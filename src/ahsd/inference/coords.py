"""Unbounded sampling coordinates for the likelihood correction.

The correction samples the 9 "slow" parameters (masses, distance, sky,
inclination, polarization, aligned spins) with Gaussian-mixture proposals and
random-walk steps. In the physical (or log) coordinates the prior has HARD
EDGES -- mass and distance boxes, spins in [a_lo, a_max], |dec| < pi/2,
0 < theta_jn < pi -- and a Gaussian mixture cannot represent a density that
stops at a boundary: it spills over and is too thin just inside, so importance
weights near the edges explode (a flat-likelihood test collapsed to an ESS of
9 of 4000). Here every bounded parameter is mapped to the whole real line:

    masses        logit of log m in the common box [log m_lo, log m_hi]
                  (one box for both masses keeps the m1 <-> m2 symmetry)
    distance      logit of d in [d_lo, d_hi]
    dec           atanh(sin dec)
    theta_jn      atanh(cos theta_jn)
    spins         logit of a in [a_lo, a_max]
    ra, psi       unchanged (periodic)

so the prior has smooth tails, mixtures fit it, and steps never leave the
support. log_jac(theta) = log |d theta / d x| converts a physical density to
one in these coordinates (exact).
"""
from __future__ import annotations

import numpy as np

from ahsd.models.lean_npe import PARAM_NAMES

_J = {n: i for i, n in enumerate(PARAM_NAMES)}
NAMES = ("mass_1", "mass_2", "luminosity_distance", "ra", "dec", "theta_jn", "psi", "a1", "a2")
IDX = [_J[n] for n in NAMES]


def _logit(z):
    return np.log(z) - np.log1p(-z)


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class SlowCoords:
    def __init__(self, prior, eps: float = 1e-12):
        m_lo = min(prior.m1_range[0], prior.m2_range[0])
        m_hi = max(prior.m1_range[1], prior.m2_range[1])
        self.lm = (float(np.log(m_lo)), float(np.log(m_hi)))
        self.d = tuple(float(v) for v in prior.dist_range)
        self.a1 = (float(prior.a1_lo), float(prior.a1_max))
        self.a2 = (float(prior.a2_lo), float(prior.a2_max))
        self.eps = eps

    def _z(self, v, box):
        return (v - box[0]) / (box[1] - box[0])

    def to_x(self, th: np.ndarray) -> np.ndarray:
        """Physical [n, 11] -> unbounded slow coordinates [n, 9]."""
        e = self.eps
        c = lambda z: np.clip(z, e, 1 - e)
        x = np.empty((len(th), len(NAMES)))
        x[:, 0] = _logit(c(self._z(np.log(th[:, _J["mass_1"]]), self.lm)))
        x[:, 1] = _logit(c(self._z(np.log(th[:, _J["mass_2"]]), self.lm)))
        x[:, 2] = _logit(c(self._z(th[:, _J["luminosity_distance"]], self.d)))
        x[:, 3] = th[:, _J["ra"]]
        x[:, 4] = np.arctanh(np.clip(np.sin(th[:, _J["dec"]]), -1 + e, 1 - e))
        x[:, 5] = np.arctanh(np.clip(np.cos(th[:, _J["theta_jn"]]), -1 + e, 1 - e))
        x[:, 6] = th[:, _J["psi"]]
        x[:, 7] = _logit(c(self._z(th[:, _J["a1"]], self.a1)))
        x[:, 8] = _logit(c(self._z(th[:, _J["a2"]], self.a2)))
        return x

    def from_x(self, x: np.ndarray) -> np.ndarray:
        """Unbounded slow coordinates [n, 9] -> physical values [n, 9]
        (in NAMES order; masses NOT sorted -- the caller sorts)."""
        v = np.empty_like(x)
        span = lambda box, s: box[0] + (box[1] - box[0]) * s
        v[:, 0] = np.exp(span(self.lm, _sigmoid(x[:, 0])))
        v[:, 1] = np.exp(span(self.lm, _sigmoid(x[:, 1])))
        v[:, 2] = span(self.d, _sigmoid(x[:, 2]))
        v[:, 3] = x[:, 3]
        v[:, 4] = np.arcsin(np.tanh(x[:, 4]))
        v[:, 5] = np.arccos(np.tanh(x[:, 5]))
        v[:, 6] = x[:, 6]
        v[:, 7] = span(self.a1, _sigmoid(x[:, 7]))
        v[:, 8] = span(self.a2, _sigmoid(x[:, 8]))
        return v

    # periodic coordinates (index -> period): kept in [0, period) after every
    # move, and mixture densities are evaluated WRAPPED (summed over the
    # neighbouring periodic images). A mixture fitted to angles in [0, 2 pi)
    # otherwise has edges there too, and samples that drifted a period away
    # (the prior does not constrain ra or psi) got huge weights against it.
    PERIODIC = {3: 2 * np.pi, 6: np.pi}

    def wrap(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        for j, per in self.PERIODIC.items():
            x[:, j] = np.mod(x[:, j], per)
        return x

    def mixture_logpdf(self, gm, x: np.ndarray, swap_masses: bool = True) -> np.ndarray:
        """log density of WRAPPED, mass-SORTED points under the mixture gm
        (fitted in these coordinates): sum over periodic images (-1, 0, +1 period
        in each periodic coordinate) and, if swap_masses, over the m1 <-> m2
        label swap."""
        import itertools
        x = self.wrap(x)
        terms = []
        variants = [x]
        if swap_masses:
            xs = x.copy()
            xs[:, [0, 1]] = x[:, [1, 0]]
            variants.append(xs)
        for v in variants:
            for shifts in itertools.product((-1, 0, 1), repeat=len(self.PERIODIC)):
                y = v.copy()
                for (j, per), k in zip(self.PERIODIC.items(), shifts):
                    y[:, j] = y[:, j] + k * per
                terms.append(gm.score_samples(y))
        return np.logaddexp.reduce(np.stack(terms), axis=0)

    def fold_wrap(self, x: np.ndarray) -> np.ndarray:
        """Map a proposed point to the sampling space: periodic angles wrapped
        and the two mass coordinates sorted (x0 >= x1; with one common box,
        logit(log m) is monotone, so this is m1 >= m2)."""
        x = self.wrap(x)
        a, b = x[:, 0].copy(), x[:, 1].copy()
        x[:, 0], x[:, 1] = np.maximum(a, b), np.minimum(a, b)
        return x

    def rw_logpdf(self, x_from: np.ndarray, x_to: np.ndarray, prec: np.ndarray,
                  logdet: float, scale: float) -> np.ndarray:
        """log density of proposing the folded, wrapped point x_to from x_from
        with a Gaussian random walk of covariance scale^2 * inv(prec): the step
        lands on any image of x_to (mass swap x periodic shifts -1/0/+1), so
        the density sums over them. Needed for the Hastings ratio when the
        step covariance depends on the current point (per-branch steps)."""
        import itertools
        d = x_to.shape[1]
        xs = x_to.copy()
        xs[:, [0, 1]] = x_to[:, [1, 0]]
        terms = []
        for v in (x_to, xs):
            for shifts in itertools.product((-1, 0, 1), repeat=len(self.PERIODIC)):
                y = v.copy()
                for (j, per), k in zip(self.PERIODIC.items(), shifts):
                    y[:, j] = y[:, j] + k * per
                r = (y - x_from) / scale
                terms.append(-0.5 * np.einsum("ni,ij,nj->n", r, prec, r))
        return (np.logaddexp.reduce(np.stack(terms), axis=0)
                - 0.5 * logdet - d * np.log(scale) - 0.5 * d * np.log(2 * np.pi))

    def log_jac(self, th: np.ndarray) -> np.ndarray:
        """log |d theta / d x| summed over the transformed parameters; -inf
        on a boundary (where the physical density has no support anyway)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.zeros(len(th))
            for n in ("mass_1", "mass_2"):
                m = th[:, _J[n]]
                z = self._z(np.log(m), self.lm)
                out += np.log(m) + np.log(self.lm[1] - self.lm[0]) + np.log(z) + np.log1p(-z)
            z = self._z(th[:, _J["luminosity_distance"]], self.d)
            out += np.log(self.d[1] - self.d[0]) + np.log(z) + np.log1p(-z)
            out += np.log(np.cos(th[:, _J["dec"]]))
            out += np.log(np.sin(th[:, _J["theta_jn"]]))
            for n, box in (("a1", self.a1), ("a2", self.a2)):
                z = self._z(th[:, _J[n]], box)
                out += np.log(box[1] - box[0]) + np.log(z) + np.log1p(-z)
        return np.where(np.isfinite(out), out, -np.inf)


class SkyFlatMixture:
    """Isotropic sky x a Gaussian mixture's conditional for the other
    coordinates given the sky (sampling coordinates of SlowCoords):

        g(x) = pi_sky(ra, x_dec) * gm(x_rest | ra, x_dec).

    Used as the correction's tempering anchor: the network's sky can sit on
    the wrong island (a cycle-slip image), and a mixture fitted to its samples
    then favours that island at every intermediate temperature. The
    conditional keeps the network's information on masses, spins and the
    extrinsic parameters given each sky position. Exact density and sampler
    (Gaussian conditionals; ra images summed as in mixture_logpdf)."""
    S = [3, 4]                      # ra, atanh(sin dec)
    R = [0, 1, 2, 5, 6, 7, 8]

    def __init__(self, gm, coords: SlowCoords):
        self.gm, self.c = gm, coords
        S, R = self.S, self.R
        cov = gm.covariances_
        self.mu_s, self.mu_r = gm.means_[:, S], gm.means_[:, R]
        css = cov[:, S][:, :, S]
        self.css_inv = np.linalg.inv(css)
        self.css_logdet = np.linalg.slogdet(css)[1]
        crs = cov[:, R][:, :, S]
        self.gain = np.einsum("kij,kjl->kil", crs, self.css_inv)              # [K, r, s]
        cond = cov[:, R][:, :, R] - np.einsum("kij,klj->kil", self.gain, crs)
        self.cond_chol = np.linalg.cholesky(cond + 1e-12 * np.eye(len(R)))
        self.logw = np.log(gm.weights_)

    def _log_resp(self, xs):
        """log w_k N(xs + image; mu_k^s, C_k^ss) for ra images -1, 0, +1:
        [n, K, 3]."""
        out = []
        for k in (-1, 0, 1):
            y = xs.copy()
            y[:, 0] = y[:, 0] + k * 2 * np.pi
            d = y[:, None, :] - self.mu_s[None]
            m = np.einsum("nki,kij,nkj->nk", d, self.css_inv, d)
            out.append(self.logw - 0.5 * (m + self.css_logdet + 2 * np.log(2 * np.pi)))
        return np.stack(out, -1)

    def log_sky_prior(self, x):
        t = np.tanh(x[:, 4])
        return -np.log(2 * np.pi) + np.log(0.5) + np.log1p(-t ** 2)

    def logpdf(self, x):
        x = self.c.wrap(x)
        lr = self._log_resp(x[:, self.S])
        log_fs = np.logaddexp.reduce(lr.reshape(len(x), -1), axis=1)
        return self.log_sky_prior(x) + self.c.mixture_logpdf(self.gm, x) - log_fs

    def sample(self, n, rng):
        xs = np.column_stack([rng.uniform(0, 2 * np.pi, n), np.arctanh(rng.uniform(-1, 1, n) * (1 - 1e-12))])
        lr = self._log_resp(xs).reshape(n, -1)
        p = np.exp(lr - lr.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        pick = (rng.uniform(size=(n, 1)) > np.cumsum(p, 1)).sum(1).clip(0, p.shape[1] - 1)
        comp, img = pick // 3, pick % 3 - 1
        ys = xs.copy()
        ys[:, 0] = ys[:, 0] + img * 2 * np.pi
        mean = self.mu_r[comp] + np.einsum("nij,nj->ni", self.gain[comp], ys - self.mu_s[comp])
        xr = mean + np.einsum("nij,nj->ni", self.cond_chol[comp], rng.standard_normal((n, len(self.R))))
        x = np.empty((n, 9))
        x[:, self.S], x[:, self.R] = xs, xr
        return self.c.fold_wrap(x)
