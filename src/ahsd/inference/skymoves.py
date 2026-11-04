"""Sky-mode jumps for the likelihood correction.

A two- or three-detector network leaves the sky DEGENERATE in a known way:
the arrival-time differences, which the data pin to ~0.1 ms, are unchanged

  - by reflecting the source direction through the plane of the three
    detector sites (three detectors: two mirror sky patches), and
  - by rotating it about the baseline of two detectors (two detectors: a
    ring of constant delay).

What tells the mirror images apart is the antenna response, so each image
needs its own polarization angle and distance. A random-walk sampler cannot
cross between such islands, and an amortized proposal often puts its mass on
one of them: on an edge-on test injection the correction found one island
per seed (distance 260 vs 410 Mpc), while dynesty's single run found two and
missed a third that holds the injected truth (all three share the H1-L1 /
H1-V1 delays to < 1 ms or differ by one wave cycle).

`SkyJumps` proposes the jump exactly: the sky is moved by the reflection or
by a uniformly random rotation (both preserve solid angle), and (psi, d_L)
are drawn so that the complex detector responses
    c_i = (F+_i (1 + cos^2 i) / 2 + 1j Fx_i cos i) / d_L
match those of the current point up to a common phase (phase is marginalized
in the likelihood): a softmax over a psi grid of the matching residual, and a
log-normal distance around the best-matching amplitude. `log_q` evaluates the
proposal density, so the move is an exact Metropolis-Hastings step.
"""
from __future__ import annotations

import numpy as np

from ahsd.models.lean_npe import PARAM_NAMES

_J = {n: i for i, n in enumerate(PARAM_NAMES)}
_trapz = getattr(np, "trapezoid", None) or np.trapz
C_LIGHT = 299792458.0


class SkyJumps:
    K = 36                 # psi grid cells on [0, pi)
    TAU = 0.05             # softmax temperature of the (relative) residual
    MIX = 0.1              # uniform share of the psi proposal
    SIG_D = 0.15           # log-normal width of the distance proposal

    def __init__(self, dets, gps: float, psds: dict, f_lo: float = 20.0, f_hi: float = 512.0):
        import bilby
        from bilby.gw.utils import greenwich_mean_sidereal_time
        self.dets = [d for d in dets if d in psds]
        ifos = [bilby.gw.detector.get_empty_interferometer(d) for d in self.dets]
        self.D = np.stack([i.detector_tensor for i in ifos])            # [k, 3, 3]
        self.r = np.stack([np.asarray(i.vertex, float) for i in ifos])  # [k, 3] (m)
        self.gmst = float(np.fmod(greenwich_mean_sidereal_time(gps), 2 * np.pi))
        w = []
        for d in self.dets:
            f = np.asarray(psds[d]["frequencies"], float)
            s = np.asarray(psds[d]["psd"], float)
            m = (f >= f_lo) & (f <= f_hi) & np.isfinite(s) & (s > 0)
            w.append(_trapz(f[m] ** (-7.0 / 3.0) / s[m], f[m]) if m.sum() > 2 else 0.0)
        w = np.asarray(w)
        self.w = w / max(w.max(), 1e-300)                                  # sensitivity weights
        order = np.argsort(-self.w)
        self.ok = len(self.dets) >= 2 and np.sum(self.w > 1e-3) >= 2
        if not self.ok:
            return
        a, b = order[0], order[1]
        base = self.r[a] - self.r[b]
        self.axis = base / np.linalg.norm(base)                          # two most sensitive sites
        self.three = int(np.sum(self.w > 1e-3)) >= 3
        # every baseline of the sensitive sites: a rotation about baseline
        # (i, j) keeps the i-j delay and moves the others -- e.g. between two
        # islands that share the H1-V1 delay and differ by one wave cycle in
        # H1-L1 (no reflection connects those)
        sens = [i for i in order if self.w[i] > 1e-3]
        axes, probes = [], []
        for u in range(len(sens)):
            for v in range(u + 1, len(sens)):
                bl = self.r[sens[u]] - self.r[sens[v]]
                axes.append(bl / np.linalg.norm(bl))
                rest = [i for i in sens if i not in (sens[u], sens[v])]
                # delay-shift move: the a-l delay (l a third site) changes
                # along the orbit about baseline (a, b)
                probes.append((self.r[sens[u]] - self.r[rest[0]]) / C_LIGHT if rest else np.zeros(3))
        self.axes, self.probes = np.array(axes), np.array(probes)
        if self.three:
            c = order[2]
            nrm = np.cross(self.r[b] - self.r[a], self.r[c] - self.r[a])
            self.normal = nrm / np.linalg.norm(nrm)

    # ── geometry ────────────────────────────────────────────────────────
    def omega(self, ra, dec):
        """Earth-fixed unit vector towards the source (bilby's convention)."""
        phi = ra - self.gmst
        return np.stack([np.cos(dec) * np.cos(phi), np.cos(dec) * np.sin(phi), np.sin(dec)], -1)

    def radec(self, om):
        om = om / np.linalg.norm(om, axis=-1, keepdims=True)
        dec = np.arcsin(np.clip(om[..., 2], -1.0, 1.0))
        ra = np.mod(np.arctan2(om[..., 1], om[..., 0]) + self.gmst, 2 * np.pi)
        return ra, dec

    def delays(self, ra, dec):
        """Arrival time at each site relative to the geocentre [n, k] (s)."""
        return -(self.omega(ra, dec) @ self.r.T) / C_LIGHT

    def reflect(self, ra, dec):
        om = self.omega(ra, dec)
        return self.radec(om - 2.0 * (om @ self.normal)[..., None] * self.normal)

    def rotate(self, ra, dec, alpha, axis=None):
        """Rodrigues rotation by alpha about `axis` ([3] or per point [n, 3];
        default the baseline of the two most sensitive sites)."""
        om = self.omega(ra, dec)
        k = self.axis if axis is None else axis
        if np.ndim(k) == 1:
            k = np.broadcast_to(k, om.shape)
        ca, sa = np.cos(alpha)[..., None], np.sin(alpha)[..., None]
        kd = np.sum(om * k, -1, keepdims=True)
        rot = om * ca + np.cross(k, om) * sa + k * kd * (1 - ca)
        return self.radec(rot)

    def antenna(self, ra, dec, psi):
        """F+ and Fx [n, k] (bilby's polarization-tensor convention)."""
        phi = ra - self.gmst
        theta = np.pi / 2 - dec
        u = np.stack([np.cos(phi) * np.cos(theta), np.cos(theta) * np.sin(phi), -np.sin(theta)], -1)
        v = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], -1)
        sp, cp = np.sin(psi)[..., None], np.cos(psi)[..., None]
        m = -u * sp - v * cp
        n = -u * cp + v * sp
        ep = np.einsum("...a,...b->...ab", m, m) - np.einsum("...a,...b->...ab", n, n)
        ex = np.einsum("...a,...b->...ab", m, n) + np.einsum("...a,...b->...ab", n, m)
        return np.einsum("kab,...ab->...k", self.D, ep), np.einsum("kab,...ab->...k", self.D, ex)

    def responses(self, ra, dec, psi, theta_jn, d=1.0):
        fp, fx = self.antenna(ra, dec, psi)
        ci = np.cos(theta_jn)[..., None]
        return (fp * (1 + ci ** 2) / 2 + 1j * fx * ci) / np.asarray(d)[..., None]

    # ── proposal ────────────────────────────────────────────────────────
    def _psi_table(self, ra_to, dec_to, theta_jn, c):
        """Per psi cell: proposal probability and best-matching distance for
        moving the responses c [n, k] to the sky (ra_to, dec_to)."""
        n = len(ra_to)
        grid = (np.arange(self.K) + 0.5) * np.pi / self.K
        psi = np.broadcast_to(grid, (n, self.K))
        r = self.responses(np.repeat(ra_to[:, None], self.K, 1), np.repeat(dec_to[:, None], self.K, 1),
                           psi, np.repeat(theta_jn[:, None], self.K, 1))      # [n, K, k]
        w = self.w
        cc = np.sum(w * np.abs(c) ** 2, -1)                                    # [n]
        rr = np.sum(w * np.abs(r) ** 2, -1)                                    # [n, K]
        rc = np.sum(w * np.conj(r) * c[:, None, :], -1)                        # [n, K]
        resid = 1.0 - np.abs(rc) ** 2 / np.maximum(rr * cc[:, None], 1e-300)  # in [0, 1]
        logits = -resid / self.TAU
        p = np.exp(logits - logits.max(1, keepdims=True))
        p = (1 - self.MIX) * p / p.sum(1, keepdims=True) + self.MIX / self.K
        # best complex scale z* = rc / rr of r / d onto c: d = 1 / |z*|
        d_opt = rr / np.maximum(np.abs(rc), 1e-300)
        return p, d_opt

    DMAX = 0.012           # delay-shift move: shift drawn uniformly in +-DMAX (s)

    def _orbit(self, om, k, u):
        """Delay of the probe pair along the orbit R(alpha) om about k:
        f(alpha) = C0 + A cos(alpha) + B sin(alpha). Returns (A, B)."""
        kw = np.sum(k * om, -1)
        A = -(om @ u if u.ndim == 1 else np.sum(om * u, -1)) + kw * np.sum(k * u, -1)
        B = -np.sum(np.cross(k, om) * u, -1)
        return A, B

    def shift_target(self, ra, dec, rng):
        """Delay-shift rotation: about a baseline (a, b) drawn uniformly, by
        the angle that changes the a-l delay (l the third site) by delta ~
        U(-DMAX, DMAX); one of the two solutions with prob 1/2. Connects sky
        islands one wave cycle apart in one delay (cycle-slip images), which
        a uniform angle rarely hits. Returns (ra', dec', log q ratio of the
        angle (reverse / forward) = log|f'(0)| - log|f'(alpha)|, valid)."""
        n = len(ra)
        j = rng.integers(0, len(self.axes), n)
        k, u = self.axes[j], self.probes[j]
        om = self.omega(ra, dec)
        A, B = self._orbit(om, k, u)
        delta = rng.uniform(-self.DMAX, self.DMAX, n)
        R = np.hypot(A, B)
        c = (A + delta) / np.maximum(R, 1e-300)
        ok = (np.abs(c) <= 1) & (R > 1e-9)
        phi = np.arctan2(B, A)
        sgn = np.where(rng.uniform(size=n) < 0.5, 1.0, -1.0)
        alpha = phi + sgn * np.arccos(np.clip(c, -1, 1))
        ra2, dec2 = self.rotate(ra, dec, alpha, k)
        with np.errstate(divide="ignore"):
            lr = np.log(np.abs(B)) - np.log(np.abs(-A * np.sin(alpha) + B * np.cos(alpha)))
        return ra2, dec2, np.where(ok & np.isfinite(lr), lr, -np.inf), ok

    def sky_target(self, ra, dec, rng, kind):
        """'reflect' (three sites): mirror through the site plane; 'rotate':
        uniform angle about a baseline drawn uniformly per point (both choices
        independent of the point, so the sky part of the proposal is
        symmetric and preserves solid angle)."""
        if kind == "reflect":
            return self.reflect(ra, dec)
        ax = self.axes[rng.integers(0, len(self.axes), len(ra))]
        rr, rd = self.rotate(ra, dec, rng.uniform(0, 2 * np.pi, len(ra)), ax)
        if kind == "mix":           # per point, reflect or rotate with prob 1/2
            fr, fd = self.reflect(ra, dec)
            m = rng.uniform(size=len(ra)) < 0.5
            rr, rd = np.where(m, fr, rr), np.where(m, fd, rd)
        return rr, rd

    def propose(self, th, rng, kind=None):
        """Jump proposal for physical rows th [n, 11]. Returns (th', log q
        of the forward draw of (psi', d') in physical units, log |J| of the
        sky map in the (ra, atanh sin dec) sampling coordinates)."""
        kind = kind or ("mix3" if self.three else "rotate")
        ra, dec = th[:, _J["ra"]], th[:, _J["dec"]]
        tj, psi, d = th[:, _J["theta_jn"]], th[:, _J["psi"]], th[:, _J["luminosity_distance"]]
        lr_sky = np.zeros(len(th))
        if kind == "shift":
            ra2, dec2, lr_sky, _ = self.shift_target(ra, dec, rng)
        elif kind == "mix3":        # per point: reflect / rotate / delay-shift, 1/3 each
            ra2, dec2 = self.sky_target(ra, dec, rng, "mix")
            sr, sd, slr, _ = self.shift_target(ra, dec, rng)
            m = rng.uniform(size=len(th)) < 1 / 3
            ra2, dec2, lr_sky = np.where(m, sr, ra2), np.where(m, sd, dec2), np.where(m, slr, 0.0)
        else:
            ra2, dec2 = self.sky_target(ra, dec, rng, kind)
        c = self.responses(ra, dec, psi, tj, d)
        p, d_opt = self._psi_table(ra2, dec2, tj, c)
        cum = np.cumsum(p, 1)
        k = np.minimum((rng.uniform(size=(len(th), 1)) > cum).sum(1), self.K - 1)
        psi2 = (k + rng.uniform(size=len(th))) * np.pi / self.K
        d2 = d_opt[np.arange(len(th)), k] * np.exp(self.SIG_D * rng.standard_normal(len(th)))
        out = th.copy()
        out[:, _J["ra"]], out[:, _J["dec"]] = ra2, dec2
        out[:, _J["psi"]], out[:, _J["luminosity_distance"]] = psi2, d2
        lq = self.log_q(th, out)
        with np.errstate(divide="ignore"):
            log_j = 2 * np.log(np.abs(np.cos(dec))) - 2 * np.log(np.abs(np.cos(dec2)))
        # the delay-shift angle's own reverse/forward density ratio rides on
        # the Jacobian term (-inf where the shift had no solution: rejected)
        return out, lq, log_j + lr_sky

    def log_q(self, th_from, th_to):
        """log density (physical psi, d_L) of drawing th_to's (psi, d) from
        th_from, given th_to's sky."""
        ra, dec = th_from[:, _J["ra"]], th_from[:, _J["dec"]]
        tj = th_from[:, _J["theta_jn"]]
        c = self.responses(ra, dec, th_from[:, _J["psi"]], tj, th_from[:, _J["luminosity_distance"]])
        p, d_opt = self._psi_table(th_to[:, _J["ra"]], th_to[:, _J["dec"]], tj, c)
        psi2 = np.mod(th_to[:, _J["psi"]], np.pi)
        k = np.minimum((psi2 / (np.pi / self.K)).astype(int), self.K - 1)
        i = np.arange(len(th_from))
        d2 = th_to[:, _J["luminosity_distance"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            lpsi = np.log(p[i, k]) - np.log(np.pi / self.K)
            z = (np.log(d2) - np.log(d_opt[i, k])) / self.SIG_D
            ld = -0.5 * z ** 2 - np.log(self.SIG_D * np.sqrt(2 * np.pi)) - np.log(d2)
        return np.where(np.isfinite(lpsi + ld), lpsi + ld, -np.inf)
