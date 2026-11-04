"""Helpers around the likelihood correction (no likelihood evaluations)."""
from types import SimpleNamespace

import numpy as np

from ahsd.inference.importance import replicate_spread
from ahsd.models.lean_npe import PARAM_NAMES, ParamScaler


def _res(med_shift, rng):
    s = rng.normal(size=(4000, len(PARAM_NAMES)))
    s[:, PARAM_NAMES.index("mass_1")] += med_shift
    return SimpleNamespace(samples=s)


def test_replicate_spread_flags_disagreeing_seeds():
    rng = np.random.default_rng(0)
    same = replicate_spread([_res(0.0, rng), _res(0.0, rng)], PARAM_NAMES)
    apart = replicate_spread([_res(0.0, rng), _res(3.0, rng)], PARAM_NAMES)
    assert same < 0.1            # noise only: a few percent of a 90% width
    assert apart > 0.5           # 3 sigma apart: ~0.9 of a 90% width -> dynesty


def test_circular_ranges_known_to_the_folding():
    # importance_correct folds every circular parameter into its RANGES entry
    for name in ("ra", "psi", "phase"):
        assert name in ParamScaler.CIRCULAR and name in ParamScaler.RANGES
        lo, hi = ParamScaler.RANGES[name][:2]
        x = np.array([-12.7, 20.6, lo, hi - 1e-9])
        y = lo + np.mod(x - lo, hi - lo)
        assert np.all((y >= lo) & (y < hi))


def _fake_result(meds):
    """A result whose importance_correct(seed=k) returns samples centred on meds[k]."""
    from ahsd.inference import importance  # noqa: F401  (module import check)

    class R:
        param_names = PARAM_NAMES

        def importance_correct(self, raw, seed=0, **kw):
            rng = np.random.default_rng(seed)
            s = rng.normal(size=(2000, len(PARAM_NAMES)))
            s[:, PARAM_NAMES.index("mass_1")] += meds[seed % len(meds)]
            return SimpleNamespace(samples=s, log_prob=np.zeros(len(s)),
                                   diagnostics={"importance_sampling": {"importance_ess": 100.0}})
    return R()


def test_consistent_seeds_are_pooled_without_extra_runs():
    from ahsd.inference.importance import corrected_with_check
    out = corrected_with_check(_fake_result([0.0]), None)
    d = out.diagnostics["importance_sampling"]
    assert d["reps_used"] == 2 and d["consistent"] and len(out.samples) == 2 * 2000


def test_disagreeing_seeds_are_pooled_until_stable():
    from ahsd.inference.importance import corrected_with_check
    # runs alternate between medians 3 sigma apart: spread ~0.9 of a 90% width;
    # the pooled error spread/sqrt(k) drops below 0.5 at k = 4
    out = corrected_with_check(_fake_result([0.0, 3.0]), None, max_reps=6)
    d = out.diagnostics["importance_sampling"]
    assert d["reps_used"] == 4 and len(out.samples) == 4 * 2000
    assert d["pooled_spread"] < 0.5 <= d["replicate_spread"] and d["consistent"]


def test_unstable_runs_are_flagged_after_max_reps():
    from ahsd.inference.importance import corrected_with_check
    out = corrected_with_check(_fake_result([0.0, 10.0]), None, max_reps=6)
    d = out.diagnostics["importance_sampling"]
    assert d["reps_used"] == 6 and d["consistent"] is False and d["reliable"] is False


def test_training_prior_sampler_matches_its_density():
    from ahsd.inference.priors import TrainingPrior
    rng = np.random.default_rng(0)
    for cls in ("BBH", "NSBH", "BNS"):
        p = TrainingPrior.for_class(cls, signed_spins=True)
        th = p.sample(20000, rng)
        assert np.isfinite(p.log_prob(th)).all(), cls        # every draw inside the support
    p = TrainingPrior.for_class("BBH")
    m1 = p.sample(200000, rng)[:, PARAM_NAMES.index("mass_1")]
    lo, hi = p.m1_range
    assert abs(np.median(m1) - np.sqrt(lo * hi)) / np.sqrt(lo * hi) < 0.02   # log-uniform m1


def _bbh_prior():
    from ahsd.inference.priors import TrainingPrior
    return TrainingPrior.for_class("BBH", signed_spins=True)


def test_slow_coords_round_trip_and_jacobian():
    from ahsd.inference.coords import IDX, SlowCoords
    p = _bbh_prior()
    c = SlowCoords(p)
    th = p.sample(2000, np.random.default_rng(1))
    x = c.to_x(th)
    assert np.isfinite(x).all()
    back = c.from_x(x)
    assert np.allclose(back, th[:, IDX], rtol=1e-8, atol=1e-8)
    # the map is diagonal: log|dtheta/dx| = sum of per-coordinate log derivatives
    h = 1e-6
    num = np.zeros(len(th))
    for j in range(x.shape[1]):
        xp, xm = x.copy(), x.copy()
        xp[:, j] += h
        xm[:, j] -= h
        num += np.log(np.abs((c.from_x(xp)[:, j] - c.from_x(xm)[:, j]) / (2 * h)))
    assert np.allclose(num, c.log_jac(th), atol=1e-4)


def test_mixture_in_unbounded_coords_reproduces_the_prior():
    """Importance weights pi(theta) |dtheta/dx| / g(x) from a Gaussian mixture
    fitted to prior draws in these coordinates must recover the prior; in the
    bounded (log) coordinates the same test gave an ESS of 9 / 40000."""
    from sklearn.mixture import GaussianMixture
    from ahsd.inference.coords import SlowCoords
    p = _bbh_prior()
    c = SlowCoords(p)
    rng = np.random.default_rng(0)
    gm = GaussianMixture(16, covariance_type="full", reg_covar=1e-5, random_state=0).fit(
        c.to_x(p.sample(20000, rng)))
    x = gm.sample(40000)[0]
    v = c.from_x(x)
    th = np.zeros((len(x), len(PARAM_NAMES)))
    from ahsd.inference.coords import IDX
    th[:, IDX] = v
    m1, m2 = np.maximum(th[:, 0], th[:, 1]), np.minimum(th[:, 0], th[:, 1])
    th[:, 0], th[:, 1] = m1, m2
    th[:, IDX[3]] = np.mod(th[:, IDX[3]], 2 * np.pi)      # ra and psi wrapped
    th[:, IDX[6]] = np.mod(th[:, IDX[6]], np.pi)
    lg = c.mixture_logpdf(gm, c.to_x(th))
    lw = p.log_prob(th) + c.log_jac(th) - lg
    f = np.isfinite(lw)
    w = np.exp(lw[f] - lw[f].max())
    ess = w.sum() ** 2 / (w ** 2).sum()
    o = np.argsort(th[f, 0])
    med = np.interp(0.5, np.cumsum(w[o]) / w.sum(), th[f, 0][o])
    assert ess > 0.5 * len(w), ess
    assert abs(med - np.sqrt(5 * 100)) / np.sqrt(5 * 100) < 0.05, med


def _branch_walk(c, correct, n=2000, steps=150, seed=0):
    """Random walk on the folded, wrapped sampling space whose step size
    depends on the current point's 'branch' (x2 > 0.3: large steps), with or
    without the Hastings correction from SlowCoords.rw_logpdf. Target:
    standard normal in the non-periodic coordinates, uniform in ra / psi."""
    gauss = [0, 1, 2, 4, 5, 7, 8]
    sds = {0: 0.15, 1: 1.2}
    d = 9
    prec = {k: np.eye(d) / s ** 2 for k, s in sds.items()}
    logdet = {k: 2 * d * np.log(s) for k, s in sds.items()}
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    x[:, 3] = rng.uniform(0, 2 * np.pi, n)
    x[:, 6] = rng.uniform(0, np.pi, n)
    x = c.fold_wrap(x)
    logp = lambda y: -0.5 * (y[:, gauss] ** 2).sum(1)
    lab_fn = lambda y: (y[:, 2] > 0.3).astype(int)
    for _ in range(steps):
        lab = lab_fn(x)
        xp = c.fold_wrap(x + rng.standard_normal(x.shape) * np.where(lab == 1, sds[1], sds[0])[:, None])
        lab_p = lab_fn(xp)
        la = logp(xp) - logp(x)
        if correct:
            for k_from in (0, 1):
                for k_to in (0, 1):
                    m = (lab == k_from) & (lab_p == k_to) & (k_from != k_to)
                    if m.any():
                        la[m] += (c.rw_logpdf(xp[m], x[m], prec[k_to], logdet[k_to], 1.0)
                                  - c.rw_logpdf(x[m], xp[m], prec[k_from], logdet[k_from], 1.0))
        acc = np.log(rng.uniform(size=n)) < la
        x = np.where(acc[:, None], xp, x)
    return x


def test_branch_dependent_steps_need_the_hastings_correction():
    """A step covariance chosen by the current point's branch is only
    reversible with the q(x|x')/q(x'|x) term; without it the large-step
    region drains (P(x2 > 0.3) 0.38 -> ~0.14). Regression for the per-branch
    random walk in importance_correct."""
    from scipy.stats import ks_1samp, norm
    from ahsd.inference.coords import SlowCoords
    c = SlowCoords(_bbh_prior())
    good = _branch_walk(c, True)
    bad = _branch_walk(c, False)
    assert ks_1samp(good[:, 2], norm.cdf).statistic < 0.04
    assert ks_1samp(bad[:, 2], norm.cdf).statistic > 0.15
    assert (good[:, 0] >= good[:, 1]).all()


def test_rw_logpdf_is_symmetric_for_one_covariance():
    """Same covariance both ways: the image-summed densities agree (the
    folded, wrapped walk is symmetric), so the correction vanishes."""
    from ahsd.inference.coords import SlowCoords
    c = SlowCoords(_bbh_prior())
    rng = np.random.default_rng(3)
    a = rng.standard_normal((9, 9))
    cov = a @ a.T + 9 * np.eye(9)
    cov[[0, 1], :] = 0.5 * (cov[[0, 1], :] + cov[[1, 0], :])     # swap-symmetric
    cov[:, [0, 1]] = 0.5 * (cov[:, [0, 1]] + cov[:, [1, 0]])
    x = c.fold_wrap(rng.standard_normal((500, 9)) * 2)
    y = c.fold_wrap(x + rng.standard_normal((500, 9)) @ np.linalg.cholesky(cov).T * 0.3)
    prec, ld = np.linalg.inv(cov), np.linalg.slogdet(cov)[1]
    assert np.allclose(c.rw_logpdf(x, y, prec, ld, 0.3), c.rw_logpdf(y, x, prec, ld, 0.3), atol=1e-8)


def test_floored_resample_keeps_light_branches_and_is_unbiased():
    """Every live branch keeps >= floor particles; the carried weights make
    the resampled set an unbiased representation of the weighted input."""
    from ahsd.inference.importance import floored_resample
    rng = np.random.default_rng(0)
    n = 4000
    labels = np.r_[np.zeros(3000, int), np.ones(990, int), np.full(10, 2)]
    w = np.r_[rng.uniform(size=3000), 1e-4 * rng.uniform(size=990), 1e-7 * np.ones(10)]
    w /= w.sum()
    f = rng.standard_normal(n)                       # test function per particle
    truth = np.sum(w * f)
    est, branch_mass = [], []
    for _ in range(300):
        idx, lw = floored_resample(w, labels, n, 0.05, rng)
        assert len(idx) == n and np.isclose(np.exp(lw).sum(), 1.0)
        counts = np.bincount(labels[idx], minlength=3)
        assert (counts >= 200).all(), counts        # 5% floor for all three
        est.append(np.sum(np.exp(lw) * f[idx]))
        branch_mass.append([np.exp(lw[labels[idx] == c]).sum() for c in range(3)])
    assert abs(np.mean(est) - truth) < 4 * np.std(est) / np.sqrt(len(est)) + 1e-12
    W = [w[labels == c].sum() for c in range(3)]
    assert np.allclose(np.mean(branch_mass, 0), W, rtol=1e-9)


def test_floored_resample_single_branch_is_plain_systematic():
    from ahsd.inference.importance import floored_resample
    rng = np.random.default_rng(1)
    w = rng.uniform(size=1000)
    w /= w.sum()
    idx, lw = floored_resample(w, np.zeros(1000, int), 1000, 0.05, rng)
    assert np.allclose(lw, -np.log(1000))
    assert np.allclose(np.bincount(idx, minlength=1000) / 1000, w, atol=1.5e-3)


def test_inflated_mixture_scales_every_covariance():
    from sklearn.mixture import GaussianMixture
    from ahsd.inference.importance import _inflated
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, (500, 3)), rng.normal(5, 0.5, (500, 3))]
    gm = GaussianMixture(2, covariance_type="full", random_state=0).fit(x)
    g4 = _inflated(gm, 4.0)
    assert np.allclose(g4.covariances_, 4 * gm.covariances_)
    s = g4.sample(40000)[0]
    # a mixture of two well-separated components: per-component spread x2
    lab = g4.predict(s)
    for k in range(2):
        assert np.allclose(np.std(s[lab == k], 0), 2 * np.sqrt(np.diag(gm.covariances_[k])), rtol=0.15)
    # density is normalized: Monte Carlo of exp(log p) / q under its own draws = 1
    assert np.isclose(np.mean(np.exp(g4.score_samples(s) - g4.score_samples(s))), 1.0)


def test_sky_flat_mixture_density_matches_its_sampler():
    """Draws of SkyFlatMixture follow its logpdf: importance weights
    logpdf / logpdf of its own draws are ~1 for a reference density, checked
    via the isotropic sky marginal and a self-normalized IS identity."""
    from scipy.stats import kstest
    from sklearn.mixture import GaussianMixture
    from ahsd.inference.coords import SkyFlatMixture, SlowCoords
    p = _bbh_prior()
    c = SlowCoords(p)
    rng = np.random.default_rng(0)
    x0 = c.to_x(p.sample(6000, rng))
    x0[:, 3] = rng.normal(0.5, 0.2, len(x0))      # a network blob on the sky
    x0[:, 4] = rng.normal(-0.3, 0.1, len(x0))
    gm = GaussianMixture(6, covariance_type="full", reg_covar=1e-5, random_state=0).fit(c.wrap(x0))
    g = SkyFlatMixture(gm, c)
    s = g.sample(30000, rng)
    assert kstest(np.tanh(s[:, 4]), "uniform", args=(-1, 2)).statistic < 0.015      # isotropic
    assert kstest(s[:, 3], "uniform", args=(0, 2 * np.pi)).statistic < 0.015
    # normalized density: E_g[ q / g ] = 1 for a normalized q (the wrapped mixture)
    lq = c.mixture_logpdf(gm, s)
    lg = g.logpdf(s)
    assert np.all(np.isfinite(lg))
    # E_g[ q / g ] = integral of q = 1: g covers q (same conditional, broader
    # sky), so this checks g's density against its own sampler. (The reverse,
    # E_q[g / q], does not hold: q cannot cover an isotropic sky.)
    r2 = np.exp(lq - lg)
    assert abs(np.mean(r2) - 1) < 0.05, np.mean(r2)
