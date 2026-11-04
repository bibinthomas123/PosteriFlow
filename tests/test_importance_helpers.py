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


def test_consistent_seeds_are_not_pooled():
    from ahsd.inference.importance import corrected_with_check
    out = corrected_with_check(_fake_result([0.0]), None)
    d = out.diagnostics["importance_sampling"]
    assert d["reps_used"] == 2 and d["consistent"] and len(out.samples) == 2000


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
