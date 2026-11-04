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
