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
