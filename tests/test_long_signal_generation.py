"""Signals longer than the 4 s window must be cut off at the window start,
not wrapped into it (the pre-2026-10-31 generator folded every BNS/NSBH and
~34% of BBH inspirals into the window: BNS correlation with the true last 4 s
was 0.54 and the stored signal 83% too loud)."""
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
logging.getLogger("bilby").setLevel(logging.ERROR)
from ahsd.data.bilby_pipeline import (BilbyPreprocessor, BilbySignalInjector,  # noqa: E402
                                      get_default_psd)

SR, T = 4096, 4.0
N = int(SR * T)
BASE = dict(theta_jn=0.7, ra=1.2, dec=0.3, psi=0.6, phase=1.1, geocent_time=0.4,
            geocent_time_gps=1369224018.0 + 0.4, a1=0.1, a2=0.05, luminosity_distance=300.0)


@pytest.fixture(scope="module")
def tools():
    return BilbySignalInjector(SR, T), BilbyPreprocessor(SR, T), get_default_psd("H1")


def _window(inj, prep, psd, p, pre_roll):
    wg = inj.wfg
    keep, wg.PRE_ROLL = wg.PRE_ROLL, pre_roll
    try:
        long_raw, i0 = wg.generate_long(p, "H1")
    finally:
        wg.PRE_ROLL = keep
    return prep.preprocess(long_raw, psd, "H1")[i0:i0 + N]


def _corr(a, b):
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


@pytest.mark.parametrize("params", [
    dict(mass_1=12.8, mass_2=7.5, event_type="BBH"),
    dict(mass_1=6.3, mass_2=1.5, event_type="NSBH", lambda_2=400.0),
])
def test_long_signal_matches_full_inspiral(tools, params):
    inj, prep, psd = tools
    p = {**BASE, **params}
    assert not inj.wfg.fits_window(p), "case must exercise the long-signal path"
    new = _window(inj, prep, psd, p, inj.wfg.PRE_ROLL)
    ref = _window(inj, prep, psd, p, 1000.0)          # whole inspiral from f_lower
    assert _corr(new, ref) > 0.999
    assert abs(np.linalg.norm(new) / np.linalg.norm(ref) - 1) < 0.01


def test_short_signal_uses_original_path(tools):
    inj, prep, psd = tools
    p = {**BASE, "mass_1": 42.0, "mass_2": 36.0, "event_type": "BBH"}
    assert inj.wfg.fits_window(p)


def test_wrapped_generation_was_wrong(tools):
    """Documents the bug: the 4 s frequency-domain generation of an NSBH
    differs materially from the true last 4 s before merger."""
    inj, prep, psd = tools
    p = {**BASE, "mass_1": 6.3, "mass_2": 1.5, "event_type": "NSBH", "lambda_2": 400.0}
    old = prep.preprocess(inj.inject_multiple(np.zeros(N), [p], "H1", psd)[0], psd, "H1")
    ref = _window(inj, prep, psd, p, 1000.0)
    assert _corr(old, ref) < 0.9
