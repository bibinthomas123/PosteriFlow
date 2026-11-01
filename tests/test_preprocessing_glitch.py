"""A detector with a glitch far louder than any signal (GW170817's L1) is
dropped to white noise, as a missing detector is in training; with no other
detector to fall back on it is kept and flagged."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("gwpy")
from ahsd.inference.preprocessing import DETECTORS, prepare_real  # noqa: E402

SR, SEC, GPS = 4096, 80, 1187008882.4


def _raw(dets, glitch=None):
    rng = np.random.default_rng(0)
    raw = {d: rng.standard_normal(SEC * SR) * 1e-21 for d in dets}
    if glitch:
        i = int((SEC / 2 - 1.0) * SR)             # 1 s before the trigger
        raw[glitch][i:i + 40] += 1e-17 * np.hanning(40)
    return prepare_real(raw, GPS, {d: GPS - SEC / 2 for d in dets}, {d: SR for d in dets})


def test_glitchy_detector_dropped():
    p = _raw(["H1", "L1"], glitch="L1")
    assert p.detectors_present == ["H1"] and "L1" not in p.asds and "L1" not in p.psds
    assert np.abs(p.strain[DETECTORS.index("L1")]).max() < 7
    assert any("glitch" in w for w in p.warnings)


def test_clean_data_untouched():
    p = _raw(["H1", "L1"])
    assert p.detectors_present == ["H1", "L1"]
    assert not any("glitch" in w for w in p.warnings)


def test_lone_detector_kept():
    p = _raw(["L1"], glitch="L1")
    assert p.detectors_present == ["L1"]
    assert any("kept" in w for w in p.warnings)


def test_raw_segment_for_importance_sampling():
    from ahsd.inference.importance import segment_duration
    from ahsd.inference.preprocessing import raw_segment
    x = {d: np.arange(SEC * SR, dtype=float) for d in ("H1", "L1")}
    seg = raw_segment(x, {"H1": GPS - SEC / 2, "L1": GPS - SEC / 2}, GPS, seconds=32)
    assert segment_duration(seg) == 32.0
    assert seg["H1"][-1] == (SEC / 2 + 2) * SR - 1          # ends at the window end
    with pytest.raises(ValueError):
        raw_segment(x, {"H1": GPS - SEC / 2, "L1": GPS - SEC / 2}, GPS, seconds=60)
