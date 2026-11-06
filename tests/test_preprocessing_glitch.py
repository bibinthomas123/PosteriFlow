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


def _noisy(n=4096 * 40, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 4096.0
    return rng.standard_normal(n) + 200.0 * np.sin(2 * np.pi * 2.0 * t)   # raw strain: low frequencies dominate


def test_gate_removes_the_glitch_and_keeps_the_rest():
    from ahsd.inference.preprocessing import GATE_HALF_S, GATE_TAPER_S, gate_glitch
    x = _noisy()
    c = 20 * 4096
    amp = 1e4
    glitched = x.copy()
    glitched[c - 50:c + 50] += amp * np.sin(2 * np.pi * 100.0 * np.arange(100) / 4096.0)
    out = gate_glitch(glitched, c, 4096.0)
    h, tp = int(GATE_HALF_S * 4096), int(GATE_TAPER_S * 4096)
    slow = 200.0 * np.sin(2 * np.pi * 2.0 * np.arange(len(x)) / 4096.0)
    # inside: the slow content, plus the glitch's own part below 10 Hz (about 5% here),
    # which the 15 Hz highpass removes downstream
    assert np.abs((out - slow)[c - h:c + h]).max() < 0.1 * amp
    # outside the gate and its taper: exactly the input
    np.testing.assert_allclose(out[:c - h - tp], glitched[:c - h - tp], atol=1e-9)
    np.testing.assert_allclose(out[c + h + tp:], glitched[c + h + tp:], atol=1e-9)


def test_known_glitches_only_inside_the_data():
    from ahsd.inference.preprocessing import KNOWN_GLITCHES, known_glitches_in
    d, g = KNOWN_GLITCHES[0]
    x = {d: np.zeros(4096 * 80)}
    assert known_glitches_in(x, {d: g - 40.0}, {d: 4096.0}) == [(d, g)]
    assert known_glitches_in(x, {d: g + 1.0}, {d: 4096.0}) == []            # another time
    assert known_glitches_in(x, {d: g - 80.05}, {d: 4096.0}) == []          # taper past the end
    assert known_glitches_in({"H1": x[d]}, {"H1": g - 40.0}, {"H1": 4096.0}) == []   # another detector

