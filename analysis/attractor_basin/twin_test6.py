"""Round 6: map the attractor basin now that it's reproducible in-distribution.

All in Gaussian design noise (fixed seed, so geometry is the only variable),
pseudo-truth params (m1 38.8, m2 33.4, dL 440) unless stated.

  DT   Δt scan: roll H1's signal by delta ms in [-12..+12]; L1 fixed.
  QS   q-truth scan at delta=+7 ms, fixed detector-frame Mc=31.3.
  TH   theta_jn = pi/2 (edge-on) at delta=+7 ms — does edge-on escape?
  CV   control: run the +7 ms basin condition and delta=0 through
       model/long_run_gated (different encoder generation) if loadable.
"""
import sys
from pathlib import Path

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))

import numpy as np

from twin_test import MODELS, DETS, analyze, design_noise
from twin_test4 import h1l1_lag_ms
import _common as C
from ahsd.data.bilby_pipeline import get_default_psd
from ahsd.inference import preprocessing as pp

SR = C.SR
NOISE_SEED = 42


def prep_with_dt(params, delta_ms):
    """Design-noise injection where H1's signal is rolled by delta_ms."""
    sig = C.raw_projected_signal(params, C.GW150914_GPS, detectors=DETS)
    sig["H1"] = np.roll(sig["H1"], int(round(delta_ms / 1000.0 * SR)))
    raw, psds = {}, {}
    for i, det in enumerate(DETS):
        raw[det] = design_noise(det, NOISE_SEED * 10 + i) + sig[det]
        psds[det] = get_default_psd(det)
    return pp.prepare_simulated(raw, psds=psds, seed=NOISE_SEED)


def m1_of(mc, q):
    return mc * (1 + q) ** 0.2 * q ** (-0.6)


def main():
    models = {name: C.load_model(path) for name, path in MODELS.items()}
    zeros_ab = np.zeros((3, 16), dtype=np.float32)
    truth = dict(C.GW150914_TRUTH)

    print("=== DT: Δt scan (design noise, pseudo-truth params) ===")
    for dms in (-12, -9, -7, -4, 0, 4, 5, 6, 7, 8, 9, 12):
        prep = prep_with_dt(truth, dms)
        lag = h1l1_lag_ms(prep.strain)
        for name, m in models.items():
            analyze(m, prep, truth, zeros_ab, f"dt {dms:+3d}ms(lag{lag:+5.1f}) / {name}")

    print("\n=== QS: q-truth scan at Δt=+7 ms (Mc fixed 31.3) ===")
    for qt in (0.50, 0.65, 0.80, 0.90, 1.00):
        m1 = m1_of(31.3, qt)
        p = dict(truth, mass_1=m1, mass_2=qt * m1)
        prep = prep_with_dt(p, +7)
        for name, m in models.items():
            analyze(m, prep, p, zeros_ab, f"q={qt:.2f} (m1 {m1:.1f}) / {name}")

    print("\n=== TH: edge-on at Δt=+7 ms ===")
    p = dict(truth, theta_jn=float(np.pi / 2))
    prep = prep_with_dt(p, +7)
    for name, m in models.items():
        analyze(m, prep, p, zeros_ab, f"edge dt+7 / {name}")

    print("\n=== CV: older-generation checkpoint control ===")
    try:
        import torch
        ck = torch.load(str(C.ROOT / "model" / "long_run_gated" / "best_model.pth"),
                        map_location="cpu", weights_only=False)
        print(f"  long_run_gated: epoch {ck['epoch']}, encoder_type "
              f"{ck['args'].get('encoder_type', 'conv')}, mass_param "
              f"{ck['args'].get('mass_param', 'm1m2')}")
        mold = C.load_model(str(C.ROOT / "model" / "long_run_gated" / "best_model.pth"))
        for dms in (0, +7):
            prep = prep_with_dt(truth, dms)
            analyze(mold, prep, truth, zeros_ab, f"CV dt{dms:+d} / long_run_gated")
    except Exception as e:
        print(f"  FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
