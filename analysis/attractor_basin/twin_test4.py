"""Twin test round 4: perturb the REAL on-source input and see what breaks
the attractor. All conditions use asd_bands=0 (round 3 B showed it inert).

  ctrl        — on-source strain as-is (collapse baseline)
  scale s     — whitened strain x s for s in 0.5/0.7/1.5/2.0 (distance move)
  roll +-dt   — circular time-shift (training's own augmentation)
  flip        — strain x -1
  swapH1L1    — exchange H1 and L1 channels (breaks cross-det geometry)
  V1 reseed   — fresh white noise in the V1 slot

Also prints the H1-L1 arrival lag (peak of whitened cross-correlation) for the
real event vs an injected twin, to test whether the pseudo-truth sky/psi even
reproduces the real inter-detector geometry.
"""
import sys
from pathlib import Path

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))

import numpy as np

from twin_test import MODELS, DETS, analyze
import _common as C

SR = C.SR


def h1l1_lag_ms(strain):
    """Lag (ms) of the peak |xcorr| between H1 and L1 whitened channels,
    searched within +-15 ms around zero, using the central 1 s."""
    from scipy.signal import correlate
    T = strain.shape[1]
    a = strain[0, T // 2 - SR // 2: T // 2 + SR // 2].astype(np.float64)
    b = strain[1, T // 2 - SR // 2: T // 2 + SR // 2].astype(np.float64)
    xc = correlate(a, b, mode="full")
    center = len(a) - 1
    w = int(0.015 * SR)
    seg = xc[center - w: center + w + 1]
    return float((np.argmax(np.abs(seg)) - w) / SR * 1000.0)


def main():
    models = {name: C.load_model(path) for name, path in MODELS.items()}
    for name, m in models.items():
        print(f"model {name}: epoch {m._ckpt_meta['epoch']} val_nll {m._ckpt_meta['val_nll']:.3f}")
    zeros_ab = np.zeros((3, 16), dtype=np.float32)
    truth = dict(C.GW150914_TRUTH)

    prep = C.prepare_gw150914()
    S = prep.strain.copy()
    print(f"\nreal on-source H1-L1 lag: {h1l1_lag_ms(S):+.2f} ms (literature: ~+7 ms L1 first)")
    try:
        prep_tw, _ = C.inject_into_real_noise(C.GW150914_GPS + 16.0, truth,
                                              detectors=DETS, seed=0)
        print(f"pseudo-truth twin H1-L1 lag: {h1l1_lag_ms(prep_tw.strain):+.2f} ms")
    except Exception as e:
        print(f"twin lag check FAILED: {e}")

    conds = [("ctrl", S)]
    for s in (0.5, 0.7, 1.5, 2.0):
        conds.append((f"scale x{s}", S * s))
    for dt in (-1.0, -0.5, +0.5, +1.0):
        conds.append((f"roll {dt:+.1f}s", np.roll(S, int(dt * SR), axis=1)))
    conds.append(("flip (x-1)", -S))
    conds.append(("swap H1<->L1", S[[1, 0, 2]]))
    rng = np.random.default_rng(7)
    Sv = S.copy(); Sv[2] = rng.standard_normal(S.shape[1]).astype(np.float32)
    conds.append(("V1 reseed", Sv))

    class P:  # minimal stand-in so analyze() can take raw arrays
        def __init__(self, strain): self.strain = strain

    for label, arr in conds:
        print(f"\n--- {label} ---")
        for name, m in models.items():
            analyze(m, P(arr.astype(np.float32)), truth, zeros_ab, f"{label} / {name}")


if __name__ == "__main__":
    main()
