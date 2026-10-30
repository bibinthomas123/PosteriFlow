"""Probe 3 result: does the mf_tokens-only ablation (no aux_mass, no
delay_aug, trained from scratch, 15 epochs) resolve the basin without those
extra levers? Reuses twin_test6/twin_test's exact basin-cell methodology
(design-noise twin at Delta t=+7ms vs -7ms, plus real GW150914) against the
new checkpoint model/lean_npe_mf_only_ablation_final/best_model.pth.
"""
import sys
from pathlib import Path

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))

import numpy as np

from twin_test import analyze
from twin_test6 import prep_with_dt
import _common as C

CKPT = str(C.ROOT / "model" / "lean_npe_mf_only_ablation_final" / "best_model.pth")


def main():
    m = C.load_model(CKPT)
    print(f"loaded: epoch {m._ckpt_meta['epoch']} val_nll {m._ckpt_meta['val_nll']:.3f} "
          f"psd_cond {m._ckpt_meta['psd_cond']}")
    zeros_ab = np.zeros((3, 16), dtype=np.float32)
    truth = dict(C.GW150914_TRUTH)

    print("\n=== DT scan (design noise, pseudo-truth params) ===")
    for dms in (-9, -7, -4, 0, 4, 7, 9):
        prep = prep_with_dt(truth, dms)
        analyze(m, prep, truth, zeros_ab, f"dt {dms:+3d}ms")

    print("\n=== QS: q-truth scan at Delta t=+7 ms (Mc fixed 31.3) ===")
    from twin_test6 import m1_of
    for qt in (0.50, 0.65, 0.80, 0.90, 1.00):
        m1 = m1_of(31.3, qt)
        p = dict(truth, mass_1=m1, mass_2=qt * m1)
        prep = prep_with_dt(p, +7)
        analyze(m, prep, p, zeros_ab, f"q={qt:.2f} (m1 {m1:.1f})")

    print("\n=== REAL GW150914 ===")
    prep_real = C.prepare_gw150914()
    ab_real = C.asd_bands_of(prep_real)
    analyze(m, prep_real, truth, ab_real, "real")


if __name__ == "__main__":
    main()
