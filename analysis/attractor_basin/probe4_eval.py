"""Probe 4 result: amortization-vs-representation control. A model trained
ONLY on a narrow, GW150914-sky-locked population (Mc in [25,40], q in
[0.3,1.0] broad, ra/dec fixed at GW150914's true sky location -- never
amortizing across the full BNS/NSBH/light-BBH population) is tested on the
ACTUAL held-out GW150914 strain (never in its training set) plus a couple of
in-narrow-distribution design twins as a sanity check.

If this model recovers sensible q on real GW150914, the population-wide
amortization was the cost paid by lean_npe_mf_scratch. If it STILL shows the
(30,30)/q->1 collapse, the architecture/optimization can't reach the answer
even without having to share weights across the whole population -- pointing
back at representation/optimization, not amortization.

NOTE: best_model.pth in this run was checkpoint-selected on val_nll computed
against the FULL dataset's validation split (symlinked in as a pragmatic
stand-in -- see REPORT.md) which is a totally different distribution from
this narrow-trained model, so best_model.pth is actually just the epoch-1,
barely-trained checkpoint. Use last_model.pth (epoch 30, fully trained on
the narrow task; rail 0.97->0.10, dist_corr 0.02->0.40 over training).
"""
import sys
from pathlib import Path

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))

import numpy as np

from twin_test import analyze, make_twin
import _common as C

CKPT = str(C.ROOT / "model" / "lean_npe_gw150914_narrow" / "last_model.pth")


def main():
    import sys as _sys
    ckpt = _sys.argv[1] if len(_sys.argv) > 1 else CKPT
    m = C.load_model(ckpt)
    print(f"loaded: epoch {m._ckpt_meta['epoch']} val_nll {m._ckpt_meta['val_nll']:.3f} "
          f"psd_cond {m._ckpt_meta['psd_cond']}")
    zeros_ab = np.zeros((3, 16), dtype=np.float32)
    truth = dict(C.GW150914_TRUTH)  # m1 38.8 m2 33.4 q 0.861, ra/dec = training sky lock

    print("\n=== in-narrow-distribution sanity check: design twins at truth params ===")
    for seed in range(3):
        prep, snr = make_twin(truth, seed)
        print(f" twin seed {seed}: network SNR opt {snr['network']['optimal_snr']:.1f}")
        analyze(m, prep, truth, zeros_ab, f"twin s{seed}")

    print("\n=== DECISIVE TEST: REAL GW150914 (held out, never in narrow training set) ===")
    prep_real = C.prepare_gw150914()
    ab_real = C.asd_bands_of(prep_real)
    analyze(m, prep_real, truth, ab_real, "real GW150914")


if __name__ == "__main__":
    main()
