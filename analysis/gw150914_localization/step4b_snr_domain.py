"""
STEP 4b - disentangle real-noise-DOMAIN from SNR-REGIME.

Fixed GW150914-like masses/orientation; vary luminosity distance in DESIGN
Gaussian noise so SNR sweeps 98 -> ~20. Recovery ratio vs SNR in the design
(training) domain. Compare to the real-O1 semi-real point (SNR~24 @ dL440,
ratio ~1.5 from step4).

If design-domain SNR~24 recovers well  -> real-noise domain is the culprit.
If design-domain SNR~24 also biases    -> low-SNR heavy-BBH regime itself.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from ahsd.data.bilby_pipeline import (BilbyNoiseGenerator, BilbySignalInjector,
                                      BilbyPreprocessor, get_default_psd)

OUT = Path(__file__).resolve().parent
model = C.load_model()
prep_bp = BilbyPreprocessor(C.SR, C.DUR)
noise_gen = BilbyNoiseGenerator(C.SR, C.DUR)
injector = BilbySignalInjector(C.SR, C.DUR)
psds = {d: get_default_psd(d) for d in ["H1", "L1", "V1"]}

rows = []
for dL in [440, 700, 1000, 1400, 1800, 2100]:
    P = dict(C.GW150914_TRUTH); P["luminosity_distance"] = float(dL)
    ratios, meds, snrs = [], [], []
    for seed in range(4):
        white, ss = {}, []
        for det in ["H1", "L1"]:
            noise = noise_gen.generate(det, psds[det], seed=5000 + seed * 7 + dL)
            strain, snr = injector.inject(noise, P, det, psds[det])
            white[det] = prep_bp.preprocess(strain, psds[det], detector=det)
            ss.append(snr)
        rng = np.random.default_rng(seed)
        strain3 = np.stack([white["H1"], white["L1"],
                            rng.standard_normal(C.T_LEN).astype(np.float32)])
        zero_ab = np.zeros((3, 16), dtype=np.float32)
        s = C.sample_from_strain(model, strain3, zero_ab, n=5000)
        med = float(np.median(s[:, C.JD]))
        meds.append(med); ratios.append(med / dL)
        snrs.append(float(np.sqrt(sum(x ** 2 for x in ss))))
    rows.append(dict(true_dL=dL, snr=round(float(np.mean(snrs)), 1),
                     dL_med=round(float(np.mean(meds)), 0),
                     ratio=round(float(np.mean(ratios)), 2),
                     ratio_sd=round(float(np.std(ratios)), 2)))
    print(f"  design dL={dL:5d}  SNR={rows[-1]['snr']:5.1f}  "
          f"dL_med={rows[-1]['dL_med']:6.0f}  ratio={rows[-1]['ratio']:.2f}")

json.dump({"design_snr_sweep": rows}, open(OUT / "step4b_snr_domain.json", "w"),
          indent=2, default=float)

print("\n===== STEP 4b : DESIGN-DOMAIN dL/SNR SWEEP =====")
print(f"{'true dL':>8s} {'SNR':>6s} {'dL med':>8s} {'ratio':>7s}")
for r in rows:
    print(f"{r['true_dL']:8d} {r['snr']:6.1f} {r['dL_med']:8.0f} {r['ratio']:7.2f}")
print("\nreal-O1 semi-real (step4): SNR~28, true dL=440, ratio~1.49")
print(f"wrote {OUT/'step4b_snr_domain.json'}")
