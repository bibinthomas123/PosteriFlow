"""
STEP 4c - is the amplitude->distance collapse specific to heavy ~equal-mass BBH?

For several source CLASSES, sweep luminosity distance in DESIGN Gaussian noise
(asd_bands=0, amplitude == distance cue) and measure the response of the
inferred distance to the true distance. A model that extracts distance has
slope d(dL_med)/d(true_dL) ~ 1 (and Spearman ~1); a collapsed one ~0.

Restrict to in-distribution SNR (roughly 15-45) so extreme-SNR OOD does not
confound.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

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

# (name, m1, m2, dL grid). dL grids chosen so SNR lands ~15-45 in design noise.
CLASSES = [
    ("light_sym  Mc~9  q1.0",  11.0, 11.0, [200, 350, 500, 700, 900]),
    ("light_asym Mc~8  q0.35", 18.0, 6.0,  [150, 250, 350, 500, 650]),
    ("moderate   Mc~17 q0.8",  22.0, 18.0, [400, 700, 1000, 1400, 1800]),
    ("heavy_GW150914 Mc~29",   38.8, 33.4, [800, 1100, 1500, 1900, 2100]),
]
zero_ab = np.zeros((3, 16), dtype=np.float32)
out = {}
for name, m1, m2, grid in CLASSES:
    pts = []
    for dL in grid:
        P = dict(C.GW150914_TRUTH)
        P.update(mass_1=float(m1), mass_2=float(m2), luminosity_distance=float(dL))
        meds, snrs = [], []
        for seed in range(3):
            white, ss = {}, []
            for det in ["H1", "L1"]:
                noise = noise_gen.generate(det, psds[det], seed=9000 + seed * 13 + int(dL))
                strain, snr = injector.inject(noise, P, det, psds[det])
                white[det] = prep_bp.preprocess(strain, psds[det], detector=det)
                ss.append(snr)
            rng = np.random.default_rng(seed)
            s3 = np.stack([white["H1"], white["L1"],
                           rng.standard_normal(C.T_LEN).astype(np.float32)])
            s = C.sample_from_strain(model, s3, zero_ab, n=4000)
            meds.append(float(np.median(s[:, C.JD])))
            snrs.append(float(np.sqrt(sum(x ** 2 for x in ss))))
        pts.append(dict(true_dL=dL, snr=round(float(np.mean(snrs)), 1),
                        dL_med=round(float(np.mean(meds)), 0)))
    # response slope over in-range SNR points
    inr = [p for p in pts if 12 <= p["snr"] <= 60]
    td = np.array([p["true_dL"] for p in inr], float)
    md = np.array([p["dL_med"] for p in inr], float)
    slope = float(np.polyfit(td, md, 1)[0]) if len(inr) >= 2 else float("nan")
    rho = float(spearmanr(td, md).correlation) if len(inr) >= 2 else float("nan")
    out[name] = dict(points=pts, slope=round(slope, 3), spearman=round(rho, 3))
    print(f"\n{name}")
    for p in pts:
        print(f"   true_dL={p['true_dL']:5d} SNR={p['snr']:5.1f} -> dL_med={p['dL_med']:6.0f}")
    print(f"   response slope={slope:+.2f}  spearman={rho:+.2f}  (1=extracts, 0=collapsed)")

json.dump(out, open(OUT / "step4c_sourceclass.json", "w"), indent=2, default=float)

print("\n===== STEP 4c : SOURCE-CLASS DISTANCE RESPONSE (design noise) =====")
print(f"{'class':26s} {'slope':>7s} {'spearman':>9s}")
for name, d in out.items():
    print(f"{name:26s} {d['slope']:7.2f} {d['spearman']:9.2f}")
print(f"\nwrote {OUT/'step4c_sourceclass.json'}")
