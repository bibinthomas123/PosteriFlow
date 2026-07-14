"""
STEP 6 - waveform-family test (no dataset regeneration).

Generate matched evaluation injections in DESIGN Gaussian noise with
IMRPhenomXPHM (higher modes + precession) vs the training approximant
IMRPhenomXP, run the CURRENT model, and compare distance bias / rank
correlation / coverage. If XPHM is not meaningfully better, waveform mismatch
is rejected as the distance-bias driver.

Uses realistic BBH draws from ParameterSampler; each draw is injected with both
approximants so the comparison is paired.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from ahsd.data.parameter_sampler import ParameterSampler
from ahsd.data.bilby_pipeline import (BilbyNoiseGenerator, BilbySignalInjector,
                                      BilbyPreprocessor, get_default_psd)

OUT = Path(__file__).resolve().parent
model = C.load_model()
sampler = ParameterSampler()
ng = BilbyNoiseGenerator(C.SR, C.DUR)
inj = BilbySignalInjector(C.SR, C.DUR)
prep = BilbyPreprocessor(C.SR, C.DUR)
psds = {d: get_default_psd(d) for d in ["H1", "L1", "V1"]}
zero_ab = torch.zeros(1, 3, 16)

N = 150
res = {"IMRPhenomXP": [], "IMRPhenomXPHM": []}
rng = np.random.default_rng(0)
n_done = 0
attempts = 0
while n_done < N and attempts < N * 4:
    attempts += 1
    p = sampler.sample_parameters()
    if p.get("event_type") != "BBH":
        continue
    seed = int(rng.integers(2 ** 31))
    row = {}
    ok = True
    for approx in ("IMRPhenomXP", "IMRPhenomXPHM"):
        pa = dict(p); pa["approximant"] = approx
        white, ss = {}, []
        for det in ["H1", "L1"]:
            noise = ng.generate(det, psds[det], seed=seed)   # SAME noise both arms
            strain, snr = inj.inject(noise, pa, det, psds[det])
            white[det] = prep.preprocess(strain, psds[det], detector=det)
            ss.append(snr)
        net = float(np.sqrt(sum(x ** 2 for x in ss)))
        if net < 6:
            ok = False; break
        r = np.random.default_rng(seed)
        s3 = torch.from_numpy(np.stack([white["H1"], white["L1"],
                              r.standard_normal(C.T_LEN).astype(np.float32)]))[None].float()
        with torch.no_grad():
            samp = model.sample_posterior(s3, rank=0, n_samples=1500, asd_bands=zero_ab)[0].numpy()
        d = samp[:, C.JD]; td = float(p["luminosity_distance"])
        lo, hi = np.quantile(d, [0.05, 0.95])
        row[approx] = dict(true_dL=td, snr=net, pred_med=float(np.median(d)),
                           in90=bool(lo <= td <= hi))
    if not ok:
        continue
    for approx in res:
        res[approx].append(row[approx])
    n_done += 1

report = {}
for approx, rows in res.items():
    td = np.array([r["true_dL"] for r in rows])
    pm = np.array([r["pred_med"] for r in rows])
    ratio = pm / td
    report[approx] = dict(
        n=len(rows), mean_snr=round(float(np.mean([r["snr"] for r in rows])), 1),
        median_dL_ratio=round(float(np.median(ratio)), 3),
        mean_abs_log_ratio=round(float(np.mean(np.abs(np.log(ratio)))), 3),
        spearman_pred_vs_true=round(float(spearmanr(td, pm).correlation), 3),
        coverage90=round(float(np.mean([r["in90"] for r in rows])), 3))

json.dump(report, open(OUT / "step6_waveform.json", "w"), indent=2, default=float)
print("\n===== STEP 6 : WAVEFORM (design noise, paired draws) =====")
print(f"{'approximant':16s} {'n':>4s} {'SNR':>6s} {'dL ratio':>9s} {'|log ratio|':>12s} {'spearman':>9s} {'cov90':>7s}")
for approx, r in report.items():
    print(f"{approx:16s} {r['n']:4d} {r['mean_snr']:6.1f} {r['median_dL_ratio']:9.3f} "
          f"{r['mean_abs_log_ratio']:12.3f} {r['spearman_pred_vs_true']:9.3f} {r['coverage90']:7.3f}")
print(f"wrote {OUT/'step6_waveform.json'}")
