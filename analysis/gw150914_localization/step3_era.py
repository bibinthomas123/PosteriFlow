"""
STEP 3 - noise-era evaluation on existing data.

Recolour the validation design signals into real bank noise from each era
(O1/O2/O3; O4 not in the bank), the same way training did, and report distance
bias / rank correlation / 90% coverage per era. Tests whether O1 is
systematically worse (it is the GW150914 era).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from remix_data import RemixDataset


def era(gps):
    if gps < 1137254417:  return "O1"
    if gps < 1187733618:  return "O2"
    return "O3"


OUT = Path(__file__).resolve().parent
model = C.load_model()
bank = C.ROOT / "data/noise_bank"

# era index per detector (order matches sorted glob == real_bank order)
era_idx = {}
for d in ("H1", "L1", "V1"):
    files = sorted(bank.glob(f"{d}_*_strain.npy"))
    era_idx[d] = np.array([era(int(f.stem.split("_")[1])) for f in files])

report = {}
N = 150
for E in ("O1", "O2", "O3"):
    ds = RemixDataset(str(C.ROOT / "data/dataset/memmap/validation"), remix=True, seed=7,
                      real_noise_dir=str(bank), real_noise_prob=1.0,
                      return_asd_bands=True, psd_bands=16)
    # restrict the bank to this era
    for d in ("H1", "L1", "V1"):
        keep = np.where(era_idx[d] == E)[0]
        if len(keep) == 0:
            keep = np.arange(len(ds.real_bank[d]))
        ds.real_bank[d] = [ds.real_bank[d][k] for k in keep]
        ds.recolor[d] = [ds.recolor[d][k] for k in keep]
    ds.set_epoch(0)
    true_dL, pred_med, in90, snrs = [], [], [], []
    for i in range(N):
        it = ds[i]
        strain = it[0][None].float()
        ab = it[4][None].float() if len(it) > 4 else None
        theta = it[1][0].numpy()
        with torch.no_grad():
            s = model.sample_posterior(strain, rank=0, n_samples=2000, asd_bands=ab)[0].numpy()
        dpost = s[:, C.JD]
        td = float(theta[C.JD])
        true_dL.append(td); pred_med.append(float(np.median(dpost)))
        lo, hi = np.quantile(dpost, [0.05, 0.95]); in90.append(lo <= td <= hi)
        snrs.append(float(it[3]))
    true_dL = np.array(true_dL); pred_med = np.array(pred_med)
    ratio = pred_med / true_dL
    report[E] = dict(
        n=N, mean_snr=round(float(np.mean(snrs)), 1),
        median_dL_ratio=round(float(np.median(ratio)), 3),
        mean_abs_log_ratio=round(float(np.mean(np.abs(np.log(ratio)))), 3),
        spearman_pred_vs_true=round(float(spearmanr(true_dL, pred_med).correlation), 3),
        coverage90=round(float(np.mean(in90)), 3))
    r = report[E]
    print(f"{E}: n={N} SNR~{r['mean_snr']}  dL ratio(med)={r['median_dL_ratio']}  "
          f"spearman={r['spearman_pred_vs_true']}  cov90={r['coverage90']}")

json.dump(report, open(OUT / "step3_era.json", "w"), indent=2, default=float)
print("\n===== STEP 3 : ERA =====")
print(f"{'era':4s} {'SNR':>6s} {'dL ratio':>9s} {'spearman':>9s} {'cov90':>7s}")
for E, r in report.items():
    print(f"{E:4s} {r['mean_snr']:6.1f} {r['median_dL_ratio']:9.3f} "
          f"{r['spearman_pred_vs_true']:9.3f} {r['coverage90']:7.3f}")
print(f"wrote {OUT/'step3_era.json'}")
