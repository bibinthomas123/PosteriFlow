#!/usr/bin/env python3
"""Does the model confuse source types, and does that drive per-type distance bias?

    python scripts/type_confusion_probe.py model/<run>/best_model.pth data/<ds> [--n 3000]

For validation events of each true source type (BBH / NSBH / BNS by true
masses), reports the fraction of posterior samples that land in each type's
mass region, and the distance bias (E[log d_true - log d_hat]) using all
samples vs only the samples of the CORRECT type. If restricting to the right
type removes most of the bias, the distance error is type mixing, not a
distance-pathway failure.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "."); sys.path.insert(0, "src")
from ahsd.inference.pipeline import load_model  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("data")
ap.add_argument("--n", type=int, default=3000); ap.add_argument("--k", type=int, default=400)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = load_model(a.model, dev)[0].eval()
ds = RemixDataset(str(Path(a.data) / "memmap" / "validation"), remix=False, seed=1234,
                  return_asd_bands=True, psd_bands=model.encoder.psd_bands if model.psd_cond else 16)
JD = PARAM_NAMES.index("luminosity_distance")


def typ(m1, m2):
    return np.where(m1 < 3, 0, np.where(m2 < 3, 1, 2))  # 0 BNS, 1 NSBH, 2 BBH


T, S = [], []
with torch.no_grad():
    for i in range(0, min(a.n, len(ds)), 64):
        it = [ds[j] for j in range(i, min(i + 64, a.n, len(ds)))]
        x = torch.stack([v[0] for v in it]).float().to(dev)
        ab = torch.stack([v[4] for v in it]).float().to(dev)
        S.append(model.sample_posterior(x, n_samples=a.k, asd_bands=ab if model.psd_cond else None).cpu().numpy())
        T.append(torch.stack([v[1][0] for v in it]).numpy())
T, S = np.concatenate(T), np.concatenate(S)
tt = typ(T[:, 0], T[:, 1])
st = typ(S[:, :, 0], S[:, :, 1])
names = ["BNS", "NSBH", "BBH"]
print(f"{a.model}: {len(T)} validation events, {a.k} samples each\n")
print("true type | posterior mass in  BNS   NSBH   BBH  | dist bias all samples | correct-type samples only")
for t in (2, 1, 0):
    k = tt == t
    if not k.any():
        continue
    frac = [(st[k] == u).mean() for u in (0, 1, 2)]
    d = S[k][:, :, JD]; dt = T[k, JD]
    b_all = np.exp(np.mean(np.log(dt) - np.log(np.median(d, axis=1))))
    corr = []
    for e in range(k.sum()):
        m = st[k][e] == t
        if m.sum() >= 20:
            corr.append(np.log(dt[e]) - np.log(np.median(d[e][m])))
    b_c = np.exp(np.mean(corr)) if corr else float("nan")
    print(f"{names[t]:9s} (n={k.sum():4d}) |      {frac[0]:5.2f}  {frac[1]:5.2f}  {frac[2]:5.2f} |"
          f"  true/pred x{b_all:.2f}          |  x{b_c:.2f}  ({len(corr)} events with >=20 right-type samples)")
