#!/usr/bin/env python3
"""Is inclination in the matched-filter trigger features, and does the
network use it?

Per validation event (one signal): |cos iota| predicted by
  npe     the model's posterior mean of |cos theta_jn|
  probe   gradient boosting on the trigger features (per-detector arrival
          time, SNR, phase; pairwise delays and phase differences)
  oracle  the same plus the TRUE sky position (ra, dec, psi): the most the
          per-detector amplitudes/phases can say once the sky is known
R^2 on held-out events, overall and at network SNR >= 15. probe >> npe means
the information is there and the network does not use it; probe ~ oracle ~ 0
means the features do not carry it.

    python scripts/inclination_probe.py --model model/lean_npe_bbh_v4/best_model.pth \
        --data data/dataset_bbh_v4 [--n 4000] [--real]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from remix_data import RemixDataset  # noqa: E402
from ahsd.inference.pipeline import load_model  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

T = 16384
J = PARAM_NAMES.index

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--real", action="store_true", help="real O3 noise instead of design Gaussian")
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = load_model(a.model, device=dev)
mf = model.encoder.mc_feat
assert getattr(mf, "trigger", False), "model has no trigger features (bank spec :trigger)"

ds = RemixDataset(f"{a.data}/memmap/validation", remix=False, seed=5, return_asd_bands=True,
                  real_noise_dir="data/noise_bank" if a.real else None,
                  real_noise_prob=1.0 if a.real else 0.0)
X, P, SN, AB = [], [], [], []
for i in range(len(ds)):
    x, p, n, snr, ab = ds[i]
    if int(n) != 1:
        continue
    X.append(x); P.append(p[0].numpy()); SN.append(float(snr)); AB.append(ab)
    if len(X) >= a.n:
        break
X = torch.stack(X).float(); P = np.stack(P); SN = np.array(SN); AB = torch.stack(AB).float()
lo_len = getattr(model, "lowrate_len", 0)

feats, npe = [], []
with torch.no_grad():
    for i in range(0, len(X), 64):
        x = X[i:i + 64].to(dev)
        ab = AB[i:i + 64].to(dev) if model.psd_cond else None
        hi, lo = x[..., :T], (x[..., T:] if lo_len else None)
        prof, _, _, pq = mf._profiles(hi, lo)
        feats.append(mf._trigger(hi, prof, pq).cpu().numpy())
        s = model.sample_posterior(x, n_samples=400, asd_bands=ab).cpu().numpy()
        npe.append(np.abs(np.cos(s[..., J("theta_jn")])).mean(1))
F = np.concatenate(feats); npe = np.concatenate(npe)
y = np.abs(np.cos(P[:, J("theta_jn")]))
sky = np.stack([np.cos(P[:, J("ra")]), np.sin(P[:, J("ra")]), np.sin(P[:, J("dec")]),
                np.cos(2 * P[:, J("psi")]), np.sin(2 * P[:, J("psi")])], 1)

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
rng = np.random.default_rng(0)
idx = rng.permutation(len(y)); tr, te = idx[: len(y) * 7 // 10], idx[len(y) * 7 // 10:]


def r2(pred, k):
    return float(1 - np.mean((y[k] - pred[k]) ** 2) / np.var(y[k]))


out = {"model": a.model, "noise": "real" if a.real else "gaussian", "n": int(len(y))}
preds = {"npe": npe}
for name, Z in (("probe", F), ("oracle", np.concatenate([F, sky], 1))):
    g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05).fit(Z[tr], y[tr])
    preds[name] = g.predict(Z)
loud = np.zeros(len(y), bool); loud[te] = SN[te] >= 15
allte = np.zeros(len(y), bool); allte[te] = True
for name, pr in preds.items():
    out[name] = {"r2_all": round(r2(pr, allte), 3), "r2_snr15": round(r2(pr, loud), 3)}
    print(f"{name:7s} R^2 |cos i|: all {out[name]['r2_all']:+.3f} | SNR>=15 {out[name]['r2_snr15']:+.3f} "
          f"(n={int(loud.sum())})", flush=True)
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
