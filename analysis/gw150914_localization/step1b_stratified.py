"""
STEP 1b - confirm the two remaining legs of the mechanism.

(1) DEGENERACY leg: does the encoder context carry the extrinsic geometry
    (inclination/sky/pol) needed to convert amplitude -> distance? If ~0, the
    distance-inclination-amplitude degeneracy cannot be broken and distance
    reverts to the chirp-mass-conditional prior.

(2) SOURCE-CLASS leg: is amplitude (SNR) and distance extraction from the
    context worse for heavy (high-Mc) BBH? Stratify context->SNR / context->dL
    R^2 by chirp-mass tercile.

Also saves contexts+params npz for reuse.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from ahsd.models.lean_npe import PARAM_NAMES
from remix_data import RemixDataset

OUT = Path(__file__).resolve().parent
model = C.load_model()
for p in model.parameters():
    p.requires_grad_(False)

ds = RemixDataset(str(C.ROOT / "data/dataset/memmap/validation"), remix=False, seed=1234)
N = min(2500, len(ds))
strain_list, theta_list, snr_list = [], [], []
for i in range(N):
    it = ds[i]
    strain_list.append(it[0]); theta_list.append(it[1][0]); snr_list.append(it[3])
strain = torch.stack(strain_list)
theta = torch.stack(theta_list).numpy()
net_snr = torch.stack(snr_list).numpy().astype(float)
with torch.no_grad():
    ctx = torch.cat([model.encoder(strain[i:i + 64]) for i in range(0, N, 64)]).numpy()

m1, m2, dL = theta[:, 0], theta[:, 1], theta[:, C.JD]
mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
np.savez(OUT / "val_contexts.npz", ctx=ctx, theta=theta, net_snr=net_snr,
         param_names=np.array(PARAM_NAMES))

kf = KFold(5, shuffle=True, random_state=0)
def r2(X, y, idx=None):
    X = X if idx is None else X[idx]
    y = y if idx is None else y[idx]
    X = X if X.ndim == 2 else X[:, None]
    return float(r2_score(y, cross_val_predict(
        HistGradientBoostingRegressor(max_depth=4, max_iter=250, random_state=0),
        X, y, cv=kf)))

# ── (1) extrinsic geometry info in the context ───────────────────────────────
targets = {n: theta[:, PARAM_NAMES.index(n)] for n in
           ["theta_jn", "ra", "dec", "psi", "luminosity_distance", "mass_1"]}
extr = {"context_R2": {}}
for n, y in targets.items():
    yy = np.log(y) if n in ("luminosity_distance", "mass_1") else y
    extr["context_R2"][n] = round(r2(ctx, yy), 3)
print("== context -> parameter R^2 ==")
for n, v in extr["context_R2"].items():
    print(f"   {n:22s} {v:+.3f}")

# ── (2) Mc-stratified amplitude & distance extraction ────────────────────────
log_snr, log_dL = np.log(net_snr), np.log(dL)
terc = np.quantile(mc, [1/3, 2/3])
bins = {"light": mc < terc[0], "medium": (mc >= terc[0]) & (mc < terc[1]),
        "heavy": mc >= terc[1]}
strat = {}
for name, mask in bins.items():
    idx = np.where(mask)[0]
    strat[name] = dict(
        n=int(mask.sum()),
        mc_range=[round(float(mc[mask].min()), 1), round(float(mc[mask].max()), 1)],
        ctx_to_logSNR_r2=round(r2(ctx, log_snr, idx), 3),
        ctx_to_logdL_r2=round(r2(ctx, log_dL, idx), 3))
print("\n== Mc-stratified extraction (within-bin R^2) ==")
print(f"{'bin':8s} {'n':>5s} {'Mc range':>14s} {'ctx->SNR':>9s} {'ctx->dL':>8s}")
for name, d in strat.items():
    print(f"{name:8s} {d['n']:5d} {str(d['mc_range']):>14s} "
          f"{d['ctx_to_logSNR_r2']:9.3f} {d['ctx_to_logdL_r2']:8.3f}")

out = {"extrinsic_info": extr, "mc_stratified": strat,
       "note": "extrinsic R2 ~0 => degeneracy unbreakable; dL R2 drop in heavy bin => source-class"}
json.dump(out, open(OUT / "step1b_stratified.json", "w"), indent=2, default=float)
print(f"\nwrote {OUT/'step1b_stratified.json'}")
