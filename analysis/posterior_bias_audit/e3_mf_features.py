#!/usr/bin/env python3
"""E3: How much (Mc, q) information is in the RAW MF features vs in the CONTEXT?

If MF features >> context, the MF->context pathway (the zero-init scalar
`mf_gate`, trained value 0.026) is destroying information that is present at
the input. That makes the gate/aggregation the primary actionable bottleneck,
not the flow and not the parameterization.
"""
import sys
from pathlib import Path
ROOT = Path("/Users/bibinthomas/Developer/PosteriFlow")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch
from experiments.remix_data import RemixDataset
from ahsd.models.lean_npe import LeanNPE

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ck = torch.load(ROOT / "model/lean_npe_mf_scratch/best_model.pth", map_location="cpu", weights_only=False)
a = dict(ck["args"])
model = LeanNPE(premerger=False, psd_cond=True, psd_bands=a.get("psd_bands", 16),
                encoder_type="coherent", mass_param="mc_q",
                geom_phase_residual=False, mf_tokens=True)
model.load_state_dict(ck["model_state_dict"]); model.to(DEV).eval()
print("mf_gate =", float(model.encoder.mf_gate))

ds = RemixDataset(str(ROOT / "data/dataset/memmap/validation"), remix=True, seed=1234,
                  real_noise_prob=0.0, return_asd_bands=True, psd_bands=a.get("psd_bands", 16))
ds.set_epoch(0)

MF, CTX, TRUTH, SNR = [], [], [], []
N, B = 1200, 32
for i0 in range(0, N, B):
    items = [ds[i] for i in range(i0, min(i0 + B, N))]
    strain = torch.stack([x[0] for x in items]).to(DEV).float()
    ab = torch.stack([x[4] for x in items]).to(DEV).float()
    pv = torch.stack([x[1] for x in items]).numpy()
    snr = torch.stack([x[3] for x in items]).numpy()
    with torch.no_grad():
        mf = model.encoder._mf_features(strain).cpu().numpy()
        ctx = model.encode(strain, ab).cpu().numpy()
    for k in range(len(items)):
        t1, t2 = pv[k, 0, 0], pv[k, 0, 1]
        TRUTH.append([np.log((t1 * t2) ** 0.6 / (t1 + t2) ** 0.2), t2 / t1])
        MF.append(mf[k]); CTX.append(ctx[k]); SNR.append(snr[k])
    if i0 % 320 == 0: print(f"  {i0}/{N}", flush=True)

MF = np.array(MF); CTX = np.array(CTX); TRUTH = np.array(TRUTH); SNR = np.array(SNR)
print("MF feature dim:", MF.shape, " context dim:", CTX.shape)
np.savez(Path(__file__).parent / "e3_raw.npz", MF=MF, CTX=CTX, TRUTH=TRUTH, SNR=SNR)

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

def r2(X, y, nonlin=False):
    X = StandardScaler().fit_transform(np.nan_to_num(X))
    est = (HistGradientBoostingRegressor(max_iter=300, random_state=0) if nonlin
           else RidgeCV(alphas=np.logspace(-3, 4, 30)))
    p = cross_val_predict(est, X, y, cv=5)
    return 1 - ((y - p) ** 2).mean() / y.var()

mc = np.exp(TRUTH[:, 0]); qq = TRUTH[:, 1]
subsets = {"ALL": np.ones(len(mc), bool),
           "Mc in bank [12,45]": (mc >= 12) & (mc <= 45),
           "Mc<12 (below bank)": mc < 12,
           "SNR>20": SNR > 20}
print("\n" + "=" * 88)
print(f"{'subset':22s} {'target':7s} {'MF_lin':>8s} {'MF_gbm':>8s} {'CTX_lin':>8s} {'CTX_gbm':>8s}")
print("=" * 88)
for nm, m in subsets.items():
    if m.sum() < 100: continue
    for j, pn in [(0, "logMc"), (1, "q")]:
        y = TRUTH[m, j]
        print(f"{nm:22s} {pn:7s} {r2(MF[m],y):8.3f} {r2(MF[m],y,True):8.3f} "
              f"{r2(CTX[m],y):8.3f} {r2(CTX[m],y,True):8.3f}")
    print("-" * 88)
