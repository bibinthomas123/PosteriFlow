#!/usr/bin/env python3
"""E1: Is the posterior width set by the ENCODER's information or by the FLOW?

For each in-distribution validation event (rank-0 signal, Gaussian/design noise):
  - context c  = encoder(strain)
  - posterior  = flow(.|c)  -> width in log Mc and in q
  - probe      = ridge/MLP fit c -> (log Mc, q) on a train split, eval held-out

If posterior_std ~= probe_residual_std  -> the flow faithfully reports the
   information in the context => ENCODER-limited.
If posterior_std >> probe_residual_std  -> the flow is broadening beyond the
   context's information => FLOW/OBJECTIVE-limited.
Also compares both against the SNR-scaling expectation for Mc.
"""
import sys, json
from pathlib import Path
ROOT = Path("/Users/bibinthomas/Developer/PosteriFlow")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

import numpy as np, torch
from experiments.remix_data import RemixDataset
from ahsd.models.lean_npe import LeanNPE

CKPT = sys.argv[1] if len(sys.argv) > 1 else "model/lean_npe_mf_scratch/best_model.pth"
N_EVENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
N_POST = 512
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

ck = torch.load(ROOT / CKPT, map_location="cpu", weights_only=False)
a = dict(ck["args"])
model = LeanNPE(premerger=a.get("premerger", False),
                psd_cond=a.get("psd_cond", False) or False,
                psd_bands=a.get("psd_bands", 16),
                encoder_type=a.get("encoder_type", "conv"),
                mass_param=a.get("mass_param", "m1m2"),
                geom_phase_residual=a.get("geom_phase_residual", False) or False,
                mf_tokens=a.get("mf_tokens", False) or False)
model.load_state_dict(ck["model_state_dict"])
model.to(DEV).eval()
print(f"loaded {CKPT}  ep={ck['epoch']} mass_param={a.get('mass_param')} mf_tokens={a.get('mf_tokens')} dev={DEV}")

ds = RemixDataset(str(ROOT / "data/dataset/memmap/validation"), remix=True, seed=1234,
                  real_noise_prob=0.0, return_asd_bands=True, psd_bands=a.get("psd_bands", 16))
ds.set_epoch(0)
print("val events:", len(ds))

CTX, TRUTH, SNR, PMED, PLO, PHI, PSTD = [], [], [], [], [], [], []
B = 32
idx = list(range(min(N_EVENTS, len(ds))))
for i0 in range(0, len(idx), B):
    items = [ds[i] for i in idx[i0:i0 + B]]
    strain = torch.stack([x[0] for x in items]).to(DEV).float()
    ab = torch.stack([x[4] for x in items]).to(DEV).float()
    pv = torch.stack([x[1] for x in items]).numpy()          # [B, max_sig, 11]
    snr = torch.stack([x[3] for x in items]).numpy()
    with torch.no_grad():
        ctx = model.encode(strain, ab)
        s = model.sample_posterior(strain, rank=0, n_samples=N_POST, asd_bands=ab)  # [B,N,11]
    s = s.cpu().numpy()
    m1, m2 = s[..., 0], s[..., 1]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    q = np.clip(m2 / np.maximum(m1, 1e-6), 1e-4, 1.0)
    for k in range(len(items)):
        t1, t2 = pv[k, 0, 0], pv[k, 0, 1]
        TRUTH.append([np.log((t1 * t2) ** 0.6 / (t1 + t2) ** 0.2), t2 / t1])
        CTX.append(ctx[k].cpu().numpy()); SNR.append(snr[k])
        lmc = np.log(mc[k])
        PMED.append([np.median(lmc), np.median(q[k])])
        PSTD.append([lmc.std(), q[k].std()])
        PLO.append([np.percentile(lmc, 5), np.percentile(q[k], 5)])
        PHI.append([np.percentile(lmc, 95), np.percentile(q[k], 95)])
    if i0 % 320 == 0: print(f"  {i0}/{len(idx)}", flush=True)

CTX = np.array(CTX); TRUTH = np.array(TRUTH); SNR = np.array(SNR)
PMED = np.array(PMED); PSTD = np.array(PSTD); PLO = np.array(PLO); PHI = np.array(PHI)
np.savez(Path(__file__).parent / "e1_raw.npz", CTX=CTX, TRUTH=TRUTH, SNR=SNR,
         PMED=PMED, PSTD=PSTD, PLO=PLO, PHI=PHI)

# ---- probes: context -> truth ----
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

n = len(CTX); ntr = int(0.7 * n)
rs = np.random.RandomState(0); perm = rs.permutation(n)
tr, te = perm[:ntr], perm[ntr:]
sc = StandardScaler().fit(CTX[tr]); Xtr, Xte = sc.transform(CTX[tr]), sc.transform(CTX[te])

names = ["log Mc", "q"]
out = {"ckpt": CKPT, "n": int(n), "median_snr": float(np.median(SNR))}
print("\n" + "=" * 96)
print(f"{'target':8s} {'probe_ridge_res':>15s} {'probe_mlp_res':>13s} {'post_std':>9s} "
      f"{'post_std/probe':>14s} {'post_bias':>10s} {'truth_std':>9s}")
print("=" * 96)
for j, nm in enumerate(names):
    y = TRUTH[:, j]
    ridge = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(Xtr, y[tr])
    r_res = float(np.std(y[te] - ridge.predict(Xte)))
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=600, random_state=0,
                       early_stopping=True).fit(Xtr, y[tr])
    m_res = float(np.std(y[te] - mlp.predict(Xte)))
    ps = float(np.mean(PSTD[te, j])); bias = float(np.mean(PMED[te, j] - y[te]))
    best = min(r_res, m_res)
    print(f"{nm:8s} {r_res:15.4f} {m_res:13.4f} {ps:9.4f} {ps/best:14.2f} {bias:10.4f} {np.std(y):9.4f}")
    out[nm] = dict(ridge_res=r_res, mlp_res=m_res, post_std=ps, ratio=ps / best,
                   post_bias=bias, truth_std=float(np.std(y)))

# ---- Mc width vs SNR (physics check: sigma_logMc should scale ~ 1/SNR) ----
print("\nMc posterior fractional width and q width vs network SNR:")
bins = [(0, 12), (12, 20), (20, 35), (35, 60), (60, 200)]
out["snr_bins"] = []
for lo, hi in bins:
    m = (SNR >= lo) & (SNR < hi)
    if m.sum() < 15: continue
    row = dict(snr_lo=lo, snr_hi=hi, n=int(m.sum()),
               med_snr=float(np.median(SNR[m])),
               sigma_logMc_post=float(np.mean(PSTD[m, 0])),
               sigma_logMc_x_snr=float(np.mean(PSTD[m, 0]) * np.median(SNR[m])),
               q_90ci_width=float(np.mean(PHI[m, 1] - PLO[m, 1])),
               q_post_std=float(np.mean(PSTD[m, 1])))
    out["snr_bins"].append(row)
    print(f"  SNR[{lo:3d},{hi:3d}) n={m.sum():4d} medSNR={row['med_snr']:6.1f} "
          f"sig_logMc={row['sigma_logMc_post']:.4f} (x SNR = {row['sigma_logMc_x_snr']:6.2f}) "
          f"q_90CIw={row['q_90ci_width']:.3f} q_std={row['q_post_std']:.3f}")

# q prior width for reference
print(f"\nq truth std (training prior) = {np.std(TRUTH[:,1]):.4f}; "
      f"mean posterior q std = {np.mean(PSTD[:,1]):.4f} "
      f"-> posterior/prior width ratio = {np.mean(PSTD[:,1])/np.std(TRUTH[:,1]):.3f}")
out["q_prior_std"] = float(np.std(TRUTH[:, 1]))
out["q_post_over_prior"] = float(np.mean(PSTD[:, 1]) / np.std(TRUTH[:, 1]))
json.dump(out, open(Path(__file__).parent / "e1_result.json", "w"), indent=1)
print("\nwrote e1_result.json")
