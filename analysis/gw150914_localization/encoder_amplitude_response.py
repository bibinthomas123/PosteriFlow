"""
How does the encoder CONTEXT respond when ONLY amplitude (distance) changes,
for a fixed heavy BBH (GW150914 intrinsics + extrinsics)?

Fix all 11 params except luminosity_distance; sweep dL (amplitude ~ 1/dL) in
DESIGN noise (asd_bands=0). For each dL, encode M noise realizations.

Decisive question: does the context move with amplitude (encoder preserves it)
while the flow's distance output stays flat (Step 4b)? -> loss is in decoding,
not the encoder.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from ahsd.data.bilby_pipeline import (BilbyNoiseGenerator, BilbySignalInjector,
                                      BilbyPreprocessor, get_default_psd)

OUT = Path(__file__).resolve().parent
model = C.load_model()
for p in model.parameters():
    p.requires_grad_(False)
prep_bp = BilbyPreprocessor(C.SR, C.DUR)
ng = BilbyNoiseGenerator(C.SR, C.DUR)
inj = BilbySignalInjector(C.SR, C.DUR)
psds = {d: get_default_psd(d) for d in ["H1", "L1", "V1"]}
zero_ab = torch.zeros(1, 3, 16)

P0 = dict(C.GW150914_TRUTH)                    # fixed heavy BBH, fixed geometry
dL_grid = [250, 350, 440, 600, 850, 1200, 1700, 2100]
M = 24                                          # noise realizations per dL

ctx_all, dL_all, snr_all = [], [], []
ctx_mean, ctx_signalonly, snr_mean, flow_dLmed = {}, {}, {}, {}
within_scatter = {}
for dL in dL_grid:
    P = dict(P0); P["luminosity_distance"] = float(dL)
    ctxs, snrs, dmeds = [], [], []
    for m in range(M):
        white, ss = {}, []
        for det in ["H1", "L1"]:
            noise = ng.generate(det, psds[det], seed=20000 + m * 31 + dL)
            strain, s = inj.inject(noise, P, det, psds[det])
            white[det] = prep_bp.preprocess(strain, psds[det], detector=det)
            ss.append(s)
        rng = np.random.default_rng(m)
        s3 = np.stack([white["H1"], white["L1"], rng.standard_normal(C.T_LEN).astype(np.float32)])
        with torch.no_grad():
            cvec = model.encoder(torch.from_numpy(s3[None]).float(), zero_ab).numpy()[0]
            samp = model.sample_posterior(torch.from_numpy(s3[None]).float(), rank=0,
                                          n_samples=1200, asd_bands=zero_ab)[0].numpy()
        ctxs.append(cvec); snrs.append(float(np.sqrt(sum(x ** 2 for x in ss))))
        dmeds.append(float(np.median(samp[:, C.JD])))
        ctx_all.append(cvec); dL_all.append(dL); snr_all.append(snrs[-1])
    ctxs = np.stack(ctxs)
    ctx_mean[dL] = ctxs.mean(0)
    within_scatter[dL] = float(np.mean(np.linalg.norm(ctxs - ctxs.mean(0), axis=1)))
    snr_mean[dL] = float(np.mean(snrs)); flow_dLmed[dL] = float(np.median(dmeds))
    # signal-only (no noise) clean amplitude response
    white = {}
    for det in ["H1", "L1"]:
        strain, _ = inj.inject(np.zeros(C.T_LEN), P, det, psds[det])
        white[det] = prep_bp.preprocess(strain, psds[det], detector=det)
    s3 = np.stack([white["H1"], white["L1"], np.zeros(C.T_LEN, np.float32)])
    with torch.no_grad():
        ctx_signalonly[dL] = model.encoder(torch.from_numpy(s3[None]).float(), zero_ab).numpy()[0]
    print(f"dL={dL:5d} SNR={snr_mean[dL]:5.1f}  flow dL_med={flow_dLmed[dL]:6.0f}  "
          f"within-noise ctx scatter={within_scatter[dL]:.2f}")

ctx_all = np.stack(ctx_all); dL_all = np.array(dL_all, float); snr_all = np.array(snr_all)

# ── metrics ──────────────────────────────────────────────────────────────────
means = np.stack([ctx_mean[d] for d in dL_grid])
amp_move = float(np.linalg.norm(ctx_mean[dL_grid[0]] - ctx_mean[dL_grid[-1]]))
typ_scatter = float(np.mean(list(within_scatter.values())))
# amplitude direction = mean(dL_min) - mean(dL_max), project mean-contexts
u = ctx_mean[dL_grid[0]] - ctx_mean[dL_grid[-1]]; u = u / (np.linalg.norm(u) + 1e-9)
proj_mean = means @ u
from scipy.stats import pearsonr, spearmanr
# eta^2: between-dL vs total variance in context (per-realization)
grand = ctx_all.mean(0)
ss_between = sum(M * np.sum((ctx_mean[d] - grand) ** 2) for d in dL_grid)
ss_total = np.sum((ctx_all - grand) ** 2)
eta2 = float(ss_between / ss_total)
# linear readout of amplitude from context (cross-validated, on the sweep)
kf = KFold(5, shuffle=True, random_state=0)
pred = cross_val_predict(Ridge(alpha=1.0), ctx_all, np.log(dL_all), cv=kf)
ridge_r2 = float(r2_score(np.log(dL_all), pred))

metrics = {
    "dL_grid": dL_grid, "M": M,
    "snr_by_dL": {d: round(snr_mean[d], 1) for d in dL_grid},
    "flow_dLmed_by_dL": {d: round(flow_dLmed[d], 0) for d in dL_grid},
    "encoder_amplitude_context_movement": round(amp_move, 2),
    "typical_within_noise_scatter": round(typ_scatter, 2),
    "movement_to_noise_ratio": round(amp_move / typ_scatter, 2),
    "eta2_context_variance_from_dL": round(eta2, 3),
    "proj_on_amp_dir_vs_logdL_pearson": round(float(pearsonr(proj_mean, np.log(dL_grid))[0]), 3),
    "context_to_logdL_ridge_R2_on_sweep": round(ridge_r2, 3),
    "flow_dLmed_vs_truedL_spearman": round(float(spearmanr(list(flow_dLmed.values()),
                                                           dL_grid).correlation), 3),
}
json.dump(metrics, open(OUT / "encoder_amplitude_response.json", "w"), indent=2, default=float)

print("\n===== ENCODER AMPLITUDE RESPONSE (fixed heavy BBH, vary dL only) =====")
print(f"amplitude-induced context movement (dL {dL_grid[0]}->{dL_grid[-1]}): {amp_move:.2f}")
print(f"typical within-noise context scatter                     : {typ_scatter:.2f}")
print(f"  movement / noise-scatter ratio                         : {metrics['movement_to_noise_ratio']}")
print(f"eta^2 (context variance explained by dL)                 : {eta2:.3f}")
print(f"proj on amp-direction vs log dL, Pearson                 : {metrics['proj_on_amp_dir_vs_logdL_pearson']}")
print(f"context->log dL ridge R2 (on sweep, CV)                  : {ridge_r2:.3f}")
print(f"FLOW output dL_med vs true dL, Spearman                  : {metrics['flow_dLmed_vs_truedL_spearman']}")
print(f"  (flow dL_med: {[int(flow_dLmed[d]) for d in dL_grid]} for true {dL_grid})")

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].plot(dL_grid, proj_mean, "-o", color="#0072B2")
ax[0].set_xlabel("true dL [Mpc] (fixed heavy BBH)"); ax[0].set_ylabel("context proj. on amplitude axis")
ax[0].set_title(f"Encoder context MOVES with amplitude\n(η²={eta2:.2f}, ctx→dL R²={ridge_r2:.2f}, move/noise={metrics['movement_to_noise_ratio']})")
ax[0].grid(alpha=0.2)
ax[1].plot(dL_grid, [flow_dLmed[d] for d in dL_grid], "-o", color="#D55E00", label="flow posterior median")
ax[1].plot(dL_grid, dL_grid, "k--", alpha=0.5, label="ideal")
ax[1].set_xlabel("true dL [Mpc]"); ax[1].set_ylabel("flow posterior-median dL")
ax[1].set_title(f"…but the FLOW output stays flat\n(Spearman={metrics['flow_dLmed_vs_truedL_spearman']})")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.2)
fig.tight_layout(); fig.savefig(OUT / "encoder_amplitude_response.png", dpi=120)
print(f"\nwrote encoder_amplitude_response.json + .png")
