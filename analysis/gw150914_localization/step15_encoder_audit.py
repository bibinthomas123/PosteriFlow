"""
STEP 1 + STEP 5 - encoder signal-strength / representation audit.

Central question: is the encoder context's distance information genuine
(amplitude-based) or merely chirp-mass-mediated (a learned P(dL|Mc) shortcut)?

Tests on the DESIGN validation split (asd_bands = 0):
  A. context -> log dL           R^2   (total)
  B. log Mc  -> log dL           R^2   (prior-correlation baseline)
  C. context -> (log dL | Mc)    R^2   (distance info BEYOND chirp mass)   <-- key
  D. context -> log net_SNR      R^2   (does the encoder encode loudness?)
  E. context -> (log SNR | Mc)   R^2   (amplitude info BEYOND chirp mass)  <-- key

Plus GW150914 vs matched injections:
  - context->SNR probe applied to GW150914 / matched design inj / matched O1 inj
  - context cosine similarity & Euclidean distance
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from remix_data import RemixDataset

OUT = Path(__file__).resolve().parent
model = C.load_model()
for p in model.parameters():
    p.requires_grad_(False)

# ── extract validation contexts + truths + SNR ───────────────────────────────
ds = RemixDataset(str(C.ROOT / "data/dataset/memmap/validation"), remix=False, seed=1234)
N = min(2000, len(ds))
strain_list, theta_list, snr_list = [], [], []
for i in range(N):
    it = ds[i]
    strain_list.append(it[0]); theta_list.append(it[1][0]); snr_list.append(it[3])
strain = torch.stack(strain_list)
theta = torch.stack(theta_list).numpy()
net_snr = torch.stack(snr_list).numpy().astype(float)
m1, m2, dL = theta[:, 0], theta[:, 1], theta[:, C.JD]
mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
with torch.no_grad():
    ctx = torch.cat([model.encoder(strain[i:i + 64]) for i in range(0, N, 64)]).numpy()
print(f"N={N} contexts {ctx.shape}  SNR range [{net_snr.min():.0f},{net_snr.max():.0f}]")

log_dL, log_mc, log_snr = np.log(dL), np.log(mc), np.log(net_snr)
kf = KFold(5, shuffle=True, random_state=0)
def gbm_r2(X, y):
    X = X if X.ndim == 2 else X[:, None]
    return float(r2_score(y, cross_val_predict(
        HistGradientBoostingRegressor(max_depth=4, max_iter=250, random_state=0), X, y, cv=kf)))
def lin_r2(X, y):
    X = X if X.ndim == 2 else X[:, None]
    return float(r2_score(y, cross_val_predict(LinearRegression(), X, y, cv=kf)))

# residualize a target on log Mc (remove the chirp-mass-predictable part)
def resid_on_mc(y):
    pred = cross_val_predict(HistGradientBoostingRegressor(max_depth=4, max_iter=250,
                             random_state=0), log_mc[:, None], y, cv=kf)
    return y - pred

R = {}
R["A_ctx_to_logdL_r2"]        = gbm_r2(ctx, log_dL)
R["B_mc_to_logdL_r2"]         = gbm_r2(log_mc, log_dL)
R["C_ctx_to_logdL_given_mc"]  = gbm_r2(ctx, resid_on_mc(log_dL))
R["D_ctx_to_logSNR_r2"]       = gbm_r2(ctx, log_snr)
R["E_ctx_to_logSNR_given_mc"] = gbm_r2(ctx, resid_on_mc(log_snr))
R["ctx_to_logMc_r2"]          = gbm_r2(ctx, log_mc)
# how much of true dL variance is Mc-predictable at all
R["frac_logdL_var_from_mc"]   = R["B_mc_to_logdL_r2"]
print("\n== encoder information content ==")
for k in ["A_ctx_to_logdL_r2", "B_mc_to_logdL_r2", "C_ctx_to_logdL_given_mc",
          "D_ctx_to_logSNR_r2", "E_ctx_to_logSNR_given_mc", "ctx_to_logMc_r2"]:
    print(f"  {k:32s} = {R[k]:+.3f}")

# ── GW150914 + matched injections: context probe + geometry ──────────────────
# train a context->log_SNR probe on validation, apply to the real/matched events
snr_probe = HistGradientBoostingRegressor(max_depth=4, max_iter=250, random_state=0).fit(ctx, log_snr)
dL_probe = HistGradientBoostingRegressor(max_depth=4, max_iter=250, random_state=0).fit(ctx, log_dL)

def ctx_of_prep(prep):
    return C.encode(model, prep)

P = dict(C.GW150914_TRUTH)
prep_real = C.prepare_gw150914()
ctx_real = ctx_of_prep(prep_real)
mf_real = C.network_mf_snr(prep_real, P)["network"]["mf_snr"]

# matched design injection (asd_bands=0) at GW150914 params, tuned so SNR~24:
from ahsd.data.bilby_pipeline import (BilbyNoiseGenerator, BilbySignalInjector,
                                      BilbyPreprocessor, get_default_psd)
prep_bp = BilbyPreprocessor(C.SR, C.DUR); ng = BilbyNoiseGenerator(C.SR, C.DUR)
inj = BilbySignalInjector(C.SR, C.DUR); psds = {d: get_default_psd(d) for d in ["H1", "L1", "V1"]}
# design SNR~24 for GW150914 masses needs dL~1800; also make a dL=440 (SNR~98) one
matched = {}
for tag, dLv in [("design_dL440_snr98", 440.0), ("design_dL1800_snr24", 1800.0)]:
    Pd = dict(P); Pd["luminosity_distance"] = dLv
    white = {}
    for det in ["H1", "L1"]:
        noise = ng.generate(det, psds[det], seed=7)
        st, _ = inj.inject(noise, Pd, det, psds[det])
        white[det] = prep_bp.preprocess(st, psds[det], detector=det)
    rng = np.random.default_rng(0)
    s3 = np.stack([white["H1"], white["L1"], rng.standard_normal(C.T_LEN).astype(np.float32)])
    with torch.no_grad():
        cvec = model.encoder(torch.from_numpy(s3[None]).float(),
                             torch.zeros(1, 3, 16)).numpy()[0]
    matched[tag] = cvec

# matched real-O1 injection (dL=440, SNR~24) from step4 machinery
prep_o1, snr_o1 = C.inject_into_real_noise(C.GW150914_GPS - 300, P)
ctx_o1 = ctx_of_prep(prep_o1)

def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def euc(a, b): return float(np.linalg.norm(a - b))

# intra-population reference distances (random val pairs)
ri = np.random.default_rng(0).integers(0, N, (500, 2))
pop_cos = np.median([cos(ctx[a], ctx[b]) for a, b in ri])
pop_euc = np.median([euc(ctx[a], ctx[b]) for a, b in ri])

events = {
    "REAL_GW150914":        dict(ctx=ctx_real, mf_snr=mf_real, true_dL=440),
    "matched_O1_dL440":     dict(ctx=ctx_o1, mf_snr=snr_o1["network"]["mf_snr"], true_dL=440),
    "matched_design_dL1800": dict(ctx=matched["design_dL1800_snr24"], mf_snr=24.0, true_dL=1800),
    "matched_design_dL440":  dict(ctx=matched["design_dL440_snr98"], mf_snr=98.0, true_dL=440),
}
probe_tbl = {}
for name, e in events.items():
    cvec = e["ctx"]
    probe_tbl[name] = dict(
        true_dL=e["true_dL"], true_mf_snr=round(float(e["mf_snr"]), 1),
        probe_pred_SNR=round(float(np.exp(snr_probe.predict(cvec[None])[0])), 1),
        probe_pred_dL=round(float(np.exp(dL_probe.predict(cvec[None])[0])), 0),
        cos_to_GW150914=round(cos(cvec, ctx_real), 3),
        euc_to_GW150914=round(euc(cvec, ctx_real), 2))
R["probe_and_geometry"] = probe_tbl
R["population_ref"] = dict(median_cos=round(pop_cos, 3), median_euc=round(pop_euc, 2))

json.dump(R, open(OUT / "step15_encoder_audit.json", "w"), indent=2, default=float)

print("\n===== STEP 1+5 : ENCODER AUDIT =====")
print(f"context->dL R2={R['A_ctx_to_logdL_r2']:+.2f} | Mc->dL R2={R['B_mc_to_logdL_r2']:+.2f} "
      f"| context->dL GIVEN Mc R2={R['C_ctx_to_logdL_given_mc']:+.2f}")
print(f"context->SNR R2={R['D_ctx_to_logSNR_r2']:+.2f} | context->SNR GIVEN Mc R2={R['E_ctx_to_logSNR_given_mc']:+.2f}")
print(f"\n{'event':24s} {'trueDL':>7s} {'trueSNR':>8s} {'probeSNR':>9s} {'probeDL':>8s} {'cos->GW':>8s} {'euc->GW':>8s}")
for name, t in probe_tbl.items():
    print(f"{name:24s} {t['true_dL']:7.0f} {t['true_mf_snr']:8.1f} {t['probe_pred_SNR']:9.1f} "
          f"{t['probe_pred_dL']:8.0f} {t['cos_to_GW150914']:8.3f} {t['euc_to_GW150914']:8.2f}")
print(f"\npopulation median cos={pop_cos:.3f} euc={pop_euc:.2f}")
print(f"wrote {OUT/'step15_encoder_audit.json'}")
