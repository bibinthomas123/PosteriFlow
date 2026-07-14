"""
Re-derive the mechanism WITHIN BBH (the event class GW150914 belongs to),
after the audit showed the pooled Mc-dL R^2=0.81 is an event-type-mixing
artifact (within BBH the true Mc-dL R^2 = 0.003).

(1) probes restricted to BBH (reuse saved contexts)
(2) end-to-end: for BBH validation events, does the posterior-median distance
    track TRUE distance / SNR, or chirp mass?
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from ahsd.models.lean_npe import PARAM_NAMES
from remix_data import RemixDataset

OUT = Path(__file__).resolve().parent
d = np.load(OUT / "val_contexts.npz")
ctx, theta, net_snr = d["ctx"], d["theta"], d["net_snr"]
m1, m2 = theta[:, 0], theta[:, 1]
mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
dL = theta[:, C.JD]
bbh = (m1 >= 5) & (m2 >= 5)
print(f"BBH subset: {bbh.sum()}/{len(bbh)}")

kf = KFold(5, shuffle=True, random_state=0)
def r2(X, y):
    X = X if X.ndim == 2 else X[:, None]
    return float(r2_score(y, cross_val_predict(
        HistGradientBoostingRegressor(max_depth=4, max_iter=250, random_state=0), X, y, cv=kf)))
def resid_on_mc(y, lmc):
    return y - cross_val_predict(HistGradientBoostingRegressor(max_depth=4, max_iter=250,
                                 random_state=0), lmc[:, None], y, cv=kf)

cb, lmc, ldl, lsnr = ctx[bbh], np.log(mc[bbh]), np.log(dL[bbh]), np.log(net_snr[bbh])
probes = {
    "mc_to_dL_BBH": r2(lmc, ldl),
    "ctx_to_dL_BBH": r2(cb, ldl),
    "ctx_to_dL_given_mc_BBH": r2(cb, resid_on_mc(ldl, lmc)),
    "ctx_to_SNR_BBH": r2(cb, lsnr),
    "ctx_to_SNR_given_mc_BBH": r2(cb, resid_on_mc(lsnr, lmc)),
    "ctx_to_mc_BBH": r2(cb, lmc),
}
print("\n== within-BBH probes ==")
for k, v in probes.items():
    print(f"   {k:26s} R2={v:+.3f}")

# ── (2) end-to-end: predicted median dL vs true dL / SNR / Mc, BBH only ───────
model = C.load_model()
ds = RemixDataset(str(C.ROOT / "data/dataset/memmap/validation"), remix=False, seed=1234)
Ntot = min(2000, len(ds))
pm, td, ts, tmc = [], [], [], []
for i in range(Ntot):
    it = ds[i]
    p = it[1][0].numpy()
    if not (p[0] >= 5 and p[1] >= 5):
        continue
    with torch.no_grad():
        s = model.sample_posterior(it[0][None].float(), rank=0, n_samples=1500)[0].numpy()
    pm.append(float(np.median(s[:, C.JD]))); td.append(float(p[C.JD]))
    ts.append(float(it[3])); tmc.append(float((p[0] * p[1]) ** 0.6 / (p[0] + p[1]) ** 0.2))
    if len(pm) >= 700:
        break
pm, td, ts, tmc = map(np.array, (pm, td, ts, tmc))
e2e = {
    "n_bbh": len(pm),
    "spearman_predmed_vs_TRUEdL": round(float(spearmanr(pm, td).correlation), 3),
    "spearman_predmed_vs_TRUEsnr": round(float(spearmanr(pm, ts).correlation), 3),
    "spearman_predmed_vs_chirpmass": round(float(spearmanr(pm, tmc).correlation), 3),
    "median_pred_dL": round(float(np.median(pm)), 0),
    "median_true_dL": round(float(np.median(td)), 0),
}
print("\n== end-to-end (BBH val events): what does posterior-median dL track? ==")
print(f"   predicted median dL vs TRUE dL   : spearman {e2e['spearman_predmed_vs_TRUEdL']:+.3f}")
print(f"   predicted median dL vs TRUE SNR  : spearman {e2e['spearman_predmed_vs_TRUEsnr']:+.3f}")
print(f"   predicted median dL vs chirp mass: spearman {e2e['spearman_predmed_vs_chirpmass']:+.3f}")
print(f"   median pred dL={e2e['median_pred_dL']}  median true dL={e2e['median_true_dL']}")

out = {"within_bbh_probes": probes, "end_to_end_bbh": e2e,
       "note": "within BBH: is distance tracked from amplitude(SNR/true dL) or chirp mass?"}
json.dump(out, open(OUT / "within_bbh.json", "w"), indent=2, default=float)
print(f"\nwrote {OUT/'within_bbh.json'}")
