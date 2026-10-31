"""Where is inclination information lost in the encoder?

Probes (held-out R^2) predicting extrinsic parameters from three places in
lean_npe_v3's encoder, on loud validation events in design noise:
  mf    - matched-filter features (per detector x template: |z|, cos/sin phase, t_peak)
  geom  - geometry features (pairwise complex coherence per band, delays)
  ctx   - the 256-D context the flow is conditioned on
Targets: |cos theta_jn|, cos 2psi, sin 2psi, sin(dec); control: log chirp mass.

raw features carry it, ctx does not -> the encoder discards it (route/skip it)
neither carries it                   -> the features are too crude (add better ones)

    python analysis/inclination_info/probe_extrinsic.py [--n 8000] [--min-snr 15]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src", ROOT / "experiments"):
    sys.path.insert(0, str(p))
from ahsd.inference.pipeline import load_model  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "model/lean_npe_v3/best_model.pth"))
ap.add_argument("--data", default=str(ROOT / "data/dataset_v3"))
ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--min-snr", type=float, default=15.0)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(a.model, dev)[0].eval()
enc = model.encoder
ds = RemixDataset(str(Path(a.data) / "memmap" / "validation"), remix=False, seed=1234,
                  return_asd_bands=True, psd_bands=enc.psd_bands)
J = {n: i for i, n in enumerate(PARAM_NAMES)}

feats = {"mf": [], "geom": [], "ctx": []}
T, kept = [], 0
with torch.no_grad():
    for i in range(0, len(ds), 128):
        it = [ds[j] for j in range(i, min(i + 128, len(ds)))]
        snr = np.array([float(v[3]) for v in it])
        keep = np.nonzero(snr >= a.min_snr)[0]
        if len(keep) == 0:
            continue
        x = torch.stack([it[k][0] for k in keep]).float().to(dev)
        ab = torch.stack([it[k][4] for k in keep]).float().to(dev)
        feats["ctx"].append(model.encode(x, ab).cpu().numpy())
        feats["geom"].append(enc._geometry_rel(torch.nan_to_num(x).clamp(-100, 100)).cpu().numpy())
        if getattr(enc, "use_mf_tokens", False):
            feats["mf"].append(enc._mf_features(x).cpu().numpy())
        T.append(np.stack([it[k][1][0].numpy() for k in keep]))
        kept += len(keep)
        if kept >= a.n:
            break
T = np.concatenate(T)
m1, m2 = T[:, J["mass_1"]], T[:, J["mass_2"]]
targets = {
    "|cos theta_jn|": np.abs(np.cos(T[:, J["theta_jn"]])),
    "cos 2psi": np.cos(2 * T[:, J["psi"]]),
    "sin 2psi": np.sin(2 * T[:, J["psi"]]),
    "sin dec": np.sin(T[:, J["dec"]]),
    "log Mc (control)": np.log((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2),
}
X = {k: np.concatenate(v) for k, v in feats.items() if v}
n = len(T); tr = np.arange(n) < int(0.8 * n)
print(f"{n} validation events with network SNR >= {a.min_snr} (design noise); "
      + ", ".join(f"{k} {v.shape[1]}-D" for k, v in X.items()), flush=True)

res = {}
print(f"\n{'target':18s} | " + " | ".join(f"{k:>5s} lin  {k:>5s} mlp" for k in X))
for tname, y in targets.items():
    row = {}
    for fname, F in X.items():
        F = StandardScaler().fit(F[tr]).transform(F)
        lin = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(F[tr], y[tr])
        mlp = MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=300,
                           early_stopping=True, random_state=0).fit(F[tr], y[tr])
        row[fname] = (r2_score(y[~tr], lin.predict(F[~tr])), r2_score(y[~tr], mlp.predict(F[~tr])))
    res[tname] = {k: {"linear_r2": float(v[0]), "mlp_r2": float(v[1])} for k, v in row.items()}
    print(f"{tname:18s} | " + " | ".join(f"{v[0]:9.2f} {v[1]:9.2f}" for v in row.values()), flush=True)

out = Path(__file__).resolve().parent / "probe_extrinsic.json"
json.dump({"model": a.model, "n_events": int(n), "min_snr": a.min_snr,
           "feature_dims": {k: int(v.shape[1]) for k, v in X.items()}, "held_out_r2": res}, open(out, "w"), indent=1)
print(f"\nwrote {out}")
