"""Offline test: do the new cross-detector polarization features carry
inclination / polarization / sky information that the current encoder lacks?

Heavy single-signal BBH from the validation set (short enough to be free of
the pre-2026-10-31 wrap bug; matched by mf_bank_bbh.npz), network SNR >= 15,
design noise (H1, L1, V1). Held-out R^2 of gradient-boosting probes from:
  old      - the encoder's current matched-filter features (mf_bank.npz)
  new      - polarization_features (pairwise amplitude ratio, relative phase, dt)
  old+new  - both
  ceiling  - new + the TRUE sky and psi (how much the ratios can give at best)

    python scripts/polarization_probe.py [--n 12000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "experiments"):
    sys.path.insert(0, str(p))
from ahsd.inference.pipeline import load_model  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402
from ahsd.models.polarization_features import polarization_features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "model/lean_npe_v3/best_model.pth"))
ap.add_argument("--data", default=str(ROOT / "data/dataset_v3"))
ap.add_argument("--n", type=int, default=12000)
ap.add_argument("--min-snr", type=float, default=15.0)
ap.add_argument("--out", default=str(ROOT / "results/polarization_probe.json"))
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
J = {n: i for i, n in enumerate(PARAM_NAMES)}

model = load_model(a.model, dev)[0].eval()
bank = np.load(ROOT / "src/ahsd/models/assets/mf_bank_bbh.npz")
bre, bim = (torch.from_numpy(bank[k]).to(dev) for k in ("bank_re", "bank_im"))
band_lo = int(bank["band_lo"])

c = Path(a.data) / "memmap" / "validation"
sig = np.load(c / "signals.npy", mmap_mode="r"); noi = np.load(c / "noise.npy", mmap_mode="r")
par = np.load(c / "params.npy")
ev = json.load(open(c / "events.json"))["events"]
# heavy single-signal BBH that fit the 4 s window (no wrap), loud enough
sel = []
for e_i, (s, n) in enumerate(ev):
    p = par[s]
    m1, m2, tc = float(p[J["mass_1"]]), float(p[J["mass_2"]]), float(p[J["geocent_time"]])
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    tau = 2.18 * (1.21 / mc) ** (5 / 3) * 5 ** (8 / 3)
    if n == 1 and m2 >= 3 and 1.2 * tau + 0.25 <= 2.0 + tc:
        sel.append((e_i, s))
print(f"{len(sel)} wrap-free single BBH events in validation", flush=True)

OLD, NEW, T = [], [], []
with torch.no_grad():
    for b0 in range(0, len(sel), 128):
        chunk = sel[b0:b0 + 128]
        x = np.stack([np.asarray(noi[e % len(noi)], np.float32) + np.asarray(sig[s], np.float32) for e, s in chunk])
        snr = np.sqrt((np.stack([np.asarray(sig[s], np.float64) for _, s in chunk]) ** 2).sum((1, 2)))
        keep = snr >= a.min_snr
        if not keep.any():
            continue
        xt = torch.from_numpy(x[keep]).to(dev)
        OLD.append(model.encoder._mf_features(xt).cpu().numpy())
        NEW.append(polarization_features(xt, bre, bim, band_lo).cpu().numpy())
        T.append(np.stack([par[s] for (_, s), k in zip(chunk, keep) if k]))
        if sum(len(t) for t in T) >= a.n:
            break
OLD, NEW, T = np.concatenate(OLD), np.concatenate(NEW), np.concatenate(T)
truth_sky = np.stack([np.sin(T[:, J["dec"]]), np.cos(T[:, J["ra"]]), np.sin(T[:, J["ra"]]),
                      np.cos(2 * T[:, J["psi"]]), np.sin(2 * T[:, J["psi"]])], 1)
targets = {"|cos theta_jn|": np.abs(np.cos(T[:, J["theta_jn"]])),
           "cos 2psi": np.cos(2 * T[:, J["psi"]]), "sin 2psi": np.sin(2 * T[:, J["psi"]]),
           "sin dec": np.sin(T[:, J["dec"]])}
sets = {"old": OLD, "new": NEW, "old+new": np.concatenate([OLD, NEW], 1),
        "ceiling": np.concatenate([NEW, truth_sky], 1)}
n = len(T); tr = np.random.default_rng(0).random(n) < 0.8
print(f"{n} events (SNR >= {a.min_snr}); feature dims: " + ", ".join(f"{k} {v.shape[1]}" for k, v in sets.items()))
print(f"\n{'target':16s} | " + " | ".join(f"{k:>8s}" for k in sets))
res = {}
for tname, y in targets.items():
    row = {}
    for sname, F in sets.items():
        if sname == "ceiling" and tname != "|cos theta_jn|":
            continue           # ceiling includes the true psi/sky: only meaningful for inclination
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(F[tr], y[tr])
        row[sname] = float(r2_score(y[~tr], m.predict(F[~tr])))
    res[tname] = row
    print(f"{tname:16s} | " + " | ".join(f"{row.get(k, float('nan')):8.2f}" for k in sets), flush=True)
json.dump({"n_events": int(n), "min_snr": a.min_snr, "held_out_r2": res},
          open(a.out, "w"), indent=1)
