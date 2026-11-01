#!/usr/bin/env python3
"""Is each encoder feature group actually used? Permutation importance.

Each group's tokens are shuffled across the events of a batch (a forward hook
permutes the batch dimension of the module that produces them), so the
network gets another event's features from that group and its own for the
rest. Reported per group, on validation events:
  dNLL     rise in the rank-0 NLL (nats); ~0 = the group is not used
  width    posterior std relative to the unshuffled model, per parameter
  err      |median - truth| relative to the unshuffled model, per parameter
"all input" (the strain itself shuffled) is the scale: everything lost.

    python scripts/feature_importance.py --model model/lean_npe_bbh_v4/best_model.pth \
        --data data/dataset_bbh_v4 [--n 1024] [--real]
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

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=1024)
ap.add_argument("--real", action="store_true")
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
model, _ = load_model(a.model, device=dev)
enc = model.encoder

ds = RemixDataset(f"{a.data}/memmap/validation", remix=False, seed=3, return_asd_bands=True,
                  real_noise_dir="data/noise_bank" if a.real else None,
                  real_noise_prob=1.0 if a.real else 0.0)
X, P, AB = [], [], []
for i in range(len(ds)):
    x, p, n, snr, ab = ds[i]
    if int(n) != 1:
        continue
    X.append(x); P.append(p[0]); AB.append(ab)
    if len(X) >= a.n:
        break
X = torch.stack(X).float(); P = torch.stack(P).float(); AB = torch.stack(AB).float()

# feature groups: name -> module whose output (batch-first) is permuted
groups = {"geometry": getattr(enc, "geom_to_tokens", None),
          "mf_tokens": getattr(enc, "mf_to_tokens", None),
          "polarization": getattr(enc, "pol_to_tokens", None)}
if getattr(enc, "mc_bank", ""):
    groups["chirp_mass_bank"] = enc.mc_feat.to_tokens
    if getattr(enc.mc_feat, "trigger", False):
        groups["timing"] = enc.mc_feat.trig_to_tokens
groups = {k: v for k, v in groups.items() if v is not None}
lowrate = getattr(model, "lowrate_len", 0) > 0

PERM = {"idx": None}


def hook(_m, _inp, out):
    return out[PERM["idx"][: out.shape[0]]] if PERM["idx"] is not None else out


def evaluate(shuffle=None, B=64):
    nll, std, err = [], [], []
    for i in range(0, len(X), B):
        x = X[i:i + B].to(dev); p = P[i:i + B].to(dev)
        ab = AB[i:i + B].to(dev) if model.psd_cond else None
        PERM["idx"] = torch.randperm(x.shape[0], device=dev) if shuffle else None
        if shuffle == "all input":
            x = x[PERM["idx"]]
        if shuffle == "context stream":
            x = torch.cat([x[..., :16384], x[PERM["idx"]][..., 16384:]], -1)
        h = groups[shuffle].register_forward_hook(hook) if shuffle in groups else None
        with torch.no_grad():
            ctx = model.encode(x, ab)
            r0 = torch.zeros(x.shape[0], dtype=torch.long, device=dev)
            nll.append(model.nll(None, p, r0, context=ctx).cpu())
            s = model.sample_posterior(x, n_samples=200, asd_bands=ab)
        if h:
            h.remove()
        m1, m2 = s[..., 0], s[..., 1]
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        mct = (p[:, 0] * p[:, 1]) ** 0.6 / (p[:, 0] + p[:, 1]) ** 0.2
        cols = {n: s[..., j] for j, n in enumerate(PARAM_NAMES)}
        cols["chirp_mass"] = mc
        tr = {n: p[:, j] for j, n in enumerate(PARAM_NAMES)}
        tr["chirp_mass"] = mct
        std.append({k: v.std(1).cpu() for k, v in cols.items()})
        err.append({k: (v.median(1).values - tr[k]).abs().cpu() for k, v in cols.items()})
    cat = lambda L: {k: torch.cat([d[k] for d in L]) for k in L[0]}
    return torch.cat(nll).mean().item(), cat(std), cat(err)


KEY = ["chirp_mass", "mass_2", "luminosity_distance", "ra", "dec",
       "theta_jn", "geocent_time"]
base_nll, base_std, base_err = evaluate()
rows = {"baseline_nll": round(base_nll, 3)}
print(f"model {a.model} | noise {'real' if a.real else 'gaussian'} | n={len(X)} | baseline NLL {base_nll:.3f}")
print(f"{'shuffled':18s}{'dNLL':>8s}  " + "  ".join(f"{k[:8]:>15s}" for k in KEY))
print(f"{'':18s}{'':>8s}  " + "  ".join(f"{'width/err':>15s}" for _ in KEY))
names = list(groups) + (["context stream"] if lowrate else []) + ["all input"]
for g in names:
    nl, sd, er = evaluate(g)
    w = {k: float((sd[k] / base_std[k].clamp_min(1e-12)).median()) for k in KEY}
    e = {k: float(er[k].median() / base_err[k].median().clamp_min(1e-12)) for k in KEY}
    rows[g] = {"dnll": round(nl - base_nll, 3), "width_ratio": w, "err_ratio": e}
    print(f"{g:18s}{nl - base_nll:>+8.2f}  " + "  ".join(f"{w[k]:>7.2f}/{e[k]:<7.2f}" for k in KEY), flush=True)
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
