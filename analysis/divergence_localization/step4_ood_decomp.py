"""STEP 4 - OOD decomposition by encoder feature group.

The reported OOD score is a single Mahalanobis on the pooled 256-D context. We
break the encoder representation into its constituent feature groups and score
each group's out-of-distribution-ness for the 6 real events against a validation
reference, to find WHICH group drives the OOD.

Groups (from CoherentEncoder):
  energy       : per-detector per-band log-energy      rel[0:48]
  coherence    : power-weighted |gamma|, cos/sin phase  rel coherence sub-blocks (144)
  arrival_time : GCC tau + peak sharpness per pair       rel tau/peak (6)
  ampratio     : per-pair log-amplitude ratio            rel (3)
  geom_embed   : geometry MLP embedding (injected tokens) g = geom_mlp(rel) (128)
  psd          : asd_bands sensitivity summary           (3x16)
  context_full : final 256-D context (== reported OOD)   out_proj

Per-group OOD = RMS z-score vs validation mean/std, and percentile vs the
held-out validation distribution of that same RMS-z score.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from _events import EVENTS, PROBLEM  # noqa: E402
from ahsd.inference.pipeline import load_model  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

MODEL = "model/long_run/best_model.pth"
VCACHE = "data/dataset/memmap/validation"
NBANK = "data/noise_bank"
N_REF = 2000
K, NPAIR, NDET = 16, 3, 3


def group_indices():
    """Index sets into the rel vector for each geometry sub-group."""
    idx = {"energy": list(range(0, NDET * K))}
    coh, arr, amp = [], [], []
    base = NDET * K
    for p in range(NPAIR):
        s = base + p * (K * 3 + 3)
        coh += list(range(s, s + K * 3))       # |g|, cos, sin
        arr += [s + K * 3, s + K * 3 + 1]      # tau, peak
        amp += [s + K * 3 + 2]                 # log-amp ratio
    idx["coherence"] = coh; idx["arrival_time"] = arr; idx["ampratio"] = amp
    return idx


def encode_groups(net, dev, strain, asd_bands):
    """Return dict group -> feature vector [B, d] for a batch of strain."""
    enc = net.encoder
    with torch.no_grad():
        strain = torch.nan_to_num(strain, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100, 100)
        rel = enc._geometry_rel(strain)                 # [B, rel_dim]
        g = enc.geom_mlp(rel)                            # [B, 128]
        ctx = net.encode(strain, asd_bands)             # [B, 256]
    rel = rel.cpu().numpy(); g = g.cpu().numpy(); ctx = ctx.cpu().numpy()
    gi = group_indices()
    out = {k: rel[:, gi[k]] for k in gi}
    out["geom_embed"] = g
    out["context_full"] = ctx
    if asd_bands is not None:
        out["psd"] = asd_bands.reshape(asd_bands.shape[0], -1).cpu().numpy()
    return out


def main():
    net, meta = load_model(MODEL)
    dev = meta["device"]
    print(f"model epoch {meta['model_epoch']} val_nll {meta['model_val_nll']:.4f}\n")

    # ---- build validation reference (mix Gaussian + real) ----
    ref = {}
    for domain, kwargs in [("gaussian", {}),
                           ("real", dict(real_noise_dir=NBANK, real_noise_prob=1.0))]:
        ds = RemixDataset(VCACHE, remix=False, seed=1234,
                          return_asd_bands=net.psd_cond, **kwargs)
        n = min(N_REF, len(ds))
        B = 128
        for start in range(0, n, B):
            idxs = range(start, min(start + B, n))
            strains, abs_ = [], []
            for i in idxs:
                item = ds[i]
                if net.psd_cond:
                    st, pv, nsig, ns, ab = item; abs_.append(ab.numpy())
                else:
                    st, pv, nsig, ns = item
                strains.append(st.numpy())
            st = torch.from_numpy(np.stack(strains)).to(dev).float()
            ab = torch.from_numpy(np.stack(abs_)).to(dev).float() if net.psd_cond else None
            grp = encode_groups(net, dev, st, ab)
            for k, v in grp.items():
                ref.setdefault(k, []).append(v)
    ref = {k: np.concatenate(v, 0) for k, v in ref.items()}
    print("reference sizes:", {k: v.shape for k, v in ref.items()})

    # per-group mean/std + reference RMS-z distribution (held-out split)
    stats = {}
    for k, v in ref.items():
        mu = v.mean(0); sd = v.std(0) + 1e-8
        z = (v - mu) / sd
        rmsz = np.sqrt((z ** 2).mean(1))         # per-event RMS z over dims
        stats[k] = dict(mu=mu, sd=sd, ref_rmsz=rmsz)

    # ---- score the 6 real events ----
    rows = {}
    for name, ev in EVENTS.items():
        d = HERE / "events_ep19" / name
        strain = torch.from_numpy(np.load(d / "strain.npy")).unsqueeze(0).to(dev).float()
        ab = None
        if net.psd_cond:
            ab = torch.from_numpy(np.load(d / "asd_bands.npy")).unsqueeze(0).to(dev).float()
        grp = encode_groups(net, dev, strain, ab)
        rows[name] = {}
        for k, v in grp.items():
            z = (v[0] - stats[k]["mu"]) / stats[k]["sd"]
            rmsz = float(np.sqrt((z ** 2).mean()))
            pct = float((stats[k]["ref_rmsz"] < rmsz).mean() * 100)
            rows[name][k] = {"rmsz": round(rmsz, 2), "pct": round(pct, 2)}

    groups = ["energy", "coherence", "arrival_time", "ampratio", "geom_embed",
              "psd", "context_full"]
    groups = [g for g in groups if g in rows["GW150914"]]

    print("\n=== per-group OOD percentile vs validation (RMS-z) ===\n")
    print(f"{'event':10s} " + " ".join(f"{g[:9]:>10}" for g in groups))
    for name in EVENTS:
        cells = " ".join(f"{rows[name][g]['pct']:10.1f}" for g in groups)
        flag = " <==" if name in PROBLEM else ""
        print(f"{name:10s} {cells}{flag}")
    print("\n=== per-group RMS z-score (magnitude) ===\n")
    print(f"{'event':10s} " + " ".join(f"{g[:9]:>10}" for g in groups))
    for name in EVENTS:
        cells = " ".join(f"{rows[name][g]['rmsz']:10.2f}" for g in groups)
        flag = " <==" if name in PROBLEM else ""
        print(f"{name:10s} {cells}{flag}")

    json.dump({"events": rows,
               "ref_rmsz_p99": {k: float(np.percentile(stats[k]["ref_rmsz"], 99))
                                for k in stats}},
              open(HERE / "step4_ood_decomp.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
