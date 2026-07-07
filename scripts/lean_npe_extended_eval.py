#!/usr/bin/env python3
"""
Extended LeanNPE evaluation on the fixed validation split:
  - distance/chirp-mass error vs network SNR (binned)
  - calibration (cov50/cov90) split by SNR band
  - performance & calibration vs overlap severity (n_signals, min |dt_c|)
Saves per-event summary CSV + JSON + plots.
"""
import argparse, glob, json, pickle, sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES

BLUE, GRAY, INK = "#3b6fd4", "#9aa4b2", "#3a3f47"
JD = PARAM_NAMES.index("luminosity_distance")
JT = PARAM_NAMES.index("geocent_time")


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def load_val(data_dir, n):
    evs = []
    for fp in sorted(glob.glob(f"{data_dir}/validation/batch_*.pkl")):
        with open(fp, "rb") as fh:
            b = pickle.load(fh)
        for s in b["samples"]:
            if str(s.get("event_type")) == "noise":
                continue
            plist = s["parameters"]
            if any(plist[0].get(k) is None for k in PARAM_NAMES):
                continue
            if s["metadata"].get("is_premerger"):
                continue  # premerger eval is separate (model v2 wasn't trained on it)
            plist = sorted(plist, key=lambda p: chirp_mass(p["mass_1"], p["mass_2"]) ** (5 / 6)
                           / max(p["luminosity_distance"], 1.0), reverse=True)
            tcs = [p["geocent_time"] for p in plist]
            min_dt = min(abs(tcs[0] - t) for t in tcs[1:]) if len(tcs) > 1 else np.nan
            evs.append({
                "strain": np.stack([np.asarray(s["detector_data"][d]["strain"], dtype=np.float32)
                                    for d in ("H1", "L1", "V1")]),
                "theta": np.array([plist[0][k] for k in PARAM_NAMES], dtype=np.float32),
                "snr": float(s.get("network_snr", 0.0)),
                "n_signals": int(s.get("n_signals", 1)),
                "min_dt": float(min_dt),
            })
            if len(evs) >= n:
                return evs
    return evs


def summarize(mask, truth, samples, label):
    """coverage + median errors for a subset of events"""
    if mask.sum() < 20:
        return None
    t, s = truth[mask], samples[mask]
    out = {"label": label, "n": int(mask.sum())}
    for j, tag in ((JD, "dist"),):
        lo50, hi50 = np.quantile(s[:, :, j], [0.25, 0.75], axis=1)
        lo90, hi90 = np.quantile(s[:, :, j], [0.05, 0.95], axis=1)
        med = np.median(s[:, :, j], axis=1)
        out[f"{tag}_cov50"] = float(((t[:, j] >= lo50) & (t[:, j] <= hi50)).mean())
        out[f"{tag}_cov90"] = float(((t[:, j] >= lo90) & (t[:, j] <= hi90)).mean())
        out[f"{tag}_medfrac"] = float(np.median(np.abs(med - t[:, j]) / t[:, j]))
    mc_t = chirp_mass(t[:, 0], t[:, 1])
    mc_s = chirp_mass(s[:, :, 0], s[:, :, 1])
    out["mc_medfrac"] = float(np.median(np.abs(np.median(mc_s, axis=1) - mc_t) / mc_t))
    # all-param mean coverage (quick scalar)
    cov90 = []
    for j in range(len(PARAM_NAMES)):
        lo, hi = np.quantile(s[:, :, j], [0.05, 0.95], axis=1)
        cov90.append(((t[:, j] >= lo) & (t[:, j] <= hi)).mean())
    out["cov90_all_mean"] = float(np.mean(cov90))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/lean_npe_v2/best_model.pth")
    ap.add_argument("--data", default="data/dataset_v2")
    ap.add_argument("--outdir", default="analysis/lean_npe_v2_extended")
    ap.add_argument("--n_events", type=int, default=2000)
    ap.add_argument("--n_post", type=int, default=400)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = LeanNPE(premerger=ckpt["args"].get("premerger", False))
    model.load_state_dict(ckpt["model_state_dict"]); model.to(device).eval()

    evs = load_val(args.data, args.n_events)
    N, P, K = len(evs), len(PARAM_NAMES), args.n_post
    print(f"events={N} (n_signals dist: {np.bincount([e['n_signals'] for e in evs])})")
    strain = torch.from_numpy(np.stack([e["strain"] for e in evs]))
    truth = np.stack([e["theta"] for e in evs])
    snr = np.array([e["snr"] for e in evs])
    nsig = np.array([e["n_signals"] for e in evs])
    min_dt = np.array([e["min_dt"] for e in evs])

    samples = np.empty((N, K, P), dtype=np.float32)
    with torch.no_grad():
        ctx_all = []
        for i in range(0, N, 64):
            ctx_all.append(model.encoder(strain[i:i + 64].to(device)))
        ctx_all = torch.cat(ctx_all)
        rank0 = torch.zeros(N, dtype=torch.long, device=device)
        fc = model._full_context(ctx_all, rank0)
        B = 48
        for i in range(0, N, B):
            f = fc[i:i + B]; nb = f.shape[0]
            fr = f.unsqueeze(1).expand(nb, K, f.shape[1]).reshape(nb * K, -1)
            z = torch.randn(nb * K, P, device=device)
            y, _ = model.flow.inverse(z, fr)
            samples[i:i + B] = model.scaler.denormalize(y).reshape(nb, K, P).cpu().numpy()

    results = {"checkpoint_epoch": ckpt["epoch"], "n_events": N, "n_post": K}

    # --- SNR bins ---
    snr_edges = [6, 9, 12, 16, 22, 32, 120]
    snr_rows = []
    for a, b in zip(snr_edges[:-1], snr_edges[1:]):
        r = summarize((snr >= a) & (snr < b), truth, samples, f"SNR {a}-{b}")
        if r: snr_rows.append(r)
    results["by_snr"] = snr_rows

    # --- calibration low vs high SNR ---
    results["calib_low_snr(<12)"] = summarize(snr < 12, truth, samples, "SNR<12")
    results["calib_high_snr(>=20)"] = summarize(snr >= 20, truth, samples, "SNR>=20")

    # --- overlap severity ---
    ov_rows = [summarize(nsig == 1, truth, samples, "single")]
    ov_rows.append(summarize(nsig == 2, truth, samples, "2 signals"))
    ov_rows.append(summarize(nsig >= 3, truth, samples, "3+ signals"))
    for a, b, lab in [(0.0, 0.3, "|dt|<0.3s"), (0.3, 1.0, "0.3-1.0s"), (1.0, 99, ">1.0s")]:
        m = (nsig > 1) & (min_dt >= a) & (min_dt < b)
        ov_rows.append(summarize(m, truth, samples, f"overlap {lab}"))
    results["by_overlap"] = [r for r in ov_rows if r]

    with open(out / "extended_report.json", "w") as f:
        json.dump(results, f, indent=2)

    # per-event CSV for future analyses
    med = np.median(samples, axis=1)
    np.savetxt(out / "per_event.csv",
               np.column_stack([snr, nsig, min_dt, truth[:, JD], med[:, JD]]),
               delimiter=",", header="snr,n_signals,min_dt,true_dist,post_med_dist", comments="")

    # --- plots ---
    plt.rcParams.update({"axes.grid": True, "grid.color": "#e8ebef", "font.size": 9,
                         "axes.edgecolor": GRAY})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    labs = [r["label"] for r in snr_rows]
    axes[0].plot(labs, [r["dist_medfrac"] for r in snr_rows], "o-", color=BLUE)
    axes[0].set_ylabel("median |frac dist error|"); axes[0].set_yscale("log")
    axes[1].plot(labs, [r["dist_cov50"] for r in snr_rows], "o-", color=BLUE, label="cov50")
    axes[1].plot(labs, [r["dist_cov90"] for r in snr_rows], "s-", color=INK, label="cov90")
    axes[1].axhline(0.5, color=BLUE, ls="--", lw=0.8); axes[1].axhline(0.9, color=INK, ls="--", lw=0.8)
    axes[1].set_ylim(0, 1); axes[1].legend(frameon=False); axes[1].set_ylabel("distance coverage")
    ol = results["by_overlap"]
    axes[2].plot([r["label"] for r in ol], [r["dist_medfrac"] for r in ol], "o-", color=BLUE)
    axes[2].set_ylabel("median |frac dist error|")
    for a in axes: a.tick_params(axis="x", rotation=30)
    fig.suptitle(f"LeanNPE v2 extended eval ({N} events)", color=INK)
    fig.tight_layout(); fig.savefig(out / "extended_eval.png", dpi=150); plt.close(fig)

    print(json.dumps(results, indent=2)[:3000])
    print("done ->", out)


if __name__ == "__main__":
    main()
