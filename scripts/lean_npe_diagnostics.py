#!/usr/bin/env python3
"""
Full calibration + conditional-inference audit for a LeanNPE checkpoint.

Computes, on the FIXED validation split (stored strain, no remixing):
  - credible-interval coverage (50/68/90/95%) for all 11 parameters
  - SBC ranks of the truth within the posterior + KS test vs uniform
  - PP calibration curves (empirical vs nominal central-interval coverage)
  - context-shuffle delta-NLL (conditional-vs-marginal, at scale)
  - distance posterior: corr with truth, error and width vs network SNR

Usage:
  python scripts/lean_npe_diagnostics.py \
      --model model/lean_npe_v2/best_model.pth --data data/dataset_v2 \
      --outdir analysis/lean_npe_v2_diagnostics --n_events 1500 --n_post 600
"""
import argparse, glob, json, pickle, sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _ROOT)
sys.path.insert(0, f"{_ROOT}/src")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES

BLUE, GRAY, INK = "#3b6fd4", "#9aa4b2", "#3a3f47"
CI_LEVELS = [0.50, 0.68, 0.90, 0.95]


def load_val(data_dir, n):
    evs = []
    for fp in sorted(glob.glob(f"{data_dir}/validation/batch_*.pkl")):
        with open(fp, "rb") as fh:
            b = pickle.load(fh)
        for s in b["samples"]:
            if str(s.get("event_type")) == "noise":
                continue
            p0 = s["parameters"][0]
            if any(p0.get(k) is None for k in PARAM_NAMES):
                continue
            # rank-0 = loudest, matching training convention
            plist = sorted(
                s["parameters"],
                key=lambda p: ((p["mass_1"] * p["mass_2"]) ** 0.6 / (p["mass_1"] + p["mass_2"]) ** 0.2)
                              ** (5 / 6) / max(p["luminosity_distance"], 1.0),
                reverse=True)
            evs.append({
                "strain": np.stack([np.asarray(s["detector_data"][d]["strain"], dtype=np.float32)
                                    for d in ("H1", "L1", "V1")]),
                "theta": np.array([plist[0][k] for k in PARAM_NAMES], dtype=np.float32),
                "snr": float(s.get("network_snr", 0.0)),
                "n_signals": int(s.get("n_signals", 1)),
            })
            if len(evs) >= n:
                return evs
    return evs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/lean_npe_v2/best_model.pth")
    ap.add_argument("--data", default="data/dataset_v2")
    ap.add_argument("--outdir", default="analysis/lean_npe_v2_diagnostics")
    ap.add_argument("--n_events", type=int, default=1500)
    ap.add_argument("--n_post", type=int, default=600)
    ap.add_argument("--noise", choices=["gaussian", "real"], default="gaussian",
                    help="'real': same validation events with deterministic real-noise "
                         "crops + signal re-coloring (needs --noise_bank)")
    ap.add_argument("--noise_bank", default="data/noise_bank")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = LeanNPE(premerger=ckpt["args"].get("premerger", False), psd_cond=ckpt["args"].get("psd_cond", False) or False, psd_bands=ckpt["args"].get("psd_bands", 16), encoder_type=ckpt["args"].get("encoder_type", "conv"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"checkpoint: epoch={ckpt['epoch']} val_nll={ckpt['val_nll']:.4f}")

    if args.noise == "real":
        # same validation events, deterministic real-noise crops + re-colored
        # signals — built by the SAME loader used in training (no duplication)
        from experiments.remix_data import RemixDataset, build_memmap_cache
        vcache = Path(args.data) / "memmap" / "validation"
        if not (vcache / "events.json").exists():
            build_memmap_cache(args.data, "validation", str(vcache))
        ds = RemixDataset(str(vcache), remix=False, seed=1234,
                          real_noise_dir=args.noise_bank, real_noise_prob=1.0)
        n = min(args.n_events, len(ds))
        items = [ds[i] for i in range(n)]
        strain = torch.stack([x[0] for x in items])
        theta = torch.stack([x[1][0] for x in items]).to(device)
        snr = np.zeros(n)  # stored SNR labels are design-PSD; omit on real noise
        N, P = n, len(PARAM_NAMES)
        print(f"events: {N} (real-noise variant)")
    else:
        evs = load_val(args.data, args.n_events)
        N, P = len(evs), len(PARAM_NAMES)
        print(f"events: {N}")
        strain = torch.from_numpy(np.stack([e["strain"] for e in evs]))
        theta = torch.from_numpy(np.stack([e["theta"] for e in evs])).to(device)
        snr = np.array([e["snr"] for e in evs])

    # ---- contexts + NLL + shuffle ----
    ctx = []
    with torch.no_grad():
        for i in range(0, N, 64):
            ctx.append(model.encoder(strain[i:i + 64].to(device)))
    ctx = torch.cat(ctx)
    rank0 = torch.zeros(N, dtype=torch.long, device=device)
    with torch.no_grad():
        nll_true = model.nll(None, theta, rank0, context=ctx)
        nll_shuf = model.nll(None, theta, rank0, context=ctx[torch.randperm(N, device=device)])
    shuf_delta = float(nll_shuf.mean() - nll_true.mean())

    # ---- posterior sampling (batched over events) ----
    K = args.n_post
    samples = np.empty((N, K, P), dtype=np.float32)
    full_ctx = model._full_context(ctx, rank0)
    B = 32  # events per inversion batch
    with torch.no_grad():
        for i in range(0, N, B):
            fc = full_ctx[i:i + B]
            nb = fc.shape[0]
            fcr = fc.unsqueeze(1).expand(nb, K, fc.shape[1]).reshape(nb * K, -1)
            z = torch.randn(nb * K, P, device=device)
            y, _ = model.flow.inverse(z, fcr)
            samples[i:i + B] = model.scaler.denormalize(y).reshape(nb, K, P).cpu().numpy()
            if (i // B) % 10 == 0:
                print(f"  sampled {i + nb}/{N}")

    truth = theta.cpu().numpy()

    # ---- coverage + SBC ranks + PP ----
    coverage = {}
    for j, name in enumerate(PARAM_NAMES):
        cov = {}
        for lev in CI_LEVELS:
            lo = np.quantile(samples[:, :, j], 0.5 - lev / 2, axis=1)
            hi = np.quantile(samples[:, :, j], 0.5 + lev / 2, axis=1)
            cov[f"{int(lev*100)}%"] = float(((truth[:, j] >= lo) & (truth[:, j] <= hi)).mean())
        coverage[name] = cov
    ranks = (samples < truth[:, None, :]).mean(axis=1)  # [N, P] in [0,1]
    ks = {name: sps.kstest(ranks[:, j], "uniform") for j, name in enumerate(PARAM_NAMES)}

    # ---- distance metrics ----
    jd = PARAM_NAMES.index("luminosity_distance")
    d_med = np.median(samples[:, :, jd], axis=1)
    d_std = samples[:, :, jd].std(axis=1)
    dcorr = float(np.corrcoef(np.log(d_med), np.log(truth[:, jd]))[0, 1])
    frac_err = (d_med - truth[:, jd]) / truth[:, jd]

    report = {
        "checkpoint": {"epoch": ckpt["epoch"], "val_nll": ckpt["val_nll"]},
        "n_events": N, "n_post": K,
        "shuffle_delta_nll": shuf_delta,
        "dist_corr_logmedian": dcorr,
        "dist_frac_err_median_abs": float(np.median(np.abs(frac_err))),
        "coverage": coverage,
        "sbc_ks": {k: {"stat": float(v.statistic), "p": float(v.pvalue)} for k, v in ks.items()},
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- plots (single-hue, recessive grids) ----
    plt.rcParams.update({"axes.edgecolor": GRAY, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "axes.grid": True, "grid.color": "#e8ebef", "grid.linewidth": 0.6,
                         "font.size": 9})

    # coverage bars
    fig, axes = plt.subplots(1, len(CI_LEVELS), figsize=(16, 4.2), sharey=True)
    xs = np.arange(P)
    for a, lev in zip(axes, CI_LEVELS):
        vals = [coverage[n][f"{int(lev*100)}%"] for n in PARAM_NAMES]
        a.bar(xs, vals, color=BLUE, width=0.62)
        a.axhline(lev, color=INK, lw=1, ls="--")
        a.set_title(f"{int(lev*100)}% CI", color=INK)
        a.set_xticks(xs); a.set_xticklabels(PARAM_NAMES, rotation=60, ha="right")
        a.set_ylim(0, 1)
    axes[0].set_ylabel("empirical coverage")
    fig.suptitle(f"LeanNPE v2 (epoch {ckpt['epoch']}) — coverage, {N} events", color=INK)
    fig.tight_layout(); fig.savefig(out / "01_coverage.png", dpi=150); plt.close(fig)

    # SBC rank histograms
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for j, name in enumerate(PARAM_NAMES):
        a = axes.flat[j]
        a.hist(ranks[:, j], bins=20, color=BLUE, edgecolor="white", linewidth=0.5)
        a.axhline(N / 20, color=INK, lw=1, ls="--")
        a.set_title(f"{name}  (KS p={ks[name].pvalue:.3f})", color=INK, fontsize=9)
    axes.flat[-1].axis("off")
    fig.suptitle("SBC ranks — flat = calibrated", color=INK)
    fig.tight_layout(); fig.savefig(out / "02_sbc_ranks.png", dpi=150); plt.close(fig)

    # PP curves
    fig, ax = plt.subplots(figsize=(6, 6))
    levs = np.linspace(0.02, 0.98, 40)
    for j, name in enumerate(PARAM_NAMES):
        emp = [np.mean(np.abs(ranks[:, j] - 0.5) <= l / 2) for l in levs]
        ax.plot(levs, emp, lw=1.4,
                color=BLUE if name == "luminosity_distance" else GRAY,
                alpha=1.0 if name == "luminosity_distance" else 0.55,
                label="luminosity_distance" if name == "luminosity_distance" else None)
    ax.plot([0, 1], [0, 1], color=INK, lw=1, ls="--")
    ax.set_xlabel("nominal central coverage"); ax.set_ylabel("empirical")
    ax.legend(frameon=False)
    ax.set_title("PP calibration (distance highlighted, others gray)", color=INK)
    fig.tight_layout(); fig.savefig(out / "03_pp_curves.png", dpi=150); plt.close(fig)

    # distance error + width vs SNR
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].scatter(snr, np.abs(frac_err), s=7, color=BLUE, alpha=0.35, edgecolors="none")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("network SNR"); axes[0].set_ylabel("|fractional distance error|")
    axes[1].scatter(snr, d_std / d_med, s=7, color=BLUE, alpha=0.35, edgecolors="none")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("network SNR"); axes[1].set_ylabel("posterior width / median")
    fig.suptitle("Distance: error and width should shrink with SNR", color=INK)
    fig.tight_layout(); fig.savefig(out / "04_distance_vs_snr.png", dpi=150); plt.close(fig)

    print(json.dumps({k: report[k] for k in
                      ("shuffle_delta_nll", "dist_corr_logmedian", "dist_frac_err_median_abs")}, indent=2))
    print("coverage (distance):", coverage["luminosity_distance"])
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
