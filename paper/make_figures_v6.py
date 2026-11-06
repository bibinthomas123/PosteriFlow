#!/usr/bin/env python3
"""Figures for the current paper version (data in paper/data/).

    python paper/make_figures_v6.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2.0, "pdf.fonttype": 42,
})
PARAMS = (("chirp_mass", r"chirp mass $\mathcal{M}$"), ("mass_ratio", "mass ratio $q$"),
          ("luminosity_distance", "distance $d_L$"))


def dingo_comparison():
    """Median |offset| from the LVK median in LVK posterior sd, four methods."""
    # aggregate values: scripts/dingo_h2h/collect.py (DINGO-IS, network, corrected)
    # and the DINGO pre-importance-sampling result files, same 24 events
    # LVK-PSD DINGO run (results/dingo_h2h_lvk), 28 events
    vals = {"LeanNPE network": (0.75, 0.40, 1.28), "LeanNPE + correction": (0.17, 0.17, 0.36),
            "DINGO network": (0.30, 0.20, 0.51), "DINGO-IS": (0.15, 0.14, 0.48)}
    style = {"LeanNPE network": (BLUE, "////"), "LeanNPE + correction": (BLUE, None),
             "DINGO network": (ORANGE, "////"), "DINGO-IS": (ORANGE, None)}
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    x = np.arange(3)
    w = 0.19
    for i, (k, v) in enumerate(vals.items()):
        c, h = style[k]
        ax.bar(x + (i - 1.5) * w, v, w * 0.9, color="white" if h else c, edgecolor=c, hatch=h,
               linewidth=1.0, label=k)
    ax.set_xticks(x, ["chirp mass", "mass ratio", "distance"])
    ax.set_ylabel("median |offset| / LVK sd")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              columnspacing=0.8, handlelength=1.4)
    ax.set_ylim(0, 1.3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "dingo_h2h.pdf")
    plt.close(fig)


def dingo_per_event():
    ev = json.load(open(DATA / "h2h_per_event.json"))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), sharey=False)
    for ax, (p, lab) in zip(axes, PARAMS):
        a = np.array([e["is_" + p] for e in ev])
        b = np.array([e["dingo_" + p] for e in ev])
        lim = max(2.5, float(np.ceil(max(np.abs(a).max(), np.abs(b).max()) * 2) / 2))
        lim = min(lim, 4.0)
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls="--")
        ax.axhline(0, color=GRID, lw=0.8)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.scatter(np.clip(b, -lim, lim), np.clip(a, -lim, lim), s=18, color=BLUE,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(lab)
        ax.set_xlabel("DINGO-IS offset (LVK sd)")
    axes[0].set_ylabel("LeanNPE corrected offset (LVK sd)")
    fig.tight_layout()
    fig.savefig(FIG / "dingo_per_event.pdf")
    plt.close(fig)


def snr_scaling():
    """Median chirp-mass posterior width x SNR per SNR bin (certification)."""
    bins = ["8-12", "12-20", "20-35", "35-60", "60+"]
    # 8-epoch runs on the 200k-event dataset; production: 10^6 events, 30 epochs
    runs = {"24-slot bank": ((2.5, 2.4, 3.3, 5.2, 7.8), BLUE, "-", "o"),
            "120-slot bank": ((2.5, 2.4, 3.3, 4.7, 6.0), ORANGE, "-", "s"),
            "trigger alignment": ((2.7, 2.7, 3.4, 5.6, 8.4), AQUA, "--", "^"),
            "SNR-weighted loss": ((2.5, 2.2, 2.9, 4.6, 6.9), YELLOW, "--", "v"),
            "production (30 epochs)": ((1.9, 2.3, 2.9, 3.1, 3.9), INK, "-", "D")}
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(bins))
    for k, (v, c, ls, m) in runs.items():
        ax.plot(x, v, color=c, ls=ls, marker=m, ms=4.5, label=k)
    ax.set_xticks(x, bins)
    ax.set_xlabel("network SNR")
    ax.set_ylabel(r"mean $\sigma_{\mathcal{M}}$ (scaled) $\times$ SNR")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "snr_scaling.pdf")
    plt.close(fig)


def bank_floor():
    """Raw bank chirp-mass error x SNR: best slot vs nearest-to-truth slot."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for name, c, lab in (("ab8_base", BLUE, "24 slots"), ("ab8_dense", ORANGE, "120 slots")):
        r = json.load(open(DATA / f"{name}.json"))["bins"]
        ks = list(r)
        snr = [r[k]["snr"] for k in ks]
        ax.plot(snr, [r[k]["max"]["err_x_snr"] for k in ks], color=c, marker="o", ms=4,
                label=f"{lab}: best slot")
        ax.plot(snr, [r[k]["oracle"]["err_x_snr"] for k in ks], color=c, ls="--", marker="o",
                ms=4, mfc="white", label=f"{lab}: slot nearest truth")
    ax.set_xscale("log")
    ax.set_xticks([10, 15, 25, 45, 80], ["10", "15", "25", "45", "80"])
    ax.set_xlabel("network SNR (bin median)")
    ax.set_ylabel(r"median $|\Delta\ln\mathcal{M}|\times$SNR")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "bank_floor.pdf")
    plt.close(fig)


def overlap_capacity():
    r = json.load(open(DATA / "overlap_capacity_bbh_v5.json"))["per_k"]
    ks = [k for k in r if int(k) <= 5]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
    colors = [BLUE, ORANGE, AQUA, YELLOW, MUTED]
    for rank in range(5):
        kk = [int(k) for k in ks if f"rank{rank}" in r[k]]
        if not kk:
            continue
        cov = [r[str(k)][f"rank{rank}"]["cov90_headline"] for k in kk]
        err = [r[str(k)][f"rank{rank}"]["mc_frac_err"] for k in kk]
        lab = "loudest" if rank == 0 else f"rank {rank}"
        axes[0].plot(kk, cov, color=colors[rank], marker="o", ms=4, label=lab)
        axes[1].plot(kk, np.array(err) * 100, color=colors[rank], marker="o", ms=4, label=lab)
    axes[0].axhline(0.9, color=MUTED, ls="--", lw=0.8)
    axes[0].set_ylabel("90% coverage (headline params)")
    axes[1].set_ylabel(r"median $|\Delta\mathcal{M}|/\mathcal{M}$ (%)")
    axes[1].set_ylim(0, 60)
    for ax in axes:
        ax.set_xlabel("signals in the segment")
        ax.set_xticks(range(1, 6))
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "overlap_capacity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    dingo_comparison()
    dingo_per_event()
    snr_scaling()
    bank_floor()
    overlap_capacity()
    print("figures written to", FIG)
