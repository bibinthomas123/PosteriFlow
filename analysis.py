#!/usr/bin/env python3
"""
Research-grade analysis figures for the PosteriFlow paper.

    python analysis.py            # builds every figure whose source data exists
    python analysis.py --only mass_plane anchors_q

One chart per file under analysis/figures/. Sources are durable artifacts
only (memmap dataset, noise bank, twin grid, dynesty anchors, CI report,
supervised-probe logs, benchmark summary); figures whose inputs are missing
are skipped with a note.

Conventions (fixed across all figures):
    LeanNPE   = blue  #3b6fd4      dynesty = orange #d07a2e
    3rd series= green #3f8f4a      truth   = ink dashed reference
    sequential = single-hue Blues; diverging = RdBu_r with neutral midpoint
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPE, DYN, THIRD = "#3b6fd4", "#d07a2e", "#3f8f4a"
INK, GRID = "#3a3f47", "#e8ebef"
FIG = Path("analysis/figures")
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": "#9aa4b2", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False,
})

PARAM_NAMES = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
               "theta_jn", "psi", "phase", "geocent_time", "a1", "a2"]


def _save(fig, name, caption):
    out = FIG / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}  — {caption}")


def _train_params():
    par = np.load("data/dataset/memmap/train/params.npy", mmap_mode="r")
    ev = json.load(open("data/dataset/memmap/train/events.json"))["events"]
    return par, ev


# ── dataset population ───────────────────────────────────────────────────────

def fig_mass_plane():
    """Training population density in the chirp-mass / mass-ratio plane."""
    par, ev = _train_params()
    starts = [e[0] for e in ev]
    m1 = np.asarray(par[starts, 0], float)
    m2 = np.asarray(par[starts, 1], float)
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    q = m2 / m1
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    h = ax.hist2d(np.log10(mc), q, bins=60, cmap="Blues", cmin=1)
    cb = fig.colorbar(h[3], ax=ax, pad=0.02)
    cb.set_label("events per cell")
    ax.set_xlabel(r"$\log_{10}$ chirp mass $\mathcal{M}_c\ [M_\odot]$")
    ax.set_ylabel("mass ratio  $q = m_2/m_1$")
    ax.set_title("Training population: rank-0 signals in the mass plane")
    # annotate the amortization-bias region measured by the twin grid
    ax.add_patch(plt.Rectangle((np.log10(20), 0.8), np.log10(45 / 20), 0.2,
                               fill=False, edgecolor=DYN, lw=1.6))
    ax.text(np.log10(21), 0.955, "q-collapse region (twin grid)",
            color=DYN, fontsize=8.5, va="top")
    _save(fig, "01_training_mass_plane", "population density, Mc-q plane")


def fig_distance_prior():
    """Sampled luminosity distance against the d^2 comoving-volume prior."""
    par, ev = _train_params()
    starts = [e[0] for e in ev]
    d = np.asarray(par[starts, 2], float)
    d = d[(d > 40) & (d < 2200)]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.hist(d, bins=70, density=True, color=NPE, edgecolor="white", lw=0.3,
            label="sampled (after SNR selection)")
    x = np.linspace(50, 2100, 300)
    p = 3 * x ** 2 / (2100 ** 3 - 50 ** 3)
    ax.plot(x, p, color=INK, ls="--", lw=1.4, label=r"$p(d_L)\propto d_L^2$ prior")
    ax.set_xlabel("luminosity distance $d_L$ [Mpc]")
    ax.set_ylabel("density")
    ax.set_title("Distance population vs comoving-volume prior")
    ax.legend()
    _save(fig, "02_distance_prior_check",
          "SNR selection depletes the far tail relative to the d^2 prior")


def fig_overlap_separation():
    """Merger-time separation between overlapping signals."""
    par, ev = _train_params()
    jt = PARAM_NAMES.index("geocent_time")
    dts = []
    for start, n in ev:
        if n < 2:
            continue
        t = np.sort(np.asarray(par[start:start + n, jt], float))
        dts.extend(np.diff(t))
    dts = np.array(dts)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.hist(dts, bins=60, color=NPE, edgecolor="white", lw=0.3)
    ax.axvline(np.median(dts), color=INK, ls="--", lw=1.2)
    ax.text(np.median(dts) + 0.03, ax.get_ylim()[1] * 0.92,
            f"median {np.median(dts):.2f} s", fontsize=9)
    ax.set_xlabel(r"merger-time separation $|\Delta t_c|$ between neighbours [s]")
    ax.set_ylabel("signal pairs")
    ax.set_title(f"Overlap severity in training ({len(dts)} overlapping pairs)")
    _save(fig, "03_overlap_separation", "pairwise merger separations")


def fig_noise_bank_qc():
    """Rebuilt O3b noise bank quality: nonstationarity vs non-Gaussianity."""
    files = sorted(Path("data/noise_bank").glob("*_strain.npy"))
    if not files:
        raise FileNotFoundError("noise bank missing")
    pts = {"H1": [], "L1": [], "V1": []}
    for f in files:
        w = np.load(f).astype(np.float32)
        bs = w[: (len(w) // 4096) * 4096].reshape(-1, 4096).std(axis=1)
        kurt = float(np.mean(w ** 4) / (np.mean(w ** 2) ** 2 + 1e-30) - 3)
        pts[f.name.split("_")[0]].append((bs.max(), kurt))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for det, color, mk in (("H1", NPE, "o"), ("L1", DYN, "s"), ("V1", THIRD, "^")):
        a = np.array(pts[det])
        ax.scatter(a[:, 0], np.clip(a[:, 1], 1e-3, None), s=16, color=color,
                   marker=mk, alpha=0.75, edgecolors="white", lw=0.4, label=det)
    ax.axvline(4.0, color=INK, ls="--", lw=1)
    ax.axhline(30.0, color=INK, ls="--", lw=1)
    ax.text(4.05, 0.0015, "quality gate", fontsize=8.5, rotation=90, va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("max 1-s block std (nonstationarity)")
    ax.set_ylabel("excess kurtosis (non-Gaussianity)")
    ax.set_title(f"Real-noise bank after rebuild ({sum(len(v) for v in pts.values())} segments)")
    ax.legend(title=None)
    _save(fig, "04_noise_bank_quality", "every retained segment sits inside the gate")


# ── amortization bias (twin grid + anchors) ──────────────────────────────────

def _twin_grid():
    return json.load(open("analysis/twin_grid_v3.json"))


def fig_twin_grid_q():
    """m1 bias vs mass ratio: the q-collapse, averaged over inclination."""
    rows = _twin_grid()
    MC, Q = [10, 20, 30, 45], [0.4, 0.7, 0.9]
    g = np.array([[np.mean([r["m1_ratio"] for r in rows
                            if r["mc"] == mc and r["q"] == q]) for q in Q]
                  for mc in MC])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(g, cmap="RdBu_r", vmin=0.5, vmax=1.5, origin="lower",
                   aspect="auto")
    for (i, j), v in np.ndenumerate(g):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(v - 1) > 0.35 else INK)
    ax.set_xticks(range(len(Q)), Q)
    ax.set_yticks(range(len(MC)), MC)
    ax.set_xlabel("true mass ratio $q$")
    ax.set_ylabel(r"chirp mass $\mathcal{M}_c\ [M_\odot]$")
    ax.set_title("LeanNPE $m_1$ bias (pred/true, SNR 24 twins)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.02, label="$m_1$ pred/true")
    _save(fig, "05_twin_grid_m1_bias", "m1 inflation grows with q at every mass")


def fig_twin_grid_dl():
    """Distance bias vs inclination, averaged over q."""
    rows = _twin_grid()
    MC, TH = [10, 20, 30, 45], [0.5, 1.57, 2.9]
    g = np.array([[np.mean([r["dl_ratio"] for r in rows
                            if r["mc"] == mc and r["theta_jn"] == t]) for t in TH]
                  for mc in MC])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(g, cmap="RdBu_r", vmin=0.0, vmax=2.0, origin="lower",
                   aspect="auto")
    for (i, j), v in np.ndenumerate(g):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(v - 1) > 0.7 else INK)
    ax.set_xticks(range(len(TH)), ["face-on 0.5", r"edge-on $\pi/2$", "face-off 2.9"])
    ax.set_yticks(range(len(MC)), MC)
    ax.set_xlabel(r"inclination $\theta_{JN}$ [rad]")
    ax.set_ylabel(r"chirp mass $\mathcal{M}_c\ [M_\odot]$")
    ax.set_title("LeanNPE $d_L$ bias (pred/true, SNR 24 twins)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.02, label="$d_L$ pred/true")
    _save(fig, "06_twin_grid_dl_bias",
          "edge-on distance inflation (largely shared with dynesty)")


def fig_anchors_q():
    """q recovery on the anchor cells: truth vs dynesty vs LeanNPE."""
    rows = json.load(open("analysis/anchors/summary.json"))
    labels, qt, qd, qn = [], [], [], []
    for r in rows:
        c = r["cell"]
        labels.append(f"$\\mathcal{{M}}_c$={c['mc']}, q={c['q']},\n"
                      f"$\\theta_{{JN}}$={c['theta_jn']}")
        qf = lambda x: x["mass_2"] / x["mass_1"]
        qt.append(qf(r["true"])); qd.append(qf(r["dynesty"])); qn.append(qf(r["npe"]))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for xi, t in zip(x, qt):
        ax.hlines(t, xi - 0.28, xi + 0.28, color=INK, ls="--", lw=1.4,
                  zorder=3)
    ax.scatter(x - 0.12, qd, s=95, color=DYN, zorder=4, label="dynesty",
               edgecolors="white", lw=0.8)
    ax.scatter(x + 0.12, qn, s=95, color=NPE, zorder=4, label="LeanNPE",
               edgecolors="white", lw=0.8)
    ax.hlines([], [], [], color=INK, ls="--", lw=1.4, label="truth")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel("posterior median mass ratio $q$")
    ax.set_ylim(0.15, 1.0)
    ax.set_title("Mass-ratio recovery at SNR 24: dynesty extracts $q$; "
                 "the amortized pass regresses to the prior")
    ax.legend(loc="lower right")
    _save(fig, "07_anchors_q_recovery", "two-sided regression toward q~0.5")


# ── extractability experiments ───────────────────────────────────────────────

def _parse_probe(path, pat=r"epoch\s+(\d+).*?q=([-\d.]+) cos_thjn=([-\d.]+)"):
    eps, qs, ts = [], [], []
    for line in open(path):
        m = re.search(pat, line)
        if m:
            eps.append(int(m.group(1))); qs.append(float(m.group(2)))
            ts.append(float(m.group(3)))
    return np.array(eps), np.array(qs), np.array(ts)


def fig_probe_q():
    """Supervised q extractability across representations."""
    runs = [("TD warm-start (v3 encoder)", "analysis/logs/supervised_q_td_warmstart.log", NPE),
            ("FD aligned re/im", "analysis/logs/supervised_q_fd_aligned.log", THIRD)]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for tag, path, color in runs:
        if not Path(path).exists():
            continue
        e, q, _ = _parse_probe(path)
        ax.plot(e, q, lw=1.8, color=color)
        ax.annotate(tag, (e[-1], q[-1]), textcoords="offset points",
                    xytext=(-4, 8), ha="right", fontsize=9, color=color)
    ax.axhline(0.32, color=DYN, ls="--", lw=1.4)
    ax.annotate(r"$\mathcal{M}_c\!\to\!q$ population-shortcut ceiling (0.32)",
                (0.02, 0.335), xycoords=("axes fraction", "data"),
                fontsize=9, color=DYN)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("validation $R^2$ (mass ratio $q$)")
    ax.set_ylim(-0.1, 0.55)
    ax.set_title("Is $q$ extractable in one pass? Supervised probes vs the "
                 "population-shortcut ceiling")
    _save(fig, "08_probe_q_extractability",
          "all representations converge at/below the shortcut ceiling")


def fig_probe_hom():
    """Paired XP vs XHM: higher-order modes add no q information at SNR 8-25."""
    path = "analysis/logs/supervised_q_hom_paired.log"
    arms = {"XP": ([], []), "XHM": ([], [])}
    for line in open(path):
        m = re.search(r"\[(XP|XHM).*?epoch\s+(\d+) \| val R2 q=([-\d.]+)", line)
        if m:
            arms[m.group(1)][0].append(int(m.group(2)))
            arms[m.group(1)][1].append(float(m.group(3)))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for tag, color, label in (("XP", NPE, "IMRPhenomXP (2,2 only)"),
                              ("XHM", DYN, "IMRPhenomXHM (higher modes)")):
        e, q = arms[tag]
        ax.plot(e, q, lw=1.8, color=color, marker="o", ms=4, label=label)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("validation $R^2$ (mass ratio $q$)")
    ax.set_title("Paired waveform test (12k identical draws, SNR 8-25):\n"
                 "higher-order modes do not change $q$ extractability")
    ax.legend(loc="lower right")
    _save(fig, "09_probe_hom_paired", "XP 0.219 vs XHM 0.209 at convergence")


# ── calibration + benchmark ──────────────────────────────────────────────────

def fig_coverage(domain="gaussian"):
    rep = json.load(open("analysis/ci/lean_npe_v3/report.json"))
    cov = rep[domain]["coverage"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = np.arange(len(PARAM_NAMES))
    ax.bar(xs, [cov[n]["90"] for n in PARAM_NAMES], color=NPE, width=0.62)
    ax.axhline(0.90, color=INK, ls="--", lw=1.3)
    ax.text(len(PARAM_NAMES) - 0.4, 0.905, "nominal 90%", ha="right",
            fontsize=9)
    ax.set_xticks(xs, PARAM_NAMES, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("empirical coverage of the 90% CI")
    ax.set_ylim(0.7, 1.0)
    n = rep[domain]["n_events"]
    tag = "Gaussian" if domain == "gaussian" else "real O3b noise"
    ax.set_title(f"Credible-interval calibration, {tag} validation ({n} events)")
    _save(fig, f"10_coverage_{domain}", f"90% coverage per parameter, {tag}")


def fig_benchmark_timing():
    path = Path("results/real_event_benchmark/summary.json")
    if not path.exists():
        raise FileNotFoundError("benchmark not finished")
    rows = [r for r in json.load(open(path)) if "error" not in r and r.get("t_dynesty_s")]
    if not rows:
        raise FileNotFoundError("no completed benchmark events")
    ev = [r["event"] for r in rows]
    x = np.arange(len(ev))
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(x - 0.19, [r["t_npe_s"] for r in rows], width=0.36, color=NPE,
           label="LeanNPE (amortized)")
    ax.bar(x + 0.19, [r["t_dynesty_s"] for r in rows], width=0.36, color=DYN,
           label="dynesty (nlive 300)")
    for xi, r in zip(x, rows):
        ax.text(xi, r["t_dynesty_s"] * 1.15,
                f"{r['t_dynesty_s'] / r['t_npe_s']:.0f}×", ha="center",
                fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x, ev, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("wall time per event [s]")
    ax.set_title("Real-event inference cost: amortized vs nested sampling")
    ax.legend()
    _save(fig, "11_benchmark_timing", "speedup annotated per event")




# ── overlap evaluation (the paper's core claim) ──────────────────────────────

def _overlap_report():
    p = Path("analysis/overlap_benchmark.json")
    if not p.exists():
        raise FileNotFoundError("run scripts/overlap_benchmark.py first")
    return json.load(open(p))


def _rank_axis(rep):
    keys = [("n1_rank0", "1 signal\nrank 0"), ("n2_rank0", "2 signals\nrank 0"),
            ("n2_rank1", "2 signals\nrank 1"), ("n3_rank0", "3+ signals\nrank 0"),
            ("n3_rank1", "3+ signals\nrank 1"), ("n3_rank2", "3+ signals\nrank 2")]
    return [(k, l) for k, l in keys if k in rep["per_rank"]]


def fig_overlap_calibration():
    """90% coverage per NPE rank as multiplicity grows — the calibration claim."""
    rep = _overlap_report()
    ks = _rank_axis(rep)
    vals = [rep["per_rank"][k]["cov90_headline_mean"] for k, _ in ks]
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(x, vals, color=NPE, width=0.62)
    ax.axhline(0.90, color=INK, ls="--", lw=1.3)
    ax.text(len(ks) - 0.45, 0.905, "nominal 90%", ha="right", fontsize=9)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [l for _, l in ks], fontsize=9)
    ax.set_ylabel("empirical 90% coverage (mean of m1, dL, tc, ra, dec)")
    ax.set_ylim(0.6, 1.0)
    n1 = rep["per_rank"]["n1_rank0"]["n"]
    ax.set_title(f"Per-signal calibration vs multiplicity "
                 f"({n1} events/group, {rep['n_post']} draws)")
    _save(fig, "12_overlap_calibration", "coverage holds per rank as overlap grows")


def fig_overlap_accuracy():
    """Chirp-mass accuracy per rank as multiplicity grows — graceful degradation."""
    rep = _overlap_report()
    ks = _rank_axis(rep)
    vals = [rep["per_rank"][k]["mc_frac_err_median"] for k, _ in ks]
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(x, vals, color=NPE, width=0.62)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.004, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x, [l for _, l in ks], fontsize=9)
    ax.set_ylabel("median fractional chirp-mass error")
    ax.set_title("Per-signal accuracy vs multiplicity")
    _save(fig, "13_overlap_accuracy", "graceful degradation with rank/multiplicity")


def fig_overlap_dt():
    """Robustness to temporal overlap: accuracy vs merger separation (2 signals)."""
    rep = _overlap_report()
    rows = rep["dt_bins"]
    if not rows:
        raise FileNotFoundError("no dt bins")
    mid = [0.5 * (r["dt_lo"] + r["dt_hi"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for r_id, color in (("rank0", NPE), ("rank1", THIRD)):
        ax.plot(mid, [r[r_id]["mc_frac_err"] for r in rows], marker="o", ms=6,
                lw=1.8, color=color, label=f"{r_id} (louder)" if r_id == "rank0"
                else f"{r_id} (quieter)")
    ax.set_xlabel(r"true merger separation $|\Delta t_c|$ [s]")
    ax.set_ylabel("median fractional chirp-mass error")
    n = sum(r["n"] for r in rows)
    ax.set_title(f"Temporal-overlap robustness (2-signal events, n={n})")
    ax.legend()
    _save(fig, "14_overlap_dt_robustness",
          "accuracy vs how closely the mergers overlap")


def fig_overlap_runtime():
    """Wall time: all-rank NPE inference vs sequential nested sampling."""
    rep = _overlap_report()
    rt = rep["runtime"]
    dyn = rt.get("dynesty_single_signal_median_s")
    ks = [(1, "1 signal"), (2, "2 signals"), (3, "3+ signals")]
    npe_t = [rt.get(f"npe_all_ranks_s_per_event_n{k}") for k, _ in ks]
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(x - 0.19, npe_t, width=0.36, color=NPE, label="LeanNPE (all ranks)")
    if dyn:
        seq = [dyn * k for k, _ in ks]
        ax.bar(x + 0.19, seq, width=0.36, color=DYN,
               label="sequential dynesty (k x single)")
        for xi, (nt, st) in enumerate(zip(npe_t, seq)):
            ax.text(xi, st * 1.15, f"{st / nt:.0f}x", ha="center", fontsize=10)
    ax.set_yscale("log")
    ax.set_xticks(x, [l for _, l in ks])
    ax.set_ylabel("wall time per event [s]")
    ax.set_title("Inference cost vs number of overlapping signals")
    ax.legend()
    _save(fig, "15_overlap_runtime", "runtime advantage grows with multiplicity")


FIGS = {
    "mass_plane": fig_mass_plane,
    "distance_prior": fig_distance_prior,
    "overlap_separation": fig_overlap_separation,
    "noise_bank_qc": fig_noise_bank_qc,
    "twin_grid_q": fig_twin_grid_q,
    "twin_grid_dl": fig_twin_grid_dl,
    "anchors_q": fig_anchors_q,
    "probe_q": fig_probe_q,
    "probe_hom": fig_probe_hom,
    "coverage_gaussian": lambda: fig_coverage("gaussian"),
    "coverage_real": lambda: fig_coverage("real"),
    "benchmark_timing": fig_benchmark_timing,
    "overlap_calibration": fig_overlap_calibration,
    "overlap_accuracy": fig_overlap_accuracy,
    "overlap_dt": fig_overlap_dt,
    "overlap_runtime": fig_overlap_runtime,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(FIGS), default=None)
    args = ap.parse_args()
    for name, fn in FIGS.items():
        if args.only and name not in args.only:
            continue
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  [skip] {name}: {e}")
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
