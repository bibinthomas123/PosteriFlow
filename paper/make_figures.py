"""Figures for paper/posteriflow.tex, computed from the stored results.

    conda run -n ahsd python paper/make_figures.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # validated categorical order
INK, MUTED = "#1f1f1e", "#6b6a63"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})
rep = lambda m: json.load(open(f"analysis/ci/lean_npe_{m}/report.json"))


def bars(ax, groups, series, values, colors, ylabel):
    x = np.arange(len(groups)); w = 0.8 / len(series)
    for k, (name, col) in enumerate(zip(series, colors)):
        ax.bar(x + (k - (len(series) - 1) / 2) * w, values[k], w * 0.92, color=col, label=name)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel(ylabel)


# 1. permutation importance ----------------------------------------------------
fi = {c: json.load(open(f"analysis/feature_importance/{c}_v4.json")) for c in ("bbh", "bns", "nsbh")}
groups = ["chirp_mass_bank", "timing", "context stream", "geometry", "all input"]
labels = ["bank\ntokens", "timing\ntokens", "context\nstream", "geometry\nbranch", "whole\ninput"]
vals = [[fi[c].get(g, {}).get("dnll", 0.0) for g in groups] for c in ("bbh", "bns", "nsbh")]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
bars(ax, labels, ["BBH", "BNS", "NSBH"], vals, C[:3], "NLL rise when shuffled (nats)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3); fig.savefig(OUT / "feature_importance.pdf"); plt.close(fig)

# 2. matched comparison with dynesty: chirp-mass width ratio vs SNR -------------
rows = sum([json.load(open(f)) for f in sorted(glob.glob("analysis/dynesty_compare_v4/part*/comparison.json"))], [])
snr = np.array([r["snr"] for r in rows])
ratio = np.array([r["chirp_mass"]["npe"]["w90"] / r["chirp_mass"]["dynesty"]["w90"] for r in rows])
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.scatter(snr, ratio, s=14, color=C[0], label="v4, per event", zorder=3)
centres = {"8-20": 13, "20-35": 27, "35-1e+09": 45}
for k, (m, lab) in enumerate((("ab40_pivot", "v5 (pivot), median"), ("ab40_fine", "fine bank, no pivot, median"))):
    md = rep(m).get("matched_dynesty") or {}
    xs = [centres[b] for b in md.get("mc_w90_ratio_by_snr", {})]
    ax.plot(xs, list(md.get("mc_w90_ratio_by_snr", {}).values()), "-o", color=C[1 + k], lw=2, ms=4, label=lab)
ax.axhline(1, color=MUTED, lw=0.8, ls="--"); ax.text(57, 1.08, "dynesty", color=MUTED, ha="right")
ax.set_xlabel("network SNR"); ax.set_ylabel("90% chirp-mass width,\nNPE / dynesty"); ax.set_yscale("log")
ax.set_yticks([0.3, 1, 3, 10]); ax.set_yticklabels(["0.3", "1", "3", "10"])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5)
fig.savefig(OUT / "dynesty_width_ratio.pdf"); plt.close(fig)

# 3. calibration in real noise: 90% coverage per parameter ----------------------
params = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec", "theta_jn", "psi", "phase", "geocent_time", "a1", "a2"]
plab = ["$m_1$", "$m_2$", "$d_L$", "ra", "dec", r"$\theta_{JN}$", r"$\psi$", r"$\phi$", "$t_c$", "$a_1$", "$a_2$"]
fig, ax = plt.subplots(figsize=(7.0, 1.9))
cov = [[rep(m)["real"]["coverage"][p]["90"] for p in params] for m in ("ab40_pivot", "bns_v5", "nsbh_v5ff")]
bars(ax, plab, ["BBH v5", "BNS v5", "NSBH v5"], cov, C[:3], "coverage of the\n90% interval")
ax.axhline(0.9, color=INK, lw=0.8, ls="--"); ax.set_ylim(0.8, 0.95); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.savefig(OUT / "coverage_real.pdf"); plt.close(fig)

# 4. timing features (BNS A/B, 8 epochs): sky and merger-time informativeness ---
ab = [("ab_base", "previous recipe"), ("ab_trig", "+ timing features"), ("ab_trig_nd", "+ timing, no delay aug.")]
keys, klab = ["ra", "dec", "geocent_time", "chirp_mass"], ["ra", "dec", "merger\ntime", "chirp\nmass"]
vals = [[rep(m)["gaussian"]["sharpness"]["informativeness"][k] for k in keys] for m, _ in ab]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
bars(ax, klab, [l for _, l in ab], vals, C[:3], "posterior width / prior width")
ax.axhline(0.85, color=MUTED, lw=0.8, ls="--"); ax.text(3.45, 0.87, "gate", color=MUTED, ha="right", fontsize=7)
ax.set_ylim(0, 1.1); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=7); fig.savefig(OUT / "timing_ab.pdf"); plt.close(fig)

# 5. catalogue: published value relative to our posterior -----------------------
R = [json.load(open(f)) for f in glob.glob("results/catalog_v5/*.json")]
R = [r for r in R if "npe" in r and r["class"] == "BBH"]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.9), sharey=True)
for ax, (p, lab) in zip(axs, (("chirp_mass", "chirp mass"), ("mass_ratio", "mass ratio"), ("luminosity_distance", "distance"))):
    cov_txt = ""
    for k, (src, name) in enumerate((("npe", "NPE"), ("is", "importance sampling"))):
        z = np.clip([r[src][p]["z"] for r in R if src in r], -4, 4)
        inn = np.mean([r[src][p]["in90"] for r in R if src in r])
        ax.hist(z, bins=np.linspace(-4, 4, 17), histtype="step", lw=2, color=C[k], label=name)
        cov_txt = (cov_txt + ", " if k else "") + f"{'NPE' if k == 0 else 'IS'} {inn:.0%}"
    ax.axvline(0, color=MUTED, lw=0.8); ax.set_title(f"{lab}\n(in 90%: {cov_txt})", fontsize=8)
    ax.set_xlabel("(published - median) / std")
axs[0].set_ylabel("BBH events")
axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=7); fig.savefig(OUT / "catalogue.pdf"); plt.close(fig)

# 6. GW200115 mode probabilities vs dynesty ----------------------------------------
mo = json.load(open("analysis/modes/nsbh_v5ff.json"))
dyn = mo["dynesty_modes"]["probs"]; cen = mo["dynesty_modes"]["centres_m1_q_a1"]
series = [("dynesty", dyn), ("NPE", mo["NPE"]["probs"]),
          ("IS", mo["IS seed0"]["probs"]), ("IS, bank-seeded", mo["IS bank-seeded seed1"]["probs"])]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
bars(ax, [f"$m_1$={c[0]:.1f}\n$a_1$={c[2]:.2f}" for c in cen], [s for s, _ in series], [v for _, v in series],
     [INK] + C[:3], "posterior mass")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7); fig.savefig(OUT / "gw200115_modes.pdf"); plt.close(fig)
# 7. inclination on matched injections: NPE, NPE + likelihood correction, dynesty ---
inc = [json.load(open(f"analysis/inclination_is/ev{i:02d}/summary.json")) for i in range(6)]
inc.sort(key=lambda r: (r["truth"]["theta_jn"], r["truth"]["ra"]))
sky = {1.95: "A", 4.2: "B"}
th_lab = lambda t: "$\\pi/2$" if abs(t - np.pi / 2) < 0.01 else f"{t:.1f}"
rows = [f"$\\theta_{{JN}}$={th_lab(r['truth']['theta_jn'])}, sky {sky[r['truth']['ra']]}" for r in inc]
series = [("dynesty", lambda r: r["dynesty"], INK), ("NPE", lambda r: r["npe"], C[0]),
          ("NPE + correction", lambda r: r["is"][0], C[1])]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.5, 1]}, sharey=True)
for k, (name, get, col) in enumerate(series):
    y = np.arange(len(inc)) + (1 - k) * 0.24
    lo = [get(r)["abs_cos_90"][0] for r in inc]; hi = [get(r)["abs_cos_90"][1] for r in inc]
    ax.hlines(y, lo, hi, color=col, lw=2.2, label=name)
    ax.plot(lo, y, "|", color=col, ms=6, mew=1.5); ax.plot(hi, y, "|", color=col, ms=6, mew=1.5)
    bx.plot([get(r)["dL_med"] / r["truth"]["luminosity_distance"] for r in inc], y, "o", color=col,
            ms=5, mec="white", mew=0.8, label=name)
for i, r in enumerate(inc):
    ax.plot(abs(np.cos(r["truth"]["theta_jn"])), i, marker="*", ls="none", color=MUTED, ms=7, zorder=4,
            label="truth" if i == 0 else None)
ax.set_yticks(range(len(inc))); ax.set_yticklabels(rows); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xlabel(r"90% interval of $|\cos\theta_{JN}|$")
bx.axvline(1, color=MUTED, lw=0.8, ls="--"); bx.set_xlabel(r"median $d_L$ / true $d_L$")
bx.set_xlim(0.4, 2.0)
ax.legend(loc="lower center", bbox_to_anchor=(0.8, 1.0), ncol=4, fontsize=7)
fig.savefig(OUT / "inclination_is.pdf"); plt.close(fig)
print("figures ->", sorted(p.name for p in OUT.glob("*.pdf")))
