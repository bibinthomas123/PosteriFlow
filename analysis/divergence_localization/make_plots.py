"""Summary figures for Steps 4 & 5 from saved JSONs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"

# ---- Step 5: calibration vs SNR ----
s5 = json.load(open(HERE / "step5_snr_bins.json"))
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for domain, color in [("gaussian", "#3b6fd4"), ("real", "#e08a1e")]:
    rows = [r for r in s5[domain]["rows"] if r["n"] >= 10]
    x = list(range(len(rows)))
    labels = [r["bin"] for r in rows]
    axes[0].plot(x, [r["cov90_sci"] for r in rows], "o-", color=color, label=f"{domain} cov90")
    axes[0].plot(x, [r["cov50_sci"] for r in rows], "s--", color=color, alpha=0.6, label=f"{domain} cov50")
    axes[1].plot(x, [r["rail_spur"] for r in rows], "o-", color=color, label=f"{domain} any-dim")
    axes[1].plot(x, [r["rail_dL_spur"] for r in rows], "^--", color=color, alpha=0.7, label=f"{domain} dL")
    axes[2].plot(x, [r["mcorr"] for r in rows], "o-", color=color, label=f"{domain} mass corr")
    axes[2].plot(x, [r["dcorr"] for r in rows], "s--", color=color, alpha=0.6, label=f"{domain} dist corr")
axes[0].axhline(0.90, color="gray", ls=":"); axes[0].axhline(0.50, color="gray", ls=":")
axes[0].set_title("Science-param coverage vs SNR\n(target 0.50/0.90)"); axes[0].set_ylabel("coverage")
axes[1].set_title("Spurious railing vs SNR"); axes[1].set_ylabel("fraction")
axes[2].set_title("Point-estimate correlation vs SNR"); axes[2].set_ylabel("Pearson r")
for ax, rows in zip(axes, [rows, rows, rows]):
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30); ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
fig.suptitle("STEP 5 — ep19 in-distribution validation: calibration collapses at high SNR, "
             "point-estimate correlation stays healthy", y=1.02)
fig.tight_layout(); fig.savefig(FIGS / "step5_calibration_vs_snr.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---- Step 4: OOD by feature group ----
s4 = json.load(open(HERE / "step4_ood_decomp.json"))["events"]
groups = ["energy", "coherence", "arrival_time", "ampratio", "geom_embed", "psd", "context_full"]
events = list(s4.keys())
fig, ax = plt.subplots(figsize=(13, 6))
w = 0.12
x = np.arange(len(groups))
for k, ev in enumerate(events):
    vals = [s4[ev][g]["pct"] for g in groups]
    c = "#d1495b" if ev in ("GW150914", "GW170814") else None
    ax.bar(x + k * w, vals, w, label=ev, color=c)
ax.set_xticks(x + w * 2.5); ax.set_xticklabels(groups, rotation=20)
ax.set_ylabel("OOD percentile vs validation (RMS-z)")
ax.set_title("STEP 4 — OOD decomposition by encoder feature group: PSD/sensitivity dominates; coherence is in-distribution")
ax.axhline(95, color="gray", ls=":"); ax.legend(fontsize=8, ncol=3)
fig.tight_layout(); fig.savefig(FIGS / "step4_ood_groups.png", dpi=130)
plt.close(fig)
print("wrote step5_calibration_vs_snr.png, step4_ood_groups.png")
