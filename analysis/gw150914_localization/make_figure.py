"""Summary figure: the amplitude->distance collapse + encoder info content."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
c4b = json.load(open(OUT / "step4b_snr_domain.json"))["design_snr_sweep"]
c4c = json.load(open(OUT / "step4c_sourceclass.json"))
enc = json.load(open(OUT / "step15_encoder_audit.json"))
extr = json.load(open(OUT / "step1b_stratified.json"))["extrinsic_info"]["context_R2"]

# colorblind-safe
COL = {"light": "#0072B2", "moderate": "#009E73", "heavy": "#D55E00", "gw": "#CC79A7"}
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# ── Panel A: predicted vs true dL (flat = ignores amplitude) ─────────────────
a = ax[0]
lim = [100, 2200]
a.plot(lim, lim, "k--", lw=1, alpha=0.6, label="ideal (dL̂ = dL)")
cls = [("light_sym  Mc~9  q1.0", COL["light"], "light BBH (Mc~9)"),
       ("moderate   Mc~17 q0.8", COL["moderate"], "moderate (Mc~17)"),
       ("heavy_GW150914 Mc~29", COL["heavy"], "heavy / GW150914 (Mc~29)")]
for key, col, lab in cls:
    pts = c4c[key]["points"]
    td = [p["true_dL"] for p in pts]; md = [p["dL_med"] for p in pts]
    a.plot(td, md, "-o", color=col, lw=2, ms=5, label=lab)
# design sweep at GW150914 masses (4b)
td = [r["true_dL"] for r in c4b]; md = [r["dL_med"] for r in c4b]
a.plot(td, md, ":s", color=COL["heavy"], lw=1.5, ms=4, alpha=0.7)
a.scatter([440], [735], s=170, marker="*", color=COL["gw"], zorder=5,
          edgecolor="k", label="real GW150914 (NPE 735)")
a.scatter([440], [466], s=90, marker="D", color="k", zorder=5, label="dynesty GW150914 (466)")
a.set_xlabel("true luminosity distance [Mpc]")
a.set_ylabel("NPE posterior-median distance [Mpc]")
a.set_title("Distance response is flat in amplitude\n(predicted dL ≈ f(chirp mass), not true dL)")
a.set_xlim(lim); a.set_ylim([100, 1300]); a.legend(fontsize=8, loc="upper left")
a.grid(alpha=0.2)

# ── Panel B: encoder information content ─────────────────────────────────────
b = ax[1]
bars = [("Mc → dL\n(prior corr.)", enc["B_mc_to_logdL_r2"], "#999999"),
        ("context → dL", enc["A_ctx_to_logdL_r2"], "#0072B2"),
        ("context → dL | Mc", enc["C_ctx_to_logdL_given_mc"], "#56B4E9"),
        ("context → SNR", enc["D_ctx_to_logSNR_r2"], "#009E73"),
        ("context → SNR | Mc", enc["E_ctx_to_logSNR_given_mc"], "#66C2A5"),
        ("context → θ_jn", extr["theta_jn"], "#D55E00"),
        ("context → ra", extr["ra"], "#E69F00")]
names = [x[0] for x in bars]; vals = [x[1] for x in bars]; cols = [x[2] for x in bars]
b.barh(range(len(bars)), vals, color=cols)
b.set_yticks(range(len(bars))); b.set_yticklabels(names, fontsize=9)
b.invert_yaxis()
for i, v in enumerate(vals):
    b.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
b.set_xlabel("probe R²  (design validation)")
b.set_xlim([0, 1.0])
b.set_title("Encoder information content\namplitude IS encoded (SNR|Mc=0.65) but unused for distance")
b.grid(alpha=0.2, axis="x")

fig.tight_layout()
fig.savefig(OUT / "summary_mechanism.png", dpi=120)
print(f"wrote {OUT/'summary_mechanism.png'}")
