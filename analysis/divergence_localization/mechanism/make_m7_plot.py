import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
m7 = json.load(open(HERE / "m7_event_selectivity.json"))
ev = m7["events"]

fig, ax = plt.subplots(1, 3, figsize=(17, 5))

# one-way base_conc trends
labels = {"SNR": ["5-12", "12-18", "18-25", "25-40"],
          "Mc": ["5-12", "12-20", "20-30", "30-60"],
          "q": ["0.1-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]}
for a, key, xlab in [(ax[0], "SNR", "network SNR bin"), (ax[1], "Mc", "chirp mass bin"),
                     (ax[2], "q", "mass ratio q bin")]:
    conc = m7["one_way"][key][0]
    a.plot(labels[key], conc, "o-", color="#d1495b", lw=2)
    a.axhline(1.0, color="gray", ls=":", label="calibrated (=1)")
    a.set_xlabel(xlab); a.set_ylabel("base concentration  ||z||²/D")
    a.set_title(f"overconfidence vs {key}"); a.grid(alpha=0.25); a.legend(fontsize=8)
fig.suptitle("M7 — overconfidence (||z||²/D) rises with SNR, chirp mass, and mass ratio "
             "(validation, ZERO PSD-OOD)", y=1.02)
fig.tight_layout(); fig.savefig(HERE / "figs/m7_oneway.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 2D quadrant heatmap + events
q = m7["quadrants_snr_gt18"]
grid = np.array([[q["Mc<20,q<0.7"]["base_conc"], q["Mc<20,q>=0.7"]["base_conc"]],
                 [q["Mc>=20,q<0.7"]["base_conc"], q["Mc>=20,q>=0.7"]["base_conc"]]])
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(grid, cmap="magma", origin="lower", aspect="auto",
               extent=[0, 2, 0, 2], vmin=1.8, vmax=3.8)
fig.colorbar(im, label="base concentration ||z||²/D  (higher = more overconfident/biased)")
for (i, j), lab in [((0, 0), "Mc<20\nq<0.7"), ((0, 1), "Mc<20\nq≥0.7"),
                    ((1, 0), "Mc≥20\nq<0.7"), ((1, 1), "Mc≥20\nq≥0.7")]:
    ax.text(j + 0.5, i + 0.5, f"{grid[i, j]:.2f}\n{lab}", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
# overlay events by (Mc>=20?, q>=0.7?) with small jitter
rng = np.random.default_rng(0)
for name, e in ev.items():
    xi = 1 if e["q"] >= 0.7 else 0
    yi = 1 if e["Mc"] >= 20 else 0
    prob = name in ("GW150914", "GW170814")
    ax.scatter(xi + 0.5 + rng.uniform(-0.28, 0.28), yi + 0.5 + rng.uniform(-0.28, 0.28),
               s=180, marker="*", color=("#39ff14" if prob else "#66ccff"),
               edgecolor="black", linewidth=0.8, zorder=5)
    ax.annotate(name.replace("GW", ""), (xi + 0.5, yi + 0.5), fontsize=7.5,
                color="white", xytext=(6, -10 if not prob else 8),
                textcoords="offset points")
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["q<0.7 (unequal)", "q≥0.7 (near-equal)"])
ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["Mc<20 (light)", "Mc≥20 (heavy)"])
ax.set_title("M7 — the victim corner (SNR>18): heavy × near-equal-mass = 3.71\n"
             "green★ = worsening events (GW150914, GW170814); blue★ = others")
fig.tight_layout(); fig.savefig(HERE / "figs/m7_corner.png", dpi=130)
plt.close(fig)
print("wrote figs/m7_oneway.png, figs/m7_corner.png")
