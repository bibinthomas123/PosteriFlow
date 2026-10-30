"""Plot M18: held-out R^2(q) per probe family and mass regime."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "m18_nonlinear_probe.json"))
MODELS = ["ridge", "mlp-256", "mlp-512-256", "rand-forest", "grad-boost"]
REGIMES = [r for r in ["light  m1<20", "mid    20<=m1<29", "HEAVY  m1>=29",
                       "HEAVY & loud (SNR>18)"] if r in D]
CQ, CM, CS = "#d1495b", "#2a9d8f", "#9aa0a6"

fig, axes = plt.subplots(1, len(REGIMES), figsize=(4.4 * len(REGIMES), 4.8),
                         sharey=True)
axes = np.atleast_1d(axes)
x = np.arange(len(MODELS))
for ax, reg in zip(axes, REGIMES):
    d = D[reg]
    q = [d["q"][m] for m in MODELS]
    mc = [d["log Mc [ctrl]"][m] for m in MODELS]
    sh = [d["q shuffled"][m] for m in MODELS]
    ax.bar(x - 0.22, q, 0.44, color=CQ, label="q  (the question)",
           edgecolor="white", zorder=3)
    ax.plot(x, mc, "o--", color=CM, lw=2, ms=6, zorder=4,
            label="log Mc  (control: well represented)")
    ax.plot(x, sh, "s:", color=CS, lw=2, ms=5, zorder=4,
            label="q shuffled  (noise floor)")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=25, ha="right")
    ax.set_title(reg.strip(), fontsize=11)
    ax.grid(alpha=0.25, axis="y", zorder=0)
axes[0].set_ylabel("held-out R²  (fit on NPE-train split, tested on val split)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("M18 — linear vs non-linear probes on the FROZEN encoder context: "
             "does extra expressivity recover the mass ratio?", y=1.02, fontsize=13)
fig.tight_layout()
(HERE / "figs").mkdir(exist_ok=True)
fig.savefig(HERE / "figs/m18_nonlinear_probe.png", dpi=130, bbox_inches="tight")
print("wrote figs/m18_nonlinear_probe.png")
