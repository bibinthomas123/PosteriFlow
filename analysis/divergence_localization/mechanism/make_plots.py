import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)

h = json.load(open(ROOT / "model/long_run/history.json"))
ep = [e["epoch"] for e in h]
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
# (a) optimizer/encoder trajectory
ax[0].plot(ep, [e["val_nll"] for e in h], "o-", color="#3b6fd4", label="val NLL")
ax[0].plot(ep, [e["train_nll"] for e in h], "s--", color="#8fb0e8", label="train NLL")
ax[0].set_ylabel("NLL"); ax[0].set_xlabel("epoch"); ax[0].legend(loc="upper right")
ax0b = ax[0].twinx()
ax0b.plot(ep, [e["shuffle_delta_nll"] for e in h], "^-", color="#e08a1e", label="shuffle Δ")
ax0b.plot(ep, [e["dist_corr"] for e in h], "d-", color="#2a9d8f", label="dist corr")
ax0b.set_ylabel("encoder metrics"); ax0b.legend(loc="lower right")
ax[0].axvspan(17, 19, color="red", alpha=0.08)
ax[0].set_title("Encoder metrics plateau (~ep16-18) while val NLL keeps falling")
# (b) grad norm + lr
ax[1].plot(ep, [e["grad_norm"] for e in h], "o-", color="#d1495b", label="grad norm")
ax[1].set_ylabel("grad norm"); ax[1].set_xlabel("epoch"); ax[1].legend(loc="upper left")
ax1b = ax[1].twinx()
ax1b.plot(ep, [e["lr"] for e in h], "s--", color="#6a4c93", label="lr")
ax1b.set_ylabel("lr"); ax1b.legend(loc="lower right")
ax[1].set_title("grad norm rises late (32→42), no explosion; lr near-peak\n(cosine sized for 140 ep)")
# (c) base-space concentration vs SNR (M3)
m3 = json.load(open(HERE / "m3_flow_internals.json"))
for name, c in [("long_run(ep19)", "#d1495b"), ("coherentC(ep40)", "#3b6fd4")]:
    ss = m3[name]["snr_split_base_conc"]
    ax[2].plot(list(ss.keys()), list(ss.values()), "o-", color=c, label=name)
ax[2].axhline(1.0, color="gray", ls=":", label="calibrated (=1)")
ax[2].set_ylabel("||z_truth||²/D  (base concentration)"); ax[2].set_xlabel("SNR bin")
ax[2].set_title("Overconfidence grows with SNR (truths pushed to base tails)")
ax[2].legend()
fig.suptitle("Mechanism: late-training flow overconfidence — encoder converged, flow keeps sharpening", y=1.02)
fig.tight_layout(); fig.savefig(FIGS / "mechanism_overview.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# (d) selection inversion
m5 = json.load(open(HERE / "m5_selection.json"))
names = list(m5["val_nll"])
vnll = [m5["val_nll"][n] for n in names]
score = [m5["calibration"][n]["railing"] + m5["calibration"][n]["cov90_err"]
         + (1 - m5["calibration"][n]["sbc_pass"]) for n in names]
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(vnll, score, s=120, color="#d1495b", zorder=5)
for n, x, y in zip(names, vnll, score):
    ax.annotate(n, (x, y), fontsize=9, xytext=(6, 4), textcoords="offset points")
ax.plot(vnll, score, "--", color="#999", zorder=1)
ax.set_xlabel("val NLL (selection metric — lower is 'better')")
ax.set_ylabel("miscalibration score (railing + |cov-.9| + SBC-fail; lower=better)")
ax.set_title("INVERSION: lower val NLL ⇒ worse calibration\ncurrent criterion picks the leftmost (most overconfident) point")
fig.tight_layout(); fig.savefig(FIGS / "selection_inversion.png", dpi=130)
plt.close(fig)
print("wrote mechanism_overview.png, selection_inversion.png")
