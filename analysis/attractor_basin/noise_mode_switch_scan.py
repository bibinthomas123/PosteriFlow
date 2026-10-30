"""Does mf_scratch's q-answer at GW150914's exact true parameters depend on
the specific noise realization? 100 synthetic twins, same source params
(GW150914_TRUTH: m1 38.8, m2 33.4, q 0.861, true ra/dec so basin-cell
geometry), independent Gaussian design-noise draws per twin (twin_test.py's
make_twin, seed 0..99). Report + plot posterior median q vs noise seed.

Consistent bias (q_med clustered near one value, e.g. all ~0.3) => ordinary
systematic bias. Wide bimodal/scattered spread (some twins near 0.3, others
near 1.0) => genuine noise-induced mode-switching, i.e. the posterior's
answer is unstable under the specific noise draw, not just biased.
"""
import sys
from pathlib import Path

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))

import numpy as np

from twin_test import make_twin
import _common as C

CKPT = str(C.ROOT / "model" / "lean_npe_mf_scratch" / "best_model.pth")
N_TWINS = 100
N_SAMP = 2000


def main():
    m = C.load_model(CKPT)
    print(f"loaded: epoch {m._ckpt_meta['epoch']} val_nll {m._ckpt_meta['val_nll']:.3f}")
    truth = dict(C.GW150914_TRUTH)
    zeros_ab = np.zeros((3, 16), dtype=np.float32)

    q_meds = np.zeros(N_TWINS)
    for seed in range(N_TWINS):
        prep, snr = make_twin(truth, seed)
        s = C.sample_from_strain(m, prep.strain, asd_bands_np=zeros_ab, n=N_SAMP, seed=0)
        m1, m2 = s[:, 0], s[:, 1]
        q = np.minimum(m1, m2) / np.maximum(m1, m2)
        q_meds[seed] = np.median(q)
        print(f"seed {seed:3d}  SNR {snr['network']['optimal_snr']:5.1f}  q_med {q_meds[seed]:.3f}")

    np.savez(SCRATCH / "noise_mode_switch_results.npz", q_meds=q_meds)

    print(f"\nn={N_TWINS}  mean {q_meds.mean():.3f}  std {q_meds.std():.3f}  "
          f"min {q_meds.min():.3f}  max {q_meds.max():.3f}")
    print(f"frac q_med>0.9: {(q_meds > 0.9).mean():.2f}   "
          f"frac q_med<0.5: {(q_meds < 0.5).mean():.2f}   "
          f"frac in [0.5,0.9]: {((q_meds >= 0.5) & (q_meds <= 0.9)).mean():.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.scatter(np.arange(N_TWINS), q_meds, s=18, alpha=0.75)
    ax.axhline(truth_q := (min(truth["mass_1"], truth["mass_2"]) /
                          max(truth["mass_1"], truth["mass_2"])),
               color="green", ls="--", label=f"true q={truth_q:.3f}")
    ax.axhline(1.0, color="red", ls=":", alpha=0.6, label="q=1 (collapse)")
    ax.set_xlabel("noise realization (seed)")
    ax.set_ylabel("posterior median q")
    ax.set_title(f"mf_scratch: q_med across {N_TWINS} independent noise draws\n"
                 f"(same source params = GW150914 true parameters)")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1.05)

    ax2 = axes[1]
    ax2.hist(q_meds, bins=25, range=(0, 1), color="steelblue", edgecolor="k", alpha=0.8)
    ax2.axvline(truth_q, color="green", ls="--", label=f"true q={truth_q:.3f}")
    ax2.set_xlabel("posterior median q")
    ax2.set_ylabel("count")
    ax2.set_title("distribution of q_med over noise realizations")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    outpath = SCRATCH / "noise_mode_switch_scan.png"
    fig.savefig(outpath, dpi=140)
    print(f"\nsaved plot -> {outpath}")


if __name__ == "__main__":
    main()
