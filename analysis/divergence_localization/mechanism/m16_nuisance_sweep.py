"""M16 - Fixed intrinsic system; vary ONE nuisance parameter at a time.

m15 found the attractor is parameter-triggered (not data-domain) and that
EDGE-ON inclination restores q while face-on/face-off collapse it. This runs the
clean version: one fixed intrinsic system (m1=38.8, m2=33.4, q=0.861), sweep each
nuisance axis (spin magnitude, inclination, polarization, phase), average over
several noise realizations to kill single-draw scatter, and plot predicted q.

All simulated: design Gaussian noise, asd_bands=0, SNR rescaled to 24 on every
point so the comparison is not confounded by loudness (m7: behaviour is strongly
SNR-dependent).
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from ahsd.data.bilby_pipeline import (BilbyWaveformGenerator, BilbyPreprocessor,
                                      get_default_psd, _GPS_REF)  # noqa: E402
from ahsd.inference.pipeline import load_model  # noqa: E402

HERE = Path(__file__).resolve().parent
SR, DUR, T_LEN, K = 4096, 4.0, 16384, 1500
TARGET_SNR, N_NOISE = 24.0, 5
TRUE_Q = 33.4 / 38.8
BASE = dict(mass_1=38.8, mass_2=33.4, luminosity_distance=440.0,
            ra=1.95, dec=-1.27, theta_jn=0.4, psi=1.75, phase=1.3,
            geocent_time=0.0, geocent_time_gps=_GPS_REF,
            a1=0.0, a2=0.0, tilt1=0.0, tilt2=0.0, event_type="BBH")


@torch.no_grad()
def qhat(net, dev, wfg, pre, params, rng, n_noise=N_NOISE):
    """Median predicted q (and m1), averaged over independent noise draws."""
    sig = {d: pre.preprocess(wfg.generate(params, d).astype(np.float64),
                             get_default_psd(d), detector=d).astype(np.float32)
           for d in ("H1", "L1", "V1")}
    s = TARGET_SNR / max(float(np.sqrt(sum((sig[d] ** 2).sum() for d in sig))), 1e-9)
    qs, m1s = [], []
    for _ in range(n_noise):
        strain = np.stack([rng.standard_normal(T_LEN).astype(np.float32) + s * sig[d]
                           for d in ("H1", "L1", "V1")])
        st = torch.from_numpy(strain).unsqueeze(0).to(dev).float()
        ctx = net._full_context(net.encode(st, torch.zeros(1, 3, 16, device=dev)),
                                torch.zeros(1, dtype=torch.long, device=dev))
        y, _ = net.flow.inverse(torch.randn(K, 11, device=dev), ctx.expand(K, -1))
        smp = net.scaler.denormalize(net.scaler.wrap(y)).cpu().numpy()
        a_ = np.maximum(smp[:, 0], smp[:, 1]); b_ = np.minimum(smp[:, 0], smp[:, 1])
        qs.append(float(np.median(b_ / a_))); m1s.append(float(np.median(a_)))
    return float(np.mean(qs)), float(np.std(qs)), float(np.mean(m1s))


def main():
    net, meta = load_model("model/lean_npe/best_model.pth"); dev = meta["device"]
    wfg, pre = BilbyWaveformGenerator(SR, DUR), BilbyPreprocessor(SR, DUR)
    rng = np.random.default_rng(0)
    print(f"fixed intrinsic: m1 38.8  m2 33.4  (TRUE q {TRUE_Q:.3f})  |  design "
          f"Gaussian noise, SNR {TARGET_SNR:.0f}, {N_NOISE} noise draws/point\n")

    sweeps = {
        "spin magnitude (a1=a2)": ("a1", np.linspace(0.0, 0.9, 10)),
        "inclination theta_jn":   ("theta_jn", np.linspace(0.0, np.pi, 13)),
        "polarization psi":       ("psi", np.linspace(0.0, np.pi, 9)),
        "phase":                  ("phase", np.linspace(0.0, 2 * np.pi, 9)),
    }
    out = {}
    for title, (key, grid) in sweeps.items():
        print(f"--- {title} ---")
        print(f"{'value':>8} {'pred q':>8} {'+/-':>6} {'m1':>7}")
        xs, qm, qe, ms = [], [], [], []
        for v in grid:
            p = dict(BASE)
            if key == "a1":
                p["a1"] = p["a2"] = float(v)
            else:
                p[key] = float(v)
            q, e, m1 = qhat(net, dev, wfg, pre, p, rng)
            xs.append(float(v)); qm.append(q); qe.append(e); ms.append(m1)
            print(f"{v:8.3f} {q:8.3f} {e:6.3f} {m1:7.1f}")
        out[title] = dict(x=xs, q=qm, q_err=qe, m1=ms)
        print()
    json.dump({"true_q": TRUE_Q, "sweeps": out},
              open(HERE / "m16_nuisance_sweep.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), sharey=True)
    for ax, (title, d) in zip(axes, out.items()):
        ax.errorbar(d["x"], d["q"], yerr=d["q_err"], fmt="o-", color="#d1495b",
                    lw=2, capsize=3, label="predicted q")
        ax.axhline(TRUE_Q, color="#2a9d8f", ls="--", lw=2, label=f"TRUE q = {TRUE_Q:.2f}")
        ax.axhspan(0.95, 1.0, color="#d1495b", alpha=0.10)
        if "theta_jn" in title:
            ax.axvline(np.pi / 2, color="#333", ls=":", lw=1.5)
            ax.annotate("edge-on", (np.pi / 2, 0.35), fontsize=9, rotation=90,
                        ha="right", color="#333")
        ax.set_xlabel(title); ax.grid(alpha=0.25); ax.set_ylim(0.25, 1.03)
    axes[0].set_ylabel("predicted mass ratio q")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("M16 — fixed intrinsic system (m1=38.8, m2=33.4, q=0.861), SNR 24, "
                 "design Gaussian noise: only INCLINATION moves the collapse "
                 "(shaded = q→1 attractor)", y=1.04)
    fig.tight_layout()
    fig.savefig(HERE / "figs/m16_nuisance_sweep.png", dpi=130, bbox_inches="tight")
    print("wrote figs/m16_nuisance_sweep.png")


if __name__ == "__main__":
    main()
