"""M21 - Compare the actual CLOUDS: NPE vs dynesty in (m1, m2) for GW150914.

m20 showed the flow's marginals carry q information that its per-sample joint
discards (E[m2]/E[m1] R^2 0.431 vs E[m2/m1] 0.252 on heavy+loud), and its
within-posterior corr(m1,m2) is -0.61. But negative correlation is PHYSICALLY
EXPECTED along constant Mc, and with no reference posterior we could not say
whether -0.61 is wrong or in which direction. Dynesty samples for GW150914
already exist (results/gw150914_dynesty_check), so the reference is free.

Distinguishes two very different bugs:
  WRONG BANANA    : the flow's cloud has the wrong shape/orientation/curvature
                    -> the joint is mismodelled
  CORRECT BANANA  : the cloud matches dynesty but the summary (median/mean) is
    + WRONG        read off badly -- e.g. a second spurious mode drags it
    ESTIMATOR      -> the joint is fine; the defect is elsewhere

Measured per cloud, in (m1,m2) and in (log Mc, q):
  covariance matrix, principal axes, eigenvalue ratio, orientation angle,
  curvature of the principal curve, and spread along/across constant Mc.

CAUTION: the NPE posterior on GW150914 is known to be BIMODAL (~70% of samples
at q>0.95 near m1~m2~30, yet the 90% CI still contains the truth). A covariance
/PCA over a bimodal cloud describes the GAP between the modes, not either mode.
So every statistic is reported globally AND per-mode, and the clouds are plotted
rather than only summarised.

Nothing is trained or modified; the NPE runs forward on real GW150914 strain.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from ahsd.inference.pipeline import load_model  # noqa: E402
from ahsd.inference.preprocessing import prepare_real, fetch_gwosc, compute_asd_bands  # noqa: E402

HERE = Path(__file__).resolve().parent
DYN = ROOT / "results/gw150914_dynesty_check/GW150914/bilby/GW150914_result.json"
GW150914_GPS = 1126259462.4
K = 20000
PUB_M1, PUB_M2 = 38.8, 33.4          # published detector-frame medians


def chirp(m1, m2):
    """Chirp mass from component masses."""
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


@torch.no_grad()
def npe_samples(net, dev):
    """Sample the current model's posterior on real GW150914 strain."""
    gps, raw, starts, srs, found = fetch_gwosc(GW150914_GPS)
    pr = prepare_real(raw, gps, starts, srs, seed=0)
    ab = compute_asd_bands(pr, psd_bands=16)
    st = torch.from_numpy(np.asarray(pr.strain, np.float32)).unsqueeze(0).to(dev)
    abt = torch.from_numpy(np.asarray(ab, np.float32)).unsqueeze(0).to(dev)
    ctx = net._full_context(net.encode(st.float(), abt.float()),
                            torch.zeros(1, dtype=torch.long, device=dev))
    y, _ = net.flow.inverse(torch.randn(K, 11, device=dev), ctx.expand(K, -1))
    s = net.scaler.denormalize(net.scaler.wrap(y)).cpu().numpy()
    m1 = np.maximum(s[:, 0], s[:, 1]); m2 = np.minimum(s[:, 0], s[:, 1])
    return m1, m2


def dynesty_samples():
    """(m1, m2) from the stored bilby/dynesty posterior."""
    c = json.load(open(DYN))["posterior"]["content"]
    m1, m2 = np.asarray(c["mass_1"], float), np.asarray(c["mass_2"], float)
    hi, lo = np.maximum(m1, m2), np.minimum(m1, m2)
    return hi, lo


def shape(x, y):
    """Covariance, principal axes, eigen-ratio, orientation, curvature.

    Curvature is fit in the cloud's OWN principal frame: project onto the major
    axis (t) and minor axis (s), fit s = a + b t + c t^2. `curv` = 2c is the raw
    second derivative; `curv_dimless` rescales by the cloud's own extent
    (2c * sd(t)^2 / sd(s)), i.e. how far it bends relative to its own width, so
    clouds of different sizes are comparable.
    """
    X = np.column_stack([x, y])
    mu = X.mean(0)
    C = np.cov(X.T)
    w, V = np.linalg.eigh(C)                 # ascending
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    ang = float(np.degrees(np.arctan2(V[1, 0], V[0, 0])))
    ang = (ang + 180) % 180                  # axis, not direction
    t = (X - mu) @ V[:, 0]
    s = (X - mu) @ V[:, 1]
    c2 = float(np.polyfit(t, s, 2)[0])
    return dict(mean=mu.tolist(), cov=C.tolist(),
                eig_major=float(w[0]), eig_minor=float(w[1]),
                eig_ratio=float(w[0] / max(w[1], 1e-12)),
                sd_major=float(np.sqrt(w[0])), sd_minor=float(np.sqrt(w[1])),
                angle_deg=ang, corr=float(C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])),
                curv=2 * c2,
                curv_dimless=float(2 * c2 * t.std() ** 2 / (s.std() + 1e-12)))


def report(tag, m1, m2, out):
    """Print + store shape stats in (m1,m2) and in (log Mc, q)."""
    mc, q = np.log(chirp(m1, m2)), m2 / m1
    a, b = shape(m1, m2), shape(mc, q)
    out[tag] = {"n": int(len(m1)), "m1_m2": a, "logMc_q": b,
                "med_m1": float(np.median(m1)), "med_m2": float(np.median(m2)),
                "med_q": float(np.median(q)), "med_Mc": float(np.exp(np.median(mc))),
                "sd_logMc": float(mc.std()), "sd_q": float(q.std()),
                "frac_q_gt_0.95": float((q > 0.95).mean())}
    print(f"\n{tag}  (n={len(m1)})")
    print(f"  medians      m1 {np.median(m1):6.2f}  m2 {np.median(m2):6.2f}  "
          f"Mc {np.exp(np.median(mc)):6.2f}  q {np.median(q):5.3f}   "
          f"frac(q>0.95) {(q > 0.95).mean():.3f}")
    print(f"  (m1,m2)      corr {a['corr']:+.3f}   sd_major {a['sd_major']:6.2f}  "
          f"sd_minor {a['sd_minor']:5.2f}   ratio {a['eig_ratio']:8.2f}   "
          f"angle {a['angle_deg']:6.1f} deg   curv_dimless {a['curv_dimless']:+7.3f}")
    print(f"  (logMc,q)    corr {b['corr']:+.3f}   sd_logMc {mc.std():.4f}  "
          f"sd_q {q.std():.4f}   ratio {b['eig_ratio']:8.2f}   "
          f"angle {b['angle_deg']:6.1f} deg   curv_dimless {b['curv_dimless']:+7.3f}")


def main():
    net, meta = load_model("model/lean_npe/best_model.pth"); dev = meta["device"]
    print(f"model epoch {meta['model_epoch']}   GW150914   NPE K={K}, "
          f"dynesty = stored reference posterior")
    dm1, dm2 = dynesty_samples()
    nm1, nm2 = npe_samples(net, dev)

    out = {}
    report("DYNESTY (reference)", dm1, dm2, out)
    report("NPE (all samples)", nm1, nm2, out)

    # The NPE cloud is bimodal; a global covariance describes neither mode.
    q = nm2 / nm1
    spur = q > 0.95
    print(f"\n  -- NPE split by mode: {spur.mean():.1%} of samples have q>0.95 --")
    if spur.sum() > 50:
        report("NPE spurious lobe (q>0.95)", nm1[spur], nm2[spur], out)
    if (~spur).sum() > 50:
        report("NPE physical lobe (q<=0.95)", nm1[~spur], nm2[~spur], out)

    print(f"\n  truth (published): m1 {PUB_M1}  m2 {PUB_M2}  "
          f"Mc {chirp(PUB_M1, PUB_M2):.2f}  q {PUB_M2/PUB_M1:.3f}")
    print("\n  wrong banana   -> NPE physical lobe disagrees with dynesty on "
          "angle / ratio / curvature")
    print("  correct banana -> physical lobe MATCHES dynesty; the defect is the "
          "spurious lobe + the summary statistic")
    json.dump(out, open(HERE / "m21_cloud_compare.json", "w"), indent=2, default=float)

    # ── plot ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    ax = axes[0]
    ax.scatter(nm1, nm2, s=3, alpha=0.10, color="#d1495b", label="NPE", rasterized=True)
    ax.scatter(dm1, dm2, s=9, alpha=0.65, color="#2a9d8f", label="dynesty (reference)")
    ax.scatter([PUB_M1], [PUB_M2], marker="*", s=420, color="black",
               edgecolor="white", zorder=6, label="published truth")
    # constant-Mc curve through the truth: the physical degeneracy direction
    mc0 = chirp(PUB_M1, PUB_M2)
    grid = np.linspace(28, 90, 400)
    m2c = []
    for a_ in grid:                      # solve chirp(a_, m2)=mc0 for m2
        lo, hi = 1.0, a_
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if chirp(a_, mid) < mc0:
                lo = mid
            else:
                hi = mid
        m2c.append(0.5 * (lo + hi))
    m2c = np.array(m2c)
    ok = (m2c > 5) & (m2c <= grid)
    ax.plot(grid[ok], m2c[ok], "k--", lw=1.6, alpha=0.8,
            label=f"constant Mc = {mc0:.1f} (physical degeneracy)")
    ax.plot([20, 90], [20, 90], ":", color="#888", lw=1.2, label="q = 1")
    ax.set_xlabel("m1 [Msun]"); ax.set_ylabel("m2 [Msun]")
    ax.set_xlim(25, 80); ax.set_ylim(15, 60)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)
    ax.set_title("GW150914 — (m1, m2) clouds")

    ax = axes[1]
    ax.scatter(chirp(nm1, nm2), nm2 / nm1, s=3, alpha=0.10, color="#d1495b",
               label="NPE", rasterized=True)
    ax.scatter(chirp(dm1, dm2), dm2 / dm1, s=9, alpha=0.65, color="#2a9d8f",
               label="dynesty")
    ax.scatter([mc0], [PUB_M2 / PUB_M1], marker="*", s=420, color="black",
               edgecolor="white", zorder=6, label="truth")
    ax.axhline(0.95, color="#d1495b", ls=":", lw=1.5, alpha=0.7,
               label="q = 0.95 (attractor)")
    ax.set_xlabel("chirp mass Mc [Msun]"); ax.set_ylabel("mass ratio q")
    ax.set_xlim(20, 45); ax.set_ylim(0.2, 1.02)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25)
    ax.set_title("GW150914 — same clouds in (Mc, q)")
    fig.suptitle("M21 — is the banana WRONG, or is the banana RIGHT and the "
                 "estimator wrong?", y=1.00, fontsize=13)
    fig.tight_layout()
    (HERE / "figs").mkdir(exist_ok=True)
    fig.savefig(HERE / "figs/m21_cloud_compare.png", dpi=130, bbox_inches="tight")
    print("wrote figs/m21_cloud_compare.png")


if __name__ == "__main__":
    main()
