"""M22 - Is the q->1 spike a real mode, or an artifact of the m1>=m2 FOLD?

m21: on GW150914 the NPE's dominant mode (68% of samples) is a near-1-D spike
lying ON the q=1 diagonal -- corr(m1,m2)=+0.996, eigen-ratio 548, sd_q=0.0036 --
at Mc 26.2, while dynesty is a clean vertical stripe at Mc 32.5 with q spread
0.55-1.0.

The q=1 diagonal is NOT a physical feature. It is the boundary of the training
convention mass_1 >= mass_2, and every analysis (here, m17-m21, validate_check-
point, infer) reconstructs q as min(s0,s1)/max(s0,s1). That FOLD is idempotent
on ordered samples but maps the m1<m2 half-plane onto the m1>m2 half-plane. So a
flow whose density merely straddles the diagonal would be FOLDED into a q~1
pile-up that never existed in its own output.

  raw violation ~0%  -> the flow respects the convention; the spike is a real
                        mode it actually placed on the diagonal
  raw violation ~50% -> the flow learned a near-symmetric blob and ignores the
                        ordering; the q~1 spike is largely manufactured by the
                        fold, and q's apparent collapse is a measurement artifact

Training data always satisfies mass_1 >= mass_2, so the target density lives on
a TRIANGLE (a hard edge along the diagonal). A smooth flow over a box cannot
represent a hard edge exactly, so where its mass goes is the question.

Nothing is trained or modified; forward passes only.
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
TEST = HERE / "m18_cache_validation_4702.npz"
GW150914_GPS = 1126259462.4
K = 20000


def chirp(m1, m2):
    """Chirp mass from component masses."""
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


@torch.no_grad()
def raw_samples(net, dev, ctx_full, k=K):
    """Flow samples in the STORED order (index 0 = mass_1, 1 = mass_2).

    No max/min folding: this is what the flow actually emits.
    """
    y, _ = net.flow.inverse(torch.randn(k, 11, device=dev), ctx_full.expand(k, -1))
    s = net.scaler.denormalize(net.scaler.wrap(y)).cpu().numpy()
    return s[:, 0], s[:, 1]


@torch.no_grad()
def gw150914_ctx(net, dev):
    """Full conditioning context for real GW150914 strain."""
    gps, raw, starts, srs, found = fetch_gwosc(GW150914_GPS)
    pr = prepare_real(raw, gps, starts, srs, seed=0)
    ab = compute_asd_bands(pr, psd_bands=16)
    st = torch.from_numpy(np.asarray(pr.strain, np.float32)).unsqueeze(0).to(dev)
    abt = torch.from_numpy(np.asarray(ab, np.float32)).unsqueeze(0).to(dev)
    return net._full_context(net.encode(st.float(), abt.float()),
                             torch.zeros(1, dtype=torch.long, device=dev))


def main():
    net, meta = load_model("model/lean_npe/best_model.pth"); dev = meta["device"]
    print(f"model epoch {meta['model_epoch']}\n")
    out = {}

    # ── 1. GW150914: does the flow respect mass_1 >= mass_2? ─────────────────
    ctx = gw150914_ctx(net, dev)
    s0, s1 = raw_samples(net, dev, ctx)
    viol = float((s0 < s1).mean())
    print("=" * 74)
    print("GW150914 — RAW flow output (no folding)")
    print("=" * 74)
    print(f"  fraction violating mass_1 >= mass_2 : {viol:.4f}"
          f"   (training data: 0.0000)")
    print(f"  raw  s0 median {np.median(s0):6.2f}   s1 median {np.median(s1):6.2f}")
    print(f"  raw  corr(s0,s1) {np.corrcoef(s0, s1)[0, 1]:+.3f}   "
          f"sd(s0) {s0.std():5.2f}  sd(s1) {s1.std():5.2f}")
    q_raw = np.minimum(s0, s1) / np.maximum(s0, s1)
    ordered = s0 >= s1
    print(f"\n  q from FOLDED samples (what every analysis reports):")
    print(f"    median {np.median(q_raw):.4f}   frac(q>0.95) {(q_raw > 0.95).mean():.4f}")
    if ordered.sum() > 50:
        qo = s1[ordered] / s0[ordered]
        print(f"  q from ONLY already-ordered samples (fold is a no-op on these):")
        print(f"    median {np.median(qo):.4f}   frac(q>0.95) {(qo > 0.95).mean():.4f}"
              f"   n={int(ordered.sum())}")
    if (~ordered).sum() > 50:
        qv = s0[~ordered] / s1[~ordered]
        print(f"  q from the VIOLATING samples (these get mirrored by the fold):")
        print(f"    median {np.median(qv):.4f}   frac(q>0.95) {(qv > 0.95).mean():.4f}"
              f"   n={int((~ordered).sum())}")
    out["GW150914"] = dict(violation_frac=viol, q_folded_med=float(np.median(q_raw)),
                           frac_q_gt95_folded=float((q_raw > 0.95).mean()),
                           corr_s0_s1=float(np.corrcoef(s0, s1)[0, 1]))

    # ── 2. the simulated population: same question, in-distribution ──────────
    z = np.load(TEST)
    C, T, snr = z["C"], z["T"], z["snr"]
    print("\n" + "=" * 74)
    print("Simulated validation population — RAW flow output")
    print("=" * 74)
    print(f"  {'regime':26}{'n':>6}{'violation frac':>16}{'frac q>0.95':>13}")
    groups = [("ALL", np.ones(len(T), bool)),
              ("light  m1<20", T[:, 0] < 20),
              ("mid    20<=m1<29", (T[:, 0] >= 20) & (T[:, 0] < 29)),
              ("HEAVY  m1>=29", T[:, 0] >= 29),
              ("HEAVY & loud (SNR>18)", (T[:, 0] >= 29) & (snr > 18))]
    sub = np.random.default_rng(0).choice(len(C), size=min(600, len(C)), replace=False)
    vio = np.empty(len(sub)); fq = np.empty(len(sub))
    for i, idx in enumerate(sub):
        c = torch.from_numpy(C[idx:idx + 1]).to(dev).float()
        full = net._full_context(c, torch.zeros(1, dtype=torch.long, device=dev))
        a, b = raw_samples(net, dev, full, k=500)
        vio[i] = (a < b).mean()
        qq = np.minimum(a, b) / np.maximum(a, b)
        fq[i] = (qq > 0.95).mean()
    for lab, m in groups:
        sel = m[sub]
        if sel.sum() < 20:
            continue
        print(f"  {lab:26}{int(sel.sum()):>6}{vio[sel].mean():>16.4f}"
              f"{fq[sel].mean():>13.4f}")
        out[lab] = dict(n=int(sel.sum()), violation_frac=float(vio[sel].mean()),
                        frac_q_gt95=float(fq[sel].mean()))

    print("\n  violation ~0.00 -> flow respects m1>=m2; the q~1 spike is a REAL mode")
    print("  violation ~0.50 -> flow ignores the ordering; the fold MANUFACTURES "
          "the spike")
    json.dump(out, open(HERE / "m22_fold_test.json", "w"), indent=2, default=float)

    # ── plot: the RAW cloud vs the diagonal ─────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    ok = s0 >= s1
    ax.scatter(s0[ok], s1[ok], s=3, alpha=0.12, color="#2a9d8f",
               label=f"ordered  m1>=m2  ({ok.mean():.0%})", rasterized=True)
    ax.scatter(s0[~ok], s1[~ok], s=3, alpha=0.12, color="#d1495b",
               label=f"VIOLATING  m1<m2  ({(~ok).mean():.0%})", rasterized=True)
    lim = [0, max(np.percentile(s0, 99.5), np.percentile(s1, 99.5))]
    ax.plot(lim, lim, "k--", lw=1.8, label="m1 = m2  (fold axis / convention edge)")
    ax.set_xlabel("raw sample slot 0  (= mass_1)")
    ax.set_ylabel("raw sample slot 1  (= mass_2)")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_title("GW150914 — what the flow ACTUALLY emits (unfolded)")

    ax = axes[1]
    ax.hist(q_raw, bins=120, color="#d1495b", alpha=0.85)
    ax.axvline(33.4 / 38.8, color="black", ls="--", lw=2, label="published q = 0.861")
    ax.set_xlabel("q after folding = min/max"); ax.set_ylabel("count")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.set_title("q as every analysis computes it (post-fold)")
    fig.suptitle("M22 — is the q→1 attractor a real mode, or the m1≥m2 fold?",
                 y=1.00, fontsize=13)
    fig.tight_layout()
    (HERE / "figs").mkdir(exist_ok=True)
    fig.savefig(HERE / "figs/m22_fold_test.png", dpi=130, bbox_inches="tight")
    print("wrote figs/m22_fold_test.png")


if __name__ == "__main__":
    main()
