"""M11 - Causal intervention: is asd_bands the TRIGGER of the q->1 attractor?

Hypothesis under test: GW150914/GW170814 fall into a spurious q->1 + railed
spin/inclination mode because their PSD/sensitivity context (asd_bands) is OOD
(they are typical O1 events; the bank is dominated by the far more sensitive
O3b/O4, so training asd_bands center at -0.49 while GW150914 sits at -0.88).

Intervention: hold the REAL strain fixed and swap ONLY asd_bands for values from
the training bulk. This is deliberately physics-INCONSISTENT (O1 strain paired
with O3b-like sensitivity) -- that is the point: it isolates the causal role of
the feature. So we read the ATTRACTOR indicators (railing, q collapse), NOT
whether m1 becomes correct.

  attractor switches off -> asd_bands OOD is the trigger  -> era-rebalance is worth it
  attractor persists     -> asd_bands is NOT the trigger  -> rebalance is wasted compute
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from ahsd.inference.pipeline import load_model  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

HERE = Path(__file__).resolve().parent
EVDIR = HERE.parent / "events_ep19"
MODEL = "model/lean_npe/best_model.pth"
K = 3000


@torch.no_grad()
def probe(net, dev, strain, ab):
    st = torch.from_numpy(strain).unsqueeze(0).to(dev).float()
    abt = torch.from_numpy(np.asarray(ab, dtype=np.float32)).unsqueeze(0).to(dev).float()
    ctx = net._full_context(net.encode(st, abt),
                            torch.zeros(1, dtype=torch.long, device=dev))
    rep = ctx.expand(K, -1)
    z = torch.randn(K, 11, device=dev)
    y, _ = net.flow.inverse(z, rep)
    yw = net.scaler.wrap(y)
    nc = ~net.scaler.circ_mask
    rail = float((yw.abs() > 0.999)[:, nc].any(1).float().mean())
    s = net.scaler.denormalize(yw).cpu().numpy()
    m1 = np.maximum(s[:, 0], s[:, 1]); m2 = np.minimum(s[:, 0], s[:, 1])
    q = m2 / m1
    return dict(rail=rail, q_med=float(np.median(q)),
                q_gt95=float((q > 0.95).mean()), m1=float(np.median(m1)),
                m2=float(np.median(m2)), dL=float(np.median(s[:, 2])))


def main():
    net, meta = load_model(MODEL)
    dev = meta["device"]
    print(f"model epoch {meta['model_epoch']} val_nll {meta['model_val_nll']:.3f}\n")

    # training-bulk asd_bands (what the model normally sees)
    ds = RemixDataset("data/dataset/memmap/validation", remix=False, seed=1234,
                      real_noise_dir="data/noise_bank", real_noise_prob=1.0,
                      return_asd_bands=True, psd_bands=16)
    tr = np.stack([ds[i][4].numpy() for i in range(300)])
    ab_mean = tr.mean(0)
    ab_draw = tr[0]
    print(f"training asd_bands: mean {ab_mean.mean():+.3f} | one draw {ab_draw.mean():+.3f}\n")

    for ev, pub in [("GW150914", (38.8, 33.4, 440.0)), ("GW170814", (30.7, 25.3, 600.0))]:
        strain = np.load(EVDIR / ev / "strain.npy")
        ab_real = np.load(EVDIR / ev / "asd_bands.npy")
        print(f"=== {ev}  (published m1 {pub[0]}, m2 {pub[1]}, dL {pub[2]}) ===")
        print(f"{'asd_bands variant':22} {'mean':>7} {'rail':>6} {'q_med':>6} "
              f"{'q>.95':>6} {'m1':>6} {'m2':>6} {'dL':>6}")
        for tag, ab in [("REAL (baseline)", ab_real),
                        ("train MEAN", ab_mean),
                        ("train DRAW (bulk)", ab_draw),
                        ("zeros (design)", np.zeros_like(ab_real))]:
            r = probe(net, dev, strain, ab)
            print(f"{tag:22} {np.mean(ab):+7.3f} {r['rail']:6.2f} {r['q_med']:6.3f} "
                  f"{r['q_gt95']:6.2f} {r['m1']:6.1f} {r['m2']:6.1f} {r['dL']:6.0f}")
        print()
    print("NOTE: swapped asd_bands are physics-inconsistent by design (O1 strain +\n"
          "O3b-like sensitivity), so judge the ATTRACTOR (rail / q>.95), not accuracy.")


if __name__ == "__main__":
    main()
