"""M12 - Does the q->1 attractor fire on SIMULATED heavy+equal-mass+loud events?

m11 falsified the asd_bands trigger: swapping sensitivity leaves the collapse
untouched (q_med 0.98-0.997, m1~m2~30 in every condition). So the attractor
lives in the strain->mass mapping. m7 measured that the heavy x near-equal-mass
x high-SNR corner already has base_conc 3.71 IN-DISTRIBUTION, while m9's clean
frac(q>0.95)=1.5% was an aggregate over all events -- which dilutes a corner
holding only a few hundred.

This asks the decisive question on SIMULATED data (truth known, no OOD):
  - do corner events collapse to q->1 the way GW150914/GW170814 do?
  - does predicted q track true q, or snap to a fixed point regardless?
  - what do GW150914-TWINS (Mc~31, q~0.86, SNR~24) recover?

  collapse on sim  -> in-distribution representation failure -> reparam earns it
  clean on sim     -> real-signal-specific (waveform systematics), reparam won't help
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from ahsd.inference.pipeline import load_model  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL = "model/lean_npe/best_model.pth"
N, K = 5000, 200


def chirp(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


@torch.no_grad()
def collect(net, dev):
    ds = RemixDataset("data/dataset/memmap/validation", remix=False, seed=1234,
                      real_noise_dir="data/noise_bank", real_noise_prob=1.0,
                      return_asd_bands=net.psd_cond)
    n = min(N, len(ds))
    T = np.empty((n, 11)); P = np.empty((n, 11)); snr = np.empty(n)
    qpred = np.empty(n); qg95 = np.empty(n); rail = np.empty(n)
    nc = (~net.scaler.circ_mask)
    B = 128
    for s in range(0, n, B):
        idx = list(range(s, min(s + B, n)))
        st, tr, ab, sv = [], [], [], []
        for i in idx:
            it = ds[i]
            if net.psd_cond:
                a, pv, ng, x, band = it; ab.append(band.numpy())
            else:
                a, pv, ng, x = it
            st.append(a.numpy()); tr.append(pv.numpy()[0]); sv.append(float(x))
        stt = torch.from_numpy(np.stack(st)).to(dev).float()
        abt = torch.from_numpy(np.stack(ab)).to(dev).float() if net.psd_cond else None
        ctx = net._full_context(net.encode(stt, abt),
                                torch.zeros(len(idx), dtype=torch.long, device=dev))
        rep = ctx.unsqueeze(1).expand(len(idx), K, ctx.shape[1]).reshape(-1, ctx.shape[1])
        z = torch.randn(len(idx) * K, 11, device=dev)
        y, _ = net.flow.inverse(z, rep)
        yw = net.scaler.wrap(y)
        rb = (yw.abs() > 0.999)[:, nc].any(1).float().reshape(len(idx), K).mean(1).cpu().numpy()
        smp = net.scaler.denormalize(yw).reshape(len(idx), K, 11).cpu().numpy()
        a_ = np.maximum(smp[:, :, 0], smp[:, :, 1]); b_ = np.minimum(smp[:, :, 0], smp[:, :, 1])
        smp[:, :, 0], smp[:, :, 1] = a_, b_
        qq = smp[:, :, 1] / smp[:, :, 0]
        for jj, i in enumerate(idx):
            T[i] = tr[jj]; P[i] = np.median(smp[jj], 0); snr[i] = sv[jj]
            qpred[i] = np.median(qq[jj]); qg95[i] = float((qq[jj] > 0.95).mean()); rail[i] = rb[jj]
    return T, P, snr, qpred, qg95, rail


def main():
    net, meta = load_model(MODEL)
    dev = meta["device"]
    print(f"model epoch {meta['model_epoch']} val_nll {meta['model_val_nll']:.3f} "
          f"(real-noise domain, n={N})\n")
    T, P, snr, qpred, qg95, rail = collect(net, dev)
    Mc = chirp(T[:, 0], T[:, 1]); qt = T[:, 1] / T[:, 0]

    corner = (Mc >= 20) & (qt >= 0.7) & (snr > 18)
    print("=== corner (Mc>=20, true q>=0.7, SNR>18) vs rest ===")
    print(f"{'group':28} {'n':>5} {'true q':>7} {'pred q':>7} {'q>.95':>6} {'rail':>6} "
          f"{'m1 true':>8} {'m1 pred':>8}")
    for lab, m in [("CORNER heavy+equal+loud", corner), ("all others", ~corner)]:
        print(f"{lab:28} {int(m.sum()):5d} {np.median(qt[m]):7.3f} {np.median(qpred[m]):7.3f} "
              f"{qg95[m].mean():6.2f} {rail[m].mean():6.2f} "
              f"{np.median(T[m,0]):8.1f} {np.median(P[m,0]):8.1f}")

    # does predicted q track true q inside the corner? (attractor => flat at ~1)
    print("\n=== inside corner: does pred q track TRUE q? (attractor => flat) ===")
    print(f"{'true q bin':14} {'n':>5} {'true q':>7} {'pred q':>7} {'q>.95':>6} {'rail':>6}")
    for lo, hi in [(0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        m = corner & (qt >= lo) & (qt < hi)
        if m.sum() >= 5:
            print(f"[{lo:.2f},{hi:.2f})   {int(m.sum()):5d} {np.median(qt[m]):7.3f} "
                  f"{np.median(qpred[m]):7.3f} {qg95[m].mean():6.2f} {rail[m].mean():6.2f}")
    # control: heavy+loud but UNEQUAL mass
    ctl = (Mc >= 20) & (qt < 0.7) & (snr > 18)
    print(f"{'CTL q<0.7':14} {int(ctl.sum()):5d} {np.median(qt[ctl]):7.3f} "
          f"{np.median(qpred[ctl]):7.3f} {qg95[ctl].mean():6.2f} {rail[ctl].mean():6.2f}")

    # GW150914 twins on simulation
    twin = (Mc > 28) & (Mc < 34) & (qt > 0.75) & (snr > 18) & (snr < 30)
    print(f"\n=== SIMULATED GW150914 TWINS (Mc 28-34, q>0.75, SNR 18-30) n={int(twin.sum())} ===")
    if twin.sum() >= 3:
        print(f"  true : m1 {np.median(T[twin,0]):5.1f}  m2 {np.median(T[twin,1]):5.1f}  "
              f"q {np.median(qt[twin]):.3f}")
        print(f"  pred : m1 {np.median(P[twin,0]):5.1f}  m2 {np.median(P[twin,1]):5.1f}  "
              f"q {np.median(qpred[twin]):.3f}  frac(q>.95) {qg95[twin].mean():.2f}  "
              f"rail {rail[twin].mean():.2f}")
        print(f"  (real GW150914: pred m1 30.2 m2 30.0 q 0.996 rail 0.69)")
    json.dump({"corner_n": int(corner.sum()),
               "corner_pred_q_med": float(np.median(qpred[corner])),
               "corner_true_q_med": float(np.median(qt[corner])),
               "corner_qgt95": float(qg95[corner].mean()),
               "corner_rail": float(rail[corner].mean()),
               "others_pred_q_med": float(np.median(qpred[~corner])),
               "others_qgt95": float(qg95[~corner].mean()),
               "twin_n": int(twin.sum()),
               "twin_pred_q_med": float(np.median(qpred[twin])) if twin.sum() else None,
               "twin_pred_m1_med": float(np.median(P[twin, 0])) if twin.sum() else None},
              open(HERE / "m12_corner_q_recovery.json", "w"), indent=2)


if __name__ == "__main__":
    main()
