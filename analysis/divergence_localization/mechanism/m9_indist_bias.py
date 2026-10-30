"""M9 - Is the (now-calibrated) model BIASED in-distribution, or only on real events?

Calibration certifies self-consistency with the training distribution; it does
not certify accuracy on OOD real data. This measures the in-distribution point-
estimate bias (posterior median vs known truth) on the simulated validation set
for the calibrated ep40 checkpoint. If in-distribution bias ~0 but real events
are off, the real-event error is the sim-to-real (OOD) gap. If in-distribution
bias is also large, the model is biased everywhere and calibration only means
"intervals wide enough to cover".
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402
from ahsd.inference.pipeline import load_model  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

HERE = Path(__file__).resolve().parent
VCACHE = "data/dataset/memmap/validation"
MODEL = "model/lean_npe/best_model.pth"
N, K = 1500, 200
M1, M2, DL = 0, 1, 2


def chirp(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


@torch.no_grad()
def main():
    net, meta = load_model(MODEL)
    dev = meta["device"]
    print(f"model epoch {meta['model_epoch']} val_nll {meta['model_val_nll']:.3f}\n")
    ds = RemixDataset(VCACHE, remix=False, seed=1234, return_asd_bands=net.psd_cond)
    n = min(N, len(ds))
    tru = np.empty((n, 11)); med = np.empty((n, 11)); snr = np.empty(n)
    B = 128
    for s in range(0, n, B):
        idx = list(range(s, min(s + B, n)))
        st, tr, ab, sn = [], [], [], []
        for i in idx:
            it = ds[i]
            if net.psd_cond:
                a, pv, ng, sv, band = it; ab.append(band.numpy())
            else:
                a, pv, ng, sv = it
            st.append(a.numpy()); tr.append(pv.numpy()[0]); sn.append(float(sv))
        stt = torch.from_numpy(np.stack(st)).to(dev).float()
        abt = torch.from_numpy(np.stack(ab)).to(dev).float() if net.psd_cond else None
        r = torch.zeros(len(idx), dtype=torch.long, device=dev)
        ctx = net._full_context(net.encode(stt, abt), r)
        crep = ctx.unsqueeze(1).expand(len(idx), K, ctx.shape[1]).reshape(-1, ctx.shape[1])
        z = torch.randn(len(idx) * K, 11, device=dev)
        y, _ = net.flow.inverse(z, crep)
        samp = net.scaler.denormalize(net.scaler.wrap(y)).reshape(len(idx), K, 11).cpu().numpy()
        a_ = np.maximum(samp[:, :, M1], samp[:, :, M2]); b_ = np.minimum(samp[:, :, M1], samp[:, :, M2])
        samp[:, :, M1], samp[:, :, M2] = a_, b_
        for jj, i in enumerate(idx):
            tru[i] = tr[jj]; med[i] = np.median(samp[jj], axis=0); snr[i] = sn[jj]

    print("=== M9: IN-DISTRIBUTION point-estimate bias (calibrated ep40) ===")
    print("median fractional bias = median((pred - truth)/truth)\n")
    print(f"{'quantity':14} {'frac bias':>10} {'|frac err|':>11} {'corr':>7}")
    rows = {}
    for lab, pj in [("mass_1", M1), ("mass_2", M2), ("luminosity_dist", DL)]:
        t, p = tru[:, pj], med[:, pj]
        fb = float(np.median((p - t) / t)); ae = float(np.median(np.abs((p - t) / t)))
        cc = float(np.corrcoef(np.log(t), np.log(p))[0, 1])
        rows[lab] = dict(frac_bias=fb, abs_frac_err=ae, corr=cc)
        print(f"{lab:14} {fb:+10.3f} {ae:11.3f} {cc:7.3f}")
    tc, pc = chirp(tru[:, M1], tru[:, M2]), chirp(med[:, M1], med[:, M2])
    fb = float(np.median((pc - tc) / tc)); ae = float(np.median(np.abs((pc - tc) / tc)))
    rows["chirp_mass"] = dict(frac_bias=fb, abs_frac_err=ae,
                              corr=float(np.corrcoef(np.log(tc), np.log(pc))[0, 1]))
    print(f"{'chirp_mass':14} {fb:+10.3f} {ae:11.3f} {rows['chirp_mass']['corr']:7.3f}")

    # high-SNR subset (where the real events sit)
    hs = snr > 15
    print(f"\n--- high-SNR subset (SNR>15, n={int(hs.sum())}) ---")
    for lab, pj in [("mass_1", M1), ("luminosity_dist", DL)]:
        t, p = tru[hs, pj], med[hs, pj]
        print(f"{lab:14} frac_bias {float(np.median((p-t)/t)):+.3f}  "
              f"|frac err| {float(np.median(np.abs((p-t)/t))):.3f}")
    tq = tru[:, M2] / tru[:, M1]; pq = med[:, M2] / med[:, M1]
    print(f"\nmass ratio q: truth med {np.median(tq):.3f} -> pred med {np.median(pq):.3f}; "
          f"frac(pred q>0.95) = {float((pq > 0.95).mean()):.3f} "
          f"(truth {float((tq > 0.95).mean()):.3f})")
    json.dump(rows, open(HERE / "m9_indist_bias.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
