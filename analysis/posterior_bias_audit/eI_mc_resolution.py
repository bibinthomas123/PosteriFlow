#!/usr/bin/env python3
"""E-I: is the MF branch's q-blindness caused by TEMPLATE Mc RESOLUTION?

The dense bank (E-B) covers q down to 0.07 and still yields q probe R^2 = 0.012,
even though dynesty resolves q to a 90% width of ~0.17 on the same data (E-H).
Matched filtering is the sufficient statistic, so the statistic itself should
carry q. Hypothesis: the bank's Mc spacing (5.9%) is ~20x coarser than the Mc
precision the data supports (dynesty reaches 0.2-0.5%), so every template near
the truth is dominated by Mc MISMATCH and the q dependence is buried under it —
the grid never resolves the Mc-q ridge it is meant to trace.

Design: one bank, fine in Mc (0.30% spacing over [9,12]), evaluated on the
validation events whose true Mc lands inside that window. The control is a
SUBSAMPLE of the same bank back to ~6% Mc spacing on the SAME events and the
SAME features — so grid resolution is the only variable.

  q R^2 rises with fine spacing  -> resolution is the cause. Matched filtering
     works but needs ~10^3-10^4x more templates near the truth, which is too
     expensive per forward pass => rules the MF-token route out as the front-end.
  q R^2 stays ~0                 -> the MF FEATURE construction is at fault
     (per-template time-argmax / asinh+phase summary discarding coherent
     structure), which is a bug to find before building any new front-end.
"""
import sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch
from experiments.remix_data import RemixDataset

SR, T_LEN = 4096, 16384
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MC_LO, MC_HI = 9.2, 11.8
N_Q = 12
COARSE_STRIDE = 20          # 0.30% * 20 = 6.0% -> matches the dense bank
N_WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def main():
    bk = np.load(ROOT / "src/ahsd/models/assets/mf_bank_fine.npz")
    mc_q = bk["mc_q"]; band_lo = int(bk["band_lo"]); Nf = int(bk["n_band"])
    H = torch.complex(torch.from_numpy(bk["bank_re"]),
                      torch.from_numpy(bk["bank_im"])).to(DEV)
    K = H.shape[1]
    n_mc = K // N_Q
    print(f"fine bank: {K} templates = {n_mc} Mc x {N_Q} q, "
          f"Mc [{mc_q[:,0].min():.2f}, {mc_q[:,0].max():.2f}], "
          f"spacing {(mc_q[N_Q,0]/mc_q[0,0]-1)*100:.2f}%")

    dt = (T_LEN / SR) / Nf
    ctr = Nf // 2
    w = min(int(1.6 / dt), ctr - 1)

    def mf_features(strain):
        B, Dn, T = strain.shape
        fd = torch.fft.rfft(strain.float().contiguous(), dim=-1)
        d = fd[..., band_lo: band_lo + Nf]
        chunk = max(1, int(3e7 / max(B * Dn * Nf * 8, 1)))
        zw = []
        for k0 in range(0, K, chunk):
            X = d.unsqueeze(2) * H[:, k0:k0 + chunk].conj().unsqueeze(0)
            z = torch.fft.ifft(X, dim=-1)
            zw.append(z[..., ctr - w: ctr + w])
        z = torch.cat(zw, dim=2)
        a = z.abs()
        idx = a.argmax(dim=-1, keepdim=True)
        zpk = torch.complex(torch.take_along_dim(z.real, idx, dim=-1).squeeze(-1),
                            torch.take_along_dim(z.imag, idx, dim=-1).squeeze(-1))
        n_bin = (idx.squeeze(-1).float() + (ctr - w))
        snr = zpk.abs() * (Nf / math.sqrt(T / 2.0))
        phase = torch.angle(zpk) + 2 * math.pi * band_lo * n_bin / Nf
        t_rel = (n_bin - ctr) * dt
        return torch.stack([torch.asinh(snr), torch.cos(phase),
                            torch.sin(phase), t_rel], dim=-1)   # [B,D,K,4]

    ds = RemixDataset(str(ROOT / "data/dataset/memmap/validation"), remix=True, seed=5,
                      real_noise_prob=0.0, return_asd_bands=False)
    ds.set_epoch(0)

    # pre-select events whose stored rank-0 Mc lands in the bank window
    cand = []
    for ei in range(len(ds)):
        start, nsig = ds.events[int(ds._index_map[ei])]
        m1, m2 = float(ds.params[start][0]), float(ds.params[start][1])
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        if MC_LO <= mc <= MC_HI:
            cand.append(ei)
        if len(cand) >= N_WANT * 2:
            break
    print(f"candidate events in Mc window: {len(cand)}")

    FEAT, MC, Q, SNR = [], [], [], []
    B = 4
    for i0 in range(0, len(cand), B):
        items = [ds[j] for j in cand[i0:i0 + B]]
        strain = torch.stack([x[0] for x in items]).to(DEV).float()
        pv = torch.stack([x[1] for x in items]).numpy()
        snr = torch.stack([x[3] for x in items]).numpy()
        with torch.no_grad():
            f = mf_features(strain).cpu().numpy()
        for k in range(len(items)):
            m1, m2 = pv[k, 0, 0], pv[k, 0, 1]
            mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
            if not (MC_LO <= mc <= MC_HI):
                continue                      # loudness re-sort moved rank 0
            FEAT.append(f[k]); MC.append(mc); Q.append(m2 / m1); SNR.append(snr[k])
        if len(MC) >= N_WANT:
            break
        if i0 % 40 == 0:
            print(f"  {len(MC)}/{N_WANT}", flush=True)

    FEAT = np.array(FEAT); MC = np.array(MC); Q = np.array(Q); SNR = np.array(SNR)
    print(f"\nevents used: {len(MC)}  Mc [{MC.min():.2f},{MC.max():.2f}]  "
          f"q [{Q.min():.2f},{Q.max():.2f}]  median SNR {np.median(SNR):.1f}")
    np.savez(Path(__file__).parent / "eI_raw.npz", MC=MC, Q=Q, SNR=SNR)

    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler

    def r2(X, y):
        X = StandardScaler().fit_transform(np.nan_to_num(X.reshape(len(X), -1)))
        p = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 4, 30)), X, y, cv=5)
        return 1 - ((y - p) ** 2).mean() / y.var()

    # template index k = i_mc * N_Q + i_q  (build loop is mc-major)
    keep_fine = np.arange(K)
    keep_coarse = np.array([i * N_Q + j for i in range(0, n_mc, COARSE_STRIDE)
                            for j in range(N_Q)])
    print(f"\ngrids: fine {len(keep_fine)} templates ({(mc_q[N_Q,0]/mc_q[0,0]-1)*100:.2f}% Mc) "
          f"| coarse {len(keep_coarse)} templates "
          f"({(mc_q[COARSE_STRIDE*N_Q,0]/mc_q[0,0]-1)*100:.1f}% Mc)")
    print("=" * 78)
    print(f"{'subset':16s} {'target':7s} {'COARSE ~6%':>12s} {'FINE 0.30%':>12s}")
    print("=" * 78)
    for tag, m in [("ALL", np.ones(len(MC), bool)), ("SNR>15", SNR > 15)]:
        if m.sum() < 60:
            continue
        for nm, y in [("q", Q[m]), ("log Mc", np.log(MC[m]))]:
            c = r2(FEAT[m][:, :, keep_coarse, :], y)
            f_ = r2(FEAT[m][:, :, keep_fine, :], y)
            print(f"{tag:16s} {nm:7s} {c:12.3f} {f_:12.3f}")
    print("=" * 78)
    print("reference: 614-template dense bank (5.9% Mc, full prior) gave q R2 = 0.012")


if __name__ == "__main__":
    main()
