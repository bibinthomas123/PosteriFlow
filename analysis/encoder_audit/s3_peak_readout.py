#!/usr/bin/env python3
"""S3: does an INTERPOLATED PEAK readout of the MF grid recover Mc?

Hypothesis under test (analysis/encoder_audit/REPORT.md §0 + follow-up): chirp-mass
precision lives in the LOCATION of the matched-filter likelihood peak, not in the
|z| values themselves. Locating a peak to a fraction of the grid spacing is a
sharply nonlinear operation that a linear probe on |z| cannot express — which
would explain why every probe of raw MF features has come back weak (R^2 0.01-0.5)
even on a 0.30%-spaced bank, while dynesty reaches sigma_logMc ~ 0.003 from the
same data.

This is deliberately NOT a probe. It is a direct point estimate:
  parabolic interpolation of the network-SNR profile around its argmax
  -> (Mc_hat, q_hat), plus local curvature (the Fisher-like width)
and the metric is the residual scatter of log(Mc_hat) vs log(Mc_true), which is
directly comparable to dynesty's sigma_logMc and free of any probe-power concern.

Reference points for the same quantity:
  dynesty (E-H, 12 events)          sigma_logMc ~ 0.0032
  LeanNPE posterior (E1, SNR 20-35) sigma_logMc ~ 0.34
If the peak readout lands near dynesty, the fix is a small change to
_mf_features and the whole SVD front-end workstream is unnecessary.
"""
import sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch
from experiments.remix_data import RemixDataset
from ahsd.models.lean_npe import LeanNPE

SR, T_LEN = 4096, 16384
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MC_LO, MC_HI = 9.25, 11.75          # inside the bank, with margin for interpolation
N_WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def parabolic(y_lo, y_0, y_hi):
    """Sub-sample peak offset in [-0.5, 0.5] from three samples around a maximum."""
    den = (y_lo - 2.0 * y_0 + y_hi)
    return np.where(np.abs(den) < 1e-12, 0.0, 0.5 * (y_lo - y_hi) / np.where(np.abs(den) < 1e-12, 1.0, den))


def main():
    bk = np.load(ROOT / "src/ahsd/models/assets/mf_bank_fine.npz")
    mc_q = bk["mc_q"]; band_lo = int(bk["band_lo"]); Nf = int(bk["n_band"])
    H = torch.complex(torch.from_numpy(bk["bank_re"]),
                      torch.from_numpy(bk["bank_im"])).to(DEV)
    K = H.shape[1]
    N_Q = 12
    n_mc = K // N_Q
    mc_grid = mc_q[::N_Q, 0].astype(np.float64)          # [n_mc] log-spaced
    q_grid = mc_q[:N_Q, 1].astype(np.float64)            # [N_Q] descending
    dlogmc = np.log(mc_grid[1] / mc_grid[0])
    print(f"bank: {n_mc} Mc x {N_Q} q, Mc [{mc_grid[0]:.2f},{mc_grid[-1]:.2f}], "
          f"dlogMc={dlogmc:.5f} ({(np.exp(dlogmc)-1)*100:.2f}%)")

    dt = (T_LEN / SR) / Nf; ctr = Nf // 2; w = min(int(1.6 / dt), ctr - 1)

    def profile(strain):
        """[B, n_mc, N_Q] network matched-filter SNR profile."""
        B, Dn, T = strain.shape
        fd = torch.fft.rfft(strain.float().contiguous(), dim=-1)
        d = fd[..., band_lo: band_lo + Nf]
        chunk = max(1, int(3e7 / max(B * Dn * Nf * 8, 1)))
        pk = []
        for k0 in range(0, K, chunk):
            X = d.unsqueeze(2) * H[:, k0:k0 + chunk].conj().unsqueeze(0)
            z = torch.fft.ifft(X, dim=-1)[..., ctr - w: ctr + w]
            pk.append(z.abs().amax(dim=-1))              # [B,D,kc]
        peak = torch.cat(pk, dim=2)
        return (peak ** 2).sum(dim=1).sqrt().reshape(B, n_mc, N_Q)

    ck = torch.load(ROOT / "model/lean_npe_mf_scratch/best_model.pth",
                    map_location="cpu", weights_only=False)
    m = LeanNPE(psd_cond=True, psd_bands=16, encoder_type="coherent",
                mass_param="mc_q", mf_tokens=True)
    m.load_state_dict(ck["model_state_dict"]); m.to(DEV).eval()

    ds = RemixDataset(str(ROOT / "data/dataset/memmap/validation"), remix=True, seed=5,
                      real_noise_prob=0.0, return_asd_bands=True, psd_bands=16)
    ds.set_epoch(0)
    cand = []
    for ei in range(len(ds)):
        s0, _ = ds.events[int(ds._index_map[ei])]
        a, b = float(ds.params[s0][0]), float(ds.params[s0][1])
        if MC_LO <= (a * b) ** 0.6 / (a + b) ** 0.2 <= MC_HI:
            cand.append(ei)
        if len(cand) >= N_WANT * 2:
            break

    MCH, QH, MCT, QT, SNR, NPMC, NPQ, CURV = [], [], [], [], [], [], [], []
    B = 4
    for i0 in range(0, len(cand), B):
        items = [ds[j] for j in cand[i0:i0 + B]]
        st = torch.stack([x[0] for x in items]).to(DEV).float()
        ab = torch.stack([x[4] for x in items]).to(DEV).float()
        pv = torch.stack([x[1] for x in items]).numpy()
        sn = torch.stack([x[3] for x in items]).numpy()
        with torch.no_grad():
            P = profile(st).cpu().numpy().astype(np.float64)
            post = m.sample_posterior(st, rank=0, n_samples=1000, asd_bands=ab).cpu().numpy()
        for k in range(len(items)):
            a, b = pv[k, 0, 0], pv[k, 0, 1]
            mct = (a * b) ** 0.6 / (a + b) ** 0.2
            if not (MC_LO <= mct <= MC_HI):
                continue
            p = P[k]
            ia, ib = np.unravel_index(np.argmax(p), p.shape)
            # sub-grid peak in the Mc direction (log-spaced -> uniform in log)
            if 0 < ia < n_mc - 1:
                d_mc = float(parabolic(p[ia - 1, ib], p[ia, ib], p[ia + 1, ib]))
                curv = float(p[ia - 1, ib] - 2 * p[ia, ib] + p[ia + 1, ib])
            else:
                d_mc, curv = 0.0, 0.0
            mc_hat = float(np.exp(np.log(mc_grid[ia]) + np.clip(d_mc, -0.5, 0.5) * dlogmc))
            # sub-grid peak in the q direction (grid is non-uniform -> interp in index)
            if 0 < ib < N_Q - 1:
                d_q = float(np.clip(parabolic(p[ia, ib - 1], p[ia, ib], p[ia, ib + 1]), -0.5, 0.5))
            else:
                d_q = 0.0
            q_hat = float(np.interp(ib + d_q, np.arange(N_Q), q_grid))
            m1p, m2p = post[k, :, 0], post[k, :, 1]
            NPMC.append(float(np.median((m1p * m2p) ** 0.6 / (m1p + m2p) ** 0.2)))
            NPQ.append(float(np.median(m2p / m1p)))
            MCH.append(mc_hat); QH.append(q_hat); MCT.append(mct); QT.append(b / a)
            SNR.append(sn[k]); CURV.append(curv)
        if len(MCT) >= N_WANT:
            break
        if i0 % 40 == 0:
            print(f"  {len(MCT)}/{N_WANT}", flush=True)

    MCH, QH = np.array(MCH), np.array(QH)
    MCT, QT = np.array(MCT), np.array(QT)
    SNR, NPMC, NPQ = np.array(SNR), np.array(NPMC), np.array(NPQ)
    np.savez(Path(__file__).parent / "s3_raw.npz", MCH=MCH, QH=QH, MCT=MCT, QT=QT,
             SNR=SNR, NPMC=NPMC, NPQ=NPQ, CURV=np.array(CURV))

    rl_mf = np.log(MCH) - np.log(MCT)
    rl_np = np.log(NPMC) - np.log(MCT)
    print(f"\nn={len(MCT)}  median SNR {np.median(SNR):.1f}  "
          f"Mc_true [{MCT.min():.2f},{MCT.max():.2f}]")
    print("=" * 84)
    print(f"{'estimator':30s} {'sigma_logMc':>12s} {'bias':>9s} {'q MAE':>8s} {'q bias':>8s}")
    print("=" * 84)
    print(f"{'MF interpolated peak':30s} {np.std(rl_mf):12.4f} {np.mean(rl_mf):+9.4f} "
          f"{np.mean(np.abs(QH - QT)):8.3f} {np.mean(QH - QT):+8.3f}")
    print(f"{'LeanNPE posterior median':30s} {np.std(rl_np):12.4f} {np.mean(rl_np):+9.4f} "
          f"{np.mean(np.abs(NPQ - QT)):8.3f} {np.mean(NPQ - QT):+8.3f}")
    print(f"{'dynesty (E-H reference)':30s} {0.0032:12.4f} {0.0018:+9.4f} {'—':>8s} {'—':>8s}")
    print("=" * 84)
    print(f"grid quantization floor (dlogMc/sqrt(12)) = {dlogmc/np.sqrt(12):.5f}")
    print("\nby SNR:")
    for lo, hi in [(0, 15), (15, 25), (25, 40), (40, 500)]:
        s = (SNR >= lo) & (SNR < hi)
        if s.sum() < 15:
            continue
        print(f"  SNR[{lo:3d},{hi:3d}) n={s.sum():4d}  MF peak sigma_logMc={np.std(rl_mf[s]):.4f}"
              f"   NPE={np.std(rl_np[s]):.4f}   ratio={np.std(rl_np[s])/max(np.std(rl_mf[s]),1e-9):6.1f}x")


if __name__ == "__main__":
    main()
