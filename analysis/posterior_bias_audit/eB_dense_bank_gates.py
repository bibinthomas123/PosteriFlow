#!/usr/bin/env python3
"""E-B: re-run both gates against the DENSE, prior-matched MF bank.

Gate 1 (alignment, E-E1): median |peak-time residual| must fall below 15 ms
                          GLOBALLY, not just where the old bank had templates.
Gate 2 (information, E3):  MF-feature -> q probe R^2 must exceed 0.25.

Deliberately standalone: `_mf_features` is pure signal processing (the bank is a
non-persistent buffer, no learned weights), so this replicates its math directly
against the new bank and never loads or risks a trained checkpoint.
"""
import sys, math
from pathlib import Path
ROOT = Path("/Users/bibinthomas/Developer/PosteriFlow")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch
from experiments.remix_data import RemixDataset

SR, T_LEN = 4096, 16384
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
BANK = sys.argv[1] if len(sys.argv) > 1 else "src/ahsd/models/assets/mf_bank_dense.npz"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600

bk = np.load(ROOT / BANK)
mc_q = bk["mc_q"]; band_lo = int(bk["band_lo"]); Nf = int(bk["n_band"])
H = torch.complex(torch.from_numpy(bk["bank_re"]), torch.from_numpy(bk["bank_im"])).to(DEV)
K = H.shape[1]
print(f"bank {BANK}: {K} templates, Mc [{mc_q[:,0].min():.2f}, {mc_q[:,0].max():.1f}], "
      f"q [{mc_q[:,1].min():.2f}, {mc_q[:,1].max():.2f}]")

dt = (T_LEN / SR) / Nf
ctr = Nf // 2
w = min(int(1.6 / dt), ctr - 1)


def mf_features(strain):
    """[B, D, K, 4] = asinh(snr), cos(phase), sin(phase), t_rel  (see _mf_features)."""
    B, Dn, T = strain.shape
    fd = torch.fft.rfft(strain.float().contiguous(), dim=-1)
    d = fd[..., band_lo: band_lo + Nf]
    chunk = max(1, int(4e7 / max(B * Dn * Nf * 8, 1)))
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
    return torch.stack([torch.asinh(snr), torch.cos(phase), torch.sin(phase), t_rel], dim=-1)


ds = RemixDataset(str(ROOT / "data/dataset/memmap/validation"), remix=True, seed=99,
                  real_noise_prob=0.0, return_asd_bands=False)
ds.set_epoch(0)

FEAT, RES, SNR, MC, Q = [], [], [], [], []
B = 8
for i0 in range(0, N, B):
    items = [ds[i] for i in range(i0, min(i0 + B, N))]
    strain = torch.stack([x[0] for x in items]).to(DEV).float()
    pv = torch.stack([x[1] for x in items]).numpy()
    snr = torch.stack([x[3] for x in items]).numpy()
    with torch.no_grad():
        f = mf_features(strain).cpu().numpy()
    asnr, t_rel = f[..., 0], f[..., 3]
    for k in range(len(items)):
        m1, m2 = pv[k, 0, 0], pv[k, 0, 1]
        bt = asnr[k].argmax(axis=1)
        t_est = np.median([t_rel[k, dd, bt[dd]] for dd in range(3)])
        RES.append(t_est - pv[k, 0, 8])
        FEAT.append(f[k].reshape(-1))
        SNR.append(snr[k]); MC.append((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2); Q.append(m2 / m1)
    if i0 % 80 == 0: print(f"  {i0}/{N}", flush=True)

RES = np.array(RES) * 1e3
FEAT = np.array(FEAT); SNR = np.array(SNR); MC = np.array(MC); Q = np.array(Q)
np.savez(Path(__file__).parent / "eB_raw.npz", RES=RES, SNR=SNR, MC=MC, Q=Q)

print("\n" + "=" * 100)
print("GATE 1 — alignment: MF peak-time residual vs truth (light-travel floor ~10 ms)")
print("=" * 100)
def rep(tag, m):
    if m.sum() < 15: return
    r = np.abs(RES[m])
    print(f"  {tag:26s} n={m.sum():4d}  median={np.median(r):7.1f} ms  p90={np.percentile(r,90):8.1f} ms"
          f"  <15ms {(r<15).mean()*100:5.1f}%")
rep("ALL", np.ones(len(RES), bool))
rep("Mc<12 (old bank blind)", MC < 12)
rep("Mc in [12,45]", (MC >= 12) & (MC <= 45))
rep("Mc>45", MC > 45)
for lo, hi in [(0, 12), (12, 20), (20, 35), (35, 500)]:
    rep(f"SNR [{lo},{hi})", (SNR >= lo) & (SNR < hi))
rep("SNR>20", SNR > 20)

med = np.median(np.abs(RES))
print(f"\n  GATE 1 (global median): {med:.1f} ms  ->  "
      f"{'PASS' if med < 15 else 'MARGINAL' if med < 50 else 'FAIL'}   [was 186.6 ms with old bank]")

print("\n" + "=" * 100)
print("GATE 2 — information: MF features -> (log Mc, q), 5-fold CV")
print("=" * 100)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(np.nan_to_num(FEAT))
for nm, y in [("log Mc", np.log(MC)), ("q", Q)]:
    for tag, m in [("ALL", np.ones(len(y), bool)), ("SNR>20", SNR > 20)]:
        p = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 4, 30)), X[m], y[m], cv=5)
        r2 = 1 - ((y[m] - p) ** 2).mean() / y[m].var()
        print(f"  {nm:7s} {tag:8s} n={m.sum():4d}  R2={r2:6.3f}")
print("\n  reference (OLD 120-template bank, same probe): log Mc ALL 0.248 / in-bank 0.552;  q ALL -0.001")
print("  GATE 2 target: q R2 > 0.25")
