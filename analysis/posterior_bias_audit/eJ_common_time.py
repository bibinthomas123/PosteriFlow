#!/usr/bin/env python3
"""E-J: does the per-template TIME ARGMAX destroy the mass information?

E-I ruled out grid resolution: a 20x finer Mc grid moved q R^2 only -0.012 ->
0.029 and log Mc 0.353 -> 0.367. So the deficit is in how the matched-filter
output is SUMMARIZED, not in what the bank covers.

The suspect is `_mf_features`' `idx = a.argmax(dim=-1)`: peak |z| is taken
INDEPENDENTLY PER TEMPLATE, i.e. every template is free to slide in time to its
own best fit. Chirp mass and coalescence time are strongly correlated in the
inspiral phase — a slightly wrong Mc is compensated almost exactly by a time
shift. Maximizing over t_c per template therefore absorbs the very mismatch that
encodes Mc (and, through the ridge, q): the profile over the bank is flattened
by construction.

The correct statistic evaluates the whole bank at a COMMON time, so templates
are compared on equal footing and the mismatch survives.

  common-time features >> per-template-argmax  -> the summary is the bug; it is
      a cheap fix inside the existing branch, no new front-end required.
  no difference                                -> time maximization is not the
      issue and the deficit is deeper in the representation.
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
N_WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def main():
    bk = np.load(ROOT / "src/ahsd/models/assets/mf_bank_fine.npz")
    band_lo = int(bk["band_lo"]); Nf = int(bk["n_band"])
    H = torch.complex(torch.from_numpy(bk["bank_re"]),
                      torch.from_numpy(bk["bank_im"])).to(DEV)
    K = H.shape[1]
    dt = (T_LEN / SR) / Nf
    ctr = Nf // 2
    w = min(int(1.6 / dt), ctr - 1)

    def both_features(strain):
        """Returns (per_template_argmax_feats, common_time_feats), each [B,D,K,3].

        Identical inputs and identical |z| series — the ONLY difference is which
        time index each template is read at."""
        B, Dn, T = strain.shape
        fd = torch.fft.rfft(strain.float().contiguous(), dim=-1)
        d = fd[..., band_lo: band_lo + Nf]
        chunk = max(1, int(3e7 / max(B * Dn * Nf * 8, 1)))
        zs = []
        for k0 in range(0, K, chunk):
            X = d.unsqueeze(2) * H[:, k0:k0 + chunk].conj().unsqueeze(0)
            z = torch.fft.ifft(X, dim=-1)
            zs.append(z[..., ctr - w: ctr + w])
        z = torch.cat(zs, dim=2)                      # [B,D,K,2w]
        a = z.abs()
        scale = Nf / math.sqrt(T / 2.0)

        def read_at(idx):
            zp = torch.complex(torch.take_along_dim(z.real, idx, dim=-1).squeeze(-1),
                               torch.take_along_dim(z.imag, idx, dim=-1).squeeze(-1))
            ph = torch.angle(zp)
            return torch.stack([torch.asinh(zp.abs() * scale),
                                torch.cos(ph), torch.sin(ph)], dim=-1)

        # (a) current: every template picks its OWN best time
        idx_own = a.argmax(dim=-1, keepdim=True)                       # [B,D,K,1]
        # (b) common time: the single best (template, time) per detector, then
        #     read EVERY template at that one time
        flat = a.reshape(B, Dn, -1).argmax(dim=-1)                     # [B,D]
        t_common = (flat % a.shape[-1]).view(B, Dn, 1, 1).expand(B, Dn, K, 1)
        return read_at(idx_own), read_at(t_common)

    ds = RemixDataset(str(ROOT / "data/dataset/memmap/validation"), remix=True, seed=5,
                      real_noise_prob=0.0, return_asd_bands=False)
    ds.set_epoch(0)
    cand = []
    for ei in range(len(ds)):
        start, _ = ds.events[int(ds._index_map[ei])]
        m1, m2 = float(ds.params[start][0]), float(ds.params[start][1])
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        if MC_LO <= mc <= MC_HI:
            cand.append(ei)
        if len(cand) >= N_WANT * 2:
            break

    FA, FB, MC, Q, SNR = [], [], [], [], []
    B = 4
    for i0 in range(0, len(cand), B):
        items = [ds[j] for j in cand[i0:i0 + B]]
        strain = torch.stack([x[0] for x in items]).to(DEV).float()
        pv = torch.stack([x[1] for x in items]).numpy()
        snr = torch.stack([x[3] for x in items]).numpy()
        with torch.no_grad():
            fa, fb = both_features(strain)
        fa, fb = fa.cpu().numpy(), fb.cpu().numpy()
        for k in range(len(items)):
            m1, m2 = pv[k, 0, 0], pv[k, 0, 1]
            mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
            if not (MC_LO <= mc <= MC_HI):
                continue
            FA.append(fa[k]); FB.append(fb[k])
            MC.append(mc); Q.append(m2 / m1); SNR.append(snr[k])
        if len(MC) >= N_WANT:
            break
        if i0 % 40 == 0:
            print(f"  {len(MC)}/{N_WANT}", flush=True)

    FA, FB = np.array(FA), np.array(FB)
    MC, Q, SNR = np.array(MC), np.array(Q), np.array(SNR)
    print(f"\nevents used: {len(MC)}  median SNR {np.median(SNR):.1f}")

    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler

    def r2(X, y):
        X = StandardScaler().fit_transform(np.nan_to_num(X.reshape(len(X), -1)))
        p = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 4, 30)), X, y, cv=5)
        return 1 - ((y - p) ** 2).mean() / y.var()

    print("\n" + "=" * 76)
    print(f"{'subset':12s} {'target':8s} {'per-template argmax':>20s} {'common time':>13s}")
    print("=" * 76)
    for tag, m in [("ALL", np.ones(len(MC), bool)), ("SNR>15", SNR > 15)]:
        if m.sum() < 60:
            continue
        for nm, y in [("q", Q[m]), ("log Mc", np.log(MC[m]))]:
            print(f"{tag:12s} {nm:8s} {r2(FA[m], y):20.3f} {r2(FB[m], y):13.3f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
