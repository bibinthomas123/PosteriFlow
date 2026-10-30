"""M8 - Would a denser corner actually help? Density vs per-event bias.

For each validation event (has truth) measure base concentration ||z||^2/D (the
overconfidence/bias signal) AND its LOCAL TRAINING DENSITY in (log Mc, q,
log SNR) space. If, AT FIXED SNR, higher density -> lower ||z||^2/D, then more
corner data helps. If ||z||^2/D is set by SNR regardless of density, data won't
fix it (overconfidence dominates). We separate the two with a controlled
regression  ||z||^2/D ~ log10 SNR + log10 density  and SNR-banded binning.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis/divergence_localization"))
from _events import EVENTS, PROBLEM, chirp_mass  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402
from ahsd.inference.pipeline import load_model  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

HERE = Path(__file__).resolve().parent
VCACHE = "data/dataset/memmap/validation"
TRAIN = ROOT / "data/dataset/memmap/train"
MODEL = "model/long_run/best_model.pth"
N = 4000
D = len(PARAM_NAMES)
KNN = 50


def standardize(Mc, q, snr, mu=None, sd=None):
    X = np.stack([np.log10(Mc), q, np.log10(np.clip(snr, 1e-3, None))], 1)
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


@torch.no_grad()
def val_zconc(net, dev):
    ds = RemixDataset(VCACHE, remix=False, seed=1234, return_asd_bands=net.psd_cond)
    n = min(N, len(ds))
    Mc = np.empty(n); q = np.empty(n); snr = np.empty(n); zc = np.empty(n)
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
        st = torch.from_numpy(np.stack(st)).to(dev).float()
        tr = np.stack(tr); trt = torch.from_numpy(tr).to(dev).float()
        abt = torch.from_numpy(np.stack(ab)).to(dev).float() if net.psd_cond else None
        r = torch.zeros(len(idx), dtype=torch.long, device=dev)
        ctx = net._full_context(net.encode(st, abt), r)
        z, _ = net.flow.forward(net.scaler.normalize(trt), ctx)
        zb = (z.pow(2).sum(1) / D).cpu().numpy()
        for jj, i in enumerate(idx):
            Mc[i] = chirp_mass(tr[jj, 0], tr[jj, 1]); q[i] = tr[jj, 1] / tr[jj, 0]
            snr[i] = sn[jj]; zc[i] = zb[jj]
    return Mc, q, snr, zc


def main():
    net, meta = load_model(MODEL)
    Mc_v, q_v, snr_v, zc = val_zconc(net, meta["device"])

    # training population in (logMc,q,logSNR) + local density via kNN radius
    p = np.load(TRAIN / "params.npy"); snr_t = np.load(HERE.parent / "_snr_cache.npy")
    Mc_t = chirp_mass(p[:, 0], p[:, 1]); q_t = p[:, 1] / p[:, 0]
    Xt, mu, sd = standardize(Mc_t, q_t, snr_t)
    Xv, _, _ = standardize(Mc_v, q_v, snr_v, mu, sd)
    tree = cKDTree(Xt)
    dk, _ = tree.query(Xv, k=KNN)                 # [N, KNN]
    rk = dk[:, -1]                                 # kNN radius (bigger = sparser)
    log_density = -3.0 * np.log10(rk + 1e-9)       # density ∝ 1/r^3

    # controlled regression: zc ~ 1 + log10 SNR + log_density
    A = np.stack([np.ones_like(zc), np.log10(snr_v), log_density], 1)
    coef, *_ = np.linalg.lstsq(A, zc, rcond=None)
    pred = A @ coef
    r2 = 1 - np.var(zc - pred) / np.var(zc)
    # standardized betas
    betas = coef[1:] * A[:, 1:].std(0) / zc.std()
    # partial correlations
    def pcorr(y, x, z):
        ry = y - np.polyval(np.polyfit(z, y, 1), z)
        rx = x - np.polyval(np.polyfit(z, x, 1), z)
        return np.corrcoef(ry, rx)[0, 1]
    pc_dens = pcorr(zc, log_density, np.log10(snr_v))
    pc_snr = pcorr(zc, np.log10(snr_v), log_density)

    print("=== M8: does local training density predict per-event bias ||z||^2/D? ===\n")
    print(f"n_val={len(zc)}  mean ||z||^2/D={zc.mean():.2f} (calibrated=1.0)\n")
    print("OLS  ||z||^2/D ~ 1 + log10(SNR) + log10(density) :")
    print(f"   intercept        {coef[0]:+.3f}")
    print(f"   log10(SNR)       {coef[1]:+.3f}   (std beta {betas[0]:+.2f})")
    print(f"   log10(density)   {coef[2]:+.3f}   (std beta {betas[1]:+.2f})")
    print(f"   R^2={r2:.3f}")
    print(f"   partial corr(||z||^2 , density | SNR) = {pc_dens:+.3f}")
    print(f"   partial corr(||z||^2 , SNR | density) = {pc_snr:+.3f}\n")

    # controlled binning: within high-SNR band, ||z||^2 vs density quartile
    for lo, hi, lab in [(5, 15, "SNR 5-15"), (18, 40, "SNR 18-40 (corner regime)")]:
        m = (snr_v >= lo) & (snr_v < hi)
        if m.sum() < 40:
            continue
        d = log_density[m]; z = zc[m]
        qs = np.quantile(d, [0, .25, .5, .75, 1.0])
        print(f"{lab}: ||z||^2/D by density quartile (Q1=sparsest .. Q4=densest):")
        cells = []
        for k in range(4):
            mm = (d >= qs[k]) & (d <= qs[k + 1])
            cells.append(f"Q{k+1}={z[mm].mean():.2f}(n{mm.sum()})")
        print("   " + "  ".join(cells))
        print(f"   Δ(Q1-Q4) = {z[(d>=qs[0])&(d<=qs[1])].mean() - z[(d>=qs[3])&(d<=qs[4])].mean():+.2f}")
    # how much would 2x / 10x density buy at fixed SNR?
    print(f"\n  => predicted ||z||^2/D drop per 2x density = {coef[2]*np.log10(2):+.3f}")
    print(f"  => predicted ||z||^2/D drop per 10x density = {coef[2]*1.0:+.3f}")
    print(f"  => irreducible SNR span (low->high SNR) = "
          f"{coef[1]*(np.log10(30)-np.log10(8)):+.3f}  (NOT fixable by data)")
    print(f"  corner gap to close: 3.71 -> ~1.0 = 2.71")

    # locate the 6 real events on the density scale
    Mc_e = np.array([chirp_mass(EVENTS[e]["m1"], EVENTS[e]["m2"]) for e in EVENTS])
    q_e = np.array([EVENTS[e]["m2"] / EVENTS[e]["m1"] for e in EVENTS])
    snr_e = np.array([EVENTS[e]["snr"] for e in EVENTS])
    Xe, _, _ = standardize(Mc_e, q_e, snr_e, mu, sd)
    dke, _ = tree.query(Xe, k=KNN)
    dens_e = -3.0 * np.log10(dke[:, -1] + 1e-9)
    dens_pct = np.array([(log_density < de).mean() * 100 for de in dens_e])
    print("\n  real events, local training-density percentile (0=sparsest):")
    for name, dp in zip(EVENTS, dens_pct):
        flag = " <==" if name in PROBLEM else ""
        print(f"    {name:10s} density pct={dp:5.1f}{flag}")

    json.dump({"coef": dict(intercept=coef[0], log_snr=coef[1], log_density=coef[2]),
               "r2": float(r2), "pcorr_density_given_snr": float(pc_dens),
               "pcorr_snr_given_density": float(pc_snr),
               "drop_per_2x_density": float(coef[2] * np.log10(2)),
               "event_density_pct": {n: float(d) for n, d in zip(EVENTS, dens_pct)}},
              open(HERE / "m8_density_vs_bias.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
