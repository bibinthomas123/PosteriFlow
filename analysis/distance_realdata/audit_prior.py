"""Effective training prior and the remaining amplitude scale (2026-11-05).

PRIOR. The generator keeps a source if its stored distance is inside the source's
horizon T = min(hi, d * snr / min_snr); remix redraws distance uniformly below T:

    pi_eff(theta, d) = pi0(theta) * [(T-lo)/(hi-lo)] * [1/(T-lo)] * 1(d < T)
                     = pi0(theta) * 1/(hi-lo) * 1(design SNR >= min_snr)

i.e. the assumed prior, cut at design SNR >= min_snr. Testable: (1) stored distances
are uniform below T; (2) weighting stored sources by 1/(T-lo) restores the assumed
mass prior; (3) real events sit far above the cut.

AMPLITUDE. Splits training/inference on Gaussian noise of known PSD.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kstest

sys.path.insert(0, "experiments")
from remix_data import RemixDataset

SR = 4096
F4 = np.fft.rfftfreq(16384, 1 / SR)


def wquant(x, w, q):
    i = np.argsort(x)
    c = np.cumsum(w[i]) / w.sum()
    return np.interp(q, c, x[i])


def wks(x, w, y):
    """Two-sample KS distance between weighted x and unweighted y."""
    g = np.sort(np.concatenate([x[:: max(1, len(x) // 4000)], y[:: max(1, len(y) // 4000)]]))
    i = np.argsort(x)
    Fx = np.interp(g, x[i], np.cumsum(w[i]) / w.sum())
    Fy = np.searchsorted(np.sort(y), g) / len(y)
    return float(np.abs(Fx - Fy).max())


def prior_test(cache, model_path, n, snr_scale):
    from ahsd.inference.pipeline import load_model
    from ahsd.inference.priors import TrainingPrior
    from ahsd.models.lean_npe import PARAM_NAMES as N
    ds = RemixDataset(cache, remix=True)
    meta = load_model(model_path, device="cpu")[1]
    pr = TrainingPrior.from_meta(meta)
    lo, hi = pr.dist_range
    ev = np.asarray(ds.events)
    single = ev[ev[:, 1] == 1]
    print(f"events {len(ev)}, single-signal {len(single)}, prior d in {lo}-{hi}")
    pick = np.random.default_rng(0).choice(len(single), min(n, len(single)), replace=False)
    S = single[pick, 0]
    jd = N.index("luminosity_distance")
    Y = np.stack([np.asarray(ds.params[s])[:11] for s in S])
    snr = np.array([np.sqrt((np.asarray(ds.signals[s][:, :16384], np.float64) ** 2).sum()) for s in S])
    d = Y[:, jd]
    print(f"stored design SNR 1/5/50%: {np.round(np.quantile(snr, [.01, .05, .5]), 2).tolist()}, below 6: {(snr < 6).mean():.3f}")
    T = np.minimum(hi, d * snr / 6.0)
    u = (d - lo) / np.maximum(T - lo, 1e-6)
    ok = (T > lo) & (u >= 0) & (u <= 1)
    print(f"(1) stored d uniform below T: KS {kstest(u[ok], 'uniform').statistic:.3f} (n={ok.sum()})")
    w = 1.0 / np.maximum(T - lo, 1.0)
    s = pr.sample(len(Y), np.random.default_rng(1))
    q = [.05, .5, .95]
    print("(2) stored vs assumed prior, raw and after 1/(T-lo) weighting")
    for i, nm in enumerate(N[:11]):
        if nm == "luminosity_distance":
            continue
        print(f"  {nm:12s} KS raw {wks(Y[:, i], np.ones(len(Y)), s[:, i]):.3f} -> weighted {wks(Y[:, i], w, s[:, i]):.3f}"
              f" | raw {np.round(np.quantile(Y[:, i], q), 2).tolist()} weighted {np.round(wquant(Y[:, i], w, q), 2).tolist()}"
              f" prior {np.round(np.quantile(s[:, i], q), 2).tolist()}")
    mc = lambda a: (a[:, 0] * a[:, 1]) ** .6 / (a[:, 0] + a[:, 1]) ** .2
    print(f"  {'chirp_mass':12s} KS raw {wks(mc(Y), np.ones(len(Y)), mc(s)):.3f} -> weighted {wks(mc(Y), w, mc(s)):.3f}")
    print(f"  {'mass_ratio':12s} KS raw {wks(Y[:, 1] / Y[:, 0], np.ones(len(Y)), s[:, 1] / s[:, 0]):.3f} -> weighted {wks(Y[:, 1] / Y[:, 0], w, s[:, 1] / s[:, 0]):.3f}")
    print(f"(3) real-event design SNR at about {snr_scale:.1f}x the O3 SNR; the catalogue's lowest SNR 10 sits near "
          f"design SNR {10 * snr_scale:.0f}, against the cut at 6")


def amp_test():
    from ahsd.data.bilby_pipeline import BilbyPreprocessor, get_default_psd, highpass_gain
    from ahsd.inference.preprocessing import _whiten_real_segment
    p = get_default_psd("H1")
    S4 = np.interp(F4, p["frequencies"], p["psd"])
    G = highpass_gain(F4, SR)
    N = 80 * SR
    f = np.fft.rfftfreq(N, 1 / SR)
    asd = np.sqrt(np.interp(f, p["frequencies"], p["psd"]))
    t = np.arange(16384) / SR
    env = np.exp(-(((t - 2) / 0.15) ** 2))
    rows = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        X = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) * asd * np.sqrt(SR * N / 4)
        y = np.fft.irfft(X, n=N)
        for fr in (60.0, 120.0, 250.0):
            h = 1e-21 * env * np.sin(2 * np.pi * fr * t)
            z = y.copy()
            i0 = 40 * SR - 8192
            z[i0:i0 + 16384] += h
            w0, a0, nm, _ = _whiten_real_segment(y, SR, 40.0)
            w1, *_ = _whiten_real_segment(z, SR, 40.0)
            inf = np.sqrt(((w1 - w0) ** 2).sum())
            H = np.fft.rfft(h) / SR
            m = F4 >= 20
            true_snr = np.sqrt(4 * (np.abs(H[m]) ** 2 / S4[m]).sum() / 4.0)
            sig = BilbyPreprocessor(SR, 4.0).preprocess(h, p, detector="H1").astype(np.float64)
            bilby = np.sqrt((sig ** 2).sum())
            filt = np.sqrt(S4) / np.maximum(a0, 1e-30) * G
            est = np.sqrt((np.fft.irfft(np.fft.rfft(sig) * filt, n=16384) ** 2).sum())
            med = np.median((a0 / (np.sqrt(S4) * G))[(F4 > 30) & (F4 < 1500)])
            rows.append((inf / true_snr, bilby / true_snr, est / bilby, nm / np.sqrt(SR / 2), est / inf, med))
    r = np.median(np.array(rows), 0)
    print("AMPLITUDE on Gaussian noise of known PSD (median of 36):")
    print(f"  inference L2 / true SNR              {r[0]:.3f}")
    print(f"  bilby design whitening L2 / true SNR {r[1]:.3f}")
    print(f"  recolour with estimated ASD, gain out {r[2]:.3f}   (estimated / true ASD, median bin {r[5]:.3f})")
    print(f"  whitening norm / ideal               {r[3]:.3f}")
    print(f"  TOTAL training / inference           {r[4]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/dataset_bbh_v6/memmap/train")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=30000)
    a = ap.parse_args()
    amp_test()
    prior_test(a.cache, a.model, a.n, snr_scale=3.0)
