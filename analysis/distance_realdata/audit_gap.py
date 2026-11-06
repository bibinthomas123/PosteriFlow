"""What is the real vs Gaussian NLL gap made of? (2026-11-06)

Same validation signals; one change per arm:
  A design-loudness signal, Gaussian noise, asd_bands 0   (the "Gaussian" number)
  B true real-noise loudness, Gaussian noise, real asd_bands   A-B: signals are quieter
  C true real-noise loudness, real noise, real asd_bands       B-C: real noise itself
  D old too-loud signal (no highpass gain), real noise, real asd_bands

CAVEAT: arms A and B use plain white noise, which costs 5.4 nats on its own because
stored Gaussian noise is tapered and empty below 20 Hz (audit_stored_noise.py), and
B also pairs it with real asd_bands. Their splits are not valid; see audit_gap_cert.py.
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "experiments")
from remix_data import RemixDataset

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--data", default="data/dataset_bbh_v6/memmap/validation")
ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()
from ahsd.inference.pipeline import load_model

ds = RemixDataset(a.data, remix=False, real_noise_dir="data/noise_bank", real_noise_prob=1.0,
                  return_asd_bands=True, psd_bands=16, single_signal=True)
rng = np.random.default_rng(0)
idx = rng.choice(ds._index_map, min(a.n, len(ds._index_map)), replace=False)
T = 16384
X = {k: [] for k in "ABCD"}
AB, Y, SNR, SNR_D = [], [], [], []
for i in idx:
    start, _ = ds.events[int(i)]
    sig = np.asarray(ds.signals[start][:, :T], np.float64)
    crops, filts = ds._real_noise_and_filters(rng)
    F = np.fft.rfft(sig, axis=-1)
    filt = np.stack(filts)
    s_new = np.fft.irfft(F * filt * ds._hp_gain, n=T, axis=-1)
    s_old = np.fft.irfft(F * filt, n=T, axis=-1)
    white = rng.standard_normal((3, T))
    X["A"].append(sig + white)
    X["B"].append(s_new + white)
    X["C"].append(s_new + crops)
    X["D"].append(s_old + crops)
    AB.append(ds._asd_bands(filts, [0, 1, 2]).numpy())
    Y.append(np.asarray(ds.params[start], np.float32))
    SNR.append(np.sqrt((s_new ** 2).sum()))
    SNR_D.append(np.sqrt((s_old ** 2).sum()))
AB, Y, SNR, SNR_D = np.stack(AB), np.stack(Y), np.array(SNR), np.array(SNR_D)
print(f"n={len(idx)} network SNR: true-loudness median {np.median(SNR):.1f}, old-loudness median {np.median(SNR_D):.1f}", flush=True)
for mp in a.models:
    model, _ = load_model(mp, device="cpu")
    d = len(model.param_names)

    def nll(key, zero_bands=False):
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                x = torch.from_numpy(np.stack(X[key][i:i + 64])).float()
                ab = torch.zeros(len(x), 3, 16) if zero_bands else torch.from_numpy(AB[i:i + 64]).float()
                th = torch.from_numpy(Y[i:i + 64, :d]).float()
                out.append(model.nll(x, th, torch.zeros(len(x), dtype=torch.long), asd_bands=ab).numpy())
        return np.concatenate(out)

    N = {"A": nll("A", True), "B": nll("B"), "C": nll("C"), "D": nll("D")}
    print(f"== {mp.split('/')[-2]}")
    for k, v in N.items():
        print(f"  {k}: mean NLL {v.mean():8.3f}  median {np.median(v):8.3f}")
    print(f"  A->C total gap (mean) {N['C'].mean() - N['A'].mean():.2f}, median {np.median(N['C'] - N['A']):.2f}")
    print(f"  A->B quieter signals  (mean) {N['B'].mean() - N['A'].mean():.2f}, median {np.median(N['B'] - N['A']):.2f}")
    print(f"  B->C real noise       (mean) {N['C'].mean() - N['B'].mean():.2f}, median {np.median(N['C'] - N['B']):.2f}")
    print(f"  A->D old-recipe gap   (mean) {N['D'].mean() - N['A'].mean():.2f}, median {np.median(N['D'] - N['A']):.2f}")
    print("  B->C by true SNR:", [(lo, round(float((N['C'] - N['B'])[(SNR >= lo) & (SNR < hi)].mean()), 2), int(((SNR >= lo) & (SNR < hi)).sum()))
                                  for lo, hi in ((0, 6), (6, 10), (10, 20), (20, 40), (40, 1e9))], flush=True)
