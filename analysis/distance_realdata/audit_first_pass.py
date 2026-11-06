"""How good is the matched-filter first pass already? (2026-11-06)

DINGO's refinement passes hold each detector's arrival time within +-1 ms. Ours gets
arrival times and a chirp mass from one matched-filter pass. If those are already at
that precision, more time-alignment passes have little to add.

Real-noise validation, single-signal events; arrival errors after removing the
median offset, per detector carrying at least 3 SNR of the signal.
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "experiments")
from remix_data import RemixDataset

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="data/dataset_bbh_v6/memmap/validation")
ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()
from ahsd.inference.pipeline import load_model
from ahsd.data.parameter_sampler import GPS_REF
from ahsd.models.lean_npe import PARAM_NAMES as N
from bilby.gw.detector import get_empty_interferometer

model, _ = load_model(a.model, device="cpu")
ds = RemixDataset(a.data, remix=False, seed=1234, real_noise_dir="data/noise_bank",
                  real_noise_prob=1.0, return_asd_bands=True, psd_bands=16, single_signal=True)
n = min(a.n, len(ds))
items = [ds[i] for i in range(n)]
x = torch.stack([i[0] for i in items]).float()
ab = torch.stack([i[4] for i in items]).float()
y = torch.stack([i[1][0] for i in items]).numpy()
snr = torch.stack([i[3] for i in items]).numpy().astype(float).reshape(n, -1)[:, 0]

arr, lmc = [], []
with torch.no_grad():
    for i in range(0, n, 64):
        model.encoder(x[i:i + 64], ab[i:i + 64])
        mf = model.encoder.mc_feat
        arr.append(mf.last_arr.numpy())
        lmc.append(mf.last_log_mc.numpy())
arr, lmc = np.concatenate(arr), np.concatenate(lmc)

ifo = {d: get_empty_interferometer(d) for d in ("H1", "L1", "V1")}
ic = N.index("geocent_time")
delay = np.array([[ifo[d].time_delay_from_geocenter(y[k, 3], y[k, 4], GPS_REF)
                   for d in ("H1", "L1", "V1")] for k in range(n)])
truth = y[:, ic, None] + delay
err = arr - truth
err = (err + 2) % 4 - 2
err -= np.median(err, 0, keepdims=True)
true_lmc = np.log((y[:, 0] * y[:, 1]) ** .6 / (y[:, 0] + y[:, 1]) ** .2)
dm = lmc - true_lmc

# per-detector optimal SNR of the stored signal, recoloured as in the input
start = [ds.events[int(ds._index_map[i])][0] for i in range(n)]
sd = np.stack([np.sqrt((np.asarray(ds.signals[s][:, :16384], np.float64) ** 2).sum(-1)) for s in start]) * snr[:, None] / np.maximum(
    np.sqrt((np.stack([np.asarray(ds.signals[s][:, :16384], np.float64) for s in start]) ** 2).sum((1, 2)))[:, None], 1e-9)

print(f"n={n}; arrival error per detector, ms; network chirp-mass error |d ln Mc|, percent")
for lo, hi in ((0, 6), (6, 10), (10, 20), (20, 40), (40, 1e9), (10, 25)):
    m = (snr >= lo) & (snr < hi)
    if not m.sum():
        continue
    e = np.abs(err[m])[sd[m] >= 3] * 1e3
    if e.size:
        print(f"  SNR {lo:>2}-{hi:<6g} n={int(m.sum()):4d}  arrival median {np.median(e):6.2f}  90% {np.quantile(e, .9):7.2f}  "
              f"within 1 ms {(e < 1).mean():.0%}  |d ln Mc| median {100 * np.median(np.abs(dm[m])):5.2f}%  90% {100 * np.quantile(np.abs(dm[m]), .9):6.2f}%")
print(f"signed d ln Mc median {100 * np.median(dm):.2f}%")
print(f"q<0.5, SNR 10-25: |d ln Mc| median {100 * np.median(np.abs(dm[(y[:, 1] / y[:, 0] < .5) & (snr >= 10) & (snr < 25)])):.2f}%; "
      f"q>=0.5 {100 * np.median(np.abs(dm[(y[:, 1] / y[:, 0] >= .5) & (snr >= 10) & (snr < 25)])):.2f}%")
