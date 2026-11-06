"""Stored Gaussian noise scores 5.4 nats better than fresh white noise. Why? (2026-11-06)

Same signals, same asd_bands (0); only the noise changes:
  G  the event's stored noise (what certification uses)
  S  stored noise from a DIFFERENT event: drops any noise/signal pairing, keeps its statistics
  W  fresh white noise
  T  fresh white noise with the Tukey(0.1) taper
  B  T, also zeroed below 20 Hz

G << S means the stored noise carries information about its own event (a leak).
G ~ S with W, T, B worse means fresh noise simply differs from what the network saw.
"""
import argparse
import sys

import numpy as np
import torch
from scipy.signal.windows import tukey

sys.path.insert(0, "experiments")
from remix_data import RemixDataset

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="data/dataset_bbh_v6/memmap/validation")
ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()
from ahsd.inference.pipeline import load_model

model, _ = load_model(a.model, device="cpu")
d = len(model.param_names)
ds = RemixDataset(a.data, remix=False, seed=1234, return_asd_bands=True, psd_bands=16)
rng = np.random.default_rng(0)
n = min(a.n, len(ds))
T = 16384
items = [ds[i] for i in range(n)]
xg = torch.stack([x[0] for x in items]).numpy().astype(np.float64)
th = torch.stack([x[1][0][:d] for x in items])
sig = np.stack([np.sum([np.asarray(ds.signals[s + j][:, :T], np.float64)
                        for j in range(k)], 0)
                for s, k in (ds.events[int(ds._index_map[i])] for i in range(n))])
res = xg - sig


def nll(x):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            b = torch.from_numpy(x[i:i + 64]).float()
            out.append(model.nll(b, th[i:i + 64].float(), torch.zeros(len(b), dtype=torch.long),
                                 asd_bands=torch.zeros(len(b), 3, 16)).numpy())
    return np.concatenate(out)


f = np.fft.rfftfreq(T, 1 / 4096)
win = tukey(T, 0.1)
white = rng.standard_normal((n, 3, T))
tap = white * win
Ft = np.fft.rfft(tap, axis=-1)
Ft[..., f < 20] = 0
band = np.fft.irfft(Ft, n=T, axis=-1)
perm = rng.permutation(n)
print("noise statistics")
P = np.abs(np.fft.rfft(res, axis=-1)) ** 2 / (T / 2 * 2)
edges = [0, 10, 20, 30, 100, 500, 1000, 1500, 1900, 2049]
print("  stored power by band", [(lo, round(float(P[..., (f >= lo) & (f < hi)].mean()), 3))
                                 for lo, hi in zip(edges[:-1], edges[1:])])
tv = (res ** 2).mean((0, 1))
print("  stored variance, 1st/10th/50th percentile of time, edges vs middle",
      round(float(tv[:50].mean()), 3), round(float(tv[T // 2 - 200:T // 2 + 200].mean()), 3))
c = (res * sig).sum(-1) / np.maximum((sig ** 2).sum(-1), 1e-9)
z = (res * sig).sum(-1) / np.maximum(np.sqrt((sig ** 2).sum(-1)), 1e-9)
print(f"  noise/signal overlap in matched-filter units: mean {z.mean():.3f}, std {z.std():.3f} (independent: 0, 1)")
print(f"  regression of stored noise on signal: median {np.median(c):.4f}")

arms = {"G": xg, "S": sig + res[perm], "W": sig + white, "T": sig + tap, "B": sig + band}
N = {k: nll(v) for k, v in arms.items()}
for k, v in N.items():
    print(f"  {k}: mean NLL {v.mean():8.2f}  median {np.median(v):8.2f}  vs G mean {(v - N['G']).mean():6.2f}", flush=True)
