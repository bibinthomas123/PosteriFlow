"""Why does certification's Gaussian NLL differ from audit_gap's arm A? (2026-11-06)

Certification: first n validation events, stored noise, all event types, rank 0.
audit_gap arm A: random single-signal events, fresh white noise, all detectors.
Here both noises are evaluated on certification's own events, then split by the
quantities that differ.
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

model, _ = load_model(a.model, device="cpu")
d = len(model.param_names)
kw = dict(remix=False, seed=1234, return_asd_bands=True, psd_bands=16)
ds_g = RemixDataset(a.data, **kw)
ds_r = RemixDataset(a.data, real_noise_dir="data/noise_bank", real_noise_prob=1.0, **kw)
rng = np.random.default_rng(0)
n = min(a.n, len(ds_g))


def nll(x, th, ab):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            out.append(model.nll(x[i:i + 64].float(), th[i:i + 64].float(),
                                 torch.zeros(len(x[i:i + 64]), dtype=torch.long),
                                 asd_bands=ab[i:i + 64].float()).numpy())
    return np.concatenate(out)


items = [ds_g[i] for i in range(n)]
real = [ds_r[i] for i in range(n)]
xg = torch.stack([x[0] for x in items])
xr = torch.stack([x[0] for x in real])
th = torch.stack([x[1][0][:d] for x in items])
abg = torch.stack([x[4] for x in items])
abr = torch.stack([x[4] for x in real])
snr_g = torch.stack([x[3] for x in items]).numpy().astype(float).reshape(n, -1)[:, 0]
snr_r = torch.stack([x[3] for x in real]).numpy().astype(float).reshape(n, -1)[:, 0]
nsig = np.array([ds_g.events[int(ds_g._index_map[i])][1] for i in range(n)])
ndet = (xg.abs().sum(-1) > 0).sum(1).numpy()

# fresh white noise on the same stored signals, only where the channel is live
xw = []
resid_std = []
for i in range(n):
    start, k = ds_g.events[int(ds_g._index_map[i])]
    sig = np.sum([np.asarray(ds_g.signals[start + j][:, :16384], np.float64) for j in range(k)], 0)
    live = (xg[i].abs().sum(-1) > 0).numpy()
    resid_std.append((xg[i].numpy() - sig)[live].std())
    w = rng.standard_normal((3, 16384))
    w[~live] = 0
    z = sig + w
    z[~live] = 0
    xw.append(z)
xw = torch.from_numpy(np.stack(xw))

G, W, R = nll(xg, th, abg), nll(xw, th, abg), nll(xr, th, abr)
print(f"n={n}  stored-noise residual std median {np.median(resid_std):.3f}, 95% {np.quantile(resid_std, .95):.3f}")
print(f"mean NLL: stored Gaussian {G.mean():.2f}, fresh white {W.mean():.2f}, real {R.mean():.2f}")
print(f"median NLL: stored Gaussian {np.median(G):.2f}, fresh white {np.median(W):.2f}, real {np.median(R):.2f}")
print(f"SNR median: Gaussian {np.median(snr_g):.1f}, real {np.median(snr_r):.1f}; "
      f"mean {snr_g.mean():.1f}, {snr_r.mean():.1f}")


def row(lab, m):
    if m.sum():
        print(f"  {lab:16s} n={int(m.sum()):5d} G {G[m].mean():7.2f} W {W[m].mean():7.2f} R {R[m].mean():7.2f} "
              f"R-G mean {(R - G)[m].mean():6.2f} median {np.median((R - G)[m]):6.2f}")


print("by number of signals")
for k in sorted(set(nsig.tolist())):
    row(f"nsig={k}", nsig == k)
print("by live detectors")
for k in sorted(set(ndet.tolist())):
    row(f"ndet={k}", ndet == k)
print("by Gaussian-domain SNR")
for lo, hi in ((0, 6), (6, 10), (10, 20), (20, 40), (40, 1e9)):
    row(f"SNR {lo}-{hi:g}", (snr_g >= lo) & (snr_g < hi))
print("by real-domain SNR")
for lo, hi in ((0, 6), (6, 10), (10, 20), (20, 40), (40, 1e9)):
    row(f"SNR {lo}-{hi:g}", (snr_r >= lo) & (snr_r < hi))
print("mean R-G once the worst tails are dropped (trim 1% each side):",
      round(float(np.sort(R - G)[n // 100:-n // 100].mean()), 2))
print("NLL quantiles 1/10/50/90/99  G", np.round(np.quantile(G, [.01, .1, .5, .9, .99]), 1).tolist())
print("NLL quantiles 1/10/50/90/99  R", np.round(np.quantile(R, [.01, .1, .5, .9, .99]), 1).tolist())
