"""What reaches the network: training samples vs real events, feature by feature (2026-11-05)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kurtosis

import sys
sys.path.insert(0, "experiments")
from remix_data import RemixDataset

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--catalog", required=True)
ap.add_argument("--data", default="data/dataset_bbh_v6/memmap/validation")
ap.add_argument("--n", type=int, default=1500)
a = ap.parse_args()

from ahsd.inference import preprocessing as P
from ahsd.inference.pipeline import load_model

DETS = ("H1", "L1", "V1")
model, meta = load_model(a.model, device="cpu")
ckpt = torch.load(a.model, map_location="cpu", weights_only=False)["args"]
bands = model.encoder.psd_bands
ds = RemixDataset(a.data, remix=True, real_noise_dir="data/noise_bank", real_noise_prob=ckpt["real_noise_prob"],
                  det_dropout=ckpt["det_dropout"], return_asd_bands=True, psd_bands=bands,
                  dist_range=tuple(ckpt["dist_range"]), dist_resample=tuple(ckpt["dist_resample"]))
ds.set_epoch(1) if hasattr(ds, "set_epoch") else None
rng = np.random.default_rng(0)

# ---- training noise, no signal
noise = {d: [] for d in DETS}
noise_ab = {d: [] for d in DETS}
for _ in range(600):
    crops, filts = ds._real_noise_and_filters(rng)
    for i, d in enumerate(DETS):
        noise[d].append(crops[i])
        noise_ab[d].append(np.array([np.log(np.maximum(filts[i], 1e-30))[s].mean() for s in ds._band_slices]))

# ---- real events: off-source noise, on-source window, asd_bands
gps_of = {e["event_name"]: e["gps_time"] for e in json.load(open("data/gwtc/gwtc_events.json"))}
rn = {d: [] for d in DETS}
ra = {d: [] for d in DETS}
xs, abs_, present, names = [], [], [], []
for f in sorted(Path(a.catalog).glob("GW*.json")):
    ev = f.stem
    try:
        gps, raw, st, srs, _ = P.fetch_gwosc(gps_of[ev], strict=False)
        off = P.prepare_real(raw, gps - 20, st, srs, seed=0)
        on = P.prepare_real(raw, gps, st, srs, seed=0)
    except Exception as ex:
        print("skip", ev, str(ex)[:60])
        continue
    ab = P.compute_asd_bands(on, psd_bands=bands)
    for i, d in enumerate(DETS):
        if d in off.detectors_present:
            rn[d].append(off.strain[i])
        if d in on.detectors_present:
            ra[d].append(ab[i])
    xs.append(on.strain)
    abs_.append(ab)
    present.append(on.detectors_present)
    names.append(ev)
print("real events read:", len(xs))


def stat(c):
    c = np.asarray(c, float)
    return c.std(), kurtosis(c), np.abs(c).max()


print("NOISE ONLY (median over crops): std, excess kurtosis, max|x|; 95% of kurtosis")
for d in DETS:
    t = np.array([stat(c) for c in noise[d]])
    r = np.array([stat(c) for c in rn[d]]) if rn[d] else None
    print(f"  {d} train std {np.median(t[:,0]):.3f} kurt {np.median(t[:,1]):.3f} (95% {np.quantile(t[:,1],.95):.2f}) max {np.median(t[:,2]):.2f}"
          + (f" | real std {np.median(r[:,0]):.3f} kurt {np.median(r[:,1]):.3f} (95% {np.quantile(r[:,1],.95):.2f}) max {np.median(r[:,2]):.2f} n={len(r)}" if r is not None else ""))

print("NOISE SPECTRUM train/real by band (should be 1)")
F = np.fft.rfftfreq(16384, 1 / 4096)
for d in DETS:
    if not rn[d]:
        continue
    B = ((20, 35), (35, 80), (80, 200), (200, 800), (800, 2000))
    lev = lambda c: [np.median(np.abs(np.fft.rfft(c))[(F >= l) & (F < h)] ** 2) for l, h in B]
    T, R = np.array([lev(c) for c in noise[d][:300]]), np.array([lev(c) for c in rn[d]])
    print("  ", d, "median-of-crops", np.round(np.sqrt(np.median(T, 0) / np.median(R, 0)), 3).tolist(),
          "| worst real/median per band", np.round(np.sqrt(R.max(0) / np.median(R, 0)), 2).tolist(),
          "| train 99%/median", np.round(np.sqrt(np.quantile(T, .99, axis=0) / np.median(T, 0)), 2).tolist())

print("ASD_BANDS: real events vs training bank (band mean; real percentile within training)")
for d in DETS:
    t = np.array(noise_ab[d])
    r = np.array(ra[d])
    if len(r):
        p = [(t[:, b] < np.median(r[:, b])).mean() for b in range(t.shape[1])]
        print(f"  {d}: train mean {t.mean():.2f}, real mean {r.mean():.2f}; percentile of real median by band {np.round(p, 2).tolist()}")

# ---- what training actually hands the network, signal included
S, AB, Y, NS, SNR = [], [], [], [], []
idx = rng.integers(0, len(ds), a.n)
for i in idx:
    o = ds[int(i)]
    S.append(o[0].numpy()), Y.append(o[1][0].numpy()), NS.append(int(o[2])), SNR.append(float(o[3])), AB.append(o[4].numpy())
S, AB, Y = np.stack(S), np.stack(AB), np.stack(Y)
X = np.stack(xs).astype(np.float32)
RAB = np.stack(abs_).astype(np.float32)
print("INPUT nan/inf train", int((~np.isfinite(S)).sum()), "real", int((~np.isfinite(X)).sum()),
      "| clamp hits |x|>100 train", int((np.abs(S) > 100).sum()), "real", int((np.abs(X) > 100).sum()))
print("signals per sample:", {k: NS.count(k) for k in sorted(set(NS))}, "| noise-only:", sum(n == 0 for n in NS))
print("detector-band zeros (dropped/absent): train", np.round((np.abs(AB).sum(-1) == 0).mean(0), 3).tolist(),
      "real", np.round((np.abs(RAB).sum(-1) == 0).mean(0), 3).tolist())
print("network SNR train quantiles 5/50/95", np.round(np.quantile([s for s, n in zip(SNR, NS) if n], [.05, .5, .95]), 1).tolist(),
      "| real events", np.round(np.quantile([json.load(open(Path(a.catalog) / f"{e}.json"))["snr"] for e in names], [.05, .5, .95]), 1).tolist())
yn = model.scaler.normalize(torch.from_numpy(Y[[n > 0 for n in NS]][:, :11]).float())
print("labels at scaler edge |y|>0.999:", np.round((yn.abs() > 0.999).float().mean(0).numpy(), 4).tolist())


# ---- encoder output: are real events inside what training produced?
def ctx(x, ab):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            c = model.encode(torch.from_numpy(x[i:i + 32]).float(), asd_bands=torch.from_numpy(ab[i:i + 32]).float())
            c = c[0] if isinstance(c, tuple) else c
            out.append(c.numpy().reshape(len(x[i:i + 32]), -1))
    return np.concatenate(out)


keep = np.array([n > 0 for n in NS])
ct, cr = ctx(S[keep], AB[keep]), ctx(X, RAB)
n_fit = len(ct) // 2
mu, cov = ct[:n_fit].mean(0), np.cov(ct[:n_fit].T) + 1e-3 * np.eye(ct.shape[1])
inv = np.linalg.pinv(cov)
md = lambda c: np.sqrt(np.einsum("ij,jk,ik->i", c - mu, inv, c - mu))
ref = md(ct[n_fit:])
mr = md(cr)
print(f"CONTEXT dim {ct.shape[1]} Mahalanobis: held-out training 50/95/99% {np.quantile(ref,.5):.1f}/{np.quantile(ref,.95):.1f}/{np.quantile(ref,.99):.1f}; "
      f"real 50/95% {np.quantile(mr,.5):.1f}/{np.quantile(mr,.95):.1f}; real above training 95%: {(mr>np.quantile(ref,.95)).mean():.0%}, above 99%: {(mr>np.quantile(ref,.99)).mean():.0%}")
worst = np.argsort(-mr)[:5]
print("  farthest real events:", [(names[i][:8], round(float(mr[i]), 1), present[i]) for i in worst])
