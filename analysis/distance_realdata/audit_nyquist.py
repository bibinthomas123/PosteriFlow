"""Does the falling ASD near Nyquist reach training or the likelihood? (2026-11-06)

GWOSC 4 kHz data is low-passed: above about 1700 Hz the measured ASD falls to 1-3% of its
300 Hz level. A real signal is filtered with the noise, so whitening leaves it alone. An
unfiltered template or injection divided by that ASD is inflated (audit_bns_hf.py: 54-70%
of a BNS injection's power at 1900-2049 Hz).

Measured here:
  training  stored signals recoloured into real noise as RemixDataset does: power per
            band and how much of the norm (the SNR) sits above 1700 Hz, per class
  bank      the recolouring filter near Nyquist
  likelihood  the unfiltered template's optimal SNR against the PSD the correction
            uses, with fmax 2048 and fmax 1700
"""
import sys

import numpy as np

sys.path.insert(0, "experiments")
from remix_data import RemixDataset

T = 16384
SR = 4096
f = np.fft.rfftfreq(T, 1 / SR)
EDGES = (20, 160, 320, 640, 1000, 1400, 1700, 1900, 2000, 2049)
MASK = [(f >= a) & (f < b) for a, b in zip(EDGES[:-1], EDGES[1:])]
print("bands", list(zip(EDGES[:-1], EDGES[1:])))

for cls, data in (("BNS", "data/dataset_bns_v4"), ("NSBH", "data/dataset_nsbh_v4"), ("BBH", "data/dataset_bbh_v6")):
    ds = RemixDataset(f"{data}/memmap/validation", remix=False, seed=1, real_noise_dir="data/noise_bank", real_noise_prob=1.0,
                      return_asd_bands=True, psd_bands=16, single_signal=True)
    rng = np.random.default_rng(0)
    shares, hi, filt_hi = [], [], []
    for k in rng.choice(ds._index_map, 300, replace=False):
        start, _ = ds.events[int(k)]
        sig = np.asarray(ds.signals[start][:, :T], np.float64)
        _, filts = ds._real_noise_and_filters(rng)
        F = np.fft.rfft(sig, axis=-1) * np.stack(filts) * ds._hp_gain
        d = int(np.argmax((np.abs(F) ** 2).sum(-1)))
        P = np.abs(F[d]) ** 2
        if P.sum() == 0:
            continue
        shares.append([P[m].sum() / P.sum() for m in MASK])
        hi.append(P[f >= 1700].sum() / P.sum())
        filt_hi.append(float(np.median(np.asarray(filts[d])[f >= 1900])))
    print(f"{cls} recoloured, loudest detector, n={len(shares)}: median {np.median(shares, 0).round(4).tolist()}", flush=True)
    print(f"   power above 1700 Hz: median {np.median(hi):.4f}, 90% {np.quantile(hi, .9):.4f}, max {np.max(hi):.4f}; SNR inflation at the median {1 / np.sqrt(1 - np.median(hi)):.3f}; "
          f"recolouring filter above 1900 Hz: median {np.median(filt_hi):.2f}", flush=True)

from ahsd.evaluation.path_audit import NS_OFFSET_S, detector_signal
from ahsd.inference.preprocessing import fetch_gwosc, prepare_real

for cls, m1, m2 in (("BNS", 1.47, 1.28), ("NSBH", 6.0, 1.4), ("BBH", 35.0, 30.0)):
    g, raw, starts, srs, _ = fetch_gwosc(1187008882.4, strict=False)
    prep = prepare_real(raw, g + NS_OFFSET_S, starts, srs, seed=0)
    p = dict(mass_1=m1, mass_2=m2, luminosity_distance=100.0, ra=1.0, dec=0.3, theta_jn=0.5, psi=0.2, phase=0.1)
    for d in prep.detectors_present[:2]:
        h = detector_signal(p, d, g + NS_OFFSET_S, event_type=cls)[-6 * SR:-2 * SR]
        H = np.fft.rfft(h) / SR
        psd = np.interp(f, prep.psds[d]["frequencies"], prep.psds[d]["psd"])
        rho = lambda fmax: np.sqrt(4 * (np.abs(H[(f >= 20) & (f < fmax)]) ** 2 / psd[(f >= 20) & (f < fmax)]).sum() / (T / SR))
        print(f"likelihood {cls} {d}: optimal SNR, fmax 2048 {rho(2049):.1f}, fmax 1700 {rho(1700):.1f}, ratio {rho(2049) / rho(1700):.3f}", flush=True)
