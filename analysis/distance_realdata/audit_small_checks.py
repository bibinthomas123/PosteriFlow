"""Two follow-ups to audit_nyquist.py (2026-11-06).

Light BBH: how much did the likelihood's 2048 Hz limit inflate them? audit_nyquist.py read
the ratio off a 4 s time-domain crop, whose edges leak power. Here it is the frequency-
domain waveform, as the likelihood sees it, against GW170817's and GW200115's measured PSDs,
for BNS and NSBH too, so that x1.5 and x1.3 get checked the same way.

Rare recoloured training signals: a few reach 40% of their power above 1700 Hz. What are
they? For the events above 10%: SNR, chirp mass, and the recolouring filter in and out of band.
"""
import sys

import numpy as np

sys.path.insert(0, "experiments")
import bilby
from ahsd.data.bilby_pipeline import _get_approximant, _get_source_model
from ahsd.inference.preprocessing import fetch_gwosc, prepare_real
from remix_data import RemixDataset

T = 16384
SR = 4096
DUR = 32.0
f4 = np.fft.rfftfreq(T, 1 / SR)

print("LIKELIHOOD: frequency-domain optimal SNR, 20 Hz to the limit", flush=True)
CASES = (("BNS", 1.47, 1.28), ("NSBH", 8.9, 1.9), ("NSBH", 6.0, 1.4), ("BBH", 8.0, 5.0), ("BBH", 12.0, 7.0),
         ("BBH", 14.0, 8.0), ("BBH", 20.0, 15.0), ("BBH", 35.0, 30.0))
for name, gps in (("GW170817 O2", 1187008882.4), ("GW200115 O3b", 1263097407.7)):
    g, raw, starts, srs, _ = fetch_gwosc(gps, strict=False)
    prep = prepare_real(raw, g + 32.0, starts, srs, seed=0)
    for cls, m1, m2 in CASES:
        source, conv = _get_source_model({"event_type": cls})
        approx = "IMRPhenomXP" if cls == "BBH" else _get_approximant({"event_type": cls})
        wfg = bilby.gw.WaveformGenerator(duration=DUR, sampling_frequency=SR, frequency_domain_source_model=source,
                                         parameter_conversion=conv,
                                         waveform_arguments={"waveform_approximant": approx, "reference_frequency": 50.0, "minimum_frequency": 20.0})
        extra = {} if cls == "BBH" else {"lambda_1": 0.0, "lambda_2": 0.0}
        h = wfg.frequency_domain_strain(dict(mass_1=m1, mass_2=m2, luminosity_distance=100.0, theta_jn=0.5, phase=0.1, psi=0.2, ra=1.0, dec=0.3,
                                             geocent_time=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, **extra))["plus"]
        f = wfg.frequency_array
        out = []
        for d in prep.detectors_present[:2]:
            psd = np.interp(f, prep.psds[d]["frequencies"], prep.psds[d]["psd"])

            def rho(fmax):
                m = (f >= 20.0) & (f < fmax)
                return float(np.sqrt(4 * (np.abs(h[m]) ** 2 / psd[m]).sum() / DUR))

            out.append(f"{d} {rho(2048.5):7.1f} -> {rho(1700.0):7.1f} (x{rho(2048.5) / rho(1700.0):.3f})")
        print(f"  {name} {cls:4s} {m1:5.1f}+{m2:4.1f}: " + " | ".join(out), flush=True)

print("RARE RECOLOURED SIGNALS", flush=True)
for cls, data in (("BNS", "data/dataset_bns_v4"), ("NSBH", "data/dataset_nsbh_v4"), ("BBH", "data/dataset_bbh_v6")):
    ds = RemixDataset(f"{data}/memmap/validation", remix=False, seed=1, real_noise_dir="data/noise_bank", real_noise_prob=1.0,
                      return_asd_bands=True, psd_bands=16, single_signal=True)
    rng = np.random.default_rng(0)
    rows = []
    for k in rng.choice(ds._index_map, 3000, replace=False):
        start, _ = ds.events[int(k)]
        sig = np.asarray(ds.signals[start][:, :T], np.float64)
        _, filts = ds._real_noise_and_filters(rng)
        F0 = np.fft.rfft(sig, axis=-1)
        fl = np.stack(filts)
        F = F0 * fl * ds._hp_gain
        P = (np.abs(F) ** 2)
        tot = P.sum()
        if tot == 0:
            continue
        hi = f4 >= 1700
        y = np.asarray(ds.params[start])
        d = int(np.argmax(P.sum(-1)))
        band = (f4 >= 30) & (f4 < 300)
        rows.append(dict(share=float(P[:, hi].sum() / tot), snr=float(np.sqrt(2 * tot) / np.sqrt(T)),
                         design_hi=float((np.abs(F0[:, hi]) ** 2).sum() / max((np.abs(F0) ** 2).sum(), 1e-30)),
                         mc=float((y[0] * y[1]) ** .6 / (y[0] + y[1]) ** .2), f_in=float(np.median(fl[d][band])),
                         f_hi=float(np.median(fl[d][f4 >= 1900]))))
    s = np.array([r["share"] for r in rows])
    big = [r for r in rows if r["share"] > .1]
    print(f"{cls}: n={len(rows)}; above 1700 Hz > 10% of power: {len(big)} ({len(big) / len(rows):.1%}), > 1%: {(s > .01).mean():.1%}", flush=True)
    for key, lab in (("snr", "network SNR"), ("mc", "chirp mass"), ("f_in", "filter 30-300 Hz"), ("f_hi", "filter above 1900 Hz")):
        a, b = np.median([r[key] for r in big]) if big else float("nan"), np.median([r[key] for r in rows])
        print(f"   {lab:22s} those events {a:8.3f}, all {b:8.3f}", flush=True)
    print(f"   power above 1700 Hz before recolouring: those events {np.median([r['design_hi'] for r in big]) if big else float('nan'):.2e}, all {np.median([r['design_hi'] for r in rows]):.2e}", flush=True)
