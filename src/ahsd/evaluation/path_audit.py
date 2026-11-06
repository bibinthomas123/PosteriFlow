"""Injections through the real-event path, for certification.

Simulated validation shares the training pipeline, so it cannot see a mismatch
between training and real events: 2026-11-05 found training signals 1.4x too loud
(distance read 1.4x far on every real event) and right ascension rotated by the
Earth's motion since the training epoch, both with every gate green. These checks
inject known signals into real detector noise and read them back through the same
path a real event takes.

Sky gates (2026-11-06, analysis/distance_realdata/audit_bns_sky2.py). BNS failed at
1.09 rad with validation at 0.20. On GW170817 at +32 s the same injections gave 0.27
rad with all detectors and 1.09 without L1, and the posteriors stayed calibrated: the
gate was scoring injections no network could localize. So:
  error of the mean ra only where a second detector carries the signal; a broad
  posterior has no meaningful mean, and "two detectors have data" does not ensure that
  searched fraction on every injection, which a wrong sky fails even when broad
  context-stream models get the inspiral through their 28 s stream, with that stream
  clear of the event's own signal and glitch
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

SR = 4096
DETS = ("H1", "L1", "V1")
AMP_RANGE = (0.90, 1.15)      # training / inference amplitude, same signal
RA_MAX = 0.5                  # rad, median |error| of the posterior's circular mean
DIST_RANGE = (0.75, 1.25)     # distance-weighted median of posterior distance / truth
SKY_MIN_SNR = 5.0             # second-loudest detector, for an injection to count in RA_MAX
SKY_MIN_N = 4                 # injections that must count, or the sky is unmeasured and fails
SEARCHED_TOL = 0.2            # |mean searched fraction - 0.5|; 2.4 sigma at 12 injections, 3.1 at 20
SKY_GRID = (36, 18)           # ra, sin(dec); finer leaves a broad posterior's cells empty
CONTEXT_BEFORE_S = 34         # the 2 s of window, 28 s of context, 4 s spare
BBH_OFFSET_S = -20.0          # before the event
NS_OFFSET_S = 32.0            # after: the 28 s context stream starts 2 s past the event


# seconds of waveform generated per class: long enough not to wrap from 20 Hz
# (BNS about 135 s, NSBH about 18 s, see bilby_pipeline)
_GEN_SECONDS = {"BBH": 8, "NSBH": 128, "BNS": 256}
TAPER_S = 1.0     # start of a cropped inspiral, so the cut adds no broadband transient
# GWOSC 4 kHz data is low-passed from about 1700 Hz (ASD down to 1-3% by 1900 Hz). A real
# signal carries that filter; without it an injection divided by the measured ASD had
# 54-70% of its BNS power above 1900 Hz, so "SNR 15" was really quieter and the amplitude
# check read 0.78 (analysis/distance_realdata/audit_bns_hf.py, 2026-11-06).
LOWPASS_HZ = (1700.0, 1900.0)


def data_lowpass(h: np.ndarray) -> np.ndarray:
    """Zero-phase cosine roll-off over LOWPASS_HZ, standing in for the data's filter."""
    f = np.fft.rfftfreq(len(h), 1.0 / SR)
    lo, hi = LOWPASS_HZ
    w = np.where(f <= lo, 1.0, np.where(f >= hi, 0.0, 0.5 * (1 + np.cos(np.pi * (f - lo) / (hi - lo)))))
    return np.fft.irfft(np.fft.rfft(h) * w, n=len(h))


def detector_signal(params: Dict, det: str, gps: float, approximant: str = None,
                    event_type: str = "BBH", before_s: int = 4) -> np.ndarray:
    """Detector strain from before_s before the geocentre merger to 4 s after. A signal
    longer than 8 s is generated in full and cut, not wrapped, with a tapered start;
    models reading the 28 s context stream need before_s >= 34 to see their inspiral."""
    import bilby
    from ahsd.data.bilby_pipeline import _get_approximant, _get_source_model
    D = _GEN_SECONDS[event_type]
    source, conversion = _get_source_model({"event_type": event_type})
    if approximant is None:
        approximant = "IMRPhenomXP" if event_type == "BBH" else _get_approximant({"event_type": event_type})
    wfg = bilby.gw.WaveformGenerator(
        duration=float(D), sampling_frequency=SR,
        frequency_domain_source_model=source, parameter_conversion=conversion,
        waveform_arguments={"waveform_approximant": approximant,
                            "reference_frequency": 50.0, "minimum_frequency": 20.0})
    extra = {} if event_type == "BBH" else {"lambda_1": 0.0, "lambda_2": 0.0}
    pol = wfg.time_domain_strain(dict(params, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0,
                                      phi_12=0.0, phi_jl=0.0, geocent_time=gps, **extra))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    h = sum(ifo.antenna_response(params["ra"], params["dec"], gps, params["psi"], m) * pol[m]
            for m in ("plus", "cross"))
    dt = ifo.time_delay_from_geocenter(params["ra"], params["dec"], gps)
    h = data_lowpass(np.roll(h, D * SR // 2 + int(round(dt * SR))))
    if D == 8:
        return h
    h = h[(D // 2 - before_s) * SR:(D // 2 + 4) * SR].copy()
    n = int(TAPER_S * SR)
    h[:n] *= np.sin(0.5 * np.pi * np.arange(n) / n) ** 2
    return h


def training_path_signal(h4: np.ndarray, det: str, asd: np.ndarray,
                         design_dir: str = "data/noise_bank", v1_design: str = "flat") -> np.ndarray:
    """A 4 s signal as RemixDataset builds it: design whitening, recolouring into
    the segment's measured ASD, filter gain included."""
    from ahsd.data.bilby_pipeline import BilbyPreprocessor, get_default_psd, highpass_gain
    from ahsd.inference.preprocessing import _design_asd
    sig = BilbyPreprocessor(SR, 4.0).preprocess(h4, get_default_psd(det, v1_design), detector=det).astype(np.float64)
    design = _design_asd(det, design_dir, v1_design)
    n = min(len(design), len(asd))
    filt = np.ones(len(h4) // 2 + 1)
    filt[:n] = np.clip(design[:n] / np.maximum(asd[:n], 1e-30), 1 / 50, 50)
    gain = highpass_gain(np.fft.rfftfreq(len(h4), 1 / SR), SR)
    return np.fft.irfft(np.fft.rfft(sig) * filt * gain, n=len(h4))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median of `values` with non-negative `weights`, interpolated between samples."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(v)
    c = np.cumsum(w[order]) - 0.5 * w[order]
    return float(np.interp(0.5 * w.sum(), c, v[order]))


def searched_fraction(ra_s: np.ndarray, dec_s: np.ndarray, ra: float, dec: float) -> float:
    """Posterior mass denser than the truth's sky position. Uniform on [0, 1] for a
    calibrated posterior, whatever its width; near 1 when it excludes the truth.

    The density comes from half the samples and the mass from the other half: read off
    one histogram, the noise in the counts ranks some cells too dense and the mean is
    0.60 on a uniform sky with 5000 samples."""
    nra, nsd = SKY_GRID

    def cell(r, d):
        i = np.minimum((np.mod(r, 2 * np.pi) / (2 * np.pi) * nra).astype(int), nra - 1)
        j = np.minimum(((np.sin(d) + 1) / 2 * nsd).astype(int), nsd - 1)
        return i * nsd + j

    density = np.bincount(cell(ra_s[::2], dec_s[::2]), minlength=nra * nsd)
    others = density[cell(ra_s[1::2], dec_s[1::2])]
    at_truth = density[cell(np.asarray(ra), np.asarray(dec))]
    return float((others > at_truth).mean() + 0.5 * (others == at_truth).mean())


def real_path_injections(model: str, sources: List[Tuple[float, float, float]], device: str = "cpu",
                         n_per_event: int = 2, target_snr: float = 15.0, num_samples: int = 5000,
                         seed: int = 0, event_type: str = "BBH") -> Dict:
    """sources: (gps, m1_det, m2_det) of events whose data provides the noise. BBH
    injections go before each event; BNS and NSBH after, because their own inspiral is
    still in band before it. Returns the medians the gates read, with sky error only
    from injections a second detector carries.

    Distance is a distance-weighted median. The injections have isotropic angles at fixed
    SNR, which sets each one's true distance. Under the uniform distance prior such events
    occur in proportion to that distance (dd/dSNR ~ d), so an ideal posterior reads about
    1.19 x truth unweighted (spread 0.74-1.85) and 1.00 weighted; measured 2026-11-06 on BNS
    and NSBH, the unweighted median was 1.18 over 8 seeds, from 0.87 to 1.43 per seed."""
    from ahsd.inference.pipeline import infer, load_model
    from ahsd.inference.preprocessing import fetch_gwosc, prepare_real
    from ahsd.models.lean_npe import PARAM_NAMES as N

    rng = np.random.default_rng(seed)
    amp, ra_err, searched, dist, truth_d, rows = [], [], [], [], [], []
    offset = BBH_OFFSET_S if event_type == "BBH" else NS_OFFSET_S
    net = load_model(model, device=device)[0]
    before_s = CONTEXT_BEFORE_S if getattr(net, "lowrate_len", 0) else 4
    v1_design = getattr(net, "v1_design", "flat")
    for gps, m1, m2 in sources:
        g, raw, starts, srs, _ = fetch_gwosc(gps, strict=False)
        t = g + offset
        noise = prepare_real(raw, t, starts, srs, seed=0)
        dets = list(noise.detectors_present)
        for k in range(n_per_event):
            p = dict(mass_1=m1, mass_2=m2, luminosity_distance=1000.0, ra=float(rng.uniform(0, 2 * np.pi)),
                     dec=float(np.arcsin(rng.uniform(-1, 1))), theta_jn=float(np.arccos(rng.uniform(-1, 1))),
                     psi=float(rng.uniform(0, np.pi)), phase=float(rng.uniform(0, 2 * np.pi)))
            hs = {d: detector_signal(p, d, t, event_type=event_type, before_s=before_s) for d in dets}

            def inject(scale):
                r = {d: np.array(raw[d], dtype=np.float64) for d in raw}
                for d in dets:
                    # every signal ends 4 s after the merger
                    i0 = int(round((t + 4.0 - starts[d]) * SR)) - len(hs[d])
                    r[d][i0:i0 + len(hs[d])] += hs[d] * scale
                return prepare_real(r, t, starts, srs, seed=0)

            first = inject(1.0)
            l2 = np.sqrt(sum(((first.strain[DETS.index(d)] - noise.strain[DETS.index(d)]) ** 2).sum() for d in dets))
            scale = target_snr / float(l2)
            prep = inject(scale)
            per = []
            for d in dets:
                i = DETS.index(d)
                a = (prep.strain[i] - noise.strain[i]).astype(np.float64)
                per.append(float(np.sqrt((a ** 2).sum())))
                if per[-1] > 4:
                    b = training_path_signal((hs[d] * scale)[-6 * SR:-2 * SR], d, noise.asds[d], v1_design=v1_design)
                    amp.append(float((a * b).sum() / (a * a).sum()))
            s = infer(strain=prep, event_type=event_type, model=model, num_samples=num_samples,
                      device=device, seed=k + 1).samples
            ra_s, dec_s = s[:, N.index("ra")], s[:, N.index("dec")]
            searched.append(searched_fraction(ra_s, dec_s, p["ra"], p["dec"]))
            z = np.exp(1j * ra_s).mean()
            err = float(abs((np.angle(z) - p["ra"] + np.pi) % (2 * np.pi) - np.pi))
            if len(per) >= 2 and sorted(per)[-2] >= SKY_MIN_SNR:
                ra_err.append(err)
            dist.append(float(np.median(s[:, N.index("luminosity_distance")]) / (1000.0 / scale)))
            truth_d.append(1000.0 / scale)
            rows.append({"event": float(g), "k": k, "ra": p["ra"], "dec": p["dec"], "snr": dict(zip(dets, per)),
                         "ra_error": err, "concentration": float(abs(z)), "searched": searched[-1],
                         "distance": dist[-1], "truth_distance": truth_d[-1]})
    return {"n": len(dist), "n_sky": len(ra_err), "amplitude": float(np.median(amp)),
            "ra_error": float(np.median(ra_err)) if ra_err else float("nan"),
            "searched": float(np.mean(searched)),
            "searched_weighted": float(np.average(searched, weights=truth_d)),
            "distance": weighted_median(dist, truth_d), "distance_unweighted": float(np.median(dist)),
            "n_eff": float(np.sum(truth_d) ** 2 / np.sum(np.square(truth_d))), "injections": rows}


def path_gates(r: Dict) -> List[Tuple[str, bool, str]]:
    """(name, passed, detail) for run_gates."""
    return [
        (f"training vs real-event signal amplitude within {AMP_RANGE[0]}-{AMP_RANGE[1]}",
         AMP_RANGE[0] <= r["amplitude"] <= AMP_RANGE[1], f"{r['amplitude']:.3f} (projection, n={r['n']} injections)"),
        (f"right ascension on real-path injections with second detector SNR >= {SKY_MIN_SNR:g}, median error < {RA_MAX} rad",
         r["n_sky"] >= SKY_MIN_N and r["ra_error"] < RA_MAX,
         f"{r['ra_error']:.3f} rad, {r['n_sky']} of {r['n']} injections count (at least {SKY_MIN_N} needed)"),
        (f"sky calibration on real-path injections, mean searched fraction 0.5 +- {SEARCHED_TOL}",
         abs(r["searched"] - 0.5) < SEARCHED_TOL,
         f"{r['searched']:.3f}, distance-weighted {r.get('searched_weighted', float('nan')):.3f}, n={r['n']}"),
        (f"distance on real-path injections, distance-weighted median posterior/truth within {DIST_RANGE[0]}-{DIST_RANGE[1]}",
         DIST_RANGE[0] <= r["distance"] <= DIST_RANGE[1],
         f"{r['distance']:.3f}x truth, unweighted {r.get('distance_unweighted', float('nan')):.3f}, effective n {r.get('n_eff', float('nan')):.0f}"),
    ]
