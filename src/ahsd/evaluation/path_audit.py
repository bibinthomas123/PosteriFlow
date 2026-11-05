"""Injections through the real-event path, for certification.

Simulated validation shares the training pipeline, so it cannot see a mismatch
between training and real events: 2026-11-05 found training signals 1.4x too loud
(distance read 1.4x far on every real event) and right ascension rotated by the
Earth's motion since the training epoch, both with every gate green. These checks
inject known signals into real detector noise and read them back through the same
path a real event takes.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

SR = 4096
DETS = ("H1", "L1", "V1")
AMP_RANGE = (0.90, 1.15)      # training / inference amplitude, same signal
RA_MAX = 0.5                  # rad, median |error| of the posterior's circular mean
DIST_RANGE = (0.75, 1.25)     # median posterior distance / truth, training prior


def detector_signal(params: Dict, det: str, gps: float, approximant: str = "IMRPhenomXP") -> np.ndarray:
    """Detector strain on an 8 s grid, geocentre merger at the centre."""
    import bilby
    wfg = bilby.gw.WaveformGenerator(
        duration=8.0, sampling_frequency=SR,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={"waveform_approximant": approximant,
                            "reference_frequency": 50.0, "minimum_frequency": 20.0})
    pol = wfg.time_domain_strain(dict(params, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0,
                                      phi_12=0.0, phi_jl=0.0, geocent_time=gps))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    h = sum(ifo.antenna_response(params["ra"], params["dec"], gps, params["psi"], m) * pol[m]
            for m in ("plus", "cross"))
    dt = ifo.time_delay_from_geocenter(params["ra"], params["dec"], gps)
    return np.roll(h, 4 * SR + int(round(dt * SR)))


def training_path_signal(h4: np.ndarray, det: str, asd: np.ndarray,
                         design_dir: str = "data/noise_bank") -> np.ndarray:
    """A 4 s signal as RemixDataset builds it: design whitening, recolouring into
    the segment's measured ASD, filter gain included."""
    from ahsd.data.bilby_pipeline import BilbyPreprocessor, get_default_psd, highpass_gain
    from ahsd.inference.preprocessing import _design_asd
    sig = BilbyPreprocessor(SR, 4.0).preprocess(h4, get_default_psd(det), detector=det).astype(np.float64)
    design = _design_asd(det, design_dir)
    n = min(len(design), len(asd))
    filt = np.ones(len(h4) // 2 + 1)
    filt[:n] = np.clip(design[:n] / np.maximum(asd[:n], 1e-30), 1 / 50, 50)
    gain = highpass_gain(np.fft.rfftfreq(len(h4), 1 / SR), SR)
    return np.fft.irfft(np.fft.rfft(sig) * filt * gain, n=len(h4))


def real_path_injections(model: str, sources: List[Tuple[float, float, float]], device: str = "cpu",
                         n_per_event: int = 2, target_snr: float = 15.0, num_samples: int = 2000,
                         seed: int = 0) -> Dict:
    """sources: (gps, m1_det, m2_det) of events whose data provides the noise; the
    injections go 20 s before each event. Returns per-injection amplitude ratios,
    ra errors and distance ratios, with the medians the gates read."""
    from ahsd.inference.pipeline import infer
    from ahsd.inference.preprocessing import fetch_gwosc, prepare_real
    from ahsd.models.lean_npe import PARAM_NAMES as N

    rng = np.random.default_rng(seed)
    amp, ra_err, dist = [], [], []
    for gps, m1, m2 in sources:
        g, raw, starts, srs, _ = fetch_gwosc(gps, strict=False)
        t = g - 20.0
        noise = prepare_real(raw, t, starts, srs, seed=0)
        dets = list(noise.detectors_present)
        for k in range(n_per_event):
            p = dict(mass_1=m1, mass_2=m2, luminosity_distance=1000.0, ra=float(rng.uniform(0, 2 * np.pi)),
                     dec=float(np.arcsin(rng.uniform(-1, 1))), theta_jn=float(np.arccos(rng.uniform(-1, 1))),
                     psi=float(rng.uniform(0, np.pi)), phase=float(rng.uniform(0, 2 * np.pi)))
            hs = {d: detector_signal(p, d, t) for d in dets}

            def inject(scale):
                r = {d: np.array(raw[d], dtype=np.float64) for d in raw}
                for d in dets:
                    i0 = int(round((t - starts[d]) * SR)) - 2 * 8192
                    r[d][i0:i0 + len(hs[d])] += hs[d] * scale
                return prepare_real(r, t, starts, srs, seed=0)

            first = inject(1.0)
            l2 = np.sqrt(sum(((first.strain[DETS.index(d)] - noise.strain[DETS.index(d)]) ** 2).sum() for d in dets))
            scale = target_snr / float(l2)
            prep = inject(scale)
            for d in dets:
                i = DETS.index(d)
                a = (prep.strain[i] - noise.strain[i]).astype(np.float64)
                if np.sqrt((a ** 2).sum()) > 4:
                    b = training_path_signal((hs[d] * scale)[8192:8192 + 16384], d, noise.asds[d])
                    amp.append(float((a * b).sum() / (a * a).sum()))
            s = infer(strain=prep, event_type="BBH", model=model, num_samples=num_samples,
                      device=device, seed=k + 1).samples
            mean_ra = np.angle(np.exp(1j * s[:, N.index("ra")]).mean())
            ra_err.append(float(abs((mean_ra - p["ra"] + np.pi) % (2 * np.pi) - np.pi)))
            dist.append(float(np.median(s[:, N.index("luminosity_distance")]) / (1000.0 / scale)))
    return {"n": len(ra_err), "amplitude": float(np.median(amp)), "ra_error": float(np.median(ra_err)),
            "distance": float(np.median(dist))}


def path_gates(r: Dict) -> List[Tuple[str, bool, str]]:
    """(name, passed, detail) for run_gates."""
    return [
        (f"training vs real-event signal amplitude within {AMP_RANGE[0]}-{AMP_RANGE[1]}",
         AMP_RANGE[0] <= r["amplitude"] <= AMP_RANGE[1], f"{r['amplitude']:.3f} (projection, n={r['n']} injections)"),
        (f"right ascension on real-path injections, median error < {RA_MAX} rad",
         r["ra_error"] < RA_MAX, f"{r['ra_error']:.3f} rad"),
        (f"distance on real-path injections, median posterior/truth within {DIST_RANGE[0]}-{DIST_RANGE[1]}",
         DIST_RANGE[0] <= r["distance"] <= DIST_RANGE[1], f"{r['distance']:.3f}x truth"),
    ]
