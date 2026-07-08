"""
Bilby/dynesty on the same event, with the SAME conventions as LeanNPE training:
identical PSDs, identical prior support, antenna patterns at the same epoch,
mass_1 >= mass_2 enforced per posterior draw, geocent_time relative to the
window center.

This is the single implementation — scripts/dynesty_compare.py imports from
here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ahsd.inference.preprocessing import DETECTORS, SAMPLE_RATE, DURATION

# Injection epoch of the dataset pipeline (parameter_sampler.GPS_REF). The
# likelihood's interferometers MUST use the same epoch or RA is rotated by an
# arbitrary sidereal phase (measured: predicted GMST shift 5.586 rad vs
# observed RA offset 5.570 rad).
GPS_REF = 1369224018.0

BILBY_NAMES = {"mass_1": "mass_1", "mass_2": "mass_2",
               "luminosity_distance": "luminosity_distance", "ra": "ra", "dec": "dec",
               "theta_jn": "theta_jn", "psi": "psi", "phase": "phase",
               "geocent_time": "geocent_time", "a1": "a_1", "a2": "a_2"}


def build_priors(center_time: float, time_halfwidth: float = 0.2,
                 mass_range=(5.0, 100.0), spin_max: float = 0.99,
                 distance_range=(50.0, 2100.0)):
    """Training-prior-matched priors; aligned spins (tilts fixed at 0)."""
    import bilby
    priors = bilby.gw.prior.BBHPriorDict()
    priors["mass_1"] = bilby.core.prior.Uniform(*mass_range, "mass_1")
    priors["mass_2"] = bilby.core.prior.Uniform(*mass_range, "mass_2")
    priors["mass_ratio"].maximum = 1.0  # overridden by component masses below
    del priors["chirp_mass"], priors["mass_ratio"]
    priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        *distance_range, name="luminosity_distance")
    priors["a_1"] = bilby.core.prior.Uniform(0, spin_max, "a_1")
    priors["a_2"] = bilby.core.prior.Uniform(0, spin_max, "a_2")
    priors["tilt_1"] = bilby.core.prior.DeltaFunction(0.0)
    priors["tilt_2"] = bilby.core.prior.DeltaFunction(0.0)
    priors["phi_12"] = bilby.core.prior.DeltaFunction(0.0)
    priors["phi_jl"] = bilby.core.prior.DeltaFunction(0.0)
    priors["geocent_time"] = bilby.core.prior.Uniform(
        center_time - time_halfwidth, center_time + time_halfwidth, "geocent_time")
    return priors


def run_dynesty(raw: Dict[str, np.ndarray], psds: Dict[str, dict],
                outdir, label: str, center_gps: float = GPS_REF,
                trigger_offset: float = 0.0, nlive: int = 300, npool: int = 4,
                mass_range=(5.0, 100.0), spin_max: float = 0.99,
                distance_range=(50.0, 2100.0), minimum_frequency: float = 20.0,
                approximant: str = "IMRPhenomXP"):
    """Nested sampling on unwhitened 4 s strains keyed by detector.

    center_gps: absolute GPS of the window center (GPS_REF for pipeline-
    generated data, the trigger GPS for real events).
    trigger_offset: expected merger offset from the window center (the
    injection's geocent_time for simulated data; 0 for trigger-centered data).
    Returns the bilby result with posterior in absolute GPS time.
    """
    import bilby
    ifos = bilby.gw.detector.InterferometerList([])
    for det in DETECTORS:
        if det not in raw:
            continue
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=np.asarray(psds[det]["frequencies"], dtype=float),
            psd_array=np.asarray(psds[det]["psd"], dtype=float))
        ifo.strain_data.set_from_time_domain_strain(
            np.asarray(raw[det], dtype=np.float64), sampling_frequency=SAMPLE_RATE,
            duration=DURATION, start_time=center_gps - DURATION / 2)
        ifo.minimum_frequency = minimum_frequency
        ifos.append(ifo)

    wfg = bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLE_RATE,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={"waveform_approximant": approximant,
                            "reference_frequency": 50.0,
                            "minimum_frequency": minimum_frequency})

    priors = build_priors(center_gps + trigger_offset, mass_range=mass_range,
                          spin_max=spin_max, distance_range=distance_range)
    like = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=wfg, priors=priors,
        phase_marginalization=True, distance_marginalization=True,
        time_marginalization=True)

    return bilby.run_sampler(
        likelihood=like, priors=priors, sampler="dynesty", nlive=nlive,
        dlogz=0.5, sample="rwalk", walks=60, naccept=20,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        outdir=str(outdir), label=label, resume=False, clean=True,
        save=True, plot=False, npool=npool)


def align_conventions(posterior, center_gps: float):
    """In-place: sort masses per draw (training convention m1 >= m2) and make
    geocent_time relative to the window center."""
    m1, m2 = posterior["mass_1"].values.copy(), posterior["mass_2"].values.copy()
    posterior["mass_1"], posterior["mass_2"] = np.maximum(m1, m2), np.minimum(m1, m2)
    posterior["geocent_time"] = posterior["geocent_time"].values - center_gps
    return posterior


def run_comparison(result, raw_strain: Dict[str, np.ndarray], outdir: str,
                   injection: Optional[Dict] = None, nlive: int = 300) -> Dict:
    """dynesty on the same event, then per-parameter offset/width comparison
    against the given PosteriorResult. Writes comparison.json under outdir."""
    import json, time
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    center = result.trigger_gps if result.trigger_gps is not None else GPS_REF
    trig_off = 0.0
    if injection and "geocent_time" in injection and result.trigger_gps is None:
        trig_off = float(injection["geocent_time"])

    t0 = time.time()
    res = run_dynesty(raw_strain, result.prepared.psds, out / "bilby", "event",
                      center_gps=center, trigger_offset=trig_off, nlive=nlive)
    t_dyn = time.time() - t0
    post = align_conventions(res.posterior, center)

    comp = {"t_dynesty_s": round(t_dyn, 1),
            "t_leannpe_s": result.diagnostics["runtime"]["total_s"]}
    for j, name in enumerate(result.param_names):
        b = post[BILBY_NAMES[name]].values
        n_ = result.samples[:, j]
        comp[name] = {
            "dynesty_med": float(np.median(b)), "dynesty_std": float(b.std()),
            "npe_med": float(np.median(n_)), "npe_std": float(n_.std()),
            "med_offset_sigma": float((np.median(n_) - np.median(b)) / (b.std() + 1e-9)),
            "width_ratio_npe_over_dynesty": float(n_.std() / (b.std() + 1e-9)),
        }
        if result.truth and name in result.truth:
            comp[name]["truth"] = float(result.truth[name])
    with open(out / "comparison.json", "w") as f:
        json.dump(comp, f, indent=2)
    return comp
