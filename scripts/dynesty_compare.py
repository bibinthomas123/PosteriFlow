#!/usr/bin/env python3
"""
LeanNPE vs Bilby/dynesty on freshly generated single-BBH events.

Uses the SAME pipeline components as dataset generation (PSDs, injector,
whitener), so both inference methods see identical data:
  - dynesty gets the unwhitened strain + PSD (standard bilby likelihood,
    phase/distance/time marginalized for tractable runtimes, posteriors
    reconstructed by bilby afterwards);
  - LeanNPE gets the whitened strain (its training representation).

Outputs per event: both posterior sample sets + summary comparison
(median offsets in sigma, width ratios, 1-D JS divergences).

Pilot mode (--n_events 1) reports wall-time per event so the batch size can
be chosen against the compute budget.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import torch

import bilby
from ahsd.data.bilby_pipeline import (BilbyNoiseGenerator, BilbySignalInjector,
                                      BilbyPreprocessor, get_default_psd)
from ahsd.data.parameter_sampler import ParameterSampler
from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES

DETS = ["H1", "L1", "V1"]
SR, DUR = 4096, 4.0
# Injection epoch: the dataset pipeline evaluates antenna patterns at the O4
# reference time (parameter_sampler.GPS_REF). The likelihood's interferometers
# MUST use the same epoch or RA is rotated by an arbitrary sidereal phase
# (measured: predicted GMST shift 5.586 rad vs observed RA offset 5.570 rad).
GPS_REF = 1369224018.0


def make_event(sampler, noise_gen, injector, prep, psds, seed, min_snr=8.0):
    """Draw a single BBH event; return unwhitened strains, whitened strains,
    params, per-detector SNR."""
    rng = np.random.default_rng(seed)
    while True:
        p = sampler.sample_parameters()
        if p.get("event_type") != "BBH":
            continue
        raw, white, snrs = {}, {}, []
        for i, det in enumerate(DETS):
            noise = noise_gen.generate(det, psds[det], seed=int(rng.integers(2**31)))
            strain, s = injector.inject(noise, p, det, psds[det])
            raw[det] = strain
            white[det] = prep.preprocess(strain, psds[det], detector=det)
            snrs.append(s)
        net = float(np.sqrt(sum(s**2 for s in snrs)))
        if net >= min_snr:
            return raw, white, p, net


def run_dynesty(raw, psds, injection, outdir, label, nlive=300):
    ifos = bilby.gw.detector.InterferometerList([])
    for det in DETS:
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=np.asarray(psds[det]["frequencies"], dtype=float),
            psd_array=np.asarray(psds[det]["psd"], dtype=float))
        ifo.strain_data.set_from_time_domain_strain(
            raw[det].astype(np.float64), sampling_frequency=SR, duration=DUR,
            start_time=GPS_REF - DUR / 2)
        ifo.minimum_frequency = 20.0
        ifos.append(ifo)

    wfg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=SR,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={"waveform_approximant": "IMRPhenomXP",
                            "reference_frequency": 50.0, "minimum_frequency": 20.0})

    priors = bilby.gw.prior.BBHPriorDict()
    # match the training prior support; aligned spins (tilts fixed at 0)
    priors["mass_1"] = bilby.core.prior.Uniform(5, 100, "mass_1")
    priors["mass_2"] = bilby.core.prior.Uniform(5, 100, "mass_2")
    priors["mass_ratio"].maximum = 1.0  # overridden by component masses below
    del priors["chirp_mass"], priors["mass_ratio"]
    priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        50, 2100, name="luminosity_distance")
    priors["a_1"] = bilby.core.prior.Uniform(0, 0.99, "a_1")
    priors["a_2"] = bilby.core.prior.Uniform(0, 0.99, "a_2")
    priors["tilt_1"] = bilby.core.prior.DeltaFunction(0.0)
    priors["tilt_2"] = bilby.core.prior.DeltaFunction(0.0)
    priors["phi_12"] = bilby.core.prior.DeltaFunction(0.0)
    priors["phi_jl"] = bilby.core.prior.DeltaFunction(0.0)
    priors["geocent_time"] = bilby.core.prior.Uniform(
        GPS_REF + injection["geocent_time"] - 0.2,
        GPS_REF + injection["geocent_time"] + 0.2, "geocent_time")

    like = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=wfg, priors=priors,
        phase_marginalization=True, distance_marginalization=True,
        time_marginalization=True)

    res = bilby.run_sampler(
        likelihood=like, priors=priors, sampler="dynesty", nlive=nlive,
        dlogz=0.5, sample="rwalk", walks=60, naccept=20,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        outdir=str(outdir), label=label, resume=False, clean=True,
        save=True, plot=False, npool=4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/lean_npe_v2/best_model.pth")
    ap.add_argument("--outdir", default="analysis/dynesty_compare")
    ap.add_argument("--n_events", type=int, default=1)
    ap.add_argument("--nlive", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = LeanNPE(premerger=ckpt["args"].get("premerger", False))
    model.load_state_dict(ckpt["model_state_dict"]); model.to(device).eval()

    sampler = ParameterSampler()
    noise_gen = BilbyNoiseGenerator(SR, DUR)
    injector = BilbySignalInjector(SR, DUR)
    prep = BilbyPreprocessor(SR, DUR)
    psds = {d: get_default_psd(d) for d in DETS}

    summaries = []
    for k in range(args.n_events):
        raw, white, p, net = make_event(sampler, noise_gen, injector, prep, psds,
                                        seed=args.seed + k)
        truth = {n: p[n] for n in PARAM_NAMES}
        print(f"\n=== event {k}: SNR={net:.1f} m1={p['mass_1']:.1f} m2={p['mass_2']:.1f} "
              f"dL={p['luminosity_distance']:.0f} ===")

        # LeanNPE (seconds)
        t0 = time.time()
        st = torch.from_numpy(np.stack([white[d] for d in DETS])).unsqueeze(0).to(device).float()
        with torch.no_grad():
            npe_samples = model.sample_posterior(st, rank=0, n_samples=3000)[0].cpu().numpy()
        t_npe = time.time() - t0
        np.save(out / f"event{k}_npe_samples.npy", npe_samples)

        # dynesty (the expensive part)
        inj = dict(mass_1=p["mass_1"], mass_2=p["mass_2"],
                   luminosity_distance=p["luminosity_distance"], ra=p["ra"], dec=p["dec"],
                   theta_jn=p["theta_jn"], psi=p["psi"], phase=p["phase"],
                   geocent_time=p["geocent_time"], a_1=p["a1"], a_2=p["a2"])
        t0 = time.time()
        res = run_dynesty(raw, psds, inj, out / f"event{k}_bilby", f"ev{k}", nlive=args.nlive)
        t_dyn = time.time() - t0

        post = res.posterior
        bilby_names = {"mass_1": "mass_1", "mass_2": "mass_2",
                       "luminosity_distance": "luminosity_distance", "ra": "ra", "dec": "dec",
                       "theta_jn": "theta_jn", "psi": "psi", "phase": "phase",
                       "geocent_time": "geocent_time", "a1": "a_1", "a2": "a_2"}
        comp = {"event": k, "snr": net, "truth": truth,
                "t_leannpe_s": round(t_npe, 2), "t_dynesty_s": round(t_dyn, 1)}
        # convention alignment before comparing:
        #  - sort masses per draw (training/dataset enforce mass_1 >= mass_2,
        #    the dynesty priors do not)
        #  - dynesty geocent_time is absolute GPS; LeanNPE's is relative to GPS_REF
        m1v, m2v = post["mass_1"].values.copy(), post["mass_2"].values.copy()
        post["mass_1"], post["mass_2"] = np.maximum(m1v, m2v), np.minimum(m1v, m2v)
        post["geocent_time"] = post["geocent_time"].values - GPS_REF
        for j, name in enumerate(PARAM_NAMES):
            b = post[bilby_names[name]].values
            n_ = npe_samples[:, j]
            comp[name] = {
                "truth": float(truth[name]),
                "dynesty_med": float(np.median(b)), "dynesty_std": float(b.std()),
                "npe_med": float(np.median(n_)), "npe_std": float(n_.std()),
                "med_offset_sigma": float((np.median(n_) - np.median(b)) / (b.std() + 1e-9)),
                "width_ratio_npe_over_dynesty": float(n_.std() / (b.std() + 1e-9)),
            }
        summaries.append(comp)
        with open(out / "comparison.json", "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"LeanNPE: {t_npe:.1f}s | dynesty: {t_dyn/60:.1f} min")

    print(f"\nwrote {out}/comparison.json")


if __name__ == "__main__":
    main()
