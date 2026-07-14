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

from ahsd.data.bilby_pipeline import (BilbyNoiseGenerator, BilbySignalInjector,
                                      BilbyPreprocessor, get_default_psd)
from ahsd.data.parameter_sampler import ParameterSampler
from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES
from ahsd.inference.dynesty_bridge import (GPS_REF, run_dynesty as _run_dynesty,
                                           align_conventions)

DETS = ["H1", "L1", "V1"]
SR, DUR = 4096, 4.0


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
    # single implementation lives in ahsd.inference.dynesty_bridge
    return _run_dynesty(raw, psds, outdir, label, center_gps=GPS_REF,
                        trigger_offset=injection["geocent_time"], nlive=nlive)


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
    model = LeanNPE(premerger=ckpt["args"].get("premerger", False), psd_cond=ckpt["args"].get("psd_cond", False) or False, psd_bands=ckpt["args"].get("psd_bands", 16), encoder_type=ckpt["args"].get("encoder_type", "conv"))
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
        # convention alignment before comparing (mass sort + relative time)
        post = align_conventions(post, GPS_REF)
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
