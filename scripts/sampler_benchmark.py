#!/usr/bin/env python3
"""Stochastic samplers on real events, against the LVK posteriors.

Same data, likelihood and priors for every sampler: the event's GWOSC strain,
LVK PSDs, a likelihood segment sized to the signal
(preprocessing.likelihood_seconds), the BBH training prior and the
higher-mode waveform the correction uses (IMRPhenomXHM, phase and distance
marginalized by bilby, time marginalized). Samples are reweighted to LVK's
priors (priors.lvk_log_weights), as the catalogue does for our correction and
for DINGO, and judged against the LVK posterior: |median offset| / LVK sd and
90% width / LVK's for detector-frame chirp mass, mass ratio and distance.

    python scripts/sampler_benchmark.py --event GW190412 --sampler nessai --npool 46
Writes results/sampler_benchmark/<sampler>/<event>.json (+ the bilby run).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULTS = {
    "dynesty": {"nlive": 1000},
    "nessai": {"nlive": 1000},
    "bilby_mcmc": {"nsamples": 2000},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--sampler", choices=list(DEFAULTS), required=True)
    ap.add_argument("--npool", type=int, default=46)
    ap.add_argument("--model", default=str(ROOT / "model/lean_npe_bbh_v5/best_model.pth"))
    ap.add_argument("--lvk", default=str(ROOT / "results/dingo_h2h/lvk"))
    ap.add_argument("--approximant", default="IMRPhenomXHM")
    ap.add_argument("--out", default=str(ROOT / "results/sampler_benchmark"))
    ap.add_argument("--seed", type=int, default=None, help="sampler seed, for repeat runs")
    ap.add_argument("--nlive", type=int, default=None, help="override the default live points")
    a = ap.parse_args()

    from ahsd.inference import infer
    from ahsd.inference.dynesty_bridge import run_dynesty
    from ahsd.inference.preprocessing import fetch_gwosc, likelihood_seconds, prepare_real, raw_segment
    from ahsd.inference.priors import TrainingPrior, chirp_mass, lvk_log_weights, resample_weighted
    from ahsd.models.lean_npe import PARAM_NAMES
    J = {n: i for i, n in enumerate(PARAM_NAMES)}

    lv = Path(a.lvk) / a.event
    if not lv.exists():
        lv = next(Path(a.lvk).glob(a.event + "*"))
    ev_name = lv.name
    need = [p.name.split("_")[0] for p in lv.glob("*_psd.txt")]      # detectors LVK analysed
    gps, raw, starts, srs, _ = fetch_gwosc(ev_name, strict=True, required=need)
    psds = {}
    for det in raw:
        f = lv / f"{det}_psd.txt"
        if f.exists():
            arr = np.loadtxt(f)
            psds[det] = {"frequencies": arr[:, 0], "psd": arr[:, 1]}
    dets = {d: raw[d] for d in raw if d in psds}       # the detectors LVK analysed
    prep = prepare_real(raw, gps, starts, srs, seed=0)
    res = infer(strain=prep, event_type="BBH", model=a.model, num_samples=4000, device="cpu", seed=0)
    prior = TrainingPrior.from_meta(res.config)
    seg_s = likelihood_seconds(res.samples)
    seg = raw_segment(dets, starts, gps, seg_s)
    mc_min = float(np.quantile(chirp_mass(res.samples[:, 0], res.samples[:, 1]), 0.005))

    outdir = Path(a.out) / a.sampler / ev_name
    outdir.mkdir(parents=True, exist_ok=True)
    settings = dict(DEFAULTS[a.sampler])
    if a.nlive is not None:
        settings["nlive"] = a.nlive
    if a.seed is not None:
        settings["seed"] = a.seed
    t0 = time.time()
    r = run_dynesty(seg, psds, outdir / "bilby", ev_name, center_gps=gps, trigger_offset=0.0,
                    npool=a.npool, prior=prior, approximant=a.approximant, mc_min=mc_min,
                    sampler=a.sampler, sampler_kwargs=settings)
    wall = time.time() - t0

    post = r.posterior
    th = np.zeros((len(post), len(PARAM_NAMES)))
    m1, m2 = post["mass_1"].values, post["mass_2"].values
    th[:, J["mass_1"]], th[:, J["mass_2"]] = np.maximum(m1, m2), np.minimum(m1, m2)
    for k in ("luminosity_distance", "ra", "dec", "theta_jn", "psi"):
        th[:, J[k]] = post[k].values
    for i in (1, 2):
        th[:, J[f"a{i}"]] = (post[f"chi_{i}"] if f"chi_{i}" in post else post[f"a_{i}"]).values
    th = resample_weighted(th, lvk_log_weights(th, prior))
    L = np.load(lv / "lvk_posterior.npz")
    ours = {"chirp_mass": chirp_mass(th[:, 0], th[:, 1]), "mass_ratio": th[:, 1] / th[:, 0],
            "luminosity_distance": th[:, J["luminosity_distance"]]}
    rec = {"event": ev_name, "sampler": a.sampler, "settings": settings,
           "log_evidence": float(r.log_evidence), "log_evidence_err": float(r.log_evidence_err),
           "log_bayes_factor": float(r.log_bayes_factor), "n_posterior": len(post),
           "approximant": a.approximant, "detectors": sorted(dets), "segment_s": seg_s,
           "wall_s": round(wall, 1), "npool": a.npool,
           "n_likelihood": int(getattr(r, "num_likelihood_evaluations", 0) or 0), "params": {}}
    for p, x in ours.items():
        y = L[p]
        rec["params"][p] = {
            "median": float(np.median(x)), "lvk_median": float(np.median(y)),
            "offset_sd": float(abs(np.median(x) - np.median(y)) / np.std(y)),
            "w90_over_lvk": float(np.subtract(*np.quantile(x, [0.95, 0.05])) / np.subtract(*np.quantile(y, [0.95, 0.05])))}
    json.dump(rec, open(Path(a.out) / a.sampler / f"{ev_name}.json", "w"), indent=1)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
