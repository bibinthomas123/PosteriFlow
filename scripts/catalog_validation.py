#!/usr/bin/env python3
"""Catalogue-wide check against published GWTC posteriors (medians).

For each GWTC-1/2.1/3 confident event above an SNR threshold: the class
model's posterior (class from the published masses), importance sampling
over the segment the model read, both reweighted to LVK-style priors
(priors.lvk_log_weights). Published detector-frame chirp mass, mass ratio
and distance are compared with our 90% intervals; the offset is reported in
units of our posterior std. One JSON per event (resumable); --part/--nparts
split the list across jobs.

    python scripts/catalog_validation.py --out results/catalog_v5 --part 0 --nparts 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")


def summary(x, pub):
    lo, med, hi = np.quantile(x, [0.05, 0.5, 0.95])
    return {"med": float(med), "lo": float(lo), "hi": float(hi), "in90": bool(lo <= pub <= hi),
            "z": float((pub - med) / (x.std() + 1e-12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_snr", type=float, default=10.0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--npool", type=int, default=30)
    ap.add_argument("--no_is", action="store_true")
    ap.add_argument("--prefetch", action="store_true", help="only download the strain (login node)")
    a = ap.parse_args()
    from ahsd.inference import infer
    from ahsd.inference.pipeline import load_model
    from ahsd.inference.preprocessing import fetch_gwosc, prepare_real, raw_segment
    from ahsd.inference.priors import TrainingPrior, lvk_log_weights, resample_weighted
    from ahsd.models.lean_npe import PARAM_NAMES
    J = PARAM_NAMES.index

    cat = json.load(open("data/gwtc/gwtc_events.json"))
    runs = ("GWTC-1-confident", "GWTC-2.1-confident", "GWTC-3-confident")
    ev = sorted([e for e in cat if e["observing_run"] in runs and e.get("mass_1_source", 0) > 0
                 and e.get("network_snr", 0) >= a.min_snr], key=lambda e: e["gps_time"])
    ev = ev[a.part::a.nparts]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{len(ev)} events in part {a.part}/{a.nparts}", flush=True)
    if a.prefetch:
        for e in ev:
            try:
                fetch_gwosc(e["gps_time"]); print("fetched", e["event_name"], flush=True)
            except Exception as ex:
                print("fetch failed", e["event_name"], str(ex)[:80], flush=True)
        return
    for e in ev:
        f = out / f"{e['event_name']}.json"
        if f.exists():
            continue
        z1 = 1 + e["redshift"]
        m1, m2 = e["mass_1_source"] * z1, e["mass_2_source"] * z1
        cls = "BNS" if m1 < 3 else ("NSBH" if m2 < 3 else "BBH")
        pub = {"chirp_mass": (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2, "mass_ratio": m2 / m1,
               "luminosity_distance": e["luminosity_distance"]}
        rec = {"event": e["event_name"], "class": cls, "snr": e["network_snr"], "published_det": pub}
        try:
            gps, raw, starts, srs, found = fetch_gwosc(e["gps_time"])
        except Exception as ex:          # not cached (compute nodes have no GWOSC): retry later
            print(e["event_name"], "no data:", str(ex)[:80], flush=True)
            continue
        try:
            prep = prepare_real(raw, gps, starts, srs, seed=0)
            res = infer(strain=prep, event_type=cls, num_samples=a.samples, device="cpu", seed=0)
            prior = TrainingPrior.from_meta(res.config)
            sets = {"npe": res.samples}
            if not a.no_is:
                model, _ = load_model(res.config["model_path"], device="cpu")
                seg = raw_segment({d: raw[d] for d in prep.detectors_present}, starts, gps,
                                  4.0 + getattr(model, "lowrate_len", 0) / 512.0)
                try:
                    c = res.importance_correct(seg, npool=a.npool, device="cpu", seed=0)
                    sets["is"] = c.samples
                    rec["is_ess"] = c.diagnostics["importance_sampling"]["importance_ess"]
                except Exception as ex:
                    rec["is_error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
            for k, s in sets.items():
                s = resample_weighted(s, lvk_log_weights(s, prior))
                mc = (s[:, 0] * s[:, 1]) ** 0.6 / (s[:, 0] + s[:, 1]) ** 0.2
                rec[k] = {"chirp_mass": summary(mc, pub["chirp_mass"]),
                          "mass_ratio": summary(s[:, 1] / s[:, 0], pub["mass_ratio"]),
                          "luminosity_distance": summary(s[:, J("luminosity_distance")], pub["luminosity_distance"])}
            print(e["event_name"], cls, {k: {p: rec[k][p]["in90"] for p in rec[k]} for k in sets}, flush=True)
        except Exception as ex:
            rec["error"] = f"{type(ex).__name__}: {str(ex)[:150]}"
            print(e["event_name"], "failed:", rec["error"], flush=True)
        json.dump(rec, open(f, "w"), indent=2)


if __name__ == "__main__":
    main()
