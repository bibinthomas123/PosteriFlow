"""BNS and NSBH real events, network alone, against published values (2026-11-06).

No LVK posteriors for these on disk, so this scores point estimates: detector-frame
chirp mass and mass ratio from the catalog's source masses and redshift, or the values
validate_checkpoint.py publishes for GW200105, which is not in the cached catalog. Sky
is scored only for GW170817, the one event with an optical position.

Reports the median and 90% interval, whether the published value is inside it, and how
much the seed changes the median.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ahsd.inference import infer
from ahsd.models.lean_npe import PARAM_NAMES as N

EM = {"GW170817": (3.4461, -0.4081)}
BNS, NSBH = "model/lean_npe_bns_v6g/best_model.pth", "model/lean_npe_nsbh_v6g/best_model.pth"
# name, class, model, GPS for events outside the catalog, source m1 m2 dL when outside it
EVENTS = (("GW170817", "BNS", BNS, None, None),
          ("GW190425", "BNS", BNS, None, None),
          ("GW200115_042309", "NSBH", NSBH, None, None),
          ("GW200105", "NSBH", NSBH, 1262276684.0, (8.9, 1.9, 280.0)),
          ("GW190814", "NSBH", NSBH, None, None))
SEEDS = (1, 2, 3)


def mc_of(m1, m2):
    return (m1 * m2) ** .6 / (m1 + m2) ** .2


def published(name, fallback):
    cat = json.load(open("data/gwtc/gwtc_events.json"))
    hit = [e for e in cat if e["event_name"] == name or e["event_name"].split("_")[0] == name.split("_")[0]]
    conf = [e for e in hit if "confident" in e["observing_run"]] or hit
    if conf:
        e = conf[-1]
        z = 1 + e["redshift"]
        m1, m2 = e["mass_1_source"] * z, e["mass_2_source"] * z
        return {"source": e["observing_run"], "chirp_mass": mc_of(m1, m2), "mass_ratio": m2 / m1,
                "luminosity_distance": e["luminosity_distance"], "snr": e.get("network_snr"), "m2_det": m2}
    m1, m2, d = fallback
    z = 1.06   # about 280 Mpc
    return {"source": "validate_checkpoint table", "chirp_mass": mc_of(m1 * z, m2 * z), "mass_ratio": m2 / m1,
            "luminosity_distance": d, "snr": None, "m2_det": m2 * z}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/distance_realdata/ns_events.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for name, cls, model, gps, fb in EVENTS:
        pub = published(name, fb)
        rows = []
        warn = []
        for seed in SEEDS:
            src = {"event": name} if gps is None else {"trigger_time": gps}
            try:
                r = infer(**src, event_type=cls, model=model, num_samples=5000, device=device, seed=seed)
            except Exception as ex:
                print(f"{name}: failed ({type(ex).__name__}: {str(ex)[:160]})", flush=True)
                rows = None
                break
            s = r.samples
            m1, m2 = s[:, N.index("mass_1")], s[:, N.index("mass_2")]
            v = {"chirp_mass": mc_of(m1, m2), "mass_ratio": m2 / m1, "luminosity_distance": s[:, N.index("luminosity_distance")]}
            row = {k: [float(x) for x in np.quantile(x_, [.05, .5, .95])] for k, x_ in v.items()}
            if name in EM:
                z = np.exp(1j * s[:, N.index("ra")]).mean()
                row["ra_err"] = float(abs((np.angle(z) - EM[name][0] + np.pi) % (2 * np.pi) - np.pi))
                row["dec_err"] = float(abs(np.median(s[:, N.index("dec")]) - EM[name][1]))
            rows.append(row)
            warn = list(r.prepared.warnings) if r.prepared is not None else []
            dets = list(r.prepared.detectors_present) if r.prepared is not None else []
            conf = (r.diagnostics.get("verdict") or {}).get("confidence")
        if rows is None:
            continue
        res[name] = {"class": cls, "published": pub, "seeds": rows, "detectors": dets, "warnings": warn, "confidence": conf}
        print(f"== {name} ({cls}) detectors {dets}, confidence {conf}; published from {pub['source']}, SNR {pub['snr']}", flush=True)
        for k in ("chirp_mass", "mass_ratio", "luminosity_distance"):
            lo, me, hi = np.mean([x[k] for x in rows], axis=0)
            seed_spread = float(np.ptp([x[k][1] for x in rows]))
            p = pub[k]
            print(f"  {k:20s} network {me:10.4f} [{lo:.4f},{hi:.4f}]  published {p:10.4f}  inside 90%: {lo <= p <= hi}  seed spread {seed_spread:.4f}", flush=True)
        if name in EM:
            print(f"  sky: ra error {np.mean([x['ra_err'] for x in rows]):.3f} rad, dec error {np.mean([x['dec_err'] for x in rows]):.3f} rad", flush=True)
        for w in warn:
            print(f"  warning: {w[:150]}", flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
