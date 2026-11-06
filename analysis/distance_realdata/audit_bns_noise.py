"""Is the BNS sky shortfall the model or GW170817's noise? (2026-11-06)

In validation BNS gets 0.19 rad with H1 and L1 similar and Virgo weak (n=195); injections
at GW170817 in that configuration got about 1 rad (n=11), while NSBH around GW200115 got
0.23 rad. The same BNS signals, through the same gate code, into noise from three runs.

Worse only at GW170817 = that stretch of noise. Worse everywhere with validation good =
injections differ from training for this class. Good everywhere = chance.
"""
import json
from pathlib import Path

import numpy as np
import torch

from ahsd.evaluation.path_audit import real_path_injections

MODEL = "model/lean_npe_bns_v6g/best_model.pth"
M = (1.46 * 1.009, 1.27 * 1.009)
NOISE = (("GW170817 O2", 1187008882.4), ("GW190814 O3a", 1249852257.0), ("GW200115 O3b", 1263097407.7))
device = "cuda" if torch.cuda.is_available() else "cpu"
out = {}
for name, gps in NOISE:
    try:
        r = real_path_injections(MODEL, [(gps, *M)], device=device, event_type="BNS", n_per_event=30, seed=11)
    except Exception as e:
        print(f"{name}: failed ({type(e).__name__}: {e})", flush=True)
        continue
    out[name] = r
    rows = r["injections"]
    print(f"== {name}: detectors {list(rows[0]['snr'])}; distance {r['distance']:.2f}; amplitude {r['amplitude']:.3f}; searched {r['searched']:.3f}", flush=True)

    def show(lab, s):
        if s:
            print(f"  {lab:28s} n={len(s):2d} ra_err {np.median([x['ra_error'] for x in s]):.2f} conc {np.median([x['concentration'] for x in s]):.2f} "
                  f"share ra_err>0.5 {np.mean([x['ra_error'] > .5 for x in s]):.0%} searched {np.mean([x['searched'] for x in s]):.2f} dist {np.median([x['distance'] for x in s]):.2f}", flush=True)

    second = lambda x: sorted(x["snr"].values())[-2]
    show("second detector >= 5", [x for x in rows if second(x) >= 5])
    if {"H1", "L1", "V1"} <= set(rows[0]["snr"]):
        ratio = lambda x: min(x["snr"]["H1"], x["snr"]["L1"]) / max(x["snr"]["H1"], x["snr"]["L1"])
        show("weak V, H/L ratio > 0.7", [x for x in rows if x["snr"]["V1"] < 5 and ratio(x) > .7])
        show("weak V, H/L ratio <= 0.7", [x for x in rows if x["snr"]["V1"] < 5 and ratio(x) <= .7 and min(x["snr"]["H1"], x["snr"]["L1"]) >= 5])
        show("V >= 5", [x for x in rows if x["snr"]["V1"] >= 5])
    Path("analysis/distance_realdata/bns_noise.json").write_text(json.dumps(out, indent=1, default=float))
