#!/usr/bin/env python3
"""Matched dynesty reference set for certification.

Collects the whitened strains dynesty_compare.py saved for each event
(event<k>_white.npy) with dynesty's chirp-mass and mass-ratio interval
widths from its comparison.json. validate_checkpoint.py then runs a model on
exactly these events and compares widths event by event, instead of
comparing a population-mean width with a reference measured on other
events (which made a 1.5x gap read as 104x).

    python scripts/build_dynesty_reference.py --parts analysis/dynesty_compare_v4/part0 ... \
        --out analysis/dynesty_reference/bbh_v4.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--parts", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
strains, truths, ref = [], [], {k: [] for k in ("mc_w50", "mc_w90", "mc_w99", "q_w90", "mc_true", "q_true", "snr")}
for part in a.parts:
    comp = json.load(open(Path(part) / "comparison.json"))
    for k, c in enumerate(comp):
        strains.append(np.load(Path(part) / f"event{k}_white.npy"))
        truths.append([c["truth"][n] for n in c["truth"]])
        net = c["snr"]
        ref["mc_w50"].append(c["chirp_mass"]["dynesty"]["w50"])
        ref["mc_w90"].append(c["chirp_mass"]["dynesty"]["w90"])
        ref["mc_w99"].append(c["chirp_mass"]["dynesty"]["w99"])
        ref["q_w90"].append(c["mass_ratio"]["dynesty"]["w90"])
        ref["mc_true"].append(c["chirp_mass"]["truth"])
        ref["q_true"].append(c["mass_ratio"]["truth"])
        ref["snr"].append(net)
    print(f"{part}: {len(comp)} events collected", flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out, strain=np.stack(strains), truth=np.array(truths),
                    param_names=np.array(list(comp[0]["truth"])), **{k: np.array(v) for k, v in ref.items()})
print(f"wrote {a.out}: {len(strains)} events")
