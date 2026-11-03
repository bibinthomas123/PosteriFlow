#!/usr/bin/env python3
"""Inclination: NPE and importance-sampled NPE against dynesty on the SAME strain.

Regenerates the injections of scripts/inclination_vs_dynesty.py (same
parameters, same noise seeds) and compares against the dynesty summaries that
script stored, so dynesty does not have to be rerun. Checks that the strain is
reproduced by re-sampling the model the dynesty run used and comparing with its
stored NPE summary.

    python scripts/inclination_is.py --index 0..5 --dynesty-dir results/inclination_bbh_v3 \
        --model model/lean_npe_bbh_v5/best_model.pth [--ref-model model/lean_npe_v3/best_model.pth]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ahsd.data.bilby_pipeline import BilbyNoiseGenerator, BilbySignalInjector, get_default_psd  # noqa: E402
from ahsd.inference.pipeline import infer  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

DETS, SR, DUR, TARGET_SNR = ["H1", "L1", "V1"], 4096, 4.0, 25.0
INCLINATIONS = (0.3, 0.9, 1.57)
SKIES = ((1.95, -1.27, 1.75), (4.2, 0.45, 0.6))
CONFIGS = [(th, sky) for sky in SKIES for th in INCLINATIONS]   # must match inclination_vs_dynesty.py

ap = argparse.ArgumentParser()
ap.add_argument("--index", type=int, required=True)
ap.add_argument("--dynesty-dir", required=True, help="outdir of inclination_vs_dynesty.py")
ap.add_argument("--model", required=True)
ap.add_argument("--ref-model", default=None, help="model the dynesty run used (strain check)")
ap.add_argument("--samples", type=int, default=5000)
ap.add_argument("--npool", type=int, default=8)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--outdir", default=str(ROOT / "results/inclination_is"))
a = ap.parse_args()

theta, (ra, dec, psi) = CONFIGS[a.index]
ref = json.load(open(Path(a.dynesty_dir) / f"ev{a.index:02d}" / "summary.json"))
out = Path(a.outdir) / f"ev{a.index:02d}"
out.mkdir(parents=True, exist_ok=True)

noise_gen, injector = BilbyNoiseGenerator(SR, DUR), BilbySignalInjector(SR, DUR)
psds = {d: get_default_psd(d) for d in DETS}
P = dict(mass_1=42.4, mass_2=36.5, theta_jn=theta, ra=ra, dec=dec, psi=psi, phase=1.3,
         geocent_time=0.0, a1=0.0, a2=0.0, event_type="BBH")
from ahsd.inference.dynesty_bridge import GPS_REF  # noqa: E402
P["geocent_time_gps"] = GPS_REF
zeros = {d: np.zeros(int(SR * DUR)) for d in DETS}
snr0 = np.sqrt(sum(injector.inject(zeros[d], {**P, "luminosity_distance": 500.0}, d, psds[d])[1] ** 2 for d in DETS))
P["luminosity_distance"] = float(500.0 * snr0 / TARGET_SNR)
assert abs(P["luminosity_distance"] - ref["truth"]["luminosity_distance"]) < 1e-3 * P["luminosity_distance"]

raw = {}
for i, d in enumerate(DETS):
    raw[d], _ = injector.inject(noise_gen.generate(d, psds[d], seed=1000 * a.index + i), P, d, psds[d])

jt, jd = PARAM_NAMES.index("theta_jn"), PARAM_NAMES.index("luminosity_distance")


def summary(s):
    c = np.abs(np.cos(s[:, jt]))
    return {"abs_cos_90": [float(np.quantile(c, .05)), float(np.quantile(c, .95))],
            "abs_cos_width": float(np.quantile(c, .95) - np.quantile(c, .05)),
            "p_edge_on": float(np.mean(c < 0.5)), "dL_med": float(np.median(s[:, jd]))}


rec = {"index": a.index, "truth": ref["truth"], "dynesty": ref["dynesty"], "model": a.model}
if a.ref_model:   # same strain? the reference model should reproduce its stored summary
    r = infer(strain=raw, psds=psds, model=a.ref_model, num_samples=a.samples, device="cpu", seed=0)
    rec["ref_npe_check"] = {"now": summary(r.samples), "stored": ref["npe"]}

res = infer(strain=raw, psds=psds, model=a.model, num_samples=a.samples, device="cpu", seed=0)
rec["npe"] = summary(res.samples)
t0 = time.time()
reps = []
for k in range(a.reps):
    c = res.importance_correct(raw, npool=a.npool, device="cpu", seed=k)
    reps.append({**summary(c.samples), "ess": c.diagnostics["importance_sampling"].get("importance_ess")})
rec["is"] = reps
rec["is_wall_s"] = round(time.time() - t0, 1)
json.dump(rec, open(out / "summary.json", "w"), indent=1)

print(f"ev{a.index:02d} theta_jn {theta} dL {P['luminosity_distance']:.0f}")
rows = [("dynesty", ref["dynesty"]), ("npe", rec["npe"])] + [(f"is seed{k}", r) for k, r in enumerate(reps)]
if a.ref_model:
    rows.insert(1, ("ref now", rec["ref_npe_check"]["now"]))
    rows.insert(1, ("ref stored", ref["npe"]))
for name, r in rows:
    print(f"  {name:11s} |cos| 90% [{r['abs_cos_90'][0]:.2f}, {r['abs_cos_90'][1]:.2f}]  "
          f"P(|cos|<0.5) {r['p_edge_on']:.2f}  dL {r['dL_med']:.0f}", flush=True)
