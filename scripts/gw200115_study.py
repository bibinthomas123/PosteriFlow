#!/usr/bin/env python3
"""GW200115: the NSBH model puts the black hole at ~11 Msun (published 6.3,
detector frame). Where does the difference come from? Not an attempt to move
the answer: each block isolates one candidate source.

  1. posterior   where 6.3 falls in the NPE posterior (broad vs inconsistent)
  2. degeneracy  m1 conditional on the aligned BH spin a1 (mass-ratio/spin)
  3. prior       reweighted to LVK-like priors: uniform component masses,
                 the aligned-spin marginal of isotropic spins, d^2 distance
  4. detectors   each detector replaced by white noise in turn
  5. model vs data  GW200115-like injections (published medians, a1 = -0.19)
                 at the event's SNR: is ~11 typical of the model (bias) or
                 specific to this stretch of data?
  6. waveform    importance sampling against the likelihood with the training
                 waveform and with another (if --is)

    python scripts/gw200115_study.py --model model/lean_npe_nsbh_v4/best_model.pth \
        --out analysis/gw200115 [--n_inj 100] [--is]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ahsd.inference import infer  # noqa: E402
from ahsd.inference.preprocessing import fetch_gwosc, prepare_real, raw_segment  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

J = PARAM_NAMES.index
PUB = {"mass_1": 6.3, "mass_2": 1.5, "luminosity_distance": 300.0, "a1": -0.19}   # detector frame
DETS = ["H1", "L1", "V1"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="model/lean_npe_nsbh_v4/best_model.pth")
ap.add_argument("--out", default="analysis/gw200115")
ap.add_argument("--n_inj", type=int, default=100)
ap.add_argument("--samples", type=int, default=20000)
ap.add_argument("--is", dest="do_is", action="store_true")
ap.add_argument("--npool", type=int, default=16)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
rep = {"model": a.model, "published_detector_frame": PUB}


def pct(x, v):
    return float((x < v).mean() * 100)


def qs(x):
    return [round(float(v), 3) for v in np.quantile(x, [0.05, 0.5, 0.95])]


# ---- 1. the posterior ----------------------------------------------------
gps, raw, starts, srs, found = fetch_gwosc("GW200115_042309")
prep = prepare_real(raw, gps, starts, srs, seed=0)
res = infer(strain=prep, event_type="NSBH", model=a.model, num_samples=a.samples, device=dev, seed=0)
S = res.samples
m1, m2, a1 = S[:, J("mass_1")], S[:, J("mass_2")], S[:, J("a1")]
rep["posterior"] = {"detectors": prep.detectors_present, "m1_q05_q50_q95": qs(m1), "m2_q05_q50_q95": qs(m2),
                    "a1_q05_q50_q95": qs(a1), "published_m1_percentile": pct(m1, PUB["mass_1"]),
                    "corr_m1_a1": float(np.corrcoef(np.log(m1), a1)[0, 1])}
print("1. posterior", rep["posterior"], flush=True)

# ---- 2. mass ratio / spin degeneracy ---------------------------------------
rep["m1_given_a1"] = {}
for lo, hi in ((-0.9, -0.3), (-0.3, 0.0), (0.0, 0.3), (0.3, 0.9)):
    k = (a1 >= lo) & (a1 < hi)
    if k.sum() > 50:
        rep["m1_given_a1"][f"[{lo},{hi})"] = {"frac": round(float(k.mean()), 3), "m1": qs(m1[k])}
print("2. m1 | a1", rep["m1_given_a1"], flush=True)

# ---- 3. prior reweighting ---------------------------------------------------
# training: BH log-uniform (1/m1), NS uniform, a1 uniform signed, distance uniform.
# LVK-like: uniform component masses, aligned-spin marginal of an isotropic spin
# with uniform magnitude, p(chi) = -ln(|chi|/amax)/(2 amax), and d^2.
amax = 0.9
d = S[:, J("luminosity_distance")]
w_mass = m1                                                   # / (1/m1)
w_spin = -np.log(np.clip(np.abs(a1), 1e-4, amax) / amax)     # / uniform
w_dist = d ** 2
rep["reweighted"] = {}
for name, w in (("uniform_masses", w_mass), ("isotropic_spin", w_spin), ("d2_distance", w_dist),
                ("all_three", w_mass * w_spin * w_dist)):
    w = w / w.sum()
    ess = float(1 / np.sum(w ** 2))
    idx = np.random.default_rng(0).choice(len(w), size=len(w), p=w)
    rep["reweighted"][name] = {"m1": qs(m1[idx]), "a1": qs(a1[idx]), "ess": round(ess),
                               "published_m1_percentile": pct(m1[idx], PUB["mass_1"])}
print("3. reweighted", rep["reweighted"], flush=True)

# ---- 4. detectors -------------------------------------------------------------
rep["drop_detector"] = {}
for det in prep.detectors_present:
    raw_d = {k: v for k, v in raw.items() if k != det}
    p_d = prepare_real(raw_d, gps, {k: starts[k] for k in raw_d}, {k: srs[k] for k in raw_d}, seed=0)
    r = infer(strain=p_d, event_type="NSBH", model=a.model, num_samples=6000, device=dev, seed=0)
    rep["drop_detector"][det] = {"m1": qs(r.samples[:, J("mass_1")]),
                                 "tc": qs(r.samples[:, J("geocent_time")])}
print("4. without detector", rep["drop_detector"], flush=True)

# ---- 5. model vs data: GW200115-like injections at the event's SNR ------------
from ahsd.data.dataset_generator import GWDatasetGenerator  # noqa: E402
import yaml  # noqa: E402
cfg = yaml.safe_load(open("configs/data_config_nsbh_v4.yaml"))
g = GWDatasetGenerator(output_dir=str(out / "_gen"), config=dict(cfg, random_seed=7))
med_tc = float(np.median(S[:, J("geocent_time")]))
X, SN = [], []
for k in range(a.n_inj):
    p = g.sampler.sample_parameters("NSBH")
    p.update(mass_1=PUB["mass_1"], mass_2=PUB["mass_2"], a1=PUB["a1"], a2=0.0,
             luminosity_distance=PUB["luminosity_distance"], lambda_1=0.0)
    p["geocent_time_gps"] += -p["geocent_time"]; p["geocent_time"] = 0.0
    s, _ = g._make_sample([p], 0.0, False, True)
    dd = s["detector_data"]
    X.append(np.stack([np.concatenate([dd[x]["strain"], dd[x]["strain_lo"]]) for x in DETS]))
    SN.append(s["network_snr"])
x = torch.from_numpy(np.stack(X)).float()
from ahsd.inference.pipeline import load_model  # noqa: E402
model, _ = load_model(a.model, device=dev)
ab = torch.zeros(len(x), 3, model.encoder.psd_bands, device=dev)
med1, inside = [], []
with torch.no_grad():
    for i in range(0, len(x), 25):
        sm = model.sample_posterior(x[i:i + 25].to(dev), n_samples=2000, asd_bands=ab[i:i + 25]).cpu().numpy()
        mm = sm[..., J("mass_1")]
        med1.append(np.median(mm, 1))
        lo, hi = np.quantile(mm, [0.05, 0.95], axis=1)
        inside.append((lo <= PUB["mass_1"]) & (PUB["mass_1"] <= hi))
med1 = np.concatenate(med1); inside = np.concatenate(inside); SN = np.array(SN)
rep["injections"] = {"n": int(len(med1)), "snr_q05_q50_q95": qs(SN),
                     "npe_m1_median_q05_q50_q95": qs(med1),
                     "truth_in_npe_90": round(float(inside.mean()), 3),
                     "frac_median_above_10": round(float((med1 > 10).mean()), 3)}
print("5. injections (truth m1 6.3, a1 -0.19)", rep["injections"], flush=True)

# ---- 6. waveform: importance sampling with two waveform models ---------------
if a.do_is:
    seg = raw_segment({k: raw[k] for k in prep.detectors_present}, starts, gps, seconds=32.0)
    rep["importance"] = {}
    for apx in ("IMRPhenomNSBH", "SEOBNRv4_ROM_NRTidalv2_NSBH"):
        try:
            c = res.importance_correct(seg, npool=a.npool, approximant=apx, device=dev)
            rep["importance"][apx] = {"m1": qs(c.samples[:, J("mass_1")]), "a1": qs(c.samples[:, J("a1")]),
                                      "ess": c.diagnostics["importance_sampling"]["importance_ess"],
                                      "published_m1_percentile": pct(c.samples[:, J("mass_1")], PUB["mass_1"])}
        except Exception as e:
            rep["importance"][apx] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print("6. importance", apx, rep["importance"][apx], flush=True)

json.dump(rep, open(out / "summary.json", "w"), indent=2)
print(f"wrote {out}/summary.json")
