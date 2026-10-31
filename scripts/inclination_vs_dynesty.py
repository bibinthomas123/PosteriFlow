"""Does LeanNPE lose inclination information that the data contains?

Same strain to both methods: a loud (network SNR 25) GW150914-like BBH in
design Gaussian noise (H1, L1, V1 — the training domain), at three
inclinations and two sky positions. dynesty (exact likelihood) shows how much
theta_jn information the data holds; LeanNPE's posterior shows how much it
extracts. If dynesty constrains theta_jn and the NPE does not, the encoder is
losing it (fixable); if neither does, it is physics.

    python scripts/inclination_vs_dynesty.py --index 0..5 [--model ...]
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
from ahsd.data.bilby_pipeline import (BilbyNoiseGenerator, BilbyPreprocessor,  # noqa: E402
                                      BilbySignalInjector, get_default_psd)
from ahsd.inference.dynesty_bridge import GPS_REF, align_conventions, run_dynesty  # noqa: E402
from ahsd.inference.pipeline import load_model  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

DETS, SR, DUR, TARGET_SNR = ["H1", "L1", "V1"], 4096, 4.0, 25.0
INCLINATIONS = (0.3, 0.9, 1.57)           # face-on, inclined, edge-on
SKIES = ((1.95, -1.27, 1.75), (4.2, 0.45, 0.6))  # (ra, dec, psi)
CONFIGS = [(th, sky) for sky in SKIES for th in INCLINATIONS]

ap = argparse.ArgumentParser()
ap.add_argument("--index", type=int, required=True, help=f"0..{len(CONFIGS) - 1}")
ap.add_argument("--model", default=str(ROOT / "model/lean_npe_v3/best_model.pth"))
ap.add_argument("--nlive", type=int, default=500)
ap.add_argument("--npool", type=int, default=8)
ap.add_argument("--outdir", default=str(ROOT / "results/inclination_dynesty"))
a = ap.parse_args()

theta, (ra, dec, psi) = CONFIGS[a.index]
out = Path(a.outdir) / f"ev{a.index:02d}"
out.mkdir(parents=True, exist_ok=True)
noise_gen, injector, prep = BilbyNoiseGenerator(SR, DUR), BilbySignalInjector(SR, DUR), BilbyPreprocessor(SR, DUR)
psds = {d: get_default_psd(d) for d in DETS}
P = dict(mass_1=42.4, mass_2=36.5, theta_jn=theta, ra=ra, dec=dec, psi=psi, phase=1.3,
         geocent_time=0.0, geocent_time_gps=GPS_REF, a1=0.0, a2=0.0, event_type="BBH")

# distance for network SNR = TARGET_SNR (SNR scales as 1/d_L)
zeros = {d: np.zeros(int(SR * DUR)) for d in DETS}
snr0 = np.sqrt(sum(injector.inject(zeros[d], {**P, "luminosity_distance": 500.0}, d, psds[d])[1] ** 2 for d in DETS))
P["luminosity_distance"] = float(500.0 * snr0 / TARGET_SNR)

raw, white, snrs = {}, {}, []
for i, d in enumerate(DETS):
    noise = noise_gen.generate(d, psds[d], seed=1000 * a.index + i)
    raw[d], s = injector.inject(noise, P, d, psds[d])
    white[d] = prep.preprocess(raw[d], psds[d], detector=d)
    snrs.append(s)
net = float(np.sqrt(sum(s ** 2 for s in snrs)))
print(f"ev{a.index:02d}: theta_jn {theta}  sky ({ra}, {dec})  dL {P['luminosity_distance']:.0f} Mpc  network SNR {net:.1f}", flush=True)

model, _ = load_model(a.model, "cpu")
x = torch.from_numpy(np.stack([white[d] for d in DETS])).float()[None]
ab = torch.zeros(1, 3, model.encoder.psd_bands) if model.psd_cond else None  # design noise
torch.manual_seed(0)
with torch.no_grad():
    npe = model.sample_posterior(x, n_samples=5000, asd_bands=ab)[0].numpy()
np.save(out / "npe_samples.npy", npe)

t0 = time.time()
res = run_dynesty(raw, psds, out / "bilby", f"ev{a.index:02d}", center_gps=GPS_REF,
                  trigger_offset=0.0, nlive=a.nlive, npool=a.npool, distance_range=(10.0, 5000.0))
post = align_conventions(res.posterior, GPS_REF)


def summary(th):
    c = np.abs(np.cos(th))
    return {"theta_med": float(np.median(th)), "abs_cos_90": [float(np.quantile(c, .05)), float(np.quantile(c, .95))],
            "abs_cos_width": float(np.quantile(c, .95) - np.quantile(c, .05)),
            "p_edge_on": float(np.mean(c < 0.5))}


jt, jd = PARAM_NAMES.index("theta_jn"), PARAM_NAMES.index("luminosity_distance")
rec = {"index": a.index, "truth": {k: P[k] for k in ("theta_jn", "ra", "dec", "psi", "luminosity_distance")},
       "net_snr": net, "dynesty_minutes": round((time.time() - t0) / 60, 1),
       "dynesty": {**summary(post["theta_jn"].values), "dL_med": float(np.median(post["luminosity_distance"]))},
       "npe": {**summary(npe[:, jt]), "dL_med": float(np.median(npe[:, jd]))}}
json.dump(rec, open(out / "summary.json", "w"), indent=1)
for m in ("dynesty", "npe"):
    r = rec[m]
    print(f"  {m:8s} |cos theta| 90% [{r['abs_cos_90'][0]:.2f}, {r['abs_cos_90'][1]:.2f}]  P(edge-on) {r['p_edge_on']:.2f}  "
          f"dL {r['dL_med']:.0f} (true {P['luminosity_distance']:.0f})", flush=True)
