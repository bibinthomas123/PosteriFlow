"""How steady is the sky gate, and is distance biased on L1+V1 events? (2026-11-06)

Stability: the gate's median moved from 1.33 to 0.81 rad with a small change in the data, so
it is rerun here with several injection seeds at the certificate's sources and size.

L1+V1: GW190425 reads 81 Mpc against 150 and GW200105 162 against 280, both inside their
intervals. If injections into that same data recover distance, the low readings are
those events' noise or our published reference; if not, the model is biased for that
detector pair. Distance ratio per noise stretch, at network SNR 15 and 12 (about the
real events'), with a bootstrap interval on the median.
"""
import json
from pathlib import Path

import numpy as np
import torch

from ahsd.evaluation.path_audit import RA_MAX, SEARCHED_TOL, SKY_MIN_N, real_path_injections

device = "cuda" if torch.cuda.is_available() else "cpu"
BNS, NSBH = "model/lean_npe_bns_v6g/best_model.pth", "model/lean_npe_nsbh_v6g/best_model.pth"
GW170817, GW190425 = 1187008882.4, 1240215503.0
GW200105, GW200115 = 1262276684.0, 1263097407.7
SOURCES = {
    "BNS": (BNS, [(GW170817, 1.46 * 1.009, 1.27 * 1.009), (GW190425, 2.0 * 1.035, 1.4 * 1.035)]),
    "NSBH": (NSBH, [(GW200105, 8.9 * 1.06, 1.9 * 1.06), (GW200115, 5.9 * 1.06, 1.4 * 1.06)]),
}


def ci(v, rng=np.random.default_rng(0)):
    v = np.asarray(v)
    b = np.median(v[rng.integers(0, len(v), (4000, len(v)))], axis=1)
    return float(np.median(v)), float(np.quantile(b, .025)), float(np.quantile(b, .975))


out = {}
print("STABILITY: 20 injections per class per seed, as certified", flush=True)
for cls, (model, src) in SOURCES.items():
    gates, pooled = [], []
    for seed in range(8):
        r = real_path_injections(model, src, device=device, event_type=cls, n_per_event=10, seed=seed)
        gates.append(dict(seed=seed, ra=r["ra_error"], n_sky=r["n_sky"], searched=r["searched"], distance=r["distance"], amplitude=r["amplitude"]))
        pooled += r["injections"]
        print(f"  {cls} seed {seed}: ra {r['ra_error']:.3f} (n={r['n_sky']}) searched {r['searched']:.3f} distance {r['distance']:.3f} amplitude {r['amplitude']:.3f}", flush=True)
    ra = np.array([g["ra"] for g in gates])
    passed = (ra < RA_MAX) & (np.array([g["n_sky"] for g in gates]) >= SKY_MIN_N) & (np.abs(np.array([g["searched"] for g in gates]) - .5) < SEARCHED_TOL)
    q = [x for x in pooled if len(x["snr"]) > 1 and sorted(x["snr"].values())[-2] >= 5]
    m, lo, hi = ci([x["ra_error"] for x in q])
    print(f"== {cls}: gate value median {np.median(ra):.3f}, range {ra.min():.3f}-{ra.max():.3f}; sky gates pass {passed.sum()} of {len(ra)}; "
          f"pooled n={len(q)} of {len(pooled)} ra {m:.3f} [{lo:.3f},{hi:.3f}]; mean searched {np.mean([x['searched'] for x in pooled]):.3f} "
          f"(ideal 0.5, se {0.289 / np.sqrt(len(pooled)):.3f})", flush=True)
    for ev in sorted({x["event"] for x in pooled}):
        e = [x for x in q if x["event"] == ev]
        if e:
            m, lo, hi = ci([x["ra_error"] for x in e])
            print(f"   noise at {ev}: counted n={len(e)} ra {m:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    out[cls] = {"gates": gates, "injections": pooled}
    Path("analysis/distance_realdata/gate_stability.json").write_text(json.dumps(out, indent=1, default=float))

print("DISTANCE, L1+V1 and controls: posterior median / truth", flush=True)
for cls, model, name, s in (("BNS", BNS, "GW190425 L1 V1", SOURCES["BNS"][1][1]), ("NSBH", NSBH, "GW200105 L1 V1", SOURCES["NSBH"][1][0]),
                            ("BNS", BNS, "GW170817 H1 L1 V1", SOURCES["BNS"][1][0]), ("NSBH", NSBH, "GW200115 H1 L1 V1", SOURCES["NSBH"][1][1])):
    for snr in (15.0, 12.0):
        r = real_path_injections(model, [s], device=device, event_type=cls, n_per_event=60, target_snr=snr, seed=5)
        m, lo, hi = ci([x["distance"] for x in r["injections"]])
        print(f"  {name:20s} SNR {snr:4.1f} n=60: distance ratio {m:.3f} [{lo:.3f},{hi:.3f}]; below 1: {np.mean([x['distance'] < 1 for x in r['injections']]):.0%}; "
              f"middle 80% {np.quantile([x['distance'] for x in r['injections']], .1):.2f}-{np.quantile([x['distance'] for x in r['injections']], .9):.2f}", flush=True)
        out[f"{name} {snr:g}"] = r
        Path("analysis/distance_realdata/gate_stability.json").write_text(json.dumps(out, indent=1, default=float))
