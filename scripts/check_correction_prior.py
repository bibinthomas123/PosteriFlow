#!/usr/bin/env python3
"""End-to-end check of the likelihood correction's measure: with a flat
likelihood the corrected samples must follow the PRIOR.

The likelihood is made flat by inflating the PSD by 1e30 (every waveform's
inner products vanish). Any missing Jacobian shows up as a distorted
distribution: the version without the log-coordinate Jacobian targeted
prior / (m1 m2 d_L), i.e. distances and masses pulled low.

    python scripts/check_correction_prior.py [--model ...] [--npool 16]
Prints quantiles of m1, m2, d_L for the corrected samples and for prior
draws, and KS distances.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "model/lean_npe_bbh_v5/best_model.pth"))
    ap.add_argument("--npool", type=int, default=16)
    ap.add_argument("--samples", type=int, default=4000)
    a = ap.parse_args()
    from scipy.stats import ks_2samp
    from ahsd.data.bilby_pipeline import BilbyNoiseGenerator, get_default_psd
    from ahsd.inference.dynesty_bridge import GPS_REF
    from ahsd.inference.importance import importance_correct
    from ahsd.inference.pipeline import infer
    from ahsd.inference.priors import TrainingPrior

    dets = ("H1", "L1", "V1")
    psds = {d: get_default_psd(d) for d in dets}
    gen = BilbyNoiseGenerator(4096, 4.0)
    raw = {d: gen.generate(d, psds[d], seed=7 + i) for i, d in enumerate(dets)}
    res = infer(strain=raw, psds=psds, model=a.model, num_samples=a.samples, device="cpu", seed=0)
    flat = {d: {"frequencies": p["frequencies"], "psd": np.asarray(p["psd"]) * 1e30} for d, p in psds.items()}
    c = importance_correct(res, raw, flat, center_gps=GPS_REF, npool=a.npool, device="cpu", seed=0,
                           seed_bank=False, f_min=20.0)
    prior = TrainingPrior.from_meta(res.config)
    pr = prior.bilby_priors(GPS_REF, 0.2).sample(20000)
    names = list(res.param_names)
    print(f"ESS {c.diagnostics['importance_sampling']['importance_ess']:.0f}; stages "
          f"{c.diagnostics['importance_sampling']['n_stages']}")
    print(f"{'param':22s} {'corrected 5/50/95%':>28s} {'prior 5/50/95%':>28s}   KS")
    worst = 0.0
    for n in ("mass_1", "mass_2", "luminosity_distance"):
        x = c.samples[:, names.index(n)]
        y = np.asarray(pr[n])
        qx, qy = np.quantile(x, [.05, .5, .95]), np.quantile(y, [.05, .5, .95])
        ks = ks_2samp(x, y).statistic
        worst = max(worst, ks)
        print(f"{n:22s} {np.array2string(qx, precision=1):>28s} {np.array2string(qy, precision=1):>28s}   {ks:.3f}")
    print("PASS" if worst < 0.06 else "FAIL", f"(worst KS {worst:.3f}; < 0.06 expected for a correct measure)")


if __name__ == "__main__":
    main()
