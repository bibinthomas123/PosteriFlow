#!/usr/bin/env python3
"""Does a method recover the posterior's modes? Scored against dynesty.

dynesty's modes: a Gaussian mixture (BIC, up to 4 components) in
(log m1, q, a1); components holding >= 5% of dynesty's mass are modes. For
each method's samples (NPE, IS, bank-seeded IS):
  recall     fraction of dynesty modes in which the method puts >= 30% of
             dynesty's mass (a broad smear over both branches with little
             mass in each does not count as recovering them)
  probs      the method's mass per mode (vs dynesty's)
  ess_mode   distinct particles per mode (IS samples are resampled, so
             distinct particles are the per-mode effective sample size)

    python scripts/mode_study.py --event GW200115_042309 --event_type NSBH \
        --model model/lean_npe_nsbh_v4/best_model.pth \
        --dynesty results/gw200115_dynesty_n1500/GW200115/bilby --out analysis/modes/nsbh_v4.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")


def feats(m1, m2, a1):
    return np.stack([np.log(m1), m2 / m1, a1], 1)


def score(name, x, gm, modes, dyn_p, out):
    lab = gm.predict(x)
    probs = [float((lab == k).mean()) for k in modes]
    recall = float(np.mean([p >= 0.3 * dp for p, dp in zip(probs, dyn_p)]))
    ess = [int(len(np.unique(x[lab == k][:, 0]))) for k in modes]
    out[name] = {"recall": recall, "probs": [round(p, 3) for p in probs], "ess_mode": ess}
    print(f"{name:18s} recall {recall:.2f}  probs {np.round(probs, 3)}  distinct/mode {ess}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--event_type", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dynesty", required=True, help="bilby result directory")
    ap.add_argument("--samples", type=int, default=8000)
    ap.add_argument("--npool", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import bilby
    from sklearn.mixture import GaussianMixture
    from ahsd.inference import infer
    from ahsd.inference.preprocessing import fetch_gwosc, prepare_real, raw_segment
    from ahsd.inference.pipeline import load_model
    from ahsd.models.lean_npe import PARAM_NAMES
    J = PARAM_NAMES.index

    post = bilby.result.read_in_result(glob.glob(a.dynesty + "/*result.json")[0]).posterior
    m1d, m2d = np.maximum(post.mass_1, post.mass_2), np.minimum(post.mass_1, post.mass_2)
    xd = feats(m1d.values, m2d.values, (post["chi_1"] if "chi_1" in post else post["a_1"]).values)
    mu, sd = xd.mean(0), xd.std(0)
    z = lambda x: (x - mu) / sd
    fits = [GaussianMixture(k, covariance_type="full", random_state=0).fit(z(xd)) for k in range(1, 5)]
    gm = min(fits, key=lambda g: g.bic(z(xd)))
    w = np.bincount(gm.predict(z(xd)), minlength=gm.n_components) / len(xd)
    modes = [k for k in range(gm.n_components) if w[k] >= 0.05]
    dyn_p = [float(w[k]) for k in modes]
    centres = [np.round(np.r_[np.exp(gm.means_[k][0] * sd[0] + mu[0]), gm.means_[k][1:] * sd[1:] + mu[1:]], 2).tolist()
               for k in modes]
    out = {"event": a.event, "model": a.model, "dynesty_modes": {"centres_m1_q_a1": centres, "probs": dyn_p}}
    print(f"dynesty: {len(modes)} modes, centres (m1, q, a1) {centres}, probs {np.round(dyn_p, 3)}")

    gps, raw, starts, srs, found = fetch_gwosc(a.event)
    prep = prepare_real(raw, gps, starts, srs, seed=0)
    res = infer(strain=prep, event_type=a.event_type, model=a.model, num_samples=a.samples, device="cpu", seed=0)
    s = res.samples
    score("NPE", z(feats(s[:, 0], s[:, 1], s[:, J("a1")])), gm, modes, dyn_p, out)
    model, _ = load_model(a.model, device="cpu")
    seg = raw_segment({d: raw[d] for d in prep.detectors_present}, starts, gps,
                      4.0 + getattr(model, "lowrate_len", 0) / 512.0)
    for name, sb in (("IS", False), ("IS bank-seeded", True)):
        for rep in (0, 1):
            try:
                c = res.importance_correct(seg, npool=a.npool, device="cpu", seed=rep, seed_bank=sb).samples
                score(f"{name} seed{rep}", z(feats(c[:, 0], c[:, 1], c[:, J("a1")])), gm, modes, dyn_p, out)
            except Exception as e:
                out[f"{name} seed{rep}"] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
                print(name, rep, "failed:", out[f"{name} seed{rep}"]["error"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
