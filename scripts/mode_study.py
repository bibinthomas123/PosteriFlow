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
    ap.add_argument("--target_ess", type=float, default=0.50, help="ESS fraction kept per tempering stage")
    ap.add_argument("--reps", type=int, default=None, help="seeds per variant (default: script's)")
    ap.add_argument("--anchor_inflate", type=float, default=4.0, help="tempering-anchor covariance inflation")
    ap.add_argument("--start", choices=("dynesty", "npe"), default="dynesty",
                    help="--invariance start: dynesty's posterior, or the network's samples (a badly "
                         "weighted population: how fast do the beta = 1 moves repair it?)")
    ap.add_argument("--rounds", type=int, nargs="+", default=[2, 6])
    ap.add_argument("--segment_z", action="store_true",
                    help="posterior mass of each dynesty mode under OUR likelihood x prior: importance "
                         "sampling per mode (mixture fitted on dynesty's samples of it, restricted to "
                         "the mode's region) with the correction's likelihood setup")
    ap.add_argument("--invariance", action="store_true",
                    help="start the correction's beta = 1 moves from dynesty's posterior and score the "
                         "modes after 2 and 6 rounds (drift = the two targets differ, or a move is wrong)")
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
    if a.segment_z:
        import multiprocessing as mp
        from ahsd.inference import importance as I
        from ahsd.inference.coords import SlowCoords
        from ahsd.inference.priors import TrainingPrior, chirp_mass, window_fmin
        prior = TrainingPrior.from_meta(res.config)
        C = SlowCoords(prior)
        th_n = res.samples
        mc_lo = float(np.quantile(chirp_mass(th_n[:, 0], th_n[:, 1]), 0.005))
        tc_lo = float(np.quantile(th_n[:, J("geocent_time")], 0.005))
        f_min = window_fmin(mc_lo, I.segment_duration(seg) - 2.0 + tc_lo)
        apx = prior.waveform()["approximant"]
        pool = mp.get_context("spawn").Pool(a.npool, initializer=I._init_worker,
                                            initargs=(seg, res.prepared.psds, res.trigger_gps, f_min, apx,
                                                      prior.event_type))
        ev = lambda t: np.concatenate(pool.map(I._eval_chunk, np.array_split(t, a.npool * 8)))  # noqa: E731
        th = np.zeros((len(post), len(PARAM_NAMES)))
        th[:, J("mass_1")], th[:, J("mass_2")] = m1d.values, m2d.values
        for k in ("luminosity_distance", "ra", "dec", "theta_jn"):
            th[:, J(k)] = post[k].values
        th[:, J("psi")] = np.mod(post["psi"].values, np.pi)
        th[:, J("a1")] = (post["chi_1"] if "chi_1" in post else post["a_1"]).values
        th[:, J("a2")] = (post["chi_2"] if "chi_2" in post else post["a_2"]).values
        lab_d = gm.predict(z(xd))
        rng = np.random.default_rng(0)
        from ahsd.inference.importance import _inflated
        logz = {}
        print(f"segment_z: f_min {f_min:.1f} Hz, {apx}, segment {I.segment_duration(seg):.0f} s", flush=True)
        for k in modes:
            xk = C.wrap(C.to_x(th[lab_d == k]))
            g = _inflated(GaussianMixture(min(6, max(1, len(xk) // 60)), covariance_type="full",
                                          reg_covar=1e-6, random_state=0).fit(xk), 1.5)
            xs = C.fold_wrap(np.asarray(g.sample(12000)[0]))
            ths = np.zeros((len(xs), len(PARAM_NAMES)))
            v = C.from_x(xs)
            ths[:, C.idx] = v
            ths[:, 0], ths[:, 1] = np.maximum(v[:, 0], v[:, 1]), np.minimum(v[:, 0], v[:, 1])
            ths[:, J("geocent_time")] = np.median(th_n[:, J("geocent_time")])
            inside = gm.predict(z(feats(ths[:, 0], ths[:, 1], ths[:, J("a1")]))) == k
            lp = prior.log_prob(ths)
            ok = np.isfinite(lp) & inside
            ll = np.full(len(ths), -np.inf)
            ll[ok] = ev(ths[ok])
            lw = ll + lp + C.log_jac(ths) - C.mixture_logpdf(g, xs)
            f = np.isfinite(lw)
            mx = lw[f].max()
            w = np.exp(lw[f] - mx)
            logz[k] = mx + np.log(w.sum() / len(lw))
            print(f"  mode {k}: inside {inside.mean():.2f}  log Z {logz[k]:.2f}  IS ESS "
                  f"{w.sum() ** 2 / (w ** 2).sum():.0f}/{len(lw)}", flush=True)
        zz = np.array([logz[k] for k in modes])
        frac = np.exp(zz - zz.max()) / np.exp(zz - zz.max()).sum()
        print(f"OUR target, mass per mode: {np.round(frac, 3)}   dynesty: {np.round(dyn_p, 3)}", flush=True)
        out["segment_z"] = {"ours": frac.tolist(), "dynesty": dyn_p}
        pool.terminate()
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        return
    if a.invariance:
        th = np.zeros((len(post), len(PARAM_NAMES)))
        th[:, J("mass_1")], th[:, J("mass_2")] = m1d.values, m2d.values
        for k in ("luminosity_distance", "ra", "dec", "theta_jn"):
            th[:, J(k)] = post[k].values
        th[:, J("psi")] = np.mod(post["psi"].values, np.pi)
        th[:, J("a1")] = (post["chi_1"] if "chi_1" in post else post["a_1"]).values
        th[:, J("a2")] = (post["chi_2"] if "chi_2" in post else post["a_2"]).values
        if a.start == "npe":
            th = res.samples[np.random.default_rng(0).choice(len(res.samples), size=len(th), replace=False)]
        score(f"{a.start} (start)", z(feats(th[:, 0], th[:, 1], th[:, J("a1")])), gm, modes, dyn_p, out)
        for rounds in a.rounds:
            r = res.importance_correct(seg, npool=a.npool, device="cpu", seed=0, start_particles=th,
                                       n_rejuvenate=rounds)
            c = r.samples
            score(f"moves x{rounds}", z(feats(c[:, 0], c[:, 1], c[:, J("a1")])), gm, modes, dyn_p, out)
            dd = r.diagnostics["importance_sampling"]
            print("   acceptance:", {k: dd.get(k) for k in ("ridge_acceptance", "ridge_acceptance_final",
                                                          "independence_acceptance", "sky_jump_acceptance")},
                  flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        return
    for name, sb in (("IS", False), ("IS bank-seeded", True)):
        for rep in (0, 1):
            try:
                c = res.importance_correct(seg, npool=a.npool, device="cpu", seed=rep, seed_bank=sb,
                                           target_ess=a.target_ess, anchor_inflate=a.anchor_inflate).samples
                score(f"{name} seed{rep}", z(feats(c[:, 0], c[:, 1], c[:, J("a1")])), gm, modes, dyn_p, out)
            except Exception as e:
                out[f"{name} seed{rep}"] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
                print(name, rep, "failed:", out[f"{name} seed{rep}"]["error"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
