#!/usr/bin/env python3
"""Two overlapping BBH signals: network ranks vs joint correction vs dynesty.

    # 1. the joint likelihood equals bilby's on a single signal (2nd signal far away)
    python scripts/joint_overlap_test.py --check
    # 2. network rank 0 / rank 1 posteriors, then the joint correction
    python scripts/joint_overlap_test.py --run --case 0 --npool 16 --out results/joint_overlap
    # 3. dynesty on the same joint model (reference; slow)
    python scripts/joint_overlap_test.py --dynesty --case 0 --npool 32 --out results/joint_overlap

Cases fix both signals' parameters and the noise seed, so --run and --dynesty
see identical strain.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DETS, SR, DUR = ["H1", "L1", "V1"], 4096, 4.0
BASE = dict(theta_jn=0.6, phase=1.3, a1=0.2, a2=-0.1, event_type="BBH")
# (signal 1, signal 2): signal 1 is the louder by Mc^(5/6)/d_L
CASES = [
    # strongly overlapping: 0.12 s apart, similar chirp masses
    (dict(mass_1=36.0, mass_2=30.0, luminosity_distance=2000.0, ra=1.9, dec=-1.2, psi=0.5, geocent_time=0.0),
     dict(mass_1=28.0, mass_2=22.0, luminosity_distance=1700.0, ra=4.1, dec=0.4, psi=1.9, geocent_time=0.12)),
    # merger times 0.4 s apart, different masses
    (dict(mass_1=50.0, mass_2=40.0, luminosity_distance=2200.0, ra=0.8, dec=0.6, psi=2.2, geocent_time=-0.2),
     dict(mass_1=20.0, mass_2=15.0, luminosity_distance=1200.0, ra=3.3, dec=-0.5, psi=0.9, geocent_time=0.2)),
    # three signals: case 0 with a quieter third, 0.3 s before the first
    (dict(mass_1=36.0, mass_2=30.0, luminosity_distance=2000.0, ra=1.9, dec=-1.2, psi=0.5, geocent_time=0.0),
     dict(mass_1=28.0, mass_2=22.0, luminosity_distance=1700.0, ra=4.1, dec=0.4, psi=1.9, geocent_time=0.12),
     dict(mass_1=18.0, mass_2=14.0, luminosity_distance=1500.0, ra=0.7, dec=0.9, psi=1.1, geocent_time=-0.3)),
    # three signals: well separated in time and mass
    (dict(mass_1=50.0, mass_2=40.0, luminosity_distance=2200.0, ra=0.8, dec=0.6, psi=2.2, geocent_time=-0.35),
     dict(mass_1=30.0, mass_2=20.0, luminosity_distance=1800.0, ra=5.2, dec=-0.9, psi=0.3, geocent_time=0.05),
     dict(mass_1=20.0, mass_2=15.0, luminosity_distance=1700.0, ra=3.3, dec=-0.5, psi=0.9, geocent_time=0.3)),
]


def make_case(i):
    from ahsd.data.bilby_pipeline import BilbyNoiseGenerator, BilbySignalInjector, get_default_psd
    from ahsd.inference.dynesty_bridge import GPS_REF
    noise_gen, inj = BilbyNoiseGenerator(SR, DUR), BilbySignalInjector(SR, DUR)
    psds = {d: get_default_psd(d) for d in DETS}
    ps = []
    for p in CASES[i]:
        q = {**BASE, **p}
        q["geocent_time_gps"] = GPS_REF + q["geocent_time"]
        ps.append(q)
    raw, snrs = {}, []
    for j, d in enumerate(DETS):
        raw[d], s = inj.inject_multiple(noise_gen.generate(d, psds[d], seed=500 + 10 * i + j), ps, d, psds[d])
        snrs.append(s)
    net = np.sqrt((np.asarray(snrs) ** 2).sum(0))
    return raw, psds, ps, net, GPS_REF


def theta_of(p):
    from ahsd.models.lean_npe import PARAM_NAMES
    return np.array([p[n] for n in PARAM_NAMES], float)


def check():
    """Joint likelihood vs bilby's GravitationalWaveTransient on one signal."""
    import bilby
    from ahsd.inference.joint_correction import JointLikelihood, to_bilby
    raw, psds, ps, net, gps = make_case(0)
    jl = JointLikelihood(raw, psds, gps, 20.0, "IMRPhenomXP")
    far = dict(ps[1], luminosity_distance=1e7)          # negligible second signal
    ifos = bilby.gw.detector.InterferometerList([ifo for ifo, _, _, _ in jl.ifos])
    ref = bilby.gw.likelihood.GravitationalWaveTransient(ifos, jl.wfg)
    for label, p1 in (("truth", ps[0]), ("offset", dict(ps[0], mass_1=37.0, phase=2.0))):
        row = np.concatenate([theta_of(p1), theta_of(far)])[None]
        ref.parameters.update(to_bilby(row[0, :11], gps))
        a, b = jl.log_lr(row)[0], ref.log_likelihood_ratio()
        print(f"{label:7s} joint {a:.4f}  bilby {b:.4f}  diff {a - b:+.2e}")
    both = np.concatenate([theta_of(ps[0]), theta_of(ps[1])])[None]
    print(f"both signals at truth: joint log LR {jl.log_lr(both)[0]:.2f}  (network SNRs {np.round(net, 1)})")


def run(a):
    from ahsd.inference.pipeline import infer
    from ahsd.inference.joint_correction import joint_correct
    raw, psds, ps, net, gps = make_case(a.case)
    out = Path(a.out) / f"case{a.case}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"case {a.case}: network SNRs {np.round(net, 1)}", flush=True)
    t = time.time()
    rs = [infer(strain=raw, psds=psds, model=a.model, num_samples=a.samples, rank=r, device="cpu", seed=r)
          for r in range(len(ps))]
    t_npe = time.time() - t
    *cs, diag = joint_correct(rs, raw, center_gps=gps, npool=a.npool, n=a.n, seed=a.seed, f_min=20.0,
                              n_global=a.n_global, target_ess=a.target_ess)
    rec = {"case": a.case, "snr": net.tolist(), "t_npe_s": round(t_npe, 1), **diag}
    names = ("mass_1", "mass_2", "luminosity_distance", "geocent_time", "ra", "dec", "theta_jn")
    for k, (p, r, c) in enumerate(zip(ps, rs, cs)):
        rec[f"signal{k + 1}"] = {}
        for nme in names:
            j = r.param_names.index(nme)
            row = {"truth": p[nme]}
            for lab, s in (("npe", r.samples[:, j]), ("joint", c.samples[:, j])):
                lo, med, hi = np.quantile(s, [0.05, 0.5, 0.95])
                row[lab] = {"med": float(med), "lo": float(lo), "hi": float(hi), "in90": bool(lo <= p[nme] <= hi)}
            rec[f"signal{k + 1}"][nme] = row
        np.save(out / f"joint_signal{k + 1}.npy", c.samples)
        np.save(out / f"npe_signal{k + 1}.npy", r.samples)
    json.dump(rec, open(out / "summary.json", "w"), indent=1)
    jc = diag["joint_correction"]
    print(f"joint: ESS {jc['ess']} of {jc['n']}, {jc['n_stages']} stages, converged {jc['converged']}, "
          f"{jc['wall_s']} s, acceptance {jc['mh_acceptance_mean']}")
    th_truth = np.concatenate([theta_of(p) for p in ps])[None]
    from ahsd.inference.joint_correction import JointLikelihood
    jl = JointLikelihood(raw, psds, gps, 20.0, "IMRPhenomXP")
    print(f"log LR at the truth {jl.log_lr(th_truth)[0]:.1f}; median over joint samples {jc['log_lr_median']}")
    for k in range(1, len(ps) + 1):
        print(f"signal {k}:")
        for nme, row in rec[f"signal{k}"].items():
            n_, j_ = row["npe"], row["joint"]
            print(f"  {nme:20s} truth {row['truth']:9.3f} | npe {n_['med']:9.3f} [{n_['lo']:.3f}, {n_['hi']:.3f}] "
                  f"{'ok' if n_['in90'] else 'MISS'} | joint {j_['med']:9.3f} [{j_['lo']:.3f}, {j_['hi']:.3f}] "
                  f"{'ok' if j_['in90'] else 'MISS'}")


def dynesty(a):
    """Nested sampling on the same joint likelihood and prior. dynesty draws
    from flat boxes over the training support; the full training prior
    density (TrainingPrior.log_prob: mass shape, distance shape, cos dec,
    sin theta_jn) and the loudness ordering enter through the likelihood, so
    the target is exactly the joint correction's."""
    import bilby
    from ahsd.inference.joint_correction import JointLikelihood, loudness
    from ahsd.inference.pipeline import load_model
    from ahsd.inference.priors import TC_HALF_WIDTH, TrainingPrior
    from ahsd.models.lean_npe import PARAM_NAMES
    raw, psds, ps, net, gps = make_case(a.case)
    out = Path(a.out) / f"case{a.case}"
    out.mkdir(parents=True, exist_ok=True)
    tp = TrainingPrior.from_meta(load_model(a.model, "cpu")[1])
    jl = JointLikelihood(raw, psds, gps, 20.0, tp.waveform()["approximant"], tp.event_type)
    names = [f"{n}_{k}" for k in (1, 2) for n in PARAM_NAMES]

    class Like(bilby.Likelihood):
        def __init__(self):
            super().__init__(parameters={n: None for n in names})

        def log_likelihood(self):
            th = np.array([[self.parameters[n] for n in names]])
            lp = tp.log_prob(th[:, :11])[0] + tp.log_prob(th[:, 11:])[0]
            if not np.isfinite(lp) or loudness(th[:, :11])[0] < loudness(th[:, 11:])[0]:
                return -np.inf
            try:
                return float(jl.log_lr(th)[0]) + float(lp)
            except Exception:
                return -np.inf

    U = bilby.core.prior.Uniform
    pri = bilby.core.prior.PriorDict()
    for k in (1, 2):
        pri[f"mass_1_{k}"] = U(*tp.m1_range); pri[f"mass_2_{k}"] = U(*tp.m2_range)
        pri[f"luminosity_distance_{k}"] = U(*tp.dist_range)
        pri[f"ra_{k}"] = U(0, 2 * np.pi, boundary="periodic"); pri[f"dec_{k}"] = U(-np.pi / 2, np.pi / 2)
        pri[f"theta_jn_{k}"] = U(0, np.pi); pri[f"psi_{k}"] = U(0, np.pi, boundary="periodic")
        pri[f"phase_{k}"] = U(0, 2 * np.pi, boundary="periodic")
        pri[f"geocent_time_{k}"] = U(-TC_HALF_WIDTH, TC_HALF_WIDTH)
        pri[f"a1_{k}"] = U(tp.a1_lo, tp.a1_max); pri[f"a2_{k}"] = U(tp.a2_lo, tp.a2_max)
    t = time.time()
    res = bilby.run_sampler(likelihood=Like(), priors=pri, sampler="dynesty", nlive=a.nlive, dlogz=0.5,
                            sample="rwalk", walks=60, naccept=20, npool=a.npool, outdir=str(out / "dynesty"),
                            label=f"case{a.case}", resume=True, plot=False)
    post = res.posterior
    rec = {"case": a.case, "t_dynesty_s": round(time.time() - t, 1), "log_evidence": res.log_evidence}
    for k in (1, 2):
        rec[f"signal{k}"] = {n: {"med": float(post[f"{n}_{k}"].median()),
                                 "lo": float(post[f"{n}_{k}"].quantile(.05)),
                                 "hi": float(post[f"{n}_{k}"].quantile(.95)), "truth": ps[k - 1][n]}
                             for n in ("mass_1", "mass_2", "luminosity_distance", "geocent_time", "ra", "dec", "theta_jn")}
    json.dump(rec, open(out / "dynesty_summary.json", "w"), indent=1)
    print(json.dumps(rec, indent=1)[:3000])


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--dynesty", action="store_true")
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--model", default=str(ROOT / "model/lean_npe_bbh_v5/best_model.pth"))
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npool", type=int, default=8)
    ap.add_argument("--nlive", type=int, default=1000)
    ap.add_argument("--n_global", type=int, default=2)
    ap.add_argument("--target_ess", type=float, default=0.2)
    ap.add_argument("--out", default=str(ROOT / "results/joint_overlap"))
    a = ap.parse_args()
    if a.check:
        check()
    elif a.run:
        run(a)
    else:
        dynesty(a)


if __name__ == "__main__":     # the correction spawns worker processes
    main()
