"""Probe 4 (amortization-vs-representation control): narrow, GW150914-sky-
locked BBH dataset. Same memmap format make_fixed_dataset.py writes (so
train_lean_npe.py / RemixDataset read it unmodified).

Design: sky position FIXED at GW150914's true (ra, dec) -- this is what pins
the H1-L1 delay to the +7ms basin geometry (analysis/attractor_basin,
REPORT.md). Chirp mass drawn near GW150914's true Mc (so the model doesn't
have to amortize across the full BNS/NSBH/light-BBH population), mass ratio
drawn BROADLY [0.3, 1.0] (so q is still a genuine inference target, not a
point mass -- this is the whole test), everything else (distance, spins,
inclination, phase, psi, t_c) drawn from the normal full priors so the
dataset is not a single delta function.

If a model trained ONLY on this narrow, still-real inference problem
recovers q on the ACTUAL GW150914 strain (held out, never in this set),
the population-wide amortization was the cost. If it still fails, the
architecture/optimization can't reach the answer even without having to
share weights across the whole population.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ahsd.data.bilby_pipeline import (BilbySignalInjector, BilbyNoiseGenerator,
                                      BilbyPreprocessor, get_default_psd)

SR, DUR, T = 4096, 4.0, 16384
DETS = ["H1", "L1", "V1"]

# GW150914 true sky location (fixes tau_HL to the +7ms basin cell)
RA, DEC = 1.95, -1.27


def masses_from_mc_q(mc, q):
    M = mc * (1.0 + q) ** 1.2 / q ** 0.6
    return M / (1.0 + q), M * q / (1.0 + q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n_noise", type=int, default=800)
    ap.add_argument("--min_snr", type=float, default=8.0)
    ap.add_argument("--mc_range", type=float, nargs=2, default=(25.0, 40.0))
    ap.add_argument("--q_range", type=float, nargs=2, default=(0.3, 1.0))
    ap.add_argument("--dist_range", type=float, nargs=2, default=(200.0, 900.0))
    ap.add_argument("--out", default=str(ROOT / "data/dataset_gw150914_narrow"))
    args = ap.parse_args()

    out = Path(args.out) / "memmap" / "train"
    out.mkdir(parents=True, exist_ok=True)
    inj = BilbySignalInjector(SR, DUR)
    ng = BilbyNoiseGenerator(SR, DUR)
    prep = BilbyPreprocessor(SR, DUR)
    psds = {d: get_default_psd(d) for d in DETS}
    rng = np.random.default_rng(2026)

    noise = np.zeros((args.n_noise, 3, T), dtype=np.float16)
    for i in range(args.n_noise):
        for di, d in enumerate(DETS):
            raw = ng.generate(d, psds[d], seed=200000 + i * 3 + di)
            noise[i, di] = prep.preprocess(raw, psds[d], detector=d).astype(np.float16)
        if (i + 1) % 200 == 0:
            print(f"  noise {i + 1}/{args.n_noise}", flush=True)
    np.save(out / "noise.npy", noise)
    print(f"noise pool {noise.shape} saved", flush=True)

    sig = np.zeros((args.n, 3, T), dtype=np.float16)
    params = np.zeros((args.n, 11), dtype=np.float32)
    snrs = []
    t0 = time.time()
    n = 0
    tries = 0
    while n < args.n:
        tries += 1
        mc = float(rng.uniform(*args.mc_range))
        q = float(rng.uniform(*args.q_range))
        m1, m2 = masses_from_mc_q(mc, q)
        p = dict(
            mass_1=m1, mass_2=m2,
            luminosity_distance=float(rng.uniform(*args.dist_range)),
            ra=RA, dec=DEC,
            theta_jn=float(rng.uniform(0.0, np.pi)),
            psi=float(rng.uniform(0.0, np.pi)),
            phase=float(rng.uniform(0.0, 2 * np.pi)),
            geocent_time=float(rng.uniform(-0.3, 0.3)),
            a1=float(rng.uniform(0.0, 0.3)), a2=float(rng.uniform(0.0, 0.3)),
        )
        white, ss = [], []
        for di, d in enumerate(DETS):
            st, s = inj.inject(np.zeros(T), p, d, psds[d])
            white.append(prep.preprocess(st, psds[d], detector=d))
            ss.append(s)
        net = float(np.sqrt(sum(x ** 2 for x in ss)))
        if net < args.min_snr:
            continue
        sig[n] = np.stack(white).astype(np.float16)
        params[n] = [p["mass_1"], p["mass_2"], p["luminosity_distance"], p["ra"],
                     p["dec"], p["theta_jn"], p["psi"], p["phase"],
                     p["geocent_time"], p["a1"], p["a2"]]
        snrs.append(net)
        n += 1
        if n % 300 == 0:
            print(f"  signal {n}/{args.n}  ({(time.time() - t0) / n * 1000:.0f} ms/ev, "
                  f"accept={n / tries:.2f})", flush=True)
    np.save(out / "signals.npy", sig)
    np.save(out / "params.npy", params)
    print(f"\nNARROW dataset: N={args.n}  Mc[{args.mc_range}]  q[{args.q_range}]  "
          f"sky=({RA},{DEC})  SNR[{min(snrs):.0f},{max(snrs):.0f}] "
          f"median {np.median(snrs):.0f}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
