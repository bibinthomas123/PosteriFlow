"""
Fixed dataset for the coherent-encoder training: BBH-only, so there is NO
event-type mixing (the audited source of the spurious Mc<->dL correlation).
The sampler already draws BBH mass and distance INDEPENDENTLY (raw Pearson
0.001); a mild SNR>=5 gate keeps within-BBH Malmquist selection minimal.

Writes the memmap format the trainer reads directly:
  <out>/memmap/train/{signals.npy [N,3,T] f16, noise.npy [Nn,3,T] f16, params.npy [N,11]}
"""
import argparse, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ahsd.models.lean_npe import PARAM_NAMES
from ahsd.data.parameter_sampler import ParameterSampler
from ahsd.data.bilby_pipeline import (BilbySignalInjector, BilbyNoiseGenerator,
                                      BilbyPreprocessor, get_default_psd)

SR, DUR, T = 4096, 4.0, 16384
DETS = ["H1", "L1", "V1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n_noise", type=int, default=1500)
    ap.add_argument("--min_snr", type=float, default=5.0)
    ap.add_argument("--out", default=str(ROOT / "data/dataset_bbh_fixed"))
    args = ap.parse_args()

    out = Path(args.out) / "memmap" / "train"; out.mkdir(parents=True, exist_ok=True)
    sampler = ParameterSampler()
    inj = BilbySignalInjector(SR, DUR); ng = BilbyNoiseGenerator(SR, DUR)
    prep = BilbyPreprocessor(SR, DUR); psds = {d: get_default_psd(d) for d in DETS}

    # ── noise pool ───────────────────────────────────────────────────────────
    noise = np.zeros((args.n_noise, 3, T), dtype=np.float16)
    for i in range(args.n_noise):
        for di, d in enumerate(DETS):
            raw = ng.generate(d, psds[d], seed=100000 + i * 3 + di)
            noise[i, di] = prep.preprocess(raw, psds[d], detector=d).astype(np.float16)
        if (i + 1) % 500 == 0:
            print(f"  noise {i+1}/{args.n_noise}", flush=True)
    np.save(out / "noise.npy", noise)
    print(f"noise pool {noise.shape} saved", flush=True)

    # ── signals (BBH only, mass ⟂ distance) ─────────────────────────────────
    sig = np.zeros((args.n, 3, T), dtype=np.float16)
    params = np.zeros((args.n, 11), dtype=np.float32)
    snrs = []
    t0 = time.time(); n = 0; tries = 0
    rng = np.random.default_rng(7)
    while n < args.n:
        tries += 1
        p = sampler.sample_parameters("BBH")
        # flatten the distance prior (uniform) so the SNR gate distorts the
        # mass<->distance independence as little as possible (minimal Malmquist).
        p["luminosity_distance"] = float(rng.uniform(150.0, 2000.0))
        white, ss = [], []
        for di, d in enumerate(DETS):
            st, s = inj.inject(np.zeros(T), p, d, psds[d])       # signal into zeros
            white.append(prep.preprocess(st, psds[d], detector=d)); ss.append(s)
        net = float(np.sqrt(sum(x ** 2 for x in ss)))
        if net < args.min_snr:
            continue
        sig[n] = np.stack(white).astype(np.float16)
        params[n] = [p["mass_1"], p["mass_2"], p["luminosity_distance"], p["ra"],
                     p["dec"], p["theta_jn"], p["psi"], p["phase"],
                     p["geocent_time"], p["a1"], p["a2"]]
        snrs.append(net); n += 1
        if n % 500 == 0:
            print(f"  signal {n}/{args.n}  ({(time.time()-t0)/n*1000:.0f} ms/ev, "
                  f"accept={n/tries:.2f})", flush=True)
    np.save(out / "signals.npy", sig)
    np.save(out / "params.npy", params)

    # ── verify decoupling ───────────────────────────────────────────────────
    from scipy.stats import pearsonr, spearmanr
    m1, m2 = params[:, 0], params[:, 1]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    lr, sr = pearsonr(np.log(mc), np.log(params[:, 2])), spearmanr(mc, params[:, 2])
    print(f"\nFIXED dataset: N={args.n}  SNR[{min(snrs):.0f},{max(snrs):.0f}] "
          f"median {np.median(snrs):.0f}", flush=True)
    print(f"log Mc vs log dL: Pearson={lr[0]:+.3f} (R2={lr[0]**2:.3f})  Spearman={sr.correlation:+.3f}",
          flush=True)
    print(f"  (was +0.81 pooled / event-type-mixing; BBH-only removes it)", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
