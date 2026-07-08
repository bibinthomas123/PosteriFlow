#!/usr/bin/env python3
"""
LeanNPE inference CLI — strain in, publication-quality posterior out.

Real GWOSC event (downloads open data, estimates PSDs, whitens, samples):
    python infer.py --event GW150914 --samples 10000 --output results/GW150914

Real data from local files (GPS trigger required; detector inferred from the
filename prefix H1/L1/V1; .hdf5/.gwf via gwpy, .npy assumed 4096 Hz raw):
    python infer.py --strain H1.hdf5 L1.hdf5 V1.hdf5 --gps 1126259462.4 \\
        --psd psds/ --output results/myevent

Simulated end-to-end validation (fresh injection through the SAME pipeline
that generated the training data; truth overlaid on every plot):
    python infer.py --inject --seed 42 --output results/injection42

Add --compare-dynesty to run bilby/dynesty on the identical data afterwards
(minutes, not seconds).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np


def load_strain_file(path: str):
    """Return (detector, strain, start_gps, sample_rate)."""
    p = Path(path)
    det = next((d for d in ("H1", "L1", "V1") if p.name.upper().startswith(d)), None)
    if det is None:
        raise ValueError(f"{p.name}: cannot infer detector — filename must start with H1/L1/V1")
    if p.suffix in (".hdf5", ".h5", ".gwf"):
        from gwpy.timeseries import TimeSeries
        ts = TimeSeries.read(str(p)) if p.suffix == ".gwf" else TimeSeries.read(str(p), format="hdf5")
        return det, ts.value, float(ts.t0.value), float(ts.sample_rate.value)
    if p.suffix == ".npy":
        return det, np.load(p), None, 4096.0
    raise ValueError(f"unsupported strain format: {p.suffix}")


def load_psds(psd_dir: str):
    """Load {det}_psd.txt (two columns: f, PSD) or {det}_psd.npy ([2, F])."""
    psds = {}
    d = Path(psd_dir)
    for det in ("H1", "L1", "V1"):
        for cand in (d / f"{det}_psd.txt", d / f"{det}_psd.npy"):
            if cand.exists():
                arr = np.loadtxt(cand) if cand.suffix == ".txt" else np.load(cand)
                arr = arr.T if arr.shape[0] > 2 else arr
                psds[det] = {"frequencies": arr[0], "psd": arr[1]}
                break
    if not psds:
        raise ValueError(f"no {{det}}_psd.txt/.npy files under {psd_dir}")
    return psds


def make_injection(seed: int, min_snr: float = 8.0):
    """One fresh BBH through the exact dataset pipeline (same components as
    training data generation). Returns (whitened strains, raw strains, truth, snr)."""
    from ahsd.data.bilby_pipeline import (BilbyNoiseGenerator, BilbySignalInjector,
                                          BilbyPreprocessor, get_default_psd)
    from ahsd.data.parameter_sampler import ParameterSampler
    from ahsd.models.lean_npe import PARAM_NAMES

    sampler = ParameterSampler()
    noise_gen = BilbyNoiseGenerator(4096, 4.0)
    injector = BilbySignalInjector(4096, 4.0)
    prep = BilbyPreprocessor(4096, 4.0)
    psds = {d: get_default_psd(d) for d in ("H1", "L1", "V1")}
    np.random.seed(seed)  # ParameterSampler draws from the global RNG
    rng = np.random.default_rng(seed)
    while True:
        p = sampler.sample_parameters()
        if p.get("event_type") != "BBH":
            continue
        raw, white, snrs = {}, {}, []
        for det in ("H1", "L1", "V1"):
            noise = noise_gen.generate(det, psds[det], seed=int(rng.integers(2**31)))
            strain, s = injector.inject(noise, p, det, psds[det])
            raw[det] = strain
            white[det] = prep.preprocess(strain, psds[det], detector=det)
            snrs.append(s)
        net = float(np.sqrt(sum(s**2 for s in snrs)))
        if net >= min_snr:
            truth = {n: float(p[n]) for n in PARAM_NAMES}
            return white, raw, truth, net, psds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src_grp = ap.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--event", help="GWOSC event name, e.g. GW150914")
    src_grp.add_argument("--strain", nargs="+", help="strain files (H1*/L1*/V1* prefixed)")
    src_grp.add_argument("--inject", action="store_true",
                         help="simulated injection through the training pipeline")
    ap.add_argument("--gps", type=float, help="trigger GPS time (required with --strain)")
    ap.add_argument("--psd", help="directory of {det}_psd.txt/.npy (else measured/design)")
    ap.add_argument("--model", default="model/lean_npe/best_model.pth")
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--rank", type=int, default=0, help="overlapping-signal rank (0=loudest)")
    ap.add_argument("--n_signals", type=int, default=1,
                    help=">1: infer all overlapping signals and rank them with "
                         "PriorityNet (single-signal events skip the ranking)")
    ap.add_argument("--priority_net", default="model/pn/priority_net_best.pth")
    ap.add_argument("--output", default=None, help="results directory (default: results/<name>)")
    ap.add_argument("--detectors", nargs="+", default=None)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "mps"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare-dynesty", action="store_true",
                    help="also run bilby/dynesty on the same data (slow)")
    ap.add_argument("--nlive", type=int, default=300)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    from ahsd.inference import infer

    psds = load_psds(args.psd) if args.psd else None
    raw_for_dynesty, truth = None, None

    if args.event:
        name = args.event
        posterior = infer(event=args.event, psds=psds, num_samples=args.samples,
                          rank=args.rank, model=args.model, detectors=args.detectors,
                          device=args.device, seed=args.seed)
    elif args.strain:
        if args.gps is None:
            ap.error("--gps is required with --strain")
        name = f"gps{args.gps:.0f}"
        strain, starts, srs = {}, {}, {}
        for f in args.strain:
            det, x, t0, sr = load_strain_file(f)
            strain[det], srs[det] = x, sr
            starts[det] = t0 if t0 is not None else args.gps - len(x) / sr / 2
        posterior = infer(strain=strain, psds=psds, trigger_time=args.gps,
                          segment_start=starts, sample_rates=srs,
                          num_samples=args.samples, rank=args.rank,
                          model=args.model, device=args.device, seed=args.seed)
        raw_for_dynesty = strain
    else:
        name = f"injection{args.seed}"
        white, raw_for_dynesty, truth, snr, inj_psds = make_injection(args.seed)
        print(f"injection: network SNR {snr:.1f}  "
              + "  ".join(f"{k}={v:.3g}" for k, v in list(truth.items())[:3]))
        posterior = infer(strain=white, psds=inj_psds, source="simulated",
                          already_whitened=True, num_samples=args.samples,
                          rank=args.rank, model=args.model, truth=truth,
                          device=args.device, seed=args.seed)

    out = Path(args.output or f"results/{name}")

    if args.n_signals > 1:
        from ahsd.inference.pipeline import infer_overlapping
        multi = infer_overlapping(
            n_signals=args.n_signals, priority_net=args.priority_net,
            strain=posterior.prepared, num_samples=args.samples,
            model=args.model, device=args.device, seed=args.seed)
        print(f"\nPriorityNet order (best first): {multi['ranking']['order']}"
              f"  scores: {multi['ranking']['priority']}")
        for r, res in enumerate(multi["results"]):
            print(f"\n--- NPE rank {r} "
                  f"(priority order {res.config.get('priority_order')}) ---")
            res.summary()
            res.save(out / f"signal_rank{r}", plots=not args.no_plots)
        with open(out / "ranking.json", "w") as f:
            json.dump(multi["ranking"], f, indent=2)
        print(f"saved -> {out}")
        return

    posterior.summary()
    rt = posterior.diagnostics["runtime"]
    print(f"\nruntime: preprocess {rt['preprocess_s']}s | encode {rt['encode_s']}s | "
          f"sampling {rt['sampling_s']}s | total {rt['total_s']}s")
    posterior.save(out, plots=not args.no_plots)
    print(f"saved -> {out}")

    if args.compare_dynesty:
        if raw_for_dynesty is None:
            print("dynesty comparison for --event mode: using the whitened-domain "
                  "data is not valid; fetching raw data again is not implemented — "
                  "use scripts/dynesty_compare.py for simulated batches.")
        else:
            print(f"\nrunning dynesty (nlive={args.nlive}) — this takes minutes ...")
            comp = posterior.compare_to_dynesty(raw_for_dynesty, str(out / "dynesty"),
                                                injection=truth, nlive=args.nlive)
            print(json.dumps({k: v for k, v in comp.items() if isinstance(v, dict)},
                             indent=2)[:2000])
            print(f"comparison -> {out}/dynesty/comparison.json")


if __name__ == "__main__":
    main()
