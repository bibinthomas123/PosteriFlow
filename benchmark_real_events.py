#!/usr/bin/env python3
"""
Real-event benchmark: LeanNPE vs bilby/dynesty on published GWTC events.

    python benchmark_real_events.py                      # all six events
    python benchmark_real_events.py --events GW150914 GW170814
    python benchmark_real_events.py --skip-dynesty       # NPE-only pass

Per event: GWOSC download -> shared preprocessing -> LeanNPE posterior AND
bilby/dynesty posterior on the identical 4 s data with identical measured
PSDs, matched priors, and identical conventions (mass_1>=mass_2, geocent_time
relative to the trigger). Metrics per parameter: 1-D KL divergence (NPE vs
dynesty), Wasserstein distance, median offset in dynesty sigmas, width ratio.
Published GWTC medians (detector frame) act as pseudo-truth for a coverage
check. Everything lands under --outdir with a summary.json + printed table.

Caveats recorded in the output: GW170817 (BNS) is analyzed from 60 Hz so the
waveform fits the 4 s window, with BNS mass/spin priors; events observed by
two detectors get a white-noise fill on the third for the NPE (flagged).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np

# name, gps, type, published medians (DETECTOR-frame masses = source*(1+z), dL Mpc) from GWTC-1
EVENTS = {
    "GW150914": {"gps": 1126259462.4, "type": "BBH",
                 "truth": {"mass_1": 38.8, "mass_2": 33.4, "luminosity_distance": 440.0}},
    "GW151226": {"gps": 1135136350.6, "type": "BBH",
                 "truth": {"mass_1": 14.9, "mass_2": 8.4, "luminosity_distance": 450.0}},
    "GW170104": {"gps": 1167559936.6, "type": "BBH",
                 "truth": {"mass_1": 37.3, "mass_2": 24.2, "luminosity_distance": 990.0}},
    "GW170608": {"gps": 1180922494.5, "type": "BBH",
                 "truth": {"mass_1": 11.8, "mass_2": 8.1, "luminosity_distance": 320.0}},
    "GW170814": {"gps": 1186741861.5, "type": "BBH",
                 "truth": {"mass_1": 34.3, "mass_2": 28.2, "luminosity_distance": 580.0}},
    "GW170817": {"gps": 1187008882.4, "type": "BNS",
                 "truth": {"mass_1": 1.48, "mass_2": 1.26, "luminosity_distance": 40.0}},
}

KL_BINS = 50


def kl_1d(p_samples: np.ndarray, q_samples: np.ndarray) -> float:
    """KL(p || q) from samples via a shared histogram (add-eps smoothing)."""
    lo = min(p_samples.min(), q_samples.min())
    hi = max(p_samples.max(), q_samples.max())
    if hi <= lo:
        return 0.0
    bins = np.linspace(lo, hi, KL_BINS + 1)
    p, _ = np.histogram(p_samples, bins=bins, density=False)
    q, _ = np.histogram(q_samples, bins=bins, density=False)
    p = p / p.sum() + 1e-10
    q = q / q.sum() + 1e-10
    return float(np.sum(p * np.log(p / q)))


def overlay_corner(npe, dyn, names, labels, save):
    try:
        import corner as _corner
        import matplotlib.pyplot as plt
        rng = [(min(npe[:, j].min(), dyn[:, j].min()),
                max(npe[:, j].max(), dyn[:, j].max())) for j in range(npe.shape[1])]
        fig = _corner.corner(dyn, labels=labels, color="#9aa4b2", range=rng,
                             hist_kwargs={"density": True}, plot_datapoints=False)
        _corner.corner(npe, fig=fig, color="#3b6fd4", range=rng,
                       hist_kwargs={"density": True}, plot_datapoints=False)
        fig.suptitle("blue = LeanNPE, gray = dynesty", y=1.0)
        fig.savefig(save, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  overlay corner skipped: {e}")


def run_event(name: str, meta: dict, args, out: Path) -> dict:
    from ahsd.inference import infer, prepare_real, fetch_gwosc
    from ahsd.inference.dynesty_bridge import (run_dynesty, align_conventions,
                                               BILBY_NAMES)
    from ahsd.models.lean_npe import PARAM_NAMES

    rec = {"event": name, "gps": meta["gps"], "type": meta["type"], "caveats": []}
    print(f"\n=== {name} ({meta['type']}) GPS {meta['gps']} ===")

    # ---- shared data: one fetch, one preprocessing ----
    t0 = time.time()
    gps, raw, starts, srs, found = fetch_gwosc(meta["gps"])
    rec["detectors"] = found
    prepared = prepare_real(raw, gps, starts, srs, seed=0)
    rec["t_fetch_s"] = round(time.time() - t0, 1)
    if len(found) < 3:
        rec["caveats"].append(f"only {found} available; NPE gets white-noise fill "
                              "for the missing detector(s)")

    # ---- LeanNPE ----
    result = infer(strain=prepared, num_samples=args.samples, model=args.model,
                   device=args.device, seed=0)
    rec["t_npe_s"] = result.diagnostics["runtime"]["total_s"]
    rec["npe_ood_flags"] = result.diagnostics["ood_flags"]
    result.truth = dict(meta["truth"])
    result.save(out / name / "npe", plots=not args.no_plots)
    npe = result.samples

    comparison = {}
    dyn_post = None
    if not args.skip_dynesty:
        # ---- dynesty on the identical data ----
        raw4 = {}
        for det in found:
            i0 = int(round((gps - starts[det]) * 4096)) - 8192
            raw4[det] = np.asarray(raw[det][i0: i0 + 16384], dtype=np.float64)
        kw = {}
        if meta["type"] == "BNS":
            kw = {"mass_range": (1.0, 3.0), "spin_max": 0.05,
                  "distance_range": (5.0, 300.0), "minimum_frequency": 60.0}
            rec["caveats"].append("BNS: f_min=60 Hz so the waveform fits the 4 s "
                                  "window; BNS mass/spin/distance priors")
        t0 = time.time()
        res = run_dynesty(raw4, prepared.psds, out / name / "bilby", name,
                          center_gps=gps, trigger_offset=0.0,
                          nlive=args.nlive, **kw)
        rec["t_dynesty_s"] = round(time.time() - t0, 1)
        dyn_post = align_conventions(res.posterior, gps)

        for j, pname in enumerate(PARAM_NAMES):
            d = dyn_post[BILBY_NAMES[pname]].values
            n_ = npe[:, j]
            comparison[pname] = {
                "npe_med": float(np.median(n_)), "npe_std": float(n_.std()),
                "dynesty_med": float(np.median(d)), "dynesty_std": float(d.std()),
                "med_offset_sigma": float((np.median(n_) - np.median(d)) / (d.std() + 1e-9)),
                "width_ratio": float(n_.std() / (d.std() + 1e-9)),
                "kl_npe_dynesty": kl_1d(n_, d),
                "wasserstein": float(__import__("scipy.stats", fromlist=["x"])
                                     .wasserstein_distance(n_, d)),
            }
        key = ["mass_1", "mass_2", "luminosity_distance", "dec", "geocent_time"]
        kj = [PARAM_NAMES.index(k) for k in key]
        overlay_corner(npe[:, kj], dyn_post[[BILBY_NAMES[k] for k in key]].values,
                       key, key, out / name / "corner_overlay.png")

    # ---- pseudo-truth checks vs published medians ----
    truth_check = {}
    for pname, tv in meta["truth"].items():
        j = PARAM_NAMES.index(pname)
        lo, hi = np.quantile(npe[:, j], [0.05, 0.95])
        entry = {"published": tv,
                 "npe_med": float(np.median(npe[:, j])),
                 "npe_in_90": bool(lo <= tv <= hi)}
        if dyn_post is not None:
            d = dyn_post[BILBY_NAMES[pname]].values
            dlo, dhi = np.quantile(d, [0.05, 0.95])
            entry["dynesty_med"] = float(np.median(d))
            entry["dynesty_in_90"] = bool(dlo <= tv <= dhi)
        truth_check[pname] = entry

    rec["comparison"] = comparison
    rec["truth_check"] = truth_check
    with open(out / name / "benchmark.json", "w") as f:
        json.dump(rec, f, indent=2, default=float)
    return rec


def print_summary(records):
    print("\n" + "=" * 100)
    print(f"{'event':<11}{'dets':>6}{'t_NPE':>8}{'t_dyn':>9}{'speedup':>9}"
          f"{'KL(Mc-ish m1)':>14}{'W1(dL)':>10}{'|Δmed| m1(σ)':>13}{'pub. in NPE 90%':>17}")
    for r in records:
        c = r.get("comparison", {})
        tc = r.get("truth_check", {})
        n_in = sum(v["npe_in_90"] for v in tc.values())
        t_dyn = r.get("t_dynesty_s")
        row = (f"{r['event']:<11}{len(r['detectors']):>6}{r['t_npe_s']:>8.1f}"
               f"{(t_dyn if t_dyn else float('nan')):>9.0f}"
               f"{(t_dyn / r['t_npe_s'] if t_dyn else float('nan')):>9.0f}")
        if c:
            row += (f"{c['mass_1']['kl_npe_dynesty']:>14.2f}"
                    f"{c['luminosity_distance']['wasserstein']:>10.1f}"
                    f"{abs(c['mass_1']['med_offset_sigma']):>13.2f}")
        else:
            row += f"{'—':>14}{'—':>10}{'—':>13}"
        row += f"{n_in:>13}/{len(tc)}"
        print(row)
        for cav in r["caveats"]:
            print(f"           note: {cav}")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", nargs="+", default=list(EVENTS),
                    choices=list(EVENTS))
    ap.add_argument("--model", default="model/lean_npe_v3/best_model.pth")
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--nlive", type=int, default=300)
    ap.add_argument("--outdir", default="results/real_event_benchmark")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-dynesty", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.events:
        try:
            records.append(run_event(name, EVENTS[name], args, out))
        except Exception as e:
            print(f"{name} FAILED: {type(e).__name__}: {e}")
            records.append({"event": name, "error": str(e), "caveats": [],
                            "detectors": [], "t_npe_s": float("nan"),
                            "truth_check": {}})
        with open(out / "summary.json", "w") as f:
            json.dump(records, f, indent=2, default=float)
    print_summary([r for r in records if "error" not in r])
    failed = [r["event"] for r in records if "error" in r]
    if failed:
        print("failed:", failed)
    print(f"\nall outputs -> {out}")


if __name__ == "__main__":
    main()
