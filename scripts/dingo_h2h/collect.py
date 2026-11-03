#!/usr/bin/env python3
"""Collect the DINGO head-to-head: DINGO (network + importance sampling)
against our network and our likelihood correction (results/catalog_v5) on the
same O3 events, each judged against the published detector-frame values with
the catalogue figure's metrics.

Run with the DINGO environment's python (it reads DINGO result files):

    ~/miniforge3/envs/dingo/bin/python scripts/dingo_h2h/collect.py

Writes analysis/dingo_h2h/summary.json and prints a table.
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results/dingo_h2h"
OURS = ROOT / "results/catalog_v5"
OUT = ROOT / "analysis/dingo_h2h"
PARAMS = ("chirp_mass", "mass_ratio", "luminosity_distance")


def wquantile(x, w, qs):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return [float(np.interp(q, c, x)) for q in qs]


def stats(x, w, pub):
    lo, med, hi = wquantile(x, w, (0.05, 0.5, 0.95))
    mean = np.average(x, weights=w)
    std = float(np.sqrt(np.average((x - mean) ** 2, weights=w)))
    return {"med": med, "lo": lo, "hi": hi, "w90": hi - lo, "in90": bool(lo <= pub <= hi),
            "z": (pub - med) / max(std, 1e-12)}


def dingo_result(ev):
    """Final (importance-sampled) DINGO result for an event, or None."""
    from dingo.gw.result import Result
    files = sorted(glob.glob(str(RUNS / ev / "result" / "*importance_sampling*.hdf5")))
    files = [f for f in files if "merge" in f] or files
    if not files:
        return None
    r = Result(file_name=files[-1])
    s = r.samples
    w = s["weights"].to_numpy() if "weights" in s else np.ones(len(s))
    out = {"file": Path(files[-1]).name, "n": int(len(s)),
           "sample_efficiency": float(getattr(r, "sample_efficiency", np.nan) or np.nan),
           "log_evidence": float(getattr(r, "log_evidence", np.nan) or np.nan)}
    return s, w, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    events = [l.strip() for l in open(ROOT / "scripts/dingo_h2h/events.txt")
              if l.strip() and not l.startswith("#")]
    rows = []
    for ev in events:
        rec = {"event": ev}
        ours = OURS / f"{ev}.json"
        if not ours.exists():
            rec["status"] = "no result of ours"
            rows.append(rec)
            continue
        o = json.load(open(ours))
        pub = o["published_det"]
        rec.update(snr=o.get("snr"), published=pub)
        for src in ("npe", "is"):
            if src in o:
                rec[src] = {p: o[src][p] for p in PARAMS if p in o[src]}
        ts, ti = RUNS / ev / "timing_sampling.txt", RUNS / ev / "timing_is.txt"
        if ts.exists() and ti.exists():
            rec["dingo_timing"] = {"sampling_s": float(ts.read_text()), "importance_s": float(ti.read_text())}
        else:   # first test event: one combined timing (its sampling step ran everything)
            t = RUNS / ev / "timing.json"
            rec["dingo_timing"] = None
            if t.exists():
                tj = json.load(open(t))
                rec["dingo_timing"] = {"total_s": tj["sampling_s"] + tj["importance_s"]}
        d = dingo_result(ev)
        if d is None:
            rec["status"] = "no DINGO result"
        else:
            s, w, meta = d
            rec["dingo_meta"] = meta
            rec["dingo"] = {p: stats(s[p].to_numpy(), w, pub[p]) for p in PARAMS
                            if p in s and pub.get(p) is not None}
            rec["status"] = "ok"
        rows.append(rec)

    ok = [r for r in rows if r["status"] == "ok"]
    summary = {"n_events": len(events), "n_compared": len(ok), "per_param": {}}
    print(f"{len(ok)} of {len(events)} events compared\n")
    print(f"{'param':22s} {'method':8s} {'in90':>6s} {'median |z|':>11s} {'median w90 / DINGO':>19s}")
    for p in PARAMS:
        summary["per_param"][p] = {}
        for src in ("dingo", "npe", "is"):
            have = [r for r in ok if src in r and p in r[src]]
            if not have:
                continue
            in90 = float(np.mean([r[src][p]["in90"] for r in have]))
            absz = float(np.median([abs(r[src][p]["z"]) for r in have]))
            ratio = float(np.median([r[src][p].get("w90", r[src][p]["hi"] - r[src][p]["lo"])
                                     / r["dingo"][p]["w90"] for r in have if p in r["dingo"]]))
            summary["per_param"][p][src] = {"n": len(have), "in90": in90, "median_abs_z": absz,
                                            "median_w90_over_dingo": ratio}
            print(f"{p:22s} {src:8s} {in90:6.0%} {absz:11.2f} {ratio:19.2f}")
    t = [r["dingo_timing"] for r in ok if r.get("dingo_timing") and "sampling_s" in r["dingo_timing"]]
    if t:
        summary["dingo_wall_s_median"] = {k: float(np.median([x[k] for x in t]))
                                          for k in ("sampling_s", "importance_s")}
        eff = [r["dingo_meta"]["sample_efficiency"] for r in ok]
        summary["dingo_sample_efficiency_median"] = float(np.nanmedian(eff))
        print(f"\nDINGO wall time (median): sampling {summary['dingo_wall_s_median']['sampling_s']:.0f} s, "
              f"importance sampling {summary['dingo_wall_s_median']['importance_s']:.0f} s; "
              f"sample efficiency {summary['dingo_sample_efficiency_median']:.1%}")
    skipped = [(r["event"], r["status"]) for r in rows if r["status"] != "ok"]
    if skipped:
        print("\nnot compared:", ", ".join(f"{e} ({s})" for e, s in skipped))
    json.dump({"summary": summary, "events": rows}, open(OUT / "summary.json", "w"), indent=1)
    print(f"\n-> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
