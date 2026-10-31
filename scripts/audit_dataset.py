#!/usr/bin/env python3
"""Full audit of a generated dataset's memmap cache (every event, not a sample).

    python scripts/audit_dataset.py data/dataset_v3 [--dist-range 8 6000]

Checks, per split: structure and overlap fraction; event-type mix; physical
parameter ranges per source type against dataset_config.yaml; the distance
distribution's shape (uniform prior vs the SNR cut); signal integrity (NaN,
inf, all-zero waveforms = failed generation, float16 saturation); per-signal
whitened SNR; noise level (whitened noise should have unit variance); and
duplicate events/noise across parallel shards. Also scans shard logs for
waveform failures. Exits non-zero if any hard check fails.
"""
import argparse
import ast
import collections
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "src")
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

J = {n: i for i, n in enumerate(PARAM_NAMES)}
FAIL, WARN = [], []


def hard(ok, msg):
    print(("  ✓ " if ok else "  ✗ ") + msg)
    if not ok:
        FAIL.append(msg)


def soft(ok, msg):
    print(("  ✓ " if ok else "  ⚠ ") + msg)
    if not ok:
        WARN.append(msg)


def source_type(m1, m2):
    return np.where(m1 < 3, "BNS", np.where(m2 < 3, "NSBH", "BBH"))


def chunked(arr, fn, step=2048):
    out = []
    for i in range(0, len(arr), step):
        out.append(fn(np.asarray(arr[i:i + step], dtype=np.float32)))
    return np.concatenate(out) if out else np.zeros(0)


def audit_logs(ds):
    print("\n[logs] generation shards")
    logs = sorted(glob.glob(f"{ds}/logs/shard_*.log")) or sorted(glob.glob(f"{ds}/logs/*.log"))
    fails, done = 0, []
    for f in logs:
        txt = open(f, errors="replace").read()
        fails += txt.count("Waveform failed")
        m = re.search(r"Done\. (\{.*\})", txt)
        if m:
            done.append(ast.literal_eval(re.sub(r"np\.str_\('([^']*)'\)", r"'\1'", m.group(1))))
    hard(fails == 0, f"waveform failures in logs: {fails}")
    if done:
        gen = sum(d["generated"] for d in done)
        rej = sum(d["rejected_snr"] for d in done)
        print(f"    shards finished {len(done)}/{len(logs)}; generated {gen:,}, rejected by SNR cut "
              f"{rej:,} ({rej / (gen + rej):.0%} of draws)")
        hard(len(done) == len(logs), "every shard log reports Done")


def audit_split(ds, split, cfg, dist_range):
    c = Path(ds) / "memmap" / split
    if not (c / "events.json").exists():
        hard(False, f"{split}: memmap cache missing at {c}")
        return None
    ev = json.load(open(c / "events.json"))
    ev = ev["events"] if isinstance(ev, dict) else ev
    par = np.load(c / "params.npy", mmap_mode="r")
    sig = np.load(c / "signals.npy", mmap_mode="r")
    noi = np.load(c / "noise.npy", mmap_mode="r")
    print(f"\n[{split}] {len(ev):,} events, {len(sig):,} signals, {len(noi):,} noise segments, "
          f"strain {sig.shape[1:]} {sig.dtype}")

    nsig = np.array([n for _, n in ev])
    hard(int(nsig.sum()) == len(sig) == len(par), "event index covers every signal exactly once")
    starts = np.array([s for s, _ in ev])
    hard(bool(np.all(np.diff(starts) == nsig[:-1])), "event index is contiguous")
    of = float((nsig > 1).mean())
    want = float(cfg.get("overlap_fraction", 0.3))
    soft(abs(of - want) < 0.05, f"overlap fraction {of:.3f} (config {want}); signals/event "
         + ", ".join(f"{k}:{int((nsig == k).sum())}" for k in sorted(set(nsig))))

    p = np.asarray(par, dtype=np.float64)
    m1, m2, d = p[:, J["mass_1"]], p[:, J["mass_2"]], p[:, J["luminosity_distance"]]
    typ = source_type(m1, m2)

    # ---- physical ranges ---------------------------------------------------
    hard(bool(np.all(np.isfinite(p))), "all parameters finite")
    hard(bool(np.all(m1 >= m2)), f"mass_1 >= mass_2 everywhere (violations {int((m1 < m2).sum())})")
    for nm, lo, hi in (("a1", 0, 1), ("a2", 0, 1), ("theta_jn", 0, np.pi), ("psi", 0, np.pi),
                       ("ra", 0, 2 * np.pi), ("phase", 0, 2 * np.pi), ("dec", -np.pi / 2, np.pi / 2)):
        v = p[:, J[nm]]
        hard(bool(np.all((v >= lo - 1e-6) & (v <= hi + 1e-6))), f"{nm} within [{lo:.3g}, {hi:.3g}]")
    tc = p[:, J["geocent_time"]]
    print(f"    geocent_time: median {np.median(tc):+.3f} s, range [{tc.min():+.2f}, {tc.max():+.2f}]")
    lo_s, hi_s = dist_range
    out = (d < lo_s) | (d > hi_s)
    hard(not out.any(), f"distance inside model support [{lo_s:g}, {hi_s:g}] Mpc ({int(out.sum())} outside)")

    # ---- event mix + per-type distance ------------------------------------
    ranges = cfg.get("distance_ranges", {})
    prior = str(cfg.get("distance_prior", "comoving_d2"))
    mix = cfg.get("event_type_distribution") or {"BBH": 1, "NSBH": 1, "BNS": 1}
    expected = {t.upper() for t, f in mix.items() if t != "noise" and float(f) > 0}
    print(f"    source mix (signals): " + ", ".join(f"{t} {(typ == t).mean():.1%}" for t in ("BBH", "NSBH", "BNS")))
    for t in ("BBH", "NSBH", "BNS"):
        k = typ == t
        if t not in expected:
            # single-class datasets: other classes must be absent, not present
            hard(not k.any(), f"{t}: absent as configured ({int(k.sum())} found)")
            continue
        if not k.any():
            hard(False, f"{t}: no signals at all")
            continue
        r_lo, r_hi = ranges.get(t, [d[k].min(), d[k].max()])
        inr = (d[k] >= r_lo - 1e-6) & (d[k] <= r_hi + 1e-6)
        hard(bool(inr.all()), f"{t}: distance within config range [{r_lo:g}, {r_hi:g}] ({int((~inr).sum())} outside)")
        # shape: deciles of (d - lo)/(hi - lo) against uniform; the SNR cut removes
        # far quiet sources, so a falling far tail is expected, a hole is not
        u = (d[k] - r_lo) / (r_hi - r_lo)
        h, _ = np.histogram(u, bins=10, range=(0, 1))
        frac = h / h.sum()
        print(f"    {t:4s} n={k.sum():6,d}  m1 [{m1[k].min():5.2f},{m1[k].max():6.2f}]  "
              f"dL [{d[k].min():7.1f},{d[k].max():7.1f}]  dL deciles (near->far, uniform=0.10): "
              + " ".join(f"{x:.2f}" for x in frac))
        soft(bool(frac.min() > 0.01), f"{t}: no empty distance decile ({prior} prior + SNR cut)")

    # ---- signal integrity ------------------------------------------------
    finite = chunked(sig, lambda x: np.isfinite(x).all(axis=(1, 2)))
    hard(bool(finite.all()), f"signals finite ({int((~finite).sum())} with NaN/inf)")
    snr = chunked(sig, lambda x: np.sqrt((x.astype(np.float64) ** 2).sum(axis=(1, 2))))
    zero = snr == 0
    hard(not zero.any(), f"no all-zero signal waveforms ({int(zero.sum())}; = failed generation)"
         + (" by type " + str({t: int((zero & (typ == t)).sum()) for t in ('BBH', 'NSBH', 'BNS')}) if zero.any() else ""))
    sat = chunked(sig, lambda x: (np.abs(x) > 6.0e4).any(axis=(1, 2)))
    hard(not sat.any(), f"no float16-saturated signals ({int(sat.sum())})")
    print("    per-signal whitened SNR (network): " + " | ".join(
        f"{t} median {np.median(snr[typ == t]):.1f} p5 {np.percentile(snr[typ == t], 5):.1f} "
        f"p95 {np.percentile(snr[typ == t], 95):.1f}" for t in ("BBH", "NSBH", "BNS") if (typ == t).any()))
    # min_snr is applied per EVENT (loudest), so quiet secondary signals are fine
    loud = np.array([snr[s:s + n].max() for s, n in ev])
    min_snr = float(cfg.get("min_snr", 0))
    soft(bool((loud >= 0.95 * min_snr).mean() > 0.99),
         f"loudest signal per event >= min_snr {min_snr:g}: {(loud >= 0.95 * min_snr).mean():.2%}")

    # ---- noise -----------------------------------------------------------
    nfin = chunked(noi, lambda x: np.isfinite(x).all(axis=(1, 2)))
    hard(bool(nfin.all()), f"noise finite ({int((~nfin).sum())} bad)")
    nstd = chunked(noi, lambda x: x.std(axis=2).mean(axis=1))
    soft(bool(0.8 < np.median(nstd) < 1.25), f"whitened noise std median {np.median(nstd):.3f} "
         f"(p1 {np.percentile(nstd, 1):.3f}, p99 {np.percentile(nstd, 99):.3f}; expect ~1)")
    soft(bool((nstd < 0.1).mean() < 0.001), f"near-silent noise segments: {int((nstd < 0.1).sum())}")

    # ---- duplicates (cross-shard seeding) ----------------------------------
    # exact comparisons (content hashes): a fingerprint of a few samples
    # collides by chance at this pool size (birthday problem)
    n_dup = len(p) - len(np.unique(np.ascontiguousarray(par).view(np.void(par.dtype.itemsize * par.shape[1]))))
    hard(n_dup == 0, f"duplicate signal parameter sets (exact): {n_dup}")
    full, det = collections.Counter(), collections.Counter()
    for i in range(0, len(noi), 4096):
        for x in np.asarray(noi[i:i + 4096]):
            full[hashlib.blake2b(x.tobytes(), digest_size=16).digest()] += 1
            for k in range(x.shape[0]):
                det[(k, hashlib.blake2b(x[k].tobytes(), digest_size=16).digest())] += 1
    n_dup_n = sum(c - 1 for c in full.values() if c > 1)
    hard(n_dup_n == 0, f"duplicate noise segments (exact, all detectors): {n_dup_n}")
    # one detector's noise repeating elsewhere: harmless at this rate (noise is
    # re-drawn per detector every epoch) but flags a seeding problem
    n_det = sum(c - 1 for c in det.values() if c > 1)
    soft(n_det <= 1e-4 * len(noi) * noi.shape[1],
         f"single-detector noise repeats (exact): {n_det} of {len(noi) * noi.shape[1]:,} detector segments")
    return {"typ": typ, "d": d, "snr": snr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--dist-range", type=float, nargs=2, default=(8.0, 6000.0))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(Path(a.dataset) / "dataset_config.yaml")) or {}
    man = Path(a.dataset) / "generation_manifest.json"
    print(f"dataset {a.dataset}: prior {cfg.get('distance_prior')}, ranges {cfg.get('distance_ranges')}")
    if man.exists():
        m = json.load(open(man))
        print(f"  manifest: {m['n_samples']:,} samples, {m['shards']} shards, splits {m['splits']}, "
              f"{m['elapsed_s'] / 60:.1f} min")
        hard(all(v > 0 for v in m["splits"].values()), "every split has batches")
    audit_logs(a.dataset)
    for split in ("train", "validation"):
        audit_split(a.dataset, split, cfg, a.dist_range)
    print(f"\nRESULT: {len(FAIL)} hard failure(s), {len(WARN)} warning(s)")
    for f in FAIL:
        print("  FAIL", f)
    for w in WARN:
        print("  WARN", w)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
