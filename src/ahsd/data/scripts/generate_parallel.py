#!/usr/bin/env python3
"""
Parallel dataset generation: N independent generate_dataset.py shards, merged
into ONE dataset with the same layout the single-process generator writes
(batches/, train|validation|test/ split links, dataset_config.yaml), so
RemixDataset / build_memmap_cache / validate_checkpoint use it unchanged.

Generation is a single CPU-bound process (~10 events/s); shards with distinct
seeds are statistically independent (parameters, noise and overlaps all derive
from the shard's random_seed), so N shards give ~N x throughput.

Example (local, 8 processes):
    python src/ahsd/data/scripts/generate_parallel.py \
        --config configs/data_config_fixed.yaml --n-samples 50000 \
        --output-dir data/dataset_v3 --shards 8 --seed 2026

On the cluster: scripts/slurm/generate_parallel.sbatch (CPU partition).

Splits are assigned per shard (train/val/test fractions within each shard),
so every split mixes all seeds. Split entries are RELATIVE symlinks, so the
dataset survives being copied or moved.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

GENERATOR = Path(__file__).resolve().parent / "generate_dataset.py"
# shard seeds are spaced so no two shards ever share a seed for any base seed
SEED_STRIDE = 1_000_003


def shard_sizes(n_total: int, n_shards: int):
    base, extra = divmod(n_total, n_shards)
    return [base + (1 if i < extra else 0) for i in range(n_shards)]


def run_shard(i, n, seed, args, shard_dir, log_path):
    """Run one generate_dataset.py process; returns (shard, returncode, seconds)."""
    cmd = [sys.executable, str(GENERATOR), "--config", args.config,
           "--n-samples", str(n), "--output-dir", str(shard_dir),
           "--random-seed", str(seed), "--no-splits", "--no-validate",
           *args.extra]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1"}
    t0 = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return i, rc, time.time() - t0


def count_batches(shard_dirs):
    return sum(len(list((d / "batches").glob("batch_*.pkl"))) for d in shard_dirs
               if (d / "batches").exists())


def merge(out: Path, shard_dirs, fracs):
    """Move every shard's batches into out/batches and build relative split
    links. Batches are interleaved round-robin across shards before the
    contiguous train/val/test cut (same rule as the single-process generator),
    so every split mixes all seeds at any dataset size."""
    batches = out / "batches"
    batches.mkdir(parents=True, exist_ok=True)
    per_shard = [sorted((d / "batches").glob("batch_*.pkl")) for d in shard_dirs]
    order = [f for k in range(max(map(len, per_shard), default=0))
             for files in per_shard if k < len(files) for f in [files[k]]]
    n = len(order)
    n_train = max(1, int(n * fracs[0]))
    n_val = max(1, int(n * fracs[1])) if n > 1 else 0
    splits = {"train": [], "validation": [], "test": []}
    for gid, f in enumerate(order):
        dest = batches / f"batch_{gid:05d}.pkl"
        shutil.move(str(f), dest)
        split = "train" if gid < n_train else ("validation" if gid < n_train + n_val else "test")
        splits[split].append(dest)
    for name, files in splits.items():
        sd = out / name
        sd.mkdir(exist_ok=True)
        for f in files:
            link = sd / f.name
            if not link.exists():
                link.symlink_to(os.path.relpath(f, sd))
    return {k: len(v) for k, v in splits.items()}, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-samples", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--shards", type=int, default=os.cpu_count(),
                    help="number of independent shards (default: CPU count)")
    ap.add_argument("--workers", type=int, default=None,
                    help="shards run at once (default: --shards)")
    ap.add_argument("--seed", type=int, default=2026, help="base seed; shard i uses seed + i*%d" % SEED_STRIDE)
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--validate", action="store_true",
                    help="run the generator's dataset validation on the merged output")
    ap.add_argument("extra", nargs="*", help="extra generate_dataset.py args, after --")
    args = ap.parse_args()

    out = Path(args.output_dir)
    if (out / "batches").exists() and any((out / "batches").iterdir()):
        sys.exit(f"{out}/batches is not empty — refusing to merge into an existing dataset")
    cfg = yaml.safe_load(open(args.config)) or {}
    fracs = (args.train_frac if args.train_frac is not None else float(cfg.get("train_frac", 0.8)),
             args.val_frac if args.val_frac is not None else float(cfg.get("val_frac", 0.1)))

    sizes = shard_sizes(args.n_samples, args.shards)
    seeds = [args.seed + i * SEED_STRIDE for i in range(args.shards)]
    shard_root = out / "_shards"
    shard_dirs = [shard_root / f"shard_{i:03d}" for i in range(args.shards)]
    (out / "logs").mkdir(parents=True, exist_ok=True)
    workers = args.workers or args.shards
    batch_size = int(cfg.get("save_batch_size", 100))
    expected_batches = sum(-(-n // batch_size) for n in sizes)
    print(f"{args.n_samples:,} samples -> {args.shards} shards x ~{sizes[0]:,}, "
          f"{workers} at a time, seeds {seeds[0]}..{seeds[-1]}", flush=True)

    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_shard, i, sizes[i], seeds[i], args, shard_dirs[i],
                            out / "logs" / f"shard_{i:03d}.log") for i in range(args.shards)]
        last = -1
        while not all(f.done() for f in futs):
            time.sleep(15)
            nb = count_batches(shard_dirs)
            if nb != last and nb > 0:
                rate = nb * batch_size / max(time.time() - t0, 1)
                eta = (expected_batches - nb) * batch_size / max(rate, 1e-9)
                print(f"  {nb}/{expected_batches} batches (~{min(nb * batch_size, args.n_samples):,} samples) "
                      f"| {rate:.0f} samples/s | ETA {eta / 60:.0f} min", flush=True)
                last = nb
        for f in futs:
            i, rc, secs = f.result()
            results[i] = (rc, secs)
    failed = [i for i, (rc, _) in results.items() if rc != 0]
    if failed:
        sys.exit(f"shards failed: {failed} — see {out}/logs/shard_XXX.log; nothing merged")

    split_counts, n_batches = merge(out, shard_dirs, fracs)
    elapsed = time.time() - t0
    with open(out / "dataset_config.yaml", "w") as f:
        yaml.safe_dump({**cfg, "n_samples": args.n_samples, "random_seed": seeds,
                        "parallel_shards": args.shards}, f, sort_keys=False)
    manifest = {"n_samples": args.n_samples, "shards": args.shards, "seeds": seeds,
                "shard_sizes": sizes, "batches": n_batches, "splits": split_counts,
                "elapsed_s": round(elapsed, 1), "config": str(args.config)}
    json.dump(manifest, open(out / "generation_manifest.json", "w"), indent=1)
    shutil.rmtree(shard_root)
    print(f"merged {n_batches} batches -> {out}  splits {split_counts}  "
          f"in {elapsed / 60:.1f} min ({args.n_samples / elapsed:.0f} samples/s)", flush=True)

    if args.validate:
        sys.path.insert(0, str(GENERATOR.parent))
        import logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.getLogger("bilby").setLevel(logging.WARNING)
        import generate_dataset as g
        g.validate_generated_dataset(out, cfg)


if __name__ == "__main__":
    main()
