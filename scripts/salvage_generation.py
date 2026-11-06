#!/usr/bin/env python3
"""Finish a parallel generation that hit its time limit: merge the shards' completed
batches, write the dataset files generate_parallel.py writes at its end, and build
the memmap cache. Does nothing to a dataset that already merged.

    python scripts/salvage_generation.py data/dataset_x --config configs/x.yaml
"""
import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "."); sys.path.insert(0, "src"); sys.path.insert(0, "src/ahsd/data/scripts")
from generate_parallel import merge

from experiments.remix_data import build_memmap_cache

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--config", required=True)
a = ap.parse_args()
out = Path(a.out)
root = out / "_shards"
if root.exists():
    dirs = sorted(d for d in root.glob("shard_*") if (d / "batches").exists())
    dropped = 0
    for d in dirs:                       # a shard killed mid-write leaves a truncated last batch
        files = sorted((d / "batches").glob("batch_*.pkl"))
        if files:
            try:
                with open(files[-1], "rb") as fh:
                    pickle.load(fh)
            except Exception:
                files[-1].unlink()
                dropped += 1
    cfg = yaml.safe_load(open(a.config))
    fracs = (float(cfg.get("train_frac", 0.8)), float(cfg.get("val_frac", 0.1)))   # as generate_parallel.py
    splits, n = merge(out, dirs, fracs)
    bs = int(cfg.get("save_batch_size", 100))
    yaml.safe_dump({**cfg, "n_samples": n * bs, "parallel_shards": len(dirs), "salvaged": True,
                    "snr_definition": "window+context" if cfg.get("lowrate_seconds") else "window"},
                   open(out / "dataset_config.yaml", "w"), sort_keys=False)
    json.dump({"batches": n, "splits": splits, "truncated_batches_dropped": dropped, "salvaged": True},
              open(out / "generation_manifest.json", "w"), indent=1)
    shutil.rmtree(root)
    print(f"merged {n} batches -> {out} {splits}; dropped {dropped} truncated", flush=True)
for split in ("train", "validation"):
    cache = out / "memmap" / split
    if not (cache / "events.json").exists():
        info = build_memmap_cache(str(out), split, str(cache))
        print(split, "events", len(info["events"]) if "events" in info else info, flush=True)
