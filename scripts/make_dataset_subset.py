#!/usr/bin/env python3
"""Make a training-size ablation of a dataset without copying any data.

    python scripts/make_dataset_subset.py data/dataset_v3 data/dataset_v3_50k --train-batches 400

The subset's train split links the first N train batches of the source; its
validation split, validation memmap cache and dataset_config.yaml are the
source's own, so models trained on the subset and on the full set are scored
on identical validation events. All links are relative. The trainer builds
the subset's train memmap on first use.
"""
import argparse
import json
import os
from pathlib import Path

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--train-batches", type=int, required=True)
a = ap.parse_args()

src, dst = Path(a.src).resolve(), Path(a.dst)
train = sorted((src / "train").glob("batch_*.pkl"))
if len(train) < a.train_batches:
    raise SystemExit(f"{src}/train has only {len(train)} batches")
(dst / "train").mkdir(parents=True, exist_ok=True)
(dst / "memmap").mkdir(exist_ok=True)
for f in train[:a.train_batches]:
    link = dst / "train" / f.name
    if not link.exists():
        link.symlink_to(os.path.relpath(f.resolve(), (dst / "train").resolve()))
for rel in ("validation", "memmap/validation"):
    link = dst / rel
    if not link.exists():
        link.symlink_to(os.path.relpath(src / rel, link.parent.resolve()))
cfg = yaml.safe_load(open(src / "dataset_config.yaml")) or {}
cfg["subset_of"] = str(a.src)
cfg["subset_train_batches"] = a.train_batches
yaml.safe_dump(cfg, open(dst / "dataset_config.yaml", "w"), sort_keys=False)
json.dump({"subset_of": str(a.src), "train_batches": a.train_batches,
           "of_total_train_batches": len(train)}, open(dst / "subset_manifest.json", "w"), indent=1)
print(f"{dst}: {a.train_batches}/{len(train)} train batches of {a.src}; validation shared")
