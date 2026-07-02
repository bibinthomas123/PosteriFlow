#!/usr/bin/env python
"""
Recompute parameter scaler statistics from a generated dataset.

Run AFTER regenerating the dataset with the corrected physics pipeline:
    conda run -n ahsd python src/ahsd/data/scripts/generate_dataset.py \
        --config configs/data_config.yaml

Then run this script:
    conda run -n ahsd python scripts/recompute_scaler_stats.py \
        --dataset data/dataset --output configs/scaler_stats.yaml

The output file can be loaded by TorchParameterScaler to override the
hardcoded statistics in parameter_scalers.py. Set scaler_stats_path in
your training config to point to this file.

Usage as verification only (no output file):
    conda run -n ahsd python scripts/recompute_scaler_stats.py \
        --dataset data/dataset
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

PARAM_NAMES = [
    "mass_1", "mass_2", "luminosity_distance",
    "ra", "dec", "theta_jn", "psi", "phase",
    "geocent_time", "a1", "a2",
]

LOG_PARAMS = {"mass_1", "mass_2", "luminosity_distance"}
LINEAR_PARAMS = {"ra", "dec", "theta_jn", "psi", "phase", "geocent_time"}
BOUNDED_PARAMS = {"a1", "a2"}


def load_dataset(dataset_dir: Path, max_batches: int = 999) -> Dict[str, np.ndarray]:
    """Load single-signal samples from all batch files."""
    batch_dir = dataset_dir / "batches"
    if not batch_dir.exists():
        batch_dir = dataset_dir
    files = sorted(batch_dir.glob("batch_*.pkl"))[:max_batches]
    if not files:
        raise FileNotFoundError(f"No batch files found in {batch_dir}")

    records: Dict[str, List[float]] = {p: [] for p in PARAM_NAMES}
    snrs: List[float] = []
    n_loaded = 0
    n_skipped = 0

    for f in files:
        with open(f, "rb") as fh:
            batch = pickle.load(fh)
        for s in batch["samples"]:
            params = s.get("parameters", [])
            if not params or s.get("n_signals", 1) > 1:
                n_skipped += 1
                continue
            p0 = params[0]
            for k in PARAM_NAMES:
                records[k].append(float(p0.get(k, np.nan)))
            snrs.append(float(s.get("network_snr", np.nan)))
            n_loaded += 1

    _log.info(f"Loaded {n_loaded} single-signal samples ({n_skipped} multi-signal skipped)")
    data = {k: np.array(v) for k, v in records.items()}
    data["network_snr"] = np.array(snrs)
    return data


def compute_stats(data: Dict[str, np.ndarray]) -> Dict:
    """Compute statistics for each parameter."""
    stats = {}
    for param, vals in data.items():
        v = vals[np.isfinite(vals)]
        if len(v) == 0:
            continue
        stat = {"n": len(v), "min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()), "std": float(v.std())}
        if param in LOG_PARAMS or param == "network_snr":
            pos = v[v > 0]
            if len(pos) > 0:
                log_v = np.log(pos)
                stat["log_mean"] = float(log_v.mean())
                stat["log_std"] = float(log_v.std())
        if param in BOUNDED_PARAMS:
            stat["bounded_mean"] = float(v.mean())
            stat["bounded_std"] = float(v.std())
        stats[param] = stat
    return stats


def print_report(stats: Dict, old_values: Dict = None):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("DATASET STATISTICS FOR SCALER CONFIGURATION")
    print("=" * 80)
    print(f"{'Parameter':25s}  {'Type':12s}  {'New stats':30s}  {'Old hardcoded':20s}")
    print("-" * 90)
    for param in PARAM_NAMES + ["network_snr"]:
        s = stats.get(param, {})
        if not s:
            continue
        if param in LOG_PARAMS or param == "network_snr":
            new_str = f"log_mean={s.get('log_mean', float('nan')):.3f}  log_std={s.get('log_std', float('nan')):.3f}"
            old_str = old_values.get(param, "N/A") if old_values else "N/A"
            kind = "log_zscore"
        elif param in BOUNDED_PARAMS:
            new_str = f"mean={s['mean']:.3f}  std={s['std']:.3f}"
            old_str = old_values.get(param, "N/A") if old_values else "N/A"
            kind = "bounded_zscore"
        else:
            new_str = f"min={s['min']:.3f}  max={s['max']:.3f}"
            old_str = ""
            kind = "linear_minmax"
        print(f"  {param:23s}  {kind:12s}  {new_str:30s}  {old_str}")
    print("=" * 80)

    print("\nCOVERAGE CHECK (no clipping expected at ±1.5×FLOW_NORM_BOUND=±4.5)")
    for param in PARAM_NAMES:
        s = stats.get(param, {})
        if not s:
            continue
        if param in LOG_PARAMS:
            log_v = np.log(np.array([s['min'], s['max']]))
            z_min = (log_v[0] - s['log_mean']) / (s['log_std'] + 1e-8)
            z_max = (log_v[1] - s['log_mean']) / (s['log_std'] + 1e-8)
            if abs(z_min) > 4.0 or abs(z_max) > 4.0:
                print(f"  WARNING: {param}: log-zscore range [{z_min:.2f}, {z_max:.2f}] — risk of clipping!")
        elif param in BOUNDED_PARAMS:
            z_min = (s['min'] - s['mean']) / (s['std'] + 1e-8)
            z_max = (s['max'] - s['mean']) / (s['std'] + 1e-8)
            if abs(z_min) > 4.0 or abs(z_max) > 4.0:
                print(f"  WARNING: {param}: zscore range [{z_min:.2f}, {z_max:.2f}] — risk of clipping!")


def save_yaml(stats: Dict, output_path: Path):
    try:
        import yaml
    except ImportError:
        _log.error("PyYAML not available; cannot save YAML output")
        return
    out = {}
    for param in PARAM_NAMES + ["network_snr"]:
        s = stats.get(param, {})
        if not s:
            continue
        if param in LOG_PARAMS or param == "network_snr":
            out[param] = {"log_mean": round(s["log_mean"], 4), "log_std": round(s["log_std"], 4)}
        elif param in BOUNDED_PARAMS:
            out[param] = {"mean": round(s["mean"], 4), "std": round(s["std"], 4)}
    with open(output_path, "w") as f:
        yaml.dump({"scaler_stats": out}, f, default_flow_style=False)
    _log.info(f"Saved scaler stats → {output_path}")


# Old hardcoded values for comparison
_OLD_HARDCODED = {
    "mass_1": "log_mean=2.660  log_std=1.354",
    "mass_2": "log_mean=1.939  log_std=1.459",
    "luminosity_distance": "log_minmax [10, 5000]",
    "a1": "mean=0.249  std=0.236",
    "a2": "mean=0.173  std=0.186",
    "network_snr": "log_mean=3.359  log_std=0.405",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/dataset",
                        help="Path to dataset directory containing batches/")
    parser.add_argument("--output", default=None,
                        help="Path to save YAML scaler stats (optional)")
    parser.add_argument("--max-batches", type=int, default=999)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    data = load_dataset(dataset_dir, max_batches=args.max_batches)
    stats = compute_stats(data)
    print_report(stats, old_values=_OLD_HARDCODED)

    if args.output:
        save_yaml(stats, Path(args.output))
    else:
        _log.info("No --output given; statistics printed only. "
                  "Regenerate dataset first, then run with --output configs/scaler_stats.yaml")


if __name__ == "__main__":
    main()
