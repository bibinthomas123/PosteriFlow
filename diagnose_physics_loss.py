#!/usr/bin/env python
"""
Diagnose why validation physics loss is 12.4x higher than training.
"""

import torch
import numpy as np
import logging
from experiments.train_priority_net import ChunkedGWDataLoader
from ahsd.models.overlap_neuralpe import OverlapNeuralPE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
import yaml
with open('configs/enhanced_training.yaml') as f:
    config = yaml.safe_load(f)

# Parameter bounds from OverlapNeuralPE
param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time']
param_bounds = {
    "mass_1": (1.0, 100.0),
    "mass_2": (1.0, 100.0),
    "luminosity_distance": (20.0, 8000.0),
    "geocent_time": (-0.1, 0.1),
    "ra": (0.0, 2 * np.pi),
    "dec": (-np.pi / 2, np.pi / 2),
    "theta_jn": (0.0, np.pi),
    "psi": (0.0, np.pi),
    "phase": (0.0, 2 * np.pi),
}

def extract_params_from_dicts(param_dicts, param_names):
    """Convert list of parameter dicts to numpy array."""
    params = []
    for p_dict in param_dicts:
        if isinstance(p_dict, dict):
            row = [p_dict.get(name, np.nan) for name in param_names]
            params.append(row)
    return np.array(params, dtype=np.float32) if params else np.array([], dtype=np.float32)

def check_bounds_violations(params, param_names, param_bounds):
    """Count parameters outside bounds."""
    if len(params) == 0:
        return {}
    violations = {}
    for i, name in enumerate(param_names):
        min_val, max_val = param_bounds[name]
        out_of_bounds = (params[:, i] < min_val) | (params[:, i] > max_val)
        n_violations = out_of_bounds.sum()
        if n_violations > 0:
            violations[name] = {
                'count': int(n_violations),
                'min_val': min_val,
                'max_val': max_val,
                'actual_min': float(params[:, i].min()),
                'actual_max': float(params[:, i].max()),
            }
    return violations

# Load training and validation data
train_loader = ChunkedGWDataLoader(dataset_path='data/dataset', split='train', max_samples=None)
val_loader = ChunkedGWDataLoader(dataset_path='data/dataset', split='validation', max_samples=None)

logger.info("=" * 80)
logger.info("TRAINING DATA ANALYSIS")
logger.info("=" * 80)

train_params_all = []
for i, sample in enumerate(train_loader.iter_all_samples()):
    if i >= 50:  # Sample first 50
        break
    params = sample.get('parameters', [])
    if isinstance(params, list) and len(params) > 0:
        train_params_all.extend(params)

if train_params_all:
    train_params = extract_params_from_dicts(train_params_all, param_names)
else:
    train_params = np.array([], dtype=np.float32)
train_violations = check_bounds_violations(train_params, param_names, param_bounds)

logger.info(f"Training samples: {len(train_params)}")
logger.info(f"Training violations: {len(train_violations)}")
for param, info in train_violations.items():
    logger.info(f"  {param}: {info['count']} violations")
    logger.info(f"    Bounds: [{info['min_val']:.2f}, {info['max_val']:.2f}]")
    logger.info(f"    Actual: [{info['actual_min']:.2f}, {info['actual_max']:.2f}]")

logger.info("\n" + "=" * 80)
logger.info("VALIDATION DATA ANALYSIS")
logger.info("=" * 80)

val_params_all = []
for i, sample in enumerate(val_loader.iter_all_samples()):
    if i >= 50:  # Sample first 50
        break
    params = sample.get('parameters', [])
    if isinstance(params, list) and len(params) > 0:
        val_params_all.extend(params)

if val_params_all:
    val_params = extract_params_from_dicts(val_params_all, param_names)
else:
    val_params = np.array([], dtype=np.float32)
val_violations = check_bounds_violations(val_params, param_names, param_bounds)

logger.info(f"Validation samples: {len(val_params)}")
logger.info(f"Validation violations: {len(val_violations)}")
for param, info in val_violations.items():
    logger.info(f"  {param}: {info['count']} violations")
    logger.info(f"    Bounds: [{info['min_val']:.2f}, {info['max_val']:.2f}]")
    logger.info(f"    Actual: [{info['actual_min']:.2f}, {info['actual_max']:.2f}]")

logger.info("\n" + "=" * 80)
logger.info("PARAMETER STATISTICS COMPARISON")
logger.info("=" * 80)

if len(train_params) > 0 and len(val_params) > 0:
    for i, name in enumerate(param_names):
        logger.info(f"\n{name}:")
        logger.info(f"  Train:  mean={train_params[:, i].mean():.4f}, std={train_params[:, i].std():.4f}, "
                    f"min={train_params[:, i].min():.4f}, max={train_params[:, i].max():.4f}")
        logger.info(f"  Val:    mean={val_params[:, i].mean():.4f}, std={val_params[:, i].std():.4f}, "
                    f"min={val_params[:, i].min():.4f}, max={val_params[:, i].max():.4f}")
        logger.info(f"  Bounds: [{param_bounds[name][0]:.4f}, {param_bounds[name][1]:.4f}]")
else:
    logger.warning("Insufficient data for statistics comparison")
