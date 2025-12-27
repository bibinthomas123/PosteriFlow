
#!/usr/bin/env python3
"""
Complete Neural PE + BiasCorrector Testing Suite
Tests OverlapNeuralPE posterior quality, scaling, and BiasCorrector performance
with comprehensive metrics: NLL, calibration, ROC-AUC, PPV, NPV, MAE, coverage
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from scipy import stats, integrate
from scipy.stats import wasserstein_distance, pearsonr, spearmanr
from scipy.special import xlogy
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# DATASET LOADER
# ============================================================================

class TestDataset(torch.utils.data.Dataset):
    """Dataset loader for neural PE testing"""
    
    def __init__(self, data_path, split='test', max_samples=None):
        self.data_path = Path(data_path)
        self.split = split
        self.samples = self._load_samples(max_samples)
        logger.info(f"✅ Loaded {len(self.samples)} {split} samples")
    
    def _load_samples(self, max_samples):
        samples = []
        
        # Try to find the split directory
        split_dir = self.data_path / self.split
        
        # If split directory doesn't exist, try all subdirectories
        if not split_dir.exists():
            logger.warning(f"⚠️  Split '{self.split}' not found in {self.data_path}")
            logger.warning(f"   Searching for any available data...")
            # Try to find any chunk files in subdirectories
            for subdir in self.data_path.iterdir():
                if subdir.is_dir():
                    chunk_files = sorted(subdir.glob('chunk_*.pkl'))
                    if chunk_files:
                        logger.info(f"   Found {len(chunk_files)} chunks in {subdir.name}")
                        split_dir = subdir
                        break
            
            # If still no chunks found, use parent directory directly
            if not (split_dir.exists()):
                logger.warning(f"   Trying parent directory {self.data_path}")
                chunk_files = sorted(self.data_path.glob('chunk_*.pkl'))
                if not chunk_files:
                    logger.error(f"No data found in {self.data_path}")
                    return samples
                split_dir = self.data_path
        
        chunk_files = sorted(split_dir.glob('chunk_*.pkl'))
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    samples.extend(chunk_data)
                    
                    if max_samples and len(samples) >= max_samples:
                        return samples[:max_samples]
            except Exception as e:
                logger.warning(f"Failed to load {chunk_file}: {e}")
                continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract strain from detector_data for all 3 detectors
        detector_data = sample.get('detector_data', {})
        
        strains = []
        for detector_name in ['H1', 'L1', 'V1']:
            det = detector_data.get(detector_name, {})
            
            strain = None
            if isinstance(det, dict):
                data = det.get('strain')
                if data is not None:
                    # Handle both numpy arrays and tensors
                    if hasattr(data, 'numpy'):
                        strain = data.numpy()
                    elif isinstance(data, np.ndarray):
                        strain = data
                    else:
                        strain = np.array(data, dtype=np.float32)
            
            if strain is None:
                strain = np.zeros(16384, dtype=np.float32)
            
            # Ensure correct length
            strain = np.asarray(strain, dtype=np.float32)
            if strain.size < 16384:
                strain = np.pad(strain.flatten(), (0, 16384 - strain.size))
            elif strain.size > 16384:
                strain = strain.flatten()[:16384]
            
            strains.append(strain)
        
        # Stack all 3 detectors: [n_det=3, n_samples=16384]
        strain_array = np.stack(strains, axis=0)
        
        # Extract true parameters from first signal
        parameters = sample.get('parameters', [{}])
        if not parameters or parameters is None:
            signal_params = {}
        else:
            signal_params = parameters[0] if isinstance(parameters, list) and len(parameters) > 0 else {}
        
        # Ensure signal_params is a dict, not None
        if signal_params is None:
            signal_params = {}
        
        # Extract params - critical fix: Handle missing keys properly
        # If a parameter is missing, we cannot use defaults (they don't match data distribution)
        # Instead, skip samples with missing parameters
        required_keys = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
                        'theta_jn', 'psi', 'phase', 'geocent_time', 'a1', 'a2']
        
        if not all(key in signal_params for key in required_keys):
            # Return zeros for missing sample (will be filtered)
            true_params = np.zeros(11, dtype=np.float32)
        else:
            true_params = np.array([
                signal_params.get('mass_1'),
                signal_params.get('mass_2'),
                signal_params.get('luminosity_distance'),
                signal_params.get('ra'),
                signal_params.get('dec'),
                signal_params.get('theta_jn'),
                signal_params.get('psi'),
                signal_params.get('phase'),
                signal_params.get('geocent_time'),
                signal_params.get('a1'),  # Primary spin magnitude
                signal_params.get('a2')   # Secondary spin magnitude
            ], dtype=np.float32)
        
        # Return: strain [3, 16384], true_params [11]
        return torch.FloatTensor(strain_array), torch.FloatTensor(true_params)


# ============================================================================
# TEST 1: POSTERIOR SANITY CHECKS
# ============================================================================

def test_posterior_sanity(model, test_loader, device, n_samples=500):
    """Test 1: Basic posterior quality and validity checks"""
    
    logger.info("=" * 80)
    logger.info("TEST 1: POSTERIOR SANITY CHECKS")
    logger.info("=" * 80)
    
    model.eval()
    results = {
        'nan_count': 0,
        'inf_count': 0,
        'out_of_bounds': 0,
        'total_samples': 0,
        'avg_rejection_rate': [],
        'valid_samples_ratio': []
    }
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
                   'theta_jn', 'psi', 'phase', 'geocent_time', 'a1', 'a2']
    
    with torch.no_grad():
        batch_count = 0
        param_oob_counts = {name: 0 for name in param_names}  # Per-param debugging
        for strain, true_params in tqdm(test_loader, desc="Sanity checks"):
            strain = strain.to(device)
            batch_size = strain.shape[0]
            
            try:
                # Sample posterior
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                samples = posterior['samples']  # [batch, n_samples, n_params]
                
                # Check for NaNs/Infs
                nan_mask = torch.isnan(samples).any(dim=-1).any(dim=0)
                inf_mask = torch.isinf(samples).any(dim=-1).any(dim=0)
                
                results['nan_count'] += nan_mask.sum().item()
                results['inf_count'] += inf_mask.sum().item()
                
                # Check bounds
                for b in range(batch_size):
                    samples_b = samples[b].cpu().numpy()
                    
                    # Count out of bounds
                    for p_idx, (p_name) in enumerate(param_names):
                        bounds = model.param_bounds[p_name]
                        out_of_bounds = ((samples_b[:, p_idx] < bounds[0]) | 
                                        (samples_b[:, p_idx] > bounds[1])).sum()
                        results['out_of_bounds'] += out_of_bounds
                        param_oob_counts[p_name] += out_of_bounds  # Track per-param
                
                results['total_samples'] += batch_size
                batch_count += 1
                
            except Exception as e:
                logger.warning(f"Sanity check failed: {e}")
                continue
    
    # Compute statistics
    total_values = results['total_samples'] * n_samples * len(param_names)
    nan_pct = (results['nan_count'] / total_values * 100) if total_values > 0 else 0
    inf_pct = (results['inf_count'] / total_values * 100) if total_values > 0 else 0
    oob_pct = (results['out_of_bounds'] / total_values * 100) if total_values > 0 else 0
    
    logger.info(f"\n📊 Sanity Check Results ({results['total_samples']} samples):")
    logger.info(f"   NaN values:         {results['nan_count']} ({nan_pct:.2f}%)")
    logger.info(f"   Inf values:         {results['inf_count']} ({inf_pct:.2f}%)")
    logger.info(f"   Out-of-bounds:      {results['out_of_bounds']} ({oob_pct:.2f}%)")
    
    # Debug: Per-parameter out-of-bounds breakdown
    logger.info(f"\n   📍 Out-of-bounds by parameter:")
    for p_name in param_names:
        count = param_oob_counts[p_name]
        if count > 0:
            pct = (count / total_values * 100) if total_values > 0 else 0
            logger.info(f"      {p_name:20s}: {count:6d} ({pct:.2f}%)")
    
    if nan_pct < 1.0 and inf_pct < 1.0 and oob_pct < 5.0:
        logger.info("   ✅ PASS: Posterior sanity checks")
    else:
        logger.warning("   ⚠️  FAIL: High anomaly rate detected")
    
    return results


# ============================================================================
# TEST 2: NEGATIVE LOG-LIKELIHOOD (NLL)
# ============================================================================

def test_nll(model, test_loader, device, n_samples=500):
     """Test 2: Compute Flow quality - CFM metrics for Flow Matching (NLL N/A)"""
     
     logger.info("\n" + "=" * 80)
     logger.info("TEST 2: FLOW QUALITY METRICS (CFM-based)")
     logger.info("=" * 80)
     
     model.eval()
     
     # Detect flow type
     flow_type = "flowmatching"  # Default
     try:
         # Try to infer flow type from config or model attributes
         if hasattr(model, 'config'):
             flow_type = model.config.get('flow_type', 'flowmatching')
         elif hasattr(model.flow, '__class__'):
             class_name = model.flow.__class__.__name__
             if 'FlowMatching' in class_name or 'CFM' in class_name or 'OT' in class_name:
                 flow_type = 'flowmatching'
             elif 'NSF' in class_name or 'NeuralSpline' in class_name:
                 flow_type = 'nsf'
     except:
         pass
     
     logger.info(f"   Flow Type Detected: {flow_type.upper()}")
     
     # For Flow Matching: Use sample-based metrics (posterior spread, recovery)
     # ALWAYS use CFM metrics - NLL doesn't apply to velocity-matching flows
     if flow_type.lower() in ['flowmatching', 'cfm', 'optimal_transport', 'nsf']:
         logger.info(f"\n   Flow Matching / CFM uses velocity matching, not likelihood")
         logger.info(f"   Computing metrics: posterior spread & parameter recovery\n")
         
         sample_spreads = []
         param_errors = []
         processed = 0
         skipped = 0
         
         with torch.no_grad():
             for strain, true_params in tqdm(test_loader, desc="Computing CFM metrics"):
                 strain = strain.to(device)
                 true_params_np = true_params.cpu().numpy()
                 batch_size = strain.shape[0]
                 
                 try:
                     # Sample posterior
                     posterior = model.sample_posterior(strain, n_samples=n_samples)
                     samples = posterior['samples'].cpu().numpy()  # [batch, n_samples, n_params]
                     
                     for b in range(batch_size):
                         # Skip incomplete samples
                         if true_params_np[b].sum() < 1e-6:
                             skipped += 1
                             continue
                         
                         # Compute posterior spread (std dev of samples)
                         posterior_std = samples[b].std(axis=0).mean()  # Average std across params
                         sample_spreads.append(posterior_std)
                         
                         # Compute parameter recovery (MAE between posterior mean and truth)
                         posterior_mean = samples[b].mean(axis=0)
                         error = np.abs(posterior_mean - true_params_np[b]).mean()
                         param_errors.append(error)
                         
                         processed += 1
                 
                 except Exception as e:
                     logger.warning(f"CFM metrics failed: {e}")
                     skipped += 1
                     continue
         
         if len(sample_spreads) > 0:
             spreads = np.array(sample_spreads)
             errors = np.array(param_errors)
             
             results = {
                 'flow_type': flow_type,
                 'metric_type': 'cfm',
                 'posterior_spread_mean': float(np.mean(spreads)),
                 'posterior_spread_std': float(np.std(spreads)),
                 'posterior_spread_median': float(np.median(spreads)),
                 'param_error_mean': float(np.mean(errors)),
                 'param_error_std': float(np.std(errors)),
                 'param_error_median': float(np.median(errors)),
                 'processed': int(processed),
                 'skipped': int(skipped)
             }
             
             logger.info(f"\n📊 Flow Matching (CFM) Quality Metrics ({processed} samples):")
             logger.info(f"   Posterior Spread (std):  {results['posterior_spread_mean']:.4f} ± {results['posterior_spread_std']:.4f}")
             logger.info(f"   Parameter Error (MAE):   {results['param_error_mean']:.4f} ± {results['param_error_std']:.4f}")
             logger.info(f"   Median spread:           {results['posterior_spread_median']:.4f}")
             logger.info(f"   Median error:            {results['param_error_median']:.4f}")
             
             # Quality assessment for CFM
             if results['param_error_mean'] < 0.1:
                 logger.info("   ✅ EXCELLENT (CFM: mean parameter error < 0.1)")
             elif results['param_error_mean'] < 0.3:
                 logger.info("   ✅ GOOD (CFM: mean parameter error < 0.3)")
             elif results['param_error_mean'] < 0.5:
                 logger.info("   ⚠️  ACCEPTABLE (CFM: mean parameter error < 0.5)")
             else:
                 logger.warning("   ❌ POOR (CFM: parameter error > 0.5, needs retraining)")
             
             if results['posterior_spread_mean'] < 0.05:
                 logger.info("   ⚠️  WARNING: Posterior too narrow (under-dispersed)")
             elif results['posterior_spread_mean'] > 0.5:
                 logger.info("   ⚠️  WARNING: Posterior too wide (over-dispersed)")
             else:
                 logger.info("   ✅ Posterior spread in healthy range")
         else:
             logger.warning("   ❌ No samples processed - check data loading")
             results = {
                 'flow_type': flow_type,
                 'metric_type': 'cfm',
                 'posterior_spread_mean': 0.0,
                 'param_error_mean': 0.0,
                 'processed': 0,
                 'skipped': 0
             }
         
         return results



# ============================================================================
# TEST 3: CALIBRATION (P-P PLOT & COVERAGE)
# ============================================================================

def test_calibration(model, test_loader, device, n_samples=500):
    """Test 3: Posterior calibration via P-P plot and coverage analysis"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: CALIBRATION TEST (P-P PLOT & COVERAGE)")
    logger.info("=" * 80)
    
    model.eval()
    percentiles_list = []
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
                   'theta_jn', 'psi', 'phase', 'geocent_time', 'a1', 'a2']
    
    with torch.no_grad():
        for strain, true_params in tqdm(test_loader, desc="Calibration"):
            strain = strain.to(device)
            true_params_np = true_params.cpu().numpy()
            batch_size = strain.shape[0]
            
            try:
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                samples = posterior['samples'].cpu().numpy()  # [batch, n_samples, n_params]
                
                for b in range(batch_size):
                    # FILTER: Skip samples with zero params (missing/incomplete data)
                    if true_params_np[b].sum() < 1e-6:  # All params near zero = missing data
                        continue
                    
                    sample_percentiles = []
                    
                    for p in range(len(param_names)):
                        posterior_p = samples[b, :, p]
                        true_val = true_params_np[b, p]
                        
                        # Compute percentile where true value lies
                        percentile = (posterior_p < true_val).mean()
                        sample_percentiles.append(percentile)
                    
                    percentiles_list.append(sample_percentiles)
            
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
                continue
    
    percentiles = np.array(percentiles_list)  # [n_samples, n_params]
    
    # Compute coverage at different confidence levels
    coverage_levels = [0.68, 0.90, 0.95]
    coverage_results = {}
    
    for level in coverage_levels:
        lower = (1 - level) / 2
        upper = 1 - lower
        
        in_interval = ((percentiles > lower) & (percentiles < upper)).mean(axis=0)
        coverage_results[level] = in_interval
        
        logger.info(f"\n📊 {level*100:.0f}% Credible Interval Coverage:")
        for i, name in enumerate(param_names):
            cov = in_interval[i] * 100
            target = level * 100
            diff = cov - target
            status = "✅" if abs(diff) < 5 else "⚠️"
            logger.info(f"   {name:20s}: {cov:5.1f}% (target {target:.0f}%, diff {diff:+5.1f}%) {status}")
    
    # Overall calibration score
    ideal_coverage = np.array([0.68, 0.90, 0.95])
    actual_coverage = np.array([coverage_results[l].mean() for l in coverage_levels])
    calibration_error = np.abs(ideal_coverage - actual_coverage).mean()
    
    logger.info(f"\n📈 Overall Calibration:")
    logger.info(f"   Mean calibration error: {calibration_error:.3f}")
    
    if calibration_error < 0.05:
        logger.info("   ✅ EXCELLENT calibration")
    elif calibration_error < 0.10:
        logger.info("   ✅ GOOD calibration")
    elif calibration_error < 0.15:
        logger.info("   ⚠️  ACCEPTABLE calibration")
    else:
        logger.warning("   ❌ POOR calibration")
    
    return {
        'coverage': coverage_results,
        'calibration_error': calibration_error,
        'percentiles': percentiles
    }


# ============================================================================
# TEST 4: POSTERIOR WIDTH & CONTRACTION
# ============================================================================

def test_posterior_width(model, test_loader, device, n_samples=500):
    """Test 4: Posterior width and contraction analysis"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: POSTERIOR WIDTH & CONTRACTION")
    logger.info("=" * 80)
    
    model.eval()
    posterior_stds = []
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
                   'theta_jn', 'psi', 'phase', 'geocent_time', 'a1', 'a2']
    
    # ✅ Parameter bounds (Dec 9, 2025): For relative width computation
    # These match the normalization bounds in the model
    param_bounds = {
        'mass_1': (1.0, 100.0),                  # M_sun
        'mass_2': (1.0, 100.0),
        'luminosity_distance': (10.0, 8000.0),  # Mpc
        'ra': (0.0, 2*np.pi),                   # radians
        'dec': (-np.pi/2, np.pi/2),
        'theta_jn': (0.0, np.pi),
        'psi': (0.0, np.pi),
        'phase': (0.0, 2*np.pi),
        'geocent_time': (-2.0, 8.0),            # seconds (from model bounds)
        'a1': (0.0, 0.99),                      # spin magnitude
        'a2': (0.0, 0.99),
    }
    
    with torch.no_grad():
        for strain, _ in tqdm(test_loader, desc="Width analysis"):
            strain = strain.to(device)
            batch_size = strain.shape[0]
            
            try:
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                samples = posterior['samples'].cpu().numpy()
                
                # Compute std per parameter across n_samples dimension
                # Shape: [batch, n_samples, param_dim] → std along axis 1 → [batch, param_dim]
                post_std = samples.std(axis=1)  # [batch, n_params]
                posterior_stds.append(post_std)
            
            except:
                continue
    
    posterior_stds = np.concatenate(posterior_stds, axis=0)
    
    # Compute mean width per parameter across batch
    mean_widths = posterior_stds.mean(axis=0)
    
    # ✅ REPORT: Absolute widths (physical units) + relative widths (% of range)
    logger.info(f"\n📊 Posterior Widths (relative to parameter bounds):")
    logger.info("   (Model applies 1.35x widening, then clamps to bounds)")
    
    relative_widths = []
    for i, name in enumerate(param_names):
        abs_width = mean_widths[i]
        
        if name in param_bounds:
            min_val, max_val = param_bounds[name]
            param_range = max_val - min_val
            rel_width_pct = (abs_width / param_range) * 100
            relative_widths.append(rel_width_pct)
            
            # Status indicators based on relative width
            if rel_width_pct < 1:
                status = "🔴 Very narrow"
            elif rel_width_pct < 5:
                status = "⚠️  Narrow"
            elif rel_width_pct < 15:
                status = "✅ Reasonable"
            else:
                status = "⚠️  Wide"
            
            logger.info(f"   {name:20s}: {abs_width:8.4f} ({rel_width_pct:5.1f}% of range) {status}")
        else:
            logger.info(f"   {name:20s}: {abs_width:8.4f} (bounds unknown)")
    
    # Overall assessment
    if len(relative_widths) > 0:
        avg_rel_width = np.mean(relative_widths)
        logger.info(f"\n   Average relative width: {avg_rel_width:.1f}% of parameter ranges")
        
        if avg_rel_width < 1:
            logger.warning("   🔴 CRITICAL: Posteriors extremely narrow across all parameters!")
            logger.warning("      This explains PIT U-shape and low coverage")
        elif avg_rel_width < 5:
            logger.warning("   ⚠️  Posteriors narrow - should be 5-15% for good calibration")
        elif avg_rel_width < 15:
            logger.info("   ✅ Posterior widths in reasonable range for good calibration")
        else:
            logger.info("   ⚠️  Posteriors quite wide - may indicate underfitting")
    
    return {
        'posterior_stds': posterior_stds,
        'mean_widths': mean_widths.tolist()
    }


# ============================================================================
# TEST 5: PARAMETER RECOVERY & BIAS
# ============================================================================

def test_parameter_recovery(model, test_loader, device, n_samples=500):
    """Test 5: Parameter recovery accuracy and bias assessment"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: PARAMETER RECOVERY & BIAS")
    logger.info("=" * 80)
    
    model.eval()
    errors = []
    relative_errors = []
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
                   'theta_jn', 'psi', 'phase', 'geocent_time', 'a1', 'a2']
    
    with torch.no_grad():
        for strain, true_params in tqdm(test_loader, desc="Recovery analysis"):
            strain = strain.to(device)
            true_params_np = true_params.cpu().numpy()
            
            try:
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                means = posterior['means'].cpu().numpy()
                
                # Compute errors
                param_errors = means - true_params_np
                param_rel_errors = param_errors / (np.abs(true_params_np) + 1.0)
                
                errors.append(param_errors)
                relative_errors.append(param_rel_errors)
            
            except:
                continue
    
    errors = np.concatenate(errors, axis=0)  # [n_samples, n_params]
    relative_errors = np.concatenate(relative_errors, axis=0)
    
    logger.info(f"\n📊 Parameter Recovery (Absolute Errors):")
    for i, name in enumerate(param_names):
        mae = np.abs(errors[:, i]).mean()
        bias = errors[:, i].mean()
        std = errors[:, i].std()
        
        logger.info(f"   {name:20s}: MAE={mae:8.4f}, Bias={bias:+8.4f}, Std={std:8.4f}")
    
    logger.info(f"\n📊 Parameter Recovery (Relative Errors %):")
    for i, name in enumerate(param_names):
        mae_rel = np.abs(relative_errors[:, i]).mean() * 100
        bias_rel = relative_errors[:, i].mean() * 100
        
        logger.info(f"   {name:20s}: {mae_rel:6.2f}% (bias {bias_rel:+6.2f}%)")
    
    # Check for systematic biases
    overall_bias = np.abs(errors.mean(axis=0)).mean()
    if overall_bias < 0.5:
        logger.info(f"   ✅ Low overall bias: {overall_bias:.4f}")
    else:
        logger.warning(f"   ⚠️  High overall bias detected: {overall_bias:.4f}")
    
    return {
        'errors': errors.tolist(),
        'relative_errors': relative_errors.tolist(),
        'mean_absolute_errors': np.abs(errors).mean(axis=0).tolist(),
        'biases': errors.mean(axis=0).tolist()
    }


# ============================================================================
# TEST 7: SCALING & PERFORMANCE
# ============================================================================

def test_scaling(model, test_loader, device, n_samples=500):
    """Test 7: Scaling and computational performance"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 7: SCALING & PERFORMANCE")
    logger.info("=" * 80)
    
    import time
    
    model.eval()
    inference_times = []
    memory_usage = []
    
    with torch.no_grad():
        for strain, _ in tqdm(test_loader, desc="Performance"):
            strain = strain.to(device)
            batch_size = strain.shape[0]
            
            try:
                # Time inference
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t0 = time.time()
                
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t1 = time.time()
                
                inference_time = (t1 - t0) / batch_size
                inference_times.append(inference_time)
            
            except:
                continue
    
    inference_times = np.array(inference_times)
    
    logger.info(f"\n⏱️  Performance Metrics:")
    logger.info(f"   Per-sample inference: {inference_times.mean():.4f}s ± {inference_times.std():.4f}s")
    logger.info(f"   Throughput:           {1/inference_times.mean():.2f} samples/sec")
    logger.info(f"   95th percentile:      {np.percentile(inference_times, 95):.4f}s")
    
    # Check if model scales well
    if inference_times.mean() < 1.0:
        logger.info("   ✅ Good scaling (< 1.0s per sample)")
    elif inference_times.mean() < 2.0:
        logger.info("   ⚠️  Acceptable scaling (< 2.0s per sample)")
    else:
        logger.warning(f"   ⚠️  Slow scaling ({inference_times.mean():.2f}s per sample)")
    
    return {
        'mean_time': float(inference_times.mean()),
        'std_time': float(inference_times.std()),
        'p95_time': float(np.percentile(inference_times, 95)),
        'throughput': float(1 / inference_times.mean())
    }


# ============================================================================
# TEST 8: ROC-AUC & CLASSIFICATION METRICS
# ============================================================================

def test_classification_metrics(model, test_loader, device, n_samples=500):
    """Test 8: ROC-AUC and precision-recall metrics for parameter accuracy"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 8: CLASSIFICATION METRICS (ROC-AUC, PPV, NPV)")
    logger.info("=" * 80)
    
    model.eval()
    predictions = []
    ground_truth = []
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
                   'theta_jn', 'psi', 'phase', 'geocent_time']
    
    with torch.no_grad():
        for strain, true_params in tqdm(test_loader, desc="Classification metrics"):
            strain = strain.to(device)
            true_params_np = true_params.cpu().numpy()
            
            try:
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                means = posterior['means'].cpu().numpy()
                stds = posterior['stds'].cpu().numpy()
                
                predictions.append(means)
                ground_truth.append(true_params_np)
            
            except:
                continue
    
    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    
    # Compute errors and classify as accurate/inaccurate
    errors = np.abs(predictions - ground_truth)
    
    logger.info(f"\n📊 Per-Parameter Classification Metrics:")
    
    auc_scores = []
    for i, name in enumerate(param_names):
        # Define "accurate" as within 10% error
        threshold = np.max(np.abs(ground_truth[:, i])) * 0.1 + 1.0
        is_accurate = errors[:, i] < threshold
        
        # Use distance from true value as score
        # Higher score = further from true value = less accurate
        pred_score = errors[:, i]
        
        if len(np.unique(is_accurate)) == 2:
            try:
                auc = roc_auc_score(is_accurate, -pred_score)  # Negate so higher = more accurate
                auc_scores.append(auc)
                
                logger.info(f"   {name:20s}: AUC={auc:.4f}")
            except:
                logger.warning(f"   {name:20s}: AUC computation failed")
        else:
            logger.warning(f"   {name:20s}: Only one class present (all accurate or all inaccurate)")
    
    # Overall AUC
    if auc_scores:
        mean_auc = np.mean(auc_scores)
        logger.info(f"\n   Mean AUC: {mean_auc:.4f}")
        
        if mean_auc > 0.90:
            logger.info("   ✅ EXCELLENT discrimination")
        elif mean_auc > 0.80:
            logger.info("   ✅ GOOD discrimination")
        elif mean_auc > 0.70:
            logger.info("   ⚠️  FAIR discrimination")
        else:
            logger.warning("   ❌ POOR discrimination")
    
    return {
        'auc_scores': auc_scores,
        'mean_auc': float(np.mean(auc_scores)) if auc_scores else 0.0
    }


# ============================================================================
# TEST 6: PIT/RANK HISTOGRAM (Probability Integral Transform)
# ============================================================================

def test_pit_histogram(model, test_loader, device, n_samples=500):
    """Test 6: PIT histogram for calibration diagnosis with enhanced KS p-value and shape analysis"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: PIT HISTOGRAM & RANK STATISTICS")
    logger.info("=" * 80)
    
    model.eval()
    pit_values = []
    pit_by_param = {}
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
                   'theta_jn', 'psi', 'phase', 'geocent_time', 'a1', 'a2']
    
    # Initialize per-parameter PIT storage
    for name in param_names:
        pit_by_param[name] = []
    
    with torch.no_grad():
        for strain, true_params in tqdm(test_loader, desc="PIT analysis"):
            strain = strain.to(device)
            true_params_np = true_params.cpu().numpy()
            batch_size = strain.shape[0]
            
            try:
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                samples = posterior['samples'].cpu().numpy()  # [batch, n_samples, n_params]
                
                for b in range(batch_size):
                    # Skip incomplete samples
                    if true_params_np[b].sum() < 1e-6:
                        continue
                    
                    for p in range(len(param_names)):
                        posterior_p = samples[b, :, p]
                        true_val = true_params_np[b, p]
                        
                        # PIT = CDF of posterior at true value
                        pit = (posterior_p < true_val).sum() / len(posterior_p)
                        pit_values.append(pit)
                        pit_by_param[param_names[p]].append(pit)
            
            except Exception as e:
                logger.warning(f"PIT failed: {e}")
                continue
    
    pit_values = np.array(pit_values)
    
    # Test uniformity with Kolmogorov-Smirnov test
    from scipy.stats import kstest, uniform
    ks_stat, ks_pval = kstest(pit_values, 'uniform')
    
    # Compute histogram bins
    pit_bins = np.linspace(0, 1, 11)
    pit_counts, _ = np.histogram(pit_values, bins=pit_bins)
    expected_count = len(pit_values) / 10
    
    logger.info(f"\n📊 PIT Histogram ({len(pit_values)} samples):")
    logger.info(f"   KS test p-value: {ks_pval:.6f} {'✅' if ks_pval > 0.05 else '⚠️'}")
    logger.info(f"   KS statistic: {ks_stat:.6f}")
    
    logger.info(f"\n   Bin counts (expect ~{expected_count:.0f} per bin):")
    for i in range(len(pit_counts)):
        bar = '█' * int(pit_counts[i] / expected_count * 20)
        logger.info(f"   [{i*0.1:.1f}-{(i+1)*0.1:.1f}): {pit_counts[i]:4d} {bar}")
    
    # Diagnose shape - compute key metrics
    lower_tail = pit_counts[0] + pit_counts[1]  # [0.0-0.2)
    upper_tail = pit_counts[8] + pit_counts[9]  # [0.8-1.0)
    middle = pit_counts[3:7].sum()              # [0.3-0.7)
    
    # Compute Anderson-Darling statistic for better uniformity test
    sorted_pit = np.sort(pit_values)
    n = len(sorted_pit)
    i = np.arange(1, n + 1)
    anderson_stat = -n - (1/n) * np.sum((2*i - 1) * (np.log(sorted_pit) + np.log(1 - sorted_pit[::-1])))
    
    logger.info(f"\n   📈 Shape Analysis:")
    logger.info(f"      Lower tail [0.0-0.2):  {lower_tail} ({lower_tail/len(pit_values)*100:.1f}%)")
    logger.info(f"      Middle [0.3-0.7):      {middle} ({middle/len(pit_values)*100:.1f}%)")
    logger.info(f"      Upper tail [0.8-1.0):  {upper_tail} ({upper_tail/len(pit_values)*100:.1f}%)")
    logger.info(f"      Anderson-Darling stat: {anderson_stat:.4f}")
    
    # Diagnose shape
    if ks_pval > 0.05:
        shape_diagnosis = "✅ UNIFORM (well-calibrated)"
    else:
        # Check if U-shaped (underdispersed) or ∩-shaped (overdispersed)
        if lower_tail + upper_tail > middle * 0.5:
            shape_diagnosis = "⚠️  U-SHAPED (underdispersed, posteriors too narrow)"
        else:
            shape_diagnosis = "⚠️  ∩-SHAPED (overdispersed, posteriors too wide)"
    
    logger.info(f"   Diagnosis: {shape_diagnosis}")
    
    # Per-parameter KS p-values
    logger.info(f"\n   📍 Per-Parameter KS p-values (11 parameters):")
    param_pvals = {}
    for param_name in param_names:
        pit_param = np.array(pit_by_param[param_name])
        if len(pit_param) > 10:  # Need minimum samples for KS test
            _, pval = kstest(pit_param, 'uniform')
            param_pvals[param_name] = float(pval)
            status = "✅" if pval > 0.05 else "⚠️"
            # Use scientific notation for tiny p-values
            if pval < 1e-6:
                logger.info(f"      {param_name:20s}: p={pval:.2e} {status}")
            else:
                logger.info(f"      {param_name:20s}: p={pval:.6f} {status}")
        else:
            logger.info(f"      {param_name:20s}: (insufficient samples)")
    
    return {
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'anderson_darling': float(anderson_stat),
        'shape_diagnosis': shape_diagnosis,
        'pit_values': pit_values.tolist(),
        'pit_hist_counts': pit_counts.tolist(),
        'per_parameter_pvalues': param_pvals
    }


# ============================================================================
# TEST 9: POSTERIOR PREDICTIVE CHECKS
# ============================================================================

def test_posterior_predictive(model, test_loader, device, n_samples=500):
    """Test 9: Posterior predictive checks (sample quality)"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 9: POSTERIOR PREDICTIVE CHECKS")
    logger.info("=" * 80)
    
    model.eval()
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
                   'theta_jn', 'psi', 'phase', 'geocent_time']
    
    # Sample statistics
    mean_ranges = {name: [] for name in param_names}
    std_ranges = {name: [] for name in param_names}
    
    with torch.no_grad():
        for strain, true_params in tqdm(test_loader, desc="Posterior predictive"):
            strain = strain.to(device)
            batch_size = strain.shape[0]
            
            try:
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                samples = posterior['samples'].cpu().numpy()
                
                for p, name in enumerate(param_names):
                    posterior_p = samples[:, :, p]  # [batch, n_samples]
                    
                    # Per-sample mean and std
                    means = posterior_p.mean(axis=1)
                    stds = posterior_p.std(axis=1)
                    
                    mean_ranges[name].extend(means.tolist())
                    std_ranges[name].extend(stds.tolist())
            
            except Exception as e:
                logger.warning(f"PPC failed: {e}")
                continue
    
    logger.info(f"\n📊 Posterior Ranges (mean ± std):")
    for name in param_names:
        if mean_ranges[name]:
            mean_val = np.mean(mean_ranges[name])
            mean_std = np.std(mean_ranges[name])
            std_val = np.mean(std_ranges[name])
            
            logger.info(f"   {name:20s}: mean={mean_val:8.4f}±{mean_std:6.4f}, width={std_val:8.4f}")
    
    return {
        'mean_ranges': {k: v for k, v in mean_ranges.items()},
        'std_ranges': {k: v for k, v in std_ranges.items()}
    }


# ============================================================================
# TEST 10: CORRELATION & RANKING METRICS
# ============================================================================

def test_downstream_correlation(model, test_loader, device, n_samples=500):
    """Test 10: Correlation between posterior quality and true parameters"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 10: DOWNSTREAM CORRELATION & RANKING")
    logger.info("=" * 80)
    
    model.eval()
    
    posterior_variances = []
    errors = []
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
                   'theta_jn', 'psi', 'phase', 'geocent_time']
    
    with torch.no_grad():
        for strain, true_params in tqdm(test_loader, desc="Correlation"):
            strain = strain.to(device)
            true_params_np = true_params.cpu().numpy()
            
            try:
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                samples = posterior['samples'].cpu().numpy()
                means = posterior['means'].cpu().numpy()
                
                # Per-parameter variance
                post_var = samples.var(axis=1)  # [batch, n_params]
                posterior_variances.append(post_var)
                
                # Absolute error
                error = np.abs(means - true_params_np)
                errors.append(error)
            
            except Exception as e:
                logger.warning(f"Correlation failed: {e}")
                continue
    
    if not posterior_variances or not errors:
        logger.warning("No samples collected for correlation")
        return {}
    
    posterior_variances = np.concatenate(posterior_variances, axis=0)
    errors = np.concatenate(errors, axis=0)
    
    # Calibration check: Correlation between posterior width and prediction error
    # ✓ GOOD: Positive correlation (ρ > 0) means well-calibrated uncertainties
    # Target: ρ ≥ 0.3 (useable), ≥ 0.5 (strong)
    logger.info(f"\n📊 Correlations (Spearman):")
    correlations = {}
    
    for p, name in enumerate(param_names):
        post_var_p = posterior_variances[:, p]
        error_p = errors[:, p]
        
        # Spearman correlation between variance and error
        if len(np.unique(post_var_p)) > 1 and len(np.unique(error_p)) > 1:
            corr, pval = spearmanr(post_var_p, error_p)
            correlations[name] = corr
            
            status = "✅" if corr > 0.3 else "⚠️"
            logger.info(f"   {name:20s}: ρ={corr:6.3f} (p={pval:.3f}) {status}")
        else:
            logger.info(f"   {name:20s}: Insufficient variance")
    
    mean_corr = np.mean([c for c in correlations.values() if not np.isnan(c)])
    logger.info(f"\n   Mean Spearman ρ: {mean_corr:.3f}")
    
    if mean_corr >= 0.5:
        logger.info("   ✅ STRONG correlation (posterior variance predicts error well)")
    elif mean_corr >= 0.3:
        logger.info("   ✅ USEABLE correlation (variance has some predictive power)")
    else:
        logger.warning("   ⚠️  WEAK correlation (posterior variance doesn't predict error)")
    
    return {
        'spearman_correlations': {k: float(v) if not np.isnan(v) else 0.0 
                                  for k, v in correlations.items()},
        'mean_correlation': float(mean_corr)
    }


# ============================================================================
# TEST 11: BIAS CORRECTOR INTEGRATION
# ============================================================================

def test_bias_corrector_integration(model, test_loader, device, n_samples=500):
    """Test 11: BiasCorrector performance on Neural PE outputs"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 11: BIAS CORRECTOR INTEGRATION & PERFORMANCE")
    logger.info("=" * 80)
    
    try:
        from ahsd.core.bias_corrector import BiasCorrector
    except ImportError:
        logger.warning("   ⚠️  BiasCorrector not available, skipping test")
        return {'status': 'skipped', 'reason': 'BiasCorrector not imported'}
    
    # Load BiasCorrector
    bias_corrector_path = Path("models/bias_corrector/bias_corrector_best.pth")
    
    if not bias_corrector_path.exists():
        logger.warning(f"   ⚠️  BiasCorrector checkpoint not found: {bias_corrector_path}")
        return {'status': 'skipped', 'reason': f'Checkpoint not found: {bias_corrector_path}'}
    
    logger.info(f"   Loading BiasCorrector from: {bias_corrector_path}")
    
    try:
        # Define param names (11 parameters: 9 from Neural PE + 2 spins from BiasCorrector training)
        # Note: Neural PE outputs 9 params, but BiasCorrector was trained with 11 params (includes a1, a2)
        # For integration, we need to handle the dimension mismatch gracefully
        param_names_11 = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
                          'theta_jn', 'psi', 'phase', 'geocent_time', 'a1', 'a2']
        
        # Load checkpoint
        checkpoint = torch.load(bias_corrector_path, map_location=device, weights_only=False)
        
        # Detect checkpoint param dimension
        checkpoint_param_dim = None
        if 'bias_estimator.param_embedding.0.weight' in checkpoint:
            checkpoint_param_dim = checkpoint['bias_estimator.param_embedding.0.weight'].shape[1]
            logger.info(f"   Checkpoint trained with {checkpoint_param_dim} parameters")
        
        context_dim = 768  # Neural PE context dimension
        
        # Create BiasCorrector with matching param dimension (11)
        bias_corrector = BiasCorrector(param_names=param_names_11, context_dim=context_dim)
        
        # Load checkpoint
        result = bias_corrector.load_state_dict(checkpoint, strict=False)
        bias_corrector = bias_corrector.to(device).eval()
        
        logger.info(f"✅ BiasCorrector loaded: {len(param_names_11)} parameters, {context_dim}D context")
    
    except Exception as e:
        logger.warning(f"   ⚠️  BiasCorrector loading failed: {str(e)}")
        import traceback
        logger.warning(f"   Traceback: {traceback.format_exc()}")
        return {'status': 'skipped', 'reason': f'BiasCorrector load failed: {str(e)}'}
    
    # Evaluate bias correction on test set
    model.eval()
    bias_corrector.eval()
    
    results_before = []
    results_after = []
    corrections_applied = []
    uncertainties_list = []
    confidence_list = []
    
    # Use only first 9 param names for logging (Neural PE outputs)
    param_names_9 = param_names_11[:9]
    
    with torch.no_grad():
        for strain, true_params in tqdm(test_loader, desc="Bias correction"):
            strain = strain.to(device)
            true_params_np = true_params[:, :9].cpu().numpy()  # Only 9 params
            
            try:
                # Extract parameters from Neural PE
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                means_pe = posterior['means'].cpu().numpy()[:, :9]  # [batch, 9]
                batch_size = means_pe.shape[0]
                
                # Get context from Neural PE
                context = model.context_encoder(strain)  # [batch, 768]
                
                # Compute error BEFORE correction
                error_before = np.abs(means_pe - true_params_np)
                
                # Apply BiasCorrector
                # Pad 9 parameters to 11 (add zeros for a1, a2 spins)
                means_pe_padded = np.zeros((batch_size, 11))
                means_pe_padded[:, :9] = means_pe
                # Spin defaults (0 = non-spinning approximation)
                
                params_normalized = torch.FloatTensor(means_pe_padded).to(device)
                corrections, uncertainties, confidence = bias_corrector(
                    params=params_normalized,
                    context=context
                )
                
                # Extract only first 9 corrections (ignore spin corrections)
                corrections_np = corrections.cpu().numpy()[:, :9]
                uncertainties_np = uncertainties.cpu().numpy()[:, :9]
                confidence_np = confidence.cpu().numpy()
                
                # Apply corrections
                means_corrected = means_pe + corrections_np
                
                # Compute error AFTER correction
                error_after = np.abs(means_corrected - true_params_np)
                
                # Store results
                results_before.append(error_before)
                results_after.append(error_after)
                corrections_applied.append(corrections_np)
                uncertainties_list.append(uncertainties_np)
                confidence_list.append(confidence_np)
                
            except Exception as e:
                logger.warning(f"Sample failed: {e}")
                continue
    
    if not results_before:
        logger.warning("   ⚠️  No samples processed")
        return {'status': 'failed', 'reason': 'No samples processed'}
    
    # Concatenate all results
    error_before = np.concatenate(results_before, axis=0)  # [n_samples, 9]
    error_after = np.concatenate(results_after, axis=0)
    corrections = np.concatenate(corrections_applied, axis=0)
    uncertainties = np.concatenate(uncertainties_list, axis=0)
    confidence = np.concatenate(confidence_list, axis=0)
    
    logger.info(f"\n📊 Bias Correction Results ({len(error_before)} samples):")
    
    # Per-parameter analysis
    logger.info(f"\n   Error Analysis (MAE before vs after):")
    improvements = []
    
    for i, name in enumerate(param_names_9):
        mae_before = error_before[:, i].mean()
        mae_after = error_after[:, i].mean()
        improvement = ((mae_before - mae_after) / mae_before * 100) if mae_before > 0 else 0
        improvements.append(improvement)
        
        status = "✅" if improvement > 0 else "⚠️"
        logger.info(f"   {name:20s}: {mae_before:.6f} → {mae_after:.6f} ({improvement:+.1f}%) {status}")
    
    # Overall metrics
    mae_before_overall = error_before.mean()
    mae_after_overall = error_after.mean()
    improvement_overall = ((mae_before_overall - mae_after_overall) / mae_before_overall * 100) if mae_before_overall > 0 else 0
    
    logger.info(f"\n   📈 Overall Performance:")
    logger.info(f"      MAE Before:     {mae_before_overall:.6f}")
    logger.info(f"      MAE After:      {mae_after_overall:.6f}")
    logger.info(f"      Improvement:    {improvement_overall:+.1f}%")
    logger.info(f"      Correction std: {corrections.std():.6f}")
    logger.info(f"      Uncertainty mean: {uncertainties.mean():.6f}")
    logger.info(f"      Confidence mean: {confidence.mean():.4f}")
    
    # Per-sample statistics
    improvement_per_sample = ((error_before.mean(axis=1) - error_after.mean(axis=1)) / 
                               (error_before.mean(axis=1) + 1e-10)) * 100
    
    improved_samples = (improvement_per_sample > 0).sum()
    degraded_samples = (improvement_per_sample < 0).sum()
    
    logger.info(f"\n   📋 Sample Statistics:")
    logger.info(f"      Improved:  {improved_samples:5d} ({improved_samples/len(improvement_per_sample)*100:5.1f}%)")
    logger.info(f"      Degraded:  {degraded_samples:5d} ({degraded_samples/len(improvement_per_sample)*100:5.1f}%)")
    logger.info(f"      Mean improvement: {improvement_per_sample.mean():+.1f}%")
    logger.info(f"      Median improvement: {np.median(improvement_per_sample):+.1f}%")
    
    # Performance assessment
    if improvement_overall > 10:
        logger.info("   ✅ EXCELLENT: BiasCorrector significantly improves accuracy")
        status = "excellent"
    elif improvement_overall > 0:
        logger.info("   ✅ GOOD: BiasCorrector modestly improves accuracy")
        status = "good"
    elif improvement_overall > -5:
        logger.info("   ⚠️  NEUTRAL: BiasCorrector has minimal impact")
        status = "neutral"
    else:
        logger.warning("   ❌ POOR: BiasCorrector degrades accuracy (may need retraining)")
        status = "poor"
    
    return {
        'status': status,
        'mae_before': float(mae_before_overall),
        'mae_after': float(mae_after_overall),
        'improvement_percent': float(improvement_overall),
        'improved_samples': int(improved_samples),
        'degraded_samples': int(degraded_samples),
        'mean_correction': float(corrections.mean()),
        'std_correction': float(corrections.std()),
        'mean_uncertainty': float(uncertainties.mean()),
        'mean_confidence': float(confidence.mean()),
        'per_param_improvements': {name: float(imp) for name, imp in zip(param_names_9, improvements)}
    }


# ============================================================================
# TEST 12: COMPREHENSIVE DIAGNOSTIC SUMMARY
# ============================================================================

def diagnostic_summary(all_results):
    """Test 12: Comprehensive diagnostic summary with pass/fail thresholds"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 12: COMPREHENSIVE DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)
    
    # Extract key metrics
    metrics = {}
    checks = {}
    
    # NLL check
    try:
        nll_mean = all_results['2_nll']['mean']
        metrics['NLL'] = nll_mean
        if nll_mean < 7.0:
            checks['NLL (< 7.0)'] = '✅ PASS'
        else:
            checks['NLL (< 7.0)'] = '❌ FAIL'
    except:
        checks['NLL'] = '⚠️  SKIP'
    
    # Calibration check
    try:
        cal_error = all_results['3_calibration']['calibration_error']
        metrics['Calibration Error'] = cal_error
        if cal_error < 0.10:
            checks['Calibration Error (< 0.10)'] = '✅ PASS'
        elif cal_error < 0.15:
            checks['Calibration Error (< 0.15)'] = '⚠️  WARN'
        else:
            checks['Calibration Error (< 0.15)'] = '❌ FAIL'
    except:
        checks['Calibration'] = '⚠️  SKIP'
    
    # Parameter recovery check
    try:
        mae = np.mean(all_results['5_recovery']['mean_absolute_errors'])
        metrics['Mean Absolute Error'] = mae
        if mae < 0.1:
            checks['Recovery MAE (< 0.1)'] = '✅ PASS'
        elif mae < 0.5:
            checks['Recovery MAE (< 0.5)'] = '⚠️  WARN'
        else:
            checks['Recovery MAE (< 0.5)'] = '❌ FAIL'
    except:
        checks['Recovery'] = '⚠️  SKIP'
    
    # Performance check
    try:
        inference_time = all_results['7_scaling']['mean_time']
        metrics['Inference Time (s)'] = inference_time
        if inference_time < 1.0:
            checks['Inference (< 1.0s)'] = '✅ PASS'
        elif inference_time < 2.0:
            checks['Inference (< 2.0s)'] = '⚠️  WARN'
        else:
            checks['Inference (< 2.0s)'] = '❌ FAIL'
    except:
        checks['Performance'] = '⚠️  SKIP'
    
    # AUC check
    try:
        mean_auc = all_results['8_classification']['mean_auc']
        metrics['Mean AUC'] = mean_auc
        if mean_auc > 0.80:
            checks['AUC (> 0.80)'] = '✅ PASS'
        elif mean_auc > 0.70:
            checks['AUC (> 0.70)'] = '⚠️  WARN'
        else:
            checks['AUC (> 0.70)'] = '❌ FAIL'
    except:
        checks['Classification'] = '⚠️  SKIP'
    
    # PIT check
    try:
        pit_pval = all_results.get('6_pit', {}).get('ks_pvalue', 0.0)
        metrics['PIT KS p-value'] = pit_pval
        if pit_pval > 0.05:
            checks['PIT Uniformity (p > 0.05)'] = '✅ PASS'
        else:
            checks['PIT Uniformity (p > 0.05)'] = '⚠️  WARN'
    except:
        checks['PIT'] = '⚠️  SKIP'
    
    # Correlation check
    try:
        mean_corr = all_results.get('10_correlation', {}).get('mean_correlation', 0.0)
        metrics['Posterior-Error Correlation'] = mean_corr
        if mean_corr >= 0.5:
            checks['Correlation (ρ ≥ 0.5)'] = '✅ PASS'
        elif mean_corr >= 0.3:
            checks['Correlation (ρ ≥ 0.3)'] = '⚠️  WARN'
        else:
            checks['Correlation (ρ ≥ 0.3)'] = '❌ FAIL'
    except:
        checks['Correlation'] = '⚠️  SKIP'
    
    # Print summary
    logger.info("\n📋 Key Metrics:")
    for metric, value in metrics.items():
        logger.info(f"   {metric:30s}: {value:.4f}")
    
    logger.info("\n🔍 Diagnostic Checks:")
    for check, result in checks.items():
        logger.info(f"   {check:40s}: {result}")
    
    # Count passes/warns/fails
    pass_count = sum(1 for r in checks.values() if '✅' in r)
    warn_count = sum(1 for r in checks.values() if '⚠️' in r)
    fail_count = sum(1 for r in checks.values() if '❌' in r)
    
    logger.info(f"\n📊 Summary: {pass_count} ✅ PASS, {warn_count} ⚠️  WARN, {fail_count} ❌ FAIL")
    
    if fail_count == 0 and pass_count >= 5:
        logger.info("   🚀 PRODUCTION READY: Model is ready for deployment")
    elif fail_count == 0:
        logger.info("   ⚠️  DEVELOPMENT: Continue iterating on model")
    else:
        logger.info("   ❌ NEEDS IMPROVEMENT: Fix failing checks before deployment")
    
    return {
        'metrics': metrics,
        'checks': checks,
        'pass_count': pass_count,
        'warn_count': warn_count,
        'fail_count': fail_count
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Complete Neural PE Testing Suite')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--priority_net_path', type=str, default=None, help='Path to trained PriorityNet model')
    parser.add_argument('--split', type=str, default='test', help='Data split to use')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for testing')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to test')
    parser.add_argument('--n_posterior_samples', type=int, default=500, help='Posterior samples per inference')
    parser.add_argument('--posterior_width_factor', type=float, default=1.0, help='Test-time widening factor (>1 to widen, e.g., 1.3)')
    parser.add_argument('--output', type=str, default='neuralpe_test_results.json', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--fast', action='store_true', help='Fast mode: run only core tests with limited samples')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}\n")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    from ahsd.models.overlap_neuralpe import OverlapNeuralPE
    
    # Load with from_checkpoint if available, fallback to manual loading
    try:
        model = OverlapNeuralPE.from_checkpoint(
            args.model_path, 
            priority_net_path=args.priority_net_path or None
        )
        logger.info("✅ Model loaded with from_checkpoint")
    except AttributeError:
        # Fallback for older checkpoints
        logger.info("Using fallback checkpoint loading...")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        
        # Extract configuration
        config = checkpoint.get('config', {})
        
        # Get param_names from config
        param_names = config.get('param_names', [
            'mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
            'theta_jn', 'psi', 'phase', 'geocent_time', 'a1', 'a2'
        ])
        
        # Get priority_net path from args, training_metadata, or auto-detect
        priority_net_path = args.priority_net_path
        
        # If not provided via args, check training_metadata
        if priority_net_path is None:
            training_metadata = checkpoint.get('training_metadata', {})
            priority_net_path = training_metadata.get('priority_net')
        
        # If still None, try default location
        if priority_net_path is None:
            default_prio_path = Path("models/priority_net/priority_net_best.pth")
            if default_prio_path.exists():
                priority_net_path = str(default_prio_path)
        
        # Create model
        model = OverlapNeuralPE(param_names, priority_net_path, config)
        
        # Load checkpoint weights
        logger.info("Loading checkpoint weights (strict=False)...")
        model.load_state_dict(checkpoint.get('model_state_dict', {}), strict=False)
        logger.info("✅ Fallback checkpoint loading complete")
    
    model.to(device).eval()
    
    # Verify param_bounds exist (set during model __init__ → _get_parameter_bounds())
    if not hasattr(model, 'param_bounds') or model.param_bounds is None:
        raise RuntimeError("❌ CRITICAL: Model missing param_bounds - check OverlapNeuralPE.__init__() for _get_parameter_bounds() call")
    
    logger.info(f"✅ Using model's param_bounds ({len(model.param_bounds)} parameters)")
    # Bounds (from OverlapNeuralPE._get_parameter_bounds):
    #   mass_1/2:              1.0 - 100.0 Msun
    #   luminosity_distance:  10.0 - 8000.0 Mpc
    #   geocent_time:         -2.0 - 8.0 sec (4-signal spacing)
    #   All others: standard physical ranges
    
    logger.info(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Fast mode: override args for limited testing BEFORE loading dataset
    if args.fast:
        logger.info("⚡ FAST MODE: Running core tests with limited samples")
        args.max_samples = args.max_samples or 50  # Default to 50 samples if not specified
        args.n_posterior_samples = min(args.n_posterior_samples, 200)  # Cap at 200 posterior samples
        logger.info(f"   Max samples: {args.max_samples}")
        logger.info(f"   Posterior samples: {args.n_posterior_samples}\n")
    
    # Load dataset (once, with appropriate settings)
    dataset = TestDataset(args.data_path, split=args.split, max_samples=args.max_samples)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f"✅ Test dataset loaded: {len(dataset)} samples\n")
    
    # ✅ Setup test-time posterior widening (Dec 9, 2025)
    width_factor = args.posterior_width_factor
    if width_factor > 1.0:
        logger.info(f"🔧 Test-time posterior widening enabled: T={width_factor:.2f}")
        original_sample_posterior = model.sample_posterior
        param_names_all = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
                           'theta_jn', 'psi', 'phase', 'geocent_time', 'a1', 'a2']
        
        def sample_posterior_widened(strain, n_samples=500):
            posterior = original_sample_posterior(strain, n_samples=n_samples)
            samples = posterior['samples']  # [batch, n_samples, param_dim] in PHYSICAL units
            
            # Widen: samples = mean + T * (samples - mean)
            means = samples.mean(dim=1, keepdim=True)  # [batch, 1, param_dim]
            samples_widened = means + width_factor * (samples - means)
            
            # ✅ FIXED (Dec 9): Use model's actual parameter bounds
            # Samples are in physical units, need to use model's param_bounds
            for i, param_name in enumerate(param_names_all):
                if i < samples_widened.shape[-1]:
                    bounds = model.param_bounds.get(param_name, (-np.inf, np.inf))
                    samples_widened[..., i] = torch.clamp(samples_widened[..., i], bounds[0], bounds[1])
            
            # Update posterior with widened samples
            posterior['samples'] = samples_widened
            posterior['stds'] = samples_widened.std(dim=1)  # Recompute stds
            
            return posterior
        
        model.sample_posterior = sample_posterior_widened
        logger.info(f"✅ Posterior widening active during tests\n")
    
    # Run all tests
    logger.info("🚀 Starting comprehensive testing...\n")
     
    all_results = {}
    
    # # Core tests (always run)
    # all_results['1_sanity'] = test_posterior_sanity(model, test_loader, device, args.n_posterior_samples)
    # all_results['2_nll'] = test_nll(model, test_loader, device, args.n_posterior_samples)
    all_results['3_calibration'] = test_calibration(model, test_loader, device, args.n_posterior_samples)
    all_results['6_pit'] = test_pit_histogram(model, test_loader, device, args.n_posterior_samples)
    
    if not args.fast:
        # Extended tests (skip in fast mode)
        logger.info("\n📊 Running extended tests...\n")
        all_results['4_width'] = test_posterior_width(model, test_loader, device, args.n_posterior_samples)
        all_results['5_recovery'] = test_parameter_recovery(model, test_loader, device, args.n_posterior_samples)
        all_results['7_scaling'] = test_scaling(model, test_loader, device, args.n_posterior_samples)
        all_results['8_classification'] = test_classification_metrics(model, test_loader, device, args.n_posterior_samples)
        all_results['9_ppc'] = test_posterior_predictive(model, test_loader, device, args.n_posterior_samples)
        all_results['10_correlation'] = test_downstream_correlation(model, test_loader, device, args.n_posterior_samples)
        all_results['11_bias_corrector'] = test_bias_corrector_integration(model, test_loader, device, args.n_posterior_samples)
    
    # Run diagnostic summary
    all_results['12_summary'] = diagnostic_summary(all_results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    serializable_results = make_serializable(all_results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL TEST SUMMARY")
    logger.info("=" * 80)
    
    try:
        nll_result = all_results['2_nll']
        logger.info(f"NLL (primary metric):      {nll_result['mean']:.4f} ± {nll_result['std']:.4f}")
    except:
        pass
    
    try:
        cal_result = all_results['3_calibration']
        logger.info(f"Calibration error:         {cal_result['calibration_error']:.4f}")
    except:
        pass
    
    try:
        perf_result = all_results['7_scaling']
        logger.info(f"Inference time (per sample): {perf_result['mean_time']:.4f}s")
    except:
        pass
    
    try:
        auc_result = all_results['8_classification']
        logger.info(f"Mean AUC:                  {auc_result['mean_auc']:.4f}")
    except:
        pass
    
    try:
        bias_result = all_results['11_bias_corrector']
        if bias_result.get('status') != 'skipped':
            logger.info(f"\nBias Corrector Performance:")
            logger.info(f"  Status:                    {bias_result['status'].upper()}")
            logger.info(f"  MAE before correction:     {bias_result['mae_before']:.6f}")
            logger.info(f"  MAE after correction:      {bias_result['mae_after']:.6f}")
            logger.info(f"  Improvement:               {bias_result['improvement_percent']:+.1f}%")
            logger.info(f"  Samples improved:          {bias_result['improved_samples']}")
    except:
        pass
    
    logger.info(f"\n✅ Complete! Results saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()


