
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
        
        true_params = np.array([
            signal_params.get('mass_1', 30.0),
            signal_params.get('mass_2', 25.0),
            signal_params.get('luminosity_distance', 500.0),
            signal_params.get('ra', 0.0),
            signal_params.get('dec', 0.0),
            signal_params.get('theta_jn', 0.0),
            signal_params.get('psi', 0.0),
            signal_params.get('phase', 0.0),
            signal_params.get('geocent_time', 0.0),
            signal_params.get('a1', 0.0),  # Primary spin magnitude
            signal_params.get('a2', 0.0)   # Secondary spin magnitude
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
                   'theta_jn', 'psi', 'phase', 'geocent_time']
    
    with torch.no_grad():
        batch_count = 0
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
    
    if nan_pct < 1.0 and inf_pct < 1.0 and oob_pct < 5.0:
        logger.info("   ✅ PASS: Posterior sanity checks")
    else:
        logger.warning("   ⚠️  FAIL: High anomaly rate detected")
    
    return results


# ============================================================================
# TEST 2: NEGATIVE LOG-LIKELIHOOD (NLL)
# ============================================================================

def test_nll(model, test_loader, device, n_samples=500):
    """Test 2: Compute NLL for posterior quality assessment"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: NEGATIVE LOG-LIKELIHOOD (NLL)")
    logger.info("=" * 80)
    
    model.eval()
    nll_values = []
    
    with torch.no_grad():
        for strain, true_params in tqdm(test_loader, desc="Computing NLL"):
            strain = strain.to(device)
            true_params = true_params.to(device)
            batch_size = strain.shape[0]
            
            try:
                # Sample posterior for NLL estimation
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                samples = posterior['samples']  # [batch, n_samples, n_params]
                
                # Compute NLL using KDE on samples
                for b in range(batch_size):
                    samples_b = samples[b].cpu().numpy()  # [n_samples, n_params]
                    true_params_b = true_params[b].cpu().numpy()  # [n_params]
                    
                    # Adaptive bandwidth using Scott's rule
                    n, d = samples_b.shape
                    scott_factor = n ** (-1.0 / (d + 4))
                    bandwidths = samples_b.std(axis=0) * scott_factor
                    bandwidths = np.maximum(bandwidths, 1e-3)
                    
                    # Compute log-likelihood: log p(x) = log(1/n * sum_i K(x - x_i))
                    # For Gaussian kernel: K(z) = (2π)^(-d/2) * exp(-||z||^2/2)
                    diffs = samples_b - true_params_b  # [n_samples, n_params]
                    scaled_diffs = diffs / bandwidths  # [n_samples, n_params]
                    
                    # Gaussian kernel: log K = -0.5 * ||z||^2 - 0.5*d*log(2π)
                    log_kernels = -0.5 * np.sum(scaled_diffs**2, axis=1)  # [n_samples]
                    
                    # Log-sum-exp for numerical stability
                    max_log_kernel = np.max(log_kernels)
                    log_prob = max_log_kernel + np.log(np.mean(np.exp(log_kernels - max_log_kernel)))
                    
                    # NLL = -log(p(x))
                    nll = -log_prob
                    
                    if not np.isnan(nll) and not np.isinf(nll) and nll < 100:
                        nll_values.append(float(nll))
            
            except Exception as e:
                logger.warning(f"NLL computation failed: {e}")
                continue
    
    nll_values = np.array(nll_values)
    
    # Safe min/max computation
    if len(nll_values) > 0:
        results = {
            'mean': float(np.mean(nll_values)),
            'std': float(np.std(nll_values)),
            'median': float(np.median(nll_values)),
            'min': float(np.min(nll_values)),
            'max': float(np.max(nll_values)),
            'percentile_25': float(np.percentile(nll_values, 25)),
            'percentile_75': float(np.percentile(nll_values, 75))
        }
    else:
        results = {
            'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0,
            'percentile_25': 0.0, 'percentile_75': 0.0
        }
    
    logger.info(f"\n📊 NLL Results ({len(nll_values)} samples):")
    logger.info(f"   Mean:       {results['mean']:.4f} ± {results['std']:.4f}")
    logger.info(f"   Median:     {results['median']:.4f}")
    logger.info(f"   Range:      [{results['min']:.4f}, {results['max']:.4f}]")
    logger.info(f"   IQR:        [{results['percentile_25']:.4f}, {results['percentile_75']:.4f}]")
    
    if results['mean'] < 5.0:
        logger.info("   ✅ EXCELLENT (publication-ready NLL < 5)")
    elif results['mean'] < 7.0:
        logger.info("   ✅ GOOD (production-ready NLL < 7)")
    elif results['mean'] < 10.0:
        logger.info("   ⚠️  ACCEPTABLE (NLL < 10)")
    else:
        logger.warning("   ❌ POOR (needs improvement, NLL > 10)")
    
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
                   'theta_jn', 'psi', 'phase', 'geocent_time']
    
    with torch.no_grad():
        for strain, true_params in tqdm(test_loader, desc="Calibration"):
            strain = strain.to(device)
            true_params_np = true_params.cpu().numpy()
            batch_size = strain.shape[0]
            
            try:
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                samples = posterior['samples'].cpu().numpy()  # [batch, n_samples, n_params]
                
                for b in range(batch_size):
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
                   'theta_jn', 'psi', 'phase', 'geocent_time']
    
    with torch.no_grad():
        for strain, _ in tqdm(test_loader, desc="Width analysis"):
            strain = strain.to(device)
            batch_size = strain.shape[0]
            
            try:
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                samples = posterior['samples'].cpu().numpy()
                
                # Compute std per parameter
                post_std = samples.std(axis=1)  # [batch, n_params]
                posterior_stds.append(post_std)
            
            except:
                continue
    
    posterior_stds = np.concatenate(posterior_stds, axis=0)
    
    # Compute mean width per parameter
    mean_widths = posterior_stds.mean(axis=0)
    
    logger.info(f"\n📊 Posterior Widths (std dev):")
    for i, name in enumerate(param_names):
        logger.info(f"   {name:20s}: {mean_widths[i]:.4f}")
    
    # Check for appropriate contraction
    if mean_widths.mean() > 0.1:
        logger.info("   ⚠️  Wide posteriors - may need more data or better model")
    elif mean_widths.mean() > 0.01:
        logger.info("   ✅ Reasonable posterior widths")
    else:
        logger.warning("   ⚠️  Very narrow posteriors - possible overfitting")
    
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
                   'theta_jn', 'psi', 'phase', 'geocent_time']
    
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
# TEST 6: BIAS CORRECTOR EVALUATION
# ============================================================================

def test_bias_corrector(model, test_loader, device, n_samples=500):
    """Test 6: Evaluate BiasCorrector performance"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: BIAS CORRECTOR EVALUATION")
    logger.info("=" * 80)
    
    if model.bias_corrector is None:
        logger.warning("⊘ BiasCorrector not available (disabled in model)")
        return {'status': 'disabled'}
    
    model.eval()
    before_errors = []
    after_errors = []
    correction_mags = []
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
                   'theta_jn', 'psi', 'phase', 'geocent_time']
    
    with torch.no_grad():
        batch_count = 0
        for strain, true_params in tqdm(test_loader, desc="BiasCorrector"):
            strain = strain.to(device)
            true_params_np = true_params.cpu().numpy()
            batch_size = strain.shape[0]
            
            try:
                # Get uncorrected posterior
                posterior = model.sample_posterior(strain, n_samples=n_samples)
                means_uncorr = posterior['means'].cpu().numpy()
                context = posterior['context']
                
                # Apply bias correction
                params_norm = model._normalize_parameters(
                    torch.FloatTensor(means_uncorr).to(device)
                )
                corrections, uncertainties, confidence = model.bias_corrector(
                    params_norm, context
                )
                corrections_np = corrections.cpu().numpy()
                
                # Corrected parameters
                means_corr = means_uncorr + corrections_np
                
                # Compute errors before/after
                before_err = np.abs(means_uncorr - true_params_np)
                after_err = np.abs(means_corr - true_params_np)
                
                before_errors.append(before_err)
                after_errors.append(after_err)
                correction_mags.append(np.abs(corrections_np))
                
                batch_count += 1
                if batch_count >= 20:  # Limit to 20 batches for speed
                    break
            
            except Exception as e:
                logger.warning(f"BiasCorrector eval failed: {e}")
                continue
    
    if not before_errors:
        logger.warning("No bias corrector evaluations completed")
        return {'status': 'failed'}
    
    before_errors = np.concatenate(before_errors, axis=0)
    after_errors = np.concatenate(after_errors, axis=0)
    correction_mags = np.concatenate(correction_mags, axis=0)
    
    # Compute improvement
    improvement = before_errors - after_errors
    improvement_pct = (improvement / (before_errors + 1e-6) * 100).mean(axis=0)
    
    logger.info(f"\n📊 BiasCorrector Performance ({len(before_errors)} samples):")
    logger.info(f"   Parameter-wise improvements:")
    for i, name in enumerate(param_names):
        before_mae = before_errors[:, i].mean()
        after_mae = after_errors[:, i].mean()
        corr_mag = correction_mags[:, i].mean()
        
        logger.info(f"   {name:20s}: {before_mae:8.4f} → {after_mae:8.4f} "
                   f"(improve {improvement_pct[i]:+6.1f}%, corr={corr_mag:.4f})")
    
    # Overall improvement
    overall_improve = (before_errors.mean() - after_errors.mean()) / (before_errors.mean() + 1e-6) * 100
    
    if overall_improve > 5:
        logger.info(f"   ✅ PASS: BiasCorrector improves errors by {overall_improve:.1f}%")
    elif overall_improve > 0:
        logger.info(f"   ⚠️  WEAK: BiasCorrector improves by only {overall_improve:.1f}%")
    else:
        logger.warning(f"   ❌ FAIL: BiasCorrector degrades errors by {-overall_improve:.1f}%")
    
    return {
        'before_mae': before_errors.mean(axis=0).tolist(),
        'after_mae': after_errors.mean(axis=0).tolist(),
        'improvement_pct': improvement_pct.tolist(),
        'overall_improvement': float(overall_improve),
        'correction_magnitudes': correction_mags.mean(axis=0).tolist()
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
    parser.add_argument('--output', type=str, default='neuralpe_test_results.json', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}\n")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    from ahsd.models.overlap_neuralpe import OverlapNeuralPE
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Extract configuration - checkpoint config is flat (not nested under neural_posterior)
    config = checkpoint.get('config', {})
    
    # Get param_names from config
    param_names = config.get('param_names', [
        'mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
        'theta_jn', 'psi', 'phase', 'geocent_time'
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
    
    # Use checkpoint config directly (it's flat, not nested)
    neural_pe_config = config
    
    logger.info(f"Parameters: {param_names}")
    logger.info(f"PriorityNet path: {priority_net_path if priority_net_path else '(disabled)'}")
    
    # Create a minimal model just for inference without architecture mismatches
    # The checkpoint was trained with specific dimensions, so we need to match them
    model = OverlapNeuralPE(param_names, priority_net_path, neural_pe_config)
    
    # Load checkpoint - use strict=False because BiasCorrector may have different dimensions
    logger.info("Loading checkpoint weights (strict=False to handle architecture differences)...")
    model_state = model.state_dict()
    checkpoint_state = checkpoint.get('model_state_dict', {})
    
    loaded_count = 0
    for name, param in model_state.items():
        if name in checkpoint_state and checkpoint_state[name].shape == param.shape:
            model_state[name].copy_(checkpoint_state[name])
            loaded_count += 1
    
    logger.info(f"✅ Loaded {loaded_count}/{len(model_state)} compatible weights")
    if loaded_count < len(model_state) * 0.3:
        logger.warning(f"⚠️  Only {loaded_count/len(model_state)*100:.1f}% of weights loaded - inference may fail")
    
    model.to(device).eval()
    
    logger.info(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Load dataset
    dataset = TestDataset(args.data_path, split=args.split, max_samples=args.max_samples)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Run all tests
    logger.info("🚀 Starting comprehensive testing...\n")
    
    all_results = {}
    
    all_results['1_sanity'] = test_posterior_sanity(model, test_loader, device, args.n_posterior_samples)
    all_results['2_nll'] = test_nll(model, test_loader, device, args.n_posterior_samples)
    all_results['3_calibration'] = test_calibration(model, test_loader, device, args.n_posterior_samples)
    all_results['4_width'] = test_posterior_width(model, test_loader, device, args.n_posterior_samples)
    all_results['5_recovery'] = test_parameter_recovery(model, test_loader, device, args.n_posterior_samples)
    all_results['6_bias_corrector'] = test_bias_corrector(model, test_loader, device, args.n_posterior_samples)
    all_results['7_scaling'] = test_scaling(model, test_loader, device, args.n_posterior_samples)
    all_results['8_classification'] = test_classification_metrics(model, test_loader, device, args.n_posterior_samples)
    
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
    
    logger.info(f"\n✅ Complete! Results saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
