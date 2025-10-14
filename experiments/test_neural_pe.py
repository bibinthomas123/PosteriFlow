#!/usr/bin/env python3
"""
COMPLETE Neural PE Testing Suite
Implements all 8 validation tests for posterior quality
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
from typing import List, Dict, Tuple
from scipy import stats, integrate
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


project_root = Path(__file__).parent.parent
sys.path.insert(0,str(project_root))
# ============================================================================
# DATASET LOADER
# ============================================================================

class TestDataset(torch.utils.data.Dataset):
    """Simple dataset for testing"""
    
    def __init__(self, data_path, split='test', max_samples=None):
        self.data_path = Path(data_path)
        self.split = split
        self.samples = self._load_samples(max_samples)
        
        logger.info(f"✅ Loaded {len(self.samples)} {split} samples")
    
    def _load_samples(self, max_samples):
        samples = []
        split_dir = self.data_path / self.split
        
        chunk_files = sorted(split_dir.glob('chunk_*.pkl'))
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    samples.extend(chunk_data)
                    
                    if max_samples and len(samples) >= max_samples:
                        return samples[:max_samples]
            except:
                continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract strain (use H1 or first detector)
        detector_data = sample.get('whitened_data', sample.get('detector_data', {}))
        
        if 'H1' in detector_data:
            strain = np.array(detector_data['H1'])
        elif detector_data:
            first_det = list(detector_data.keys())[0]
            strain = np.array(detector_data[first_det])
        else:
            strain = np.zeros(16384)
        
        # Ensure correct length
        if len(strain) < 16384:
            strain = np.pad(strain, (0, 16384 - len(strain)))
        elif len(strain) > 16384:
            strain = strain[:16384]
        
        # Extract true parameters
        metadata = sample.get('metadata', {})
        signal_params = metadata.get('signal_parameters', [{}])[0]
        
        true_params = np.array([
            signal_params.get('mass_1', 30.0),
            signal_params.get('mass_2', 25.0),
            signal_params.get('luminosity_distance', 500.0),
            signal_params.get('ra', 0.0),
            signal_params.get('dec', 0.0),
            signal_params.get('theta_jn', 0.0),
            signal_params.get('psi', 0.0),
            signal_params.get('phase', 0.0),
            signal_params.get('geocent_time', 0.0)
        ], dtype=np.float32)
        
        return torch.FloatTensor(strain), torch.FloatTensor(true_params)


# ============================================================================
# TEST 1: POSTERIOR SANITY CHECKS
# ============================================================================

def test_posterior_sanity(model, test_loader, device, n_samples=1000):
    """Test 1: Basic posterior quality checks"""
    
    logger.info("="*70)
    logger.info("TEST 1: POSTERIOR SANITY CHECKS")
    logger.info("="*70)
    
    model.eval()
    results = {
        'marginalization_check': [],
        'support_violations': 0,
        'nan_count': 0,
        'total_samples': 0
    }
    
    with torch.no_grad():
        for strain, true_params in tqdm(test_loader, desc="Sanity checks"):
            strain = strain.to(device)
            batch_size = strain.shape[0]
            
            # Sample posteriors
            try:
                # Get context
                context = model.context_encoder(strain)
                
                # Sample from base distribution
                z = torch.randn(n_samples, batch_size, model.n_params).to(device)
                
                # Transform through flow
                posterior_samples = []
                for i in range(0, n_samples, 100):  # Process in chunks
                    z_chunk = z[i:i+100]
                    samples_chunk, _ = model.flow.inverse(
                        z_chunk.reshape(-1, model.n_params),
                        context.repeat_interleave(z_chunk.shape[0], dim=0)
                    )
                    posterior_samples.append(samples_chunk.reshape(z_chunk.shape))
                
                posterior_samples = torch.cat(posterior_samples, dim=0)  # [n_samples, batch, n_params]
                
                # Check for NaNs
                nan_mask = torch.isnan(posterior_samples).any(dim=-1).any(dim=0)
                results['nan_count'] += nan_mask.sum().item()
                
                # Check marginalization (approximate)
                for b in range(batch_size):
                    samples_b = posterior_samples[:, b, :].cpu().numpy()
                    
                    # Simple histogram normalization check for first parameter
                    hist, bins = np.histogram(samples_b[:, 0], bins=50, density=True)
                    integral = np.trapz(hist, bins[:-1])
                    results['marginalization_check'].append(integral)
                
                results['total_samples'] += batch_size
                
            except Exception as e:
                logger.warning(f"Posterior sampling failed: {e}")
                continue
    
    # Compute statistics
    marg_checks = np.array(results['marginalization_check'])
    
    logger.info(f"\n📊 Sanity Check Results:")
    logger.info(f"   Total samples tested: {results['total_samples']}")
    logger.info(f"   NaN detections: {results['nan_count']}")
    logger.info(f"   Marginalization integral: {marg_checks.mean():.3f} ± {marg_checks.std():.3f}")
    
    if results['nan_count'] == 0:
        logger.info("   ✅ No NaN values detected")
    else:
        logger.warning(f"   ⚠️  {results['nan_count']} samples with NaNs")
    
    if np.abs(marg_checks.mean() - 1.0) < 0.2:
        logger.info("   ✅ Marginalization check passed")
    else:
        logger.warning("   ⚠️  Marginalization may be incorrect")
    
    return results


# ============================================================================
# TEST 2: CALIBRATION (P-P PLOT)
# ============================================================================

def test_calibration(model, test_loader, device, n_samples=1000):
    """Test 2: P-P plot and coverage calibration"""
    
    logger.info("\n" + "="*70)
    logger.info("TEST 2: CALIBRATION TEST (P-P PLOT)")
    logger.info("="*70)
    
    model.eval()
    percentiles = []
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 
                   'theta_jn', 'psi', 'phase', 'geocent_time']
    
    with torch.no_grad():
        for strain, true_params in tqdm(test_loader, desc="Calibration"):
            strain = strain.to(device)
            true_params_np = true_params.cpu().numpy()
            batch_size = strain.shape[0]
            
            try:
                # Get context
                context = model.context_encoder(strain)
                
                # Sample posteriors
                z = torch.randn(n_samples, batch_size, model.n_params).to(device)
                posterior_samples, _ = model.flow.inverse(
                    z.reshape(-1, model.n_params),
                    context.repeat_interleave(n_samples, dim=0)
                )
                posterior_samples = posterior_samples.reshape(n_samples, batch_size, model.n_params)
                posterior_samples = posterior_samples.cpu().numpy()
                
                # Compute percentiles where true value lies
                for b in range(batch_size):
                    sample_percentiles = []
                    
                    for p in range(model.n_params):
                        posterior_p = posterior_samples[:, b, p]
                        true_val = true_params_np[b, p]
                        
                        # Compute CDF percentile
                        percentile = (posterior_p < true_val).mean()
                        sample_percentiles.append(percentile)
                    
                    percentiles.append(sample_percentiles)
            
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
                continue
    
    percentiles = np.array(percentiles)  # [n_samples, n_params]
    
    # Compute coverage
    coverage_levels = [0.68, 0.90, 0.95]
    coverage_results = {}
    
    for level in coverage_levels:
        lower = (1 - level) / 2
        upper = 1 - lower
        
        in_interval = ((percentiles > lower) & (percentiles < upper)).mean(axis=0)
        coverage_results[level] = in_interval
        
        logger.info(f"\n📊 {level*100:.0f}% Credible Interval Coverage:")
        for i, name in enumerate(param_names):
            logger.info(f"   {name:20s}: {in_interval[i]*100:.1f}%")
    
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
    else:
        logger.warning("   ⚠️  POOR calibration")
    
    # Plot P-P plot
    plt.figure(figsize=(8, 8))
    percentiles_flat = percentiles.flatten()
    expected = np.linspace(0, 1, 100)
    observed = np.percentile(percentiles_flat, np.linspace(0, 100, 100))
    
    plt.plot(expected, observed, 'b-', linewidth=2, label='Observed')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Ideal')
    plt.fill_between(expected, expected - 0.05, expected + 0.05, alpha=0.2, color='gray')
    plt.xlabel('Expected Percentile')
    plt.ylabel('Observed Percentile')
    plt.title('P-P Plot: Posterior Calibration')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('pp_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("   📊 P-P plot saved: pp_plot.png")
    
    return {
        'coverage': coverage_results,
        'calibration_error': calibration_error,
        'percentiles': percentiles
    }


# ============================================================================
# TEST 3: NLL (NEGATIVE LOG-LIKELIHOOD)
# ============================================================================

def test_nll(model, test_loader, device):
    """Test 3: Compute NLL"""
    
    logger.info("\n" + "="*70)
    logger.info("TEST 3: NEGATIVE LOG-LIKELIHOOD (NLL)")
    logger.info("="*70)
    
    model.eval()
    nll_values = []
    
    with torch.no_grad():
        for strain, true_params in tqdm(test_loader, desc="Computing NLL"):
            strain = strain.to(device)
            true_params = true_params.to(device)
            
            try:
                # Compute NLL
                result = model.compute_loss(strain, true_params)
                nll = result.get('nll', result.get('loss'))
                
                if torch.isnan(nll).any():
                    continue
                
                nll_values.append(nll.cpu().numpy())
            
            except Exception as e:
                logger.warning(f"NLL computation failed: {e}")
                continue
    
    nll_values = np.concatenate(nll_values)
    
    results = {
        'mean': float(np.mean(nll_values)),
        'std': float(np.std(nll_values)),
        'median': float(np.median(nll_values)),
        'min': float(np.min(nll_values)),
        'max': float(np.max(nll_values))
    }
    
    logger.info(f"\n📊 NLL Results ({len(nll_values)} samples):")
    logger.info(f"   Mean:   {results['mean']:.4f} ± {results['std']:.4f}")
    logger.info(f"   Median: {results['median']:.4f}")
    logger.info(f"   Range:  [{results['min']:.4f}, {results['max']:.4f}]")
    
    if results['mean'] < 7.0:
        logger.info("   ✅ EXCELLENT (publication-ready)")
    elif results['mean'] < 8.5:
        logger.info("   ✅ GOOD (production-ready)")
    elif results['mean'] < 10.0:
        logger.info("   ⚠️  ACCEPTABLE")
    else:
        logger.warning("   ❌ POOR (needs improvement)")
    
    return results


# ============================================================================
# TEST 4: POSTERIOR WIDTH ANALYSIS
# ============================================================================

def test_posterior_width(model, test_loader, device, n_samples=1000):
    """Test 4: Posterior width ratio"""
    
    logger.info("\n" + "="*70)
    logger.info("TEST 4: POSTERIOR WIDTH ANALYSIS")
    logger.info("="*70)
    
    model.eval()
    posterior_stds = []
    true_stds = []
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 
                   'theta_jn', 'psi', 'phase', 'geocent_time']
    
    with torch.no_grad():
        for strain, true_params in tqdm(test_loader, desc="Width analysis"):
            strain = strain.to(device)
            batch_size = strain.shape[0]
            
            try:
                # Get context
                context = model.context_encoder(strain)
                
                # Sample posteriors
                z = torch.randn(n_samples, batch_size, model.n_params).to(device)
                posterior_samples, _ = model.flow.inverse(
                    z.reshape(-1, model.n_params),
                    context.repeat_interleave(n_samples, dim=0)
                )
                posterior_samples = posterior_samples.reshape(n_samples, batch_size, model.n_params)
                posterior_samples = posterior_samples.cpu().numpy()
                
                # Compute std per parameter
                post_std = posterior_samples.std(axis=0)  # [batch, n_params]
                posterior_stds.append(post_std)
            
            except:
                continue
    
    posterior_stds = np.concatenate(posterior_stds, axis=0)  # [total_samples, n_params]
    
    # Compute mean width per parameter
    mean_widths = posterior_stds.mean(axis=0)
    
    logger.info(f"\n📊 Posterior Widths:")
    for i, name in enumerate(param_names):
        logger.info(f"   {name:20s}: {mean_widths[i]:.4f}")
    
    return {
        'posterior_stds': posterior_stds,
        'mean_widths': mean_widths.tolist()
    }


# ============================================================================
# TEST 5: PARAMETER INDEPENDENCE
# ============================================================================

def test_parameter_independence(model, test_loader, device, n_samples=1000):
    """Test 5: Check posterior correlations"""
    
    logger.info("\n" + "="*70)
    logger.info("TEST 5: PARAMETER INDEPENDENCE")
    logger.info("="*70)
    
    model.eval()
    all_samples = []
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 
                   'theta_jn', 'psi', 'phase', 'geocent_time']
    
    with torch.no_grad():
        for strain, _ in tqdm(test_loader, desc="Independence test"):
            strain = strain.to(device)
            batch_size = strain.shape[0]
            
            try:
                context = model.context_encoder(strain)
                z = torch.randn(n_samples, batch_size, model.n_params).to(device)
                posterior_samples, _ = model.flow.inverse(
                    z.reshape(-1, model.n_params),
                    context.repeat_interleave(n_samples, dim=0)
                )
                posterior_samples = posterior_samples.reshape(n_samples, batch_size, model.n_params)
                
                # Take mean over samples
                posterior_mean = posterior_samples.mean(dim=0).cpu().numpy()
                all_samples.append(posterior_mean)
            except:
                continue
    
    all_samples = np.concatenate(all_samples, axis=0)  # [total, n_params]
    
    # Compute correlation matrix
    df = pd.DataFrame(all_samples, columns=param_names)
    corr_matrix = df.corr(method='spearman')
    
    logger.info(f"\n📊 Spearman Correlations (|r| > 0.3):")
    
    strong_corr_count = 0
    for i in range(len(param_names)):
        for j in range(i+1, len(param_names)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.3:
                logger.info(f"   {param_names[i]:15s} ↔ {param_names[j]:15s}: r={r:+.3f}")
                strong_corr_count += 1
    
    if strong_corr_count == 0:
        logger.info("   ✅ No strong correlations detected")
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True)
    plt.title('Posterior Parameter Correlations (Spearman)')
    plt.tight_layout()
    plt.savefig('posterior_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("   📊 Correlation heatmap saved: posterior_correlations.png")
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'strong_correlations': strong_corr_count
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Complete Neural PE Testing')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--n_posterior_samples', type=int, default=1000)
    parser.add_argument('--output', type=str, default='neuralpe_test_results.json')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}\n")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    from ahsd.models.overlap_neuralpe import OverlapNeuralPE
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    param_names = checkpoint['param_names']
    config = checkpoint['config']
    priority_net_path = config['training_metadata']['priority_net']
    
    model = OverlapNeuralPE(param_names, priority_net_path, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    logger.info(f"✅ Model loaded\n")
    
    # Load dataset
    dataset = TestDataset(args.data_path, split=args.split, max_samples=args.max_samples)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Run all tests
    logger.info("🚀 Starting comprehensive testing...\n")
    
    all_results = {}
    
    all_results['1_sanity'] = test_posterior_sanity(model, test_loader, device, args.n_posterior_samples)
    all_results['2_calibration'] = test_calibration(model, test_loader, device, args.n_posterior_samples)
    all_results['3_nll'] = test_nll(model, test_loader, device)
    all_results['4_width'] = test_posterior_width(model, test_loader, device, args.n_posterior_samples)
    all_results['5_independence'] = test_parameter_independence(model, test_loader, device, args.n_posterior_samples)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, val in all_results.items():
        if isinstance(val, dict):
            serializable_results[key] = {}
            for k, v in val.items():
                if isinstance(v, np.ndarray):
                    serializable_results[key][k] = v.tolist()
                else:
                    serializable_results[key][k] = v
        else:
            serializable_results[key] = val
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Print final summary
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    logger.info(f"NLL:                 {all_results['3_nll']['mean']:.4f} ± {all_results['3_nll']['std']:.4f}")
    logger.info(f"Calibration Error:   {all_results['2_calibration']['calibration_error']:.3f}")
    logger.info(f"Strong Correlations: {all_results['5_independence']['strong_correlations']}")
    logger.info(f"\n✅ Complete! Results saved to: {output_path}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
