#!/usr/bin/env python3
"""
Proper Simulation-Based Calibration (SBC) for Distance Parameter.

Gold standard test:
1. Rank histogram for log(D_L) - should be uniform
2. Coverage vs SNR - should be flat at 68%
3. Median bias vs true distance - should be centered at zero

This is the ONLY way to verify if posterior is actually calibrated.
"""

import numpy as np
import torch
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistanceSBC:
    """SBC for luminosity distance parameter."""
    
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained Neural PE model."""
        try:
            import yaml
            from ahsd.models.overlap_neuralpe import OverlapNeuralPE
            
            logger.info(f"Loading model from {model_path}...")
            
            # Load config
            config_path = Path("configs/enhanced_training.yaml")
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model with required arguments (must match training config)
            param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time', 'a1', 'a2']
            
            self.model = OverlapNeuralPE(
                param_names=param_names,
                priority_net_path="models/prioritynet/priority_net_best.pth",
                config=config,
                device=self.device
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model.eval()
            
            logger.info(f"✅ Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_validation_data(self, data_path, max_samples=None):
        """Load validation dataset."""
        data_path = Path(data_path)
        samples = []
        
        chunk_files = sorted(data_path.glob("chunk_*.pkl"))
        for chunk_file in chunk_files:
            if max_samples and len(samples) >= max_samples:
                break
            with open(chunk_file, "rb") as f:
                chunk_data = pickle.load(f)
                samples.extend(chunk_data[:max_samples - len(samples) if max_samples else len(chunk_data)])
        
        logger.info(f"✅ Loaded {len(samples)} samples")
        return samples
    
    def extract_distance_data(self, samples):
        """Extract distance, SNR, and event type from samples."""
        distances_true = []
        snrs_true = []
        event_types = []
        
        for sample in samples:
            if isinstance(sample, dict):
                params_list = sample.get("parameters", [])
                if isinstance(params_list, list) and len(params_list) > 0:
                    params = params_list[0]
                    if isinstance(params, dict):
                        dist = params.get("luminosity_distance", np.nan)
                        snr = params.get("target_snr", np.nan)
                        event_type = sample.get("event_type", "unknown")
                        
                        if not np.isnan(dist) and not np.isnan(snr) and dist > 0 and snr > 0:
                            distances_true.append(float(dist))
                            snrs_true.append(float(snr))
                            event_types.append(str(event_type))
        
        return np.array(distances_true), np.array(snrs_true), event_types
    
    def generate_posterior_samples(self, samples, n_posterior=100):
        """Generate posterior samples for validation data using trained model."""
        if self.model is None:
            logger.error("❌ Model not loaded")
            return None
        
        distance_samples_all = []
        
        logger.info(f"Generating {n_posterior} posterior samples for {len(samples)} data points...")
        
        with torch.no_grad():
            for idx, sample in enumerate(samples):
                if idx % 50 == 0:
                    logger.info(f"  Processing {idx}/{len(samples)}...")
                
                try:
                    # Extract strain data
                    detector_data = sample.get("detector_data", {})
                    strain_list = []
                    
                    for detector in ["H1", "L1", "V1"]:
                        if detector in detector_data:
                            det_data = detector_data[detector]
                            if isinstance(det_data, dict) and "strain" in det_data:
                                strain = det_data["strain"]
                                if isinstance(strain, np.ndarray):
                                    strain_tensor = torch.from_numpy(strain).float().to(self.device)
                                else:
                                    strain_tensor = torch.tensor(strain, dtype=torch.float32, device=self.device)
                                strain_list.append(strain_tensor)
                            else:
                                strain_list.append(torch.zeros(16384, dtype=torch.float32, device=self.device))
                        else:
                            strain_list.append(torch.zeros(16384, dtype=torch.float32, device=self.device))
                    
                    strain_batch = torch.stack(strain_list).unsqueeze(0)  # [1, 3, 16384]
                    
                    # Sample posterior
                    posterior_samples = self.model.sample_posterior(
                        strain_batch,
                        n_samples=n_posterior,
                        return_all_samples=True
                    )  # [1, n_posterior, 10]
                    
                    # Extract distance samples (index 2: luminosity_distance - must match param_names order)
                    distance_samples = posterior_samples[0, :, 2].cpu().numpy()
                    distance_samples_all.append(distance_samples)
                    
                except Exception as e:
                    logger.warning(f"  Sample {idx} failed: {e}")
                    # Add NaN placeholder
                    distance_samples_all.append(np.full(n_posterior, np.nan))
        
        logger.info(f"✅ Generated posterior samples")
        return distance_samples_all
    
    def compute_rank_histogram(self, distance_samples_all, distances_true):
        """
        Compute rank histogram for distance parameter.
        
        For each sample, rank the true distance among posterior samples.
        Histogram should be uniform [0, n_posterior_samples].
        """
        ranks = []
        
        logger.info(f"Computing ranks for {len(distances_true)} samples...")
        
        for i, true_distance in enumerate(distances_true):
            if i % 100 == 0:
                logger.info(f"  Processing {i}/{len(distances_true)}...")
            
            # Get posterior samples for this data point (sample, distance, posterior_samples)
            if i < len(distance_samples_all):
                posterior_samples = distance_samples_all[i]  # [n_posterior_samples]
                
                # Compute rank: how many posterior samples are < true value?
                rank = np.sum(posterior_samples < true_distance)
                ranks.append(rank)
        
        return np.array(ranks)
    
    def compute_coverage(self, distance_samples_all, distances_true, ci=0.68):
        """Compute coverage: fraction of true values in credible interval."""
        coverage_all = []
        coverage_by_snr = {}
        
        for i, true_distance in enumerate(distances_true):
            if i < len(distance_samples_all):
                posterior_samples = distance_samples_all[i]
                
                # Compute credible interval
                lower = np.percentile(posterior_samples, (1 - ci) / 2 * 100)
                upper = np.percentile(posterior_samples, (1 + ci) / 2 * 100)
                
                in_ci = (lower <= true_distance <= upper)
                coverage_all.append(1.0 if in_ci else 0.0)
        
        return np.array(coverage_all)
    
    def compute_bias(self, distance_samples_all, distances_true):
        """Compute median bias: median(posterior) - true."""
        biases = []
        errors = []
        
        for i, true_distance in enumerate(distances_true):
            if i < len(distance_samples_all):
                posterior_samples = distance_samples_all[i]
                
                median_pred = np.median(posterior_samples)
                bias = median_pred - true_distance
                error = np.abs(bias)
                
                biases.append(bias)
                errors.append(error)
        
        return np.array(biases), np.array(errors)
    
    def plot_rank_histogram(self, ranks, output_path="rank_histogram.png"):
        """Plot rank histogram - should be uniform."""
        n_bins = len(np.unique(ranks))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        counts, bins, patches = ax.hist(ranks, bins=n_bins, alpha=0.7, edgecolor='black')
        
        expected = len(ranks) / n_bins
        ax.axhline(expected, color='red', linestyle='--', linewidth=2, label=f'Expected (uniform): {expected:.1f}')
        
        sigma = np.sqrt(expected)
        ax.fill_between(range(n_bins+1), expected - sigma, expected + sigma, 
                        alpha=0.2, color='red', label='±1σ band')
        
        ax.set_xlabel('Rank of True Distance', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('SBC Rank Histogram for Luminosity Distance\n(Should be Uniform)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Saved: {output_path}")
        plt.close()
    
    def plot_coverage_vs_snr(self, coverage, snrs, output_path="coverage_vs_snr.png"):
        """Plot coverage as function of SNR."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Coverage by SNR bin
        ax = axes[0]
        snr_bins = [(0, 15), (15, 25), (25, 40), (40, 100)]
        snr_labels = ['Weak\n5-15', 'Low\n15-25', 'Medium\n25-40', 'Strong\n40+']
        coverages = []
        
        for snr_min, snr_max in snr_bins:
            mask = (snrs >= snr_min) & (snrs < snr_max)
            if np.sum(mask) > 0:
                cov = np.mean(coverage[mask])
                coverages.append(cov)
            else:
                coverages.append(np.nan)
        
        bars = ax.bar(snr_labels, coverages, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axhline(0.68, color='red', linestyle='--', linewidth=2, label='Target (68%)')
        ax.axhline(0.68 - 0.05, color='orange', linestyle=':', linewidth=1.5, label='±5% band')
        ax.axhline(0.68 + 0.05, color='orange', linestyle=':', linewidth=1.5)
        
        for i, (bar, cov) in enumerate(zip(bars, coverages)):
            height = bar.get_height()
            if not np.isnan(cov):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{cov:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Coverage (90% CI)', fontsize=12)
        ax.set_title('Coverage by SNR Regime', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Coverage scatter
        ax = axes[1]
        scatter = ax.scatter(snrs, coverage, alpha=0.5, s=30, c=coverage, cmap='RdYlGn')
        ax.axhline(0.68, color='red', linestyle='--', linewidth=2, label='Target')
        ax.set_xlabel('True SNR', fontsize=12)
        ax.set_ylabel('Coverage (90% CI)', fontsize=12)
        ax.set_title('Coverage vs True SNR', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Coverage')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Saved: {output_path}")
        plt.close()
    
    def plot_bias_vs_distance(self, biases, distances_true, snrs, output_path="bias_vs_distance.png"):
        """Plot median bias vs true distance."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Bias vs true distance
        ax = axes[0, 0]
        scatter = ax.scatter(distances_true, biases, c=snrs, cmap='viridis', alpha=0.6, s=30)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('True Distance (Mpc)', fontsize=12)
        ax.set_ylabel('Median Bias (Mpc)', fontsize=12)
        ax.set_title('Distance Bias vs True Distance', fontsize=13, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('SNR')
        ax.grid(True, alpha=0.3)
        
        # Bias by SNR bin
        ax = axes[0, 1]
        snr_bins = [(0, 15), (15, 25), (25, 40), (40, 100)]
        snr_labels = ['5-15', '15-25', '25-40', '40+']
        bias_means = []
        bias_stds = []
        
        for snr_min, snr_max in snr_bins:
            mask = (snrs >= snr_min) & (snrs < snr_max)
            if np.sum(mask) > 0:
                bias_means.append(np.mean(biases[mask]))
                bias_stds.append(np.std(biases[mask]))
            else:
                bias_means.append(0)
                bias_stds.append(0)
        
        x_pos = np.arange(len(snr_labels))
        ax.bar(x_pos, bias_means, yerr=bias_stds, alpha=0.7, edgecolor='black', capsize=5)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(snr_labels)
        ax.set_ylabel('Mean Bias (Mpc)', fontsize=12)
        ax.set_title('Distance Bias by SNR Regime', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Absolute error vs distance
        ax = axes[1, 0]
        errors = np.abs(biases)
        scatter = ax.scatter(distances_true, errors, c=snrs, cmap='viridis', alpha=0.6, s=30)
        ax.set_xlabel('True Distance (Mpc)', fontsize=12)
        ax.set_ylabel('|Median Bias| (Mpc)', fontsize=12)
        ax.set_title('Absolute Error vs True Distance', fontsize=13, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('SNR')
        ax.grid(True, alpha=0.3)
        
        # Histogram of biases
        ax = axes[1, 1]
        ax.hist(biases, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero bias')
        ax.axvline(np.mean(biases), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(biases):.1f} Mpc')
        ax.set_xlabel('Median Bias (Mpc)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Distance Biases', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Saved: {output_path}")
        plt.close()
    
    def print_sbc_report(self, ranks, coverage, biases, snrs, distances_true):
        """Print comprehensive SBC report."""
        print("\n" + "="*90)
        print("SIMULATION-BASED CALIBRATION (SBC) REPORT - LUMINOSITY DISTANCE")
        print("="*90)
        
        # Rank histogram test
        print(f"\n1️⃣ RANK HISTOGRAM TEST (log D_L)")
        print("-" * 90)
        print(f"   N samples: {len(ranks)}")
        print(f"   N posterior samples: {int(np.max(ranks)) + 1}")
        print(f"   Rank range: [{int(np.min(ranks))}, {int(np.max(ranks))}]")
        
        # Kolmogorov-Smirnov test
        if len(ranks) > 0:
            rank_cdf = ranks / (np.max(ranks) + 1)
            ks_stat, ks_pval = stats.kstest(rank_cdf, 'uniform')
            print(f"   KS test: stat={ks_stat:.4f}, p-value={ks_pval:.4f}")
            
            if ks_pval > 0.05:
                print(f"   ✅ RANKS ARE UNIFORM (p > 0.05) - Posterior is well-calibrated!")
            else:
                print(f"   🔴 RANKS ARE NON-UNIFORM (p < 0.05) - Posterior is biased!")
        
        # Coverage test
        print(f"\n2️⃣ COVERAGE TEST (90% Credible Interval)")
        print("-" * 90)
        overall_coverage = np.mean(coverage)
        print(f"   Overall coverage: {overall_coverage:.1%} (target: 68%)")
        
        if abs(overall_coverage - 0.68) < 0.05:
            print(f"   ✅ COVERAGE IS GOOD (within ±5% of target)")
        elif abs(overall_coverage - 0.68) < 0.10:
            print(f"   🟡 COVERAGE IS ACCEPTABLE (within ±10% of target)")
        else:
            print(f"   🔴 COVERAGE IS POOR (>±10% from target)")
        
        print(f"\n   Coverage by SNR regime:")
        snr_bins = [(0, 15), (15, 25), (25, 40), (40, 100)]
        snr_labels = ['Weak (5-15)', 'Low (15-25)', 'Medium (25-40)', 'Strong (40+)']
        
        for (snr_min, snr_max), label in zip(snr_bins, snr_labels):
            mask = (snrs >= snr_min) & (snrs < snr_max)
            if np.sum(mask) > 0:
                cov = np.mean(coverage[mask])
                n = np.sum(mask)
                status = "✅" if abs(cov - 0.68) < 0.05 else "🟡" if abs(cov - 0.68) < 0.10 else "🔴"
                print(f"      {status} {label:20s}: {cov:.1%} (N={n})")
        
        # Distance bias test
        print(f"\n3️⃣ DISTANCE BIAS TEST (median posterior - true)")
        print("-" * 90)
        print(f"   Mean bias: {np.mean(biases):.1f} Mpc")
        print(f"   Median bias: {np.median(biases):.1f} Mpc")
        print(f"   Std bias: {np.std(biases):.1f} Mpc")
        print(f"   Max |bias|: {np.max(np.abs(biases)):.1f} Mpc")
        
        if np.std(biases) < 50:
            print(f"   ✅ BIAS IS SMALL (σ < 50 Mpc)")
        elif np.std(biases) < 100:
            print(f"   🟡 BIAS IS MODERATE (50 < σ < 100 Mpc)")
        else:
            print(f"   🔴 BIAS IS LARGE (σ > 100 Mpc)")
        
        print(f"\n   Bias by SNR regime:")
        for (snr_min, snr_max), label in zip(snr_bins, snr_labels):
            mask = (snrs >= snr_min) & (snrs < snr_max)
            if np.sum(mask) > 0:
                mean_bias = np.mean(biases[mask])
                std_bias = np.std(biases[mask])
                n = np.sum(mask)
                status = "✅" if abs(mean_bias) < 50 and std_bias < 100 else "🔴"
                print(f"      {status} {label:20s}: {mean_bias:+7.1f} ± {std_bias:6.1f} Mpc (N={n})")
        
        # Final verdict
        print("\n" + "="*90)
        print("🎯 FINAL VERDICT")
        print("="*90)
        
        is_uniform = ks_pval > 0.05 if len(ranks) > 0 else False
        is_calibrated = abs(overall_coverage - 0.68) < 0.05
        is_unbiased = np.std(biases) < 50
        
        if is_uniform and is_calibrated and is_unbiased:
            print("✅ MODEL IS WELL-CALIBRATED")
            print("   - Rank histogram is uniform")
            print("   - Coverage is at target (68%)")
            print("   - Distance bias is small (<50 Mpc)")
            print("\n   ➜ NEXT: Deploy model or run full inference")
        elif is_calibrated and is_unbiased:
            print("🟡 MODEL IS PARTIALLY CALIBRATED")
            print("   - Coverage and bias are acceptable")
            print("   - But rank histogram shows deviation")
            print("\n   ➜ NEXT: Add distance-specific loss or recalibrate")
        else:
            print("🔴 MODEL IS POORLY CALIBRATED")
            print(f"   - Rank test: {'PASS ✅' if is_uniform else 'FAIL 🔴'}")
            print(f"   - Coverage: {'PASS ✅' if is_calibrated else 'FAIL 🔴'}")
            print(f"   - Bias: {'PASS ✅' if is_unbiased else 'FAIL 🔴'}")
            print("\n   ➜ CRITICAL: Fix loss function or data generation")
        
        print("=" * 90 + "\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SBC for distance parameter")
    parser.add_argument("--model-path", default="models/neuralpe2/best_model.pth",
                       help="Path to trained model checkpoint")
    parser.add_argument("--data-path", default="data/dataset2/validation",
                       help="Path to validation data")
    parser.add_argument("--n-samples", type=int, default=200,
                       help="Number of samples for SBC")
    parser.add_argument("--n-posterior", type=int, default=300,
                       help="Number of posterior samples per data point")
    parser.add_argument("--device", default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--output-dir", default="sbc_results",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize SBC with model
    sbc = DistanceSBC(model_path=args.model_path, device=device)
    
    if sbc.model is None:
        logger.error("❌ Failed to load model. Cannot proceed.")
        return 1
    
    # Load data
    samples = sbc.load_validation_data(args.data_path, max_samples=args.n_samples)
    distances_true, snrs_true, event_types = sbc.extract_distance_data(samples)
    
    if len(distances_true) == 0:
        logger.error("❌ No valid samples extracted")
        return 1
    
    logger.info(f"✅ Extracted {len(distances_true)} samples for SBC")
    
    # Generate REAL posterior samples from model
    distance_samples_all = sbc.generate_posterior_samples(samples, n_posterior=args.n_posterior)
    
    if distance_samples_all is None or len(distance_samples_all) == 0:
        logger.error("❌ Failed to generate posterior samples")
        return 1
    
    # Compute SBC metrics
    logger.info("Computing SBC metrics...")
    ranks = sbc.compute_rank_histogram(distance_samples_all, distances_true)
    coverage = sbc.compute_coverage(distance_samples_all, distances_true)
    biases, errors = sbc.compute_bias(distance_samples_all, distances_true)
    
    # Print report
    sbc.print_sbc_report(ranks, coverage, biases, snrs_true, distances_true)
    
    # Plot results
    logger.info("Generating plots...")
    sbc.plot_rank_histogram(ranks, output_path=f"{args.output_dir}/01_rank_histogram.png")
    sbc.plot_coverage_vs_snr(coverage, snrs_true, output_path=f"{args.output_dir}/02_coverage_vs_snr.png")
    sbc.plot_bias_vs_distance(biases, distances_true, snrs_true, output_path=f"{args.output_dir}/03_bias_vs_distance.png")
    
    logger.info(f"✅ SBC complete! Results saved to {args.output_dir}/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
