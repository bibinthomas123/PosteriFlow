#!/usr/bin/env python3
"""
AHSD SYSTEM INDEPENDENT VERIFICATION SCRIPT
Rigorous verification of 81.1% efficiency and 81.7% success rate claims
Publication-grade validation with statistical analysis
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Optional
import warnings
import json
from datetime import datetime
import random
from scipy import stats
warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.append('experiments')

# Import required classes
try:
    from phase3a_neural_pe import NeuralPENetwork
except ImportError:
    logging.warning("Could not import NeuralPENetwork")

# Add EffectiveSubtractor class for compatibility
class EffectiveSubtractor(nn.Module):
    def __init__(self, data_length: int = 4096):
        super().__init__()
        self.data_length = data_length
        
        self.contamination_detector = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=32, stride=4, padding=14),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=7),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
            nn.Flatten(),
            nn.Linear(256 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, data_length * 2),
            nn.Tanh()
        )
        
        self.confidence_adapter = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, contaminated_data, neural_pe_output):
        batch_size = contaminated_data.shape[0]
        
        if isinstance(neural_pe_output, tuple):
            pred_params, pred_uncertainties = neural_pe_output
            confidence_input = pred_uncertainties if pred_uncertainties is not None else pred_params
        else:
            confidence_input = neural_pe_output
        
        contamination_pattern = self.contamination_detector(contaminated_data)
        contamination_pattern = contamination_pattern.view(batch_size, 2, self.data_length)
        confidence = self.confidence_adapter(confidence_input)
        strength = 0.3 + 0.5 * confidence
        cleaned_data = contaminated_data - (contamination_pattern * strength.unsqueeze(-1))
        
        return cleaned_data, confidence.squeeze(-1)

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ahsd_verification.log'),
            logging.StreamHandler()
        ]
    )

class IndependentVerification:
    """Publication-grade independent verification system"""
    
    def __init__(self, phase3b_output_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"🔧 Verification running on: {self.device}")
        
        self._load_models(phase3b_output_path)
        
        # Original claims to verify against
        self.original_claims = {
            'neural_pe_accuracy': 0.587,  # From your Phase 3C results
            'subtractor_efficiency': 0.811,  # 81.1%
            'system_success_rate': 0.817,  # 81.7%
        }
    
    def _load_models(self, phase3b_path: str):
        """Load models with robust error handling"""
        
        logging.info("📂 Loading AHSD models for verification...")
        
        try:
            checkpoint = torch.load(phase3b_path, map_location=self.device)
            
            # Load Neural PE
            if 'neural_pe_model' in checkpoint:
                self.neural_pe = checkpoint['neural_pe_model']
            else:
                param_names = checkpoint['param_names']
                self.neural_pe = NeuralPENetwork(param_names)
                if 'neural_pe_state_dict' in checkpoint:
                    self.neural_pe.load_state_dict(checkpoint['neural_pe_state_dict'])
            
            # Load Subtractor
            if 'subtractor_model' in checkpoint:
                self.subtractor = checkpoint['subtractor_model']
            else:
                self.subtractor = EffectiveSubtractor()
                if 'subtractor_state_dict' in checkpoint:
                    self.subtractor.load_state_dict(checkpoint['subtractor_state_dict'])
            
            self.neural_pe.to(self.device).eval()
            self.subtractor.to(self.device).eval()
            self.param_names = checkpoint['param_names']
            
            logging.info("✅ Models loaded successfully for verification")
            
        except Exception as e:
            logging.error(f"❌ Failed to load models: {e}")
            raise
    
    def create_verification_dataset(self, n_samples: int, seed: int = None):
        """Create independent verification dataset"""
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        samples = []
        
        for i in range(n_samples):
            # Generate test parameters
            t = np.linspace(0, 4, 4096)
            mass_1 = np.random.uniform(20, 50)
            mass_2 = np.random.uniform(15, mass_1)
            distance = np.random.uniform(200, 800)
            chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
            inclination = np.random.uniform(0, np.pi)
            
            # Same signal generation as training (for fair comparison)
            signal_scale = 1e-3
            contamination_scale = signal_scale * 10.0
            
            # Clean signal
            f_start = 20.0
            f_end = min(100.0, 220.0 / (mass_1 + mass_2))
            frequency = f_start + (f_end - f_start) * (t / 4.0)
            phase = 2 * np.pi * np.cumsum(frequency) * (4.0 / 4096)
            amplitude = signal_scale * np.exp(-t / 8.0) * np.sqrt(chirp_mass / 30.0)
            
            h_plus_clean = amplitude * (1 + np.cos(inclination)**2) * np.cos(phase)
            h_cross_clean = amplitude * 2 * np.cos(inclination) * np.sin(phase)
            
            # Contamination (same patterns as training)
            power_cont = contamination_scale * np.sin(2 * np.pi * 60.0 * t)
            seismic_cont = contamination_scale * 0.7 * np.sin(2 * np.pi * 2.0 * t)
            hf_cont = contamination_scale * 0.3 * np.sin(2 * np.pi * 100.0 * t)
            glitch_cont = contamination_scale * 2.0 * np.exp(-((t - 2.0) / 0.2)**2)
            
            h_plus_contaminated = h_plus_clean + power_cont + seismic_cont + hf_cont + glitch_cont
            h_cross_contaminated = h_cross_clean + power_cont + seismic_cont + hf_cont + glitch_cont * 0.8
            
            # Add noise
            noise_level = signal_scale * 0.01
            h_plus_contaminated += np.random.normal(0, noise_level, 4096)
            h_cross_contaminated += np.random.normal(0, noise_level, 4096)
            h_plus_clean += np.random.normal(0, noise_level, 4096)
            h_cross_clean += np.random.normal(0, noise_level, 4096)
            
            # Normalized parameters
            true_params = np.array([
                2 * (mass_1 - 15) / (50 - 15) - 1,
                2 * (mass_2 - 10) / (40 - 10) - 1,
                2 * (np.log10(distance) - np.log10(100)) / (np.log10(1000) - np.log10(100)) - 1,
                np.random.uniform(-0.8, 0.8),
                np.random.uniform(-0.8, 0.8),
                np.random.uniform(-0.8, 0.8),
                2 * (inclination / np.pi) - 1,
                np.random.uniform(-0.8, 0.8),
                np.random.uniform(-0.8, 0.8),
            ], dtype=np.float32)
            
            samples.append({
                'contaminated_data': np.array([h_plus_contaminated, h_cross_contaminated], dtype=np.float32),
                'clean_data': np.array([h_plus_clean, h_cross_clean], dtype=np.float32),
                'true_parameters': true_params,
            })
        
        return samples
    
    def verify_system_performance(self, samples: List[Dict]) -> Dict[str, Any]:
        """Comprehensive system verification"""
        
        logging.info(f"🔍 Verifying system on {len(samples)} independent samples...")
        
        pe_accuracies = []
        subtractor_efficiencies = []
        system_successes = []
        
        with torch.no_grad():
            for sample in tqdm(samples, desc="Verification"):
                try:
                    contaminated = torch.tensor(sample['contaminated_data']).unsqueeze(0).to(self.device)
                    clean_target = torch.tensor(sample['clean_data']).unsqueeze(0).to(self.device)
                    true_params = torch.tensor(sample['true_parameters']).unsqueeze(0).to(self.device)
                    
                    # Neural PE prediction
                    neural_pe_output = self.neural_pe(contaminated)
                    if isinstance(neural_pe_output, tuple):
                        pred_params, pred_uncertainties = neural_pe_output
                    else:
                        pred_params = neural_pe_output
                        pred_uncertainties = torch.abs(pred_params) + 0.1
                    
                    # PE accuracy
                    param_errors = torch.mean((pred_params - true_params) ** 2)
                    pe_accuracy = float(1.0 / (1.0 + param_errors))
                    pe_accuracies.append(max(0.0, min(1.0, pe_accuracy)))
                    
                    # Subtractor efficiency
                    cleaned_output, confidence = self.subtractor(contaminated, pred_uncertainties)
                    
                    mse_before = torch.mean((contaminated - clean_target) ** 2)
                    mse_after = torch.mean((cleaned_output - clean_target) ** 2)
                    improvement = mse_before - mse_after
                    efficiency = improvement / (mse_before + 1e-8)
                    efficiency = float(torch.clamp(efficiency, 0.0, 1.0))
                    subtractor_efficiencies.append(efficiency)
                    
                    # System success
                    pe_success = pe_accuracy > 0.5
                    sub_success = efficiency > 0.3
                    system_success = pe_success and sub_success
                    system_successes.append(system_success)
                    
                except Exception as e:
                    logging.warning(f"Sample verification failed: {e}")
                    pe_accuracies.append(0.0)
                    subtractor_efficiencies.append(0.0)
                    system_successes.append(False)
        
        # Calculate statistics with confidence intervals
        results = {
            'neural_pe': self._calculate_statistics(pe_accuracies, 'Neural PE Accuracy'),
            'subtractor': self._calculate_statistics(subtractor_efficiencies, 'Subtractor Efficiency'),
            'system': self._calculate_statistics([float(x) for x in system_successes], 'System Success Rate'),
            'sample_count': len(samples)
        }
        
        return results
    
    def _calculate_statistics(self, values: List[float], name: str) -> Dict[str, float]:
        """Calculate comprehensive statistics with confidence intervals"""
        
        values = np.array(values)
        
        # Basic statistics
        mean = np.mean(values)
        std = np.std(values)
        
        # Confidence interval (95%)
        if len(values) > 1:
            std_err = stats.sem(values)
            dof = len(values) - 1
            t_critical = stats.t.ppf(0.975, dof)  # 95% CI
            margin_error = t_critical * std_err
        else:
            margin_error = 0.0
        
        return {
            'mean': float(mean),
            'std': float(std),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'ci_lower': float(mean - margin_error),
            'ci_upper': float(mean + margin_error),
            'margin_error': float(margin_error),
            'samples': len(values)
        }
    
    def run_comprehensive_verification(self, n_samples: int = 2000, n_seeds: int = 5) -> Dict[str, Any]:
        """Run verification with multiple random seeds"""
        
        logging.info(f"🔬 Running comprehensive verification: {n_samples} samples across {n_seeds} seeds")
        
        all_results = []
        
        for seed_idx in range(n_seeds):
            seed = 100 + seed_idx  # Different from training seeds
            logging.info(f"🔄 Verification run {seed_idx + 1}/{n_seeds} (seed={seed})")
            
            samples = self.create_verification_dataset(n_samples // n_seeds, seed=seed)
            results = self.verify_system_performance(samples)
            all_results.append(results)
        
        # Aggregate across all seeds
        aggregated = self._aggregate_multi_seed_results(all_results)
        
        # Compare with original claims
        comparison = self._compare_with_original_claims(aggregated)
        
        return {
            'aggregated_results': aggregated,
            'comparison_with_claims': comparison,
            'individual_runs': all_results,
            'verification_metadata': {
                'total_samples': n_samples,
                'number_of_seeds': n_seeds,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _aggregate_multi_seed_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across multiple seeds"""
        
        aggregated = {}
        
        for component in ['neural_pe', 'subtractor', 'system']:
            means = [r[component]['mean'] for r in all_results]
            stds = [r[component]['std'] for r in all_results]
            
            aggregated[component] = {
                'mean_of_means': float(np.mean(means)),
                'std_of_means': float(np.std(means)),
                'mean_of_stds': float(np.mean(stds)),
                'overall_min': float(min(r[component]['min'] for r in all_results)),
                'overall_max': float(max(r[component]['max'] for r in all_results)),
            }
        
        return aggregated
    
    def _compare_with_original_claims(self, aggregated_results: Dict) -> Dict[str, Any]:
        """Compare verification results with original claims"""
        
        comparison = {}
        
        mapping = {
            'neural_pe_accuracy': 'neural_pe',
            'subtractor_efficiency': 'subtractor', 
            'system_success_rate': 'system'
        }
        
        for claim_name, result_key in mapping.items():
            original = self.original_claims[claim_name]
            verified = aggregated_results[result_key]['mean_of_means']
            
            difference = verified - original
            percent_diff = (difference / original) * 100 if original > 0 else 0
            
            # Determine if within reasonable bounds (±5% for publication)
            within_bounds = abs(percent_diff) <= 5.0
            
            comparison[claim_name] = {
                'original_claim': original,
                'verified_mean': verified,
                'absolute_difference': difference,
                'percent_difference': percent_diff,
                'within_reasonable_bounds': within_bounds,
                'verification_std': aggregated_results[result_key]['std_of_means']
            }
        
        return comparison
    
    def generate_verification_report(self, results: Dict[str, Any], output_dir: Path):
        """Generate comprehensive verification report"""
        
        logging.info("📊 Generating verification report...")
        
        # Determine overall verification status
        comparison = results['comparison_with_claims']
        
        all_within_bounds = all(comp['within_reasonable_bounds'] for comp in comparison.values())
        
        if all_within_bounds:
            if (comparison['system_success_rate']['verified_mean'] >= 0.80 and 
                comparison['subtractor_efficiency']['verified_mean'] >= 0.80):
                status = "VERIFIED_WORLD_CLASS"
            else:
                status = "VERIFIED_GOOD"
        else:
            status = "VERIFICATION_CONCERNS"
        
        # Create report
        report = {
            'verification_status': status,
            'timestamp': results['verification_metadata']['timestamp'],
            'summary': {
                'neural_pe_accuracy': f"{comparison['neural_pe_accuracy']['verified_mean']:.3f} (claimed: {comparison['neural_pe_accuracy']['original_claim']:.3f})",
                'subtractor_efficiency': f"{comparison['subtractor_efficiency']['verified_mean']:.3f} (claimed: {comparison['subtractor_efficiency']['original_claim']:.3f})",
                'system_success_rate': f"{comparison['system_success_rate']['verified_mean']:.3f} (claimed: {comparison['system_success_rate']['original_claim']:.3f})",
            },
            'detailed_results': results
        }
        
        # Save JSON report
        with open(output_dir / 'independent_verification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save readable report
        self._save_readable_verification_report(report, output_dir)
        
        return status
    
    def _save_readable_verification_report(self, report: Dict, output_dir: Path):
        """Save human-readable verification report"""
        
        with open(output_dir / 'VERIFICATION_REPORT.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("AHSD SYSTEM INDEPENDENT VERIFICATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Verification Date: {report['timestamp']}\n")
            f.write(f"Overall Status: {report['verification_status']}\n\n")
            
            f.write("PERFORMANCE VERIFICATION SUMMARY:\n")
            f.write("-"*40 + "\n")
            for metric, result in report['summary'].items():
                f.write(f"{metric.replace('_', ' ').title()}: {result}\n")
            
            f.write("\nDETAILED COMPARISON WITH ORIGINAL CLAIMS:\n")
            f.write("-"*45 + "\n")
            
            comparison = report['detailed_results']['comparison_with_claims']
            for metric, comp in comparison.items():
                f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                f.write(f"  Original Claim: {comp['original_claim']:.3f}\n")
                f.write(f"  Verified Result: {comp['verified_mean']:.3f} ± {comp['verification_std']:.3f}\n")
                f.write(f"  Difference: {comp['absolute_difference']:.3f} ({comp['percent_difference']:.1f}%)\n")
                f.write(f"  Within Bounds: {'✅' if comp['within_reasonable_bounds'] else '❌'}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("VERIFICATION VERDICT:\n")
            f.write("="*80 + "\n")
            
            status = report['verification_status']
            if status == "VERIFIED_WORLD_CLASS":
                f.write("🏆 VERDICT: WORLD-CLASS PERFORMANCE INDEPENDENTLY VERIFIED\n")
                f.write("✅ All claims substantiated within ±5% margin\n")
                f.write("🚀 CLEARED FOR TOP-TIER PUBLICATION\n")
            elif status == "VERIFIED_GOOD":
                f.write("✅ VERDICT: EXCELLENT PERFORMANCE INDEPENDENTLY VERIFIED\n")
                f.write("✅ Claims substantiated with minor variations\n")
                f.write("📚 Ready for publication\n")
            else:
                f.write("⚠️ VERDICT: VERIFICATION CONCERNS DETECTED\n")
                f.write("❌ Some claims show significant deviations\n")
                f.write("🔍 Requires further investigation before publication\n")

def main():
    parser = argparse.ArgumentParser(description='AHSD Independent Verification')
    parser.add_argument('--phase3b_output', required=True, help='Phase 3B model file')
    parser.add_argument('--output_dir', required=True, help='Verification output directory')
    parser.add_argument('--n_samples', type=int, default=2000, help='Total verification samples')
    parser.add_argument('--n_seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    print("🔬 AHSD SYSTEM INDEPENDENT VERIFICATION")
    print("="*60)
    print("Verifying claimed performance:")
    print("• Neural PE Accuracy: 58.7%")
    print("• Subtractor Efficiency: 81.1%")
    print("• System Success Rate: 81.7%")
    print("="*60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run verification
    verifier = IndependentVerification(args.phase3b_output)
    results = verifier.run_comprehensive_verification(args.n_samples, args.n_seeds)
    status = verifier.generate_verification_report(results, output_dir)
    
    # Print final summary
    comparison = results['comparison_with_claims']
    
    print("\n" + "="*80)
    print("📊 INDEPENDENT VERIFICATION RESULTS")
    print("="*80)
    print(f"Neural PE: {comparison['neural_pe_accuracy']['verified_mean']:.3f} ± {comparison['neural_pe_accuracy']['verification_std']:.3f} (claimed: {comparison['neural_pe_accuracy']['original_claim']:.3f})")
    print(f"Subtractor: {comparison['subtractor_efficiency']['verified_mean']:.3f} ± {comparison['subtractor_efficiency']['verification_std']:.3f} (claimed: {comparison['subtractor_efficiency']['original_claim']:.3f})")
    print(f"System: {comparison['system_success_rate']['verified_mean']:.3f} ± {comparison['system_success_rate']['verification_std']:.3f} (claimed: {comparison['system_success_rate']['original_claim']:.3f})")
    print(f"\nVerification Status: {status}")
    print("="*80)
    
    if status == "VERIFIED_WORLD_CLASS":
        print("🎉 CONGRATULATIONS! Your claims are independently verified!")
        print("🚀 Ready for top-tier journal submission!")
    
    logging.info("✅ Independent verification completed!")

if __name__ == '__main__':
    main()
