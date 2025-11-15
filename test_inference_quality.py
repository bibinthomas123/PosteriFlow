#!/usr/bin/env python3
"""
Comprehensive inference pipeline quality test.
Tests posterior sampling, credible intervals, and estimation accuracy.
"""

import torch
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_signal(
    params: np.ndarray,
    sample_rate: float = 4096.0,
    duration: float = 4.0,
    noise_std: float = 1e-23,
    snr: float = 20.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic GW signal with noise.
    
    Returns:
        (strain_h1, strain_l1, signal_clean)
    """
    mass_1, mass_2, distance, ra, dec, theta_jn, psi, phase, geocent_time = params
    
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate
    
    # Create simple sinusoidal chirp-like signal
    f0 = 100  # Hz
    df_dt = 5  # Hz/s (frequency sweep)
    f = f0 + df_dt * t
    omega = 2 * np.pi * f
    phase_evolution = np.cumsum(omega) * 2 * np.pi / sample_rate
    
    # Amplitude envelope (Gaussian around geocent_time)
    t_window = 1.0  # Window width
    envelope = np.exp(-(t - geocent_time) ** 2 / (2 * t_window ** 2))
    
    # Signal amplitude (inversely proportional to distance)
    amplitude = 1e-21 * (400.0 / distance)  # 400 Mpc reference
    
    # Create waveform
    signal = amplitude * envelope * np.sin(phase_evolution + phase)
    
    # Add noise
    noise_h1 = np.random.normal(0, noise_std, n_samples).astype(np.float32)
    noise_l1 = np.random.normal(0, noise_std, n_samples).astype(np.float32)
    
    strain_h1 = (noise_h1 + signal).astype(np.float32)
    strain_l1 = (noise_l1 + signal * 0.95).astype(np.float32)  # L1 slightly different amplitude
    
    return strain_h1, strain_l1, signal.astype(np.float32)


def run_inference_test(
    model_path: str = 'models/neural_pe/best_model.pth',
    config_path: str = 'configs/enhanced_training.yaml',
    n_test_cases: int = 5,
    n_posterior_samples: int = 1000,
    device: str = 'cuda'
) -> dict:
    """Run inference quality test with synthetic signals."""
    
    logger.info("\n" + "="*100)
    logger.info("INFERENCE PIPELINE QUALITY TEST")
    logger.info("="*100)
    
    # Import pipeline
    try:
        from src.ahsd.inference.inference_pipeline import InferencePipeline, InferenceConfig
        logger.info(f"✓ InferencePipeline imported")
    except ImportError as e:
        logger.error(f"Failed to import InferencePipeline: {e}")
        return {}
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}\n")
    
    # Initialize pipeline
    logger.info(f"STEP 1: Initialize Pipeline")
    logger.info("-" * 100)
    try:
        pipeline = InferencePipeline(
            model_path=model_path,
            config_path=config_path,
            device=device,
            inference_config=InferenceConfig(
                device=device,
                n_posterior_samples=n_posterior_samples,
                batch_size=1
            )
        )
        logger.info(f"✓ Pipeline initialized with {n_posterior_samples} posterior samples\n")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    # Test cases: vary parameters
    test_cases = []
    for i in range(n_test_cases):
        mass_1 = np.random.uniform(20, 50)
        mass_2 = np.random.uniform(10, mass_1)
        distance = np.random.uniform(100, 800)
        ra = np.random.uniform(0, 2*np.pi)
        dec = np.random.uniform(-np.pi/2, np.pi/2)
        theta_jn = np.random.uniform(0, np.pi)
        psi = np.random.uniform(0, np.pi)
        phase = np.random.uniform(0, 2*np.pi)
        geocent_time = np.random.uniform(0.5, 2.5)
        
        params = np.array([mass_1, mass_2, distance, ra, dec, theta_jn, psi, phase, geocent_time])
        test_cases.append(params)
    
    # Run inference on each test case
    logger.info(f"STEP 2: Run Inference on {n_test_cases} Test Cases")
    logger.info("-" * 100)
    
    results = {
        'test_cases': n_test_cases,
        'n_samples': n_posterior_samples,
        'device': device,
        'model_path': model_path,
        'timestamp': datetime.now().isoformat(),
        'test_results': []
    }
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time']
    
    for test_idx, true_params in enumerate(test_cases):
        logger.info(f"\nTest Case {test_idx + 1}/{n_test_cases}")
        logger.info("-" * 100)
        
        try:
            # Generate synthetic signal
            strain_h1, strain_l1, signal = generate_synthetic_signal(
                true_params,
                sample_rate=4096.0,
                duration=4.0,
                noise_std=1e-23
            )
            
            strain_data = torch.tensor(
                [[strain_h1, strain_l1]], 
                dtype=torch.float32,
                device=device
            )
            
            # Log true parameters
            logger.info("True parameters:")
            for name, val in zip(param_names, true_params):
                logger.info(f"  {name:20s}: {val:10.4f}")
            
            # Run inference
            logger.info("\nRunning posterior sampling...")
            posteriors = pipeline.get_posteriors(strain_data, n_samples=n_posterior_samples)
            logger.info(f"✓ Got {posteriors['samples'].shape[1]} posterior samples")
            
            # Compute credible intervals
            logger.info("Computing credible intervals...")
            intervals = pipeline.get_credible_intervals(strain_data, credibility=0.90, n_samples=n_posterior_samples)
            logger.info(f"✓ Credible intervals computed")
            
            # Compute posterior statistics
            logger.info("Computing posterior statistics...")
            stats = pipeline.get_posterior_statistics(strain_data, n_samples=n_posterior_samples)
            
            # Compare with truth
            logger.info("\nPosterior vs True Parameters:")
            logger.info(f"{'Parameter':<20} {'True':<12} {'Mean':<12} {'Median':<12} {'Std':<12} {'In CI?':<8}")
            logger.info("-" * 100)
            
            errors = []
            in_ci_count = 0
            test_result = {
                'case': test_idx + 1,
                'true_params': true_params.tolist(),
                'parameters': {}
            }
            
            for i, name in enumerate(param_names):
                true_val = true_params[i]
                mean_val = stats[name]['mean']
                median_val = stats[name]['median']
                std_val = stats[name]['std']
                ci = intervals[name]
                in_ci = ci['lower'] <= true_val <= ci['upper']
                
                error = abs(mean_val - true_val)
                errors.append(error)
                if in_ci:
                    in_ci_count += 1
                
                ci_str = "✓" if in_ci else "✗"
                logger.info(f"{name:<20} {true_val:<12.4f} {mean_val:<12.4f} {median_val:<12.4f} {std_val:<12.4f} {ci_str:<8}")
                
                test_result['parameters'][name] = {
                    'true': float(true_val),
                    'posterior_mean': float(mean_val),
                    'posterior_median': float(median_val),
                    'posterior_std': float(std_val),
                    'ci_lower': float(ci['lower']),
                    'ci_upper': float(ci['upper']),
                    'error': float(error),
                    'in_ci': bool(in_ci)
                }
            
            # Summary metrics for this test case
            mae = np.mean(errors)
            rms = np.sqrt(np.mean(np.array(errors) ** 2))
            ci_coverage = in_ci_count / len(param_names)
            
            logger.info("\nTest Case Summary:")
            logger.info(f"  Mean Absolute Error:     {mae:.6f}")
            logger.info(f"  RMS Error:               {rms:.6f}")
            logger.info(f"  Credible Interval Coverage: {ci_coverage:.1%} ({in_ci_count}/{len(param_names)})")
            logger.info(f"  Max Error:               {np.max(errors):.6f}")
            
            test_result['metrics'] = {
                'mean_absolute_error': float(mae),
                'rms_error': float(rms),
                'ci_coverage': float(ci_coverage),
                'max_error': float(np.max(errors))
            }
            
            results['test_results'].append(test_result)
            
        except Exception as e:
            logger.error(f"Test case {test_idx + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            results['test_results'].append({
                'case': test_idx + 1,
                'error': str(e)
            })
    
    # Compute aggregate statistics
    logger.info("\n" + "="*100)
    logger.info("AGGREGATE RESULTS")
    logger.info("="*100)
    
    successful_tests = [r for r in results['test_results'] if 'metrics' in r]
    
    if successful_tests:
        all_mae = [r['metrics']['mean_absolute_error'] for r in successful_tests]
        all_rms = [r['metrics']['rms_error'] for r in successful_tests]
        all_coverage = [r['metrics']['ci_coverage'] for r in successful_tests]
        
        logger.info(f"\nSuccessful Tests: {len(successful_tests)}/{n_test_cases}")
        logger.info(f"Mean Absolute Error:     {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
        logger.info(f"RMS Error:               {np.mean(all_rms):.6f} ± {np.std(all_rms):.6f}")
        logger.info(f"CI Coverage (mean):      {np.mean(all_coverage):.1%} ± {np.std(all_coverage):.1%}")
        logger.info(f"Min/Max CI Coverage:     {np.min(all_coverage):.1%} / {np.max(all_coverage):.1%}")
        
        results['aggregate'] = {
            'successful_tests': len(successful_tests),
            'mean_absolute_error_mean': float(np.mean(all_mae)),
            'mean_absolute_error_std': float(np.std(all_mae)),
            'rms_error_mean': float(np.mean(all_rms)),
            'rms_error_std': float(np.std(all_rms)),
            'ci_coverage_mean': float(np.mean(all_coverage)),
            'ci_coverage_std': float(np.std(all_coverage)),
            'ci_coverage_min': float(np.min(all_coverage)),
            'ci_coverage_max': float(np.max(all_coverage))
        }
        
        # Quality assessment
        logger.info("\n" + "-"*100)
        logger.info("QUALITY ASSESSMENT")
        logger.info("-"*100)
        
        mae_threshold = 1.0  # meters or Mpc
        coverage_threshold = 0.85  # 85%
        
        if np.mean(all_mae) < mae_threshold:
            logger.info(f"✅ MAE {np.mean(all_mae):.4f} < {mae_threshold} (EXCELLENT)")
        elif np.mean(all_mae) < mae_threshold * 1.5:
            logger.info(f"⚠️  MAE {np.mean(all_mae):.4f} < {mae_threshold*1.5} (GOOD)")
        else:
            logger.info(f"❌ MAE {np.mean(all_mae):.4f} (NEEDS IMPROVEMENT)")
        
        if np.mean(all_coverage) > coverage_threshold:
            logger.info(f"✅ CI Coverage {np.mean(all_coverage):.1%} > {coverage_threshold:.0%} (EXCELLENT)")
        elif np.mean(all_coverage) > coverage_threshold * 0.9:
            logger.info(f"⚠️  CI Coverage {np.mean(all_coverage):.1%} (GOOD)")
        else:
            logger.info(f"❌ CI Coverage {np.mean(all_coverage):.1%} (NEEDS IMPROVEMENT)")
    
    logger.info("\n" + "="*100 + "\n")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test inference pipeline quality')
    parser.add_argument('--model', type=str, default='models/neural_pe/best_model.pth')
    parser.add_argument('--config', type=str, default='configs/enhanced_training.yaml')
    parser.add_argument('--test-cases', type=int, default=5)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-results', type=str, default=None)
    
    args = parser.parse_args()
    
    results = run_inference_test(
        model_path=args.model,
        config_path=args.config,
        n_test_cases=args.test_cases,
        n_posterior_samples=args.n_samples,
        device=args.device
    )
    
    # Save results if requested
    if args.save_results:
        output_path = Path(args.save_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    main()
