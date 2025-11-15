#!/usr/bin/env python3
"""
Test inference pipeline with best model
"""

import torch
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("\n" + "="*80)
    logger.info("TEST: Inference Pipeline with Best Model")
    logger.info("="*80 + "\n")
    
    # Import pipeline
    from src.ahsd.inference.inference_pipeline import InferencePipeline, InferenceConfig
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========== STEP 1: Initialize pipeline ==========
    logger.info("STEP 1: Initialize InferencePipeline")
    logger.info("-" * 80)
    
    try:
        pipeline = InferencePipeline(
            model_path='models/neural_pe/best_model.pth',
            config_path='configs/enhanced_training.yaml',
            device=str(device),
            inference_config=InferenceConfig(
                device=str(device),
                n_posterior_samples=200,  # Reduced for speed
                verbose=True
            )
        )
        logger.info("✓ Pipeline initialized\n")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== STEP 2: Create test strain data ==========
    logger.info("STEP 2: Create test strain data")
    logger.info("-" * 80)
    
    true_params = {
        'mass_1': 35.0,
        'mass_2': 30.0,
        'distance': 400.0,
        'ra': 2.5,
        'dec': -0.3,
        'theta_jn': 1.2,
        'psi': 0.8,
        'phase': 1.5,
        'geocent_time': 0.5
    }
    
    logger.info("True parameters:")
    for k, v in true_params.items():
        logger.info(f"  {k:20s}: {v:.3f}")
    
    # Create simple strain data
    sample_rate = 4096
    duration = 4
    n_samples = sample_rate * duration
    noise_std = 1e-23
    
    strain_h1 = np.random.normal(0, noise_std, n_samples).astype(np.float32)
    strain_l1 = np.random.normal(0, noise_std, n_samples).astype(np.float32)
    
    # Add simple signal
    signal_amplitude = 1e-21
    t = np.arange(n_samples) / sample_rate
    signal = signal_amplitude * np.sin(2 * np.pi * 100 * t)
    signal = signal * np.exp(-(t - true_params['geocent_time'])**2 / 0.1)
    
    strain_h1 += signal.astype(np.float32)
    strain_l1 += signal.astype(np.float32) * 0.9
    
    strain_data = torch.tensor([[strain_h1, strain_l1]], dtype=torch.float32).to(device)
    
    logger.info(f"Strain data shape: {strain_data.shape}\n")
    
    # ========== STEP 3: Get posteriors ==========
    logger.info("STEP 3: Sample from posterior")
    logger.info("-" * 80)
    
    try:
        posteriors = pipeline.get_posteriors(strain_data, n_samples=200)
        logger.info(f"✓ Posterior sampling successful")
        logger.info(f"  Samples shape: {posteriors['samples'].shape}")
        logger.info(f"  Means shape: {posteriors['means'].shape}\n")
    except Exception as e:
        logger.error(f"Posterior sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== STEP 4: Get credible intervals ==========
    logger.info("STEP 4: Compute credible intervals (90%)")
    logger.info("-" * 80)
    
    try:
        intervals = pipeline.get_credible_intervals(strain_data, n_samples=200)
        
        param_names = list(intervals.keys())
        logger.info(f"\n{'Parameter':<25} {'Median':<15} {'90% Interval':<35}")
        logger.info("-" * 75)
        
        for name in param_names:
            ci = intervals[name]
            interval_str = f"[{ci['lower']:.3f}, {ci['upper']:.3f}]"
            logger.info(f"{name:<25} {ci['median']:>12.3f}  {interval_str:>35}")
        
        logger.info()
    except Exception as e:
        logger.error(f"Credible interval computation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== STEP 5: Compare with truth ==========
    logger.info("STEP 5: Compare predictions with true values")
    logger.info("-" * 80)
    
    try:
        true_params_array = np.array([
            true_params['mass_1'],
            true_params['mass_2'],
            true_params['distance'],
            true_params['ra'],
            true_params['dec'],
            true_params['theta_jn'],
            true_params['psi'],
            true_params['phase'],
            true_params['geocent_time']
        ])
        
        comparison = pipeline.compare_to_truth(strain_data, true_params_array, n_samples=200)
        
        logger.info(f"\n{'Parameter':<25} {'True':<12} {'Predicted':<12} {'Error':<12} {'In CI?':<8}")
        logger.info("-" * 75)
        
        for param_name, data in comparison['parameters'].items():
            true_val = data['true']
            pred_val = data['posterior_mean']
            error = data['error']
            in_ci = "✓ YES" if data['within_credible_interval'] else "✗ NO"
            
            logger.info(f"{param_name:<25} {true_val:>10.3f}  {pred_val:>10.3f}  {error:>10.3f}  {in_ci:>8}")
        
        logger.info("\n" + "="*80)
        logger.info("SUMMARY METRICS")
        logger.info("="*80)
        
        metrics = comparison['global_metrics']
        logger.info(f"Parameters in credible interval: {metrics['parameters_in_ci']:.1%}")
        logger.info(f"Mean absolute error:             {metrics['mean_absolute_error']:.4f}")
        logger.info(f"RMS error:                       {metrics['rms_error']:.4f}")
        logger.info(f"Max error:                       {metrics['max_error']:.4f}")
        
        logger.info("\n✅ Inference test complete!\n")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == '__main__':
    main()
