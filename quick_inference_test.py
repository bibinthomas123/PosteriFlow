#!/usr/bin/env python3
"""
Quick inference test - Direct posterior sampling (fastest)
Skips iterative extraction, goes straight to parameter estimation
"""

import torch
import numpy as np
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("\n" + "="*80)
    logger.info("QUICK INFERENCE TEST - Neural PE Posterior Sampling")
    logger.info("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}\n")
    
    # ========== STEP 1: Prepare test data ==========
    logger.info("STEP 1: Prepare test strain data")
    logger.info("-" * 80)
    
    # True parameters (if we injected a signal)
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
    
    logger.info("Injected true parameters:")
    for key, val in true_params.items():
        logger.info(f"  {key:20s}: {val:10.3f}")
    
    # Create realistic strain data
    sample_rate = 4096
    duration = 4
    n_samples = sample_rate * duration
    noise_std = 1e-23
    
    strain_h1 = np.random.normal(0, noise_std, n_samples).astype(np.float32)
    strain_l1 = np.random.normal(0, noise_std, n_samples).astype(np.float32)
    
    # Inject simple signal
    signal_amplitude = 1e-21
    t = np.arange(n_samples) / sample_rate
    signal = signal_amplitude * np.sin(2 * np.pi * 100 * t)
    signal = signal * np.exp(-(t - true_params['geocent_time'])**2 / 0.1)
    
    strain_h1 += signal.astype(np.float32)
    strain_l1 += signal.astype(np.float32) * 0.9
    
    strain_data = torch.tensor(
        [[strain_h1, strain_l1]],
        dtype=torch.float32
    ).to(device)
    
    logger.info(f"\nStrain data: shape={strain_data.shape}, device={device}\n")
    
    # ========== STEP 2: Load model ==========
    logger.info("STEP 2: Load trained model")
    logger.info("-" * 80)
    
    try:
        from src.ahsd.models.overlap_neuralpe import OverlapNeuralPE
        
        config_path = Path('configs/enhanced_training.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        param_names = [
            'mass_1', 'mass_2', 'luminosity_distance',
            'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time'
        ]
        
        priority_net_path = Path('models/priority_net_checkpoint.pt')
        model = OverlapNeuralPE(
            param_names=param_names,
            priority_net_path=str(priority_net_path),
            config=config,
            device=str(device)
        )
        
        checkpoint_path = Path('models/neural_pe/best_model.pth')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info("✓ Model loaded successfully\n")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== STEP 3: Run posterior sampling ==========
    logger.info("STEP 3: Sample from posterior distribution")
    logger.info("-" * 80)
    
    try:
        with torch.no_grad():
            posterior = model.sample_posterior(
                strain_data=strain_data,
                n_samples=500
            )
        
        logger.info("✓ Posterior sampling complete\n")
        
    except Exception as e:
        logger.error(f"Posterior sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== STEP 4: Display results ==========
    logger.info("STEP 4: Parameter Estimates")
    logger.info("="*80)
    
    samples = posterior['samples'][0].cpu().numpy()  # [500, 9]
    param_names_list = param_names
    
    logger.info(f"\n{'Parameter':<25} {'Median':<15} {'Std Dev':<15} {'90% Interval':<35}")
    logger.info("-" * 90)
    
    results = {}
    for i, name in enumerate(param_names_list):
        param_samples = samples[:, i]
        
        median = np.median(param_samples)
        std = np.std(param_samples)
        lower = np.percentile(param_samples, 5)
        upper = np.percentile(param_samples, 95)
        
        results[name] = {
            'median': median,
            'std': std,
            'lower': lower,
            'upper': upper
        }
        
        interval_str = f"[{lower:.3f}, {upper:.3f}]"
        logger.info(f"{name:<25} {median:>12.3f}±{std:>8.3f}  {interval_str:>35}")
    
    # ========== STEP 5: Compare with injected ==========
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: Predicted vs Injected Values")
    logger.info("="*80)
    
    logger.info(f"\n{'Parameter':<25} {'Injected':<15} {'Predicted':<15} {'Error':<12} {'Within 90%?':<12}")
    logger.info("-" * 90)
    
    errors = []
    within_credible = []
    
    for name in param_names_list:
        if name in true_params:
            true_val = true_params[name]
            pred_val = results[name]['median']
            error = abs(pred_val - true_val)
            within = results[name]['lower'] <= true_val <= results[name]['upper']
            
            errors.append(error)
            within_credible.append(within)
            
            status = "✓ YES" if within else "✗ NO"
            logger.info(
                f"{name:<25} {true_val:>12.3f}    {pred_val:>12.3f}    "
                f"{error:>10.3f}  {status:>12}"
            )
    
    # ========== STEP 6: Summary statistics ==========
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    percent_within = 100.0 * np.sum(within_credible) / len(within_credible)
    
    logger.info(f"\nMean Absolute Error:        {mean_error:.4f}")
    logger.info(f"Max Absolute Error:         {max_error:.4f}")
    logger.info(f"Parameters within 90% CI:   {percent_within:.1f}%")
    logger.info(f"Posterior samples shape:    {posterior['samples'].shape}")
    logger.info(f"Context dimension:          {posterior.get('context', torch.tensor([])).shape}")
    
    logger.info("\n✅ Inference complete!\n")

if __name__ == '__main__':
    main()
