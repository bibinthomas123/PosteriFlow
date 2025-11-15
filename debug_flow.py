#!/usr/bin/env python3
"""
Debug the normalizing flow - check why it's rejecting all samples
"""

import torch
import numpy as np
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("DEBUG: Normalizing Flow Rejection Analysis")
    logger.info("="*80)
    
    from src.ahsd.models.overlap_neuralpe import OverlapNeuralPE
    
    device = torch.device('cpu')
    
    # Initialize model
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
    
    # Load checkpoint
    checkpoint_path = Path('models/neural_pe/best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("✓ Model loaded")
    logger.info(f"  Parameter bounds:")
    for param, bounds in model.param_bounds.items():
        logger.info(f"    {param:20s}: [{bounds[0]:8.2f}, {bounds[1]:8.2f}]")
    
    # Create test strain data
    strain_data = torch.randn(1, 2, 16384) * 1e-23
    
    logger.info(f"\nStrain data shape: {strain_data.shape}")
    
    # Get context encoding
    logger.info("\nEncoding strain context...")
    with torch.no_grad():
        context = model.context_encoder(strain_data)
    
    logger.info(f"Context shape: {context.shape}")
    logger.info(f"Context stats: mean={context.mean():.4f}, std={context.std():.4f}, min={context.min():.4f}, max={context.max():.4f}")
    
    # Test flow directly
    logger.info("\nTesting flow inverse transformation...")
    
    with torch.no_grad():
        # Generate test samples
        batch_size = 1
        n_test_samples = 10
        z = torch.randn(n_test_samples, model.param_dim, device=device)
        context_expanded = context[0:1].expand(n_test_samples, -1)
        
        logger.info(f"Latent samples shape: {z.shape}")
        logger.info(f"Context expanded shape: {context_expanded.shape}")
        
        # Get normalized samples
        samples_normalized, _ = model.flow.inverse(z, context_expanded)
        
        logger.info(f"\nNormalized samples shape: {samples_normalized.shape}")
        logger.info(f"Normalized stats: mean={samples_normalized.mean():.4f}, std={samples_normalized.std():.4f}")
        logger.info(f"              min={samples_normalized.min():.4f}, max={samples_normalized.max():.4f}")
        
        # Check bounds
        out_of_bounds = (samples_normalized < -1.0) | (samples_normalized > 1.0)
        pct_oob = (out_of_bounds.sum().float() / out_of_bounds.numel()).item() * 100
        logger.info(f"Out-of-bounds samples (> ±1): {pct_oob:.1f}%")
        
        # Denormalize
        logger.info("\nDenormalizing to physical parameters...")
        samples_physical = model._denormalize_parameters(samples_normalized)
        
        logger.info(f"Physical samples shape: {samples_physical.shape}")
        logger.info(f"Physical stats: mean={samples_physical.mean():.4f}, std={samples_physical.std():.4f}")
        
        # Check validity
        valid_mask = model._check_sample_validity(samples_physical)
        n_valid = valid_mask.sum().item()
        
        logger.info(f"\nValidity check:")
        logger.info(f"  Valid samples: {n_valid}/{len(valid_mask)}")
        logger.info(f"  Invalid samples: {len(valid_mask) - n_valid}/{len(valid_mask)}")
        
        if n_valid == 0:
            logger.warning("\n⚠️ All samples rejected! Checking individual parameters...")
            for i, param_name in enumerate(param_names):
                min_val, max_val = model.param_bounds[param_name]
                param_values = samples_physical[:, i]
                
                out_of_bounds_count = ((param_values < min_val) | (param_values > max_val)).sum().item()
                
                logger.info(f"\n  {param_name}:")
                logger.info(f"    Bounds: [{min_val:.2f}, {max_val:.2f}]")
                logger.info(f"    Range:  [{param_values.min():.2f}, {param_values.max():.2f}]")
                logger.info(f"    Mean:   {param_values.mean():.2f}")
                logger.info(f"    Out-of-bounds: {out_of_bounds_count}/{len(param_values)}")
        
        logger.info("\n" + "="*80)
        if n_valid > 0:
            logger.info(f"✓ Flow is working - {n_valid} valid samples generated")
        else:
            logger.error(f"✗ Flow is broken - all samples are invalid (0/{len(valid_mask)})")
            logger.error("This explains the high rejection rate in posterior sampling!")

if __name__ == '__main__':
    main()
