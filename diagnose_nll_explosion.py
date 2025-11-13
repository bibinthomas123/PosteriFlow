#!/usr/bin/env python3
"""
Diagnostic script to analyze NLL explosion and physics loss.

The training logs show:
- Train Loss: 5525.9 (NLL: 12.12 bits)
- Physics Loss: 5513.75 (99.8% of total loss!)
- Val Loss: 8.39 (NLL: 8.28 bits)

This suggests:
1. Flow outputs are extremely out-of-bounds during training
2. Bounds penalty (0.1 weight) too weak or not working
3. Sample loss (0.1 weight) not constraining flow properly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
import logging
from ahsd.models.overlap_neuralpe import OverlapNeuralPE
from experiments.train_priority_net import ChunkedGWDataLoader
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_bounds_violations():
    """Analyze parameter bounds violations during training."""
    
    # Load config
    with open('configs/enhanced_training.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    priority_net_path = "models/priority_net/priority_net_best.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    param_names = config['neural_posterior']['param_names']
    model = OverlapNeuralPE(
        param_names=param_names,
        priority_net_path=priority_net_path,
        config=config['neural_posterior'],
        device=device,
        event_type="BBH"
    )
    
    # Load sample data
    data_loader = ChunkedGWDataLoader(
        data_dir="data/training",
        split="train",
        batch_size=4,
        shuffle=False
    )
    
    logger.info("=" * 80)
    logger.info("ANALYZING BOUNDS VIOLATIONS")
    logger.info("=" * 80)
    
    # Get first batch
    try:
        for strain_data, parameters, n_signals, metadata in data_loader:
            strain_data = strain_data.to(device)
            parameters = parameters.to(device)
            target_params = parameters[:, 0, :]
            
            logger.info(f"\n1. TARGET PARAMETERS (Ground Truth)")
            logger.info("-" * 40)
            for i, name in enumerate(param_names):
                min_val, max_val = model.param_bounds[name]
                true_vals = target_params[:, i].cpu().detach().numpy()
                logger.info(f"  {name:20s}: min={true_vals.min():.2f}, max={true_vals.max():.2f}, bounds=[{min_val}, {max_val}]")
                violations = ((true_vals < min_val) | (true_vals > max_val)).sum()
                if violations > 0:
                    logger.warning(f"    ⚠️  {violations}/{len(true_vals)} samples OUT OF BOUNDS")
            
            logger.info(f"\n2. NORMALIZED PARAMETERS (for flow)")
            logger.info("-" * 40)
            normalized = model._normalize_parameters(target_params)
            norm_out_of_bounds = ((normalized < -1.0) | (normalized > 1.0)).sum().item()
            logger.info(f"  Out of bounds: {norm_out_of_bounds}/{normalized.numel()}")
            
            # Compute physics loss for ground truth
            context = model.context_encoder(strain_data)
            physics_loss_gt = model._compute_physics_loss(target_params)
            logger.info(f"\n3. PHYSICS LOSS (Ground Truth)")
            logger.info("-" * 40)
            logger.info(f"  Physics Loss: {physics_loss_gt.item():.6f}")
            
            # Now sample from flow and check bounds
            logger.info(f"\n4. FLOW SAMPLES (from inverse)")
            logger.info("-" * 40)
            
            n_samples = 100
            z = torch.randn(n_samples, len(param_names), device=device)
            context_expanded = context[0:1].expand(n_samples, -1)
            
            samples_norm, _ = model.flow.inverse(z, context_expanded)
            logger.info(f"  Normalized samples range: [{samples_norm.min():.4f}, {samples_norm.max():.4f}]")
            
            # Check bounds violations
            out_of_bounds_count = ((samples_norm < -1.0) | (samples_norm > 1.0)).sum().item()
            logger.info(f"  Out of bounds: {out_of_bounds_count}/{samples_norm.numel()} ({100*out_of_bounds_count/samples_norm.numel():.1f}%)")
            
            # Denormalize and check physical bounds
            samples_physical = model._denormalize_parameters(samples_norm)
            logger.info(f"\n5. PHYSICAL SAMPLES (after denormalization)")
            logger.info("-" * 40)
            for i, name in enumerate(param_names):
                min_val, max_val = model.param_bounds[name]
                phys_vals = samples_physical[:, i].cpu().detach().numpy()
                logger.info(f"  {name:20s}: range=[{phys_vals.min():.2f}, {phys_vals.max():.2f}], bounds=[{min_val}, {max_val}]")
                violations = ((phys_vals < min_val) | (phys_vals > max_val)).sum()
                if violations > 0:
                    logger.warning(f"    ⚠️  {violations}/{len(phys_vals)} samples OUT OF BOUNDS")
            
            # Compute physics loss on samples
            physics_loss_samples = model._compute_physics_loss(samples_physical)
            logger.info(f"\n6. PHYSICS LOSS (Flow Samples)")
            logger.info("-" * 40)
            logger.info(f"  Physics Loss: {physics_loss_samples.item():.6f}")
            
            # Sample loss computation
            logger.info(f"\n7. SAMPLE LOSS (Nov 13 constraint)")
            logger.info("-" * 40)
            out_of_bounds = F.relu(torch.abs(samples_norm) - 1.0)
            sample_loss = torch.mean(out_of_bounds**2)
            logger.info(f"  Raw sample loss: {sample_loss.item():.6f}")
            logger.info(f"  Weighted (w=0.1): {0.1 * sample_loss.item():.6f}")
            logger.info(f"  Max out-of-bounds: {out_of_bounds.max():.4f}")
            
            logger.info(f"\n8. WEIGHT ASSESSMENT")
            logger.info("-" * 40)
            logger.info(f"  Physics loss weight: {config['neural_posterior'].get('physics_loss_weight', 1.0)}")
            logger.info(f"  Bounds penalty weight: {config['neural_posterior'].get('bounds_penalty_weight', 0.1)}")
            logger.info(f"  Sample loss weight: {config['neural_posterior'].get('sample_loss_weight', 0.1)}")
            logger.info(f"  Ratio physics_loss / sample_loss: {physics_loss_samples.item() / sample_loss.item():.1f}x")
            
            break
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    diagnose_bounds_violations()
