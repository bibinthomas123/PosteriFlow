#!/usr/bin/env python3
"""
Test VelocityNet and FlowMatching gradient flow in isolation.
Run this to identify if gradients are vanishing in flow components.

Usage:
    python experiments/test_flow_gradients.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import logging

from src.ahsd.models.flows import VelocityNet, FlowMatchingPosterior

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_velocity_net_gradients():
    """Check if velocity network receives gradients"""
    
    logger.info("\n" + "="*70)
    logger.info("TEST 1: VelocityNet Gradient Flow")
    logger.info("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Create velocity network with simpler config (2 layers for debugging)
    vel_net = VelocityNet(
        features=9,
        context_features=512,
        hidden_dim=256,
        num_layers=2,  # Start with 2 layers for debugging
        dropout=0.1
    ).to(device)
    
    # Dummy inputs
    batch_size = 4
    z = torch.randn(batch_size, 9, device=device, requires_grad=True)
    t = torch.rand(batch_size, device=device)
    context = torch.randn(batch_size, 512, device=device)
    
    # Forward pass
    velocity = vel_net(z, t, context)
    
    # Compute loss
    loss = velocity.mean()
    
    # Backward pass
    loss.backward()
    
    # Check input gradients
    logger.info(f"\n✅ Forward pass successful")
    logger.info(f"  Output shape: {velocity.shape}")
    logger.info(f"  Loss: {loss.item():.6f}")
    
    if z.grad is not None:
        logger.info(f"  z.grad exists: ✅ norm={z.grad.norm().item():.6f}")
    else:
        logger.error(f"  z.grad exists: ❌ NO GRADIENTS")
    
    # Check layer gradients
    logger.info(f"\n📊 Per-Layer Gradient Statistics:")
    vanishing_layers = []
    
    for name, param in vel_net.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.data.norm().item()
            ratio = grad_norm / (param_norm + 1e-8)
            
            if grad_norm < 1e-6:
                status = "⚠️ VANISHING"
                vanishing_layers.append(name)
            elif grad_norm < 1e-4:
                status = "⚠️ SMALL"
            else:
                status = "✅ OK"
            
            logger.info(
                f"  {status} {name:50s} | "
                f"grad_norm={grad_norm:.6f} | param_norm={param_norm:.6f} | ratio={ratio:.6f}"
            )
    
    # Summary
    if vanishing_layers:
        logger.error(f"\n🔴 CRITICAL: Vanishing gradients in {len(vanishing_layers)} layers:")
        for layer in vanishing_layers:
            logger.error(f"   - {layer}")
        return False
    else:
        logger.info(f"\n✅ All layers have healthy gradients!")
        return True


def test_flow_matching_gradients():
    """Check if FlowMatching receives gradients"""
    
    logger.info("\n" + "="*70)
    logger.info("TEST 2: FlowMatching Posterior Gradient Flow")
    logger.info("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Create flow with simpler config
    flow = FlowMatchingPosterior(
        features=9,
        context_features=512,
        hidden_dim=256,
        num_layers=2,  # 2 layers for debugging
        solver_steps=5,
        dropout=0.1
    ).to(device)
    
    # Dummy inputs
    batch_size = 4
    x = torch.randn(batch_size, 9, device=device, requires_grad=True)
    context = torch.randn(batch_size, 512, device=device)
    
    # Forward pass
    z, log_det = flow(x, context)
    
    # Loss: reconstruction error + divergence regularization
    loss = z.mean() + log_det.mean()
    
    # Backward
    loss.backward()
    
    # Check input gradients
    logger.info(f"\n✅ Forward pass successful")
    logger.info(f"  z shape: {z.shape}")
    logger.info(f"  log_det shape: {log_det.shape}")
    logger.info(f"  Loss: {loss.item():.6f}")
    
    if x.grad is not None:
        logger.info(f"  x.grad exists: ✅ norm={x.grad.norm().item():.6f}")
    else:
        logger.error(f"  x.grad exists: ❌ NO GRADIENTS")
    
    # Check velocity network inside flow
    logger.info(f"\n📊 VelocityNet Gradient Statistics (inside FlowMatching):")
    vanishing_layers = []
    
    for name, param in flow.velocity_net.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.data.norm().item()
            ratio = grad_norm / (param_norm + 1e-8)
            
            if grad_norm < 1e-6:
                status = "⚠️ VANISHING"
                vanishing_layers.append(name)
            elif grad_norm < 1e-4:
                status = "⚠️ SMALL"
            else:
                status = "✅ OK"
            
            logger.info(
                f"  {status} {name:50s} | "
                f"grad_norm={grad_norm:.6f} | param_norm={param_norm:.6f} | ratio={ratio:.6f}"
            )
    
    # Summary
    if vanishing_layers:
        logger.error(f"\n🔴 CRITICAL: Vanishing gradients in {len(vanishing_layers)} layers:")
        for layer in vanishing_layers:
            logger.error(f"   - {layer}")
        return False
    else:
        logger.info(f"\n✅ All VelocityNet layers have healthy gradients!")
        return True


def test_simple_loss():
    """Test with even simpler configuration to isolate issues"""
    
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Simple Configuration (1 layer, small dims)")
    logger.info("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Minimal configuration
    vel_net = VelocityNet(
        features=9,
        context_features=64,  # Much smaller context
        hidden_dim=128,       # Much smaller hidden
        num_layers=1,         # Only 1 layer
        dropout=0.0           # No dropout
    ).to(device)
    
    batch_size = 2
    z = torch.randn(batch_size, 9, device=device, requires_grad=True)
    t = torch.rand(batch_size, device=device)
    context = torch.randn(batch_size, 64, device=device)
    
    # Forward
    velocity = vel_net(z, t, context)
    loss = (velocity ** 2).mean()
    
    # Backward
    loss.backward()
    
    logger.info(f"✅ Minimal config works")
    logger.info(f"  Loss: {loss.item():.6f}")
    
    if z.grad is not None:
        logger.info(f"  Gradient norm: {z.grad.norm().item():.6f}")
        return True
    else:
        logger.error(f"  ❌ No gradients even in minimal config!")
        return False


if __name__ == '__main__':
    logger.info("\n" + "="*70)
    logger.info("FLOW GRADIENT TEST SUITE")
    logger.info("="*70)
    
    results = {
        'velocity_net': test_velocity_net_gradients(),
        'flow_matching': test_flow_matching_gradients(),
        'minimal': test_simple_loss()
    }
    
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n✅ All gradient tests passed! Flow is learning correctly.")
        exit(0)
    else:
        logger.error("\n❌ Some gradient tests failed. Check output above for details.")
        exit(1)
