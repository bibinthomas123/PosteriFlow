#!/usr/bin/env python3
"""
Test script to validate the AdaptiveRankingLoss fix.
Verifies that correct rankings have low loss and incorrect rankings have high loss.
"""

import torch
from ahsd.core.priority_net import AdaptiveRankingLoss

def test_ranking_loss():
    """Test the fixed AdaptiveRankingLoss implementation."""
    loss_fn = AdaptiveRankingLoss(base_margin=0.05)
    
    print("=" * 70)
    print("Testing AdaptiveRankingLoss (VECTORIZED FIX)")
    print("=" * 70)
    
    # Test 1: Correct ranking (predictions match targets)
    print("\n✓ Test 1: Correct Ranking")
    predictions_correct = torch.tensor([0.1, 0.5, 0.9], requires_grad=True)
    targets = torch.tensor([0.1, 0.5, 0.9])
    loss_correct = loss_fn(predictions_correct, targets)
    print(f"  Predictions: {predictions_correct.data}")
    print(f"  Targets:     {targets}")
    print(f"  Loss: {loss_correct.item():.6f}")
    print(f"  Expected: ~0.0 (correct ranking should have minimal loss)")
    assert loss_correct.item() < 0.01, f"Loss too high: {loss_correct.item()}"
    print("  ✓ PASS")
    
    # Test 2: Reversed ranking (complete mismatch)
    print("\n✓ Test 2: Reversed Ranking")
    predictions_wrong = torch.tensor([0.9, 0.5, 0.1], requires_grad=True)
    targets_same = torch.tensor([0.1, 0.5, 0.9])
    loss_wrong = loss_fn(predictions_wrong, targets_same)
    print(f"  Predictions: {predictions_wrong.data}")
    print(f"  Targets:     {targets_same}")
    print(f"  Loss: {loss_wrong.item():.6f}")
    print(f"  Expected: >0.3 (reversed ranking should have high loss)")
    assert loss_wrong.item() > 0.3, f"Loss too low: {loss_wrong.item()}"
    print("  ✓ PASS")
    
    # Test 3: Partial mismatch (one pair correct, one pair reversed)
    print("\n✓ Test 3: Partial Mismatch")
    predictions_partial = torch.tensor([0.5, 0.3, 0.9], requires_grad=True)
    targets_same = torch.tensor([0.1, 0.5, 0.9])
    loss_partial = loss_fn(predictions_partial, targets_same)
    print(f"  Predictions: {predictions_partial.data}")
    print(f"  Targets:     {targets_same}")
    print(f"  Loss: {loss_partial.item():.6f}")
    print(f"  Expected: 0.0 < loss < 0.3 (partial recovery)")
    assert 0.0 < loss_partial.item() < 0.3, f"Loss out of range: {loss_partial.item()}"
    print("  ✓ PASS")
    
    # Test 4: Gradients flow correctly
    print("\n✓ Test 4: Gradient Flow")
    predictions_grad = torch.tensor([0.9, 0.5, 0.1], requires_grad=True)
    targets_grad = torch.tensor([0.1, 0.5, 0.9])
    loss_grad = loss_fn(predictions_grad, targets_grad)
    loss_grad.backward()
    print(f"  Gradients: {predictions_grad.grad}")
    print(f"  Expected: Negative values (push predictions down to match targets)")
    assert predictions_grad.grad is not None, "No gradients computed"
    assert (predictions_grad.grad < 0).any(), "Gradients should be negative"
    print("  ✓ PASS")
    
    # Test 5: With SNR weights
    print("\n✓ Test 5: With SNR Weights")
    predictions_snr = torch.tensor([0.9, 0.5, 0.1], requires_grad=True)
    targets_snr = torch.tensor([0.1, 0.5, 0.9])
    snr_weights = torch.tensor([0.5, 1.0, 2.0])
    loss_snr = loss_fn(predictions_snr, targets_snr, snr_weights=snr_weights)
    print(f"  Predictions: {predictions_snr.data}")
    print(f"  Targets:     {targets_snr}")
    print(f"  SNR weights: {snr_weights}")
    print(f"  Loss with weights: {loss_snr.item():.6f}")
    print(f"  Expected: High loss (weighted version of reversed ranking)")
    assert loss_snr.item() > 0.0, "Loss should be positive for wrong ranking"
    print("  ✓ PASS")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - AdaptiveRankingLoss fix is correct!")
    print("=" * 70)

if __name__ == "__main__":
    test_ranking_loss()
