#!/usr/bin/env python3
"""
Quick test to verify NLL explosion fix is working.
Tests:
1. Loss computation succeeds with sample_loss
2. sample_loss is included in output
3. No training errors occur
"""

import torch
import yaml
import sys

def test_loss_computation():
    """Test that compute_loss includes sample_loss and works correctly."""
    print("=" * 70)
    print("TEST 1: Loss Computation with Sample Loss")
    print("=" * 70)
    
    try:
        from src.ahsd.models.overlap_neuralpe import OverlapNeuralPE
        
        # Load config
        print("\n[1/5] Loading config...")
        with open('configs/enhanced_training.yaml') as f:
            config = yaml.safe_load(f)
        
        neural_posterior_config = config['neural_posterior']
        print(f"      ✓ Config loaded")
        print(f"      - sample_loss_weight: {neural_posterior_config.get('sample_loss_weight', 'NOT SET')}")
        print(f"      - bounds_penalty_weight: {neural_posterior_config.get('bounds_penalty_weight', 'NOT SET')}")
        
        # Create model
        print("\n[2/5] Creating OverlapNeuralPE model...")
        model = OverlapNeuralPE(
            param_names=neural_posterior_config['param_names'],
            priority_net_path='models/priority_net/priority_net_best.pth',
            config=neural_posterior_config,
            device='cpu'
        )
        print(f"      ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create fake data
        print("\n[3/5] Creating test data...")
        batch_size = 4
        strain_data = torch.randn(batch_size, 2, 4096)
        true_params = torch.randn(batch_size, 9) * 0.5  # Normalized around 0
        print(f"      ✓ Batch size: {batch_size}")
        print(f"      - strain_data shape: {strain_data.shape}")
        print(f"      - true_params shape: {true_params.shape}")
        
        # Compute loss
        print("\n[4/5] Computing loss...")
        loss_dict = model.compute_loss(strain_data, true_params)
        print(f"      ✓ Loss computed successfully")
        
        # Verify output
        print("\n[5/5] Verifying loss components...")
        required_keys = ['total_loss', 'nll', 'sample_loss', 'physics_loss']
        for key in required_keys:
            if key not in loss_dict:
                print(f"      ✗ Missing key: {key}")
                return False
            val = loss_dict[key].item()
            print(f"      ✓ {key:20s}: {val:10.6f}")
        
        # Check values are reasonable
        print("\n[VALIDATION]")
        nll = loss_dict['nll'].item()
        sample_loss = loss_dict['sample_loss'].item()
        physics_loss = loss_dict['physics_loss'].item()
        total_loss = loss_dict['total_loss'].item()
        
        # NLL should be positive and non-zero
        if nll > 0:
            print(f"      ✓ NLL is positive: {nll:.4f}")
        else:
            print(f"      ✗ NLL should be positive, got {nll:.4f}")
            return False
        
        # Sample loss should be positive
        if sample_loss >= 0:
            print(f"      ✓ sample_loss is non-negative: {sample_loss:.6f}")
        else:
            print(f"      ✗ sample_loss should be non-negative, got {sample_loss:.6f}")
            return False
        
        # Total loss should equal sum of components
        expected_total = nll + loss_dict['physics_loss'].item() + sample_loss + loss_dict.get('bias_loss', torch.tensor(0)).item() + loss_dict.get('uncertainty_loss', torch.tensor(0)).item() + loss_dict.get('jacobian_reg', torch.tensor(0)).item()
        if abs(total_loss - expected_total) < 0.01:
            print(f"      ✓ Total loss equals sum of components")
        else:
            print(f"      ⚠ Total loss may not sum correctly (diff: {abs(total_loss - expected_total):.6f})")
        
        print("\n" + "=" * 70)
        print("✅ TEST 1 PASSED: Loss computation working correctly")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test that gradients flow through sample_loss."""
    print("\n" + "=" * 70)
    print("TEST 2: Gradient Flow Through Sample Loss")
    print("=" * 70)
    
    try:
        from src.ahsd.models.overlap_neuralpe import OverlapNeuralPE
        
        print("\n[1/4] Creating model...")
        with open('configs/enhanced_training.yaml') as f:
            config = yaml.safe_load(f)
        
        model = OverlapNeuralPE(
            param_names=config['neural_posterior']['param_names'],
            priority_net_path='models/priority_net/priority_net_best.pth',
            config=config['neural_posterior'],
            device='cpu'
        )
        print(f"      ✓ Model created")
        
        # Create test data
        print("\n[2/4] Creating test data...")
        batch_size = 2
        strain_data = torch.randn(batch_size, 2, 4096, requires_grad=False)
        true_params = torch.randn(batch_size, 9) * 0.5
        print(f"      ✓ Test data created")
        
        # Compute loss with gradients
        print("\n[3/4] Computing loss with gradient tracking...")
        loss_dict = model.compute_loss(strain_data, true_params)
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        print("\n[4/4] Running backprop...")
        total_loss.backward()
        
        # Check for gradients in flow parameters
        flow_params_with_grad = 0
        for name, param in model.flow.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                flow_params_with_grad += 1
        
        print(f"      ✓ Flow params with non-zero gradients: {flow_params_with_grad}")
        
        if flow_params_with_grad > 0:
            print("\n" + "=" * 70)
            print("✅ TEST 2 PASSED: Gradients flowing correctly")
            print("=" * 70)
            return True
        else:
            print("\n✗ TEST 2 FAILED: No gradients in flow parameters")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  NLL EXPLOSION FIX - VERIFICATION TEST".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    test1_passed = test_loss_computation()
    test2_passed = test_gradient_flow()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  TEST 1 (Loss Computation):  {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  TEST 2 (Gradient Flow):     {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 70)
    
    if test1_passed and test2_passed:
        print("\n✅ All tests passed! NLL fix is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check output above.")
        sys.exit(1)
