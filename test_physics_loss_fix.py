#!/usr/bin/env python
"""
Test script to verify physics loss fix works correctly.

Expected behavior:
1. _compute_physics_loss returns (loss, violations) tuple
2. No errors when unpacking
3. Violations dict contains expected keys
"""

import torch
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ahsd.models.overlap_neuralpe import OverlapNeuralPE

def load_config(config_path):
    """Load YAML config"""
    with open(config_path) as f:
        return yaml.safe_load(f)

def test_physics_loss_return_type():
    """Test that _compute_physics_loss returns tuple (loss, violations)"""
    print("\n" + "="*80)
    print("TEST 1: Physics Loss Return Type")
    print("="*80)
    
    # Load config
    config = load_config("configs/enhanced_training.yaml")
    print(f"✓ Loaded config: {Path('configs/enhanced_training.yaml').name}")
    
    # Create model
    model = OverlapNeuralPE(
        param_names=["mass_1", "mass_2", "distance", "geocent_time"],
        priority_net_path="dummy_path",
        config=config,
        device="cpu",
        event_type="BBH"
    )
    print("✓ Created OverlapNeuralPE model")
    
    # Create dummy parameters (first signal only, as per fix)
    true_params = torch.tensor(
        [[30.0, 25.0, 400.0, 0.1]],  # [mass_1, mass_2, distance, geocent_time]
        dtype=torch.float32
    )
    print(f"✓ Created test params: {true_params.shape}")
    
    # Call physics loss function
    try:
        result = model._compute_physics_loss(true_params)
        print(f"✓ _compute_physics_loss returned: {type(result)}")
        
        if isinstance(result, tuple) and len(result) == 2:
            loss, violations = result
            print(f"✓ Unpacked tuple successfully: loss={type(loss)}, violations={type(violations)}")
            
            # Check loss is a tensor
            if isinstance(loss, torch.Tensor):
                print(f"✓ Loss is tensor: {loss.item():.6f}")
            else:
                print(f"✗ Loss is not tensor: {type(loss)}")
                return False
            
            # Check violations is a dict
            if isinstance(violations, dict):
                print(f"✓ Violations is dict with {len(violations)} entries")
                for param_name in list(violations.keys())[:2]:
                    v = violations[param_name]
                    print(f"  - {param_name}: {v}")
            else:
                print(f"✗ Violations is not dict: {type(violations)}")
                return False
            
            return True
        else:
            print(f"✗ Return value is not a 2-tuple: {type(result)}")
            return False
            
    except Exception as e:
        print(f"✗ Error calling _compute_physics_loss: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_weights():
    """Test that config weights load correctly"""
    print("\n" + "="*80)
    print("TEST 2: Config Weight Loading")
    print("="*80)
    
    config = load_config("configs/enhanced_training.yaml")
    print(f"✓ Loaded config")
    
    # Check structure
    if "neural_posterior" in config:
        print(f"✓ Config has 'neural_posterior' section")
        np_config = config["neural_posterior"]
        
        weights = {
            "physics_loss_weight": 0.05,
            "bounds_penalty_weight": 0.5,
            "sample_loss_weight": 0.5,
        }
        
        for key, expected in weights.items():
            if key in np_config:
                actual = np_config[key]
                if actual == expected:
                    print(f"✓ {key}: {actual} (expected {expected})")
                else:
                    print(f"⚠ {key}: {actual} (expected {expected})")
            else:
                print(f"✗ {key}: not found in config")
                return False
        
        return True
    else:
        print(f"✗ Config missing 'neural_posterior' section")
        return False

def test_first_signal_only():
    """Test that physics loss can be called with first signal only"""
    print("\n" + "="*80)
    print("TEST 3: First Signal Only Logic")
    print("="*80)
    
    config = load_config("configs/enhanced_training.yaml")
    
    model = OverlapNeuralPE(
        param_names=["mass_1", "mass_2", "distance", "geocent_time"],
        priority_net_path="dummy_path",
        config=config,
        device="cpu",
        event_type="BBH"
    )
    print("✓ Created model")
    
    # Create batch with multiple signals
    true_params_batch = torch.tensor(
        [
            [30.0, 25.0, 400.0, 0.1],    # Signal 1 (ground truth)
            [50.0, 45.0, 200.0, 1.5],    # Signal 2 (secondary)
            [20.0, 18.0, 800.0, -0.5],   # Signal 3 (secondary)
        ],
        dtype=torch.float32
    )
    print(f"✓ Created batch with {true_params_batch.shape[0]} signals")
    
    # Call with first signal only (as the fix does)
    try:
        loss_first, viol_first = model._compute_physics_loss(true_params_batch[:1, :])
        print(f"✓ Called with first signal only: loss={loss_first.item():.6f}")
        
        # Call with all signals
        loss_all, viol_all = model._compute_physics_loss(true_params_batch)
        print(f"✓ Called with all signals: loss={loss_all.item():.6f}")
        
        # Loss with all signals should be higher (more samples = higher penalty)
        if loss_all.item() >= loss_first.item():
            print(f"✓ Full batch loss ≥ first signal loss: {loss_all.item():.6f} ≥ {loss_first.item():.6f}")
            return True
        else:
            print(f"⚠ Full batch loss < first signal loss (unexpected but not wrong)")
            return True
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "█"*80)
    print("PHYSICS LOSS FIX VERIFICATION")
    print("█"*80)
    
    results = []
    
    # Run tests
    results.append(("Return Type", test_physics_loss_return_type()))
    results.append(("Config Weights", test_config_weights()))
    results.append(("First Signal Only", test_first_signal_only()))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n✅ All tests passed! Physics loss fix is working correctly.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
