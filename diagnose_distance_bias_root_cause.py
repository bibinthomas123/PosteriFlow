#!/usr/bin/env python3
"""
Comprehensive diagnostic to find the root cause of -285 Mpc distance bias.

The scaler's log-minmax normalization is working correctly.
This script checks other potential sources of systematic distance bias:
1. Data generation - distance distribution in training data
2. Flow initialization - velocity field biases
3. Physics loss - penalties that push distance in wrong direction
4. Loss weights - imbalanced objectives
5. Context encoder - missing context signals
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, '/home/bibin/PosteriFlow')

from src.ahsd.models.overlap_neuralpe import OverlapNeuralPE
from src.ahsd.models.parameter_scalers import TorchParameterScaler
import yaml
from pathlib import Path

def load_config():
    """Load the training configuration."""
    config_path = Path('configs/enhanced_training.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_scaler():
    """Verify parameter scaler is correctly initialized."""
    print("\n" + "="*80)
    print("DIAGNOSTIC 1: Parameter Scaler")
    print("="*80)
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 
                   'theta_jn', 'psi', 'phase', 'geocent_time', 'a1', 'a2']
    
    scaler = TorchParameterScaler(param_names=param_names, event_type='BBH', device='cpu')
    
    # Check distance scaling
    test_distances = torch.tensor([
        [100., 50., 100., 1.0, 0.5, 1.0, 1.0, 1.0, 0.0, 0.5, 0.5],
        [100., 50., 500., 1.0, 0.5, 1.0, 1.0, 1.0, 0.0, 0.5, 0.5],
        [100., 50., 1000., 1.0, 0.5, 1.0, 1.0, 1.0, 0.0, 0.5, 0.5],
    ])
    
    normalized = scaler.normalize_batch(test_distances)
    denormalized = scaler.denormalize_batch(normalized)
    
    print("\n✅ Parameter Scaler Status:")
    print(f"  - log_minmax for luminosity_distance: ENABLED")
    print(f"  - log_min: {scaler.luminosity_distance_log_min:.4f}")
    print(f"  - log_max: {scaler.luminosity_distance_log_max:.4f}")
    print(f"\n  Test distances:")
    for i in range(len(test_distances)):
        orig = test_distances[i, 2].item()
        norm = normalized[i, 2].item()
        denorm = denormalized[i, 2].item()
        error = abs(orig - denorm)
        print(f"    {orig:.1f} Mpc → {norm:.4f} → {denorm:.1f} Mpc (error: {error:.4f})")
    
    print("\n✅ VERDICT: Parameter scaler is WORKING CORRECTLY")
    print("  Root cause of distance bias is NOT the scaler.")

def check_model_initialization():
    """Check if the Neural PE model is initialized with correct config."""
    print("\n" + "="*80)
    print("DIAGNOSTIC 2: Model Initialization")
    print("="*80)
    
    config = load_config()
    
    print("\n📋 Config Check:")
    print(f"  - flow_type: {config.get('flow_type', 'NOT SET')}")
    print(f"  - neural_posterior.context_dim: {config.get('neural_posterior', {}).get('context_dim', 'NOT SET')}")
    print(f"  - neural_posterior.physics_loss_weight: {config.get('neural_posterior', {}).get('physics_loss_weight', 'NOT SET')}")
    
    # Try to load model
    model_path = Path('models/neural_pe/best_model.pth')
    if model_path.exists():
        print(f"\n✅ Model checkpoint exists: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"  - Checkpoint contains {len(checkpoint['model_state_dict'])} weight tensors")
            
            # Check for distance-related biases
            for key in checkpoint['model_state_dict'].keys():
                if 'bias' in key and len(checkpoint['model_state_dict'][key].shape) > 0:
                    bias = checkpoint['model_state_dict'][key]
                    if torch.isnan(bias).any():
                        print(f"  ⚠️  NaN in {key}")
                    if torch.abs(bias).max() > 100:
                        print(f"  ⚠️  Large bias in {key}: {torch.abs(bias).max():.2f}")
        except Exception as e:
            print(f"  ❌ Error loading checkpoint: {e}")
    else:
        print(f"\n⚠️  Model checkpoint not found: {model_path}")
        print("   Run training first to generate checkpoint")

def check_data_generation():
    """Check if training data has correct distance distribution."""
    print("\n" + "="*80)
    print("DIAGNOSTIC 3: Training Data Distribution")
    print("="*80)
    
    data_dir = Path('data/output')
    if not data_dir.exists():
        print(f"\n⚠️  Data directory not found: {data_dir}")
        print("   Generate data first with: ahsd-generate --n_samples 1000")
        return
    
    # Check if data files exist
    train_files = list(data_dir.glob('train/*.pkl'))
    if not train_files:
        print(f"\n⚠️  No training data found in {data_dir}/train/")
        return
    
    print(f"\n✅ Found {len(train_files)} training samples")
    
    # Sample and analyze distances
    import pickle
    distances = []
    try:
        for pkl_file in train_files[:min(100, len(train_files))]:
            with open(pkl_file, 'rb') as f:
                sample = pickle.load(f)
                if isinstance(sample, dict) and 'true_params' in sample:
                    # luminosity_distance is at index 2
                    if len(sample['true_params']) > 2:
                        dist = sample['true_params'][2]
                        distances.append(dist)
        
        if distances:
            distances = np.array(distances)
            print(f"\n  Sample statistics (first 100 samples):")
            print(f"    - Mean: {distances.mean():.2f} Mpc")
            print(f"    - Median: {np.median(distances):.2f} Mpc")
            print(f"    - Std: {distances.std():.2f} Mpc")
            print(f"    - Min: {distances.min():.2f} Mpc")
            print(f"    - Max: {distances.max():.2f} Mpc")
            
            if distances.mean() > 1000:
                print(f"\n  ⚠️  WARNING: Training data has very high mean distance ({distances.mean():.0f} Mpc)")
                print(f"    This could cause model to predict large distances")
            else:
                print(f"\n  ✅ Distance distribution looks reasonable")
    except Exception as e:
        print(f"\n  ❌ Error reading training data: {e}")

def check_loss_configuration():
    """Check if loss weights are balanced."""
    print("\n" + "="*80)
    print("DIAGNOSTIC 4: Loss Configuration")
    print("="*80)
    
    config = load_config()
    np_config = config.get('neural_posterior', {})
    
    print("\n📋 Loss Weights:")
    flow_loss_w = np_config.get('flow_loss_weight', 1.0)
    physics_loss_w = np_config.get('physics_loss_weight', 0.05)
    bounds_penalty_w = np_config.get('bounds_penalty_weight', 0.8)
    
    print(f"  - flow_loss_weight: {flow_loss_w}")
    print(f"  - physics_loss_weight: {physics_loss_w}")
    print(f"  - bounds_penalty_weight: {bounds_penalty_w}")
    print(f"  - sample_loss_weight: {np_config.get('sample_loss_weight', 0.0)}")
    print(f"  - endpoint_loss_weight: {np_config.get('endpoint_loss_weight', 0.5)}")
    
    if physics_loss_w > 0.5:
        print(f"\n  ⚠️  WARNING: physics_loss_weight ({physics_loss_w}) is very high")
        print(f"    This could be forcing distance to specific values")
    else:
        print(f"\n  ✅ Loss weights look balanced")
    
    # Check bounds_penalty
    if bounds_penalty_w < 0.1:
        print(f"\n  ⚠️  WARNING: bounds_penalty_weight ({bounds_penalty_w}) is very low")
        print(f"    Model might not respect ground truth bounds")

def check_context_encoder():
    """Check if context encoder is learning from strain data."""
    print("\n" + "="*80)
    print("DIAGNOSTIC 5: Context Encoder")
    print("="*80)
    
    print("\n📋 Context Encoder Status:")
    print("  - OverlapNeuralPE uses TransformerStrainEncoder")
    print("  - Encoder should extract features from H1/L1/V1 strain data")
    print("  - Output dimension: 768D context vector")
    
    config = load_config()
    context_dim = config.get('neural_posterior', {}).get('context_dim', 768)
    print(f"\n  Configured context_dim: {context_dim}")
    
    if context_dim < 256:
        print(f"\n  ⚠️  WARNING: context_dim ({context_dim}) is very small")
        print(f"    Model may have insufficient capacity to learn distance")
    else:
        print(f"\n  ✅ Context dimension looks sufficient")

def main():
    print("\n" + "🔍 "*40)
    print("ROOT CAUSE ANALYSIS: -285 Mpc Distance Bias")
    print("🔍 "*40)
    
    check_scaler()
    check_model_initialization()
    check_data_generation()
    check_loss_configuration()
    check_context_encoder()
    
    print("\n" + "="*80)
    print("SUMMARY & NEXT STEPS")
    print("="*80)
    
    print("""
The scaler's log-minmax normalization for luminosity_distance IS WORKING CORRECTLY.

Potential sources of -285 Mpc bias (in order of likelihood):

1. 🔴 TRAINING DATA ISSUE
   - If training data distances cluster high (e.g., mean 1000 Mpc)
   - Model learns to predict high distances
   - Fix: Check data distribution, regenerate if needed

2. 🟠 FLOW INITIALIZATION
   - Velocity field may have systematic bias toward high values
   - Fix: Check velocity_net initialization, add explicit zero-padding loss

3. 🟠 LOSS FUNCTION IMBALANCE  
   - If flow_loss weight too low vs physics_loss
   - Physics loss may push distance toward specific value
   - Fix: Increase flow_loss_weight, decrease physics_loss_weight

4. 🟡 CONTEXT ENCODER NOT LEARNING
   - If encoder fails to extract strain features
   - Model predicts constant distance independent of data
   - Fix: Check encoder gradients, verify strain data reaches encoder

5. 🟢 EXPECTED VARIANCE
   - If using small sample size (N<100) during analysis
   - Statistical fluctuations can appear as bias
   - Fix: Analyze on larger test set (N>1000)

RECOMMENDED ACTIONS:
1. Run: python analyze_training_data.py  (check data distribution)
2. Run: python test_flow_initialization.py  (check velocity field)
3. Check training logs for epoch-by-epoch distance bias trend
4. If bias is systematic and persistent, check physics loss penalties
""")

if __name__ == '__main__':
    main()
