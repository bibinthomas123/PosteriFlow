#!/usr/bin/env python3
"""
Simple script to compute and display SNR-distance correlation
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ahsd.models.overlap_neuralpe import OverlapNeuralPE
from ahsd.utils.universal_config import load_config

def compute_and_display_correlation():
    """Compute SNR-distance correlation from a trained model"""
    
    print("\n" + "="*80)
    print("SNR-Distance Correlation Computation")
    print("="*80)
    
    # Load config
    config = load_config("configs/enhanced_training.yaml")
    print(f"✅ Loaded config")
    
    # Create model
    param_names = [
        'mass_1', 'mass_2', 'a1', 'a2', 
        'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'geocent_time',
        'luminosity_distance'
    ]
    
    model = OverlapNeuralPE(
        param_names=param_names,
        priority_net_path=None,
        config=config,
        device='cpu'
    )
    model.training = True
    print(f"✅ Created model")
    
    # Initialize epoch data
    model._epoch_distance_data = {'snr': [], 'log_scale': []}
    
    # Simulate some batches
    print(f"\n--- Running {3} simulated batches ---")
    for batch_idx in range(3):
        batch_size = 16
        
        # Create data with varying amplitudes
        strain_data_list = []
        for i in range(batch_size):
            # Vary amplitude to create SNR variation
            amplitude_factor = 0.5 + (i % 4) * 0.3  # 0.5 to 1.4
            strain_h1 = torch.randn(1, 16384) * amplitude_factor
            strain_l1 = torch.randn(1, 16384) * amplitude_factor
            strain_v1 = torch.randn(1, 16384) * amplitude_factor
            strain_data_list.append(torch.stack([strain_h1, strain_l1, strain_v1], dim=1))
        
        strain_data = torch.cat(strain_data_list, dim=0)
        true_params = torch.randn(batch_size, 1, 10)
        
        # Run forward pass
        # (SNR and log_scale are accumulated INSIDE compute_loss automatically)
        try:
            loss_dict = model.compute_loss(strain_data, true_params)
            # loss can be a dict or tensor depending on marginalization
            if isinstance(loss_dict, dict):
                total_loss = loss_dict.get('total_loss', loss_dict.get('loss', 0))
                if hasattr(total_loss, 'item'):
                    loss_val = total_loss.item()
                else:
                    loss_val = float(total_loss)
            else:
                loss_val = loss_dict.item()
            
            print(f"Batch {batch_idx}: loss={loss_val:.4f}")
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
    
    # Compute correlation
    print(f"\n--- Computing Correlation ---")
    snr_values = np.array(model._epoch_distance_data['snr'], dtype=np.float32)
    log_scale_values = np.array(model._epoch_distance_data['log_scale'], dtype=np.float32)
    
    print(f"\nSNR values: {len(snr_values)} samples")
    print(f"  Mean: {snr_values.mean():.6f}")
    print(f"  Std: {snr_values.std():.6f}")
    print(f"  Min/Max: {snr_values.min():.6f} / {snr_values.max():.6f}")
    
    print(f"\nLog-Distance-Scale values: {len(log_scale_values)} samples")
    print(f"  Mean: {log_scale_values.mean():.6f}")
    print(f"  Std: {log_scale_values.std():.6f}")
    print(f"  Min/Max: {log_scale_values.min():.6f} / {log_scale_values.max():.6f}")
    
    # Compute correlation
    if len(snr_values) > 10 and snr_values.std() > 1e-6 and log_scale_values.std() > 1e-6:
        corr_matrix = np.corrcoef(log_scale_values, snr_values)
        correlation = corr_matrix[0, 1]
        
        print(f"\n--- CORRELATION RESULT ---")
        print(f"Correlation(log_distance_scale, SNR) = {correlation:.6f}")
        print(f"(Expected: negative value around -0.3 to -0.7)")
        
        if np.isfinite(correlation):
            print(f"✅ VALID correlation (finite)")
        else:
            print(f"❌ INVALID correlation (NaN or Inf)")
    else:
        print(f"\n--- CANNOT COMPUTE CORRELATION ---")
        print(f"SNR std: {snr_values.std():.2e} (need > 1e-6)")
        print(f"Log-scale std: {log_scale_values.std():.2e} (need > 1e-6)")
        print(f"Samples: {len(snr_values)} (need > 10)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    compute_and_display_correlation()
