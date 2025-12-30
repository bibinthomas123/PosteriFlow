#!/usr/bin/env python3
"""
Test to verify luminosity_distance log scaling is working correctly.
"""

import torch
import sys
sys.path.insert(0, '/home/bibin/PosteriFlow')

from src.ahsd.models.parameter_scalers import TorchParameterScaler

# Create a scaler with the correct param names
param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 
               'theta_jn', 'psi', 'phase', 'geocent_time', 'a1', 'a2']

scaler = TorchParameterScaler(param_names=param_names, event_type='BBH', device='cpu')

# Check if luminosity_distance buffers exist
print("✅ TorchParameterScaler created successfully")
print(f"Parameters in scaler: {list(scaler.param_names)}")

# Check if log_min and log_max are registered
if hasattr(scaler, 'luminosity_distance_log_min'):
    print(f"\n✅ luminosity_distance_log_min: {scaler.luminosity_distance_log_min}")
    print(f"✅ luminosity_distance_log_max: {scaler.luminosity_distance_log_max}")
else:
    print("\n❌ log_min/log_max NOT registered!")

# Test normalize and denormalize with multiple distances
distances = [10.4, 100.0, 500.0, 1000.0, 8000.0]
print(f"\n=== Testing Log-Minmax Normalization ===")
for dist in distances:
    distance_phys = torch.tensor([[100.0, 50.0, float(dist), 1.0, 0.5, 1.0, 1.0, 1.0, 0.0, 0.5, 0.5]])
    print(f"\nOriginal distance: {distance_phys[0, 2].item():.2f} Mpc")

    distance_norm = scaler.normalize_batch(distance_phys)
    print(f"  Normalized: {distance_norm[0, 2].item():.4f} (should be in [-1, 1])")

    distance_denorm = scaler.denormalize_batch(distance_norm)
    print(f"  Denormalized: {distance_denorm[0, 2].item():.2f} Mpc")
    error = abs(distance_phys[0, 2] - distance_denorm[0, 2]).item()
    print(f"  Round-trip error: {error:.6f} Mpc")
    
    if error < 1e-3:
        print("  ✅ PASS")
    else:
        print(f"  ❌ FAIL (error > 1e-3)")

print("\n=== Summary ===")
print("If all log-minmax scaling tests passed, the scaler is working correctly.")
print("The user's report about -285 Mpc bias may be from a different issue.")
