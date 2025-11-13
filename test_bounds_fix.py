#!/usr/bin/env python
"""
Quick test to verify geocent_time and luminosity_distance bounds fix.
"""

import numpy as np
from src.ahsd.models.overlap_neuralpe import OverlapNeuralPE

print("=" * 80)
print("PARAMETER BOUNDS FIX VERIFICATION (Nov 13, 2025)")
print("=" * 80)

# Create a mock model to test bounds
config = {'enable_event_specific_priors': True}

# Create instance
class MockModel:
    pass

model = OverlapNeuralPE.__new__(OverlapNeuralPE)
model.param_names = [
    'mass_1', 'mass_2', 'luminosity_distance', 'geocent_time',
    'ra', 'dec', 'theta_jn', 'psi', 'phase'
]

# Get bounds using the actual method
bounds = model._get_parameter_bounds()

print("\n✓ FIXED BOUNDS:")
print(f"  geocent_time:         {bounds['geocent_time']}")
print(f"  luminosity_distance:  {bounds['luminosity_distance']}")

print("\n✓ ACTUAL DATA RANGES (from diagnostic):")
actual_ranges = {
    'geocent_time': (-1.77, 6.63),  # 99th percentile: 6.05s
    'luminosity_distance': (15.9, 1170),
    'mass_1': (1.2, 73.3),
    'mass_2': (1.0, 60.8),
    'ra': (0.017, 6.27),
    'dec': (-1.33, 1.41),
    'theta_jn': (0.17, 3.11),
    'psi': (0.03, 3.12),
    'phase': (0.03, 6.24),
}

print("\n✓ VERIFICATION - All data within bounds:")
all_pass = True
for param in bounds:
    min_bound, max_bound = bounds[param]
    actual_min, actual_max = actual_ranges.get(param, (0, 0))
    
    # Check if actual data is within bounds
    within = actual_min >= min_bound and actual_max <= max_bound
    status = "✓" if within else "✗"
    
    if not within:
        all_pass = False
    
    print(f"  {status} {param:20} [{actual_min:8.2f}, {actual_max:8.2f}] ⊆ "
          f"[{min_bound:8.2f}, {max_bound:8.2f}]")

print("\n" + "=" * 80)
if all_pass:
    print("✓ ALL PARAMETERS WITHIN BOUNDS - FIX VERIFIED")
else:
    print("✗ SOME PARAMETERS OUT OF BOUNDS - FIX INCOMPLETE")
print("=" * 80)
