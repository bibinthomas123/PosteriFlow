#!/usr/bin/env python3
"""Test quota mode SNR enforcement"""
import sys
sys.path.insert(0, '/home/bibinathomas/PosteriFlow')

from src.ahsd.data.config import SNR_DISTRIBUTION
import numpy as np
from collections import Counter

print("=" * 80)
print("QUOTA MODE DIAGNOSTIC TEST")
print("=" * 80)

print("\n1. SNR_DISTRIBUTION from config:")
for regime, prob in SNR_DISTRIBUTION.items():
    print(f"   {regime:8s}: {prob:5.2f} ({prob*100:5.1f}%)")

# Simulate quota computation like in dataset_generator.py
print("\n2. Simulating quota computation for 100 samples (with overlaps):")
n_samples = 100
n_regular_single = int(0.3 * n_samples)
n_regular_overlap = int(0.5 * n_samples)  
n_edge = int(0.2 * n_samples)

print(f"   n_samples: {n_samples}")
print(f"   n_regular_single: {n_regular_single}")
print(f"   n_regular_overlap: {n_regular_overlap}")
print(f"   n_edge_cases: {n_edge}")

# Estimate signals
expected_signals_per_overlap = 3.8
total_signals_est = int(round(
    n_regular_single
    + n_regular_overlap * expected_signals_per_overlap
    + n_edge  # include_extremes = true
))

print(f"   expected_signals_per_overlap: {expected_signals_per_overlap}")
print(f"   total_signals_est: {total_signals_est}")

# Compute quotas
quotas_snr = {}
regimes = list(SNR_DISTRIBUTION.keys())
for r in regimes:
    quotas_snr[r] = int(round(total_signals_est * float(SNR_DISTRIBUTION.get(r, 0.0))))

# Balance rounding
rem = total_signals_est - sum(quotas_snr.values())
idx = 0
while rem > 0:
    quotas_snr[regimes[idx % len(regimes)]] += 1
    idx += 1
    rem -= 1

print(f"\n3. Computed SNR quotas:")
for regime, quota in quotas_snr.items():
    expected_pct = SNR_DISTRIBUTION[regime] * 100
    actual_pct = quota / total_signals_est * 100
    print(f"   {regime:8s}: {quota:4d} ({actual_pct:5.1f}%) expected {expected_pct:5.1f}%")

total_quota = sum(quotas_snr.values())
print(f"   Total: {total_quota}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
if abs(total_quota - total_signals_est) < 2:
    print("✓ Quota computation is correct")
else:
    print(f"✗ Quota computation mismatch: {total_quota} vs {total_signals_est}")

# Check if quotas match config distribution
all_good = True
for regime, quota in quotas_snr.items():
    expected_pct = SNR_DISTRIBUTION[regime] * 100
    actual_pct = quota / total_signals_est * 100
    if abs(actual_pct - expected_pct) > 1:
        all_good = False
        print(f"✗ {regime}: {actual_pct:.1f}% vs expected {expected_pct:.1f}%")

if all_good:
    print("✓ All quotas match config distribution")
else:
    print("Note: Small differences are expected due to rounding")
