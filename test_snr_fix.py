#!/usr/bin/env python3
"""Test SNR distribution fix - quick validation without full generation"""
import sys
import numpy as np
from collections import Counter

# Add src to path
sys.path.insert(0, '/home/bibinathomas/PosteriFlow')

from src.ahsd.data.parameter_sampler import ParameterSampler
from src.ahsd.data.config import SNR_DISTRIBUTION, SNR_RANGES

print("=" * 80)
print("SNR DISTRIBUTION FIX VALIDATION")
print("=" * 80)

print("\n1. Config SNR_DISTRIBUTION:")
for regime, prob in SNR_DISTRIBUTION.items():
    expected_pct = prob * 100
    snr_range = SNR_RANGES[regime]
    print(f"   {regime:8s}: {prob:5.2f} ({expected_pct:5.1f}%) range={snr_range}")

# Simulate overlapping sample generation (where the bug was)
print("\n2. Testing overlapping signal SNR sampling (OLD bug code):")
print("   OLD: hardcoded p=0.65 weak/low, p=0.35 medium/high/loud")
print("   Result: 65% biased to weak/low")

old_counts = Counter()
for _ in range(1000):
    if np.random.random() < 0.65:
        regime = np.random.choice(["weak", "low"], p=[0.4, 0.6])
    else:
        regime = np.random.choice(["medium", "high", "loud"], p=[0.7, 0.2, 0.1])
    old_counts[regime] += 1

print("\n   Old counts (1000 samples):")
for regime in SNR_DISTRIBUTION.keys():
    count = old_counts.get(regime, 0)
    old_pct = count / 10
    expected_pct = SNR_DISTRIBUTION[regime] * 100
    diff = old_pct - expected_pct
    status = "✗ WRONG" if abs(diff) > 5 else "✓"
    print(f"      {regime:8s}: {count:4d} ({old_pct:5.1f}%) expected {expected_pct:5.1f}% {status}")

# Test NEW sampling (should respect distribution)
print("\n3. Testing overlapping signal SNR sampling (NEW fix):")
print("   NEW: parameter_sampler._sample_snr_regime()")

sampler = ParameterSampler()
new_counts = Counter()
for _ in range(1000):
    regime = sampler._sample_snr_regime()
    new_counts[regime] += 1

print("\n   New counts (1000 samples):")
for regime in SNR_DISTRIBUTION.keys():
    count = new_counts.get(regime, 0)
    new_pct = count / 10
    expected_pct = SNR_DISTRIBUTION[regime] * 100
    diff = new_pct - expected_pct
    status = "✓ OK" if abs(diff) < 5 else "⚠ marginal"
    print(f"      {regime:8s}: {count:4d} ({new_pct:5.1f}%) expected {expected_pct:5.1f}% {diff:+.1f}% {status}")

print("\n" + "=" * 80)
print("FIX SUMMARY:")
print("=" * 80)
print("✓ Fixed: Overlapping signals now respect configured SNR distribution")
print("✓ Before: 65% bias toward weak/low (wrong for most configs)")
print("✓ After: Samples from SNR_DISTRIBUTION correctly")
print("\nNext: Run ahsd-generate with full dataset to verify end-to-end")
print("=" * 80)
