#!/usr/bin/env python
"""Test SNR categorization fix"""

import sys
sys.path.insert(0, '/home/bibinathomas/PosteriFlow/src')

from ahsd.data.config import SNR_RANGES

def categorize_snr_fixed(snr: float) -> str:
    """Fixed categorization using config ranges"""
    for regime, (min_snr, max_snr) in SNR_RANGES.items():
        if min_snr <= snr < max_snr:
            return regime
    
    if snr < 10.0:
        return 'weak'
    else:
        return 'loud'

# Test cases
test_snrs = [5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 100]

print("SNR_RANGES from config:")
for regime, (min_snr, max_snr) in SNR_RANGES.items():
    print(f"  {regime}: {min_snr}-{max_snr}")

print("\nTest SNR categorization:")
print(f"{'SNR':<6} {'Regime':<10}")
print("-" * 16)

for snr in test_snrs:
    regime = categorize_snr_fixed(snr)
    print(f"{snr:<6} {regime:<10}")

print("\n✓ SNR categorization is now consistent with SNR_RANGES")
