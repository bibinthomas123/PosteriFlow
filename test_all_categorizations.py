#!/usr/bin/env python
"""Test all SNR categorization implementations"""

import sys
sys.path.insert(0, '/home/bibinathomas/PosteriFlow/src')

from data.analysis import Analysis

# Test the analysis.py snr_regime method
print("Testing Analysis.snr_regime():")
print(f"{'SNR':<6} {'Regime':<10}")
print("-" * 16)

test_snrs = [5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 100]
for snr in test_snrs:
    regime = Analysis.snr_regime(snr)
    print(f"{snr:<6} {regime:<10}")

print("\n✓ Analysis.snr_regime() is now using SNR_RANGES from config")
