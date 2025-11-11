#!/usr/bin/env python3
"""
Test the improved PSD validation with log-space analysis.
"""

import numpy as np

# Simulate our noise PSD (from debug output)
psd_range = np.random.uniform(1e-45, 1e-44, size=100)

print("=" * 80)
print("PSD VALIDATION TEST")
print("=" * 80)

# OLD method (broken for tiny values)
psd_std_old = np.std(psd_range)
psd_mean_old = np.mean(psd_range)
ratio_old = psd_std_old / psd_mean_old if psd_mean_old > 0 else 0

print(f"\n1. OLD LINEAR METHOD:")
print(f"   PSD mean: {psd_mean_old:.2e}")
print(f"   PSD std:  {psd_std_old:.2e}")
print(f"   Ratio (std/mean): {ratio_old:.6f}")
print(f"   Threshold check (ratio < 0.01): {ratio_old < 0.01}")
print(f"   Result: {'FAIL - PSD too uniform' if ratio_old < 0.01 else 'PASS'}")

# NEW method (robust log-space)
log_psd = np.log10(np.maximum(psd_range, 1e-50))
log_std = np.std(log_psd)
log_mean = np.mean(log_psd)

print(f"\n2. NEW LOG-SPACE METHOD:")
print(f"   Log PSD mean: {log_mean:.4f}")
print(f"   Log PSD std:  {log_std:.4f}")
print(f"   Threshold check (log_std < 0.05): {log_std < 0.05}")
print(f"   Result: {'FAIL - PSD too uniform' if log_std < 0.05 else 'PASS'}")

# Show that with proper coloring, log_std will be higher
print(f"\n3. WITH FREQUENCY-DEPENDENT COLORING:")
# Simulate PSD with 10x variation across frequencies
freqs = np.arange(50, 2000, step=5)
# Power-law spectrum
psd_colored = 1e-44 * (freqs / 100.0)**(-0.5)
psd_colored += np.random.normal(0, psd_colored * 0.1)  # Add some noise

log_psd_colored = np.log10(np.maximum(psd_colored, 1e-50))
log_std_colored = np.std(log_psd_colored)
psd_std_colored = np.std(psd_colored)
psd_mean_colored = np.mean(psd_colored)
ratio_colored = psd_std_colored / psd_mean_colored

print(f"   PSD std:  {psd_std_colored:.2e}")
print(f"   PSD mean: {psd_mean_colored:.2e}")
print(f"   Linear ratio: {ratio_colored:.6f}")
print(f"   Log std:  {log_std_colored:.4f}")
print(f"   Threshold check (log_std < 0.05): {log_std_colored < 0.05}")
print(f"   Result: {'FAIL - PSD too uniform' if log_std_colored < 0.05 else 'PASS'}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("Log-space analysis is more robust for detecting frequency dependence")
print("in tiny PSD values where linear std can suffer from underflow/precision issues.")
print("=" * 80)
