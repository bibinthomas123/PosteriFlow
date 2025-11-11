#!/usr/bin/env python3
"""
Test extreme case variance analysis fix.

This validates that the z-score-based status indicators correctly
identify normal statistical variance vs. actual generation errors.
"""

import numpy as np
from scipy.stats import norm

def calculate_z_score(count, expected_count, total_extreme, type_fraction):
    """Calculate z-score for an extreme case type."""
    if total_extreme <= 1 or type_fraction <= 0:
        return 0.0
    
    # Standard error for multinomial distribution
    std_error = np.sqrt(
        total_extreme * type_fraction * (1 - type_fraction)
    )
    
    z_score = (count - expected_count) / std_error if std_error > 0 else 0
    return z_score


def get_status(z_score):
    """Get status based on z-score."""
    if abs(z_score) <= 1.0:
        return "✓✓"  # Within 1σ (68%)
    elif abs(z_score) <= 2.0:
        return "✓"   # Within 2σ (95%)
    else:
        return "⚠"   # Outside 2σ (rare, p < 0.05)


# Test data from the analysis
n_total = 5000
extreme_fraction = 0.03
n_extreme = int(n_total * extreme_fraction)  # 150

types = {
    "near_simultaneous_mergers": 0.25,
    "extreme_mass_ratio": 0.15,
    "high_spin_aligned": 0.15,
    "weak_strong_overlaps": 0.25,
    "noise_confused_overlaps": 0.15,
    "long_duration_bns_overlaps": 0.05,
}

actual = {
    "extreme_mass_ratio": 24,
    "high_spin_aligned": 24,
    "long_duration_bns_overlaps": 11,
    "near_simultaneous_mergers": 32,
    "noise_confused_overlaps": 19,
    "weak_strong_overlaps": 28,
}

print("=" * 90)
print("EXTREME CASE VARIANCE FIX VALIDATION")
print("=" * 90)
print()
print(f"Total samples: {n_total}")
print(f"Extreme fraction: {extreme_fraction * 100:.1f}%")
print(f"Total extreme samples: {n_extreme}")
print()
print(f"{'Type':<35} {'Expected':>8} {'Actual':>8} {'Z-score':>10} {'Status':>8}")
print("─" * 90)

total_chi2 = 0
all_within_2sigma = True

for name in sorted(types.keys()):
    type_frac = types[name]
    expected = n_extreme * type_frac
    actual_count = actual[name]
    
    z_score = calculate_z_score(actual_count, expected, n_extreme, type_frac)
    status = get_status(z_score)
    
    # Track chi-square contribution
    chi2_contrib = (actual_count - expected) ** 2 / expected if expected > 0 else 0
    total_chi2 += chi2_contrib
    
    if abs(z_score) > 2.0:
        all_within_2sigma = False
    
    print(
        f"{name:<35} {expected:>8.1f} {actual_count:>8} {z_score:>+10.2f} {status:>8}"
    )

print("─" * 90)
print()

# Chi-square test
from scipy.stats import chi2
df = len(types) - 1
p_value = 1 - chi2.cdf(total_chi2, df)

print(f"Chi-square goodness of fit:")
print(f"  χ² = {total_chi2:.2f}")
print(f"  df = {df}")
print(f"  p-value = {p_value:.4f}")
print()

print("VERDICT:")
if p_value > 0.05 and all_within_2sigma:
    print("  ✓✓ PASS - All deviations within 2σ (p > 0.05)")
    print("  Status: No generation errors detected, variance is normal")
elif all_within_2sigma:
    print("  ✓ PASS - All individual deviations within 2σ")
    print("  Status: Normal statistical variance")
else:
    print("  ⚠ REVIEW - Some deviations exceed 2σ")
    print("  Status: Potential systematic bias or generation error")

print()
print("=" * 90)
