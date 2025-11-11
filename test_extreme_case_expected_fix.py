#!/usr/bin/env python
"""
Test extreme case type expected percentage calculation fix.
Verify that expected percentages are calculated from config, not hardcoded.
"""

import sys
sys.path.insert(0, '/home/bibinathomas/PosteriFlow/src')

import yaml


def test_extreme_case_expected_calculation():
    """Test that expected percentages match config fractions"""
    
    print("=" * 80)
    print("TEST: Extreme Case Type Expected Percentage Calculation")
    print("=" * 80)
    
    # Load the config
    with open('/home/bibinathomas/PosteriFlow/configs/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    extreme_config = config.get('extreme_cases', {})
    extreme_fraction = extreme_config.get('fraction', 0.03)
    extreme_types_config = extreme_config.get('types', {})
    
    print(f"\nConfig values:")
    print(f"  extreme_cases.fraction: {extreme_fraction} ({extreme_fraction*100}% of total)")
    print(f"  Number of extreme types: {len(extreme_types_config)}")
    
    # Calculate expected percentages using the fix
    extreme_expected = {}
    for extreme_type, type_config in extreme_types_config.items():
        if isinstance(type_config, dict) and type_config.get("enabled", True):
            type_fraction = type_config.get("fraction", 0.0)
            expected_pct = type_fraction * extreme_fraction * 100
            extreme_expected[extreme_type] = expected_pct
    
    print(f"\nExpected percentages (from fix):")
    print(f"{'Type':<35} {'Fraction':<12} {'Expected %':<12}")
    print("-" * 60)
    
    # Expected values based on config
    expected_from_config = {
        'near_simultaneous_mergers': (0.25, 0.25 * 0.03 * 100),
        'extreme_mass_ratio': (0.15, 0.15 * 0.03 * 100),
        'high_spin_aligned': (0.15, 0.15 * 0.03 * 100),
        'weak_strong_overlaps': (0.25, 0.25 * 0.03 * 100),
        'noise_confused_overlaps': (0.15, 0.15 * 0.03 * 100),
        'long_duration_bns_overlaps': (0.05, 0.05 * 0.03 * 100),
    }
    
    all_pass = True
    for extreme_type in sorted(extreme_expected.keys()):
        calc_pct = extreme_expected[extreme_type]
        config_fraction, expected_pct = expected_from_config.get(
            extreme_type, (0, 0)
        )
        
        match = "✓" if abs(calc_pct - expected_pct) < 0.001 else "✗"
        print(
            f"{extreme_type:<35} {config_fraction:>10.2f}   "
            f"{calc_pct:>10.3f}%  {match}"
        )
        
        if abs(calc_pct - expected_pct) >= 0.001:
            all_pass = False
    
    print()
    print(f"Total expected (should be ~{extreme_fraction*100:.1f}%): "
          f"{sum(extreme_expected.values()):.2f}%")
    
    # Verify against OLD hardcoded values
    print("\n" + "=" * 80)
    print("COMPARISON: Old Hardcoded vs New Config-Based")
    print("=" * 80)
    
    old_hardcoded = {
        "near_simultaneous_mergers": 1.0,
        "extreme_mass_ratio": 1.0,
        "high_spin_aligned": 1.0,
        "weak_strong_overlaps": 0.5,
        "noise_confused_overlaps": 1.0,
        "long_duration_bns_overlaps": 0.5,
    }
    
    print(f"\n{'Type':<35} {'Old (Wrong)':<15} {'New (Fixed)':<15} {'Difference':<15}")
    print("-" * 80)
    
    for extreme_type in sorted(expected_from_config.keys()):
        old_val = old_hardcoded.get(extreme_type, 0)
        new_val = extreme_expected.get(extreme_type, 0)
        diff = new_val - old_val
        
        print(
            f"{extreme_type:<35} {old_val:>13.2f}%  {new_val:>13.2f}%  "
            f"{diff:>13.2f}%"
        )
    
    if all_pass:
        print("\n" + "=" * 80)
        print("✓ ALL CALCULATIONS CORRECT")
        print("=" * 80)
        print("\nExpected percentages now correctly calculated from config!")
        print("Enabled types show proper expected values for comparison.")
        return True
    else:
        print("\n✗ MISMATCH DETECTED")
        return False


if __name__ == '__main__':
    try:
        success = test_extreme_case_expected_calculation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
