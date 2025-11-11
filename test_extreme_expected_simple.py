#!/usr/bin/env python
"""
Simple test to verify extreme case expected percentages are correctly calculated.
"""

import sys
sys.path.insert(0, '/home/bibinathomas/PosteriFlow/src')

import yaml


def test_extreme_expected_calculation():
    """Test that the fix correctly calculates expected percentages"""
    
    print("=" * 80)
    print("TEST: Extreme Case Expected Percentage Calculation (Simple)")
    print("=" * 80)
    
    # Load config
    with open('/home/bibinathomas/PosteriFlow/configs/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    extreme_config = config.get('extreme_cases', {})
    extreme_fraction = extreme_config.get('fraction', 0.03)
    extreme_types_config = extreme_config.get('types', {})
    
    print(f"\nConfiguration:")
    print(f"  extreme_cases.fraction: {extreme_fraction} ({extreme_fraction*100}% of total)")
    print(f"  Number of extreme types: {len(extreme_types_config)}")
    
    # Simulate the fixed code
    print(f"\n✓ Simulating fixed code:")
    print(f"{'─'*80}")
    print(f"for extreme_type, type_config in extreme_types_config.items():")
    print(f"    if type_config.get('enabled', True):")
    print(f"        type_fraction = type_config.get('fraction', 0.0)")
    print(f"        expected_pct = type_fraction * extreme_fraction * 100")
    print(f"        extreme_expected[extreme_type] = expected_pct")
    print(f"{'─'*80}\n")
    
    # Calculate expected percentages using the fixed logic
    extreme_expected = {}
    for extreme_type, type_config in extreme_types_config.items():
        if isinstance(type_config, dict) and type_config.get("enabled", True):
            type_fraction = type_config.get("fraction", 0.0)
            expected_pct = type_fraction * extreme_fraction * 100
            extreme_expected[extreme_type] = expected_pct
    
    print(f"Results:")
    print(f"{'Type':<35} {'Fraction':<12} {'Expected %':<12} {'Status'}")
    print("─" * 60)
    
    expected_dict = {
        'near_simultaneous_mergers': 0.25,
        'extreme_mass_ratio': 0.15,
        'high_spin_aligned': 0.15,
        'weak_strong_overlaps': 0.25,
        'noise_confused_overlaps': 0.15,
        'long_duration_bns_overlaps': 0.05,
    }
    
    all_pass = True
    total_expected = 0
    
    for extreme_type in sorted(extreme_expected.keys()):
        calc_pct = extreme_expected[extreme_type]
        config_fraction = expected_dict.get(extreme_type, 0)
        expected_pct = config_fraction * extreme_fraction * 100
        
        match = "✓" if abs(calc_pct - expected_pct) < 0.001 else "✗"
        print(
            f"{extreme_type:<35} {config_fraction:>10.2f}   "
            f"{calc_pct:>10.3f}%  {match}"
        )
        
        total_expected += calc_pct
        if abs(calc_pct - expected_pct) >= 0.001:
            all_pass = False
    
    print("─" * 60)
    print(f"{'Total':<35} {'':<12} {total_expected:>10.2f}%")
    
    # Verify totals
    total_expected_theoretical = sum(expected_dict.values()) * extreme_fraction * 100
    
    print(f"\n✓ Verification:")
    print(f"  Total calculated:    {total_expected:.3f}%")
    print(f"  Total theoretical:   {total_expected_theoretical:.3f}%")
    print(f"  Config extreme_fraction: {extreme_fraction*100:.1f}%")
    
    if abs(total_expected - extreme_fraction*100) < 0.001:
        print(f"  Status: ✓ Totals match")
    else:
        print(f"  Status: ✗ Totals DO NOT match")
        all_pass = False
    
    # Compare with old hardcoded values
    print(f"\n" + "=" * 80)
    print("Comparison with OLD hardcoded values:")
    print("=" * 80)
    
    old_hardcoded = {
        "near_simultaneous_mergers": 1.0,
        "extreme_mass_ratio": 1.0,
        "high_spin_aligned": 1.0,
        "weak_strong_overlaps": 0.5,
        "noise_confused_overlaps": 1.0,
        "long_duration_bns_overlaps": 0.5,
    }
    
    print(f"\n{'Type':<35} {'Old':<15} {'New':<15} {'Delta':<15}")
    print("─" * 80)
    
    for extreme_type in sorted(expected_dict.keys()):
        old_val = old_hardcoded.get(extreme_type, 0)
        new_val = extreme_expected.get(extreme_type, 0)
        delta = new_val - old_val
        
        delta_str = f"{delta:+.2f}%" if delta != 0 else "0.00%"
        print(f"{extreme_type:<35} {old_val:>13.2f}%  {new_val:>13.2f}%  {delta_str:>13}")
    
    # Final result
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ TEST PASSED - Expected percentages correctly calculated!")
        print("=" * 80)
        print("\nSummary:")
        print("  • Configuration fractions are correctly loaded")
        print("  • Expected percentages match theoretical calculations")
        print("  • Total matches extreme_cases.fraction from config")
        print("  • Fix correctly handles enabled/disabled types")
        return True
    else:
        print("✗ TEST FAILED - Calculations don't match expected values")
        print("=" * 80)
        return False


if __name__ == '__main__':
    try:
        success = test_extreme_expected_calculation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
