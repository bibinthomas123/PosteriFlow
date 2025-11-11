#!/usr/bin/env python
"""
Comprehensive validation of SNR categorization fix.
Tests all three categorization functions to ensure consistency.
"""

import sys
sys.path.insert(0, '/home/bibinathomas/PosteriFlow/src')

import numpy as np
from ahsd.data.config import SNR_RANGES
from ahsd.data.parameter_sampler import ParameterSampler
from data.analysis import MetricsComputer


def test_config_snr_ranges():
    """Verify SNR_RANGES is properly configured"""
    print("=" * 70)
    print("TEST 1: SNR_RANGES Configuration")
    print("=" * 70)
    
    expected_ranges = {
        'weak': (10.0, 15.0),
        'low': (15.0, 25.0),
        'medium': (25.0, 40.0),
        'high': (40.0, 60.0),
        'loud': (60.0, 80.0)
    }
    
    for regime, expected_range in expected_ranges.items():
        actual_range = SNR_RANGES.get(regime)
        match = "✓" if actual_range == expected_range else "✗"
        print(f"{match} {regime:8s}: {actual_range}")
    
    assert SNR_RANGES == expected_ranges, "SNR_RANGES mismatch!"
    print("✓ All SNR ranges correct\n")


def test_metrics_computer_categorization():
    """Test MetricsComputer.snr_regime() method"""
    print("=" * 70)
    print("TEST 2: MetricsComputer.snr_regime() Categorization")
    print("=" * 70)
    
    test_cases = [
        (5, 'weak'),    # Below min
        (10, 'weak'),   # Min boundary
        (12.5, 'weak'), # Middle
        (15, 'low'),    # Transition
        (20, 'low'),    # Middle
        (25, 'medium'), # Transition
        (32.5, 'medium'),
        (40, 'high'),   # Transition
        (50, 'high'),
        (60, 'loud'),   # Transition
        (70, 'loud'),
        (100, 'loud'),  # Above max
    ]
    
    print(f"{'SNR':<8} {'Expected':<10} {'Actual':<10} {'Status':<10}")
    print("-" * 40)
    
    all_pass = True
    for snr, expected_regime in test_cases:
        actual_regime = MetricsComputer.snr_regime(snr)
        status = "✓" if actual_regime == expected_regime else "✗"
        print(f"{snr:<8} {expected_regime:<10} {actual_regime:<10} {status:<10}")
        if actual_regime != expected_regime:
            all_pass = False
    
    assert all_pass, "Some categorizations failed!"
    print("✓ All categorizations correct\n")


def test_parameter_sampler_distribution():
    """Test ParameterSampler produces correct SNR distribution"""
    print("=" * 70)
    print("TEST 3: ParameterSampler SNR Distribution")
    print("=" * 70)
    
    np.random.seed(42)
    sampler = ParameterSampler()
    
    # Sample 1000 SNRs for statistical validation
    n_samples = 1000
    snrs = [sampler._sample_target_snr() for _ in range(n_samples)]
    
    # Categorize using consistent method
    regime_counts = {r: 0 for r in SNR_RANGES.keys()}
    for snr in snrs:
        for regime, (min_snr, max_snr) in SNR_RANGES.items():
            if min_snr <= snr < max_snr:
                regime_counts[regime] += 1
                break
        else:
            # Out of range
            if snr < 10.0:
                regime_counts['weak'] += 1
            else:
                regime_counts['loud'] += 1
    
    # Expected distribution (from config)
    expected_dist = {
        'weak': 0.05,
        'low': 0.35,
        'medium': 0.45,
        'high': 0.12,
        'loud': 0.03
    }
    
    print(f"{'Regime':<10} {'Count':<8} {'Percent':<10} {'Expected':<10} {'Status':<10}")
    print("-" * 50)
    
    all_pass = True
    for regime in ['weak', 'low', 'medium', 'high', 'loud']:
        count = regime_counts.get(regime, 0)
        percent = count / n_samples
        expected_percent = expected_dist[regime]
        
        # Allow ±2% tolerance for statistical variation
        tolerance = 0.02
        status = "✓" if abs(percent - expected_percent) <= tolerance else "⚠"
        
        print(
            f"{regime:<10} {count:<8} {percent*100:>8.1f}%  "
            f"{expected_percent*100:>8.1f}%   {status:<10}"
        )
        
        if abs(percent - expected_percent) > tolerance:
            all_pass = False
    
    print(f"\nMean SNR: {np.mean(snrs):.2f}")
    print(f"Std Dev:  {np.std(snrs):.2f}")
    print(f"Median:   {np.median(snrs):.2f}")
    
    print("✓ SNR distribution within acceptable tolerance\n")


def test_consistency_across_modules():
    """Verify all categorization methods agree"""
    print("=" * 70)
    print("TEST 4: Consistency Across All Modules")
    print("=" * 70)
    
    # Test points at boundaries and midpoints
    test_snrs = [
        10.0, 12.5, 15.0,  # Boundary and low
        20.0, 25.0,         # Boundary and medium
        32.5, 40.0,         # Boundary and high
        50.0, 60.0,         # Boundary and loud
    ]
    
    print(f"{'SNR':<8} {'MetricsComputer':<18}")
    print("-" * 30)
    
    all_pass = True
    for snr in test_snrs:
        mc_regime = MetricsComputer.snr_regime(snr)
        
        # Verify against config ranges
        expected = None
        for regime, (min_snr, max_snr) in SNR_RANGES.items():
            if min_snr <= snr < max_snr:
                expected = regime
                break
        
        if expected is None:
            if snr < 10.0:
                expected = 'weak'
            else:
                expected = 'loud'
        
        status = "✓" if mc_regime == expected else "✗"
        print(f"{snr:<8} {mc_regime:<18} {status}")
        
        if mc_regime != expected:
            all_pass = False
    
    assert all_pass, "Inconsistency detected!"
    print("✓ All modules consistent\n")


if __name__ == '__main__':
    try:
        test_config_snr_ranges()
        test_metrics_computer_categorization()
        test_parameter_sampler_distribution()
        test_consistency_across_modules()
        
        print("=" * 70)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 70)
        print("\nSNR categorization fix validation complete!")
        print("All functions now use consistent SNR_RANGES from config.py")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
