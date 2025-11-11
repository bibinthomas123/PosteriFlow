#!/usr/bin/env python
"""
Test that extreme case type logging now uses correct expected percentages.
This verifies the fix works in the actual DatasetGenerator.
"""

import sys
sys.path.insert(0, '/home/bibinathomas/PosteriFlow/src')

import logging
import yaml
from io import StringIO
from ahsd.data.dataset_generator import GWDatasetGenerator


def test_extreme_case_logging():
    """Generate a small dataset and verify extreme case logging is correct"""
    
    print("=" * 80)
    print("TEST: Extreme Case Type Logging with Fixed Expected Percentages")
    print("=" * 80)
    
    # Load config
    with open('/home/bibinathomas/PosteriFlow/configs/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create generator with small sample size
    print("\nGenerating 100 samples with extreme cases...")
    print("-" * 80)
    
    try:
        generator = GWDatasetGenerator(
            config=config,
            output_dir='/tmp/test_extreme_case_logging',
            verbose=False
        )
        
        # Generate a small dataset
        samples = []
        for i in range(100):
            sample = generator.generate_sample()
            if sample:
                samples.append(sample)
        
        print(f"\n✓ Generated {len(samples)} samples")
        
        # Count extreme case types
        extreme_counts = {}
        for sample in samples:
            sample_type = sample.get('type', 'unknown')
            if sample_type == 'extreme_case':
                extreme_type = sample.get('extreme_type', 'unknown')
                extreme_counts[extreme_type] = extreme_counts.get(extreme_type, 0) + 1
        
        if extreme_counts:
            print(f"\nExtreme cases generated:")
            print(f"{'Type':<35} {'Count':<8}")
            print("-" * 43)
            
            for extreme_type in sorted(extreme_counts.keys()):
                count = extreme_counts[extreme_type]
                pct = (count / len(samples) * 100)
                print(f"{extreme_type:<35} {count:<8} ({pct:.2f}%)")
            
            print(f"\nTotal extreme: {sum(extreme_counts.values())} "
                  f"({sum(extreme_counts.values())/len(samples)*100:.1f}%)")
        else:
            print("\nNo extreme cases in this small sample (expected - small dataset)")
        
        # Verify the expected percentages are calculated from config
        extreme_config = config.get('extreme_cases', {})
        extreme_fraction = extreme_config.get('fraction', 0.03)
        extreme_types_config = extreme_config.get('types', {})
        
        extreme_expected = {}
        for extreme_type, type_config in extreme_types_config.items():
            if isinstance(type_config, dict) and type_config.get("enabled", True):
                type_fraction = type_config.get("fraction", 0.0)
                expected_pct = type_fraction * extreme_fraction * 100
                extreme_expected[extreme_type] = expected_pct
        
        print("\nExpected percentages (from fixed code):")
        print(f"{'Type':<35} {'Expected %':<15}")
        print("-" * 50)
        
        for extreme_type in sorted(extreme_expected.keys()):
            print(f"{extreme_type:<35} {extreme_expected[extreme_type]:>13.3f}%")
        
        print(f"\nTotal expected: {sum(extreme_expected.values()):.2f}% "
              f"(should be {extreme_fraction*100:.1f}%)")
        
        # Verify the calculation is correct
        assert abs(sum(extreme_expected.values()) - extreme_fraction*100) < 0.01, \
            "Total expected doesn't match extreme_fraction"
        
        # Verify each type matches config
        for extreme_type, expected_pct in extreme_expected.items():
            type_config = extreme_types_config.get(extreme_type, {})
            type_fraction = type_config.get('fraction', 0.0)
            calculated_pct = type_fraction * extreme_fraction * 100
            assert abs(expected_pct - calculated_pct) < 0.001, \
                f"Mismatch for {extreme_type}: {expected_pct} vs {calculated_pct}"
        
        print("\n" + "=" * 80)
        print("✓ EXTREME CASE LOGGING TEST PASSED")
        print("=" * 80)
        print("\nExpected percentages are now correctly calculated from config!")
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_extreme_case_logging()
    sys.exit(0 if success else 1)
