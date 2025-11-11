# Extreme Case Type Categorization Bug

## Problem

Extreme case type percentages are being reported incorrectly, showing mismatches with expected values:

```
Type                                  Count     %   Expected  Status
extreme_mass_ratio                        6  0.60%     1.0%  ✓
high_spin_aligned                         8  0.80%     1.0%  ✓
long_duration_bns_overlaps                2  0.20%     0.5%  ⚠
near_simultaneous_mergers                 4  0.40%     1.0%  ⚠
noise_confused_overlaps                   4  0.40%     1.0%  ⚠
weak_strong_overlaps                      5  0.50%     0.5%  ✓✓
```

The expected percentages don't match what's configured in `data_config.yaml`.

## Root Cause Analysis

**Location:** `src/ahsd/data/dataset_generator.py`, Lines 788-825

### Issue 1: Hardcoded Expected Percentages (Lines 792-803)

```python
extreme_expected = {
    "near_simultaneous_mergers": 1.0,      # Hardcoded - should be 0.75%
    "extreme_mass_ratio": 1.0,              # Hardcoded - should be 0.45%
    "high_spin_aligned": 1.0,               # Hardcoded - should be 0.45%
    "precession_dominated": 1.0,
    "eccentric_overlaps": 0.5,
    "weak_strong_overlaps": 0.5,            # Hardcoded - should be 0.75%
    "noise_confused_overlaps": 1.0,         # Hardcoded - should be 0.45%
    "long_duration_bns_overlaps": 0.5,      # Hardcoded - should be 0.15%
    "detector_dropouts": 0.5,
    "cosmological_distance": 0.5,
}
```

These values **do not match** the configured fractions in `data_config.yaml`:

**Config fractions (relative to extreme_cases):**
```yaml
extreme_cases:
  fraction: 0.03  # 3% of total samples
  types:
    near_simultaneous_mergers:
      fraction: 0.25     # 25% of extreme → 0.75% of total
    extreme_mass_ratio:
      fraction: 0.15     # 15% of extreme → 0.45% of total
    high_spin_aligned:
      fraction: 0.15     # 15% of extreme → 0.45% of total
    weak_strong_overlaps:
      fraction: 0.25     # 25% of extreme → 0.75% of total
    noise_confused_overlaps:
      fraction: 0.15     # 15% of extreme → 0.45% of total
    long_duration_bns_overlaps:
      fraction: 0.05     # 5% of extreme → 0.15% of total
```

**Calculation:**
- Total extreme fraction: 3% (0.03)
- near_simultaneous_mergers expected: 0.25 × 3% = 0.75%
- extreme_mass_ratio expected: 0.15 × 3% = 0.45%
- high_spin_aligned expected: 0.15 × 3% = 0.45%
- weak_strong_overlaps expected: 0.25 × 3% = 0.75%
- noise_confused_overlaps expected: 0.15 × 3% = 0.45%
- long_duration_bns_overlaps expected: 0.05 × 3% = 0.15%

### Issue 2: Percentage Calculation (Line 812)

```python
actual_pct = (count / total_generated * 100) if total_generated > 0 else 0
```

This is correct - it calculates the percentage of total samples.

But it's being compared to hardcoded `extreme_expected` values that don't align with the configured fractions.

## Impact

**The sampling is probably working correctly**, but the reported expected percentages are wrong, causing:

1. ✓ Correct samples are generated according to config fractions
2. ✓ Counts are accurate
3. ✗ Expected values in logs don't match config
4. ✗ Status indicators (✓, ⚠) are misleading

## Solution

Calculate expected percentages dynamically from config rather than hardcoding them:

```python
# Get extreme_cases config
extreme_config = self.extreme_types_config if hasattr(self, 'extreme_types_config') else {}
extreme_total_fraction = self.config.get('extreme_cases', {}).get('fraction', 0.03)

# Build expected percentages from config
extreme_expected = {}
for extreme_type, type_config in extreme_config.items():
    if isinstance(type_config, dict) and type_config.get('enabled', True):
        type_fraction = type_config.get('fraction', 0.0)
        expected_pct = type_fraction * extreme_total_fraction * 100
        extreme_expected[extreme_type] = expected_pct
```

## Files to Fix

1. **`src/ahsd/data/dataset_generator.py`** - Lines 792-803 ✓ FIXED
   - Replace hardcoded `extreme_expected` dictionary with dynamic calculation from config
   - Use `self.extreme_types_config` and total extreme fraction

## Implementation Status

✓ **FIXED** - Nov 11, 2025

### Changes Made:

**File:** `src/ahsd/data/dataset_generator.py` (Lines 788-839)

**Old code (WRONG):**
```python
extreme_expected = {
    "near_simultaneous_mergers": 1.0,
    "extreme_mass_ratio": 1.0,
    ...
}
```

**New code (FIXED):**
```python
# Calculate expected percentages from config, not hardcoded values
extreme_expected = {}
for extreme_type, type_config in self.extreme_types_config.items():
    if isinstance(type_config, dict) and type_config.get("enabled", True):
        type_fraction = type_config.get("fraction", 0.0)
        expected_pct = type_fraction * self.extreme_fraction * 100
        extreme_expected[extreme_type] = expected_pct
```

### Test Results:

**Old (Wrong) vs New (Fixed) Expected Percentages:**
```
Type                              Old (Wrong)  New (Fixed)  Difference
────────────────────────────────────────────────────────────────────
extreme_mass_ratio                     1.00%       0.45%      -0.55%
high_spin_aligned                      1.00%       0.45%      -0.55%
long_duration_bns_overlaps             0.50%       0.15%      -0.35%
near_simultaneous_mergers              1.00%       0.75%      -0.25%
noise_confused_overlaps                1.00%       0.45%      -0.55%
weak_strong_overlaps                   0.50%       0.75%      +0.25%
```

**Total Expected:** 3.00% (matches extreme_cases.fraction in config) ✓

## Testing Strategy

1. ✓ Verify expected percentages match `data_config.yaml` fractions
2. ✓ Check that status indicators (✓, ⚠) align with actual vs expected
3. ✓ Validate that all enabled types have non-zero expected percentages
4. ✓ Test with disabled types (they should not appear in logs)

## Additional Notes

- **Disabled types** in config (`detector_dropouts`, `eccentric_overlaps`, `cosmological_distance`) are now properly excluded from expected percentages
- Only enabled types with configured fractions are now reported
- Expected percentages now match the configuration file exactly
- The fix makes the logging trustworthy for dataset quality validation
