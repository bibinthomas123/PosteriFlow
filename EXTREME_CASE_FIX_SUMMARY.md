# Extreme Case Type Expected Percentage Fix - Complete Analysis

**Date:** November 11, 2025  
**Status:** ✓ FIXED AND VALIDATED

## Problem Summary

Extreme case type expected percentages were **hardcoded with wrong values**, causing misleading status indicators in dataset generation logs:

```
Reported with OLD code (wrong):
Type                                  Count     %   Expected  Status
extreme_mass_ratio                        6  0.60%     1.0%  ✓
high_spin_aligned                         8  0.80%     1.0%  ✓
long_duration_bns_overlaps                2  0.20%     0.5%  ⚠
near_simultaneous_mergers                 4  0.40%     1.0%  ⚠
noise_confused_overlaps                   4  0.40%     1.0%  ⚠
weak_strong_overlaps                      5  0.50%     0.5%  ✓✓
```

The expected values (1.0%, 0.5%) were **completely arbitrary**, not derived from the configuration.

## Root Cause

**File:** `src/ahsd/data/dataset_generator.py`, Lines 792-803

The code had a **hardcoded dictionary** of expected percentages:

```python
extreme_expected = {
    "near_simultaneous_mergers": 1.0,      # Wrong!
    "extreme_mass_ratio": 1.0,              # Wrong!
    "high_spin_aligned": 1.0,               # Wrong!
    "weak_strong_overlaps": 0.5,            # Wrong!
    "noise_confused_overlaps": 1.0,         # Wrong!
    "long_duration_bns_overlaps": 0.5,      # Wrong!
    ...
}
```

These hardcoded values **didn't match the configuration**, which actually specified:

**From `configs/data_config.yaml`:**
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

## Calculation Logic

Expected percentage for each extreme type:
```
expected_pct = type_fraction × extreme_total_fraction × 100

Example:
- near_simultaneous_mergers expected = 0.25 × 0.03 × 100 = 0.75%
- extreme_mass_ratio expected = 0.15 × 0.03 × 100 = 0.45%
- long_duration_bns_overlaps expected = 0.05 × 0.03 × 100 = 0.15%
```

## Impact

**The sampling was correct**, but the **expected values were wrong**, causing:

1. ✓ Actual samples generated correctly per config
2. ✓ Counts were accurate
3. ✗ Expected percentages reported were arbitrary hardcoded values
4. ✗ Status indicators (✓, ⚠) were misleading and unreliable

## Solution

Replace hardcoded dictionary with **dynamic calculation from configuration**:

```python
# ✓ NEW: Calculate expected percentages from config
extreme_expected = {}
for extreme_type, type_config in self.extreme_types_config.items():
    if isinstance(type_config, dict) and type_config.get("enabled", True):
        type_fraction = type_config.get("fraction", 0.0)
        expected_pct = type_fraction * self.extreme_fraction * 100
        extreme_expected[extreme_type] = expected_pct
```

**Benefits:**
- Expected values now match the configuration file
- Disabled types are automatically excluded
- Changes to config are reflected in logs without code changes
- Status indicators are now reliable

## Files Changed

1. **`src/ahsd/data/dataset_generator.py`** - Lines 791-839
   - Removed hardcoded `extreme_expected` dictionary
   - Added dynamic calculation from `self.extreme_types_config`
   - Only includes enabled types (respects config settings)

## Validation Results

### Test: Expected Percentage Calculation

**Before (Old Hardcoded Values):**
```
near_simultaneous_mergers:    1.00%  (WRONG - should be 0.75%)
extreme_mass_ratio:            1.00%  (WRONG - should be 0.45%)
high_spin_aligned:             1.00%  (WRONG - should be 0.45%)
weak_strong_overlaps:          0.50%  (WRONG - should be 0.75%)
noise_confused_overlaps:       1.00%  (WRONG - should be 0.45%)
long_duration_bns_overlaps:    0.50%  (WRONG - should be 0.15%)
```

**After (Config-Based Calculation):**
```
near_simultaneous_mergers:    0.75%  ✓
extreme_mass_ratio:           0.45%  ✓
high_spin_aligned:            0.45%  ✓
weak_strong_overlaps:         0.75%  ✓
noise_confused_overlaps:      0.45%  ✓
long_duration_bns_overlaps:   0.15%  ✓
────────────────────────────────────
Total:                        3.00%  ✓ (matches config)
```

### Difference Analysis

```
Type                            Old (Wrong)  New (Fixed)  Difference
────────────────────────────────────────────────────────────────────
extreme_mass_ratio                  1.00%       0.45%       -0.55%
high_spin_aligned                   1.00%       0.45%       -0.55%
long_duration_bns_overlaps          0.50%       0.15%       -0.35%
near_simultaneous_mergers           1.00%       0.75%       -0.25%
noise_confused_overlaps             1.00%       0.45%       -0.55%
weak_strong_overlaps                0.50%       0.75%       +0.25%
```

## Next Steps

Future dataset generation runs will now show **accurate expected percentages** in the logs:

```bash
python src/ahsd/data/scripts/generate_dataset.py \
    --config configs/data_config.yaml \
    --num-samples 10000
```

**Expected output:**
```
⚡ EXTREME CASE TYPES (Publication Quality):
Type                              Count     %   Expected  Status
────────────────────────────────────────────────────────────────
extreme_mass_ratio                  42   0.42%     0.45%  ✓
high_spin_aligned                   43   0.43%     0.45%  ✓
long_duration_bns_overlaps          14   0.14%     0.15%  ✓
near_simultaneous_mergers           77   0.77%     0.75%  ✓✓
noise_confused_overlaps             45   0.45%     0.45%  ✓✓
weak_strong_overlaps                75   0.75%     0.75%  ✓✓
────────────────────────────────────────────────────────────────
Total extreme cases               296   2.96%
```

Status indicators (✓, ✓✓, ⚠) are now **meaningful and reliable** for validating dataset quality.

## Related Fixes

This is the second categorization bug found and fixed:
1. **SNR Categorization Bug** - Fixed Nov 11, 2025
   - Issue: Wrong SNR regime boundaries in three functions
   - Fix: Use SNR_RANGES from config consistently

2. **Extreme Case Categorization Bug** - Fixed Nov 11, 2025
   - Issue: Hardcoded expected percentages
   - Fix: Calculate from config dynamically

Both issues stemmed from hardcoding expected values instead of deriving them from configuration.
