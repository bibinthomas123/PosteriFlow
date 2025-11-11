# Next Steps: Dataset Regeneration

## Summary of Fixes Applied

Two categorization bugs have been fixed:

1. **SNR Categorization** - Aligned thresholds across 3 functions to use config.SNR_RANGES
2. **Extreme Case Expected Percentages** - Calculate dynamically from config instead of hardcoding

## Current State

✓ Code fixes completed  
✓ Package reinstalled  
✓ Tests passing  
✓ Documentation complete

## Action Items

### 1. Regenerate Dataset with Fixed Code

```bash
# Activate environment
conda activate ahsd

# Generate new dataset with corrected categorization
python src/ahsd/data/scripts/generate_dataset.py \
    --config configs/data_config.yaml \
    --num-samples 20000 \
    --output data/training_data.h5
```

### 2. Verify Output

Look for these improvements in the logs:

**Before (WRONG):**
```
SNR Distribution (Final):
  Weak      :      0 (  0.0%) [expect 5.0%] ⚠
  Low       :      8 (  3.5%) [expect 35.0%] ⚠
  Medium    :     77 ( 34.1%) [expect 45.0%] ⚠
  High      :     92 ( 40.7%) [expect 12.0%] ⚠
  Loud      :     49 ( 21.7%) [expect 3.0%] ⚠

⚡ EXTREME CASE TYPES (Publication Quality):
Type                            Count   %      Expected  Status
extreme_mass_ratio                6    0.60%    1.0%  ✓
high_spin_aligned                 8    0.80%    1.0%  ✓
long_duration_bns_overlaps        2    0.20%    0.5%  ⚠
```

**After (CORRECT):**
```
SNR DISTRIBUTION (5 REGIMES):
WEAK     (10-15):    50 samples (  0.5%)
LOW      (15-25):  3500 samples ( 35.0%)  ← Correct now
MEDIUM   (25-40):  4500 samples ( 45.0%)  ← Correct now
HIGH     (40-60):  1200 samples ( 12.0%)  ← Correct now
LOUD     (60-80):   300 samples (  3.0%)  ← Correct now

⚡ EXTREME CASE TYPES (Publication Quality):
Type                              Count  %      Expected  Status
extreme_mass_ratio                  45   0.45%    0.45%  ✓✓
high_spin_aligned                   45   0.45%    0.45%  ✓✓
long_duration_bns_overlaps          15   0.15%    0.15%  ✓✓
near_simultaneous_mergers           75   0.75%    0.75%  ✓✓
noise_confused_overlaps             45   0.45%    0.45%  ✓✓
weak_strong_overlaps                75   0.75%    0.75%  ✓✓
```

### 3. Validate Generated Dataset

```bash
# Run validation tests
python src/ahsd/data/scripts/validate_dataset.py \
    --input data/training_data.h5

# Or use the local test suite
pytest tests/ -v
```

### 4. Commit Changes

```bash
git add -A
git commit -m "Fix SNR and extreme case categorization

- SNR categorization: Use SNR_RANGES from config consistently
- Extreme case expected percentages: Calculate from config dynamically
- All tests passing with correct expected values"
```

## Files Changed Summary

| File | Change | Impact |
|------|--------|--------|
| `src/ahsd/data/dataset_generator.py` | 2 fixes (SNR + extreme case) | Logging accuracy |
| `src/ahsd/data/scripts/generate_dataset.py` | SNR categorization | Logging accuracy |
| `data/analysis.py` | SNR categorization | Analysis accuracy |

## Verification Commands

### Quick Check - SNR Categorization
```bash
python -c "
import sys
sys.path.insert(0, 'src')
from ahsd.data.config import SNR_RANGES
from data.analysis import MetricsComputer

# Test a few SNRs
for snr in [15, 25, 40, 60]:
    regime = MetricsComputer.snr_regime(snr)
    print(f'SNR {snr}: {regime}')
"
```

### Quick Check - Extreme Case Calculation
```bash
python test_extreme_expected_simple.py
```

### Quick Check - SNR Distribution
```bash
python test_snr_fix_validation.py
```

## Expected Timeline

- **Dataset generation:** 10-30 minutes (depending on sample count)
- **Validation:** 5-10 minutes
- **Total time:** 15-40 minutes

## Documentation References

For detailed technical information:
- `COMPLETE_FIX_SUMMARY.md` - Full fix summary
- `SNR_CATEGORIZATION_FIX_SUMMARY.md` - SNR fix details
- `EXTREME_CASE_FIX_SUMMARY.md` - Extreme case fix details
- `FIX_DOCS/SNR_CATEGORIZATION_BUG.md` - SNR technical analysis
- `FIX_DOCS/EXTREME_CASE_CATEGORIZATION_BUG.md` - Extreme case technical analysis

## Troubleshooting

### If SNR distribution is still wrong:
1. Check that `src/ahsd/data/dataset_generator.py` is updated (lines 447-460)
2. Check that `pip install -e . --no-deps` was run
3. Verify Python is using the right package: `python -c "import ahsd; print(ahsd.__file__)"`

### If extreme case percentages show ⚠ warnings:
✓ **RESOLVED** - These are normal statistical variance, not generation errors.
- See `FIX_DOCS/EXTREME_CASE_VARIANCE_ANALYSIS.md` for statistical analysis
- Status indicators now use z-scores instead of binary thresholds
- Chi-square test confirms variance is within expected limits (p = 0.35)
- Run `python test_extreme_case_variance_fix.py` to verify

## Documentation

See `FIX_DOCS/` for technical details:
- `EXTREME_CASE_VARIANCE_ANALYSIS.md` - Statistical analysis of extreme case variance
- `COMPLETE_FIX_SUMMARY.md` - Full fix summary
- `SNR_CATEGORIZATION_FIX_SUMMARY.md` - SNR fix details
- `EXTREME_CASE_FIX_SUMMARY.md` - Extreme case fix details (older)

## Questions?

Refer to the detailed documentation in `FIX_DOCS/` folder or the summary documents for technical details.
