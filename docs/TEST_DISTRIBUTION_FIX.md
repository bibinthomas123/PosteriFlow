# Testing Guide: Class Imbalance Distribution Fix

## Overview
This guide helps verify that the event type distribution (BBH/BNS/NSBH) is now properly balanced.

## Quick Test

### 1. Generate Small Test Dataset
```bash
cd /home/bibinathomas/PosteriFlow

# Generate 1000 samples with 35% overlap
python -m ahsd.data.scripts.generate_dataset \
  --n-samples 1000 \
  --output-dir data/test_distribution_fix \
  --overlap-fraction 0.35 \
  --edge-case-fraction 0.05 \
  --add-glitches \
  --no-save-complete \
  --verbose
```

Expected time: 2-5 minutes (includes ~3s calibration overhead)

### 2. Check Output Logs
Look for the event distribution table at the end of generation. Expected output:

```
2025-11-08 XX:XX:XX - ahsd.data.dataset_generator - INFO - 📊 EVENT TYPE DISTRIBUTION:
2025-11-08 XX:XX:XX - ahsd.data.dataset_generator - INFO - ────────────────────────────────────────────────────────────────────────────────
2025-11-08 XX:XX:XX - ahsd.data.dataset_generator - INFO - Signal-level distribution (individual signals):
2025-11-08 XX:XX:XX - ahsd.data.dataset_generator - INFO - Type            Count   Actual Expected     Diff  Status
2025-11-08 XX:XX:XX - ahsd.data.dataset_generator - INFO - ────────────────────────────────────────────────────────────────────────────────
2025-11-08 XX:XX:XX - ahsd.data.dataset_generator - INFO - BBH            460±30    46.0%    46.0%    ±3%  ✓
2025-11-08 XX:XX:XX - ahsd.data.dataset_generator - INFO - BNS            320±25    32.0%    32.0%    ±2%  ✓
2025-11-08 XX:XX:XX - ahsd.data.dataset_generator - INFO - NSBH           170±15    17.0%    17.0%    ±2%  ✓
2025-11-08 XX:XX:XX - ahsd.data.dataset_generator - INFO - noise            50±8     5.0%     5.0%    ±1%  ✓
```

**Success Criteria:**
- ✓ BBH: 46% ± 3%
- ✓ BNS: 32% ± 2%
- ✓ NSBH: 17% ± 2%
- ✓ noise: 5% ± 1%

### 3. Verify Calibration Ran
Look for calibration message in logs:
```
Calibrating parameter sampler for quota-aware event type sampling...
✓ Calibration complete: P(snr_regime | event_type) ready for conditional sampling
```

## Detailed Test Procedure

### Step 1: Clean Previous Test Data
```bash
rm -rf data/test_distribution_fix data/test_distribution_fix_detailed
```

### Step 2: Test with Different Overlap Fractions

**Test A: Low Overlap (10%)**
```bash
python -m ahsd.data.scripts.generate_dataset \
  --n-samples 500 \
  --output-dir data/test_overlap_low \
  --overlap-fraction 0.10 \
  --random-seed 42 \
  --no-save-complete
```

Expected: ~500 single signals + ~50 overlaps = ~600+ signals total
Check distribution is still ~46/32/17

**Test B: High Overlap (50%)**
```bash
python -m ahsd.data.scripts.generate_dataset \
  --n-samples 500 \
  --output-dir data/test_overlap_high \
  --overlap-fraction 0.50 \
  --random-seed 42 \
  --no-save-complete
```

Expected: ~250 single + ~250 overlaps (2-3 signals each) = ~800+ signals total
Check distribution is still ~46/32/17 even with many overlaps

### Step 3: Verify Reproducibility
```bash
# Generate same dataset twice with same seed
python -m ahsd.data.scripts.generate_dataset \
  --n-samples 1000 \
  --output-dir data/test_reproducible_1 \
  --random-seed 123 \
  --no-save-complete

python -m ahsd.data.scripts.generate_dataset \
  --n-samples 1000 \
  --output-dir data/test_reproducible_2 \
  --random-seed 123 \
  --no-save-complete
```

Expected: Identical distribution tables in both logs

## Manual Verification Script

Create a quick Python script to analyze distributions:

```python
#!/usr/bin/env python3
import pickle
from pathlib import Path
from collections import Counter

def analyze_distribution(dataset_dir: str):
    """Analyze event type distribution from generated dataset."""
    
    dataset_path = Path(dataset_dir)
    
    # Find sample files
    train_dir = dataset_path / 'train'
    if not train_dir.exists():
        print(f"Error: {train_dir} not found")
        return
    
    # Load samples from train split
    sample_files = sorted(train_dir.glob('*chunk*.pkl'))[:5]  # First 5 chunks
    
    signal_types = Counter()
    sample_types = Counter()
    
    for chunk_file in sample_files:
        with open(chunk_file, 'rb') as f:
            samples = pickle.load(f)
            
        for sample in samples:
            if sample is None:
                continue
            
            # Track sample-level type
            sample_type = sample.get('type', 'unknown')
            sample_types[sample_type] += 1
            
            # Track signal-level types
            params = sample.get('parameters', [])
            if params:
                if isinstance(params, list):
                    for p in params:
                        if isinstance(p, dict):
                            sig_type = p.get('type', 'unknown')
                            signal_types[sig_type] += 1
                elif isinstance(params, dict):
                    sig_type = params.get('type', 'unknown')
                    signal_types[sig_type] += 1
    
    # Print results
    print("\n" + "="*70)
    print("SIGNAL-LEVEL DISTRIBUTION")
    print("="*70)
    
    total_signals = sum(signal_types.values())
    expected = {'BBH': 0.46, 'BNS': 0.32, 'NSBH': 0.17, 'noise': 0.05}
    
    print(f"{'Type':<15} {'Count':<10} {'Actual':<10} {'Expected':<10} {'Diff':<10}")
    print("-"*70)
    
    for event_type in sorted(signal_types.keys()):
        count = signal_types[event_type]
        actual = count / total_signals * 100 if total_signals > 0 else 0
        exp = expected.get(event_type, 0.0) * 100
        diff = actual - exp
        status = "✓" if abs(diff) <= 3 else "✗"
        
        print(f"{event_type:<15} {count:<10} {actual:<9.1f}% {exp:<9.1f}% {diff:+.1f}% {status}")
    
    print(f"\nTotal signals: {total_signals}")
    print("\n" + "="*70)
    print("SAMPLE-LEVEL DISTRIBUTION")
    print("="*70)
    
    total_samples = sum(sample_types.values())
    print(f"{'Type':<15} {'Count':<10} {'Percent':<10}")
    print("-"*70)
    
    for event_type in sorted(sample_types.keys()):
        count = sample_types[event_type]
        pct = count / total_samples * 100 if total_samples > 0 else 0
        print(f"{event_type:<15} {count:<10} {pct:<9.1f}%")
    
    print(f"\nTotal samples: {total_samples}")

if __name__ == '__main__':
    import sys
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/test_distribution_fix'
    analyze_distribution(dataset_dir)
```

Save as `analyze_distribution.py` and run:
```bash
python analyze_distribution.py data/test_distribution_fix
```

## Expected Behavior After Fix

### Before Fix
```
BBH: 94.4% ❌ (expected 46%)
BNS: 3.8%  ❌ (expected 32%)
NSBH: 1.8% ❌ (expected 17%)
```

### After Fix
```
BBH: 46% ± 3% ✓
BNS: 32% ± 2% ✓
NSBH: 17% ± 2% ✓
noise: 5% ± 1% ✓
```

## Debugging Tips

### If Distribution is Still Imbalanced

1. **Check that quota_mode is enabled:**
   ```bash
   grep "quota_mode" /home/bibinathomas/PosteriFlow/src/ahsd/data/scripts/generate_dataset.py
   ```
   Should see: `'quota_mode': config.get('quota_mode', True)`

2. **Verify method name is fixed:**
   ```bash
   grep "event_type_given_snr" /home/bibinathomas/PosteriFlow/src/ahsd/data/dataset_generator.py | grep 3
   ```
   Should show corrected method name (no `_regime`)

3. **Check calibration runs:**
   ```bash
   python -m ahsd.data.scripts.generate_dataset \
     --n-samples 100 \
     --output-dir data/test_debug \
     --verbose 2>&1 | grep -i calibrat
   ```
   Should see calibration messages

4. **Verify quota computation:**
   Add `--quota-verbose` flag (if supported) to see quota allocation

### If Calibration Fails

The code has a fallback: if calibration fails, it logs a warning and continues with marginal distributions. This won't prevent generation but may reduce accuracy.

To debug:
```bash
python -c "
from ahsd.data.parameter_sampler import ParameterSampler
sampler = ParameterSampler()
try:
    cal = sampler.empirical_calibrate(n_samples=100)
    print('Calibration successful')
    print(cal)
except Exception as e:
    print(f'Calibration failed: {e}')
    import traceback
    traceback.print_exc()
"
```

## Performance Expectations

### Time Breakdown (1000 samples, 35% overlap)
- Calibration: ~3-5 seconds
- Generation: ~1-2 minutes
- Total: ~2-3 minutes

### Memory Usage
- Typical: 500MB - 1GB
- Peak: ~1.5GB during overlap generation

## Reporting Issues

If distribution is still imbalanced after applying fixes:

1. Generate test dataset with `--verbose` flag
2. Capture logs to file:
   ```bash
   python -m ahsd.data.scripts.generate_dataset \
     --n-samples 1000 \
     --output-dir data/debug_distribution \
     --verbose 2>&1 | tee distribution_debug.log
   ```
3. Check for error messages in logs
4. Verify all three changes are applied correctly

## Next Steps

Once distribution is verified:
1. Train a model with balanced dataset
2. Compare validation performance with old (imbalanced) dataset
3. Update AGENTS.md with quota_mode recommendation
4. Document in model training guide
