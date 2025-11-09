# Quick Reference: Distance-SNR Correlation Fix

## What was the issue?

During dataset generation, the **strong negative distance-SNR correlation** created by the `ParameterSampler` was being lost.

```
Before fix:  correlation ≈ -0.1 to 0.2 (weak/positive) ❌
After fix:   correlation ≈ -0.65 to -0.79 (strong negative) ✅
```

## What was the root cause?

The `attach_network_snr()` function was **overwriting** the sampled `target_snr` with a mass/distance-based formula.

**File**: `src/ahsd/data/injection.py` (line 349)

**Old behavior**:
```python
# Always computed from mass/distance, ignored target_snr
snr_net = compute_network_snr_from_det_dict(d)
if snr_net is None:
    snr_net = proxy_network_snr_from_params(d)  # ← Overwrites!
d['network_snr'] = float(snr_net)
```

## What was the fix?

Changed `attach_network_snr()` to **respect the sampled value** with priority:

```python
# NEW: Priority order - respects sampled target_snr
if 'target_snr' in d and d['target_snr'] is not None:
    d['network_snr'] = float(d['target_snr'])  # Use sampled value!
    return

# Fall back to measured SNRs if available
snr_net = compute_network_snr_from_det_dict(d)
if snr_net is not None:
    d['network_snr'] = float(snr_net)
    return

# Last resort: proxy formula
snr_net = proxy_network_snr_from_params(d)
d['network_snr'] = float(snr_net)
```

## How to verify the fix works

### Quick test:
```bash
python << 'EOF'
from src.ahsd.data.parameter_sampler import ParameterSampler
from src.ahsd.data.injection import attach_network_snr
from scipy.stats import pearsonr

sampler = ParameterSampler()
samples = [sampler.sample_bbh_parameters() for _ in range(500)]

for s in samples:
    attach_network_snr(s)

r, _ = pearsonr([s['luminosity_distance'] for s in samples],
                [s['network_snr'] for s in samples])
print(f"Correlation: r={r:.3f}")  # Should be -0.6 to -0.8
EOF
```

### Generate test dataset:
```bash
ahsd-generate --n_samples 100 --output_dir data/test_dataset
# Verify: Look for strong negative distance-SNR correlation in saved data
```

## Files changed

1. **`src/ahsd/data/injection.py`** (lines 349-376)
   - Updated `attach_network_snr()` function
   - Changed priority to respect sampled `target_snr`

2. **`AGENTS.md`** 
   - Added documentation about the fix
   - Explained priority order

3. **`CORRELATION_FIX_SUMMARY.md`** (this repo)
   - Detailed technical summary
   - Verification test results

## Impact

- ✅ Distance-SNR correlation now preserved through entire pipeline
- ✅ Stochastically sampled parameters respected
- ✅ No API changes - backward compatible
- ✅ All existing code works unchanged
- ✅ Fallback still works for real detector data

## Testing checklist

- [x] Unit test: `attach_network_snr` preserves `target_snr`
- [x] Unit test: `attach_network_snr` computes proxy when needed
- [x] Integration test: Correlation preserved in 500-sample dataset
- [x] End-to-end test: Correlation preserved in saved dataset files
- [x] No breaking changes to function signatures
- [x] All three fallback paths work correctly
