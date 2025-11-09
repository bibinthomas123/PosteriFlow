# Analysis and Fix for Class Imbalance Issue

## Problem Summary

The dataset shows severe class imbalance:
- **BBH: 94.4%** (expected 46.0%, +48.4% overrepresented)
- **BNS: 3.8%** (expected 32.0%, -28.2% underrepresented)  
- **NSBH: 1.8%** (expected 17.0%, -15.2% underrepresented)

## Root Causes

### 1. **Method Name Typo (Line 3263)**
**File:** `src/ahsd/data/dataset_generator.py`

In `_generate_overlapping_sample()`:
```python
event_type = self.parameter_sampler.event_type_given_snr_regime(snr_regime)  # ❌ Wrong method name
```

Should be:
```python
event_type = self.parameter_sampler.event_type_given_snr(snr_regime)  # ✅ Correct
```

The method `event_type_given_snr_regime()` doesn't exist in `ParameterSampler`. This causes an exception that's silently caught, falling back to `_sample_event_type()` which uses the global 46/32/17 distribution.

### 2. **No Quota Enforcement in Overlapping Samples**
**File:** `src/ahsd/data/dataset_generator.py` (lines 1295-1350)

The main generation loop has two paths:
- **Quota mode enabled** (line 1300): Uses `forced_signals` to enforce quotas per signal
- **Quota mode disabled** (default): Overlapping signals sample from global distribution without quota control

**Config Issue:** The generator defaults to `quota_mode = False` (line 982), and the script doesn't pass `quota_mode=True` in the config.

### 3. **Diversity Logic in Overlaps (Lines 3267-3279)**
When generating multiple signals in an overlap, if the first signal is BBH (46% chance), subsequent signals avoid BBH type and must sample from {BNS, NSBH} only. However, this doesn't re-balance the quotas—it just creates local diversity.

Over 22,860 overlap samples with ~2.5 signals each (~57k signals):
- First signal of each overlap: 46% BBH (26k BBH)
- Subsequent signals: Must avoid first type but sample from global weights
- Result: BBH still dominates with 78k out of 83k signals

## Solutions

### Quick Fix (Method Name Only)
```python
# In src/ahsd/data/dataset_generator.py, line 3263
- event_type = self.parameter_sampler.event_type_given_snr_regime(snr_regime)
+ event_type = self.parameter_sampler.event_type_given_snr(snr_regime)
```

**Impact:** Minimal—allows conditional sampling by SNR regime but still uses global distribution.

### Comprehensive Fix (Quota Enforcement)

**Option 1: Enable Quota Mode in Config**
```yaml
# config.yaml
quota_mode: true
expected_signals_per_overlap: 2.5
```

**Option 2: Pass quota_mode in generate_dataset.py**
```python
# Line 158-165 in generate_dataset.py
generator = GWDatasetGenerator(
    output_dir=output_dir,
    ...,
    config={
        **config,
        'quota_mode': True,  # ✅ Add this
        'expected_signals_per_overlap': 2.5
    }
)
```

**Option 3: Re-balance Overlap Diversity Logic (Lines 3267-3279)**

Replace the current diversity logic with quota-aware sampling:
```python
# After sampling first signal
if i > 0 and n_signals > 1:
    first_type = signal_params_list[0]['type']
    available_types = [t for t in ['BBH', 'BNS', 'NSBH'] if t != first_type]
    
    # OLD: Uses global weights, ignoring quotas
    # weights = [EVENT_TYPE_DISTRIBUTION.get(t, 0.0) for t in available_types]
    
    # NEW: Use quota-aware conditional distribution
    if hasattr(self.parameter_sampler, 'conditional_snr') and self.parameter_sampler.conditional_snr:
        # Invert P(snr|type) → P(type|snr) using Bayes' rule
        event_type = self.parameter_sampler.event_type_given_snr(snr_regime)
        if event_type == first_type:
            # Resample excluding first type
            available_types = [t for t in ['BBH', 'BNS', 'NSBH'] if t != first_type]
            weights = [EVENT_TYPE_DISTRIBUTION.get(t, 0.0) for t in available_types]
            ...
```

## Implementation Steps

### Step 1: Fix Method Name Typo (Priority: High)
```bash
# Edit line 3263 in dataset_generator.py
```

### Step 2: Enable Quota Mode (Priority: High)
**Option A:** Update config YAML
```yaml
quota_mode: true
expected_signals_per_overlap: 2.5
```

**Option B:** Update generate_dataset.py
Add to line 164:
```python
'quota_mode': config.get('quota_mode', True),  # Default to True for balanced distribution
```

### Step 3: Verify Calibration (Priority: Medium)
Ensure empirical calibration is run:
```python
# In dataset_generator.py, around line 900
sampler.empirical_calibrate(n_samples=2000, random_seed=42)
```

## Testing

Generate a small dataset and verify distribution:
```bash
python -m ahsd.data.scripts.generate_dataset \
  --n-samples 1000 \
  --output-dir data/test_distribution \
  --output-format pkl \
  --no-save-complete
```

Check the logs for the event type distribution report. Expected after fix:
- BBH: ~46% (tolerance ±5%)
- BNS: ~32% (tolerance ±5%)
- NSBH: ~17% (tolerance ±5%)

## Expected Outcome

After implementing all fixes:
- **Signal-level distribution** will match config (46/32/17)
- **Sample-level distribution** will reflect realistic overlaps
- **No silent failures** in overlap event type sampling
- **Quota enforcement** ensures balance across SNR regimes and event types
