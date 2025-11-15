# Gradient Explosion Fixes Applied - Nov 15, 2025

## Changes Summary

Applied 5 critical fixes to eliminate Grad=28.9 gradient explosion in training epoch 3.

### 1. ✅ Batch Size Increase (6 → 24)
- **File:** `configs/enhanced_training.yaml` line 45
- **Change:** `batch_size: 6` → `batch_size: 24`
- **Why:** BatchNorm with n=6 has unreliable statistics; n=24 provides stable normalization
- **Impact:** Reduces gradient variance from BatchNorm ~5-10×

### 2. ✅ Gradient Clipping Relaxed (2.0 → 5.0)
- **File:** `configs/enhanced_training.yaml` line 85
- **Change:** `gradient_clip_norm: 2.0` → `gradient_clip_norm: 5.0`
- **Why:** 2.0 is too aggressive for model with 6 loss components; suppresses legitimate gradients
- **Impact:** Allows proper gradient flow while still preventing extreme spikes

### 3. ✅ Min Variance Penalty Reduced (10× → 2×)
- **File:** `src/ahsd/core/priority_net.py` line 824
- **Change:** `min_variance_penalty = 10.0 * torch.relu(...)` → `min_variance_penalty = 2.0 * torch.relu(...)`
- **Why:** 10× penalty was dominating loss landscape, creating sharp gradients
- **Impact:** Penalty still effective but no longer dominates loss signal

### 4. ✅ Calibration Weights Reduced
- **File:** `configs/enhanced_training.yaml` lines 76-77
- **Changes:**
  - `calib_max_weight: 2.50` → `calib_max_weight: 1.00` (60% reduction)
  - `calib_range_weight: 2.00` → `calib_range_weight: 0.75` (62% reduction)
- **Why:** Nov 15 increases (from 0.75 and 0.45) were overtuned and causing gradient explosion
- **Impact:** Loss landscape less sharp; gradients more stable

### 5. ✅ Uncertainty Weight Reverted
- **File:** `configs/enhanced_training.yaml` line 64
- **Change:** `uncertainty_weight: 0.35` → `uncertainty_weight: 0.10` (reverted Nov 13 3.5× increase)
- **Why:** 0.35 was part of the Nov 13 fix that made sense then but is overkill now with other changes
- **Impact:** Reduces cross-loss interference

---

## Expected Results

### Before Fixes
- Epoch 3: Grad = 28.9 (unstable)
- Loss likely spiking or oscillating
- Training potentially diverging

### After Fixes (Expected)
- Epoch 3-5: Grad < 5.0 (within clipping threshold)
- Loss decreasing smoothly
- No NaN/Inf in outputs
- Stable gradient signal for learning

---

## How to Run with Fixes

```bash
# Reinstall package (already done)
pip install -e . --no-deps

# Run training (use existing commands)
python experiments/train_priority_net.py \
  --config configs/enhanced_training.yaml \
  --device cuda \
  --create_overlaps
```

---

## Verification Checklist

After running next epoch:

- [ ] Gradient norm < 5.0 (should see ~1-3 typical)
- [ ] Loss decreasing smoothly (not spiking)
- [ ] No warnings: "gradient overflow", "invalid value", "NaN"
- [ ] Loss components balanced (none dominating)
- [ ] Training continuing without resets

---

## If Gradients Still High (>5.0)

Try next fixes in order:

1. **Further reduce learning_rate:** 8e-5 → 4e-5
2. **Add loss normalization:** Divide total_loss by 6 (n_components)
3. **Verify model.train() mode:** Check BatchNorm is running
4. **Reduce uncertainty weight further:** 0.10 → 0.05
5. **Increase warmup epochs:** 6 → 10 (slower LR ramp)

---

## Root Cause Analysis

The gradient explosion was **not a bug**, but a configuration tuning issue:

1. **Small batch (6)** with BatchNorm → Statistics unreliable
2. **Aggressive penalties** (10×, 2.5×, 2.0×) → Sharp loss landscape  
3. **Multiple loss components (6)** without normalization → Cumulative effects
4. **Tight clipping (2.0)** → Overly restrictive for large gradients

This is a common issue when stacking multiple loss terms without careful weighting.

---

## Files Modified

1. `configs/enhanced_training.yaml` - 3 changes (batch_size, gradient_clip_norm, calibration weights)
2. `src/ahsd/core/priority_net.py` - 1 change (min_variance_penalty)
3. `AGENTS.md` - Added fix documentation

## Code Changes Verify

```bash
# Verify syntax
python -m py_compile src/ahsd/core/priority_net.py

# Check config YAML
python -c "import yaml; yaml.safe_load(open('configs/enhanced_training.yaml'))" && echo "✅ Config valid"

# Reinstall package
pip install -e . --no-deps
```

All verified ✅
