# High Gradient Diagnosis Report (Grad=28.9)

## Summary
Training is experiencing exploding gradients (28.9) despite gradient clipping enabled. This is a sign of sharp loss landscape or unstable loss scaling.

---

## 1. GRADIENT CLIPPING ANALYSIS ✓ GOOD

**Current Setting:**
- `gradient_clip_norm: 2.0` (config/enhanced_training.yaml line 85)
- Code: `torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)` (priority_net.py line 1673)

**Status:** ✅ Clipping is implemented and active

**Issue:** Gradient of 28.9 means:
- BEFORE clipping: Likely ~100-300 (clipped to 2.0)
- AFTER clipping: Clamped to 2.0, but then somehow 28.9 reported
- **Possible cause:** Gradient norm reported BEFORE clip_grad_norm is applied, or clip not covering all parameters

---

## 2. LEARNING RATE ANALYSIS ⚠️ POTENTIALLY TOO HIGH

**Current Setting:**
- `learning_rate: 8.0e-5` (config line 42)

**Analysis:**
- This is **very conservative** for AdamW (typical 1e-4 to 5e-4)
- But combined with high gradients, even conservative LR can cause instability
- **Warmup helps:** `warmup_epochs: 6, warmup_start_factor: 0.02` should gradually ramp up

**Recommended Check:**
- Is warmup actually active in epoch 3? (Warmup should be epochs 1-6)
- If epoch 3 is still in warmup phase, effective LR might be ~3.2e-5 (OK)
- If epoch 3 is post-warmup, LR=8e-5 with grad=28.9 gives `update_size = 8e-5 * 28.9 ≈ 2.3e-3` (large step)

---

## 3. BATCH NORMALIZATION ANALYSIS ✓ PRESENT

**Current Usage:**
- TemporalStrainEncoder: BatchNorm1d in conv_blocks (line 92)
- Used in multiple decoder layers
- Running mode during training: should normalize per-batch statistics

**Potential Issues:**
- Batch size: 6 (very small!) — BatchNorm statistics unreliable with n<16
- **This is likely a major contributor** to gradient instability
- Small batch → high variance in batch stats → inconsistent gradient signals

**Verification Needed:** 
- Check if BatchNorm is in eval mode by accident: `model.train()` must be called

---

## 4. LOSS SCALING ANALYSIS ⚠️ MULTIPLE ISSUES FOUND

### Issue A: Excessive Loss Component Weights
```
loss_weights:
  - ranking: 0.70  ← Large, adaptive pairwise loss
  - mse: 0.20      ← Small, regression component
  - uncertainty: 0.10
  + uncertainty_bounds: depends on params
  + calib_loss: depends on params
  + min_variance_penalty: up to 10.0 (!!!)
```

**Problem:** Total loss is sum of 6 components with NO normalization. If batch has:
- High variance in targets → calibration penalties spike
- Low SNR signals → ranking loss explodes
- Min variance penalty: 10× multiplicative factor (can be ~10 by itself!)

### Issue B: Min Variance Penalty is Explosive
```python
# Line 824
min_variance_penalty = 10.0 * torch.relu(0.5 - variance_ratio)
```

If `pred_std << target_std`, this alone can be 5-10, dominating the loss.

### Issue C: Calibration Weights are Very Aggressive
```yaml
calib_max_weight: 2.50   # Nov 15 increased from 0.75 (3.3× stronger!)
calib_range_weight: 2.00 # Nov 15 increased from 0.45 (4.4× stronger!)
```

With ratio-based penalties that can be [0, 1], these contribute 0-2.5 to loss.
Combined with ranking (0.7) + mse (0.2) = total ~4-5 minimum.

### Issue D: Ranking Loss Dynamics Unknown
```python
# Line 747 - uses AdaptiveRankingLoss
ranking_loss = self.ranking_loss_fn(...)
```

AdaptiveRankingLoss uses pairwise comparisons. With batch_size=6:
- (6 choose 2) = 15 pairs
- Each pair creates a gradient signal
- If all 15 pairs are "wrong" order, gradient accumulates 15× signal

---

## 5. RECOMMENDED FIXES

### IMMEDIATE (High Priority - Apply Now)

**Fix #1: Reduce Batch Size Effect with Gradient Accumulation**
```python
# Instead of batch_size=6, use:
# - batch_size=24 with gradient_accumulation_steps=1 (better batch norm stats)
# - Or batch_size=6 with gradient_accumulation_steps=4 (same effective size)
```
**Why:** Larger batch → more stable BatchNorm → lower gradient variance

**Fix #2: Increase Gradient Clipping Threshold**
```yaml
# In enhanced_training.yaml line 85, change:
gradient_clip_norm: 2.0  # ← TOO AGGRESSIVE
# To:
gradient_clip_norm: 5.0  # More reasonable for large models
```
**Why:** 2.0 is very tight. With 6+ loss components, clipping to 5.0 is still safe.

**Fix #3: Reduce Min Variance Penalty Magnitude**
```python
# Line 824 in priority_net.py, change:
min_variance_penalty = 10.0 * torch.relu(0.5 - variance_ratio)
# To:
min_variance_penalty = 2.0 * torch.relu(0.5 - variance_ratio)  # Reduce 10→2
```
**Why:** 10× is way too aggressive. A 2× penalty is still effective.

**Fix #4: Add Loss Component Normalization**
```python
# After computing all loss components, add:
total_loss = (
    effective_mse_weight * mse_loss
    + effective_ranking_weight * ranking_loss
    + self.uncertainty_weight * uncertainty_loss
    + uncertainty_bounds_loss
    + calib_loss
    + min_variance_penalty
)

# NORMALIZE by number of components to prevent scale drift
n_components = 6
total_loss = total_loss / n_components  # or use average instead of sum
```
**Why:** Prevents any single component from dominating gradient signals.

### SECONDARY (Medium Priority - Apply After Immediate Fixes)

**Fix #5: Reduce Calibration Weights**
```yaml
# In enhanced_training.yaml, reduce aggressive Nov 15 increases:
calib_max_weight: 1.0    # Down from 2.50
calib_range_weight: 0.75 # Down from 2.00
```
**Why:** These are meant to guide, not dominate. Current values too aggressive.

**Fix #6: Check if BatchNorm is Running Correctly**
```python
# Add diagnostic logging in trainer:
print(f"Model training mode: {model.training}")
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm1d):
        print(f"  {name}: training={module.training}, momentum={module.momentum}")
```

**Fix #7: Reduce Uncertainty Weight Post-Convergence**
```yaml
# Line 64, reduce:
uncertainty_weight: 0.10  # Down from 0.35 (Nov 13 increase was 3.5×)
```
**Why:** Uncertainty calibration is good, but 0.35 may be overkill at this stage.

---

## 3. VALIDATION CHECKLIST

After applying fixes, verify:
- [ ] Epoch 3 gradient < 5.0 (should be 0.5-2.0 with clip=5.0)
- [ ] Loss decreases smoothly (not spiking)
- [ ] Batch norm running stats updating (check with `model.eval()` mode)
- [ ] No NaN/Inf in loss or predictions
- [ ] Training loss < validation loss (normal pattern)

---

## 4. ROOT CAUSE SUMMARY

The high gradient (28.9) is likely caused by:

1. **Small batch size (6)** with BatchNorm → high variance in normalization stats
2. **Aggressive loss weights** (min_variance_penalty=10.0, calib_max=2.5)
3. **Tight gradient clipping (2.0)** clipping before other stabilization kicks in
4. **Multiple loss components (6)** without normalization → cumulative effects

The fix is to:
1. Increase batch size OR use gradient accumulation
2. Reduce aggressive penalty coefficients
3. Relax gradient clipping threshold
4. Add loss normalization

---

## Implementation Order

1. **TODAY:** Apply Fixes #1, #2, #3
2. **NEXT RUN:** Monitor epoch 3-5 gradients
3. **IF STILL HIGH:** Apply Fixes #4, #5
4. **IF NEEDED:** Reduce learning rate by 2× (to 4e-5)

This is a configuration tuning issue, not a code bug. The model is fine; it just needs better hyperparameter balance.
