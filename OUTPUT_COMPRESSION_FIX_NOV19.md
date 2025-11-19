# Output Compression Fix - Nov 19, 2025

## Problem Statement

**Observations from training logs (Epoch 38-39):**
- Predictions: `[0.0753, 0.7673]` (range = 0.6920)
- Targets: `[0.1101, 0.9500]` (range = 0.8399)
- **Compression ratio: 82.3%** (should be ~100%)
- Maximum gap: predictions 0.173 below target ceiling (0.767 vs 0.950)
- Model stuck in narrow band, not utilizing full [0, 1] range

**Impact**: Model cannot distinguish high-priority signals (targets near 0.95); ceiling crushed at 0.77

---

## Root Cause Analysis

### 1. Hard Clipping Killing Gradients (PRIMARY)
**Location**: `src/ahsd/core/priority_net.py` line 566
```python
prio = torch.clamp(prio, min=0.0, max=1.0)  # ← KILLS GRADIENTS
```

**How it breaks range expansion**:
- Affine transformation tries to expand output beyond 1.0 to reach target 0.95
- Example: if unclamped output is 1.15, clipping to 1.0 eliminates gradient signal
- Loss function cannot push gain higher when all gradients flatten at boundary
- Result: Model stops trying to expand, settles at 0.767

**Equation**:
```
unclamped_output = base_output * gain + bias
clamped_output = clamp(unclamped_output, 0, 1)
∂loss/∂gain when clamped = 0  (no gradient signal!)
```

### 2. Weak Calibration Loss vs Strong MSE (SECONDARY)
**Config before fix**:
- `mse_weight: 0.20` (pulls toward safe zone, minimizes error)
- `calib_max_weight: 0.50` (pushes ceiling upward)

**Gradient magnitude analysis**:
- MSE gradient: `O(0.20 × 2 × error)` ≈ 0.01-0.1
- Calibration gradient: `O(0.50 × ratio_penalty)` ≈ 0.1
- **Ratio**: MSE ~2× stronger, pulling down defeats calibration pushing up

**Mechanism**:
```
max_ratio_penalty = max(0, 1 - pred_max/tgt_max)
                  = max(0, 1 - 0.767/0.950)
                  = 0.193
calib_loss = 0.50 × 0.193 = 0.0965  (tiny!)
mse_loss = 0.20 × (0.767 - 0.950)² = 0.0066  (but × 2 gradient = 0.013)
→ MSE dominates, prevents range expansion
```

### 3. No Affine Parameter Logging
- Hard clipping issue was invisible during training
- No way to detect that `gain` was being clamped at [1.2, 2.5]
- Could not diagnose "is gain trained correctly?"

---

## Solution Implementation

### Fix #1: Remove Hard Clipping (CRITICAL)
**File**: `src/ahsd/core/priority_net.py` line 566

**Change**:
```python
# BEFORE
prio = torch.clamp(prio, min=0.0, max=1.0)

# AFTER  
# NOTE: Removed hard clipping. Use soft penalty in loss instead.
# Loss handles bounds via bounds_penalty; forward allows unclamped output.
```

**Why this works**:
- Preserves gradient flow for affine gain to expand output
- Clipping moved to loss function as `bounds_penalty` (soft, not hard)
- Optimizer naturally learns to respect bounds without hard wall

### Fix #2: Increase Calibration Weights 5×
**File**: `configs/enhanced_training.yaml` lines 123-128

**Changes**:
```yaml
# BEFORE (Nov 15)
calib_mean_weight: 0.30
calib_max_weight: 0.50
calib_range_weight: 0.40
mse_weight: 0.20

# AFTER (Nov 19)
calib_mean_weight: 0.50     # 1.67× increase
calib_max_weight: 2.50      # 5× increase (CRITICAL)
calib_range_weight: 2.00    # 5× increase (CRITICAL)
mse_weight: 0.05            # 4× decrease
```

**Weight balance now**:
- MSE gradient ≈ 0.05 × 2 × error ≈ 0.005
- Calibration gradient ≈ 2.50 × 0.193 ≈ 0.48
- **Ratio**: Calibration now 100× stronger!

**Loss formula impact**:
```python
total_loss = 0.05 * mse_loss
           + 0.70 * ranking_loss
           + 0.10 * uncertainty_loss
           + 2.50 * max_gap_penalty        # ← DOMINANT force now
           + 2.00 * range_penalty          # ← STRONG variance penalty
           + 5.00 * bounds_penalty         # ← SOFT constraint (without clipping)
```

### Fix #3: Add Affine Parameter Logging
**Files**: 
- `src/ahsd/core/priority_net.py` lines 1695-1700 (trainer)
- `experiments/train_priority_net.py` lines 1824-1825 (display)

**What's tracked**:
- `affine_gain`: Should trend from init toward 2.0-2.5 as range expands
- `affine_bias`: Should adjust to shift ceiling/floor appropriately

**Expected log**:
```
Epoch 1: Gain=1.80, Bias=-0.05  (just initialized)
Epoch 5: Gain=2.15, Bias=-0.08  (gaining capacity)
Epoch 10: Gain=2.35, Bias=-0.12 (approaching max bounds)
Epoch 20: Gain=2.40, Bias=-0.15 (saturated, supporting 0.95 ceiling)
```

**If Gain stays at 1.2-1.4**: Problem persists (needs further investigation)
**If Gain increases to 2.3+**: Fix is working (predictions expanding)

---

## Expected Training Trajectory

### Epoch 1-5: Rapid Expansion
```
Epoch 1:
  Predictions: [0.10, 0.55]  (starting range)
  Gain=1.80
  
Epoch 3:
  Predictions: [0.08, 0.82]  (expanding!)
  Gain=2.20, Bias=-0.10
  MAE starting to drop as range increases
  
Epoch 5:
  Predictions: [0.05, 0.88]  (nearly there)
  Gain=2.35, Bias=-0.13
```

### Epoch 10-20: Convergence
```
Epoch 10:
  Predictions: [0.06, 0.92]  (90%+ expansion)
  Gain=2.40, Bias=-0.15
  MAE ≈ 0.02-0.03
  Spearman correlation maintained >0.85
  
Epoch 20:
  Predictions: [0.07, 0.93]  (stable full range)
  Gain=2.41, Bias=-0.15 (saturated at bounds)
  MAE ≈ 0.015-0.020
  Ready for validation
```

### Epoch 50: Final State
- Predictions: [0.08, 0.95] (100% expansion achieved)
- Range compression: 100% (vs 82% before)
- MAE: 0.01-0.015 (vs 0.08-0.09 before)
- Gain: 2.5 (at max bound)
- Uncertainty correlation: >0.30

---

## Validation Checklist

Before considering fix successful:

- [ ] **Affine gain increases** from ~1.8 to 2.3+ by epoch 5
- [ ] **Prediction max reaches** 0.90+ by epoch 10 (vs 0.767 before)
- [ ] **Prediction range** increases to >0.85 (vs 0.69 before)
- [ ] **MAE drops** to <0.03 by epoch 10 (vs 0.08 before)
- [ ] **Spearman >0.85** maintained throughout
- [ ] **Ranking loss** stays stable (0.001-0.01)
- [ ] **No gradient explosion** (Grad < 2.0)
- [ ] **Hard bounds penalty** used softly (no clipping artifacts)

---

## Why This Fix Works

### Physics of Range Expansion
```
Goal: pred_max → 0.95, pred_min stays ~0.08
Current: pred_max = 0.767 (stuck)

Mechanism:
1. Remove hard clipping → allows gradient flow beyond 1.0
2. Increase calib_max weight → strong penalty for underexpansion
3. Reduce mse_weight → stop pulling toward safe zone
4. Result: Unconstrained output 1.15 → clamped softly to 0.95 by loss
```

### Why Soft Constraint (Loss) > Hard Constraint (Clipping)
```
Hard clipping:
  loss = f(clamp(affine_output))
  ∂loss/∂gain = 0 when clamped  (no learning)
  
Soft penalty:
  loss = f(affine_output) + penalty_if_out_of_bounds
  ∂loss/∂gain ≠ 0 always  (continuous learning)
  → Optimizer expands until penalty outweighs benefit
```

---

## Monitoring During Training

Watch for these failure modes:

### Failure Mode 1: Gain Doesn't Increase
```
Epoch 5: Gain=1.80, Epoch 10: Gain=1.81
→ Affine parameters not training
→ Check: learning_rate too low? Bounds too restrictive?
```

### Failure Mode 2: Predictions Overshoot
```
Predictions: [−0.5, 1.5]  (out of bounds)
bounds_penalty high (>10.0)
→ Penalty working correctly; model will learn to constrain
→ OK if transient, fix if persistent after epoch 10
```

### Failure Mode 3: Loss Oscillates Wildly
```
Epoch 1: Loss=0.15, Epoch 2: Loss=0.35, Epoch 3: Loss=0.08
→ Calibration weight 2.5 might be too aggressive
→ Reduce to 1.5-2.0 if unstable
```

---

## Rollback Plan

If range expansion causes issues:
1. Revert `calib_max_weight: 2.50 → 1.0`
2. Revert `mse_weight: 0.05 → 0.15`
3. Re-add hard clipping: `prio = torch.clamp(prio, 0, 1)`
4. Retrain from checkpoint

But this would return to 82% compression. Better: tune weights incrementally if needed.

---

## Code Changes Summary

| File | Lines | Change | Impact |
|------|-------|--------|--------|
| `priority_net.py` | 566 | Remove hard clipping | Allows gradient flow for expansion |
| `priority_net.py` | 1695-1700 | Add affine logging | Visibility into gain/bias training |
| `enhanced_training.yaml` | 92 | mse_weight: 0.20→0.05 | Reduces shrinkage pressure |
| `enhanced_training.yaml` | 126-128 | calib weights 5× increase | Dominant expansion signal |
| `train_priority_net.py` | 1824-1825 | Log Gain/Bias in postfix | Real-time monitoring |

---

## Commit Message

```
fix: Remove hard clipping to enable range expansion in PriorityNet

Issue: Output predictions compressed to 69% of target range (0.08-0.77 vs 0.11-0.95)
Root cause: Hard clipping at line 566 killed gradient signal for affine gain expansion
Solution: Replace hard clipping with soft bounds penalty in loss function

Changes:
- Remove torch.clamp() from forward pass (line 566)
- Increase calib_max_weight 0.50→2.50 (5× dominant expansion signal)
- Decrease mse_weight 0.20→0.05 (reduce shrinkage pressure)
- Add affine gain/bias logging for visibility

Expected: Predictions expand to full [0,1] range by epoch 20
Monitoring: Track affine_gain (should reach 2.3+) and pred_max (should reach 0.95)

This is the critical fix for Block 4 (Calibration) test failure
```

