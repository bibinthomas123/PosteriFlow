# Prediction Compression Fix - Quick Start Guide

## What Was Fixed?
Model predictions were stuck in range (0.297, 0.471) — only 25% of the target range (0.263, 0.950).

**Root Causes:**
- Calibration penalties too weak (0.05 each, should be 0.15 and 0.40)
- Output layer bias initialized to 0.2 instead of 0.5 (midpoint)
- Weight initialization std too small (0.01 instead of 0.05)
- No explicit range regularization penalty
- Affine parameters (gain/bias) unconstrained during training

## What Changed?

### Config Changes
Added 6 parameters to `configs/enhanced_training.yaml` under `priority_net`:
```yaml
calib_mean_weight: 0.15           # 3× stronger than before
calib_max_weight: 0.40            # 8× stronger than before
output_bias_init: 0.5             # Midpoint instead of 0.2
output_weight_std: 0.05           # Doubled from 0.01
log_affine_params: true           # Enable monitoring
clamp_affine_gain: [0.7, 1.5]    # Prevent compression
```

### Code Changes
6 strategic changes in `src/ahsd/core/priority_net.py`:
1. **PriorityLoss.__init__:** Accept configurable calibration weights
2. **PriorityLoss.forward():** Use strong penalties + new range loss
3. **PriorityNet.__init__:** Load bias/std from config
4. **PriorityNet.forward():** Clamp affine parameters
5. **TrainerForPriorityNet:** Pass config to loss function

## Expected Results

| Metric | Before | After (by epoch 30) | Improvement |
|--------|--------|---|---|
| Prediction range | 0.174 | ≥0.50 | **+287%** |
| Max gap | 0.687 | <0.10 | **-93%** |
| Compression ratio | 25% | ≥73% | **+192%** |

## Launch Training

### Step 1: Verify Setup
```bash
bash START_TRAINING_NOW.sh
```

### Step 2: Start Training
```bash
nohup python experiments/train_priority_net.py \
  --config configs/enhanced_training.yaml \
  --create_overlaps > nohup.out 2>&1 &
```

### Step 3: Monitor Progress
```bash
# Watch live logs (Ctrl+C to exit)
tail -f nohup.out | grep -E "Epoch|val_loss|Calib gaps|Affine"

# Or check specific metrics
tail -20 nohup.out | grep "Max gap"
tail -20 nohup.out | grep "Affine:"
```

## Monitoring Checklist

### Epoch 5 (First Check)
- [ ] No errors in logs
- [ ] Prediction range expanding toward 0.30
- [ ] Calibration loss visible in loss breakdown

### Epoch 15 (Mid-Check)
- [ ] Prediction range ≥ 0.45
- [ ] Max gap ≤ 0.20
- [ ] Validation loss still decreasing

### Epoch 30 (Success Check)
- [ ] Prediction range ≥ 0.60
- [ ] **Max gap < 0.10** ✓ (PRIMARY GOAL)
- [ ] Mean gap < 0.05
- [ ] Training converged (loss plateau)

## Quick Debugging

### Problem: Range Not Expanding
```bash
# Check config loaded correctly
grep "calib_max_weight" nohup.out

# If missing, config not loaded - check configs/enhanced_training.yaml
# If stuck, try increasing calib_max_weight to 0.50
```

### Problem: Affine Params Hitting Bounds
**This is normal!** Means gradient flow is working. Verify:
```bash
grep "Affine:" nohup.out | tail -10
# Should show gain between 0.7-1.5, bias between -0.1-0.1
```

### Problem: Loss Exploding
```bash
# Reduce calibration weights temporarily
# Edit configs/enhanced_training.yaml:
#   calib_mean_weight: 0.10  (from 0.15)
#   calib_max_weight: 0.20   (from 0.40)
```

## Files to Monitor

- **Training log:** `nohup.out`
- **Config:** `configs/enhanced_training.yaml` (lines 59-71)
- **Model checkpoint:** `models/` directory
- **Metrics:** Check `outputs/` after training

## Documentation

### Complete Guides
- `FIX_DOCS/PREDICTION_COMPRESSION_FIX.md` — Full technical details
- `COMPRESSION_FIX_SUMMARY.md` — Implementation overview
- `MONITORING_CHECKLIST.md` — Detailed epoch-by-epoch guide
- `CALIBRATION_PENALTY_ANALYSIS.md` — Loss component math

### Quick References
- `PRIORITY_NET_CONFIG_QUICK_REFERENCE.md` — All config options
- `START_TRAINING_NOW.sh` — Automated setup script

## Success Criteria (All Must Pass)
1. ✓ No training errors
2. ✓ Prediction range ≥ 0.50
3. ✓ Max gap < 0.10
4. ✓ Validation loss decreasing
5. ✓ MAE improving

If all 5 criteria met by epoch 40: **FIX SUCCESSFUL**

## Next Steps After Training

1. Evaluate calibration on test set
2. Compare metrics with baseline (pre-Nov 12)
3. Test on real GWOSC events
4. Analyze prediction distribution (histograms)
5. Deploy to production if metrics improved

## Support

If training gets stuck:
1. Check `MONITORING_CHECKLIST.md` for your specific issue
2. Review `CALIBRATION_PENALTY_ANALYSIS.md` for loss behavior
3. Refer to `COMPRESSION_FIX_SUMMARY.md` for implementation details

---

**Remember:** This is a calibration fix. Training may take 30-50 epochs to fully converge. Be patient and monitor the progression in the checklist.
