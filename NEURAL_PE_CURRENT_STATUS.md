# Neural PE Training - Current Status (Nov 13, 2025)

## Overview
Training the OverlapNeuralPE model for joint posterior estimation of overlapping gravitational wave signals.

## Latest Status

### ✅ Fixes Completed Today

1. **Weight Reading Logic** (morning)
   - Fixed config reading to prioritize `neural_posterior` section
   - Set correct defaults: physics_loss_weight=0.05, bounds_penalty=0.5, sample_loss=0.5

2. **Physics Loss Rebalancing** (09:55 AM)
   - Reduced physics_loss_weight: 1.0 → 0.05 (soft constraint)
   - Increased bounds_penalty_weight: 0.1 → 0.5
   - Increased sample_loss_weight: 0.1 → 2.0 (then later 0.5)
   - Result: Total loss dropped from 27580 to ~1390

3. **Parameter Bounds Fixes** (13:00 PM)
   - geocent_time: [-0.1, 0.1] → [-2.0, 8.0]s (covers overlapping signal spacing i*1.5)
   - luminosity_distance: [20, 8000] → [10, 8000] Mpc (allows rare nearby events)
   - Result: Eliminated spurious physics penalty violations on edge cases

4. **Physics Loss - First Signal Only** (10:30 AM)
   - Restrict physics loss to first signal (ground truth)
   - Secondary signals in overlaps are intentionally out-of-bounds
   - Added debug logging for parameter violation detection
   - Expected: Physics loss raw ~2-10 instead of 27568

### 🔍 Current Issues Remaining

1. **NLL Still High** (after fixes)
   - Train NLL: 12.1 bits (target: 1-3 bits)
   - Diagnosis: Weights were reading as defaults despite config; now fixed
   - Action: Re-run training with corrected weight reading

2. **Train-Val Gap**
   - Before fixes: 27571 (massive)
   - After weight fixes: ~1388 (still high)
   - Expected after physics loss first-signal fix: <10

## Configuration

**File:** `configs/enhanced_training.yaml`

### Loss Weights
```yaml
neural_posterior:
  physics_loss_weight: 0.05           # Soft physics constraint
  bounds_penalty_weight: 0.5          # Hard ground truth protection
  sample_loss_weight: 0.5             # Flow output regularization
```

### Parameter Bounds
```yaml
param_bounds:
  mass_1: [1.0, 100.0]
  mass_2: [1.0, 100.0]
  distance: [10.0, 8000.0]           # ✅ Fixed: 20 → 10 Mpc
  geocent_time: [-2.0, 8.0]          # ✅ Fixed: [-0.1, 0.1] → [-2.0, 8.0]s
  # ... other params
```

## Loss Function Architecture

```
Total Loss = NLL + Jacobian Reg + Physics Loss (weighted) + Bias Loss + Uncertainty Loss + Sample Loss

Where:
- NLL: Flow likelihood (main objective)
- Physics Loss: Bounds violations + mass ordering + spin bounds
- Sample Loss: Penalize flow samples outside [-1, 1] range
- Bounds Penalty (in Physics): Quadratic penalty for parameter violations
```

## Next Steps

1. **Run Diagnostic Training** (5-10 epochs)
   ```bash
   python experiments/phase3a_neural_pe.py \
     --config configs/enhanced_training.yaml \
     --epochs 10 \
     --log_level DEBUG
   ```
   Expected: NLL → 3-5 bits, total loss → 5-10

2. **Monitor Batch 0 Losses**
   - Physics loss (raw): Should be 1-10, not 27568
   - Parameter violations: Should show zeros for clean first signal
   - NLL: Should dominate loss signal (>50% of total)

3. **If NLL still high:**
   - Check if weights are being read from config (log at startup)
   - Verify normalization bounds match physical bounds
   - Inspect flow.log_prob() values for ground truth samples

4. **Long-term Training** (once diagnostics pass)
   ```bash
   python experiments/phase3a_neural_pe.py \
     --config configs/enhanced_training.yaml \
     --epochs 100
   ```

## Key Commands

```bash
# Activate environment
source /home/bibinathomas/miniconda3/etc/profile.d/conda.sh && conda activate ahsd

# Install latest changes
pip install -e . --no-deps

# Run training with debug output
python experiments/phase3a_neural_pe.py \
  --config configs/enhanced_training.yaml \
  --output_dir outputs/neural_pe_debug \
  --epochs 5 \
  --log_level DEBUG

# Monitor specific logs
tail -f outputs/neural_pe_debug/training.log | grep "BATCH 0"
```

## Performance Targets

| Metric | Current | Target | Epoch |
|--------|---------|--------|-------|
| NLL (bits) | 12.1 | 2-3 | 15 |
| Physics Loss (raw) | 27568 | 1-10 | 1 |
| Total Loss | 1390 | <10 | 50 |
| Train-Val Gap | 1388 | <1 | 50 |
| Sample Loss | ? | <0.5 | 50 |

## Related Documentation

- `FIX_DOCS/PHYSICS_LOSS_FIRST_SIGNAL_FIX.md` - Latest fix details
- `FIX_DOCS/NLL_EXPLOSION_ROOT_CAUSE.md` - Loss analysis
- `FIX_DOCS/GEOCENT_TIME_BOUNDS_FIX.md` - Parameter bounds fixes
- `NEURAL_PE_GUIDE.md` - Architecture overview

## Code Files Modified Today

1. `src/ahsd/models/overlap_neuralpe.py`
   - Lines 768, 814, 859: Weight reading fixes
   - Line 765: Physics loss first-signal-only
   - Line 908: Return violations dict

2. `configs/enhanced_training.yaml`
   - Lines 142-146: Loss weight rebalancing
   - Lines 114-115: Parameter bounds fixes

3. `experiments/phase3a_neural_pe.py`
   - Lines 433-442: Debug logging for violations

## Debugging Tips

If physics loss is still high after training:
```python
# In phase3a_neural_pe.py, add at epoch 0, batch 0:
physics_violations = loss_dict.get('physics_violations', {})
for param, viol in physics_violations.items():
    if viol['lower'] > 0 or viol['upper'] > 0:
        print(f"⚠️ {param}: {viol}")
```

If NLL doesn't improve:
```python
# Check flow likelihood on ground truth
log_prob = model.flow.log_prob(true_params_norm, context)
print(f"Log prob: {log_prob.mean():.4f}")  # Should be > -5.0 (not -50.0)
```
