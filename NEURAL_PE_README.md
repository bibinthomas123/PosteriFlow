# Neural PE Training - Quick Reference

## Latest Status (Nov 13, 2025)

✅ **All critical fixes applied and verified**

- Config weights now load correctly
- Physics loss balanced (0.05 weight)
- First signal only constraint in place
- Ready for diagnostic training

## Quick Start

```bash
# Setup
cd /home/bibinathomas/PosteriFlow
source /home/bibinathomas/miniconda3/etc/profile.d/conda.sh
conda activate ahsd
pip install -e . --no-deps

# Run training
python experiments/phase3a_neural_pe.py \
  --config configs/enhanced_training.yaml \
  --output_dir outputs/neural_pe_v1 \
  --epochs 10
```

## What Changed?

### Fix 1: Config Reading
- Weights now read from `configs/enhanced_training.yaml`
- Priority: `neural_posterior` → top-level → defaults

### Fix 2: Loss Balance
Physics loss weight: 1.0 → 0.05 (soft constraint)
- Before: Physics = 99.8% of total loss
- After: Physics = ~2-3% of total loss

### Fix 3: Physics Scope
Physics loss now applies to first signal only
- Secondary signals are edge cases (no constraints)
- Ground truth gets proper physics penalties

## Expected Results

**Epoch 1:**
```
Total Loss: ~12.0
NLL: ~10.0 bits
Physics Loss (raw): ~2.0 × 0.05 = 0.1
```

**Epoch 10:**
```
Total Loss: ~5.0
NLL: ~3.5 bits
Physics Loss (raw): ~0.2 × 0.05 = 0.01
```

**Epoch 50:**
```
Total Loss: <3.0
NLL: 2-3 bits (TARGET)
Physics Loss (raw): ~0.1 × 0.05 ≈ 0.005
```

## Key Files

| File | Purpose |
|------|---------|
| `configs/enhanced_training.yaml` | Training config, loss weights |
| `src/ahsd/models/overlap_neuralpe.py` | Model code (physics loss fixed) |
| `experiments/phase3a_neural_pe.py` | Training script (debug logging) |
| `FIX_DOCS/PHYSICS_LOSS_FIRST_SIGNAL_FIX.md` | Detailed fix explanation |
| `FIX_DOCS/NEURAL_PE_FIXES_SUMMARY_NOV13.md` | Full summary |

## Monitoring

Check logs during training:
```bash
tail -f outputs/neural_pe_v1/training.log | grep "EPOCH\|LOSS\|BATCH 0"
```

Look for:
- NLL decreasing each epoch
- Physics loss staying low (<10 raw)
- No parameter violations in first signal
- Total loss smooth curve

## If Something Goes Wrong

**Physics loss still 27,568?**
- Verify `true_params[:1, :]` in line 765
- Check Python restarted (old bytecode issue)

**NLL not improving?**
- Verify config loaded (check startup log)
- Try smaller learning rate (1e-4)

**Parameter violations shown?**
- First signal shouldn't have any
- Check dataset ground truth bounds

## Advanced Tuning

Edit `configs/enhanced_training.yaml`:
```yaml
neural_posterior:
  physics_loss_weight: 0.02       # Less strict
  sample_loss_weight: 1.0         # More flow regularization
```

## Success Checklist

Before calling training "done":
- [ ] Epoch 1 NLL < 15 bits
- [ ] Epoch 10 NLL < 5 bits
- [ ] Epoch 50 NLL in [2, 3] bits
- [ ] No NaN/Inf in loss components
- [ ] Train-Val gap < 1
- [ ] Parameter violations = 0 for first signal

## Related Documentation

- **Quick Start:** `NEURAL_PE_TRAINING_QUICK_START.md`
- **Current Status:** `NEURAL_PE_CURRENT_STATUS.md`
- **Full Summary:** `FIX_DOCS/NEURAL_PE_FIXES_SUMMARY_NOV13.md`
- **Physics Loss Details:** `FIX_DOCS/PHYSICS_LOSS_FIRST_SIGNAL_FIX.md`
- **Architecture:** `NEURAL_PE_GUIDE.md`

## Support

All fixes documented in `FIX_DOCS/` folder. Check there for:
- Detailed explanations
- Code diffs
- Verification procedures
- Expected performance curves
