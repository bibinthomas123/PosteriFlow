# Neural PE Training - Quick Start Guide

## What's Changed (Nov 13, 2025)

Three major fixes applied to stabilize training:

### 1️⃣ Config Reading (fixed weights not loading)
- Physics loss now reads from `configs/enhanced_training.yaml`
- Correct defaults: physics=0.05, bounds=0.5, sample=0.5

### 2️⃣ Loss Weight Rebalancing 
- Reduced physics loss dominance (was 99.8% of total)
- Increased sample loss to constrain flow outputs
- Result: Better NLL optimization

### 3️⃣ Physics Loss - First Signal Only
- Secondary signals in overlaps are edge cases
- Physics constraints now apply to first signal only
- Added debug logging for parameter violations

## Quick Start

### 1. Set up environment
```bash
source /home/bibinathomas/miniconda3/etc/profile.d/conda.sh
conda activate ahsd
pip install -e . --no-deps
```

### 2. Run diagnostic training (5-10 epochs)
```bash
cd /home/bibinathomas/PosteriFlow
python experiments/phase3a_neural_pe.py \
  --config configs/enhanced_training.yaml \
  --output_dir outputs/neural_pe_v1 \
  --epochs 10 \
  --batch_size 32 \
  --log_level DEBUG
```

### 3. Monitor logs
```bash
# In another terminal
tail -f outputs/neural_pe_v1/training.log | grep "BATCH 0"
```

### 4. Expected results (Epoch 1)
```
[BATCH 0 LOSS BREAKDOWN - Epoch 1]
  Total Loss: ~12.0000
  NLL: ~10.0000
  Physics Loss (raw): ~2.0000 × 0.05 = 0.1000
  Sample Loss (raw): ~1.0000 × 0.5 = 0.5000
  
  [PARAMETER VIOLATIONS]
    mass_1: Range: [8.5, 56.3], Lower: 0, Upper: 0
    mass_2: Range: [5.2, 35.8], Lower: 0, Upper: 0
    (All should have 0 violations)
```

## What to Check

✅ **Good signs:**
- NLL decreases each epoch (target: 12 → 8 → 5 → 3 bits)
- Physics loss raw stays 1-10 (not 27568)
- No parameter violations logged
- Train-Val gap closes (<5 by epoch 10)

❌ **Red flags:**
- Physics loss still 27000+: Ground truth bounds issue
- NLL increases: Learning rate too high
- Parameter violations: Bounds need adjustment
- GPU memory explosion: Batch size too large

## Loss Interpretation

```
Total Loss = NLL + Jacobian Reg + 0.05*Physics + Bias + Uncertainty + Sample Loss

- NLL (main loss): Should dominate (>50% of total)
- Physics Loss: Should be tiny after weighting (0.05x)
- Sample Loss: Keeps flow outputs bounded
```

## Typical Training Progression

| Epoch | NLL | Physics (raw) | Total | Notes |
|-------|-----|---------------|-------|-------|
| 1 | 12.0 | 2.0 | 12.3 | First signal clean, physics loss tiny |
| 5 | 8.0 | 0.5 | 8.5 | NLL improving, sample loss working |
| 10 | 5.0 | 0.3 | 5.4 | Convergence visible |
| 50 | 2.5 | 0.1 | 3.0 | Target range reached |

## Configuration File

**Location:** `configs/enhanced_training.yaml`

Key sections:
```yaml
# Loss weights (total components)
neural_posterior:
  physics_loss_weight: 0.05           # Soft physics
  bounds_penalty_weight: 0.5          # Hard ground truth
  sample_loss_weight: 0.5             # Flow regularization

# Parameter bounds (prevents out-of-range predictions)
param_bounds:
  mass_1: [1.0, 100.0]
  geocent_time: [-2.0, 8.0]          # Fixed for overlaps
  luminosity_distance: [10.0, 8000.0] # Fixed for edge cases
```

## Troubleshooting

### Physics loss still high (>100)?
1. Check ground truth labels: `python -c "import pickle; d=pickle.load(open('data/test/train/chunk_0000.pkl','rb')); print([s[0] for s in d[:3]])"` 
2. Verify normalization doesn't change parameter ranges
3. Check `param_bounds` match actual data

### NLL not improving?
1. Verify config is loaded: Look for "Loading enhanced_training.yaml" in logs
2. Check learning rate: Default 1e-3 may be too aggressive
3. Inspect `flow.log_prob()` values: Should be -5 to 5, not -100

### GPU memory errors?
1. Reduce batch_size: `--batch_size 16` instead of 32
2. Reduce context window: Edit `ContextEncoder` input size
3. Check for data leaks: Gradients shouldn't accumulate between batches

## Advanced: Custom Loss Weights

Edit `configs/enhanced_training.yaml`:
```yaml
neural_posterior:
  physics_loss_weight: 0.02      # More exploration, less constraint
  bounds_penalty_weight: 1.0     # Strict ground truth
  sample_loss_weight: 1.0        # Strong output regularization
```

Then train:
```bash
python experiments/phase3a_neural_pe.py \
  --config configs/enhanced_training.yaml \
  --custom_tag "test_weights_v1"
```

## Files You Need

- `configs/enhanced_training.yaml` - Main config
- `src/ahsd/models/overlap_neuralpe.py` - Model definition
- `experiments/phase3a_neural_pe.py` - Training script
- `data/test/train/` - Training data

## Next Steps

1. **Run diagnostic (now)**
   - 10 epochs, watch epoch 1-3 metrics
   - Takes ~15-30 minutes on GPU

2. **Analyze results (after diagnostic)**
   - If NLL → 3 bits: Ready for production training
   - If NLL plateaus: Adjust weights, re-run

3. **Production training (once validated)**
   - 100+ epochs
   - Save checkpoints every 10 epochs
   - Monitor val loss for early stopping

## Support

See detailed docs:
- `FIX_DOCS/PHYSICS_LOSS_FIRST_SIGNAL_FIX.md` - Why first-signal-only
- `FIX_DOCS/GEOCENT_TIME_BOUNDS_FIX.md` - Parameter bounds
- `FIX_DOCS/NLL_EXPLOSION_ROOT_CAUSE.md` - Loss analysis
- `NEURAL_PE_GUIDE.md` - Architecture details
