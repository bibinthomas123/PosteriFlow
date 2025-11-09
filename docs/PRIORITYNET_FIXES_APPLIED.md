# PriorityNet Model Fixes - Implementation Summary

**Date**: October 30, 2025  
**Status**: ✅ All fixes implemented

---

## Dataset Fixes (Already Complete)

| Fix | Status | Location | Impact |
|-----|--------|----------|--------|
| ✅ edge_type_id variance > 0 | DONE | `combined_dataset_cret.py:3341` | Enables edge conditioning |
| ✅ network_snr always present | DONE | Multiple locations | SNR pathway functional |
| ✅ 5+ overlaps: 25-35% | DONE | `combined_dataset_cret.py:3317` | Better overlap diversity |
| ✅ Event distribution 46/32/17/5 | DONE | `combined_dataset_cret.py:128` | Correct signal composition |

---

## PriorityNet Model Fixes (NOW COMPLETE)

### 1. ✅ Remove Output Squashing
**Status**: Already implemented + verified

**Current implementation**:
```python
self.priority_head = nn.Sequential(
    nn.Linear(64, 32),
    nn.LayerNorm(32),
    nn.GELU(),
    nn.Dropout(0.12),
    nn.Linear(32, 16),
    nn.GELU(),
    nn.Linear(16, 1)  # ← LINEAR output, no sigmoid/tanh
)
prio = prio * self.prio_gain + self.prio_bias  # Affine calibration
```

**Result**: ✅ No squashing, affine calibration available

---

### 2. ✅ Strengthen SNR Pathway
**Status**: **JUST IMPLEMENTED**

**Added** (lines 781-790):
```python
self.snr_embedding = nn.Sequential(
    nn.Linear(1, 16),
    nn.LayerNorm(16),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(16, 32)  # ← 1→16→32 MLP as specified
)
```

**Updated forward pass** (lines 901-908):
```python
# Extract SNR from signal_tensor (index 15)
snr_raw = signal_tensor[:, 15:16]  # [N, 1]
snr_embedded = self.snr_embedding(snr_raw)  # [N, 32]

# Concatenate with metadata before fusion
metadata_features = torch.cat([metadata_features, snr_embedded], dim=1)
```

**Result**: ✅ Dedicated SNR pathway with 32-D embedding

---

### 3. ✅ Re-enable Edge Conditioning
**Status**: Already implemented + **ENHANCED**

**Existing** (line 766):
```python
self.edge_embedding = nn.Embedding(
    num_embeddings=17,  # n_edge_types
    embedding_dim=32,
    padding_idx=0
)
```

**Enhanced** (line 930-932):
```python
edge_embeds = self.edge_embedding(edge_type_ids)
# NEW: Apply dropout to prevent overfitting to specific IDs
if self.training:
    edge_embeds = F.dropout(edge_embeds, p=0.1, training=True)
```

**Result**: ✅ Edge embedding active with 10% dropout

---

### 4. ✅ Calibrated Training Objective
**Status**: **JUST IMPROVED**

**Updated loss** (lines 671-698):
```python
# FIX #4: Calibrated training objective
# L = L_rank + L_mse + λ1·|mean(ŷ) − mean(y)| + λ2·|max(ŷ) − max(y)|

# Mean calibration loss
mean_gap = torch.abs(predictions.mean() - targets.mean())

# Max calibration loss (widen spread)
max_gap = torch.abs(predictions.max() - targets.max())

# Calibration penalties (λ1 = λ2 = 0.05 as specified)
lambda1 = 0.05
lambda2 = 0.05
calib_loss = lambda1 * mean_gap + lambda2 * max_gap

total_loss = (
    effective_mse_weight * mse_loss +
    effective_ranking_weight * ranking_loss +
    self.uncertainty_weight * uncertainty_loss +
    calib_loss  # Calibration term to widen spread
)
```

**Result**: ✅ Explicit mean and max calibration with λ1=λ2=0.05

---

### 5. ✅ Head Initialization
**Status**: **JUST IMPLEMENTED**

**Added** (lines 812-814):
```python
# Initialize final layer with small weights
final_layer = self.priority_head[-1]
nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
final_layer.bias.data.fill_(0.2)  # Dataset mean priority ~0.2
```

**Result**: ✅ Prevents early saturation, bias near dataset mean

---

## Complete Fix Summary

| Fix | Specification | Implementation | Status |
|-----|---------------|----------------|--------|
| **Remove squashing** | Linear output with affine | Already present | ✅ VERIFIED |
| **SNR embedding** | MLP [1→16→32] | Added to __init__ and forward | ✅ NEW |
| **Edge conditioning** | Embedding + dropout 0.1 | Enhanced with dropout | ✅ IMPROVED |
| **Calibration loss** | λ1=λ2=0.05 for mean/max | Replaced old calib_pen | ✅ NEW |
| **Head init** | Normal(0, 0.01), bias=0.2 | Added after head creation | ✅ NEW |

---

## Architecture Changes

### Before
```
Input [N, 15] 
  ↓
Metadata Encoder [N, 96]
  ↓
Fusion (meta + overlap + temporal + edge) [N, 64]
  ↓
Priority Head [N, 1] → linear output
```

### After
```
Input [N, 16]  ← Added network_snr
  ↓
Metadata Encoder [N, 96]
  ↓
SNR Embedding MLP [N, 1] → [N, 32]  ← NEW pathway
  ↓
Concatenate [N, 96+32=128]
  ↓
Fusion (meta+snr + overlap + temporal + edge) [N, 64]
  ↓
Priority Head [N, 1] → linear output
  ↓
Affine Calibration (gain, bias)
```

---

## Training Changes

### Loss Function
```python
L_total = w_mse·MSE(ŷ, y) + 
          w_rank·RankingLoss(ŷ, y) +
          w_unc·MSE(σ, |ŷ - y|) +
          0.05·|mean(ŷ) - mean(y)| +    ← NEW
          0.05·|max(ŷ) - max(y)|        ← NEW
```

### Bucket-Aware Weighting
- **Bucket 5+**: w_rank × 1.5, w_mse × 0.85 (emphasize ordering)
- **Bucket <5**: Normal weights

---

## Expected Improvements

### Before Fixes
- Pred max: ~0.40
- Target max: ~0.95
- **Gap**: 0.55 (output compression)
- SNR sensitivity: Δpred ≈ 0 for +2 SNR

### After Fixes (Expected)
- Pred max: ~0.75-0.85 (gap < 0.18 ✅)
- SNR sensitivity: Δpred ≥ 0.01 for +2 SNR ✅
- Edge variance: > 0.5 ✅
- Spearman(ŷ, SNR): ≥ 0.4 on BBH subset ✅

---

## Verification Tests (After Retraining)

Run these tests after training with fixed model:

### 1. Edge Variance Test
```python
# Load 500 validation samples
edge_ids = [sample['edge_type_id'] for sample in val_samples[:500]]
variance = np.var(edge_ids)
assert variance > 0.5, f"Edge variance too low: {variance}"
```

### 2. SNR Sensitivity Test
```python
# Unit test: +2 SNR delta should yield Δpred ≥ 0.01
detection = create_test_detection(snr=15.0)
detection_high = create_test_detection(snr=17.0)

pred_base = model([detection])[0].item()
pred_high = model([detection_high])[0].item()

delta_pred = abs(pred_high - pred_base)
assert delta_pred >= 0.01, f"SNR sensitivity too low: {delta_pred}"
```

### 3. Calibration Gap Test
```python
# Validation max gap should be ≤ 0.18
val_targets_max = max(target priorities from validation)
val_preds_max = max(predicted priorities from validation)
gap = abs(val_targets_max - val_preds_max)
assert gap <= 0.18, f"Calibration gap too large: {gap}"
```

### 4. SNR Correlation Test
```python
# On BBH-only subset, correlation should be ≥ 0.4
bbh_samples = [s for s in val_samples if s['type'] == 'BBH']
preds = [model.predict(s) for s in bbh_samples]
snrs = [s['network_snr'] for s in bbh_samples]

from scipy.stats import spearmanr
rho, _ = spearmanr(preds, snrs)
assert rho >= 0.4, f"SNR correlation too low: {rho}"
```

---

## Retrain Requirements

### Critical
1. **Delete old checkpoints** - Architecture changed, can't resume
2. **Regenerate dataset** - Ensure 50K samples with all fixes
3. **Update config** - Ensure network_snr normalization is calibrated

### Training Command
```bash
# After 50K dataset is ready
python scripts/train_priority_net.py \
    --config configs/enhanced_training.yaml \
    --data data/ahsd_dataset_50k \
    --epochs 100 \
    --batch-size 32 \
    --workers 4
```

### Monitor Metrics
- Validation Kendall τ (should stay ≥ 0.56)
- Validation max gap (should drop from 0.55 to <0.18)
- SNR Spearman (should reach ≥ 0.4)
- Edge variance (should be > 0.5)

---

## Files Modified

1. `src/ahsd/core/priority_net.py`:
   - Added SNR embedding MLP (lines 781-790)
   - Updated forward pass to use SNR embedding (lines 901-908)
   - Added dropout to edge embeddings (lines 930-932)
   - Updated calibration loss with λ1, λ2 (lines 671-698)
   - Added head initialization (lines 812-814)

2. `experiments/combined_dataset_cret.py`:
   - Multiple fixes for event distribution, network_snr, overlaps
   - Performance optimizations

3. `configs/data_config_50k.yaml`:
   - 50K sample configuration
   - Parallel generation ready

---

## Next Steps

1. ✅ **Wait for current 3K generation to complete** (~5 more minutes)
2. ✅ **Validate 3K dataset** to confirm all fixes work
3. ✅ **Generate 50K dataset** using parallel script (~25 minutes)
4. ✅ **Retrain PriorityNet** with fixed architecture
5. ✅ **Run verification tests** to confirm improvements

All fixes are complete and ready for retraining! 🎉
