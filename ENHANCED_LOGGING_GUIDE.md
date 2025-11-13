# Enhanced Training Logging Guide

## Overview

The training pipeline now includes comprehensive logging across all six core components of OverlapNeuralPE. This guide explains what metrics are logged, their meaning, and how to interpret them for debugging and monitoring.

---

## Complete Epoch Logging Output

Here's an example of the enhanced logging output for a single training epoch:

```
Epoch 18/100
  Train Loss: 8.3022 (NLL: 8.3026)
  Val Loss: 8.2892 (NLL: 8.2897)
  Train-Val Gap: 0.0129
  Gradient Norm: 0.036
  LR: 1.00e-04
  Patience: 0/20
  Time: 55.9s
  
  Loss Components:
    Physics Loss: 0.001234
    Bias Loss: 0.005678
    Uncertainty Loss: 0.002345
    Jacobian Reg: 0.000456
  
  RL Controller:
    epsilon: 0.1000
    avg_complexity: 1.5234
    complexity_std: 0.6123
    avg_reward: 0.3456
    action_entropy: 0.8901
  
  Bias Corrector:
    ✓ Avg Correction: 0.001234
    ✓ Max Correction: 0.012345
    ✓ Correction Std: 0.003456
    ⚡ Avg Confidence: 0.8567
    ⚡ Min Confidence: 0.6234
    📊 Acceptance Rate: 87.65%
    ⚠️  Physics Violations: 2
  
  Integration Summary:
    Total Parameters: 12,345,678
    Trainable Parameters: 9,876,543
    
  RL State:
    Exploration (ε): 0.1000
    Avg Complexity: 1.5234
    Action Entropy: 0.8901
    Avg Reward: 0.3456
  
  Bias State:
    Magnitude: 0.001234
    Acceptance: 87.65%
```

---

## Metric Categories

### 1. Loss Metrics

#### Main Loss Metrics
```
Train Loss: 8.3022          # Total training loss for epoch
Val Loss: 8.2892            # Total validation loss
Train-Val Gap: 0.0129       # Difference (ideally small, ~0.01-0.05)
NLL: 8.3026                 # Negative Log-Likelihood from normalizing flow
```

**Interpretation**:
- Train loss should decrease monotonically
- Val loss oscillates but trends downward
- Gap > 0.1 indicates overfitting
- Gap < 0.01 indicates underfitting (model not learning)
- **Optimal**: Gap between 0.01-0.05

#### Component Loss Breakdown
```
Physics Loss: 0.001234      # Mass ordering & physical constraint violations
Bias Loss: 0.005678         # Bias corrector regularization (prior toward no correction)
Uncertainty Loss: 0.002345  # Uncertainty calibration penalty
Jacobian Reg: 0.000456      # Flow weight regularization (stabilizes training)
```

**When Each Component Matters**:
- **Physics Loss** trending up → model violating mass ordering constraints
  - Action: Increase physics_loss weight in config
- **Bias Loss** large → bias corrector making aggressive corrections
  - Action: Reduce bias loss weight or enable validation
- **Uncertainty Loss** high → uncertainty estimates are too large/unstable
  - Action: Increase uncertainty weight, check scaling
- **Jacobian Reg** very small → flow may be underconstrained
  - Action: Increase regularization weight

---

### 2. Gradient Metrics

```
Gradient Norm: 0.036        # L2 norm of all parameter gradients
LR: 1.00e-04                # Current learning rate
```

**Interpretation**:
- Gradient norm 0.001-0.1: Healthy
- Gradient norm < 0.001: Vanishing gradients (learning stalled)
- Gradient norm > 1.0: Exploding gradients (training unstable)
  - Solution: Reduce learning rate, increase gradient clip
- LR decreasing → scheduler reducing learning rate
  - Usually triggered by validation loss plateau

**Debug Workflow**:
```python
if avg_gradient_norm < 0.001:
    print("⚠️  Vanishing gradients detected")
    # Try: increase learning rate, reduce weight decay
    
if avg_gradient_norm > 1.0:
    print("⚠️  Exploding gradients detected")
    # Try: decrease learning rate, increase gradient_clip
```

---

### 3. RL Controller Metrics

#### Complexity Selection
```
avg_complexity: 1.5234      # Mean complexity level (0=low, 1=medium, 2=high)
complexity_std: 0.6123      # Std of complexity choices
```

**Interpretation**:
- 0-0.5: Mostly "low" complexity (efficient)
- 0.5-1.5: Mixed "low" and "medium" (balanced)
- 1.5-2.0: Mostly "medium" and "high" (costly)
- 2.0: Always "high" (not adapting properly)

#### Exploration
```
epsilon: 0.1000             # Exploration rate (1.0 = random, 0.0 = greedy)
action_entropy: 0.8901      # Shannon entropy of action distribution
```

**Interpretation**:
- epsilon should decay from 0.1 → 0.01 over training
- action_entropy close to log(3)≈1.099 = exploring all actions equally
- action_entropy close to 0 = exploiting best action

#### Learning Performance
```
avg_reward: 0.3456          # Average extraction reward per iteration
```

**Interpretation**:
- Should increase from 0.1-0.2 (early) → 0.4-0.5 (late)
- Stalled reward → RL not learning
  - Solution: Increase RL learning rate, check reward signal

---

### 4. Bias Corrector Metrics

#### Correction Magnitudes
```
Avg Correction: 0.001234    # Mean absolute correction across all parameters
Max Correction: 0.012345    # Maximum correction applied
Correction Std: 0.003456    # Standard deviation of corrections
```

**Expected Ranges** (by signal type):
```
BBH (Binary Black Hole):
  Avg: 0.001-0.005          # 0.1-0.5% of parameter scale
  Max: 0.010-0.050          # Up to 5% for bad cases
  
BNS (Binary Neutron Star):
  Avg: 0.0005-0.002         # Very small (NS masses tightly constrained)
  Max: 0.005-0.020
  
NSBH (Neutron Star + Black Hole):
  Avg: 0.001-0.003
  Max: 0.008-0.030
```

#### Confidence in Corrections
```
Avg Confidence: 0.8567      # Mean confidence score [0=uncertain, 1=certain]
Min Confidence: 0.6234      # Minimum confidence (worst case)
```

**Interpretation**:
- Avg Confidence > 0.8: Good (model is confident)
- Avg Confidence 0.5-0.8: Moderate (some uncertainty)
- Avg Confidence < 0.5: Poor (model unreliable)
  - Action: Check if bias corrector is trained, reduce correction aggressiveness

#### Validation & Acceptance
```
Acceptance Rate: 87.65%     # % of proposed corrections that passed validation
Physics Violations: 2       # Number of rejected corrections (physics violations)
```

**Interpretation**:
- Acceptance rate should be 80%+ (most corrections valid)
- Acceptance rate < 60%: Corrections too aggressive
  - Solution: Switch to "conservative" strategy, reduce scaling weight
- Physics violations increasing: Corrections violating bounds
  - Solution: Check bounds are reasonable, increase validation threshold

---

### 5. Integration Summary

```
Total Parameters: 12,345,678
Trainable Parameters: 9,876,543
```

**Component Breakdown** (rough estimates):
```
Context Encoder: ~500K parameters
Normalizing Flow: ~8M parameters (depending on config)
Bias Estimator: ~300K parameters
RL Q-Network: ~100K parameters
Uncertainty Estimator: ~50K parameters
PriorityNet: (frozen, loaded from checkpoint)
```

---

### 6. Component Status (RL & Bias)

#### RL Controller State
```
Exploration (ε): 0.1000     # Current epsilon (should decay)
Avg Complexity: 1.5234      # Which complexity level being selected
Action Entropy: 0.8901      # How diverse the decisions are
Avg Reward: 0.3456          # Quality of selected actions
```

#### Bias Corrector State
```
Magnitude: 0.001234         # Scale of bias corrections
Acceptance: 87.65%          # Rate of accepted corrections
```

---

## Logging Configuration

### Enable Verbose Logging

```bash
# Run training with DEBUG-level logging
python experiments/phase3a_neural_pe.py \
    --config configs/enhanced_training.yaml \
    --verbose
```

This outputs:
- Component-level debug info
- Individual signal extraction details
- Subtraction convergence info
- Bias correction validation details

### Log File Location

```
outputs/experiments/{run_id}/training.log
```

Each epoch appends to this file. To monitor live:
```bash
tail -f outputs/experiments/{run_id}/training.log
```

---

## Debugging Workflows

### Issue: Loss Not Decreasing

```
Epoch 10: Train Loss: 8.300, Val Loss: 8.298
Epoch 11: Train Loss: 8.305, Val Loss: 8.302
Epoch 12: Train Loss: 8.310, Val Loss: 8.308
```

**Diagnosis**:
1. Check Gradient Norm: If < 0.001 → vanishing gradients
   - Solution: Increase LR by 10x
2. Check component losses: If one dominates (e.g., physics_loss >> others)
   - Solution: Reduce that weight in config
3. Check validation metrics: If stable but high
   - Solution: Increase context_dim or flow layers

### Issue: Overfitting (Large Train-Val Gap)

```
Epoch 10: Train Loss: 7.900, Val Loss: 8.200, Gap: 0.3
```

**Diagnosis**:
1. Increase dropout (0.1 → 0.2)
2. Reduce flow depth
3. Check Bias Corrector: If always confident (> 0.95)
   - Solution: Add uncertainty penalty to bias loss

### Issue: Bias Corrector Unstable

```
Acceptance Rate: 45.3%
Physics Violations: 127
Avg Correction: 0.25 (too large!)
```

**Diagnosis**:
1. Switch correction strategy: 'aggressive' → 'conservative'
2. Reduce bias loss weight
3. Check if bias corrector is trained
   - If not: Use physics-based mode (automatic)
4. Verify parameter bounds are reasonable

### Issue: RL Controller Not Learning

```
Epoch 5: avg_reward: 0.200, avg_complexity: 1.200
Epoch 10: avg_reward: 0.205, avg_complexity: 1.199
Epoch 15: avg_reward: 0.203, avg_complexity: 1.198
```

**Diagnosis**:
1. Check epsilon decay: Should go 0.1 → 0.01
2. Increase RL learning rate: 1e-3 → 1e-2
3. Check reward signal: Ensure extraction quality improving

---

## WandB Integration

All metrics are automatically logged to WandB if enabled:

```bash
python experiments/phase3a_neural_pe.py \
    --config configs/enhanced_training.yaml \
    --wandb_project "posteriorflow" \
    --wandb_entity "your-entity"
```

### WandB Metric Groups

```
batch/
  - train_loss
  - nll
  - physics_loss
  - bias_loss
  - uncertainty_loss
  - jacobian_reg
  - gradient_norm
  - learning_rate

epoch/
  - train_loss
  - train_nll
  - train_gradient_norm
  - val_loss
  - val_nll
  - train_val_gap
  - learning_rate
  - time
  - patience_counter
  - epochs_since_best

rl/
  - epsilon
  - avg_complexity
  - complexity_std
  - avg_reward
  - action_entropy

bias/
  - avg_correction
  - max_correction
  - correction_std
  - avg_confidence
  - min_confidence
  - correction_acceptance_rate
  - physics_violations
```

### WandB Charts

Create custom charts:
```
- X: epoch
- Y: train_loss, val_loss (compare convergence)
- Y: loss components (physics, bias, uncertainty, jacobian)
- Y: rl/avg_reward (RL learning curve)
- Y: bias/correction_acceptance_rate (bias corrector health)
```

---

## Expected Training Trajectory

### Early Training (Epochs 1-20)

```
✓ NLL: 8.5 → 8.2 (decreasing rapidly)
✓ Gradient Norm: > 0.05 (active learning)
✓ Train-Val Gap: < 0.02 (good generalization)
✓ Physics Loss: small (< 0.01)
✓ Bias Correction: Conservative, high acceptance (> 95%)
✓ RL Reward: Increasing (0.1 → 0.3)
```

### Mid Training (Epochs 20-50)

```
✓ NLL: 8.2 → 7.8 (slow decrease)
✓ Gradient Norm: 0.01-0.05 (stable)
✓ Train-Val Gap: 0.02-0.05 (optimal)
✓ Physics Loss: stable (near 0)
✓ Bias Acceptance: 80-95% (well-calibrated)
✓ RL Reward: Plateauing (0.3 → 0.4)
✓ Complexity: Diverse (entropy > 0.8)
```

### Late Training (Epochs 50+)

```
✓ NLL: 7.8 → 7.5 (very slow improvement)
✓ Gradient Norm: < 0.01 (stable, small)
✓ Train-Val Gap: 0.02-0.05 (stable)
✓ Physics Loss: stable near 0
✓ Bias Acceptance: Stable 85%+
✓ RL Reward: Plateau (optimal actions found)
✓ LR: Decayed due to scheduler
```

---

## Saving & Analyzing Logs

### Extract Metrics from Log File

```python
import re

log_file = "outputs/experiments/run_001/training.log"
metrics = {'epoch': [], 'train_loss': [], 'val_loss': []}

with open(log_file) as f:
    for line in f:
        match = re.search(r'Epoch (\d+)', line)
        if match:
            metrics['epoch'].append(int(match.group(1)))
        
        match = re.search(r'Train Loss: ([\d.]+)', line)
        if match:
            metrics['train_loss'].append(float(match.group(1)))
        
        match = re.search(r'Val Loss: ([\d.]+)', line)
        if match:
            metrics['val_loss'].append(float(match.group(1)))

# Plot
import matplotlib.pyplot as plt
plt.plot(metrics['epoch'], metrics['train_loss'], label='Train')
plt.plot(metrics['epoch'], metrics['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

---

## Summary

Enhanced logging provides visibility into:
1. **Loss Components**: Where gradients flow
2. **Gradient Health**: Training stability
3. **RL Behavior**: Complexity adaptation
4. **Bias Correction**: Systematic error removal
5. **Integration Health**: Overall pipeline status

Use these metrics to:
- Monitor training progress
- Detect convergence issues early
- Debug component interactions
- Optimize hyperparameters
- Track experimental results
