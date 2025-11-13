# Logging Examples & Interpretation

## Before vs After Enhancement

### BEFORE: Basic Logging
```
Epoch 18/100
  Train Loss: 8.3022 (NLL: 8.3026)
  Val Loss: 8.2892 (NLL: 8.2897)
  Train-Val Gap: 0.0129
  Gradient Norm: 0.036
  LR: 1.00e-04
  Patience: 0/20
  Time: 55.9s
  RL Controller:
    epsilon: 0.1000
```

**Problem**: Hard to diagnose training issues. Where is the loss coming from? Is bias correction working?

---

### AFTER: Enhanced Logging
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

**Benefit**: Complete visibility into all components. Easy to spot issues.

---

## Real-World Scenarios

### Scenario 1: Healthy Training Progression

**Epoch 5** (Early training - rapid learning):
```
Epoch 5/100
  Train Loss: 8.5234 (NLL: 8.5240)
  Val Loss: 8.6123 (NLL: 8.6130)
  Train-Val Gap: -0.0889           ← Note: negative (underfitting, okay early)
  Gradient Norm: 0.1234            ← Strong gradients
  LR: 1.00e-04
  Patience: 5/20
  Time: 52.3s
  
  Loss Components:
    Physics Loss: 0.0450           ← High early (model learning constraints)
    Bias Loss: 0.0234
    Uncertainty Loss: 0.0567
    Jacobian Reg: 0.0001
  
  Bias Corrector:
    ✓ Avg Correction: 0.0050       ← Learning to correct
    📊 Acceptance Rate: 92.34%      ← High confidence
    ⚠️  Physics Violations: 3       ← Some corrections rejected
  
  RL State:
    Exploration (ε): 0.1000        ← Full exploration
    Avg Complexity: 1.2345         ← Using varied complexity
    Avg Reward: 0.1234             ← Learning starting
```

**Interpretation**: ✅ **All systems healthy**
- Strong gradient learning
- Physics loss high (expected, will decrease)
- Bias corrector exploring space
- RL discovering good actions

---

**Epoch 25** (Mid training - convergence):
```
Epoch 25/100
  Train Loss: 8.1234 (NLL: 8.1240)
  Val Loss: 8.1456 (NLL: 8.1462)
  Train-Val Gap: -0.0222           ← Flip to slightly negative
  Gradient Norm: 0.0567            ← Moderate gradients
  LR: 1.00e-04
  Patience: 0/20
  Time: 56.1s
  
  Loss Components:
    Physics Loss: 0.0023           ← Decreased 20× (good)
    Bias Loss: 0.0012
    Uncertainty Loss: 0.0034
    Jacobian Reg: 0.0001
  
  Bias Corrector:
    ✓ Avg Correction: 0.0012       ← Refined to right scale
    📊 Acceptance Rate: 88.92%      ← Good validation
    ⚠️  Physics Violations: 1       ← Rare
  
  RL State:
    Exploration (ε): 0.0823        ← Decayed (exploiting)
    Avg Complexity: 1.5678         ← Learned good range
    Avg Reward: 0.3456             ← Strong performance
```

**Interpretation**: ✅ **Converging well**
- Physics loss decreased significantly
- Bias corrections refined & accurate
- RL settled into good strategy
- No early stopping (best still improving)

---

**Epoch 50** (Late training - fine-tuning):
```
Epoch 50/100
  Train Loss: 7.8901 (NLL: 7.8907)
  Val Loss: 7.9012 (NLL: 7.9018)
  Train-Val Gap: -0.0111           ← Very small gap
  Gradient Norm: 0.0234            ← Small but stable
  LR: 6.31e-05                     ← Scheduler reduced
  Patience: 3/20
  Time: 56.8s
  
  Loss Components:
    Physics Loss: 0.0001           ← Nearly 0 (excellent)
    Bias Loss: 0.0003
    Uncertainty Loss: 0.0001
    Jacobian Reg: 0.0001
  
  Bias Corrector:
    ✓ Avg Correction: 0.0008       ← Stable
    📊 Acceptance Rate: 91.45%      ← Consistent
    ⚠️  Physics Violations: 0       ← None!
  
  RL State:
    Exploration (ε): 0.0314        ← Mostly exploit
    Avg Complexity: 1.6234         ← Locked in strategy
    Avg Reward: 0.4123             ← Peak performance
```

**Interpretation**: ✅ **Fully converged**
- All component losses minimal
- Perfect physics compliance
- RL exploiting learned policy
- Scheduler helping fine-tune

---

### Scenario 2: Overfitting Problem

**Epoch 40** (Signs of overfitting):
```
Epoch 40/100
  Train Loss: 7.6234 (NLL: 7.6240)
  Val Loss: 8.1567 (NLL: 8.1573)
  Train-Val Gap: 0.5333            ← ⚠️ HUGE GAP!
  Gradient Norm: 0.0134            ← Declining
  LR: 1.00e-04
  Patience: 15/20                  ← Close to early stopping
  Time: 57.2s
  
  Loss Components:
    Physics Loss: 0.0001
    Bias Loss: 0.0234              ← ⚠️ Growing
    Uncertainty Loss: 0.0567       ← ⚠️ Growing
    Jacobian Reg: 0.0001
  
  Bias Corrector:
    ✓ Avg Correction: 0.0234       ← ⚠️ Larger
    📊 Acceptance Rate: 73.45%      ← ⚠️ Dropped
    ⚠️  Physics Violations: 23      ← ⚠️ Many rejections
  
  RL State:
    Exploration (ε): 0.0456        ← Stuck
    Avg Complexity: 2.0000         ← ⚠️ ALWAYS high!
    Avg Reward: 0.3123             ← Declining
```

**Interpretation**: ❌ **Overfitting detected**

**Immediate Actions**:
1. **Stop training** - Next few epochs will break
2. **Reload best checkpoint** (Epoch 30)
3. **Adjust config**:
   ```yaml
   # In enhanced_training.yaml
   dropout: 0.1 → 0.2              # Increase dropout
   flow_dropout: 0.15 → 0.25
   n_flow_layers: 6 → 4            # Reduce capacity
   context_dim: 512 → 256
   ```
4. **Resume training** with reduced capacity

---

### Scenario 3: Bias Corrector Failing

**Epoch 15** (Bias corrector unstable):
```
Epoch 15/100
  Train Loss: 8.4567 (NLL: 8.4573)
  Val Loss: 8.3456 (NLL: 8.3462)
  Train-Val Gap: 0.1111
  Gradient Norm: 0.0456
  
  Loss Components:
    Physics Loss: 0.0001
    Bias Loss: 0.2345              ← ⚠️ VERY HIGH
    Uncertainty Loss: 0.0012
    Jacobian Reg: 0.0001
  
  Bias Corrector:
    ✓ Avg Correction: 0.1234       ← ⚠️ Way too large
    ✓ Max Correction: 0.5678       ← ⚠️ Extreme
    ⚡ Avg Confidence: 0.2345      ← ⚠️ Very uncertain
    📊 Acceptance Rate: 34.56%      ← ⚠️ Most rejected
    ⚠️  Physics Violations: 456     ← ⚠️ MANY
```

**Interpretation**: ❌ **Bias corrector broken**

**Root Causes & Solutions**:

1. **Bias corrector not trained** (using untrained neural network):
   ```python
   if model.bias_corrector.is_trained == False:
       # Force physics-based mode
       model.bias_corrector.is_trained = True  # Skip neural path
       # Or: model.bias_corrector = None  (disable completely)
   ```

2. **Correction weights too aggressive**:
   ```yaml
   bias_corrector:
     strategy: 'aggressive'  → 'conservative'
   # Or manually:
   model.bias_corrector.current_strategy = 'conservative'
   ```

3. **Bounds too tight**:
   ```python
   # Check physics bounds
   bounds = model.bias_corrector.physics_bounds
   # Increase max_correction values if legitimate signals rejected
   ```

---

### Scenario 4: Vanishing Gradients

**Epoch 8** (Gradients dying):
```
Epoch 8/100
  Train Loss: 8.5600 (NLL: 8.5606)   ← Loss not decreasing!
  Val Loss: 8.5601 (NLL: 8.5607)
  Train-Val Gap: -0.0001
  Gradient Norm: 0.00023             ← ⚠️ VANISHING!
  LR: 1.00e-04
  Time: 50.1s
  
  Loss Components:
    Physics Loss: 0.0450
    Bias Loss: 0.0234
    Uncertainty Loss: 0.0567
    Jacobian Reg: 0.0001
```

**Interpretation**: ❌ **Vanishing gradients**

**Solutions**:

1. **Increase learning rate**:
   ```yaml
   optimizer:
     lr: 1.00e-04 → 1.00e-03  (10×)
   ```

2. **Reduce weight decay**:
   ```yaml
   optimizer:
     weight_decay: 1e-5 → 1e-6
   ```

3. **Reduce gradient clipping threshold**:
   ```yaml
   training:
     gradient_clip: 1.0 → 10.0  (allow larger gradients)
   ```

4. **Try different scheduler**:
   ```yaml
   scheduler:
     type: 'cosine_annealing'  # Instead of ReduceLROnPlateau
   ```

---

### Scenario 5: RL Not Learning

**Epoch 30** (RL controller stuck):
```
Epoch 30/100
  RL State:
    Exploration (ε): 0.0951         ← Not decaying properly
    Avg Complexity: 0.9999          ← STUCK at one level
    Action Entropy: 0.0012          ← ⚠️ ZERO entropy (always same action)
    Avg Reward: 0.1234              ← Stalled (no improvement)
  
  Loss Components:
    (rest normal, but RL contribution not visible)
```

**Interpretation**: ❌ **RL not learning optimal policy**

**Solutions**:

1. **Check reward signal**:
   ```python
   # Is reward actually based on extraction quality?
   # Verify: _compute_extraction_reward() uses correct metrics
   ```

2. **Increase RL learning rate**:
   ```yaml
   rl_controller:
     learning_rate: 1e-3 → 1e-2  (10×)
   ```

3. **Reduce epsilon decay**:
   ```yaml
   rl_controller:
     epsilon_decay: 0.995 → 0.99  (slower decay = more exploration)
   ```

4. **Check action space**:
   ```python
   # Verify complexity levels are actually different
   complexity_configs = {
       'low': {'flow_layers': 2, 'inference_samples': 250},    ← Make more different
       'medium': {'flow_layers': 6, 'inference_samples': 750},
       'high': {'flow_layers': 12, 'inference_samples': 2000}
   }
   ```

---

## Metric Ranges Reference

### Loss Metrics
```
Phase          NLL Range      Train-Val Gap   Status
Early (1-10)   8.5-8.0       -0.05 to 0.05   Normal (underfitting okay)
Mid (10-50)    8.0-7.5       0.01 to 0.05    Optimal
Late (50+)     7.5-7.0       0.02 to 0.05    Converged
```

### Gradient Metrics
```
Gradient Norm          Status               Action
> 1.0                  Exploding            Reduce LR or increase clip
0.1 - 1.0              Good                 Continue
0.01 - 0.1             Normal               Continue
0.001 - 0.01           Weak                 Monitor
< 0.001                Vanishing            Increase LR / Reduce regularization
```

### RL Metrics
```
Metric                 Healthy Range        Issue
epsilon                0.1 → 0.01           Not decaying: stuck at ~0.1
avg_complexity         1.0 - 2.0            Always same: stuck action
action_entropy         > 0.8                = 0: no exploration
avg_reward             0.1 → 0.4            Stalled: RL not learning
```

### Bias Metrics
```
Metric                 Range                 Status
Avg Correction         0.0001-0.01          Normal
                       > 0.05               Too aggressive
Acceptance Rate        > 80%                Good
                       < 60%                Corrections invalid
Confidence             > 0.8                Trustworthy
                       < 0.5                Unreliable
Physics Violations     < 5 per epoch        Good
                       > 50 per epoch       Validation failing
```

---

## Quick Reference: What to Check

**Training stalled (loss not decreasing)**:
→ Check Gradient Norm (vanishing?) → Check Loss Components → Check Learning Rate

**Overfitting (large train-val gap)**:
→ Check component losses → Check Bias Acceptance Rate → Check RL Complexity

**Bias corrector unreliable**:
→ Check Acceptance Rate → Check Avg/Max Correction → Check is_trained flag

**RL not improving**:
→ Check epsilon decay → Check action entropy → Check avg reward trend

---

## Logging to CSV for Analysis

Create a simple script to extract metrics:

```python
import re
import csv

log_file = "outputs/experiments/run_001/training.log"
csv_file = "metrics.csv"

metrics = {}

with open(log_file) as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    # Extract epoch
    match = re.search(r'Epoch (\d+)/', line)
    if match:
        epoch = int(match.group(1))
        metrics[epoch] = {}
    
    # Extract metrics
    patterns = {
        'train_loss': r'Train Loss: ([\d.]+)',
        'val_loss': r'Val Loss: ([\d.]+)',
        'nll': r'NLL: ([\d.]+)',
        'gradient_norm': r'Gradient Norm: ([\d.]+)',
        'physics_loss': r'Physics Loss: ([\d.]+)',
        'bias_loss': r'Bias Loss: ([\d.]+)',
        'uncertainty_loss': r'Uncertainty Loss: ([\d.]+)',
        'bias_acceptance': r'Acceptance Rate: ([\d.]+)%',
        'avg_reward': r'Avg Reward: ([\d.]+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match and epoch in metrics:
            metrics[epoch][key] = float(match.group(1))

# Write to CSV
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['epoch'] + list(metrics[min(metrics.keys())].keys()))
    writer.writeheader()
    for epoch, row in sorted(metrics.items()):
        writer.writerow({'epoch': epoch, **row})

print(f"Saved metrics to {csv_file}")
```

Then analyze in Excel/Pandas:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('metrics.csv')
df.plot(x='epoch', y=['train_loss', 'val_loss'])
plt.show()
```

