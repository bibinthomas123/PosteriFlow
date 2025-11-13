# Overlap Neural PE Architecture Guide

## Overview

The **OverlapNeuralPE** is a unified, best-in-class neural parameter estimation pipeline designed for extracting gravitational wave signals from overlapping multi-detector data. It integrates six core components into a seamless end-to-end system that handles signal prioritization, adaptive complexity control, posterior estimation, systematic bias correction, iterative signal subtraction, and uncertainty quantification.

This guide provides a comprehensive walkthrough of the architecture, core components, and how they work together to enable robust parameter inference for gravitational wave astronomy.

---

## Architecture Overview

```
Input: Multi-detector Strain Data
       ↓
       ├─→ [1] PriorityNet → Signal Priorities & Rankings
       │
       ├─→ [2] Context Encoder → High-dim Signal Features (512-dim)
       │       ↓
       ├─→ [3] Normalizing Flow → Posterior Distribution (FlowMatching/RealNVP)
       │       ↓
       ├─→ [4] RL Controller → Adaptive Complexity Selection
       │       ↓
       ├─→ [5] Bias Corrector → Systematic Bias Removal
       │       ↓
       ├─→ [6] Adaptive Subtractor → Iterative Signal Extraction
       │       ↓
Output: Extracted Signals with Bias-Corrected Parameters & Uncertainties
```

---

## Component 1: PriorityNet (Signal Ranking)

**Purpose**: Identify which signals to extract first and their relative importance.

**Implementation**:
- Pre-trained, frozen neural network that analyzes residual strain data
- Outputs priority scores (0 to 1) and optionally signal uncertainties
- Takes as input: edge-conditioned strain data with detection information

**Key Features**:
- Edge conditioning captures signal boundary effects
- Learned ranking correlates with signal SNR and detectability
- Frozen parameters prevent drift during overlap training

**Usage in Pipeline**:
```python
detections = self._residual_to_detections(residual_data)
priorities, _ = self.priority_net(detections)
```

---

## Component 2: Context Encoder

**Purpose**: Extract compact, informative representations from multi-detector strain data.

**Architecture**:
```
Strain Data [batch, 2_detectors, 16384_samples]
    ↓
Per-detector CNN: Conv1d layers + BatchNorm + ReLU
    ↓ (2× for H1/L1 detectors)
Detector Features: [batch, 128, 64]
    ↓
Concatenate: [batch, 256, 64]
    ↓
Flatten: [batch, 16384]
    ↓
Fusion Network: 3 FC layers with LayerNorm
    ↓
Context Vector: [batch, 512]
```

**Key Improvements**:
- **Hidden dimension increased to 512** for richer feature representation
- **Dropout layers** after each CNN block (0.1) prevent overfitting
- **Layer normalization** stabilizes gradient flow through deep fusion
- **Per-detector encoding** captures instrument-specific characteristics

**Example Output**:
```python
context = self.context_encoder(strain_data)  # [batch, 512]
# context is normalized: (context - mean) / (std + 1e-6)
```

---

## Component 3: Normalizing Flow (Posterior Estimation)

**Purpose**: Learn a flexible conditional distribution p(parameters | data) for Bayesian inference.

**Architecture Options**:

### FlowMatching (Default - Optimal)
- **Optimal Transport Conditional Flow Matching (OT-CFM)**
- Fewer, more expressive layers (4 vs 8)
- Better convergence for high-dimensional posteriors
- Uses solver_steps=10 for accurate ODE integration

```python
flow = create_flow_model(
    flow_type='flowmatching',
    features=9,              # Parameter dimension
    context_features=512,    # From context encoder
    hidden_dim=256,          # High capacity
    num_layers=4,            # Few, expressive layers
    solver_steps=10,
    dropout=0.1
)
```

### RealNVP (Fallback)
- Affine coupling layers with masked transformations
- More stable training but less expressive
- Good for quick prototyping

```python
flow = create_flow_model(
    flow_type='realnvp',
    features=9,
    context_features=512,
    hidden_features=256,
    num_layers=8,
    num_blocks_per_layer=2
)
```

**Posterior Sampling**:
```python
# Sample from standard normal
z = torch.randn(batch_size, n_samples, param_dim)

# Transform through inverse flow (latent → parameter space)
samples, _ = self.flow.inverse(z, context)  # Physical units directly
```

---

## Component 4: RL Controller (Adaptive Complexity)

**Purpose**: Dynamically adjust computational budget based on pipeline state.

**Complexity Levels**:

| Level | Flow Layers | Inference Samples | Use Case |
|-------|------------|------------------|----------|
| Low | 4 | 500 | High residual power, many signals remaining |
| Medium | 8 | 1000 | Standard extraction |
| High | 12 | 2000 | Final signal, low SNR, high precision needed |

**State Representation**:
```python
state = {
    'remaining_signals': int,           # Iterations left
    'residual_power': float,            # Mean squared residual
    'current_snr': float,               # Residual RMS power
    'extraction_success_rate': float    # Fraction of successful extractions
}
```

**Decision Loop**:
```python
complexity = self.rl_controller.get_complexity_level(state, training=training)
# Returns: 'low', 'medium', or 'high'

# Apply complexity settings
n_samples = complexity_configs[complexity]['inference_samples']
result = self.extract_single_signal(residual_data, complexity)
```

**Learning Mechanism**:
- Q-network trained on extraction accuracy vs. computational cost
- Experience replay buffer stores (state, action, reward, next_state)
- Reward = extraction accuracy + bias correction confidence - uncertainty penalty

---

## Component 5: Bias Corrector (Systematic Error Removal)

**Purpose**: Remove systematic biases introduced by overlapping signals, limited detector information, or model limitations.

### Two-Mode Operation

#### Mode 1: Physics-Based Correction (Default, No Training Required)
```python
if self.is_trained == False:
    corrections = self._apply_physics_based_correction(
        original_values, original_uncertainties, iteration_idx, total_iterations
    )
```

**Physics Model**:
- **Masses**: Underestimated by ~2% due to degeneracy with distance
  - Correction: `-0.02 * mass_value * hierarchy_factor`
- **Distance**: Overestimated by ~10% in overlapping signals
  - Correction: `+0.10 * distance * hierarchy_factor`
- **Sky Position**: Localization biases scale with poor geometry
  - Correction: `~0.05 * sky_uncertainty * hierarchy_factor`
- **Time**: Small systematic delays from signal processing
  - Correction: `~0.001s (±1ms) * hierarchy_factor`

**Hierarchy Factor**: Increases bias for later iterations (signals become weaker):
```
hierarchy_factor = 1.0 + 0.3 * (iteration_idx / total_iterations)
```

#### Mode 2: Neural Network Correction (After Training)
```python
if self.is_trained == True:
    corrections, uncertainties = self.bias_estimator(params_normalized, context)
    corrections *= self.correction_scales
    corrections = torch.clamp(corrections, -0.2, 0.2)  # Max 20% correction
```

### BiasEstimator Architecture

```
Parameters [batch, 9] ──→ Embedding Layer (128-dim) ──→ 64-dim
Context [batch, 512] ──→ Embedding Layer (64-dim) ──→ 32-dim
                         │
                         ├─→ Concatenate: 96-dim
                         │
                         ├─→ Transformer Encoder (3 layers, 8 heads)
                         │
                         └─→ 96-dim features
                             │
                             ├─→ Mass Corrector (→ 2 outputs) × mass_scale
                             ├─→ Distance Corrector (→ 1 output) × distance_scale
                             ├─→ Sky Corrector (→ 2 outputs) × sky_scale
                             ├─→ Time Corrector (→ 1 output) × time_scale
                             └─→ Extra Corrector (→ 3 outputs) × extra_scale
                             
                             └─→ Uncertainty Head (Softplus)
```

### Physics-Based Constraints

**Parameter Bounds** (Hard Limits):
```python
bounds = {
    'mass_1': (1.0, 100.0) M☉,
    'mass_2': (1.0, 100.0) M☉,
    'luminosity_distance': (20.0, 8000.0) Mpc,
    'geocent_time': (-0.1, 0.1) s,
    'ra': (0.0, 2π),
    'dec': (-π/2, π/2),
    'theta_jn': (0.0, π),
    'psi': (0.0, π),
    'phase': (0.0, 2π)
}
```

**Correction Magnitude Limits**:
- Masses: ≤ 8% of value
- Distance: ≤ 30% of value
- Sky position: ≤ 20% of parameter space
- Time: ≤ 2 ms absolute
- Angles: ≤ 25% of parameter space

### Integration with Extraction Pipeline

```python
# Step 1: Extract parameters with flow
params_means, params_stds, context = extract_single_signal(...)

# Step 2: Apply bias correction
if self.bias_corrector is not None:
    params_normalized = self._normalize_parameters(params_means)
    corrections, bias_uncertainties, confidence = self.bias_corrector(
        params_normalized, context
    )
    
    # Apply correction
    params_corrected = params_means + corrections
    params_corrected = torch.clamp(params_corrected, 
                                   min_bounds, max_bounds)
    
    # Update uncertainties (quadrature sum)
    params_stds_final = torch.sqrt(params_stds**2 + bias_uncertainties**2)
else:
    params_corrected = params_means
    params_stds_final = params_stds

all_extracted.append({
    'params': params_corrected,
    'uncertainties': params_stds_final,
    'bias_corrected': True,
    'confidence': confidence.mean()
})
```

### Validation & Quality Control

Each correction undergoes comprehensive validation:

```python
validation = {
    'physics_valid': check_bounds(corrected_params),
    'magnitude_acceptable': check_correction_sizes(corrections),
    'uncertainty_reasonable': check_uncertainty_ratios(uncertainties),
    'overall_valid': physics_valid and magnitude_acceptable
}

if validation['overall_valid']:
    apply_correction()
else:
    reject_correction()
    return original_parameters
```

---

## Component 6: Adaptive Subtractor (Iterative Extraction)

**Purpose**: Subtract extracted signals from residual data to isolate subsequent overlapping sources.

### Extraction-Subtraction Loop

```python
for iteration in range(self.max_iterations):
    # [Components 1-5 above]
    
    # Subtract extracted signal from residual
    residual_dict = self._tensor_to_detector_dict(residual_data)
    
    residual_dict, metadata, uncertainties = self.adaptive_subtractor.extract_and_subtract(
        residual_dict,
        detection_idx=iteration
    )
    
    # Convert back to tensor
    residual_data = self._detector_dict_to_tensor(residual_dict)
    
    # Update pipeline state for RL
    pipeline_state['residual_power'] = torch.mean(residual_data ** 2)
    
    # Early stopping if residual too low
    if pipeline_state['residual_power'] < 0.001:
        break
```

### Subtraction Strategy

- **Waveform Regeneration**: Uses extracted parameters to synthesize original signal
- **Matched Filtering**: Cross-correlates with residual to find optimal phase/amplitude
- **Multi-detector Consistency**: Ensures subtraction is coherent across H1, L1, V1
- **Uncertainty Propagation**: Tracks subtraction uncertainties through iterations

### Convergence Criteria

```
STOPPING CONDITIONS:
├─ Residual power < 0.001 (99% power removed)
├─ SNR of next signal < detection threshold
├─ Maximum iterations reached (typically 5)
└─ Subtraction uncertainty exceeds correction confidence
```

---

## Training Workflow

### Loss Function (Multi-Component)

```python
total_loss = (
    flow_loss +                     # Main NLL from normalizing flow
    jacobian_reg +                  # Flow weight regularization (1e-4)
    0.1 * physics_loss +            # Physical constraint penalties
    bias_loss +                     # Bias correction regularization
    uncertainty_loss                # Uncertainty calibration
)
```

### Forward Pass Example

```python
# 1. Encode strain data
context = self.context_encoder(strain_data)

# 2. Normalize ground truth parameters
true_params_norm = self._normalize_parameters(true_params)

# 3. Compute flow likelihood
log_prob = self.flow.log_prob(true_params_norm, context)
flow_loss = -log_prob.mean()

# 4. Physics constraint loss
mass_violation = F.relu(params[:, m2_idx] - params[:, m1_idx])
physics_loss = torch.mean(mass_violation**2)

# 5. Bias correction loss (if enabled)
if self.bias_corrector is not None:
    corrections, _, confidence = self.bias_corrector(true_params_norm, context)
    # Encourage small corrections on average (prior toward no correction)
    bias_loss = 0.05 * (torch.mean(torch.abs(corrections)) + 
                        0.1 * torch.mean(-confidence))

# 6. Backward pass
total_loss.backward()
optimizer.step()
```

### Hyperparameters to Tune

```yaml
context_dim: 512              # Context encoder output dimension
n_flow_layers: 6              # Normalizing flow depth
max_iterations: 5             # Max signals to extract

dropout: 0.1                  # Dropout rate in context encoder
flow_dropout: 0.15           # Dropout in flow model

# Flow configuration
flow_config:
  type: 'flowmatching'        # 'flowmatching', 'realnvp', or 'maf'
  hidden_features: 256        # Flow network hidden dimension
  num_layers: 4               # Fewer for FlowMatching
  num_blocks_per_layer: 2
  solver_steps: 10            # ODE solver steps

# RL Controller
rl_controller:
  complexity_levels: ['low', 'medium', 'high']
  learning_rate: 1e-3
  epsilon: 0.1                # Exploration rate
  epsilon_decay: 0.995
  memory_size: 10000

# Bias Corrector
bias_corrector:
  enabled: true
  training_epochs: 200        # If training
  strategy: 'balanced'        # 'conservative', 'balanced', 'aggressive'
```

---

## Inference Pipeline

### Single Signal Extraction

```python
# Simple case: extract one signal
model = OverlapNeuralPE(param_names, priority_net_path, config)

result = model.extract_single_signal(
    strain_data,           # [batch, n_det, n_samples]
    complexity='medium'    # 'low' | 'medium' | 'high'
)

# Returns:
# {
#     'means': [batch, param_dim],
#     'stds': [batch, param_dim],
#     'samples': [batch, n_samples, param_dim],
#     'uncertainties': [batch, param_dim],
#     'context': [batch, 512]
# }
```

### Overlapping Signal Extraction

```python
# Complex case: iteratively extract overlapping signals
results = model.extract_overlapping_signals(
    strain_data,                  # [batch, n_det, n_samples]
    true_params=None,             # Optional for RL training
    training=False
)

# Returns:
# {
#     'all_signals': [
#         {
#             'params': [batch, param_dim],
#             'uncertainties': [batch, param_dim],
#             'priority': [...],
#             'iteration': int,
#             'complexity': str,
#             'bias_corrected': bool
#         },
#         ...
#     ],
#     'residual': [batch, n_det, n_samples],
#     'quality_metric': float,
#     'extraction_summary': {...}
# }
```

### Posterior Sampling

```python
# Generate full posterior distribution samples
samples_dict = model.sample_posterior(
    strain_data,          # [batch, n_det, n_samples]
    n_samples=1000        # Number of posterior samples
)

# Returns:
# {
#     'samples': [batch, n_samples, param_dim],    # Full posterior
#     'means': [batch, param_dim],
#     'stds': [batch, param_dim],
#     'uncertainties': [batch, param_dim],
#     'context': [batch, 512]
# }

# Use for visualization/analysis
plt.hist(samples_dict['samples'][0, :, 0], bins=50)  # mass_1 posterior
```

---

## Performance Tracking

### Built-in Metrics

The model automatically tracks:

```python
# Training metrics
model.performance_tracker = {
    'training_losses': deque(maxlen=1000),           # Latest 1000 losses
    'validation_metrics': deque(maxlen=100),
    'complexity_history': deque(maxlen=1000),        # RL actions
    'inference_times': deque(maxlen=1000),           # Per-signal timing
    'rl_rewards': deque(maxlen=1000)                 # RL training rewards
}
```

### Integration Summary

```python
summary = model.get_integration_summary()

# Contains:
# - Component status (enabled/disabled, parameter counts)
# - Configuration snapshot
# - Training metrics
# - RL metrics (epsilon, avg complexity, etc.)
# - Bias correction metrics (correction rates, acceptance rate)
```

### Bias Correction Metrics

```python
bias_metrics = model.get_bias_metrics()

# Returns:
# {
#     'avg_correction': float,           # Mean correction magnitude
#     'max_correction': float,
#     'avg_confidence': float,           # Mean confidence scores
#     'physics_violations': int,         # Rejected corrections
#     'correction_acceptance_rate': float
# }
```

---

## Best Practices

### Configuration Tips

1. **For High-SNR Signals**:
   ```yaml
   rl_controller:
     complexity_levels:
       low: {flow_layers: 2, inference_samples: 250}
       medium: {flow_layers: 4, inference_samples: 500}
       high: {flow_layers: 8, inference_samples: 1500}
   ```

2. **For Overlapping Sources**:
   - Enable bias corrector (mitigates interference artifacts)
   - Use `max_iterations: 5-7` (more signals likely)
   - Increase `context_dim` to 768 for complex scenes

3. **For Production Inference**:
   ```yaml
   dropout: 0.05              # Reduce during inference
   flow_dropout: 0.05
   use_real_noise_prob: 0.1   # Add realistic noise background
   ```

### Common Failure Modes & Fixes

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Bias blow-up** | Corrections larger than signal values | Reduce `bias_loss` weight, enable validation |
| **Poor flow convergence** | High NLL after training | Increase `context_dim` to 768, use FlowMatching |
| **RL not learning** | Constant complexity selection | Increase RL `learning_rate` to 1e-2, reduce `epsilon_decay` |
| **Residual not decreasing** | Subtraction ineffective | Check waveform generation params, verify matched filtering |

### Debugging Workflow

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
model = OverlapNeuralPE(..., config)

# Check integration status
summary = model.get_integration_summary()
print(f"Components active: {len(summary['components'])}")
print(f"Total parameters: {summary['metrics']['total_parameters']}")

# Monitor extraction step-by-step
for iteration in range(5):
    # Logs will show:
    # - PriorityNet priorities
    # - RL complexity selection
    # - Flow sampling time
    # - Bias correction confidence
    # - Subtraction residual power
```

---

## Key Implementation Notes

1. **Parameter Normalization**: All parameters normalized to [-1, 1] for flow training
   ```python
   normalized = 2 * (params - min_val) / (max_val - min_val) - 1
   ```

2. **Context Normalization**: Context centered and scaled before flow
   ```python
   context = (context - mean) / (std + 1e-6)
   ```

3. **Device Management**: All tensors on GPU (cuda or fallback to cpu)
   ```python
   self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
   ```

4. **Gradient Handling**: 
   - PriorityNet frozen (no gradients)
   - Flow backpropagates through context encoder
   - Bias corrector trains separately or uses physics priors

5. **Batch Processing**: All operations vectorized for batch_size > 1
   - Shape consistency: [batch, ...] throughout

---

## Architecture Strengths

✅ **Integrated Pipeline**: All components work seamlessly together
✅ **Adaptive Complexity**: RL adjusts computational budget in real-time
✅ **Bias Correction**: Removes systematic errors from overlapping signals
✅ **Principled Uncertainty**: Full Bayesian posterior via normalizing flows
✅ **Physics-Informed**: Constraints and priors throughout
✅ **Production-Ready**: Checkpoint loading, metrics tracking, error handling

---

## References

- OverlapNeuralPE: `src/ahsd/models/overlap_neuralpe.py`
- BiasCorrector: `src/ahsd/core/bias_corrector.py`
- Normalizing Flows: `src/ahsd/models/flows.py`
- Adaptive Subtraction: `src/ahsd/core/adaptive_subtractor.py`
- RL Controller: `src/ahsd/models/rl_controller.py`
- PriorityNet: `src/ahsd/core/priority_net.py`
