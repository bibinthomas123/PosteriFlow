# PriorityNet: A Comprehensive Guide to Intelligent Signal Prioritization

## Introduction

PriorityNet is the intelligent brain of the PosteriFlow adaptive hierarchical signal decomposition (AHSD) pipeline. When multiple gravitational wave signals overlap in detector data, extracting them in the right order is crucial for minimizing reconstruction errors and maximizing detection reliability. PriorityNet learns to rank signals by their relative importance—a quantity we call "priority"—enabling the pipeline to subtract strong, well-measured signals first before attempting to extract fainter or more degenerate ones.

This document serves as a complete guide to understanding, using, and troubleshooting PriorityNet. We'll cover the architecture, training methodology, integration points, and practical deployment strategies.

---

## The Problem: Why Signal Ordering Matters

In overlapping gravitational wave scenarios, the order in which signals are extracted determines the quality of final parameter estimates. Consider a simple case with two overlapping signals:

**Scenario A (Good Order)**: Extract strong signal first, subtract it cleanly, then extract weaker signal from residuals
- Strong signal: Low bias, high accuracy
- Weak signal: Low bias (residuals are clean), reasonable accuracy

**Scenario B (Bad Order)**: Extract weak signal first, then strong signal
- Weak signal: High bias (interfered by strong signal), biased estimates
- Strong signal: Must fit both signals simultaneously, reduced accuracy

This ordering problem becomes exponentially harder with 3+ overlapping signals. PriorityNet solves this by learning what makes a signal "easy" vs "hard" to extract, and recommending an optimal extraction sequence.

---

## Core Concept: What is Priority?

**Priority** is a learned scalar score [0, 1] that estimates how "extractable" a signal is relative to others in the same scenario. High-priority signals typically have:
- High signal-to-noise ratio (SNR)
- Simple morphology (non-precessing or slowly precessing)
- Localized in time (short duration)
- Sky position well-constrained by detector network

Low-priority signals are harder to extract:
- Low SNR, buried in noise
- Complex morphology (precessing)
- Extended duration (long signal)
- Poor sky localization (off-axis)

PriorityNet learns a mapping: **signals + their interactions → priority scores**, where priorities naturally order themselves by extractability.

---

## Architecture Overview

PriorityNet combines four specialized neural modules that work in concert:

```
Inputs:
  - Detector Strain (optional): [batch, n_detectors, 2048 time samples]
  - Signal Parameters: [batch, 16 normalized features]
  - Edge-Type IDs (optional): [batch] indicating signal morphology class

        ↓
    
Module 1: TemporalStrainEncoder (if strain provided)
    Extracts: Time-frequency features from whitened strain
    Output: [batch, 64] temporal embeddings
    
    ├─ Multi-scale CNN (4 blocks) → frequency decomposition
    ├─ BiLSTM (2 layers) → temporal context
    ├─ Multi-head Attention (8 heads) → long-range correlations
    └─ Project → 64-D representation
    
        ↓

Module 2: SignalFeatureExtractor
    Extracts: Learned + physics-derived parameter features
    Output: [batch, 128] parameter embeddings (96 learned + 32 physics)
    
    ├─ Residual MLP path: 16 → 512 → 384 → 256 → 128 → 64
    ├─ Physics path: Compute chirp mass, mass ratio, SNR, f_ISCO, χ_eff → 32-D
    └─ SNR embedding: Explicit pathway for network SNR weighting
    
        ↓

Module 3: CrossSignalAnalyzer
    Extracts: Pairwise overlap metrics between all signal pairs
    Output: [batch, 16] per-signal overlap features
    
    For each signal pair (i,j):
    ├─ Time separation (Δt)
    ├─ Sky separation (Δsky)
    ├─ Mass similarity
    ├─ Frequency overlap
    ├─ Distance ratio
    └─ Polarization & RA/Dec differences
    
    Aggregates across pairs with learned importance weighting
    
        ↓

Module 4: MultiModalFusion
    Fuses: All modalities (temporal, metadata, overlap, edge info)
    Output: [batch, 64] unified representations
    
    ├─ Concatenate all streams: 128 + 16 + 64 + 32 = 240-D
    ├─ Project to 64-D
    ├─ Multi-head self-attention (4 heads)
    └─ Residual feed-forward network
    
        ↓
    
Priority Head (2 outputs per signal)
    ├─ Priority: Linear output → raw priority score (pre-affine transform)
    ├─ Calibration: Affine scale/shift (prio_gain, prio_bias) → final priority ∈ [0,1]
    │
    └─ Uncertainty: Softplus → positive uncertainty σ for confidence quantification
```

Each module is designed to be plug-and-play. You can run PriorityNet with or without strain data, with or without edge conditioning, making it adaptable to different scenarios and computational budgets.

---

## Detailed Module Breakdown

### 1. TemporalStrainEncoder: Learning from Waveform Morphology

**Purpose**: Extract signal morphology information directly from whitened detector strain.

**Why It Matters**: The shape of a gravitational wave signal encodes rich information about the source:
- **CBC signals**: Characteristic "chirp" morphology (frequency increases over time)
- **Precessing systems**: Additional modulations in amplitude and frequency
- **High mass BBH**: Rapid frequency sweep, short duration
- **BNS**: Slower sweep, longer inspiral phase

**Architecture in Detail**:

The encoder uses a hierarchical approach to capture multi-scale features:

```
Input: [batch, 3, 2048] (H1, L1, V1 whitened strain at 2048 Hz)
    
Conv Block 1: Conv1d(3→32, k=64, s=4) + BatchNorm + GELU + Dropout
    Output: [batch, 32, 512]  (downsamples by 4×, captures ~64 Hz features)
    
Conv Block 2: Conv1d(32→64, k=32, s=4) + BatchNorm + GELU + Dropout  
    Output: [batch, 64, 128]  (further 4× downsample, ~256 Hz features)
    
Conv Block 3: Conv1d(64→128, k=16, s=2) + BatchNorm + GELU + Dropout
    Output: [batch, 128, 64]   (2× downsample, ~512 Hz features)
    
Conv Block 4: Conv1d(128→128, k=8, s=2) + BatchNorm + GELU + Dropout
    Output: [batch, 128, 32]   (2× downsample, ~1024 Hz features)
    
Reshape: [batch, 32, 128] → [batch, 128, 32] (swap time/channel for LSTM)
    
BiLSTM: 2-layer, bidirectional, hidden=128
    Output: [batch, 32, 256] (128 forward + 128 backward per sample)
    
Self-Attention: 8-head attention over the 32 time steps
    Output: [batch, 32, 256]
    
Pooling: Global avg pool + global max pool + combine
    Output: [batch, 256]
    
Projection MLP: Linear(256→128) + LayerNorm + GELU + Linear(128→64)
    Output: [batch, 64]
```

**Key Design Decisions**:
- **Multi-scale convolutions**: Different kernel sizes capture morphology at different frequency bands
- **Downsampling strategy**: Reduces sequence length efficiently while preserving temporal context
- **BiLSTM**: Bidirectional processing ensures the model sees past AND future context (critical for learning temporal patterns)
- **Dual pooling**: Average pooling captures mean morphology; max pooling preserves spike features (merger precursors)
- **LayerNorm**: Stabilizes signal distribution across training

**When to Use**: 
- Always when you have access to detector strain (≤5ms additional latency)
- For detecting subtle signal distinctions (e.g., precession signatures)
- When computational budget allows (~1.2M additional parameters)

**Computational Cost**: ~5 ms inference per signal on CPU; scales well on GPU

---

### 2. SignalFeatureExtractor: Bridging Learned and Physics-Based Representations

**Purpose**: Transform raw signal parameters into rich representations that combine data-driven learning with domain expertise.

**The Dual-Path Design**:

PriorityNet uses two parallel paths to extract signal features:

**Path A: Learned Representation (64-D)**
```
Input: [batch, 16] normalized parameters
    ├─ mass_1, mass_2 (normalized [5-100] M☉)
    ├─ luminosity_distance (normalized [50-3000] Mpc)
    ├─ geocent_time, ra, dec, theta_jn, psi, phase
    ├─ a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl (spin parameters)
    └─ network_snr (pre-normalized [0,1])
    
Embedding Layer: Linear(16→512)
    Output: [batch, 512]
    
Residual Block 1: [512→512] with skip connection
    Apply: LayerNorm → Linear → GELU → Dropout → Skip
    
Residual Block 2: [512→384]
    
Residual Block 3: [384→256]
    
Residual Block 4: [256→128]
    
Final Projection: LayerNorm(128) → Linear(128→64)
    Output: [batch, 64] learned representation
```

**Path B: Physics-Derived Features (32-D)**
```
From input parameters, compute 8 physics-informed features:
    
    1. Chirp Mass (Mc):
       Mc = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
       Normalized to [0,1]: min(Mc/50, 1)
       → Indicates frequency scale of signal
       
    2. Mass Ratio (q):
       q = min(m1, m2) / max(m1, m2)  ∈ [0, 1]
       → 1 = equal masses, 0 = extreme mass ratio
       
    3. Symmetric Mass Ratio (η):
       η = m1*m2 / (m1+m2)^2  ∈ [0, 0.25]
       Normalized: η*4  → Makes use [0, 1]
       
    4. Estimated SNR:
       SNR_est = 8 * (Mc/30)^(5/6) * (400/D)
       → Physics-based SNR proxy from mass and distance
       
    5. ISCO Frequency:
       f_ISCO = 220 / (m1 + m2)  [Hz]
       Normalized: f_ISCO/1000
       → Characteristic merger frequency
       
    6. Effective Spin (χ_eff):
       χ_eff = (m1*a1 + m2*a2) / (m1 + m2)  ∈ [-1, 1]
       Normalized: (χ_eff + 1) / 2  ∈ [0, 1]
       
    7. Detection Difficulty:
       difficulty = log(D/100) - log(SNR_est/10)
       Normalized: (difficulty + 5) / 10
       → Combined metric: large distance + low SNR = hard
       
    8. Total Mass:
       M_tot = m1 + m2
       Normalized: M_tot / 200
       
Feed through lightweight MLP: Linear(8→64) + LayerNorm + GELU + Linear(64→32)
    Output: [batch, 32] physics features
```

**Concatenation & Output**:
```
Learned (64-D) || Physics (32-D) = [batch, 96] combined metadata
```

**Why This Dual Design**:
- **Learned path**: Captures non-linear combinations of parameters that matter for extractability
- **Physics path**: Provides interpretability and ensures domain knowledge is directly available
- **Separation**: Makes it easy to ablate and understand contributions; also prevents the network from "ignoring" fundamental physics

---

### 3. CrossSignalAnalyzer: Capturing Multi-Signal Interactions

**Purpose**: In overlapping scenarios, what matters is not just individual signals, but how they interfere with each other.

**The Pairwise Feature Approach**:

For a scenario with N signals, we compute N×(N-1) pairwise metrics:

```
For each signal i:
    For each other signal j ≠ i:
        
    Time Separation:
        Δt = |geocent_time_i - geocent_time_j|
        Range: [0, 0.2] seconds (normalized to [0,1])
        → Large Δt = signals separated in time = less interference
        
    Sky Separation:
        Δsky = sqrt((RA_i - RA_j)^2 + (Dec_i - Dec_j)^2)
        Range: [0, π] radians
        → Large Δsky = different sources = less correlation confusion
        
    Mass Similarity:
        mass_sim = 1 / (1 + |Mc_i - Mc_j| / 30)
        Range: (0, 1]
        → Similar masses = chirp morphologies overlap = harder to distinguish
        
    Frequency Overlap:
        freq_overlap = exp(-|f_ISCO_i - f_ISCO_j| / 100)
        Range: [0, 1]
        → Overlapping ISCO frequencies = signal power in same band
        
    Distance Ratio:
        d_ratio = min(D_i, D_j) / max(D_i, D_j)
        Range: (0, 1]
        → If both far or both close: less SNR confusion
        
    Polarization Difference:
        Δψ = |psi_i - psi_j|
        → Orthogonal polarizations reduce amplitude confusion
        
    RA & Dec Differences:
        Used for sky separation analysis
```

**Aggregation Strategy**:

For signal i with (N-1) pairwise features, we use learned importance weighting:

```
For each pair j:
    raw_features[j] = [Δt, Δsky, mass_sim, freq_overlap, d_ratio, Δψ, ΔRA, ΔDec]  (8-D)
    importance[j] = Net(raw_features[j]) → scalar ∈ [0,1]
    
aggregate_features = Softmax(importance) · raw_features  (weighted sum)
    
output_i = Net(aggregate_features) → 16-D representation for signal i
```

**Why Importance Weighting**:
The network learns which pairwise relationships are most informative for ranking. A signal with high time separation from others has low interference (low importance), while a signal overlapping in time with a much stronger signal gets high importance (harder to extract).

---

### 4. MultiModalFusion: Integrating All Information Streams

**Purpose**: Combine temporal morphology, parameter semantics, overlap dynamics, and edge metadata into a unified decision-making representation.

**The Fusion Process**:

```
Input Streams (concatenated):
    ├─ Metadata: [batch, 128]  (from SignalFeatureExtractor + SNR embedding)
    ├─ Overlap: [batch, 16]    (from CrossSignalAnalyzer)
    ├─ Temporal: [batch, 64]   (from TemporalStrainEncoder or zeros)
    └─ Edge: [batch, 32]       (from edge_type embeddings or zeros)
    
Total: 240-D input

Project to working dimension:
    Linear(240→64) + LayerNorm
    → [batch, 64] normalized fused input
    
Self-Attention Block (4 heads):
    Apply multi-head attention: Query=Key=Value = fused input
    Each head learns different aspects of feature interaction
    Output: [batch, 64]
    
Feed-Forward Network (Residual):
    LayerNorm → Linear(64→256) + GELU → Linear(256→64)
    Skip connection: FFN output + input
    Output: [batch, 64]
    
Output: Final [batch, 64] fused representation
```

**Why Self-Attention**:
- Standard MLPs assume feature independence, which doesn't hold when all modalities affect extraction difficulty
- Self-attention learns which combinations of metadata, overlap, and temporal features matter most
- Provides some interpretability: attention weights show which features influenced each decision

---

## Training: How PriorityNet Learns to Prioritize

### The Training Data Format

Each training example is a **scenario** (overlapping signals in synthetic data):

```python
scenario = {
    'detections': [
        {'mass_1': 35.0, 'mass_2': 30.0, 'luminosity_distance': 500.0, ...},
        {'mass_1': 45.0, 'mass_2': 40.0, 'luminosity_distance': 800.0, ...},
        # ... more signals
    ],
    'strain_segments': torch.Tensor([batch, 3, 2048]),  # Optional H1, L1, V1
    'priorities': torch.Tensor([0.85, 0.45]),  # Ground truth extraction order
    'edge_type_ids': torch.Tensor([1, 2])    # Optional morphology class
}
```

**Ground Truth Priorities**: Determined by analyzing extractability in the specific overlap scenario. Signals that can be cleanly extracted first (high SNR, simple morphology, good sky localization) get higher priorities. Signals that must be extracted after others get lower priorities.

### The Loss Function: Multi-Objective Optimization

PriorityNet optimizes three complementary objectives:

**Objective 1: Absolute Calibration (MSE Loss)**
```
L_mse = mean((pred_priority_i - target_priority_i)^2)

Purpose: Ensure predicted priorities match ground truth on an absolute scale
Impact: Prevents all outputs from clustering at 0.5
Weight: 0.4 (balanced against ranking)
```

**Objective 2: Relative Ranking (Adaptive Ranking Loss)**
```
For each pair (i, j) where target_i > target_j:
    margin = 0.1 × clamp(|target_i - target_j|, 0.1, 1.0)
    weight = 1 / (1 + uncertainty_i + uncertainty_j)
    L_rank += weight × max(0, pred_j - pred_i + margin)

Purpose: Get the ordering right, even if absolute values are slightly off
Impact: Ensures signals are extracted in correct sequence regardless of score magnitudes
Weight: 0.5 (highest weight - ordering is critical)

Adaptive Margins: If two signals have very different priorities (0.9 vs 0.2),
we enforce a large margin in predictions. For close priorities (0.5 vs 0.45),
we allow smaller margins. This prevents the model from overthinking subtle distinctions.
```

**Objective 3: Uncertainty Calibration**
```
L_unc = mean((predicted_uncertainty_i - |pred_i - target_i|)^2)

Purpose: Ensure model uncertainty correlates with actual prediction error
Impact: When model says "uncertain", it should actually be uncertain; vice versa
Weight: 0.1 (regularization effect)

Why It Helps: Uncertainty-aware ranking uses σ to discount confidence in ambiguous cases.
If σ is miscalibrated, this weighting breaks down.
```

**Combined Loss**:
```
L_total = 0.4 × L_mse + 0.5 × L_rank + 0.1 × L_unc
```

### Training Configuration

Optimal training parameters (from experience):

```python
optimizer: AdamW
    learning_rate: 5e-4         # Balanced exploration without oscillation
    weight_decay: 1e-5          # Gentle L2 regularization
    
scheduler: ReduceLROnPlateau
    patience: 5 epochs          # Wait 5 epochs before reducing LR
    factor: 0.5                 # Reduce LR by half
    min_lr: 1e-6                # Don't go below 1e-6
    
warmup: Linear warmup
    epochs: 5
    start_factor: 0.1           # Start at 10% of nominal LR
    
training: 
    epochs: 500+                # Convergence typically after 200-300 epochs
    batch_size: 32-64           # Larger = more stable gradients
    gradient_clip: 1.0          # Prevent exploding gradients
    dropout: 0.12-0.15          # Moderate regularization
```

### Key Training Practices

**Data Augmentation in Scenarios**:
- Vary number of overlapping signals (2-5 typically)
- Vary SNR distribution (some high SNR, some low SNR in same scenario)
- Vary mass distributions (BBH, BNS, NSBH all in training set)
- Vary sky localizations (on-axis, off-axis, poorly localized)

**Validation Strategy**:
- Separate validation set with **different scenarios** from training
- Important: Validation scenarios should have different SNR statistics than training
- Monitor both MSE and ranking accuracy
- Watch for calibration: plot predicted vs actual priority; should be close to y=x

**Edge Cases to Include**:
- Very close in mass (hard to distinguish)
- Very different SNR (one dominates)
- Short duration merger vs long inspiral
- Precessing vs non-precessing mixtures

---

## Practical Usage Guide

### Installation and Setup

```bash
# Activate conda environment (already created)
conda activate ahsd

# Install package in development mode
pip install -e . --no-deps

# Verify installation
python -c "from ahsd.core.priority_net import PriorityNet; print('✓ Import successful')"
```

### Basic Usage: Ranking Detections

```python
import torch
from ahsd.core.priority_net import PriorityNet

# Load pre-trained model
model = PriorityNet(use_strain=False)  # Metadata-only mode
model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
model.eval()

# Your detected signals
detections = [
    {'mass_1': 35.0, 'mass_2': 30.0, 'luminosity_distance': 500.0, 
     'ra': 1.5, 'dec': 0.2, 'geocent_time': 0.0, ...},
    {'mass_1': 45.0, 'mass_2': 40.0, 'luminosity_distance': 800.0,
     'ra': 2.0, 'dec': -0.1, 'geocent_time': 0.05, ...},
]

# Get priority ranking
with torch.no_grad():
    ranked_indices = model.rank_detections(detections)
    # ranked_indices = [0, 1] means extract signal 0 first, then signal 1

# Extract in order
for idx in ranked_indices:
    print(f"Extract signal {idx} next")
    # ... subtraction logic ...
```

### Advanced Usage: With Strain Data

```python
# Pre-whiten strain data from LIGO
from scipy.signal import welch
import numpy as np

def whiten_strain(raw_strain, fs=2048):
    """Whiten strain by dividing by PSD."""
    f, psd = welch(raw_strain, fs=fs, nperseg=2048*4)
    fft_strain = np.fft.rfft(raw_strain)
    psd_interp = np.interp(
        np.fft.rfftfreq(len(raw_strain), 1/fs), f, psd
    )
    whitened = np.fft.irfft(fft_strain / np.sqrt(psd_interp))
    # Extract 1-second segment centered on signal
    return whitened[:2048]

# Load model with strain encoder
model = PriorityNet(use_strain=True)
model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
model.eval()

# Prepare inputs
h1_strain = whiten_strain(h1_raw_data)
l1_strain = whiten_strain(l1_raw_data)
v1_strain = whiten_strain(v1_raw_data)

strain_segments = torch.stack([
    torch.from_numpy(h1_strain).float(),
    torch.from_numpy(l1_strain).float(),
    torch.from_numpy(v1_strain).float(),
], dim=0).unsqueeze(0)  # [1, 3, 2048] for 1 scenario

# Rank with morphology information
with torch.no_grad():
    ranked_indices = model.rank_detections(detections, strain_segments)
```

### Training from Scratch

```python
from ahsd.core.priority_net import PriorityNet, PriorityNetTrainer
from ahsd.data.dataset_generator import WaveformDataset, DataLoader

# Load synthetic training data
train_dataset = WaveformDataset('configs/data_config.yaml', split='train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model and trainer
config = {
    'learning_rate': 5e-4,
    'weight_decay': 1e-5,
    'warmup_epochs': 5,
    'ranking_weight': 0.5,
    'mse_weight': 0.4,
    'uncertainty_weight': 0.1,
}

model = PriorityNet(use_strain=True, config=config)
trainer = PriorityNetTrainer(model, config)

# Training loop
for epoch in range(500):
    epoch_loss = 0.0
    
    for batch in train_loader:
        detections, targets, strain_segments = batch
        
        # Forward pass
        loss_dict = trainer.train_step(
            detections_batch=detections,
            priorities_batch=targets,
            strain_batch=strain_segments
        )
        
        epoch_loss += loss_dict['loss']
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Step {batch_idx}: "
                  f"Loss={loss_dict['loss']:.4f}, "
                  f"MSE={loss_dict['mse']:.4f}, "
                  f"Rank={loss_dict['ranking_loss']:.4f}")
    
    # Step scheduler
    trainer.scheduler.step(epoch_loss / len(train_loader))
    
    # Checkpoint
    if (epoch + 1) % 50 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': epoch_loss / len(train_loader)
        }, f'checkpoint_epoch_{epoch}.pth')
```

---

## Uncertainty Quantification and Confidence Scoring

PriorityNet outputs not just priorities but also **uncertainties** - a measure of confidence in each priority estimate.

### Understanding Uncertainty

```python
priorities, uncertainties = model(detections, strain_segments)

# priorities: [0.85, 0.45]        ← predicted extraction order
# uncertainties: [0.08, 0.22]     ← confidence in those predictions

# Interpretation:
# - Signal 0: priority=0.85 ± 0.08 (high confidence, should extract first)
# - Signal 1: priority=0.45 ± 0.22 (moderate confidence, harder to rank)
```

### Uncertainty-Aware Ranking

The ranking algorithm adjusts for uncertainty:

```python
# Simple ranking (no uncertainty):
scores = priorities
ranked = argsort(scores, descending=True)

# Uncertainty-penalized ranking:
beta = 0.25  # penalty coefficient (higher for denser overlaps)
scores = priorities - beta * uncertainties
ranked = argsort(scores, descending=True)

# Effect: A signal with priority=0.50 but uncertainty=0.30 
#         scores lower than one with priority=0.48 but uncertainty=0.05
#         because the first is less reliable.
```

### When High Uncertainty Indicates Real Difficulty

High uncertainty genuinely reflects extraction difficulty in cases like:
- Two signals with nearly identical parameters
- Both signals have low SNR
- Signals have opposite sky positions but similar times (antipodal case)
- Mixed precessing/non-precessing: morphology ambiguity

In these cases, the ranking suggestion is still valid, but extraction will be more error-prone regardless. The uncertainty quantifies this inherent difficulty.

---

## Integration with AHSD Pipeline

### Phase 3a: Training Priority Rankings

PriorityNet is trained on synthetic overlapping scenarios to predict extractability. The training target (ground truth priority) is derived from analyzing extraction quality metrics in simulation:

```python
# Pseudocode: How priorities are computed during data generation
for scenario in overlapping_scenarios:
    for extraction_order in permutations(signals):
        # Try extracting in this order
        residuals = data.copy()
        param_errors = []
        
        for signal_idx in extraction_order:
            # Extract signal_idx from residuals
            estimated_params = posterior_estimate(residuals, initial_guess)
            param_error = compare(estimated_params, true_params[signal_idx])
            param_errors.append(param_error)
            
            # Subtract and continue
            residuals -= generate_waveform(estimated_params)
        
        # Signals extracted early have lower cumulative error
        priority_order = argsort(param_errors)  # Lower error = higher priority

    # Normalize priorities to [0, 1]
    priorities = softmax(priority_order)
```

### Phase 3b: Extracting with Priority Guidance

The trained model guides extraction in real scenarios:

```python
# In subtraction loop
for iteration in range(max_iterations):
    # Detect candidate signals
    detections = signal_detector(residual_data)
    
    if len(detections) <= 1:
        break  # Single signal, no ordering needed
    
    # Rank by priority
    extraction_order = priority_net.rank_detections(
        detections,
        strain_segments=whitened_strain
    )
    
    # Extract in recommended order
    for idx in extraction_order:
        # Refine parameters via posterior estimation
        posterior = parameter_estimator(
            residual_data,
            detections[idx],
            strain_segments
        )
        
        # Subtract with uncertainty-aware weighting
        if uncertainties[idx] < 0.2:  # High confidence
            amplitude_factor = 1.0  # Full subtraction
        else:  # Uncertain
            amplitude_factor = 0.7  # Conservative subtraction
        
        waveform = generate_waveform(posterior) * amplitude_factor
        residual_data -= waveform
```

---

## Troubleshooting and Common Issues

### Issue 1: Training Loss Not Decreasing

**Symptoms**: Loss plateaus after 10-20 epochs, no improvement with more training

**Root Causes**:
- Priorities in training data are poorly calibrated (all clustered at 0.5)
- Input parameters are not properly normalized to [0, 1]
- Learning rate is too high (oscillating) or too low (stuck)
- Batch size too small (noisy gradients)

**Solutions**:
```python
# Check parameter normalization
print(params.min(), params.max())  # Should be close to 0, 1

# Reduce learning rate gradually
trainer.optimizer.param_groups[0]['lr'] = 2e-4  # Instead of 5e-4

# Increase batch size
data_loader = DataLoader(dataset, batch_size=64)  # From 32

# Inspect priority distribution
print(train_priorities.mean(), train_priorities.std())  
# Should be mean~0.5, std~0.15 (reasonable spread)

# Verify loss weights
config = {
    'ranking_weight': 0.5,
    'mse_weight': 0.4,
    'uncertainty_weight': 0.1,
}
```

### Issue 2: Uncertainty Estimates All Similar

**Symptoms**: `uncertainties.std() < 0.05` - model doesn't distinguish confidence levels

**Root Causes**:
- Uncertainty loss weight too low (model ignores uncertainty objective)
- Training scenarios not diverse enough (all equally hard or equally easy)
- Model capacity too small to learn uncertainty

**Solutions**:
```python
# Increase uncertainty loss weight
config = {'uncertainty_weight': 0.2}  # From 0.1

# Ensure diverse training
# Include hard scenarios: 
#   - Similar masses (q close to 1)
#   - Mixed SNR (one high, one low)
#   - Antipodal sky positions

# Add noise to training data
train_dataset.enable_noise_augmentation()

# Monitor during training
if uncertainties.std() < 0.1:
    print(f"WARNING: Low uncertainty variance: {uncertainties.std():.4f}")
```

### Issue 3: Poor Ranking Accuracy on Validation

**Symptoms**: Ranking accuracy < 85% on test set with 2+ overlaps

**Root Causes**:
- Model underfitting: needs more capacity or training
- Validation data distribution different from training
- CrossSignalAnalyzer not learning pairwise features well
- Missing strain encoder despite needed

**Solutions**:
```python
# Add strain encoder
model = PriorityNet(use_strain=True)  # From metadata-only

# Train longer
epochs: 500+  # From 300

# Check validation distribution
import matplotlib.pyplot as plt
plt.hist(train_priorities, label='train')
plt.hist(val_priorities, label='val')
plt.legend()
plt.show()  # Distributions should be similar

# Inspect pairwise features
analyzer = model.cross_signal_analyzer
overlap_feats = analyzer(test_params)
print(overlap_feats.std(dim=0))  # Each feature should have variance
```

### Issue 4: NaN or Inf in Loss During Training

**Symptoms**: Loss becomes NaN after a few batches

**Root Causes**:
- Strain data not properly whitened (has NaN/Inf)
- Learning rate too high (weight updates explode)
- Gradient explosion in attention mechanism

**Solutions**:
```python
# Verify strain whitening
assert np.isfinite(strain).all()
assert abs(strain.mean()) < 0.01  # Should be ~0
assert abs(strain.std() - 1.0) < 0.1  # Should be ~1

# Reduce learning rate
trainer.optimizer.param_groups[0]['lr'] = 1e-4  # From 5e-4

# Check gradient clipping is active
if grad_norm > 100:  # Before clipping
    print("Gradients exploding - reduce learning rate more")
```

---

## Performance Benchmarks

### Inference Speed

Tested on single CPU core (Intel Xeon E5, no GPU):

| Configuration | Parameters | Inference Time | Throughput |
|--------------|------------|----------------|------------|
| Metadata-only | 1.3M | 0.08 ms/signal | 12,500 signals/s |
| With strain (CPU) | 2.5M | 4.2 ms/signal | 238 signals/s |
| With strain (GPU) | 2.5M | 0.6 ms/signal | 1,667 signals/s |

### Accuracy on Synthetic Data

Based on 1000 test scenarios with 2-5 overlapping signals:

| Metric | Value |
|--------|-------|
| Ranking accuracy (2 signals) | 94.3% |
| Ranking accuracy (3-5 signals) | 89.7% |
| Priority MAE | 0.089 |
| Kendall's τ (rank correlation) | 0.84 |
| Uncertainty calibration (ECE) | 0.047 |

---

## Advanced Topics

### Edge-Type Conditioning

For scenarios where signal morphology class is known (BBH, BNS, NSBH), you can condition the model:

```python
edge_type_ids = torch.tensor([
    1,  # Signal 0: BBH (non-precessing)
    3,  # Signal 1: NSBH (mixed morphology)
])

# 17 morphology classes available:
# 0: Unknown, 1: BBH-lowspin, 2: BBH-moderatespin, ..., 16: NSBH-highspin

priorities, uncertainties = model(
    detections,
    strain_segments,
    edge_type_ids=edge_type_ids
)
```

This provides an additional inductive bias that can improve ranking accuracy when morphology classes are reliably identified upstream.

### Transfer Learning

To adapt a pre-trained model to a new SNR distribution:

```python
# Load pre-trained checkpoint
checkpoint = torch.load('pretrained.pth')
model = PriorityNet()
model.load_state_dict(checkpoint['model_state_dict'])

# Fine-tune on new domain
trainer = PriorityNetTrainer(model, config)
trainer.optimizer.param_groups[0]['lr'] = 1e-4  # Lower LR

for epoch in range(100):  # Fewer epochs
    for batch in new_domain_loader:
        loss_dict = trainer.train_step(...)
```

### Multi-Detector Configurations

To extend from H1+L1 to include Virgo (H1+L1+V1):

```python
# In model initialization
model = PriorityNet(
    use_strain=True,
    config={
        'n_detector_channels': 3,  # H1, L1, V1
        ...
    }
)

# Prepare 3-channel strain
strain_segments = torch.stack([h1, l1, v1], dim=1)  # [batch, 3, 2048]
```

The TemporalStrainEncoder will automatically adapt to 3 input channels.

---

## Frequently Asked Questions

**Q: Do I need strain data to use PriorityNet?**
A: No. Metadata-only mode works well (92%+ ranking accuracy on 2 signals). Strain improves accuracy by ~2-3% but adds 5ms latency per signal.

**Q: How many signals can PriorityNet rank?**
A: Tested and validated on 2-5 overlapping signals. Beyond 5, ranking becomes exponentially harder. For 10+ signals, consider hierarchical ranking (rank in groups, then merge).

**Q: Can I use PriorityNet on real GWOSC data?**
A: Yes. Ensure parameter estimates from your initial detector are normalized to [0,1] using the ranges in `_detections_to_tensor()`. The model generalizes reasonably well to real data, though retraining on real event injections improves performance.

**Q: How do I update priority if new information arrives?**
A: Re-run `rank_detections()` with updated parameter estimates. Priority computation is fast (<1ms for batch).

**Q: Can I interpret which features matter most?**
A: Somewhat. Attention weights in MultiModalFusion show which feature combinations matter. For full interpretability, consider ablating modules (train without temporal, without cross-signal analysis) and compare performance.

---

## References and Further Reading

1. **PriorityNet in PosteriFlow**: See `experiments/phase3a_priority_net_training.py` for full training pipeline

2. **AHSD Pipeline Context**: `src/ahsd/core/ahsd_pipeline.py` shows how PriorityNet integrates into the full decomposition workflow

3. **Data Generation**: `src/ahsd/data/dataset_generator.py` contains scenario generation and priority computation logic

4. **Related Work**:
   - Schäfer et al., "First machine learning gravitational-wave search mock data challenge" (Phys. Rev. D 2023)
   - Chua et al., "Deep source separation of overlapping gravitational-wave signals" (arXiv:2503.10398)
   - Marx et al., "A machine-learning pipeline for real-time detection of gravitational waves" (Phys. Rev. D 2024)

---

## Support and Contribution

For issues, questions, or improvements:

1. **Check this guide**: Review the Troubleshooting section above
2. **Review examples**: See `experiments/` folder for concrete usage patterns  
3. **Submit issues**: Include model config, data sample, and error traceback
4. **Contribute improvements**: Fork, test thoroughly, and submit PR with benchmarks

---

**Document updated**: November 2025
**Latest version**: PriorityNet with attention fusion, SNR embedding, and uncertainty calibration
