# Inference Setup & Model Usage Guide

**Date:** November 14, 2025  
**Purpose:** How to setup and use the OverlapNeuralPE model for inference testing

---

## Model Architecture Overview

```
OverlapNeuralPE (Neural Posterior Estimation)
├── 1. Normalizing Flow (NSF)           → Posterior distribution
├── 2. Context Encoder                   → Encodes strain data features  
├── 3. Uncertainty Estimator             → Prediction confidence
├── 4. Bias Corrector                    → Hierarchical bias correction
├── 5. Priority Net (optional)           → Signal prioritization
├── 6. RL Controller                     → Adaptive complexity
└── 7. Adaptive Subtractor               → Iterative signal extraction
```

---

## Current Data Flow

### Training Loop (phase3a_neural_pe.py)
```
DataLoader
    ↓
(strain_data, parameters, n_signals, metadata)
    ↓
model.compute_loss(strain_data, target_params)
    ↓
loss.backward() → optimizer.step()
```

### Key Methods
- **`compute_loss()`** - Used during training (lines 535, 419)
- **`extract_overlapping_signals()`** - Iterative extraction (line 512)
- **`extract()`** - Public API wrapper (line 681)
- **`sample_posterior()`** - Draw samples from learned posterior (line 328)

---

## When/How extract() is Called

### Current Usage Status
```
❌ NOT called during normal training (only compute_loss is used)
⚠️  Available but not hooked into training loop
✅ Can be used for inference/testing
```

### Proper Usage Pattern

#### Option A: Training Mode (collect RL data)
```python
# During training with ground truth parameters
result = model.extract_overlapping_signals(
    strain_data=batch_strain,           # [batch, n_det, n_samples]
    true_params=batch_true_params,      # Ground truth for RL reward
    training=True                       # Enable RL data collection
)
# Returns:
# {
#     'extracted_signals': [...],       # Detected signals
#     'residual': tensor,               # Remaining strain
#     'iterations': int,                # How many extracted
#     'rl_metrics': {...}               # RL controller state
# }
```

#### Option B: Inference Mode (no ground truth)
```python
# During inference without ground truth
result = model.extract_overlapping_signals(
    strain_data=test_strain,            # [batch, n_det, n_samples]
    true_params=None,                   # No ground truth
    training=False                      # Use residual power as reward
)
# Returns same structure but with inference-only metrics
```

#### Option C: Direct Posterior Sampling
```python
# For single event parameter estimation
strain_tensor = torch.from_numpy(strain_data).unsqueeze(0)
posterior_samples = model.sample_posterior(
    strain_data=strain_tensor,
    n_samples=1000,
    batch_size=100
)
# Returns: {'samples': [batch, n_samples, 9 params], 'means': [...], ...}
```

---

## Inference Testing Setup

### Step 1: Load Trained Model
```python
import torch
from src.ahsd.models.overlap_neuralpe import OverlapNeuralPE
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configuration
config_path = Path('configs/enhanced_training.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

# Initialize model
model = OverlapNeuralPE(config, device=device)

# Load checkpoint
checkpoint = torch.load('models/neural_pe/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Step 2: Prepare Test Data
```python
import numpy as np
from torch.utils.data import DataLoader

# Option A: From dataset
test_dataset = # your dataset loader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Option B: Generate test event
strain_h1 = np.random.randn(16384) * 1e-23  # 4s at 4096 Hz
strain_l1 = np.random.randn(16384) * 1e-23
strain_data = torch.tensor([
    [strain_h1, strain_l1]
], dtype=torch.float32).to(device)  # [1, 2, 16384]
```

### Step 3: Run Inference
```python
with torch.no_grad():
    # Method 1: Full extraction with iterative subtraction
    result = model.extract_overlapping_signals(
        strain_data=strain_data,
        true_params=None,
        training=False
    )
    extracted_signals = result['extracted_signals']
    
    # Method 2: Single signal posterior sampling
    posterior = model.sample_posterior(
        strain_data=strain_data,
        n_samples=1000
    )
    posterior_means = posterior['means']
    posterior_stds = posterior['stds']
    
    # Method 3: Get metrics
    metrics = model.get_metrics()
```

---

## Using ahsd_pipeline.py for Inference Testing

### Current Status of ahsd_pipeline.py
```
✅ Components present:
  - BiasCorrector integration
  - AdaptiveSubtractor integration
  - Signal prioritization (heuristic + PriorityNet)
  - Performance metrics computation
  
❌ Issues:
  - Built for pre-trained prioritization
  - No normalization flow integration
  - No uncertainty estimation
  - No RL controller
  
⚠️  Can be used as: Lighter inference wrapper (no deep learning)
```

### Compatibility Assessment

| Feature | ahsd_pipeline.py | OverlapNeuralPE | Needed for |
|---------|-----------------|-----------------|-----------|
| Signal Extraction | ✅ Heuristic | ✅ Neural | Posterior inference |
| Bias Correction | ✅ Yes | ✅ Yes | Parameter quality |
| Prioritization | ✅ Heuristic | ✅ PriorityNet | Multi-signal ordering |
| Uncertainty Quantif. | ❌ No | ✅ Yes | Confidence bounds |
| RL Adaptive Complexity | ❌ No | ✅ Yes | Dynamic tuning |
| Normalizing Flow | ❌ No | ✅ Yes | Posterior distribution |

### Recommendation: Create Hybrid Inference Pipeline

Rather than trying to make ahsd_pipeline.py compatible, **create a new inference wrapper**:

```python
# NEW: src/ahsd/inference/inference_pipeline.py

class InferencePipeline:
    """
    Unified inference pipeline combining:
    - OverlapNeuralPE (neural extraction)
    - BiasCorrector (parameter correction)
    - MetricsComputer (quality assessment)
    """
    
    def __init__(self, model_path, config_path, device='cuda'):
        self.model = OverlapNeuralPE.from_checkpoint(model_path, config_path, device)
        self.device = device
    
    def extract(self, strain_data, return_samples=False, n_samples=1000):
        """
        Extract signals from strain data.
        
        Args:
            strain_data: [batch, n_det, n_samples] tensor
            return_samples: Whether to return posterior samples
            n_samples: Number of posterior samples
        
        Returns:
            {
                'signals': [extracted signals],
                'parameters': {means, stds, quantiles},
                'metrics': {quality, snr, ...},
                'posterior_samples': (optional) [batch, n_samples, 9 params]
            }
        """
        with torch.no_grad():
            # Extract using neural flow
            result = self.model.extract_overlapping_signals(
                strain_data=strain_data,
                true_params=None,
                training=False
            )
            
            # Get posterior samples if requested
            if return_samples:
                posterior = self.model.sample_posterior(
                    strain_data=strain_data,
                    n_samples=n_samples
                )
                result['posterior_samples'] = posterior['samples']
            
            return result
    
    def get_posteriors(self, strain_data, n_samples=1000):
        """Get parameter posterior distributions."""
        return self.model.sample_posterior(strain_data, n_samples)
    
    def get_credible_intervals(self, strain_data, credibility=0.90):
        """Get credible intervals for parameters."""
        posterior = self.get_posteriors(strain_data, n_samples=5000)
        samples = posterior['samples'][0]  # [n_samples, 9 params]
        
        param_names = ['mass_1', 'mass_2', 'distance', 'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time']
        intervals = {}
        
        for i, name in enumerate(param_names):
            param_samples = samples[:, i]
            lower = np.percentile(param_samples, (1 - credibility) / 2 * 100)
            upper = np.percentile(param_samples, (1 + credibility) / 2 * 100)
            intervals[name] = {'lower': lower, 'upper': upper, 'median': np.median(param_samples)}
        
        return intervals
```

---

## How to Hook Extract() into Training

Currently `extract()` is NOT called during training. To enable RL data collection:

### Modification 1: Update Training Loop
**File:** `experiments/phase3a_neural_pe.py` (line ~419)

```python
# Current (training only):
loss_dict = self.model.compute_loss(strain_data, target_params)

# Option A: Call extract() with training=True
# (After ensuring true_params are available in batch)
result = self.model.extract(
    strain_data=strain_data,
    training=True  # Enable RL
)
loss_dict = self.model.compute_loss(strain_data, target_params)

# Option B: Separate extraction and loss computation
# (Extract for RL, then compute loss)
if batch_idx % 10 == 0:  # Every N batches
    extraction_result = self.model.extract(
        strain_data=strain_data,
        training=True
    )
    rl_metrics = extraction_result.get('rl_metrics', {})
    self.logger.info(f"RL Metrics: {rl_metrics}")
```

### Modification 2: Prepare True Parameters in Batch
**File:** Data loader or collate_fn

```python
def collate_fn(batch):
    """
    Collate function that includes ground truth parameters for RL.
    """
    strain_list = [item['strain'] for item in batch]
    params_list = [item['parameters'] for item in batch]
    
    strain_data = torch.stack(strain_list)  # [batch, n_det, n_samples]
    parameters = torch.stack(params_list)   # [batch, n_signals, 9 params]
    
    # Extract first signal true parameters for RL reward
    true_params_list = []
    for i in range(len(batch)):
        # Get first signal (highest priority) true parameters
        true_params_list.append(parameters[i, 0, :])
    
    true_params = torch.stack(true_params_list)  # [batch, 9]
    
    return strain_data, parameters, true_params, metadata
```

---

## Inference Testing Script Template

```python
#!/usr/bin/env python3
"""
Inference testing script for OverlapNeuralPE
"""

import torch
import numpy as np
import logging
from pathlib import Path
from src.ahsd.models.overlap_neuralpe import OverlapNeuralPE
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    logger.info("Loading model...")
    config_path = Path('configs/enhanced_training.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model = OverlapNeuralPE(config, device=device)
    checkpoint = torch.load('models/neural_pe/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate test data
    logger.info("Generating test data...")
    strain_h1 = torch.randn(1, 16384, device=device) * 1e-23
    strain_l1 = torch.randn(1, 16384, device=device) * 1e-23
    strain_data = torch.stack([strain_h1, strain_l1], dim=1)  # [1, 2, 16384]
    
    # Test 1: Extract with iteration
    logger.info("Test 1: Iterative extraction...")
    with torch.no_grad():
        result = model.extract(strain_data=strain_data, training=False)
        logger.info(f"  Extracted {len(result['extracted_signals'])} signals")
    
    # Test 2: Sample posterior
    logger.info("Test 2: Posterior sampling...")
    with torch.no_grad():
        posterior = model.sample_posterior(strain_data=strain_data, n_samples=100)
        logger.info(f"  Posterior shape: {posterior['samples'].shape}")
        logger.info(f"  Means: {posterior['means']}")
    
    # Test 3: Get metrics
    logger.info("Test 3: Model metrics...")
    metrics = model.get_metrics()
    logger.info(f"  Metrics: {metrics}")
    
    logger.info("✅ All tests passed!")

if __name__ == '__main__':
    test_inference()
```

---

## Data Format Specifications

### Input Format (strain_data)
```
Shape: [batch, n_detectors, n_samples]
- batch: Number of events (usually 1 for inference)
- n_detectors: 2 (H1, L1) or 3 (H1, L1, V1)
- n_samples: 16384 (4 seconds at 4096 Hz)
- dtype: torch.float32
- Device: CUDA or CPU
- Value range: ~1e-23 to 1e-20 (strain units)
```

### Output Format (extracted_signals)
```python
[
    {
        'parameters': array[9],          # mass_1, mass_2, distance, ra, dec, theta_jn, psi, phase, time
        'parameter_uncertainties': array[9],
        'snr': float,
        'quality': float,
        'iteration': int,
        'complexity': 'low'|'medium'|'high',
        'bias_corrected': bool,
        'posterior_summary': {
            'mass_1': {'median': 30.5, 'std': 2.1, ...},
            ...
        }
    },
    ...
]
```

### Output Format (posterior samples)
```python
{
    'samples': tensor[batch, n_samples, 9],    # Raw samples
    'means': tensor[batch, 9],                  # Parameter means
    'stds': tensor[batch, 9],                   # Parameter stds
    'quantiles': tensor[batch, 9, 5],          # 5%, 25%, 50%, 75%, 95%
    'context': tensor[batch, context_dim]      # Encoded input features
}
```

---

## Success Criteria for Inference

- [ ] Model loads without errors
- [ ] Inference runs in <1s per event (GPU)
- [ ] Posterior samples are within physical bounds
- [ ] Posterior credible intervals contain ground truth (when known)
- [ ] Metrics show meaningful values (epsilon < 0.1, rewards non-zero)
- [ ] RL controller selects varying complexity levels
- [ ] Bias correction applied successfully

---

## Next Steps

1. **Create InferencePipeline wrapper** (recommended)
   - File: `src/ahsd/inference/inference_pipeline.py`
   - Provides clean API for inference

2. **Add inference entry point**
   - File: `src/ahsd/cli/inference.py`
   - Command-line tool for inference

3. **Hook extract() into training** (for RL)
   - Modify: `experiments/phase3a_neural_pe.py`
   - Enable RL data collection during training

4. **Create inference test suite**
   - File: `tests/test_inference.py`
   - Verify inference functionality

5. **Document model loading**
   - Update: `README.md`
   - Add inference examples
