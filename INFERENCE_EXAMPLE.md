# Inference Example: Real Data → Predicted Parameters

## Quick Example

**You provide:** Strain data from LIGO/Virgo detectors  
**Model does:** Neural posterior estimation with uncertainty quantification  
**You get:** Predicted masses, distances, sky position + uncertainties

---

## Step-by-Step Example

### INPUT: Real Gravitational Wave Strain Data

```python
import torch
import numpy as np

# Real detector data (4 seconds at 4096 Hz)
strain_h1 = load_from_detector('H1_data.hdf5')  # Shape: [16384]
strain_l1 = load_from_detector('L1_data.hdf5')  # Shape: [16384]

# Combine into tensor: [batch=1, detectors=2, samples=16384]
strain_data = torch.tensor([
    [strain_h1, strain_l1]
], dtype=torch.float32)

print(strain_data.shape)
# Output: torch.Size([1, 2, 16384])
```

---

### PROCESSING: Neural Posterior Estimation

```python
from src.ahsd.models.overlap_neuralpe import OverlapNeuralPE
import yaml

# Load trained model
config = yaml.safe_load(open('configs/enhanced_training.yaml'))
model = OverlapNeuralPE(config, device='cuda')
checkpoint = torch.load('models/neural_pe/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    posterior = model.sample_posterior(
        strain_data=strain_data,
        n_samples=1000  # Draw 1000 samples from posterior
    )
```

**What happens internally:**
1. **Context Encoder** reads strain data → extracts features
2. **Normalizing Flow (NSF)** learns posterior distribution from features
3. **1000 samples** drawn from the learned posterior
4. **Samples denormalized** from [-1,1] range to physical units

---

### OUTPUT: Predicted Parameters with Uncertainties

```python
# Extract and display results
param_names = [
    'mass_1',              # Primary mass
    'mass_2',              # Secondary mass
    'luminosity_distance', # Distance to source
    'ra', 'dec',          # Sky position
    'theta_jn',           # Inclination angle
    'psi',                # Polarization angle
    'phase',              # Orbital phase
    'geocent_time'        # Time of merger
]

# Get samples for first event in batch
samples = posterior['samples'][0]  # Shape: [1000, 9]

print("=" * 80)
print("PARAMETER ESTIMATES")
print("=" * 80)

for i, name in enumerate(param_names):
    param_samples = samples[:, i].cpu().numpy()
    
    # Compute statistics from 1000 samples
    median = np.median(param_samples)
    std = np.std(param_samples)
    lower_90 = np.percentile(param_samples, 5)    # 5th percentile
    upper_90 = np.percentile(param_samples, 95)   # 95th percentile
    
    print(f"\n{name}")
    print(f"  Median (point estimate): {median:.3f}")
    print(f"  Std Dev (uncertainty):   ±{std:.3f}")
    print(f"  90% Credible Interval:   [{lower_90:.3f}, {upper_90:.3f}]")
```

**Example Output:**

```
================================================================================
PARAMETER ESTIMATES
================================================================================

mass_1
  Median (point estimate): 35.247
  Std Dev (uncertainty):   ±2.145
  90% Credible Interval:   [31.892, 39.561]

mass_2
  Median (point estimate): 29.834
  Std Dev (uncertainty):   ±1.987
  90% Credible Interval:   [26.543, 33.891]

luminosity_distance
  Median (point estimate): 398.456
  Std Dev (uncertainty):   ±52.321
  90% Credible Interval:   [311.234, 502.876]

ra
  Median (point estimate): 2.543
  Std Dev (uncertainty):   ±0.234
  90% Credible Interval:   [2.134, 2.945]

dec
  Median (point estimate): -0.312
  Std Dev (uncertainty):   ±0.189
  90% Credible Interval:   [-0.657, 0.045]

theta_jn
  Median (point estimate): 1.234
  Std Dev (uncertainty):   ±0.456
  90% Credible Interval:   [0.456, 1.987]

psi
  Median (point estimate): 0.823
  Std Dev (uncertainty):   ±0.234
  90% Credible Interval:   [0.412, 1.234]

phase
  Median (point estimate): 1.567
  Std Dev (uncertainty):   ±0.345
  90% Credible Interval:   [0.923, 2.156]

geocent_time
  Median (point estimate): 0.487
  Std Dev (uncertainty):   ±0.012
  90% Credible Interval:   [0.463, 0.511]
```

---

## Interpreting Results

### What Do These Numbers Mean?

| Parameter | Meaning | Example | Interpretation |
|-----------|---------|---------|-----------------|
| **mass_1** | Primary black hole mass | 35.2 ± 2.1 M☉ | Most likely 35.2 M☉, but could be 31-40 M☉ |
| **mass_2** | Secondary black hole mass | 29.8 ± 2.0 M☉ | Binary companion mass |
| **luminosity_distance** | Distance to source | 398 ± 52 Mpc | ~1.3 billion light-years away |
| **ra, dec** | Sky coordinates | ra=2.54, dec=-0.31 rad | Where to point telescope |
| **theta_jn** | How tilted the orbit is | 1.23 ± 0.46 rad | 0=face-on, π/2=edge-on |
| **psi** | Polarization angle | 0.82 ± 0.23 rad | Orientation of GW wave |
| **phase** | Orbital phase at detection | 1.57 ± 0.35 rad | Position in binary orbit |
| **geocent_time** | Merger timestamp | 0.49 ± 0.01 s | When the black holes collided |

---

## Real Data Workflow

### If You Have Real GWOSC Data:

```python
from gwpy.timeseries import TimeSeries
import torch

# Download real data from GWOSC
h1_data = TimeSeries.fetch_open_data(
    'H1',
    start='2019-04-10 02:30:00',
    end='2019-04-10 02:40:00'
)

l1_data = TimeSeries.fetch_open_data(
    'L1',
    start='2019-04-10 02:30:00',
    end='2019-04-10 02:40:00'
)

# Resample to 4096 Hz and extract a 4-second window
h1 = h1_data.resample(4096).value[:]  # numpy array
l1 = l1_data.resample(4096).value[:]

# Normalize to zero mean, unit variance
h1 = (h1 - h1.mean()) / h1.std()
l1 = (l1 - l1.mean()) / l1.std()

# Create tensor for inference
strain_data = torch.tensor([[h1[:16384], l1[:16384]]], dtype=torch.float32)

# Run inference (as shown above)
```

---

## What About Multiple Signals?

If data contains overlapping signals (multiple simultaneous events):

```python
# Model can extract multiple signals iteratively
result = model.extract_overlapping_signals(
    strain_data=strain_data,
    training=False
)

extracted_signals = result['extracted_signals']
# List of dicts, each containing:
# {
#     'parameters': [mass_1, mass_2, distance, ra, dec, theta_jn, psi, phase, time],
#     'parameter_uncertainties': [std_1, std_2, ...],
#     'snr': 12.5,
#     'quality': 0.87,
#     'posterior_samples': [1000, 9]  # Full posterior for this signal
# }

for i, signal in enumerate(extracted_signals):
    print(f"\nSignal {i+1}:")
    print(f"  Mass 1: {signal['parameters'][0]:.1f} ± {signal['parameter_uncertainties'][0]:.1f} M☉")
    print(f"  Mass 2: {signal['parameters'][1]:.1f} ± {signal['parameter_uncertainties'][1]:.1f} M☉")
    print(f"  SNR: {signal['snr']:.1f}")
```

---

## Complete Working Example

Run this to see the full workflow:

```bash
python example_inference_workflow.py
```

This will:
1. ✅ Create synthetic strain data (or load real data)
2. ✅ Load trained model
3. ✅ Run inference on GPU/CPU
4. ✅ Display results with uncertainties
5. ✅ Compare predictions vs. true values
6. ✅ Generate visualization plots

---

## Key Points

- **Input:** Strain data from 2-3 detectors (H1, L1, V1)
- **Output:** 9 physical parameters + full uncertainty quantification
- **Speed:** ~0.5-1.0s per event on GPU
- **Accuracy:** Posterior includes all parameter correlations via flow
- **Robust:** Works with real GWOSC data and multiple overlapping signals
