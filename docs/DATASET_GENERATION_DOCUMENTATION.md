# PosteriFlow Dataset Generation Documentation

## Overview

This document describes the complete dataset generation pipeline for the PosteriFlow project, which creates synthetic gravitational wave (GW) data for training neural posterior estimation and priority network models. The pipeline handles realistic signal generation, overlapping event simulation, noise injection, and comprehensive preprocessing.

**Version**: 2.0 (Physics-Validated, November 2025)
**Key Features**: 
- Realistic O4 LIGO/Virgo sensitivity simulation
- Multi-event overlapping scenarios
- Physics-aware SNR-distance correlation
- Comprehensive edge case handling
- Mixed event-type overlaps (BBH+BNS, BBH+NSBH, BNS+NSBH)

---

## Table of Contents

1. [Configuration System](#configuration-system)
2. [Component Architecture](#component-architecture)
3. [Parameter Sampling](#parameter-sampling)
4. [Waveform Generation](#waveform-generation)
5. [Signal Injection](#signal-injection)
6. [Preprocessing Pipeline](#preprocessing-pipeline)
7. [Dataset Generation Workflow](#dataset-generation-workflow)
8. [Key Formulas and Physics](#key-formulas-and-physics)
9. [Data Structure](#data-structure)
10. [Configuration Reference](#configuration-reference)

---

## Configuration System

### Location
`/home/bibinathomas/PosteriFlow/configs/data_config.yaml`

### Core Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_samples` | 10,000 | Total dataset size for training |
| `sample_rate` | 4,096 Hz | LIGO standard detector sampling rate |
| `duration` | 4.0 s | Data segment length per sample |
| `detectors` | [H1, L1, V1] | Active interferometers (Hanford, Livingston, Virgo) |

### SNR Distribution

Targets realistic O4 detection rates:

```yaml
snr_ranges:
  weak:   [10.0, 15.0]   # Near detection threshold
  low:    [12.0, 22.0]   # Low but confident
  medium: [18.0, 35.0]   # Most typical events
  high:   [30.0, 50.0]   # Loud events
  loud:   [45.0, 65.0]   # Very loud/nearby

snr_distribution:
  weak:   0.05  # 5% near threshold
  low:    0.35  # 35% low SNR
  medium: 0.45  # 45% medium SNR (bulk)
  high:   0.12  # 12% high SNR
  loud:   0.03  # 3% very loud
```

### Event Type Distribution

```yaml
event_type_distribution:
  BBH:   0.46  # Black hole binaries (most common)
  BNS:   0.32  # Binary neutron stars (rare, important)
  NSBH:  0.17  # Neutron star + black hole
  noise: 0.05  # Pure noise samples
```

### Distance Ranges (O4/Design Sensitivity)

```yaml
distance_ranges:
  BBH:  [50.0, 2000.0]    # O4 horizon ~1000-1200 Mpc
  BNS:  [10.0, 180.0]     # O4 realistic ~150-170 Mpc
  NSBH: [20.0, 600.0]     # Intermediate scale
```

### Overlap Configuration

```yaml
overlap_fraction: 0.45         # 45% of samples contain overlaps
edge_case_fraction: 0.08       # 8% edge cases
extreme_cases:
  enabled: true
  fraction: 0.03              # Only 3% extreme samples
```

---

## Component Architecture

```
┌─────────────────────────────────────────────┐
│      GWDatasetGenerator (orchestrator)      │
├─────────────────────────────────────────────┤
│                                             │
├─→ ParameterSampler      (BBH, BNS, NSBH)   │
├─→ WaveformGenerator     (PyCBC + fallback)  │
├─→ NoiseGenerator        (colored noise)     │
├─→ PSDManager            (detector psds)     │
├─→ SignalInjector        (overlap mixing)    │
├─→ DataPreprocessor      (whitening, filter) │
├─→ OverlappingSignalSimulator (physics)     │
└─→ DatasetWriter         (HDF5/PKL output)   │
```

### Key Classes

#### 1. **ParameterSampler** (`parameter_sampler.py`)
Generates astrophysically realistic GW parameters.

**Methods:**
- `sample_bbh_parameters(snr_regime, is_edge_case)` → BBH parameter dict
- `sample_bns_parameters(snr_regime, is_edge_case)` → BNS parameter dict
- `sample_nsbh_parameters(snr_regime, is_edge_case)` → NSBH parameter dict
- `calibrate_snr_by_event_type(n_samples)` → Empirical SNR conditioning
- `event_type_given_snr(snr_regime)` → Bayes-inverted event type

#### 2. **WaveformGenerator** (`waveform_generator.py`)
Generates realistic gravitational waveforms.

**Methods:**
- `generate_waveform(params, detector_name)` → Time-domain strain (with fallback chain)
- `generate_pycbc_waveform(params, detector_name)` → PyCBC waveform
- `generate_analytical_waveform(params, detector_name)` → Post-Newtonian fallback
- `generate_aligned_spin_waveform()` → 3.5PN TaylorT4 aligned-spin
- `generate_tidal_waveform()` → Tidal corrections for BNS/NSBH
- `generate_precessing_waveform()` → Precessing BBH waveforms
- `compute_optimal_snr(signal, psd)` → Matched-filter SNR

#### 3. **SignalInjector** (`injection.py`)
Injects signals into noise with precise SNR control.

**Methods:**
- `inject_signal(noise, params, detector_name, psd_dict)` → Single signal injection
- `inject_overlapping_signals(noise, signal_params_list, detector_name, psd_dict)` → Multiple signals
- `create_overlapping_scenario(n_signals, snr_range, overlap_window)` → Parameter sets for overlaps
- `calculate_network_snr(detector_snrs)` → Combined SNR from detectors

**Helper Functions:**
- `attach_network_snr(d)` → Priority-based SNR attachment:
  1. Sampled `target_snr` (highest priority)
  2. Matched-filter SNRs from detector data
  3. Fallback proxy formula
- `proxy_network_snr_from_params(d)` → SNR ∝ (M_c)^(5/6) / distance

#### 4. **DataPreprocessor** (`preprocessing.py`)
Preprocessing for ML training.

**Methods:**
- `preprocess(strain, psd_dict, whiten, bandpass, remove_edges)` → Complete pipeline
- `whiten_data(strain, psd_dict)` → Frequency-domain whitening
- `bandpass_filter(strain, f_low, f_high)` → 8-pole Butterworth filter
- `highpass_filter(strain, cutoff)` → High-pass filter
- `apply_tukey_window(strain, alpha)` → Edge tapering
- `validate_data(strain)` → Quality checks

#### 5. **GWDatasetGenerator** (`dataset_generator.py`)
Main orchestrator for complete dataset generation.

**Key Methods:**
- `generate_dataset()` → Main generation pipeline
- `_generate_sample_with_simulator()` → Physics-based generation
- `_generate_sample_with_priority()` → Priority calculation
- `_generate_single_sample()` → Legacy generation
- `_classify_signal_type(mass_1, mass_2)` → BBH/BNS/NSBH classification
- `_compute_snr_from_strain(strain, detector_name)` → SNR from detector data

---

## Parameter Sampling

### Overview

Parameters are sampled from astrophysically realistic distributions, with special attention to SNR-distance and mass-distance correlations.

### BBH Parameter Sampling

```python
def sample_bbh_parameters(snr_regime=None, is_edge_case=False):
    # 1. Mass sampling
    m1_raw = lognormal(mean=ln(28), sigma=0.30) clipped to [8, 60]
    m2_raw = lognormal(mean=ln(22), sigma=0.32) clipped to [8, 60]
    enforce mass_ratio_min = 0.1
    
    # 2. Compute derived masses
    total_mass = m1 + m2
    chirp_mass = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
    mass_ratio = m2 / m1
    
    # 3. Sample target SNR from regime distribution
    target_snr = sample_from_snr_regime(snr_regime)
    
    # 4. Derive distance from SNR using chirp-mass scaling
    # CRITICAL: This creates the tight SNR-distance correlation
    luminosity_distance = reference_distance * (chirp_mass / reference_mass)^(5/6) * 
                         (reference_snr / target_snr)
    
    # 5. Clip to valid range [50, 2000] Mpc
    luminosity_distance = clip(luminosity_distance, d_min, d_max)
    
    # 6. Spin parameters (isotropic)
    a1 = beta(2, 5) clipped to [0, 0.99]
    a2 = beta(2, 5) clipped to [0, 0.99]
    tilt1, tilt2 = arccos(uniform(-1, 1))
    
    # 7. Sky location (isotropic)
    ra = uniform(0, 2π)
    dec = arcsin(uniform(-1, 1))
    theta_jn = arccos(uniform(-1, 1))  # Inclination
    psi = uniform(0, π)
    phase = uniform(0, 2π)
```

### BNS Parameter Sampling

```python
def sample_bns_parameters(snr_regime=None, is_edge_case=False):
    # 1. Tight mass range (~1.4 Msun neutron stars)
    m1 = normal(1.40, 0.15) clipped to [1.0, 2.5]
    m2 = normal(1.40, 0.20) clipped to [1.0, 2.5]
    
    # 2. Chirp mass (narrow distribution)
    chirp_mass = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
    
    # 3. SNR sampling
    target_snr = sample_from_snr_regime(snr_regime)
    
    # 4. Distance from SNR
    luminosity_distance = reference_distance * (chirp_mass / reference_mass)^(5/6) * 
                         (reference_snr / target_snr)
    luminosity_distance = clip(luminosity_distance, 10.0, 180.0)
    
    # 5. Tidal parameters
    lambda_1 = lognormal(ln(400), 0.7) * (1.4/m1)^5
    lambda_2 = lognormal(ln(400), 0.7) * (1.4/m2)^5
    lambda_tilde = (16/13) * [detailed formula]
    
    # 6. Low spins
    a1, a2 = uniform(0, 0.05)
    tilt1, tilt2 = 0
```

### NSBH Parameter Sampling

```python
def sample_nsbh_parameters(snr_regime=None, is_edge_case=False):
    # 1. Mass composition
    ns_mass = uniform(1.2, 2.0)
    
    # BH mass diversity (light/medium/heavy)
    if bh_mass_type == 'light':
        bh_mass = uniform(3.0, 8.0)
    elif bh_mass_type == 'medium':
        bh_mass = uniform(8.0, 25.0)
    else:  # heavy
        bh_mass = uniform(25.0, 50.0)
    
    # 2. Chirp mass
    chirp_mass = (bh_mass * ns_mass)^(3/5) / (bh_mass + ns_mass)^(1/5)
    
    # 3. MASS-AWARE SNR SAMPLING (crucial for NSBH)
    base_snr = sample_from_snr_regime(snr_regime)
    
    # Adjust target SNR based on BH mass to decouple mass from distance
    # Heavier BHs → larger chirp_mass → would naturally need larger distances
    # Boost SNR to keep distances uniform
    if bh_mass_type == 'light':
        target_snr = base_snr               # Baseline
    elif bh_mass_type == 'medium':
        target_snr = base_snr * 1.25       # +25% boost
    else:  # heavy
        target_snr = base_snr * 1.55       # +55% boost
    
    # 4. Distance from target SNR
    luminosity_distance = reference_distance * (chirp_mass / reference_mass)^(5/6) * 
                         (reference_snr / target_snr)
    luminosity_distance = clip(luminosity_distance, 20.0, 600.0)
    
    # 5. Tidal parameters (mass-dependent)
    if total_mass <= 6.0:
        approximant = 'IMRPhenomPv2_NRTidal'
        # Tidal EOS-dependent lambda_2
    else:
        approximant = 'IMRPhenomPv2'
        lambda_2 = 0.0
```

### Reference Parameters

**Critical calibration constants** (must be consistent across all modules):

```python
reference_snr = 35.0       # Baseline SNR
reference_mass = 30.0      # Reference chirp mass (Msun)
reference_distance = 400.0 # Reference distance (Mpc)
```

These define the SNR scaling formula:
```
SNR = reference_snr × (M_chirp / reference_mass)^(5/6) × (reference_distance / distance)
```

### Edge Case Sampling

Applied to ~8% of dataset, modifying parameters for robustness:

**BBH edge cases:**
- Short-duration (high mass): m1 ∈ [60, 100], m2 ∈ [50, m1]
- Extreme mass ratio: q ∈ [0.05, 0.15]

**BNS edge cases:**
- Long inspiral: f_lower ∈ [10, 15] Hz (standard: 35 Hz)

**NSBH edge cases:**
- Extreme BH masses: m_BH ∈ [50, 100] Msun

---

## Waveform Generation

### Generation Pipeline

```
generate_waveform(params, detector_name)
  ├─ Try: PyCBC (full physics, accurate)
  ├─ If fails: Analytical post-Newtonian
  │   ├─ Tidal corrections (BNS/NSBH)
  │   ├─ Precession (Pv2 approximants)
  │   ├─ Aligned-spin (3.5PN TaylorT4)
  ├─ If fails: Simple chirp (ultimate fallback)
  └─ Rescale to target SNR (if not PyCBC)
```

### Aligned-Spin Waveform (3.5PN TaylorT4)

**Key Formula:**
```
Frequency evolution:
  θ = time_to_merger / (5 M_chirp)
  v = θ^(-1/8)
  f = v^3 / (8π M_chirp)

Phase (3.5PN with spin):
  ψ = -(1/η) × [
    v^(-5) +
    (3715/1008 + 55η/12) v^(-3) +
    (-10π + (113/3 + 19η/3)χ_eff) v^(-2) +
    (higher order terms) v^(-1)
  ]

Amplitude (strain):
  A = (M_chirp^(5/6) / d_L) × f^(-7/6)
  A_scaled = A × 2×10^(-23) × (1 + cos²(θ_jn)) / 2

Strain:
  h(t) = A × sin(2π cumsum(f) / f_s + ψ)
```

**Calibration:** The factor `2×10^(-23)` is empirically tuned to match target SNR for 30 Msun at 400 Mpc.

### Tidal Waveform Corrections

For BNS and some NSBH systems:

```
Base waveform: h_base = generate_aligned_spin_waveform(...)

Tidal phase correction (5PN):
  x = (π M_chirp f)^(2/3)
  Δψ_tidal = -(39/2) λ̃ x^5 / M_chirp^5
  
where λ̃ is the effective tidal deformability:
  λ̃ = (16/13) × [(m1 + 12m2)m1^4 λ1 + (m2 + 12m1)m2^4 λ2] / M_total^5

Frequency domain application:
  h_tidal(f) = h_base(f) × exp(-i Δψ_tidal)
```

### Precessing Waveform Modulation

For systems with significant spin tilts (tilt1, tilt2 > 0.1):

```
Precession frequency:
  Ω_prec = (2 + 3q/2) / (1 + q) × Ω_orb
  
where q = m2/m1 is the mass ratio

Effective precessing spin:
  χ_p = max(
    a1 sin(tilt1),
    a2 × (4 + 3q) / (4 + 3/q) × sin(tilt2)
  )

Modulation amplitude:
  mod_amplitude = 0.4 × tanh(2 χ_p)

Applied modulation:
  h_prec(t) = h_base(t) × [1 + mod_amplitude cos(2π Ω_prec t + φ12)]
```

### SNR Scaling

When using fallback waveforms (not PyCBC):

```python
# Compute expected SNR from distance and mass
expected_snr = reference_snr × (chirp_mass / reference_mass)^(5/6) × 
               (reference_distance / distance)

# Scale factor
scale_factor = target_snr / expected_snr

# Rescaled waveform
h_scaled = h × scale_factor
```

---

## Signal Injection

### Single Signal Injection

```python
def inject_signal(noise, params, detector_name, psd_dict):
    # 1. Generate waveform
    signal = generate_waveform(params, detector_name)
    
    # 2. Resize to match noise length
    signal = resize_signal(signal, len(noise))
    
    # 3. Scale to target SNR
    target_snr = params.get('target_snr', 15.0)
    scaled_signal, actual_snr = scale_to_target_snr(signal, noise, target_snr, psd_dict)
    
    # 4. Combine with noise
    injected = noise + scaled_signal
    
    return injected, metadata
```

### Overlapping Signal Injection

For multiple signals with time clustering:

```python
def inject_overlapping_signals(noise, signal_params_list, detector_name, psd_dict):
    combined_signal = zeros(len(noise))
    
    for params in signal_params_list:
        # 1. Generate individual waveform
        signal = generate_waveform(params, detector_name)
        signal = resize_signal(signal, len(noise))
        
        # 2. Scale to target SNR
        scaled_signal, actual_snr = scale_to_target_snr(signal, noise, target_snr, psd_dict)
        
        # 3. Apply time offset for overlap
        if time_offset != 0:
            scaled_signal = apply_time_shift(scaled_signal, time_offset)
        
        # 4. Accumulate
        combined_signal += scaled_signal
    
    # 5. Combine with noise
    injected = noise + combined_signal
    
    return injected, metadata_list
```

### Network SNR Calculation

```python
def calculate_network_snr(detector_snrs: Dict[str, float]) -> float:
    """
    Network SNR combines individual detector SNRs (incoherent combination)
    Formula: ρ_net = √(ρ_H1² + ρ_L1² + ρ_V1²)
    """
    network_snr_sq = sum(snr**2 for snr in detector_snrs.values())
    return sqrt(network_snr_sq)
```

### SNR Attachment Priority

```python
def attach_network_snr(detection_dict):
    """
    Determine network SNR with priority order:
    
    1. HIGH: Use sampled target_snr if present
       - Preserves stochastic SNR sampling from parameter distribution
    
    2. MEDIUM: Compute from per-detector matched-filter SNRs
       - Uses actual detector response
    
    3. LOW: Use proxy formula based on mass and distance
    """
    
    # Priority 1: Sampled target_snr
    if 'target_snr' in d and d['target_snr'] is not None:
        d['network_snr'] = float(d['target_snr'])
        return
    
    # Priority 2: Per-detector SNRs
    snrs = []
    for key in ['snr_H1', 'snr_L1', 'snr_V1']:
        if key in d:
            snrs.append(float(d[key]))
    if snrs:
        d['network_snr'] = float(sqrt(sum(s**2 for s in snrs)))
        return
    
    # Priority 3: Proxy formula
    m1 = float(d.get('mass_1', 30.0))
    m2 = float(d.get('mass_2', 25.0))
    d_l = float(d.get('luminosity_distance', 100.0))
    
    M_chirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    snr = reference_snr * (M_chirp / reference_mass)**(5/6) * 
          (reference_distance / d_l)
    
    d['network_snr'] = float(snr)
```

---

## Preprocessing Pipeline

### Complete Preprocessing Sequence

```
Raw strain data
  ↓
1. DC offset removal
  ↓
2. Bandpass filter [f_low, f_high] (8-pole Butterworth)
  ↓
3. Frequency-domain whitening using detector PSD
  ↓
4. Tukey window taper (α=0.1)
  ↓
Preprocessed strain (float32)
```

### Whitening Formula

```
Whitening in frequency domain:

1. FFT: h̃(f) = FFT[h(t)] / N

2. Interpolate PSD to match FFT frequencies

3. Whiten: h̃_white(f) = h̃(f) / √(S_n(f) × f_s / 2)

4. IFFT: h_white(t) = IFFT[h̃_white(f)]

5. High-pass filter (10 Hz) to remove artifacts
```

### Bandpass Filter

8-pole Butterworth bandpass filter [f_low, f_high]:
- Normalized frequencies: [f_low/f_nyquist, f_high/f_nyquist]
- Applied forward-backward (filtfilt) for zero phase shift

### Tukey Window

Alpha parameter controls edge taper:
```
window = tukey(N, alpha=0.1)
strain_tapered = strain × window
```

---

## Dataset Generation Workflow

### Main Generation Loop

```python
def generate_dataset(n_samples=10000, overlap_fraction=0.45, ...):
    
    # 1. Initialize components
    psd_manager = PSDManager()
    psds = psd_manager.load_detector_psds(detectors)
    parameter_sampler = ParameterSampler()
    waveform_generator = WaveformGenerator()
    injector = SignalInjector()
    preprocessor = DataPreprocessor()
    
    samples = []
    
    # 2. Generate samples
    for sample_id in range(n_samples):
        
        # Decide: single or overlap
        is_overlap = random() < overlap_fraction
        
        if is_overlap:
            n_signals = sample_overlap_size(p_heavy=0.45, base_lambda=2.2)
            signal_params = []
            
            for i in range(n_signals):
                event_type = sample_event_type()
                
                if event_type == 'BBH':
                    params = parameter_sampler.sample_bbh_parameters()
                elif event_type == 'BNS':
                    params = parameter_sampler.sample_bns_parameters()
                else:
                    params = parameter_sampler.sample_nsbh_parameters()
                
                signal_params.append(params)
            
            # Mixed event types (≥30% of overlaps)
            signal_params = ensure_mixed_event_types(signal_params, p_mix=0.50)
            
            # Clustered times for overlap
            times = sample_clustered_times(n_signals, duration, overlap_window=0.6)
            for i, t in enumerate(times):
                signal_params[i]['geocent_time'] = t
            
            # Generate noise
            detector_data = {}
            for detector_name in detectors:
                psd_dict = psds[detector_name]
                noise = noise_generator.generate_colored_noise(psd_dict)
                
                # Inject overlapping signals
                injected, metadata = injector.inject_overlapping_signals(
                    noise, signal_params, detector_name, psd_dict
                )
                
                # Preprocess
                if preprocess:
                    injected = preprocessor.preprocess(injected, psd_dict)
                
                detector_data[detector_name] = {
                    'strain': injected.astype(float32),
                    'metadata': metadata
                }
            
            # Compute priorities (SNRs)
            priorities = []
            for params in signal_params:
                priority = compute_snr_from_params(params, detector_data)
                priorities.append(priority)
            
            priorities = rescale_priorities(priorities)
            
            sample = {
                'sample_id': f'overlap_{sample_id:06d}',
                'type': 'overlap',
                'is_overlap': True,
                'n_signals': n_signals,
                'parameters': signal_params,
                'priorities': priorities,
                'detector_data': detector_data,
                'edge_type_id': encode_edge_type(signal_params),
                'metadata': {...}
            }
        
        else:  # Single signal
            event_type = sample_event_type()
            params = sample_parameters(event_type)
            
            detector_data = {}
            for detector_name in detectors:
                psd_dict = psds[detector_name]
                noise = noise_generator.generate_colored_noise(psd_dict)
                
                signal = waveform_generator.generate_waveform(params, detector_name)
                injected, metadata = injector.inject_signal(
                    noise, params, detector_name, psd_dict
                )
                
                if preprocess:
                    injected = preprocessor.preprocess(injected, psd_dict)
                
                detector_data[detector_name] = {
                    'strain': injected.astype(float32),
                    'metadata': metadata
                }
            
            priority = compute_snr_from_params(params, detector_data)
            
            sample = {
                'sample_id': f'single_{sample_id:06d}',
                'type': params['type'],
                'is_overlap': False,
                'n_signals': 1,
                'parameters': [params],
                'priorities': [priority],
                'detector_data': detector_data,
                'metadata': {...}
            }
        
        # Track statistics and save
        track_sample(sample)
        samples.append(sample)
        
        if len(samples) % save_batch_size == 0:
            writer.save_batch(samples)
            samples = []
    
    return samples, statistics
```

### Overlap Size Sampling

```python
def sample_overlap_size(p_heavy=0.45, base_lambda=2.2, heavy_min=4, heavy_max=6):
    """
    Mixture distribution for number of overlapping signals:
    
    45% probability: Draw from heavy tail (4-6 signals)
    55% probability: Draw from light distribution (1-4 signals)
    
    This creates realistic multi-event scenarios while keeping bulk simple.
    """
    if random() < p_heavy:
        # Heavy tail: 4-6 signals
        return randint(heavy_min, heavy_max + 1)
    else:
        # Light: Poisson truncated to [1,4]
        val = poisson(base_lambda)
        return clip(val, 1, 4)
```

### Clustered Time Sampling

```python
def sample_clustered_times(rng, n_signals, duration, overlap_window=0.6):
    """
    Generate geocent_time offsets clustered within overlap_window.
    Ensures signals overlap temporally.
    """
    # Random center time (not at edges)
    center = uniform(-0.5 * (duration - overlap_window), 
                    0.5 * (duration - overlap_window))
    
    # Gaussian cluster around center
    offsets = normal(0, overlap_window / 6.0, size=n_signals)
    times = center + offsets
    
    # Clip to valid range
    return clip(times, -duration/2 + 0.01, duration/2 - 0.01)
```

### Mixed Event Type Enforcement

```python
def ensure_mixed_event_types(signal_params_list, p_mix=0.50):
    """
    With probability p_mix, force signals beyond first to be different types
    from the first signal. Creates cross-type overlaps (BBH+BNS, etc.)
    """
    if len(signal_params_list) <= 1:
        return signal_params_list
    
    if random() >= p_mix:
        return signal_params_list
    
    first_type = signal_params_list[0].get('type', 'BBH')
    
    for i in range(1, len(signal_params_list)):
        current_type = signal_params_list[i].get('type', 'BBH')
        
        if current_type == first_type:
            available_types = [t for t in ['BBH', 'BNS', 'NSBH'] if t != first_type]
            new_type = choice(available_types)
            signal_params_list[i]['type'] = new_type
    
    return signal_params_list
```

### Priority Rescaling

```python
def rescale_priorities(y_raw):
    """
    Rescale priority list from [min, max] to [0, 1] with mild gamma < 1.
    Expands headroom for calibration (avoids compression at extremes).
    """
    y_arr = array(y_raw, dtype=float32)
    y_min, y_max = min(y_arr), max(y_arr)
    
    if y_max - y_min < 1e-6:
        # Constant: return 0.5
        return [0.5] * len(y_arr)
    
    # Normalize to [0, 1]
    y = (y_arr - y_min) / (y_max - y_min)
    
    # Apply gamma expansion
    y = clip(y ** 0.9, 0.0, 1.0)
    
    return y.tolist()
```

### Edge Type Encoding

```python
def encode_edge_type(signal_params_list):
    """
    Map overlap size to stable integer ID for edge conditioning in PriorityNet.
    
    0 → Single signal
    3 → Pairwise overlap (2 signals)
    6 → Triple overlap (3 signals)
    7+ → Heavy overlaps (4+ signals)
    """
    n = len([p for p in signal_params_list if p is not None])
    
    if n == 1:
        return 0
    elif n == 2:
        return 3
    elif n == 3:
        return 6
    else:
        return 7
```

---

## Key Formulas and Physics

### SNR-Distance Scaling

**Fundamental relationship:**
```
SNR ∝ √(M_chirp) / distance

Or with full 3.5PN:
SNR = reference_snr × (M_chirp / reference_mass)^(5/6) × 
      (reference_distance / distance)
```

**Reference parameters (CRITICAL - must match across all modules):**
- reference_snr = 35.0
- reference_mass = 30.0 Msun
- reference_distance = 400.0 Mpc

### Chirp Mass

```
M_chirp = (m1 × m2)^(3/5) / (m1 + m2)^(1/5)

Properties:
- Dominates waveform frequency evolution
- Central to SNR calculation
- Tightly constrained by GW observations
```

### Effective Spin

```
χ_eff = (m1 × a1 × cos(tilt1) + m2 × a2 × cos(tilt2)) / (m1 + m2)

- Affects phase evolution at 3.5PN
- Bounded to [-1, 1] for physical systems
- Tilt angles ∈ [0, π]
```

### Optimal SNR Computation

**Matched-filter SNR formula:**
```
ρ² = 4 × Δf × Σ_f |h̃(f)|² / S_n(f)

Where:
- h̃(f) = FFT of waveform / N
- S_n(f) = Power spectral density of detector noise
- Δf = Frequency resolution
- The factor 4 accounts for positive/negative frequencies and PSD convention
```

**Implementation:**
```python
def compute_optimal_snr(signal, psd_dict):
    N = len(signal)
    signal_fft = fft(signal) / float(N)
    freq_array = rfftfreq(N, 1/sample_rate)
    
    psd_interp = interp(freq_array, psd_freqs, psd)
    psd_interp = max(psd_interp, 1e-50)  # Avoid division by zero
    
    df = freq_array[1] - freq_array[0]
    integrand = abs(signal_fft)**2 / psd_interp
    
    snr_squared = 4.0 * sum(integrand) * df
    return sqrt(max(snr_squared, 0))
```

### Antenna Response

**Detector response to GW polarizations:**
```
h(t) = F_+ × h_+(t) + F_× × h_×(t)

Where F_+ and F_× are antenna response factors that depend on:
- Source location: RA, Dec
- GW polarization angle: ψ
- Detector orientation: latitude, longitude, rotation

For isotropic source:
<|h|²> ≈ 0.4  (averaged over sky)
```

### Redshift and Cosmological Distance

```
Redshift from luminosity distance:
z = √(1 + 2Ω_m × d_L / (c² H_0⁻²)) - 1

Comoving distance:
d_C = d_L / (1 + z)

Effective source-frame masses:
m_source = m_detector / (1 + z)
```

---

## Data Structure

### Sample Dictionary

```python
{
    'sample_id': str,                      # Unique identifier
    'type': str,                           # 'BBH', 'BNS', 'NSBH', 'overlap', 'noise'
    'is_overlap': bool,                    # True if multiple signals
    'n_signals': int,                      # Number of signals (1-8)
    'parameters': list | dict,             # Signal parameters (single or list)
    'priorities': list,                    # SNR-based priorities [0, 1]
    'detector_data': {
        'H1': {
            'strain': ndarray[float32],    # Whitened, preprocessed strain
            'metadata': dict                # Injection metadata
        },
        'L1': {...},
        'V1': {...}
    },
    'edge_type_id': int,                   # Overlap size encoding [0, 3, 6, 7]
    'is_edge_case': bool,                  # True if edge case sample
    'metadata': {
        'sample_id': int,
        'n_signals': int,
        'mean_snr': float,
        'max_snr': float,
        'generator': str,
        'scenario_type': str,
        ...
    }
}
```

### Parameter Dictionary (Single Signal)

```python
{
    'type': str,                           # 'BBH', 'BNS', 'NSBH'
    'mass_1': float,                       # Primary mass (Msun)
    'mass_2': float,                       # Secondary mass (Msun)
    'total_mass': float,                   # m1 + m2
    'chirp_mass': float,                   # (m1 m2)^(3/5) / (m1+m2)^(1/5)
    'mass_ratio': float,                   # m2/m1
    'symmetric_mass_ratio': float,         # (m1 m2) / (m1+m2)²
    
    # Spins
    'a1': float,                           # Primary spin magnitude [0, 0.99]
    'a2': float,                           # Secondary spin magnitude
    'tilt1': float,                        # Primary spin tilt angle [0, π]
    'tilt2': float,                        # Secondary spin tilt angle
    'effective_spin': float,               # χ_eff
    'phi12': float,                        # Relative azimuth
    'phi_jl': float,                       # Precession phase
    
    # Distance and redshift
    'luminosity_distance': float,          # Mpc
    'redshift': float,                     # Cosmological redshift
    'comoving_distance': float,            # Mpc
    
    # Sky location
    'ra': float,                           # Right ascension [0, 2π]
    'dec': float,                          # Declination [-π/2, π/2]
    'theta_jn': float,                     # Inclination (observer frame) [0, π]
    'psi': float,                          # Polarization angle [0, π]
    
    # Merger
    'geocent_time': float,                 # Time of merger (relative)
    'phase': float,                        # Coalescence phase [0, 2π]
    
    # Waveform model
    'approximant': str,                    # e.g., 'IMRPhenomD'
    'f_lower': float,                      # Hz
    'f_ref': float,                        # Reference frequency (Hz)
    'f_upper': float,                      # Upper frequency cutoff (Hz)
    
    # Tidal (BNS/NSBH)
    'lambda_1': float,                     # Tidal deformability body 1
    'lambda_2': float,                     # Tidal deformability body 2
    'lambda_tilde': float,                 # Effective tidal deformability
    
    # SNR and metadata
    'target_snr': float,                   # Sampled target SNR [5, 100]
    'network_snr': float,                  # Computed network SNR
    'is_real_event': bool,                 # True if from GWTC
    'edge_case': bool,                     # True if edge case sample
    'edge_case_type': str,                 # Type of edge case
    
    # Additional (NSBH specific)
    'bh_mass_type': str,                   # 'light', 'medium', 'heavy'
    'eos_type': str,                       # EOS for BNS: 'soft', 'medium', 'stiff'
    'approximant_type': str,               # 'tidal', 'non_precessing', 'precessing'
}
```

---

## Configuration Reference

### Complete YAML Structure

```yaml
# Dataset generation
n_samples: 10000
output_dir: "data/dataset"
output_format: "pkl"
save_batch_size: 100

# Detector configuration
sample_rate: 4096
duration: 4.0
detectors:
  - H1
  - L1
  - V1

# Overlap configuration
overlap_fraction: 0.45
edge_case_fraction: 0.08

# SNR distribution
snr_ranges:
  weak: [10.0, 15.0]
  low: [12.0, 22.0]
  medium: [18.0, 35.0]
  high: [30.0, 50.0]
  loud: [45.0, 65.0]

snr_distribution:
  weak: 0.05
  low: 0.35
  medium: 0.45
  high: 0.12
  loud: 0.03

# Event type distribution
event_type_distribution:
  BBH: 0.46
  BNS: 0.32
  NSBH: 0.17
  noise: 0.05

# Mass ranges
mass_ranges:
  BBH: [5.0, 100.0]
  BNS: [1.0, 2.5]
  NSBH: [1.0, 100.0]

# Distance ranges
distance_ranges:
  BBH: [50.0, 2000.0]
  BNS: [10.0, 180.0]
  NSBH: [20.0, 600.0]

# Waveform model
approximant: "IMRPhenomXAS"
f_ref: 20.0
f_lower: 20.0
f_upper: 1024.0

# Pre-merger events
premerger_config:
  enabled: true
  fraction: 0.08
  time_to_merger_range: [0.5, 3.0]
  event_types:
    - BNS
    - NSBH
  min_snr: 8

# Edge cases
edge_cases:
  physical_extremes:
    enabled: true
    fraction: 0.25
    types:
      high_mass_ratio:
        fraction: 0.25
        q_range: [0.05, 0.15]
      # ... more types ...

# Extreme cases
extreme_cases:
  enabled: true
  fraction: 0.03
  types:
    near_simultaneous_mergers:
      enabled: true
      fraction: 0.25
      delta_t_range: [0.1, 0.3]
    # ... more types ...

# Processing
add_glitches: true
preprocess: true
noise_augmentation_k: 1
save_complete: false

# Splits
create_splits: true
train_frac: 0.80
val_frac: 0.10
test_frac: 0.10
chunk_size: 500

# Random seed
random_seed: 42

# Debug
debug_snr_diagnostic: false
debug_snr_limit: 50

# Validation
validation:
  inclination_isotropy_pvalue_min: 0.05
  mass_distance_correlation_max:
    BBH: 0.55
    BNS: 0.35
    NSBH: 0.45
  min_samples_per_type:
    BBH: 100
    BNS: 50
    NSBH: 50
```

---

## Entry Points

### Command-Line Generation

```bash
# Activate environment
conda activate ahsd

# Generate dataset
ahsd-generate --config configs/data_config.yaml

# Train model
ahsd-train --config configs/training_config.yaml

# Validate
ahsd-validate --dataset data/dataset/train_split.pkl
```

### Programmatic Usage

```python
from ahsd.data.dataset_generator import GWDatasetGenerator
from ahsd.data.config import load_config

config = load_config('configs/data_config.yaml')

generator = GWDatasetGenerator(
    output_dir='data/dataset',
    sample_rate=4096,
    duration=4.0,
    config=config
)

dataset = generator.generate_dataset(
    n_samples=10000,
    overlap_fraction=0.45,
    edge_case_fraction=0.08,
    preprocess=True
)
```

---

## Testing & Validation

### Quick Dataset Check

```python
import pickle

# Load sample
with open('data/dataset/train_split.pkl', 'rb') as f:
    samples = pickle.load(f)

# Inspect
sample = samples[0]
print(f"Type: {sample['type']}")
print(f"SNR: {sample.get('priorities', [])}")
print(f"Detectors: {list(sample['detector_data'].keys())}")
print(f"Strain shape: {sample['detector_data']['H1']['strain'].shape}")
print(f"Strain dtype: {sample['detector_data']['H1']['strain'].dtype}")
```

### Validation Checks

```python
def validate_dataset(samples):
    """Basic dataset validation"""
    
    for i, sample in enumerate(samples):
        # Check structure
        assert 'type' in sample
        assert 'parameters' in sample
        assert 'detector_data' in sample
        assert 'priorities' in sample
        
        # Check parameters
        for params in sample.get('parameters', []):
            assert 'mass_1' in params
            assert 'mass_2' in params
            assert params['mass_1'] >= params['mass_2']
            assert 'target_snr' in params or 'luminosity_distance' in params
        
        # Check detector data
        for det_name in ['H1', 'L1', 'V1']:
            strain = sample['detector_data'][det_name]['strain']
            assert len(strain) == 4096 * 4
            assert strain.dtype == np.float32
            assert np.all(np.isfinite(strain))
        
        # Check priorities
        assert len(sample['priorities']) == len(sample['parameters'])
        assert all(0 <= p <= 1 for p in sample['priorities'])
    
    print(f"✓ All {len(samples)} samples validated")
```

---

## Performance Metrics

### Typical Generation Speed

- **Single sample generation**: ~1-2 seconds (with PyCBC)
- **Full 10k dataset**: ~3-4 hours on single CPU
- **Memory per sample**: ~50-100 MB (detector data + metadata)

### Recommended Settings for Different Scales

| Scale | n_samples | duration | sample_rate | RAM Required |
|-------|-----------|----------|-------------|-------------|
| Debug | 100 | 4.0 | 4096 | 5 GB |
| Medium | 5,000 | 4.0 | 4096 | 250 GB |
| Full | 50,000 | 4.0 | 4096 | 2.5 TB |
| Production | 100,000+ | 4.0 | 4096 | >5 TB |

---

## References and Further Reading

- PyCBC Documentation: https://pycbc.org/
- IMRPhenomD: Husa et al., PRD 93, 044006 (2016)
- GW Data Analysis: Allen et al., "GW100" review
- LIGO/Virgo O4 Configuration: https://observing.ligo.org/

---

*Last Updated: November 2025*
*Version: 2.0 - Physics-Validated*
