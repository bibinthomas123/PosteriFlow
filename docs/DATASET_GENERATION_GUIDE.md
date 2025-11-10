# PosteriFlow Dataset Generation Guide
## A Narrative Explanation of How We Create Training Data for Gravitational Wave Detection

---

## The Big Picture: What Are We Trying to Do?

Imagine you're building a system to detect and analyze gravitational wave signals from colliding neutron stars or black holes. You need lots of training data, but you can't just use real observations because:

1. **Real events are rare** - LIGO might detect a few dozen events per year
2. **We need variety** - To train a robust model, we need signals with many different masses, distances, overlapping patterns, and noise conditions
3. **We need labels** - We know exactly what signals we injected (masses, distances, SNRs), so we can train models to predict these parameters
4. **We need realistic complexity** - Multiple signals overlapping, detector glitches, varying noise levels

So we **generate synthetic data** by:
1. Creating realistic gravitational wave signals
2. Mixing multiple signals together to create overlapping events
3. Injecting them into realistic detector noise
4. Processing the data to match what a detector sees
5. Packaging it all with the ground truth labels

---

## What Data Are We Creating?

Each sample in our dataset contains:

### The Signal Part
- **Gravitational wave strain**: The actual detector output (a time series of numbers)
- **Three separate detector views**: H1 (Hanford), L1 (Livingston), V1 (Virgo) - each detector sees a slightly different version of the signal based on its orientation
- **Preprocessed and whitened**: We apply standard processing so the model sees data like a real detector would deliver

### The Label Part
- **Source parameters**: What caused the signal?
  - Mass of object 1 and object 2
  - How far away was it?
  - How fast was it spinning?
  - What angle were we observing from?
  
- **Priority ranking**: How "loud" was each signal relative to others?
  - SNR (signal-to-noise ratio) tells us how obvious the signal is
  - If two signals overlap, which one is more important to detect first?

### The Context Part
- **Sample ID**: A unique identifier
- **Type of signal**: BBH (black hole merger), BNS (neutron star merger), or NSBH (mixed)
- **Metadata**: Information about how the sample was generated

---

## The Generation Pipeline: Step by Step

Think of the pipeline as an assembly line. Each stage transforms the data:

```
Stage 1: Sample Parameters
  ↓
  Choose: Will this be a BBH, BNS, or NSBH signal?
  Sample: Random masses, spins, distances (from realistic distributions)
  Result: A "recipe" for a gravitational wave
  
Stage 2: Generate Waveform
  ↓
  Take the recipe from Stage 1
  Calculate: What does this signal look like?
  Use: Physics-based formulas or PyCBC (the gold standard)
  Result: A synthetic gravitational wave signal
  
Stage 3: Create Overlap (if needed)
  ↓
  Sometimes we want 2, 3, or even 4+ signals at the same time
  We mix them with time offsets so they overlap
  Result: Multiple waveforms combined
  
Stage 4: Generate Noise
  ↓
  Create realistic detector noise
  Match: The noise profile of real LIGO/Virgo detectors
  Result: Realistic background noise
  
Stage 5: Inject Signals into Noise
  ↓
  Add the signal(s) to the noise
  Scale: Make signals have specific SNRs (strength levels)
  Result: Detector output with signal + noise
  
Stage 6: Preprocess
  ↓
  Whiten: Remove the color (frequency structure) of noise
  Filter: Keep only frequencies where GW signals live (30-500 Hz)
  Taper: Smooth the edges to reduce artifacts
  Result: Clean, preprocessed detector data ready for ML
  
Stage 7: Package
  ↓
  Organize: All detector data + parameters + metadata
  Save: Write to disk in efficient format
  Result: One sample in our training dataset
```

---

## Understanding the Key Decisions

### Decision 1: How Many Signals in This Sample?

```
Random draw:
  45% of the time: This sample will have MULTIPLE overlapping signals
  55% of the time: This sample will have ONE signal
  
When multiple signals:
  - How many? Could be 2, 3, 4, or even more
  - We use a "mixture" distribution that prefers 2-3 but sometimes does 4+
  - This matches realistic scenarios where detectors might catch multiple events

Why overlapping signals matter:
  - Real detectors will sometimes see multiple events at once
  - Our model needs to learn to: detect them, separate them, rank them
  - Overlapping signals are harder to analyze → good test for the model
```

### Decision 2: What Types of Events?

```
When we sample a signal type:
  46% Black Hole Binaries (BBH)
    - Heaviest signals, last only seconds, strong
    - Masses: 5-100 solar masses
    
  32% Binary Neutron Stars (BNS)
    - Lighter, last longer (~minutes of observable signal), rare
    - Masses: 1.0-2.5 solar masses each
    - Have tidal effects (they deform each other)
    
  17% Neutron Star + Black Hole (NSBH)
    - Mixed: one light (NS), one heavy (BH)
    - Masses: NS (1-2.5), BH (3-100)
    - Intermediate behavior
    
  5% Pure Noise
    - No signal at all, just detector background

Why mix types in overlaps:
  - Sometimes a BBH + BNS might overlap (realistic scenario)
  - Model needs to recognize them separately and rank them
  - We force ~30% of overlaps to have mixed types for learning diversity
```

### Decision 3: How Strong Should the Signals Be?

```
We don't just randomly generate signals. We sample from realistic SNR (strength) distributions:

5%   Very weak (SNR: 10-15)        - Just detectable, barely above noise
35%  Weak (SNR: 12-22)             - Clear but not loud
45%  Medium (SNR: 18-35)           - Typical for real detections (most common)
12%  Loud (SNR: 30-50)             - Really obvious signals
3%   Very loud (SNR: 45-65)        - Nearby or massive events

Why this distribution?
  - Matches actual LIGO/Virgo detection rates
  - Most signals are medium strength (far away or light objects)
  - Need some very weak ones to teach model detection threshold
  - Need some loud ones to teach model saturation behavior
```

---

## The Physics Behind the Numbers

### The Golden Relationship: SNR vs Distance

The most important formula in our system:

```
SNR = K × (M_chirp)^(5/6) / distance

What this means in words:
  - Heavier signals are stronger (mass to the 5/6 power)
  - Closer signals are stronger (inversely with distance)
  - These two things are tightly linked
  
Example:
  - A 30 solar mass merger at 400 Mpc gives SNR ~35
  - Same merger at 200 Mpc (2x closer) gives SNR ~55 (1.6x stronger)
  - Same merger at 800 Mpc (2x farther) gives SNR ~22 (1.6x weaker)

How we use this:
  1. We decide on a target SNR (from the distribution)
  2. We use this formula BACKWARDS to calculate: "What distance gives this SNR?"
  3. This ensures SNR and distance are always physically consistent
  4. Model learns the real physics automatically
```

### Chirp Mass: The Magic Number

```
What is chirp mass?
  M_c = (m1 × m2)^(3/5) / (m1 + m2)^(1/5)
  
Why it's magic:
  - Determines how fast the signal sweeps up in frequency
  - Controls the signal amplitude
  - The PRIMARY thing that determines signal strength
  - It's what we can best measure from gravitational waves
  
Example:
  - 30 + 30 Msun system → M_c ≈ 26 Msun
  - 20 + 40 Msun system → M_c ≈ 28 Msun (similar!)
  - 10 + 10 Msun system → M_c ≈ 8.9 Msun (much lighter → weaker signal)
```

### How Distance Gets Derived from SNR

```
Here's what actually happens when we generate a signal:

Step 1: Choose masses (randomly from physics distributions)
        m1 = 30 Msun, m2 = 25 Msun
        
Step 2: Calculate chirp mass
        M_c = (30 × 25)^(3/5) / 55^(1/5) ≈ 26 Msun

Step 3: Sample a target SNR
        SNR_target = 25 (from the weak/medium distribution)

Step 4: Calculate distance that gives this SNR
        d = reference_distance × (M_c / reference_mass)^(5/6) × 
            (reference_snr / SNR_target)
        d = 400 × (26/30)^(5/6) × (35/25)
        d ≈ 550 Mpc

Step 5: Generate signal with these parameters
        The signal gets created with all parameters consistent
        
Why this matters:
  - SNR and distance are NOT independent
  - They're linked by physics
  - Model learns to respect this relationship
```

---

## The Waveform Generation Process

### Where Do Signals Come From?

We have a fallback chain of methods to generate signals:

```
Try Method 1: PyCBC (Best - actual physics)
  ├─ Uses Einstein's equations to solve for the signal
  ├─ Accounts for: spins, precession, tidal effects, orbital eccentricity
  ├─ Highly accurate but slower
  └─ If successful: USE THIS

If PyCBC fails:
  Try Method 2: Analytical Post-Newtonian (Good - approximate physics)
  ├─ Uses mathematical approximations of GW equations
  ├─ Handles: tidal effects, precession, aligned spins
  ├─ Faster, usually accurate enough
  └─ If successful: USE THIS
  
If that fails:
  Try Method 3: Simple Chirp (Fallback - basic physics)
  ├─ Very simple frequency sweep
  ├─ Only gets amplitude and frequency envelope right
  └─ Always works, but less accurate

The idea:
  - Always try the most accurate method first
  - Fall back to simpler methods if needed
  - Every sample gets a signal, one way or another
```

### What Does the Waveform Look Like?

```
Example: BBH signal near merger

Time before merger (seconds) | What's happening
──────────────────────────────────────────────────
4.0 seconds before          | Frequency ~30 Hz, very weak
                            | Hard to see in detector noise
                            
2.0 seconds before          | Frequency ~100 Hz, getting stronger
                            | Signal becoming visible
                            
0.5 seconds before          | Frequency ~300 Hz, very loud
                            | Signal dominates the output
                            
0 (merger)                  | Frequency peaks, abrupt cutoff
                            | Peak amplitude reached
                            
After merger                | Ringdown - dampened oscillations
                            | Final black hole settles down

The key feature:
  - Frequency INCREASES with time (chirp)
  - Amplitude INCREASES with time
  - These two together make it look like a "chirping bird"
```

### Overlapping Signals Example

```
Scenario: Two BBH signals overlap

Signal 1: Mass ratio m1/m2 = 2, merges at t=0.5 sec
Signal 2: Mass ratio m1/m2 = 1, merges at t=1.0 sec

What we do:
  Generate signal 1: Complete waveform from start to merger
  Generate signal 2: Complete waveform from start to merger
  Shift signal 2 in time by +0.5 seconds
  Add them together: h_total = h1 + h2
  
Result: The detector sees both signals
  - First half: Only signal 1 visible
  - Middle: Both signals present (interference)
  - Second half: Only signal 2 visible
  
Model must learn to:
  - Recognize signal 1 in isolation
  - Recognize signal 2 in isolation
  - Separate them when overlapping
  - Rank them (which one is stronger?)
```

---

## Processing: From Raw Signal to ML-Ready Data

### Why Preprocessing Matters

```
Raw detector output has problems:
  - Low frequencies dominated by thermal noise, instrument drift (1/f noise)
  - High frequencies are too noisy to use
  - Detector has "color" - some frequencies louder than others
  
We solve this with three steps:

Step 1: WHITEN (remove color from noise)
  Problem: Noise is not white (equal at all frequencies)
           Detector noise is much louder at low frequencies
  Solution: Measure noise power at each frequency (Power Spectral Density)
            Divide by the noise at each frequency
            Normalize to equal power across spectrum
  Result: All frequencies now equally important
  
Step 2: BANDPASS (keep only useful frequencies)
  Problem: GW signals are in 30-500 Hz range
           Below 30 Hz: instrument noise dominates
           Above 500 Hz: too noisy, no GW signal
  Solution: Apply Butterworth filter [30 Hz, 500 Hz]
  Result: Keep signal, reject noise outside band
  
Step 3: TAPER (smooth edges)
  Problem: Sharp frequency cutoffs cause artifacts
  Solution: Multiply by Tukey window (smooth rise at edges)
  Result: Clean data without filter ringing
```

### Example: Before and After

```
Original raw data (4 seconds):
  Time series with lots of noise
  Signal buried in noise
  Hard to see anything
  
After whitening:
  Noise level more uniform across time
  Signal starting to become visible
  Background more consistent
  
After bandpass filter:
  Frequencies outside [30, 500] Hz removed
  Signal much more visible
  Noise still present but filtered
  
After tapering:
  Edges smoothed
  No artificial spikes from filter edges
  Ready for neural network
  
Final result: Clean, preprocessed strain data
  Signal is visible
  Noise is realistic
  Model can learn to detect and measure the signal
```

---

## Data Organization: What Does One Sample Look Like?

### The Sample Dictionary

```
One sample (dictionary) contains:

Basic Info:
  sample_id: "overlap_001234"          → Unique identifier
  type: "overlap"                      → This is a multi-signal sample
  is_overlap: true                     → Contains multiple signals
  n_signals: 3                         → Specifically 3 signals
  
The Signals:
  parameters: [
    {
      type: "BBH"                      → Black hole binary
      mass_1: 35.2 Msun
      mass_2: 28.5 Msun
      chirp_mass: 30.1 Msun
      target_snr: 22.5                 → How strong this signal is
      luminosity_distance: 450 Mpc     → How far away it is
      a1: 0.7, a2: 0.3                 → Spin magnitudes
      ra: 2.45, dec: 0.82              → Where in the sky
      geocent_time: 0.5                → When it merges (relative to sample)
      ... (many more parameters)
    },
    {
      type: "BNS"                      → Neutron star binary
      mass_1: 1.45 Msun
      mass_2: 1.38 Msun
      target_snr: 18.3
      luminosity_distance: 150 Mpc
      ... (similar parameters)
    },
    {
      type: "NSBH"                     → Mixed system
      mass_1: 20.0 Msun (black hole)
      mass_2: 1.5 Msun (neutron star)
      target_snr: 15.7
      ... (similar parameters)
    }
  ]
  
Priorities (Labels for ML):
  priorities: [0.78, 0.62, 0.44]       → Normalized SNR for each signal
                                        → Signal 1 is strongest
                                        → Model must learn to rank them
  
Detector Data (The actual observations):
  detector_data: {
    "H1": {                            → Hanford detector
      strain: [array of 16384 numbers] → 4 seconds × 4096 Hz sampling
              each number is the detector output
              float32 data type
              
      metadata: {
        target_snr: 22.5
        actual_snr: 21.8
        detector: "H1"
        injection_time: 0.5
      }
    },
    "L1": {                            → Livingston detector
      strain: [array of 16384 numbers] → Same signal, slightly different
              based on detector orientation
      metadata: {...}
    },
    "V1": {                            → Virgo detector (Italy)
      strain: [array of 16384 numbers]
      metadata: {...}
    }
  }
  
Edge Type Information:
  edge_type_id: 6                      → Triple overlap (3 signals)
                                        → Used for graph conditioning
  
Quality Flags:
  is_edge_case: false                  → Not a physically extreme case
  
Metadata:
  metadata: {
    mean_snr: 18.8                     → Average SNR across signals
    max_snr: 22.5                      → Strongest signal
    generator: "simulator"             → How it was generated
    scenario_type: "mixed_overlap"     → Type of scenario
  }
```

---

## How We Ensure Physical Realism

### The Mass-Distance Correlation Issue

```
Problem we faced:
  If we independently sampled masses and distances,
  we'd get unphysical combinations
  
Example of unphysical:
  m1=80 Msun, m2=70 Msun, distance=10000 Mpc
  → This would be a distant merger of very heavy objects
  → In real universe: we mostly detect nearby heavy OR distant light
  
Solution we implemented:
  1. Sample mass distribution (independent)
  2. Sample SNR distribution (independent)
  3. Calculate distance from SNR using physics formula
     → This creates realistic correlation
  
Result:
  - Heavy systems (large M_c) naturally end up closer to average distance
  - Light systems naturally end up farther
  - Correlations match real LIGO/Virgo observations
```

### The Mixed Event Types

```
Why it matters:
  Real detectors sometimes see a BBH and BNS at the same time
  Models must learn to: recognize both, separate them, rank them
  
Old approach:
  Most overlaps were same-type (BBH+BBH, BNS+BNS)
  Easier to generate, but unrealistic
  
New approach:
  ≥30% of overlaps mix event types:
    - BBH + BNS (e.g., SNR: 25 + 18)
    - BBH + NSBH (e.g., SNR: 30 + 12)
    - BNS + NSBH (e.g., SNR: 20 + 15)
  
Implementation:
  We have a function: ensure_mixed_event_types()
  It randomly converts 50% of overlapping signals to different types
  Result: Diverse, realistic training scenarios
```

### Edge Cases: Teaching the Model Robustness

```
~8% of samples are "edge cases" - physically or observationally extreme:

Physical Extremes (3%):
  - Extreme mass ratios: q < 0.15 (very unequal mass pairs)
  - High spins: a > 0.8 (rapidly rotating objects)
  - Eccentric mergers: small orbital eccentricity
  - Precessing systems: significant spin tilts
  
Why include them:
  - Model must handle rare but real scenarios
  - Teaches robustness to parameter space boundaries
  
Observational Extremes (3%):
  - Strong detector glitches overlapping signals
  - Non-stationary noise (PSD changes over time)
  - Detector dropout (one detector is offline)
  - Sky position effects (signal strength varies with location)
  
Why include them:
  - Real detectors face these challenges
  - Model learns to work in imperfect conditions
  
Statistical Extremes (2%):
  - Multimodal posteriors (multiple solutions possible)
  - Heavy-tailed distributions
  - Weak priors (uninformative)
  
Why include them:
  - Model learns about solution degeneracies
  - Teaches uncertainty quantification
```

---

## The Complete Dataset Size

```
For a full training set of 10,000 samples:

Composition:
  9,000 synthetic signals (90%)
  1,000 real GWTC events (10%)
  
Synthetic breakdown:
  4,500 single signals
    ├─ 2,070 BBH (46%)
    ├─ 1,440 BNS (32%)
    └─  990 NSBH (17%)
    
  4,500 overlapping signals
    ├─ 1,000 edge cases
    └─ 3,500 normal cases
    
  Pure noise samples: 500 (5%)

Event Type Distribution:
  BBH: 4,600 samples (largest, most common in universe)
  BNS: 3,200 samples (rarer, but important for physics)
  NSBH: 1,700 samples (intermediate)
  Noise: 500 samples (baseline)

SNR Distribution:
  Weak (SNR 10-15): 500 (5%)
  Low (SNR 12-22): 3,500 (35%)
  Medium (SNR 18-35): 4,500 (45%)
  High (SNR 30-50): 1,200 (12%)
  Loud (SNR 45-65): 300 (3%)

Distance Coverage:
  BBH: 50-2000 Mpc (universe scale)
  BNS: 10-180 Mpc (local universe)
  NSBH: 20-600 Mpc (intermediate)

Train/Val/Test Split:
  Training: 8,000 (80%) - used to optimize model
  Validation: 1,000 (10%) - used to tune hyperparameters
  Testing: 1,000 (10%) - held out for final evaluation
```

---

## How We Use This Data in Training

```
Loading the Dataset:
  1. Load 10,000 samples from disk (HDF5 or pickle format)
  2. For each sample:
     - Extract strain data from H1, L1, V1
     - Stack into tensor: shape [3, 16384] (3 detectors × 4096 Hz × 4 sec)
     - Extract parameters: mass, distance, SNR, etc.
     - Extract priorities: ranking (0-1 scale)
  
Training Loop:
  For each epoch (full pass through dataset):
    1. Shuffle the 8,000 training samples
    2. Create batches of 32 or 64 samples
    3. For each batch:
       - Feed strain data to neural network
       - Network predicts: masses, distances, SNR, ranking
       - Compare predictions to ground truth (from parameters)
       - Calculate loss (how wrong were we?)
       - Update model weights to reduce loss
  
Validation:
  After each epoch:
    1. Run model on 1,000 validation samples
    2. Calculate metrics: accuracy, correlation, ranking loss
    3. If validation improves: save model checkpoint
    4. If validation plateaus: reduce learning rate
  
Testing:
  Final evaluation on held-out test set:
    1. Run model on 1,000 test samples (model never saw these)
    2. Calculate final performance metrics
    3. Report accuracy, uncertainty calibration, edge case performance
```

---

## Key Design Decisions and Why We Made Them

### 1. Why Three Detectors?

```
H1, L1, V1 provides:
  - TRIANGULATION: Multiple views help locate the source
  - NOISE MITIGATION: Real events coherent across detectors, glitches localized
  - POLARIZATION: Different detector orientations see different polarizations
  - REDUNDANCY: If one detector fails, others can still detect signals
  
In our training:
  - Model learns to use information from all three
  - Can handle cases where one detector is noisy
  - Learns to combine evidence for stronger detection
```

### 2. Why 4 Seconds Duration?

```
4 seconds = good compromise between:
  - BBH signals: Last ~1-5 seconds in sensitive band
  - BNS signals: Last ~100-1000 seconds (but we capture early part)
  - Computational cost: 4096 Hz × 4 sec = 16,384 samples (manageable)
  - Memory: 3 detectors × 16,384 × float32 = ~200 KB per sample
  
Too short (1 second):
  - Would miss long inspiral phases of heavy systems
  
Too long (10 seconds):
  - Computational cost becomes prohibitive
  - Memory usage explodes
```

### 3. Why Overlap Signals This Way?

```
Instead of synthetic overlaps, why not:
  - Use real overlapping GWTC events? 
    → Too rare, can't generate enough variety
  
Our approach (synthetic overlaps with time clustering):
  - Realistic: Signals cluster in time like real scenarios
  - Controllable: We decide how many, what types, what SNRs
  - Diverse: Can generate thousands of different combinations
  - Labeled: We know exactly what signals are there
  
The clustering ensures:
  - Signals overlap for a measurable time window
  - Not just touching at edges (unrealistic)
  - Temporal proximity matters (like real events)
```

### 4. Why SNR-Distance Coupling?

```
Problem with naive approach:
  Sample mass → Sample distance → Calculate SNR
  → Masses and distances are independent
  → Creates unrealistic parameter space
  
Problem fixed with our approach:
  Sample mass → Sample SNR → Calculate distance
  → SNR and distance are coupled by physics
  → Creates realistic parameter correlations
  → Model learns real physics automatically
  
Real universe observation:
  - Most detected events are at moderate distance (400-500 Mpc)
  - Very heavy nearby events (strong SNR)
  - Very light distant events (weak SNR)
  - NOT: random combinations of mass and distance
```

---

## Running the Generation

### Command Line (Simple)

```bash
# Activate the environment
conda activate ahsd

# Generate dataset using config file
ahsd-generate --config configs/data_config.yaml

# What happens:
#   1. Reads configuration (n_samples, SNR ranges, etc.)
#   2. Initializes all components
#   3. Generates 10,000 samples one by one
#   4. Saves batches of 100 to disk
#   5. Prints progress and statistics
#   Takes ~3-4 hours on typical CPU
```

### Python Script (Detailed Control)

```python
from ahsd.data.dataset_generator import GWDatasetGenerator
from ahsd.data.config import load_config

# Load configuration
config = load_config('configs/data_config.yaml')

# Create generator
gen = GWDatasetGenerator(
    output_dir='data/dataset',
    sample_rate=4096,
    duration=4.0,
    config=config
)

# Generate dataset
dataset = gen.generate_dataset(
    n_samples=10000,
    overlap_fraction=0.45,      # 45% multi-signal samples
    edge_case_fraction=0.08,    # 8% edge cases
    preprocess=True,            # Apply whitening, filtering
    save_batch_size=100         # Save every 100 samples
)

# Result: 10,000 samples saved to disk in train/val/test splits
```

---

## Checking If Everything Worked

### Quick Sanity Check

```python
import pickle
import numpy as np

# Load one sample
with open('data/dataset/train_split.pkl', 'rb') as f:
    samples = pickle.load(f)

sample = samples[0]

# Check structure exists
assert sample['type'] in ['BBH', 'BNS', 'NSBH', 'overlap', 'noise']
assert 'parameters' in sample
assert 'detector_data' in sample
assert 'priorities' in sample

# Check detector data
for det in ['H1', 'L1', 'V1']:
    strain = sample['detector_data'][det]['strain']
    # Should be: 4 seconds × 4096 Hz = 16,384 samples
    assert len(strain) == 16384
    # Should be float32
    assert strain.dtype == np.float32
    # Should be finite (no NaN or Inf)
    assert np.all(np.isfinite(strain))
    # Should have reasonable magnitude
    assert np.max(np.abs(strain)) < 1e-18

# Check parameters
params = sample['parameters'][0]
assert params['mass_1'] >= params['mass_2']
assert params['luminosity_distance'] > 0
assert 'target_snr' in params or 'network_snr' in params

# Check priorities
assert len(sample['priorities']) == len(sample['parameters'])
assert all(0 <= p <= 1 for p in sample['priorities'])

print("✓ Sample structure is valid!")
print(f"  Type: {sample['type']}")
print(f"  Signals: {sample['n_signals']}")
print(f"  SNR: {sample['priorities']}")
```

### Full Dataset Validation

```python
def validate_full_dataset(dataset_path):
    """Check entire dataset for consistency"""
    
    with open(dataset_path, 'rb') as f:
        samples = pickle.load(f)
    
    print(f"Loaded {len(samples)} samples")
    
    stats = {
        'types': {},
        'n_signals': {},
        'snr_ranges': {'min': 999, 'max': 0},
        'errors': []
    }
    
    for i, sample in enumerate(samples):
        # Count event types
        event_type = sample['type']
        stats['types'][event_type] = stats['types'].get(event_type, 0) + 1
        
        # Count overlap sizes
        n = sample['n_signals']
        stats['n_signals'][n] = stats['n_signals'].get(n, 0) + 1
        
        # Track SNR range
        for priority in sample['priorities']:
            stats['snr_ranges']['min'] = min(stats['snr_ranges']['min'], priority)
            stats['snr_ranges']['max'] = max(stats['snr_ranges']['max'], priority)
        
        # Check detector data integrity
        try:
            for det in ['H1', 'L1', 'V1']:
                strain = sample['detector_data'][det]['strain']
                assert len(strain) == 16384
                assert strain.dtype == np.float32
                assert np.all(np.isfinite(strain))
        except Exception as e:
            stats['errors'].append((i, str(e)))
    
    # Print results
    print("\nEvent Type Distribution:")
    for event_type, count in sorted(stats['types'].items()):
        pct = 100 * count / len(samples)
        print(f"  {event_type}: {count} ({pct:.1f}%)")
    
    print("\nSignal Count Distribution:")
    for n_sig, count in sorted(stats['n_signals'].items()):
        pct = 100 * count / len(samples)
        print(f"  {n_sig} signals: {count} ({pct:.1f}%)")
    
    print(f"\nSNR Range: {stats['snr_ranges']['min']:.3f} to {stats['snr_ranges']['max']:.3f}")
    print(f"Errors found: {len(stats['errors'])}")
    
    if len(stats['errors']) > 0:
        print("\nFirst few errors:")
        for idx, error in stats['errors'][:5]:
            print(f"  Sample {idx}: {error}")
```

---

## Summary: The Whole Pipeline at a Glance

```
┌─────────────────────────────────────────────────────────┐
│         DATASET GENERATION PIPELINE SUMMARY             │
└─────────────────────────────────────────────────────────┘

INPUT:
  Configuration file (data_config.yaml)
    ├─ Dataset size: 10,000 samples
    ├─ SNR distribution: 5% weak, 35% low, 45% medium, 12% high, 3% loud
    ├─ Event types: 46% BBH, 32% BNS, 17% NSBH, 5% noise
    ├─ Overlaps: 45% of samples have multiple signals
    └─ Edge cases: 8% physically or observationally extreme

PROCESSING STAGES:

Stage 1: Parameter Sampling
  Sample: Masses, spins, distances, sky locations
  Ensure: Physics consistency (SNR ↔ distance coupling)
  Result: 10,000 unique parameter sets

Stage 2: Waveform Generation
  Method 1: PyCBC (accurate, slower)
  Method 2: Analytical PN (good approximation)
  Method 3: Simple chirp (fallback)
  Result: 10,000 time-domain waveforms

Stage 3: Noise Generation
  Create: Colored noise matching LIGO/Virgo PSD
  Replicate: Three independent detector noise realizations
  Result: Realistic background for each detector

Stage 4: Signal Injection
  Single: Mix waveform + noise for each detector
  Multiple: Mix 2-4+ overlapping waveforms with time offsets
  Result: Signal + noise for each detector

Stage 5: Preprocessing
  Whiten: Remove frequency-dependent noise structure
  Filter: Keep only 30-500 Hz (GW signal band)
  Taper: Smooth edges to reduce artifacts
  Result: ML-ready data (whitened, filtered, tapered)

Stage 6: Priority Assignment
  Compute: SNR for each signal from detector data
  Normalize: Scale to [0, 1] range
  Rank: Model learns to predict these priorities
  Result: Ground truth labels for training

Stage 7: Packaging
  Organize: All detector data + parameters + metadata
  Format: Efficient pickle or HDF5 format
  Save: Batch saves to disk every 100 samples
  Result: 10,000 complete samples

OUTPUT:
  Training set: 8,000 samples (80%)
  Validation set: 1,000 samples (10%)
  Test set: 1,000 samples (10%)
  
  Each sample contains:
    - Strain from 3 detectors (H1, L1, V1)
    - Waveform parameters (masses, distances, etc.)
    - Ground truth priorities (SNRs)
    - Metadata (how it was generated)

PROPERTIES OF RESULTING DATA:
  ✓ Physically realistic (SNR-distance coupling)
  ✓ Diverse (many types, SNRs, overlaps, edge cases)
  ✓ Well-labeled (know true parameters and priorities)
  ✓ Preprocessed (whitened, filtered, ready for ML)
  ✓ Matches LIGO/Virgo characteristics
  ✓ Imbalanced like real observations (more BBH, less BNS)
  ✓ Includes overlapping signals (realistic challenge)

TIME TO GENERATE:
  Single sample: 1-2 seconds
  10,000 samples: 3-4 hours (parallel batch processing)
  
DATA SIZE:
  Per sample: ~100 MB (strain data + metadata)
  Total: ~1 TB for 10,000 samples
```

---

## Key Takeaways

1. **Physics First**: Our system generates realistic data by embedding physics into the sampling process (SNR ↔ distance coupling)

2. **Realistic Diversity**: We include single signals, overlaps, edge cases, and noise to prepare the model for real detector challenges

3. **Proper Fallbacks**: Waveform generation has backup methods to ensure every sample gets a signal

4. **Comprehensive Preprocessing**: Data is whitened, filtered, and tapered to match what real detectors deliver to ML systems

5. **Ground Truth Labels**: Unlike real detector data, we know exactly what signals are present and can train models to detect and measure them

6. **Scalable Design**: The pipeline can generate datasets of any size, with configurable parameters for research flexibility

---

*This document explains the **WHAT**, **WHY**, and **HOW** of dataset generation.*
*For detailed mathematical formulas and API references, see DATASET_GENERATION_DOCUMENTATION.md*

*Last Updated: November 2025*
