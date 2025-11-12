# 🌊 PosteriFlow — Adaptive Hierarchical Signal Decomposition (AHSD)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()

> **A next-generation gravitational-wave analysis system that detects, decomposes, and characterizes overlapping signals in real-time using neural posterior estimation and adaptive signal subtraction.**

---

## 🎯 What is PosteriFlow?

PosteriFlow is a cutting-edge machine learning pipeline for gravitational-wave astronomy that solves a critical problem: **how to extract multiple overlapping signals from noisy gravitational-wave detector data**.

### The Core Problem
Modern gravitational-wave detectors (LIGO, Virgo) detect weak signals buried in noise. When **multiple sources merge simultaneously**, their signals overlap, creating a complex mixture that traditional methods cannot easily separate. PosteriFlow uses **hierarchical neural networks** to:

1. **Prioritize signals** - Determine which sources to extract first
2. **Estimate parameters** - Rapidly infer masses, distances, spins using neural inference
3. **Subtract adaptively** - Remove extracted signals while preserving fainter ones
4. **Quantify uncertainty** - Provide calibrated confidence intervals for all estimates

### Why This Matters
- **Multi-messenger astronomy**: Early warnings for neutron star mergers enable electromagnetic follow-up
- **Population statistics**: Extracting overlapping events improves population constraints on compact object formation
- **Real-time decision-making**: LIGO alert system can trigger faster with overlapping signals disentangled
- **Scientific discovery**: Overlaps may reveal unexpected binary characteristics (precession, eccentricity)

---

## 🏗️ Architecture Overview

### Three-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│         RAW GRAVITATIONAL-WAVE DATA (H1, L1, V1)             │
│     Detector noise + overlapping GW signals + glitches       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────────┐
          │  PHASE 1: NEURAL POSTERIOR       │
          │  ESTIMATION (Neural PE)          │
          │  ─────────────────────────────   │
          │  • Likelihood-free inference     │
          │  • Multi-detector coherence      │
          │  • Uncertainty quantification    │
          └──────────────┬───────────────────┘
                         │
          Parameter estimates + uncertainties
          (mass_1, mass_2, distance, sky position, spins)
                         │
                         ▼
          ┌──────────────────────────────────┐
          │  PHASE 2: PRIORITY NET            │
          │  Signal Ranking & Selection       │
          │  ─────────────────────────────   │
          │  • Temporal encoding (CNN+BiLSTM)│
          │  • Cross-signal feature analysis  │
          │  • Uncertainty-aware ranking      │
          │  • Predicts extraction order      │
          └──────────────┬───────────────────┘
                         │
                     Ordered list of signals
                     (which to remove first)
                         │
                         ▼
          ┌──────────────────────────────────┐
          │  PHASE 3: ADAPTIVE SUBTRACTOR     │
          │  Iterative Signal Removal         │
          │  ─────────────────────────────   │
          │  • Uncertainty-weighted subtraction
          │  • Cross-detector coherence       │
          │  • Bias correction               │
          │  • Residual quality monitoring    │
          └──────────────┬───────────────────┘
                         │
                         ▼
       ┌──────────────────────────────────────────┐
       │  EXTRACTED SIGNALS & RESIDUAL NOISE      │
       │  • Individual source parameters          │
       │  • Parameter uncertainties               │
       │  • Signal-to-noise metrics               │
       │  • Residual quality assessment           │
       └──────────────────────────────────────────┘
```

### Key Neural Components

#### **Neural PE (Parameter Estimation)**
- Likelihood-free inference using normalizing flows
- Simultaneous estimation of ~15 binary parameters
- Fast inference: <100ms for 4-second segment
- Uncertainty quantification via posterior ensemble
- Handles contamination via data augmentation

#### **PriorityNet (Signal Prioritization)**
- Temporal CNN encoder: Multi-scale time-frequency features
- BiLSTM encoder: Temporal dependencies in strain data
- Cross-signal analyzer: Quantifies signal overlap and interaction
- Output: Ranking of signals + confidence in order
- Enables optimal extraction strategy

#### **Adaptive Subtractor**
- Uses Neural PE uncertainties to weight residuals
- Subtracts strongest signal first (per PriorityNet)
- Bias correction: Accounts for parameter estimation errors
- Iterative: Updates estimates after each subtraction
- Quality monitoring: Validates residual Gaussianity

---

## 💾 Data Pipeline

### Synthetic Dataset Generation

PosteriFlow generates realistic synthetic gravitational-wave data for training:

```
REAL LIGO/VIRGO CHARACTERISTICS
├─ Detector network (H1, L1, V1)
├─ Realistic PSDs from O4 sensitivity
├─ Real glitches & contamination
├─ Physics-accurate waveforms (IMRPhenomXAS)
└─ Realistic source populations

                    ▼

PARAMETERS SAMPLED (Physics-Constrained)
├─ Masses (BBH: 5-100 M☉, BNS: 1-2.5 M☉)
├─ Spins (aligned & precessing)
├─ Distance (~log-uniform, Malmquist bias)
├─ Sky position (uniform on sphere)
└─ Binary merger epoch

                    ▼

SIGNAL GENERATION
├─ GW waveform synthesis (PyCBC)
├─ Detector response (antenna patterns)
├─ SNR-dependent distance scaling
└─ Parameter-distance correlation (physics-validated)

                    ▼

CONTAMINATION INJECTION
├─ Real LIGO noise (GWOSC, 10-25× speedup via caching)
├─ Neural synthetic noise (10,000× faster than GWOSC)
├─ Line glitches (60 Hz, harmonics)
├─ Transient glitches (blips, scattered light)
├─ PSD drift (multiple epochs)
└─ Detector dropout scenarios

                    ▼

OVERLAP CREATION (45% realistic rate)
├─ 2-signal overlaps (direct mergers)
├─ Multi-signal overlaps (up to 8 signals)
├─ Partial overlaps (different durations)
└─ Subtle ranking (important for prioritization)

                    ▼

EDGE CASE SAMPLING (8% of dataset)
├─ Physical extremes (high mass-ratio, spins)
├─ Observational extremes (strong glitches)
├─ Statistical extremes (multimodal posteriors)
└─ Overlapping extremes (subtle ranking)

                    ▼

FINAL DATASET (25,000+ samples)
├─ Detector strain (H1, L1, V1) + preprocessing
├─ Ground-truth parameters
├─ Network SNR & quality metrics
├─ Metadata for analysis
└─ Train/val/test splits (80/10/10)
```

### Data Statistics

```
SIGNAL TYPE DISTRIBUTION:
├─ Binary Black Hole (BBH):    46% → Loudest, most common
├─ Binary Neutron Star (BNS):  32% → Rare, long duration, crucial for EW
├─ NS-BH (NSBH):               17% → Intermediate
└─ Noise only:                  5% → Background characterization

OVERLAP STATISTICS:
├─ Single signals:      55% of samples
├─ Overlapping:         45% of samples
│  ├─ 2-3 signals:      35%
│  ├─ 4-5 signals:       8%
│  └─ 6+ signals:        2%
└─ Average: 2.25 signals per sample

SNR DISTRIBUTION (O4 REALISTIC):
├─ Weak (10-15):        5%
├─ Low (15-25):        35%  ← Most detections
├─ Medium (25-40):     45%
├─ High (40-60):       12%
└─ Loud (60-80):        3%

PARAMETER RANGES:
├─ Masses:    3-200 M☉  (detector frame)
├─ Distances: 10-18,000 Mpc
├─ Spins:     0-0.99
└─ SNR:       3-100
```

### Advanced Features

**Real Noise Integration (10-25× speedup)**
- Pre-downloaded GWOSC segments (133 cached files)
- Three-level fallback: cache → on-demand → synthetic
- 10% real noise mixing for enhanced realism

**Neural Noise Generation (10,000× speedup)**
- FMPE pre-trained models (Gaussian_network.pickle)
- Colored Gaussian & non-Gaussian variants
- Falls back gracefully if models unavailable


**TransformerStrainEncoder Enhancement**
- State-of-the-art strain encoding
- Attention-based temporal modeling
- Outperforms CNN+BiLSTM baselines


---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/bibinthomas123/PosteriFlow.git
cd PosteriFlow

# Initialize conda (first time only)
conda init

# Activate environment
conda activate ahsd

# Install package in development mode
pip install -e . --no-deps
```

**Important:** The conda environment `ahsd` exists and contains all dependencies. Never recreate it.

### 2. Generate Training Data

```bash
# Generate 25,000 samples (default, ~1.5-2 hours)
python src/ahsd/data/scripts/generate_dataset.py \
    --config configs/data_config.yaml \
    --num-samples 25000

# Custom parameters
python src/ahsd/data/scripts/generate_dataset.py \
    --config configs/data_config.yaml \
    --num-samples 50000 \
    --output-dir data/dataset_custom
```

### 3. Train Phase 1: Neural PE

```bash
# Train neural parameter estimation network
python experiments/phase3a_neural_pe.py \
    --config configs/enhanced_training.yaml \
    --batch-size 32 \
    --epochs 100

# Monitor training
tensorboard --logdir outputs/
```

### 4. Train Phase 2: PriorityNet

```bash
# Train signal prioritization network
python experiments/train_priority_net.py \
    --config configs/priority_net.yaml \
    --create-overlaps \
    --batch-size 16

# Resume from checkpoint
python experiments/train_priority_net.py \
    --resume outputs/prioritynet_checkpoint.pth \
    --create-overlaps
```

### 5. Evaluate & Validate

```bash
# Full validation suite
python experiments/phase3c_validation.py \
    --phase3a_output outputs/phase3a_output_X/ \
    --phase3b_output outputs/phase3b_production/ \
    --n_samples 2000 \
    --seeds 5

# Expected output:
# ✅ System Success Rate: 82.1%
# ✅ Neural PE Accuracy: 0.582 ± 0.087
# ✅ Subtraction Efficiency: 81.1%
```

---

## 📊 Performance Results

### System-Level Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **System Success Rate** | 82.1% | End-to-end detection of all signals |
| **Average Efficiency (η)** | 81.1% | Residual energy reduction |
| **Latency per 4s segment** | 156 ms | Dual-channel (H1, L1) |
| **Throughput** | 25.6 seg/s | Real-time capable |
| **Memory (8GB VRAM)** | Fits | Batch inference supported |

### Phase 1: Neural PE Accuracy

| Dataset | APE (mean) | APE (std) | Comments |
|---------|-----------|----------|----------|
| Clean (training) | 0.802 | 0.012 | Physics-perfect data |
| Contaminated (validation) | 0.582 | 0.087 | Realistic noise |
| After subtraction | 0.645 | 0.074 | Improved residuals |

### Phase 2: PriorityNet Ranking

| Metric | Value | Target |
|--------|-------|--------|
| Top-K Precision@1 | 96.6% | >95% |
| Ranking Correlation | 0.605 | >0.50 |
| Priority Accuracy | 94.6% | >90% |
| Calibration Error | <0.05 | <0.10 |

### Phase 3: Multi-Seed Verification

```
METRIC STABILITY ACROSS 5 SEEDS (200 samples each):
├─ Neural PE Accuracy:  0.582 ± 0.004  (variation: 0.1%)
├─ Subtraction η:       0.811 ± 0.001  (variation: <0.1%)
├─ System Success:      0.821 ± 0.008  (variation: 1.0%)
└─ Statistical significance: Cohen's d > 2.0
```

---

## 📁 Project Structure

```
PosteriFlow/
├── 📁 src/ahsd/                    # Main package
│   ├── 📁 core/                    # Core algorithms
│   │   ├── priority_net.py          # Signal prioritization (PriorityNet)
│   │   ├── adaptive_subtractor.py   # Adaptive subtraction + NeuralPE
│   │   ├── ahsd_pipeline.py         # Full end-to-end pipeline
│   │   └── bias_corrector.py        # Parameter bias correction
│   ├── 📁 data/                    # Data generation & preprocessing
│   │   ├── dataset_generator.py     # Main dataset generator
│   │   ├── waveform_generator.py    # GW waveform synthesis (PyCBC)
│   │   ├── noise_generator.py       # Synthetic noise + glitches
│   │   ├── neural_noise_generator.py # FMPE neural noise (10k× speedup)
│   │   ├── parameter_sampler.py     # Physics-constrained sampling
│   │   ├── psd_manager.py          # Power spectral density management
│   │   ├── gwtc_loader.py          # Real GWOSC data loading
│   │   ├── injection.py            # Signal injection into noise
│   │   ├── preprocessing.py        # Whitening, normalization
│   │   └── config.py               # Config loading & validation
│   ├── 📁 models/                  # Neural network architectures
│   │   ├── neural_pe.py            # Neural PE normalizing flow
│   │   ├── overlap_neuralpe.py      # Multi-signal PE variant
│   │   ├── transformer_encoder.py   # TransformerStrainEncoder
│   │   ├── flows.py                # Flow architectures
│   │   └── rl_controller.py         # RL-based control (future)
│   ├── 📁 evaluation/              # Metrics & analysis
│   │   └── metrics.py              # APE, efficiency, ranking metrics
│   └── 📁 utils/                   # Utilities
│       ├── config.py               # Configuration classes
│       ├── logging.py              # Logging setup
│       └── data_format.py           # Data standardization
├── 📁 experiments/                 # Training & evaluation scripts
│   ├── phase3a_neural_pe.py        # Neural PE training
│   ├── train_priority_net.py        # PriorityNet training
│   ├── data_generation.py          # Dataset generation wrapper
│   └── phase3c_validation.py        # Multi-seed validation
├── 📁 configs/                     # Configuration files (YAML)
│   ├── data_config.yaml            # Data generation parameters
│   ├── enhanced_training.yaml      # Training hyperparameters
│   ├── priority_net.yaml           # PriorityNet config
│   └── inference.yaml              # Inference settings
├── 📁 tests/                       # Unit & integration tests
│   ├── test_dataset_generation.py
│   ├── test_neural_pe.py
│   ├── test_priority_net.py
│   └── test_integration.py
├── 📁 models/                      # Trained model checkpoints
│   ├── neural_pe_best.pth
│   └── prioritynet_checkpoint.pth
├── 📁 data/                        # Generated datasets
│   ├── dataset/
│   │   ├── train.pkl
│   │   ├── val.pkl
│   │   └── test.pkl
│   └── Gaussian_network.pickle     # FMPE model (neural noise)
├── 📁 outputs/                     # Experiment results
│   ├── phase3a_output_X/
│   ├── phase3b_production/
│   └── logs/
├── 📁 gw_segments_cleaned/         # Pre-cached GWOSC segments
│   └── [133 real noise segments]
├── 📁 notebooks/                   # Analysis & visualization
├── 📁 docs/                        # Additional documentation
├── pyproject.toml                  # Package metadata & dependencies
├── setup.py                        # Package setup
├── AGENTS.md                       # Development guidelines
└── README.md                       # This file
```

---

## 🔧 Configuration System

All parameters are controlled via YAML configuration files in `configs/`:

### data_config.yaml - Dataset Generation

```yaml
# Core parameters
n_samples: 25000              # Number of samples to generate
sample_rate: 4096             # Hz (LIGO standard)
duration: 4.0                 # seconds
detectors: [H1, L1, V1]      # Detector network

# Signal characteristics
overlap_fraction: 0.45        # Realistic O4 rate
edge_case_fraction: 0.08      # Physical/statistical extremes
create_overlaps: true         # Enable multi-signal generation

# Contamination
add_glitches: true
neural_noise_enabled: true    # 10,000× speedup
neural_noise_prob: 0.5        # 50% neural, 50% synthetic
use_real_noise_prob: 0.1      # 10% real GWOSC (cached)

# Event distribution (realistic O4)
event_type_distribution:
  BBH: 0.46                   # Most common
  BNS: 0.32                   # Rare but important
  NSBH: 0.17                  # Intermediate
  noise: 0.05                 # Background
```

### enhanced_training.yaml - Neural PE Training

```yaml
# Hyperparameters
learning_rate: 0.0005
batch_size: 32
epochs: 100
weight_decay: 1e-5

# Loss weights
loss_weights:
  mse: 0.35                   # Parameter estimation
  ranking: 0.50               # Ranking loss
  uncertainty: 0.15           # Calibration

# Data augmentation
augment_contamination: true
noise_augmentation_k: 1.0
preprocess: true
```

### priority_net.yaml - Signal Prioritization

```yaml
# Architecture
temporal_encoder_dim: 128
hidden_dim: 256
num_heads: 8                  # Multi-head attention

# Training
learning_rate: 0.0002
batch_size: 16
epochs: 80
create_overlaps: true         # Enable multi-signal training
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
pytest

# Specific test
pytest tests/test_priority_net.py::TestPriorityNet::test_forward_pass -v

# With coverage
pytest --cov=ahsd --cov-report=html

# Verbose with print statements
pytest -v -s

# Specific test file
pytest tests/test_neural_pe.py
```

### Key Test Suites

| Test | Purpose | Location |
|------|---------|----------|
| Neural PE | Forward pass, loss computation | `tests/test_neural_pe.py` |
| PriorityNet | Signal ranking, feature extraction | `tests/test_priority_net.py` |
| Dataset | Data generation, splits, validation | `tests/test_dataset_generation.py` |
| Integration | End-to-end pipeline | `tests/test_integration.py` |

---

## 💡 How to Use PosteriFlow

### Use Case 1: Train on Custom Data

1. Prepare real GW data in HDF5 format
2. Implement data reader in `src/ahsd/data/gwtc_loader.py`
3. Update `data_config.yaml` with real data paths
4. Run training pipeline

### Use Case 2: Parameter Estimation on New Events

```python
from ahsd.core.adaptive_subtractor import NeuralPE
import numpy as np

# Load strain data
strain_data = {
    'H1': np.load('H1_data.npy'),
    'L1': np.load('L1_data.npy'),
    'V1': np.load('V1_data.npy'),
}

# Quick estimation
pe = NeuralPE()
result = pe.quick_estimate(strain_data)

print(f"Mass 1: {result['mass_1_mean']:.1f} M☉")
print(f"Distance: {result['luminosity_distance_mean']:.0f} Mpc")
print(f"SNR: {result['network_snr']:.1f}")
```

### Use Case 3: Signal Decomposition Pipeline

```python
from ahsd.core.ahsd_pipeline import AHSDPipeline

# Initialize pipeline
pipeline = AHSDPipeline(
    neural_pe_model='models/neural_pe_best.pth',
    priority_net_model='models/prioritynet_best.pth',
    subtractor_model='models/subtractor_best.pth',
)

# Process 4-second segment
result = pipeline.run(strain_data={
    'H1': h1_strain,
    'L1': l1_strain,
    'V1': v1_strain,
})

# Extracted signals
for i, signal in enumerate(result['extracted_signals']):
    print(f"\nSignal {i+1}:")
    print(f"  Mass 1: {signal['mass_1']:.1f} M☉")
    print(f"  SNR: {signal['snr']:.1f}")
    print(f"  Confidence: {signal['priority_score']:.2f}")
```

---

## 🔬 Scientific Details

### Neural Posterior Estimation (Phase 1)

**Approach:** Likelihood-free inference using normalizing flows
- **Input:** Multi-detector strain (whitened, windowed)
- **Output:** Posterior samples of ~15 astrophysical parameters
- **Speed:** <100ms per 4s segment
- **Training:** On clean synthetic waveforms + augmented contamination

**Key Features:**
- Amortized inference: Single network for all parameters
- Uncertainty quantification: Full posterior ensemble
- Multi-detector coherence: Combines H1, L1, V1 optimally
- Robust to PSD variation: Data augmentation during training

### Signal Prioritization (Phase 2: PriorityNet)

**Approach:** Deep learning on temporal strain features
- **Architecture:** CNN (multi-scale) + BiLSTM (temporal) + Attention (context)
- **Input:** Whitened strain for multiple signals
- **Output:** Ranking order (which signal to subtract first)
- **Training:** On overlapping synthetic signals

**Why Prioritization Matters:**
- Extracting loud signal first reduces noise floor
- Removes contamination bias on faint signals
- Improves overall parameter estimation accuracy
- Handles multimodal posteriors better

### Adaptive Subtraction (Phase 3)

**Approach:** Iterative removal with uncertainty weighting
- **Step 1:** Identify signal with highest priority
- **Step 2:** Subtract using Neural PE parameters + uncertainties
- **Step 3:** Bias correction: Account for parameter errors
- **Step 4:** Validate residual Gaussianity
- **Step 5:** Repeat for remaining signals

**Uncertainty Weighting:**
- Larger uncertainties → weaker subtraction (preserve signal)
- Calibrated uncertainties → correct bias
- Cross-detector coherence check

---

## 📚 References

### Key Papers

1. **PyCBC Waveforms**: [arXiv:1508.01844](https://arxiv.org/abs/1508.01844)
   - GW waveform generation and detection

2. **LIGO Data Conditioning**: [arXiv:2002.01606](https://arxiv.org/abs/2002.01606)
   - Real gravitational-wave detector noise

3. **Normalizing Flows**: [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)
   - Flexible density estimation (used in Neural PE)

4. **DINGO**: [arXiv:2105.12151](https://arxiv.org/abs/2105.12151)
   - Deep inference for GW observations (basis for neural noise models)

### Data Sources

- **GWOSC**: [gwosc.readthedocs.io](https://gwosc.readthedocs.io/)
  - Public gravitational-wave detector data
- **GWTC-3**: [arXiv:2105.15615](https://arxiv.org/abs/2105.15615)
  - LIGO-Virgo third catalogs of GW transients

---

## 🤝 Contributing

### Development Workflow

1. **Create feature branch**: `git checkout -b feature/description`
2. **Code style**: Follow AGENTS.md guidelines
3. **Test**: Run `pytest` before committing
4. **Format**: `black . && isort . && flake8 .`
5. **Commit message**: Descriptive, explain "why"
6. **Push & PR**: Create pull request with summary

### Code Standards

- **Type hints:** Always (required for all functions)
- **Docstrings:** NumPy format for classes and methods
- **Line length:** 100 characters (black formatter)
- **Testing:** Unit tests for new modules
- **Coverage:** Aim for >80% for new code

---

## 📞 Support & Resources

### Documentation
- **Docs** - Use this folder to understand the core functionality and how to run the code

### Commands

```bash
# Data generation
ahsd-generate --config configs/data_config.yaml

# Validation
ahsd-validate --dataset data/dataset/train.pkl

# Analysis
ahsd-analyze --input-data data.hdf5 --output results.pkl

# Model training
python experiments/phase3a_neural_pe.py --config configs/enhanced_training.yaml

# Validation
python experiments/phase3c_validation.py --phase3a_output outputs/phase3a_output_X/ \
    --phase3b_output outputs/phase3b_production/ --n_samples 2000 --seeds 5
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details

---

## 👤 Author & Citation

**Author:** Bibin Thomas  
**Email:** bibinthomas951@gmail.com  
**Repository:** https://github.com/bibinthomas123/PosteriFlow

### Citation

If you use PosteriFlow in your research, please cite:

```bibtex
@software{thomas2025posteriflow,
  title={PosteriFlow: Adaptive Hierarchical Signal Decomposition 
         for Overlapping Gravitational Waves},
  author={Thomas, Bibin},
  year={2025},
  url={https://github.com/bibinthomas123/PosteriFlow}
}
```

---

## 🌟 Acknowledgments

PosteriFlow builds on foundational work from:
- **LIGO-Virgo Collaboration** for detector design and data access
- **PyCBC** for waveform generation
- **Bilby** for Bayesian inference tools
- **GWpy** for detector data handling
- **DINGO** for neural density estimation techniques

---

**Built for the next generation of gravitational-wave astronomy** 🌌

*Last Updated: November 12, 2025*
