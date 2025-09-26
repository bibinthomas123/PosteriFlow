# 🌊 AHSD — Adaptive Hierarchical Signal Decomposition

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Paper-brightgreen.svg)]()

> *A cutting-edge, real-time gravitational-wave data conditioning system that intelligently removes contamination while preserving precious astrophysical signals.*

---

## 🚀 Overview

AHSD is a sophisticated **two-phase neural pipeline** designed for next-generation gravitational wave astronomy:

* **🧠 Phase 1**: Neural Posterior Estimation (PE) of source parameters and uncertainties
* **🎯 Phase 2**: Uncertainty-aware Adaptive Subtraction for intelligent contamination removal  
* **🔬 Phase 3A–3C**: Comprehensive training, validation, and independent multi-seed verification

**Key Features:**
* ⚡ **Real-time processing** (0.156s per 4s segment)
* 🎨 **Built-in synthetic data generation** with LIGO-like contamination
* 🌍 **Real data adaptation** capabilities
* 📊 **Production-ready** with 82.1% system success rate

---

## 📦 Installation

### Quick Start
```bash
# Create environment
conda env create -f environment.yaml
conda activate ahsd

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, numpy; print('✅ PyTorch:', torch.__version__, '| NumPy:', numpy.__version__)"
```

---

## 🎛️ Data Generation

Synthetic data is **dynamically generated** according to configurations in `configs/*.yaml`.

### 📋 Default Configuration
```yaml
Sampling Rate: 4096 Hz
Segment Length: 4 seconds
Channels: Dual (H1/L1)
Contamination:
  - 60 Hz line noise
  - 2 Hz seismic interference  
  - 100 Hz narrowband artifacts
  - Transient glitches (~2s)
  - Low-level Gaussian noise
```

### 🔧 Customization
* **Data parameters**: Edit `configs/data.yaml`
* **Real data integration**: Implement readers in `data/` directory

---

## 🎯 Usage Guide

### Phase 1 — 🧠 Neural Posterior Estimation

Train the neural PE network on physics-perfect waveforms:

```bash
python experiments/phase3a_neural_pe.py --config configs/phase3a.yaml
```

**📁 Outputs:** `outputs/phase3a_output_X/` *(checkpoints, metrics, logs)*

---

### Phase 2 — 🎯 Adaptive Subtractor

Train the intelligent subtractor using Phase 1 outputs:

```bash
python experiments/phase3b_subtractor.py \
    --phase3a_output outputs/phase3a_output_X/ \
    --config configs/phase3b.yaml
```

**📁 Outputs:** `outputs/phase3b_production/` *(models, metrics, learning curves)*

---

### Phase 3 — 🔬 Evaluation & Verification

#### 📊 Phase 3A: Accuracy Check
```bash
python experiments/phase3a_eval.py \
    --phase3a_output outputs/phase3a_output_X/ \
    --config configs/phase3a_eval.yaml
```

#### 🎯 Phase 3B: Validation
```bash
python experiments/phase3c_validation.py \
    --phase3b_output outputs/phase3b_production/ \
    --config configs/phase3c.yaml
```

#### ✅ Phase 3C: Multi-Seed Verification
```bash
python experiments/phase3c_validation.py \
    --phase3b_output outputs/phase3b_production/phase3b_working_output.pth \
    --n_samples 2000 --seeds 5 --verbose
```

---

### ⚡ Quick Inference Test

```bash
python experiments/run_inference.py \
    --phase3a outputs/phase3a_output_X/best.pth \
    --phase3b outputs/phase3b_production/phase3b_working_output.pth \
    --config configs/inference.yaml
```

---

## 📊 DATASET STRUCTURE

### 🎲 Challenge Level Distribution
```
Easy:                227 samples (0.9%)
Medium:            15,990 samples (62.0%)
Very Hard:            592 samples (2.3%)
Extreme:            1,175 samples (4.6%)
Real:                  52 samples (0.2%)
Real Inspired:      2,940 samples (11.4%)
Real Multi:         4,814 samples (18.7%)
```

### 🔬 Data Type Breakdown
```
Pure Synthetic:     5,250 samples (45.7%)
Real Augmented:     2,940 samples (25.6%)
Real Multi-Aug:     2,184 samples (19.0%)
Extreme Scenarios:    600 samples (5.2%)
Low SNR Challenge:    300 samples (2.6%)
High SNR Pristine:    150 samples (1.3%)
Real Background:       52 samples (0.5%)
```

### 📈 Parameter Statistics
| Parameter | Min | Mean | Max | Std |
|-----------|-----|------|-----|-----|
| **Mass 1** | 3.1 M☉ | 36.4 M☉ | 201.6 M☉ | 16.7 M☉ |
| **Mass 2** | 1.0 M☉ | 26.4 M☉ | 149.9 M☉ | 10.8 M☉ |
| **Distance** | 10 Mpc | 1,654 Mpc | 17,638 Mpc | 2,198 Mpc |
| **SNR** | 3.0 | 12.6 | 100.0 | 9.8 |

### 🎯 Dataset Overview
```yaml
Total Scenarios: 11,476
Total Signals: 25,790
Avg Signals/Scenario: 2.25
Creation: 2025-09-24 10:25:49
Diversity Score: 0.440 (High: 29.6%, Very High: 27.6%)
```

---

## 📊 PHASE 2 RESULTS

### 🎯 PriorityNet Training Success
```
✅ Production PriorityNet Training: COMPLETED
📊 Average Ranking Correlation: 0.6050 ± 0.6747
📊 Average Top-K Precision: 0.9660 ± 0.1049  
📊 Average Priority Accuracy: 0.9458 ± 0.0257
```

---

## 🏆 Performance Metrics

### 🎯 Phase 3B: Subtraction Efficiency
* **Mean Efficiency**: η = **81.1%** on validation & independent verification
* **Stability**: Variation ~0.000–0.001 across 5 seeds (extremely consistent)
* **Training Progress**: 1-2% → 60-70% → **81.1%** by epoch 25-30

### 🧠 Phase 3A: Neural PE Accuracy  
* **Validation Accuracy**: APE ≈ **0.582 ± 0.087** (contaminated data)
* **Training Accuracy**: APE ≈ **0.802 ± 0.012** (clean data)
* **Impact**: Uncertainty-driven adaptation improves η by **7-10%** absolute

### ✅ Phase 3C: System Validation
* **System Success Rate**: **82.1%** (τPE = 0.5, τsub = 0.3)
* **Multi-seed Verification**: 5 seeds × 200 samples each
* **Metric Stability**: All deltas < 0.05 (PE: -0.5%, Sub: 0.0%, System: +0.4%)
* **Statistical Significance**: Cohen's d > 2.0 vs. baseline methods

### ⚡ Runtime Performance
* **End-to-end Latency**: **0.156s** per 4s segment (dual channel)
* **Throughput**: **25.6 segments/second**
* **Model Sizes**: Neural PE ~3M params, Subtractor ~39M params
* **Memory**: Fits in **8GB VRAM** for batch inference

---

## 📁 Project Structure

```
🌊 AHSD/
├── 📁 configs/               # Configuration files
│   ├── 📄 data_config.yaml            # Data generation settings
│   ├── 📄 enhanced_training.yaml      # Training
│   ├── 📄 experiment_config.yaml      # Experiment
│   ├── 📄 model_config.yaml           # Model config
├── 📁 data/                  # Data adapters and loaders
├── 📁 experiments/           # Main experiment scripts
│   ├── 🐍 phase2_priority_net.py
│   ├── 🐍 phase3a_adaptive_subtractor.py
│   ├── 🐍 phase3b_neural_pe.py
│   ├── 🐍 phase3c_validation.py
├── 📁 outputs/               # Generated outputs and checkpoints
├── 📄 requirements.txt       # Python dependencies
└── 📖 README.md
```

---

## 🎯 Key Notes

* **🎛️ Configuration-Driven**: All parameters controlled via `configs/*.yaml`
* **🌍 Real Data Ready**: Implement custom readers in `data/` directory  
* **🔄 Reproducible**: Seeded runs with comprehensive logging

---

## 🚀 Getting Started

1. **Clone & Setup**: Follow installation instructions
2. **Generate Data**: Use default configs or customize as needed
3. **Train Pipeline**: Run Phase 1 → Phase 2 → Phase 3
4. **Validate**: Execute comprehensive verification suite
5. **Deploy**: Use single-segment inference for real-time processing

---

*Built for the next generation of gravitational wave astronomy* 🌌