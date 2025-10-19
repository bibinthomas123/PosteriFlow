# AHSD: Neural Parameter Estimation for Overlapping Gravitational Waves

**Fast, accurate parameter estimation for multiple gravitational wave signals using deep learning**

---

## What We're Doing

When LIGO/Virgo detects gravitational waves, we need to figure out what caused them - things like the masses of black holes, how far away they are, and where in the sky they came from. 

**The Problem:** Sometimes multiple gravitational wave signals arrive at the same time and overlap. Traditional methods can only handle one signal at a time, taking hours or days to analyze.

**Our Solution:** A two-stage neural network system that:
1. **Ranks** overlapping signals by importance (PriorityNet)
2. **Extracts** parameters for each signal in order (OverlapNeuralPE)

**Result:** Analysis in under 1 second instead of hours, while handling overlaps that traditional methods can't.

---

## What We Used

### Core Technologies
- **Python 3.9** - Programming language
- **PyTorch 2.0** - Deep learning framework
- **PyCBC** - Gravitational wave waveform generation

### Neural Network Components
- **PriorityNet** (~2.5M parameters) - Ranks which signal to extract first
- **Normalizing Flows** (~7.5M parameters) - Learns posterior distributions
- **Reinforcement Learning** - Adapts model complexity on-the-fly
- **Bias Correction** (~0.5M parameters) - Removes systematic errors

### Training Data
- **30,000 synthetic gravitational wave events**
- Mix of binary black holes (BBH), binary neutron stars (BNS), and NSBH
- 30% have overlapping signals (2-5 concurrent events)

---

# Quick Start 

## 1. Clone the Repository

```bash
git clone https://github.com/bibinthomas123/Posteriflow.git
cd Posteriflow

```
2. Create the Conda Environment
the repository includes an environment.yaml file, you can set up the environment directly using:

```bash
conda env create -f environment.yaml
```

3. Activate the Environment
```bash
conda activate ahsd
```

**System Requirements:**
- 16GB RAM (minimum)
- GPU optional but recommended (8GB VRAM)
- 50GB free disk space

---

### 2. Generate Training Data (4-6 hours)

Generate 30,000 training samples
```bash
python experiments/generate_dataset.py
--n_samples 30000
--output data/training_30k/
```

This creates:
- 24,000 training samples
- 3,000 validation samples
- 3,000 test samples

---

### 3. Train Models (10-15 hours total)

**Step 1: Train PriorityNet (2-3 hours)**
```bash
python experiments/phase1_priority_net.py
--data_dir data/training_30k/
--output_dir models/priority_net/
--epochs 100
```


**Step 2: Train OverlapNeuralPE (8-12 hours)**
```bash
python experiments/phase3_neural_pe.py
--data_dir data/training_30k/
--priority_net models/priority_net/priority_net_best.pth
--output_dir models/neuralpe/
--epochs 200
```


---

### 4. Analyze Gravitational Waves (<1 second)

Analyze a single event
```bash
python scripts/analyze_event.py
--event GW150914
--priority_net models/priority_net/priority_net_best.pth
--neuralpe models/neuralpe/best_model.pth
--output results/GW150914/
```

**Output:**
- Parameter estimates (masses, distance, sky location)
- Full posterior distributions
- Corner plots
- Waveform reconstructions

---

## How It Works

```
Step 1: Detect overlapping signals
↓
Step 2: PriorityNet ranks them by importance
[Signal A: priority 0.85, Signal B: priority 0.42]
↓
Step 3: Extract Signal A first (higher priority)
→ Estimate parameters (mass1=35.6, mass2=29.8, ...)
↓
Step 4: Subtract Signal A from data
↓
Step 5: Extract Signal B from residual
→ Estimate parameters
↓
Step 6: Done! Both signals analyzed
```
## Citation
```
@article{yourlastname2025ahsd,
title={Neural Parameter Estimation for Overlapping Gravitational Waves},
author={Your Name},
year={2025}
}
```

**Questions?** Start with the [Quick Start](#quick-start) section above, or check `docs/` for a detailed walkthrough.

**Last Updated:** October 2025