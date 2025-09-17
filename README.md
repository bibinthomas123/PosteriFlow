# PosteriFlow 

# PosteriFlow: Adaptive Hierarchical Signal Decomposition

A deep learning pipeline for analyzing overlapping gravitational wave signals using real LIGO-Virgo-KAGRA data.

## Installation

1. **Create conda environment:**
conda create -n ahsd python=3.11
conda activate ahsd

text

2. **Install dependencies:**
Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

Install gravitational wave packages
pip install gwpy pycbc lalsuite bilby

Install ML packages
pip install nflows scikit-learn xgboost

Install utilities
pip install hydra-core wandb tqdm pytest seaborn h5py astropy

Install AHSD package
pip install -e .

text

## Quick Start with Real Data

1. **Run the complete real data pipeline:**
python experiments/real_data_pipeline.py
--config configs/experiment_config.yaml
--output_dir results/real_data_run
--n_scenarios 100
--use_wandb
--verbose

text

2. **Analyze results:**
import pickle
with open('results/real_data_run/evaluation_results.pkl', 'rb') as f:
results = pickle.load(f)
print(f"Mean parameter bias: {results['summary_metrics']['ahsd']['mean_bias']:.3f}")

text

## Pipeline Components

- **PriorityNet**: Neural ranking of signal extraction order
- **Adaptive Subtractor**: Fast parameter estimation with uncertainty quantification  
- **Bias Corrector**: ML-based correction of hierarchical biases
- **Real Data Integration**: Direct use of GWTC-4.0 catalog and strain data

## Performance

On GWTC-4.0 test scenarios:
- Parameter bias reduction: ~40% vs standard methods
- Processing speed: ~5x faster than joint analysis
- Recovery rate: >90% for SNR > 10 signals

## Citation

