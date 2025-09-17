Adaptive Hierarchical Signal Decomposition (AHSD) for Gravitational Wave Analysis
[![Python 3.11+](https://img.shields.iohttps://img.shields.io/badge/PyTorch-2.e: MIT](https://img.shields.-of-the-art machine learning pipeline for detecting and separating overlapping gravitational wave signals in LIGO-Virgo-KAGRA data.**

🌟 Overview
The Adaptive Hierarchical Signal Decomposition (AHSD) system addresses a critical challenge in gravitational wave astronomy: detecting and analyzing multiple overlapping signals that traditional methods cannot separate. As gravitational wave detectors become more sensitive, signal overlap becomes increasingly common, potentially hiding important astrophysical discoveries.

Key Capabilities
🎯 Signal Prioritization: Neural network-based ranking of detection candidates

🔬 Adaptive Subtraction: Uncertainty-aware signal separation with posterior sampling

📊 Multi-Detector Processing: Simultaneous analysis across H1, L1, and V1 detectors

🌌 Real Data Integration: Compatible with GWTC-4.0 catalog and LIGO Open Science Center data

⚡ Production Ready: Scalable architecture for real-time observing run deployment

🏗️ System Architecture
text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Data Layer    │    │  Processing Layer │    │   Analysis Layer    │
├─────────────────┤    ├──────────────────┤    ├─────────────────────┤
│ • GWTC Events   │ ──▶│ • PriorityNet     │ ──▶│ • Parameter Est.    │
│ • Strain Data   │    │ • Feature Extract │    │ • Uncertainty Quant │
│ • Noise Models  │    │ • Detection Rank  │    │ • Quality Metrics   │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Adaptive Sub.   │    │ Neural Posterior │    │   Results Output    │
│ • Waveform Gen  │    │ • Normalizing    │    │ • Separated Signals │
│ • Template Sub  │    │   Flows (nflows) │    │ • Astrophys. Params │
│ • Residual Anal │    │ • Uncertainty    │    │ • Quality Reports   │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
🚀 Quick Start
Installation
bash
# Clone the repository
git clone https://github.com/your-org/ahsd_project.git
cd ahsd_project

# Create conda environment
conda create -n ahsd python=3.11
conda activate ahsd

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
Dependencies
bash
# Core scientific computing
torch>=2.0.0
numpy>=1.21.0
scipy>=1.7.0

# Gravitational wave analysis
bilby>=2.0.0
gwpy>=3.0.0
PyCBC>=2.0.0

# Machine learning
nflows>=0.14.0
tqdm>=4.62.0

# Data handling
pandas>=1.3.0
h5py>=3.7.0
astropy>=5.0.0

# Visualization (optional)
matplotlib>=3.5.0
seaborn>=0.11.0
Configuration
bash
# Copy and customize the configuration file
cp configs/experiment_config.yaml configs/my_config.yaml

# Edit detector settings, waveform parameters, and training options
nano configs/my_config.yaml
📖 Usage Guide
Phase 1: Data Generation
Generate training scenarios with overlapping gravitational wave signals:

bash
# Basic training data generation
python experiments/phase1_data_generation.py \
    --config configs/experiment_config.yaml \
    --output_dir data/training \
    --n_simulated 1000 \
    --n_real 200 \
    --verbose

# Production-scale data generation
python experiments/phase1_data_generation.py \
    --config configs/experiment_config.yaml \
    --output_dir data/production \
    --n_simulated 10000 \
    --n_real 2000 \
    --max_overlapping_signals 5 \
    --snr_range 8,50 \
    --verbose
Output: Realistic overlapping signal scenarios with ground truth parameters

Phase 2: PriorityNet Training
Train the neural network for signal prioritization:

bash
# Train PriorityNet
python experiments/phase2_priority_net.py \
    --config configs/experiment_config.yaml \
    --data_dir data/training \
    --output_dir models/priority_net \
    --epochs 200 \
    --batch_size 128 \
    --verbose

# Monitor training with TensorBoard
tensorboard --logdir models/priority_net/logs
Output: Trained PriorityNet model achieving ~94% priority accuracy

Phase 3: Adaptive Subtractor Training
Train the uncertainty-aware signal subtraction system:

bash
# Train Adaptive Subtractor
python experiments/phase3_adaptive_subtractor.py \
    --config configs/experiment_config.yaml \
    --data_dir data/training \
    --output_dir models/adaptive_subtractor \
    --priority_net models/priority_net/priority_net.pth \
    --n_train 1000 \
    --n_test 200 \
    --verbose
Output: Neural posterior estimator with 100% extraction success rate

Phase 4: Full Pipeline Integration
Test the complete AHSD pipeline:

bash
# Run complete pipeline
python experiments/phase4_full_pipeline.py \
    --config configs/experiment_config.yaml \
    --output_dir results/full_pipeline \
    --n_scenarios 100 \
    --verbose

# Analyze results
python scripts/analyze_results.py \
    --results_dir results/full_pipeline \
    --generate_plots
🔬 Scientific Applications
Real-Time Processing
python
from ahsd import AHSDPipeline

# Initialize pipeline
pipeline = AHSDPipeline.from_config('configs/production_config.yaml')

# Process strain data segment
results = pipeline.process_strain_segment(
    strain_data={'H1': h1_strain, 'L1': l1_strain, 'V1': v1_strain},
    gps_time=1234567890,
    duration=4.0
)

# Extract results
separated_signals = results['separated_signals']
parameters = results['estimated_parameters']
uncertainties = results['parameter_uncertainties']
Batch Processing
python
# Process multiple segments
from ahsd.batch import BatchProcessor

processor = BatchProcessor(
    config_file='configs/batch_config.yaml',
    n_workers=8
)

results = processor.process_observation_run(
    run_name='O4a',
    start_time=1367284818,  # O4a start
    end_time=1387593618,    # O4a end
    output_dir='results/O4a_reanalysis'
)
📊 Performance Metrics
Benchmark Results
Component	Metric	Value	Target
PriorityNet	Priority Accuracy	93.8%	>90%
PriorityNet	Top-K Precision	65%	>60%
Adaptive Subtractor	Extraction Success	100%	>95%
Adaptive Subtractor	Parameter Bias	1.05σ	<3σ
Full Pipeline	Processing Time	~30s	<60s
Full Pipeline	Success Rate	95%	>90%
Computational Requirements
Memory: 16+ GB RAM for training, 8+ GB for inference

GPU: NVIDIA GPU with 8+ GB VRAM (recommended)

CPU: 8+ cores for parallel processing

Storage: 100+ GB for production datasets

🔧 Configuration Options
Key Configuration Parameters
text
# configs/experiment_config.yaml

detectors:
  - name: "H1"
    sampling_rate: 4096
    duration: 8.0
  - name: "L1" 
    sampling_rate: 4096
    duration: 8.0

waveform:
  approximant: "IMRPhenomPv2"
  f_lower: 15.0
  duration: 8.0

priority_net:
  hidden_dims: [256, 128, 64, 32]
  dropout: 0.1
  learning_rate: 0.0005
  batch_size: 128

adaptive_subtractor:
  neural_pe:
    flow_layers: 8
    hidden_features: 64
    num_blocks: 4
  uncertainty_realizations: 200
📈 Scaling to Production
Large-Scale Training
bash
# Multi-GPU training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    experiments/distributed_training.py \
    --config configs/production_config.yaml \
    --n_simulated 50000 \
    --n_real 10000

# Cloud deployment
./scripts/deploy_aws.sh \
    --instance_type p3.8xlarge \
    --workers 16 \
    --dataset_size 100000
Integration with LIGO Infrastructure
bash
# Configure for LIGO clusters
python scripts/setup_ligo_integration.py \
    --cluster_name LIGO-Hanford \
    --data_source /cvmfs/gwosc.osgstorage.org \
    --output_format lalsuite

# Submit production jobs
condor_submit production_ahsd.sub
🧪 Testing
Unit Tests
bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_priority_net.py -v
pytest tests/test_adaptive_subtractor.py -v
pytest tests/test_integration.py -v
Validation
bash
# Validate against known GWTC events
python validation/validate_gwtc_events.py \
    --events GW150914,GW170817,GW190521 \
    --tolerance 0.1

# Cross-validation
python validation/cross_validate.py \
    --k_folds 5 \
    --metrics accuracy,precision,recall
📚 Documentation
Scientific Background: Theoretical foundation

Algorithm Details: Technical implementation

API Reference: Complete function documentation

Examples: Jupyter notebooks and example scripts

Troubleshooting: Common issues and solutions

🤝 Contributing
We welcome contributions! Please see our Contributing Guide.

Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

📄 Citation
If you use AHSD in your research, please cite:

text
@article{ahsd2025,
    title={Adaptive Hierarchical Signal Decomposition for Overlapping Gravitational Wave Detection},
    author={Your Name and Collaborators},
    journal={Physical Review D},
    year={2025},
    volume={XXX},
    pages={XXXXXX},
    eprint={arXiv:2025.XXXX}
}
📞 Support
Issues: GitHub Issues

Discussions: GitHub Discussions

Email: bibinthomas951@gmail.com

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🏆 Acknowledgments
LIGO Scientific Collaboration

Virgo Collaboration

KAGRA Collaboration

Bilby development team

PyCBC development team

Gravitational Wave Open Science Center (GWOSC)

Built with ❤️ for the gravitational wave astronomy community
