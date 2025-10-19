# AHSD Data Generation Module

**Advanced Hierarchical Signal Decomposition (AHSD) Dataset Generator for Overlapping Gravitational Waves**

This module provides a comprehensive toolkit for generating realistic gravitational wave datasets with overlapping signals, designed specifically for AHSD pipeline development and validation.

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Structure](#module-structure)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Dataset Format](#dataset-format)
- [API Reference](#api-reference)
- [Citation](#citation)

---

## ✨ Features

### Core Capabilities

- **Realistic Signal Generation**: BBH, BNS, and NSBH waveforms with PyCBC and analytical fallbacks
- **Overlapping Scenarios**: 2-4 simultaneous signals with configurable time offsets
- **GWTC Integration**: Load and manipulate real gravitational wave events from GWTC-4
- **Astrophysical Realism**: 
  - Independent mass/distance sampling (decorrelated parameters)
  - Realistic tidal effects for BNS/NSBH
  - Spin-precession support
  - Cosmological redshift corrections
- **Advanced Noise Modeling**:
  - Colored Gaussian noise from detector PSDs
  - Realistic glitches (blips, whistles, scratches, wandering lines)
  - Spectral lines and violin modes
- **Quality Control**:
  - Strict SNR validation
  - Data quality checks
  - Comprehensive metadata tracking

### Technical Features

- **Multi-detector Support**: H1, L1, V1 with realistic antenna patterns
- **Flexible I/O**: HDF5, pickle, JSON formats with batch processing
- **Preprocessing Pipeline**: Whitening, bandpass filtering, edge tapering
- **Scalability**: Parallel processing and incremental saving
- **Reproducibility**: Complete parameter logging and random seed control

---

## 🚀 Installation

### Prerequisites

python >= 3.8
numpy >= 1.24.0
scipy >= 1.10.0



### Install from source

Clone the repository
git clone https://github.com/bibinthomas123/PosteriFlow.git <br>

cd PosteriFlow

Install with data generation dependencies
pip install -e .

Or with all dependencies
pip install -e .[all]



### Install PyCBC (required for high-fidelity waveforms)

Via conda (recommended)
conda install -c conda-forge pycbc lalsuite

Or via pip
pip install pycbc lalsuite



### Optional: GWpy for real data access

pip install gwpy



---

## ⚡ Quick Start

### Generate a Simple Dataset

from ahsd.data import GWDatasetGenerator

Initialize generator
generator = GWDatasetGenerator(
output_dir="data/my_dataset",
sample_rate=4096,
duration=4.0,
detectors=['H1', 'L1']
)

Generate 1000 samples
summary = generator.generate_dataset(
n_samples=1000,
overlap_fraction=0.05, # 5% overlapping signals
edge_case_fraction=0.15, # 15% edge cases
save_batch_size=100,
add_glitches=True,
preprocess=True
)

print(f"Dataset generated: {summary['output_dir']}")
print(f"Generation time: {summary['elapsed_time']:.1f}s")



### Load and Inspect Dataset

from ahsd.data.io_utils import DatasetReader

reader = DatasetReader()

Load batch
batch = reader.load_hdf5("data/my_dataset/batch_00000.h5")

print(f"Batch contains {len(batch['samples'])} samples")
print(f"First sample ID: {batch['samples']['sample_id']}")
print(f"Detectors: {batch['metadata']['detectors']}")



### Command-Line Usage

Generate dataset from command line
ahsd-generate
--n-samples 10000
--output-dir data/ahsd_overlaps
--overlap-fraction 0.1
--detectors H1 L1 V1
--add-glitches
--preprocess

Validate generated dataset
ahsd-validate --dataset-dir data/ahsd_overlaps


## 📚 Usage Examples

### Example 1: Custom Parameter Sampling

from ahsd.data import ParameterSampler

sampler = ParameterSampler()

Sample BBH parameters
bbh_params = sampler.sample_bbh_parameters(
snr_regime='high', # SNR 20-30
is_edge_case=False
)

print(f"Mass 1: {bbh_params['mass_1']:.2f} M☉")
print(f"Mass 2: {bbh_params['mass_2']:.2f} M☉")
print(f"Distance: {bbh_params['luminosity_distance']:.0f} Mpc")
print(f"Redshift: {bbh_params['redshift']:.3f}")
print(f"Target SNR: {bbh_params['target_snr']:.1f}")



### Example 2: Generate Overlapping Scenario

from ahsd.data import SignalInjector, NoiseGenerator, PSDManager

Initialize components
psd_manager = PSDManager()
psds = psd_manager.load_detector_psds(['H1', 'L1'])

noise_gen = NoiseGenerator()
injector = SignalInjector()

Create overlapping scenario
from ahsd.data.injection import SignalInjector
scenario_params = injector.create_overlapping_scenario(
n_signals=3,
snr_range=(10, 25),
overlap_window=0.5 # seconds
)

Generate noise
noise_h1 = noise_gen.generate_colored_noise(psds['H1'])

Inject overlapping signals
injected_h1, metadata = injector.inject_overlapping_signals(
noise_h1, scenario_params, 'H1', psds['H1']
)

print(f"Injected {len(scenario_params)} overlapping signals")
print(f"Time offsets: {[p['time_offset'] for p in scenario_params]}")



### Example 3: Load Real GWTC Events

from ahsd.data import GWTCLoader

loader = GWTCLoader()

Get all GWTC-4 events
events = loader.get_gwtc_events(catalog='GWTC-4')

print(f"Found {len(events)} events")
print(events[['event_name', 'mass_1_source', 'mass_2_source', 'network_snr']].head())

Create synthetic overlaps from real events
overlaps = loader.create_synthetic_overlaps(
events,
n_overlaps=100,
overlap_window=0.5
)

print(f"Created {len(overlaps)} synthetic overlapping scenarios")



### Example 4: Custom Preprocessing Pipeline

from ahsd.data import DataPreprocessor

preprocessor = DataPreprocessor(
sample_rate=4096,
duration=4.0,
f_low=20.0,
f_high=2048.0
)

Load raw strain
import numpy as np
raw_strain = np.random.randn(16384) # Example data

Preprocess
processed = preprocessor.preprocess(
raw_strain,
psd_dict=psds['H1'],
whiten=True,
bandpass=True,
remove_edges=True
)

Validate
report = preprocessor.validate_data(processed)
print(f"Validation passed: {report['passed']}")
print(f"RMS: {report['metrics']['rms']:.2e}")



### Example 5: Advanced Dataset Generation with Custom Config

from ahsd.data import GWDatasetGenerator
from ahsd.data.config import (
EVENT_TYPE_DISTRIBUTION,
SNR_DISTRIBUTION
)

Customize distributions
custom_event_dist = {
'BBH': 0.70, # More BBH
'BNS': 0.20,
'NSBH': 0.10,
'noise': 0.0 # No noise-only samples
}

custom_snr_dist = {
'weak': 0.05,
'low': 0.15,
'medium': 0.30,
'high': 0.35, # More high-SNR
'loud': 0.15
}

Override in config
import ahsd.data.config as config
config.EVENT_TYPE_DISTRIBUTION = custom_event_dist
config.SNR_DISTRIBUTION = custom_snr_dist

Generate
generator = GWDatasetGenerator(output_dir="data/high_snr_dataset")
summary = generator.generate_dataset(
n_samples=5000,
overlap_fraction=0.10,
edge_case_fraction=0.20
)



---

## ⚙️ Configuration

### Default Parameters (ahsd/data/config.py)

Acquisition
SAMPLE_RATE = 4096 # Hz
DURATION = 4.0 # seconds

Detectors
DETECTORS = ['H1', 'L1', 'V1']

SNR Ranges
SNR_RANGES = {
'weak': (8, 10),
'low': (10, 14),
'medium': (14, 20),
'high': (20, 30),
'loud': (30, 50)
}

Event Types
EVENT_TYPE_DISTRIBUTION = {
'BBH': 0.55,
'BNS': 0.25,
'NSBH': 0.15,
'noise': 0.05
}

Mass Ranges (M☉)
MASS_RANGES = {
'BBH': {'m1': (5.0, 100.0), 'm2': (5.0, 100.0)},
'BNS': {'m1': (1.0, 2.5), 'm2': (1.0, 2.5)},
'NSBH': {'m1': (3.0, 100.0), 'm2': (1.0, 2.5)}
}

Distance Ranges (Mpc)
DISTANCE_RANGES = {
'BBH': (100.0, 2000.0),
'BNS': (10.0, 300.0),
'NSBH': (20.0, 800.0)
}

Cosmology (Planck 2018)
COSMO_H0 = 67.4
COSMO_OMEGA_M = 0.315
COSMO_OMEGA_LAMBDA = 0.685



### Custom Configuration

Create custom config file: my_config.yaml
my_config.yaml
sample_rate: 8192
duration: 8.0
detectors: ['H1', 'L1', 'V1']
overlap_fraction: 0.15
event_distribution:
BBH: 0.6
BNS: 0.3
NSBH: 0.1

Load and use
import yaml
from ahsd.data import GWDatasetGenerator

with open('my_config.yaml') as f:
config = yaml.safe_load(f)

generator = GWDatasetGenerator(
output_dir="data/custom",
sample_rate=config['sample_rate'],
duration=config['duration'],
detectors=config['detectors']
)

---

## 🔧 API Reference

### Core Classes

#### `GWDatasetGenerator`

Main class for dataset generation.

generator = GWDatasetGenerator(
output_dir: str = "data/output",
sample_rate: int = 4096,
duration: float = 4.0,
detectors: List[str] = ['H1', 'L1', 'V1']
)

summary = generator.generate_dataset(
n_samples: int = 1000,
overlap_fraction: float = 0.05,
edge_case_fraction: float = 0.15,
save_batch_size: int = 100,
add_glitches: bool = True,
preprocess: bool = True
) -> Dict



#### `ParameterSampler`

Sample astrophysical parameters.

sampler = ParameterSampler()

bbh_params = sampler.sample_bbh_parameters(
snr_regime: str, # 'weak', 'low', 'medium', 'high', 'loud'
is_edge_case: bool = False
) -> Dict

bns_params = sampler.sample_bns_parameters(
snr_regime: str,
is_edge_case: bool = False
) -> Dict

nsbh_params = sampler.sample_nsbh_parameters(
snr_regime: str,
is_edge_case: bool = False
) -> Dict



#### `SignalInjector`

Inject signals with SNR control.

injector = SignalInjector(sample_rate=4096, duration=4.0)

Single injection
injected, metadata = injector.inject_signal(
noise: np.ndarray,
params: Dict,
detector_name: str,
psd_dict: Dict = None
) -> Tuple[np.ndarray, Dict]

Overlapping injections
injected, metadata_list = injector.inject_overlapping_signals(
noise: np.ndarray,
signal_params_list: List[Dict],
detector_name: str,
psd_dict: Dict = None
) -> Tuple[np.ndarray, List[Dict]]



#### `GWTCLoader`

Load real GWTC events.

loader = GWTCLoader(data_dir="data/gwtc")

events_df = loader.get_gwtc_events(
catalog: str = "GWTC-4"
) -> pd.DataFrame

strain = loader.download_strain(
event_name: str,
detector: str = 'H1',
duration: int = 4,
sample_rate: int = 4096
) -> Optional[np.ndarray]

overlaps = loader.create_synthetic_overlaps(
events_df: pd.DataFrame,
n_overlaps: int = 100,
overlap_window: float = 0.5
) -> List[Dict]



---

## 📖 Citation

If you use this dataset generation module in your research, please cite:
```
@software{posteriflow_data_2025,
author = {Thomas, Bibin},
title = {PosteriFlow Data: Overlapping Gravitational Wave Dataset Generator},
year = {2025},
url = {https://github.com/bibinthomas123/PosteriFlow},
version = {1.0.0}
}
```




## 📄 License

MIT License - see [LICENSE](../LICENSE) for details.

---

## 🐛 Known Issues & Limitations

1. **PyCBC Dependency**: Some waveform approximants require PyCBC installation
2. **Memory Usage**: Large datasets (>10,000 samples) may require incremental processing
3. **Waveform Approximants**: Not all approximants support all parameter combinations
4. **Glitch Realism**: Simplified glitch models compared to real LIGO artifacts

---

## 📞 Support

- **Issues**: https://github.com/bibinthomas123/PosteriFlow/issues
- **Email**: bibinthomas951@gmail.com
- **Documentation**: https://github.com/bibinthomas123/PosteriFlow/docs

---

**Last Updated**: October 2025  
**Version**: 1.0.0