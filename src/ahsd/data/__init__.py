"""
AHSD Data Module
===============
Complete gravitational wave dataset generation and management for AHSD pipeline.

Main Components:
- Dataset generation with overlapping signals
- PSD management and noise generation
- Waveform injection and parameter sampling
- GWTC catalog integration
- Preprocessing and quality validation
"""

from .dataset_generator import GWDatasetGenerator
from .parameter_sampler import ParameterSampler
from .psd_manager import PSDManager
from .waveform_generator import WaveformGenerator
from .injection import SignalInjector
from .preprocessing import DataPreprocessor
from .gwtc_loader import GWTCLoader
from .noise_generator import NoiseGenerator, RealNoiseGenerator

__all__ = [
    'GWDatasetGenerator',
    'ParameterSampler',
    'PSDManager',
    'WaveformGenerator',
    'SignalInjector',
    'DataPreprocessor',
    'GWTCLoader',
    'NoiseGenerator',
    'RealNoiseGenerator'
]

__version__ = '1.0.0'
