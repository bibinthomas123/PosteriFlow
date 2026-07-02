"""
AHSD Data Module
================
Gravitational-wave dataset generation using bilby for all physics.
"""

from .dataset_generator import GWDatasetGenerator, PARAM_NAMES, DETECTORS
from .parameter_sampler import ParameterSampler
from .bilby_pipeline import (
    BilbyWaveformGenerator as WaveformGenerator,
    BilbyNoiseGenerator as NoiseGenerator,
    BilbySignalInjector as SignalInjector,
    BilbyPreprocessor as DataPreprocessor,
    get_default_psd,
)
from . import snr_utils

# Optional: GWTC catalog loader (requires extra deps)
try:
    from .gwtc_loader import GWTCLoader
except ImportError:
    GWTCLoader = None

__all__ = [
    "GWDatasetGenerator",
    "PARAM_NAMES",
    "DETECTORS",
    "ParameterSampler",
    "WaveformGenerator",
    "NoiseGenerator",
    "SignalInjector",
    "DataPreprocessor",
    "get_default_psd",
    "GWTCLoader",
    "snr_utils",
]

__version__ = "2.0.0"
