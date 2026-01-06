"""
Utility functions and configurations.
"""

from .config import AHSDConfig, DetectorConfig, WaveformConfig, PriorityNetConfig
from .logging import setup_logging
from .config_loader import load_enhanced_config, validate_config, log_config, get_config_value, ConfigDict
from .universal_config import (
    UniversalConfigReader,
    load_config,
    get_config_value as get_universal_config_value,
    validate_config as validate_universal_config,
    get_reader,
)

# Delay waveforms import since it depends on bilby which has dependency issues
def __getattr__(name):
    if name == 'WaveformUtilities':
        from .waveforms import WaveformUtilities
        return WaveformUtilities
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AHSDConfig",
    "DetectorConfig",
    "WaveformConfig", 
    "PriorityNetConfig",
    "setup_logging",
    "load_enhanced_config",
    "validate_config",
    "log_config",
    "get_config_value",
    "ConfigDict",
    "WaveformUtilities",
    # New universal config API
    "UniversalConfigReader",
    "load_config",
    "get_universal_config_value",
    "validate_universal_config",
    "get_reader",
]
