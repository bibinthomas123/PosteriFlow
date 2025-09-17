"""
AHSD: Adaptive Hierarchical Signal Decomposition

A deep learning pipeline for analyzing overlapping gravitational wave signals
using real LIGO-Virgo-KAGRA data.
"""

__version__ = "1.0.0"
__author__ = "AHSD Team"
__email__ = "ahsd@example.com"

from .core.ahsd_pipeline import AHSDPipeline
from .utils.config import AHSDConfig
from .data.gwtc_loader import GWTCDataLoader

__all__ = [
    "AHSDPipeline",
    "AHSDConfig", 
    "GWTCDataLoader"
]
