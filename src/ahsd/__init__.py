"""
AHSD: Adaptive Hierarchical Signal Decomposition

A deep learning pipeline for analyzing overlapping gravitational wave signals
using real LIGO-Virgo-KAGRA data.
"""

"""AHSD Package initialization."""

__version__ = "1.0.0"

# Make sure the package can be imported
try:
    from .utils.config import AHSDConfig
    from .utils.data_format import standardize_strain_data
    from .core.priority_net import PriorityNet
    from .core.adaptive_subtractor import AdaptiveSubtractor
    from .core.bias_corrector import BiasCorrector
    from .core.ahsd_pipeline import AHSDPipeline
    
    __all__ = [
        'AHSDConfig',
        'standardize_strain_data', 
        'PriorityNet',
        'AdaptiveSubtractor',
        'BiasCorrector',
        'AHSDPipeline'
    ]
    
except ImportError as e:
    print(f"Warning: Some AHSD components could not be imported: {e}")
    __all__ = []
