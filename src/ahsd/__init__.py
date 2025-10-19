"""
AHSD: Adaptive Hierarchical Signal Decomposition

A deep learning pipeline for analyzing overlapping gravitational wave signals
using real LIGO-Virgo-KAGRA data.
"""

__version__ = "1.0.0"

try:
    from .utils.config import AHSDConfig
    from .utils.data_format import standardize_strain_data
    from .core.priority_net import PriorityNet
    from .core.adaptive_subtractor import AdaptiveSubtractor
    from .core.bias_corrector import BiasCorrector
    from .core.ahsd_pipeline import AHSDPipeline
    from . import data  
    
    __all__ = [
        'AHSDConfig',
        'standardize_strain_data', 
        'PriorityNet',
        'AdaptiveSubtractor',
        'BiasCorrector',
        'AHSDPipeline',
        'data'  
    ]
    
except ImportError as e:
    print(f"Warning: Some AHSD components could not be imported: {e}")
    __all__ = []
