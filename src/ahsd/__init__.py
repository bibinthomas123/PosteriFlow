"""
AHSD / PosteriFlow: amortized neural posterior estimation (LeanNPE) for
overlapping compact-binary signals in LIGO-Virgo strain.
"""

__version__ = "1.0.0"

try:
    from .utils.data_format import standardize_strain_data
    from .core.priority_net import PriorityNet
    from . import data

    __all__ = [
        'standardize_strain_data',
        'PriorityNet',
        'data'
    ]
    
except ImportError as e:
    print(f"Warning: Some AHSD components could not be imported: {e}")
    __all__ = []
