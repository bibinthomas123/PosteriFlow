"""
Machine learning models for parameter estimation and control.
"""

from .neural_pe import NeuralPosteriorEstimator
from .transformer_encoder import TransformerStrainEncoder
from .parameter_scalers import ParameterScaler, TorchParameterScaler

try:
    from .overlap_neuralpe import OverlapNeuralPE, ContextEncoder
except Exception as e:
    # Skip if overlap_neuralpe fails (depends on flows which has issues)
    import logging
    logging.warning(f"Could not import OverlapNeuralPE: {e}")
    OverlapNeuralPE = None
    ContextEncoder = None

__all__ = [
    "NeuralPosteriorEstimator",
    "TransformerStrainEncoder",
    "OverlapNeuralPE",
    "ContextEncoder",
    "ParameterScaler",
    "TorchParameterScaler",
]
