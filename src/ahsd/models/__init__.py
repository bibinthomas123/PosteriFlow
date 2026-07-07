"""
Machine learning models for parameter estimation and control.

OverlapNeuralPE was removed 2026-07-07: its context encoder was measured to
produce a constant (event-independent) context — a pure marginal fit (see
analysis/context_conditioning_test.json). LeanNPE is its replacement; the
old code remains in git history.
"""

from .transformer_encoder import TransformerStrainEncoder
from .parameter_scalers import ParameterScaler, TorchParameterScaler
from .lean_npe import LeanNPE, LeanStrainEncoder, ParamScaler

__all__ = [
    "TransformerStrainEncoder",
    "LeanNPE",
    "LeanStrainEncoder",
    "ParamScaler",
    "ParameterScaler",
    "TorchParameterScaler",
]
