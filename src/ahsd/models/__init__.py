"""
Machine learning models for parameter estimation and control.

OverlapNeuralPE was removed 2026-07-07: its context encoder was measured to
produce a constant (event-independent) context — a pure marginal fit (see
analysis/context_conditioning_test.json). LeanNPE is its replacement; the
old code remains in git history. Its ParameterScaler / TorchParameterScaler
(parameter_scalers.py) went with it on 2026-10-31; LeanNPE uses ParamScaler.
"""

from .transformer_encoder import TransformerStrainEncoder
from .lean_npe import LeanNPE, LeanStrainEncoder, ParamScaler

__all__ = [
    "TransformerStrainEncoder",
    "LeanNPE",
    "LeanStrainEncoder",
    "ParamScaler",
]
