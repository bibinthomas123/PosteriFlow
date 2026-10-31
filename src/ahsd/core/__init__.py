"""
Core components for overlapping-signal analysis.

PriorityNet ranks candidate signals for the overlap path
(ahsd.inference.ranking / infer_overlapping); OutputCalibrator calibrates its
scores. Parameter estimation itself is LeanNPE (ahsd.models.lean_npe), whose
rank conditioning is the ranking actually relied on for posteriors.
"""

from .priority_net import PriorityNet, SignalFeatureExtractor, PriorityNetTrainer

__all__ = ["PriorityNet", "SignalFeatureExtractor", "PriorityNetTrainer"]
