"""Config dataclasses for the AHSD overlap pipeline (detectors, waveform,
PriorityNet, adaptive subtractor, bias corrector) plus their YAML loader.
Note: LeanNPE training is configured via argparse in experiments/train_lean_npe.py,
not these classes."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
import numpy as np

@dataclass
class DetectorConfig:
    """Detector settings: name, optional PSD file, sampling rate, segment duration."""
    name: str
    psd_file: Optional[str] = None
    sampling_rate: int = 4096
    duration: float = 8.0  

@dataclass
class WaveformConfig:
    """Waveform-generation settings: approximant, low/reference frequency, duration."""
    approximant: str = "IMRPhenomPv2"
    f_lower: float = 20.0
    f_ref: float = 20.0
    duration: float = 8.0  

@dataclass
class PriorityNetConfig:
    """PriorityNet architecture/training hyperparameters (layer widths, dropout,
    learning rate, batch size)."""
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    dropout: float = 0.25  
    learning_rate: float = 0.0005  
    batch_size: int = 16
