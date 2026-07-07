#!/usr/bin/env python3
"""
Phase 2: PriorityNet with Integrated Dataset Loading
Reads directly from your 20K chunked dataset structure
Enhanced with train/validation/test splits and comprehensive evaluation

MPS COMPATIBILITY NOTES (Apple Silicon M4):
  - All float64 → float32 (MPS does not support float64)
  - pin_memory=False when using MPS (crashes otherwise)
  - persistent_workers=False on MPS (fork-based multiprocessing issues on macOS)
  - num_workers=0 on MPS (safest; avoids spawn/fork conflicts)
  - torch.load map_location uses get_device() helper
  - torch.where scalar arguments always constructed on correct device
  - scipy / numpy work is kept on CPU (correct—they don't use MPS)
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import yaml
from typing import List, Dict, Tuple, Any, Optional, Union, Iterator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import random
from datetime import datetime
from scipy.stats import spearmanr, kendalltau
import time
import traceback
import gc
import math
from torch.utils.data import WeightedRandomSampler

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ahsd.core.priority_net import PriorityNet, PriorityNetTrainer
from ahsd.utils.config_loader import (
    load_enhanced_config,
    validate_config,
    log_config,
    get_config_value,
)


# ============================================================================
# MPS-COMPATIBLE DEVICE HELPER
# ============================================================================

def get_device() -> torch.device:
    """
    Return the best available device in priority order:
      CUDA  →  MPS (Apple Silicon)  →  CPU

    MPS is available from PyTorch 1.12+ on macOS 12.3+ with Apple Silicon.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def is_mps_device(device: torch.device) -> bool:
    return device.type == "mps"


def dataloader_kwargs(device: torch.device) -> dict:
    """
    Return DataLoader kwargs that are safe for the target device.

    MPS restrictions:
      - pin_memory must be False  (MPS memory is unified; pinning is a no-op and
        raises a warning/error depending on PyTorch version)
      - persistent_workers=False  (macOS uses 'spawn' for multiprocessing;
        persistent workers with forked processes cause deadlocks)
      - num_workers=0             (safest default on MPS; avoids all fork/spawn
        issues with Python multiprocessing on macOS)
    """
    if is_mps_device(device):
        return {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }
    # CUDA / CPU: use faster settings
    return {
        "num_workers": 1,
        "pin_memory": True,
        "persistent_workers": True,
    }


def to_device_float32(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Move tensor to device, casting to float32.

    MPS does not support float64 (double).  Any float64 tensor passed to an
    MPS-backed op will raise RuntimeError.  Always cast before moving.
    """
    if tensor.dtype == torch.float64:
        tensor = tensor.float()  # → float32
    return tensor.to(device)


def scalar_tensor(value: float, device: torch.device) -> torch.Tensor:
    """Create a float32 scalar tensor on the correct device (for torch.where)."""
    return torch.tensor(value, dtype=torch.float32, device=device)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("phase2_priority_net_complete.log"),
            logging.StreamHandler(),
        ],
    )


# ============================================================================
# EDGE TYPE ENCODING
# ============================================================================

def encode_edge_type(dets):
    """
    Match dataset_generator.py encoding scheme.
    0  → single signal
    3  → pairwise overlap
    6  → triple overlap
    7+ → higher overlaps (4+)
    """
    if isinstance(dets, str):
        return None
    if not dets:
        return 0
    if isinstance(dets, dict):
        dets = [dets]
    size = len([d for d in dets if d is not None])
    if size == 1:
        return 0
    elif size == 2:
        return 3
    elif size == 3:
        return 6
    else:
        return 7


# ============================================================================
# PRIORITY CALCULATION  (CPU / numpy — no device dependency)
# ============================================================================

def calculate_priority(
    signal: Dict[str, Any], is_overlap: bool = False, time_lag_ms: float = 0.0
) -> float:
    """Enhanced priority calculation with SNR sensitivity boost."""
    target_snr = signal.get("target_snr", 0.0)
    if target_snr == 0.0:
        target_snr = signal.get("network_snr", 0.0)
    if target_snr == 0.0:
        h1_data = signal.get("H1", {})
        l1_data = signal.get("L1", {})
        if isinstance(h1_data, dict) and isinstance(l1_data, dict):
            h1_snr = float(h1_data.get("optimal_snr", 0.0))
            l1_snr = float(l1_data.get("optimal_snr", 0.0))
            target_snr = np.sqrt(h1_snr ** 2 + l1_snr ** 2)
    target_snr = float(target_snr)
    if target_snr < 8.0:
        return 0.0

    if target_snr < 12.0:
        base_priority = target_snr ** 0.9
    elif target_snr < 20.0:
        base_priority = target_snr ** 1.15
    elif target_snr < 35.0:
        base_priority = target_snr ** 1.25
    else:
        base_priority = target_snr ** 1.1

    edge_bonus = 0.0
    snr_context = np.clip(target_snr / 30.0, 0.0, 1.0)

    try:
        mass_1 = float(signal.get("mass_1", 10.0))
        mass_2 = float(signal.get("mass_2", 10.0))
        if mass_1 > 0 and mass_2 > 0:
            q = max(mass_1, mass_2) / min(mass_1, mass_2)
            if q > 8.0:
                edge_bonus += 0.10 * snr_context
            elif q > 4.0:
                edge_bonus += 0.05 * snr_context
    except Exception:
        pass

    try:
        a1 = abs(float(signal.get("a1", signal.get("a_1", 0.0))))
        a2 = abs(float(signal.get("a2", signal.get("a_2", 0.0))))
        a_max = max(a1, a2)
        if a_max > 0.8:
            edge_bonus += 0.12 * snr_context
        elif a_max > 0.5:
            edge_bonus += 0.06 * snr_context
    except Exception:
        pass

    try:
        eccentricity = float(signal.get("eccentricity", 0.0))
        if eccentricity > 0.1:
            edge_bonus += 0.15 * snr_context
    except Exception:
        pass

    edge_bonus = min(edge_bonus, 0.40)

    if is_overlap:
        if time_lag_ms < 500:
            overlap_penalty = 0.70
        elif time_lag_ms < 1000:
            overlap_penalty = 0.82
        else:
            overlap_penalty = 0.95
    else:
        overlap_penalty = 1.0

    distance_bonus = 1.0
    try:
        luminosity_distance = float(signal.get("luminosity_distance", 500.0))
        if luminosity_distance < 100:
            distance_bonus = 1.18
        elif luminosity_distance < 200:
            distance_bonus = 1.10
        elif luminosity_distance < 400:
            distance_bonus = 1.04
    except Exception:
        pass

    return float(base_priority * (1.0 + edge_bonus) * overlap_penalty * distance_bonus)


def estimate_scenario_difficulty(priorities: np.ndarray) -> float:
    if len(priorities) < 2:
        return 1.0
    prios_sorted = np.sort(priorities)[::-1]
    gap = float(prios_sorted[0] - prios_sorted[1])
    difficulty = 1.0 / np.clip(gap + 0.1, 0.1, 2.0)
    return float(np.clip(difficulty, 0.5, 2.0))


def create_weighted_sampler(
    dataset,
    n_signals_threshold: int = 5,
    oversample_factor: float = 1.35,
    use_difficulty: bool = True,
):
    weights = []
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            n = len(sample.get("detections", []))
            weight = oversample_factor if n >= n_signals_threshold else 1.0
            if use_difficulty and n >= 2:
                priorities_raw = sample.get("priorities", [])
                if isinstance(priorities_raw, np.ndarray):
                    priorities = priorities_raw.astype(np.float32)
                else:
                    priorities = np.array(list(priorities_raw), dtype=np.float32)
                if len(priorities) >= 2:
                    difficulty = estimate_scenario_difficulty(priorities)
                    weight *= difficulty
            weights.append(weight)
        except Exception as e:
            logging.debug(f"Weight calculation failed at {i}: {e}")
            weights.append(1.0)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


# ============================================================================
# CHUNKED DATA LOADER
# ============================================================================

class ChunkedGWDataLoader:
    """Data loader for 20K chunked GW dataset structure."""

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.max_samples = max_samples
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self._load_split_info()
        self.last_processed = 0

    @property
    def count(self) -> int:
        if not hasattr(self, "_count_cache"):
            self._count_cache = self.count_samples(ignore_max=False)
        return self._count_cache

    def count_samples(self, ignore_max: bool = False, cap: Optional[int] = None) -> int:
        old_max = self.max_samples
        if ignore_max:
            self.max_samples = None
        cnt = 0
        for _ in self.iter_all_samples():
            cnt += 1
            if cap is not None and cnt >= cap:
                break
        self.max_samples = old_max
        return cnt

    def _load_split_info(self):
        split_dir = self.dataset_path / self.split
        # Also try common abbreviations: "validation" → "val", "val" → "validation"
        if not split_dir.exists():
            alternates = {"validation": "val", "val": "validation"}
            alt = alternates.get(self.split)
            if alt and (self.dataset_path / alt).exists():
                self.split = alt
                split_dir = self.dataset_path / self.split
            else:
                raise FileNotFoundError(f"Split directory not found: {split_dir}")
        split_info_file = split_dir / "split_info.json"
        if split_info_file.exists():
            with open(split_info_file, "r") as f:
                self.split_info = json.load(f)
        else:
            # Support both chunk_*.pkl (legacy) and batch_*.pkl (current generator)
            chunk_files = sorted(split_dir.glob("chunk_*.pkl"))
            batch_files = sorted(split_dir.glob("batch_*.pkl"))
            if batch_files and not chunk_files:
                self._batch_files = batch_files
                self.split_info = {
                    "n_chunks": len(batch_files),
                    "chunk_size": 100,
                    "file_pattern": "batch_XXXX.pkl",
                }
            else:
                self._batch_files = None
                self.split_info = {
                    "n_chunks": len(chunk_files),
                    "chunk_size": 500,
                    "file_pattern": "chunk_XXXX.pkl",
                }
        self.n_chunks = self.split_info["n_chunks"]
        self.chunk_size = self.split_info["chunk_size"]
        self.logger.info(
            f" {self.split}: {self.n_chunks} chunks, {self.chunk_size} samples per chunk"
        )

    @staticmethod
    def _extract_samples(chunk_data) -> list:
        """Handle both list-of-samples and {'samples': [...]} dict formats."""
        if isinstance(chunk_data, dict):
            return chunk_data.get("samples", [])
        return list(chunk_data)

    def iter_all_samples(self) -> Iterator[Dict]:
        total_yielded = 0
        self.logger.info(f"Loading {self.split} chunks (streaming)...")
        split_dir = self.dataset_path / self.split
        # Use batch_files list if detected, otherwise build chunk file names
        files = getattr(self, "_batch_files", None)
        if files is None:
            files = [split_dir / f"chunk_{i:04d}.pkl" for i in range(self.n_chunks)]
        for chunk_file in files:
            if not chunk_file.exists():
                self.logger.warning(f"Chunk file not found: {chunk_file}")
                continue
            try:
                with open(chunk_file, "rb") as f:
                    chunk_data = pickle.load(f)
                for sample in self._extract_samples(chunk_data):
                    yield sample
                    total_yielded += 1
                    if self.max_samples and total_yielded >= self.max_samples:
                        self.logger.info(f"Reached max samples limit: {self.max_samples}")
                        return
                del chunk_data
            except Exception as e:
                self.logger.warning(f"Failed to load chunk {chunk_file.name}: {e}")
                continue

    def _convert_noise_sample_to_scenario(self, sample: Dict) -> Optional[Dict]:
        try:
            metadata = sample.get("metadata", {})
            noise_param = {
                "mass_1": 0.0, "mass_2": 0.0, "luminosity_distance": 0.0,
                "network_snr": 0.0,
                "individual_snrs": {
                    det: 0.0 for det in metadata.get("detector_network", ["H1", "L1"])
                },
                "ra": 0.0, "dec": 0.0, "theta_jn": 0.0, "psi": 0.0,
                "phase": 0.0, "geocent_time": 0.0, "a1": 0.0, "a2": 0.0,
                "approximant": "noise", "event_type": "noise",
                "edge_case": False, "edge_case_type": None,
                "detector_network": metadata.get("detector_network", ["H1", "L1"]),
                "sample_id": sample.get("sample_id", "unknown"),
                "noise_type": metadata.get("noise_type", "gaussian"),
                "glitch_present": metadata.get("glitch_present", False),
            }
            return {
                "scenario_id": sample.get("sample_id", "unknown"),
                "true_parameters": [noise_param],
                "baseline_biases": [],
                "detector_data": sample.get("detector_data", {}),
                "whitened_data": sample.get("whitened_data", {}),
                "metadata": metadata,
            }
        except Exception as e:
            self.logger.debug(f"Failed to convert noise sample: {e}")
            return None

    def convert_to_priority_scenarios(
        self,
        limit: Optional[int] = None,
        create_overlaps: bool = False,
        overlap_probability: float = 0.3,
        reservoir_max: int = 2000,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        assert 0.0 <= overlap_probability <= 1.0
        scenarios: List[Dict] = []
        failed_count = 0
        success_count = 0
        singles_seen = 0
        artificial_created = 0
        reservoir: List[Dict] = []
        rng = random.Random(seed) if seed is not None else random

        def maybe_add_to_reservoir(sample: Dict):
            nonlocal reservoir, singles_seen
            if len(reservoir) < reservoir_max:
                reservoir.append(sample)
            else:
                j = rng.randint(0, singles_seen)
                if j < reservoir_max:
                    reservoir[j] = sample

        iterator = self.iter_all_samples()
        processed = 0
        pbar = tqdm(
            iterator, desc="Converting (streaming)", disable=not self.verbose, total=limit
        )
        for sample in pbar:
            if limit is not None and processed >= limit:
                break
            processed += 1
            if not isinstance(sample, Dict):
                failed_count += 1
                continue
            if sample.get("type") == "overlap":
                scenario = self._convert_overlap_sample_to_scenario(sample)
                if scenario is not None:
                    scenarios.append(scenario)
                    success_count += 1
                else:
                    failed_count += 1
            else:
                scenario = self._convert_single_sample_to_scenario(sample)
                singles_seen += 1 if scenario is not None else 0
                if scenario is not None:
                    scenarios.append(scenario)
                    success_count += 1
                    maybe_add_to_reservoir(sample)
                else:
                    failed_count += 1
            if create_overlaps and overlap_probability > 0.0 and len(reservoir) >= 2:
                target = int(singles_seen * overlap_probability)
                while artificial_created < target:
                    art = self._create_artificial_overlap_scenario(reservoir)
                    if art is not None:
                        scenarios.append(art)
                        success_count += 1
                    artificial_created += 1

        print(f"Total processed: {processed}")
        print(f"Singles seen: {singles_seen}")
        print(f"Artificial overlaps created: {artificial_created}")
        print(f"\nConversion complete: Success={success_count}, Failed={failed_count}")
        if (success_count + failed_count) > 0:
            print(
                f"  Success rate: {success_count/(success_count+failed_count)*100:.1f}%"
            )
        if processed > 0 and failed_count > 0.5 * processed:
            print(f"\n⚠️  WARNING: High failure rate ({failed_count}/{processed})")
        self.proccessed = processed
        return scenarios

    def _convert_multi_sample_to_scenario(self, sample: Dict) -> Optional[Dict]:
        try:
            metadata = sample.get("metadata", {})
            signal_parameters = metadata.get("signal_parameters", [])
            if len(signal_parameters) < 2:
                if len(signal_parameters) == 1:
                    return self._convert_single_sample_to_scenario(sample)
                return None
            return self._convert_overlap_sample_to_scenario(sample)
        except Exception as e:
            self.logger.debug(f"Failed to convert multi-sample: {e}")
            return None

    def _extract_priorities_from_dataset(
        self, sample: Dict, signal_params: List[Dict]
    ) -> Optional[List[float]]:
        p = sample.get("priorities")
        if isinstance(p, (int, float, np.floating)):
            p = [float(p)]
        if isinstance(p, list) and all(isinstance(v, (int, float, np.floating)) for v in p):
            if len(p) == len(signal_params):
                return [float(v) for v in p]
            if len(signal_params) == 1 and len(p) >= 1:
                return [float(p[0])]
        keys = ["priority", "target_priority", "label_priority", "priority_score"]
        vals = []
        for par in signal_params:
            if not isinstance(par, dict):
                return None
            v = next((par[k] for k in keys if k in par), None)
            if not isinstance(v, (int, float, np.floating)):
                return None
            vals.append(float(v))
        return vals if len(vals) == len(signal_params) else None

    def _resolve_edge_type(self, sample: Dict) -> tuple:
        md = sample.get("metadata") or {}
        edge_type_id = sample.get("edge_type_id", md.get("edge_type_id"))
        if edge_type_id is not None:
            try:
                return None, int(edge_type_id)
            except Exception:
                pass
        signal_params = md.get("signal_parameters", sample.get("parameters", []))
        if isinstance(signal_params, dict):
            signal_params = [signal_params]
        if not signal_params:
            return None, 0
        n_signals = len([p for p in signal_params if p is not None])
        if n_signals == 1:
            return None, 0
        elif n_signals == 2:
            return None, 3
        elif n_signals == 3:
            return None, 6
        else:
            return None, 7

    def _convert_single_sample_to_scenario(self, sample: Dict) -> Optional[Dict]:
        if not isinstance(sample, dict):
            return None
        sample_type = sample.get("type", "unknown")
        if hasattr(sample_type, "item"):
            sample_type = sample_type.item()
        sample_type = str(sample_type)
        if sample_type == "noise":
            return None
        if sample.get("is_overlap") or sample_type == "overlap":
            signal_params = (sample.get("metadata") or {}).get("signal_parameters")
        else:
            signal_params = sample.get("parameters")
            if signal_params is None:
                signal_params = (sample.get("metadata") or {}).get("signal_parameters")
        if signal_params is None:
            self._missing_params_count = getattr(self, "_missing_params_count", 0) + 1
            if self._missing_params_count in (1,) or self._missing_params_count % 1000 == 0:
                self.logger.warning(
                    f"Missing signal_parameters (count: {self._missing_params_count}). "
                    f"Sample type: {sample_type}"
                )
            return None
        if isinstance(signal_params, dict):
            signal_params = [signal_params]
        elif not isinstance(signal_params, list):
            return None
        try:
            priorities = self._extract_priorities_from_dataset(sample, signal_params)
            if priorities is None:
                return None
            detections = list(signal_params)
            edge_case_type, edge_type_id = self._resolve_edge_type(sample)
            if edge_type_id is None:
                edge_type_id = encode_edge_type(detections)
            return {
                "detections": detections,
                # Store as float32 tensor immediately — avoids float64 issues on MPS
                "priorities": torch.tensor(priorities, dtype=torch.float32),
                "sample_type": sample_type,
                "sample_id": sample.get("sample_id", sample.get("id", "unknown")),
                "is_edge_case": sample.get("is_edge_case", False),
                "edge_case_type": edge_case_type,
                "edge_type_id": int(edge_type_id),
                "detector_data": sample.get("detector_data", {}),
            }
        except Exception as e:
            self._conversion_error_count = getattr(self, "_conversion_error_count", 0) + 1
            if self._conversion_error_count <= 10:
                self.logger.debug(f"Conversion error: {e}")
            return None

    def _convert_overlap_sample_to_scenario(self, sample: Dict) -> Optional[Dict]:
        if not isinstance(sample, dict):
            return None
        metadata = sample.get("metadata", {}) or {}
        signal_params = metadata.get("signal_parameters", [])
        if not signal_params:
            signal_params = sample.get("parameters", [])
        if isinstance(signal_params, dict):
            signal_params = [signal_params]
        if not signal_params:
            return None
        try:
            priorities = self._extract_priorities_from_dataset(sample, signal_params)
            if priorities is None:
                return None
            detections = list(signal_params)
            edge_case_type, edge_type_id = self._resolve_edge_type(sample)
            if edge_type_id is None:
                edge_type_id = encode_edge_type(detections)
            return {
                "detections": detections,
                "priorities": torch.tensor(priorities, dtype=torch.float32),
                "sample_type": "overlap",
                "sample_id": sample.get("sample_id", sample.get("id", "unknown")),
                "is_edge_case": sample.get("is_edge_case", False),
                "edge_case_type": edge_case_type,
                "edge_type_id": int(edge_type_id),
                "detector_data": sample.get("detector_data", {}),
            }
        except Exception as e:
            self.logger.debug(f"Overlap conversion error: {e}")
            return None

    def _create_artificial_overlap_scenario(self, single_samples: List[Dict]) -> Optional[Dict]:
        try:
            n_signals = random.choice([2, 3])
            if len(single_samples) < n_signals:
                return None
            selected = random.sample(single_samples, n_signals)
            first_det = selected[0].get("detector_data", {})
            if not isinstance(first_det, dict) or not first_det:
                return None
            detectors = list(first_det.keys())
            combined_detector_data = {}
            for det in detectors:
                base = selected[0]["detector_data"][det]
                base_strain = base.get("strain") if isinstance(base, dict) else base
                if base_strain is None:
                    return None
                base_strain = np.asarray(base_strain, dtype=np.float32)  # float32!
                combined_detector_data[det] = {
                    "strain": np.zeros_like(base_strain, dtype=np.float32)
                }
            detections = []
            for i, s in enumerate(selected):
                md = s.get("metadata", {}) or {}
                sp = md.get("signal_parameters") or s.get("parameters")
                if sp is None:
                    continue
                if isinstance(sp, dict):
                    sp = [sp]
                if not isinstance(sp, list) or len(sp) == 0:
                    continue
                sig = sp[0]
                snr_reduction = random.uniform(0.6, 0.8)
                for det in detectors:
                    det_blob = s.get("detector_data", {}).get(det)
                    if det_blob is None:
                        continue
                    det_strain = det_blob.get("strain") if isinstance(det_blob, dict) else det_blob
                    if det_strain is None:
                        continue
                    det_strain = np.asarray(det_strain, dtype=np.float32)
                    m = min(len(combined_detector_data[det]["strain"]), len(det_strain))
                    if m <= 0:
                        continue
                    combined_detector_data[det]["strain"][:m] += det_strain[:m] * snr_reduction
                det_params = {
                    "mass_1": sig.get("mass_1", 30.0),
                    "mass_2": sig.get("mass_2", 25.0),
                    "luminosity_distance": sig.get("luminosity_distance", 500.0),
                    "network_snr": float(md.get("network_snr", sig.get("network_snr", 10.0)))
                    * snr_reduction,
                    "ra": sig.get("ra", 0.0), "dec": sig.get("dec", 0.0),
                    "theta_jn": sig.get("theta_jn", 0.0), "psi": sig.get("psi", 0.0),
                    "phase": sig.get("phase", 0.0),
                    "geocent_time": i * random.uniform(0.2, 1.0),
                    "a1": sig.get("a1", 0.0), "a2": sig.get("a2", 0.0),
                    "approximant": sig.get("approximant", "IMRPhenomD"),
                    "event_type": (
                        s.get("event_type") or md.get("event_type")
                        or sig.get("event_type") or "BBH"
                    ),
                    "detector_network": detectors,
                    "is_overlap": True, "artificial": True,
                }
                detections.append(det_params)
            if not detections:
                return None
            priorities = [calculate_priority(p, is_overlap=True) for p in detections]
            scenario_id = f"artificial_overlap_{random.randint(1000, 9999)}"
            n_det = len(detections)
            edge_type_id = 3 if n_det == 2 else (6 if n_det == 3 else 7)
            return {
                "detections": detections,
                "priorities": torch.tensor(priorities, dtype=torch.float32),
                "sample_type": "overlap",
                "sample_id": scenario_id,
                "is_edge_case": False,
                "edge_case_type": None,
                "edge_type_id": edge_type_id,
                "detector_data": combined_detector_data,
                "metadata": {
                    "event_type": "overlap", "n_signals": n_det,
                    "overlap_type": "multi_signal", "detector_network": detectors,
                    "artificial": True, "scenario_type": "artificial_overlap",
                },
            }
        except Exception as e:
            self.logger.debug(f"Failed to create artificial overlap: {e}")
            return None


# ============================================================================
# DATASET
# ============================================================================

class PriorityNetDataset(Dataset):
    """Enhanced PriorityNet dataset with signal-type awareness."""

    def __init__(self, scenarios: List[Dict], split_name: str = "train"):
        self.data = []
        self.split_name = split_name
        self.normalize_priorities = True
        self.logger = logging.getLogger(__name__)
        # Initialise stats so _normalize_priorities never reads uninitialised attrs
        self.priority_min = 0.0
        self.priority_max = 100.0

        bbh_count = bns_count = nsbh_count = noise_count = overlap_count = 0
        self.logger.info(
            f"📊 Processing {len(scenarios)} scenarios for {split_name} dataset..."
        )

        for scenario_id, scenario in enumerate(scenarios):
            try:
                detections = scenario.get("detections", [])
                priorities = scenario.get("priorities", None)
                if not detections or priorities is None:
                    if scenario_id < 5:
                        self.logger.warning(
                            f"  Scenario {scenario_id} missing data. "
                            f"Has detections: {bool(detections)}, "
                            f"Has priorities: {priorities is not None}, "
                            f"Keys: {list(scenario.keys())}"
                        )
                    continue

                # Always work in float32 —— MPS incompatible with float64
                if isinstance(priorities, torch.Tensor):
                    priorities = priorities.float()  # ensure float32
                elif isinstance(priorities, list):
                    priorities = torch.tensor(priorities, dtype=torch.float32)
                else:
                    priorities = torch.tensor(
                        np.asarray(priorities, dtype=np.float32), dtype=torch.float32
                    )

                if len(detections) != len(priorities):
                    self.logger.warning(
                        f"  Scenario {scenario_id} length mismatch: "
                        f"{len(detections)} detections vs {len(priorities)} priorities"
                    )
                    continue

                # Clean on CPU — no device ops yet
                priorities = torch.where(
                    torch.isnan(priorities), torch.tensor(0.5, dtype=torch.float32), priorities
                )
                priorities = torch.where(
                    torch.isinf(priorities), torch.tensor(1.0, dtype=torch.float32), priorities
                )

                class_ids = []
                for detection in detections:
                    evt_raw = detection.get("event_type", detection.get("type", None))
                    evt = str(evt_raw).strip().lower() if evt_raw is not None else None
                    if evt and "bbh" in evt:
                        class_id = 0; bbh_count += 1
                    elif evt and "bns" in evt:
                        class_id = 1; bns_count += 1
                    elif evt and "nsbh" in evt:
                        class_id = 2; nsbh_count += 1
                    else:
                        class_id = 3; noise_count += 1
                    class_ids.append(class_id)

                edge_type_id = scenario.get("edge_type_id", 0)

                self.data.append({
                    "scenario_id": scenario.get("sample_id", f"scenario_{scenario_id}"),
                    "detections": detections,
                    "priorities": priorities,
                    "class_ids": torch.tensor(class_ids, dtype=torch.long),
                    "edge_type_id": edge_type_id,
                    "is_edge_case": scenario.get("is_edge_case", False),
                    "edge_case_type": scenario.get("edge_case_type", None),
                    "metadata": scenario.get("metadata", {}),
                    "detector_data": scenario.get("detector_data", {}),
                })
                if len(detections) > 1:
                    overlap_count += 1
            except Exception as e:
                if scenario_id < 5:
                    self.logger.warning(f"  Error processing scenario {scenario_id}: {e}")
                continue

        total = bbh_count + bns_count + nsbh_count + noise_count
        self.logger.info(
            f"\n📈 {split_name.upper()} PriorityNet dataset created: {len(self.data)} scenarios"
        )
        if total > 0:
            self.logger.info(f"   BBH: {bbh_count} ({bbh_count/total*100:.1f}%)")
            self.logger.info(f"   BNS: {bns_count} ({bns_count/total*100:.1f}%)")
            self.logger.info(f"   NSBH: {nsbh_count} ({nsbh_count/total*100:.1f}%)")
            self.logger.info(f"   Noise: {noise_count} ({noise_count/total*100:.1f}%)")
        if len(self.data) > 0:
            self.logger.info(
                f"   Overlap: {overlap_count} ({overlap_count/len(self.data)*100:.1f}%)"
            )
        if len(self.data) == 0:
            self.logger.error(f"❌ NO VALID SCENARIOS FOR {split_name.upper()} DATASET!")

        if self.normalize_priorities:
            self._compute_priority_stats()

    def _compute_priority_stats(self):
        all_priorities = []
        for item in self.data:
            p = item.get("priorities")
            if isinstance(p, torch.Tensor):
                p = p.numpy()
            elif isinstance(p, list):
                p = np.array(p)
            if p is not None:
                all_priorities.extend([float(v) for v in p if v > 0])
        arr = np.array(all_priorities, dtype=np.float32)
        if len(arr) > 0:
            self.priority_min = float(arr.min())
            self.priority_max = float(arr.max())
            self.logger.info(f"📊 Priority stats ({self.split_name}):")
            self.logger.info(
                f"   Raw: [{self.priority_min:.2f}, {self.priority_max:.2f}]"
            )
            self.logger.info(f"   Mean: {arr.mean():.2f} ± {arr.std():.2f}")

    def _normalize_priorities(self, priorities: np.ndarray) -> np.ndarray:
        """Normalize to [0.05, 0.95].  All arithmetic in float32."""
        priorities = np.asarray(priorities, dtype=np.float32)
        priorities = np.maximum(priorities, 1e-6)
        if np.all((priorities >= 0.0) & (priorities <= 1.0)):
            return np.clip(priorities, 0.01, 0.99)
        if self.priority_max > self.priority_min and self.priority_max > 1.5:
            normalized = (priorities - self.priority_min) / (
                self.priority_max - self.priority_min
            )
            normalized = np.clip(normalized, 0.0, 1.0)
            return (normalized * 0.90 + 0.05).astype(np.float32)
        return np.clip(priorities, 0.01, 0.99).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scenario = self.data[idx]
        detections = scenario["detections"]
        priorities = scenario["priorities"]

        # Always numpy for _normalize_priorities, always float32
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.numpy().astype(np.float32)
        else:
            priorities = np.asarray(priorities, dtype=np.float32)

        if self.normalize_priorities:
            priorities = self._normalize_priorities(priorities)

        priorities = torch.tensor(priorities, dtype=torch.float32)

        n_detections = len(detections)
        edge_type_id = scenario.get("edge_type_id", 0)
        edge_type_ids = torch.full((n_detections,), edge_type_id, dtype=torch.long)

        return {
            "detections": detections,
            "priorities": priorities,
            "class_ids": scenario["class_ids"],
            "edge_type_ids": edge_type_ids,
            "scenario_id": scenario["scenario_id"],
            "detector_data": scenario.get("detector_data", {}),
        }


# ============================================================================
# UTILITY
# ============================================================================

def _config_get(cfg: Any, key: str, default: Any) -> Any:
    try:
        if hasattr(cfg, key):
            return getattr(cfg, key)
        if isinstance(cfg, dict):
            return cfg.get(key, default)
    except Exception:
        pass
    return default


def assemble_detector_strains(
    detector_data: Dict,
    detectors: List[str] = None,
    default_length: int = 16384,
) -> Optional[torch.Tensor]:
    """
    Assemble strain data into a stacked float32 tensor [n_detectors, time_samples].

    MPS NOTE: Returns CPU tensor intentionally — it will be moved to the target
    device inside the training/eval loop just before the forward pass.  Moving
    per-batch is safer than storing pre-moved tensors in Dataset.__getitem__.
    """
    if detectors is None:
        detectors = ["H1", "L1", "V1"]
    if not detector_data:
        return None

    reference_shape = None
    for detector in detectors:
        if detector in detector_data:
            det_entry = detector_data[detector]
            strain_data = (
                det_entry.get("strain") if isinstance(det_entry, dict) else det_entry
            )
            if strain_data is not None:
                try:
                    reference_shape = np.asarray(strain_data).shape
                    break
                except Exception:
                    pass

    if reference_shape is None:
        reference_shape = (default_length,)

    strain_list = []
    for detector in detectors:
        strain_data = None
        if detector in detector_data:
            det_entry = detector_data[detector]
            if isinstance(det_entry, dict):
                strain_data = det_entry.get("strain")
            elif isinstance(det_entry, (np.ndarray, torch.Tensor)):
                strain_data = det_entry

            if strain_data is not None:
                if isinstance(strain_data, np.ndarray):
                    # Cast to float32 — MPS doesn't support float64
                    strain_data = torch.from_numpy(strain_data.astype(np.float32))
                elif isinstance(strain_data, torch.Tensor):
                    strain_data = strain_data.float()
                else:
                    strain_data = None

                if strain_data is not None and strain_data.shape != reference_shape:
                    strain_data = None

        if strain_data is None:
            strain_data = torch.zeros(reference_shape, dtype=torch.float32)

        strain_list.append(strain_data)

    if len(strain_list) == len(detectors):
        return torch.stack(strain_list)  # CPU tensor; moved to device in the loop
    logging.error(
        f"Incomplete strain data: {len(strain_list)}/{len(detectors)} detectors"
    )
    return None


def collate_priority_batch(batch):
    """
    Collate function for PriorityNet batches.

    All tensors returned are on CPU.  The training loop moves them to the target
    device (MPS / CUDA / CPU) immediately before each forward/backward pass.
    This avoids holding device memory across batches and is the recommended
    pattern for MPS (which uses unified memory anyway).
    """
    detections_batch = []
    priorities_batch = []
    edge_type_ids_batch = []
    strain_batch = []
    snr_values_batch = []
    detector_data_batch = []

    for item in batch:
        if isinstance(item, dict):
            dets = item["detections"]
            prios = item["priorities"]  # already float32

            edge_id_scalar = item.get("edge_type_id", None)
            if edge_id_scalar is None:
                edge_ids_tensor = item.get("edge_type_ids", None)
                edge_ids = (
                    edge_ids_tensor
                    if isinstance(edge_ids_tensor, torch.Tensor)
                    else None
                )
            else:
                edge_ids = torch.full((len(dets),), edge_id_scalar, dtype=torch.long)

            snr_vals = item.get("snr_values", None)
            detector_data = item.get("detector_data", {})
            strain_tensor = assemble_detector_strains(detector_data) if detector_data else None

            if strain_tensor is not None and not hasattr(collate_priority_batch, "_logged"):
                logging.info(
                    f"[COLLATE] Strain shape: {strain_tensor.shape}, "
                    f"detectors: {list(detector_data.keys())}"
                )
                collate_priority_batch._logged = True

            detections_batch.append(dets)
            priorities_batch.append(prios)
            edge_type_ids_batch.append(edge_ids)
            strain_batch.append(strain_tensor)
            snr_values_batch.append(snr_vals)
            detector_data_batch.append(detector_data)
        else:
            dets = item[0]
            prios = item[1]
            edge_ids = item[2] if len(item) > 2 else None
            strain = item[3] if len(item) > 3 else None
            snr_vals = item[4] if len(item) > 4 else None
            detections_batch.append(dets)
            priorities_batch.append(prios)
            edge_type_ids_batch.append(edge_ids)
            strain_batch.append(strain)
            snr_values_batch.append(snr_vals)
            detector_data_batch.append({})

    return (
        detections_batch,
        priorities_batch,
        edge_type_ids_batch,
        strain_batch,
        snr_values_batch,
        detector_data_batch,
    )


# ============================================================================
# TRAINING
# ============================================================================

def _move_strain_to_device(
    strain: Optional[torch.Tensor], device: torch.device
) -> Optional[torch.Tensor]:
    """Move strain tensor to device, ensuring float32."""
    if strain is None:
        return None
    return to_device_float32(strain, device)


def _move_priorities_to_device(
    prios: torch.Tensor, device: torch.device
) -> torch.Tensor:
    return to_device_float32(prios, device)


def train_priority_net_with_validation(
    config,
    train_dataset,
    val_dataset,
    output_dir: Path,
    resume_state: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    PriorityNet training with MPS support, warmup ramp, validation,
    resume support, and per-epoch rank diagnostics.
    """
    logging.info("🧠  Training PriorityNet with validation...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ device
    device = get_device()
    logging.info(f"🖥️  Training device: {device}")
    dl_kwargs = dataloader_kwargs(device)

    # -------------------------------------------------------- resume / init
    if resume_state is not None:
        model = resume_state["model"]
        trainer = resume_state["trainer"]
        start_epoch = resume_state["start_epoch"]
        best_val_loss = resume_state["best_val_loss"]
        patience_counter = resume_state["patience_counter"]
        training_metrics = resume_state["training_metrics"]
        logging.info(f"🔄 Resuming from epoch {start_epoch}")
    else:
        use_strain = get_config_value(config, "priority_net.use_strain", True, bool)
        use_edge_conditioning = get_config_value(
            config, "priority_net.use_edge_conditioning", True, bool
        )
        n_edge_types = get_config_value(config, "priority_net.n_edge_types", 19, int)
        priority_net_config = (
            config.get("priority_net", config) if hasattr(config, "get") else config
        )
        model = PriorityNet(
            config=priority_net_config,
            use_strain=use_strain,
            use_edge_conditioning=use_edge_conditioning,
            n_edge_types=n_edge_types,
        ).to(device)
        trainer = PriorityNetTrainer(model, priority_net_config)
        start_epoch = 0
        best_val_loss = float("inf")
        patience_counter = 0
        training_metrics = {
            "train_losses": [], "val_losses": [],
            "train_ranking_losses": [], "train_priority_losses": [],
            "val_ranking_losses": [], "val_priority_losses": [],
            "grad_norms": [], "learning_rates": [], "val_spearman": [],
            "epochs_completed": 0, "best_epoch": 0,
            "warmup_epochs": get_config_value(config, "warmup_epochs", 10),
        }

    batch_size = get_config_value(config, "batch_size", 16, int)
    n_epochs = get_config_value(config, "epochs", 250, int)
    patience = get_config_value(config, "patience", 30, int)
    warmup_epochs = get_config_value(config, "warmup_epochs", 5, int)

    logging.info(
        f"📝 Training config: epochs={start_epoch+1}→{n_epochs}, "
        f"batch={batch_size}, patience={patience}, device={device}"
    )

    train_sampler = create_weighted_sampler(
        train_dataset, n_signals_threshold=5, oversample_factor=1.35
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_priority_batch,
        sampler=train_sampler,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_priority_batch,
        **dl_kwargs,
    )

    training_start_time = time.time()
    affine_mode = False
    affine_epochs_left = 0
    prev_prev_mae = None
    prev_mae = None

    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()

        # -------------------------------------------------------- TRAIN
        model.train()
        train_losses, train_ranking_losses, train_priority_losses, train_grad_norms = (
            [], [], [], []
        )

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{n_epochs}")
        for batch in train_pbar:
            detections_batch = batch[0]
            priorities_batch = batch[1]
            edge_type_ids_batch = batch[2] if len(batch) > 2 else None
            strain_batch = batch[3] if len(batch) > 3 else None
            snr_values_batch = batch[4] if len(batch) > 4 else None

            # Move priorities and strains to device (float32 enforced)
            priorities_on_device = [
                _move_priorities_to_device(p, device) for p in priorities_batch
            ]
            strains_on_device = (
                [_move_strain_to_device(s, device) for s in strain_batch]
                if strain_batch is not None
                else None
            )

            try:
                loss_info = trainer.train_step(
                    detections_batch,
                    priorities_on_device,
                    strain_batch=strains_on_device,
                    edge_type_ids_batch=edge_type_ids_batch,
                    snr_values_batch=snr_values_batch,
                )
            except TypeError:
                loss_info = trainer.train_step(detections_batch, priorities_on_device)

            train_losses.append(loss_info["loss"])
            train_ranking_losses.append(loss_info.get("ranking_loss", 0.0))
            train_priority_losses.append(loss_info.get("priority_loss", 0.0))
            train_grad_norms.append(loss_info.get("grad_norm", 0.0))
            train_pbar.set_postfix({
                "Loss": f"{loss_info['loss']:.2e}",
                "Prior": f"{loss_info.get('priority_loss', 0):.2e}",
                "Grad": f"{loss_info.get('grad_norm', 0):.2e}",
                "Gain": f"{loss_info.get('affine_gain', 0):.2f}",
                "Bias": f"{loss_info.get('affine_bias', 0):.2f}",
            })

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        avg_ranking_loss = float(np.mean(train_ranking_losses)) if train_ranking_losses else 0.0
        avg_priority_loss = float(np.mean(train_priority_losses)) if train_priority_losses else 0.0
        avg_grad_norm = float(np.mean(train_grad_norms)) if train_grad_norms else 0.0
        training_metrics["train_losses"].append(avg_train_loss)
        training_metrics["train_ranking_losses"].append(avg_ranking_loss)
        training_metrics["train_priority_losses"].append(avg_priority_loss)
        training_metrics["grad_norms"].append(avg_grad_norm)

        # -------------------------------------------------------- VALIDATE
        model.eval()
        val_ranking_losses, val_priority_losses = [], []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{n_epochs}")
            for batch in val_pbar:
                detections_batch = batch[0]
                priorities_batch = batch[1]
                edge_type_ids_batch = batch[2] if len(batch) > 2 else None
                strain_batch_v = batch[3] if len(batch) > 3 else None
                snr_values_batch = batch[4] if len(batch) > 4 else None
                detector_data_batch = batch[5] if len(batch) > 5 else None

                batch_ranking = 0.0
                batch_priority = 0.0
                valid_batches = 0

                for idx, (detections, target_priorities) in enumerate(
                    zip(detections_batch, priorities_batch)
                ):
                    if not detections or len(target_priorities) == 0:
                        continue

                    # Move to device — float32 enforced
                    target_priorities = _move_priorities_to_device(target_priorities, device)

                    strain_segments = (
                        _move_strain_to_device(strain_batch_v[idx], device)
                        if strain_batch_v is not None
                        else None
                    )
                    edge_ids_item = (
                        edge_type_ids_batch[idx]
                        if edge_type_ids_batch is not None
                        else None
                    )
                    if edge_ids_item is not None:
                        edge_ids_item = edge_ids_item.to(device)

                    if strain_segments is None:
                        n = len(detections)
                        strain_segments = torch.zeros(
                            (n, 3, 2048), dtype=torch.float32, device=device
                        )

                    try:
                        detector_data = (
                            detector_data_batch[idx]
                            if detector_data_batch and idx < len(detector_data_batch)
                            else {}
                        )
                        predicted_priorities, uncertainties = model(
                            detections,
                            strain_segments=strain_segments,
                            edge_type_ids=edge_ids_item,
                            detector_data=detector_data,
                        )
                    except Exception as e:
                        logging.debug(f"Validation step error: {e}")
                        continue

                    min_len = min(len(predicted_priorities), len(target_priorities))
                    if min_len == 0:
                        continue

                    pred_slice = predicted_priorities[:min_len]
                    target_slice = target_priorities[:min_len].to(pred_slice.device)
                    unc_slice = uncertainties[:min_len]

                    snr_vals = (
                        snr_values_batch[idx] if snr_values_batch is not None else None
                    )
                    if snr_vals is not None:
                        snr_vals = snr_vals[:min_len].to(pred_slice.device)

                    losses = trainer.loss_fn(
                        pred_slice, target_slice, unc_slice, snr_values=snr_vals
                    )
                    batch_ranking += float(losses["ranking"])
                    batch_priority += float(losses["mse"])
                    valid_batches += 1

                if valid_batches > 0:
                    val_ranking_losses.append(batch_ranking / valid_batches)
                    val_priority_losses.append(batch_priority / valid_batches)
                    val_pbar.set_postfix({
                        "Rank": f"{batch_ranking/valid_batches:.2e}",
                        "Prior": f"{batch_priority/valid_batches:.2e}",
                    })

        avg_val_ranking = float(np.mean(val_ranking_losses)) if val_ranking_losses else 0.0
        avg_val_priority = float(np.mean(val_priority_losses)) if val_priority_losses else 0.0
        avg_val_loss = avg_val_ranking + avg_val_priority
        training_metrics["val_losses"].append(avg_val_loss)
        training_metrics["val_ranking_losses"].append(avg_val_ranking)
        training_metrics["val_priority_losses"].append(avg_val_priority)
        training_metrics["epochs_completed"] = epoch + 1

        current_lr = trainer.optimizer.param_groups[0]["lr"]
        training_metrics["learning_rates"].append(float(current_lr))

        # ---------------------------------------- PER-EPOCH SPEARMAN (CPU)
        # Pull predictions to CPU numpy — scipy works on CPU only
        val_corr = float("nan")
        with torch.no_grad():
            sample_preds, sample_targets = [], []
            buckets = {
                "1": {"preds": [], "tgts": []},
                "2": {"preds": [], "tgts": []},
                "3-4": {"preds": [], "tgts": []},
                "5+": {"preds": [], "tgts": []},
            }
            for batch in val_loader:
                detections_batch = batch[0]
                priorities_batch = batch[1]
                edge_type_ids_batch = batch[2] if len(batch) > 2 else None
                strain_batch_v = batch[3] if len(batch) > 3 else None
                detector_data_batch = batch[5] if len(batch) > 5 else None

                for i, (detections, priorities) in enumerate(
                    zip(detections_batch, priorities_batch)
                ):
                    n = len(priorities)
                    key = (
                        "1" if n == 1
                        else "2" if n == 2
                        else "3-4" if 3 <= n <= 4
                        else "5+"
                    )
                    if n < 2:
                        continue

                    priorities_dev = _move_priorities_to_device(priorities, device)
                    strain_segments = (
                        _move_strain_to_device(strain_batch_v[i], device)
                        if strain_batch_v is not None
                        else None
                    )
                    edge_ids_item = (
                        edge_type_ids_batch[i].to(device)
                        if edge_type_ids_batch is not None and edge_type_ids_batch[i] is not None
                        else None
                    )
                    if strain_segments is None:
                        strain_segments = torch.zeros(
                            (len(detections), 3, 2048), dtype=torch.float32, device=device
                        )

                    try:
                        detector_data = (
                            detector_data_batch[i]
                            if detector_data_batch and i < len(detector_data_batch)
                            else {}
                        )
                        pred, _ = model(
                            detections,
                            strain_segments=strain_segments,
                            edge_type_ids=edge_ids_item,
                            detector_data=detector_data,
                        )
                        # Pull to CPU float32 numpy — avoid float64
                        pred_np = pred.detach().cpu().float().numpy()
                        tgt_np = priorities_dev.detach().cpu().float().numpy()
                        m = min(len(pred_np), len(tgt_np))
                        if m < 2:
                            continue
                        p = pred_np[:m].ravel()
                        t = tgt_np[:m].ravel()
                        sample_preds.extend(p.tolist())
                        sample_targets.extend(t.tolist())
                        buckets[key]["preds"].extend(p.tolist())
                        buckets[key]["tgts"].extend(t.tolist())
                    except Exception as e:
                        logging.debug(f"Per-epoch corr forward failed: {e}")
                        continue

        if len(sample_preds) >= 10:
            preds_np = np.array(sample_preds, dtype=np.float32)
            tgts_np = np.array(sample_targets, dtype=np.float32)

            corr, pval = spearmanr(preds_np, tgts_np)
            if np.isfinite(corr):
                val_corr = float(corr)

            mae = float(np.mean(np.abs(preds_np - tgts_np)))
            rmse = float(np.sqrt(np.mean((preds_np - tgts_np) ** 2)))

            plateau = (
                prev_mae is not None and prev_prev_mae is not None
                and (prev_prev_mae - prev_mae) < 1e-4
                and (prev_mae - mae) < 1e-4
            )
            spearman_ok = np.isfinite(val_corr) and val_corr >= 0.90
            if spearman_ok and plateau and not affine_mode:
                logging.info("🎯 Entering 1-epoch affine calibration")
                trainer.set_affine_calibration(
                    enable=True, base_lr=trainer.optimizer.param_groups[0]["lr"]
                )
                affine_mode = True
                affine_epochs_left = 1

            prev_prev_mae = prev_mae
            prev_mae = mae

            if hasattr(trainer, "loss_fn") and hasattr(trainer.loss_fn, "lambda_calib"):
                trainer.loss_fn.lambda_calib = (
                    3e-4 if (mae < 0.02 and preds_np.max() < 0.55) else 1e-4
                )

            logging.info(
                f"   📈 Eval: MAE={mae:.3e}, RMSE={rmse:.3e}, "
                f"pred=[{preds_np.min():.3e}, {preds_np.max():.3e}], "
                f"tgt=[{tgts_np.min():.3e}, {tgts_np.max():.3e}]"
            )
            logging.info(f"   📊 Spearman: {corr:.3f} (p={pval:.7f})")

            for key, buf in buckets.items():
                if len(buf["preds"]) >= 10:
                    try:
                        sp, sp_p = spearmanr(buf["preds"], buf["tgts"])
                        kt, kt_p = kendalltau(buf["preds"], buf["tgts"], variant="b")
                        logging.info(
                            f"   🔎 Bucket {key}: Spearman={sp:.3f} (p={sp_p:.1e}), "
                            f"Kendallτ={kt:.3f} (p={kt_p:.1e})"
                        )
                    except Exception as e:
                        logging.debug(f"Bucket {key} corr failed: {e}")

        training_metrics["val_spearman"].append(val_corr)

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time
        logging.info(
            f"Epoch {epoch+1}/{n_epochs}: "
            f"Train={avg_train_loss:.6e}, Val={avg_val_loss:.6e}, "
            f"LR={current_lr:.2e}, Time={epoch_time:.1f}s"
        )

        # ---------------------------------------- LR SCHEDULING
        old_lr = trainer.optimizer.param_groups[0]["lr"]
        if hasattr(trainer, "warmup_scheduler") and epoch < warmup_epochs:
            trainer.warmup_scheduler.step()
            new_lr = trainer.optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                logging.info(f"   🔼 Warmup LR: {old_lr:.2e} → {new_lr:.2e}")
        else:
            if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                trainer.scheduler.step(avg_val_loss)
                new_lr = trainer.optimizer.param_groups[0]["lr"]
                if new_lr < old_lr:
                    logging.info(f"🔻 LR REDUCED: {old_lr:.2e} → {new_lr:.2e}")
                    patience_counter = 0
                else:
                    num_bad = trainer.scheduler.num_bad_epochs
                    pat_sched = trainer.scheduler.patience
                    logging.info(
                        f"   ✅ LR unchanged: {old_lr:.2e} (bad epochs: {num_bad}/{pat_sched})"
                    )
            else:
                trainer.scheduler.step()
                new_lr = trainer.optimizer.param_groups[0]["lr"]
                if new_lr != old_lr:
                    logging.info(f"🔻 LR changed: {old_lr:.2e} → {new_lr:.2e}")

        if affine_mode:
            affine_epochs_left -= 1
            if affine_epochs_left <= 0:
                logging.info("✅ Exiting affine calibration; restoring full training")
                trainer.set_affine_calibration(enable=False)
                affine_mode = False

        # ---------------------------------------- CHECKPOINT
        if avg_val_loss < best_val_loss - 1e-6:
            improvement = best_val_loss - avg_val_loss
            best_val_loss = avg_val_loss
            patience_counter = 0
            training_metrics["best_epoch"] = epoch + 1

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": trainer.scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "training_metrics": training_metrics,
                "total_training_time": total_time,
                "model_config": config.__dict__ if hasattr(config, "__dict__") else config,
                "use_transformer_encoder": model.use_transformer_encoder,
                "model_architecture": {
                    "use_strain": model.use_strain,
                    "use_edge_conditioning": model.use_edge_conditioning,
                    "use_transformer_encoder": model.use_transformer_encoder,
                    "n_edge_types": model.n_edge_types,
                    "strain_encoder_type": (
                        "whisper"
                        if (
                            hasattr(model, "strain_encoder")
                            and hasattr(model.strain_encoder, "use_whisper")
                            and model.strain_encoder.use_whisper
                        )
                        else "lightweight"
                    ),
                },
                "training_config": {
                    "patience": patience,
                    "patience_counter": patience_counter,
                    "best_epoch": training_metrics.get("best_epoch", epoch),
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "warmup_epochs": warmup_epochs,
                },
                "training_data": {
                    k: training_metrics.get(k, [])
                    for k in [
                        "train_losses", "val_losses", "train_ranking_losses",
                        "train_priority_losses", "val_ranking_losses", "val_priority_losses",
                        "grad_norms", "learning_rates", "val_spearman",
                    ]
                },
            }
            torch.save(checkpoint, output_dir / "priority_net_best.pth")
            logging.info(
                f"✅ Best model saved (val_loss: {avg_val_loss:.2e}, "
                f"improvement: {improvement:.2e})"
            )
        else:
            patience_counter += 1
            logging.info(
                f"   No improvement for {patience_counter} epochs "
                f"({patience - patience_counter} remaining)"
            )
            if patience_counter >= patience:
                logging.info(f"⏹️  Early stopping at epoch {epoch+1}")
                break

    logging.info(
        f"🎉 Training complete. Best epoch: {training_metrics['best_epoch']}, "
        f"Best val loss: {best_val_loss:.2e}"
    )
    return training_metrics


# ============================================================================
# DATA SPLITS
# ============================================================================

def create_data_splits(
    scenarios: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    logging.info(
        f" Creating splits: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test"
    )
    train_scenarios, temp_scenarios = train_test_split(
        scenarios, test_size=(val_ratio + test_ratio), random_state=42
    )
    val_scenarios, test_scenarios = train_test_split(
        temp_scenarios,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=42,
    )
    logging.info(
        f"   Train: {len(train_scenarios)}, Val: {len(val_scenarios)}, Test: {len(test_scenarios)}"
    )
    return train_scenarios, val_scenarios, test_scenarios


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_priority_net(
    model,
    dataset,
    split_name: str = "test",
    debug_plots: bool = False,
    out_dir: str = "outputs/priority_net",
):
    """MPS-compatible evaluation — all numpy/scipy work on CPU."""
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs(out_dir, exist_ok=True)

    logging.info(f"🔍 Evaluating PriorityNet on {split_name} set...")
    start_time = time.time()

    model.eval()
    device = next(model.parameters()).device  # matches training device

    correlations = []
    kendalls_all = []
    spearmans_all = []
    precisions = []
    lens = []
    pairwise_accuracies = []

    successful_evaluations = 0
    total_multidet = 0
    total_scenarios = 0

    with torch.no_grad():
        for item_idx, item in enumerate(tqdm(dataset, desc=f"Evaluating {split_name}")):
            try:
                detections = item["detections"]
                true_priorities = item["priorities"]
                total_scenarios += 1

                if len(detections) < 2:
                    continue
                total_multidet += 1

                edge_type_id = item.get("edge_type_id", None)
                edge_ids = (
                    torch.full(
                        (len(detections),), edge_type_id, dtype=torch.long, device=device
                    )
                    if edge_type_id is not None
                    else None
                )

                strain_segments = None
                detector_data = item.get("detector_data", {})
                if detector_data:
                    st = assemble_detector_strains(detector_data)
                    if st is not None:
                        # Shape: [n_det, time]; model expects [1, n_det, time] or [n_det, time]
                        strain_segments = to_device_float32(st, device)

                try:
                    pred_priorities, uncertainties = model(
                        detections,
                        strain_segments=strain_segments,
                        edge_type_ids=edge_ids,
                        detector_data=detector_data,
                    )
                except Exception as e:
                    if item_idx < 5:
                        logging.error(f"Forward pass failed at {item_idx}: {e}")
                    continue

                # Pull to CPU float32 numpy — avoids MPS → float64 issues
                pred_np = pred_priorities.detach().cpu().float().numpy()
                true_np = (
                    true_priorities.detach().cpu().float().numpy()
                    if isinstance(true_priorities, torch.Tensor)
                    else np.asarray(true_priorities, dtype=np.float32)
                )

                m = min(len(pred_np), len(true_np))
                if m < 2:
                    continue

                pred_np = pred_np[:m]
                true_np = true_np[:m]

                if not (np.isfinite(pred_np).all() and np.isfinite(true_np).all()):
                    continue
                if np.allclose(true_np, true_np[0]) or np.allclose(pred_np, pred_np[0]):
                    continue
                if np.std(pred_np) < 1e-8 or np.std(true_np) < 1e-8:
                    continue

                # Pairwise accuracy
                n_correct = sum(
                    (true_np[i] > true_np[j]) == (pred_np[i] > pred_np[j])
                    for i in range(m)
                    for j in range(i + 1, m)
                )
                n_pairs = m * (m - 1) // 2
                pairwise_acc = n_correct / max(1, n_pairs)

                # Z-score on CPU float32
                t_z = (true_np - true_np.mean()) / (true_np.std() + 1e-8)
                p_z = (pred_np - pred_np.mean()) / (pred_np.std() + 1e-8)

                sp_corr, _ = spearmanr(t_z, p_z)
                kd_corr, _ = kendalltau(t_z, p_z)
                spearmans_all.append(float(sp_corr) if np.isfinite(sp_corr) else 0.0)
                kendalls_all.append(float(kd_corr) if np.isfinite(kd_corr) else 0.0)

                corr_val = (
                    float(kd_corr) if m < 3 else float(sp_corr)
                )
                if not np.isfinite(corr_val):
                    corr_val = 0.0

                correlations.append(corr_val)
                pairwise_accuracies.append(pairwise_acc)
                lens.append(m)

                k = min(3, m)
                true_topk = set(np.argsort(true_np)[-k:])
                pred_topk = set(np.argsort(pred_np)[-k:])
                precisions.append(len(true_topk & pred_topk) / k)

                successful_evaluations += 1

            except Exception as e:
                if item_idx < 5:
                    logging.error(f"Error at {item_idx}: {e}\n{traceback.format_exc()}")
                continue

    success_rate = successful_evaluations / max(total_multidet, 1)
    results = {
        "split": split_name,
        "n_samples": successful_evaluations,
        "total_multidet": total_multidet,
        "total_scenarios": total_scenarios,
        "success_rate": success_rate,
        "failure_rate": 1.0 - success_rate,
        "eval_time_sec": float(time.time() - start_time),
    }

    if correlations:
        corrs_arr = np.array(correlations, dtype=np.float32)
        prec_arr = np.array(precisions, dtype=np.float32)
        lens_arr = np.array(lens, dtype=np.int32)
        spears_arr = np.array(spearmans_all, dtype=np.float32)
        kend_arr = np.array(kendalls_all, dtype=np.float32)
        pairwise_arr = np.array(pairwise_accuracies, dtype=np.float32)

        results.update({
            "avg_correlation": float(np.mean(corrs_arr)),
            "std_correlation": float(np.std(corrs_arr)),
            "median_correlation": float(np.median(corrs_arr)),
            "min_correlation": float(np.min(corrs_arr)),
            "max_correlation": float(np.max(corrs_arr)),
            "fraction_positive_corr": float(np.mean(corrs_arr > 0.0)),
            "avg_spearman": (
                float(np.nanmean(spears_arr[lens_arr >= 3])) if spears_arr.size else 0.0
            ),
            "avg_kendall": (
                float(np.nanmean(kend_arr[lens_arr < 3])) if kend_arr.size else 0.0
            ),
            "avg_topk_precision": float(np.mean(prec_arr)),
            "std_topk_precision": float(np.std(prec_arr)),
            "avg_pairwise_accuracy": float(np.mean(pairwise_arr)),
            "std_pairwise_accuracy": float(np.std(pairwise_arr)),
        })

        try:
            np.save(os.path.join(out_dir, f"{split_name}_correlations.npy"), corrs_arr)
            np.save(os.path.join(out_dir, f"{split_name}_precisions.npy"), prec_arr)
        except Exception as e:
            logging.debug(f"Metric dump failed: {e}")

        logging.info(
            f"{split_name.upper()}: {successful_evaluations}/{total_multidet} multi-det scenarios"
        )
        logging.info(
            f"   Corr: {results['avg_correlation']:.3f} ± {results['std_correlation']:.3f}"
        )
        logging.info(
            f"   Spearman(m≥3): {results['avg_spearman']:.3f} | "
            f"Kendall(m<3): {results['avg_kendall']:.3f}"
        )
        logging.info(
            f"   Pairwise Acc: {results['avg_pairwise_accuracy']:.3f} | "
            f"P@3: {results['avg_topk_precision']:.3f} | "
            f"Time: {results['eval_time_sec']:.1f}s"
        )
    else:
        logging.warning("❌ No successful correlations computed")

    return results


# ============================================================================
# CHECKPOINT
# ============================================================================

def load_checkpoint(
    checkpoint_path: Optional[str], config, device=None
) -> Optional[Dict[str, Any]]:
    """Load checkpoint with MPS-safe map_location."""
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        logging.info("No checkpoint found, starting fresh training")
        return None

    logging.info(f"📂 Found checkpoint: {checkpoint_path}")

    try:
        if device is None:
            device = get_device()

        # map_location must be a string for MPS ("mps") or torch.device
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        def cfg_get(cfg, key, default):
            if isinstance(cfg, dict):
                if key in cfg:
                    return cfg[key]
                if "priority_net" in cfg and isinstance(cfg["priority_net"], dict):
                    if key in cfg["priority_net"]:
                        return cfg["priority_net"][key]
                return default
            if hasattr(cfg, key):
                return getattr(cfg, key)
            if hasattr(cfg, "priority_net"):
                pn = getattr(cfg, "priority_net")
                if hasattr(pn, key):
                    return getattr(pn, key)
            return default

        checkpoint_encoder_type = checkpoint.get("use_transformer_encoder", False)
        config_encoder_type = cfg_get(config, "use_transformer_encoder", False)
        if checkpoint_encoder_type != config_encoder_type:
            logging.warning(
                f"⚠️  Encoder type mismatch (ckpt={checkpoint_encoder_type}, "
                f"cfg={config_encoder_type}). Starting fresh."
            )
            return None

        arch = checkpoint.get("model_architecture", {})
        model = PriorityNet(
            config,
            use_strain=arch.get("use_strain", True),
            use_edge_conditioning=arch.get("use_edge_conditioning", True),
            n_edge_types=arch.get("n_edge_types", 19),
        ).to(device)

        pretrained = checkpoint.get("model_state_dict", {})
        current = model.state_dict()
        missing = [k for k in current if k not in pretrained]
        unexpected = [k for k in pretrained if k not in current]
        shape_mismatch = [
            k for k in pretrained
            if k in current and current[k].shape != pretrained[k].shape
        ]
        had_mismatches = bool(missing or unexpected or shape_mismatch)

        try:
            model.load_state_dict(pretrained, strict=True)
            logging.info("✅ Model state loaded (strict=True)")
        except RuntimeError:
            compatible = {
                k: v for k, v in pretrained.items()
                if k in current and current[k].shape == v.shape
            }
            model.load_state_dict(compatible, strict=False)
            logging.warning(
                f"⚠️  Loaded {len(compatible)}/{len(pretrained)} compatible keys"
            )

        priority_net_config = (
            config.get("priority_net", config) if hasattr(config, "get") else config
        )
        trainer = PriorityNetTrainer(model, priority_net_config)

        if not had_mismatches:
            if "optimizer_state_dict" in checkpoint and trainer.optimizer is not None:
                try:
                    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    logging.info("✅ Optimizer restored")
                except Exception as e:
                    logging.warning(f"⚠️  Optimizer not restored: {e}")
            if "scheduler_state_dict" in checkpoint and trainer.scheduler is not None:
                try:
                    trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    logging.info("✅ Scheduler restored")
                except Exception as e:
                    logging.warning(f"⚠️  Scheduler not restored: {e}")
        else:
            logging.info("🔄 Optimizer/scheduler reset (architecture changes)")

        train_cfg = checkpoint.get("training_config", {})
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        patience_counter = train_cfg.get("patience_counter", 0)

        ckpt_metrics = checkpoint.get("training_metrics", {})
        if not ckpt_metrics:
            td = checkpoint.get("training_data", {})
            ckpt_metrics = {k: td.get(k, []) for k in [
                "train_losses", "val_losses", "train_ranking_losses",
                "val_ranking_losses", "train_priority_losses", "val_priority_losses",
                "grad_norms", "learning_rates", "val_spearman",
            ]}
            ckpt_metrics["best_epoch"] = train_cfg.get("best_epoch", 0)
            ckpt_metrics["epochs_completed"] = start_epoch - 1

        logging.info(
            f"✅ Checkpoint loaded: epoch {start_epoch}, "
            f"best_val_loss={best_val_loss:.2e}, "
            f"patience={patience_counter}/{cfg_get(config, 'patience', 30)}"
        )

        return {
            "model": model,
            "trainer": trainer,
            "start_epoch": start_epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "training_metrics": ckpt_metrics,
            "checkpoint_path": checkpoint_path,
        }

    except Exception as e:
        logging.error(f"❌ Failed to load checkpoint: {e}\n{traceback.format_exc()}")
        logging.warning("⚠️ Starting from scratch")
        return None


# ============================================================================
# PLOT
# ============================================================================

def plot_enhanced_training_curves(training_metrics: Dict, output_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    warmup_epochs = training_metrics.get("warmup_epochs", 10)
    best_epoch = training_metrics.get("best_epoch", 0)
    val_losses = training_metrics.get("val_losses", [])
    epochs = list(range(len(val_losses)))
    lrs = training_metrics.get("learning_rates", None)

    def _plot(ax, data, title, ylabel, color="steelblue"):
        if data:
            ax.plot(epochs[:len(data)], data, linewidth=2, color=color)
        ax.axvline(warmup_epochs, color="gray", linestyle="--", alpha=0.5, label="Warmup End")
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5, label="Best")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], val_losses, "Total Loss (Val)", "Loss")
    _plot(axes[0, 1], training_metrics.get("val_ranking_losses", []), "Ranking Loss (Val)", "Loss")
    _plot(axes[0, 2], training_metrics.get("val_priority_losses", []), "Priority Loss (Val)", "MSE")

    grad_norms = training_metrics.get("grad_norms", [])
    if grad_norms:
        axes[1, 0].plot(epochs[:len(grad_norms)], grad_norms, linewidth=2, color="purple")
    axes[1, 0].set_yscale("log"); axes[1, 0].set_title("Gradient Norm")
    axes[1, 0].grid(True, alpha=0.3)

    if lrs:
        axes[1, 1].plot(epochs[:len(lrs)], lrs, linewidth=2, color="orange")
        axes[1, 1].set_yscale("log"); axes[1, 1].set_title("Learning Rate")
        axes[1, 1].grid(True, alpha=0.3)

    _plot(axes[1, 2], val_losses, "Validation Loss (Zoomed)", "Loss", "red")

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: PriorityNet — MPS compatible"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--create_overlaps", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    device = get_device()
    logging.info(f"🚀 Starting Phase 2 PriorityNet | device={device}")

    if not Path(args.dataset_path).exists():
        logging.error(f"❌ Dataset path not found: {args.dataset_path}")
        return
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        logging.error("❌ Split ratios must sum to 1.0")
        return

    try:
        config = load_enhanced_config(Path(args.config))
        log_config(config, logging.getLogger())
        validate_config(config, logging.getLogger())
    except Exception as e:
        logging.error(f"❌ Failed to load config: {e}")
        raise

    logging.info("📂 Loading GW datasets...")
    train_loader = ChunkedGWDataLoader(
        args.dataset_path, split="train", max_samples=args.max_samples
    )
    val_loader = ChunkedGWDataLoader(
        args.dataset_path, split="validation", max_samples=args.max_samples
    )
    test_loader = ChunkedGWDataLoader(
        args.dataset_path, split="test", max_samples=args.max_samples
    )

    logging.info("🔄 Converting to PriorityNet scenarios...")
    train_scenarios = train_loader.convert_to_priority_scenarios(
        create_overlaps=args.create_overlaps, overlap_probability=0.3
    )
    val_scenarios = val_loader.convert_to_priority_scenarios(
        create_overlaps=args.create_overlaps, overlap_probability=0.2
    )
    test_scenarios = test_loader.convert_to_priority_scenarios(
        create_overlaps=args.create_overlaps, overlap_probability=0.2
    )

    train_dataset = PriorityNetDataset(train_scenarios, "train")
    val_dataset = PriorityNetDataset(val_scenarios, "validation")
    test_dataset = PriorityNetDataset(test_scenarios, "test")

    if len(train_dataset) == 0:
        logging.error("❌ No valid training scenarios")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Multi-detection diagnostic
    logging.info("=" * 60)
    detection_counts: Dict[int, int] = {}
    for item in train_dataset:
        n = len(item.get("detections", []))
        detection_counts[n] = detection_counts.get(n, 0) + 1
    multi_det = sum(v for k, v in detection_counts.items() if k >= 2)
    total_items = len(train_dataset)
    logging.info("📊 Detection distribution (train):")
    for n in sorted(detection_counts):
        logging.info(
            f"   {n} det: {detection_counts[n]} ({detection_counts[n]/total_items*100:.1f}%)"
        )
    pct = multi_det / total_items * 100
    level = "🟢" if pct > 30 else ("🟡" if pct > 10 else "🔴")
    logging.info(f"{level} Multi-detection: {multi_det}/{total_items} ({pct:.1f}%)")
    logging.info("=" * 60)

    # Checkpoint
    checkpoint_path = None
    if args.no_resume:
        logging.info("🔄 --no-resume: starting fresh")
    elif args.resume:
        checkpoint_path = args.resume
    else:
        default_ckpt = output_dir / "priority_net_best.pth"
        if default_ckpt.exists():
            checkpoint_path = str(default_ckpt)
            logging.info(f"🔍 Auto-resuming from {checkpoint_path}")

    resume_state = load_checkpoint(checkpoint_path, config, device)

    training_metrics = train_priority_net_with_validation(
        config, train_dataset, val_dataset, output_dir, resume_state=resume_state
    )

    # Final evaluation
    train_results = val_results = test_results = None
    try:
        logging.info("📊 Loading best model for evaluation...")
        best_ckpt = torch.load(
            output_dir / "priority_net_best.pth",
            weights_only=False,
            map_location=device,
        )
        eval_config = best_ckpt.get("model_config", config)
        if isinstance(eval_config, dict) and "priority_net" in eval_config:
            eval_config = eval_config["priority_net"]
        model = PriorityNet(
            eval_config, use_strain=True, use_edge_conditioning=True, n_edge_types=19
        ).to(device)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=False)

        train_results = evaluate_priority_net(model, train_dataset, "train", out_dir=str(output_dir))
        val_results = evaluate_priority_net(model, val_dataset, "validation", out_dir=str(output_dir))
        test_results = evaluate_priority_net(model, test_dataset, "test", out_dir=str(output_dir))
    except Exception as e:
        logging.warning(f"⚠️  Evaluation skipped: {e}")

    final_results = {
        "training_metrics": training_metrics,
        "evaluation_results": {
            "train": train_results,
            "validation": val_results,
            "test": test_results,
        },
        "model_config": config.__dict__ if hasattr(config, "__dict__") else {},
        "dataset_info": {
            "train_scenarios": len(train_scenarios),
            "val_scenarios": len(val_scenarios),
            "test_scenarios": len(test_scenarios),
        },
        "device": str(device),
    }
    with open(output_dir / "complete_results.pkl", "wb") as f:
        pickle.dump(final_results, f)

    plot_enhanced_training_curves(training_metrics, output_dir)

    print("\n" + "=" * 70)
    print("🎉  PHASE 2 COMPLETE — PRIORITYNET")
    print(f"🖥️  Device: {device}")
    print("=" * 70)
    if test_results:
        corr = test_results.get("avg_correlation", 0.0)
        print(f"✅ TEST Ranking Correlation : {corr*100:.1f}%")
        print(f"✅ VAL  Ranking Correlation : {val_results.get('avg_correlation', 0)*100:.1f}%")
        grade = "🟢 Production ready!" if corr > 0.85 else ("🟡 Good" if corr > 0.70 else "🔴 Needs work")
        print(grade)
    print(f"\n💾 Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()