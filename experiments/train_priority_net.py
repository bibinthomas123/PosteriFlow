#!/usr/bin/env python3
"""
COMPLETE Phase 2: PriorityNet with Integrated Dataset Loading
Reads directly from your 20K chunked dataset structure
Enhanced with train/validation/test splits and comprehensive evaluation
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


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("phase2_priority_net_complete.log"), logging.StreamHandler()],
    )


def encode_edge_type(dets):
    """
    FIXED: Match dataset_generator.py encoding scheme.
    Map overlap size to a stable int ID used in generated datasets:
    0: single signal
    3: pairwise overlap
    6: triple overlap
    7+: higher overlaps (4+)

    Args:
        dets: List of detection parameters (or single dict)

    Returns:
        Integer ID matching dataset generator scheme
    """
    if isinstance(dets, str):
        # Handle string edge case types by returning None (fallback to auto-classification)
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
        return 7  # For size 4+, use 7


# ============================================================================
# FIXED: Priority Calculation
# ============================================================================


def calculate_priority(signal: Dict[str, Any], is_overlap: bool = False, time_lag_ms: float = 0.0) -> float:
    """
    Enhanced priority calculation with SNR sensitivity boost and context-aware edge bonuses.
    
    IMPROVEMENTS (Nov 16, 2025):
    - SNR sensitivity: Non-linear scaling with amplification in 15-35 range
    - Edge bonuses: Context-aware (scale with SNR)
    - Distance bonus: Closer events are higher priority (easier to extract)
    - Overlap penalty: Depends on signal separation
    """
    # Step 1: Get target SNR (it's directly in signal dict!)
    target_snr = signal.get("target_snr", 0.0)

    # Fallback to network_snr if exists
    if target_snr == 0.0:
        target_snr = signal.get("network_snr", 0.0)

    # Try detector-specific SNRs
    if target_snr == 0.0:
        h1_data = signal.get("H1", {})
        l1_data = signal.get("L1", {})
        if isinstance(h1_data, dict) and isinstance(l1_data, dict):
            h1_snr = float(h1_data.get("optimal_snr", 0.0))
            l1_snr = float(l1_data.get("optimal_snr", 0.0))
            target_snr = np.sqrt(h1_snr**2 + l1_snr**2)

    target_snr = float(target_snr)

    if target_snr < 8.0:
        return 0.0

    # ========================================================================
    # A1: SNR SENSITIVITY BOOST (Non-linear scaling for better discrimination)
    # ========================================================================
    # The key insight: SNR 15-35 range is most discriminative.
    # Use power-law scaling to amplify discrimination in the mid-range.
    # Power < 1 suppresses low SNR, power > 1 amplifies.
    
    if target_snr < 8.0:
        base_priority = 0.0  # Already handled above
    elif target_snr < 12.0:
        # Very low SNR: suppress slightly (power = 0.9)
        base_priority = target_snr ** 0.9
    elif target_snr < 20.0:
        # Low-mid SNR: amplify moderately (power = 1.15) for better discrimination
        base_priority = target_snr ** 1.15
    elif target_snr < 35.0:
        # High sensitivity zone: strong amplification (power = 1.25)
        base_priority = target_snr ** 1.25
    else:
        # Very high SNR: saturation with power = 1.1 (diminishing returns)
        base_priority = target_snr ** 1.1
    edge_bonus = 0.0

    # ========================================================================
    # A2: CONTEXT-AWARE EDGE BONUSES (Scale with SNR)
    # ========================================================================
    # Edge cases are more interesting when SNR is moderate.
    # Keep bonuses as relative multipliers applied after SNR sensitivity.
    
    snr_context = np.clip(target_snr / 30.0, 0.0, 1.0)  # Normalize to [0,1] at SNR=30

    # High mass ratio (stronger impact at moderate SNR)
    try:
        mass_1 = float(signal.get("mass_1", 10.0))
        mass_2 = float(signal.get("mass_2", 10.0))
        if mass_1 > 0 and mass_2 > 0:
            q = max(mass_1, mass_2) / min(mass_1, mass_2)
            if q > 8.0:
                edge_bonus += 0.10 * snr_context  # High mass ratio: up to 0.1 bonus
            elif q > 4.0:
                edge_bonus += 0.05 * snr_context  # Moderate mass ratio
    except:
        pass

    # High spin (spinning objects are more scientifically interesting)
    try:
        a1 = abs(float(signal.get("a1", signal.get("a_1", 0.0))))
        a2 = abs(float(signal.get("a2", signal.get("a_2", 0.0))))
        a_max = max(a1, a2)
        if a_max > 0.8:
            edge_bonus += 0.12 * snr_context  # Highly spinning: up to 0.12
        elif a_max > 0.5:
            edge_bonus += 0.06 * snr_context  # Moderately spinning
    except:
        pass

    # Eccentric (orbital eccentricity is rare and important)
    try:
        eccentricity = float(signal.get("eccentricity", 0.0))
        if eccentricity > 0.1:
            edge_bonus += 0.15 * snr_context  # Context-aware eccentricity bonus
    except:
        pass

    # Cap edge bonus (multiplicative, so small multipliers)
    edge_bonus = min(edge_bonus, 0.40)  # Max 40% boost from edge features
    
    # ========================================================================
    # A3: OVERLAP PENALTY REFINEMENT (Distance-dependent)
    # ========================================================================
    if is_overlap:
        if time_lag_ms < 500:
            overlap_penalty = 0.70  # Close overlaps: significant penalty
        elif time_lag_ms < 1000:
            overlap_penalty = 0.82  # Medium overlaps: moderate penalty
        else:
            overlap_penalty = 0.95  # Distant overlaps: minor penalty
    else:
        overlap_penalty = 1.0
    
    # ========================================================================
    # A4: DISTANCE-BASED PRIORITIZATION (Closer = higher priority)
    # ========================================================================
    # Closer events are intrinsically higher priority (easier to extract accurately)
    distance_bonus = 1.0
    try:
        luminosity_distance = float(signal.get("luminosity_distance", 500.0))
        if luminosity_distance < 100:
            distance_bonus = 1.18  # Very close: 18% bonus
        elif luminosity_distance < 200:
            distance_bonus = 1.10  # Close: 10% bonus
        elif luminosity_distance < 400:
            distance_bonus = 1.04  # Moderate: 4% bonus
        # else: distance_bonus = 1.0 (no bonus for distant)
    except:
        pass

    # Combine all factors
    priority = base_priority * (1.0 + edge_bonus) * overlap_penalty * distance_bonus

    return float(priority)


def estimate_scenario_difficulty(priorities: np.ndarray) -> float:
    """
    Estimate scenario difficulty based on priority gaps.
    
    Hard scenarios (close priorities) should be oversampled for better learning.
    
    Difficulty is inverse of top-2 priority gap:
    - Large gap (easy): difficulty ≈ 0.5
    - Small gap (hard): difficulty ≈ 2.0 (oversampled 4x)
    
    Args:
        priorities: Array of signal priorities
    
    Returns:
        difficulty weight in range [0.5, 2.0]
    """
    if len(priorities) < 2:
        return 1.0
    
    # Sort in descending order
    prios_sorted = np.sort(priorities)[::-1]
    
    # Gap between top-2 priorities
    gap = float(prios_sorted[0] - prios_sorted[1])
    
    # Difficulty is inverse of gap: small gap = large difficulty
    # gap=0.5 -> difficulty=2.0 (hard)
    # gap=5.0 -> difficulty=0.5 (easy)
    difficulty = 1.0 / np.clip(gap + 0.1, 0.1, 2.0)
    
    # Clamp to reasonable range
    return float(np.clip(difficulty, 0.5, 2.0))


def create_weighted_sampler(dataset, n_signals_threshold=5, oversample_factor=1.35, use_difficulty=True):
    """
    Create sampler that oversamples hard scenarios.
    
    IMPROVEMENTS (Nov 16, 2025):
    - Oversample scenarios with ≥5 signals (more training signal)
    - Oversample hard scenarios (close priorities = hard to rank)
    
    Args:
        dataset: PriorityNetDataset
        n_signals_threshold: Min signals to trigger oversampling
        oversample_factor: Base oversampling multiplier for n≥threshold
        use_difficulty: If True, apply difficulty-based weighting
    
    Returns:
        WeightedRandomSampler
    """
    weights = []
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            n = len(sample.get("detections", []))
            
            # Base weight: oversample multi-detection scenarios
            weight = oversample_factor if n >= n_signals_threshold else 1.0
            
            # Difficulty weighting: hard scenarios get higher weight
            if use_difficulty and n >= 2:
                priorities_raw = sample.get("priorities", [])
                if isinstance(priorities_raw, np.ndarray):
                    priorities = priorities_raw.astype(np.float32)
                else:
                    # Convert to list first to avoid NumPy 2.0 copy keyword issues
                    priorities = np.array(list(priorities_raw), dtype=np.float32)
                if len(priorities) >= 2:
                    difficulty = estimate_scenario_difficulty(priorities)
                    weight *= difficulty  # Hard scenarios: up to 2x more weight
            
            weights.append(weight)
        except Exception as e:
            logging.debug(f"Weight calculation failed at {i}: {e}")
            weights.append(1.0)

    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


class ChunkedGWDataLoader:
    """
    Data loader for your 20K chunked GW dataset structure
    Converts GW samples to PriorityNet training scenarios
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize chunked GW dataset loader

        Args:
            dataset_path: Path to 'newDataset' directory
            split: 'train', 'validation', or 'test'
            max_samples: Maximum samples to load (None = all)
        """

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.max_samples = max_samples
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose

        # Load split information
        self._load_split_info()

        # Load all samples from chunks
        # self.samples = self._load_all_samples()

        # self.logger.info(f" {split.upper()} dataset loaded: {len(self.samples)} samples")

        self.last_processed = 0
        self.verbose = verbose

    @property
    def count(self) -> int:
        if not hasattr(self, "_count_cache"):
            self._count_cache = self.count_samples(ignore_max=False)
        return self._count_cache

    def count_samples(self, ignore_max: bool = False, cap: Optional[int] = None) -> int:
        """Count samples by streaming (respects max unless ignore_max=True)."""
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
        """Load split information"""

        split_dir = self.dataset_path / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Load split info
        split_info_file = split_dir / "split_info.json"
        if split_info_file.exists():
            with open(split_info_file, "r") as f:
                self.split_info = json.load(f)
        else:
            # Fallback: count chunk files
            chunk_files = list(split_dir.glob("chunk_*.pkl"))
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

    def iter_all_samples(self) -> Iterator[Dict]:
        """Stream samples from chunks."""
        total_yielded = 0
        self.logger.info(f"Loading {self.split} chunks (streaming)...")

        for chunk_id in range(self.n_chunks):
            chunk_file = self.dataset_path / self.split / f"chunk_{chunk_id:04d}.pkl"

            if not chunk_file.exists():
                self.logger.warning(f"Chunk file not found: {chunk_file}")
                continue

            try:
                with open(chunk_file, "rb") as f:
                    chunk_data = pickle.load(f)

                for sample in chunk_data:
                    yield sample
                    total_yielded += 1

                    if self.max_samples and total_yielded >= self.max_samples:
                        self.logger.info(f"Reached max samples limit: {self.max_samples}")
                        return

                # Let chunk_data go out of scope, or explicitly free
                del chunk_data

            except Exception as e:
                self.logger.warning(f"Failed to load chunk {chunk_id}: {e}")
                continue

    def _convert_noise_sample_to_scenario(self, sample: Dict) -> Optional[Dict]:
        """Convert noise sample to PriorityNet scenario"""

        try:
            metadata = sample.get("metadata", {})

            # Create noise parameter entry
            noise_param = {
                "mass_1": 0.0,
                "mass_2": 0.0,
                "luminosity_distance": 0.0,
                "network_snr": 0.0,
                "individual_snrs": {
                    det: 0.0 for det in metadata.get("detector_network", ["H1", "L1"])
                },
                "ra": 0.0,
                "dec": 0.0,
                "theta_jn": 0.0,
                "psi": 0.0,
                "phase": 0.0,
                "geocent_time": 0.0,
                "a1": 0.0,
                "a2": 0.0,
                "approximant": "noise",
                "event_type": "noise",
                "edge_case": False,
                "edge_case_type": None,
                "detector_network": metadata.get("detector_network", ["H1", "L1"]),
                "sample_id": sample.get("sample_id", "unknown"),
                "noise_type": metadata.get("noise_type", "gaussian"),
                "glitch_present": metadata.get("glitch_present", False),
            }

            scenario = {
                "scenario_id": sample.get("sample_id", "unknown"),
                "true_parameters": [noise_param],
                "baseline_biases": [],
                "detector_data": sample.get("detector_data", {}),
                "whitened_data": sample.get("whitened_data", {}),
                "metadata": metadata,
            }

            return scenario

        except Exception as e:
            self.logger.debug(f"Failed to convert noise sample: {e}")
            return None

    def convert_to_priority_scenarios(
        self,
        limit: Optional[int] = None,
        create_overlaps: bool = False,
        overlap_probability: float = 0.3,
        reservoir_max: int = 2000,  # bounded memory for singles used to create overlaps
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """
        Streaming conversion without materializing self.samples.
        Uses a bounded reservoir of single samples to create artificial overlaps.
        """
        assert 0.0 <= overlap_probability <= 1.0

        scenarios: List[Dict] = []
        failed_count = 0
        success_count = 0

        # Stats and bounded reservoir for single samples
        singles_seen = 0
        artificial_created = 0
        reservoir: List[Dict] = []
        rng = random.Random(seed) if seed is not None else random

        def maybe_add_to_reservoir(sample: Dict):
            nonlocal reservoir, singles_seen
            # Reservoir sampling (Algorithm R): uniform subset of singles with O(k) memory
            if len(reservoir) < reservoir_max:
                reservoir.append(sample)
            else:
                # Replace with decreasing probability
                j = rng.randint(0, singles_seen)  # inclusive
                if j < reservoir_max:
                    reservoir[j] = sample

        # Stream from chunks; do not use self.samples
        iterator = self.iter_all_samples()  # implement Fix 1 generator on your class
        processed = 0

        pbar = tqdm(iterator, desc="Converting (streaming)", disable=not self.verbose, total=limit)
        for sample in pbar:
            if limit is not None and processed >= limit:
                break
            processed += 1

            if not isinstance(sample, Dict):
                failed_count += 1
                continue

            # Convert singles vs overlaps
            if sample.get("type") == "overlap":
                scenario = self._convert_overlap_sample_to_scenario(sample)
                if scenario is not None:
                    scenarios.append(scenario)
                    success_count += 1
                else:
                    failed_count += 1
            else:
                # Single sample path
                scenario = self._convert_single_sample_to_scenario(sample)
                singles_seen += 1 if scenario is not None else 0

                if scenario is not None:
                    scenarios.append(scenario)
                    success_count += 1
                    # Maintain bounded reservoir of singles
                    maybe_add_to_reservoir(sample)
                else:
                    failed_count += 1

            # Incremental creation of artificial overlaps to match probability target
            if create_overlaps and overlap_probability > 0.0 and len(reservoir) >= 2:
                target = int(singles_seen * overlap_probability)
                while artificial_created < target:
                    # Reuse your existing method which draws internally from provided singles
                    artificial_scenario = self._create_artificial_overlap_scenario(reservoir)
                    if artificial_scenario is not None:
                        scenarios.append(artificial_scenario)
                        success_count += 1
                    artificial_created += 1

        print(f"Total processed: {processed}")
        print(f"Singles seen: {singles_seen}")
        print(f"Artificial overlaps created: {artificial_created}")
        print(f"\nConversion complete:")
        print(f"  Success: {success_count}")
        print(f"  Failed: {failed_count}")
        if (success_count + failed_count) > 0:
            print(f"  Success rate: {success_count/(success_count+failed_count)*100:.1f}%")

        # Warn on high failure rate
        if processed > 0 and failed_count > 0.5 * processed:
            print(f"\n⚠️  WARNING: High failure rate ({failed_count}/{processed})")
            print("  Check dataset structure with inspect_dataset.py")

        self.proccessed = processed
        return scenarios

    def _convert_multi_sample_to_scenario(self, sample: Dict) -> Optional[Dict]:
        """Convert multi-signal GW sample to PriorityNet scenario (legacy support)"""

        try:
            metadata = sample.get("metadata", {})
            signal_parameters = metadata.get("signal_parameters", [])

            if len(signal_parameters) < 2:
                # If it's actually a single signal, convert as single
                if len(signal_parameters) == 1:
                    return self._convert_single_sample_to_scenario(sample)
                else:
                    return None

            # This is essentially the same as overlap conversion
            return self._convert_overlap_sample_to_scenario(sample)

        except Exception as e:
            self.logger.debug(f"Failed to convert multi-sample: {e}")
            return None

    def _extract_priorities_from_dataset(
        self, sample: Dict, signal_params: List[Dict]
    ) -> Optional[List[float]]:
        """
        Extract priorities aligned to signal_params from the dataset.

        Order of precedence:
        1) sample['priorities'] (scalar or list)
        2) per-parameter keys in each param dict: ['priority', 'target_priority', 'label_priority', 'priority_score']

        Returns:
            list[float] with length == len(signal_params), or None if unavailable/mismatched.
        """
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

    def _resolve_edge_type(self, sample: Dict) -> tuple[Optional[str], Optional[int]]:
        """
        Resolve edge_type_id from dataset.

        FIXED: The dataset stores edge_type_id based on overlap size:
        - 0: single signal
        - 3: pairwise overlap (2 signals)
        - 6: triple overlap (3 signals)
        - 7: heavy overlap (4+ signals)

        This is generated during dataset creation and is the ground truth.
        """
        md = sample.get("metadata") or {}

        # Prefer dataset-provided edge_type_id (most reliable)
        edge_type_id = sample.get("edge_type_id", md.get("edge_type_id"))
        if edge_type_id is not None:
            try:
                return None, int(edge_type_id)
            except Exception:
                pass

        # Fallback: Compute from signal count if edge_type_id missing
        signal_params = md.get("signal_parameters", sample.get("parameters", []))
        if isinstance(signal_params, dict):
            signal_params = [signal_params]

        if not signal_params:
            return None, 0  # single signal

        # Encode based on overlap size (must match dataset_generator.py)
        n_signals = len([p for p in signal_params if p is not None])
        if n_signals == 1:
            return None, 0
        elif n_signals == 2:
            return None, 3
        elif n_signals == 3:
            return None, 6
        else:  # 4+
            return None, 7

    def _convert_single_sample_to_scenario(self, sample: Dict) -> Optional[Dict]:
        """
        Convert a single, labeled sample into a PriorityNet scenario using dataset-provided priorities.

        Behavior:
        - Skips noise samples to keep schema consistent.
        - Reads priorities from sample['priorities'] (or per-parameter fields); drops the sample if labels are missing/mismatched.
        - Ensures edge_type_id is an int by preferring dataset ID, otherwise using a default 'none' class.
        """
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
                    f"Sample type: {sample_type}, is_overlap: {sample.get('is_overlap')}"
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

            detections = [params for params in signal_params]
            edge_case_type, edge_type_id = self._resolve_edge_type(sample)
            if edge_type_id is None:
                # Fallback: compute from signal count
                edge_type_id = encode_edge_type(detections)

            return {
                "detections": detections,
                "priorities": torch.tensor(priorities, dtype=torch.float32),
                "sample_type": sample_type,
                "sample_id": sample.get("sample_id", sample.get("id", "unknown")),
                "is_edge_case": sample.get("is_edge_case", False),
                "edge_case_type": edge_case_type,
                "edge_type_id": int(edge_type_id),
                "detector_data": sample.get("detector_data", {}),  # ✅ Preserve strain data
            }
        except Exception as e:
            self._conversion_error_count = getattr(self, "_conversion_error_count", 0) + 1
            if self._conversion_error_count <= 10:
                self.logger.debug(f"Conversion error: {e}")
            return None

    def _convert_overlap_sample_to_scenario(self, sample: Dict) -> Optional[Dict]:
        """
        Convert a labeled overlap sample to a PriorityNet scenario using dataset-provided priorities.

        Behavior:
        - Resolves signal parameters from metadata.signal_parameters or parameters.
        - Reads priorities from sample['priorities'] (or per-parameter fields); drops the sample if labels are missing/mismatched.
        - Ensures edge_type_id is an int by preferring dataset ID, otherwise using a default 'none' class.
        """
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

            detections = [params for params in signal_params]
            edge_case_type, edge_type_id = self._resolve_edge_type(sample)
            if edge_type_id is None:
                # Fallback: compute from signal count
                edge_type_id = encode_edge_type(detections)

            return {
                "detections": detections,
                "priorities": torch.tensor(priorities, dtype=torch.float32),
                "sample_type": "overlap",
                "sample_id": sample.get("sample_id", sample.get("id", "unknown")),
                "is_edge_case": sample.get("is_edge_case", False),
                "edge_case_type": edge_case_type,
                "edge_type_id": int(edge_type_id),
                "detector_data": sample.get("detector_data", {}),  # ✅ Preserve strain data
            }
        except Exception as e:
            self.logger.debug(f"Overlap conversion error: {e}")
            return None

    def _create_artificial_overlap_scenario(self, single_samples: List[Dict]) -> Optional[Dict]:
        """
        Create a synthetic overlap scenario by mixing detector strains from sampled singles.

        Behavior:
        - Samples 2–3 singles, aligns detector strain lengths, and sums scaled strains.
        - Builds detections from first signal of each single and computes priorities on-the-fly for synthetic labels.
        - Returns a labeled scenario with 'detections', 'priorities', and 'detector_data'.
        """
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
                base_strain = np.asarray(base_strain)
                combined_detector_data[det] = {
                    "strain": np.zeros_like(base_strain, dtype=base_strain.dtype)
                }

            detections = []
            for i, s in enumerate(selected):
                md = s.get("metadata", {}) or {}
                sp = md.get("signal_parameters")
                if sp is None:
                    sp = s.get("parameters")
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
                    det_strain = np.asarray(det_strain)
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
                    "ra": sig.get("ra", 0.0),
                    "dec": sig.get("dec", 0.0),
                    "theta_jn": sig.get("theta_jn", 0.0),
                    "psi": sig.get("psi", 0.0),
                    "phase": sig.get("phase", 0.0),
                    "geocent_time": i * random.uniform(0.2, 1.0),
                    "a1": sig.get("a1", 0.0),
                    "a2": sig.get("a2", 0.0),
                    "approximant": sig.get("approximant", "IMRPhenomD"),
                    "event_type": s.get("event_type")
                    or md.get("event_type")
                    or sig.get("event_type")
                    or "BBH",
                    "detector_network": detectors,
                    "is_overlap": True,
                    "artificial": True,
                }
                detections.append(det_params)

            if not detections:
                return None

            priorities = [calculate_priority(p, is_overlap=True) for p in detections]
            scenario_id = f"artificial_overlap_{random.randint(1000, 9999)}"

            # Encode edge type based on overlap size for diversity
            n_signals = len(detections)
            if n_signals == 2:
                edge_type_id = 3  # pairwise overlap
            elif n_signals == 3:
                edge_type_id = 6  # triple overlap
            else:  # 4+
                edge_type_id = 7  # heavy overlap

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
                    "event_type": "overlap",
                    "n_signals": len(detections),
                    "overlap_type": "multi_signal",
                    "detector_network": detectors,
                    "artificial": True,
                    "scenario_type": "artificial_overlap",
                },
            }
        except Exception as e:
            self.logger.debug(f"Failed to create artificial overlap: {e}")
            return None


class PriorityNetDataset(Dataset):
    """Enhanced PriorityNet dataset with signal-type awareness"""

    def __init__(self, scenarios: List[Dict], split_name: str = "train"):
        """
        Initialize PriorityNet dataset from converted scenarios.

        Args:
            scenarios: List of scenario dicts from convert_to_priority_scenarios
            split_name: Name of dataset split ('train', 'validation', 'test')
        """

        self.data = []
        self.split_name = split_name
        self.normalize_priorities = True
        self.logger = logging.getLogger(__name__)

        bbh_count = bns_count = nsbh_count = noise_count = overlap_count = 0

        self.logger.info(f"📊 Processing {len(scenarios)} scenarios for {split_name} dataset...")

        for scenario_id, scenario in enumerate(scenarios):
            try:
                # ✅ NEW: Use 'detections' and 'priorities' fields from convert_to_priority_scenarios
                detections = scenario.get("detections", [])
                priorities = scenario.get("priorities", None)

                if not detections or priorities is None:
                    if scenario_id < 5:  # Log first few failures
                        self.logger.warning(
                            f"  Scenario {scenario_id} missing data. "
                            f"Has detections: {bool(detections)}, "
                            f"Has priorities: {priorities is not None}, "
                            f"Keys: {list(scenario.keys())}"
                        )
                    continue

                # Ensure priorities is tensor
                if not isinstance(priorities, torch.Tensor):
                    priorities = scenario["priorities"]
                    if isinstance(priorities, torch.Tensor):
                        priorities = priorities.numpy()
                    elif isinstance(priorities, list):
                        priorities = np.array(priorities)

                    if self.normalize_priorities:
                        priorities = self._normalize_priorities(priorities)

                    priorities_tensor = torch.tensor(priorities, dtype=torch.float32)

                # Verify length match
                if len(detections) != len(priorities):
                    self.logger.warning(
                        f"  Scenario {scenario_id} length mismatch: "
                        f"{len(detections)} detections vs {len(priorities)} priorities"
                    )
                    continue

                # Clean priorities (no NaN/Inf)
                priorities = torch.where(torch.isnan(priorities), torch.tensor(0.5), priorities)
                priorities = torch.where(torch.isinf(priorities), torch.tensor(1.0), priorities)

                # ✅ Extract class IDs from detections
                class_ids = []
                for detection in detections:
                    evt_raw = detection.get("event_type", detection.get("type", None))
                    evt = str(evt_raw).strip().lower() if evt_raw is not None else None

                    if evt and "bbh" in evt:
                        class_id = 0
                        bbh_count += 1
                    elif evt and "bns" in evt:
                        class_id = 1
                        bns_count += 1
                    elif evt and "nsbh" in evt:
                        class_id = 2
                        nsbh_count += 1
                    else:
                        class_id = 3  # noise or unknown
                        noise_count += 1

                    class_ids.append(class_id)

                # ✅ Get edge case information
                edge_type_id = scenario.get("edge_type_id", 0)
                is_edge_case = scenario.get("is_edge_case", False)
                edge_case_type = scenario.get("edge_case_type", None)

                # Store complete scenario
                scenario_data = {
                    "scenario_id": scenario.get("sample_id", f"scenario_{scenario_id}"),
                    "detections": detections,
                    "priorities": priorities,
                    "class_ids": torch.tensor(class_ids, dtype=torch.long),
                    "edge_type_id": edge_type_id,
                    "is_edge_case": is_edge_case,
                    "edge_case_type": edge_case_type,
                    "metadata": scenario.get("metadata", {}),
                    "detector_data": scenario.get("detector_data", {}),  # ✅ Preserve strain data
                }

                self.data.append(scenario_data)

                # Track overlaps
                if len(detections) > 1:
                    overlap_count += 1

            except Exception as e:
                if scenario_id < 5:
                    self.logger.warning(f"  Error processing scenario {scenario_id}: {e}")
                continue

        # Log statistics
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
        """Compute global priority statistics for normalization."""
        all_priorities = []
        for item in self.data:
            priorities = item.get("priorities")
            if isinstance(priorities, torch.Tensor):
                priorities = priorities.numpy()
            elif isinstance(priorities, list):
                priorities = np.array(priorities)
            if priorities is not None:
                all_priorities.extend([float(p) for p in priorities if p > 0])

        all_priorities = np.array(all_priorities)

        if len(all_priorities) > 0:
            # ✅ CRITICAL FIX: Use RAW priorities, not log!
            self.priority_min = float(all_priorities.min())
            self.priority_max = float(all_priorities.max())

            self.logger.info(f"📊 Priority stats ({self.split_name}):")
            self.logger.info(f"   Raw: [{self.priority_min:.2f}, {self.priority_max:.2f}]")
            self.logger.info(f"   Mean: {all_priorities.mean():.2f} ± {all_priorities.std():.2f}")
        else:
            self.priority_min = 0.0
            self.priority_max = 100.0  # Safe default

    def _normalize_priorities(self, priorities: np.ndarray) -> np.ndarray:
        """
        Normalize priorities intelligently based on their range.

        ✅ NEW LOGIC (Nov 11, 2025):
        - If priorities are ALREADY in [0, 1]: Use as-is (no double-normalization)
        - If priorities are RAW SNR (5-100): Normalize to [0.05, 0.95]

        This prevents information loss when dataset already provides normalized values.
        """
        priorities = np.asarray(priorities, dtype=np.float32)
        priorities = np.maximum(priorities, 1e-6)

        # ✅ CRITICAL FIX: Detect if already normalized
        # If ALL values are in [0, 1], they're already normalized
        if np.all((priorities >= 0.0) & (priorities <= 1.0)):
            # Already normalized! Just ensure safe bounds
            return np.clip(priorities, 0.01, 0.99)

        # ✅ LEGACY: Only rescale if detected raw SNR range (> 1.5)
        # This handles old datasets with SNR values like 5-100
        if self.priority_max > self.priority_min and self.priority_max > 1.5:
            normalized = (priorities - self.priority_min) / (self.priority_max - self.priority_min)
            normalized = np.clip(normalized, 0.0, 1.0)
            normalized = normalized * 0.90 + 0.05  # [0.05, 0.95]
            return normalized

        # Default: already normalized or small range - use directly
        return np.clip(priorities, 0.01, 0.99)

    # Legacy function - not used in current pipeline

    # def _compute_extraction_priorities(self, signals: List[Dict],
    #                             baseline_biases: Optional[List[Dict]] = None) -> torch.Tensor:
    #     """
    #     COMPLETE  Enhanced priority computation for all signal types

    #     Handles:
    #     - Noise samples (lowest priority)
    #     - BBH, BNS, NSBH with optimized parameters
    #     - Overlap scenarios
    #     - Edge cases
    #     - Original event type preservation
    #     - Distance/mass/SNR scaling
    #     - Bias corrections
    #     """

    #     n_signals = len(signals)
    #     priorities = torch.zeros(n_signals)

    #     for i, signal in enumerate(signals):
    #         try:
    #             #  Use original event_type, handle noise properly (normalize to upper)
    #             evt_raw = signal.get('event_type', 'BBH')
    #             event_type = str(evt_raw).strip().upper() if evt_raw is not None else 'BBH'

    #             # Handle noise samples - give them lowest but non-zero priority
    #             if event_type == 'NOISE':
    #                 # Noise gets consistent low priority  with small random variation
    #                 priorities[i] = 0.0

    #                 # Do not proceed to further classification/logging for noise
    #                 continue

    #             # Basic signal properties with safety checks (robust casting)
    #             def _as_float(val, default):
    #                 try:
    #                     return float(val)
    #                 except Exception:
    #                     return float(default)

    #             snr = max(0.1, _as_float(signal.get('network_snr', 10.0), 10.0))
    #             m1 = max(0.1, _as_float(signal.get('mass_1', 30.0), 30.0))
    #             m2 = max(0.1, _as_float(signal.get('mass_2', 25.0), 25.0))
    #             distance = max(1.0, _as_float(signal.get('luminosity_distance', 500.0), 500.0))

    #             # Ensure proper mass ordering
    #             if m2 > m1:
    #                 m1, m2 = m2, m1

    #             # Use original event type instead of mass-based classification
    #             if event_type not in ['BBH', 'BNS', 'NSBH']:
    #                 # Fallback to mass-based classification for unknown/invalid types
    #                 if m1 <= 3.0 and m2 <= 3.0:
    #                     signal_type = 'BNS'
    #                 elif (m1 <= 3.0 and m2 > 3.0) or (m1 > 3.0 and m2 <= 3.0):
    #                     signal_type = 'NSBH'
    #                 else:
    #                     signal_type = 'BBH'

    #                 # Avoid misleading logs for noise which is handled above
    #                 if event_type not in ['NOISE']:
    #                     self.logger.debug(f"Signal {i}: Unknown event_type '{event_type}', classified as {signal_type}")
    #             else:
    #                 signal_type = event_type

    #             # Derived quantities with safety checks
    #             total_mass = m1 + m2
    #             if total_mass > 0:
    #                 chirp_mass = (m1 * m2)**(3/5) / total_mass**(1/5)
    #                 mass_ratio = m2 / m1
    #                 symmetric_mass_ratio = (m1 * m2) / total_mass**2
    #             else:
    #                 chirp_mass = 1.0
    #                 mass_ratio = 0.5
    #                 symmetric_mass_ratio = 0.25

    #             #  ENHANCED: SNR priority with logarithmic scaling for high SNR
    #             if snr >= 50.0:
    #                 snr_priority = 1.4 + 0.1 * np.log10(snr / 50.0)  # Extra bonus for very high SNR
    #             elif snr >= 20.0:
    #                 snr_priority = 1.0 + 0.15 * np.log10(snr / 20.0)  # Bonus for high SNR
    #             elif snr >= 12.0:
    #                 snr_priority = snr / 12.0  # Linear scaling in medium range
    #             elif snr >= 8.0:
    #                 snr_priority = 0.7 + 0.3 * (snr - 8.0) / 4.0  # Gentle scaling for low SNR
    #             else:
    #                 snr_priority = 0.4 + 0.3 * snr / 8.0  # Very low SNR handling

    #             snr_priority = min(snr_priority, 1.5)  # Cap at 1.5

    #             #  ENHANCED: Mass priority optimized for each signal type
    #             if signal_type == 'BNS':
    #                 # BNS: Favor canonical masses, don't heavily penalize outliers
    #                 if 2.0 <= total_mass <= 3.5:  # Canonical BNS range
    #                     mass_priority = 1.0
    #                 elif 1.8 <= total_mass < 2.0:  # Light BNS
    #                     mass_priority = 0.95
    #                 elif 3.5 < total_mass <= 4.5:  # Heavy BNS
    #                     mass_priority = 0.9
    #                 elif 4.5 < total_mass <= 6.0:  # Very heavy BNS (still detectable)
    #                     mass_priority = 0.8
    #                 else:  # Extreme cases
    #                     mass_priority = 0.7

    #             elif signal_type == 'NSBH':
    #                 # NSBH: Wide mass range acceptance
    #                 if 4.0 <= total_mass <= 35.0:  # Standard NSBH
    #                     mass_priority = 1.0
    #                 elif 2.5 <= total_mass < 4.0:  # Light NSBH
    #                     mass_priority = 0.9
    #                 elif 35.0 < total_mass <= 80.0:  # Heavy NSBH
    #                     mass_priority = 0.95
    #                 elif 80.0 < total_mass <= 150.0:  # Very heavy NSBH
    #                     mass_priority = 0.85
    #                 else:  # Extreme cases
    #                     mass_priority = 0.75

    #             else:  # BBH
    #                 #  ENHANCED: Excellent support for all BBH masses
    #                 if 15.0 <= total_mass <= 50.0:  # Stellar mass BBH (optimal)
    #                     mass_priority = 1.0
    #                 elif 50.0 < total_mass <= 100.0:  # Intermediate mass BBH
    #                     mass_priority = 1.05  # Slight bonus for intermediate mass
    #                 elif 100.0 < total_mass <= 200.0:  # Heavy BBH (GW190521-like)
    #                     mass_priority = 1.1   # Bonus for heavy BBH (astrophysically interesting)
    #                 elif 200.0 < total_mass <= 400.0:  # Very heavy BBH
    #                     mass_priority = 1.0   # Still very valuable
    #                 elif 8.0 <= total_mass < 15.0:  # Light BBH
    #                     mass_priority = 0.95
    #                 elif 5.0 <= total_mass < 8.0:  # Very light BBH
    #                     mass_priority = 0.9
    #                 else:  # Extreme masses
    #                     mass_priority = 0.8

    #             #  ENHANCED: Distance priority with better distant signal support
    #             if chirp_mass > 0:
    #                 # Chirp mass scaling for detection range
    #                 chirp_mass_factor = (chirp_mass / 30.0)**(5/6)

    #                 # Signal-type dependent horizon scaling
    #                 if signal_type == 'BBH':
    #                     base_horizon = 1000.0  # BBH detectable further
    #                 elif signal_type == 'NSBH':
    #                     base_horizon = 800.0   # NSBH intermediate
    #                 else:  # BNS
    #                     base_horizon = 200.0   # BNS closer detection

    #                 effective_horizon = base_horizon * chirp_mass_factor
    #             else:
    #                 effective_horizon = 500.0

    #             if distance <= effective_horizon:
    #                 distance_priority = 1.0
    #             elif distance <= 2.0 * effective_horizon:
    #                 # Gradual falloff for distant signals
    #                 distance_priority = 0.6 + 0.4 * (2.0 * effective_horizon - distance) / effective_horizon
    #             elif distance <= 5.0 * effective_horizon:
    #                 # Extended range for very high mass systems
    #                 distance_priority = 0.3 + 0.3 * (5.0 * effective_horizon - distance) / (3.0 * effective_horizon)
    #             else:
    #                 # Very distant signals still get some priority
    #                 distance_priority = max(0.1, effective_horizon / distance)

    #             #  ENHANCED: Advanced detectability factors
    #             base_detectability = 1.0

    #             # Chirp mass bonus (higher chirp mass = louder signal)
    #             if chirp_mass >= 60.0:  # Very high chirp mass
    #                 chirp_bonus = 0.2
    #             elif chirp_mass >= 40.0:  # High chirp mass
    #                 chirp_bonus = 0.15
    #             elif chirp_mass >= 25.0:  # Moderate high chirp mass
    #                 chirp_bonus = 0.1
    #             elif chirp_mass >= 15.0:  # Standard chirp mass
    #                 chirp_bonus = 0.05
    #             elif chirp_mass >= 5.0:   # Low chirp mass
    #                 chirp_bonus = 0.02
    #             else:  # Very low chirp mass
    #                 chirp_bonus = 0.0

    #             # Mass ratio factor (symmetric masses generally easier to detect)
    #             if mass_ratio >= 0.9:  # Nearly equal masses
    #                 mass_ratio_bonus = 0.08
    #             elif mass_ratio >= 0.8:  # Close to equal
    #                 mass_ratio_bonus = 0.05
    #             elif mass_ratio >= 0.6:  # Moderate asymmetry
    #                 mass_ratio_bonus = 0.03
    #             elif mass_ratio >= 0.4:  # High asymmetry
    #                 mass_ratio_bonus = 0.01
    #             elif mass_ratio >= 0.2:  # Very high asymmetry
    #                 mass_ratio_bonus = 0.0
    #             else:  # Extreme asymmetry
    #                 mass_ratio_bonus = -0.02  # Small penalty for extreme asymmetry

    #             # Symmetric mass ratio factor (peaks at 0.25)
    #             eta_factor = 4.0 * symmetric_mass_ratio * (1.0 - symmetric_mass_ratio)
    #             eta_bonus = 0.05 * eta_factor

    #             detectability = base_detectability + chirp_bonus + mass_ratio_bonus + eta_bonus

    #             #  ENHANCED: Special handling for extreme and interesting cases
    #             extreme_bonus = 0.0

    #             # Very high mass bonus (interesting astrophysics)
    #             if total_mass >= 150.0:
    #                 extreme_bonus += 0.15  # Very interesting for astrophysics
    #             elif total_mass >= 80.0:
    #                 extreme_bonus += 0.1   # High mass bonus
    #             elif total_mass >= 50.0:
    #                 extreme_bonus += 0.05  # Moderate mass bonus

    #             # Very distant but high SNR bonus (rare but important)
    #             if distance >= 3000.0 and snr >= 15.0:
    #                 extreme_bonus += 0.15  # Exceptional detection
    #             elif distance >= 2000.0 and snr >= 12.0:
    #                 extreme_bonus += 0.1   # Very good distant detection
    #             elif distance >= 1000.0 and snr >= 10.0:
    #                 extreme_bonus += 0.05  # Good distant detection

    #             # Very high SNR bonus (regardless of other factors)
    #             if snr >= 100.0:
    #                 extreme_bonus += 0.2   # Exceptional SNR
    #             elif snr >= 50.0:
    #                 extreme_bonus += 0.15  # Very high SNR
    #             elif snr >= 30.0:
    #                 extreme_bonus += 0.1   # High SNR bonus

    #             # Low frequency bonus (longer inspiral, more information)
    #             f_lower = _as_float(signal.get('f_lower', 20.0), 20.0)
    #             if f_lower <= 10.0:
    #                 extreme_bonus += 0.1   # Very low frequency start
    #             elif f_lower <= 15.0:
    #                 extreme_bonus += 0.05  # Low frequency start

    #             # Edge case bonus
    #             if signal.get('edge_case', False):
    #                 edge_case_type = signal.get('edge_case_type', 'unknown')
    #                 if edge_case_type in ['high_spin', 'eccentric']:
    #                     extreme_bonus += 0.08  # Higher bonus for challenging cases
    #                 elif edge_case_type in ['short_bbh', 'long_bns']:
    #                     extreme_bonus += 0.06  # Medium bonus
    #                 else:
    #                     extreme_bonus += 0.04  # Standard edge case bonus

    #             # Overlap handling bonus
    #             if signal.get('is_overlap', False):
    #                 n_overlapping = signal.get('n_overlapping_signals', 1)
    #                 if n_overlapping >= 3:
    #                     extreme_bonus += 0.1   # Complex overlap scenario
    #                 elif n_overlapping == 2:
    #                     extreme_bonus += 0.05  # Standard overlap

    #                 # Time separation factor
    #                 time_sep = abs(signal.get('time_separation', 0.0))
    #                 if time_sep < 0.5:
    #                     extreme_bonus += 0.05  # Close in time (harder)

    #             # Spin magnitude bonus
    #             a1 = _as_float(signal.get('a1', 0.0), 0.0)
    #             a2 = _as_float(signal.get('a2', 0.0), 0.0)
    #             max_spin = max(abs(a1), abs(a2))
    #             if max_spin >= 0.9:
    #                 extreme_bonus += 0.08  # Very high spin
    #             elif max_spin >= 0.7:
    #                 extreme_bonus += 0.05  # High spin
    #             elif max_spin >= 0.5:
    #                 extreme_bonus += 0.02  # Moderate spin

    #             # Tidal parameter bonus (for BNS/NSBH)
    #             if signal_type in ['BNS', 'NSBH']:
    #                 lambda_1 = _as_float(signal.get('lambda_1', 0), 0.0)
    #                 lambda_2 = _as_float(signal.get('lambda_2', 0), 0.0)
    #                 max_lambda = max(lambda_1, lambda_2)
    #                 if max_lambda > 1000:
    #                     extreme_bonus += 0.05  # High tidal deformability
    #                 elif max_lambda > 500:
    #                     extreme_bonus += 0.03  # Moderate tidal effects

    #             #  ENHANCED: Baseline bias penalty (if available)
    #             bias_penalty = 0.0
    #             if baseline_biases and i < len(baseline_biases) and baseline_biases[i]:
    #                 try:
    #                     bias_values = [abs(float(b)) for b in baseline_biases[i].values()
    #                                 if isinstance(b, (int, float)) and not np.isnan(float(b))]
    #                     if bias_values:
    #                         bias_magnitude = np.mean(bias_values)
    #                         # Scale penalty based on bias severity
    #                         if bias_magnitude > 0.5:
    #                             bias_penalty = 0.15  # Severe bias
    #                         elif bias_magnitude > 0.3:
    #                             bias_penalty = 0.12  # High bias
    #                         elif bias_magnitude > 0.1:
    #                             bias_penalty = 0.08  # Moderate bias
    #                         else:
    #                             bias_penalty = 0.04  # Small bias
    #                 except Exception as e:
    #                     self.logger.debug(f"Bias calculation error for signal {i}: {e}")
    #                     bias_penalty = 0.0

    #             #  ENHANCED: SNR regime bonus/penalty
    #             snr_regime = signal.get('snr_regime', 'medium')
    #             snr_regime_modifier = 0.0
    #             if snr_regime == 'loud':
    #                 snr_regime_modifier = 0.05   # Loud signals get small bonus
    #             elif snr_regime == 'weak':
    #                 snr_regime_modifier = 0.03   # Weak signals get small bonus (challenging)
    #             elif snr_regime == 'low':
    #                 snr_regime_modifier = 0.02   # Low SNR get small bonus

    #             #  OPTIMIZED: Final priority formula
    #             # Weights: SNR is most important, then distance, then mass
    #             base_priority = (
    #                 0.40 * snr_priority +      # SNR weight slightly reduced
    #                 0.30 * distance_priority + # Distance weight increased
    #                 0.25 * mass_priority +     # Mass weight
    #                 0.05 * detectability       # Detectability factors
    #             ) * (1.0 + extreme_bonus + snr_regime_modifier) - bias_penalty

    #             #  REDUCED: Even smaller hierarchy penalty to avoid artificial ordering
    #             hierarchy_penalty = i * 0.002  # Very small penalty

    #             #  ENHANCED: Adaptive minimum priority based on signal quality
    #             if snr >= 15.0:
    #                 min_priority = 0.4  # High SNR signals
    #             elif snr >= 10.0:
    #                 min_priority = 0.35  # Medium SNR signals
    #             elif snr >= 8.0:
    #                 min_priority = 0.3   # Low SNR signals
    #             else:
    #                 min_priority = 0.25  # Very low SNR signals

    #             # Special minimum for edge cases and overlaps
    #             if signal.get('edge_case', False) or signal.get('is_overlap', False):
    #                 min_priority = max(min_priority, 0.35)

    #             final_priority = max(min_priority, base_priority - hierarchy_penalty)

    #             # Ensure reasonable bounds
    #             final_priority = min(max(final_priority, 0.1), 1.0)

    #             priorities[i] = final_priority

    #         except Exception as e:
    #             self.logger.warning(f"Error computing priority for signal {i}: {e}")
    #             # Assign safe default priority
    #             priorities[i] = 0.5
    #             continue

    #     # Final validation and normalization
    #     priorities = torch.clamp(priorities, min=0.01, max=1.0)

    #     # Ensure no NaN or infinite values
    #     priorities = torch.where(torch.isnan(priorities), torch.tensor(0.5), priorities)
    #     priorities = torch.where(torch.isinf(priorities), torch.tensor(1.0), priorities)

    #     return priorities

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single scenario with NORMALIZED priorities."""

        scenario = self.data[idx]

        detections = scenario["detections"]
        priorities = scenario["priorities"]

        # ✅ Convert to numpy if needed
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.numpy()

        # ✅ NORMALIZE using dataset-wide stats (computed in __init__)
        if self.normalize_priorities:
            priorities = self._normalize_priorities(priorities)

        # Convert back to tensor
        priorities = torch.tensor(priorities, dtype=torch.float32)

        class_ids = scenario["class_ids"]
        edge_type_id = scenario.get("edge_type_id", 0)

        n_detections = len(detections)
        edge_type_ids = torch.full((n_detections,), edge_type_id, dtype=torch.long)

        # ✅ Include detector_data with strain segments
        detector_data = scenario.get("detector_data", {})

        return {
            "detections": detections,
            "priorities": priorities,  # ✅ Now properly normalized [0.05-0.95]
            "class_ids": class_ids,
            "edge_type_ids": edge_type_ids,
            "scenario_id": scenario["scenario_id"],
            "detector_data": detector_data,  # ✅ For strain encoder
        }


def _config_get(cfg: Any, key: str, default: Any) -> Any:
    """Safely read a config value from either an object or a dict."""
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
    default_length: int = 16384
) -> Optional[torch.Tensor]:
    """
    Assemble strain data from multiple detectors into a stacked tensor.
    
    Extracts strain data from detector_data dict, handles format conversions,
    and backfills missing detectors with zeros. Ensures consistent shape across
    all detectors by using the first available shape as reference.
    
    CRITICAL FIX: Iterates through detectors in strict order and backfills missing
    detectors with zeros at their correct position to preserve detector ordering.
    
    Args:
        detector_data: Dict mapping detector names to data (e.g., {"H1": {...}, "L1": {...}})
        detectors: List of detector names to extract (default: ["H1", "L1", "V1"])
        default_length: Default length for zero-padding if no reference available (default: 16384)
        
    Returns:
        Stacked tensor [n_detectors, time_samples] or None if detector_data is empty
        
    Example:
        >>> detector_data = {
        ...     "H1": {"strain": np.array([...])},
        ...     "L1": {"strain": torch.tensor([...])}
        ... }
        >>> strain_tensor = assemble_detector_strains(detector_data)
        >>> strain_tensor.shape
        torch.Size([3, 16384])
    """
    if detectors is None:
        detectors = ["H1", "L1", "V1"]
    
    if not detector_data:
        logging.debug("Empty detector_data provided to assemble_detector_strains")
        return None
    
    strain_list = []
    reference_shape = None
    available_detectors = []
    
    # FIRST PASS: Find reference shape from first valid strain data
    for detector in detectors:
        if detector in detector_data:
            det_entry = detector_data[detector]
            
            # Handle both dict and direct array formats
            strain_data = None
            if isinstance(det_entry, dict):
                strain_data = det_entry.get("strain", None)
            elif isinstance(det_entry, (np.ndarray, torch.Tensor)):
                strain_data = det_entry
            
            if strain_data is not None:
                try:
                    if isinstance(strain_data, np.ndarray):
                        reference_shape = strain_data.shape
                        available_detectors.append(detector)
                        break
                    elif isinstance(strain_data, torch.Tensor):
                        reference_shape = strain_data.shape
                        available_detectors.append(detector)
                        break
                except Exception as e:
                    logging.debug(f"Error getting reference shape from {detector}: {e}")
                    pass
    
    # If still no reference shape, use default
    if reference_shape is None:
        reference_shape = (default_length,)
        logging.debug(f"No valid strain data found in detector_data. Using default shape: {reference_shape}")
    
    # SECOND PASS: Extract and backfill in detector order
    for detector in detectors:
        strain_data = None
        
        # Extract strain data from detector_data
        if detector in detector_data:
            det_entry = detector_data[detector]
            
            # Handle both dict and direct array formats
            if isinstance(det_entry, dict):
                strain_data = det_entry.get("strain", None)
            elif isinstance(det_entry, (np.ndarray, torch.Tensor)):
                strain_data = det_entry
            
            if strain_data is not None:
                # Convert to torch tensor if needed
                if isinstance(strain_data, np.ndarray):
                    strain_data = torch.from_numpy(strain_data).float()
                elif isinstance(strain_data, torch.Tensor):
                    strain_data = strain_data.float()
                else:
                    strain_data = None
                
                # Check shape consistency
                if strain_data is not None:
                    if strain_data.shape != reference_shape:
                        logging.debug(
                            f"Detector {detector} shape mismatch: "
                            f"{strain_data.shape} != {reference_shape}, using zeros"
                        )
                        strain_data = None
        
        # Backfill missing detectors with zeros (preserving order)
        if strain_data is None:
            strain_data = torch.zeros(reference_shape, dtype=torch.float32)
        
        strain_list.append(strain_data)
    
    if len(strain_list) == len(detectors):
        result = torch.stack(strain_list)
        if result.shape[0] != 3:
            logging.error(f"CRITICAL: Expected 3 detectors, got {result.shape[0]}. Shape: {result.shape}")
        return result
    else:
        logging.error(f"Incomplete strain data: {len(strain_list)}/{len(detectors)} detectors")
        return None


def collate_priority_batch(batch):
    """
    Collate function for PriorityNet datasets with optional SNR per signal and strain segments.

    Accepts either dict samples or tuple samples; ensures edge_type_ids is present.
    Extracts strain segments from detector_data (H1, L1, V1) and passes them through.

    Args:
        batch: Iterable of samples where each sample is a dict with keys:
               - 'detections': List[Dict]
               - 'priorities': torch.Tensor [n]
               - 'edge_type_ids': Optional[torch.LongTensor [n]]
               - 'snr_values': Optional[torch.Tensor [n]]
               - 'detector_data': Optional[Dict] with H1/L1/V1 strain data
               or a tuple (detections, priorities, [edge_type_ids], [snr_values], [strain])

    Returns:
        Tuple of 5 lists:
          detections_batch: List[List[Dict]]
          priorities_batch: List[torch.Tensor]
          edge_type_ids_batch: List[torch.LongTensor]
          strain_batch: List[Optional[torch.Tensor]] - strain segments [n_detectors, time_samples]
          snr_values_batch: List[Optional[torch.Tensor]]
    """
    detections_batch = []
    priorities_batch = []
    edge_type_ids_batch = []
    strain_batch = []
    snr_values_batch = []

    for item in batch:
        if isinstance(item, dict):
            dets = item["detections"]
            prios = item["priorities"]

            # ✅ Use dataset-provided edge_type_id (scalar) or edge_type_ids (tensor)
            edge_id_scalar = item.get("edge_type_id", None)  # single int per scenario
            if edge_id_scalar is None:
                # Fallback: check for edge_type_ids (tensor) from dataset
                edge_ids_tensor = item.get("edge_type_ids", None)
                if edge_ids_tensor is not None and isinstance(edge_ids_tensor, torch.Tensor):
                    edge_ids = edge_ids_tensor
                else:
                    edge_ids = None  # Let model handle via zeros internally
            else:
                # Expand scalar to per-signal tensor for model API
                edge_ids = torch.full((len(dets),), edge_id_scalar, dtype=torch.long)

            snr_vals = item.get("snr_values", None)

            # ✅ EXTRACT STRAIN SEGMENTS from detector_data
            detector_data = item.get("detector_data", {})
            strain_tensor = assemble_detector_strains(detector_data) if detector_data else None
            
            # DEBUG: Log strain shape for first few batches
            if strain_tensor is not None and not hasattr(collate_priority_batch, '_logged_first'):
                logging.info(f"[COLLATE DEBUG] Strain shape: {strain_tensor.shape}, detector_data keys: {list(detector_data.keys())}")
                collate_priority_batch._logged_first = True

            detections_batch.append(dets)
            priorities_batch.append(prios)
            edge_type_ids_batch.append(edge_ids)
            strain_batch.append(strain_tensor)
            snr_values_batch.append(snr_vals)
        else:
            # Tuple format: (detections, priorities, [edge_type_ids], [strain], [snr_values])
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

    return detections_batch, priorities_batch, edge_type_ids_batch, strain_batch, snr_values_batch


def train_priority_net_with_validation(config, train_dataset, val_dataset, output_dir: Path, resume_state: Optional[Dict] = None) -> Dict[str, Any]:
    """
    PriorityNet training with warmup ramp, validation, resume support,
    and per-epoch rank diagnostics (overall Spearman; per-bucket Spearman/Kendall).
    """
    import time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from scipy.stats import spearmanr, kendalltau
    import logging

    logging.info("🧠  Training PriorityNet with validation...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resume or init
    if resume_state is not None:
        model = resume_state["model"]
        trainer = resume_state["trainer"]
        start_epoch = resume_state["start_epoch"]
        best_val_loss = resume_state["best_val_loss"]
        patience_counter = resume_state["patience_counter"]
        training_metrics = resume_state["training_metrics"]
        logging.info(f"🔄 Resuming training from epoch {start_epoch}")
        logging.info(f"   Previous best val loss: {best_val_loss:.2e}")
    else:
        # Get config values - allow overrides via command line or use config defaults
        use_strain = get_config_value(config, "priority_net.use_strain", True, bool)
        use_edge_conditioning = get_config_value(
            config, "priority_net.use_edge_conditioning", True, bool
        )
        n_edge_types = get_config_value(config, "priority_net.n_edge_types", 19, int)

        # Extract priority_net config section for model
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
            "train_losses": [],
            "val_losses": [],
            "train_ranking_losses": [],
            "train_priority_losses": [],
            "val_ranking_losses": [],
            "val_priority_losses": [],
            "grad_norms": [],
            "learning_rates": [],
            "val_spearman": [],
            "epochs_completed": 0,
            "best_epoch": 0,
            "warmup_epochs": get_config_value(config, "warmup_epochs", 10),
        }

    # Read top-level training settings using unified config access
    batch_size = get_config_value(config, "batch_size", 16, int)
    n_epochs = get_config_value(config, "epochs", 250, int)
    patience = get_config_value(config, "patience", 30, int)
    warmup_epochs = get_config_value(config, "warmup_epochs", 5, int)

    logging.info("📝 Training configuration:")
    logging.info(f"   Epochs: {start_epoch + 1} → {n_epochs}")
    logging.info(f"   Batch size: {batch_size}")
    logging.info(f"   Patience: {patience}")

    train_sampler = create_weighted_sampler(
        train_dataset, n_signals_threshold=5, oversample_factor=1.35
    )
    # Efficient data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_priority_batch,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_priority_batch,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )

    training_start_time = time.time()

    # Affine calibration state (scoped to this training run)
    affine_mode = False
    affine_epochs_left = 0
    prev_prev_mae = None
    prev_mae = None

    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()

        # Train
        model.train()
        train_losses, train_ranking_losses, train_priority_losses, train_grad_norms = [], [], [], []

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{n_epochs}")
        for batch in train_pbar:
            # Support (detections_batch, priorities_batch, edge_type_ids_batch[, strain_batch][, snr_values_batch])
            detections_batch = batch[0]
            priorities_batch = batch[1]
            edge_type_ids_batch = batch[2] if len(batch) > 2 else None
            strain_batch = batch[3] if len(batch) > 3 else None
            snr_values_batch = batch[4] if len(batch) > 4 else None  # ALWAYS DEFINED

            try:
                loss_info = trainer.train_step(
                    detections_batch,
                    priorities_batch,
                    strain_batch=strain_batch,
                    edge_type_ids_batch=edge_type_ids_batch,
                    snr_values_batch=snr_values_batch,
                )
            except TypeError:
                # Backward compatibility if train_step has old signature
                loss_info = trainer.train_step(detections_batch, priorities_batch)

            train_losses.append(loss_info["loss"])
            train_ranking_losses.append(loss_info.get("ranking_loss", 0.0))
            train_priority_losses.append(loss_info.get("priority_loss", 0.0))
            train_grad_norms.append(loss_info.get("grad_norm", 0.0))
            train_pbar.set_postfix(
                {
                    "Loss": f"{loss_info['loss']:.2e}",
                    "Prior": f"{loss_info.get('priority_loss', 0):.2e}",
                    "Grad": f"{loss_info.get('grad_norm', 0):.2e}",
                    "Gain": f"{loss_info.get('affine_gain', 0):.2f}",  # NEW: Track range expansion
                    "Bias": f"{loss_info.get('affine_bias', 0):.2f}",  # NEW: Track offset
                }
            )

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        avg_ranking_loss = float(np.mean(train_ranking_losses)) if train_ranking_losses else 0.0
        avg_priority_loss = float(np.mean(train_priority_losses)) if train_priority_losses else 0.0
        avg_grad_norm = float(np.mean(train_grad_norms)) if train_grad_norms else 0.0

        training_metrics["train_losses"].append(avg_train_loss)
        training_metrics["train_ranking_losses"].append(avg_ranking_loss)
        training_metrics["train_priority_losses"].append(avg_priority_loss)
        training_metrics["grad_norms"].append(avg_grad_norm)

        # Validation
        model.eval()
        val_ranking_losses, val_priority_losses = [], []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch + 1}/{n_epochs}")
            for batch in val_pbar:
                # Support (detections_batch, priorities_batch, edge_type_ids_batch[, strain_batch][, snr_values_batch])
                detections_batch = batch[0]
                priorities_batch = batch[1]
                edge_type_ids_batch = batch[2] if len(batch) > 2 else None
                strain_batch = batch[3] if len(batch) > 3 else None
                snr_values_batch = batch[4] if len(batch) > 4 else None

                batch_ranking = 0.0
                batch_priority = 0.0
                valid_batches = 0

                for idx, (detections, target_priorities) in enumerate(
                    zip(detections_batch, priorities_batch)
                ):
                    if not detections or len(target_priorities) == 0:
                        continue

                    # Per-scenario aux
                    strain_segments = None if strain_batch is None else strain_batch[idx]
                    edge_ids_item = (
                        None if edge_type_ids_batch is None else edge_type_ids_batch[idx]
                    )

                    # Ensure consistent strain shape when missing
                    if strain_segments is None:
                        n = len(detections)
                        strain_segments = torch.zeros((n, 3, 2048), dtype=torch.float32)

                    try:
                        predicted_priorities, uncertainties = model(
                            detections, strain_segments=strain_segments, edge_type_ids=edge_ids_item
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

                    # Use the class-based loss fn for consistency
                    # losses = trainer.loss_fn(pred_slice, target_slice, unc_slice)
                    snr_vals = None if snr_values_batch is None else snr_values_batch[idx]
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
                    val_pbar.set_postfix(
                        {
                            "Rank": f"{batch_ranking / valid_batches:.2e}",
                            "Prior": f"{batch_priority / valid_batches:.2e}",
                        }
                    )

        avg_val_ranking = float(np.mean(val_ranking_losses)) if val_ranking_losses else 0.0
        avg_val_priority = float(np.mean(val_priority_losses)) if val_priority_losses else 0.0
        avg_val_loss = avg_val_ranking + avg_val_priority

        training_metrics["val_losses"].append(avg_val_loss)
        training_metrics["val_ranking_losses"].append(avg_val_ranking)
        training_metrics["val_priority_losses"].append(avg_val_priority)
        training_metrics["epochs_completed"] = epoch + 1

        current_lr = trainer.optimizer.param_groups[0]["lr"]
        training_metrics["learning_rates"].append(float(current_lr))

        # Per-epoch Spearman (overall) + per-bucket Spearman/Kendall over full val set
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
                # Support (detections_batch, priorities_batch, edge_type_ids_batch[, strain_batch][, snr_values_batch])
                detections_batch = batch[0]
                priorities_batch = batch[1]
                edge_type_ids_batch = batch[2] if len(batch) > 2 else None
                strain_batch = batch[3] if len(batch) > 3 else None
                snr_values_batch = batch[4] if len(batch) > 4 else None

                for i, (detections, priorities) in enumerate(
                    zip(detections_batch, priorities_batch)
                ):
                    n = len(priorities)
                    key = "1" if n == 1 else ("2" if n == 2 else ("3-4" if 3 <= n <= 4 else "5+"))
                    if n < 2:
                        continue

                    strain_segments = None if strain_batch is None else strain_batch[i]
                    edge_ids_item = None if edge_type_ids_batch is None else edge_type_ids_batch[i]

                    if strain_segments is None:
                        strain_segments = torch.zeros(
                            (len(detections), 3, 2048), dtype=torch.float32
                        )

                    try:
                        pred, _ = model(
                            detections, strain_segments=strain_segments, edge_type_ids=edge_ids_item
                        )
                        pred_np = (
                            pred.detach().cpu().numpy()
                            if isinstance(pred, torch.Tensor)
                            else np.asarray(pred)
                        )
                        tgt_np = (
                            priorities.detach().cpu().numpy()
                            if isinstance(priorities, torch.Tensor)
                            else np.asarray(priorities)
                        )
                        m = min(len(pred_np), len(tgt_np))
                        if m < 2:
                            continue
                        p = pred_np[:m].ravel()
                        t = tgt_np[:m].ravel()
                        sample_preds.extend(p)
                        sample_targets.extend(t)
                        buckets[key]["preds"].extend(p)
                        buckets[key]["tgts"].extend(t)

                        # Optional: collect SNR if you want bucket-aware diagnostics
                        # snr_vals = None if snr_values_batch is None else snr_values_batch[i]
                        # if snr_vals is not None: snr_vals = snr_vals[:m].cpu().numpy()

                    except Exception as e:
                        logging.debug(f"Per-epoch corr forward failed: {e}")
                        continue

        if len(sample_preds) >= 10:
            preds_np = np.asarray(sample_preds)
            tgts_np = np.asarray(sample_targets)

            corr, pval = spearmanr(preds_np, tgts_np)
            if np.isfinite(corr):
                val_corr = float(corr)

            mae = float(np.mean(np.abs(preds_np - tgts_np)))
            rmse = float(np.sqrt(np.mean((preds_np - tgts_np) ** 2)))
            spearman_ok = np.isfinite(val_corr) and (val_corr >= 0.88)
            # Require two-epoch plateau and higher rank
            plateau = (
                (prev_mae is not None and prev_prev_mae is not None)
                and ((prev_prev_mae - prev_mae) < 1e-4)
                and ((prev_mae - mae) < 1e-4)
            )
            spearman_ok = np.isfinite(val_corr) and (val_corr >= 0.90)
            if spearman_ok and plateau and not affine_mode:
                logging.info("🎯 Entering 1-epoch affine calibration (prio_gain/bias only)")
                trainer.set_affine_calibration(
                    enable=True, base_lr=trainer.optimizer.param_groups[0]["lr"]
                )
                affine_mode = True
                affine_epochs_left = 1

            prev_prev_mae = prev_mae
            prev_mae = mae

            if hasattr(trainer, "loss_fn") and hasattr(trainer.loss_fn, "lambda_calib"):
                if mae < 0.02 and preds_np.max() < 0.55:  # calibration saturation detected
                    trainer.loss_fn.lambda_calib = 3e-4  # 3× stronger calibration pull
                else:
                    trainer.loss_fn.lambda_calib = 1e-4

            logging.info(
                f"   📈 Eval (epoch): MAE={mae:.3e}, RMSE={rmse:.3e}, "
                f"predictions=[{preds_np.min():.3e}, {preds_np.max():.3e}], std ={preds_np.std():.3e}; "
                f"targets=[{tgts_np.min():.3e}, {tgts_np.max():.3e}]"
            )
            logging.info(f"   📊 Per-epoch Spearman: {corr:.3f} (p={pval:.7f})")

            for key, buf in buckets.items():
                preds, tgts = buf["preds"], buf["tgts"]
                if len(preds) >= 10 and len(tgts) >= 10:
                    try:
                        sp, sp_p = spearmanr(preds, tgts)
                        kt, kt_p = kendalltau(preds, tgts, variant="b")
                        logging.info(
                            f"   🔎 Bucket {key}: Spearman={sp:.3f} (p={sp_p:.1e}), Kendallτ={kt:.3f} (p={kt_p:.1e})"
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

        # Warmup and LR scheduling (gate plateau during warmup; step on epoch MAE)
        old_lr = trainer.optimizer.param_groups[0]["lr"]

        # Use validation loss for scheduler
        scheduler_metric = avg_val_loss

        if hasattr(trainer, "warmup_scheduler") and epoch < warmup_epochs:
            trainer.warmup_scheduler.step()
            new_lr = trainer.optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                logging.info(f"   🔼 Warmup LR: {old_lr:.2e} → {new_lr:.2e}")
        else:
            if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                logging.info(
                    f"📊 LR Scheduler: ReduceLROnPlateau monitoring metric={scheduler_metric:.6e}"
                )
                trainer.scheduler.step(scheduler_metric)
                new_lr = trainer.optimizer.param_groups[0]["lr"]
                if new_lr < old_lr:
                    logging.info(
                        f"🔻 Learning rate REDUCED: {old_lr:.2e} → {new_lr:.2e} (patience triggered!)"
                    )
                    patience_counter = 0  # reset early-stopping patience on LR drop
                else:
                    num_bad_epochs = trainer.scheduler.num_bad_epochs
                    patience_sched = trainer.scheduler.patience
                    logging.info(
                        f"   ✅ LR unchanged: {old_lr:.2e} (bad epochs: {num_bad_epochs}/{patience_sched})"
                    )
            else:
                logging.info(f"📊 LR Scheduler: {type(trainer.scheduler).__name__} (step-based)")
                trainer.scheduler.step()
                new_lr = trainer.optimizer.param_groups[0]["lr"]
                if new_lr != old_lr:
                    logging.info(f"🔻 Learning rate changed: {old_lr:.2e} → {new_lr:.2e}")

        # Exit affine mode after the configured short window
        if affine_mode:
            affine_epochs_left -= 1
            if affine_epochs_left <= 0:
                logging.info("✅ Exiting affine calibration; restoring full training")
                trainer.set_affine_calibration(enable=False)
                affine_mode = False

        # Save best model / early stopping (based on val loss for continuity)
        # Inside the training loop, after validation, when model improves:
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
                
                # 🔧 ARCHITECTURE METADATA (CRITICAL FOR RESUMPTION)
                "model_architecture": {
                    "use_strain": model.use_strain,
                    "use_edge_conditioning": model.use_edge_conditioning,
                    "use_transformer_encoder": model.use_transformer_encoder,
                    "n_edge_types": model.n_edge_types,
                    # ✅ Track actual encoder type (Whisper or lightweight)
                    "strain_encoder_type": (
                        "whisper" 
                        if (hasattr(model, 'strain_encoder') and 
                            hasattr(model.strain_encoder, 'use_whisper') and 
                            model.strain_encoder.use_whisper) 
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
                
                # Training curves data
                "training_data": {
                    "train_losses": training_metrics.get("train_losses", []),
                    "val_losses": training_metrics.get("val_losses", []),
                    "train_ranking_losses": training_metrics.get("train_ranking_losses", []),
                    "train_priority_losses": training_metrics.get("train_priority_losses", []),
                    "val_ranking_losses": training_metrics.get("val_ranking_losses", []),
                    "val_priority_losses": training_metrics.get("val_priority_losses", []),
                    "grad_norms": training_metrics.get("grad_norms", []),
                    "learning_rates": training_metrics.get("learning_rates", []),
                    "val_spearman": training_metrics.get("val_spearman", []),
                },
            }
            
            torch.save(checkpoint, output_dir / "priority_net_best.pth")
            logging.info(
                f"✅ Best model saved (val_loss: {avg_val_loss:.2e}, improvement: {improvement:.2e})"
            )

        else:
            patience_counter += 1
            logging.info(
                f"   No improvement for {patience_counter} epochs ({patience - patience_counter} remaining)"
            )
            if patience_counter >= patience:
                logging.info(f"⏹️  Early stopping at epoch {epoch+1}")
                break

    logging.info(
        f"🎉 Training completed! Best epoch: {training_metrics['best_epoch']}, Best val loss: {best_val_loss:.2e}"
    )
    return training_metrics


def create_data_splits(
    scenarios: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create stratified train/validation/test splits"""

    logging.info(
        f" Creating scenario splits: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test"
    )

    # First split: train vs (val + test)
    train_scenarios, temp_scenarios = train_test_split(
        scenarios, test_size=(val_ratio + test_ratio), random_state=42
    )

    # Second split: val vs test
    val_scenarios, test_scenarios = train_test_split(
        temp_scenarios, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42
    )

    logging.info(f" Scenario splits created:")
    logging.info(f"   Train: {len(train_scenarios)} scenarios")
    logging.info(f"   Validation: {len(val_scenarios)} scenarios")
    logging.info(f"   Test: {len(test_scenarios)} scenarios")

    return train_scenarios, val_scenarios, test_scenarios


def plot_enhanced_training_curves(training_metrics: Dict, output_dir: Path):
    """Plot comprehensive training curves with warmup phase highlighted."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    warmup_epochs = training_metrics.get("warmup_epochs", 10)
    best_epoch = training_metrics.get("best_epoch", 0)

    # Extract commonly used sequences with safe defaults
    epochs = training_metrics.get(
        "epochs", list(range(len(training_metrics.get("val_losses", []))))
    )
    lrs = training_metrics.get("lrs", None)
    val_losses = training_metrics.get("val_losses", training_metrics.get("val_losses", []))

    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, val_losses, label="Val", linewidth=2)
    ax.axvline(warmup_epochs, color="gray", linestyle="--", alpha=0.5, label="Warmup End")
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5, label="Best Model")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ranking Loss
    ax = axes[0, 1]
    ax.plot(epochs, training_metrics.get("val_ranking_losses", []), label="Val", linewidth=2)
    ax.axvline(warmup_epochs, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ranking Loss")
    ax.set_title("Ranking Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Priority Loss
    ax = axes[0, 2]
    ax.plot(epochs, training_metrics.get("val_priority_losses", []), label="Val", linewidth=2)
    ax.axvline(warmup_epochs, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Priority Loss (MSE)")
    ax.set_title("Priority Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient Norms (training stability)
    ax = axes[1, 0]
    grad_norms = training_metrics.get("grad_norms", [])
    if grad_norms:
        ax.plot(epochs, grad_norms, linewidth=2, color="purple")
    ax.axvline(warmup_epochs, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm (Training Stability)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Learning Rate Schedule
    ax = axes[1, 1]
    if lrs:
        ax.plot(epochs, lrs, linewidth=2, color="orange")
        ax.axvline(warmup_epochs, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
        try:
            ax.axhspan(lrs[0] * 0.1, lrs[0], alpha=0.2, color="yellow", label="Warmup Range")
        except Exception:
            pass
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation Loss (Zoomed)
    ax = axes[1, 2]
    ax.plot(epochs, val_losses, linewidth=2, color="red")
    ax.axvline(warmup_epochs, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    if val_losses:
        ax.legend([f"Best: {min(val_losses):.4f}"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss (Zoomed)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.close()


def evaluate_priority_net(
    model, dataset, split_name="test", debug_plots=False, out_dir="outputs/priority_net"
):
    """
    Enhanced evaluation for PriorityNet with proper model forward pass.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs(out_dir, exist_ok=True)

    logging.info(f"🔍 Evaluating PriorityNet on {split_name} set...")
    start_time = time.time()

    model.eval()
    device = next(model.parameters()).device

    # Results tracking
    correlations = []
    kendalls_all = []
    spearmans_all = []
    precisions = []
    lens = []
    pairwise_accuracies = []  # NEW: Track pairwise ranking accuracy

    successful_evaluations = 0
    total_multidet = 0
    total_scenarios = 0

    with torch.no_grad():
        for item_idx, item in enumerate(tqdm(dataset, desc=f"Evaluating {split_name}")):
            try:
                detections = item["detections"]
                true_priorities = item["priorities"]
                total_scenarios += 1

                # Skip single detection (can't compute correlation)
                if len(detections) < 2:
                    continue

                total_multidet += 1

                # CORRECT: Use full model forward pass with edge_type_id and strain segments
                try:
                    edge_type_id = item.get("edge_type_id", None)
                    if edge_type_id is not None:
                        edge_ids = torch.full((len(detections),), edge_type_id, dtype=torch.long)
                    else:
                        edge_ids = None
                    
                    # Extract strain segments from detector_data
                    strain_segments = None
                    detector_data = item.get("detector_data", {})
                    if detector_data:
                        strain_tensor = assemble_detector_strains(detector_data)
                        if strain_tensor is not None:
                            # Add batch dimension: [n_detectors, time] -> [1, n_detectors, time]
                            strain_segments = strain_tensor.unsqueeze(0).to(device)
                    
                    pred_priorities, uncertainties = model(
                        detections, 
                        strain_segments=strain_segments,
                        edge_type_ids=edge_ids
                    )
                except Exception as e:
                    if item_idx < 5:
                        logging.error(f"Forward pass failed at {item_idx}: {e}")
                    continue

                # Convert predictions to numpy
                try:
                    if isinstance(pred_priorities, torch.Tensor):
                        pred_priorities = pred_priorities.detach().cpu().numpy()
                    else:
                        pred_priorities = np.asarray(pred_priorities, dtype=np.float64)

                    if isinstance(true_priorities, torch.Tensor):
                        true_priorities = true_priorities.detach().cpu().numpy()
                    else:
                        true_priorities = np.asarray(true_priorities, dtype=np.float64)
                except Exception as e:
                    if item_idx < 5:
                        logging.error(f"Array conversion failed at {item_idx}: {e}")
                    continue

                # Align lengths
                m = min(len(pred_priorities), len(true_priorities))
                if m < 2:
                    continue

                pred_priorities = pred_priorities[:m].astype(np.float64, copy=False)
                true_priorities = true_priorities[:m].astype(np.float64, copy=False)

                # Validate arrays
                if not (np.isfinite(pred_priorities).all() and np.isfinite(true_priorities).all()):
                    continue

                # Skip degenerate cases (all equal priorities)
                if np.allclose(true_priorities, true_priorities[0]) or np.allclose(
                    pred_priorities, pred_priorities[0]
                ):
                    if item_idx < 5:
                        logging.debug(f"Degenerate equal-priority scenario {item_idx}, skipping")
                    continue

                # Skip zero-variance cases
                if np.std(pred_priorities) < 1e-8 or np.std(true_priorities) < 1e-8:
                    if item_idx < 5:
                        logging.debug(f"No variation in scenario {item_idx}")
                    continue

                # ========================================================================
                # PAIRWISE RANKING ACCURACY (NEW METRIC - Nov 16, 2025)
                # ========================================================================
                # Measure what fraction of pairwise rankings are correct
                # This is more robust than correlation for hard scenarios
                n_correct_pairs = 0
                n_total_pairs = 0
                
                for i in range(len(detections)):
                    for j in range(i + 1, len(detections)):
                        # Ground truth ordering
                        true_order = true_priorities[i] > true_priorities[j]
                        # Predicted ordering
                        pred_order = pred_priorities[i] > pred_priorities[j]
                        
                        if true_order == pred_order:
                            n_correct_pairs += 1
                        n_total_pairs += 1
                
                pairwise_accuracy = n_correct_pairs / max(1, n_total_pairs)

                # Z-score normalization per scenario
                t_mean, t_std = true_priorities.mean(), true_priorities.std()
                p_mean, p_std = pred_priorities.mean(), pred_priorities.std()

                t_z = (true_priorities - t_mean) / (t_std + 1e-8)
                p_z = (pred_priorities - p_mean) / (p_std + 1e-8)

                # Compute correlations
                try:
                    sp_corr, _ = spearmanr(t_z, p_z)
                    kd_corr, _ = kendalltau(t_z, p_z)

                    spearmans_all.append(float(sp_corr) if np.isfinite(sp_corr) else 0.0)
                    kendalls_all.append(float(kd_corr) if np.isfinite(kd_corr) else 0.0)

                    # Use Kendall for m<3, Spearman for m>=3
                    if m < 3:
                        corr_val = float(kd_corr) if np.isfinite(kd_corr) else 0.0
                    else:
                        corr_val = float(sp_corr) if np.isfinite(sp_corr) else 0.0

                    # Fallback to rank correlation if needed
                    if not np.isfinite(corr_val):
                        true_ranks = np.argsort(np.argsort(t_z))
                        pred_ranks = np.argsort(np.argsort(p_z))
                        corr_val = float(np.corrcoef(true_ranks, pred_ranks)[0, 1])
                        if not np.isfinite(corr_val):
                            corr_val = 0.0

                    correlations.append(corr_val)
                    pairwise_accuracies.append(pairwise_accuracy)  # NEW: Track pairwise
                    lens.append(m)

                except Exception as e:
                    if item_idx < 5:
                        logging.error(f"Correlation failed at {item_idx}: {e}")
                    continue

                # Precision@k
                k = min(3, m)
                true_topk = set(np.argsort(true_priorities)[-k:])
                pred_topk = set(np.argsort(pred_priorities)[-k:])
                precision = len(true_topk & pred_topk) / k
                precisions.append(float(precision))

                successful_evaluations += 1

                # Debug output for first few scenarios
                if successful_evaluations <= 5:
                    logging.debug(
                        f"✅ Scenario {item_idx}: m={m}, corr={corr_val:.3f}, prec={precision:.3f}"
                    )
                    logging.debug(f"   True: {true_priorities[:min(3,m)]}")
                    logging.debug(f"   Pred: {pred_priorities[:min(3,m)]}")

            except Exception as e:
                if item_idx < 5:
                    logging.error(f"General error at {item_idx}: {e}")
                    logging.error(traceback.format_exc())
                continue

    # Compile results
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
        corrs_arr = np.asarray(correlations, dtype=np.float64)
        prec_arr = np.asarray(precisions, dtype=np.float64)
        lens_arr = np.asarray(lens, dtype=np.int32)
        spears_arr = np.asarray(spearmans_all, dtype=np.float64) if spearmans_all else np.array([])
        kend_arr = np.asarray(kendalls_all, dtype=np.float64) if kendalls_all else np.array([])
        pairwise_arr = np.asarray(pairwise_accuracies, dtype=np.float64) if pairwise_accuracies else np.array([])

        results.update(
            {
                "avg_correlation": float(np.mean(corrs_arr)),
                "std_correlation": float(np.std(corrs_arr)),
                "median_correlation": float(np.median(corrs_arr)),
                "min_correlation": float(np.min(corrs_arr)),
                "max_correlation": float(np.max(corrs_arr)),
                "fraction_positive_corr": float(np.mean(corrs_arr > 0.0)),
                "avg_spearman": (
                    float(np.nanmean(spears_arr[lens_arr >= 3])) if spears_arr.size else 0.0
                ),
                "avg_kendall": float(np.nanmean(kend_arr[lens_arr < 3])) if kend_arr.size else 0.0,
                "avg_topk_precision": float(np.mean(prec_arr)),
                "std_topk_precision": float(np.std(prec_arr)),
                "avg_pairwise_accuracy": float(np.mean(pairwise_arr)) if pairwise_arr.size else 0.0,
                "std_pairwise_accuracy": float(np.std(pairwise_arr)) if pairwise_arr.size else 0.0,
            }
        )

        # Save metrics
        try:
            np.save(os.path.join(out_dir, f"{split_name}_correlations.npy"), corrs_arr)
            np.save(os.path.join(out_dir, f"{split_name}_precisions.npy"), prec_arr)
        except Exception as e:
            logging.debug(f"Metric dump failed: {e}")

        logging.info(
            f"{split_name.upper()} evaluation: {successful_evaluations}/{total_multidet} multi-detection scenarios"
        )
        logging.info(
            f"   Total scenarios: {total_scenarios} | Success: {results['success_rate']:.3f} | Failure: {results['failure_rate']:.3f}"
        )
        logging.info(
            f"   Corr (selected): {results['avg_correlation']:.3f} ± {results['std_correlation']:.3f}"
        )
        logging.info(
            f"   Spearman(avg, m≥3): {results['avg_spearman']:.3f} | Kendall(avg, m<3): {results['avg_kendall']:.3f}"
        )
        logging.info(
            f"   Pairwise Accuracy: {results['avg_pairwise_accuracy']:.3f} ± {results['std_pairwise_accuracy']:.3f}"
        )
        logging.info(
            f"   Precision@3: {results['avg_topk_precision']:.3f} | Time: {results['eval_time_sec']:.2f}s"
        )
    else:
        logging.warning("❌ No successful correlations computed")
        logging.warning(f"   Multi-detection scenarios found: {total_multidet}")
        logging.warning("   Check: Are priorities being predicted correctly?")

    return results


# def load_checkpoint(
#     checkpoint_path: Optional[str], config, device=None
# ) -> Optional[Dict[str, Any]]:
#     """
#     Load a PriorityNet training checkpoint safely (backward-compatible).

#     Tolerates missing or unexpected keys caused by architectural evolution
#     (e.g., new heads like overlap_head) by filtering/merging the model state
#     and loading with strict=False. Optimizer and scheduler states are restored
#     when present. Designed for unattended (nohup) resume.

#     Args:
#         checkpoint_path: Path to checkpoint file (or None to start fresh).
#         config: Training configuration (dict or object) used to rebuild model/trainer.
#         device: Optional torch.device; inferred from CUDA availability by default.

#     Returns:
#         Dict with resume state: model, trainer, start_epoch, best_val_loss, patience_counter,
#         training_metrics, checkpoint_path; or None if no checkpoint or on failure.
#     """
#     # Check if checkpoint exists
#     if checkpoint_path is None or not Path(checkpoint_path).exists():
#         logging.info("No checkpoint found, starting fresh training")
#         return None

#     logging.info(f"📂 Found checkpoint: {checkpoint_path}")
#     logging.info(f"🔄 Auto-resuming training...")

#     try:
#         if device is None:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

#         # Helper to extract config value (same logic as cfg_get in PriorityNet)
#         def get_config_value(cfg, key, default):
#             """Extract config value from top level or nested priority_net section."""
#             if isinstance(cfg, dict):
#                 if key in cfg:
#                     return cfg[key]
#                 if "priority_net" in cfg and isinstance(cfg["priority_net"], dict):
#                     if key in cfg["priority_net"]:
#                         return cfg["priority_net"][key]
#                 return default
#             else:
#                 if hasattr(cfg, key):
#                     return getattr(cfg, key)
#                 if hasattr(cfg, "priority_net"):
#                     priority_net_cfg = getattr(cfg, "priority_net")
#                     if hasattr(priority_net_cfg, key):
#                         return getattr(priority_net_cfg, key)
#                 return default

#         # Check encoder type compatibility
#         checkpoint_encoder_type = checkpoint.get("use_transformer_encoder", False)
#         config_encoder_type = get_config_value(config, "use_transformer_encoder", False)

#         if checkpoint_encoder_type != config_encoder_type:
#             logging.warning(
#                 f"⚠️  Encoder type mismatch: checkpoint={checkpoint_encoder_type}, config={config_encoder_type}"
#             )
#             logging.warning(
#                 f"   Checkpoint expects {'Transformer' if checkpoint_encoder_type else 'CNN+BiLSTM'}, config specifies {'Transformer' if config_encoder_type else 'CNN+BiLSTM'}"
#             )
#             logging.warning(f"   Starting fresh training with config encoder type")
#             return None

#         # Extract model architecture from checkpoint (for proper restoration)
#         checkpoint_arch = checkpoint.get("model_architecture", {})
#         checkpoint_use_strain = checkpoint_arch.get("use_strain", True)
#         checkpoint_use_edge = checkpoint_arch.get("use_edge_conditioning", True)
#         checkpoint_use_transformer = checkpoint_arch.get("use_transformer_encoder", False)
#         checkpoint_n_edge_types = checkpoint_arch.get("n_edge_types", 19)

#         # Create model with checkpoint's architecture if available, else use config
#         model = PriorityNet(
#             config,
#             use_strain=checkpoint_use_strain if checkpoint_arch else True,
#             use_edge_conditioning=checkpoint_use_edge if checkpoint_arch else True,
#             n_edge_types=checkpoint_n_edge_types if checkpoint_arch else 19,
#         ).to(device)

#         # Load state dict with proper error handling
#         pretrained = checkpoint.get("model_state_dict", {})
#         try:
#             model.load_state_dict(pretrained, strict=True)
#             logging.info("✅ Model state loaded (strict=True, all keys matched)")
#         except RuntimeError as e:
#             # Strict load failed - check if it's due to architecture mismatch
#             current_sd = model.state_dict()
#             filtered = {
#                 k: v
#                 for k, v in pretrained.items()
#                 if k in current_sd and current_sd[k].shape == v.shape
#             }
#             missing = [k for k in current_sd.keys() if k not in filtered]
#             unexpected = [k for k in pretrained.keys() if k not in current_sd]

#             if missing or unexpected:
#                 logging.warning(f"⚠️ Architecture mismatch detected during checkpoint load")
#                 if missing:
#                     logging.warning(
#                         f"   Missing keys (will use init): {missing[:5]}{' ...' if len(missing)>5 else ''}"
#                     )
#                 if unexpected:
#                     logging.warning(
#                         f"   Unexpected keys (will ignore): {unexpected[:5]}{' ...' if len(unexpected)>5 else ''}"
#                     )

#                 # Load only matching weights
#                 model.load_state_dict(filtered, strict=False)
#                 logging.info(f"✅ Loaded {len(filtered)}/{len(pretrained)} checkpoint weights")
#             else:
#                 raise

#         had_shape_mismatches = (
#             len(missing) > 0 or len(unexpected) > 0 if "missing" in locals() else False
#         )

#         # Create trainer
#         trainer = PriorityNetTrainer(model, config)

#         # Load optimizer and scheduler states when available
#         # BUT: Reset optimizer if model architecture changed (shape mismatches detected)
#         if (
#             "optimizer_state_dict" in checkpoint
#             and trainer.optimizer is not None
#             and not had_shape_mismatches
#         ):
#             try:
#                 trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#                 logging.info("✅ Optimizer state restored")
#             except Exception as e:
#                 logging.warning(f"Optimizer state not loaded: {e}")
#         elif had_shape_mismatches:
#             logging.info(
#                 "🔄 Optimizer reset due to model architecture changes (shape mismatches detected)"
#             )
#             # Optimizer is already initialized fresh in PriorityNetTrainer.__init__()

#         if "scheduler_state_dict" in checkpoint and trainer.scheduler is not None:
#             try:
#                 trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
#             except Exception as e:
#                 logging.warning(f"Scheduler state not loaded: {e}")

#         # Extract training state from checkpoint or config
#         checkpoint_training_config = checkpoint.get("training_config", {})
#         start_epoch = checkpoint.get("epoch", 0) + 1
#         best_val_loss = checkpoint.get("best_val_loss", float("inf"))
#         patience_counter = checkpoint_training_config.get("patience_counter", 0)

#         # ✅ Restore all training data from checkpoint
#         checkpoint_training_data = checkpoint.get("training_data", {})
#         training_metrics = checkpoint.get(
#             "training_metrics",
#             {
#                 "train_losses": checkpoint_training_data.get("train_losses", []),
#                 "val_losses": checkpoint_training_data.get("val_losses", []),
#                 "train_ranking_losses": checkpoint_training_data.get("train_ranking_losses", []),
#                 "val_ranking_losses": checkpoint_training_data.get("val_ranking_losses", []),
#                 "train_priority_losses": checkpoint_training_data.get("train_priority_losses", []),
#                 "val_priority_losses": checkpoint_training_data.get("val_priority_losses", []),
#                 "grad_norms": checkpoint_training_data.get("grad_norms", []),
#                 "learning_rates": checkpoint_training_data.get("learning_rates", []),
#                 "val_spearman": checkpoint_training_data.get("val_spearman", []),
#                 "warmup_epochs": get_config_value(config, "warmup_epochs", 5),
#                 "best_epoch": checkpoint_training_config.get("best_epoch", 0),
#                 "epochs_completed": start_epoch - 1,
#             },
#         )

#         logging.info(f"✅ Checkpoint loaded successfully!")
#         logging.info(f"   📍 Resuming from epoch {start_epoch}")
#         logging.info(f"   📊 Best validation loss: {best_val_loss:.2e}")
#         patience_val = get_config_value(config, "patience", 30)
#         logging.info(f"   ⏱️  Patience counter: {patience_counter}/{patience_val}")

#         return {
#             "model": model,
#             "trainer": trainer,
#             "start_epoch": start_epoch,
#             "best_val_loss": best_val_loss,
#             "patience_counter": patience_counter,
#             "training_metrics": training_metrics,
#             "checkpoint_path": checkpoint_path,
#         }

#     except Exception as e:
#         logging.error(f"❌ Failed to load checkpoint: {e}")
#         logging.error(f"Traceback: {traceback.format_exc()}")
#         logging.warning("⚠️ Starting from scratch instead")
#         return None


def load_checkpoint(checkpoint_path: Optional[str], config, device=None) -> Optional[Dict[str, Any]]:
    """
    Load a PriorityNet training checkpoint with full validation and compatibility.

    Validates:
    1. Encoder type matches (Transformer vs CNN+BiLSTM)
    2. Architecture components compatible
    3. State dict keys/shapes match
    4. Training metrics structure valid

    Args:
        checkpoint_path: Path to checkpoint file (or None to start fresh).
        config: Training configuration (dict or object) used to rebuild model/trainer.
        device: Optional torch.device; inferred from CUDA availability by default.

    Returns:
        Dict with resume state: model, trainer, start_epoch, best_val_loss, patience_counter,
        training_metrics, checkpoint_path; or None if no checkpoint or on failure.
    """
    # 1. CHECK EXISTENCE
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        logging.info("No checkpoint found, starting fresh training")
        return None

    logging.info(f"📂 Found checkpoint: {checkpoint_path}")
    logging.info(f"🔄 Validating checkpoint...")

    try:
        # 2. LOAD CHECKPOINT
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 3. CONFIG ACCESSOR (handles nested priority_net)
        def cfg_get(cfg, key, default):
            """Extract config value from top or nested priority_net section."""
            if isinstance(cfg, dict):
                if key in cfg:
                    return cfg[key]
                if "priority_net" in cfg and isinstance(cfg["priority_net"], dict):
                    if key in cfg["priority_net"]:
                        return cfg["priority_net"][key]
                return default
            else:
                if hasattr(cfg, key):
                    return getattr(cfg, key)
                if hasattr(cfg, "priority_net"):
                    pn_cfg = getattr(cfg, "priority_net")
                    if hasattr(pn_cfg, key):
                        return getattr(pn_cfg, key)
                return default

        # 4. ENCODER TYPE COMPATIBILITY CHECK
        checkpoint_encoder_type = checkpoint.get("use_transformer_encoder", False)
        config_encoder_type = cfg_get(config, "use_transformer_encoder", False)

        # ✅ NEW: Check actual encoder type used (Whisper vs Lightweight)
        model_arch = checkpoint.get("model_architecture", {})
        actual_encoder_type = model_arch.get("strain_encoder_type", "lightweight")
        
        logging.info(
            f"📊 Encoder architecture: config={config_encoder_type}, "
            f"checkpoint={checkpoint_encoder_type}, actual={actual_encoder_type}"
        )

        if checkpoint_encoder_type != config_encoder_type:
            logging.warning(
                f"⚠️  Encoder use_transformer_encoder flag mismatch: "
                f"checkpoint={checkpoint_encoder_type}, config={config_encoder_type}"
            )
            logging.warning(
                f"   Checkpoint expects {'Transformer' if checkpoint_encoder_type else 'CNN+BiLSTM'}, "
                f"config specifies {'Transformer' if config_encoder_type else 'CNN+BiLSTM'}"
            )
            logging.warning(f"   Cannot safely resume with mismatched architecture. Starting fresh.")
            return None

        # 5. EXTRACT ARCHITECTURE (with fallbacks)
        checkpoint_arch = checkpoint.get("model_architecture", {})
        checkpoint_use_strain = checkpoint_arch.get("use_strain", True)
        checkpoint_use_edge = checkpoint_arch.get("use_edge_conditioning", True)
        checkpoint_n_edge_types = checkpoint_arch.get("n_edge_types", 19)

        # 6. CREATE MODEL WITH CHECKPOINT ARCHITECTURE
        model = PriorityNet(
            config,
            use_strain=checkpoint_use_strain,
            use_edge_conditioning=checkpoint_use_edge,
            n_edge_types=checkpoint_n_edge_types,
        ).to(device)

        # 7. STATE DICT VALIDATION & LOADING
        pretrained_state = checkpoint.get("model_state_dict", {})
        current_state = model.state_dict()

        # Validate keys/shapes
        missing = []
        unexpected = []
        shape_mismatches = []

        for k in current_state.keys():
            if k not in pretrained_state:
                missing.append(k)
            elif current_state[k].shape != pretrained_state[k].shape:
                shape_mismatches.append((k, current_state[k].shape, pretrained_state[k].shape))

        for k in pretrained_state.keys():
            if k not in current_state:
                unexpected.append(k)

        # Log detailed mismatch info
        if missing or unexpected or shape_mismatches:
            logging.warning(f"⚠️  State dict mismatch detected during checkpoint load")
            if shape_mismatches:
                logging.warning(f"   Shape mismatches ({len(shape_mismatches)}):")
                for k, exp, got in shape_mismatches[:3]:  # Show first 3
                    logging.warning(f"      {k}: expected {exp}, got {got}")
                if len(shape_mismatches) > 3:
                    logging.warning(f"      ... and {len(shape_mismatches)-3} more")

            if missing and len(missing) <= 10:
                logging.warning(f"   Missing keys: {missing}")
            elif missing:
                logging.warning(f"   Missing {len(missing)} keys (too many to list)")

            if unexpected and len(unexpected) <= 10:
                logging.warning(f"   Unexpected keys: {unexpected}")
            elif unexpected:
                logging.warning(f"   Unexpected {len(unexpected)} keys (too many to list)")

        # Attempt strict load first
        had_mismatches = False
        try:
            model.load_state_dict(pretrained_state, strict=True)
            logging.info("✅ Model state loaded (strict=True, perfect match)")
        except RuntimeError as e:
            # Strict load failed, try non-strict with key filtering
            if shape_mismatches or missing or unexpected:
                # Only load keys that match exactly
                compatible_keys = {
                    k: v
                    for k, v in pretrained_state.items()
                    if k in current_state and current_state[k].shape == v.shape
                }
                model.load_state_dict(compatible_keys, strict=False)
                loaded_count = len(compatible_keys)
                total_count = len(pretrained_state)
                logging.warning(
                    f"⚠️  Loaded {loaded_count}/{total_count} compatible state dict keys"
                )
                logging.warning(f"   Architecture changes detected but loading what we can")
                had_mismatches = True
            else:
                # Unexpected error
                raise

        # 8. CREATE TRAINER
        trainer = PriorityNetTrainer(model, config)

        # 9. RESTORE OPTIMIZER & SCHEDULER (skip if architecture changed)
        if not had_mismatches:
            if "optimizer_state_dict" in checkpoint and trainer.optimizer is not None:
                try:
                    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    logging.info("✅ Optimizer state restored")
                except Exception as e:
                    logging.warning(f"⚠️  Could not restore optimizer state: {e}")

            if "scheduler_state_dict" in checkpoint and trainer.scheduler is not None:
                try:
                    trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    logging.info("✅ Scheduler state restored")
                except Exception as e:
                    logging.warning(f"⚠️  Could not restore scheduler state: {e}")
        else:
            logging.info("🔄 Optimizer/scheduler reset due to architecture changes")

        # 10. RESTORE TRAINING STATE
        checkpoint_training_config = checkpoint.get("training_config", {})
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        patience_counter = checkpoint_training_config.get("patience_counter", 0)

        # 11. RESTORE TRAINING METRICS (with full backward compat)
        checkpoint_metrics = checkpoint.get("training_metrics", {})
        if not checkpoint_metrics:
            # Fallback to training_data if training_metrics not present
            checkpoint_metrics = {
                "train_losses": checkpoint.get("training_data", {}).get("train_losses", []),
                "val_losses": checkpoint.get("training_data", {}).get("val_losses", []),
                "train_ranking_losses": checkpoint.get("training_data", {}).get(
                    "train_ranking_losses", []
                ),
                "val_ranking_losses": checkpoint.get("training_data", {}).get(
                    "val_ranking_losses", []
                ),
                "train_priority_losses": checkpoint.get("training_data", {}).get(
                    "train_priority_losses", []
                ),
                "val_priority_losses": checkpoint.get("training_data", {}).get(
                    "val_priority_losses", []
                ),
                "grad_norms": checkpoint.get("training_data", {}).get("grad_norms", []),
                "learning_rates": checkpoint.get("training_data", {}).get("learning_rates", []),
                "val_spearman": checkpoint.get("training_data", {}).get("val_spearman", []),
                "best_epoch": checkpoint_training_config.get("best_epoch", 0),
                "epochs_completed": start_epoch - 1,
            }
        else:
            # Ensure all expected keys present
            expected_keys = [
                "train_losses",
                "val_losses",
                "train_ranking_losses",
                "val_ranking_losses",
                "train_priority_losses",
                "val_priority_losses",
                "grad_norms",
                "learning_rates",
                "val_spearman",
                "best_epoch",
                "epochs_completed",
            ]
            for key in expected_keys:
                if key not in checkpoint_metrics:
                    checkpoint_metrics[key] = checkpoint.get("training_data", {}).get(key, [])

        # 12. LOG RESUMPTION INFO
        logging.info(f"✅ Checkpoint loaded successfully!")
        logging.info(f"   📍 Resuming from epoch {start_epoch}")
        logging.info(f"   📊 Best validation loss: {best_val_loss:.2e}")
        patience_val = cfg_get(config, "patience", 30)
        logging.info(f"   ⏱️  Patience counter: {patience_counter}/{patience_val}")
        logging.info(
            f"   🏗️  Architecture: use_strain={checkpoint_use_strain}, "
            f"use_edge={checkpoint_use_edge}, use_transformer={checkpoint_encoder_type}"
        )

        return {
            "model": model,
            "trainer": trainer,
            "start_epoch": start_epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "training_metrics": checkpoint_metrics,
            "checkpoint_path": checkpoint_path,
        }

    except Exception as e:
        logging.error(f"❌ Failed to load checkpoint: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        logging.warning("⚠️ Starting from scratch instead")
        return None


def main():
    """Main training function with automatic checkpoint resumption."""

    # ========================================================================
    # ARGUMENT PARSING
    # ========================================================================
    parser = argparse.ArgumentParser(
        description="Complete Phase 2: PriorityNet with Integrated Dataset Loading"
    )
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--dataset_path", required=True, help="Path to newDataset directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum samples per split (None = all)"
    )
    parser.add_argument(
        "--create_overlaps", action="store_true", help="Create artificial overlap scenarios"
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training scenarios ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation scenarios ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test scenarios ratio")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint file to resume from"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch even if checkpoint exists",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(args.verbose)
    logging.info("🚀 Starting Complete Phase 2: PriorityNet with Integrated Dataset Loading")

    # ========================================================================
    # VALIDATE ARGUMENTS
    # ========================================================================
    if not Path(args.dataset_path).exists():
        logging.error(f"❌ Dataset path not found: {args.dataset_path}")
        return

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        logging.error("❌ Split ratios must sum to 1.0")
        return

    # ========================================================================
    # LOAD CONFIGURATION
    # ========================================================================
    try:
        config = load_enhanced_config(Path(args.config))
        log_config(config, logging.getLogger())
        validate_config(config, logging.getLogger())
    except Exception as e:
        logging.error(f"❌ Failed to load config: {e}")
        raise

    # ========================================================================
    # LOAD GW DATASETS
    # ========================================================================
    logging.info("📂 Loading GW datasets...")

    train_loader = ChunkedGWDataLoader(
        args.dataset_path, split="train", max_samples=args.max_samples
    )
    val_loader = ChunkedGWDataLoader(
        args.dataset_path, split="validation", max_samples=args.max_samples
    )
    test_loader = ChunkedGWDataLoader(args.dataset_path, split="test", max_samples=args.max_samples)

    # ========================================================================
    # CONVERT TO PRIORITYNET SCENARIOS
    # ========================================================================
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

    # ========================================================================
    # CREATE PRIORITYNET DATASETS
    # ========================================================================
    train_dataset = PriorityNetDataset(train_scenarios, "train")
    val_dataset = PriorityNetDataset(val_scenarios, "validation")
    test_dataset = PriorityNetDataset(test_scenarios, "test")

    if len(train_dataset) == 0:
        logging.error("❌ No valid training scenarios for PriorityNet")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # MULTI-DETECTION DIAGNOSTIC
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("MULTI-DETECTION DIAGNOSTIC")
    logging.info("=" * 80)

    multi_det_count = 0
    single_det_count = 0
    detection_counts = {}

    for item in train_dataset:
        n_det = len(item.get("detections", []))
        detection_counts[n_det] = detection_counts.get(n_det, 0) + 1

        if n_det >= 2:
            multi_det_count += 1
        elif n_det == 1:
            single_det_count += 1

    logging.info(f"\n📊 Detection Count Distribution:")
    for n_det in sorted(detection_counts.keys()):
        count = detection_counts[n_det]
        pct = count / len(train_dataset) * 100
        logging.info(f"   {n_det} detection(s): {count} ({pct:.1f}%)")

    logging.info(f"\n📈 Summary:")
    logging.info(
        f"   Multi-detection (2+): {multi_det_count} ({multi_det_count/len(train_dataset)*100:.1f}%)"
    )
    logging.info(
        f"   Single-detection (1): {single_det_count} ({single_det_count/len(train_dataset)*100:.1f}%)"
    )

    if multi_det_count < len(train_dataset) * 0.1:
        logging.error(
            f"🔴 CRITICAL: Only {multi_det_count/len(train_dataset)*100:.1f}% multi-detection!"
        )
        logging.error("   Ranking loss will be ZERO most of the time.")
        logging.error("   STOP and fix before training!")
    elif multi_det_count < len(train_dataset) * 0.3:
        logging.warning(
            f"🟡 WARNING: Only {multi_det_count/len(train_dataset)*100:.1f}% multi-detection."
        )
        logging.warning("   Ranking supervision will be limited but training can proceed.")
    else:
        logging.info(
            f"\n🟢 Good! {multi_det_count/len(train_dataset)*100:.1f}% multi-detection scenarios."
        )

    logging.info("=" * 80 + "\n")

    # ========================================================================
    # ✅ AUTOMATIC CHECKPOINT HANDLING (Perfect for nohup)
    # ========================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine checkpoint path
    checkpoint_path = None

    if args.no_resume:
        # User explicitly wants to start fresh
        logging.info("🔄 --no-resume flag set, starting fresh training")
        checkpoint_path = None
    elif args.resume:
        # User specified explicit checkpoint path
        checkpoint_path = args.resume
        logging.info(f"🔍 Using specified checkpoint: {checkpoint_path}")
    else:
        # Check for default checkpoint
        default_checkpoint = output_dir / "priority_net_best.pth"
        if default_checkpoint.exists():
            checkpoint_path = str(default_checkpoint)
            logging.info(f"🔍 Found default checkpoint: {checkpoint_path}")

    # ✅ Load checkpoint automatically (no user prompts!)
    resume_state = load_checkpoint(checkpoint_path=checkpoint_path, config=config, device=device)

    # ========================================================================
    # 🚀 TRAINING
    # ========================================================================

    # Train Model
    training_metrics = train_priority_net_with_validation(
        config, train_dataset, val_dataset, output_dir, resume_state=resume_state
    )

    # ========================================================================
    # 📊 EVALUATION
    # ========================================================================

    # Final evaluation (skip if checkpoint loading fails due to config mismatch)
    train_results = None
    val_results = None
    test_results = None

    try:
        logging.info("📊 Loading best model for final evaluation...")
        best_checkpoint = torch.load(
            output_dir / "priority_net_best.pth", weights_only=False, map_location="cpu"
        )

        # Use saved config from checkpoint if available (ensures architecture match)
        eval_config = best_checkpoint.get("model_config", config)
        if isinstance(eval_config, dict) and "priority_net" in eval_config:
            eval_config = eval_config["priority_net"]

        model = PriorityNet(
            eval_config, use_strain=True, use_edge_conditioning=True, n_edge_types=19
        )
        model.load_state_dict(best_checkpoint["model_state_dict"], strict=False)

        # Comprehensive evaluation
        logging.info("📊 Evaluating model on all splits...")
        train_results = evaluate_priority_net(model, train_dataset, "train")
        val_results = evaluate_priority_net(model, val_dataset, "validation")
        test_results = evaluate_priority_net(model, test_dataset, "test")
    except Exception as e:
        logging.warning(f"⚠️  Skipping final evaluation due to checkpoint loading error: {e}")

    # Combine all results
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
            "train_samples": getattr(train_loader, "count", train_loader.count_samples()),
            "val_samples": getattr(val_loader, "count", val_loader.count_samples()),
            "test_samples": getattr(test_loader, "count", test_loader.count_samples()),
        },
    }

    # Save results
    with open(output_dir / "complete_results.pkl", "wb") as f:
        pickle.dump(final_results, f)

    # ========================================================================
    # 🎉 FINAL SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("🎉 COMPLETE PHASE 2 - PRIORITYNET WITH INTEGRATED DATASET LOADING")
    print("=" * 80)
    print(f"✅ TEST SET Ranking Correlation: {test_results['avg_correlation']*100:.1f}%")
    print(f"✅ VALIDATION SET Ranking Correlation: {val_results['avg_correlation']*100:.1f}%")

    if test_results["avg_correlation"] > 0.85:
        print("🟢 EXCELLENT - Production ready!")
    elif test_results["avg_correlation"] > 0.70:
        print("🟡 GOOD - Continue improvements")
    else:
        print("🔴 NEEDS WORK - Consider retraining")

    print(f"\n📊 DATASET STATISTICS:")
    print(
        f"   Training: {final_results['dataset_info']['train_samples']:,} samples → {len(train_scenarios):,} scenarios"
    )
    print(
        f"   Validation: {final_results['dataset_info']['val_samples']:,} samples → {len(val_scenarios):,} scenarios"
    )
    print(
        f"   Test: {final_results['dataset_info']['test_samples']:,} samples → {len(test_scenarios):,} scenarios"
    )

    print(f"\n💾 Results saved to: {output_dir}")
    print(f"📈 See training_curves.png for training visualization")
    print("=" * 80)


if __name__ == "__main__":
    main()