"""
GW Dataset Generator

Clean, physics-correct pipeline:
  1. Sample parameters from proper astrophysical priors (ParameterSampler)
  2. Generate per-detector colored noise (BilbyNoiseGenerator)
  3. Inject GW signal at natural amplitude — no SNR targeting (BilbySignalInjector)
  4. Measure matched-filter SNR; reject samples below min_snr
  5. Whiten injected strain (BilbyPreprocessor)
  6. Save in batches

Main class: GWDatasetGenerator
"""

import logging
import pickle
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .parameter_sampler import ParameterSampler
from .bilby_pipeline import (
    BilbyWaveformGenerator,
    BilbyNoiseGenerator,
    BilbySignalInjector,
    BilbyPreprocessor,
    get_default_psd,
)
from .io_utils import DatasetWriter, MetadataManager

_log = logging.getLogger(__name__)

# 11 parameters the NPE is trained on (geocent_time = offset in [-2,7] s)
PARAM_NAMES: List[str] = [
    "mass_1", "mass_2", "luminosity_distance",
    "ra", "dec", "theta_jn", "psi", "phase",
    "geocent_time", "a1", "a2",
]

DETECTORS: List[str] = ["H1", "L1", "V1"]


# ─────────────────────────────────────────────────────────────────────────────
# Encode edge type (kept for io_utils compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def encode_edge_type(detected: list) -> int:
    n = len([d for d in (detected or []) if d])
    return max(0, n - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _param_vector(p: Dict) -> Dict:
    """Extract the 11-parameter dict the NPE needs, plus tilt1/tilt2=0."""
    return {
        name: float(p.get(name, 0.0)) for name in PARAM_NAMES
    } | {"tilt1": 0.0, "tilt2": 0.0}


def _network_snr(per_det_snr: Dict[str, float]) -> float:
    """Coherent network SNR = sqrt(sum(SNR_i^2))."""
    return float(np.sqrt(sum(v ** 2 for v in per_det_snr.values())))


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

class GWDatasetGenerator:
    """
    Generate synthetic GW datasets for NPE training.

    Parameters
    ----------
    output_dir : str | Path
    config : dict
        Reads the following keys (all optional, with defaults):
          n_samples           : 5000
          sample_rate         : 4096
          duration            : 4.0
          detectors           : [H1, L1, V1]
          save_batch_size     : 100
          min_snr             : 6.0   – reject samples below this network SNR
          max_retries         : 20    – per-sample rejection retries
          overlap_fraction    : 0.4   – fraction of samples with ≥2 signals
          max_overlapping     : 4     – max signals per sample
          add_glitches        : false
          preprocess          : true  – whiten strain
          noise_augmentation_k: 1    – noise realizations per parameter draw
          create_splits       : true
          train_frac          : 0.80
          val_frac            : 0.10
          test_frac           : 0.10
          random_seed         : 42
          real_noise_prob     : 0.0  – fraction of samples using real GWOSC noise
          real_noise_cache_dir: data/gwosc_cache
    """

    def __init__(self, output_dir: str = "data/dataset",
                 sample_rate: int = 4096,
                 duration: float = 4.0,
                 detectors: Optional[List[str]] = None,
                 output_format: str = "pkl",
                 config: Optional[Dict] = None):
        cfg = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = int(cfg.get("sample_rate", sample_rate))
        self.duration = float(cfg.get("duration", duration))
        self.detectors = list(cfg.get("detectors", detectors or DETECTORS))
        self.output_format = output_format.lower()
        self.config = cfg

        seed = cfg.get("random_seed", 42)
        self._rng = np.random.default_rng(seed)

        # Sub-components
        self.sampler = ParameterSampler(cfg)
        self.noise_gen = BilbyNoiseGenerator(
            self.sample_rate, self.duration,
            real_noise_prob=float(cfg.get("real_noise_prob", 0.0)),
            cache_dir=cfg.get("real_noise_cache_dir", None),
        )
        self.injector = BilbySignalInjector(
            self.sample_rate, self.duration,
            f_lower=float(cfg.get("f_lower", 20.0)),
            f_ref=float(cfg.get("f_ref", 50.0)),
        )
        self.preprocessor = BilbyPreprocessor(
            self.sample_rate, self.duration,
            f_low=float(cfg.get("f_lower", 20.0)),
        )
        self.writer = DatasetWriter(str(self.output_dir), format="pkl")

        # Pre-compute default PSDs once
        self._psd_cache: Dict[str, Dict] = {
            det: get_default_psd(det) for det in self.detectors
        }

        _log.info(
            f"GWDatasetGenerator: {self.detectors}, sr={self.sample_rate} Hz, "
            f"dur={self.duration}s, output={self.output_dir}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        n_samples: Optional[int] = None,
        overlap_fraction: Optional[float] = None,
        create_splits: Optional[bool] = None,
        **kwargs,
    ) -> Dict:
        """
        Generate the full dataset.

        Returns a summary dict with generation statistics.
        """
        cfg = self.config
        n_samples = int(n_samples or cfg.get("n_samples", 5000))
        overlap_fraction = float(overlap_fraction if overlap_fraction is not None
                                 else cfg.get("overlap_fraction", 0.4))
        max_overlap = int(cfg.get("max_overlapping_signals", cfg.get("max_overlapping", 4)))
        min_snr = float(cfg.get("min_snr", 6.0))
        max_retries = int(cfg.get("max_retries", 20))
        add_glitches = bool(cfg.get("add_glitches", False))
        do_preprocess = bool(cfg.get("preprocess", True))
        k_noise = int(cfg.get("noise_augmentation_k", 1))
        save_batch = int(cfg.get("save_batch_size", 100))
        create_splits = bool(create_splits if create_splits is not None
                              else cfg.get("create_splits", True))

        _log.info(f"Generating {n_samples} samples "
                  f"(overlap_frac={overlap_fraction:.2f}, min_snr={min_snr:.1f})")

        samples: List[Dict] = []
        batch_id = 0
        stats = {
            "generated": 0, "rejected_snr": 0, "rejected_waveform": 0,
            "event_types": {}, "snr_mean": [], "n_signals_dist": {},
        }

        with tqdm(total=n_samples, desc="Generating samples") as pbar:
            param_idx = 0
            while stats["generated"] < n_samples:
                # Decide number of signals for this sample
                if self._rng.uniform() < overlap_fraction:
                    n_sig = int(self._rng.integers(2, max_overlap + 1))
                else:
                    n_sig = 1

                # Draw parameters
                param_sets = [self.sampler.sample_parameters() for _ in range(n_sig)]

                # k noise realizations per parameter set
                for k in range(k_noise):
                    if stats["generated"] >= n_samples:
                        break

                    sample, rej = self._make_sample(
                        param_sets, min_snr, add_glitches, do_preprocess,
                        seed_offset=param_idx * k_noise + k,
                    )

                    if sample is None:
                        stats["rejected_snr"] += 1
                        continue

                    sample["sample_id"] = str(uuid.uuid4())[:12]
                    samples.append(sample)
                    stats["generated"] += 1
                    et = sample.get("event_type", "unknown")
                    stats["event_types"][et] = stats["event_types"].get(et, 0) + 1
                    stats["snr_mean"].append(sample.get("network_snr", 0.0))
                    ns = sample.get("n_signals", 1)
                    stats["n_signals_dist"][ns] = stats["n_signals_dist"].get(ns, 0) + 1
                    pbar.update(1)

                    # Save batch
                    if len(samples) >= save_batch:
                        self._save_batch(samples, batch_id)
                        batch_id += 1
                        samples.clear()

                param_idx += 1

        # Flush remainder
        if samples:
            self._save_batch(samples, batch_id)

        # Create splits
        if create_splits:
            self._create_splits()

        # Save metadata
        if stats["snr_mean"]:
            stats["snr_mean_value"] = float(np.mean(stats["snr_mean"]))
        stats.pop("snr_mean", None)
        _log.info(f"Done. {stats}")
        return stats

    # ── Core sample generation ────────────────────────────────────────────────

    def _make_sample(self, param_sets: List[Dict],
                     min_snr: float,
                     add_glitches: bool,
                     do_preprocess: bool,
                     seed_offset: int = 0) -> Tuple[Optional[Dict], int]:
        """
        Generate noise, inject signals, whiten.
        Returns (sample_dict, 0) or (None, 1) if SNR below threshold.
        """
        detector_data: Dict[str, Dict] = {}
        per_det_snr: Dict[str, float] = {}

        for i, det in enumerate(self.detectors):
            psd = self._psd_cache[det]
            seed = int(self._rng.integers(0, 2 ** 31)) + seed_offset * 100 + i

            # 1. Generate noise
            noise = self.noise_gen.generate(det, psd, seed=seed)
            if add_glitches:
                noise = self.noise_gen.add_glitches(noise)

            # 2. Inject all signals; track per-detector SNR for primary signal
            if param_sets[0].get("event_type") == "noise":
                strain = noise.astype(np.float64)
                det_snr = 0.0
            else:
                strain, snrs = self.injector.inject_multiple(noise, param_sets, det, psd)
                det_snr = snrs[0] if snrs else 0.0  # primary signal SNR
            per_det_snr[det] = float(det_snr)

            # 3. Whiten
            if do_preprocess:
                whitened = self.preprocessor.preprocess(strain, psd)
            else:
                whitened = strain.astype(np.float32)

            detector_data[det] = {"strain": whitened, "snr": float(det_snr)}

        # 4. SNR gate (primary signal network SNR)
        net_snr = _network_snr(per_det_snr)
        is_noise_only = param_sets[0].get("event_type") == "noise"
        if not is_noise_only and net_snr < min_snr:
            return None, 1

        # 5. Build sample dict
        primary = param_sets[0]
        return {
            "detector_data": detector_data,
            "parameters": [_param_vector(p) for p in param_sets],
            "n_signals": len(param_sets),
            "network_snr": float(net_snr),
            "per_detector_snr": per_det_snr,
            "event_type": primary.get("event_type", "BBH"),
            "metadata": {
                "chirp_mass": float(primary.get("chirp_mass", 0.0)),
                "mass_ratio": float(primary.get("mass_ratio", 1.0)),
                "luminosity_distance": float(primary.get("luminosity_distance", 0.0)),
                "n_signals": len(param_sets),
            },
        }, 0

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _save_batch(self, samples: List[Dict], batch_id: int) -> None:
        batch_dir = self.output_dir / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        path = batch_dir / f"batch_{batch_id:05d}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"samples": samples, "batch_id": batch_id,
                         "n_samples": len(samples)}, f)
        _log.debug(f"Saved batch {batch_id}: {len(samples)} samples → {path.name}")

    def _create_splits(self) -> None:
        """Distribute existing batches into train/val/test splits."""
        train_frac = float(self.config.get("train_frac", 0.80))
        val_frac = float(self.config.get("val_frac", 0.10))
        batch_dir = self.output_dir / "batches"
        batch_files = sorted(batch_dir.glob("batch_*.pkl"))
        if not batch_files:
            return
        n = len(batch_files)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        n_test = max(1, n - n_train - n_val)

        splits = {"train": batch_files[:n_train],
                  "val": batch_files[n_train: n_train + n_val],
                  "test": batch_files[n_train + n_val:]}

        for split_name, files in splits.items():
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for bf in files:
                dest = split_dir / bf.name
                if not dest.exists():
                    dest.symlink_to(bf.resolve())
        _log.info(
            f"Splits: train={len(splits['train'])} "
            f"val={len(splits['val'])} test={len(splits['test'])} batches"
        )


    def generate_dataset(self, n_samples=None, overlap_fraction=None,
                          create_splits=None, **kwargs) -> Dict:
        """Alias for generate() — backwards-compatibility with generate_dataset.py."""
        self.config = {**self.config, **kwargs}
        return self.generate(n_samples=n_samples, overlap_fraction=overlap_fraction,
                             create_splits=create_splits)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _run_from_config(config_path: str) -> None:
    import yaml
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    gen = GWDatasetGenerator(
        output_dir=cfg.get("output_dir", "data/dataset"),
        config=cfg,
    )
    gen.generate()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        _run_from_config(sys.argv[1])
    else:
        print("Usage: python -m ahsd.data.dataset_generator <config.yaml>")
