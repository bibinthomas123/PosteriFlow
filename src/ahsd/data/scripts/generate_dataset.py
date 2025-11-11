#!/usr/bin/env python3
"""
Complete Dataset Generation Script for AHSD Overlapping GW Analysis
====================================================================

Generates production-ready datasets with:
- Configurable overlapping scenarios
- Multiple detector networks
- Realistic astrophysical distributions
- Quality validation
- Multiple output formats

Usage:
    python generate_dataset.py --config config.yaml
    python generate_dataset.py --n-samples 10000 --output-dir data/my_dataset
    ahsd-generate --n-samples 10000 --overlap-fraction 0.1
"""

import argparse
import logging
import sys
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
from collections import Counter

# Import AHSD data modules
from ahsd.data import GWDatasetGenerator
from ahsd.data.config import (
    SAMPLE_RATE,
    DURATION,
    DETECTORS,
    EVENT_TYPE_DISTRIBUTION,
    SNR_DISTRIBUTION,
    OVERLAP_FRACTION,
    EDGE_CASE_FRACTION,
    SNR_RANGES,
)
from ahsd.data.io_utils import DatasetReader, MetadataManager


def setup_logging(output_dir: Path, verbose: bool = False):
    """Setup comprehensive logging"""

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"generation_{time.strftime('%Y%m%d_%H%M%S')}.log"

    log_level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")

    return logger


def load_config(config_file: Optional[str]) -> Dict:
    """Load configuration from YAML file"""

    if config_file is None:
        return {}

    config_path = Path(config_file)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: Dict) -> Dict:
    """Validate and fill in default values"""

    validated = {
        "n_samples": config.get("n_samples", 1000),
        "sample_rate": config.get("sample_rate", SAMPLE_RATE),
        "duration": config.get("duration", DURATION),
        "detectors": config.get("detectors", DETECTORS),
        "output_dir": config.get("output_dir", "data/output"),
        "output_format": config.get("output_format", "pkl_compressed"),
        "overlap_fraction": config.get("overlap_fraction", OVERLAP_FRACTION),
        "edge_case_fraction": config.get("edge_case_fraction", EDGE_CASE_FRACTION),
        "save_batch_size": config.get("save_batch_size", 100),
        "add_glitches": config.get("add_glitches", True),
        "preprocess": config.get("preprocess", True),
        "save_complete": config.get("save_complete", True),
        "validate_output": config.get("validate_output", True),
        "random_seed": config.get("random_seed", None),
        "event_distribution": config.get("event_distribution", EVENT_TYPE_DISTRIBUTION),
        "snr_distribution": config.get("snr_distribution", SNR_DISTRIBUTION),
        "create_splits": config.get("create_splits", True),
        "train_frac": config.get("train_frac", 0.8),
        "val_frac": config.get("val_frac", 0.1),
        "test_frac": config.get("test_frac", 0.1),
        "chunk_size": config.get("chunk_size", 100),  # ✅ ADD THIS
        "noise_augmentation_k": config.get("noise_augmentation_k", 1),  # ✅ ADD THIS
        "debug_snr_diagnostic": config.get("debug_snr_diagnostic", False),
        # ✅ ADD THESE TWO CRITICAL SECTIONS:
        "edge_cases": config.get("edge_cases", {}),
        "extreme_cases": config.get("extreme_cases", {}),
    }

    # Validate ranges
    if validated["n_samples"] < 1:
        raise ValueError("n_samples must be >= 1")

    if not 0 <= validated["overlap_fraction"] <= 1:
        raise ValueError("overlap_fraction must be in [0, 1]")

    if validated["output_format"] not in ["hdf5", "pkl", "pkl_compressed", "both"]:
        raise ValueError(f"Invalid output_format: {validated['output_format']}")

    total_frac = validated["train_frac"] + validated["val_frac"] + validated["test_frac"]
    if abs(total_frac - 1.0) > 0.01:
        raise ValueError(f"Split fractions must sum to 1.0, got {total_frac}")

    return validated


def apply_custom_distributions(config: Dict):
    """Apply custom event and SNR distributions if specified"""

    if "event_distribution" in config:
        import ahsd.data.config as data_config

        data_config.EVENT_TYPE_DISTRIBUTION = config["event_distribution"]
        logging.info(f"Applied custom event distribution: {config['event_distribution']}")

    if "snr_distribution" in config:
        import ahsd.data.config as data_config

        data_config.SNR_DISTRIBUTION = config["snr_distribution"]
        logging.info(f"Applied custom SNR distribution: {config['snr_distribution']}")


def generate_dataset_from_config(config: Dict) -> Dict:
    """Generate dataset from configuration with proper error handling."""

    logger = logging.getLogger(__name__)

    try:
        # Extract parameters
        n_samples = config.get("n_samples", 1000)
        output_dir = config.get("output_dir", "data/dataset")

        # Initialize generator
        generator = GWDatasetGenerator(
            output_dir=output_dir,
            sample_rate=config.get("sample_rate", 4096),
            duration=config.get("duration", 4.0),
            detectors=config.get("detectors", ["H1", "L1", "V1"]),
            output_format=config.get("output_format", "pkl"),
            config={
                **config,
                "quota_mode": config.get(
                    "quota_mode", True
                ),  # ✅ Enable quota enforcement by default
                "expected_signals_per_overlap": config.get("expected_signals_per_overlap", 2.5),
            },
        )

        # Generate dataset
        summary = generator.generate_dataset(
            n_samples=n_samples,
            overlap_fraction=config.get("overlap_fraction", 0.5),
            edge_case_fraction=config.get("edge_case_fraction", 0.15),
            save_batch_size=config.get("save_batch_size", 100),
            add_glitches=config.get("add_glitches", True),
            preprocess=config.get("preprocess", True),
            save_complete=config.get("save_complete", False),
            create_splits=config.get("create_splits", True),
            train_frac=config.get("train_frac", 0.8),
            val_frac=config.get("val_frac", 0.1),
            test_frac=config.get("test_frac", 0.1),
            chunk_size=config.get("chunk_size", 100),
            noise_augmentation_k=config.get("noise_augmentation_k", 1),
            config=config,
        )

        # ✅ FIX: Handle both complete generation and early return cases
        logger.info("")
        logger.info("=" * 70)
        logger.info("DATASET GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"✓ Generated {summary.get('n_samples', 0):,} samples")

        generation_time = summary.get("generation_time")
        if generation_time and generation_time > 0:
            logger.info(
                f"✓ Time elapsed: {generation_time:.1f}s ({generation_time/60:.1f}m) @ {summary.get('samples_per_second', 0):.2f} samples/s"
            )
        else:
            logger.info(f"✓ Time elapsed: 0.0s (0.0m) @ 0.00 samples/s")

        logger.info(f"✓ Output directory: {summary.get('output_dir', output_dir)}")

        # ✅ FIX: Only log n_batches if it exists
        if "n_batches" in summary:
            logger.info(f"✓ Batches created: {summary['n_batches']}")

        # ✅ FIX: Check if this was a resume/skip
        if summary.get("resumed", False):
            logger.info("✓ Resumed from existing dataset")
        if summary.get("already_complete", False):
            logger.info("✓ Dataset already complete, no generation needed")

        logger.info("=" * 70)

        # Validation
        if config.get("validate_output", True):
            logger.info("")
            logger.info("=" * 70)
            logger.info("VALIDATING GENERATED DATASET")
            logger.info("=" * 70)

            validation = validate_generated_dataset(Path(output_dir), config)

            if validation["passed"]:
                logger.info("✓ Validation PASSED")
            else:
                logger.warning("✗ Validation FAILED")
                if validation.get("errors"):
                    logger.error("Errors:")
                    for error in validation["errors"]:
                        logger.error(f"  - {error}")
                if validation.get("warnings"):
                    logger.warning("Warnings:")
                    for warning in validation["warnings"]:
                        logger.warning(f"  - {warning}")

        return summary

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


def _categorize_snr(snr: float) -> str:
    """
    Categorize SNR into one of 5 regimes using configured SNR_RANGES.

    Args:
        snr: Signal-to-noise ratio

    Returns:
        Category string: 'weak', 'low', 'medium', 'high', or 'loud'
    """
    from ahsd.data.config import SNR_RANGES

    # Check each regime's bounds for consistency
    for regime, (min_snr, max_snr) in SNR_RANGES.items():
        if min_snr <= snr < max_snr:
            return regime

    # Handle out-of-range values
    if snr < 10.0:  # Below weak minimum
        return "weak"
    else:  # Above loud maximum
        return "loud"


def validate_premerger_samples(samples: List[Dict], logger) -> bool:
    """
    Validate pre-merger samples have correct metadata.

    Args:
        samples: List of sample dicts
        logger: Logger instance

    Returns:
        True if validation passed
    """
    if not samples:
        logger.info("    ℹ No samples to validate")
        return True

    premerger_count = 0
    valid_premerger = 0

    for sample in samples:
        if sample is None:
            continue

        # Check if this is a pre-merger sample
        metadata = sample.get("metadata", {})
        is_premerger = (
            metadata.get("phase") == "inspiral_only"
            or metadata.get("merger_in_window") is False
            or "time_to_merger" in metadata
        )

        if is_premerger:
            premerger_count += 1

            # Validate has time_to_merger
            if "time_to_merger" in metadata:
                ttm = metadata["time_to_merger"]
                if ttm > 0:
                    valid_premerger += 1

    if premerger_count > 0:
        logger.info(f"    ✓ Pre-merger samples: {premerger_count} ({valid_premerger} valid)")
        return valid_premerger == premerger_count
    else:
        logger.info(f"    ℹ No pre-merger samples found")
        return True


def validate_generated_dataset(output_dir: Path, config: Dict) -> Dict:
    """
    Validate generated dataset - works with or without complete file.

    Checks:
    - File existence
    - Sample counts (from splits if no complete file)
    - Data integrity (from splits)
    - Parameter distributions (from splits)

    Args:
        output_dir: Path to dataset output directory
        config: Configuration dict

    Returns:
        Validation report dict with passed/failed status
    """

    logger = logging.getLogger(__name__)
    reader = DatasetReader()

    validation_report = {"passed": True, "errors": [], "warnings": [], "checks": {}}

    logger.info("\n" + "=" * 80)
    logger.info("DATASET VALIDATION")
    logger.info("=" * 80)

    # Check 1: Output files exist
    logger.info("  [1/6] Checking output files...")

    # Check for splits (primary storage method)
    splits_exist = {
        "train": (output_dir / "train").exists(),
        "validation": (output_dir / "validation").exists(),
        "test": (output_dir / "test").exists(),
    }

    validation_report["checks"]["splits_exist"] = splits_exist

    if all(splits_exist.values()):
        logger.info(f"    ✓ All splits found")
        validation_report["checks"]["has_splits"] = True
    else:
        missing = [k for k, v in splits_exist.items() if not v]
        validation_report["warnings"].append(f"Missing splits: {missing}")
        logger.warning(f"    ⚠ Missing splits: {missing}")

    # Check for complete file (optional)
    complete_file = None
    if config.get("save_complete", False):
        if config.get("output_format") == "pkl_compressed":
            complete_file = output_dir / "complete_dataset.pkl.gz"
        else:
            complete_file = output_dir / "complete_dataset.pkl"

        if complete_file and complete_file.exists():
            validation_report["checks"]["complete_file_exists"] = True
            logger.info(f"    ✓ Complete file found: {complete_file.name}")
        else:
            validation_report["warnings"].append("No complete dataset file (using splits only)")
            logger.info(f"    ℹ No complete file (splits-only mode)")
    else:
        logger.info(f"    ℹ Complete file disabled (memory-optimized mode)")
        validation_report["checks"]["complete_file_exists"] = False

    # Check 2: Load and verify sample count
    logger.info("  [2/6] Verifying sample count...")

    dataset = None
    samples = []

    # Try loading from complete file first
    if complete_file and complete_file.exists():
        try:
            dataset = reader.load_pkl(str(complete_file))
            samples = dataset.get("samples", [])
            actual_samples = len(samples)
            expected_samples = config["n_samples"]

            if actual_samples == expected_samples:
                validation_report["checks"]["sample_count"] = True
                logger.info(f"    ✓ Sample count matches: {actual_samples:,}")
            else:
                validation_report["errors"].append(
                    f"Sample count mismatch: expected {expected_samples:,}, got {actual_samples:,}"
                )
                validation_report["passed"] = False
                logger.error(f"    ✗ Expected {expected_samples:,}, got {actual_samples:,}")

        except Exception as e:
            validation_report["warnings"].append(f"Could not load complete file: {e}")
            logger.warning(f"    ⚠ Could not load complete file: {e}")

    # Load from splits if no complete file
    if not samples and splits_exist["train"]:
        logger.info(f"    ℹ Loading from splits for validation...")
        try:
            # Load samples from train split (first 5 chunks for better sampling)
            train_dir = output_dir / "train"
            chunk_files = sorted(train_dir.glob("*chunk*.pkl"))[:5]

            for chunk_file in chunk_files:
                try:
                    import pickle

                    with open(chunk_file, "rb") as f:
                        chunk_data = pickle.load(f)
                        if isinstance(chunk_data, list):
                            samples.extend(chunk_data)
                        elif isinstance(chunk_data, dict) and "samples" in chunk_data:
                            samples.extend(chunk_data["samples"])
                except Exception as e:
                    logger.warning(f"    ⚠ Failed to load {chunk_file.name}: {e}")

            if samples:
                logger.info(f"    ✓ Loaded {len(samples):,} samples from {len(chunk_files)} chunks")
                validation_report["checks"]["sample_count"] = "partial"
            else:
                validation_report["warnings"].append("Could not load samples from splits")
                logger.warning(f"    ⚠ Could not load samples from splits")

        except Exception as e:
            validation_report["warnings"].append(f"Failed to load from splits: {e}")
            logger.warning(f"    ⚠ Failed to load from splits: {e}")

    # ✅ Early exit if no samples loaded
    if not samples:
        logger.error("    ✗ No samples could be loaded for validation!")
        validation_report["errors"].append("No samples loaded")
        validation_report["passed"] = False
        return validation_report

    logger.info(f"    ✓ Loaded {len(samples):,} samples for validation")

    # Check 3: Verify overlap fraction
    logger.info("  [3/6] Checking overlap distribution...")

    try:
        n_overlap = sum(1 for s in samples if s and s.get("is_overlap", False))
        n_single = len(samples) - n_overlap
        actual_overlap_frac = n_overlap / len(samples) if samples else 0
        expected_overlap_frac = config.get("overlap_fraction", 0.4)

        tolerance = 0.10  # ✅ 10% tolerance (more lenient for partial data)

        logger.info(f"    Single events: {n_single:,} ({(1-actual_overlap_frac)*100:.1f}%)")
        logger.info(f"    Overlap events: {n_overlap:,} ({actual_overlap_frac*100:.1f}%)")

        if abs(actual_overlap_frac - expected_overlap_frac) < tolerance:
            validation_report["checks"]["overlap_fraction"] = True
            logger.info(f"    ✓ Within tolerance (expected: {expected_overlap_frac:.1%})")
        else:
            diff = abs(actual_overlap_frac - expected_overlap_frac)
            validation_report["warnings"].append(
                f"Overlap fraction off by {diff:.1%} "
                f"(expected {expected_overlap_frac:.1%}, got {actual_overlap_frac:.1%})"
            )
            logger.warning(f"    ⚠ Off by {diff:.1%} (expected: {expected_overlap_frac:.1%})")

    except Exception as e:
        validation_report["warnings"].append(f"Could not verify overlap fraction: {e}")
        logger.warning(f"    ⚠ Could not verify overlap fraction: {e}")

    # Check 4: Data integrity
    logger.info("  [4/6] Checking data integrity...")

    try:
        import numpy as np

        integrity_checks = {
            "has_strain_data": 0,
            "has_parameters": 0,
            "has_valid_snr": 0,
            "has_detector_data": 0,
            "valid_samples": 0,
        }

        check_count = min(500, len(samples))  # ✅ Check up to 500 samples

        for i, sample in enumerate(samples[:check_count]):
            if sample is None:
                continue

            integrity_checks["valid_samples"] += 1

            # Check detector data
            if "detector_data" in sample and sample["detector_data"] is not None:
                integrity_checks["has_detector_data"] += 1

                for det_name, det_data in sample["detector_data"].items():
                    if det_data is None:
                        continue

                    if "strain" in det_data and det_data["strain"] is not None:
                        integrity_checks["has_strain_data"] += 1

                        # Check for NaN/Inf
                        strain = det_data["strain"]
                        if not np.all(np.isfinite(strain)):
                            validation_report["errors"].append(
                                f"Sample {i} contains NaN/Inf in {det_name}"
                            )
                            validation_report["passed"] = False

            # Check parameters
            if "parameters" in sample and sample["parameters"] is not None:
                integrity_checks["has_parameters"] += 1

                params = sample["parameters"]
                if isinstance(params, list):
                    params = params[0] if len(params) > 0 else None

                if params is not None and isinstance(params, dict):
                    if "target_snr" in params and params["target_snr"] > 0:
                        integrity_checks["has_valid_snr"] += 1

        validation_report["checks"]["integrity"] = integrity_checks

        logger.info(f"    ✓ Checked {check_count:,} samples:")
        logger.info(f"      Valid samples: {integrity_checks['valid_samples']}")
        logger.info(f"      With detector data: {integrity_checks['has_detector_data']}")
        logger.info(f"      With parameters: {integrity_checks['has_parameters']}")
        logger.info(f"      With valid SNR: {integrity_checks['has_valid_snr']}")

    except Exception as e:
        validation_report["warnings"].append(f"Data integrity check failed: {e}")
        logger.warning(f"    ⚠ Data integrity check failed: {e}")

    # Check 5: Event type distribution
    logger.info("  [5/6] Checking event type distribution...")

    try:
        event_counts = Counter()
        for sample in samples:
            if sample is None:
                continue
            event_type = sample.get("type", "unknown")
            event_counts[event_type] += 1

        total = sum(event_counts.values())

        logger.info(f"    Event type distribution ({total:,} samples):")
        for event_type in sorted(event_counts.keys()):
            count = event_counts[event_type]
            pct = count / total * 100 if total > 0 else 0
            logger.info(f"      {event_type:10s}: {count:>6,} ({pct:>5.1f}%)")

        validation_report["checks"]["event_distribution"] = dict(event_counts)

    except Exception as e:
        validation_report["warnings"].append(f"Event type distribution check failed: {e}")
        logger.warning(f"    ⚠ Event type check failed: {e}")

    # Check 6: SNR distribution
    logger.info("  [6/6] Checking SNR distribution...")

    try:
        import numpy as np

        snr_distribution = Counter({"weak": 0, "low": 0, "medium": 0, "high": 0, "loud": 0})
        snr_values = []

        for sample in samples:
            if sample is None:
                continue

            # Get SNR from parameters
            snr = None
            params = sample.get("parameters")

            if params:
                if isinstance(params, list):
                    # Overlapping - take max SNR
                    snrs = [
                        p.get("target_snr", 0)
                        for p in params
                        if isinstance(p, dict) and p.get("target_snr", 0) > 0
                    ]
                    if snrs:
                        snr = max(snrs)
                elif isinstance(params, dict):
                    snr = params.get("target_snr")

            if snr is not None and snr > 0:
                snr_values.append(snr)
                category = _categorize_snr(snr)
                snr_distribution[category] += 1

        total_snr_samples = sum(snr_distribution.values())

        if total_snr_samples > 0:
            logger.info(f"    SNR distribution ({total_snr_samples:,} signals):")
            logger.info(
                f"      Mean: {np.mean(snr_values):.2f}, Median: {np.median(snr_values):.2f}"
            )

            for category in ["weak", "low", "medium", "high", "loud"]:
                count = snr_distribution[category]
                fraction = count / total_snr_samples
                bar = "█" * int(fraction * 50)
                logger.info(
                    f"      {category.capitalize():6s}: {count:>6,} ({fraction*100:>5.1f}%) {bar}"
                )

            validation_report["checks"]["snr_distribution"] = dict(snr_distribution)
        else:
            logger.warning(f"    ⚠ No valid SNR values found")
            validation_report["warnings"].append("No valid SNR values")

    except Exception as e:
        validation_report["warnings"].append(f"SNR distribution check failed: {e}")
        logger.warning(f"    ⚠ SNR distribution check failed: {e}")

    # Check 7: Pre-merger samples
    logger.info("  [7/7] Validating pre-merger samples...")

    try:
        premerger_valid = validate_premerger_samples(samples, logger)
        validation_report["checks"]["premerger_valid"] = premerger_valid
    except Exception as e:
        validation_report["warnings"].append(f"Pre-merger validation failed: {e}")
        logger.warning(f"    ⚠ Pre-merger validation failed: {e}")

    # Final summary
    logger.info("\n" + "=" * 80)
    if validation_report["passed"] and not validation_report["errors"]:
        logger.info("✅ DATASET VALIDATION PASSED")
    else:
        logger.error("❌ DATASET VALIDATION FAILED")

        if validation_report["errors"]:
            logger.error(f"\nErrors ({len(validation_report['errors'])}):")
            for error in validation_report["errors"]:
                logger.error(f"  • {error}")

    if validation_report["warnings"]:
        logger.warning(f"\nWarnings ({len(validation_report['warnings'])}):")
        for warning in validation_report["warnings"]:
            logger.warning(f"  • {warning}")

    logger.info("=" * 80 + "\n")

    return validation_report


def parse_arguments():
    """Parse command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Generate AHSD Overlapping GW Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from config file
  %(prog)s --config my_config.yaml
  
  # Quick generation with defaults
  %(prog)s --n-samples 1000 --output-dir data/test
  
  # Production dataset
  %(prog)s --n-samples 50000 --overlap-fraction 0.1 --output-format pkl_compressed
  
  # Custom detector network
  %(prog)s --n-samples 5000 --detectors H1 L1 V1
        """,
    )

    # Configuration
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # Dataset parameters
    parser.add_argument("--n-samples", type=int, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, help="Output directory path")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["hdf5", "pkl", "pkl_compressed", "both"],
        help="Output file format",
    )

    # Signal parameters
    parser.add_argument(
        "--overlap-fraction", type=float, help="Fraction of samples with overlapping signals (0-1)"
    )
    parser.add_argument(
        "--edge-case-fraction", type=float, help="Fraction of edge case samples (0-1)"
    )

    # Detector configuration
    parser.add_argument(
        "--detectors", nargs="+", choices=["H1", "L1", "V1"], help="Detector network (e.g., H1 L1)"
    )
    parser.add_argument("--sample-rate", type=int, help="Sample rate in Hz")
    parser.add_argument("--duration", type=float, help="Sample duration in seconds")

    # Processing options
    parser.add_argument("--no-glitches", action="store_true", help="Disable glitch injection")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable preprocessing")
    parser.add_argument("--no-validate", action="store_true", help="Skip output validation")
    parser.add_argument(
        "--no-save-complete", action="store_true", help="Do not save complete dataset file"
    )

    # Advanced options
    parser.add_argument("--save-batch-size", type=int, help="Batch size for saving")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility")

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal logging")

    # splits
    parser.add_argument(
        "--no-splits", action="store_true", help="Do not create train/val/test splits"
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.8, help="Training set fraction (default: 0.8)"
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.1, help="Validation set fraction (default: 0.1)"
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.1, help="Test set fraction (default: 0.1)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100, help="Samples per chunk file (default: 100)"
    )
    parser.add_argument(
        "--noise-augmentation",
        type=int,
        default=1,
        help="Noise augmentation factor for training set (default: 1, no augmentation)",
    )
    parser.add_argument(
        "--debug-snr",
        action="store_true",
        help="Enable debug SNR diagnostics (logs target/pre/actual SNR for first N samples)",
    )

    return parser.parse_args()


def main():
    """Main entry point"""

    args = parse_arguments()

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = {}

    # Override with command-line arguments
    if args.n_samples:
        config["n_samples"] = args.n_samples
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.output_format:
        config["output_format"] = args.output_format
    if args.overlap_fraction is not None:
        config["overlap_fraction"] = args.overlap_fraction
    if args.edge_case_fraction is not None:
        config["edge_case_fraction"] = args.edge_case_fraction
    if args.detectors:
        config["detectors"] = args.detectors
    if args.sample_rate:
        config["sample_rate"] = args.sample_rate
    if args.duration:
        config["duration"] = args.duration
    if args.save_batch_size:
        config["save_batch_size"] = args.save_batch_size
    if args.random_seed is not None:
        config["random_seed"] = args.random_seed
    if getattr(args, "debug_snr", False):
        config["debug_snr_diagnostic"] = True

    # Processing flags
    if args.no_glitches:
        config["add_glitches"] = False
    if args.no_preprocess:
        config["preprocess"] = False
    if args.no_validate:
        config["validate_output"] = False
    if args.no_save_complete:
        config["save_complete"] = False

    # Validate configuration
    try:
        config = validate_config(config)
    except Exception as e:
        print(f"Error: Invalid configuration - {e}")
        sys.exit(1)

    # Setup logging
    output_dir = Path(config["output_dir"])
    logger = setup_logging(output_dir, verbose=args.verbose)

    # Generate dataset
    try:
        summary = generate_dataset_from_config(config)

        logger.info("=" * 70)
        logger.info("SUCCESS: Dataset generation completed successfully!")
        logger.info("=" * 70)

        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\nDataset generation interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"FAILED: Dataset generation encountered an error", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
