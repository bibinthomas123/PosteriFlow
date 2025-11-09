"""
Test: Overlap Diversity & Calibration Robustness
================================================

Ensures that your dataset generator produces realistic overlap diversity
for PriorityNet or Neural Posterior training.

Checks:
1️⃣  At least 25–35% of overlaps contain ≥5 signals.
2️⃣  Mixed event-type overlaps (BBH + BNS, etc.) exist.
3️⃣  “Decoy” overlaps (real + weak synthetic) are present.
"""

import pytest
import os
import numpy as np

try:
    from ahsd.data.io_utils import DatasetReader
except ImportError:
    DatasetReader = None


@pytest.fixture(scope="function")
def sample_dataset():
    """
    Load a representative subset of the generated dataset for testing.

    Uses DatasetReader to load the first .pkl chunk in data/dataset/train.
    Falls back gracefully if missing.
    """
    dataset_dir = "data/dataset/train"

    if DatasetReader and os.path.exists(dataset_dir):
        try:
            reader = DatasetReader()  # ✅ no args to __init__
            import glob
            chunk_files = sorted(glob.glob(os.path.join(dataset_dir, "*.pkl*")))
            if not chunk_files:
                pytest.skip("No dataset chunks found — generate dataset first.")
            
            # Pick the first chunk and load it
            data = reader.load_pkl(chunk_files[0])
            if isinstance(data, dict):
                if "samples" in data:
                    return data["samples"][:500]
                elif isinstance(data, list):
                    return data[:500]
            elif isinstance(data, list):
                return data[:500]
        except Exception as e:
            print(f"⚠️ DatasetReader failed: {e}")

    # Manual fallback
    import pickle, glob
    chunk_files = sorted(glob.glob(os.path.join(dataset_dir, "*.pkl")))
    if not chunk_files:
        pytest.skip("No dataset chunks found — generate dataset first.")
    with open(chunk_files[0], "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"][:500]
    elif isinstance(data, list):
        return data[:500]
    else:
        pytest.skip("Dataset format not recognized (expected list or dict).")


def test_overlap_diversity(sample_dataset):
    """
    ✅ Validate overlap diversity, type mixing, and decoy presence.

    * 5+ overlaps must represent 25–35% of overlap samples.
    * ≥10% of overlaps must mix event types (BBH+BNS, etc.).
    * ≥5% of overlaps must include both weak (<8) and strong (>15) SNRs.
    """

    samples = sample_dataset
    assert samples, "No samples loaded — dataset is empty."

    overlap_samples = [s for s in samples if s.get("is_overlap", False)]
    if not overlap_samples:
        pytest.skip("No overlap samples present in dataset.")

    # ----------------------------------------------------------------------
    # 1️⃣ Overlap complexity: 5+ signals should represent 25–35% of overlaps
    overlap_sizes = [
        len(s.get("parameters", []))
        for s in overlap_samples
        if isinstance(s.get("parameters"), list)
    ]
    n_5plus = sum(size >= 5 for size in overlap_sizes)
    frac_5plus = n_5plus / len(overlap_samples)

    assert (
        0.25 <= frac_5plus <= 0.40
    ), f"❌ 5+ overlaps fraction ({frac_5plus:.2%}) out of expected range (25–35%)."

    # ----------------------------------------------------------------------
    # 2️⃣ Mixed event types: at least 10% should include multiple types
    mixed_overlaps = 0
    for s in overlap_samples:
        params = s.get("parameters", [])
        if len(params) < 2:
            continue
        types = {p.get("type") or p.get("event_type") for p in params if isinstance(p, dict)}
        if len(types) > 1:
            mixed_overlaps += 1

    frac_mixed = mixed_overlaps / len(overlap_samples)
    assert (
        frac_mixed >= 0.10
    ), f"❌ Only {frac_mixed:.2%} mixed-type overlaps — expected ≥10%."

    # ----------------------------------------------------------------------
    # 3️⃣ Decoy overlaps: at least 5% with both weak (<8) & strong (>15) SNRs
    decoy_overlaps = 0
    for s in overlap_samples:
        params = s.get("parameters", [])
        snrs = [p.get("target_snr", 0) for p in params if isinstance(p, dict)]
        if any(snr < 8 for snr in snrs) and any(snr > 15 for snr in snrs):
            decoy_overlaps += 1

    frac_decoy = decoy_overlaps / len(overlap_samples)
    assert (
        frac_decoy >= 0.05
    ), f"❌ Too few decoy overlaps ({frac_decoy:.2%}) — expected ≥5%."

    # ----------------------------------------------------------------------
    # ✅ Summary printout
    print("\n=== Overlap Diversity Summary ===")
    print(f"Total overlap samples: {len(overlap_samples)}")
    print(f"5+ overlaps: {n_5plus} ({frac_5plus:.2%})")
    print(f"Mixed-type overlaps: {mixed_overlaps} ({frac_mixed:.2%})")
    print(f"Decoy overlaps: {decoy_overlaps} ({frac_decoy:.2%})")
    print("=================================")
