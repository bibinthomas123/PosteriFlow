#!/usr/bin/env python3
import pickle
import os
from collections import Counter
import sys

def extract_samples(batch_data):
    """Return the list of samples contained in the batch.
    Your batch dict has:
       - batch_id: int
       - n_samples: int
       - samples: list-of-dicts   (unnamed key)
    The only list in the dict is the samples list.
    """

    for k, v in batch_data.items():
        if isinstance(v, list):
            return v
    return None


def analyze_dataset(batches_dir):
    n_signals_counts = Counter()
    overlap_samples = 0
    heavy_overlaps = 0
    total_samples = 0

    print(f"Analyzing batches in: {batches_dir}")

    batch_files = [
        f for f in os.listdir(batches_dir)
        if f.startswith("batch_") and f.endswith(".pkl")
    ]
    batch_files.sort()

    for batch_file in batch_files[::]:      # analyze first 5
        batch_path = os.path.join(batches_dir, batch_file)
        print(f"Loading {batch_file}...")

        try:
            with open(batch_path, "rb") as f:
                batch_data = pickle.load(f)
        except Exception as e:
            print(f"Error reading {batch_file}: {e}")
            continue

        # Extract the list of samples
        samples = extract_samples(batch_data)
        if samples is None:
            print(f"ERROR: No sample list found in {batch_file}")
            continue

        print(f"  Found {len(samples)} samples")

        # Process every sample
        for sample in samples:
            total_samples += 1

            if not isinstance(sample, dict):
                print(f"  Unexpected sample type: {type(sample)}")
                continue

            n_signals = sample.get("n_signals", 1)
            is_overlap = sample.get("is_overlap", False)

            if is_overlap or n_signals > 1:
                overlap_samples += 1
                n_signals_counts[n_signals] += 1

                if n_signals >= 5:
                    heavy_overlaps += 1
                    sid = sample.get("sample_id", "unknown")
                    print(f"  Heavy overlap: {n_signals} signals in sample {sid}")

    # Print summary
    print("\nAnalysis Results:")
    print(f"Total samples analyzed: {total_samples}")
    print(f"Overlap samples: {overlap_samples}")
    print(f"Heavy overlaps (5+ signals): {heavy_overlaps}")

    frac_overlap = overlap_samples / max(total_samples, 1)
    print(f"Fraction overlapped: {frac_overlap:.3f}")

    if overlap_samples > 0:
        print("\nOverlap size distribution:")
        for n in sorted(n_signals_counts.keys()):
            print(f"  {n} signals: {n_signals_counts[n]} samples")

        frac_5plus = heavy_overlaps / max(overlap_samples, 1)
        print(f"Fraction 5+ among overlaps: {frac_5plus:.3f}")

    if heavy_overlaps > 0:
        print("✓ SUCCESS: Found heavy overlaps (5+ signals).")
        return True
    else:
        print("✗ No heavy overlaps found.")
        return False


if __name__ == "__main__":
    batches_dir = "data/dataset/batches"
    if not os.path.exists(batches_dir):
        print(f"Batches directory not found: {batches_dir}")
        sys.exit(1)

    ok = analyze_dataset(batches_dir)
    sys.exit(0 if ok else 1)
