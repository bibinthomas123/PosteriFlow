#!/usr/bin/env python3
"""
Check if network_snr is present in dataset samples
"""

import pickle
import os
from collections import Counter

def extract_samples(batch_data):
    """Return the list of samples contained in the batch."""
    for k, v in batch_data.items():
        if isinstance(v, list):
            return v
    return None

def check_snr_presence(batches_dir, max_batches=3):
    """Check if network_snr is present in samples"""
    total_samples = 0
    samples_with_snr = 0
    snr_values = []
    event_types = Counter()

    print(f"Checking network_snr presence in: {batches_dir}")
    print("=" * 60)

    batch_files = [
        f for f in os.listdir(batches_dir)
        if f.startswith("batch_") and (f.endswith(".pkl") or f.endswith(".pkl.gz"))
    ]
    batch_files.sort()

    for batch_file in batch_files[:max_batches]:
        batch_path = os.path.join(batches_dir, batch_file)
        print(f"\nLoading {batch_file}...")

        try:
            if batch_file.endswith(".pkl.gz"):
                import gzip
                with gzip.open(batch_path, "rb") as f:
                    batch_data = pickle.load(f)
            else:
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

        # Check each sample
        for i, sample in enumerate(samples):
            total_samples += 1

            if not isinstance(sample, dict):
                continue

            # Check event type
            event_type = sample.get('type', 'unknown')
            event_types[event_type] += 1

            # Check if sample has parameters
            params = sample.get('parameters', [])

            if isinstance(params, list) and len(params) > 0:
                # Check first parameter dict for network_snr
                first_param = params[0]
                if isinstance(first_param, dict):
                    snr = first_param.get('network_snr')
                    if snr is not None:
                        samples_with_snr += 1
                        snr_values.append(float(snr))
                        if samples_with_snr <= 5:  # Show first few
                            print(f"    Sample {i}: network_snr = {snr}")
                    else:
                        if total_samples <= 10:  # Show first few missing
                            print(f"    Sample {i}: network_snr MISSING")
                else:
                    if total_samples <= 5:
                        print(f"    Sample {i}: parameters[0] is not dict: {type(first_param)}")
            elif total_samples <= 5:
                print(f"    Sample {i}: no parameters list or empty")

    # Summary
    print(f"\n{'='*60}")
    print("SNR PRESENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples checked: {total_samples}")
    print(f"Samples with network_snr: {samples_with_snr}")
    print(".1f")

    if snr_values:
        print("SNR Statistics:")
        print(".1f")
        print(".1f")
        print(".1f")

        # Show distribution
        print("SNR Distribution (first 10 values):")
        for i, snr in enumerate(snr_values[:10]):
            print("6.2f")

    print("Event Type Distribution:")
    for event_type, count in sorted(event_types.items()):
        pct = count / total_samples * 100 if total_samples > 0 else 0
        print(".1f")

    print(f"\n{'='*60}")
    if samples_with_snr > 0:
        print("✅ network_snr is PRESENT in dataset!")
        if samples_with_snr == total_samples:
            print("✅ All samples have network_snr populated.")
        else:
            print(f"⚠️  Only {samples_with_snr}/{total_samples} samples have network_snr.")
        return True
    else:
        print("❌ network_snr is MISSING from all samples!")
        print("💡 This needs to be fixed in dataset generation.")
        return False

if __name__ == "__main__":
    import sys
    batches_dir = sys.argv[1] if len(sys.argv) > 1 else "data/dataset/batches"
    if not os.path.exists(batches_dir):
        print(f"Batches directory not found: {batches_dir}")
        exit(1)

    success = check_snr_presence(batches_dir)
    exit(0 if success else 1)
