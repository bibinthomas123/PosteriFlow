#!/usr/bin/env python3
"""Test that real noise is actually in the detector_data."""

import tempfile
import pickle
from pathlib import Path
from ahsd.data.dataset_generator import GWDatasetGenerator

with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir) / "test_dataset"
    
    print(f"\n{'='*80}")
    print("CHECKING DETECTOR DATA FOR REAL NOISE")
    print(f"{'='*80}\n")
    
    # Generate dataset
    gen = GWDatasetGenerator(output_dir=str(output_dir), detectors=["H1", "L1"])
    print(f"Real noise probability: {gen.use_real_noise_prob * 100:.0f}%")
    print(f"Real noise generators: {list(gen.real_noise_generators.keys())}")
    
    metadata = gen.generate_dataset(n_samples=50)
    
    # Check batch files
    batch_file = output_dir / "batches" / "batch_00000.pkl"
    if batch_file.exists():
        with open(batch_file, "rb") as f:
            batch = pickle.load(f)
        
        samples = batch["samples"]
        print(f"\nBatch contains {len(samples)} samples")
        
        # Track real noise by detector
        h1_real = 0
        l1_real = 0
        
        # Also check sample-level noise_type
        sample_level_real = 0
        
        for i, sample in enumerate(samples):
            if "detector_data" in sample:
                det_data = sample["detector_data"]
                if "H1" in det_data and "noise_type" in det_data["H1"]:
                    if det_data["H1"]["noise_type"] == "real":
                        h1_real += 1
                if "L1" in det_data and "noise_type" in det_data["L1"]:
                    if det_data["L1"]["noise_type"] == "real":
                        l1_real += 1
            
            if "noise_type" in sample:
                noise_types = sample["noise_type"]
                if isinstance(noise_types, dict):
                    if any(v == "real" for v in noise_types.values()):
                        sample_level_real += 1
        
        total = len(samples)
        print(f"\nDetector-level Real Noise:")
        print(f"  H1 real noise: {h1_real}/{total} ({100*h1_real//total if total else 0}%)")
        print(f"  L1 real noise: {l1_real}/{total} ({100*l1_real//total if total else 0}%)")
        print(f"  Sample-level (any det real): {sample_level_real}/{total} ({100*sample_level_real//total if total else 0}%)")
        
        expected = int(gen.use_real_noise_prob * total)
        print(f"\n  Expected: ~{expected} real samples (at 30% probability)")
        
        if h1_real > 0 or l1_real > 0 or sample_level_real > 0:
            print(f"\n✓ SUCCESS: Real LIGO/Virgo noise is being used!")
        else:
            print(f"\n✗ No real noise found in this batch")
            print(f"\nSample structure check:")
            if samples:
                print(f"  Sample keys: {list(samples[0].keys())}")
                if "detector_data" in samples[0]:
                    print(f"  Detector data keys (H1): {list(samples[0]['detector_data']['H1'].keys())}")
    else:
        print(f"Batch file not found: {batch_file}")

print(f"\n{'='*80}\n")
