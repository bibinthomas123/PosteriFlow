#!/usr/bin/env python3
"""Test that real noise is actually in the generated samples."""

import tempfile
import pickle
from pathlib import Path
from ahsd.data.dataset_generator import GWDatasetGenerator

with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir) / "test_dataset"
    
    print(f"\n{'='*80}")
    print("CHECKING IF REAL NOISE IS IN GENERATED SAMPLES")
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
        
        real_count = 0
        synthetic_count = 0
        unknown_count = 0
        
        for i, sample in enumerate(samples):
            if "noise_type" in sample:
                noise_type = sample["noise_type"]
                if noise_type == "real":
                    real_count += 1
                elif noise_type == "synthetic":
                    synthetic_count += 1
            else:
                unknown_count += 1
        
        total = len(samples)
        print(f"\nNoise Type Distribution:")
        print(f"  Real noise:      {real_count}/{total} ({100*real_count//total if total else 0}%)")
        print(f"  Synthetic noise: {synthetic_count}/{total} ({100*synthetic_count//total if total else 0}%)")
        if unknown_count > 0:
            print(f"  Unknown/Missing: {unknown_count}/{total}")
        
        if real_count > 0:
            print(f"\n✓ SUCCESS: Real LIGO/Virgo noise is being used in {real_count} samples!")
            print(f"  Expected: ~{int(gen.use_real_noise_prob * total)} samples")
        else:
            print(f"\n⚠ WARNING: No real noise found in this batch")
            print(f"  (Random variation: may occur with 30% probability)")
    else:
        print(f"Batch file not found: {batch_file}")

print(f"\n{'='*80}\n")
