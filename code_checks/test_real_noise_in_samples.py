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
    print(f"Real noise probability: {gen.use_real_noise_prob}")
    print(f"Real noise generators: {list(gen.real_noise_generators.keys())}")
    
    metadata = gen.generate_dataset(n_samples=30)
    
    # Check batch files
    batch_file = output_dir / "batches" / "batch_00000.pkl"
    if batch_file.exists():
        with open(batch_file, "rb") as f:
            batch = pickle.load(f)
        
        print(f"\nBatch contains {len(batch)} samples")
        
        real_count = 0
        synthetic_count = 0
        
        for i, sample in enumerate(batch):
            if "noise_type" in sample:
                noise_type = sample["noise_type"]
                if noise_type == "real":
                    real_count += 1
                elif noise_type == "synthetic":
                    synthetic_count += 1
        
        total = real_count + synthetic_count
        if total > 0:
            print(f"\nNoise Type Distribution:")
            print(f"  Real noise:      {real_count}/{total} ({100*real_count//total}%)")
            print(f"  Synthetic noise: {synthetic_count}/{total} ({100*synthetic_count//total}%)")
            
            if real_count > 0:
                print(f"\n✓ SUCCESS: Real LIGO/Virgo noise is being used in {real_count} samples!")
            else:
                print(f"\n✗ ISSUE: No real noise in dataset (expected ~{int(0.3*total)} samples)")
        else:
            print(f"\n✗ No samples with noise_type field found")
            print(f"\nSample keys: {batch[0].keys() if batch else 'empty'}")
    else:
        print(f"Batch file not found: {batch_file}")

print(f"\n{'='*80}\n")
