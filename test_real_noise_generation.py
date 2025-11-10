#!/usr/bin/env python3
"""Test that real noise is being generated in the dataset."""

import logging
import tempfile
import h5py
from pathlib import Path
from ahsd.data.dataset_generator import GWDatasetGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create temporary output directory
with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir) / "test_dataset"
    output_dir.mkdir(parents=True)
    
    print(f"\n{'='*80}")
    print("GENERATING SMALL TEST DATASET WITH REAL NOISE")
    print(f"{'='*80}\n")
    print(f"Output directory: {output_dir}")
    
    try:
        # Create dataset generator (20 samples with 30% real noise)
        gen = GWDatasetGenerator(
            output_dir=str(output_dir),
            detectors=["H1", "L1"]
        )
        
        print(f"\n✓ DatasetGenerator created")
        print(f"  Real noise probability: {gen.use_real_noise_prob}")
        print(f"  Real noise generators: {list(gen.real_noise_generators.keys())}")
        for det, gen_obj in gen.real_noise_generators.items():
            if gen_obj:
                print(f"    {det}: {len(gen_obj.noise_segments)} segments loaded")
            else:
                print(f"    {det}: None (not initialized)")
        
        # Generate the dataset
        print(f"\nGenerating 20 samples...")
        metadata = gen.generate_dataset(n_samples=20)
        
        # Check the generated dataset
        dataset_file = output_dir / "dataset.h5"
        if dataset_file.exists():
            print(f"\n✓ Dataset file created: {dataset_file}")
            
            with h5py.File(dataset_file, "r") as f:
                # Count samples with real vs synthetic noise
                real_noise_count = 0
                synthetic_noise_count = 0
                
                for key in f.keys():
                    if key.startswith("sample_"):
                        if "noise_type" in f[key].attrs:
                            noise_type = f[key].attrs["noise_type"]
                            if isinstance(noise_type, bytes):
                                noise_type = noise_type.decode()
                            
                            if noise_type == "real":
                                real_noise_count += 1
                            else:
                                synthetic_noise_count += 1
                
                total = real_noise_count + synthetic_noise_count
                print(f"\n  Samples with real noise: {real_noise_count}/{total} ({100*real_noise_count//total}%)")
                print(f"  Samples with synthetic noise: {synthetic_noise_count}/{total}")
                
                if real_noise_count > 0:
                    print(f"\n✓ SUCCESS: Real noise is being generated!")
                else:
                    print(f"\n✗ ISSUE: No real noise found in dataset")
        else:
            print(f"✗ Dataset file not created")
            
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("Test complete!")
print(f"{'='*80}\n")
