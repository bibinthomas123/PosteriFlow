#!/usr/bin/env python3
"""Check the structure of batch files."""

import tempfile
import pickle
from pathlib import Path
from ahsd.data.dataset_generator import GWDatasetGenerator

with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir) / "test_dataset"
    
    gen = GWDatasetGenerator(output_dir=str(output_dir), detectors=["H1", "L1"])
    metadata = gen.generate_dataset(n_samples=5)
    
    # Check what's in the batch file
    batch_file = output_dir / "batches" / "batch_00000.pkl"
    if batch_file.exists():
        with open(batch_file, "rb") as f:
            batch = pickle.load(f)
        
        print(f"Batch type: {type(batch)}")
        print(f"Batch length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
        
        if isinstance(batch, dict):
            print(f"Batch keys: {list(batch.keys())}")
            # Check first key
            first_key = list(batch.keys())[0]
            print(f"\nFirst entry ({first_key}):")
            print(f"  Type: {type(batch[first_key])}")
            if isinstance(batch[first_key], dict):
                print(f"  Keys: {list(batch[first_key].keys())}")
                if "noise_type" in batch[first_key]:
                    print(f"  noise_type: {batch[first_key]['noise_type']}")
        elif isinstance(batch, list):
            print(f"Batch is a list with {len(batch)} items")
            if batch:
                print(f"  First item type: {type(batch[0])}")
                if isinstance(batch[0], dict):
                    print(f"  First item keys: {list(batch[0].keys())}")
