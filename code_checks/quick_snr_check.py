#!/usr/bin/env python3
"""Quick SNR correlation check using actual dataset generation"""

import numpy as np
import logging
from src.ahsd.data.dataset_generator import GWDatasetGenerator
from pathlib import Path
import pickle
import sys

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')

# Use data directory instead of temp
output_dir = Path('/data/snr_check_dataset')
output_dir.mkdir(exist_ok=True, parents=True)

generator = GWDatasetGenerator(
    output_dir=output_dir,
    sample_rate=4096,
    duration=4.0,
    detectors=['H1', 'L1', 'V1'],
    output_format='pkl'
)

print("Generating 500 samples for correlation check...")
summary = generator.generate_dataset(
    n_samples=500,
    overlap_fraction=0.35,
    edge_case_fraction=0.08,
    save_batch_size=100,
    add_glitches=False,
    preprocess=False,
    save_complete=False,
    create_splits=False
)

print("\n2️⃣  Distance-SNR Correlation (expect negative):")

# Find all pkl files in batches subdirectory
pkl_files = sorted(output_dir.glob('batches/*.pkl'))
print(f"Found {len(pkl_files)} batch files\n")

for event_type in ['BBH', 'BNS', 'NSBH']:
    distances = []
    snrs = []
    
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                batch_data = pickle.load(f)
                
                # Handle batch structure
                samples = batch_data.get('samples', [])
                if not samples:
                    continue
                
                for sample in samples:
                    if sample.get('type') == event_type:
                        d = sample.get('luminosity_distance')
                        s = sample.get('network_snr')
                        if d is not None and s is not None:
                            distances.append(d)
                            snrs.append(s)
        except Exception as e:
            pass  # Skip problematic files
    
    if len(distances) > 10:
        distances = np.array(distances)
        snrs = np.array(snrs)
        corr = np.corrcoef(distances, snrs)[0, 1]
        print(f"✓    ⚠️ {event_type}: r={corr:.3f}")
    else:
        print(f"    ✗ {event_type}: insufficient samples ({len(distances)})")

print(f"\nDataset saved to: {output_dir}")
