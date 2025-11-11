"""Debug script to check if priorities are inverted in targets."""
import sys
sys.path.insert(0, '/home/bibinathomas/PosteriFlow')

import numpy as np
import torch
import pickle
from pathlib import Path
from experiments.train_priority_net import PriorityNetDataset

# Load validation chunks directly
chunk_dir = Path("/home/bibinathomas/PosteriFlow/data/dataset/validation")
chunk_files = sorted(chunk_dir.glob("chunk_*.pkl"))[:1]  # Load first chunk

samples = []
for chunk_file in chunk_files:
    with open(chunk_file, 'rb') as f:
        chunk_data = pickle.load(f)
    samples.extend(chunk_data)
    if len(samples) >= 20:
        break

print(f"✅ Loaded {len(samples)} raw samples")

# Now create the dataset (which applies the fix)
try:
    dataset = PriorityNetDataset(samples, split_name='validation')
    print(f"✅ Created PriorityNetDataset with {len(dataset)} scenarios")
    
    # Check if fix works
    for idx in range(min(5, len(dataset))):
        try:
            item = dataset[idx]
            detections = item['detections']
            priorities = item['priorities'].numpy() if isinstance(item['priorities'], torch.Tensor) else item['priorities']
            print(f"\nDataset Sample {idx}: n_detections={len(detections)}, n_priorities={len(priorities)}")
            
            if len(detections) > 0 and len(priorities) >= 2:
                snrs = []
                for det in detections:
                    snr = det.get('network_snr', det.get('target_snr', 0.0))
                    snrs.append(float(snr))
                snrs = np.array(snrs)
                
                snr_sorted = np.argsort(snrs)[::-1]
                prio_sorted = np.argsort(priorities)[::-1]
                print(f"  SNR ranking:  {snr_sorted}")
                print(f"  Prio ranking: {prio_sorted}")
                print(f"  Match: {np.array_equal(snr_sorted, prio_sorted)}")
        except Exception as e:
            print(f"  Error: {e}")
    
except Exception as e:
    print(f"❌ Failed to create dataset: {e}")
    print(f"Sample keys: {list(samples[0].keys())}")

# Check a few samples
found = 0
print(f"Checking samples for n_signals >= 2...")
for idx, sample in enumerate(samples[:100]):
    # Samples have 'parameters' not 'detections'
    detections = sample.get('detections', sample.get('parameters', []))
    priorities = sample.get('priorities', None)
    n_signals = sample.get('n_signals', len(detections))
    
    print(f"  Sample {idx}: n_signals={n_signals}, has_priorities={priorities is not None}, len(detections)={len(detections)}")
    
    if priorities is None or n_signals < 2:
        continue
    
    found += 1
    if found > 5:
        break
    
    if isinstance(priorities, torch.Tensor):
        priorities = priorities.numpy()
    elif isinstance(priorities, list):
        priorities = np.array(priorities)
    
    print(f"\n===== Sample {idx} (type={sample.get('type')}, n_signals={sample.get('n_signals')}) =====")
    print(f"N detections: {len(detections)}")
    print(f"Priorities shape: {priorities.shape}, values: {priorities}")
    
    # Extract SNRs from parameters
    snrs = []
    for i, param in enumerate(detections):
        snr = param.get('network_snr', param.get('target_snr', 0.0))
        snrs.append(float(snr))
    snrs = np.array(snrs)
    print(f"Network SNRs: {snrs}")
    
    # Check if they correlate correctly
    if len(snrs) > 1:
        snr_sorted = np.argsort(snrs)[::-1]  # descending order (highest SNR first)
        prio_sorted = np.argsort(priorities)[::-1]  # descending order (highest priority first)
        print(f"SNR ranking (desc): indices={snr_sorted}, values={snrs[snr_sorted]}")
        print(f"Priority ranking (desc): indices={prio_sorted}, values={priorities[prio_sorted]}")
        print(f"Rankings match: {np.array_equal(snr_sorted, prio_sorted)}")
        
        # Also check ascending (is one inverted?)
        snr_sorted_asc = np.argsort(snrs)
        prio_sorted_asc = np.argsort(priorities)
        print(f"SNR ranking (asc): indices={snr_sorted_asc}, values={snrs[snr_sorted_asc]}")
        print(f"Priority ranking (asc): indices={prio_sorted_asc}, values={priorities[prio_sorted_asc]}")
        print(f"Rankings match (asc): {np.array_equal(snr_sorted_asc, prio_sorted_asc)}")
