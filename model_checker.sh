#!/bin/bash

echo "🚀 Generating 40,000 samples in 8 batches of 5,000 each"
echo "This is SAFE for 8GB RAM and will complete successfully!"
echo ""

for i in {1..3}; do
    echo "========================================"
    echo "📦 BATCH $i/3 (1,000 samples)"
    echo "========================================"
    
    python experiments/phase1_data_generation.py \
        --total_samples 1000 \
        --output_dir data/batch_$i \
        --seed $((42 + i)) \
        --verbose
    
    if [ $? -ne 0 ]; then
        echo "❌ Batch $i failed. Check logs."
        exit 1
    fi
    
    echo ""
    echo "✅ Batch $i complete!"
    echo "Memory freed. Cooling down for 30s..."
    sleep 30
    echo ""
done

echo "========================================"
echo "🎉 All 8 batches generated successfully!"
echo "Now merging into single dataset..."
echo "========================================"

# Merge script
python -c "
import pickle
from pathlib import Path
import json
import shutil

print('🔀 Merging 8 batches...')

output_dir = Path('data/data_merged')
output_dir.mkdir(parents=True, exist_ok=True)

for split in ['train', 'validation', 'test']:
    print(f'  Processing {split} split...')
    split_dir = output_dir / split
    split_dir.mkdir(exist_ok=True)
    
    all_samples = []
    for i in range(1, 9):
        batch_split = Path(f'data/batch_{i}/{split}')
        if batch_split.exists():
            for chunk in sorted(batch_split.glob('chunk_*.pkl')):
                with open(chunk, 'rb') as f:
                    all_samples.extend(pickle.load(f))
    
    print(f'    Collected {len(all_samples)} samples')
    
    # Save in chunks of 100
    chunk_size = 100
    n_chunks = (len(all_samples) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, len(all_samples))
        chunk = all_samples[start:end]
        
        chunk_file = split_dir / f'chunk_{chunk_idx:04d}.pkl'
        with open(chunk_file, 'wb') as f:
            pickle.dump(chunk, f, protocol=4)
    
    # Save split info
    split_info = {
        'n_samples': len(all_samples),
        'n_chunks': n_chunks,
        'chunk_size': chunk_size
    }
    with open(split_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f'    ✅ Saved {n_chunks} chunks for {split}')

# Copy metadata from first batch
print('📋 Copying metadata...')
shutil.copytree('data/batch_1/detector_psds', 
                output_dir / 'detector_psds', 
                dirs_exist_ok=True)

if Path('data/batch_1/dataset_metadata.json').exists():
    shutil.copy('data/batch_1/dataset_metadata.json', 
                output_dir / 'dataset_metadata.json')

if Path('data/batch_1/parameter_scalers.json').exists():
    shutil.copy('data/batch_1/parameter_scalers.json', 
                output_dir / 'parameter_scalers.json')

print('')
print('✅ MERGE COMPLETE!')
print('📁 Final dataset location: data/data_merged/')
print('')
"

echo "========================================"
echo "✅ 40,000 SAMPLES READY!"
echo "📁 Location: data/data_merged/"
echo "========================================"
