#!/usr/bin/env python3
"""
Test full dataset generation with noise validation
"""
import numpy as np
import logging
import sys
sys.path.insert(0, '/home/bibinathomas/PosteriFlow/src')

logging.basicConfig(level=logging.WARNING)  # Suppress verbose logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from ahsd.data.dataset_generator import GWDatasetGenerator

def main():
    print("=" * 80)
    print("FULL DATASET GENERATION TEST")
    print("=" * 80)
    
    try:
        # Create minimal generator with real noise disabled
        config = {
            'use_real_noise_prob': 0,  # Use synthetic noise only
            'disable_real_noise': True
        }
        gen = GWDatasetGenerator(
            output_dir='/tmp/test_dataset',
            sample_rate=4096,
            duration=4.0,
            detectors=['H1', 'L1', 'V1'],
            output_format='pkl',
            config=config
        )
        
        print("\nGenerating 10 samples...")
        print("-" * 80)
        
        zero_count = 0
        samples_generated = 0
        
        for i in range(10):
            try:
                sample = gen._generate_single_sample(
                    sample_id=i,
                    is_edge_case=False,
                    add_glitches=False,
                    preprocess=False
                )
                samples_generated += 1
                
                # Check noise in detector_data
                for det_name in ['H1', 'L1', 'V1']:
                    if det_name in sample.get('detector_data', {}):
                        det_data = sample['detector_data'][det_name]
                        if 'noise' in det_data:
                            noise = det_data['noise']
                            # Use same check as analysis.py
                            if np.allclose(noise, 0, atol=1e-30):
                                zero_count += 1
                                print(f"Sample {i}, {det_name}: ZERO (allclose)")
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
        
        print("\n" + "=" * 80)
        print(f"Samples generated: {samples_generated}/10")
        print(f"Zero-noise channels found: {zero_count}/30")
        
        if zero_count == 0:
            print("✓ VALIDATION PASSED - No dead channels")
            return 0
        else:
            print(f"⚠️  {zero_count} dead channels found")
            return 1
            
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
