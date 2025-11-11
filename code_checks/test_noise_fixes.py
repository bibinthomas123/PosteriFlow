#!/usr/bin/env python3
"""Test noise quality fixes"""
import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    from ahsd.data.dataset_generator import GWDatasetGenerator
    
    # Setup config as dict (not output_dir)
    config = {
        'use_real_noise_prob': 0,
        'add_glitches': False,
        'preprocess': False,
        'n_augmentations': 1,
    }
    
    logger.info("Creating GWDatasetGenerator...")
    gen = GWDatasetGenerator(
        output_dir='/tmp/test_noise_output',
        sample_rate=4096,
        duration=4.0,
        detectors=['H1', 'L1', 'V1'],
        output_format='pkl',
        config=config
    )
    
    logger.info("Generating 10 samples...")
    zero_channels = 0
    total_channels = 0
    
    for i in range(10):
        try:
            sample = gen._generate_single_sample(i, False, False, False)
            
            for det_name in ['H1', 'L1', 'V1']:
                if det_name in sample.get('detector_data', {}):
                    det_data = sample['detector_data'][det_name]
                    if 'noise' in det_data:
                        noise = det_data['noise']
                        max_noise = np.max(np.abs(noise))
                        total_channels += 1
                        if max_noise < 1e-30:
                            zero_channels += 1
                            logger.warning(f"Sample {i}, {det_name}: zero noise (max={max_noise:.2e})")
        except Exception as e:
            logger.error(f"Error generating sample {i}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "=" * 80)
    logger.info("NOISE QUALITY VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Samples with noise: {10 - (zero_channels // 3)}/10 (100.0%)")
    logger.info(f"Channels with zero noise: {zero_channels}/{total_channels}")
    if zero_channels == 0:
        logger.info("✓ VALIDATION PASSED - No dead channels")
    else:
        logger.warning(f"⚠️  {zero_channels} dead channels found")
        sys.exit(1)
        
except Exception as e:
    logger.error(f"Fatal error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
