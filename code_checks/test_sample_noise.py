#!/usr/bin/env python3
"""
Test that actual sample generation produces non-zero noise
"""
import numpy as np
import logging
import sys
sys.path.insert(0, '/home/bibinathomas/PosteriFlow/src')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from ahsd.data.psd_manager import PSDManager
from ahsd.data.noise_generator import NoiseGenerator

def main():
    logger.info("=" * 80)
    logger.info("SAMPLE NOISE GENERATION TEST")
    logger.info("=" * 80)
    
    # Setup
    psd_manager = PSDManager(sample_rate=4096, duration=4.0)
    psds = psd_manager.load_detector_psds(['H1', 'L1', 'V1'])
    noise_gen = NoiseGenerator(sample_rate=4096, duration=4.0)
    
    # Generate 10 samples per detector and check noise
    logger.info("\nGenerating noise for 10 samples per detector:")
    logger.info("-" * 80)
    
    total_samples = 0
    zero_samples = 0
    
    for det_name in ['H1', 'L1', 'V1']:
        psd_dict = psds[det_name]
        detector_zero_count = 0
        
        for sample_id in range(10):
            noise = noise_gen.generate_colored_noise(psd_dict)
            
            # Check using np.allclose (as in analysis.py)
            is_zero_allclose = np.allclose(noise, 0, atol=1e-30)
            
            # Also check max amplitude
            max_amp = np.max(np.abs(noise))
            
            total_samples += 1
            if is_zero_allclose:
                detector_zero_count += 1
                zero_samples += 1
                logger.warning(f"  {det_name} sample {sample_id}: ZERO (allclose), max_amp={max_amp:.2e}")
            else:
                # Only log first one
                if sample_id == 0:
                    logger.info(f"  {det_name}: OK (max_amp={max_amp:.2e})")
        
        logger.info(f"  {det_name}: {10-detector_zero_count}/10 non-zero")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Total: {total_samples - zero_samples}/{total_samples} samples non-zero")
    if zero_samples > 0:
        logger.warning(f"⚠️  {zero_samples} zero-noise samples found")
        return 1
    else:
        logger.info("✓ All samples have non-zero noise")
        return 0

if __name__ == '__main__':
    sys.exit(main())
