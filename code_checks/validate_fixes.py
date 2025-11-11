#!/usr/bin/env python3
"""
Direct validation of noise quality fixes
Tests PSD variation and noise generation without full dataset pipeline
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
    logger.info("NOISE QUALITY VALIDATION - FIXES")
    logger.info("=" * 80)
    
    # Load PSDs
    psd_manager = PSDManager(sample_rate=4096, duration=4.0)
    psds = psd_manager.load_detector_psds(['H1', 'L1', 'V1'])
    
    logger.info("\n1️⃣  PSD VARIATION CHECK:")
    logger.info("-" * 80)
    
    all_psds_good = True
    for det_name, psd_dict in psds.items():
        psd = psd_dict['psd']
        freqs = psd_dict['frequencies']
        
        # Filter to 50-2000 Hz
        mask = (freqs >= 50) & (freqs <= 2000)
        psd_filtered = psd[mask]
        
        if len(psd_filtered) > 0:
            mean_psd = np.mean(psd_filtered)
            std_psd = np.std(psd_filtered)
            variation = std_psd / mean_psd if mean_psd > 0 else 0
            
            logger.info(f"{det_name}:")
            logger.info(f"  PSD mean (50-2000 Hz): {mean_psd:.2e}")
            logger.info(f"  PSD std: {std_psd:.2e}")
            logger.info(f"  Variation (std/mean): {variation:.4f}")
            
            if variation > 0.1:
                logger.info(f"  ✓ Good variation")
            else:
                logger.warning(f"  ⚠️  FAILED: Poor variation (need > 0.1)")
                all_psds_good = False
    
    logger.info("\n2️⃣  COLORED NOISE GENERATION CHECK:")
    logger.info("-" * 80)
    
    noise_gen = NoiseGenerator(sample_rate=4096, duration=4.0)
    all_noises_good = True
    
    for det_name in ['H1', 'L1', 'V1']:
        psd_dict = psds[det_name]
        amplitudes = []
        
        # Generate 10 noise samples per detector
        for i in range(10):
            noise = noise_gen.generate_colored_noise(psd_dict)
            amp = np.max(np.abs(noise))
            amplitudes.append(amp)
        
        min_amp = np.min(amplitudes)
        avg_amp = np.mean(amplitudes)
        
        logger.info(f"{det_name}:")
        logger.info(f"  Avg max amplitude: {avg_amp:.2e}")
        logger.info(f"  Min max amplitude: {min_amp:.2e}")
        
        if min_amp > 1e-30:
            logger.info(f"  ✓ All samples non-zero")
        else:
            logger.warning(f"  ⚠️  FAILED: Dead channel found (max < 1e-30)")
            all_noises_good = False
    
    logger.info("\n" + "=" * 80)
    if all_psds_good and all_noises_good:
        logger.info("✓ VALIDATION PASSED")
        logger.info("=" * 80)
        return 0
    else:
        logger.warning("⚠️  VALIDATION FAILED")
        logger.info("=" * 80)
        return 1

if __name__ == '__main__':
    sys.exit(main())
