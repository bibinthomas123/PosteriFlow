#!/usr/bin/env python3
"""
Quick test to validate PSD and noise generation quality after fixes
"""
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ahsd.data.psd_manager import PSDManager
from ahsd.data.noise_generator import NoiseGenerator

def test_psd_variation():
    """Test that PSD has realistic frequency variation"""
    psd_manager = PSDManager(sample_rate=4096, duration=4.0)
    psds = psd_manager.load_detector_psds(['H1', 'L1', 'V1'])
    
    logger.info("=" * 80)
    logger.info("PSD VARIATION TEST")
    logger.info("=" * 80)
    
    for det_name, psd_dict in psds.items():
        psd = psd_dict['psd']
        freqs = psd_dict['frequencies']
        
        # Calculate variation metric
        # Filter to 50-2000 Hz range
        mask = (freqs >= 50) & (freqs <= 2000)
        psd_filtered = psd[mask]
        
        if len(psd_filtered) > 0:
            mean_psd = np.mean(psd_filtered)
            std_psd = np.std(psd_filtered)
            variation = std_psd / mean_psd if mean_psd > 0 else 0
            
            logger.info(f"\n{det_name}:")
            logger.info(f"  Mean PSD (50-2000 Hz): {mean_psd:.2e}")
            logger.info(f"  Std PSD: {std_psd:.2e}")
            logger.info(f"  Variation (std/mean): {variation:.4f}")
            
            if variation > 0.1:
                logger.info(f"  ✓ Good PSD variation (std/mean = {variation:.4f} > 0.1)")
            else:
                logger.warning(f"  ⚠️  Poor PSD variation (std/mean = {variation:.4f} <= 0.1)")

def test_noise_generation():
    """Test that noise generation produces non-zero samples"""
    noise_gen = NoiseGenerator(sample_rate=4096, duration=4.0)
    psd_manager = PSDManager(sample_rate=4096, duration=4.0)
    psds = psd_manager.load_detector_psds(['H1', 'L1', 'V1'])
    
    logger.info("\n" + "=" * 80)
    logger.info("NOISE GENERATION TEST")
    logger.info("=" * 80)
    
    zero_count = 0
    for det_name in ['H1', 'L1', 'V1']:
        psd_dict = psds[det_name]
        
        # Generate 10 noise samples
        max_amplitudes = []
        for i in range(10):
            noise = noise_gen.generate_colored_noise(psd_dict)
            max_amp = np.max(np.abs(noise))
            max_amplitudes.append(max_amp)
            
            if max_amp < 1e-30:
                zero_count += 1
        
        avg_amp = np.mean(max_amplitudes)
        min_amp = np.min(max_amplitudes)
        
        logger.info(f"\n{det_name} (10 samples):")
        logger.info(f"  Avg max amplitude: {avg_amp:.2e}")
        logger.info(f"  Min max amplitude: {min_amp:.2e}")
        logger.info(f"  Std of max amplitudes: {np.std(max_amplitudes):.2e}")
        
        if min_amp > 1e-30:
            logger.info(f"  ✓ All samples have non-zero noise")
        else:
            logger.warning(f"  ⚠️  Some samples have near-zero noise")
    
    logger.info(f"\n✓ Total zero-noise samples found: {zero_count}/30")

if __name__ == '__main__':
    test_psd_variation()
    test_noise_generation()
    logger.info("\n" + "=" * 80)
    logger.info("TESTS COMPLETE")
    logger.info("=" * 80)
