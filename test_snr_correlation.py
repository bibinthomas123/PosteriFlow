#!/usr/bin/env python3
"""Quick test to verify SNR-distance correlation after fix"""

import numpy as np
import logging
from src.ahsd.data.parameter_sampler import ParameterSampler
from src.ahsd.data.injection import attach_network_snr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_snr_correlation(n_samples=2000):
    """Test distance-SNR correlation for each event type"""
    
    sampler = ParameterSampler()
    
    event_types = {
        'BBH': sampler.sample_bbh_parameters,
        'BNS': sampler.sample_bns_parameters,
        'NSBH': sampler.sample_nsbh_parameters,
    }
    
    for event_type, sample_func in event_types.items():
        distances = []
        snrs = []
        
        for _ in range(n_samples):
            params = sample_func()
            attach_network_snr(params)
            
            distances.append(params.get('luminosity_distance', 0))
            snrs.append(params.get('network_snr', 0))
        
        distances = np.array(distances)
        snrs = np.array(snrs)
        
        # Calculate correlation
        correlation = np.corrcoef(distances, snrs)[0, 1]
        
        logger.info(f"\n{event_type}:")
        logger.info(f"  Distance range: {distances.min():.1f} - {distances.max():.1f} Mpc")
        logger.info(f"  SNR range: {snrs.min():.2f} - {snrs.max():.2f}")
        logger.info(f"  Distance-SNR correlation: r={correlation:.3f} (expect ≈ -0.8 to -1.0)")

if __name__ == '__main__':
    test_snr_correlation()
