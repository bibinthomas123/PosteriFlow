#!/usr/bin/env python
"""Quick validation of SNR-Distance correlation fix."""

import numpy as np
from scipy.stats import pearsonr, spearmanr
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.ahsd.data.parameter_sampler import ParameterSampler


def validate_snr_correlations(n_samples=500):
    """Generate samples and validate SNR-distance correlations."""
    
    sampler = ParameterSampler()
    
    # Storage
    results = {}
    
    for event_type in ['BBH', 'BNS', 'NSBH']:
        distances = []
        snrs = []
        
        logger.info(f'\n✓ Generating {n_samples} {event_type} samples...')
        
        for i in range(n_samples):
            if i % 100 == 0:
                logger.info(f'  {i}/{n_samples}')
            
            if event_type == 'BBH':
                params = sampler.sample_bbh_parameters()
            elif event_type == 'BNS':
                params = sampler.sample_bns_parameters()
            else:
                params = sampler.sample_nsbh_parameters()
            
            distances.append(params['luminosity_distance'])
            snrs.append(params['target_snr'])
        
        # Correlations
        pearson_r, pearson_p = pearsonr(distances, snrs)
        spearman_rho, spearman_p = spearmanr(distances, snrs)
        
        results[event_type] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p,
            'd_min': min(distances),
            'd_max': max(distances),
            'snr_min': min(snrs),
            'snr_max': max(snrs)
        }
        
        # Log results
        logger.info(f'\n2️⃣  Distance-SNR Correlation for {event_type}:')
        logger.info(f'   Pearson r = {pearson_r:.4f} (p={pearson_p:.2e})')
        logger.info(f'   Spearman ρ = {spearman_rho:.4f} (p={spearman_p:.2e})')
        
        # Check thresholds
        if pearson_r < -0.6:
            logger.info(f'   ✅ STRONG NEGATIVE CORRELATION')
        elif pearson_r < -0.3:
            logger.info(f'   ⚠️  WEAK CORRELATION (needs improvement)')
        else:
            logger.info(f'   ❌ VERY WEAK CORRELATION (broken physics)')
    
    return results


if __name__ == '__main__':
    logger.info('=' * 70)
    logger.info('SNR-DISTANCE CORRELATION VALIDATION')
    logger.info('=' * 70)
    
    results = validate_snr_correlations(n_samples=500)
    
    logger.info('\n' + '=' * 70)
    logger.info('SUMMARY')
    logger.info('=' * 70)
    
    thresholds = {
        'BBH': -0.65,
        'BNS': -0.75,
        'NSBH': -0.60
    }
    
    all_pass = True
    for event_type, threshold in thresholds.items():
        r = results[event_type]['pearson_r']
        status = '✅ PASS' if r < threshold else '❌ FAIL'
        all_pass = all_pass and (r < threshold)
        logger.info(f'{event_type}: r={r:.4f} (threshold={threshold}) {status}')
    
    logger.info('=' * 70)
    if all_pass:
        logger.info('✅ ALL TESTS PASSED - SNR correlation fix is working!')
    else:
        logger.info('❌ SOME TESTS FAILED - Issue still present')
    logger.info('=' * 70)
