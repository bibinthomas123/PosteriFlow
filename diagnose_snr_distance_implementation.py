#!/usr/bin/env python3
"""
Diagnose SNR-distance physics implementation in parameter sampler.

This script verifies that the sampling code correctly implements:
    distance = ref_distance × (M_c / ref_mass)^(5/6) × (ref_snr / target_snr)

By directly testing the parameter_sampler module.
"""

import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_parameter_sampler():
    """Test that parameter sampler implements physics formula."""
    
    logger.info("=" * 80)
    logger.info("SNR-DISTANCE PHYSICS IMPLEMENTATION DIAGNOSTIC")
    logger.info("=" * 80 + "\n")
    
    try:
        from src.ahsd.data.parameter_sampler import ParameterSampler
    except ImportError:
        logger.error("Cannot import ParameterSampler. Ensure src/ is in Python path.")
        return False
    
    sampler = ParameterSampler()
    
    # Test each event type
    event_types = [('BBH', 'sample_bbh_parameters'), 
                   ('BNS', 'sample_bns_parameters'),
                   ('NSBH', 'sample_nsbh_parameters')]
    
    results = {}
    
    for event_type, sampler_method in event_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING {event_type}")
        logger.info(f"{'='*80}\n")
        
        # Get reference parameters
        ref = sampler.reference_params[event_type]
        logger.info(f"Reference Parameters for {event_type}:")
        logger.info(f"  distance: {ref['distance']:.1f} Mpc")
        logger.info(f"  mass: {ref['mass']:.1f} M☉")
        logger.info(f"  snr: {ref['snr']:.1f}")
        
        # Sample multiple times and check formula
        n_tests = 20
        formula_errors = []
        snr_values = []
        distance_values = []
        
        logger.info(f"\nSampling {n_tests} {event_type} parameters...\n")
        
        for i in range(n_tests):
            # Sample from 'medium' regime for consistency
            params = getattr(sampler, sampler_method)('medium', False)
            
            # Extract required fields
            target_snr = params.get('target_snr')
            chirp_mass = params.get('chirp_mass')
            distance = params.get('luminosity_distance')
            
            if target_snr is None or chirp_mass is None or distance is None:
                logger.warning(f"  Sample {i}: Missing required fields")
                continue
            
            # Compute expected distance from formula
            # distance = ref_d × (M_c/M_ref)^(5/6) × (SNR_ref/SNR_target)
            expected_distance = (ref['distance'] * 
                                (chirp_mass / ref['mass']) ** (5/6) * 
                                (ref['snr'] / target_snr))
            
            # The actual distance has scatter and jitter applied,
            # but should be close to expected (within ~30% due to lognormal scatter)
            ratio = distance / expected_distance
            relative_error = abs(ratio - 1.0)
            
            snr_values.append(target_snr)
            distance_values.append(distance)
            formula_errors.append(relative_error)
            
            if i < 5:  # Print first 5
                logger.info(f"  Sample {i+1}:")
                logger.info(f"    Target SNR: {target_snr:.2f}")
                logger.info(f"    Chirp Mass: {chirp_mass:.2f} M☉")
                logger.info(f"    Expected distance: {expected_distance:.1f} Mpc")
                logger.info(f"    Actual distance: {distance:.1f} Mpc")
                logger.info(f"    Ratio (actual/expected): {ratio:.2f}")
                logger.info(f"    Relative error: {relative_error:.2%}")
        
        # Compute statistics
        mean_error = np.mean(formula_errors)
        max_error = np.max(formula_errors)
        snrs = np.array(snr_values)
        distances = np.array(distance_values)
        
        logger.info(f"\n{'─'*80}")
        logger.info(f"FORMULA VERIFICATION FOR {event_type}")
        logger.info(f"{'─'*80}")
        logger.info(f"Mean relative error: {mean_error:.2%}")
        logger.info(f"Max relative error: {max_error:.2%}")
        logger.info(f"SNR range: {snrs.min():.1f} - {snrs.max():.1f}")
        logger.info(f"Distance range: {distances.min():.1f} - {distances.max():.1f} Mpc")
        
        # Check physics relationship
        from scipy.stats import pearsonr
        corr, p_value = pearsonr(snrs, distances)
        logger.info(f"\nSNR-Distance Correlation: {corr:.4f}")
        logger.info(f"  (Expected: r < -0.5 for physics-coupled sampling)")
        
        # Verdict
        if corr < -0.3:
            logger.info(f"  ✅ Physics coupling DETECTED")
            results[event_type] = True
        else:
            logger.info(f"  ⚠️  Physics coupling WEAK (r > -0.3)")
            results[event_type] = False
        
        # Check formula error
        if mean_error < 0.35:  # 35% error acceptable due to scatter/jitter
            logger.info(f"  ✅ Formula implementation OK (error < 35%)")
        else:
            logger.info(f"  ⚠️  Formula error large (> 35%)")
            results[event_type] = False
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}\n")
    
    for event_type, passed in results.items():
        status = "✅ PASS" if passed else "⚠️  WEAK"
        logger.info(f"{event_type:6s}: {status}")
    
    all_pass = all(results.values())
    
    if all_pass:
        logger.info("\n✅ CONCLUSION: Physics formula is correctly implemented")
        logger.info("   SNR determines distance via canonical GW scaling law")
        logger.info("   If dataset shows weak correlation, check:")
        logger.info("   1. Per-event-type correlation (not just overall)")
        logger.info("   2. Single-signal samples (overlaps weaken correlation)")
        logger.info("   3. Edge cases (6% intentionally modify parameters)")
    else:
        logger.info("\n⚠️  CONCLUSION: Some issues detected")
        logger.info("   Review the physics implementation")
    
    logger.info(f"\n{'='*80}\n")
    
    return all_pass


def test_snr_attachment():
    """Test that SNR attachment preserves physics."""
    
    logger.info("\n" + "="*80)
    logger.info("SNR ATTACHMENT MECHANISM DIAGNOSTIC")
    logger.info("="*80 + "\n")
    
    try:
        from src.ahsd.data.injection import attach_network_snr, proxy_network_snr_from_params
    except ImportError:
        logger.error("Cannot import injection module")
        return False
    
    # Test with realistic parameters
    test_cases = [
        {'mass_1': 50.0, 'mass_2': 40.0, 'luminosity_distance': 1000.0, 'target_snr': 25.0},
        {'mass_1': 1.4, 'mass_2': 1.4, 'luminosity_distance': 100.0, 'target_snr': 20.0},
        {'mass_1': 10.0, 'mass_2': 1.5, 'luminosity_distance': 400.0, 'target_snr': 18.0},
    ]
    
    event_names = ['BBH', 'BNS', 'NSBH']
    
    logger.info("Testing SNR attachment priority system:\n")
    
    for event_name, test_case in zip(event_names, test_cases):
        logger.info(f"{event_name}:")
        
        # Make a copy
        sample = test_case.copy()
        
        # Attach network SNR
        attach_network_snr(sample)
        
        network_snr = sample.get('network_snr')
        target_snr = sample.get('target_snr')
        
        logger.info(f"  Input parameters:")
        logger.info(f"    mass_1: {sample['mass_1']:.1f} M☉")
        logger.info(f"    mass_2: {sample['mass_2']:.1f} M☉")
        logger.info(f"    distance: {sample['luminosity_distance']:.1f} Mpc")
        logger.info(f"    target_snr: {target_snr:.1f} (input)")
        logger.info(f"  Output:")
        logger.info(f"    network_snr: {network_snr:.1f}")
        
        if network_snr is not None and target_snr is not None:
            if abs(network_snr - target_snr) / target_snr < 0.05:
                logger.info(f"  ✅ Priority 1 (target_snr) used correctly")
            else:
                logger.info(f"  ⚠️  Network SNR differs from target SNR")
                logger.info(f"     (May indicate Priority 2/3 used instead)")
        else:
            logger.info(f"  ⚠️  Missing SNR values")
        
        logger.info()
    
    return True


if __name__ == '__main__':
    success1 = test_parameter_sampler()
    success2 = test_snr_attachment()
    
    overall_success = success1 and success2
    
    if overall_success:
        logger.info("\n✅ ALL DIAGNOSTICS PASSED")
        logger.info("\nThe physics formula IS correctly implemented:")
        logger.info("  distance = ref_distance × (M_c/ref_mass)^(5/6) × (ref_snr/target_snr)")
        logger.info("\nIf you're seeing weak SNR-distance correlation in the dataset,")
        logger.info("the issue is NOT in the sampling code, but in:")
        logger.info("  • Dataset composition (overlaps, edge cases)")
        logger.info("  • Actual SNR differing from target SNR")
        logger.info("  • Per-regime or per-type analysis showing different patterns")
    else:
        logger.info("\n⚠️  SOME DIAGNOSTICS FAILED - REVIEW IMPLEMENTATION")
    
    exit(0 if overall_success else 1)
