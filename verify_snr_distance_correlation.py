#!/usr/bin/env python3
"""
Verify SNR-distance correlation in dataset to diagnose physics violations.
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_samples(data_path, max_samples=None):
    """Load samples from pkl chunk files in data directory with flexible metadata extraction."""
    data_path = Path(data_path)
    
    snrs = []
    distances = []
    event_types = []
    
    chunk_files = sorted(data_path.glob('chunk_*.pkl'))
    logger.info(f"Found {len(chunk_files)} chunk files")
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'rb') as f:
                chunk = pickle.load(f)
            
            # chunk is a list of samples
            if chunk is None:
                logger.debug(f"Empty chunk in {chunk_file}")
                continue
                
            if not isinstance(chunk, list):
                logger.warning(f"Chunk format unexpected in {chunk_file}")
                continue
            
            for sample in chunk:
                if max_samples and len(snrs) >= max_samples:
                    break
                
                if sample is None:
                    continue
                
                # Extract SNR and distance - try multiple locations
                event_type = sample.get('type', 'unknown') if isinstance(sample, dict) else 'unknown'
                
                snr = None
                distance = None
                
                # Try to extract from metadata (new format)
                metadata = sample.get('metadata', {})
                if metadata:
                    snr = metadata.get('target_snr', None)
                    distance = metadata.get('luminosity_distance', None)
                
                # If not found in metadata, try parameters
                if snr is None or distance is None:
                    params_list = sample.get('parameters', [])
                    if params_list:
                        params = params_list[0] if isinstance(params_list, list) else params_list
                        if isinstance(params, dict):
                            if snr is None:
                                snr = params.get('target_snr', None) or params.get('network_snr', None)
                            if distance is None:
                                distance = params.get('luminosity_distance', None)
                
                # Validate values
                if snr is not None and distance is not None:
                    try:
                        snr_float = float(snr)
                        dist_float = float(distance)
                        
                        # Sanity check ranges
                        if 5.0 <= snr_float <= 500.0 and 1.0 <= dist_float <= 10000.0:
                            snrs.append(snr_float)
                            distances.append(dist_float)
                            event_types.append(event_type)
                    except (TypeError, ValueError):
                        continue
            
            if max_samples and len(snrs) >= max_samples:
                break
        except Exception as e:
            logger.debug(f"Error loading {chunk_file}: {e}")
            continue
    
    return np.array(snrs), np.array(distances), event_types

def analyze_correlation(snrs, distances, event_types, verbose=False):
    """Analyze SNR-distance correlation."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"SNR-DISTANCE CORRELATION ANALYSIS")
    logger.info(f"{'='*80}\n")
    
    logger.info(f"Total samples: {len(snrs)}")
    logger.info(f"\nSNR Statistics:")
    logger.info(f"  Mean: {np.mean(snrs):.2f}")
    logger.info(f"  Std:  {np.std(snrs):.2f}")
    logger.info(f"  Range: [{np.min(snrs):.1f}, {np.max(snrs):.1f}]")
    
    logger.info(f"\nDistance Statistics:")
    logger.info(f"  Mean: {np.mean(distances):.1f} Mpc")
    logger.info(f"  Std:  {np.std(distances):.1f} Mpc")
    logger.info(f"  Range: [{np.min(distances):.1f}, {np.max(distances):.1f}] Mpc")
    
    # Overall correlation
    pearson_r, pearson_p = pearsonr(snrs, distances)
    spearman_r, spearman_p = spearmanr(snrs, distances)
    
    logger.info(f"\n{'─'*80}")
    logger.info(f"OVERALL CORRELATION (all samples)")
    logger.info(f"{'─'*80}")
    logger.info(f"Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    logger.info(f"Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")
    
    target = -0.78
    status = "✅ GOOD" if pearson_r < -0.5 else "❌ BROKEN"
    logger.info(f"\nTarget: {target:.2f}")
    logger.info(f"Status: {status}")
    
    # Per-event-type correlation
    logger.info(f"\n{'─'*80}")
    logger.info(f"CORRELATION BY EVENT TYPE")
    logger.info(f"{'─'*80}")
    
    event_types_unique = list(set(event_types))
    for event_type in sorted(event_types_unique):
        mask = np.array(event_types) == event_type
        snrs_evt = snrs[mask]
        distances_evt = distances[mask]
        
        if len(snrs_evt) < 2:
            logger.info(f"{event_type}: only {len(snrs_evt)} sample(s), skipping")
            continue
        
        r, p = pearsonr(snrs_evt, distances_evt)
        logger.info(f"{event_type} (n={len(snrs_evt):4d}): r={r:+.4f} (p={p:.2e})")
    
    # SNR regime analysis
    logger.info(f"\n{'─'*80}")
    logger.info(f"CORRELATION BY SNR REGIME")
    logger.info(f"{'─'*80}")
    
    regimes = [
        ("Weak (5-10)", 5, 10),
        ("Low (10-20)", 10, 20),
        ("Medium (20-40)", 20, 40),
        ("High (40-70)", 40, 70),
        ("Loud (>70)", 70, np.inf)
    ]
    
    for regime_name, snr_min, snr_max in regimes:
        mask = (snrs >= snr_min) & (snrs < snr_max)
        snrs_regime = snrs[mask]
        distances_regime = distances[mask]
        
        if len(snrs_regime) < 2:
            logger.info(f"{regime_name}: {len(snrs_regime)} samples (too few)")
            continue
        
        r, p = pearsonr(snrs_regime, distances_regime)
        logger.info(f"{regime_name}: r={r:+.4f} ({len(snrs_regime)} samples)")
    
    # Detailed diagnosis
    logger.info(f"\n{'='*80}")
    logger.info(f"DIAGNOSIS")
    logger.info(f"{'='*80}\n")
    
    if pearson_r > -0.3:
        logger.info("🔴 CRITICAL: SNR-distance correlation is BROKEN")
        logger.info("   • SNR and distance are nearly independent")
        logger.info("   • Network cannot learn physics relationship")
        logger.info("   • Fix: Resample distance from target_snr using formula:")
        logger.info("     distance = ref_d * (Mc/M_ref)^(5/6) * (ref_snr / target_snr)")
        logger.info("   • Then regenerate 50K dataset")
        return False
    elif pearson_r < -0.5:
        logger.info("✅ GOOD: SNR-distance correlation is healthy")
        logger.info("   • Physics relationship is present in data")
        logger.info("   • Network should be able to learn")
        return True
    else:
        logger.info("⚠️  MARGINAL: SNR-distance correlation is weak")
        logger.info("   • Some physics relationship present, but not strong")
        logger.info("   • Consider regenerating with stronger coupling")
        return False

def main():
    parser = argparse.ArgumentParser(description='Verify SNR-distance correlation')
    parser.add_argument('--data-path', type=str, required=True, help='Path to train data directory')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to analyze')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    snrs, distances, event_types = load_samples(args.data_path, args.max_samples)
    
    if len(snrs) == 0:
        logger.error("No samples loaded!")
        return False
    
    success = analyze_correlation(snrs, distances, event_types, args.verbose)
    
    logger.info(f"\n{'='*80}")
    if success:
        logger.info("✅ Dataset physics is GOOD - Safe to train")
    else:
        logger.info("❌ Dataset physics is BROKEN - DO NOT TRAIN")
    logger.info(f"{'='*80}\n")
    
    return success

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
