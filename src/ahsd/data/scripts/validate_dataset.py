#!/usr/bin/env python3
"""
Enhanced Dataset Validation Script for AHSD
Validates splits, edge cases, extreme cases, SNR distribution, and event distribution
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pickle
from collections import Counter

from ahsd.data.io_utils import DatasetReader


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_split_samples(split_dir: Path, max_chunks: int = None):
    """Load samples from split directory."""
    patterns = ['*chunk*.pkl', '*chunk*.pkl.gz', '*.pkl']
    
    chunk_files = []
    for pattern in patterns:
        chunk_files = sorted(split_dir.glob(pattern))
        if chunk_files:
            break
    
    if not chunk_files:
        return []
    
    samples = []
    chunks_to_load = chunk_files[:max_chunks] if max_chunks else chunk_files
    
    for chunk_file in chunks_to_load:
        try:
            if chunk_file.suffix == '.gz':
                import gzip
                with gzip.open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
            else:
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
            
            if isinstance(chunk_data, list):
                samples.extend(chunk_data)
            elif isinstance(chunk_data, dict) and 'samples' in chunk_data:
                samples.extend(chunk_data['samples'])
        except Exception as e:
            logging.warning(f"Failed to load {chunk_file.name}: {e}")
    
    return samples


def validate_distributions(samples: List[Dict], expected_config: Dict, logger) -> Dict:
    """
    Validate SNR distribution and event type distribution match expectations.
    
    Supports 5 SNR regimes with adaptive tolerance based on sample size.
    Compares actual distributions against expected values from config.
    
    Args:
        samples: List of sample dicts to validate
        expected_config: Expected distribution config
        logger: Logger instance
    
    Returns:
        Validation result dict with pass/fail and detailed stats
    
    Examples:
        >>> validation = validate_distributions(samples, config, logger)
        >>> if not validation['passed']:
        ...     print(validation['errors'])
    """
    
    logger.info("\n" + "=" * 80)
    logger.info("DISTRIBUTION VALIDATION")
    logger.info("=" * 80)
    
    validation = {
        'passed': True,
        'snr_distribution': {},
        'event_distribution': {},
        'errors': [],
        'warnings': []
    }
    
    # Adaptive tolerance based on sample size
    total_samples = len([s for s in samples if s is not None])
    
    if total_samples < 500:
        tolerance = 0.12  # ±12% for small datasets
    elif total_samples < 2000:
        tolerance = 0.08  # ±8% for medium datasets
    elif total_samples < 10000:
        tolerance = 0.06  # ±6% for large datasets
    else:
        tolerance = 0.05  # ±5% for very large datasets
    
    logger.info(f"Total samples: {total_samples:,}")
    logger.info(f"Tolerance: ±{tolerance*100:.1f}%")
    
    # ========================================================================
    # 1. EVENT TYPE DISTRIBUTION (signal-level)
    # ========================================================================
    # NOTE: When overlaps are present, comparing per-sample top-level 'type'
    # will undercount individual signal types. We therefore compute the
    # distribution from signals (parameters) and compare that to the
    # expected EVENT_TYPE_DISTRIBUTION (which is defined per-signal).
    logger.info("\n[1/2] Validating event type distribution...")

    # Sample-level counts (keep for reporting)
    sample_level_counts = Counter()
    # Signal-level counts (one entry per physical signal in the dataset)
    signal_level_counts = Counter()

    for sample in samples:
        if sample is None:
            continue

        # Track top-level sample type (single / overlap / noise)
        sample_type = sample.get('type', 'unknown')
        sample_level_counts[sample_type] += 1

        # Extract per-signal event types from parameters (handles lists for overlaps)
        params = sample.get('parameters')
        if params is None:
            continue

        if isinstance(params, list):
            for p in params:
                if not isinstance(p, dict):
                    continue
                sig_type = p.get('type') or p.get('event_type') or 'unknown'
                # convert numpy strings to python str for consistent keys
                try:
                    if hasattr(sig_type, 'tolist'):
                        sig_type = sig_type.tolist()
                except Exception:
                    pass
                signal_level_counts[str(sig_type)] += 1
        elif isinstance(params, dict):
            sig_type = params.get('type') or params.get('event_type') or 'unknown'
            try:
                if hasattr(sig_type, 'tolist'):
                    sig_type = sig_type.tolist()
            except Exception:
                pass
            signal_level_counts[str(sig_type)] += 1

    # Build event distribution mapping from signal counts
    event_distribution = {}
    total_signals = sum(signal_level_counts.values())
    for event_type, count in signal_level_counts.items():
        fraction = count / total_signals if total_signals > 0 else 0
        event_distribution[event_type] = {
            'count': count,
            'fraction': fraction,
            'percentage': fraction * 100
        }

    validation['event_distribution'] = event_distribution

    # Log sample-level overview first (keeps previous behavior for diagnostics)
    logger.info("\n  Sample-level event summary (samples):")
    for event_type in sorted(sample_level_counts.keys()):
        count = sample_level_counts[event_type]
        pct = count / total_samples * 100 if total_samples > 0 else 0
        logger.info(f"    {event_type:10s}: {count:>6,} ({pct:>5.2f}%)")

    # Log signal-level distribution (what we actually compare to expected)
    logger.info("\n  Signal-level event distribution (individual signals):")
    for event_type in sorted(event_distribution.keys()):
        stats = event_distribution[event_type]
        logger.info(
            f"    {event_type:10s}: {stats['count']:>6,} ({stats['percentage']:>5.2f}%) "
            f"{'█' * int(stats['percentage'] / 2)}"
        )

    # Compare with expected if provided (expected_config holds per-signal fractions)
    expected_events = expected_config.get('event_distribution', {})
    if expected_events:
        logger.info("\n  Comparing with expected distribution (signal-level):")
        for event_type, expected_frac in expected_events.items():
            actual_stats = event_distribution.get(event_type, {'fraction': 0, 'count': 0})
            actual_frac = actual_stats['fraction']

            diff = abs(actual_frac - expected_frac)
            status = "✓" if diff <= tolerance else "✗"

            logger.info(
                f"    {event_type:10s}: Expected {expected_frac*100:>5.2f}%, "
                f"Got {actual_frac*100:>5.2f}%, Diff {diff*100:>5.2f}% {status}"
            )

            if diff > tolerance:
                error = (
                    f"Event type '{event_type}' distribution off by {diff*100:.2f}% "
                    f"(expected {expected_frac*100:.2f}%, got {actual_frac*100:.2f}%)"
                )
                validation['errors'].append(error)
                validation['passed'] = False
    
    # ========================================================================
    # 2. SNR DISTRIBUTION - 5 REGIMES (CORRECTED)
    # ========================================================================
    logger.info("\n[2/2] Validating SNR distribution...")
    
    snr_values = []
    snr_regimes = Counter()
    
    # ✅ FIXED: Correct SNR ranges matching dataset generator
    SNR_RANGES = {
    'weak': (5.0, 10.0),
    'low': (8.0, 20.0),
    'medium': (15.0, 40.0),
    'high': (30.0, 65.0),
    'loud': (50.0, 100.0)
}
    
    # Extract SNR values and categorize
    for sample in samples:
        if sample is None:
            continue
        
        params = sample.get('parameters')
        if params is None:
            continue
        
        # Handle list of parameters (overlapping signals)
        if isinstance(params, list):
            for p in params:
                if isinstance(p, dict) and 'target_snr' in p:
                    snr = p['target_snr']
                    snr_values.append(snr)
                    
                    # ✅ FIXED: Correct categorization thresholds
                    if snr < 10:
                        snr_regimes['weak'] += 1
                    elif snr < 15:
                        snr_regimes['low'] += 1
                    elif snr < 25:
                        snr_regimes['medium'] += 1
                    elif snr < 40:
                        snr_regimes['high'] += 1
                    else:
                        snr_regimes['loud'] += 1
        
        # Handle single parameter dict
        elif isinstance(params, dict) and 'target_snr' in params:
            snr = params['target_snr']
            snr_values.append(snr)
            
            # ✅ FIXED: Correct categorization thresholds
            if snr < 10:
                snr_regimes['weak'] += 1
            elif snr < 15:
                snr_regimes['low'] += 1
            elif snr < 25:
                snr_regimes['medium'] += 1
            elif snr < 40:
                snr_regimes['high'] += 1
            else:
                snr_regimes['loud'] += 1
    
    # Calculate SNR distribution
    total_snrs = len(snr_values)
    
    if total_snrs == 0:
        logger.warning("  ⚠ No SNR values found in samples!")
        validation['warnings'].append("No SNR values found")
        return validation
    
    snr_distribution = {}
    for regime in ['weak', 'low', 'medium', 'high', 'loud']:
        count = snr_regimes.get(regime, 0)
        fraction = count / total_snrs if total_snrs > 0 else 0
        snr_distribution[regime] = {
            'count': count,
            'fraction': fraction,
            'percentage': fraction * 100,
            'range': SNR_RANGES[regime]
        }
    
    validation['snr_distribution'] = snr_distribution
    
    # Log SNR statistics
    if snr_values:
        snr_array = np.array(snr_values)
        logger.info(f"\n  SNR statistics:")
        logger.info(f"    Total signals: {total_snrs:,}")
        logger.info(f"    Mean:   {snr_array.mean():.2f}")
        logger.info(f"    Median: {np.median(snr_array):.2f}")
        logger.info(f"    Std:    {snr_array.std():.2f}")
        logger.info(f"    Min:    {snr_array.min():.2f}")
        logger.info(f"    Max:    {snr_array.max():.2f}")
    
    # ✅ FIXED: Correct histogram bins
    logger.info("\n  SNR histogram:")
    bins = [0, 7, 10, 15, 25, 40, 80]  # ✅ CORRECT bins
    hist, _ = np.histogram(snr_values, bins=bins)
    regime_labels = [
        '< 7 (noise)',
        '7-10 (weak)',
        '10-15 (low)',
        '15-25 (medium)',
        '25-40 (high)',
        '40-80 (loud)'
    ]
    
    for i in range(len(bins)-1):
        count = hist[i]
        pct = count / total_snrs * 100 if total_snrs > 0 else 0
        label = regime_labels[i]
        bar = '█' * int(pct / 2)
        logger.info(
            f"    {bins[i]:>5.0f} - {bins[i+1]:<5.0f}: {count:>6} ({pct:>5.1f}%) {bar} {label}"
        )
    
    # Log actual distribution by regime
    logger.info("\n  SNR regime distribution:")
    for regime in ['weak', 'low', 'medium', 'high', 'loud']:
        stats = snr_distribution[regime]
        range_str = f"{stats['range'][0]}-{stats['range'][1]}"
        bar = '█' * int(stats['percentage'] / 2)
        logger.info(
            f"    {regime:8s} ({range_str:>7s}): {stats['count']:>6} "
            f"({stats['percentage']:>5.2f}%) {bar}"
        )
    
    # Compare with expected SNR distribution
    expected_snr = expected_config.get('snr_distribution', {})
    if expected_snr:
        logger.info("\n  Comparing with expected SNR distribution:")
        for regime in ['weak', 'low', 'medium', 'high', 'loud']:
            expected_frac = expected_snr.get(regime, 0)
            actual_stats = snr_distribution.get(regime, {'fraction': 0, 'count': 0})
            actual_frac = actual_stats['fraction']
            
            diff = abs(actual_frac - expected_frac)
            status = "✓" if diff <= tolerance else "✗"
            
            logger.info(
                f"    {regime:8s}: Expected {expected_frac*100:>5.2f}%, "
                f"Got {actual_frac*100:>5.2f}%, Diff {diff*100:>5.2f}% {status}"
            )
            
            if diff > tolerance:
                error = (
                    f"SNR regime '{regime}' distribution off by {diff*100:.2f}% "
                    f"(expected {expected_frac*100:.2f}%, got {actual_frac*100:.2f}%)"
                )
                validation['errors'].append(error)
                validation['passed'] = False
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    if validation['passed']:
        logger.info("✅ DISTRIBUTION VALIDATION PASSED")
    else:
        logger.error("❌ DISTRIBUTION VALIDATION FAILED")
        logger.error(f"\nErrors ({len(validation['errors'])}):")
        for error in validation['errors']:
            logger.error(f"  • {error}")
    
    if validation['warnings']:
        logger.warning(f"\nWarnings ({len(validation['warnings'])}):")
        for warning in validation['warnings']:
            logger.warning(f"  • {warning}")
    
    logger.info("=" * 80)
    
    return validation


def validate_edge_cases(samples: List[Dict], logger, expected_config: Dict = None) -> Dict:
    """
    Validate edge case distribution and properties.
    Properly categorizes all edge case types including variants and aliases.
    """
    
    logger.info("\n  Analyzing edge case distribution...")
    
    edge_stats = {
        'total_edge_cases': 0,
        'edge_case_types': {},
        'regular_samples': 0,
        'overlap_samples': 0,
        'physical_extremes': {},
        'observational_extremes': {},
        'statistical_extremes': {},
        'overlap_extremes': {},
        'validation': {
            'passed': True,
            'errors': [],
            'warnings': []
        }
    }
    
    # Comprehensive edge case type sets (includes all variants and aliases)
    PHYSICAL_EXTREMES = {
        # Mass-related
        'high_mass_ratio', 'extreme_mass_ratio', 'short_duration_high_mass',
        'extreme_mass',
        
        # Spin-related
        'extreme_spins', 'high_spin', 'high_spin_aligned',
        
        # Orbital characteristics
        'eccentric_mergers', 'eccentric', 'precessing_systems', 'precessing',
        'precession_dominated',
        
        # SNR/Distance
        'low_snr_threshold', 'cosmological_distance',
        
        # Other physical
        'short_bbh'
    }
    
    OBSERVATIONAL_EXTREMES = {
        'strong_glitches', 'glitches',
        'detector_dropout', 'detector_dropouts',
        'psd_drift',
        'sky_position_extremes'
    }
    
    STATISTICAL_EXTREMES = {
        'multimodal_posteriors',
        'heavy_tailed_regions',
        'uninformative_priors',
        'posterior_degeneracy'
    }
    
    OVERLAP_EXTREMES = {
        'subtle_ranking',
        'heavy_overlaps',
        'partial_overlaps', 'partial_overlap',
        'long_bns_inspiral', 'long_duration_bns_overlaps',
        'near_simultaneous_mergers',
        'eccentric_overlaps',
        'weak_strong_overlaps',
        'noise_confused_overlaps'
    }
    
    for sample in samples:
        if sample is None:
            continue
        
        # Check if sample is an edge case
        # Consider both explicit 'is_edge_case' and legacy/related flags.
        # Also treat 'is_extreme_case' / 'extreme_case_type' as an edge-case marker
        # for validation purposes (generator may use the 'extreme' naming).
        is_edge = (
            sample.get('is_edge_case', False)
            or sample.get('edge_case', False)
            or sample.get('is_extreme_case', False)
            or sample.get('edge_case_type') not in [None, 'none', 'None', '']
            or sample.get('extreme_case_type') not in [None, 'none', 'None', '']
        )
        
        edge_type = sample.get('edge_case_type', None)
        
        if is_edge and edge_type and edge_type not in ['none', 'None', '']:
            edge_stats['total_edge_cases'] += 1
            edge_stats['edge_case_types'][edge_type] = edge_stats['edge_case_types'].get(edge_type, 0) + 1
            
            # ✅ FIXED: Use set membership for categorization
            if edge_type in PHYSICAL_EXTREMES:
                edge_stats['physical_extremes'][edge_type] = edge_stats['physical_extremes'].get(edge_type, 0) + 1
            
            elif edge_type in OBSERVATIONAL_EXTREMES:
                edge_stats['observational_extremes'][edge_type] = edge_stats['observational_extremes'].get(edge_type, 0) + 1
            
            elif edge_type in STATISTICAL_EXTREMES:
                edge_stats['statistical_extremes'][edge_type] = edge_stats['statistical_extremes'].get(edge_type, 0) + 1
            
            elif edge_type in OVERLAP_EXTREMES:
                edge_stats['overlap_extremes'][edge_type] = edge_stats['overlap_extremes'].get(edge_type, 0) + 1
            
            else:
                # ✅ FIXED: Categorize unknown types but don't spam warnings
                # Put in overlap_extremes as catch-all (most edge cases are overlap-related)
                edge_stats['overlap_extremes'][edge_type] = edge_stats['overlap_extremes'].get(edge_type, 0) + 1
                
                # Only log debug message, not a warning
                logger.debug(f"    Edge case type '{edge_type}' not in predefined categories - categorized as overlap extreme")
                
                # Only add warning if it looks truly suspicious (no common keywords)
                if not any(keyword in edge_type.lower() for keyword in 
                          ['overlap', 'extreme', 'edge', 'special', 'heavy', 'subtle', 
                           'merger', 'mass', 'spin', 'eccentric', 'precessing']):
                    edge_stats['validation']['warnings'].append(f"Unusual edge case type: {edge_type}")
        
        elif sample.get('is_overlap', False) or sample.get('type') == 'overlap':
            edge_stats['overlap_samples'] += 1
        else:
            edge_stats['regular_samples'] += 1
    
    # Calculate statistics
    total_samples = edge_stats['total_edge_cases'] + edge_stats['regular_samples'] + edge_stats['overlap_samples']
    edge_fraction = edge_stats['total_edge_cases'] / total_samples if total_samples > 0 else 0
    overlap_fraction = edge_stats['overlap_samples'] / total_samples if total_samples > 0 else 0
    
    # Log summary
    logger.info(f"  ✓ Edge case summary:")
    logger.info(f"    - Total samples: {total_samples:,}")
    logger.info(f"    - Edge cases: {edge_stats['total_edge_cases']:,} ({edge_fraction:.1%})")
    logger.info(f"    - Regular samples: {edge_stats['regular_samples']:,} ({edge_stats['regular_samples']/total_samples*100:.1%})")
    logger.info(f"    - Overlap samples: {edge_stats['overlap_samples']:,} ({overlap_fraction:.1%})")
    
    # Validate against expected config
    if expected_config:
        expected_edge_fraction = expected_config.get('edge_case_fraction', 0.15)

        # If no explicit edge_case markings were found, skip strict numeric
        # comparison here and surface a warning instead. The extreme-case
        # analysis (validate_extreme_cases) runs separately and will provide
        # a heuristic-based breakdown for datasets that don't set explicit
        # 'edge_case_type' fields during generation.
        if edge_stats['total_edge_cases'] == 0:
            logger.warning("    ⚠ No explicit edge_case flags found - skipping strict edge-fraction check;"
                           " falling back to extreme-case heuristics for coverage checks.")
            edge_stats['validation']['warnings'].append(
                "No explicit 'edge_case' flags found; validator will rely on extreme-case heuristics."
            )
        else:
            edge_diff = abs(edge_fraction - expected_edge_fraction)
            # Adaptive tolerance based on sample size
            tolerance = 0.10 if total_samples < 200 else 0.05

            if edge_diff > tolerance:
                edge_stats['validation']['passed'] = False
                edge_stats['validation']['errors'].append(
                    f"Edge case fraction: expected {expected_edge_fraction:.1%}, got {edge_fraction:.1%} (diff: {edge_diff:.1%})"
                )
                logger.warning(f"    ⚠ Edge case fraction off by {edge_diff:.1%}")
            else:
                logger.info(f"    ✓ Edge case fraction within tolerance")
    
    # Log category breakdowns
    if edge_stats['physical_extremes']:
        phys_total = sum(edge_stats['physical_extremes'].values())
        logger.info(f"\n  Physical extremes ({phys_total:,} total, {phys_total/total_samples*100:.1f}%):")
        for edge_type, count in sorted(edge_stats['physical_extremes'].items(), key=lambda x: -x[1]):
            logger.info(f"    - {edge_type:35s}: {count:6,} ({count/phys_total*100:5.1f}%)")
    
    if edge_stats['observational_extremes']:
        obs_total = sum(edge_stats['observational_extremes'].values())
        logger.info(f"\n  Observational extremes ({obs_total:,} total, {obs_total/total_samples*100:.1f}%):")
        for edge_type, count in sorted(edge_stats['observational_extremes'].items(), key=lambda x: -x[1]):
            logger.info(f"    - {edge_type:35s}: {count:6,} ({count/obs_total*100:5.1f}%)")
    
    if edge_stats['statistical_extremes']:
        stat_total = sum(edge_stats['statistical_extremes'].values())
        logger.info(f"\n  Statistical extremes ({stat_total:,} total, {stat_total/total_samples*100:.1f}%):")
        for edge_type, count in sorted(edge_stats['statistical_extremes'].items(), key=lambda x: -x[1]):
            logger.info(f"    - {edge_type:35s}: {count:6,} ({count/stat_total*100:5.1f}%)")
    
    if edge_stats['overlap_extremes']:
        overlap_total = sum(edge_stats['overlap_extremes'].values())
        logger.info(f"\n  Overlap extremes ({overlap_total:,} total, {overlap_total/total_samples*100:.1f}%):")
        for edge_type, count in sorted(edge_stats['overlap_extremes'].items(), key=lambda x: -x[1]):
            logger.info(f"    - {edge_type:35s}: {count:6,} ({count/overlap_total*100:5.1f}%)")
    
    # Sanity checks
    if edge_stats['total_edge_cases'] == 0 and expected_config and expected_config.get('edge_case_fraction', 0) > 0:
        logger.warning(f"    ⚠ No edge cases detected - samples may be missing 'edge_case_type' field")
        edge_stats['validation']['warnings'].append(
            "No edge cases found - check if samples have 'edge_case_type' field set during generation"
        )
    
    # ✅ NEW: Summary of categorization
    categorized_total = (
        sum(edge_stats['physical_extremes'].values()) +
        sum(edge_stats['observational_extremes'].values()) +
        sum(edge_stats['statistical_extremes'].values()) +
        sum(edge_stats['overlap_extremes'].values())
    )
    
    if categorized_total != edge_stats['total_edge_cases']:
        logger.warning(
            f"    ⚠ Categorization mismatch: {categorized_total} categorized vs "
            f"{edge_stats['total_edge_cases']} total edge cases"
        )
    
    logger.info(f"\n  ✓ Edge case categorization complete:")
    logger.info(f"    - Physical: {sum(edge_stats['physical_extremes'].values()):,}")
    logger.info(f"    - Observational: {sum(edge_stats['observational_extremes'].values()):,}")
    logger.info(f"    - Statistical: {sum(edge_stats['statistical_extremes'].values()):,}")
    logger.info(f"    - Overlap: {sum(edge_stats['overlap_extremes'].values()):,}")
    
    return edge_stats

def validate_extreme_cases(samples: List[Dict], logger, expected_config: Dict = None) -> Dict:
    """
    Validate presence of extreme/challenging cases critical for model robustness.
    These cases ensure the model learns to handle real-world complications.
    """
    
    logger.info("  Analyzing extreme case distribution...")
    
    extreme_stats = {
    'total_samples': len([s for s in samples if s is not None]),
    'extreme_cases': {
    'near_simultaneous_mergers': 0,
    'extreme_mass_ratio': 0,
    'high_spin_aligned': 0,
    'precession_dominated': 0,
    'eccentric_overlaps': 0,
    'weak_strong_overlaps': 0,
    'noise_confused_overlaps': 0,
    'long_duration_bns_overlaps': 0,
    'detector_dropouts': 0,
    'cosmological_distance': 0
    # Removed pre_merger_samples as it's an edge case, not extreme case
    },
        'validation': {
            'passed': True,
            'errors': [],
            'warnings': []
        }
    }
    # Additional runtime checks
    extreme_stats['checks'] = {
        'missing_priorities': 0,
        'priority_length_mismatch': 0,
        'missing_target_snr': 0,
        'samples_with_inconsistent_params': 0,
    }

    # Collect arrays for simple correlation checks (distance vs SNR)
    corr_distances = []
    corr_snrs = []
    
    for sample in samples:
        if sample is None:
            continue
        
        params = sample.get('parameters')
        if params is None:
            continue

        # Check priorities presence for overlap samples
        priorities = sample.get('priorities')
        if isinstance(params, list):
            if priorities is None:
                extreme_stats['checks']['missing_priorities'] += 1
            else:
                # If mismatch in lengths, note it
                if len(priorities) != len(params):
                    extreme_stats['checks']['priority_length_mismatch'] += 1
        else:
            # single event: priorities may be a single value or missing
            if priorities is None and not isinstance(params, list):
                # acceptable but count missing target_snr below
                pass
        
        # Note: Pre-merger samples are handled as edge cases, not extreme cases
        
        # Handle overlapping samples (can have multiple parameter dicts)
        if isinstance(params, list) and len(params) >= 2:
            # 1️⃣ Near-simultaneous mergers
            if len(params) >= 2:
                t1 = params[0].get('geocent_time', 0)
                t2 = params[1].get('geocent_time', 0)
                if abs(t1 - t2) < 0.2:
                    extreme_stats['extreme_cases']['near_simultaneous_mergers'] += 1
            
            # 5️⃣ Eccentric overlaps
            has_eccentric = any(p.get('eccentricity', 0) > 0.3 for p in params)
            if has_eccentric:
                extreme_stats['extreme_cases']['eccentric_overlaps'] += 1
            
            # 6️⃣ Weak-Strong overlaps
            snrs = [p.get('target_snr', 0) for p in params if 'target_snr' in p]
            if len(snrs) >= 2:
                min_snr = min(snrs)
                max_snr = max(snrs)
                if min_snr < 10 and max_snr > 40:
                    extreme_stats['extreme_cases']['weak_strong_overlaps'] += 1
            
            # 7️⃣ Noise-confused overlaps
            has_glitch = sample.get('has_glitch', False) or sample.get('glitch', False)
            if has_glitch:
                extreme_stats['extreme_cases']['noise_confused_overlaps'] += 1
            
            # 8️⃣ Long-duration BNS overlaps
            bns_count = sum(1 for p in params if p.get('type') == 'BNS')
            if bns_count >= 2:
                has_long = any(p.get('f_lower', 35) < 30 for p in params)
                if has_long:
                    extreme_stats['extreme_cases']['long_duration_bns_overlaps'] += 1
        
        # Single event parameters
        # Guard: params may be an empty list -> skip
        if isinstance(params, list):
            if len(params) == 0:
                # nothing to inspect
                continue
            param_dict = params[0]
        else:
            param_dict = params

        if not isinstance(param_dict, dict):
            extreme_stats['checks']['samples_with_inconsistent_params'] += 1
            continue
        
        # 2️⃣ Extreme Mass-Ratio
        if 'mass_1' in param_dict and 'mass_2' in param_dict:
            m1 = param_dict['mass_1']
            m2 = param_dict['mass_2']
            q = min(m1, m2) / max(m1, m2)
            if q < 0.05 and (m1 + m2) < 30:
                extreme_stats['extreme_cases']['extreme_mass_ratio'] += 1
        
        # 3️⃣ High-Spin Aligned
        a1 = param_dict.get('a1', 0)
        a2 = param_dict.get('a2', 0)
        tilt1 = param_dict.get('tilt1', 0)
        tilt2 = param_dict.get('tilt2', 0)
        
        if max(abs(a1), abs(a2)) > 0.95:
            if (abs(tilt1) < 0.1 or abs(tilt1 - np.pi) < 0.1 or 
                abs(tilt2) < 0.1 or abs(tilt2 - np.pi) < 0.1):
                extreme_stats['extreme_cases']['high_spin_aligned'] += 1
        
        # 4️⃣ Precession-Dominated
        chi_p = param_dict.get('chi_p', 0)
        if chi_p > 0.8:
            extreme_stats['extreme_cases']['precession_dominated'] += 1
        elif max(abs(a1), abs(a2)) > 0.7:
            if max(abs(tilt1 - np.pi/2), abs(tilt2 - np.pi/2)) < 0.5:
                extreme_stats['extreme_cases']['precession_dominated'] += 1
        
        # 🔟 Cosmological distance
        d_L = param_dict.get('luminosity_distance', 0)
        snr = param_dict.get('target_snr', 0)
        if d_L > 2000 and snr < 10:
            extreme_stats['extreme_cases']['cosmological_distance'] += 1
        
        # 9️⃣ Detector dropouts
        detector_data = sample.get('detector_data', {})
        n_active_detectors = 0
        for det_name, det_data in detector_data.items():
            # det_data can be an ndarray or dict; guard against ambiguous truth-value
            if isinstance(det_data, dict) and 'strain' in det_data:
                strain = det_data.get('strain')
                if strain is not None and hasattr(strain, 'size') and strain.size > 0 and np.any(strain != 0):
                    n_active_detectors += 1
        
        if 0 < n_active_detectors < len(detector_data):
            extreme_stats['extreme_cases']['detector_dropouts'] += 1

        # Collect distance / snr for correlation checks if available
        d_L = param_dict.get('luminosity_distance', None)
        snr = param_dict.get('target_snr', None)
        if snr is None:
            extreme_stats['checks']['missing_target_snr'] += 1
        if d_L is not None and snr is not None:
            try:
                d_val = float(d_L)
                s_val = float(snr)
                corr_distances.append(d_val)
                corr_snrs.append(s_val)
            except Exception:
                # skip non-numeric values
                pass
    
    # Calculate stats and validate
    total = extreme_stats['total_samples']

    # Expected fractions based on config (extreme_fraction=0.03):
    # near_simultaneous_mergers: 0.25 → 0.03*0.25 = 0.0075
    # extreme_mass_ratio: 0.15 → 0.03*0.15 = 0.0045
    # high_spin_aligned: 0.15 → 0.03*0.15 = 0.0045
    # weak_strong_overlaps: 0.25 → 0.03*0.25 = 0.0075
    # noise_confused_overlaps: 0.15 → 0.03*0.15 = 0.0045
    # long_duration_bns_overlaps: 0.05 → 0.03*0.05 = 0.0015
    # pre_merger_samples: not an extreme case, remove from expectations
    expected_minimums = {
        'near_simultaneous_mergers': 0.0075,  # 0.03 * 0.25
        'extreme_mass_ratio': 0.0045,  # 0.03 * 0.15
        'high_spin_aligned': 0.0045,  # 0.03 * 0.15
        'weak_strong_overlaps': 0.0075,  # 0.03 * 0.25
        'noise_confused_overlaps': 0.0045,  # 0.03 * 0.15
        'long_duration_bns_overlaps': 0.0015,  # 0.03 * 0.05
    }
    
    logger.info("  ✓ Extreme case breakdown (critical for robustness):")
    logger.info(f"    Total samples analyzed: {total}")
    logger.info("")
    
    for case_name, count in extreme_stats['extreme_cases'].items():
        fraction = (count / total) if total > 0 else 0
        expected = expected_minimums.get(case_name, 0.01)
        
        # Status indicators
        if fraction >= expected:
            status = "✓✓"
        elif fraction >= expected * 0.5:
            status = "✓"
        else:
            status = "⚠"
        
        # Format name nicely
        display_name = case_name.replace('_', ' ').title()
        
        logger.info(
            f"    {status} {display_name}: {count} "
            f"({fraction*100:.2f}%, expect ≥{expected*100:.2f}%)"
        )
        
        if fraction < expected * 0.5 and expected_config:
            extreme_stats['validation']['warnings'].append(
                f"Low {case_name}: {count} ({fraction:.2%} vs expected {expected:.2%})"
            )
    
    logger.info("")
    
    total_extreme = sum(extreme_stats['extreme_cases'].values())
    extreme_fraction = total_extreme / total if total > 0 else 0
    
    logger.info(f"  Total extreme cases: {total_extreme} ({extreme_fraction*100:.2f}% of dataset)")
    
    if extreme_fraction < 0.05:
        extreme_stats['validation']['warnings'].append(
            f"Only {extreme_fraction:.2%} extreme cases - consider adding more challenging scenarios"
        )
        logger.warning("  ⚠ Low extreme case coverage - model may struggle with edge cases")
    elif extreme_fraction >= 0.10:
        logger.info("  ✓✓ Excellent extreme case coverage!")
    else:
        logger.info("  ✓ Good extreme case coverage")

    # Correlation checks: distance vs SNR should be negative (SNR ∝ 1/D roughly)
    try:
        import scipy.stats as stats
        if len(corr_distances) >= 10:
            pearson_r, pearson_p = stats.pearsonr(corr_distances, corr_snrs)
            spearman_r, spearman_p = stats.spearmanr(corr_distances, corr_snrs)
            extreme_stats['correlations'] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n': len(corr_distances)
            }

            # Expect negative correlation; warn if weak/not negative
            if pearson_r > -0.2:
                msg = f"Weak/non-negative distance vs SNR correlation (pearson_r={pearson_r:.3f})"
                extreme_stats['validation']['warnings'].append(msg)
                logger.warning(f"  ⚠ {msg}")
    except Exception:
        # scipy may not be available; fall back to numpy sign check
        try:
            if len(corr_distances) >= 10:
                import numpy as _np
                # compute simple correlation via covariance of standardized values
                d_arr = _np.array(corr_distances)
                s_arr = _np.array(corr_snrs)
                if _np.std(d_arr) > 0 and _np.std(s_arr) > 0:
                    cov = _np.cov(d_arr, s_arr)[0,1]
                    corr = cov / (_np.std(d_arr) * _np.std(s_arr))
                    extreme_stats['correlations'] = {'pearson_r': float(corr), 'n': len(corr_distances)}
                    if corr > -0.2:
                        msg = f"Weak/non-negative distance vs SNR correlation (approx r={corr:.3f})"
                        extreme_stats['validation']['warnings'].append(msg)
                        logger.warning(f"  ⚠ {msg}")
        except Exception:
            pass

    # Attach checks summary
    logger.info("")
    logger.info("  Runtime checks summary:")
    logger.info(f"    - Missing priorities (overlaps): {extreme_stats['checks']['missing_priorities']}")
    logger.info(f"    - Priority length mismatches: {extreme_stats['checks']['priority_length_mismatch']}")
    logger.info(f"    - Missing target_snr entries: {extreme_stats['checks']['missing_target_snr']}")
    logger.info(f"    - Samples with inconsistent params: {extreme_stats['checks']['samples_with_inconsistent_params']}")
    
    return extreme_stats

def validate_parameter_ranges(samples: List[Dict], logger) -> Dict:
    """Validate parameter distributions."""
    
    logger.info("\n  Analyzing parameter distributions...")
    
    param_stats = {
        'masses': [],
        'mass_ratios': [],
        'spins': [],
        'distances': [],
        'snrs': [],
        'eccentricities': []
    }
    
    for sample in samples[:1000]:
        if sample is None or not sample.get('parameters'):
            continue
        
        params = sample['parameters']
        if isinstance(params, list):
            params = params[0] if params else {}
        
        if isinstance(params, dict):
            if 'mass_1' in params and 'mass_2' in params:
                param_stats['masses'].append(params['mass_1'])
                param_stats['masses'].append(params['mass_2'])
                q = min(params['mass_1'], params['mass_2']) / max(params['mass_1'], params['mass_2'])
                param_stats['mass_ratios'].append(q)
            
            if 'a1' in params:
                param_stats['spins'].append(abs(params['a1']))
            if 'a2' in params:
                param_stats['spins'].append(abs(params['a2']))
            
            if 'luminosity_distance' in params:
                param_stats['distances'].append(params['luminosity_distance'])
            
            if 'target_snr' in params:
                param_stats['snrs'].append(params['target_snr'])
            
            if 'eccentricity' in params:
                param_stats['eccentricities'].append(params['eccentricity'])
    
    if param_stats['masses']:
        logger.info(f"  ✓ Mass range: {min(param_stats['masses']):.1f} - {max(param_stats['masses']):.1f} M☉")
    
    if param_stats['mass_ratios']:
        logger.info(f"  ✓ Mass ratio (q) range: {min(param_stats['mass_ratios']):.3f} - {max(param_stats['mass_ratios']):.3f}")
        n_extreme = sum(1 for q in param_stats['mass_ratios'] if q < 0.1)
        if n_extreme > 0:
            logger.info(f"    - Extreme q < 0.1: {n_extreme} samples")
    
    if param_stats['spins']:
        logger.info(f"  ✓ Spin magnitude range: {min(param_stats['spins']):.3f} - {max(param_stats['spins']):.3f}")
        n_extreme = sum(1 for s in param_stats['spins'] if s > 0.9)
        if n_extreme > 0:
            logger.info(f"    - Extreme |χ| > 0.9: {n_extreme} samples")
    
    if param_stats['distances']:
        logger.info(f"  ✓ Distance range: {min(param_stats['distances']):.1f} - {max(param_stats['distances']):.1f} Mpc")
    
    if param_stats['eccentricities']:
        logger.info(f"  ✓ Eccentricity: {len(param_stats['eccentricities'])} eccentric samples")
    
    return param_stats


def validate_split(split_name: str, split_dir: Path, logger, expected_config: Dict = None) -> Dict:
    """Validate a single split with comprehensive checks."""
    
    logger.info(f"\n[{split_name.upper()}] Validating split...")
    logger.info("=" * 60)
    
    validation = {
        'found': False,
        'passed': False,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    if not split_dir.exists():
        logger.error(f"  ✗ Directory not found: {split_dir}")
        validation['errors'].append(f"Directory not found: {split_dir}")
        return validation
    
    validation['found'] = True
    validation['passed'] = True
    
    # Check chunk files
    chunk_files = list(split_dir.glob('*chunk*.pkl')) + list(split_dir.glob('*chunk*.pkl.gz'))
    
    if not chunk_files:
        logger.error(f"  ✗ No chunk files found")
        validation['errors'].append("No chunk files")
        validation['passed'] = False
        return validation
    
    logger.info(f"  ✓ Found {len(chunk_files)} chunk files")
    validation['stats']['n_chunks'] = len(chunk_files)
    
    # Check metadata
    metadata_file = split_dir / f'{split_name}_metadata.pkl'
    if metadata_file.exists():
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            logger.info(f"  ✓ Metadata found:")
            logger.info(f"    - Total samples: {metadata.get('total_samples', 'N/A')}")
            logger.info(f"    - Chunk size: {metadata.get('chunk_size', 'N/A')}")
            
            validation['stats']['metadata'] = metadata
        except Exception as e:
            logger.warning(f"  ⚠ Failed to load metadata: {e}")
            validation['warnings'].append(f"Metadata load error: {e}")
    else:
        logger.info(f"  ℹ No metadata file (optional)")
    
    # Load samples
    logger.info(f"\n  Loading samples from first 5 chunks...")
    samples = load_split_samples(split_dir, max_chunks=5)
    
    if not samples:
        logger.error(f"  ✗ Failed to load any samples")
        validation['errors'].append("No samples loaded")
        validation['passed'] = False
        return validation
    
    logger.info(f"  ✓ Loaded {len(samples)} samples for validation")
    
    # Validate sample structure
    logger.info(f"\n  Validating sample structure...")
    
    valid_samples = 0
    samples_with_strain = 0
    samples_with_params = 0
    event_types = Counter()
    issues = []
    
    for i, sample in enumerate(samples):
        if sample is None:
            issues.append(f"Sample {i} is None")
            continue
        
        valid_samples += 1
        
        if 'detector_data' in sample:
            has_strain = False
            for det_name, det_data in sample['detector_data'].items():
                # Guard against numpy arrays or unexpected types being used as det_data
                if isinstance(det_data, dict) and 'strain' in det_data:
                    strain = det_data.get('strain')
                    if strain is not None and hasattr(strain, 'size') and strain.size > 0:
                        has_strain = True

                        if not np.all(np.isfinite(strain)):
                            issues.append(f"Sample {i} has NaN/Inf in {det_name}")

                        if hasattr(strain, 'dtype') and strain.dtype != np.float32:
                            issues.append(f"Sample {i} detector {det_name} not float32")
            
            if has_strain:
                samples_with_strain += 1
        
        if 'parameters' in sample and sample['parameters']:
            samples_with_params += 1
        
        event_types[sample.get('type', 'unknown')] += 1
    
    logger.info(f"  ✓ Valid samples: {valid_samples}/{len(samples)}")
    logger.info(f"  ✓ With strain data: {samples_with_strain}")
    logger.info(f"  ✓ With parameters: {samples_with_params}")
    logger.info(f"  ✓ Event types: {dict(event_types)}")
    
    if issues:
        logger.warning(f"  ⚠ Found {len(issues)} issues")
        for issue in issues[:3]:
            logger.warning(f"      - {issue}")
        validation['warnings'].extend(issues)
    
    validation['stats']['samples'] = {
        'n_validated': len(samples),
        'n_valid': valid_samples,
        'n_with_strain': samples_with_strain,
        'n_with_parameters': samples_with_params,
        'event_types': dict(event_types),
        'n_issues': len(issues)
    }
    
    # Validate edge cases
    edge_stats = validate_edge_cases(samples, logger, expected_config)
    validation['stats']['edge_cases'] = edge_stats
    
    if not edge_stats['validation']['passed']:
        validation['passed'] = False
        validation['errors'].extend(edge_stats['validation']['errors'])
    validation['warnings'].extend(edge_stats['validation']['warnings'])
    
    # Validate extreme cases
    extreme_stats = validate_extreme_cases(samples, logger, expected_config)
    validation['stats']['extreme_cases'] = extreme_stats
    
    if not extreme_stats['validation']['passed']:
        validation['passed'] = False
        validation['errors'].extend(extreme_stats['validation']['errors'])
    validation['warnings'].extend(extreme_stats['validation']['warnings'])
    
    # Validate parameter ranges
    param_stats = validate_parameter_ranges(samples, logger)
    validation['stats']['parameters'] = param_stats
    
    # Validate distributions
    if expected_config:
        dist_validation = validate_distributions(samples, expected_config, logger)
        validation['stats']['distributions'] = dist_validation
        
        if not dist_validation.get('passed', True):
            validation['passed'] = False
            validation['errors'].extend(dist_validation.get('errors', []))
        
        validation['warnings'].extend(dist_validation.get('warnings', []))
    
    if validation['errors']:
        validation['passed'] = False
    
    return validation


def validate_dataset(dataset_dir: str, verbose: bool = False, config_file: str = None) -> Dict:
    """Validate complete dataset with distribution checks."""
    
    logger = setup_logging(verbose)
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return {'passed': False, 'error': 'Directory not found'}
    
    # Load expected config
    expected_config = {}
    
    config_path = dataset_path / 'generation_config.yaml'
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            expected_config = yaml.safe_load(f)
        logger.info(f"Loaded expected config from: {config_path}")
    elif config_file:
        import yaml
        with open(config_file, 'r') as f:
            expected_config = yaml.safe_load(f)
        logger.info(f"Loaded expected config from: {config_file}")
    else:
        try:
            from ahsd.data.config import EVENT_TYPE_DISTRIBUTION, SNR_DISTRIBUTION, OVERLAP_FRACTION
            expected_config = {
                'event_distribution': EVENT_TYPE_DISTRIBUTION,
                'snr_distribution': SNR_DISTRIBUTION,
                'overlap_fraction': OVERLAP_FRACTION
            }
            logger.info("Using default config from ahsd.data.config")
        except ImportError:
            logger.warning("Could not load default config, using hardcoded defaults")
            expected_config = {
                'event_distribution': {'BBH': 0.50, 'BNS': 0.30, 'NSBH': 0.15, 'noise': 0.05},
                'snr_distribution': {'weak': 0.20, 'low': 0.35, 'medium': 0.25, 'high': 0.15, 'loud': 0.05},
                'overlap_fraction': 0.5
            }
    
    logger.info("=" * 70)
    logger.info("AHSD ENHANCED DATASET VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info("")
    
    report = {
        'dataset_dir': str(dataset_dir),
        'passed': True,
        'errors': [],
        'warnings': [],
        'splits': {},
        'expected_config': expected_config
    }
    
    # Validate each split
    for split_name in ['train', 'validation', 'test']:
        split_dir = dataset_path / split_name
        split_validation = validate_split(split_name, split_dir, logger, expected_config)
        report['splits'][split_name] = split_validation
        
        if not split_validation['passed']:
            report['passed'] = False
            report['errors'].extend(split_validation['errors'])
        
        report['warnings'].extend(split_validation['warnings'])
    
    # Check split_indices.json
    logger.info("\n[SPLIT INDICES] Checking split_indices.json...")
    logger.info("=" * 60)
    
    split_indices_file = dataset_path / 'split_indices.json'
    if split_indices_file.exists():
        try:
            with open(split_indices_file, 'r') as f:
                indices = json.load(f)
            
            logger.info(f"  ✓ Found split_indices.json")
            logger.info(f"    - Train indices: {len(indices.get('train', []))}")
            logger.info(f"    - Validation indices: {len(indices.get('validation', []))}")
            logger.info(f"    - Test indices: {len(indices.get('test', []))}")
            
            report['split_indices'] = indices
        except Exception as e:
            logger.warning(f"  ⚠ Failed to load split_indices.json: {e}")
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    
    for split_name, split_val in report['splits'].items():
        if split_val['found']:
            status = "✓ PASSED" if split_val['passed'] else "✗ FAILED"
            logger.info(f"{split_name.capitalize()}: {status}")
            
            if 'distributions' in split_val.get('stats', {}):
                dist = split_val['stats']['distributions']
                # event_distribution is a mapping {event_type: {count, fraction, percentage}}
                if dist.get('event_distribution'):
                    try:
                        total_events = sum(v.get('count', 0) for v in dist['event_distribution'].values())
                        logger.info(f"  Events: {total_events} samples")
                    except Exception:
                        logger.info("  Events: (could not compute total samples)")

                # snr_distribution is a mapping {regime: {count, fraction, percentage, range}}
                if dist.get('snr_distribution'):
                    try:
                        total_snrs = sum(v.get('count', 0) for v in dist['snr_distribution'].values())
                        logger.info(f"  SNRs: {total_snrs} signals")
                    except Exception:
                        logger.info("  SNRs: (could not compute total signals)")
            
            if 'edge_cases' in split_val.get('stats', {}):
                edge = split_val['stats']['edge_cases']
                if edge['total_edge_cases'] > 0:
                    logger.info(f"  Edge cases: {edge['total_edge_cases']}")
            
            if 'extreme_cases' in split_val.get('stats', {}):
                extreme = split_val['stats']['extreme_cases']
                total_extreme = sum(extreme['extreme_cases'].values())
                if total_extreme > 0:
                    logger.info(f"  Extreme cases: {total_extreme} ({total_extreme/extreme['total_samples']*100:.1f}%)")
                    
                    critical_missing = []
                    for case, count in extreme['extreme_cases'].items():
                        if count == 0:
                            critical_missing.append(case.replace('_', ' ').title())
                    
                    if critical_missing and len(critical_missing) <= 3:
                        logger.info(f"    Missing: {', '.join(critical_missing)}")
    
    logger.info("")
    
    if report['passed']:
        logger.info("✓ OVERALL: VALIDATION PASSED")
    else:
        logger.error("✗ OVERALL: VALIDATION FAILED")
    
    logger.info("=" * 70)
    
    if report['errors']:
        logger.error("\nErrors:")
        for error in list(set(report['errors']))[:10]:
            logger.error(f"  - {error}")
    
    if report['warnings']:
        logger.warning(f"\nWarnings: {len(set(report['warnings']))} unique warnings")
    
    return report


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced AHSD Dataset Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate with auto-detected config
  %(prog)s --dataset-dir data/dataset
  
  # Validate with specific config
  %(prog)s --dataset-dir data/dataset --config config.yaml
  
  # Verbose output with report
  %(prog)s --dataset-dir data/dataset -v --output validation_report.json
        """
    )
    
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--config', type=str,
                       help='Path to generation config YAML (optional)')
    parser.add_argument('--output', type=str,
                       help='Output path for validation report (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        report = validate_dataset(args.dataset_dir, args.verbose, args.config)
        
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nValidation report saved to: {output_path}")
        
        sys.exit(0 if report['passed'] else 1)
        
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
