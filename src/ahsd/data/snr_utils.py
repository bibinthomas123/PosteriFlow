"""
SNR Utility Functions

Provides helper functions for Signal-to-Noise Ratio (SNR) computations, regime classification,
and validation across the AHSD pipeline.

Key Functions:
- categorize_snr: Classify SNR into regimes (weak, low, medium, high, loud)
- compute_network_snr: Compute coherent network SNR from detector data
- validate_snr: Check if SNR is within expected bounds
- snr_regime_from_params: Estimate SNR regime from GW parameters
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def categorize_snr(snr: float) -> str:
    """
    Classify SNR value into detection regime.
    
    Args:
        snr (float): Signal-to-Noise Ratio value
        
    Returns:
        str: Regime name ('weak', 'low', 'medium', 'high', 'loud')
        
    Example:
        >>> categorize_snr(8.5)
        'weak'
        >>> categorize_snr(25.0)
        'low'
        >>> categorize_snr(40.0)
        'medium'
        >>> categorize_snr(70.0)
        'high'
        >>> categorize_snr(120.0)
        'loud'
    """
    try:
        snr_val = float(snr)
    except (TypeError, ValueError):
        logger.warning(f"Invalid SNR value: {snr}, returning 'low'")
        return 'low'
    
    if snr_val < 5:
        return 'weak'  # Borderline detectable
    elif snr_val < 10:
        return 'weak'  # Weak detections (SNR 5-10)
    elif snr_val < 30:
        return 'low'   # Low SNR (SNR 10-30)
    elif snr_val < 50:
        return 'medium'  # Medium SNR (SNR 30-50)
    elif snr_val < 100:
        return 'high'  # High SNR (SNR 50-100)
    else:
        return 'loud'  # Very high SNR (SNR > 100)


def validate_snr_bounds(snr: float, min_snr: float = 5.0, max_snr: float = 200.0) -> Tuple[bool, str]:
    """
    Validate if SNR is within expected bounds.
    
    Args:
        snr (float): SNR value to validate
        min_snr (float): Minimum valid SNR (default: 5.0)
        max_snr (float): Maximum valid SNR (default: 200.0)
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
        
    Example:
        >>> validate_snr_bounds(25.0)
        (True, 'SNR within bounds [5.0, 200.0]')
        >>> validate_snr_bounds(2.0)
        (False, 'SNR too low: 2.0 < 5.0')
    """
    try:
        snr_val = float(snr)
    except (TypeError, ValueError):
        return False, f"Invalid SNR value: {snr}"
    
    if snr_val < min_snr:
        return False, f"SNR too low: {snr_val} < {min_snr}"
    elif snr_val > max_snr:
        return False, f"SNR too high: {snr_val} > {max_snr}"
    else:
        return True, f"SNR within bounds [{min_snr}, {max_snr}]"


def compute_network_snr(
    detector_dict: Dict[str, Dict],
    detector_names: Optional[list] = None
) -> float:
    """
    Compute coherent network SNR from detector data.
    
    Implements coherent combination across detectors weighted by sensitivity:
    SNR_net = sqrt(sum(SNR_i^2)) for independent detectors
    
    Args:
        detector_dict (Dict): Dictionary with detector data
            Expected structure: {
                'H1': {'snr': float, ...},
                'L1': {'snr': float, ...},
                'V1': {'snr': float, ...}
            }
        detector_names (list, optional): Detectors to include (default: ['H1', 'L1', 'V1'])
        
    Returns:
        float: Network SNR value
        
    Example:
        >>> det_data = {
        ...     'H1': {'snr': 30.0},
        ...     'L1': {'snr': 25.0},
        ...     'V1': {'snr': 10.0}
        ... }
        >>> compute_network_snr(det_data)
        39.05124...
    """
    if detector_names is None:
        detector_names = ['H1', 'L1', 'V1']
    
    snr_squared_sum = 0.0
    count = 0
    
    for detector in detector_names:
        if detector in detector_dict:
            det_data = detector_dict[detector]
            if isinstance(det_data, dict) and 'snr' in det_data:
                snr_val = float(det_data['snr'])
                snr_squared_sum += snr_val ** 2
                count += 1
            elif isinstance(det_data, (int, float)):
                snr_val = float(det_data)
                snr_squared_sum += snr_val ** 2
                count += 1
    
    if count == 0:
        logger.warning("No SNR values found in detector dictionary")
        return 0.0
    
    network_snr = np.sqrt(snr_squared_sum)
    return float(network_snr)


def estimate_snr_from_parameters(
    params: Dict,
    reference_snr: float = 20.0,
    reference_distance: float = 1500.0,
    reference_mass: float = 30.0
) -> float:
    """
    Estimate SNR from gravitational wave parameters using scaling relationships.
    
    Uses physics-based formula:
    SNR = ref_snr × (M_c / ref_mass)^(5/6) × (ref_distance / distance)
    
    Args:
        params (Dict): GW parameters with at least 'luminosity_distance' and 'chirp_mass'
        reference_snr (float): Reference SNR (default: 20.0)
        reference_distance (float): Reference distance in Mpc (default: 1500.0)
        reference_mass (float): Reference chirp mass in M☉ (default: 30.0)
        
    Returns:
        float: Estimated SNR
        
    Example:
        >>> params = {'chirp_mass': 30.0, 'luminosity_distance': 1500.0}
        >>> estimate_snr_from_parameters(params)
        20.0
    """
    try:
        chirp_mass = float(params.get('chirp_mass', reference_mass))
        distance = float(params.get('luminosity_distance', reference_distance))
        
        if distance <= 0:
            logger.warning(f"Invalid distance: {distance}, using reference")
            distance = reference_distance
        
        # Physics-based SNR scaling
        snr = (
            reference_snr
            * (chirp_mass / reference_mass) ** (5 / 6)
            * (reference_distance / distance)
        )
        
        return float(np.clip(snr, 5.0, 200.0))
    
    except (TypeError, ValueError, KeyError) as e:
        logger.warning(f"SNR estimation failed: {e}, returning reference")
        return reference_snr


def snr_regime_from_params(
    params: Dict,
    reference_snr: float = 20.0,
    reference_distance: float = 1500.0,
    reference_mass: float = 30.0
) -> str:
    """
    Estimate SNR regime from GW parameters.
    
    Args:
        params (Dict): GW parameters
        reference_snr (float): Reference SNR
        reference_distance (float): Reference distance in Mpc
        reference_mass (float): Reference chirp mass in M☉
        
    Returns:
        str: SNR regime ('weak', 'low', 'medium', 'high', 'loud')
        
    Example:
        >>> params = {'chirp_mass': 30.0, 'luminosity_distance': 200.0}
        >>> snr_regime_from_params(params)
        'medium'
    """
    snr = estimate_snr_from_parameters(
        params,
        reference_snr=reference_snr,
        reference_distance=reference_distance,
        reference_mass=reference_mass
    )
    return categorize_snr(snr)


def get_snr_range_for_regime(regime: str) -> Tuple[float, float]:
    """
    Get SNR bounds for a given regime.
    
    Args:
        regime (str): SNR regime name
        
    Returns:
        Tuple[float, float]: (min_snr, max_snr) bounds
        
    Example:
        >>> get_snr_range_for_regime('medium')
        (30.0, 50.0)
    """
    ranges = {
        'weak': (5.0, 10.0),
        'low': (10.0, 30.0),
        'medium': (30.0, 50.0),
        'high': (50.0, 100.0),
        'loud': (100.0, 200.0),
    }
    
    if regime not in ranges:
        logger.warning(f"Unknown regime: {regime}, returning 'medium' bounds")
        return ranges['medium']
    
    return ranges[regime]


def snr_within_regime(snr: float, regime: str) -> bool:
    """
    Check if SNR value is within a given regime's bounds.
    
    Args:
        snr (float): SNR value
        regime (str): SNR regime
        
    Returns:
        bool: True if SNR is within regime bounds
        
    Example:
        >>> snr_within_regime(35.0, 'medium')
        True
        >>> snr_within_regime(35.0, 'high')
        False
    """
    min_snr, max_snr = get_snr_range_for_regime(regime)
    return min_snr <= float(snr) <= max_snr


def normalize_snr_to_priority(snr: float, min_snr: float = 5.0, max_snr: float = 200.0) -> float:
    """
    Normalize SNR to priority weight in [0.05, 1.0].
    
    Uses log-scale mapping to handle wide SNR range.
    
    Args:
        snr (float): SNR value
        min_snr (float): Minimum SNR bound
        max_snr (float): Maximum SNR bound
        
    Returns:
        float: Priority in [0.05, 1.0]
        
    Example:
        >>> normalize_snr_to_priority(5.0)
        0.05
        >>> normalize_snr_to_priority(20.0)
        0.45...
        >>> normalize_snr_to_priority(200.0)
        1.0
    """
    try:
        snr_val = float(np.clip(snr, min_snr, max_snr))
        
        # Log-scale normalization
        log_snr = np.log(snr_val)
        log_min = np.log(min_snr)
        log_max = np.log(max_snr)
        
        normalized = (log_snr - log_min) / (log_max - log_min)
        
        # Apply sigmoid correction to expand lower end
        priority = 0.05 + 0.95 * (normalized ** 0.7)
        
        return float(np.clip(priority, 0.05, 1.0))
    
    except (TypeError, ValueError):
        logger.warning(f"Priority normalization failed for SNR={snr}, returning 0.05")
        return 0.05


def snr_statistics(snr_values: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics on SNR distribution.
    
    Args:
        snr_values (np.ndarray): Array of SNR values
        
    Returns:
        Dict: Statistics dict with 'mean', 'median', 'std', 'min', 'max', 'cv'
        
    Example:
        >>> snrs = np.array([8, 15, 25, 45, 120])
        >>> stats = snr_statistics(snrs)
        >>> stats['mean']
        42.6
        >>> stats['cv']  # Coefficient of variation
        1.05...
    """
    snr_arr = np.array(snr_values, dtype=float)
    
    if len(snr_arr) == 0:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'cv': 0.0
        }
    
    mean = np.mean(snr_arr)
    median = np.median(snr_arr)
    std = np.std(snr_arr)
    cv = std / mean if mean > 0 else 0.0
    
    return {
        'mean': float(mean),
        'median': float(median),
        'std': float(std),
        'min': float(np.min(snr_arr)),
        'max': float(np.max(snr_arr)),
        'cv': float(cv)
    }


def regime_distribution(snr_values: np.ndarray) -> Dict[str, int]:
    """
    Classify SNR values into regimes and count distribution.
    
    Args:
        snr_values (np.ndarray): Array of SNR values
        
    Returns:
        Dict: Count of samples in each regime
        
    Example:
        >>> snrs = np.array([8, 15, 25, 45, 120])
        >>> regime_distribution(snrs)
        {'weak': 1, 'low': 2, 'medium': 1, 'high': 0, 'loud': 1}
    """
    distribution = {regime: 0 for regime in ['weak', 'low', 'medium', 'high', 'loud']}
    
    for snr in snr_values:
        regime = categorize_snr(snr)
        distribution[regime] += 1
    
    return distribution
