"""Data format standardization for AHSD pipeline."""

import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def standardize_strain_data(data: Any) -> Dict[str, np.ndarray]:
    """Convert any data format to standard: Dict[str, np.ndarray]"""
    
    if isinstance(data, dict):
        standardized = {}
        for detector, strain_data in data.items():
            try:
                if isinstance(strain_data, dict):
                    # Format: {'strain': array, 'times': array, ...}
                    if 'strain' in strain_data:
                        strain_array = np.array(strain_data['strain'])
                    else:
                        # Try to get first array-like value
                        strain_array = np.array(list(strain_data.values())[0])
                elif hasattr(strain_data, 'value'):
                    # gwpy TimeSeries
                    strain_array = np.array(strain_data.value)
                elif hasattr(strain_data, 'data'):
                    # Other time series formats
                    strain_array = np.array(strain_data.data)
                else:
                    # Direct array
                    strain_array = np.array(strain_data)
                
                # Validate and clean
                if len(strain_array) > 0 and np.any(np.isfinite(strain_array)):
                    standardized[detector] = strain_array.astype(np.float32)
                else:
                    logger.warning(f"Invalid data for {detector}, using noise")
                    standardized[detector] = np.random.normal(0, 1e-23, 32768).astype(np.float32)
                    
            except Exception as e:
                logger.warning(f"Error processing {detector}: {e}")
                standardized[detector] = np.random.normal(0, 1e-23, 32768).astype(np.float32)
                
        return standardized
    else:
        logger.warning("Unknown data format, returning default noise")
        return {
            'H1': np.random.normal(0, 1e-23, 32768).astype(np.float32),
            'L1': np.random.normal(0, 1e-23, 32768).astype(np.float32),
            'V1': np.random.normal(0, 1e-23, 32768).astype(np.float32)
        }

def validate_data_format(data: Dict[str, np.ndarray]) -> bool:
    """Validate standardized data format."""
    if not isinstance(data, dict):
        return False
    
    for detector, strain in data.items():
        if not isinstance(strain, np.ndarray):
            return False
        if len(strain) == 0:
            return False
        if not np.all(np.isfinite(strain)):
            return False
    
    return True

def prepare_detection_features(signal_params: Dict) -> Dict:
    """Prepare detection features from signal parameters."""
    
    detection = signal_params.copy()
    
    # Ensure all required features are present
    required_features = [
        'mass_1', 'mass_2', 'luminosity_distance', 'network_snr',
        'ra', 'dec', 'a_1', 'a_2', 'theta_jn', 'psi'
    ]
    
    for feature in required_features:
        if feature not in detection:
            # Set reasonable defaults
            if 'mass' in feature:
                detection[feature] = 30.0
            elif feature == 'luminosity_distance':
                detection[feature] = 500.0
            elif feature == 'network_snr':
                detection[feature] = 12.0
            elif feature in ['ra', 'dec', 'theta_jn', 'psi']:
                detection[feature] = 0.0
            else:
                detection[feature] = 0.0
    
    # Compute derived quantities
    m1, m2 = detection['mass_1'], detection['mass_2']
    detection['chirp_mass_source'] = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    detection['total_mass_source'] = m1 + m2
    
    return detection
