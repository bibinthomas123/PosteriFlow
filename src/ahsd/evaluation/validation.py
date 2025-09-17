"""
Result validation and consistency checks.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats

class ResultValidator:
    """Validate AHSD analysis results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_results(self, result: Dict) -> Dict:
        """
        Comprehensive validation of analysis results.
        
        Parameters:
        -----------
        result : dict
            AHSD analysis result
            
        Returns:
        --------
        dict
            Validation report
        """
        
        validation_report = {
            'passed': True,
            'warnings': [],
            'errors': [],
            'checks': {}
        }
        
        try:
            # Check basic structure
            structure_check = self._check_result_structure(result)
            validation_report['checks']['structure'] = structure_check
            if not structure_check['passed']:
                validation_report['passed'] = False
                validation_report['errors'].extend(structure_check['errors'])
            
            # Check extracted signals
            if 'extracted_signals' in result:
                signals_check = self._check_extracted_signals(result['extracted_signals'])
                validation_report['checks']['signals'] = signals_check
                if not signals_check['passed']:
                    validation_report['warnings'].extend(signals_check['warnings'])
            
            # Check parameter ranges
            params_check = self._check_parameter_ranges(result)
            validation_report['checks']['parameters'] = params_check
            if not params_check['passed']:
                validation_report['warnings'].extend(params_check['warnings'])
            
            # Check performance metrics
            performance_check = self._check_performance_metrics(result)
            validation_report['checks']['performance'] = performance_check
            
            # Check consistency
            consistency_check = self._check_consistency(result)
            validation_report['checks']['consistency'] = consistency_check
            if not consistency_check['passed']:
                validation_report['warnings'].extend(consistency_check['warnings'])
                
        except Exception as e:
            validation_report['passed'] = False
            validation_report['errors'].append(f"Validation failed: {e}")
            self.logger.error(f"Result validation failed: {e}")
        
        return validation_report
    
    def _check_result_structure(self, result: Dict) -> Dict:
        """Check basic result structure."""
        
        required_keys = ['extracted_signals', 'performance_metrics']
        missing_keys = []
        
        for key in required_keys:
            if key not in result:
                missing_keys.append(key)
        
        return {
            'passed': len(missing_keys) == 0,
            'errors': [f"Missing required key: {key}" for key in missing_keys],
            'required_keys': required_keys,
            'missing_keys': missing_keys
        }
    
    def _check_extracted_signals(self, extracted_signals: List[Dict]) -> Dict:
        """Check extracted signals validity."""
        
        warnings = []
        
        if len(extracted_signals) == 0:
            warnings.append("No signals extracted")
        
        for i, signal in enumerate(extracted_signals):
            # Check posterior summary
            if 'posterior_summary' not in signal:
                warnings.append(f"Signal {i}: Missing posterior_summary")
                continue
            
            posterior = signal['posterior_summary']
            
            # Check required parameters
            required_params = ['mass_1', 'mass_2', 'luminosity_distance']
            for param in required_params:
                if param not in posterior:
                    warnings.append(f"Signal {i}: Missing parameter {param}")
                else:
                    param_data = posterior[param]
                    if not isinstance(param_data, dict):
                        warnings.append(f"Signal {i}: Invalid parameter data for {param}")
                    elif 'median' not in param_data:
                        warnings.append(f"Signal {i}: Missing median for {param}")
        
        return {
            'passed': len(warnings) == 0,
            'warnings': warnings
        }
    
    def _check_parameter_ranges(self, result: Dict) -> Dict:
        """Check if parameters are in physically reasonable ranges."""
        
        warnings = []
        
        parameter_ranges = {
            'mass_1': (1.0, 200.0),  # Solar masses
            'mass_2': (1.0, 200.0),
            'luminosity_distance': (10.0, 10000.0),  # Mpc
            'a_1': (0.0, 0.99),  # Spin magnitudes
            'a_2': (0.0, 0.99),
            'network_snr': (3.0, 1000.0)  # SNR
        }
        
        extracted_signals = result.get('extracted_signals', [])
        
        for i, signal in enumerate(extracted_signals):
            posterior = signal.get('posterior_summary', {})
            
            for param, (min_val, max_val) in parameter_ranges.items():
                if param in posterior:
                    param_data = posterior[param]
                    if isinstance(param_data, dict) and 'median' in param_data:
                        value = param_data['median']
                        if not (min_val <= value <= max_val):
                            warnings.append(
                                f"Signal {i}: {param}={value:.3f} outside reasonable range "
                                f"[{min_val}, {max_val}]"
                            )
        
        return {
            'passed': len(warnings) == 0,
            'warnings': warnings
        }
    
    def _check_performance_metrics(self, result: Dict) -> Dict:
        """Check performance metrics validity."""
        
        warnings = []
        metrics = result.get('performance_metrics', {})
        
        # Check processing time
        if 'total_extraction_time' in metrics:
            time_taken = metrics['total_extraction_time']
            if time_taken < 0:
                warnings.append("Negative processing time")
            elif time_taken > 3600:  # 1 hour
                warnings.append(f"Very long processing time: {time_taken:.1f}s")
        
        # Check signal count consistency
        n_extracted = metrics.get('n_extracted_signals', 0)
        actual_signals = len(result.get('extracted_signals', []))
        
        if n_extracted != actual_signals:
            warnings.append(
                f"Inconsistent signal count: metrics={n_extracted}, "
                f"actual={actual_signals}"
            )
        
        return {
            'passed': len(warnings) == 0,
            'warnings': warnings
        }
    
    def _check_consistency(self, result: Dict) -> Dict:
        """Check internal consistency of results."""
        
        warnings = []
        
        extracted_signals = result.get('extracted_signals', [])
        
        # Check mass ordering (m1 >= m2)
        for i, signal in enumerate(extracted_signals):
            posterior = signal.get('posterior_summary', {})
            
            if 'mass_1' in posterior and 'mass_2' in posterior:
                m1 = posterior['mass_1'].get('median', 0)
                m2 = posterior['mass_2'].get('median', 0)
                
                if m1 < m2:
                    warnings.append(f"Signal {i}: mass_1 < mass_2 (should be m1 >= m2)")
        
        # Check spin magnitudes
        for i, signal in enumerate(extracted_signals):
            posterior = signal.get('posterior_summary', {})
            
            for spin_param in ['a_1', 'a_2']:
                if spin_param in posterior:
                    spin_value = posterior[spin_param].get('median', 0)
                    if spin_value < 0 or spin_value >= 1.0:
                        warnings.append(
                            f"Signal {i}: {spin_param}={spin_value:.3f} outside [0, 1)"
                        )
        
        return {
            'passed': len(warnings) == 0,
            'warnings': warnings
        }
    
    def validate_training_data(self, training_scenarios: List[Dict]) -> Dict:
        """Validate training data quality."""
        
        validation_report = {
            'passed': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        if not training_scenarios:
            validation_report['passed'] = False
            validation_report['errors'].append("No training scenarios provided")
            return validation_report
        
        # Check scenario structure
        required_keys = ['true_parameters', 'injected_data']
        missing_scenarios = []
        
        for i, scenario in enumerate(training_scenarios):
            for key in required_keys:
                if key not in scenario:
                    missing_scenarios.append(f"Scenario {i}: missing {key}")
        
        if missing_scenarios:
            validation_report['warnings'].extend(missing_scenarios)
        
        # Collect statistics
        n_signals_per_scenario = []
        snr_values = []
        
        for scenario in training_scenarios:
            true_params = scenario.get('true_parameters', [])
            n_signals_per_scenario.append(len(true_params))
            
            # Extract SNR values if available
            target_snrs = scenario.get('target_snrs', [])
            snr_values.extend(target_snrs)
        
        validation_report['statistics'] = {
            'n_scenarios': len(training_scenarios),
            'mean_signals_per_scenario': np.mean(n_signals_per_scenario) if n_signals_per_scenario else 0,
            'mean_snr': np.mean(snr_values) if snr_values else 0,
            'min_snr': np.min(snr_values) if snr_values else 0,
            'max_snr': np.max(snr_values) if snr_values else 0
        }
        
        return validation_report
    
    def check_data_quality(self, data: Dict) -> Dict:
        """Check strain data quality."""
        
        quality_report = {
            'passed': True,
            'warnings': [],
            'statistics': {}
        }
        
        for det_name, strain in data.items():
            if hasattr(strain, '__len__') and len(strain) > 0:
                strain_array = np.array(strain)
                
                # Check for NaNs or Infs
                if not np.all(np.isfinite(strain_array)):
                    quality_report['passed'] = False
                    quality_report['warnings'].append(f"{det_name}: Contains NaN/Inf values")
                
                # Check dynamic range
                strain_std = np.std(strain_array)
                strain_max = np.max(np.abs(strain_array))
                
                if strain_std == 0:
                    quality_report['warnings'].append(f"{det_name}: Zero variance (constant data)")
                
                if strain_max > 1e-18:  # Very large strain
                    quality_report['warnings'].append(f"{det_name}: Unusually large strain amplitude")
                
                # Statistics
                quality_report['statistics'][det_name] = {
                    'length': len(strain_array),
                    'mean': float(np.mean(strain_array)),
                    'std': float(strain_std),
                    'max_abs': float(strain_max),
                    'finite_fraction': float(np.mean(np.isfinite(strain_array)))
                }
        
        return quality_report
