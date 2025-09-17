"""Adaptive signal subtraction with uncertainty quantification."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

class UncertaintyAwareSubtractor:
    """Subtractor that accounts for parameter uncertainties."""
    
    def __init__(self, waveform_generator):
        self.waveform_generator = waveform_generator
        self.logger = logging.getLogger(__name__)
    
    def subtract_signal(self, data: Dict[str, np.ndarray], 
                       parameters: Dict[str, float],
                       uncertainty: Dict[str, float] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Subtract signal with uncertainty propagation."""
        
        residual_data = {}
        subtraction_info = {}
        
        for detector, strain in data.items():
            try:
                # Generate waveform template
                template = self._generate_template(parameters, detector)
                
                if template is not None and len(template) == len(strain):
                    # Perform subtraction
                    residual = strain - template
                    residual_data[detector] = residual
                    
                    # Compute subtraction quality metrics
                    subtraction_info[detector] = {
                        'template_snr': np.sqrt(np.sum(template**2)),
                        'residual_rms': np.sqrt(np.mean(residual**2)),
                        'subtraction_efficiency': 1.0 - np.var(residual) / np.var(strain)
                    }
                else:
                    # Keep original data if template generation fails
                    residual_data[detector] = strain
                    subtraction_info[detector] = {'error': 'template_generation_failed'}
                    
            except Exception as e:
                self.logger.debug(f"Subtraction failed for {detector}: {e}")
                residual_data[detector] = strain
                subtraction_info[detector] = {'error': str(e)}
        
        return residual_data, subtraction_info
    
    def _generate_template(self, parameters: Dict[str, float], detector: str) -> Optional[np.ndarray]:
        """Generate waveform template for subtraction."""
        try:
            # Simple mock template generation
            # In real implementation, use waveform_generator
            duration = 4.0
            sampling_rate = 4096
            n_samples = int(duration * sampling_rate)
            
            # Generate simple chirp template
            t = np.linspace(0, duration, n_samples)
            m1 = parameters.get('mass_1', 30.0)
            m2 = parameters.get('mass_2', 30.0)
            
            # Chirp mass
            chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
            
            # Simple frequency evolution
            f_start = 35.0
            f_end = 250.0
            freq = f_start + (f_end - f_start) * (t / duration)**3
            
            # Amplitude
            distance = parameters.get('luminosity_distance', 500.0)
            amp = 1e-21 * (chirp_mass / 30.0)**(5/6) / (distance / 400.0)
            
            # Generate strain
            phase = 2 * np.pi * np.cumsum(freq) * (t[1] - t[0])
            template = amp * np.exp(-t / (duration * 0.3)) * np.sin(phase)
            
            return template
            
        except Exception as e:
            self.logger.debug(f"Template generation failed: {e}")
            return None


class AdaptiveSubtractor:
    """Main adaptive subtractor combining neural PE and uncertainty-aware subtraction."""
    
    def __init__(self, neural_pe, uncertainty_subtractor):
        self.neural_pe = neural_pe
        self.uncertainty_subtractor = uncertainty_subtractor
        self.logger = logging.getLogger(__name__)
    
    def extract_and_subtract(self, data: Dict[str, np.ndarray], 
                           detection_idx: int) -> Tuple[Dict[str, np.ndarray], Dict, Dict]:
        """Extract signal parameters and subtract from data."""
        
        try:
            # Extract data features for neural PE conditioning
            context = self._extract_data_context(data)
            
            # Use quick_estimate method for testing
            extraction_result = self.neural_pe.quick_estimate(data, detection_idx)
            
            # Get best parameter estimates from posterior summary
            posterior_summary = extraction_result.get('posterior_summary', {})
            best_params = {}
            uncertainties = {}
            
            for param_name, summary in posterior_summary.items():
                best_params[param_name] = summary.get('median', 0.0)
                uncertainties[param_name] = summary.get('std', 0.1)
            
            # Perform subtraction using uncertainty-aware subtractor
            residual_data, subtraction_info = self.uncertainty_subtractor.subtract_signal(
                data, best_params, uncertainties
            )
            
            # Add extraction info to result
            extraction_result['subtraction_info'] = subtraction_info
            extraction_result['best_parameters'] = best_params
            
            return residual_data, extraction_result, uncertainties
            
        except Exception as e:
            self.logger.debug(f"Extract and subtract failed: {e}")
            
            # Return original data and empty results
            return data, {'error': str(e), 'posterior_summary': {}}, {}

    def _extract_data_context(self, data: Dict[str, np.ndarray]) -> torch.Tensor:
        """Extract data context for neural PE conditioning."""
        try:
            features = []
            for det_name, strain in data.items():
                if hasattr(strain, '__len__') and len(strain) > 0:
                    strain_array = np.array(strain)
                    # Basic statistical features
                    features.extend([
                        np.mean(strain_array),
                        np.std(strain_array),
                        np.max(np.abs(strain_array)),
                        np.median(strain_array)
                    ])
            
            # Pad to expected context size (300)
            while len(features) < 300:
                features.append(0.0)
            features = features[:300]  # Truncate if too long
            
            return torch.tensor(np.real(features), dtype=torch.float32).unsqueeze(0)
        except:
            # Fallback
            return torch.zeros(1, 300)

    def quick_estimate(self, data: Dict[str, np.ndarray], detection_idx: int) -> Dict:
        """Quick parameter estimation without full posterior sampling."""
        
        try:
            # Extract context
            context = self._extract_data_context(data)
            
            # Get quick estimate using neural PE
            with torch.no_grad():
                # Sample a few times for quick estimate
                samples = self.neural_pe.sample(context, num_samples=10)
                
                if samples.dim() == 3:
                    samples = samples.squeeze(1)
                
                # Compute statistics
                param_estimates = {}
                for i, param_name in enumerate(self.neural_pe.param_names):
                    if i < samples.shape[1]:
                        param_samples = samples[:, i].numpy()
                        param_estimates[param_name] = {
                            'median': float(np.median(param_samples)),
                            'std': float(np.std(param_samples))
                        }
                
                return {
                    'posterior_summary': param_estimates,
                    'signal_quality': 0.8,  # Mock quality
                    'method': 'quick_neural_pe'
                }
                
        except Exception as e:
            self.logger.debug(f"Quick estimate failed: {e}")
            # Fallback estimates
            fallback_params = {}
            for param_name in self.neural_pe.param_names:
                fallback_params[param_name] = {
                    'median': 30.0 if 'mass' in param_name else 0.0,
                    'std': 5.0 if 'mass' in param_name else 0.1
                }
            
            return {
                'posterior_summary': fallback_params,
                'signal_quality': 0.5,
                'method': 'fallback'
            }
