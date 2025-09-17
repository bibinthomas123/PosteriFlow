import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from ..models.neural_pe import NeuralPosteriorEstimator
import bilby
import logging
from scipy.signal import find_peaks
from gwpy.timeseries import TimeSeries

class UncertaintyAwareSubtractor:
    """Subtract signals accounting for parameter uncertainties using real GW data"""
    
    def __init__(self, waveform_generator, sampling_rate: int = 4096):
        self.waveform_generator = waveform_generator
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__)
        
    def subtract_with_uncertainty(self, 
                                 data: Dict, 
                                 posterior_samples: Dict, 
                                 n_realizations: int = 100) -> Tuple[Dict, Dict]:
        """Subtract signal accounting for parameter uncertainty"""
        
        try:
            # Generate multiple waveform realizations from posterior
            waveform_realizations = self._generate_waveform_realizations(
                posterior_samples, n_realizations
            )
            
            # Compute uncertainty-weighted subtraction
            subtracted_data = {}
            subtraction_uncertainty = {}
            
            for det_name, strain in data.items():
                if det_name in waveform_realizations:
                    # Ensure strain is numpy array
                    if hasattr(strain, 'value'):
                        strain_array = strain.value
                    else:
                        strain_array = np.array(strain)
                    
                    # Get waveform realizations for this detector
                    det_waveforms = waveform_realizations[det_name]  # [n_realizations, n_samples]
                    
                    # Match lengths
                    min_len = min(len(strain_array), det_waveforms.shape[1])
                    strain_cropped = strain_array[:min_len]
                    waveforms_cropped = det_waveforms[:, :min_len]
                    
                    # Compute mean and std of waveform at each time sample
                    mean_waveform = np.mean(waveforms_cropped, axis=0)
                    std_waveform = np.std(waveforms_cropped, axis=0)
                    
                    # Subtract mean waveform
                    residual = strain_cropped - mean_waveform
                    
                    subtracted_data[det_name] = residual
                    subtraction_uncertainty[det_name] = std_waveform
                else:
                    # If waveform generation failed, return original data
                    self.logger.warning(f"No waveform generated for {det_name}, using original data")
                    subtracted_data[det_name] = strain
                    subtraction_uncertainty[det_name] = np.zeros_like(strain)
                    
        except Exception as e:
            self.logger.error(f"Subtraction failed: {e}")
            # Fallback: return original data
            subtracted_data = {det: np.array(strain) for det, strain in data.items()}
            subtraction_uncertainty = {det: np.zeros_like(strain) for det, strain in data.items()}
            
        return subtracted_data, subtraction_uncertainty
    
    def _generate_waveform_realizations(self, 
                                      posterior_samples: Dict, 
                                      n_realizations: int) -> Dict:
        """Generate multiple waveform realizations from posterior samples"""
        
        realizations = {}
        
        # Sample parameters from posterior
        sample_keys = list(posterior_samples.keys())
        if not sample_keys:
            return {}
            
        n_samples = len(posterior_samples[sample_keys[0]])
        if n_samples == 0:
            return {}
            
        sample_indices = np.random.choice(n_samples, min(n_realizations, n_samples), replace=True)
        
        # Get detector names from config or default
        detector_names = ['H1', 'L1', 'V1']
        
        for det_name in detector_names:
            det_realizations = []
            
            for idx in sample_indices:
                try:
                    # Extract parameters for this realization
                    params = {key: values[idx] if hasattr(values, '__getitem__') else values 
                             for key, values in posterior_samples.items()}
                    
                    # Generate waveform for this detector
                    waveform = self._generate_detector_waveform(params, det_name)
                    
                    if waveform is not None and len(waveform) > 0:
                        det_realizations.append(waveform)
                        
                except Exception as e:
                    self.logger.debug(f"Failed to generate waveform realization {idx} for {det_name}: {e}")
                    continue
            
            if det_realizations:
                # Ensure all realizations have same length
                min_len = min(len(w) for w in det_realizations)
                realizations[det_name] = np.array([w[:min_len] for w in det_realizations])
            
        return realizations
    
    def _generate_detector_waveform(self, params: Dict, detector_name: str) -> Optional[np.ndarray]:
        """Generate waveform for specific detector and parameters"""
        
        try:
            # Get detector object
            detector = bilby.gw.detector.get_detector(detector_name)
            
            # Ensure all required parameters are present with defaults
            default_params = {
                'mass_1': 30.0, 'mass_2': 30.0, 'luminosity_distance': 500.0,
                'theta_jn': 0.0, 'psi': 0.0, 'phase': 0.0, 'geocent_time': 0.0,
                'ra': 0.0, 'dec': 0.0, 'a_1': 0.0, 'a_2': 0.0,
                'tilt_1': 0.0, 'tilt_2': 0.0, 'phi_12': 0.0, 'phi_jl': 0.0
            }
            
            # Update with provided parameters
            full_params = default_params.copy()
            full_params.update(params)
            
            # Validate parameter ranges
            full_params['mass_1'] = max(1.0, min(100.0, float(full_params['mass_1'])))
            full_params['mass_2'] = max(1.0, min(100.0, float(full_params['mass_2'])))
            full_params['luminosity_distance'] = max(10.0, min(5000.0, float(full_params['luminosity_distance'])))
            
            # Generate polarizations
            waveform_polarizations = self.waveform_generator.frequency_domain_strain(full_params)
            
            # Project to detector
            strain = detector.project_wave(
                waveform_polarizations['plus'],
                waveform_polarizations['cross'], 
                full_params['ra'], full_params['dec'], full_params['psi']
            )
            
            return strain.time_domain_strain
            
        except Exception as e:
            self.logger.debug(f"Waveform generation failed for {detector_name}: {e}")
            return None

class AdaptiveSubtractor:
    """Main adaptive subtraction class for real GW data"""
    
    def __init__(self, neural_pe: NeuralPosteriorEstimator, 
                 uncertainty_subtractor: UncertaintyAwareSubtractor):
        self.neural_pe = neural_pe
        self.uncertainty_subtractor = uncertainty_subtractor
        self.logger = logging.getLogger(__name__)
        
    def extract_and_subtract(self, data: Dict, signal_priority: int) -> Tuple[Dict, Dict, Dict]:
        """Extract signal parameters and subtract from data"""
        
        try:
            # Quick parameter estimation
            self.logger.debug(f"Running neural PE for signal {signal_priority}")
            posterior_samples, posterior_summary = self.neural_pe.quick_estimate(data)
            
            # Uncertainty-aware subtraction
            self.logger.debug(f"Subtracting signal {signal_priority} with uncertainty")
            residual_data, subtraction_uncertainty = self.uncertainty_subtractor.subtract_with_uncertainty(
                data, posterior_samples, n_realizations=50  # Reduced for speed
            )
            
            # Compute signal quality metrics
            signal_quality = self._assess_signal_quality(posterior_summary, subtraction_uncertainty)
            
            extraction_result = {
                'signal_id': signal_priority,
                'posterior_samples': posterior_samples,
                'posterior_summary': posterior_summary,
                'subtraction_uncertainty': subtraction_uncertainty,
                'signal_quality': signal_quality,
                'extraction_method': 'neural_pe'
            }
            
            return residual_data, extraction_result, subtraction_uncertainty
            
        except Exception as e:
            self.logger.error(f"Extraction failed for signal {signal_priority}: {e}")
            
            # Fallback: return original data with minimal extraction result
            fallback_result = {
                'signal_id': signal_priority,
                'posterior_samples': {},
                'posterior_summary': self._generate_fallback_summary(),
                'subtraction_uncertainty': {det: np.zeros_like(strain) for det, strain in data.items()},
                'signal_quality': 0.0,
                'extraction_method': 'fallback'
            }
            
            return data, fallback_result, {}
    
    def _assess_signal_quality(self, posterior_summary: Dict, subtraction_uncertainty: Dict) -> float:
        """Assess quality of signal extraction"""
        
        quality_score = 0.0
        
        try:
            # SNR-based quality (if available)
            if 'network_snr' in posterior_summary:
                snr = posterior_summary['network_snr']['median']
                quality_score += min(snr / 20.0, 1.0) * 0.4
            
            # Parameter uncertainty quality
            param_uncertainties = []
            for param_name in ['mass_1', 'mass_2', 'luminosity_distance']:
                if param_name in posterior_summary:
                    median = posterior_summary[param_name]['median']
                    std = posterior_summary[param_name]['std']
                    if median > 0:
                        rel_uncertainty = std / median
                        param_uncertainties.append(rel_uncertainty)
            
            if param_uncertainties:
                avg_uncertainty = np.mean(param_uncertainties)
                quality_score += max(0, 1.0 - avg_uncertainty) * 0.4
            
            # Subtraction quality
            if subtraction_uncertainty:
                avg_sub_uncertainty = np.mean([np.mean(unc) for unc in subtraction_uncertainty.values()])
                quality_score += max(0, 1.0 - avg_sub_uncertainty * 1e20) * 0.2
            
        except Exception as e:
            self.logger.debug(f"Quality assessment failed: {e}")
            quality_score = 0.5  # Default moderate quality
            
        return min(max(quality_score, 0.0), 1.0)
    
    def _generate_fallback_summary(self) -> Dict:
        """Generate fallback posterior summary"""
        fallback_params = {
            'mass_1': {'median': 30.0, 'std': 10.0, 'quantiles': [20.0, 25.0, 35.0, 40.0]},
            'mass_2': {'median': 30.0, 'std': 10.0, 'quantiles': [20.0, 25.0, 35.0, 40.0]},
            'luminosity_distance': {'median': 500.0, 'std': 200.0, 'quantiles': [300.0, 400.0, 600.0, 700.0]},
            'network_snr': {'median': 10.0, 'std': 2.0, 'quantiles': [8.0, 9.0, 11.0, 12.0]}
        }
        return fallback_params
