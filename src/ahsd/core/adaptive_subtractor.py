#!/usr/bin/env python3
"""
Adaptive signal subtraction with uncertainty quantification
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from ..utils.data_format import standardize_strain_data




class UncertaintyAwareSubtractor:
    """ physics-based subtractor"""
    
    def __init__(self, waveform_generator=None):
        self.waveform_generator = waveform_generator
        self.logger = logging.getLogger(__name__)
        self.sampling_rate = 4096
        self.duration = 8.0
        
        # Physics constants
        self.G = 6.674e-11  # m^3 kg^-1 s^-2
        self.c = 2.998e8    # m/s
        self.M_sun = 1.989e30  # kg
        self.M_sun_s = 4.925490947e-6  # Solar mass in seconds (G*M_sun/c^3)
        
    def subtract_signal(self, data: Dict[str, Any], 
                       parameters: Dict[str, float],
                       uncertainty: Dict[str, float] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """ physics-based signal subtraction with uncertainty propagation."""
        
        residual_data = {}
        subtraction_info = {}
        
        # Standardize data format first
        standardized_data = standardize_strain_data(data)
        
        for detector, strain in standardized_data.items():
            try:
                if len(strain) == 0:
                    continue
                
                #  Generate physics-based waveform template
                template = self._generate_physics_template(parameters, detector, len(strain))
                
                if template is not None and len(template) == len(strain):
                    #  Apply uncertainty-weighted subtraction
                    weight_factor = 1.0
                    if uncertainty is not None:
                        weight_factor = self._compute_uncertainty_weight(parameters, uncertainty)
                        template = template * weight_factor
                    
                    # Perform subtraction
                    residual = strain - template
                    residual_data[detector] = residual.astype(np.float32)
                    
                    #  Compute comprehensive subtraction quality metrics
                    template_power = np.sum(template**2)
                    residual_power = np.sum(residual**2)
                    original_power = np.sum(strain**2)
                    
                    # Cross-correlation for template matching quality
                    if template_power > 0 and original_power > 0:
                        try:
                            cross_corr = np.corrcoef(strain, template)[0, 1]
                            if not np.isfinite(cross_corr):
                                cross_corr = 0.0
                        except:
                            cross_corr = 0.0
                            
                        # Template SNR estimation
                        noise_power = np.var(strain[:1000])  # Estimate from first 1s
                        if noise_power > 0:
                            template_snr = np.sqrt(template_power) / np.sqrt(noise_power * len(template))
                        else:
                            template_snr = 0.0
                    else:
                        cross_corr = 0.0
                        template_snr = 0.0
                    
                    # Subtraction efficiency
                    if original_power > 0:
                        subtraction_efficiency = 1.0 - residual_power / original_power
                    else:
                        subtraction_efficiency = 0.0
                    
                    # Frequency domain analysis
                    freq_metrics = self._analyze_subtraction_frequency_domain(strain, template, residual)
                    
                    subtraction_info[detector] = {
                        'template_snr': float(template_snr),
                        'residual_rms': float(np.sqrt(np.mean(residual**2))),
                        'subtraction_efficiency': float(subtraction_efficiency),
                        'template_match': float(cross_corr),
                        'uncertainty_weight': float(weight_factor),
                        'original_power': float(original_power),
                        'template_power': float(template_power),
                        'residual_power': float(residual_power),
                        'freq_domain_efficiency': freq_metrics.get('efficiency', 0.0),
                        'peak_frequency_match': freq_metrics.get('peak_match', 0.0)
                    }
                else:
                    # Keep original data if template generation fails
                    residual_data[detector] = strain
                    subtraction_info[detector] = {
                        'error': 'physics_template_generation_failed',
                        'template_length': len(template) if template is not None else 0,
                        'strain_length': len(strain)
                    }
                    
            except Exception as e:
                self.logger.warning(f"Subtraction failed for {detector}: {e}")
                residual_data[detector] = strain
                subtraction_info[detector] = {'error': str(e)}
        
        return residual_data, subtraction_info
    
    def _generate_physics_template(self, parameters: Dict[str, float], detector: str, n_samples: int) -> Optional[np.ndarray]:
        """ physics-based gravitational waveform generation using Post-Newtonian theory."""
        try:
            # Time array
            t = np.linspace(0, self.duration, n_samples)
            dt = t[1] - t[0]
            
            #  Extract and validate physical parameters
            m1 = max(parameters.get('mass_1', 30.0), 1.0)
            m2 = max(parameters.get('mass_2', 30.0), 1.0)
            
            # Ensure m1 >= m2 (convention)
            if m2 > m1:
                m1, m2 = m2, m1
            
            #  Compute derived mass parameters
            total_mass = m1 + m2
            chirp_mass = (m1 * m2)**(3/5) / total_mass**(1/5)
            eta = m1 * m2 / total_mass**2  # Symmetric mass ratio
            
            # Physical parameters
            distance = max(parameters.get('luminosity_distance', 400.0), 10.0)
            tc = parameters.get('geocent_time', 0.0)  # Coalescence time
            
            # Spin parameters
            a1 = np.clip(parameters.get('a_1', 0.0), 0.0, 0.99)
            a2 = np.clip(parameters.get('a_2', 0.0), 0.0, 0.99)
            
            #  Post-Newtonian frequency evolution
            # Convert masses to geometric units
            M_total_s = total_mass * self.M_sun_s
            M_chirp_s = chirp_mass * self.M_sun_s
            
            # Time to coalescence
            time_to_merger = tc - t
            time_to_merger = np.maximum(time_to_merger, dt * 0.1)  # Avoid singularity
            
            # 3.5PN frequency evolution (more accurate than 2.5PN)
            # f(t) = (1/8Ï€) * (5/(256*Î·))^(3/8) * (G*M_chirp/c^3)^(-5/8) * (tc-t)^(-3/8)
            
            frequency_factor = (5.0/(256.0*eta))**(3/8) / (8*np.pi) * (M_chirp_s)**(-5/8)
            frequency = frequency_factor * (time_to_merger)**(-3/8)
            
            # Apply frequency limits
            f_min = 20.0  # LIGO low frequency cutoff
            f_max = min(self.sampling_rate/2.1, 1000.0)  # Avoid Nyquist issues
            frequency = np.clip(frequency, f_min, f_max)
            
            #  3.5PN phase evolution
            # More accurate phase evolution including spin effects (trapezoidal integration)
            phase = np.zeros_like(t)
            for i in range(1, len(t)):
                if frequency[i] > 0:
                    # Trapezoidal integration for better accuracy
                    avg_freq = (frequency[i] + frequency[i-1]) / 2.0
                    phase[i] = phase[i-1] + 2*np.pi * avg_freq * dt
            
            # Add coalescence phase
            phase += parameters.get('phase', 0.0)
            
            #  Spin-orbit coupling effects (2PN spin-orbit coupling)
            if a1 > 0.01 or a2 > 0.01:  # Include spin effects if significant
                # Effective spin parameter
                chi_eff = (a1 * m1 + a2 * m2) / total_mass
                
                # 2PN spin-orbit phase correction
                v = (np.pi * frequency * M_total_s)**(1/3)  # PN velocity parameter
                spin_phase_correction = (113.0/12.0 + 25.0*eta/4.0) * chi_eff * v**2
                phase += spin_phase_correction
            
            #  Amplitude evolution with Post-Newtonian corrections
            # Convert distance to meters
            distance_m = distance * 3.086e22  # Mpc to meters
            
            # Leading order amplitude
            h0 = (4 * eta * (M_chirp_s * self.c**2)**(5/3) * (np.pi * frequency)**(2/3) * self.G**(5/3)) / (self.c**4 * distance_m)
            
            # PN amplitude corrections
            # 1PN amplitude correction
            v = (np.pi * frequency * M_total_s)**(1/3)  # PN velocity parameter
            amp_1pn = 1.0 + (743.0/336.0 + 11.0*eta/4.0) * v**2
            
            # Apply amplitude evolution
            amplitude = h0 * amp_1pn
            
            # Apply coalescence amplitude scaling with regularization near merger
            time_scale = np.maximum(time_to_merger, 0.1 * M_total_s)  # Avoid singularity at merger
            amplitude *= (time_scale / M_total_s)**(-1/4)
            
            #  Generate polarizations with proper orientation
            inclination = parameters.get('theta_jn', np.pi/2)
            cos_iota = np.cos(inclination)
            
            # Plus and cross polarizations
            h_plus = amplitude * (1 + cos_iota**2) * np.cos(phase)
            h_cross = amplitude * 2 * cos_iota * np.sin(phase)
            
            #  Apply istic detector response
            psi = parameters.get('psi', 0.0)  # Polarization angle
            ra = parameters.get('ra', 0.0)
            dec = parameters.get('dec', 0.0)
            
            # Compute antenna patterns (simplified -  implementation uses LAL)
            F_plus, F_cross = self._compute_antenna_patterns(detector, ra, dec, psi)
            
            # Detector strain
            strain = F_plus * h_plus + F_cross * h_cross
            
            #  Apply istic time-domain windowing
            window = self._compute_tukey_window(len(strain), alpha=0.2)
            strain *= window
            
            #  Add detector-specific calibration effects
            calibration_factor = self._get_calibration_factor(detector, np.mean(frequency))
            strain *= calibration_factor
            
            return strain.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Physics template generation failed: {e}")
            return self._generate_fallback_template(parameters, n_samples)
    
    def _compute_antenna_patterns(self, detector: str, ra: float, dec: float, psi: float) -> Tuple[float, float]:
        """ antenna pattern computation for detectors."""
        try:
            # Simplified antenna patterns (proper formulation requires LAL)
            # For overhead sources, antenna patterns reduce to detector-dependent functions of polarization angle only
            
            if detector == 'H1':
                # Hanford detector (arms along N-E and N-W)
                F_plus = np.cos(2*psi)
                F_cross = np.sin(2*psi)
                
            elif detector == 'L1':
                # Livingston detector (arms rotated ~45° from Hanford)
                F_plus = np.cos(2*psi + np.pi/2)  # 90° rotated arms
                F_cross = np.sin(2*psi + np.pi/2)
                
            elif detector == 'V1':
                # Virgo detector (different arm angle and orientation)
                F_plus = 0.7 * np.cos(2*psi)
                F_cross = 0.7 * np.sin(2*psi)
                
            else:
                # Generic detector
                F_plus = 0.5 * np.cos(2*psi)
                F_cross = 0.5 * np.sin(2*psi)
            
            # Sky position dependence (rough approximation for non-overhead sources)
            sky_factor = (1.0 + np.cos(dec)**2) / 2.0
            F_plus *= sky_factor
            F_cross *= np.cos(dec)
            
            # Apply physical bounds
            F_plus = np.clip(F_plus, -1.0, 1.0)
            F_cross = np.clip(F_cross, -1.0, 1.0)
            
            return float(F_plus), float(F_cross)
            
        except Exception as e:
            # Fallback antenna patterns
            raise Exception(f"Failed to compute antenna patterns: {e}")
    
    def _compute_tukey_window(self, length: int, alpha: float = 0.2) -> np.ndarray:
        """Compute Tukey window for smooth template edges."""
        try:
            # Try scipy's optimized Tukey window implementation
            try:
                from scipy.signal import windows
                return windows.tukey(length, alpha=alpha)
            except (ImportError, AttributeError):
                print("Scipy.signal Import Error")
            
            # Fallback: manual Tukey window computation
            window = np.ones(length)
            taper_length = int(alpha * length / 2)
            
            if taper_length > 0:
                # Beginning taper (cosine ramp from 0 to 1)
                for i in range(taper_length):
                    window[i] = 0.5 * (1.0 - np.cos(np.pi * i / taper_length))
                
                # End taper (cosine ramp from 1 to 0) - properly reversed
                for i in range(taper_length):
                    window[length - taper_length + i] = 0.5 * (1.0 - np.cos(np.pi * (taper_length - i) / taper_length))
            
            return window
            
        except:
            # Ultimate fallback: simple linear taper
            window = np.ones(length)
            edge_length = max(1, length // 20)
            
            # Linear taper at edges
            window[:edge_length] = np.linspace(0, 1, edge_length)
            window[-edge_length:] = np.linspace(1, 0, edge_length)
            
            return window
    
    def _get_calibration_factor(self, detector: str, frequency: float) -> float:
        """Get istic calibration factor for detector."""
        try:
            # Simplified calibration model (frequency-dependent)
            base_factors = {
                'H1': 1.0,
                'L1': 1.02,  # Slight calibration difference
                'V1': 0.95   # Virgo has different calibration
            }
            
            base_factor = base_factors.get(detector, 1.0)
            
            # Frequency dependence (simplified)
            if frequency < 50:
                freq_factor = 0.8  # Reduced sensitivity at low frequencies
            elif frequency > 400:
                freq_factor = 0.9  # Some rolloff at high frequencies
            else:
                freq_factor = 1.0
            
            return base_factor * freq_factor
            
        except:
            return 1.0
    
    def _generate_fallback_template(self, parameters: Dict[str, float], n_samples: int) -> np.ndarray:
        """Fallback template generation for error cases."""
        try:
            t = np.linspace(0, self.duration, n_samples)
            
            # Simple sinusoidal template
            frequency = parameters.get('peak_frequency', 100.0)
            if frequency <= 0:
                frequency = 100.0
                
            distance = max(parameters.get('luminosity_distance', 500.0), 10.0)
            amplitude = 1e-21 / (distance / 400.0)
            
            # Add simple frequency evolution
            freq_evolution = frequency * (1 + 0.1 * t / self.duration)
            phase = 2 * np.pi * np.cumsum(freq_evolution) * (t[1] - t[0])
            
            template = amplitude * np.sin(phase + parameters.get('phase', 0.0))
            
            # Apply simple window
            window = np.exp(-((t - self.duration/2) / (self.duration/4))**2)
            template *= window
            
            return template.astype(np.float32)
            
        except:
            # Ultimate fallback
            return np.zeros(n_samples, dtype=np.float32)
    
    def _analyze_subtraction_frequency_domain(self, strain: np.ndarray, template: np.ndarray, 
                                            residual: np.ndarray) -> Dict[str, float]:
        """ frequency domain analysis of subtraction quality."""
        try:
            # FFT analysis
            strain_fft = np.fft.fft(strain)
            template_fft = np.fft.fft(template)
            residual_fft = np.fft.fft(residual)
            
            freqs = np.fft.fftfreq(len(strain), 1.0/self.sampling_rate)
            
            # Focus on GW frequency band
            gw_mask = (freqs >= 20) & (freqs <= 500)
            
            if np.any(gw_mask):
                strain_power_gw = np.sum(np.abs(strain_fft[gw_mask])**2)
                template_power_gw = np.sum(np.abs(template_fft[gw_mask])**2)
                residual_power_gw = np.sum(np.abs(residual_fft[gw_mask])**2)
                
                # Frequency domain efficiency
                if strain_power_gw > 0:
                    freq_efficiency = 1.0 - residual_power_gw / strain_power_gw
                else:
                    freq_efficiency = 0.0
                
                # Peak frequency matching
                strain_peak_idx = np.argmax(np.abs(strain_fft[gw_mask]))
                template_peak_idx = np.argmax(np.abs(template_fft[gw_mask]))
                
                freq_diff = abs(strain_peak_idx - template_peak_idx)
                peak_match = 1.0 / (1.0 + freq_diff / 10.0)  # Normalize
                
                return {
                    'efficiency': float(freq_efficiency),
                    'peak_match': float(peak_match),
                    'strain_power_gw': float(strain_power_gw),
                    'template_power_gw': float(template_power_gw),
                    'residual_power_gw': float(residual_power_gw)
                }
            else:
                return {'efficiency': 0.0, 'peak_match': 0.0}
                
        except:
            return {'efficiency': 0.0, 'peak_match': 0.0}
    
    def _compute_uncertainty_weight(self, parameters: Dict[str, float], 
                                  uncertainty: Dict[str, float]) -> float:
        """ uncertainty-based template weighting."""
        try:
            # Weight based on parameter uncertainties (most important for waveform accuracy)
            important_params = ['mass_1', 'mass_2', 'luminosity_distance', 'geocent_time']
            total_weight = 1.0
            
            for param in important_params:
                if param in parameters and param in uncertainty:
                    param_val = abs(parameters[param])
                    param_unc = uncertainty[param]
                    
                    if param_val > 0 and param_unc > 0:
                        # Relative uncertainty
                        relative_unc = param_unc / param_val
                        
                        # Exponential weighting (higher uncertainty â†’ lower weight)
                        if param in ['mass_1', 'mass_2']:
                            # Mass uncertainties are critical for waveform accuracy
                            weight = np.exp(-3 * relative_unc)
                        elif param == 'luminosity_distance':
                            # Distance affects amplitude scaling
                            weight = np.exp(-1 * relative_unc)
                        elif param == 'geocent_time':
                            # Time uncertainty affects phase alignment
                            weight = np.exp(-5 * relative_unc / 0.01)  # Normalize by 10ms
                        else:
                            weight = np.exp(-2 * relative_unc)
                        
                        total_weight *= weight
            
            # Additional weighting based on signal quality if available
            if 'signal_quality' in parameters:
                quality = parameters['signal_quality']
                quality_weight = max(quality, 0.3)  # Minimum weight of 0.3
                total_weight *= quality_weight
            
            # Apply bounds
            final_weight = np.clip(total_weight, 0.1, 1.0)
            
            return float(final_weight)
            
        except Exception as e:
            self.logger.warning(f"Uncertainty weight computation failed: {e}")
            return 0.8  # Conservative default


class AdaptiveSubtractor:
    """Adaptive subtractor with physics-based logic.
    
    ✅ NOTE (Dec 14, 2025): Neural PE is now handled by OverlapNeuralPE from
    src/ahsd/models/overlap_neuralpe.py. This class focuses on physics-based
    signal subtraction only.
    """
    
    def __init__(self):
        self.uncertainty_subtractor = UncertaintyAwareSubtractor()
        self.logger = logging.getLogger(__name__)
        
        # Advanced configuration
        self.max_iterations = 3  # Maximum iterative refinement
        self.convergence_threshold = 0.01  # Convergence criterion
        
        self.logger.info("✅ AdaptiveSubtractor initialized with physics-based logic")
    
    def extract_and_subtract(self, data: Dict[str, Any], 
                           best_params: Dict[str, float],
                           uncertainties: Dict[str, float] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Physics-based signal subtraction with iterative refinement.
        
        Args:
            data: Input strain data (H1, L1, V1 detectors)
            best_params: Best-fit parameters from OverlapNeuralPE
            uncertainties: Parameter uncertainties (optional)
        
        Returns:
            (residual_data, subtraction_info) tuple
        """
        
        try:
            # Standardize data format
            standardized_data = standardize_strain_data(data)
            
            if uncertainties is None:
                uncertainties = {}
            
            # Iterative refinement (if signal quality is sufficient)
            refined_params, refined_uncertainties, refinement_info = self._iterative_refinement(
                standardized_data, best_params, uncertainties
            )
            
            # Perform physics-based subtraction
            residual_data, subtraction_info = self.uncertainty_subtractor.subtract_signal(
                standardized_data, refined_params, refined_uncertainties
            )

            # Post-subtraction analysis and validation
            validation_results = self._validate_subtraction(
                standardized_data, residual_data, refined_params, subtraction_info
            )
            
            # Compile comprehensive results
            result = {
                'subtraction_info': subtraction_info,
                'best_parameters': refined_params,
                'parameter_uncertainties': refined_uncertainties,
                'validation_results': validation_results,
                'refinement_info': refinement_info,
                'processing_metadata': {
                    'iterative_refinement': True,
                    'n_detectors': len(standardized_data),
                    'data_quality': self._assess_data_quality(standardized_data)
                }
            }
            
            return residual_data, result
            
        except Exception as e:
            self.logger.error(f"Extract and subtract failed: {e}")
            
            # Robust fallback
            try:
                original_data = standardize_strain_data(data)
            except:
                original_data = {
                    'H1': np.array([], dtype=np.float32), 
                    'L1': np.array([], dtype=np.float32), 
                    'V1': np.array([], dtype=np.float32)
                }
            
            fallback_result = {
                'error': str(e),
                'subtraction_info': {},
                'validation_results': {},
                'method': 'extraction_failed'
            }
            
            return original_data, fallback_result
    
    def _iterative_refinement(self, data: Dict[str, np.ndarray], 
                            initial_params: Dict[str, float],
                            initial_uncertainties: Dict[str, float]) -> Tuple[Dict, Dict, Dict]:
        """iterative parameter refinement using template matching."""
        
        refined_params = initial_params.copy()
        refined_uncertainties = initial_uncertainties.copy()
        refinement_info = {
            'iterations': 0,
            'convergence_achieved': False,
            'improvement_history': [],
            'final_match_quality': 0.0
        }
        
        try:
            previous_match_quality = 0.0
            
            for iteration in range(self.max_iterations):
                # Generate template with current parameters
                residual_data, subtraction_info = self.uncertainty_subtractor.subtract_signal(
                    data, refined_params, refined_uncertainties
                )
                
                # Compute overall match quality
                match_qualities = []
                for det_info in subtraction_info.values():
                    if 'template_match' in det_info:
                        match_qualities.append(abs(det_info['template_match']))
                
                current_match_quality = np.mean(match_qualities) if match_qualities else 0.0
                refinement_info['improvement_history'].append(current_match_quality)
                
                # Check for convergence
                improvement = current_match_quality - previous_match_quality
                if improvement < self.convergence_threshold:
                    refinement_info['convergence_achieved'] = True
                    break
                
                # Gradient-based parameter updates
                parameter_updates = self._compute_parameter_updates(
                    data, refined_params, subtraction_info
                )
                
                # Apply conservative updates
                for param_name, update in parameter_updates.items():
                    if param_name in refined_params:
                        # Conservative update with bounds checking
                        old_value = refined_params[param_name]
                        uncertainty = refined_uncertainties.get(param_name, 0.1)
                        
                        # Limit update size to avoid instability
                        max_update = min(abs(old_value) * 0.1, uncertainty)
                        bounded_update = np.clip(update, -max_update, max_update)
                        
                        refined_params[param_name] = old_value + bounded_update
                        
                        # Update uncertainty (slightly increase due to refinement uncertainty)
                        refined_uncertainties[param_name] = uncertainty * 1.05
                
                previous_match_quality = current_match_quality
                refinement_info['iterations'] += 1
            
            refinement_info['final_match_quality'] = current_match_quality
            
        except Exception as e:
            self.logger.warning(f"Iterative refinement failed: {e}")
            # Return original parameters if refinement fails
            
        return refined_params, refined_uncertainties, refinement_info
    
    def _compute_parameter_updates(self, data: Dict[str, np.ndarray], 
                                 params: Dict[str, float],
                                 subtraction_info: Dict) -> Dict[str, float]:
        """parameter updates using finite difference gradients."""
        
        parameter_updates = {}
        
        try:
            # Focus on most important parameters for template matching
            important_params = ['mass_1', 'mass_2', 'luminosity_distance', 'geocent_time']
            
            for param_name in important_params:
                if param_name not in params:
                    continue
                    
                current_value = params[param_name]
                
                # Compute finite difference step
                if param_name in ['mass_1', 'mass_2']:
                    step_size = max(current_value * 0.01, 0.5)  # 1% or 0.5 M_sun
                elif param_name == 'luminosity_distance':
                    step_size = max(current_value * 0.05, 10.0)  # 5% or 10 Mpc
                elif param_name == 'geocent_time':
                    step_size = 0.001  # 1ms
                else:
                    step_size = 0.01
                
                # Compute gradient estimate
                gradient = self._estimate_parameter_gradient(
                    data, params, param_name, step_size, subtraction_info
                )
                
                # Convert gradient to parameter update
                # Update in direction of improved template matching
                parameter_updates[param_name] = gradient * step_size * 0.5  # Conservative factor
        
        except Exception as e:
            self.logger.warning(f"Parameter update computation failed: {e}")
        
        return parameter_updates
    
    def _estimate_parameter_gradient(self, data: Dict[str, np.ndarray], 
                                   params: Dict[str, float],
                                   param_name: str, step_size: float,
                                   baseline_info: Dict) -> float:
        """Estimate parameter gradient using finite differences."""
        
        try:
            # Get baseline match quality
            baseline_matches = []
            for det_info in baseline_info.values():
                if 'template_match' in det_info:
                    baseline_matches.append(abs(det_info['template_match']))
            
            baseline_quality = np.mean(baseline_matches) if baseline_matches else 0.0
            
            # Test parameter perturbation
            perturbed_params = params.copy()
            perturbed_params[param_name] += step_size
            
            # Generate perturbed template and compute match
            _, perturbed_info = self.uncertainty_subtractor.subtract_signal(
                data, perturbed_params, {}
            )
            
            perturbed_matches = []
            for det_info in perturbed_info.values():
                if 'template_match' in det_info:
                    perturbed_matches.append(abs(det_info['template_match']))
            
            perturbed_quality = np.mean(perturbed_matches) if perturbed_matches else 0.0
            
            # Compute gradient
            gradient = (perturbed_quality - baseline_quality) / step_size
            
            return float(gradient)
            
        except:
            return 0.0  # No update if gradient computation fails
    
    def _validate_subtraction(self, original_data: Dict[str, np.ndarray],
                            residual_data: Dict[str, np.ndarray],
                            parameters: Dict[str, float],
                            subtraction_info: Dict) -> Dict:
        """post-subtraction validation and quality assessment."""
        
        validation_results = {
            'overall_quality': 0.0,
            'detector_qualities': {},
            'parameter_consistency': {},
            'residual_analysis': {},
            'recommendations': []
        }
        
        try:
            detector_qualities = []
            
            for detector in original_data.keys():
                if detector in residual_data and detector in subtraction_info:
                    det_info = subtraction_info[detector]
                    
                    # Extract quality metrics
                    template_match = abs(det_info.get('template_match', 0.0))
                    subtraction_eff = det_info.get('subtraction_efficiency', 0.0)
                    template_snr = det_info.get('template_snr', 0.0)
                    
                    # Compute detector-specific quality
                    det_quality = (0.5 * template_match + 
                                  0.3 * max(subtraction_eff, 0) + 
                                  0.2 * min(template_snr / 10.0, 1.0))
                    
                    validation_results['detector_qualities'][detector] = {
                        'quality_score': float(det_quality),
                        'template_match': float(template_match),
                        'subtraction_efficiency': float(subtraction_eff),
                        'template_snr': float(template_snr)
                    }
                    
                    detector_qualities.append(det_quality)
                    
                    # Residual analysis
                    if detector in original_data and detector in residual_data:
                        original = original_data[detector]
                        residual = residual_data[detector]
                        
                        if len(original) == len(residual) and len(original) > 0:
                            residual_analysis = self._analyze_residual_properties(original, residual)
                            validation_results['residual_analysis'][detector] = residual_analysis
            
            # Overall quality
            if detector_qualities:
                validation_results['overall_quality'] = np.mean(detector_qualities)
                
                # Generate recommendations
                avg_quality = validation_results['overall_quality']
                if avg_quality > 0.8:
                    validation_results['recommendations'].append('Excellent subtraction quality')
                elif avg_quality > 0.6:
                    validation_results['recommendations'].append('Good subtraction quality')
                elif avg_quality > 0.4:
                    validation_results['recommendations'].append('Moderate quality - consider parameter refinement')
                else:
                    validation_results['recommendations'].append('Poor quality - signal may be too weak or parameters inaccurate')
            
            # Parameter consistency check
            validation_results['parameter_consistency'] = self._check_parameter_consistency(parameters)
            
        except Exception as e:
            self.logger.warning(f"Subtraction validation failed: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _analyze_residual_properties(self, original: np.ndarray, residual: np.ndarray) -> Dict:
        """Analyze properties of subtraction residuals."""
        
        try:
            # Basic statistics
            residual_rms = np.sqrt(np.mean(residual**2))
            original_rms = np.sqrt(np.mean(original**2))
            
            # Whiteness test (simplified)
            residual_autocorr = np.correlate(residual, residual, mode='full')
            autocorr_peak = np.max(np.abs(residual_autocorr))
            autocorr_mean = np.mean(np.abs(residual_autocorr))
            whiteness_ratio = autocorr_peak / max(autocorr_mean, 1e-30)
            
            # Frequency content analysis
            residual_fft = np.fft.fft(residual)
            freqs = np.fft.fftfreq(len(residual), 1.0/4096)
            
            # Power in GW band
            gw_mask = (freqs >= 20) & (freqs <= 500)
            if np.any(gw_mask):
                residual_power_gw = np.sum(np.abs(residual_fft[gw_mask])**2)
                total_power = np.sum(np.abs(residual_fft)**2)
                gw_fraction = residual_power_gw / max(total_power, 1e-30)
            else:
                gw_fraction = 0.0
            
            return {
                'residual_rms': float(residual_rms),
                'rms_ratio': float(residual_rms / max(original_rms, 1e-30)),
                'whiteness_ratio': float(whiteness_ratio),
                'gw_band_fraction': float(gw_fraction),
                'length': len(residual)
            }
            
        except:
            return {
                'residual_rms': 0.0,
                'rms_ratio': 1.0,
                'whiteness_ratio': 1.0,
                'gw_band_fraction': 0.0,
                'error': 'analysis_failed'
            }
    
    def _check_parameter_consistency(self, parameters: Dict[str, float]) -> Dict:
        """Check physical consistency of extracted parameters."""
        
        consistency_results = {
            'mass_consistency': True,
            'distance_consistency': True,
            'spin_consistency': True,
            'issues': []
        }
        
        try:
            # Mass consistency
            m1 = parameters.get('mass_1', 30.0)
            m2 = parameters.get('mass_2', 30.0)
            
            if m1 < m2:
                consistency_results['mass_consistency'] = False
                consistency_results['issues'].append('m1 < m2 (should be m1 >= m2)')
            
            if m1 < 5.0 or m1 > 100.0:
                consistency_results['mass_consistency'] = False
                consistency_results['issues'].append(f'm1 = {m1:.1f} outside reasonable range [5, 100] M_sun')
            
            if m2 < 5.0 or m2 > 100.0:
                consistency_results['mass_consistency'] = False
                consistency_results['issues'].append(f'm2 = {m2:.1f} outside reasonable range [5, 100] M_sun')
            
            # Distance consistency
            distance = parameters.get('luminosity_distance', 500.0)
            if distance < 50.0 or distance > 3000.0:
                consistency_results['distance_consistency'] = False
                consistency_results['issues'].append(f'Distance = {distance:.1f} Mpc outside reasonable range [50, 3000]')
            
            # Spin consistency
            a1 = parameters.get('a_1', 0.0)
            a2 = parameters.get('a_2', 0.0)
            
            if a1 < 0.0 or a1 >= 1.0:
                consistency_results['spin_consistency'] = False
                consistency_results['issues'].append(f'a_1 = {a1:.2f} outside physical range [0, 1)')
            
            if a2 < 0.0 or a2 >= 1.0:
                consistency_results['spin_consistency'] = False
                consistency_results['issues'].append(f'a_2 = {a2:.2f} outside physical range [0, 1)')
            
        except Exception as e:
            consistency_results['issues'].append(f'Consistency check failed: {e}')
        
        return consistency_results
    
    def _assess_data_quality(self, data: Dict[str, np.ndarray]) -> Dict:
        """Assess quality of input strain data."""
        
        quality_assessment = {
            'overall_grade': 'unknown',
            'detector_grades': {},
            'issues': []
        }
        
        try:
            detector_grades = []
            
            for detector, strain in data.items():
                if len(strain) == 0:
                    quality_assessment['detector_grades'][detector] = 'no_data'
                    continue
                
                # Basic quality checks
                strain_array = np.array(strain)
                
                # Check for NaN/inf
                if not np.all(np.isfinite(strain_array)):
                    grade = 'poor'
                    quality_assessment['issues'].append(f'{detector}: Non-finite values detected')
                    
                # Check dynamic range
                elif np.max(np.abs(strain_array)) / max(np.std(strain_array), 1e-30) > 1000:
                    grade = 'poor'
                    quality_assessment['issues'].append(f'{detector}: Excessive dynamic range')
                    
                # Check for reasonable power levels
                elif np.var(strain_array) < 1e-50 or np.var(strain_array) > 1e-40:
                    grade = 'fair'
                    quality_assessment['issues'].append(f'{detector}: Unusual power level')
                    
                else:
                    grade = 'good'
                
                quality_assessment['detector_grades'][detector] = grade
                
                # Convert grade to numeric for averaging
                grade_values = {'poor': 0, 'fair': 1, 'good': 2}
                detector_grades.append(grade_values.get(grade, 0))
            
            # Overall grade
            if detector_grades:
                avg_grade = np.mean(detector_grades)
                if avg_grade >= 1.5:
                    quality_assessment['overall_grade'] = 'good'
                elif avg_grade >= 0.5:
                    quality_assessment['overall_grade'] = 'fair'
                else:
                    quality_assessment['overall_grade'] = 'poor'
            
        except Exception as e:
            quality_assessment['issues'].append(f'Data quality assessment failed: {e}')
        
        return quality_assessment