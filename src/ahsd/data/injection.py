import numpy as np
from typing import Dict, List, Tuple, Optional
import bilby
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import logging
from ..utils.config import AHSDConfig

class RealDataSignalInjector:
    """Inject synthetic gravitational wave signals into real detector noise"""
    
    def __init__(self, config: AHSDConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.waveform_generator = self._setup_waveform_generator()
        
    def _setup_waveform_generator(self):
        """Setup bilby waveform generator"""
        
        return bilby.gw.WaveformGenerator(
            duration=self.config.waveform.duration,
            sampling_frequency=self.config.detectors.sampling_rate,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(
                waveform_approximant=self.config.waveform.approximant,
                reference_frequency=self.config.waveform.f_ref,
            )
        )
    
    def create_overlapping_injection(self, 
                                   background_data: Dict,
                                   signal_parameters: List[Dict],
                                   target_snrs: Optional[List[float]] = None) -> Tuple[Dict, Dict]:
        """Create overlapping injection scenario from real background data"""
        
        if not signal_parameters:
            return background_data, {}
        
        injected_data = {}
        signal_contributions = {}
        
        # Use provided SNRs or generate random ones
        if target_snrs is None:
            target_snrs = [np.random.uniform(8, 20) for _ in signal_parameters]
        
        for det_name, background_strain in background_data.items():
            try:
                # Ensure background is a numpy array
                if hasattr(background_strain, 'value'):
                    background_array = background_strain.value
                else:
                    background_array = np.array(background_strain)
                
                # Start with background
                total_strain = background_array.copy()
                signal_contributions[det_name] = {}
                
                # Inject each signal
                for i, (params, target_snr) in enumerate(zip(signal_parameters, target_snrs)):
                    try:
                        # Generate waveform
                        strain_signal = self._generate_detector_waveform(params, det_name)
                        
                        if strain_signal is not None:
                            # Scale to achieve target SNR
                            scaled_signal = self._scale_to_target_snr(
                                strain_signal, target_snr, background_array, det_name
                            )
                            
                            # Add to total (ensure same length)
                            min_len = min(len(total_strain), len(scaled_signal))
                            total_strain[:min_len] += scaled_signal[:min_len]
                            
                            # Store contribution
                            signal_contributions[det_name][i] = scaled_signal
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to inject signal {i} in {det_name}: {e}")
                        continue
                
                injected_data[det_name] = total_strain
                
            except Exception as e:
                self.logger.error(f"Failed to process detector {det_name}: {e}")
                injected_data[det_name] = background_strain
                signal_contributions[det_name] = {}
        
        return injected_data, signal_contributions
    
    def _generate_detector_waveform(self, params: Dict, detector_name: str) -> Optional[np.ndarray]:
        """Generate waveform for specific detector"""
        
        try:
            # Get detector object
            detector = bilby.gw.detector.get_detector(detector_name)
            
            # Ensure parameters are complete with realistic defaults
            complete_params = self._complete_parameters(params)
            
            # Generate polarizations
            waveform_polarizations = self.waveform_generator.frequency_domain_strain(complete_params)
            
            # Project to detector
            strain = detector.project_wave(
                waveform_polarizations['plus'],
                waveform_polarizations['cross'],
                complete_params['ra'], 
                complete_params['dec'], 
                complete_params['psi']
            )
            
            return strain.time_domain_strain
            
        except Exception as e:
            self.logger.debug(f"Waveform generation failed for {detector_name}: {e}")
            return None
    
    def _complete_parameters(self, params: Dict) -> Dict:
        """Ensure all required parameters are present with realistic values"""
        
        complete_params = {
            # Masses (solar masses)
            'mass_1': 35.0,
            'mass_2': 30.0,
            
            # Distance and location
            'luminosity_distance': 500.0,  # Mpc
            'ra': np.random.uniform(0, 2*np.pi),
            'dec': np.arcsin(np.random.uniform(-1, 1)),
            
            # Orientation
            'theta_jn': np.arccos(np.random.uniform(-1, 1)),
            'psi': np.random.uniform(0, np.pi),
            'phase': np.random.uniform(0, 2*np.pi),
            
            # Time
            'geocent_time': 0.0,
            
            # Spins
            'a_1': 0.0,
            'a_2': 0.0,
            'tilt_1': 0.0,
            'tilt_2': 0.0,
            'phi_12': 0.0,
            'phi_jl': 0.0,
        }
        
        # Update with provided parameters
        complete_params.update(params)
        
        # Ensure masses are ordered correctly
        if complete_params['mass_1'] < complete_params['mass_2']:
            complete_params['mass_1'], complete_params['mass_2'] = complete_params['mass_2'], complete_params['mass_1']
        
        # Validate parameter ranges
        complete_params['mass_1'] = max(1.0, min(100.0, complete_params['mass_1']))
        complete_params['mass_2'] = max(1.0, min(100.0, complete_params['mass_2']))
        complete_params['luminosity_distance'] = max(10.0, min(5000.0, complete_params['luminosity_distance']))
        complete_params['a_1'] = max(0.0, min(0.99, complete_params['a_1']))
        complete_params['a_2'] = max(0.0, min(0.99, complete_params['a_2']))
        
        return complete_params
    
    def _scale_to_target_snr(self, waveform: np.ndarray, target_snr: float, 
                           background: np.ndarray, detector_name: str) -> np.ndarray:
        """Scale waveform to achieve target SNR in background noise"""
        
        try:
            # Estimate current SNR (simplified)
            signal_power = np.var(waveform)
            noise_power = np.var(background)
            
            if signal_power > 0 and noise_power > 0:
                current_snr = np.sqrt(signal_power / noise_power)
                if current_snr > 0:
                    scale_factor = target_snr / current_snr
                    return waveform * scale_factor
            
            # Fallback: simple scaling
            noise_std = np.std(background)
            if noise_std > 0:
                scale_factor = target_snr * noise_std / max(np.std(waveform), 1e-25)
                return waveform * scale_factor
            
        except Exception as e:
            self.logger.debug(f"SNR scaling failed for {detector_name}: {e}")
        
        # Return original if scaling fails
        return waveform
    
    def generate_realistic_parameters(self, n_signals: int) -> List[Dict]:
        """Generate realistic BBH parameters based on GWTC observations"""
        
        parameters = []
        
        for i in range(n_signals):
            # Mass distribution based on GWTC observations
            # Primary mass: power law + peak
            if np.random.random() < 0.7:
                # Power law component
                mass_1 = self._sample_power_law(5, 50, -2.3)
            else:
                # Peak component around 35 solar masses
                mass_1 = np.random.normal(35, 5)
                mass_1 = np.clip(mass_1, 20, 50)
            
            # Secondary mass
            mass_2 = np.random.uniform(5, mass_1)
            
            # Distance: uniform in comoving volume
            distance = self._sample_power_law(100, 2000, 2)
            
            # Spins: based on GWTC measurements
            a_1 = self._sample_beta_spin()
            a_2 = self._sample_beta_spin()
            
            # Isotropic sky location
            ra = np.random.uniform(0, 2*np.pi)
            dec = np.arcsin(np.random.uniform(-1, 1))
            
            # Random orientations
            theta_jn = np.arccos(np.random.uniform(-1, 1))
            psi = np.random.uniform(0, np.pi)
            phase = np.random.uniform(0, 2*np.pi)
            
            # Small time offsets for overlapping
            if i == 0:
                geocent_time = 0.0
            else:
                geocent_time = np.random.uniform(-0.5, 0.5)
            
            # Spin orientations
            tilt_1 = np.arccos(np.random.uniform(-1, 1))
            tilt_2 = np.arccos(np.random.uniform(-1, 1))
            phi_12 = np.random.uniform(0, 2*np.pi)
            phi_jl = np.random.uniform(0, 2*np.pi)
            
            params = {
                'mass_1': mass_1,
                'mass_2': mass_2,
                'luminosity_distance': distance,
                'ra': ra,
                'dec': dec,
                'theta_jn': theta_jn,
                'psi': psi,
                'phase': phase,
                'geocent_time': geocent_time,
                'a_1': a_1,
                'a_2': a_2,
                'tilt_1': tilt_1,
                'tilt_2': tilt_2,
                'phi_12': phi_12,
                'phi_jl': phi_jl,
                'signal_id': i
            }
            
            parameters.append(params)
        
        return parameters
    
    def _sample_power_law(self, min_val: float, max_val: float, alpha: float) -> float:
        """Sample from power law distribution"""
        
        if alpha == -1:
            return min_val * np.exp(np.random.random() * np.log(max_val / min_val))
        else:
            u = np.random.random()
            return (min_val**(alpha + 1) + u * (max_val**(alpha + 1) - min_val**(alpha + 1)))**(1.0 / (alpha + 1))
    
    def _sample_beta_spin(self) -> float:
        """Sample spin magnitude from Beta distribution fit to GWTC data"""
        
        # Beta distribution parameters fit to GWTC spin measurements
        a, b = 1.5, 3.0
        return np.random.beta(a, b) * 0.99  # Cap at 0.99
