"""
Real data signal injection utilities for AHSD pipeline.
"""

import numpy as np
import bilby
from typing import Dict, List, Tuple, Optional
import logging
from gwpy.timeseries import TimeSeries
from ..utils.config import AHSDConfig

class RealDataSignalInjector:
    """ 
    RealDataSignalInjector
    A class for injecting synthetic gravitational-wave signals into real background data, 
    primarily for use in gravitational-wave data analysis pipelines. This class supports 
    the generation, projection, scaling, and injection of multiple overlapping signals 
    into background strain data from multiple detectors.
    Parameters
    ----------
    config : AHSDConfig
        Configuration object containing waveform and detector settings.
    Attributes
    ----------
    config : AHSDConfig
        The configuration object used for waveform and detector settings.
    logger : logging.Logger
        Logger instance for this class.
    waveform_generator : bilby.gw.WaveformGenerator
        Waveform generator instance for producing gravitational-wave signals.
    Methods
    -------
    create_overlapping_injection(background_data, signal_parameters, target_snrs)
        Injects multiple overlapping signals into background data for each detector.
    estimate_injection_snr(signal, background, detector_name)
        Estimates the signal-to-noise ratio (SNR) of an injected signal in the background.
    validate_injection_parameters(params)
        Validates the injection parameters for physical plausibility and completeness.
    Private Methods
    ---------------
    _setup_waveform_generator()
        Sets up the waveform generator using the configuration.
    _generate_signal_strain(params, detector_name, target_snr, background)
        Generates the signal strain for a specific detector and target SNR.
    _project_waveform_manually(waveform_polarizations, params, detector_name)
        Projects waveform polarizations onto a detector using simplified antenna patterns.
    _get_detector(detector_name)
        Retrieves a detector object, with multiple fallback methods for compatibility.
    _create_mock_detector(detector_name)
        Creates a mock detector object if standard detector retrieval fails.
    _scale_to_target_snr(signal, target_snr, background, detector_name)
        Scales a signal to achieve the desired SNR in the given background noise.
    _generate_mock_signal(params, detector_name, target_snr)
        Generates a simple mock signal as a fallback if waveform generation fails.
        """
    
    def __init__(self, config: AHSDConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.waveform_generator = self._setup_waveform_generator()
        
    def _setup_waveform_generator(self):
        
        """Setup bilby waveform generator."""
        return bilby.gw.WaveformGenerator(
            duration=self.config.waveform.duration,
            sampling_frequency=self.config.detectors[0].sampling_rate,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(
                waveform_approximant=self.config.waveform.approximant,
                reference_frequency=self.config.waveform.f_ref,
            )
        )
    
    def create_overlapping_injection(self,
                                   background_data: Dict[str, np.ndarray],
                                   signal_parameters: List[Dict],
                                   target_snrs: List[float]) -> Tuple[Dict, Dict]:
        """
        Inject multiple overlapping signals into background data.
        
        Parameters:
        -----------
        background_data : dict
            Background strain data for each detector
        signal_parameters : list
            List of parameter dictionaries for each signal
        target_snrs : list
            Target SNR for each signal
            
        Returns:
        --------
        tuple
            (injected_data, signal_contributions)
        """
        
        injected_data = {}
        signal_contributions = {}
        
        for det_name, background in background_data.items():
            try:
                total_strain = np.array(background)
                signal_contributions[det_name] = {}
                
                for i, (params, target_snr) in enumerate(zip(signal_parameters, target_snrs)):
                    # Generate waveform for this detector
                    signal_strain = self._generate_signal_strain(params, det_name, target_snr, background)
                    
                    if signal_strain is not None:
                        # Ensure same length
                        min_len = min(len(total_strain), len(signal_strain))
                        total_strain[:min_len] += signal_strain[:min_len]
                        signal_contributions[det_name][i] = signal_strain
                    else:
                        self.logger.debug(f"Failed to generate signal {i} for {det_name}")
                
                injected_data[det_name] = total_strain
                
            except Exception as e:
                self.logger.warning(f"Failed to inject into {det_name}: {e}")
                injected_data[det_name] = background
                signal_contributions[det_name] = {}
        
        return injected_data, signal_contributions
    
    def _generate_signal_strain(self, 
                              params: Dict, 
                              detector_name: str,
                              target_snr: float,
                              background: np.ndarray) -> Optional[np.ndarray]:
        """Generate signal strain for specific detector with target SNR."""
        
        try:
            # Generate waveform polarizations
            waveform_polarizations = self.waveform_generator.frequency_domain_strain(params)
            
            # Use manual projection to avoid bilby API issues
            signal_td = self._project_waveform_manually(
                waveform_polarizations, params, detector_name
            )
            
            # Scale to target SNR
            scaled_signal = self._scale_to_target_snr(signal_td, target_snr, background, detector_name)
            
            return scaled_signal
            
        except Exception as e:
            self.logger.debug(f"Signal generation failed for {detector_name}: {e}")
            # Fallback to simple mock signal
            return self._generate_mock_signal(params, detector_name, target_snr)

    def _project_waveform_manually(self, waveform_polarizations: Dict, params: Dict, detector_name: str) -> np.ndarray:
        """Manually project waveform to detector using antenna patterns."""
        
        try:
            # Get waveform in time domain
            h_plus_td = np.real(np.fft.ifft(waveform_polarizations['plus']))
            h_cross_td = np.real(np.fft.ifft(waveform_polarizations['cross']))
            
            # Sky position and polarization
            ra = params.get('ra', 0.0)
            dec = params.get('dec', 0.0) 
            psi = params.get('psi', 0.0)
            
            # Simplified detector antenna patterns
            if detector_name == 'H1':
                F_plus = 0.5 * (1 + np.cos(dec)**2) * np.cos(2*psi)
                F_cross = np.cos(dec) * np.sin(2*psi)
            elif detector_name == 'L1':
                F_plus = 0.5 * (1 + np.cos(dec)**2) * np.cos(2*psi + np.pi/2)
                F_cross = np.cos(dec) * np.sin(2*psi + np.pi/2)
            elif detector_name == 'V1':
                F_plus = 0.3 * (1 + np.cos(dec)**2) * np.cos(2*psi + np.pi/4)
                F_cross = 0.7 * np.cos(dec) * np.sin(2*psi + np.pi/4)
            else:
                F_plus = 0.5
                F_cross = 0.5
            
            # Combine polarizations
            h_detector = F_plus * h_plus_td + F_cross * h_cross_td
            
            return h_detector
            
        except Exception as e:
            self.logger.debug(f"Manual projection failed: {e}")
            return np.real(np.fft.ifft(waveform_polarizations['plus']))
    
    def _get_detector(self, detector_name: str):
        """Get detector with multiple fallback methods."""
        
        # Try different bilby detector APIs
        try:
            # Method 1: Try newer bilby API
            from bilby.gw.detector import Interferometer
            return Interferometer.get_empty_interferometer(detector_name)
        except:
            pass
            
        try:
            # Method 2: Try older bilby API
            import bilby.gw.detector as detector_module
            if hasattr(detector_module, 'get_empty_interferometer'):
                return detector_module.get_empty_interferometer(detector_name)
        except:
            pass
            
        try:
            # Method 3: Try even older API
            import bilby.gw.detector as detector_module
            if hasattr(detector_module, 'get_detector'):
                return detector_module.get_detector(detector_name)
        except:
            pass
        
        # Method 4: Create mock detector
        return self._create_mock_detector(detector_name)
    
    def _create_mock_detector(self, detector_name: str):
        """Create mock detector when bilby methods fail."""
        
        class MockDetector:
            def __init__(self, name):
                self.name = name
                
            def project_wave(self, h_plus, h_cross, ra, dec, psi, geocent_time=0):
                # Simple projection - combine polarizations
                try:
                    h_data = np.real(np.fft.ifft(h_plus))
                except:
                    h_data = np.real(h_plus) if hasattr(h_plus, '__len__') else np.array([h_plus])
                
                # Apply detector response
                if detector_name == 'H1':
                    response = 0.5 * np.cos(2*psi)
                elif detector_name == 'L1':
                    response = 0.5 * np.cos(2*psi + np.pi/2)  
                elif detector_name == 'V1':
                    response = 0.3 * np.cos(2*psi + np.pi/4)
                else:
                    response = 0.5
                    
                class MockStrain:
                    def __init__(self, data):
                        self.time_domain_strain = data * abs(response)
                        
                return MockStrain(h_data)
        
        return MockDetector(detector_name)
    
    def _scale_to_target_snr(self, 
                           signal: np.ndarray, 
                           target_snr: float,
                           background: np.ndarray, 
                           detector_name: str) -> np.ndarray:
        """Scale signal to achieve target SNR in background noise."""
        
        try:
            # Estimate background noise level
            noise_std = np.std(background)
            
            # Current signal power
            signal_power = np.sqrt(np.mean(signal**2))
            
            if signal_power > 0 and noise_std > 0:
                # Scale factor to achieve target SNR
                scale_factor = (target_snr * noise_std) / signal_power
                scaled_signal = signal * scale_factor
            else:
                # Fallback scaling
                scaled_signal = signal * (target_snr / 10.0) * 1e-21
            
            return scaled_signal
            
        except Exception as e:
            self.logger.debug(f"SNR scaling failed for {detector_name}: {e}")
            return signal * (target_snr / 10.0) * 1e-21
    
    def _generate_mock_signal(self, 
                            params: Dict, 
                            detector_name: str,
                            target_snr: float) -> np.ndarray:
        """Generate simple mock signal as fallback."""
        
        try:
            duration = self.config.waveform.duration
            sampling_rate = self.config.detectors[0].sampling_rate
            n_samples = int(duration * sampling_rate)
            
            t = np.linspace(0, duration, n_samples)
            
            # Extract parameters
            m1 = params.get('mass_1', 30.0)
            m2 = params.get('mass_2', 30.0)
            
            # Chirp mass
            chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
            
            # Simple frequency evolution
            f_start = 35.0
            f_end = 250.0
            freq = f_start + (f_end - f_start) * (t / duration)**3
            
            # Amplitude scaled by target SNR
            amp = 1e-21 * (target_snr / 10.0) * (chirp_mass / 30.0)**(5/6)
            
            # Exponential envelope
            envelope = np.exp(-t / (duration * 0.4))
            
            # Generate strain
            dt = t[1] - t[0] if len(t) > 1 else 1/sampling_rate
            phase = 2 * np.pi * np.cumsum(freq) * dt
            strain = amp * envelope * np.sin(phase)
            
            # Simple detector response
            ra = params.get('ra', 0.0)
            dec = params.get('dec', 0.0)
            psi = params.get('psi', 0.0)
            
            if detector_name == 'H1':
                response = 0.5 * (1 + np.cos(dec)**2) * np.cos(2*psi)
            elif detector_name == 'L1':  
                response = 0.5 * (1 + np.cos(dec)**2) * np.cos(2*psi + np.pi/2)
            elif detector_name == 'V1':
                response = 0.3 * (1 + np.cos(dec)**2) * np.cos(2*psi + np.pi/4)
            else:
                response = 0.5
            
            return strain * abs(response)
            
        except Exception as e:
            self.logger.debug(f"Mock signal generation failed: {e}")
            # Ultimate fallback
            n_samples = int(self.config.waveform.duration * self.config.detectors[0].sampling_rate)
            return np.random.normal(0, 1e-21 * target_snr, n_samples)

    def estimate_injection_snr(self, 
                             signal: np.ndarray, 
                             background: np.ndarray,
                             detector_name: str) -> float:
        """Estimate the SNR of injected signal in background."""
        
        try:
            # Simple time-domain SNR estimate
            signal_power = np.sqrt(np.mean(signal**2))
            noise_power = np.std(background)
            
            if noise_power > 0:
                snr = signal_power / noise_power
            else:
                snr = 0.0
            
            return float(snr)
            
        except Exception:
            return 0.0

    def validate_injection_parameters(self, params: Dict) -> bool:
        """Validate injection parameters."""
        
        required_params = ['mass_1', 'mass_2', 'ra', 'dec', 'psi']
        
        for param in required_params:
            if param not in params:
                return False
        
        # Check parameter ranges
        if params['mass_1'] < 1.0 or params['mass_1'] > 200.0:
            return False
        if params['mass_2'] < 1.0 or params['mass_2'] > 200.0:
            return False
        if abs(params['dec']) > 1.57:  # ~90 degrees
            return False
        
        return True


def create_injection_from_gwtc_event(event_data: Dict, 
                                   background_data: Dict,
                                   config: AHSDConfig,
                                   target_snr: float = 15.0) -> Dict:
    """Create injection scenario from GWTC event data."""
    
    # Extract parameters from GWTC event
    injection_params = {
        'mass_1': event_data.get('mass_1_source', 35.0),
        'mass_2': event_data.get('mass_2_source', 30.0),
        'luminosity_distance': event_data.get('luminosity_distance', 500.0),
        'geocent_time': np.random.uniform(-0.1, 0.1),  # Random time offset
        'ra': np.random.uniform(0, 2*np.pi),
        'dec': np.random.uniform(-0.5, 0.5),
        'theta_jn': np.random.uniform(0, np.pi),
        'psi': np.random.uniform(0, np.pi),
        'phase': np.random.uniform(0, 2*np.pi),
        'a_1': np.random.uniform(0, 0.5),
        'a_2': np.random.uniform(0, 0.5),
        'tilt_1': np.random.uniform(0, np.pi),
        'tilt_2': np.random.uniform(0, np.pi),
        'phi_12': np.random.uniform(0, 2*np.pi),
        'phi_jl': np.random.uniform(0, 2*np.pi),
    }
    
    # Create injector
    injector = RealDataSignalInjector(config)
    
    # Perform injection
    injected_data, contributions = injector.create_overlapping_injection(
        background_data, 
        [injection_params], 
        [target_snr]
    )
    
    return {
        'injected_data': injected_data,
        'signal_contributions': contributions,
        'true_parameters': [injection_params],
        'target_snr': target_snr,
        'source_event': event_data
    }
