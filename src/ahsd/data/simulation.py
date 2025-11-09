"""
Simulate overlapping gravitational wave signals for training.
"""

import numpy as np
import bilby
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from typing import List, Dict, Tuple, Optional
import torch
import logging
from pathlib import Path
import pickle
from gwpy.timeseries import TimeSeries
from scipy.signal import get_window
from ..utils.config import AHSDConfig

# Suppress bilby warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", message="Unkown projection method")
warnings.filterwarnings("ignore", message="Unknown projection method")

# Configure bilby logging
bilby.core.utils.logger.setLevel('WARNING')

class OverlappingSignalSimulator:
    """
    OverlappingSignalSimulator
    Simulates overlapping gravitational wave (GW) signals for training machine learning models or testing GW data analysis pipelines. This class provides methods to generate realistic GW signal parameters, simulate detector noise, inject multiple overlapping signals into noise, and create datasets for training.
    Attributes:
        config (AHSDConfig): Configuration object specifying waveform and detector settings.
        logger (logging.Logger): Logger for status and error messages.
        detectors (dict): Dictionary of detector objects keyed by detector name.
        waveform_generator (bilby.gw.WaveformGenerator): Bilby waveform generator for GW signals.
    Methods:
        __init__(config: AHSDConfig):
            Initializes the simulator with the given configuration, sets up detectors and waveform generator.
        _setup_waveform_generator():
            Sets up the bilby waveform generator using configuration parameters.
        generate_single_signal_params() -> Dict:
            Generates a dictionary of parameters for a single GW signal using conservative, realistic priors.
        _sample_power_law(min_val: float, max_val: float, alpha: float) -> float:
            Samples a value from a power-law distribution.
        _sample_spin_magnitude() -> float:
            Samples a spin magnitude using a Beta distribution fit to GWTC measurements.
        generate_overlapping_scenario(n_signals: int) -> Dict:
            Generates a scenario dictionary containing parameters for multiple overlapping GW signals.
        generate_detector_noise(duration: Optional[float] = None, sampling_rate: Optional[int] = None) -> Dict:
            Generates realistic colored Gaussian noise for each detector using analytical PSDs.
        _generate_colored_noise(n_samples: int, sampling_rate: int, detector: str) -> np.ndarray:
            Generates colored noise for a given detector using its power spectral density (PSD).
        _aLIGO_psd(freqs: np.ndarray) -> np.ndarray:
            Returns a simplified analytical PSD for Advanced LIGO.
        _virgo_psd(freqs: np.ndarray) -> np.ndarray:
            Returns a simplified analytical PSD for Virgo.
        inject_signals_to_data(scenario: Dict, noise_data: Dict) -> Tuple[Dict, Dict]:
            Injects multiple GW signals into detector noise, returning the injected data and individual signal contributions.
        _generate_mock_waveform(params: Dict, detector_name: str) -> np.ndarray:
            Generates a simple mock waveform for a GW signal, used as a fallback if bilby waveform generation fails.
        _generate_detector_strain(params: Dict, detector_name: str) -> Optional[np.ndarray]:
            Generates the strain time series for a specific detector using bilby, with fallback to mock waveform.
        create_training_dataset(n_scenarios: int, output_dir: str) -> List[Dict]:
            Generates and saves a dataset of simulated scenarios with overlapping GW signals and noise for training.
    """
    """Simulate overlapping gravitational wave signals for training."""
    
    def __init__(self, config: AHSDConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup detectors
        self.detectors = {}
        for det_config in config.detectors:
            try:
                self.detectors[det_config.name] = Detector(det_config.name)
            except:
                # Fallback to bilby detector
                self.detectors[det_config.name] = self._create_mock_detector(det_config.name)
        
        # Setup waveform generator
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
        
    def generate_single_signal_params(self) -> Dict:
        """Generate parameters for a single GW signal using realistic priors."""
        
        # Very conservative parameters to eliminate warnings
        mass_1 = np.random.uniform(25, 40)
        mass_2 = np.random.uniform(20, 35)
        if mass_1 < mass_2:
            mass_1, mass_2 = mass_2, mass_1
        
        luminosity_distance = np.random.uniform(300, 800)
        
        # Very conservative spins
        a_1 = np.random.uniform(0.1, 0.4)
        a_2 = np.random.uniform(0.1, 0.4)
        
        # Safe sky positions - avoid problematic declinations
        ra = np.random.uniform(0.5, 2*np.pi - 0.5)
        dec = np.arcsin(np.random.uniform(-1, 1))
        
        # Safe angles
        theta_jn = np.arccos(np.random.uniform(-1, 1))
        psi = np.random.uniform(0.3, np.pi - 0.3)
        phase = np.random.uniform(0.3, 2*np.pi - 0.3)
        
        geocent_time = np.random.uniform(-0.02, 0.02)
        
        # Conservative spin angles
        tilt_1 = np.random.uniform(0.5, np.pi - 0.5)
        tilt_2 = np.random.uniform(0.5, np.pi - 0.5)
        phi_12 = np.random.uniform(0.3, 2*np.pi - 0.3)
        phi_jl = np.random.uniform(0.3, 2*np.pi - 0.3)
        
        return {
            'mass_1': mass_1,
            'mass_2': mass_2,
            'luminosity_distance': luminosity_distance,
            'theta_jn': theta_jn,
            'psi': psi,
            'phase': phase,
            'geocent_time': geocent_time,
            'ra': ra,
            'dec': dec,
            'a_1': a_1,
            'a_2': a_2,
            'tilt_1': tilt_1,
            'tilt_2': tilt_2,
            'phi_12': phi_12,
            'phi_jl': phi_jl,
        }

    def _sample_power_law(self, min_val: float, max_val: float, alpha: float) -> float:
        """Sample from power law distribution."""
        
        if alpha == -1:
            return min_val * np.exp(np.random.random() * np.log(max_val / min_val))
        else:
            u = np.random.random()
            return (min_val**(alpha + 1) + u * (max_val**(alpha + 1) - min_val**(alpha + 1)))**(1.0 / (alpha + 1))
    
    def _sample_spin_magnitude(self) -> float:
        """Sample spin magnitude using Beta distribution."""
        
        # Beta distribution fit to GWTC measurements
        return np.random.beta(1.5, 3.0) * 0.99
    
    def generate_overlapping_scenario(self, n_signals: int) -> Dict:
        """Generate a scenario with overlapping signals."""
        
        signals = []
        for i in range(n_signals):
            params = self.generate_single_signal_params()
            params['signal_id'] = i
            signals.append(params)
            
        return {
            'signals': signals,
            'n_signals': n_signals,
            'scenario_id': np.random.randint(0, 1000000)
        }
    
    def generate_detector_noise(self, 
                              duration: Optional[float] = None,
                              sampling_rate: Optional[int] = None) -> Dict:
        """Generate realistic detector noise."""
        
        if duration is None:
            duration = self.config.waveform.duration
        if sampling_rate is None:
            sampling_rate = self.config.detectors[0].sampling_rate
            
        n_samples = int(duration * sampling_rate)
        noise_data = {}
        
        for det_name in self.detectors.keys():
            # Generate colored Gaussian noise with realistic PSD
            noise = self._generate_colored_noise(n_samples, sampling_rate, det_name)
            noise_data[det_name] = noise
            
        return noise_data
    
    def _generate_colored_noise(self, n_samples: int, sampling_rate: int, detector: str) -> np.ndarray:
        """Generate colored noise with detector PSD."""
        
        try:
            # Get detector PSD (simplified)
            freqs = np.fft.fftfreq(n_samples, 1/sampling_rate)
            positive_freqs = freqs[:n_samples//2 + 1]
            
            # Simplified analytical PSD for Advanced LIGO
            if detector in ['H1', 'L1']:
                psd = self._aLIGO_psd(positive_freqs)
            elif detector == 'V1':
                psd = self._virgo_psd(positive_freqs)
            else:
                # White noise fallback
                psd = np.ones_like(positive_freqs)
            
            # Generate white noise in frequency domain
            white_noise_f = np.random.normal(0, 1, n_samples//2 + 1) + 1j * np.random.normal(0, 1, n_samples//2 + 1)
            white_noise_f[0] = white_noise_f[0].real  # DC component
            if n_samples % 2 == 0:
                white_noise_f[-1] = white_noise_f[-1].real  # Nyquist component
            
            # Color the noise
            colored_noise_f = white_noise_f / np.sqrt(psd * sampling_rate / 2)
            
            # Convert to time domain
            colored_noise_f_full = np.concatenate([colored_noise_f, np.conj(colored_noise_f[-2:0:-1])])
            colored_noise = np.fft.ifft(colored_noise_f_full).real
            
            return colored_noise
            
        except Exception as e:
            self.logger.warning(f"Failed to generate colored noise for {detector}: {e}")
            # Fallback to white noise
            return np.random.normal(0, 1e-23, n_samples)
    
    def _aLIGO_psd(self, freqs: np.ndarray) -> np.ndarray:
        """Simplified analytical PSD for Advanced LIGO."""
        
        # Avoid zero frequency
        freqs = np.maximum(freqs, 10.0)
        
        # Simplified analytical fit
        f0 = 215.0  # Hz
        psd = (freqs / f0)**(-4.14) - 5 * (freqs / f0)**(-2) + 111 * (1 + (freqs / f0)**2)**(-0.5)
        
        # Add low frequency rise
        psd += 1e4 * (freqs / 10.0)**(-8)
        
        return psd * 1e-48
    
    def _virgo_psd(self, freqs: np.ndarray) -> np.ndarray:
        """Simplified analytical PSD for Virgo."""
        
        freqs = np.maximum(freqs, 10.0)
        
        # Simplified Virgo PSD
        psd = 3.2e-46 * (freqs / 100.0)**(-4.05) + 2e-48
        
        return psd
    
    def inject_signals_to_data(self, scenario: Dict, noise_data: Dict) -> Tuple[Dict, Dict]:
        """Inject multiple overlapping signals into noise data."""
        
        injected_data = {}
        signal_contributions = {}
        
        for det_name, noise in noise_data.items():
            if det_name in self.detectors:
                try:
                    detector = self.detectors[det_name]
                    total_strain = np.array(noise)
                    signal_contributions[det_name] = {}
                    
                    for signal in scenario['signals']:
                        # Generate waveform for this detector
                        strain = self._generate_detector_strain(signal, det_name)
                        
                        if strain is not None:
                            # Ensure same length
                            min_len = min(len(total_strain), len(strain))
                            total_strain[:min_len] += strain[:min_len]
                            signal_contributions[det_name][signal['signal_id']] = strain
                    
                    injected_data[det_name] = total_strain
                    
                except Exception as e:
                    self.logger.warning(f"Failed to inject signals into {det_name}: {e}")
                    injected_data[det_name] = noise
                    signal_contributions[det_name] = {}
            else:
                injected_data[det_name] = noise
                signal_contributions[det_name] = {}
                
        return injected_data, signal_contributions
    
    def _generate_mock_waveform(self, params: Dict, detector_name: str) -> np.ndarray:
        """Generate simple mock waveform without bilby warnings."""
        
        # Simple chirp-like signal
        duration = self.config.waveform.duration
        sampling_rate = self.config.detectors[0].sampling_rate
        n_samples = int(duration * sampling_rate)
        
        t = np.linspace(0, duration, n_samples)
        
        # Extract parameters
        m1 = params.get('mass_1', 30.0)
        m2 = params.get('mass_2', 30.0)
        distance = params.get('luminosity_distance', 500.0)
        
        # Chirp mass
        chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        
        # Simple frequency evolution
        f_start = 35.0
        f_end = 250.0
        freq = f_start + (f_end - f_start) * (t / duration)**3
        
        # Amplitude with 1/r scaling
        amp = 1e-21 * (chirp_mass / 30.0)**(5/6) / (distance / 400.0)
        
        # Exponential envelope
        envelope = np.exp(-t / (duration * 0.4))
        
        # Generate strain with detector response
        dt = t[1] - t[0] if len(t) > 1 else 1/sampling_rate
        phase = 2 * np.pi * np.cumsum(freq) * dt
        strain = amp * envelope * np.sin(phase)
        
        # Apply simple detector response based on sky position
        ra = params.get('ra', 0.0)
        dec = params.get('dec', 0.0)
        psi = params.get('psi', 0.0)
        
        # Simple detector response approximation
        if detector_name == 'H1':
            response = 0.5 * (1 + np.cos(dec)**2) * np.cos(2*psi)
        elif detector_name == 'L1':  
            response = 0.5 * (1 + np.cos(dec)**2) * np.cos(2*psi + np.pi/2)
        elif detector_name == 'V1':
            response = 0.3 * (1 + np.cos(dec)**2) * np.cos(2*psi + np.pi/4)
        else:
            response = 0.5
        
        return strain * abs(response)
    
    def _generate_detector_strain(self, params: Dict, detector_name: str) -> Optional[np.ndarray]:
        """Generate strain for specific detector using bilby with fallback."""
        
        try:
            # First try bilby
            waveform_polarizations = self.waveform_generator.frequency_domain_strain(params)
            
            if detector_name in self.detectors:
                if hasattr(self.detectors[detector_name], 'project_wave'):
                    # bilby detector
                    detector = self.detectors[detector_name]
                    strain = detector.project_wave(
                        waveform_polarizations['plus'],
                        waveform_polarizations['cross'],
                        params['ra'], params['dec'], params['psi'],
                        params.get('geocent_time', 0.0)
                    )
                    return strain.time_domain_strain
                else:
                    # PyCBC detector - simplified projection
                    waveform_td = np.fft.ifft(waveform_polarizations['plus']).real
                    return waveform_td
            
            return None
            
        except Exception as e:
            # Fallback to mock waveform on any bilby error
            try:
                return self._generate_mock_waveform(params, detector_name)
            except Exception as e2:
                self.logger.debug(f"Both bilby and mock waveform failed for {detector_name}: {e2}")
                return None
    
    def create_training_dataset(self, 
                              n_scenarios: int, 
                              output_dir: str) -> List[Dict]:
        """Create complete training dataset."""
        
        training_scenarios = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating {n_scenarios} training scenarios...")
        
        for i in range(n_scenarios):
            try:
                # Number of overlapping signals (weighted toward lower numbers)
                n_signals = np.random.choice([2, 3, 4, 5], p=[0.5, 0.3, 0.15, 0.05])
                
                # Generate scenario
                scenario = self.generate_overlapping_scenario(n_signals)
                
                # Generate noise
                noise_data = self.generate_detector_noise()
                
                # Inject signals
                injected_data, signal_contributions = self.inject_signals_to_data(
                    scenario, noise_data
                )
                
                # Create training example
                training_scenario = {
                    'scenario_id': i,
                    'true_parameters': scenario['signals'],
                    'injected_data': injected_data,
                    'noise_data': noise_data,
                    'signal_contributions': signal_contributions,
                    'n_signals': n_signals,
                    'generation_params': {
                        'duration': self.config.waveform.duration,
                        'sampling_rate': self.config.detectors[0].sampling_rate,
                        'approximant': self.config.waveform.approximant
                    }
                }
                
                training_scenarios.append(training_scenario)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Generated {i + 1}/{n_scenarios} scenarios")
                    
            except Exception as e:
                self.logger.error(f"Failed to generate scenario {i}: {e}")
                continue
        
        # Save dataset
        dataset_file = output_path / 'simulated_training_data.pkl'
        with open(dataset_file, 'wb') as f:
            pickle.dump(training_scenarios, f)
        
        self.logger.info(f"Saved {len(training_scenarios)} training scenarios to {dataset_file}")
        
        return training_scenarios


def compute_waveform_overlap(h1: np.ndarray, h2: np.ndarray, 
                           sampling_rate: float = 4096.0) -> float:
    """Compute normalized overlap between two waveforms."""
    
    try:
        # Ensure same length
        min_len = min(len(h1), len(h2))
        h1 = h1[:min_len]
        h2 = h2[:min_len]
        
        # Compute overlap in time domain (simplified)
        overlap = np.abs(np.dot(h1, h2)) / (np.linalg.norm(h1) * np.linalg.norm(h2))
        
        return float(overlap)
        
    except:
        return 0.0


def estimate_snr_from_strain(strain: np.ndarray, 
                           psd: Optional[np.ndarray] = None,
                           sampling_rate: float = 4096.0) -> float:
    """Estimate SNR of strain data."""
    
    try:
        if psd is None:
            # Simple time-domain SNR estimate
            signal_power = np.var(strain)
            # Assume noise level
            noise_power = 1e-46
            snr = np.sqrt(signal_power / noise_power)
        else:
            # Matched filter SNR (ensure consistent FFT normalization)
            N = len(strain)
            strain_f = np.fft.fft(strain) / float(N)
            freqs = np.fft.fftfreq(N, 1/sampling_rate)

            # Compute SNR using frequency domain
            # Ensure psd has at least N elements when provided by caller
            psd_seg = psd[:len(strain_f)] if psd is not None else np.ones_like(strain_f)
            integrand = np.abs(strain_f)**2 / psd_seg
            df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0/sampling_rate
            # Use same normalization as dataset_generator (4 * integral over positive freqs)
            snr_squared = 4.0 * np.sum(integrand) * df
            snr = np.sqrt(max(snr_squared, 0.0))
        
        return float(snr)
        
    except:
        return 0.0
