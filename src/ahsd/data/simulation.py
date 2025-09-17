import numpy as np
import bilby
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from typing import List, Dict, Tuple, Optional
import torch
import logging
from gwpy.timeseries import TimeSeries
from scipy.signal import get_window
from ..utils.config import AHSDConfig

class OverlappingSignalSimulator:
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
                self.detectors[det_config.name] = bilby.gw.detector.get_detector(det_config.name)
        
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
        
        # Component masses with power law + peak distribution
        if np.random.random() < 0.7:
            # Power law component
            mass_1 = self._sample_power_law(5, 80, -2.35)
        else:
            # Peak around 35 solar masses  
            mass_1 = np.clip(np.random.normal(35, 5), 20, 50)
        
        # Secondary mass
        mass_2 = np.random.uniform(5, mass_1)
        
        # Ensure mass ordering
        if mass_1 < mass_2:
            mass_1, mass_2 = mass_2, mass_1
        
        # Distance: uniform in comoving volume
        luminosity_distance = self._sample_power_law(100, 2000, 2)
        
        # Spins based on GWTC measurements
        a_1 = self._sample_spin_magnitude()
        a_2 = self._sample_spin_magnitude()
        
        # Isotropic sky location
        ra = np.random.uniform(0, 2*np.pi)
        dec = np.arcsin(np.random.uniform(-1, 1))
        
        # Orientation angles
        theta_jn = np.arccos(np.random.uniform(-1, 1))
        psi = np.random.uniform(0, np.pi)
        phase = np.random.uniform(0, 2*np.pi)
        
        # Time offset for overlapping
        geocent_time = np.random.uniform(-0.5, 0.5)
        
        # Spin orientations
        tilt_1 = np.arccos(np.random.uniform(-1, 1))
        tilt_2 = np.arccos(np.random.uniform(-1, 1))
        phi_12 = np.random.uniform(0, 2*np.pi)
        phi_jl = np.random.uniform(0, 2*np.pi)
        
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
    
    def _generate_detector_strain(self, params: Dict, detector_name: str) -> Optional[np.ndarray]:
        """Generate strain for specific detector using bilby."""
        
        try:
            # Generate polarizations
            waveform_polarizations = self.waveform_generator.frequency_domain_strain(params)
            
            # Get detector
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
                    detector = self.detectors[detector_name]
                    # This would need proper PyCBC projection implementation
                    # For now, return a simplified version
                    waveform_td = np.fft.ifft(waveform_polarizations['plus']).real
                    return waveform_td
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Waveform generation failed for {detector_name}: {e}")
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
        import pickle
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
            # Matched filter SNR
            strain_f = np.fft.fft(strain)
            freqs = np.fft.fftfreq(len(strain), 1/sampling_rate)
            
            # Compute SNR using frequency domain
            integrand = np.abs(strain_f)**2 / psd[:len(strain_f)]
            snr_squared = 4 * np.sum(integrand) / sampling_rate
            snr = np.sqrt(snr_squared)
        
        return float(snr)
        
    except:
        return 0.0
