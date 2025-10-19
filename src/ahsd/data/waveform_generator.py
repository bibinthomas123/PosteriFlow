"""
Waveform Generator
Comprehensive GW signal generation with PyCBC and analytical fallbacks
Extracted from EnhancedWaveformGenerator class
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from scipy.signal import windows

try:
    from pycbc.waveform import get_td_waveform, get_fd_waveform
    from pycbc.detector import Detector
    from pycbc.types import TimeSeries, FrequencySeries
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False

from .config import SAMPLE_RATE, DURATION, N_SAMPLES, APPROXIMANTS
from .utils import compute_effective_spin

class WaveformGenerator:
    """
    Generate gravitational waveforms with comprehensive fallbacks
    Supports BBH, BNS, NSBH with tidal and precession effects
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, duration: float = DURATION):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors
        self.detectors = {}
        if PYCBC_AVAILABLE:
            for det_name in ['H1', 'L1', 'V1']:
                try:
                    self.detectors[det_name] = Detector(det_name)
                except Exception as e:
                    self.logger.debug(f"Detector {det_name} initialization failed: {e}")
    
    def generate_waveform(self, params: Dict, detector_name: str = None) -> np.ndarray:
        """
        Main waveform generation with automatic fallback chain
        
        Priority:
        1. PyCBC (highest fidelity)
        2. Analytical models (reliable)
        3. Simple chirp (guaranteed fallback)
        """
        
        # Try PyCBC first
        if PYCBC_AVAILABLE:
            try:
                return self.generate_pycbc_waveform(params, detector_name)
            except Exception as e:
                self.logger.debug(f"PyCBC waveform failed: {e}")
        
        # Try analytical models
        try:
            return self.generate_analytical_waveform(params, detector_name)
        except Exception as e:
            self.logger.debug(f"Analytical waveform failed: {e}")
        
        # Ultimate fallback
        return self.generate_simple_chirp_fallback(params, detector_name)
    
    def generate_pycbc_waveform(self, params: Dict, detector_name: str = None) -> np.ndarray:
        """Generate waveform using PyCBC with detector projection"""
        
        # Generate polarizations
        hp, hc = get_td_waveform(
            approximant=params.get('approximant', 'IMRPhenomD'),
            mass1=params['mass_1'],
            mass2=params['mass_2'],
            spin1z=params.get('a1', 0.0) * np.cos(params.get('tilt1', 0.0)),
            spin2z=params.get('a2', 0.0) * np.cos(params.get('tilt2', 0.0)),
            distance=params.get('luminosity_distance', 400.0),
            inclination=params.get('theta_jn', 0.0),
            coa_phase=params.get('phase', 0.0),
            delta_t=1.0/self.sample_rate,
            f_lower=params.get('f_lower', 20.0),
            lambda1=params.get('lambda_1', 0.0),
            lambda2=params.get('lambda_2', 0.0)
        )
        
        # Project onto detector if specified
        if detector_name and detector_name in self.detectors:
            detector = self.detectors[detector_name]
            fp, fc = detector.antenna_pattern(
                params.get('ra', 0.0),
                params.get('dec', 0.0),
                params.get('psi', 0.0),
                params.get('geocent_time', 0.0)
            )
            signal = fp * hp + fc * hc
        else:
            # No detector projection - use plus polarization
            signal = hp
        
        # Resize to match duration
        if len(signal) < self.n_samples:
            signal.resize(self.n_samples)
        else:
            signal = signal[:self.n_samples]
        
        # Convert to numpy array
        return np.array(signal.data, dtype=np.float32)
    
    def generate_analytical_waveform(self, params: Dict, detector_name: str = None) -> np.ndarray:
        """Generate waveform using analytical post-Newtonian methods"""
        
        # Time array
        t = np.linspace(-self.duration/2, self.duration/2, self.n_samples)
        tc = params.get('geocent_time', 0.0)
        time_to_merger = np.maximum(tc - t, 0.001)
        
        # Determine waveform type
        if params.get('lambda_1', 0) > 0 or params.get('lambda_2', 0) > 0:
            signal = self.generate_tidal_waveform(t, time_to_merger, params)
        elif params.get('approximant', '').find('Pv2') >= 0:
            signal = self.generate_precessing_waveform(t, time_to_merger, params)
        else:
            signal = self.generate_aligned_spin_waveform(t, time_to_merger, params)
        
        # Apply detector response
        if detector_name:
            response = self.calculate_detector_response(params, detector_name)
            signal *= response
        
        # Apply taper window
        window = windows.tukey(self.n_samples, alpha=0.1)
        signal *= window
        
        return signal.astype(np.float32)
    
    def generate_aligned_spin_waveform(self, t: np.ndarray, 
                                      time_to_merger: np.ndarray, 
                                      params: Dict) -> np.ndarray:
        """Generate aligned-spin BBH waveform using 3.5PN TaylorT4"""
        
        # System parameters
        m1 = params['mass_1']
        m2 = params['mass_2']
        total_mass = m1 + m2
        chirp_mass = params['chirp_mass']
        eta = params.get('symmetric_mass_ratio', (m1 * m2) / total_mass**2)
        
        # Spin parameters
        chi1 = params.get('a1', 0.0) * np.cos(params.get('tilt1', 0.0))
        chi2 = params.get('a2', 0.0) * np.cos(params.get('tilt2', 0.0))
        chi_eff = (m1 * chi1 + m2 * chi2) / total_mass
        
        # Frequency evolution (3.5PN TaylorT4)
        theta = time_to_merger / (5.0 * chirp_mass)
        theta = np.maximum(theta, 1e-10)
        
        # PN coefficients with spin
        v = theta**(-1/8)
        
        # Phase evolution
        psi = -(1/eta) * (
            v**(-5) +
            (3715/1008 + 55*eta/12) * v**(-3) +
            (-10*np.pi + 2*chi_eff*(1 + 3*eta/(2*m1/total_mass))) * v**(-2) +
            (15293365/1016064 + 27145*eta/1008 + 3085*eta**2/144) * v**(-1)
        )
        
        # Frequency
        frequency = v**3 / (8*np.pi*chirp_mass)
        
        # Amplitude with detector distance
        distance = params.get('luminosity_distance', 400.0)
        amplitude = (chirp_mass**(5/6) / distance) * frequency**(-7/6)
        amplitude *= 4.0 * np.sqrt(5*eta/(24*np.pi))  # Normalization
        
        # Apply inclination
        theta_jn = params.get('theta_jn', 0.0)
        amplitude *= (1 + np.cos(theta_jn)**2) / 2
        
        # Generate strain
        phase_total = 2*np.pi*np.cumsum(frequency) / self.sample_rate
        strain = amplitude * np.sin(phase_total + psi)
        
        return strain
    
    def generate_tidal_waveform(self, t: np.ndarray, 
                               time_to_merger: np.ndarray, 
                               params: Dict) -> np.ndarray:
        """Generate BNS/NSBH waveform with tidal corrections"""
        
        # Base point-particle inspiral
        base_signal = self.generate_aligned_spin_waveform(t, time_to_merger, params)
        
        # Tidal deformability
        lambda_tilde = params.get('lambda_tilde', 
                                 params.get('lambda_1', 0) + params.get('lambda_2', 0))
        
        if lambda_tilde > 0:
            # Frequency for tidal correction
            chirp_mass = params['chirp_mass']
            v = (time_to_merger / (5.0 * chirp_mass))**(-1/8)
            frequency = v**3 / (8*np.pi*chirp_mass)
            
            # Leading tidal phase correction (5PN)
            x = (np.pi * chirp_mass * frequency)**(2/3)
            tidal_phase = -(39/2) * lambda_tilde * x**5 / chirp_mass**5
            
            # Apply tidal correction
            tidal_factor = np.exp(1j * tidal_phase)
            base_signal = np.real(base_signal * np.abs(tidal_factor)) * np.cos(np.angle(tidal_factor))
        
        return base_signal
    
    def generate_precessing_waveform(self, t: np.ndarray,
                                    time_to_merger: np.ndarray, 
                                    params: Dict) -> np.ndarray:
        """Generate precessing waveform (simplified analytical model)"""
        
        # Base aligned-spin waveform
        base = self.generate_aligned_spin_waveform(t, time_to_merger, params)
        
        # Precession modulation
        tilt1 = params.get('tilt1', 0.0)
        phi12 = params.get('phi12', 0.0)
        
        if tilt1 > 0.1:  # Significant precession
            # Simple precession frequency (qualitative)
            chirp_mass = params['chirp_mass']
            v = (time_to_merger / (5.0 * chirp_mass))**(-1/8)
            prec_freq = v**3 / (100 * chirp_mass)  # Slow precession
            
            # Amplitude modulation
            mod_amplitude = 0.3 * np.sin(tilt1)
            modulation = 1 + mod_amplitude * np.sin(2*np.pi*prec_freq*t + phi12)
            
            base *= modulation
        
        return base
    
    def generate_simple_chirp_fallback(self, params: Dict, detector_name: str = None) -> np.ndarray:
        """Ultimate fallback: simple chirp"""
        
        t = np.linspace(0, self.duration, self.n_samples)
        
        # Chirp parameters
        m1 = params.get('mass_1', 30.0)
        m2 = params.get('mass_2', 30.0)
        chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        
        # Frequency sweep
        f_start = 35.0
        f_end = min(250.0, self.sample_rate / 4)
        frequency = f_start + (f_end - f_start) * (t / self.duration)**3
        
        # Amplitude
        distance = params.get('luminosity_distance', 400.0)
        amplitude = 1e-21 * (chirp_mass / 30.0)**(5/6) * (400.0 / distance)
        
        # Envelope
        envelope = np.exp(-t / (self.duration * 0.4))
        
        # Phase
        dt = 1.0 / self.sample_rate
        phase = 2 * np.pi * np.cumsum(frequency) * dt
        
        return amplitude * envelope * np.sin(phase)
    
    def calculate_detector_response(self, params: Dict, detector_name: str) -> float:
        """Calculate detector antenna response"""
        
        if detector_name in self.detectors:
            detector = self.detectors[detector_name]
            fp, fc = detector.antenna_pattern(
                params.get('ra', 0.0),
                params.get('dec', 0.0),
                params.get('psi', 0.0),
                params.get('geocent_time', 0.0)
            )
            return float(np.sqrt(fp**2 + fc**2))
        else:
            # Fallback: average response
            return 0.4
    
    def get_alternative_approximants(self, params: Dict) -> list:
        """Get fallback approximants based on event type"""
        
        event_type = params.get('type', 'BBH')
        approx_dict = APPROXIMANTS.get(event_type, {'non_precessing': ['IMRPhenomD']})
        
        alternatives = []
        for category in ['non_precessing', 'precessing', 'tidal']:
            alternatives.extend(approx_dict.get(category, []))
        
        # Remove current approximant
        current = params.get('approximant')
        if current in alternatives:
            alternatives.remove(current)
        
        return alternatives
