"""
Waveform generation and manipulation utilities.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, List
import logging
from scipy import signal
import bilby

class WaveformUtilities:
    """Utilities for gravitational waveform generation and processing."""
    
    def __init__(self, 
                 duration: float = 4.0,
                 sampling_frequency: float = 4096.0,
                 f_lower: float = 20.0):
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.f_lower = f_lower
        self.logger = logging.getLogger(__name__)
        
    def generate_bbh_waveform(self, 
                             parameters: Dict,
                             approximant: str = "IMRPhenomPv2") -> Dict:
        """
        Generate binary black hole waveform.
        
        Parameters:
        -----------
        parameters : dict
            BBH parameters (masses, spins, etc.)
        approximant : str
            Waveform approximant
            
        Returns:
        --------
        dict
            Dictionary containing plus and cross polarizations
        """
        
        try:
            # Setup waveform generator
            waveform_generator = bilby.gw.WaveformGenerator(
                duration=self.duration,
                sampling_frequency=self.sampling_frequency,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                waveform_arguments=dict(
                    waveform_approximant=approximant,
                    reference_frequency=self.f_lower,
                )
            )
            
            # Generate waveform
            waveform_polarizations = waveform_generator.frequency_domain_strain(parameters)
            
            return waveform_polarizations
            
        except Exception as e:
            self.logger.error(f"Waveform generation failed: {e}")
            return None
    
    def project_waveform_to_detector(self,
                                   waveform_polarizations: Dict,
                                   detector_name: str,
                                   ra: float,
                                   dec: float, 
                                   psi: float,
                                   geocent_time: float = 0.0) -> Optional[np.ndarray]:
        """
        Project waveform to specific detector.
        
        Parameters:
        -----------
        waveform_polarizations : dict
            Plus and cross polarizations
        detector_name : str
            Detector name (H1, L1, V1)
        ra, dec, psi : float
            Sky location and polarization angle
        geocent_time : float
            GPS time at geocenter
            
        Returns:
        --------
        np.ndarray or None
            Strain time series for detector
        """
        
        try:
            # Get detector
            detector = bilby.gw.detector.get_detector(detector_name)
            
            # Project waveform
            strain = detector.project_wave(
                waveform_polarizations['plus'],
                waveform_polarizations['cross'],
                ra, dec, psi, geocent_time
            )
            
            return strain.time_domain_strain
            
        except Exception as e:
            self.logger.error(f"Waveform projection failed for {detector_name}: {e}")
            return None
    
    def compute_waveform_overlap(self,
                               h1: np.ndarray,
                               h2: np.ndarray,
                               psd: Optional[np.ndarray] = None) -> float:
        """
        Compute overlap between two waveforms.
        
        Parameters:
        -----------
        h1, h2 : np.ndarray
            Waveform time series
        psd : np.ndarray, optional
            Power spectral density for weighting
            
        Returns:
        --------
        float
            Overlap value (0 to 1)
        """
        
        try:
            # Ensure same length
            min_len = min(len(h1), len(h2))
            h1 = h1[:min_len]
            h2 = h2[:min_len]
            
            if psd is None:
                # Simple time-domain overlap
                overlap = np.abs(np.dot(h1, h2)) / (np.linalg.norm(h1) * np.linalg.norm(h2))
            else:
                # Frequency-domain overlap with PSD weighting
                h1_f = np.fft.fft(h1)
                h2_f = np.fft.fft(h2)
                
                # Truncate PSD to match FFT length
                psd_trunc = psd[:len(h1_f)//2 + 1]
                
                # Inner product
                integrand = np.conj(h1_f[:len(psd_trunc)]) * h2_f[:len(psd_trunc)] / psd_trunc
                inner_product = 4 * np.real(np.sum(integrand)) / self.sampling_frequency
                
                # Normalization
                norm1 = self.compute_waveform_norm(h1, psd)
                norm2 = self.compute_waveform_norm(h2, psd)
                
                overlap = inner_product / (norm1 * norm2)
            
            return float(np.abs(overlap))
            
        except Exception as e:
            self.logger.error(f"Overlap computation failed: {e}")
            return 0.0
    
    def compute_waveform_norm(self,
                            waveform: np.ndarray,
                            psd: Optional[np.ndarray] = None) -> float:
        """Compute norm of waveform."""
        
        if psd is None:
            return np.linalg.norm(waveform)
        else:
            waveform_f = np.fft.fft(waveform)
            psd_trunc = psd[:len(waveform_f)//2 + 1]
            
            integrand = np.abs(waveform_f[:len(psd_trunc)])**2 / psd_trunc
            norm_squared = 4 * np.sum(integrand) / self.sampling_frequency
            
            return np.sqrt(norm_squared)
    
    def estimate_snr(self,
                    waveform: np.ndarray,
                    psd: np.ndarray) -> float:
        """
        Estimate optimal SNR of waveform in colored noise.
        
        Parameters:
        -----------
        waveform : np.ndarray
            Waveform time series
        psd : np.ndarray
            Power spectral density
            
        Returns:
        --------
        float
            Optimal SNR
        """
        
        try:
            waveform_f = np.fft.fft(waveform)
            freqs = np.fft.fftfreq(len(waveform), 1/self.sampling_frequency)
            
            # Keep only positive frequencies
            positive_freqs = freqs[:len(freqs)//2 + 1]
            waveform_f_pos = waveform_f[:len(positive_freqs)]
            
            # Interpolate PSD to match frequency grid
            if len(psd) != len(positive_freqs):
                psd_interp = np.interp(positive_freqs, 
                                     np.linspace(0, self.sampling_frequency/2, len(psd)),
                                     psd)
            else:
                psd_interp = psd
            
            # Avoid division by zero
            psd_interp[psd_interp <= 0] = np.inf
            
            # Compute SNR
            integrand = np.abs(waveform_f_pos)**2 / psd_interp
            snr_squared = 4 * np.sum(integrand) / self.sampling_frequency
            
            return np.sqrt(snr_squared)
            
        except Exception as e:
            self.logger.error(f"SNR estimation failed: {e}")
            return 0.0
    
    def apply_tukey_window(self, 
                          waveform: np.ndarray,
                          alpha: float = 0.1) -> np.ndarray:
        """Apply Tukey window to waveform."""
        
        window = signal.windows.tukey(len(waveform), alpha)
        return waveform * window
    
    def bandpass_filter(self,
                       waveform: np.ndarray,
                       f_low: float = 20.0,
                       f_high: float = 1000.0,
                       order: int = 8) -> np.ndarray:
        """Apply bandpass filter to waveform."""
        
        try:
            # Normalized frequencies
            nyquist = self.sampling_frequency / 2
            low = f_low / nyquist
            high = f_high / nyquist
            
            # Design filter
            b, a = signal.butter(order, [low, high], btype='band')
            
            # Apply filter
            filtered_waveform = signal.filtfilt(b, a, waveform)
            
            return filtered_waveform
            
        except Exception as e:
            self.logger.error(f"Bandpass filtering failed: {e}")
            return waveform

def compute_chirp_mass(m1: float, m2: float) -> float:
    """Compute chirp mass from component masses."""
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)

def compute_total_mass(m1: float, m2: float) -> float:
    """Compute total mass from component masses."""
    return m1 + m2

def compute_mass_ratio(m1: float, m2: float) -> float:
    """Compute mass ratio (q = m2/m1, q <= 1)."""
    return min(m1, m2) / max(m1, m2)

def compute_symmetric_mass_ratio(m1: float, m2: float) -> float:
    """Compute symmetric mass ratio."""
    total_mass = m1 + m2
    return (m1 * m2) / (total_mass**2)
