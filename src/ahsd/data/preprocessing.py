"""
Data Preprocessing Module
Whitening, bandpass filtering, and quality validation
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from scipy.signal import butter, filtfilt, welch, windows
from scipy.interpolate import interp1d

from .config import SAMPLE_RATE, DURATION, F_LOWER, F_UPPER

class DataPreprocessor:
    """
    Preprocess GW strain data for analysis
    Implements whitening, filtering, and quality checks
    """
    
    def __init__(self, 
                 sample_rate: int = SAMPLE_RATE,
                 duration: float = DURATION,
                 f_low: float = F_LOWER,
                 f_high: float = F_UPPER):
        self.sample_rate = sample_rate
        self.duration = duration
        self.f_low = f_low
        self.f_high = f_high
        self.n_samples = int(sample_rate * duration)
        self.logger = logging.getLogger(__name__)
    
    def preprocess(self,
                  strain: np.ndarray,
                  psd_dict: Dict = None,
                  whiten: bool = True,
                  bandpass: bool = True,
                  remove_edges: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            strain: Raw strain data
            psd_dict: PSD for whitening
            whiten: Apply whitening
            bandpass: Apply bandpass filter
            remove_edges: Apply edge tapering
            
        Returns:
            Preprocessed strain
        """
        
        # Ensure correct dtype
        data = np.array(strain, dtype=np.float64)
        
        # Remove DC offset
        data = data - np.mean(data)
        
        # Bandpass filter
        if bandpass:
            data = self.bandpass_filter(data, self.f_low, self.f_high)
        
        # Whiten
        if whiten and psd_dict is not None:
            data = self.whiten_data(data, psd_dict)
        
        # Edge tapering
        if remove_edges:
            data = self.apply_tukey_window(data, alpha=0.1)
        
        return data.astype(np.float32)
    
    def whiten_data(self, strain: np.ndarray, psd_dict: Dict) -> np.ndarray:
        """
        Frequency-domain whitening using PSD
        """
        
        try:
            # FFT
            strain_fft = np.fft.rfft(strain)
            freqs_fft = np.fft.rfftfreq(len(strain), 1.0/self.sample_rate)
            
            # Get PSD
            if 'psd' in psd_dict:
                psd = psd_dict['psd']
                if hasattr(psd, 'numpy'):
                    psd = psd.numpy()
                psd_freqs = psd_dict['frequencies']
            else:
                # Estimate PSD from data
                psd_freqs, psd = welch(strain, fs=self.sample_rate, 
                                      nperseg=self.sample_rate)
            
            # Interpolate PSD to FFT frequencies
            psd_interp = interp1d(
                psd_freqs, psd,
                bounds_error=False,
                fill_value=(psd[0], psd[-1])
            )(freqs_fft)
            
            # Avoid division by zero
            psd_interp = np.maximum(psd_interp, 1e-50)
            
            # Whiten
            whitened_fft = strain_fft / np.sqrt(psd_interp * self.sample_rate / 2)
            
            # IFFT
            whitened = np.fft.irfft(whitened_fft, n=len(strain))
            
            # High-pass to remove low-freq artifacts
            whitened = self.highpass_filter(whitened, 10.0)
            
            return whitened
            
        except Exception as e:
            self.logger.warning(f"Whitening failed: {e}")
            return strain
    
    def bandpass_filter(self,
                       strain: np.ndarray,
                       f_low: float,
                       f_high: float,
                       order: int = 8) -> np.ndarray:
        """Apply Butterworth bandpass filter"""
        
        try:
            nyquist = self.sample_rate / 2
            low = max(f_low / nyquist, 1e-6)
            high = min(f_high / nyquist, 0.99)
            
            if low >= high:
                return strain
            
            b, a = butter(order, [low, high], btype='band')
            filtered = filtfilt(b, a, strain)
            
            return filtered
            
        except Exception as e:
            self.logger.warning(f"Bandpass filtering failed: {e}")
            return strain
    
    def highpass_filter(self,
                       strain: np.ndarray,
                       cutoff: float,
                       order: int = 4) -> np.ndarray:
        """Apply high-pass filter"""
        
        try:
            nyquist = self.sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            if normalized_cutoff >= 1.0:
                return strain
            
            b, a = butter(order, normalized_cutoff, btype='high')
            filtered = filtfilt(b, a, strain)
            
            return filtered
            
        except Exception as e:
            self.logger.warning(f"High-pass filtering failed: {e}")
            return strain
    
    def apply_tukey_window(self, strain: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """Apply Tukey window to reduce edge effects"""
        
        try:
            window = windows.tukey(len(strain), alpha)
            return strain * window
        except:
            return strain
    
    def validate_data(self, strain: np.ndarray) -> Dict:
        """
        Validate data quality
        
        Returns:
            Dictionary with validation results
        """
        
        report = {
            'passed': True,
            'warnings': [],
            'metrics': {}
        }
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(strain)):
            report['passed'] = False
            report['warnings'].append('Contains NaN or Inf values')
        
        # Check length
        if len(strain) != self.n_samples:
            report['warnings'].append(
                f'Length mismatch: expected {self.n_samples}, got {len(strain)}'
            )
        
        # Statistical checks
        report['metrics'] = {
            'length': len(strain),
            'mean': float(np.mean(strain)),
            'std': float(np.std(strain)),
            'max': float(np.max(strain)),
            'min': float(np.min(strain)),
            'rms': float(np.sqrt(np.mean(strain**2))),
            'finite_fraction': float(np.mean(np.isfinite(strain)))
        }
        
        # Check for saturation
        if report['metrics']['max'] > 1e-18:
            report['warnings'].append('Possible saturation detected')
        
        # Check for abnormal statistics
        if report['metrics']['std'] < 1e-25 or report['metrics']['std'] > 1e-20:
            report['warnings'].append(f"Unusual noise level: {report['metrics']['std']:.2e}")
        
        return report
    
    def compute_spectrogram(self,
                           strain: np.ndarray,
                           nperseg: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram for visualization
        
        Returns:
            (frequencies, times, spectrogram)
        """
        
        from scipy.signal import spectrogram
        
        if nperseg is None:
            nperseg = self.sample_rate // 4
        
        f, t, Sxx = spectrogram(
            strain,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=nperseg//2
        )
        
        return f, t, Sxx
