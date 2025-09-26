"""
Data preprocessing utilities for AHSD pipeline.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import signal
from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

class DataPreprocessor:
    """DataPreprocessor
    A class for preprocessing gravitational wave strain data from multiple detectors.
    Provides a modular pipeline for common preprocessing steps such as length adjustment,
    glitch removal, filtering, whitening, windowing, and validation. Designed to be
    configurable via a user-supplied configuration object.
    Parameters
    ----------
    config : object
        Configuration object containing detector and waveform parameters.
    Attributes
    ----------
    sampling_rate : int
        Sampling rate in Hz (default: 4096, can be overridden by config).
    duration : float
        Duration of the data segment in seconds (default: 4.0, can be overridden by config).
    f_low : float
        Low-frequency cutoff for filtering in Hz (default: 20.0, can be overridden by config).
    f_high : float
        High-frequency cutoff for filtering in Hz (default: 1024.0).
    Methods
    -------
    preprocess(data: Dict) -> Dict
        Main preprocessing pipeline for raw strain data from each detector.
    _preprocess_detector_data(strain: np.ndarray, detector: str) -> np.ndarray
        Preprocess data for a single detector, applying all steps.
    _ensure_correct_length(strain: np.ndarray) -> np.ndarray
        Ensure strain data has the correct length by truncating or zero-padding.
    _remove_glitches(strain: np.ndarray, threshold: float = 5.0) -> np.ndarray
        Remove glitches using a simple amplitude thresholding method.
    _highpass_filter(strain: np.ndarray, f_low: Optional[float] = None) -> np.ndarray
        Apply a high-pass Butterworth filter to remove low-frequency noise.
    _notch_filter(strain: np.ndarray) -> np.ndarray
        Apply notch filters to remove power line harmonics (e.g., 60 Hz and harmonics).
    _whiten(strain: np.ndarray, detector: str) -> np.ndarray
        Whiten strain data using a power spectral density (PSD) estimate.
    _apply_window(strain: np.ndarray, window_type: str = 'tukey', alpha: float = 0.1) -> np.ndarray
        Apply a window function (Tukey, Hann, or Hamming) to reduce edge effects.
    compute_psd(strain: np.ndarray, method: str = 'welch') -> Tuple[np.ndarray, np.ndarray]
        Compute the power spectral density (PSD) of the strain data.
    bandpass_filter(strain: np.ndarray, f_low: float, f_high: float, order: int = 8) -> np.ndarray
        Apply a bandpass Butterworth filter to the strain data.
    resample_data(strain: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray
        Resample strain data to a target sampling rate.
    validate_data(data: Dict) -> Dict
        Validate the quality of preprocessed data, checking for NaNs, length mismatches, and basic statistics.
    Logging
    -------
    Uses the standard Python logging module for debug, warning, and error messages during preprocessing steps.
    """    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Default processing parameters
        self.sampling_rate = 4096  # Hz
        self.duration = 4.0  # seconds
        self.f_low = 20.0  # Hz
        self.f_high = 1024.0  # Hz
        
        # Update from config if available
        if hasattr(config, 'detectors') and config.detectors:
            self.sampling_rate = config.detectors[0].sampling_rate
            self.duration = config.detectors[0].duration
            
        if hasattr(config, 'waveform'):
            self.f_low = config.waveform.f_lower
    
    def preprocess(self, data: Dict) -> Dict:
        """
        Main preprocessing pipeline.
        
        Parameters:
        -----------
        data : dict
            Raw strain data for each detector
            
        Returns:
        --------
        dict
            Preprocessed strain data
        """
        
        processed_data = {}
        
        for det_name, strain in data.items():
            try:
                # Convert to numpy array if needed
                if hasattr(strain, 'value'):
                    strain_array = strain.value
                elif hasattr(strain, 'data'):
                    strain_array = strain.data
                else:
                    strain_array = np.array(strain)
                
                # Apply preprocessing steps
                processed_strain = self._preprocess_detector_data(strain_array, det_name)
                
                processed_data[det_name] = processed_strain
                
            except Exception as e:
                self.logger.warning(f"Preprocessing failed for {det_name}: {e}")
                # Use original data if preprocessing fails
                processed_data[det_name] = np.array(strain)
        
        return processed_data
    
    def _preprocess_detector_data(self, strain: np.ndarray, detector: str) -> np.ndarray:
        """Preprocess data for single detector."""
        
        # Step 1: Ensure correct length
        strain = self._ensure_correct_length(strain)
        
        # Step 2: Remove glitches (simple approach)
        strain = self._remove_glitches(strain)
        
        # Step 3: High-pass filter to remove low-frequency noise
        strain = self._highpass_filter(strain)
        
        # Step 4: Notch filter for power line harmonics
        strain = self._notch_filter(strain)
        
        # Step 5: Whitening (optional, usually done in analysis)
        #strain = self._whiten(strain, detector)
        
        # Step 6: Apply window
        strain = self._apply_window(strain)
        
        return strain
    
    def _ensure_correct_length(self, strain: np.ndarray) -> np.ndarray:
        """Ensure strain has correct length."""
        
        target_length = int(self.duration * self.sampling_rate)
        
        if len(strain) > target_length:
            # Truncate from center
            start_idx = (len(strain) - target_length) // 2
            strain = strain[start_idx:start_idx + target_length]
        elif len(strain) < target_length:
            # Zero-pad
            pad_length = target_length - len(strain)
            pad_before = pad_length // 2
            pad_after = pad_length - pad_before
            strain = np.pad(strain, (pad_before, pad_after), mode='constant')
        
        return strain
    
    def _remove_glitches(self, strain: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """Simple glitch removal using amplitude threshold ( version)."""
        
        try:
            strain_array = np.array(strain, dtype=float)
            
            # Simple approach: clip extreme values
            std_val = np.std(strain_array)
            mean_val = np.mean(strain_array)
            
            if std_val > 0:
                # Clip values beyond threshold * std
                lower_bound = mean_val - threshold * std_val
                upper_bound = mean_val + threshold * std_val
                
                clipped_strain = np.clip(strain_array, lower_bound, upper_bound)
                
                # Count clipped values
                clipped_count = np.sum((strain_array < lower_bound) | (strain_array > upper_bound))
                if clipped_count > 0:
                    self.logger.debug(f"Clipped {clipped_count} outlier samples")
                
                return clipped_strain
            else:
                return strain_array
            
        except Exception as e:
            self.logger.debug(f"Glitch removal failed: {e}")
            return np.array(strain, dtype=float)
    
    def _highpass_filter(self, strain: np.ndarray, f_low: Optional[float] = None) -> np.ndarray:
        """Apply high-pass filter."""
        
        if f_low is None:
            f_low = self.f_low
        
        try:
            # Design high-pass filter
            nyquist = self.sampling_rate / 2
            normalized_freq = f_low / nyquist
            
            if normalized_freq < 1.0:
                b, a = signal.butter(8, normalized_freq, btype='high')
                filtered_strain = signal.filtfilt(b, a, strain)
                return filtered_strain
            
        except Exception as e:
            self.logger.debug(f"High-pass filtering failed: {e}")
        
        return strain
    
    def _notch_filter(self, strain: np.ndarray) -> np.ndarray:
        """Apply notch filters for power line harmonics."""
        
        # Common power line frequencies (60 Hz and harmonics for US)
        notch_freqs = [60, 120, 180]  # Hz
        
        try:
            filtered_strain = strain.copy()
            
            for freq in notch_freqs:
                if freq < self.sampling_rate / 2:
                    # Design notch filter
                    Q = 30  # Quality factor
                    w0 = freq / (self.sampling_rate / 2)  # Normalized frequency
                    b, a = signal.iirnotch(w0, Q)
                    filtered_strain = signal.filtfilt(b, a, filtered_strain)
            
            return filtered_strain
            
        except Exception as e:
            self.logger.debug(f"Notch filtering failed: {e}")
            return strain
    
    def _whiten(self, strain: np.ndarray, detector: str) -> np.ndarray:
        """Whiten strain data using PSD estimate."""
        
        try:
            # Estimate PSD
            freqs, psd = signal.welch(strain, 
                                    fs=self.sampling_rate,
                                    nperseg=self.sampling_rate//2,
                                    overlap=0.5)
            
            # Avoid division by zero
            psd[psd <= 0] = np.min(psd[psd > 0])
            
            # Whiten in frequency domain
            strain_fft = np.fft.fft(strain)
            freqs_fft = np.fft.fftfreq(len(strain), 1/self.sampling_rate)
            
            # Interpolate PSD to match FFT frequencies
            psd_interp = np.interp(np.abs(freqs_fft), freqs, psd)
            
            # Apply whitening
            whitened_fft = strain_fft / np.sqrt(psd_interp)
            whitened_strain = np.real(np.fft.ifft(whitened_fft))
            
            return whitened_strain
            
        except Exception as e:
            self.logger.debug(f"Whitening failed for {detector}: {e}")
            return strain
    
    def _apply_window(self, strain: np.ndarray, window_type: str = 'tukey', alpha: float = 0.1) -> np.ndarray:
        """Apply window function to reduce edge effects."""
        
        try:
            if window_type == 'tukey':
                window = signal.windows.tukey(len(strain), alpha)
            elif window_type == 'hann':
                window = signal.windows.hann(len(strain))
            elif window_type == 'hamming':
                window = signal.windows.hamming(len(strain))
            else:
                # No windowing
                return strain
            
            windowed_strain = strain * window
            return windowed_strain
            
        except Exception as e:
            self.logger.debug(f"Windowing failed: {e}")
            return strain
    
    def compute_psd(self, strain: np.ndarray, method: str = 'welch') -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density."""
        
        try:
            if method == 'welch':
                freqs, psd = signal.welch(strain,
                                        fs=self.sampling_rate,
                                        nperseg=self.sampling_rate,
                                        overlap=0.5)
            elif method == 'periodogram':
                freqs, psd = signal.periodogram(strain, fs=self.sampling_rate)
            else:
                raise ValueError(f"Unknown PSD method: {method}")
            
            return freqs, psd
            
        except Exception as e:
            self.logger.error(f"PSD computation failed: {e}")
            # Return dummy PSD
            freqs = np.linspace(0, self.sampling_rate/2, len(strain)//2 + 1)
            psd = np.ones_like(freqs)
            return freqs, psd
    
    def bandpass_filter(self, strain: np.ndarray, 
                       f_low: float, f_high: float, 
                       order: int = 8) -> np.ndarray:
        """Apply bandpass filter."""
        
        try:
            nyquist = self.sampling_rate / 2
            low = f_low / nyquist
            high = f_high / nyquist
            
            # Ensure frequencies are in valid range
            low = max(low, 1e-6)
            high = min(high, 0.99)
            
            if low >= high:
                self.logger.warning("Invalid bandpass frequencies")
                return strain
            
            b, a = signal.butter(order, [low, high], btype='band')
            filtered_strain = signal.filtfilt(b, a, strain)
            
            return filtered_strain
            
        except Exception as e:
            self.logger.debug(f"Bandpass filtering failed: {e}")
            return strain
    
    def resample_data(self, strain: np.ndarray, 
                     original_rate: int, 
                     target_rate: int) -> np.ndarray:
        """Resample data to target sampling rate."""
        
        if original_rate == target_rate:
            return strain
        
        try:
            # Use scipy's resample
            n_samples_new = int(len(strain) * target_rate / original_rate)
            resampled_strain = signal.resample(strain, n_samples_new)
            
            return resampled_strain
            
        except Exception as e:
            self.logger.error(f"Resampling failed: {e}")
            return strain
    
    def validate_data(self, data: Dict) -> Dict:
        """Validate preprocessed data quality."""
        
        validation_report = {
            'passed': True,
            'warnings': [],
            'statistics': {}
        }
        
        for det_name, strain in data.items():
            try:
                strain_array = np.array(strain)
                
                # Check for NaN/Inf
                if not np.all(np.isfinite(strain_array)):
                    validation_report['passed'] = False
                    validation_report['warnings'].append(f"{det_name}: Contains NaN/Inf values")
                
                # Check length
                expected_length = int(self.duration * self.sampling_rate)
                if len(strain_array) != expected_length:
                    validation_report['warnings'].append(
                        f"{det_name}: Length mismatch - expected {expected_length}, got {len(strain_array)}"
                    )
                
                # Statistical checks
                strain_std = np.std(strain_array)
                strain_max = np.max(np.abs(strain_array))
                
                validation_report['statistics'][det_name] = {
                    'length': len(strain_array),
                    'std': float(strain_std),
                    'max_abs': float(strain_max),
                    'finite_fraction': float(np.mean(np.isfinite(strain_array)))
                }
                
            except Exception as e:
                validation_report['warnings'].append(f"{det_name}: Validation error - {e}")
        
        return validation_report

def estimate_noise_floor(strain: np.ndarray, sampling_rate: int = 4096) -> float:
    """Estimate noise floor of strain data."""
    
    try:
        # Use high-frequency content to estimate noise
        freqs, psd = signal.welch(strain, fs=sampling_rate)
        
        # Use frequencies above 1000 Hz (typically dominated by noise)
        high_freq_mask = freqs > 1000
        if np.any(high_freq_mask):
            noise_floor = np.median(psd[high_freq_mask])
        else:
            noise_floor = np.median(psd)
        
        return float(noise_floor)
        
    except Exception:
        return 1e-46  # Fallback estimate

def compute_data_quality_metrics(strain: np.ndarray, sampling_rate: int = 4096) -> Dict:
    """Compute data quality metrics."""
    
    metrics = {}
    
    try:
        # Time domain metrics
        metrics['rms'] = float(np.sqrt(np.mean(strain**2)))
        metrics['peak'] = float(np.max(np.abs(strain)))
        metrics['kurtosis'] = float(signal.kurtosis(strain))
        metrics['skewness'] = float(signal.skew(strain))
        
        # Frequency domain metrics
        freqs, psd = signal.welch(strain, fs=sampling_rate)
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
        metrics['spectral_centroid'] = float(spectral_centroid)
        
        # Spectral bandwidth
        spectral_variance = np.sum(((freqs - spectral_centroid)**2) * psd) / np.sum(psd)
        metrics['spectral_bandwidth'] = float(np.sqrt(spectral_variance))
        
    except Exception as e:
        # Return default metrics if computation fails
        metrics = {
            'rms': 0.0, 'peak': 0.0, 'kurtosis': 0.0, 'skewness': 0.0,
            'spectral_centroid': 0.0, 'spectral_bandwidth': 0.0
        }
    
    return metrics
