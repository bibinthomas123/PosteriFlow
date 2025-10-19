"""
Signal Injection Module
Handles injection of single and overlapping GW signals with SNR control
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

try:
    from pycbc.filter import matched_filter, sigma
    from pycbc.types import TimeSeries
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False

from .config import SAMPLE_RATE, DURATION, N_SAMPLES
from .waveform_generator import WaveformGenerator

class SignalInjector:
    """
    Inject GW signals into noise with precise SNR control
    Supports overlapping signals for AHSD pipeline
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, duration: float = DURATION):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.logger = logging.getLogger(__name__)
        self.waveform_generator = WaveformGenerator(sample_rate, duration)
    
    def inject_signal(self, 
                     noise: np.ndarray, 
                     params: Dict, 
                     detector_name: str,
                     psd_dict: Dict = None) -> Tuple[np.ndarray, Dict]:
        """
        Inject single signal with target SNR
        
        Returns:
            (injected_data, injection_metadata)
        """
        
        # Generate waveform
        signal = self.waveform_generator.generate_waveform(params, detector_name)
        
        # Ensure correct length
        signal = self._resize_signal(signal, len(noise))
        
        # Scale to target SNR
        target_snr = params.get('target_snr', 15.0)
        scaled_signal, actual_snr = self._scale_to_target_snr(
            signal, noise, target_snr, psd_dict
        )
        
        # Combine
        injected = noise + scaled_signal
        
        # Metadata
        metadata = {
            'detector': detector_name,
            'target_snr': target_snr,
            'actual_snr': actual_snr,
            'signal_peak': float(np.max(np.abs(scaled_signal))),
            'injection_time': params.get('geocent_time', 0.0)
        }
        
        return injected, metadata
    
    def inject_overlapping_signals(self,
                                   noise: np.ndarray,
                                   signal_params_list: List[Dict],
                                   detector_name: str,
                                   psd_dict: Dict = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Inject multiple overlapping signals
        
        Args:
            noise: Background noise
            signal_params_list: List of parameter dicts for each signal
            detector_name: Detector name
            psd_dict: PSD for SNR calculation
            
        Returns:
            (injected_data, list of metadata for each signal)
        """
        
        combined_signal = np.zeros(len(noise), dtype=np.float32)
        metadata_list = []
        
        for i, params in enumerate(signal_params_list):
            # Generate individual waveform
            signal = self.waveform_generator.generate_waveform(params, detector_name)
            signal = self._resize_signal(signal, len(noise))
            
            # Scale to target SNR
            target_snr = params.get('target_snr', 15.0)
            scaled_signal, actual_snr = self._scale_to_target_snr(
                signal, noise, target_snr, psd_dict
            )
            
            # Add time offset for overlap
            time_offset = params.get('time_offset', 0.0)
            if time_offset != 0:
                scaled_signal = self._apply_time_shift(scaled_signal, time_offset)
            
            # Accumulate
            combined_signal += scaled_signal
            
            # Store metadata
            metadata_list.append({
                'signal_index': i,
                'detector': detector_name,
                'target_snr': target_snr,
                'actual_snr': actual_snr,
                'time_offset': time_offset,
                'signal_peak': float(np.max(np.abs(scaled_signal)))
            })
        
        # Combine with noise
        injected = noise + combined_signal
        
        return injected, metadata_list
    
    def _scale_to_target_snr(self,
                            signal: np.ndarray,
                            noise: np.ndarray,
                            target_snr: float,
                            psd_dict: Dict = None) -> Tuple[np.ndarray, float]:
        """
        Scale signal to achieve target SNR
        
        Returns:
            (scaled_signal, actual_snr)
        """
        
        # Try PyCBC optimal SNR calculation
        if PYCBC_AVAILABLE and psd_dict and 'psd' in psd_dict:
            try:
                return self._scale_optimal_snr_pycbc(signal, target_snr, psd_dict)
            except Exception as e:
                self.logger.debug(f"PyCBC SNR scaling failed: {e}")
        
        # Fallback: simple noise-based scaling
        return self._scale_simple_snr(signal, noise, target_snr)
    
    def _scale_optimal_snr_pycbc(self,
                                signal: np.ndarray,
                                target_snr: float,
                                psd_dict: Dict) -> Tuple[np.ndarray, float]:
        """Scale using PyCBC optimal SNR"""
        
        # Convert to PyCBC TimeSeries
        signal_ts = TimeSeries(signal, delta_t=1.0/self.sample_rate)
        psd = psd_dict['psd']
        
        # Calculate optimal SNR
        sig = sigma(signal_ts, psd=psd, low_frequency_cutoff=20.0)
        
        if sig > 0:
            scale_factor = target_snr / sig
        else:
            scale_factor = 1.0
        
        scaled_signal = signal * scale_factor
        actual_snr = float(sig * scale_factor)
        
        return scaled_signal, actual_snr
    
    def _scale_simple_snr(self,
                         signal: np.ndarray,
                         noise: np.ndarray,
                         target_snr: float) -> Tuple[np.ndarray, float]:
        """Simple SNR scaling based on noise RMS"""
        
        # Estimate noise level
        noise_rms = np.std(noise)
        
        # Signal power
        signal_rms = np.sqrt(np.mean(signal**2))
        
        if signal_rms > 0 and noise_rms > 0:
            # Current SNR estimate
            current_snr = signal_rms / noise_rms
            
            # Scale factor
            scale_factor = target_snr / current_snr
        else:
            # Fallback
            scale_factor = target_snr * noise_rms / 1e-21
        
        scaled_signal = signal * scale_factor
        actual_snr = float(target_snr)  # Approximate
        
        return scaled_signal, actual_snr
    
    def _apply_time_shift(self, signal: np.ndarray, time_offset: float) -> np.ndarray:
        """Apply time shift to signal for overlap creation"""
        
        shift_samples = int(time_offset * self.sample_rate)
        
        if shift_samples == 0:
            return signal
        
        shifted = np.zeros_like(signal)
        
        if shift_samples > 0:
            # Shift forward
            if shift_samples < len(signal):
                shifted[shift_samples:] = signal[:-shift_samples]
        else:
            # Shift backward
            abs_shift = abs(shift_samples)
            if abs_shift < len(signal):
                shifted[:-abs_shift] = signal[abs_shift:]
        
        return shifted
    
    def _resize_signal(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Resize signal to match target length"""
        
        if len(signal) == target_length:
            return signal
        elif len(signal) > target_length:
            return signal[:target_length]
        else:
            # Pad with zeros
            padded = np.zeros(target_length, dtype=signal.dtype)
            padded[:len(signal)] = signal
            return padded
    
    def calculate_network_snr(self, 
                             detector_snrs: Dict[str, float]) -> float:
        """Calculate network SNR from individual detector SNRs"""
        
        network_snr_sq = sum(snr**2 for snr in detector_snrs.values())
        return float(np.sqrt(network_snr_sq))
    
    def create_overlapping_scenario(self,
                                   n_signals: int,
                                   snr_range: Tuple[float, float],
                                   overlap_window: float = 0.5) -> List[Dict]:
        """
        Create parameter sets for overlapping scenario
        
        Args:
            n_signals: Number of overlapping signals
            snr_range: (min_snr, max_snr) for individual signals
            overlap_window: Time window for overlaps (seconds)
            
        Returns:
            List of parameter dicts with time offsets
        """
        
        from .parameter_sampler import ParameterSampler
        
        sampler = ParameterSampler()
        signal_params = []
        
        # Base time
        base_time = 0.0
        
        for i in range(n_signals):
            # Sample parameters
            snr_regime = 'medium'  # Can be randomized
            params = sampler.sample_bbh_parameters(snr_regime, is_edge_case=False)
            
            # Random SNR
            params['target_snr'] = np.random.uniform(*snr_range)
            
            # Time offset for overlap
            if i == 0:
                params['time_offset'] = 0.0
                params['geocent_time'] = base_time
            else:
                offset = np.random.uniform(-overlap_window/2, overlap_window/2)
                params['time_offset'] = offset
                params['geocent_time'] = base_time + offset
            
            signal_params.append(params)
        
        return signal_params


    def _compute_optimal_snr(self, waveform, psd_dict):
        """Compute optimal matched-filter SNR"""
        from numpy.fft import rfft, rfftfreq
        
        N = len(waveform)
        dt = 1.0 / self.sample_rate
        freqs = rfftfreq(N, dt)
        wf_fft = rfft(waveform)
        
        psd_freqs = psd_dict['frequencies']
        psd = psd_dict['psd']
        psd_interp = np.interp(freqs, psd_freqs, psd)
        psd_interp = np.maximum(psd_interp, 1e-50)
        
        delta_f = freqs[1] - freqs[0]
        integrand = np.abs(wf_fft)**2 / psd_interp
        
        snr_squared = 4 * np.sum(integrand) * delta_f
        return np.sqrt(snr_squared)
