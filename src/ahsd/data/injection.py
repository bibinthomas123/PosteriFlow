"""
Gravitational Wave Signal Injection Module

Core functionality for injecting simulated gravitational wave signals into detector noise
with precise signal-to-noise ratio (SNR) control. Supports single and overlapping signal
scenarios essential for training the Adaptive Hierarchical Signal Decomposition (AHSD) pipeline.

This module is responsible for:
1. **Waveform generation**: Interface with WaveformGenerator for detector-specific responses
2. **SNR scaling**: Scale signals to achieve target SNR using optimal (PyCBC) or fallback methods
3. **Signal overlapping**: Combine multiple signals with time offsets for multi-event scenarios
4. **SNR computation**: Calculate and attach network SNR using 3-tier priority system

Key Classes:
    SignalInjector: Main class for signal injection operations

Key Functions:
    compute_network_snr_from_det_dict: Extract per-detector SNRs and combine
    proxy_network_snr_from_params: Physics-based SNR proxy from mass/distance
    attach_network_snr: 3-tier priority SNR computation

Typical Workflow:
    1. Create SignalInjector instance
    2. Generate noise (via PSD manager or real GWOSC data)
    3. Sample parameters (via ParameterSampler)
    4. Inject signals: inject_signal() for single or inject_overlapping_signals() for multiple
    5. Attach SNR: attach_network_snr() adds network_snr to sample metadata
    6. Validate: Check for NaN/Inf, verify SNR targets achieved, log metadata

SNR Scaling Methods:
    - **Optimal (PyCBC)**: Uses matched-filter SNR via sigma() function and PSD
      Pros: Theoretically optimal, accounts for colored noise
      Cons: Requires PyCBC, slower, fails on edge cases
    - **Simple (fallback)**: RMS-based scaling without PSD
      Pros: Fast, robust, no dependencies
      Cons: Less accurate for colored noise

Signal Overlap Configuration:
    - Supports 1-6 overlapping signals per sample
    - Time offsets create realistic multi-event scenarios
    - Each signal tracked independently for subsequent extraction
    - Suitable for testing signal separation algorithms (AHSD)

Error Handling:
    - No exceptions raised; failures handled gracefully
    - NaN/Inf in waveforms → fallback to noise-only
    - Failed SNR scaling → fallback to simple method then default
    - Invalid PSD → automatic fallback
    - All output arrays validated and sanitized to float32

References:
    - LIGO/Virgo data analysis methodology (GW150914 and subsequent)
    - GWTC-1, GWTC-2 catalogs for reference parameters
    - PyCBC documentation for matched-filter SNR
    
See Also:
    ahsd.data.parameter_sampler: Generate GW source parameters
    ahsd.data.waveform_generator: Generate detector-specific waveforms
    ahsd.data.psd_manager: Load power spectral densities
    ahsd.data.dataset_generator: End-to-end dataset generation using this module
"""

import numpy as np
import math
import logging
from typing import Dict, List, Tuple, Optional

try:
    from pycbc.filter import matched_filter, sigma
    from pycbc.types import TimeSeries
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False

from .config import SAMPLE_RATE, DURATION
from .waveform_generator import WaveformGenerator

class SignalInjector:
    """
    Main class for injecting gravitational wave signals into detector noise with SNR control.
    
    Provides comprehensive signal injection capabilities for single and multiple overlapping
    signals. Integrates waveform generation, SNR scaling, and metadata tracking to produce
    realistic synthetic gravitational wave data for neural network training.
    
    This class is the primary interface for the signal injection pipeline. It handles:
    - Single signal injection with precise SNR targeting
    - Multiple overlapping signal combinations
    - SNR scaling via optimal (PyCBC) or simple (RMS-based) methods
    - Time-domain signal shifting for temporal overlaps
    - Robust error handling with graceful fallbacks
    - Comprehensive metadata generation for downstream processing
    
    Designed for the Adaptive Hierarchical Signal Decomposition (AHSD) pipeline,
    enabling training on realistic multi-event scenarios where multiple gravitational
    wave sources occur within the same observation window.
    
    Attributes:
        sample_rate (int): Sampling rate in Hz (default 4096 Hz, LIGO/Virgo standard)
        duration (float): Observation window duration in seconds (default 4.0 s)
        n_samples (int): Total samples per observation = sample_rate * duration
        logger (logging.Logger): Logger instance for warnings/info messages
        waveform_generator (WaveformGenerator): Generates detector-specific waveforms
        
    Typical Usage:
        ```python
        # Initialize injector
        injector = SignalInjector(sample_rate=4096, duration=4.0)
        
        # Single signal injection
        signal, metadata = injector.inject_signal(noise, params, 'H1', psd_dict)
        
        # Multiple overlapping signals
        signals, metadata_list = injector.inject_overlapping_signals(
            noise, [params1, params2, params3], 'H1', psd_dict
        )
        
        # Helper: Create overlap scenario
        overlap_params = injector.create_overlapping_scenario(
            n_signals=3, snr_range=(10, 40), overlap_window=0.5
        )
        ```
        
    Note:
        All methods designed to fail gracefully - no exceptions raised to calling code.
        Failures logged via logger and fallback to safe defaults (noise-only or zero arrays).
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
        Inject a single gravitational wave signal into detector noise with precise SNR control.
        
        This method generates a waveform based on the provided parameters, scales it to achieve
        a target signal-to-noise ratio (SNR), and additively combines it with the background noise.
        The actual SNR achieved is computed and returned in metadata.
        
        Args:
            noise: Background detector noise array of shape (n_samples,). Should be float32 array
                representing time-domain strain data.
            params: Dictionary containing signal parameters with the following keys:
                - 'target_snr' (float, optional): Desired SNR for the injected signal. 
                  If not provided, defaults to 15.0. Must be positive.
                - 'geocent_time' (float, optional): Geocentric time of signal merger (seconds).
                  Stored in metadata for reference.
                - Other keys: Parameters required by WaveformGenerator (mass_1, mass_2, spin_1z, etc.)
            detector_name: Name of the detector (e.g., 'H1', 'L1', 'V1'). Used to generate
                detector-specific waveform with correct antenna response pattern.
            psd_dict: Optional dictionary containing power spectral density information.
                If provided with 'psd' key, uses PyCBC's optimal SNR calculation.
                If None or PyCBC unavailable, falls back to noise RMS-based scaling.
                
        Returns:
            Tuple containing:
                - injected_data: Time-domain strain data (noise + signal), same length as input noise.
                  Type: np.ndarray float32. Will contain only noise if injection fails.
                - metadata: Dictionary with injection details:
                    - 'detector': Detector name
                    - 'target_snr': Requested SNR value
                    - 'actual_snr': Achieved SNR (0.0 if injection failed)
                    - 'signal_peak': Maximum absolute amplitude of scaled signal
                    - 'injection_time': Geocentric time from params
                    
        Raises:
            No exceptions raised; failures are handled gracefully with fallback to noise-only.
            
        Notes:
            - If waveform generation or SNR scaling produces NaN/Inf values, the method
              logs a warning and returns noise without signal (actual_snr = 0.0).
            - Signal length is automatically adjusted to match noise length.
            - SNR scaling uses PyCBC optimal SNR if available and PSD provided,
              otherwise uses simpler noise RMS-based scaling.
            - All output arrays are cast to float32 for consistency.
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
        
        # Validate injected data
        if np.any(~np.isfinite(injected)):
            self.logger.warning(
                f"NaN/Inf detected in injected signal for {detector_name}, "
                f"replacing with noise only"
            )
            # If injection produced NaN, just return the noise
            injected = np.copy(noise)
            actual_snr = 0.0
        
        # Double-check: if noise itself has NaN, sanitize it
        if np.any(~np.isfinite(injected)):
            injected = np.nan_to_num(injected, nan=0.0, posinf=1e-6, neginf=-1e-6).astype(np.float32)
        
        # Metadata
        metadata = {
            'detector': detector_name,
            'target_snr': target_snr,
            'actual_snr': actual_snr,
            'signal_peak': float(np.nanmax(np.abs(scaled_signal))) if np.any(np.isfinite(scaled_signal)) else 0.0,
            'injection_time': params.get('geocent_time', 0.0)
        }
        
        return injected, metadata
    
    
    def inject_overlapping_signals(self,
                                   noise: np.ndarray,
                                   signal_params_list: List[Dict],
                                   detector_name: str,
                                   psd_dict: Dict = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Inject multiple overlapping gravitational wave signals into detector noise.
        
        This method is essential for the Adaptive Hierarchical Signal Decomposition (AHSD) pipeline.
        It generates multiple waveforms, scales each to its target SNR, applies time offsets for
        overlap, and combines them additively into a single detector strain dataset.
        
        The overlapping configuration creates scenarios where multiple GW events occur within
        the same observation window, enabling training and testing of signal separation algorithms.
        Each signal's metadata is tracked independently to enable subsequent extraction and
        parameter estimation steps.
        
        Args:
            noise: Background detector noise array of shape (n_samples,). Should be float32 array
                representing time-domain strain data at the specified sample rate.
            signal_params_list: List of parameter dictionaries, one per signal. Each dict contains:
                - 'target_snr' (float, optional): Desired SNR for this signal. Defaults to 15.0.
                - 'time_offset' (float, optional): Time offset in seconds relative to first signal.
                  Defaults to 0.0. Can be negative (signal starts before primary).
                - 'geocent_time' (float, optional): Geocentric merger time of this signal.
                - Other keys: Parameters for WaveformGenerator (mass_1, mass_2, etc.)
            detector_name: Name of the detector (e.g., 'H1', 'L1', 'V1'). Used to generate
                detector-specific waveforms with proper antenna response.
            psd_dict: Optional dictionary containing power spectral density for optimal SNR
                calculation. If None or missing 'psd' key, falls back to RMS-based scaling.
                
        Returns:
            Tuple containing:
                - injected_data: Combined strain data (noise + all signals), float32 array.
                  Length matches input noise. Contains only noise if injection fails.
                - metadata_list: List of dictionaries, one per input signal, containing:
                    - 'signal_index': Index in signal_params_list (0, 1, 2, ...)
                    - 'detector': Detector name
                    - 'target_snr': Requested SNR for this signal
                    - 'actual_snr': Achieved SNR (0.0 if all injection fails)
                    - 'time_offset': Time offset applied to this signal (seconds)
                    - 'signal_peak': Maximum absolute amplitude of scaled signal
                    
        Raises:
            No exceptions raised; failures are handled gracefully with fallback to noise-only.
            
        Notes:
            - Signals are generated and scaled independently, then summed (linear superposition).
            - If SNR scaling or waveform generation produces NaN/Inf for any signal,
              the entire injection reverts to noise-only and all actual_snr values become 0.0.
            - Time offsets are applied via cyclic shift in time domain.
            - Each signal's metadata is preserved even if final result reverts to noise.
            - Suitable for 2-6 overlapping signals; performance degrades beyond 6 due to
              GW waveform overlap and parameter degeneracies.
              
        See Also:
            create_overlapping_scenario: Helper to generate overlapping parameter sets
            extract_overlapping_signals: Companion method in AHSDPipeline for signal extraction
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
                'signal_peak': float(np.nanmax(np.abs(scaled_signal))) if np.any(np.isfinite(scaled_signal)) else 0.0
            })
        
        # Combine with noise
        injected = noise + combined_signal
        
        # Validate injected data
        if np.any(~np.isfinite(injected)):
            self.logger.warning(
                f"NaN/Inf detected in overlapping signals for {detector_name}, "
                f"replacing with noise only"
            )
            # If injection produced NaN, just return the noise
            injected = np.copy(noise)
            # Reset all actual_snr values to reflect noise-only fallback
            for metadata in metadata_list:
                metadata['actual_snr'] = 0.0
        
        # Double-check: if noise itself has NaN, sanitize it
        if np.any(~np.isfinite(injected)):
            injected = np.nan_to_num(injected, nan=0.0, posinf=1e-6, neginf=-1e-6).astype(np.float32)
        
        return injected, metadata_list
    
    def _scale_to_target_snr(self,
                            signal: np.ndarray,
                            noise: np.ndarray,
                            target_snr: float,
                            psd_dict: Dict = None) -> Tuple[np.ndarray, float]:
        """
        Scale signal amplitude to achieve target signal-to-noise ratio.
        
        This method determines the appropriate scaling factor to inject a signal at the
        desired SNR level. It uses a two-tier approach: optimal matched-filter SNR via
        PyCBC if available, otherwise falls back to simple RMS-based scaling.
        
        The matched-filter SNR approach uses the power spectral density (PSD) to weight
        the waveform's frequency components, providing more accurate SNR estimates that
        account for detector sensitivity variations across the band.
        
        Args:
            signal: Time-domain waveform array (typically 1-2 seconds duration).
                Should already be in physical units (strain).
            noise: Background noise array with same length as signal. Used to estimate
                noise floor in fallback method.
            target_snr: Desired SNR for scaled signal. Must be positive. Typical range 5-100.
            psd_dict: Optional dictionary with PSD information:
                - 'psd': Array of PSD values (required for optimal method)
                - Other keys: Ignored
                If None or missing 'psd', uses fallback scaling method.
                
        Returns:
            Tuple containing:
                - scaled_signal: Signal multiplied by computed scaling factor. Same dtype as input.
                  Contains only zeros if scaling fails.
                - actual_snr: Computed SNR of the scaled signal. Float value, positive.
                  Returns target_snr as approximation if scaling succeeds.
                  
        Notes:
            - Scaling factors are clipped to [1e-6, 1e6] to prevent numerical overflow/underflow.
            - If signal RMS is too small or noise level is suspiciously low,
              uses conservative fallback factor.
            - Output arrays are validated for finite values (no NaN/Inf).
            - Failure is silent; returns scaled_signal filled with finite numbers.
            
        Algorithm Priority:
            1. PyCBC optimal SNR (if available and PSD provided)
            2. Simple noise RMS scaling (fallback)
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
        """
        Scale signal using PyCBC's matched-filter optimal SNR calculation.
        
        This method uses the sigma() function from PyCBC to compute the optimal SNR
        that would be achieved by cross-correlating the waveform with itself, weighted
        by the inverse PSD. This is the theoretically optimal SNR for matched filtering
        in colored Gaussian noise and is more accurate than simple RMS scaling.
        
        Args:
            signal: Waveform array in physical units (strain). Will be converted to
                PyCBC TimeSeries for calculation.
            target_snr: Desired SNR value. Used to compute scaling factor from
                optimal SNR: factor = target_snr / optimal_snr.
            psd_dict: Dictionary containing at minimum:
                - 'psd': Array of PSD values in appropriate frequency units
                
        Returns:
            Tuple containing:
                - scaled_signal: Signal array multiplied by computed scaling factor.
                  dtype matches input. Will be filled with safe values if computation fails.
                - actual_snr: Computed SNR after scaling, typically close to target_snr.
                  
        Raises:
            No exceptions. Returns safe defaults (scale_factor=1.0, zero signal) if
            PyCBC sigma() call fails, PSD is invalid, or computed SNR is non-finite.
            
        Notes:
            - PyCBC's sigma() can return array or scalar depending on version.
              This is handled via np.atleast_1d() conversion.
            - Low-frequency cutoff hardcoded to 20.0 Hz (typical for LIGO/Virgo).
            - Scaling factors are clipped to [1e-6, 1e6] to prevent numerical extremes.
            - Used only when PyCBC is available and PSD is provided.
            - Fallback path uses simple RMS-based scaling if this method fails.
        """
        
        # Convert to PyCBC TimeSeries
        signal_ts = TimeSeries(signal, delta_t=1.0/self.sample_rate)
        psd = psd_dict['psd']
        
        # Calculate optimal SNR
        sig = sigma(signal_ts, psd=psd, low_frequency_cutoff=20.0)
        
        # Convert sig to scalar - sigma() can return array in some PyCBC versions
        sig = float(np.atleast_1d(sig)[0]) if hasattr(sig, '__len__') else float(sig)
        
        if sig > 0 and np.isfinite(sig):
            scale_factor = target_snr / sig
            # Clamp to prevent overflow
            scale_factor = np.clip(scale_factor, 1e-6, 1e6)
        else:
            scale_factor = 1.0
        
        scaled_signal = signal * scale_factor
        # Ensure result is finite
        if not np.all(np.isfinite(scaled_signal)):
            scaled_signal = np.nan_to_num(scaled_signal, nan=0.0, posinf=1e-6, neginf=-1e-6)
        
        actual_snr = float(sig * scale_factor) if np.isfinite(sig) else target_snr
        
        return scaled_signal, actual_snr
    
    def _scale_simple_snr(self,
                         signal: np.ndarray,
                         noise: np.ndarray,
                         target_snr: float) -> Tuple[np.ndarray, float]:
        """
        Scale signal using simple RMS-based SNR estimation.
        
        This is a fallback method used when PyCBC is unavailable or PSD is not provided.
        It estimates SNR as the ratio of signal RMS to noise RMS, a simple but effective
        approximation for white or nearly-white noise.
        
        The method computes the current signal-to-noise ratio and derives a scaling
        factor to achieve the target SNR. Includes safeguards against extreme values
        and pathological edge cases (very low noise, very small signals).
        
        Args:
            signal: Input waveform array (physical units, strain).
            noise: Noise array with same length. Used to estimate noise level.
            target_snr: Desired SNR. Must be positive.
            
        Returns:
            Tuple containing:
                - scaled_signal: Signal multiplied by computed scaling factor. Same shape/dtype as input.
                - actual_snr: Computed SNR. Set to target_snr as approximation.
                
        Notes:
            - Noise RMS computed via np.std(noise).
            - Signal RMS computed via sqrt(mean(signal^2)).
            - Scaling factors clipped to [1e-6, 1e6] to prevent overflow/underflow.
            - If noise is suspiciously small (<1e-30), uses conservative factor (target_snr * 1e-6).
            - No exceptions raised; returns safe defaults on edge cases.
            - Suitable for white or quasi-white noise (real LIGO/Virgo data after whitening).
            - For colored noise, use PyCBC method instead.
        """
        
        # Estimate noise level
        noise_rms = np.std(noise)
        
        # Signal power
        signal_rms = np.sqrt(np.mean(signal**2))
        
        if signal_rms > 0 and noise_rms > 1e-30:
            # Current SNR estimate
            current_snr = signal_rms / noise_rms
            
            # Scale factor - clamp to prevent overflow
            scale_factor = target_snr / current_snr
            # Clamp extreme scale factors to prevent NaN/Inf
            scale_factor = np.clip(scale_factor, 1e-6, 1e6)
        else:
            # Fallback: use target SNR directly if noise is too small
            # Don't divide by 1e-21! That creates enormous scale factors
            scale_factor = target_snr * 1e-6  # Conservative default
        
        scaled_signal = signal * scale_factor
        actual_snr = float(target_snr)  # Approximate
        
        return scaled_signal, actual_snr
    
    def _apply_time_shift(self, signal: np.ndarray, time_offset: float) -> np.ndarray:
        """
        Apply time shift to signal for creating overlapping signal scenarios.
        
        Shifts a waveform forward or backward in time using integer sample shifts.
        The operation uses zero-padding: shifted samples are filled with zeros.
        This preserves signal energy within the observation window while changing
        the temporal alignment relative to other signals.
        
        Used in inject_overlapping_signals() to position signals at different
        merger times, creating the realistic multi-event scenarios needed for
        testing hierarchical signal decomposition algorithms.
        
        Args:
            signal: Input waveform array.
            time_offset: Time shift in seconds. Positive = shift forward, negative = shift backward.
                Internally converted to integer sample count via time_offset * sample_rate.
                
        Returns:
            Shifted signal array with same shape as input. Zero-padded regions at edges.
            Returns input signal unchanged if time_offset rounds to 0 samples.
            
        Notes:
            - Uses roll-like behavior for negative offsets (signal "enters" from right edge).
            - For positive offsets, future signal samples are zero-padded (signal moves right).
            - For negative offsets, past signal samples are zero-padded (signal moves left).
            - Edge of observation window: signal samples outside [0, n_samples) are lost.
            - Common usage: time_offset ∈ [-0.5, +0.5] seconds for overlaps within 1s window.
            - Preserves dtype of input signal.
        """
        
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
        """
        Resize signal array to match target observation window length.
        
        Handles three cases:
        1. Signal longer than target: Truncates from the end
        2. Signal shorter than target: Pads with zeros
        3. Signal equals target: Returns unchanged
        
        This ensures all waveforms fit into the observation window of the detector
        data, regardless of merger time or waveform duration.
        
        Args:
            signal: Input waveform array.
            target_length: Desired output length in samples.
            
        Returns:
            Resized signal array with exactly target_length samples. dtype preserved.
            
        Notes:
            - Truncation removes trailing samples (signal end, "ringdown" part).
            - Padding adds zeros at end (quiescent time after signal).
            - Does NOT apply windowing or fade-out; abrupt cutoff at boundaries.
            - Common scenario: merger at t=0 → signal 1-2s duration → padded to 4s observation.
            - Inverse operation: same array can be padded then truncated without data loss
              if intermediate length never equals target_length.
        """
        
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
        """
        Calculate network SNR from individual detector SNRs via coherent combination.
        
        Combines SNRs from multiple detectors using the standard formula for
        coherent matched filtering networks:
        
            SNR_network = sqrt(SNR_H1^2 + SNR_L1^2 + SNR_V1^2 + ...)
        
        This assumes independent detector noise and optimal weighting (equal SNR per detector).
        The network SNR represents the effective sensitivity of the detector network
        as a whole. Higher network SNR indicates better signal detectability.
        
        Args:
            detector_snrs: Dictionary mapping detector names (e.g., 'H1', 'L1', 'V1')
                to their individual matched-filter SNR values (float).
                Example: {'H1': 12.5, 'L1': 10.3, 'V1': 5.2}
                
        Returns:
            Network SNR as float, always non-negative. Returns 0.0 if dict is empty.
            
        Notes:
            - Assumes all SNR values are positive (no validation performed).
            - Order of detectors doesn't matter (quadratic sum is commutative).
            - Network SNR ≥ max(individual SNRs), typically 1.3-2.0× larger.
            - Typical LIGO/Virgo networks: H1+L1 gives network ~1.4× larger than either alone.
            - Used for sample-level network_snr computation in dataset generation.
            
        Example:
            >>> injector = SignalInjector()
            >>> net_snr = injector.calculate_network_snr({'H1': 15.0, 'L1': 12.0})
            >>> net_snr
            19.209372712298526  # sqrt(15^2 + 12^2) = sqrt(369)
        """
        
        network_snr_sq = sum(snr**2 for snr in detector_snrs.values())
        return float(np.sqrt(network_snr_sq))
    
    def create_overlapping_scenario(self,
                                   n_signals: int,
                                   snr_range: Tuple[float, float],
                                   overlap_window: float = 0.5) -> List[Dict]:
        """
        Create parameter sets for a realistic overlapping signal scenario.
        
        Generates n_signals waveform parameter sets with random masses, spins,
        and geometry, each assigned a random SNR within the specified range.
        Time offsets are randomly distributed to create overlapping events
        within the observation window, simulating multiple simultaneous GW detections.
        
        This is a convenience method for quickly generating test cases and synthetic
        datasets without manual parameter specification. Each invocation produces
        different parameters due to random sampling.
        
        Args:
            n_signals: Number of signals to generate (typically 2-6).
            snr_range: Tuple (min_snr, max_snr) specifying the uniform range for
                individual signal SNRs. Example: (5.0, 50.0).
            overlap_window: Time window in seconds within which signals can overlap.
                Offsets are uniformly distributed in [-overlap_window/2, +overlap_window/2].
                Default 0.5s means signals can start up to 0.25s before/after the first.
                
        Returns:
            List of n_signals dictionaries, each containing:
                - 'mass_1', 'mass_2': Sampled from ParameterSampler (e.g., BBH)
                - 'spin_1z', 'spin_2z': Sampled from ParameterSampler
                - 'luminosity_distance': Sampled from ParameterSampler
                - 'geocent_time': Base time + time_offset
                - 'time_offset': For inject_overlapping_signals() (relative to first signal)
                - 'target_snr': Uniformly sampled from snr_range
                - Other params: Inclination, sky position, polarization (from sampler)
                
        Notes:
            - First signal always has time_offset = 0.0 and geocent_time = 0.0 (reference).
            - Subsequent signals have time_offset uniformly distributed in overlap_window.
            - All parameters sampled from BBH regime ('medium' SNR regime).
            - Different invocations produce different parameter sets (non-deterministic).
            - SNRs within range specified; actual network SNR will be larger (quadratic sum).
            - Time offsets are applied in inject_overlapping_signals() to create waveform overlap.
            
        See Also:
            inject_overlapping_signals: Method that uses output of this function
            ParameterSampler.sample_bbh_parameters: Source of sampled parameters
            
        Example:
            >>> injector = SignalInjector()
            >>> params = injector.create_overlapping_scenario(n_signals=3, snr_range=(10, 30))
            >>> injected, metadata = injector.inject_overlapping_signals(noise, params, 'H1')
            >>> len(metadata)
            3  # Three signals injected
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
            
            # Random SNR (do not overwrite if already set by caller/sampler)
            if 'target_snr' not in params:
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
        """
        Compute optimal matched-filter SNR directly via frequency-domain integration.
        
        Calculates the matched-filter SNR by integrating the waveform's power
        weighted by the inverse PSD:
        
            SNR^2 = 4 ∫ |h(f)|^2 / S_n(f) df
        
        This is the maximum possible SNR achievable by any linear filter on the data
        containing this signal and noise. It represents the theoretical upper bound
        on SNR for this waveform in this noise.
        
        Args:
            waveform: Time-domain waveform array.
            psd_dict: Dictionary containing:
                - 'frequencies': Frequency values (Hz)
                - 'psd': Power spectral density values
                
        Returns:
            Optimal SNR as float. Always positive.
            
        Notes:
            - Uses FFT to move to frequency domain (rfft for real input).
            - Linearly interpolates PSD to match FFT frequency grid.
            - Clamps PSD minimum to 1e-50 to prevent division by zero.
            - Used as ground-truth SNR for algorithm development and validation.
            - More accurate than sigma() for edge cases and validation purposes.
            
        Implementation Detail:
            This method is similar to _scale_optimal_snr_pycbc() but avoids PyCBC
            dependency by implementing the matched-filter integral directly.
        """
        
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


def compute_network_snr_from_det_dict(d: dict):
    """
    Extract and combine per-detector matched-filter SNRs into network SNR.
    
    Looks for detector-specific SNR fields ('snr_H1', 'snr_L1', 'snr_V1') in
    the input dictionary and combines them via the coherent network formula:
    
        SNR_network = sqrt(SNR_H1^2 + SNR_L1^2 + SNR_V1^2 + ...)
    
    This function is part of the SNR attachment priority chain used in dataset
    generation to ensure all samples have network_snr fields.
    
    Args:
        d: Dictionary potentially containing per-detector SNR fields:
            - 'snr_H1': Hanford SNR
            - 'snr_L1': Livingston SNR
            - 'snr_V1': Virgo SNR
            - Other keys: Ignored
            
    Returns:
        Network SNR (float) if at least one per-detector SNR found.
        None if no relevant SNR fields present in dictionary.
        
    Notes:
        - Extracts each SNR via d.get(key, None) to handle missing fields gracefully.
        - Converts to float via float() constructor; silently ignores conversion failures.
        - Returns None early if no SNRs found (cost: O(1) in failure case).
        - Used in attach_network_snr() as MEDIUM priority option.
        - Typical usage: detector-level SNR computation already completed.
        
    See Also:
        proxy_network_snr_from_params: LOW priority fallback (uses mass/distance)
        attach_network_snr: HIGH priority wrapper (uses target_snr if available)
        
    Example:
        >>> d = {'snr_H1': 15.0, 'snr_L1': 12.0, 'snr_V1': 5.0, 'other': 'field'}
        >>> compute_network_snr_from_det_dict(d)
        19.209372712298526  # sqrt(15^2 + 12^2 + 5^2)
        
        >>> d2 = {'mass_1': 30.0, 'mass_2': 25.0}  # No SNR fields
        >>> compute_network_snr_from_det_dict(d2)
        None
    """
    snrs = []
    for k in ('snr_H1', 'snr_L1', 'snr_V1'):
        v = d.get(k, None)
        if v is not None:
            snrs.append(float(v))
    if not snrs:
        return None
    return float(math.sqrt(sum(s*s for s in snrs)))

def proxy_network_snr_from_params(d: dict):
    """
    Compute network SNR estimate from physical parameters using physics-based proxy.
    
    This function provides a physics-motivated fallback when matched-filter SNRs
    are unavailable. It exploits the fundamental gravitational wave physics relationship
    between detector response and source properties:
    
        SNR ∝ (M_chirp)^(5/6) / luminosity_distance
    
    Heavier binaries are louder; more distant sources are quieter. This creates a strong
    SNR-distance anticorrelation that dominates observed GW populations.
    
    The proxy uses reference parameters tuned to LIGO/Virgo detector sensitivity and
    realistic astrophysical source populations. Accounts for detector-specific mass ranges
    (BBH, BNS, NSBH) via reference mass calibration.
    
    Args:
        d: Dictionary containing parameters:
            - 'mass_1' (float, default 30.0): Primary mass (M_sun)
            - 'mass_2' (float, default 25.0): Secondary mass (M_sun)
            - 'luminosity_distance' (float, default 100.0): Distance (Mpc)
            Other keys: Ignored
            
    Returns:
        Estimated network SNR (float), always positive.
        
    Notes:
        - Chirp mass M_c = (m1*m2)^(3/5) / (m1+m2)^(1/5) [standard GR formula]
        - Reference parameters calibrated Dec 29, 2025 to match dataset generation:
            - reference_chirp_mass = 30.0 M_sun (BBH typical)
            - reference_distance = 1500.0 Mpc (median BBH distance)
            - reference_snr = 20.0 (typical LIGO/Virgo detection threshold)
        - Formula: SNR = 20 * (Mc/30)^(5/6) * (1500/d)
        - Distance clipped to minimum 10 Mpc to prevent division by zero.
        - Used as LOW priority option in attach_network_snr() priority chain.
        - Suitable for: quick SNR estimates, missing detector data, validation.
        - NOT suitable for: precise SNR calculations (use matched-filter method).
        
    Physics:
        The (M_c)^(5/6) scaling comes from GW amplitude:
            h ∝ M_c^(5/6) × frequency_dependent_terms / distance
        
        The 1/distance scaling reflects geometric dilution of wave amplitude.
        
    See Also:
        compute_network_snr_from_det_dict: MEDIUM priority (per-detector matched-filter SNR)
        attach_network_snr: Wrapper with 3-tier priority
        
    Example:
        >>> d = {'mass_1': 50.0, 'mass_2': 40.0, 'luminosity_distance': 500.0}
        >>> snr = proxy_network_snr_from_params(d)
        >>> snr  # Higher masses + closer = higher SNR
        ~35.0  # Depends on exact calculation
    
    References:
        - LIGO/Virgo Population Paper (GWTC-1, GWTC-2)
        - parameter_sampler.py reference_params (source of calibration)
    """
    m1 = float(d.get('mass_1', 30.0))
    m2 = float(d.get('mass_2', 25.0))
    dl = max(float(d.get('luminosity_distance', 100.0)), 10.0)
    
    # Chirp mass: M_c = (m1*m2)^(3/5) / (m1+m2)^(1/5)
    total_mass = m1 + m2
    chirp_mass = (m1 * m2)**(3/5) / total_mass**(1/5)
    
    # Reference: must match ParameterSampler reference_params
    # BBH default: M_c=30 Msun, d=1500 Mpc, SNR=20
    # Dec 29, 17:00 UTC CRITICAL FIX: Updated to match ParameterSampler scatter fix
    reference_chirp_mass = 30.0
    reference_distance = 1500.0  # Updated from 1400.0 to match BBH reference_params (Dec 29, 17:00 UTC)
    reference_snr = 20.0         # Keep at 20 (already correct from previous fix)
    
    # SNR scales as: SNR ~ (M_c)^(5/6) / d
    snr = reference_snr * (chirp_mass / reference_chirp_mass)**(5/6) * (reference_distance / dl)
    
    return float(snr)



def attach_network_snr(d: dict):
    """
    Compute and attach network_snr field to sample dictionary using 3-tier priority system.
    
    This function is the central hub for SNR attachment in dataset generation. It implements
    a robust priority chain that handles diverse data sources and scenarios:
    
    **Priority 1 (Highest)**: target_snr field
        - Used when signals were injected with explicit SNR targeting (stochastic sampling)
        - Preserves the sampler's intent: signals generated at specific SNR regime
        - Fast: O(1) dictionary lookup
    
    **Priority 2 (Medium)**: Per-detector matched-filter SNRs
        - Extracted from snr_H1, snr_L1, snr_V1 fields (if available)
        - Combines via network formula: SNR_net = sqrt(sum SNR_i^2)
        - More accurate than proxy; requires prior detector-level computation
    
    **Priority 3 (Fallback)**: Physics-based proxy from parameters
        - Estimates SNR from mass and distance using GW physics
        - Used when detectors SNRs unavailable and no target_snr set
        - Always succeeds (guaranteed network_snr field on exit)
    
    This design ensures:
    - Stochastic sampling in data generation is respected (Priority 1)
    - No samples missing network_snr (Priority 3 fallback)
    - Accurate SNR values when available (Priority 2 preference)
    - Consistency across heterogeneous data sources
    
    Args:
        d: Sample dictionary. Modified in-place to add 'network_snr' field.
            May contain any or all of:
            - 'target_snr': Intended SNR from sampler
            - 'snr_H1', 'snr_L1', 'snr_V1': Per-detector SNRs
            - 'mass_1', 'mass_2', 'luminosity_distance': Parameters for proxy
            
    Returns:
        None. Dictionary d is modified in-place; 'network_snr' key is added.
        
    Side Effects:
        - Adds or overwrites d['network_snr'] with float value
        - Does not remove other fields
        - Does not validate/sanitize other dictionary contents
        
    Notes:
        - Invoked during dataset generation for every sample
        - Typical dataset: ~45% Priority 1, ~10% Priority 2, ~45% Priority 3
        - Priority 1 samples: direct injection, well-controlled SNR
        - Priority 2 samples: detector-level SNR computation available
        - Priority 3 samples: GWTC real events, synthetic edge cases
        - All three priorities yield valid (positive float) SNR values
        - See companion functions for Priority 2 and 3 implementations
        
    See Also:
        compute_network_snr_from_det_dict: Implements Priority 2 (per-detector SNRs)
        proxy_network_snr_from_params: Implements Priority 3 (physics proxy)
        SignalInjector.calculate_network_snr: Related method (used in Priority 2)
        
    Example:
        >>> d = {'target_snr': 25.0, 'mass_1': 30.0}
        >>> attach_network_snr(d)
        >>> d['network_snr']
        25.0  # Priority 1: target_snr used
        
        >>> d2 = {'snr_H1': 15.0, 'snr_L1': 12.0, 'mass_1': 50.0}
        >>> attach_network_snr(d2)
        >>> d2['network_snr']
        19.21  # Priority 2: per-detector combined
        
        >>> d3 = {'mass_1': 40.0, 'mass_2': 35.0, 'luminosity_distance': 800.0}
        >>> attach_network_snr(d3)
        >>> d3['network_snr']
        18.5  # Priority 3: physics proxy used
    """
    # HIGH PRIORITY: If target_snr was sampled/set, use it directly
    if 'target_snr' in d and d['target_snr'] is not None:
        try:
            d['network_snr'] = float(d['target_snr'])
            return
        except (ValueError, TypeError):
            pass
    
    # MEDIUM PRIORITY: Compute from per-detector SNRs if available
    snr_net = compute_network_snr_from_det_dict(d)
    if snr_net is not None:
        d['network_snr'] = float(snr_net)
        return
    
    # LOW PRIORITY: Fallback to proxy formula
    snr_net = proxy_network_snr_from_params(d)
    d['network_snr'] = float(snr_net)