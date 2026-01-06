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
    Complete GW strain preprocessing pipeline with quality assurance.
    
    This class implements production-ready preprocessing for gravitational wave strain data
    acquired from LIGO/Virgo detectors. The pipeline includes:
    
    - **Bandpass filtering**: Isolate GW signal band (default 20-8192 Hz)
    - **Whitening**: Normalize noise spectrum to unit variance
    - **Edge tapering**: Apply Tukey window to reduce boundary artifacts
    - **Quality validation**: Check for NaN/Inf, saturation, unusual noise levels
    
    **Physics Background**:
    LIGO/Virgo detectors output strain h(t) = [signal + noise] in the gravitational wave
    band. Raw data is strongly colored (noise varies with frequency). Preprocessing normalizes
    the data for subsequent analysis and reduces instrumental artifacts.
    
    **Processing Steps**:
    1. DC offset removal (subtract mean)
    2. Bandpass filtering (isolate GW band, remove low-frequency seismic, high-frequency electronics)
    3. Whitening (divide by PSD in frequency domain, normalize spectrum)
    4. Edge tapering (smooth boundaries with Tukey window to prevent spectral leakage)
    5. Quality validation (detect NaN/Inf, saturation, unusual statistics)
    
    **Key Parameters**:
    - sample_rate: Sampling frequency in Hz (typically 4096 or 16384)
    - duration: Data duration in seconds (typically 1-4s per segment)
    - f_low: Lower bandpass cutoff (default 20 Hz, below BBH band for safety margin)
    - f_high: Upper bandpass cutoff (default 8192 Hz, Nyquist for 16384 Hz sampling)
    
    **Numerical Stability**:
    Whitening can amplify noise at low frequencies where PSD is small. Implementation uses:
    - Clipped PSD minimum (1e-30) to prevent division by zero
    - High-pass filtering after whitening to remove low-frequency artifacts
    - Intermediate NaN/Inf sanitization to detect numerical errors
    - Fallback to original strain on processing failure
    
    **Typical Usage**:
    ```python
    preprocessor = DataPreprocessor(sample_rate=4096, duration=4.0)
    preprocessed_strain = preprocessor.preprocess(raw_strain, psd_dict=psd)
    quality_report = preprocessor.validate_data(preprocessed_strain)
    ```
    
    Attributes:
        sample_rate (int): Sampling frequency in Hz
        duration (float): Data duration in seconds
        f_low (float): Lower bandpass cutoff frequency in Hz
        f_high (float): Upper bandpass cutoff frequency in Hz
        n_samples (int): Total number of samples = sample_rate × duration
        logger: Python logger for warnings and debugging
    """
    
    def __init__(self, 
                 sample_rate: int = SAMPLE_RATE,
                 duration: float = DURATION,
                 f_low: float = F_LOWER,
                 f_high: float = F_UPPER):
        """
        Initialize the data preprocessor with sampling parameters.
        
        Args:
            sample_rate (int, default from config):
                Sampling frequency in Hz. Typical values:
                - 4096 Hz: Standard LIGO/Virgo detector sample rate (2010-2015)
                - 16384 Hz: Enhanced LIGO/Virgo sample rate (O3 onwards)
                - Must be positive integer
                
            duration (float, default from config):
                Data segment duration in seconds. Typical values:
                - 1.0s: Short segments for low-latency analysis
                - 4.0s: Standard GWTC analysis segments (overlap between events)
                - Must be positive
                
            f_low (float, default from config):
                Lower bandpass cutoff frequency in Hz. Typical values:
                - 10-20 Hz: Below BBH signal band (200-1000 Hz)
                - 25-35 Hz: For BNS searches (narrow band)
                - Prevents low-frequency seismic noise and DC drift
                
            f_high (float, default from config):
                Upper bandpass cutoff frequency in Hz. Typical values:
                - 8192 Hz: For 16384 Hz sampling (near Nyquist)
                - 2048 Hz: For 4096 Hz sampling (near Nyquist)
                - Removes high-frequency electronics noise
                
        Returns:
            None (initializes instance)
            
        Side Effects:
            Creates logger for this instance, computes n_samples
            
        Notes:
            - All parameters must be physically reasonable for GW analysis
            - No validation of ranges (assumes valid from config)
            - n_samples = sample_rate × duration should be power-of-2 for FFT efficiency
        """
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
        Execute complete strain preprocessing pipeline.
        
        Applies a sequence of signal processing steps to prepare raw LIGO/Virgo strain
        for GW analysis. Steps are applied in order with safety checks for NaN/Inf at
        each stage. Each step is independently optional for flexibility.
        
        **Processing Order**:
        1. Convert to float64 (preserve precision during processing)
        2. Check for NaN/Inf in input (sanitize if found)
        3. Remove DC offset (subtract mean)
        4. Bandpass filter (optional, isolate GW band)
        5. Whiten (optional, normalize spectrum)
        6. Edge tapering (optional, smooth boundaries)
        7. Final NaN/Inf check (convert to float32 for output)
        
        **Quality Assurance**:
        - Input and output validated for NaN/Inf
        - Processing failures logged as warnings, input returned unchanged
        - Intermediate results sanitized before being passed to next stage
        - Final output cast to float32 for memory efficiency
        
        **Typical Workflow**:
        ```python
        # Full pipeline (all steps enabled)
        clean_strain = preprocessor.preprocess(raw_strain, psd_dict=psd)
        
        # Custom: whitening only (skip bandpass and tapering)
        whitened_only = preprocessor.preprocess(raw_strain, psd_dict=psd,
                                                bandpass=False, remove_edges=False)
        
        # Custom: filtering only (skip whitening)
        filtered_only = preprocessor.preprocess(raw_strain, 
                                                whiten=False)
        ```
        
        Args:
            strain (np.ndarray):
                Raw strain data from detector(s). Shape: (n_samples,) or higher dimensional.
                Expected dtype: float or int (will be converted to float64).
                May contain NaN/Inf (will be sanitized).
                Typical values: ±1e-18 m (LIGO strain units)
                
            psd_dict (Dict, optional):
                Power Spectral Density for whitening. Required if whiten=True.
                Expected keys:
                - 'psd': Array of PSD values, shape (n_freqs,), units [strain^2/Hz]
                - 'frequencies': Array of frequencies, shape (n_freqs,), units [Hz]
                Can also contain PyTorch tensor (will be converted to numpy)
                
            whiten (bool, default True):
                If True, apply frequency-domain whitening using psd_dict.
                Converts colored noise (varying spectrum) to white noise (flat spectrum).
                Requires psd_dict. If False, skips whitening step.
                
            bandpass (bool, default True):
                If True, apply Butterworth bandpass filter using [f_low, f_high].
                Isolates gravitational wave signal band, removes seismic/electronics noise.
                If False, skips bandpass filtering.
                
            remove_edges (bool, default True):
                If True, apply Tukey window tapering to edges (alpha=0.1).
                Reduces spectral leakage from FFT boundary conditions.
                If False, skips tapering (edges kept at full amplitude).
                
        Returns:
            np.ndarray:
                Preprocessed strain data, dtype=float32 for memory efficiency.
                Shape matches input (typically (n_samples,)).
                All values finite (NaN/Inf sanitized).
                Statistics depend on options:
                - With whitening: RMS ~1.0 (normalized)
                - Without whitening: RMS ~1e-19 to 1e-20 (detector units)
                
        Raises:
            None (no exceptions raised, all errors logged as warnings)
            
        Notes:
            - Input not modified (numpy creates copies during processing)
            - Processing can fail gracefully (returns original strain if error)
            - Output shape matches input shape exactly
            - NaN/Inf handling: replaced with 0.0 (conservative), ±1e-6 (extremes)
            - Memory: requires ~2× input size (float64 during processing, float32 output)
            
        Side Effects:
            Logs warnings if NaN/Inf detected, processing errors occur, or fallbacks used
            
        Example:
            >>> import numpy as np
            >>> strain_raw = np.random.normal(0, 1e-19, 16384)
            >>> preprocessor = DataPreprocessor(sample_rate=4096, duration=4.0)
            >>> strain_clean = preprocessor.preprocess(strain_raw, psd_dict=psd, whiten=True)
            >>> assert strain_clean.dtype == np.float32
            >>> assert strain_clean.shape == strain_raw.shape
        """
         
         # Ensure correct dtype
        data = np.array(strain, dtype=np.float64)
        
        # Check for NaN/Inf early
        if np.any(~np.isfinite(data)):
            self.logger.warning("Input strain contains NaN/Inf, sanitizing before preprocessing")
            data = np.nan_to_num(data, nan=0.0, posinf=1e-6, neginf=-1e-6)
        
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
        
        # Final check: if NaN appears after preprocessing, sanitize
        if np.any(~np.isfinite(data)):
            self.logger.warning("Output strain contains NaN/Inf after preprocessing, sanitizing")
            data = np.nan_to_num(data, nan=0.0, posinf=1e-6, neginf=-1e-6)
        
        return data.astype(np.float32)
    
    def whiten_data(self, strain: np.ndarray, psd_dict: Dict) -> np.ndarray:
        """
        Normalize strain using Power Spectral Density (PSD) whitening.
        
        Converts colored noise (frequency-dependent power) to white noise (flat spectrum)
        by dividing in frequency domain by the PSD. Critical preprocessing step for
        gravitational wave searches where many analysis algorithms assume white noise.
        
        **Physics Background**:
        Raw LIGO/Virgo data is strongly colored: power is high at low frequencies
        (seismic noise) and low at high frequencies (thermal noise). Whitening normalizes
        the spectrum so all frequencies have equal power, improving signal detectability
        and parameter estimation accuracy.
        
        **Algorithm**:
        1. FFT strain to frequency domain: h_f = FFT(strain)
        2. Interpolate PSD to match FFT frequency grid
        3. Compute whitening factor: sqrt(psd) in frequency domain
        4. Divide FFT by whitening factor: h_white_f = h_f / sqrt(psd)
        5. Inverse FFT back to time domain
        6. High-pass filter at 10 Hz to remove low-frequency artifacts
        
        **Numerical Stability**:
        - PSD minimum clipped to 1e-30 to prevent division by zero
        - High-pass filter removes noise amplification at frequencies where PSD→0
        - Intermediate NaN/Inf checks preserve numerical integrity
        - Forward-backward filtering preserves phase information
        
        **Key Parameters**:
        - PSD interpolation: Linear spline to match FFT frequency resolution
        - High-pass filter: 4th-order Butterworth at 10 Hz
        - Clipping: Prevents division by zero for extremely small PSD values
        
        Args:
            strain (np.ndarray):
                Input strain data, shape (n_samples,), dtype float (any).
                Expected values: ±1e-18 to ±1e-21 m (LIGO/Virgo strain units).
                May contain DC offset (will be handled).
                
            psd_dict (Dict):
                Power Spectral Density dictionary. Must contain:
                - 'psd': Array of PSD values, shape (n_freqs,), units [strain^2/Hz]
                - 'frequencies': Array of frequencies, shape (n_freqs,), units [Hz]
                Alternatively, can be torch.Tensor (automatically converted to numpy).
                PSD should be computed from quiet data segment (no injections).
                
        Returns:
            np.ndarray:
                Whitened strain, shape (n_samples,), dtype matches input.
                Statistics: RMS ≈ 1.0 (normalized), mean ≈ 0.0 (zero-centered).
                Spectrum: Flat across all frequencies (white).
                Returns original strain if whitening fails.
                
        Raises:
            None (exceptions logged as warnings, graceful fallback to input)
            
        Notes:
            - Whitening always includes high-pass filtering at 10 Hz
            - PSD interpolation uses linear splines (fast, smooth)
            - FFT length matches strain length (no padding/truncation)
            - Forward-backward filtering prevents phase distortion
            - Memory: requires ~2× input size for FFT arrays
            
        Side Effects:
            - Logs warnings if PSD missing/invalid or processing fails
            - Modifies strain in-place during FFT computation (numpy behavior)
            
        Example:
            >>> strain_colored = np.random.normal(0, 1e-19, 16384)
            >>> psd_dict = {'psd': np.ones(8193), 'frequencies': np.linspace(0, 4096, 8193)}
            >>> preprocessor = DataPreprocessor(sample_rate=4096)
            >>> strain_white = preprocessor.whiten_data(strain_colored, psd_dict)
            >>> # Result: flat spectrum, RMS≈1.0, low frequencies suppressed
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
            
            # Avoid division by zero (use larger minimum to avoid numerical issues)
            psd_interp = np.maximum(psd_interp, 1e-30)
            
            # Compute whitening denominator safely
            whitening_denom = np.sqrt(psd_interp * self.sample_rate / 2)
            whitening_denom = np.maximum(whitening_denom, 1e-15)  # Clamp minimum
            
            # Whiten
            whitened_fft = strain_fft / whitening_denom
            
            # Sanitize FFT result before IFFT
            if np.any(~np.isfinite(whitened_fft)):
                self.logger.warning("Whitened FFT contains NaN/Inf, using original FFT")
                whitened_fft = strain_fft
            
            # IFFT
            whitened = np.fft.irfft(whitened_fft, n=len(strain))
            
            # Sanitize time-domain result
            if np.any(~np.isfinite(whitened)):
                self.logger.warning("Whitened time-domain contains NaN/Inf, using original strain")
                return strain
            
            # High-pass to remove low-freq artifacts
            whitened = self.highpass_filter(whitened, 10.0)
            
            # Final sanitization
            if np.any(~np.isfinite(whitened)):
                self.logger.warning("High-pass filtered whitened contains NaN/Inf, using original strain")
                return strain
            
            return whitened
            
        except Exception as e:
            self.logger.warning(f"Whitening failed: {e}")
            return strain
    
    def bandpass_filter(self,
                       strain: np.ndarray,
                       f_low: float,
                       f_high: float,
                       order: int = 8) -> np.ndarray:
        """
        Apply Butterworth bandpass filter to isolate gravitational wave signal band.
        
        Removes out-of-band noise using an infinite impulse response (IIR) Butterworth
        filter. Zero-phase filtering via forward-backward application (filtfilt) ensures
        no phase distortion.
        
        **Filter Design**:
        Butterworth filter (maximally flat passband response):
        - Order: Typically 4-8 (higher order = steeper rolloff, more group delay)
        - Cutoff frequencies: f_low and f_high define passband
        - Rolloff: 20 dB/decade per filter order outside passband
        
        **Physics Motivation**:
        - Low frequencies (< 20 Hz): Seismic noise, instrumental drift
        - Signal band (20-8000 Hz): Gravitational wave signal for most compact objects
        - High frequencies (> 8000 Hz): Quantum noise, RF contamination
        
        **Numerical Stability**:
        - Normalized cutoff frequencies clamped to valid range [1e-6, 0.99]
        - Forward-backward filtering (filtfilt) prevents instability
        - Returns original strain if filter design fails
        
        **Computational Details**:
        - Forward-backward filtering via scipy.signal.filtfilt
        - Eliminates group delay distortion (phase linear)
        - Doubles effective filter order (8 → 16 after bidirectional application)
        - Computational cost: O(N × order) where N = number of samples
        
        Args:
            strain (np.ndarray):
                Input strain data, shape (n_samples,), dtype float (any).
                Expected values: ±1e-18 to ±1e-20 m (detector units).
                
            f_low (float):
                Lower cutoff frequency in Hz. Band gain -3dB at this frequency.
                Typical values: 10-50 Hz (below GW signal band).
                Must be positive and < f_high.
                
            f_high (float):
                Upper cutoff frequency in Hz. Band gain -3dB at this frequency.
                Typical values: 500-8000 Hz (above GW signal band).
                Must be > f_low and < sample_rate/2.
                
            order (int, default 8):
                Filter order (number of poles). Valid range: 1-10 typical.
                Higher order = steeper rolloff but more potential instability.
                Effective order doubled after bidirectional application (8 → 16).
                
        Returns:
            np.ndarray:
                Bandpass filtered strain, shape (n_samples,), dtype matches input.
                Out-of-band frequencies attenuated.
                Phase information preserved (forward-backward filtering).
                Returns original strain unchanged if filter fails.
                
        Raises:
            None (all errors logged as warnings, input returned)
            
        Notes:
            - Uses scipy.signal.butter for filter design
            - Uses scipy.signal.filtfilt for zero-phase application
            - Normalized cutoff = f_cutoff / (sample_rate/2)
            - Valid normalized range: (0, 1)
            - Group delay at cutoff frequency = ~1/(2π×f_cutoff)
            - Forward-backward doubles delay (unavoidable with zero-phase requirement)
            
        Side Effects:
            Logs warnings if filter design fails
            
        Example:
            >>> strain = np.random.normal(0, 1e-19, 16384)
            >>> preprocessor = DataPreprocessor(sample_rate=4096)
            >>> filtered = preprocessor.bandpass_filter(strain, f_low=20, f_high=1000)
            >>> # Signal between 20-1000 Hz preserved, outside attenuated
        """
        
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
        """
        Apply Butterworth high-pass filter to remove low-frequency noise.
        
        Attenuates frequencies below the cutoff while preserving higher frequencies.
        Commonly used to:
        - Remove DC offset and slow drift in detector
        - Eliminate seismic noise after whitening
        - Remove gravitational wave memory effects
        
        **Filter Design**:
        - Butterworth high-pass with maximally flat passband
        - Order: Typically 2-4 (higher = steeper rolloff)
        - Cutoff: Frequencies above this have -3dB gain
        - Rolloff: 20 dB/decade per order
        
        **Use Case - Whitening Cleanup**:
        Whitening can amplify low-frequency noise where PSD is very small. High-pass
        filtering at 10 Hz after whitening removes these artifacts while preserving
        GW signal (which starts ~20 Hz for most sources).
        
        **Numerical Stability**:
        - Normalized cutoff clamped to valid range [0, 1)
        - Forward-backward filtering prevents instability
        - Graceful fallback to original strain on error
        
        Args:
            strain (np.ndarray):
                Input strain data, shape (n_samples,), dtype float (any).
                Expected values: ±1e-18 to ±1e-20 m (detector units).
                
            cutoff (float):
                Cutoff frequency in Hz. Frequencies below this attenuated.
                Typical values:
                - 1-10 Hz: Aggressive low-frequency removal
                - 20-50 Hz: Preserves most GW signal band
                Must be positive and < sample_rate/2.
                
            order (int, default 4):
                Filter order. Higher = steeper rolloff, more stability concerns.
                Typical: 2-4. Effective order doubled after bidirectional application.
                
        Returns:
            np.ndarray:
                High-pass filtered strain, shape (n_samples,), dtype matches input.
                Frequencies below cutoff attenuated.
                Phase preserved (forward-backward filtering).
                Returns original strain if filter fails.
                
        Raises:
            None (all errors logged as warnings, input returned)
            
        Notes:
            - Uses scipy.signal.butter with btype='high'
            - Uses scipy.signal.filtfilt for zero-phase response
            - Normalized cutoff = cutoff / (sample_rate/2)
            - Typical use: highpass_filter(whitened, cutoff=10.0) to clean whitening artifacts
            
        Side Effects:
            Logs warning if filter design fails
            
        Example:
            >>> strain_with_drift = np.random.normal(0, 1e-19, 16384) + 1e-16*np.linspace(-1, 1, 16384)
            >>> preprocessor = DataPreprocessor(sample_rate=4096)
            >>> cleaned = preprocessor.highpass_filter(strain_with_drift, cutoff=5.0)
            >>> # Low-frequency drift removed, high-frequency signal preserved
        """
        
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
        """
        Apply Tukey (tapered cosine) window to reduce spectral leakage.
        
        Smoothly tapers the edges of the strain data to zero while keeping the center
        unmodified. Reduces spectral leakage caused by abrupt boundaries in FFT.
        
        **Why Taper Edges?**
        When applying FFT, signal is implicitly assumed periodic. Abrupt edges create
        discontinuities → spectral leakage (power "smears" across frequencies). Tapering
        smoothly zeros the edges, eliminating discontinuities while minimizing side lobe
        power.
        
        **Tukey Window Design**:
        - Center (1-alpha) fraction: Amplitude = 1.0 (unmodified)
        - Edges (alpha/2 fraction each): Amplitude tapers from 0 → 1 via raised cosine
        - Typical alpha: 0.05-0.25 (trade-off between edge smoothing and center preservation)
        - alpha=0: Rectangular window (no tapering)
        - alpha=1: Hann window (full cosine taper)
        
        **Physics Impact**:
        - Reduces spectral leakage at GW frequencies
        - Improves SNR for long-duration signals
        - Minimal impact if signal contained well within center (1-alpha) fraction
        - Edge artifacts suppressed below noise floor
        
        **Numerical Stability**:
        - Simple element-wise multiplication (very stable)
        - Returns original strain unchanged if windowing fails
        
        Args:
            strain (np.ndarray):
                Input strain data, shape (n_samples,), dtype float (any).
                Edge samples will be progressively suppressed.
                
            alpha (float, default 0.1):
                Taper parameter controlling fraction of samples windowed.
                Valid range: 0.0 (no taper) to 1.0 (full Hann taper)
                Typical: 0.05-0.25
                - 0.1: 10% of edges tapered (5% each side)
                - 0.25: 25% of edges tapered (12.5% each side)
                
        Returns:
            np.ndarray:
                Windowed strain, shape (n_samples,), dtype matches input.
                Center: Unmodified (1-alpha fraction)
                Edges: Smoothly tapered to zero (alpha/2 fraction each side)
                Returns original strain if windowing fails.
                
        Raises:
            None (exceptions caught, original strain returned)
            
        Notes:
            - Implementation uses scipy.signal.windows.tukey
            - Operation: output = strain × window(alpha)
            - Low-power operation (no FFT, one multiplication per sample)
            - Typically used last in preprocessing pipeline before analysis
            - Reduces energy by factor (1 - alpha/2) due to tapering
            
        Side Effects:
            None (no exceptions raised, silent fallback if error)
            
        Example:
            >>> strain = np.ones(1000)  # Rectangular signal
            >>> preprocessor = DataPreprocessor()
            >>> windowed = preprocessor.apply_tukey_window(strain, alpha=0.1)
            >>> # windowed[0:50] tapers from 0 to 1 (5% of 1000)
            >>> # windowed[50:950] = 1.0 (unmodified)
            >>> # windowed[950:1000] tapers from 1 to 0 (5% of 1000)
        """
        
        try:
            window = windows.tukey(len(strain), alpha)
            return strain * window
        except:
            return strain
    
    def validate_data(self, strain: np.ndarray) -> Dict:
        """
        Comprehensive data quality validation and statistical assessment.
        
        Performs multi-level validation of preprocessed strain data to detect
        numerical errors, instrumental artifacts, and processing failures. Returns
        detailed metrics for quality assurance and debugging.
        
        **Validation Checks**:
        1. **Finiteness**: All values must be finite (no NaN/Inf)
        2. **Shape**: Length must match expected n_samples (sample_rate × duration)
        3. **Statistics**: Mean, std, RMS, max, min within detector-typical ranges
        4. **Saturation**: Peak amplitude (max) below LIGO saturation threshold
        5. **Noise Level**: Standard deviation in expected range for detector
        
        **Quality Thresholds** (LIGO/Virgo typical):
        - Max amplitude: < 1e-18 m (saturation usually occurs near 1e-17)
        - Std dev: 1e-25 to 1e-20 m (whitened data ~1.0, raw data ~1e-20)
        - Finite fraction: 1.0 (all values finite, no NaN/Inf)
        - Mean: ~0.0 (DC offset removed)
        
        **Use Cases**:
        - Post-processing validation: Check preprocessor output quality
        - Batch processing: Monitor data quality before neural network inference
        - Debugging: Identify stages where processing fails (NaN introduction, etc.)
        - Data quality reporting: Generate quality metrics for event candidates
        
        **Return Structure**:
        ```python
        {
            'passed': bool,           # True if all critical checks pass
            'warnings': [str, ...],   # List of detected issues (non-fatal)
            'metrics': {
                'length': int,        # Number of samples
                'mean': float,        # Mean amplitude
                'std': float,         # Standard deviation (noise level)
                'max': float,         # Maximum amplitude
                'min': float,         # Minimum amplitude
                'rms': float,         # Root mean square
                'finite_fraction': float  # Fraction of finite values (0.0-1.0)
            }
        }
        ```
        
        Args:
            strain (np.ndarray):
                Strain data to validate, shape (n_samples,), dtype any (coercible to float).
                Typically preprocessed output from preprocess() method.
                May contain various data qualities (used for quality assessment).
                
        Returns:
            Dict:
                Validation report with structure detailed above:
                - 'passed' (bool): True if all critical checks satisfied
                - 'warnings' (list): Non-fatal issues detected (never raises exception)
                - 'metrics' (dict): Statistical measures for quality assessment
                
                Typical metrics for different data types:
                - Raw strain: std~1e-20, max~1e-18, finite_fraction=1.0
                - Whitened strain: std~1.0, max~10-50, finite_fraction=1.0
                - Failed processing: finite_fraction<1.0, std out of range, warnings populated
                
        Raises:
            None (all failures logged as warnings, graceful return)
            
        Notes:
            - All checks are non-fatal (exceptions never raised)
            - Warnings are informative (do not prevent downstream processing)
            - Metrics always computed regardless of pass/fail status
            - Used for monitoring, not for filtering (returns data anyway)
            - Saturation threshold 1e-18 m typical for LIGO, varies by detector
            
        Side Effects:
            None (read-only validation, no modifications)
            
        Example:
            >>> preprocessor = DataPreprocessor(sample_rate=4096, duration=4.0)
            >>> strain = np.random.normal(0, 1e-20, 16384)  # synthetic
            >>> report = preprocessor.validate_data(strain)
            >>> if not report['passed']:
            ...     print(f"Data quality issues: {report['warnings']}")
            >>> print(f"Noise level (std): {report['metrics']['std']:.2e}")
            
        Production Usage:
            ```python
            # Monitor batch processing
            for batch in data_loader:
                strain = preprocess_batch(batch)
                report = preprocessor.validate_data(strain)
                if not report['passed']:
                    logger.warning(f"Batch {i} issues: {report['warnings']}")
                    # Decide: skip batch, flag for manual review, or continue
            ```
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
        Compute time-frequency spectrogram (power spectrogram) for analysis and visualization.
        
        Generates a 2D time-frequency representation of strain data using overlapping
        windowed Fourier transforms. The spectrogram reveals how frequency content changes
        over time, essential for visualizing gravitational wave signals and identifying
        instrumental artifacts.
        
        **Physics Application**:
        Gravitational wave signals from inspiraling compact objects exhibit "chirps": the
        frequency increases over time as the objects spiral inward. The spectrogram clearly
        shows this frequency evolution, enabling:
        - **Signal detection**: Identify characteristic chirp patterns
        - **Signal localization**: Determine start/end times and frequency range
        - **Artifact identification**: Distinguish instrumental lines from astrophysical signal
        - **Quality assessment**: Detect glitches, gaps, or unusual noise behavior
        
        **Algorithm**:
        1. Partition strain into overlapping segments (default: 50% overlap)
        2. Apply Hann window to each segment (reduces spectral leakage)
        3. Compute FFT for each windowed segment
        4. Compute power (magnitude-squared FFT)
        5. Return frequency grid, time grid, and power matrix
        
        **Time-Frequency Resolution Trade-off**:
        - **Frequency resolution**: Δf = sample_rate / nperseg (Hz)
        - **Time resolution**: Δt = nperseg / sample_rate (seconds)
        - Longer segments (large nperseg): Better frequency resolution, worse time localization
        - Shorter segments (small nperseg): Better time localization, worse frequency resolution
        - Default (nperseg = sample_rate/4): Balanced for typical GW signals
        
        **Default Parameters**:
        - nperseg: sample_rate // 4 (0.25s segments at 4096 Hz = 1024 samples)
        - Window: Hann (automatic in scipy.signal.spectrogram)
        - Overlap: 50% (noverlap = nperseg // 2)
        - Resolution: Δf ≈ 4 Hz, Δt ≈ 0.125s (at 4096 Hz sample rate)
        
        **Output Interpretation**:
        ```
        f ∈ [0, sample_rate/2]          # Frequency array (Hz)
        t ∈ [0, duration]               # Time array (seconds)
        Sxx ∈ [nperseg/2, len(f)]       # Power array (strain^2/Hz)
        
        To plot: import matplotlib
                 plt.pcolormesh(t, f, 10*np.log10(Sxx))  # dB scale
                 plt.ylabel('Frequency (Hz)')
                 plt.xlabel('Time (s)')
                 plt.colorbar(label='Power (dB)')
        ```
        
        **Typical Usage**:
        - **Visualization**: Inspect preprocessed data quality
        - **Signal hunting**: Find chirps, transients, or unusual artifacts
        - **Vetting**: Determine if signal candidate is astrophysical or instrumental
        - **Debugging**: Compare before/after preprocessing to diagnose issues
        
        Args:
            strain (np.ndarray):
                Input strain data, shape (n_samples,), dtype float (any).
                Typically preprocessed output from preprocess() method.
                Expected values: ±1.0 (whitened) or ±1e-20 (raw), units strain (dimensionless).
                May contain signal, noise, or artifacts.
                
            nperseg (int, optional):
                Length of each FFT window in samples. Controls time-frequency resolution.
                Default: sample_rate // 4 (0.25 seconds of data per FFT window)
                Typical values:
                - sample_rate // 2: Better frequency resolution, coarser time resolution
                - sample_rate // 4: Balanced (recommended)
                - sample_rate // 8: Better time localization, noisier spectrum
                Must be positive. Larger values → better frequency resolution, worse time localization.
                
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                Three-element tuple:
                
                1. frequencies (np.ndarray):
                   Frequency array, shape (n_freqs,), units Hz.
                   Range: [0, sample_rate/2] (up to Nyquist frequency).
                   Spacing: sample_rate / nperseg (uniform).
                   
                2. times (np.ndarray):
                   Time array, shape (n_times,), units seconds.
                   Range: [0, duration] (approximately, depends on window overlap).
                   Spacing: nperseg / (2 × sample_rate) (50% overlap).
                   
                3. spectrogram_power (np.ndarray):
                   Power spectrogram, shape (n_freqs, n_times), units strain^2/Hz.
                   Rows: Frequencies (0 to Nyquist)
                   Columns: Time windows (0 to duration)
                   Values: Positive (power is non-negative)
                   Typical range: 1e-50 (noise floor) to 1e-40 (strong signal)
                   
        Raises:
            None (scipy.signal.spectrogram handles all edge cases)
            
        Notes:
            - Window function: Hann window (automatic, scipy default)
            - Overlap: 50% (noverlap = nperseg // 2)
            - FFT length: Same as nperseg (no zero-padding)
            - Normalization: Power per unit frequency (PSD style)
            - Returns raw power (linear scale), user should log-scale for dB visualization
            - Computation: O(n_times × nperseg × log(nperseg)) with FFT
            
        Side Effects:
            None (read-only, no modifications to input or state)
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
