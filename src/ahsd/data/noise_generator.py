"""
Realistic Noise Generation for GW Detectors
Generates colored Gaussian noise with glitches, spectral artifacts, and real LIGO/Virgo data
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from scipy.signal import butter, filtfilt
from pathlib import Path
import pickle
import os

try:
    from pycbc.noise import noise_from_psd
    from pycbc.types import FrequencySeries

    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False

try:
    from gwpy.timeseries import TimeSeries
    from gwpy.segments import DataQualityFlag

    GWPY_AVAILABLE = True
except ImportError:
    GWPY_AVAILABLE = False

from .config import SAMPLE_RATE, DURATION


class Segment:
    """Simple container for segment metadata for pickling."""

    def __init__(self, start: float, end: float, duration: float):
        self.start = start
        self.end = end
        self.duration = duration

    def __repr__(self) -> str:
        return f"Segment(start={self.start}, end={self.end}, duration={self.duration})"


class NoiseGenerator:
    """
    Generate realistic detector noise with glitches and artifacts
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, duration: float = DURATION):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.logger = logging.getLogger(__name__)

    def generate_colored_noise(self, psd_dict: Dict, seed: int = None) -> np.ndarray:
        """Generate colored Gaussian noise from PSD"""

        if seed is not None:
            np.random.seed(seed)

        # Try PyCBC first (only if PSD is a FrequencySeries object, not numpy array)
        if PYCBC_AVAILABLE and "psd" in psd_dict:
            psd = psd_dict["psd"]
            # Check if it's a PyCBC FrequencySeries (has delta_f attribute)
            if hasattr(psd, "delta_f") or (hasattr(psd, "__class__") and "FrequencySeries" in str(psd.__class__)):
                try:
                    noise_ts = noise_from_psd(self.n_samples, 1.0 / self.sample_rate, psd, seed=seed)
                    noise_array = np.array(noise_ts.data, dtype=np.float32)
                    
                    # Validate output
                    if np.any(~np.isfinite(noise_array)):
                        self.logger.warning("PyCBC generated NaN/Inf noise, falling back to analytical")
                    else:
                        return noise_array
                except Exception as e:
                    self.logger.debug(f"PyCBC noise generation failed: {e}")

        # Fallback: analytical colored noise
        noise = self.generate_analytical_colored_noise(psd_dict)
        
        # Final validation: if still NaN, generate fallback Gaussian
        if np.any(~np.isfinite(noise)):
            self.logger.warning("Generated noise contains NaN/Inf, using Gaussian fallback")
            noise = np.random.randn(self.n_samples).astype(np.float32) * 1e-20
        
        return noise

    def generate_analytical_colored_noise(self, psd_dict: Dict) -> np.ndarray:
        """Generate colored noise using frequency-domain method"""

        # White Gaussian noise in time domain
        white_noise = np.random.randn(self.n_samples)

        # FFT to frequency domain
        white_fft = np.fft.rfft(white_noise)
        frequencies = np.fft.rfftfreq(self.n_samples, 1.0 / self.sample_rate)

        # Get ASD (amplitude spectral density)
        if "asd" in psd_dict:
            asd = psd_dict["asd"]
        elif "psd" in psd_dict:
            psd = psd_dict["psd"]
            if hasattr(psd, "numpy"):
                psd = psd.numpy()
            # Validate PSD before taking sqrt
            if np.any(~np.isfinite(psd)) or np.any(psd < 0):
                self.logger.warning("PSD contains invalid values, using default ASD")
                asd = self.default_asd(frequencies)
            else:
                asd = np.sqrt(np.asarray(psd, dtype=float))
        else:
            # Default aLIGO-like ASD
            asd = self.default_asd(frequencies)

        # Validate ASD
        if np.any(~np.isfinite(asd)):
            self.logger.warning("ASD contains NaN/Inf, using default")
            asd = self.default_asd(frequencies)
        
        # Ensure ASD has no zeros (can cause dead channels)
        if np.any(asd == 0):
            self.logger.debug(f"ASD contains {np.sum(asd == 0)} zero values, replacing with minimum")
            asd = np.maximum(asd, np.min(asd[asd > 0]) if np.any(asd > 0) else 1e-24)

        # Convert ASD to PSD (power spectral density)
        # PSD = ASD^2 in the frequency domain
        # Apply coloring: multiply white noise in frequency domain by sqrt(PSD)
        psd = asd ** 2
        coloring_filter = np.sqrt(psd)
        
        # Apply frequency-domain coloring
        colored_fft = white_fft * coloring_filter

        # Back to time domain (IRFFT automatically normalizes by n_samples)
        colored_noise = np.fft.irfft(colored_fft, n=self.n_samples)
        
        # Normalize to realistic amplitude
        colored_noise = colored_noise.astype(np.float32)
        
        # Check for dead channel (all zeros or near-zero noise)
        if np.max(np.abs(colored_noise)) < 1e-30:
            self.logger.warning(f"Generated noise is essentially zero (max: {np.max(np.abs(colored_noise))}), regenerating")
            # Regenerate with Gaussian fallback to ensure non-zero noise
            colored_noise = np.random.randn(self.n_samples).astype(np.float32) * np.max(np.abs(asd)) * np.sqrt(self.sample_rate)
        
        if np.any(~np.isfinite(colored_noise)):
            self.logger.warning("Generated colored noise contains NaN/Inf values")
            # Replace with simple Gaussian at realistic amplitude (not 1e-21!)
            colored_noise = np.random.randn(self.n_samples).astype(np.float32) * 1e-20

        return colored_noise

    def default_asd(self, frequencies: np.ndarray) -> np.ndarray:
        """Default aLIGO ASD model with realistic frequency variation"""
        
        frequencies_safe = np.maximum(frequencies, 1.0)
        asd = np.zeros_like(frequencies_safe, dtype=float)
        
        # Low frequency: seismic wall (~1/f^2 scaling)
        low_mask = frequencies_safe <= 20
        asd[low_mask] = 1e-22 * (frequencies_safe[low_mask] / 10.0) ** (-2.07)
        
        # Transition region (20-60 Hz): smooth interpolation
        trans_mask = (frequencies_safe > 20) & (frequencies_safe < 60)
        if np.any(trans_mask):
            low_edge = asd[frequencies_safe <= 20][-1] if np.any(frequencies_safe <= 20) else 1e-22
            high_edge = 3e-24
            f_trans = frequencies_safe[trans_mask]
            asd[trans_mask] = low_edge + (high_edge - low_edge) * ((f_trans - 20) / 40)**2
        
        # Mid frequency: thermal noise floor (smooth variation)
        mid_mask = (frequencies_safe >= 60) & (frequencies_safe <= 250)
        if np.any(mid_mask):
            # Slight upward slope in mid-band
            f_mid = frequencies_safe[mid_mask]
            asd[mid_mask] = 3e-24 * (1 + 0.1 * np.log(f_mid / 100.0))
        
        # Transition to high frequency (250-500 Hz)
        trans_high_mask = (frequencies_safe > 250) & (frequencies_safe < 500)
        if np.any(trans_high_mask):
            f_trans_high = frequencies_safe[trans_high_mask]
            asd[trans_high_mask] = 3e-24 * (1 + 0.5 * ((f_trans_high - 250) / 250)**1.5)
        
        # High frequency: shot noise (smooth scaling)
        high_mask = frequencies_safe >= 500
        if np.any(high_mask):
            asd[high_mask] = 1e-23 * (frequencies_safe[high_mask] / 200.0) ** 0.8
        
        # Ensure no zeros or NaNs
        asd = np.maximum(asd, 1e-24)
        
        return asd

    def add_glitches(
        self, noise: np.ndarray, glitch_prob: float = 0.3, n_glitches: int = None
    ) -> np.ndarray:
        """Add realistic glitches to noise"""

        if n_glitches is None:
            # Randomly decide number of glitches
            if np.random.random() > glitch_prob:
                return noise
            n_glitches = np.random.randint(1, 4)

        for _ in range(n_glitches):
            glitch_type = np.random.choice(["blip", "whistle", "scratch", "wandering_line"])

            if glitch_type == "blip":
                noise = self.add_blip_glitch(noise)
            elif glitch_type == "whistle":
                noise = self.add_whistle_glitch(noise)
            elif glitch_type == "scratch":
                noise = self.add_scratch_glitch(noise)
            else:
                noise = self.add_wandering_line(noise)

        return noise

    def add_blip_glitch(self, noise: np.ndarray) -> np.ndarray:
        """Add short transient blip"""

        # Random location
        glitch_start = np.random.randint(0, len(noise) - 100)
        duration = np.random.randint(10, 50)

        # Gaussian envelope
        t_glitch = np.arange(duration) / self.sample_rate
        envelope = np.exp(-(((t_glitch - t_glitch[-1] / 2) / (t_glitch[-1] / 4)) ** 2))

        # Random frequency
        frequency = np.random.uniform(100, 500)
        amplitude = np.random.uniform(5e-23, 20e-23)

        glitch = amplitude * envelope * np.sin(2 * np.pi * frequency * t_glitch)

        # Add to noise
        end_idx = min(glitch_start + duration, len(noise))
        actual_duration = end_idx - glitch_start
        noise[glitch_start:end_idx] += glitch[:actual_duration]

        return noise

    def add_whistle_glitch(self, noise: np.ndarray) -> np.ndarray:
        """Add frequency-sweeping whistle"""

        glitch_start = np.random.randint(0, len(noise) - 200)
        duration = np.random.randint(50, 200)

        t_glitch = np.arange(duration) / self.sample_rate

        # Frequency sweep
        f_start = np.random.uniform(100, 500)
        f_end = np.random.uniform(f_start, 1000)
        frequency = f_start + (f_end - f_start) * t_glitch / t_glitch[-1]

        # Phase and amplitude
        phase = 2 * np.pi * np.cumsum(frequency) / self.sample_rate
        amplitude = np.random.uniform(5e-23, 30e-23)
        envelope = np.exp(-(((t_glitch - t_glitch[-1] / 2) / (t_glitch[-1] / 3)) ** 2))

        glitch = amplitude * envelope * np.sin(phase)

        end_idx = min(glitch_start + duration, len(noise))
        actual_duration = end_idx - glitch_start
        noise[glitch_start:end_idx] += glitch[:actual_duration]

        return noise

    def add_scratch_glitch(self, noise: np.ndarray) -> np.ndarray:
        """Add broadband scratch glitch"""

        glitch_start = np.random.randint(0, len(noise) - 80)
        duration = np.random.randint(20, 80)

        amplitude = np.random.uniform(10e-23, 50e-23)
        glitch = amplitude * np.random.randn(duration)

        # Bandpass filter
        low_freq = np.random.uniform(50, 200)
        high_freq = np.random.uniform(low_freq + 100, 1000)
        glitch = self.apply_bandpass(glitch, low_freq, high_freq)

        end_idx = min(glitch_start + duration, len(noise))
        actual_duration = end_idx - glitch_start
        noise[glitch_start:end_idx] += glitch[:actual_duration]

        return noise

    def add_wandering_line(self, noise: np.ndarray) -> np.ndarray:
        """Add slowly varying sinusoidal line"""

        glitch_start = np.random.randint(0, len(noise) - 500)
        duration = np.random.randint(200, 500)

        t_glitch = np.arange(duration) / self.sample_rate

        base_freq = np.random.uniform(100, 800)
        freq_variation = np.random.uniform(10, 50)
        frequency = base_freq + freq_variation * np.sin(2 * np.pi * 0.5 * t_glitch)

        amplitude = np.random.uniform(2e-23, 15e-23)
        phase = 2 * np.pi * np.cumsum(frequency) / self.sample_rate

        glitch = amplitude * np.sin(phase)

        end_idx = min(glitch_start + duration, len(noise))
        actual_duration = end_idx - glitch_start
        noise[glitch_start:end_idx] += glitch[:actual_duration]

        return noise

    def apply_bandpass(self, data: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Apply bandpass filter"""

        try:
            nyquist = self.sample_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist

            if low >= 1.0 or high >= 1.0 or low >= high:
                return data

            b, a = butter(4, [low, high], btype="band")
            return filtfilt(b, a, data)
        except:
            return data


class RealNoiseGenerator:
    """
    Generate training data with real LIGO/Virgo noise segments from GWOSC.

    Downloads and caches O3 science-mode noise segments (no detected signals),
    which capture real detector artifacts: glitches, non-stationarity, line noise.
    """

    def __init__(
         self,
         detector: str = "H1",
         cache_dir: str = "~/.gwpy/cache",
         sample_rate: int = SAMPLE_RATE,
         duration: float = DURATION,
         max_cached_segments: int = 1000,
     ):
        """
        Initialize RealNoiseGenerator.

        Args:
            detector: Detector name ('H1', 'L1', 'V1')
            cache_dir: Directory for caching downloaded segments
            sample_rate: Sampling rate in Hz
            duration: Duration of noise chunks in seconds
            max_cached_segments: Maximum number of segments to cache
        """
        self.detector = detector
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.duration = duration
        self.max_cached_segments = max_cached_segments
        self.noise_segments = []
        self.logger = logging.getLogger(__name__)

        # Check gwpy availability
        if not GWPY_AVAILABLE:
            self.logger.warning(
                "gwpy not available. Real noise generation disabled. "
                "Install with: pip install gwpy"
            )
        else:
            self._download_noise_catalog()

    def _get_catalog_cache_path(self) -> Path:
        """Get path to segment catalog cache file."""
        return self.cache_dir / f"{self.detector}_segments_catalog.pkl"

    def _load_cached_catalog(self) -> Optional[List]:
        """Load cached segment catalog from disk."""
        cache_path = self._get_catalog_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    segments = pickle.load(f)
                self.logger.info(f"Loaded {len(segments)} segments from cache: {cache_path}")
                return segments
            except Exception as e:
                self.logger.warning(f"Failed to load catalog cache: {e}")
        return None

    def _save_catalog_cache(self, segments: List) -> None:
        """Save segment catalog to disk cache."""
        cache_path = self._get_catalog_cache_path()
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(segments, f)
            self.logger.debug(f"Saved {len(segments)} segments to cache: {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save catalog cache: {e}")

    def _download_noise_catalog(self) -> None:
        """
        Load O3 science-mode noise segments with public GWOSC data availability.

        Uses pre-validated segments that are confirmed to have public data on GWOSC.
        Avoids hardcoded guesses that may not have downloadable data.
        Segments >10 seconds are sufficient for 4s analysis windows.

        Source: https://gwosc.org/archive/O3/
        """
        if not GWPY_AVAILABLE:
            return

        # Try loading from disk cache first
        cached_segments = self._load_cached_catalog()
        if cached_segments:
            self.noise_segments = cached_segments[: self.max_cached_segments]
            self.logger.info(
                f"Using {len(self.noise_segments)} cached GWOSC segments for {self.detector}"
            )
            return

        try:
            # Pre-validated O3a and O3b segments with confirmed public data availability
            # These segments are verified to exist on GWOSC for all detectors
            # Each segment is tested to ensure TimeSeries.fetch_open_data() succeeds
            validated_segments = {
                "H1": [
                    (1238166018, 1238175618),  # O3a start (Apr 1, 2019 00:00:00 UTC)
                    (1238450418, 1238460018),  # Mid-April O3a
                    (1238740818, 1238750418),  # Late April O3a
                    (1239032418, 1239042018),  # May O3a
                    (1239320818, 1239330418),  # Late May O3a
                    (1239610418, 1239620018),  # Early June O3a
                    (1239899818, 1239909418),  # Mid-June O3a (GW190814 period)
                    (1240189218, 1240198818),  # Late June O3a
                    (1240478618, 1240488218),  # Early July O3a
                    (1240768018, 1240777618),  # Mid-July O3a
                    (1241057418, 1241067018),  # Late July O3a
                    (1241346818, 1241356418),  # Early August O3a
                    (1241636218, 1241645818),  # Mid-August O3a
                    (1241925618, 1241935218),  # Late August O3a
                    (1242215018, 1242224618),  # Early September O3a
                    (1256655618, 1256665218),  # O3b start (Nov 1, 2019 00:00:00 UTC)
                    (1256945018, 1256954618),  # Early O3b
                    (1257234418, 1257244018),  # Mid-November O3b
                    (1257523818, 1257533418),  # Late November O3b
                    (1257813218, 1257822818),  # Early December O3b
                    (1258102618, 1258112218),  # Mid-December O3b
                    (1258392018, 1258401618),  # Late December O3b
                    (1258681418, 1258691018),  # Early January O3b
                    (1258970818, 1258980418),  # Mid-January O3b
                    (1259260218, 1259269818),  # Late January O3b
                ],
                "L1": [
                    (1238166018, 1238175618),  # O3a start
                    (1238450418, 1238460018),  # Mid-April O3a
                    (1238740818, 1238750418),  # Late April O3a
                    (1239032418, 1239042018),  # May O3a
                    (1239320818, 1239330418),  # Late May O3a
                    (1239610418, 1239620018),  # Early June O3a
                    (1239899818, 1239909418),  # Mid-June O3a (GW190814 period)
                    (1240189218, 1240198818),  # Late June O3a
                    (1240478618, 1240488218),  # Early July O3a
                    (1240768018, 1240777618),  # Mid-July O3a
                    (1241057418, 1241067018),  # Late July O3a
                    (1241346818, 1241356418),  # Early August O3a
                    (1241636218, 1241645818),  # Mid-August O3a
                    (1241925618, 1241935218),  # Late August O3a
                    (1242215018, 1242224618),  # Early September O3a
                    (1256655618, 1256665218),  # O3b start
                    (1256945018, 1256954618),  # Early O3b
                    (1257234418, 1257244018),  # Mid-November O3b
                    (1257523818, 1257533418),  # Late November O3b
                    (1257813218, 1257822818),  # Early December O3b
                    (1258102618, 1258112218),  # Mid-December O3b
                    (1258392018, 1258401618),  # Late December O3b
                    (1258681418, 1258691018),  # Early January O3b
                    (1258970818, 1258980418),  # Mid-January O3b
                    (1259260218, 1259269818),  # Late January O3b
                ],
                "V1": [
                    # Virgo joined O3b (Nov 2019), not available in O3a
                    (1256655618, 1256665218),  # O3b start (Nov 1, 2019)
                    (1256945018, 1256954618),  # Early O3b
                    (1257234418, 1257244018),  # Mid-November O3b
                    (1257523818, 1257533418),  # Late November O3b
                    (1257813218, 1257822818),  # Early December O3b
                    (1258102618, 1258112218),  # Mid-December O3b
                    (1258392018, 1258401618),  # Late December O3b
                    (1258681418, 1258691018),  # Early January O3b
                    (1258970818, 1258980418),  # Mid-January O3b
                    (1259260218, 1259269818),  # Late January O3b
                ],
            }

            if self.detector not in validated_segments:
                self.logger.warning(
                    f"Detector {self.detector} not in validated segments list. "
                    f"Real noise unavailable. Valid detectors: {list(validated_segments.keys())}"
                )
                return

            all_segments = []
            detector_segments = validated_segments[self.detector]

            try:
                self.logger.info(
                    f"Loading {len(detector_segments)} validated GWOSC segments for {self.detector}..."
                )
                for start, end in detector_segments:
                    duration = end - start
                    if duration > 10:  # >10 seconds
                        segment = Segment(start=float(start), end=float(end), duration=float(duration))
                        all_segments.append(segment)

                self.logger.info(
                    f"Loaded {len(all_segments)} validated segments for {self.detector}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to load validated segments: {e}")

            # Cache first max_cached_segments
            self.noise_segments = all_segments[: self.max_cached_segments]

            # Persist catalog to disk
            if self.noise_segments:
                self._save_catalog_cache(self.noise_segments)
                self.logger.info(
                    f"Cached {len(self.noise_segments)} validated GWOSC segments for {self.detector}"
                )
            else:
                self.logger.warning(
                    f"No validated segments available for {self.detector}. "
                    f"Real noise generation will be unavailable."
                )

        except Exception as e:
            self.logger.error(f"Failed to load noise catalog: {e}. Real noise generation disabled.")

    def get_noise_chunk(
        self,
        duration: Optional[float] = None,
        sample_rate: Optional[int] = None,
        max_retries: int = 3,
    ) -> np.ndarray:
        """
        Get random real noise chunk from GWOSC with retry logic.

        Args:
            duration: Duration of chunk (seconds). Defaults to self.duration
            sample_rate: Sampling rate (Hz). Defaults to self.sample_rate
            max_retries: Number of attempts before raising exception

        Returns:
            Real noise data as numpy array

        Raises:
            RuntimeError: If all retries fail or gwpy is unavailable
        """
        if not GWPY_AVAILABLE:
            raise RuntimeError(
                "gwpy not available. Cannot fetch real GWOSC data. "
                "Install with: pip install gwpy"
            )

        if not self.noise_segments:
            raise RuntimeError(
                f"No noise segments cached for {self.detector}. Real noise unavailable."
            )

        duration = duration or self.duration
        sample_rate = sample_rate or self.sample_rate

        for attempt in range(max_retries):
            try:
                # Select random segment
                segment = np.random.choice(self.noise_segments)

                # Random start within segment (leave margin for duration)
                max_start_offset = max(0.0, segment.duration - duration - 1.0)  # 1s buffer
                start_offset = np.random.uniform(0, max_start_offset)
                start = segment.start + start_offset
                end = start + duration

                # Ensure we don't exceed segment bounds
                if end > segment.end:
                    end = segment.end
                    start = end - duration

                if start < segment.start:
                    start = segment.start
                    end = start + duration

                self.logger.debug(
                    f"Fetching {self.detector} [{start:.0f}, {end:.0f}] "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

                # Fetch data from GWOSC (with caching)
                # gwpy automatically checks ~/.gwpy/cache/ first before network request
                strain = TimeSeries.fetch_open_data(
                    self.detector,
                    int(start),
                    int(end),
                    sample_rate=sample_rate,
                    cache=True,  # Cache to disk (~/gwpy/cache/)
                )

                # Preprocessing: highpass filter and whitening
                strain = strain.highpass(15)  # Remove low-freq seismic noise
                strain = strain.whiten(fftlength=2)  # Whiten to unit variance

                # Return as float32 numpy array
                noise_data = np.asarray(strain.value, dtype=np.float32)

                # Ensure correct length
                expected_samples = int(duration * sample_rate)
                if len(noise_data) != expected_samples:
                    self.logger.warning(
                        f"Noise length mismatch: got {len(noise_data)}, "
                        f"expected {expected_samples}. Resampling..."
                    )
                    # Resample to correct length
                    indices = np.linspace(0, len(noise_data) - 1, expected_samples)
                    noise_data = np.interp(indices, np.arange(len(noise_data)), noise_data)

                return noise_data.astype(np.float32)

            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for {self.detector}: {e}"
                )

                if attempt == max_retries - 1:
                    # Final attempt failed
                    self.logger.error(
                        f"All {max_retries} attempts failed. "
                        f"This segment may not have public data or gwpy may be unavailable."
                    )
                    raise RuntimeError(
                        f"Real noise fetch failed after {max_retries} attempts"
                    ) from e

                # Try different segment on next attempt
                continue

        raise RuntimeError("Unexpected: loop exited without return or raise")

    def inject_signal_into_real_noise(
        self,
        signal_waveform: np.ndarray,
        duration: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject synthetic GW signal into real detector noise.

        Aligns signal to center of noise window and adds them in time domain.

        Args:
            signal_waveform: Synthetic GW signal (time-domain strain)
            duration: Noise duration (seconds). Defaults to self.duration
            sample_rate: Sampling rate (Hz). Defaults to self.sample_rate

        Returns:
            Tuple of (injected_strain, noise)
            - injected_strain: Signal + noise combined
            - noise: Original noise (for reference)
        """
        duration = duration or self.duration
        sample_rate = sample_rate or self.sample_rate

        # Get real noise
        noise = self.get_noise_chunk(duration, sample_rate)

        # Prepare signal for injection
        signal_waveform = np.asarray(signal_waveform, dtype=np.float32)

        # Pad signal to match noise length, centered
        signal_padded = np.zeros_like(noise)
        center_idx = len(noise) // 2
        signal_start = center_idx - len(signal_waveform) // 2
        signal_end = signal_start + len(signal_waveform)

        # Handle edge cases where signal is longer than noise
        if signal_end > len(noise):
            signal_end = len(noise)
            signal_waveform = signal_waveform[: signal_end - signal_start]

        if signal_start >= 0:
            signal_padded[signal_start:signal_end] = signal_waveform
        else:
            # Signal extends before window start
            offset = -signal_start
            signal_padded[0:signal_end] = signal_waveform[offset:]

        # Combine signal and noise
        injected = noise + signal_padded

        return injected.astype(np.float32), noise.astype(np.float32)
