"""
Realistic Noise Generation for GW Detectors
Generates colored Gaussian noise with glitches and spectral artifacts
"""

import numpy as np
import logging
from typing import Dict, List
from scipy.signal import butter, filtfilt

try:
    from pycbc.noise import noise_from_psd
    from pycbc.types import FrequencySeries
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False

from .config import SAMPLE_RATE, DURATION

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
        
        # Try PyCBC first
        if PYCBC_AVAILABLE and 'psd' in psd_dict and hasattr(psd_dict['psd'], 'data'):
            try:
                psd = psd_dict['psd']
                noise_ts = noise_from_psd(self.n_samples, 1.0/self.sample_rate, psd, seed=seed)
                return np.array(noise_ts.data, dtype=np.float32)
            except Exception as e:
                self.logger.debug(f"PyCBC noise generation failed: {e}")
        
        # Fallback: analytical colored noise
        return self.generate_analytical_colored_noise(psd_dict)
    
    def generate_analytical_colored_noise(self, psd_dict: Dict) -> np.ndarray:
        """Generate colored noise using frequency-domain method"""
        
        # White Gaussian noise in time domain
        white_noise = np.random.randn(self.n_samples)
        
        # FFT to frequency domain
        white_fft = np.fft.rfft(white_noise)
        frequencies = np.fft.rfftfreq(self.n_samples, 1.0/self.sample_rate)
        
        # Get ASD (amplitude spectral density)
        if 'asd' in psd_dict:
            asd = psd_dict['asd']
        elif 'psd' in psd_dict:
            psd = psd_dict['psd']
            if hasattr(psd, 'numpy'):
                psd = psd.numpy()
            asd = np.sqrt(psd)
        else:
            # Default aLIGO-like ASD
            asd = self.default_asd(frequencies)
        
        # Apply coloring
        colored_fft = white_fft * asd * np.sqrt(self.sample_rate / 2)
        
        # Back to time domain
        colored_noise = np.fft.irfft(colored_fft, n=self.n_samples)
        
        return colored_noise.astype(np.float32)
    
    def default_asd(self, frequencies: np.ndarray) -> np.ndarray:
        """Default aLIGO ASD model"""
        
        asd = np.ones_like(frequencies) * 1e-23
        
        # Low frequency: seismic wall
        low_mask = frequencies < 40
        if np.any(low_mask):
            asd[low_mask] = 1e-22 * (frequencies[low_mask] / 40.0)**(-2.07)
        
        # Mid frequency: thermal noise floor
        mid_mask = (frequencies >= 40) & (frequencies < 200)
        if np.any(mid_mask):
            asd[mid_mask] = 3e-24
        
        # High frequency: shot noise
        high_mask = frequencies >= 200
        if np.any(high_mask):
            asd[high_mask] = 1e-23 * (frequencies[high_mask] / 200.0)
        
        return asd
    
    def add_glitches(self, noise: np.ndarray, glitch_prob: float = 0.3, n_glitches: int = None) -> np.ndarray:
        """Add realistic glitches to noise"""
        
        if n_glitches is None:
            # Randomly decide number of glitches
            if np.random.random() > glitch_prob:
                return noise
            n_glitches = np.random.randint(1, 4)
        
        for _ in range(n_glitches):
            glitch_type = np.random.choice(['blip', 'whistle', 'scratch', 'wandering_line'])
            
            if glitch_type == 'blip':
                noise = self.add_blip_glitch(noise)
            elif glitch_type == 'whistle':
                noise = self.add_whistle_glitch(noise)
            elif glitch_type == 'scratch':
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
        envelope = np.exp(-((t_glitch - t_glitch[-1]/2) / (t_glitch[-1]/4))**2)
        
        # Random frequency
        frequency = np.random.uniform(100, 500)
        amplitude = np.random.uniform(5e-23, 20e-23)
        
        glitch = amplitude * envelope * np.sin(2*np.pi*frequency*t_glitch)
        
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
        phase = 2*np.pi*np.cumsum(frequency) / self.sample_rate
        amplitude = np.random.uniform(5e-23, 30e-23)
        envelope = np.exp(-((t_glitch - t_glitch[-1]/2) / (t_glitch[-1]/3))**2)
        
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
        frequency = base_freq + freq_variation * np.sin(2*np.pi*0.5*t_glitch)
        
        amplitude = np.random.uniform(2e-23, 15e-23)
        phase = 2*np.pi*np.cumsum(frequency) / self.sample_rate
        
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
            
            b, a = butter(4, [low, high], btype='band')
            return filtfilt(b, a, data)
        except:
            return data
