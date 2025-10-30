"""
PSD (Power Spectral Density) Manager
Handles detector noise curves with PyCBC and analytical fallbacks
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from pathlib import Path

try:
    from pycbc import psd as pycbc_psd
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False

from .config import SAMPLE_RATE, DURATION

class PSDManager:
    """
    Manage detector PSDs with comprehensive fallback system
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, duration: float = DURATION):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.logger = logging.getLogger(__name__)
        self.psds = {}
    
    def load_detector_psds(self, detectors: List[str] = None) -> Dict:
        """Load PSDs for all requested detectors"""
        if detectors is None:
            detectors = ['H1', 'L1', 'V1']
        
        psds = {}
        
        for detector_name in detectors:
            try:
                if PYCBC_AVAILABLE:
                    psds[detector_name] = self._load_pycbc_psd(detector_name)
                else:
                    psds[detector_name] = self._create_analytical_psd(detector_name)
                    
                self.logger.info(f"✓ {detector_name} PSD loaded")
                
            except Exception as e:
                self.logger.warning(f"✗ {detector_name} PSD failed: {e}")
                psds[detector_name] = self._create_analytical_psd(detector_name)
        
        self.psds = psds
        return psds
    
    def _load_pycbc_psd(self, detector_name: str) -> Dict:
        """Load PSD using PyCBC analytical models"""
        delta_f = 1.0 / self.duration
        flen = self.n_samples // 2 + 1
        
        psd_names = self._get_psd_names(detector_name)
        
        for psd_name in psd_names:
            try:
                psd = pycbc_psd.from_string(psd_name, flen, delta_f, 10.0)
                return {
                    'psd': psd,
                    'frequencies': psd.sample_frequencies.numpy(),
                    'asd': np.sqrt(psd.numpy()),
                    'name': psd_name,
                    'source': 'pycbc'
                }
            except Exception:
                continue
        
        # Fallback
        psd = pycbc_psd.from_string('aLIGOZeroDetHighPower', flen, delta_f, 10.0)
        return {
            'psd': psd,
            'frequencies': psd.sample_frequencies.numpy(),
            'asd': np.sqrt(psd.numpy()),
            'name': 'aLIGOZeroDetHighPower',
            'source': 'pycbc_fallback'
        }
    
    def _create_analytical_psd(self, detector_name: str) -> Dict:
        """Create analytical PSD model (fallback)"""
        delta_f = 1.0 / self.duration
        flen = self.n_samples // 2 + 1
        frequencies = np.arange(flen) * delta_f
        
        # Advanced aLIGO analytical model
        psd = np.ones(flen) * 1e-47
        
        # Low frequency: seismic wall
        low_mask = frequencies < 40
        if np.any(low_mask):
            f_low = frequencies[low_mask]
            psd[low_mask] = 1e-44 * (f_low / 40.0)**(-4.14)
        
        # Mid frequency: thermal noise floor
        mid_mask = (frequencies >= 40) & (frequencies < 200)
        if np.any(mid_mask):
            psd[mid_mask] = 3e-48
        
        # High frequency: shot noise
        high_mask = frequencies >= 200
        if np.any(high_mask):
            f_high = frequencies[high_mask]
            psd[high_mask] = 1e-47 * (f_high / 200.0)**2
        
        # Detector-specific adjustments
        if detector_name == 'V1':
            psd *= 2.0  # Virgo is ~2x worse
        
        # Add spectral lines
        psd = self._add_spectral_lines(frequencies, psd, detector_name)
        
        return {
            'psd': psd,
            'frequencies': frequencies,
            'asd': np.sqrt(psd),
            'name': f'analytical_{detector_name}',
            'source': 'analytical'
        }
    
    def _add_spectral_lines(self, frequencies: np.ndarray, 
                           psd: np.ndarray, 
                           detector_name: str) -> np.ndarray:
        """Add realistic spectral lines to PSD"""
        # Power line harmonics
        if detector_name == 'V1':
            line_freqs = [50, 100, 150, 200, 250]  # European 50 Hz
        else:
            line_freqs = [60, 120, 180, 240, 300]  # US 60 Hz
        
        for line_freq in line_freqs:
            if line_freq < frequencies.max():
                line_idx = np.argmin(np.abs(frequencies - line_freq))
                for offset in range(-1, 2):
                    idx = line_idx + offset
                    if 0 <= idx < len(psd):
                        line_strength = 3.0 * np.exp(-offset**2 / 0.5)
                        psd[idx] *= line_strength
        
        # Violin modes
        violin_modes = self._get_violin_modes(detector_name)
        for mode_freq in violin_modes:
            if mode_freq < frequencies.max():
                mode_idx = np.argmin(np.abs(frequencies - mode_freq))
                for offset in range(-2, 3):
                    idx = mode_idx + offset
                    if 0 <= idx < len(psd):
                        mode_strength = 2.0 * np.exp(-offset**2 / 2.0)
                        psd[idx] *= mode_strength
        
        return psd
    
    def _get_psd_names(self, detector_name: str) -> List[str]:
        """Get ordered list of PSD names to try"""
        if detector_name == 'V1':
            return ['AdvVirgo', 'Virgo', 'aLIGOZeroDetHighPower']
        return ['aLIGOZeroDetHighPower', 'aLIGODesignSensitivityP1200087']
    
    def _get_violin_modes(self, detector_name: str) -> List[float]:
        """Get violin mode frequencies for detector"""
        modes = {
            'H1': [347.0, 694.0, 1041.0, 1388.0],
            'L1': [331.0, 662.0, 993.0, 1324.0],
            'V1': [350.0, 700.0, 1050.0, 1400.0]
        }
        return modes.get(detector_name, [350.0, 700.0, 1050.0])
    
    def get_psd(self, detector_name: str) -> Optional[Dict]:
        """Retrieve loaded PSD for detector"""
        return self.psds.get(detector_name)
    
    def save_psds(self, output_dir: str):
        """Save PSDs to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for detector_name, psd_info in self.psds.items():
            psd_file = output_path / f'{detector_name}_psd.npz'
            psd_array = psd_info['psd']
            if hasattr(psd_array, 'numpy'):
                psd_array = psd_array.numpy()
            
            np.savez(psd_file,
                    frequencies=psd_info['frequencies'],
                    psd=psd_array,
                    source=psd_info['source'],
                    name=psd_info['name'])
        
        self.logger.info(f"✓ PSDs saved to {output_path}")
