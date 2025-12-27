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
                psd_arr = psd.numpy()
                
                # PyCBC PSDs need scaling to match physical units (strain^2/Hz)
                # Empirically scale to ~1e-46 at 100 Hz for aLIGO
                freqs = psd.sample_frequencies.numpy()
                idx_100 = np.argmin(np.abs(freqs - 100.0))
                if psd_arr[idx_100] > 0:
                    scaling = 1e-46 / psd_arr[idx_100]
                    psd_arr = psd_arr * scaling
                
                # Clamp ASD to realistic values (1e-23 to 1e-22 strain per sqrt Hz)
                asd = np.sqrt(np.maximum(psd_arr, 1e-46))
                asd = np.maximum(asd, 1e-22)  # Ensure minimum realistic noise floor
                return {
                    'psd': asd ** 2,  # Recompute PSD from clamped ASD
                    'frequencies': freqs,
                    'asd': asd,
                    'name': psd_name,
                    'source': 'pycbc'
                }
            except Exception:
                continue
        
        # Fallback
        psd = pycbc_psd.from_string('aLIGOZeroDetHighPower', flen, delta_f, 10.0)
        psd_arr = psd.numpy()
        freqs = psd.sample_frequencies.numpy()
        idx_100 = np.argmin(np.abs(freqs - 100.0))
        if psd_arr[idx_100] > 0:
            scaling = 1e-46 / psd_arr[idx_100]
            psd_arr = psd_arr * scaling
        
        # Clamp ASD to realistic values (1e-23 to 1e-22 strain per sqrt Hz)
        asd = np.sqrt(np.maximum(psd_arr, 1e-46))
        asd = np.maximum(asd, 1e-22)  # Ensure minimum realistic noise floor
        return {
            'psd': asd ** 2,  # Recompute PSD from clamped ASD
            'frequencies': freqs,
            'asd': asd,
            'name': 'aLIGOZeroDetHighPower',
            'source': 'pycbc_fallback'
        }
    
    def _create_analytical_psd(self, detector_name: str) -> Dict:
        """Create realistic analytical PSD model"""
        delta_f = 1.0 / self.duration
        flen = self.n_samples // 2 + 1
        frequencies = np.arange(flen) * delta_f
        
        # Avoid zero frequency issues
        frequencies_safe = np.maximum(frequencies, 1.0)
        
        if detector_name == 'V1':
            psd = self._create_virgo_psd(frequencies_safe)
        else:
            psd = self._create_aLIGO_psd(frequencies_safe)
        
        # Add spectral lines for realism
        psd = self._add_spectral_lines(frequencies, psd, detector_name)
        
        # Clamp ASD to realistic values (1e-23 to 1e-22 strain per sqrt Hz)
        asd = np.sqrt(np.maximum(psd, 1e-46))
        asd = np.maximum(asd, 1e-22)  # Ensure minimum realistic noise floor
        return {
            'psd': asd ** 2,  # Recompute PSD from clamped ASD
            'frequencies': frequencies,
            'asd': asd,
            'name': f'analytical_{detector_name}',
            'source': 'analytical'
        }
    
    def _create_aLIGO_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """Realistic aLIGO PSD model based on O3 sensitivity"""
        # ASD reference: ~3e-23 at 100 Hz for aLIGO
        # PSD = ASD^2, so reference PSD ~9e-46 at 100 Hz
        
        psd = np.zeros_like(frequencies, dtype=float)
        
        # Low frequency: seismic wall (~1/f^2 scaling)
        # ASD ~ 1e-21 * (f/10)^(-2.07)
        low_mask = frequencies <= 20
        psd[low_mask] = (1e-21 * (frequencies[low_mask] / 10.0)**(-2.07))**2
        
        # Transition region (20-60 Hz): smooth interpolation
        trans_mask = (frequencies > 20) & (frequencies < 60)
        low_edge = psd[frequencies <= 20][-1] if np.any(frequencies <= 20) else 1e-43
        high_edge = (3e-23)**2  # Mid-freq baseline ~9e-46
        f_trans = frequencies[trans_mask]
        psd[trans_mask] = low_edge + (high_edge - low_edge) * ((f_trans - 20) / 40)**2
        
        # Mid frequency: thermal noise floor (relatively flat)
        # ASD ~ 2-4e-23, so PSD ~ 4-16e-46
        mid_mask = (frequencies >= 60) & (frequencies <= 250)
        psd[mid_mask] = (3e-23)**2
        
        # Transition to high frequency (250-500 Hz): smooth
        trans_high_mask = (frequencies > 250) & (frequencies < 500)
        f_trans_high = frequencies[trans_high_mask]
        psd[trans_high_mask] = ((3e-23)**2) * (1 + 0.5 * ((f_trans_high - 250) / 250)**1.5)
        
        # High frequency: shot noise (~f^2 scaling)
        # ASD ~ 1e-22 * (f/200)^1
        high_mask = frequencies >= 500
        psd[high_mask] = (1e-22 * (frequencies[high_mask] / 200.0))**2
        
        return psd
    
    def _create_virgo_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """Realistic Virgo PSD model (higher noise than aLIGO)"""
        # Virgo is ~2-3x less sensitive than aLIGO
        # Reference ASD ~ 5-7e-23 at 100 Hz
        
        psd = np.zeros_like(frequencies, dtype=float)
        
        # Low frequency: seismic wall
        # ASD ~ 3e-21 * (f/10)^(-2.0)
        low_mask = frequencies <= 20
        psd[low_mask] = (3e-21 * (frequencies[low_mask] / 10.0)**(-2.0))**2
        
        # Transition region
        trans_mask = (frequencies > 20) & (frequencies < 60)
        low_edge = psd[frequencies <= 20][-1] if np.any(frequencies <= 20) else 1e-42
        high_edge = (7e-23)**2  # Virgo reference at ~100 Hz
        f_trans = frequencies[trans_mask]
        psd[trans_mask] = low_edge + (high_edge - low_edge) * ((f_trans - 20) / 40)**2
        
        # Mid frequency: broader plateau with stronger features
        # ASD ~ 5-8e-23
        mid_mask = (frequencies >= 60) & (frequencies <= 300)
        psd[mid_mask] = (6e-23)**2
        
        # Transition to high frequency
        trans_high_mask = (frequencies > 300) & (frequencies < 500)
        f_trans_high = frequencies[trans_high_mask]
        psd[trans_high_mask] = ((6e-23)**2) * (1 + 0.3 * ((f_trans_high - 300) / 200)**1.5)
        
        # High frequency: shot noise
        # ASD ~ 2e-22 * (f/200)^0.8
        high_mask = frequencies >= 500
        psd[high_mask] = (2e-22 * (frequencies[high_mask] / 200.0)**0.8)**2
        
        return psd
    
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
