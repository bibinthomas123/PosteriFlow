#!/usr/bin/env python3
"""
Combined Gravitational Wave Dataset Generator
==============================================
All modules merged into a single standalone script.

This script combines all functionality from:
- utils.py: Utility functions (cosmology, SNR)
- psd_manager.py: Power Spectral Density management
- parameter_sampler.py: Astrophysical parameter sampling
- waveform_generator.py: GW waveform generation
- noise_generator.py: Realistic detector noise
- preprocessing.py: Data preprocessing (whitening, filtering)
- gwtc_loader.py: Real GW event catalog loading
- injection.py: Signal injection with SNR control
- simulation.py: Overlapping signal simulation
- dataset_generator.py: Main dataset generation orchestration

Usage:
    python combined_gw_dataset_generator.py --n_samples 1000 --output_dir ./data

Author: AHSD Pipeline
Date: 2025-10-30
"""

import numpy as np
import logging
import sys
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
import time
import gc
import pickle
import json
from collections import Counter
from functools import wraps
import random
import math
import warnings
import h5py
from scipy.stats import truncnorm, beta
from scipy.signal import butter, filtfilt, welch, windows, get_window, spectrogram
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import brentq
import yaml
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor as ProcessPoolExecutor, as_completed
import multiprocessing
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== OPTIONAL IMPORTS ==========
try:
    from pycbc import psd as pycbc_psd
    from pycbc.waveform import get_td_waveform, get_fd_waveform
    from pycbc.detector import Detector
    from pycbc.types import TimeSeries, FrequencySeries
    from pycbc.noise import noise_from_psd
    from pycbc.filter import matched_filter, sigma
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False
    logger.warning("PyCBC not available. Using analytical fallbacks.")

try:
    from gwpy.timeseries import TimeSeries as GWpyTimeSeries
    GWPY_AVAILABLE = True
except ImportError:
    GWPY_AVAILABLE = False
    logger.warning("GWpy not available. Real strain download disabled.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available. Some features may be limited.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("Requests not available. GWTC loading disabled.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import bilby
    bilby.core.utils.logger.setLevel('WARNING')
    BILBY_AVAILABLE = True
except ImportError:
    BILBY_AVAILABLE = False
    logger.warning("Bilby not available. Some simulation features disabled.")

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available. JIT compilation disabled.")
    

# Suppress warnings
warnings.filterwarnings("ignore", message="Unkown projection method")
warnings.filterwarnings("ignore", message="Unknown projection method")


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Sampling and duration
SAMPLE_RATE = 4096  # Hz
DURATION = 4.0  # seconds

# Detector configuration
DETECTORS = ['H1', 'L1', 'V1']

# Frequency range
F_LOWER = 20.0  # Hz
F_UPPER = 1024.0  # Hz

# Event type distribution
# These values are used throughout the code for:
# - Sampling event types in single/overlap/edge case generation
# - Validation logging to compare actual vs expected distributions
# Adjust these to match your desired signal composition
EVENT_TYPE_DISTRIBUTION = {
    'BBH': 0.46,    # Binary Black Hole mergers (46%)
    'BNS': 0.32,    # Binary Neutron Star mergers (32%)
    'NSBH': 0.17,   # Neutron Star-Black Hole mergers (17%)
    'noise': 0.05   # Noise-only samples (5%)
}

# SNR regime distribution
# These values control the distribution of SNR values across signals
# Used for sampling and validation logging
# Adjust these to match your desired SNR composition
SNR_DISTRIBUTION = {
    'weak': 0.15,    # SNR 5-10 (15%)
    'low': 0.35,     # SNR 10-15 (35%)
    'medium': 0.30,  # SNR 15-25 (30%)
    'high': 0.15,    # SNR 25-40 (15%)
    'loud': 0.05     # SNR >40 (5%)
}

# SNR ranges for each regime
SNR_RANGES = {
    'weak': (5.0, 10.0),
    'low': (10.0, 15.0),
    'medium': (15.0, 25.0),
    'high': (25.0, 40.0),
    'loud': (40.0, 60.0)
}

# Mass ranges (solar masses)
MASS_RANGES = {
    'BBH': (5.0, 100.0),
    'BNS': (1.0, 2.5),
    'NSBH': (1.0, 100.0)
}

# Distance ranges (Mpc)
DISTANCE_RANGES = {
    "BBH": (100.0, 2000.0),
    "BNS": (10.0, 200.0),
    "NSBH": (20.0, 800.0)
}


# Overlapping and edge case fractions
OVERLAP_FRACTION = 0.20
EDGE_CASE_FRACTION = 0.15

# Cosmology parameters
COSMO_H0 = 67.9  # km/s/Mpc
COSMO_OMEGA_M = 0.3065
COSMO_OMEGA_LAMBDA = 0.6935

# Approximants
APPROXIMANTS = {
    'BBH': {
        'non_precessing': ['SEOBNRv4', 'IMRPhenomXAS', 'IMRPhenomD'],
        'precessing': ['SEOBNRv4P', 'IMRPhenomPv2']
    },
    'BNS': {
        'tidal': ['TaylorF2', 'IMRPhenomD_NRTidal']
    },
    'NSBH': {
        'non_precessing': ['IMRPhenomD_NRTidal', 'IMRPhenomXAS', 'IMRPhenomD'],
        'precessing': ['IMRPhenomPv2_NRTidal', 'IMRPhenomPv2']
    }
}

logger.info(f"Configuration loaded: SAMPLE_RATE={SAMPLE_RATE}, DURATION={DURATION}")



# ================================================================================
# UTILS.PY
# ================================================================================

"""
Utility functions for AHSD data module
Cosmology calculations, SNR computation, effective spin, etc.
"""

import numpy as np
import logging
from typing import Optional


logger = logging.getLogger(__name__)

def calculate_redshift(luminosity_distance: float) -> Optional[float]:
    """Calculate redshift from luminosity distance using Planck18 cosmology."""
    from astropy.cosmology import Planck18 as cosmo
    import astropy.units as u
    from scipy.optimize import brentq
    import numpy as np
    import warnings

    if luminosity_distance <= 0:
        return 0.0

    try:
        def residual(z):
            d_L_model = cosmo.luminosity_distance(z).to(u.Mpc).value
            return d_L_model - luminosity_distance

        z = brentq(residual, 0.0, 5.0)
        return float(z)

    except Exception as e:
        warnings.warn(f"Redshift calculation failed ({e}); using fallback approximation.")
        # H0 = 67.4 km/s/Mpc → approximate z = H0 * D_L / c (for small z)
        z_approx = luminosity_distance * 67.4 / 3e5
        return float(np.clip(z_approx, 0.0, 5.0))

    

def calculate_comoving_distance(z: float) -> float:
    """Calculate comoving distance from redshift using Astropy for consistency"""
    if z <= 0:
        return 0.0

    try:
        from astropy.cosmology import Planck18 as cosmo
        import astropy.units as u
        d_C = cosmo.comoving_distance(z).to(u.Mpc).value
        return float(d_C)
    except ImportError:
        # Fallback to numerical integration if astropy not available
        from scipy.integrate import quad
        H0 = COSMO_H0
        Om_m = COSMO_OMEGA_M
        Om_L = COSMO_OMEGA_LAMBDA
        c = 299792.458

        def integrand(zp):
            E_z = np.sqrt(Om_m * (1 + zp)**3 + Om_L)
            return 1.0 / E_z

        result, _ = quad(integrand, 0, z, limit=100)
        d_C = (c / H0) * result
        return float(d_C)

def sample_redshift_from_comoving_volume(z_min: float = 0.001, z_max: float = 2.0) -> float:
    """
    Sample redshift from uniform comoving volume prior.

    This ensures proper decoupling of mass and distance by sampling z
    from a distribution proportional to comoving volume dV_c/dz.

    Args:
        z_min: Minimum redshift
        z_max: Maximum redshift

    Returns:
        Redshift value
    """
    try:
        from astropy.cosmology import Planck18 as cosmo
        import astropy.units as u
        from scipy.integrate import quad
        from scipy.interpolate import interp1d

        # Define the comoving volume integrand dV_c/dz = 4π χ^2 dc/dz
        # where χ is comoving distance
        def dV_dz(z):
            chi = cosmo.comoving_distance(z).to(u.Mpc).value
            dchi_dz = cosmo.comoving_distance(z + 0.001).to(u.Mpc).value - chi
            return 4 * np.pi * chi**2 * (dchi_dz / 0.001)

        # Create a grid of z values and corresponding volumes
        z_grid = np.linspace(z_min, z_max, 1000)
        vol_grid = np.array([dV_dz(z) for z in z_grid])

        # Create CDF for sampling
        cdf = np.cumsum(vol_grid)
        cdf /= cdf[-1]  # Normalize

        # Sample from CDF
        u = np.random.uniform(0, 1)
        idx = np.searchsorted(cdf, u)

        if idx >= len(z_grid) - 1:
            return float(z_grid[-1])
        elif idx <= 0:
            return float(z_grid[0])
        else:
            # Linear interpolation
            frac = (u - cdf[idx-1]) / (cdf[idx] - cdf[idx-1])
            return float(z_grid[idx-1] + frac * (z_grid[idx] - z_grid[idx-1]))

    except ImportError:
        # Fallback: log-uniform in redshift (approximation)
        logger.warning("Astropy not available for proper comoving volume sampling. Using log-uniform approximation.")
        log_z_min = np.log(1 + z_min)
        log_z_max = np.log(1 + z_max)
        log_z = np.random.uniform(log_z_min, log_z_max)
        return float(np.exp(log_z) - 1)

def compute_luminosity_distance_from_redshift(z: float) -> float:
    """
    Compute luminosity distance from redshift using Planck18 cosmology.

    Args:
        z: Redshift

    Returns:
        Luminosity distance in Mpc
    """
    try:
        from astropy.cosmology import Planck18 as cosmo
        import astropy.units as u
        d_L = cosmo.luminosity_distance(z).to(u.Mpc).value
        return float(d_L)
    except ImportError:
        # Fallback approximation: d_L ≈ (c/H0) * z for small z
        # More accurate: integrate 1/E(z) dz
        from scipy.integrate import quad
        H0 = COSMO_H0
        Om_m = COSMO_OMEGA_M
        Om_L = COSMO_OMEGA_LAMBDA
        c = 299792.458  # km/s

        def integrand(zp):
            E_z = np.sqrt(Om_m * (1 + zp)**3 + Om_L)
            return 1.0 / E_z

        integral, _ = quad(integrand, 0, z, limit=100)
        d_L = (c / H0) * integral
        return float(d_L)

def compute_effective_spin(m1: float, m2: float, 
                          a1: float, a2: float, 
                          tilt1: float, tilt2: float) -> float:
    """Compute effective inspiral spin parameter χ_eff

    Parameters:
        m1, m2: masses (solar masses)
        a1, a2: spin magnitudes (0-1)
        tilt1, tilt2: tilt angles (in RADIANS from binary axis)

    Formula: χ_eff = (a1*m1*cos(θ1) + a2*m2*cos(θ2)) / (m1 + m2)
    """
    # Ensure tilts are in radians (convert from degrees if needed)
    # If tilt values are > π, assume they're in degrees
    tilt1_rad = np.radians(tilt1) if tilt1 > np.pi else tilt1
    tilt2_rad = np.radians(tilt2) if tilt2 > np.pi else tilt2

    # Validate spin magnitudes
    a1_clip = np.clip(a1, 0, 1)
    a2_clip = np.clip(a2, 0, 1)

    total_mass = m1 + m2

    # Effective spin: weighted sum of spin projections
    chi_eff = (a1_clip * m1 * np.cos(tilt1_rad) + a2_clip * m2 * np.cos(tilt2_rad)) / total_mass

    # Clip to physically valid range [-1, 1]
    chi_eff = np.clip(chi_eff, -1.0, 1.0)

    return float(chi_eff)


# --- PHYSICAL SNR util: canonical chirp mass scaling -------------------------
@numba.jit(nopython=True) if NUMBA_AVAILABLE else lambda f: f
def chirp_mass(m1: float, m2: float) -> float:
    """Return the chirp mass for masses m1, m2 (in same units)."""
    M = float(m1 + m2)
    if M <= 0:
        return 0.0
    return (m1 * m2)**(3.0/5.0) / (M**(1.0/5.0))

def compute_physical_snr_from_params(params: dict,
                                    snr_ref: float = None,
                                    reference_mass: float = None,
                                    reference_distance: float = None,
                                    min_snr: float = 4.5,
                                    max_snr: float = 1000.0) -> float:
    """
    Compute a physically-consistent network SNR from params using:
      snr = snr_ref * (M_chirp / reference_mass)^(5/6) * (reference_distance / D)
    Params keys expected: 'mass_1', 'mass_2', 'luminosity_distance' (or 'distance').
    If any reference values are None, fallback to sensible defaults if available on `params`.
    """
    # try common param names
    m1 = float(params.get('mass_1', params.get('m1', 30.0)))
    m2 = float(params.get('mass_2', params.get('m2', 30.0)))
    d = float(params.get('luminosity_distance', params.get('distance', params.get('d', 1.0))))
    # sensible defaults if not passed (these should be replaced by your class-level refs)
    if snr_ref is None:
        snr_ref = float(params.get('_snr_ref', 25.0))
    if reference_mass is None:
        reference_mass = float(params.get('_reference_mass', 30.0))
    if reference_distance is None:
        reference_distance = float(params.get('_reference_distance', 100.0))

    M_ch = chirp_mass(m1, m2)
    if M_ch <= 0 or d <= 0:
        return float(np.clip(snr_ref, min_snr, max_snr))
    snr = snr_ref * ( (M_ch / reference_mass) ** (5.0/6.0) ) * (reference_distance / d)
    # guard rails
    return float(np.clip(snr, min_snr, max_snr))


def compute_snr_model(params: dict, model: str = 'physical', **kwargs) -> float:
    """
    Unified entry point for SNR calculation. model choices: 'physical', 'proxy'
    - 'physical' uses chirp-mass / distance scaling
    - 'proxy' should call your existing proxy_network_snr_from_params(params)
    """
    if model == 'physical':
        return compute_physical_snr_from_params(params, **kwargs)
    elif model == 'proxy':
        # expect a proxy function to exist in your module
        return proxy_network_snr_from_params(params)
    else:
        raise ValueError("Unknown snr model: %s" % model)



# ================================================================================
# PSD_MANAGER.PY
# ================================================================================

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


## ================================================================================
# PARAMETER_SAMPLER.PY
## ================================================================================

"""
Parameter Sampler for GW Signals
Implements astrophysically realistic parameter distributions
"""

import numpy as np
import logging
from typing import Dict, List
from scipy.stats import truncnorm, beta




def _draw_until_snr(make_params_fn, snr_range, attach_snr_fn, max_tries: int = 3000):
    """
    Rejection-sampling loop:
    repeatedly draw params until computed SNR lies inside snr_range.
    Robust against missing target_snr or attach_network_snr_safe variants.
    """
    low, high = snr_range
    last = None

    for attempt in range(max_tries):
        params = make_params_fn()

        # Compute SNR safely
        try:
            snr_val = attach_snr_fn(params, mutate_params=False)
        except TypeError:
            # fallback if function doesn't support the flag
            snr_val = attach_snr_fn(params)
        except Exception:
            snr_val = None

        if snr_val is None:
            # fallback to proxy
            snr_val = proxy_network_snr_from_params(params)

        # still None? skip
        if snr_val is None:
            last = params
            continue

        params["network_snr"] = float(snr_val)
        last = params

        # Accept if within range
        if low <= snr_val <= high:
            params["accepted"] = True
            params["rejection_sampling_exhausted"] = False
            return params

    # fallback: return last, mark as failed
    if last is not None:
        last["accepted"] = False
        last["rejection_sampling_exhausted"] = True
        return last

    return {"accepted": False, "error": "Failed to generate sample"}

@numba.jit(nopython=True) if NUMBA_AVAILABLE else lambda f: f
def _sample_powerlaw(primary_min: float, primary_max: float, alpha: float) -> float:
        """Draw from p(m) ∝ m^{-alpha} on [primary_min, primary_max] by inverse-transform."""
        if primary_min <= 0:
            primary_min = 1e-3
        if abs(alpha - 1.0) < 1e-12:
            r = np.random.uniform(0, 1)
            return primary_min * (primary_max / primary_min) ** r
        r = np.random.uniform(0, 1)
        pow_min = primary_min ** (1.0 - alpha)
        pow_max = primary_max ** (1.0 - alpha)
        return (pow_min + r * (pow_max - pow_min)) ** (1.0 / (1.0 - alpha))
    
def _sample_mass_ratio(q_min: float = 0.05, q_max: float = 1.0, beta_param: float = 0.0) -> float:
        """Draw q from p(q) ∝ q^{beta_param} on [q_min, q_max]. beta_param=0 -> uniform."""
        if q_min >= q_max:
            return float(q_min)
        if abs(beta_param) < 1e-12:
            return float(np.random.uniform(q_min, q_max))
        r = np.random.uniform(0, 1)
        pow_min = q_min ** (beta_param + 1)
        pow_max = q_max ** (beta_param + 1)
        return float((pow_min + r * (pow_max - pow_min)) ** (1.0 / (beta_param + 1)))


@numba.jit(nopython=True) if NUMBA_AVAILABLE else lambda f: f
def _sample_volume_distance(d_min: float, d_max: float) -> float:
    """Volume-weighted distance (P(d) ∝ d^2)."""
    u = np.random.uniform(0, 1)
    return float((d_min**3 + u * (d_max**3 - d_min**3)) ** (1.0/3.0))


class ParameterSampler:
    """
    Sample astrophysically realistic GW parameters.
    Uses SNR_DISTRIBUTION and EVENT_TYPE_DISTRIBUTION from config.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mass_ranges = MASS_RANGES
        self.distance_ranges = DISTANCE_RANGES
        self.snr_ranges = SNR_RANGES
        assert hasattr(self, 'snr_ranges'), "ParameterSampler: snr_ranges not initialized"
        
        #  Store distributions from config
        self.snr_distribution = SNR_DISTRIBUTION
        self.event_type_distribution = EVENT_TYPE_DISTRIBUTION
        
        self.reference_snr = 75
        self.reference_mass = 30.0
        self.reference_distance = 400.0
        
        self.stats = {
            'event_types': {'BBH': 0, 'BNS': 0, 'NSBH': 0},
            'edge_cases': {
                'short_bbh': 0,
                'extreme_mass_ratio': 0,
                'extreme_mass': 0,
                'long_bns_inspiral': 0,
                'high_spin': 0
            },
            'snr_regimes': {regime: 0 for regime in self.snr_ranges.keys()}
        }
    
    def _sample_snr_regime(self) -> str:
        """Sample SNR regime from configured distribution."""
        regimes = list(self.snr_distribution.keys())
        probs = list(self.snr_distribution.values())
        return np.random.choice(regimes, p=probs)

    def _sample_target_snr_from_distribution(self) -> float:
        """Sample target SNR from the desired distribution."""
        regimes = list(SNR_DISTRIBUTION.keys())
        probs = list(SNR_DISTRIBUTION.values())

        # Sample regime according to desired distribution
        regime = np.random.choice(regimes, p=probs)

        # Sample SNR within that regime
        min_snr, max_snr = SNR_RANGES[regime]

        if regime == 'weak':
            # For weak SNR, use uniform distribution
            snr = np.random.uniform(min_snr, max_snr)
        elif regime == 'low':
            # For low SNR, slightly skewed toward higher values
            snr = np.random.uniform(min_snr, max_snr)
        elif regime == 'medium':
            # For medium SNR, uniform
            snr = np.random.uniform(min_snr, max_snr)
        elif regime == 'high':
            # For high SNR, uniform
            snr = np.random.uniform(min_snr, max_snr)
        else:  # loud
            # For loud SNR, power-law decay
            alpha = 2.0  # Shape parameter for power-law
            u = np.random.uniform(0, 1)
            snr = min_snr + (max_snr - min_snr) * (1 - u)**(1/(alpha-1))

        return float(np.clip(snr, 5.0, 100.0))

    def _sample_target_snr(self, snr_regime: str = None) -> float:
        """
        Sample target SNR from specified regime or from distribution.
        
        Args:
            snr_regime: Specific regime ('low', 'medium', 'high') or None to sample from distribution
        
        Returns:
            Target SNR value
        """
        # New: accept optional event_type conditioning via self.conditional_snr
        # Signature backward compatible: snr_regime may be provided by caller.
        def _draw_from_regime(regime):
            snr_min, snr_max = self.snr_ranges[regime]
            return float(np.random.uniform(snr_min, snr_max))

        if snr_regime is not None:
            target_snr = _draw_from_regime(snr_regime)
            # Track statistics
            self.stats['snr_regimes'][snr_regime] = self.stats['snr_regimes'].get(snr_regime, 0) + 1
            return target_snr

        # If caller didn't ask for a specific regime, sample optionally conditioned on event_type
        # (caller can pass event_type by setting attribute self._sampling_event_type before calling
        # or by using the new helper sample_target_snr_for_event).
        event_type = getattr(self, '_sampling_event_type', None)
        if event_type and hasattr(self, 'conditional_snr') and event_type in self.conditional_snr:
            regimes = list(self.conditional_snr[event_type].keys())
            probs = [self.conditional_snr[event_type][r] for r in regimes]
            # numerical stability
            s = sum(probs)
            if s <= 0:
                regimes = list(self.snr_distribution.keys())
                probs = list(self.snr_distribution.values())
            else:
                probs = [p / s for p in probs]
            regime = np.random.choice(regimes, p=probs)
            target = _draw_from_regime(regime)
            self.stats['snr_regimes'][regime] = self.stats['snr_regimes'].get(regime, 0) + 1
            return float(target)

        # Fallback to global sampling
        snr_regime = self._sample_snr_regime()
        snr_min, snr_max = self.snr_ranges[snr_regime]
        target_snr = float(np.random.uniform(snr_min, snr_max))
        self.stats['snr_regimes'][snr_regime] = self.stats['snr_regimes'].get(snr_regime, 0) + 1
        return target_snr
    

    def _compute_target_snr_from_params(self, mass_1: float, mass_2: float, 
                                    luminosity_distance: float) -> float:
        """Compute SNR purely from independent mass and distance."""
        M_total = mass_1 + mass_2
        M_chirp = (mass_1 * mass_2)**(3/5) / M_total**(1/5)
        
        # Reference calibration (MUST match WaveformGenerator._crude_snr_rescale)
        target_snr = 75.0 * (M_chirp / 30.0)**(5/6) * (400.0 / luminosity_distance)
        return float(np.clip(target_snr, 5.0, 100.0))

            
    def _sample_distance_from_prior(self, event_type: str = 'BBH') -> float:
        d_min, d_max = self.distance_ranges.get(event_type, (10, 10000))

        # Log-uniform prior (standard for GW astronomy)
        log_d_min = np.log10(d_min)
        log_d_max = np.log10(d_max)
        log_d_L = np.random.uniform(log_d_min, log_d_max)
        d_L = 10.0 ** log_d_L

        return float(d_L)
 
    def _sample_distance_independently(self, event_type: str) -> float:
        """Distance sampling - COMPLETELY INDEPENDENT of mass."""
        d_min, d_max = self.distance_ranges.get(event_type, (10.0, 1000.0))
        # Use uniform sampling for all types to ensure perfect independence
        d_L = np.random.uniform(d_min, d_max)
        return float(d_L)



    def sample_bbh_parameters(self, snr_regime: Optional[str] = None, is_edge_case: bool = False) -> Dict:
        """
        Decoupled BBH sampler: intrinsic masses and redshifts sampled independently.
        Detector-frame masses computed from redshift, distance from cosmology.
        """
        # Sample intrinsic source-frame masses (independent of distance)
        m1_src = float(_sample_powerlaw(5.0, 100.0, alpha=2.3))
        m2_src = float(_sample_powerlaw(5.0, 100.0, alpha=2.3))
        m1_src += np.random.uniform(-0.01, 0.01)
        m2_src += np.random.uniform(-0.01, 0.01)
        mass_1_src, mass_2_src = max(m1_src, m2_src), min(m1_src, m2_src)

        # Sample redshift independently from comoving volume prior
        z = sample_redshift_from_comoving_volume(z_min=0.001, z_max=2.0)

        # Compute luminosity distance from cosmology
        d_L = compute_luminosity_distance_from_redshift(z)

        # Ensure distance is within reasonable bounds (GW detection range)
        d_min, d_max = self.distance_ranges.get("BBH", (100.0, 2000.0))
        d_L = float(np.clip(d_L, d_min, d_max))
        z = calculate_redshift(d_L)  # Recalculate z if distance was clipped

        # Compute detector-frame masses
        mass_1 = mass_1_src * (1 + z)
        mass_2 = mass_2_src * (1 + z)
        total_mass = mass_1 + mass_2
        chirp_mass = (mass_1 * mass_2)**(3 / 5) / (total_mass**(1 / 5))
        chirp_mass_src = (mass_1_src * mass_2_src)**(3 / 5) / ((mass_1_src + mass_2_src)**(1 / 5))

        # Sample target SNR (now independent of mass/distance)
        if snr_regime:
            target_snr = self._sample_target_snr(snr_regime)
        else:
            # Sample from overall distribution
            target_snr = self._sample_target_snr_from_distribution()

        # Spins and geometry (all uniform / isotropic)
        a1 = float(np.random.beta(2, 5))
        a2 = float(np.random.beta(2, 5))
        tilt1 = float(np.arccos(np.random.uniform(-1, 1)))
        tilt2 = float(np.arccos(np.random.uniform(-1, 1)))
        ra = float(np.random.uniform(0, 2 * np.pi))
        dec = float(np.arcsin(np.random.uniform(-1, 1)))
        theta_jn = float(np.arccos(np.random.uniform(-1, 1)))
        psi = float(np.random.uniform(0, np.pi))
        phase = float(np.random.uniform(0, 2 * np.pi))
        geocent_time = float(np.random.uniform(-0.1, 0.1))

        # Compute comoving distance
        d_C = calculate_comoving_distance(z) if z else d_L

        # Edge cases (handle carefully to maintain decoupling)
        edge_case_type = None
        if is_edge_case:
            if np.random.rand() < 0.5:
                edge_case_type = "extreme_mass_ratio"
                # Resample source-frame masses for extreme case
                mass_1_src = np.random.uniform(30, 80) / (1 + z)  # Convert back to source frame
                mass_2_src = np.random.uniform(5, 30) / (1 + z)
                mass_1 = mass_1_src * (1 + z)
                mass_2 = mass_2_src * (1 + z)
            else:
                edge_case_type = "short_bbh"
                # Resample source-frame masses for extreme case
                mass_1_src = np.random.uniform(60, 100) / (1 + z)
                mass_2_src = np.random.uniform(60, 100) / (1 + z)
                mass_1 = mass_1_src * (1 + z)
                mass_2 = mass_2_src * (1 + z)
            total_mass = mass_1 + mass_2
            chirp_mass = (mass_1 * mass_2)**(3 / 5) / (total_mass**(1 / 5))
            chirp_mass_src = (mass_1_src * mass_2_src)**(3 / 5) / ((mass_1_src + mass_2_src)**(1 / 5))

        params = {
            "mass_1": float(mass_1), "mass_2": float(mass_2),
            "mass_1_source": float(mass_1_src), "mass_2_source": float(mass_2_src),
            "total_mass": float(total_mass), "chirp_mass": float(chirp_mass),
            "chirp_mass_source": float(chirp_mass_src),
            "mass_ratio": float(mass_2 / mass_1),
            "symmetric_mass_ratio": float((mass_1 * mass_2) / total_mass**2),
            "a1": a1, "a2": a2, "tilt1": tilt1, "tilt2": tilt2,
            "phi12": float(np.random.uniform(0, 2 * np.pi)),
            "phi_jl": float(np.random.uniform(0, 2 * np.pi)),
            "luminosity_distance": float(d_L), "redshift": float(z), "comoving_distance": float(d_C),
            "ra": ra, "dec": dec, "theta_jn": theta_jn,
            "psi": psi, "phase": phase, "geocent_time": geocent_time,
            "f_lower": 20.0, "f_ref": 20.0, "approximant": "IMRPhenomD",
            "target_snr": target_snr,
            "network_snr": target_snr,
            "lambda_1": 0.0, "lambda_2": 0.0,
            "type": "BBH", "edge_case": is_edge_case, "edge_case_type": edge_case_type,
            "accepted": True
        }

        if snr_regime:
            params["requested_snr_range"] = self.snr_ranges[snr_regime]
        return params


    def sample_bns_parameters(self, snr_regime: Optional[str] = None, is_edge_case: bool = False) -> Dict:
        """Decoupled BNS sampler: intrinsic masses and redshifts sampled independently."""
        # Sample intrinsic source-frame masses (independent of distance)
        m1_src = float(np.clip(np.random.normal(1.35, 0.18), 1.0, 2.5))
        m2_src = float(np.clip(np.random.normal(1.25, 0.22), 1.0, 2.5))
        mass_1_src, mass_2_src = max(m1_src, m2_src), min(m1_src, m2_src)

        # Sample redshift independently from comoving volume prior
        z = sample_redshift_from_comoving_volume(z_min=0.001, z_max=1.0)  # BNS typically closer

        # Compute luminosity distance from cosmology
        d_L = compute_luminosity_distance_from_redshift(z)

        # Ensure distance is within reasonable bounds
        d_min, d_max = self.distance_ranges.get("BNS", (10.0, 1200.0))
        d_L = float(np.clip(d_L, d_min, d_max))
        z = calculate_redshift(d_L)  # Recalculate z if distance was clipped

        # Compute detector-frame masses
        mass_1 = mass_1_src * (1 + z)
        mass_2 = mass_2_src * (1 + z)
        total_mass = mass_1 + mass_2
        chirp_mass = (mass_1 * mass_2)**(3/5) / (total_mass**(1/5))
        chirp_mass_src = (mass_1_src * mass_2_src)**(3/5) / ((mass_1_src + mass_2_src)**(1/5))

        # Sample target SNR (now independent of mass/distance)
        if snr_regime:
            target_snr = self._sample_target_snr(snr_regime)
        else:
            # Sample from overall distribution
            target_snr = self._sample_target_snr_from_distribution()

        theta_jn = float(np.arccos(np.random.uniform(-1, 1)))
        ra = float(np.random.uniform(0, 2*np.pi))
        dec = float(np.arcsin(np.random.uniform(-1, 1)))
        psi = float(np.random.uniform(0, np.pi))
        phase = float(np.random.uniform(0, 2*np.pi))
        geocent_time = float(np.random.uniform(-0.1, 0.1))
        a1 = float(np.random.uniform(0.0, 0.05))
        a2 = float(np.random.uniform(0.0, 0.05))
        f_lower = 25.0 if is_edge_case else 35.0

        # Compute comoving distance
        d_C = calculate_comoving_distance(z) if z else d_L

        params = {
            "mass_1": mass_1, "mass_2": mass_2,
            "mass_1_source": mass_1_src, "mass_2_source": mass_2_src,
            "total_mass": total_mass, "chirp_mass": chirp_mass,
            "chirp_mass_source": chirp_mass_src,
            "mass_ratio": mass_2 / mass_1, "symmetric_mass_ratio": (mass_1 * mass_2) / total_mass**2,
            "luminosity_distance": d_L, "redshift": z, "comoving_distance": d_C,
            "a1": a1, "a2": a2, "tilt1": 0.0, "tilt2": 0.0,
            "ra": ra, "dec": dec, "theta_jn": theta_jn,
            "psi": psi, "phase": phase,
            "f_lower": f_lower, "f_ref": 50.0, "approximant": "IMRPhenomD_NRTidal",
            "lambda_1": float(np.clip(np.random.lognormal(np.log(400), 0.7) * (1.4/mass_1_src)**5, 50, 5000)),  # Use source mass
            "lambda_2": float(np.clip(np.random.lognormal(np.log(400), 0.7) * (1.4/mass_2_src)**5, 50, 5000)),  # Use source mass
            "type": "BNS", "edge_case": is_edge_case,
            "edge_case_type": "long_inspiral" if is_edge_case else None,
            "geocent_time": geocent_time,
            "target_snr": target_snr,
            "network_snr": target_snr,
            "accepted": True
        }

        if snr_regime:
            params["requested_snr_range"] = self.snr_ranges[snr_regime]
        return params

    def sample_nsbh_parameters(self, snr_regime: Optional[str] = None, is_edge_case: bool = False) -> Dict:
        """Decoupled NSBH sampler: intrinsic masses and redshifts sampled independently."""
        # Sample intrinsic source-frame masses (independent of distance)
        ns_mass_src = float(np.clip(np.random.normal(1.35, 0.12), 1.0, 2.5))
        bh_mass_src = float(_sample_powerlaw(3.0, 100.0, alpha=2.3))
        mass_1_src, mass_2_src = max(bh_mass_src, ns_mass_src), min(bh_mass_src, ns_mass_src)

        # Sample redshift independently from comoving volume prior
        z = sample_redshift_from_comoving_volume(z_min=0.001, z_max=1.5)  # NSBH intermediate range

        # Compute luminosity distance from cosmology
        d_L = compute_luminosity_distance_from_redshift(z)

        # Ensure distance is within reasonable bounds
        d_min, d_max = self.distance_ranges.get("NSBH", (20.0, 1600.0))
        d_L = float(np.clip(d_L, d_min, d_max))
        z = calculate_redshift(d_L)  # Recalculate z if distance was clipped

        # Compute detector-frame masses
        mass_1 = mass_1_src * (1 + z)
        mass_2 = mass_2_src * (1 + z)
        total_mass = mass_1 + mass_2
        chirp_mass = (mass_1 * mass_2)**(3/5) / (total_mass**(1/5))
        chirp_mass_src = (mass_1_src * mass_2_src)**(3/5) / ((mass_1_src + mass_2_src)**(1/5))

        # Sample target SNR (now independent of mass/distance)
        if snr_regime:
            target_snr = self._sample_target_snr(snr_regime)
        else:
            # Sample from overall distribution
            target_snr = self._sample_target_snr_from_distribution()

        a1 = float(np.random.uniform(0.0, 0.99))
        a2 = float(np.random.uniform(0.0, 0.05))
        tilt1 = float(np.random.uniform(0, np.pi/3))
        tilt2 = 0.0
        ra = float(np.random.uniform(0, 2*np.pi))
        dec = float(np.arcsin(np.random.uniform(-1, 1)))
        theta_jn = float(np.arccos(np.random.uniform(-1, 1)))
        psi = float(np.random.uniform(0, np.pi))
        phase = float(np.random.uniform(0, 2*np.pi))

        # Compute comoving distance
        d_C = calculate_comoving_distance(z) if z else d_L

        params = {
            "mass_1": mass_1, "mass_2": mass_2,
            "mass_1_source": mass_1_src, "mass_2_source": mass_2_src,
            "total_mass": total_mass, "chirp_mass": chirp_mass,
            "chirp_mass_source": chirp_mass_src,
            "mass_ratio": mass_2 / mass_1, "symmetric_mass_ratio": (mass_1 * mass_2) / total_mass**2,
            "luminosity_distance": d_L, "redshift": z, "comoving_distance": d_C,
            "a1": a1, "a2": a2, "tilt1": tilt1, "tilt2": tilt2,
            "phi12": float(np.random.uniform(0, 2*np.pi)), "phi_jl": float(np.random.uniform(0, 2*np.pi)),
            "ra": ra, "dec": dec, "theta_jn": theta_jn, "psi": psi, "phase": phase,
            "geocent_time": 0.0, "f_lower": 20.0, "f_ref": 50.0,
            "approximant": "IMRPhenomPv2_NRTidal" if total_mass <= 6.0 else "IMRPhenomPv2",
            "type": "NSBH", "edge_case": is_edge_case,
            "edge_case_type": "extreme_mass" if is_edge_case else None,
            "target_snr": target_snr,
            "network_snr": target_snr,
            "accepted": True
        }

        if snr_regime:
            params["requested_snr_range"] = self.snr_ranges[snr_regime]
        return params

    def recompute_target_snr_from_params(self, params: Dict, overwrite: bool = True) -> float:
        """
        Recompute target_snr from params using canonical chirp-mass scaling.
        If overwrite==False, do not write back to params (just return computed value).
        """
        # set reference defaults from self if available
        snr_ref = getattr(self, 'reference_snr', params.get('_snr_ref', 25.0))
        reference_mass = getattr(self, 'reference_mass', params.get('_reference_mass', 30.0))
        reference_distance = getattr(self, 'reference_distance', params.get('_reference_distance', 100.0))
        computed = compute_physical_snr_from_params(
            params,
            snr_ref=snr_ref,
            reference_mass=reference_mass,
            reference_distance=reference_distance
        )
        if overwrite:
            params['target_snr'] = computed
        return computed

    
    def calibrate_snr_by_event_type(self, n_samples: int = 2000, random_seed: int = None) -> Dict:
        """
        Empirically estimate P(snr_regime | event_type) using the same sampling
        priors used by the sampler. For BBH use power-law draws (no lognormal clustering).
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        conditional = {}
        for et in ['BBH', 'BNS', 'NSBH']:
            counts = {r: 0 for r in self.snr_ranges.keys()}
            for i in range(n_samples):
                if et == 'BBH':
                    # power-law primary and secondary (independent draws)
                    m1 = _sample_powerlaw(5.0, 100.0, alpha=2.3)
                    m2 = _sample_powerlaw(5.0, 100.0, alpha=2.3)
                    if m2 > m1:
                        m1, m2 = m2, m1
                    dmin, dmax = self.distance_ranges['BBH']
                elif et == 'BNS':
                    m1 = float(np.clip(np.random.normal(1.40, 0.15), 1.0, 2.5))
                    m2 = float(np.clip(np.random.normal(1.40, 0.20), 1.0, 2.5))
                    if m2 > m1:
                        m1, m2 = m2, m1
                    dmin, dmax = self.distance_ranges['BNS']
                else:  # NSBH
                    ns_mass = float(np.random.uniform(1.2, 2.0))
                    bh_mass = float(np.clip(_sample_powerlaw(3.0, 100.0, alpha=2.3), 3.0, 100.0))
                    m1, m2 = bh_mass, ns_mass
                    dmin, dmax = self.distance_ranges['NSBH']

                # sample distance with volume weighting (realistic)
                u = np.random.random()
                d = (dmin**3 + u * (dmax**3 - dmin**3))**(1/3)

                # chirp and SNR estimate
                M_total = m1 + m2
                M_chirp = (m1 * m2)**(3/5) / M_total**(1/5)
                snr = self.reference_snr * (M_chirp / self.reference_mass)**(5/6) * (self.reference_distance / d)

                # categorize
                overall_min = min(r[0] for r in self.snr_ranges.values())
                overall_max = max(r[1] for r in self.snr_ranges.values())
                if snr < overall_min:
                    counts['weak'] += 1
                elif snr >= overall_max:
                    counts['loud'] += 1
                else:
                    for regime, (mn, mx) in self.snr_ranges.items():
                        if mn <= snr < mx:
                            counts[regime] += 1
                            break

            total = float(n_samples)
            conditional[et] = {r: counts[r] / total for r in counts}

        self.conditional_snr = conditional
        return conditional




    def empirical_calibrate(self, n_samples: int = 2000, random_seed: int = None) -> Dict:
        """Empirically estimate P(snr_regime | event_type) by using the sampler's
        own sampling routines. This avoids model/ordering mismatches between a
        separate forward model and the actual sampler logic.

        This method temporarily clears any existing `self.conditional_snr` to
        avoid recursive conditioning during the calibration run, and restores
        sampler stats after completion.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Save and restore stats to avoid polluting sampler state
        saved_stats = None
        try:
            saved_stats = dict(self.stats)
        except Exception:
            saved_stats = None

        # Temporarily remove any existing conditional map
        prev_cond = getattr(self, 'conditional_snr', None)
        if hasattr(self, 'conditional_snr'):
            try:
                delattr(self, 'conditional_snr')
            except Exception:
                self.conditional_snr = None

        regimes = list(self.snr_ranges.keys())
        conditional = {}

        for et in ['BBH', 'BNS', 'NSBH']:
            counts = {r: 0 for r in regimes}
            for i in range(n_samples):
                try:
                    if et == 'BBH':
                        params = self.sample_bbh_parameters(None, False)
                    elif et == 'BNS':
                        params = self.sample_bns_parameters(None, False)
                    else:
                        params = self.sample_nsbh_parameters(None, False)

                    if 'target_snr' in params and not np.isnan(params['target_snr']):
                        snr = float(params['target_snr'])
                    else:
                        snr = compute_physical_snr_from_params(
                            params,
                            snr_ref=self.reference_snr,
                            reference_mass=self.reference_mass,
                            reference_distance=self.reference_distance
                        )
                except Exception:
                    snr = float('nan')

                # Categorize
                categorized = False
                for r, (mn, mx) in self.snr_ranges.items():
                    try:
                        if mn <= snr < mx:
                            counts[r] += 1
                            categorized = True
                            break
                    except Exception:
                        continue

                if not categorized:
                    # Out-of-range values map to weak/loud
                    mins = [rng[0] for rng in self.snr_ranges.values()]
                    maxs = [rng[1] for rng in self.snr_ranges.values()]
                    overall_min = min(mins)
                    overall_max = max(maxs)
                    if np.isnan(snr) or snr < overall_min:
                        counts['weak'] += 1
                    elif snr >= overall_max:
                        counts['loud'] += 1

            total = float(n_samples)
            conditional[et] = {r: counts[r] / total for r in counts}

        # Store calibration
        self.conditional_snr = conditional

        # Restore saved stats to avoid observer effects
        if saved_stats is not None:
            try:
                self.stats = saved_stats
            except Exception:
                pass

        # Restore previous conditional if caller expects no persistent change
        # (we intentionally keep the new empirical map by default)
        return conditional

    def event_type_given_snr(self, snr_regime: str):
        """Return a sampled event type conditioned on an SNR regime.

        Uses the empirical conditional map `self.conditional_snr` when available
        and the configured `self.event_type_distribution` as priors. Falls back
        to the prior distribution if no calibration is present.
        """
        types = list(self.event_type_distribution.keys())

        # If we have an empirical conditional P(snr|type), invert with Bayes:
        # P(type|snr) ∝ P(snr|type) * P(type)
        if hasattr(self, 'conditional_snr') and self.conditional_snr:
            weights = []
            for t in types:
                p_reg_given_t = float(self.conditional_snr.get(t, {}).get(snr_regime, 0.0))
                p_t = float(self.event_type_distribution.get(t, 0.0))
                weights.append(p_reg_given_t * p_t)
            s = sum(weights)
            if s <= 0:
                probs = [self.event_type_distribution.get(t, 0.0) for t in types]
                s2 = sum(probs)
                if s2 <= 0:
                    probs = [1.0 / len(types)] * len(types)
                else:
                    probs = [p / s2 for p in probs]
            else:
                probs = [w / s for w in weights]
            return np.random.choice(types, p=probs)

        # Fallback to prior event-type distribution
        probs = [self.event_type_distribution.get(t, 0.0) for t in types]
        s = sum(probs)
        if s <= 0:
            probs = [1.0 / len(types)] * len(types)
        else:
            probs = [p / s for p in probs]
        return np.random.choice(types, p=probs)


# ================================================================================
# WAVEFORM_GENERATOR.PY
# ================================================================================

"""
Waveform Generator
Comprehensive GW signal generation with PyCBC and analytical fallbacks
Extracted from EnhancedWaveformGenerator class
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from scipy.signal import windows

try:
    from pycbc.waveform import get_td_waveform, get_fd_waveform
    from pycbc.detector import Detector
    from pycbc.types import TimeSeries, FrequencySeries
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False


class WaveformGenerator:
    """
    Generate gravitational waveforms with comprehensive fallbacks
    Supports BBH, BNS, NSBH with tidal and precession effects
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, duration: float = DURATION):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors
        self.detectors = {}
        if PYCBC_AVAILABLE:
            for det_name in ['H1', 'L1', 'V1']:
                try:
                    self.detectors[det_name] = Detector(det_name)
                except Exception as e:
                    self.logger.debug(f"Detector {det_name} initialization failed: {e}")
   
    def generate_waveform(self, params: Dict, detector_name: str = None, psd: np.ndarray = None, allow_waveform_rescale: bool = False) -> np.ndarray:
        """
        Generate waveform and DO NOT silently rescale physical parameters to hit target_snr.
        If allow_waveform_rescale==True (legacy) and psd provided, rescaling will be applied.
        """
        used_pycbc = False
        signal = None
        if PYCBC_AVAILABLE:
            try:
                signal = self.generate_pycbc_waveform(params, detector_name)
                used_pycbc = True
            except Exception:
                signal = None

        if signal is None:
            try:
                signal = self.generate_analytical_waveform(params, detector_name)
            except Exception:
                try:
                    signal = self.generate_simple_chirp_fallback(params, detector_name)
                except Exception:
                    signal = np.zeros(self.n_samples, dtype=np.float32)

        # LEGACY: only rescale if explicitly allowed by caller
        target_snr = params.get('target_snr', None)
        if allow_waveform_rescale and (target_snr is not None):
            if used_pycbc:
                # PyCBC accepts distance etc.; prefer not to rescale
                pass
            else:
                if psd is not None:
                    try:
                        signal = self.rescale_to_target_snr(signal, psd, target_snr)
                    except Exception:
                        # fallback: crude rescale
                        signal = self._crude_snr_rescale(signal, target_snr, params)
                else:
                    signal = self._crude_snr_rescale(signal, target_snr, params)

        # Ensure length and dtype
        sig = np.asarray(signal, dtype=np.float32)
        if len(sig) < self.n_samples:
            sig = np.pad(sig, (0, self.n_samples - len(sig)), mode='constant')
        else:
            sig = sig[:self.n_samples]
        return sig

    def _crude_snr_rescale(self, signal: np.ndarray, target_snr: float, params: Dict) -> np.ndarray:
        """
        Fallback crude scaling (legacy) — kept for compatibility but not used by default.
        """
        chirp_mass = params.get('chirp_mass', 30.0)
        distance = params.get('luminosity_distance', 400.0)
        reference_snr = 75.0
        reference_mass = 30.0
        reference_distance = 400.0
        expected_snr = reference_snr * (chirp_mass / reference_mass)**(5/6) * (reference_distance / distance)
        if expected_snr > 0:
            scale_factor = float(target_snr) / float(expected_snr)
        else:
            scale_factor = float(target_snr) / 10.0
        return signal * float(scale_factor)

    def generate_pycbc_waveform(self, params: Dict, detector_name: str = None) -> np.ndarray:
        """Generate waveform using PyCBC with detector projection"""
        
        # Generate polarizations
        hp, hc = get_td_waveform(
            approximant=params.get('approximant', 'IMRPhenomD'),
            mass1=params['mass_1'],
            mass2=params['mass_2'],
            spin1z=params.get('a1', 0.0) * np.cos(params.get('tilt1', 0.0)),
            spin2z=params.get('a2', 0.0) * np.cos(params.get('tilt2', 0.0)),
            distance=params.get('luminosity_distance', 400.0),
            inclination=params.get('theta_jn', 0.0),
            coa_phase=params.get('phase', 0.0),
            delta_t=1.0/self.sample_rate,
            f_lower=params.get('f_lower', 20.0),
            lambda1=params.get('lambda_1', 0.0),
            lambda2=params.get('lambda_2', 0.0)
        )
        
        # Project onto detector if specified
        if detector_name and detector_name in self.detectors:
            detector = self.detectors[detector_name]
            fp, fc = detector.antenna_pattern(
                params.get('ra', 0.0),
                params.get('dec', 0.0),
                params.get('psi', 0.0),
                params.get('geocent_time', 0.0)
            )
            signal = fp * hp + fc * hc
        else:
            # No detector projection - use plus polarization
            signal = hp
        
        # Resize to match duration
        if len(signal) < self.n_samples:
            signal.resize(self.n_samples)
        else:
            signal = signal[:self.n_samples]
        
        # Convert to numpy array
        return np.array(signal.data, dtype=np.float32)
    
    def generate_analytical_waveform(self, params: Dict, detector_name: str = None) -> np.ndarray:
        """Generate waveform using analytical post-Newtonian methods"""
        
        # Time array
        t = np.linspace(-self.duration/2, self.duration/2, self.n_samples)
        tc = params.get('geocent_time', 0.0)
        time_to_merger = np.maximum(tc - t, 0.001)
        
        # Determine waveform type
        if params.get('lambda_1', 0) > 0 or params.get('lambda_2', 0) > 0:
            signal = self.generate_tidal_waveform(t, time_to_merger, params)
        elif params.get('approximant', '').find('Pv2') >= 0:
            signal = self.generate_precessing_waveform(t, time_to_merger, params)
        else:
            signal = self.generate_aligned_spin_waveform(t, time_to_merger, params)
        
        # Apply detector response
        if detector_name:
            response = self.calculate_detector_response(params, detector_name)
            signal *= response
        
        # Apply taper window
        window = windows.tukey(self.n_samples, alpha=0.1)
        signal *= window
        
        return signal.astype(np.float32)
    
    def generate_aligned_spin_waveform(self, t: np.ndarray, 
                                time_to_merger: np.ndarray, 
                                params: Dict) -> np.ndarray:
        """
        Generate aligned-spin BBH waveform using 3.5PN TaylorT4.
        
         CORRECTED: Amplitude gives SNR ~ 15 for 30 Msun at 400 Mpc
        
        Args:
            t: Time array
            time_to_merger: Time until merger for each sample
            params: Signal parameters dict
        
        Returns:
            Waveform in physical units (strain)
        """
        
        # System parameters
        m1 = params['mass_1']
        m2 = params['mass_2']
        total_mass = m1 + m2
        chirp_mass = params['chirp_mass']
        eta = params.get('symmetric_mass_ratio', (m1 * m2) / total_mass**2)
        
        # Spin parameters
        chi1 = params.get('a1', 0.0) * np.cos(params.get('tilt1', 0.0))
        chi2 = params.get('a2', 0.0) * np.cos(params.get('tilt2', 0.0))
        chi_eff = (m1 * chi1 + m2 * chi2) / total_mass
        
        # Frequency evolution (3.5PN TaylorT4)
        theta = time_to_merger / (5.0 * chirp_mass)
        theta = np.maximum(theta, 1e-10)
        v = theta**(-1/8)
        
        # Phase evolution (3.5PN with spin corrections)
        psi = -(1/eta) * (
            v**(-5) +
            (3715/1008 + 55*eta/12) * v**(-3) +
            (-10*np.pi + (113/3 + 19*eta/3)*chi_eff) * v**(-2) +
            (15293365/1016064 + 27145*eta/1008 + 3085*eta**2/144) * v**(-1)
        )
        
        # Frequency
        frequency = v**3 / (8*np.pi*chirp_mass)
        
        #  AMPLITUDE CALIBRATION
        distance = params.get('luminosity_distance', 400.0)
        amplitude = (chirp_mass**(5/6) / distance) * frequency**(-7/6)
        
        #  CRITICAL: Empirical scale factor tuned to match target SNR distribution
        # 5e-23 gives SNR ~ 75 for 30 Msun at 400 Mpc (adjusted scaling)
        amplitude *= 5e-23

        # Apply inclination
        theta_jn = params.get('theta_jn', 0.0)
        amplitude *= (1 + np.cos(theta_jn)**2) / 2
        
        # Generate strain
        phase_total = 2*np.pi*np.cumsum(frequency) / self.sample_rate
        strain = amplitude * np.sin(phase_total + psi)
        
        return strain.astype(np.float32)


    def generate_tidal_waveform(self, t: np.ndarray, 
                           time_to_merger: np.ndarray, 
                           params: Dict) -> np.ndarray:
        
        """Generate BNS/NSBH waveform with tidal corrections."""
        
        # Base point-particle inspiral
        base_signal = self.generate_aligned_spin_waveform(t, time_to_merger, params)
        
        # Tidal deformability
        lambda_tilde = params.get('lambda_tilde', 
                                params.get('lambda_1', 0) + params.get('lambda_2', 0))
        
        if lambda_tilde > 0:
            # Frequency for tidal correction
            chirp_mass = params['chirp_mass']
            v = (time_to_merger / (5.0 * chirp_mass))**(-1/8)
            frequency = v**3 / (8*np.pi*chirp_mass)
            
            # Leading tidal phase correction (5PN)
            x = (np.pi * chirp_mass * frequency)**(2/3)
            
            #  FIX: Compute average tidal phase for frequency-domain correction
            # Time-domain phase shift → single frequency-domain phase value
            avg_tidal_phase = np.mean(-(39/2) * lambda_tilde * x**5 / chirp_mass**5)
            
            # Apply correction in frequency domain
            base_signal_fft = np.fft.fft(base_signal)
            freqs = np.fft.fftfreq(len(base_signal), 1/self.sample_rate)
            
            #  Phase correction: scalar applied to positive frequencies
            phase_correction = np.ones_like(freqs, dtype=complex)
            phase_correction[freqs > 0] = np.exp(-1j * avg_tidal_phase)
            
            corrected_fft = base_signal_fft * phase_correction
            return np.real(np.fft.ifft(corrected_fft)).astype(np.float32)
        
        return base_signal

        
    
    def generate_precessing_waveform(self, t: np.ndarray,
                                time_to_merger: np.ndarray, 
                                params: Dict) -> np.ndarray:
        """Generate precessing waveform with proper precession frequency."""
        
        # Base aligned-spin waveform
        base = self.generate_aligned_spin_waveform(t, time_to_merger, params)
        
        # Precession parameters
        tilt1 = params.get('tilt1', 0.0)
        tilt2 = params.get('tilt2', 0.0)
        phi12 = params.get('phi12', 0.0)
        
        # Only apply precession if tilts are significant
        if max(tilt1, tilt2) > 0.1:  # Significant precession
            # System parameters
            m1 = params['mass_1']
            m2 = params['mass_2']
            total_mass = m1 + m2
            chirp_mass = params['chirp_mass']
            mass_ratio = m2 / m1
            
            # FIX:  Proper precession frequency
            # Precession frequency ~ Ω_orb * (S_perp / L)
            # For leading order: Ω_prec ~ (2 + 3q/2) / (1 + q) * Ω_orb
            
            v = (time_to_merger / (5.0 * chirp_mass))**(-1/8)
            orbital_freq = v**3 / (8*np.pi*chirp_mass)
            
            # Precession frequency factor
            prec_factor = (2.0 + 1.5*mass_ratio) / (1.0 + mass_ratio)
            prec_freq = prec_factor * orbital_freq
            
            # Precession amplitude depends on spin perpendicular components
            a1 = params.get('a1', 0.0)
            a2 = params.get('a2', 0.0)
            
            # Effective precessing spin χ_p
            chi_p = max(
                a1 * np.sin(tilt1),
                a2 * (4.0 + 3.0*mass_ratio) / (4.0 + 3.0/mass_ratio) * np.sin(tilt2)
            )
            
            # Modulation amplitude (0 to ~0.4 for strong precession)
            mod_amplitude = 0.4 * np.tanh(2.0 * chi_p)  # Smooth saturation
            
            # Apply precession modulation
            # Both amplitude and phase modulation
            amp_modulation = 1.0 + mod_amplitude * np.cos(2*np.pi*prec_freq*t + phi12)
            phase_modulation = mod_amplitude * 0.5 * np.sin(2*np.pi*prec_freq*t + phi12)
            
            base *= amp_modulation
            base = np.roll(base, int(np.mean(phase_modulation) * self.sample_rate / (2*np.pi)))
        
        return base

    
    def generate_simple_chirp_fallback(self, params: Dict, detector_name: str = None) -> np.ndarray:
        """Ultimate fallback: simple chirp"""
        
        t = np.linspace(0, self.duration, self.n_samples)
        
        # Chirp parameters
        m1 = params.get('mass_1', 30.0)
        m2 = params.get('mass_2', 30.0)
        chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        
        # Frequency sweep
        f_start = 35.0
        f_end = min(250.0, self.sample_rate / 4)
        frequency = f_start + (f_end - f_start) * (t / self.duration)**3
        
        # Amplitude
        distance = params.get('luminosity_distance', 400.0)
        amplitude = 2.5e-21 * (chirp_mass / 30.0)**(5/6) * (400.0 / distance)
        
        # Envelope
        envelope = np.exp(-t / (self.duration * 0.4))
        
        # Phase
        dt = 1.0 / self.sample_rate
        phase = 2 * np.pi * np.cumsum(frequency) * dt
        
        return amplitude * envelope * np.sin(phase)
    
    def calculate_detector_response(self, params: Dict, detector_name: str) -> float:
        """Calculate detector antenna response"""
        
        if detector_name in self.detectors:
            detector = self.detectors[detector_name]
            fp, fc = detector.antenna_pattern(
                params.get('ra', 0.0),
                params.get('dec', 0.0),
                params.get('psi', 0.0),
                params.get('geocent_time', 0.0)
            )
            return float(np.sqrt(fp**2 + fc**2))
        else:
            # Fallback: average response
            return 0.4
    
    def get_alternative_approximants(self, params: Dict) -> list:
        """Get fallback approximants based on event type"""
        
        event_type = params.get('type', 'BBH')
        approx_dict = APPROXIMANTS.get(event_type, {'non_precessing': ['IMRPhenomD']})
        
        alternatives = []
        for category in ['non_precessing', 'precessing', 'tidal']:
            alternatives.extend(approx_dict.get(category, []))
        
        # Remove current approximant
        current = params.get('approximant')
        if current in alternatives:
            alternatives.remove(current)
        
        return alternatives

    def _compute_optimal_snr(self, waveform, psd_dict):
        """Compute optimal matched-filter SNR - FIXED"""
        from numpy.fft import rfft, rfftfreq
        
        N = len(waveform)
        dt = 1.0 / self.sample_rate
        freqs = rfftfreq(N, dt)
        wf_fft = rfft(waveform)
        
        psd_freqs = psd_dict['frequencies']
        psd = psd_dict['psd']
        
        # FIX: Proper extrapolation (use edge values, don't extrapolate)
        psd_interp = np.interp(freqs, psd_freqs, psd, 
                            left=psd[0], right=psd[-1])
        
        # FIX: Reasonable floor (1e-20 is typical LIGO PSD minimum, NOT 1e-50!)
        min_psd = np.min(psd[psd > 0]) if np.any(psd > 0) else 1e-20
        psd_floor = max(min_psd * 0.1, 1e-20)
        psd_interp = np.maximum(psd_interp, psd_floor)
        
        # FIX: Validation
        if np.any(np.isnan(psd_interp)) or np.any(np.isinf(psd_interp)):
            self.logger.warning("PSD has NaN/inf; using default SNR=15.0")
            return 15.0
        
        delta_f = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        # Ensure no division by zero
        safe_psd = np.where(psd_interp > 0, psd_interp, 1e-20)
        integrand = np.abs(wf_fft)**2 / safe_psd
        snr_squared = 4 * np.sum(integrand) * delta_f
        snr = np.sqrt(snr_squared)
        
        # FIX: Clip to reasonable range
        snr = np.clip(snr, 1.0, 100.0)
        
        if np.isnan(snr) or np.isinf(snr):
            return 15.0
        
        return float(snr)



    def rescale_to_target_snr(self, signal: np.ndarray, psd: np.ndarray, 
                            target_snr: float) -> np.ndarray:
        """
        Rescale waveform to match target SNR.
        
        Args:
            signal: Time-domain waveform
            psd: Power spectral density
            target_snr: Desired SNR
        
        Returns:
            Rescaled waveform
        """
        current_snr = self.compute_optimal_snr(signal, psd)
        
        if current_snr > 0:
            scale_factor = target_snr / current_snr
            return signal * scale_factor
        else:
            # Fallback: use crude amplitude scaling
            self.logger.warning("SNR computation failed, using amplitude scaling")
            desired_amplitude = target_snr * 1e-21
            current_amplitude = np.max(np.abs(signal))
            if current_amplitude > 0:
                return signal * (desired_amplitude / current_amplitude)
        
        return signal


# ================================================================================
# NOISE_GENERATOR.PY
# ================================================================================

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


# ================================================================================
# PREPROCESSING.PY
# ================================================================================

"""
Data Preprocessing Module
Whitening, bandpass filtering, and quality validation
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from scipy.signal import butter, filtfilt, welch, windows
from scipy.interpolate import interp1d


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
            psd_interp = np.maximum(psd_interp, 1e-20)
            
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


# ================================================================================
# GWTC_LOADER.PY
# ================================================================================

"""
GWTC Catalog Loader
Load real gravitational wave events from GWTC catalogs
"""

import numpy as np
import pandas as pd
import logging
import requests
import json
from typing import Dict, List, Optional
from pathlib import Path

try:
    from gwpy.timeseries import TimeSeries
    GWPY_AVAILABLE = True
except ImportError:
    GWPY_AVAILABLE = False

class GWTCLoader:
    """
    Load and process real GW event data from GWTC catalogs
    Supports GWTC-1, GWTC-2, GWTC-3, GWTC-4
    """
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.gwosc_base_url = "https://gwosc.org"
    
    def get_gwtc_events(self, catalog: str = "GWTC-4") -> pd.DataFrame:
        """
        Fetch all events from GWTC catalog
        
        Args:
            catalog: Catalog name (GWTC-1, GWTC-2, GWTC-3, GWTC-4, or all)
            
        Returns:
            DataFrame with event parameters
        """
        
        endpoints = [
            f"https://gwosc.org/eventapi/json/{catalog}/",
            "https://gwosc.org/eventapi/json/GWTC-4.0/",
            "https://gwosc.org/eventapi/json/confident/",
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    events_data = response.json()
                    
                    if 'events' in events_data:
                        return self._parse_events_dict(events_data['events'])
                    elif isinstance(events_data, list):
                        return self._parse_events_list(events_data)
                        
            except requests.RequestException as e:
                self.logger.debug(f"Failed to fetch from {endpoint}: {e}")
                continue
        
        # Fallback to hardcoded high-confidence events
        self.logger.warning("API fetch failed, using hardcoded events")
        return self._get_hardcoded_events()
    
    def _parse_events_dict(self, events_dict: Dict) -> pd.DataFrame:
        """Parse GWOSC old dict format"""
        
        events_list = []
        for event_name, info in events_dict.items():
            events_list.append({
                'event_name': event_name,
                'gps_time': info.get('GPS', 0.0),
                'mass_1_source': info.get('mass_1_source', 0.0),
                'mass_2_source': info.get('mass_2_source', 0.0),
                'chirp_mass_source': info.get('chirp_mass_source', 0.0),
                'total_mass_source': info.get('total_mass_source', 0.0),
                'luminosity_distance': info.get('luminosity_distance', 0.0),
                'redshift': info.get('redshift', 0.0),
                'network_snr': info.get('network_matched_filter_snr', 0.0),
                'far': info.get('far', 1e10),
                'observing_run': info.get('observing_run', ''),
                'detectors': info.get('detectors', [])
            })
        
        return pd.DataFrame(events_list)
    
    def _parse_events_list(self, events_list: List) -> pd.DataFrame:
        """Parse GWOSC new list format"""
        
        parsed = []
        for event in events_list:
            if isinstance(event, dict):
                parsed.append({
                    'event_name': event.get('name', 'Unknown'),
                    'gps_time': event.get('GPS', event.get('gps_time', 0.0)),
                    'mass_1_source': event.get('mass_1_source', 0.0),
                    'mass_2_source': event.get('mass_2_source', 0.0),
                    'chirp_mass_source': event.get('chirp_mass_source', 0.0),
                    'total_mass_source': event.get('total_mass_source', 0.0),
                    'luminosity_distance': event.get('luminosity_distance', 0.0),
                    'redshift': event.get('redshift', 0.0),
                    'network_snr': event.get('network_matched_filter_snr', 0.0),
                    'far': event.get('far', 1e10),
                    'observing_run': event.get('observing_run', ''),
                    'detectors': event.get('detectors', [])
                })
        
        return pd.DataFrame(parsed)
    
    def _get_hardcoded_events(self) -> pd.DataFrame:
        """Hardcoded high-confidence GWTC events"""
        
        events = [
            {
                'event_name': 'GW150914', 'gps_time': 1126259462.4,
                'mass_1_source': 36.2, 'mass_2_source': 29.1,
                'chirp_mass_source': 30.5, 'total_mass_source': 65.3,
                'luminosity_distance': 410.0, 'redshift': 0.09,
                'network_snr': 23.7, 'far': 2.0e-7,
                'observing_run': 'O1', 'detectors': ['H1', 'L1']
            },
            {
                'event_name': 'GW151226', 'gps_time': 1135136350.6,
                'mass_1_source': 14.2, 'mass_2_source': 7.5,
                'chirp_mass_source': 8.9, 'total_mass_source': 21.8,
                'luminosity_distance': 440.0, 'redshift': 0.09,
                'network_snr': 13.0, 'far': 1.0e-6,
                'observing_run': 'O1', 'detectors': ['H1', 'L1']
            },
            {
                'event_name': 'GW170817', 'gps_time': 1187008882.4,
                'mass_1_source': 1.6, 'mass_2_source': 1.2,
                'chirp_mass_source': 1.2, 'total_mass_source': 2.8,
                'luminosity_distance': 40.0, 'redshift': 0.009,
                'network_snr': 32.4, 'far': 1.0e-9,
                'observing_run': 'O2', 'detectors': ['H1', 'L1', 'V1']
            },
            {
                'event_name': 'GW190521', 'gps_time': 1242442967.4,
                'mass_1_source': 85.0, 'mass_2_source': 66.0,
                'chirp_mass_source': 72.0, 'total_mass_source': 151.0,
                'luminosity_distance': 5300.0, 'redshift': 0.82,
                'network_snr': 14.7, 'far': 1.4e-4,
                'observing_run': 'O3a', 'detectors': ['H1', 'L1', 'V1']
            },
        ]
        
        return pd.DataFrame(events)
    
    def download_strain(self,
                       event_name: str,
                       detector: str = 'H1',
                       duration: int = 4,
                       sample_rate: int = 4096) -> Optional[np.ndarray]:
        """Download strain data for event"""
        
        if not GWPY_AVAILABLE:
            self.logger.error("gwpy not available for strain download")
            return None
        
        try:
            # Get event GPS time
            gps_time = self._get_event_gps_time(event_name)
            if gps_time is None:
                return None
            
            start_time = gps_time - duration // 2
            end_time = gps_time + duration // 2
            
            strain = TimeSeries.fetch_open_data(
                detector, start_time, end_time,
                sample_rate=sample_rate
            )
            
            return np.array(strain.value, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Strain download failed for {event_name}: {e}")
            return None
    
    def _get_event_gps_time(self, event_name: str) -> Optional[float]:
        """Get GPS time for event"""
        
        try:
            response = requests.get(
                f"https://gwosc.org/eventapi/json/{event_name}/",
                timeout=5
            )
            if response.status_code == 200:
                return response.json().get('GPS')
        except:
            pass
        
        # Fallback to hardcoded
        events_df = self._get_hardcoded_events()
        event_row = events_df[events_df['event_name'] == event_name]
        if not event_row.empty:
            return float(event_row.iloc[0]['gps_time'])
        
        return None
    
    def create_synthetic_overlaps(self,
                                 events_df: pd.DataFrame,
                                 n_overlaps: int = 100,
                                 overlap_window: float = 0.5) -> List[Dict]:
        """
        Create synthetic overlapping scenarios from real events
        
        Args:
            events_df: DataFrame with GWTC events
            n_overlaps: Number of overlapping scenarios to create
            overlap_window: Time window for overlaps (seconds)
            
        Returns:
            List of overlapping scenario dicts
        """
        
        if events_df.empty or len(events_df) < 2:
            return []
        
        # Filter quality events
        quality_events = events_df[
            (events_df['network_snr'] > 10) &
            (events_df['mass_1_source'] > 5)
        ]
        
        if len(quality_events) < 2:
            quality_events = events_df
        
        overlaps = []
        
        for i in range(n_overlaps):
            # Number of signals (2-4)
            n_signals = np.random.choice([2, 3, 4], p=[0.6, 0.3, 0.1])
            
            if len(quality_events) >= n_signals:
                selected = quality_events.sample(n_signals)
                base_time = selected.iloc[0]['gps_time']
                
                # Create time offsets
                time_offsets = [0.0] + [
                    np.random.uniform(-overlap_window/2, overlap_window/2)
                    for _ in range(n_signals - 1)
                ]
                
                overlaps.append({
                    'scenario_id': i,
                    'n_signals': n_signals,
                    'central_gps_time': base_time,
                    'events': [
                        {**event.to_dict(), 'time_offset': time_offsets[j]}
                        for j, (_, event) in enumerate(selected.iterrows())
                    ]
                })
        
        return overlaps


# ================================================================================
# INJECTION.PY
# ================================================================================

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
        """Compute optimal matched-filter SNR"""
        from numpy.fft import rfft, rfftfreq
        
        N = len(waveform)
        dt = 1.0 / self.sample_rate
        freqs = rfftfreq(N, dt)
        wf_fft = rfft(waveform)
        
        psd_freqs = psd_dict['frequencies']
        psd = psd_dict['psd']
        psd_interp = np.interp(freqs, psd_freqs, psd, 
                       left=psd, right=psd[-1])
        
        min_psd = np.min(psd[psd > 0]) if np.any(psd > 0) else 1e-20
        psd_floor = max(min_psd * 0.1, 1e-20)
        psd_interp = np.maximum(psd_interp, psd_floor)
        
        if np.any(np.isnan(psd_interp)) or np.any(np.isinf(psd_interp)):
            self.logger.warning("PSD has NaN/inf; using default SNR=15.0")
            return 15.0

        delta_f = freqs - freqs if len(freqs) > 1 else 1.0
        if delta_f <= 0:
            self.logger.warning("Invalid frequency spacing; using default SNR=15.0")
            return 15.0

        integrand = np.abs(wf_fft)**2 / psd_interp
        snr_squared = 4 * np.sum(integrand) * delta_f
        snr = np.sqrt(snr_squared)

        # FIX: Clip to reasonable range
        snr = np.clip(snr, 1.0, 100.0)

        if np.isnan(snr) or np.isinf(snr):
            self.logger.warning(f"SNR computation resulted in {snr}; using default 15.0")
            return 15.0

        return float(snr)


def compute_network_snr_from_det_dict(d: dict):
    """
    Compute sqrt(sum s_i^2) from per-detector SNRs if present.
    Returns None if no per-detector SNRs are found.
    """
    snrs = []
    for k in ('snr_H1', 'snr_L1', 'snr_V1'):
        v = d.get(k, None)
        if v is not None:
            snrs.append(float(v))
    if not snrs:
        return None
    return float(math.sqrt(sum(s*s for s in snrs)))


def proxy_network_snr_from_params(d: dict) -> float:
    """
    PROXY SNR that does NOT use mass or distance inputs.
    - If 'target_snr' is present (explicit desired SNR), return it with tiny additive noise.
    - Otherwise return a stochastic SNR draw independent of physical params.
    This prevents implicit mass-distance correlation.
    """
    if not isinstance(d, dict):
        return float(np.clip(np.random.lognormal(mean=np.log(15.0), sigma=0.25), 4.0, 100.0))

    # Prefer explicit target_snr (small additive measurement noise only)
    target = d.get("target_snr", None)
    if target is not None and not (isinstance(target, float) and np.isnan(target)):
        # Add small additive noise (NOT multiplicative scaling)
        noise = np.random.normal(0.0, 0.3)  # small absolute noise in SNR units
        return float(np.clip(float(target) + noise, 4.0, 100.0))

    # No target provided — sample a stochastic proxy independent of mass/distance
    # lognormal centered ~15 with modest sigma; fully independent.
    snr_proxy = np.random.lognormal(mean=np.log(15.0), sigma=0.25)
    return float(np.clip(snr_proxy, 4.0, 100.0))




def attach_network_snr(d: dict):
    """
    Attach 'network_snr' to detection dict d in-place.
    Priority: matched-filter net SNR; fallback to proxy.
    """
    snr_net = compute_network_snr_from_det_dict(d)
    if snr_net is None:
        snr_net = proxy_network_snr_from_params(d)
    d['network_snr'] = float(snr_net)

# ================================================================================
# SIMULATION.PY
# ================================================================================

"""
Simulate overlapping gravitational wave signals for training.
"""

import numpy as np
import bilby
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from typing import List, Dict, Tuple, Optional
import torch
import logging
from pathlib import Path
import pickle
from gwpy.timeseries import TimeSeries
from scipy.signal import get_window

# Suppress bilby warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", message="Unkown projection method")
warnings.filterwarnings("ignore", message="Unknown projection method")

# Configure bilby logging
bilby.core.utils.logger.setLevel('WARNING')

class OverlappingSignalSimulator:
    """
    OverlappingSignalSimulator
    Simulates overlapping gravitational wave (GW) signals for training machine learning models or testing GW data analysis pipelines. This class provides methods to generate realistic GW signal parameters, simulate detector noise, inject multiple overlapping signals into noise, and create datasets for training.
    Attributes:
        config (AHSDConfig): Configuration object specifying waveform and detector settings.
        logger (logging.Logger): Logger for status and error messages.
        detectors (dict): Dictionary of detector objects keyed by detector name.
        waveform_generator (bilby.gw.WaveformGenerator): Bilby waveform generator for GW signals.
    Methods:
        __init__(config: AHSDConfig):
            Initializes the simulator with the given configuration, sets up detectors and waveform generator.
        _setup_waveform_generator():
            Sets up the bilby waveform generator using configuration parameters.
        generate_single_signal_params() -> Dict:
            Generates a dictionary of parameters for a single GW signal using conservative, realistic priors.
        _sample_power_law(min_val: float, max_val: float, alpha: float) -> float:
            Samples a value from a power-law distribution.
        _sample_spin_magnitude() -> float:
            Samples a spin magnitude using a Beta distribution fit to GWTC measurements.
        generate_overlapping_scenario(n_signals: int) -> Dict:
            Generates a scenario dictionary containing parameters for multiple overlapping GW signals.
        generate_detector_noise(duration: Optional[float] = None, sampling_rate: Optional[int] = None) -> Dict:
            Generates realistic colored Gaussian noise for each detector using analytical PSDs.
        _generate_colored_noise(n_samples: int, sampling_rate: int, detector: str) -> np.ndarray:
            Generates colored noise for a given detector using its power spectral density (PSD).
        _aLIGO_psd(freqs: np.ndarray) -> np.ndarray:
            Returns a simplified analytical PSD for Advanced LIGO.
        _virgo_psd(freqs: np.ndarray) -> np.ndarray:
            Returns a simplified analytical PSD for Virgo.
        inject_signals_to_data(scenario: Dict, noise_data: Dict) -> Tuple[Dict, Dict]:
            Injects multiple GW signals into detector noise, returning the injected data and individual signal contributions.
        _generate_mock_waveform(params: Dict, detector_name: str) -> np.ndarray:
            Generates a simple mock waveform for a GW signal, used as a fallback if bilby waveform generation fails.
        _generate_detector_strain(params: Dict, detector_name: str) -> Optional[np.ndarray]:
            Generates the strain time series for a specific detector using bilby, with fallback to mock waveform.
        create_training_dataset(n_scenarios: int, output_dir: str) -> List[Dict]:
            Generates and saves a dataset of simulated scenarios with overlapping GW signals and noise for training.
    """
    """Simulate overlapping gravitational wave signals for training."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup detectors
        self.detectors = {}
        for det_config in config.detectors:
            try:
                self.detectors[det_config.name] = Detector(det_config.name)
            except:
                # Fallback to bilby detector
                self.detectors[det_config.name] = self._create_mock_detector(det_config.name)
        
        # Setup waveform generator
        self.waveform_generator = self._setup_waveform_generator()
        
    def _setup_waveform_generator(self):
        """Setup bilby waveform generator."""
        
        return bilby.gw.WaveformGenerator(
            duration=self.config.waveform.duration,
            sampling_frequency=self.config.detectors[0].sampling_rate,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(
                waveform_approximant=self.config.waveform.approximant,
                reference_frequency=self.config.waveform.f_ref,
            )
        )
        
    def generate_single_signal_params(self) -> Dict:
        """Generate parameters for a single GW signal using realistic priors."""
        from astropy.cosmology import Planck18

        #  MASS SAMPLING
        mass_1 = np.random.uniform(25, 40)
        mass_2 = np.random.uniform(20, 35)
        if mass_1 < mass_2:
            mass_1, mass_2 = mass_2, mass_1

        #  COSMOLOGY (NEW!)
        z = np.random.uniform(0.01, 0.5)
        d_L = Planck18.luminosity_distance(z).to('Mpc').value
        d_C = d_L / (1 + z)

        #  SPINS
        a_1 = np.random.uniform(0.1, 0.4)
        a_2 = np.random.uniform(0.1, 0.4)

        #  SKY POSITIONS
        ra = np.random.uniform(0.5, 2*np.pi - 0.5)
        dec = np.random.uniform(-0.3, 0.3)

        #  ANGLES (ISOTROPIC + FULL RANGE)
        theta_jn = np.arccos(np.random.uniform(-1, 1))
        psi = np.random.uniform(0, np.pi)
        phase = np.random.uniform(0, 2*np.pi)

        geocent_time = np.random.uniform(-0.02, 0.02)

        #  SPIN ANGLES 
        tilt_1 = np.random.uniform(0, np.pi)
        tilt_2 = np.random.uniform(0, np.pi)
        phi_12 = np.random.uniform(0, 2*np.pi)
        phi_jl = np.random.uniform(0, 2*np.pi)

        total_mass = mass_1 + mass_2
        chirp_mass = (mass_1 * mass_2)**(3/5) / total_mass**(1/5)
        mass_ratio = mass_2 / mass_1

        effective_spin = (mass_1 * a_1 * np.cos(tilt_1) + 
                        mass_2 * a_2 * np.cos(tilt_2)) / total_mass

        return {
            'mass_1': mass_1,
            'mass_2': mass_2,
            'total_mass': total_mass,           # NEW!
            'chirp_mass': chirp_mass,           # NEW!
            'mass_ratio': mass_ratio,           # NEW!
            'luminosity_distance': d_L,         # CHANGED (was hardcoded)
            'comoving_distance': d_C,           # NEW!
            'redshift': z,                      # NEW!
            'theta_jn': theta_jn,
            'psi': psi,
            'phase': phase,
            'geocent_time': geocent_time,
            'ra': ra,
            'dec': dec,
            'a1': a_1,                          # NOTE: Changed from 'a_1' to 'a1'
            'a2': a_2,                          # NOTE: Changed from 'a_2' to 'a2'
            'effective_spin': effective_spin,   # NEW!
            'tilt_1': tilt_1,
            'tilt_2': tilt_2,
            'phi_12': phi_12,
            'phi_jl': phi_jl,
            'lambda_1': 0.0,
            'lambda_2': 0.0,
            'type': 'BBH',                      # NEW! (was 'approximant')
        }


    def _sample_powerlaw(primary_min: float, primary_max: float, alpha: float) -> float:
        """Draw from p(m) ∝ m^{-alpha} on [primary_min, primary_max] by inverse-transform."""
        if primary_min <= 0:
            primary_min = 1e-3
        if abs(alpha - 1.0) < 1e-12:
            r = np.random.uniform(0, 1)
            return primary_min * (primary_max / primary_min) ** r
        r = np.random.uniform(0, 1)
        pow_min = primary_min ** (1.0 - alpha)
        pow_max = primary_max ** (1.0 - alpha)
        return (pow_min + r * (pow_max - pow_min)) ** (1.0 / (1.0 - alpha))
    
    def _sample_mass_ratio(q_min: float = 0.05, q_max: float = 1.0, beta_param: float = 0.0) -> float:
        """Draw q from p(q) ∝ q^{beta_param} on [q_min, q_max]. beta_param=0 -> uniform."""
        if q_min >= q_max:
            return float(q_min)
        if abs(beta_param) < 1e-12:
            return float(np.random.uniform(q_min, q_max))
        r = np.random.uniform(0, 1)
        pow_min = q_min ** (beta_param + 1)
        pow_max = q_max ** (beta_param + 1)
        return float((pow_min + r * (pow_max - pow_min)) ** (1.0 / (beta_param + 1)))

    def _sample_volume_distance(d_min: float, d_max: float) -> float:
        """Volume-weighted distance (P(d) ∝ d^2)."""
        u = np.random.uniform(0, 1)
        return float((d_min**3 + u * (d_max**3 - d_min**3)) ** (1.0/3.0))

    def _draw_until_snr(make_params_fn: Callable[[], Dict],
                    snr_range: Tuple[float, float],
                    attach_snr_fn: Callable,
                    max_tries: int = 3000) -> Dict:
        """
        Rejection-sampling: draw independent physics params and accept only if
        the computed intrinsic SNR lies within snr_range.

        Key points:
        - attach_snr_fn called with mutate_params=False whenever possible,
        so physical params are never mutated during acceptance testing.
        - If attach_snr_fn does not support the flag, fallback safely to proxy.
        - Returns a params dict with 'network_snr', 'accepted' and optionally
        'rejection_sampling_exhausted'.
        """
        low, high = snr_range
        last = None

        for attempt in range(max_tries):
            params = make_params_fn()

            # Compute SNR without mutating physics params
            try:
                snr_val = attach_snr_fn(params, mutate_params=False)
            except TypeError:
                # function doesn't accept mutate flag — call and expect it to mutate
                try:
                    attach_snr_fn(params)
                    snr_val = params.get('network_snr', None)
                except Exception:
                    snr_val = None
            except Exception:
                snr_val = None

            # If attach returned None, use proxy (still independent)
            if snr_val is None:
                try:
                    snr_val = proxy_network_snr_from_params(params)
                except Exception:
                    snr_val = None

            # Still None? remember last and continue
            if snr_val is None:
                last = params
                continue

            # Do not write back any amplitude scaling; only annotate network_snr copy
            params['network_snr'] = float(snr_val)
            last = params

            if low <= snr_val <= high:
                params['accepted'] = True
                params['rejection_sampling_exhausted'] = False
                return params

        # exhausted: return last with flags
        if last is not None:
            last['accepted'] = False
            last['rejection_sampling_exhausted'] = True
            return last

        # extreme fallback
        return {'accepted': False, 'error': 'Failed to generate sample'}

    
    def _sample_spin_magnitude(self) -> float:
        """Sample spin magnitude using Beta distribution."""
        
        # Beta distribution fit to GWTC measurements
        return np.random.beta(1.5, 3.0) * 0.99
    
    def generate_overlapping_scenario(self, n_signals: int) -> Dict:
        """Generate a scenario with overlapping signals."""
        
        signals = []
        for i in range(n_signals):
            params = self.generate_single_signal_params()
            params['signal_id'] = i
            signals.append(params)
            
        return {
            'signals': signals,
            'n_signals': n_signals,
            'scenario_id': np.random.randint(0, 1000000)
        }
    
    def generate_detector_noise(self, 
                              duration: Optional[float] = None,
                              sampling_rate: Optional[int] = None) -> Dict:
        """Generate realistic detector noise."""
        
        if duration is None:
            duration = self.config.waveform.duration
        if sampling_rate is None:
            sampling_rate = self.config.detectors[0].sampling_rate
            
        n_samples = int(duration * sampling_rate)
        noise_data = {}
        
        for det_name in self.detectors.keys():
            # Generate colored Gaussian noise with realistic PSD
            noise = self._generate_colored_noise(n_samples, sampling_rate, det_name)
            noise_data[det_name] = noise
            
        return noise_data
    
    def _generate_colored_noise(self, n_samples: int, sampling_rate: int, detector: str) -> np.ndarray:
        """Generate colored noise with detector PSD."""
        
        try:
            # Get detector PSD (simplified)
            freqs = np.fft.fftfreq(n_samples, 1/sampling_rate)
            positive_freqs = freqs[:n_samples//2 + 1]
            
            # Simplified analytical PSD for Advanced LIGO
            if detector in ['H1', 'L1']:
                psd = self._aLIGO_psd(positive_freqs)
            elif detector == 'V1':
                psd = self._virgo_psd(positive_freqs)
            else:
                # White noise fallback
                psd = np.ones_like(positive_freqs)
            
            # Generate white noise in frequency domain
            white_noise_f = np.random.normal(0, 1, n_samples//2 + 1) + 1j * np.random.normal(0, 1, n_samples//2 + 1)
            white_noise_f[0] = white_noise_f[0].real  # DC component
            if n_samples % 2 == 0:
                white_noise_f[-1] = white_noise_f[-1].real  # Nyquist component
            
            # Color the noise
            colored_noise_f = white_noise_f / np.sqrt(psd * sampling_rate / 2)
            
            # Convert to time domain
            colored_noise_f_full = np.concatenate([colored_noise_f, np.conj(colored_noise_f[-2:0:-1])])
            colored_noise = np.fft.ifft(colored_noise_f_full).real
            
            return colored_noise
            
        except Exception as e:
            self.logger.warning(f"Failed to generate colored noise for {detector}: {e}")
            # Fallback to white noise
            return np.random.normal(0, 1e-23, n_samples)
    
    def _aLIGO_psd(self, freqs: np.ndarray) -> np.ndarray:
        """Simplified analytical PSD for Advanced LIGO."""
        
        # Avoid zero frequency
        freqs = np.maximum(freqs, 10.0)
        
        # Simplified analytical fit
        f0 = 215.0  # Hz
        psd = (freqs / f0)**(-4.14) - 5 * (freqs / f0)**(-2) + 111 * (1 + (freqs / f0)**2)**(-0.5)
        
        # Add low frequency rise
        psd += 1e4 * (freqs / 10.0)**(-8)
        
        return psd * 1e-48
    
    def _virgo_psd(self, freqs: np.ndarray) -> np.ndarray:
        """Simplified analytical PSD for Virgo."""
        
        freqs = np.maximum(freqs, 10.0)
        
        # Simplified Virgo PSD
        psd = 3.2e-46 * (freqs / 100.0)**(-4.05) + 2e-48
        
        return psd
    
    def inject_signals_to_data(self, scenario: Dict, noise_data: Dict) -> Tuple[Dict, Dict]:
        """Inject multiple overlapping signals into noise data."""
        
        injected_data = {}
        signal_contributions = {}
        
        for det_name, noise in noise_data.items():
            if det_name in self.detectors:
                try:
                    detector = self.detectors[det_name]
                    total_strain = np.array(noise)
                    signal_contributions[det_name] = {}
                    
                    for signal in scenario['signals']:
                        # Generate waveform for this detector
                        strain = self._generate_detector_strain(signal, det_name)
                        
                        if strain is not None:
                            # Ensure same length
                            min_len = min(len(total_strain), len(strain))
                            total_strain[:min_len] += strain[:min_len]
                            signal_contributions[det_name][signal['signal_id']] = strain
                    
                    injected_data[det_name] = total_strain
                    
                except Exception as e:
                    self.logger.warning(f"Failed to inject signals into {det_name}: {e}")
                    injected_data[det_name] = noise
                    signal_contributions[det_name] = {}
            else:
                injected_data[det_name] = noise
                signal_contributions[det_name] = {}
                
        return injected_data, signal_contributions
    
    def _generate_mock_waveform(self, params: Dict, detector_name: str) -> np.ndarray:
        """Generate simple mock waveform without bilby warnings."""
        
        # Simple chirp-like signal
        duration = self.config.waveform.duration
        sampling_rate = self.config.detectors[0].sampling_rate
        n_samples = int(duration * sampling_rate)
        
        t = np.linspace(0, duration, n_samples)
        
        # Extract parameters
        m1 = params.get('mass_1', 30.0)
        m2 = params.get('mass_2', 30.0)
        distance = params.get('luminosity_distance', 500.0)
        
        # Chirp mass
        chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        
        # Simple frequency evolution
        f_start = 35.0
        f_end = 250.0
        freq = f_start + (f_end - f_start) * (t / duration)**3
        
        # Amplitude with 1/r scaling
        amp = 1e-21 * (chirp_mass / 30.0)**(5/6) / (distance / 400.0)
        
        # Exponential envelope
        envelope = np.exp(-t / (duration * 0.4))
        
        # Generate strain with detector response
        dt = t[1] - t[0] if len(t) > 1 else 1/sampling_rate
        phase = 2 * np.pi * np.cumsum(freq) * dt
        strain = amp * envelope * np.sin(phase)
        
        # Apply simple detector response based on sky position
        ra = params.get('ra', 0.0)
        dec = params.get('dec', 0.0)
        psi = params.get('psi', 0.0)
        
        # Simple detector response approximation
        if detector_name == 'H1':
            response = 0.5 * (1 + np.cos(dec)**2) * np.cos(2*psi)
        elif detector_name == 'L1':  
            response = 0.5 * (1 + np.cos(dec)**2) * np.cos(2*psi + np.pi/2)
        elif detector_name == 'V1':
            response = 0.3 * (1 + np.cos(dec)**2) * np.cos(2*psi + np.pi/4)
        else:
            response = 0.5
        
        return strain * abs(response)
    
    def _generate_detector_strain(self, params: Dict, detector_name: str) -> Optional[np.ndarray]:
        """Generate strain for specific detector using bilby with fallback."""
        
        try:
            # First try bilby
            waveform_polarizations = self.waveform_generator.frequency_domain_strain(params)
            
            if detector_name in self.detectors:
                if hasattr(self.detectors[detector_name], 'project_wave'):
                    # bilby detector
                    detector = self.detectors[detector_name]
                    strain = detector.project_wave(
                        waveform_polarizations['plus'],
                        waveform_polarizations['cross'],
                        params['ra'], params['dec'], params['psi'],
                        params.get('geocent_time', 0.0)
                    )
                    return strain.time_domain_strain
                else:
                    # PyCBC detector - simplified projection
                    waveform_td = np.fft.ifft(waveform_polarizations['plus']).real
                    return waveform_td
            
            return None
            
        except Exception as e:
            # Fallback to mock waveform on any bilby error
            try:
                return self._generate_mock_waveform(params, detector_name)
            except Exception as e2:
                self.logger.debug(f"Both bilby and mock waveform failed for {detector_name}: {e2}")
                return None
    
    def create_training_dataset(self, 
                              n_scenarios: int, 
                              output_dir: str) -> List[Dict]:
        """Create complete training dataset."""
        
        training_scenarios = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating {n_scenarios} training scenarios...")
        
        for i in range(n_scenarios):
            try:
                # Number of overlapping signals (weighted toward lower numbers)
                n_signals = np.random.choice([2, 3, 4, 5], p=[0.5, 0.3, 0.15, 0.05])
                
                # Generate scenario
                scenario = self.generate_overlapping_scenario(n_signals)
                
                # Generate noise
                noise_data = self.generate_detector_noise()
                
                # Inject signals
                injected_data, signal_contributions = self.inject_signals_to_data(
                    scenario, noise_data
                )
                
                # Create training example
                training_scenario = {
                    'scenario_id': i,
                    'true_parameters': scenario['signals'],
                    'injected_data': injected_data,
                    'noise_data': noise_data,
                    'signal_contributions': signal_contributions,
                    'n_signals': n_signals,
                    'generation_params': {
                        'duration': self.config.waveform.duration,
                        'sampling_rate': self.config.detectors[0].sampling_rate,
                        'approximant': self.config.waveform.approximant
                    }
                }
                
                training_scenarios.append(training_scenario)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Generated {i + 1}/{n_scenarios} scenarios")
                    
            except Exception as e:
                self.logger.error(f"Failed to generate scenario {i}: {e}")
                continue
        
        # Save dataset
        dataset_file = output_path / 'simulated_training_data.pkl'
        with open(dataset_file, 'wb') as f:
            pickle.dump(training_scenarios, f)
        
        self.logger.info(f"Saved {len(training_scenarios)} training scenarios to {dataset_file}")
        
        return training_scenarios


def compute_waveform_overlap(h1: np.ndarray, h2: np.ndarray, 
                           sampling_rate: float = 4096.0) -> float:
    """Compute normalized overlap between two waveforms."""
    
    try:
        # Ensure same length
        min_len = min(len(h1), len(h2))
        h1 = h1[:min_len]
        h2 = h2[:min_len]
        
        # Compute overlap in time domain (simplified)
        overlap = np.abs(np.dot(h1, h2)) / (np.linalg.norm(h1) * np.linalg.norm(h2))
        
        return float(overlap)
        
    except:
        return 0.0


def estimate_snr_from_strain(strain: np.ndarray, 
                           psd: Optional[np.ndarray] = None,
                           sampling_rate: float = 4096.0) -> float:
    """Estimate SNR of strain data."""
    
    try:
        if psd is None:
            # Simple time-domain SNR estimate - use RMS amplitude ratio
            rms_signal = np.sqrt(np.mean(strain**2))
            # Assume typical detector noise RMS ~ 1e-20 m/sqrt(Hz) * sqrt(bandwidth)
            # For 4s duration, bandwidth ~ 1/4 Hz, so RMS noise ~ 1e-20 * 0.5 ~ 5e-21
            rms_noise = 5e-21
            if rms_signal <= 0 or rms_noise <= 0:
                return 10.0  # Return default for zero signal or noise
            # SNR = signal RMS / noise RMS
            snr = rms_signal / rms_noise
            # Cap at reasonable maximum
            snr = min(snr, 100.0)
        else:
            # Matched filter SNR (ensure consistent FFT normalization)
            N = len(strain)
            strain_f = np.fft.rfft(strain)
            freqs = np.fft.rfftfreq(N, 1/sampling_rate)
            
            # Match PSD length to FFT output
            if hasattr(psd, 'numpy'):
                psd_array = psd.numpy()
            else:
                psd_array = np.asarray(psd)
            
            # Ensure PSD is same length as FFT output
            if len(psd_array) != len(strain_f):
                # Interpolate or truncate PSD to match
                if len(psd_array) > len(strain_f):
                    psd_seg = psd_array[:len(strain_f)]
                else:
                    # Simple repetition if too short
                    psd_seg = np.ones(len(strain_f)) * 1e-46
                    psd_seg[:len(psd_array)] = psd_array
            else:
                psd_seg = psd_array
            
            # Prevent division by zero or very small values
            psd_seg = np.maximum(psd_seg, 1e-20)

            # Compute SNR squared
            # Additional safety: ensure no zero denominators
            safe_psd = np.where(psd_seg > 0, psd_seg, 1e-20)
            integrand = np.abs(strain_f)**2 / safe_psd
            
            # Remove any inf/nan values
            integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)
            
            df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0/sampling_rate
            # Use 4 * integral for one-sided PSD
            snr_squared = 4.0 * np.sum(integrand) * df
            
            # Clip to reasonable range before sqrt
            snr_squared = np.clip(snr_squared, 0.0, 6400.0)  # Max SNR 80
            snr = np.sqrt(snr_squared)
        
        # Final safety check
        if np.isnan(snr) or np.isinf(snr):
            return 15.0  # Return reasonable default
        
        return float(np.clip(snr, 0.0, 80.0))
        
    except Exception as e:
        # Log error for debugging but don't crash
        return 15.0  # Return reasonable default


# ================================================================================
# DATASET_GENERATOR.PY
# ================================================================================

"""
Main Dataset Generator
Orchestrates complete dataset generation pipeline
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Tuple
from tqdm import tqdm
import time
import gc
import glob
import pickle
from collections import Counter
from functools import wraps
import random
from collections import defaultdict
from functools import partial

_EDGE_MAP = {}  # Global edge type mapping

def sample_overlap_size():
    """70% dense overlaps (5–8), 30% lighter (2–4) for robustness.

    This ensures ~28-33% of all scenarios have 5+ signals
    (70% of overlaps * ~40% overlap fraction = ~28%)
    """
    if random.random() < 0.70:  # Increased from 60% to 70%
        return random.choice([5, 6, 7, 8])
    return random.choice([2, 3, 4])

def build_overlap_group(pool, n):
    """
    Build a group of n detections by time proximity and mixed types.
    pool: list of candidate detection dicts with 'geocent_time'.
    """
    if not pool or n <= 0:
        return []
    seed = random.choice(pool)
    candidates = sorted(pool, key=lambda d: abs(d.get('geocent_time', 0.0) - seed.get('geocent_time', 0.0)))
    group = [seed]
    for d in candidates:
        if len(group) >= n:
            break
        if d is seed:
            continue
        group.append(d)
    return group

def encode_edge_type(dets):
    """
    Map the set of detectors present and overlap size to a stable int ID.
    Handles None, empty lists, and dicts without 'detectors' gracefully.
    """
    if not dets:
        return 0
    detectors = set()
    for d in dets:
        if d is None:  # ← FIX: skip None entries
            continue
        if isinstance(d, dict):
            detectors |= set(d.get('detectors', []))
    size = len([d for d in dets if d is not None])  # ← FIX: count only non-None
    key = (tuple(sorted(detectors)), size)
    if key not in _EDGE_MAP:
        _EDGE_MAP[key] = len(_EDGE_MAP)
    return _EDGE_MAP[key]


def attach_network_snr_safe(params_or_list, mutate_params: bool = True):
    """
    Attach or compute network SNR. If mutate_params=False, return SNR (float)
    and DO NOT modify the input dict(s). If mutate_params=True, write in-place
    and return None (legacy behaviour).

    IMPORTANT: this function never changes physical parameters (masses/distances).
    """
    if params_or_list is None:
        return None

    # List handler: compute per-item and return mean if mutate_params==False
    if isinstance(params_or_list, list):
        snrs = []
        for p in params_or_list:
            if p is None or not isinstance(p, dict):
                continue
            val = attach_network_snr_safe(p, mutate_params=mutate_params)
            if not mutate_params and val is not None:
                snrs.append(val)
        return float(np.mean(snrs)) if snrs else None

    # Single dict case
    d = params_or_list
    snr_val = None

    # 1) If detector-level SNRs are present, use compute_network_snr_from_det_dict
    try:
        if isinstance(d, dict) and ("detector_data" in d or "snr_H1" in d or "network" in d):
            snr_val = compute_network_snr_from_det_dict(d)
    except Exception:
        snr_val = None

    # 2) Fallback to proxy (explicitly NOT depending on mass/distance)
    if snr_val is None:
        snr_val = proxy_network_snr_from_params(d)

    if snr_val is None:
        return None

    if mutate_params:
        # Only write the scalar; do not mutate physical fields (no amplitude_scaling etc.)
        d["network_snr"] = float(snr_val)
        return None
    else:
        return float(snr_val)



def rescale_priorities(y_raw):
    """
    Rescale list/array y_raw to [0,1] with mild gamma < 1 to expand headroom.
    """
    y_arr = np.asarray(y_raw, dtype=np.float32)
    ymin, ymax = float(np.min(y_arr)), float(np.max(y_arr))
    if ymax - ymin < 1e-6:
        return np.clip(y_arr * 0 + 0.5, 0.0, 1.0).tolist()
    y = (y_arr - ymin) / (ymax - ymin)
    y = np.clip(y ** 0.9, 0.0, 1.0)
    return y.tolist()

def maybe_inject_decoy(detections, p=0.30):
    """
    Optionally append a decoy (weaker copy) to force label separation.
    """
    if len(detections) == 0 or random.random() >= p:
        return detections
    d0 = detections[0]
    d_decoy = dict(d0)
    if d_decoy.get('network_snr') is not None:
        d_decoy['network_snr'] = max(0.0, float(d_decoy['network_snr']) * 0.7)
    d_decoy['luminosity_distance'] = float(d_decoy.get('luminosity_distance', 500.0)) * 1.3
    attach_network_snr(d_decoy)
    detections.append(d_decoy)
    return detections




# Add to the GWDatasetGenerator class

class GWDatasetGenerator:
    """
    Main class for generating complete GW datasets
    Supports HDF5 and PKL output formats
    """
    
    def __init__(self, 
             output_dir: str = "data/output",
             sample_rate: int = SAMPLE_RATE,
             duration: float = DURATION,
             detectors: List[str] = None,
             output_format: str = 'pkl',
             config: Dict = None,
             parameter_sampler=None):  
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.duration = duration
        self.detectors = detectors or DETECTORS
        self.output_format = output_format.lower()

        # Ensure snr_ranges is properly initialized
        self.snr_ranges = SNR_RANGES
        assert hasattr(self, 'snr_ranges'), "snr_ranges not initialized"
        
        #  FIX: Initialize extreme config properly
        self.config = config if config is not None else {}
        self.extreme_config = self.config.get('extreme_cases', {})
        self.extreme_enabled = self.extreme_config.get('enabled', True)  # Enable by default
        self.extreme_fraction = self.extreme_config.get('fraction', 0.10)  # 10% for extreme cases
        self.stats = {
            'event_types': {'BBH': 0, 'BNS': 0, 'NSBH': 0},
            'edge_cases': {
                'short_bbh': 0,
                'extreme_mass_ratio': 0,
                'extreme_mass': 0,
                'long_bns_inspiral': 0,
                'high_spin': 0
            },
            'snr_regimes': {regime: 0 for regime in self.snr_ranges.keys()}
        }


        # Initialize default extreme types config if not provided
        default_extreme_types = {
            'near_simultaneous_mergers': {'enabled': True, 'fraction': 0.08},
            'extreme_mass_ratio': {'enabled': True, 'fraction': 0.08},
            'high_spin_aligned': {'enabled': True, 'fraction': 0.08},
            'precession_dominated': {'enabled': True, 'fraction': 0.08},
            'eccentric_overlaps': {'enabled': True, 'fraction': 0.1},  # Increased
            'weak_strong_overlaps': {'enabled': True, 'fraction': 0.08},
            'noise_confused_overlaps': {'enabled': True, 'fraction': 0.08},
            'long_duration_bns_overlaps': {'enabled': True, 'fraction': 0.08},
            'detector_dropouts': {'enabled': True, 'fraction': 0.1},  # Increased
            'cosmological_distance': {'enabled': True, 'fraction': 0.1}  # Increased
        }
        self.extreme_types_config = self.extreme_config.get('types', default_extreme_types)
        
        # Setup logging FIRST (simulator needs it)
        self.logger = logging.getLogger(__name__)
        
        #  FIX: Initialize simulator with proper config conversion
        try:
            # Create AHSDConfig object from dict
            from types import SimpleNamespace
            
            sim_config = SimpleNamespace()
            
            # Detector configs
            sim_config.detectors = []
            for det_name in self.detectors:
                det_cfg = SimpleNamespace(
                    name=det_name,
                    sampling_rate=sample_rate
                )
                sim_config.detectors.append(det_cfg)
            
            # Waveform config
            sim_config.waveform = SimpleNamespace(
                duration=duration,
                approximant=self.config.get('approximant', 'IMRPhenomPv2'),
                f_ref=self.config.get('f_ref', 20.0)
            )
            
            # Initialize simulator
            self.simulation = OverlappingSignalSimulator(sim_config)
            self.logger.info(" Initialized OverlappingSignalSimulator")
            
        except Exception as e:
            self.logger.warning(f"Simulator initialization failed: {e}")
            self.logger.info("Falling back to legacy generation with SNR estimation")
            self.simulation = None
        
        self.logger.info(f"Extreme cases: {'enabled' if self.extreme_enabled else 'disabled'} "
                        f"(fraction={self.extreme_fraction})")
        
        # Validate format
        if self.output_format not in ['hdf5', 'pkl', 'pkl_compressed', 'both']:
            raise ValueError(f"Invalid output_format: {self.output_format}")
        
        # Initialize components
        self.logger.info("Initializing dataset generator components...")
        self.psd_manager = PSDManager(sample_rate, duration)
        # Allow injection of a pre-calibrated ParameterSampler for conditioning
        if parameter_sampler is not None:
            self.parameter_sampler = parameter_sampler
        else:
            self.parameter_sampler = ParameterSampler()
        self.waveform_generator = WaveformGenerator(sample_rate, duration)
        self.noise_generator = NoiseGenerator(sample_rate, duration)
        self.injector = SignalInjector(sample_rate, duration)
        self.preprocessor = DataPreprocessor(sample_rate, duration)
        self.writer = DatasetWriter(output_dir, format=output_format)
        self.metadata_manager = MetadataManager()
        
        # Load PSDs
        self.logger.info(f"Loading PSDs for detectors: {self.detectors}")
        self.psds = self.psd_manager.load_detector_psds(self.detectors)
        
        self.logger.info(f"✓ Dataset generator initialized (output format: {output_format})")
        # Debug SNR diagnostic counters (can be enabled via config['debug_snr_diagnostic'])
        self._debug_snr_count = 0
        self._debug_snr_limit = int(self.config.get('debug_snr_limit', 50))
    
    def create_noise_augmentations(self, sample: Dict, k: int) -> List[Dict]:
        """
        Create k augmented versions with different noise realizations
        
        Args:
            sample: Original sample
            k: Number of augmentations (including original)
            
        Returns:
            List of k augmented samples
        """
        
        if k <= 1:
            return [sample]
        
        augmented_samples = [sample]  # Include original
        
        # Get signal for this sample (if not noise-only)
        has_signal = sample['type'] != 'noise'
        
        for aug_idx in range(1, k):
            # Create augmented copy
            aug_sample = {
                'sample_id': f"{sample['sample_id']}_aug{aug_idx}",
                'type': sample['type'],
                'is_overlap': sample.get('is_overlap', False),
                'is_edge_case': sample.get('is_edge_case', False),
                'parameters': sample.get('parameters'),
                'detector_data': {}
            }
            
            # Generate new noise realization for each detector
            for det_name in self.detectors:
                psd_dict = self.psds[det_name]
                
                # Generate fresh noise
                new_noise = self.noise_generator.generate_colored_noise(psd_dict)
                
                if has_signal and 'detector_data' in sample:
                    # Extract signal from original (assuming signal + noise structure)
                    # For simplicity, we regenerate the signal with same params
                    original_data = sample['detector_data'].get(det_name, {})
                    
                    if sample.get('is_overlap', False):
                        # Overlapping signals
                        params_list = sample['parameters']
                        injected, metadata_list = self.injector.inject_overlapping_signals(
                            new_noise, params_list, det_name, psd_dict
                        )
                    else:
                        # Single signal
                        params = sample['parameters']
                        if params:
                            injected, metadata = self.injector.inject_signal(
                                new_noise, params, det_name, psd_dict
                            )
                        else:
                            injected = new_noise
                            metadata = {'noise_only': True}
                else:
                    # Noise-only
                    injected = new_noise
                    metadata = {'noise_only': True}
                
                # Preprocess if needed
                if self.preprocessor:
                    injected = self.preprocessor.preprocess(injected, psd_dict)
                
                aug_sample['detector_data'][det_name] = {
                    'strain': injected.astype(np.float32),
                    'metadata': metadata if not sample.get('is_overlap') else metadata_list
                }
            
            augmented_samples.append(aug_sample)
        
        return augmented_samples
    
   
    def _select_joint_cell(self):
        """Pick (snr_regime, event_type) cell from joint quotas."""
        available = {k: v for k, v in getattr(self, 'joint_quotas', {}).items() if v > 0}
        if not available:
            sr = self._sample_snr_regime()
            et = self._sample_event_type()
            return sr, et

        deterministic = bool(self.config.get('quota_deterministic', False))
        if deterministic:
            keys = sorted(available.keys(), key=lambda k: (available[k], k[0], k[1]), reverse=True)
            chosen = keys[0]
            self.joint_quotas[chosen] -= 1
            return chosen

        keys = list(available.keys())
        vals = np.array([available[k] for k in keys], dtype=float)
        probs = vals / vals.sum()
        idx_choice = np.random.choice(len(keys), p=probs)
        chosen = keys[idx_choice]
        self.joint_quotas[chosen] -= 1
        return chosen

    def _select_from_quotas(self, quotas_dict):
        """Select key from quotas dict, decrementing count."""
        available = {k: v for k, v in quotas_dict.items() if v > 0}
        if not available:
            return None
        keys = list(available.keys())
        vals = np.array([available[k] for k in keys], dtype=float)
        probs = vals / vals.sum()
        idx_choice = np.random.choice(len(keys), p=probs)
        chosen = keys[idx_choice]
        quotas_dict[chosen] -= 1
        return chosen

    def generate_dataset(self,
                    n_samples: int = 1000,
                    overlap_fraction: float = OVERLAP_FRACTION,
                    edge_case_fraction: float = EDGE_CASE_FRACTION,
                    save_batch_size: int = 100,
                    add_glitches: bool = True,
                    preprocess: bool = True,
                    save_complete: bool = False,
                    create_splits: bool = True,
                    train_frac: float = 0.8,
                    val_frac: float = 0.1,
                    test_frac: float = 0.1,
                    chunk_size: int = 100,
                    noise_augmentation_k: int = 1,
                    config: Dict = None) -> Dict:
        """
        Memory-optimized dataset generation with comprehensive tracking and logging.
        Safe for 32GB RAM. Auto-resume capable.
        """
        
        # Adjust for noise augmentation: compute base samples needed
        if noise_augmentation_k > 1:
            base_sample_count = n_samples // noise_augmentation_k
        else:
            base_sample_count = n_samples
        
        n_synthetic = int(base_sample_count * 0.9)
        n_real = base_sample_count - n_synthetic
       
        # Initialize tracking dictionaries
        # event_type_counts: counts samples by top-level sample.type (e.g. 'single', 'overlap', 'BBH', etc.)
        # signal_type_counts: counts individual signals inside samples (BBH/BNS/NSBH) so distribution
        # comparisons reflect per-signal expectations in configs.
        
        event_type_counts = Counter()
        signal_type_counts = Counter()
        snr_regime_counts = Counter()
        edge_case_type_counts = Counter()
        extreme_case_type_counts = Counter()
        
        def _categorize_snr(snr: float) -> str:
            """Categorize SNR into 5 bins: weak, low, medium, high, loud"""
            if snr < 10:
                return 'weak'
            elif snr < 15:
                return 'low'
            elif snr < 25:
                return 'medium'
            elif snr < 40:
                return 'high'
            else:
                return 'loud'

        
        def _track_sample(sample):
            """Track statistics for generated sample"""
            if sample is None:
                return
            # Track top-level sample type (single vs overlap vs noise)
            event_type = sample.get('type', 'unknown')
            event_type_counts[event_type] += 1

            # Track SNR per signal and also maintain per-signal event type counts
            params = sample.get('parameters')
            if params:
                if isinstance(params, list):
                    # Overlapping - track each signal
                    for p in params:
                        if isinstance(p, dict):
                            # signal-level event type (BBH/BNS/NSBH)
                            sig_type = p.get('type') or p.get('event_type') or 'unknown'
                            signal_type_counts[sig_type] += 1
                            if p.get('target_snr') is not None:
                                regime = _categorize_snr(p['target_snr'])
                                snr_regime_counts[regime] += 1
                elif isinstance(params, dict):
                    sig_type = params.get('type') or params.get('event_type') or 'unknown'
                    signal_type_counts[sig_type] += 1
                    if params.get('target_snr') is not None:
                        regime = _categorize_snr(params['target_snr'])
                        snr_regime_counts[regime] += 1

        def _normalize_sample(sample):
            """Normalize parameter dicts inside a sample so downstream analysis finds consistent keys.

            Ensures each per-signal parameter dict has a 'type' key (tries sample-level hints or mass-based inference),
            and normalizes common alternate key names (a_1 -> a1, tilt_1 -> tilt1).
            """
            if sample is None:
                return
            params = sample.get('parameters')

            def _normalize_param_dict(p):
                if not isinstance(p, dict):
                    return p
                # Coerce numpy/string types to native str where appropriate
                for key in ['type', 'event_type', 'category', 'class', 'source_type']:
                    if key in p and p.get(key) is not None:
                        try:
                            val = p.get(key)
                            # numpy strings -> python str
                            if hasattr(val, 'tolist'):
                                val = val.tolist()
                            p['type'] = str(val)
                            break
                        except Exception:
                            continue

                # Ensure 'type' exists; attempt multiple inference strategies
                if 'type' not in p or p.get('type') in [None, '']:
                    # 1) event_type key
                    if 'event_type' in p and p.get('event_type'):
                        p['type'] = str(p.get('event_type'))
                    else:
                        # 2) name prefix (e.g., 'BBH_30_20')
                        name = p.get('name') or p.get('event_name') or ''
                        if isinstance(name, str) and name.upper().startswith(('BBH', 'BNS', 'NSBH')):
                            if name.upper().startswith('BBH'):
                                p['type'] = 'BBH'
                            elif name.upper().startswith('BNS'):
                                p['type'] = 'BNS'
                            elif name.upper().startswith('NSBH'):
                                p['type'] = 'NSBH'
                        else:
                            # 3) mass-based inference (many variants covered)
                            m1 = p.get('mass_1') or p.get('mass1') or p.get('m1')
                            try:
                                if m1 is None:
                                    # Fallback to sample-level type
                                    p['type'] = sample.get('type', 'unknown')
                                else:
                                    m1f = float(m1)
                                    if m1f < 3.0:
                                        p['type'] = 'BNS'
                                    elif m1f < 8.0:
                                        p['type'] = 'NSBH'
                                    else:
                                        p['type'] = 'BBH'
                            except Exception:
                                p['type'] = sample.get('type', 'unknown')
                # Normalize alternate key names
                if 'a_1' in p and 'a1' not in p:
                    p['a1'] = p.pop('a_1')
                if 'a_2' in p and 'a2' not in p:
                    p['a2'] = p.pop('a_2')
                if 'tilt_1' in p and 'tilt1' not in p:
                    p['tilt1'] = p.pop('tilt_1')
                if 'tilt_2' in p and 'tilt2' not in p:
                    p['tilt2'] = p.pop('tilt_2')
                # Also handle alternate mass key names
                if 'mass1' in p and 'mass_1' not in p:
                    p['mass_1'] = p.pop('mass1')
                if 'mass2' in p and 'mass_2' not in p:
                    p['mass_2'] = p.pop('mass2')
                if 'm1' in p and 'mass_1' not in p:
                    p['mass_1'] = p.pop('m1')
                if 'm2' in p and 'mass_2' not in p:
                    p['mass_2'] = p.pop('m2')

                # Ensure numeric numpy scalars are converted to native Python types for JSON/pickle friendliness
                for k, v in list(p.items()):
                    try:
                        if hasattr(v, 'item'):
                            p[k] = v.item()
                    except Exception:
                        pass
                return p

            if isinstance(params, list):
                sample['parameters'] = [_normalize_param_dict(p) for p in params]
            elif isinstance(params, dict):
                sample['parameters'] = _normalize_param_dict(params)
            
            # Track edge/extreme cases
            if sample.get('is_edge_case', False):
                edge_type = sample.get('edge_case_type', 'none')
                edge_case_type_counts[edge_type] += 1
            
            if sample.get('is_extreme_case', False):
                extreme_type = sample.get('extreme_case_type', 'none')
                extreme_case_type_counts[extreme_type] += 1
        
        def _log_current_stats(checkpoint_num, total_generated):
            """Log comprehensive generation statistics with detailed breakdowns"""
            self.logger.info("")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"CHECKPOINT {checkpoint_num}: Generated {total_generated:,} samples")
            elapsed = time.time() - start_time
            rate = total_generated / elapsed if elapsed > 0 else 0
            remaining = total_to_generate - total_generated
            eta = remaining / rate if rate > 0 else 0
            self.logger.info(f"Progress: {total_generated/total_to_generate*100:.1f}% | "
                            f"Rate: {rate:.2f}/s | ETA: {eta/60:.1f}m")
            self.logger.info(f"{'='*80}")
            self.logger.info("")
            
            # ========================================================================
            # EVENT TYPE DISTRIBUTION (signal-level and sample-level)
            # ========================================================================
            self.logger.info("📊 EVENT TYPE DISTRIBUTION:")
            self.logger.info(f"{'─'*80}")

            # Signal-level counts (individual BBH/BNS/NSBH signals)
            total_signals = sum(signal_type_counts.values())
            
            # Use EVENT_TYPE_DISTRIBUTION from config instead of hardcoded values
            expected = EVENT_TYPE_DISTRIBUTION.copy()

            self.logger.info("Signal-level distribution (individual signals):")
            self.logger.info(f"{'Type':<12} {'Count':>8} {'Actual':>8} {'Expected':>8} {'Diff':>8}  {'Status'}")
            self.logger.info(f"{'─'*80}")

            for event_type in ['BBH', 'BNS', 'NSBH', 'noise']:
                count = signal_type_counts.get(event_type, 0)
                actual_pct = (count / total_signals * 100) if total_signals > 0 else 0
                exp_pct = expected.get(event_type, 0.0) * 100
                diff = actual_pct - exp_pct
                status = "✓" if abs(diff) < 10 else "⚠"
                self.logger.info(
                    f"{event_type:<12} {count:>8,} {actual_pct:>7.1f}% "
                    f"{exp_pct:>7.1f}% {diff:>+7.1f}%  {status}"
                )

            self.logger.info(f"{'─'*80}")
            self.logger.info(f"{'TOTAL SIGNALS':<12} {total_signals:>8,}")
            self.logger.info("")

            # Sample-level counts (how many samples are of each top-level type)
            self.logger.info("Sample-level distribution (samples):")
            self.logger.info(f"{'Type':<12} {'Count':>8} {'Actual':>8}")
            self.logger.info(f"{'─'*80}")
            total_samples = sum(event_type_counts.values())
            for event_type in ['single', 'overlap', 'BBH', 'BNS', 'NSBH', 'noise']:
                count = event_type_counts.get(event_type, 0)
                actual_pct = (count / total_samples * 100) if total_samples > 0 else 0
                self.logger.info(f"{event_type:<12} {count:>8,} {actual_pct:>7.1f}%")

            self.logger.info("")
            
            # ========================================================================
            # SNR DISTRIBUTION (5 REGIMES)
            # ========================================================================
            self.logger.info("📈 SNR DISTRIBUTION (Astrophysically Realistic):")
            self.logger.info(f"{'─'*80}")
            
            total_snr = sum(snr_regime_counts.values())
            
            # Use SNR_DISTRIBUTION from config instead of hardcoded values
            # Convert to percentages
            expected_snr = {k: v * 100.0 for k, v in SNR_DISTRIBUTION.items()}
            
            self.logger.info(f"{'Regime':<12} {'Range':>12} {'Count':>8} {'Actual':>8} {'Expected':>8} {'Diff':>8}  {'Status'}")
            self.logger.info(f"{'─'*80}")
            
            snr_ranges = {
                'weak': '8-10 Hz',
                'low': '10-15 Hz',
                'medium': '15-25 Hz',
                'high': '25-40 Hz',
                'loud': '40+ Hz'
            }
            
            for regime in ['weak', 'low', 'medium', 'high', 'loud']:
                count = snr_regime_counts.get(regime, 0)
                actual_pct = (count / total_snr * 100) if total_snr > 0 else 0
                exp_pct = expected_snr[regime]
                diff = actual_pct - exp_pct
                status = "✓" if abs(diff) < 5 else "⚠"
                
                self.logger.info(
                    f"{regime.capitalize():<12} {snr_ranges[regime]:>12} {count:>8,} "
                    f"{actual_pct:>7.1f}% {exp_pct:>7.1f}% {diff:>+7.1f}%  {status}"
                )
            
            self.logger.info(f"{'─'*80}")
            self.logger.info(f"{'TOTAL SIGNALS':<12} {'':<12} {total_snr:>8,}")
            
            if total_snr > 0:
                snr_mean = sum(
                    count * {'weak': 9, 'low': 12.5, 'medium': 20, 'high': 32.5, 'loud': 45}[regime]
                    for regime, count in snr_regime_counts.items()
                ) / total_snr
                self.logger.info(f"Mean SNR (approx): {snr_mean:.1f}")
            
            self.logger.info("")

            # ====================================================================
            # Joint quotas diagnostic (if available)
            # ====================================================================
            try:
                if quota_mode and (bool(self.config.get('quota_verbose', False)) or bool(self.config.get('quota_debug', False))):
                    try:
                        jq = joint_quotas
                        per_snr = {}
                        per_event = {}
                        for (r, e), v in jq.items():
                            per_snr[r] = per_snr.get(r, 0) + v
                            per_event[e] = per_event.get(e, 0) + v

                        self.logger.info('Joint quota remaining (per-snr): ' + str(per_snr))
                        self.logger.info('Joint quota remaining (per-event): ' + str(per_event))
                        try:
                            self.logger.info('\n' + _format_joint_table(jq))
                        except Exception:
                            self.logger.debug(f"Joint quotas raw: {jq}")
                    except NameError:
                        # joint_quotas not defined in scope yet
                        pass
            except Exception:
                pass
            
            # ========================================================================
            # EDGE CASES BREAKDOWN
            # ========================================================================
            if edge_case_type_counts:
                self.logger.info("🔧 EDGE CASE TYPES:")
                self.logger.info(f"{'─'*80}")
                
                edge_categories = {
                    'Physical Extremes': ['high_mass_ratio', 'extreme_spins', 'eccentric_mergers', 
                                        'precessing_systems', 'short_duration_high_mass', 'low_snr_threshold'],
                    'Observational': ['strong_glitches', 'psd_drift', 'detector_dropout', 'sky_position_extremes'],
                    'Statistical': ['multimodal_posteriors', 'heavy_tailed_regions', 'uninformative_priors'],
                    'Overlapping': ['subtle_ranking', 'heavy_overlaps', 'partial_overlaps']
                }
                
                for category, edge_types in edge_categories.items():
                    category_total = sum(edge_case_type_counts.get(et, 0) for et in edge_types)
                    if category_total > 0:
                        self.logger.info(f"\n{category}: {category_total:,} samples")
                        for edge_type in edge_types:
                            count = edge_case_type_counts.get(edge_type, 0)
                            if count > 0:
                                pct = (count / category_total * 100)
                                self.logger.info(f"  {edge_type:<35s}: {count:>6,} ({pct:>5.1f}%)")
                
                self.logger.info(f"{'─'*80}")
                self.logger.info(f"Total edge cases: {sum(edge_case_type_counts.values()):,}")
                self.logger.info("")
            
            # ========================================================================
            # EXTREME CASES BREAKDOWN
            # ========================================================================
            if extreme_case_type_counts:
                self.logger.info("⚡ EXTREME CASE TYPES (Publication Quality):")
                self.logger.info(f"{'─'*80}")
                
                extreme_expected = {
                    'near_simultaneous_mergers': 1.0,
                    'extreme_mass_ratio': 1.0,
                    'high_spin_aligned': 1.0,
                    'precession_dominated': 1.0,
                    'eccentric_overlaps': 0.5,
                    'weak_strong_overlaps': 0.5,
                    'noise_confused_overlaps': 1.0,
                    'long_duration_bns_overlaps': 0.5,
                    'detector_dropouts': 0.5,
                    'cosmological_distance': 0.5
                }
                
                self.logger.info(f"{'Type':<35} {'Count':>8} {'%':>7} {'Expected':>8}  {'Status'}")
                self.logger.info(f"{'─'*80}")
                
                total_extreme = sum(extreme_case_type_counts.values())
                
                for extreme_type in sorted(extreme_case_type_counts.keys()):
                    count = extreme_case_type_counts[extreme_type]
                    actual_pct = (count / total_generated * 100) if total_generated > 0 else 0
                    exp_pct = extreme_expected.get(extreme_type, 0.5)
                    
                    if actual_pct >= exp_pct:
                        status = "✓✓"
                    elif actual_pct >= exp_pct * 0.5:
                        status = "✓"
                    else:
                        status = "⚠"
                    
                    self.logger.info(
                        f"{extreme_type:<35} {count:>8,} {actual_pct:>6.2f}% "
                        f"{exp_pct:>7.1f}%  {status}"
                    )
                
                self.logger.info(f"{'─'*80}")
                self.logger.info(f"{'Total extreme cases':<35} {total_extreme:>8,} "
                                f"{total_extreme/total_generated*100:>6.1f}%")
                self.logger.info("")
            
            # ========================================================================
            # SAMPLE COMPOSITION SUMMARY
            # ========================================================================
            self.logger.info("📦 SAMPLE COMPOSITION:")
            self.logger.info(f"{'─'*80}")
            
            n_single = event_type_counts.get('BBH', 0) + event_type_counts.get('BNS', 0) + \
                    event_type_counts.get('NSBH', 0) + event_type_counts.get('noise', 0)
            n_overlap_actual = event_type_counts.get('overlap', 0)
            n_edge = sum(edge_case_type_counts.values())
            n_extreme = sum(extreme_case_type_counts.values())
            
            self.logger.info(f"Single events:     {n_single:>8,} ({n_single/total_generated*100:>5.1f}%)")
            self.logger.info(f"Overlap events:    {n_overlap_actual:>8,} ({n_overlap_actual/total_generated*100:>5.1f}%)")
            self.logger.info(f"Edge cases:        {n_edge:>8,} ({n_edge/total_generated*100:>5.1f}%)")
            self.logger.info(f"Extreme cases:     {n_extreme:>8,} ({n_extreme/total_generated*100:>5.1f}%)")
            self.logger.info(f"{'─'*80}")
            self.logger.info(f"Total analyzed:    {total_generated:>8,} (100.0%)")
            
            self.logger.info(f"{'='*80}")
            self.logger.info("")

        # Update config if provided
        if config is not None:
            for key, value in config.items():
                if key not in self.config:
                    self.config[key] = value
        
        # Reload extreme config
        self.extreme_config = self.config.get('extreme_cases', {})
        self.extreme_enabled = self.extreme_config.get('enabled', False)
        self.extreme_fraction = self.extreme_config.get('fraction', 0.10)
        self.extreme_types_config = self.extreme_config.get('types', {})
        
        # Check for existing batches
        batch_dir = self.output_dir / 'batches'
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        existing_batch_files = sorted(glob.glob(str(batch_dir / 'batch*.pkl')))
        existing_sample_count = 0
        
        if existing_batch_files:
            self.logger.info("=" * 80)
            self.logger.info("CHECKING EXISTING BATCHES")
            self.logger.info("=" * 80)
            self.logger.info(f"Found {len(existing_batch_files)} existing batch files")
            
            for batch_file in existing_batch_files:
                try:
                    with open(batch_file, 'rb') as f:
                        batch_data = pickle.load(f)
                        
                        if isinstance(batch_data, list):
                            batch_count = len(batch_data)
                        elif isinstance(batch_data, dict):
                            if 'samples' in batch_data:
                                batch_count = len(batch_data['samples'])
                            elif 'data' in batch_data:
                                batch_count = len(batch_data['data'])
                            else:
                                batch_count = 1
                        else:
                            batch_count = 1
                        
                        existing_sample_count += batch_count
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load {Path(batch_file).name}: {e}")
            
            self.logger.info(f"✓ Total existing samples: {existing_sample_count:,}")
            self.logger.info("=" * 80)
            self.logger.info("")
        
        if existing_sample_count >= n_samples:
            self.logger.info("=" * 80)
            self.logger.info("✓ TARGET SAMPLES ALREADY GENERATED!")
            self.logger.info("=" * 80)
            self.logger.info(f"Existing: {existing_sample_count:,} / Target: {n_samples:,}")
            
            if create_splits:
                train_dir = self.output_dir / 'train'
                if not train_dir.exists():
                    self.logger.info("")
                    self.logger.info("Creating splits from existing batches...")
                    self._create_splits_from_batches(train_frac, val_frac, test_frac, chunk_size, noise_augmentation_k)
                else:
                    self.logger.info("Splits already exist - skipping")
            
            self.logger.info("=" * 80)
            
            return {
                'n_samples': existing_sample_count,
                'generation_time': 0,
                'samples_per_second': 0,
                'output_dir': str(self.output_dir),
                'output_format': self.output_format,
                'resumed': True,
                'already_complete': True
            }
        
        # Calculate remaining samples
        remaining = n_samples - existing_sample_count
        
        self.logger.info("=" * 80)
        if existing_sample_count > 0:
            self.logger.info("RESUMING DATASET GENERATION")
        else:
            self.logger.info("STARTING DATASET GENERATION")
        self.logger.info(f"Target: {n_samples:,} samples")
        self.logger.info(f"Remaining: {remaining:,} samples")
        self.logger.info("=" * 80)
        self.logger.info("")
        
        start_time = time.time()
        
        # ========================================================================
        # CALCULATE SAMPLE DISTRIBUTION
        # ========================================================================
        
        n_overlap = int(base_sample_count * overlap_fraction)
        n_regular = base_sample_count - n_overlap
        
        # Pre-merger samples
        premerger_fraction = self.config.get('premerger_fraction', 0.15)
        n_premerger = int(base_sample_count * premerger_fraction)
        
        # Edge case breakdown
        edge_case_counts = {}
        edge_config = self.config.get('edge_cases', {})
        
        self.logger.info("Calculating sample distribution...")
        self.logger.info("")
        
        n_edge_total = int(base_sample_count * edge_case_fraction)
        # Physical extremes
        phys_config = edge_config.get('physical_extremes', {'enabled': True, 'fraction': 0.3, 'types': {'high_mass_ratio': {'fraction': 0.5}, 'extreme_spins': {'fraction': 0.3}, 'eccentric_mergers': {'fraction': 0.2}}})
        if phys_config.get('enabled', True):
            n_phys = int(n_edge_total * phys_config.get('fraction', 0.25))
            self.logger.info(f"Physical extremes: {n_phys:,} samples ({phys_config.get('fraction', 0.15):.1%})")
            for edge_type, type_config in phys_config.get('types', {}).items():
                n_type = int(n_phys * type_config.get('fraction', 0.0))
                if n_type > 0:
                    edge_case_counts[edge_type] = n_type
                    self.logger.info(f"  - {edge_type}: {n_type:,}")
        
        # Observational extremes
        obs_config = edge_config.get('observational_extremes', {'enabled': True, 'fraction': 0.4, 'types': {'strong_glitches': {'fraction': 0.4}, 'psd_drift': {'fraction': 0.3}, 'detector_dropout': {'fraction': 0.3}}})
        if obs_config.get('enabled', True):
            n_obs = int(n_edge_total * obs_config.get('fraction', 0.25))
            self.logger.info(f"Observational extremes: {n_obs:,} samples ({obs_config.get('fraction', 0.10):.1%})")
            for edge_type, type_config in obs_config.get('types', {}).items():
                n_type = int(n_obs * type_config.get('fraction', 0.0))
                if n_type > 0:
                    edge_case_counts[edge_type] = n_type
                    self.logger.info(f"  - {edge_type}: {n_type:,}")
        
        # Statistical extremes
        stat_config = edge_config.get('statistical_extremes', {'enabled': True, 'fraction': 0.2, 'types': {'multimodal_posteriors': {'fraction': 0.6}, 'heavy_tailed_regions': {'fraction': 0.4}}})
        if stat_config.get('enabled', True):
            n_stat = int(n_edge_total * stat_config.get('fraction', 0.10))
            self.logger.info(f"Statistical extremes: {n_stat:,} samples ({stat_config.get('fraction', 0.10):.1%})")
            for edge_type, type_config in stat_config.get('types', {}).items():
                n_type = int(n_stat * type_config.get('fraction', 0.0))
                if n_type > 0:
                    edge_case_counts[edge_type] = n_type
                    self.logger.info(f"  - {edge_type}: {n_type:,}")
        
        # Overlapping extremes
        overlap_config = edge_config.get('overlapping_extremes', {'enabled': True, 'fraction': 0.1, 'types': {'subtle_ranking': {'fraction': 0.5}, 'heavy_overlaps': {'fraction': 0.5}}})
        if overlap_config.get('enabled', True):
            n_overlap_edge = int(n_edge_total * overlap_config.get('fraction', 0.25))
            self.logger.info(f"Overlapping extremes: {n_overlap_edge:,} samples ({overlap_config.get('fraction', 0.10):.1%})")
            for edge_type, type_config in overlap_config.items():
                if edge_type in ['enabled', 'fraction']:
                    continue
                n_type = int(n_overlap_edge * type_config.get('fraction', 0.0))
                if n_type > 0:
                    edge_case_counts[edge_type] = n_type
                    self.logger.info(f"  - {edge_type}: {n_type:,}")
        
        # Adjust regular samples
        total_edge = sum(edge_case_counts.values())
        n_regular_single = max(0, n_regular - total_edge - n_premerger)
        n_regular_overlap = n_overlap
        
        # ========================================================================
        # FINAL BREAKDOWN SUMMARY
        # ========================================================================
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("SAMPLE BREAKDOWN SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Regular single events:    {n_regular_single:>6,} ({n_regular_single/n_samples*100:>5.1f}%)")
        self.logger.info(f"Pre-merger samples:       {n_premerger:>6,} ({n_premerger/n_samples*100:>5.1f}%)")
        self.logger.info(f"Regular overlaps:         {n_regular_overlap:>6,} ({n_regular_overlap/n_samples*100:>5.1f}%)")
        self.logger.info(f"Edge cases (total):       {total_edge:>6,} ({total_edge/n_samples*100:>5.1f}%)")

        if self.extreme_enabled:
            expected_extreme = int(n_samples * self.extreme_fraction)
            self.logger.info(f"Extreme cases (expected): {expected_extreme:>6,} ({self.extreme_fraction*100:>5.1f}%)")

        self.logger.info(f"{'─' * 40}")
        self.logger.info(f"Total:                    {n_samples:>6,} (100.0%)")
        self.logger.info("=" * 80)
        self.logger.info("")

        #  ADD THIS SECTION HERE:
        # ========================================================================
        # GENERATION PLAN (with Deterministic Extremes)
        # ========================================================================
        # Calculate extreme case counts deterministically
        extreme_case_counts = {}
        if self.extreme_enabled:
            # Calculate exact counts for each extreme case type
            n_expected_extremes = int(n_samples * self.extreme_fraction)

            # Calculate counts for each enabled extreme type
            enabled_extreme_types = {
                name: config for name, config in self.extreme_types_config.items() if config.get('enabled', True)
            }

            if enabled_extreme_types:
                for extreme_type, type_config in enabled_extreme_types.items():
                    fraction = type_config.get('fraction', 0.1)
                    n_type = int(n_expected_extremes * fraction)
                    if n_type > 0:
                        extreme_case_counts[extreme_type] = n_type

            total_extreme_cases = sum(extreme_case_counts.values())
            # Adjust regular samples
            n_regular_actual = max(n_regular_single - total_extreme_cases, 0)

            # Compute base logical samples (not augmented)
            base_samples = (
                n_regular_actual
                + total_extreme_cases
                + n_premerger
                + n_regular_overlap
                + total_edge
            )

            # ✅ Final total for progress bar (multiply by augmentation if enabled)
            noise_augmentation_k = int(self.config.get('noise_augmentation_k', 1))
            total_to_generate = base_samples * noise_augmentation_k
        else:
            # No extreme cases
            total_extreme_cases = 0
            n_regular_actual = n_regular_single
            base_samples = n_regular_single + n_premerger + n_regular_overlap + total_edge + n_real
            noise_augmentation_k = int(self.config.get('noise_augmentation_k', 1))
            total_to_generate = base_samples * noise_augmentation_k

        self.preprocess_enabled = preprocess

        # Quota mode configuration: when enabled, enforce SNR-regime and event-type
        # marginals by selecting regimes/types from computed quotas.
        quota_mode = bool(self.config.get('quota_mode', False))
        quota_max_attempts = int(self.config.get('quota_max_attempts', 10))

        # Quota bookkeeping (populated only if quota_mode=True)
        quotas_snr = {}
        quotas_event = {}

        if quota_mode:
            # Estimate total signals to allocate quotas across. Use expected overlap size 2.5
            expected_signals_per_overlap = float(self.config.get('expected_signals_per_overlap', 2.5))
            # By default we enforce quotas only on the regular single + overlapping signals.
            # Pre-merger samples and explicit edge-case samples are generated in separate
            # loops and may not consume quotas unless configured via 'quota_include_extremes'.
            include_extremes = bool(self.config.get('quota_include_extremes', False))
            total_signals_est = int(round(
                n_regular_single + n_regular_overlap * expected_signals_per_overlap + (total_edge if include_extremes else 0)
            ))

            # Per-regime quotas (rounding and distribute remainder)
            regimes = list(SNR_DISTRIBUTION.keys())
            for r in regimes:
                quotas_snr[r] = int(round(total_signals_est * float(SNR_DISTRIBUTION.get(r, 0.0))))
            # Balance rounding error
            rem = total_signals_est - sum(quotas_snr.values())
            idx = 0
            while rem > 0:
                quotas_snr[regimes[idx % len(regimes)]] += 1
                idx += 1
                rem -= 1

            # Per-event-type quotas
            types = list(EVENT_TYPE_DISTRIBUTION.keys())
            for t in types:
                quotas_event[t] = int(round(total_signals_est * float(EVENT_TYPE_DISTRIBUTION.get(t, 0.0))))
            rem2 = total_signals_est - sum(quotas_event.values())
            idx = 0
            while rem2 > 0:
                quotas_event[types[idx % len(types)]] += 1
                idx += 1
                rem2 -= 1
            # Build joint quotas using iterative proportional fitting (IPF)
            # This enforces joint (snr_regime x event_type) quotas consistent with the marginal
            # quotas_snr and quotas_event. The result is an integer table summing to total_signals_est.
            def _compute_joint_quotas(row_counts: Dict[str, int], col_counts: Dict[str, int], total: int):
                # Order rows and cols for deterministic behavior
                rows = list(row_counts.keys())
                cols = list(col_counts.keys())
                r = np.array([row_counts[k] for k in rows], dtype=float)
                c = np.array([col_counts[k] for k in cols], dtype=float)

                # Start with outer product of marginals as initial guess (avoid zeros)
                # Use small epsilon to avoid division by zero
                eps = 1e-12
                M = np.outer(r + eps, c + eps)

                # Normalize to have sum = total (floating)
                if M.sum() <= 0:
                    M = np.ones_like(M)
                M = M / M.sum() * float(total)

                # IPF iterations: alternate row/col scaling
                for _ in range(2000):
                    # scale rows
                    row_sums = M.sum(axis=1)
                    # avoid division by zero
                    row_scale = np.where(row_sums > 0, r / row_sums, 0.0)
                    M = (M.T * row_scale).T

                    # scale cols
                    col_sums = M.sum(axis=0)
                    col_scale = np.where(col_sums > 0, c / col_sums, 0.0)
                    M = M * col_scale

                    # convergence check (max absolute diff of margins)
                    if (np.max(np.abs(M.sum(axis=1) - r)) < 1e-6) and (np.max(np.abs(M.sum(axis=0) - c)) < 1e-6):
                        break

                # Now round to integers while preserving total via largest fractional parts
                floored = np.floor(M).astype(int)
                remainder = int(total - floored.sum())
                if remainder > 0:
                    # distribute remaining counts by fractional parts
                    fracs = (M - np.floor(M)).flatten()
                    indices = np.argsort(-fracs)[:remainder]
                    for idx in indices:
                        i = idx // M.shape[1]
                        j = idx % M.shape[1]
                        floored[i, j] += 1

                # Build dict mapping (row, col) -> count
                joint = {}
                for i, row in enumerate(rows):
                    for j, col in enumerate(cols):
                        joint[(row, col)] = int(floored[i, j])
                return joint

            joint_quotas = _compute_joint_quotas(quotas_snr, quotas_event, total_signals_est)
            self.joint_quotas = joint_quotas
            self.event_quotas = quotas_event.copy()

            # Diagnostic formatter for joint_quotas
            def _format_joint_table(joint_table):
                rows = sorted(set(r for r, _ in joint_table.keys()))
                cols = sorted(set(c for _, c in joint_table.keys()))
                header = ['SNR\\Event'] + cols
                lines = ['\t'.join(header)]
                for r in rows:
                    row = [r] + [str(joint_table.get((r, c), 0)) for c in cols]
                    lines.append('\t'.join(row))
                return '\n'.join(lines)

            if bool(self.config.get('quota_verbose', False)) or bool(self.config.get('quota_debug', False)):
                try:
                    self.logger.info('Joint quotas (snr_regime x event_type):\n' + _format_joint_table(joint_quotas))
                except Exception:
                    self.logger.info('Joint quotas: ' + str(joint_quotas))


            # Keep deprecated helpers for compatibility (they will consult joint_quotas)
            def _select_snr_from_quota():
                # pick joint cell then return its snr
                snr_choice, _ = self._select_joint_cell()
                return snr_choice

            def _select_event_for_snr(snr_regime):
                # Prefer event types in joint_quotas for the given snr_regime
                # Try to find any (snr_regime, event) with positive quota
                candidates = [(k, v) for k, v in joint_quotas.items() if k[0] == snr_regime and v > 0]
                if candidates:
                    # choose proportional to remaining
                    keys = [k for k, _ in candidates]
                    vals = np.array([v for _, v in candidates], dtype=float)
                    probs = vals / vals.sum()
                    idx = np.random.choice(len(keys), p=probs)
                    chosen = keys[idx][1]
                    joint_quotas[(snr_regime, chosen)] -= 1
                    return chosen

                # fallback to selecting any joint cell
                sr, et = self._select_joint_cell()
                return et

        # ========================================================================
        # MEMORY-OPTIMIZED GENERATION WITH TRACKING
        # ========================================================================
        samples = []
        batch_id = len(existing_batch_files)
        sample_id = existing_sample_count
        total_generated = 0

        last_log_time = time.time()
        log_interval = 600  # Log stats every 10 minutes
        checkpoint_interval = 5000  # Detailed stats every 5000 samples
        forced_signals = None

        self.logger.info("=" * 80)
        self.logger.info("STARTING SAMPLE GENERATION")
        self.logger.info("=" * 80)
        self.logger.info("")

        
        with tqdm(total=total_to_generate, desc="Generating samples",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            # Helper function for noise augmentation
            def add_sample_with_augmentation(sample, samples_list):
                """Add sample to list with optional noise augmentation"""
                _normalize_sample(sample)
                _track_sample(sample)

                if noise_augmentation_k > 1:
                    augmented_samples = self.create_noise_augmentations(sample, noise_augmentation_k)
                    samples_list.extend(augmented_samples)
                    return len(augmented_samples)  # Return total samples added
                else:
                    samples_list.append(sample)
                    return 1
            
            # [1/3] Generate regular single samples and deterministic extreme cases
            if self.extreme_enabled and extreme_case_counts:
                self.logger.info(f"[1/3] Generating deterministic extreme cases...")
                for extreme_type, count in extreme_case_counts.items():
                    if count > 0:
                        # 24 spaces (6 levels)
                        self.logger.info(f"  Generating {count:,} {extreme_type} samples...")
                        for i in range(count):
                            # 28 spaces (7 levels) - ALL content inside for loop
                            sample = self._generate_extreme_case(sample_id, extreme_type)
                            n_samples_added = add_sample_with_augmentation(sample, samples)
                            sample_id += 1
                            total_generated += n_samples_added
                            pbar.update(n_samples_added)
                            
                            # 28 spaces - checkpoint check
                            if total_generated % checkpoint_interval == 0:
                                # 32 spaces (8 levels)
                                _log_current_stats(total_generated // checkpoint_interval, total_generated)
                            
                            # 28 spaces - batch save
                            if len(samples) >= save_batch_size:
                                # 32 spaces
                                self._save_batch(batch_id, samples)
                                batch_id += 1
                                samples = []
                                gc.collect()
                            
                            # 28 spaces - timing
                            current_time = time.time()
                            if current_time - last_log_time >= log_interval:
                                # 32 spaces
                                self._log_progress(start_time, total_generated, total_to_generate)
                                last_log_time = current_time


                        # Generate pure regular samples (non-extreme)
            n_pure_regular = n_regular_single - total_extreme_cases
            # Reserve some for real GWTC events
            # n_gwtc = min(int(n_pure_regular * 0.05), 50)  # 5% or up to 50 real events
            n_synthetic_regular = n_regular_single - total_extreme_cases

            if n_synthetic_regular > 0:
                self.logger.info(f"[2/3] Generating {n_synthetic_regular:,} synthetic regular samples...")
                for i in range(n_synthetic_regular):
                    if quota_mode:
                        snr_choice, evt_choice = self._select_joint_cell()
                        sample = self._generate_single_sample(
                            sample_id=sample_id,
                            is_edge_case=False,
                            add_glitches=add_glitches,
                            preprocess=preprocess,
                            snr_regime=snr_choice,
                            forced_event_type=evt_choice
                        )
                        sample = self._ensure_sample_priorities(sample)
                    else:
                        if self.simulation is not None:
                            try:
                                sample = self._generate_sample_with_simulator(
                                    sample_id,
                                    n_signals=1
                                )
                            except Exception as e:
                                self.logger.warning(f"Simulator failed for sample {sample_id}: {e}")
                                sample = self._generate_single_sample(
                                    sample_id=sample_id,
                                    is_edge_case=False,
                                    add_glitches=add_glitches,
                                    preprocess=preprocess
                                )
                                sample = self._ensure_sample_priorities(sample)
                        else:
                            sample = self._generate_single_sample(
                                sample_id=sample_id,
                                is_edge_case=False,
                                add_glitches=add_glitches,
                                preprocess=preprocess
                            )
                            priorities = []
                            parameters = sample.get('parameters', [])
                            if not isinstance(parameters, list):
                                parameters = [parameters] if parameters else []
                            for params in parameters:
                                if not isinstance(params, dict):
                                    continue
                                priority = self._estimate_snr_from_params(params)
                                if 'target_snr' not in params:
                                    self._set_target_snr(params, priority, reason='regular_single_assign')
                                priorities.append(priority)
                            sample['priorities'] = priorities if priorities else [15.0]

                    # ✅ Always add the sample regardless of the generation path
                    n_samples_added = add_sample_with_augmentation(sample, samples)
                    sample_id += 1
                    total_generated += n_samples_added
                    pbar.update(n_samples_added)

                    # Optimized: Save less frequently to reduce I/O overhead
                    if len(samples) >= save_batch_size * 2:
                        self._save_batch(batch_id, samples)
                        batch_id += 1
                        samples = []
                        gc.collect()

                    if total_generated % checkpoint_interval == 0:
                        _log_current_stats(total_generated // checkpoint_interval, total_generated)

                    current_time = time.time()
                    if current_time - last_log_time >= log_interval:
                        self._log_progress(start_time, total_generated, total_to_generate)
                        last_log_time = current_time

            

            # [3/3] Generate pre-merger samples
            if n_premerger > 0:
                self.logger.info(f"[3/3] Generating {n_premerger:,} pre-merger samples...")
                
                for i in range(n_premerger):
                    sample = self._generate_pre_merger_sample(sample_id)
                    n_samples_added = add_sample_with_augmentation(sample, samples)
                    total_generated += n_samples_added

                    # ✅ Progress bar increments by one base event, not augmented copies
                    pbar.update(1)
                    
                    if total_generated % checkpoint_interval == 0:
                        _log_current_stats(total_generated // checkpoint_interval, total_generated)
                    
                    if len(samples) >= save_batch_size:
                        self._save_batch(batch_id, samples)
                        batch_id += 1
                        samples = []
                        gc.collect()
                    
                    current_time = time.time()
                    if current_time - last_log_time >= log_interval:
                        self._log_progress(start_time, total_generated, total_to_generate)
                        last_log_time = current_time

            # [2.5/3] Generate explicit edge cases
            for edge_type, count in edge_case_counts.items():
                if count > 0:
                    self.logger.info(f"Generating {count:,} {edge_type} edge cases...")
                    for i in range(count):
                        sample = self._generate_edge_case(sample_id, edge_type)
                        n_samples_added = add_sample_with_augmentation(sample, samples)
                        sample_id += 1
                        total_generated += n_samples_added
                        pbar.update(n_samples_added)

                        if len(samples) >= save_batch_size:
                            self._save_batch(batch_id, samples)
                            batch_id += 1
                            samples = []
                            gc.collect()

            # [3/3] Generate regular overlapping samples
            if n_regular_overlap > 0:
                self.logger.info(f"[3/3] Generating {n_regular_overlap:,} overlapping samples...")

                # Prepare tasks for parallel generation
                overlap_tasks = []
                for i in range(n_regular_overlap):
                    task = {
                        'sample_id': sample_id + i,
                        'quota_mode': quota_mode,
                        'add_glitches': add_glitches,
                        'preprocess': preprocess,
                        'simulation': self.simulation
                    }
                    overlap_tasks.append(task)

                # Generate overlapping samples sequentially
                for t in overlap_tasks:
                    sample_id = t['sample_id']
                    quota_mode = t['quota_mode']
                    add_glitches = t['add_glitches']
                    preprocess = t['preprocess']
                    simulation = t['simulation']

                    if quota_mode:
                        n_sigs = np.random.choice([2, 3, 4], p=[0.6, 0.3, 0.1])
                        forced_signals = []
                        for _ in range(n_sigs):
                            evt_choice = self._select_from_quotas(self.event_quotas)
                            if evt_choice is None:
                                evt_choice = self._sample_event_type()
                            forced_signals.append({'event_type': evt_choice})
                        sample = self._generate_overlapping_sample(sample_id, False, add_glitches, preprocess, forced_signals=forced_signals)
                    else:
                        if simulation is not None:
                            try:
                                n_sigs = np.random.choice([2, 3, 4], p=[0.6, 0.3, 0.1])
                                sample = self._generate_sample_with_simulator(sample_id, n_signals=n_sigs)
                            except Exception:
                                sample = self._generate_overlapping_sample(sample_id, False, add_glitches, preprocess)
                        else:
                            sample = self._generate_overlapping_sample(sample_id, False, add_glitches, preprocess)

                    # Assign priorities
                    if sample and 'priorities' not in sample:
                        priorities = []
                        for params in sample.get('parameters', []):
                            if not isinstance(params, dict): continue
                            priority = self._estimate_snr_from_params(params)
                            if 'target_snr' not in params:
                                self._set_target_snr(params, priority)
                            priorities.append(priority)
                        sample['priorities'] = priorities

                    if sample:
                        n_samples_added = add_sample_with_augmentation(sample, samples)
                        total_generated += n_samples_added
                        pbar.update(n_samples_added)

                        if total_generated % checkpoint_interval == 0:
                            _log_current_stats(total_generated // checkpoint_interval, total_generated)

                        if len(samples) >= save_batch_size:
                            self._save_batch(batch_id, samples)
                            batch_id += 1
                            samples = []
                            gc.collect()

            
                # =======================================================================
                # 4. GENERATE REAL GWTC EVENTS
                # =======================================================================

            # Generate real events from GWTC catalogs
            if n_real > 0:
                self.logger.info(f" Generating {n_real:,} real GWTC events...")
                samples = []  # Reset batch

                successful_real_events = 0
                for attempt in range(n_real * 2):  # Allow up to 2x attempts to get n_real successful samples
                    if successful_real_events >= n_real:
                        break

                    try:
                        sample = self.generate_sample_from_gwtc(sample_id)
                        if sample is not None:
                            _normalize_sample(sample)
                            _track_sample(sample)
                            samples.append(sample)
                            successful_real_events += 1
                            total_generated += 1

                            # Save in batches
                            if len(samples) >= save_batch_size:
                                self._save_batch(batch_id, samples)
                                batch_id += 1
                                samples = []
                                gc.collect()

                            # Progress logging
                            if total_generated % checkpoint_interval == 0:
                                _log_current_stats(total_generated // checkpoint_interval, total_generated)

                            current_time = time.time()
                            if current_time - last_log_time >= log_interval:
                                self._log_progress(start_time, total_generated, total_to_generate)
                                last_log_time = current_time
                        else:
                            self.logger.debug(f"GWTC event {sample_id} skipped (invalid parameters)")

                    except Exception as e:
                        self.logger.warning(f"Error generating GWTC sample {sample_id}: {e}")

                    sample_id += 1  # Always increment sample_id

                if successful_real_events < n_real:
                    self.logger.warning(f"Only generated {successful_real_events}/{n_real} real GWTC events due to data quality issues")

                self.logger.info(f"✓ Generated {successful_real_events:,} real GWTC events")

            # Decorrelate distances and save in batches
            samples = self.stratified_decorrelate_distances(samples)
            while samples:
                batch = samples[:save_batch_size]
                self._save_batch(batch_id, batch)
                batch_id += 1
                samples = samples[save_batch_size:]
                gc.collect()
        
        generation_time = time.time() - start_time
        total_samples = existing_sample_count + total_generated
        
        # ========================================================================
        # FINAL STATISTICS
        # ========================================================================
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("FINAL GENERATION STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"✓ Generated {total_generated:,} new samples in {generation_time/60:.1f}m")
        self.logger.info(f"  Rate: {total_generated/generation_time:.2f} samples/sec")
        self.logger.info("")
        
        # Event type final summary: show signal-level (individual signals) and sample-level
        self.logger.info("Event Type Distribution (Final):")
        total_signals = sum(signal_type_counts.values())
        self.logger.info("  Signal-level (individual signals):")
        for event_type in ['BBH', 'BNS', 'NSBH', 'noise']:
            count = signal_type_counts.get(event_type, 0)
            pct = (count / total_signals * 100) if total_signals > 0 else 0
            self.logger.info(f"    {event_type:8s}: {count:6,} ({pct:5.1f}%)")

        self.logger.info("  Sample-level (samples):")
        for event_type in sorted(event_type_counts.keys()):
            count = event_type_counts[event_type]
            pct = (count / total_generated * 100) if total_generated > 0 else 0
            self.logger.info(f"    {event_type:8s}: {count:6,} ({pct:5.1f}%)")
        
        # SNR final summary
        total_snr = sum(snr_regime_counts.values())
        self.logger.info("")
        self.logger.info("SNR Distribution (Final):")
        for regime in ['weak', 'low', 'medium', 'high', 'loud']:
            count = snr_regime_counts.get(regime, 0)
            pct = (count / total_snr * 100) if total_snr > 0 else 0
            expected_pct = {'weak': 15, 'low': 35, 'medium': 30, 'high': 15, 'loud': 5}[regime]
            diff = abs(pct - expected_pct)
            status = "✓" if diff < 5 else "⚠"
            self.logger.info(f"  {regime.capitalize():10s}: {count:6,} ({pct:5.1f}%) [expect {expected_pct}%] {status}")
        
        # Edge/Extreme case summary
        if edge_case_type_counts:
            self.logger.info("")
            self.logger.info("Edge Case Summary:")
            for edge_type, count in sorted(edge_case_type_counts.items()):
                self.logger.info(f"  {edge_type:30s}: {count:4,}")
        
        if extreme_case_type_counts:
            self.logger.info("")
            self.logger.info("Extreme Case Summary:")
            for extreme_type, count in sorted(extreme_case_type_counts.items()):
                self.logger.info(f"  {extreme_type:30s}: {count:4,}")
        
        self.logger.info("=" * 80)
        self.logger.info("")
        
        # CREATE SPLITS
        if create_splits:
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("CREATING SPLITS FROM BATCHES")
            self.logger.info("=" * 80)
            self._create_splits_from_batches(train_frac, val_frac, test_frac, chunk_size, noise_augmentation_k)
        
        # Save PSDs
        self.logger.info("")
        self.logger.info("Saving detector PSDs...")
        psd_dir = self.output_dir / 'detector_psds'
        psd_dir.mkdir(exist_ok=True)
        
        for detector_name, psd_info in self.psds.items():
            psd_file = psd_dir / f'{detector_name}_psd.npz'
            psd_array = psd_info['psd']
            if hasattr(psd_array, 'numpy'):
                psd_array = psd_array.numpy()
            
            np.savez(psd_file,
                    frequencies=psd_info['frequencies'],
                    psd=psd_array,
                    source=psd_info['source'],
                    name=psd_info['name'])
        
        self.logger.info("✓ PSDs saved")
        
        # Generate summary with statistics
        summary = {
            'n_samples': total_samples,
            'n_batches': batch_id + 1,
            'generation_time': generation_time,
            'samples_per_second': total_generated / generation_time if generation_time > 0 else 0,
            'output_dir': str(self.output_dir),
            'output_format': self.output_format,
            'elapsed_time': generation_time,
            'resumed': existing_sample_count > 0,
            'configuration': {
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'detectors': self.detectors,
                'overlap_fraction': overlap_fraction,
                'edge_case_fraction': edge_case_fraction,
                'premerger_fraction': premerger_fraction,
                'edge_cases': edge_case_counts
            },
            'statistics': {
                'event_types': dict(event_type_counts),
                'snr_regimes': dict(snr_regime_counts),
                'edge_cases': dict(edge_case_type_counts),
                'extreme_cases': dict(extreme_case_type_counts)
            }
        }

        # Attach joint_quotas to summary if present (convert to nested mapping)
        try:
            if quota_mode and 'joint_quotas' in locals():
                jq = locals().get('joint_quotas')
                nested = {}
                for (r, e), v in jq.items():
                    nested.setdefault(r, {})[e] = int(v)
                summary['statistics']['joint_quotas'] = nested
        except Exception:
            # best-effort: skip adding joint quotas
            pass
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("DATASET GENERATION COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Total samples: {total_samples:,}")
        self.logger.info(f"Generation time: {generation_time/60:.1f}m")
        self.logger.info("=" * 80)
        
        self.writer.save_json('generation_summary.json', summary)
        return summary



    def stratified_decorrelate_distances(self, samples: List = None):
        """
        Breaks mass–distance correlation within each event type
        while preserving each type's overall distance range and histogram.
        Uses parallel processing for computations.
        """
        max_workers = min(multiprocessing.cpu_count(), 4)  # Limit for I/O/computation tasks

        groups = defaultdict(list)
        for i, s in enumerate(samples):
            event_type = s.get('type') or s.get('parameters', {}).get('type', 'unknown')
            if event_type != 'noise':  # Skip noise samples
                groups[event_type].append(i)

        # Prepare tasks for parallel processing
        tasks = []
        for t, idxs in groups.items():
            if len(idxs) < 5:
                continue
            dvals = []
            for i in idxs:
                sample = samples[i]
                # All signal samples have parameters nested
                if 'parameters' in sample and isinstance(sample['parameters'], dict) and 'luminosity_distance' in sample['parameters']:
                    dvals.append(sample['parameters']['luminosity_distance'])
                else:
                    self.logger.warning(f"Sample {i} missing luminosity_distance in parameters, skipping")
                    continue
            if dvals:
                tasks.append((t, idxs, dvals))

        # Process in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(self._decorrelate_type, task): task for task in tasks}
            for future in as_completed(future_to_task):
                t, idxs, shuffled_dvals = future.result()
                for i, d in zip(idxs, shuffled_dvals):
                    sample = samples[i]
                    # All signal samples have parameters nested
                    if 'parameters' in sample and isinstance(sample['parameters'], dict):
                        params = sample['parameters']
                        params['luminosity_distance'] = float(d)
                        try:
                            z = calculate_redshift(d)
                            d_C = calculate_comoving_distance(z) if z else d
                        except Exception:
                            z = max(0.0, d * COSMO_H0 / 3e5)
                            d_C = d / (1.0 + z) if z > 0 else d
                        params['redshift'] = float(z)
                        params['comoving_distance'] = float(min(d_C, d))
        return samples

    def _decorrelate_type(self, task):
        """Worker function for parallel decorrelation of one type."""
        t, idxs, dvals = task
        np.random.shuffle(dvals)
        return t, idxs, dvals

    def _generate_edge_case_sample(self, sample_id: int, edge_type: str, config: Dict) -> Dict:
        """Route to appropriate edge case generator."""
        
        # Physical extremes
        if edge_type == 'high_mass_ratio':
            return self._generate_high_mass_ratio_sample(sample_id, config)
        elif edge_type == 'extreme_spins':
            return self._generate_extreme_spin_sample(sample_id, config)
        elif edge_type == 'eccentric_mergers':
            return self._generate_eccentric_merger_sample(sample_id, config)
        elif edge_type == 'precessing_systems':
            return self._generate_precessing_system_sample(sample_id, config)
        elif edge_type == 'short_duration_high_mass':
            return self._generate_short_duration_high_mass_sample(sample_id, config)
        elif edge_type == 'low_snr_threshold':
            return self._generate_low_snr_threshold_sample(sample_id, config)
        
        # Observational extremes
        elif edge_type == 'strong_glitches':
            return self._generate_strong_glitch_sample(sample_id, config)
        elif edge_type == 'detector_dropout':
            return self._generate_detector_dropout_sample(sample_id, config)
        elif edge_type == 'psd_drift':
            return self._generate_psd_drift_sample(sample_id, config)
        elif edge_type == 'sky_position_extremes':
            return self._generate_sky_position_extreme_sample(sample_id, config)
        
        # Statistical extremes
        elif edge_type == 'multimodal_posteriors':
            return self._generate_multimodal_posterior_sample(sample_id, config)
        elif edge_type == 'heavy_tailed_regions':
            return self._generate_heavy_tailed_sample(sample_id, config)
        elif edge_type == 'uninformative_priors':
            return self._generate_uninformative_prior_sample(sample_id, config)
        
        # Overlapping extremes
        elif edge_type == 'subtle_ranking':
            return self._generate_subtle_ranking_overlap(sample_id, config)
        elif edge_type == 'heavy_overlaps':
            return self._generate_heavy_overlap(sample_id, config)
        elif edge_type == 'partial_overlaps':
            return self._generate_partial_overlap_sample(sample_id, config)
        
        else:
            self.logger.warning(f"Unknown edge case type: {edge_type}")
            return None


    def _log_progress(self, start_time, generated, total):
        """Log generation progress."""
        elapsed = time.time() - start_time
        rate = generated / elapsed if elapsed > 0 else 0
        eta = (total - generated) / rate if rate > 0 else 0
        
        self.logger.info(
            f"Rate: {rate:.2f} samples/s | "
            f"ETA: {eta/60:.1f}m | "
            f"Elapsed: {elapsed/60:.1f}m"
        )


    def _create_splits_from_batches(self, train_frac, val_frac, test_frac, chunk_size, noise_augmentation_k):
        """Load batches and create splits (memory efficient)."""
        import gc
        
        self.logger.info("Creating splits from saved batches...")
        
        # Load all batches
        all_samples = self._load_all_batches()
        
        if not all_samples:
            self.logger.error("No samples to split!")
            return
        
        self.logger.info(f"Loaded {len(all_samples):,} samples total")
        
        # Create splits
        splits = self._create_splits(all_samples, train_frac, val_frac, test_frac, stratify=True)
        
        # Clear all_samples from memory
        del all_samples
        gc.collect()
        
        # Save splits in chunks
        compress = (self.output_format == 'pkl_compressed')
        
        for split_name in ['train', 'validation', 'test']:
            split_samples = splits[split_name]['samples']
            
            split_metadata = {
                'split': split_name,
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'detectors': self.detectors,
                'total_samples': len(split_samples),
                'chunk_size': chunk_size
            }
            
            self.logger.info(f"Saving {split_name} split: {len(split_samples):,} samples...")
            
            self.writer.save_split_chunks(
                split_name,
                split_samples,
                split_metadata,
                chunk_size=chunk_size,
                compress=compress
            )
            
            # Clear samples from memory
            splits[split_name]['samples'] = []
            gc.collect()
        
        # Save split indices
        split_indices = {
            'train': splits['train']['indices'],
            'validation': splits['validation']['indices'],
            'test': splits['test']['indices']
        }
        
        import json
        with open(self.output_dir / 'split_indices.json', 'w') as f:
            json.dump(split_indices, f, indent=2)
        
        self.logger.info("✓ All splits saved")


    def _load_all_batches(self):
        """Load all batches from disk."""
        import glob
        import pickle
        
        batch_dir = self.output_dir / 'batches'
        batch_files = sorted(glob.glob(str(batch_dir / 'batch*.pkl')))
        
        if not batch_files:
            self.logger.error(f"No batch files found in {batch_dir}")
            return []
        
        self.logger.info(f"Loading {len(batch_files)} batches...")
        all_samples = []
        
        for batch_file in tqdm(batch_files, desc="Loading batches"):
            try:
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    if isinstance(batch_data, list):
                        all_samples.extend(batch_data)
                    elif isinstance(batch_data, dict) and 'samples' in batch_data:
                        all_samples.extend(batch_data['samples'])
            except Exception as e:
                self.logger.warning(f"Failed to load {batch_file}: {e}")
        
        self.logger.info(f"Loaded {len(all_samples):,} samples")
        return all_samples


    def _create_splits(self, 
                   all_samples: List[Dict],
                   train_frac: float = 0.8,
                   val_frac: float = 0.1,
                   test_frac: float = 0.1,
                   stratify: bool = True) -> Dict:
        """
        Split dataset into train/val/test with optional stratification.
        """
        
        import random
        
        #  ADD THIS: Filter out None samples
        all_samples = [s for s in all_samples if s is not None]
        
        if len(all_samples) == 0:
            raise ValueError("No valid samples to create splits!")
        
        n_total = len(all_samples)
        self.logger.info(f"Creating splits from {n_total} valid samples")
        
        if stratify:
            # Group samples by event type
            type_groups = {}
            for i, sample in enumerate(all_samples):
                #  Extra safety check (though should be unnecessary now)
                if sample is None:
                    continue
                
                event_type = sample.get('type', 'unknown')
                if event_type not in type_groups:
                    type_groups[event_type] = []
                type_groups[event_type].append(i)
            
            # Split each group proportionally
            train_indices = []
            val_indices = []
            test_indices = []
            
            for event_type, indices in type_groups.items():
                random.shuffle(indices)
                n_type = len(indices)
                
                n_train = int(n_type * train_frac)
                n_val = int(n_type * val_frac)
                
                train_indices.extend(indices[:n_train])
                val_indices.extend(indices[n_train:n_train + n_val])
                test_indices.extend(indices[n_train + n_val:])
            
            # Shuffle the splits
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)
            
        else:
            # Simple random split
            indices = list(range(n_total))
            random.shuffle(indices)
            
            n_train = int(n_total * train_frac)
            n_val = int(n_total * val_frac)
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
        
        #  Filter None when creating splits (extra safety)
        splits = {
            'train': {
                'samples': [all_samples[i] for i in train_indices if all_samples[i] is not None],
                'indices': train_indices,
                'n_samples': len(train_indices)
            },
            'validation': {
                'samples': [all_samples[i] for i in val_indices if all_samples[i] is not None],
                'indices': val_indices,
                'n_samples': len(val_indices)
            },
            'test': {
                'samples': [all_samples[i] for i in test_indices if all_samples[i] is not None],
                'indices': test_indices,
                'n_samples': len(test_indices)
            }
        }
        
        self.logger.info("Dataset splits created:")
        self.logger.info(f"  Train:      {len(train_indices)} samples ({len(train_indices)/n_total*100:.1f}%)")
        self.logger.info(f"  Validation: {len(val_indices)} samples ({len(val_indices)/n_total*100:.1f}%)")
        self.logger.info(f"  Test:       {len(test_indices)} samples ({len(test_indices)/n_total*100:.1f}%)")
        
        return splits


    # ============================================================================
    # 1. PHYSICAL EXTREMES
    # ============================================================================

    def _generate_high_mass_ratio_sample(self, sample_id: int, config: Dict) -> Dict:
        """Generate sample with extreme mass ratio (q << 1)."""
        q = np.random.uniform(0.01, 0.1)
        mass_1 = np.random.uniform(5.0, 50.0)
        mass_2 = mass_1 * q
        
        params = self.parameter_sampler.sample_bbh_parameters('medium', False)
        params['mass_1'] = float(mass_1)
        params['mass_2'] = float(mass_2)
        params['mass_ratio'] = float(q)
        
        return self._generate_sample_from_params(sample_id, params, 'high_mass_ratio')


    def _generate_psd_drift_sample(self, sample_id: int, config: Dict) -> Dict:
        """Generate sample with PSD drift."""
        
        edge_cases = config.get('edge_cases', {})
        obs_config = edge_cases.get('observational_extremes', {})
        types = obs_config.get('types', {})
        psd_config = types.get('psd_drift', {})
        
        use_multiple_epochs = psd_config.get('use_multiple_epochs', True)
        
        event_type = self._sample_event_type()
        params = self._sample_parameters(event_type)
        
        detector_data = {}
        
        for detector_name in self.detectors:
            psd_info = self.psds[detector_name]
            
            if use_multiple_epochs:
                half_duration = self.duration / 2
                half_samples = int(half_duration * self.sample_rate)
                
                noise1 = self.noise_generator.generate_colored_noise(psd_info)[:half_samples]
                
                drift_factor = np.random.uniform(0.5, 2.0)
                psd_drifted = psd_info['psd'] * drift_factor
                psd_info_drift = {
                    'psd': psd_drifted,
                    'frequencies': psd_info['frequencies']
                }
                noise2 = self.noise_generator.generate_colored_noise(psd_info_drift)[:half_samples]
                
                combined_noise = np.concatenate([noise1, noise2])
            else:
                combined_noise = np.zeros(int(self.duration * self.sample_rate))
                n_samples = len(combined_noise)
                
                for i in range(n_samples):
                    drift_factor = 1.0 + (i / n_samples) * np.random.uniform(-0.5, 0.5)
                    psd_t = psd_info['psd'] * drift_factor
                    combined_noise[i] = np.random.normal(0, np.sqrt(np.mean(psd_t)))
            
            signal = self.waveform_generator.generate_waveform(params, detector_name)
            combined_strain = combined_noise + signal
            
            if self.preprocess_enabled:
                combined_strain = self.preprocessor.preprocess(combined_strain, psd_info)
            
            detector_data[detector_name] = {
                'strain': combined_strain.astype(np.float32),
                'psd': psd_info['psd'],
                'frequencies': psd_info['frequencies']
            }
        
        #  Calculate priority
        priority = self._estimate_snr_from_params(params)
        if 'target_snr' not in params:
            self._set_target_snr(params, priority, reason='psd_drift_assign')
        
        # Attach network_snr
        attach_network_snr_safe(params)
        # Validate network_snr is present
        if isinstance(params, dict) and ('network_snr' not in params or params['network_snr'] is None):
            params['network_snr'] = 15.0  # Fallback
        elif isinstance(params, list):
            for p in params:
                if isinstance(p, dict) and ('network_snr' not in p or p['network_snr'] is None):
                    p['network_snr'] = 15.0  # Fallback
        
        return {
            'sample_id': f'psd_drift_{sample_id:06d}',  #  Fixed
            'type': event_type,
            'is_overlap': False,
            'n_signals': 1,  #  Added
            'is_edge_case': True,
            'edge_case_type': 'psd_drift',
            'parameters': params,  # Dict for single
            'priorities': [priority],  #  Added
            'detector_data': detector_data,
            'metadata': {
                'sample_id': sample_id,  #  Fixed
                'event_type': event_type,
                'psd_drift': True,
                'multiple_epochs': use_multiple_epochs
            }
        }

    def _generate_sky_position_extreme_sample(self, sample_id: int, config: Dict) -> Dict:
        """Generate sample with extreme sky position."""
        
        edge_cases = config.get('edge_cases', {})
        obs_config = edge_cases.get('observational_extremes', {})
        types = obs_config.get('types', {})
        sky_config = types.get('sky_position_extremes', {})
        
        use_uniform_sky = sky_config.get('use_uniform_sky', True)
        
        event_type = self._sample_event_type()
        params = self._sample_parameters(event_type)
        
        if use_uniform_sky:
            params['ra'] = np.random.uniform(0, 2 * np.pi)
            params['dec'] = np.arcsin(np.random.uniform(-1, 1))
        else:
            if np.random.random() < 0.5:
                params['dec'] = np.random.uniform(np.pi/2 - 0.2, np.pi/2)
            else:
                params['dec'] = np.random.uniform(-np.pi/2, -np.pi/2 + 0.2)
            params['ra'] = np.random.uniform(0, 2 * np.pi)
        
        detector_data = {}
        
        for detector_name in self.detectors:
            psd_info = self.psds[detector_name]
            noise = self.noise_generator.generate_colored_noise(psd_info)
            signal = self.waveform_generator.generate_waveform(params, detector_name)
            combined_strain = noise + signal
            
            if self.preprocess_enabled:
                combined_strain = self.preprocessor.preprocess(combined_strain, psd_info)
            
            detector_data[detector_name] = {
                'strain': combined_strain.astype(np.float32),
                'psd': psd_info['psd'],
                'frequencies': psd_info['frequencies']
            }
        
        #  Calculate priority
        priority = self._estimate_snr_from_params(params)
        if 'target_snr' not in params:
            self._set_target_snr(params, priority, reason='sky_extreme_assign')
        
        # Attach network_snr
        attach_network_snr_safe(params)
        
        return {
            'sample_id': f'sky_extreme_{sample_id:06d}',  #  Fixed
            'type': event_type,
            'is_overlap': False,
            'n_signals': 1,  #  Added
            'is_edge_case': True,
            'edge_case_type': 'sky_position_extremes',
            'parameters': params,  # Dict for single
            'priorities': [priority],  #  Added
            'detector_data': detector_data,
            'metadata': {
                'sample_id': sample_id,  #  Fixed
                'event_type': event_type,
                'ra': params['ra'],
                'dec': params['dec'],
                'near_pole': abs(params['dec']) > np.pi/2 - 0.2
            }
        }

    def _generate_heavy_tailed_sample(self, sample_id: int, config: Dict) -> Dict:
        """
        Generate sample with parameters from heavy-tailed distributions.
        Tests model's ability to handle rare, extreme parameter values.
        """
        # Extract config
        edge_cases = config.get('edge_cases', {})
        stat_config = edge_cases.get('statistical_extremes', {})
        types = stat_config.get('types', {})
        heavy_config = types.get('heavy_tailed_regions', {})
        
        log_uniform_distance = heavy_config.get('log_uniform_distance', True)
        
        # Sample event
        event_type = self._sample_event_type()
        params = self._sample_parameters(event_type)
        
        # Override with heavy-tailed distributions
        if log_uniform_distance:
            from astropy.cosmology import Planck18
            z = np.random.uniform(0.01, 0.5)
            d_L = Planck18.luminosity_distance(z).to('Mpc').value
            d_C = d_L / (1 + z)
            
            params['redshift'] = float(z)
            params['luminosity_distance'] = float(d_L)
            params['comoving_distance'] = float(d_C)
            assert d_C <= d_L
            # Recompute target_snr to remain consistent with overridden distance
            try:
                self.parameter_sampler.recompute_target_snr_from_params(params)
            except Exception:
                pass
        
        # Heavy-tailed mass distribution (Student-t like)
        if event_type == 'BBH':
            # Sample from tails
            if np.random.random() < 0.5:
                # High mass tail
                params['mass_1'] = np.random.uniform(50, 100)
                params['mass_2'] = np.random.uniform(30, params['mass_1'])
            else:
                # Low mass tail (unusual for BBH)
                params['mass_1'] = np.random.uniform(3, 8)
                params['mass_2'] = np.random.uniform(2, params['mass_1'])
        
        # Generate sample
        # If distance was overridden, recompute target_snr to remain consistent
        try:
            self.parameter_sampler.recompute_target_snr_from_params(params)
        except Exception:
            pass

        return self._generate_sample_from_params(sample_id, params, 'heavy_tailed_regions')
    
    def _generate_uninformative_prior_sample(self, sample_id: int, config: Dict) -> Dict:
        """
        Generate sample with very broad parameter ranges (uninformative priors).
        Tests model's ability to infer parameters with minimal prior information.
        """
        # Extract config
        edge_cases = config.get('edge_cases', {})
        stat_config = edge_cases.get('statistical_extremes', {})
        types = stat_config.get('types', {})
        prior_config = types.get('uninformative_priors', {})

        broad_multiplier = prior_config.get('broad_prior_multiplier', 10.0)
        sample_intrinsic = prior_config.get('sample_intrinsic_params', False)
        mass_range = prior_config.get('mass_range', [1.0, 100.0])
        spin_range = prior_config.get('spin_range', [0.0, 0.99])
        distance_range = prior_config.get('distance_range', [10.0, 5000.0])
        angle_range = prior_config.get('angle_range', [0.0, 2 * np.pi])
        inclination_range = prior_config.get('inclination_range', [-1.0, 1.0])
        polarization_range = prior_config.get('polarization_range', [0.0, np.pi])
        phase_range = prior_config.get('phase_range', [0.0, 2 * np.pi])

        # Sample event type
        event_type = self._sample_event_type()

        if sample_intrinsic:
            # Sample all parameters from broad uniform distributions
            # Masses
            mass_1 = np.random.uniform(mass_range[0], mass_range[1])
            mass_2 = np.random.uniform(mass_range[0], mass_1)  # Ensure m1 >= m2

            # Spins
            a1 = np.random.uniform(spin_range[0], spin_range[1])
            a2 = np.random.uniform(spin_range[0], spin_range[1])

            # Spin orientations
            tilt1 = np.arccos(np.random.uniform(-1.0, 1.0))
            tilt2 = np.arccos(np.random.uniform(-1.0, 1.0))

            # Orbital parameters
            phi12 = np.random.uniform(0.0, 2 * np.pi)
            phi_jl = np.random.uniform(0.0, 2 * np.pi)

            # Build parameter dict manually with all required fields
            total_mass = mass_1 + mass_2
            chirp_mass = (mass_1 * mass_2)**(3/5) / total_mass**(1/5)
            effective_spin = compute_effective_spin(mass_1, mass_2, a1, a2, tilt1, tilt2)

            params = {
                'name': f'{event_type}_{int(mass_1)}_{int(mass_2)}_uninformative',
                'type': event_type,
                'mass_1': float(mass_1),
                'mass_2': float(mass_2),
                'total_mass': float(total_mass),
                'chirp_mass': float(chirp_mass),
                'mass_ratio': float(mass_2 / mass_1),
                'symmetric_mass_ratio': float((mass_1 * mass_2) / total_mass**2),
                'a_1': float(a1),
                'a_2': float(a2),
                'tilt_1': float(tilt1),
                'tilt_2': float(tilt2),
                'effective_spin': float(effective_spin),
                'phi_12': float(phi12),
                'phi_jl': float(phi_jl),
                'f_lower': 20.0,
                'f_ref': 20.0,
                'approximant': 'IMRPhenomD',  # Default for uninformative priors
                'lambda_1': 0.0,  # Tidal deformability (0 for BBH)
                'lambda_2': 0.0,
                'is_real_event': False,
                'edge_case': True,
                'edge_case_type': 'uninformative_priors'
            }
        else:
            # Use normal intrinsic parameters, only broaden extrinsic
            params = self._sample_parameters(event_type)

        # Always override extrinsic parameters with broad ranges
        params['luminosity_distance'] = np.random.uniform(
            distance_range[0],
            distance_range[1] * broad_multiplier
        )

        # Sky location
        params['ra'] = np.random.uniform(angle_range[0], angle_range[1])
        params['dec'] = np.arcsin(np.random.uniform(inclination_range[0], inclination_range[1]))
        params['theta_jn'] = np.arccos(np.random.uniform(-1.0, 1.0))  # Inclination angle

        # Polarization and phase
        params['psi'] = np.random.uniform(polarization_range[0], polarization_range[1])
        params['phase'] = np.random.uniform(phase_range[0], phase_range[1])

        # Time
        params['geocent_time'] = np.random.uniform(-0.1, 0.1)

        # Cosmology calculations
        z = calculate_redshift(params['luminosity_distance'])
        params['redshift'] = float(z) if z is not None else 0.0
        params['luminosity_distance'] = float(params['luminosity_distance'])
        # Exact: d_C = d_L / (1 + z)
        d_C = params['luminosity_distance'] / (1 + params['redshift']) if params['redshift'] > 0 else params['luminosity_distance']
        params['comoving_distance'] = float(d_C)

        # Compute target SNR based on the broad distance
        try:
            self.parameter_sampler.recompute_target_snr_from_params(params)
        except Exception:
            # Fallback: estimate SNR
            params['target_snr'] = self._estimate_snr_from_params(params)

        # Generate sample
        return self._generate_sample_from_params(sample_id, params, 'uninformative_priors')

    def _generate_subtle_ranking_overlap(self, sample_id: int, config: Dict) -> Dict:
        """
        Generate overlapping signals with very similar SNRs (hard to rank).
        Tests prioritization network's ability to distinguish close SNRs.
        """
        # Extract config
        edge_cases = config.get('edge_cases', {})
        overlap_config = edge_cases.get('overlapping_extremes', {})
        subtle_config = overlap_config.get('subtle_ranking', {})
        
        snr_diff_range = subtle_config.get('snr_difference_range', [0.5, 2.5])
        
        # Generate 2-3 signals with similar SNRs
        n_signals = np.random.choice([2, 3], p=[0.7, 0.3])
        
        # Base SNR
        base_snr = np.random.uniform(10, 20)
        
        parameters_list = []
        for i in range(n_signals):
            event_type = self._sample_event_type()
            params = self._sample_parameters(event_type)
            
            # Set similar SNRs
            snr_offset = np.random.uniform(snr_diff_range[0], snr_diff_range[1])
            if 'target_snr' not in params:
                self._set_target_snr(params, base_snr + (i - 1) * snr_offset, reason='subtle_ranking_base_snr')
            
            # Slightly offset times
            params['geocent_time'] = i * 0.5  # 0.5s apart
            
            parameters_list.append(params)
        
        # Generate combined sample
        return self.generate_custom_overlap(parameters_list, sample_id, is_edge_case=True)

    def _generate_pre_merger_sample(self, sample_id: int) -> Dict:
        """
        Generate pre-merger sample for early warning training.
        
        The data window ends BEFORE the merger happens, containing only the 
        inspiral phase. The model must learn to:
        - Detect that a signal is present (inspiral)
        - Predict that a merger is coming soon
        - Estimate time_to_merger
        - Infer parameters from incomplete data
        
        This is critical for early warning systems.
        """
        
        premerger_config = self.config.get('premerger_config', {})
    
        # Check if enabled
        if not premerger_config.get('enabled', True):
            sample = self._generate_single_sample(
                sample_id=sample_id,
                is_edge_case=False,
                add_glitches=False,
                preprocess=True
            )
            # Ensure priorities and normalization
            sample = self._ensure_sample_priorities(sample)
            return sample
        
        # Extract configuration parameters
        time_to_merger_range = premerger_config.get('time_to_merger_range', [0.5, 5.0])
        event_types = premerger_config.get('event_types', ['BBH', 'BNS', 'NSBH'])
        min_snr = premerger_config.get('min_snr', 8)
        
        # Sample event type and parameters (use configured global distribution but restricted to allowed list)
        event_type = self._sample_event_type_subset(event_types)
        params = self._sample_parameters(event_type)
        
        # Ensure detectable SNR: if sampler provided one, enforce minimum; otherwise set to min_snr
        if 'target_snr' in params:
            try:
                self._set_target_snr(params, max(float(params.get('target_snr', 0)), min_snr), reason='premerger_min_enforce')
            except Exception:
                self._set_target_snr(params, min_snr, reason='premerger_min_enforce_fail')
        else:
            self._set_target_snr(params, min_snr, reason='premerger_min_default')
        
        # Sample time to merger
        time_to_merger = np.random.uniform(time_to_merger_range[0], time_to_merger_range[1])
        
        # Shift merger time outside window
        params['geocent_time'] = self.duration / 2 + time_to_merger
        
        # Generate detector data
        detector_data = {}
        
        for detector_name in self.detectors:
            psd_dict = self.psds[detector_name]
            noise = self.noise_generator.generate_colored_noise(psd_dict)
            
            try:
                strain = self.waveform_generator.generate_waveform(params, detector_name)
                
                expected_length = int(self.duration * self.sample_rate)
                if len(strain) > expected_length:
                    strain = strain[:expected_length]
                elif len(strain) < expected_length:
                    strain = np.pad(strain, (0, expected_length - len(strain)), mode='constant')
                
                combined = noise + strain
                
                if self.preprocess_enabled:
                    combined = self.preprocessor.preprocess(combined, psd_dict)
                
                detector_data[detector_name] = {
                    'strain': combined.astype(np.float32),
                    'psd': psd_dict['psd'],
                    'frequencies': psd_dict['frequencies']
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to generate pre-merger waveform for {detector_name}: {e}")
                detector_data[detector_name] = {
                    'strain': noise.astype(np.float32),
                    'psd': psd_dict['psd'],
                    'frequencies': psd_dict['frequencies']
                }
        
        #  Calculate priority
        priority = self._estimate_snr_from_params(params)
        # Only set target_snr if not already present (preserve sampled values)
        if 'target_snr' not in params:
            self._set_target_snr(params, priority, reason='pre_merger_assign')
        
        # Construct sample
        sample = {
            'sample_id': f'pre_merger_{sample_id:06d}',  #  Changed from 'id' to 'sample_id'
            'type': event_type,
            'is_overlap': False,
            'n_signals': 1,  #  Added
            'is_edge_case': True,
            'edge_case_type': 'pre_merger_samples',
            'is_premerger': True,
            'parameters': params,  # Dict for single
            'priorities': [priority],  #  Added
            'detector_data': detector_data,
            'metadata': {
                'sample_id': f'pre_merger_{sample_id:06d}',
                'event_type': event_type,
                'time_to_merger': time_to_merger,
                'merger_in_window': False,
                'phase': 'inspiral_only',
                'window_end_to_merger_seconds': time_to_merger,
                'merger_time': params['geocent_time'],
                'window_duration': self.duration,
                'contains_merger': False,
                'is_complete_signal': False
            }
        }
        
        # Optional debug SNR logging: compare sampled target_snr vs pre-injection estimate and actual injected SNR
        try:
            debug_enabled = bool(self.config.get('debug_snr_diagnostic', False))
        except Exception:
            debug_enabled = False

        if debug_enabled and self._debug_snr_count < self._debug_snr_limit:
            try:
                ref_det = self.detectors[0]
                ref_psd = self.psds.get(ref_det)
                # For overlapping, target_snrs already computed earlier as 'target_snrs'
                pre_calc = []
                actuals = []
                for i, params in enumerate(signal_params_list):
                    try:
                        wf = self.waveform_generator.generate_waveform(params, ref_det)
                        pre = float(self.injector._compute_optimal_snr(wf, ref_psd)) if ref_psd is not None else float('nan')
                    except Exception as e:
                        pre = float('nan')
                    pre_calc.append(pre)

                # actual SNRs from metadata_list inside detector_data for ref_det
                det_meta_list = []
                det_entry = detector_data.get(ref_det, {})
                if det_entry:
                    det_meta_list = det_entry.get('metadata', [])

                # detector metadata list corresponds to each injected signal
                for m in det_meta_list:
                    try:
                        actuals.append(float(m.get('actual_snr', m.get('target_snr', float('nan')))))
                    except Exception:
                        actuals.append(float('nan'))

                self.logger.info(f"[SNR-DIAG] {sample['sample_id']}: sampled_targets={target_snrs} pre_estimates={pre_calc} actuals={actuals}")
            except Exception as e:
                self.logger.debug(f"SNR debug logging failed for {sample.get('sample_id')}: {e}")

            self._debug_snr_count += 1

        return sample

    def _generate_extreme_spin_sample(self, sample_id: int, config: Dict) -> Dict:
        """Generate sample with near-maximal spins."""
        spin_config = config.get('edge_cases', {}).get('physical_extremes', {}).get('types', {}).get('extreme_spins', {})
        
        params = self.parameter_sampler.sample_bbh_parameters('medium', False)
        
        # Near-maximal spins
        params['a1'] = float(np.random.uniform(0.9, 0.998))
        params['a2'] = float(np.random.uniform(0.9, 0.998))
        
        # Random alignment
        if np.random.random() < spin_config['aligned_fraction']:
            params['tilt1'] = 0.0
            params['tilt2'] = 0.0
        else:
            params['tilt1'] = float(np.random.uniform(0, np.pi))
            params['tilt2'] = float(np.random.uniform(0, np.pi))
        
        return self._generate_sample_from_params(sample_id, params, 'extreme_spins')


    def _generate_heavy_overlap(self, sample_id: int, config: Dict) -> Dict:
        """
        Generate sample with many (5-8) overlapping signals.
        Tests model's ability to detect and separate many simultaneous events.
        """
        # Extract config
        edge_cases = config.get('edge_cases', {})
        overlap_config = edge_cases.get('overlapping_extremes', {})
        heavy_config = overlap_config.get('heavy_overlaps', {})
        
        n_signals_range = heavy_config.get('n_signals_range', [5, 8])
        n_signals = np.random.randint(n_signals_range[0], n_signals_range[1] + 1)
        
        # Generate many signals
        parameters_list = []
        for i in range(n_signals):
            # Use configured EVENT_TYPE_DISTRIBUTION for sampling event types
            event_type = self._sample_event_type()
            params = self._sample_parameters(event_type)
            
            # Spread out in time
            params['geocent_time'] = np.random.uniform(-1, 1)
            
            parameters_list.append(params)
        
        return self.generate_custom_overlap(parameters_list, sample_id, is_edge_case=True)


    def _generate_eccentric_merger_sample(self, sample_id: int, config: Dict) -> Dict:
        """Generate sample with residual eccentricity."""
        ecc_config = config.get('edge_cases', {}).get('physical_extremes', {}).get('types', {}).get('eccentric_mergers', {})
        
        params = self.parameter_sampler.sample_bbh_parameters('medium', False)
        params['eccentricity'] = float(np.random.uniform(
            ecc_config['eccentricity_range'][0],
            ecc_config['eccentricity_range'][1]
        ))
        
        return self._generate_sample_from_params(sample_id, params, 'eccentric')


    def _generate_precessing_system_sample(self, sample_id: int, config: Dict) -> Dict:
        """Generate sample with random spin orientations (precession)."""
        params = self.parameter_sampler.sample_bbh_parameters('medium', False)
        
        # Random spin magnitudes
        params['a1'] = float(np.random.uniform(0.3, 0.9))
        params['a2'] = float(np.random.uniform(0.3, 0.9))
        
        # Random orientations (isotropic)
        params['tilt1'] = float(np.arccos(np.random.uniform(-1, 1)))
        params['tilt2'] = float(np.arccos(np.random.uniform(-1, 1)))
        params['phi12'] = float(np.random.uniform(0, 2*np.pi))
        params['phi_jl'] = float(np.random.uniform(0, 2*np.pi))
        
        return self._generate_sample_from_params(sample_id, params, 'precessing')


    def _generate_short_duration_high_mass_sample(self, sample_id: int, config: Dict) -> Dict:
        """Generate high-mass BBH with few inspiral cycles."""
        mass_config = config.get('edge_cases', {}).get('physical_extremes', {}).get('types', {}).get('short_duration_high_mass', {})
        
        params = self.parameter_sampler.sample_bbh_parameters('medium', False)
        params['mass_1'] = float(np.random.uniform(mass_config['mass_range'][0], mass_config['mass_range'][1]))
        params['mass_2'] = float(np.random.uniform(20, params['mass_1']))
        
        # Higher starting frequency → fewer cycles
        params['f_lower'] = float(np.random.uniform(40.0, 100.0))
        
        return self._generate_sample_from_params(sample_id, params, 'short_duration_high_mass')


    def _generate_low_snr_threshold_sample(self, sample_id: int, config: Dict) -> Dict:
        """Generate near-threshold low-SNR sample."""
        snr_config = config.get('edge_cases', {}).get('physical_extremes', {}).get('types', {}).get('low_snr_threshold', {})
        
        params = self.parameter_sampler.sample_bbh_parameters('low', False)
        if 'target_snr' not in params:
            self._set_target_snr(params, float(np.random.uniform(snr_config['snr_range'][0], snr_config['snr_range'][1])), reason='low_snr_threshold')
        
        return self._generate_sample_from_params(sample_id, params, 'low_snr_threshold')
    
        
    def _generate_sample_from_params(self, sample_id: int, params: Dict, edge_type: str) -> Dict:
        """
        Generate detector data from given parameters.
        Used by edge case generators that need specific parameter control.
        """
        
        detector_data = {}
        
        for detector_name in self.detectors:
            psd_dict = self.psds[detector_name]
            
            # Generate noise
            noise = self.noise_generator.generate_colored_noise(psd_dict)
            
            # Generate and inject signal
            signal = self.waveform_generator.generate_waveform(params, detector_name)
            
            # Combine
            combined = noise + signal
            
            # Preprocess
            if self.preprocess_enabled:
                combined = self.preprocessor.preprocess(combined, psd_dict)
            
            detector_data[detector_name] = {
                'strain': combined.astype(np.float32),
                'psd': psd_dict['psd'],
                'frequencies': psd_dict['frequencies']
            }
        
        #sample target_Snr randomly for extreme cases to ensure coverage in all reagimes
        if 'target_snr' not in params:
            snr_regime = self._sample_snr_regime()
            params['target_snr'] = self._sample_target_snr(snr_regime)
        
        # Set priority to target_snr
        priority = params['target_snr']

        # Attach network_snr
        attach_network_snr_safe(params)

        #  Create sample with all required fields
        sample = {
            'sample_id': f'{edge_type}_{sample_id:06d}',
            'type': params.get('type', 'BBH'),
            'is_overlap': False,
            'n_signals': 1,
            'is_edge_case': True,
            'edge_case_type': edge_type,
            'parameters': params,  # Dict for single
            'priorities': [priority],  # Must be list
            'detector_data': detector_data,
            'metadata': {
                'sample_id': sample_id,
                'edge_case_type': edge_type,
                'signal_parameters': params
            }
        }
        
        # Optional debug SNR logging for single events
        try:
            debug_enabled = bool(self.config.get('debug_snr_diagnostic', False))
        except Exception:
            debug_enabled = False

        if debug_enabled and self._debug_snr_count < self._debug_snr_limit:
            try:
                ref_det = self.detectors[0]
                ref_psd = self.psds.get(ref_det)
                if params:
                    # Pre-injection SNR estimate from waveform
                    try:
                        wf = self.waveform_generator.generate_waveform(params, ref_det)
                        pre = float(self.injector._compute_optimal_snr(wf, ref_psd)) if ref_psd is not None else float('nan')
                    except Exception:
                        pre = float('nan')

                    # actual SNR reported by injector for reference detector
                    meta = detector_data.get(ref_det, {}).get('metadata', {})
                    actual = meta.get('actual_snr', meta.get('target_snr', float('nan')))

                    # self.logger.info(f"[SNR-DIAG] {sample['sample_id']}: target={params.get('target_snr')} pre_estimate={pre} actual={actual}")
                else:
                    self.logger.info(f"[SNR-DIAG] {sample['sample_id']}: noise-only")
            except Exception as e:
                self.logger.debug(f"SNR debug logging failed for {sample.get('sample_id')}: {e}")

            self._debug_snr_count += 1

        return sample

    # ============================================================================
    # 2. OBSERVATIONAL EXTREMES
    # ============================================================================

    def _generate_strong_glitch_sample(self, sample_id: int, config: Dict) -> Dict:
        """Generate sample with strong glitches."""
        
        #  Safe config navigation with defaults
        edge_cases = config.get('edge_cases', {})
        obs_extremes = edge_cases.get('observational_extremes', {})
        types_config = obs_extremes.get('types', {})
        glitch_config = types_config.get('strong_glitches', {
            'glitch_prob': 0.8,
            'glitch_types': ['blip', 'whistle', 'scattered_light']
        })
        
        # Extract parameters with defaults
        glitch_prob = glitch_config.get('glitch_prob', 0.8)
        glitch_types = glitch_config.get('glitch_types', ['blip', 'whistle', 'scattered_light'])
        
        # Generate base sample
        sample = self._generate_single_sample(
            sample_id=sample_id,
            is_edge_case=True,
            add_glitches=False,  # We'll add manually
            preprocess=True
        )
        

        
        # Add strong glitch if probability check passes
        if np.random.random() < glitch_prob:
            glitch_type = np.random.choice(glitch_types)

            # Add glitch to detector data
            for det_name in sample.get('detector_data', {}).keys():
                det_data = sample['detector_data'][det_name]
                if 'strain' in det_data:
                    original_strain = det_data['strain']
                    glitched_strain = self._inject_glitch(original_strain, glitch_type)
                    det_data['strain'] = glitched_strain
                    # Mark as modified
                    det_data['has_glitch'] = True

            sample['has_glitch'] = True
            sample['glitch_type'] = glitch_type
        
        sample['edge_case_type'] = 'strong_glitches'
        
        # Ensure priorities exist and are normalized
        sample = self._ensure_sample_priorities(sample)

        return sample

    def _inject_glitch(self, strain, glitch_type):
        """Inject specific glitch type into strain data."""
        N = len(strain)
        t = np.arange(N) / self.sample_rate
        # Estimate local noise floor from the provided strain (use as scale)
        noise_std = float(np.std(strain)) if np.std(strain) > 0 else 1e-12
        
        if glitch_type == 'blip':
            # Short Gaussian transient scaled to local noise
            t_glitch = np.random.uniform(0.5, 3.5)
            sigma = np.random.uniform(0.01, 0.05)
            # amplitude in noise-relative units
            amplitude_scale = np.random.uniform(5, 20)
            amplitude = amplitude_scale * noise_std
            glitch = amplitude * np.exp(-((t - t_glitch)**2) / (2 * sigma**2))
        
        elif glitch_type == 'whistle':
            # Chirping noise artifact scaled to local noise
            f0 = np.random.uniform(100, 500)
            f1 = np.random.uniform(f0, 1000)
            t_start = np.random.uniform(0.5, 2.0)
            duration = np.random.uniform(0.1, 0.5)
            mask = (t >= t_start) & (t < t_start + duration)
            phase = 2 * np.pi * (f0 * (t - t_start) + 0.5 * (f1 - f0) / duration * (t - t_start)**2)
            glitch = np.zeros_like(strain)
            # per-sample amplitude scale relative to noise
            whistle_scale = np.random.uniform(2, 10)
            glitch[mask] = (whistle_scale * noise_std) * np.sin(phase[mask])
        
        elif glitch_type == 'scattered_light':
            # Low-frequency modulation scaled to local noise
            f_scatter = np.random.uniform(10, 60)
            scatter_scale = np.random.uniform(1, 5)
            amplitude = scatter_scale * noise_std
            glitch = amplitude * np.sin(2 * np.pi * f_scatter * t)
        
        return strain + glitch


    def _generate_detector_dropout_sample(self, sample_id: int, config: Dict) -> Dict:
        """
        Generate sample with detector dropout.
        
        Simulates scenarios where one or more detectors are offline or have
        data quality issues. This tests the model's ability to work with
        incomplete detector networks.
        
        Args:
            sample_id: Unique sample identifier
            config: Full configuration dict
        
        Returns:
            Sample dict with some detectors zeroed out
        """
        
        # Safe config navigation
        edge_cases = config.get('edge_cases', {})
        obs_extremes = edge_cases.get('observational_extremes', {})
        types = obs_extremes.get('types', {})
        dropout_config = types.get('detector_dropout', {})
        
        # Extract parameters with defaults
        dropout_prob = dropout_config.get('dropout_prob', 0.3)
        min_active = dropout_config.get('min_active_detectors', 1)
        
        # Generate base sample with all detectors
        sample = self._generate_single_sample(
            sample_id=sample_id,
            is_edge_case=True,
            add_glitches=False,
            preprocess=True
        )
        
        # Determine if we apply dropout
        if np.random.random() < dropout_prob:
            # Calculate how many detectors to drop
            n_detectors = len(self.detectors)
            
            if n_detectors > min_active:
                # Drop 1 to (n_detectors - min_active) detectors
                max_dropout = n_detectors - min_active
                n_to_drop = np.random.randint(1, max_dropout + 1)
                
                # Randomly select which detectors to keep active
                n_active = n_detectors - n_to_drop
                active_detectors = list(np.random.choice(
                    self.detectors, 
                    size=n_active, 
                    replace=False
                ))
                
                # Zero out dropped detectors
                for det_name in sample['detector_data'].keys():
                    if det_name not in active_detectors:
                        # Set strain to zeros
                        strain_shape = sample['detector_data'][det_name]['strain'].shape
                        sample['detector_data'][det_name]['strain'] = np.zeros(
                            strain_shape, 
                            dtype=np.float32
                        )
                
                # Update metadata
                sample['active_detectors'] = active_detectors
                sample['dropped_detectors'] = [d for d in self.detectors if d not in active_detectors]
                sample['n_active_detectors'] = n_active
                sample['n_dropped_detectors'] = n_to_drop
                sample['has_dropout'] = True
                
                self.logger.debug(
                    f"Sample {sample_id}: Detector dropout - "
                    f"{n_to_drop} dropped, {n_active} active: {active_detectors}"
                )
            else:
                # Can't drop any without violating min_active constraint
                sample['active_detectors'] = list(self.detectors)
                sample['dropped_detectors'] = []
                sample['n_active_detectors'] = n_detectors
                sample['n_dropped_detectors'] = 0
                sample['has_dropout'] = False
        else:
            # No dropout applied
            sample['active_detectors'] = list(self.detectors)
            sample['dropped_detectors'] = []
            sample['n_active_detectors'] = len(self.detectors)
            sample['n_dropped_detectors'] = 0
            sample['has_dropout'] = False
        
        # Mark as edge case
        sample['edge_case_type'] = 'detector_dropout'
        
        # Update metadata
        if 'metadata' not in sample:
            sample['metadata'] = {}
        
        sample['metadata'].update({
            'edge_case_type': 'detector_dropout',
            'active_detectors': sample['active_detectors'],
            'dropped_detectors': sample['dropped_detectors'],
            'n_active': sample['n_active_detectors']
        })
        
        # Ensure priorities exist and are normalized
        sample = self._ensure_sample_priorities(sample)

        return sample



    # ============================================================================
    # 3. STATISTICAL EXTREMES
    # ============================================================================

    def _generate_multimodal_posterior_sample(self, sample_id: int, config: Dict) -> Dict:
        """
        Generate sample with multimodal posterior distribution.
        
        Creates scenarios with parameter degeneracies that lead to multiple 
        posterior modes during inference. This tests the model's ability to 
        handle ambiguous parameter combinations that produce similar signals.
        
        Scientific basis:
        - Inclination degeneracy: θ_JN ↔ π - θ_JN with distance compensation
        - Spin-inclination degeneracy: aligned vs anti-aligned with θ flip
        - Mass-distance degeneracy: chirp mass determines SNR scaling
        
        Args:
            sample_id: Unique sample identifier
            config: Full configuration dict
        
        Returns:
            Sample dict with parameters that create posterior multimodality
        """
        
        # Safe config navigation
        edge_cases = config.get('edge_cases', {})
        stat_config = edge_cases.get('statistical_extremes', {})
        types = stat_config.get('types', {})
        multimodal_config = types.get('multimodal_posteriors', {})
        
        # Configuration
        degeneracy_type = multimodal_config.get('degeneracy_type', 'inclination')
        
        # Generate base sample
        base_sample = self._generate_single_sample(
            sample_id=sample_id,
            is_edge_case=True,
            add_glitches=False,
            preprocess=True
        )
        
        if base_sample is None:
            self.logger.warning("Failed to generate multimodal posterior sample")
            return None
        
        # Handle noise samples (no signal to create degeneracy with)
        if base_sample.get('type') == 'noise' or not base_sample.get('parameters'):
            base_sample['edge_case_type'] = 'multimodal_posteriors'
            base_sample['priorities'] = base_sample.get('priorities', [10.0])
            base_sample['n_signals'] = 0
            self.logger.debug(f"Sample {sample_id}: Noise sample, skipping multimodal posterior generation")
            return base_sample
        
        # Extract parameters - HANDLE LIST FORMAT
        params_list = base_sample.get('parameters', [])
        
        # Ensure params_list is actually a list
        if not isinstance(params_list, list):
            params_list = [params_list] if params_list else []
        
        # Skip if no valid parameters
        if not params_list or not params_list[0]:
            base_sample['edge_case_type'] = 'multimodal_posteriors'
            base_sample['priorities'] = base_sample.get('priorities', [10.0])
            base_sample['n_signals'] = len(params_list)
            return base_sample
        
        # Get first (and typically only) param dict from list
        params = params_list[0]

        # Ensure luminosity_distance exists
        if 'luminosity_distance' not in params:
            params['luminosity_distance'] = 100.0  # Default distance

        # Apply specific degeneracy based on type
        if degeneracy_type == 'inclination':
            # Inclination degeneracy: face-on ↔ face-off
            params['theta_jn'] = np.random.uniform(0.0, 0.3)  # Nearly face-on
            
            # Create degenerate parameter set
            degenerate_params = params.copy()
            degenerate_params['theta_jn'] = np.pi - params['theta_jn']  # Face-off
            
            # Distance must scale to maintain SNR
            cos_theta_original = np.cos(params['theta_jn'])
            cos_theta_degenerate = np.cos(degenerate_params['theta_jn'])
            
            amplitude_ratio = (1 + cos_theta_original**2) / (1 + cos_theta_degenerate**2)
            degenerate_params['luminosity_distance'] = params['luminosity_distance'] * amplitude_ratio
            # Keep degenerate params' target_snr consistent with the adjusted distance
            try:
                self.parameter_sampler.recompute_target_snr_from_params(degenerate_params)
            except Exception:
                pass
            
        elif degeneracy_type == 'spin_inclination':
            # Aligned spin with inclination flip
            params['a_1'] = np.random.uniform(0.7, 0.95)  # High spin
            params['tilt_1'] = np.random.uniform(0.0, 0.2)  # Nearly aligned
            params['theta_jn'] = np.random.uniform(0.0, 0.4)
            
            degenerate_params = params.copy()
            degenerate_params['tilt_1'] = np.pi - params['tilt_1']  # Anti-aligned
            degenerate_params['theta_jn'] = np.pi - params['theta_jn']
            
        elif degeneracy_type == 'mass_distance':
            # Chirp mass-distance degeneracy
            # Only set target_snr if sampler hasn't provided one
            if 'target_snr' not in params:
                self._set_target_snr(params, np.random.uniform(8, 12), reason='mass_distance_degeneracy')
            
            degenerate_params = params.copy()
            mass_scale = np.random.uniform(0.9, 1.1)
            degenerate_params['mass_1'] = params['mass_1'] * mass_scale
            degenerate_params['mass_2'] = params['mass_2'] * mass_scale
            
            # Recompute chirp mass
            M1 = degenerate_params['mass_1']
            M2 = degenerate_params['mass_2']
            degenerate_params['chirp_mass'] = (M1 * M2)**(3/5) / (M1 + M2)**(1/5)
            
            # Adjust distance - compute original chirp mass if not present
            if 'chirp_mass' in params:
                original_Mc = params['chirp_mass']
            else:
                m1, m2 = params['mass_1'], params['mass_2']
                original_Mc = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
            
            new_Mc = degenerate_params['chirp_mass']
            degenerate_params['luminosity_distance'] = params['luminosity_distance'] * (new_Mc / original_Mc)**(5/6)
            # Keep the degenerate parameter's target_snr consistent with its new distance
            try:
                self.parameter_sampler.recompute_target_snr_from_params(degenerate_params)
            except Exception:
                pass
            
        else:
            # Default: simple inclination flip
            degenerate_params = params.copy()
            if 'theta_jn' in degenerate_params:
                degenerate_params['theta_jn'] = np.pi - degenerate_params['theta_jn']
        
        # IMPORTANT: Update the params_list with modified params
        params_list[0] = params
        base_sample['parameters'] = params_list
        
        # Attach network_snr to all params (including modified ones)
        attach_network_snr_safe(params_list)
        
        # Update sample with multimodal metadata
        base_sample['edge_case_type'] = 'multimodal_posteriors'
        
        if 'metadata' not in base_sample:
            base_sample['metadata'] = {}
        
        base_sample['metadata'].update({
            'multimodal': True,
            'degeneracy_type': degeneracy_type,
            'primary_parameters': params,
            'degenerate_parameters': degenerate_params,
            'n_modes': 2,
            'description': f"{degeneracy_type} degeneracy creates bimodal posterior"
        })
        
        self.logger.debug(
            f"Sample {sample_id}: Multimodal posterior with {degeneracy_type} degeneracy"
        )
        
        # Ensure priorities exist
        if 'priorities' not in base_sample or not base_sample.get('priorities'):
            priorities = []
            parameters = base_sample.get('parameters', [])
            
            if not isinstance(parameters, list):
                parameters = [parameters]
                base_sample['parameters'] = parameters
            
            for p in parameters:
                if isinstance(p, dict):
                    priority = self._estimate_snr_from_params(p)
                    # Respect any existing sampled target_snr (don't overwrite)
                    if 'target_snr' not in p:
                        self._set_target_snr(p, priority, reason='multimodal_posteriors')
                    priorities.append(priority)
            
            base_sample['priorities'] = priorities if priorities else [15.0]
            base_sample['n_signals'] = len(priorities)
        
        return base_sample

    # ============================================================================
    # 4. OVERLAPPING EXTREMES
    # ============================================================================

    def _generate_partial_overlap_sample(self, sample_id: int, config: Dict) -> Dict:
        """Generate sample with partial temporal overlap (50-200ms)."""
        
        edge_cases = config.get('edge_cases', {})
        overlap_config = edge_cases.get('overlapping_extremes', {})
        types = overlap_config.get('types', {})
        partial_config = types.get('partial_overlaps', {})
        
        n_signals = partial_config.get('n_signals', 2)
        overlap_time_ms = partial_config.get('overlap_time_ms', [50, 200])
        
        time_offset = np.random.uniform(overlap_time_ms[0], overlap_time_ms[1]) / 1000.0
        
        parameters_list = []
        for i in range(n_signals):
            event_type = self._sample_event_type()
            params = self._sample_parameters(event_type)
            
            if i > 0:
                params['geocent_time'] = time_offset * i
            else:
                params['geocent_time'] = 0.0
            
            parameters_list.append(params)
        
        detector_data = {}
        
        for detector_name in self.detectors:
            psd_dict = self.psds[detector_name]
            noise = self.noise_generator.generate_colored_noise(psd_dict)
            combined = noise.copy()
            
            metadata_list = []
            for params in parameters_list:
                signal = self.waveform_generator.generate_waveform(params, detector_name)
                
                shift_samples = int(params['geocent_time'] * self.sample_rate)
                if shift_samples > 0:
                    signal = np.roll(signal, shift_samples)
                    signal[:shift_samples] = 0
                
                combined += signal
                metadata_list.append({'time_offset': params['geocent_time']})
            
            if self.preprocess_enabled:
                combined = self.preprocessor.preprocess(combined, psd_dict)
            
            detector_data[detector_name] = {
                'strain': combined.astype(np.float32),
                'psd': psd_dict['psd'],
                'frequencies': psd_dict['frequencies'],
                'metadata': metadata_list
            }
        
        #  Calculate priorities
        priorities = []
        for params in parameters_list:
            priority = self._estimate_snr_from_params(params)
            # Respect any existing sampled target_snr (don't overwrite)
            if 'target_snr' not in params:
                self._set_target_snr(params, priority, reason='partial_overlap')
            priorities.append(priority)
        
        return {
            'sample_id': f'partial_overlap_{sample_id:06d}',  #  Fixed
            'type': 'overlap',
            'is_overlap': n_signals > 1,
            'n_signals': n_signals,  #  Added
            'is_edge_case': True,
            'edge_case_type': 'partial_overlaps',
            'parameters': parameters_list,  #  Already list
            'priorities': priorities,  #  Added
            'detector_data': detector_data,
            'metadata': {
                'sample_id': sample_id,  #  Fixed
                'overlap_time_ms': time_offset * 1000,
                'n_signals': n_signals,
                'edge_case_type': 'partial_overlaps',
                'temporal_separation': 'partial'
            }
        }


    # ============================================================================
    # HELPER METHOD
    # ============================================================================

    def _save_batch(self, batch_id: int, samples: List[Dict]):
        """Save batch to disk in specified format"""
        
        metadata = {
            'batch_id': batch_id,
            'n_samples': len(samples),
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'detectors': self.detectors
        }
        
        # Integrity diagnostic: detect any parameters with clipped/high target_snr
        clipped_entries = []
        try:
            import traceback
            for si, sample in enumerate(samples):
                params_list = sample.get('parameters', [])
                for pi, params in enumerate(params_list):
                    if isinstance(params, dict) and 'target_snr' in params:
                        try:
                            val = float(params.get('target_snr', 0.0))
                        except Exception:
                            val = 0.0
                        if val >= 80.0:
                            entry = {
                                'sample_index': si,
                                'sample_id': sample.get('sample_id'),
                                'param_index': pi,
                                'target_snr': val
                            }
                            clipped_entries.append(entry)
                            # Log a short stack for diagnostics (where save was triggered)
                            #self.logger.warning(f"[SNR-PRE-SAVE] Detected high target_snr={val} in batch {batch_id} " f"sample_index={si} param_index={pi} sample_id={sample.get('sample_id')}")
                            # stack = ''.join(traceback.format_stack(limit=6))
                            # self.logger.warning(stack)
        except Exception:
            # Ignore diagnostics failures — do not block saving
            clipped_entries = clipped_entries

        # Attach simple diagnostic summary to metadata
        if clipped_entries:
            metadata['clipped_target_snr_count'] = len(clipped_entries)
            # Keep a small sample of clipped entries
            metadata['clipped_entries'] = clipped_entries[:50]

        if self.output_format == 'hdf5':
            self.writer.save_batch_hdf5(batch_id, samples, metadata)
        elif self.output_format == 'pkl':
            self.writer.save_batch_pkl(batch_id, samples, metadata, compress=False)
        elif self.output_format == 'pkl_compressed':
            self.writer.save_batch_pkl(batch_id, samples, metadata, compress=True)
        elif self.output_format == 'both':
            self.writer.save_batch_hdf5(batch_id, samples, metadata)
            self.writer.save_batch_pkl(batch_id, samples, metadata, compress=True)



    def _estimate_snr_from_params(self, params: Dict) -> float:
        """
        Estimate expected SNR from signal parameters.
        
        Uses simplified chirp mass scaling:
            SNR ∝ M_chirp^(5/6) / D
        
        Where:
            M_chirp = chirp mass in solar masses
            D = luminosity distance in Mpc
        
        Args:
            params: Signal parameters dict with mass_1, mass_2, luminosity_distance
        
        Returns:
            Estimated SNR (float)
        
        Raises:
            ValueError: If required parameters missing or invalid
        
        Examples:
            >>> params = {'mass_1': 30, 'mass_2': 30, 'luminosity_distance': 400}
            >>> snr = self._estimate_snr_from_params(params)
            >>> print(f"Expected SNR: {snr:.1f}")
            Expected SNR: 15.0
        """
        try:
            # Extract parameters
            m1 = params.get('mass_1')
            m2 = params.get('mass_2')
            distance = params.get('luminosity_distance')
            
            # Validate required parameters
            if m1 is None or m2 is None:
                raise ValueError("Missing mass_1 or mass_2")
            if distance is None or distance <= 0:
                raise ValueError(f"Invalid luminosity_distance: {distance}")
            
            # Ensure masses are physical
            if m1 <= 0 or m2 <= 0:
                raise ValueError(f"Non-physical masses: m1={m1}, m2={m2}")
            
            # Calculate chirp mass
            # M_chirp = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
            M_total = m1 + m2
            M_chirp = (m1 * m2)**(3/5) / M_total**(1/5)
            
            #  SNR scaling formula
            # Reference: M_chirp=30 Msun at D=400 Mpc → SNR ≈ 75
            # SNR ∝ M_chirp^(5/6) / D_L
            reference_snr = 75
            reference_mass = 30.0  # Solar masses
            reference_distance = 400.0  # Mpc
            
            snr_estimate = reference_snr * (M_chirp / reference_mass)**(5/6) * (reference_distance / distance)
            
            #  Clamp to reasonable range
            # Minimum SNR: 5 (detection threshold)
            # Maximum SNR: 80 (very loud event for training data)
            unclipped = float(snr_estimate)
            clipped = float(np.clip(unclipped, 5.0, 80.0))

            # # Diagnostic: log when unclipped value exceeds the upper cap (likely cause of clipping)
            # try:
            #     if clipped >= 79.999 and unclipped > 80.0:
            #         import traceback
            #         short_params = {k: params.get(k) for k in ('mass_1', 'mass_2', 'luminosity_distance')}
            #         self.logger.warning(f"[SNR-CLIP] _estimate_snr_from_params clipped {unclipped:.2f} -> {clipped:.2f}; params={short_params}")
            #         stack = ''.join(traceback.format_stack(limit=6))
            #         self.logger.warning(stack)
            # except Exception:
            #     pass

            return clipped
        
        except (TypeError, ValueError, ZeroDivisionError) as e:
            # Log error and return default
            self.logger.debug(f"SNR estimation failed: {e}")
            return 15.0  # Default fallback
    
    
    def _ensure_sample_priorities(self, sample: Dict) -> Dict:
        """
        Ensure that a sample has 'priorities' and 'n_signals' populated.
        
        This normalizes parameters to a list, reads existing 'target_snr' from params
        (or computes if missing), and sets sample['priorities'] and sample['n_signals'].
        
        IMPORTANT: Does NOT overwrite existing target_snr values!
        """
        if sample is None:
            return sample

        # Normalize parameters to a list so n_signals reflects real number of signals
        parameters = sample.get('parameters', [])
        if parameters is None:
            parameters = []
        if not isinstance(parameters, list):
            parameters = [parameters]
        
        # Ensure sample stores normalized parameters
        sample['parameters'] = parameters

        if 'priorities' not in sample or not sample.get('priorities'):
            priorities = []

            for params in parameters:
                if not isinstance(params, dict):
                    continue
                
                try:
                    #  USE EXISTING target_snr if available (don't recompute!)
                    priority = params.get('target_snr')
                    
                    if priority is None:
                        # Only compute if target_snr doesn't exist
                        priority = float(self._estimate_snr_from_params(params))
                        if 'target_snr' not in params:
                            self._set_target_snr(params, priority, reason='_ensure_sample_priorities_compute')  # Set it ONLY if missing
                    else:
                        # Already exists, use it as-is
                        priority = float(priority)
                        
                except Exception as e:
                    self.logger.debug(f"Priority computation failed: {e}")
                    priority = params.get('target_snr', 15.0)
                    if 'target_snr' not in params:
                        self._set_target_snr(params, priority, reason='_ensure_sample_priorities_exception')
                
                priorities.append(priority)

            # If we couldn't compute priorities (no params), keep a sensible default list
            sample['priorities'] = priorities if priorities else [15.0]

        # Derive n_signals from the normalized parameters list (true signal count)
        sample['n_signals'] = len(parameters) if parameters else 1

        return sample

    def _set_target_snr(self, params: Dict, value, reason: str = None):
        """Helper to assign target_snr and log a stack trace when a clipped/very large value is set.

        This centralizes assignments so we can diagnose where clipped 80.0 values originate.
        """
        try:
            val = float(value)
            # Clip to maximum allowed SNR
            val = min(val, 80.0)
        except Exception:
            val = value

        # Assign into params dict
        if isinstance(params, dict):
            params['target_snr'] = val

        # If value is effectively clipped/high, log stack for diagnostics
        # try:
        #     if isinstance(val, (int, float)) and val >= 80.0:
        #         import traceback
        #         self.logger.warning(f"[SNR-TRACE] Assigned clipped target_snr={val} reason={reason}")
        #         stack = ''.join(traceback.format_stack(limit=6))
        #         self.logger.warning(stack)
        # except Exception:
        #     # Never fail generation due to diagnostics
        #     pass

        return val

    
    def _generate_single_sample(self, sample_id: int, is_edge_case: bool,
                            add_glitches: bool, preprocess: bool,
                            snr_regime: Optional[str] = None,
                            forced_event_type: Optional[str] = None) -> Dict:
            """Generate single non-overlapping sample

            Optional arguments allow callers (quota mode) to force the SNR regime
            and/or the event type. If not provided, the method falls back to the
            usual sampler-driven behavior.
            """
            # Prefer caller-provided regime/type when present (quota-mode integration)
            if snr_regime is None:
                snr_regime = self._sample_snr_regime()

            if forced_event_type is not None:
                event_type = forced_event_type
            else:
                try:
                    event_type = self.parameter_sampler.event_type_given_snr(snr_regime)
                except Exception:
                    # Fallback to prior sampling if the sampler doesn't provide inversion
                    event_type = self._sample_event_type()

            # Generate parameters based on event type
            if event_type == 'BBH':
                params = self.parameter_sampler.sample_bbh_parameters(snr_regime, is_edge_case)
            elif event_type == 'BNS':
                params = self.parameter_sampler.sample_bns_parameters(snr_regime, is_edge_case)
            elif event_type == 'NSBH':
                params = self.parameter_sampler.sample_nsbh_parameters(snr_regime, is_edge_case)
            elif event_type == 'noise':
                params = {'type': 'noise', 'target_snr': 0.0, 'network_snr': 0.0}
            else:
                params = None

            if params is None:
                self.logger.error(f"Failed to sample parameters for event_type: {event_type}")
                return None

            attach_network_snr_safe(params)
        
            if params is None:
                print(f"ERROR: {event_type} parameter sampling returned None!")
                return None

            if params['type'] != 'noise':
                target_snr = self._sample_target_snr(snr_regime)
                params['target_snr'] = target_snr
            else:
                target_snr = 0.0

            if params is None:
                print("ERROR: params became None!")
                return None

            # The actual SNR will be computed during injection
            # It depends on: distance (already sampled), mass, and target SNR
            # NOT forced by distance selection

            # Generate data for each detector
            detector_data = {}
            for detector_name in self.detectors:
                psd_dict = self.psds[detector_name]

                # Generate noise
                noise = self.noise_generator.generate_colored_noise(psd_dict)

                # Add glitches if requested
                if add_glitches:
                    noise = self.noise_generator.add_glitches(noise, glitch_prob=0.1)

                # Inject signal if not noise
                if params['type'] == 'noise':
                    injected = noise
                    metadata = {}
                else:
                    injected, metadata = self.injector.inject_signal(
                        noise, params, detector_name, psd_dict
                    )

                # Preprocess if requested
                # OPTIMIZATION: Skip preprocessing for overlaps if configured
                skip_overlap_preprocessing = bool(self.config.get('skip_overlap_preprocessing', False))
                if preprocess and not skip_overlap_preprocessing:
                    injected = self.preprocessor.preprocess(injected, psd_dict)

                detector_data[detector_name] = {
                    'strain': injected,
                    'metadata': metadata
                }

            # Construct sample
            sample = {
                'sample_id': sample_id,
                'sample_type': 'single',
                'is_overlap': False,  # ✅ Explicitly mark as non-overlap
                'is_edge_case': is_edge_case,
                'detector_data': detector_data,
                'parameters': params,
                'metadata': {
                    'snr_regime': snr_regime,
                    'event_type': event_type,
                'target_snr': params.get('target_snr', 15.0),
            }
            }

            return sample

    def _sample_target_snr(self, snr_regime: str = None) -> float:
        """
        Sample target SNR from specified regime or from distribution.
        
        Args:
            snr_regime: Specific regime ('low', 'medium', 'high') or None to sample from distribution
        
        Returns:
            Target SNR value
        """
        # New: accept optional event_type conditioning via self.conditional_snr
        # Signature backward compatible: snr_regime may be provided by caller.
        def _draw_from_regime(regime):
            snr_min, snr_max = self.snr_ranges[regime]
            return float(np.random.uniform(snr_min, snr_max))

        if snr_regime is not None:
            target_snr = _draw_from_regime(snr_regime)
            # Track statistics
            self.stats['snr_regimes'][snr_regime] = self.stats['snr_regimes'].get(snr_regime, 0) + 1
            return target_snr

        # If caller didn't ask for a specific regime, sample optionally conditioned on event_type
        # (caller can pass event_type by setting attribute self._sampling_event_type before calling
        # or by using the new helper sample_target_snr_for_event).
        event_type = getattr(self, '_sampling_event_type', None)
        if event_type and hasattr(self, 'conditional_snr') and event_type in self.conditional_snr:
            regimes = list(self.conditional_snr[event_type].keys())
            probs = [self.conditional_snr[event_type][r] for r in regimes]
            # numerical stability
            s = sum(probs)
            if s <= 0:
                regimes = list(self.snr_distribution.keys())
                probs = list(self.snr_distribution.values())
            else:
                probs = [p / s for p in probs]
            regime = np.random.choice(regimes, p=probs)
            target = _draw_from_regime(regime)
            self.stats['snr_regimes'][regime] = self.stats['snr_regimes'].get(regime, 0) + 1
            return float(target)

        # Fallback to global sampling
        snr_regime = self._sample_snr_regime()
        snr_min, snr_max = self.snr_ranges[snr_regime]
        target_snr = float(np.random.uniform(snr_min, snr_max))
        self.stats['snr_regimes'][snr_regime] = self.stats['snr_regimes'].get(snr_regime, 0) + 1
        return target_snr
    
    def _generate_overlapping_sample(self, sample_id: int, is_edge_case: bool,
                                        add_glitches: bool, preprocess: bool,
                                        forced_signals: Optional[List[Dict]] = None) -> Dict:
            """Generate sample with overlapping signals"""

            # OPTIMIZATION: Reduce overlap complexity if configured
            fast_overlap_mode = bool(self.config.get('fast_overlap_mode', False))
            if fast_overlap_mode:
                n_signals = random.choice([2, 3])  # Limit to 2-3 signals max
            else:
                n_signals = sample_overlap_size()

            signal_params_list = []
            target_snrs = []

            # Generate parameters for each signal
            for i in range(n_signals):
                # If caller provided forced_signals info, use it for this index
                if forced_signals and i < len(forced_signals):
                    info = forced_signals[i]
                    snr_regime = info.get('snr_regime')
                    event_type = info.get('event_type')
                else:
                    # Default behavior: sample regime then event type via sampler
                    snr_regime = self._sample_snr_regime()
                    try:
                        event_type = self.parameter_sampler.event_type_given_snr(snr_regime)
                    except Exception:
                        event_type = self._sample_event_type()

                if event_type == 'BBH':
                    params = self.parameter_sampler.sample_bbh_parameters(snr_regime, is_edge_case)
                elif event_type == 'BNS':
                    params = self.parameter_sampler.sample_bns_parameters(snr_regime, is_edge_case)
                else:
                    params = self.parameter_sampler.sample_nsbh_parameters(snr_regime, is_edge_case)

                params['time_offset'] = np.random.uniform(-0.25, 0.25) if i > 0 else 0.0

                # Sample target SNR from the regime to match distribution
                target_snr = self._sample_target_snr(snr_regime)
                params['target_snr'] = target_snr
                target_snrs.append(float(target_snr))

                signal_params_list.append(params)

            #  FIX #1: SORT TARGETS BY SNR (highest SNR first) for priority learning
            # This ensures the priority network learns to rank signals correctly
            sorted_indices = np.argsort(target_snrs)[::-1]  # Descending order by SNR
            signal_params_list = [signal_params_list[i] for i in sorted_indices]
            target_snrs = [target_snrs[i] for i in sorted_indices]

            # Assign priorities proportional to SNR for stronger correlation
            max_snr = max(target_snrs) if target_snrs else 1.0
            priorities = [snr / max_snr for snr in target_snrs]
            for params, priority in zip(signal_params_list, priorities):
                params['priority'] = float(priority)


            # Generate data for each detector (OPTIMIZED: parallel detector processing)
            detector_data = {}
            psd_dicts = {det: self.psds[det] for det in self.detectors}

            # OPTIMIZATION: Pre-generate noise for all detectors in batch
            noises = {}
            for detector_name in self.detectors:
                psd_dict = psd_dicts[detector_name]
                noises[detector_name] = self.noise_generator.generate_colored_noise(psd_dict)
            if add_glitches:
            # OPTIMIZATION: Use configurable glitch probability for overlaps
                glitch_prob = float(self.config.get('overlap_glitch_prob', 0.2))
                noises[detector_name] = self.noise_generator.add_glitches(
                            noises[detector_name], glitch_prob=glitch_prob)

            # OPTIMIZATION: Inject signals for all detectors efficiently
            for detector_name in self.detectors:
                psd_dict = psd_dicts[detector_name]
                noise = noises[detector_name]

                injected, metadata_list = self.injector.inject_overlapping_signals(
                    noise, signal_params_list, detector_name, psd_dict
                )

                if preprocess:
                    injected = self.preprocessor.preprocess(injected, psd_dict)

                detector_data[detector_name] = {
                    'strain': injected,
                    'metadata_list': metadata_list
                }

            # Determine primary and secondary targets
            snrs_array = np.array(target_snrs)
            top_2_indices = np.argsort(snrs_array)[-2:][::-1]  # Two highest SNR indices
            primary_idx = top_2_indices[0]
            secondary_idx = top_2_indices[1] if len(top_2_indices) > 1 else primary_idx
            
            # Attach network_snr to ALL parameters in the list
            attach_network_snr_safe(signal_params_list)

            # Construct the sample
            sample = {
                'sample_id': sample_id,
                'sample_type': 'overlap',
                'is_edge_case': is_edge_case,
                'detector_data': detector_data,
                'parameters': signal_params_list,
                'metadata': {
                    'n_signals': n_signals,
                    'target_snrs': target_snrs,
                    'primary_idx': primary_idx,
                    'secondary_idx': secondary_idx,
                }
            }

            return sample
    
    
    def _sample_event_type(self) -> str:
        """Sample event type from distribution"""
        types = list(EVENT_TYPE_DISTRIBUTION.keys())
        probs = list(EVENT_TYPE_DISTRIBUTION.values())
        return np.random.choice(types, p=probs)

    def _sample_event_type_subset(self, subset: List[str]) -> str:
        """Sample an event type restricted to a given subset but weighted by the global distribution.

        If the subset is empty or None, falls back to the global distribution.
        """
        if not subset:
            return self._sample_event_type()
        # Build weighted probabilities from global distribution for the subset
        global_dist = EVENT_TYPE_DISTRIBUTION
        filtered = {k: global_dist.get(k, 0.0) for k in subset}
        total = sum(filtered.values())
        if total <= 0:
            # Fall back to uniform over subset
            return np.random.choice(subset)
        probs = [filtered[k] / total for k in subset]
        return np.random.choice(subset, p=probs)
    
    def _sample_snr_regime(self) -> str:
        """Sample SNR regime from distribution"""
        regimes = list(SNR_DISTRIBUTION.keys())
        probs = list(SNR_DISTRIBUTION.values())
        return np.random.choice(regimes, p=probs)



    def _calculate_type_counts(self):
        """Calculate how many samples of each extreme type to generate."""
        self.type_counts = {}
        
        if not self.enabled:
            return
        
        for type_name, type_config in self.extreme_types_config.items():
            if type_config.get('enabled', True):
                fraction = type_config.get('fraction', 0.1)
                self.type_counts[type_name] = fraction
                

    def _sample_parameters(self, event_type: str) -> Dict:
        """Sample parameters for given event type."""
        snr_regime = self._sample_snr_regime()
        
        if event_type == 'BBH':
            return self.parameter_sampler.sample_bbh_parameters(snr_regime, False)
        elif event_type == 'BNS':
            return self.parameter_sampler.sample_bns_parameters(snr_regime, False)
        elif event_type == 'NSBH':
            return self.parameter_sampler.sample_nsbh_parameters(snr_regime, False)
        else:
            return self.parameter_sampler.sample_bbh_parameters(snr_regime, False)

    
    def _should_generate_extreme_case(self) -> Tuple[bool, Optional[str]]:
        """Determine if current sample should be an extreme case."""
        if not self.extreme_enabled:
            return False, None
        
        if np.random.random() > self.extreme_fraction:
            return False, None
        
        enabled_types = {
            name: config for name, config in self.extreme_types_config.items()
            if config.get('enabled', True)
        }
        
        if not enabled_types:
            return False, None
        
        types = list(enabled_types.keys())
        fractions = [enabled_types[t].get('fraction', 0.1) for t in types]
        
        total = sum(fractions)
        if total == 0:
            return False, None
        
        probs = [f / total for f in fractions]
        extreme_type = np.random.choice(types, p=probs)
        
        return True, extreme_type

    # ========================================================================
    # 1️⃣ NEAR-SIMULTANEOUS MERGERS (Δt < 0.2s)
    # ========================================================================
    
    def _generate_near_simultaneous_mergers(self, sample_id: int) -> Dict:
        """
        Generate overlapping signals with merger times within 0.2 seconds.
        Critical for training PriorityNet to disentangle closely-spaced events.
        """
        config = self.extreme_types_config.get('near_simultaneous_mergers', {})
        
        delta_t_range = config.get('delta_t_range', [0.05, 0.2])
        min_overlap = config.get('min_overlap', 2)
        max_overlap = config.get('max_overlap', 3)
        event_pairs = config.get('event_type_pairs', [['BBH', 'BBH']])
        
        # Determine number of overlapping signals
        n_signals = np.random.randint(min_overlap, max_overlap + 1)
        
        # Generate base parameters for each signal
        parameters_list = []
        
        # Reference merger time
        t_ref = 1187008882.0  # GW150914-like GPS time
        
        for i in range(n_signals):
            # Select event type pair
            event_type = np.random.choice([pair[i % len(pair)] for pair in event_pairs])
            
            # Generate standard parameters
            params = self._sample_parameters(event_type)
            
            # Set merger time with small separation
            if i == 0:
                params['geocent_time'] = t_ref
            else:
                # Add small time offset
                delta_t = np.random.uniform(delta_t_range[0], delta_t_range[1])
                params['geocent_time'] = t_ref + delta_t * (1 if np.random.random() > 0.5 else -1)
            
            # Ensure SNRs are reasonable (not too weak)
            target_snr = np.random.uniform(10, 30)
            if 'target_snr' not in params:
                self._set_target_snr(params, target_snr, reason='near_simultaneous_target')
            
            parameters_list.append(params)
        
        # Generate overlapping sample
        sample = self.generate_custom_overlap(parameters_list, sample_id, is_edge_case=False)
        
        # Mark as extreme case
        sample['is_extreme_case'] = True
        sample['extreme_case_type'] = 'extreme_near_simultaneous_mergers'
        sample['extreme_metadata'] = {
            'delta_t_actual': max([p['geocent_time'] for p in parameters_list]) - 
                            min([p['geocent_time'] for p in parameters_list]),
            'n_signals': n_signals
        }
        
        return sample
    
    # ========================================================================
    # 2️⃣ EXTREME MASS-RATIO INSPIRALS (q < 0.05)
    # ========================================================================
    
    def _generate_extreme_mass_ratio(self, sample_id: int) -> Dict:
        """
        Generate extreme mass-ratio binaries (q < 0.1, M_total < 30 M_☉).
        
        Scientific Motivation:
            Tests waveform models and parameter estimation for asymmetric systems.
            Challenges spin-orbit coupling models and higher-order PN corrections.
        
        Physical Scenario:
            Stellar-mass BH + stellar-mass compact object (e.g., GW190814 with 
            q ~ 0.112 [Abbott et al., ApJL 2020]), or potential IMBH-stellar systems.
        
        Astrophysical Context:
            Expected from globular cluster dynamics and hierarchical mergers.
            Formation channels: dynamical capture, primordial binaries.
        
        Parameter Ranges:
            - Mass ratio: q ∈ [0.01, 0.1]
            - Total mass: M_total ∈ [10, 30] M_☉
            - Primary mass: m₁ ∈ [9, 27] M_☉
            - Secondary mass: m₂ = q × m₁
        
        Returns:
            Sample with extreme q, enhanced higher harmonics
        """
        
        config = self.extreme_types_config.get('extreme_mass_ratio', {})
        q_range = config.get('q_range', [0.01, 0.1])
        total_mass_range = config.get('total_mass_range', [10.0, 30.0])
        
        # Sample extreme mass ratio
        q = np.random.uniform(q_range[0], q_range[1])
        m_total = np.random.uniform(total_mass_range[0], total_mass_range[1])
        
        # Compute masses
        m1 = m_total / (1 + q)
        m2 = m1 * q
        
        # Derived quantities
        chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        eta = (m1 * m2) / (m1 + m2)**2
        
        # Sample other parameters - balance event types across extreme cases
        # Extreme mass ratios: favor NSBH over BBH (more astrophysical)
        event_types = ['BBH', 'NSBH', 'BNS']
        weights = [0.4, 0.5, 0.1]  # Shift toward NSBH for extreme mass ratios
        event_type = np.random.choice(event_types, p=weights)
        params = self._sample_parameters(event_type)
        
        # Override mass parameters
        params.update({
            'mass_1': m1,
            'mass_2': m2,
            'chirp_mass': chirp_mass,
            'symmetric_mass_ratio': eta,
            'mass_ratio': q,
            'total_mass': m_total,
            'type': 'BBH'
        })

        # For extreme low q, increase f_lower to fit waveform in duration
        if q < 0.05:
            params['f_lower'] = 50.0
        
        # Generate sample
        sample = self._generate_sample_from_params(sample_id, params, 'extreme_mass_ratio')
        
        
        # Ensure priorities exist and are normalized
        sample = self._ensure_sample_priorities(sample)

        sample['is_extreme_case'] = True
        sample['extreme_case_type'] = 'extreme_mass_ratio'
        sample['extreme_metadata'] = {
            'mass_ratio': float(q),
            'total_mass_msun': float(m_total),
            'primary_mass_msun': float(m1),
            'secondary_mass_msun': float(m2),
            'chirp_mass_msun': float(chirp_mass),
            'symmetric_mass_ratio': float(eta),
            'higher_harmonics_enhanced': True
        }
        
        return sample

    # ========================================================================
    # 3️⃣ HIGH-SPIN ALIGNED/ANTI-ALIGNED (|χ| > 0.95)
    # ========================================================================
    
    def _generate_high_spin_aligned(self, sample_id: int) -> Dict:
        """
        Generate systems with near-maximal aligned spins (χ_eff > 0.8).
        
        Scientific Motivation:
            Tests spin-orbit coupling, spin-induced precession, and higher-order
            spin effects in waveform models. Critical for distinguishing formation
            channels [Gerosa & Berti, PRD 2017].
        
        Physical Scenario:
            Rapid rotation inherited from stellar progenitors (isolated evolution)
            or dynamical assembly with spin alignment from accretion disks.
        
        Astrophysical Context:
            χ_eff distribution peaks near zero for field binaries [GWTC-3].
            High χ_eff (>0.7) suggests isolated evolution or hierarchical formation.
        
        Parameter Ranges:
            - Dimensionless spins: |a₁|, |a₂| ∈ [0.8, 0.998]
            - Tilt angles: θ₁, θ₂ ∈ [0, 0.2] rad (nearly aligned)
            - Effective spin: χ_eff ∈ [0.8, 0.95]
        
        Returns:
            Sample with extreme aligned spins, enhanced ringdown
        """
        
        config = self.extreme_types_config.get('high_spin_aligned', {})
        spin_range = config.get('spin_range', [0.8, 0.998])
        tilt_max = config.get('tilt_max', 0.2)  # radians
        
        # Sample parameters - balance event types for high spin systems
        # High spins occur in all types, but balance to reduce BBH dominance
        event_types = ['BBH', 'NSBH', 'BNS']
        weights = [0.6, 0.3, 0.1]  # More balanced distribution
        event_type = np.random.choice(event_types, p=weights)
        params = self._sample_parameters(event_type)
        
        # High aligned spins
        params['a1'] = np.random.uniform(spin_range[0], spin_range[1])
        params['a2'] = np.random.uniform(spin_range[0], spin_range[1])
        
        # Small tilt angles (aligned)
        params['tilt1'] = np.random.uniform(0.0, tilt_max)
        params['tilt2'] = np.random.uniform(0.0, tilt_max)
        
        # Compute effective spin
        m1 = params['mass_1']
        m2 = params['mass_2']
        chi_eff = (m1 * params['a1'] * np.cos(params['tilt1']) + 
                m2 * params['a2'] * np.cos(params['tilt2'])) / (m1 + m2)
        
        params['chi_eff'] = chi_eff
        
        # Generate sample
        sample = self._generate_sample_from_params(sample_id, params, 'high_spin_aligned')
        
        # Ensure priorities exist and are normalized
        sample = self._ensure_sample_priorities(sample)

        sample['is_extreme_case'] = True
        sample['extreme_case_type'] = 'high_spin_aligned'
        sample['extreme_metadata'] = {
            'spin_1': float(params['a1']),
            'spin_2': float(params['a2']),
            'tilt_1_rad': float(params['tilt1']),
            'tilt_2_rad': float(params['tilt2']),
            'chi_eff': float(chi_eff),
            'spin_alignment': 'aligned',
            'formation_channel_hint': 'isolated_evolution'
        }
        
        return sample

    # ========================================================================
    # 4️⃣ PRECESSION-DOMINATED (χ_p > 0.8)
    # ========================================================================
    
    def _generate_precession_dominated(self, sample_id: int) -> Dict:
        """
        Generate systems with strong spin-precession (χ_p > 0.5).
        
        Scientific Motivation:
            Tests precession-handling in waveform models and parameter estimation.
            Precession breaks degeneracies but complicates inference [Schmidt et al., PRD 2015].
        
        Physical Scenario:
            Misaligned spins from supernova kicks or dynamical capture.
            Causes orbital plane precession with period τ_prec ~ (c³/GM) × (r/rg)³.
        
        
        Astrophysical Context:
            χ_p distribution suggests ~20% of BBH have measurable precession [GWTC-3].
            Strong precession indicates dynamical formation or natal kicks.
        
        Parameter Ranges:
            - In-plane spin: χ_p ∈ [0.5, 0.9]
            - Tilt angles: θ₁, θ₂ ∈ [π/4, 3π/4] rad (misaligned)
            - Azimuthal angles: φ₁₂ ∈ [0, 2π] rad
        
        Returns:
            Sample with strong precession, modulated amplitude
        """
        
        config = self.extreme_types_config.get('precession_dominated', {})
        chi_p_range = config.get('chi_p_range', [0.5, 0.9])
        
        # Sample parameters - balance event types for precession systems
        # Precession occurs in all types, balance to meet target distributions
        event_types = ['BBH', 'NSBH', 'BNS']
        weights = [0.5, 0.35, 0.15]  # Favor target distribution ratios
        event_type = np.random.choice(event_types, p=weights)
        params = self._sample_parameters(event_type)
        
        # Moderate to high spins
        params['a1'] = np.random.uniform(0.5, 0.95)
        params['a2'] = np.random.uniform(0.5, 0.95)
        
        # Misaligned tilts (for precession)
        params['tilt1'] = np.random.uniform(np.pi/4, 3*np.pi/4)
        params['tilt2'] = np.random.uniform(np.pi/4, 3*np.pi/4)
        
        # Azimuthal angles
        params['phi12'] = np.random.uniform(0, 2*np.pi)
        params['phi_jl'] = np.random.uniform(0, 2*np.pi)
        
        # Compute effective precession spin
        m1 = params['mass_1']
        m2 = params['mass_2']
        q = min(m1, m2) / max(m1, m2)
        
        # Schmidt et al. (2015) formula
        A1 = 2 + 3*q/2
        A2 = 2 + 3/(2*q)
        
        S1_perp = m1**2 * params['a1'] * np.sin(params['tilt1'])
        S2_perp = m2**2 * params['a2'] * np.sin(params['tilt2'])
        
        chi_p = max(A1*S1_perp, A2*S2_perp) / (A1 * m1**2)
        params['chi_p'] = min(chi_p, 1.0)  # Cap at 1
        
        # Ensure chi_p in desired range
        if params['chi_p'] < chi_p_range[0]:
            # Boost spin magnitudes
            scale = chi_p_range[0] / params['chi_p']
            params['a1'] = min(params['a1'] * scale, 0.998)
            params['a2'] = min(params['a2'] * scale, 0.998)
            params['chi_p'] = chi_p_range[0]
        
        # Generate sample
        sample = self._generate_sample_from_params(sample_id, params, 'precession_dominated')
        
        # Ensure priorities exist and are normalized
        sample = self._ensure_sample_priorities(sample)

        sample['is_extreme_case'] = True
        sample['extreme_case_type'] = 'precession_dominated'
        sample['extreme_metadata'] = {
            'chi_p': float(params['chi_p']),
            'spin_1': float(params['a1']),
            'spin_2': float(params['a2']),
            'tilt_1_rad': float(params['tilt1']),
            'tilt_2_rad': float(params['tilt2']),
            'phi_12_rad': float(params.get('phi12', 0)),
            'precession_strength': 'strong',
            'formation_channel_hint': 'dynamical_or_kicks'
        }
        
        return sample

    # ========================================================================
    # 5️⃣ ECCENTRIC OVERLAPS (e > 0.3)
    # ========================================================================
    
    def _generate_eccentric_overlaps(self, sample_id: int) -> Dict:
        """
        Generate overlapping systems with measurable eccentricity (e₀ > 0.1).
        
        Scientific Motivation:
            Tests handling of eccentric waveforms in overlapping scenarios.
            Eccentricity breaks degeneracies but requires specialized models.
        
        Physical Scenario:
            Dynamically formed binaries retain eccentricity at merger.
            Expected from dense stellar environments (GCs, AGN disks).
        
        Astrophysical Context:
            Current waveform models (IMRPhenom, SEOBNRv4) assume e=0.
            Eccentricity detection requires specialized approximants [Hinder et al.].
        
        Parameter Ranges:
            - Eccentricity at 10 Hz: e₁₀ ∈ [0.1, 0.4]
            - 2-3 overlapping signals
            - Enhanced higher harmonics
        
        Returns:
            Overlapping sample with eccentric systems
        """
        
        config = self.extreme_types_config.get('eccentric_overlaps', {})
        ecc_range = config.get('eccentricity_range', [0.1, 0.4])
        n_signals_range = config.get('n_signals_range', [2, 3])
        
        # Number of overlapping signals
        n_signals = np.random.randint(n_signals_range[0], n_signals_range[1] + 1)
        
        parameters_list = []
        for i in range(n_signals):
            event_type = self._sample_event_type()
            params = self._sample_parameters(event_type)
            
            # Add eccentricity
            params['eccentricity'] = np.random.uniform(ecc_range[0], ecc_range[1])
            
            # Time offset for overlap
            if i > 0:
                params['geocent_time'] = np.random.uniform(0.5, 1.5)
            
            parameters_list.append(params)
        
        # Generate overlap
        sample = self.generate_custom_overlap(parameters_list, sample_id, is_edge_case=False)
        
        # Ensure priorities exist and are normalized
        sample = self._ensure_sample_priorities(sample)

        sample['is_extreme_case'] = True
        sample['extreme_case_type'] = 'eccentric_overlaps'
        sample['extreme_metadata'] = {
            'n_signals': n_signals,
            'eccentricities': [p.get('eccentricity', 0) for p in parameters_list],
            'mean_eccentricity': np.mean([p.get('eccentricity', 0) for p in parameters_list]),
            'formation_channel': 'dynamical',
            'waveform_approximant_needed': 'EccentricTD'
        }
        
        return sample

    # ========================================================================
    # 6️⃣ WEAK-STRONG SNR OVERLAPS
    # ========================================================================
    
    def _generate_weak_strong_overlaps(self, sample_id: int) -> Dict:
        """
        Generate overlapping signals with extreme SNR disparity (SNR_ratio > 5:1).
        
        Scientific Motivation:
            Tests PriorityNet's ability to detect weak signals in presence of 
            loud foreground events. Critical for catalog completeness in high-rate
            scenarios where fainter signals may be masked [Nitz et al., ApJ 2020].
        
        Physical Scenario:
            Nearby loud BBH (SNR ~ 40-80) overlapping with distant weak event
            (SNR ~ 8-12). Challenges subtraction and residual analysis pipelines.
        
        Astrophysical Context:
            Volume scaling: N(SNR > ρ) ∝ ρ⁻³ for uniform spatial distribution.
            Expect ~10% of detected events to have sub-threshold companions.
        
        Parameter Ranges:
            - Strong signal: SNR ∈ [40, 80], d_L ∈ [100, 400] Mpc
            - Weak signal: SNR ∈ [8, 12], d_L ∈ [800, 2000] Mpc
            - SNR ratio: ρ_strong / ρ_weak ∈ [5, 10]
        
        Returns:
            Overlapping sample with extreme SNR hierarchy
        """
        
        config = self.extreme_types_config.get('weak_strong_overlaps', {})
        strong_snr_range = config.get('strong_snr_range', [40, 80])
        weak_snr_range = config.get('weak_snr_range', [8, 12])
        
        parameters_list = []
        
        # Strong foreground signal (prefer compact-object types, bias from configured distribution)
        types = ['BBH', 'NSBH']
        probs = [EVENT_TYPE_DISTRIBUTION.get(t, 0.0) for t in types]
        total_p = sum(probs)
        if total_p <= 0:
            probs = [0.5, 0.5]
        else:
            probs = [p / total_p for p in probs]

        event_type_strong = np.random.choice(types, p=probs)
        params_strong = self._sample_parameters(event_type_strong)
        # Do not overwrite any existing sampler-provided target_snr
        if 'target_snr' not in params_strong:
            try:
                self._set_target_snr(params_strong, np.random.uniform(strong_snr_range[0], strong_snr_range[1]),
                                     reason='weak_strong_assign_strong')
            except Exception:
                params_strong['target_snr'] = float(np.random.uniform(strong_snr_range[0], strong_snr_range[1]))
        params_strong['luminosity_distance'] = np.random.uniform(100, 400)
        try:
            self.parameter_sampler.recompute_target_snr_from_params(params_strong)
        except Exception:
            pass
        params_strong['geocent_time'] = 0.0
        parameters_list.append(params_strong)
        
        # Weak background signal
        event_type_weak = self._sample_event_type()
        params_weak = self._sample_parameters(event_type_weak)
        if 'target_snr' not in params_weak:
            try:
                self._set_target_snr(params_weak, np.random.uniform(weak_snr_range[0], weak_snr_range[1]),
                                     reason='weak_strong_assign_weak')
            except Exception:
                params_weak['target_snr'] = float(np.random.uniform(weak_snr_range[0], weak_snr_range[1]))
        params_weak['luminosity_distance'] = np.random.uniform(800, 2000)
        try:
            self.parameter_sampler.recompute_target_snr_from_params(params_weak)
        except Exception:
            pass
        params_weak['geocent_time'] = np.random.uniform(-0.5, 0.5)  # Small offset
        parameters_list.append(params_weak)
        
        # Generate overlap
        sample = self.generate_custom_overlap(parameters_list, sample_id, is_edge_case=False)
        
        
        # Ensure priorities exist and are normalized
        sample = self._ensure_sample_priorities(sample)

        sample['is_extreme_case'] = True
        sample['extreme_case_type'] = 'weak_strong_overlaps'
        sample['extreme_metadata'] = {
            'strong_snr': float(params_strong['target_snr']),
            'weak_snr': float(params_weak['target_snr']),
            'snr_ratio': float(params_strong['target_snr'] / params_weak['target_snr']),
            'strong_distance_mpc': float(params_strong['luminosity_distance']),
            'weak_distance_mpc': float(params_weak['luminosity_distance']),
            'detection_challenge': 'subtraction_required',
            'catalog_completeness_test': True
        }
        
        return sample

    # ========================================================================
    # 7️⃣ NOISE-CONFUSED OVERLAPS
    # ========================================================================
    
        
    def _generate_noise_confused_overlaps(self, sample_id: int) -> Dict:
        """Generate overlapping signals with glitch contamination."""

        config = self.extreme_types_config.get('noise_confused_overlaps', {})
        n_signals = config.get('n_signals', 2)
        glitch_prob = config.get('glitch_probability', 0.8)
        glitch_types = config.get('glitch_types', ['blip', 'whistle', 'scattered_light'])
        glitch_placement = config.get('glitch_placement', 'random')

        # Generate overlapping signals
        parameters_list = []
        for i in range(n_signals):
            event_type = self._sample_event_type()
            params = self._sample_parameters(event_type)
            if i > 0:
                params['geocent_time'] = np.random.uniform(0.3, 1.0)
            parameters_list.append(params)

        sample = self.generate_custom_overlap(parameters_list, sample_id, is_edge_case=False)

        # ✅ ADD: Validate cosmology
        for idx, params in enumerate(parameters_list):
            if 'luminosity_distance' in params and 'redshift' in params:
                d_L = params['luminosity_distance']
                z = params['redshift']
                expected_d_C = d_L / (1 + z)

                if 'comoving_distance' in params:
                    actual_d_C = params['comoving_distance']
                    if abs(actual_d_C - expected_d_C) / expected_d_C > 0.01:
                        self.logger.warning(f"Signal {idx} cosmology error > 1%")

        sample = self._ensure_sample_priorities(sample)

        # ✅ ADD: Safe detector data access
        if 'detector_data' not in sample:
            self.logger.warning(f"No detector_data in sample {sample_id}")
            sample['is_extreme_case'] = True
            return sample

        if np.random.random() < glitch_prob:
            glitch_type = np.random.choice(glitch_types)

            for detector_name, detector_data in sample['detector_data'].items():
                if 'strain' not in detector_data:
                    continue

                strain = detector_data['strain'].copy()  # ✅ COPY!

                # Determine glitch time
                if glitch_placement == 'random':
                    glitch_time = np.random.randint(len(strain) // 4, 3 * len(strain) // 4)
                elif glitch_placement == 'near_signal':
                    glitch_time = len(strain) // 2 + np.random.randint(-100, 100)
                else:
                    glitch_time = np.random.choice([
                        np.random.randint(0, len(strain) // 4),
                        np.random.randint(3 * len(strain) // 4, len(strain))
                    ])

                # Inject glitch
                if glitch_type == 'blip':
                    glitch_width = np.random.randint(20, 100)
                    glitch_amplitude = np.random.uniform(5, 15) * np.std(strain)
                    glitch = glitch_amplitude * np.exp(
                        -((np.arange(len(strain)) - glitch_time)**2) / (2 * glitch_width**2)
                    )
                    strain += glitch

                elif glitch_type == 'whistle':
                    t = np.linspace(0, len(strain) / self.sample_rate, len(strain))
                    f0 = np.random.uniform(50, 200)
                    f1 = np.random.uniform(f0 + 50, 400)
                    freq = f0 + (f1 - f0) * (t / t[-1])
                    phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate

                    # ✅ FIXED: Better decay (Hann window)
                    hann = np.hanning(len(strain))
                    glitch_amplitude = np.random.uniform(3, 10) * np.std(strain)
                    glitch = glitch_amplitude * np.sin(phase) * hann
                    strain += glitch

                elif glitch_type == 'scattered_light':
                    freq_mod = np.random.uniform(10, 50)
                    phase_mod = 2 * np.pi * freq_mod * np.arange(len(strain)) / self.sample_rate

                    # ✅ FIXED: Additive, not multiplicative
                    glitch_amplitude = np.random.uniform(2, 8) * np.std(strain)
                    glitch = glitch_amplitude * (1 + np.sin(phase_mod)) / 2
                    strain += glitch

                sample['detector_data'][detector_name]['strain'] = strain.astype(np.float32)

            sample['has_glitch'] = True
            sample['glitch_type'] = glitch_type

        sample['is_extreme_case'] = True
        sample['extreme_case_type'] = 'noise_confused_overlaps'
        sample['extreme_metadata'] = {
            'n_signals': n_signals,
            'glitch_present': sample.get('has_glitch', False),
            'glitch_type': sample.get('glitch_type', 'none'),
            'glitch_placement': glitch_placement,
            'data_quality_challenge': 'high',
            'veto_requirement': 'strict'
        }

        return sample
    
    # ========================================================================
    # 8️⃣ LONG-DURATION BNS OVERLAPS
    # ========================================================================
    
    def _generate_long_duration_bns_overlaps(self, sample_id: int) -> Dict:
        """
        Generate overlapping BNS systems with long inspiral in-band (T > 60s).
        
        Scientific Motivation:
            Tests computational efficiency and long-duration waveform handling.
            BNS signals spend minutes in LIGO band, challenging real-time analysis
            [Aasi et al., CQG 2015].
        
        Physical Scenario:
            Multiple BNS systems with f_low ~ 10-20 Hz entering band simultaneously.
            Requires tracking phase evolution over O(10³) cycles.
        
        Astrophysical Context:
            BNS rate: R_BNS ~ 320 Gpc⁻³yr⁻¹ [GWTC-3]
            Long duration enables early warning but increases overlap probability.
        
        Parameter Ranges:
            - Component masses: m ∈ [1.0, 2.0] M_☉
            - Starting frequency: f_low ∈ [10, 20] Hz
            - In-band time: T_inspiral > 60 s
            - Tidal deformability: Λ ∈ [0, 5000]
        
        Returns:
            Overlapping BNS sample with tidal effects
        """
        
     
    
        config = self.extreme_types_config.get('long_duration_bns_overlaps', {})
        
        #  FIX: Read correct config keys
        min_signals = config.get('min_signals', 2)
        max_signals = config.get('max_signals', 3)
        n_signals = np.random.randint(min_signals, max_signals + 1)
        
        f_lower_max = config.get('f_lower_max', 25)
        overlap_required = config.get('overlap_required', True)
        
        # ⚠️ duration_min is conceptual - we can't fit 30s in 4s window
        # Instead: use low f_lower to simulate end of long inspiral
        
        parameters_list = []
        
        for i in range(n_signals):
            params = self._sample_parameters('BNS')
            
            # Low starting frequency → represents end of long inspiral
            params['f_lower'] = np.random.uniform(10, f_lower_max)
            
            # BNS-specific masses
            params['mass_1'] = np.random.uniform(1.0, 2.0)
            params['mass_2'] = np.random.uniform(1.0, params['mass_1'])
            
            # Tidal parameters
            params['lambda_1'] = np.random.uniform(0, 5000)
            params['lambda_2'] = np.random.uniform(0, 5000)
            
            # Compute tidal deformability
            m1 = params['mass_1']
            m2 = params['mass_2']
            q = m2 / m1
            eta = (m1 * m2) / (m1 + m2)**2
            
            lambda_tilde = (8.0/13) * (
                (1 + 7*eta - 31*eta**2) * (params['lambda_1'] + params['lambda_2']) +
                np.sqrt(1 - 4*eta) * (1 + 9*eta - 11*eta**2) * 
                (params['lambda_1'] - params['lambda_2'])
            )
            
            params['lambda_tilde'] = lambda_tilde
            
            #  FIX: Create OVERLAPS if required
            if overlap_required and i > 0:
                # Overlap in last 2 seconds before merger
                params['geocent_time'] = np.random.uniform(-2.0, 2.0)
            else:
                # Well-separated
                params['geocent_time'] = i * 1.5 if i > 0 else 0.0
            
            params['type'] = 'BNS'
            parameters_list.append(params)
        
        # Generate overlap
        sample = self.generate_custom_overlap(parameters_list, sample_id, is_edge_case=False)
        
        # Ensure priorities exist and are normalized
        sample = self._ensure_sample_priorities(sample)

        sample['is_extreme_case'] = True
        sample['extreme_case_type'] = 'long_duration_bns_overlaps'
        sample['extreme_metadata'] = {
            'n_signals': n_signals,
            'f_lower_hz': [p['f_lower'] for p in parameters_list],
            'lambda_tilde': [p.get('lambda_tilde', 0) for p in parameters_list],
            'note': 'Last 4s of long-duration BNS inspiral',
            'equivalent_full_duration_s': '30+',  # What it represents
            'computational_challenge': 'low_frequency_long_inspiral',
            'early_warning_potential': True
        }
        
        return sample


    # ========================================================================
    # 9️⃣ DETECTOR DROPOUTS
    # ========================================================================
    
    def _generate_detector_dropouts(self, sample_id: int) -> Dict:
        """
        Generate signals with 1-2 detector outages during event.
        
        Scientific Motivation:
            Tests robustness to incomplete detector networks. Real O3 experienced
            ~30% duty cycle per detector [Abbott et al., PRX 2021].
        
        Physical Scenario:
            Signal visible in H1 only (L1 offline), or H1+V1 (L1 dropout).
            Degrades sky localization and parameter estimation.
        
        Astrophysical Context:
            Future multi-detector networks (LIGO-Virgo-KAGRA-LIGO India) increase
            redundancy, but individual detector failures remain probable.
        
        Network Configurations:
            - H1 only: Δθ ~ 360° (no localization)
            - H1+L1: Δθ ~ 20 deg² (good)
            - H1+V1: Δθ ~ 100 deg² (moderate)
            - Full network: Δθ ~ 10 deg² (excellent)
        
        Returns:
            Sample with zeroed-out detector data
        """
        
        config = self.extreme_types_config.get('detector_dropouts', {})
        dropout_prob = config.get('dropout_probability', 0.5)
        min_active = config.get('min_active_detectors', 1)
        
        # Generate base sample
        sample = self._generate_single_sample(
            sample_id=sample_id,
            is_edge_case=False,
            add_glitches=False,
            preprocess=True
        )
        
        if np.random.random() < dropout_prob:
            # Drop 1-2 detectors
            n_detectors = len(self.detectors)
            n_to_drop = np.random.randint(1, min(3, n_detectors - min_active + 1))
            
            detectors_to_drop = list(np.random.choice(
                self.detectors, 
                size=n_to_drop, 
                replace=False
            ))
            
            # Zero out dropped detectors
            for det in detectors_to_drop:
                if det in sample['detector_data']:
                    sample['detector_data'][det]['strain'] *= 0
            
            active_dets = [d for d in self.detectors if d not in detectors_to_drop]
            
            sample['active_detectors'] = active_dets
            sample['dropped_detectors'] = detectors_to_drop
            
            # Estimate sky localization degradation
            if len(active_dets) == 1:
                sky_area_deg2 = 41253  # Full sky
            elif len(active_dets) == 2:
                if 'H1' in active_dets and 'L1' in active_dets:
                    sky_area_deg2 = 20
                else:
                    sky_area_deg2 = 100
            else:
                sky_area_deg2 = 10
            
            sample['sky_localization_deg2'] = sky_area_deg2
        
        # Ensure priorities exist and are normalized
        sample = self._ensure_sample_priorities(sample)

        sample['is_extreme_case'] = True
        sample['extreme_case_type'] = 'detector_dropouts'
        sample['extreme_metadata'] = {
            'n_active_detectors': len(sample.get('active_detectors', self.detectors)),
            'active_detectors': sample.get('active_detectors', self.detectors),
            'dropped_detectors': sample.get('dropped_detectors', []),
            'sky_localization_deg2': sample.get('sky_localization_deg2', 10),
            'parameter_estimation_degradation': 'moderate_to_severe'
        }
        
        return sample

    # ========================================================================
    # 🔟 COSMOLOGICAL-SCALE DISTANCES
    # ========================================================================
    
    def _generate_cosmological_distance(self, sample_id: int) -> Dict:
        """
        Generate signals at cosmological distances (z > 0.5, d_L > 2 Gpc).
        
        Scientific Motivation:
            Tests low-SNR detection and Malmquist bias corrections.
            Probes cosmological merger rate evolution ρ(z) [Fishbach & Holz, ApJL 2017].
        
        Physical Scenario:
            High-mass BBH (M_total > 80 M_☉) at z ~ 0.5-1.0.
            Source-frame masses scaled by (1+z).
        
        Astrophysical Context:
            GWTC-3 reports z_max ~ 0.9 (GW190521).
            Future detectors (A+, Voyager) will probe z > 2.
        
        Parameter Ranges:
            - Redshift: z ∈ [0.5, 1.0]
            - Luminosity distance: d_L ∈ [2000, 5000] Mpc
            - Source-frame masses: M_source = M_detector / (1+z)
            - Detector-frame masses: M ∈ [80, 150] M_☉
        
        Cosmology:
            H₀ = 67.4 km/s/Mpc, Ωₘ = 0.315, ΩΛ = 0.685 [Planck 2018]
        
        Returns:
            Sample with cosmologically redshifted parameters
        """
        
        config = self.extreme_types_config.get('cosmological_distance', {})
        z_range = config.get('redshift_range', [0.5, 1.0])
        
        # Sample redshift
        z = np.random.uniform(z_range[0], z_range[1])
        
        # Compute luminosity distance using astropy for consistency
        from astropy.cosmology import Planck18 as cosmo
        import astropy.units as u
        d_L = cosmo.luminosity_distance(z).to(u.Mpc).value

        # Compute comoving distance consistently
        d_C = d_L / (1 + z)

        # Verify relationship
        assert d_C <= d_L, f"Unphysical: d_C={d_C} > d_L={d_L}"
        
        # Sample source-frame masses
        M_source_total = np.random.uniform(40, 75)  # Source frame
        q = np.random.uniform(0.5, 1.0)

        M1_source = M_source_total / (1 + q)
        M2_source = M1_source * q

        # Detector-frame masses (redshifted)
        M1_detector = M1_source * (1 + z)
        M2_detector = M2_source * (1 + z)
        
        # Generate parameters - balance event types for cosmological distances
        # High-redshift favors massive systems, but balance for target distribution
        event_types = ['BBH', 'NSBH', 'BNS']
        weights = [0.6, 0.3, 0.1]  # Compromise between physics and target distribution
        event_type = np.random.choice(event_types, p=weights)
        params = self._sample_parameters(event_type)

        # Override with cosmological values (consistent with astropy cosmology)
        params.update({
        'mass_1': M1_detector,
        'mass_2': M2_detector,
        'luminosity_distance': d_L,
        'comoving_distance': d_C,  #  Now consistent
        'redshift': z,
        'mass_1_source': M1_source,
        'mass_2_source': M2_source,
        'chirp_mass': (M1_detector * M2_detector)**(3/5) / (M1_detector + M2_detector)**(1/5),
        'chirp_mass_source': (M1_source * M2_source)**(3/5) / (M1_source + M2_source)**(1/5),
        'f_lower': 50.0  # Higher f_lower for cosmological distances
        })
        
        # Low SNR expected - only set if sampler didn't specify
        if 'target_snr' not in params:
            # self._set_target_snr(params, np.random.uniform(8, 12), reason='cosmological_distance_default')
            snr_regime = self._sample_snr_regime()
            params['target_snr'] = self._sample_snr_regime(snr_regime)
        
        # Generate sample
        sample = self._generate_sample_from_params(sample_id, params, 'cosmological_distance')
        
        
        # Ensure priorities exist and are normalized
        sample = self._ensure_sample_priorities(sample)

        sample['is_extreme_case'] = True
        sample['extreme_case_type'] = 'cosmological_distance'
        sample['extreme_metadata'] = {
            'redshift': float(z),
            'luminosity_distance_mpc': float(d_L),
            'comoving_distance_mpc': float(d_L / (1 + z)),
            'source_frame_mass_1_msun': float(M1_source),
            'source_frame_mass_2_msun': float(M2_source),
            'detector_frame_mass_1_msun': float(M1_detector),
            'detector_frame_mass_2_msun': float(M2_detector),
            'time_dilation_factor': float(1 + z),
            'cosmology': 'Planck2018',
            'H0_km_s_Mpc': 67.4,
            'low_snr_cosmological': True
        }
        
        return sample

    # ========================================================================
    # MAIN GENERATION DISPATCHER
    # ========================================================================
    
    def _generate_edge_case(self, sample_id: int, edge_type: str) -> Dict:
        """Generate explicit edge case of specified type."""

        # Use config from YAML file if available, otherwise fall back to defaults
        config = self.config if self.config else {}

        # Dispatch to appropriate edge case generation method
        if edge_type == 'high_mass_ratio':
            return self._generate_high_mass_ratio_sample(sample_id, config)
        elif edge_type == 'extreme_spins':
            return self._generate_edge_case_sample(sample_id, 'extreme_spins', config)
        elif edge_type == 'eccentric_mergers':
            return self._generate_edge_case_sample(sample_id, 'eccentric_mergers', config)
        elif edge_type == 'strong_glitches':
            return self._generate_edge_case_sample(sample_id, 'strong_glitches', config)
        elif edge_type == 'psd_drift':
            return self._generate_psd_drift_sample(sample_id, config)
        elif edge_type == 'detector_dropout':
            return self._generate_edge_case_sample(sample_id, 'detector_dropout', config)
        elif edge_type == 'multimodal_posteriors':
            return self._generate_multimodal_posterior_sample(sample_id, config)
        elif edge_type == 'heavy_tailed_regions':
            return self._generate_heavy_tailed_sample(sample_id, config)
        elif edge_type == 'subtle_ranking':
            return self._generate_subtle_ranking_overlap(sample_id, config)
        elif edge_type == 'heavy_overlaps':
            return self._generate_edge_case_sample(sample_id, 'heavy_overlaps', config)
        elif edge_type == 'uninformative_priors':
            return self._generate_uninformative_prior_sample(sample_id, config)
        elif edge_type == 'precessing_systems':
            return self._generate_edge_case_sample(sample_id, 'precessing_systems', config)
        elif edge_type == 'short_duration_high_mass':
            return self._generate_edge_case_sample(sample_id, 'short_duration_high_mass', config)
        elif edge_type == 'low_snr_threshold':
            return self._generate_edge_case_sample(sample_id, 'low_snr_threshold', config)
        elif edge_type == 'sky_position_extremes':
            return self._generate_edge_case_sample(sample_id, 'sky_position_extremes', config)
        elif edge_type == 'pre_merger_samples':
            return self._generate_pre_merger_sample(sample_id)
        else:
            self.logger.warning(f"Unknown edge case type: {edge_type}, generating regular sample")
            return self._generate_single_sample(sample_id, is_edge_case=True, add_glitches=True, preprocess=True)

    def _generate_extreme_case(self, sample_id: int, extreme_type: str) -> Dict:
        """
        Generate extreme case of specified type.
        
        Args:
            sample_id: Unique sample identifier
            extreme_type: One of the 10 extreme case types
            
        Returns:
            Generated sample dictionary
        """
        dispatch = {
            'near_simultaneous_mergers': self._generate_near_simultaneous_mergers,  
            'extreme_mass_ratio': self._generate_extreme_mass_ratio,                
            'high_spin_aligned': self._generate_high_spin_aligned,                  
            'precession_dominated': self._generate_precession_dominated,            
            'eccentric_overlaps': self._generate_eccentric_overlaps,                
            'weak_strong_overlaps': self._generate_weak_strong_overlaps,            
            'noise_confused_overlaps': self._generate_noise_confused_overlaps,      
            'long_duration_bns_overlaps': self._generate_long_duration_bns_overlaps,
            'detector_dropouts': self._generate_detector_dropouts,                  
            'cosmological_distance': self._generate_cosmological_distance,
            'pre_merger_samples': self._generate_pre_merger_sample,         
        }
        
        if extreme_type not in dispatch:
            self.logger.warning(f"Unknown extreme case type: {extreme_type}, generating regular sample")
            return self._generate_single_sample(
                sample_id=sample_id,
                is_edge_case=False,
                add_glitches=False,
                preprocess=True
            )
            
            
        return dispatch[extreme_type](sample_id)


    def _generate_near_simultaneous_mergers(self, sample_id: int) -> Dict:
        """
        Generate overlapping signals with Δt < 0.2s between mergers.
        
        Scientific motivation: Tests PriorityNet's ability to resolve near-simultaneous
        events, critical for high-rate observing runs (O5+) where event rate may 
        exceed 1 per minute.
        
        Physical scenario: Two independent binary mergers occurring within light-travel
        time across detectors (~20ms for H1-L1), requiring coherent multi-detector
        analysis for separation.
        
        Returns:
            Sample with 2-3 overlapping signals, Δt ∈ [50ms, 200ms]
        """
        
        # Configuration
        config = self.extreme_types_config.get('near_simultaneous_mergers', {})
        delta_t_range = config.get('delta_t_range', [0.05, 0.2])
        n_signals_range = config.get('n_signals_range', [2, 3])
        
        # Sample number of signals
        n_signals = np.random.randint(n_signals_range[0], n_signals_range[1] + 1)
        
        # Get event type pairs from config
        event_pairs = config.get('event_type_pairs', [['BBH', 'BBH'], ['BBH', 'BNS'], ['BNS', 'BNS']])
        
        #  NEW (CORRECT - pick one pair, then extend to n_signals):
        selected_pair = event_pairs[np.random.randint(len(event_pairs))]
        
        # Extend to match n_signals
        event_types = []
        for i in range(n_signals):
            if i < len(selected_pair):
                event_types.append(selected_pair[i])
            else:
                # If need more signals than pair has, randomly pick from pair
                event_types.append(selected_pair[np.random.randint(len(selected_pair))])
        
        # Generate parameters for each signal
        parameters_list = []
        actual_delta_ts = []
        
        for i in range(n_signals):
            event_type = event_types[i]
            params = self._sample_parameters(event_type)
            
            # Set merger times
            if i == 0:
                params['geocent_time'] = 0.0
            else:
                delta_t = np.random.uniform(delta_t_range[0], delta_t_range[1])
                params['geocent_time'] = delta_t
                actual_delta_ts.append(delta_t)
            
            parameters_list.append(params)
        
        # Generate overlap
        sample = self.generate_custom_overlap(parameters_list, sample_id, is_edge_case=False)
        
        #  At end of EVERY extreme case function: ensure priorities and n_signals
        sample = self._ensure_sample_priorities(sample)

        # Add metadata
        sample['is_extreme_case'] = True
        sample['extreme_case_type'] = 'near_simultaneous_mergers'
        sample['extreme_metadata'] = {
            'delta_t_max_ms': max(actual_delta_ts) * 1000 if actual_delta_ts else 0.0,
            'delta_t_min_ms': min(actual_delta_ts) * 1000 if actual_delta_ts else 0.0,
            'n_signals': n_signals,
            'event_types': event_types,
            'merger_times': [p['geocent_time'] for p in parameters_list],
            'temporal_overlap': 'near_simultaneous'
        }
        
        return sample

    

    def generate_custom_overlap(self, parameters_list: List[Dict], sample_id: int, is_edge_case: bool = False) -> Dict:
        """
        Generate overlapping scenario from parameter list.

        Args:
            parameters_list: List of parameter dicts for each signal (or None/empty for noise-only)
            sample_id: Unique sample identifier
            is_edge_case: Whether this is an edge case sample

        Returns:
            Sample dict with overlapping signals and proper priorities.
            If parameters_list is empty/None, returns noise-only overlap sample.

        Raises:
            TypeError: If parameters_list is not a list/tuple
        """

        # ========================================================================
        # INPUT VALIDATION
        # ========================================================================
        if parameters_list is None:
            self.logger.warning(
                f"generate_custom_overlap (sample {sample_id}): "
                f"received None - treating as empty list (noise-only)"
            )
            parameters_list = []

        if not isinstance(parameters_list, (list, tuple)):
            raise TypeError(
                f"generate_custom_overlap (sample {sample_id}): "
                f"parameters_list must be a list or tuple, got {type(parameters_list).__name__}"
            )

        n_signals = len(parameters_list)

        # ========================================================================
        # NOISE-ONLY FALLBACK
        # ========================================================================
        if n_signals == 0:
            self.logger.warning(
                f"generate_custom_overlap (sample {sample_id}): "
                f"empty parameters_list - generating noise-only overlap sample"
            )

            detector_data = {}

            for detector_name in self.detectors:
                psd_dict = self.psds[detector_name]

                # Generate noise-only strain
                noise = self.noise_generator.generate_colored_noise(psd_dict)
                combined = noise

                # Apply preprocessing if enabled
                if self.preprocess_enabled:
                    try:
                        combined = self.preprocessor.preprocess(combined, psd_dict)
                    except Exception as e:
                        self.logger.debug(f"Preprocessing failed for detector {detector_name}: {e}")

                detector_data[detector_name] = {
                    'strain': combined.astype(np.float32),
                    'psd': psd_dict.get('psd', None),
                    'frequencies': psd_dict.get('frequencies', None)
                }

            sample = {
                'id': sample_id,
                'sample_id': f'overlap_{sample_id:06d}',
                'type': 'overlap',
                'is_overlap': False,
                'n_signals': 0,
                'is_edge_case': is_edge_case,
                'parameters': [],
                'priorities': [],
                'detector_data': detector_data,
                'metadata': {
                    'sample_id': sample_id,
                    'n_signals': 0,
                    'mean_snr': 0.0,
                    'max_snr': 0.0,
                    'generator': 'custom_overlap',
                    'noise_only': True
                }
            }

            return sample

        # ========================================================================
        # ✅ COSMOLOGY VALIDATION - FIX FOR 16 VIOLATIONS!
        # ========================================================================
        for i, params in enumerate(parameters_list):
            if 'luminosity_distance' in params and 'redshift' in params:
                d_L = params['luminosity_distance']
                z = params['redshift']
                expected_d_C = d_L / (1 + z)

                actual_d_C = params.get('comoving_distance', expected_d_C)

                # Check and correct if needed
                if actual_d_C > d_L or abs(actual_d_C - expected_d_C) / expected_d_C > 0.01:
                    # self.logger.warning(
                    #     f"Sample {sample_id}, signal {i}: Correcting cosmology. "
                    #     f"d_C was {actual_d_C:.1f}, now {expected_d_C:.1f}"
                    # )
                    params['comoving_distance'] = float(expected_d_C)

        # ========================================================================
        # STANDARD OVERLAP GENERATION
        # ========================================================================
        detector_data = {}
        signal_contributions = {det: {} for det in self.detectors}

        for detector_name in self.detectors:
            psd_dict = self.psds[detector_name]

            # Generate noise
            noise = self.noise_generator.generate_colored_noise(psd_dict)
            combined = noise.copy()

            # Inject all signals
            for i, params in enumerate(parameters_list):
                try:
                    # Generate waveform
                    strain = self.waveform_generator.generate_waveform(params, detector_name)

                    # Ensure correct length
                    expected_length = int(self.duration * self.sample_rate)
                    if len(strain) > expected_length:
                        strain = strain[:expected_length]
                    elif len(strain) < expected_length:
                        strain = np.pad(strain, (0, expected_length - len(strain)), mode='constant')

                    # Store individual contribution
                    signal_contributions[detector_name][i] = strain

                    # Add to combined signal
                    combined += strain

                except Exception as e:
                    self.logger.warning(f"Failed to generate overlap signal {i} for {detector_name}: {e}")
                    signal_contributions[detector_name][i] = np.zeros(int(self.duration * self.sample_rate))

            # Preprocess if enabled
            if self.preprocess_enabled:
                combined = self.preprocessor.preprocess(combined, psd_dict)

            # Store detector data
            detector_data[detector_name] = {
                'strain': combined.astype(np.float32),
                'psd': psd_dict['psd'],
                'frequencies': psd_dict['frequencies']
            }

        # ========================================================================
        # CALCULATE PRIORITIES
        # ========================================================================
        priorities = []
        for i, params in enumerate(parameters_list):
            # Try to compute SNR from individual contributions
            max_snr = 0.0
            for detector_name in self.detectors:
                if i in signal_contributions[detector_name]:
                    individual_strain = signal_contributions[detector_name][i]

                    # Estimate SNR
                    snr = self._estimate_snr_from_params(params)
                    max_snr = max(max_snr, snr)

            # Fallback if SNR calculation failed
            if max_snr < 5.0:
                max_snr = self._estimate_snr_from_params(params)

            priority = float(np.clip(max_snr, 7.0, 80.0))
            # Do not overwrite an existing sampled target_snr — only set if missing
            if 'target_snr' not in params:
                self._set_target_snr(params, priority, reason='custom_overlap_priority')
            priorities.append(priority)

        # Attach network_snr to all parameters
        attach_network_snr_safe(parameters_list)

        # ========================================================================
        # CONSTRUCT SAMPLE
        # ========================================================================
        sample = {
            'id': sample_id,
            'sample_id': f'overlap_{sample_id:06d}',
            'type': 'overlap',
            'is_overlap': n_signals > 1,
            'n_signals': n_signals,
            'is_edge_case': is_edge_case,
            'parameters': parameters_list,
            'priorities': priorities,
            'detector_data': detector_data,
            'metadata': {
                'sample_id': sample_id,
                'n_signals': n_signals,
                'mean_snr': float(np.mean(priorities)),
                'max_snr': float(np.max(priorities)),
                'generator': 'custom_overlap',
                'noise_only': False
            }
        }

        return sample
    
    
    def generate_sample_with_simulator(self, sample_id: int, n_signals: int = None) -> Dict:
        """
        Generate sample using OverlappingSignalSimulator.
        This provides COMPLETE waveforms with proper SNR calculation.
        
        Args:
            sample_id: Unique sample ID
            n_signals: Number of overlapping signals (None = random 2-4)
        
        Returns:
            Sample dict with detector_data, parameters, and priorities
        """
        
        if n_signals is None:
            n_signals = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
        
        # Generate scenario using simulator
        scenario = self.simulator.generate_overlapping_scenario(n_signals)
        
        # Generate detector noise
        noise_data = self.simulator.generate_detector_noise(
            duration=self.config.waveform.duration,
            sampling_rate=self.config.detectors[0].sampling_rate
        )
        
        # Inject signals
        injected_data, signal_contributions = self.simulator.inject_signals_to_data(
            scenario, noise_data
        )
        
        #  Calculate priorities (SNR) for each signal
        priorities = []
        parameters = []
        
        for signal in scenario['signals']:
            # Extract signal parameters
            params = {
                'mass_1': signal['mass_1'],
                'mass_2': signal['mass_2'],
                'total_mass': signal['mass_1'] + signal['mass_2'],
                'chirp_mass': ((signal['mass_1'] * signal['mass_2'])**(3/5) / 
                            (signal['mass_1'] + signal['mass_2'])**(1/5)),
                'mass_ratio': signal['mass_2'] / signal['mass_1'],
                'luminosity_distance': signal['luminosity_distance'],
                'theta_jn': signal['theta_jn'],
                'psi': signal['psi'],
                'phase': signal['phase'],
                'ra': signal['ra'],
                'dec': signal['dec'],
                'a1': signal.get('a_1', 0.0),
                'a2': signal.get('a_2', 0.0),
                'tilt1': signal.get('tilt_1', 0.0),
                'tilt2': signal.get('tilt_2', 0.0),
                'geocent_time': signal.get('geocent_time', 0.0),
                'type': self._classify_signal_type(signal['mass_1'], signal['mass_2'])
            }
            
            #  Calculate SNR for this signal across all detectors
            max_snr = 0.0
            for detector_name, signal_strain in signal_contributions[detector_name].items():
                if signal['signal_id'] == signal_strain:
                    snr = estimate_snr_from_strain(
                        signal_strain,
                        sampling_rate=self.config.detectors[0].sampling_rate
                    )
                    max_snr = max(max_snr, snr)
            
            # Use luminosity distance to estimate SNR if strain method fails
            if max_snr < 5:
                # Fallback: use canonical physical SNR
                max_snr = compute_physical_snr_from_params(
                    {'mass_1': signal['mass_1'],
                     'mass_2': signal['mass_2'],
                     'luminosity_distance': signal['luminosity_distance'],
                     '_snr_ref': 15.0,
                     '_reference_mass': 30.0,
                     '_reference_distance': 500.0
                    }
                )
            
            priority = np.clip(max_snr, 7.0, 80.0)
            priorities.append(float(priority))
            # Do not overwrite an existing sampled target_snr
            if 'target_snr' not in params:
                self._set_target_snr(params, float(priority), reason='simulator_overlap_priority')
            parameters.append(params)
        
        # Convert detector data to your format
        detector_data = {}
        for detector_name, strain in injected_data.items():
            detector_data[detector_name] = {
                'strain': strain.astype(np.float32),
                'noise': noise_data[detector_name].astype(np.float32)
            }
        
        return {
            'sample_id': f'sim_{sample_id:06d}',
            'type': 'overlap' if n_signals > 1 else parameters[0]['type'],
            'is_overlap': n_signals > 1,
            'n_signals': n_signals,
            'is_edge_case': False,
            'parameters': parameters,  # List of parameter dicts
            'priorities': priorities,  #  List of SNRs (one per signal)
            'detector_data': detector_data,
            'metadata': {
                'sample_id': sample_id,
                'generator': 'OverlappingSignalSimulator',
                'n_signals': n_signals,
                'mean_snr': float(np.mean(priorities)),
                'max_snr': float(np.max(priorities))
            }
        }


    def generate_sample_from_gwtc(self, sample_id: int, event_name: str = None) -> Dict:
        """
        Generate sample from real GWTC event.
        Great for validation and augmentation with real data.
        
        Args:
            sample_id: Unique sample ID
            event_name: Specific event (e.g., 'GW150914'), or None for random
        
        Returns:
            Sample dict with real event parameters
        """
        
        # Lazy initialize GWTC loader
        if not hasattr(self, 'gwtc_loader'):
            from ahsd.data.gwtc_loader import GWTCLoader
            self.gwtc_loader = GWTCLoader()
            self.gwtc_events = self.gwtc_loader.get_gwtc_events()
            self.logger.info(f" Loaded {len(self.gwtc_events)} GWTC events")
        
        # Select event
        if event_name is None:
            # Filter high-quality events
            quality_events = self.gwtc_events[
                (self.gwtc_events['network_snr'] > 10) &
                (self.gwtc_events['mass_1_source'] > 5)
            ]
            if len(quality_events) == 0:
                quality_events = self.gwtc_events
            
            event = quality_events.sample(1).iloc[0]
        else:
            event = self.gwtc_events[self.gwtc_events['event_name'] == event_name].iloc[0]
        
        # Convert GWTC parameters to your format with validation
        try:
            mass_1 = float(event['mass_1_source'])
            mass_2 = float(event['mass_2_source'])
            luminosity_distance = float(event['luminosity_distance'])

            # Check for NaN or invalid values
            if np.isnan(mass_1) or np.isnan(mass_2) or np.isnan(luminosity_distance):
                self.logger.warning(f"GWTC event {event['event_name']} has NaN values, skipping")
                return None

            if mass_1 <= 0 or mass_2 <= 0 or luminosity_distance <= 0:
                self.logger.warning(f"GWTC event {event['event_name']} has invalid values, skipping")
                return None

            params = {
                'mass_1': mass_1,
                'mass_2': mass_2,
                'total_mass': float(event['total_mass_source']),
                'chirp_mass': float(event['chirp_mass_source']),
                'mass_ratio': mass_2 / mass_1,
                'luminosity_distance': luminosity_distance,
                'redshift': float(event['redshift']),
                'target_snr': float(event['network_snr']),  #  Real SNR from GWTC!
                'type': self._classify_signal_type(mass_1, mass_2),
                'event_name': event['event_name'],
                'gps_time': float(event['gps_time'])
            }
        except (KeyError, ValueError, TypeError) as e:
            self.logger.warning(f"Failed to extract parameters from GWTC event {event.get('event_name', 'unknown')}: {e}")
            return None
        
        # Try to download real strain data (use robust config access)
        
        # ============================================================================
        # CONFIG ACCESS HELPERS - UNIFIED PATTERN
        # ============================================================================

        def _get_config_value(*keys, default=None):
            """
            Unified config accessor supporting both dict and attribute access.
            
            This method handles the dual nature of self.config (can be dict or object)
            and provides a consistent interface with proper fallbacks.
            
            Args:
                *keys: Nested keys to traverse (e.g., 'waveform', 'duration')
                default: Default value if key not found or None
            
            Returns:
                Config value or default
            
            Examples:
                >>> # Access nested value
                >>> duration = self._get_config_value('waveform', 'duration', default=4.0)
                >>> # Access top-level value
                >>> n_samples = self._get_config_value('n_samples', default=1000)
            """
            value = self.config
            
            for key in keys:
                if value is None:
                    return default
                
                # Try dict-style access first (most common)
                if isinstance(value, dict):
                    value = value.get(key)
                # Fallback to attribute access (for dataclass/object configs)
                elif hasattr(value, key):
                    value = getattr(value, key, None)
                else:
                    return default
            
            # Return value if found and not None, otherwise default
            return value if value is not None else default


        def _get_config_duration() -> float:
            """Get waveform duration with fallback to self.duration."""
            return float(_get_config_value('waveform', 'duration', default=self.duration))


        def _get_config_approximant() -> str:
            """Get waveform approximant with fallback to 'IMRPhenomPv2'."""
            return str(_get_config_value('waveform', 'approximant', default='IMRPhenomXAS'))


        def _get_config_f_ref() -> float:
            """Get reference frequency with fallback to 20.0 Hz."""
            return float(_get_config_value('waveform', 'f_ref', default=20.0))


        def _get_config_f_lower() -> float:
            """Get lower frequency cutoff with fallback to 20.0 Hz."""
            return float(_get_config_value('waveform', 'f_lower', default=20.0))

        def _get_extreme_type_config(type_name: str) -> Dict:
            """
            Get configuration for specific extreme case type.
            
            Args:
                type_name: Extreme case type (e.g., 'near_simultaneous_mergers')
            
            Returns:
                Config dict for that type, or empty dict if not found
            
            Examples:
                >>> config = self._get_extreme_type_config('extreme_mass_ratio')
                >>> q_max = config.get('q_max', 0.05)
            """
            extreme_config = _get_config_value('extreme_cases', 'types', default={})
            
            if isinstance(extreme_config, dict):
                return extreme_config.get(type_name, {})
            
            return {}


        def _get_detector_sampling(det_name: str) -> int:
            # Try attribute-style config first
            try:
                dets = getattr(self.config, 'detectors', None)
                if dets:
                    for d in dets:
                        name = getattr(d, 'name', None)
                        sr = getattr(d, 'sampling_rate', None)
                        if name == det_name and sr:
                            return int(sr)
                    # fallback to first detector entry
                    first_sr = getattr(dets[0], 'sampling_rate', None)
                    if first_sr:
                        return int(first_sr)
            except Exception:
                pass

            # Try dict-style config
            try:
                dets = self.config.get('detectors', None)
                if isinstance(dets, list) and len(dets) > 0:
                    for d in dets:
                        if isinstance(d, dict) and d.get('name') == det_name and d.get('sampling_rate'):
                            return int(d.get('sampling_rate'))
                    if isinstance(dets[0], dict) and dets[0].get('sampling_rate'):
                        return int(dets[0].get('sampling_rate'))
            except Exception:
                pass

            # Final fallback
            return int(self.sample_rate)

        duration_cfg = _get_config_duration()

        detector_data = {}
        for detector_name in ['H1', 'L1', 'V1']:
            if detector_name in event.get('detectors', []):
                sample_rate_cfg = _get_detector_sampling(detector_name)
                try:
                    strain = self.gwtc_loader.download_strain(
                        event['event_name'],
                        detector=detector_name,
                        duration=duration_cfg,
                        sample_rate=sample_rate_cfg
                    )
                except Exception as e:
                    self.logger.debug(f"Strain download exception for {detector_name}: {e}")
                    strain = None

                if strain is not None:
                    detector_data[detector_name] = {
                        'strain': strain.astype(np.float32),
                        'is_real_data': True,
                        'psd': self.psds.get(detector_name, {}).get('psd') if hasattr(self, 'psds') else None,
                        'frequencies': self.psds.get(detector_name, {}).get('frequencies') if hasattr(self, 'psds') else None
                    }

        # If strain download failed for all detectors, synthesize per-detector strain using GWTC params
        if len(detector_data) == 0:
            # self.logger.warning(f"Real strain download failed for {event['event_name']}, synthesizing signals using GWTC parameters")
            # Build synthetic strains per detector present in the event
            for detector_name in event.get('detectors', []):
                sample_rate_cfg = _get_detector_sampling(detector_name)
                # Create a local waveform generator matching requested sampling and duration
                try:
                    local_wg = WaveformGenerator(sample_rate=sample_rate_cfg, duration=duration_cfg)
                    synth = local_wg.generate_waveform(params, detector_name)
                except Exception as e:
                    self.logger.debug(f"Waveform generation failed for {detector_name}: {e}")
                    # Fallback to global waveform generator and resample if necessary
                    try:
                        synth = self.waveform_generator.generate_waveform(params, detector_name)
                    except Exception as e2:
                        self.logger.warning(f"Fallback waveform generation failed: {e2}")
                        synth = np.zeros(int(self.sample_rate * self.duration), dtype=np.float32)

                # Ensure dtype and length
                try:
                    synth = np.asarray(synth, dtype=np.float32)
                except Exception:
                    synth = np.array(synth, dtype=np.float32)

                detector_data[detector_name] = {
                    'strain': synth,
                    'is_real_data': False,
                    'psd': self.psds.get(detector_name, {}).get('psd') if hasattr(self, 'psds') else None,
                    'frequencies': self.psds.get(detector_name, {}).get('frequencies') if hasattr(self, 'psds') else None
                }

        return {
            'sample_id': f'gwtc_{sample_id:06d}_{event["event_name"]}',
            'type': params['type'],
            'is_overlap': False,
            'n_signals': 1,
            'is_edge_case': False,
            'is_real_event': True,
            'parameters': params,  # Dict for single
            'priorities': [params['target_snr']],  #  Real SNR from GWTC!
            'detector_data': detector_data,
            'metadata': {
                'sample_id': sample_id,
                'generator': 'GWTC',
                'event_name': event['event_name'],
                'observing_run': event.get('observing_run', 'Unknown'),
                'far': float(event.get('far', 0.0))
            }
        }


    def _compute_snr_from_params(self, params: Dict, detector_data: Optional[Dict] = None) -> float:
        """
        Compute SNR for a signal from parameters.
        Uses either actual detector data or distance-based estimate.
        """
        
        if detector_data is not None:
            # Calculate from actual strain if available
            max_snr = 0.0
            for detector_name, data in detector_data.items():
                try:
                    strain = data.get('strain')
                    if strain is not None and len(strain) > 0:
                        snr = estimate_snr_from_strain(
                            strain,
                            psd=data.get('psd'),
                            sampling_rate=self.sample_rate
                        )
                        max_snr = max(max_snr, snr)
                except:
                    continue
            
            if max_snr > 5:
                unclipped = float(max_snr)
                clipped = float(np.clip(unclipped, 7.0, 80.0))
                try:
                    if clipped >= 79.999 and unclipped > 80.0:
                        import traceback
                        short_params = {k: params.get(k) for k in ('mass_1', 'mass_2', 'luminosity_distance')}
                        self.logger.warning(f"[SNR-CLIP] _compute_snr_from_params clipped {unclipped:.2f} -> {clipped:.2f}; params={short_params}")
                        # stack = ''.join(traceback.format_stack(limit=6))
                        # self.logger.warning(stack)
                except Exception:
                    pass

                return clipped
        
        # Fallback: Distance-based SNR estimate
        distance = params.get('luminosity_distance', 500.0)
        m1 = params.get('mass_1', 30.0)
        m2 = params.get('mass_2', 30.0)
        
        # Use canonical physical SNR
        snr = compute_physical_snr_from_params(
            params,
            snr_ref=15.0,
            reference_mass=30.0,
            reference_distance=500.0
        )
        
        # Add random factor for realism (±20%)
        snr *= np.random.uniform(0.8, 1.2)
        
        return float(np.clip(snr, 7.0, 80.0))

    def _generate_sample_with_priority(self, sample_id: int, 
                                    generator_type: str = 'simulator',
                                    event_name: str = None,
                                    **kwargs) -> Dict:
        """
        Generate sample using specified generator with proper priority calculation.
        
        Args:
            sample_id: Unique sample ID
            generator_type: 'simulator' or 'gwtc'
            event_name: For GWTC, specific event name
            **kwargs: Additional generation parameters
        
        Returns:
            Sample dict with priorities properly set
        """
        
        if generator_type == 'simulator':
            # Use OverlappingSignalSimulator
            n_signals = kwargs.get('n_signals', np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1]))
            
            scenario = self.simulator.generate_overlapping_scenario(n_signals)
            noise_data = self.simulator.generate_detector_noise(
                duration=self.duration,
                sampling_rate=self.sample_rate
            )
            injected_data, signal_contributions = self.simulator.inject_signals_to_data(
                scenario, noise_data
            )
            
            priorities = []
            parameters = []
            
            for signal in scenario['signals']:
                params = {
                    'mass_1': signal['mass_1'],
                    'mass_2': signal['mass_2'],
                    'luminosity_distance': signal['luminosity_distance'],
                    'total_mass': signal['mass_1'] + signal['mass_2'],
                    'chirp_mass': ((signal['mass_1'] * signal['mass_2'])**(3/5) / 
                                (signal['mass_1'] + signal['mass_2'])**(1/5)),
                    'type': self._classify_signal_type(signal['mass_1'], signal['mass_2']),
                    **{k: v for k, v in signal.items() if k not in ['signal_id']}
                }
                
                # Compute SNR from detector data
                detector_data_for_snr = {}
                for detector_name in self.detectors:
                    if detector_name in signal_contributions:
                        signal_strain = signal_contributions[detector_name].get(signal['signal_id'])
                        if signal_strain is not None:
                            detector_data_for_snr[detector_name] = {'strain': signal_strain}
                
                priority = self._compute_snr_from_params(params, detector_data_for_snr)
                # Do not overwrite an existing sampled target_snr
                if 'target_snr' not in params:
                    self._set_target_snr(params, priority, reason='sim_scenario_compute')
                priorities.append(priority)
                parameters.append(params)
            
            detector_data = {det: {'strain': injected_data[det].astype(np.float32)} 
                            for det in injected_data.keys()}
            
            return {
                'sample_id': f'sim_{sample_id:06d}',
                'type': 'overlap' if n_signals > 1 else parameters[0]['type'],
                'is_overlap': n_signals > 1,
                'n_signals': n_signals,
                'parameters': parameters,
                'priorities': priorities,  #  Proper SNR-based priorities!
                'detector_data': detector_data,
                'metadata': {
                    'sample_id': sample_id,
                    'generator': 'OverlappingSignalSimulator',
                    'mean_snr': float(np.mean(priorities)),
                    'max_snr': float(np.max(priorities))
                }
            }
        
        elif generator_type == 'gwtc':
            # Use GWTC real events
            if not hasattr(self, 'gwtc_loader'):
                from ahsd.data.gwtc_loader import GWTCLoader
                self.gwtc_loader = GWTCLoader()
                self.gwtc_events = self.gwtc_loader.get_gwtc_events()
            
            if event_name:
                event = self.gwtc_events[self.gwtc_events['event_name'] == event_name].iloc[0]
            else:
                quality_events = self.gwtc_events[
                    (self.gwtc_events['network_snr'] > 10) &
                    (self.gwtc_events['mass_1_source'] > 5)
                ]
                event = quality_events.sample(1).iloc[0] if len(quality_events) > 0 else self.gwtc_events.sample(1).iloc[0]
            
            params = {
                'mass_1': float(event['mass_1_source']),
                'mass_2': float(event['mass_2_source']),
                'luminosity_distance': float(event['luminosity_distance']),
                'target_snr': float(event['network_snr']),  # Real SNR from GWTC!
                'type': self._classify_signal_type(event['mass_1_source'], event['mass_2_source']),
                'event_name': event['event_name']
            }
            
            return {
                'sample_id': f'gwtc_{sample_id:06d}_{event["event_name"]}',
                'type': params['type'],
                'is_overlap': False,
                'n_signals': 1,
                'parameters': params,  # Dict for single
                'priorities': [params['target_snr']],  #  Real GWTC SNR!
                'detector_data': {},  # Implement strain download if needed
                'metadata': {
                    'sample_id': sample_id,
                    'generator': 'GWTC',
                    'event_name': event['event_name']
                }
            }
        
        else:
            # Fallback to your existing generation with SNR estimation
            sample = self._generate_single_sample(sample_id, **kwargs)
            
            #  FIX: Add priority calculation to existing samples
            if 'priorities' not in sample or all(p == 0 for p in sample.get('priorities', [])):
                priorities = []
                for params in sample.get('parameters', []):
                    detector_data = sample.get('detector_data', {})
                    priority = self._compute_snr_from_params(params, detector_data)
                    priorities.append(priority)
                    # Do not overwrite existing sampled SNR
                    if 'target_snr' not in params:
                        self._set_target_snr(params, priority, reason='fallback_generator_assign')
                
                sample['priorities'] = priorities
            
            return sample
        
 
    def _generate_sample_with_simulator(self, sample_id: int, n_signals: int = 1) -> Dict:
        """
        Generate sample using OverlappingSignalSimulator for realistic physics.
        
        Args:
            sample_id: Unique sample identifier
            n_signals: Number of signals to generate (1 for single, 2+ for overlap)
        
        Returns:
            Sample dict with proper structure and priorities
            
        Raises:
            ValueError: If simulator is not initialized
            RuntimeError: If signal generation fails
        """
        
        # ========================================================================
        # VALIDATION
        # ========================================================================
        if self.simulation is None:
            raise ValueError(
                f"Cannot generate sample {sample_id}: "
                f"OverlappingSignalSimulator not initialized. "
                f"Check __init__ and ensure simulation is set."
            )
        
        if n_signals < 1:
            raise ValueError(f"n_signals must be >= 1, got {n_signals}")
        
        try:
            # ====================================================================
            # GENERATE SCENARIO
            # ====================================================================
            scenario = self.simulation.generate_overlapping_scenario(n_signals)
            
            # Extract parameters from scenario
            parameters_list = scenario['signals']
            
            # ====================================================================
            # GENERATE DETECTOR NOISE
            # ====================================================================
            noise_data = self.simulation.generate_detector_noise(
                duration=self.duration,
                sampling_rate=self.sample_rate
            )
            
            # ====================================================================
            # INJECT SIGNALS INTO NOISE
            # ====================================================================
            #  FIXED: Pass entire scenario dict (API expects scenario, not signals)
            detector_data, signal_contributions = self.simulation.inject_signals_to_data(
                scenario=scenario,        #  Changed from signals=scenario['signals']
                noise_data=noise_data
            )
            
            # ====================================================================
            # CALCULATE PRIORITIES (SNRs)
            # ====================================================================
            priorities = []
            
            for signal in scenario['signals']:
                signal_id = signal['signal_id']
                max_snr = 0.0
                detector_snrs = {}
                
                # Proper nested dict access with validation
                for detector_name in self.detectors:
                    # Check if detector exists in signal_contributions
                    if detector_name not in signal_contributions:
                        self.logger.warning(
                            f"Detector {detector_name} not found in signal_contributions "
                            f"for sample {sample_id}, signal {signal_id}"
                        )
                        continue
                    
                    detector_signals = signal_contributions[detector_name]
                    
                    # Validate structure
                    if not isinstance(detector_signals, dict):
                        self.logger.warning(
                            f"signal_contributions[{detector_name}] is not a dict "
                            f"(got {type(detector_signals).__name__}) for sample {sample_id}"
                        )
                        continue
                    
                    # Check if this signal exists for this detector
                    if signal_id not in detector_signals:
                        self.logger.debug(
                            f"Signal {signal_id} not found in {detector_name} contributions "
                            f"for sample {sample_id}"
                        )
                        continue
                    
                    signal_strain = detector_signals[signal_id]
                    
                    # Validate strain data
                    if signal_strain is None or len(signal_strain) == 0:
                        self.logger.warning(
                            f"Empty signal strain for signal {signal_id} in {detector_name} "
                            f"for sample {sample_id}"
                        )
                        continue
                    
                    # Compute SNR from strain
                    try:
                        snr = self._compute_snr_from_strain(signal_strain, detector_name)
                        max_snr = max(max_snr, snr)
                        detector_snrs[detector_name] = float(snr)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to compute SNR for signal {signal_id} in {detector_name}: {e}"
                        )
                
                # Fallback if SNR computation failed for all detectors
                if max_snr < 5.0:
                    self.logger.warning(
                        f"SNR computation failed for signal {signal_id} in sample {sample_id}, "
                        f"using parameter-based estimate"
                    )
                    max_snr = self._estimate_snr_from_params(signal)
                
                # Clip and store priority
                # Diagnostic: if the raw computed SNR is above the clipping threshold,
                # log the per-detector contributions so we can trace the origin.
                if float(max_snr) > 80.0:
                    try:
                        self.logger.warning(
                            f"[SNR-ORIGIN] sample={sample_id} signal={signal_id} raw_max_snr={max_snr:.3f} "
                            f"detector_snrs={detector_snrs}"
                        )
                    except Exception:
                        self.logger.warning(f"[SNR-ORIGIN] sample={sample_id} signal={signal_id} raw_max_snr={max_snr}")

                priority = float(np.clip(max_snr, 7.0, 80.0))
                # Respect any pre-sampled value: do not overwrite
                if 'target_snr' not in signal:
                    # Use central helper so we get diagnostic traces for clipped values
                    try:
                        self._set_target_snr(signal, priority, reason='simulator_assign')
                    except Exception:
                        # Best-effort: fall back to direct assignment if helper is unavailable
                        signal['target_snr'] = float(priority)
                priorities.append(priority)
            
            # ====================================================================
            # CONSTRUCT SAMPLE
            # ====================================================================
            is_overlap = (n_signals > 1)
            sample_type = 'overlap' if is_overlap else scenario['signals'][0].get('type', 'BBH')
            
            sample = {
                'id': sample_id,
                'sample_id': f"{'overlap' if is_overlap else 'single'}_{sample_id:06d}",
                'type': sample_type,
                'is_overlap': is_overlap,
                'n_signals': n_signals,
                'parameters': parameters_list,
                'priorities': priorities,
                'detector_data': detector_data,
                'metadata': {
                    'sample_id': sample_id,
                    'n_signals': n_signals,
                    'mean_snr': float(np.mean(priorities)) if priorities else 0.0,
                    'max_snr': float(np.max(priorities)) if priorities else 0.0,
                    'generator': 'simulator',
                    'scenario_type': scenario.get('scenario_type', 'unknown')
                }
            }
            
            return sample
            
        except Exception as e:
            self.logger.error(
                f"Failed to generate sample {sample_id} with simulator "
                f"(n_signals={n_signals}): {e}",
                exc_info=True
            )
            raise RuntimeError(
                f"Simulator generation failed for sample {sample_id}: {e}"
            ) from e

    def _classify_signal_type(self, mass_1: float, mass_2: float) -> str:
        """
        Classify signal type based on component masses.
        BBH: Both masses > 3 Msun
        BNS: Both masses < 3 Msun
        NSBH: One mass > 3, other < 3
        """
        NEUTRON_STAR_MAX_MASS = 3.0
        
        if mass_1 > NEUTRON_STAR_MAX_MASS and mass_2 > NEUTRON_STAR_MAX_MASS:
            return 'BBH'
        elif mass_1 < NEUTRON_STAR_MAX_MASS and mass_2 < NEUTRON_STAR_MAX_MASS:
            return 'BNS'
        else:
            return 'NSBH'
    
    def _compute_snr_from_strain(self, strain: np.ndarray, detector_name: str) -> float:
        """
        Compute optimal SNR from signal strain and detector PSD.
        
        Args:
            strain: Signal strain array
            detector_name: Name of detector (H1, L1, V1)
        
        Returns:
            Optimal matched-filter SNR
        """
        
        try:
            # Get PSD for this detector
            psd_dict = self.psds[detector_name]
            psd = psd_dict['psd']
            frequencies = psd_dict['frequencies']
            
            # FFT of strain (normalize by N to get proper Fourier amplitudes)
            # Note: failing to normalize by N inflates |~h|^2 and thus the SNR by a factor ~N^2.
            N = len(strain)
            strain_fft = np.fft.rfft(strain) / float(N)
            freq_array = np.fft.rfftfreq(len(strain), d=1.0/self.sample_rate)
            
            # Interpolate PSD to match frequency array
            psd_interp = np.interp(freq_array, frequencies, psd)
            
            # Avoid division by zero
            psd_interp[psd_interp == 0] = np.inf
            
            # Compute optimal SNR (matched-filter formula)
            integrand = np.abs(strain_fft)**2 / psd_interp
            
            # Integrate (using trapezoidal rule)
            df = freq_array[1] - freq_array[0]
            snr_squared = 4.0 * np.sum(integrand) * df
            
            snr = np.sqrt(max(snr_squared, 0.0))
            
            return float(snr)
            
        except Exception as e:
            self.logger.warning(f"SNR computation failed for {detector_name}: {e}")
            return 0.0



# # ================================================================================
# IO UTILS FOR DATASET WRITING
# # ================================================================================

class DatasetWriter:
    """
    Write datasets to HDF5 or PKL format with comprehensive metadata
    """
    
    def __init__(self, output_dir: str, format: str = 'hdf5'):
        """
        Args:
            output_dir: Output directory path
            format: 'hdf5', 'pkl', or 'pkl_compressed'
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format.lower()
        self.logger = logging.getLogger(__name__)
    
    def save_batch_pkl(self,
                      batch_id: int,
                      samples: List[Dict],
                      metadata: Dict = None,
                      compress: bool = False) -> Path:
        """
        Save batch of samples to pickle file
        
        Args:
            batch_id: Batch identifier
            samples: List of sample dictionaries
            metadata: Metadata dictionary
            compress: Use gzip compression
            
        Returns:
            Path to saved file
        """
        
        if compress:
            filename = f"batch_{batch_id:05d}.pkl.gz"
        else:
            filename = f"batch_{batch_id:05d}.pkl"
        
        batch_dir = self.output_dir / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        filepath = batch_dir / filename
        
        try:
            # Create batch data structure
            batch_data = {
                'metadata': metadata or {},
                'samples': samples,
                'batch_id': batch_id,
                'n_samples': len(samples)
            }
            
            # Add metadata fields
            if metadata:
                batch_data['metadata']['batch_id'] = batch_id
                batch_data['metadata']['n_samples'] = len(samples)
            
            # Save with or without compression
            if compress:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.debug(f"✓ Batch {batch_id} saved to PKL ({len(samples)} samples)")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save PKL batch {batch_id}: {e}")
            raise
    
    def save_complete_dataset_pkl(self,
                                  filename: str,
                                  all_samples: List[Dict],
                                  metadata: Dict = None,
                                  compress: bool = True) -> Path:
        """
        Save entire dataset to single pickle file
        
        Args:
            filename: Output filename
            all_samples: List of all samples
            metadata: Dataset metadata
            compress: Use gzip compression
            
        Returns:
            Path to saved file
        """
        
        if compress and not filename.endswith('.gz'):
            if not filename.endswith('.pkl'):
                filename = filename + '.pkl.gz'
            else:
                filename = filename + '.gz'
        elif not compress and not filename.endswith('.pkl'):
            filename = filename + '.pkl'
        
        filepath = self.output_dir / filename
        
        try:
            
            # Ensure metadata has format_version
            if metadata is None:
                metadata = {}
            
            if 'format_version' not in metadata:
                metadata['format_version'] = '1.0.0'
            
            if 'creation_time' not in metadata:
                from datetime import datetime
                metadata['creation_time'] = datetime.now().isoformat()
                
                
            dataset = {
                'metadata': metadata or {},
                'samples': all_samples,
                'n_samples': len(all_samples),
                'format_version': '1.0.0'
            }
            
            # Add statistics
            from .io_utils import MetadataManager
            meta_manager = MetadataManager()
            stats = meta_manager.compute_dataset_statistics(all_samples)
            dataset['statistics'] = stats
            
            # Save
            if compress:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                self.logger.info(f"✓ Complete dataset saved to {filepath} (compressed)")
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                self.logger.info(f"✓ Complete dataset saved to {filepath}")
            
            # Get file size
            size_mb = filepath.stat().st_size / (1024 * 1024)
            self.logger.info(f"  File size: {size_mb:.2f} MB")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save complete dataset: {e}")
            raise
    
    def save_batch_hdf5(self,
                       batch_id: int,
                       samples: List[Dict],
                       metadata: Dict = None) -> Path:
        """Save batch of samples to HDF5"""
        
        filename = f"batch_{batch_id:05d}.h5"
        filepath = self.output_dir / filename
        
        try:
            with h5py.File(filepath, 'w') as f:
                # Metadata
                if metadata:
                    meta_group = f.create_group('metadata')
                    self._write_dict_to_group(meta_group, metadata)
                    meta_group.attrs['n_samples'] = len(samples)
                
                # Samples
                samples_group = f.create_group('samples')
                
                for i, sample in enumerate(samples):
                    sample_group = samples_group.create_group(f'sample_{i:05d}')
                    
                    # Sample data
                    for key, value in sample.items():
                        if isinstance(value, np.ndarray):
                            sample_group.create_dataset(key, data=value, compression='gzip')
                        elif isinstance(value, dict):
                            subgroup = sample_group.create_group(key)
                            self._write_dict_to_group(subgroup, value)
                        elif isinstance(value, (int, float, str, bool)):
                            sample_group.attrs[key] = value
            
            self.logger.debug(f"✓ Batch {batch_id} saved to HDF5 ({len(samples)} samples)")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save HDF5 batch {batch_id}: {e}")
            raise
    
    def _write_dict_to_group(self, group: h5py.Group, data_dict: Dict):
        """Recursively write dictionary to HDF5 group"""
        
        for key, value in data_dict.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._write_dict_to_group(subgroup, value)
            elif isinstance(value, (np.ndarray, list)):
                group.create_dataset(key, data=value)
            elif isinstance(value, (int, float, str, bool, type(None))):
                group.attrs[key] = value if value is not None else 'None'
            elif isinstance(value, (tuple, set)):
                group.attrs[key] = str(value)
    
    def save_pickle(self, filename: str, data: Any, compress: bool = False) -> Path:
        """Save arbitrary data to pickle file"""
        
        if compress and not filename.endswith('.gz'):
            filename = filename + '.gz'
        
        filepath = self.output_dir / filename
        
        try:
            if compress:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f"✓ Saved pickle to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save pickle: {e}")
            raise
    
    def save_json(self, filename: str, data: Dict) -> Path:
        """Save data to JSON file"""
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"✓ Saved JSON to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON: {e}")
            raise
    
    def save_yaml(self, filename: str, data: Dict) -> Path:
        """Save data to YAML file"""
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            self.logger.info(f"✓ Saved YAML to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save YAML: {e}")
            raise

    def save_split_chunks(self,
                     split_name: str,
                     samples: List[Dict],
                     metadata: Dict = None,
                     chunk_size: int = 100,
                     compress: bool = True) -> Path:
        """
        Save dataset split in chunks with comprehensive metadata.
        
        Args:
            split_name: 'train', 'validation', or 'test'
            samples: List of samples for this split
            metadata: Split metadata
            chunk_size: Samples per chunk file
            compress: Use gzip compression
            
        Returns:
            Path to split directory
        """
        
        split_dir = self.output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        n_chunks = (len(samples) + chunk_size - 1) // chunk_size
        
        self.logger.info(f"Saving {split_name} split: {len(samples)} samples in {n_chunks} chunks")
        
        # Save chunks
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(samples))
            chunk_samples = samples[start_idx:end_idx]
            
            # Save chunk
            if compress:
                chunk_file = split_dir / f'chunk_{chunk_idx:04d}.pkl.gz'
                with gzip.open(chunk_file, 'wb') as f:
                    pickle.dump(chunk_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                chunk_file = split_dir / f'chunk_{chunk_idx:04d}.pkl'
                with open(chunk_file, 'wb') as f:
                    pickle.dump(chunk_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.debug(f"  Chunk {chunk_idx+1}/{n_chunks}: {len(chunk_samples)} samples")
        
        # Prepare comprehensive metadata
        split_metadata = {
            'split': split_name,  
            'n_samples': len(samples),
            'n_chunks': n_chunks,
            'chunk_size': chunk_size,
            'compressed': compress,
            'file_pattern': 'chunk_XXXX.pkl.gz' if compress else 'chunk_XXXX.pkl',
            'created_at': datetime.now(timezone.utc).isoformat(),  #  UTC ISO format
            'chunk_files': [f'chunk_{i:04d}.pkl' + ('.gz' if compress else '') 
                        for i in range(n_chunks)]
        }
        
        if metadata:
            split_metadata.update(metadata)
        
        json_file = split_dir / 'split_info.json'
        with open(json_file, 'w') as f:
            json.dump(split_metadata, f, indent=2, default=str)
        
        pkl_file = split_dir / f'{split_name}_metadata.pkl'
        with open(pkl_file, 'wb') as f:
            pickle.dump(split_metadata, f)
        
        self.logger.info(f"✓ {split_name} split saved ({n_chunks} chunks)")
        self.logger.info(f"  - Metadata: {json_file.name}, {pkl_file.name}")
        
        return split_dir

class DatasetReader:
    """
    Read datasets from various formats including pickle
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_pkl(self, filepath: str, compressed: bool = None) -> Dict:
        """
        Load dataset from pickle file
        
        Args:
            filepath: Path to pickle file
            compressed: True if gzipped, None for auto-detect
            
        Returns:
            Dataset dictionary
        """
        
        filepath = Path(filepath)
        
        # Auto-detect compression
        if compressed is None:
            compressed = filepath.suffix == '.gz' or str(filepath).endswith('.pkl.gz')
        
        try:
            if compressed:
                with gzip.open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            
            self.logger.info(f"✓ Loaded from {filepath}")
            
            # Log dataset info
            if isinstance(data, dict):
                if 'n_samples' in data:
                    self.logger.info(f"  Samples: {data['n_samples']}")
                if 'samples' in data:
                    self.logger.info(f"  Samples: {len(data['samples'])}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load pickle: {e}")
            raise
    
    def load_batch_pkl(self, filepath: str) -> Dict:
        """Load batch from pickle file"""
        return self.load_pkl(filepath)
    
    def load_hdf5(self, filepath: str) -> Dict:
        """Load dataset from HDF5 file"""
        
        try:
            with h5py.File(filepath, 'r') as f:
                data = {}
                
                # Load metadata
                if 'metadata' in f:
                    data['metadata'] = self._read_group_to_dict(f['metadata'])
                
                # Load data
                if 'data' in f:
                    data['data'] = self._read_group_to_dict(f['data'])
                
                # Load samples if present
                if 'samples' in f:
                    data['samples'] = self._read_samples_group(f['samples'])
            
            self.logger.info(f"✓ Loaded from {filepath}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load HDF5: {e}")
            raise
    
    def _read_group_to_dict(self, group: h5py.Group) -> Dict:
        """Recursively read HDF5 group to dictionary"""
        
        result = {}
        
        # Read attributes
        for key, value in group.attrs.items():
            if value == 'None':
                result[key] = None
            else:
                result[key] = value
        
        # Read datasets and subgroups
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                result[key] = np.array(item)
            elif isinstance(item, h5py.Group):
                result[key] = self._read_group_to_dict(item)
        
        return result
    
    def _read_samples_group(self, samples_group: h5py.Group) -> List[Dict]:
        """Read samples group to list of dicts"""
        
        samples = []
        
        for sample_key in sorted(samples_group.keys()):
            sample_group = samples_group[sample_key]
            sample = self._read_group_to_dict(sample_group)
            samples.append(sample)
        
        return samples
    
    def load_pickle(self, filepath: str) -> Any:
        """Load arbitrary data from pickle file"""
        return self.load_pkl(filepath)
    
    def load_json(self, filepath: str) -> Dict:
        """Load data from JSON file"""
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.logger.info(f"✓ Loaded JSON from {filepath}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON: {e}")
            raise
    
    def load_yaml(self, filepath: str) -> Dict:
        """Load data from YAML file"""
        
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            self.logger.info(f"✓ Loaded YAML from {filepath}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load YAML: {e}")
            raise

    def load_split_chunks(self, split_dir: str) -> List[Dict]:
        """
        Load all chunks from a split directory
        
        Args:
            split_dir: Path to split directory (e.g., 'data/dataset/train')
            
        Returns:
            List of all samples
        """
        
        split_path = Path(split_dir)
        
        # Load metadata
        metadata_file = split_path / 'split_info.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.logger.info(f"Loading {metadata['n_samples']} samples from {metadata['n_chunks']} chunks")
        
        # Find all chunk files
        chunk_files = sorted(split_path.glob('chunk_*.pkl*'))
        
        all_samples = []
        for chunk_file in chunk_files:
            samples = self.load_pkl(str(chunk_file))
            all_samples.extend(samples)
        
        self.logger.info(f"✓ Loaded {len(all_samples)} samples from {split_path.name}")
        
        return all_samples


class MetadataManager:
    """
    Manage dataset metadata and statistics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def create_metadata(self,
                   n_samples: int,
                   config: Dict,
                   statistics: Dict = None) -> Dict:
        """
        Create comprehensive dataset metadata dictionary.
        
        Includes creation timestamp (UTC), configuration, version info,
        system details, and optional statistics.
        
        Args:
            n_samples: Total number of samples in dataset
            config: Generation configuration dictionary
            statistics: Optional statistics dict (sample counts, distributions, etc.)
        
        Returns:
            Metadata dictionary with all relevant dataset information
        
        Example:
            >>> metadata = gen.create_metadata(
            ...     n_samples=50000,
            ...     config={'duration': 4.0, 'sample_rate': 4096},
            ...     statistics={'mean_snr': 18.5, 'n_overlaps': 15000}
            ... )
        """
        
        from datetime import datetime, timezone
        import sys
        import platform
        
        # ========================================================================
        # CORE METADATA
        # ========================================================================
        metadata = {
            # Timestamps
            'creation_time': datetime.now(timezone.utc).isoformat(),  #  UTC ISO format
            'creation_timestamp_utc': datetime.now(timezone.utc).timestamp(),  # Unix timestamp
            
            # Dataset info
            'n_samples': n_samples,
            'format': 'AHSD-GW-Dataset',
            'format_version': '1.0.0',
            'schema_version': '1.0.0',
            
            # Configuration
            'configuration': config.copy() if config else {},
            
            # Generator info
            'generator': {
                'name': 'GWDatasetGenerator',
                'version': '1.0.0',
                'class': self.__class__.__name__
            }
        }
        
        # ========================================================================
        # SYSTEM INFORMATION (for reproducibility)
        # ========================================================================
        metadata['system_info'] = {
            'platform': sys.platform,
            'platform_version': platform.platform(),
            'python_version': sys.version.split()[0],
            'python_implementation': platform.python_implementation(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
        
        # ========================================================================
        # GIT INFORMATION (optional, for reproducibility)
        # ========================================================================
        git_commit = self._get_git_commit()
        if git_commit:
            metadata['git_info'] = {
                'commit': git_commit,
                'branch': self._get_git_branch(),
                'dirty': self._is_git_dirty()
            }
        
        # ========================================================================
        # DETECTOR CONFIGURATION
        # ========================================================================
        metadata['detector_config'] = {
            'detectors': self.detectors,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'n_samples_per_segment': int(self.duration * self.sample_rate)
        }
        
        # ========================================================================
        # STATISTICS (optional)
        # ========================================================================
        if statistics:
            metadata['statistics'] = statistics
        
        # ========================================================================
        # PACKAGE VERSIONS (for reproducibility)
        # ========================================================================
        metadata['dependencies'] = self._get_package_versions()
        
        return metadata


    # ========================================================================
    # HELPER METHODS FOR METADATA
    # ========================================================================

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            return result.stdout.strip()
        except Exception:
            return None


    def _get_git_branch(self) -> Optional[str]:
        """Get current git branch name."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            return result.stdout.strip()
        except Exception:
            return None


    def _is_git_dirty(self) -> bool:
        """Check if git working directory has uncommitted changes."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            return len(result.stdout.strip()) > 0
        except Exception:
            return False


    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key dependencies."""
        versions = {}
        
        packages = [
            'numpy',
            'scipy',
            'pycbc',
            'lalsuite',
            'h5py',
            'torch',
            'gwpy'
        ]
        
        for package in packages:
            try:
                import importlib
                mod = importlib.import_module(package)
                versions[package] = getattr(mod, '__version__', 'unknown')
            except ImportError:
                versions[package] = 'not_installed'
            except Exception:
                versions[package] = 'unknown'
        
        return versions


    def compute_dataset_statistics(self, samples: List[Dict]) -> Dict:
        """Compute comprehensive dataset statistics"""
        
        stats = {
            'n_total': len(samples),
            'event_types': {},
            'snr_distribution': {},
            'overlap_statistics': {},
            'detector_coverage': {}
        }
        
        # Event type distribution
        event_types = [s.get('type', 'unknown') for s in samples]
        for et in set(event_types):
            stats['event_types'][et] = event_types.count(et)
        
        # SNR statistics
        snrs = [s.get('target_snr', 0) for s in samples if 'target_snr' in s]
        if snrs:
            stats['snr_distribution'] = {
                'mean': float(np.mean(snrs)),
                'std': float(np.std(snrs)),
                'min': float(np.min(snrs)),
                'max': float(np.max(snrs)),
                'median': float(np.median(snrs))
            }
        
        # Overlap statistics
        n_overlaps = sum(1 for s in samples if s.get('is_overlap', False))
        stats['overlap_statistics'] = {
            'n_overlapping': n_overlaps,
            'n_single': len(samples) - n_overlaps,
            'overlap_fraction': n_overlaps / len(samples) if samples else 0
        }
        
        return stats


# ================================================================================
# MAIN EXECUTION BLOCK - CONFIG-FILE DRIVEN
# ================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate gravitational wave dataset for AHSD training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file argument
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration YAML file (required)')

    args = parser.parse_args()

    # Load configuration file
    if not args.config:
        logger.error("Configuration file is required!")
        logger.error("Usage: python combined_gw_dataset_generator.py --config ahsd_config.yaml")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # Load YAML config
    try:
        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        sys.exit(1)

    # Set random seed if provided
    if config.get('random_seed'):
        seed = config['random_seed']
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f"Random seed set to: {seed}")

    # Print configuration
    logger.info("="*80)
    logger.info("GRAVITATIONAL WAVE DATASET GENERATOR")
    logger.info("AHSD Pipeline - Configuration-Driven")
    logger.info("="*80)
    logger.info("Configuration from file:")
    logger.info(f"  Number of Samples: {config.get('n_samples', 'N/A')}")
    logger.info(f"  Output Directory: {config.get('output_dir', 'N/A')}")
    logger.info(f"  Sample Rate: {config.get('sample_rate', 'N/A')} Hz")
    logger.info(f"  Duration: {config.get('duration', 'N/A')} s")
    logger.info(f"  Detectors: {config.get('detectors', ['N/A'])}")
    logger.info(f"  Overlap Fraction: {config.get('overlap_fraction', 'N/A')}")
    logger.info(f"  Edge Case Fraction: {config.get('edge_case_fraction', 'N/A')}")
    logger.info(f"  Save Batch Size: {config.get('save_batch_size', 'N/A')}")
    logger.info(f"  Noise Augmentation: {config.get('noise_augmentation_k', 'N/A')}x")
    logger.info("")
    logger.info("Dependencies:")
    logger.info(f"  PyCBC: {'✓ Available' if PYCBC_AVAILABLE else '✗ Not available (using fallbacks)'}")
    logger.info(f"  GWpy: {'✓ Available' if GWPY_AVAILABLE else '✗ Not available'}")
    logger.info(f"  Pandas: {'✓ Available' if PANDAS_AVAILABLE else '✗ Not available'}")
    logger.info(f"  Requests: {'✓ Available' if REQUESTS_AVAILABLE else '✗ Not available'}")
    logger.info(f"  Bilby: {'✓ Available' if BILBY_AVAILABLE else '✗ Not available'}")
    logger.info(f"  tqdm: {'✓ Available' if TQDM_AVAILABLE else '✗ Not available'}")
    logger.info(f"  NUMBA: {'✓ Available' if NUMBA_AVAILABLE else '✗ Not available'}")
    logger.info("="*80)

    try:
        # Initialize generator
        logger.info("Initializing dataset generator...")
        generator = GWDatasetGenerator(
            output_dir=config['output_dir'],
            sample_rate=config['sample_rate'],
            duration=config['duration'],
            detectors=config['detectors'],
            output_format=config.get('output_format', 'pkl'),
            config=config
        )
        logger.info("Generator initialized successfully!")

        # Generate dataset
        logger.info("")
        logger.info("Starting dataset generation...")
        start_time = time.time()

        summary = generator.generate_dataset(
            n_samples=config['n_samples'],
            overlap_fraction=config['overlap_fraction'],
            edge_case_fraction=config['edge_case_fraction'],
            save_batch_size=config['save_batch_size'],
            add_glitches=config['add_glitches'],
            preprocess=config['preprocess'],
            save_complete=config['save_complete'],
            create_splits=config['create_splits'],
            train_frac=config['train_frac'],
            val_frac=config['val_frac'],
            test_frac=config['test_frac'],
            chunk_size=config['chunk_size'],
            noise_augmentation_k=config['noise_augmentation_k'],
            config=config
        )

        elapsed = time.time() - start_time

        # Print summary
        logger.info("")
        logger.info("="*80)
        logger.info(" GENERATION COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        logger.info(f"Samples Generated: {summary.get('n_samples', 'N/A')}")
        if config['n_samples'] > 0:
            logger.info(f"Average Time per Sample: {elapsed/config['n_samples']:.3f} seconds")
        logger.info("")
        logger.info("Dataset Statistics:")
        stats = summary.get('statistics', {})
        logger.info(f"  Event Types: {stats.get('event_types', {})}")
        logger.info(f"  SNR Regimes: {stats.get('snr_regimes', {})}")
        logger.info(f"  Edge Cases: {stats.get('edge_cases', {})}")
        logger.info(f"  Extreme Cases: {stats.get('extreme_cases', {})}")
        logger.info("")
        logger.info(f"Output Location: {config['output_dir']}")
        logger.info("="*80)

        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("⚠ Generation interrupted by user (Ctrl+C)")
        logger.warning("Partial results may be saved in output directory")
        sys.exit(1)

    except Exception as e:
        logger.error("")
        logger.error("="*80)
        logger.error("❌ GENERATION FAILED!")
        logger.error("="*80)
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("")
        logger.error("Please check:")
        logger.error("  1. Configuration file is valid YAML")
        logger.error("  2. All required config parameters are present")
        logger.error("  3. All required dependencies are installed")
        logger.error("  4. Output directory exists and is writable")
        logger.error("  5. Sufficient disk space is available")
        logger.error("="*80)
        sys.exit(1)
