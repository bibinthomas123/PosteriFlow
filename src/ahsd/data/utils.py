"""
Utility functions for AHSD data module
Cosmology calculations, SNR computation, effective spin, etc.
"""

import numpy as np
import logging
from typing import Optional

from .config import COSMO_H0, COSMO_OMEGA_M, COSMO_OMEGA_LAMBDA

logger = logging.getLogger(__name__)

def calculate_redshift(luminosity_distance: float) -> Optional[float]:
    """
    Calculate redshift from luminosity distance using ΛCDM cosmology
    
    Args:
        luminosity_distance: Luminosity distance in Mpc
        
    Returns:
        Redshift z, or None if calculation fails
    """
    try:
        from scipy.integrate import quad
        from scipy.optimize import brentq
        
        def E(z):
            return np.sqrt(COSMO_OMEGA_M * (1 + z)**3 + COSMO_OMEGA_LAMBDA)
        
        def d_L_calc(z):
            if z <= 0:
                return 0.0
            integral, _ = quad(lambda zp: 1.0 / E(zp), 0, z)
            return (1 + z) * (299792.458 / COSMO_H0) * integral
        
        def residual(z):
            return d_L_calc(z) - luminosity_distance
        
        # Find redshift
        z = brentq(residual, 0.0, 10.0)
        return float(z)
        
    except Exception as e:
        logger.debug(f"Redshift calculation failed: {e}")
        # Hubble approximation for small z
        z_approx = luminosity_distance * COSMO_H0 / 299792.458
        return float(np.clip(z_approx, 0.0, 10.0))

def calculate_comoving_distance(z: float) -> float:
    """Calculate comoving distance from redshift"""
    try:
        from scipy.integrate import quad
        
        def E(zp):
            return np.sqrt(COSMO_OMEGA_M * (1 + zp)**3 + COSMO_OMEGA_LAMBDA)
        
        integral, _ = quad(lambda zp: 1.0 / E(zp), 0, z)
        d_C = (299792.458 / COSMO_H0) * integral
        return float(d_C)
        
    except Exception:
        # Approximation
        return float(z * 299792.458 / COSMO_H0)

def compute_effective_spin(m1: float, m2: float, 
                          a1: float, a2: float, 
                          tilt1: float, tilt2: float) -> float:
    """Compute effective inspiral spin parameter χ_eff"""
    total_mass = m1 + m2
    chi_eff = (m1 * a1 * np.cos(tilt1) + m2 * a2 * np.cos(tilt2)) / total_mass
    return float(chi_eff)
