"""
Parameter Sampler for GW Signals
Implements astrophysically realistic parameter distributions
"""

import numpy as np
import logging
from typing import Dict,List
from scipy.stats import truncnorm, beta

from .config import MASS_RANGES, DISTANCE_RANGES, SNR_RANGES
from .utils import calculate_redshift, calculate_comoving_distance, compute_effective_spin

class ParameterSampler:
    """
    Sample astrophysically realistic GW parameters
    Based on GWTC-3 population models with independent sampling to avoid correlations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mass_ranges = MASS_RANGES
        self.distance_ranges = DISTANCE_RANGES
        self.snr_ranges = SNR_RANGES
        
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
    def sample_bbh_parameters(self, snr_regime: str, is_edge_case: bool = False) -> Dict:
        """
        Generate complete BBH parameters with INDEPENDENT mass and distance sampling
        Implements decorrelation strategies to avoid parameter degeneracies
        """
        
        # Mass sampling with decorrelation
        if is_edge_case and np.random.random() < 0.3:
            mass_1 = np.random.uniform(60, 100)
            mass_2 = np.random.uniform(50, mass_1)
            edge_case_type = 'short_bbh'
        elif is_edge_case and np.random.random() < 0.5:
            mass_1 = np.random.uniform(30, 80)
            q = np.random.uniform(0.05, 0.15)
            mass_2 = mass_1 * q
            edge_case_type = 'extreme_mass_ratio'
        else:
            # ✅ Sample with DIFFERENT widths to decorrelate
            m1_raw = np.clip(np.random.lognormal(mean=np.log(25.0), sigma=0.35), 5.0, 80.0)
            m2_raw = np.clip(np.random.lognormal(mean=np.log(20.0), sigma=0.40), 5.0, 80.0)
            
            # Enforce minimum mass ratio
            q_min = 0.1
            m2_raw = max(m2_raw, q_min * m1_raw)
            
            # Order by convention
            mass_1, mass_2 = (m1_raw, m2_raw) if m1_raw >= m2_raw else (m2_raw, m1_raw)
            
            # ✅ Add jitter to break deterministic pairing
            mass_1 += np.random.uniform(-0.05, 0.05)
            mass_2 += np.random.uniform(-0.05, 0.05)
            
            # Clip to valid range
            mass_1 = float(np.clip(mass_1, 5.0, 100.0))
            mass_2 = float(np.clip(mass_2, 5.0, min(100.0, mass_1)))
            
            edge_case_type = 'none'
        
        # Distance sampling - INDEPENDENT (uniform in volume)
        d_min, d_max = self.distance_ranges['BBH']
        u_d = np.random.random()
        luminosity_distance = (d_min**3 + u_d * (d_max**3 - d_min**3))**(1/3)
        
        # Spin sampling - INDEPENDENT
        a1 = float(np.clip(np.random.beta(2, 5), 0, 0.99))
        a2 = float(np.clip(np.random.beta(2, 5), 0, 0.99))
        
        cos_tilt1 = np.random.uniform(-1.0, 1.0)
        cos_tilt2 = np.random.uniform(-1.0, 1.0)
        tilt1 = float(np.arccos(cos_tilt1))
        tilt2 = float(np.arccos(cos_tilt2))
        
        phi12 = float(np.random.uniform(0, 2*np.pi))
        phi_jl = float(np.random.uniform(0, 2*np.pi))
        
        # Sky location - isotropic
        ra = np.random.uniform(0, 2*np.pi)
        dec = np.arcsin(np.random.uniform(-1, 1))
        theta_jn = np.arccos(np.random.uniform(-1, 1))
        psi = np.random.uniform(0, np.pi)
        phase = np.random.uniform(0, 2*np.pi)
        geocent_time = np.random.uniform(-0.1, 0.1)
        
        # Approximant
        approximant = 'IMRPhenomD'
        
        # Derived quantities
        total_mass = mass_1 + mass_2
        chirp_mass = (mass_1 * mass_2)**(3/5) / total_mass**(1/5)
        mass_ratio = mass_2 / mass_1
        symmetric_mass_ratio = (mass_1 * mass_2) / total_mass**2
        
        # Target SNR
        snr_min, snr_max = self.snr_ranges[snr_regime]
        target_snr = np.random.uniform(snr_min, snr_max)
        
        # Cosmology
        z = calculate_redshift(luminosity_distance)
        d_C = calculate_comoving_distance(z) if z is not None else luminosity_distance/(1+z)
        
        return {
            'name': f'BBH_{int(mass_1)}_{int(mass_2)}',
            'type': 'BBH',
            'mass_1': float(mass_1),
            'mass_2': float(mass_2),
            'total_mass': float(total_mass),
            'chirp_mass': float(chirp_mass),
            'mass_ratio': float(mass_ratio),
            'symmetric_mass_ratio': float(symmetric_mass_ratio),
            'a1': float(a1),
            'a2': float(a2),
            'tilt1': float(tilt1),
            'tilt2': float(tilt2),
            'effective_spin': compute_effective_spin(mass_1, mass_2, a1, a2, tilt1, tilt2),
            'phi12': float(phi12),
            'phi_jl': float(phi_jl),
            'luminosity_distance': float(luminosity_distance),
            'redshift': float(z),
            'comoving_distance': float(d_C),
            'ra': float(ra),
            'dec': float(dec),
            'theta_jn': float(theta_jn),
            'psi': float(psi),
            'phase': float(phase),
            'geocent_time': float(geocent_time),
            'f_lower': 20.0,
            'f_ref': 20.0,
            'approximant': approximant,
            'target_snr': float(target_snr),
            'lambda_1': 0.0,
            'lambda_2': 0.0,
            'is_real_event': False,
            'edge_case': is_edge_case,
            'edge_case_type': edge_case_type
        }
        
    def sample_bns_parameters(self, snr_regime: str, is_edge_case: bool = False) -> Dict:
        """Generate BNS parameters with  mass correlation."""
        
        if is_edge_case:
            f_lower = np.random.uniform(25.0, 30.0)
            edge_type = 'long_bns_inspiral'
        else:
            f_lower = 35.0
            edge_type = None
        
        #  FIX: Sample independently with different widths + jitter
        m1_raw = np.clip(np.random.normal(1.40, 0.15), 1.0, 2.5)
        m2_raw = np.clip(np.random.normal(1.40, 0.20), 1.0, 2.5)  # wider Ïƒ
        
        # Enforce ordering
        mass_1, mass_2 = (m1_raw, m2_raw) if m1_raw >= m2_raw else (m2_raw, m1_raw)
        
        #  FIX: Tiny jitter to break determinism
        mass_1 += np.random.uniform(-0.01, 0.01)
        mass_2 += np.random.uniform(-0.01, 0.01)
        
        # Clip to valid range
        mass_1 = float(np.clip(mass_1, 1.0, 2.5))
        mass_2 = float(np.clip(mass_2, 1.0, min(2.5, mass_1)))
        
        total_mass = mass_1 + mass_2
        chirp_mass = (mass_1 * mass_2)**(3/5) / total_mass**(1/5)
        
        # Distance INDEPENDENT of mass (uniform in volume)
        d_min, d_max = 10.0, 300.0
        u_d = np.random.random()
        luminosity_distance = float((d_min**3 + u_d * (d_max**3 - d_min**3))**(1/3))
        
        # Target SNR
        snr_min, snr_max = self.snr_ranges[snr_regime]
        target_snr = np.random.uniform(snr_min, snr_max)
        
        # Tidal parameters
        lambda_1 = np.clip(np.random.lognormal(np.log(400), 0.7) * (1.4/mass_1)**5, 50, 5000)
        lambda_2 = np.clip(np.random.lognormal(np.log(400), 0.7) * (1.4/mass_2)**5, 50, 5000)
        
        # Low spins
        a1, a2 = np.random.uniform(0.0, 0.05, 2)
        tilt1, tilt2 = 0.0, 0.0
        
        # Isotropic inclination
        cos_theta_jn = np.random.uniform(-1, 1)
        theta_jn = float(np.arccos(cos_theta_jn))
        
        # Sky location
        ra = np.random.uniform(0, 2*np.pi)
        dec = np.arcsin(np.random.uniform(-1, 1))
        psi = np.random.uniform(0, np.pi)
        phase = np.random.uniform(0, 2*np.pi)
        
        # Cosmology
        z = calculate_redshift(luminosity_distance)
        d_C = calculate_comoving_distance(z) if z else luminosity_distance/(1+z)
        
        return {
            'mass_1': float(mass_1),
            'mass_2': float(mass_2),
            'total_mass': float(total_mass),
            'chirp_mass': float(chirp_mass),
            'mass_ratio': float(mass_2/mass_1),
            'symmetric_mass_ratio': float((mass_1 * mass_2) / (total_mass ** 2)),
            'luminosity_distance': float(luminosity_distance),
            'redshift': float(z),
            'comoving_distance': float(d_C),
            'target_snr': float(target_snr),
            'a1': float(a1),
            'a2': float(a2),
            'tilt1': float(tilt1),
            'tilt2': float(tilt2),
            'effective_spin': compute_effective_spin(mass_1, mass_2, a1, a2, tilt1, tilt2),
            'ra': float(ra), 'dec': float(dec),
            'theta_jn': float(theta_jn),
            'psi': float(psi),
            'phase': float(phase),
            'f_lower': f_lower,
            'f_ref': 50.0,
            'approximant': 'IMRPhenomD_NRTidal',
            'lambda_1': float(lambda_1),
            'lambda_2': float(lambda_2),
            'lambda_tilde': float((16/13) * ((mass_1+12*mass_2)*mass_1**4*lambda_1 + 
                                            (mass_2+12*mass_1)*mass_2**4*lambda_2) / total_mass**5),
            'edge_case': is_edge_case,
            'edge_case_type': edge_type,
            'type': 'BNS',
            'geocent_time': float(np.random.uniform(-0.1, 0.1))
        }

    def sample_nsbh_parameters(self, snr_regime: str, is_edge_case: bool = False) -> Dict:
        """Generate complete NSBH parameters with mass-aware approximant selection"""
        
        # Neutron star mass
        ns_mass = np.random.uniform(1.2, 2.0)
        
        # Black hole mass with diversity
        if is_edge_case:
            edge_type = 'extreme_mass'
            self.stats['edge_cases']['extreme_mass'] += 1
            bh_mass = np.random.uniform(50.0, 100.0)
        else:
            edge_type = None
            bh_mass_type = np.random.choice(['light', 'medium', 'heavy'])
            
            if bh_mass_type == 'light':
                bh_mass = np.random.uniform(3.0, 8.0)
            elif bh_mass_type == 'medium':
                bh_mass = np.random.uniform(8.0, 25.0)
            else:  # heavy
                bh_mass = np.random.uniform(25.0, 50.0)
        
        mass_1, mass_2 = bh_mass, ns_mass
        total_mass = mass_1 + mass_2
        
        
        # ========================================================================
        # CRITICAL: Mass-aware approximant selection
        # ========================================================================
        if total_mass <= 6.0:
            # Low-mass NSBH: tidal effects matter
            approximant =  'IMRPhenomPv2_NRTidal'
            approximant_type = 'tidal'
        else:
            # High-mass NSBH: tidal effects negligible
            approximant = 'IMRPhenomPv2'
            approximant_type = 'non_precessing'
        
        # Distance and SNR
        snr_min, snr_max = self.snr_ranges[snr_regime]
        target_snr = np.random.uniform(snr_min, snr_max)
        
        chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)

        # Sample distance independently - uniform in comoving volume
        d_min, d_max = 20.0, 800.0  # Mpc (realistic NSBH horizon)
        u_d = np.random.random()
        luminosity_distance = float((d_min**3 + u_d * (d_max**3 - d_min**3))**(1/3))

        # Target SNR is guide only
        target_snr = np.random.uniform(snr_min, snr_max)
                
        # Black hole spin
        if approximant_type == 'tidal' or np.random.random() < 0.6:
            a1 = np.random.uniform(0.0, 0.99)
        else:
            a1 = 0.0
        
        # NS spin is small
        a2 = np.random.uniform(0.0, 0.05)
        
        # Spin orientations
        if 'Pv2' in approximant:
            tilt1 = np.random.uniform(0, np.pi/3)
            phi12 = np.random.uniform(0, 2*np.pi)
            phi_jl = np.random.uniform(0, 2*np.pi)
        else:
            tilt1 = 0.0
            phi12 = phi_jl = 0.0
        
        tilt2 = 0.0
        
        if total_mass > 30 and luminosity_distance < 100:
            self.logger.debug(f"High-mass NSBH at low distance: M={total_mass:.1f}, D={luminosity_distance:.1f}")
            
            
        # Sky location
        ra = np.random.uniform(0, 2*np.pi)
        dec = np.arcsin(np.random.uniform(-1, 1))
        cos_theta_jn = np.random.uniform(-1, 1)
        theta_jn = float(np.arccos(cos_theta_jn))
        psi = np.random.uniform(0, np.pi)
        phase = np.random.uniform(0, 2*np.pi)
        geocent_time = 0.0
        
        # Tidal parameters (only for low-mass systems)
        lambda_1 = 0
        if approximant_type == 'tidal':
            eos_type = np.random.choice(['soft', 'medium', 'stiff'])
            if eos_type == 'soft':
                lambda_2 = np.random.lognormal(np.log(800), 0.5) * (1.4 / ns_mass)**5
            elif eos_type == 'medium':
                lambda_2 = np.random.lognormal(np.log(400), 0.7) * (1.4 / ns_mass)**5
            else:
                lambda_2 = np.random.lognormal(np.log(200), 0.8) * (1.4 / ns_mass)**5
            lambda_2 = np.clip(lambda_2, 0, 3000)
        else:
            eos_type = 'N/A'
            lambda_2 = 0
            
        d_L = luminosity_distance  # The luminosity_distance variable already computed
        z = calculate_redshift(d_L)
        d_C = calculate_comoving_distance(z) if z is not None else d_L / (1.0 + z)
        
        return {
            'mass_1': mass_1,
            'mass_2': mass_2,
            'total_mass': total_mass,
            'chirp_mass': chirp_mass,
            'mass_ratio': mass_2 / mass_1,
            'symmetric_mass_ratio': (mass_1 * mass_2) / total_mass**2,
            'luminosity_distance': float(d_L),
            'redshift': float(z),              
            'comoving_distance': float(d_C),  
            'target_snr': target_snr,
            'a1': a1, 'a2': a2,
            'tilt1': tilt1, 'tilt2': tilt2,
            'phi12': phi12, 'phi_jl': phi_jl,
            'effective_spin': compute_effective_spin(mass_1, mass_2, a1, a2, tilt1, tilt2),
            'ra': ra, 'dec': dec,
            'theta_jn': theta_jn, 'psi': psi, 'phase': phase,
            'geocent_time': geocent_time,
            'f_lower': 20.0,
            'f_ref': 50.0,
            'approximant': approximant,
            'approximant_type': approximant_type,
            'eccentricity': 0.0,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'eos_type': eos_type,
            'bh_mass_type': bh_mass_type if not is_edge_case else 'extreme',
            'edge_case': is_edge_case,
            'edge_case_type': edge_type
        }


    def sample_pure_noise(self, detector_network: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate a pure-noise sample per detector using the configured PSDs.
        Returns a dict {detector_name: noise_time_series}
        """
        noise = {}
        for det_name in detector_network:
            # Reuse your realistic noise generator if available
            try:
                ts = generate_realistic_noise(det_name)
            except Exception:
                # Fallback: white gaussian noise tapered to avoid edges
                ts = np.random.normal(0.0, 1.0, self.n_samples).astype(np.float64)
                # light taper at ends
                taper_len = max(16, int(0.01 * self.n_samples))
                window = np.ones(self.n_samples, dtype=np.float64)
                ramp = np.hanning(2 * taper_len)
                window[:taper_len] = ramp[:taper_len]
                window[-taper_len:] = ramp[-taper_len:]
                ts *= window
            noise[det_name] = ts
        return noise


