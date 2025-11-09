"""
Parameter Sampler for GW Signals
Implements astrophysically realistic parameter distributions
"""

import numpy as np
import logging
from typing import Dict, List
from scipy.stats import truncnorm, beta

from .config import (
    MASS_RANGES, DISTANCE_RANGES, SNR_RANGES, 
    SNR_DISTRIBUTION, EVENT_TYPE_DISTRIBUTION
)
from .utils import calculate_redshift, calculate_comoving_distance, compute_effective_spin


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
        
        # ✅ Store distributions from config
        self.snr_distribution = SNR_DISTRIBUTION
        self.event_type_distribution = EVENT_TYPE_DISTRIBUTION
        
        self.reference_snr = 35  # Increased from 15 for stronger correlation
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
    
    def sample_bbh_parameters(self, snr_regime: str = None, is_edge_case: bool = False) -> Dict:
        """
        Generate complete BBH parameters with volumetric distance sampling and
        strong stochastic scatter on SNR to reduce deterministic distance–SNR coupling.
        """
        # bookkeeping & RNG
        self.stats['event_types']['BBH'] += 1
        rng = np.random.default_rng()

        # --- Mass sampling (unchanged logic, RNG switched) ---
        if is_edge_case and rng.random() < 0.3:
            mass_1 = rng.uniform(60, 100)
            mass_2 = rng.uniform(50, mass_1)
            edge_case_type = 'short_bbh'
            self.stats['edge_cases']['short_bbh'] += 1
        elif is_edge_case and rng.random() < 0.5:
            mass_1 = rng.uniform(30, 80)
            q = rng.uniform(0.05, 0.15)
            mass_2 = mass_1 * q
            edge_case_type = 'extreme_mass_ratio'
            self.stats['edge_cases']['extreme_mass_ratio'] += 1
        else:
            m1_raw = np.clip(rng.lognormal(mean=np.log(25.0), sigma=0.35), 5.0, 80.0)
            m2_raw = np.clip(rng.lognormal(mean=np.log(20.0), sigma=0.40), 5.0, 80.0)
            q_min = 0.1
            m2_raw = max(m2_raw, q_min * m1_raw)
            mass_1, mass_2 = (m1_raw, m2_raw) if m1_raw >= m2_raw else (m2_raw, m1_raw)
            mass_1 += rng.uniform(-0.05, 0.05)
            mass_2 += rng.uniform(-0.05, 0.05)
            mass_1 = float(np.clip(mass_1, 5.0, 100.0))
            mass_2 = float(np.clip(mass_2, 5.0, min(100.0, mass_1)))
            edge_case_type = 'none'

        total_mass = mass_1 + mass_2
        chirp_mass = (mass_1 * mass_2) ** (3 / 5) / total_mass ** (1 / 5)
        mass_ratio = mass_2 / mass_1
        symmetric_mass_ratio = (mass_1 * mass_2) / total_mass ** 2

        # ✅ FIX: Sample target_snr first and derive distance from it (matching BNS/NSBH)
        self._sampling_event_type = 'BBH'
        try:
            target_snr = self._sample_target_snr(snr_regime)
        finally:
            self._sampling_event_type = None

        # Compute distance consistent with target_snr using chirp-mass scaling
        luminosity_distance = (self.reference_distance *
                       (chirp_mass / self.reference_mass) ** (5 / 6) *
                       (self.reference_snr / target_snr))

        # Add minimal jitter and clamp to valid range
        jitter_factor = rng.uniform(0.999, 1.001)  # Minimal jitter to preserve correlation
        luminosity_distance *= jitter_factor
        d_min, d_max = self.distance_ranges['BBH']
        luminosity_distance = float(np.clip(luminosity_distance, d_min, d_max))

        # --- Spins (unchanged, RNG) ---
        a1 = float(np.clip(rng.beta(2, 5), 0, 0.99))
        a2 = float(np.clip(rng.beta(2, 5), 0, 0.99))
        cos_tilt1 = rng.uniform(-1.0, 1.0)
        cos_tilt2 = rng.uniform(-1.0, 1.0)
        tilt1 = float(np.arccos(cos_tilt1))
        tilt2 = float(np.arccos(cos_tilt2))
        phi12 = float(rng.uniform(0, 2 * np.pi))
        phi_jl = float(rng.uniform(0, 2 * np.pi))

        # --- Sky location & isotropic inclination (keeps isotropy) ---
        ra = float(rng.uniform(0, 2 * np.pi))
        dec = float(np.arcsin(rng.uniform(-1, 1)))
        cos_theta_jn = rng.uniform(-1.0, 1.0)
        theta_jn = float(np.arccos(cos_theta_jn))
        psi = float(rng.uniform(0, np.pi))
        phase = float(rng.uniform(0, 2 * np.pi))
        geocent_time = float(rng.uniform(-0.1, 0.1))

        # --- Cosmology ---
        z = calculate_redshift(luminosity_distance)
        d_C = calculate_comoving_distance(z) if z is not None else luminosity_distance / (1 + (z or 0))

        # --- Package output ---
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
            'redshift': float(z) if z is not None else 0.0,
            'comoving_distance': float(d_C),
            'ra': float(ra),
            'dec': float(dec),
            'theta_jn': float(theta_jn),
            'psi': float(psi),
            'phase': float(phase),
            'geocent_time': float(geocent_time),
            'f_lower': 20.0,
            'f_ref': 20.0,
            'approximant': 'IMRPhenomD',
            'target_snr': float(target_snr),
            'lambda_1': 0.0,
            'lambda_2': 0.0,
            'is_real_event': False,
            'edge_case': is_edge_case,
            'edge_case_type': edge_case_type
        }


    
    def sample_bns_parameters(self, snr_regime: str = None, is_edge_case: bool = False) -> Dict:
        """Generate BNS parameters."""
        
        self.stats['event_types']['BNS'] += 1
        
        if is_edge_case:
            f_lower = float(np.random.uniform(10.0, 15.0))
            edge_type = 'long_bns_inspiral'
            self.stats['edge_cases']['long_bns_inspiral'] += 1
        else:
            f_lower = 35.0
            edge_type = None
        
        # Sample independently with different widths + jitter
        m1_raw = np.clip(np.random.normal(1.40, 0.15), 1.0, 2.5)
        m2_raw = np.clip(np.random.normal(1.40, 0.20), 1.0, 2.5)
        
        # Enforce ordering
        mass_1, mass_2 = (m1_raw, m2_raw) if m1_raw >= m2_raw else (m2_raw, m1_raw)
        
        # Tiny jitter to break determinism
        mass_1 += np.random.uniform(-0.01, 0.01)
        mass_2 += np.random.uniform(-0.01, 0.01)
        
        # Clip to valid range
        mass_1 = float(np.clip(mass_1, 1.0, 2.5))
        mass_2 = float(np.clip(mass_2, 1.0, min(2.5, mass_1)))
        
        total_mass = mass_1 + mass_2
        chirp_mass = (mass_1 * mass_2)**(3/5) / total_mass**(1/5)
        
        # ✅ FIX: Sample target_snr first and derive distance from it (allow conditioning by event type)
        self._sampling_event_type = 'BNS'
        try:
            target_snr = self._sample_target_snr(snr_regime)
        finally:
            self._sampling_event_type = None

        # Compute expected distance from target_snr using chirp-mass scaling
        luminosity_distance = (self.reference_distance *
                       (chirp_mass / self.reference_mass)**(5/6) *
                       (self.reference_snr / target_snr))

        # Add minimal jitter and clamp
        jitter_factor = np.random.uniform(0.999, 1.001)  # Minimal jitter to preserve correlation
        luminosity_distance *= jitter_factor
        d_min, d_max = self.distance_ranges['BNS']  # Use config bounds: (10.0, 180.0)
        luminosity_distance = float(np.clip(luminosity_distance, d_min, d_max))
        
        # Tidal parameters
        lambda_1 = float(np.clip(np.random.lognormal(np.log(400), 0.7) * (1.4/mass_1)**5, 50, 5000))
        lambda_2 = float(np.clip(np.random.lognormal(np.log(400), 0.7) * (1.4/mass_2)**5, 50, 5000))
        
        # Low spins
        a1, a2 = float(np.random.uniform(0.0, 0.05)), float(np.random.uniform(0.0, 0.05))
        tilt1, tilt2 = 0.0, 0.0
        
        # Isotropic inclination
        cos_theta_jn = np.random.uniform(-1, 1)
        theta_jn = float(np.arccos(cos_theta_jn))
        
        # Sky location
        ra = float(np.random.uniform(0, 2*np.pi))
        dec = float(np.arcsin(np.random.uniform(-1, 1)))
        psi = float(np.random.uniform(0, np.pi))
        phase = float(np.random.uniform(0, 2*np.pi))
        
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
            'redshift': float(z) if z is not None else 0.0,
            'comoving_distance': float(d_C),
            'target_snr': float(target_snr),
            'a1': float(a1),
            'a2': float(a2),
            'tilt1': float(tilt1),
            'tilt2': float(tilt2),
            'effective_spin': compute_effective_spin(mass_1, mass_2, a1, a2, tilt1, tilt2),
            'ra': float(ra), 
            'dec': float(dec),
            'theta_jn': float(theta_jn),
            'psi': float(psi),
            'phase': float(phase),
            'f_lower': float(f_lower),
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
    
    def sample_nsbh_parameters(self, snr_regime: str = None, is_edge_case: bool = False) -> Dict:
        """Generate complete NSBH parameters."""
        
        self.stats['event_types']['NSBH'] += 1
        
        # Neutron star mass
        ns_mass = float(np.random.uniform(1.2, 2.0))
        
        # Black hole mass with diversity
        if is_edge_case:
            edge_type = 'extreme_mass'
            self.stats['edge_cases']['extreme_mass'] += 1
            bh_mass = float(np.random.uniform(50.0, 100.0))
            bh_mass_type = 'extreme'
        else:
            edge_type = None
            bh_mass_type = np.random.choice(['light', 'medium', 'heavy'])
            
            if bh_mass_type == 'light':
                bh_mass = float(np.random.uniform(3.0, 8.0))
            elif bh_mass_type == 'medium':
                bh_mass = float(np.random.uniform(8.0, 25.0))
            else:  # heavy
                bh_mass = float(np.random.uniform(25.0, 50.0))
        
        mass_1, mass_2 = bh_mass, ns_mass
        total_mass = mass_1 + mass_2
        
        # Mass-aware approximant selection
        if total_mass <= 6.0:
            approximant = 'IMRPhenomPv2_NRTidal'
            approximant_type = 'tidal'
        else:
            approximant = 'IMRPhenomPv2'
            approximant_type = 'non_precessing'
        
        # ✅ FIX: Sample target_snr first and derive distance from chirp mass (allow conditioning by event type)
        self._sampling_event_type = 'NSBH'
        try:
            target_snr = self._sample_target_snr(snr_regime)
        finally:
            self._sampling_event_type = None

        chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)

        # Compute distance consistent with target_snr
        luminosity_distance = (self.reference_distance *
        (chirp_mass / self.reference_mass)**(5/6) *
        (self.reference_snr / target_snr))
        # Clamp and minimal jitter
        jitter_factor = np.random.uniform(0.999, 1.001)  # Minimal jitter to preserve correlation
        luminosity_distance *= jitter_factor
        d_min, d_max = self.distance_ranges['NSBH']  # Use config bounds: (20.0, 600.0)
        luminosity_distance = float(np.clip(luminosity_distance, d_min, d_max))
        
        # Black hole spin
        if approximant_type == 'tidal' or np.random.random() < 0.6:
            a1 = float(np.random.uniform(0.0, 0.99))
        else:
            a1 = 0.0
        
        # NS spin is small
        a2 = float(np.random.uniform(0.0, 0.05))
        
        # Spin orientations
        if 'Pv2' in approximant:
            tilt1 = float(np.random.uniform(0, np.pi/3))
            phi12 = float(np.random.uniform(0, 2*np.pi))
            phi_jl = float(np.random.uniform(0, 2*np.pi))
        else:
            tilt1 = 0.0
            phi12 = phi_jl = 0.0
        
        tilt2 = 0.0
        
        # Sky location
        ra = float(np.random.uniform(0, 2*np.pi))
        dec = float(np.arcsin(np.random.uniform(-1, 1)))
        cos_theta_jn = np.random.uniform(-1, 1)
        theta_jn = float(np.arccos(cos_theta_jn))
        psi = float(np.random.uniform(0, np.pi))
        phase = float(np.random.uniform(0, 2*np.pi))
        geocent_time = 0.0
        
        # Tidal parameters
        lambda_1 = 0.0
        if approximant_type == 'tidal':
            eos_type = np.random.choice(['soft', 'medium', 'stiff'])
            if eos_type == 'soft':
                lambda_2 = float(np.random.lognormal(np.log(800), 0.5) * (1.4 / ns_mass)**5)
            elif eos_type == 'medium':
                lambda_2 = float(np.random.lognormal(np.log(400), 0.7) * (1.4 / ns_mass)**5)
            else:
                lambda_2 = float(np.random.lognormal(np.log(200), 0.8) * (1.4 / ns_mass)**5)
            lambda_2 = float(np.clip(lambda_2, 0, 3000))
        else:
            eos_type = 'N/A'
            lambda_2 = 0.0
        
        d_L = luminosity_distance
        z = calculate_redshift(d_L)
        d_C = calculate_comoving_distance(z) if z is not None else d_L
        
        return {
            'mass_1': float(mass_1),
            'mass_2': float(mass_2),
            'total_mass': float(total_mass),
            'chirp_mass': float(chirp_mass),
            'mass_ratio': float(mass_2 / mass_1),
            'symmetric_mass_ratio': float((mass_1 * mass_2) / total_mass**2),
            'luminosity_distance': float(d_L),
            'redshift': float(z) if z is not None else 0.0,
            'comoving_distance': float(d_C),
            'target_snr': float(target_snr),
            'a1': float(a1), 
            'a2': float(a2),
            'tilt1': float(tilt1), 
            'tilt2': float(tilt2),
            'phi12': float(phi12), 
            'phi_jl': float(phi_jl),
            'effective_spin': compute_effective_spin(mass_1, mass_2, a1, a2, tilt1, tilt2),
            'ra': float(ra), 
            'dec': float(dec),
            'theta_jn': float(theta_jn), 
            'psi': float(psi), 
            'phase': float(phase),
            'geocent_time': float(geocent_time),
            'f_lower': 20.0,
            'f_ref': 50.0,
            'approximant': approximant,
            'approximant_type': approximant_type,
            'eccentricity': 0.0,
            'lambda_1': float(lambda_1),
            'lambda_2': float(lambda_2),
            'eos_type': eos_type,
            'bh_mass_type': bh_mass_type,
            'edge_case': is_edge_case,
            'edge_case_type': edge_type,
            'type': 'NSBH'
        }

    def recompute_target_snr_from_params(self, params: Dict) -> float:
        """
        Recompute and overwrite the 'target_snr' field in a params dict using the
        sampler's reference calibration.

        Formula: target_snr = reference_snr * (M_chirp / reference_mass)^(5/6) * (reference_distance / D)

        This method is intended to be called when callers override 'luminosity_distance'
        or mass parameters after sampling, to keep the explicit 'target_snr' consistent
        with the parameter set.

        Returns the recomputed target_snr (float).
        """
        try:
            m1 = float(params.get('mass_1'))
            m2 = float(params.get('mass_2'))
            d = float(params.get('luminosity_distance'))
        except Exception:
            # Missing or invalid inputs; leave target_snr unchanged if present
            return float(params.get('target_snr', self.reference_snr))

        if d <= 0:
            return float(params.get('target_snr', self.reference_snr))

        M_total = m1 + m2
        M_chirp = (m1 * m2)**(3/5) / M_total**(1/5)

        target = self.reference_snr * (M_chirp / self.reference_mass)**(5/6) * (self.reference_distance / d)

        # Clip to reasonable physical range
        target = float(np.clip(target, 5.0, 100.0))

        params['target_snr'] = target
        return target

    def calibrate_snr_by_event_type(self, n_samples: int = 2000, random_seed: int = None) -> Dict:
        """Empirically estimate P(snr_regime | event_type) by forward-sampling masses and distances.

        This draws masses according to the per-type priors used in the sampler, draws distances
        from the configured distance ranges (volume weighting), computes the approximate SNR
        using the sampler's reference calibration, and records the fraction of samples falling
        into each SNR regime from `SNR_RANGES`.

        Returns the conditional distribution mapping {event_type: {regime: fraction}} and stores
        it in `self.conditional_snr` for later sampling.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        conditional = {}
        for et in ['BBH', 'BNS', 'NSBH']:
            counts = {r: 0 for r in self.snr_ranges.keys()}
            for i in range(n_samples):
                # Sample masses according to lightweight priors used in sampling
                if et == 'BBH':
                    m1 = np.clip(np.random.lognormal(mean=np.log(25.0), sigma=0.35), 5.0, 100.0)
                    m2 = np.clip(np.random.lognormal(mean=np.log(20.0), sigma=0.40), 5.0, 100.0)
                    if m2 > m1:
                        m1, m2 = m2, m1
                    # enforce min q
                    q_min = 0.1
                    if m2 < q_min * m1:
                        m2 = q_min * m1
                    dmin, dmax = self.distance_ranges['BBH']
                elif et == 'BNS':
                    m1 = np.clip(np.random.normal(1.40, 0.15), 1.0, 2.5)
                    m2 = np.clip(np.random.normal(1.40, 0.20), 1.0, 2.5)
                    if m2 > m1:
                        m1, m2 = m2, m1
                    dmin, dmax = self.distance_ranges['BNS']
                else:  # NSBH
                    ns_mass = float(np.random.uniform(1.2, 2.0))
                    bh_mass = float(np.random.uniform(3.0, 50.0))
                    m1, m2 = bh_mass, ns_mass
                    dmin, dmax = self.distance_ranges['NSBH']

                # Volume-weighted distance draw
                u = np.random.random()
                d = (dmin**3 + u * (dmax**3 - dmin**3))**(1/3)

                # Compute chirp mass and SNR
                M_total = m1 + m2
                M_chirp = (m1 * m2)**(3/5) / M_total**(1/5)
                snr = self.reference_snr * (M_chirp / self.reference_mass)**(5/6) * (self.reference_distance / d)

                # categorize with clamping: below min -> 'weak', above max -> 'loud'
                mins = [rng[0] for rng in self.snr_ranges.values()]
                maxs = [rng[1] for rng in self.snr_ranges.values()]
                overall_min = min(mins)
                overall_max = max(maxs)
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
            if total > 0:
                conditional[et] = {r: counts[r] / total for r in counts}
            else:
                conditional[et] = {r: 1.0/len(self.snr_ranges) for r in self.snr_ranges}

        # Store calibration
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

                    snr = float(params.get('target_snr', float('nan')))
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
