"""
Gravitational-wave parameter sampler.

Samples from proper astrophysical priors. SNR is NOT targeted here — it is
measured after waveform generation and injection. The caller decides whether to
accept a sample based on measured SNR.

Parameters returned match the 11-parameter NPE output set:
  mass_1, mass_2, luminosity_distance, ra, dec, theta_jn, psi, phase,
  geocent_time, a1, a2

geocent_time is stored as seconds offset from GPS_REF so the parameter
scaler's linear_minmax(min=-2, max=7) applies without wrapping.
"""

import numpy as np
import logging
from typing import Dict, Optional, List

_log = logging.getLogger(__name__)

# O4 era reference GPS time (2023-05-24 18:00:00 UTC)
GPS_REF: float = 1369224018.0

# ── Distance priors (Mpc) ──────────────────────────────────────────────────────
_DIST_MIN = {"BBH": 50.0, "BNS": 10.0, "NSBH": 20.0}
_DIST_MAX = {"BBH": 5000.0, "BNS": 600.0, "NSBH": 2000.0}

# ── Mass bounds (M☉) ──────────────────────────────────────────────────────────
_MASS_BBH = (5.0, 100.0)
_MASS_BNS = (1.0, 2.5)
_MASS_NSBH_BH = (3.0, 100.0)
_MASS_NSBH_NS = (1.0, 2.5)

# ── Spin bounds ───────────────────────────────────────────────────────────────
_SPIN_BBH = (0.0, 0.99)
_SPIN_BNS = (0.0, 0.05)
_SPIN_NS = (0.0, 0.05)
_SPIN_BH = (0.0, 0.99)

# ── Default event-type fractions ──────────────────────────────────────────────
_DEFAULT_FRACTIONS = {"BBH": 0.55, "BNS": 0.20, "NSBH": 0.20, "noise": 0.05}


class ParameterSampler:
    """
    Sample GW parameters from astrophysical priors.

    Distance is drawn from a volume-weighted prior P(d) ∝ d².
    No SNR targeting — measured SNR drives accept/reject downstream.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        seed = cfg.get("random_seed", None)
        self.rng = np.random.default_rng(seed)

        raw_fracs = cfg.get("event_type_distribution", _DEFAULT_FRACTIONS)
        total = sum(raw_fracs.values()) or 1.0
        self._types = list(raw_fracs)
        self._probs = np.array([raw_fracs[t] / total for t in self._types])

        # Allow config to override distance bounds
        dist_cfg = cfg.get("distance_ranges", {})
        self._dist_min = {
            "BBH": dist_cfg.get("BBH", [_DIST_MIN["BBH"]])[0],
            "BNS": dist_cfg.get("BNS", [_DIST_MIN["BNS"]])[0],
            "NSBH": dist_cfg.get("NSBH", [_DIST_MIN["NSBH"]])[0],
        }
        self._dist_max = {
            "BBH": dist_cfg.get("BBH", [None, _DIST_MAX["BBH"]])[-1],
            "BNS": dist_cfg.get("BNS", [None, _DIST_MAX["BNS"]])[-1],
            "NSBH": dist_cfg.get("NSBH", [None, _DIST_MAX["NSBH"]])[-1],
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def sample_parameters(self, event_type: Optional[str] = None) -> Dict:
        """
        Draw one complete set of GW parameters.

        Returns a dict with keys:
          mass_1, mass_2, luminosity_distance, ra, dec, theta_jn, psi, phase,
          geocent_time (offset in s from GPS_REF), a1, a2,
          chirp_mass, mass_ratio, event_type, geocent_time_gps
        """
        if event_type is None:
            event_type = self.rng.choice(self._types, p=self._probs)

        if event_type == "noise":
            return self._noise_params()

        p: Dict = {"event_type": event_type}
        p.update(self._sample_masses(event_type))
        p["luminosity_distance"] = self._sample_distance(event_type)
        p["ra"] = float(self.rng.uniform(0.0, 2 * np.pi))
        p["dec"] = float(np.arcsin(self.rng.uniform(-1.0, 1.0)))
        p["theta_jn"] = float(np.arccos(self.rng.uniform(-1.0, 1.0)))
        p["psi"] = float(self.rng.uniform(0.0, np.pi))
        p["phase"] = float(self.rng.uniform(0.0, 2 * np.pi))
        # geocent_time as offset [-2, 7] s; absolute GPS kept separately
        t_off = float(self.rng.uniform(-2.0, 7.0))
        p["geocent_time"] = t_off
        p["geocent_time_gps"] = GPS_REF + t_off
        p.update(self._sample_spins(event_type))
        m1, m2 = p["mass_1"], p["mass_2"]
        p["chirp_mass"] = float((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)
        p["mass_ratio"] = float(m2 / m1)
        return p

    def sample_batch(self, n: int, event_type: Optional[str] = None) -> List[Dict]:
        return [self.sample_parameters(event_type) for _ in range(n)]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sample_distance(self, event_type: str) -> float:
        """P(d) ∝ d² (uniform in comoving volume at low z)."""
        d_min = self._dist_min.get(event_type, 50.0)
        d_max = self._dist_max.get(event_type, 5000.0)
        # Inverse-CDF of d² prior: d = (d_min³ + u*(d_max³-d_min³))^(1/3)
        u = self.rng.uniform(0.0, 1.0)
        d3 = d_min ** 3 + u * (d_max ** 3 - d_min ** 3)
        return float(d3 ** (1.0 / 3.0))

    def _sample_masses(self, event_type: str) -> Dict:
        et = event_type.upper()
        if et == "BBH":
            lo, hi = _MASS_BBH
            # Flat-in-log for m1 (power-law P(m)∝1/m), flat for m2 in [lo, m1]
            log_m1 = self.rng.uniform(np.log(lo), np.log(hi))
            m1 = float(np.exp(log_m1))
            m2 = float(self.rng.uniform(lo, m1))
        elif et == "BNS":
            lo, hi = _MASS_BNS
            m1 = float(self.rng.uniform(lo, hi))
            m2 = float(self.rng.uniform(lo, m1))
        else:  # NSBH
            bh_lo, bh_hi = _MASS_NSBH_BH
            ns_lo, ns_hi = _MASS_NSBH_NS
            m1 = float(np.exp(self.rng.uniform(np.log(bh_lo), np.log(bh_hi))))
            m2 = float(self.rng.uniform(ns_lo, ns_hi))
        return {"mass_1": max(m1, m2), "mass_2": min(m1, m2)}

    def _sample_spins(self, event_type: str) -> Dict:
        et = event_type.upper()
        if et == "BNS":
            a1 = float(self.rng.uniform(*_SPIN_BNS))
            a2 = float(self.rng.uniform(*_SPIN_BNS))
        elif et == "NSBH":
            a1 = float(self.rng.uniform(*_SPIN_BH))   # BH component
            a2 = float(self.rng.uniform(*_SPIN_NS))   # NS component
        else:
            a1 = float(self.rng.uniform(*_SPIN_BBH))
            a2 = float(self.rng.uniform(*_SPIN_BBH))
        return {"a1": a1, "a2": a2}

    def _noise_params(self) -> Dict:
        """Placeholder parameters for noise-only samples (no signal)."""
        return {
            "event_type": "noise",
            "mass_1": 30.0, "mass_2": 30.0,
            "luminosity_distance": 1000.0,
            "ra": 0.0, "dec": 0.0,
            "theta_jn": 0.0, "psi": 0.0, "phase": 0.0,
            "geocent_time": 0.0,
            "geocent_time_gps": GPS_REF,
            "a1": 0.0, "a2": 0.0,
            "chirp_mass": 26.1, "mass_ratio": 1.0,
        }
