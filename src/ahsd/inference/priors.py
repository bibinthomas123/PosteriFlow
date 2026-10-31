"""
The prior a LeanNPE checkpoint was trained on, per source class.

Single source of truth for importance sampling (log density of the training
prior) and the dynesty bridge (bilby priors with the same support). Bounds
come from the generator (ahsd.data.parameter_sampler); the distance range and
shape come from the dataset the checkpoint was trained on, which the trainer
records in its args (dataset_distance_ranges, dataset_distance_prior).

    prior = TrainingPrior.from_meta(meta, event_type="BNS")
    log_pi = prior.log_prob(theta)              # [N, 11] physical
    priors = prior.bilby_priors(center_time)    # bilby PriorDict
    wf = prior.waveform()                       # source model, approximant
    f_min = window_fmin(mc_min, t_merger)       # band that fits a 4 s segment

The waveform model is the generator's for the class, with zero tidal
deformability (the generator never sets lambda_1/lambda_2, so bilby uses 0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ahsd.data import parameter_sampler as ps
from ahsd.models.lean_npe import PARAM_NAMES

_J = {n: i for i, n in enumerate(PARAM_NAMES)}

# merger-time half-width: the sampler draws +-1.5 s, time-shift augmentation
# adds up to +-0.1 s; the scaler range is +-1.6 s
TC_HALF_WIDTH = 1.6

F_LOWER = 20.0


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def window_fmin(mc_min: float, t_merger: float, f_lower: float = F_LOWER) -> float:
    """Lowest template frequency for a likelihood on one 4 s segment: the
    signal of the lightest chirp mass considered must lie between the segment
    start and the merger (t_merger s after the start), with the generator's
    margin (1.2 x leading-order chirp time + 0.25 s, BilbyWaveformGenerator.
    fits_window). Below it a frequency-domain template wraps around the
    segment. This is also the band of the signal that is inside the window
    the network sees, so the comparison uses the same information."""
    tau_max = max((t_merger - 0.25) / 1.2, 0.05)
    f = 100.0 * (2.18 * (1.21 / max(mc_min, 1e-3)) ** (5.0 / 3.0) / tau_max) ** (3.0 / 8.0)
    return float(max(f_lower, f))


# per class: (mass_1 range, mass_2 range, a1 max, a2 max)
_CLASS = {
    "BBH": (ps._MASS_BBH, ps._MASS_BBH, ps._SPIN_BBH[1], ps._SPIN_BBH[1]),
    "NSBH": (ps._MASS_NSBH_BH, ps._MASS_NSBH_NS, ps._SPIN_BH[1], ps._SPIN_NS[1]),
    "BNS": (ps._MASS_BNS, ps._MASS_BNS, ps._SPIN_BNS[1], ps._SPIN_BNS[1]),
}


@dataclass(frozen=True)
class TrainingPrior:
    event_type: str
    m1_range: Tuple[float, float]
    m2_range: Tuple[float, float]
    a1_max: float
    a2_max: float
    dist_range: Tuple[float, float]
    dist_shape: str                       # "uniform" | "comoving_d2"

    @classmethod
    def for_class(cls, event_type: str, dist_ranges: Optional[Dict] = None,
                  dist_shape: Optional[str] = None) -> "TrainingPrior":
        """Prior of one source class; dist_ranges is the dataset's
        {class: [lo, hi]} (sampler defaults when absent)."""
        et = event_type.upper()
        if et not in _CLASS:
            raise ValueError(f"unknown event_type {event_type!r}; expected one of {list(_CLASS)}")
        m1, m2, a1, a2 = _CLASS[et]
        rng = (dist_ranges or {}).get(et) or (ps._DIST_MIN[et], ps._DIST_MAX[et])
        shape = (dist_shape or "comoving_d2").lower()
        if shape not in ("uniform", "comoving_d2"):
            raise ValueError(f"unknown distance prior {dist_shape!r}")
        return cls(et, tuple(m1), tuple(m2), float(a1), float(a2),
                   (float(rng[0]), float(rng[-1])), shape)

    @classmethod
    def from_meta(cls, meta: Dict, event_type: Optional[str] = None) -> "TrainingPrior":
        """From load_model() metadata or a PosteriorResult config. A
        single-class model uses its own class; a mixed model uses
        `event_type` (the class the event is analysed as), default BBH."""
        trained = meta.get("train_event_types") or []
        if len(trained) == 1:
            et = trained[0]
        else:
            et = event_type or meta.get("event_type") or "BBH"
        return cls.for_class(et, meta.get("train_distance_ranges"),
                             meta.get("train_distance_prior"))

    @property
    def mc_min(self) -> float:
        """Smallest chirp mass in the support."""
        return float(chirp_mass(self.m1_range[0], self.m2_range[0]))

    def waveform(self) -> Dict:
        """The generator's waveform model for this class: bilby source model,
        parameter conversion, approximant and the sampler's conversion
        function."""
        import bilby
        from ahsd.data.bilby_pipeline import _APPROXIMANTS, _get_source_model
        source, conversion = _get_source_model({"event_type": self.event_type})
        bns = self.event_type == "BNS"
        return {"source_model": source, "parameter_conversion": conversion,
                "approximant": _APPROXIMANTS[self.event_type.lower()],
                "sampler_conversion": (bilby.gw.conversion.generate_all_bns_parameters if bns
                                       else bilby.gw.conversion.generate_all_bbh_parameters)}

    # ── log density ──────────────────────────────────────────────────────
    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        """Log density of the training prior (up to a constant), theta [N, 11]
        physical; -inf outside the support."""
        m1, m2 = theta[:, _J["mass_1"]], theta[:, _J["mass_2"]]
        d = theta[:, _J["luminosity_distance"]]
        dec, th = theta[:, _J["dec"]], theta[:, _J["theta_jn"]]
        tc = theta[:, _J["geocent_time"]]
        a1, a2 = theta[:, _J["a1"]], theta[:, _J["a2"]]
        (m1_lo, m1_hi), (m2_lo, m2_hi) = self.m1_range, self.m2_range
        d_lo, d_hi = self.dist_range

        ok = ((m1 >= m1_lo) & (m1 <= m1_hi) & (m2 >= m2_lo) & (m2 <= m2_hi) & (m2 <= m1) &
              (d >= d_lo) & (d <= d_hi) & (np.abs(tc) < TC_HALF_WIDTH) &
              (a1 >= 0) & (a1 <= self.a1_max) & (a2 >= 0) & (a2 <= self.a2_max) &
              (th > 0) & (th < np.pi) & (np.abs(dec) < np.pi / 2))
        m1c, m2c = np.maximum(m1, 1e-6), np.maximum(m2, 1e-6)
        if self.event_type == "BBH":
            # log m1 uniform, log m2 uniform on [lo, log m1]: 1/(m1 m2 log(m1/lo))
            lp = -(np.log(m1c) + np.log(m2c)
                   + np.log(np.maximum(np.log(m1c / m2_lo), 1e-9)))
        elif self.event_type == "NSBH":
            # BH log-uniform, NS uniform (independent)
            lp = -np.log(m1c)
        else:
            # BNS: m1 uniform, m2 uniform on [lo, m1]: 1/(m1 - lo)
            lp = -np.log(np.maximum(m1 - m2_lo, 1e-9))
        if self.dist_shape == "comoving_d2":
            lp = lp + 2.0 * np.log(np.maximum(d, 1e-6))
        lp = lp + np.log(np.maximum(np.cos(dec), 1e-12)) + np.log(np.maximum(np.sin(th), 1e-12))
        lp[~ok] = -np.inf
        return lp

    # ── bilby priors (dynesty) ───────────────────────────────────────────
    def bilby_priors(self, center_time: float, time_halfwidth: float = 0.2):
        """bilby PriorDict with the training support: per-class mass and
        spin bounds, the dataset's distance range and shape; aligned spins.
        Black-hole masses are log-uniform, neutron-star masses uniform; the
        posterior is sorted to mass_1 >= mass_2 afterwards
        (dynesty_bridge.align_conventions)."""
        import bilby
        P = bilby.core.prior
        priors = bilby.gw.prior.BBHPriorDict()
        del priors["chirp_mass"], priors["mass_ratio"]
        bh = {"BBH": (True, True), "NSBH": (True, False), "BNS": (False, False)}[self.event_type]
        for name, rng, log in (("mass_1", self.m1_range, bh[0]), ("mass_2", self.m2_range, bh[1])):
            priors[name] = (P.LogUniform if log else P.Uniform)(*rng, name=name)
        d_lo, d_hi = self.dist_range
        priors["luminosity_distance"] = (
            P.Uniform(d_lo, d_hi, name="luminosity_distance") if self.dist_shape == "uniform"
            else P.PowerLaw(alpha=2, minimum=d_lo, maximum=d_hi, name="luminosity_distance"))
        priors["a_1"] = P.Uniform(0, self.a1_max, "a_1")
        priors["a_2"] = P.Uniform(0, self.a2_max, "a_2")
        for k in ("tilt_1", "tilt_2", "phi_12", "phi_jl"):
            priors[k] = P.DeltaFunction(0.0)
        if self.event_type == "BNS":
            # the BNS source model needs tidal parameters; training used 0
            priors["lambda_1"] = P.DeltaFunction(0.0, name="lambda_1")
            priors["lambda_2"] = P.DeltaFunction(0.0, name="lambda_2")
        priors["geocent_time"] = P.Uniform(center_time - time_halfwidth,
                                           center_time + time_halfwidth, "geocent_time")
        return priors
