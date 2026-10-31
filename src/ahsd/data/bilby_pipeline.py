"""
Bilby-based gravitational-wave data pipeline.

Handles waveform generation, noise generation (synthetic + real GWOSC),
signal injection, and whitening. All physics is delegated to bilby/LALSuite.

Key design choice: SNR is MEASURED after injection, not targeted.
The caller (dataset_generator) decides whether to accept or reject a sample.

Real noise support: set GWOSC_CACHE_DIR to a directory containing pre-downloaded
*.gwf or bilby-format *.npz noise segments. Falls back to synthetic colored noise
when real segments are unavailable.
"""

import copy
import logging
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bilby
from bilby.gw.detector import (
    Interferometer,
    PowerSpectralDensity,
    get_empty_interferometer,
)
from bilby.gw import WaveformGenerator as _BilbyWFG
import bilby.gw.source as _gwsource
import bilby.gw.conversion as _gwconv

_log = logging.getLogger(__name__)

# Directory for cached real noise segments (set via env or config)
GWOSC_CACHE_DIR = os.environ.get("GWOSC_CACHE_DIR", "data/gwosc_cache")

# Fixed reference GPS time (O4 era) — MUST match parameter_sampler.GPS_REF.
# All data windows are centered on this time: [GPS_REF - T/2, GPS_REF + T/2].
# Mergers are sampled within [GPS_REF - T/2 + margin, GPS_REF + T/2 - margin].
_GPS_REF: float = 1369224018.0  # 2023-05-24 18:00:00 UTC

# ── Approximant selection ─────────────────────────────────────────────────────
_APPROXIMANTS = {
    "bbh": "IMRPhenomXP",
    "bns": "IMRPhenomD_NRTidalv2",
    "nsbh": "IMRPhenomNSBH",
}
_DEFAULT_APPROX = "IMRPhenomXP"

# ── Parameter name mapping (internal → bilby) ─────────────────────────────────
_RENAME = {"a1": "a_1", "a2": "a_2"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_bilby_params(params: Dict) -> Dict:
    """Rename keys and supply required defaults bilby needs."""
    p = dict(params)
    for old, new in _RENAME.items():
        if old in p and new not in p:
            p[new] = p.pop(old)
    # Aligned-spin defaults when tilt angles are not sampled
    p.setdefault("tilt_1", 0.0)
    p.setdefault("tilt_2", 0.0)
    p.setdefault("phi_12", 0.0)
    p.setdefault("phi_jl", 0.0)
    p.setdefault("chi_1", p.get("a_1", 0.0))
    p.setdefault("chi_2", p.get("a_2", 0.0))
    # Use absolute GPS time for bilby; geocent_time_gps is set by parameter_sampler
    if "geocent_time_gps" in p and "geocent_time" in p:
        p["geocent_time"] = p.pop("geocent_time_gps")
    return p


def _get_approximant(params: Dict) -> str:
    """Pick the waveform approximant: an explicit `approximant` if given, else
    the default for the event type (BNS/NSBH/BBH)."""
    explicit = params.get("approximant")
    if explicit:
        return explicit
    et = str(params.get("event_type", "bbh")).lower()
    for key in ("bns", "nsbh", "bbh"):
        if key in et:
            return _APPROXIMANTS[key]
    return _DEFAULT_APPROX


def _get_source_model(params: Dict):
    """Return the (frequency-domain source model, parameter-conversion) pair for
    the event type — binary-neutron-star for BNS, binary-black-hole otherwise."""
    et = str(params.get("event_type", "bbh")).lower()
    if "bns" in et:
        return _gwsource.lal_binary_neutron_star, _gwconv.convert_to_lal_binary_neutron_star_parameters
    return _gwsource.lal_binary_black_hole, _gwconv.convert_to_lal_binary_black_hole_parameters


def _make_psd(psd_dict: Optional[Dict]) -> PowerSpectralDensity:
    """Build a bilby PowerSpectralDensity from a {frequencies, psd} dict, or
    fall back to the aLIGO design curve when none is provided."""
    if psd_dict and "psd" in psd_dict and "frequencies" in psd_dict:
        freqs = np.asarray(psd_dict["frequencies"], dtype=np.float64)
        psd = np.maximum(np.asarray(psd_dict["psd"], dtype=np.float64), 1e-55)
        return PowerSpectralDensity(frequency_array=freqs, psd_array=psd)
    return PowerSpectralDensity.from_aligo()


_IFO_TEMPLATES: Dict[str, Interferometer] = {}


def _empty_ifo(detector_name: str) -> Interferometer:
    """Private copy of bilby's empty Interferometer for `detector_name`.

    get_empty_interferometer() re-reads the detector definition and its
    default PSD text file on every call (genfromtxt: ~84% of dataset
    generation time, worse on network filesystems). Build it once per
    detector and deep-copy, so every caller still gets its own mutable
    Interferometer with the same default PSD."""
    if detector_name not in _IFO_TEMPLATES:
        _IFO_TEMPLATES[detector_name] = get_empty_interferometer(detector_name)
    return copy.deepcopy(_IFO_TEMPLATES[detector_name])


def _make_ifo(detector_name: str, psd_dict: Optional[Dict],
              sample_rate: float, duration: float) -> Interferometer:
    """Construct an empty bilby Interferometer for `detector_name` with the
    given PSD and a [20 Hz, Nyquist] analysis band."""
    ifo = _empty_ifo(detector_name)
    ifo.power_spectral_density = _make_psd(psd_dict)
    ifo.minimum_frequency = 20.0
    ifo.maximum_frequency = sample_rate / 2.0
    return ifo


# ─────────────────────────────────────────────────────────────────────────────
# Real GWOSC noise cache
# ─────────────────────────────────────────────────────────────────────────────

class _GWOSCCache:
    """
    Loads pre-downloaded real noise segments and serves random windows.

    Expected format: directory of .npz files, each with keys:
      strain   – float64 array, length = sample_rate * full_segment_duration
      sample_rate – scalar
      detector    – str
    """

    def __init__(self, cache_dir: str = GWOSC_CACHE_DIR):
        self._cache_dir = Path(cache_dir)
        self._segments: Dict[str, List[np.ndarray]] = {}
        self._sample_rates: Dict[str, float] = {}
        self._loaded = False

    def _load(self):
        """Lazily load all cached real-noise `.npz` segments from the cache
        directory into memory (once); silently skips unreadable files."""
        if self._loaded:
            return
        self._loaded = True
        if not self._cache_dir.exists():
            return
        for npz_path in sorted(self._cache_dir.glob("*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=False)
                det = str(data["detector"])
                sr = float(data["sample_rate"])
                strain = data["strain"].astype(np.float64)
                self._segments.setdefault(det, []).append(strain)
                self._sample_rates[det] = sr
                _log.debug(f"Loaded GWOSC segment: {npz_path.name} ({det})")
            except Exception as e:
                _log.debug(f"Skipping {npz_path.name}: {e}")

    def get_segment(self, detector: str, duration: float,
                    sample_rate: float, rng: np.random.Generator) -> Optional[np.ndarray]:
        """Return a random window of real noise or None if unavailable."""
        self._load()
        segs = self._segments.get(detector)
        if not segs:
            return None
        seg = segs[rng.integers(len(segs))]
        n_want = int(sample_rate * duration)
        # Resample if stored at a different rate
        sr_stored = self._sample_rates.get(detector, sample_rate)
        if abs(sr_stored - sample_rate) > 1.0:
            from scipy.signal import resample
            n_stored = int(sr_stored * duration)
            if len(seg) < n_stored:
                return None
            start = rng.integers(0, max(1, len(seg) - n_stored))
            window = seg[start: start + n_stored]
            window = resample(window, n_want)
        else:
            if len(seg) < n_want:
                return None
            start = rng.integers(0, max(1, len(seg) - n_want))
            window = seg[start: start + n_want]
        return window.astype(np.float64)

    def available_detectors(self) -> List[str]:
        """Return the detectors for which real-noise segments are cached."""
        self._load()
        return list(self._segments.keys())


_gwosc_cache = _GWOSCCache()


# ─────────────────────────────────────────────────────────────────────────────
# Waveform generation
# ─────────────────────────────────────────────────────────────────────────────

class BilbyWaveformGenerator:
    """
    Generate projected time-domain GW strain for a single detector.

    Uses bilby's WaveformGenerator + ifo.get_detector_response for correct
    antenna patterns and inter-detector time delays.
    """

    def __init__(self, sample_rate: float, duration: float,
                 f_lower: float = 20.0, f_upper: Optional[float] = None,
                 f_ref: float = 50.0):
        self.sample_rate = int(sample_rate)
        self.duration = float(duration)
        self.n_samples = int(sample_rate * duration)
        self.f_lower = f_lower
        self.f_upper = f_upper or sample_rate / 2.0
        self.f_ref = f_ref

    def generate(self, params: Dict, detector: str) -> np.ndarray:
        """Return float64 time-domain strain, length = n_samples."""
        try:
            return self._generate(params, detector)
        except Exception as e:
            _log.warning(f"Waveform failed [{detector}] ({type(e).__name__}: {e}); zeros")
            return np.zeros(self.n_samples, dtype=np.float64)

    def _generate(self, params: Dict, detector: str) -> np.ndarray:
        """Generate the projected time-domain detector response: build the bilby
        waveform generator for the event type, project the polarizations onto
        `detector` over a merger-centered window, and IFFT to the time domain."""
        p = _to_bilby_params(params)
        source_model, param_conv = _get_source_model(p)
        approximant = _get_approximant(p)

        wfg = _BilbyWFG(
            duration=self.duration,
            sampling_frequency=self.sample_rate,
            frequency_domain_source_model=source_model,
            parameter_conversion=param_conv,
            waveform_arguments={
                "waveform_approximant": approximant,
                "reference_frequency": self.f_ref,
                "minimum_frequency": self.f_lower,
            },
        )
        polarizations = wfg.frequency_domain_strain(p)

        # Fixed window: [GPS_REF - T/2, GPS_REF + T/2] so the merger appears at
        # position (geocent_time_gps - start_time)*fs within the array, not always at the end.
        start_time = _GPS_REF - self.duration / 2.0
        ifo = _empty_ifo(detector)
        ifo.minimum_frequency = self.f_lower
        ifo.maximum_frequency = self.f_upper
        ifo.set_strain_data_from_zero_noise(
            sampling_frequency=self.sample_rate,
            duration=self.duration,
            start_time=start_time,
        )
        signal_fd = ifo.get_detector_response(
            waveform_polarizations=polarizations, parameters=p
        )
        # IFFT: bilby freq array 0..f_nyq with step 1/duration
        signal_td = np.fft.irfft(signal_fd, n=self.n_samples) * self.sample_rate
        return signal_td.astype(np.float64)

    # ── long signals ────────────────────────────────────────────────────────
    # A frequency-domain waveform sampled at 1/duration represents only
    # `duration` seconds: anything longer WRAPS circularly when irfft'd. Every
    # BNS (~135 s in band from 20 Hz), every NSBH (~18 s) and ~34% of BBH are
    # longer than the 4 s window, so they used to be stored with their whole
    # inspiral folded into it — unlike real data, which holds only the last
    # 4 s before merger (measured 2026-10-31).

    PRE_ROLL = 8.0   # s of signal kept before the window start (whitening support)
    POST = 2.0       # s between window end and long-segment end (Tukey roll-off)

    @staticmethod
    def chirp_time(params: Dict, f: float) -> float:
        """Leading-order (Newtonian) time from frequency `f` to merger [s]."""
        m1, m2 = float(params["mass_1"]), float(params["mass_2"])
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        return 2.18 * (1.21 / mc) ** (5.0 / 3.0) * (100.0 / f) ** (8.0 / 3.0)

    def fits_window(self, params: Dict) -> bool:
        """True if the whole signal from f_lower lies between the window start
        and the merger (with margin), i.e. the 4 s FD generation cannot wrap."""
        t_merger = self.duration / 2.0 + float(params.get("geocent_time", 0.0))
        return 1.2 * self.chirp_time(params, self.f_lower) + 0.25 <= t_merger

    def generate_long(self, params: Dict, detector: str) -> Tuple[np.ndarray, int]:
        """Projected raw strain over a long segment that contains the window,
        so a signal longer than the window is cut off — not wrapped — at the
        window start. Returns (strain [n_long], index of the window start).
        Whiten the whole array, then crop [i0, i0 + n_samples): the same
        order as real data (prepare_real whitens a long segment, then crops)."""
        p = _to_bilby_params(params)
        source_model, param_conv = _get_source_model(p)
        T = self.duration
        t_c = float(params.get("geocent_time", 0.0))
        # start the waveform PRE_ROLL s before the window (never above f_ref, so
        # the phase convention of the labels is unchanged)
        tau_needed = T / 2.0 + t_c + self.PRE_ROLL
        f_start = 100.0 * (2.18 * (1.21 / self._mc(params)) ** (5.0 / 3.0) / tau_needed) ** (3.0 / 8.0)
        f_start = float(min(self.f_ref, max(self.f_lower, f_start)))
        tau = self.chirp_time(params, f_start)
        need = 1.25 * tau + T / 2.0 - t_c + self.POST + 2.0
        D = float(2 ** int(np.ceil(np.log2(max(need, 2 * T)))))
        wfg = _BilbyWFG(
            duration=D, sampling_frequency=self.sample_rate,
            frequency_domain_source_model=source_model, parameter_conversion=param_conv,
            waveform_arguments={"waveform_approximant": _get_approximant(p),
                                "reference_frequency": self.f_ref,
                                "minimum_frequency": f_start})
        polarizations = wfg.frequency_domain_strain(p)
        window_start = _GPS_REF - T / 2.0
        long_start = window_start - (D - T - self.POST)
        ifo = _empty_ifo(detector)
        ifo.minimum_frequency = self.f_lower
        ifo.maximum_frequency = self.f_upper
        ifo.set_strain_data_from_zero_noise(sampling_frequency=self.sample_rate,
                                            duration=D, start_time=long_start)
        signal_fd = ifo.get_detector_response(waveform_polarizations=polarizations, parameters=p)
        n_long = int(round(D * self.sample_rate))
        long_td = np.fft.irfft(signal_fd, n=n_long) * self.sample_rate
        i0 = int(round((D - T - self.POST) * self.sample_rate))
        return long_td.astype(np.float64), i0

    @staticmethod
    def _mc(params: Dict) -> float:
        m1, m2 = float(params["mass_1"]), float(params["mass_2"])
        return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2

    # Keep old API compatible
    def generate_waveform(self, params: Dict, detector_name: str, psd=None) -> np.ndarray:
        """Back-compat alias for `generate` returning float32 strain."""
        return self.generate(params, detector_name).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Noise generation
# ─────────────────────────────────────────────────────────────────────────────

class BilbyNoiseGenerator:
    """
    Generate detector noise: real GWOSC segments when available, else bilby
    synthetic colored Gaussian noise from a design-sensitivity PSD.
    """

    def __init__(self, sample_rate: float, duration: float,
                 real_noise_prob: float = 0.0,
                 cache_dir: Optional[str] = None):
        self.sample_rate = int(sample_rate)
        self.duration = float(duration)
        self.n_samples = int(sample_rate * duration)
        self.real_noise_prob = float(real_noise_prob)
        self._rng = np.random.default_rng()
        if cache_dir:
            global _gwosc_cache
            _gwosc_cache = _GWOSCCache(cache_dir)

    def generate(self, detector: str, psd_dict: Optional[Dict] = None,
                 seed: Optional[int] = None) -> np.ndarray:
        """
        Return float64 noise array of length n_samples.
        Uses real GWOSC noise with probability real_noise_prob.
        """
        rng = np.random.default_rng(seed)
        if self.real_noise_prob > 0 and rng.uniform() < self.real_noise_prob:
            seg = _gwosc_cache.get_segment(detector, self.duration, self.sample_rate, rng)
            if seg is not None:
                return seg
        return self._synthetic_noise(detector, psd_dict, seed)

    def _synthetic_noise(self, detector: str, psd_dict: Optional[Dict],
                         seed: Optional[int]) -> np.ndarray:
        """Generate synthetic colored Gaussian noise via bilby, falling back to
        the manual FFT generator if bilby fails."""
        try:
            return self._bilby_noise(detector, psd_dict, seed)
        except Exception as e:
            _log.warning(f"Bilby noise failed ({e}); FFT fallback")
            return self._fft_noise(psd_dict, seed)

    def _bilby_noise(self, detector: str, psd_dict: Optional[Dict],
                     seed: Optional[int]) -> np.ndarray:
        """Draw colored Gaussian noise from the detector PSD using bilby's
        `set_strain_data_from_power_spectral_density`."""
        if seed is not None:
            np.random.seed(seed % 2 ** 32)  # legacy API takes 32-bit seeds only
            # bilby 2.x draws noise from its own generator, which
            # np.random.seed does not reach: without this, --random-seed
            # reproduced parameters but never the noise
            bilby.core.utils.random.seed(seed)
        ifo = _make_ifo(detector, psd_dict, self.sample_rate, self.duration)
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=self.sample_rate,
            duration=self.duration,
            start_time=0.0,
        )
        return ifo.strain_data.time_domain_strain.astype(np.float64)

    def _fft_noise(self, psd_dict: Optional[Dict], seed: Optional[int]) -> np.ndarray:
        """Manual colored-noise fallback: color white Gaussian noise by the PSD
        in the frequency domain with the correct E[|X|^2]=n*S_n scaling."""
        rng = np.random.default_rng(seed)
        n = self.n_samples
        white = rng.standard_normal(n)
        if psd_dict and "psd" in psd_dict and "frequencies" in psd_dict:
            freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
            psd_interp = np.interp(freqs, psd_dict["frequencies"], psd_dict["psd"])
            psd_interp = np.maximum(psd_interp, 1e-55)
            wf = np.fft.rfft(white)
            # Correct scaling: E[|X[k]|²] = n·S_n(f_k), so amplitude = sqrt(n·S_n/2)
            # Using sample_rate/2 would be 2× wrong for duration > 1s
            wf *= np.sqrt(psd_interp * n / 2.0)
            white = np.fft.irfft(wf, n=n)
        return white.astype(np.float64)

    # Keep old API compatible
    def generate_colored_noise(self, psd_dict=None, seed=None) -> np.ndarray:
        """Back-compat alias: generate one H1 noise realization."""
        return self.generate("H1", psd_dict, seed)

    def add_glitches(self, noise: np.ndarray, glitch_prob: float = 0.1,
                     n_glitches: int = 3) -> np.ndarray:
        """Inject simple burst glitches for data augmentation."""
        if np.random.rand() > glitch_prob:
            return noise
        rng = np.random.default_rng()
        out = noise.copy()
        n_g = rng.integers(1, n_glitches + 1)
        noise_std = float(np.std(noise)) or 1.0
        for _ in range(n_g):
            center = rng.integers(0, len(out))
            width = rng.integers(20, 200)
            amp = rng.uniform(2.0, 8.0) * noise_std
            t = np.arange(-width, width + 1)
            glitch = amp * np.exp(-t ** 2 / (2 * (width / 3) ** 2)) * np.sin(2 * np.pi * t / width)
            lo = max(0, center - width)
            hi = min(len(out), center + width + 1)
            g_lo = lo - (center - width)
            g_hi = g_lo + (hi - lo)
            out[lo:hi] += glitch[g_lo:g_hi]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Whitening
# ─────────────────────────────────────────────────────────────────────────────

class BilbyPreprocessor:
    """
    Whiten strain using bilby's matched-filter normalisation:
        h_white(f) = h(f) / sqrt(PSD(f) / (4 df))
    This gives unit-variance white Gaussian noise when PSD is correct.
    """

    def __init__(self, sample_rate: float, duration: float,
                 f_low: float = 20.0, f_high: Optional[float] = None):
        self.sample_rate = int(sample_rate)
        self.duration = float(duration)
        self.f_low = f_low
        self.f_high = f_high or sample_rate / 2.0
        self.n_samples = int(sample_rate * duration)

    def preprocess(self, strain: np.ndarray, psd_dict: Optional[Dict],
                   detector: str = "H1", whiten: bool = True, **kwargs) -> np.ndarray:
        """Whiten `strain` with the detector PSD (bilby matched-filter norm),
        or return it unchanged when `whiten=False`."""
        if not whiten:
            return np.asarray(strain, dtype=np.float32)
        try:
            return self._bilby_whiten(strain, psd_dict, detector)
        except Exception as e:
            _log.warning(f"Whitening failed ({e}); returning raw strain")
            return np.asarray(strain, dtype=np.float32)

    def _bilby_whiten(self, strain: np.ndarray, psd_dict: Optional[Dict],
                      detector: str = "H1") -> np.ndarray:
        """Whiten one detector's time-domain strain via bilby's
        `whitened_time_domain_strain` (unit-variance output for a correct PSD),
        padded/cropped back to the input length."""
        strain64 = np.asarray(strain, dtype=np.float64)
        n = len(strain64)
        dur = n / self.sample_rate
        ifo = _make_ifo(detector, psd_dict, self.sample_rate, dur)
        ifo.strain_data.set_from_time_domain_strain(
            strain64, sampling_frequency=self.sample_rate,
            duration=dur, start_time=0.0,
        )
        whitened = ifo.whitened_time_domain_strain
        if len(whitened) > n:
            whitened = whitened[:n]
        elif len(whitened) < n:
            whitened = np.pad(whitened, (0, n - len(whitened)))
        return whitened.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Signal injection  (SNR is measured, not targeted)
# ─────────────────────────────────────────────────────────────────────────────

class BilbySignalInjector:
    """
    Inject GW signals into noise and measure matched-filter SNR.

    No SNR targeting. The waveform is injected at its natural amplitude
    (set by luminosity_distance). The caller receives the measured SNR and
    decides whether to keep the sample.
    """

    def __init__(self, sample_rate: float, duration: float,
                 f_lower: float = 20.0, f_ref: float = 50.0):
        self.sample_rate = int(sample_rate)
        self.duration = float(duration)
        self.n_samples = int(sample_rate * duration)
        self.f_lower = f_lower
        self.f_ref = f_ref
        self.wfg = BilbyWaveformGenerator(sample_rate, duration,
                                          f_lower=f_lower, f_ref=f_ref)

    def inject(self, noise: np.ndarray, params: Dict,
               detector: str,
               psd_dict: Optional[Dict] = None) -> Tuple[np.ndarray, float]:
        """
        Inject one signal into noise.

        Returns:
            strain  – float64 array (noise + signal)
            snr     – measured optimal matched-filter SNR (0 if waveform failed)
        """
        signal = self.wfg.generate(params, detector).astype(np.float64)
        signal = self._resize(signal, len(noise))
        snr = self.measure_snr(signal, psd_dict)
        return (noise.astype(np.float64) + signal), snr

    def inject_multiple(self, noise: np.ndarray, params_list: List[Dict],
                        detector: str,
                        psd_dict: Optional[Dict] = None) -> Tuple[np.ndarray, List[float]]:
        """
        Inject multiple signals into noise.

        Returns combined strain and per-signal SNR list.
        """
        combined = noise.astype(np.float64)
        snrs: List[float] = []
        for p in params_list:
            sig = self.wfg.generate(p, detector).astype(np.float64)
            sig = self._resize(sig, len(noise))
            snrs.append(self.measure_snr(sig, psd_dict))
            combined += sig
        return combined, snrs

    def measure_snr(self, waveform: np.ndarray,
                    psd_dict: Optional[Dict]) -> float:
        """
        Optimal matched-filter SNR: sqrt(4 ∫ |h̃(f)|²/S_n(f) df).
        """
        try:
            return self._snr_integral(waveform, psd_dict)
        except Exception as e:
            _log.debug(f"SNR measurement failed: {e}")
            return 0.0

    def _snr_integral(self, waveform: np.ndarray,
                      psd_dict: Optional[Dict]) -> float:
        """Optimal matched-filter SNR sqrt(4 int |h(f)|^2 / S_n(f) df) above
        f_lower, using the continuous-FT normalization of rfft."""
        n = len(waveform)
        dt = 1.0 / self.sample_rate
        freqs = np.fft.rfftfreq(n, dt)
        # rfft returns DFT coefficients X[k] = sum_n x[n]*exp(-2πikn/N)
        # which approximates H_c(f_k) * sample_rate (continuous FT × sample_rate).
        # The matched-filter SNR uses the continuous FT:
        #   ρ² = 4·df·∑ |H_c(f_k)|² / S_n(f_k) = 4·df·∑ |rfft(h)[k]|² / (fs²·S_n(f_k))
        hf = np.fft.rfft(waveform.astype(np.float64)) / self.sample_rate
        if psd_dict and "psd" in psd_dict and "frequencies" in psd_dict:
            psd_vals = np.interp(freqs, psd_dict["frequencies"], psd_dict["psd"])
        else:
            psd_obj = PowerSpectralDensity.from_aligo()
            psd_vals = np.interp(freqs, psd_obj.frequency_array, psd_obj.psd_array)
        psd_vals = np.maximum(psd_vals, 1e-55)
        mask = freqs >= self.f_lower
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0 / self.duration
        snr_sq = 4.0 * df * float(np.sum(np.abs(hf[mask]) ** 2 / psd_vals[mask]))
        return float(np.sqrt(max(snr_sq, 0.0)))

    def _resize(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        """Crop or zero-pad `arr` to exactly `target_len` samples."""
        if len(arr) == target_len:
            return arr
        if len(arr) > target_len:
            return arr[:target_len]
        return np.pad(arr, (0, target_len - len(arr)))

    # ── Compatibility shim ──────────────────────────────────────────────────

    def inject_signal(self, noise, params, detector_name, psd_dict=None):
        """Compatibility shim: inject one signal and return (float32 strain,
        metadata dict with detector / measured SNR / injection time)."""
        strain, snr = self.inject(noise, params, detector_name, psd_dict)
        meta = {"detector": detector_name, "actual_snr": snr,
                "injection_time": params.get("geocent_time_gps",
                                              params.get("geocent_time", 0.0))}
        return strain.astype(np.float32), meta

    def inject_overlapping_signals(self, noise, signal_params_list, detector_name,
                                   psd_dict=None):
        """Compatibility shim: inject multiple overlapping signals into `noise`
        and return the combined float32 strain plus per-signal metadata."""
        combined, snrs = self.inject_multiple(noise, signal_params_list,
                                               detector_name, psd_dict)
        meta_list = [{"detector": detector_name, "signal_index": i,
                      "actual_snr": snrs[i]} for i in range(len(snrs))]
        return combined.astype(np.float32), meta_list


# ── Convenience: get default analytic PSD for a detector ────────────────────

def get_default_psd(detector: str) -> Dict:
    """Return analytic aLIGO/AdVirgo PSD as {frequencies, psd} dict."""
    try:
        if detector == "V1":
            psd_obj = PowerSpectralDensity.from_advancedvirgo()
        else:
            psd_obj = PowerSpectralDensity.from_aligo()
        return {
            "frequencies": psd_obj.frequency_array,
            "psd": psd_obj.psd_array,
        }
    except Exception:
        # Minimal fallback: flat 1e-46 PSD
        freqs = np.linspace(1.0, 2048.0, 4096)
        return {"frequencies": freqs, "psd": np.full(len(freqs), 1e-46)}


# ── Aliases for dataset_generator import compatibility ──────────────────────
WaveformGenerator = BilbyWaveformGenerator
NoiseGenerator = BilbyNoiseGenerator
SignalInjector = BilbySignalInjector
DataPreprocessor = BilbyPreprocessor
