"""
Inference-time preprocessing: raw detector strain -> the exact representation
LeanNPE was trained on.

Two paths, both delegating to the SAME code used to build the training data
(no re-implementation of the physics):

  simulated / design-PSD path
      whitening via ahsd.data.bilby_pipeline.BilbyPreprocessor — identical to
      dataset_generator's whitening of stored training strain.

  real-detector path
      the noise-bank recipe (scripts/download_gwosc_noise_bank.py +
      experiments/remix_data.RemixDataset): highpass 15 Hz, measured ASD
      (median estimator, robust to the signal being in-segment), MANUAL
      whitening  irfft(rfft(x)/ASD), 2 s edge trim, unit noise floor from an
      off-source region. This is the representation the real-noise fine-tune
      saw during training.

The 4 s analysis window is centered on the trigger time, so the model's
geocent_time posterior is the offset of the merger from the trigger —
the same convention as training (window centered on GPS_REF, geocent_time
relative to GPS_REF).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ahsd.data.bilby_pipeline import BilbyPreprocessor, get_default_psd, highpass_gain

log = logging.getLogger(__name__)

DETECTORS = ["H1", "L1", "V1"]
SAMPLE_RATE = 4096
DURATION = 4.0
T_LEN = int(SAMPLE_RATE * DURATION)
FETCH_SECONDS = 80          # context fetched around the trigger (PSD estimation and
                            # the 28 s low-rate stream before the window)
LOWRATE_SECONDS = 28        # multi-rate models: seconds before the window ...
LOWRATE_RATE = 512          # ... at this rate (must match the training dataset)
EDGE_TRIM_S = 2             # whitening filter edge artifacts to discard
HIGHPASS_HZ = 15.0
ASD_FFTLENGTH = 8
ASD_OVERLAP = 4
GLITCH_SIGMA = 50.0         # whitened excursion above any signal in the training
                            # sets (per-sample peak max 47.5σ, BBH at SNR 524;
                            # Gaussian max ≈ 4.5σ): a detector above it is absent


@dataclass
class PreparedData:
    """Model-ready strain plus everything needed to interpret/reconstruct it."""
    strain: np.ndarray                      # [3, T_LEN] float32, model input order H1,L1,V1
    trigger_gps: Optional[float]            # absolute GPS of the window center (None for simulated)
    detectors_present: List[str]            # detectors with real data (others are white-noise fill)
    asds: Dict[str, np.ndarray]             # per-detector ASD on the 4 s rfft grid (whitening used)
    psds: Dict[str, dict]                   # {frequencies, psd} dicts (bilby/dynesty-compatible)
    source: str                             # "simulated" | "real"
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    quality: Dict[str, dict] = field(default_factory=dict)  # per-detector checks
    strain_lo: Optional[np.ndarray] = None  # [3, 28 s @ 512 Hz] context before the window (multi-rate models)


# ── quality checks ───────────────────────────────────────────────────────────

def _quality_checks(det: str, x: np.ndarray, warnings: List[str]) -> dict:
    """Checks on a whitened, unit-floor 4 s window. Appends human-readable
    warnings; returns the raw numbers for the metadata record."""
    q = {}
    q["finite"] = bool(np.isfinite(x).all())
    if not q["finite"]:
        warnings.append(f"{det}: NaN/Inf in strain (sanitized to 0)")
    q["std"] = float(np.nanstd(x))
    if not (0.5 < q["std"] < 3.0):
        warnings.append(f"{det}: whitened std {q['std']:.2f} far from 1 — "
                        "PSD mismatch or non-stationary data (out-of-distribution)")
    q["max_abs"] = float(np.nanmax(np.abs(x))) if x.size else 0.0
    if q["max_abs"] > 40.0:
        warnings.append(f"{det}: |strain| peaks at {q['max_abs']:.0f}σ — likely glitch/saturation")
    # excess kurtosis of the off-center half: Gaussian noise ≈ 0
    edge = np.concatenate([x[: T_LEN // 4], x[-T_LEN // 4:]])
    m2 = np.mean(edge ** 2) + 1e-12
    q["excess_kurtosis"] = float(np.mean(edge ** 4) / m2 ** 2 - 3.0)
    if q["excess_kurtosis"] > 3.0:
        warnings.append(f"{det}: off-source excess kurtosis {q['excess_kurtosis']:.1f} — "
                        "non-Gaussian noise (glitches) in window")
    # saturation / dead channel: long runs of identical samples
    q["repeated_frac"] = float(np.mean(np.diff(x) == 0.0))
    if q["repeated_frac"] > 0.01:
        warnings.append(f"{det}: {q['repeated_frac']:.1%} repeated samples — dropouts/saturation?")
    return q


def _white_noise_fill(rng: np.random.Generator) -> np.ndarray:
    """Stand-in for a missing detector: unit white noise = whitened noise with
    no signal, the closest in-distribution input the model can receive."""
    return rng.standard_normal(T_LEN).astype(np.float32)


# ── real-data path ───────────────────────────────────────────────────────────

def _whiten_real_segment(x: np.ndarray, sr_in: float, trigger_offset_s: float,
                         asd_freqs: Optional[np.ndarray] = None,
                         asd_vals: Optional[np.ndarray] = None):
    """Noise-bank whitening recipe applied to one detector's raw segment.

    x: raw strain, sr_in Hz, trigger at trigger_offset_s seconds from x[0].
    asd_freqs/asd_vals: user-supplied ASD; when None, measured from x with the
    median estimator (robust to the signal being in the segment).

    Returns (window[T_LEN] float32 unit-floor, asd_on_4s_grid, norm_factor,
    low-rate context stream before the window or None if the segment is too
    short).
    """
    from gwpy.timeseries import TimeSeries

    ts = TimeSeries(np.asarray(x, dtype=np.float64), sample_rate=sr_in)
    if abs(sr_in - SAMPLE_RATE) > 1.0:
        ts = ts.resample(SAMPLE_RATE)
    ts = ts.highpass(HIGHPASS_HZ)

    if asd_vals is None:
        asd = ts.asd(fftlength=ASD_FFTLENGTH, overlap=ASD_OVERLAP, method="median")
        asd_freqs, asd_vals = asd.frequencies.value, asd.value
    else:
        # a supplied ASD describes unfiltered strain; the data here is filtered
        # and every ASD below is post-filter, as the measured one is
        asd_freqs = np.asarray(asd_freqs, float)
        asd_vals = np.asarray(asd_vals, float) * highpass_gain(asd_freqs, SAMPLE_RATE, HIGHPASS_HZ)

    v = ts.value
    n = len(v)
    freqs_seg = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    asd_seg = np.interp(freqs_seg, asd_freqs, asd_vals,
                        left=asd_vals[0], right=asd_vals[-1])
    wf = np.fft.rfft(v) / np.maximum(asd_seg, 1e-30)
    # Below the highpass corner both numerator and ASD are suppressed and
    # their ratio is dominated by leakage/filter transients — measured to
    # carry ~100% of the output power as sub-15 Hz junk. Zero it: the band
    # is outside the analysis range (f_min = 20 Hz) and removing it restores
    # a unit white floor (verified: block stds 0.97-1.08, kurtosis 0.01).
    wf[freqs_seg < HIGHPASS_HZ + 3.0] = 0.0
    w = np.fft.irfft(wf, n=n)

    trim = EDGE_TRIM_S * SAMPLE_RATE
    w = w[trim:-trim]
    trig_i = int(round((trigger_offset_s - EDGE_TRIM_S) * SAMPLE_RATE))

    # unit noise floor measured OFF-SOURCE (exclude ±2 s around the trigger so
    # a loud signal does not inflate the normalization)
    mask = np.ones(len(w), dtype=bool)
    mask[max(0, trig_i - 2 * SAMPLE_RATE): trig_i + 2 * SAMPLE_RATE] = False
    norm = float(w[mask].std()) if mask.any() else float(w.std())
    w = (w / (norm + 1e-30)).astype(np.float32)

    i0 = trig_i - T_LEN // 2
    if i0 < 0 or i0 + T_LEN > len(w):
        raise ValueError(f"trigger too close to segment edge (need ±{DURATION/2 + EDGE_TRIM_S}s)")
    window = w[i0: i0 + T_LEN]
    try:
        from ahsd.data.bilby_pipeline import to_lowrate
        lo = to_lowrate(w, i0, SAMPLE_RATE, LOWRATE_SECONDS, LOWRATE_RATE)
    except ValueError:
        lo = None

    freqs_win = np.fft.rfftfreq(T_LEN, 1.0 / SAMPLE_RATE)
    asd_win = np.interp(freqs_win, asd_freqs, asd_vals,
                        left=asd_vals[0], right=asd_vals[-1]).astype(np.float32)
    return window, asd_win, norm, lo


def prepare_real(strain: Dict[str, np.ndarray], trigger_gps: float,
                 segment_start_gps: Dict[str, float],
                 sample_rates: Dict[str, float],
                 psds: Optional[Dict[str, dict]] = None,
                 seed: int = 0, drop_glitches: bool = True) -> PreparedData:
    """Real (or externally supplied) detector data -> model representation.

    strain: raw (unwhitened) strain per detector, ideally >= 16 s around the
    trigger for a stable in-segment ASD estimate (64 s recommended).
    psds: optional user-supplied {det: {frequencies, psd}}; otherwise the ASD
    is measured from the data (median estimator).
    drop_glitches: a detector whose window or context stream exceeds
    GLITCH_SIGMA is replaced by white noise, as a dropped detector is in
    training, provided another detector remains (GW170817: L1 glitch at 137σ
    gave chirp mass 1.38 and merger -0.58 s; without L1, 1.205 and +0.10 s).
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    warnings: List[str] = []
    out = np.empty((3, T_LEN), dtype=np.float32)
    out_lo = np.empty((3, LOWRATE_SECONDS * LOWRATE_RATE), dtype=np.float32)
    have_lo = True
    asds, psd_out, quality = {}, {}, {}
    present = []

    for i, det in enumerate(DETECTORS):
        if det not in strain or strain[det] is None:
            out[i] = _white_noise_fill(rng)
            out_lo[i] = rng.standard_normal(out_lo.shape[1]).astype(np.float32)
            warnings.append(f"{det}: no data — filled with unit white noise "
                            "(posterior uses remaining detectors only)")
            continue
        sr = float(sample_rates.get(det, SAMPLE_RATE))
        off = trigger_gps - float(segment_start_gps[det])
        user_asd = None
        if psds and det in psds:
            p = psds[det]
            user_asd = (np.asarray(p["frequencies"], float),
                        np.sqrt(np.asarray(p["psd"], float)))
        win, asd_win, norm, lo = _whiten_real_segment(
            strain[det], sr, off,
            asd_freqs=user_asd[0] if user_asd else None,
            asd_vals=user_asd[1] if user_asd else None)
        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
        quality[det] = _quality_checks(det, win, warnings)
        quality[det]["whitening_norm"] = norm
        out[i] = win
        if lo is None:
            have_lo = False
        else:
            out_lo[i] = np.nan_to_num(lo, nan=0.0, posinf=0.0, neginf=0.0)
        asds[det] = asd_win
        freqs_win = np.fft.rfftfreq(T_LEN, 1.0 / SAMPLE_RATE)
        # asds stay post-filter (asd_bands, whitening); psds describe the unfiltered
        # strain the likelihood reads, so the filter's gain comes out
        gain = np.maximum(highpass_gain(freqs_win, SAMPLE_RATE, HIGHPASS_HZ), 1e-3)
        psd_out[det] = {"frequencies": freqs_win, "psd": (asd_win.astype(np.float64) / gain) ** 2}
        present.append(det)

    if not present:
        raise ValueError("no detector data supplied")
    glitchy = [d for d in present
               if max(quality[d]["max_abs"],
                      float(np.abs(out_lo[DETECTORS.index(d)]).max()) if have_lo else 0.0) > GLITCH_SIGMA]
    for d in glitchy:
        if not drop_glitches or len(present) == 1:
            warnings.append(f"{d}: excursion above {GLITCH_SIGMA:.0f}σ kept (no other detector "
                            "to fall back on) — posterior unreliable")
            continue
        i = DETECTORS.index(d)
        out[i] = _white_noise_fill(rng)
        out_lo[i] = rng.standard_normal(out_lo.shape[1]).astype(np.float32)
        asds.pop(d); psd_out.pop(d); present.remove(d)
        quality[d]["dropped_glitch"] = True
        warnings.append(f"{d}: excursion above {GLITCH_SIGMA:.0f}σ (glitch) — dropped and filled "
                        "with unit white noise; gate or subtract the glitch to use this detector")
    return PreparedData(strain=out, trigger_gps=float(trigger_gps),
                        detectors_present=present, asds=asds, psds=psd_out,
                        source="real", warnings=warnings,
                        timings={"preprocess_s": round(time.time() - t0, 3)},
                        quality=quality, strain_lo=out_lo if have_lo else None)


_DESIGN_ASD_CACHE: Dict[tuple, np.ndarray] = {}


def _design_asd(det: str, design_asd_dir: str) -> Optional[np.ndarray]:
    """Load (and cache) detector `det`'s design ASD from `design_asd_dir`, used
    as the reference in the design-vs-measured sensitivity summary. None if
    missing."""
    key = (det, design_asd_dir)
    if key not in _DESIGN_ASD_CACHE:
        p = Path(design_asd_dir) / f"design_asd_{det}.npy"
        _DESIGN_ASD_CACHE[key] = np.load(p).astype(np.float64) if p.exists() else None
    return _DESIGN_ASD_CACHE[key]


def compute_asd_bands(prepared: PreparedData, psd_bands: int = 16,
                      design_asd_dir: str = "data/noise_bank",
                      clamp: float = 50.0) -> np.ndarray:
    """Per-detector sensitivity summary for a PSD-conditioned LeanNPE, matching
    the training definition (RemixDataset._asd_bands): band-mean of
    log(clip(design_ASD / measured_ASD)). 0 for design-whitened / missing
    detectors. Returns [3, psd_bands] float32."""
    freqs = np.fft.rfftfreq(T_LEN, 1.0 / SAMPLE_RATE)
    edges = np.geomspace(20.0, SAMPLE_RATE / 2.0, psd_bands + 1)
    slices = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        slices.append(idx if len(idx) else np.array([int(np.argmin(np.abs(freqs - lo)))]))
    ab = np.zeros((3, psd_bands), dtype=np.float32)
    for i, det in enumerate(DETECTORS):
        design = _design_asd(det, design_asd_dir)
        if det not in prepared.asds or design is None:
            continue  # white-fill / no design ASD -> design-like (0)
        meas = np.asarray(prepared.asds[det], dtype=np.float64)
        n = min(len(design), len(meas))
        ratio = np.clip(design[:n] / np.maximum(meas[:n], 1e-30), 1.0 / clamp, clamp)
        logr = np.log(ratio)
        ab[i] = [float(logr[s[s < n]].mean()) if (s < n).any() else 0.0 for s in slices]
    return ab


def likelihood_seconds(samples: np.ndarray, f_min: float = 20.0, minimum: float = 4.0,
                       quantile: float = 0.005) -> float:
    """Likelihood segment length (s) for a posterior's samples [N, >=2
    masses first, PARAM_NAMES order], as LVK sizes analyses: the
    leading-order chirp time from f_min of the lightest plausible chirp mass,
    plus the time from the earliest plausible merger to the window end (2 s
    minus the merger-time offset; 3.6 s if unknown) and 0.2 s, rounded up to
    a power of two (>= minimum).

    A fixed 16 s let a long, light template fit a transient in the 12 s
    before the window: GW190513 with LVK PSDs went to chirp mass 7.0 (LVK
    29.9) for every seed; with a 4 s segment it gives 31.6 [27.7, 35.7]."""
    m1, m2 = samples[:, 0], samples[:, 1]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    mc_lo = float(np.quantile(mc[np.isfinite(mc)], quantile))
    tau = 5.0 / 256.0 * (np.pi * f_min) ** (-8.0 / 3.0) * (mc_lo * 4.925490947e-6) ** (-5.0 / 3.0)
    to_end = 3.6
    if samples.shape[1] > 8:
        from ahsd.models.lean_npe import PARAM_NAMES
        tc = samples[:, PARAM_NAMES.index("geocent_time")]
        tc = tc[np.isfinite(tc)]
        if len(tc):
            to_end = DURATION / 2 - float(np.quantile(tc, quantile))
    need = tau + to_end + 0.2
    return float(max(minimum, 2.0 ** np.ceil(np.log2(need))))


def raw_segment(strain: Dict[str, np.ndarray], segment_start_gps: Dict[str, float],
                center_gps: float, seconds: float = DURATION) -> Dict[str, np.ndarray]:
    """Unwhitened strain ending at the window end (center + 2 s), `seconds`
    long, per detector (4096 Hz input, e.g. from fetch_gwosc): what
    importance sampling and dynesty need. For a multi-rate model pass
    seconds = 4 + 28 so the likelihood covers the context it read."""
    out = {}
    n = int(round(seconds * SAMPLE_RATE))
    for det, x in strain.items():
        i1 = int(round((center_gps + DURATION / 2 - segment_start_gps[det]) * SAMPLE_RATE))
        if i1 - n < 0 or i1 > len(x):
            raise ValueError(f"{det}: data does not cover {seconds:.0f} s before the window end")
        out[det] = np.asarray(x[i1 - n:i1], np.float64)
    return out


def _cached_gwosc(det: str, t0: float, t1: float):
    """A GWOSC strain file for `det` covering [t0, t1] from the local download
    cache (no network), cropped and at SAMPLE_RATE; None if none is cached."""
    import re
    try:
        from astropy.utils.data import download_file, get_cached_urls
        from gwpy.timeseries import TimeSeries
    except ImportError:
        return None
    pat = re.compile(rf"/{det[0]}-{det}_(?:GWOSC|LOSC)_[^/]*?-(\d+)-(\d+)\.hdf5$")
    for url in get_cached_urls():
        m = pat.search(url)
        if not m or not (int(m[1]) <= t0 and int(m[1]) + int(m[2]) >= t1):
            continue
        try:
            ts = TimeSeries.read(download_file(url, cache=True), format="hdf5.gwosc").crop(t0, t1)
            if abs(float(ts.sample_rate.value) - SAMPLE_RATE) > 1e-6:
                ts = ts.resample(SAMPLE_RATE)
            return ts
        except Exception:
            continue
    return None


MIN_BEFORE_S, MIN_AFTER_S = 34.0, 4.0   # valid data a gappy detector must keep around the trigger


def _finite_run_around(x: np.ndarray, i: int):
    """(start, stop) of the contiguous finite stretch of x containing index i,
    or None if x[i] is not finite."""
    if not (0 <= i < len(x)) or not np.isfinite(x[i]):
        return None
    bad = ~np.isfinite(x)
    lo = np.flatnonzero(bad[:i])
    hi = np.flatnonzero(bad[i:])
    return (int(lo[-1]) + 1 if len(lo) else 0, i + int(hi[0]) if len(hi) else len(x))


def fetch_gwosc(event_or_gps, detectors: Optional[List[str]] = None,
                fetch_seconds: int = FETCH_SECONDS, strict: bool = False,
                required: Optional[List[str]] = None):
    """Download raw strain around a GWOSC event name or GPS time.

    Returns (trigger_gps, strain dict, segment_start dict, sample_rate dict,
    detectors found). GPS lookup goes through ahsd.data.gwtc_loader.GWTCLoader
    (cached GWOSC catalog).
    """
    from gwpy.timeseries import TimeSeries

    if isinstance(event_or_gps, str):
        from ahsd.data.gwtc_loader import GWTCLoader
        gps = GWTCLoader()._get_event_gps_time(event_or_gps)
        if gps is None:
            raise ValueError(f"could not resolve GPS time for {event_or_gps!r}")
    else:
        gps = float(event_or_gps)

    dets = detectors or DETECTORS
    strain, starts, srs, found, failed = {}, {}, {}, [], []
    for det in dets:
        try:
            for attempt in range(3):          # GWOSC downloads fail transiently
                try:
                    ts = TimeSeries.fetch_open_data(det, gps - fetch_seconds / 2,
                                                    gps + fetch_seconds / 2,
                                                    sample_rate=SAMPLE_RATE, cache=True)
                    break
                except Exception:
                    if attempt == 2:
                        # gwpy asks the GWOSC API for the file list on every
                        # fetch, even when the files are cached; when the API
                        # is down (2026-11-04: timeouts from every node) read a
                        # cached file covering the span directly
                        ts = _cached_gwosc(det, gps - fetch_seconds / 2, gps + fetch_seconds / 2)
                        if ts is None:
                            raise
                        log.warning(f"{det}: GWOSC unreachable; using the locally cached file")
                        break
                    time.sleep(5 * (attempt + 1))
            # GWOSC serves non-observing detectors as NaN-filled segments
            # (e.g. H1 during GW190425); treat them as absent so the window
            # gets the same white-noise fill as a dropped detector in
            # training, not a zeroed channel the model has never seen.
            x, t0, sr = ts.value, float(ts.t0.value), float(ts.sample_rate.value)
            nan_frac = float(np.isnan(x).mean())
            if nan_frac > 0.01:
                # A gap ANYWHERE in the fetched span used to drop the detector:
                # GW190513's H1 has 25% NaN in the 80 s fetch but valid data
                # around the event (LVK analysed it), and without H1 the
                # correction locked onto a wrong low-mass mode. Keep the
                # finite stretch around the trigger when it covers what the
                # network and likelihood read (32 s before the window end).
                cut = _finite_run_around(x, int(round((gps - t0) * sr)))
                need_lo, need_hi = int(MIN_BEFORE_S * sr), int(MIN_AFTER_S * sr)
                i_trig = int(round((gps - t0) * sr))
                if cut is None or i_trig - cut[0] < need_lo or cut[1] - i_trig < need_hi:
                    log.warning(f"{det}: {nan_frac:.0%} NaN in GWOSC data (not observing) — treated as absent")
                    continue
                log.warning(f"{det}: {nan_frac:.0%} NaN in the fetched span; using the "
                            f"{(cut[1] - cut[0]) / sr:.0f} s of valid data around the trigger")
                x, t0 = x[cut[0]:cut[1]], t0 + cut[0] / sr
            strain[det] = x
            starts[det] = t0
            srs[det] = sr
            found.append(det)
        except Exception as e:
            log.warning(f"{det}: GWOSC fetch failed ({str(e)[:80]})")
            failed.append(det)
    if not found:
        raise RuntimeError(f"no open data available around GPS {gps}")
    # strict: a failed download of a detector that is needed (`required`;
    # default all) is an error. With GWOSC unreachable a detector that was
    # not observing (no cached file ever) looks the same as a failure, so the
    # caller names the detectors the analysis needs (e.g. those LVK analysed)
    if required is not None:
        failed = [d for d in failed if d in required]
    if strict and failed:
        # a download ERROR is not a non-observing detector: analysing without
        # it silently degrades the result (2026-11-04: GWOSC timeouts on the
        # compute nodes left three events with Virgo alone)
        raise RuntimeError(f"GWOSC fetch failed for {failed}; retry (prefetch on a node with internet)")
    return gps, strain, starts, srs, found


# ── simulated path ───────────────────────────────────────────────────────────

def prepare_simulated(strain: Dict[str, np.ndarray],
                      psds: Optional[Dict[str, dict]] = None,
                      already_whitened: bool = False,
                      seed: int = 0) -> PreparedData:
    """Simulated strain (the dataset pipeline's 4 s window centered on
    GPS_REF) -> model representation, whitened by the SAME BilbyPreprocessor
    used in dataset generation. geocent_time posteriors are offsets from the
    window center."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    prep = BilbyPreprocessor(SAMPLE_RATE, DURATION)
    psds = psds or {d: get_default_psd(d) for d in DETECTORS}
    warnings: List[str] = []
    out = np.empty((3, T_LEN), dtype=np.float32)
    asds, quality, present = {}, {}, []

    freqs_win = np.fft.rfftfreq(T_LEN, 1.0 / SAMPLE_RATE)
    for i, det in enumerate(DETECTORS):
        if det not in strain or strain[det] is None:
            out[i] = _white_noise_fill(rng)
            warnings.append(f"{det}: no data — filled with unit white noise")
            continue
        x = np.asarray(strain[det], dtype=np.float64)
        if len(x) != T_LEN:
            raise ValueError(f"{det}: expected {T_LEN} samples (4 s @ 4096 Hz), got {len(x)}")
        w = x.astype(np.float32) if already_whitened else prep.preprocess(x, psds[det], detector=det)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        quality[det] = _quality_checks(det, w, warnings)
        out[i] = w
        p = psds[det]
        asds[det] = np.sqrt(np.interp(freqs_win, p["frequencies"], p["psd"])).astype(np.float32)
        present.append(det)

    return PreparedData(strain=out, trigger_gps=None, detectors_present=present,
                        asds=asds, psds={d: psds[d] for d in present},
                        source="simulated", warnings=warnings,
                        timings={"preprocess_s": round(time.time() - t0, 3)},
                        quality=quality)
