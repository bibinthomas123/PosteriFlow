"""
Remix data pipeline for component-storage (v2) datasets.

The v2 generator stores whitened noise and each whitened signal SEPARATELY
per event (float16). This module:

  1. build_memmap_cache(): one-time conversion of the pickle chunks into
     np.memmap arrays for cheap random access (no 22 GB RAM residency).
  2. RemixDataset: assembles a fresh training example every access:
       - noise drawn at random from the ENTIRE noise pool (any event's noise,
         including pure-noise samples) -> the model can never memorize a
         noise realization;
       - per-signal circular time shift (same shift across detectors, so
         inter-detector delays are preserved), geocent_time relabeled;
       - per-signal amplitude rescale s with EXACT distance relabel d/s
         (leading-order GR: strain amplitude scales as 1/d_L) -> direct
         supervision for the amplitude->distance mapping.

Signals within an event are kept sorted by the loudness proxy
Mc^(5/6)/d_L so that rank-conditioning stays consistent (loudness order is
re-evaluated AFTER distance rescaling).
"""

import glob
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PARAM_NAMES = [
    "mass_1", "mass_2", "luminosity_distance",
    "ra", "dec", "theta_jn", "psi", "phase",
    "geocent_time", "a1", "a2",
]
MAX_SIGNALS = 5
T_LEN = 16384
IDX_DIST = PARAM_NAMES.index("luminosity_distance")
IDX_TIME = PARAM_NAMES.index("geocent_time")

# Distance window the rescale augmentation is allowed to move an event into.
# DERIVED from ParamScaler.RANGES (with a small margin) rather than hardcoded:
# these used to be the literals 45.0/2100.0 against a scaler range of
# 40.0/2200.0, so the two could drift apart silently and a rescale could relabel
# an event to a distance that normalize() then CLAMPS -- training the flow on a
# target that is not the event's true distance.
def _dist_guard_window(margin: float = 0.05, dist_range: tuple | None = None):
    from ahsd.models.lean_npe import ParamScaler
    lo, hi = dist_range or ParamScaler.LEGACY_DIST_RANGE
    return lo * (1.0 + margin), hi * (1.0 - margin)

# GPS spans of the observing runs. Real-noise segments are tagged by era so the
# sampling can be era-WEIGHTED: the bank covers O1..O4, but is dominated by the
# far more sensitive O3b/O4, which pulls the training asd_bands distribution
# toward "sensitive" while real O1/O2 events sit in the insensitive tail
# (measured: H1 asd_bands train mean -0.47 vs GW150914 -0.88). Weighting the
# eras lets the training sensitivity distribution match the deployment era.
OBSERVING_RUNS = (
    ("O1",  1126051217, 1137254417),
    ("O2",  1164556817, 1187733618),
    ("O3a", 1238166018, 1253977218),
    ("O3b", 1256655618, 1269363618),
    ("O4",  1368975618, 1500000000),
)


def _era_of(gps: int) -> str:
    """Observing-run label for a GPS time ('other' if outside every run span)."""
    for name, lo, hi in OBSERVING_RUNS:
        if lo <= gps <= hi:
            return name
    return "other"


def _loudness(m1, m2, d):
    """Deterministic loudness proxy (amplitude ~ Mc^(5/6)/d_L) for ordering
    overlapping signals so rank labels stay consistent across the pipeline."""
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    return mc ** (5.0 / 6.0) / max(d, 1.0)


def build_memmap_cache(data_dir: str, split: str, out_dir: str) -> dict:
    """Convert {data_dir}/{split}/batch_*.pkl into memmaps under out_dir.

    Writes: noise.npy [Nn,3,T] f16 (noise pool: ALL events incl. pure noise),
            signals.npy [M,3,T] f16, params.npy [M,11] f32,
            events.json (per signal-event: [sig_start, n_sig]).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{data_dir}/{split}/batch_*.pkl"))
    assert files, f"no chunks under {data_dir}/{split}"

    # pass 1: count
    n_noise, n_sig = 0, 0
    for fp in files:
        with open(fp, "rb") as fh:
            b = pickle.load(fh)
        for s in b["samples"]:
            dd = s["detector_data"]["H1"]
            if "noise" not in dd:
                continue
            n_noise += 1
            if str(s.get("event_type")) != "noise":
                n_sig += min(len(s.get("parameters", [])), MAX_SIGNALS)
    print(f"[{split}] noise pool={n_noise}  total signals={n_sig}")

    noise_mm = np.lib.format.open_memmap(out / "noise.npy", mode="w+",
                                         dtype=np.float16, shape=(n_noise, 3, T_LEN))
    sig_mm = np.lib.format.open_memmap(out / "signals.npy", mode="w+",
                                       dtype=np.float16, shape=(n_sig, 3, T_LEN))
    par_mm = np.lib.format.open_memmap(out / "params.npy", mode="w+",
                                       dtype=np.float32, shape=(n_sig, len(PARAM_NAMES)))
    events = []  # (sig_start, n_sig) per signal-event
    ni, si = 0, 0
    for fp in files:
        with open(fp, "rb") as fh:
            b = pickle.load(fh)
        for s in b["samples"]:
            dd = s["detector_data"]
            if "noise" not in dd["H1"]:
                continue
            noise_mm[ni] = np.stack([dd[d]["noise"] for d in ("H1", "L1", "V1")])
            ni += 1
            if str(s.get("event_type")) == "noise":
                continue
            plist = s.get("parameters", [])[:MAX_SIGNALS]
            sigs = [np.stack([dd[d]["signals"][k] for d in ("H1", "L1", "V1")])
                    for k in range(len(plist))]
            order = sorted(range(len(plist)), reverse=True,
                           key=lambda k: _loudness(plist[k]["mass_1"], plist[k]["mass_2"],
                                                   plist[k]["luminosity_distance"]))
            start = si
            for k in order:
                sig_mm[si] = sigs[k]
                par_mm[si] = [plist[k].get(p, 0.0) for p in PARAM_NAMES]
                si += 1
            events.append([start, len(plist)])
    noise_mm.flush(); sig_mm.flush(); par_mm.flush()
    meta = {"n_noise": int(ni), "n_signals": int(si), "events": events}
    with open(out / "events.json", "w") as f:
        json.dump(meta, f)
    print(f"[{split}] cache written -> {out}  (noise {ni}, signals {si}, events {len(events)})")
    return meta


class RemixDataset(Dataset):
    """On-the-fly remixed training examples from a memmap cache.

    Optional real-noise mixing (real_noise_dir + real_noise_prob): with the
    given probability an event's noise is drawn from 4 s crops of real
    (GWOSC, per-segment-whitened) detector noise instead of the Gaussian
    pool, and the event's design-whitened signals are RE-COLORED into that
    segment's whitening via the exact linear transform
        sig_seg = irfft( rfft(sig_design) * ASD_design / ASD_measured ).
    All other augmentations (noise swap, time shift, distance rescale,
    loudness re-sort) are applied identically for both noise sources.
    Defaults (real_noise_prob=0.0) reproduce the previous behavior exactly.
    """

    def __init__(self, cache_dir: str, time_shift_max: float = 0.1,
                 dist_scale_range: tuple = (0.75, 1.33), sample_rate: int = 4096,
                 remix: bool = True, seed: int = 0,
                 real_noise_dir: str | None = None, real_noise_prob: float = 0.0,
                 recolor_clamp: float = 50.0, det_dropout: float = 0.0,
                 return_asd_bands: bool = False, psd_bands: int = 16,
                 real_noise_era_weights: dict | None = None,
                 det_keep_weights: list | None = None,
                 basin_oversample: int = 0,
                 delay_aug: float = 0.0, delay_aug_candidates: int = 64,
                 return_event_idx: bool = False,
                 dist_range: tuple | None = None):
        cache = Path(cache_dir)
        self.cache_dir = cache
        self.noise = np.load(cache / "noise.npy", mmap_mode="r")
        self.signals = np.load(cache / "signals.npy", mmap_mode="r")
        self.params = np.load(cache / "params.npy", mmap_mode="r")
        with open(cache / "events.json") as f:
            meta = json.load(f)
        self.events = meta["events"]
        self.n_noise = meta["n_noise"]

        # ---- basin-cell oversampling (analysis/attractor_basin) ----
        # The {tau_HL in [+4,+10] ms} x {m1>=29} cell collapses to a constant
        # (30,30) answer on converged models because prior-weighted NLL never
        # pays enough to resolve it (feature fixes measured insufficient:
        # phase-slope helped loud only, phase-residual inert, dropout inert).
        # Each cell event gets `basin_oversample` EXTRA epoch slots; every
        # copy remixes independently (fresh noise/shift/rescale) since the
        # augmentation RNG is seeded by the epoch SLOT, not the event.
        self._index_map = np.arange(len(self.events))
        if basin_oversample and remix:
            tau = self._tau_hl_cached(cache)
            rank0 = np.array([e[0] for e in self.events])
            m1 = np.asarray(self.params[rank0])[:, 0]
            cell = (tau >= 4.0) & (tau <= 10.0) & (m1 >= 29.0)
            extra = np.repeat(self._index_map[cell], int(basin_oversample))
            self._index_map = np.concatenate([self._index_map, extra])
            print(f"[RemixDataset] basin oversample: {int(cell.sum())} cell events "
                  f"(tau_HL [+4,+10] ms & m1>=29) x{int(basin_oversample)} extra "
                  f"-> {len(self._index_map)} samples/epoch")
        # ---- differential-delay augmentation (analysis/attractor_basin) ----
        # With probability `delay_aug` a signal is remapped to a new sky:
        # per-detector arrival delays re-rolled EXACTLY (FFT phase ramp) and
        # ra/dec/psi relabeled honestly (antenna-matched candidate + closed-
        # form psi rotation; see delay_aug.py). This is the only augmentation
        # that varies inter-detector delay — it breaks the measured-tau <-> q
        # correlation behind the (30,30) attractor instead of reweighting it.
        self.return_event_idx = bool(return_event_idx)
        self.delay_aug = float(delay_aug)
        self.sky_remap = None
        if self.delay_aug > 0.0 and remix:
            from delay_aug import SkyRemapper
            self.sky_remap = SkyRemapper(str(cache), n_candidates=delay_aug_candidates,
                                         dist_window=_dist_guard_window(dist_range=dist_range))
            print(f"[RemixDataset] delay aug: p={self.delay_aug}/signal, "
                  f"{delay_aug_candidates} sky candidates")
        self.shift_max_samp = int(time_shift_max * sample_rate)
        self.dist_lo, self.dist_hi = dist_scale_range
        # must match the model's ParamScaler distance support (LeanNPE dist_range)
        self.dist_guard_lo, self.dist_guard_hi = _dist_guard_window(dist_range=dist_range)
        self.remix = remix
        self.seed = seed
        self.epoch = 0  # bump via set_epoch() for fresh noise pairings
        # Detector-dropout augmentation: with this probability an event keeps
        # only a random subset of detectors; dropped detectors are replaced
        # with unit white noise — EXACTLY the fill the inference pipeline uses
        # for missing detectors (e.g. two-detector O1/O2 events), so train and
        # deployment see the same representation.
        self.det_dropout = float(det_dropout)
        # non-empty proper subsets of (H1, L1, V1) to KEEP
        self._keep_configs = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
        # Optional non-uniform weighting over those configs. Uniform means the
        # (H1, L1) config — the one every real 2-detector O1/O2 event has — is
        # drawn only 1/6 of the time dropout fires (~3.3% of events at
        # det_dropout=0.2), leaving that deployment configuration rare in
        # training. Weighting it up matches train to deployment.
        self._keep_p = None
        if det_keep_weights is not None:
            w = np.asarray(det_keep_weights, dtype=np.float64)
            if w.shape != (len(self._keep_configs),) or (w < 0).any() or w.sum() <= 0:
                raise ValueError(
                    f"det_keep_weights must be {len(self._keep_configs)} "
                    f"non-negative weights (order {self._keep_configs}), got {det_keep_weights}")
            self._keep_p = w / w.sum()

        # ---- optional PSD-conditioning feature (sensitivity summary) ----
        # asd_bands[d, b] = band-mean of log(design_ASD / whitening_ASD) for
        # detector d: 0 for design-whitened events, negative where the detector
        # is less sensitive than design (the recolor filter = design/measured).
        # Lets a psd_cond model convert whitened amplitude -> physical SNR.
        self.return_asd_bands = bool(return_asd_bands)
        self.psd_bands = int(psd_bands)
        if self.return_asd_bands:
            freqs = np.fft.rfftfreq(T_LEN, 1.0 / sample_rate)      # [T_LEN//2+1]
            edges = np.geomspace(20.0, sample_rate / 2.0, self.psd_bands + 1)
            self._band_slices = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                idx = np.where((freqs >= lo) & (freqs < hi))[0]
                self._band_slices.append(idx if len(idx) else
                                         np.array([np.argmin(np.abs(freqs - lo))]))

        # ---- optional real-noise bank ----
        self.real_noise_prob = float(real_noise_prob)
        self.real_bank = None
        if real_noise_dir and self.real_noise_prob > 0.0:
            bank_dir = Path(real_noise_dir)
            design = {d: np.load(bank_dir / f"design_asd_{d}.npy") for d in ("H1", "L1", "V1")}
            self.real_bank = {d: [] for d in ("H1", "L1", "V1")}
            self.recolor = {d: [] for d in ("H1", "L1", "V1")}  # parallel filters
            self._seg_era = {d: [] for d in ("H1", "L1", "V1")}  # parallel era labels
            for d in ("H1", "L1", "V1"):
                for f in sorted(bank_dir.glob(f"{d}_*_strain.npy")):
                    asd_f = Path(str(f).replace("_strain", "_asd"))
                    if not asd_f.exists():
                        continue
                    strain = np.load(f)  # f16, per-segment-whitened
                    asd = np.load(asd_f).astype(np.float32)
                    filt = np.clip(design[d] / np.maximum(asd, 1e-30),
                                   1.0 / recolor_clamp, recolor_clamp).astype(np.float32)
                    self.real_bank[d].append(strain)
                    self.recolor[d].append(filt)
                    m = re.search(r"_(\d+)_strain", f.name)
                    self._seg_era[d].append(_era_of(int(m.group(1))) if m else "other")
            counts = {d: len(v) for d, v in self.real_bank.items()}
            if min(counts.values()) == 0:
                raise ValueError(f"real-noise bank incomplete under {bank_dir}: {counts}")

            # per-detector segment sampling distribution (None = uniform)
            self._seg_p = {d: None for d in ("H1", "L1", "V1")}
            if real_noise_era_weights:
                for d in ("H1", "L1", "V1"):
                    w = np.array([float(real_noise_era_weights.get(e, 1.0))
                                  for e in self._seg_era[d]], dtype=np.float64)
                    if w.sum() <= 0:
                        raise ValueError(
                            f"real_noise_era_weights zero out every {d} segment "
                            f"(eras present: {sorted(set(self._seg_era[d]))})")
                    self._seg_p[d] = w / w.sum()
            print(f"[RemixDataset] real-noise bank: {counts} segments, "
                  f"p(real)={self.real_noise_prob}")
            for d in ("H1", "L1", "V1"):
                era_n = {e: self._seg_era[d].count(e) for e in sorted(set(self._seg_era[d]))}
                line = f"  {d}: " + " ".join(f"{e}={n}" for e, n in era_n.items())
                if self._seg_p[d] is not None:
                    # effective (post-weighting) probability mass per era
                    eff = {e: float(self._seg_p[d][[i for i, x in enumerate(self._seg_era[d])
                                                    if x == e]].sum())
                           for e in era_n}
                    line += "  ->  p(era)=" + " ".join(f"{e}:{v:.2f}" for e, v in eff.items())
                print(line)
            if self._keep_p is not None:
                print(f"  det keep-config p: " + " ".join(
                    f"{cfg}:{p:.2f}" for cfg, p in zip(self._keep_configs, self._keep_p)))

    def _tau_hl_cached(self, cache: Path) -> np.ndarray:
        """Per-event H1-L1 lag (ms) of the rank-0 stored signal component
        (noise-free xcorr argmax, +-15 ms window — same estimator as the
        analysis/attractor_basin density audit). Computed once, cached as
        tau_hl.npy beside the memmaps."""
        f = cache / "tau_hl.npy"
        if f.exists():
            tau = np.load(f)
            if tau.shape[0] == len(self.events):
                return tau
        maxlag = int(0.015 * 4096)
        rank0 = np.array([e[0] for e in self.events])
        tau = np.empty(len(rank0), dtype=np.float32)
        for i in range(0, len(rank0), 512):
            s = np.asarray(self.signals[rank0[i:i + 512]], dtype=np.float32)
            X = np.fft.rfft(s[:, 0]) * np.conj(np.fft.rfft(s[:, 1]))
            cc = np.fft.irfft(X)
            win = np.concatenate([cc[:, -maxlag:], cc[:, :maxlag + 1]], axis=1)
            tau[i:i + 512] = (np.argmax(np.abs(win), axis=1) - maxlag) / 4096 * 1000.0
        np.save(f, tau)
        print(f"[RemixDataset] tau_HL sidecar computed -> {f}")
        return tau

    def set_epoch(self, epoch: int):
        """Set the current epoch so per-worker RNGs draw fresh noise/signal
        pairings each epoch (must be called before iterating; workers respawn
        per epoch and read this)."""
        self.epoch = epoch

    def __len__(self):
        return len(self._index_map)

    def _draw_seg(self, d, rng):
        """Index of a real-noise segment for detector `d`: era-weighted when
        real_noise_era_weights was given, uniform otherwise."""
        n = len(self.real_bank[d])
        p = self._seg_p[d]
        return int(rng.choice(n, p=p)) if p is not None else int(rng.integers(n))

    def _real_noise_and_filters(self, rng, seg_override=None):
        """Random 4 s crop per detector from the real bank + that segment's
        re-coloring filter. Time-flip+sign decorrelates reused segments.

        seg_override: optional {det: seg_idx} to FIX which segment is used
        (skips _draw_seg); only the crop position/sign still draw from `rng`.
        Used by get_multiview(noise_mode="real") to hold the noise DOMAIN
        (and hence asd_bands, which is a function of the segment's recolor
        filter) identical across views, so only the fine noise instantiation
        varies -- the real-noise analogue of force_design=True."""
        crops, filts = [], []
        for d in ("H1", "L1", "V1"):
            k = seg_override[d] if seg_override is not None else self._draw_seg(d, rng)
            seg = self.real_bank[d][k]
            i0 = int(rng.integers(0, seg.shape[0] - T_LEN))
            c = seg[i0:i0 + T_LEN].astype(np.float32)
            if rng.uniform() < 0.5:
                c = -c[::-1].copy()
            crops.append(c)
            filts.append(self.recolor[d][k])
        return np.stack(crops), filts

    def _remix_event(self, start, nsig, rng_sig, rng_noise,
                     noise_idx_override=None, force_design=False,
                     force_real_segs=None):
        """Core remix logic, split into two independent RNG streams so a
        signal's IDENTITY (delay-aug resky, distance rescale, time shift,
        detector-dropout SCENARIO -- which detectors are even present) can be
        held fixed across multiple draws while only the NOISE INSTANTIATION
        varies (rng_sig / rng_noise respectively). __getitem__ uses one-shot
        streams per epoch slot; get_multiview() replays the same rng_sig seed
        across K views of one event so they share identity and differ only in
        noise (analysis/attractor_basin addendum: noise-consistency training,
        targeting the MF-token branch's noise-realization sensitivity found
        by grafting on GW150914).

        force_real_segs: optional {det: seg_idx} forcing REAL noise from
        those exact segments (skips the real_noise_prob coin flip and segment
        draw) -- get_multiview(noise_mode="real")'s mechanism for holding the
        noise DOMAIN (hence asd_bands) fixed across views while the crop
        position/sign (rng_noise) still varies per view."""
        if force_real_segs is not None:
            use_real = True
        elif force_design:
            use_real = False
        else:
            use_real = self.real_bank is not None and rng_noise.uniform() < self.real_noise_prob
        if use_real:
            strain, recolor_filts = self._real_noise_and_filters(
                rng_noise, seg_override=force_real_segs)
        else:
            noise_idx = (noise_idx_override if noise_idx_override is not None
                        else int(rng_noise.integers(self.n_noise)))
            strain = self.noise[noise_idx].astype(np.float32)
            recolor_filts = None

        pv = np.zeros((MAX_SIGNALS, len(PARAM_NAMES)), dtype=np.float32)
        entries = []
        sig_sum = np.zeros_like(strain)
        for k in range(nsig):
            sig = self.signals[start + k].astype(np.float32)
            par = self.params[start + k].copy()
            if self.remix:
                if self.sky_remap is not None and rng_sig.uniform() < self.delay_aug:
                    sig = self.sky_remap.remap(sig, par, rng_sig)
                # distance rescale (exact relabel), clamped to scaler range
                s = float(rng_sig.uniform(self.dist_lo, self.dist_hi))
                d_new = par[IDX_DIST] / s
                if not (self.dist_guard_lo < d_new < self.dist_guard_hi):
                    s, d_new = 1.0, par[IDX_DIST]
                sig = sig * s
                par[IDX_DIST] = d_new
                # circular time shift, same across detectors
                if abs(par[IDX_TIME]) < 1.45 and self.shift_max_samp > 0:
                    ds = int(rng_sig.integers(-self.shift_max_samp, self.shift_max_samp + 1))
                    if ds != 0:
                        sig = np.roll(sig, ds, axis=-1)
                        par[IDX_TIME] += ds / 4096.0
            sig_sum += sig
            entries.append(par)

        if recolor_filts is not None:
            # re-color the per-detector signal sum from design whitening into
            # the chosen segment's whitening (exact: whitening is diagonal in
            # frequency, so it commutes with the scale/shift applied above)
            for di in range(3):
                sig_sum[di] = np.fft.irfft(np.fft.rfft(sig_sum[di]) * recolor_filts[di],n=T_LEN).astype(np.float32)
        strain += sig_sum

        kept_dets = [0, 1, 2]
        if self.remix and self.det_dropout > 0.0 and rng_sig.uniform() < self.det_dropout:
            ci = (int(rng_sig.choice(len(self._keep_configs), p=self._keep_p))
                  if self._keep_p is not None
                  else int(rng_sig.integers(len(self._keep_configs))))
            keep = self._keep_configs[ci]
            kept_dets = list(keep)
            for di in range(3):
                if di not in keep:
                    if use_real:
                        det_name = ("H1", "L1", "V1")[di]
                        # replace with another real-noise crop from this
                        # detector -- same fixed segment if force_real_segs
                        # is active (keeps asd_bands identical across views)
                        k = (force_real_segs[det_name] if force_real_segs is not None
                            else self._draw_seg(det_name, rng_noise))
                        seg = self.real_bank[det_name][k]
                        i0 = int(rng_noise.integers(0, seg.shape[0] - T_LEN))
                        c = seg[i0:i0 + T_LEN].astype(np.float32)
                        if rng_noise.uniform() < 0.5:
                            c = -c[::-1].copy()
                        strain[di] = c
                    else:
                        strain[di] = rng_noise.standard_normal(T_LEN).astype(np.float32)

        # Network SNR of the (rescaled/shifted/re-colored/summed) whitened
        # injection: for unit-variance whitened data the optimal matched-filter
        # SNR is exactly the L2 norm of the whitened signal, so this stays
        # correct under every augmentation above WITHOUT any stored SNR (which
        # would be stale after distance rescaling and real-noise re-coloring).
        # Sum only over KEPT detectors — a dropped detector's signal is gone.
        net_snr = float(np.sqrt((sig_sum[kept_dets] ** 2).sum()))

        # re-sort by loudness AFTER rescaling so rank labels stay consistent
        entries.sort(key=lambda p: _loudness(p[0], p[1], p[IDX_DIST]), reverse=True)
        for k, par in enumerate(entries):
            pv[k] = par

        out = (torch.from_numpy(strain),
               torch.from_numpy(pv),
               torch.tensor(nsig, dtype=torch.long),
               torch.tensor(net_snr, dtype=torch.float32))
        if self.return_asd_bands:
            out = out + (self._asd_bands(recolor_filts, kept_dets),)
        return out

    def __getitem__(self, i):
        # i is the epoch SLOT; the map may point several slots at one basin-
        # cell event. Seeding by slot keeps every copy an independent remix.
        event_idx = int(self._index_map[i])
        start, nsig = self.events[event_idx]
        # integer stream tags (0=sig, 1=noise): np.random seed tuples accept
        # only non-negative ints, not strings.
        rng_sig = np.random.default_rng((self.seed, self.epoch, i, 0))
        rng_noise = np.random.default_rng((self.seed, self.epoch, i, 1))
        noise_idx_override = None if self.remix else i % self.n_noise
        out = self._remix_event(start, nsig, rng_sig, rng_noise, noise_idx_override)
        if self.return_event_idx:
            # Appended LAST (after asd_bands, if present) so any existing
            # caller unpacking a fixed-length tuple is unaffected unless it
            # explicitly opts in via return_event_idx=True. Used by the
            # semi-amortized training hard-example selector
            # (experiments/semi_amortized.py) to look up basin-cell
            # membership per batch example via basin_cell_mask().
            out = out + (torch.tensor(event_idx, dtype=torch.long),)
        return out

    def basin_cell_event_indices(self) -> np.ndarray:
        """Underlying event indices (into self.events, NOT the possibly-
        oversampled _index_map) in {tau_HL in [+4,+10] ms, m1>=29} -- the
        diagnosed failure population. Cached; safe regardless of
        basin_oversample. Feeds the multiview sampler for noise-consistency
        training (analysis/attractor_basin addendum)."""
        if not hasattr(self, "_basin_cell_idx_cache"):
            tau = self._tau_hl_cached(self.cache_dir)
            rank0 = np.array([e[0] for e in self.events])
            m1 = np.asarray(self.params[rank0])[:, 0]
            cell = (tau >= 4.0) & (tau <= 10.0) & (m1 >= 29.0)
            self._basin_cell_idx_cache = np.nonzero(cell)[0]
        return self._basin_cell_idx_cache

    def basin_cell_mask(self) -> np.ndarray:
        """Boolean [len(self.events)] membership mask, O(1) per-event lookup
        version of basin_cell_event_indices() -- used by the semi-amortized
        training hard-example selector (experiments/semi_amortized.py) to
        tag each batch example without a linear scan."""
        if not hasattr(self, "_basin_cell_bool_cache"):
            mask = np.zeros(len(self.events), dtype=bool)
            mask[self.basin_cell_event_indices()] = True
            self._basin_cell_bool_cache = mask
        return self._basin_cell_bool_cache

    def get_multiview(self, event_idx: int, k: int, epoch: int, group_id: int,
                      noise_mode: str = "design") -> list:
        """K independently-NOISED views of the SAME underlying event: draws
        that define signal IDENTITY (delay-aug resky, distance rescale, time
        shift, det-dropout scenario) are held fixed across views by replaying
        the same rng_sig seed; only the noise instantiation varies per view
        via a fresh rng_noise.

        noise_mode="design": views are Gaussian design noise (asd_bands 0 for
        every view) -- no sensitivity-context confound, but ALSO no exposure
        to real noise's fine structure. Measured (2026-07-20,
        model/lean_npe_noiseinv): improved the SYNTHETIC basin probe further
        (worst-row rail 0.15->0.07) but did NOT transfer to real GW150914 --
        rail went 74.0%->81.6%, slightly worse. Design-noise invariance
        doesn't touch the actual failure mode.

        noise_mode="real": views are real noise from the SAME fixed segment
        per detector (chosen once per group, independent of view index) --
        only the crop POSITION and sign flip vary per view via rng_noise.
        Holding the segment fixed keeps each view's recolor filter (hence
        asd_bands) identical, so there's still no sensitivity-context
        confound, but now the varying quantity is real noise's actual fine
        structure -- directly targeting the mechanism the grafting test
        isolated (analysis/attractor_basin addendum, 2026-07-20: the MF
        branch's collapse on GW150914 was caused by that exact segment's
        local noise realization, not the astrophysical signal or the
        broadband domain). Requires a real-noise bank (real_noise_dir).

        Training-only (remix required)."""
        assert self.remix, "get_multiview requires remix=True (training only)"
        assert noise_mode in ("design", "real")
        if noise_mode == "real":
            assert self.real_bank is not None, \
                "get_multiview(noise_mode='real') requires real_noise_dir"
        start, nsig = self.events[int(event_idx)]
        # integer stream tags: 2=multiview-sig, 3=multiview-noise, 4=multiview
        # -domain (distinct from __getitem__'s 0/1 so slot-indexed and
        # multiview draws never collide even if event_idx == some epoch slot).
        force_real_segs = None
        if noise_mode == "real":
            domain_rng = np.random.default_rng((self.seed, epoch, 4, group_id, event_idx))
            force_real_segs = {d: self._draw_seg(d, domain_rng) for d in ("H1", "L1", "V1")}
        views = []
        for v in range(k):
            rng_sig = np.random.default_rng(
                (self.seed, epoch, 2, group_id, event_idx))
            rng_noise = np.random.default_rng(
                (self.seed, epoch, 3, group_id, event_idx, v))
            views.append(self._remix_event(
                start, nsig, rng_sig, rng_noise,
                force_design=(noise_mode == "design"),
                force_real_segs=force_real_segs))
        return views

    def _asd_bands(self, recolor_filts, kept_dets):
        """[3, psd_bands] sensitivity summary. 0 for design-whitened / dropped
        detectors; band-mean log(design/measured)=log(recolor) for real ones."""
        ab = np.zeros((3, self.psd_bands), dtype=np.float32)
        if recolor_filts is not None:
            for di in range(3):
                if di not in kept_dets:
                    continue
                logf = np.log(np.maximum(recolor_filts[di], 1e-30))
                ab[di] = [float(logf[s].mean()) for s in self._band_slices]
        return torch.from_numpy(ab)
