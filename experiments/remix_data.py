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


def _loudness(m1, m2, d):
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
                 recolor_clamp: float = 50.0, det_dropout: float = 0.0):
        cache = Path(cache_dir)
        self.noise = np.load(cache / "noise.npy", mmap_mode="r")
        self.signals = np.load(cache / "signals.npy", mmap_mode="r")
        self.params = np.load(cache / "params.npy", mmap_mode="r")
        with open(cache / "events.json") as f:
            meta = json.load(f)
        self.events = meta["events"]
        self.n_noise = meta["n_noise"]
        self.shift_max_samp = int(time_shift_max * sample_rate)
        self.dist_lo, self.dist_hi = dist_scale_range
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

        # ---- optional real-noise bank ----
        self.real_noise_prob = float(real_noise_prob)
        self.real_bank = None
        if real_noise_dir and self.real_noise_prob > 0.0:
            bank_dir = Path(real_noise_dir)
            design = {d: np.load(bank_dir / f"design_asd_{d}.npy") for d in ("H1", "L1", "V1")}
            self.real_bank = {d: [] for d in ("H1", "L1", "V1")}
            self.recolor = {d: [] for d in ("H1", "L1", "V1")}  # parallel filters
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
            counts = {d: len(v) for d, v in self.real_bank.items()}
            if min(counts.values()) == 0:
                raise ValueError(f"real-noise bank incomplete under {bank_dir}: {counts}")
            print(f"[RemixDataset] real-noise bank: {counts} segments, "
                  f"p(real)={self.real_noise_prob}")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.events)

    def _real_noise_and_filters(self, rng):
        """Random 4 s crop per detector from the real bank + that segment's
        re-coloring filter. Time-flip+sign decorrelates reused segments."""
        crops, filts = [], []
        for d in ("H1", "L1", "V1"):
            k = int(rng.integers(len(self.real_bank[d])))
            seg = self.real_bank[d][k]
            i0 = int(rng.integers(0, seg.shape[0] - T_LEN))
            c = seg[i0:i0 + T_LEN].astype(np.float32)
            if rng.uniform() < 0.5:
                c = -c[::-1].copy()
            crops.append(c)
            filts.append(self.recolor[d][k])
        return np.stack(crops), filts

    def __getitem__(self, i):
        start, nsig = self.events[i]
        rng = np.random.default_rng((self.seed, self.epoch, i))

        use_real = self.real_bank is not None and rng.uniform() < self.real_noise_prob
        if use_real:
            strain, recolor_filts = self._real_noise_and_filters(rng)
        else:
            noise_idx = int(rng.integers(self.n_noise)) if self.remix else i % self.n_noise
            strain = self.noise[noise_idx].astype(np.float32)
            recolor_filts = None

        pv = np.zeros((MAX_SIGNALS, len(PARAM_NAMES)), dtype=np.float32)
        entries = []
        sig_sum = np.zeros_like(strain)
        for k in range(nsig):
            sig = self.signals[start + k].astype(np.float32)
            par = self.params[start + k].copy()
            if self.remix:
                # distance rescale (exact relabel), clamped to scaler range
                s = float(rng.uniform(self.dist_lo, self.dist_hi))
                d_new = par[IDX_DIST] / s
                if not (45.0 < d_new < 2100.0):
                    s, d_new = 1.0, par[IDX_DIST]
                sig = sig * s
                par[IDX_DIST] = d_new
                # circular time shift, same across detectors
                if abs(par[IDX_TIME]) < 1.45 and self.shift_max_samp > 0:
                    ds = int(rng.integers(-self.shift_max_samp, self.shift_max_samp + 1))
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

        if self.remix and self.det_dropout > 0.0 and rng.uniform() < self.det_dropout:
            keep = self._keep_configs[int(rng.integers(len(self._keep_configs)))]
            for di in range(3):
                if di not in keep:
                    if use_real:                                                        
                        det_name = ("H1", "L1", "V1")[di]
                        # replace with another real-noise crop from this detector
                        k = int(rng.integers(len(self.real_bank[det_name])))
                        seg = self.real_bank[det_name][k]
                        i0 = int(rng.integers(0, seg.shape[0] - T_LEN))
                        c = seg[i0:i0 + T_LEN].astype(np.float32)
                        if rng.uniform() < 0.5:
                            c = -c[::-1].copy()
                        strain[di] = c
                    else:
                        strain[di] = rng.standard_normal(T_LEN).astype(np.float32)

        # re-sort by loudness AFTER rescaling so rank labels stay consistent
        entries.sort(key=lambda p: _loudness(p[0], p[1], p[IDX_DIST]), reverse=True)
        for k, par in enumerate(entries):
            pv[k] = par

        return (torch.from_numpy(strain),
                torch.from_numpy(pv),
                torch.tensor(nsig, dtype=torch.long))
