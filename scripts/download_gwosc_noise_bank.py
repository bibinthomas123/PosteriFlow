#!/usr/bin/env python
"""
Download a training-grade real-noise bank from GWOSC O3b.

Per segment saves BOTH:
  - {det}_{gps}_strain.npy : 64 s of whitened strain (float16), whitened by
    the segment's own measured PSD (gwpy .whiten()) — the representation a
    real search sees;
  - {det}_{gps}_asd.npy    : the measured ASD interpolated onto the rfft grid
    of a 4 s / 4096 Hz window (8193 bins, float32) — lets the training loader
    re-color stored design-whitened signals into this segment's whitening:
        sig_seg = irfft( rfft(sig_design) * ASD_design / ASD_measured )

Usage: python scripts/download_gwosc_noise_bank.py --per_det 120 --outdir gw_noise_bank
"""
import argparse, os, sys
import numpy as np
from gwpy.timeseries import TimeSeries

SR = 4096
SEG = 64          # seconds per segment
WIN_N = 16384     # 4 s window the training pipeline uses
O3B_START, O3B_END = 1256655618, 1269363618
O1_START, O1_END = 1126623617, 1136649617
O2_START, O2_END = 1164556817, 1187733618
O4A_START, O4A_END = 1368979218, 1389456018
O4B_START, O4B_END = 1396796418, 1422118818

# Observing-run GPS ranges for the multi-era bank. Real-event inference spans
# O1-O3, so the noise/texture family must too (measured: O3b-only training
# gives +1-1.5 sigma distance bias on O1/O2 events). Virgo only observed from
# late O2 onward.
RUN_RANGES = {
    "O1":  (O1_START, O1_END),
    "O2":  (O2_START, O2_END),
    "O3b": (O3B_START, O3B_END),
    "O4a": (O4A_START, O4A_END),
    "O4b":  (O4B_START, O4B_END),
}
DET_RUNS = {
    "H1": ("O1", "O2", "O3b", "O4a", "O4b"),
    "L1": ("O1", "O2", "O3b", "O4a", "O4b"),
    "V1": ("O2", "O3b", "O4a", "O4b"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_det", type=int, default=120)
    ap.add_argument("--outdir", default="gw_noise_bank")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    freqs_win = np.fft.rfftfreq(WIN_N, 1.0 / SR)
    # randomized GPS grid PER DETECTOR, spanning that detector's observing runs
    # (shuffled so eras interleave and the bank fills uniformly across runs)
    det_candidates = {}
    for det, runs in DET_RUNS.items():
        cands = []
        for run in runs:
            lo, hi = RUN_RANGES[run]
            grid = np.arange(lo + 10000, hi - 10000, 3600)
            CANDIDATES_PER_RUN = 1500

            cands.append(
                rng.choice(
                    grid,
                    size=min(CANDIDATES_PER_RUN, len(grid)),
                    replace=False,
                )
            )
        c = np.concatenate(cands)
        rng.shuffle(c)
        det_candidates[det] = c

    def fetch_one(det, gps):
        fs = f"{args.outdir}/{det}_{gps}_strain.npy"
        if os.path.exists(fs):
            return None  # already have it
        try:
            ts = TimeSeries.fetch_open_data(det, gps, gps + SEG, sample_rate=SR, cache=False)
            ts = ts.highpass(15)
            asd = ts.asd(fftlength=8, overlap=4)  # measured ASD of this segment
            asd_freq = asd.frequencies.value
            asd_val = asd.value

            if (
                len(asd_freq) == 0
                or len(asd_val) == 0
                or not np.isfinite(asd_val).all()
                or np.any(asd_val <= 0)
            ):
                raise ValueError("invalid ASD")
            # Whiten MANUALLY with that exact ASD (freq division + unit-std
            # normalization) instead of gwpy .whiten(): guarantees the noise's
            # whitening filter and the training loader's signal re-coloring
            # denominator are the SAME array — compatible by construction, not
            # merely by matching estimator settings.
            x = ts.value.astype(np.float64)
            n = len(x)
            freqs_seg = np.fft.rfftfreq(n, 1.0 / SR)
            asd_seg = np.interp(
                freqs_seg,
                asd_freq,
                asd_val,
                left=asd_val[0],
                right=asd_val[-1],
            )

            if (
                not np.isfinite(asd_seg).all()
                or np.any(asd_seg <= 0)
            ):
                raise ValueError("bad interpolated ASD")
            wf = np.fft.rfft(x) / np.maximum(asd_seg, 1e-30)
            # Below the 15 Hz highpass corner the numerator/ASD ratio is
            # leakage-dominated junk that was measured to carry ~100% of the
            # output power (and to wreck the unit-std normalization). Zero it —
            # the analysis band starts at 20 Hz.
            wf[freqs_seg < 18.0] = 0.0
            w = np.fft.irfft(wf, n=n)
            w = w[SR * 2: -SR * 2]                      # trim filter/edge artifacts
            w = (w / (w.std() + 1e-30)).astype(np.float32)  # unit noise floor
            arr = w
            if len(arr) < SEG * SR // 2 or not np.isfinite(arr).all():
                raise ValueError("bad strain")
            # quality gate: keep mild real-noise non-stationarity, reject
            # data artifacts (dropouts, saturations, calibration junk)
            bs = arr[: (len(arr) // SR) * SR].reshape(-1, SR).std(axis=1)
            kurt = float(np.mean(arr ** 4) / (np.mean(arr ** 2) ** 2 + 1e-30) - 3.0)
            if kurt > 30.0 or bs.max() > 4.0 or not (0.7 < np.median(bs) < 1.3):
                raise ValueError(
                    f"failed quality gate "
                    f"(kurt={kurt:.1f}, "
                    f"maxstd={bs.max():.2f}, "
                    f"medianstd={np.median(bs):.2f})"
                )
            asd_i = np.interp(freqs_win, asd.frequencies.value, asd.value,
                              left=asd.value[0], right=asd.value[-1]).astype(np.float32)
            if not np.isfinite(asd_i).all() or (asd_i <= 0).any():
                raise ValueError("bad asd")
            np.save(fs, arr.astype(np.float16))
            np.save(fs.replace("_strain", "_asd"), asd_i)
            return True
        except Exception as e:
            print(f"[{det}] {gps} skip: {str(e)[:60]}", flush=True)
            return False

    # Round-robin across detectors so all three fill up together — a usable
    # (if small) bank exists almost immediately and grows uniformly.
    got = {d: sum(1 for f in os.listdir(args.outdir)
                  if f.startswith(f"{d}_") and f.endswith("_strain.npy"))
           for d in ("H1", "L1", "V1")}
    print("existing:", got)
    pos = {d: 0 for d in ("H1", "L1", "V1")}
    while not all(got[d] >= args.per_det or pos[d] >= len(det_candidates[d])
                  for d in ("H1", "L1", "V1")):
        for det in ("H1", "L1", "V1"):
            if got[det] >= args.per_det or pos[det] >= len(det_candidates[det]):
                continue
            gps = int(det_candidates[det][pos[det]])
            pos[det] += 1
            if fetch_one(det, gps):
                got[det] += 1
                print(f"[{det}] {gps} ok ({got[det]}/{args.per_det})", flush=True)
    print("done:", got)


if __name__ == "__main__":
    main()
