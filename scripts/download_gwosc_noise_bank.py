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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_det", type=int, default=120)
    ap.add_argument("--outdir", default="gw_noise_bank")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    freqs_win = np.fft.rfftfreq(WIN_N, 1.0 / SR)
    # randomized GPS grid, same candidates for every detector
    candidates = np.sort(rng.choice(
        np.arange(O3B_START + 10000, O3B_END - 10000, 3600), size=600, replace=False))

    def fetch_one(det, gps):
        fs = f"{args.outdir}/{det}_{gps}_strain.npy"
        if os.path.exists(fs):
            return None  # already have it
        try:
            ts = TimeSeries.fetch_open_data(det, gps, gps + SEG, sample_rate=SR, cache=False)
            ts = ts.highpass(15)
            asd = ts.asd(fftlength=8, overlap=4)  # measured ASD of this segment
            # Whiten MANUALLY with that exact ASD (freq division + unit-std
            # normalization) instead of gwpy .whiten(): guarantees the noise's
            # whitening filter and the training loader's signal re-coloring
            # denominator are the SAME array — compatible by construction, not
            # merely by matching estimator settings.
            x = ts.value.astype(np.float64)
            n = len(x)
            freqs_seg = np.fft.rfftfreq(n, 1.0 / SR)
            asd_seg = np.interp(freqs_seg, asd.frequencies.value, asd.value,
                                left=asd.value[0], right=asd.value[-1])
            w = np.fft.irfft(np.fft.rfft(x) / np.maximum(asd_seg, 1e-30), n=n)
            w = w[SR * 2: -SR * 2]                      # trim filter/edge artifacts
            w = (w / (w.std() + 1e-30)).astype(np.float32)  # unit noise floor
            arr = w
            if len(arr) < SEG * SR // 2 or not np.isfinite(arr).all():
                raise ValueError("bad strain")
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
    for gps in candidates:
        if all(v >= args.per_det for v in got.values()):
            break
        for det in ("H1", "L1", "V1"):
            if got[det] >= args.per_det:
                continue
            r = fetch_one(det, int(gps))
            if r:
                got[det] += 1
                print(f"[{det}] {int(gps)} ok ({got[det]}/{args.per_det})", flush=True)
    print("done:", got)


if __name__ == "__main__":
    main()
