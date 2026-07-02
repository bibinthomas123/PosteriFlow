#!/usr/bin/env python
"""
Download real O3/O4 noise segments from GWOSC for training.

Usage:
    conda run -n ahsd python scripts/download_gwosc_noise.py \
        --output data/gwosc_cache \
        --n-segments 50 \
        --detectors H1 L1 V1 \
        --run O3

Saves .npz files consumable by BilbyNoiseGenerator (real_noise_prob > 0).
Each file: strain (float64), sample_rate, detector.

Set real_noise_prob in data_config.yaml to use them:
    real_noise_prob: 0.3          # 30% real noise
    real_noise_cache_dir: data/gwosc_cache
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# Known O3a science-mode GPS windows (H1, L1, V1 all online)
# Source: GWTC-3 data release, O3a epoch
_O3A_QUIET_TIMES = [
    1238166018,  # 2019-04-01 15:00 UTC
    1238770818,  # 2019-04-08 15:00 UTC
    1239375618,  # 2019-04-15 15:00 UTC
    1240585218,  # 2019-04-29 15:00 UTC
    1241189418,  # 2019-05-06 15:00 UTC
    1241794218,  # 2019-05-13 15:00 UTC
    1242398418,  # 2019-05-20 15:00 UTC
    1243003218,  # 2019-05-27 15:00 UTC
    1243608018,  # 2019-06-03 15:00 UTC
    1244212218,  # 2019-06-10 15:00 UTC
]

# O3b epoch
_O3B_QUIET_TIMES = [
    1256655618,  # 2019-11-01 15:00 UTC
    1257260418,  # 2019-11-08 15:00 UTC
    1257865218,  # 2019-11-15 15:00 UTC
    1258469418,  # 2019-11-22 15:00 UTC
    1259074818,  # 2019-11-29 15:00 UTC
    1259679618,  # 2019-12-06 15:00 UTC
    1260284418,  # 2019-12-13 15:00 UTC
    1260889218,  # 2019-12-20 15:00 UTC
]

_GPS_TIMES = {"O3a": _O3A_QUIET_TIMES, "O3b": _O3B_QUIET_TIMES,
              "O3": _O3A_QUIET_TIMES + _O3B_QUIET_TIMES}


def download_segment(detector: str, gps_time: int, duration: float,
                     sample_rate: float, output_path: Path) -> bool:
    """Download one noise segment via bilby/gwpy and save as .npz."""
    try:
        import bilby
        ifo = bilby.gw.detector.get_interferometer_with_open_data(
            detector,
            trigger_time=gps_time,
            duration=duration + 8,   # extra for edge effects
            psd_offset=-512,
            psd_duration=512,
            roll_off=0.4,
            minimum_frequency=20.0,
            sampling_frequency=sample_rate,
            cache=False,
        )
        strain = ifo.strain_data.time_domain_strain
        # Trim to requested duration from center
        n_want = int(duration * sample_rate)
        if len(strain) > n_want:
            start = (len(strain) - n_want) // 2
            strain = strain[start: start + n_want]
        elif len(strain) < n_want:
            _log.warning(f"Segment too short: {len(strain)} < {n_want}")
            return False

        np.savez_compressed(
            output_path,
            strain=strain.astype(np.float64),
            sample_rate=np.array(sample_rate),
            detector=np.array(detector),
            gps_time=np.array(gps_time),
        )
        _log.info(f"Saved {output_path.name} ({detector}, GPS={gps_time})")
        return True

    except Exception as e:
        _log.warning(f"Failed [{detector} GPS={gps_time}]: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/gwosc_cache")
    p.add_argument("--n-segments", type=int, default=30,
                   help="Number of segments per detector")
    p.add_argument("--detectors", nargs="+", default=["H1", "L1", "V1"])
    p.add_argument("--run", choices=["O3a", "O3b", "O3"], default="O3")
    p.add_argument("--duration", type=float, default=64.0,
                   help="Segment duration in seconds (longer = more noise variety)")
    p.add_argument("--sample-rate", type=float, default=4096.0)
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    gps_times = _GPS_TIMES[args.run]
    # Stride through available times; offset each iteration by 200s for variety
    saved = {det: 0 for det in args.detectors}
    total = 0

    for i in range(args.n_segments):
        t_base = gps_times[i % len(gps_times)]
        t = t_base + (i // len(gps_times)) * 200

        for det in args.detectors:
            if saved[det] >= args.n_segments:
                continue
            fname = out_dir / f"{det}_{t}_{int(args.duration)}s.npz"
            if fname.exists():
                _log.info(f"Already exists: {fname.name}")
                saved[det] += 1
                total += 1
                continue
            ok = download_segment(det, t, args.duration, args.sample_rate, fname)
            if ok:
                saved[det] += 1
                total += 1

    _log.info(f"Downloaded {total} segments → {out_dir}")
    _log.info("Per detector: " + ", ".join(f"{d}={saved[d]}" for d in args.detectors))
    _log.info(
        "\nTo use in training, set in data_config.yaml:\n"
        "  real_noise_prob: 0.3\n"
        f"  real_noise_cache_dir: {out_dir}"
    )


if __name__ == "__main__":
    main()
