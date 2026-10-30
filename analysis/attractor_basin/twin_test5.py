"""Round 5: inter-detector time delays — training-data audit + delay-corrected twins.

  K  AUDIT: measure H1-L1 lags of loud stored validation events. If they are
     all ~0 ms, the dataset generator shares the Δt=0 bug and every real
     two-detector event is geometrically OOD. If they spread over ±10 ms,
     training geometry is fine and only the study helper was buggy.
  H  delay-corrected twin (pseudo-truth sky -> real Δt) into off-source real
     noise at -300 s — does the collapse now reproduce?
  I  mirror control: delays negated.
  J  delay-corrected twin in Gaussian design noise (dL 440 and 1564).

Every built condition prints its measured H1-L1 lag as a self-check.
"""
import glob
import pickle
import sys
from pathlib import Path

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))

import numpy as np

from twin_test import MODELS, DETS, analyze, design_noise
from twin_test4 import h1l1_lag_ms
import _common as C
from ahsd.data.bilby_pipeline import get_default_psd
from ahsd.inference import preprocessing as pp

SR = C.SR


def det_delays(params, gps_epoch, detectors=DETS):
    """Per-detector geocentric time delays (s) via bilby."""
    import bilby
    out = {}
    for det in detectors:
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        out[det] = float(ifo.time_delay_from_geocenter(
            params["ra"], params["dec"], gps_epoch))
    return out


def delayed_signal(params, gps_epoch, sign=+1, detectors=DETS):
    """_common's projected signal with per-detector time delays applied
    (sign=-1 mirrors the geometry)."""
    sig = C.raw_projected_signal(params, gps_epoch, detectors=detectors)
    dts = det_delays(params, gps_epoch, detectors)
    for det in detectors:
        sig[det] = np.roll(sig[det], sign * int(round(dts[det] * SR)))
    return sig, dts


def inject_real_delayed(off_gps, params, sign=+1, seed=0):
    """inject_into_real_noise with the delay-corrected signal."""
    gps, strain, starts, srs, found = pp.fetch_gwosc(
        off_gps, detectors=list(DETS), fetch_seconds=64)
    sig, dts = delayed_signal(params, off_gps, sign=sign, detectors=tuple(found))
    raw = {}
    for det in found:
        v = np.array(strain[det], dtype=np.float64)
        c = int(round((off_gps - starts[det]) * SR))
        i0 = c - pp.T_LEN // 2
        v[i0:i0 + pp.T_LEN] = v[i0:i0 + pp.T_LEN] + sig[det]
        raw[det] = v
    prep = pp.prepare_real(raw, off_gps, starts, srs, seed=seed)
    return prep, dts


def make_design_delayed(params, seed, sign=+1):
    sig, dts = delayed_signal(params, C.GW150914_GPS, sign=sign)
    raw, psds = {}, {}
    for i, det in enumerate(DETS):
        raw[det] = design_noise(det, seed * 10 + i) + sig[det]
        psds[det] = get_default_psd(det)
    return pp.prepare_simulated(raw, psds=psds, seed=seed), dts


def audit_training_delays(data_dir="data/dataset", max_events=12):
    """H1-L1 xcorr lag for the loudest single-signal validation events."""
    print("=== K: training-data H1-L1 delay audit ===")
    rows = []
    for fp in sorted(glob.glob(f"{data_dir}/validation/batch_*.pkl"))[:4]:
        with open(fp, "rb") as fh:
            b = pickle.load(fh)
        for s in b["samples"]:
            if str(s.get("event_type")) == "noise" or int(s.get("n_signals", 1)) != 1:
                continue
            snr = float(s.get("network_snr", 0.0))
            if snr < 18:
                continue
            st = np.stack([np.asarray(s["detector_data"][d]["strain"], dtype=np.float32)
                           for d in ("H1", "L1", "V1")])
            p0 = s["parameters"][0]
            rows.append((snr, h1l1_lag_ms(st), float(p0["ra"]), float(p0["dec"])))
            if len(rows) >= max_events:
                break
        if len(rows) >= max_events:
            break
    for snr, lag, ra, dec in rows:
        print(f"  snr {snr:5.1f}  lag {lag:+6.2f} ms   ra {ra:+.2f} dec {dec:+.2f}")
    lags = np.array([r[1] for r in rows])
    print(f"  -> n={len(lags)}  mean|lag| {np.abs(lags).mean():.2f} ms  "
          f"max|lag| {np.abs(lags).max():.2f} ms  (physical range up to ~10 ms)")


def main():
    audit_training_delays()

    models = {name: C.load_model(path) for name, path in MODELS.items()}
    zeros_ab = np.zeros((3, 16), dtype=np.float32)
    truth = dict(C.GW150914_TRUTH)
    print(f"\npseudo-truth sky delays at event GPS: "
          f"{ {k: round(v*1000, 2) for k, v in det_delays(truth, C.GW150914_GPS).items()} } ms")

    print("\n=== H: delay-corrected twin, off-source real noise (-300 s) ===")
    prep, _ = inject_real_delayed(C.GW150914_GPS - 300.0, truth, sign=+1)
    ab = C.asd_bands_of(prep)
    print(f"  measured twin lag: {h1l1_lag_ms(prep.strain):+.2f} ms (real event: +6.84)")
    for name, m in models.items():
        analyze(m, prep, truth, ab, f"H delay+ / {name}")

    print("\n=== I: mirror control (delays negated) ===")
    prep_m, _ = inject_real_delayed(C.GW150914_GPS - 300.0, truth, sign=-1)
    ab_m = C.asd_bands_of(prep_m)
    print(f"  measured mirror lag: {h1l1_lag_ms(prep_m.strain):+.2f} ms")
    for name, m in models.items():
        analyze(m, prep_m, truth, ab_m, f"I delay- / {name}")

    print("\n=== J: delay-corrected twin, Gaussian design noise ===")
    for dl in (440.0, 1564.0):
        p = dict(truth, luminosity_distance=dl)
        prep_j, _ = make_design_delayed(p, seed=int(dl))
        print(f"  dL {dl:.0f}: lag {h1l1_lag_ms(prep_j.strain):+.2f} ms")
        for name, m in models.items():
            analyze(m, prep_j, p, zeros_ab, f"J dL{dl:.0f} / {name}")


if __name__ == "__main__":
    main()
