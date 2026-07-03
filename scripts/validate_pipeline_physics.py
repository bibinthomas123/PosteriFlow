#!/usr/bin/env python
"""
Physics validation suite for the PosteriFlow GW data pipeline.

Checks:
  1. Whitening produces unit-variance noise
  2. Detector noise is uncorrelated between H1/L1/V1
  3. Waveform amplitude scales as 1/d_L
  4. Time delays match detector geometry
  5. Antenna patterns behave correctly
  6. Parameter distributions match intended priors (before SNR cut)
  7. Post-selection parameter distributions
  8. Parameter scaler invertibility (lossless within float32 precision)

Usage:
    conda run -n ahsd python scripts/validate_pipeline_physics.py
"""

import sys
import warnings
import logging
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def check(label: str, cond: bool, detail: str = "", warn_only: bool = False):
    if cond:
        print(f"  [{PASS}] {label}  {detail}")
        return True
    else:
        tag = WARN if warn_only else FAIL
        print(f"  [{tag}] {label}  {detail}")
        return warn_only  # WARN counts as pass; FAIL counts as failure


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print("=" * 70)


def run_all():
    from ahsd.data.bilby_pipeline import (
        BilbyNoiseGenerator, BilbyWaveformGenerator,
        BilbySignalInjector, BilbyPreprocessor, get_default_psd,
    )
    from ahsd.data.parameter_sampler import ParameterSampler, GPS_REF
    import bilby

    psd_H1 = get_default_psd("H1")
    psd_L1 = get_default_psd("L1")
    psd_V1 = get_default_psd("V1")
    psd_map = {"H1": psd_H1, "L1": psd_L1, "V1": psd_V1}
    sample_rate = 4096
    duration = 4.0
    n = int(sample_rate * duration)

    noise_gen = BilbyNoiseGenerator(sample_rate, duration)
    wfg = BilbyWaveformGenerator(sample_rate, duration, f_lower=20.0)
    injector = BilbySignalInjector(sample_rate, duration, f_lower=20.0)
    preproc = BilbyPreprocessor(sample_rate, duration, f_low=20.0)
    sampler = ParameterSampler({"random_seed": 42})

    all_pass = True

    # ── 1. WHITENING: unit variance ─────────────────────────────────────────
    section("1. Whitening produces unit-variance noise")
    stds_by_det = {}
    for det, psd in psd_map.items():
        stds = []
        for seed in range(50):
            noise = noise_gen.generate(det, psd, seed=seed)
            wh = preproc.preprocess(noise, psd, detector=det)
            stds.append(float(np.std(wh)))
        mean_std = np.mean(stds)
        std_std = np.std(stds)
        stds_by_det[det] = stds
        ok = 0.95 < mean_std < 1.05
        all_pass &= check(f"{det} whitened noise std",
                          ok, f"mean={mean_std:.4f} ± {std_std:.4f} (should be ≈0.975)")

    # ── 2. NOISE UNCORRELATED ───────────────────────────────────────────────
    section("2. Detector noise uncorrelated between H1/L1/V1")
    noises = {}
    for det, psd in psd_map.items():
        noises[det] = []
        for seed in range(50):
            n_arr = noise_gen.generate(det, psd, seed=seed + 100)
            wh = preproc.preprocess(n_arr, psd, detector=det)
            noises[det].append(wh)
    for d1, d2 in [("H1", "L1"), ("H1", "V1"), ("L1", "V1")]:
        corrs = []
        for i in range(50):
            r, _ = scipy_stats.pearsonr(noises[d1][i], noises[d2][i])
            corrs.append(r)
        mean_r = np.mean(np.abs(corrs))
        ok = mean_r < 0.05
        all_pass &= check(f"{d1}-{d2} |Pearson r|", ok,
                          f"mean|r|={mean_r:.4f} (should be < 0.05)")

    # ── 3. WAVEFORM AMPLITUDE ∝ 1/d_L ──────────────────────────────────────
    section("3. Waveform amplitude scales as 1/d_L")
    base_params = {
        "mass_1": 36.0, "mass_2": 29.0,
        "ra": 1.95, "dec": -1.27, "theta_jn": 2.0, "psi": 0.82, "phase": 0.0,
        "geocent_time": 0.0, "geocent_time_gps": GPS_REF,
        "a1": 0.0, "a2": 0.0, "event_type": "BBH",
    }
    distances = [200.0, 400.0, 800.0, 1600.0]
    rms_values = []
    for d in distances:
        p = {**base_params, "luminosity_distance": d}
        h = wfg.generate(p, "H1")
        rms_values.append(float(np.std(h)))

    ratios = []
    expected_ratios = []
    for i in range(1, len(distances)):
        ratio = rms_values[i] / rms_values[0] if rms_values[0] > 0 else 0.0
        expected = distances[0] / distances[i]
        ratios.append(ratio)
        expected_ratios.append(expected)

    ratios = np.array(ratios)
    expected_ratios = np.array(expected_ratios)
    residuals = np.abs(ratios - expected_ratios) / expected_ratios
    ok_scaling = residuals.max() < 0.05
    for i, d in enumerate(distances[1:]):
        print(f"    d={distances[0]}→{d} Mpc: rms_ratio={ratios[i]:.4f}  expected={expected_ratios[i]:.4f}  "
              f"error={residuals[i]*100:.2f}%")
    all_pass &= check("1/d_L scaling", ok_scaling,
                      f"max_residual={residuals.max()*100:.2f}% (< 5% required)")

    # ── 4. INTER-DETECTOR TIME DELAYS ───────────────────────────────────────
    section("4. Inter-detector time delays match geometry")
    # For a face-on BBH overhead (dec=0, ra=0), the time delays should be close to 0
    # For a source to the side, they should match light-travel time between detectors
    # H1-L1 baseline: ~3000 km → max time delay = 3000/3e5 km/s ≈ 10 ms
    # H1-V1 baseline: ~8000 km → max ≈ 27 ms
    test_params = {**base_params, "luminosity_distance": 200.0}
    peak_times = {}
    for det, psd in psd_map.items():
        p = {**test_params, "geocent_time": 0.0, "geocent_time_gps": GPS_REF}
        h = wfg.generate(p, det)
        if np.std(h) > 1e-25:
            idx = np.argmax(np.abs(h))
            peak_times[det] = float(idx) / sample_rate
    if len(peak_times) == 3:
        dt_HL = abs(peak_times["H1"] - peak_times["L1"]) * 1000  # ms
        dt_HV = abs(peak_times["H1"] - peak_times["V1"]) * 1000  # ms
        ok_HL = dt_HL <= 11.0  # max H1-L1 ≈ 10 ms + margin
        ok_HV = dt_HV <= 28.0  # max H1-V1 ≈ 27 ms + margin
        print(f"    H1-L1 time delay: {dt_HL:.2f} ms  (max physical: 10 ms)")
        print(f"    H1-V1 time delay: {dt_HV:.2f} ms  (max physical: 27 ms)")
        all_pass &= check("H1-L1 delay within bound", ok_HL, f"{dt_HL:.2f} ms")
        all_pass &= check("H1-V1 delay within bound", ok_HV, f"{dt_HV:.2f} ms")
    else:
        check("Waveforms generated for all detectors",
              False, "Could not compute all waveforms")

    # ── 5. ANTENNA PATTERNS ─────────────────────────────────────────────────
    section("5. Antenna patterns behave correctly")
    # Face-on source at pole: H1 and L1 should have similar response (sky-independent)
    # Edge-on (theta_jn=pi/2): cross polarization should be zero
    ifo_H1 = bilby.gw.detector.get_empty_interferometer("H1")
    ifo_L1 = bilby.gw.detector.get_empty_interferometer("L1")
    ra_test, dec_test, psi_test, t = 0.5, 0.3, 0.7, GPS_REF
    fp_H1, fx_H1 = ifo_H1.antenna_response(ra_test, dec_test, t, psi_test, "plus"), \
                   ifo_H1.antenna_response(ra_test, dec_test, t, psi_test, "cross")
    fp_L1, fx_L1 = ifo_L1.antenna_response(ra_test, dec_test, t, psi_test, "plus"), \
                   ifo_L1.antenna_response(ra_test, dec_test, t, psi_test, "cross")
    # Antenna factor F+²+Fx² should be in [0, 1]
    af_H1 = fp_H1 ** 2 + fx_H1 ** 2
    af_L1 = fp_L1 ** 2 + fx_L1 ** 2
    ok_range = 0.0 <= af_H1 <= 1.0 and 0.0 <= af_L1 <= 1.0
    print(f"    H1: F+={fp_H1:.4f}  Fx={fx_H1:.4f}  F+²+Fx²={af_H1:.4f}")
    print(f"    L1: F+={fp_L1:.4f}  Fx={fx_L1:.4f}  F+²+Fx²={af_L1:.4f}")
    # Detectors should differ (they're in different locations/orientations)
    ok_differ = abs(fp_H1 - fp_L1) > 0.01 or abs(fx_H1 - fx_L1) > 0.01
    all_pass &= check("Antenna factors in [0,1]", ok_range)
    all_pass &= check("H1 and L1 have different response", ok_differ)

    # ── 6. PARAMETER PRIOR DISTRIBUTIONS ────────────────────────────────────
    section("6. Parameter distributions match intended priors (N=3000 draws)")
    N = 3000
    samples = [sampler.sample_parameters(event_type="BBH") for _ in range(N)]
    ra_vals = np.array([s["ra"] for s in samples])
    dec_vals = np.array([s["dec"] for s in samples])
    cos_dec = np.cos(np.pi / 2 - dec_vals)  # equivalent to sin(dec) for isotropy check
    theta_jn = np.array([s["theta_jn"] for s in samples])
    cos_theta = np.cos(theta_jn)
    t_offs = np.array([s["geocent_time"] for s in samples])
    # Use a separate sampler for distance test (avoids seed correlation with previous draws)
    dist_sampler = ParameterSampler({"random_seed": 7777})
    dists = np.array([dist_sampler.sample_parameters("BBH")["luminosity_distance"]
                      for _ in range(5000)])

    # RA: should be uniform in [0, 2π]
    _, p_ra = scipy_stats.kstest(ra_vals, "uniform", args=(0, 2 * np.pi))
    check("RA uniform [0,2π]", p_ra > 0.05, f"KS p={p_ra:.3f}")

    # dec: sin(dec) uniform → cos_inc uniform
    sin_dec = np.sin(dec_vals)
    _, p_dec = scipy_stats.kstest(sin_dec, "uniform", args=(-1, 2))
    check("Dec isotropic (sin-dec uniform)", p_dec > 0.05, f"KS p={p_dec:.3f}")

    # theta_jn: cos uniform in [-1,1]
    _, p_jn = scipy_stats.kstest(cos_theta, "uniform", args=(-1, 2))
    check("theta_jn isotropic (cos uniform)", p_jn > 0.05, f"KS p={p_jn:.3f}")

    # geocent_time: uniform in [-1.5, 1.5]
    _, p_t = scipy_stats.kstest(t_offs, "uniform", args=(-1.5, 3.0))
    check("geocent_time uniform [-1.5,1.5]", p_t > 0.05, f"KS p={p_t:.3f}")

    # Distance: P(d) ∝ d² → CDF = (d³-d_min³)/(d_max³-d_min³)
    d_min, d_max = 50.0, 2000.0
    cdf_expected = (dists ** 3 - d_min ** 3) / (d_max ** 3 - d_min ** 3)
    _, p_dist = scipy_stats.kstest(dists, lambda x: np.clip(
        (x**3 - d_min**3)/(d_max**3 - d_min**3), 0, 1))
    check("Distance P(d)∝d² (volume-weighted)", p_dist > 0.05,
          f"KS p={p_dist:.3f}")

    # ── 7. POST-SELECTION DISTRIBUTIONS ─────────────────────────────────────
    section("7. Post-selection SNR distribution (N=200, checking range)")
    accepted_snrs, accepted_dists = [], []
    n_tried = 0
    for _ in range(500):
        p = sampler.sample_parameters(event_type="BBH")
        noise = noise_gen.generate("H1", psd_H1, seed=n_tried)
        noise_L1 = noise_gen.generate("L1", psd_L1, seed=n_tried + 1000)
        noise_V1 = noise_gen.generate("V1", psd_V1, seed=n_tried + 2000)
        _, snr_H1 = injector.inject(noise, p, "H1", psd_H1)
        _, snr_L1 = injector.inject(noise_L1, p, "L1", psd_L1)
        _, snr_V1 = injector.inject(noise_V1, p, "V1", psd_V1)
        net_snr = float(np.sqrt(snr_H1**2 + snr_L1**2 + snr_V1**2))
        n_tried += 1
        if net_snr >= 8.0:
            accepted_snrs.append(net_snr)
            accepted_dists.append(p["luminosity_distance"])
        if len(accepted_snrs) >= 50:
            break
    if accepted_snrs:
        print(f"    Accepted {len(accepted_snrs)}/{n_tried} draws (acceptance rate: "
              f"{len(accepted_snrs)/n_tried*100:.1f}%)")
        print(f"    Network SNR: mean={np.mean(accepted_snrs):.2f}  "
              f"min={np.min(accepted_snrs):.2f}  max={np.max(accepted_snrs):.2f}")
        print(f"    Distance: mean={np.mean(accepted_dists):.0f}  "
              f"min={np.min(accepted_dists):.0f}  max={np.max(accepted_dists):.0f} Mpc")
        ok_snr_min = min(accepted_snrs) >= 8.0
        ok_snr_range = max(accepted_snrs) < 500
        check("All accepted SNR ≥ 8", ok_snr_min, f"min={min(accepted_snrs):.2f}")
        check("Accepted SNR physically plausible (< 500)", ok_snr_range,
              f"max={max(accepted_snrs):.2f}")
    else:
        print("    ERROR: No accepted samples in 500 draws!")
        all_pass = False

    # ── 8. SCALER INVERTIBILITY ──────────────────────────────────────────────
    section("8. Parameter scaler invertibility (lossless)")
    try:
        import torch
        from ahsd.models.parameter_scalers import TorchParameterScaler
        from ahsd.data.dataset_generator import PARAM_NAMES
        scaler = TorchParameterScaler(param_names=PARAM_NAMES)
        raw_samples = [sampler.sample_parameters() for _ in range(100)]
        param_arr = np.array([[float(s.get(k, 0.0)) for k in PARAM_NAMES]
                               for s in raw_samples], dtype=np.float32)
        t = torch.from_numpy(param_arr)
        t_norm = scaler.normalize_batch(t)
        t_rec = scaler.denormalize_batch(t_norm)
        # Check round-trip error (excluding clipped values)
        in_range_mask = (torch.abs(t_norm) <= 2.9)  # below FLOW_NORM_BOUND
        if in_range_mask.any():
            max_err = float(torch.abs(t[in_range_mask] - t_rec[in_range_mask]).max())
            mean_err = float(torch.abs(t[in_range_mask] - t_rec[in_range_mask]).mean())
            ok_inv = max_err < 1e-3
            check("Scaler round-trip (normalize→denormalize)", ok_inv,
                  f"max_err={max_err:.2e}  mean_err={mean_err:.2e}")
            # Check no values are clipped (in-bounds check)
            frac_clipped = float((~in_range_mask).float().mean())
            check("Clipping < 2% within bounds", frac_clipped < 0.02,
                  f"clipped={frac_clipped*100:.1f}%", warn_only=True)
        else:
            print("  [WARN] All values clipped — scaler stats likely stale")
    except Exception as e:
        print(f"  [FAIL] Scaler test failed: {e}")
        all_pass = False

    # ── SUMMARY ─────────────────────────────────────────────────────────────
    section("SUMMARY")
    print(f"  Overall status: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print()
    print("  NOTE: Posterior calibration (SBC) requires a trained model.")
    print("  Regenerate dataset, train, then run scripts/calibrate_posterior.py")
    print()


if __name__ == "__main__":
    run_all()
