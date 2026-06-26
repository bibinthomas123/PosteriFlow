#!/usr/bin/env python3
"""
PosteriFlow Validation Suite
=============================
Single-command validation of dataset, model, and training health.

Usage:
    python scripts/validate.py --data-dir data/dataset
    python scripts/validate.py --data-dir data/dataset --model model/neuralpe_fresh/best_model.pth
    python scripts/validate.py --data-dir data/dataset --model model/... --strict

Exit codes:
    0 — all checks passed (or only warnings)
    1 — one or more FAIL checks
"""

import sys, os, argparse, json, time, logging
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

# ---------------------------------------------------------------------------
# Tolerances (override with --tolerances-json)
# ---------------------------------------------------------------------------
DEFAULT_TOL = {
    # Dataset
    "whitened_noise_std_min": 0.40,
    "whitened_noise_std_max": 1.60,
    "whitened_signal_std_min": 0.30,
    "detector_correlation_max": 0.15,
    "geocent_time_zero_frac_max": 0.02,   # <2% exact zeros
    "mass_ordering_fail_frac_max": 0.005,  # <0.5% samples with m1 < m2
    "spin_range_min": 0.0,
    "spin_range_max": 1.0,
    "distance_range_min": 5.0,
    "distance_range_max": 6000.0,
    "nan_inf_frac_max": 0.001,
    "snr_distance_spearman_min": -0.40,   # negative correlation expected; scatter from SNR regimes weakens it
    # Model
    "roundtrip_max_abs": 5e-3,
    "roundtrip_mean_abs": 1e-4,
    "flow_nll_max": 100.0,               # NLL per sample
    "flow_forward_inverse_max": 5e-4,    # |z - inv(fwd(z))| max — NSF splines are not exactly invertible
    "posterior_std_min": 0.001,          # samples must have non-trivial spread
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("validate")


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------
class Results:
    def __init__(self):
        self._rows = []       # (section, name, status, detail)

    def add(self, section, name, passed, detail="", warn=False):
        if passed is None:
            status = "SKIP"
        elif warn:
            status = "WARN"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        self._rows.append((section, name, status, detail))
        icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "SKIP": "—"}[status]
        log.info(f"  [{icon}] {name}: {detail}")

    def summary(self):
        failures = [r for r in self._rows if r[2] == "FAIL"]
        warnings = [r for r in self._rows if r[2] == "WARN"]
        skips    = [r for r in self._rows if r[2] == "SKIP"]
        passes   = [r for r in self._rows if r[2] == "PASS"]
        log.info("")
        log.info("=" * 70)
        log.info(f"SUMMARY  {len(passes)} passed  {len(warnings)} warnings  "
                 f"{len(failures)} failed  {len(skips)} skipped")
        if failures:
            log.info("\nFAILURES:")
            for _, name, _, detail in failures:
                log.info(f"  ✗ {name}: {detail}")
        log.info("=" * 70)
        return len(failures) == 0

    def to_dict(self):
        return [{"section": s, "name": n, "status": st, "detail": d}
                for s, n, st, d in self._rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_chunks(split_dir, max_chunks=None):
    import pickle, gzip
    split_dir = Path(split_dir)
    # Support chunk_*.pkl, chunk_*.pkl.gz, and batch_*.pkl.gz
    for pattern in ("chunk_*.pkl", "chunk_*.pkl.gz", "batch_*.pkl.gz", "batch_*.pkl"):
        files = sorted(split_dir.glob(pattern))
        if files:
            break
    if max_chunks:
        files = files[:max_chunks]
    samples = []
    for f in files:
        try:
            opener = gzip.open if f.suffix == ".gz" else open
            with opener(f, "rb") as fh:
                data = pickle.load(fh)
            samples.extend(data if isinstance(data, list) else data.get("samples", []))
        except Exception as e:
            log.warning(f"Failed to load {f.name}: {e}")
    return samples


def find_splits(data_dir):
    d = Path(data_dir)
    splits = {}
    # Named train/val/test subdirs
    for name in ("train", "validation", "val", "test"):
        p = d / name
        for pat in ("chunk_*.pkl", "chunk_*.pkl.gz", "batch_*.pkl.gz", "batch_*.pkl"):
            if p.is_dir() and list(p.glob(pat)):
                splits[name] = p
                break
    # Flat batches/ dir (ahsd-generate default output)
    batches_dir = d / "batches"
    if not splits and batches_dir.is_dir():
        for pat in ("batch_*.pkl.gz", "batch_*.pkl", "chunk_*.pkl"):
            if list(batches_dir.glob(pat)):
                splits["train"] = batches_dir
                break
    return splits


def spearman(x, y):
    from scipy.stats import spearmanr
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    r, _ = spearmanr(x[mask], y[mask])
    return float(r)


# ===========================================================================
# Section 1: DATASET
# ===========================================================================
def validate_dataset(data_dir, res, tol, max_chunks=10):
    log.info("")
    log.info("━" * 70)
    log.info("SECTION 1 — DATASET")
    log.info("━" * 70)

    splits = find_splits(data_dir)
    if not splits:
        res.add("dataset", "splits-found", False, f"No splits found in {data_dir}")
        return
    res.add("dataset", "splits-found", True, f"Found: {list(splits.keys())}")

    # Load a representative set from the first available split
    primary = splits.get("train") or next(iter(splits.values()))
    samples = load_chunks(primary, max_chunks=max_chunks)
    if not samples:
        res.add("dataset", "samples-loaded", False, "0 samples loaded")
        return
    res.add("dataset", "samples-loaded", True, f"{len(samples)} samples from train split")

    _validate_parameter_ranges(samples, res, tol)
    _validate_noise_whitening(samples, res, tol)
    _validate_detector_independence(samples, res, tol)
    _validate_snr_distance_scaling(samples, res, tol)
    _validate_serialization(samples, res, tol)
    _validate_nan_inf(samples, res, tol)


def _validate_parameter_ranges(samples, res, tol):
    log.info("\n  [1.1] Parameter ranges & distributions")
    PARAM_BOUNDS = {
        "mass_1":              (1.0,  200.0),
        "mass_2":              (0.5,  200.0),
        "luminosity_distance": (tol["distance_range_min"], tol["distance_range_max"]),
        "ra":                  (0.0,  2 * np.pi),
        "dec":                 (-np.pi / 2, np.pi / 2),
        "theta_jn":            (0.0,  np.pi),
        "psi":                 (0.0,  np.pi),
        "phase":               (0.0,  2 * np.pi),
        "geocent_time":        (-5.0, 10.0),
        "a1":                  (tol["spin_range_min"], tol["spin_range_max"]),
        "a2":                  (tol["spin_range_min"], tol["spin_range_max"]),
    }

    param_vals = {k: [] for k in PARAM_BOUNDS}
    mass_ordering_fails = 0
    geocent_zeros = 0
    n_signal = 0

    for s in samples:
        p = s.get("parameters")
        if not p:
            continue
        if isinstance(p, list):
            p = p[0]
        if not isinstance(p, dict):
            continue
        n_signal += 1
        for key in PARAM_BOUNDS:
            v = p.get(key)
            if v is not None and np.isfinite(float(v)):
                param_vals[key].append(float(v))
        m1, m2 = p.get("mass_1"), p.get("mass_2")
        if m1 is not None and m2 is not None:
            if float(m1) < float(m2):
                mass_ordering_fails += 1
        gt = p.get("geocent_time")
        if gt is not None and float(gt) == 0.0:
            geocent_zeros += 1

    for key, (lo, hi) in PARAM_BOUNDS.items():
        vals = np.array(param_vals[key])
        if len(vals) == 0:
            res.add("dataset", f"param-range-{key}", None, "not present in samples")
            continue
        out_of_range = np.sum((vals < lo) | (vals > hi))
        pct = 100.0 * out_of_range / len(vals)
        passed = pct < 0.5
        res.add("dataset", f"param-range-{key}", passed,
                f"n={len(vals)}, [{vals.min():.3g}, {vals.max():.3g}], "
                f"out-of-range={out_of_range} ({pct:.2f}%)")

    if n_signal > 0:
        fail_frac = mass_ordering_fails / n_signal
        res.add("dataset", "mass-ordering-m1>=m2", fail_frac <= tol["mass_ordering_fail_frac_max"],
                f"{mass_ordering_fails}/{n_signal} have m1 < m2 ({100*fail_frac:.2f}%)")
        zero_frac = geocent_zeros / n_signal
        res.add("dataset", "geocent-time-no-zeros", zero_frac <= tol["geocent_time_zero_frac_max"],
                f"{geocent_zeros}/{n_signal} have geocent_time=0.0 ({100*zero_frac:.2f}%)")


def _validate_noise_whitening(samples, res, tol):
    log.info("\n  [1.2] Noise & whitening statistics")

    # The stored 'noise' array is raw physical noise (~1e-23 strain), NOT whitened.
    # The stored 'strain' is the whitened (noise + signal) output used by the model.
    # To check whitening quality, use strain from noise-only samples where signal=0.
    noise_only_stds, all_strain_stds = [], []
    for s in samples[:300]:
        is_noise_only = s.get("type") == "noise"
        for det in ("H1", "L1", "V1"):
            dd = s.get("detector_data", {}).get(det, {})
            strain = dd.get("strain")
            if strain is not None:
                std = float(np.std(strain.astype(np.float64)))
                all_strain_stds.append(std)
                if is_noise_only:
                    noise_only_stds.append(std)

    lo, hi = tol["whitened_noise_std_min"], tol["whitened_noise_std_max"]
    if noise_only_stds:
        mean_std = np.mean(noise_only_stds)
        res.add("dataset", "noise-std-whitened",
                lo <= mean_std <= hi,
                f"noise-only strain: mean std={mean_std:.4f} (expected [{lo:.2f}, {hi:.2f}]), "
                f"n={len(noise_only_stds)}")
    else:
        res.add("dataset", "noise-std-whitened", None,
                "no noise-only samples in this chunk — cannot verify whitening amplitude")

    if all_strain_stds:
        arr = np.array(all_strain_stds)
        lo_floor = tol["whitened_signal_std_min"]
        pct_ok = 100.0 * np.mean(arr > lo_floor)
        res.add("dataset", "signal-std-above-noise-floor",
                pct_ok > 80.0,
                f"{pct_ok:.1f}% of detector strain > {lo_floor:.2f} (mean={arr.mean():.4f})")
        # Check for anomalous 1e-10 strain (pre-fix signature)
        n_anomalous = np.sum((arr > 1e-11) & (arr < 1e-9))
        res.add("dataset", "strain-not-anomalous-1e-10",
                n_anomalous == 0,
                f"{n_anomalous} detector samples with std in [1e-11, 1e-9] (should be 0)")
    else:
        res.add("dataset", "signal-std-above-noise-floor", None, "no strain arrays found")


def _validate_detector_independence(samples, res, tol):
    log.info("\n  [1.3] Detector independence")
    h1_noise, l1_noise, v1_noise = [], [], []

    for s in samples[:200]:
        dd = s.get("detector_data", {})
        n_h1 = dd.get("H1", {}).get("noise")
        n_l1 = dd.get("L1", {}).get("noise")
        n_v1 = dd.get("V1", {}).get("noise")
        if n_h1 is not None and n_l1 is not None and len(n_h1) == len(n_l1):
            h1_noise.append(n_h1.astype(np.float64))
            l1_noise.append(n_l1.astype(np.float64))
        if n_v1 is not None and n_h1 is not None and len(n_v1) == len(n_h1):
            v1_noise.append(n_v1.astype(np.float64))

    max_corr = tol["detector_correlation_max"]
    if h1_noise:
        corrs = [np.corrcoef(h1, l1)[0, 1] for h1, l1 in
                 zip(h1_noise[:50], l1_noise[:50])]
        mean_corr = float(np.mean(np.abs(corrs)))
        res.add("dataset", "H1-L1-noise-independence",
                mean_corr < max_corr,
                f"mean |corr|={mean_corr:.4f} (max allowed {max_corr})")
    else:
        res.add("dataset", "H1-L1-noise-independence", None, "no noise arrays found")

    if v1_noise:
        corrs = [np.corrcoef(h1, v1)[0, 1] for h1, v1 in
                 zip(h1_noise[:50], v1_noise[:50])]
        mean_corr = float(np.mean(np.abs(corrs)))
        res.add("dataset", "H1-V1-noise-independence",
                mean_corr < max_corr,
                f"mean |corr|={mean_corr:.4f} (max allowed {max_corr})")
    else:
        res.add("dataset", "H1-V1-noise-independence", None, "V1 noise not found")


def _validate_snr_distance_scaling(samples, res, tol):
    log.info("\n  [1.4] SNR-distance scaling")
    distances, snrs = [], []
    for s in samples[:500]:
        p = s.get("parameters")
        if not p:
            continue
        if isinstance(p, list):
            p = p[0]
        if not isinstance(p, dict):
            continue
        d = p.get("luminosity_distance")
        snr = p.get("target_snr")
        if d is not None and snr is not None:
            distances.append(float(d))
            snrs.append(float(snr))

    if len(distances) < 20:
        res.add("dataset", "snr-distance-anticorrelation", None,
                f"only {len(distances)} samples with both distance and SNR")
        return

    r = spearman(np.array(distances), np.array(snrs))
    min_r = tol["snr_distance_spearman_min"]
    res.add("dataset", "snr-distance-anticorrelation",
            r <= min_r,
            f"Spearman r={r:.3f} (expected <= {min_r}; SNR should decrease with distance)")


def _validate_serialization(samples, res, tol):
    log.info("\n  [1.5] Serialization integrity")
    REQUIRED_KEYS = {"sample_id", "parameters", "detector_data", "type"}
    REQUIRED_DET_KEYS = {"strain", "psd", "frequencies"}

    missing_keys, missing_det_keys = 0, 0
    total = len(samples)

    for s in samples[:500]:
        missing = REQUIRED_KEYS - set(s.keys())
        if missing:
            missing_keys += 1
            continue
        for det in ("H1", "L1"):
            dd = s.get("detector_data", {}).get(det, {})
            missing_d = REQUIRED_DET_KEYS - set(dd.keys())
            if missing_d:
                missing_det_keys += 1

    res.add("dataset", "samples-have-required-keys",
            missing_keys == 0,
            f"{missing_keys}/{min(total,500)} samples missing top-level keys")
    res.add("dataset", "detector-data-has-required-keys",
            missing_det_keys == 0,
            f"{missing_det_keys} H1/L1 detector entries missing strain/psd/frequencies")


def _validate_nan_inf(samples, res, tol):
    log.info("\n  [1.6] NaN/Inf scan")
    n_nan_inf = 0
    n_checked = 0
    for s in samples[:300]:
        for det in ("H1", "L1", "V1"):
            dd = s.get("detector_data", {}).get(det, {})
            strain = dd.get("strain")
            if strain is not None:
                n_checked += 1
                if not np.all(np.isfinite(strain)):
                    n_nan_inf += 1

    frac = n_nan_inf / n_checked if n_checked else 0
    res.add("dataset", "strain-finite",
            frac <= tol["nan_inf_frac_max"],
            f"{n_nan_inf}/{n_checked} detector arrays contain NaN/Inf ({100*frac:.3f}%)")


# ===========================================================================
# Section 2: MODEL
# ===========================================================================
def validate_model(model_path, data_dir, res, tol, device="cpu"):
    log.info("")
    log.info("━" * 70)
    log.info("SECTION 2 — MODEL")
    log.info("━" * 70)

    if not model_path or not Path(model_path).exists():
        res.add("model", "model-loadable", None, "no model path provided — skipping model checks")
        return

    # Load model
    try:
        from ahsd.models.overlap_neuralpe import OverlapNeuralPE
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        param_names = checkpoint.get("param_names", [
            "mass_1", "mass_2", "luminosity_distance", "ra", "dec",
            "theta_jn", "psi", "phase", "geocent_time", "a1", "a2",
        ])
        model = OverlapNeuralPE(
            param_names=param_names,
            priority_net_path=None,
            config=config,
            device=device,
        )
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        model.eval()
        res.add("model", "model-loadable", True, f"loaded from {model_path}")
    except Exception as e:
        res.add("model", "model-loadable", False, str(e))
        return

    _validate_parameter_scaler(res, tol)
    _validate_flow_forward_inverse(model, res, tol, device)
    _validate_nll(model, data_dir, res, tol, device)
    _validate_posterior_samples(model, data_dir, res, tol, device)
    _validate_nan_inf_model(model, data_dir, res, tol, device)


def _validate_parameter_scaler(res, tol):
    log.info("\n  [2.1] Parameter scaler round-trip")
    try:
        from ahsd.models.parameter_scalers import TorchParameterScaler
        PARAM_NAMES = [
            "mass_1", "mass_2", "luminosity_distance", "ra", "dec",
            "theta_jn", "psi", "phase", "geocent_time", "a1", "a2",
        ]
        scaler = TorchParameterScaler(PARAM_NAMES)
        rng = np.random.default_rng(42)
        batch = torch.tensor(np.column_stack([
            rng.uniform(5, 80, 512),
            rng.uniform(1, 40, 512),
            rng.uniform(20, 5000, 512),
            rng.uniform(0, 2 * np.pi, 512),
            rng.uniform(-np.pi / 2, np.pi / 2, 512),
            rng.uniform(0, np.pi, 512),
            rng.uniform(0, np.pi, 512),
            rng.uniform(0, 2 * np.pi, 512),
            rng.uniform(-1.8, 6.9, 512),
            rng.uniform(0, 0.9, 512),
            rng.uniform(0, 0.9, 512),
        ]), dtype=torch.float32)
        normed = scaler.normalize_batch(batch)
        recovered = scaler.denormalize_batch(normed)
        diff = (recovered - batch).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        res.add("model", "scaler-roundtrip-max",
                max_err < tol["roundtrip_max_abs"],
                f"max |error|={max_err:.2e} (tol {tol['roundtrip_max_abs']:.0e})")
        res.add("model", "scaler-roundtrip-mean",
                mean_err < tol["roundtrip_mean_abs"],
                f"mean |error|={mean_err:.2e} (tol {tol['roundtrip_mean_abs']:.0e})")
    except Exception as e:
        res.add("model", "scaler-roundtrip", False, str(e))


def _validate_flow_forward_inverse(model, res, tol, device):
    log.info("\n  [2.2] Flow forward/inverse consistency")
    try:
        flow = model.flow
        n_params = 11
        context_dim = model.context_dim_with_snr
        torch.manual_seed(0)
        z_orig = torch.randn(64, n_params).to(device)
        ctx = torch.randn(64, context_dim).to(device)
        with torch.no_grad():
            # forward: z → x
            x, log_jac = flow.flow._transform(z_orig, context=ctx)
            # inverse: x → z_back
            z_back, _ = flow.flow._transform.inverse(x, context=ctx)
        err = (z_back - z_orig).abs()
        max_err = err.max().item()
        res.add("model", "flow-forward-inverse",
                max_err < tol["flow_forward_inverse_max"],
                f"max |z - inv(fwd(z))|={max_err:.2e} (tol {tol['flow_forward_inverse_max']:.0e})")
    except Exception as e:
        res.add("model", "flow-forward-inverse", None, f"skipped: {e}")


def _build_strain_batch(data_dir, n=8, device="cpu"):
    splits = find_splits(data_dir)
    if not splits:
        return None, None
    primary = splits.get("validation") or splits.get("val") or next(iter(splits.values()))
    samples = load_chunks(primary, max_chunks=2)
    if not samples:
        return None, None
    strain_list, param_list = [], []
    for s in samples:
        if len(strain_list) >= n:
            break
        dd = s.get("detector_data", {})
        detectors = ["H1", "L1", "V1"]
        arrs = [dd.get(d, {}).get("strain") for d in detectors]
        if any(a is None for a in arrs):
            continue
        lengths = [len(a) for a in arrs]
        if len(set(lengths)) != 1:
            continue
        p = s.get("parameters")
        if p is None:
            continue
        if isinstance(p, list):
            p = p[0]
        PARAM_NAMES = ["mass_1","mass_2","luminosity_distance","ra","dec",
                       "theta_jn","psi","phase","geocent_time","a1","a2"]
        param_row = [float(p.get(k, 0)) for k in PARAM_NAMES]
        strain_list.append(np.stack([a.astype(np.float32) for a in arrs]))
        param_list.append(param_row)
    if not strain_list:
        return None, None
    strain_t = torch.tensor(np.stack(strain_list)).to(device)
    param_t  = torch.tensor(np.array(param_list, dtype=np.float32)).to(device)
    return strain_t, param_t


def _validate_nll(model, data_dir, res, tol, device):
    log.info("\n  [2.3] NLL correctness")
    strain, params = _build_strain_batch(data_dir, n=8, device=device)
    if strain is None:
        res.add("model", "nll-finite", None, "no validation samples available")
        return
    try:
        with torch.no_grad():
            loss_dict = model.compute_loss(strain, params.unsqueeze(1))
        nll = loss_dict.get("flow_nll", loss_dict.get("total_loss", torch.tensor(float("nan"))))
        nll_val = nll.item() if hasattr(nll, "item") else float(nll)
        res.add("model", "nll-finite",
                np.isfinite(nll_val),
                f"NLL={nll_val:.4f}")
        res.add("model", "nll-reasonable",
                nll_val < tol["flow_nll_max"],
                f"NLL={nll_val:.4f} (max allowed {tol['flow_nll_max']})")
    except Exception as e:
        res.add("model", "nll-finite", False, str(e))


def _validate_posterior_samples(model, data_dir, res, tol, device):
    log.info("\n  [2.4] Posterior sample validity")
    strain, params = _build_strain_batch(data_dir, n=4, device=device)
    if strain is None:
        res.add("model", "posterior-samples-valid", None, "no validation samples available")
        return
    try:
        with torch.no_grad():
            out = model.sample_posterior(strain[:2], n_samples=200, return_all_samples=True)
        if isinstance(out, dict):
            samples_t = out.get("samples", out.get("all_samples"))
        else:
            samples_t = out
        samples_np = samples_t.cpu().numpy() if hasattr(samples_t, "cpu") else np.array(samples_t)
        # shape: [batch, n_samples, n_params]
        has_nan = not np.all(np.isfinite(samples_np))
        res.add("model", "posterior-no-nan-inf", not has_nan,
                f"shape={samples_np.shape}, NaN/Inf={has_nan}")
        stds = samples_np.std(axis=1)   # [batch, n_params]
        min_std = stds.min()
        res.add("model", "posterior-has-spread",
                min_std > tol["posterior_std_min"],
                f"min std across params/batch={min_std:.4f} (min allowed {tol['posterior_std_min']})")
    except Exception as e:
        res.add("model", "posterior-samples-valid", None, f"skipped: {e}")


def _validate_nan_inf_model(model, data_dir, res, tol, device):
    log.info("\n  [2.5] NaN/Inf in model forward pass")
    strain, params = _build_strain_batch(data_dir, n=8, device=device)
    if strain is None:
        res.add("model", "forward-pass-finite", None, "no data available")
        return
    try:
        with torch.no_grad():
            loss_dict = model.compute_loss(strain, params.unsqueeze(1))
        any_nan = any(
            (torch.isnan(v) | torch.isinf(v)).any().item()
            for v in loss_dict.values()
            if isinstance(v, torch.Tensor)
        )
        res.add("model", "forward-pass-finite", not any_nan,
                "NaN/Inf in loss tensors" if any_nan else "all loss tensors finite")
    except Exception as e:
        res.add("model", "forward-pass-finite", False, str(e))


# ===========================================================================
# Section 3: TRAINING HEALTH (checkpoint history)
# ===========================================================================
def validate_training(model_dir, res, tol):
    log.info("")
    log.info("━" * 70)
    log.info("SECTION 3 — TRAINING HEALTH")
    log.info("━" * 70)

    if not model_dir:
        res.add("training", "training-history", None, "no model dir provided")
        return

    model_dir = Path(model_dir)

    # Load training history YAML if available
    history_path = model_dir / "training_history.yaml"
    run_log = next(
        (p for p in (model_dir / "run.log", model_dir / "training.log") if p.exists()),
        None,
    )

    if history_path.exists():
        import yaml
        with open(history_path) as f:
            history = yaml.safe_load(f) or {}
        _check_loss_trajectory(history, res)
        _check_gradient_norms(history, res)
        _check_parameter_biases(history, res)
    elif run_log is not None:
        _check_run_log(run_log, res)
    else:
        res.add("training", "training-history", None,
                f"no training_history.yaml or run.log in {model_dir}")


def _check_loss_trajectory(history, res):
    log.info("\n  [3.1] Loss trajectory")
    val_losses = history.get("val_loss", [])
    if not val_losses:
        res.add("training", "val-loss-present", None, "val_loss not in history")
        return
    arr = np.array([v for v in val_losses if v is not None and np.isfinite(v)])
    if len(arr) < 2:
        res.add("training", "val-loss-finite", len(arr) > 0, f"only {len(arr)} finite val losses")
        return
    res.add("training", "val-loss-finite", True,
            f"{len(arr)} epochs, range=[{arr.min():.4f}, {arr.max():.4f}]")
    # Check monotone improvement over last half of training
    half = len(arr) // 2
    improving = arr[-1] < arr[half]
    res.add("training", "val-loss-improving",
            improving,
            f"epoch {half} → last: {arr[half]:.4f} → {arr[-1]:.4f}",
            warn=not improving)


def _check_gradient_norms(history, res):
    log.info("\n  [3.2] Gradient norms")
    gnorms = history.get("grad_norm", [])
    if not gnorms:
        res.add("training", "grad-norms", None, "grad_norm not in history")
        return
    arr = np.array([v for v in gnorms if v is not None and np.isfinite(v)])
    if len(arr) == 0:
        return
    spikes = np.sum(arr > 100.0)
    res.add("training", "grad-norm-spikes",
            spikes < max(3, 0.05 * len(arr)),
            f"spikes >100: {spikes}/{len(arr)} ({100*spikes/len(arr):.1f}%)",
            warn=spikes > 0)
    diverged = arr[-1] > 200 if len(arr) > 0 else False
    res.add("training", "grad-norm-not-diverged",
            not diverged,
            f"final grad_norm={arr[-1]:.2f}")


def _check_parameter_biases(history, res):
    log.info("\n  [3.3] Parameter biases")
    biases = history.get("parameter_biases", {})
    if not biases:
        res.add("training", "parameter-biases", None, "no bias data in history")
        return
    CRITICAL_PARAMS = ["luminosity_distance", "mass_1", "geocent_time"]
    for param in CRITICAL_PARAMS:
        bias = biases.get(param)
        if bias is None:
            res.add("training", f"bias-{param}", None, "not recorded")
            continue
        final_bias = float(bias[-1]) if isinstance(bias, list) else float(bias)
        # bias in normalised units — warn if |bias| > 0.5 sigma
        res.add("training", f"bias-{param}",
                abs(final_bias) < 1.0,
                f"|bias|={abs(final_bias):.4f} normalised units",
                warn=abs(final_bias) > 0.5)


def _check_run_log(run_log, res):
    log.info("\n  [3.1-3.3] Parsing run.log")
    text = run_log.read_text(errors="replace")
    # Quick checks on log content
    has_nan_warn = "NaN" in text or "nan loss" in text.lower()
    has_inf_warn = "Inf" in text or "inf loss" in text.lower()
    # Count epoch lines
    import re
    epochs = re.findall(r"Epoch\s+(\d+)", text)
    res.add("training", "run-log-epochs",
            len(epochs) > 0,
            f"{len(epochs)} epoch lines found in log")
    res.add("training", "run-log-no-nan-explosion",
            not has_nan_warn,
            "NaN warnings detected in log" if has_nan_warn else "no NaN warnings",
            warn=has_nan_warn)
    res.add("training", "run-log-no-inf-explosion",
            not has_inf_warn,
            "Inf warnings detected in log" if has_inf_warn else "no Inf warnings",
            warn=has_inf_warn)


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="PosteriFlow validation suite")
    parser.add_argument("--data-dir",  default="data/dataset",
                        help="Dataset root directory (default: data/dataset)")
    parser.add_argument("--model",     default=None,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--model-dir", default=None,
                        help="Model run directory (for training_history.yaml / run.log)")
    parser.add_argument("--max-chunks", type=int, default=10,
                        help="Max dataset chunks to load per split (default: 10)")
    parser.add_argument("--device",    default="cpu",
                        help="Torch device for model checks (default: cpu)")
    parser.add_argument("--strict",    action="store_true",
                        help="Exit 1 even for warnings")
    parser.add_argument("--tolerances-json", default=None,
                        help="JSON file overriding default tolerances")
    parser.add_argument("--output-json", default=None,
                        help="Write results to JSON file")
    args = parser.parse_args()

    tol = dict(DEFAULT_TOL)
    if args.tolerances_json:
        with open(args.tolerances_json) as f:
            tol.update(json.load(f))

    res = Results()
    t0 = time.time()

    validate_dataset(args.data_dir, res, tol, max_chunks=args.max_chunks)

    model_dir = args.model_dir
    if not model_dir and args.model:
        model_dir = str(Path(args.model).parent)
    validate_model(args.model, args.data_dir, res, tol, device=args.device)
    validate_training(model_dir, res, tol)

    elapsed = time.time() - t0
    log.info(f"\nValidation completed in {elapsed:.1f}s")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"results": res.to_dict(), "tolerances": tol, "elapsed_s": elapsed}, f, indent=2)
        log.info(f"Results written to {args.output_json}")

    passed = res.summary()
    if not passed:
        sys.exit(1)
    if args.strict and any(r[2] == "WARN" for r in res._rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
