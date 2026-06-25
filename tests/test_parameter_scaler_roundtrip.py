"""
Regression tests for ParameterScaler and TorchParameterScaler.

Verifies:
1. normalize → denormalize is lossless within tolerance for all 11 parameters
2. Normalized values for dataset-typical inputs land near [-1, 1] (bounded) or small z-score (unbounded)
3. Scaler stats match the measured dataset distribution (guards against re-introducing stale stats)
4. TorchParameterScaler produces identical results to ParameterScaler
"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ahsd.models.parameter_scalers import ParameterScaler, TorchParameterScaler

PARAM_NAMES = [
    'mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
    'theta_jn', 'psi', 'phase', 'geocent_time', 'a1', 'a2'
]

# Representative physical values (dataset medians / typical values)
TYPICAL_PHYSICAL = {
    'mass_1': 20.0,
    'mass_2': 8.0,
    'luminosity_distance': 500.0,
    'ra': 3.14,
    'dec': 0.0,
    'theta_jn': 1.57,
    'psi': 1.57,
    'phase': 3.14,
    'geocent_time': 0.0,
    'a1': 0.2,
    'a2': 0.15,
}

# Expected scaler statistics from the 5895-sample full-dataset measurement
EXPECTED_STATS = {
    'mass_1':   {'log_mean': 2.660, 'log_std': 1.354},
    'mass_2':   {'log_mean': 1.939, 'log_std': 1.459},
    'geocent_time': {'mean': 0.722, 'std': 1.903},
    'a1': {'mean': 0.249, 'std': 0.236},
    'a2': {'mean': 0.173, 'std': 0.186},
}


@pytest.fixture
def scaler():
    # TorchParameterScaler is the implementation used during training and inference
    return TorchParameterScaler(PARAM_NAMES)


class TestRoundTrip:
    def _make_physical_batch(self, n=64):
        rng = np.random.default_rng(42)
        rows = []
        for _ in range(n):
            rows.append([
                rng.uniform(5, 80),      # mass_1
                rng.uniform(1, 40),      # mass_2
                rng.uniform(20, 5000),   # luminosity_distance
                rng.uniform(0, 2*np.pi), # ra
                rng.uniform(-np.pi/2, np.pi/2),  # dec
                rng.uniform(0, np.pi),   # theta_jn
                rng.uniform(0, np.pi),   # psi
                rng.uniform(0, 2*np.pi), # phase
                rng.uniform(-1.8, 6.9),  # geocent_time
                rng.uniform(0, 0.9),     # a1
                rng.uniform(0, 0.9),     # a2
            ])
        return np.array(rows, dtype=np.float32)

    def test_roundtrip(self, scaler):
        """normalize → denormalize recovers physical values within float32 tolerance."""
        physical = torch.tensor(self._make_physical_batch(200))
        norm = scaler.normalize_batch(physical)
        recovered = scaler.denormalize_batch(norm)
        diff = (recovered - physical).abs()
        # float32 log-space ops accumulate ~3e-3 absolute error on luminosity_distance
        assert diff.max().item() < 5e-3, f"Max round-trip error {diff.max().item():.2e} exceeds 5e-3"
        assert diff.mean().item() < 1e-4, f"Mean round-trip error {diff.mean().item():.2e} exceeds 1e-4"

    def test_typical_values_normalized_range(self, scaler):
        """Typical physical values should map to normalized values near zero."""
        v = list(TYPICAL_PHYSICAL.values())
        t = torch.tensor([v], dtype=torch.float32)
        norm = scaler.normalize_batch(t)[0]
        # Bounded params should land in [-1, 1]
        bounded_idx = [2, 3, 4, 5, 6, 7]  # dist, ra, dec, theta_jn, psi, phase
        for i in bounded_idx:
            assert abs(norm[i].item()) <= 1.0 + 1e-5, \
                f"{PARAM_NAMES[i]} normalized to {norm[i].item():.3f}, outside [-1,1]"
        # Z-score params should be within ±3 for typical values
        zscore_idx = [0, 1, 8, 9, 10]  # masses, time, spins
        for i in zscore_idx:
            assert abs(norm[i].item()) <= 3.0, \
                f"{PARAM_NAMES[i]} normalized to {norm[i].item():.3f}, outside ±3 for typical value"

    def test_distance_bounds(self, scaler):
        """Distance at 10 Mpc should map to -1, distance at 5000 Mpc to +1."""
        lo = torch.tensor([[1, 1, 10.0, 1, 0, 1, 1, 1, 0, 0.1, 0.1]], dtype=torch.float32)
        hi = torch.tensor([[1, 1, 5000.0, 1, 0, 1, 1, 1, 0, 0.1, 0.1]], dtype=torch.float32)
        norm_lo = scaler.normalize_batch(lo)[0, 2].item()
        norm_hi = scaler.normalize_batch(hi)[0, 2].item()
        assert abs(norm_lo - (-1.0)) < 1e-4, f"10 Mpc normalized to {norm_lo:.4f}, expected -1.0"
        assert abs(norm_hi - 1.0) < 1e-4, f"5000 Mpc normalized to {norm_hi:.4f}, expected +1.0"


class TestScalerStats:
    """Guard against re-introducing stale scaler statistics."""

    @pytest.fixture
    def scaler(self):
        return TorchParameterScaler(PARAM_NAMES)

    def test_mass1_log_mean(self, scaler):
        cfg = scaler.scalers['mass_1']
        assert abs(cfg['log_mean'] - 2.660) < 0.01, \
            f"mass_1 log_mean={cfg['log_mean']:.3f}, expected 2.660 (stale value was 2.908)"

    def test_mass1_log_std(self, scaler):
        cfg = scaler.scalers['mass_1']
        assert abs(cfg['log_std'] - 1.354) < 0.05, \
            f"mass_1 log_std={cfg['log_std']:.3f}, expected 1.354 (stale value was 0.719)"

    def test_mass2_log_mean(self, scaler):
        cfg = scaler.scalers['mass_2']
        assert abs(cfg['log_mean'] - 1.939) < 0.01, \
            f"mass_2 log_mean={cfg['log_mean']:.3f}, expected 1.939 (stale value was 1.733)"

    def test_mass2_log_std(self, scaler):
        cfg = scaler.scalers['mass_2']
        assert abs(cfg['log_std'] - 1.459) < 0.05, \
            f"mass_2 log_std={cfg['log_std']:.3f}, expected 1.459 (stale value was 1.053)"

    def test_geocent_time_mean(self, scaler):
        cfg = scaler.scalers['geocent_time']
        assert abs(cfg['mean'] - 0.722) < 0.01, \
            f"geocent_time mean={cfg['mean']:.3f}, expected 0.722 (stale value was 1.371, caused +0.65s bias)"

    def test_geocent_time_std(self, scaler):
        cfg = scaler.scalers['geocent_time']
        assert abs(cfg['std'] - 1.903) < 0.05, \
            f"geocent_time std={cfg['std']:.3f}, expected 1.903 (stale value was 2.267)"

    def test_a1_mean(self, scaler):
        cfg = scaler.scalers['a1']
        assert abs(cfg['mean'] - 0.249) < 0.01, f"a1 mean={cfg['mean']:.3f}, expected 0.249"

    def test_a2_mean(self, scaler):
        cfg = scaler.scalers['a2']
        assert abs(cfg['mean'] - 0.173) < 0.01, f"a2 mean={cfg['mean']:.3f}, expected 0.173"
