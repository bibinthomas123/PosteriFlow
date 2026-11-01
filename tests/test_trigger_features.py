"""Trigger features: from the best template's complex SNR series, recover a
known arrival time per detector (sub-millisecond), the phase per detector,
and a second merger-time mode."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
BANK = Path(__file__).resolve().parents[1] / "src/ahsd/models/assets/mc_bank_bbh.npz"
pytestmark = pytest.mark.skipif(not BANK.exists(), reason="build with: python src/ahsd/models/build_mc_bank.py --cls bbh")
SR, T = 4096, 16384


@pytest.fixture(scope="module")
def mf():
    from ahsd.models.mc_features import McFeatures
    return McFeatures("mc_bank_bbh.npz:trigger", 80, 4016)


def _data(mf, k, t_det, alpha, amp=20.0, second=None):
    """Whitened rows holding template k at merger time t_det[d] (s from the
    window centre) with phase alpha[d], plus unit white noise."""
    H = torch.complex(mf.win_re, mf.win_im)[:, k].numpy()                # [D, F]
    f = np.arange(mf.band_lo, mf.band_lo + mf.n_band) / (T / SR)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, T))
    for d in range(3):
        for tt, a, s in [(t_det[d], alpha[d], amp)] + ([second] if second and d == 0 else []):
            spec = np.zeros(T // 2 + 1, complex)
            spec[mf.band_lo:mf.band_lo + mf.n_band] = H[d] * np.exp(-2j * np.pi * f * (tt + 2.0)) * np.exp(1j * a)
            x[d] += np.fft.irfft(spec, n=T) * s * math.sqrt(T / 2.0)
    return torch.from_numpy(x).float()[None]


def test_arrival_times_and_phases(mf):
    k = 300 * mf.n_q
    t_det, alpha = np.array([0.1003, 0.1071, 0.0842]), np.array([0.3, -1.2, 2.5])
    x = _data(mf, k, t_det, alpha, amp=500.0)
    prof, _, _, pq = mf._profiles(x)
    f = mf._trigger(x, prof, pq)[0].numpy()
    snr2 = np.sinh(f[[5, 9, 13]]) ** 2
    assert abs(f[0] * 1.5 - (snr2 * t_det).sum() / snr2.sum()) < 5e-5   # SNR^2-weighted arrival
    tau = f[[4, 8, 12]] * 0.012
    np.testing.assert_allclose(tau[1:] - tau[0], (t_det - t_det[0])[1:], atol=2e-5)
    phi = np.arctan2(f[[7, 11, 15]], f[[6, 10, 14]])
    assert np.abs(np.angle(np.exp(1j * (phi - alpha)))).max() < 0.1     # absolute phase per detector


def test_second_time_mode(mf):
    k = 300 * mf.n_q
    t = np.array([-0.4, -0.4, -0.4])
    x = _data(mf, k, t, np.zeros(3), amp=15.0, second=(0.8, 0.0, 12.0))
    prof, _, _, pq = mf._profiles(x)
    f = mf._trigger(x, prof, pq)[0].numpy()
    assert abs(f[0] * 1.5 + 0.4) < 2e-3 and abs(f[2] * 1.5 - 0.8) < 2e-3


def test_tokens_and_no_trigger_unchanged(mf):
    from ahsd.models.mc_features import McFeatures
    x = torch.randn(2, 3, T)
    assert mf(x).shape == (2, 4, 192)
    plain = McFeatures("mc_bank_bbh.npz", 80, 4016)
    assert plain(x).shape == (2, 2, 192) and not hasattr(plain, "trig_mlp")
