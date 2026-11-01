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


def test_fast_lengths_and_doubling_maxpool():
    """Feature version 3 speed-ups: the doubling max-filter equals
    max_pool1d on the circularly padded axis exactly, FFT lengths are
    2^a 3^b, and the profiles match version 2 (finer lag grid only)."""
    from ahsd.models.mc_features import McFeatures, circ_maxpool, fast_len
    assert [fast_len(n) for n in (4016, 8032, 6048, 32128)] == [4096, 8192, 6144, 32768]
    a = torch.rand(5, 3, 1000)
    for w in (1, 3, 12, 24):
        ref = torch.nn.functional.max_pool1d(torch.nn.functional.pad(a.reshape(-1, 1, 1000), (w, w), mode="circular"),
                                             2 * w + 1, stride=1).reshape(a.shape)
        assert torch.equal(circ_maxpool(a, w), ref)
    m3 = McFeatures("mc_bank_bbh.npz", 80, 4016)
    m2 = McFeatures("mc_bank_bbh.npz", 80, 4016)
    m2.feat_version.fill_(2.0); m2._sync_flags()
    x = _data(m3, 300 * m3.n_q, np.full(3, 0.3), np.zeros(3), amp=12.0)
    p3, p2 = m3._profiles(x)[0], m2._profiles(x)[0]
    assert float(p3.max()) == pytest.approx(float(p2.max()), rel=1e-2)   # the peak; noise maxima move a little
    assert m3.estimate(x).item() == pytest.approx(m2.estimate(x).item(), rel=2e-3)


def test_pivot_shift_is_consistent():
    """pivot=True: the flow sees parameters minus the bank's estimates; the
    inverse adds them back, so forward(inverse(z)) == z, and the density of
    a parameter equals the flow density of its offset."""
    from ahsd.models.lean_npe import LeanNPE
    torch.manual_seed(0)
    m = LeanNPE(psd_cond=True, encoder_type="coherent", mass_param="mc_q", dist_range=(80, 5000),
                mc_bank="mc_bank_bbh.npz:trigger", pivot=True).eval()
    assert m.flow.pivot_idx == [m.scaler.internal_names.index("chirp_mass"),
                                m.scaler.internal_names.index("geocent_time")]
    x = _data(m.encoder.mc_feat, 300 * m.encoder.mc_feat.n_q, np.full(3, 0.4), np.zeros(3), amp=30.0)
    ab = torch.zeros(1, 3, 16)
    with torch.no_grad():
        ctx = m.encode(x, ab)
        assert ctx.shape[1] == m.context_dim + 2
        mc_hat = float(torch.exp(m.encoder.mc_feat.last_log_mc))
        fc = m._full_context(ctx, torch.zeros(1, dtype=torch.long))
        z = torch.randn(1, 11)
        y, _ = m.flow.inverse(z, fc)
        z2, _ = m.flow.forward(y, fc)
        assert torch.allclose(z, z2, atol=1e-4)
        s = m.sample_posterior(x, n_samples=5, asd_bands=ab)
        assert torch.isfinite(s).all()
    t_hat = float(m.encoder.mc_feat.last_t)
    assert abs(t_hat - 0.4) < 2e-3 and mc_hat > 0
    # an old checkpoint (no buffer) loads without the pivot
    sd = {k: v for k, v in m.state_dict().items() if k != "pivot"}
    old = LeanNPE(psd_cond=True, encoder_type="coherent", mass_param="mc_q", dist_range=(80, 5000),
                  mc_bank="mc_bank_bbh.npz:trigger")
    old.load_state_dict(sd)
    assert old.flow.pivot_idx == [] and not bool(old.pivot)
