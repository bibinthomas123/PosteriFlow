"""Trigger alignment of the encoder's stem input (mc_bank option 'align')."""
import numpy as np
import torch

from ahsd.models.coherent_encoder import shift_to_centre


def test_shift_puts_each_detector_arrival_at_the_centre():
    sr, T = 4096.0, 16384
    t = np.arange(T) / sr - T / sr / 2                     # seconds from the window centre
    arr = np.array([[0.31, -0.42, 0.0071], [-1.2, 0.5, 0.25]])
    x = np.zeros((2, 3, T))
    for b in range(2):
        for d in range(3):
            x[b, d] = np.exp(-0.5 * ((t - arr[b, d]) / 2e-3) ** 2)   # pulse at the arrival
    y = shift_to_centre(torch.tensor(x), torch.tensor(arr), sr).numpy()
    for b in range(2):
        for d in range(3):
            centroid = (y[b, d] * t).sum() / y[b, d].sum()
            assert abs(centroid) < 1e-5, (b, d, centroid)


def test_shift_is_norm_preserving_and_invertible():
    """Exact for band-limited input (no power in the Nyquist bin, where a
    fractional shift of a real signal is not representable; whitened strain
    carries none above the analysis band)."""
    rng = np.random.default_rng(0)
    X = np.fft.rfft(rng.standard_normal((4, 3, 16384)), axis=-1)
    X[..., -1] = 0.0
    x = torch.tensor(np.fft.irfft(X, n=16384, axis=-1))
    a = torch.tensor(rng.uniform(-1.5, 1.5, (4, 3)))
    y = shift_to_centre(x, a)
    assert torch.allclose((y ** 2).sum(-1), (x ** 2).sum(-1), rtol=1e-6)
    back = shift_to_centre(y, -a)
    assert torch.allclose(back, x, atol=1e-4)
