import numpy as np
from ahsd.data.waveform_generator import WaveformGenerator


def test_snr_scales_with_amplitude():
    wg = WaveformGenerator(sample_rate=1024, duration=1.0)
    # simple sinusoid at 50 Hz
    t = np.linspace(0, wg.duration, wg.n_samples, endpoint=False)
    freq = 50.0
    signal = np.sin(2 * np.pi * freq * t)

    # PSD: flat (white noise) with value 1.0 across rfft bins
    freqs = np.fft.rfftfreq(len(signal), 1.0 / wg.sample_rate)
    psd = np.ones_like(freqs)

    snr1 = wg.compute_optimal_snr(signal, psd)
    alpha = 3.7
    snr2 = wg.compute_optimal_snr(signal * alpha, psd)

    # SNR should scale linearly with amplitude
    assert snr1 > 0
    assert np.allclose(snr2, snr1 * alpha, rtol=1e-6, atol=1e-8)
