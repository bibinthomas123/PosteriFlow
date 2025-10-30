import numpy as np
from ahsd.data.preprocessing import DataPreprocessor
from ahsd.utils.waveforms import WaveformUtilities


def test_whitening_preserves_snr_scaling():
    sampling_frequency = 1024
    duration = 1.0
    n_samples = int(sampling_frequency * duration)

    # simple sinusoid at 50 Hz
    t = np.linspace(0, duration, n_samples, endpoint=False)
    freq = 50.0
    base_wave = np.sin(2 * np.pi * freq * t)

    # PSD: flat (white noise) with value 1.0 across positive-frequency bins
    pos_freqs = np.fft.rfftfreq(n_samples, 1.0 / sampling_frequency)
    psd = np.ones_like(pos_freqs)
    psd_dict = {'psd': psd, 'frequencies': pos_freqs}

    preproc = DataPreprocessor(sample_rate=sampling_frequency, duration=duration)
    wu = WaveformUtilities(duration=duration, sampling_frequency=sampling_frequency)

    # Two amplitude scalings
    a1 = 1.0
    a2 = 3.7

    w1 = a1 * base_wave
    w2 = a2 * base_wave

    # Compute SNR in frequency-domain (before whitening) for reference
    snr1_ref = wu.estimate_snr(w1, psd)
    snr2_ref = wu.estimate_snr(w2, psd)

    # Whiten both waveforms using the preprocessor
    w1_white = preproc.whiten_data(w1, psd_dict)
    w2_white = preproc.whiten_data(w2, psd_dict)

    # After whitening, use unit PSD (whitened data should have white noise)
    unit_psd = np.ones_like(pos_freqs)
    snr1_white = wu.estimate_snr(w1_white, unit_psd)
    snr2_white = wu.estimate_snr(w2_white, unit_psd)

    # Sanity checks
    assert snr1_ref > 0
    assert snr2_ref > 0

    # SNR should scale approximately linearly with amplitude both before and after whitening
    ratio_ref = snr2_ref / snr1_ref
    ratio_white = snr2_white / snr1_white

    assert np.isfinite(ratio_ref)
    assert np.isfinite(ratio_white)

    # Allow small numerical tolerance
    assert np.allclose(ratio_ref, a2 / a1, rtol=1e-6, atol=1e-8)
    assert np.allclose(ratio_white, a2 / a1, rtol=1e-5, atol=1e-7)
