"""Training and real-event paths must give a signal the same whitened amplitude.

The bank's ASD is measured after a non-flat highpass and training signals never
pass through it; without the gain, training signals were ~1.4x too loud and the
network read every real event ~1.4x too far (analysis/distance_realdata/).
"""
import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from ahsd.data.bilby_pipeline import BilbyPreprocessor, get_default_psd, highpass_gain
from ahsd.inference.preprocessing import _whiten_real_segment, prepare_real

SR, N = 4096, 80 * 4096
F4 = np.fft.rfftfreq(16384, 1 / SR)


def _noise(seed=0):
    p = get_default_psd("H1")
    f = np.fft.rfftfreq(N, 1 / SR)
    asd = np.sqrt(np.interp(f, p["frequencies"], p["psd"]))
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) * asd * np.sqrt(SR * N / 4)
    return np.fft.irfft(x, n=N), p


def _burst(freq):
    t = np.arange(16384) / SR
    return 1e-21 * np.exp(-(((t - 2) / 0.1) ** 2)) * np.sin(2 * np.pi * freq * t)


def test_gain_matches_the_filter():
    x = np.random.default_rng(0).normal(size=2 ** 18)
    y = TimeSeries(x, sample_rate=SR).highpass(15).value
    f = np.fft.rfftfreq(len(x), 1 / SR)
    ratio = np.abs(np.fft.rfft(y) / np.fft.rfft(x))
    for lo, hi in ((35, 50), (50, 80), (80, 120), (200, 400)):
        m = (f >= lo) & (f < hi)
        assert np.median(ratio[m]) == pytest.approx(np.median(highpass_gain(f[m])), abs=0.01)
    assert highpass_gain(np.array([50.0]))[0] < 0.75   # the filter really is not flat


@pytest.mark.parametrize("freq", [50.0, 100.0, 300.0])
def test_training_and_inference_paths_agree(freq):
    noise, p = _noise()
    h = _burst(freq)
    i0 = 40 * SR - 8192
    with_sig = noise.copy()
    with_sig[i0:i0 + len(h)] += h
    w0, asd, _, _ = _whiten_real_segment(noise, SR, 40.0)
    w1, _, _, _ = _whiten_real_segment(with_sig, SR, 40.0)
    inference = np.sqrt(((w1 - w0) ** 2).sum())

    sig = BilbyPreprocessor(SR, 4.0).preprocess(h, p, detector="H1").astype(float)
    design = np.sqrt(np.interp(F4, p["frequencies"], p["psd"]))
    filt = design / np.maximum(asd, 1e-30) * highpass_gain(F4)
    training = np.sqrt((np.fft.irfft(np.fft.rfft(sig) * filt, n=16384) ** 2).sum())
    assert training / inference == pytest.approx(1.0, abs=0.08)


def test_prepared_psds_describe_unfiltered_strain():
    noise, p = _noise()
    prep = prepare_real({"H1": noise}, 40.0, {"H1": 0.0}, {"H1": SR}, seed=0)
    true = np.interp(F4, p["frequencies"], p["psd"])
    m = (F4 > 30) & (F4 < 1500)
    assert np.median(np.sqrt(prep.psds["H1"]["psd"][m] / true[m])) == pytest.approx(1.0, abs=0.08)
    # asds remain post-filter, the quantity asd_bands is defined on
    band = (F4 > 40) & (F4 < 60)
    assert np.median(prep.asds["H1"][band] / np.sqrt(true[band])) < 0.8


def test_supplied_psd_gives_unit_noise_and_true_amplitude():
    noise, p = _noise()
    h = _burst(100.0)
    i0 = 40 * SR - 8192
    with_sig = noise.copy()
    with_sig[i0:i0 + len(h)] += h
    f, a = np.asarray(p["frequencies"], float), np.sqrt(np.asarray(p["psd"], float))
    w0, _, norm, _ = _whiten_real_segment(noise, SR, 40.0, asd_freqs=f, asd_vals=a)
    w1, _, _, _ = _whiten_real_segment(with_sig, SR, 40.0, asd_freqs=f, asd_vals=a)
    assert norm / np.sqrt(SR / 2) == pytest.approx(1.0, abs=0.08)
    H = np.fft.rfft(h) / SR
    S = np.interp(F4, f, a ** 2)
    m = F4 >= 20
    snr = np.sqrt(4 * (np.abs(H[m]) ** 2 / S[m]).sum() / 4.0)
    assert np.sqrt(((w1 - w0) ** 2).sum()) / snr == pytest.approx(1.0, abs=0.08)


def test_certification_training_path_matches_inference():
    """path_audit's copy of the training path must stay equal to the real path."""
    from ahsd.evaluation.path_audit import AMP_RANGE, training_path_signal
    noise, p = _noise()
    h = _burst(100.0)
    with_sig = noise.copy()
    i0 = 40 * SR - 8192
    with_sig[i0:i0 + len(h)] += h
    w0, asd, _, _ = _whiten_real_segment(noise, SR, 40.0)
    w1, _, _, _ = _whiten_real_segment(with_sig, SR, 40.0)
    a = (w1 - w0).astype(float)
    b = training_path_signal(h, "H1", asd)
    assert AMP_RANGE[0] <= (a * b).sum() / (a * a).sum() <= AMP_RANGE[1]
