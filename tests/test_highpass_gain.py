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


def test_gap_gate_compares_equal_snr_not_raw_means():
    """Real events are quieter, so a large raw gap with small matched gaps must pass."""
    from ahsd.evaluation.snr_gap import nll_by_snr, snr_gap_gate
    rng = np.random.default_rng(0)
    nll_of = lambda snr: 5.0 - 6.0 * np.log(snr)        # one curve: NLL set by SNR alone
    sg = rng.uniform(10, 40, 4000)
    sr = sg * 0.4                                       # same events, quieter
    sr = sr[sr >= 6]
    g, r = nll_by_snr(nll_of(sg), sg), nll_by_snr(nll_of(sr), sr)
    assert nll_of(sr).mean() - nll_of(sg).mean() > 3
    name, ok, detail = snr_gap_gate(g, r)
    assert ok, detail
    # the same quiet events with a real penalty must fail
    assert not snr_gap_gate(g, nll_by_snr(nll_of(sr) + 4.0, sr))[1]


def test_gap_gate_fails_when_nothing_was_measured():
    from ahsd.evaluation.snr_gap import nll_by_snr, snr_gap_gate
    few = nll_by_snr(np.zeros(5), np.full(5, 15.0))
    name, ok, detail = snr_gap_gate(few, few)
    assert not ok and "no bin" in detail


@pytest.mark.parametrize("cls,m1,m2", [("NSBH", 8.0, 1.4), ("BNS", 1.5, 1.3)])
def test_long_signal_is_cut_not_wrapped(cls, m1, m2):
    """Merger at the centre of 8 s, inspiral before it, nothing wrapped in after it."""
    pytest.importorskip("lal")
    from ahsd.evaluation.path_audit import detector_signal
    p = dict(mass_1=m1, mass_2=m2, luminosity_distance=100.0, ra=1.0, dec=0.3, theta_jn=0.5,
             psi=0.2, phase=0.1)
    h = detector_signal(p, "H1", 1187008882.4, event_type=cls)
    assert len(h) == 8 * 4096
    peak = int(np.argmax(np.abs(h)))
    assert abs(peak - 4 * 4096) < 0.05 * 4096
    assert h[0] == 0.0                                       # tapered start
    assert np.abs(h[5 * 4096:]).max() < 1e-2 * np.abs(h).max()
    assert np.abs(h[4096:3 * 4096]).max() > 1e-3 * np.abs(h).max()   # inspiral kept


@pytest.mark.parametrize("cls,m1,m2", [("NSBH", 8.0, 1.4), ("BNS", 1.5, 1.3)])
def test_context_models_get_the_inspiral_through_the_context_stream(cls, m1, m2):
    """Merger 34 s in, 4 s after it: inspiral covers the window and the 28 s before it."""
    pytest.importorskip("lal")
    from ahsd.evaluation.path_audit import CONTEXT_BEFORE_S, detector_signal
    p = dict(mass_1=m1, mass_2=m2, luminosity_distance=100.0, ra=1.0, dec=0.3, theta_jn=0.5,
             psi=0.2, phase=0.1)
    h = detector_signal(p, "H1", 1187008882.4, event_type=cls, before_s=CONTEXT_BEFORE_S)
    assert len(h) == (CONTEXT_BEFORE_S + 4) * 4096
    assert abs(int(np.argmax(np.abs(h))) - CONTEXT_BEFORE_S * 4096) < 0.05 * 4096
    assert np.abs(h[2 * 4096:6 * 4096]).max() > 0                    # earliest context has signal
    assert np.abs(h[-3 * 4096:]).max() < 1e-2 * np.abs(h).max()      # nothing wrapped


def test_searched_fraction_separates_inside_from_excluded():
    from ahsd.evaluation.path_audit import searched_fraction
    rng = np.random.default_rng(0)
    ra = rng.normal(2.0, 0.05, 20000)
    dec = rng.normal(0.3, 0.05, 20000)
    assert searched_fraction(ra, dec, 2.0, 0.3) < 0.5
    assert searched_fraction(ra, dec, 5.0, -0.8) > 0.95


def test_searched_fraction_is_calibrated_for_a_broad_posterior():
    """Truth drawn from the same uniform sky: the mean is 0.5 although nothing localizes."""
    from ahsd.evaluation.path_audit import searched_fraction
    rng = np.random.default_rng(1)
    v = []
    for _ in range(300):
        ra, dec = rng.uniform(0, 2 * np.pi, 5000), np.arcsin(rng.uniform(-1, 1, 5000))
        v.append(searched_fraction(ra, dec, rng.uniform(0, 2 * np.pi), np.arcsin(rng.uniform(-1, 1))))
    assert abs(np.mean(v) - 0.5) < 0.05


def _gates(**kw):
    from ahsd.evaluation.path_audit import path_gates
    r = dict(n=20, n_sky=10, amplitude=1.0, ra_error=0.3, searched=0.55, distance=1.0)
    r.update(kw)
    return {name.split(" on ")[0]: ok for name, ok, _ in path_gates(r)}


def test_sky_gates_pass_a_good_model():
    assert all(_gates().values())


def test_sky_gate_fails_when_too_few_injections_count():
    """An unmeasured sky must not pass, even with a small error."""
    assert not _gates(n_sky=3)["right ascension"]
    assert not _gates(n_sky=0, ra_error=float("nan"))["right ascension"]


def test_sky_gates_fail_a_wrong_sky():
    assert not _gates(ra_error=1.1)["right ascension"]
    assert not _gates(searched=0.85)["sky calibration"]


def test_searched_fraction_is_calibrated_for_a_narrow_posterior():
    from ahsd.evaluation.path_audit import searched_fraction
    rng = np.random.default_rng(2)
    v = []
    for _ in range(300):
        c = rng.uniform(1, 5), rng.uniform(-0.8, 0.8)
        draw = lambda n: (rng.normal(c[0], 0.1, n), rng.normal(c[1], 0.1, n))
        tr, td = draw(1)
        v.append(searched_fraction(*draw(5000), float(tr[0]), float(td[0])))
    assert abs(np.mean(v) - 0.5) < 0.05


@pytest.mark.parametrize("cls,m1,m2", [("BNS", 1.5, 1.3), ("NSBH", 6.0, 1.4), ("BBH", 35.0, 30.0)])
def test_injection_carries_the_data_lowpass(cls, m1, m2):
    """Nothing above the data's roll-off, where dividing by the falling ASD would inflate it."""
    pytest.importorskip("lal")
    from ahsd.evaluation.path_audit import detector_signal
    p = dict(mass_1=m1, mass_2=m2, luminosity_distance=100.0, ra=1.0, dec=0.3, theta_jn=0.5,
             psi=0.2, phase=0.1)
    h = detector_signal(p, "H1", 1187008882.4, event_type=cls)
    P = np.abs(np.fft.rfft(h)) ** 2
    f = np.fft.rfftfreq(len(h), 1 / 4096)
    assert P[f >= 1900].sum() < 1e-3 * P.sum()      # only the start taper's leakage


def test_data_lowpass_keeps_the_band_and_the_phase():
    from ahsd.evaluation.path_audit import data_lowpass
    t = np.arange(4096 * 4) / 4096
    x = np.sin(2 * np.pi * 300.0 * t)
    np.testing.assert_allclose(data_lowpass(x), x, atol=1e-8)
    assert np.abs(data_lowpass(np.sin(2 * np.pi * 2000.0 * t))).max() < 1e-8


def test_likelihood_stops_below_the_data_lowpass():
    from ahsd.evaluation.path_audit import LOWPASS_HZ
    from ahsd.inference.priors import LIKELIHOOD_FMAX
    assert LIKELIHOOD_FMAX <= LOWPASS_HZ[0]


def test_recolouring_holds_the_filter_above_the_data_lowpass():
    """The clamp at Nyquist must not reach signals; below the limit nothing changes."""
    import sys
    sys.path.insert(0, "experiments")
    from remix_data import hold_above_band
    f = np.fft.rfftfreq(16384, 1 / 4096)
    filt = np.where(f < 1700, 1.0 + f / 2000, 50.0).astype(np.float32)
    out = hold_above_band(filt, f)
    np.testing.assert_array_equal(out[f < 1700], filt[f < 1700])
    assert out[f >= 1700].max() < 2.0 and len(set(out[f >= 1700].tolist())) == 1
    assert filt[-1] == 50.0                     # asd_bands keeps the measured filter


def test_noise_holdout_is_disjoint_and_spans_every_run():
    """Held-out segments are never trained on, cover each observing run, and the split
    needs only the fraction, so training and certification agree."""
    import sys
    from pathlib import Path
    sys.path.insert(0, "experiments")
    from remix_data import _era_of, noise_split_files
    gps = list(range(1126000000, 1137000000, 200000)) + list(range(1238000000, 1269000000, 200000))
    files = [Path(f"H1_{g}_strain.npy") for g in gps]
    tr, va = noise_split_files(files, 0.1, "train"), noise_split_files(files, 0.1, "val")
    assert not set(tr) & set(va) and set(tr) | set(va) == set(files)
    assert abs(len(va) / len(files) - 0.1) < 0.02
    era = lambda fs: {_era_of(int(f.name.split("_")[1])) for f in fs}
    assert era(va) == era(files)
    assert noise_split_files(files, 0.0, "train") == files and noise_split_files(files, 0.1, None) == files
    assert va == noise_split_files(list(reversed(files))[::-1], 0.1, "val")


def test_virgo_design_curve_is_real_and_old_models_keep_theirs():
    from ahsd.data.bilby_pipeline import design_asd_file, get_default_psd
    f = lambda p, x: float(np.sqrt(np.interp(x, p["frequencies"], p["psd"])))
    v = get_default_psd("V1")
    assert f(v, 20.0) > 5 * f(v, 100.0) and f(v, 1000.0) > 2 * f(v, 100.0)     # not flat
    assert f(v, 100.0) != f(get_default_psd("H1"), 100.0)                      # not aLIGO
    old = get_default_psd("V1", "flat")
    assert f(old, 20.0) == f(old, 1000.0)
    assert design_asd_file("V1") == "design_asd_V1.npy"                        # nothing recorded: flat
    assert design_asd_file("V1", "advirgo") == "design_asd_V1_advirgo.npy"
    assert design_asd_file("H1", "advirgo") == "design_asd_H1.npy"


def test_weighted_median():
    from ahsd.evaluation.path_audit import weighted_median
    assert weighted_median([1, 2, 3], [1, 1, 1]) == pytest.approx(2.0)
    assert weighted_median([1, 2, 3], [0, 0, 1]) == pytest.approx(3.0, abs=1e-9)
    assert weighted_median([3, 1, 2], [1, 1, 1]) == pytest.approx(2.0)


def test_distance_gate_centres_an_ideal_posterior():
    """Isotropic inclination at fixed SNR: an ideal posterior reads high unweighted and 1 weighted."""
    from ahsd.evaluation.path_audit import weighted_median
    amp = lambda c: np.sqrt(((1 + c ** 2) / 2) ** 2 + c ** 2) / np.sqrt(2)
    grid = np.linspace(0, 1, 20001)
    order = np.argsort(amp(grid))
    post = amp(grid)[order][np.searchsorted(np.cumsum(amp(grid)[order]) / amp(grid).sum(), 0.5)]
    truth = amp(np.random.default_rng(0).uniform(0, 1, 200000))
    ratio = post / truth
    assert 1.15 < np.median(ratio) < 1.23
    assert abs(weighted_median(ratio, truth) - 1.0) < 0.01


def test_gate_detail_reads_old_reports():
    from ahsd.evaluation.path_audit import path_gates
    assert all(ok for _, ok, _ in path_gates(dict(n=20, n_sky=10, amplitude=1.0, ra_error=0.3, searched=0.5, distance=1.0)))

