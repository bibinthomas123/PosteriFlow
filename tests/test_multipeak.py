"""Multi-signal chirp-mass peaks of the bank (mc_bank option 'multipeak')."""
import numpy as np
import pytest
import torch

pytest.importorskip("bilby")


def test_peeling_finds_a_quieter_overlapping_signal():
    from ahsd.data.bilby_pipeline import (BilbyNoiseGenerator, BilbyPreprocessor,
                                          BilbySignalInjector, get_default_psd)
    from ahsd.inference.dynesty_bridge import GPS_REF
    from ahsd.models.lean_npe import LeanNPE
    m = LeanNPE(psd_cond=True, encoder_type="coherent", mass_param="mc_q", pivot=True, spin_signed=True,
                mc_bank="mc_bank_bbh_spin_fine.npz:trigger,multipeak").eval()
    dets, sr, dur = ["H1", "L1", "V1"], 4096, 4.0
    ng, inj, prep = BilbyNoiseGenerator(sr, dur), BilbySignalInjector(sr, dur), BilbyPreprocessor(sr, dur)
    psds = {d: get_default_psd(d) for d in dets}
    sigs = [(36., 29., 500., 0.5), (16., 11., 450., -0.8)]
    white = []
    for i, d in enumerate(dets):
        x = ng.generate(d, psds[d], seed=i)
        for m1, m2, dl, tc in sigs:
            x, _ = inj.inject(x, dict(mass_1=m1, mass_2=m2, luminosity_distance=dl, theta_jn=0.4, ra=1.2,
                                      dec=0.3, psi=0.7, phase=1.0, geocent_time=tc, geocent_time_gps=GPS_REF + tc,
                                      a1=0.1, a2=-0.2, event_type="BBH"), d, psds[d])
        white.append(prep.preprocess(x, psds[d], detector=d))
    x = torch.tensor(np.stack(white), dtype=torch.float32)[None]
    with torch.no_grad():
        ctx = m.encode(x, torch.zeros(1, 3, m.encoder.psd_bands))
    mf = m.encoder.mc_feat
    mc = np.exp(mf.last_peak_log_mc[0].numpy())
    t = mf.last_peak_t[0].numpy()
    chirp = lambda a, b: (a * b) ** 0.6 / (a + b) ** 0.2  # noqa: E731
    assert abs(mc[0] / chirp(36, 29) - 1) < 0.03 and abs(t[0] - 0.5) < 0.02
    assert abs(mc[1] / chirp(16, 11) - 1) < 0.03 and abs(t[1] + 0.8) < 0.02
    # rank r gets peak r's pivot: the flow context differs between ranks only there
    f0, f1 = m._full_context(ctx, torch.tensor([0])), m._full_context(ctx, torch.tensor([1]))
    assert f0.shape == f1.shape and not torch.allclose(f0[:, -2:], f1[:, -2:])
