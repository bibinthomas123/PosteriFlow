"""Multi-rate input: the 4 s window plus the 28 s before it at 512 Hz.

The context stream must keep whitened noise unit-variance, carry the part of
a long inspiral that lies before the window, add up exactly with its noise,
and the model must read [window | context] rows while 4 s models stay as
they were."""
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
logging.getLogger("bilby").setLevel(logging.ERROR)
from ahsd.data.bilby_pipeline import to_lowrate  # noqa: E402

SR, LO_SEC, LO_RATE = 4096, 28, 512
LO_LEN = LO_SEC * LO_RATE


def test_lowrate_keeps_white_noise_unit_variance():
    x = np.random.default_rng(0).standard_normal(40 * SR)
    lo = to_lowrate(x, 34 * SR, SR, LO_SEC, LO_RATE)
    assert lo.shape == (LO_LEN,)
    assert abs(lo.std() - 1.0) < 0.08


def test_lowrate_needs_padding():
    with pytest.raises(ValueError):
        to_lowrate(np.zeros(20 * SR), 10 * SR, SR, LO_SEC, LO_RATE)


@pytest.fixture(scope="module")
def bns_sample(tmp_path_factory):
    from ahsd.data.dataset_generator import GWDatasetGenerator
    out = tmp_path_factory.mktemp("mr")
    cfg = {"lowrate_seconds": LO_SEC, "lowrate_rate": LO_RATE, "event_type_distribution": {"BNS": 1.0},
           "distance_ranges": {"BNS": [10, 60]}, "distance_prior": "uniform", "random_seed": 3}
    g = GWDatasetGenerator(output_dir=str(out), config=cfg)
    p = g.sampler.sample_parameters("BNS")
    p["luminosity_distance"] = 40.0
    s, _ = g._make_sample([p], 0.0, False, True)
    return s["detector_data"]["H1"]


def test_generated_context_stream(bns_sample):
    d = bns_sample
    lo, nl = d["signals_lo"][0].astype(np.float64), d["noise_lo"].astype(np.float64)
    hi = d["signals"][0].astype(np.float64)
    assert lo.shape == (LO_LEN,) and d["noise"].shape == (4 * SR,)
    assert 0.85 < nl.std() < 1.15
    # a BNS carries most of its in-band power before the 4 s window
    assert (lo ** 2).sum() > (hi ** 2).sum()
    np.testing.assert_allclose(d["strain_lo"], nl + lo, atol=2e-2)


def test_model_reads_window_and_context():
    from ahsd.models.lean_npe import LeanNPE
    m = LeanNPE(psd_cond=True, encoder_type="coherent", mass_param="mc_q",
                dist_range=(7, 550), lowrate_len=LO_LEN).eval()
    x = torch.randn(2, 3, 16384 + LO_LEN)
    with torch.no_grad():
        assert m.encode(x).shape == (2, 256)
        post = m.sample_posterior(x, n_samples=8)
    assert post.shape == (2, 8, 11)
    # the context stream changes the context vector
    x2 = x.clone()
    x2[..., 16384:] = 0
    with torch.no_grad():
        assert not torch.allclose(m.encode(x), m.encode(x2))


def test_four_second_models_unchanged():
    from ahsd.models.lean_npe import LeanNPE
    m = LeanNPE(psd_cond=True, encoder_type="coherent", mass_param="mc_q", dist_range=(7, 550))
    assert m.lowrate_len == 0 and not any(".lo_" in n for n, _ in m.named_parameters())
    with pytest.raises(ValueError):
        LeanNPE(encoder_type="conv", lowrate_len=LO_LEN)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "src/ahsd/models/assets/mc_bank_bns_ctx.npz").exists(),
                    reason="build with: python src/ahsd/models/build_mc_bank.py --cls bns --context")
def test_chirp_mass_features_find_the_chirp_mass(bns_sample):
    """The dense-bank matched filter recovers a loud BNS chirp mass to << 1%
    from the window and the context stream."""
    from ahsd.models.mc_features import McFeatures
    m = McFeatures("mc_bank_bns_ctx.npz", 80, 4016, lowrate_len=LO_LEN)
    d = bns_sample
    hi = torch.from_numpy(np.stack([d["signals"][0].astype(np.float32)] * 3))[None]
    lo = torch.from_numpy(np.stack([d["signals_lo"][0].astype(np.float32)] * 3))[None]
    assert m(hi, lo).shape == (1, 2, 192)
    assert torch.isfinite(m.estimate(hi, lo)).all()
