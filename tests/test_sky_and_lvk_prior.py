"""Sky position at the event's own epoch; LVK comparison prior."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ahsd.data.parameter_sampler import GPS_REF
from ahsd.inference.importance import symmetrized_log_q
from ahsd.inference.pipeline import ra_rotation
from ahsd.inference.priors import comoving_log_pdf
from ahsd.models.lean_npe import PARAM_NAMES, LeanNPE

GPS = 1239082262.1


def _prep(source, gps=GPS):
    return SimpleNamespace(source=source, trigger_gps=gps, strain=np.random.default_rng(0).normal(size=(3, 16384)).astype(np.float32),
                           strain_lo=None)


def test_rotation_only_on_real_events():
    assert ra_rotation(_prep("simulated")) == 0.0
    assert ra_rotation(_prep("real", GPS_REF)) == pytest.approx(0.0, abs=1e-9)
    assert abs(ra_rotation(_prep("real"))) > 0.01


@pytest.mark.parametrize("det", ["H1", "L1", "V1"])
def test_rotated_sky_has_the_same_detector_response(det):
    """The physical statement: (ra at GPS_REF) and (rotated ra at GPS) are one source."""
    import bilby
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    rot = ra_rotation(_prep("real"))
    for ra, dec, psi in ((0.3, 0.4, 1.0), (2.5, -1.0, 0.2), (5.0, 0.1, 2.5)):
        for m in ("plus", "cross"):
            a = ifo.antenna_response(ra, dec, GPS_REF, psi, m)
            b = ifo.antenna_response((ra + rot) % (2 * np.pi), dec, GPS, psi, m)
            assert a == pytest.approx(b, abs=2e-3)
        ta = ifo.time_delay_from_geocenter(ra, dec, GPS_REF)
        tb = ifo.time_delay_from_geocenter((ra + rot) % (2 * np.pi), dec, GPS)
        assert ta == pytest.approx(tb, abs=2e-5)


def test_flow_density_undoes_the_rotation():
    """The correction must evaluate the flow at the sky it sampled from."""
    torch.manual_seed(0)
    model = LeanNPE().eval()
    rng = np.random.default_rng(1)
    theta = model.scaler.denormalize(torch.from_numpy(rng.uniform(-0.8, 0.8, (8, 11))).float()).numpy().astype(np.float64)
    j1, j2 = PARAM_NAMES.index("mass_1"), PARAM_NAMES.index("mass_2")
    theta[:, j1], theta[:, j2] = theta[:, [j1, j2]].max(1), theta[:, [j1, j2]].min(1)
    jra = PARAM_NAMES.index("ra")
    rotated = theta.copy()
    rotated[:, jra] = np.mod(theta[:, jra] + ra_rotation(_prep("real")), 2 * np.pi)
    a = symmetrized_log_q(theta, model, _prep("simulated"))
    b = symmetrized_log_q(rotated, model, _prep("real"))
    assert np.allclose(a, b, atol=1e-2)


def test_comoving_prior_matches_bilby_and_is_below_d2_far_out():
    from bilby.gw.prior import UniformSourceFrame
    d = np.array([300.0, 1000.0, 2500.0, 5000.0])
    ours = comoving_log_pdf(d)
    bil = UniformSourceFrame(minimum=100, maximum=6000, name="luminosity_distance", cosmology="Planck15").ln_prob(d)
    assert np.allclose(ours - ours[0], bil - bil[0], atol=0.01)
    d2 = 2 * np.log(d)
    assert (ours - ours[0])[-1] < (d2 - d2[0])[-1] - 0.5   # favours far sources less than d^2


def test_wider_encoder_round_trips_through_saved_args():
    """A load site rebuilds a wider encoder from args alone."""
    from ahsd.models.lean_npe import LeanNPE
    a = {"encoder_kwargs": {"d_model": 240, "n_layers": 4}}
    m = LeanNPE(encoder_type="coherent", psd_cond=True, encoder_kwargs=a["encoder_kwargs"])
    n = LeanNPE(encoder_type="coherent", psd_cond=True, encoder_kwargs=a.get("encoder_kwargs"))
    n.load_state_dict(m.state_dict())
    x = torch.randn(2, 3, 16384)
    assert n.encode(x, torch.zeros(2, 3, 16)).shape[0] == 2
