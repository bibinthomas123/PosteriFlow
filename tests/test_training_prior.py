"""TrainingPrior must match the generator: every parameter set the sampler
draws for a class lies inside that class's prior support, the support ends
where the sampler's does, and the bilby priors carry the same bounds."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ahsd.data.parameter_sampler import ParameterSampler  # noqa: E402
from ahsd.inference.priors import TrainingPrior  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

RANGES = {"BBH": [100.0, 4000.0], "BNS": [10.0, 400.0], "NSBH": [20.0, 1200.0]}


def _draw(event_type, n=3000):
    s = ParameterSampler({"random_seed": 1, "distance_prior": "uniform", "distance_ranges": RANGES})
    return np.array([[p[k] for k in PARAM_NAMES] for p in s.sample_batch(n, event_type)])


@pytest.mark.parametrize("et", ["BBH", "NSBH", "BNS"])
def test_sampler_draws_inside_support(et):
    prior = TrainingPrior.for_class(et, RANGES, "uniform")
    assert np.isfinite(prior.log_prob(_draw(et))).all()


@pytest.mark.parametrize("et", ["BBH", "NSBH", "BNS"])
def test_support_bounds_match_sampler(et):
    prior = TrainingPrior.for_class(et, RANGES, "uniform")
    th = _draw(et, 400)
    j = PARAM_NAMES.index
    d = th.copy(); d[:, j("luminosity_distance")] = RANGES[et][1] * 1.05
    assert np.isneginf(prior.log_prob(d)).all()
    d = th.copy(); d[:, j("luminosity_distance")] = RANGES[et][0] * 0.95
    assert np.isneginf(prior.log_prob(d)).all()
    a = th.copy(); a[:, j("a1")] = prior.a1_max + 0.01
    assert np.isneginf(prior.log_prob(a)).all()


def test_class_and_dataset_ranges_come_from_meta():
    meta = {"train_event_types": ["BNS"], "train_distance_prior": "uniform",
            "train_distance_ranges": RANGES}
    p = TrainingPrior.from_meta(meta, event_type="BBH")      # single-class model wins
    assert p.event_type == "BNS" and p.dist_range == (10.0, 400.0)
    mixed = dict(meta, train_event_types=["BBH", "BNS", "NSBH"])
    assert TrainingPrior.from_meta(mixed, "NSBH").dist_range == (20.0, 1200.0)


def test_bns_mass_density_matches_sampler():
    """BNS: m1 ~ U[1, 2.5], m2 ~ U[1, m1] gives p(m1) flat; the prior density
    1/(m1 - 1) integrated over m2 must also be flat in m1."""
    prior = TrainingPrior.for_class("BNS", RANGES, "uniform")
    th = _draw("BNS", 4000)
    m1 = th[:, 0]
    h, _ = np.histogram(m1, bins=6, range=(1.0, 2.5))
    assert h.min() / h.max() > 0.8                 # sampler: flat in m1
    assert np.isfinite(prior.log_prob(th)).all()


@pytest.mark.parametrize("et,shape", [("BBH", "uniform"), ("BNS", "comoving_d2")])
def test_bilby_priors_bounds(et, shape):
    pytest.importorskip("bilby")
    prior = TrainingPrior.for_class(et, RANGES, shape)
    bp = prior.bilby_priors(1369224018.0)
    assert bp["luminosity_distance"].minimum == RANGES[et][0]
    assert bp["luminosity_distance"].maximum == RANGES[et][1]
    assert bp["mass_1"].minimum == prior.m1_range[0] and bp["mass_2"].maximum == prior.m2_range[1]
    assert bp["a_1"].maximum == prior.a1_max


def test_tides_sampled_and_in_priors():
    """Neutron stars carry a tidal deformability drawn from the generator's
    prior; black holes carry none. dynesty priors and IS draws use the same."""
    s = ParameterSampler({"random_seed": 2})
    bns, nsbh, bbh = (s.sample_parameters(t) for t in ("BNS", "NSBH", "BBH"))
    assert bns["lambda_1"] > 0 and bns["lambda_2"] > 0
    assert nsbh["lambda_1"] == 0 and nsbh["lambda_2"] > 0
    assert bbh["lambda_1"] == bbh["lambda_2"] == 0
    rng = np.random.default_rng(0)
    l1, l2 = TrainingPrior.for_class("NSBH", RANGES).sample_tides(500, rng)
    assert (l1 == 0).all() and 0 <= l2.min() and l2.max() <= 5000


def test_merger_time_gate_flags_wrong_mode():
    """A real event's window is centred on its trigger: a posterior merger
    ~1 s away (the GW190425/GW200115 failure) is a strong refine indicator."""
    from types import SimpleNamespace
    from ahsd.inference.gating import refinement_gate
    from ahsd.inference.result import PosteriorResult
    th = _draw("NSBH", 400)
    j = PARAM_NAMES.index("geocent_time")
    for tc, expect in ((0.0, False), (1.0, True)):
        x = th.copy(); x[:, j] = tc + np.random.default_rng(1).normal(0, 0.01, len(x))
        x[:, 0], x[:, 1] = 8.0, 1.4                      # narrow chirp mass
        r = SimpleNamespace(samples=x, param_names=list(PARAM_NAMES), prepared=None,
                            median={n: float(np.median(x[:, i])) for i, n in enumerate(PARAM_NAMES)},
                            trigger_gps=1263097407.7,
                            diagnostics={"verdict": {"confidence": "HIGH"}, "boundary_railing_frac": 0.0})
        r.modes = lambda p, r=r: PosteriorResult.modes(r, p)
        g = refinement_gate(r)
        assert any("trigger time" in m for m in g["strong_indicators"]) == expect


def test_merger_time_modes():
    """A bimodal merger-time posterior: the dominant mode is reported (the
    median falls between the modes) and the second mode is flagged."""
    from types import SimpleNamespace
    from ahsd.inference.gating import refinement_gate
    from ahsd.inference.result import PosteriorResult
    rng = np.random.default_rng(3)
    th = _draw("NSBH", 3000)
    j = PARAM_NAMES.index("geocent_time")
    th[:, j] = np.where(rng.random(3000) < 0.65, rng.normal(0.01, 0.005, 3000), rng.normal(1.1, 0.02, 3000))
    m = PosteriorResult.modes(SimpleNamespace(samples=th, param_names=list(PARAM_NAMES)))
    assert abs(m[0]["center"] - 0.01) < 0.01 and abs(m[0]["frac"] - 0.65) < 0.03
    assert abs(m[1]["center"] - 1.1) < 0.02
    th[:, 0], th[:, 1] = 8.0, 1.4
    r = SimpleNamespace(samples=th, param_names=list(PARAM_NAMES), prepared=None,
                        median={n: float(np.median(th[:, i])) for i, n in enumerate(PARAM_NAMES)},
                        trigger_gps=1263097407.7, modes=lambda p: m,
                        diagnostics={"verdict": {"confidence": "HIGH"}, "boundary_railing_frac": 0.0})
    g = refinement_gate(r)
    assert not any("trigger time" in s for s in g["strong_indicators"])
    assert any("bimodal" in s for s in g.get("moderate_indicators", []))


def test_signed_spins_sampler_and_prior():
    """signed_spins datasets draw aligned components in [-max, max]; the
    prior's support and its bilby priors (chi_i) follow."""
    s = ParameterSampler({"random_seed": 4, "signed_spins": True, "distance_prior": "uniform",
                          "distance_ranges": RANGES})
    th = np.array([[p[k] for k in PARAM_NAMES] for p in s.sample_batch(2000, "NSBH")])
    a1 = th[:, PARAM_NAMES.index("a1")]
    assert a1.min() < -0.5 and a1.max() > 0.5
    signed = TrainingPrior.for_class("NSBH", RANGES, "uniform", signed_spins=True)
    assert np.isfinite(signed.log_prob(th)).all()
    assert np.isneginf(TrainingPrior.for_class("NSBH", RANGES, "uniform").log_prob(th[a1 < 0])).all()
    assert TrainingPrior.from_meta({"train_event_types": ["NSBH"], "train_signed_spins": True}).signed_spins
    pytest.importorskip("bilby")
    bp = signed.bilby_priors(1369224018.0)
    assert "chi_1" in bp and "a_1" not in bp and bp["chi_1"].minimum == -signed.a1_max


def test_negative_spin_reaches_the_waveform():
    """bilby rebuilds a_i/tilt_i from chi_i: the sign must survive (it was
    lost before 2026-11-01, every spin came out aligned)."""
    pytest.importorskip("bilby")
    from ahsd.data.bilby_pipeline import _to_bilby_params
    p = _to_bilby_params({"a1": -0.5, "a2": 0.3})
    assert p["a_1"] == 0.5 and abs(p["tilt_1"] - np.pi) < 1e-12 and abs(p["chi_1"] + 0.5) < 1e-12
    assert p["tilt_2"] == 0.0 and p["chi_2"] == 0.3
    q = _to_bilby_params({"a1": 0.5, "tilt_1": np.pi})
    assert abs(q["chi_1"] + 0.5) < 1e-12


def test_spin_convention_travels_with_the_checkpoint():
    import torch
    from ahsd.models.lean_npe import LeanNPE
    signed = LeanNPE(spin_signed=True)
    y = signed.scaler.normalize(torch.tensor([[30.0, 20.0, 400, 1, 0.1, 1, 1, 1, 0.0, -0.6, 0.2]]))
    assert -1 < float(y[0, PARAM_NAMES.index("a1")]) < 0
    sd = {k: v for k, v in signed.state_dict().items()}
    fresh = LeanNPE()
    fresh.load_state_dict(sd)
    assert fresh.scaler.spin_signed                          # buffer wins over the default
    sd.pop("spin_signed")
    legacy = LeanNPE(spin_signed=True)
    legacy.load_state_dict(sd)                               # pre-buffer checkpoint: magnitudes
    assert not legacy.scaler.spin_signed


def _phys_batch(n=64, seed=0):
    import torch
    rng = np.random.default_rng(seed)
    m1 = rng.uniform(10, 60, n); m2 = m1 * rng.uniform(0.3, 0.95, n)
    th = np.stack([m1, m2, rng.uniform(200, 3000, n), rng.uniform(0.1, 6.2, n),
                   np.arcsin(rng.uniform(-0.95, 0.95, n)), rng.uniform(0.2, 2.9, n),
                   rng.uniform(0.1, 3.0, n), rng.uniform(0.1, 6.2, n), rng.uniform(-1.4, 1.4, n),
                   rng.uniform(-0.8, 0.8, n), rng.uniform(-0.8, 0.8, n)], 1)
    return torch.tensor(th, dtype=torch.float64)


def test_sky_frame_round_trip_and_h1_time():
    """(ra, dec, geocent_time) <-> (azimuth, cos_zenith, time_h1) is exact,
    and time_h1 is the geocent time plus bilby's H1 delay."""
    pytest.importorskip("bilby")
    import torch, bilby
    from ahsd.models.lean_npe import ParamScaler
    from ahsd.data.parameter_sampler import GPS_REF
    sc = ParamScaler(PARAM_NAMES, mass_param="mc_q", dist_range=(80, 5000), spin_signed=True, sky_frame=True)
    sc.lo, sc.hi = sc.lo.double(), sc.hi.double()
    th = _phys_batch()
    back = sc.denormalize(sc.normalize(th))
    assert torch.allclose(back, th, rtol=1e-6, atol=1e-6)
    h1 = bilby.gw.detector.get_empty_interferometer("H1")
    z = sc._to_internal(th)
    for i in range(5):
        d = h1.time_delay_from_geocenter(float(th[i, 3]), float(th[i, 4]), GPS_REF + float(th[i, 8]))
        assert abs(float(z[i, 8]) - (float(th[i, 8]) + d)) < 2e-6


@pytest.mark.parametrize("sky", [False, True])
def test_log_abs_det_jacobian_matches_numeric(sky):
    """The physical proposal density in importance sampling needs the exact
    log|dy/dtheta| of normalize(): linear maps, log axes, (m1,m2)->(Mc,q)
    and, with sky_frame, the sky rotation."""
    if sky:
        pytest.importorskip("bilby")
    import torch
    from ahsd.models.lean_npe import ParamScaler
    sc = ParamScaler(PARAM_NAMES, mass_param="mc_q", dist_range=(80, 5000), spin_signed=True, sky_frame=sky)
    sc.lo, sc.hi = sc.lo.double(), sc.hi.double()
    th = _phys_batch(6, seed=1)
    for i in range(len(th)):
        J = torch.autograd.functional.jacobian(lambda v: sc.normalize(v[None])[0], th[i])
        num = torch.linalg.slogdet(J)[1]
        assert abs(float(num - sc.log_abs_det_jacobian(th[i][None])[0])) < 1e-6


def test_precessing_coordinates():
    """Precessing models: tilt -> cos(tilt) round trip, circular azimuths,
    and the Jacobian includes |sin tilt|."""
    import torch
    from ahsd.models.lean_npe import LeanNPE, ParamScaler, PRECESSION_NAMES
    names = list(PARAM_NAMES) + PRECESSION_NAMES
    sc = ParamScaler(names, mass_param="mc_q", dist_range=(80, 5000))
    sc.lo, sc.hi = sc.lo.double(), sc.hi.double()
    th = torch.cat([_phys_batch(6, 2).abs(), torch.tensor([[0.4, 2.0, 1.0, 5.0]] * 6, dtype=torch.float64)], 1)
    th[:, 9:11] = th[:, 9:11].clamp(0.05, 0.9)
    assert "cos_tilt_1" in sc.internal_names and "phi_jl" in sc.CIRCULAR
    assert torch.allclose(sc.denormalize(sc.normalize(th)), th, atol=1e-6)
    for i in range(3):
        J = torch.autograd.functional.jacobian(lambda v: sc.normalize(v[None])[0], th[i])
        assert abs(float(torch.linalg.slogdet(J)[1] - sc.log_abs_det_jacobian(th[i][None])[0])) < 1e-6
    m = LeanNPE(psd_cond=True, encoder_type="coherent", mass_param="mc_q", dist_range=(80, 5000), precessing=True)
    assert len(m.param_names) == 15 and m.flow.features == 15


HEAVY = dict(train_bbh_mass_range=[5.0, 200.0], train_bbh_mc_max=90.0,
             train_event_types=["BBH"], train_distance_ranges=RANGES, train_distance_prior="uniform")


def _mc(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def test_default_bbh_support_is_unchanged():
    """Models trained before bbh_mass_range keep the old prior, and the sampler its draws."""
    from ahsd.data.parameter_sampler import ParameterSampler
    p = TrainingPrior.from_meta({"train_event_types": ["BBH"], "train_distance_ranges": RANGES})
    assert p.m1_range == (5.0, 100.0) and p.mc_max is None
    s = ParameterSampler({"random_seed": 3})._sample_masses("BBH")
    rng = np.random.default_rng(3)
    log_m1 = rng.uniform(np.log(5.0), np.log(100.0))
    assert abs(s["mass_1"] - np.exp(log_m1)) < 1e-9


def test_heavy_config_reaches_low_q_at_high_chirp_mass():
    """The whole point: GW190706's region exists in the data, inside the chirp-mass cap."""
    from ahsd.data.parameter_sampler import ParameterSampler
    s = ParameterSampler({"random_seed": 0, "bbh_mass_range": [5.0, 200.0], "bbh_mc_max": 90.0})
    m = np.array([[d["mass_1"], d["mass_2"]] for d in (s._sample_masses("BBH") for _ in range(20000))])
    mc, q = _mc(m[:, 0], m[:, 1]), m[:, 1] / m[:, 0]
    assert m[:, 0].max() > 150 and mc.max() <= 90.0
    high = (mc >= 70) & (mc < 86)
    assert high.sum() > 100 and (q[high] < 0.6).mean() > 0.4


def test_prior_follows_the_checkpoint_mass_support():
    p = TrainingPrior.from_meta(HEAVY)
    assert p.m1_range == (5.0, 200.0) and p.mc_max == 90.0
    th = p.sample(4000, np.random.default_rng(0))
    m1, m2 = th[:, 0], th[:, 1]
    assert _mc(m1, m2).max() <= 90.0 and m1.max() > 100
    assert np.isfinite(p.log_prob(th)).all()
    bad = th[:1].copy()
    bad[0, 0], bad[0, 1] = 190.0, 180.0     # inside the box, chirp mass 160
    assert p.log_prob(bad)[0] == -np.inf


def test_bilby_priors_apply_the_chirp_mass_cap():
    pytest.importorskip("bilby")
    d = TrainingPrior.from_meta(HEAVY).bilby_priors(0.0).sample(500)
    assert _mc(d["mass_1"], d["mass_2"]).max() <= 90.0
    assert d["mass_1"].max() > 100

