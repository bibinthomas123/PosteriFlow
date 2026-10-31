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
    th = _draw("NSBH", 400)
    j = PARAM_NAMES.index("geocent_time")
    for tc, expect in ((0.0, False), (1.0, True)):
        x = th.copy(); x[:, j] = tc + np.random.default_rng(1).normal(0, 0.01, len(x))
        x[:, 0], x[:, 1] = 8.0, 1.4                      # narrow chirp mass
        r = SimpleNamespace(samples=x, param_names=list(PARAM_NAMES), prepared=None,
                            median={n: float(np.median(x[:, i])) for i, n in enumerate(PARAM_NAMES)},
                            trigger_gps=1263097407.7,
                            diagnostics={"verdict": {"confidence": "HIGH"}, "boundary_railing_frac": 0.0})
        g = refinement_gate(r)
        assert any("trigger time" in m for m in g["strong_indicators"]) == expect
