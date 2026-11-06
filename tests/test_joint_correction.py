"""Joint correction helpers for K signals (2026-11-06). The sampler itself is checked against
the truth and dynesty by scripts/joint_overlap_test.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ahsd.inference.joint_correction import P, _J, _from_x, _gm_logpdf, _log_jac, _to_x, loudness  # noqa: E402


def _row(K, seed=0):
    rng = np.random.default_rng(seed)
    th = np.zeros((20, K * P))
    for k in range(K):
        o = k * P
        th[:, o + _J["mass_1"]] = rng.uniform(30, 60, 20)
        th[:, o + _J["mass_2"]] = rng.uniform(10, 29, 20)
        th[:, o + _J["luminosity_distance"]] = rng.uniform(500, 3000, 20)
        th[:, o + _J["ra"]] = rng.uniform(0, 6, 20)
        th[:, o + _J["psi"]] = rng.uniform(0, 3, 20)
        th[:, o + _J["phase"]] = rng.uniform(0, 6, 20)
    return th


@pytest.mark.parametrize("K", [2, 3, 4])
def test_coordinates_round_trip_every_block(K):
    th = _row(K)
    np.testing.assert_allclose(_from_x(_to_x(th)), th, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("K", [2, 3])
def test_jacobian_covers_every_block(K):
    th = _row(K)
    want = sum(np.log(th[:, k * P + _J[n]]) for k in range(K) for n in ("mass_1", "mass_2", "luminosity_distance"))
    np.testing.assert_allclose(_log_jac(th), want)


@pytest.mark.parametrize("K", [2, 3])
def test_from_x_sorts_masses_in_every_block(K):
    x = _to_x(_row(K))
    for k in range(K):
        a, b = k * P + _J["mass_1"], k * P + _J["mass_2"]
        x[:, [a, b]] = x[:, [b, a]]
    th = _from_x(x)
    for k in range(K):
        assert (th[:, k * P + _J["mass_1"]] >= th[:, k * P + _J["mass_2"]]).all()


class _Fake:
    """Records the arrays it is scored on; density depends on the first coordinate."""
    def __init__(self):
        self.calls = 0

    def score_samples(self, x):
        self.calls += 1
        return -0.5 * (x[:, _J["mass_1"]] ** 2)


@pytest.mark.parametrize("K,orderings", [(2, 4), (3, 8), (4, 16)])
def test_sorted_density_sums_every_label_ordering(K, orderings):
    gm = _Fake()
    x = _to_x(_row(K))
    out = _gm_logpdf(gm, x)
    assert gm.calls == orderings and out.shape == (len(x),)


def test_two_block_density_matches_the_old_four_term_sum():
    gm = _Fake()
    x = _to_x(_row(2))
    i1, i2 = _J["mass_1"], _J["mass_2"]
    terms = []
    for s1 in (False, True):
        for s2 in (False, True):
            xs = x.copy()
            if s1:
                xs[:, [i1, i2]] = x[:, [i2, i1]]
            if s2:
                xs[:, [P + i1, P + i2]] = x[:, [P + i2, P + i1]]
            terms.append(_Fake().score_samples(xs))
    np.testing.assert_allclose(_gm_logpdf(gm, x), np.logaddexp.reduce(np.stack(terms), axis=0))


def test_third_signal_far_away_changes_nothing():
    """The summed-waveform likelihood of K=3 with a negligible third equals K=2."""
    pytest.importorskip("lal")
    from ahsd.data.bilby_pipeline import BilbyNoiseGenerator, get_default_psd
    from ahsd.inference.dynesty_bridge import GPS_REF
    from ahsd.inference.joint_correction import JointLikelihood
    DETS = ("H1", "L1")
    gen = BilbyNoiseGenerator(4096, 4.0)
    psds = {d: get_default_psd(d) for d in DETS}
    raw = {d: gen.generate(d, psds[d], seed=i) for i, d in enumerate(DETS)}
    jl = JointLikelihood(raw, psds, GPS_REF, 20.0, "IMRPhenomXP")
    th = np.zeros((1, 3 * P))
    for k, (m1, m2, d) in enumerate(((36.0, 30.0, 2000.0), (28.0, 22.0, 1700.0), (20.0, 15.0, 1e8))):
        o = k * P
        th[0, o + _J["mass_1"]], th[0, o + _J["mass_2"]], th[0, o + _J["luminosity_distance"]] = m1, m2, d
        th[0, o + _J["ra"]], th[0, o + _J["dec"]], th[0, o + _J["theta_jn"]] = 1.0 + k, 0.3, 0.6
    a = jl.log_lr(th[:, :2 * P])[0]
    assert jl.log_lr(th[:, :2 * P], n_signals=2)[0] == pytest.approx(a)
    assert jl.log_lr(th)[0] == pytest.approx(a, abs=1e-3)
    assert jl.log_lr(th[:, :P])[0] != pytest.approx(a, abs=1e-3)


def test_loudness_orders_a_block():
    th = _row(1)
    assert loudness(th).shape == (20,)
