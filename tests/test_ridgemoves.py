"""Mass-ratio / spin ridge move (ahsd.inference.ridgemoves)."""
import numpy as np

from ahsd.models.lean_npe import PARAM_NAMES

J = {n: i for i, n in enumerate(PARAM_NAMES)}


def _prior():
    from test_importance_helpers import _bbh_prior
    return _bbh_prior()


def test_move_keeps_chirp_mass_and_phase_term():
    from ahsd.inference.ridgemoves import RidgeMove, phase_B
    rng = np.random.default_rng(0)
    th = _prior().sample(2000, rng)
    mv = RidgeMove()
    # fix the random choices to check one f_ref
    mv.F_REFS = (50.0,)
    out, corr = mv.propose(th, rng)
    ok = np.isfinite(corr)
    assert ok.mean() > 0.8
    mc = lambda t: (t[:, 0] * t[:, 1]) ** 0.6 / (t[:, 0] + t[:, 1]) ** 0.2  # noqa: E731
    assert np.allclose(mc(out[ok]), mc(th[ok]), rtol=1e-10)
    b0 = phase_B(th[ok, 0], th[ok, 1], th[ok, J["a1"]], th[ok, J["a2"]], 50.0)
    b1 = phase_B(out[ok, 0], out[ok, 1], out[ok, J["a1"]], out[ok, J["a2"]], 50.0)
    assert np.allclose(b0, b1, rtol=1e-9)
    assert (out[ok, 0] >= out[ok, 1]).all()


def test_jacobian_matches_finite_differences():
    from ahsd.inference.ridgemoves import log_jac_u, phase_B
    rng = np.random.default_rng(1)
    th = _prior().sample(50, rng)
    f = 80.0
    for t in th[:10]:
        m1, m2, c1, c2 = t[0], t[1], t[J["a1"]], t[J["a2"]]
        def u(x):
            M = x[0] + x[1]
            eta = x[0] * x[1] / M ** 2
            return np.array([M * eta ** 0.6, eta, phase_B(x[0], x[1], x[2], c2, f)])
        x0 = np.array([m1, m2, c1])
        h = 1e-6 * np.maximum(np.abs(x0), 1e-3)
        jac = np.column_stack([(u(x0 + e) - u(x0 - e)) / (2 * e[i]) for i, e in enumerate(np.diag(h))])
        assert np.isclose(np.log(abs(np.linalg.det(jac))), log_jac_u(m1, m2, c1, f), atol=1e-4)


def test_move_leaves_the_prior_invariant():
    """MH with the move, target = training prior: the distribution of
    (m1, m2, chi1) stays the prior's."""
    from scipy.stats import ks_2samp
    from ahsd.inference.ridgemoves import RidgeMove
    p = _prior()
    rng = np.random.default_rng(2)
    th = p.sample(6000, rng)
    ref = p.sample(20000, rng)
    mv = RidgeMove()
    acc_all = []
    for _ in range(40):
        tp, corr = mv.propose(th, rng)
        la = p.log_prob(tp) - p.log_prob(th) + corr
        acc = np.log(rng.uniform(size=len(th))) < np.where(np.isfinite(la), la, -np.inf)
        th = np.where(acc[:, None], tp, th)
        acc_all.append(acc.mean())
    assert np.mean(acc_all) > 0.05
    for k in ("mass_1", "mass_2", "a1"):
        assert ks_2samp(th[:, J[k]], ref[:, J[k]]).statistic < 0.03, k


def test_distance_inclination_move_keeps_amplitude():
    from ahsd.inference.ridgemoves import DistanceInclinationMove, _amp
    rng = np.random.default_rng(3)
    th = _prior().sample(3000, rng)
    out, corr = DistanceInclinationMove().propose(th, rng)
    ok = np.isfinite(corr)
    assert ok.mean() > 0.8
    k0 = _amp(np.cos(th[ok, J["theta_jn"]])) / th[ok, J["luminosity_distance"]]
    k1 = _amp(np.cos(out[ok, J["theta_jn"]])) / out[ok, J["luminosity_distance"]]
    assert np.allclose(k0, k1, rtol=1e-10)
    other = [j for n, j in J.items() if n not in ("theta_jn", "luminosity_distance")]
    assert np.array_equal(out[:, other], th[:, other])


def test_distance_inclination_jacobian_matches_finite_differences():
    from ahsd.inference.ridgemoves import _amp, log_jac_k
    rng = np.random.default_rng(4)
    th = _prior().sample(20, rng)
    for t in th[:10]:
        x0 = np.array([t[J["theta_jn"]], t[J["luminosity_distance"]]])

        def u(x):
            c = np.cos(x[0])
            return np.array([c, _amp(c) / x[1]])
        h = 1e-6 * np.maximum(np.abs(x0), 1e-3)
        jac = np.column_stack([(u(x0 + e) - u(x0 - e)) / (2 * e[i]) for i, e in enumerate(np.diag(h))])
        assert np.isclose(np.log(abs(np.linalg.det(jac))), log_jac_k(x0[0], x0[1]), atol=1e-4)


def test_distance_inclination_move_leaves_the_prior_invariant():
    """MH with the move, target = training prior: theta_jn and d_L stay
    distributed as the prior's."""
    from scipy.stats import ks_2samp
    from ahsd.inference.ridgemoves import DistanceInclinationMove
    p = _prior()
    rng = np.random.default_rng(5)
    th = p.sample(6000, rng)
    ref = p.sample(20000, rng)
    mv = DistanceInclinationMove()
    acc_all = []
    for _ in range(40):
        tp, corr = mv.propose(th, rng)
        la = p.log_prob(tp) - p.log_prob(th) + corr
        acc = np.log(rng.uniform(size=len(th))) < np.where(np.isfinite(la), la, -np.inf)
        th = np.where(acc[:, None], tp, th)
        acc_all.append(acc.mean())
    assert np.mean(acc_all) > 0.05
    for k in ("theta_jn", "luminosity_distance"):
        assert ks_2samp(th[:, J[k]], ref[:, J[k]]).statistic < 0.03, k
