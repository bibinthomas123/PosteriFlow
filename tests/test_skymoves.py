"""Sky-mode jumps of the likelihood correction (ahsd.inference.skymoves)."""
import numpy as np
import pytest

bilby = pytest.importorskip("bilby")

from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

J = {n: i for i, n in enumerate(PARAM_NAMES)}
GPS = 1369224018.0


def _jumps(dets=("H1", "L1", "V1")):
    from ahsd.data.bilby_pipeline import get_default_psd
    from ahsd.inference.skymoves import SkyJumps
    return SkyJumps(dets, GPS, {d: get_default_psd(d) for d in dets})


def _sky(n, rng):
    return rng.uniform(0, 2 * np.pi, n), np.arcsin(rng.uniform(-1, 1, n))


def test_antenna_matches_bilby():
    s = _jumps()
    rng = np.random.default_rng(0)
    ra, dec = _sky(20, rng)
    psi = rng.uniform(0, np.pi, 20)
    fp, fx = s.antenna(ra, dec, psi)
    for k, d in enumerate(s.dets):
        ifo = bilby.gw.detector.get_empty_interferometer(d)
        ref_p = [ifo.antenna_response(a, b, GPS, p, "plus") for a, b, p in zip(ra, dec, psi)]
        ref_x = [ifo.antenna_response(a, b, GPS, p, "cross") for a, b, p in zip(ra, dec, psi)]
        assert np.allclose(fp[:, k], ref_p, atol=1e-10)
        assert np.allclose(fx[:, k], ref_x, atol=1e-10)


def test_reflection_keeps_every_delay_and_rotation_the_baseline_delay():
    s = _jumps()
    rng = np.random.default_rng(1)
    ra, dec = _sky(500, rng)
    t0 = s.delays(ra, dec)
    for k, d in enumerate(s.dets):        # delays agree with bilby's
        ifo = bilby.gw.detector.get_empty_interferometer(d)
        assert np.isclose(t0[0, k], ifo.time_delay_from_geocenter(ra[0], dec[0], GPS), atol=1e-9)
    ra2, dec2 = s.reflect(ra, dec)
    t1 = s.delays(ra2, dec2)
    assert np.allclose(t1 - t1[:, :1], t0 - t0[:, :1], atol=1e-9)
    ra3, dec3 = s.reflect(ra2, dec2)                        # involution
    assert np.allclose(s.omega(ra3, dec3), s.omega(ra, dec), atol=1e-9)
    ra4, dec4 = s.rotate(ra, dec, rng.uniform(0, 2 * np.pi, 500))
    t4 = s.delays(ra4, dec4)
    a, b = np.argsort(-s.w)[:2]
    assert np.allclose(t4[:, a] - t4[:, b], t0[:, a] - t0[:, b], atol=1e-9)


def test_rotation_about_each_baseline_keeps_that_delay():
    s = _jumps()
    rng = np.random.default_rng(4)
    ra, dec = _sky(300, rng)
    t0 = s.delays(ra, dec)
    pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
    assert len(s.axes) == 3
    for ax in s.axes:
        ra2, dec2 = s.rotate(ra, dec, rng.uniform(0, 2 * np.pi, 300), ax)
        t1 = s.delays(ra2, dec2)
        kept = [np.allclose(t1[:, i] - t1[:, j], t0[:, i] - t0[:, j], atol=1e-9) for i, j in pairs]
        assert sum(kept) == 1          # exactly the pair whose baseline it is


def test_delay_shift_moves_one_delay_by_at_most_dmax():
    s = _jumps()
    rng = np.random.default_rng(5)
    ra, dec = _sky(2000, rng)
    t0 = s.delays(ra, dec)
    ra2, dec2, lr, ok = s.shift_target(ra, dec, rng)
    assert ok.mean() > 0.3
    t1 = s.delays(ra2[ok], dec2[ok])
    t0 = t0[ok]
    pairs = [(0, 1), (0, 2), (1, 2)]
    for row0, row1 in zip(t0[:50], t1[:50]):
        ch = [abs((row1[i] - row1[j]) - (row0[i] - row0[j])) for i, j in pairs]
        assert min(ch) < 1e-9                      # the rotation axis' pair is kept
        assert max(ch) <= 2 * s.DMAX + 1e-9


@pytest.mark.parametrize("dets,kind", [(("H1", "L1", "V1"), None), (("H1", "L1", "V1"), "shift"),
                                       (("H1", "L1"), None)])
def test_jump_leaves_the_target_invariant(dets, kind):
    """MH with the jump (log_q both ways + sky Jacobian) keeps an isotropic
    sky x uniform psi x uniform distance target exactly."""
    from scipy.stats import kstest
    s = _jumps(dets)
    rng = np.random.default_rng(2)
    n = 4000
    th = np.zeros((n, len(PARAM_NAMES)))
    th[:, J["ra"]], th[:, J["dec"]] = _sky(n, rng)
    th[:, J["psi"]] = rng.uniform(0, np.pi, n)
    th[:, J["theta_jn"]] = np.arccos(rng.uniform(-1, 1, n))
    th[:, J["luminosity_distance"]] = rng.uniform(100, 1000, n)
    # isotropic sky x uniform psi, distance, in the correction's sky
    # coordinates (ra, atanh sin dec): density cos^2 dec (log_j is the
    # Jacobian in these coordinates, plus a delay-shift angle's q ratio)
    def logp(t):
        dd = t[:, J["luminosity_distance"]]
        return np.where((dd > 100) & (dd < 1000), 2 * np.log(np.cos(t[:, J["dec"]])), -np.inf)
    acc_all = []
    for _ in range(25):
        tp, lq_f, log_j = s.propose(th, rng, kind)
        lq_r = s.log_q(tp, th)
        la = logp(tp) - logp(th) + lq_r - lq_f + log_j
        acc = np.log(rng.uniform(size=n)) < np.where(np.isfinite(la), la, -np.inf)
        th = np.where(acc[:, None], tp, th)
        acc_all.append(acc.mean())
    assert np.mean(acc_all) > 0.05
    assert kstest(np.sin(th[:, J["dec"]]), "uniform", args=(-1, 2)).statistic < 0.03
    assert kstest(th[:, J["psi"]], "uniform", args=(0, np.pi)).statistic < 0.03
    assert kstest(th[:, J["luminosity_distance"]], "uniform", args=(100, 900)).statistic < 0.03
    assert kstest(th[:, J["ra"]], "uniform", args=(0, 2 * np.pi)).statistic < 0.03
