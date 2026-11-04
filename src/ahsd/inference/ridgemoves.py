"""Mass-ratio / spin ridge move for the likelihood correction.

At fixed chirp mass the inspiral phase still trades the mass ratio against
the aligned spin: a lighter, more anti-aligned primary and a heavier, less
spinning one produce nearly the same waveform (GW200115: dynesty's posterior
is one curved ridge from (m1, q, chi1) = (3.8, 0.61, -0.81) to (5.7, 0.30,
-0.31)). A random walk crosses such a ridge, it does not follow it, and the
correction kept one stretch of it (seed-dependent).

The move steps the symmetric mass ratio eta at fixed chirp mass and sets
chi1 so that the 1PN + 1.5PN phase term at a reference frequency is
unchanged:

    B = 3 / (128 eta v^5) [ (3715/756 + 55 eta / 9) v^2 + (4 beta - 16 pi) v^3 ],
    v = (pi M f_ref)^(1/3),  M = Mc eta^(-3/5),
    beta = sum_i chi_i (113 (m_i / M)^2 + 75 eta) / 12,

i.e. a symmetric random walk in eta in the coordinates u = (Mc, eta, B) with
chi2 and everything else fixed. It is exact Metropolis-Hastings once the
Jacobian |d u / d (m1, m2, chi1)| enters the acceptance (`log_jac_u`).
f_ref and the step size are drawn independently of the point.
"""
from __future__ import annotations

import numpy as np

from ahsd.models.lean_npe import PARAM_NAMES

_J = {n: i for i, n in enumerate(PARAM_NAMES)}
MSUN_S = 4.925490947e-6          # G M_sun / c^3 (s)


def _beta(m1, m2, c1, c2):
    M = m1 + m2
    eta = m1 * m2 / M ** 2
    return (c1 * (113 * (m1 / M) ** 2 + 75 * eta) + c2 * (113 * (m2 / M) ** 2 + 75 * eta)) / 12.0


def _v(M, f_ref):
    return (np.pi * M * MSUN_S * f_ref) ** (1.0 / 3.0)


def phase_B(m1, m2, c1, c2, f_ref):
    M = m1 + m2
    eta = m1 * m2 / M ** 2
    v = _v(M, f_ref)
    return 3.0 / (128.0 * eta * v ** 5) * ((3715.0 / 756.0 + 55.0 * eta / 9.0) * v ** 2
                                          + (4.0 * _beta(m1, m2, c1, c2) - 16.0 * np.pi) * v ** 3)


def log_jac_u(m1, m2, c1, f_ref):
    """log |d (Mc, eta, B) / d (m1, m2, chi1)| (block triangular: Mc, eta do
    not depend on chi1)."""
    M = m1 + m2
    eta = m1 * m2 / M ** 2
    mc = M * eta ** 0.6
    dmc1 = mc * (0.6 / m1 - 0.2 / M)
    dmc2 = mc * (0.6 / m2 - 0.2 / M)
    de1 = m2 * (m2 - m1) / M ** 3
    de2 = m1 * (m1 - m2) / M ** 3
    det = dmc1 * de2 - dmc2 * de1
    v = _v(M, f_ref)
    dB = 3.0 / (128.0 * eta * v ** 5) * 4.0 * v ** 3 * (113 * (m1 / M) ** 2 + 75 * eta) / 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.abs(det)) + np.log(np.abs(dB))


class RidgeMove:
    F_REFS = (30.0, 50.0, 80.0, 120.0)
    SIGMAS = (0.005, 0.015, 0.04)          # eta steps

    def propose(self, th, rng):
        """Returns (th', log_a correction) where the correction is
        -[log|du/dtheta|(th') - log|du/dtheta|(th)] (add the change of the
        sampling-coordinate log-Jacobian and of the target yourself)."""
        n = len(th)
        m1, m2 = th[:, _J["mass_1"]], th[:, _J["mass_2"]]
        c1, c2 = th[:, _J["a1"]], th[:, _J["a2"]]
        f_ref = np.asarray(self.F_REFS)[rng.integers(0, len(self.F_REFS), n)]
        sig = np.asarray(self.SIGMAS)[rng.integers(0, len(self.SIGMAS), n)]
        M = m1 + m2
        eta = m1 * m2 / M ** 2
        mc = M * eta ** 0.6
        B = phase_B(m1, m2, c1, c2, f_ref)
        eta2 = eta + sig * rng.standard_normal(n)
        ok = (eta2 > 0) & (eta2 <= 0.25)
        eta2 = np.clip(eta2, 1e-6, 0.25)
        M2 = mc * eta2 ** -0.6
        root = np.sqrt(np.maximum(1 - 4 * eta2, 0.0))
        n1, n2 = M2 * (1 + root) / 2, M2 * (1 - root) / 2
        v = _v(M2, f_ref)
        beta2 = (B * 128.0 * eta2 * v ** 5 / 3.0 - (3715.0 / 756.0 + 55.0 * eta2 / 9.0) * v ** 2) \
            / (4.0 * v ** 3) + 4.0 * np.pi
        k2 = 113 * (n2 / M2) ** 2 + 75 * eta2
        k1 = 113 * (n1 / M2) ** 2 + 75 * eta2
        d1 = (12.0 * beta2 - c2 * k2) / k1
        out = th.copy()
        out[:, _J["mass_1"]], out[:, _J["mass_2"]], out[:, _J["a1"]] = n1, n2, d1
        corr = -(log_jac_u(n1, n2, d1, f_ref) - log_jac_u(m1, m2, c1, f_ref))
        return out, np.where(ok & np.isfinite(corr), corr, -np.inf)
