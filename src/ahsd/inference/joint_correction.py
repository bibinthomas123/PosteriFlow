"""
Exact likelihood correction for two or more overlapping signals.

Written for two signals below; K signals work the same way, with 11 K parameters and the
ordering loudness(1) >= ... >= loudness(K). Checked against the truth for three, and
against joint dynesty for two (scripts/joint_overlap_test.py).

The network answers one rank-conditioned query per signal (rank 0 = louder by
Mc^(5/6)/d_L, rank 1 = quieter); each answer is a marginal posterior that
treats the other signal as part of the noise. When the signals overlap in
time and frequency they are not independent: the likelihood of the data
contains a cross term <h1, h2>. This module samples the JOINT posterior

    p(theta_1, theta_2 | d)  ∝  L(d | h(theta_1) + h(theta_2)) pi(theta_1) pi(theta_2)
                                 x 1[loudness(theta_1) >= loudness(theta_2)]

by tempered sequential Monte Carlo from the product proposal
q(theta_1 | d, rank 0) q(theta_2 | d, rank 1), with the exact Gaussian
likelihood of the summed waveforms. The ordering constraint is the training
convention, so "signal 1" always means the louder one (no label switching).

Unlike the single-signal correction (importance.py), phase and merger time
are NOT marginalized analytically: with two signals the cross term couples
both phases and both times and there is no closed form. All 22 parameters
are sampled; the tempering ladder is longer but the result stays exact. The
likelihood function here (JointLikelihood) is the one the dynesty reference
uses too (scripts/joint_overlap_test.py), so the comparison is exact by
construction.

    from ahsd.inference.joint_correction import joint_correct
    r0, r1 = infer(..., rank=0), infer(..., rank=1)
    c0, c1, diag = joint_correct([r0, r1], raw_strain, center_gps=..., npool=16)
"""
from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional

import numpy as np

from ahsd.models.lean_npe import PARAM_NAMES
from ahsd.inference.preprocessing import DETECTORS, SAMPLE_RATE, DURATION

_J = {n: i for i, n in enumerate(PARAM_NAMES)}
P = len(PARAM_NAMES)
# coordinates the sampler moves in, per signal block: log masses/distance,
# angles raw (wrapped after each move), merger time and spins raw
_LOG = ("mass_1", "mass_2", "luminosity_distance")
_PERIOD = {"ra": 2 * np.pi, "psi": np.pi, "phase": 2 * np.pi}


def loudness(theta: np.ndarray) -> np.ndarray:
    """Training ranking proxy Mc^(5/6) / d_L (remix_data._loudness)."""
    m1, m2, d = theta[:, _J["mass_1"]], theta[:, _J["mass_2"]], theta[:, _J["luminosity_distance"]]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    return mc ** (5.0 / 6.0) / d


def to_bilby(p: np.ndarray, center_gps: float, lambdas=(0.0, 0.0)) -> Dict[str, float]:
    """One signal's 11 physical parameters -> bilby parameter dict (signed
    aligned spins as magnitude + tilt; merger time absolute)."""
    g = {n: float(p[j]) for n, j in _J.items()}
    return dict(mass_1=g["mass_1"], mass_2=g["mass_2"], luminosity_distance=g["luminosity_distance"],
                ra=g["ra"], dec=g["dec"], theta_jn=g["theta_jn"], psi=g["psi"], phase=g["phase"],
                geocent_time=center_gps + g["geocent_time"],
                a_1=abs(g["a1"]), a_2=abs(g["a2"]),
                tilt_1=np.pi if g["a1"] < 0 else 0.0, tilt_2=np.pi if g["a2"] < 0 else 0.0,
                phi_12=0.0, phi_jl=0.0, lambda_1=float(lambdas[0]), lambda_2=float(lambdas[1]))


class JointLikelihood:
    """log L(d | h1 + h2) - log L(d | 0) for 22-parameter rows [theta_1, theta_2],
    Gaussian noise with the given PSDs, over [f_min, f_max]:
        <d, s> - <s, s> / 2,  s = sum over signals of the detector response,
        <a, b> = 4 df Re sum conj(a) b / S(f).
    Built once per worker; also used as the dynesty reference likelihood."""

    def __init__(self, raw, psds, center_gps, f_min, approximant, event_type="BBH"):
        import bilby
        from ahsd.inference.importance import segment_duration
        from ahsd.inference.priors import LIKELIHOOD_FMAX, TrainingPrior
        self.prior = TrainingPrior.for_class(event_type)
        wf = self.prior.waveform()
        self.center = float(center_gps)
        dur = segment_duration(raw)
        start = center_gps + DURATION / 2 - dur
        self.ifos = []
        for det in DETECTORS:
            if det not in raw or det not in psds:
                continue
            ifo = bilby.gw.detector.get_empty_interferometer(det)
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
                frequency_array=np.asarray(psds[det]["frequencies"], float),
                psd_array=np.asarray(psds[det]["psd"], float))
            ifo.strain_data.set_from_time_domain_strain(
                np.asarray(raw[det], np.float64), sampling_frequency=SAMPLE_RATE,
                duration=dur, start_time=start)
            ifo.minimum_frequency = f_min
            ifo.maximum_frequency = LIKELIHOOD_FMAX
            f = ifo.frequency_array
            m = ifo.frequency_mask & (f >= f_min)
            psd = ifo.power_spectral_density_array[m]
            self.ifos.append((ifo, m, ifo.frequency_domain_strain[m], 4.0 / dur / psd))
            self.idx = getattr(self, "idx", []) + [np.where(m)[0]]
            self.nf = len(f)
        self.dur = dur
        self.wfg = bilby.gw.WaveformGenerator(
            duration=dur, sampling_frequency=SAMPLE_RATE,
            frequency_domain_source_model=wf["source_model"],
            parameter_conversion=wf["parameter_conversion"],
            waveform_arguments={"waveform_approximant": approximant,
                                "reference_frequency": 50.0, "minimum_frequency": f_min})
        self.rng = np.random.default_rng()

    def responses(self, rows: np.ndarray, k: int):
        """Per-detector frequency-domain responses of signal block k of each row."""
        out = []
        l1s, l2s = self.prior.sample_tides(len(rows), self.rng)
        for i, row in enumerate(rows):
            par = to_bilby(row[k * P:(k + 1) * P], self.center, (l1s[i], l2s[i]))
            pol = self.wfg.frequency_domain_strain(par)
            out.append([ifo.get_detector_response(pol, par)[m] for ifo, m, _, _ in self.ifos])
        return out

    def log_lr(self, rows: np.ndarray, n_signals: Optional[int] = None) -> np.ndarray:
        n_signals = n_signals or rows.shape[1] // P
        out = np.empty(len(rows))
        resp = [self.responses(rows, k) for k in range(n_signals)]
        for i in range(len(rows)):
            ll = 0.0
            for j, (_, _, d, w) in enumerate(self.ifos):
                s = sum(resp[k][i][j] for k in range(n_signals))
                ll += float(np.sum((np.conj(d) * s).real * w) - 0.5 * np.sum((np.abs(s) ** 2) * w))
            out[i] = ll if np.isfinite(ll) else -np.inf
        return out


    def tp_propose(self, rows: np.ndarray, k: int, beta: float, rng, tc_half: float):
        """Independence proposal for signal k's (merger time, phase) given all
        other parameters, from the tempered likelihood itself.

        With s_k the response at the current (t, phi) and r = d - s_other,
        log L(t + tau, phi + D) = const + Re(z(tau) e^{i m D}), where
        z(tau) = <r, s_k e^{-2 pi i f tau}> for every time shift tau at once
        (one zero-padded FFT; 0.12 ms grid) and m = +-2 for a dominant-mode
        waveform (checked per row from a pi/4 phase step). The proposal over
        the whole merger-time prior is
            q(t) ~ I0(beta |z(t)|) on the grid, uniform within a cell,
            q(phi | t) = von Mises in m phi with concentration beta |z|,
        the exact conditional of L^beta on the grid. Because it spans every
        merger time it moves particles between the likelihood's cycle-spaced
        local maxima, which random-walk steps cannot (a 20 ms cycle skip held
        a whole run 100 nats below the truth). Returns new rows and log q of
        the new and the old (t, phi); the Metropolis ratio, which also
        carries the mixture anchor's factor, is formed by the caller.
        Rows whose waveform is not a single (2, 2) rotation are returned
        unchanged with log q = 0 for both (always accepted as a no-op)."""
        others = [j for j in range(rows.shape[1] // P) if j != k]
        jt, jp = k * P + _J["geocent_time"], k * P + _J["phase"]
        N = 4 * (self.nf - 1)
        dtau = self.dur / N
        tau = np.arange(N) * dtau
        tau = np.where(tau > self.dur / 2, tau - self.dur, tau)       # signed shifts
        new = rows.copy()
        lq_new, lq_old = np.zeros(len(rows)), np.zeros(len(rows))
        s_k = self.responses(rows, k)
        s_others = [self.responses(rows, j) for j in others]
        rows_q = rows.copy()
        rows_q[:, jp] = rows_q[:, jp] + np.pi / 4
        s_q = self.responses(rows_q, k)
        for i in range(len(rows)):
            # phase rotation of this waveform: s(phi + pi/4) = s(phi) e^{i m pi/4}
            j0 = int(np.argmax(np.abs(s_k[i][0])))
            ratio = s_q[i][0][j0] / s_k[i][0][j0] if abs(s_k[i][0][j0]) > 0 else 0.0
            if not (abs(abs(ratio) - 1) < 1e-3 and abs(ratio.real) < 1e-3):
                continue
            m = 2.0 * np.sign(ratio.imag)
            c = np.zeros(self.nf, complex)
            for jd, (_, _, d, w) in enumerate(self.ifos):
                c[self.idx[jd]] += np.conj(d - sum(s[i][jd] for s in s_others)) * s_k[i][jd] * w
            z_rel = np.fft.fft(c, n=N)                     # sum_f c e^{-2 pi i f tau}
            t_old, ph_old = rows[i, jt], rows[i, jp]
            t_abs = t_old + tau
            ok = np.abs(t_abs) < tc_half
            za = z_rel[ok] * np.exp(-1j * m * ph_old)     # absolute phase reference
            kap = beta * np.abs(za)
            # log I0 (stable for large arguments)
            from scipy.special import i0e
            logp = np.log(i0e(kap)) + kap
            logp -= np.logaddexp.reduce(logp)
            cells = t_abs[ok]
            pick = rng.choice(len(cells), p=np.exp(logp))
            t_new = cells[pick] + rng.uniform(-dtau / 2, dtau / 2)
            mu = -np.angle(za[pick])
            u = rng.vonmises(mu, kap[pick]) if kap[pick] > 0 else rng.uniform(-np.pi, np.pi)
            ph_new = np.mod((u + 2 * np.pi * rng.integers(0, 2)) / m, 2 * np.pi)
            def logq(t, ph):
                cidx = int(np.argmin(np.abs(cells - t)))
                if abs(cells[cidx] - t) > dtau:            # outside the grid support
                    return -np.inf
                kk = kap[cidx]
                # von Mises in u = m phi (periodic closed form); choosing one of
                # the |m| = 2 phase branches with probability 1/2 makes the
                # density of phi equal to that of u
                lv = (kk * np.cos(m * ph + np.angle(za[cidx]))
                      - np.log(2 * np.pi * i0e(kk)) - kk)
                return logp[cidx] - np.log(dtau) + lv

            new[i, jt], new[i, jp] = t_new, ph_new
            lq_new[i], lq_old[i] = logq(t_new, ph_new), logq(t_old, ph_old)
        return new, lq_new, lq_old


# ── worker pool (module-level for multiprocessing spawn) ─────────────────────
_W = {}


def _init(raw, psds, center_gps, f_min, approximant, event_type):
    import os
    _W["like"] = JointLikelihood(raw, psds, center_gps, f_min, approximant, event_type)
    _W["like"].rng = np.random.default_rng((int(center_gps * 1000) % 2 ** 32, os.getpid()))


def _eval(rows):
    try:
        return _W["like"].log_lr(rows)
    except Exception:          # waveform failures (e.g. spins past the Kerr bound) -> -inf row by row
        out = np.empty(len(rows))
        for i, r in enumerate(rows):
            try:
                out[i] = _W["like"].log_lr(r[None])[0]
            except Exception:
                out[i] = -np.inf
        return out


def _tp(args):
    rows, k, beta, seed, tc_half = args
    rng = np.random.default_rng(seed)
    try:
        return _W["like"].tp_propose(rows, k, beta, rng, tc_half)
    except Exception:          # waveform failures: leave the rows unchanged
        z = np.zeros(len(rows))
        return rows.copy(), z, z


_SLOW = [_J[n] for n in ("mass_1", "mass_2", "luminosity_distance", "ra", "dec",
                         "theta_jn", "psi", "a1", "a2")]


def _slow(block: np.ndarray) -> np.ndarray:
    """One signal's [n, 11] physical parameters -> its 9 'slow' coordinates
    (log masses / distance first; merger time and phase excluded)."""
    y = block[:, _SLOW].astype(np.float64).copy()
    y[:, :3] = np.log(y[:, :3])
    return y


def _slow_logpdf(gm, y: np.ndarray) -> np.ndarray:
    """Mixture density of sorted slow coordinates (m1 <-> m2 swap summed)."""
    ys = y.copy()
    ys[:, [0, 1]] = y[:, [1, 0]]
    return np.logaddexp(gm.score_samples(y), gm.score_samples(ys))


def _log_jac(th: np.ndarray) -> np.ndarray:
    """log |d theta / d x| of the log coordinates (both signals' masses and
    distances): every physical density gets this factor so that it is a
    density in the coordinates the walk and the mixture live in."""
    return sum(np.log(th[:, k * P + _J[n]]) for k in range(th.shape[1] // P) for n in _LOG)


def _to_x(th: np.ndarray) -> np.ndarray:
    """Physical [n, 11 K] -> sampler coordinates (log masses / distance)."""
    x = th.astype(np.float64).copy()
    for k in range(th.shape[1] // P):
        for n in _LOG:
            x[:, k * P + _J[n]] = np.log(x[:, k * P + _J[n]])
    return x


def _from_x(x: np.ndarray) -> np.ndarray:
    """Sampler coordinates -> physical: exponentiate, wrap angles, sort masses."""
    th = x.copy()
    for k in range(x.shape[1] // P):
        o = k * P
        for n in _LOG:
            th[:, o + _J[n]] = np.exp(th[:, o + _J[n]])
        for n, per in _PERIOD.items():
            th[:, o + _J[n]] = np.mod(th[:, o + _J[n]], per)
        m1, m2 = th[:, o + _J["mass_1"]].copy(), th[:, o + _J["mass_2"]].copy()
        th[:, o + _J["mass_1"]], th[:, o + _J["mass_2"]] = np.maximum(m1, m2), np.minimum(m1, m2)
    return th


def _gm_logpdf(gm, x):
    """Mixture density of SORTED samples: sum over the m1<->m2 swap in each
    signal block (2^K label orderings), so the density matches the sorted
    convention the samples are kept in."""
    outs = []
    for swaps in itertools.product((False, True), repeat=x.shape[1] // P):
        xs = x.copy()
        for k, sw in enumerate(swaps):
            if sw:
                a, b = k * P + _J["mass_1"], k * P + _J["mass_2"]
                xs[:, [a, b]] = x[:, [b, a]]
        outs.append(gm.score_samples(xs))
    return np.logaddexp.reduce(np.stack(outs), axis=0)


def residual_strain(raw: Dict[str, np.ndarray], psds, center_gps: float, theta: np.ndarray,
                    prior) -> Dict[str, np.ndarray]:
    """Raw strain minus one signal's waveform (time domain, per detector),
    for parameters theta [11]: what the quieter signal sees once the louder
    one's estimate is removed. Uses the same waveform model and detector
    projection as the likelihood (bilby, frequency domain -> irfft)."""
    import bilby
    from ahsd.inference.importance import segment_duration
    wf = prior.waveform()
    dur = segment_duration(raw)
    start = center_gps + DURATION / 2 - dur
    wfg = bilby.gw.WaveformGenerator(
        duration=dur, sampling_frequency=SAMPLE_RATE,
        frequency_domain_source_model=wf["source_model"],
        parameter_conversion=wf["parameter_conversion"],
        waveform_arguments={"waveform_approximant": wf["approximant"],
                            "reference_frequency": 50.0, "minimum_frequency": 20.0})
    par = to_bilby(theta, center_gps)
    pol = wfg.frequency_domain_strain(par)
    out = {}
    for det, x in raw.items():
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.strain_data.set_from_time_domain_strain(np.zeros(len(x)), sampling_frequency=SAMPLE_RATE,
                                                   duration=dur, start_time=start)
        fd = ifo.get_detector_response(pol, par)
        out[det] = np.asarray(x, np.float64) - np.fft.irfft(fd, n=len(x)) * SAMPLE_RATE
    return out


def best_fit_time_phase(raw: Dict[str, np.ndarray], psds, center_gps: float, theta: np.ndarray,
                        approximant: str) -> np.ndarray:
    """theta [11] with merger time and phase replaced by their best fit to the
    data (the peak of the FFT time/phase conditional at a high temperature,
    JointLikelihood.tp_propose); the second signal slot is switched off."""
    jl = JointLikelihood(raw, psds, center_gps, 20.0, approximant)
    far = theta.copy()
    far[_J["luminosity_distance"]] = 1e7
    row = np.concatenate([theta, far])[None]
    new, _, _ = jl.tp_propose(row, 0, 50.0, np.random.default_rng(0), 1.6)   # beta 50: ~the maximum
    out = theta.copy()
    out[_J["geocent_time"]], out[_J["phase"]] = new[0, _J["geocent_time"]], new[0, _J["phase"]]
    return out


def joint_correct(results: List, raw_strain: Dict[str, np.ndarray], center_gps: Optional[float] = None,
                  npool: int = 8, n: int = 3000, seed: int = 0, f_min: Optional[float] = None,
                  device: str = "cpu", max_stages: int = 60, n_moves: int = 10, n_tp: int = 2,
                  refine_proposal: bool = True, n_global: int = 2, target_ess: float = 0.2,
                  n_refine: int = 2):
    """Joint posterior of K overlapping signals from their rank-ordered network
    posteriors. Returns (result_signal1, ..., result_signalK, diagnostics); the
    diagnostics include the effective sample size, the number of tempering
    stages and log Z (the evidence of the K-signal model relative to noise, a
    by-product of SMC usable to compare numbers of signals)."""
    import multiprocessing as mp
    from sklearn.mixture import GaussianMixture
    from ahsd.inference.dynesty_bridge import GPS_REF
    from ahsd.inference.importance import segment_duration
    from ahsd.inference.pipeline import load_model
    from ahsd.inference.priors import TC_HALF_WIDTH, TrainingPrior, chirp_mass, window_fmin
    from ahsd.inference.result import PosteriorResult

    K = len(results)
    if not 2 <= K <= 4:       # the network queries up to five; four is what it trains on
        raise ValueError(f"joint_correct handles two to four signals, got {K}")
    r0 = results[0]
    if len(r0.param_names) > P:
        raise NotImplementedError("precessing posteriors are not supported by the joint correction")
    t0 = time.time()
    center = center_gps if center_gps is not None else (
        r0.trigger_gps if r0.trigger_gps is not None else GPS_REF)
    model, _ = load_model(r0.config["model_path"], device=device)
    prior = TrainingPrior.from_meta(r0.config)
    rng = np.random.default_rng(seed)

    def clean(r):
        keep = np.ones(len(r.samples), bool) if r.rail_mask is None else ~r.rail_mask
        return r.samples[keep][:, :P]
    raws = [clean(r) for r in results]
    refined = [s.copy() for s in raws]
    refine = {}
    if refine_proposal:
        # Proposal refinement (exactness is untouched: the joint stage below
        # weights against the exact two-signal likelihood, the proposal only
        # has to put particles in the right region). The network's rank-1
        # answer can span the whole sky; random-walk steps in 22 dimensions
        # cannot search it (a run settled in the wrong sky mode of signal 2,
        # 57 nats below the truth). The single-signal correction integrates
        # time and phase out analytically and fixes sky and masses well:
        # correct signal 1 on the data, subtract its median waveform, correct
        # signal 2 on the residual. Merger time and phase in those results
        # are the network's; the joint stage's time/phase move fixes them.
        from ahsd.inference.importance import corrected_with_check
        try:
            psds_ = r0.prepared.psds
            appr = prior.waveform()["approximant"]
            # The single-signal correction marginalizes merger time and phase:
            # its samples carry the NETWORK's time and phase, so subtracting a
            # median waveform used an arbitrary phase and could leave (or add)
            # most of the louder signal in the residual, which the quieter one
            # was then fitted to. Subtract the best-fit waveform instead (slow
            # parameters: corrected median; time and phase: FFT best fit),
            # and alternate: signal 2 on d - s1, signal 1 on d - s2, ...
            done, last = [], {}

            def correct_one(k):
                # subtract the others loudest first, each fitted to what is left
                d = raw_strain
                for j in done:
                    if j != k:
                        tj = best_fit_time_phase(d, psds_, center, np.median(refined[j], axis=0), appr)
                        d = residual_strain(d, psds_, center, tj, prior)
                last[k] = corrected_with_check(results[k], d, reps=1, npool=npool, device=device)
                refined[k] = clean(last[k])
                if k not in done:
                    done.append(k)

            for k in range(K):                      # first pass: loudest to quietest
                correct_one(k)
            for _ in range(max(1, n_refine) - 1):
                for k in range(K):
                    correct_one(k)
            refine = {"rounds": int(max(1, n_refine)),
                      **{f"signal{k + 1}_ess": last[k].diagnostics["importance_sampling"].get("importance_ess")
                         for k in range(K)}}
        except Exception as e:     # fall back to the network posteriors
            refine = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    m = min(*(len(s) for s in refined), n)
    # Per-signal proposal pools: the refined (single-signal corrected)
    # samples, the raw network samples, and a defensive share of prior draws.
    # A pool built from the refined samples alone inherits their failure: a
    # quiet signal 2 fitted the residual of an imperfectly subtracted signal 1
    # and its pool held only signal 1's sky (case 0: log LR 277 vs 303 at the
    # truth). The prior share guarantees every sky region is proposable.
    def make_pool(refined, raw):
        return np.concatenate([refined[rng.permutation(len(refined))[:m]],
                               raw[rng.permutation(len(raw))[:m // 2]],
                               prior.sample(max(1, m // 4), rng)])
    pools = [make_pool(s, r) for s, r in zip(refined, raws)]
    # [2 (K - 1) m, 11 K]: more draws than the mixture below has room for per dimension
    pairs = np.concatenate([p[rng.integers(len(p), size=2 * (K - 1) * m)] for p in pools], axis=1)
    # block proposals for the global move: a mixture per signal over the
    # "slow" parameters (merger time and phase are drawn from their exact
    # conditional, JointLikelihood.tp_propose)
    q_blocks = [GaussianMixture(n_components=12, covariance_type="full", reg_covar=1e-4,
                                random_state=seed).fit(_slow(pool)) for pool in pools]

    def ordered_mask(th):
        """Signals in the training's loudness order, loudest first."""
        loud = [loudness(th[:, k * P:(k + 1) * P]) for k in range(K)]
        return np.all([a_ >= b_ for a_, b_ in zip(loud[:-1], loud[1:])], axis=0)

    def log_prior(th):
        lp = sum(prior.log_prob(th[:, k * P:(k + 1) * P]) for k in range(K))
        return np.where(ordered_mask(th), lp, -np.inf)

    dur = segment_duration(raw_strain)
    if f_min is None:     # the band that fits the segment for every signal's lightest proposal templates
        both = np.concatenate(refined)
        mc_lo = float(np.quantile(chirp_mass(both[:, _J["mass_1"]], both[:, _J["mass_2"]]), 0.005))
        tc_lo = float(np.quantile(both[:, _J["geocent_time"]], 0.005))
        f_min = window_fmin(mc_lo, dur - DURATION / 2 + tc_lo)
    approximant = prior.waveform()["approximant"]
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(npool, initializer=_init,
                    initargs=(raw_strain, r0.prepared.psds, center, f_min, approximant, prior.event_type))

    def eval_ll(th):
        return np.concatenate(pool.map(_eval, np.array_split(th, max(npool * 4, 4))))

    try:
        # Initial particles are drawn from a mixture fitted to the network's
        # rank-0 x rank-1 pairs, and the tempering path runs from that mixture
        # g0 to L pi. Weighting the network's own samples against a mixture
        # anchor (as a first version did) puts the anchor/flow density
        # mismatch into the weights before the likelihood has any say: in 22
        # dimensions it collapsed every particle onto a few copies at
        # beta = 1e-4 and the run "converged" where neither signal fits
        # (log LR -9 against 303 at the truth; 2026-11-04). Drawing from g0
        # makes its density exact, so only the likelihood moves the weights.
        ordered = pairs[ordered_mask(pairs)]
        gm0 = GaussianMixture(n_components=16, covariance_type="full", reg_covar=1e-5,
                              random_state=seed).fit(_to_x(ordered))
        theta = _from_x(np.asarray(gm0.sample(m)[0]))
        x = _to_x(theta)
        lg0 = _gm_logpdf(gm0, x)
        lpi = log_prior(theta)
        lt1 = np.full(m, -np.inf)
        ok = np.isfinite(lpi)
        lt1[ok] = eval_ll(theta[ok]) + lpi[ok] + _log_jac(theta[ok])
        lg = lg0                         # the particles' density is the mixture's, in x
        beta, betas, log_z, accs = 0.0, [], 0.0, []

        def ess(lw):
            f = np.isfinite(lw)
            if f.sum() < 10:
                return 0.0
            v = np.exp(lw[f] - lw[f].max())
            return float(v.sum() ** 2 / (v ** 2).sum() / len(lw))

        mv = {"scale": 0.3}
        tp_acc, glob_acc = [], []

        def accept(state, thp, beta, lq_diff=0.0):
            """Metropolis step to proposals thp for the tempered target
            t_beta = g0^(1-beta) (L pi)^beta (densities in x); lq_diff =
            log q(new) - log q(old) for an asymmetric proposal."""
            theta, x, lt1, lg0 = state
            xp = _to_x(thp)
            lpp = log_prior(thp)
            t1p = np.full(m, -np.inf)
            okp = np.isfinite(lpp)
            if okp.any():
                t1p[okp] = eval_ll(thp[okp]) + lpp[okp] + _log_jac(thp[okp])
            fin = np.isfinite(xp).all(axis=1)      # overflowed steps: rejected
            g0p = np.full(m, -np.inf)
            if fin.any():
                g0p[fin] = _gm_logpdf(gm0, xp[fin])
            la = (1 - beta) * (g0p - lg0) + beta * (t1p - lt1) - lq_diff
            acc = np.log(rng.uniform(size=m)) < np.where(np.isfinite(la), la, -np.inf)
            return (np.where(acc[:, None], thp, theta), np.where(acc[:, None], xp, x),
                    np.where(acc, t1p, lt1), np.where(acc, g0p, lg0)), float(acc.mean())

        def moves(state, beta):
            """Random-walk block moves, then the time/phase proposal per signal
            (JointLikelihood.tp_propose); all leave t_beta invariant."""
            for _ in range(n_moves):
                for k in range(K):
                    sl = slice(k * P, (k + 1) * P)
                    x = state[1]
                    cov = np.cov(x[:, sl].T)
                    # masses are sorted after the step (a fold across
                    # m1 = m2): the step must be swap-symmetric for the
                    # folded walk to keep detailed balance (importance.py)
                    i1, i2 = _J["mass_1"], _J["mass_2"]
                    pc = cov.copy()
                    pc[[i1, i2], :] = pc[[i2, i1], :]
                    pc[:, [i1, i2]] = pc[:, [i2, i1]]
                    cov = 0.5 * (cov + pc) + 1e-9 * np.eye(P)
                    step = rng.standard_normal((m, P)) @ np.linalg.cholesky(cov).T
                    xp = x.copy()
                    xp[:, sl] = x[:, sl] + mv["scale"] * step
                    state, a_ = accept(state, _from_x(xp), beta)
                    accs.append(a_)
                    mv["scale"] *= float(np.exp(a_ - 0.25))
            for _ in range(n_tp):
                for k in range(K):
                    chunks = np.array_split(state[0], max(npool * 4, 4))
                    res = pool.map(_tp, [(c, k, beta, int(rng.integers(2 ** 31)), TC_HALF_WIDTH)
                                         for c in chunks])
                    thp = np.concatenate([r_[0] for r_ in res])
                    lqd = np.concatenate([r_[1] - r_[2] for r_ in res])
                    state, a_ = accept(state, thp, beta, lqd)
                    tp_acc.append(a_)
            # Global block moves: a whole signal (masses, distance, sky,
            # inclination, polarization, spins) is redrawn from its block
            # mixture, then its merger time and phase from their exact
            # tempered conditional. Exact MH with the reverse-move density;
            # this lets a signal jump between sky modes and out of the other
            # signal's residual, which local steps cannot.
            for _ in range(n_global):
                for k in range(K):
                    th = state[0]
                    o = k * P
                    cols = [o + j for j in _SLOW]
                    y_new = np.asarray(q_blocks[k].sample(m)[0])[rng.permutation(m)]
                    vals = y_new.copy()
                    vals[:, :3] = np.exp(vals[:, :3])
                    thp = th.copy()
                    thp[:, cols] = vals
                    thp = _from_x(_to_x(thp))                  # sorted masses, wrapped angles
                    lq_new_slow = _slow_logpdf(q_blocks[k], _slow(thp[:, o:o + P]))
                    lq_old_slow = _slow_logpdf(q_blocks[k], _slow(th[:, o:o + P]))
                    split = lambda z: np.array_split(z, max(npool * 4, 4))
                    fw = pool.map(_tp, [(c, k, beta, int(rng.integers(2 ** 31)), TC_HALF_WIDTH)
                                        for c in split(thp)])
                    bw = pool.map(_tp, [(c, k, beta, int(rng.integers(2 ** 31)), TC_HALF_WIDTH)
                                        for c in split(th)])
                    thp2 = np.concatenate([r_[0] for r_ in fw])
                    lqn = np.concatenate([r_[1] for r_ in fw])
                    lqo = np.concatenate([r_[2] for r_ in bw])
                    lqd = (lq_new_slow + lqn) - (lq_old_slow + lqo)
                    # a row whose time/phase draw was skipped in one direction
                    # only (not a single-mode rotation) cannot be balanced: reject
                    lqd = np.where((lqn == 0) ^ (lqo == 0), np.inf, lqd)
                    lqd = np.where(np.isfinite(lqd), lqd, np.inf)
                    state, a_ = accept(state, thp2, beta, lqd)
                    glob_acc.append(a_)
            return state

        for stage in range(max_stages):
            def lw_at(bn):      # target_bn vs the current particles' density
                return (1 - bn) * lg0 + bn * lt1 - lg
            if ess(lw_at(1.0)) >= target_ess:
                nb = 1.0
            else:
                lo_, hi_ = beta, 1.0
                for _ in range(40):
                    mid = 0.5 * (lo_ + hi_)
                    lo_, hi_ = (mid, hi_) if ess(lw_at(mid)) >= target_ess else (lo_, mid)
                nb = max(lo_, beta + 1e-4)
            lw = lw_at(nb)
            f = np.isfinite(lw)
            mx = lw[f].max()
            # evidence increment of the step from target_beta to target_nb
            w_inc = np.where(f, np.exp(lw - mx), 0.0)
            log_z += np.log(w_inc.mean()) + mx
            beta = nb
            betas.append(round(beta, 4))
            w = w_inc / w_inc.sum()
            if beta >= 1.0:
                break
            # systematic resampling
            u = (rng.uniform() + np.arange(m)) / m
            idx = np.searchsorted(np.cumsum(w), u).clip(0, m - 1)
            state = (theta[idx], x[idx], lt1[idx], lg0[idx])
            state = moves(state, beta)
            theta, x, lt1, lg0 = state
            # after resampling + invariant moves the particles follow t_beta
            lg = (1 - beta) * lg0 + beta * lt1
        # final self-normalized weights at beta = 1, then rejuvenation moves
        # at beta = 1 (invariant for the posterior) so the resampled copies
        # spread out again
        ess_final = float(1.0 / np.sum(w ** 2))
        u = (rng.uniform() + np.arange(m)) / m
        idx = np.searchsorted(np.cumsum(w), u).clip(0, m - 1)
        state = moves((theta[idx], x[idx], lt1[idx], lg0[idx]), 1.0)
        out = state[0]
        log_lr_out = state[2] - log_prior(out) - _log_jac(out)
    finally:
        pool.close()
        pool.join()

    diag = {"joint_correction": {
        "n_signals": K, "ess": round(ess_final, 1), "n": int(m), "n_stages": len(betas), "betas": betas,
        "converged": bool(beta >= 1.0), "log_evidence_vs_anchor": float(log_z),
        "mh_acceptance_mean": round(float(np.mean(accs)), 3) if accs else None,
        "tp_acceptance_mean": round(float(np.mean(tp_acc)), 3) if tp_acc else None,
        "global_acceptance_mean": round(float(np.mean(glob_acc)), 3) if glob_acc else None,
        "log_lr_median": round(float(np.median(log_lr_out)), 1),
        "unique_frac": round(len(np.unique(out[:, 0])) / len(out), 3),
        "f_min": round(float(f_min), 1), "wall_s": round(time.time() - t0, 1),
        "proposal_refinement": refine,
        "note": f"joint posterior of {K} signals ({K * P} parameters, summed-waveform "
                "likelihood); phase and merger time are sampled, not marginalized"}}
    res = []
    for k, r in enumerate(results):
        res.append(PosteriorResult(
            samples=out[:, k * P:(k + 1) * P], log_prob=np.zeros(len(out)),
            param_names=list(PARAM_NAMES), trigger_gps=r.trigger_gps, truth=r.truth,
            prepared=r.prepared, diagnostics=diag, config={**r.config, "joint_corrected": True},
            rail_mask=None))
    return (*res, diag)
