"""
Importance-sampling correction for LeanNPE posteriors (DINGO-IS style,
Dax et al. 2023): the amortized posterior is a PROPOSAL; reweighting by
    w = L(theta) * pi(theta) / q(theta | d) 
against the true bilby likelihood makes the result asymptotically exact —
in particular it removes the mass-ratio prior-reversion bias wherever the
proposal covers the truth (our q posteriors are wide, so it does).

    corrected = importance_correct(result, raw_strain, psds,
                                   center_gps=..., npool=4)

Cost: one waveform + inner product per sample (~50-100 ms each,
parallelized) — seconds-to-minutes per event, still 10-100x faster than
nested sampling, with the importance ESS as a built-in honesty meter.

Density bookkeeping (exactness requires both):
  - mass ordering: samples are sorted to m1 >= m2 after the flow, so the
    proposal density is the SYMMETRIZED flow density q(m1,m2) + q(m2,m1);
  - boundary-railed samples are excluded (the clamp invalidates their
    recorded density; they sit at prior edges by construction).
Circular wrapping (ra/phase/psi) folds at most a few-percent density error
on wrapped samples; second-order, documented, ignored.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch

from ahsd.models.lean_npe import PARAM_NAMES
from ahsd.inference.preprocessing import DETECTORS, SAMPLE_RATE, DURATION

# training-prior constants (parameter_sampler)
M_LO, M_HI = 5.0, 100.0
D_LO, D_HI = 50.0, 2100.0
TC_HW = 1.6           # scaler range; sampler draws +-1.5
A_HI = 0.99

_J = {n: i for i, n in enumerate(PARAM_NAMES)}


def log_prior_training(theta: np.ndarray) -> np.ndarray:
    """Closed-form log of the BBH training prior (unnormalized constants
    kept: they cancel in self-normalized IS). theta: [N, 11] physical."""
    m1, m2 = theta[:, _J["mass_1"]], theta[:, _J["mass_2"]]
    d = theta[:, _J["luminosity_distance"]]
    dec = theta[:, _J["dec"]]
    th = theta[:, _J["theta_jn"]]
    tc = theta[:, _J["geocent_time"]]
    a1, a2 = theta[:, _J["a1"]], theta[:, _J["a2"]]

    lp = np.zeros(len(theta))
    ok = ((m2 > M_LO) & (m1 < M_HI) & (m2 <= m1) &
          (d > D_LO) & (d < D_HI) & (np.abs(tc) < TC_HW) &
          (a1 >= 0) & (a1 <= A_HI) & (a2 >= 0) & (a2 <= A_HI) &
          (th > 0) & (th < np.pi) & (np.abs(dec) < np.pi / 2))
    # conditional log-uniform masses: p ∝ 1/(m1 m2 log(m1/lo))
    lp -= np.log(m1) + np.log(m2) + np.log(np.maximum(np.log(m1 / M_LO), 1e-9))
    lp += 2.0 * np.log(d)                      # comoving-volume distance
    lp += np.log(np.maximum(np.cos(dec), 1e-12))   # isotropic dec
    lp += np.log(np.maximum(np.sin(th), 1e-12))    # isotropic inclination
    lp[~ok] = -np.inf
    return lp


def symmetrized_log_q(theta_np, model, prepared, rank: int = 0,
                      device: str = "cpu", batch: int = 2048) -> np.ndarray:
    """Flow density of SORTED samples: log[q(m1,m2,..) + q(m2,m1,..)],
    physical units. theta_np: [N, 11]."""
    from ahsd.inference.pipeline import _log_prob_physical

    theta = torch.from_numpy(theta_np.astype(np.float32))
    strain = torch.from_numpy(prepared.strain).unsqueeze(0).to(device).float()
    with torch.no_grad():
        ctx = model.encoder(strain)
    r = torch.full((1,), rank, dtype=torch.long, device=device)
    fc1 = model._full_context(ctx, r)

    j1, j2 = _J["mass_1"], _J["mass_2"]
    out = np.empty(len(theta))
    with torch.no_grad():
        for i in range(0, len(theta), batch):
            t = theta[i:i + batch].to(device)
            fc = fc1.expand(len(t), -1)
            y_a = model.scaler.normalize(t)
            t_sw = t.clone()
            t_sw[:, [j1, j2]] = t[:, [j2, j1]]
            y_b = model.scaler.normalize(t_sw)
            la = _log_prob_physical(model, y_a, fc)
            lb = _log_prob_physical(model, y_b, fc)
            out[i:i + batch] = torch.logaddexp(la, lb).cpu().numpy()
    return out


# ── likelihood workers (module-level for multiprocessing spawn) ──────────────

_WORKER = {}


def _init_worker(raw, psds, center_gps, f_min, approximant):
    import bilby
    ifos = bilby.gw.detector.InterferometerList([])
    for det in DETECTORS:
        if det not in raw:
            continue
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=np.asarray(psds[det]["frequencies"], float),
            psd_array=np.asarray(psds[det]["psd"], float))
        ifo.strain_data.set_from_time_domain_strain(
            np.asarray(raw[det], np.float64), sampling_frequency=SAMPLE_RATE,
            duration=DURATION, start_time=center_gps - DURATION / 2)
        ifo.minimum_frequency = f_min
        ifos.append(ifo)
    wfg = bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLE_RATE,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={"waveform_approximant": approximant,
                            "reference_frequency": 50.0,
                            "minimum_frequency": f_min})
    # Phase and time are marginalized ANALYTICALLY: without this, the
    # likelihood demands ~0.1 rad / ~1 ms alignment that a wide amortized
    # proposal essentially never achieves (measured: ESS 1.4/7759 naive vs
    # usable with marginalization). Weights then live in the 9-D "slow"
    # space; see importance_correct for the matching proposal correction.
    pri = bilby.core.prior.PriorDict()
    pri["phase"] = bilby.core.prior.Uniform(0, 2 * np.pi, boundary="periodic")
    pri["geocent_time"] = bilby.core.prior.Uniform(
        center_gps - 1.6, center_gps + 1.6)
    like = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=wfg, priors=pri,
        phase_marginalization=True, time_marginalization=True)
    _WORKER["like"] = like
    _WORKER["center"] = center_gps
    _WORKER["noise_ll"] = like.noise_log_likelihood()


def _eval_chunk(rows):
    like = _WORKER["like"]
    out = np.empty(len(rows))
    for k, th in enumerate(rows):
        p = {n: float(th[j]) for n, j in _J.items()}
        like.parameters = dict(
            mass_1=p["mass_1"], mass_2=p["mass_2"],
            luminosity_distance=p["luminosity_distance"], ra=p["ra"],
            dec=p["dec"], theta_jn=p["theta_jn"], psi=p["psi"],
            phase=0.0, geocent_time=_WORKER["center"],  # marginalized out
            time_jitter=0.0,  # bilby time-marginalization bookkeeping param
            a_1=p["a1"], a_2=p["a2"], tilt_1=0.0, tilt_2=0.0,
            phi_12=0.0, phi_jl=0.0)
        try:
            out[k] = like.log_likelihood() - _WORKER["noise_ll"]
        except Exception:
            out[k] = -np.inf
    return out


def importance_correct(result, raw_strain: Dict[str, np.ndarray],
                       psds: Dict[str, dict], center_gps: float,
                       model_path: Optional[str] = None,
                       npool: int = 4, f_min: float = 20.0,
                       approximant: str = "IMRPhenomXP",
                       device: str = "cpu"):
    """Reweight a PosteriorResult against the true likelihood.

    raw_strain: UNWHITENED 4 s strains keyed by detector (same data the
    NPE saw, pre-whitening). center_gps: absolute GPS of the window center
    (result.trigger_gps for real events; GPS_REF for pipeline-simulated).
    Returns a new PosteriorResult (resampled) with IS diagnostics attached.
    """
    import multiprocessing as mp
    from ahsd.inference.pipeline import load_model
    from ahsd.inference.result import PosteriorResult

    t0 = time.time()
    model, _ = load_model(model_path or result.config["model_path"], device=device)

    keep = np.ones(len(result.samples), bool)
    if result.rail_mask is not None:
        keep &= ~result.rail_mask
    theta = result.samples[keep]
    n = len(theta)

    rank = int(result.config.get("rank", 0))
    log_q = symmetrized_log_q(theta, model, result.prepared, rank, device)
    log_pi = log_prior_training(theta)

    # Marginalized-space correction: the likelihood integrates phase and tc
    # analytically, so the proposal density must be the corresponding
    # MARGINAL. Approximation: the flow's phase conditional is ~uniform
    # (drop a constant); its tc dependence is divided out via the 1-D
    # marginal KDE of the tc samples. Corrected slow-space proposal:
    #   log q_slow = log q_joint - log KDE_tc(tc) - log(1/2pi)
    from scipy.stats import gaussian_kde
    jt = _J["geocent_time"]
    kde_tc = gaussian_kde(theta[:, jt])
    log_q = log_q - np.log(np.maximum(kde_tc(theta[:, jt]), 1e-30)) \
                  + np.log(2 * np.pi)

    # ── adaptive tempered importance sampling ────────────────────────────
    # A wide amortized proposal cannot bridge a likelihood whose log varies
    # by hundreds of nats in one hop (measured: ESS ~1). Climb a beta-ladder:
    # at each rung pick the largest beta whose tempered weights keep
    # ESS >= target, resample, refit a Gaussian-mixture proposal (in slow
    # space: masses/distance in log), redraw, re-evaluate. The FINAL rung is
    # plain self-normalized IS against a mixture whose density is known
    # exactly — asymptotically exact, no approximation carried over.
    from sklearn.mixture import GaussianMixture

    SLOW = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
            "theta_jn", "psi", "a1", "a2"]
    sj = [_J[p] for p in SLOW]
    LOGD = np.array([True, True, True, False, False, False, False, False, False])

    def to_slow(th):
        x = th[:, sj].astype(np.float64).copy()
        x[:, LOGD] = np.log(x[:, LOGD])
        return x

    def from_slow(x, tc_pool, ph_pool, rng):
        th = np.zeros((len(x), len(PARAM_NAMES)))
        xx = x.copy()
        xx[:, LOGD] = np.exp(xx[:, LOGD])
        # enforce the m1 >= m2 convention (mixture density symmetrized below)
        m1 = np.maximum(xx[:, 0], xx[:, 1]); m2 = np.minimum(xx[:, 0], xx[:, 1])
        xx[:, 0], xx[:, 1] = m1, m2
        th[:, sj] = xx
        k = rng.integers(0, len(tc_pool), len(x))
        th[:, _J["geocent_time"]] = tc_pool[k]   # NOT IS-corrected (marginalized)
        th[:, _J["phase"]] = ph_pool[k]          # NOT IS-corrected (marginalized)
        return th

    def gm_logpdf_sym(gm, x):
        xs = x.copy()
        xs[:, [0, 1]] = x[:, [1, 0]]
        return np.logaddexp(gm.score_samples(x), gm.score_samples(xs))

    rng = np.random.default_rng(0)
    tc_pool = theta[:, _J["geocent_time"]].copy()
    ph_pool = theta[:, _J["phase"]].copy()

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(npool, initializer=_init_worker,
                    initargs=(raw_strain, psds, center_gps, f_min, approximant))

    def eval_ll(th):
        return np.concatenate(pool.map(_eval_chunk,
                                       np.array_split(th, max(npool * 8, 8))))

    def pick_beta(log_t1, log_t0, log_g_s, lo, target_frac):
        """Geometric path: log target_b = (1-b) log_t0 + b log_t1, where
        t0 = original flow proposal and t1 = L * pi. Largest b with
        ESS >= target under the current stage proposal g_s."""
        def ess_at(b):
            lw = (1 - b) * log_t0 + b * log_t1 - log_g_s
            m = np.isfinite(lw)
            if m.sum() < 10:
                return 0.0
            v = np.exp(lw[m] - lw[m].max())
            return (v.sum() ** 2 / (v ** 2).sum()) / len(log_t1)
        if ess_at(1.0) >= target_frac:
            return 1.0
        a, b = lo, 1.0
        for _ in range(40):
            mid = 0.5 * (a + b)
            if ess_at(mid) >= target_frac:
                a = mid
            else:
                b = mid
        return max(a, lo + 1e-4)

    target_frac, n_stage, MAX_STAGES = 0.20, n, 25
    # Path anchor: a mixture SURROGATE of the proposal, fitted on the flow
    # samples. Anchoring on the flow density itself fails: its razor tails
    # assign ~e^-10^4 to mixture draws and stall the ladder (measured).
    gm0 = GaussianMixture(n_components=10, covariance_type="full",
                          reg_covar=1e-5, random_state=0).fit(to_slow(theta))
    cur_theta, cur_log_g = theta, log_q
    cur_log_g0 = gm_logpdf_sym(gm0, to_slow(theta))
    cur_log_pi = log_pi
    log_l = eval_ll(cur_theta)
    beta_prev, betas = 0.0, []
    for stage in range(MAX_STAGES):
        log_t1 = log_l + cur_log_pi          # log(L * pi)
        beta = pick_beta(log_t1, cur_log_g0, cur_log_g, beta_prev, target_frac)
        betas.append(round(beta, 4))
        log_w = (1 - beta) * cur_log_g0 + beta * log_t1 - cur_log_g
        fin = np.isfinite(log_w)
        log_w = np.where(fin, log_w, -np.inf)
        w = np.exp(log_w - log_w[fin].max())
        w = np.where(np.isfinite(w), w, 0.0)
        if w.sum() == 0:
            pool.terminate()
            raise RuntimeError("importance weights vanished at "
                               f"beta={beta:.3f} — no likelihood support")
        w /= w.sum()
        if beta >= 1.0:
            break
        # rejuvenate: refit mixture on the tempered posterior, redraw
        u = (rng.uniform() + np.arange(n_stage)) / n_stage
        idx_r = np.searchsorted(np.cumsum(w), u).clip(0, len(w) - 1)
        xs = to_slow(cur_theta[idx_r])
        xs += rng.standard_normal(xs.shape) * (xs.std(0, keepdims=True) * 0.05 + 1e-6)
        gm = GaussianMixture(n_components=min(10, max(2, len(np.unique(idx_r)) // 30)),
                             covariance_type="full", reg_covar=1e-5,
                             random_state=0).fit(xs)
        gm.covariances_ *= 1.69   # defensive widening (x1.3 per axis)
        # refresh cached cholesky after widening
        from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
        gm.precisions_cholesky_ = _compute_precision_cholesky(
            gm.covariances_, "full")
        xn, _ = gm.sample(n_stage)
        cur_theta = from_slow(np.asarray(xn), tc_pool, ph_pool, rng)
        xs_cur = to_slow(cur_theta)
        cur_log_g = gm_logpdf_sym(gm, xs_cur)
        cur_log_g0 = gm_logpdf_sym(gm0, xs_cur)
        cur_log_pi = log_prior_training(cur_theta)
        log_l = eval_ll(cur_theta)
        beta_prev = beta
    pool.terminate()
    if beta < 1.0:
        raise RuntimeError(f"tempering did not reach beta=1 within {MAX_STAGES} "
                           f"stages (ladder: {betas}) — proposal cannot bridge "
                           "this likelihood; treat NPE result as final")

    theta = cur_theta
    ess = float(1.0 / np.sum(w ** 2))
    u = (rng.uniform() + np.arange(n_stage)) / n_stage
    idx = np.searchsorted(np.cumsum(w), u).clip(0, len(w) - 1)
    log_pi = cur_log_pi

    diag = {**result.diagnostics,
            "importance_sampling": {
                "n_proposal": int(len(result.samples)),
                "n_evaluated": int(n),
                "railed_excluded": int((~keep).sum()),
                "importance_ess": round(ess, 1),
                "beta_ladder": betas,
                "n_stages": len(betas),
                "efficiency": round(ess / n, 4),
                "wall_s": round(time.time() - t0, 1),
                "approximant": approximant,
                "note": "phase and geocent_time are analytically marginalized "
                        "in the weights; their columns in the corrected "
                        "samples retain PROPOSAL values (use the NPE result "
                        "for timing).",
            }}
    return PosteriorResult(
        samples=theta[idx], log_prob=(log_l + log_pi)[idx],
        param_names=list(result.param_names), trigger_gps=result.trigger_gps,
        truth=result.truth, prepared=result.prepared, diagnostics=diag,
        config={**result.config, "importance_corrected": True}, rail_mask=None)
