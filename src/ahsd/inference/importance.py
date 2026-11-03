"""
Likelihood correction for LeanNPE posteriors: the amortized posterior is
the PROPOSAL of a tempered sequential Monte Carlo sampler (branch-preserving
resampling, Metropolis moves, bank-seeded particles). Weighting by
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

_J = {n: i for i, n in enumerate(PARAM_NAMES)}


def symmetrized_log_q(theta_np, model, prepared, rank: int = 0,
                      device: str = "cpu", batch: int = 2048) -> np.ndarray:
    """Flow density of SORTED samples: log[q(m1,m2,..) + q(m2,m1,..)],
    physical units. theta_np: [N, 11]."""
    from ahsd.inference.pipeline import _log_prob_physical

    theta = torch.from_numpy(theta_np.astype(np.float32))
    from ahsd.inference.pipeline import model_asd_bands, model_input
    strain = torch.from_numpy(model_input(model, prepared)).unsqueeze(0).to(device).float()
    with torch.no_grad():
        # the same conditional that drew the samples: sensitivity input included
        ctx = model.encode(strain, model_asd_bands(model, prepared, device))
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


def _init_worker(raw, psds, center_gps, f_min, approximant, event_type="BBH"):
    """Multiprocessing pool initializer: build the bilby GW likelihood once per
    worker (phase- and time-marginalized analytically) and cache it + the noise
    log-likelihood in the module-level `_WORKER` dict for `_eval_chunk`."""
    import bilby
    from ahsd.inference.priors import TrainingPrior
    prior = TrainingPrior.for_class(event_type)
    wf = prior.waveform()
    # tides are nuisance parameters, not NPE outputs: each likelihood
    # evaluation draws them from the prior. With q(theta, lambda) =
    # q(theta) pi(lambda) the weight L pi / q is unchanged, so IS marginalizes
    # them exactly (at the cost of some extra variance)
    _WORKER["prior"] = prior
    import os
    _WORKER["rng"] = np.random.default_rng((int(center_gps * 1000) % 2 ** 32, os.getpid()))
    # segment: the window plus whatever precedes it in `raw` (it ends at the
    # window end), so a multi-rate model's 28 s of context is in the likelihood
    dur = segment_duration(raw)
    start = center_gps + DURATION / 2 - dur
    ifos = bilby.gw.detector.InterferometerList([])
    for det in DETECTORS:
        if det not in raw or det not in psds:      # absent, or dropped (glitch) in preprocessing
            continue
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=np.asarray(psds[det]["frequencies"], float),
            psd_array=np.asarray(psds[det]["psd"], float))
        ifo.strain_data.set_from_time_domain_strain(
            np.asarray(raw[det], np.float64), sampling_frequency=SAMPLE_RATE,
            duration=dur, start_time=start)
        ifo.minimum_frequency = f_min
        ifos.append(ifo)
    wfg = bilby.gw.WaveformGenerator(
        duration=dur, sampling_frequency=SAMPLE_RATE,
        frequency_domain_source_model=wf["source_model"],
        parameter_conversion=wf["parameter_conversion"],
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
    _WORKER["start"] = start
    _WORKER["noise_ll"] = like.noise_log_likelihood()


def _eval_chunk(rows):
    """Worker task: return the noise-subtracted log-likelihood for a chunk of
    parameter rows using the cached marginalized likelihood (-inf on failure)."""
    like = _WORKER["like"]
    out = np.empty(len(rows))
    l1s, l2s = _WORKER["prior"].sample_tides(len(rows), _WORKER["rng"])
    for k, th in enumerate(rows):
        p = {n: float(th[j]) for n, j in _J.items()}
        like.parameters = dict(
            mass_1=p["mass_1"], mass_2=p["mass_2"],
            luminosity_distance=p["luminosity_distance"], ra=p["ra"],
            dec=p["dec"], theta_jn=p["theta_jn"], psi=p["psi"],
            # marginalized out. With time marginalization bilby searches the
            # time prior RELATIVE to geocent_time = the data start time (it
            # replaces the prior by that delta itself in run_sampler); passing
            # the window centre shifts the search 2 s late and misses the
            # signal entirely
            phase=0.0, geocent_time=_WORKER["start"],
            time_jitter=0.0,  # bilby time-marginalization bookkeeping param
            # signed aligned spins: magnitude and tilt pi when negative
            a_1=abs(p["a1"]), a_2=abs(p["a2"]),
            tilt_1=np.pi if p["a1"] < 0 else 0.0, tilt_2=np.pi if p["a2"] < 0 else 0.0,
            phi_12=0.0, phi_jl=0.0, lambda_1=float(l1s[k]), lambda_2=float(l2s[k]))
        try:
            out[k] = like.log_likelihood() - _WORKER["noise_ll"]
        except Exception:
            out[k] = -np.inf
    return out


def _bank_seeds(model, result, theta, rng, d_snr2: float = 9.0, per_slot: int = 400):
    """NPE samples with (m1, m2, a1) moved to each plausible slot of the
    chirp-mass bank (peak SNR^2 within d_snr2 of the best slot), jittered;
    None when the model has no multi-slot bank."""
    mf = getattr(model.encoder, "mc_feat", None)
    if mf is None or getattr(mf, "n_q", 1) < 2 or result.prepared is None:
        return None
    from ahsd.inference.pipeline import model_asd_bands, model_input
    dev = next(model.parameters()).device
    x = torch.from_numpy(model_input(model, result.prepared)).unsqueeze(0).to(dev).float()
    with torch.no_grad():
        model.encode(x, model_asd_bands(model, result.prepared, dev))
    sl = mf.last_slot[0].cpu().numpy().reshape(-1, 2)                 # (norm log Mc, asinh SNR)
    snr2 = np.sinh(sl[:, 1]) ** 2
    out = []
    for j in np.where(snr2 >= snr2.max() - d_snr2)[0]:
        mc = float(np.exp((sl[j, 0] + 1) / 2 * (mf.log_hi - mf.log_lo) + mf.log_lo))
        q = float(mf.slot_q_chi[j, 0])
        chi = float(mf.slot_q_chi[j, 1]) if mf.slot_q_chi.shape[1] > 1 else None
        th = theta[rng.integers(0, len(theta), per_slot)].copy()
        mcj = mc * np.exp(rng.normal(0, 0.01, per_slot))
        qj = np.clip(q + rng.normal(0, 0.05, per_slot), 0.02, 1.0)
        M = mcj * (1 + qj) ** 1.2 / qj ** 0.6
        th[:, _J["mass_1"]], th[:, _J["mass_2"]] = M / (1 + qj), M * qj / (1 + qj)
        if chi is not None:
            th[:, _J["a1"]] = np.clip(chi + rng.normal(0, 0.1, per_slot), -0.89, 0.89)
        out.append(th)
    return np.concatenate(out) if out else None


def segment_duration(raw: Dict[str, np.ndarray]) -> float:
    """Seconds covered by the raw strains (all detectors the same length)."""
    lens = {len(v) for v in raw.values()}
    if len(lens) != 1:
        raise ValueError(f"raw strains differ in length: {sorted(lens)}")
    return lens.pop() / SAMPLE_RATE


def importance_correct(result, raw_strain: Dict[str, np.ndarray],
                       psds: Dict[str, dict], center_gps: float,
                       model_path: Optional[str] = None,
                       npool: int = 4, f_min: Optional[float] = None,
                       approximant: Optional[str] = None,
                       device: str = "cpu", seed: int = 0, seed_bank: Optional[bool] = None):
    """Reweight a PosteriorResult against the true likelihood.

    raw_strain: UNWHITENED strains keyed by detector, ending at the window end
    (centre + 2 s): the 4 s window, or for a multi-rate model the window plus
    the 28 s before it (32 s), so the likelihood covers what the NPE read. center_gps: absolute GPS of the window center
    (result.trigger_gps for real events; GPS_REF for pipeline-simulated).
    Returns a new PosteriorResult (resampled) with IS diagnostics attached.
    """
    import multiprocessing as mp
    from ahsd.inference.pipeline import load_model
    from ahsd.inference.result import PosteriorResult

    t0 = time.time()
    model, _ = load_model(model_path or result.config["model_path"], device=device)
    caveats = []
    dur = segment_duration(raw_strain)
    lo_s = getattr(model, "lowrate_len", 0) / 512.0
    if lo_s and dur < DURATION + lo_s - 1e-6:
        # the network also read the 28 s before the window: a shorter likelihood
        # is a wider target than the proposal, so the weights are less reliable
        caveats.append(f"multi-rate model: IS likelihood covers {dur:.0f} s, the proposal "
                       f"used {DURATION + lo_s:.0f} s; pass the window plus the "
                       f"{lo_s:.0f} s before it")
        import logging
        logging.getLogger(__name__).warning(caveats[-1])

    keep = np.ones(len(result.samples), bool)
    if result.rail_mask is not None:
        keep &= ~result.rail_mask
    theta = result.samples[keep]
    n = len(theta)

    rank = int(result.config.get("rank", 0))
    log_q = symmetrized_log_q(theta, model, result.prepared, rank, device)
    # the prior the model was trained on, for the class the event is analysed as
    from ahsd.inference.priors import TrainingPrior
    prior = TrainingPrior.from_meta(result.config)
    log_pi = prior.log_prob(theta)
    # waveform model of the class; band that fits the 4 s segment for the
    # lightest chirp mass / earliest merger among the proposal samples
    from ahsd.inference.priors import chirp_mass, window_fmin
    approximant = approximant or prior.waveform()["approximant"]
    if f_min is None:
        mc_lo = float(np.quantile(chirp_mass(theta[:, _J["mass_1"]], theta[:, _J["mass_2"]]), 0.005))
        tc_lo = float(np.quantile(theta[:, _J["geocent_time"]], 0.005))
        f_min = window_fmin(mc_lo, dur - DURATION / 2 + tc_lo)
    # A raised f_min can cut most of a heavy signal: on a 4 s segment with an
    # early merger the band that fits starts above the merger frequency, the
    # likelihood is nearly flat in mass and the particles drift to the prior's
    # edge, the same way for every seed (measured: an SNR-10, M=94 injection
    # with tc = -1.4 s gave f_min 85.6 Hz and m1 6 against a true 86). Flag it
    # when the leading-order merger frequency of >10% of the proposal lies
    # below f_min; pass a longer segment (data before the window) to fix it.
    m_tot = theta[:, _J["mass_1"]] + theta[:, _J["mass_2"]]
    band_cut = float(np.mean(4400.0 / m_tot < f_min))
    if band_cut > 0.1:
        caveats.append(f"likelihood band starts at {f_min:.0f} Hz, above the merger frequency "
                       f"of {band_cut:.0%} of the proposal: the correction sees little of the "
                       f"signal and is not reliable; pass a longer segment")

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

    # ── adaptive tempered importance sampling (SMC) ──────────────────────
    # A wide amortized proposal cannot bridge a likelihood whose log varies
    # by hundreds of nats in one hop (measured: ESS ~1). Climb a beta-ladder:
    # at each rung pick the largest beta whose tempered weights keep
    # ESS >= target, resample, and move the particles with Metropolis steps
    # that leave the tempered target invariant (sequential Monte Carlo;
    # slow space: masses/distance in log). The last rung (beta = 1) is
    # self-normalized IS from the previous tempered target.
    from sklearn.mixture import GaussianMixture

    SLOW = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
            "theta_jn", "psi", "a1", "a2"]
    sj = [_J[p] for p in SLOW]
    LOGD = np.array([True, True, True, False, False, False, False, False, False])

    def to_slow(th):
        """Map physical params to the 9-D "slow" IS space (masses/distance in
        log), dropping the analytically-marginalized phase and time."""
        x = th[:, sj].astype(np.float64).copy()
        x[:, LOGD] = np.log(x[:, LOGD])
        return x

    def from_slow(x, tc_pool, ph_pool, rng):
        """Inverse of `to_slow`: exponentiate log-dims, enforce m1 >= m2, and
        re-attach time/phase by resampling the marginalized pools."""
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
        """Gaussian-mixture log-density symmetrized under the m1<->m2 label
        swap (the mixture is fit in slow space where mass ordering is enforced)."""
        xs = x.copy()
        xs[:, [0, 1]] = x[:, [1, 0]]
        return np.logaddexp(gm.score_samples(x), gm.score_samples(xs))

    rng = np.random.default_rng(seed)
    tc_pool = theta[:, _J["geocent_time"]].copy()
    ph_pool = theta[:, _J["phase"]].copy()

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(npool, initializer=_init_worker,
                    initargs=(raw_strain, psds, center_gps, f_min, approximant,
                              prior.event_type))

    def eval_ll(th):
        """Evaluate the log-likelihood for all rows of `th` in parallel across
        the worker pool."""
        return np.concatenate(pool.map(_eval_chunk,
                                       np.array_split(th, max(npool * 8, 8))))

    def pick_beta(log_t1, log_t0, log_g_s, lo, target_frac):
        """Geometric path: log target_b = (1-b) log_t0 + b log_t1, where
        t0 = original flow proposal and t1 = L * pi. Largest b with
        ESS >= target under the current stage proposal g_s."""
        def ess_at(b):
            """Fractional effective sample size of the tempered weights at
            inverse-temperature `b` under the current stage proposal."""
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

    # 15 moves per stage: with 5 the particles stayed too close to their
    # resampled parents (GW200115: m1 90% width 0.4 vs dynesty's 1.5)
    target_frac, MAX_STAGES, N_MOVES = 0.20, 40, 15
    # Path anchor: a mixture SURROGATE of the proposal, fitted on the flow
    # samples. Anchoring on the flow density itself fails: its razor tails
    # assign ~e^-10^4 to mixture draws and stall the ladder (measured).
    gm0 = GaussianMixture(n_components=10, covariance_type="full",
                          reg_covar=1e-5, random_state=0).fit(to_slow(theta))

    def branches(x, k_max=4):
        """Cluster labels of particles in slow space (BIC over 1..k_max
        Gaussian components, fitted on a subsample): the posterior's separate
        branches, e.g. GW200115's light-BH / anti-aligned-spin and heavier /
        small-spin modes."""
        z = (x - x.mean(0)) / (x.std(0) + 1e-12)
        sub = z[rng.choice(len(z), size=min(len(z), 3000), replace=False)]
        fits = [GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=0).fit(sub)
                for k in range(1, k_max + 1)]
        best = min(fits, key=lambda g: g.bic(sub))
        return best.predict(z)

    def resample(w, labels):
        """Systematic resampling within each branch, branch counts set by
        branch weight (residual allocation): unbiased, and a branch holding
        weight >= 1/n is never lost to resampling noise, which is what
        collapsed GW200115 onto one branch in some runs and the other in
        others."""
        out = []
        cls = np.unique(labels)
        W = np.array([w[labels == c].sum() for c in cls])
        n_c = np.floor(W * n).astype(int)
        rest = n - n_c.sum()
        if rest > 0:
            n_c[np.argsort(-(W * n - n_c))[:rest]] += 1
        for c, k in zip(cls, n_c):
            if k == 0:
                continue
            ids = np.where(labels == c)[0]
            wc = w[ids] / w[ids].sum()
            u = (rng.uniform() + np.arange(k)) / k
            out.append(ids[np.searchsorted(np.cumsum(wc), u).clip(0, len(ids) - 1)])
        return np.concatenate(out)

    def mh_moves(th, log_t1, log_g0, beta, scale):
        """Random-walk Metropolis moves leaving the tempered target
        t_b = g0^(1-b) (L pi)^b invariant (slow space; time and phase are
        marginalized in L). A mixture refit alone cannot climb onto the
        likelihood's narrow correlated ridge: on GW150914 the best of 400
        NPE samples sat 16 nats below the peak and the ladder stalled at
        beta ~ 0.06 although every 1-D marginal agreed with dynesty."""
        x = to_slow(th)
        # proposal covariance per branch (a pooled covariance spans the gap
        # between branches: steps too large inside each, ~0 acceptance)
        labels = branches(x)
        eye = 1e-9 * np.eye(x.shape[1])
        chol_all = np.linalg.cholesky(np.cov(x.T) + eye)
        chols = {c: (np.linalg.cholesky(np.cov(x[labels == c].T) + eye)
                     if (labels == c).sum() > 3 * x.shape[1] else chol_all) for c in np.unique(labels)}
        acc_all = []
        for _ in range(N_MOVES):
            step = rng.standard_normal(x.shape)
            for c, L in chols.items():
                m = labels == c
                step[m] = step[m] @ L.T
            xp = x + scale * step
            thp = from_slow(xp, tc_pool, ph_pool, rng)
            thp[:, _J["geocent_time"]] = th[:, _J["geocent_time"]]
            thp[:, _J["phase"]] = th[:, _J["phase"]]
            xp = to_slow(thp)
            lpp = prior.log_prob(thp)
            t1p = np.full(len(thp), -np.inf)
            ok = np.isfinite(lpp)
            if ok.any():
                t1p[ok] = eval_ll(thp[ok]) + lpp[ok]
            g0p = gm_logpdf_sym(gm0, xp)
            log_a = (1 - beta) * (g0p - log_g0) + beta * (t1p - log_t1)
            acc = np.log(rng.uniform(size=len(x))) < np.where(np.isfinite(log_a), log_a, -np.inf)
            th = np.where(acc[:, None], thp, th)
            x = np.where(acc[:, None], xp, x)
            log_t1 = np.where(acc, t1p, log_t1)
            log_g0 = np.where(acc, g0p, log_g0)
            acc_all.append(acc.mean())
            # adapt towards ~25% acceptance
            scale *= float(np.exp(acc.mean() - 0.25))
        return th, log_t1, log_g0, scale, float(np.mean(acc_all))

    cur_theta, cur_log_g = theta, log_q
    # default: seed from the bank whenever it has several (q, spin) slots
    # (GW200115 mode recall vs dynesty 0.25-0.75 -> 0.75-1.0 with NSBH v5ff)
    if seed_bank is None:
        seed_bank = getattr(getattr(model.encoder, "mc_feat", None), "n_q", 1) > 1
    seeds = _bank_seeds(model, result, theta, rng) if seed_bank else None
    if seeds is not None and len(seeds):
        # initial particles drawn from a mixture fitted to NPE samples plus
        # bank-seeded copies (each plausible (q, spin) slot of the matched
        # filter): a secondary branch the NPE smears over gets particles from
        # the start. The mixture density is known, so the weights stay exact.
        pool_x = to_slow(np.concatenate([theta, seeds]))
        gms = GaussianMixture(n_components=min(16, 8 + len(seeds) // 200), covariance_type="full",
                              reg_covar=1e-5, random_state=seed).fit(pool_x)
        xn, _ = gms.sample(n)
        cur_theta = from_slow(np.asarray(xn), tc_pool, ph_pool, rng)
        cur_log_g = gm_logpdf_sym(gms, to_slow(cur_theta))
        log_pi = prior.log_prob(cur_theta)
    cur_log_g0 = gm_logpdf_sym(gm0, to_slow(cur_theta))
    log_l = eval_ll(cur_theta)
    if not np.isfinite(log_l).any():
        pool.terminate()
        raise RuntimeError(f"likelihood is -inf for every sample: the {approximant} "
                           "waveform failed (model not available in this LAL installation?)")
    log_t1 = log_l + log_pi              # log(L * pi)
    beta_prev, betas, accs = 0.0, [], []
    scale = 2.38 / np.sqrt(len(SLOW)) * 0.5
    for stage in range(MAX_STAGES):
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
        # SMC step: resample by the tempered weights, then Metropolis moves
        # at this beta. The moved particles follow t_beta, so the next
        # stage's proposal density is t_beta itself (unnormalized: the
        # weights are self-normalized), and pick_beta / the weights above
        # stay exact.
        idx_r = resample(w, branches(to_slow(cur_theta)))
        cur_theta, log_t1, cur_log_g0, scale, acc = mh_moves(
            cur_theta[idx_r], log_t1[idx_r], cur_log_g0[idx_r], beta, scale)
        accs.append(round(acc, 3))
        cur_log_g = (1 - beta) * cur_log_g0 + beta * log_t1
        beta_prev = beta
    pool.terminate()
    if beta < 1.0:
        raise RuntimeError(f"tempering did not reach beta=1 within {MAX_STAGES} "
                           f"stages (ladder: {betas}) — proposal cannot bridge "
                           "this likelihood; treat NPE result as final")
    theta = cur_theta
    ess = float(1.0 / np.sum(w ** 2))
    u = (rng.uniform() + np.arange(n)) / n
    idx = np.searchsorted(np.cumsum(w), u).clip(0, len(w) - 1)

    diag = {**result.diagnostics,
            "importance_sampling": {
                "n_proposal": int(len(result.samples)),
                "n_evaluated": int(n),
                "railed_excluded": int((~keep).sum()),
                "importance_ess": round(ess, 1),
                "beta_ladder": betas,
                "mh_acceptance": accs,
                # distinct particles among the returned samples: low values
                # mean the moves did not spread the resampled particles
                "unique_frac": round(float(len(np.unique(theta[idx][:, 0]))) / n, 3),
                "n_stages": len(betas),
                "efficiency": round(ess / n, 4),
                "wall_s": round(time.time() - t0, 1),
                "approximant": approximant,
                "f_min": round(f_min, 1),
                "band_cut_frac": round(band_cut, 3),
                "reliable": band_cut <= 0.1,
                "caveats": caveats,
                "note": "phase and geocent_time are analytically marginalized "
                        "in the weights; their columns in the corrected "
                        "samples retain PROPOSAL values (use the NPE result "
                        "for timing).",
            }}
    out = theta[idx].copy()
    # the Metropolis moves are unconstrained random walks: angles leave their
    # range (the likelihood is periodic, so the weights are right). Fold them
    # back, or quantiles, medians and plots of ra/psi/phase come out wrong.
    from ahsd.models.lean_npe import ParamScaler
    for j, name in enumerate(result.param_names):
        if name in ParamScaler.CIRCULAR and name in ParamScaler.RANGES:
            lo, hi = ParamScaler.RANGES[name][:2]
            out[:, j] = lo + np.mod(out[:, j] - lo, hi - lo)
    return PosteriorResult(
        samples=out, log_prob=log_t1[idx],
        param_names=list(result.param_names), trigger_gps=result.trigger_gps,
        truth=result.truth, prepared=result.prepared, diagnostics=diag,
        config={**result.config, "importance_corrected": True}, rail_mask=None)


def replicate_spread(results, param_names, names=("mass_1", "mass_2", "luminosity_distance")):
    """Largest spread of replicate medians, in units of the mean 90% width,
    over `names`. Independent-seed corrections of a multimodal posterior can
    come out on different branches; a spread above 0.5 of a 90% width means
    the correction is not reliable for that event (hand it to dynesty)."""
    worst = 0.0
    for pname in names:
        j = list(param_names).index(pname)
        meds = [np.median(r.samples[:, j]) for r in results]
        w90 = np.mean([np.subtract(*np.quantile(r.samples[:, j], [0.95, 0.05])) for r in results])
        worst = max(worst, float(np.ptp(meds) / max(w90, 1e-12)))
    return worst


def corrected_with_check(result, raw_strain, reps: int = 2, tol: float = 0.5, **kw):
    """importance_correct with `reps` independent seeds. Returns the seed-0
    corrected PosteriorResult; its diagnostics["importance_sampling"] gains
    replicate_spread and consistent (spread < tol). kw go to
    PosteriorResult.importance_correct (npool, device, ...)."""
    runs = [result.importance_correct(raw_strain, seed=k, **kw) for k in range(max(1, reps))]
    d = runs[0].diagnostics["importance_sampling"]
    if len(runs) > 1:
        spread = replicate_spread(runs, result.param_names)
        d["replicate_spread"] = round(spread, 3)
        d["consistent"] = spread < tol
        d["reliable"] = bool(d.get("reliable", True) and d["consistent"])
    return runs[0]
