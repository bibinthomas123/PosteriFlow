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
    from ahsd.inference.pipeline import _log_prob_physical, ra_rotation

    # samples report the true sky; the flow's variable is the sky at training epoch
    theta_np = theta_np.copy()
    theta_np[:, _J["ra"]] = np.mod(theta_np[:, _J["ra"]] - ra_rotation(prepared), 2 * np.pi)
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


def _init_worker(raw, psds, center_gps, f_min, approximant, event_type="BBH",
                 explicit_phase=False):
    """Multiprocessing pool initializer: build the bilby GW likelihood once per
    worker (phase- and time-marginalized analytically) and cache it + the noise
    log-likelihood in the module-level `_WORKER` dict for `_eval_chunk`."""
    import bilby
    from ahsd.inference.priors import LIKELIHOOD_FMAX, TrainingPrior
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
        ifo.maximum_frequency = LIKELIHOOD_FMAX
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
        phase_marginalization=not explicit_phase, time_marginalization=True)
    _WORKER["explicit_phase"] = bool(explicit_phase)
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
            # phase: marginalized analytically (exact for the dominant mode
            # only) unless sampled explicitly (higher-mode waveforms)
            phase=p["phase"] if _WORKER.get("explicit_phase") else 0.0,
            geocent_time=_WORKER["start"],
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


def _wq(v, w, qs=(0.1, 0.5, 0.9)):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return [float(v[o][min(np.searchsorted(c, q * c[-1]), len(v) - 1)]) for q in qs]


def _stage_summary(beta, theta, w, labels):
    """Weighted distance / dec quantiles and branch weights of one SMC stage."""
    cls = np.unique(labels)
    return {"beta": round(float(beta), 4),
            "ess_frac": round(float(1.0 / np.sum(w ** 2) / len(w)), 4),
            "dL_q": [round(x, 1) for x in _wq(theta[:, _J["luminosity_distance"]], w)],
            "dec_q": [round(x, 3) for x in _wq(theta[:, _J["dec"]], w)],
            # weighted mass in dec bins (sky islands show up as separate bins)
            "dec_hist": [round(float(v), 3) for v in np.histogram(
                theta[:, _J["dec"]], bins=[-1.6, -0.9, 0.1, 1.6], weights=w)[0]],
            "branch_w": [round(float(w[labels == c].sum()), 3) for c in cls],
            "branch_dL": [round(float(np.median(theta[labels == c, _J["luminosity_distance"]])), 0) for c in cls]}


def _inflated(gm, k: float):
    """Copy of a fitted full-covariance GaussianMixture with every component
    covariance multiplied by k (same weights and means)."""
    import copy
    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
    g = copy.deepcopy(gm)
    g.covariances_ = gm.covariances_ * float(k)
    g.precisions_cholesky_ = _compute_precision_cholesky(g.covariances_, "full")
    g.precisions_ = np.array([np.linalg.inv(c) for c in g.covariances_])
    return g


def floored_resample(w, labels, n, branch_floor, rng):
    """Systematic resampling within each branch, with a FLOOR on every
    branch's particle count; the resampled particles carry their
    branch's weight (log W_c / n_c) instead of uniform weights.

    Allocating counts in proportion to branch weight loses any branch
    whose weight dips below ~1/n at an intermediate temperature, and it
    never comes back: the anchor g0^(1-beta) favours wherever the network
    put its mass, so a true mode the network missed is suppressed mid-
    ladder and regains its weight only near beta = 1. Seeds then landed
    on one mode or the other (edge-on injection: distance 260 vs 410
    Mpc; GW200115). With a floor every branch stays populated, and the
    carried weights keep the particle set an exact representation of
    the tempered target (Metropolis moves leave it invariant, so they
    act on weighted particles just as well). Returns (indices, log
    weights)."""
    cls = np.unique(labels)
    W = np.array([w[labels == c].sum() for c in cls])
    live = W > 1e-12 * W.sum()
    n_live = int(live.sum())
    floor = min(max(1, int(branch_floor * n)), n // (2 * n_live)) if n_live > 1 else 0
    n_c = np.zeros(len(cls), int)
    n_c[live] = floor
    free = n - n_c.sum()
    share = np.where(live, W, 0.0) / W[live].sum() * free
    n_c += np.floor(share).astype(int)
    rest = n - n_c.sum()
    if rest > 0:      # largest remainders, live branches only
        frac = np.where(live, share - np.floor(share), -1.0)
        n_c[np.argsort(-frac)[:rest]] += 1
    out, lws = [], []
    for c, k, Wc in zip(cls, n_c, W):
        if k == 0:
            continue
        ids = np.where(labels == c)[0]
        wc = w[ids] / w[ids].sum()
        u = (rng.uniform() + np.arange(k)) / k
        out.append(ids[np.searchsorted(np.cumsum(wc), u).clip(0, len(ids) - 1)])
        lws.append(np.full(k, np.log(Wc / W.sum()) - np.log(k)))
    lw = np.concatenate(lws)
    return np.concatenate(out), lw - np.log(np.exp(lw - lw.max()).sum()) - lw.max()


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
                       device: str = "cpu", seed: int = 0, seed_bank: Optional[bool] = None,
                       defensive: float = 0.2, target_ess: float = 0.50, n_moves: int = 15, branch_floor: float = 0.05,
                       n_ind: int = 3, n_rejuvenate: int = 2, n_sky: int = 2,
                       sky_seeds: bool = True, n_sky_final: int = 12,
                       anchor_inflate: float = 4.0, anchor_sky_flat: bool = True,
                       start_particles: Optional[np.ndarray] = None,
                       n_ridge: int = 2, n_ridge_final: int = 6,
                       explicit_phase: Optional[bool] = None, anchor_flat_spins: bool = True):
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
    if len(result.param_names) > len(_J):
        raise NotImplementedError(
            "likelihood correction of a precessing posterior is not implemented: "
            "its likelihood would treat the spins as aligned")
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

    # Higher-mode waveforms: the analytic phase marginalization assumes the
    # phase enters as one overall rotation, which holds for the dominant
    # (2, 2) mode only (each (l, m) mode rotates by m * phase). Sample the
    # phase as a coordinate then.
    if explicit_phase is None:
        explicit_phase = "HM" in str(approximant).upper()
    SLOW = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
            "theta_jn", "psi", "a1", "a2"] + (["phase"] if explicit_phase else [])

    # Sampling coordinates: unbounded (logit / atanh) for every bounded
    # parameter, periodic angles wrapped, mixture densities wrapped and
    # symmetrized over the mass swap (ahsd.inference.coords). In the old
    # log coordinates the prior's hard edges made Gaussian-mixture proposals
    # and anchors degenerate (flat-likelihood check: final ESS 9 of 4000).
    from ahsd.inference.coords import SlowCoords
    C = SlowCoords(prior, with_phase=explicit_phase)
    sj = C.idx

    def log_jac(th):
        """log |d theta / d x| of the sampling coordinates: every physical
        density (likelihood x prior, flow) is converted to them with it."""
        return C.log_jac(th)

    def to_slow(th):
        """Physical params -> wrapped unbounded sampling coordinates (9-D:
        phase and time are marginalized and dropped; 10-D with an explicit
        phase)."""
        return C.wrap(C.to_x(th))

    def from_slow(x, tc_pool, ph_pool, rng):
        """Inverse of `to_slow` (angles wrapped, m1 >= m2 enforced), with
        time/phase re-attached from the marginalized pools."""
        th = np.zeros((len(x), len(PARAM_NAMES)))
        v = C.from_x(C.wrap(x))
        m1 = np.maximum(v[:, 0], v[:, 1]); m2 = np.minimum(v[:, 0], v[:, 1])
        v[:, 0], v[:, 1] = m1, m2
        th[:, sj] = v
        k = rng.integers(0, len(tc_pool), len(x))
        th[:, _J["geocent_time"]] = tc_pool[k]   # NOT IS-corrected (marginalized)
        if not explicit_phase:
            th[:, _J["phase"]] = ph_pool[k]      # NOT IS-corrected (marginalized)
        return th

    def gm_logpdf_sym(gm, x):
        """Mixture density of wrapped, mass-sorted sampling points (periodic
        images and the m1 <-> m2 swap summed)."""
        return C.mixture_logpdf(gm, x)

    rng = np.random.default_rng(seed)
    tc_pool = theta[:, _J["geocent_time"]].copy()
    ph_pool = theta[:, _J["phase"]].copy()

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(npool, initializer=_init_worker,
                    initargs=(raw_strain, psds, center_gps, f_min, approximant,
                              prior.event_type, explicit_phase))

    def eval_ll(th):
        """Evaluate the log-likelihood for all rows of `th` in parallel across
        the worker pool."""
        return np.concatenate(pool.map(_eval_chunk,
                                       np.array_split(th, max(npool * 8, 8))))

    def pick_beta(log_t1, log_t0, log_g_s, lo, target_frac, lw_carry=0.0):
        """Geometric path: log target_b = (1-b) log_t0 + b log_t1, where
        t0 = original flow proposal and t1 = L * pi. Largest b with
        ESS >= target under the current stage proposal g_s."""
        def ess_at(b):
            """Fractional effective sample size of the tempered weights at
            inverse-temperature `b` under the current stage proposal."""
            lw = lw_carry + (1 - b) * log_t0 + b * log_t1 - log_g_s
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
    # ESS 0.5 per stage (not 0.2): with 20% the ~7-stage ladder left the
    # final particles descended from a few early ancestors, and the mass of
    # each posterior branch depended on which island those sat on (edge-on
    # injection: cycle-slip island 27-92% across seeds vs ~6%); with 50%,
    # ~12 stages, 3.5-17%
    target_frac, MAX_STAGES, N_MOVES = float(target_ess), 80, int(n_moves)
    # Path anchor: a mixture SURROGATE of the proposal, fitted on the flow
    # samples. Anchoring on the flow density itself fails: its razor tails
    # assign ~e^-10^4 to mixture draws and stall the ladder (measured).
    gm0 = GaussianMixture(n_components=10, covariance_type="full",
                          reg_covar=1e-5, random_state=0).fit(to_slow(theta))

    def cluster_features(x):
        """Clustering space: the sky as a unit vector and psi as the angle
        2 psi (the raw ra cut at 0 / 2 pi split islands near ra = 0 and
        merged others), the rest unchanged."""
        ra, sd_ = x[:, 3], np.tanh(x[:, 4])
        cd_ = np.sqrt(np.maximum(1 - sd_ ** 2, 0.0))
        cols = [x[:, :3], cd_ * np.cos(ra), cd_ * np.sin(ra), sd_, x[:, 5],
                np.cos(2 * x[:, 6]), np.sin(2 * x[:, 6]), x[:, 7:9]]
        if x.shape[1] > 9:                   # explicit phase, as an angle
            cols += [np.cos(x[:, 9]), np.sin(x[:, 9])]
        return np.column_stack(cols)

    def branches(x, k_max=6, classifier=False):
        """Cluster labels of particles in slow space (BIC over 1..k_max
        Gaussian components, fitted on a subsample): the posterior's separate
        branches, e.g. GW200115's light-BH / anti-aligned-spin and heavier /
        small-spin modes, or an edge-on source's sky islands. classifier=True
        also returns the labelling as a fixed function of a point (for
        state-dependent proposals)."""
        f = cluster_features(x)
        mu, sd = f.mean(0), f.std(0) + 1e-12
        z = (f - mu) / sd
        sub = z[rng.choice(len(z), size=min(len(z), 3000), replace=False)]
        fits = [GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=0).fit(sub)
                for k in range(1, k_max + 1)]
        best = min(fits, key=lambda g: g.bic(sub))
        if classifier:
            return best.predict(z), (lambda y: best.predict((cluster_features(y) - mu) / sd))
        return best.predict(z)

    def resample(w, labels):
        return floored_resample(w, labels, n, branch_floor, rng)

    ind, ind_acc, sky_acc, ridge_acc = {"gm": None}, [], [], []
    from ahsd.inference.ridgemoves import DistanceInclinationMove, RidgeMove
    # ridge moves: mass-ratio / spin, and distance / inclination
    ridges = (RidgeMove(), DistanceInclinationMove())
    anchor = {"g": None}
    from ahsd.inference.skymoves import SkyJumps
    sky = SkyJumps([d for d in DETECTORS if d in raw_strain], center_gps, psds,
                   f_lo=f_min or 20.0)

    def log_dd_dx(th):
        """log |d d_L / d x_d| of the distance's sampling coordinate."""
        lo, hi = C.d
        z = (th[:, _J["luminosity_distance"]] - lo) / (hi - lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(hi - lo) + np.log(z) + np.log1p(-z)

    def mh_moves(th, log_t1, log_g0, beta, scale):
        """Random-walk Metropolis moves leaving the tempered target
        t_b = g0^(1-b) (L pi)^b invariant (slow space; time and phase are
        marginalized in L). A mixture refit alone cannot climb onto the
        likelihood's narrow correlated ridge: on GW150914 the best of 400
        NPE samples sat 16 nats below the peak and the ladder stalled at
        beta ~ 0.06 although every 1-D marginal agreed with dynesty."""
        x = to_slow(th)
        # proposal covariance per branch (a pooled covariance spans the gap
        # between branches: steps too large inside each, ~0 acceptance).
        # The branch is that of the CURRENT point at every step, and the
        # Hastings ratio carries q(x | x') / q(x' | x) whenever x and x' sit
        # in different branches. Without it the kernel is not reversible: a
        # particle's covariance was fixed by its starting branch, and on a
        # toy target with two step sizes the region with the larger steps
        # was drained (P 0.38 -> 0.14; tests/test_importance_helpers.py).
        # On a posterior ridge that the clustering cuts into segments
        # (GW200115's m1-q-spin degeneracy) particles cross branches all
        # the time, and mass collected in the narrow segments.
        labels, classify = branches(x, classifier=True)
        eye = 1e-9 * np.eye(x.shape[1])
        # Proposed masses are SORTED (m1 >= m2) after each step, i.e. the walk
        # is folded across m1 = m2. A folded walk is symmetric (detailed
        # balance holds without a correction term) only if the step
        # distribution is invariant under swapping the two mass coordinates;
        # a raw sample covariance is not, and the kernel then does not leave
        # the target invariant (flat-likelihood check: masses biased low).
        # Average each covariance with its mass-swapped copy.
        def swap_sym(c):
            pc = c.copy()
            pc[[0, 1], :] = pc[[1, 0], :]
            pc[:, [0, 1]] = pc[:, [1, 0]]
            return 0.5 * (c + pc)
        chol_all = np.linalg.cholesky(swap_sym(np.cov(x.T)) + eye)
        chols = {c: (np.linalg.cholesky(swap_sym(np.cov(x[labels == c].T)) + eye)
                     if (labels == c).sum() > 3 * x.shape[1] else chol_all) for c in np.unique(labels)}
        # precision and log-determinant of each branch's step covariance
        precs = {c: np.linalg.inv(L @ L.T) for c, L in chols.items()}
        logdets = {c: 2.0 * np.sum(np.log(np.diag(L))) for c, L in chols.items()}

        def log_q(x_from, x_to, lab_from):
            out = np.empty(len(x_from))
            for c in chols:
                m = lab_from == c
                if m.any():
                    out[m] = C.rw_logpdf(x_from[m], x_to[m], precs[c], logdets[c], scale)
            return out

        acc_all = []
        for _ in range(N_MOVES):
            labels = classify(x)
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
                t1p[ok] = eval_ll(thp[ok]) + lpp[ok] + log_jac(thp[ok])
            # a large step in log mass / distance can overflow to inf; such
            # rows are rejected (sklearn refuses non-finite input: GW200129
            # failed with "Input X contains infinity")
            fin = np.isfinite(xp).all(axis=1)
            g0p = np.full(len(xp), -np.inf)
            if fin.any():
                g0p[fin] = anchor_logpdf(xp[fin])
            log_a = (1 - beta) * (g0p - log_g0) + beta * (t1p - log_t1)
            # Hastings correction where the step changed branch (same branch:
            # the folded, wrapped walk is symmetric and the terms cancel)
            lab_p = np.full(len(xp), -1)
            lab_p[fin] = classify(xp[fin])
            cross = fin & (lab_p != labels) & np.isfinite(log_a)
            if cross.any():
                log_a[cross] += (log_q(xp[cross], x[cross], lab_p[cross])
                                 - log_q(x[cross], xp[cross], labels[cross]))
            acc = np.log(rng.uniform(size=len(x))) < np.where(np.isfinite(log_a), log_a, -np.inf)
            th = np.where(acc[:, None], thp, th)
            x = np.where(acc[:, None], xp, x)
            log_t1 = np.where(acc, t1p, log_t1)
            log_g0 = np.where(acc, g0p, log_g0)
            acc_all.append(acc.mean())
            # adapt towards ~25% acceptance
            scale *= float(np.exp(acc.mean() - 0.25))
        # Independence moves from the broad initial mixture (bank seeds +
        # defensive prior draws: it covers every branch). Random-walk steps
        # never cross between separated branches, so after resampling a
        # branch's weight was frozen at whatever resampling noise gave it --
        # GW200115's branch weights changed from seed to seed. A proposal
        # drawn independently of the current point lets particles move
        # between branches in proportion to their target mass (exact MH).
        gm_i = ind["gm"]
        for _ in range(n_ind):
            xi = np.asarray(gm_i.sample(len(x))[0])[rng.permutation(len(x))]
            thp = from_slow(xi, tc_pool, ph_pool, rng)
            thp[:, _J["geocent_time"]] = th[:, _J["geocent_time"]]
            thp[:, _J["phase"]] = th[:, _J["phase"]]
            xp = to_slow(thp)
            fin = np.isfinite(xp).all(axis=1)
            lqp = np.full(len(xp), -np.inf)
            g0p = np.full(len(xp), -np.inf)
            if fin.any():
                lqp[fin] = gm_logpdf_sym(gm_i, xp[fin])
                g0p[fin] = anchor_logpdf(xp[fin])
            lqc = gm_logpdf_sym(gm_i, x)
            lpp = prior.log_prob(thp)
            t1p = np.full(len(thp), -np.inf)
            ok = np.isfinite(lpp) & fin
            if ok.any():
                t1p[ok] = eval_ll(thp[ok]) + lpp[ok] + log_jac(thp[ok])
            log_a = (1 - beta) * (g0p - log_g0) + beta * (t1p - log_t1) - (lqp - lqc)
            acc = np.log(rng.uniform(size=len(x))) < np.where(np.isfinite(log_a), log_a, -np.inf)
            th = np.where(acc[:, None], thp, th)
            x = np.where(acc[:, None], xp, x)
            log_t1 = np.where(acc, t1p, log_t1)
            log_g0 = np.where(acc, g0p, log_g0)
            ind_acc.append(float(acc.mean()))
        # Sky-mode jumps (ahsd.inference.skymoves): reflect the sky through
        # the detector plane (or rotate it about the baseline for two
        # detectors), with (psi, d_L) drawn to match the detector responses.
        # Exact MH: forward/reverse proposal densities of (psi, d_L) in the
        # sampling coordinates and the sky map's Jacobian.
        for _ in range(n_sky if sky.ok else 0):
            thp, lq_f, log_j = sky.propose(th, rng)
            lq_r = sky.log_q(thp, th)
            xp = to_slow(thp)
            fin = np.isfinite(xp).all(axis=1)
            g0p = np.full(len(xp), -np.inf)
            if fin.any():
                g0p[fin] = anchor_logpdf(xp[fin])
            lpp = prior.log_prob(thp)
            t1p = np.full(len(thp), -np.inf)
            ok = np.isfinite(lpp) & fin
            if ok.any():
                t1p[ok] = eval_ll(thp[ok]) + lpp[ok] + log_jac(thp[ok])
            with np.errstate(divide="ignore", invalid="ignore"):
                log_a = ((1 - beta) * (g0p - log_g0) + beta * (t1p - log_t1)
                         + (lq_r + log_dd_dx(th)) - (lq_f + log_dd_dx(thp)) + log_j)
            acc = np.log(rng.uniform(size=len(x))) < np.where(np.isfinite(log_a), log_a, -np.inf)
            th = np.where(acc[:, None], thp, th)
            x = np.where(acc[:, None], xp, x)
            log_t1 = np.where(acc, t1p, log_t1)
            log_g0 = np.where(acc, g0p, log_g0)
            sky_acc.append(float(acc.mean()))
        # Mass-ratio / spin ridge moves (ahsd.inference.ridgemoves): step eta
        # at fixed chirp mass with chi1 set to keep the 1PN + 1.5PN phase,
        # i.e. along the q - chi1 degeneracy a random walk only crosses.
        # Distance / inclination ridge moves: step cos(theta_jn) at fixed
        # dominant-mode amplitude _amp(cos theta_jn) / d_L.
        for _, mv in [(i, m) for i in range(n_ridge) for m in ridges]:
            thp, corr = mv.propose(th, rng)
            xp = to_slow(thp)
            fin = np.isfinite(xp).all(axis=1) & np.isfinite(corr)
            g0p = np.full(len(xp), -np.inf)
            if fin.any():
                g0p[fin] = anchor_logpdf(xp[fin])
            lpp = prior.log_prob(thp)
            t1p = np.full(len(thp), -np.inf)
            ok = np.isfinite(lpp) & fin
            if ok.any():
                t1p[ok] = eval_ll(thp[ok]) + lpp[ok] + log_jac(thp[ok])
            with np.errstate(invalid="ignore"):
                log_a = ((1 - beta) * (g0p - log_g0) + beta * (t1p - log_t1)
                         - (log_jac(thp) - log_jac(th)) + corr)
            acc = np.log(rng.uniform(size=len(x))) < np.where(np.isfinite(log_a), log_a, -np.inf)
            th = np.where(acc[:, None], thp, th)
            x = np.where(acc[:, None], xp, x)
            log_t1 = np.where(acc, t1p, log_t1)
            log_g0 = np.where(acc, g0p, log_g0)
            ridge_acc.append(float(acc.mean()))
        return th, log_t1, log_g0, scale, float(np.mean(acc_all))

    def sky_moves_beta1(th, log_t1):
        """One sky-jump MH step targeting L * pi (beta = 1)."""
        thp, lq_f, log_j = sky.propose(th, rng)
        lq_r = sky.log_q(thp, th)
        lpp = prior.log_prob(thp)
        t1p = np.full(len(thp), -np.inf)
        ok = np.isfinite(lpp) & np.isfinite(to_slow(thp)).all(axis=1)
        if ok.any():
            t1p[ok] = eval_ll(thp[ok]) + lpp[ok] + log_jac(thp[ok])
        with np.errstate(divide="ignore", invalid="ignore"):
            log_a = (t1p - log_t1) + (lq_r + log_dd_dx(th)) - (lq_f + log_dd_dx(thp)) + log_j
        acc = np.log(rng.uniform(size=len(th))) < np.where(np.isfinite(log_a), log_a, -np.inf)
        sky_acc_final.append(float(acc.mean()))
        return np.where(acc[:, None], thp, th), np.where(acc, t1p, log_t1)

    def ridge_moves_beta1(th, log_t1, mv):
        """One ridge MH step (move `mv`) targeting L * pi (beta = 1)."""
        thp, corr = mv.propose(th, rng)
        lpp = prior.log_prob(thp)
        t1p = np.full(len(thp), -np.inf)
        ok = np.isfinite(lpp) & np.isfinite(corr) & np.isfinite(to_slow(thp)).all(axis=1)
        if ok.any():
            t1p[ok] = eval_ll(thp[ok]) + lpp[ok] + log_jac(thp[ok])
        with np.errstate(invalid="ignore"):
            log_a = (t1p - log_t1) - (log_jac(thp) - log_jac(th)) + corr
        acc = np.log(rng.uniform(size=len(th))) < np.where(np.isfinite(log_a), log_a, -np.inf)
        ridge_acc_final.append(float(acc.mean()))
        return np.where(acc[:, None], thp, th), np.where(acc, t1p, log_t1)

    sky_acc_final, ridge_acc_final = [], []
    cur_theta, cur_log_g = theta, log_q + log_jac(theta)   # flow density, in x
    # default: seed from the bank whenever it has several (q, spin) slots
    # (GW200115 mode recall vs dynesty 0.25-0.75 -> 0.75-1.0 with NSBH v5ff)
    if seed_bank is None:
        seed_bank = getattr(getattr(model.encoder, "mc_feat", None), "n_q", 1) > 1
    seeds = _bank_seeds(model, result, theta, rng) if seed_bank else None
    if defensive > 0:
        # Defensive importance sampling: a share of PRIOR draws in the initial
        # mixture. Where the likelihood is weak (masses on a quiet event, the
        # mass ratio) the corrected posterior must approach the prior, but
        # weights cannot create particles where the network posterior has
        # none: with a flat likelihood the m1 median stayed at 11 against the
        # prior's 22 (scripts/check_correction_prior.py). The mixture density
        # is known, so the weights stay exact.
        k = int(round(len(theta) * defensive / max(1e-9, 1 - defensive)))
        pdraw = prior.sample(k, rng)
        seeds = pdraw if seeds is None else np.concatenate([seeds, pdraw])
    if sky_seeds and sky.ok:
        # mirror-image copies of the network samples (the other sky island of
        # each, with matched psi and distance), so an island the network
        # missed is populated from the first stage and covered by the anchor
        # (reflections and rotations about every baseline: the network's sky
        # may sit on a cycle-slip island, not on a mirror of the true one)
        sub = theta[rng.choice(len(theta), size=len(theta), replace=True)]
        mir = sky.propose(sub, rng)[0]
        mir = mir[np.isfinite(prior.log_prob(mir))]
        seeds = mir if seeds is None else np.concatenate([seeds, mir])
    if seeds is not None and len(seeds):
        # initial particles drawn from a mixture fitted to NPE samples plus
        # bank-seeded copies (each plausible (q, spin) slot of the matched
        # filter): a secondary branch the NPE smears over gets particles from
        # the start. The mixture density is known, so the weights stay exact.
        pool_x = to_slow(np.concatenate([theta, seeds]))
        gms = GaussianMixture(n_components=min(24, 8 + len(seeds) // 200), covariance_type="full",
                              reg_covar=1e-5, random_state=seed).fit(pool_x)
        # Tempering ANCHOR and initial draws: the mixture with every
        # covariance inflated by anchor_inflate. With the fitted mixture
        # itself the intermediate targets g0^(1-b) (L pi)^b were dominated by
        # wherever the network put its mass (log-density contrast ~10 nats
        # between its blob and a sky island or ridge segment it missed), so
        # such regions were populated too late to be weighted correctly:
        # seeds disagreed on an edge-on injection's sky islands (main island
        # 16% / 83% / 15%) and GW200115's m1-q-spin ridge collapsed onto one
        # segment. Inflating by k divides that contrast by k.
        gm0 = _inflated(gms, anchor_inflate) if anchor_inflate > 1 else gms
    if anchor_sky_flat and sky.ok:
        # ... and isotropic on the sky (coords.SkyFlatMixture): the network's
        # sky island is not favoured over the others it may have missed
        from ahsd.inference.coords import SkyFlatMixture
        anchor["g"] = SkyFlatMixture(gm0, C, flat_spins=anchor_flat_spins)
        anchor_logpdf = anchor["g"].logpdf
    else:
        anchor_logpdf = lambda x: gm_logpdf_sym(gm0, x)   # noqa: E731
    if (seeds is not None and len(seeds)) or anchor["g"] is not None:
        xn = anchor["g"].sample(n, rng) if anchor["g"] is not None else np.asarray(gm0.sample(n)[0])
        cur_theta = from_slow(np.asarray(xn), tc_pool, ph_pool, rng)
        cur_log_g = anchor_logpdf(to_slow(cur_theta))
        log_pi = prior.log_prob(cur_theta)
        # The broad mixture is also the tempering ANCHOR. The path
        # t_b = g0^(1-b) (L pi)^b with g0 fitted to the network samples alone
        # inherits their razor-thin tails: where the network puts no mass
        # (high masses on a quiet event) g0^(1-b) suppresses the target until
        # b ~ 1, and the last step cannot repopulate it. More Metropolis moves
        # made it worse (flat-likelihood check: m1 median 14.6 -> 11.4 vs the
        # prior's 22.2) because they followed those excluding targets more
        # faithfully. With the broad anchor no intermediate target excludes
        # any part of the prior. (gm0 is the inflated mixture set above.)
    # the independence moves draw from the broadest available mixture
    ind["gm"] = gms if (seeds is not None and len(seeds)) else gm0
    cur_log_g0 = anchor_logpdf(to_slow(cur_theta))
    log_l = eval_ll(cur_theta)
    if not np.isfinite(log_l).any():
        pool.terminate()
        raise RuntimeError(f"likelihood is -inf for every sample: the {approximant} "
                           "waveform failed (model not available in this LAL installation?)")
    log_t1 = log_l + log_pi + log_jac(cur_theta)   # log(L * pi), in x
    beta_prev, betas, accs, trace = 0.0, [], [], []
    lw_carry = np.zeros(n)               # log weights carried through resampling
    scale = 2.38 / np.sqrt(len(SLOW)) * 0.5
    if start_particles is not None:
        # diagnostic: skip the tempering and start the beta = 1 moves from
        # given particles (e.g. a reference posterior): any drift of their
        # distribution under the moves exposes a kernel that does not leave
        # the posterior invariant
        cur_theta = np.asarray(start_particles, float)
        log_t1 = eval_ll(cur_theta) + prior.log_prob(cur_theta) + log_jac(cur_theta)
        cur_log_g0 = anchor_logpdf(to_slow(cur_theta))
        n = len(cur_theta)
        w = np.full(n, 1.0 / n)
        beta, betas = 1.0, [1.0]
    else:
        for stage in range(MAX_STAGES):
            beta = pick_beta(log_t1, cur_log_g0, cur_log_g, beta_prev, target_frac, lw_carry)
            betas.append(round(beta, 4))
            log_w = lw_carry + (1 - beta) * cur_log_g0 + beta * log_t1 - cur_log_g
            fin = np.isfinite(log_w)
            log_w = np.where(fin, log_w, -np.inf)
            w = np.exp(log_w - log_w[fin].max())
            w = np.where(np.isfinite(w), w, 0.0)
            if w.sum() == 0:
                pool.terminate()
                raise RuntimeError("importance weights vanished at "
                                   f"beta={beta:.3f} — no likelihood support")
            w /= w.sum()
            trace.append(_stage_summary(beta, cur_theta, w, branches(to_slow(cur_theta))))
            if beta >= 1.0:
                break
            # SMC step: resample by the tempered weights, then Metropolis moves
            # at this beta. The moved particles follow t_beta, so the next
            # stage's proposal density is t_beta itself (unnormalized: the
            # weights are self-normalized), and pick_beta / the weights above
            # stay exact.
            idx_r, lw_carry = resample(w, branches(to_slow(cur_theta)))
            cur_theta, log_t1, cur_log_g0, scale, acc = mh_moves(
                cur_theta[idx_r], log_t1[idx_r], cur_log_g0[idx_r], beta, scale)
            accs.append(round(acc, 3))
            cur_log_g = (1 - beta) * cur_log_g0 + beta * log_t1
            beta_prev = beta
    if beta < 1.0:
        pool.terminate()
        raise RuntimeError(f"tempering did not reach beta=1 within {MAX_STAGES} "
                           f"stages (ladder: {betas}) — proposal cannot bridge "
                           "this likelihood; treat NPE result as final")
    ess = float(1.0 / np.sum(w ** 2))
    # Final resampling, then REJUVENATION: Metropolis moves at beta = 1 leave
    # the posterior invariant and spread the resampled copies out again.
    # Without them the returned samples stood on a handful of distinct
    # values in tightly coupled directions: on an edge-on injection the
    # median distance moved by 80-190 Mpc from seed to seed (dynesty's sigma
    # ~65 Mpc), whatever the proposal settings.
    u = (rng.uniform() + np.arange(n)) / n
    idx0 = np.searchsorted(np.cumsum(w), u).clip(0, len(w) - 1)
    rj_theta, rj_t1 = cur_theta[idx0], log_t1[idx0]
    rejuv_trace = []
    for _ in range(max(1, n_rejuvenate)):
        rj_theta, rj_t1, _g0, scale, acc = mh_moves(
            rj_theta, rj_t1, cur_log_g0[idx0], 1.0, scale)
        rejuv_trace.append(_stage_summary(1.0, rj_theta, np.full(n, 1.0 / n), branches(to_slow(rj_theta))))
    # Island balancing: sky jumps alone at beta = 1 (one likelihood call
    # each). They move particles between sky islands in proportion to the
    # islands' posterior mass (exact MH), which the tempering cannot do once
    # an island was found late.
    for _ in range(n_sky_final if sky.ok else 0):
        rj_theta, rj_t1 = sky_moves_beta1(rj_theta, rj_t1)
    for _ in range(n_ridge_final):
        for mv in ridges:
            rj_theta, rj_t1 = ridge_moves_beta1(rj_theta, rj_t1, mv)
    accs.append(round(acc, 3))
    pool.terminate()
    theta, log_t1 = rj_theta, rj_t1
    idx = np.arange(n)                   # the rejuvenated set IS the sample

    diag = {**result.diagnostics,
            "importance_sampling": {
                "n_proposal": int(len(result.samples)),
                "n_evaluated": int(n),
                "railed_excluded": int((~keep).sum()),
                "importance_ess": round(ess, 1),
                "beta_ladder": betas,
                # per stage: weighted quantiles of distance / sky and the
                # weight of each slow-space branch (where a mode is lost)
                "stage_trace": trace,
                "mh_acceptance": accs,
                "independence_acceptance": round(float(np.mean(ind_acc)), 3) if ind_acc else None,
                "sky_jump_acceptance": round(float(np.mean(sky_acc)), 4) if sky_acc else None,
                "sky_jump_acceptance_final": round(float(np.mean(sky_acc_final)), 4) if sky_acc_final else None,
                "ridge_acceptance": round(float(np.mean(ridge_acc)), 4) if ridge_acc else None,
                "ridge_acceptance_final": round(float(np.mean(ridge_acc_final)), 4) if ridge_acc_final else None,
                # distinct particles among the returned samples: low values
                # mean the moves did not spread the resampled particles
                "unique_frac": round(float(len(np.unique(theta[idx][:, 0]))) / n, 3),
                "rejuv_trace": rejuv_trace,
                "n_stages": len(betas),
                "efficiency": round(ess / n, 4),
                "wall_s": round(time.time() - t0, 1),
                "approximant": approximant,
                "f_min": round(f_min, 1),
                "band_cut_frac": round(band_cut, 3),
                "reliable": band_cut <= 0.1,
                "caveats": caveats,
                "explicit_phase": bool(explicit_phase),
                "note": ("geocent_time is analytically marginalized in the weights; "
                         "its column retains PROPOSAL values (use the NPE result for "
                         "timing). Phase is sampled explicitly (higher-mode waveform)."
                         if explicit_phase else
                         "phase and geocent_time are analytically marginalized "
                         "in the weights; their columns in the corrected "
                         "samples retain PROPOSAL values (use the NPE result "
                         "for timing)."),
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


def corrected_with_check(result, raw_strain, reps: int = 2, tol: float = 0.5, max_reps: int = 6,
                         max_spread: float = 1.0, **kw):
    """importance_correct with `reps` independent seeds; when they disagree
    (replicate spread >= tol, typical of multimodal posteriors whose branch
    weights vary from run to run), more seeds are run, up to `max_reps`, and
    all runs are POOLED: each run is a consistent estimate of the posterior,
    so pooling k of them reduces the run-to-run error by sqrt(k). The result
    is reliable when that pooled error, spread / sqrt(k), is below tol;
    otherwise it is flagged for nested sampling. Returns the (pooled)
    PosteriorResult; diagnostics["importance_sampling"] gains
    replicate_spread, reps_used, pooled_spread, consistent and reliable.
    kw go to PosteriorResult.importance_correct (npool, device, ...)."""
    import copy
    runs = [result.importance_correct(raw_strain, seed=k, **kw) for k in range(max(1, reps))]
    d = runs[0].diagnostics["importance_sampling"]
    if len(runs) < 2:
        return runs[0]
    spread = replicate_spread(runs, result.param_names)
    while spread / np.sqrt(len(runs)) >= tol and len(runs) < max_reps:
        runs.append(result.importance_correct(raw_strain, seed=len(runs), **kw))
        spread = replicate_spread(runs, result.param_names)
    pooled_spread = spread / np.sqrt(len(runs))
    # pool every run (equal weight: equal sample counts). Consistent runs are
    # pooled too: each is a valid estimate, so discarding one wasted half the
    # likelihood evaluations
    out = copy.copy(runs[0])
    out.samples = np.concatenate([r.samples for r in runs])
    out.log_prob = np.concatenate([r.log_prob for r in runs])
    out.diagnostics = copy.deepcopy(runs[0].diagnostics)
    d = out.diagnostics["importance_sampling"]
    d["importance_ess"] = float(sum(r.diagnostics["importance_sampling"]["importance_ess"] for r in runs))
    d["replicate_spread"] = round(spread, 3)
    d["reps_used"] = len(runs)
    d["pooled_spread"] = round(float(pooled_spread), 3)
    # Seeds more than ~one 90% width apart sat in different posterior modes:
    # pooling them averages mode weights that are each run's accident, so the
    # pooled spread is no stability measure (GW190513 with H1 missing: spread
    # 1.04, pooled 0.43 -> "consistent", q 0.06 against LVK's 0.48)
    d["consistent"] = bool(pooled_spread < tol and spread <= max_spread)
    # The correction refines the network's posterior; it must not move the
    # chirp mass (the network's best-measured parameter) outside the network's
    # own support. GW190513 with LVK PSDs: network 90% [22.4, 35.7], every
    # seed agreed on 7.0 (LVK 29.9) -- a consistent but wrong mode, which the
    # replicate test cannot see.
    def mc(smp):
        with np.errstate(invalid="ignore"):
            v = (smp[:, 0] * smp[:, 1]) ** 0.6 / (smp[:, 0] + smp[:, 1]) ** 0.2
        return v[np.isfinite(v)]
    inside = None
    prop = getattr(result, "samples", None)
    if prop is not None and len(mc(prop)) and len(mc(out.samples)):
        lo, hi = np.quantile(mc(prop), [0.005, 0.995])
        inside = bool(lo <= float(np.median(mc(out.samples))) <= hi)
    d["mc_inside_proposal"] = inside
    d["reliable"] = bool(d.get("reliable", True) and d["consistent"] and inside is not False)
    return out
