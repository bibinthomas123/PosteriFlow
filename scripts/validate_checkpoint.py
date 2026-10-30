#!/usr/bin/env python3
"""
CI validation suite: everything a LeanNPE checkpoint must pass, in one command,
out as one self-contained HTML report with PASS/FAIL gates.

    python scripts/validate_checkpoint.py --model model/lean_npe_v3/best_model.pth

Runs on the fixed validation split (Gaussian domain) and, when a noise bank is
present, the deterministic real-noise variant of the same events:
  - NLL + context-shuffle delta (conditional-vs-marginal)
  - credible-interval coverage at 50/68/90/95% for all 11 parameters
  - SBC ranks + KS tests, PP curves
  - boundary-railing statistics
Then real-data smoke tests: 6 GWTC events through the production pipeline
(timing must land on each published GPS; GW170729 at dL~2750 Mpc lies outside
the training prior and live-tests the OOD/refinement gate).
Also fits <checkpoint_dir>/ood_stats.npz if missing, so the OOD detector is
armed for every validated checkpoint.

Exit code 0 = all gates pass, 1 = at least one FAIL (CI-friendly).
"""
import logging
logging.getLogger("bilby").setLevel(logging.WARNING)  # waveform-generator INFO spam
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _ROOT)
sys.path.insert(0, f"{_ROOT}/src")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES

BLUE, GRAY, INK, RED = "#3b6fd4", "#9aa4b2", "#3a3f47", "#c65a49"
CI_LEVELS = [0.50, 0.68, 0.90, 0.95]
plt.rcParams.update({"axes.edgecolor": GRAY, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
                     "grid.color": "#e8ebef", "grid.linewidth": 0.6, "font.size": 9})


def fig_to_b64(fig) -> str:
    """Encode a matplotlib figure as a base64 PNG data URI for embedding in the
    HTML report."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# Parameters whose natural scale is multiplicative — width is measured in log.
_LOG_PARAMS = {"mass_1", "mass_2", "luminosity_distance", "chirp_mass"}
# Periodic parameters and their period — measured with a circular statistic, not
# a linear std (see sharpness_metrics). Matches ParamScaler.CIRCULAR.
_CIRCULAR = {"ra": 2 * np.pi, "phase": 2 * np.pi, "psi": np.pi}
_SNR_BINS = ((0, 12), (12, 20), (20, 35), (35, 60), (60, 10000))

# Informativeness (posterior std / prior std) ACHIEVED BY DYNESTY on identical
# strain — the physical optimum, not a guess. Measured 2026-07-21 over 12
# simulated events, SNR ladder 12/18/25/40, Mc 6-40, q 0.18-0.91:
# analysis/posterior_bias_audit/eH_dynesty/summary.json. dynesty's median 90%
# widths (Mc 0.31% fractional, q 0.169) converted to std via the Gaussian factor
# 3.2897 and divided by the validation-split prior stds (log Mc 1.1447, q 0.3082).
#
# This replaces the hand-set thresholds these gates shipped with. The basin
# report made exactly this criticism of its own gate ("set by eyeballing
# collapsed-vs-healthy runs, not by physics"); anchoring to a measured reference
# is the fix. When the reference is re-measured (e.g. with realistic spins),
# update here and the gates move with it.
#
# SPIN SENSITIVITY (measured 2026-07-21, eH_spin/, n=4 with a1,a2 ~ U[0,0.9]):
# the 12-event sweep above used ZERO-SPIN injections while the training prior
# draws spins. Controlling for mass (Mc<13), dynesty's achievable q width nearly
# triples with realistic spins, 0.120 -> 0.292 — the q-chi_eff degeneracy is real
# — while its Mc width does NOT loosen. So `mass_ratio` below uses the SPIN-
# INCLUSIVE optimum (0.292/3.2897/0.3082 = 0.288); holding the model to the
# zero-spin 0.167 would demand better-than-achievable on the actual training
# distribution. `chirp_mass` keeps the 12-event value, which is the LOOSER of the
# two measurements (the spin draw's low-mass events gave a tighter 0.0012).
# Both choices are deliberately conservative. The q reference rests on only 4
# events — re-measure with ~12 spinning events before treating it as settled.
_DYNESTY_REF = {"chirp_mass": 0.00276, "mass_ratio": 0.2880}
# An amortized estimator within this factor of a reference sampler's width is
# doing well (DINGO reports near-parity). Generous, and still fails the current
# generation by ~52x on Mc and ~1.8x on q.
_SHARPNESS_TOL = 3.0


def _mc_q(m1, m2):
    """Chirp mass and mass ratio from component masses (any array shape).

    Floors both components at a positive value first: the fractional powers are
    NaN for non-positive input, and a single NaN would silently poison a whole
    informativeness statistic (and, worse, slip past a `v > threshold` gate,
    since NaN compares False against everything)."""
    m1 = np.maximum(np.asarray(m1, dtype=float), 1e-9)
    m2 = np.maximum(np.asarray(m2, dtype=float), 1e-9)
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2, np.clip(m2 / m1, 1e-6, 1.0)


def sharpness_metrics(samples, truth, ctx, snr):
    """Diagnostics that measure how INFORMATIVE the posterior is.

    Coverage, SBC and PP are all invariant to posterior WIDTH: an estimator that
    returns the prior for a parameter passes every one of them exactly. The
    2026-07-21 audit (analysis/posterior_bias_audit/REPORT.md) measured a model
    with textbook-clean calibration (90% coverage 0.901) whose q posterior was
    89% as wide as the prior and whose Mc was ~20x wider than a dynesty reference
    on the same strain. Nothing in this suite could see that. These three
    statistics can:

      informativeness  mean posterior std / prior std, per parameter. ~1 means
                       the parameter is UNMEASURED (posterior == prior); ->0 means
                       sharply measured. Reported for the 11 sampled parameters
                       plus derived (chirp_mass, mass_ratio), which is where the
                       mass information actually lives.

      snr_scaling      sigma_theta * SNR across SNR bins. A matched-filter-limited
                       estimator holds this ~constant (sigma ~ 1/SNR). If it RISES
                       with SNR the encoder is saturating — extracting a fixed
                       amount of information no matter how much the data carries.
                       Reported as `ratio` = (last bin)/(first bin); ~1 is healthy.

      flow_vs_encoder  mean posterior std / residual of a ridge probe ctx->truth.
                       ~1 means the flow faithfully reports the information in its
                       conditioning vector, so posterior width is set by the
                       ENCODER and a bigger flow cannot help. >>1 would mean the
                       flow is broadening beyond what the context supports.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler

    out = {"informativeness": {}, "snr_scaling": {}, "flow_vs_encoder": {}}

    # per-parameter series: (name, posterior samples [N,K], truth [N])
    series = [(nm, samples[:, :, j], truth[:, j]) for j, nm in enumerate(PARAM_NAMES)]
    mc_s, q_s = _mc_q(samples[:, :, 0], samples[:, :, 1])
    mc_t, q_t = _mc_q(truth[:, 0], truth[:, 1])
    series += [("chirp_mass", mc_s, mc_t), ("mass_ratio", q_s, q_t)]

    ctx_np = ctx.detach().cpu().numpy() if hasattr(ctx, "detach") else np.asarray(ctx)
    Xs = StandardScaler().fit_transform(ctx_np)

    for nm, s, t in series:
        if nm in _CIRCULAR:
            # A plain std ratio is WRONG for a periodic parameter: a sharp but
            # bimodal sky/polarization posterior (the physically expected shape —
            # e.g. the antipodal sky degeneracy) has a large linear std while
            # being highly informative, so it would be mislabelled "unmeasured".
            # Use the mean resultant length R instead: R->1 concentrated, R->0
            # uniform. Report 1-R so the scale matches the linear case (1.0 =
            # uninformative), and flag it as a different statistic.
            per = _CIRCULAR[nm]
            R = np.abs(np.exp(2j * np.pi * s / per).mean(axis=1))   # [N]
            out["informativeness"][nm] = float(1.0 - R.mean())
            out.setdefault("circular_params", []).append(nm)
            continue
        if nm in _LOG_PARAMS:
            s, t = np.log(np.maximum(s, 1e-9)), np.log(np.maximum(t, 1e-9))
        post_sd = s.std(axis=1)                      # [N]
        prior_sd = float(t.std())
        if not np.isfinite(prior_sd) or prior_sd <= 0:
            continue
        out["informativeness"][nm] = float(post_sd.mean() / prior_sd)

        # encoder-vs-flow attribution (ridge is deliberately weak: it lower-bounds
        # the context's information, so the ratio is an UPPER bound on flow slack)
        try:
            pred = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 4, 20)), Xs, t, cv=5)
            res = float(np.std(t - pred))
            if res > 1e-9:
                out["flow_vs_encoder"][nm] = float(post_sd.mean() / res)
        except Exception:
            pass

        if nm in ("chirp_mass", "luminosity_distance", "mass_ratio"):
            bins = []
            for lo, hi in _SNR_BINS:
                m = (snr >= lo) & (snr < hi)
                if m.sum() < 15:
                    continue
                bins.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                             "med_snr": float(np.median(snr[m])),
                             "sigma": float(post_sd[m].mean()),
                             "sigma_x_snr": float(post_sd[m].mean() * np.median(snr[m]))})
            if len(bins) >= 2:
                out["snr_scaling"][nm] = {
                    "bins": bins,
                    "ratio": float(bins[-1]["sigma_x_snr"] / max(bins[0]["sigma_x_snr"], 1e-9))}
    return out


def sharpness_figure(sh, tag):
    """Two-panel figure: informativeness bars + sigma*SNR trend."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.8))
    names = list(sh["informativeness"].keys())
    vals = [sh["informativeness"][n] for n in names]
    cols = [BLUE if v < 0.5 else GRAY for v in vals]
    a1.bar(np.arange(len(names)), vals, color=cols, width=0.62)
    a1.axhline(1.0, color=INK, lw=1, ls="--")
    a1.axhline(0.5, color=INK, lw=0.8, ls=":")
    a1.set_xticks(np.arange(len(names)))
    a1.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    a1.set_ylabel("posterior std / prior std")
    a1.set_ylim(0, 1.15)
    a1.set_title("informativeness (1.0 = parameter unmeasured)")
    for nm, d in sh["snr_scaling"].items():
        xs = [b["med_snr"] for b in d["bins"]]
        ys = [b["sigma_x_snr"] for b in d["bins"]]
        a2.plot(xs, np.array(ys) / max(ys[0], 1e-9), marker="o", lw=1.4, label=nm)
    a2.axhline(1.0, color=INK, lw=1, ls="--")
    a2.set_xscale("log")
    a2.set_xlabel("network SNR")
    a2.set_ylabel("σ×SNR  (normalized to first bin)")
    a2.set_title("SNR scaling (flat = matched-filter-limited)")
    a2.legend(fontsize=7)
    fig.suptitle(f"{tag}: posterior sharpness")
    fig.tight_layout()
    return fig_to_b64(fig)


def evaluate_domain(model, ds, n_events, n_post, device, tag):
    """All sample-based metrics + figures for one data domain."""
    n = min(n_events, len(ds))
    items = [ds[i] for i in range(n)]
    strain = torch.stack([x[0] for x in items])
    theta = torch.stack([x[1][0] for x in items]).to(device)
    # PSD-conditioned models: use the domain's per-event sensitivity summary
    # (nonzero for the real-noise domain, ~0 for Gaussian/design).
    asd = torch.stack([x[4] for x in items]) if len(items[0]) > 4 else None
    # net SNR per event — needed by the sharpness diagnostics to resolve posterior
    # width against the information actually present in the data.
    snr = torch.stack([x[3] for x in items]).numpy().astype(float)
    N, P = n, len(PARAM_NAMES)

    ctx = []
    with torch.no_grad():
        for i in range(0, N, 64):
            ab = asd[i:i + 64].to(device).float() if asd is not None else None
            ctx.append(model.encode(strain[i:i + 64].to(device).float(), ab))
    ctx = torch.cat(ctx)
    rank0 = torch.zeros(N, dtype=torch.long, device=device)
    with torch.no_grad():
        nll_t = model.nll(None, theta, rank0, context=ctx)
        nll_s = model.nll(None, theta, rank0, context=ctx[torch.randperm(N, device=device)])

    K = n_post
    samples = np.empty((N, K, P), dtype=np.float32)
    railed = np.zeros((N, K), dtype=bool)
    full_ctx = model._full_context(ctx, rank0)
    with torch.no_grad():
        for i in range(0, N, 32):
            fc = full_ctx[i:i + 32]
            nb = fc.shape[0]
            fcr = fc.unsqueeze(1).expand(nb, K, fc.shape[1]).reshape(nb * K, -1)
            z = torch.randn(nb * K, P, device=device)
            y, _ = model.flow.inverse(z, fcr)
            y = model.scaler.wrap(y)  # circular wrap; bounded clamp
            # SPURIOUS railing only: a sample pinned at a bound in a dim whose
            # TRUE value is not near that bound. Bound pile-up where the truth
            # actually sits at the edge (spins ~0, face-on inclination) is the
            # posterior doing its job, not miscalibration.
            y_true = model.scaler.normalize(theta[i:i + 32]).unsqueeze(1)  # [nb,1,P]
            yr = y.reshape(nb, K, P)
            spurious = ((yr > 0.999) & (y_true < 0.9) |
                        (yr < -0.999) & (y_true > -0.9)) & ~model.scaler.circ_mask
            railed[i:i + 32] = spurious.any(dim=2).cpu().numpy()
            samples[i:i + 32] = model.scaler.denormalize(y).reshape(nb, K, P).cpu().numpy()

    truth = theta.cpu().numpy()
    coverage = {name: {} for name in PARAM_NAMES}
    for j, name in enumerate(PARAM_NAMES):
        for lev in CI_LEVELS:
            lo = np.quantile(samples[:, :, j], 0.5 - lev / 2, axis=1)
            hi = np.quantile(samples[:, :, j], 0.5 + lev / 2, axis=1)
            coverage[name][f"{int(lev*100)}"] = float(
                ((truth[:, j] >= lo) & (truth[:, j] <= hi)).mean())
    # SBC ranks: plain linear ranks are a valid SBC statistic for every
    # parameter (rank uniformity holds for any fixed ordering) — circular
    # parameters are fine now that sampling WRAPS instead of clamping at the
    # period seam (the old edge pile-up was a sampling artifact, not a rank-
    # statistic problem)
    ranks = (samples < truth[:, None, :]).mean(axis=1)
    ks = {name: sps.kstest(ranks[:, j], "uniform") for j, name in enumerate(PARAM_NAMES)}

    jd = PARAM_NAMES.index("luminosity_distance")
    d_med = np.median(samples[:, :, jd], axis=1)
    metrics = {
        "n_events": N, "n_post": K,
        "nll": float(nll_t.mean()),
        "shuffle_delta_nll": float(nll_s.mean() - nll_t.mean()),
        "dist_corr": float(np.corrcoef(np.log(d_med), np.log(truth[:, jd]))[0, 1]),
        "distance_by_type": distance_by_type(samples[:, :, jd], truth, model),
        "coverage": coverage,
        "sbc_ks_p": {k: float(v.pvalue) for k, v in ks.items()},
        "railing_frac": float(railed.mean()),
        # Width diagnostics — the only metrics here that are NOT invariant to
        # posterior sharpness (see sharpness_metrics).
        "sharpness": sharpness_metrics(samples, truth, ctx, snr),
    }

    figs = {}
    # coverage bars
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    xs = np.arange(P)
    for a, lev in zip(axes, CI_LEVELS):
        a.bar(xs, [coverage[nm][f"{int(lev*100)}"] for nm in PARAM_NAMES],
              color=BLUE, width=0.62)
        a.axhline(lev, color=INK, lw=1, ls="--")
        a.set_title(f"{int(lev*100)}% CI")
        a.set_xticks(xs)
        a.set_xticklabels(PARAM_NAMES, rotation=60, ha="right", fontsize=7)
        a.set_ylim(0, 1)
    fig.suptitle(f"{tag}: coverage ({N} events)")
    figs["coverage"] = fig_to_b64(fig)
    # SBC ranks
    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    for j, name in enumerate(PARAM_NAMES):
        a = axes.flat[j]
        a.hist(ranks[:, j], bins=20, color=BLUE, edgecolor="white", linewidth=0.4)
        a.axhline(N / 20, color=INK, lw=1, ls="--")
        a.set_title(f"{name} (KS p={ks[name].pvalue:.3f})", fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle(f"{tag}: SBC ranks")
    figs["sbc"] = fig_to_b64(fig)
    # PP curves
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    levs = np.linspace(0.02, 0.98, 40)
    for j, name in enumerate(PARAM_NAMES):
        emp = [np.mean(np.abs(ranks[:, j] - 0.5) <= l / 2) for l in levs]
        hl = name == "luminosity_distance"
        ax.plot(levs, emp, lw=1.3, color=BLUE if hl else GRAY, alpha=1 if hl else 0.5)
    ax.plot([0, 1], [0, 1], color=INK, lw=1, ls="--")
    ax.set_xlabel("nominal")
    ax.set_ylabel("empirical")
    ax.set_title(f"{tag}: PP calibration")
    figs["pp"] = fig_to_b64(fig)
    figs["sharpness"] = sharpness_figure(metrics["sharpness"], tag)
    return metrics, figs


def gate(name, ok, detail):
    """Build one pass/fail gate record {name, pass, detail}."""
    return {"gate": name, "pass": bool(ok), "detail": detail}


def distance_by_type(d_samp, truth, model):
    """Distance calibration per source class, conditioned on the PREDICTION.

    The pooled metrics (dist_corr, coverage) average BBH/NSBH/BNS together, and
    BBH dominates: on mf_scratch they read green while BNS distances were 2.3x
    too far with 65% cov90 (measured 2026-10-30). Conditioning on the truth
    instead (bias per true-distance bin) is NOT a calibration test — a correct
    posterior is pulled toward the prior there — so bias is measured as
    E[log d_true - log d_hat_median], which is 0 for a calibrated posterior.
    """
    d_true = truth[:, PARAM_NAMES.index("luminosity_distance")]
    m1, m2 = truth[:, 0], truth[:, 1]
    lo, hi = (float(v) for v in model.dist_range.tolist())
    out = {}
    for name, sel in (("BBH", m2 >= 3.0), ("NSBH", (m2 < 3.0) & (m1 >= 3.0)),
                      ("BNS", m1 < 3.0)):
        n = int(sel.sum())
        if n == 0:
            continue
        ds, dt = d_samp[sel], d_true[sel]
        rank = (ds < dt[:, None]).mean(axis=1)
        out[name] = {
            "n": n,
            "true_over_pred": float(np.exp(np.mean(np.log(dt) - np.log(np.median(ds, axis=1))))),
            "sbc_rank_mean": float(rank.mean()),
            "cov90": float(((rank > 0.05) & (rank < 0.95)).mean()),
            # truths the model cannot represent at all (clamped at its support)
            "outside_support_frac": float(((dt < lo) | (dt > hi)).mean()),
        }
    return out


def run_gates(gaus, real, gw_ok, gw_summary, basin=None):
    """Evaluate all CI acceptance gates over the Gaussian/real-domain metrics
    and the GWTC real-event results; returns the list of gate records."""
    gates = []
    gates.append(gate("gaussian shuffle ΔNLL > 5 nats",
                      gaus["shuffle_delta_nll"] > 5, f"{gaus['shuffle_delta_nll']:.2f}"))
    for lev, tol in (("50", 0.07), ("90", 0.05)):
        bad = [n for n in PARAM_NAMES
               if abs(gaus["coverage"][n][lev] - int(lev) / 100) > tol]
        gates.append(gate(f"gaussian {lev}% coverage within ±{tol} (≤2 exceptions)",
                          len(bad) <= 2, f"violations: {bad or 'none'}"))
    n_ks_ok = sum(p > 1e-3 for p in gaus["sbc_ks_p"].values())
    gates.append(gate("gaussian SBC KS p>1e-3 for ≥9/11 params",
                      n_ks_ok >= 9, f"{n_ks_ok}/11"))
    gates.append(gate("gaussian spurious railing < 5%",
                      gaus["railing_frac"] < 0.05, f"{gaus['railing_frac']:.2%}"))
    gates.append(gate("distance correlation > 0.5",
                      gaus["dist_corr"] > 0.5, f"{gaus['dist_corr']:.3f}"))
    bt = {k: v for k, v in (gaus.get("distance_by_type") or {}).items() if v["n"] >= 30}
    if bt:
        bad = [f"{k} x{v['true_over_pred']:.2f} cov90 {v['cov90']:.2f}"
               for k, v in bt.items()
               if not (0.8 < v["true_over_pred"] < 1.25 and v["cov90"] > 0.80)]
        gates.append(gate("distance unbiased per source type (|bias| < 25%, cov90 > 0.80; n>=30)",
                          not bad, "; ".join(bad) or ", ".join(
                              f"{k} x{v['true_over_pred']:.2f}" for k, v in bt.items())))

    # ---- sharpness gates (2026-07-21 audit) --------------------------------
    # Every gate above is invariant to posterior WIDTH. These are not. They are
    # expected to FAIL on the current generation of checkpoints — that is the
    # point: the failure they describe is real, was invisible to this suite, and
    # is the open work item (REPORT.md ranks 1-2). Do not relax them to green.
    sh = gaus.get("sharpness") or {}
    inf = sh.get("informativeness", {})
    for nm, label in (("chirp_mass", "chirp-mass"), ("mass_ratio", "mass-ratio")):
        if nm not in inf:
            continue
        ref = _DYNESTY_REF[nm]
        thr = ref * _SHARPNESS_TOL
        v = inf[nm]
        ratio = v / ref if np.isfinite(v) and ref > 0 else float("nan")
        gates.append(gate(
            f"{label} informativeness within {_SHARPNESS_TOL:g}x dynesty ({thr:.4f})",
            np.isfinite(v) and v < thr,
            f"{v:.4f} vs dynesty-achievable {ref:.4f} -> {ratio:.1f}x the optimum "
            f"(1.0 = parameter unmeasured)"))
    # non-finite counts as unmeasured: NaN compares False against any threshold,
    # so an unguarded `v > 0.85` would let a degenerate parameter through.
    uninformative = [k for k, v in inf.items() if not np.isfinite(v) or v > 0.85]
    gates.append(gate("no parameter is effectively unmeasured (post/prior > 0.85)",
                      not uninformative, f"unmeasured: {uninformative or 'none'}"))
    mc_scale = sh.get("snr_scaling", {}).get("chirp_mass")
    if mc_scale:
        gates.append(gate("Mc σ×SNR flat in SNR (ratio < 2.0; rising = encoder saturating)",
                          mc_scale["ratio"] < 2.0,
                          " → ".join(f"{b['sigma_x_snr']:.1f}" for b in mc_scale["bins"])
                          + f"  (ratio {mc_scale['ratio']:.2f})"))
    fve = sh.get("flow_vs_encoder", {})
    if "chirp_mass" in fve:
        gates.append(gate("[diagnostic] Mc post-width / encoder-probe residual",
                          True,
                          f"{fve['chirp_mass']:.2f} — ≈1 means ENCODER-limited "
                          f"(a larger flow cannot help); ≫1 would implicate the flow"))
    if real is not None:
        gates.append(gate("real-noise shuffle ΔNLL > 5 nats",
                          real["shuffle_delta_nll"] > 5, f"{real['shuffle_delta_nll']:.2f}"))
        bad = [n for n in PARAM_NAMES if abs(real["coverage"][n]["90"] - 0.90) > 0.07]
        gates.append(gate("real-noise 90% coverage within ±0.07 (≤2 exceptions)",
                          len(bad) <= 2, f"violations: {bad or 'none'}"))
        gates.append(gate("real vs gaussian NLL gap < 3 nats",
                          abs(real["nll"] - gaus["nll"]) < 3,
                          f"gaussian {gaus['nll']:.2f} vs real {real['nll']:.2f}"))
    gates.append(gate("real-event smoke tests (6 GWTC events, |t_c|<0.1s)", gw_ok, gw_summary))
    if basin is not None:
        from ahsd.evaluation.basin_probe import basin_gate
        b_ok, b_detail = basin_gate(basin)
        gates.append(gate("basin probe (τ≈+7ms heavy cell): q>0.95 < 0.30 & rail < 0.20",
                          b_ok, b_detail))
    return gates


def html_report(out_path, meta, gaus, gfigs, real, rfigs, gw, gates):
    """Render the full validation HTML report (per-domain metrics, figures, and
    the gate summary) to `out_path`."""
    ok = all(g["pass"] for g in gates)
    rows = "".join(
        f"<tr class={'pass' if g['pass'] else 'fail'}><td>{'PASS' if g['pass'] else 'FAIL'}</td>"
        f"<td>{g['gate']}</td><td>{g['detail']}</td></tr>" for g in gates)

    def domain_html(tag, m, figs):
        """Render the HTML block for one validation domain (metrics table +
        figures)."""
        if m is None:
            return f"<h2>{tag}</h2><p>skipped (no noise bank)</p>"
        inf = (m.get("sharpness") or {}).get("informativeness", {})
        sharp = ""
        if inf:
            sharp = ("<p>informativeness (posterior/prior width, 1.0 = unmeasured): "
                     + " | ".join(f"{k} <b>{inf[k]:.2f}</b>"
                                  for k in ("chirp_mass", "mass_ratio", "luminosity_distance")
                                  if k in inf) + "</p>")
        return (f"<h2>{tag}</h2>"
                f"<p>NLL {m['nll']:.3f} | shuffle Δ {m['shuffle_delta_nll']:.2f} | "
                f"dist corr {m['dist_corr']:.3f} | railing {m['railing_frac']:.2%} | "
                f"{m['n_events']} events × {m['n_post']} draws</p>"
                + sharp
                + "".join(f'<img src="data:image/png;base64,{b}"/>' for b in figs.values()))

    gw_html = "<p>skipped</p>"
    if gw:
        rows_gw = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in gw.items())
        gw_html = f"<table>{rows_gw}</table>"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>LeanNPE checkpoint validation</title><style>
body{{font-family:-apple-system,sans-serif;margin:2em auto;max-width:1100px;color:{INK}}}
img{{max-width:100%;margin:6px 0}} table{{border-collapse:collapse;margin:1em 0}}
td{{border:1px solid #d7dce2;padding:4px 10px;font-size:14px}}
tr.pass td:first-child{{background:#e3f0e3;font-weight:600}}
tr.fail td:first-child{{background:#f6dcd7;font-weight:600}}
.badge{{display:inline-block;padding:4px 14px;border-radius:6px;color:#fff;
background:{'#3f8f4a' if ok else '#c65a49'};font-weight:700}}</style></head><body>
<h1>LeanNPE checkpoint validation <span class="badge">{'PASS' if ok else 'FAIL'}</span></h1>
<p>checkpoint: <code>{meta['model']}</code> (epoch {meta['epoch']}, val NLL {meta['val_nll']:.4f})<br>
run: {meta['timestamp']} | device {meta['device']} | wall {meta['wall_s']:.0f}s</p>
<h2>Gates</h2><table><tr><td></td><td>gate</td><td>detail</td></tr>{rows}</table>
{domain_html('Gaussian validation', gaus, gfigs)}
{domain_html('Real-noise validation', real, rfigs)}
<h2>Real-event smoke tests</h2>{gw_html}
</body></html>"""
    Path(out_path).write_text(html)
    return ok


def main():
    """CLI: load a checkpoint, run the SBC/coverage/railing diagnostics on both
    validation domains plus the 6 GWTC real-event smoke tests, and write the
    HTML validation report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="path to remix_data/ directory")
    ap.add_argument("--noise_bank", default="data/noise_bank")
    ap.add_argument("--outdir", default=None,
                    help="default: analysis/ci/<checkpoint dir name>")
    ap.add_argument("--n_events", type=int, default=2000,
                    help="validation events per domain (5000 available; 2000 gives coverage MC error ~1.1%% at the 50%% level vs 1.8%% at 768)")
    ap.add_argument("--n_post", type=int, default=400)
    ap.add_argument("--skip_gw150914", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    t_start = time.time()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.outdir or f"analysis/ci/{Path(args.model).parent.name}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = LeanNPE(premerger=ckpt["args"].get("premerger", False),
                    psd_cond=ckpt["args"].get("psd_cond", False) or False,
                    psd_bands=ckpt["args"].get("psd_bands", 16),
                    encoder_type=ckpt["args"].get("encoder_type", "conv"),
                    mass_param=ckpt["args"].get("mass_param", "m1m2"),
                    geom_phase_residual=ckpt["args"].get("geom_phase_residual", False),
                    mf_tokens=ckpt["args"].get("mf_tokens", False), mf_shape_norm=ckpt["args"].get("mf_shape_norm", False), mf_token_fusion=ckpt["args"].get("mf_token_fusion", "add"), q_embed_dim=ckpt["args"].get("q_embed_dim", 0))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"checkpoint: epoch {ckpt['epoch']}, val NLL {ckpt['val_nll']:.4f}")

    from experiments.remix_data import RemixDataset, build_memmap_cache
    vcache = Path(args.data) / "memmap" / "validation"
    if not (vcache / "events.json").exists():
        build_memmap_cache(args.data, "validation", str(vcache))

    print("Gaussian validation ...")
    ds_g = RemixDataset(str(vcache), remix=False, seed=1234,
                        return_asd_bands=model.psd_cond,
                        psd_bands=model.encoder.psd_bands if model.psd_cond else 16)
    gaus, gfigs = evaluate_domain(model, ds_g, args.n_events, args.n_post, device,
                                  "Gaussian")

    real, rfigs = None, {}
    if args.noise_bank and Path(args.noise_bank).exists():
        try:
            print("Real-noise validation ...")
            ds_r = RemixDataset(str(vcache), remix=False, seed=1234,
                                real_noise_dir=args.noise_bank, real_noise_prob=1.0,
                                return_asd_bands=model.psd_cond,
                                psd_bands=model.encoder.psd_bands if model.psd_cond else 16)
            real, rfigs = evaluate_domain(model, ds_r, args.n_events, args.n_post,
                                          device, "Real noise")
        except Exception as e:
            print(f"real-noise validation skipped: {e}")

    # arm the OOD detector for this checkpoint
    ood_path = Path(args.model).parent / "ood_stats.npz"
    if not ood_path.exists():
        print("fitting OOD context stats ...")
        from ahsd.inference.ood import fit_context_stats
        print(fit_context_stats(model, str(vcache), str(ood_path), device=device))

    gw, gw_ok, gw_summary = None, True, "skipped"
    # GW170729 (dL ~2750 Mpc) sits OUTSIDE the training distance prior — it is
    # a deliberate live test of the OOD/refinement gate on every CI run.

    REAL_EVENTS = [
            ("GW150914", {"m1": 38.8, "m2": 33.4, "dL": 440}),
            ("GW151226", {"m1": 13.7, "m2": 7.7, "dL": 440}),
            ("GW170104", {"m1": 31.0, "m2": 20.1, "dL": 960}),
            ("GW170608", {"m1": 12.0, "m2": 7.0, "dL": 320}),
            ("GW170729", {"m1": 50.6, "m2": 34.3, "dL": 2750}),
            ("GW170814", {"m1": 30.7, "m2": 25.3, "dL": 600}),
        ]
    
    if not args.skip_gw150914:
        from ahsd.inference import infer
        gw, fails = {}, []
        for name, pub in REAL_EVENTS:
            print(f"{name} smoke test ...")
            try:
                res = infer(event=name, num_samples=3000, model=args.model,
                            device=device)
                med = res.median
                v = res.diagnostics.get("verdict", {})
                g = res.diagnostics.get("refinement", {})
                tc_ok = abs(med["geocent_time"]) < 0.1
                if not tc_ok:
                    fails.append(f"{name}: t_c offset {med['geocent_time']:+.3f}s")
                gw[name] = (
                    f"m1 {med['mass_1']:.1f} (pub {pub['m1']}) | "
                    f"m2 {med['mass_2']:.1f} (pub {pub['m2']}) | "
                    f"dL {med['luminosity_distance']:.0f} (pub {pub['dL']}) | "
                    f"t_c {med['geocent_time']:+.3f}s | "
                    f"rail {res.diagnostics['boundary_railing_frac']:.1%} | "
                    f"conf {v.get('confidence', '?')} | "
                    f"gate: {'REFINE' if g.get('refine') else 'as-is'}"
                    + (f" ({','.join(g.get('untrusted_params', []))})"
                       if g.get('untrusted_params') else ""))
                res.save(out / name.lower(), plots=True)
            except Exception as e:
                fails.append(f"{name}: {type(e).__name__}: {e}")
                gw[name] = f"FAILED: {e}"
        gw_ok = not fails
        gw_summary = (f"all {len(REAL_EVENTS)} events ran, |t_c|<0.1s"
                      if gw_ok else "; ".join(fails)[:200])

    basin_rows = None
    try:
        from ahsd.evaluation.basin_probe import build_bank, evaluate_bank
        basin_rows = evaluate_bank(model, build_bank(), device)
    except Exception as e:
        print(f"basin probe failed ({type(e).__name__}: {e})")

    meta = {"model": args.model, "epoch": ckpt["epoch"], "val_nll": ckpt["val_nll"],
            "device": device, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_s": time.time() - t_start}
    gates = run_gates(gaus, real, gw_ok, gw_summary, basin=basin_rows)
    ok = html_report(out / "report.html", meta, gaus, gfigs, real, rfigs, gw, gates)

    with open(out / "report.json", "w") as f:
        json.dump({"meta": meta, "gaussian": gaus, "real": real, "gw150914": gw,
                   "basin_probe": basin_rows, "gates": gates}, f, indent=2, default=float)
    for g in gates:
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['gate']}  ({g['detail']})")
    print(f"report -> {out}/report.html  |  overall: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
