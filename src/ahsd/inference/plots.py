"""
Publication-quality posterior plots for PosteriorResult.

Waveform reconstruction reuses BilbyWaveformGenerator (the training injector's
waveform path) and whitens reconstructions with the SAME per-detector ASD the
data was whitened with, so data and model live in identical units.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ahsd.inference.preprocessing import DETECTORS, SAMPLE_RATE, DURATION, T_LEN

BLUE, GRAY, INK, RED = "#3b6fd4", "#9aa4b2", "#3a3f47", "#c65a49"

LABELS = {
    "mass_1": r"$m_1\ [M_\odot]$", "mass_2": r"$m_2\ [M_\odot]$",
    "luminosity_distance": r"$d_L$ [Mpc]", "ra": "RA [rad]", "dec": "dec [rad]",
    "theta_jn": r"$\theta_{JN}$ [rad]", "psi": r"$\psi$ [rad]",
    "phase": r"$\phi$ [rad]", "geocent_time": r"$t_c$ [s]",
    "a1": r"$a_1$", "a2": r"$a_2$",
}

_STYLE = {"axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": INK,
          "ytick.color": INK, "axes.grid": True, "grid.color": "#e8ebef",
          "grid.linewidth": 0.6, "font.size": 9}


def corner_plot(result, save: Optional[str] = None, params: Optional[list] = None):
    """Corner plot via the `corner` package when installed, otherwise a
    self-contained matplotlib fallback (2-D hists + 1-D marginals)."""
    names = params or result.param_names
    idx = [result.param_names.index(n) for n in names]
    data = result.samples[:, idx]
    truths = ([result.truth.get(n) for n in names] if result.truth else None)

    try:
        import corner as _corner
        fig = _corner.corner(
            data, labels=[LABELS.get(n, n) for n in names],
            truths=truths, truth_color=RED, color=BLUE,
            quantiles=[0.05, 0.5, 0.95], show_titles=True,
            title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 9},
            hist_kwargs={"density": True})
    except ImportError:
        fig = _corner_fallback(data, names, truths)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def _corner_fallback(data, names, truths):
    k = len(names)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(k, k, figsize=(2.0 * k, 2.0 * k))
        for i in range(k):
            for j in range(k):
                a = axes[i, j]
                if j > i:
                    a.axis("off"); continue
                if i == j:
                    a.hist(data[:, i], bins=50, color=BLUE, density=True)
                    if truths and truths[i] is not None:
                        a.axvline(truths[i], color=RED, lw=1.2)
                else:
                    a.hist2d(data[:, j], data[:, i], bins=40, cmap="Blues")
                    if truths and truths[j] is not None and truths[i] is not None:
                        a.plot(truths[j], truths[i], "s", color=RED, ms=4)
                if i == k - 1:
                    a.set_xlabel(LABELS.get(names[j], names[j]), fontsize=7)
                else:
                    a.set_xticklabels([])
                if j == 0 and i > 0:
                    a.set_ylabel(LABELS.get(names[i], names[i]), fontsize=7)
                else:
                    a.set_yticklabels([])
        fig.tight_layout()
    return fig


def marginals_plot(result, save: Optional[str] = None):
    names = result.param_names
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(3, 4, figsize=(15, 9))
        for j, n in enumerate(names):
            a = axes.flat[j]
            a.hist(result.samples[:, j], bins=60, color=BLUE, density=True,
                   edgecolor="white", linewidth=0.3)
            med = np.median(result.samples[:, j])
            a.axvline(med, color=INK, lw=1, ls="--", label=f"median {med:.3g}")
            if result.truth and n in result.truth:
                a.axvline(result.truth[n], color=RED, lw=1.3, label="truth")
            a.set_title(LABELS.get(n, n), color=INK, fontsize=10)
            a.legend(frameon=False, fontsize=7)
        axes.flat[-1].axis("off")
        fig.suptitle("Posterior marginals", color=INK)
        fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150); plt.close(fig)
    return fig


def cdf_plot(result, save: Optional[str] = None):
    names = result.param_names
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(3, 4, figsize=(15, 9))
        for j, n in enumerate(names):
            a = axes.flat[j]
            xs = np.sort(result.samples[:, j])
            a.plot(xs, np.linspace(0, 1, len(xs)), color=BLUE, lw=1.4)
            if result.truth and n in result.truth:
                a.axvline(result.truth[n], color=RED, lw=1.2)
            a.set_title(LABELS.get(n, n), color=INK, fontsize=10)
            a.set_ylim(0, 1)
        axes.flat[-1].axis("off")
        fig.suptitle("Posterior CDFs (red = truth)", color=INK)
        fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150); plt.close(fig)
    return fig


def reconstruction_plot(result, save: Optional[str] = None, n_draws: int = 50):
    """Whitened data vs waveform reconstructions from posterior draws, plus
    the residual after subtracting the MAP waveform."""
    from ahsd.data.bilby_pipeline import BilbyWaveformGenerator, _GPS_REF

    prep = result.prepared
    if prep is None:
        raise ValueError("no PreparedData attached — cannot reconstruct")
    wfg = BilbyWaveformGenerator(SAMPLE_RATE, DURATION)

    rng = np.random.default_rng(0)
    lp = result.log_prob
    if result.rail_mask is not None and (~result.rail_mask).any():
        lp = np.where(result.rail_mask, -np.inf, lp)  # match map_estimate
    draw_idx = np.concatenate([[int(np.argmax(lp))],  # MAP first
                               rng.choice(len(result.samples),
                                          size=min(n_draws - 1, len(result.samples)),
                                          replace=False)])

    def whitened_waveform(theta_row, det):
        p = {n: float(theta_row[j]) for j, n in enumerate(result.param_names)}
        # geocent_time is an offset from the window center; the waveform
        # generator's window is centered on _GPS_REF
        p["geocent_time_gps"] = _GPS_REF + p["geocent_time"]
        p["event_type"] = "BBH"
        sig = wfg.generate(p, det)
        asd = prep.asds[det].astype(np.float64)
        # whiten with the SAME ASD the data used; 4·df matched-filter norm
        # matches BilbyPreprocessor's whitening convention
        df = 1.0 / DURATION
        white = np.fft.irfft(np.fft.rfft(sig) / (asd / np.sqrt(4.0 * df)) / SAMPLE_RATE,
                             n=T_LEN)
        return white.astype(np.float32)

    t = np.arange(T_LEN) / SAMPLE_RATE - DURATION / 2
    # zoom around the inferred merger, not the window center (simulated events
    # place the merger anywhere in the window)
    tc_med = result.median["geocent_time"]
    xlim = (max(-DURATION / 2, tc_med - 0.6), min(DURATION / 2, tc_med + 0.25))
    dets = prep.detectors_present
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(len(dets), 2, figsize=(15, 3.1 * len(dets)),
                                 squeeze=False)
        for r, det in enumerate(dets):
            di = DETECTORS.index(det)
            data = prep.strain[di]
            a = axes[r, 0]
            a.plot(t, data, color=GRAY, lw=0.4, alpha=0.8, label="whitened data")
            map_wf = None
            for q, i in enumerate(draw_idx):
                wf = whitened_waveform(result.samples[i], det)
                if q == 0:
                    map_wf = wf
                    a.plot(t, wf, color=RED, lw=1.0, label="MAP waveform")
                else:
                    a.plot(t, wf, color=BLUE, lw=0.3, alpha=0.15)
            a.set_xlim(*xlim)
            a.set_title(f"{det}: data + posterior waveforms", color=INK, fontsize=10)
            a.set_xlabel("t - trigger [s]"); a.legend(frameon=False, fontsize=8)
            a2 = axes[r, 1]
            a2.plot(t, data - map_wf, color=GRAY, lw=0.4)
            a2.set_xlim(*xlim)
            res_std = float((data - map_wf).std())
            a2.set_title(f"{det}: residual after MAP subtraction (std {res_std:.2f}, "
                         "unit noise -> 1.0)", color=INK, fontsize=10)
            a2.set_xlabel("t - trigger [s]")
        fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150); plt.close(fig)
    return fig
