"""
PosteriorResult: samples + diagnostics + summaries + persistence for one event.

Everything downstream of the model lives here: point estimates, credible
intervals, covariance/correlation, plotting entry points, bilby-compatible
export, and a reproducibility record (model checkpoint, git commit,
preprocessing settings) written next to the samples.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ahsd.models.lean_npe import PARAM_NAMES

CI_LEVELS = [0.50, 0.68, 0.90, 0.95]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class PosteriorResult:
    samples: np.ndarray                     # [N, 11] physical units
    log_prob: np.ndarray                    # [N] log q(theta|d) in physical units
    param_names: List[str]
    trigger_gps: Optional[float]            # geocent_time samples are offsets from this
    truth: Optional[Dict[str, float]] = None
    prepared: object = None                 # PreparedData (strain, ASDs, quality)
    diagnostics: Dict = field(default_factory=dict)
    config: Dict = field(default_factory=dict)
    rail_mask: Optional[np.ndarray] = None   # samples clamped at a parameter bound

    # ── point estimates ──────────────────────────────────────────────────────
    @property
    def median(self) -> Dict[str, float]:
        return {n: float(np.median(self.samples[:, j])) for j, n in enumerate(self.param_names)}

    @property
    def mean(self) -> Dict[str, float]:
        return {n: float(self.samples[:, j].mean()) for j, n in enumerate(self.param_names)}

    @property
    def map_estimate(self) -> Dict[str, float]:
        # boundary-clamped samples have unreliable density values; exclude
        # them from the argmax unless everything railed
        lp = self.log_prob
        if self.rail_mask is not None and (~self.rail_mask).any():
            lp = np.where(self.rail_mask, -np.inf, lp)
        i = int(np.argmax(lp))
        return {n: float(self.samples[i, j]) for j, n in enumerate(self.param_names)}

    def credible_interval(self, level: float = 0.90) -> Dict[str, tuple]:
        lo = np.quantile(self.samples, 0.5 - level / 2, axis=0)
        hi = np.quantile(self.samples, 0.5 + level / 2, axis=0)
        return {n: (float(lo[j]), float(hi[j])) for j, n in enumerate(self.param_names)}

    def covariance(self) -> np.ndarray:
        return np.cov(self.samples.T)

    def correlation(self) -> np.ndarray:
        return np.corrcoef(self.samples.T)

    # ── summary ──────────────────────────────────────────────────────────────
    def summary(self, print_it: bool = True) -> Dict:
        med, mp = self.median, self.map_estimate
        cis = {f"{int(l*100)}%": self.credible_interval(l) for l in CI_LEVELS}
        rows = []
        for j, n in enumerate(self.param_names):
            row = {"param": n, "median": med[n], "mean": self.mean[n], "map": mp[n],
                   "std": float(self.samples[:, j].std())}
            for lab, ci in cis.items():
                row[f"ci{lab}"] = ci[n]
            if self.truth and n in self.truth:
                row["truth"] = float(self.truth[n])
                lo, hi = cis["90%"][n]
                row["truth_in_90"] = bool(lo <= self.truth[n] <= hi)
            rows.append(row)
        out = {"n_samples": len(self.samples), "trigger_gps": self.trigger_gps,
               "parameters": rows, "diagnostics": self.diagnostics}
        if print_it:
            hdr = f"{'param':<22}{'median':>12}{'map':>12}{'90% CI':>28}"
            hdr += f"{'truth':>10}" if self.truth else ""
            print(hdr)
            print("-" * len(hdr))
            for r in rows:
                lo, hi = r["ci90%"]
                line = (f"{r['param']:<22}{r['median']:>12.4g}{r['map']:>12.4g}"
                        f"{f'[{lo:.4g}, {hi:.4g}]':>28}")
                if self.truth and "truth" in r:
                    flag = "" if r["truth_in_90"] else "  <-- outside 90%"
                    line += f"{r['truth']:>10.4g}{flag}"
                print(line)
            if self.trigger_gps is not None:
                print(f"\ngeocent_time is offset from trigger GPS {self.trigger_gps:.4f}; "
                      f"absolute merger GPS median = "
                      f"{self.trigger_gps + med['geocent_time']:.4f}")
            for w in (self.prepared.warnings if self.prepared else []):
                print(f"WARNING: {w}")
            g = self.diagnostics.get("refinement")
            if g:
                tag = "REFINEMENT RECOMMENDED" if g["refine"] else "usable as-is"
                print(f"\nRefinement gate: {tag}")
                for s_ in g["strong_indicators"]:
                    print(f"  [strong]   {s_}")
                for m_ in g["moderate_indicators"]:
                    print(f"  [moderate] {m_}")
            v = self.diagnostics.get("verdict")
            if v:
                if v["confidence"] == "HIGH":
                    print("Confidence: HIGH")
                else:
                    print(f"Warning: confidence {v['confidence']} — "
                          "event may lie outside the training distribution.")
                for r in v["reasons"]:
                    print(f"  - {r}")
        return out

    # ── plots (delegated) ────────────────────────────────────────────────────
    def corner(self, save: Optional[str] = None, **kw):
        from ahsd.inference.plots import corner_plot
        return corner_plot(self, save=save, **kw)

    def plot_marginals(self, save: Optional[str] = None):
        from ahsd.inference.plots import marginals_plot
        return marginals_plot(self, save=save)

    def plot_cdfs(self, save: Optional[str] = None):
        from ahsd.inference.plots import cdf_plot
        return cdf_plot(self, save=save)

    def plot_reconstruction(self, save: Optional[str] = None, n_draws: int = 50):
        from ahsd.inference.plots import reconstruction_plot
        return reconstruction_plot(self, save=save, n_draws=n_draws)

    # ── bilby compatibility ──────────────────────────────────────────────────
    def to_bilby(self, label: str = "leannpe"):
        """Return a bilby.core.result.Result whose posterior DataFrame uses
        bilby's parameter names and absolute GPS geocent_time."""
        import pandas as pd
        import bilby
        rename = {"a1": "a_1", "a2": "a_2"}
        data = {}
        for j, n in enumerate(self.param_names):
            col = self.samples[:, j].astype(np.float64)
            if n == "geocent_time" and self.trigger_gps is not None:
                col = col + self.trigger_gps
            data[rename.get(n, n)] = col
        data["log_likelihood"] = self.log_prob  # closest available analogue
        center = self.trigger_gps if self.trigger_gps is not None else 0.0
        try:
            from ahsd.inference.dynesty_bridge import build_priors
            priors = build_priors(center)
        except Exception:
            priors = None
        res = bilby.core.result.Result(
            label=label, outdir=".",
            search_parameter_keys=list(data.keys())[:-1],
            priors=priors,
            posterior=pd.DataFrame(data),
            meta_data={"analysis": "LeanNPE amortized NPE",
                       "model": self.config.get("model_path"),
                       "model_epoch": self.config.get("model_epoch"),
                       "git_commit": _git_commit(),
                       "trigger_gps": self.trigger_gps,
                       "diagnostics": self.diagnostics},
            sampler_kwargs={"num_samples": len(self.samples)})
        return res

    def importance_correct(self, raw_strain: Dict[str, np.ndarray],
                           center_gps: Optional[float] = None, npool: int = 4,
                           **kw):
        """Exact reweighting against the true bilby likelihood (DINGO-IS
        style); removes proposal bias (notably mass ratio) wherever the NPE
        posterior covers the truth. See ahsd.inference.importance."""
        from ahsd.inference.importance import importance_correct
        from ahsd.inference.dynesty_bridge import GPS_REF
        center = center_gps if center_gps is not None else (
            self.trigger_gps if self.trigger_gps is not None else GPS_REF)
        return importance_correct(self, raw_strain, self.prepared.psds,
                                  center_gps=center, npool=npool, **kw)

    def compare_to_dynesty(self, raw_strain: Dict[str, np.ndarray],
                           outdir: str, injection: Optional[Dict] = None,
                           nlive: int = 300):
        """Run bilby/dynesty on the SAME data (identical priors/PSDs/epoch as
        the training pipeline) and return the per-parameter comparison dict.
        raw_strain: unwhitened 4 s strains keyed by detector."""
        from ahsd.inference.dynesty_bridge import run_comparison
        return run_comparison(self, raw_strain, outdir, injection=injection, nlive=nlive)

    # ── prior reweighting ────────────────────────────────────────────────────
    def reweight_to_uniform_masses(self, m_lo: float = 5.0):
        """Importance-reweight the posterior from the TRAINING mass prior
        (log-uniform in m1; log-uniform in m2 given m2 < m1) to the standard
        LVC uniform-in-component-masses prior. Distance (d^2 ~ comoving
        volume) and spin (uniform) priors already match LVC conventions, so
        masses are the only dimension that needs correcting when comparing
        against published posteriors.

        w(theta) = p_target/p_train ∝ m1 * m2 * log(m1/m_lo)

        Returns a NEW PosteriorResult with systematically resampled draws;
        the importance ESS is recorded in diagnostics["reweighting"].
        """
        j1 = self.param_names.index("mass_1")
        j2 = self.param_names.index("mass_2")
        m1 = np.maximum(self.samples[:, j1], m_lo * (1 + 1e-6))
        m2 = self.samples[:, j2]
        w = m1 * m2 * np.log(m1 / m_lo)
        w = np.clip(w, 0, np.quantile(w, 0.999))  # guard the far tail
        w = w / w.sum()
        ess = float(1.0 / (w ** 2).sum())
        # systematic resampling
        n = len(w)
        u = (np.random.default_rng(0).uniform() + np.arange(n)) / n
        idx = np.searchsorted(np.cumsum(w), u).clip(0, n - 1)
        out = PosteriorResult(
            samples=self.samples[idx], log_prob=self.log_prob[idx],
            param_names=list(self.param_names), trigger_gps=self.trigger_gps,
            truth=self.truth, prepared=self.prepared,
            diagnostics={**self.diagnostics,
                         "reweighting": {"target": "uniform component masses",
                                         "importance_ess": round(ess, 1),
                                         "n": n}},
            config=dict(self.config),
            rail_mask=(self.rail_mask[idx] if self.rail_mask is not None else None))
        return out

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, outdir: str, plots: bool = True) -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "posterior_samples.npy", self.samples)
        np.save(out / "posterior_log_prob.npy", self.log_prob)
        # CSV for anything that reads tabular posteriors (bilby, pandas, R)
        header = ",".join(self.param_names + ["log_prob"])
        np.savetxt(out / "posterior_samples.csv",
                   np.column_stack([self.samples, self.log_prob]),
                   delimiter=",", header=header, comments="")
        meta = {
            "param_names": self.param_names,
            "trigger_gps": self.trigger_gps,
            "truth": self.truth,
            "summary": self.summary(print_it=False),
            "covariance": self.covariance().tolist(),
            "correlation": self.correlation().tolist(),
            "diagnostics": self.diagnostics,
            "config": self.config,
            "reproducibility": {
                "git_commit": _git_commit(),
                **{k: self.config.get(k) for k in
                   ("model_path", "model_epoch", "model_val_nll", "premerger")},
            },
            "preprocessing": ({
                "source": self.prepared.source,
                "detectors_present": self.prepared.detectors_present,
                "warnings": self.prepared.warnings,
                "quality": self.prepared.quality,
                "timings": self.prepared.timings,
            } if self.prepared is not None else None),
        }
        with open(out / "result.json", "w") as f:
            json.dump(meta, f, indent=2, default=float)
        if self.prepared is not None:
            np.save(out / "model_input_strain.npy", self.prepared.strain)
            for det, asd in self.prepared.asds.items():
                np.save(out / f"asd_{det}.npy", asd)
        if plots:
            self.corner(save=str(out / "corner.png"))
            self.plot_marginals(save=str(out / "marginals.png"))
            self.plot_cdfs(save=str(out / "cdfs.png"))
            try:
                self.plot_reconstruction(save=str(out / "reconstruction.png"))
            except Exception as e:
                print(f"reconstruction plot skipped: {e}")
        return out

