"""
Refinement gate: "does this event need sampler refinement?" decided from ALL
independent indicators, not a single confidence threshold.

Indicators (each validated separately in this project):
  1. input-quality / confidence verdict  (caught GW170817's L1 glitch)
  2. context OOD percentile              (Mahalanobis vs validation cloud)
  3. bias-map region                     (twin-grid measured amortization bias:
                                          the q-collapse and edge-on-dL cells)
  4. posterior width                     (uninformative posterior = the data
                                          was not understood)
  5. boundary railing                    (prior-edge pile-up where the truth
                                          is not expected at an edge)

Aggregation: strong indicators refine on their own; two or more moderate
indicators refine together. The output carries every indicator's value and
reason so the decision is auditable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# thresholds (moderate, strong) per indicator
OOD_PCT = (99.0, 100.0)
RAIL = (0.05, 0.20)
MC_WIDTH_FRAC = (0.6, 1.0)      # 90% CI width / median for chirp mass

_GRID_CACHE: Optional[list] = None


def _twin_grid():
    global _GRID_CACHE
    if _GRID_CACHE is None:
        p = Path(__file__).resolve().parents[3] / "analysis" / "twin_grid_v3.json"
        _GRID_CACHE = json.load(open(p)) if p.exists() else []
    return _GRID_CACHE


# The twin grid + dynesty anchors show the q collapse is an ATTRACTOR:
# truths on BOTH sides (0.4 and 0.9) get reported as q ~ 0.5, so any event
# whose REPORTED q falls in this band has unconstrained masses. dL bias is
# deliberately excluded from severity: the anchors showed the edge-on dL
# inflation is largely shared with dynesty (physical degeneracy).
Q_ATTRACTOR = (0.30, 0.80)


def _bias_region(mc: float, q: float, theta_jn: float) -> Dict:
    """Posterior-median lookup in the measured bias map (mass biases only)."""
    out = {"q_in_attractor": bool(Q_ATTRACTOR[0] < q < Q_ATTRACTOR[1] and mc > 8)}
    grid = _twin_grid()
    if grid:
        cell = min(grid, key=lambda r: (np.log(max(mc, 1) / r["mc"]) ** 2 +
                                        (q - r["q"]) ** 2 * 4 +
                                        (theta_jn - r["theta_jn"]) ** 2 * 0.3))
        m1b = abs(cell["m1_ratio"] - 1)
        out.update({"cell": {k: cell[k] for k in ("mc", "q", "theta_jn")},
                    "m1_bias": round(cell["m1_ratio"], 2),
                    "dl_bias": round(cell["dl_ratio"], 2),
                    "severe_mass_bias": bool(m1b > 0.4)})
    else:
        out.update({"cell": None, "severe_mass_bias": bool(q > 0.8)})
    return out


def refinement_gate(result) -> Dict:
    """Aggregate all indicators into {'refine': bool, 'reasons': [...]}."""
    strong, moderate, indicators = [], [], {}
    d = result.diagnostics
    names = result.param_names

    # 1. confidence verdict (input quality + aggregated flags)
    verdict = d.get("verdict", {})
    lvl = verdict.get("confidence", "HIGH")
    indicators["confidence"] = lvl
    if lvl == "LOW":
        strong.append("confidence LOW: " + "; ".join(verdict.get("reasons", [])[:2]))
    elif lvl == "MEDIUM":
        moderate.append("confidence MEDIUM")

    # 2. context OOD distance
    ood = d.get("context_ood")
    if ood:
        pct = ood["percentile_vs_validation"]
        indicators["ood_percentile"] = pct
        if pct >= OOD_PCT[1]:
            strong.append(f"context beyond all validation events "
                          f"(Mahalanobis {ood['mahalanobis']})")
        elif pct >= OOD_PCT[0]:
            moderate.append(f"context in top {100 - pct:.1f}% OOD tail")

    # 3. bias-map region -> PARAMETER-LEVEL distrust (the q collapse is an
    # attractor: a reported q in the band could have come from anywhere)
    untrusted = []
    med = result.median
    mc = (med["mass_1"] * med["mass_2"]) ** 0.6 / (med["mass_1"] + med["mass_2"]) ** 0.2
    q = med["mass_2"] / med["mass_1"]
    region = _bias_region(mc, q, med["theta_jn"])
    indicators["bias_region"] = region
    if region.get("severe_mass_bias"):
        strong.append(f"posterior median in severe mass-bias cell {region['cell']} "
                      f"(m1 x{region.get('m1_bias')})")
    if region["q_in_attractor"]:
        untrusted += ["mass_1", "mass_2"]

    # 3b. soft input warnings (e.g. missing-detector white-noise fill) count
    # as a moderate indicator here even when the confidence verdict stays HIGH
    soft = [w for w in (result.prepared.warnings if result.prepared else [])
            if "filled with unit white noise" in w]
    if soft:
        moderate.append(soft[0])

    # 4. posterior width (uninformative = not understood)
    j1, j2 = names.index("mass_1"), names.index("mass_2")
    mc_s = ((result.samples[:, j1] * result.samples[:, j2]) ** 0.6 /
            (result.samples[:, j1] + result.samples[:, j2]) ** 0.2)
    lo, hi = np.quantile(mc_s, [0.05, 0.95])
    wfrac = float((hi - lo) / max(np.median(mc_s), 1e-9))
    indicators["mc_width_frac"] = round(wfrac, 3)
    if wfrac > MC_WIDTH_FRAC[1]:
        strong.append(f"chirp-mass posterior uninformative (90% width "
                      f"{wfrac:.0%} of median)")
    elif wfrac > MC_WIDTH_FRAC[0]:
        moderate.append(f"chirp-mass posterior wide ({wfrac:.0%} of median)")

    # 5. spurious boundary railing
    rail = d.get("boundary_railing_frac", 0.0)
    indicators["railing_frac"] = rail
    if rail > RAIL[1]:
        strong.append(f"{rail:.0%} of samples rail at prior bounds")
    elif rail > RAIL[0]:
        moderate.append(f"{rail:.0%} boundary railing")

    refine_global = bool(strong) or len(moderate) >= 2
    if refine_global:
        rec = ("run sampler refinement (compare_to_dynesty) before quoting "
               "these posteriors")
    elif untrusted:
        rec = (f"posterior usable EXCEPT {sorted(set(untrusted))}: reported "
               "mass ratio sits in the measured prior-reversion attractor "
               "band — refine with the sampler if masses matter")
    else:
        rec = "amortized posterior usable as-is"
    return {"refine": bool(refine_global or untrusted),
            "refine_global": refine_global,
            "untrusted_params": sorted(set(untrusted)),
            "strong_indicators": strong,
            "moderate_indicators": moderate,
            "indicators": indicators,
            "recommendation": rec}
