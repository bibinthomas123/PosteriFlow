"""
Out-of-distribution detection for LeanNPE inference.

The encoder context is a 256-d summary of the event; events unlike anything
in training land far from the training context cloud. We measure Mahalanobis
distance to that cloud and calibrate it EMPIRICALLY: the verdict compares the
event's distance to the distribution of distances of held-out validation
events (no Gaussianity assumption on the context).

    fit_context_stats(model, cache_dir, out_path)   # once per checkpoint
    score = score_context(ctx, stats)               # at inference

infer() picks stats up automatically from <checkpoint_dir>/ood_stats.npz and
combines the Mahalanobis percentile with input-quality flags and boundary
railing into a single HIGH / MEDIUM / LOW confidence verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def fit_context_stats(model, cache_dir: str, out_path: str,
                      n_events: int = 2000, device: str = "cpu",
                      batch: int = 64) -> dict:
    """Encode up to n_events validation events; store context mean, inverse
    covariance (shrinkage-regularized), and the empirical distance CDF."""
    import sys
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from experiments.remix_data import RemixDataset

    ds = RemixDataset(cache_dir, remix=False, seed=1234)
    n = min(n_events, len(ds))
    ctxs = []
    with torch.no_grad():
        for i in range(0, n, batch):
            st = torch.stack([ds[k][0] for k in range(i, min(i + batch, n))])
            ctxs.append(model.encoder(st.to(device).float()).cpu().numpy())
    ctx = np.concatenate(ctxs).astype(np.float64)          # [n, C]

    mean = ctx.mean(axis=0)
    xc = ctx - mean
    cov = xc.T @ xc / (len(ctx) - 1)
    # Ledoit-Wolf-style shrinkage keeps the inverse stable for C=256, n~2000
    lam = 0.1
    cov = (1 - lam) * cov + lam * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    cov_inv = np.linalg.inv(cov)

    d = np.sqrt(np.einsum("ij,jk,ik->i", xc, cov_inv, xc))
    ref = np.sort(d)
    np.savez(out_path, mean=mean, cov_inv=cov_inv, ref_distances=ref)
    return {"n": int(n), "median_distance": float(np.median(ref)),
            "p99_distance": float(np.quantile(ref, 0.99))}


def load_context_stats(checkpoint_path: str) -> Optional[dict]:
    p = Path(checkpoint_path).parent / "ood_stats.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {"mean": z["mean"], "cov_inv": z["cov_inv"],
            "ref_distances": z["ref_distances"]}


def score_context(ctx: np.ndarray, stats: dict) -> Dict[str, float]:
    """Mahalanobis distance of one event context + its percentile among
    training-domain validation events (percentile 100 = farther than
    everything seen in validation)."""
    xc = ctx.astype(np.float64) - stats["mean"]
    d = float(np.sqrt(xc @ stats["cov_inv"] @ xc))
    pct = float(np.searchsorted(stats["ref_distances"], d) /
                len(stats["ref_distances"]) * 100.0)
    return {"mahalanobis": round(d, 2), "percentile_vs_validation": round(pct, 1)}


def confidence_verdict(ood_score: Optional[Dict], rail_frac: float,
                       quality_warnings: list) -> Dict:
    """Aggregate into HIGH / MEDIUM / LOW with reasons. Conservative: any
    strong signal downgrades."""
    reasons = []
    level = "HIGH"

    if ood_score is not None:
        pct = ood_score["percentile_vs_validation"]
        if pct >= 100.0:
            level = "LOW"
            reasons.append(f"context farther from training than ALL validation "
                           f"events (Mahalanobis {ood_score['mahalanobis']})")
        elif pct > 99.0:
            level = "MEDIUM"
            reasons.append(f"context in the top {100 - pct:.1f}% distance tail")
    if rail_frac > 0.20:
        level = "LOW"
        reasons.append(f"{rail_frac:.0%} of posterior samples rail at prior bounds")
    elif rail_frac > 0.05 and level == "HIGH":
        level = "MEDIUM"
        reasons.append(f"{rail_frac:.0%} boundary railing")
    hard = [w for w in quality_warnings if "far from 1" in w or "glitch" in w
            or "NaN" in w or "dropouts" in w]
    if hard:
        level = "LOW" if len(hard) > 1 else ("MEDIUM" if level == "HIGH" else level)
        reasons.extend(hard)
    soft = [w for w in quality_warnings if w not in hard]
    if soft and level == "HIGH":
        reasons.extend(soft)  # informational (e.g. white-noise fill)

    return {"confidence": level, "reasons": reasons}
