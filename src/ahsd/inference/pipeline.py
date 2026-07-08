"""
infer(): one call from detector strain to a PosteriorResult.

    from ahsd.inference import infer
    posterior = infer(event="GW150914", num_samples=10000)
    posterior = infer(strain={"H1": h1, "L1": l1}, trigger_time=gps,
                      segment_start={...}, sample_rates={...})
    posterior = infer(strain={"H1": ..., "L1": ..., "V1": ...},
                      source="simulated", truth=injection_params)

The model, preprocessing, and parameter conventions are exactly those of
training: whitened unit-floor strain, 4 s @ 4096 Hz, geocent_time relative to
the window center, mass_1 >= mass_2, rank-0 = loudest signal.
"""

from __future__ import annotations

import resource
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES
from ahsd.inference.preprocessing import (PreparedData, prepare_real,
                                          prepare_simulated, fetch_gwosc)
from ahsd.inference.result import PosteriorResult

_MODEL_CACHE: dict = {}


def load_model(path: str, device: Optional[str] = None):
    """Load (and cache) a LeanNPE checkpoint. Returns (model, ckpt_meta)."""
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    key = (str(Path(path).resolve()), device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LeanNPE(premerger=ckpt["args"].get("premerger", False))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    meta = {"model_path": str(path), "model_epoch": int(ckpt.get("epoch", -1)),
            "model_val_nll": float(ckpt.get("val_nll", float("nan"))),
            "premerger": bool(ckpt["args"].get("premerger", False)),
            "device": device}
    _MODEL_CACHE[key] = (model, meta)
    return model, meta


def _log_prob_physical(model: LeanNPE, y_norm: torch.Tensor,
                       full_ctx: torch.Tensor) -> torch.Tensor:
    """log q(theta|d) in PHYSICAL units for normalized samples y_norm.

    flow.log_prob returns the NEGATIVE log prob in normalized space; the
    change of variables to physical units adds the ParamScaler Jacobian
    log|dy/dtheta| = sum_j [ log 2 - log(hi_j - lo_j) - (log theta_j if log-dim) ].
    """
    # same density path as training (model.nll): forward transform + standard-
    # normal base with zero log_sigma; the generic flow.log_prob is not wired
    # for the PSDScaledNormal base
    neg_logq_y = model.flow.compute_psd_aware_nll(
        y_norm, full_ctx, torch.zeros_like(y_norm))
    theta = model.scaler.denormalize(y_norm)
    span = model.scaler.hi - model.scaler.lo                      # [11]
    jac = (np.log(2.0) - torch.log(span)).sum()                   # scalar, linear dims
    log_theta = torch.where(model.scaler.log_mask,
                            torch.log(theta.clamp_min(1e-6)),
                            torch.zeros_like(theta))
    return -neg_logq_y + jac - log_theta.sum(dim=1)


def _memory_mb() -> dict:
    m = {"rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)}
    if torch.backends.mps.is_available():
        try:
            m["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / 1e6, 1)
        except Exception:
            pass
    return m


def infer(strain: Optional[Dict[str, np.ndarray]] = None,
          psds: Optional[Dict[str, dict]] = None,
          trigger_time: Optional[float] = None,
          event: Optional[str] = None,
          event_type: str = "BBH",
          num_samples: int = 10000,
          rank: int = 0,
          model: str = "model/lean_npe/best_model.pth",
          source: str = "auto",
          already_whitened: bool = False,
          segment_start: Optional[Dict[str, float]] = None,
          sample_rates: Optional[Dict[str, float]] = None,
          detectors: Optional[list] = None,
          truth: Optional[Dict[str, float]] = None,
          device: Optional[str] = None,
          seed: int = 0,
          batch_size: int = 4096) -> PosteriorResult:
    """Run LeanNPE inference on one event. See module docstring for modes.

    rank: which overlapping signal to infer (0 = loudest).
    event_type: informational for now (the model is trained on the full
    BBH/NSBH/BNS mix; priors are baked into training).
    """
    if event_type.upper() not in ("BBH", "NSBH", "BNS"):
        raise ValueError(f"unknown event_type {event_type!r}")
    t_total = time.time()
    torch.manual_seed(seed)

    # ---- data preparation ----
    if isinstance(strain, PreparedData):
        prepared = strain
    elif event is not None or (strain is None and trigger_time is not None):
        t0 = time.time()
        gps, raw, starts, srs, found = fetch_gwosc(event if event is not None else trigger_time,
                                                   detectors=detectors)
        fetch_s = time.time() - t0
        prepared = prepare_real(raw, gps, starts, srs, psds=psds, seed=seed)
        prepared.timings["gwosc_fetch_s"] = round(fetch_s, 2)
    elif strain is not None and source in ("auto", "real") and trigger_time is not None:
        prepared = prepare_real(strain, trigger_time,
                                segment_start or {d: trigger_time - len(v) / 2 / 4096
                                                  for d, v in strain.items()},
                                sample_rates or {d: 4096.0 for d in strain},
                                psds=psds, seed=seed)
    elif strain is not None:
        prepared = prepare_simulated(strain, psds=psds,
                                     already_whitened=already_whitened, seed=seed)
    else:
        raise ValueError("provide `event`, or `strain` (+ `trigger_time` for real data)")

    # ---- model ----
    net, meta = load_model(model, device=device)
    dev = meta["device"]

    # ---- encode + sample + log-prob ----
    t0 = time.time()
    st = torch.from_numpy(prepared.strain).unsqueeze(0).to(dev).float()
    with torch.no_grad():
        ctx = net.encoder(st)
    encode_s = time.time() - t0

    t0 = time.time()
    r = torch.full((1,), rank, dtype=torch.long, device=dev)
    full_ctx = net._full_context(ctx, r)                       # [1, C]
    P = len(PARAM_NAMES)
    samples = np.empty((num_samples, P), dtype=np.float64)
    logq = np.empty(num_samples, dtype=np.float64)
    railed = np.zeros(num_samples, dtype=bool)
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            k = min(batch_size, num_samples - i)
            fc = full_ctx.expand(k, -1)
            z = torch.randn(k, P, device=dev)
            y, _ = net.flow.inverse(z, fc)
            y = net.scaler.wrap(y)  # circular params wrap; bounded ones clamp
            # railing counts only non-circular dims (bound pile-up there is
            # the OOD signal; circular dims wrap by construction)
            railed[i:i + k] = ((y.abs() > 0.999) & ~net.scaler.circ_mask
                               ).any(dim=1).cpu().numpy()
            samples[i:i + k] = net.scaler.denormalize(y).cpu().numpy()
            logq[i:i + k] = _log_prob_physical(net, y, fc).cpu().numpy()
    sample_s = time.time() - t0

    # enforce the training mass convention on the output
    j1, j2 = PARAM_NAMES.index("mass_1"), PARAM_NAMES.index("mass_2")
    m1, m2 = samples[:, j1].copy(), samples[:, j2].copy()
    samples[:, j1], samples[:, j2] = np.maximum(m1, m2), np.minimum(m1, m2)

    # ---- diagnostics ----
    rail_frac = float(railed.mean())
    diagnostics = {
        "runtime": {"preprocess_s": prepared.timings.get("preprocess_s"),
                    "gwosc_fetch_s": prepared.timings.get("gwosc_fetch_s"),
                    "encode_s": round(encode_s, 3),
                    "sampling_s": round(sample_s, 3),
                    "total_s": round(time.time() - t_total, 3)},
        "memory": _memory_mb(),
        # flow samples are i.i.d. -> ESS = N; railing measures prior-boundary
        # pile-up (a real quality signal for NPE)
        "n_effective": num_samples,
        "boundary_railing_frac": round(rail_frac, 4),
        "log_prob": {"median": float(np.median(logq)), "max": float(logq.max())},
        "ood_flags": list(prepared.warnings),
    }
    if rail_frac > 0.05:
        diagnostics["ood_flags"].append(
            f"{rail_frac:.1%} of samples rail at the parameter bounds — the event "
            "may lie outside the training prior (out-of-distribution)")

    # context-space OOD score (needs <checkpoint_dir>/ood_stats.npz — build it
    # with ahsd.inference.ood.fit_context_stats once per checkpoint)
    from ahsd.inference.ood import (load_context_stats, score_context,
                                    confidence_verdict)
    stats = load_context_stats(meta["model_path"])
    ood_score = None
    if stats is not None:
        ood_score = score_context(ctx[0].cpu().numpy(), stats)
        diagnostics["context_ood"] = ood_score
    diagnostics["verdict"] = confidence_verdict(ood_score, rail_frac,
                                                prepared.warnings)

    config = {**meta, "num_samples": num_samples, "rank": rank, "seed": seed,
              "event": event, "event_type": event_type, "source": prepared.source,
              "detectors_present": prepared.detectors_present}

    out = PosteriorResult(samples=samples, log_prob=logq, param_names=list(PARAM_NAMES),
                          trigger_gps=prepared.trigger_gps, truth=truth,
                          prepared=prepared, diagnostics=diagnostics, config=config,
                          rail_mask=railed)
    # multi-indicator refinement gate (confidence + OOD + bias map + width
    # + railing) — the auditable "do I trust this?" decision
    from ahsd.inference.gating import refinement_gate
    out.diagnostics["refinement"] = refinement_gate(out)
    return out


def infer_overlapping(n_signals: int, priority_net: str = "model/pn/priority_net_best.pth",
                      **infer_kwargs) -> dict:
    """Posteriors for all k overlapping signals + PriorityNet ordering.

    Runs infer() once per NPE rank (0 = loudest by training convention); for
    k > 1 the candidates are then scored by PriorityNet (posterior point
    estimates + strain segments around each merger). k = 1 skips PriorityNet.

    Returns {"results": [PosteriorResult per NPE rank], "ranking": {...}}
    where ranking["order"][0] is the highest-priority candidate's NPE rank.
    """
    if n_signals < 1:
        raise ValueError("n_signals must be >= 1")
    results = []
    for r in range(n_signals):
        res = infer(rank=r, **infer_kwargs)
        results.append(res)
        # reuse the prepared data for the remaining ranks (identical input)
        infer_kwargs["strain"] = res.prepared
        infer_kwargs.pop("event", None)
        infer_kwargs.pop("trigger_time", None)

    from ahsd.inference.ranking import rank_overlapping
    ranking = {"order": [0], "priority": None, "uncertainty": None,
               "priority_net": None}
    if n_signals > 1:
        try:
            ranking = rank_overlapping(results, results[0].prepared.strain,
                                       priority_net=priority_net)
        except Exception as e:
            ranking["error"] = f"PriorityNet ranking failed: {e}"
    for r, res in enumerate(results):
        res.config["npe_rank"] = r
        if ranking.get("priority") is not None:
            res.config["priority_score"] = ranking["priority"][r]
            res.config["priority_order"] = ranking["order"].index(r)
    return {"results": results, "ranking": ranking}
