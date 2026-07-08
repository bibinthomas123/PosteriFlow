"""
PriorityNet-based ranking of overlapping signals at inference time.

LeanNPE is rank-conditioned (rank 0 = loudest by the training convention),
so for an event with k overlapping signals we draw one posterior per rank.
PriorityNet then scores those candidate signals (posterior point estimates +
strain segments around each merger) and returns the extraction/priority
order. Single-signal events skip PriorityNet entirely — there is nothing to
rank.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

_PN_CACHE: dict = {}


def _get_pn(path: str, device: str):
    key = (path, device)
    if key not in _PN_CACHE:
        from ahsd.core.priority_net import load_priority_net
        _PN_CACHE[key] = load_priority_net(path, device=device)
    return _PN_CACHE[key]


def _snr_proxy(m1: float, m2: float, dl: float) -> float:
    """Crude network-SNR estimate from the leading-order amplitude scaling
    Mc^(5/6)/d_L, normalized so a (30+30) Msun binary at 400 Mpc ~ SNR 25."""
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    return float(np.clip(25.0 * (mc ** (5.0 / 6.0) / 15.9) * (400.0 / max(dl, 1.0)),
                         0.0, 100.0))


def rank_overlapping(results: List, strain: np.ndarray,
                     priority_net: str = "model/pn/priority_net_best.pth",
                     device: str = "cpu",
                     sample_rate: int = 4096) -> dict:
    """Rank per-rank PosteriorResults with PriorityNet.

    results: one PosteriorResult per NPE rank (index = NPE loudness rank).
    strain:  [3, T] whitened model input (segments around each candidate's
             merger are cut from it for PriorityNet's temporal branch).
    Returns {"order": [...], "priority": [...], "uncertainty": [...]} where
    order[0] is the highest-priority candidate's NPE rank.
    """
    k = len(results)
    if k <= 1:  # nothing to rank
        return {"order": list(range(k)), "priority": None, "uncertainty": None,
                "priority_net": None}

    pn = _get_pn(priority_net, device)

    detections, segments = [], []
    T = strain.shape[-1]
    seg_len = 2048
    for res in results:
        med = res.median
        det = {n: med[n] for n in ("mass_1", "mass_2", "luminosity_distance",
                                   "ra", "dec", "geocent_time", "theta_jn",
                                   "psi", "phase")}
        det["a_1"], det["a_2"] = med["a1"], med["a2"]
        det["tilt_1"] = det["tilt_2"] = det["phi_12"] = det["phi_jl"] = 0.0
        det["network_snr"] = _snr_proxy(med["mass_1"], med["mass_2"],
                                        med["luminosity_distance"])
        detections.append(det)
        # 0.5 s strain segment centered on this candidate's inferred merger
        c = int(round((med["geocent_time"] + 2.0) * sample_rate))
        i0 = int(np.clip(c - seg_len // 2, 0, T - seg_len))
        segments.append(strain[:, i0:i0 + seg_len])

    seg_t = torch.from_numpy(np.stack(segments)).float()
    order = pn.rank_detections(detections, seg_t)
    with torch.no_grad():
        prio, unc = pn(detections, seg_t)
    return {"order": [int(i) for i in order],
            "priority": [round(float(p), 4) for p in prio],
            "uncertainty": [round(float(u), 4) for u in unc],
            "priority_net": priority_net}
