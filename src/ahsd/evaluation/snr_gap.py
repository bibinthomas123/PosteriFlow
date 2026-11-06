"""Real vs Gaussian NLL gap at matched SNR, for certification.

Recolouring into measured noise makes the same validation events quieter (bbh_v6g:
median SNR 13.6 Gaussian, 8.1 real), and NLL depends strongly on SNR, so the raw
difference of the two means mostly measures loudness: 7.5 nats overall, but
1.3 and 0.4 nats at SNR 10-20 and 20-40 (2026-11-06,
analysis/distance_realdata/audit_gap_cert.py).

Each domain is binned by its own SNR. The events inside a bin differ between the two
domains, so this compares populations, not paired events. The gate covers the bins
where real events sit; quieter and louder bins and the raw gap stay in the report.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

BINS = ((6.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, np.inf))
GATED = ("10-20", "20-40")
LIMIT = 3.0        # nats
MIN_EVENTS = 30    # per domain, for a bin to count


def _label(lo: float, hi: float) -> str:
    return f"{lo:g}+" if np.isinf(hi) else f"{lo:g}-{hi:g}"


def nll_by_snr(nll: np.ndarray, snr: np.ndarray) -> Dict[str, Dict[str, float]]:
    """{bin: {"n", "nll"}} for one domain, binned by that domain's own SNR."""
    nll, snr = np.asarray(nll, dtype=float), np.asarray(snr, dtype=float).reshape(len(nll), -1)[:, 0]
    out = {}
    for lo, hi in BINS:
        m = (snr >= lo) & (snr < hi)
        out[_label(lo, hi)] = {"n": int(m.sum()), "nll": float(nll[m].mean()) if m.any() else float("nan")}
    return out


def gaps(gaus: Dict, real: Dict) -> Dict[str, float]:
    """real - gaussian mean NLL per bin with enough events in both domains."""
    return {b: real[b]["nll"] - gaus[b]["nll"] for b in gaus
            if gaus[b]["n"] >= MIN_EVENTS and real[b]["n"] >= MIN_EVENTS}


def snr_gap_gate(gaus: Dict, real: Dict) -> Tuple[str, bool, str]:
    """(name, passed, detail). Fails if no gated bin has enough events: an unmeasured
    gap must not pass."""
    g = gaps(gaus, real)
    tested = [b for b in GATED if b in g]
    ok = bool(tested) and all(abs(g[b]) < LIMIT for b in tested)
    detail = ", ".join(f"SNR {b}: {v:+.2f}" for b, v in g.items()) or "no bin has enough events"
    return f"real vs gaussian NLL gap < {LIMIT:g} nats at matched SNR (bins {', '.join(GATED)})", ok, detail
