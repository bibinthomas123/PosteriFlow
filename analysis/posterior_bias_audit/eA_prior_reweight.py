#!/usr/bin/env python3
"""E-A: how much of the m1 upper tail is INHERITED TRAINING PRIOR vs model error?

The NPE returns p(theta|x) under ITS OWN training prior. GWTC/Bilby quote
posteriors under a prior that is uniform in detector-frame (m1, m2). The
training prior here is log-flat in both masses (BBH) mixed with BNS/NSBH, so it
carries far more weight at extreme mass ratio than GWTC's does. Since the model
is measurably ~uninformative about q (audit REPORT.md Sec.1), it returns roughly
that prior -- and m1 = Mc (1+q)^0.2 q^(-0.6) amplifies the low-q tail.

Reweighting is exact for a prior swap:
    p_target(theta|x)  ∝  p_train(theta|x) * p_target(theta) / p_train(theta)

Masses only. Distance/sky/spin priors are left alone: they do not enter m1, and
reweighting them too would only cost effective sample size.

p_train(m1,m2) is estimated EMPIRICALLY from the shipped training parameters
(the dataset is a BBH/BNS/NSBH mixture, so the analytic 1/(m1 m2) BBH form does
not describe it), via a smoothed 2-D histogram in (log m1, log m2).
"""
import sys
from pathlib import Path
ROOT = Path("/Users/bibinthomas/Developer/PosteriFlow")
import numpy as np

GWTC = {  # m1, m2, Mc, q, and the GWTC-1 90% CI on m1 (source-frame, approx)
    "gw150914": (35.6, 30.6, 28.6, 0.86, (32.3, 40.5)),
    "gw151226": (13.7, 7.7, 8.9, 0.56, (10.5, 19.4)),
    "gw170104": (31.0, 20.1, 21.4, 0.65, (25.1, 39.0)),
    "gw170608": (11.0, 7.6, 7.9, 0.69, (9.5, 15.9)),
    "gw170729": (50.2, 34.0, 35.4, 0.68, (38.5, 66.5)),
    "gw170814": (30.6, 25.2, 24.1, 0.83, (27.2, 35.4)),
}

# ---- empirical training prior in (log m1, log m2) ----
p = np.asarray(np.load(ROOT / "data/dataset/memmap/train/params.npy", mmap_mode="r")[:200000])
tm1, tm2 = p[:, 0], p[:, 1]
u, v = np.log(tm1), np.log(tm2)
EDG = np.linspace(np.log(1.0), np.log(105.0), 61)
H, _, _ = np.histogram2d(u, v, bins=[EDG, EDG], density=True)
# light smoothing so empty cells in the tail do not produce infinite weights
from scipy.ndimage import gaussian_filter
H = gaussian_filter(H, 1.0)
H = np.maximum(H, H.max() * 1e-4)


def log_p_train(m1, m2):
    """log density of the TRAINING prior in physical (m1, m2)."""
    i = np.clip(np.digitize(np.log(m1), EDG) - 1, 0, H.shape[0] - 1)
    j = np.clip(np.digitize(np.log(m2), EDG) - 1, 0, H.shape[1] - 1)
    # p(m1,m2) = p_log(log m1, log m2) / (m1 m2)
    return np.log(H[i, j]) - np.log(m1) - np.log(m2)


def wq(x, w, qs):
    """weighted quantiles"""
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return np.interp(qs, c, x)


print("E-A: prior reweighting to a GWTC-style uniform-in-(m1,m2) mass prior")
print("=" * 104)
print(f"{'event':10s} {'m1 med':>7s} {'m1 90% CI (as-is)':>19s} | {'m1 med':>7s} "
      f"{'m1 90% CI (reweighted)':>23s} | {'ESS':>6s} | {'GWTC m1 90% CI':>16s}")
print("=" * 104)
rows = []
for ev, (M1, M2, MC, Q, gci) in GWTC.items():
    f = ROOT / f"analysis/ci/lean_npe_mf_scratch/{ev}/posterior_samples.npy"
    s = np.load(f)
    m1, m2 = s[:, 0], s[:, 1]
    keep = (m1 >= 1) & (m2 >= 1) & (m1 <= 105) & (m2 <= 105)
    m1, m2 = m1[keep], m2[keep]
    # target: uniform in (m1, m2) -> log p_target = const
    lw = -log_p_train(m1, m2)
    lw -= lw.max()
    w = np.exp(lw)
    ess = w.sum() ** 2 / (w ** 2).sum()
    a_lo, a_med, a_hi = np.percentile(m1, [5, 50, 95])
    r_lo, r_med, r_hi = wq(m1, w, [0.05, 0.50, 0.95])
    print(f"{ev:10s} {a_med:7.2f} [{a_lo:6.2f},{a_hi:7.2f}] | {r_med:7.2f} "
          f"[{r_lo:6.2f},{r_hi:7.2f}] | {ess:6.0f} | [{gci[0]:5.1f},{gci[1]:5.1f}]")
    rows.append((ev, a_hi, r_hi, gci[1], ess, len(m1)))

print("=" * 104)
print("\nm1 90% UPPER limit — the quantity in the audit brief:")
print(f"{'event':10s} {'as-is':>8s} {'reweighted':>11s} {'GWTC':>7s} "
      f"{'tail removed by prior':>22s} {'residual vs GWTC':>17s}")
for ev, a_hi, r_hi, g_hi, ess, n in rows:
    removed = (a_hi - r_hi) / max(a_hi - g_hi, 1e-9)
    print(f"{ev:10s} {a_hi:8.1f} {r_hi:11.1f} {g_hi:7.1f} "
          f"{removed*100:21.0f}% {r_hi/g_hi:16.2f}x")
print("\nESS = effective sample size out of ~3000; low ESS means the reweighted")
print("numbers are noisy and the two priors disagree strongly in that region.")
