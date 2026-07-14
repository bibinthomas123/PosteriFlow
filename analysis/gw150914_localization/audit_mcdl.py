"""
AUDIT - is the Mc-dL R^2 ~ 0.81 a real dataset property or a probe artifact?

(A) raw ParameterSampler draws (no selection): are Mc and dL independent?
(B) apply the generator's SNR gate: does it induce the correlation?
(C) stored training/validation data: measure directly, stratify by true SNR.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ahsd.data.parameter_sampler import ParameterSampler

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]


def corr(mc, dL):
    l1, l2 = np.log(mc), np.log(dL)
    return dict(pearson_log=round(float(pearsonr(l1, l2)[0]), 3),
                r2_log=round(float(pearsonr(l1, l2)[0] ** 2), 3),
                spearman=round(float(spearmanr(l1, l2).correlation), 3),
                pearson_linear=round(float(pearsonr(mc, dL)[0]), 3), n=len(mc))


report = {}

# ── (A) raw sampler draws, NO selection ──────────────────────────────────────
smp = ParameterSampler()
N = 60000
rows = [smp.sample_parameters("BBH") for _ in range(N)]
mc = np.array([r["chirp_mass"] for r in rows])
dL = np.array([r["luminosity_distance"] for r in rows])
report["A_raw_sampler_unfiltered"] = corr(mc, dL)
print("A) raw ParameterSampler (no SNR gate):", report["A_raw_sampler_unfiltered"])

# ── (B) apply the SNR gate using the REAL injector SNR (with geometry) ────────
from ahsd.data.bilby_pipeline import (BilbyNoiseGenerator, BilbySignalInjector,
                                      get_default_psd)
inj = BilbySignalInjector(4096, 4.0)
psds = {d: get_default_psd(d) for d in ["H1", "L1", "V1"]}
Nb = 3000
snr = np.zeros(Nb)
for i in range(Nb):
    p = rows[i]
    zeros = np.zeros(int(4096 * 4.0))
    net2 = 0.0
    for det in ["H1", "L1", "V1"]:
        _, s = inj.inject(zeros, p, det, psds[det])
        net2 += float(s) ** 2
    snr[i] = np.sqrt(net2)
mcb = mc[:Nb]; dLb = dL[:Nb]
report["B_injector_snr_range"] = [round(float(snr.min()), 1), round(float(snr.max()), 1)]
report["B_unfiltered_subset"] = corr(mcb, dLb)
for thr in (6.0, 8.0, 12.0, 20.0):
    m = snr >= thr
    if m.sum() > 50:
        report[f"B_after_SNRgate_ge{int(thr)}"] = {**corr(mcb[m], dLb[m]),
                                                   "frac_kept": round(float(m.mean()), 3)}
        print(f"B) after SNR>={thr:>4}: {report[f'B_after_SNRgate_ge{int(thr)}']}")

# ── (C) stored data: true optimal SNR from whitened signals ──────────────────
def stored_snr_and_params(split, nmax=None):
    p = np.load(ROOT / f"data/dataset/memmap/{split}/params.npy")
    sig = np.load(ROOT / f"data/dataset/memmap/{split}/signals.npy", mmap_mode="r")
    n = len(p) if nmax is None else min(nmax, len(p))
    snr = np.empty(n)
    for i0 in range(0, n, 2000):
        i1 = min(i0 + 2000, n)
        chunk = np.asarray(sig[i0:i1], dtype=np.float32)
        snr[i0:i1] = np.sqrt((chunk ** 2).sum(axis=(1, 2)))  # ||whitened signal||
    return p[:n], snr

pv, sv = stored_snr_and_params("validation")
mcv = (pv[:, 0] * pv[:, 1]) ** 0.6 / (pv[:, 0] + pv[:, 1]) ** 0.2
dLv = pv[:, 2]
report["C_stored_val_snr_floor"] = round(float(np.percentile(sv, 0.5)), 1)
report["C_stored_val_snr_range"] = [round(float(sv.min()), 1), round(float(sv.max()), 1)]
report["C_stored_val_full"] = corr(mcv, dLv)
report["C_stored_val_SNRgt20"] = corr(mcv[sv > 20], dLv[sv > 20])
bins = [(6, 10), (10, 15), (15, 25), (25, 200)]
report["C_stored_val_snr_binned"] = {}
for lo, hi in bins:
    m = (sv >= lo) & (sv < hi)
    if m.sum() > 50:
        report["C_stored_val_snr_binned"][f"{lo}-{hi}"] = {**corr(mcv[m], dLv[m]),
                                                          "n": int(m.sum())}
print("\nC) stored validation:")
print("   full        :", report["C_stored_val_full"])
print("   SNR>20      :", report["C_stored_val_SNRgt20"])
for k, v in report["C_stored_val_snr_binned"].items():
    print(f"   SNR {k:8s}:", v)

json.dump(report, open(OUT / "audit_mcdl.json", "w"), indent=2, default=float)

# ── scatter plots ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].scatter(mc, dL, s=3, alpha=0.15, color="#999999")
ax[0].set_title(f"(A) raw sampler, no selection\nPearson(logMc,logdL)={report['A_raw_sampler_unfiltered']['pearson_log']}")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("chirp mass"); ax[0].set_ylabel("luminosity distance [Mpc]")

sc = ax[1].scatter(mcb, dLb, s=6, c=snr, cmap="viridis", vmin=4, vmax=40)
# overlay SNR=8 selection locus  dL ~ Mc^(5/6)
mm = np.linspace(mcb.min(), mcb.max(), 100)
for thr, K in [(8, None)]:
    keep = snr >= thr
    ax[1].scatter(mcb[keep], dLb[keep], s=10, edgecolor="k", facecolor="none", alpha=0.25)
ax[1].set_title(f"(B) injector SNR; kept SNR>=8 (circled)\nafter cut Pearson={report.get('B_after_SNRgate_ge8',{}).get('pearson_log','-')}")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("chirp mass"); ax[1].set_ylabel("dL [Mpc]")
plt.colorbar(sc, ax=ax[1], label="network SNR")

sc2 = ax[2].scatter(mcv, dLv, s=3, c=np.clip(sv, 4, 50), cmap="viridis", alpha=0.5)
ax[2].set_title(f"(C) stored validation set (real gate)\nPearson(logMc,logdL)={report['C_stored_val_full']['pearson_log']}")
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("chirp mass"); ax[2].set_ylabel("dL [Mpc]")
plt.colorbar(sc2, ax=ax[2], label="true SNR (clip)")
fig.tight_layout(); fig.savefig(OUT / "audit_mcdl_scatter.png", dpi=110)
print(f"\nwrote {OUT/'audit_mcdl.json'} and audit_mcdl_scatter.png")
