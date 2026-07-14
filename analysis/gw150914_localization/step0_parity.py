"""
STEP 0 - prior & pipeline parity between LeanNPE and dynesty on GW150914.

Verifies both methods solve the same inference problem and dumps/overlays the
intermediate arrays (PSD, whitened strain, matched-filter SNR).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C

OUT = Path(__file__).resolve().parent
report = {}

# ── prepare event through the LeanNPE pipeline ───────────────────────────────
model = C.load_model()
prep = C.prepare_gw150914()
report["detectors_present"] = prep.detectors_present
report["preproc_warnings"] = prep.warnings
report["preproc_quality"] = {d: {k: round(float(v), 4) for k, v in q.items()}
                             for d, q in prep.quality.items()}

# LeanNPE posterior (canonical model)
s = C.sample(model, prep, n=6000)
report["leannpe_dL"] = C.summ(s[:, C.JD])
report["leannpe_m1"] = C.summ(s[:, 0])
report["leannpe_m2"] = C.summ(s[:, 1])

# ── (A) network matched-filter SNR on the SAME whitened input ────────────────
# at published GW150914 truth and at the dynesty MAP
mf_truth = C.network_mf_snr(prep, C.GW150914_TRUTH)
report["mf_snr_at_truth"] = mf_truth

# ── (B) load the existing dynesty benchmark result for GW150914 ──────────────
bench = C.ROOT / "results" / "real_event_benchmark" / "GW150914"
dyn_summary = None
try:
    allb = json.load(open(bench.parent / "summary.json"))
    dyn_summary = next(e for e in allb if e["event"] == "GW150914")
except Exception as e:
    dyn_summary = {"error": str(e)}
report["dynesty_benchmark_comparison"] = dyn_summary.get("comparison", dyn_summary)

# dynesty posterior samples + its PSD, from the bilby result json
dyn_dL = dyn_psd = None
try:
    br = json.load(open(bench / "bilby" / "GW150914_result.json"))
    post = br.get("posterior", {}).get("content", {})
    if "luminosity_distance" in post:
        dyn_dL = np.asarray(post["luminosity_distance"], float)
    # PSD stored per-interferometer under meta_data / interferometers
    md = br.get("meta_data", {}) or {}
    dyn_psd = md
except Exception as e:
    report["dynesty_json_error"] = str(e)

if dyn_dL is not None:
    report["dynesty_dL"] = C.summ(dyn_dL)
    # network SNR that dynesty reconstructs (from posterior meta if present)

# ── (C) PSD parity: LeanNPE whitening ASD^2 vs dynesty's consumed PSD ─────────
# dynesty consumes result.prepared.psds (== LeanNPE whitening PSD) BY CONSTRUCTION
# (dynesty_bridge.run_comparison passes result.prepared.psds). Re-verify the
# benchmark's stored PSD matches the current pipeline PSD.
psd_parity = {}
try:
    ifos = (json.load(open(bench / "bilby" / "GW150914_result.json"))
            .get("meta_data", {}).get("likelihood", {}).get("interferometers", {}))
    for det in prep.detectors_present:
        cur_freqs = prep.psds[det]["frequencies"]
        cur_psd = np.asarray(prep.psds[det]["psd"], float)
        info = ifos.get(det, {})
        if "power_spectral_density" in info:
            bfreq = np.asarray(info["power_spectral_density"].get("frequency_array", []), float)
            bpsd = np.asarray(info["power_spectral_density"].get("psd_array", []), float)
            if bfreq.size and bpsd.size:
                bi = np.interp(cur_freqs, bfreq, bpsd)
                band = (cur_freqs >= 20) & (cur_freqs <= 1024)
                frac = np.abs(bi[band] - cur_psd[band]) / (cur_psd[band] + 1e-40)
                psd_parity[det] = dict(median_frac_diff=float(np.median(frac)),
                                       p95_frac_diff=float(np.percentile(frac, 95)))
        report.setdefault("dynesty_ifo_meta_keys", list(info.keys()))
except Exception as e:
    psd_parity["error"] = str(e)
report["psd_parity_bench_vs_current"] = psd_parity

# ── (D) prior parity: reveal LeanNPE's learned distance prior ─────────────────
# Feed PURE white noise (no signal) with GW150914-like asd_bands => the model's
# learned conditional prior. Compare to training d^2 and dynesty UniformSourceFrame.
rng = np.random.default_rng(0)
noise = rng.standard_normal((3, C.T_LEN)).astype(np.float32)
ab = C.asd_bands_of(prep)
zero_ab = np.zeros_like(ab)                       # design-like sensitivity
noise_gw_ab = C.sample_from_strain(model, noise, ab, n=6000)
noise_design_ab = C.sample_from_strain(model, noise, zero_ab, n=6000)
report["learned_prior_dL_noise_gw150914_asdbands"] = C.summ(noise_gw_ab[:, C.JD])
report["learned_prior_dL_noise_design_asdbands"] = C.summ(noise_design_ab[:, C.JD])

# analytic training priors for reference (d^2 on [40,2200] vs UniformSourceFrame)
dd = np.linspace(40, 2200, 100000)
w_d2 = dd ** 2
med_d2 = dd[np.searchsorted(np.cumsum(w_d2) / w_d2.sum(), 0.5)]
report["analytic_prior_medians"] = {
    "training_d2_[40,2200]": float(med_d2),
    "note": "dynesty uses UniformSourceFrame(50,2100); LeanNPE trained ~d^2"}

# ── plots: PSD + whitened strain overlay ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for det, axc in zip(prep.detectors_present, [axes[0, 0], axes[0, 1]]):
    f = prep.psds[det]["frequencies"]
    axc.loglog(f, np.sqrt(prep.psds[det]["psd"]), label=f"{det} measured ASD (whitening)")
    axc.set_xlim(15, 2048); axc.set_title(f"{det} ASD"); axc.legend(fontsize=8)
    axc.set_xlabel("Hz")
# whitened strain, zoom on merger (center)
t = np.arange(C.T_LEN) / C.SR - C.DUR / 2
for i, det in enumerate(prep.detectors_present):
    ax = axes[1, i]
    ax.plot(t, prep.strain[i], lw=0.5)
    ax.set_xlim(-0.25, 0.1); ax.set_title(f"{det} whitened (model input), zoom")
    ax.set_xlabel("s from trigger")
fig.tight_layout()
fig.savefig(OUT / "step0_psd_strain.png", dpi=100)

json.dump(report, open(OUT / "step0_parity.json", "w"), indent=2, default=float)

# ── console table ────────────────────────────────────────────────────────────
print("\n===== STEP 0 : PARITY =====")
print(f"detectors present: {prep.detectors_present}")
print(f"preproc warnings: {prep.warnings}")
print("\n-- network matched-filter SNR on LeanNPE's whitened input (at truth) --")
for k, v in mf_truth.items():
    print(f"  {k:9s} optimal={v['optimal_snr']:6.2f}  mf={v['mf_snr']:6.2f}")
print(f"\n  (published GW150914 network SNR ~= 24)")
print("\n-- distance recovery --")
print(f"  LeanNPE dL   median={report['leannpe_dL']['median']:.0f}  ci90={[round(x) for x in report['leannpe_dL']['ci90']]}")
if dyn_dL is not None:
    print(f"  dynesty dL   median={report['dynesty_dL']['median']:.0f}  ci90={[round(x) for x in report['dynesty_dL']['ci90']]}")
print(f"  truth 440")
print("\n-- LeanNPE learned prior (pure noise input) --")
print(f"  dL | GW150914 asd_bands : median={report['learned_prior_dL_noise_gw150914_asdbands']['median']:.0f}")
print(f"  dL | design   asd_bands : median={report['learned_prior_dL_noise_design_asdbands']['median']:.0f}")
print(f"  analytic d^2[40,2200] median={med_d2:.0f}")
print("\n-- PSD parity (benchmark vs current pipeline) --")
print(f"  {psd_parity}")
print(f"\nwrote {OUT/'step0_parity.json'}")
