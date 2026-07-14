"""
STEP 4 - semi-real injection.

Inject a KNOWN synthetic IMRPhenomXP signal (GW150914-like params) into
OFF-SOURCE real O1 strain around GW150914, whiten with the SAME prepare_real
path, and run LeanNPE.

  biased   -> failure lives in the real-noise domain
  recovered-> failure is specific to the actual GW150914 signal

Controls:
  (R) real GW150914 itself
  (S) same synthetic signal in DESIGN Gaussian noise (training domain)
  multiple off-source epochs to average over noise realisation
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C

OUT = Path(__file__).resolve().parent
model = C.load_model()

# GW150914-like injection params (detector-frame pseudo-truth)
P = dict(C.GW150914_TRUTH)
report = {"inj_params": P}

# ── (R) real GW150914 ────────────────────────────────────────────────────────
prep_real = C.prepare_gw150914()
s_real = C.sample(model, prep_real, n=6000)
mf_real = C.network_mf_snr(prep_real, P)
report["real_gw150914"] = dict(dL=C.summ(s_real[:, C.JD]),
                               m1=C.summ(s_real[:, 0]), m2=C.summ(s_real[:, 1]),
                               net_mf_snr=mf_real["network"]["mf_snr"])

# ── (I) semi-real injections into off-source O1 noise ────────────────────────
# off-source epochs: comfortably clear of the event (>200 s away, both directions)
offsets = [-400, -300, -200, 200, 300]
inj_runs = []
for off in offsets:
    off_gps = C.GW150914_GPS + off
    try:
        prep_inj, snr = C.inject_into_real_noise(off_gps, P)
        s = C.sample(model, prep_inj, n=6000)
        rec = dict(off_s=off, net_mf_snr=round(snr["network"]["mf_snr"], 2),
                   dL=C.summ(s[:, C.JD]), m1=C.summ(s[:, 0]), m2=C.summ(s[:, 1]),
                   asd_bands_mean=float(np.mean(C.asd_bands_of(prep_inj)[:2])),
                   warnings=prep_inj.warnings)
        inj_runs.append(rec)
        print(f"  off {off:+5d}s  SNR={rec['net_mf_snr']:5.1f}  "
              f"dL med={rec['dL']['median']:6.0f} ci90={[round(x) for x in rec['dL']['ci90']]}")
    except Exception as e:
        print(f"  off {off:+5d}s  FAILED: {str(e)[:100]}")
        inj_runs.append(dict(off_s=off, error=str(e)))
report["semireal_injections"] = inj_runs

# ── (S) same signal in DESIGN Gaussian noise (training domain reference) ──────
# Use the bilby_pipeline design noise + injector, whiten with design PSD.
from ahsd.data.bilby_pipeline import (BilbyNoiseGenerator, BilbySignalInjector,
                                      BilbyPreprocessor, get_default_psd)
prep_bp = BilbyPreprocessor(C.SR, C.DUR)
noise_gen = BilbyNoiseGenerator(C.SR, C.DUR)
injector = BilbySignalInjector(C.SR, C.DUR)
psds = {d: get_default_psd(d) for d in ["H1", "L1", "V1"]}
design_runs = []
for seed in range(4):
    white = {}
    snrs = []
    for det in ["H1", "L1"]:
        noise = noise_gen.generate(det, psds[det], seed=1000 + seed)
        strain, snr = injector.inject(noise, {**P, "a1": 0.0, "a2": 0.0}, det, psds[det])
        white[det] = prep_bp.preprocess(strain, psds[det], detector=det)
        snrs.append(snr)
    rng = np.random.default_rng(seed)
    strain3 = np.stack([white["H1"], white["L1"], rng.standard_normal(C.T_LEN).astype(np.float32)])
    zero_ab = np.zeros((3, 16), dtype=np.float32)   # design-whitened -> asd_bands 0
    s = C.sample_from_strain(model, strain3, zero_ab, n=6000)
    net = float(np.sqrt(sum(x ** 2 for x in snrs)))
    design_runs.append(dict(seed=seed, net_snr=round(net, 2), dL=C.summ(s[:, C.JD]),
                            m1=C.summ(s[:, 0])))
    print(f"  design seed {seed}  SNR={net:5.1f}  dL med={design_runs[-1]['dL']['median']:6.0f}")
report["design_injections"] = design_runs

json.dump(report, open(OUT / "step4_semireal.json", "w"), indent=2, default=float)

# ── summary table ────────────────────────────────────────────────────────────
print("\n===== STEP 4 : SEMI-REAL INJECTION =====")
print(f"injection truth dL = {P['luminosity_distance']} Mpc,  m1={P['mass_1']} m2={P['mass_2']}")
print(f"\n{'scenario':32s} {'SNR':>6s} {'dL med':>8s} {'dL/true':>8s} {'ci90':>16s}")
r = report["real_gw150914"]
print(f"{'REAL GW150914':32s} {r['net_mf_snr']:6.1f} {r['dL']['median']:8.0f} "
      f"{r['dL']['median']/440:8.2f} {str([round(x) for x in r['dL']['ci90']]):>16s}")
for rec in inj_runs:
    if "error" in rec:
        continue
    print(f"{'semi-real O1 off '+str(rec['off_s'])+'s':32s} {rec['net_mf_snr']:6.1f} "
          f"{rec['dL']['median']:8.0f} {rec['dL']['median']/440:8.2f} "
          f"{str([round(x) for x in rec['dL']['ci90']]):>16s}")
for rec in design_runs:
    print(f"{'design gaussian seed '+str(rec['seed']):32s} {rec['net_snr']:6.1f} "
          f"{rec['dL']['median']:8.0f} {rec['dL']['median']/440:8.2f} "
          f"{str([round(x) for x in rec['dL']['ci90']]):>16s}")
inj_ok = [rec for rec in inj_runs if "error" not in rec]
if inj_ok:
    med_ratio = np.median([rec["dL"]["median"] / 440 for rec in inj_ok])
    print(f"\nmedian(semi-real dL/true) = {med_ratio:.2f}")
print(f"\nwrote {OUT/'step4_semireal.json'}")
