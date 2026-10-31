"""Disentangle Possibility 2 (waveform mismatch) from Possibility 3 (specific
noise realization) for the lean_npe_mf_scratch GW150914 collapse.

Preprocessing is now ruled out (verify_preprocessing_path.py: byte-identical,
deterministic). The off-source twin test (real_noise_basin_check2.py) showed
that a SYNTHETIC signal (correct masses/geometry, zero-spin IMRPhenomXP) in
GENERIC real O1 noise does NOT collapse (q_med 0.44-0.51, rail 2-4%) while
the REAL GW150914 event (real signal + its own specific noise) DOES (rail
74%). Two candidate explanations remain:
  (2) waveform mismatch: the real signal's content (true phase evolution,
      spin, subtle systematics) differs from the synthetic zero-spin twin
      in a way that matters.
  (3) specific noise realization: GW150914 happened to land in a particular
      O1 noise fluctuation that the model responds to differently than
      generic off-source noise, independent of signal content.

Test: reconstruct GW150914's real signal at the published (approximate)
parameters, SUBTRACT it from the real raw strain at the correct GPS
location -> an approximate "local residual" (real noise realization AT
GW150914's exact time and segment, with most of the deterministic signal
removed). Inject the SAME plain synthetic twin signal (used in the
off-source test) into this residual instead of into distant off-source
noise, and evaluate.

  - if this now COLLAPSES like the real event -> supports (3): the local
    noise realization is the driver, independent of signal content (since
    both this test and the off-source twin use the IDENTICAL synthetic
    signal -- only the noise differs).
  - if this stays CLEAN like the off-source twin -> the local noise isn't
    special either; the residual (imperfect) subtraction and/or the real
    signal's own content is implicated -> reopens (2) / imperfect
    subtraction as confounds.

Caveat (stated up front, not just at the end): subtraction uses
GW150914_TRUTH (m1=38.8, m2=33.4, zero spin, IMRPhenomXP) as a stand-in for
the true signal -- an approximation, not the exact LVC posterior (real
spins, precession, calibration uncertainty are NOT removed). A collapsed
result here is still informative (noise realization implicated even under
imperfect subtraction); a clean result is only suggestive against (3), since
leftover subtraction residual could itself explain a clean result by
accident. Read together with the off-source twins, not in isolation.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis/gw150914_localization"))

from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES
from ahsd.inference.preprocessing import fetch_gwosc, prepare_real, T_LEN, SAMPLE_RATE
from _common import GW150914_TRUTH, GW150914_GPS, raw_projected_signal
from twin_test4 import h1l1_lag_ms

CKPT = str(ROOT / "model/lean_npe_mf_scratch/best_model.pth")
DETS = ("H1", "L1")
# the exact synthetic twin signal used in real_noise_basin_check2.py (basin
# probe BASE params), so results are directly comparable
SYN = dict(mass_1=38.8, mass_2=33.4, luminosity_distance=440.0,
          theta_jn=2.9, ra=1.95, dec=-1.27, psi=1.75, phase=1.3,
          geocent_time=0.0, a1=0.0, a2=0.0)


def run_model(model, raw, gps, starts, srs, label, truth):
    prep = prepare_real(raw, gps, starts, srs, seed=0)
    lag = h1l1_lag_ms(prep.strain)
    from _common import asd_bands_of
    ab = asd_bands_of(prep)
    st = torch.from_numpy(prep.strain).unsqueeze(0).float()
    abt = torch.from_numpy(ab).unsqueeze(0).float()
    with torch.no_grad():
        s = model.sample_posterior(st, rank=0, n_samples=3000, asd_bands=abt)
    m1, m2 = s[0, :, 0], s[0, :, 1]
    q = torch.minimum(m1, m2) / torch.maximum(m1, m2)
    yn = model.scaler.normalize(s[0])
    yt = model.scaler.normalize(torch.tensor([[truth.get(k, 0.0) for k in PARAM_NAMES]]))[0]
    noncirc = ~model.scaler.circ_mask
    spur = (((yn > 0.999) & (yt < 0.9)) | ((yn < -0.999) & (yt > -0.9)))[:, noncirc]
    rail = float(spur.any(-1).float().mean())
    print(f"  [{label}] lag {lag:+.2f}ms  m1 {float(m1.median()):.1f}  m2 {float(m2.median()):.1f}  "
          f"q_med {float(q.median()):.3f}  q>0.95 {float((q > 0.95).float().mean()):.3f}  rail {rail:.3f}")
    return dict(q_med=float(q.median()), rail=rail)


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = LeanNPE(psd_cond=a.get("psd_cond", True), psd_bands=a.get("psd_bands", 16),
                    encoder_type=a.get("encoder_type", "coherent"),
                    mass_param=a.get("mass_param", "mc_q"),
                    geom_dropout=0.0, geom_phase_residual=a.get("geom_phase_residual", False),
                    mf_tokens=a.get("mf_tokens", False),
                    mf_shape_norm=a.get("mf_shape_norm", False), mf_token_fusion=a.get("mf_token_fusion", "add"), mf_bank=a.get("mf_bank", "mf_bank.npz"), pol_features=a.get("pol_features", False), q_embed_dim=a.get("q_embed_dim", 0))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    print("fetching real GW150914 raw strain (64s) ...")
    gps, raw0, starts, srs, found = fetch_gwosc("GW150914", detectors=list(DETS))
    print(f"  gps={gps}  detectors={found}")

    print("\n=== baseline: real event, unmodified ===")
    run_model(model, {d: raw0[d].copy() for d in found}, gps, starts, srs,
             "real GW150914", GW150914_TRUTH)

    print("\nreconstructing approximate true signal (GW150914_TRUTH, zero-spin "
          "IMRPhenomXP) for subtraction ...")
    true_sig = raw_projected_signal(GW150914_TRUTH, GW150914_GPS, detectors=found)

    def place(det):
        return int(round((GW150914_GPS - starts[det]) * SAMPLE_RATE)) - T_LEN // 2

    print("\n=== residual: real strain MINUS reconstructed true signal ===")
    raw_resid = {}
    for det in found:
        v = raw0[det].copy()
        i0 = place(det)
        v[i0:i0 + T_LEN] = v[i0:i0 + T_LEN] - true_sig[det]
        raw_resid[det] = v
    run_model(model, {d: raw_resid[d].copy() for d in found}, gps, starts, srs,
             "residual only (no injection)", GW150914_TRUTH)

    print("\n=== residual + SAME synthetic twin signal used in the off-source test ===")
    syn_sig = raw_projected_signal(SYN, GW150914_GPS, detectors=found)
    raw_test = {}
    for det in found:
        v = raw_resid[det].copy()
        i0 = place(det)
        v[i0:i0 + T_LEN] = v[i0:i0 + T_LEN] + syn_sig[det]
        raw_test[det] = v
    r_local = run_model(model, {d: raw_test[d].copy() for d in found}, gps, starts, srs,
                        "residual + synthetic twin (LOCAL noise)", SYN)

    print("\ncomparison:")
    print("  off-source twin (real_noise_basin_check2.py, 3 trials): "
          "q_med 0.44-0.51, rail 0.02-0.04  [generic O1 noise]")
    print(f"  this test (LOCAL noise at GW150914's exact time): "
          f"q_med {r_local['q_med']:.3f}, rail {r_local['rail']:.3f}")
    print("  real GW150914 event: q_med ~1.0, rail 0.740  [real signal + local noise]")


if __name__ == "__main__":
    main()
