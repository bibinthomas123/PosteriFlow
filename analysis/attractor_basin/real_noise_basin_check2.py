"""Diagnostic v2 (corrected): does the basin probe (design noise) predict
real-event behavior for lean_npe_mf_scratch?

v1 (real_noise_basin_check.py) had a UNITS BUG: it recolored the RAW
(unwhitened) output of basin_probe._projected() as if it were already
design-whitened, then added it to noise_bank's pre-whitened real strain.
That's an amplitude-scale mismatch (off by ~the ASD itself across
frequency), not a real test — it produced garbage masses (m1~13, m2~6),
not a genuine "resolved" result. Retracted.

v2 uses the ALREADY-VALIDATED twin methodology from
analysis/attractor_basin/twin_test5.py (round 5, the one that first
reproduced the collapse with a known-truth injection): inject the raw
delay-correct signal directly into REAL unwhitened off-source GWOSC strain,
then let ahsd.inference.preprocessing.prepare_real do the whitening with
that segment's own measured PSD. No manual recoloring, no unit risk.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis/gw150914_localization"))
sys.path.insert(0, str(ROOT / "analysis/attractor_basin"))

from ahsd.evaluation.basin_probe import BASE
from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES
from twin_test5 import inject_real_delayed, det_delays
from twin_test4 import h1l1_lag_ms
from _common import asd_bands_of

CKPT = ROOT / "model/lean_npe_mf_scratch/best_model.pth"
GW150914_GPS = 1126259462.4


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = LeanNPE(psd_cond=a.get("psd_cond", True), psd_bands=a.get("psd_bands", 16),
                    encoder_type=a.get("encoder_type", "coherent"),
                    mass_param=a.get("mass_param", "mc_q"),
                    geom_dropout=0.0,
                    geom_phase_residual=a.get("geom_phase_residual", False),
                    mf_tokens=a.get("mf_tokens", False),
                    mf_shape_norm=a.get("mf_shape_norm", False), mf_token_fusion=a.get("mf_token_fusion", "add"), mf_bank=a.get("mf_bank", "mf_bank.npz"), pol_features=a.get("pol_features", False), q_embed_dim=a.get("q_embed_dim", 0))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    truth = dict(BASE, geocent_time=0.0)
    dts = det_delays(truth, GW150914_GPS)
    print(f"pseudo-truth sky delays at GW150914 GPS: "
          f"{ {k: round(v*1000, 2) for k, v in dts.items()} } ms")
    print(f"truth: m1={truth['mass_1']} m2={truth['mass_2']} "
          f"(q={truth['mass_2']/truth['mass_1']:.3f}) dL={truth['luminosity_distance']}")

    offsets = [-300.0, -600.0, +300.0]
    for off in offsets:
        off_gps = GW150914_GPS + off
        prep, _ = inject_real_delayed(off_gps, truth, sign=+1, seed=int(abs(off)))
        lag = h1l1_lag_ms(prep.strain)
        print(f"\n=== off-source {off:+.0f}s (GPS {off_gps:.1f}) ===")
        print(f"  measured H1-L1 lag: {lag:+.2f} ms (target +6.84)")

        st = torch.from_numpy(prep.strain).unsqueeze(0).float()
        ab = asd_bands_of(prep)
        abt = torch.from_numpy(ab).unsqueeze(0).float()
        with torch.no_grad():
            s = model.sample_posterior(st, rank=0, n_samples=2000, asd_bands=abt)
        m1, m2 = s[0, :, 0], s[0, :, 1]
        q = torch.minimum(m1, m2) / torch.maximum(m1, m2)
        yn = model.scaler.normalize(s[0])
        yt = model.scaler.normalize(torch.tensor([[truth.get(k, 0.0) for k in PARAM_NAMES]]))[0]
        noncirc = ~model.scaler.circ_mask
        spur = (((yn > 0.999) & (yt < 0.9)) | ((yn < -0.999) & (yt > -0.9)))[:, noncirc]
        rail = float(spur.any(-1).float().mean())
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        mct = (truth["mass_1"] * truth["mass_2"]) ** 0.6 / (truth["mass_1"] + truth["mass_2"]) ** 0.2
        print(f"  m1_med {float(m1.median()):.1f}  m2_med {float(m2.median()):.1f}  "
              f"q_med {float(q.median()):.3f}  q>0.95 {float((q > 0.95).float().mean()):.3f}  "
              f"rail {rail:.3f}  Mc_err {float(mc.median())/mct - 1.0:+.3f}")

    print("\n(basin probe / design noise, snr85 row: q_med 0.325, q>0.95 0.02, rail 0.016, Mc_err -0.021)")
    print("(real GW150914 (GWOSC strain, includes actual event): m1 31.0 m2 30.9 rail 0.740)")


if __name__ == "__main__":
    main()
