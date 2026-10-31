"""Causal isolation: WHICH token branch (geometry vs matched-filter) carries
the local-noise-triggered collapse found in subtraction_test.py?

subtraction_test.py showed: the SAME synthetic signal collapses in noise
local to GW150914's exact segment but not in generic off-source noise. This
mirrors round 8 of the original investigation (analysis/attractor_basin/
REPORT.md), which used token grafting to prove the geometry branch carried
the (30,30) attractor causally. lean_npe_mf_scratch has TWO extra-token
branches now (geometry + matched-filter, additively combined:
gtok_final = gtok_geom + mf_gate * gtok_mf). This script grafts each branch
independently onto an otherwise-clean off-source conv representation to see
which one, alone, is sufficient to install the collapse.

4 conditions, all using the OFF-SOURCE strain's conv/energy path (the "carrier")
and varying only which strain's geometry/MF tokens are attached:
  off+off   (baseline: should be clean, matches the twin result)
  loc+loc   (baseline: should collapse, matches subtraction_test result)
  loc+off   (LOCAL geometry tokens, OFF-SOURCE MF tokens)
  off+loc   (OFF-SOURCE geometry tokens, LOCAL MF tokens)
If loc+off collapses -> geometry branch carries it.
If off+loc collapses -> MF branch carries it.
If both/neither -> distributed or carrier-dependent (would need a same-strain
conv-path control too).

Also dumps the raw geometry (tau, coherence) and MF (best-matching template
Mc/q, peak SNR profile) numbers for both encodings, and the two segments'
measured ASDs, to look for WHAT differs.
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
from _common import GW150914_TRUTH, GW150914_GPS, raw_projected_signal, asd_bands_of

CKPT = str(ROOT / "model/lean_npe_mf_scratch/best_model.pth")
DETS = ("H1", "L1")
SYN = dict(mass_1=38.8, mass_2=33.4, luminosity_distance=440.0,
          theta_jn=2.9, ra=1.95, dec=-1.27, psi=1.75, phase=1.3,
          geocent_time=0.0, a1=0.0, a2=0.0)
OFF_GPS = GW150914_GPS - 300.0


def load_model():
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
    return model


def build_local_strain(found, starts, raw0):
    true_sig = raw_projected_signal(GW150914_TRUTH, GW150914_GPS, detectors=found)
    syn_sig = raw_projected_signal(SYN, GW150914_GPS, detectors=found)
    raw = {}
    for det in found:
        v = raw0[det].copy()
        i0 = int(round((GW150914_GPS - starts[det]) * SAMPLE_RATE)) - T_LEN // 2
        v[i0:i0 + T_LEN] = v[i0:i0 + T_LEN] - true_sig[det] + syn_sig[det]
        raw[det] = v
    return raw


def build_offsource_strain(found):
    gps, raw0, starts, srs, found2 = fetch_gwosc(OFF_GPS, detectors=list(found), fetch_seconds=64)
    syn_sig = raw_projected_signal(SYN, OFF_GPS, detectors=found2)
    raw = {}
    for det in found2:
        v = raw0[det].copy()
        i0 = int(round((OFF_GPS - starts[det]) * SAMPLE_RATE)) - T_LEN // 2
        v[i0:i0 + T_LEN] = v[i0:i0 + T_LEN] + syn_sig[det]
        raw[det] = v
    return raw, gps, starts, srs, found2


@torch.no_grad()
def encode_parts(model, strain_t, ab_t):
    """Returns (gtok_geom, gtok_mf, conv_carrier_fn) -- gtok_mf already
    includes the mf_gate scale, matching forward()'s combination."""
    enc = model.encoder
    strain_c = torch.nan_to_num(strain_t, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
    g = enc.geom_mlp(enc._geometry_rel(strain_c))
    gtok_geom = enc.geom_to_tokens(g).reshape(-1, enc.n_geom_tokens, enc.d_model)
    gtok_mf = torch.zeros_like(gtok_geom)
    mf_raw = None
    if enc.use_mf_tokens:
        mf_raw = enc._mf_features(strain_c)
        m = enc.mf_mlp(mf_raw)
        gtok_mf = enc.mf_gate * enc.mf_to_tokens(m).reshape(-1, enc.n_geom_tokens, enc.d_model)
    return gtok_geom, gtok_mf, mf_raw


@torch.no_grad()
def context_from_tokens(model, carrier_strain_t, carrier_ab_t, gtok):
    enc = model.encoder
    strain_c = torch.nan_to_num(carrier_strain_t, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
    feats, _ = enc._compute_feats(strain_c, carrier_ab_t, extra_tokens=gtok)
    return enc.out_proj(feats)


@torch.no_grad()
def sample_from_context(model, ctx, rank=0, n_samples=3000):
    B = ctx.shape[0]
    r = torch.full((B,), rank, dtype=torch.long, device=ctx.device)
    full_ctx = model._full_context(ctx, r)
    ctx_rep = full_ctx.unsqueeze(1).expand(B, n_samples, full_ctx.shape[1]).reshape(B * n_samples, -1)
    z = torch.randn(B * n_samples, len(model.param_names), device=ctx.device)
    y, _ = model.flow.inverse(z, ctx_rep)
    y = model.scaler.wrap(y).reshape(B, n_samples, -1)
    return model.scaler.denormalize(y)


def summarize(samples, truth):
    m1, m2 = samples[0, :, 0], samples[0, :, 1]
    q = torch.minimum(m1, m2) / torch.maximum(m1, m2)
    return dict(m1=float(m1.median()), m2=float(m2.median()), q_med=float(q.median()),
               q95=float((q > 0.95).float().mean()))


def main():
    model = load_model()

    print("fetching GW150914 on-source raw strain ...")
    gps0, raw0, starts0, srs0, found0 = fetch_gwosc("GW150914", detectors=list(DETS))
    raw_local = build_local_strain(found0, starts0, raw0)
    prep_local = prepare_real(raw_local, gps0, starts0, srs0, seed=0)
    ab_local = asd_bands_of(prep_local)

    print("fetching off-source raw strain (-300s) ...")
    raw_off, gps1, starts1, srs1, found1 = build_offsource_strain(found0)
    prep_off = prepare_real(raw_off, gps1, starts1, srs1, seed=1)
    ab_off = asd_bands_of(prep_off)

    st_local = torch.from_numpy(prep_local.strain).unsqueeze(0).float()
    st_off = torch.from_numpy(prep_off.strain).unsqueeze(0).float()
    ab_local_t = torch.from_numpy(ab_local).unsqueeze(0).float()
    ab_off_t = torch.from_numpy(ab_off).unsqueeze(0).float()

    print("\n=== measured ASD comparison (H1, band means) ===")
    freqs = np.fft.rfftfreq(T_LEN, 1.0 / SAMPLE_RATE)
    for lo, hi, label in [(20, 60, "20-60Hz"), (60, 150, "60-150Hz"),
                          (150, 300, "150-300Hz"), (300, 512, "300-512Hz")]:
        band = (freqs >= lo) & (freqs < hi)
        loc_h1 = float(np.mean(prep_local.asds["H1"][band]))
        off_h1 = float(np.mean(prep_off.asds["H1"][band]))
        print(f"  H1 {label}: local {loc_h1:.3e}  off-source {off_h1:.3e}  "
              f"ratio {loc_h1/off_h1:.3f}")

    print("\n=== token grafting (carrier = off-source conv path) ===")
    g_local, mf_local, mfraw_local = encode_parts(model, st_local, ab_local_t)
    g_off, mf_off, mfraw_off = encode_parts(model, st_off, ab_off_t)

    combos = {
        "off_geom + off_mf (baseline clean)": (g_off, mf_off, st_off, ab_off_t),
        "loc_geom + loc_mf (baseline collapsed)": (g_local, mf_local, st_local, ab_local_t),
        "loc_geom + off_mf, off carrier": (g_local, mf_off, st_off, ab_off_t),
        "off_geom + loc_mf, off carrier": (g_off, mf_local, st_off, ab_off_t),
    }
    for label, (gg, mm, carrier, ab) in combos.items():
        gtok = gg + mm
        ctx = context_from_tokens(model, carrier, ab, gtok)
        samples = sample_from_context(model, ctx)
        r = summarize(samples, SYN)
        print(f"  [{label}] m1 {r['m1']:.1f}  m2 {r['m2']:.1f}  "
              f"q_med {r['q_med']:.3f}  q>0.95 {r['q95']:.3f}")

    if mfraw_local is not None:
        print("\n=== MF template match (best (Mc,q) grid point, per detector) ===")
        enc = model.encoder
        K = enc.mf_re.shape[1]
        mc_q = None
        bank_path = ROOT / "src/ahsd/models/assets/mf_bank.npz"
        bank = np.load(bank_path)
        mc_q = bank["mc_q"]
        for name, feats in [("local", mfraw_local), ("off-source", mfraw_off)]:
            f = feats.reshape(1, 3, K, 4)[0]
            for di, det in enumerate(("H1", "L1")):
                snr = torch.sinh(f[di, :, 0])
                k = int(snr.argmax())
                print(f"  {name} {det}: best template Mc={mc_q[k,0]:.1f} q={mc_q[k,1]:.2f} "
                      f"peak SNR={float(snr[k]):.1f}  (top-3 SNR: "
                      f"{sorted(snr.tolist(), reverse=True)[:3]})")


if __name__ == "__main__":
    main()
