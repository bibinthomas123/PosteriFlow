"""Offline MF-token go/no-go probe (docs/mf_token_encoder_design.md).

Question: in the quiet basin cell {GW150914 sky (+7 ms), Mc ~= 31, SNR ~ 24,
H1+L1}, which feature set linearly decodes mass ratio?

  mf       — matched-filter overlap features against a 120-template (Mc, q)
             grid (the proposed encoder addition)
  geom     — the coherent encoder's ACTUAL geometry vector
             (CoherentEncoder._geometry_rel on the same strain)
  context  — the frozen pslope2 encoder context (256-d). If this already
             decodes q, the encoder is fine and the flow is the problem —
             MF tokens would be the wrong surgery.
  snr_ctrl — whitened-norm-only control (catches amplitude leakage: at fixed
             Mc and distance, SNR varies slightly with q).

240 injections: q uniform in [0.92, 1.0] (n=120, class 1) and [0.40, 0.75]
(n=120, class 0); everything else frozen at the probe-bank BASE (incl.
distance 1564 Mpc -> SNR ~ 24); fresh design-noise seed each. Readouts:
5-fold CV AUC (logistic, L2) and ridge R^2 for continuous q.

Decision rule (design doc + reviewer): implement MF tokens ONLY if
mf AUC >= 0.8 AND mf clearly beats context (>= +0.15 AUC).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ahsd.evaluation.basin_probe import (BASE, DETS, GPS_REF, T_LEN, SR, DUR,
                                         _projected, _design_noise)
from ahsd.inference.preprocessing import prepare_simulated
from ahsd.data.bilby_pipeline import get_default_psd

OUT = Path(__file__).resolve().parent
MC_BASE = (BASE["mass_1"] * BASE["mass_2"]) ** 0.6 / (BASE["mass_1"] + BASE["mass_2"]) ** 0.2
F_LO, F_HI = 20.0, 1024.0
WIN_S = 0.3          # peak-search half-window around the merger (window centre)


def masses_from_mc_q(mc, q):
    M = mc * (1.0 + q) ** 1.2 / q ** 0.6
    return M / (1.0 + q), M * q / (1.0 + q)


# ---------------------------------------------------------------- injections
def build_injections(n_per_class=120, dist=1564.0, seed0=5000):
    rng = np.random.default_rng(42)
    qs = np.concatenate([rng.uniform(0.92, 1.00, n_per_class),
                         rng.uniform(0.40, 0.75, n_per_class)])
    strains, labels = [], []
    for i, q in enumerate(qs):
        m1, m2 = masses_from_mc_q(MC_BASE, float(q))
        p = dict(BASE, mass_1=m1, mass_2=m2, luminosity_distance=dist)
        sig = _projected(p, +1)
        raw = {det: _design_noise(det, seed0 + i * 10 + k) + sig[det]
               for k, det in enumerate(DETS)}
        prep = prepare_simulated(raw, seed=seed0 + i)
        strains.append(prep.strain)
        labels.append(float(q))
        if (i + 1) % 40 == 0:
            print(f"  injections {i + 1}/{len(qs)}", flush=True)
    return np.stack(strains).astype(np.float32), np.array(labels)


# ---------------------------------------------------------------- MF bank
def build_bank(n_mc=20, q_grid=(1.0, 0.85, 0.7, 0.55, 0.4, 0.25)):
    """Design-whitened, unit-norm FD templates per detector: [D, K, Nf_band]."""
    import bilby
    freqs = np.fft.rfftfreq(T_LEN, 1.0 / SR)
    band = (freqs >= F_LO) & (freqs < F_HI)
    asd = {}
    for det in DETS:
        p = get_default_psd(det)
        asd[det] = np.sqrt(np.interp(freqs, np.asarray(p["frequencies"], float),
                                     np.asarray(p["psd"], float),
                                     left=np.inf, right=np.inf))
    wfg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=SR,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={"waveform_approximant": "IMRPhenomXP",
                            "reference_frequency": 50.0, "minimum_frequency": F_LO})
    bank = {det: [] for det in DETS}
    mcs = np.geomspace(12.0, 45.0, n_mc)
    kk = 0
    for mc in mcs:
        for q in q_grid:
            m1, m2 = masses_from_mc_q(float(mc), float(q))
            hp = wfg.frequency_domain_strain(
                dict(mass_1=m1, mass_2=m2, luminosity_distance=500.0,
                     theta_jn=0.0, psi=0.0, phase=0.0, a_1=0.0, a_2=0.0,
                     tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0,
                     ra=0.0, dec=0.0, geocent_time=0.0))["plus"]
            for det in DETS:
                hw = (hp / asd[det])[band]
                hw = hw / np.sqrt((np.abs(hw) ** 2).sum())
                bank[det].append(hw.astype(np.complex64))
            kk += 1
    print(f"  bank: {kk} templates x {len(DETS)} dets", flush=True)
    return {det: np.stack(v) for det, v in bank.items()}, band, mcs


def mf_features(strains, bank, band):
    """Per (det, template): peak |z| in the merger window, refined via the
    full-resolution overlap, plus cos/sin of phase and peak time. [N, D*K*4]"""
    N = strains.shape[0]
    ctr = T_LEN // 2
    w = int(WIN_S * SR)
    feats = []
    for i in range(N):
        row = []
        for di, det in enumerate(DETS):
            D = np.fft.rfft(strains[i, di].astype(np.float64))[band]      # [Nf]
            # analytic overlap: positive-frequency product only, zero-padded
            # negative half -> complex z(t) whose |.| is the envelope
            # (K x len-16384 iffts are affordable at N=240 offline)
            K = bank[det].shape[0]
            full = np.zeros((K, T_LEN // 2 + 1), dtype=np.complex128)
            full[:, np.where(band)[0]] = D[None, :] * np.conj(bank[det])
            z = np.fft.ifft(full, n=T_LEN, axis=1)
            zwin = z[:, ctr - w: ctr + w]
            k_t = np.argmax(np.abs(zwin), axis=1)
            zpk = zwin[np.arange(K), k_t]
            row += [np.abs(zpk), np.cos(np.angle(zpk)), np.sin(np.angle(zpk)),
                    (k_t - w) / SR]
        feats.append(np.concatenate(row))
        if (i + 1) % 60 == 0:
            print(f"  mf {i + 1}/{N}", flush=True)
    return np.stack(feats)


# ---------------------------------------------------------------- encoder sets
def encoder_features(strains):
    import torch
    from ahsd.models.lean_npe import LeanNPE
    ck = torch.load(ROOT / "model/lean_npe_pslope2/best_model.pth",
                    map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = LeanNPE(psd_cond=a.get("psd_cond", True),
                    psd_bands=a.get("psd_bands", 16),
                    encoder_type=a.get("encoder_type", "coherent"),
                    mass_param=a.get("mass_param", "mc_q"),
                    geom_dropout=a.get("geom_dropout", 0.0),
                    geom_phase_residual=a.get("geom_phase_residual", False),
                    mf_tokens=a.get("mf_tokens", False),
                    mf_shape_norm=a.get("mf_shape_norm", False), mf_token_fusion=a.get("mf_token_fusion", "add"), q_embed_dim=a.get("q_embed_dim", 0))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    xs = torch.from_numpy(strains).float()
    ab = torch.zeros(xs.shape[0], 3, a.get("psd_bands", 16))
    ctx, geom = [], []
    with torch.no_grad():
        for i in range(0, xs.shape[0], 32):
            b = xs[i:i + 32]
            ctx.append(model.encode(b, ab[: b.shape[0]]).numpy())
            geom.append(model.encoder._geometry_rel(b).numpy())
    return np.concatenate(ctx), np.concatenate(geom)


# ---------------------------------------------------------------- probes
def probe(X, y_cls, y_q, name):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.03, max_iter=5000))
    auc = cross_val_score(clf, X, y_cls, scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    reg = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    r2 = cross_val_score(reg, X, y_q, scoring="r2",
                         cv=KFold(5, shuffle=True, random_state=0)).mean()
    print(f"{name:10s}  dim {X.shape[1]:5d}   AUC {auc:.3f}   q-R2 {r2:+.3f}", flush=True)
    return auc, r2


def main():
    print("== injections ==", flush=True)
    strains, q = build_injections()
    y = (q > 0.9).astype(int)
    print("== MF bank ==", flush=True)
    bank, band, _ = build_bank()
    print("== MF overlaps ==", flush=True)
    Xmf = mf_features(strains, bank, band)
    print("== encoder passes (pslope2 frozen) ==", flush=True)
    Xctx, Xgeom = encoder_features(strains)
    Xsnr = np.sqrt((strains.astype(np.float64) ** 2).sum(axis=(1, 2)))[:, None]

    print("\n== results (5-fold CV) ==", flush=True)
    res = {}
    res["snr_ctrl"] = probe(Xsnr, y, q, "snr_ctrl")
    res["geom"] = probe(Xgeom, y, q, "geom")
    res["context"] = probe(Xctx, y, q, "context")
    res["mf"] = probe(Xmf, y, q, "mf")
    np.savez(OUT / "mf_probe_results.npz", q=q,
             auc={k: v[0] for k, v in res.items()},
             Xmf=Xmf.astype(np.float32), Xctx=Xctx.astype(np.float32))
    go = res["mf"][0] >= 0.8 and res["mf"][0] >= res["context"][0] + 0.15
    print(f"\nGO/NO-GO (mf>=0.8 and mf>=context+0.15): "
          f"{'GO — implement MF tokens' if go else 'NO-GO'}", flush=True)


if __name__ == "__main__":
    main()
