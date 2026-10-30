"""GW150914 twin test on the mcq model (+ old m1m2 model as reference).

Conditions:
  real      — actual GWOSC GW150914 strain (control; expect the (30,30) collapse)
  twin      — published params injected into Gaussian design noise, 6 seeds
  twin-edge — same but theta_jn = pi/2 (edge-on), 2 seeds

Both models sample every condition. Reports median m1/m2/Mc/q, frac(q>0.95),
spurious railing (trainer's definition), and which params rail.
"""
import sys
from pathlib import Path

ROOT = Path("/Users/bibinthomas/Developer/PosteriFlow")
sys.path.insert(0, str(ROOT / "analysis" / "gw150914_localization"))

import numpy as np
import torch

import _common as C
from ahsd.data.bilby_pipeline import get_default_psd
from ahsd.inference import preprocessing as pp
from ahsd.models.lean_npe import PARAM_NAMES

MODELS = {
    "mcq": str(ROOT / "model" / "lean_npe_mcq" / "best_model.pth"),
    "m1m2-old": str(ROOT / "model" / "lean_npe" / "best_model.pth"),
}
N_SAMP = 4000
DETS = ("H1", "L1")


def design_noise(det, seed):
    """Raw colored Gaussian noise from the design PSD via bilby (same PSDs as
    the dataset pipeline)."""
    import bilby
    p = get_default_psd(det)
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=np.asarray(p["frequencies"], dtype=float),
        psd_array=np.asarray(p["psd"], dtype=float))
    np.random.seed(seed)
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=C.SR, duration=C.DUR, start_time=0.0)
    return np.asarray(ifo.strain_data.time_domain_strain, dtype=np.float64)


def make_twin(params, seed):
    """PreparedData for `params` injected into fresh Gaussian design noise,
    whitened by the same BilbyPreprocessor path as dataset generation."""
    sig = C.raw_projected_signal(params, C.GW150914_GPS, detectors=DETS)
    raw, psds = {}, {}
    for i, det in enumerate(DETS):
        raw[det] = design_noise(det, seed * 10 + i) + sig[det]
        psds[det] = get_default_psd(det)
    prep = pp.prepare_simulated(raw, psds=psds, seed=seed)
    # prepare_simulated does not record whitening_norm (prepare_real does);
    # recover it as the scale ratio between _common's whitening convention and
    # the prepared strain so network_mf_snr's template lives in data units.
    for i, det in enumerate(pp.DETECTORS):
        if det not in prep.detectors_present:
            continue
        asd_win = np.asarray(prep.asds[det], dtype=np.float64)
        ww = C.whiten_like_pipeline(raw[det], asd_win)
        prep.quality[det]["whitening_norm"] = float(
            np.std(ww) / (np.std(prep.strain[i]) + 1e-30))
    snr = C.network_mf_snr(prep, params)
    return prep, snr


def analyze(model, prep, truth, asd_bands_np, label):
    s = C.sample_from_strain(model, prep.strain, asd_bands_np=asd_bands_np,
                             n=N_SAMP, seed=0)
    m1, m2 = s[:, 0], s[:, 1]
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    scaler = model.scaler
    yn = scaler.normalize(torch.from_numpy(s).float()).numpy()
    yt = scaler.normalize(torch.tensor([[truth[k] for k in PARAM_NAMES]],
                                       dtype=torch.float32)).numpy()[0]
    noncirc = (~scaler.circ_mask).numpy()
    spur = ((yn > 0.999) & (yt[None, :] < 0.9)) | ((yn < -0.999) & (yt[None, :] > -0.9))
    spur = spur[:, noncirc]
    rail_any = float(spur.any(axis=1).mean())
    per_param = {PARAM_NAMES[j]: float(spur[:, k].mean())
                 for k, j in enumerate(np.where(noncirc)[0])}
    railed = {k: v for k, v in sorted(per_param.items(), key=lambda kv: -kv[1]) if v > 0.01}
    print(f"  {label:24} m1 {np.median(m1):5.1f}  m2 {np.median(m2):5.1f}  "
          f"Mc {np.median(mc):5.1f}  q {np.median(q):.3f} "
          f"[{np.quantile(q, .05):.2f},{np.quantile(q, .95):.2f}]  "
          f"q>0.95 {float((q > 0.95).mean()):.3f}  rail {rail_any:.3f}  {railed}")
    return dict(m1=float(np.median(m1)), m2=float(np.median(m2)),
                q=float(np.median(q)), fq95=float((q > 0.95).mean()), rail=rail_any)


def main():
    models = {}
    for name, path in MODELS.items():
        m = C.load_model(path)
        models[name] = m
        print(f"model {name}: epoch {m._ckpt_meta['epoch']} "
              f"val_nll {m._ckpt_meta['val_nll']:.3f} psd_cond {m._ckpt_meta['psd_cond']}")
    psd_bands = 16
    zeros_ab = np.zeros((3, psd_bands), dtype=np.float32)

    print("\n=== REAL GW150914 (control) ===")
    prep_real = C.prepare_gw150914()
    ab_real = C.asd_bands_of(prep_real, psd_bands)
    truth = dict(C.GW150914_TRUTH)
    for name, m in models.items():
        analyze(m, prep_real, truth, ab_real, f"real / {name}")

    print(f"\n=== TWIN in Gaussian design noise (published params, theta_jn={truth['theta_jn']}) ===")
    agg = {name: [] for name in models}
    for seed in range(6):
        prep, snr = make_twin(truth, seed)
        print(f" twin seed {seed}: network SNR opt {snr['network']['optimal_snr']:.1f} "
              f"mf {snr['network']['mf_snr']:.1f}")
        for name, m in models.items():
            r = analyze(m, prep, truth, zeros_ab, f"twin s{seed} / {name}")
            agg[name].append(r)

    print("\n=== TWIN edge-on (theta_jn = pi/2) ===")
    edge = dict(truth, theta_jn=float(np.pi / 2))
    for seed in range(2):
        prep, snr = make_twin(edge, 100 + seed)
        print(f" edge seed {seed}: network SNR opt {snr['network']['optimal_snr']:.1f} "
              f"mf {snr['network']['mf_snr']:.1f}")
        for name, m in models.items():
            analyze(m, prep, edge, zeros_ab, f"edge s{seed} / {name}")

    print("\n=== TWIN AGGREGATE (6 Gaussian twins, published params) ===")
    for name, rows in agg.items():
        med = lambda k: float(np.median([r[k] for r in rows]))
        print(f"  {name:10} m1 {med('m1'):5.1f}  m2 {med('m2'):5.1f}  q {med('q'):.3f}  "
              f"q>0.95 {med('fq95'):.3f}  rail {med('rail'):.3f}   "
              f"(truth m1 38.8 m2 33.4 q 0.861)")


if __name__ == "__main__":
    main()
