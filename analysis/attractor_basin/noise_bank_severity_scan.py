"""Does GW150914's specific noise segment sit outside the severity range the
real-noise-bank training diversity actually covers?

Consistency-loss trend in lean_npe_noiseinv_real's training log converged to
near-zero (0.0015-0.0025) -- the mechanism clearly worked WITHIN the training
distribution (basin-cell signals x real-noise-bank segments), yet real
GW150914 rail barely moved (74.0% -> 79.3%). Two remaining explanations:
  (2) GW150914's exact segment is an outlier the bank's O1-era diversity
      doesn't cover -- consistency training over in-bank segments can't
      teach robustness to an out-of-bank severity level.
  (3) even in-bank severity levels aren't handled if the LOSS/architecture
      doesn't transfer what it learns to arbitrary new real strain.

This script computes the SAME diagnostic local_noise_grafting.py used on
GW150914 (peak matched-filter SNR of the basin-probe synthetic signal,
correctly delayed, injected into a noise segment) across a sample of the O1
noise-bank segments used in training, and compares the spread to GW150914's
own measured degradation (H1 18.5 / L1 12.4 vs generic off-source H1 24.9 /
L1 14.4 -- a ~26%/14% SNR suppression).
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "analysis/gw150914_localization"))

from ahsd.evaluation.basin_probe import BASE, T_LEN, SR
from ahsd.models.lean_npe import LeanNPE
from ahsd.data.bilby_pipeline import BilbyPreprocessor, get_default_psd
from _common import raw_projected_signal, GW150914_GPS

NOISE_BANK = ROOT / "data/noise_bank"
CKPT = ROOT / "model/lean_npe_mf_scratch/best_model.pth"
SYN = dict(mass_1=38.8, mass_2=33.4, luminosity_distance=440.0,
          theta_jn=2.9, ra=1.95, dec=-1.27, psi=1.75, phase=1.3,
          geocent_time=0.0, a1=0.0, a2=0.0)


def recolor_filter(det, seg_gps, clamp=50.0):
    design = np.load(NOISE_BANK / f"design_asd_{det}.npy").astype(np.float64)
    meas = np.load(NOISE_BANK / f"{det}_{seg_gps}_asd.npy").astype(np.float64)
    n = min(len(design), len(meas))
    filt = np.clip(design[:n] / np.maximum(meas[:n], 1e-30), 1.0 / clamp, clamp)
    return filt.astype(np.float32)


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = LeanNPE(psd_cond=a.get("psd_cond", True), psd_bands=a.get("psd_bands", 16),
                    encoder_type=a.get("encoder_type", "coherent"),
                    mass_param=a.get("mass_param", "mc_q"), geom_dropout=0.0,
                    geom_phase_residual=a.get("geom_phase_residual", False),
                    mf_tokens=True, mf_shape_norm=a.get("mf_shape_norm", False), mf_token_fusion=a.get("mf_token_fusion", "add"), mf_bank=a.get("mf_bank", "mf_bank.npz"), pol_features=a.get("pol_features", False), q_embed_dim=a.get("q_embed_dim", 0))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    enc = model.encoder
    K = enc.mf_re.shape[1]

    # O1-era segments actually present in the noise bank (same pool
    # get_multiview(noise_mode="real") draws from)
    h1_files = sorted(NOISE_BANK.glob("H1_*_strain.npy"))
    l1_files = sorted(NOISE_BANK.glob("L1_*_strain.npy"))
    h1_o1 = [f.stem.split("_")[1] for f in h1_files
            if 1126051217 <= int(f.stem.split("_")[1]) <= 1137254417]
    l1_o1 = [f.stem.split("_")[1] for f in l1_files
            if 1126051217 <= int(f.stem.split("_")[1]) <= 1137254417]
    print(f"O1-era segments available: H1={len(h1_o1)} L1={len(l1_o1)}")

    # design-whiten the raw projected signal FIRST (bilby matched-filter norm,
    # same formula BilbyPreprocessor/the dataset generator use to build the
    # stored signals.npy that RemixDataset's recolor filter is meant to act
    # on) -- v1 of this script skipped this and applied the recolor filter
    # (design_asd/measured_asd) directly to RAW strain, the exact units bug
    # already caught and retracted once this session (real_noise_basin_check.py).
    prep = BilbyPreprocessor(SR, T_LEN / SR)
    raw_sig = raw_projected_signal(SYN, GW150914_GPS, detectors=("H1", "L1"))
    sig = {det: prep.preprocess(raw_sig[det], get_default_psd(det), detector=det)
          for det in ("H1", "L1")}
    rng = np.random.default_rng(0)
    n_trials = min(25, len(h1_o1), len(l1_o1))
    h1_sample = rng.choice(h1_o1, n_trials, replace=False)
    l1_sample = rng.choice(l1_o1, n_trials, replace=False)

    h1_snrs, l1_snrs = [], []
    for i in range(n_trials):
        filt_h1 = recolor_filter("H1", h1_sample[i])
        filt_l1 = recolor_filter("L1", l1_sample[i])
        rec_h1 = np.fft.irfft(np.fft.rfft(sig["H1"]) * filt_h1, n=T_LEN).astype(np.float32)
        rec_l1 = np.fft.irfft(np.fft.rfft(sig["L1"]) * filt_l1, n=T_LEN).astype(np.float32)
        noise_h1 = np.load(NOISE_BANK / f"H1_{h1_sample[i]}_strain.npy").astype(np.float32)
        noise_l1 = np.load(NOISE_BANK / f"L1_{l1_sample[i]}_strain.npy").astype(np.float32)
        i0h = int(rng.integers(0, noise_h1.shape[0] - T_LEN))
        i0l = int(rng.integers(0, noise_l1.shape[0] - T_LEN))
        strain = np.stack([noise_h1[i0h:i0h + T_LEN] + rec_h1,
                           noise_l1[i0l:i0l + T_LEN] + rec_l1,
                           rng.standard_normal(T_LEN).astype(np.float32)])
        with torch.no_grad():
            feats = enc._mf_features(torch.from_numpy(strain).unsqueeze(0).float())
        f = feats.reshape(1, 3, K, 4)[0]
        snr_h1 = float(torch.sinh(f[0, :, 0]).max())
        snr_l1 = float(torch.sinh(f[1, :, 0]).max())
        h1_snrs.append(snr_h1)
        l1_snrs.append(snr_l1)

    h1_snrs, l1_snrs = np.array(h1_snrs), np.array(l1_snrs)
    print(f"\nbank sample (n={n_trials}) peak MF SNR:")
    print(f"  H1: mean {h1_snrs.mean():.1f}  std {h1_snrs.std():.1f}  "
          f"min {h1_snrs.min():.1f}  max {h1_snrs.max():.1f}")
    print(f"  L1: mean {l1_snrs.mean():.1f}  std {l1_snrs.std():.1f}  "
          f"min {l1_snrs.min():.1f}  max {l1_snrs.max():.1f}")
    print(f"\nGW150914's own segment (from local_noise_grafting.py, 2026-07-20): "
          f"H1=18.5  L1=12.4")
    print(f"generic off-source control: H1=24.9  L1=14.4")
    h1_z = (18.5 - h1_snrs.mean()) / max(h1_snrs.std(), 1e-6)
    l1_z = (12.4 - l1_snrs.mean()) / max(l1_snrs.std(), 1e-6)
    print(f"\nGW150914's degradation in z-score units vs bank sample: "
          f"H1 z={h1_z:.2f}  L1 z={l1_z:.2f}")
    below = int((h1_snrs < 18.5).sum())
    print(f"bank segments with H1 SNR <= GW150914's 18.5: {below}/{n_trials} "
          f"({100*below/n_trials:.0f}%)")


if __name__ == "__main__":
    main()
