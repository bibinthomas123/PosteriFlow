"""M13 - Exact-parameter GW150914 TWIN injected into REAL O1 noise.

Separates "the model cannot do these parameters" from "the REAL GW150914 signal
has structure the training waveform lacks". We synthesize GW150914's PUBLISHED
parameters through the SAME generation path the training data uses (bilby
IMRPhenomXP, aligned spins, design-whitened, then re-colored into a real O1
segment exactly as RemixDataset does) and ask the calibrated ep40 model to
recover it.

  twin recovers m1~38.8      -> model handles these params; the REAL signal is the
                                odd one out (waveform systematics: precession /
                                higher modes absent from aligned-spin training)
  twin collapses to (30,30)  -> the model cannot represent this corner at all;
                                an in-distribution representation failure
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from ahsd.data.bilby_pipeline import (BilbyWaveformGenerator, BilbyPreprocessor,
                                      get_default_psd)  # noqa: E402
from ahsd.inference.pipeline import load_model  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL = "model/lean_npe/best_model.pth"
SR, DUR, T_LEN = 4096, 4.0, 16384
K = 3000

# GW150914 published (detector-frame) + plausible extrinsics; aligned spins ~0
TWIN = dict(mass_1=38.8, mass_2=33.4, luminosity_distance=440.0,
            ra=1.95, dec=-1.27, theta_jn=2.9, psi=1.75, phase=1.3,
            geocent_time=0.0, a1=0.0, a2=0.0, tilt1=0.0, tilt2=0.0,
            event_type="BBH")


@torch.no_grad()
def probe(net, dev, strain, ab):
    st = torch.from_numpy(strain).unsqueeze(0).to(dev).float()
    abt = torch.from_numpy(ab.astype(np.float32)).unsqueeze(0).to(dev).float()
    ctx = net._full_context(net.encode(st, abt),
                            torch.zeros(1, dtype=torch.long, device=dev))
    y, _ = net.flow.inverse(torch.randn(K, 11, device=dev), ctx.expand(K, -1))
    yw = net.scaler.wrap(y)
    nc = ~net.scaler.circ_mask
    rail = float((yw.abs() > 0.999)[:, nc].any(1).float().mean())
    s = net.scaler.denormalize(yw).cpu().numpy()
    m1 = np.maximum(s[:, 0], s[:, 1]); m2 = np.minimum(s[:, 0], s[:, 1])
    q = m2 / m1
    return dict(m1=float(np.median(m1)), m2=float(np.median(m2)),
                q=float(np.median(q)), qg95=float((q > 0.95).mean()),
                dL=float(np.median(s[:, 2])), rail=rail)


def main():
    net, meta = load_model(MODEL); dev = meta["device"]
    print(f"model epoch {meta['model_epoch']} val_nll {meta['model_val_nll']:.3f}\n")

    wfg = BilbyWaveformGenerator(SR, DUR)
    pre = BilbyPreprocessor(SR, DUR)
    ds = RemixDataset("data/dataset/memmap/validation", remix=False, seed=1234,
                      real_noise_dir="data/noise_bank", real_noise_prob=1.0,
                      return_asd_bands=True, psd_bands=16)

    # design-whitened twin signal per detector (exactly how the dataset stores signals)
    sig_w = {}
    for det in ("H1", "L1", "V1"):
        raw = wfg.generate(TWIN, det)
        sig_w[det] = pre.preprocess(raw, get_default_psd(det), detector=det).astype(np.float32)
    snr_design = float(np.sqrt(sum((sig_w[d] ** 2).sum() for d in ("H1", "L1", "V1"))))
    print(f"twin design-whitened network SNR = {snr_design:.1f}  "
          f"(real GW150914 ~24)\n")

    # inject into REAL O1 segments, re-colored exactly as RemixDataset does
    o1 = {d: [k for k, e in enumerate(ds._seg_era[d]) if e == "O1"] for d in ("H1", "L1", "V1")}
    print(f"O1 segments available: " + " ".join(f"{d}={len(o1[d])}" for d in o1))
    # Match the REAL GW150914 network SNR (~24). The recolored amplitude depends
    # on the segment's sensitivity and on our (arbitrary) extrinsics, so we
    # rescale amplitude by s and relabel dL -> dL/s — exactly the physically
    # consistent distance rescale RemixDataset applies. Without this the twin
    # lands at SNR~10 and is NOT comparable (behaviour is strongly SNR-dependent).
    TARGET_SNR = 24.0
    rng = np.random.default_rng(0)
    print(f"\n{'trial':8} {'SNR':>6} {'dL_true':>8} {'m1':>6} {'m2':>6} {'q':>6} "
          f"{'q>.95':>6} {'dL_pred':>8} {'rail':>6}")
    for t in range(4):
        rec = {}
        ab = np.zeros((3, 16), dtype=np.float32)
        noise = np.zeros((3, T_LEN), dtype=np.float32)
        for di, d in enumerate(("H1", "L1", "V1")):
            if not o1[d]:      # V1 has no O1 data -> white-noise fill (as GW150914)
                noise[di] = rng.standard_normal(T_LEN).astype(np.float32)
                rec[d] = np.zeros(T_LEN, dtype=np.float32)   # V1 absent for GW150914
                continue
            k = int(rng.choice(o1[d]))
            seg = ds.real_bank[d][k]
            i0 = int(rng.integers(0, seg.shape[0] - T_LEN))
            noise[di] = seg[i0:i0 + T_LEN].astype(np.float32)
            filt = ds.recolor[d][k]
            rec[d] = np.fft.irfft(np.fft.rfft(sig_w[d]) * filt, n=T_LEN).astype(np.float32)
            logf = np.log(np.maximum(filt, 1e-30))
            ab[di] = [float(logf[s].mean()) for s in ds._band_slices]
        snr_now = float(np.sqrt(sum((rec[d] ** 2).sum() for d in rec)))
        s = TARGET_SNR / max(snr_now, 1e-9)          # amplitude scale
        dL_true = TWIN["luminosity_distance"] / s     # exact GR relabel
        strain = noise.copy()
        for di, d in enumerate(("H1", "L1", "V1")):
            strain[di] = strain[di] + s * rec[d]
        snr_final = float(np.sqrt(sum(((s * rec[d]) ** 2).sum() for d in rec)))
        r = probe(net, dev, strain, ab)
        print(f"O1 #{t:<5} {snr_final:6.1f} {dL_true:8.0f} {r['m1']:6.1f} {r['m2']:6.1f} "
              f"{r['q']:6.3f} {r['qg95']:6.2f} {r['dL']:8.0f} {r['rail']:6.2f}")
    print(f"\n  twin truth   m1 {TWIN['mass_1']:.1f}  m2 {TWIN['mass_2']:.1f}  "
          f"q {TWIN['mass_2']/TWIN['mass_1']:.3f}   (dL relabelled per trial, SNR matched to {TARGET_SNR:.0f})")
    print(f"  REAL GW150914 -> m1 30.2  m2 30.0  q 0.996  dL 703  rail 0.69")


if __name__ == "__main__":
    main()
