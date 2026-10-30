"""M14 - Substitution ladder: WHERE does pred q jump 0.46 -> 0.99?

Everything data-side has been eliminated (m11: asd_bands swap leaves the
attractor intact; m12: sim events in REAL noise never collapse; strain std /
kurtosis / spectra all in-distribution). So we walk from a simulated twin to the
actual event, changing ONE thing per rung, and record predicted q. The rung where
q jumps is where the problem enters.

  1. sim twin + design Gaussian noise      (training path, asd_bands = 0)
  2. sim twin + real O1 bank noise         (training path, re-colored, asd_bands from filter)
  3. sim twin injected into RAW O1 GWOSC   -> prepare_real()   (inference whitening path)
  4. ACTUAL GW150914                       -> prepare_real()

  1->2 isolates real NOISE. 2->3 isolates the PREPROCESSING path.
  3->4 isolates the REAL SIGNAL (only the signal differs between 3 and 4).

All rungs SNR-matched to ~24 (the real event's), because behaviour is strongly
SNR-dependent (m7).
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from ahsd.data.bilby_pipeline import (BilbyWaveformGenerator, BilbyPreprocessor,
                                      BilbySignalInjector, get_default_psd,
                                      _GPS_REF)  # noqa: E402
from ahsd.inference.pipeline import load_model  # noqa: E402
from ahsd.inference.preprocessing import prepare_real, fetch_gwosc, compute_asd_bands  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL = "model/lean_npe/best_model.pth"
SR, DUR, T_LEN, K = 4096, 4.0, 16384, 3000
GW150914_GPS = 1126259462.4
TARGET_SNR = 24.0

# published detector-frame masses + plausible extrinsics; aligned spins.
# geocent_time_gps is REQUIRED (bilby wants absolute GPS) -- passing only
# geocent_time silently places the signal at GPS 0 and wraps it.
TWIN = dict(mass_1=38.8, mass_2=33.4, luminosity_distance=440.0,
            ra=1.95, dec=-1.27, theta_jn=0.4, psi=1.75, phase=1.3,
            geocent_time=0.0, geocent_time_gps=_GPS_REF,
            a1=0.0, a2=0.0, tilt1=0.0, tilt2=0.0, event_type="BBH")


@torch.no_grad()
def probe(net, dev, strain, ab):
    st = torch.from_numpy(np.asarray(strain, np.float32)).unsqueeze(0).to(dev).float()
    abt = torch.from_numpy(np.asarray(ab, np.float32)).unsqueeze(0).to(dev).float()
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


def row(tag, snr, r):
    print(f"{tag:44} {snr:6.1f} {r['m1']:6.1f} {r['m2']:6.1f} {r['q']:6.3f} "
          f"{r['qg95']:6.2f} {r['dL']:7.0f} {r['rail']:6.2f}")


def main():
    net, meta = load_model(MODEL); dev = meta["device"]
    print(f"model epoch {meta['model_epoch']} val_nll {meta['model_val_nll']:.3f}")
    wfg = BilbyWaveformGenerator(SR, DUR)
    pre = BilbyPreprocessor(SR, DUR)
    inj = BilbySignalInjector(SR, DUR)
    rng = np.random.default_rng(0)

    raw_sig = {d: wfg.generate(TWIN, d).astype(np.float64) for d in ("H1", "L1", "V1")}
    sig_w = {d: pre.preprocess(raw_sig[d], get_default_psd(d), detector=d).astype(np.float32)
             for d in ("H1", "L1", "V1")}
    print(f"twin design-whitened net SNR {np.sqrt(sum((sig_w[d]**2).sum() for d in sig_w)):.1f}"
          f"  (rescaled to {TARGET_SNR:.0f} on every rung)\n")
    print(f"{'rung':44} {'SNR':>6} {'m1':>6} {'m2':>6} {'q':>6} {'q>.95':>6} "
          f"{'dL':>7} {'rail':>6}")

    # ── rung 1: sim twin + design Gaussian noise (training path) ──────────────
    s1 = TARGET_SNR / np.sqrt(sum((sig_w[d] ** 2).sum() for d in sig_w))
    strain = np.stack([rng.standard_normal(T_LEN).astype(np.float32)
                       + s1 * sig_w[d] for d in ("H1", "L1", "V1")])
    row("1. sim twin + design Gaussian noise", TARGET_SNR,
        probe(net, dev, strain, np.zeros((3, 16), np.float32)))

    # ── rung 2: sim twin + real O1 bank noise (training path, re-colored) ─────
    ds = RemixDataset("data/dataset/memmap/validation", remix=False, seed=1234,
                      real_noise_dir="data/noise_bank", real_noise_prob=1.0,
                      return_asd_bands=True, psd_bands=16)
    o1 = {d: [k for k, e in enumerate(ds._seg_era[d]) if e == "O1"] for d in ("H1", "L1", "V1")}
    rec, noise, ab = {}, np.zeros((3, T_LEN), np.float32), np.zeros((3, 16), np.float32)
    for di, d in enumerate(("H1", "L1", "V1")):
        if not o1[d]:
            noise[di] = rng.standard_normal(T_LEN).astype(np.float32)
            rec[d] = np.zeros(T_LEN, np.float32); continue
        k = int(rng.choice(o1[d])); seg = ds.real_bank[d][k]
        i0 = int(rng.integers(0, seg.shape[0] - T_LEN))
        noise[di] = seg[i0:i0 + T_LEN].astype(np.float32)
        filt = ds.recolor[d][k]
        rec[d] = np.fft.irfft(np.fft.rfft(sig_w[d]) * filt, n=T_LEN).astype(np.float32)
        logf = np.log(np.maximum(filt, 1e-30))
        ab[di] = [float(logf[s].mean()) for s in ds._band_slices]
    s2 = TARGET_SNR / np.sqrt(sum((rec[d] ** 2).sum() for d in rec))
    strain = np.stack([noise[di] + s2 * rec[d] for di, d in enumerate(("H1", "L1", "V1"))])
    row("2. sim twin + real O1 bank noise", TARGET_SNR, probe(net, dev, strain, ab))

    # ── rung 3: sim twin injected into RAW O1 GWOSC -> prepare_real() ─────────
    off_gps = GW150914_GPS - 400.0     # signal-free O1 time, same era/detectors
    try:
        gps, raw, starts, srs, found = fetch_gwosc(off_gps)
        pr_n = prepare_real(raw, gps, starts, srs, seed=0)
        scale = {}
        for d in found:
            snr1 = inj.measure_snr(raw_sig[d], pr_n.psds[d])
            scale[d] = snr1
        net_snr1 = float(np.sqrt(sum(v ** 2 for v in scale.values())))
        s3 = TARGET_SNR / max(net_snr1, 1e-9)
        raw_inj = {}
        for d in raw:
            x = np.asarray(raw[d], np.float64).copy()
            if d in found:
                i0 = int(round((gps - starts[d]) * SR)) - T_LEN // 2
                x[i0:i0 + T_LEN] += s3 * raw_sig[d]
            raw_inj[d] = x
        pr3 = prepare_real(raw_inj, gps, starts, srs, seed=0)
        ab3 = compute_asd_bands(pr3, psd_bands=16)
        row("3. sim twin in RAW O1 -> prepare_real()", TARGET_SNR,
            probe(net, dev, pr3.strain, ab3))
    except Exception as e:
        print(f"3. sim twin in RAW O1 -> prepare_real()   SKIPPED: {type(e).__name__}: {e}")

    # ── rung 4: the actual event ─────────────────────────────────────────────
    gps, raw, starts, srs, found = fetch_gwosc(GW150914_GPS)
    pr4 = prepare_real(raw, gps, starts, srs, seed=0)
    ab4 = compute_asd_bands(pr4, psd_bands=16)
    row("4. ACTUAL GW150914 -> prepare_real()", 24.0, probe(net, dev, pr4.strain, ab4))
    print(f"\n  twin truth: m1 {TWIN['mass_1']} m2 {TWIN['mass_2']} q "
          f"{TWIN['mass_2']/TWIN['mass_1']:.3f}   (dL relabelled by the SNR rescale)")


if __name__ == "__main__":
    main()
