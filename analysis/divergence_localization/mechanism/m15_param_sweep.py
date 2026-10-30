"""M15 - WHICH parameter triggers the q->1 collapse? (pure simulation)

m14 rung 1 reproduced the real-event attractor on a fully simulated twin in
design Gaussian noise (q 0.994, m1~m2~30.3 for truth 38.8/33.4) -- so the trigger
is the PARAMETERS, not the data domain. m12 missed this because it matched
(Mc, q, SNR) but left spins/inclination random; the m14 twin has a1=a2=0 and
theta_jn=0.4, matching the genuinely low-spin, face-on real GW150914.

Sweep one axis at a time at fixed heavy+near-equal masses, design Gaussian noise,
SNR matched to 24, asd_bands=0. Report predicted q and masses. Also report
SPURIOUS railing (truth-aware) since a1=a2=0 sits ON the prior bound, where
railing is legitimate and the naive metric is misleading.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from ahsd.data.bilby_pipeline import (BilbyWaveformGenerator, BilbyPreprocessor,
                                      get_default_psd, _GPS_REF)  # noqa: E402
from ahsd.inference.pipeline import load_model  # noqa: E402

SR, DUR, T_LEN, K = 4096, 4.0, 16384, 2000
TARGET_SNR = 24.0
BASE = dict(mass_1=38.8, mass_2=33.4, luminosity_distance=440.0,
            ra=1.95, dec=-1.27, theta_jn=0.4, psi=1.75, phase=1.3,
            geocent_time=0.0, geocent_time_gps=_GPS_REF,
            a1=0.0, a2=0.0, tilt1=0.0, tilt2=0.0, event_type="BBH")


@torch.no_grad()
def run(net, dev, wfg, pre, params, rng):
    sig = {d: pre.preprocess(wfg.generate(params, d).astype(np.float64),
                             get_default_psd(d), detector=d).astype(np.float32)
           for d in ("H1", "L1", "V1")}
    n0 = np.sqrt(sum((sig[d] ** 2).sum() for d in sig))
    s = TARGET_SNR / max(float(n0), 1e-9)
    strain = np.stack([rng.standard_normal(T_LEN).astype(np.float32) + s * sig[d]
                       for d in ("H1", "L1", "V1")])
    st = torch.from_numpy(strain).unsqueeze(0).to(dev).float()
    ab = torch.zeros(1, 3, 16, device=dev)
    ctx = net._full_context(net.encode(st, ab),
                            torch.zeros(1, dtype=torch.long, device=dev))
    y, _ = net.flow.inverse(torch.randn(K, 11, device=dev), ctx.expand(K, -1))
    yw = net.scaler.wrap(y)
    # truth in scaled space -> SPURIOUS railing (railed where truth is NOT at that bound)
    tv = np.array([[params[k] for k in ["mass_1", "mass_2", "luminosity_distance",
                                        "ra", "dec", "theta_jn", "psi", "phase",
                                        "geocent_time", "a1", "a2"]]], np.float32)
    tv[0, 2] = params["luminosity_distance"] / s          # exact GR relabel
    yt = net.scaler.normalize(torch.from_numpy(tv).to(dev))
    nc = ~net.scaler.circ_mask
    spur = (((yw > 0.999) & (yt < 0.9)) | ((yw < -0.999) & (yt > -0.9)))[:, nc].any(1)
    smp = net.scaler.denormalize(yw).cpu().numpy()
    m1 = np.maximum(smp[:, 0], smp[:, 1]); m2 = np.minimum(smp[:, 0], smp[:, 1])
    q = m2 / m1
    return dict(m1=float(np.median(m1)), m2=float(np.median(m2)), q=float(np.median(q)),
                qg95=float((q > 0.95).mean()), rail_spur=float(spur.float().mean()),
                dL_true=float(tv[0, 2]))


def main():
    net, meta = load_model(MODEL := "model/lean_npe/best_model.pth"); dev = meta["device"]
    print(f"model epoch {meta['model_epoch']}   base: m1 38.8 m2 33.4 (q 0.861), "
          f"design Gaussian noise, SNR {TARGET_SNR:.0f}\n")
    wfg, pre = BilbyWaveformGenerator(SR, DUR), BilbyPreprocessor(SR, DUR)
    rng = np.random.default_rng(0)
    hdr = f"{'variant':26} {'m1':>6} {'m2':>6} {'q':>6} {'q>.95':>6} {'railSpur':>9}"

    print("--- sweep SPIN (a1=a2), theta_jn=0.4 ---"); print(hdr)
    for a in [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]:
        p = dict(BASE, a1=a, a2=a)
        r = run(net, dev, wfg, pre, p, rng)
        print(f"a1=a2={a:<20.2f} {r['m1']:6.1f} {r['m2']:6.1f} {r['q']:6.3f} "
              f"{r['qg95']:6.2f} {r['rail_spur']:9.2f}")

    print("\n--- sweep INCLINATION (theta_jn), a1=a2=0 ---"); print(hdr)
    for th in [0.0, 0.4, 0.8, 1.57, 2.4, 2.9]:
        p = dict(BASE, theta_jn=th)
        r = run(net, dev, wfg, pre, p, rng)
        print(f"theta_jn={th:<17.2f} {r['m1']:6.1f} {r['m2']:6.1f} {r['q']:6.3f} "
              f"{r['qg95']:6.2f} {r['rail_spur']:9.2f}")

    print("\n--- sweep INCLINATION at a1=a2=0.4 (mid spin) ---"); print(hdr)
    for th in [0.4, 1.57, 2.9]:
        p = dict(BASE, theta_jn=th, a1=0.4, a2=0.4)
        r = run(net, dev, wfg, pre, p, rng)
        print(f"theta_jn={th:<8.2f} a=0.4    {r['m1']:6.1f} {r['m2']:6.1f} {r['q']:6.3f} "
              f"{r['qg95']:6.2f} {r['rail_spur']:9.2f}")

    print("\n--- sweep TRUE q (vary m2), a1=a2=0, theta_jn=0.4 ---"); print(hdr)
    for m2 in [12.0, 20.0, 27.0, 33.4, 38.0]:
        p = dict(BASE, mass_2=m2)
        r = run(net, dev, wfg, pre, p, rng)
        print(f"m2={m2:<5.1f} (q={m2/38.8:.2f})     {r['m1']:6.1f} {r['m2']:6.1f} "
              f"{r['q']:6.3f} {r['qg95']:6.2f} {r['rail_spur']:9.2f}")

    print("\n--- sweep MASS SCALE (q fixed 0.861), a1=a2=0, theta_jn=0.4 ---"); print(hdr)
    for f in [0.3, 0.5, 0.75, 1.0]:
        p = dict(BASE, mass_1=38.8 * f, mass_2=33.4 * f)
        r = run(net, dev, wfg, pre, p, rng)
        print(f"m1={38.8*f:<5.1f} m2={33.4*f:<5.1f}     {r['m1']:6.1f} {r['m2']:6.1f} "
              f"{r['q']:6.3f} {r['qg95']:6.2f} {r['rail_spur']:9.2f}")


if __name__ == "__main__":
    main()
