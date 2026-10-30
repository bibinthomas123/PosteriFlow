#!/usr/bin/env python3
"""E-E1: the alignment gate for a phase-preserving (SVD) front-end.

A frequency-domain SVD basis is only compact for near-ALIGNED waveforms: a time
shift tau multiplies the FD waveform by exp(-2i pi f tau), and across [20,1024]
Hz that is ~15 cycles of phase ramp at tau=15 ms but ~1500 cycles at tau=1.5 s.
The previous SVD attempt (analysis/posterior_bias_audit/REPORT.md Sec.11) built
its basis with tau ~ +/-15 ms and deployed it on data spanning +/-1.5 s, a 100x
mismatch — which is why it failed, and why DINGO needs GNPE.

So before rebuilding a basis, the question is whether we can estimate absolute
coalescence time per detector well enough to pre-align. The MF-token branch
already computes exactly this: `_mf_features` returns, per (detector, template),
the peak time `t_rel` of the matched-filter envelope over a +/-1.6 s window at
~1 ms resolution. This script measures how well the best-matching template's
peak time recovers the true coalescence time.

GATE: median |residual| must approach the light-travel floor (per-detector
arrival times differ from geocent_time by up to ~10 ms, which an actual
per-detector alignment scheme would absorb). A median residual of order 10 ms
=> the estimator is at the physical floor and the SVD path is viable. A median
residual of order 100 ms or more => it is not, and the fallback is to reduce the
conv stem's 256x stride instead.
"""
import sys
from pathlib import Path
ROOT = Path("/Users/bibinthomas/Developer/PosteriFlow")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch
from experiments.remix_data import RemixDataset
from ahsd.models.lean_npe import LeanNPE

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ck = torch.load(ROOT / "model/lean_npe_mf_scratch/best_model.pth", map_location="cpu",
                weights_only=False)
model = LeanNPE(psd_cond=True, psd_bands=16, encoder_type="coherent",
                mass_param="mc_q", mf_tokens=True)
model.load_state_dict(ck["model_state_dict"]); model.to(DEV).eval()
enc = model.encoder
mc_q = np.load(Path(enc.__class__.__module__ and ROOT / "src/ahsd/models/assets/mf_bank.npz"))["mc_q"]
print(f"bank: {mc_q.shape[0]} templates, Mc in [{mc_q[:,0].min():.1f}, {mc_q[:,0].max():.1f}], "
      f"q in [{mc_q[:,1].min():.2f}, {mc_q[:,1].max():.2f}]")

ds = RemixDataset(str(ROOT / "data/dataset/memmap/validation"), remix=True, seed=99,
                  real_noise_prob=0.0, return_asd_bands=True, psd_bands=16)
ds.set_epoch(0)

N, B = 1000, 25
RES, TRUE_T, SNR, MC, Q, BESTMC = [], [], [], [], [], []
for i0 in range(0, N, B):
    items = [ds[i] for i in range(i0, min(i0 + B, N))]
    strain = torch.stack([x[0] for x in items]).to(DEV).float()
    pv = torch.stack([x[1] for x in items]).numpy()
    snr = torch.stack([x[3] for x in items]).numpy()
    with torch.no_grad():
        f = enc._mf_features(strain)                      # [B, D*K*4]
    D, K = 3, mc_q.shape[0]
    f = f.reshape(len(items), D, K, 4).cpu().numpy()
    asnr, t_rel = f[..., 0], f[..., 3]                    # asinh(snr), seconds
    for k in range(len(items)):
        m1, m2 = pv[k, 0, 0], pv[k, 0, 1]
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        # best-matching template per detector, then median across detectors
        bt = asnr[k].argmax(axis=1)                       # [D] best template idx
        t_est = np.median([t_rel[k, d, bt[d]] for d in range(D)])
        # loudest detector's choice of template (for the Mc-recovery check)
        d_loud = asnr[k].max(axis=1).argmax()
        BESTMC.append(mc_q[bt[d_loud], 0])
        RES.append(t_est - pv[k, 0, 8]); TRUE_T.append(pv[k, 0, 8])
        SNR.append(snr[k]); MC.append(mc); Q.append(m2 / m1)
    if i0 % 250 == 0: print(f"  {i0}/{N}", flush=True)

RES = np.array(RES) * 1e3          # ms
SNR, MC, Q = np.array(SNR), np.array(MC), np.array(Q)
BESTMC = np.array(BESTMC)
np.savez(Path(__file__).parent / "eE1_raw.npz", RES=RES, SNR=SNR, MC=MC, Q=Q,
         TRUE_T=np.array(TRUE_T), BESTMC=BESTMC)

def rep(tag, m):
    if m.sum() < 20: return
    r = np.abs(RES[m])
    print(f"  {tag:28s} n={m.sum():4d}  median={np.median(r):8.1f} ms  "
          f"p90={np.percentile(r,90):9.1f} ms  |  <5ms {(r<5).mean()*100:5.1f}%  "
          f"<15ms {(r<15).mean()*100:5.1f}%  <50ms {(r<50).mean()*100:5.1f}%")

print("\n" + "=" * 108)
print("MF peak-time residual vs true coalescence time  (light-travel floor ~10 ms)")
print("=" * 108)
rep("ALL", np.ones(len(RES), bool))
print()
rep("Mc in bank [12,45]", (MC >= 12) & (MC <= 45))
rep("Mc BELOW bank (<12)", MC < 12)
rep("Mc ABOVE bank (>45)", MC > 45)
print()
for lo, hi in [(0, 12), (12, 20), (20, 35), (35, 60), (60, 500)]:
    rep(f"SNR [{lo},{hi})", (SNR >= lo) & (SNR < hi))
print()
rep("in-bank AND SNR>20", (MC >= 12) & (MC <= 45) & (SNR > 20))
rep("in-bank AND SNR>20 AND q>0.5", (MC >= 12) & (MC <= 45) & (SNR > 20) & (Q > 0.5))

r = np.abs(RES)
print("\n" + "=" * 108)
print(f"GATE (median |residual| over all events): {np.median(r):.1f} ms")
if np.median(r) < 15:
    print("  PASS -> at/near the light-travel floor; SVD front-end viable, proceed to E-E2.")
elif np.median(r) < 50:
    print("  MARGINAL -> usable only where the bank covers; fix bank coverage (E-B) and re-run.")
else:
    print("  FAIL -> alignment cannot be recovered from the current MF bank.")
print("=" * 108)
