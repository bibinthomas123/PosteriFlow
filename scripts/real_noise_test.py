#!/usr/bin/env python3
"""
Real-detector-noise robustness test for LeanNPE.

Injects freshly generated (design-PSD-whitened) signals into random 4 s crops
of the real whitened O3 segments in gw_segments/, runs LeanNPE, and compares
coverage / distance metrics against the Gaussian-noise validation results.

Caveat (by construction): signals are whitened with the design PSD while the
real noise was whitened with its measured PSD — the residual mismatch IS the
robustness being tested. Segments with glitches/non-stationarity are included
deliberately.
"""
import argparse, glob, json, sys, time
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import torch

from ahsd.data.bilby_pipeline import BilbySignalInjector, BilbyPreprocessor, get_default_psd
from ahsd.data.parameter_sampler import ParameterSampler
from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES

DETS = ["H1", "L1", "V1"]
SR, DUR, T = 4096, 4.0, 16384
JD = PARAM_NAMES.index("luminosity_distance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/lean_npe_v2/best_model.pth")
    ap.add_argument("--segments", default="gw_segments")
    ap.add_argument("--outdir", default="analysis/real_noise_test")
    ap.add_argument("--n_events", type=int, default=300)
    ap.add_argument("--n_post", type=int, default=400)
    ap.add_argument("--min_snr", type=float, default=8.0)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(7)
    torch.manual_seed(7)

    segs = {d: [np.load(f).astype(np.float32) for f in sorted(glob.glob(f"{args.segments}/{d}_*.npy"))]
            for d in DETS}
    print({d: len(v) for d, v in segs.items()}, "segments")

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = LeanNPE(premerger=ckpt["args"].get("premerger", False), psd_cond=ckpt["args"].get("psd_cond", False) or False, psd_bands=ckpt["args"].get("psd_bands", 16), encoder_type=ckpt["args"].get("encoder_type", "conv"))
    model.load_state_dict(ckpt["model_state_dict"]); model.to(device).eval()

    sampler = ParameterSampler()
    injector = BilbySignalInjector(SR, DUR)
    prep = BilbyPreprocessor(SR, DUR)
    psds = {d: get_default_psd(d) for d in DETS}

    def real_crop(det):
        s = segs[det][rng.integers(len(segs[det]))]
        i0 = rng.integers(0, len(s) - T)
        c = s[i0:i0 + T].copy()
        if rng.uniform() < 0.5:
            c = -c[::-1].copy()  # time-flip+sign: decorrelates reused segments
        return c

    events, t0 = [], time.time()
    while len(events) < args.n_events:
        p = sampler.sample_parameters()
        if p.get("event_type") == "noise":
            continue
        strains, snrs = [], []
        ok = True
        for det in DETS:
            sig, snr = injector.inject(np.zeros(T), p, det, psds[det])
            wsig = prep.preprocess(sig, psds[det], detector=det)
            if not np.isfinite(wsig).all():
                ok = False; break
            strains.append(real_crop(det) + wsig)
            snrs.append(snr)
        if not ok:
            continue
        net = float(np.sqrt(sum(s**2 for s in snrs)))
        if net < args.min_snr:
            continue
        events.append({"strain": np.stack(strains),
                       "theta": np.array([p[k] for k in PARAM_NAMES], dtype=np.float32),
                       "snr": net})
        if len(events) % 50 == 0:
            print(f"  built {len(events)}/{args.n_events} ({time.time()-t0:.0f}s)")

    N, P, K = len(events), len(PARAM_NAMES), args.n_post
    strain = torch.from_numpy(np.stack([e["strain"] for e in events]))
    truth = np.stack([e["theta"] for e in events])
    snr = np.array([e["snr"] for e in events])

    samples = np.empty((N, K, P), dtype=np.float32)
    with torch.no_grad():
        ctx = torch.cat([model.encoder(strain[i:i+64].to(device).float()) for i in range(0, N, 64)])
        rank0 = torch.zeros(N, dtype=torch.long, device=device)
        # shuffle test on real noise
        th = torch.from_numpy(truth).to(device)
        nll_t = model.nll(None, th, rank0, context=ctx)
        nll_s = model.nll(None, th, rank0, context=ctx[torch.randperm(N, device=device)])
        fc = model._full_context(ctx, rank0)
        for i in range(0, N, 48):
            f = fc[i:i+48]; nb = f.shape[0]
            fr = f.unsqueeze(1).expand(nb, K, f.shape[1]).reshape(nb*K, -1)
            z = torch.randn(nb*K, P, device=device)
            y, _ = model.flow.inverse(z, fr)
            samples[i:i+48] = model.scaler.denormalize(y).reshape(nb, K, P).cpu().numpy()

    report = {"n_events": N, "checkpoint_epoch": ckpt["epoch"],
              "nll_matched": float(nll_t.mean()), "shuffle_delta_nll": float(nll_s.mean()-nll_t.mean())}
    cov = {}
    for j, name in enumerate(PARAM_NAMES):
        lo50, hi50 = np.quantile(samples[:, :, j], [0.25, 0.75], axis=1)
        lo90, hi90 = np.quantile(samples[:, :, j], [0.05, 0.95], axis=1)
        cov[name] = {"50%": float(((truth[:, j] >= lo50) & (truth[:, j] <= hi50)).mean()),
                     "90%": float(((truth[:, j] >= lo90) & (truth[:, j] <= hi90)).mean())}
    report["coverage"] = cov
    dmed = np.median(samples[:, :, JD], axis=1)
    report["dist_corr_logmedian"] = float(np.corrcoef(np.log(dmed), np.log(truth[:, JD]))[0, 1])
    report["dist_frac_err_median_abs"] = float(np.median(np.abs(dmed - truth[:, JD]) / truth[:, JD]))

    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("done ->", out)


if __name__ == "__main__":
    main()
