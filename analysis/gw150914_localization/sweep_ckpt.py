"""Fixed-mass amplitude sweep on a given checkpoint (design noise, signal+noise,
averaged over realizations). Isolates amplitude->distance from Mc/inclination."""
import argparse, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from ahsd.data.bilby_pipeline import (BilbyNoiseGenerator, BilbySignalInjector,
                                      BilbyPreprocessor, get_default_psd)

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
ap.add_argument("--M", type=int, default=12); args = ap.parse_args()
model = C.load_model(args.model)
print("loaded", args.model, "enc", model._ckpt_meta)
prep = BilbyPreprocessor(C.SR, C.DUR); ng = BilbyNoiseGenerator(C.SR, C.DUR)
inj = BilbySignalInjector(C.SR, C.DUR); psds = {d: get_default_psd(d) for d in ["H1","L1","V1"]}
zab = torch.zeros(1, 3, 16)
P0 = dict(C.GW150914_TRUTH)
grid = [300, 450, 650, 950, 1400, 1900]
meds, snrs = [], []
for dL in grid:
    P = dict(P0); P["luminosity_distance"] = float(dL); dm = []
    for m in range(args.M):
        white, ss = {}, []
        for det in ["H1","L1"]:
            noise = ng.generate(det, psds[det], seed=4000+m*17+dL)
            st, s = inj.inject(noise, P, det, psds[det]); white[det] = prep.preprocess(st, psds[det], detector=det); ss.append(s)
        rng = np.random.default_rng(m)
        s3 = torch.from_numpy(np.stack([white["H1"], white["L1"], rng.standard_normal(C.T_LEN).astype(np.float32)]))[None].float()
        with torch.no_grad():
            samp = model.sample_posterior(s3, rank=0, n_samples=800, asd_bands=zab)[0].numpy()
        dm.append(float(np.median(samp[:, C.JD])))
    meds.append(float(np.mean(dm))); snrs.append(float(np.sqrt(sum(x**2 for x in ss))))
slope = float(np.polyfit(np.log(grid), np.log(meds), 1)[0])
rho = float(spearmanr(grid, meds).correlation)
print("\nfixed heavy-BBH amplitude sweep (design noise, signal+noise):")
for dL, mm, sn in zip(grid, meds, snrs):
    print("  true dL=%5d  SNR=%5.1f  -> flow dL_med=%6.0f" % (dL, sn, mm))
print("  slope d(log dL_med)/d(log dL) = %+.2f   Spearman = %+.2f   (1=ideal, 0=flat)" % (slope, rho))
