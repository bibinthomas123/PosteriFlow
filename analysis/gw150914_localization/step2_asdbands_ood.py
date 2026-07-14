"""
STEP 2 - is GW150914's PSD representation (asd_bands, the psd_cond input) OOD?

Compares the actual asd_bands vector the encoder receives against the training
asd_bands distribution reproduced from the noise bank via RemixDataset's own
recipe (band-mean log(design/measured)). Reports percentile, nearest-neighbour
distance, Mahalanobis distance -- overall and restricted to the same era (O1).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from remix_data import RemixDataset


def era(gps):
    if gps < 1137254417:  return "O1"
    if gps < 1187733618:  return "O2"
    return "O3"


OUT = Path(__file__).resolve().parent
# build the bank recolor filters via RemixDataset, then band-average like training
ds = RemixDataset(str(C.ROOT / "data/dataset/memmap/validation"), remix=False, seed=0,
                  real_noise_dir=str(C.ROOT / "data/noise_bank"), real_noise_prob=1.0,
                  return_asd_bands=True, psd_bands=16)
slices = ds._band_slices
bank_dir = C.ROOT / "data/noise_bank"

pool = {"H1": [], "L1": [], "V1": []}
eras = {"H1": [], "L1": [], "V1": []}
for d in ("H1", "L1", "V1"):
    files = sorted(bank_dir.glob(f"{d}_*_strain.npy"))
    for k, f in enumerate(files):
        filt = ds.recolor[d][k]
        logf = np.log(np.maximum(filt, 1e-30))
        vec = np.array([float(logf[s].mean()) for s in slices], dtype=np.float32)
        pool[d].append(vec)
        gps = int(f.stem.split("_")[1])
        eras[d].append(era(gps))
    pool[d] = np.stack(pool[d]); eras[d] = np.array(eras[d])

era_counts = {d: {e: int((eras[d] == e).sum()) for e in ("O1", "O2", "O3")} for d in pool}
print("bank era composition:", era_counts)

# GW150914 asd_bands (inference recipe)
prep = C.prepare_gw150914()
ab_gw = C.asd_bands_of(prep)  # [3,16]; H1,L1 real, V1=0

report = {"bank_era_counts": era_counts, "detectors": {}}
for i, d in enumerate(("H1", "L1")):
    v = ab_gw[i].astype(np.float64)
    P = pool[d].astype(np.float64)
    # percentile per band + overall (mean over bands of |percentile-50|)
    pct = np.array([(P[:, b] < v[b]).mean() * 100 for b in range(16)])
    # nearest neighbour (overall + O1-only)
    nn_all = float(np.min(np.linalg.norm(P - v, axis=1)))
    o1 = P[eras[d] == "O1"]
    nn_o1 = float(np.min(np.linalg.norm(o1 - v, axis=1))) if len(o1) else float("nan")
    # Mahalanobis vs pool
    mu = P.mean(0); cov = np.cov(P.T) + 1e-6 * np.eye(16)
    maha = float(np.sqrt((v - mu) @ np.linalg.inv(cov) @ (v - mu)))
    # typical intra-pool NN + Mahalanobis for reference
    nn_typ = float(np.median([np.min(np.linalg.norm(np.delete(P, j, 0) - P[j], axis=1))
                              for j in np.random.default_rng(0).integers(0, len(P), 80)]))
    maha_typ = float(np.median([np.sqrt((P[j] - mu) @ np.linalg.inv(cov) @ (P[j] - mu))
                                for j in range(len(P))]))
    report["detectors"][d] = dict(
        asd_bands=v.round(3).tolist(),
        band_percentiles=pct.round(1).tolist(),
        n_bands_beyond_95pct=int(((pct > 95) | (pct < 5)).sum()),
        nn_euclid_all=round(nn_all, 3), nn_euclid_o1=round(nn_o1, 3),
        nn_typical_intra_pool=round(nn_typ, 3),
        mahalanobis=round(maha, 2), mahalanobis_typical=round(maha_typ, 2))
    print(f"\n{d}: bands beyond [5,95]pct = {report['detectors'][d]['n_bands_beyond_95pct']}/16")
    print(f"   NN(all)={nn_all:.2f}  NN(O1 only)={nn_o1:.2f}  typical intra-pool NN={nn_typ:.2f}")
    print(f"   Mahalanobis={maha:.2f}  (typical intra-pool median={maha_typ:.2f})")

json.dump(report, open(OUT / "step2_asdbands_ood.json", "w"), indent=2, default=float)
print(f"\nwrote {OUT/'step2_asdbands_ood.json'}")
