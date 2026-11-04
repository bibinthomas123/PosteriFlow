#!/usr/bin/env python3
"""How many overlapping signals can the network handle?

Builds events with k = 1..k_max signals by summing k independent held-out
(validation) signal components and a noise realization -- the composition
the training remix uses -- orders them by the loudness proxy the model is
trained with (Mc^(5/6) / d_L), and queries every rank the model has
(rank embedding size: model.max_signals). Training data hold 1-4 signals;
k = 5 is beyond them (rank 4 never trained) and k > max_signals cannot be
queried at all.

Per (k, rank): 90% coverage of the headline parameters and of each mass,
median fractional chirp-mass / distance errors, and the median optimal SNR
of that rank's signal (quiet ranks are expected to be broad).

    python scripts/overlap_capacity.py --model model/lean_npe_bbh_v5/best_model.pth \
        --data data/dataset_bbh_v4 [--per_k 300] [--k_max 6] [--out analysis/overlap_capacity.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ahsd.inference.pipeline import load_model  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402
from overlap_benchmark import calib_block, mchirp, sample_ranks  # noqa: E402

J = {n: i for i, n in enumerate(PARAM_NAMES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="dataset root (uses memmap/validation)")
    ap.add_argument("--per_k", type=int, default=300)
    ap.add_argument("--k_max", type=int, default=6)
    ap.add_argument("--n_post", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.model, dev)[0].eval()
    R = int(model.max_signals)
    c = Path(a.data) / "memmap" / "validation"
    ev = json.load(open(c / "events.json"))["events"]
    sig, noi, par = (np.load(c / f, mmap_mode="r") for f in ("signals.npy", "noise.npy", "params.npy"))
    singles = np.array([s for s, n in ev if n == 1])
    rng = np.random.default_rng(a.seed)
    print(f"model {a.model}: rank embedding size {R}; {len(singles)} held-out single signals", flush=True)
    report = {"model": a.model, "max_signals": R, "per_k": {}}
    for k in range(1, a.k_max + 1):
        strains, truths, snrs = [], [], []
        for _ in range(a.per_k):
            comp = rng.choice(singles, size=k, replace=False)
            p = np.stack([np.asarray(par[s][:len(PARAM_NAMES)], np.float64) for s in comp])
            m1, m2 = np.maximum(p[:, 0], p[:, 1]), np.minimum(p[:, 0], p[:, 1])
            p[:, 0], p[:, 1] = m1, m2
            order = np.argsort(-(mchirp(m1, m2) ** (5 / 6) / p[:, J["luminosity_distance"]]))
            x = np.asarray(noi[rng.integers(len(noi))], np.float32).copy()
            snr = []
            for s in comp:
                g = np.asarray(sig[s], np.float32)
                x += g
                snr.append(float(np.sqrt((g.astype(np.float64) ** 2).sum())))
            strains.append(x)
            truths.append(p[order])
            snrs.append(np.asarray(snr)[order])
        strain = torch.from_numpy(np.stack(strains))
        nq = min(k, R)
        smp = sample_ranks(model, strain, nq, a.n_post, dev)
        rows = {}
        for r in range(nq):
            tr = np.stack([t[r] for t in truths])
            blk = calib_block(smp[:, r], tr)
            rows[f"rank{r}"] = {
                "cov90_headline": blk["cov90_headline_mean"],
                "cov90_m1": blk["cov90"]["mass_1"], "cov90_m2": blk["cov90"]["mass_2"],
                "cov90_dL": blk["cov90"]["luminosity_distance"],
                "mc_frac_err": blk["mc_frac_err_median"], "dl_frac_err": blk["dl_frac_err_median"],
                "median_snr": round(float(np.median([s[r] for s in snrs])), 1)}
            print(f"k={k} rank {r}: SNR {rows[f'rank{r}']['median_snr']:5.1f} | cov90 headline "
                  f"{blk['cov90_headline_mean']:.2f}  m1 {blk['cov90']['mass_1']:.2f}  "
                  f"dL {blk['cov90']['luminosity_distance']:.2f} | Mc err {blk['mc_frac_err_median']:.3f}  "
                  f"dL err {blk['dl_frac_err_median']:.3f}", flush=True)
        if k > R:
            print(f"k={k}: ranks {R}..{k - 1} cannot be queried (rank embedding size {R})", flush=True)
        report["per_k"][str(k)] = rows
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
