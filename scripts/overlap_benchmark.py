#!/usr/bin/env python3
"""
Overlap evaluation — the paper's core claim: per-signal posteriors that stay
calibrated as the number of overlapping signals and their temporal overlap
increase, at a fraction of sequential-sampling cost.

    python scripts/overlap_benchmark.py --model model/lean_npe_v3/best_model.pth

Produces analysis/overlap_benchmark.json with four sections:
  per_rank      coverage/SBC/accuracy per NPE rank, grouped by multiplicity
  dt_bins       2-signal events binned by true merger separation |dt|
  ranking       PriorityNet ordering accuracy vs true loudness order
  runtime       measured k-rank NPE wall time vs sequential dynesty baseline
Figures are rendered from the JSON by analysis.py (one chart per file).
"""
import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _ROOT)
sys.path.insert(0, f"{_ROOT}/src")

import numpy as np
import torch
from scipy import stats as sps

from experiments.remix_data import RemixDataset
from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES

JM1, JM2 = PARAM_NAMES.index("mass_1"), PARAM_NAMES.index("mass_2")
JD, JT = PARAM_NAMES.index("luminosity_distance"), PARAM_NAMES.index("geocent_time")
HEADLINE = ["mass_1", "luminosity_distance", "geocent_time", "ra", "dec"]


def mchirp(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def sample_ranks(model, strain, max_rank, K, device, batch_ev=24):
    """Posterior draws for ranks 0..max_rank-1 of each event.
    strain [N,3,T] -> samples [N, max_rank, K, 11]."""
    N = strain.shape[0]
    out = np.empty((N, max_rank, K, len(PARAM_NAMES)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, batch_ev):
            st = strain[i:i + batch_ev].to(device).float()
            ctx = model.encoder(st)
            nb = ctx.shape[0]
            for r in range(max_rank):
                rk = torch.full((nb,), r, dtype=torch.long, device=device)
                fc = model._full_context(ctx, rk)
                fcr = fc.unsqueeze(1).expand(nb, K, fc.shape[1]).reshape(nb * K, -1)
                z = torch.randn(nb * K, len(PARAM_NAMES), device=device)
                y, _ = model.flow.inverse(z, fcr)
                y = model.scaler.wrap(y)
                s = model.scaler.denormalize(y).reshape(nb, K, -1).cpu().numpy()
                m1 = np.maximum(s[..., JM1], s[..., JM2])
                m2 = np.minimum(s[..., JM1], s[..., JM2])
                s[..., JM1], s[..., JM2] = m1, m2
                out[i:i + nb, r] = s
    return out


def calib_block(samples, truth):
    """samples [n,K,11], truth [n,11] -> coverage/err/SBC summary."""
    med = np.median(samples, axis=1)
    blk = {"n": int(len(truth))}
    for lev, tag in ((0.50, "cov50"), (0.90, "cov90")):
        lo = np.quantile(samples, 0.5 - lev / 2, axis=1)
        hi = np.quantile(samples, 0.5 + lev / 2, axis=1)
        inside = (truth >= lo) & (truth <= hi)
        blk[tag] = {p: round(float(inside[:, j].mean()), 3)
                    for j, p in enumerate(PARAM_NAMES)}
        blk[f"{tag}_headline_mean"] = round(float(np.mean(
            [inside[:, PARAM_NAMES.index(p)].mean() for p in HEADLINE])), 3)
    mc_t = mchirp(truth[:, JM1], truth[:, JM2])
    mc_p = mchirp(med[:, JM1], med[:, JM2])
    blk["mc_frac_err_median"] = round(float(np.median(np.abs(mc_p - mc_t) / mc_t)), 4)
    blk["dl_frac_err_median"] = round(float(np.median(
        np.abs(med[:, JD] - truth[:, JD]) / truth[:, JD])), 4)
    blk["tc_abs_err_median_s"] = round(float(np.median(
        np.abs(med[:, JT] - truth[:, JT]))), 4)
    ranks = (samples < truth[:, None, :]).mean(axis=1)
    blk["sbc_ks_p"] = {p: round(float(sps.kstest(ranks[:, j], "uniform").pvalue), 4)
                       for j, p in enumerate(PARAM_NAMES)}
    return blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/lean_npe_v3/best_model.pth")
    ap.add_argument("--dataset ", default="data/dataset/memmap/validation")
    ap.add_argument("--per_group", type=int, default=400,
                    help="events per multiplicity group (1,2,3+)")
    ap.add_argument("--n_post", type=int, default=400)
    ap.add_argument("--out", default="analysis/overlap_benchmark.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = LeanNPE(premerger=ckpt["args"].get("premerger", False), psd_cond=ckpt["args"].get("psd_cond", False) or False, psd_bands=ckpt["args"].get("psd_bands", 16), encoder_type=ckpt["args"].get("encoder_type", "conv"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"model: {args.model} (epoch {ckpt['epoch']})")

    ds = RemixDataset(args.dataset, remix=False, seed=1234)
    groups = {1: [], 2: [], 3: []}   # 3 == "3+"
    for i in range(len(ds)):
        _, nsig = ds.events[i]
        g = min(int(nsig), 3)
        if len(groups[g]) < args.per_group:
            groups[g].append(i)
        if all(len(v) >= args.per_group for v in groups.values()):
            break
    print("group sizes:", {k: len(v) for k, v in groups.items()})

    report = {"model": args.model, "epoch": int(ckpt["epoch"]),
              "n_post": args.n_post, "per_rank": {}, "dt_bins": [],
              "ranking": {}, "runtime": {}}

    # ---- per-rank calibration by multiplicity ----
    truths, strains, samples = {}, {}, {}
    for g, idxs in groups.items():
        items = [ds[i] for i in idxs]
        strain = torch.stack([x[0] for x in items])
        pv = torch.stack([x[1] for x in items]).numpy()      # [n, 5, 11]
        nsig = np.array([int(x[2]) for x in items])
        max_rank = min(g, 3)
        t0 = time.time()
        smp = sample_ranks(model, strain, max_rank, args.n_post, device)
        wall = time.time() - t0
        truths[g], strains[g], samples[g] = pv, strain, smp
        report["runtime"][f"npe_all_ranks_s_per_event_n{g}"] = round(wall / len(idxs), 3)
        for r in range(max_rank):
            has = nsig > r
            blk = calib_block(smp[has, r], pv[has, r])
            report["per_rank"][f"n{g}_rank{r}"] = blk
            print(f"  n={g} rank {r}: cov90(headline) {blk['cov90_headline_mean']}"
                  f"  Mc err {blk['mc_frac_err_median']:.3f}"
                  f"  dL err {blk['dl_frac_err_median']:.3f}  (n={blk['n']})")

    # ---- dt robustness (2-signal events) ----
    pv2, smp2 = truths[2], samples[2]
    dt = np.abs(pv2[:, 0, JT] - pv2[:, 1, JT])
    edges = [0.0, 0.25, 0.5, 1.0, 2.0, 3.2]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dt >= lo) & (dt < hi)
        if m.sum() < 15:
            continue
        row = {"dt_lo": lo, "dt_hi": hi, "n": int(m.sum())}
        for r in (0, 1):
            blk = calib_block(smp2[m, r], pv2[m, r])
            row[f"rank{r}"] = {"cov90_headline": blk["cov90_headline_mean"],
                               "mc_frac_err": blk["mc_frac_err_median"],
                               "dl_frac_err": blk["dl_frac_err_median"]}
        report["dt_bins"].append(row)
        print(f"  dt [{lo},{hi}): n={row['n']} "
              f"r0 Mc {row['rank0']['mc_frac_err']:.3f} "
              f"r1 Mc {row['rank1']['mc_frac_err']:.3f}")

    # ---- PriorityNet ranking accuracy ----
    try:
        from ahsd.core.priority_net import load_priority_net
        from ahsd.inference.ranking import _snr_proxy
        pn = load_priority_net("model/pn/priority_net_best.pth", device="cpu")
        top1, taus, top1_dt = [], [], []
        pv2m = np.median(samples[2], axis=2)                 # [n, ranks, 11]
        for k in range(len(pv2)):
            dets, segs = [], []
            for r in range(2):
                med = {p: float(pv2m[k, r, j]) for j, p in enumerate(PARAM_NAMES)}
                det = {p: med[p] for p in ("mass_1", "mass_2", "luminosity_distance",
                                           "ra", "dec", "geocent_time", "theta_jn",
                                           "psi", "phase")}
                det["a_1"], det["a_2"] = med["a1"], med["a2"]
                det["tilt_1"] = det["tilt_2"] = det["phi_12"] = det["phi_jl"] = 0.0
                det["network_snr"] = _snr_proxy(med["mass_1"], med["mass_2"],
                                                med["luminosity_distance"])
                dets.append(det)
                c = int(round((med["geocent_time"] + 2.0) * 4096))
                i0 = int(np.clip(c - 1024, 0, 16384 - 2048))
                segs.append(strains[2][k, :, i0:i0 + 2048].numpy())
            order = pn.rank_detections(dets, torch.from_numpy(np.stack(segs)).float())
            # truth loudness order is (0, 1) by construction (sorted storage)
            top1.append(order[0] == 0)
            taus.append(sps.kendalltau([0, 1], order).statistic)
            top1_dt.append((dt[k], order[0] == 0))
        report["ranking"] = {
            "n": len(top1),
            "top1_accuracy": round(float(np.mean(top1)), 3),
            "kendall_tau_mean": round(float(np.nanmean(taus)), 3)}
        # top-1 accuracy binned by dt
        arr = np.array(top1_dt)
        by_dt = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
            if m.sum() >= 15:
                by_dt.append({"dt_lo": lo, "dt_hi": hi,
                              "top1": round(float(arr[m, 1].mean()), 3),
                              "n": int(m.sum())})
        report["ranking"]["top1_by_dt"] = by_dt
        print(f"  PriorityNet top-1 {report['ranking']['top1_accuracy']}, "
              f"tau {report['ranking']['kendall_tau_mean']}")
    except Exception as e:
        report["ranking"] = {"error": str(e)}
        print(f"  ranking skipped: {e}")

    # ---- runtime vs sequential sampling ----
    bench = Path("results/real_event_benchmark/summary.json")
    dyn_s = None
    if bench.exists():
        rows = [r for r in json.load(open(bench))
                if "error" not in r and r.get("t_dynesty_s") and r["event"] != "GW170817"]
        dyn_s = float(np.median([r["t_dynesty_s"] for r in rows]))
    report["runtime"]["dynesty_single_signal_median_s"] = dyn_s
    report["runtime"]["note"] = (
        "sequential baseline = k x single-signal dynesty (median over the "
        "real-event benchmark BBHs); a joint 11k-dimensional sampler or "
        "hierarchical subtraction would be slower still")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
