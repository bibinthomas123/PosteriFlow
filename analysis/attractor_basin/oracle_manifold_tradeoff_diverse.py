"""Expanded version of oracle_manifold_tradeoff.py with a more diverse event
sample: 12 events spanning the FULL true-q range in even deciles (not just
the extreme low/high thirds), so the lambda-vs-recovery trade-off isn't
just characterized at the two extremes of the basin cell.
"""
import sys
from pathlib import Path

import numpy as np
import torch

SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH))
ROOT = SCRATCH.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "gw150914_localization"))

from mf_probe import build_injections, MC_BASE  # noqa: E402
from mf_embedding_tap_v2 import load_model  # noqa: E402
import _common as C  # noqa: E402
from oracle_manifold_tradeoff import (median_q, optimize_with_manifold_penalty,  # noqa: E402
                                      m1_of, N_PC)

CKPT = ROOT / "model" / "lean_npe_mf_scratch" / "best_model.pth"
LAMBDAS = [0.0, 0.01, 0.1, 0.3, 1.0, 3.0]
N_EVENTS = 12
N_STEPS = 250


def main():
    model = load_model(str(CKPT))
    strains, q_true = build_injections()
    xs = torch.from_numpy(strains).float()
    ab = torch.zeros(xs.shape[0], model.encoder.n_detectors, model.encoder.psd_bands)
    with torch.no_grad():
        ctxs = model.encode(xs, ab)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=N_PC).fit(ctxs.numpy())
    mean_vec = torch.from_numpy(pca.mean_.astype(np.float32))
    components = torch.from_numpy(pca.components_.astype(np.float32))
    inv_std = torch.from_numpy((1.0 / np.sqrt(pca.explained_variance_)).astype(np.float32))

    # 12 events spanning deciles of the FULL true-q range (diverse, not just extremes)
    order = np.argsort(q_true)
    decile_positions = np.linspace(0, len(order) - 1, N_EVENTS).astype(int)
    pick = order[decile_positions].tolist()
    print(f"picked {len(pick)} events, true_q range: "
          f"[{q_true[pick].min():.3f}, {q_true[pick].max():.3f}]", flush=True)

    rank0 = torch.zeros(1, dtype=torch.long)
    rank_vec = model.rank_embed(rank0)[0]
    truth_base = dict(C.GW150914_TRUTH)

    # summary accumulators: for each lambda, collect |q_rec - q_true| and D across all events
    summary = {lam: {"abs_err": [], "D": []} for lam in LAMBDAS}

    for i in pick:
        q_t = float(q_true[i])
        m1_t = m1_of(MC_BASE, q_t)
        m2_t = q_t * m1_t
        truth_phys = [m1_t, m2_t, truth_base["luminosity_distance"], truth_base["ra"],
                      truth_base["dec"], truth_base["theta_jn"], truth_base["psi"],
                      truth_base["phase"], truth_base["geocent_time"],
                      truth_base["a1"], truth_base["a2"]]

        print(f"\n=== event {i}  true_q={q_t:.3f} ===", flush=True)
        print(f"{'lambda':>8} | {'D(c,M)':>8} | {'q_recovered':>11} | {'abs_err':>8}", flush=True)
        for lam in LAMBDAS:
            full_opt, D_final, nll_final = optimize_with_manifold_penalty(
                model, ctxs[i], rank_vec, truth_phys,
                mean_vec, components, inv_std, lam, steps=N_STEPS)
            q_rec = median_q(model, full_opt)
            err = abs(q_rec - q_t)
            summary[lam]["abs_err"].append(err)
            summary[lam]["D"].append(D_final)
            print(f"{lam:8.3f} | {D_final:8.2f} | {q_rec:11.3f} | {err:8.3f}", flush=True)

    print("\n\n== SUMMARY across all 12 events ==", flush=True)
    print(f"{'lambda':>8} | {'mean D(c,M)':>12} | {'mean |err|':>11} | {'median |err|':>13}", flush=True)
    for lam in LAMBDAS:
        d = summary[lam]
        print(f"{lam:8.3f} | {np.mean(d['D']):12.2f} | {np.mean(d['abs_err']):11.3f} | "
              f"{np.median(d['abs_err']):13.3f}", flush=True)


if __name__ == "__main__":
    main()
