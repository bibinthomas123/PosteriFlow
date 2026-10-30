"""How different is the encoder's real context from the oracle context that
makes the flow recover the true q (flow_oracle_context_test.py)? The
plumbing is confirmed correct (encoder->q_embed_head->flow all wired live);
oracle_vs_ridge_direction.py showed the oracle shift is orthogonal to the
naive linear q-direction. This characterizes the geometry of the gap
directly: cosine similarity / Euclidean distance between real and oracle
context, which dimensions carry the movement (concentrated vs spread), and
where the oracle points land relative to the whole basin-cell population in
PCA space -- "close but slightly off" vs "a completely different region."
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

CKPT = ROOT / "model" / "lean_npe_mf_scratch" / "best_model.pth"
N_STEPS = 300
LR = 0.05


def m1_of(mc, q):
    return mc * (1 + q) ** 0.2 * q ** (-0.6)


def optimize_context_for_truth(model, context0, rank_vec, truth_phys, steps=N_STEPS, lr=LR):
    full0 = torch.cat([context0, rank_vec], dim=0).detach()
    full = full0.clone().requires_grad_(True)
    y_true = model.scaler.normalize(torch.tensor([truth_phys], dtype=torch.float32))
    log_sigma_psd = torch.zeros_like(y_true)
    opt = torch.optim.Adam([full], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = model.flow.compute_psd_aware_nll(y_true, full.unsqueeze(0), log_sigma_psd).mean()
        loss.backward()
        opt.step()
    return full.detach()


def concentration_report(delta_ctx, top_k=(5, 10, 20)):
    """How concentrated is |delta| across the 256 context dims? Report what
    fraction of the total L2 energy the top-k dims carry (1.0/256*k would be
    the 'perfectly spread/uniform' baseline for comparison)."""
    energy = delta_ctx ** 2
    total = energy.sum()
    order = np.argsort(-energy)
    out = {}
    for k in top_k:
        frac = float(energy[order[:k]].sum() / total)
        out[k] = frac
    return out, order


def main():
    model = load_model(str(CKPT))
    strains, q_true = build_injections()
    xs = torch.from_numpy(strains).float()
    ab = torch.zeros(xs.shape[0], model.encoder.n_detectors, model.encoder.psd_bands)
    with torch.no_grad():
        ctxs = model.encode(xs, ab)  # [240, 256]

    rank0 = torch.zeros(1, dtype=torch.long)
    rank_vec = model.rank_embed(rank0)[0]
    truth_base = dict(C.GW150914_TRUTH)

    rng = np.random.default_rng(0)
    idx_low = np.argsort(q_true)[: len(q_true) // 3]
    idx_high = np.argsort(q_true)[-len(q_true) // 3:]
    pick = list(rng.choice(idx_low, 4, replace=False)) + list(rng.choice(idx_high, 4, replace=False))

    oracle_ctxs = np.zeros((len(pick), ctxs.shape[1]), dtype=np.float32)
    real_ctxs_picked = np.zeros_like(oracle_ctxs)

    print(f"\n{'event':>6} {'true_q':>7} | {'cos(real,oracle)':>16} | {'||real-oracle||':>16} | "
          f"{'||real||':>9} | top5/10/20 energy frac (256-d context only)", flush=True)
    for j, i in enumerate(pick):
        q_t = float(q_true[i])
        m1_t = m1_of(MC_BASE, q_t)
        m2_t = q_t * m1_t
        truth_phys = [m1_t, m2_t, truth_base["luminosity_distance"], truth_base["ra"],
                      truth_base["dec"], truth_base["theta_jn"], truth_base["psi"],
                      truth_base["phase"], truth_base["geocent_time"],
                      truth_base["a1"], truth_base["a2"]]

        full_opt = optimize_context_for_truth(model, ctxs[i], rank_vec, truth_phys)
        ctx_real = ctxs[i].numpy()
        ctx_oracle = full_opt.numpy()[: ctxs.shape[1]]  # drop the rank part, context only

        oracle_ctxs[j] = ctx_oracle
        real_ctxs_picked[j] = ctx_real

        cos = float(np.dot(ctx_real, ctx_oracle) /
                    (np.linalg.norm(ctx_real) * np.linalg.norm(ctx_oracle) + 1e-12))
        dist = float(np.linalg.norm(ctx_real - ctx_oracle))
        rnorm = float(np.linalg.norm(ctx_real))

        delta = ctx_oracle - ctx_real
        conc, order = concentration_report(delta)
        top5_dims = order[:5].tolist()

        print(f"{i:6d} {q_t:7.3f} | {cos:16.3f} | {dist:16.2f} | {rnorm:9.2f} | "
              f"{conc[5]:.2f}/{conc[10]:.2f}/{conc[20]:.2f}  (top5 dims: {top5_dims})", flush=True)

    print("\n(concentration fracs: 'perfectly spread' baseline for top5/10/20 out of 256 "
          f"dims = {5/256:.3f}/{10/256:.3f}/{20/256:.3f} -- much higher observed values mean "
          "the movement is concentrated in a handful of dimensions, not spread uniformly)")

    # PCA: fit on the FULL basin-cell population (240 real contexts), project
    # both the population and the 8 oracle points into it.
    print("\n== PCA: where do oracle contexts land relative to the real-context population? ==", flush=True)
    from sklearn.decomposition import PCA
    all_real = ctxs.numpy()  # [240, 256]
    pca = PCA(n_components=10).fit(all_real)
    pop_proj = pca.transform(all_real)
    oracle_proj = pca.transform(oracle_ctxs)
    picked_proj = pca.transform(real_ctxs_picked)

    pop_mean = pop_proj.mean(axis=0)
    pop_std = pop_proj.std(axis=0)
    print(f"explained variance (first 5 PCs): {np.round(pca.explained_variance_ratio_[:5], 3)}")
    print(f"\n{'event':>6} | PC1,PC2 real (z-score) | PC1,PC2 oracle (z-score) | "
          f"oracle Mahalanobis-ish dist from pop mean (10 PCs, in std units)", flush=True)
    for j, i in enumerate(pick):
        real_z = (picked_proj[j, :2] - pop_mean[:2]) / (pop_std[:2] + 1e-9)
        oracle_z = (oracle_proj[j, :2] - pop_mean[:2]) / (pop_std[:2] + 1e-9)
        maha = float(np.sqrt((((oracle_proj[j] - pop_mean) / (pop_std + 1e-9)) ** 2).sum()))
        print(f"{i:6d} | ({real_z[0]:+.2f},{real_z[1]:+.2f})       | "
              f"({oracle_z[0]:+.2f},{oracle_z[1]:+.2f})        | {maha:.2f}", flush=True)

    real_maha = np.sqrt((((picked_proj - pop_mean) / (pop_std + 1e-9)) ** 2).sum(axis=1))
    print(f"\nfor reference, typical REAL point's own Mahalanobis-ish distance from pop mean "
          f"(10 PCs): mean={real_maha.mean():.2f}")


if __name__ == "__main__":
    main()
