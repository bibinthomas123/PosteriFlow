"""Is the oracle solution reachable NEAR the encoder's own latent manifold,
or does correct q-recovery require leaving the manifold entirely?

oracle_context_geometry.py showed the unconstrained oracle context sits
7-9x farther from the basin-cell population's center than any real context
ever lands. That result alone can't distinguish two very different stories:
  (a) unconstrained optimization exploits directions/regions the flow was
      never trained on (pure extrapolation artifact) -- OR
  (b) the flow's genuinely useful q-dependence really does require moving
      far from anywhere the encoder currently produces (a real, hard
      latent-geometry mismatch).

This traces the actual Pareto frontier by solving, for a sweep of lambda:

    min_c   NLL(theta_true | c)  +  lambda * D(c, M_encoder)^2

where D(c, M) is a whitened-PCA Mahalanobis-style distance to the
basin-cell population's own distribution (fit once, held fixed). Reports
q-recovery vs. D(c,M) at each lambda. If q recovers well even at LARGE
lambda (small D forced), correct q is reachable near the manifold and this
is a representation-learning/optimization problem end-to-end training could
plausibly solve. If q-recovery collapses as soon as D is constrained to
anywhere near a real context's typical distance (~2.8, from
oracle_context_geometry.py), that's direct evidence of a fundamental
latent-geometry mismatch, not just insufficient training.
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
N_STEPS = 400
LR = 0.05
LAMBDAS = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
N_PC = 20


def m1_of(mc, q):
    return mc * (1 + q) ** 0.2 * q ** (-0.6)


def median_q(model, full_context_row, seed=0, n=2000):
    rank0 = torch.zeros(1, dtype=torch.long)
    fc = full_context_row.detach().unsqueeze(0).expand(n, -1)
    torch.manual_seed(seed)
    z = torch.randn(n, len(model.param_names))
    with torch.no_grad():
        y, _ = model.flow.inverse(z, fc)
        s = model.scaler.denormalize(y).cpu().numpy()
    m1, m2 = s[:, 0], s[:, 1]
    return float(np.median(np.minimum(m1, m2) / np.maximum(m1, m2)))


def optimize_with_manifold_penalty(model, context0, rank_vec, truth_phys,
                                   mean_vec, components, inv_std, lam,
                                   steps=N_STEPS, lr=LR):
    """context0: [256]. mean_vec/components/inv_std define the fixed,
    precomputed whitened-PCA distance-to-manifold penalty D(c)^2 =
    ||components @ (c-mean) * inv_std||^2, differentiable w.r.t. c."""
    full0 = torch.cat([context0, rank_vec], dim=0).detach()
    full = full0.clone().requires_grad_(True)
    y_true = model.scaler.normalize(torch.tensor([truth_phys], dtype=torch.float32))
    log_sigma_psd = torch.zeros_like(y_true)
    opt = torch.optim.Adam([full], lr=lr)
    ctx_dim = context0.shape[0]
    for _ in range(steps):
        opt.zero_grad()
        nll = model.flow.compute_psd_aware_nll(y_true, full.unsqueeze(0), log_sigma_psd).mean()
        c = full[:ctx_dim]
        z_pc = (components @ (c - mean_vec)) * inv_std  # [N_PC]
        D2 = (z_pc ** 2).sum()
        loss = nll + lam * D2
        loss.backward()
        opt.step()
    c_final = full[:ctx_dim].detach()
    z_pc_final = (components @ (c_final - mean_vec)) * inv_std
    D_final = float(z_pc_final.norm())
    return full.detach(), D_final, float(nll.item())


def main():
    model = load_model(str(CKPT))
    strains, q_true = build_injections()
    xs = torch.from_numpy(strains).float()
    ab = torch.zeros(xs.shape[0], model.encoder.n_detectors, model.encoder.psd_bands)
    with torch.no_grad():
        ctxs = model.encode(xs, ab)

    # Fixed manifold definition: PCA on the full 240-event basin-cell population.
    from sklearn.decomposition import PCA
    pca = PCA(n_components=N_PC).fit(ctxs.numpy())
    mean_vec = torch.from_numpy(pca.mean_.astype(np.float32))
    components = torch.from_numpy(pca.components_.astype(np.float32))          # [N_PC, 256]
    inv_std = torch.from_numpy((1.0 / np.sqrt(pca.explained_variance_)).astype(np.float32))  # [N_PC]

    # typical real-context D(c,M) for reference (should be small/moderate by construction)
    with torch.no_grad():
        real_D = []
        for i in range(0, ctxs.shape[0], 40):
            c = ctxs[i]
            z_pc = (components @ (c - mean_vec)) * inv_std
            real_D.append(float(z_pc.norm()))
    print(f"typical real-context D(c,M) (whitened {N_PC}-PC distance): "
          f"mean={np.mean(real_D):.2f} range=[{min(real_D):.2f},{max(real_D):.2f}]\n", flush=True)

    rank0 = torch.zeros(1, dtype=torch.long)
    rank_vec = model.rank_embed(rank0)[0]
    truth_base = dict(C.GW150914_TRUTH)

    rng = np.random.default_rng(0)
    idx_low = np.argsort(q_true)[: len(q_true) // 3]
    idx_high = np.argsort(q_true)[-len(q_true) // 3:]
    pick = list(rng.choice(idx_low, 2, replace=False)) + list(rng.choice(idx_high, 2, replace=False))

    for i in pick:
        q_t = float(q_true[i])
        m1_t = m1_of(MC_BASE, q_t)
        m2_t = q_t * m1_t
        truth_phys = [m1_t, m2_t, truth_base["luminosity_distance"], truth_base["ra"],
                      truth_base["dec"], truth_base["theta_jn"], truth_base["psi"],
                      truth_base["phase"], truth_base["geocent_time"],
                      truth_base["a1"], truth_base["a2"]]

        print(f"=== event {i}  true_q={q_t:.3f} ===", flush=True)
        print(f"{'lambda':>8} | {'D(c,M)':>8} | {'q_recovered':>11} | {'final NLL':>10}", flush=True)
        for lam in LAMBDAS:
            full_opt, D_final, nll_final = optimize_with_manifold_penalty(
                model, ctxs[i], rank_vec, truth_phys,
                mean_vec, components, inv_std, lam)
            q_rec = median_q(model, full_opt)
            print(f"{lam:8.3f} | {D_final:8.2f} | {q_rec:11.3f} | {nll_final:10.2f}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
