"""Is the oracle-found context shift (flow_oracle_context_test.py) aligned
with the linear q-informative direction a ridge regression finds
(flow_context_sensitivity.py), or does it point somewhere else entirely?

context_spread_basin_vs_control.py showed the basin cell has SUBSTANTIAL
natural context variation along the ridge direction (std ~2.4 in
standardized units) -- more than a non-basin control region, not less. Yet
flow_context_sensitivity.py found amplifying that exact direction by 3x
barely moves the flow's output, while an UNCONSTRAINED gradient search
(flow_oracle_context_test.py) easily finds a working solution ~2x the
context's own norm away. If the oracle delta is nearly ORTHOGONAL to the
ridge direction, that resolves the puzzle: the flow's actual sensitivity to
q lives along a direction standard linear probing never looks at, so
"the context is linearly q-informative" and "the context moves the flow's q
output" are almost unrelated questions.
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
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

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


def main():
    model = load_model(str(CKPT))
    strains, q_true = build_injections()
    xs = torch.from_numpy(strains).float()
    ab = torch.zeros(xs.shape[0], model.encoder.n_detectors, model.encoder.psd_bands)
    with torch.no_grad():
        ctxs = model.encode(xs, ab)

    # ridge direction (same methodology as flow_context_sensitivity.py)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(ctxs.numpy())
    ctx_std = scaler.transform(ctxs.numpy())
    ridge = Ridge(alpha=100.0).fit(ctx_std, q_true)
    w_ridge_std = ridge.coef_ / (np.linalg.norm(ridge.coef_) + 1e-12)  # unit dir, STANDARDIZED space
    # convert to RAW context space (undo standardization) for direct comparison
    # with the oracle delta, which lives in raw context units
    w_ridge_raw = w_ridge_std / scaler.scale_
    w_ridge_raw = w_ridge_raw / np.linalg.norm(w_ridge_raw)

    rank0 = torch.zeros(1, dtype=torch.long)
    rank_vec = model.rank_embed(rank0)[0]
    truth_base = dict(C.GW150914_TRUTH)

    rng = np.random.default_rng(0)
    idx_low = np.argsort(q_true)[: len(q_true) // 3]
    idx_high = np.argsort(q_true)[-len(q_true) // 3:]
    pick = list(rng.choice(idx_low, 3, replace=False)) + list(rng.choice(idx_high, 3, replace=False))

    print(f"\n{'event':>6} {'true_q':>7} | {'cos(delta_oracle, w_ridge)':>27} | "
          f"{'cos(delta_ctx_only, w_ridge)':>29}", flush=True)
    for i in pick:
        q_t = float(q_true[i])
        m1_t = m1_of(MC_BASE, q_t)
        m2_t = q_t * m1_t
        truth_phys = [m1_t, m2_t, truth_base["luminosity_distance"], truth_base["ra"],
                      truth_base["dec"], truth_base["theta_jn"], truth_base["psi"],
                      truth_base["phase"], truth_base["geocent_time"],
                      truth_base["a1"], truth_base["a2"]]

        full_opt = optimize_context_for_truth(model, ctxs[i], rank_vec, truth_phys)
        full0 = torch.cat([ctxs[i], rank_vec], dim=0)
        delta_full = (full_opt - full0).numpy()          # [context+rank] space
        delta_ctx = delta_full[: ctxs.shape[1]]            # just the context part

        cos_full = float(np.dot(delta_full[:len(w_ridge_raw)], w_ridge_raw) /
                         (np.linalg.norm(delta_full[:len(w_ridge_raw)]) + 1e-12))
        cos_ctx = float(np.dot(delta_ctx, w_ridge_raw) /
                        (np.linalg.norm(delta_ctx) + 1e-12))

        print(f"{i:6d} {q_t:7.3f} | {cos_full:27.3f} | {cos_ctx:29.3f}", flush=True)

    print("\n(cos ~ +-1 = oracle moved ALONG the ridge direction; "
          "cos ~ 0 = oracle moved ORTHOGONAL to it -- a direction linear "
          "probing never sees)")


if __name__ == "__main__":
    main()
