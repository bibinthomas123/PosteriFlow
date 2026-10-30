"""Does the flow actually USE the q-information that's demonstrably present
in the context vector (mf_embedding_tap_v2.py measured final-context q-AUC
0.889/R2 0.584 for lean_npe_mf_concat)? The encoder fix improved
representation but not the posterior (REPORT.md 2026-07-22 addendum) -- this
tests the next link in the chain directly, by MANUALLY perturbing a
strain-derived context along the exact direction a ridge regression says is
most q-predictive, holding the strain (and hence the encoder) out of the
loop entirely, and watching what the FLOW does.

  1. Fit ridge(context -> q) on basin-cell contexts; take its unit weight
     direction w_hat as the "q-informative direction" in context space.
  2. For a handful of individual basin-cell events (spanning low/high true
     q), perturb that event's OWN real context along +-3 sigma of w_hat
     (sigma = std of real inter-event variation along w_hat) and resample
     from the flow with a FIXED noise seed, tracking median q vs perturbation.
  3. Control: same-magnitude perturbation along a random direction, and a
     full context-zeroing control, to separate "ignores the q-direction
     specifically" from "ignores context in general" from "is just noisy".

If median q tracks the perturbation monotonically and by a meaningful
amount, the conditioning path is transmitting q information fine and the
bottleneck is elsewhere (e.g. the encoder just doesn't spread real events
far enough along this direction). If it's flat regardless of a *manually
injected* large perturbation, the flow is demonstrably ignoring information
we know is present -- direct evidence for "conditioning interface/flow
ignoring context" as the bottleneck.
"""
import sys
from pathlib import Path

import numpy as np
import torch

SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH))
ROOT = SCRATCH.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mf_probe import build_injections  # noqa: E402
from mf_embedding_tap_v2 import load_model, DEVICE  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

CKPT = ROOT / "model" / "lean_npe_mf_concat" / "best_model.pth"
N_SAMP = 2000
N_EVENTS = 8
PERTURB_SIGMAS = [-3, -2, -1, 0, 1, 2, 3]


@torch.no_grad()
def get_context(model, strain, ab):
    return model.encode(strain, ab)  # [B, context_dim]


@torch.no_grad()
def sample_q_from_context(model, context_row, seed=0, n=N_SAMP):
    """context_row: [context_dim] single event's (possibly perturbed) context."""
    context_row = context_row.to(DEVICE)
    rank0 = torch.zeros(1, dtype=torch.long, device=DEVICE)
    fc = model._full_context(context_row.unsqueeze(0), rank0)   # [1, context_dim+rank_dim]
    fcr = fc.expand(n, -1)
    torch.manual_seed(seed)
    z = torch.randn(n, len(PARAM_NAMES), device=DEVICE)
    y, _ = model.flow.inverse(z, fcr)
    s = model.scaler.denormalize(y).cpu().numpy()
    m1, m2 = s[:, 0], s[:, 1]
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    return float(np.median(q)), float(np.std(q))


def main():
    print("== building basin-cell injections ==", flush=True)
    strains, q_true = build_injections()
    model = load_model(str(CKPT))
    xs = torch.from_numpy(strains).float().to(DEVICE)
    ab = torch.zeros(xs.shape[0], model.encoder.n_detectors, model.encoder.psd_bands, device=DEVICE)

    print("== computing contexts ==", flush=True)
    ctxs = []
    for i in range(0, xs.shape[0], 32):
        ctxs.append(get_context(model, xs[i:i + 32], ab[i:i + 32]).cpu().numpy())
    ctxs = np.concatenate(ctxs)  # [N, context_dim]

    print("== fitting ridge(context -> q) for the q-informative direction ==", flush=True)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(ctxs)
    ctxs_std = scaler.transform(ctxs)
    ridge = Ridge(alpha=100.0).fit(ctxs_std, q_true)
    w = ridge.coef_ / (np.linalg.norm(ridge.coef_) + 1e-12)     # unit direction, STANDARDIZED space
    proj = ctxs_std @ w
    sigma = float(np.std(proj))
    print(f"ridge R2 (train, sanity): {ridge.score(ctxs_std, q_true):.3f}  "
          f"proj std (sigma) = {sigma:.3f}", flush=True)

    rng = np.random.default_rng(0)
    idx_low = np.argsort(q_true)[: len(q_true) // 3]
    idx_high = np.argsort(q_true)[-len(q_true) // 3:]
    pick = list(rng.choice(idx_low, N_EVENTS // 2, replace=False)) + \
          list(rng.choice(idx_high, N_EVENTS - N_EVENTS // 2, replace=False))

    w_rand = rng.normal(size=w.shape)
    w_rand /= np.linalg.norm(w_rand) + 1e-12

    print(f"\n== per-event perturbation scan (n_samp={N_SAMP}, "
          f"sigmas={PERTURB_SIGMAS}) ==", flush=True)
    print(f"{'event':>6} {'true_q':>7} | {'q-direction perturbation (median q per sigma)':<60} | "
          f"{'random-dir (same |t|)':<40} | zero-ctx", flush=True)
    for i in pick:
        c0_std = ctxs_std[i]
        row_qdir, row_rand = [], []
        for t in PERTURB_SIGMAS:
            c_pert_std = c0_std + t * sigma * w
            c_pert = scaler.inverse_transform(c_pert_std[None])[0]
            qm, _ = sample_q_from_context(model, torch.from_numpy(c_pert).float())
            row_qdir.append(qm)
        for t in (-3, 3):
            c_pert_std = c0_std + t * sigma * w_rand
            c_pert = scaler.inverse_transform(c_pert_std[None])[0]
            qm, _ = sample_q_from_context(model, torch.from_numpy(c_pert).float())
            row_rand.append(qm)
        qm_zero, _ = sample_q_from_context(model, torch.zeros(ctxs.shape[1]))
        qdir_str = " ".join(f"{v:.3f}" for v in row_qdir)
        rand_str = " ".join(f"{v:.3f}" for v in row_rand)
        print(f"{i:6d} {q_true[i]:7.3f} | {qdir_str:<60} | {rand_str:<40} | {qm_zero:.3f}", flush=True)


if __name__ == "__main__":
    main()
