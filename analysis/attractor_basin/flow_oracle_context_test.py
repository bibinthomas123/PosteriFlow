"""Oracle-context test: if the flow is GIVEN a context that's been directly
optimized (gradient descent, flow weights frozen) to maximize likelihood at
the TRUE parameters, can it recover the correct mass ratio? This answers the
question flow_context_sensitivity.py's manual perturbation left open: that
test showed the flow ignores a HAND-PICKED (ridge-regression) direction, but
a linear-regression direction is not necessarily the flow's OWN preferred
direction for expressing q. Optimizing the context directly against the
flow's own log_prob removes that ambiguity.

  - If optimization finds a context (via a small, plausible perturbation)
    that makes the flow sample the correct q: the flow HAS the capacity to
    express q-dependence, it's just never given (or never learns to build)
    a context that triggers it. The bottleneck is fully upstream (encoder/
    conditioning-interface), not the flow's decoding capacity.
  - If optimization cannot find such a context even with a large budget: the
    bottleneck is inside the flow's conditioning mechanism itself -- it
    structurally cannot express the needed q-dependence regardless of what
    context it's given.

Uses lean_npe_mf_scratch (per-layer dense context injection is Identical
in kind to lean_npe_mf_concat's; mf_scratch is used here since it's the
deploy candidate and the model on which the original (30,30) collapse and
flow_context_sensitivity.py's negative result were both measured).
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

from mf_probe import build_injections  # noqa: E402
from mf_embedding_tap_v2 import load_model  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

CKPT = ROOT / "model" / "lean_npe_mf_scratch" / "best_model.pth"
N_SAMP = 2000
N_STEPS = 300
LR = 0.05


def median_q(model, full_context_row, seed=0, n=N_SAMP):
    rank0 = torch.zeros(1, dtype=torch.long)
    fc = full_context_row.detach().unsqueeze(0).expand(n, -1)
    torch.manual_seed(seed)
    z = torch.randn(n, len(PARAM_NAMES))
    with torch.no_grad():
        y, _ = model.flow.inverse(z, fc)
        s = model.scaler.denormalize(y).cpu().numpy()
    m1, m2 = s[:, 0], s[:, 1]
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    return float(np.median(q))


def optimize_context_for_truth(model, context0, rank_vec, truth_phys, steps=N_STEPS, lr=LR):
    """Gradient descent on the [context+rank] flow-input vector (model
    weights frozen) to MAXIMIZE the flow's log-likelihood at the true
    parameters -- i.e. find the context an oracle encoder WOULD have
    produced for this exact event, if it existed."""
    full0 = torch.cat([context0, rank_vec], dim=0).detach()
    full = full0.clone().requires_grad_(True)
    y_true = model.scaler.normalize(torch.tensor([truth_phys], dtype=torch.float32))
    log_sigma_psd = torch.zeros_like(y_true)  # matches LeanNPE.nll()'s own convention
    opt = torch.optim.Adam([full], lr=lr)
    for step in range(steps):
        opt.zero_grad()
        # NSFPosteriorFlow.log_prob() calls nflows' Flow.log_prob(context=...),
        # which is incompatible with the PSDScaledNormal base distribution
        # (its .log_prob() takes log_sigma_psd, not context) -- the actual
        # training path (LeanNPE.nll()) uses compute_psd_aware_nll() instead,
        # which goes through forward()+base_dist.log_prob() directly and is
        # what every checkpoint here was actually optimized against.
        nll = model.flow.compute_psd_aware_nll(y_true, full.unsqueeze(0), log_sigma_psd)
        loss = nll.mean()
        loss.backward()
        opt.step()
    delta_norm = float((full - full0).norm())
    orig_norm = float(full0.norm())
    return full.detach(), delta_norm, orig_norm, float(loss.item())


def main():
    print("== building basin-cell injections ==", flush=True)
    strains, q_true = build_injections()
    model = load_model(str(CKPT))
    xs = torch.from_numpy(strains).float()
    ab = torch.zeros(xs.shape[0], model.encoder.n_detectors, model.encoder.psd_bands)

    with torch.no_grad():
        ctxs = model.encode(xs, ab)
    rank0 = torch.zeros(1, dtype=torch.long)
    rank_vec = model.rank_embed(rank0)[0]

    rng = np.random.default_rng(0)
    idx_low = np.argsort(q_true)[: len(q_true) // 3]
    idx_high = np.argsort(q_true)[-len(q_true) // 3:]
    pick = list(rng.choice(idx_low, 4, replace=False)) + list(rng.choice(idx_high, 4, replace=False))

    # Need PHYSICAL truth params in PARAM_NAMES order for scaler.normalize.
    # build_injections' basin cell fixes Mc=MC_BASE and everything else at
    # the GW150914-truth-like fixed values except q; reconstruct m1,m2 from
    # (Mc, q) the same way twin_test6.m1_of does.
    from mf_probe import MC_BASE
    import _common as C
    truth_base = dict(C.GW150914_TRUTH)

    def m1_of(mc, q):
        return mc * (1 + q) ** 0.2 * q ** (-0.6)

    print(f"\n{'event':>6} {'true_q':>7} | {'q before':>9} | {'q after (oracle)':>16} | "
          f"{'|delta context|':>15} | {'|context|':>9} | {'final NLL':>10}", flush=True)
    for i in pick:
        q_t = float(q_true[i])
        m1_t = m1_of(MC_BASE, q_t)
        m2_t = q_t * m1_t
        truth_phys = [m1_t, m2_t, truth_base["luminosity_distance"], truth_base["ra"],
                      truth_base["dec"], truth_base["theta_jn"], truth_base["psi"],
                      truth_base["phase"], truth_base["geocent_time"],
                      truth_base["a1"], truth_base["a2"]]

        full0 = torch.cat([ctxs[i], rank_vec], dim=0)
        q_before = median_q(model, full0)

        full_opt, dnorm, onorm, final_nll = optimize_context_for_truth(
            model, ctxs[i], rank_vec, truth_phys)
        q_after = median_q(model, full_opt)

        print(f"{i:6d} {q_t:7.3f} | {q_before:9.3f} | {q_after:16.3f} | "
              f"{dnorm:15.2f} | {onorm:9.2f} | {final_nll:10.2f}", flush=True)


if __name__ == "__main__":
    main()
