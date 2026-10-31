"""Does the flow actually use the DEDICATED q-conditioning channel
(--q_embed_dim)? Direct follow-up to flow_context_sensitivity.py, which
found the flow's conditioning was equally (in)sensitive to the
q-informative direction as to a random one when that information had to be
discovered inside the 256-d shared context. Now that a small, explicitly-
supervised channel exists specifically for this, this checks: does
perturbing THAT channel move the sampled mass ratio more than an equal-size
perturbation elsewhere in context?

Also reports the q_pred_head's own regression accuracy (is the embedding
itself learning to predict q at all) and repeats the QS truth-tracking scan
and a real-GW150914 check, so this is a full go/no-go, not just the
perturbation test alone.

Usage: conda run -n ahsd python analysis/attractor_basin/q_embed_channel_sensitivity.py [ckpt_path]
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
from twin_test import analyze, make_twin  # noqa: E402
from twin_test6 import prep_with_dt, m1_of  # noqa: E402
import _common as C  # noqa: E402
from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES  # noqa: E402

DEFAULT_CKPT = ROOT / "model" / "lean_npe_q_embed_check" / "best_model.pth"
N_SAMP = 2000


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = LeanNPE(psd_cond=a.get("psd_cond", True), psd_bands=a.get("psd_bands", 16),
                    encoder_type=a.get("encoder_type", "coherent"),
                    mass_param=a.get("mass_param", "mc_q"),
                    geom_dropout=a.get("geom_dropout", 0.0),
                    geom_phase_residual=a.get("geom_phase_residual", False),
                    mf_tokens=a.get("mf_tokens", False),
                    mf_shape_norm=a.get("mf_shape_norm", False),
                    mf_token_fusion=a.get("mf_token_fusion", "add"), mf_bank=a.get("mf_bank", "mf_bank.npz"), pol_features=a.get("pol_features", False),
                    q_embed_dim=a.get("q_embed_dim", 0))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # twin_test.py's analyze() -> _common.sample_from_strain() reads this
    # (normally set by _common.load_model(), which we bypass here to pass
    # q_embed_dim through).
    model._ckpt_meta = dict(epoch=ck.get("epoch"), val_nll=ck.get("val_nll"),
                            psd_cond=a.get("psd_cond"), psd_bands=a.get("psd_bands"))
    print(f"loaded {Path(path).name}: epoch {ck.get('epoch')} val_nll {ck.get('val_nll'):.4f} "
          f"q_embed_dim={a.get('q_embed_dim', 0)} aux_q_embed={a.get('aux_q_embed', 0)}",
          flush=True)
    assert model.q_embed_dim > 0, "checkpoint was not trained with --q_embed_dim"
    return model


@torch.no_grad()
def sample_q_from_full_vector(model, full_context_row, seed=0, n=N_SAMP):
    """full_context_row: [context_dim + q_embed_dim] -- ALREADY includes
    whatever (possibly perturbed) q-embedding should be used; bypasses
    model._full_context, which would otherwise recompute q_embed_head from
    a plain 256-d context and silently discard any injected perturbation."""
    rank0 = torch.zeros(1, dtype=torch.long)
    rank_vec = model.rank_embed(rank0)  # [1, rank_dim]
    fc = torch.cat([full_context_row.unsqueeze(0), rank_vec], dim=1)  # [1, context+q_embed+rank]
    fcr = fc.expand(n, -1)
    torch.manual_seed(seed)
    z = torch.randn(n, len(PARAM_NAMES))
    y, _ = model.flow.inverse(z, fcr)
    s = model.scaler.denormalize(y).cpu().numpy()
    m1, m2 = s[:, 0], s[:, 1]
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    return float(np.median(q))


def part1_dedicated_channel_test(model):
    print("\n== Part 1: dedicated-channel perturbation test ==", flush=True)
    strains, q_true = build_injections()
    xs = torch.from_numpy(strains).float()
    ab = torch.zeros(xs.shape[0], model.encoder.n_detectors, model.encoder.psd_bands)

    with torch.no_grad():
        ctxs = model.encode(xs, ab)              # [N, 256]
        q_embeds = model.q_embed_head(ctxs)       # [N, q_embed_dim]
        q_preds = model.q_pred_head(q_embeds)     # [N, 2] normalized (Mc,q)

    # q_pred_head's own accuracy: does the embedding predict q at all?
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    r2_embed = cross_val_score(reg, q_embeds.numpy(), q_true, cv=KFold(5, shuffle=True, random_state=0),
                               scoring="r2").mean()
    print(f"q_embed_head output -> q, 5-fold ridge R2: {r2_embed:+.3f}  "
          f"(is the DEDICATED embedding itself informative)", flush=True)

    rng = np.random.default_rng(0)
    idx_low = np.argsort(q_true)[: len(q_true) // 3]
    idx_high = np.argsort(q_true)[-len(q_true) // 3:]
    pick = list(rng.choice(idx_low, 4, replace=False)) + list(rng.choice(idx_high, 4, replace=False))

    q_emb_std = float(q_embeds.numpy().std())
    ctx_std = float(ctxs.numpy().std())

    print(f"\n{'event':>6} {'true_q':>7} | base(t=0) | zero q_embed | q_embed x3 | "
          f"random-ctx-dir x3 (same |delta|)", flush=True)
    for i in pick:
        c0 = ctxs[i]
        qe0 = q_embeds[i]

        # baseline: normal context + normal q_embed
        base_full = torch.cat([c0, qe0])
        q_base = sample_q_from_full_vector(model, base_full)

        # zero out the dedicated channel entirely
        zero_full = torch.cat([c0, torch.zeros_like(qe0)])
        q_zero = sample_q_from_full_vector(model, zero_full)

        # scale the dedicated channel by 3x (amplify whatever it's saying)
        amp_full = torch.cat([c0, qe0 * 3.0])
        q_amp = sample_q_from_full_vector(model, amp_full)

        # equal-magnitude perturbation to the BASE context instead (random direction)
        rand_dir = torch.from_numpy(rng.normal(size=c0.shape).astype(np.float32))
        rand_dir = rand_dir / rand_dir.norm() * (qe0.norm() * 2.0)  # match perturbation norm to (3x-1)*qe0 norm roughly
        c_pert = c0 + rand_dir
        randctx_full = torch.cat([c_pert, qe0])
        q_randctx = sample_q_from_full_vector(model, randctx_full)

        print(f"{i:6d} {q_true[i]:7.3f} | {q_base:.3f}    | {q_zero:.3f}       | "
              f"{q_amp:.3f}     | {q_randctx:.3f}", flush=True)


def part2_qs_scan(model):
    print("\n== Part 2: QS truth-tracking scan (Delta t=+7ms, Mc fixed 31.3) ==", flush=True)
    truth = dict(C.GW150914_TRUTH)
    zeros_ab = np.zeros((3, model.encoder.psd_bands), dtype=np.float32)
    for qt in (0.50, 0.65, 0.80, 0.90, 1.00):
        m1 = m1_of(31.3, qt)
        p = dict(truth, mass_1=m1, mass_2=qt * m1)
        prep = prep_with_dt(p, +7)
        analyze(model, prep, p, zeros_ab, f"q={qt:.2f} (m1 {m1:.1f})")


def part3_real_event(model):
    print("\n== Part 3: real GW150914 ==", flush=True)
    truth = dict(C.GW150914_TRUTH)
    prep_real = C.prepare_gw150914()
    ab_real = C.asd_bands_of(prep_real)
    analyze(model, prep_real, truth, ab_real, "real GW150914")


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CKPT)
    model = load_model(ckpt)
    part1_dedicated_channel_test(model)
    part2_qs_scan(model)
    part3_real_event(model)


if __name__ == "__main__":
    main()
