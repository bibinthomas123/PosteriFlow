"""Training-free lower bound on the concat-fusion experiment: reuse
mf_scratch's OWN already-trained geometry/MF token embeddings (frozen, as
optimized for "add"-mode fusion) and just recombine them via concatenation
instead of addition, post-hoc. Concatenation is provably information-
preserving for a linear probe (a regularized logistic regression on
[gtok; mftok] can always fall back to "ignore gtok, use mftok's own optimal
linear weights" as a special case), so this measures how much of the
additive merge's information loss is attributable to the OPERATION itself
(destructive interference between two summed vectors) vs. something only a
full concat-mode training run could reveal. Not the same as post-training
convergence (gradients would further adapt the embeddings for the new
architecture), but a fast, zero-cost sanity check available right now.
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
from mf_embedding_tap_v2 import load_model, probe  # noqa: E402

CKPT = ROOT / "model" / "lean_npe_mf_scratch" / "best_model.pth"


@torch.no_grad()
def tap(model, strain):
    enc = model.encoder
    strain_c = torch.nan_to_num(strain, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)

    g = enc.geom_mlp(enc._geometry_rel(strain_c))
    gtok = enc.geom_to_tokens(g).reshape(-1, enc.n_geom_tokens, enc.d_model)

    raw = enc._mf_features(strain_c)
    m = enc.mf_mlp(raw)
    mftok_pre = enc.mf_to_tokens(m)
    mftok_gated = enc.mf_gate * mftok_pre.reshape(-1, enc.n_geom_tokens, enc.d_model)

    add_merge = (gtok + mftok_gated).reshape(gtok.shape[0], -1)
    concat_merge = torch.cat([gtok, mftok_gated], dim=1).reshape(gtok.shape[0], -1)

    return dict(gtok_alone=gtok.reshape(gtok.shape[0], -1),
               mftok_alone=mftok_gated.reshape(gtok.shape[0], -1),
               add_merge=add_merge,
               concat_merge_frozen=concat_merge)


def main():
    print("== building basin-cell injections ==", flush=True)
    strains, q = build_injections()
    y = (q > 0.9).astype(int)

    model = load_model(str(CKPT))
    xs = torch.from_numpy(strains).float()

    acc = None
    for i in range(0, xs.shape[0], 32):
        b = xs[i:i + 32]
        out = tap(model, b)
        if acc is None:
            acc = {k: [] for k in out}
        for k, v in out.items():
            acc[k].append(v.numpy())
    acc = {k: np.concatenate(v) for k, v in acc.items()}

    print("\n== frozen-weight merge-operation A/B test (mf_scratch's own trained "
          "embeddings, no training done here) ==", flush=True)
    probe(acc["gtok_alone"], y, q, "gtok alone (geometry)")
    probe(acc["mftok_alone"], y, q, "mftok alone (MF, gated)")
    probe(acc["add_merge"], y, q, "gtok + mftok  (as-trained, ACTUAL)")
    probe(acc["concat_merge_frozen"], y, q, "[gtok ; mftok]  (post-hoc concat, frozen)")


if __name__ == "__main__":
    main()
