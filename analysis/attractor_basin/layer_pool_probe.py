"""Probes 1+2 of the "where does MF information get lost" investigation
(2026-07-21 follow-up to REPORT.md's mf_probe.py go/no-go).

mf_probe.py already showed: raw MF-overlap features linearly decode q in the
basin cell at AUC 0.989, while the (frozen, pslope2) encoder CONTEXT only
gets 0.821. That's a real drop somewhere between the correlation and the
final pooled vector. This script localizes WHERE, on the actual deployed
checkpoint (model/lean_npe_mf_scratch), which has mf_tokens trained in from
scratch (not the frozen pslope2 checkpoint mf_probe.py used):

  Probe 1 (layer-wise): mean-pool each of the 3 fusion-transformer layers'
    token outputs and probe q/Mc at each depth, plus the true final pooled
    context (via the model's own pool_attn). If AUC is already high at
    layer 0 or 1 and DROPS by the final context, the loss is in later fusion
    layers or in pool_attn/out_proj, not in the token-building stage.

  Probe 2 (pool attention entropy): the predecessor architecture
    (OverlapNeuralPE) had a diagnosed attention-pooling collapse (entropy
    pinned at 1.0, i.e. near-uniform/non-selective). Check whether the
    CURRENT pool_attn (8 learned queries, MultiheadAttention) shows the same
    signature post-MF-tokens, which has never been directly measured.

Usage: conda run -n ahsd python analysis/attractor_basin/layer_pool_probe.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH))
ROOT = SCRATCH.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mf_probe import build_injections, MC_BASE  # noqa: E402
from ahsd.models.lean_npe import LeanNPE  # noqa: E402

CKPT = ROOT / "model/lean_npe_mf_scratch/best_model.pth"


def load_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = LeanNPE(psd_cond=a.get("psd_cond", True), psd_bands=a.get("psd_bands", 16),
                    encoder_type=a.get("encoder_type", "coherent"),
                    mass_param=a.get("mass_param", "mc_q"),
                    geom_dropout=a.get("geom_dropout", 0.0),
                    geom_phase_residual=a.get("geom_phase_residual", False),
                    mf_tokens=a.get("mf_tokens", False),
                    mf_shape_norm=a.get("mf_shape_norm", False), mf_token_fusion=a.get("mf_token_fusion", "add"), q_embed_dim=a.get("q_embed_dim", 0))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"loaded {CKPT.name}: encoder_type={a.get('encoder_type')} "
          f"mf_tokens={a.get('mf_tokens')} mass_param={a.get('mass_param')}", flush=True)
    return model


@torch.no_grad()
def manual_forward_capture(model, strain, asd_bands):
    """Reimplements CoherentEncoder.forward + LeanStrainEncoder._compute_feats
    step by step so every intermediate stage is inspectable. Must stay in
    exact sync with coherent_encoder.py / lean_npe.py forward logic."""
    enc = model.encoder
    B, D, T = strain.shape
    strain_c = torch.nan_to_num(strain, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)

    # geometry tokens
    g = enc.geom_mlp(enc._geometry_rel(strain_c))
    gtok = enc.geom_to_tokens(g).reshape(-1, enc.n_geom_tokens, enc.d_model)
    if enc.use_mf_tokens:
        m = enc.mf_mlp(enc._mf_features(strain_c))
        mftok = enc.mf_to_tokens(m).reshape(-1, enc.n_geom_tokens, enc.d_model)
        gtok = gtok + enc.mf_gate * mftok
    extra_tokens = gtok

    # conv token stream (copy of LeanStrainEncoder._compute_feats, pre-fusion)
    w = enc.n_energy_windows
    win = strain_c[:, :, : (T // w) * w].reshape(B, D, w, -1)
    log_energy = torch.log((win ** 2).mean(dim=-1) + 1e-8)
    energy_feat = enc.energy_mlp(log_energy.reshape(B, -1))

    x = torch.asinh(strain_c).reshape(B * D, 1, T)
    tokens = enc.stem(x).transpose(1, 2)
    L = tokens.shape[1]
    tokens = tokens + enc.pos(L).unsqueeze(0)
    tokens = tokens.reshape(B, D, L, -1)
    det_ids = torch.arange(D, device=strain.device)
    tokens = tokens + enc.detector_embed(det_ids)[None, :, None, :]
    tokens = tokens.reshape(B, D * L, -1)
    tokens = torch.cat([extra_tokens, tokens], dim=1)

    # walk the fusion transformer layer by layer, capturing mean-pooled state
    layer_reprs = []
    xt = tokens
    for layer in enc.fusion.layers:
        xt = layer(xt)
        layer_reprs.append(xt.mean(dim=1).clone())  # [B, d_model]

    # true pooling stage (matches the model's actual forward exactly)
    q = enc.pool_queries.unsqueeze(0).expand(B, -1, -1)
    pooled, attn_w = enc.pool_attn(q, xt, xt, need_weights=True, average_attn_weights=True)
    # attn_w: [B, n_pool_queries, seq_len]

    feats = [pooled.reshape(B, -1), energy_feat]
    if enc.psd_bands > 0:
        if asd_bands is None:
            asd_bands = strain.new_zeros(B, enc.n_detectors, enc.psd_bands)
        feats.append(enc.noise_mlp(asd_bands.reshape(B, -1)))
    final_context = model.encoder.out_proj(torch.cat(feats, dim=1))

    return layer_reprs, pooled, attn_w, final_context, extra_tokens


def probe(X, y_cls, y_q, name):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=5000))
    auc = cross_val_score(clf, X, y_cls, scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    reg = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    r2 = cross_val_score(reg, X, y_q, scoring="r2",
                         cv=KFold(5, shuffle=True, random_state=0)).mean()
    print(f"{name:22s} dim {X.shape[1]:5d}   AUC {auc:.3f}   q-R2 {r2:+.3f}", flush=True)
    return auc, r2


def entropy_of_attn(attn_w):
    """attn_w: [B, n_queries, seq_len] (already softmax'd, sums to 1 over
    seq_len). Returns per-query mean entropy (nats) and the uniform-baseline
    entropy (log(seq_len)) for reference — near-baseline = non-selective
    pooling (the predecessor's diagnosed failure mode)."""
    p = attn_w.clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1)          # [B, n_queries]
    seq_len = attn_w.shape[-1]
    return ent.mean(dim=0).numpy(), float(np.log(seq_len))


def main():
    print("== building basin-cell injections (q in [0.40,0.75] vs [0.92,1.00]) ==", flush=True)
    strains, q = build_injections()
    y = (q > 0.9).astype(int)
    mc_true = np.full(q.shape, MC_BASE, dtype=np.float32)  # Mc fixed in this cell by design

    model = load_model()
    xs = torch.from_numpy(strains).float()
    ab = torch.zeros(xs.shape[0], model.encoder.n_detectors, model.encoder.psd_bands)

    layer_ctx = [[] for _ in range(3)]
    pooled_ctx, final_ctx, ent_all = [], [], []
    seq_len = None
    for i in range(0, xs.shape[0], 32):
        b = xs[i:i + 32]
        bab = ab[i:i + 32]
        layer_reprs, pooled, attn_w, final_context, _ = manual_forward_capture(model, b, bab)
        for li, lr in enumerate(layer_reprs):
            layer_ctx[li].append(lr.numpy())
        pooled_ctx.append(pooled.reshape(pooled.shape[0], -1).numpy())
        final_ctx.append(final_context.numpy())
        ent, base = entropy_of_attn(attn_w)
        ent_all.append(ent)
        seq_len = attn_w.shape[-1]

    layer_ctx = [np.concatenate(c) for c in layer_ctx]
    pooled_ctx = np.concatenate(pooled_ctx)
    final_ctx = np.concatenate(final_ctx)
    ent_all = np.stack(ent_all)  # [n_batches, n_pool_queries]

    print("\n== Probe 1: layer-wise Mc/q linear probe (5-fold CV) ==", flush=True)
    print("(Mc is FIXED in this cell by construction — Mc-R2 near 0 is expected "
          "and just a sanity check that we are not leaking distance/SNR info)", flush=True)
    for li, X in enumerate(layer_ctx):
        probe(X, y, q, f"fusion layer {li}")
    probe(pooled_ctx, y, q, "pool_attn output")
    probe(final_ctx, y, q, "final context (out_proj)")

    print("\n== Probe 2: pool_attn entropy (nats), baseline = log(seq_len) "
          f"= {np.log(seq_len):.3f} (uniform/non-selective) ==", flush=True)
    mean_ent = ent_all.mean(axis=0)
    for qi, e in enumerate(mean_ent):
        frac_of_uniform = e / np.log(seq_len)
        print(f"  query {qi}: entropy {e:.3f}  ({frac_of_uniform*100:.1f}% of uniform)", flush=True)
    print(f"  mean over queries: {mean_ent.mean():.3f}  "
          f"({mean_ent.mean()/np.log(seq_len)*100:.1f}% of uniform)", flush=True)

    np.savez(SCRATCH / "layer_pool_probe_results.npz",
             q=q, seq_len=seq_len, mean_entropy=mean_ent,
             **{f"layer{i}_ctx": c for i, c in enumerate(layer_ctx)},
             pooled_ctx=pooled_ctx, final_ctx=final_ctx)
    print("\nsaved -> layer_pool_probe_results.npz", flush=True)


if __name__ == "__main__":
    main()
