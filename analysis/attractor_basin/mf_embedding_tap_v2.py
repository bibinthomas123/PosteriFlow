"""Generalized version of mf_embedding_tap.py + layer_pool_probe.py that
works for BOTH mf_token_fusion modes ("add" legacy vs "concat", 2026-07-22
architecture experiment), so the same information trace can be run on
lean_npe_mf_scratch (add) and lean_npe_mf_concat (concat) for direct
comparison. Traces:

    raw _mf_features() -> mf_mlp -> mf_to_tokens -> *mf_gate
        -> [add: + gtok | concat: cat with gtok] = extra_tokens
        -> fusion layer 0 -> ... -> pool_attn -> final context (out_proj)

Usage: conda run -n ahsd python analysis/attractor_basin/mf_embedding_tap_v2.py [ckpt_path]
Default ckpt: model/lean_npe_mf_concat/best_model.pth
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
from ahsd.models.lean_npe import LeanNPE  # noqa: E402

DEFAULT_CKPT = ROOT / "model" / "lean_npe_mf_concat" / "best_model.pth"

# Diagnostic/analysis scripts in this directory default to CPU (fine for
# quick one-off probes) but MPS is available on this machine and these
# scripts have gotten heavy enough (oracle context optimization, manifold
# sweeps) that it's worth using it. DEVICE is exported so other scripts can
# `from mf_embedding_tap_v2 import DEVICE` and move their own tensors (xs,
# asd_bands, etc.) there too -- moving only the model isn't enough, every
# input tensor fed to it must match.
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))


def load_model(path, device=DEVICE):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = LeanNPE(psd_cond=a.get("psd_cond", True), psd_bands=a.get("psd_bands", 16),
                    encoder_type=a.get("encoder_type", "coherent"),
                    mass_param=a.get("mass_param", "mc_q"),
                    geom_dropout=a.get("geom_dropout", 0.0),
                    geom_phase_residual=a.get("geom_phase_residual", False),
                    mf_tokens=a.get("mf_tokens", False),
                    mf_shape_norm=a.get("mf_shape_norm", False),
                    mf_token_fusion=a.get("mf_token_fusion", "add"), mf_bank=a.get("mf_bank", "mf_bank.npz"), pol_features=a.get("pol_features", False), q_embed_dim=a.get("q_embed_dim", 0))
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"loaded {Path(path).name}: epoch {ck.get('epoch')} val_nll {ck.get('val_nll'):.4f} "
          f"mf_token_fusion={a.get('mf_token_fusion', 'add')} mf_tokens={a.get('mf_tokens')} "
          f"device={device}", flush=True)
    return model


@torch.no_grad()
def manual_forward_capture(model, strain, asd_bands):
    """Reimplements CoherentEncoder.forward + LeanStrainEncoder._compute_feats
    step by step, capturing every MF-branch intermediate stage plus each
    fusion-layer mean-pooled representation and the final pooled/context."""
    enc = model.encoder
    B, D, T = strain.shape
    strain_c = torch.nan_to_num(strain, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)

    g = enc.geom_mlp(enc._geometry_rel(strain_c))
    gtok = enc.geom_to_tokens(g).reshape(-1, enc.n_geom_tokens, enc.d_model)
    use_type_embed = enc.use_mf_tokens and enc.mf_token_fusion == "concat"
    if use_type_embed:
        gtok = gtok + enc.token_type_embed[0]

    mf_stages = {}
    if enc.use_mf_tokens:
        raw = enc._mf_features(strain_c)
        m = enc.mf_mlp(raw)
        mftok_pre = enc.mf_to_tokens(m)
        mftok_gated = enc.mf_gate * mftok_pre.reshape(-1, enc.n_geom_tokens, enc.d_model)
        mf_stages = dict(raw_mf=raw, mf_mlp_out=m,
                         mftok_pre_gate=mftok_pre.reshape(mftok_pre.shape[0], -1),
                         mftok_gated=mftok_gated.reshape(mftok_gated.shape[0], -1))
        if enc.mf_token_fusion == "concat":
            mftok_gated = mftok_gated + enc.token_type_embed[1]
            extra_tokens = torch.cat([gtok, mftok_gated], dim=1)
        else:
            extra_tokens = gtok + mftok_gated
    else:
        extra_tokens = gtok
    mf_stages["extra_tokens"] = extra_tokens.reshape(extra_tokens.shape[0], -1)

    # conv token stream
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

    layer_reprs = []
    xt = tokens
    for layer in enc.fusion.layers:
        xt = layer(xt)
        layer_reprs.append(xt.mean(dim=1).clone())

    q = enc.pool_queries.unsqueeze(0).expand(B, -1, -1)
    pooled, attn_w = enc.pool_attn(q, xt, xt, need_weights=True, average_attn_weights=True)

    feats = [pooled.reshape(B, -1), energy_feat]
    if enc.psd_bands > 0:
        if asd_bands is None:
            asd_bands = strain.new_zeros(B, enc.n_detectors, enc.psd_bands)
        feats.append(enc.noise_mlp(asd_bands.reshape(B, -1)))
    final_context = model.encoder.out_proj(torch.cat(feats, dim=1))

    return mf_stages, layer_reprs, pooled.reshape(B, -1), final_context


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


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CKPT)
    print("== building basin-cell injections (q in [0.40,0.75] vs [0.92,1.00]) ==", flush=True)
    strains, q = build_injections()
    y = (q > 0.9).astype(int)

    model = load_model(ckpt)
    xs = torch.from_numpy(strains).float().to(DEVICE)
    ab = torch.zeros(xs.shape[0], model.encoder.n_detectors, model.encoder.psd_bands, device=DEVICE)

    mf_acc, layer_acc, pooled_acc, final_acc = None, None, [], []
    for i in range(0, xs.shape[0], 32):
        b, bab = xs[i:i + 32], ab[i:i + 32]
        mf_stages, layer_reprs, pooled, final_context = manual_forward_capture(model, b, bab)
        if mf_acc is None:
            mf_acc = {k: [] for k in mf_stages}
            layer_acc = [[] for _ in layer_reprs]
        for k, v in mf_stages.items():
            mf_acc[k].append(v.cpu().numpy())
        for li, lr in enumerate(layer_reprs):
            layer_acc[li].append(lr.cpu().numpy())
        pooled_acc.append(pooled.cpu().numpy())
        final_acc.append(final_context.cpu().numpy())

    mf_acc = {k: np.concatenate(v) for k, v in mf_acc.items()}
    layer_acc = [np.concatenate(c) for c in layer_acc]
    pooled_acc = np.concatenate(pooled_acc)
    final_acc = np.concatenate(final_acc)

    print(f"\n== full information trace ({model.encoder.mf_token_fusion}-fusion) ==", flush=True)
    order = [k for k in ("raw_mf", "mf_mlp_out", "mftok_pre_gate", "mftok_gated")
            if k in mf_acc] + ["extra_tokens"]
    for name in order:
        probe(mf_acc[name], y, q, name)
    for li, X in enumerate(layer_acc):
        probe(X, y, q, f"fusion layer {li}")
    probe(pooled_acc, y, q, "pool_attn output")
    probe(final_acc, y, q, "final context (out_proj)")


if __name__ == "__main__":
    main()
