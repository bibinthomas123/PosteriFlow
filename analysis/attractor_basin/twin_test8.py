"""Round 8: white-box causality — do the geometry tokens carry the collapse?

For basin input (design twin Δt+7), non-basin input (Δt-7), and the real
GW150914, sample the posterior with:
  normal   — encoder as-is
  gtok=0   — geometry tokens zeroed
  graft:X  — geometry tokens computed from a donor strain X

If zeroing/grafting un-collapses the basin input (and grafting basin geometry
onto a non-basin strain collapses it), the causal path is geom_mlp -> tokens.
"""
import sys
from pathlib import Path

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))

import numpy as np
import torch

from twin_test6 import prep_with_dt
import _common as C
from ahsd.models.lean_npe import PARAM_NAMES

MODELS = {
    "mcq": str(C.ROOT / "model" / "lean_npe_mcq" / "best_model.pth"),
    "m1m2-old": str(C.ROOT / "model" / "lean_npe" / "best_model.pth"),
}
N = 4000


def ctx_with_gtok(model, strain_np, ab_np, donor_np=None, zero=False):
    """Encoder context with controllable geometry tokens."""
    enc = model.encoder
    s = torch.from_numpy(strain_np[None]).float()
    s = torch.nan_to_num(s, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
    src = s if donor_np is None else torch.nan_to_num(
        torch.from_numpy(donor_np[None]).float(), nan=0.0).clamp(-100.0, 100.0)
    with torch.no_grad():
        g = enc.geom_mlp(enc._geometry_rel(src))
        gtok = enc.geom_to_tokens(g).reshape(-1, enc.n_geom_tokens, enc.d_model)
        if zero:
            gtok = torch.zeros_like(gtok)
        ab = torch.from_numpy(ab_np[None]).float() if ab_np is not None else None
        feats, _ = enc._compute_feats(s, ab, extra_tokens=gtok)
        return enc.out_proj(feats)


def sample_from_ctx(model, ctx, truth, label):
    rank0 = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        fc = model._full_context(ctx, rank0)
        fcr = fc.unsqueeze(1).expand(1, N, fc.shape[1]).reshape(N, -1)
        torch.manual_seed(0)
        z = torch.randn(N, len(PARAM_NAMES))
        y, _ = model.flow.inverse(z, fcr)
        s = model.scaler.denormalize(y).cpu().numpy()
    m1, m2 = s[:, 0], s[:, 1]
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    yn = model.scaler.normalize(torch.from_numpy(s).float()).numpy()
    yt = model.scaler.normalize(torch.tensor([[truth[k] for k in PARAM_NAMES]],
                                             dtype=torch.float32)).numpy()[0]
    noncirc = (~model.scaler.circ_mask).numpy()
    spur = (((yn > 0.999) & (yt[None] < 0.9)) | ((yn < -0.999) & (yt[None] > -0.9)))[:, noncirc]
    print(f"  {label:34} m1 {np.median(m1):5.1f}  m2 {np.median(m2):5.1f}  "
          f"q {np.median(q):.3f}  q>0.95 {float((q > 0.95).mean()):.3f}  "
          f"rail {float(spur.any(1).mean()):.3f}")


def main():
    models = {n: C.load_model(p) for n, p in MODELS.items()}
    truth = dict(C.GW150914_TRUTH)
    zeros_ab = np.zeros((3, 16), dtype=np.float32)

    basin = prep_with_dt(truth, +7).strain.astype(np.float32)
    clean = prep_with_dt(truth, -7).strain.astype(np.float32)
    prep_real = C.prepare_gw150914()
    real = prep_real.strain.astype(np.float32)
    ab_real = C.asd_bands_of(prep_real)

    for name, m in models.items():
        print(f"\n### {name} ###")
        print(" basin strain (dt+7):")
        sample_from_ctx(m, ctx_with_gtok(m, basin, zeros_ab), truth, "normal")
        sample_from_ctx(m, ctx_with_gtok(m, basin, zeros_ab, zero=True), truth, "gtok=0")
        sample_from_ctx(m, ctx_with_gtok(m, basin, zeros_ab, donor_np=clean), truth, "graft: dt-7 geometry")
        print(" non-basin strain (dt-7):")
        sample_from_ctx(m, ctx_with_gtok(m, clean, zeros_ab), truth, "normal")
        sample_from_ctx(m, ctx_with_gtok(m, clean, zeros_ab, donor_np=basin), truth, "graft: dt+7 geometry")
        print(" real GW150914:")
        sample_from_ctx(m, ctx_with_gtok(m, real, ab_real), truth, "normal")
        sample_from_ctx(m, ctx_with_gtok(m, real, ab_real, zero=True), truth, "gtok=0")
        sample_from_ctx(m, ctx_with_gtok(m, real, ab_real, donor_np=clean), truth, "graft: dt-7 geometry")


if __name__ == "__main__":
    main()
