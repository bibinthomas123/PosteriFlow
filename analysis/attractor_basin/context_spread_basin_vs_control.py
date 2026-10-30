"""Why doesn't the encoder produce contexts separated enough along q for the
flow to use (flow_oracle_context_test.py: the flow CAN recover q exactly
given an oracle context ~2x the natural context's norm away)? Hypothesis:
this is the SAME basin-cell geometry collapse the original attractor
investigation found (analysis/attractor_basin/REPORT.md Sec 5.2) -- not a
general encoder limitation, but specific to the {tau_HL~+7ms, heavy, quiet}
cell, where the geometry tokens map many different true q to one near-
constant context.

Test: fit the q-informative ridge direction on the basin-cell contexts
(as flow_context_sensitivity.py did), then compare the STD of context
projected onto that direction for:
  (a) the basin cell itself (tau_HL~+7ms, heavy, quiet -- build_injections)
  (b) a matched CONTROL set at the same true-q values and Mc, but Delta t=-7ms
      (non-basin geometry; twin_test6.prep_with_dt)

If (a)'s spread is much smaller than (b)'s, the collapse is basin-cell-
specific (confirms/extends the known mechanism). If they're similar, the
encoder has a general q-representation-scale problem, not a cell-specific
one.
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
from twin_test6 import prep_with_dt  # noqa: E402
import _common as C  # noqa: E402

CKPT = ROOT / "model" / "lean_npe_mf_scratch" / "best_model.pth"


def m1_of(mc, q):
    return mc * (1 + q) ** 0.2 * q ** (-0.6)


def main():
    model = load_model(str(CKPT))
    truth_base = dict(C.GW150914_TRUTH)

    print("== (a) basin-cell contexts (tau~+7ms, heavy, quiet, from build_injections) ==", flush=True)
    strains, q_basin = build_injections()
    xs = torch.from_numpy(strains).float()
    ab = torch.zeros(xs.shape[0], model.encoder.n_detectors, model.encoder.psd_bands)
    with torch.no_grad():
        ctx_basin = model.encode(xs, ab).numpy()

    print("== (b) matched CONTROL contexts (Delta t=-7ms, same Mc, same q range) ==", flush=True)
    rng = np.random.default_rng(0)
    q_control = rng.uniform(q_basin.min(), q_basin.max(), size=60)
    ctx_control = []
    for i, qc in enumerate(q_control):
        m1 = m1_of(MC_BASE, qc)
        p = dict(truth_base, mass_1=m1, mass_2=qc * m1)
        prep = prep_with_dt(p, -7)
        strain = torch.from_numpy(prep.strain.astype(np.float32))[None]
        abz = torch.zeros(1, model.encoder.n_detectors, model.encoder.psd_bands)
        with torch.no_grad():
            c = model.encode(strain, abz).numpy()[0]
        ctx_control.append(c)
        if (i + 1) % 20 == 0:
            print(f"  control {i+1}/{len(q_control)}", flush=True)
    ctx_control = np.array(ctx_control)

    print("\n== fitting q-informative ridge direction on BASIN-CELL contexts ==", flush=True)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(ctx_basin)
    ctx_basin_std = scaler.transform(ctx_basin)
    ridge = Ridge(alpha=100.0).fit(ctx_basin_std, q_basin)
    w = ridge.coef_ / (np.linalg.norm(ridge.coef_) + 1e-12)
    r2_basin = ridge.score(ctx_basin_std, q_basin)

    proj_basin = ctx_basin_std @ w
    ctx_control_std = scaler.transform(ctx_control)  # SAME scaler (basin-cell stats)
    proj_control = ctx_control_std @ w

    print(f"ridge R2 on basin cell (in-sample): {r2_basin:.3f}")
    print(f"\nprojection onto q-direction (standardized context units):")
    print(f"  basin cell   (n={len(proj_basin)}): std={proj_basin.std():.3f}  "
          f"range=[{proj_basin.min():.2f},{proj_basin.max():.2f}]")
    print(f"  control(-7ms)(n={len(proj_control)}): std={proj_control.std():.3f}  "
          f"range=[{proj_control.min():.2f},{proj_control.max():.2f}]")
    print(f"\nratio control_std/basin_std: {proj_control.std()/max(proj_basin.std(),1e-9):.2f}x")

    # also compare RAW (unstandardized) context vector norm-of-difference
    # across the full q range, as a magnitude-level (not just projected) check
    basin_spread = np.linalg.norm(ctx_basin[np.argmax(q_basin)] - ctx_basin[np.argmin(q_basin)])
    control_spread = np.linalg.norm(ctx_control[np.argmax(q_control)] - ctx_control[np.argmin(q_control)])
    print(f"\n||context(q_max) - context(q_min)|| :")
    print(f"  basin cell: {basin_spread:.3f}")
    print(f"  control:    {control_spread:.3f}")


if __name__ == "__main__":
    main()
