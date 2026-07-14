"""STEP 7 - Flow tail audit.

The task frames this as 'StandardScaler + tail_bound=3'. The ACTUAL pipeline
uses ParamScaler (fixed physical<->[-1,1] map, HARD-CLAMPED to [-1,1]), feeding
an NSF with tail_bound=5.0 (global) / FLOW_NORM_BOUND=3.0 (per-param). We report
what actually feeds the spline:

  1. post-clamp scaled targets |z| (what the flow trains on): bounded by 1 by
     construction -> compare to tail_bound.
  2. PRE-clamp normalized targets: how far training params extend beyond +/-1,
     i.e. does the scaler CLIP real training data (creating learned boundary
     pile-up)?  This distinguishes 'railing learned from clipped targets' from
     'flow extrapolation on OOD contexts'.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ahsd.models.lean_npe import ParamScaler, PARAM_NAMES  # noqa: E402
from ahsd.models.parameter_scalers import FLOW_NORM_BOUND  # noqa: E402

MEMMAP = HERE.parents[1] / "data/dataset/memmap/train"
TAIL_BOUND = 5.0   # LeanNPE passes tail_bound=5.0 to NSFPosteriorFlow
scaler = ParamScaler()


def normalize_no_clamp(x):
    """ParamScaler.normalize WITHOUT the final clamp to [-1,1]."""
    lo, hi, logm = scaler.lo, scaler.hi, scaler.log_mask
    x = torch.where(logm, torch.log(x.clamp_min(1e-6)), x)
    return (2 * (x - lo) / (hi - lo) - 1)


def main():
    p = np.load(MEMMAP / "params.npy")
    x = torch.from_numpy(p.astype(np.float32))
    z_clamp = scaler.normalize(x).numpy()            # [-1,1] (post-clamp)
    z_raw = normalize_no_clamp(x).numpy()            # pre-clamp

    print("=== STEP 7: FLOW TAIL AUDIT ===")
    print(f"FLOW_NORM_BOUND (per-param clamp) = {FLOW_NORM_BOUND}, "
          f"tail_bound (spline) = {TAIL_BOUND}\n")
    print(f"{'param':20s} {'|z| med':>8} {'|z| 95%':>8} {'|z| max':>8} "
          f"{'%>±1(clip)':>11} {'%>±3':>7} {'%>±5':>7}")
    summary = {}
    for j, name in enumerate(PARAM_NAMES):
        az = np.abs(z_raw[:, j])
        med = float(np.percentile(az, 50)); p95 = float(np.percentile(az, 95))
        mx = float(az.max())
        clip = float((az > 1.0).mean() * 100)
        gt3 = float((az > 3.0).mean() * 100)
        gt5 = float((az > 5.0).mean() * 100)
        summary[name] = dict(med=med, p95=p95, max=mx, pct_clip=clip,
                             pct_gt3=gt3, pct_gt5=gt5)
        print(f"{name:20s} {med:8.3f} {p95:8.3f} {mx:8.3f} {clip:11.2f} "
              f"{gt3:7.3f} {gt5:7.3f}")

    # aggregate over all params
    az_all = np.abs(z_raw)
    print(f"\nALL params pooled: max|z| = {az_all.max():.3f}, "
          f"frac > ±3 = {(az_all > 3).mean()*100:.4f}%, "
          f"frac > ±5 = {(az_all > 5).mean()*100:.4f}%")
    print(f"Post-clamp targets (what flow trains on): max|z| = "
          f"{np.abs(z_clamp).max():.4f}  (== 1.0 by construction)")

    summary["_pooled"] = dict(max_abs_z_raw=float(az_all.max()),
                              frac_gt3=float((az_all > 3).mean()),
                              frac_gt5=float((az_all > 5).mean()),
                              max_abs_z_clamped=float(np.abs(z_clamp).max()))
    json.dump(summary, open(HERE / "step7_flow_tail.json", "w"), indent=2)


if __name__ == "__main__":
    main()
