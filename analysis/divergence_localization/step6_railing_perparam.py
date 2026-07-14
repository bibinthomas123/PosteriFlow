"""STEP 6 - Per-parameter railing localization (and ep18->ep19 delta for Step 3).

Railing = posterior samples piling at the ParamScaler [-1,1] boundary for a
NON-circular parameter (circular ra/phase/psi wrap, so boundary pile-up there
is not railing). We normalize the stored physical posteriors back through the
exact ParamScaler and count |scaled|>0.999 per parameter, for ep18 (stored CI
posteriors) and ep19 (current checkpoint). Also reports robust posterior width
(scaled IQR) to test the 'flow got sharper' hypothesis independent of clamping.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _events import EVENTS  # noqa: E402
from ahsd.models.lean_npe import ParamScaler, PARAM_NAMES  # noqa: E402

DERIVED = ["chirp_mass", "mass_ratio"]
scaler = ParamScaler()
CIRC = scaler.circ_mask.numpy()
RAIL_EPS = 0.999


def scaled(phys):
    """physical [N,11] -> scaled [-1,1] via exact ParamScaler."""
    return scaler.normalize(torch.from_numpy(phys.astype(np.float32))).numpy()


def rail_per_param(phys):
    z = scaled(phys)
    out = {}
    for j, name in enumerate(PARAM_NAMES):
        if CIRC[j]:
            out[name] = None  # circular: wraps, not railing
        else:
            out[name] = float((np.abs(z[:, j]) > RAIL_EPS).mean())
    return out, z


def iqr_scaled(z):
    return {PARAM_NAMES[j]: float(np.subtract(*np.percentile(z[:, j], [75, 25])))
            for j in range(len(PARAM_NAMES))}


def load_ep18(ev):
    p = f"analysis/ci/long_run_ep18/{ev.lower()}/posterior_samples.npy"
    return np.load(p)


def load_ep19(ev):
    return np.load(HERE / "events_ep19" / ev / "samples_phys.npy")


def main():
    result = {}
    print("=== STEP 6: PER-PARAMETER RAILING  (ep18 -> ep19) ===\n")
    non_circ = [p for j, p in enumerate(PARAM_NAMES) if not CIRC[j]]
    hdr = f"{'event':10s} {'epoch':5s} " + " ".join(f"{p[:6]:>6}" for p in non_circ)
    print(hdr)
    for ev in EVENTS:
        result[ev] = {}
        for tag, loader in [("ep18", load_ep18), ("ep19", load_ep19)]:
            try:
                phys = loader(ev)
            except Exception as e:
                print(f"{ev} {tag}: MISSING ({e})"); continue
            rp, z = rail_per_param(phys)
            result[ev][tag] = {"rail": rp, "iqr_scaled": iqr_scaled(z)}
            cells = " ".join(f"{rp[p]*100:6.0f}" for p in non_circ)
            print(f"{ev:10s} {tag:5s} {cells}")
        print()

    # width (scaled IQR) delta: mean over non-circular params
    print("Robust posterior width (mean scaled IQR over non-circular params):\n")
    print(f"{'event':10s} {'ep18':>8} {'ep19':>8} {'ratio 19/18':>12}")
    for ev in EVENTS:
        if "ep18" in result[ev] and "ep19" in result[ev]:
            w18 = np.mean([result[ev]["ep18"]["iqr_scaled"][p] for p in non_circ])
            w19 = np.mean([result[ev]["ep19"]["iqr_scaled"][p] for p in non_circ])
            print(f"{ev:10s} {w18:8.3f} {w19:8.3f} {w19/w18:12.2f}")

    # ---- SPURIOUS railing on science params, ep18 vs ep19 (vs published) ----
    # identical methodology on identical events -> clean within-run regression.
    import torch as _t
    print("\n\n=== SPURIOUS science-param railing, ep18 vs ep19 (vs published) ===")
    print("(% samples railing a bound where the PUBLISHED value is not near it)\n")
    sci = ["mass_1", "mass_2", "luminosity_distance"]
    sidx = [PARAM_NAMES.index(p) for p in sci]
    print(f"{'event':10s} {'epoch':5s} {'m1_spur':>8} {'m2_spur':>8} {'dL_spur':>8}")
    spur_out = {}
    for ev in EVENTS:
        spur_out[ev] = {}
        for tag, loader in [("ep18", load_ep18), ("ep19", load_ep19)]:
            try:
                phys = loader(ev)
            except Exception:
                continue
            z = scaled(phys)
            pub = [EVENTS[ev]["m1"], EVENTS[ev]["m2"], EVENTS[ev]["dL"]]
            pub_full = np.median(phys, 0).copy()[None, :]
            for val, j in zip(pub, sidx):
                pub_full[0, j] = val
            y_true = scaler.normalize(_t.from_numpy(pub_full.astype(np.float32))).numpy()[0]
            row = {}
            for p, j in zip(sci, sidx):
                hi = (z[:, j] > 0.999) & (y_true[j] < 0.9)
                lo = (z[:, j] < -0.999) & (y_true[j] > -0.9)
                row[p] = float((hi | lo).mean())
            spur_out[ev][tag] = row
            print(f"{ev:10s} {tag:5s} {row['mass_1']*100:8.0f} "
                  f"{row['mass_2']*100:8.0f} {row['luminosity_distance']*100:8.0f}")
        print()
    result["_science_spurious"] = spur_out
    json.dump(result, open(HERE / "step6_railing.json", "w"), indent=2)
    print("Note: railing shown as % of samples at |scaled|>0.999 (non-circular).")


if __name__ == "__main__":
    main()
