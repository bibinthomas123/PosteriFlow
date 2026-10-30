"""M1 - Verify flow overconfidence (posterior sharpening), ep18 -> ep19.

Clean within-run comparison using STORED posteriors (no weights needed):
ep18 = analysis/ci/long_run_ep18/<ev>/posterior_samples.npy
ep19 = analysis/divergence_localization/events_ep19/<ev>/samples_phys.npy
Both from identical events/preprocessing, 3000 samples, rank 0.

Per parameter: physical std, 90% CI width, scaled variance, scaled differential
entropy (Gaussian approx 0.5 ln(2 pi e var) + KDE), and a joint-entropy proxy.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis/divergence_localization"))
from _events import EVENTS, PROBLEM  # noqa: E402
from ahsd.models.lean_npe import ParamScaler, PARAM_NAMES  # noqa: E402

HERE = Path(__file__).resolve().parent
scaler = ParamScaler()
CIRC = scaler.circ_mask.numpy()
SCI = [PARAM_NAMES.index(p) for p in ("mass_1", "mass_2", "luminosity_distance")]
# ep18 diag val NLL vs ep19 diag val NLL (from history/CI)
VAL_NLL = {"ep18": 0.996, "ep19": 0.864}


def scaled(phys):
    return scaler.normalize(torch.from_numpy(phys.astype(np.float32))).numpy()


def gauss_entropy(var):
    return 0.5 * np.log(2 * np.pi * np.e * np.maximum(var, 1e-12))


def metrics(phys):
    z = scaled(phys)
    out = {}
    for j, name in enumerate(PARAM_NAMES):
        col_p = phys[:, j]
        col_z = z[:, j]
        ci90_p = np.subtract(*np.percentile(col_p, [95, 5]))
        var_z = float(np.var(col_z))
        out[name] = dict(std_phys=float(np.std(col_p)),
                         ci90_phys=float(ci90_p),
                         var_scaled=var_z,
                         ci90_scaled=float(np.subtract(*np.percentile(col_z, [95, 5]))),
                         H_scaled=float(gauss_entropy(var_z)))
    # joint-entropy proxy: sum of scaled log-std over NON-circular params
    nc = [j for j in range(len(PARAM_NAMES)) if not CIRC[j]]
    out["_joint"] = dict(
        sum_logstd_scaled=float(sum(0.5 * np.log(out[PARAM_NAMES[j]]["var_scaled"]) for j in nc)),
        mean_ci90_scaled_sci=float(np.mean([out[PARAM_NAMES[j]]["ci90_scaled"] for j in SCI])),
        mean_var_scaled_sci=float(np.mean([out[PARAM_NAMES[j]]["var_scaled"] for j in SCI])),
    )
    return out


def load_ep18(ev):
    return np.load(f"analysis/ci/long_run_ep18/{ev.lower()}/posterior_samples.npy")


def load_ep19(ev):
    return np.load(HERE.parent / "events_ep19" / ev / "samples_phys.npy")


def main():
    result = {"val_nll": VAL_NLL}
    print("=== M1: POSTERIOR SHARPENING ep18 -> ep19 ===")
    print(f"val NLL improved {VAL_NLL['ep18']} -> {VAL_NLL['ep19']} "
          f"(lower=better)\n")

    # per-event science-param width + scaled variance ratio
    print("Science-param posterior WIDTH (mean scaled 90% CI) and VARIANCE, "
          "ep18 -> ep19:\n")
    print(f"{'event':10s} {'ci90_18':>8} {'ci90_19':>8} {'ratio':>6} | "
          f"{'var_18':>7} {'var_19':>7} {'ratio':>6}")
    for ev in EVENTS:
        m18 = metrics(load_ep18(ev)); m19 = metrics(load_ep19(ev))
        result[ev] = {"ep18": m18, "ep19": m19}
        c18 = m18["_joint"]["mean_ci90_scaled_sci"]; c19 = m19["_joint"]["mean_ci90_scaled_sci"]
        v18 = m18["_joint"]["mean_var_scaled_sci"]; v19 = m19["_joint"]["mean_var_scaled_sci"]
        flag = " <==" if ev in PROBLEM else ""
        print(f"{ev:10s} {c18:8.3f} {c19:8.3f} {c19/c18:6.2f} | "
              f"{v18:7.4f} {v19:7.4f} {v19/v18:6.2f}{flag}")

    # per-parameter width ratio, averaged over events (which params sharpen?)
    print("\nPer-parameter scaled-variance ratio (ep19/ep18), mean over 6 events:\n")
    print(f"{'param':20s} {'var18':>8} {'var19':>8} {'ratio':>6} {'H18':>7} {'H19':>7}")
    for j, name in enumerate(PARAM_NAMES):
        v18 = np.mean([result[ev]["ep18"][name]["var_scaled"] for ev in EVENTS])
        v19 = np.mean([result[ev]["ep19"][name]["var_scaled"] for ev in EVENTS])
        h18 = np.mean([result[ev]["ep18"][name]["H_scaled"] for ev in EVENTS])
        h19 = np.mean([result[ev]["ep19"][name]["H_scaled"] for ev in EVENTS])
        tag = " (sci)" if j in SCI else (" (circ)" if CIRC[j] else "")
        print(f"{name:20s} {v18:8.4f} {v19:8.4f} {v19/v18:6.2f} {h18:7.2f} {h19:7.2f}{tag}")

    json.dump(result, open(HERE / "m1_overconfidence.json", "w"), indent=2, default=float)
    print("\n(scaled = ParamScaler [-1,1] space; H = Gaussian differential entropy)")


if __name__ == "__main__":
    main()
