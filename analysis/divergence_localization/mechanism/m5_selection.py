"""M5 - Checkpoint-selection simulation.

Does the current criterion (best val NLL) systematically pick the overconfident
checkpoint? Compare within the coherent + fixed-dataset lineage where val NLL is
comparable: coherentC(ep40), long_run(ep18), long_run(ep19). Calibration axes
(railing, coverage, SBC) come from the stored CI reports (ep18, coherentC) and
from the Step-5 measurement (ep19); val_nll is the trainer's own selection
metric (history.json / checkpoint meta).
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CIDIR = HERE.parents[2] / "analysis/ci"


def ci_gaussian(gen):
    r = json.load(open(CIDIR / gen / "report.json"))
    g = r["gaussian"]
    cov90 = np.mean([v["90"] for v in g["coverage"].values()])
    sbc_pass = np.mean([p > 1e-3 for p in g["sbc_ks_p"].values()])
    return dict(railing=g["railing_frac"],
                cov90_err=abs(cov90 - 0.90),
                cov90=cov90,
                sbc_pass=sbc_pass,
                ci_nll=g["nll"])


def ep19_from_step5():
    s = json.load(open(HERE.parent / "step5_snr_bins.json"))["gaussian"]
    rows = [r for r in s["rows"] if r["n"] >= 10]
    tot = sum(r["n"] for r in rows)
    cov90 = sum(r["cov90_sci"] * r["n"] for r in rows) / tot
    rail = s["overall"]["rail_spur"]
    # SBC: distance KS p across bins is ~1e-19..1e-251 -> 0/representative fail
    sbc_pass = 0.0   # measured: distance SBC p ~ 0 in every bin (Step 5)
    return dict(railing=rail, cov90_err=abs(cov90 - 0.90), cov90=cov90,
                sbc_pass=sbc_pass, ci_nll=float("nan"))


def main():
    # stored selection metric (trainer's val_nll) for the lineage
    VAL_NLL = {"coherentC(ep40)": 1.368, "long_run(ep18)": 0.996,
               "long_run(ep19)": 0.864}
    cal = {
        "coherentC(ep40)": ci_gaussian("lean_coherentC"),
        "long_run(ep18)": ci_gaussian("long_run_ep18"),
        "long_run(ep19)": ep19_from_step5(),
    }
    names = list(VAL_NLL)

    print("=== M5: CHECKPOINT-SELECTION SIMULATION (coherent+fixed lineage) ===\n")
    print(f"{'checkpoint':17s} {'val_nll':>8} {'railing':>8} {'cov90':>7} "
          f"{'|cov-.9|':>8} {'SBCpass':>8}")
    for n in names:
        c = cal[n]
        print(f"{n:17s} {VAL_NLL[n]:>8.3f} {c['railing']:>8.3f} {c['cov90']:>7.2f} "
              f"{c['cov90_err']:>8.3f} {c['sbc_pass']:>8.2f}")

    # combined calibration score: lower = better (railing + cov error + SBC fail)
    def cal_score(n):
        c = cal[n]
        return c["railing"] + c["cov90_err"] + (1 - c["sbc_pass"])

    criteria = {
        "min val_nll (CURRENT)": lambda n: VAL_NLL[n],
        "min railing": lambda n: cal[n]["railing"],
        "min |cov90-0.90|": lambda n: cal[n]["cov90_err"],
        "max SBC pass (=min -pass)": lambda n: -cal[n]["sbc_pass"],
        "min combined calib score": cal_score,
    }
    print("\n=== which checkpoint each criterion SELECTS ===\n")
    for label, fn in criteria.items():
        pick = min(names, key=fn)
        print(f"  {label:28s} -> {pick}")

    print("\ncombined calib score (lower=better):",
          {n: round(cal_score(n), 3) for n in names})
    json.dump({"val_nll": VAL_NLL, "calibration": cal,
               "selected": {label: min(names, key=fn) for label, fn in criteria.items()}},
              open(HERE / "m5_selection.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
