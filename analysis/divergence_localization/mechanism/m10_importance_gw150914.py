"""M10 - Does importance reweighting kill the spurious q->1 attractor?

Measured context: on GW150914 the calibrated ep40 NPE puts ~70% of its posterior
mass in a spurious q->1 mode (m1~m2~30), so the MEDIAN is wrong (30.2 vs pub
38.8) even though the 90% CI covers the truth (3/3). Dynesty on the identical
data recovers m1 39.9 / dL 454, proving the mode is a model artifact with poor
likelihood, and that the NPE proposal COVERS the truth -- exactly the
precondition for importance sampling.

This reweights the NPE samples against the true likelihood and reports the
median before/after vs published and vs dynesty.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from ahsd.inference import infer, prepare_real, fetch_gwosc  # noqa: E402
from ahsd.inference.importance import importance_correct  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL = "model/lean_npe/best_model.pth"
GPS = 1126259462.4
PUB = {"mass_1": 38.8, "mass_2": 33.4, "luminosity_distance": 440.0}
DYN = {"mass_1": 39.90, "mass_2": 35.41, "luminosity_distance": 453.7}


def qstats(res):
    m1, m2 = res.samples[:, 0], res.samples[:, 1]
    q = m2 / m1
    return float(np.median(q)), float((q > 0.95).mean())


def main():
    gps, raw, starts, srs, found = fetch_gwosc(GPS)
    print(f"detectors: {found}")
    prepared = prepare_real(raw, gps, starts, srs, seed=0)

    res = infer(strain=prepared, num_samples=4000, model=MODEL, seed=0)
    qm, qf = qstats(res)
    print(f"\nNPE (before): m1 {res.median['mass_1']:.1f}  m2 {res.median['mass_2']:.1f}  "
          f"dL {res.median['luminosity_distance']:.0f} | q med {qm:.3f} frac(q>.95) {qf:.2f}")

    # raw UNWHITENED 4 s crop centered on the trigger (same data the NPE saw)
    raw4 = {}
    for det in found:
        i0 = int(round((gps - starts[det]) * 4096)) - 8192
        raw4[det] = np.asarray(raw[det][i0: i0 + 16384], dtype=np.float64)

    t0 = time.time()
    out = importance_correct(res, raw4, prepared.psds, center_gps=gps,
                             model_path=MODEL, npool=4)
    dt = time.time() - t0
    qm2, qf2 = qstats(out)
    print(f"NPE (reweighted): m1 {out.median['mass_1']:.1f}  m2 {out.median['mass_2']:.1f}  "
          f"dL {out.median['luminosity_distance']:.0f} | q med {qm2:.3f} "
          f"frac(q>.95) {qf2:.2f}   [{dt:.0f}s]")
    print("IS diagnostics:", json.dumps(out.diagnostics.get("reweighting", {}), default=float))

    print(f"\n{'param':22} {'published':>10} {'dynesty':>9} {'NPE before':>11} {'NPE after':>10}")
    rows = {}
    for p in PUB:
        b, a = float(res.median[p]), float(out.median[p])
        rows[p] = {"published": PUB[p], "dynesty": DYN[p], "npe_before": b, "npe_after": a}
        print(f"{p:22} {PUB[p]:10.1f} {DYN[p]:9.1f} {b:11.1f} {a:10.1f}")
    rows["_q"] = {"before": {"med": qm, "frac_gt95": qf},
                  "after": {"med": qm2, "frac_gt95": qf2}}
    json.dump(rows, open(HERE / "m10_importance_gw150914.json", "w"), indent=2, default=float)
    out.save(HERE / "gw150914_reweighted", plots=False)


if __name__ == "__main__":
    main()
