"""STEP 3 - Real-event trajectory across available checkpoints.

LIMITATION (documented): the trainer saves ONLY best_model.pth (best val NLL);
no per-epoch checkpoints were ever written, and the model/ dir is being
overwritten by an active run. So a clean single-run epoch-15..latest trajectory
does not exist on disk. What DOES exist: stored CI reports (per-event
result.json with medians, railing, OOD) across model generations. We extract
those, focusing on the coherent-encoder + fixed-dataset lineage that the
current `long_run` belongs to, and order them by (lineage, epoch).
"""
import json, glob, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _events import EVENTS, PROBLEM, chirp_mass  # noqa: E402

EVENT_NAMES = list(EVENTS.keys())

# ordered generations to include (label -> ci dir). Coherent lineage last.
GENERATIONS = [
    ("v6", "lean_v6"),
    ("v7_aux", "lean_v7_aux"),
    ("v8_psd", "lean_v8_psdcond"),
    ("v8_psd_ft", "lean_v8_psdcond_ft"),
    ("coherentC(ep40)", "lean_coherentC"),
    ("long_run(ep18)", "long_run_ep18"),
]


def parse_event(path):
    r = json.load(open(path))
    params = {p["param"]: p for p in r["summary"]["parameters"]}
    d = r["summary"].get("diagnostics") or r["diagnostics"]
    m1 = params["mass_1"]["median"]; m2 = params["mass_2"]["median"]
    dL = params["luminosity_distance"]["median"]
    ood = d.get("context_ood") or {}
    return {
        "m1": m1, "m2": m2, "Mc": chirp_mass(m1, m2),
        "q": m2 / m1 if m1 else float("nan"),
        "dL": dL,
        "rail": d.get("boundary_railing_frac", float("nan")),
        "ood_pct": ood.get("percentile_vs_validation", float("nan")),
        "ood_maha": ood.get("mahalanobis", float("nan")),
        "conf": (d.get("verdict") or {}).get("confidence", "?"),
    }


def main():
    traj = {}  # gen_label -> {event -> row}
    for label, cid in GENERATIONS:
        traj[label] = {}
        for ev in EVENT_NAMES:
            p = f"analysis/ci/{cid}/{ev.lower()}/result.json"
            if os.path.exists(p):
                traj[label][ev] = parse_event(p)
    json.dump(traj, open(HERE / "step3_trajectory.json", "w"), indent=2)

    # ---- console: per-event error vs published, across generations ----
    print("=== STEP 3: REAL-EVENT TRAJECTORY (generation proxy) ===\n")
    print("Absolute m1 error (|median - published|), by generation:\n")
    hdr = f"{'event':10s} " + " ".join(f"{g:>15s}" for g, _ in GENERATIONS)
    print(hdr)
    for ev in EVENT_NAMES:
        pub = EVENTS[ev]["m1"]
        cells = []
        for g, _ in GENERATIONS:
            r = traj[g].get(ev)
            cells.append(f"{abs(r['m1']-pub):15.1f}" if r else f"{'--':>15s}")
        flag = " <==" if ev in PROBLEM else ""
        print(f"{ev:10s} " + " ".join(cells) + flag)

    for metric, lbl, fmt in [("rail", "railing fraction", "{:15.2f}"),
                             ("ood_pct", "OOD percentile", "{:15.1f}"),
                             ("Mc", "chirp-mass median", "{:15.1f}")]:
        print(f"\n{lbl}, by generation:\n")
        print(hdr)
        for ev in EVENT_NAMES:
            cells = []
            for g, _ in GENERATIONS:
                r = traj[g].get(ev)
                cells.append(fmt.format(r[metric]) if r else f"{'--':>15s}")
            flag = " <==" if ev in PROBLEM else ""
            print(f"{ev:10s} " + " ".join(cells) + flag)

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(len(GENERATIONS)))
    xlabels = [g for g, _ in GENERATIONS]
    metrics = [("m1_err", "|m1 - pub|  (Msun)"),
               ("rail", "railing fraction"),
               ("ood_pct", "OOD percentile"),
               ("dL", "dL median (Mpc)")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for ev in EVENT_NAMES:
            ys = []
            for g, _ in GENERATIONS:
                r = traj[g].get(ev)
                if not r:
                    ys.append(np.nan); continue
                if metric == "m1_err":
                    ys.append(abs(r["m1"] - EVENTS[ev]["m1"]))
                else:
                    ys.append(r[metric])
            c = "#d1495b" if ev in PROBLEM else None
            lw = 2.4 if ev in PROBLEM else 1.2
            ax.plot(xs, ys, marker="o", label=ev, color=c, lw=lw)
        if metric == "dL":
            for ev in EVENT_NAMES:
                ax.axhline(EVENTS[ev]["dL"], color="gray", ls=":", lw=0.4)
        ax.set_xticks(xs); ax.set_xticklabels(xlabels, rotation=30, ha="right",
                                              fontsize=8)
        ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Real-event metrics across model generations "
                 "(red = worsening events)", y=0.995)
    fig.tight_layout()
    fig.savefig(HERE / "figs/step3_trajectory.png", dpi=120)
    print("\nsaved figs/step3_trajectory.png")


if __name__ == "__main__":
    main()
