"""STEP 2 - Heavy-BBH coverage audit.

Joint distribution of chirp mass, total mass, mass ratio, network SNR, and
luminosity distance over the full training set (memmap params + cached SNR).
Overlays the 6 GWTC events and quantifies how close each sits to the edge of
the training distribution (marginal percentiles + local joint density).
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _events import EVENTS, PROBLEM, event_table  # noqa: E402

MEMMAP = HERE.parents[1] / "data/dataset/memmap/train"
FIGS = HERE / "figs"

# param column order (verified against batch dicts / ParamScaler.PARAM_NAMES)
COL = {"mass_1": 0, "mass_2": 1, "luminosity_distance": 2}


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def main():
    p = np.load(MEMMAP / "params.npy")
    snr = np.load(HERE / "_snr_cache.npy")
    m1, m2 = p[:, 0], p[:, 1]
    dL = p[:, 2]
    Mc = chirp_mass(m1, m2)
    Mtot = m1 + m2
    q = m2 / m1

    data = {"chirp_mass": Mc, "total_mass": Mtot, "mass_ratio": q,
            "network_snr": snr, "luminosity_distance": dL}
    evt = event_table()

    # ---- per-event edge diagnostics ----
    def pctile(a, v):
        return float((a < v).mean() * 100)

    rows = {}
    for name, e in evt.items():
        ev_vals = {"chirp_mass": e["Mc"], "total_mass": e["Mtot"],
                   "mass_ratio": e["q"], "network_snr": e["snr"],
                   "luminosity_distance": e["dL"]}
        marg = {k: round(pctile(data[k], ev_vals[k]), 1) for k in data}
        # local joint density in (Mc, SNR): count training within a relative box
        box_mc = (Mc > e["Mc"] * 0.85) & (Mc < e["Mc"] * 1.15)
        box_snr = (snr > e["snr"] * 0.8) & (snr < e["snr"] * 1.2)
        n_joint = int((box_mc & box_snr).sum())
        # (Mtot, q) box
        box_mt = (Mtot > e["Mtot"] * 0.85) & (Mtot < e["Mtot"] * 1.15)
        box_q = (q > e["q"] - 0.1) & (q < e["q"] + 0.1)
        n_mtq = int((box_mt & box_q).sum())
        rows[name] = {"marginal_pctile": marg,
                      "n_train_in_McSNR_box(±15%Mc,±20%SNR)": n_joint,
                      "n_train_in_Mtot_q_box(±15%Mtot,±0.1q)": n_mtq}

    summary = {
        "n_train": int(p.shape[0]),
        "training_ranges": {k: [round(float(v.min()), 2), round(float(v.max()), 2),
                                round(float(np.percentile(v, 99)), 2)]
                            for k, v in data.items()},
        "note_ranges": "[min, max, p99]",
        "events": rows,
    }
    json.dump(summary, open(HERE / "step2_mass.json", "w"), indent=2)

    # ---- pair plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["chirp_mass", "total_mass", "mass_ratio", "network_snr",
            "luminosity_distance"]
    labels = ["Mc", "Mtot", "q", "SNR", "dL"]
    nsub = np.random.default_rng(0).choice(p.shape[0], 12000, replace=False)
    K = len(keys)
    fig, axes = plt.subplots(K, K, figsize=(16, 16))
    clip = {"network_snr": (0, 120), "luminosity_distance": (0, 2300),
            "chirp_mass": (0, 60), "total_mass": (0, 130), "mass_ratio": (0, 1)}
    for i in range(K):
        for j in range(K):
            ax = axes[i, j]
            ki, kj = keys[i], keys[j]
            if i == j:
                ax.hist(data[ki], bins=60, color="#3b6fd4", alpha=0.6)
                for name, e in evt.items():
                    c = "#d1495b" if name in PROBLEM else "#111"
                    ax.axvline({"chirp_mass": e["Mc"], "total_mass": e["Mtot"],
                                "mass_ratio": e["q"], "network_snr": e["snr"],
                                "luminosity_distance": e["dL"]}[ki],
                               color=c, ls="--", lw=1.0)
                ax.set_yticks([])
            else:
                ax.scatter(data[kj][nsub], data[ki][nsub], s=2, alpha=0.05,
                           color="#3b6fd4", rasterized=True)
                for name, e in evt.items():
                    c = "#d1495b" if name in PROBLEM else "#111"
                    xv = {"chirp_mass": e["Mc"], "total_mass": e["Mtot"],
                          "mass_ratio": e["q"], "network_snr": e["snr"],
                          "luminosity_distance": e["dL"]}
                    ax.scatter(xv[kj], xv[ki], s=45, marker="*", color=c,
                               edgecolor="white", linewidth=0.5, zorder=5)
                if kj in clip:
                    ax.set_xlim(*clip[kj])
                if ki in clip:
                    ax.set_ylim(*clip[ki])
            if i == K - 1:
                ax.set_xlabel(labels[j])
            if j == 0:
                ax.set_ylabel(labels[i])
    fig.suptitle("Training joint distribution (blue) with GWTC events "
                 "(red=worsening, black=other)", y=0.99)
    fig.tight_layout()
    fig.savefig(FIGS / "step2_pairplot.png", dpi=110)
    plt.close(fig)

    # ---- focused Mc-vs-SNR density (the key joint) ----
    fig, ax = plt.subplots(figsize=(9, 7))
    h = ax.hist2d(Mc, snr, bins=[np.linspace(0, 55, 80), np.linspace(0, 120, 80)],
                  cmap="viridis", cmin=1)
    fig.colorbar(h[3], ax=ax, label="training count")
    for name, e in evt.items():
        c = "#ff5566" if name in PROBLEM else "#ffffff"
        ax.scatter(e["Mc"], e["snr"], s=120, marker="*", color=c,
                   edgecolor="black", linewidth=0.8, zorder=5)
        ax.annotate(name.replace("GW", ""), (e["Mc"], e["snr"]),
                    fontsize=8, color=c, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("chirp mass (detector-frame)"); ax.set_ylabel("network SNR")
    ax.set_title("Training density in (chirp mass, SNR); stars = GWTC events")
    fig.tight_layout()
    fig.savefig(FIGS / "step2_mc_snr_density.png", dpi=130)
    plt.close(fig)

    # ---- console ----
    print("=== STEP 2: HEAVY-BBH COVERAGE ===\n")
    print("Training ranges [min, max, p99]:")
    for k, v in summary["training_ranges"].items():
        print(f"  {k:22s} {v}")
    print()
    hdr = f"{'event':10s} {'Mc':>6} {'Mtot':>6} {'q':>5} {'SNR':>5} {'dL':>6} | " \
          f"{'%Mc':>5} {'%Mtot':>6} {'%q':>5} {'%SNR':>5} {'%dL':>5} | " \
          f"{'nMcSNR':>6} {'nMtq':>5}"
    print(hdr)
    for name, e in evt.items():
        r = rows[name]; mp = r["marginal_pctile"]
        flag = " <==" if name in PROBLEM else ""
        print(f"{name:10s} {e['Mc']:6.1f} {e['Mtot']:6.1f} {e['q']:5.2f} "
              f"{e['snr']:5.1f} {e['dL']:6.0f} | "
              f"{mp['chirp_mass']:5.1f} {mp['total_mass']:6.1f} "
              f"{mp['mass_ratio']:5.1f} {mp['network_snr']:5.1f} "
              f"{mp['luminosity_distance']:5.1f} | "
              f"{r['n_train_in_McSNR_box(±15%Mc,±20%SNR)']:6d} "
              f"{r['n_train_in_Mtot_q_box(±15%Mtot,±0.1q)']:5d}{flag}")
    print("\n(marginal columns = percentile of the event value within the "
          "training marginal; nMcSNR/nMtq = local joint counts)")


if __name__ == "__main__":
    main()
