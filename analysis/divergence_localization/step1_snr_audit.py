"""STEP 1 - Training SNR distribution audit.

Network optimal matched-filter SNR of a WHITENED signal is exactly its L2 norm
(remix_data.py: net_snr = sqrt((sig_sum[kept]**2).sum())). We verified this
convention reproduces the stored measured `network_snr` distribution to <5%.
So we read the signal-only whitened components from the memmap and take the
per-signal 3-detector L2 norm as the network SNR.

Outputs: percentiles, threshold counts, histogram+CDF PNG, per-event local
density, JSON summary.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _events import EVENTS, PROBLEM  # noqa: E402

MEMMAP = HERE.parents[1] / "data/dataset/memmap/train"
OUT = HERE
FIGS = HERE / "figs"

# remix distance-rescale: amplitude x s, s ~ U(0.75, 1.33) => SNR x s
DIST_SCALE = (0.75, 1.33)


def per_signal_snr(sig_path, chunk=2000):
    s = np.load(sig_path, mmap_mode="r")
    n = s.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunk):
        blk = np.asarray(s[i:i + chunk], dtype=np.float32)
        out[i:i + chunk] = np.sqrt((blk.astype(np.float64) ** 2).sum(axis=(1, 2)))
    return out


def loudest_per_event(snr, events):
    """events: list of [sig_start, n_sig]; event owns snr[start:start+n_sig].
    Return the max (rank-0 loudest) SNR per event."""
    out = []
    for start, nsig in events:
        if nsig <= 0:
            continue
        out.append(snr[start:start + nsig].max())
    return np.array(out)


def main():
    snr = per_signal_snr(MEMMAP / "signals.npy")
    ev = json.load(open(MEMMAP / "events.json"))
    loud = loudest_per_event(snr, ev["events"])

    # remix-widened effective per-signal SNR (MC over the uniform scale)
    rng = np.random.default_rng(0)
    scale = rng.uniform(*DIST_SCALE, size=snr.shape)
    snr_eff = snr * scale

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    thresholds = [10, 15, 18, 20, 22, 25, 30, 40, 50]

    def describe(a):
        return {
            "n": int(a.size),
            "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "median": float(np.median(a)),
            "percentiles": {str(p): float(np.percentile(a, p)) for p in pcts},
            "n_above": {str(t): int((a > t).sum()) for t in thresholds},
            "frac_above": {str(t): float((a > t).mean()) for t in thresholds},
        }

    summary = {
        "convention": "network SNR = L2 norm of 3-detector whitened signal (remix_data.py)",
        "per_signal": describe(snr),
        "per_signal_remix_widened": describe(snr_eff),
        "loudest_per_event": describe(loud),
    }

    # per-event local density: percentile of each event's SNR + density in a
    # +/-2 SNR window around it (per-signal distribution and rank-0 distribution)
    ev_rows = {}
    for name, meta in EVENTS.items():
        s0 = meta["snr"]
        pct_all = float((snr < s0).mean() * 100)
        pct_loud = float((loud < s0).mean() * 100)
        win = (snr >= s0 - 2) & (snr <= s0 + 2)
        win_loud = (loud >= s0 - 2) & (loud <= s0 + 2)
        ev_rows[name] = {
            "snr": s0,
            "percentile_all_signals": round(pct_all, 1),
            "percentile_loudest_per_event": round(pct_loud, 1),
            "n_signals_within_2snr": int(win.sum()),
            "n_loudest_within_2snr": int(win_loud.sum()),
            "n_signals_louder": int((snr > s0).sum()),
        }
    summary["events"] = ev_rows

    json.dump(summary, open(OUT / "step1_snr.json", "w"), indent=2)

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    bins = np.linspace(0, 100, 101)
    ax1.hist(snr, bins=bins, color="#3b6fd4", alpha=0.55, label="per-signal (all)")
    ax1.hist(loud, bins=bins, color="#e08a1e", alpha=0.55,
             label="loudest per event (rank-0)")
    ax1.set_xlabel("network SNR"); ax1.set_ylabel("count")
    ax1.set_title("Training SNR histogram (0-100 clip)")
    ax1.set_yscale("log")
    for name, meta in EVENTS.items():
        c = "#d1495b" if name in PROBLEM else "#2b2b2b"
        ax1.axvline(meta["snr"], color=c, ls="--", lw=1.4)
        ax1.text(meta["snr"], ax1.get_ylim()[1] * 0.6, name.replace("GW", ""),
                 rotation=90, fontsize=7, color=c, ha="right", va="top")
    ax1.legend(fontsize=8)

    xs = np.sort(snr); cdf = np.arange(1, xs.size + 1) / xs.size
    ax2.plot(xs, cdf, color="#3b6fd4", label="per-signal CDF")
    xl = np.sort(loud); cdfl = np.arange(1, xl.size + 1) / xl.size
    ax2.plot(xl, cdfl, color="#e08a1e", label="loudest-per-event CDF")
    ax2.set_xlim(0, 100); ax2.set_xlabel("network SNR"); ax2.set_ylabel("CDF")
    ax2.set_title("Cumulative distribution")
    for name, meta in EVENTS.items():
        c = "#d1495b" if name in PROBLEM else "#2b2b2b"
        ax2.axvline(meta["snr"], color=c, ls="--", lw=1.0)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "step1_snr.png", dpi=130)

    # ---- console tables ----
    print("=== STEP 1: TRAINING SNR DISTRIBUTION ===\n")
    for label, key in [("PER-SIGNAL (all 60306)", "per_signal"),
                       ("LOUDEST-PER-EVENT (rank-0)", "loudest_per_event"),
                       ("PER-SIGNAL remix-widened", "per_signal_remix_widened")]:
        d = summary[key]
        print(f"[{label}]  n={d['n']}  median={d['median']:.1f}  "
              f"mean={d['mean']:.1f}  max={d['max']:.0f}")
        print("  pct:", " ".join(f"{p}={d['percentiles'][str(p)]:.1f}" for p in pcts))
        print("  n>thr:", " ".join(f">{t}:{d['n_above'][str(t)]}({d['frac_above'][str(t)]*100:.1f}%)"
                                    for t in thresholds))
        print()
    print(f"{'event':10s} {'SNR':>5} {'pctile(all)':>11} {'pctile(loud)':>12} "
          f"{'n±2(all)':>9} {'n±2(loud)':>10} {'n louder':>9}")
    for name, r in ev_rows.items():
        flag = " <== PROBLEM" if name in PROBLEM else ""
        print(f"{name:10s} {r['snr']:5.1f} {r['percentile_all_signals']:11.1f} "
              f"{r['percentile_loudest_per_event']:12.1f} "
              f"{r['n_signals_within_2snr']:9d} {r['n_loudest_within_2snr']:10d} "
              f"{r['n_signals_louder']:9d}{flag}")


if __name__ == "__main__":
    main()
