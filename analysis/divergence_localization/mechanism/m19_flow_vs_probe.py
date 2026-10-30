"""M19 - Decompose the probe-vs-flow gap in q: lost information, or just bias?

m18 found the frozen context supports R^2(q) ~ 0.47 on heavy systems for EVERY
probe family (ridge == MLP == forest == boosting), while the flow's own
E[q|context] -- read from the same context -- scores 0.015 (heavy) / 0.241
(heavy & loud). R^2 charges for bias and for lost correlation at once, and those
mean different things:

  * if the flow merely SHRINKS q toward the population mean, the ranking of
    events survives; an affine rescale a + b*qflow would recover it. Then the
    information reached the flow's output and the defect is calibration.
  * if the flow's q barely correlates with truth, the information did not
    survive into the posterior at all.

r^2 (squared Pearson) is exactly R^2-after-optimal-affine-recalibration, so
comparing r^2 against raw R^2 separates the two. The regression slope b says
which direction: b << 1 is shrinkage toward the mean.

Reads m18's cached contexts; nothing is trained or modified in the NPE.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FIT, TEST = HERE / "m18_cache_train_12000.npz", HERE / "m18_cache_validation_4702.npz"


def _r2(yt, yp):
    """Coefficient of determination against the mean of yt."""
    return 1.0 - float(((yt - yp) ** 2).sum()) / float(((yt - yt.mean()) ** 2).sum())


def main():
    z = np.load(TEST)
    T, snr, qmean, qmed = z["T"], z["snr"], z["qmean"], z["qmed"]
    qt = T[:, 1] / T[:, 0]
    probe = json.load(open(HERE / "m18_nonlinear_probe.json"))

    groups = [("light  m1<20", T[:, 0] < 20),
              ("mid    20<=m1<29", (T[:, 0] >= 20) & (T[:, 0] < 29)),
              ("HEAVY  m1>=29", T[:, 0] >= 29),
              ("HEAVY & loud (SNR>18)", (T[:, 0] >= 29) & (snr > 18))]

    print("Flow's E[q|ctx] vs the probe ceiling on the same events.\n")
    print(f"{'regime':24}{'probe':>8}{'flow R2':>9}{'flow r':>8}{'flow r^2':>10}"
          f"{'slope':>8}{'true sd':>9}{'flow sd':>9}")
    out = {}
    for lab, m in groups:
        yt, yp = qt[m], qmean[m]
        r = float(np.corrcoef(yt, yp)[0, 1])
        b = float(np.polyfit(yp, yt, 1)[0])       # regress truth on prediction
        row = dict(n=int(m.sum()), probe_r2=probe[lab]["best_probe_q_r2"],
                   flow_r2=_r2(yt, yp), flow_r=r, flow_r2_affine=r * r,
                   slope=b, sd_true=float(yt.std()), sd_flow=float(yp.std()))
        out[lab] = row
        print(f"{lab:24}{row['probe_r2']:>8.3f}{row['flow_r2']:>9.3f}{r:>8.3f}"
              f"{r*r:>10.3f}{b:>8.2f}{row['sd_true']:>9.3f}{row['sd_flow']:>9.3f}")

    print("\n  flow r^2 ~= probe  -> info IS in the posterior; defect is affine "
          "miscalibration/shrinkage")
    print("  flow r^2 << probe  -> info decodable from the context did not reach "
          "the posterior")
    print("  slope >> 1 with small flow sd -> flow compresses q toward the "
          "population mean")
    json.dump(out, open(HERE / "m19_flow_vs_probe.json", "w"), indent=2, default=float)
    print(f"\nwrote {HERE / 'm19_flow_vs_probe.json'}")


if __name__ == "__main__":
    main()
