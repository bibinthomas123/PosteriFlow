"""Dynesty reference posterior for the frozen basin-probe injections.

The basin gate (basin_probe.basin_gate: q>0.95 mass < 0.30, rail < 0.20) was
set by eyeballing collapsed-vs-healthy NPE runs, not by physics. The probe's
quiet row has truth q = 0.86 at SNR ~24 with 2 detectors — the OPTIMAL
posterior necessarily keeps some mass at q > 0.95, and nobody has measured
how much. This script nested-samples the exact probe injections (same
signal, same noise seed, same priors as training) so the gate can be
calibrated against the achievable optimum instead of a guess:

  - if dynesty's q>0.95 mass is ~>= 0.30, the gate demands better-than-Bayes
    and must be recalibrated to the dynesty value (+ margin);
  - if it is well below 0.30, the residual NPE deficit is real headroom.

Usage:  conda run -n ahsd python analysis/attractor_basin/dynesty_reference.py [row ...]
        (rows default to "basin snr24"; e.g. add "basin edgeon" "basin q.65")
Writes results under analysis/attractor_basin/dynesty_ref/.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ahsd.evaluation.basin_probe import ROWS, GPS_REF, DETS, _projected, _design_noise
from ahsd.data.bilby_pipeline import get_default_psd
from ahsd.inference.dynesty_bridge import run_dynesty

OUT = Path(__file__).resolve().parent / "dynesty_ref"


def main():
    want = sys.argv[1:] or ["basin snr24"]
    rows = [r for r in ROWS if r[0] in want]
    assert rows, f"no probe rows match {want}; have {[r[0] for r in ROWS]}"
    for label, p, sign, seed in rows:
        sig = _projected(p, sign)
        raw = {det: _design_noise(det, seed * 10 + i) + sig[det]
               for i, det in enumerate(DETS)}
        psds = {det: get_default_psd(det) for det in DETS}
        tag = label.replace(" ", "_").replace(".", "")
        result = run_dynesty(raw, psds, OUT, tag, center_gps=GPS_REF,
                             trigger_offset=0.0, nlive=300, npool=4)
        post = result.posterior
        m1 = np.maximum(post["mass_1"].values, post["mass_2"].values)
        m2 = np.minimum(post["mass_1"].values, post["mass_2"].values)
        q = m2 / m1
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        mct = (p["mass_1"] * p["mass_2"]) ** 0.6 / (p["mass_1"] + p["mass_2"]) ** 0.2
        line = (f"[dynesty ref] {label}: q_med {np.median(q):.3f}  "
                f"q>0.95 {float((q > 0.95).mean()):.3f}  "
                f"q90CI [{np.percentile(q, 5):.2f},{np.percentile(q, 95):.2f}]  "
                f"Mc_err {float(np.median(mc)) / mct - 1.0:+.3f}  "
                f"(truth q {p['mass_2'] / p['mass_1']:.2f})")
        print(line, flush=True)
        with open(OUT / "summary.txt", "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
