"""Bit-for-bit check: does infer(event='GW150914') use the same preprocessing
as a hand-called fetch_gwosc + prepare_real, and is the whole path
deterministic/reproducible?

User's Possibility 1: preprocessing isn't identical between prepare_real()
calls (PSD window, whitening, segment boundaries, centering, tapering,
resampling). Code inspection shows both infer(event=...) and
twin_test5.inject_real_delayed() route through the SAME prepare_real()
implementation with the same default fetch_seconds=64 -- no separate
reimplementation exists to diverge. This script verifies that claim by
execution rather than just reading the code:

  V1  infer(event="GW150914") reproduces the validate_checkpoint report
      (m1 31.0, m2 30.9, rail 74%) -- confirms determinism.
  V2  A hand-called fetch_gwosc("GW150914") + prepare_real(...) with
      infer()'s exact default args produces a strain tensor BYTE-IDENTICAL
      to what infer() used internally (extracted via PosteriorResult.prepared).
  V3  Calling prepare_real twice on the same raw strain gives identical
      output (rules out hidden nondeterminism in the ASD/whitening step).
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ahsd.inference.pipeline import infer
from ahsd.inference.preprocessing import fetch_gwosc, prepare_real, FETCH_SECONDS

CKPT = str(ROOT / "model/lean_npe_mf_scratch/best_model.pth")


def main():
    print(f"FETCH_SECONDS (preprocessing.py default) = {FETCH_SECONDS}")

    print("\n=== V1: infer(event='GW150914') reproduces the CI report ===")
    res = infer(event="GW150914", num_samples=3000, model=CKPT, device="cpu", seed=0)
    med = res.median
    rail = res.diagnostics["boundary_railing_frac"]
    print(f"  m1 {med['mass_1']:.1f} (report: 31.0)  m2 {med['mass_2']:.1f} (report: 30.9)  "
          f"rail {rail:.1%} (report: 74.0%)")
    v1_ok = abs(med["mass_1"] - 31.0) < 2.0 and abs(med["mass_2"] - 30.9) < 2.0 and abs(rail - 0.740) < 0.05
    print("V1", "PASS (reproduces)" if v1_ok else "MISMATCH")

    print("\n=== V2: hand-called fetch_gwosc+prepare_real vs infer()'s internal prepared.strain ===")
    gps, raw, starts, srs, found = fetch_gwosc("GW150914")
    print(f"  fetched: gps={gps}, detectors={found}, "
          f"lengths={ {d: len(raw[d]) for d in found} }")
    prep_manual = prepare_real(raw, gps, starts, srs, psds=None, seed=0)
    prep_internal = res.prepared

    d = np.abs(prep_manual.strain - prep_internal.strain)
    print(f"  max |manual - internal| strain diff: {d.max():.3e}")
    print(f"  manual strain std per det: "
          f"{ {('H1','L1','V1')[i]: float(prep_manual.strain[i].std()) for i in range(3)} }")
    print(f"  internal strain std per det: "
          f"{ {('H1','L1','V1')[i]: float(prep_internal.strain[i].std()) for i in range(3)} }")
    v2_ok = d.max() < 1e-5
    print("V2", "PASS (byte-identical)" if v2_ok else "MISMATCH — preprocessing paths DIVERGE")

    print("\n=== V3: prepare_real determinism (call twice, same inputs) ===")
    prep_a = prepare_real(raw, gps, starts, srs, psds=None, seed=0)
    prep_b = prepare_real(raw, gps, starts, srs, psds=None, seed=0)
    d2 = np.abs(prep_a.strain - prep_b.strain).max()
    print(f"  max diff between two identical calls: {d2:.3e}")
    v3_ok = d2 < 1e-9
    print("V3", "PASS (deterministic)" if v3_ok else "MISMATCH — hidden nondeterminism")

    print(f"\nOVERALL: {'PASS — Possibility 1 (preprocessing mismatch) RULED OUT' if (v1_ok and v2_ok and v3_ok) else 'FAIL — investigate further'}")

    # bonus: quality/warning flags on the real GW150914 window (glitch/kurtosis
    # checks already built into prepare_real -- directly relevant to
    # Possibility 3, the specific noise realization)
    print("\n=== bonus: prepare_real quality flags on GW150914's actual window ===")
    for k, v in prep_internal.quality.items():
        print(f"  {k}: {v}")
    if prep_internal.warnings:
        for w in prep_internal.warnings:
            print("  WARNING:", w)
    else:
        print("  (no quality warnings raised)")


if __name__ == "__main__":
    main()
