"""LVK posteriors for BNS and NSBH events, for score_ns_events.py. Login node only.

GWTC-2.1 and GWTC-3 come from the Zenodo releases fetch_lvk.py reads; GW170817 from the
GWTC-1 release at the LIGO DCC. Keeps chirp mass, mass ratio and distance in the detector
frame, and the analysis used. Low-spin analyses are preferred for neutron stars.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_lvk import release_files

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/ns_lvk"
GWTC1 = "https://dcc.ligo.org/public/0157/P1800370/005/GW170817_GWTC-1.hdf5"
EVENTS = ("GW190425", "GW190814", "GW200105", "GW200115")


def pick(runs):
    """A low-spin analysis if there is one, else the first with a neutron-star model."""
    order = ("LowSpin", "lowSpin", "low_spin", "NSBH", "NRTidal", "XPHM", "C01")
    for key in order:
        hit = [r for r in runs if key in r]
        if hit:
            return hit[0]
    return runs[0]


def save(d, mc, q, dl, run, n=None):
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "lvk_posterior.npz", chirp_mass=mc, mass_ratio=q, luminosity_distance=dl, analysis=run)
    print(f"{d.name}: {run}, {len(q)} samples; Mc {np.median(mc):.4f} sd {mc.std():.4f}, q {np.median(q):.3f}, dL {np.median(dl):.1f}", flush=True)


def main():
    import h5py
    tmp = OUT / "_download"
    tmp.mkdir(parents=True, exist_ok=True)
    f = tmp / "GW170817_GWTC-1.hdf5"
    try:
        subprocess.run(["wget", "-q", "-c", "-O", str(f), GWTC1], check=True)
        with h5py.File(f, "r") as h:
            run = "IMRPhenomPv2NRT_lowSpin_posterior"
            s = h[run]
            m1, m2 = s["m1_detector_frame_Msun"], s["m2_detector_frame_Msun"]
            save(OUT / "GW170817", (m1 * m2) ** .6 / (m1 + m2) ** .2, m2 / m1, s["luminosity_distance_Mpc"], run)
    except Exception as e:
        print("GW170817 failed:", type(e).__name__, str(e)[:200], flush=True)
    f.unlink(missing_ok=True)
    files = release_files()
    for ev in EVENTS:
        match = ev if ev in files else next((k for k in files if k.startswith(ev + "_")), None)
        if match is None:
            print(ev, "no release file", flush=True)
            continue
        _, url, key = files[match]
        f = tmp / key
        try:
            subprocess.run(["wget", "-q", "-c", "-O", str(f), url], check=True)
            with h5py.File(f, "r") as h:
                runs = [k for k in h.keys() if k.startswith("C01:")] or [k for k in h.keys() if "version" not in k and "history" not in k]
                print(ev, "analyses:", runs, flush=True)
                run = pick(runs)
                s = h[run]["posterior_samples"]
                save(OUT / ev, np.asarray(s["chirp_mass"]), np.asarray(s["mass_ratio"]), np.asarray(s["luminosity_distance"]), run)
        except Exception as e:
            print(ev, "failed:", type(e).__name__, str(e)[:200], flush=True)
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
