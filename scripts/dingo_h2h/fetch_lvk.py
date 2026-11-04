#!/usr/bin/env python3
"""Official LVK products for the head-to-head events, from the GWTC-2.1 and
GWTC-3 parameter-estimation data releases (Zenodo 6513631, 5546663, 8177023):

  - the PSDs the LVK analyses used, per detector, written as two-column text
    files for DINGO's psd-dict (DINGO's own 128 s Welch estimate gave NaNs in
    band for 5 events and importance-sampling efficiencies of 0-5%);
  - the LVK posterior samples (detector-frame chirp mass, mass ratio,
    luminosity distance), the judge for every method in the comparison.

Run on the LOGIN node (downloads; compute nodes have no internet). Each
release file is ~0.2-0.8 GB; only the extracted products are kept.

    python scripts/dingo_h2h/fetch_lvk.py [--keep]
"""
import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/dingo_h2h/lvk"
RECORDS = (6513631, 5546663, 8177023)


def release_files():
    files = {}
    for r in RECORDS:
        d = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{r}", timeout=60))
        for f in d["files"]:
            k = f["key"]
            if k.endswith(".h5") and "PEDataRelease" in k:
                ev = k.split("-")[-1].split("_PEDataRelease")[0]
                pref = 0 if k.endswith("_cosmo.h5") else 1          # smaller file first
                if ev not in files or pref < files[ev][0]:
                    files[ev] = (pref, f["links"]["self"], k)
    return files


def extract(h5path, out_dir):
    import h5py
    with h5py.File(h5path, "r") as h:
        runs = [k for k in h.keys() if k.startswith("C01:")]
        run = next((r for r in runs if "IMRPhenomXPHM" in r), runs[0] if runs else None)
        if run is None:
            raise ValueError("no C01 analysis in file")
        g = h[run]
        for det, arr in g["psds"].items():
            a = np.asarray(arr)
            np.savetxt(out_dir / f"{det}_psd.txt", a)
        s = g["posterior_samples"]
        names = s.dtype.names
        mc = s["chirp_mass"] if "chirp_mass" in names else None
        q = s["mass_ratio"]
        dl = s["luminosity_distance"]
        np.savez(out_dir / "lvk_posterior.npz", chirp_mass=mc, mass_ratio=q, luminosity_distance=dl,
                 analysis=run)
        return run, sorted(g["psds"].keys()), len(q)


def intact(d: Path) -> bool:
    """Cached products load as numbers (a catalogue_validation bug once wrote
    event records over PSD files)."""
    if not (d / "lvk_posterior.npz").exists():
        return False
    try:
        for f in d.glob("*_psd.txt"):
            a = np.loadtxt(f)
            if a.ndim != 2 or a.shape[1] != 2:
                return False
    except ValueError:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the downloaded release files")
    a = ap.parse_args()
    events = [l.strip() for l in open(ROOT / "scripts/dingo_h2h/events.txt") if l.strip() and not l.startswith("#")]
    files = release_files()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_download"
    tmp.mkdir(exist_ok=True)
    summary = {}
    for ev in events:
        d = OUT / ev
        if intact(d):
            summary[ev] = "ok (cached)"
            continue
        # releases name some events in full (GW190412 -> GW190412_053044)
        match = ev if ev in files else next((k for k in files if k.startswith(ev + "_")), None)
        if match is None:
            summary[ev] = "no release file"
            print(ev, summary[ev], flush=True)
            continue
        _, url, key = files[match]
        f = tmp / key
        subprocess.run(["wget", "-q", "-c", "-O", str(f), url], check=True)
        d.mkdir(exist_ok=True)
        try:
            run, dets, n = extract(f, d)
            summary[ev] = f"ok: {run}, PSDs {dets}, {n} samples"
        except Exception as e:
            summary[ev] = f"extract failed: {type(e).__name__}: {str(e)[:80]}"
        if not a.keep:
            f.unlink(missing_ok=True)
        print(ev, summary[ev], flush=True)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
