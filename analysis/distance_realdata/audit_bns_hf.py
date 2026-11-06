"""Where does the power above 640 Hz come from? (2026-11-06)

audit_bns_amp.py: a BNS injection has 4% of its whitened power at 320-640 Hz and 56-70%
above 640 Hz, where an inspiral has almost none. If that power is an artifact, "network
SNR 15" overstates the physical signal, and the injections are really quieter.

Fine bands for: the raw waveform, the real-path signal, the training-path signal, and the
signals stored in the training data (design-whitened, as generated). Also the ASD near
Nyquist: GWOSC 4 kHz data is low-passed, and dividing an unfiltered injection by that
falling ASD would inflate it.
"""
import sys

import numpy as np

sys.path.insert(0, "experiments")
from ahsd.evaluation.path_audit import CONTEXT_BEFORE_S, NS_OFFSET_S, SR, detector_signal, training_path_signal
from ahsd.inference.preprocessing import fetch_gwosc, prepare_real
from remix_data import RemixDataset

DETS = ("H1", "L1", "V1")
EDGES = (20, 160, 320, 640, 1000, 1400, 1700, 1900, 2000, 2049)
T = 16384
f = np.fft.rfftfreq(T, 1 / SR)
MASK = [(f >= a) & (f < b) for a, b in zip(EDGES[:-1], EDGES[1:])]


def share(x):
    P = np.abs(np.fft.rfft(x)) ** 2
    return np.array([P[m].sum() for m in MASK]) / P.sum()


print("bands", list(zip(EDGES[:-1], EDGES[1:])))
for name, gps in (("GW170817 O2", 1187008882.4), ("GW190814 O3a", 1249852257.0)):
    g, raw, starts, srs, _ = fetch_gwosc(gps, strict=False)
    t = g + NS_OFFSET_S
    noise = prepare_real(raw, t, starts, srs, seed=0)
    for d in noise.detectors_present[:2]:
        asd = noise.asds[d]
        idx = [np.argmin(abs(f - x)) for x in (300, 640, 1000, 1400, 1700, 1900, 2000, 2040, 2048)]
        print(f"{name} {d} ASD at 300..2048 Hz relative to 300 Hz:", (asd[idx] / asd[idx[0]]).round(3).tolist(), flush=True)
        p = dict(mass_1=1.47, mass_2=1.28, luminosity_distance=100.0, ra=1.0, dec=0.3, theta_jn=0.5, psi=0.2, phase=0.1)
        hs = {x: detector_signal(p, x, t, event_type="BNS", before_s=CONTEXT_BEFORE_S) for x in noise.detectors_present}
        r = {x: np.array(raw[x], dtype=np.float64) for x in raw}
        for x in noise.detectors_present:
            i0 = int(round((t + 4.0 - starts[x]) * SR)) - len(hs[x])
            r[x][i0:i0 + len(hs[x])] += hs[x]
        prep = prepare_real(r, t, starts, srs, seed=0, drop_glitches=False)
        i = DETS.index(d)
        a = (prep.strain[i] - noise.strain[i]).astype(np.float64)
        h4 = hs[d][-6 * SR:-2 * SR]
        b = training_path_signal(h4, d, noise.asds[d])
        print(f"  raw h, 4 s                 {share(h4).round(4).tolist()}")
        print(f"  real path, whitened        {share(a).round(3).tolist()}  norm {np.linalg.norm(a):.1f}")
        print(f"  training path, recoloured  {share(b).round(3).tolist()}  norm {np.linalg.norm(b):.1f}", flush=True)

for cls, data in (("BNS", "data/dataset_bns_v4"), ("NSBH", "data/dataset_nsbh_v4"), ("BBH", "data/dataset_bbh_v6")):
    ds = RemixDataset(f"{data}/memmap/validation", remix=False, single_signal=True)
    rng = np.random.default_rng(0)
    rows = []
    for k in rng.choice(ds._index_map, 200, replace=False):
        start, _ = ds.events[int(k)]
        sig = np.asarray(ds.signals[start][:, :T], np.float64)
        d = int(np.argmax((sig ** 2).sum(-1)))
        if (sig[d] ** 2).sum() > 25:
            rows.append(share(sig[d]))
    print(f"{cls} stored training signals, loudest detector, n={len(rows)}: median {np.median(rows, 0).round(3).tolist()}", flush=True)
