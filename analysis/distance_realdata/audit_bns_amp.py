"""Why is the BNS amplitude check 0.78 in O3 noise and 0.97 at GW170817? (2026-11-06)

The check projects the real-path signal a on the training-path signal b: a.b / a.a. It
falls if b is quieter than a, and also if the two are merely out of phase. So, per
detector and per noise stretch, for BNS and NSBH signals:
  norm ratio      |b| / |a|: loudness alone
  match           a.b / (|a||b|): shape alone; 1 = same waveform
  best lag        samples of shift that maximize the match
  by band         projection in frequency bands, to see where it goes wrong
The gate's number is norm ratio x match.
"""
import numpy as np

from ahsd.evaluation.path_audit import CONTEXT_BEFORE_S, NS_OFFSET_S, SR, detector_signal, training_path_signal
from ahsd.inference.preprocessing import fetch_gwosc, prepare_real

DETS = ("H1", "L1", "V1")
BANDS = ((15, 40), (40, 80), (80, 160), (160, 320), (320, 640), (640, 2048))
SOURCES = {"BNS": (1.46 * 1.009, 1.27 * 1.009), "NSBH": (6.0, 1.4)}
NOISE = (("GW170817 O2", 1187008882.4), ("GW190814 O3a", 1249852257.0), ("GW200115 O3b", 1263097407.7))

for name, gps in NOISE:
    g, raw, starts, srs, _ = fetch_gwosc(gps, strict=False)
    t = g + NS_OFFSET_S
    noise = prepare_real(raw, t, starts, srs, seed=0)
    dets = list(noise.detectors_present)
    for cls, (m1, m2) in SOURCES.items():
        rng = np.random.default_rng(3)
        res = {d: [] for d in dets}
        for k in range(6):
            p = dict(mass_1=m1, mass_2=m2, luminosity_distance=100.0, ra=float(rng.uniform(0, 2 * np.pi)),
                     dec=float(np.arcsin(rng.uniform(-1, 1))), theta_jn=float(np.arccos(rng.uniform(-1, 1))),
                     psi=float(rng.uniform(0, np.pi)), phase=float(rng.uniform(0, 2 * np.pi)))
            hs = {d: detector_signal(p, d, t, event_type=cls, before_s=CONTEXT_BEFORE_S) for d in dets}

            def inject(scale):
                r = {d: np.array(raw[d], dtype=np.float64) for d in raw}
                for d in dets:
                    i0 = int(round((t + 4.0 - starts[d]) * SR)) - len(hs[d])
                    r[d][i0:i0 + len(hs[d])] += hs[d] * scale
                return prepare_real(r, t, starts, srs, seed=0)

            first = inject(1.0)
            l2 = np.sqrt(sum(((first.strain[DETS.index(d)] - noise.strain[DETS.index(d)]) ** 2).sum() for d in dets))
            scale = 15.0 / float(l2)
            prep = inject(scale)
            for d in dets:
                i = DETS.index(d)
                a = (prep.strain[i] - noise.strain[i]).astype(np.float64)
                if np.linalg.norm(a) < 4:
                    continue
                b = training_path_signal((hs[d] * scale)[-6 * SR:-2 * SR], d, noise.asds[d])
                A, B = np.fft.rfft(a), np.fft.rfft(b)
                f = np.fft.rfftfreq(len(a), 1 / SR)
                cc = np.fft.irfft(A * np.conj(B), n=len(a)) / (np.linalg.norm(a) * np.linalg.norm(b))
                lag = int(np.argmax(cc))
                lag = lag - len(a) if lag > len(a) // 2 else lag
                band = [float((A[m] * np.conj(B[m])).real.sum() / max((np.abs(A[m]) ** 2).sum(), 1e-30))
                        for m in (((f >= lo) & (f < hi)) for lo, hi in BANDS)]
                frac = [float((np.abs(A[m]) ** 2).sum() / (np.abs(A) ** 2).sum()) for m in (((f >= lo) & (f < hi)) for lo, hi in BANDS)]
                res[d].append(dict(proj=float((a * b).sum() / (a * a).sum()), norm=float(np.linalg.norm(b) / np.linalg.norm(a)),
                                   match=float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))), lag=lag,
                                   best=float(cc.max()), band=band, frac=frac))
        print(f"== {name} {cls}; bands {BANDS}", flush=True)
        for d in dets:
            v = res[d]
            if not v:
                continue
            f = lambda k: float(np.median([x[k] for x in v]))
            print(f"  {d} n={len(v)} projection {f('proj'):.3f} = norm {f('norm'):.3f} x match {f('match'):.3f}; best lag {f('lag'):+.0f} samples, match there {f('best'):.3f}", flush=True)
            print(f"     band projection {np.median([x['band'] for x in v], axis=0).round(2).tolist()}", flush=True)
            print(f"     share of a's power {np.median([x['frac'] for x in v], axis=0).round(3).tolist()}", flush=True)
