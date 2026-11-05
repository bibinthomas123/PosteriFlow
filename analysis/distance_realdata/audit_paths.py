"""Audit of training vs real-event mismatches (2026-11-05), run on v6 and the fixed code.

  shape    the same signal through both paths: per-band amplitude, time lag, overlap
  curves   design ASD file vs the PSD that whitened the signals; bank vs real ASD recipe
  prior    TrainingPrior.sample vs the labels training actually sees
  params   injections through both paths: per-parameter error and coverage against truth
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

from diagnose import signal_8s

SR = 4096
F4 = np.fft.rfftfreq(16384, 1 / SR)
BANDS = ((20, 35), (35, 50), (50, 80), (80, 120), (120, 200), (200, 400), (400, 800), (800, 1600))
DETS = ("H1", "L1", "V1")
EVENTS = ("GW190412", "GW190408_181802", "GW200129_065458", "GW190828_063405", "GW190503_185404", "GW191222_033537")


def gps_table():
    return {e["event_name"]: e["gps_time"] for e in json.load(open("data/gwtc/gwtc_events.json"))}


def training_signal(h4, d, asd, gain):
    from ahsd.data.bilby_pipeline import BilbyPreprocessor, get_default_psd
    from ahsd.inference.preprocessing import _design_asd
    sig = BilbyPreprocessor(SR, 4.0).preprocess(h4, get_default_psd(d), detector=d).astype(np.float64)
    design = _design_asd(d, "data/noise_bank")
    n = min(len(design), len(asd))
    filt = np.ones(8193)
    filt[:n] = np.clip(design[:n] / np.maximum(asd[:n], 1e-30), 1 / 50, 50)
    return np.fft.irfft(np.fft.rfft(sig) * filt * gain, n=16384)


def shape():
    from ahsd.data.bilby_pipeline import highpass_gain
    from ahsd.inference import preprocessing as P
    G = highpass_gain(F4, SR)
    gps_of, rng = gps_table(), np.random.default_rng(3)
    amp, lag, ov = {b: [] for b in BANDS}, [], []
    for ev in EVENTS:
        gps, raw, st, srs, _ = P.fetch_gwosc(gps_of[ev], strict=False)
        t = gps - 20
        p0 = P.prepare_real(raw, t, st, srs, seed=0)
        for m1, m2 in ((60., 45.), (40., 30.), (15., 10.)):
            par = dict(mass_1=m1, mass_2=m2, luminosity_distance=300., ra=float(rng.uniform(0, 6.28)),
                       dec=float(np.arcsin(rng.uniform(-1, 1))), theta_jn=float(np.arccos(rng.uniform(-1, 1))),
                       psi=1., phase=1.)
            hs = {d: signal_8s(par, d, t, "IMRPhenomXP") for d in p0.detectors_present}
            r = {d: np.array(raw[d], float) for d in raw}
            for d, h in hs.items():
                i0 = int(round((t - st[d]) * SR)) - 16384
                r[d][i0:i0 + len(h)] += h
            p1 = P.prepare_real(r, t, st, srs, seed=0)
            for d, h in hs.items():
                i = DETS.index(d)
                a = (p1.strain[i] - p0.strain[i]).astype(np.float64)
                if np.sqrt((a ** 2).sum()) < 8:
                    continue
                b = training_signal(h[8192:8192 + 16384], d, p0.asds[d], G)
                A, B = np.fft.rfft(a), np.fft.rfft(b)
                for lo, hi in BANDS:
                    m = (F4 >= lo) & (F4 < hi)
                    if (np.abs(A[m]) ** 2).sum() > 0.02 * (np.abs(A) ** 2).sum():
                        amp[(lo, hi)].append(np.sqrt((np.abs(B[m]) ** 2).sum() / (np.abs(A[m]) ** 2).sum()))
                cc = np.fft.irfft(A * np.conj(B), n=16384)
                k = int(np.argmax(cc))
                lag.append(((k + 8192) % 16384 - 8192) / SR * 1e3)
                ov.append(float(cc.max() / np.sqrt((a ** 2).sum() * (b ** 2).sum()) / 1.0 * 1.0))
    print("SHAPE training/inference amplitude by band (median [5,95%], n), corrected filter")
    for b, v in amp.items():
        if v:
            print(f"  {b}: {np.median(v):.3f} [{np.quantile(v, .05):.3f}, {np.quantile(v, .95):.3f}] n={len(v)}")
    print(f"  time lag ms: median {np.median(lag):+.3f} range [{min(lag):+.3f}, {max(lag):+.3f}] n={len(lag)}")
    print(f"  overlap: median {np.median(ov):.4f}, 5% {np.quantile(ov, .05):.4f}")


def curves():
    from ahsd.data.bilby_pipeline import get_default_psd
    from ahsd.inference import preprocessing as P
    from gwpy.timeseries import TimeSeries
    print("CURVES design file ASD / analytic ASD that whitened the signals, by band")
    for d in DETS:
        p = get_default_psd(d)
        tr = np.sqrt(np.interp(F4, p["frequencies"], p["psd"]))
        des = P._design_asd(d, "data/noise_bank")
        n = min(len(des), len(tr))
        r = des[:n] / np.maximum(tr[:n], 1e-40)
        print(f"  {d} len {len(des)}:", [round(float(np.median(r[(F4[:n] >= a) & (F4[:n] < b)])), 3) for a, b in BANDS])
    gps_of = gps_table()
    gps, raw, st, srs, _ = P.fetch_gwosc(gps_of["GW190412"], strict=False)
    x = np.array(raw["H1"], float)
    real = TimeSeries(x, sample_rate=SR).highpass(15).asd(fftlength=8, overlap=4, method="median")
    bank = TimeSeries(x[:64 * SR], sample_rate=SR).highpass(15).asd(fftlength=8, overlap=4)
    r = np.interp(F4, bank.frequencies.value, bank.value) / np.interp(F4, real.frequencies.value, real.value)
    print("  bank recipe (64 s) / real recipe (80 s) ASD, same data:",
          [round(float(np.median(r[(F4 >= a) & (F4 < b)])), 3) for a, b in BANDS])


def prior(data):
    from scipy.stats import ks_2samp
    from ahsd.inference.pipeline import load_model
    from ahsd.inference.priors import TrainingPrior
    from ahsd.models.lean_npe import PARAM_NAMES
    print("PRIOR memmap:", sorted(Path(glob.glob(f"{data}/memmap")[0]).glob("*"))[:8])
    files = glob.glob(f"{data}/memmap/**/*param*.npy", recursive=True)
    print("  param files", files[:4])
    y = np.load(sorted(f for f in files if "train" in f)[0] if any("train" in f for f in files) else files[0], mmap_mode="r")
    print("  shape", y.shape)
    y = np.asarray(y[:: max(1, len(y) // 50000)]).reshape(-1, y.shape[-1])
    y = y[y[:, 0] > 0]
    model, meta = load_model(sys.argv[sys.argv.index("--model") + 1], device="cpu")
    pr = TrainingPrior.from_meta(meta)
    print("  prior", pr.dist_shape, pr.dist_range, pr.m1_range, pr.m2_range, "signed", pr.signed_spins)
    s = pr.sample(len(y), np.random.default_rng(0))
    for i, n in enumerate(PARAM_NAMES):
        q = lambda v: np.round(np.quantile(v, [.05, .5, .95]), 3)
        print(f"  {n:22s} KS {ks_2samp(y[:, i], s[:, i]).statistic:.3f}  data {q(y[:, i])}  prior {q(s[:, i])}")


def params(a):
    from ahsd.data.bilby_pipeline import highpass_gain
    from ahsd.inference import preprocessing as P
    from ahsd.inference.pipeline import infer
    from ahsd.models.lean_npe import PARAM_NAMES
    import dataclasses
    J = {n: i for i, n in enumerate(PARAM_NAMES)}
    ones = np.ones_like(F4)
    gps_of = gps_table()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.catalog).glob("GW*.json"))[a.part::a.nparts]
    for f in files:
        cat = json.load(open(f))
        ev = cat["event"]
        dest = out / f"{ev}.json"
        if dest.exists():
            continue
        rec = {"event": ev, "inj": []}
        try:
            gps, raw, st, srs, _ = P.fetch_gwosc(gps_of[ev], strict=False)
            t = gps - 20.0
            p0 = P.prepare_real(raw, t, st, srs, seed=0)
            rng = np.random.default_rng(sum(map(ord, ev)) + 7)
            pub = cat["published_det"]
            mc, q = pub["chirp_mass"], pub["mass_ratio"]
            m1 = mc * (1 + q) ** 0.2 / q ** 0.6
            for k in range(a.n_inj):
                par = dict(mass_1=m1, mass_2=q * m1, luminosity_distance=1000.0,
                           ra=float(rng.uniform(0, 2 * np.pi)), dec=float(np.arcsin(rng.uniform(-1, 1))),
                           theta_jn=float(np.arccos(rng.uniform(-1, 1))), psi=float(rng.uniform(0, np.pi)),
                           phase=float(rng.uniform(0, 2 * np.pi)))
                dets = list(p0.detectors_present)
                hs = {d: signal_8s(par, d, t, "IMRPhenomXP") for d in dets}

                def inject(sc):
                    r = {d: np.array(raw[d], float) for d in raw}
                    for d in dets:
                        i0 = int(round((t - st[d]) * SR)) - 16384
                        r[d][i0:i0 + len(hs[d])] += hs[d] * sc
                    return P.prepare_real(r, t, st, srs, seed=0)

                p1 = inject(1.0)
                l2 = float(np.sqrt(sum(((p1.strain[DETS.index(d)] - p0.strain[DETS.index(d)]) ** 2).sum() for d in dets)))
                sc = cat["snr"] / l2
                pa = inject(sc)
                sb = p0.strain.copy()
                for d in dets:
                    i = DETS.index(d)
                    sb[i] = (p0.strain[i] + training_signal((hs[d] * sc)[8192:8192 + 16384], d, p0.asds[d], ones)).astype(np.float32)
                pb = dataclasses.replace(p0, strain=sb)
                truth = dict(par, luminosity_distance=1000.0 / sc, geocent_time=0.0)
                row = {"truth": {n: float(truth[n]) for n in ("ra", "dec", "theta_jn", "psi", "geocent_time", "luminosity_distance")},
                       "mc": float(mc), "q": float(q), "gmst_note": t}
                for name, pr in (("inference", pa), ("training", pb)):
                    s = infer(strain=pr, event_type="BBH", model=a.model, num_samples=a.samples, device="cpu", seed=k + 1).samples
                    m = (s[:, J["mass_1"]] * s[:, J["mass_2"]]) ** .6 / (s[:, J["mass_1"]] + s[:, J["mass_2"]]) ** .2
                    v = {"chirp_mass": m, "mass_ratio": s[:, J["mass_2"]] / s[:, J["mass_1"]]}
                    for n in ("ra", "dec", "theta_jn", "geocent_time", "luminosity_distance"):
                        v[n] = s[:, J[n]]
                    row[name] = {n: [float(x) for x in np.quantile(u, [.05, .5, .95])] for n, u in v.items()}
                    # sky: circular mean/spread so a fixed offset in ra shows up
                    z = np.exp(1j * s[:, J["ra"]])
                    row[name]["ra_circ"] = [float(np.angle(z.mean())), float(np.abs(z.mean()))]
                rec["inj"].append(row)
            print(ev, "ok", flush=True)
        except Exception as ex:
            rec["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            print(ev, "failed", rec["error"], flush=True)
        dest.write_text(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True)
    ap.add_argument("--model")
    ap.add_argument("--data")
    ap.add_argument("--catalog")
    ap.add_argument("--out")
    ap.add_argument("--n_inj", type=int, default=2)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args()
    if a.section == "quick":
        shape(); curves(); prior(a.data)
    else:
        params(a)
