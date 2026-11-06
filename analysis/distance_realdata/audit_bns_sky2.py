"""Why is the BNS sky wrong on injections? Three candidate causes (2026-11-06)

audit_bns_sky.py found real path = training path, validation good (0.20 rad), and
injections broad and far off (1.2-1.4 rad). Those injections differed from training
in ways that also apply to both of its paths:
  detector     GW170817 at +20 s loses L1 to the glitch rule; H1 was off for GW190425
  no context   the signal was cut to 8 s, so the 28 s context stream had no inspiral
  own signal   at +20 s the context reaches 10 s before the event, so it holds the
               event's own inspiral and merger

All injections here go at +32 s, context clear of the event and its glitch. Arms, same
injections and seeds:
  full         all detectors, inspiral through the whole context stream
  no L1        L1 strain removed before prepare_real (the real GW170817 situation)
  8 s cut      the signal as the gate injects it now
Validation is binned by second-loudest detector SNR, with the context stream built
as RemixDataset does.

Searched fraction on a 36x18 grid with 5000 samples: the 72x36 grid with 2000 samples
leaves cells of a broad posterior empty, and a truth in an empty cell reads 1.0.

Reading: full good and 8 s cut bad = the gate's injections are wrong. Full good and
no L1 bad = the missing detector. Full bad with validation good at the same second
SNR = something else differs from training.

CAVEAT: the searched fractions here read one histogram, which biases them high for
broad posteriors (0.60 for a calibrated uniform sky with 5000 samples, on 36x18). The
gate uses path_audit.searched_fraction, which is unbiased.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "experiments")
sys.path.insert(0, "analysis/distance_realdata")
import audit_bns_sky as base

base.NRA, base.NSD = 36, 18
base.DEG2_PER_CELL = 41252.96 / (36 * 18)
DETS = ("H1", "L1", "V1")
SAMPLES = 5000
T = 16384
BINS = ((0, 3), (3, 5), (5, 8), (8, 1e9))
OFFSET = 32.0   # context starts 2 s after the event; the injection ends before the fetch does
BEFORE = 34     # s of inspiral kept before the merger: the 2 s of window, 28 s of context, 4 s spare


def metrics(s, ra, dec):
    from ahsd.models.lean_npe import PARAM_NAMES as N
    m = base.sky_metrics(s[:, N.index("ra")], s[:, N.index("dec")], ra, dec)
    tc = s[:, N.index("geocent_time")]
    m["tc_std"] = float(np.std(tc))
    m["tc_iqr"] = float(np.subtract(*np.quantile(tc, [.75, .25])))
    return m


def report(rows, label):
    if not rows:
        print(f"  {label:34s} n=0", flush=True)
        return
    f = lambda k: float(np.median([r[k] for r in rows]))
    print(f"  {label:34s} n={len(rows):3d} ra_err {f('ra_err'):.2f} conc {f('conc'):.2f} sep {f('sep_med'):.2f} "
          f"searched mean {np.mean([r['searched'] for r in rows]):.2f} (>0.9: {np.mean([r['searched'] > .9 for r in rows]):.0%}) "
          f"area90 {f('area90'):.0f} deg2 | tc std {f('tc_std'):.4f} iqr {f('tc_iqr'):.4f} s", flush=True)


def validation(model_path, data, device, per_bin=100, lo=10.0, hi=25.0):
    from remix_data import RemixDataset
    from ahsd.inference.pipeline import load_model
    from ahsd.models.lean_npe import PARAM_NAMES as N
    model, _ = load_model(model_path, device=device)
    ds = RemixDataset(f"{data}/memmap/validation", remix=False, seed=1234, real_noise_dir="data/noise_bank",
                      real_noise_prob=1.0, return_asd_bands=True, psd_bands=16, single_signal=True)
    rng = np.random.default_rng(0)
    rows = {b: [] for b in BINS}
    for i in rng.permutation(ds._index_map):
        if all(len(v) >= per_bin for v in rows.values()):
            break
        start, _ = ds.events[int(i)]
        sig = np.asarray(ds.signals[start], np.float64)
        crops, filts = ds._real_noise_and_filters(rng)
        s = np.fft.irfft(np.fft.rfft(sig[:, :T], axis=-1) * np.stack(filts) * ds._hp_gain, n=T, axis=-1)
        ctx = np.fft.irfft(np.fft.rfft(sig[:, T:], axis=-1) * np.stack(ds._last_recolor_lo) * ds._hp_gain_lo,
                          n=ds.lo_len, axis=-1)
        per = np.sqrt((s ** 2).sum(-1))
        net = float(np.sqrt((per ** 2).sum()))
        if not lo <= net < hi:
            continue
        second = float(np.sort(per)[-2])
        b = next(b for b in BINS if b[0] <= second < b[1])
        if len(rows[b]) >= per_bin:
            continue
        x = torch.from_numpy(np.concatenate([s, ctx], axis=-1) + crops).float()[None].to(device)
        ab = ds._asd_bands(filts, [0, 1, 2])[None].to(device).float()
        with torch.no_grad():
            out = model.sample_posterior(x, rank=0, n_samples=SAMPLES, asd_bands=ab)[0].cpu().numpy()
        y = np.asarray(ds.params[start], np.float32)
        m = metrics(out, float(y[N.index("ra")]), float(y[N.index("dec")]))
        m["second_snr"], m["net_snr"] = second, net
        rows[b].append(m)
    return rows


def injections(model, cls, name, m1, m2, n, device, drop=None, cut=False):
    """Network SNR 15 at +32 s, the context clear of the event. Arm "full" always; "no
    <drop>" removes that detector's strain; "cut" keeps only the last 4 s of inspiral."""
    from ahsd.evaluation.path_audit import SR, TAPER_S, detector_signal
    from ahsd.inference.pipeline import infer
    from ahsd.inference.preprocessing import fetch_gwosc, prepare_real
    g, raw, starts, srs, _ = fetch_gwosc(name, strict=False, gate_known_glitches=False)
    t = g + OFFSET
    noise = prepare_real(raw, t, starts, srs, seed=0)
    dets = list(noise.detectors_present)
    print(f"{name} +{OFFSET:g} s: detectors {dets}; warnings {noise.warnings}", flush=True)
    rng = np.random.default_rng(1)
    out = []
    for k in range(n):
        p = dict(mass_1=m1, mass_2=m2, luminosity_distance=100.0, ra=float(rng.uniform(0, 2 * np.pi)),
                 dec=float(np.arcsin(rng.uniform(-1, 1))), theta_jn=float(np.arccos(rng.uniform(-1, 1))),
                 psi=float(rng.uniform(0, np.pi)), phase=float(rng.uniform(0, 2 * np.pi)))
        hs = {d: detector_signal(p, d, t, event_type=cls, before_s=BEFORE) for d in dets}

        def added(scale, short=False):
            r = {d: np.array(raw[d], dtype=np.float64) for d in raw}
            for d in dets:
                skip = (BEFORE - 4) * SR if short else 0
                h = (hs[d] * scale)[skip:].copy()
                if short:
                    w = int(TAPER_S * SR)
                    h[:w] *= np.sin(0.5 * np.pi * np.arange(w) / w) ** 2
                i0 = int(round((t - starts[d]) * SR)) - (BEFORE * SR - skip)
                r[d][i0:i0 + len(h)] += h
            return r

        first = prepare_real(added(1.0), t, starts, srs, seed=0, drop_glitches=False)
        per = {d: float(np.sqrt(((first.strain[DETS.index(d)] - noise.strain[DETS.index(d)]) ** 2).sum())) for d in dets}
        scale = 15.0 / float(np.sqrt(sum(v ** 2 for v in per.values())))
        row = {"event": name, "k": k, "ra": p["ra"], "dec": p["dec"], "det_snr": {d: per[d] * scale for d in dets}}

        def run(r):
            prep = prepare_real(r, t, starts, srs, seed=0)
            s = infer(strain=prep, event_type=cls, model=model, num_samples=SAMPLES, device=device, seed=k + 1).samples
            return metrics(s, p["ra"], p["dec"])

        r = added(scale)
        row["full"] = run(r)
        if drop:
            row["dropped"] = run({d: v for d, v in r.items() if d != drop})
        if cut:
            row["cut"] = run(added(scale, short=True))
        out.append(row)
        print(f"  {name} {k}: " + " | ".join(f"{a} ra_err {row[a]['ra_err']:.2f} searched {row[a]['searched']:.2f}"
                                             for a in ("full", "dropped", "cut") if a in row)
              + f" | SNR { {d: round(v, 1) for d, v in row['det_snr'].items()} }", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/lean_npe_bns_v6g/best_model.pth")
    ap.add_argument("--data", default="data/dataset_bns_v4")
    ap.add_argument("--out", default="analysis/distance_realdata/bns_sky2.json")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m170817 = (1.46 * 1.009, 1.27 * 1.009)
    m190425 = (2.0 * 1.035, 1.4 * 1.035)

    print("VALIDATION, real noise, network SNR 10-25, by second-loudest detector SNR", flush=True)
    val = validation(a.model, a.data, device)
    for b, rows in val.items():
        report(rows, f"second detector SNR {b[0]}-{b[1]:g}")

    inj = injections(a.model, "BNS", "GW170817", *m170817, a.n, device, drop="L1", cut=True)
    inj2 = injections(a.model, "BNS", "GW190425", *m190425, a.n // 2, device)
    Path(a.out).write_text(json.dumps({"val": {f"{b[0]}-{b[1]:g}": r for b, r in val.items()},
                                       "gw170817": inj, "gw190425": inj2}, indent=1, default=float))
    print("INJECTIONS, target network SNR 15, +32 s")
    report([r["full"] for r in inj], "GW170817 full, H1 L1 V1")
    report([r["dropped"] for r in inj], "GW170817 no L1")
    report([r["cut"] for r in inj], "GW170817 8 s cut")
    report([r["full"] for r in inj2], "GW190425 full, L1 V1")
    second = np.array([np.sort(list(r["det_snr"].values()))[-2] for r in inj])
    err = np.array([r["full"]["ra_err"] for r in inj])
    for lo, hi in BINS:
        m = (second >= lo) & (second < hi)
        if m.any():
            print(f"  GW170817 full, second detector SNR {lo}-{hi:g}: n={int(m.sum())} ra_err {np.median(err[m]):.2f}")
