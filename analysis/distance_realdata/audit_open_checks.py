"""Open checks after the BNS sky diagnosis (2026-11-06).

validation, real noise, network SNR 10-25, BNS v6g and BBH v8:
  sky calibration   mean and fraction above 0.9 of the unbiased searched fraction
                    (the earlier 15-19% came from the biased one)
  merger time       how much mass sits in the tails, how far out, on which events
  H1+L1, V1 weak    is 1.1 rad on GW170817 injections a ring? Small H1-L1 delay error
                    with large ra error means on the ring, in the wrong place on it

GW170817 glitch: can gating restore L1 instead of dropping it?
  real event        L1 dropped vs L1 gated, scored against the optical counterpart
  injections, +32 s all detectors vs a matching hole in L1 vs no L1, known truth

The gate acts only on content above 10 Hz, so it adds no low-frequency leakage to the
strain, which is dominated by it. Width and taper are my choice, not LVK's procedure
(LVK subtracted the glitch for final parameter estimation). Now the standard:
preprocessing.KNOWN_GLITCHES.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "experiments")

T = 16384
SR = 4096
SAMPLES = 5000
DETS = ("H1", "L1", "V1")
GLITCH_BEFORE_S = 1.1  # before the merger, as published; located in the data below
# optical counterpart, and detector-frame chirp mass from the LVK low-spin analysis
EM_RA, EM_DEC = 3.4461, -0.4081
MC_DET = 1.1975


def circ(a, b):
    return float(abs((a - b + np.pi) % (2 * np.pi) - np.pi))


def sky(ra_s, dec_s, ra, dec):
    from ahsd.evaluation.path_audit import SKY_GRID, searched_fraction
    nra, nsd = SKY_GRID
    H, _, _ = np.histogram2d(np.mod(ra_s, 2 * np.pi), np.sin(dec_s), bins=[nra, nsd],
                             range=[[0, 2 * np.pi], [-1, 1]])
    p = np.sort((H / H.sum()).ravel())[::-1]
    z = np.exp(1j * ra_s).mean()
    return {"ra_err": circ(float(np.angle(z)), ra), "conc": float(abs(z)),
            "dec_err": float(abs(np.median(dec_s) - dec)),
            "searched": searched_fraction(ra_s, dec_s, ra, dec),
            "area90": float((np.searchsorted(np.cumsum(p), 0.9) + 1) * 41252.96 / (nra * nsd))}


def hl_delay(ra_s, dec_s, ra, dec, gps):
    """H1-L1 arrival difference: median error and spread, ms."""
    from bilby.gw.detector import get_empty_interferometer as ifo_of
    h, l = ifo_of("H1"), ifo_of("L1")
    d = lambda r, e: h.time_delay_from_geocenter(r, e, gps) - l.time_delay_from_geocenter(r, e, gps)
    pick = np.random.default_rng(0).choice(len(ra_s), 300)
    v = np.array([d(ra_s[j], dec_s[j]) for j in pick])
    return float(abs(np.median(v) - d(ra, dec)) * 1e3), float(np.std(v) * 1e3)


def tail(tc):
    m = np.median(tc)
    a = np.abs(tc - m)
    return {"tc_iqr": float(np.subtract(*np.quantile(tc, [.75, .25]))), "tc_std": float(np.std(tc)),
            "tail_50ms": float((a > 0.05).mean()), "tail_200ms": float((a > 0.2).mean()),
            "tc_q999": float(np.quantile(a, .999))}


def validation(model_path, data, device, n):
    from remix_data import RemixDataset
    from ahsd.data.parameter_sampler import GPS_REF
    from ahsd.inference.pipeline import load_model
    from ahsd.models.lean_npe import PARAM_NAMES as N
    model, _ = load_model(model_path, device=device)
    ds = RemixDataset(f"{data}/memmap/validation", remix=False, seed=1234, real_noise_dir="data/noise_bank",
                      real_noise_prob=1.0, return_asd_bands=True, psd_bands=16, single_signal=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in rng.permutation(ds._index_map):
        if len(rows) >= n:
            break
        start, _ = ds.events[int(i)]
        sig = np.asarray(ds.signals[start], np.float64)
        crops, filts = ds._real_noise_and_filters(rng)
        hi = np.fft.irfft(np.fft.rfft(sig[:, :T], axis=-1) * np.stack(filts) * ds._hp_gain, n=T, axis=-1)
        per = np.sqrt((hi ** 2).sum(-1))
        net = float(np.sqrt((per ** 2).sum()))
        if not 10.0 <= net < 25.0:
            continue
        if ds.lo_len:
            ctx = np.fft.irfft(np.fft.rfft(sig[:, T:], axis=-1) * np.stack(ds._last_recolor_lo) * ds._hp_gain_lo,
                               n=ds.lo_len, axis=-1)
            x = np.concatenate([hi, ctx], axis=-1) + crops
        else:
            x = hi + crops
        ab = ds._asd_bands(filts, [0, 1, 2])[None].to(device).float()
        with torch.no_grad():
            s = model.sample_posterior(torch.from_numpy(x).float()[None].to(device), rank=0, n_samples=SAMPLES,
                                       asd_bands=ab)[0].cpu().numpy()
        y = np.asarray(ds.params[start], np.float32)
        ra, dec = float(y[N.index("ra")]), float(y[N.index("dec")])
        ra_s, dec_s = s[:, N.index("ra")], s[:, N.index("dec")]
        m = sky(ra_s, dec_s, ra, dec)
        m.update(tail(s[:, N.index("geocent_time")]))
        m["delay_err"], m["delay_std"] = hl_delay(ra_s, dec_s, ra, dec, GPS_REF)
        m.update(net=net, snr=[float(v) for v in per], tc_true=float(y[N.index("geocent_time")]))
        rows.append(m)
    return rows


def med(rows, k):
    return float(np.median([r[k] for r in rows]))


def report_sky(rows, label):
    if not rows:
        print(f"  {label:32s} n=0", flush=True)
        return
    print(f"  {label:32s} n={len(rows):3d} ra_err {med(rows, 'ra_err'):.2f} conc {med(rows, 'conc'):.2f} "
          f"dec_err {med(rows, 'dec_err'):.2f} area90 {med(rows, 'area90'):.0f} | delay err {med(rows, 'delay_err'):.2f} "
          f"spread {med(rows, 'delay_std'):.2f} ms | searched {np.mean([r['searched'] for r in rows]):.3f}", flush=True)


def report_validation(name, rows):
    print(f"== {name}, n={len(rows)}", flush=True)
    s = np.array([r["searched"] for r in rows])
    k = int((s > .9).sum())
    print(f"  searched: mean {s.mean():.3f} (se {s.std() / np.sqrt(len(s)):.3f}); above 0.9 {k / len(s):.1%} (ideal 10%, "
          f"binomial se {np.sqrt(.09 / len(s)):.1%}); deciles {np.histogram(s, bins=10, range=(0, 1))[0].tolist()}", flush=True)
    snr = np.array([r["snr"] for r in rows])
    hl = (snr[:, 0] >= 5) & (snr[:, 1] >= 5) & (snr[:, 2] < 5)
    v = snr[:, 2] >= 5
    report_sky([r for r, m in zip(rows, hl) if m], "H1,L1 >= 5, V1 < 5")
    report_sky([r for r, m in zip(rows, v) if m], "V1 >= 5")
    t50 = np.array([r["tail_50ms"] for r in rows])
    print(f"  merger time: iqr {med(rows, 'tc_iqr') * 1e3:.1f} ms, std {med(rows, 'tc_std') * 1e3:.0f} ms; mass beyond 50 ms median {np.median(t50):.2%}, "
          f"mean {t50.mean():.2%}, 90th pct event {np.quantile(t50, .9):.2%}; beyond 200 ms mean {np.mean([r['tail_200ms'] for r in rows]):.2%}; "
          f"99.9% sample within {med(rows, 'tc_q999') * 1e3:.0f} ms", flush=True)
    net = np.array([r["net"] for r in rows])
    print(f"  tail mass vs SNR: 10-15 {t50[net < 15].mean():.2%}, 15-25 {t50[net >= 15].mean():.2%}; events with over 5% in tail {np.mean(t50 > .05):.1%}", flush=True)


def glitch_time(raw, starts, srs, g):
    """GPS of L1's peak whitened excursion, in the window around the event."""
    from ahsd.inference.preprocessing import prepare_real
    p = prepare_real(raw, g, starts, srs, seed=0, drop_glitches=False)
    w = p.strain[1]
    i = int(np.argmax(np.abs(w)))
    t = g - 2.0 + i / SR
    print(f"L1 window peak {np.abs(w).max():.1f} sigma at {t - g:+.3f} s from the trigger (published {-GLITCH_BEFORE_S:+.1f} s)", flush=True)
    return t


def post_gate_check(raw, starts, srs, t):
    from ahsd.inference.preprocessing import prepare_real
    p = prepare_real(raw, t, starts, srs, seed=0, drop_glitches=False)
    q = p.quality["L1"]["max_abs"]
    lo = float(np.abs(p.strain_lo[1]).max()) if p.strain_lo is not None else 0.0
    print(f"  L1 after gate: window peak {q:.1f} sigma, context peak {lo:.1f} sigma", flush=True)


def real_event(model, device):
    from ahsd.inference.pipeline import infer
    from ahsd.inference.preprocessing import fetch_gwosc, gate_glitch, prepare_real
    from ahsd.models.lean_npe import PARAM_NAMES as N
    g, raw, starts, srs, _ = fetch_gwosc("GW170817", strict=False, gate_known_glitches=False)
    tg = glitch_time(raw, starts, srs, g)
    r2 = dict(raw)
    r2["L1"] = gate_glitch(np.array(raw["L1"], np.float64), int(round((tg - starts["L1"]) * srs["L1"])), srs["L1"])
    post_gate_check(r2, starts, srs, g)
    out = {}
    for lab, r in (("L1 dropped by the glitch rule", raw), ("L1 gated", r2)):
        prep = prepare_real(r, g, starts, srs, seed=0)
        print(f"  {lab}: detectors {prep.detectors_present}", flush=True)
        rows = []
        for seed in (1, 2, 3):
            s = infer(strain=prep, event_type="BNS", model=model, num_samples=SAMPLES, device=device, seed=seed).samples
            m1, m2 = s[:, N.index("mass_1")], s[:, N.index("mass_2")]
            mc = (m1 * m2) ** .6 / (m1 + m2) ** .2
            m = sky(s[:, N.index("ra")], s[:, N.index("dec")], EM_RA, EM_DEC)
            m.update(mc=float(np.median(mc)), mc_lo=float(np.quantile(mc, .05)), mc_hi=float(np.quantile(mc, .95)),
                     dl=float(np.median(s[:, N.index("luminosity_distance")])), tc=float(np.median(s[:, N.index("geocent_time")])))
            rows.append(m)
            print(f"    seed {seed}: ra_err {m['ra_err']:.2f} dec_err {m['dec_err']:.2f} searched {m['searched']:.2f} area90 {m['area90']:.0f} deg2 | "
                  f"Mc {m['mc']:.4f} [{m['mc_lo']:.4f},{m['mc_hi']:.4f}] (published {MC_DET}) | dL {m['dl']:.0f} Mpc | tc {m['tc']:+.4f} s", flush=True)
        out[lab] = rows
    return out


def injections(model, device, n):
    from ahsd.evaluation.path_audit import CONTEXT_BEFORE_S, NS_OFFSET_S, detector_signal
    from ahsd.inference.pipeline import infer
    from ahsd.inference.preprocessing import fetch_gwosc, gate_glitch, prepare_real
    from ahsd.models.lean_npe import PARAM_NAMES as N
    g, raw, starts, srs, _ = fetch_gwosc("GW170817", strict=False, gate_known_glitches=False)
    t = g + NS_OFFSET_S
    noise = prepare_real(raw, t, starts, srs, seed=0)
    dets = list(noise.detectors_present)
    print(f"injections at +{NS_OFFSET_S:g} s: detectors {dets}", flush=True)
    m1, m2 = 1.46 * 1.009, 1.27 * 1.009
    mc_true = (m1 * m2) ** .6 / (m1 + m2) ** .2
    rng = np.random.default_rng(1)
    rows = []
    for k in range(n):
        p = dict(mass_1=m1, mass_2=m2, luminosity_distance=100.0, ra=float(rng.uniform(0, 2 * np.pi)),
                 dec=float(np.arcsin(rng.uniform(-1, 1))), theta_jn=float(np.arccos(rng.uniform(-1, 1))),
                 psi=float(rng.uniform(0, np.pi)), phase=float(rng.uniform(0, 2 * np.pi)))
        hs = {d: detector_signal(p, d, t, event_type="BNS", before_s=CONTEXT_BEFORE_S) for d in dets}

        def added(scale):
            r = {d: np.array(raw[d], dtype=np.float64) for d in raw}
            for d in dets:
                i0 = int(round((t + 4.0 - starts[d]) * SR)) - len(hs[d])
                r[d][i0:i0 + len(hs[d])] += hs[d] * scale
            return r

        first = prepare_real(added(1.0), t, starts, srs, seed=0, drop_glitches=False)
        per = {d: float(np.sqrt(((first.strain[DETS.index(d)] - noise.strain[DETS.index(d)]) ** 2).sum())) for d in dets}
        scale = 15.0 / float(np.sqrt(sum(v ** 2 for v in per.values())))
        r = added(scale)
        hole = dict(r)
        hole["L1"] = gate_glitch(r["L1"], int(round((t - GLITCH_BEFORE_S - starts["L1"]) * srs["L1"])), srs["L1"])
        row = {"k": k, "det_snr": {d: per[d] * scale for d in dets}}
        for lab, rr in (("full", r), ("gated", hole), ("no_l1", {d: v for d, v in r.items() if d != "L1"})):
            s = infer(strain=prepare_real(rr, t, starts, srs, seed=0), event_type="BNS", model=model,
                      num_samples=SAMPLES, device=device, seed=k + 1).samples
            ra_s, dec_s = s[:, N.index("ra")], s[:, N.index("dec")]
            m = sky(ra_s, dec_s, p["ra"], p["dec"])
            a, b = s[:, N.index("mass_1")], s[:, N.index("mass_2")]
            m["mc_err"] = float(abs(np.median((a * b) ** .6 / (a + b) ** .2) - mc_true))
            m["dist_ratio"] = float(np.median(s[:, N.index("luminosity_distance")]) / (100.0 / scale))
            m["delay_err"], m["delay_std"] = hl_delay(ra_s, dec_s, p["ra"], p["dec"], t)
            row[lab] = m
        rows.append(row)
        print(f"  {k}: " + " | ".join(f"{a} ra {row[a]['ra_err']:.2f} srch {row[a]['searched']:.2f}" for a in ("full", "gated", "no_l1"))
              + f" | SNR { {d: round(v, 1) for d, v in row['det_snr'].items()} }", flush=True)
    print("INJECTION SUMMARY, network SNR 15")
    for a in ("full", "gated", "no_l1"):
        report_sky([r[a] for r in rows], a)
        print(f"     Mc error {np.median([r[a]['mc_err'] for r in rows]):.5f} Msun, distance ratio {np.median([r[a]['dist_ratio'] for r in rows]):.2f}", flush=True)
    weak = [r["full"] for r in rows if r["det_snr"]["H1"] >= 5 and r["det_snr"]["L1"] >= 5 and r["det_snr"]["V1"] < 5]
    strong = [r["full"] for r in rows if r["det_snr"]["V1"] >= 5]
    report_sky(weak, "full, H1,L1 >= 5, V1 < 5")
    report_sky(strong, "full, V1 >= 5")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bns", default="model/lean_npe_bns_v6g/best_model.pth")
    ap.add_argument("--bbh", default="model/lean_npe_bbh_v8/best_model.pth")
    ap.add_argument("--out", default="analysis/distance_realdata/open_checks.json")
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_inj", type=int, default=20)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    print("REAL GW170817", flush=True)
    res["real"] = real_event(a.bns, device)
    res["inj"] = injections(a.bns, device, a.n_inj)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print("VALIDATION, real noise, network SNR 10-25", flush=True)
    res["val_bns"] = validation(a.bns, "data/dataset_bns_v4", device, a.n_val)
    report_validation("BNS v6g", res["val_bns"])
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    res["val_bbh"] = validation(a.bbh, "data/dataset_bbh_v6", device, a.n_val)
    report_validation("BBH v8", res["val_bbh"])
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
