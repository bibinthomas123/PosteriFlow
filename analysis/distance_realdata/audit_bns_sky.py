"""Why does BNS fail the real-path sky gate (1.09 rad)? (2026-11-06)

Same injections through two paths, then validation as a control:
  real path       signal in raw strain, through the real-event pipeline
  training path   same signal recoloured as RemixDataset does, added to the same noise
  validation      real-noise validation events at SNR 10-25, for BNS and BBH

Reading the result:
  real path bad, training path good       a real-path or class-injection bug
  both bad, validation good               injections differ from training
  all bad with the true sky inside the    the circular-mean metric fails on broad or
  credible region (searched < 0.9)        ring posteriors; the model may be fine
  searched fraction near 1                the posterior excludes the truth: a real failure
  delay right, ra wrong                   on the sky ring, in the wrong place along it

CAVEAT: the searched fractions here read one histogram, which biases them high for
broad posteriors (0.60 for a calibrated uniform sky with 5000 samples, on 36x18). The
gate uses path_audit.searched_fraction, which is unbiased.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

NRA, NSD = 72, 36
DEG2_PER_CELL = 41252.96 / (NRA * NSD)
DETS = ("H1", "L1", "V1")


def circ(a, b):
    return float(abs((a - b + np.pi) % (2 * np.pi) - np.pi))


def sky_metrics(ra_s, dec_s, ra, dec):
    """Circular-mean ra error, concentration, sky separation, searched fraction, 90% area."""
    z = np.exp(1j * ra_s).mean()
    H, _, _ = np.histogram2d(np.mod(ra_s, 2 * np.pi), np.sin(dec_s), bins=[NRA, NSD],
                             range=[[0, 2 * np.pi], [-1, 1]])
    p = (H / H.sum()).ravel()
    order = np.argsort(-p)
    c = np.cumsum(p[order])
    cell = min(int(np.mod(ra, 2 * np.pi) / (2 * np.pi) * NRA), NRA - 1) * NSD + min(int((np.sin(dec) + 1) / 2 * NSD), NSD - 1)
    searched = float(c[int(np.where(order == cell)[0][0])])
    sep = np.arccos(np.clip(np.sin(dec_s) * np.sin(dec) + np.cos(dec_s) * np.cos(dec) * np.cos(ra_s - ra), -1, 1))
    return {"ra_err": circ(float(np.angle(z)), ra), "conc": float(abs(z)), "dec_err": float(abs(np.median(dec_s) - dec)),
            "sep_med": float(np.median(sep)), "searched": searched,
            "area90": float((np.searchsorted(c, 0.9) + 1) * DEG2_PER_CELL)}


def summary(rows, label):
    if not rows:
        return
    f = lambda k: np.median([r[k] for r in rows])
    print(f"  {label:26s} n={len(rows):3d} ra_err {f('ra_err'):.2f} conc {f('conc'):.2f} dec_err {f('dec_err'):.2f} "
          f"sep {f('sep_med'):.2f} searched {f('searched'):.2f} (>0.9: {np.mean([r['searched'] > .9 for r in rows]):.0%}) "
          f"area90 {f('area90'):.0f} deg2", flush=True)


def injections(model, sources, n, cls, device, samples):
    from ahsd.evaluation.path_audit import SR, detector_signal, training_path_signal
    from ahsd.inference.pipeline import infer, ra_rotation
    from ahsd.inference.preprocessing import fetch_gwosc, prepare_real
    from ahsd.models.lean_npe import PARAM_NAMES as N
    from bilby.gw.detector import get_empty_interferometer as ifo_of
    rng = np.random.default_rng(0)
    out = []
    for name, m1, m2 in sources:
        g, raw, starts, srs, _ = fetch_gwosc(name, strict=False, gate_known_glitches=False)
        t = g + 20.0
        noise = prepare_real(raw, t, starts, srs, seed=0)
        dets = list(noise.detectors_present)
        rot = ra_rotation(noise)
        ifo = {d: ifo_of(d) for d in dets}
        print(f"{name} dets {dets} ra rotation {rot:.3f} rad", flush=True)
        for k in range(n):
            p = dict(mass_1=m1, mass_2=m2, luminosity_distance=100.0, ra=float(rng.uniform(0, 2 * np.pi)),
                     dec=float(np.arcsin(rng.uniform(-1, 1))), theta_jn=float(np.arccos(rng.uniform(-1, 1))),
                     psi=float(rng.uniform(0, np.pi)), phase=float(rng.uniform(0, 2 * np.pi)))
            hs = {d: detector_signal(p, d, t, event_type=cls) for d in dets}

            def inject(scale):
                r = {d: np.array(raw[d], dtype=np.float64) for d in raw}
                for d in dets:
                    i0 = int(round((t - starts[d]) * SR)) - 2 * 8192
                    r[d][i0:i0 + len(hs[d])] += hs[d] * scale
                return prepare_real(r, t, starts, srs, seed=0)

            first = inject(1.0)
            per = {d: float(np.sqrt(((first.strain[DETS.index(d)] - noise.strain[DETS.index(d)]) ** 2).sum())) for d in dets}
            scale = 15.0 / float(np.sqrt(sum(v ** 2 for v in per.values())))
            real = inject(scale)
            st = noise.strain.copy()
            for d in dets:
                i = DETS.index(d)
                st[i] = (noise.strain[i] + training_path_signal((hs[d] * scale)[8192:8192 + 16384], d, noise.asds[d])).astype(np.float32)
            train = dataclasses.replace(noise, strain=st)
            row = {"event": name, "k": k, "dets": dets, "ra": p["ra"], "dec": p["dec"],
                   "det_snr": {d: per[d] * scale for d in dets}}
            for lab, pr in (("real", real), ("train", train)):
                s = infer(strain=pr, event_type=cls, model=model, num_samples=samples, device=device, seed=k + 1).samples
                ra_s, dec_s = s[:, N.index("ra")], s[:, N.index("dec")]
                m = sky_metrics(ra_s, dec_s, p["ra"], p["dec"])
                # same samples read with no rotation and with it applied twice: a wrong
                # convention would make one of these good
                m["ra_err_norot"] = circ(float(np.angle(np.exp(1j * (ra_s - rot)).mean())), p["ra"])
                m["ra_err_double"] = circ(float(np.angle(np.exp(1j * (ra_s + rot)).mean())), p["ra"])
                m["tc"] = float(np.median(s[:, N.index("geocent_time")]))
                m["tc_std"] = float(np.std(s[:, N.index("geocent_time")]))
                if len(dets) >= 2:
                    a, b = ifo[dets[0]], ifo[dets[1]]
                    truth = a.time_delay_from_geocenter(p["ra"], p["dec"], t) - b.time_delay_from_geocenter(p["ra"], p["dec"], t)
                    pick = np.random.default_rng(k).choice(len(s), 300)
                    d = [a.time_delay_from_geocenter(ra_s[j], dec_s[j], t) - b.time_delay_from_geocenter(ra_s[j], dec_s[j], t) for j in pick]
                    m["delay_err_ms"] = float(abs(np.median(d) - truth) * 1e3)
                    m["delay_std_ms"] = float(np.std(d) * 1e3)
                row[lab] = m
            out.append(row)
            print(f"  {name} {k}: real ra_err {row['real']['ra_err']:.2f} searched {row['real']['searched']:.2f} "
                  f"delay {row['real'].get('delay_err_ms', float('nan')):.2f} ms | train ra_err {row['train']['ra_err']:.2f} "
                  f"searched {row['train']['searched']:.2f} | tc {row['real']['tc']:+.4f}", flush=True)
    return out


def validation(model_path, data, device, n, samples, lo=10.0, hi=25.0):
    import sys
    sys.path.insert(0, "experiments")
    from remix_data import RemixDataset
    from ahsd.inference.pipeline import load_model
    from ahsd.models.lean_npe import PARAM_NAMES as N
    model, _ = load_model(model_path, device=device)
    ds = RemixDataset(f"{data}/memmap/validation", remix=False, seed=1234, real_noise_dir="data/noise_bank",
                      real_noise_prob=1.0, return_asd_bands=True, psd_bands=16, single_signal=True)
    rows, i = [], 0
    while len(rows) < n and i < len(ds):
        it = ds[i]
        i += 1
        if not lo <= float(it[3]) < hi:
            continue
        with torch.no_grad():
            s = model.sample_posterior(it[0][None].to(device).float(), rank=0, n_samples=samples,
                                       asd_bands=it[4][None].to(device).float())[0].cpu().numpy()
        y = it[1][0].numpy()
        rows.append(sky_metrics(s[:, N.index("ra")], s[:, N.index("dec")], float(y[N.index("ra")]), float(y[N.index("dec")])))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/lean_npe_bns_v6g/best_model.pth")
    ap.add_argument("--bbh", default="model/lean_npe_bbh_v6g/best_model.pth")
    ap.add_argument("--out", default="analysis/distance_realdata/bns_sky.json")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # detector-frame masses at the certificate's published values and redshifts
    src = [("GW170817", 1.46 * 1.009, 1.27 * 1.009), ("GW190425", 2.0 * 1.035, 1.4 * 1.035)]

    print("VALIDATION, real noise, SNR 10-25, same metric", flush=True)
    vb = validation(a.model, "data/dataset_bns_v4", device, 300, 1000)
    summary(vb, "BNS validation")
    vc = validation(a.bbh, "data/dataset_bbh_v6", device, 300, 1000)
    summary(vc, "BBH validation")

    rows = injections(a.model, src, a.n, "BNS", device, 2000)
    Path(a.out).write_text(json.dumps({"inj": rows, "val_bns": vb, "val_bbh": vc}, indent=1, default=float))
    print("INJECTIONS, target network SNR 15")
    for ev, _, _ in src:
        for lab in ("real", "train"):
            summary([r[lab] for r in rows if r["event"] == ev], f"{ev} {lab} path")
        rr = [r["real"] for r in rows if r["event"] == ev]
        print(f"    no rotation ra_err {np.median([r['ra_err_norot'] for r in rr]):.2f}, double {np.median([r['ra_err_double'] for r in rr]):.2f}; "
              f"delay error {np.median([r.get('delay_err_ms', np.nan) for r in rr]):.2f} ms, spread {np.median([r.get('delay_std_ms', np.nan) for r in rr]):.2f} ms; "
              f"tc {np.median([r['tc'] for r in rr]):+.4f} s, tc spread {np.median([r['tc_std'] for r in rr]):.4f} s")
        print(f"    per-detector SNR, median: " + str({d: round(float(np.median([r['det_snr'][d] for r in rows if r['event'] == ev])), 1)
                                                        for d in next(r['dets'] for r in rows if r['event'] == ev)}))
