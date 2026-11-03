#!/usr/bin/env python3
"""Where does the network get unequal masses wrong, and is it the data?

Per held-out single-signal event (validation split, Gaussian noise): posterior
samples from the model, then coverage of the 90% interval, median bias and
width for m1, m2, mass ratio and chirp mass, binned by TRUE mass ratio and by
SNR. Also reports how the training split populates those bins (the SNR cut
removes quiet, unequal-mass systems preferentially).

    python scripts/q_bias_probe.py --model model/lean_npe_bbh_v5/best_model.pth \
        --data data/dataset_bbh_v4 [--n 4000] [--out analysis/q_bias/bbh_v5.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ahsd.inference.pipeline import load_model  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

J = {n: i for i, n in enumerate(PARAM_NAMES)}
Q_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
SNR_BINS = [(0, 12), (12, 20), (20, 1e9)]


def split(data, name):
    c = Path(data) / "memmap" / name
    ev = json.load(open(c / "events.json"))["events"]
    return (np.load(c / "signals.npy", mmap_mode="r"), np.load(c / "noise.npy", mmap_mode="r"),
            np.load(c / "params.npy", mmap_mode="r"), ev)


def derived(m1, m2):
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    return q, mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.model, dev)[0].eval()

    # ---- training population per bin (all single-signal training events) ----
    sig_t, _, par_t, ev_t = split(a.data, "train")
    single = [s for s, n in ev_t if n == 1]
    pt = np.asarray(par_t[single][:, :len(PARAM_NAMES)], np.float64)
    qt, _ = derived(pt[:, J["mass_1"]], pt[:, J["mass_2"]])
    pop = {f"q {lo:.2f}-{min(hi, 1):.2f}": int(((qt >= lo) & (qt < hi)).sum()) for lo, hi in Q_BINS}
    print(f"training single-signal events: {len(qt)}; by true q: {pop}")

    # ---- held-out posteriors ----
    sig, noi, par, ev = split(a.data, "validation")
    sel = [(e, s) for e, (s, n) in enumerate(ev) if n == 1][: a.n]
    rows = []
    with torch.no_grad():
        for b0 in range(0, len(sel), 64):
            ch = sel[b0:b0 + 64]
            x = np.stack([np.asarray(noi[e % len(noi)], np.float32) + np.asarray(sig[s], np.float32)
                          for e, s in ch])
            snr = np.sqrt((np.stack([np.asarray(sig[s], np.float64) for _, s in ch]) ** 2).sum((1, 2)))
            xt = torch.from_numpy(x).to(dev)
            ab = torch.zeros(len(ch), 3, model.encoder.psd_bands, device=dev) if model.psd_cond else None
            smp = model.sample_posterior(xt, n_samples=a.samples, asd_bands=ab).cpu().numpy()
            for i, (_, s) in enumerate(ch):
                t = np.asarray(par[s][:len(PARAM_NAMES)], np.float64)
                m1s, m2s = smp[i, :, J["mass_1"]], smp[i, :, J["mass_2"]]
                qs, mcs = derived(m1s, m2s)
                qt_, mct = derived(t[J["mass_1"]], t[J["mass_2"]])
                rec = {"q_true": float(qt_), "snr": float(snr[i])}
                for name, post, truth in (("m1", np.maximum(m1s, m2s), max(t[0], t[1])),
                                          ("m2", np.minimum(m1s, m2s), min(t[0], t[1])),
                                          ("q", qs, qt_), ("mc", mcs, mct)):
                    lo, med, hi = np.quantile(post, [0.05, 0.5, 0.95])
                    rec[name] = {"in90": bool(lo <= truth <= hi), "logbias": float(np.log(med / truth)),
                                 "relw90": float((hi - lo) / truth)}
                rows.append(rec)

    def table(key, bins, label):
        out = {}
        print(f"\nby {label}: n | 90% coverage m1 m2 q Mc | median log-bias m1 q Mc | median rel. width q Mc")
        for lo, hi in bins:
            r = [x for x in rows if lo <= x[key] < hi]
            if not r:
                continue
            cov = {p: float(np.mean([x[p]["in90"] for x in r])) for p in ("m1", "m2", "q", "mc")}
            bias = {p: float(np.median([x[p]["logbias"] for x in r])) for p in ("m1", "q", "mc")}
            wid = {p: float(np.median([x[p]["relw90"] for x in r])) for p in ("q", "mc")}
            tag = f"{lo:g}-{min(hi, 1.0) if key == 'q_true' else hi:g}"
            out[tag] = {"n": len(r), "cov90": cov, "logbias": bias, "relw90": wid}
            print(f"  {tag:10s} {len(r):5d} | {cov['m1']:.2f} {cov['m2']:.2f} {cov['q']:.2f} {cov['mc']:.2f} | "
                  f"{bias['m1']:+.3f} {bias['q']:+.3f} {bias['mc']:+.3f} | {wid['q']:.2f} {wid['mc']:.3f}")
        return out

    res = {"model": a.model, "train_pop_by_q": pop, "n": len(rows),
           "by_q": table("q_true", Q_BINS, "true mass ratio"),
           "by_snr": table("snr", SNR_BINS, "SNR")}
    lowq = [x for x in rows if x["q_true"] < 0.25]
    res["low_q_by_snr"] = {}
    print("\nq < 0.25 only, by SNR:")
    for lo, hi in SNR_BINS:
        r = [x for x in lowq if lo <= x["snr"] < hi]
        if r:
            c = float(np.mean([x["m1"]["in90"] for x in r])); b = float(np.median([x["m1"]["logbias"] for x in r]))
            res["low_q_by_snr"][f"{lo}-{hi}"] = {"n": len(r), "m1_cov90": c, "m1_logbias": b}
            print(f"  SNR {lo}-{hi}: n {len(r)}  m1 coverage {c:.2f}  m1 median log-bias {b:+.3f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
