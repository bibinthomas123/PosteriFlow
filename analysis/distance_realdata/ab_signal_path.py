"""Controlled A/B for the amplitude mismatch (2026-11-05).

The noise bank's ASD is measured after gwpy's 15 Hz highpass, whose passband is
not flat, while stored training signals never pass through that filter. Same noise,
same source, same asd_bands; only the signal path differs:

  inference  signal added to raw strain and whitened by the real-event path
  training   design-whitened signal x (design ASD / measured ASD), added to the
             whitened noise, as RemixDataset does

The source is scaled so its inference-path L2 (the true SNR) equals the event's
network SNR. If the mismatch causes the bias, the training-path copy recovers the
true distance and the inference-path copy reads far.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from diagnose import signal_8s, summarise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--lvk", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_inj", type=int, default=3)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args()

    from ahsd.data.bilby_pipeline import BilbyPreprocessor, get_default_psd
    from ahsd.inference.pipeline import infer
    from ahsd.inference.preprocessing import _design_asd, fetch_gwosc, prepare_real
    from ahsd.inference.priors import TrainingPrior
    from ahsd.models.lean_npe import PARAM_NAMES

    J = PARAM_NAMES.index("luminosity_distance")
    DETS = ("H1", "L1", "V1")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    whitener = BilbyPreprocessor(4096, 4.0)
    gps_of = {e["event_name"]: e["gps_time"] for e in json.load(open("data/gwtc/gwtc_events.json"))}
    for f in sorted(Path(a.catalog).glob("GW*.json"))[a.part::a.nparts]:
        cat = json.load(open(f))
        ev = cat["event"]
        dest = out / f"{ev}.json"
        if dest.exists():
            continue
        V = Path(a.lvk) / ev / "lvk_posterior.npz"
        if V.exists():
            V = np.load(V, allow_pickle=True)
            mc, q = float(np.median(V["chirp_mass"])), float(np.median(V["mass_ratio"]))
        else:
            mc, q = cat["published_det"]["chirp_mass"], cat["published_det"]["mass_ratio"]
        m1 = mc * (1 + q) ** 0.2 / q ** 0.6
        rec = {"event": ev, "snr": cat["snr"], "injections": []}
        try:
            gps, raw, starts, srs, _ = fetch_gwosc(gps_of[ev], strict=False)
            t = gps - 20.0
            prep0 = prepare_real(raw, t, starts, srs, seed=0)
            rec["detectors"] = list(prep0.detectors_present)
            rng = np.random.default_rng(sum(map(ord, ev)))
            for k in range(a.n_inj):
                p = dict(mass_1=m1, mass_2=q * m1, luminosity_distance=1000.0,
                         ra=float(rng.uniform(0, 2 * np.pi)), dec=float(np.arcsin(rng.uniform(-1, 1))),
                         theta_jn=float(np.arccos(rng.uniform(-1, 1))), psi=float(rng.uniform(0, np.pi)),
                         phase=float(rng.uniform(0, 2 * np.pi)))
                dets = list(prep0.detectors_present)
                hs = {d: signal_8s(p, d, t, "IMRPhenomXP") for d in dets}

                def inject(scale):
                    r = {d: np.array(raw[d], dtype=np.float64) for d in raw}
                    for d in dets:
                        i0 = int(round((t - starts[d]) * 4096)) - 16384
                        r[d][i0:i0 + len(hs[d])] += hs[d] * scale
                    return prepare_real(r, t, starts, srs, seed=0)

                p1 = inject(1.0)
                l2 = np.sqrt(sum(((p1.strain[DETS.index(d)] - prep0.strain[DETS.index(d)]) ** 2).sum() for d in dets))
                scale = rec["snr"] / l2
                truth = 1000.0 / scale
                pa = inject(scale)
                sb = prep0.strain.copy()
                amp = {}
                for d in dets:
                    i = DETS.index(d)
                    h4 = (hs[d] * scale)[8192:8192 + 16384]
                    sig = whitener.preprocess(h4, get_default_psd(d), detector=d).astype(np.float64)
                    design = _design_asd(d, "data/noise_bank")
                    n = min(len(design), len(prep0.asds[d]))
                    filt = np.ones(8193)
                    filt[:n] = np.clip(design[:n] / np.maximum(prep0.asds[d][:n], 1e-30), 1 / 50, 50)
                    st = np.fft.irfft(np.fft.rfft(sig) * filt, n=16384)
                    sb[i] = (prep0.strain[i] + st).astype(np.float32)
                    amp[d] = {"inference": float(np.sqrt(((pa.strain[i] - prep0.strain[i]) ** 2).sum())),
                              "training": float(np.sqrt((st ** 2).sum()))}
                pb = dataclasses.replace(prep0, strain=sb)
                row = {"truth_dL": truth, "theta_jn": p["theta_jn"], "amp": amp}
                for name, pr in (("inference", pa), ("training", pb)):
                    res = infer(strain=pr, event_type="BBH", model=a.model, num_samples=a.samples,
                                device="cpu", seed=k + 1)
                    row[name] = summarise(res.samples, TrainingPrior.from_meta(res.config), J)
                rec["injections"].append(row)
            print(ev, "ok", flush=True)
        except Exception as ex:
            rec["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            print(ev, "failed:", rec["error"], flush=True)
        dest.write_text(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main()
