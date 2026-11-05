"""Why does the network read every real event too far? (2026-11-05)

For each catalogue event, with the network and preprocessing used in production:

  event       raw (training-prior) and LVK-reweighted distance, detectors, asd_bands
  injections  LVK-median masses at the event's network SNR, random sky/orientation,
              zero spin, added to the event's own raw data 20 s before the trigger
              and run through the real-event path. The truth is known, so this
              separates "the path/network is biased" from "real signals differ".
  amplitude   the injected signal's whitened L2 norm through the inference path and
              the training path (design whitening + recolor filter), against the
              exact matched-filter SNR. All three should agree.

Results: one JSON per event, resumable.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def signal_8s(params, det, gps, approximant):
    """Detector strain on an 8 s grid with the geocentre merger at the centre."""
    import bilby
    sr, dur = 4096, 8.0
    wfg = bilby.gw.WaveformGenerator(
        duration=dur, sampling_frequency=sr,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={"waveform_approximant": approximant,
                            "reference_frequency": 50.0, "minimum_frequency": 20.0})
    inj = dict(params, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0,
               geocent_time=gps)
    pol = wfg.time_domain_strain(inj)
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    h = sum(ifo.antenna_response(inj["ra"], inj["dec"], gps, inj["psi"], m) * pol[m]
            for m in ("plus", "cross"))
    dt = ifo.time_delay_from_geocenter(inj["ra"], inj["dec"], gps)
    return np.roll(h, int(sr * dur / 2) + int(round(dt * sr)))


def optimal_snr(h, asd_win):
    """Exact matched-filter SNR of an 8 s series against an ASD on the 4 s grid."""
    sr = 4096.0
    f = np.fft.rfftfreq(len(h), 1 / sr)
    asd = np.interp(f, np.fft.rfftfreq(16384, 1 / sr), asd_win)
    hf = np.fft.rfft(h) / sr
    m = f >= 20.0
    return float(np.sqrt(4 * (np.abs(hf[m]) ** 2 / asd[m] ** 2).sum() / (len(h) / sr)))


def summarise(samples, prior, J):
    from ahsd.inference.priors import lvk_log_weights, resample_weighted
    d = samples[:, J]
    rw = resample_weighted(samples, lvk_log_weights(samples, prior))[:, J]
    q = lambda v: [float(x) for x in np.quantile(v, [0.05, 0.5, 0.95])]
    return {"raw": q(d), "reweighted": q(rw)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--catalog", required=True, help="catalogue folder giving events and SNRs")
    ap.add_argument("--lvk", required=True, help="folder with <event>/lvk_posterior.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--approximant", default="IMRPhenomXP")
    ap.add_argument("--n_inj", type=int, default=2)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args()

    from ahsd.data.bilby_pipeline import BilbyPreprocessor, get_default_psd
    from ahsd.inference.pipeline import infer, load_model
    from ahsd.inference.preprocessing import _design_asd, compute_asd_bands, fetch_gwosc, prepare_real
    from ahsd.inference.priors import TrainingPrior
    from ahsd.models.lean_npe import PARAM_NAMES

    J = PARAM_NAMES.index("luminosity_distance")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    whitener = BilbyPreprocessor(4096, 4.0)
    n_bands = load_model(a.model, device="cpu")[0].encoder.psd_bands
    gps_of = {e["event_name"]: e["gps_time"] for e in json.load(open("data/gwtc/gwtc_events.json"))}
    files = sorted(Path(a.catalog).glob("GW*.json"))[a.part::a.nparts]
    for f in files:
        rec_cat = json.load(open(f))
        ev = rec_cat["event"]
        dest = out / f"{ev}.json"
        if dest.exists():
            continue
        lvk_file = Path(a.lvk) / ev / "lvk_posterior.npz"
        if lvk_file.exists():
            V = np.load(lvk_file, allow_pickle=True)
            mc, q_ = float(np.median(V["chirp_mass"])), float(np.median(V["mass_ratio"]))
            d_lvk = float(np.median(V["luminosity_distance"]))
        else:
            p = rec_cat["published_det"]
            mc, q_, d_lvk = p["chirp_mass"], p["mass_ratio"], p["luminosity_distance"]
        m1 = mc * (1 + q_) ** 0.2 / q_ ** 0.6
        rec = {"event": ev, "snr": rec_cat["snr"], "mc_lvk": mc, "q_lvk": q_, "d_lvk": d_lvk}
        try:
            gps, raw, starts, srs, found = fetch_gwosc(gps_of[ev], strict=True)
            prep = prepare_real(raw, gps, starts, srs, seed=0)
            res = infer(strain=prep, event_type="BBH", model=a.model, num_samples=a.samples,
                        device="cpu", seed=0)
            prior = TrainingPrior.from_meta(res.config)
            bands = compute_asd_bands(prep, psd_bands=n_bands)
            rec["detectors"] = list(prep.detectors_present)
            rec["asd_bands_mean"] = {d: float(bands[i].mean())
                                     for i, d in enumerate(("H1", "L1", "V1"))}
            rec["window_std"] = {d: float(prep.strain[i].std())
                                 for i, d in enumerate(("H1", "L1", "V1"))}
            rec["event_dL"] = summarise(res.samples, prior, J)

            rng = np.random.default_rng(sum(map(ord, ev)))
            t_inj = gps - 20.0
            prep0 = prepare_real(raw, t_inj, starts, srs, seed=0)
            rec["offsource_window_std"] = {d: float(prep0.strain[i].std())
                                           for i, d in enumerate(("H1", "L1", "V1"))}
            rec["injections"] = []
            for k in range(a.n_inj):
                p = dict(mass_1=m1, mass_2=q_ * m1, luminosity_distance=1000.0,
                         ra=float(rng.uniform(0, 2 * np.pi)),
                         dec=float(np.arcsin(rng.uniform(-1, 1))),
                         theta_jn=float(np.arccos(rng.uniform(-1, 1))),
                         psi=float(rng.uniform(0, np.pi)), phase=float(rng.uniform(0, 2 * np.pi)))
                dets = list(prep0.detectors_present)
                hs = {d: signal_8s(p, d, t_inj, a.approximant) for d in dets}
                snr0 = np.sqrt(sum(optimal_snr(hs[d], prep0.asds[d]) ** 2 for d in dets))
                scale = snr0 / rec["snr"]             # strain amplitude ~ 1/d
                truth = 1000.0 * scale
                raw_inj, amp = {d: np.array(raw[d], dtype=np.float64) for d in raw}, {}
                for d in dets:
                    h = hs[d] / scale
                    if abs(srs[d] - 4096) > 1:
                        raise ValueError(f"{d} sample rate {srs[d]}")
                    i0 = int(round((t_inj - starts[d]) * 4096)) - 16384
                    raw_inj[d][i0:i0 + len(h)] += h
                    amp[d] = {"snr_opt": optimal_snr(h, prep0.asds[d])}
                prep_i = prepare_real(raw_inj, t_inj, starts, srs, seed=0)
                for i, d in enumerate(("H1", "L1", "V1")):
                    if d not in dets:
                        continue
                    h = hs[d] / scale
                    amp[d]["l2_inference"] = float(np.sqrt(((prep_i.strain[i] - prep0.strain[i]) ** 2).sum()))
                    # training path: design whitening of the 4 s window, then the
                    # recolor filter into this data's measured ASD
                    h4 = h[8192:8192 + 16384]
                    sig = whitener.preprocess(h4, get_default_psd(d), detector=d).astype(np.float64)
                    design = _design_asd(d, "data/noise_bank")
                    n = min(len(design), len(prep0.asds[d]))
                    filt = np.ones(8193)
                    filt[:n] = np.clip(design[:n] / np.maximum(prep0.asds[d][:n], 1e-30), 1 / 50, 50)
                    sig_t = np.fft.irfft(np.fft.rfft(sig) * filt, n=16384)
                    amp[d]["l2_design"] = float(np.sqrt((sig ** 2).sum()))
                    amp[d]["l2_training"] = float(np.sqrt((sig_t ** 2).sum()))
                res_i = infer(strain=prep_i, event_type="BBH", model=a.model, num_samples=a.samples,
                              device="cpu", seed=k + 1)
                rec["injections"].append({"truth_dL": truth, "theta_jn": p["theta_jn"],
                                          "dL": summarise(res_i.samples, prior, J), "amp": amp})
            print(ev, "ok", flush=True)
        except Exception as ex:
            rec["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            print(ev, "failed:", rec["error"], flush=True)
        json.dump(rec, open(dest, "w"), indent=1)


if __name__ == "__main__":
    main()
