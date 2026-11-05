#!/usr/bin/env python3
"""Does a continuous local search over (chirp mass, q, chi) remove the bank's
chirp-mass error floor? (Experiment 3; bank_floor_probe.py is 1-2.)

Per held-out single signal: the dense bank's best slot and its interpolated
peak ln Mc ("bank"); then templates generated on the fly (same recipe as the
bank: IMRPhenomXP, design-PSD whitened, 4 s circular window, equal aligned
spins) on a local grid around it, scored with the encoder's own coincident
network matched-filter statistic (McFeatures._peak_snr2, fine-stage lags):

  local   two zoom rounds over ln Mc x ln q x chi (13 x 7 x 7, then half the
          ranges around the best), parabolic peak in ln Mc at the best (q, chi)
  truthqc ln Mc scanned with the TRUE q and the TRUE component spins (a1, a2)
          held fixed: what the statistic gives when nothing but chirp mass is
          free -- if this still rises with SNR the floor is not (q, chi) at all

Reports median |err ln Mc| x SNR per SNR bin (flat = no floor).

    python analysis/bank_floor/continuous_search_probe.py --model model/lean_npe_ab8_dense/best_model.pth \
        --data data/dataset_bbh_v6 --n_loud 250 --n_quiet 250 --workers 46 --out analysis/bank_floor/continuous.json
"""
import argparse
import json
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis/bank_floor"))

from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

J = {n: i for i, n in enumerate(PARAM_NAMES)}
BINS = ((8, 12), (12, 20), (20, 35), (35, 60), (60, 1e9))
SR, T_LEN, F_LO, F_HI, DETS = 4096, 16384, 20.0, 1024.0, ("H1", "L1", "V1")
_G = {}


def _init():
    import bilby
    logging.getLogger("bilby").setLevel(logging.ERROR)
    from ahsd.data.bilby_pipeline import get_default_psd
    psd = {d: get_default_psd(d) for d in DETS}
    fw = np.fft.rfftfreq(T_LEN, 1.0 / SR)
    band = (fw >= F_LO) & (fw < F_HI)
    _G.update(bilby=bilby, fw=fw, lo=int(np.argmax(band)), nb=int(band.sum()), gens={},
              asd={d: np.sqrt(np.interp(fw, psd[d]["frequencies"], psd[d]["psd"], left=np.inf, right=np.inf))
                   for d in DETS})


def template(args):
    """[D, n_band] complex64 unit-norm window template (build_mc_bank recipe)."""
    m1, m2, a1, a2 = args
    bilby = _G["bilby"]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    fmin = round(max(F_LO, 100.0 * (2.18 * (1.21 / mc) ** (5 / 3) / 3.6) ** (3 / 8)), 1)
    if fmin not in _G["gens"]:
        _G["gens"][fmin] = bilby.gw.WaveformGenerator(
            duration=T_LEN / SR, sampling_frequency=SR,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments={"waveform_approximant": "IMRPhenomXP",
                                "reference_frequency": 50.0, "minimum_frequency": fmin})
    hp = _G["gens"][fmin].frequency_domain_strain(dict(
        mass_1=m1, mass_2=m2, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0,
        a_1=abs(a1), a_2=abs(a2), tilt_1=np.pi if a1 < 0 else 0.0, tilt_2=np.pi if a2 < 0 else 0.0,
        phi_12=0.0, phi_jl=0.0, ra=0.0, dec=0.0, geocent_time=0.0))["plus"]
    lo, nb = _G["lo"], _G["nb"]
    rows = []
    for d in DETS:
        h = (hp / _G["asd"][d])[lo:lo + nb]
        rows.append(h / np.sqrt((np.abs(h) ** 2).sum() + 1e-300))
    return np.stack(rows).astype(np.complex64)


def m12(lmc, q):
    mc = np.exp(lmc)
    m1 = mc * (1 + q) ** 0.2 / q ** 0.6
    return m1, m1 * q


def parab(x, y):
    k = int(np.argmax(y))
    if 0 < k < len(y) - 1:
        den = y[k - 1] - 2 * y[k] + y[k + 1]
        if den < 0:
            dk = np.clip(0.5 * (y[k - 1] - y[k + 1]) / den, -1, 1)
            return x[k] + dk * (x[1] - x[0])
    return x[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_loud", type=int, default=250, help="events with SNR >= 35")
    ap.add_argument("--n_quiet", type=int, default=250, help="events with SNR < 35")
    ap.add_argument("--workers", type=int, default=46)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from ahsd.inference.pipeline import load_model
    from bank_floor_probe import peak_interp
    torch.set_num_threads(max(1, a.workers))
    model = load_model(a.model, "cpu")[0].eval()
    mf = model.encoder.mc_feat
    margs = torch.load(a.model, map_location="cpu", weights_only=False)["args"]
    b = np.load(ROOT / "src/ahsd/models/assets" / str(margs["mc_bank"]).split(":")[0])
    slots = b["mc_q"][:int(b["n_q"]), 1:3]
    c = Path(a.data) / "memmap" / "validation"
    ev = json.load(open(c / "events.json"))["events"]
    sig, noi, par = (np.load(c / f, mmap_mode="r") for f in ("signals.npy", "noise.npy", "params.npy"))
    singles = [(e, s) for e, (s, n) in enumerate(ev) if n == 1]
    snr_of = lambda s: float(np.sqrt((np.asarray(sig[s], np.float64) ** 2).sum()))  # noqa: E731
    loud, quiet = [], []
    for e, s in singles:
        x = snr_of(s)
        (loud if x >= 35 else quiet).append((e, s, x))
        if len(loud) >= a.n_loud and len(quiet) >= a.n_quiet:
            break
    sel = loud[:a.n_loud] + quiet[:a.n_quiet]
    print(f"{len(sel)} events ({len(loud[:a.n_loud])} with SNR >= 35)", flush=True)
    sw1 = mf.n_band / np.sqrt(T_LEN / 2.0)

    def score(d, H):
        Ht = torch.from_numpy(np.stack(H)).permute(1, 0, 2)[None]      # [1, D, K, F]
        with torch.no_grad():
            return mf._peak_snr2(d, Ht, 2 * mf.n_band, 2 * sw1, T_LEN / SR)[0].numpy()

    rows = []
    with Pool(a.workers, initializer=_init) as pool:
        for i, (e, s, snr) in enumerate(sel):
            x = torch.from_numpy((np.asarray(noi[e % len(noi)], np.float32) + np.asarray(sig[s], np.float32))[None])
            with torch.no_grad():
                d = torch.fft.rfft(x.float(), dim=-1)[..., mf.band_lo:mf.band_lo + mf.n_band]
                prof_q = mf._profiles(x)[-1]
                valid = prof_q > -0.5
                e_bank, _ = peak_interp(prof_q.amax(-1), valid.any(-1), mf.log_grid, mf.dlog)
                lmc_s, pk_s = peak_interp(prof_q.transpose(1, 2), valid.transpose(1, 2), mf.log_grid, mf.dlog)
            j = int(pk_s[0].argmax())
            lmc0, lq0, chi0 = float(lmc_s[0, j]), float(np.log(slots[j, 0])), float(slots[j, 1])
            # two zoom rounds
            span = np.array([0.03, 0.3, 0.15])
            for _ in range(2):
                g_l = lmc0 + np.linspace(-span[0], span[0], 13)
                g_q = np.clip(lq0 + np.linspace(-span[1], span[1], 7), np.log(0.05), 0.0)
                g_c = np.clip(chi0 + np.linspace(-span[2], span[2], 7), -0.95, 0.95)
                pts = [(l_, q_, c_) for l_ in g_l for q_ in g_q for c_ in g_c]
                H = pool.map(template, [(*m12(l_, np.exp(q_)), c_, c_) for l_, q_, c_ in pts], chunksize=8)
                sc = score(d, H).reshape(13, 7, 7)
                il, iq, ic = np.unravel_index(sc.argmax(), sc.shape)
                lmc0 = float(parab(g_l, sc[:, iq, ic]))
                lq0, chi0 = float(g_q[iq]), float(g_c[ic])
                span = span / 2
            e_local = lmc0
            # truth (q, a1, a2) held fixed; only chirp mass free
            t = np.asarray(par[s][:len(PARAM_NAMES)], np.float64)
            m1, m2 = max(t[0], t[1]), min(t[0], t[1])
            a1, a2 = (t[J["a1"]], t[J["a2"]]) if t[0] >= t[1] else (t[J["a2"]], t[J["a1"]])
            lt = np.log((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)
            g_l = lt + np.linspace(-0.03, 0.03, 31)
            H = pool.map(template, [(*m12(l_, m2 / m1), a1, a2) for l_ in g_l], chunksize=2)
            e_truth = float(parab(g_l, score(d, H)))
            rows.append((snr, float(e_bank[0]) - lt, e_local - lt, e_truth - lt))
            if i % 25 == 0:
                r = np.array(rows)
                print(f"  {i}/{len(sel)}  median |err| x SNR so far: bank {np.median(np.abs(r[:, 1]) * r[:, 0]):.3f}"
                      f"  local {np.median(np.abs(r[:, 2]) * r[:, 0]):.3f}  truthqc {np.median(np.abs(r[:, 3]) * r[:, 0]):.3f}",
                      flush=True)
    r = np.array(rows)
    names = ["bank", "local", "truthqc"]
    rep = {"model": a.model, "n": len(r), "bins": {}}
    print("\nmedian |err ln Mc| x SNR (flat = no floor); raw % and signed median % in brackets")
    print(f"{'SNR bin':9s} {'n':>4s} " + " ".join(f"{n:>22s}" for n in names))
    for lo, hi in BINS:
        k = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if k.sum() < 5:
            continue
        s = float(np.median(r[k, 0]))
        v = [float(np.median(np.abs(r[k, 1 + i]))) for i in range(3)]
        sg = [float(np.median(r[k, 1 + i])) for i in range(3)]
        rep["bins"][f"{lo}-{hi:g}"] = {"n": int(k.sum()), "snr": s,
                                      **{n: {"err_x_snr": x * s, "raw_pct": 100 * x, "signed_pct": 100 * g}
                                         for n, x, g in zip(names, v, sg)}}
        print(f"{lo:>3}-{hi:<5g} {k.sum():4d} " + " ".join(f"{x * s:6.3f} ({100 * x:4.2f}, {100 * g:+5.2f})"
                                                           for x, g in zip(v, sg)))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=1)
        np.save(Path(a.out).with_suffix(".npy"), r)


if __name__ == "__main__":
    main()
