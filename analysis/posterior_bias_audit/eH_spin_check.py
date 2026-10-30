#!/usr/bin/env python3
"""E-H: the missing reference measurement — dynesty vs LeanNPE on IDENTICAL strain.

Every comparison in this audit so far has been NPE-vs-published-GWTC on real
events, where the prior, the waveform family and the noise all differ. That
cannot separate "the encoder throws information away" from "the information is
not there at these SNRs". This does: same injection, same noise realization,
same priors, one reference sampler and one amortized estimator.

Decision logic:
  dynesty q also broad   -> the q deficit is largely PHYSICAL; stop chasing q,
                            target Mc (which has no such alternative explanation).
  dynesty q tight        -> the q deficit is a MODEL deficit; a q channel is worth
                            building (3-D (Mc,q,chi_eff) bank, or a new front-end).
The Mc ratio is reported alongside and is the primary number either way.

Injections are drawn from the TRAINING prior (parameter_sampler), then distance-
scaled onto an SNR ladder so the comparison is resolved in SNR rather than
confounded by it. Uses basin_probe._projected, which applies per-detector time
delays correctly (unlike gw150914_localization/_common.py — see REPORT.md Sec.0).
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch

from ahsd.evaluation.basin_probe import _projected, _design_noise, GPS_REF, DETS
from ahsd.data.bilby_pipeline import get_default_psd
from ahsd.inference.dynesty_bridge import run_dynesty
from ahsd.inference.preprocessing import prepare_simulated
from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES

OUT = Path(__file__).resolve().parent / "eH_spin"
OUT.mkdir(exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_EVENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NLIVE = int(sys.argv[2]) if len(sys.argv) > 2 else 300


SR, DUR = 4096, 4.0
_N = int(SR * DUR); _DT = 1.0 / SR; _DF = 1.0 / DUR
_FREQS = np.fft.rfftfreq(_N, _DT)
_BAND = (_FREQS >= 20.0) & (_FREQS < 1024.0)
_PSD = None


def net_snr(sig):
    """Optimal matched-filter network SNR: sqrt(sum_d 4 int |h~|^2/S df).

    NOT the L2 norm of `prepare_simulated(...).strain`: that path estimates the
    PSD from the data it is given, so whitening a SIGNAL-ONLY array self-
    normalizes and the amplitude (hence the distance dependence) largely cancels
    — measured 154.6 -> 129.9 for a 3.55x distance increase that must give 3.55x.
    This analytic form reproduces basin_probe's own SNR labels (87.0 vs "snr85",
    24.5 vs "snr24").
    """
    global _PSD
    if _PSD is None:
        _PSD = {d: np.interp(_FREQS,
                             np.asarray(get_default_psd(d)["frequencies"], float),
                             np.asarray(get_default_psd(d)["psd"], float),
                             left=np.inf, right=np.inf) for d in DETS}
    tot = 0.0
    for d in DETS:
        ht = np.fft.rfft(sig[d]) * _DT
        tot += 4 * np.sum((np.abs(ht[_BAND]) ** 2) / _PSD[d][_BAND]) * _DF
    return float(np.sqrt(tot))


def mcq(m1, m2):
    """Chirp mass and mass ratio, with components SORTED so q <= 1.

    bilby's posterior columns are not sorted to m1 >= m2, so reading
    `posterior["mass_ratio"]` directly yields values above 1 and a meaningless
    90% width (measured 6.18 on ev0). dynesty_reference.py sorts for this reason.
    """
    m1, m2 = np.asarray(m1, float), np.asarray(m2, float)
    hi, lo = np.maximum(m1, m2), np.minimum(m1, m2)
    return (hi * lo) ** 0.6 / (hi + lo) ** 0.2, np.clip(lo / np.maximum(hi, 1e-9), 1e-6, 1.0)


def main():
    """Run the sweep.

    MUST stay inside a function behind a __main__ guard: run_dynesty uses
    bilby's multiprocessing pool (npool>1), and macOS spawns rather than forks,
    so every child re-imports this module. With the sweep at module level each
    child re-ran the whole sweep, and the resulting chaos surfaced as a dozen
    instant "dynesty FAILED:" lines with empty exception messages.
    """
    # ---- draw a spread of events: BBH masses over the SNR ladder ----
    rng = np.random.default_rng(777)
    SNR_LADDER = [12.0, 18.0, 25.0, 40.0]
    events = []
    for i in range(N_EVENTS):
        m1 = float(np.exp(rng.uniform(np.log(6.0), np.log(60.0))))
        m2 = float(np.exp(rng.uniform(np.log(5.0), np.log(m1))))
        events.append(dict(mass_1=max(m1, m2), mass_2=min(m1, m2),
                           luminosity_distance=500.0,
                           ra=float(rng.uniform(0, 2 * np.pi)),
                           dec=float(np.arcsin(rng.uniform(-1, 1))),
                           theta_jn=float(np.arccos(rng.uniform(-1, 1))),
                           psi=float(rng.uniform(0, np.pi)),
                           phase=float(rng.uniform(0, 2 * np.pi)),
                           geocent_time=0.0,
                           # REALISTIC aligned spins: the E-H sweep used a1=a2=0 while
                           # the training prior draws U[0,0.99]. Zero-spin is in-
                           # distribution but at the edge, and may flatter dynesty
                           # (which marginalizes spin). This re-run tests that caveat.
                           a1=float(rng.uniform(0.0, 0.9)),
                           a2=float(rng.uniform(0.0, 0.9)),
                           _snr_target=SNR_LADDER[i % len(SNR_LADDER)]))

    ck = torch.load(ROOT / "model/lean_npe_mf_scratch/best_model.pth", map_location="cpu",
                    weights_only=False)
    model = LeanNPE(psd_cond=True, psd_bands=16, encoder_type="coherent",
                    mass_param="mc_q", mf_tokens=True)
    model.load_state_dict(ck["model_state_dict"]); model.to(DEV).eval()
    psds = {det: get_default_psd(det) for det in DETS}

    rows = []
    for i, p in enumerate(events):
        t0 = time.time()
        tgt = p.pop("_snr_target")
        # Scale distance to hit the target network SNR (SNR ∝ 1/d_L exactly), staying
        # inside the NPE's training distance prior so the comparison is in-distribution
        # for both estimators. If the target is unreachable within that range the
        # achieved SNR is simply recorded and reported.
        sig = _projected(p, +1)
        p["luminosity_distance"] = float(np.clip(
            p["luminosity_distance"] * net_snr(sig) / tgt, 60.0, 2100.0))
        sig = _projected(p, +1)
        snr = net_snr(sig)
        raw = {det: _design_noise(det, 1000 + i * 10 + k) + sig[det] for k, det in enumerate(DETS)}
        prep = prepare_simulated(raw, seed=i)

        mc_t, q_t = mcq(p["mass_1"], p["mass_2"])

        # ---- NPE on this exact strain ----
        st = torch.from_numpy(prep.strain).float().unsqueeze(0).to(DEV)
        ab = torch.zeros(1, 3, 16, device=DEV)
        with torch.no_grad():
            s = model.sample_posterior(st, rank=0, n_samples=4000, asd_bands=ab)[0].cpu().numpy()
        n_mc, n_q = mcq(s[:, 0], s[:, 1])

        # ---- dynesty on the same raw strain ----
        try:
            res = run_dynesty(raw, psds, OUT, f"ev{i:02d}", center_gps=GPS_REF,
                              trigger_offset=0.0, nlive=NLIVE, npool=4)
            post = res.posterior
            # derive from components + sort (see mcq) rather than trusting bilby's
            # own chirp_mass/mass_ratio columns, whose convention is not m1 >= m2
            d_mc, d_q = mcq(np.asarray(post["mass_1"]), np.asarray(post["mass_2"]))
        except Exception as e:
            # print the full traceback, not str(e): the multiprocessing-spawn bug
            # above raised exceptions with EMPTY messages, so `str(e)[:120]` showed
            # "dynesty FAILED:" and nothing else, hiding the cause completely.
            import traceback
            print(f"[ev{i:02d}] dynesty FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue

        def w90(x): return float(np.percentile(x, 95) - np.percentile(x, 5))
        row = dict(i=i, snr=snr, mc_true=mc_t, q_true=q_t,
                   m1=p["mass_1"], m2=p["mass_2"], dL=p["luminosity_distance"],
                   npe_mc_med=float(np.median(n_mc)), dyn_mc_med=float(np.median(d_mc)),
                   npe_mc_w90=w90(n_mc), dyn_mc_w90=w90(d_mc),
                   npe_q_med=float(np.median(n_q)), dyn_q_med=float(np.median(d_q)),
                   npe_q_w90=w90(n_q), dyn_q_w90=w90(d_q),
                   npe_sig_logmc=float(np.std(np.log(n_mc))),
                   dyn_sig_logmc=float(np.std(np.log(d_mc))),
                   secs=time.time() - t0)
        rows.append(row)
        print(f"[ev{i:02d}] SNR{snr:5.1f} Mc_t{mc_t:6.2f} q_t{q_t:.2f} | "
              f"Mc w90 npe {row['npe_mc_w90']:7.2f} dyn {row['dyn_mc_w90']:6.3f} "
              f"({row['npe_mc_w90']/max(row['dyn_mc_w90'],1e-9):6.1f}x) | "
              f"q w90 npe {row['npe_q_w90']:.3f} dyn {row['dyn_q_w90']:.3f} "
              f"({row['npe_q_w90']/max(row['dyn_q_w90'],1e-9):4.1f}x) | {row['secs']:.0f}s", flush=True)
        json.dump(rows, open(OUT / "summary.json", "w"), indent=1)

    if rows:
        print("\n" + "=" * 96)
        mcr = np.array([r["npe_mc_w90"] / max(r["dyn_mc_w90"], 1e-9) for r in rows])
        qr = np.array([r["npe_q_w90"] / max(r["dyn_q_w90"], 1e-9) for r in rows])
        dq = np.array([r["dyn_q_w90"] for r in rows])
        print(f"n={len(rows)}  NPE/dynesty 90% width ratio — Mc: median {np.median(mcr):.1f}x "
              f"(range {mcr.min():.1f}-{mcr.max():.1f})")
        print(f"                                            q : median {np.median(qr):.1f}x "
              f"(range {qr.min():.1f}-{qr.max():.1f})")
        print(f"  dynesty's OWN q 90% width: median {np.median(dq):.3f} (prior support ~0.99)")
        print("  -> dynesty q broad (>0.4) => q deficit largely PHYSICAL; target Mc.")
        print("  -> dynesty q tight (<0.2) => q deficit is a MODEL deficit.")



if __name__ == "__main__":
    main()