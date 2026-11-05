#!/usr/bin/env python3
"""Reduced waveform basis for the encoder's frequency-domain projection
branch (models/aligned_projection.py): the leading right-singular vectors of
whitened BBH waveforms over the encoder band (20-1024 Hz at 0.25 Hz).

Each waveform has its merger at t = 0 of the 4 s frame (what the projection
aligns the data to, using the bank's per-detector arrival times). Templates
start where they fit the window (3 s of inspiral at most), as the cropped data
in the window do. Both polarizations, whitened by the H1 and the V1 design
noise, are rows of the matrix; the basis comes from the eigenvectors of its
Gram matrix.

    python src/ahsd/models/build_aligned_basis.py --n 8000 --k 128 --out aligned_basis_bbh.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

SR, DUR, F_LO, F_HI = 4096, 4.0, 20.0, 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000, help="waveforms")
    ap.add_argument("--k", type=int, default=128, help="basis size")
    ap.add_argument("--max_tau", type=float, default=3.0, help="longest inspiral kept (s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="aligned_basis_bbh.npz")
    ap.add_argument("--gram_out", default=None, help="save this shard's Gram matrix and stop")
    ap.add_argument("--merge", nargs="+", default=None, help="Gram shards to sum into the basis")
    a = ap.parse_args()
    if a.merge:
        return finish(sum(np.load(f)["gram"] for f in a.merge), a.k, a.out,
                      int(np.load(a.merge[0])["band_lo"]), int(np.load(a.merge[0])["n_band"]))

    import bilby
    from ahsd.data.bilby_pipeline import get_default_psd
    from ahsd.inference.priors import TrainingPrior

    prior = TrainingPrior.for_class("BBH")
    wf = prior.waveform()
    freqs = np.fft.rfftfreq(int(SR * DUR), 1.0 / SR)
    band = (freqs >= F_LO) & (freqs < F_HI)
    band_lo, n_band = int(np.argmax(band)), int(band.sum())
    fb = freqs[band]
    asds = [np.sqrt(np.interp(fb, get_default_psd(d)["frequencies"], get_default_psd(d)["psd"]))
            for d in ("H1", "V1")]

    rng = np.random.default_rng(a.seed)
    (m_lo, m_hi), a_max = prior.m1_range, prior.a1_max
    gram = np.zeros((n_band, n_band), np.complex128)
    rows, done, failed = 0, 0, 0
    while done < a.n:
        m1, m2 = np.sort(rng.uniform(m_lo, m_hi, 2))[::-1]
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        # start frequency: at most max_tau of inspiral (leading order)
        f0 = 100.0 * (2.18 * (1.21 / mc) ** (5 / 3) / a.max_tau) ** (3 / 8)
        f0 = max(F_LO, f0)
        chi = rng.uniform(-a_max, a_max, 2)
        p = dict(mass_1=m1, mass_2=m2, luminosity_distance=400.0, theta_jn=rng.uniform(0, np.pi),
                 phase=0.0, psi=0.0, ra=0.0, dec=0.0, geocent_time=0.0,
                 a_1=abs(chi[0]), a_2=abs(chi[1]), tilt_1=0.0 if chi[0] >= 0 else np.pi,
                 tilt_2=0.0 if chi[1] >= 0 else np.pi, phi_12=0.0, phi_jl=0.0)
        gen = bilby.gw.WaveformGenerator(
            duration=DUR, sampling_frequency=SR, frequency_domain_source_model=wf["source_model"],
            parameter_conversion=wf["parameter_conversion"],
            waveform_arguments={"waveform_approximant": wf["approximant"], "reference_frequency": 50.0,
                                "minimum_frequency": f0})
        try:
            pol = gen.frequency_domain_strain(p)
        except Exception:
            failed += 1
            continue
        for key in ("plus", "cross"):
            h = np.asarray(pol[key])[band] * (fb >= f0)
            for asd in asds:
                w = h / asd
                nrm = np.linalg.norm(w)
                if nrm > 0:
                    w = w / nrm
                    gram += np.outer(w.conj(), w)
                    rows += 1
        done += 1
        if done % 1000 == 0:
            print(f"{done}/{a.n} waveforms ({failed} failed)", flush=True)

    if a.gram_out:
        np.savez(a.gram_out, gram=gram, band_lo=band_lo, n_band=n_band)
        print(f"{rows} rows -> {a.gram_out}")
        return
    finish(gram, a.k, a.out, band_lo, n_band)


def finish(gram, k, out_name, band_lo, n_band):
    # right-singular vectors of the row matrix = eigenvectors of its Gram matrix
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    v = evecs[:, :k].T                                       # [K, F]: coefficient c = d @ v.conj().T
    captured = float(evals[:k].sum() / evals.sum())
    print(f"first {k} vectors capture {captured:.4%} of the training waveforms' power")
    out = Path(out_name)
    if not out.is_absolute() and out.parent == Path("."):
        out = Path(__file__).parent / "assets" / out
    np.savez(out, v_re=v.real.astype(np.float32), v_im=v.imag.astype(np.float32),
             band_lo=band_lo, n_band=n_band, captured=captured, eigvals=evals[: 4 * k])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
