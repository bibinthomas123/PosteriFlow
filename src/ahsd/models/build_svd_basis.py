"""
Build the reduced-order (SVD) basis for the coherent encoder [stage 1].

Per-detector whitened data factorize as d_d(f) = P_d · Y(f;Mc,q,χ) · e^{2πifτ_d},
so a basis that spans {whitened Y(f) × time-shift} represents every detector's
data up to the complex prefactor P_d. We stack whitened (2,2)-dominant plus-
polarisation waveforms over the BBH mass/spin range, each multiplied by a random
arrival-time phase ramp, and take the complex SVD.

Output: svd_basis.npz with Bre,Bim [Nb, Nf] (complex basis rows), f_lo/f_hi/df.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

SR, DUR = 4096, 4.0
F_LO, F_HI = 20.0, 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_waveforms", type=int, default=800)
    ap.add_argument("--n_basis", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT / "src/ahsd/models/assets/svd_basis.npz"))
    ap.add_argument("--approximant", default="IMRPhenomXP")
    args = ap.parse_args()

    import bilby
    from ahsd.data.bilby_pipeline import get_default_psd

    df = 1.0 / DUR
    freqs = np.fft.rfftfreq(int(SR * DUR), 1.0 / SR)          # [0 .. 2048], df=0.25
    band = (freqs >= F_LO) & (freqs < F_HI)
    fb = freqs[band]
    Nf = int(band.sum())
    # design ASD on the band (average H1/L1 -> a representative whitening)
    asd = {}
    for d in ("H1", "L1", "V1"):
        p = get_default_psd(d)
        asd[d] = np.sqrt(np.interp(fb, p["frequencies"], p["psd"]))
    asd_ref = np.mean([asd["H1"], asd["L1"]], axis=0)

    wfg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=SR,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={"waveform_approximant": args.approximant,
                            "reference_frequency": 50.0, "minimum_frequency": F_LO})
    rng = np.random.default_rng(0)
    W = np.zeros((args.n_waveforms, Nf), dtype=np.complex128)
    n = 0
    while n < args.n_waveforms:
        m1 = float(np.exp(rng.uniform(np.log(5), np.log(100))))
        m2 = float(np.exp(rng.uniform(np.log(5), np.log(m1))))
        pars = dict(mass_1=m1, mass_2=m2, luminosity_distance=500.0,
                    theta_jn=float(np.arccos(rng.uniform(-1, 1))), psi=0.0, phase=float(rng.uniform(0, 2*np.pi)),
                    a_1=float(rng.uniform(0, 0.9)), a_2=float(rng.uniform(0, 0.9)),
                    tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0,
                    ra=0.0, dec=0.0, geocent_time=0.0)
        pol = wfg.frequency_domain_strain(pars)
        h = pol["plus"][band] / asd_ref                       # whitened plus polarisation
        if not np.all(np.isfinite(h)) or np.linalg.norm(h) < 1e-30:
            continue
        tau = rng.uniform(-0.015, 0.015)                      # random arrival-time shift
        h = h * np.exp(-2j * np.pi * fb * tau)
        W[n] = h / (np.linalg.norm(h) + 1e-30)
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{args.n_waveforms}")

    # complex SVD; basis rows = top-Nb right-singular vectors (conjugated for projection)
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    Nb = min(args.n_basis, Vh.shape[0])
    B = Vh[:Nb]                                               # [Nb, Nf], rows orthonormal
    captured = float((S[:Nb] ** 2).sum() / (S ** 2).sum())
    print(f"Nf={Nf} Nb={Nb} captured signal power={captured:.5f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, Bre=B.real.astype(np.float32), Bim=B.imag.astype(np.float32),
             f_lo=F_LO, f_hi=F_HI, df=df, band_lo_idx=int(np.argmax(band)),
             nf=Nf, captured_power=captured)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
