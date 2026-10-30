"""
Build the matched-filter template bank for the MF-token encoder branch
(docs/mf_token_encoder_design.md) [stage 1 of 2; stage 2 is
CoherentEncoder(mf_tokens=True), which consumes assets/mf_bank.npz].

Grid: log-spaced chirp mass x mass ratio, zero aligned spin, (2,2)-dominant
plus polarization, whitened per detector by the DESIGN ASD and unit-L2-
normalized over the [20, 1024) Hz analysis band — so the per-detector
overlap of unit-variance whitened data with a template is in SNR units after
the fixed scale in the encoder. Offline go/no-go for this statistic:
analysis/attractor_basin/mf_probe.py (2026-07-19: in-cell q AUC 0.989 vs
0.532 for the geometry features and 0.821 for the frozen pslope2 context).

Usage: conda run -n ahsd python src/ahsd/models/build_mf_bank.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

SR, DUR = 4096, 4.0
T_LEN = int(SR * DUR)
F_LO, F_HI = 20.0, 1024.0
DETS = ("H1", "L1", "V1")


def masses_from_mc_q(mc, q):
    M = mc * (1.0 + q) ** 1.2 / q ** 0.6
    return M / (1.0 + q), M * q / (1.0 + q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_mc", type=int, default=20)
    ap.add_argument("--mc_range", type=float, nargs=2, default=(12.0, 45.0))
    ap.add_argument("--q_grid", type=float, nargs="+",
                    default=(1.0, 0.85, 0.7, 0.55, 0.4, 0.25))
    ap.add_argument("--approximant", default="IMRPhenomXP")
    ap.add_argument("--m_min", type=float, default=1.0,
                    help="skip cells whose lighter component falls below this (Msun)")
    ap.add_argument("--m_max", type=float, default=105.0,
                    help="skip cells whose heavier component exceeds this (Msun)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "assets/mf_bank.npz"))
    args = ap.parse_args()

    import bilby
    from ahsd.data.bilby_pipeline import get_default_psd

    freqs = np.fft.rfftfreq(T_LEN, 1.0 / SR)
    band = (freqs >= F_LO) & (freqs < F_HI)
    band_lo = int(np.argmax(band))
    asd = {}
    for det in DETS:
        p = get_default_psd(det)
        asd[det] = np.sqrt(np.interp(freqs, np.asarray(p["frequencies"], float),
                                     np.asarray(p["psd"], float),
                                     left=np.inf, right=np.inf))
    wfg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=SR,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={"waveform_approximant": args.approximant,
                            "reference_frequency": 50.0, "minimum_frequency": F_LO})

    mcs = np.geomspace(*args.mc_range, args.n_mc)
    bank = []
    labels = []
    n_skip = 0
    for mc in mcs:
        for q in args.q_grid:
            m1, m2 = masses_from_mc_q(float(mc), float(q))
            # Skip (Mc, q) cells that fall outside the physical component-mass
            # support the dataset actually samples (ParamScaler.RANGES caps both
            # components at [1, 105] Msun). Harmless for the original coarse
            # grid, which sat well inside; REQUIRED for a prior-matched grid,
            # where low Mc combined with small q asks for sub-solar companions
            # (e.g. Mc=1.0, q=0.07 -> m2 = 0.36 Msun) that no waveform model
            # should be asked to produce and that no training event can match.
            if m2 < args.m_min or m1 > args.m_max:
                n_skip += 1
                continue
            hp = wfg.frequency_domain_strain(
                dict(mass_1=m1, mass_2=m2, luminosity_distance=500.0,
                     theta_jn=0.0, psi=0.0, phase=0.0, a_1=0.0, a_2=0.0,
                     tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0,
                     ra=0.0, dec=0.0, geocent_time=0.0))["plus"]
            row = []
            for det in DETS:
                hw = (hp / asd[det])[band]
                row.append((hw / np.sqrt((np.abs(hw) ** 2).sum())).astype(np.complex64))
            bank.append(np.stack(row))            # [D, Nfb]
            labels.append((float(mc), float(q)))
    bank = np.stack(bank).transpose(1, 0, 2)      # [D, K, Nfb]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, bank_re=bank.real.astype(np.float32),
             bank_im=bank.imag.astype(np.float32),
             mc_q=np.array(labels, dtype=np.float32),
             band_lo=band_lo, n_band=int(band.sum()),
             f_lo=F_LO, f_hi=F_HI, t_len=T_LEN, sr=SR,
             approximant=args.approximant)
    print(f"wrote {out}: bank {bank.shape} "
          f"({bank.nbytes / 1e6:.1f} MB), band_lo={band_lo}, Nfb={int(band.sum())}, "
          f"skipped {n_skip} unphysical (Mc, q) cells")


if __name__ == "__main__":
    main()
