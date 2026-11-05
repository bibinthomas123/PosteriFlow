"""
Build a dense chirp-mass template bank for the encoder's chirp-mass features
(models/mc_features.py).

Window templates are self-contained whitened chirps: generated over 4 s from
the frequency where the chirp is 3.6 s from merger (so it neither wraps nor
carries a cut edge that would move under the matched filter's time search),
20-1024 Hz band. Context templates (multi-rate models) are built like the data:
a long waveform whitened by the design ASD, the 28 s before the window taken
with to_lowrate (rfft at 512 Hz), merger at the window centre. Each template is
unit-norm per detector, so overlaps with unit-variance whitened data are in
SNR units.

A 0.2% chirp-mass grid with an interpolated peak measures BNS chirp mass to
~0.15% from the window alone at window SNR >= 8 (scratch probe 2026-10-31);
the current network's posterior is ~8% wide.

    python src/ahsd/models/build_mc_bank.py --cls bns --context
    python src/ahsd/models/build_mc_bank.py --cls nsbh --context
    python src/ahsd/models/build_mc_bank.py --cls bbh
    python src/ahsd/models/build_mc_bank.py --cls bbh_spin
    python src/ahsd/models/build_mc_bank.py --cls nsbh_spin --context
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

SR, T_LEN = 4096, 16384
F_LO, F_HI = 20.0, 1024.0
DETS = ("H1", "L1", "V1")
LO_SEC, LO_RATE = 28, 512

# per class: chirp-mass range, log spacing, mass-ratio grid, approximant
CLASSES = {
    "bns": dict(mc=(0.86, 2.2), dlog=0.002, q=(0.85,), approx="IMRPhenomD"),
    "nsbh": dict(mc=(1.35, 11.5), dlog=0.003, q=(0.4, 0.15, 0.04), approx="IMRPhenomD"),
    "bbh": dict(mc=(4.2, 90.0), dlog=0.005, q=(0.9, 0.5, 0.25), approx="IMRPhenomXP"),
    # spin-aware BBH bank: at fixed chirp mass, mass ratio and aligned spin
    # shift the phase (the Mc-q-chi degeneracy), so a zero-spin grid biases
    # the chirp mass of spinning systems; 4 q x 4 aligned spins per chirp mass
    "bbh_spin": dict(mc=(4.2, 90.0), dlog=0.01, q=(0.9, 0.6, 0.35, 0.15),
                     chi=(0.0, 0.3, 0.6, 0.85), approx="IMRPhenomXP"),
    # finer, signed spin-aware BBH bank: the 1% chirp-mass grid and spins >= 0
    # of bbh_spin capped the chirp-mass precision at high SNR (bank estimate
    # ~2%; matched width vs dynesty 1.3x at SNR 8-20 but 4.5x at SNR 35+ in
    # the 8-epoch A/B, 2026-11-01), and v4 data have anti-aligned spins
    "bbh_spin_fine": dict(mc=(4.2, 90.0), dlog=0.005, q=(0.9, 0.6, 0.35, 0.15),
                          chi=(-0.6, -0.3, 0.0, 0.3, 0.6, 0.85), approx="IMRPhenomXP"),
    # dense BBH bank (2026-11-04): 24 (q, chi) slots left a chirp-mass error
    # floor of 1.5-2% at any SNR (error x SNR 0.21 -> 1.69 from SNR 10 to 87;
    # events far from a slot 2x worse): the best slot compensates an off-grid
    # q / spin with a shifted chirp mass. 10 q x 12 equal aligned spins.
    "bbh_spin_dense": dict(mc=(4.2, 90.0), dlog=0.005,
                           q=(0.95, 0.8, 0.65, 0.52, 0.42, 0.33, 0.25, 0.18, 0.12, 0.08),
                           chi=(-0.75, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9),
                           approx="IMRPhenomXP"),
    # signed NSBH bank: anti-aligned black-hole spins too (GW200115's light-BH
    # branch has a1 ~ -0.7; the v4 data draw a1 in [-0.9, 0.9])
    "nsbh_spin_signed": dict(mc=(1.35, 11.5), dlog=0.01, q=(0.4, 0.15, 0.06, 0.025),
                             chi=(-0.8, -0.5, -0.2, 0.0, 0.3, 0.6, 0.85), chi_primary_only=True,
                             approx="IMRPhenomD"),
    # spin-aware NSBH bank: black-hole spin only (neutron-star spins are
    # drawn from [0, 0.05]); NSBH mass ratios span 0.01-0.8
    "nsbh_spin": dict(mc=(1.35, 11.5), dlog=0.01, q=(0.4, 0.15, 0.06, 0.025),
                      chi=(0.0, 0.3, 0.6, 0.85), chi_primary_only=True, approx="IMRPhenomD"),
}


def masses_from_mc_q(mc, q):
    M = mc * (1.0 + q) ** 1.2 / q ** 0.6
    return M / (1.0 + q), M * q / (1.0 + q)


def masses_on_grid(mc, q, m_min=0.9):
    """Component masses for (mc, q), raising q toward 1 where the lighter
    mass would fall below m_min, so every chirp mass keeps every grid slot
    (the feature module reshapes the bank to [n_mc, n_q])."""
    m1, m2 = masses_from_mc_q(mc, q)
    if m2 >= m_min:
        return m1, m2, q
    lo, hi = q, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if masses_from_mc_q(mc, mid)[1] >= m_min else (mid, hi)
    m1, m2 = masses_from_mc_q(mc, hi)
    return m1, m2, hi


def spins(chi, cfg):
    """Aligned spins as bilby magnitudes and tilts (chi < 0: anti-aligned,
    tilt pi); the secondary carries chi too unless chi_primary_only."""
    a2 = 0.0 if cfg.get("chi_primary_only") else chi
    return dict(a_1=abs(chi), a_2=abs(a2), tilt_1=np.pi if chi < 0 else 0.0,
                tilt_2=np.pi if a2 < 0 else 0.0)


def chirp_time(mc, f):
    return 2.18 * (1.21 / mc) ** (5.0 / 3.0) * (100.0 / f) ** (8.0 / 3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", choices=list(CLASSES), required=True)
    ap.add_argument("--context", action="store_true",
                    help="also build context-stream templates (multi-rate models)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = CLASSES[a.cls]
    import bilby
    logging.getLogger("bilby").setLevel(logging.ERROR)
    from ahsd.data.bilby_pipeline import get_default_psd, to_lowrate

    psd = {d: get_default_psd(d) for d in DETS}
    fw = np.fft.rfftfreq(T_LEN, 1.0 / SR)
    band = (fw >= F_LO) & (fw < F_HI)
    band_lo, n_band = int(np.argmax(band)), int(band.sum())
    gens, asds = {}, {}

    def long_whitened(m1, m2, mc, chi=0.0):
        """Whitened long series per detector, merger 4 s before the end."""
        need = chirp_time(mc, F_LO) + LO_SEC + 12.0
        D = float(2 ** int(np.ceil(np.log2(max(need, 64.0)))))
        if D not in gens:
            gens[D] = bilby.gw.WaveformGenerator(
                duration=D, sampling_frequency=SR,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                waveform_arguments={"waveform_approximant": cfg["approx"],
                                    "reference_frequency": 50.0, "minimum_frequency": F_LO})
            fl = np.fft.rfftfreq(int(D * SR), 1.0 / SR)
            asds[D] = (fl, {d: np.sqrt(np.interp(fl, psd[d]["frequencies"], psd[d]["psd"],
                                                 left=np.inf, right=np.inf)) for d in DETS})
        hp = gens[D].frequency_domain_strain(dict(
            mass_1=m1, mass_2=m2, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0,
            **spins(chi, cfg), phi_12=0.0, phi_jl=0.0,
            ra=0.0, dec=0.0, geocent_time=D - 4.0))["plus"]
        fl, asd = asds[D]
        n = int(D * SR)
        return D, {d: np.fft.irfft(np.where(fl >= F_LO, hp / asd[d], 0), n=n) for d in DETS}

    win_gens = {}

    def window_template(m1, m2, mc, chi=0.0):
        """[D, n_band] self-contained chirp ending at merger (circular)."""
        fmin = round(max(F_LO, 100.0 * (2.18 * (1.21 / mc) ** (5 / 3) / 3.6) ** (3 / 8)), 1)
        if fmin not in win_gens:
            win_gens[fmin] = bilby.gw.WaveformGenerator(
                duration=T_LEN / SR, sampling_frequency=SR,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                waveform_arguments={"waveform_approximant": cfg["approx"],
                                    "reference_frequency": 50.0, "minimum_frequency": fmin})
        hp = win_gens[fmin].frequency_domain_strain(dict(
            mass_1=m1, mass_2=m2, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0,
            **spins(chi, cfg), phi_12=0.0, phi_jl=0.0,
            ra=0.0, dec=0.0, geocent_time=0.0))["plus"]
        rows = []
        for d in DETS:
            asd_w = np.sqrt(np.interp(fw, psd[d]["frequencies"], psd[d]["psd"], left=np.inf, right=np.inf))
            h = (hp / asd_w)[band_lo:band_lo + n_band]
            rows.append(h / np.sqrt((np.abs(h) ** 2).sum()))
        return np.stack(rows)

    mcs = np.exp(np.arange(np.log(cfg["mc"][0]), np.log(cfg["mc"][1]), cfg["dlog"]))
    chis = cfg.get("chi", (0.0,))
    win, ctx, labels = [], [], []
    for i, mc in enumerate(mcs):
        for q, chi in [(q, c) for q in cfg["q"] for c in chis]:
            m1, m2, q = masses_on_grid(float(mc), float(q))
            win.append(window_template(m1, m2, mc, chi))
            if a.context:
                D, wl = long_whitened(m1, m2, mc, chi)
                i0 = int((D - 6.0) * SR)              # merger at the window centre
                c_row = []
                for d in DETS:
                    hc = np.fft.rfft(to_lowrate(wl[d], i0, SR, LO_SEC, LO_RATE))
                    c_row.append(hc / np.sqrt((np.abs(hc) ** 2).sum() + 1e-30))
                ctx.append(np.stack(c_row))
            labels.append((float(mc), float(q), float(chi)))
        if i % 50 == 0:
            print(f"  {i}/{len(mcs)} chirp masses", flush=True)
    win = np.stack(win).transpose(1, 0, 2).astype(np.complex64)        # [D, K, Nf]
    out = Path(a.out or Path(__file__).parent / f"assets/mc_bank_{a.cls}{'_ctx' if a.context else ''}.npz")
    extra = {}
    if a.context:
        c = np.stack(ctx).transpose(1, 0, 2).astype(np.complex64)      # [D, K, Fc]
        extra = dict(ctx_re=c.real, ctx_im=c.imag, ctx_len=LO_SEC * LO_RATE)
    np.savez(out, win_re=win.real, win_im=win.imag, mc_q=np.array(labels, np.float32),
             mc_grid=mcs.astype(np.float32), dlog=cfg["dlog"], n_q=len(cfg["q"]) * len(chis),
             band_lo=band_lo, n_band=n_band, approximant=cfg["approx"], **extra)
    print(f"wrote {out}: {win.shape[1]} templates ({len(mcs)} chirp masses x {len(cfg['q'])} q x {len(chis)} spins), "
          f"{out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
