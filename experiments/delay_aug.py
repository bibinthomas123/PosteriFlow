"""Differential inter-detector delay augmentation (sky remapping).

Root cause being attacked (analysis/attractor_basin REPORT + addendum):
RemixDataset's augmentations never vary inter-detector delay — time shift is
a common-mode roll; noise swap, distance rescale and re-coloring leave tau
untouched. So the geometry branch sees exactly ONE tau per stored event, AND
the training set carries a spurious measured-tau <-> mass-ratio correlation
(the polarization phase folds into any correlation-based lag for narrow-band
heavy signals, piling heavy q>0.8 events at tau in [+5,+9] ms at 3x the
mirror-bin rate). Both together let the coherent encoder learn the degenerate
{tau~+7 ms, heavy} -> (30,30) cell mapping that captures GW150914 — and
basin-cell oversampling REINFORCES it, because the oversampled cell itself is
q-skewed. This augmentation randomizes each signal's sky geometry, which (a)
gives the geometry branch fresh tau every epoch, and (b) makes measured tau
independent of the masses, so NLL stops paying for the shortcut and the only
way to keep resolving q is the conv/chirp pathway.

A stored per-detector projected signal is remapped to a NEW sky position with
labels kept honest:

  delays  — exact. Each detector's series is shifted by
            dt_new(det) - dt_old(det) via an FFT phase ramp (subsample
            precision). The generator used one fixed epoch
            (parameter_sampler.GPS_REF) for every event, so dt_old is
            computable from the ra/dec labels alone — verified against the
            stored components to <0.06 ms (phase-slope lag on 300 light
            events, scratchpad preflight 2026-07-18).
  antenna — approximate. The new sky is CHOSEN (lowest antenna-response
            mismatch among `n_candidates` random grid points) and psi is
            relabeled with the closed-form optimal polarization rotation
            (F+ + iFx transforms as e^{-2i psi}, so the best common rotation
            over detectors is a phase fit). The signal's amplitude content
            then stays near-consistent with the new (ra, dec, psi) labels;
            the residual mismatch is mild label noise on the sky-amplitude
            coupling, on augmented samples only. Detectors are weighted by
            stored signal energy so the match spends its freedom where the
            signal actually is.

Runtime is pure numpy — no bilby/lal in DataLoader workers. Detector
geometry (vertex positions, detector tensors, GMST at GPS_REF) is extracted
from bilby once and cached beside the memmaps as delay_aug_geom.npz; the
numpy delay/antenna math was verified to match bilby to machine precision
(same preflight).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# MUST match ahsd.data.parameter_sampler.GPS_REF (the fixed window epoch every
# stored event was projected at).
GPS_REF = 1369224018.0
C_SI = 299792458.0
SR = 4096
DETS = ("H1", "L1", "V1")
IDX_DIST, IDX_RA, IDX_DEC, IDX_PSI = 2, 3, 4, 6   # into remix_data.PARAM_NAMES


def _load_geom(cache_dir: str) -> dict:
    """Detector vertices [3,3] m, detector tensors [3,3,3], GMST at GPS_REF.
    Cached beside the memmaps so workers (and later runs) never import bilby."""
    f = Path(cache_dir) / "delay_aug_geom.npz"
    if f.exists():
        z = np.load(f)
        return {"vertex": z["vertex"], "dtensor": z["dtensor"], "gmst": float(z["gmst"])}
    import bilby
    from bilby.gw.utils import greenwich_mean_sidereal_time
    ifos = [bilby.gw.detector.get_empty_interferometer(d) for d in DETS]
    geom = {"vertex": np.stack([i.geometry.vertex for i in ifos]).astype(np.float64),
            "dtensor": np.stack([i.geometry.detector_tensor for i in ifos]).astype(np.float64),
            "gmst": float(greenwich_mean_sidereal_time(GPS_REF)) % (2 * np.pi)}
    np.savez(f, **geom)
    return geom


class SkyRemapper:
    """Per-signal sky remap: exact delay re-roll + antenna-matched relabel."""

    def __init__(self, cache_dir: str, grid_n: int = 16384, n_candidates: int = 64,
                 tau_window: float = 0.35e-3):
        g = _load_geom(cache_dir)
        self.vertex, self.dtensor, self.gmst = g["vertex"], g["dtensor"], g["gmst"]
        self.n_candidates = int(n_candidates)
        self.tau_window = float(tau_window)
        # quasi-uniform sky grid (Fibonacci sphere) at the fixed epoch
        i = np.arange(grid_n)
        self.grid_dec = np.arcsin(1.0 - 2.0 * (i + 0.5) / grid_n)
        self.grid_ra = (i * np.pi * (3.0 - np.sqrt(5.0))) % (2 * np.pi)
        self.grid_delay = self._delays(self.grid_ra, self.grid_dec)    # [N,3] s
        self.grid_f0 = self._antenna0(self.grid_ra, self.grid_dec)     # [N,3] c128
        # tau_HL-sorted view for STRATIFIED candidate draws: matching antenna
        # response over the whole sphere picks skies near the original (or its
        # pattern-degenerate images) and drags the new delay back toward the
        # old one (measured corr(tau_old, tau_new) = +0.81) — defeating the
        # purpose. Instead ONE random grid point fixes the target tau_HL
        # (isotropic by construction, independent of the old sky) and the
        # antenna match only chooses along that tau's ring.
        tau_hl = self.grid_delay[:, 0] - self.grid_delay[:, 1]
        self._tau_order = np.argsort(tau_hl)
        self._tau_sorted = tau_hl[self._tau_order]
        self._rfreq = np.fft.rfftfreq(16384, 1.0 / SR)                 # T_LEN bins

    # ---- vectorized geometry (bilby conventions, verified to machine eps) ----
    def _nhat(self, ra, dec):
        """[K,3] unit vectors toward the source in Earth-fixed frame."""
        gha = self.gmst - np.asarray(ra)
        dec = np.asarray(dec)
        return np.stack([np.cos(dec) * np.cos(gha),
                         -np.cos(dec) * np.sin(gha),
                         np.sin(dec)], axis=-1)

    def _delays(self, ra, dec):
        """time_delay_from_geocenter per detector, [K,3] seconds."""
        return -self._nhat(ra, dec) @ self.vertex.T / C_SI

    def _antenna0(self, ra, dec):
        """F+(psi=0) + i Fx(psi=0) per detector, [K,3] complex.
        F(psi) = e^{-2i psi} F(0), so one complex number per (sky, det) spans
        every polarization angle."""
        theta, phi = np.pi / 2.0 - np.asarray(dec), np.asarray(ra) - self.gmst
        u = np.stack([np.cos(phi) * np.cos(theta),
                      np.cos(theta) * np.sin(phi), -np.sin(theta)], axis=-1)
        v = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
        # psi=0: m0=-v, n0=-u  ->  e+ = vv - uu,  ex = uv + vu
        ep = v[:, :, None] * v[:, None, :] - u[:, :, None] * u[:, None, :]
        ex = u[:, :, None] * v[:, None, :] + v[:, :, None] * u[:, None, :]
        fp = np.einsum("dab,kab->kd", self.dtensor, ep)
        fx = np.einsum("dab,kab->kd", self.dtensor, ex)
        return fp + 1j * fx

    def remap(self, sig: np.ndarray, par: np.ndarray, rng) -> np.ndarray:
        """Remap one signal [3,T] to a new sky; relabels par[ra,dec,psi] in
        place, returns the delay-shifted signal."""
        ra, dec, psi = float(par[IDX_RA]), float(par[IDX_DEC]), float(par[IDX_PSI])
        d_old = self._delays([ra], [dec])[0]                          # [3]
        g = np.exp(-2j * psi) * self._antenna0([ra], [dec])[0]        # [3]
        w = (sig.astype(np.float64) ** 2).sum(-1)                     # energy/det

        # isotropic target tau_HL, then candidates from that stratum only
        tau_star = self._tau_sorted[rng.integers(len(self._tau_sorted))]
        lo = int(np.searchsorted(self._tau_sorted, tau_star - self.tau_window))
        hi = int(np.searchsorted(self._tau_sorted, tau_star + self.tau_window))
        cand = self._tau_order[rng.integers(lo, max(hi, lo + 1),
                                            size=min(self.n_candidates, max(hi - lo, 1)))]
        f = self.grid_f0[cand]                                        # [K,3]
        # best polarization rotation AND overall scale per candidate:
        #   min_{alpha>0, psi'} sum_d w_d |g_d - alpha e^{-2i psi'} f_d|^2
        # -> e^{-2i psi'} = S/|S|, alpha = |S|/A  (weighted phase Procrustes).
        # The rotation becomes the new psi label; the scale becomes an EXACT
        # distance relabel (amplitude ~ F/d_L, so response scaled by alpha ==
        # distance divided by alpha — same leading-order GR relation the
        # distance-rescale augmentation relies on). Candidates are ranked by
        # the SHAPE-only residual B - |S|^2/A: the honest-label noise left
        # over after psi and distance have absorbed what they can.
        S = (w[None, :] * np.conj(f) * g[None, :]).sum(-1)            # [K]
        A = (w[None, :] * np.abs(f) ** 2).sum(-1)                     # [K]
        B = (w * np.abs(g) ** 2).sum()
        k = int(np.argmin(B - np.abs(S) ** 2 / np.maximum(A, 1e-30)))
        c = int(cand[k])
        psi_new = float((-np.angle(S[k]) / 2.0) % np.pi)
        alpha = float(np.abs(S[k]) / max(A[k], 1e-30))
        # d' = d/alpha, alpha clamped so the relabel stays inside the prior
        # box (and within a sane range even for near-degenerate matches)
        d_old_lab = float(par[IDX_DIST])
        alpha = float(np.clip(alpha, max(0.5, d_old_lab / 2100.0),
                              min(2.0, d_old_lab / 45.0)))

        # exact per-detector delay change; s'(t) = s(t - delta) moves each
        # detector's arrival to the new sky's delay (subsample FFT shift;
        # whitening is diagonal in frequency, so it commutes with the shift)
        delta = self.grid_delay[c] - d_old                            # [3] s
        fr = (self._rfreq if sig.shape[-1] == 16384
              else np.fft.rfftfreq(sig.shape[-1], 1.0 / SR))
        Sf = np.fft.rfft(sig, axis=-1)
        sig = np.fft.irfft(Sf * np.exp(-2j * np.pi * fr[None, :] * delta[:, None]),
                           n=sig.shape[-1], axis=-1).astype(np.float32)

        par[IDX_RA] = self.grid_ra[c]
        par[IDX_DEC] = self.grid_dec[c]
        par[IDX_PSI] = psi_new
        par[IDX_DIST] = d_old_lab / alpha
        return sig
