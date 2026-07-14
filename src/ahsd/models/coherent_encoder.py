"""
CoherentEncoder (Path B, v2) — the PROVEN conv encoder (LeanStrainEncoder,
which conditions well) made GEOMETRY-AWARE. The geometry is injected as
conditioning tokens INTO the transformer (cross-fused before pooling), not
concatenated at the end — so the convolutional representation is geometry-aware.

Geometry features (per reviewer feedback), all from the band-limited unitary FD
d_d(f) = rfft(x_d)/√T over [20,1024] Hz, K=16 log bands:

  1. per-detector per-band log mean-power        E_d^b = <|d_d|^2>_b
  2. per-PAIR POWER-WEIGHTED complex coherence    γ_ij^b = Σ_b d_i d_j* / Σ_b |d_i||d_j|
     (loud inspiral bins dominate; one noisy bin no longer counts as much), given
     to the net as MAGNITUDE + PHASE  (|γ|, cosφ, sinφ)  — not Re/Im.
  3. explicit inter-detector ARRIVAL-TIME DELAY   τ_ij = argmax_τ |Σ_f d_i d_j* e^{-i2πfτ}|
     via the GCC peak within the physical light-travel window, + a peak-sharpness
     coherence-strength scalar. (The single most important geometric quantity.)
  4. per-pair overall log-amplitude ratio          r_ij = log(ΣE_i) - log(ΣE_j)

Caveat (agreed): these are informative summaries on raw band-limited data, NOT
matched-filtered sufficient statistics, so γ degrades gracefully as SNR drops.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn

from ahsd.models.lean_npe import LeanStrainEncoder

# torch.fft.rfft emits a benign internal "output was resized" deprecation on
# some CPU builds regardless of input contiguity/dtype; it does not affect the
# result. Silence just that message.
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r".*output with one or more elements was resized.*")

SR, T_LEN = 4096, 16384
F_LO, F_HI = 20.0, 1024.0


class CoherentEncoder(LeanStrainEncoder):
    def __init__(self, geometry_bands: int = 16, geom_hidden: int = 128,
                 n_geom_tokens: int = 4, tau_max_ms: float = 30.0, **kw):
        super().__init__(**kw)
        self.K = int(geometry_bands)
        self.n_geom_tokens = int(n_geom_tokens)
        self.d_model = self.detector_embed.embedding_dim
        self.n_rfft = T_LEN // 2 + 1

        freqs = np.fft.rfftfreq(T_LEN, 1.0 / SR)
        band = (freqs >= F_LO) & (freqs < F_HI)
        self.band_lo = int(np.argmax(band))
        self.Nf = int(band.sum())
        fb = freqs[band]
        edges = np.geomspace(F_LO, F_HI, self.K + 1)
        Bsum = torch.zeros(self.K, self.Nf)                    # 0/1 band membership
        for k in range(self.K):
            Bsum[k] = torch.from_numpy(((fb >= edges[k]) & (fb < edges[k + 1])).astype(np.float32))
        self.register_buffer("Bsum", Bsum)
        self.register_buffer("bcount", Bsum.sum(1).clamp_min(1.0))   # bins per band
        # normalized lag axis for the GCC search window
        self.maxlag = int(tau_max_ms * 1e-3 * SR)
        self.register_buffer("lags_norm",
                             torch.arange(-self.maxlag, self.maxlag + 1).float() / self.maxlag)

        self.pairs = [(i, j) for i in range(self.n_detectors)
                      for j in range(i + 1, self.n_detectors)]
        rel_dim = (self.n_detectors * self.K            # log-energy
                   + len(self.pairs) * self.K * 3       # |γ|, cosφ, sinφ per band
                   + len(self.pairs) * 2                # τ_ij, peak-sharpness
                   + len(self.pairs))                   # log-amp ratio
        self.geom_mlp = nn.Sequential(
            nn.Linear(rel_dim, geom_hidden), nn.GELU(),
            nn.Linear(geom_hidden, geom_hidden), nn.GELU(),
        )
        self.geom_to_tokens = nn.Linear(geom_hidden, self.n_geom_tokens * self.d_model)

    def _tau_features(self, Xr, Xi, Ei, Ej):
        """GCC arrival-time delay + peak sharpness for one pair. Xr,Xi: band-
        limited cross-spectrum Re/Im [B,Nf]; Ei,Ej: total band-energy [B]."""
        B = Xr.shape[0]
        Xfull = torch.zeros(B, self.n_rfft, dtype=torch.complex64, device=Xr.device)
        Xfull[:, self.band_lo: self.band_lo + self.Nf] = torch.complex(Xr, Xi)
        cc = torch.fft.irfft(Xfull, n=T_LEN, dim=-1)           # [B, T] cross-correlation
        win = torch.cat([cc[:, -self.maxlag:], cc[:, :self.maxlag + 1]], dim=1)  # τ∈[-max,+max]
        a = win.abs()
        k = a.argmax(-1)
        tau = self.lags_norm[k].unsqueeze(-1)                  # [B,1] in [-1,1]
        peak = (a.max(-1).values / (a.mean(-1) + 1e-8)).unsqueeze(-1)  # peak-to-avg sharpness
        return tau, peak

    def _geometry_rel(self, strain):
        B = strain.shape[0]
        fd = torch.fft.rfft(strain.float().contiguous(), norm="ortho", dim=-1)
        d = fd[..., self.band_lo: self.band_lo + self.Nf]      # [B,D,Nf]
        dr, di = d.real, d.imag
        P = dr ** 2 + di ** 2                                   # per-bin power
        amp = torch.sqrt(P + 1e-12)                            # |d| per bin
        E_band = (P @ self.Bsum.T) / self.bcount               # [B,D,K] mean power/band
        feats = [torch.log(E_band + 1e-8).reshape(B, -1)]
        for (i, j) in self.pairs:
            Xr = (dr[:, i] * dr[:, j] + di[:, i] * di[:, j])   # Re(d_i d_j*) [B,Nf]
            Xi = (di[:, i] * dr[:, j] - dr[:, i] * di[:, j])   # Im(d_i d_j*)
            # POWER-WEIGHTED coherence: numerator=Σ d_i d_j*, denom=Σ|d_i||d_j|
            num_r = Xr @ self.Bsum.T                            # [B,K]
            num_i = Xi @ self.Bsum.T
            den = (amp[:, i] * amp[:, j]) @ self.Bsum.T + 1e-8
            gr, gi = num_r / den, num_i / den
            gmag = torch.sqrt(gr ** 2 + gi ** 2) + 1e-8
            feats += [gmag, gr / gmag, gi / gmag]               # |γ|, cosφ, sinφ  [B,K]
            Ei, Ej = P[:, i].sum(-1), P[:, j].sum(-1)
            tau, peak = self._tau_features(Xr, Xi, Ei, Ej)
            feats += [tau, peak]                                # [B,1] each
            feats.append((torch.log(Ei + 1e-8) - torch.log(Ej + 1e-8)).unsqueeze(-1))
        return torch.cat(feats, dim=-1)                         # [B, rel_dim]

    def forward(self, strain: torch.Tensor, asd_bands=None) -> torch.Tensor:
        strain = torch.nan_to_num(strain, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
        g = self.geom_mlp(self._geometry_rel(strain))           # [B, geom_hidden]
        gtok = self.geom_to_tokens(g).reshape(-1, self.n_geom_tokens, self.d_model)
        feats, _ = self._compute_feats(strain, asd_bands, extra_tokens=gtok)
        return self.out_proj(feats)
