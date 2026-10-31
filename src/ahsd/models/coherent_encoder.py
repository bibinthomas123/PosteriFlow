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

import math
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
    """Geometry-aware strain encoder: the norm-free conv/transformer base
    (`LeanStrainEncoder`) plus an explicit inter-detector geometry branch
    (power-weighted complex coherence |γ|/phase per band and GCC arrival-time
    delays) injected as conditioning tokens INTO the transformer — so the
    convolutional representation is fused with sky/distance geometry before
    pooling. Selected via `LeanNPE(encoder_type="coherent")`."""

    def __init__(self, geometry_bands: int = 16, geom_hidden: int = 128,
                 n_geom_tokens: int = 4, tau_max_ms: float = 30.0,
                 geom_dropout: float = 0.0, phase_residual: bool = False,
                 mf_tokens: bool = False, mf_shape_norm: bool = False,
                 mf_token_fusion: str = "add", mf_bank: str = "mf_bank.npz",
                 pol_features: bool = False, n_pol_tokens: int = 2, **kw):
        super().__init__(**kw)
        self.mf_bank = str(mf_bank)
        # Cross-detector polarization tokens (models/polarization_features.py):
        # pairwise amplitude ratio, relative phase and arrival-time difference
        # of the best template's complex SNR. The encoder's other features were
        # measured to carry no inclination/psi and almost no sky information
        # (analysis/inclination_info, probe R^2 ~ 0); these give sin(dec) R^2
        # 0.62 and resolve inclination jointly with sky. Needs the MF bank.
        self.use_pol_features = bool(pol_features)
        if self.use_pol_features and not mf_tokens:
            raise ValueError("pol_features requires mf_tokens (it uses the MF template bank)")
        self.geom_dropout = float(geom_dropout)
        self.phase_residual = bool(phase_residual)
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

        # band-center frequencies for delay removal in the residual-phase
        # feature; persistent=False so pre-residual checkpoints still load
        # strict — this buffer is derived, not learned.
        self.register_buffer("fband", torch.from_numpy(fb.astype(np.float32)),
                             persistent=False)

        self.pairs = [(i, j) for i in range(self.n_detectors)
                      for j in range(i + 1, self.n_detectors)]
        rel_dim = (self.n_detectors * self.K            # log-energy
                   + len(self.pairs) * self.K * 3       # |γ|, cosφ, sinφ per band
                   + len(self.pairs) * 2                # τ_ij, peak-sharpness
                   + len(self.pairs)                    # log-amp ratio
                   # residual polarization phase after delay removal (cos,sin),
                   # appended at the END of the vector so pre-residual
                   # geom_mlp weights map onto the same columns (init_from
                   # zero-pads the new tail)
                   + (len(self.pairs) * 2 if self.phase_residual else 0))
        self.geom_mlp = nn.Sequential(
            nn.Linear(rel_dim, geom_hidden), nn.GELU(),
            nn.Linear(geom_hidden, geom_hidden), nn.GELU(),
        )
        self.geom_to_tokens = nn.Linear(geom_hidden, self.n_geom_tokens * self.d_model)

        # ---- matched-filter mass tokens (docs/mf_token_encoder_design.md) ----
        # Per-detector complex overlaps against a fixed (Mc, q) template grid:
        # the |z| pattern over the grid is the profile likelihood over masses —
        # the chirp-phase statistic the conv stem cannot extract at low SNR and
        # the geometry features provably lack (in-cell q AUC: mf 0.989 vs geom
        # 0.532, analysis/attractor_basin/mf_probe.py). Template peak-time
        # differences double as the polarization-immune delay estimator.
        self.use_mf_tokens = bool(mf_tokens)
        # "add": legacy gated-additive merge into the geometry-token slots
        # (gtok = gtok + mf_gate*mftok) -- measured 2026-07-22
        # (analysis/attractor_basin/mf_embedding_tap_v2.py) to be the single
        # largest information-loss stage in the whole MF pathway (in-cell q
        # AUC 0.883 -> 0.738), bigger than the 1440->128 mf_mlp embedding
        # compression itself: a near-uninformative geometry signal (AUC
        # 0.532 alone) summed into a clean informative one (0.883) appears
        # to corrupt it. "concat": MF tokens are independent tokens
        # appended alongside (not summed into) the geometry tokens -- lets
        # the fusion transformer learn how to combine the two streams
        # instead of forcing a fixed linear (additive) combination before
        # attention ever sees them.
        assert mf_token_fusion in ("add", "concat")
        self.mf_token_fusion = mf_token_fusion
        # SHAPE-normalized overlap channel (analysis/posterior_bias_audit, E-K).
        # The |z| channel below is asinh(SNR) in ABSOLUTE units, which is
        # dominated by distance/amplitude; the mass-ratio information lives in
        # the RELATIVE shape of the profile across the template grid. Measured on
        # a 0.30%-Mc-spaced bank (n=344): a probe of the raw profile recovers q at
        # R^2 0.005-0.048, while the SAME profile divided by its own peak gives
        # 0.178 (0.219 at SNR>15). Off by default so every existing checkpoint
        # keeps its exact feature dimension and loads strict.
        self.mf_shape_norm = bool(mf_shape_norm)
        if self.use_mf_tokens:
            from pathlib import Path
            # mf_bank: a file name under models/assets/ or a path. Per-class
            # models need their own bank: the original covers Mc 12-45 only,
            # i.e. NO templates for BNS, NSBH or light BBH (2026-10-31).
            bank_path = Path(self.mf_bank)
            if not bank_path.is_absolute() and bank_path.parent == Path("."):
                bank_path = Path(__file__).parent / "assets" / bank_path
            bk = np.load(bank_path)
            assert int(bk["band_lo"]) == self.band_lo and int(bk["n_band"]) == self.Nf, \
                "mf_bank.npz band does not match encoder band — rebuild with build_mf_bank.py"
            # persistent=False: the 12 MB bank is a deterministic build artifact
            # loaded from assets, so checkpoints stay small and old ones load.
            self.register_buffer("mf_re", torch.from_numpy(bk["bank_re"]), persistent=False)
            self.register_buffer("mf_im", torch.from_numpy(bk["bank_im"]), persistent=False)
            K = self.mf_re.shape[1]
            self.mf_channels = 5 if self.mf_shape_norm else 4
            self.mf_mlp = nn.Sequential(
                nn.Linear(self.n_detectors * K * self.mf_channels, 256), nn.GELU(),
                nn.Linear(256, geom_hidden), nn.GELU(),
            )
            self.mf_to_tokens = nn.Linear(geom_hidden, self.n_geom_tokens * self.d_model)
            # zero-init gate: in "add" mode this exactly preserves an
            # --init_from checkpoint's pretrained function (a zero
            # CONCATENATED token would still soak up attention weight and
            # perturb the pooled values even at gate=0, but a zero ADDITIVE
            # contribution is a true no-op); in "concat" mode it just means
            # the MF tokens fade in from an all-zero state by gradient,
            # same as any other freshly-initialized token.
            self.mf_gate = nn.Parameter(torch.zeros(1))
            if self.mf_token_fusion == "concat":
                # Modality/type embedding (segment-embedding style): with
                # independent tokens the transformer must otherwise infer
                # "this token is geometry vs MF" purely from value content.
                # Zero-init so it starts as a true no-op and only carries a
                # signal once gradients decide it's useful.
                self.token_type_embed = nn.Parameter(torch.zeros(2, self.d_model))
        if self.use_pol_features:
            from ahsd.models.polarization_features import N_FEATURES
            self.n_pol_tokens = int(n_pol_tokens)
            self.pol_mlp = nn.Sequential(
                nn.LayerNorm(N_FEATURES), nn.Linear(N_FEATURES, 128), nn.GELU(),
                nn.Linear(128, 128), nn.GELU())
            self.pol_to_tokens = nn.Linear(128, self.n_pol_tokens * self.d_model)
            # zero-init marker telling the transformer which tokens these are
            self.pol_type_embed = nn.Parameter(torch.zeros(self.d_model))

    def _tau_features(self, Xr, Xi, Ei, Ej):
        """Phase-slope arrival-time delay + alignment strength for one pair.
        Xr,Xi: band-limited cross-spectrum Re/Im [B,Nf].

        Replaces the GCC-argmax lag: for narrow-band (heavy) signals any
        correlation-peak lag conflates the true delay with the inter-detector
        polarization phase (shift up to a half-cycle at the carrier, ~3.5-7 ms
        for ~30+30 systems) — the many-to-one measured-tau~+7ms cell behind
        the GW150914 (30,30) attractor (analysis/attractor_basin, addendum).
        The cross-spectrum phase is phi(f) = 2*pi*f*dt + dphi; increments
        between bins s apart, angle(X_{k+s} conj(X_k)) = 2*pi*s*df*dt, cancel
        the constant dphi exactly and stay wrap-free for |dt| <= 30 ms at
        s*df = 4 Hz, so a power-weighted mean of increment angles is a clean
        group-delay estimate. Strength = circular concentration of the
        increment phasors (fit quality; ~1 clean, ~0 noise)."""
        X = torch.complex(Xr, Xi)                              # [B,Nf]
        s = 16                                                 # s*df = 4 Hz
        P = X[:, s:] * X[:, :-s].conj()                        # [B,Nf-s]
        w = P.abs() + 1e-12
        ang = torch.atan2(P.imag, P.real)
        df = SR / T_LEN
        delay_raw = (w * ang).sum(-1) / (w.sum(-1) * 2 * np.pi * s * df)   # seconds
        # negated: the GCC-argmax this replaces peaks at -delay for
        # X = d_i conj(d_j), and the pretrained geom_mlp expects that sign.
        tau = (-delay_raw * SR / self.maxlag).clamp(-1.5, 1.5).unsqueeze(-1)
        strength = (P.sum(-1).abs() / w.sum(-1)).unsqueeze(-1)  # [B,1] in [0,1]
        res = None
        if self.phase_residual:
            # residual polarization phase AFTER delay removal: the constant
            # the slope estimator cancels is exactly what disambiguates the
            # quiet heavy tau~+7ms cell (basin snr24 stayed collapsed with
            # delay alone), so hand it to the net explicitly as (cos, sin) of
            # angle(sum_k X_k e^{-i 2 pi f_k dt}).
            th = 2 * np.pi * self.fband[None, :] * delay_raw[:, None]      # [B,Nf]
            c, sn = torch.cos(th), torch.sin(th)
            Rr = (Xr * c + Xi * sn).sum(-1)
            Ri = (Xi * c - Xr * sn).sum(-1)
            n = torch.sqrt(Rr ** 2 + Ri ** 2) + 1e-12
            res = torch.stack([Rr / n, Ri / n], dim=-1)                    # [B,2]
        return tau, strength, res

    def _geometry_rel(self, strain):
        """Compute the inter-detector relational geometry vector [B, rel_dim]:
        per-detector per-band log-energy, then per detector pair the
        power-weighted complex coherence (|γ|, cosφ, sinφ per band), the GCC
        arrival-time delay + peak sharpness, and the log-amplitude ratio."""
        B = strain.shape[0]
        fd = torch.fft.rfft(strain.float().contiguous(), norm="ortho", dim=-1)
        d = fd[..., self.band_lo: self.band_lo + self.Nf]      # [B,D,Nf]
        dr, di = d.real, d.imag
        P = dr ** 2 + di ** 2                                   # per-bin power
        amp = torch.sqrt(P + 1e-12)                            # |d| per bin
        E_band = (P @ self.Bsum.T) / self.bcount               # [B,D,K] mean power/band
        feats = [torch.log(E_band + 1e-8).reshape(B, -1)]
        residuals = []
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
            tau, peak, res = self._tau_features(Xr, Xi, Ei, Ej)
            feats += [tau, peak]                                # [B,1] each
            feats.append((torch.log(Ei + 1e-8) - torch.log(Ej + 1e-8)).unsqueeze(-1))
            if res is not None:
                residuals.append(res)
        feats += residuals                                      # appended at END
        return torch.cat(feats, dim=-1)                         # [B, rel_dim]

    def _mf_features(self, strain):
        """[B, D*K*4] matched-filter summary: per (detector, template) the
        peak overlap |z| (SNR units, asinh-compressed), cos/sin of the phase
        at peak, and the peak time relative to the window centre.

        The band-limited product spectrum is ifft'd over ONLY its Nf band
        bins: since z(t) is exactly band-limited, this is an alias-free
        complex baseband series at ~1 ms sampling (heterodyned by f_lo; the
        known phase ramp 2*pi*band_lo*n/Nf is added back for the phase
        feature) — full envelope fidelity with no noise folding and no
        [B,D,K,T] full-length ifft."""
        B, Dn, T = strain.shape
        fd = torch.fft.rfft(strain.float().contiguous(), dim=-1)
        d = fd[..., self.band_lo: self.band_lo + self.Nf]          # [B,D,Nf]
        H = torch.complex(self.mf_re, self.mf_im)                  # [D,K,Nf]
        K = H.shape[1]
        dt = (T / SR) / self.Nf                                    # ~0.996 ms
        ctr = self.Nf // 2
        w = min(int(1.6 / dt), ctr - 1)                            # merger window
        chunk = max(1, int(6e7 / max(B * Dn * self.Nf * 8, 1)))
        zw = []
        for k0 in range(0, K, chunk):
            X = d.unsqueeze(2) * H[:, k0:k0 + chunk].conj().unsqueeze(0)
            z = torch.fft.ifft(X, dim=-1)                          # [B,D,kc,Nf]
            zw.append(z[..., ctr - w: ctr + w])
        z = torch.cat(zw, dim=2)                                   # [B,D,K,2w]
        a = z.abs()
        idx = a.argmax(dim=-1, keepdim=True)                       # [B,D,K,1]
        # gather real/imag separately: complex gather is unsupported on MPS
        zpk = torch.complex(
            torch.take_along_dim(z.real, idx, dim=-1).squeeze(-1),
            torch.take_along_dim(z.imag, idx, dim=-1).squeeze(-1))  # [B,D,K]
        n_bin = (idx.squeeze(-1).float() + (ctr - w))              # bin in bb frame
        # |z| -> SNR units: ifft has 1/Nf; Var(sum D H*) = T/2 for unit-
        # variance whitened noise and unit-norm templates (unnormalized rfft)
        snr = zpk.abs() * (self.Nf / math.sqrt(T / 2.0))
        phase = torch.angle(zpk) + 2 * math.pi * self.band_lo * n_bin / self.Nf
        t_rel = (n_bin - ctr) * dt                                 # seconds
        chans = [torch.asinh(snr), torch.cos(phase), torch.sin(phase), t_rel]
        if self.mf_shape_norm:
            # Profile SHAPE: peak overlap divided by this detector's best
            # template. Scale-free, so it is invariant to distance/amplitude and
            # exposes the across-grid structure that carries mass ratio (E-K).
            chans.append(snr / snr.amax(dim=-1, keepdim=True).clamp_min(1e-9))
        feats = torch.stack(chans, dim=-1)                         # [B,D,K,4 or 5]
        return feats.reshape(B, -1)

    def forward(self, strain: torch.Tensor, asd_bands=None) -> torch.Tensor:
        """Whitened strain [B, 3, T] (+ optional asd_bands) -> context [B,
        context_dim]. Builds geometry tokens from `_geometry_rel` (and, when
        enabled, matched-filter mass tokens from `_mf_features`), injects them
        into the transformer alongside the conv tokens, and projects the pooled
        result to the context dimension."""
        strain = torch.nan_to_num(strain, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
        g = self.geom_mlp(self._geometry_rel(strain))           # [B, geom_hidden]
        gtok = self.geom_to_tokens(g).reshape(-1, self.n_geom_tokens, self.d_model)
        use_type_embed = self.use_mf_tokens and self.mf_token_fusion == "concat"
        if use_type_embed:
            # Added BEFORE geom_dropout so a dropped-out sample's geometry
            # tokens (incl. their type marker) still zero out completely --
            # no constant "geometry was dropped" leak into the transformer.
            gtok = gtok + self.token_type_embed[0]
        # Per-SAMPLE token dropout (all 4 tokens together, train only): the conv
        # path must stay self-sufficient so the geometry tokens cannot learn a
        # degenerate cell mapping (analysis/attractor_basin: the tau~+7ms basin
        # was carried causally by these tokens).
        if self.training and self.geom_dropout > 0.0:
            keep = (torch.rand(gtok.shape[0], 1, 1, device=gtok.device)
                    >= self.geom_dropout).to(gtok.dtype)
            gtok = gtok * keep
        if self.use_mf_tokens:
            # Computed AFTER geom_dropout, so the MF pathway survives when the
            # ambiguity-prone geometry tokens are dropped — it is the pathway
            # the model is SUPPOSED to rely on for masses.
            m = self.mf_mlp(self._mf_features(strain))
            mftok = self.mf_gate * self.mf_to_tokens(m).reshape(-1, self.n_geom_tokens, self.d_model)
            if self.mf_token_fusion == "concat":
                mftok = mftok + self.token_type_embed[1]
                gtok = torch.cat([gtok, mftok], dim=1)
            else:
                gtok = gtok + mftok
        if self.use_pol_features:
            # after geom_dropout: always present, like the MF pathway
            from ahsd.models.polarization_features import polarization_features
            pf = polarization_features(strain, self.mf_re, self.mf_im, self.band_lo)
            ptok = self.pol_to_tokens(self.pol_mlp(pf)).reshape(-1, self.n_pol_tokens, self.d_model)
            gtok = torch.cat([gtok, ptok + self.pol_type_embed], dim=1)
        feats, _ = self._compute_feats(strain, asd_bands, extra_tokens=gtok)
        return self.out_proj(feats)
