"""
LeanNPE: minimal amortized neural posterior estimator for overlapping CBC signals.

Every design choice here reverses a *measured* failure of OverlapNeuralPE
(see analysis/context_conditioning_test.json: context std across events
0.0000, shuffle-NLL delta 0.0000, probe R^2 < 0 for all 11 parameters):

1. NO per-sample normalization on the amplitude path. The strain is already
   whitened (unit noise floor), so absolute signal amplitude IS the
   distance/SNR cue. The old pipeline stacked per-detector std-normalization,
   InstanceNorm x3, and per-parameter LayerNorm, which together produced a
   context that was constant across events. Here the convs see
   asinh-compressed raw strain (amplitude-monotone, bounded), and explicit
   per-window log-energy features carry amplitude losslessly.

2. ONE flat context vector with simple attention pooling. The 11-query
   per-parameter readout was measured to collapse (attention entropy pinned
   at 1.0, post-attention outputs cos-sim 0.9997); the diversity losses that
   fought it only sculpted statistics. Flat conditioning is the
   literature-standard (DINGO) and was never shown to be the actual
   bottleneck.

3. Signal-rank conditioning. The old loss trained the SAME context to
   predict every overlapping signal's parameters with no indication of which
   signal was queried, making the optimal solution a mixture, not a
   posterior. Here the flow context includes an embedding of the queried
   signal's rank.

4. Pure NLL objective. No aux/diversity/calibration/Jacobian terms.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ahsd.models.flows import NSFPosteriorFlow

PARAM_NAMES = [
    "mass_1", "mass_2", "luminosity_distance",
    "ra", "dec", "theta_jn", "psi", "phase",
    "geocent_time", "a1", "a2",
]


class ParamScaler:
    """Fixed, invertible map physical params <-> [-1, 1] (log-space for
    masses/distance, linear for angles/time/spins). Deterministic: no fitted
    statistics to drift between train/eval.

    `mass_param` selects the coordinates the FLOW models for the mass pair. The
    physical interface is unchanged in both cases -- callers always pass and
    receive (mass_1, mass_2, ...) -- only the flow's internal target moves:

    "m1m2" (default, legacy)
        Flow models (log m1, log m2). The convention mass_1 >= mass_2 then makes
        the support a TRIANGLE: a diagonal hard edge that nothing enforces,
        because m1 and m2 are scaled independently. Measured consequence
        (analysis/.../mechanism/m22): on GW150914 27.7% of posterior samples
        land at m1 < m2, where the training density is exactly zero, and the
        dominant mode centres ON the diagonal (m1~m2~30, straddling it 59/41).

    "mc_q"
        Flow models (log Mc, q). The support becomes an axis-aligned BOX, which
        is exactly the flow's own output domain, and q <= 1 makes
        mass_1 >= mass_2 hold BY CONSTRUCTION -- no ordering fold at sampling.
        Mc is also the best-measured mass combination, so it gets its own
        coordinate instead of being an emergent function of two.

    "mc_eta" is deliberately NOT offered: eta = q/(1+q)^2 has deta/dq = 0 at
    q = 1, so p(eta) carries an inverse-sqrt singularity at the equal-mass edge
    -- it concentrates density at precisely the boundary that is already the
    measured failure, and squeezes a broad q posterior (dynesty's GW150914 spans
    q 0.55-1.0) into the top ~9% of eta's range.
    """

    # (lo, hi, log?) covering the dataset generation priors with margin
    RANGES = {
        "mass_1":              (1.0, 105.0, True),
        "mass_2":              (1.0, 105.0, True),
        # Mc in [m/2^0.2 for m=1 .. 105] = [0.87, 91.4]; margin either side.
        "chirp_mass":          (0.8, 95.0, True),
        # q = m2/m1 in (1/105, 1] = (0.0095, 1]; linear, +1 <-> equal mass.
        "mass_ratio":          (0.008, 1.0, False),
        "luminosity_distance": (40.0, 2200.0, True),
        "ra":                  (0.0, 2 * math.pi, False),
        "dec":                 (-math.pi / 2, math.pi / 2, False),
        "theta_jn":            (0.0, math.pi, False),
        "psi":                 (0.0, math.pi, False),
        "phase":               (0.0, 2 * math.pi, False),
        "geocent_time":        (-1.6, 1.6, False),
        "a1":                  (0.0, 1.0, False),
        "a2":                  (0.0, 1.0, False),
    }

    # Parameters whose range equals one full period: wrapping in normalized
    # space is EXACT (ra, phase: 2*pi; psi: pi). Flow samples that land past a
    # bound belong at the other end of the circle, not clamped at the edge.
    CIRCULAR = ("ra", "phase", "psi")

    # Legacy luminosity-distance support. Labels outside it are CLAMPED by
    # normalize(), so a 15 Mpc BNS trains as "40 Mpc" and a 2750 Mpc event
    # (GW170729) can never be returned (measured 2026-10-30: BNS < 40 Mpc SBC
    # rank 0.00). New models pass a wider dist_range; LeanNPE stores it in the
    # state dict so every load site picks it up without new constructor args.
    LEGACY_DIST_RANGE = (40.0, 2200.0)

    def __init__(self, param_names: List[str] = PARAM_NAMES, premerger: bool = False,
                 mass_param: str = "m1m2", dist_range: Optional[tuple] = None):
        """premerger=True widens geocent_time to cover early-warning events
        whose merger lies up to ~3 s past the window end (t_c up to +5 s).
        Must match between training and inference (stored in checkpoint args).

        mass_param selects the flow's mass coordinates ("m1m2" or "mc_q"); it
        must also match between training and inference.

        dist_range (lo, hi) Mpc overrides the luminosity-distance support
        (default LEGACY_DIST_RANGE).
        """
        self.dist_range = tuple(float(v) for v in (dist_range or self.LEGACY_DIST_RANGE))
        if mass_param not in ("m1m2", "mc_q"):
            raise ValueError(f"mass_param must be 'm1m2' or 'mc_q', got {mass_param!r}")
        self.param_names = param_names
        self.premerger = premerger
        self.mass_param = mass_param
        # names of the coordinates the FLOW sees (differs from param_names only
        # in the first two slots, and only for mc_q)
        if mass_param == "mc_q":
            self.internal_names = ["chirp_mass", "mass_ratio"] + list(param_names[2:])
        else:
            self.internal_names = list(param_names)
        lo, hi, lg = [], [], []
        for p in self.internal_names:
            l, h, g = self.RANGES[p]
            if p == "geocent_time" and premerger:
                l, h = -1.6, 5.2
            if p == "luminosity_distance":
                l, h = self.dist_range
            lo.append(math.log(l) if g else l)
            hi.append(math.log(h) if g else h)
            lg.append(g)
        self.lo = torch.tensor(lo, dtype=torch.float32)
        self.hi = torch.tensor(hi, dtype=torch.float32)
        self.log_mask = torch.tensor(lg, dtype=torch.bool)
        self.circ_mask = torch.tensor([p in self.CIRCULAR for p in self.internal_names],
                                      dtype=torch.bool)

    def to(self, device):
        """Move the scaler's range/mask buffers to `device` (call after moving
        the model so normalize/denormalize run on the same device as the flow)."""
        self.lo = self.lo.to(device)
        self.hi = self.hi.to(device)
        self.log_mask = self.log_mask.to(device)
        self.circ_mask = self.circ_mask.to(device)
        return self

    def wrap(self, y: torch.Tensor) -> torch.Tensor:
        """Map raw flow output into [-1, 1]: modular wrap for circular
        parameters (exact), clamp for bounded ones."""
        wrapped = torch.remainder(y + 1.0, 2.0) - 1.0
        return torch.where(self.circ_mask, wrapped, y.clamp(-1.0, 1.0))

    # NOTE: both maps index with `...` and concatenate on the LAST axis. Callers
    # pass [batch, 11] (nll) and [batch, n_samples, 11] (sample_posterior)
    # alike, and the surrounding scale/log steps are pure broadcasting, so these
    # must stay rank-agnostic too. Indexing [:, 0] silently slices the wrong
    # axis for 3-D input and breaks the m1>=m2 guarantee.
    def _to_internal(self, x: torch.Tensor) -> torch.Tensor:
        """physical (m1, m2, ...) -> the flow's coordinates."""
        if self.mass_param == "m1m2":
            return x
        m1 = x[..., 0].clamp_min(1e-6)
        m2 = x[..., 1].clamp_min(1e-6)
        mc = (m1 * m2).pow(0.6) / (m1 + m2).pow(0.2)
        q = (m2 / m1).clamp(1e-4, 1.0)
        return torch.cat([mc.unsqueeze(-1), q.unsqueeze(-1), x[..., 2:]], dim=-1)

    def _from_internal(self, z: torch.Tensor) -> torch.Tensor:
        """the flow's coordinates -> physical (m1, m2, ...).

        Mc = M * eta^(3/5) with eta = q/(1+q)^2, so M = Mc (1+q)^(6/5) q^(-3/5);
        then m1 = M/(1+q), m2 = qM/(1+q). q <= 1 gives m1 >= m2 exactly, which is
        why no max/min fold is needed downstream.
        """
        if self.mass_param == "m1m2":
            return z
        mc = z[..., 0].clamp_min(1e-6)
        q = z[..., 1].clamp(1e-4, 1.0)
        M = mc * (1.0 + q).pow(1.2) / q.pow(0.6)
        m1 = M / (1.0 + q)
        m2 = M * q / (1.0 + q)
        return torch.cat([m1.unsqueeze(-1), m2.unsqueeze(-1), z[..., 2:]], dim=-1)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """physical [batch, 11] -> [-1, 1]"""
        x = self._to_internal(x)
        x = torch.where(self.log_mask, torch.log(x.clamp_min(1e-6)), x)
        return (2 * (x - self.lo) / (self.hi - self.lo) - 1).clamp(-1.0, 1.0)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> physical"""
        x = (y.clamp(-1.0, 1.0) + 1) / 2 * (self.hi - self.lo) + self.lo
        x = torch.where(self.log_mask, torch.exp(x), x)
        return self._from_internal(x)


class SinusoidalPositions(nn.Module):
    """Fixed (non-learned) sinusoidal positional encodings, precomputed to
    `max_len` and sliced per call — added to the conv-token sequence so the
    transformer can use token order along the strain."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, n: int) -> torch.Tensor:
        """Return the first `n` positional-encoding rows, shape [n, d_model]."""
        return self.pe[:n]


class LeanStrainEncoder(nn.Module):
    """Whitened 3-detector strain -> single context vector.

    Norm-free conv stem on asinh(strain) (already ~unit-scale inputs), small
    pre-norm transformer for cross-detector fusion, learned-query attention
    pooling, plus an explicit log-energy branch computed from RAW strain so
    absolute amplitude (the distance cue) survives no matter what the
    transformer's internal LayerNorms do.
    """

    def __init__(self, n_detectors: int = 3, d_model: int = 192,
                 n_layers: int = 3, n_heads: int = 6, n_pool_queries: int = 8,
                 n_energy_windows: int = 16, context_dim: int = 256,
                 dropout: float = 0.05, psd_bands: int = 0):
        super().__init__()
        self.n_detectors = n_detectors
        self.n_energy_windows = n_energy_windows
        self.context_dim = context_dim
        # PSD-conditioning branch (psd_bands>0): per-detector log-ASD band
        # summary of the whitening filter relative to design. Whitened
        # amplitude = distance x detector-sensitivity; without this the model
        # cannot separate a loud/near source from a loud/sensitive detector and
        # reads quieter (less-sensitive, e.g. O1/O2) events as farther. Feeding
        # the sensitivity lets it convert whitened amplitude -> physical SNR.
        self.psd_bands = psd_bands

        # Norm-free stem (shared across detectors). 16384 -> 61 tokens.
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=8), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=16, stride=4), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4), nn.GELU(),
            nn.Conv1d(128, d_model, kernel_size=4, stride=2), nn.GELU(),
        )
        self.detector_embed = nn.Embedding(n_detectors, d_model)
        self.pos = SinusoidalPositions(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.pool_queries = nn.Parameter(torch.randn(n_pool_queries, d_model) / math.sqrt(d_model))
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Explicit amplitude branch: per-detector, per-window log mean-square
        # of the RAW whitened strain. For unit-noise data this is ~log(1 + SNR
        # density in the window): a direct, un-normalizable distance/SNR/
        # time-of-loudest-window cue.
        self.energy_mlp = nn.Sequential(
            nn.Linear(n_detectors * n_energy_windows, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
        )

        noise_dim = 0
        if psd_bands > 0:
            self.noise_mlp = nn.Sequential(
                nn.Linear(n_detectors * psd_bands, 64), nn.GELU(),
                nn.Linear(64, 32), nn.GELU(),
            )
            noise_dim = 32

        self.out_proj = nn.Sequential(
            nn.Linear(n_pool_queries * d_model + 64 + noise_dim, 512), nn.GELU(),
            nn.Linear(512, context_dim),
        )

    def _compute_feats(self, strain: torch.Tensor,
                       asd_bands: Optional[torch.Tensor] = None,
                       extra_tokens: Optional[torch.Tensor] = None):
        """Concatenated pre-projection features [B, feat_dim] + the nan-cleaned
        strain (so subclasses can add branches without recomputing cleanup).
        extra_tokens [B, n, d_model] are prepended to the transformer input so a
        subclass can make the fused representation geometry-aware before pooling."""
        B, D, T = strain.shape
        strain = torch.nan_to_num(strain, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)

        # Energy branch from RAW strain (before any compression)
        w = self.n_energy_windows
        win = strain[:, :, : (T // w) * w].reshape(B, D, w, -1)
        log_energy = torch.log((win ** 2).mean(dim=-1) + 1e-8)  # [B, D, w]
        energy_feat = self.energy_mlp(log_energy.reshape(B, -1))  # [B, 64]

        # Token branch on asinh-compressed strain (amplitude-monotone, bounded)
        x = torch.asinh(strain).reshape(B * D, 1, T)
        tokens = self.stem(x).transpose(1, 2)  # [B*D, L, d_model]
        L = tokens.shape[1]
        tokens = tokens + self.pos(L).unsqueeze(0)
        tokens = tokens.reshape(B, D, L, -1)
        det_ids = torch.arange(D, device=strain.device)
        tokens = tokens + self.detector_embed(det_ids)[None, :, None, :]
        tokens = tokens.reshape(B, D * L, -1)

        # geometry-aware conditioning: prepend extra tokens so self-attention
        # fuses them with the conv tokens BEFORE pooling.
        if extra_tokens is not None:
            tokens = torch.cat([extra_tokens, tokens], dim=1)

        tokens = self.fusion(tokens)

        q = self.pool_queries.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, tokens, tokens)  # [B, nq, d_model]

        feats = [pooled.reshape(B, -1), energy_feat]
        if self.psd_bands > 0:
            # Default to zeros = design-sensitivity reference when not supplied.
            # Correct for design-whitened data (simulated/Gaussian domains);
            # real-noise paths MUST pass measured asd_bands to get distance right.
            if asd_bands is None:
                asd_bands = strain.new_zeros(B, self.n_detectors, self.psd_bands)
            feats.append(self.noise_mlp(asd_bands.reshape(B, -1)))
        return torch.cat(feats, dim=1), strain

    def forward(self, strain: torch.Tensor,
                asd_bands: Optional[torch.Tensor] = None) -> torch.Tensor:
        """strain: [B, n_det, T] raw whitened -> context [B, context_dim].
        asd_bands: [B, n_det, psd_bands] log-ASD-vs-design summary of the
        whitening filter (required iff psd_bands>0; ~0 for design-whitened
        data, negative where the detector is less sensitive than design)."""
        feats, _ = self._compute_feats(strain, asd_bands)
        return self.out_proj(feats)


class LeanNPE(nn.Module):
    """Encoder + rank embedding + NSF flow. Pure NLL training."""

    def __init__(self, param_names: List[str] = PARAM_NAMES,
                 context_dim: int = 256, rank_dim: int = 32, max_signals: int = 5,
                 flow_layers: int = 10, flow_hidden: int = 256, flow_bins: int = 16,
                 encoder_kwargs: Optional[dict] = None, premerger: bool = False,
                 psd_cond: bool = False, psd_bands: int = 16,
                 encoder_type: str = "conv", mass_param: str = "m1m2",
                 geom_dropout: float = 0.0, geom_phase_residual: bool = False,
                 mf_tokens: bool = False, mf_shape_norm: bool = False,
                 mf_token_fusion: str = "add", q_embed_dim: int = 0,
                 dist_range: Optional[tuple] = None):
        super().__init__()
        self.param_names = param_names
        self.premerger = premerger
        self.max_signals = max_signals
        self.context_dim = context_dim
        self.encoder_type = encoder_type
        # Dedicated, explicitly-supervised mass-ratio conditioning channel
        # (analysis/attractor_basin/REPORT.md, 2026-07-22 flow-conditioning
        # addendum): flow_context_sensitivity.py showed the flow's per-layer
        # dense context->hidden map is equally (in)sensitive to the
        # q-informative direction in the 256-d context as to a random
        # direction of the same size -- the flow isn't preferentially
        # routing through q, even though it's linearly present. Rather than
        # asking the flow to DISCOVER that direction unaided among 256
        # largely-redundant dims, hand it a small (q_embed_dim-wide)
        # embedding trained with its own auxiliary regression loss
        # (q_pred_head) and concatenate it onto the flow's own context
        # input. 0 = disabled (default, exact prior behavior/checkpoint
        # compatibility).
        self.q_embed_dim = int(q_embed_dim)
        if self.q_embed_dim > 0:
            self.q_embed_head = nn.Sequential(
                nn.Linear(context_dim, 64), nn.GELU(),
                nn.Linear(64, self.q_embed_dim))
            self.q_pred_head = nn.Linear(self.q_embed_dim, 2)
        # the coherent encoder always ingests asd_bands (phase-coherent geometry
        # front-end); treat it as psd-conditioned regardless of the flag.
        self.psd_cond = psd_cond or (encoder_type == "coherent")
        # mass_param only changes the flow's target coordinates; weights, shapes
        # and every physical-space interface are identical between the two.
        self.mass_param = mass_param
        self.scaler = ParamScaler(param_names, premerger=premerger,
                                  mass_param=mass_param, dist_range=dist_range)
        # Persisted so a checkpoint carries its own distance support: any load
        # site (all of them build LeanNPE without dist_range) gets the trained
        # range from load_state_dict. Checkpoints that predate this buffer get
        # the legacy range via _load_from_state_dict below.
        self.register_buffer("dist_range", torch.tensor(self.scaler.dist_range,
                                                        dtype=torch.float32))

        enc_kw = dict(encoder_kwargs or {})
        if encoder_type == "coherent":
            from ahsd.models.coherent_encoder import CoherentEncoder
            self.encoder = CoherentEncoder(context_dim=context_dim, psd_bands=psd_bands,
                                           geom_dropout=geom_dropout,
                                           phase_residual=geom_phase_residual,
                                           mf_tokens=mf_tokens,
                                           mf_shape_norm=mf_shape_norm,
                                           mf_token_fusion=mf_token_fusion, **enc_kw)
        else:
            if psd_cond:
                enc_kw["psd_bands"] = psd_bands
            self.encoder = LeanStrainEncoder(context_dim=context_dim, **enc_kw)
        self.rank_embed = nn.Embedding(max_signals, rank_dim)

        self.flow = NSFPosteriorFlow(
            features=len(param_names),
            context_features=context_dim + self.q_embed_dim + rank_dim,
            hidden_features=flow_hidden,
            num_layers=flow_layers,
            num_bins=flow_bins,
            tail_bound=5.0,
            dropout=0.0,
            temperature_scale=1.0,
        )

    def _full_context(self, context: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
        """Concatenate the encoder context [B, context_dim] (+ the dedicated
        q-embedding, when enabled) with the queried signal's rank embedding
        [B, rank_dim] to form the flow's conditioning vector."""
        parts = [context]
        if self.q_embed_dim > 0:
            parts.append(self.q_embed_head(context))
        parts.append(self.rank_embed(rank))
        return torch.cat(parts, dim=1)

    def encode(self, strain: torch.Tensor,
               asd_bands: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Whitened strain [B, 3, T] -> context vector [B, context_dim].
        `asd_bands` [B, n_det, psd_bands] is the sensitivity summary required
        when psd_cond=True (0 for design-whitened data); ignored otherwise."""
        return self.encoder(strain, asd_bands) if self.psd_cond else self.encoder(strain)

    def nll(self, strain: torch.Tensor, params_phys: torch.Tensor,
            rank: torch.Tensor, context: Optional[torch.Tensor] = None,
            asd_bands: Optional[torch.Tensor] = None) -> torch.Tensor:
        """params_phys: [B, 11] physical units; rank: [B] long. Returns [B].
        asd_bands: [B, n_det, psd_bands] required when psd_cond=True."""
        if context is None:
            context = self.encode(strain, asd_bands)
        ctx = self._full_context(context, rank)
        y = self.scaler.normalize(params_phys)
        log_sigma = torch.zeros_like(y)
        return self.flow.compute_psd_aware_nll(y, ctx, log_sigma)

    @torch.no_grad()
    def sample_posterior(self, strain: torch.Tensor, rank: int = 0,
                         n_samples: int = 256,
                         asd_bands: Optional[torch.Tensor] = None) -> torch.Tensor:
        """strain: [B, 3, T] -> samples [B, n_samples, 11] in PHYSICAL units.
        asd_bands: [B, n_det, psd_bands] required when psd_cond=True."""
        context = self.encode(strain, asd_bands)
        B = context.shape[0]
        r = torch.full((B,), rank, dtype=torch.long, device=context.device)
        ctx = self._full_context(context, r)
        ctx_rep = ctx.unsqueeze(1).expand(B, n_samples, ctx.shape[1]).reshape(B * n_samples, -1)
        z = torch.randn(B * n_samples, len(self.param_names), device=context.device)
        y, _ = self.flow.inverse(z, ctx_rep)
        y = self.scaler.wrap(y).reshape(B, n_samples, -1)
        return self.scaler.denormalize(y)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Pre-dist_range checkpoints lack the buffer: they were trained on the
        legacy support, so fill it in (strict loads keep working). Then rebuild
        the scaler so normalize/denormalize use the checkpoint's range."""
        key = prefix + "dist_range"
        if key not in state_dict:
            state_dict[key] = torch.tensor(ParamScaler.LEGACY_DIST_RANGE, dtype=torch.float32)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        rng = tuple(float(v) for v in self.dist_range.tolist())
        if rng != self.scaler.dist_range:
            self.scaler = ParamScaler(self.param_names, premerger=self.premerger,
                                      mass_param=self.mass_param,
                                      dist_range=rng).to(self.dist_range.device)

    def to(self, *args, **kwargs):
        """Move the model to a device/dtype and keep the (non-Parameter)
        ParamScaler buffers in sync, so normalize/denormalize stay on-device."""
        out = super().to(*args, **kwargs)
        dev = next(self.parameters()).device
        self.scaler.to(dev)
        return out
