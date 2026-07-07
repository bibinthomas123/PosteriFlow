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
    statistics to drift between train/eval."""

    # (lo, hi, log?) covering the dataset generation priors with margin
    RANGES = {
        "mass_1":              (1.0, 105.0, True),
        "mass_2":              (1.0, 105.0, True),
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

    def __init__(self, param_names: List[str] = PARAM_NAMES, premerger: bool = False):
        """premerger=True widens geocent_time to cover early-warning events
        whose merger lies up to ~3 s past the window end (t_c up to +5 s).
        Must match between training and inference (stored in checkpoint args)."""
        self.param_names = param_names
        self.premerger = premerger
        lo, hi, lg = [], [], []
        for p in param_names:
            l, h, g = self.RANGES[p]
            if p == "geocent_time" and premerger:
                l, h = -1.6, 5.2
            lo.append(math.log(l) if g else l)
            hi.append(math.log(h) if g else h)
            lg.append(g)
        self.lo = torch.tensor(lo, dtype=torch.float32)
        self.hi = torch.tensor(hi, dtype=torch.float32)
        self.log_mask = torch.tensor(lg, dtype=torch.bool)

    def to(self, device):
        self.lo = self.lo.to(device)
        self.hi = self.hi.to(device)
        self.log_mask = self.log_mask.to(device)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """physical [batch, 11] -> [-1, 1]"""
        x = torch.where(self.log_mask, torch.log(x.clamp_min(1e-6)), x)
        return (2 * (x - self.lo) / (self.hi - self.lo) - 1).clamp(-1.0, 1.0)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> physical"""
        x = (y.clamp(-1.0, 1.0) + 1) / 2 * (self.hi - self.lo) + self.lo
        return torch.where(self.log_mask, torch.exp(x), x)


class SinusoidalPositions(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, n: int) -> torch.Tensor:
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
                 dropout: float = 0.05):
        super().__init__()
        self.n_detectors = n_detectors
        self.n_energy_windows = n_energy_windows
        self.context_dim = context_dim

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

        self.out_proj = nn.Sequential(
            nn.Linear(n_pool_queries * d_model + 64, 512), nn.GELU(),
            nn.Linear(512, context_dim),
        )

    def forward(self, strain: torch.Tensor) -> torch.Tensor:
        """strain: [B, n_det, T] raw whitened -> context [B, context_dim]"""
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

        tokens = self.fusion(tokens)

        q = self.pool_queries.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, tokens, tokens)  # [B, nq, d_model]

        context = self.out_proj(torch.cat([pooled.reshape(B, -1), energy_feat], dim=1))
        return context


class LeanNPE(nn.Module):
    """Encoder + rank embedding + NSF flow. Pure NLL training."""

    def __init__(self, param_names: List[str] = PARAM_NAMES,
                 context_dim: int = 256, rank_dim: int = 32, max_signals: int = 5,
                 flow_layers: int = 8, flow_hidden: int = 192, flow_bins: int = 16,
                 encoder_kwargs: Optional[dict] = None, premerger: bool = False):
        super().__init__()
        self.param_names = param_names
        self.max_signals = max_signals
        self.scaler = ParamScaler(param_names, premerger=premerger)

        self.encoder = LeanStrainEncoder(context_dim=context_dim, **(encoder_kwargs or {}))
        self.rank_embed = nn.Embedding(max_signals, rank_dim)

        self.flow = NSFPosteriorFlow(
            features=len(param_names),
            context_features=context_dim + rank_dim,
            hidden_features=flow_hidden,
            num_layers=flow_layers,
            num_bins=flow_bins,
            tail_bound=3.0,
            dropout=0.0,
            temperature_scale=1.0,
            use_masked_context=False,
        )
        # Learnable temperature was a symptom-patch in the old model; pin it.
        self.flow.temperature.requires_grad_(False)

    def _full_context(self, context: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
        return torch.cat([context, self.rank_embed(rank)], dim=1)

    def encode(self, strain: torch.Tensor) -> torch.Tensor:
        return self.encoder(strain)

    def nll(self, strain: torch.Tensor, params_phys: torch.Tensor,
            rank: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """params_phys: [B, 11] physical units; rank: [B] long. Returns [B]."""
        if context is None:
            context = self.encoder(strain)
        ctx = self._full_context(context, rank)
        y = self.scaler.normalize(params_phys)
        log_sigma = torch.zeros_like(y)
        return self.flow.compute_psd_aware_nll(y, ctx, log_sigma)

    @torch.no_grad()
    def sample_posterior(self, strain: torch.Tensor, rank: int = 0,
                         n_samples: int = 256) -> torch.Tensor:
        """strain: [B, 3, T] -> samples [B, n_samples, 11] in PHYSICAL units."""
        context = self.encoder(strain)
        B = context.shape[0]
        r = torch.full((B,), rank, dtype=torch.long, device=context.device)
        ctx = self._full_context(context, r)
        ctx_rep = ctx.unsqueeze(1).expand(B, n_samples, ctx.shape[1]).reshape(B * n_samples, -1)
        z = torch.randn(B * n_samples, len(self.param_names), device=context.device)
        y, _ = self.flow.inverse(z, ctx_rep)
        y = y.reshape(B, n_samples, -1)
        return self.scaler.denormalize(y)

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        dev = next(self.parameters()).device
        self.scaler.to(dev)
        return out
