#!/usr/bin/env python3
"""
Enhanced PriorityNet for intelligent signal extraction ordering in overlapping GW scenarios.
Includes temporal encoding, multi-detector coherence, and uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import logging


class TemporalStrainEncoder(nn.Module):
    """
    CNN + BiLSTM + Multi-head Attention encoder for whitened strain segments.

    This module extracts multi-scale time-frequency features from multi-detector strain
    inputs, models temporal dependencies with a bidirectional LSTM, and aggregates long-
    range context via self-attention, projecting to a fixed 64-D representation suitable
    for downstream fusion.
    """

    def __init__(self, input_length: int = 2048, n_detectors: int = 3, hidden_dim: int = 128):
        """
        Initialize the temporal strain encoder.

        Args:
            input_length: Number of time samples per detector channel (e.g., 2048 for 1s at 2048 Hz).
            n_detectors: Number of detector channels (e.g., H1, L1, V1 → 3).
            hidden_dim: Hidden size for the BiLSTM per direction (total BiLSTM output is 2×hidden_dim).
        """
        super().__init__()

        # Multi-scale CNN for time-frequency features
        # Architecture based on LIGO ML pipelines (2048 Hz, 1s segments)
        self.conv_blocks = nn.ModuleList(
            [
                self._conv_block(n_detectors, 32, kernel_size=64, stride=4),
                self._conv_block(32, 64, kernel_size=32, stride=4),
                self._conv_block(64, 128, kernel_size=16, stride=2),
                self._conv_block(128, 128, kernel_size=8, stride=2),
            ]
        )

        # Calculate sequence length after convolutions
        seq_len = input_length
        for _ in range(4):
            seq_len = seq_len // 4 if seq_len > 64 else seq_len // 2
        self.seq_len = max(seq_len, 8)

        # BiLSTM for temporal dependencies (captures chirp evolution)
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # Multi-head self-attention for long-range dependencies
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=8, dropout=0.1, batch_first=True
        )

        # Projection to fixed dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
        )

    def _conv_block(self, in_channels, out_channels, kernel_size, stride):
        """
        Construct a 1D convolution block with BatchNorm, GELU, and Dropout.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Convolution kernel width.
            stride: Convolution stride (downsampling factor).

        Returns:
            A sequential module: Conv1d → BatchNorm1d → GELU → Dropout.
        """
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, strain_segments: torch.Tensor) -> torch.Tensor:
        """
        Encode whitened strain segments into a 64-D temporal representation.

        Args:
            strain_segments: Tensor of shape [batch, n_detectors, time_samples].

        Returns:
            A tensor of shape [batch, 64] with aggregated temporal features.
        """
        x = strain_segments

        # Multi-scale convolution
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Reshape for LSTM: [batch, seq_len, features]
        x = x.transpose(1, 2)

        # BiLSTM encoding
        lstm_out, _ = self.bilstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling + max pooling
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool, _ = torch.max(attn_out, dim=1)
        combined = avg_pool + max_pool

        # Project to fixed dimension
        encoded = self.projection(combined)

        return encoded


class CrossSignalAnalyzer(nn.Module):
    """
    Pairwise overlap analyzer with optional learned attention over pairwise relations.

    Computes optional attention-weighted aggregates of pairwise features (e.g., time separation,
    sky separation, mass/frequency overlap proxies, distance ratio, polarization diff),
    yielding a 16-D overlap feature per signal within a multi-signal scenario.
    """

    def __init__(self, use_attention=True, importance_hidden_dim=16):
        """
        Initialize the pairwise feature and importance networks.

        Args:
            use_attention: Whether to use attention weighting for pairwise features (default True).
            importance_hidden_dim: Hidden dimension for importance network (default 16).
        """
        super().__init__()

        self.use_attention = use_attention

        # Pairwise feature extractor
        self.overlap_net = nn.Sequential(
            nn.Linear(8, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 16)
        )

        # Learns which pairwise relationships matter most (optional)
        self.importance_net = nn.Sequential(
            nn.Linear(8, importance_hidden_dim),
            nn.LayerNorm(importance_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(importance_hidden_dim, 1),
        )

    def forward(self, params_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute optional attention-weighted pairwise overlap features for a scenario.

        Args:
            params_batch: Tensor [n_signals, 15] of normalized signal parameters.

        Returns:
            Tensor [n_signals, 16] with aggregated overlap features per signal.
        """
        n_signals = params_batch.shape[0]

        if n_signals < 2:
            # No overlap for single signal
            return torch.zeros(n_signals, 16, device=params_batch.device)

        overlap_scores = []

        for i in range(n_signals):
            # Compute overlap metrics with all other signals
            pairwise_features = []
            pairwise_importance = []

            for j in range(n_signals):
                if i == j:
                    continue

                # Time separation (from geocent_time)
                dt = torch.abs(params_batch[i, 5] - params_batch[j, 5])

                # Sky position separation (RA, Dec)
                dra = torch.abs(params_batch[i, 3] - params_batch[j, 3])
                ddec = torch.abs(params_batch[i, 4] - params_batch[j, 4])
                sky_sep = torch.sqrt(dra**2 + ddec**2)

                # Mass similarity (chirp mass overlap indicator)
                m1_i = params_batch[i, 0] * 95 + 5
                m2_i = params_batch[i, 1] * 95 + 5
                m1_j = params_batch[j, 0] * 95 + 5
                m2_j = params_batch[j, 1] * 95 + 5

                mc_i = (m1_i * m2_i) ** (3 / 5) / (m1_i + m2_i) ** (1 / 5)
                mc_j = (m1_j * m2_j) ** (3 / 5) / (m1_j + m2_j) ** (1 / 5)
                mass_similarity = 1.0 / (1.0 + torch.abs(mc_i - mc_j) / 30.0)

                # Frequency overlap estimate
                f_isco_i = 220.0 / (m1_i + m2_i)
                f_isco_j = 220.0 / (m1_j + m2_j)
                freq_overlap = torch.exp(-torch.abs(f_isco_i - f_isco_j) / 100.0)

                # Distance ratio (SNR confusion indicator)
                dist_i = params_batch[i, 2] * 2950 + 50
                dist_j = params_batch[j, 2] * 2950 + 50
                dist_ratio = torch.minimum(dist_i, dist_j) / torch.maximum(dist_i, dist_j)

                # Polarization angles
                dpsi = torch.abs(params_batch[i, 7] - params_batch[j, 7])

                features = torch.stack(
                    [dt, sky_sep, mass_similarity, freq_overlap, dist_ratio, dpsi, dra, ddec]
                )

                pairwise_features.append(features)

                # Learned importance score for this pair (if attention enabled)
                if self.use_attention:
                    importance = self.importance_net(features)
                    pairwise_importance.append(importance)

            if len(pairwise_features) > 0:
                # Stack all pairwise features
                pairwise_tensor = torch.stack(pairwise_features)  # [n-1, 8]

                if self.use_attention and len(pairwise_importance) > 0:
                    # Attention-weighted aggregation across pairs
                    importance_tensor = torch.stack(pairwise_importance)  # [n-1, 1]
                    attention_weights = F.softmax(importance_tensor, dim=0)  # [n-1, 1]
                    # Weighted sum of pairwise features
                    weighted_overlap = (attention_weights * pairwise_tensor).sum(dim=0)  # [8]
                else:
                    # Simple mean aggregation (no attention)
                    weighted_overlap = pairwise_tensor.mean(dim=0)  # [8]

                overlap_scores.append(weighted_overlap)
            else:
                overlap_scores.append(torch.zeros(8, device=params_batch.device))

        overlap_tensor = torch.stack(overlap_scores)  # [n_signals, 8]
        overlap_features = self.overlap_net(overlap_tensor)  # [n_signals, 16]

        return overlap_features


class ResidualBlock(nn.Module):
    """
    Residual MLP block with pre-layer normalization and optional projection.

    Applies LayerNorm → Linear → GELU → Dropout, then adds a skip connection
    (identity or linear projection if dimensions differ), improving gradient flow
    and stabilizing deeper stacks in the feature extractor.
    """

    def __init__(self, in_dim, out_dim, dropout=0.15):
        """
        Initialize a ResidualBlock.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            dropout: Dropout probability applied after activation.
        """
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Skip connection projection
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        """
        Forward pass for the residual block.

        Args:
            x: Input tensor of shape [..., in_dim].

        Returns:
            Output tensor of shape [..., out_dim] after residual connection.
        """
        # Pre-norm
        identity = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Residual
        return x + self.skip(identity)


class SignalFeatureExtractor(nn.Module):
    """
    Enhanced metadata feature extractor with residual connections and physics encodings.

    Transforms normalized signal parameters into a learned representation via stacked
    residual MLP blocks (projected to 64-D), augments with analytic physics-derived
    features (32-D), and concatenates them into a 96-D metadata representation.
    """

    def __init__(self, input_dim: int = 16, config=None):
        """
        Initialize the feature extractor.

        Args:
            input_dim: Number of normalized scalar parameters per signal (default 16, including network_snr).
            config: Dict or object with fields hidden_dims and dropout; falls back to defaults when absent.
        """
        super().__init__()

        if isinstance(config, dict) or hasattr(config, "get"):
            # Try top level first, then nested priority_net
            hidden_dims = config.get("hidden_dims", None)
            if hidden_dims is None and "priority_net" in config:
                hidden_dims = config["priority_net"].get("hidden_dims", [640, 512, 384, 256])
            if hidden_dims is None:
                hidden_dims = [640, 512, 384, 256]
            
            dropout = config.get("dropout", None)
            if dropout is None and "priority_net" in config:
                dropout = config["priority_net"].get("dropout", 0.25)
            if dropout is None:
                dropout = 0.25
        elif config is not None:
            hidden_dims = getattr(config, "hidden_dims", None)
            if hidden_dims is None and hasattr(config, "priority_net"):
                pn_cfg = getattr(config, "priority_net")
                hidden_dims = getattr(pn_cfg, "hidden_dims", [640, 512, 384, 256])
            if hidden_dims is None:
                hidden_dims = [640, 512, 384, 256]
            
            dropout = getattr(config, "dropout", None)
            if dropout is None and hasattr(config, "priority_net"):
                pn_cfg = getattr(config, "priority_net")
                dropout = getattr(pn_cfg, "dropout", 0.25)
            if dropout is None:
                dropout = 0.25
        else:
            hidden_dims = [640, 512, 384, 256]
            dropout = 0.25

        # Input embedding
        self.input_embed = nn.Linear(input_dim, hidden_dims[0])

        # Residual blocks
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dims[0], hidden_dims[0], dropout),
                ResidualBlock(hidden_dims[0], hidden_dims[1], dropout),
                ResidualBlock(hidden_dims[1], hidden_dims[2], dropout),
                ResidualBlock(hidden_dims[2], hidden_dims[3], dropout),
            ]
        )

        # Final projection to 64-D
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dims[3]), nn.Linear(hidden_dims[3], 64)
        )

        # Physics encoder (8 → 32)
        self.physics_encoder = nn.Sequential(
            nn.Linear(8, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode normalized parameters into a 96-D metadata feature.

        Args:
            x: Tensor [N, input_dim] of normalized scalar parameters.

        Returns:
            Tensor [N, 96] formed by concatenating learned (64-D) and physics (32-D) encodings.
        """
        # Network features
        features = self.input_embed(x)
        for block in self.blocks:
            features = block(features)
        network_features = self.output_proj(features)

        # Physics features
        physics_features = self._compute_physics_features(x)
        physics_encoded = self.physics_encoder(physics_features)

        # Combine
        return torch.cat([network_features, physics_encoded], dim=1)

    def _compute_physics_features(self, params: torch.Tensor) -> torch.Tensor:
        """
        Compute analytic physics-derived features from normalized parameters.

        Denormalizes selected parameters to physically meaningful scales, computes
        derived quantities (chirp mass, symmetric mass ratio, f_ISCO, χ_eff proxy,
        SNR estimate, difficulty), and returns an 8-D normalized feature vector.

        Args:
            params: Tensor [N, input_dim] of normalized parameters.

        Returns:
            Tensor [N, 8] with physics-derived features in [0, 1] where applicable.
        """
        batch_size = params.shape[0]
        physics_features = torch.zeros(batch_size, 8, device=params.device)

        try:
            # Extract basic parameters (denormalized)
            m1 = params[:, 0] * 95 + 5
            m2 = params[:, 1] * 95 + 5
            # ← CRITICAL FIX (Nov 15): Denormalize distance using log-scale (inverse of log-space encoding)
            # params[:, 2] is in [0, 1], maps to log10(d) in [log10(50), log10(800)] = [1.699, 2.903]
            log_min = np.log10(50.0)
            log_max = np.log10(800.0)
            distance = 10 ** (params[:, 2] * (log_max - log_min) + log_min)

            # Derived quantities
            total_mass = m1 + m2
            chirp_mass = (m1 * m2) ** (3 / 5) / total_mass ** (1 / 5)
            mass_ratio = torch.minimum(m1, m2) / torch.maximum(m1, m2)
            eta = m1 * m2 / (total_mass**2)

            # SNR estimation
            estimated_snr = (
                8.0 * (chirp_mass / 30.0) ** (5 / 6) * (400.0 / torch.clamp(distance, min=50.0))
            )

            # Frequency estimates
            f_isco = 220.0 / total_mass

            # Effective spin
            if params.shape[1] >= 11:
                a1 = params[:, 9] * 0.99
                a2 = params[:, 10] * 0.99
                chi_eff = (m1 * a1 + m2 * a2) / total_mass
            else:
                chi_eff = torch.zeros_like(m1)

            # FIX #7: Circular SNR-Distance Dependencies
            # Difficulty should use distance only (not both distance and derived SNR which creates redundancy)
            # Farther objects are harder to detect → difficulty proportional to distance
            difficulty = torch.log(distance / 100.0)

            # Normalize features
            physics_features[:, 0] = torch.clamp(chirp_mass / 50.0, 0, 1)
            physics_features[:, 1] = mass_ratio
            physics_features[:, 2] = eta * 4
            physics_features[:, 3] = torch.clamp(estimated_snr / 50.0, 0, 1)
            physics_features[:, 4] = torch.clamp(f_isco / 1000.0, 0, 1)
            physics_features[:, 5] = torch.clamp((chi_eff + 1) / 2, 0, 1)
            physics_features[:, 6] = torch.clamp((difficulty + 5) / 10, 0, 1)
            physics_features[:, 7] = torch.clamp(total_mass / 200.0, 0, 1)

        except Exception as e:
            logging.debug(f"Physics feature computation failed: {e}")
            physics_features.fill_(0.5)

        return physics_features


class MultiModalFusion(nn.Module):
    """
    Cross-modal fusion block with self-attention and residual FFN.

    Concatenates metadata (96-D), overlap (16-D), temporal strain (64-D), and edge
    embeddings (32-D), projects to an output_dim space, applies a single-token
    self-attention block and a residual feed-forward network to produce a unified
    fused embedding for downstream heads.
    """

    def __init__(
        self,
        metadata_dim=96,
        overlap_dim=16,
        temporal_dim=64,
        edge_dim=32,
        output_dim=64,
        num_heads=4,
        use_attention=True,
        attention_dropout=0.08,
    ):
        """
        Initialize the multi-modal fusion module.

        Args:
            metadata_dim: Input dimension of metadata features.
            overlap_dim: Input dimension of overlap features.
            temporal_dim: Input dimension of temporal strain features.
            edge_dim: Input dimension of edge-type embeddings.
            output_dim: Working dimension for attention/FFN projections.
            num_heads: Number of attention heads for the self-attention block.
            use_attention: Whether to use self-attention (default True).
            attention_dropout: Dropout rate for attention module (default 0.08).
        """
        super().__init__()

        self.use_attention = use_attention
        total_dim = metadata_dim + overlap_dim + temporal_dim + edge_dim

        # Input projection
        self.input_proj = nn.Linear(total_dim, output_dim)
        self.input_norm = nn.LayerNorm(output_dim)  # stabilize fused distribution

        # Cross-modal self-attention (optional)
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                output_dim, num_heads, dropout=attention_dropout, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(output_dim)
            logging.info(
                f"✅ MultiModalFusion: attention enabled ({num_heads} heads, dropout={attention_dropout})"
            )
        else:
            logging.info("⚠️  MultiModalFusion: attention disabled (projection + FFN only)")

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(attention_dropout),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.ffn_norm = nn.LayerNorm(output_dim)

    def forward(self, metadata, overlap, temporal, edge):
        """
        Fuse modalities into a single 64-D token embedding.

        Args:
            metadata: Tensor [N, metadata_dim] from the metadata extractor.
            overlap: Tensor [N, overlap_dim] from the overlap analyzer.
            temporal: Tensor [N, temporal_dim] from the strain encoder (zeros if absent).
            edge: Tensor [N, edge_dim] from edge-type embeddings (zeros if absent).

        Returns:
            Tensor [N, output_dim] fused representation for downstream heads.
        """
        # Concatenate all modalities
        combined = torch.cat([metadata, overlap, temporal, edge], dim=-1)

        # Project to output dimension
        x = self.input_proj(combined)
        x = self.input_norm(x)
        x = x.unsqueeze(1)  # [batch, 1, dim]

        # Optional self-attention with residual
        if self.use_attention:
            residual = x
            attn_out, _ = self.attention(x, x, x)
            x = residual + attn_out
            x = self.attn_norm(x)

        # FFN with residual
        residual = x
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        x = self.ffn_norm(x)

        return x.squeeze(1)


class AdaptiveRankingLoss(nn.Module):
    """
    Adaptive pairwise ranking loss with learned margin scaling and optional pair weights.

    Computes hinge-like violations over all pairs (i, j) where targets differ by ≥ 0.05,
    with a margin proportional to the target gap and a multiplicative margin_scale
    (e.g., larger for dense overlaps). Optional per-sample SNR weights can be averaged
    per pair to emphasize de-noising (e.g., weak SNR cases) during ranking.
    """

    def __init__(self, base_margin: float = 0.08):
        """
        Initialize the adaptive ranking loss.

        Args:
            base_margin: Base margin factor multiplied by clamped |Δtarget| to form pairwise margins.
                         Increased from 0.05 to 0.08 (Nov 24, 2025) for stronger ranking separation.
        """
        super().__init__()
        self.base_margin = base_margin

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
        snr_weights: Optional[torch.Tensor] = None,
        margin_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the adaptive pairwise ranking loss.

        Args:
            predictions: Tensor [n] of predicted priorities for a scenario.
            targets: Tensor [n] of target priorities for the same scenario.
            uncertainties: Optional tensor [n] of predicted uncertainties (unused here; kept for API cohesion).
            snr_weights: Optional tensor [n] of per-sample SNR-derived weights; if provided,
                         each pair (i, j) is weighted by 0.5·(w_i + w_j).
            margin_scale: Scalar multiplier applied to the base margin to emphasize certain scenarios.

        Returns:
            A scalar tensor with the average pairwise violation over considered pairs.
        """
        n_samples = predictions.shape[0]
        if n_samples < 2:
            return torch.zeros(1, device=predictions.device, requires_grad=True).squeeze()

        loss = predictions.new_tensor(0.0, requires_grad=True)
        count = 0

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                target_diff = torch.abs(targets[i] - targets[j])
                if target_diff < 0.05:
                    continue

                margin = (
                    margin_scale * self.base_margin * torch.clamp(target_diff, min=0.1, max=1.0)
                )

                pair_w = 1.0
                if snr_weights is not None:
                    pair_w = 0.5 * (snr_weights[i] + snr_weights[j])

                if targets[i] > targets[j]:
                    violation = torch.clamp(predictions[j] - predictions[i] + margin, min=0.0)
                else:
                    violation = torch.clamp(predictions[i] - predictions[j] + margin, min=0.0)

                loss = loss + pair_w * violation
                count += 1

        return loss / count if count > 0 else predictions.new_tensor(0.0, requires_grad=True)


class PriorityLoss(nn.Module):
    """
    Composite loss for priority learning with calibration and robustness components.

    Combines:
    - Weighted MSE (with optional SNR emphasis) for scale calibration,
    - Adaptive pairwise ranking loss to preserve correct ordering (with larger margins for dense overlaps),
    - Uncertainty regression toward |error| to keep σ informative,
    - A light batch-level mean/std calibration penalty to correct global offset/compression.
    """

    def __init__(
        self,
        ranking_weight: float = 0.4,
        mse_weight: float = 0.5,
        uncertainty_weight: float = 0.35,
        use_snr_weighting: bool = True,
        label_smoothing: float = 0.05,
        lambda_calib: float = 0.3,
        uncertainty_lower_bound: float = 0.01,
        uncertainty_upper_bound: float = 0.50,
        uncertainty_bounds_weight: float = 0.05,
        calib_mean_weight: float = 0.20,
        calib_max_weight: float = 1.50,
        calib_range_weight: float = 1.00,
    ):
        """
        Initialize PriorityLoss.

        Args:
            ranking_weight: Weight for the pairwise ranking component.
            mse_weight: Weight for the regression (MSE) component.
            uncertainty_weight: Weight for the uncertainty calibration component.
            use_snr_weighting: Enable SNR-based weighting in 5+ overlaps for hard cases.
            label_smoothing: Smooth targets toward 0.5 by ε to reduce overfitting to extremes.
            lambda_calib: Batch mean/std alignment penalty coefficient.
            uncertainty_lower_bound: Minimum uncertainty value (prevent collapse).
            uncertainty_upper_bound: Maximum uncertainty value (prevent explosion).
            uncertainty_bounds_weight: Penalty weight for uncertainty bounds violations.
            calib_mean_weight: Weight for mean calibration loss (Nov 13 fix).
            calib_max_weight: Weight for max range expansion loss (Nov 13 fix).
            calib_range_weight: Weight for range regularization loss (Nov 13 fix).
        """
        super().__init__()
        self.ranking_weight = ranking_weight
        self.mse_weight = mse_weight
        self.uncertainty_weight = uncertainty_weight
        self.use_snr_weighting = use_snr_weighting
        self.label_smoothing = label_smoothing
        self.lambda_calib = lambda_calib
        self.uncertainty_lower_bound = uncertainty_lower_bound
        self.uncertainty_upper_bound = uncertainty_upper_bound
        self.uncertainty_bounds_weight = uncertainty_bounds_weight
        self.calib_mean_weight = calib_mean_weight  # NEW (Nov 13)
        self.calib_max_weight = calib_max_weight    # NEW (Nov 13)
        self.calib_range_weight = calib_range_weight  # NEW (Nov 13)
        self.ranking_loss_fn = AdaptiveRankingLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor,
        snr_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss and return individual components for logging.

        Args:
            predictions: Predicted priorities [n].
            targets: Target priorities [n].
            uncertainties: Predicted uncertainties [n].
            snr_values: Optional per-sample SNRs [n] to emphasize weak-SNR samples (used when n ≥ 5).

        Returns:
            Dict[str, Tensor] with keys: 'total', 'mse', 'ranking', 'uncertainty'.
        """
        # Label smoothing for regression
        if self.label_smoothing > 0:
            smoothed_targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smoothed_targets = targets

        # Bucket 5+ detection by scenario length
        n_signals = predictions.shape[0]
        bucket5 = n_signals >= 5

        # Low-SNR emphasis weights (only for bucket 5+)
        if self.use_snr_weighting and (snr_values is not None) and bucket5:
            snr_w = 2.0 * torch.exp(-snr_values / 15.0) + 0.5
            snr_w = snr_w / snr_w.mean().clamp_min(1e-6)
        else:
            snr_w = torch.ones_like(predictions)

        # Weighted MSE
        mse_loss = (snr_w * (predictions - smoothed_targets) ** 2).mean()

        # Adaptive ranking with larger margin and pair weights in bucket 5+
        ranking_loss = self.ranking_loss_fn(
            predictions,
            targets,
            uncertainties,
            snr_weights=snr_w if bucket5 else None,
            margin_scale=(2.0 if bucket5 else 1.2),  # INCREASED (Nov 24): 1.6→2.0 for 5+, 1.0→1.2 for <5 (target pairwise acc 0.85+)
        )

        # ✅ FIXED (Nov 19 22:45): Uncertainty calibration with correlation loss
        # The issue: pure MSE doesn't enforce correlation, it can achieve low loss with random unc
        # Solution: Add explicit correlation penalty to force learning
        pred_error = torch.abs(predictions - targets)
        
        # Clamp error to [0.001, 1.0] to avoid log(0) and extreme scales
        clamped_error = torch.clamp(pred_error, min=0.001, max=1.0)
        
        # Part 1: Direct MSE regression toward error magnitude (L2 regression)
        mse_unc_loss = F.mse_loss(uncertainties, clamped_error.detach())
        
        # Part 2: Correlation penalty (CRITICAL FIX)
        # Penalize low correlation between uncertainty and error
        # cor = (unc - mean_unc) · (err - mean_err) / (std_unc · std_err)
        unc_centered = uncertainties - uncertainties.mean()
        err_centered = clamped_error - clamped_error.mean()
        
        # Compute Pearson correlation
        cov = (unc_centered * err_centered).mean()
        std_unc = uncertainties.std(unbiased=False).clamp(min=1e-6)
        std_err = clamped_error.std(unbiased=False).clamp(min=1e-6)
        correlation = cov / (std_unc * std_err + 1e-6)
        
        # Penalty: maximize correlation (target cor >= 0.5)
        correlation_loss = torch.relu(0.5 - correlation).pow(2)
        
        # Combine three components
        uncertainty_loss = 0.5 * mse_unc_loss + 0.3 * correlation_loss + 0.2 * (
            torch.relu(-correlation) * 10  # Additional penalty for negative correlation
        )
        
        # Uncertainty bounds penalty (prevent collapse/explosion)
        unc_lower_penalty = torch.relu(self.uncertainty_lower_bound - uncertainties).mean()
        unc_upper_penalty = torch.relu(uncertainties - self.uncertainty_upper_bound).mean()
        uncertainty_bounds_loss = self.uncertainty_bounds_weight * (unc_lower_penalty + unc_upper_penalty)

        # FIX #4: Aggressive range expansion (Nov 15 DEEP FIX)
        # L = L_rank + L_mse + λ1·|mean(ŷ) − mean(y)| + λ2·(1 - max(ŷ)/max(y)) + λ3·(range(y)/range(ŷ))
        
        # Use configured calibration weights (Nov 15 DEEP FIX)
        lambda1 = self.calib_mean_weight  # mean alignment
        lambda2 = self.calib_max_weight   # max range expansion
        lambda3 = self.calib_range_weight # range regularization
        
        # Mean calibration loss: pull mean to target mean
        mean_gap = torch.abs(predictions.mean() - targets.mean())
        
        # Max calibration loss (CRITICAL): ratio-based penalty to push ceiling to target ceiling
        # If pred_max < target_max, large penalty. If pred_max >= target_max, zero penalty.
        # This creates strong gradient signal to expand predictions upward.
        target_max = targets.max()
        pred_max = predictions.max()
        max_ratio_penalty = torch.relu(1.0 - (pred_max + 1e-6) / (target_max + 1e-6))
        max_gap = max_ratio_penalty  # Ratio-based, not absolute
        
        # Range regularization: penalize when prediction range < target range
        target_range = targets.max() - targets.min()
        pred_range = predictions.max() - predictions.min()
        range_ratio_penalty = torch.relu(1.0 - (pred_range + 1e-6) / (target_range + 1e-6))
        range_penalty = range_ratio_penalty  # Ratio-based, not absolute
        
        # Diagnostic logging (every 100 batches in training)
        # if torch.rand(1).item() < 0.01:  # 1% sample
        #     logging.debug(
        #         f"Range expansion: tgt_range=[{targets.min():.3f}, {targets.max():.3f}] ({target_range:.3f}), "
        #         f"pred_range=[{predictions.min():.3f}, {predictions.max():.3f}] ({pred_range:.3f}), "
        #         f"mean_gap={mean_gap:.4f}, max_penalty={max_gap:.4f}, range_penalty={range_penalty:.4f}"
        #     )
        
        calib_loss = lambda1 * mean_gap + lambda2 * max_gap + lambda3 * range_penalty
        
        # CRITICAL (Nov 15): Minimum variance penalty
        # If predictions have very low variance, penalize heavily regardless of other metrics
        # This prevents the "all predictions near 0.5" collapse
        min_variance_penalty = 0.0
        pred_std = predictions.std(unbiased=False)
        target_std = targets.std(unbiased=False)
        
        # Penalize if prediction std < 50% of target std
        if pred_std < 0.5 * target_std and target_std > 1e-4:
            variance_ratio = pred_std / (target_std + 1e-8)
            min_variance_penalty = 2.0 * torch.relu(0.5 - variance_ratio)  # ⬇️ Reduced from 10.0 (too aggressive)
        
        # CRITICAL (Nov 16 FIX): Hard bounds penalty to prevent out-of-range predictions
        # Penalize predictions outside [0, 1] range BEFORE clipping in forward pass
        # This encourages optimizer to learn valid predictions rather than relying on clipping
        out_of_bounds_low = torch.relu(-predictions).mean()  # Penalize < 0
        out_of_bounds_high = torch.relu(predictions - 1.0).mean()  # Penalize > 1
        bounds_penalty = 5.0 * (out_of_bounds_low + out_of_bounds_high)  # Strong penalty

        # Bucket-aware weights
        effective_ranking_weight = self.ranking_weight
        effective_mse_weight = self.mse_weight

        if bucket5:
            # Emphasize ordering in dense overlaps; slightly down-weight MSE
            effective_ranking_weight = self.ranking_weight * 1.5
            effective_mse_weight = self.mse_weight * 0.85

        total_loss = (
            effective_mse_weight * mse_loss
            + effective_ranking_weight * ranking_loss
            + self.uncertainty_weight * uncertainty_loss
            + uncertainty_bounds_loss  # NEW: Bounds penalty
            + calib_loss  # Calibration term to widen spread
            + min_variance_penalty  # CRITICAL: Prevent variance collapse
            + bounds_penalty  # CRITICAL (Nov 16): Hard bounds enforcement
        )

        return {
            "total": total_loss,
            "mse": mse_loss.detach(),
            "ranking": ranking_loss.detach(),
            "uncertainty": uncertainty_loss.detach(),
        }


class PriorityNet(nn.Module):
    """
    Enhanced PriorityNet with config-driven architecture and multi-modal fusion.

    Encodes metadata, overlap structure, temporal strain, and edge-type context;
    fuses them with attention; and predicts per-signal priorities with calibrated
    uncertainty estimates. Includes a lightweight affine calibrator (gain/bias).
    """

    def __init__(
        self,
        config=None,
        use_strain: bool = None,
        use_edge_conditioning: bool = None,
        n_edge_types: int = None,
    ):
        """
        Initialize PriorityNet components and read configuration safely.

        Args:
            config: Dict or object providing hidden_dims, dropout, use_strain, use_edge_conditioning, n_edge_types.
            use_strain: Override to enable/disable temporal strain encoder (default from config).
            use_edge_conditioning: Override to enable/disable edge-type embedding (default from config).
            n_edge_types: Size of edge-type vocabulary (default from config or 17).
        """
        super().__init__()

        # ✅ FIXED: Accept both dict and object-style configs
        if config is not None and (isinstance(config, dict) or hasattr(config, "__dict__")):
            print("🔧 Using provided configuration for PriorityNet.")
            self.config = config
        else:
            print("🔧 Using default configuration for PriorityNet.")
            self.config = self._default_config()

        def cfg_get(key, default):
            """
            Read config value, checking both top level and nested 'priority_net' section.
            Handles both dict and object-style configs.
            """
            if isinstance(self.config, dict):
                # Try top level first
                if key in self.config:
                    return self.config[key]
                # Try nested priority_net section
                if "priority_net" in self.config and isinstance(self.config["priority_net"], dict):
                    if key in self.config["priority_net"]:
                        return self.config["priority_net"][key]
                return default
            else:
                # Object-style config
                if hasattr(self.config, key):
                    return getattr(self.config, key)
                # Try nested priority_net section
                if hasattr(self.config, "priority_net"):
                    priority_net_cfg = getattr(self.config, "priority_net")
                    if hasattr(priority_net_cfg, key):
                        return getattr(priority_net_cfg, key)
                return default

        # Flags
        self.use_strain = use_strain if use_strain is not None else cfg_get("use_strain", True)
        self.use_edge_conditioning = (
            use_edge_conditioning
            if use_edge_conditioning is not None
            else cfg_get("use_edge_conditioning", True)
        )
        self.n_edge_types = (
            n_edge_types if n_edge_types is not None else cfg_get("n_edge_types", 17)
        )

        # Temporal strain encoder (optional) - support both CNN+BiLSTM and Transformer
        self.use_transformer_encoder = cfg_get("use_transformer_encoder", False)

        if self.use_strain:
            if self.use_transformer_encoder:
                # Lazy import to avoid circular dependency
                from ahsd.models.transformer_encoder import TransformerStrainEncoder

                # Use Transformer-based encoder (lightweight, matches checkpoint training)
                self.strain_encoder = TransformerStrainEncoder(
                    use_whisper=False,  # Use lightweight transformer to match checkpoint
                    freeze_layers=4,
                    input_length=2048,
                    n_detectors=2,  # H1, L1 detectors
                    output_dim=64,
                )
                logging.info(
                    f"   ✅ Strain encoder: TransformerStrainEncoder (use_transformer_encoder=True)"
                )
            else:
                # Fall back to CNN+BiLSTM (legacy)
                self.strain_encoder = TemporalStrainEncoder(
                    input_length=2048, n_detectors=3, hidden_dim=128
                )
                logging.info(f"   ℹ️  Strain encoder: TemporalStrainEncoder (CNN+BiLSTM)")
            temporal_dim = 64
        else:
            temporal_dim = 0

        # Cross-signal overlap analyzer with configurable attention
        overlap_use_attention = cfg_get("overlap_use_attention", True)
        print("Overlap use attention:", overlap_use_attention)
        overlap_importance_hidden = cfg_get("overlap_importance_hidden", 32)
        self.cross_signal_analyzer = CrossSignalAnalyzer(
            use_attention=overlap_use_attention, importance_hidden_dim=overlap_importance_hidden
        )
        if overlap_use_attention:
            logging.info(
                f"   ✅ Overlap analyzer: attention enabled (hidden_dim={overlap_importance_hidden})"
            )
        else:
            logging.info(f"   ⚠️  Overlap analyzer: attention disabled (mean aggregation)")

        self.signal_encoder = SignalFeatureExtractor(config=self.config)

        # Edge case embedding with padding_idx=0 for 'none'
        if self.use_edge_conditioning:
            self.edge_embedding = nn.Embedding(
                num_embeddings=self.n_edge_types, embedding_dim=32, padding_idx=0
            )
            # Initialize edge embeddings with small uniform values to control variance
            with torch.no_grad():
                self.edge_embedding.weight.uniform_(-0.05, 0.05)
                # Keep padding vector as zero
                self.edge_embedding.weight[0].fill_(0.0)
            edge_dim = 32
        else:
            edge_dim = 0

        # Log configuration
        logging.info(f"🔍 PriorityNet Configuration:")
        logging.info(f"   use_strain: {self.use_strain} → temporal_dim: {temporal_dim}")
        logging.info(
            f"   use_edge_conditioning: {self.use_edge_conditioning} → edge_dim: {edge_dim}"
        )
        logging.info(f"   n_edge_types: {self.n_edge_types}")
        if hasattr(self.config, "hidden_dims"):
            logging.info(f"   hidden_dims: {self.config.hidden_dims}")
        if hasattr(self.config, "dropout"):
            logging.info(f"   dropout: {self.config.dropout}")

        # NEW: Dedicated SNR embedding pathway (FIX #2: Strengthen SNR pathway)
        self.snr_embedding = nn.Sequential(
            nn.Linear(1, 16), nn.LayerNorm(16), nn.GELU(), nn.Dropout(0.1), nn.Linear(16, 32)
        )

        # Modal fusion with configurable attention (updated to include SNR embedding)
        # total_dim = metadata(96) + overlap(16) + temporal(64) + edge(32) + snr(32) = 240
        use_modal_fusion_attention = cfg_get("use_modal_fusion", True)
        attention_num_heads = cfg_get("attention_num_heads", 4)
        attention_dropout = cfg_get("attention_dropout", 0.08)

        self.modal_fusion = MultiModalFusion(
            metadata_dim=96 + 32,  # Concatenate SNR embedding with metadata
            overlap_dim=16,
            temporal_dim=64,
            edge_dim=32,
            output_dim=64,
            num_heads=attention_num_heads,
            use_attention=use_modal_fusion_attention,
            attention_dropout=attention_dropout,
        )
        # Optional: overlap density head (not returned by forward; use in trainer if desired)
        self.overlap_head = nn.Linear(64, 4)

        if use_modal_fusion_attention:
            logging.info(
                f"   ✅ Modal fusion: attention enabled ({attention_num_heads} heads, dropout={attention_dropout})"
            )
        else:
            logging.info(f"   ⚠️  Modal fusion: attention disabled (projection + FFN only)")

        # FIX #1: Priority head without sigmoid/tanh (already linear)
        # FIX #5: Initialize with small weights and bias near dataset mean
        self.priority_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(cfg_get("dropout", 0.12)),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),  # priority output (linear, no squashing)
        )

        # Initialize final layer with stronger gradients and higher bias (Nov 15 DEEP FIX)
        final_layer = self.priority_head[-1]
        output_weight_std = cfg_get("output_weight_std", 0.10)  # Read from config, default 0.10 (10x)
        nn.init.normal_(final_layer.weight, mean=0.0, std=output_weight_std)  # ⬆️ 10x larger for even steeper gradients
        output_bias_init = cfg_get("output_bias_init", 0.45)  # Read from config, default 0.45 (below mean)
        final_layer.bias.data.fill_(output_bias_init)  # Initialize to 0.45 to leave headroom for expansion

        self.uncertainty_head = nn.Sequential(nn.Linear(64, 1), nn.Softplus(beta=2.0))  # sigma > 0, ⬆️ beta 2.0 for stronger gradients (Nov 13 fix)

        # Affine calibration parameters with bounds (Nov 13 fix)
        # CRITICAL: Initialize gain higher to push expansion from start
        self.prio_gain = nn.Parameter(torch.tensor(1.8))  # Start at 1.8x to force range expansion
        self.prio_bias = nn.Parameter(torch.tensor(-0.05))  # Slight downward bias to lower floor
        
        # Store affine bounds as instance variables for forward pass (Nov 15 DEEP FIX)
        self.affine_gain_min = cfg_get("affine_gain_min", 1.2)   # ↑ from 0.7 — prevent collapse
        self.affine_gain_max = cfg_get("affine_gain_max", 2.5)   # ↑ from 1.5 — allow aggressive expansion
        self.affine_bias_min = cfg_get("affine_bias_min", -0.2)  # ↑ from -0.1 — push floor down
        self.affine_bias_max = cfg_get("affine_bias_max", 0.05)  # ↓ from 0.1 — prevent floor rise

        logging.info(
            f"🔍 PriorityNet Configuration: use_strain={self.use_strain}, use_edge_conditioning={self.use_edge_conditioning}, n_edge_types={self.n_edge_types}"
        )
        logging.info(
            f"   dropout={cfg_get('dropout', 0.2)}, hidden_dims={cfg_get('hidden_dims', [512,384,256,128])}"
        )
        logging.info(f"✅ Enhanced PriorityNet initialized with attention fusion")

    def _default_config(self):
        """
        Provide a minimal default configuration object for PriorityNet.

        Returns:
            A dynamic object with default fields matching enhanced_training.yaml structure.
        """
        return type(
            "Config",
            (),
            {
                # Architecture
                "hidden_dims": [640, 512, 384, 256],
                "dropout": 0.25,
                # Model flags
                "use_strain": True,
                "use_edge_conditioning": True,
                "n_edge_types": 19,
                "use_transformer_encoder": False,  # ✅ FIXED: Default to False (CNN+BiLSTM), not Transformer
                # Optimizer
                "optimizer": "AdamW",
                "learning_rate": 8.0e-5,
                "weight_decay": 1.5e-5,
                # Training schedule
                "batch_size": 48,
                "epochs": 250,
                "patience": 15,
                "warmup_epochs": 20,
                "warmup_start_factor": 0.02,
                # Scheduler
                "scheduler": "ReduceLROnPlateau",
                "scheduler_patience": 8,
                "scheduler_factor": 0.5,
                "min_lr": 1.0e-6,
                # Loss function
                "ranking_weight": 0.60,
                "mse_weight": 0.30,
                "uncertainty_weight": 0.10,
                "use_snr_weighting": True,
                "loss_scale_factor": 0.005,
                "label_smoothing": 0.02,
                # Gradient management
                "gradient_clip_norm": 2.0,
                "gradient_log_threshold": 0.5,
                # Attention/Modal fusion
                "use_modal_fusion": True,
                "attention_num_heads": 8,
                "attention_dropout": 0.15,
                # Overlap handling
                "overlap_use_attention": True,
                "overlap_importance_hidden": 32,
            },
        )()

    def _to_device_tensor(self, arr, device):
        """
        Convert input to a float32 tensor on the specified device with non-blocking transfer.

        Args:
            arr: Tensor or array-like to convert.
            device: Target device (e.g., 'cuda' or 'cpu').

        Returns:
            A torch.float32 tensor on the given device.
        """
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=torch.float32, non_blocking=True)
        return torch.as_tensor(np.asarray(arr), device=device, dtype=torch.float32)

    def forward(
        self,
        detections: List[Dict],
        strain_segments: Optional[torch.Tensor] = None,
        edge_type_ids: Optional[torch.Tensor] = None,
        training: bool = False,
        detector_dropout_prob: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with cross-modal attention fusion.

        Args:
            detections: List of per-signal dictionaries containing normalized/unnormalized parameters.
            strain_segments: Optional tensor [N, n_detectors, time_samples] of whitened strain.
            edge_type_ids: Optional tensor [N] of edge-type ids; 0 is padding for 'none'.
            training: Unused flag (reserved for future stochastic behavior).
            detector_dropout_prob: Unused argument placeholder for future extensions.

        Returns:
            priorities: Tensor [N] of predicted priorities (pre-calibrated by affine).
            uncertainties: Tensor [N] of positive uncertainties via Softplus.
        """
        # Check if detections is empty (supports Tensor, list, or other sequences)
        if isinstance(detections, torch.Tensor):
            if detections.numel() == 0:
                return torch.empty(0), torch.empty(0)
        else:
            if len(detections) == 0:
                return torch.empty(0), torch.empty(0)

        device = next(self.parameters()).device
        try:
            signal_tensor = self._detections_to_tensor(detections).to(device)
            if signal_tensor is None or signal_tensor.numel() == 0:
                n = len(detections)
                return torch.zeros(n, device=device), torch.ones(n, device=device)

            # Modal encodings
            metadata_features = self.signal_encoder(signal_tensor)  # [N, 96]
            overlap_features = self.cross_signal_analyzer(signal_tensor)  # [N, 16]

            # NEW: Extract and embed SNR (FIX #2: Strengthen SNR pathway)
            # network_snr is at index 15 in signal_tensor
            snr_raw = signal_tensor[:, 15:16]  # [N, 1] - already normalized to [0,1]
            snr_embedded = self.snr_embedding(snr_raw)  # [N, 32]

            # Concatenate SNR embedding with metadata
            metadata_features = torch.cat(
                [metadata_features, snr_embedded], dim=1
            )  # [N, 96+32=128]

            # Temporal features (zeros if absent)
            if self.use_strain and strain_segments is not None and strain_segments.numel() > 0:
                strain_seg = strain_segments.to(device)
                N = signal_tensor.shape[0]  # Number of detections
                
                # ✅ FIX: Handle strain_segments shape properly
                # Case 1: [n_detectors, time_samples] - shared strain, replicate for each detection
                # Case 2: [N, n_detectors, time_samples] - unique strain per detection
                if strain_seg.dim() == 2:
                    # Shared strain: [n_detectors, T] → [N, n_detectors, T]
                    strain_seg = strain_seg.unsqueeze(0).expand(N, -1, -1)
                elif strain_seg.dim() == 3 and strain_seg.shape[0] == 1:
                    # Single batch: [1, n_detectors, T] → [N, n_detectors, T]
                    strain_seg = strain_seg.expand(N, -1, -1)
                
                # Adapt detector count for TransformerStrainEncoder (expects 2: H1, L1)
                if self.use_transformer_encoder and strain_seg.shape[1] > 2:
                    strain_seg = strain_seg[:, :2, :]  # Use only H1, L1 (first 2 detectors)
                
                try:
                    temporal_features = self.strain_encoder(strain_seg)  # [N, 64]
                except Exception as e:
                    logging.error(f"Strain encoder failed: {e}", exc_info=True)
                    temporal_features = torch.zeros((signal_tensor.shape[0], 64), device=device)
                
                if temporal_features.shape[0] != N or temporal_features.shape[-1] != 64:
                    logging.error(f"Strain encoder shape mismatch: expected [{N}, 64], got {temporal_features.shape}")
                    temporal_features = torch.zeros((signal_tensor.shape[0], 64), device=device)

            else:
                temporal_features = torch.zeros((signal_tensor.shape[0], 64), device=device)

                # Edge conditioning inputs (IDs provided by caller or default 0)
            # FIX #3: Edge conditioning with dropout
            if self.use_edge_conditioning:
                if edge_type_ids is None:
                    edge_type_ids = torch.zeros(
                        signal_tensor.shape[0], dtype=torch.long, device=device
                    )
                else:
                    edge_type_ids = edge_type_ids.to(device=device, dtype=torch.long)
                    if edge_type_ids.dim() == 0:
                        edge_type_ids = edge_type_ids.unsqueeze(0).expand(signal_tensor.shape[0])
                    elif len(edge_type_ids) != signal_tensor.shape[0]:
                        edge_type_ids = torch.zeros(
                            signal_tensor.shape[0], dtype=torch.long, device=device
                        )
                edge_embeds = self.edge_embedding(edge_type_ids)  # [N, 32]
                # Apply dropout to edge embeddings to prevent overfitting
                if self.training:
                    edge_embeds = F.dropout(edge_embeds, p=0.1, training=True)
            else:
                edge_embeds = torch.zeros((signal_tensor.shape[0], 32), device=device)

            # Optional lightweight eval diagnostics for modality activity (rate-limited)
            if not self.training and torch.rand(1).item() < 0.001:  # 0.1% instead of 2%
                try:
                    n_sigs = signal_tensor.shape[0]
                    if n_sigs >= 2:  # only log when overlap is expected
                        logging.info(
                            f"   [n={n_sigs}] meta:{metadata_features.std():.2e} overlap:{overlap_features.std():.2e} temp:{temporal_features.std():.2e} edge:{edge_embeds.std():.2e} snr:{snr_embedded.std():.2e}"
                        )
                except Exception:
                    pass

            # Fusion
            fused = self.modal_fusion(
                metadata_features, overlap_features, temporal_features, edge_embeds
            )  # [N, 64]

            # Heads
            prio = self.priority_head(fused).squeeze(-1)  # [N], linear (no sigmoid)
            
            # Apply clamped affine transform (Nov 13 fix: prevent saturation)
            # Use instance variables set in __init__
            gain_clamped = torch.clamp(self.prio_gain, min=self.affine_gain_min, max=self.affine_gain_max)
            bias_clamped = torch.clamp(self.prio_bias, min=self.affine_bias_min, max=self.affine_bias_max)
            prio = prio * gain_clamped + bias_clamped
            
            # NOTE (Nov 19 FIX): REMOVED hard clipping here!
            # Hard clipping at inference killed gradients during training:
            # - When loss wanted pred_max to reach 0.95, affine gained > 2.5
            # - Clipping to 1.0 destroyed the gradient signal
            # - Calibration loss had no effect (clipping eliminated all variation)
            # 
            # New strategy: Use soft penalty in loss function (bounds_penalty)
            # instead of hard clipping. Loss handles bounds; forward pass allows unclamped output.
            
            sigma = self.uncertainty_head(fused).squeeze(-1)  # [N], positive

            return prio, sigma

        except Exception as e:
            logging.error(f"Forward pass error: {e}")
            n = len(detections)
            return torch.zeros(n, device=device), torch.ones(n, device=device)

    def rank_detections(
        self, detections: List[Dict], strain_segments: Optional[torch.Tensor] = None
    ) -> List[int]:
        """
        Rank detections by an uncertainty-penalized priority score.

        The score is priorities − β·uncertainty, where β is larger for dense overlaps
        (n ≥ 5) to discount uncertain picks in difficult scenes.

        Args:
            detections: List of detection dicts matching the forward API.
            strain_segments: Optional strain tensor forwarded to the encoder.

        Returns:
            A list of indices sorted by descending score (best-first).
        """
        if not detections:
            return []
        try:
            with torch.no_grad():
                priorities, uncertainties = self.forward(detections, strain_segments)
                if priorities.numel() == 0 or len(priorities) != len(detections):
                    return list(range(len(detections)))
                n = len(detections)
                beta = 0.25 if n >= 5 else 0.10  # stronger penalty for 5+
                scores = priorities - beta * uncertainties
                return torch.argsort(scores, descending=True).tolist()
        except Exception as e:
            logging.warning(f"Ranking failed: {e}")
            return self._snr_fallback_ranking(detections)

    def _detections_to_tensor(self, detections: List[Dict]) -> torch.Tensor:
        """
        Convert detection dictionaries to a normalized tensor of features.

        Extracts a fixed 16-D parameter vector per detection (added network_snr),
        denormalizes when needed, clamps to valid ranges, and normalizes to [0, 1],
        with safe fallbacks.

        Args:
            detections: List of detection dicts possibly containing nested 'median'/'mean' values.

        Returns:
            Tensor [N, 16] of normalized features suitable for the feature extractor.
        """
        param_names = [
            "mass_1",
            "mass_2",
            "luminosity_distance",
            "ra",
            "dec",
            "geocent_time",
            "theta_jn",
            "psi",
            "phase",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "phi_12",
            "phi_jl",
            "network_snr",  # ← NEW: 16th feature
        ]
        ranges = {
            "mass_1": (5.0, 100.0),
            "mass_2": (5.0, 100.0),
            "luminosity_distance": (50.0, 800.0),  # ← CRITICAL FIX (Nov 15): Narrowed to 750 Mpc span for 4.2× higher sensitivity vs 150-1500 (1450 span). 33% change (150→200) now = 6.7% of normalized range vs 3.4% before
            "ra": (0.0, 2 * np.pi),
            "dec": (-np.pi / 2, np.pi / 2),
            "geocent_time": (-0.1, 0.1),
            "theta_jn": (0.0, np.pi),
            "psi": (0.0, np.pi),
            "phase": (0.0, 2 * np.pi),
            "a_1": (0.0, 0.99),
            "a_2": (0.0, 0.99),
            "tilt_1": (0.0, np.pi),
            "tilt_2": (0.0, np.pi),
            "phi_12": (0.0, 2 * np.pi),
            "phi_jl": (0.0, 2 * np.pi),
            "network_snr": (0.0, 1.0),  # ← FIXED: SNR now pre-normalized to [0,1]
        }
        defaults = {
            "mass_1": 35.0,
            "mass_2": 30.0,
            "luminosity_distance": 500.0,
            "ra": 1.0,
            "dec": 0.0,
            "geocent_time": 0.0,
            "theta_jn": np.pi / 2,
            "psi": 0.0,
            "phase": 0.0,
            "a_1": 0.0,
            "a_2": 0.0,
            "tilt_1": 0.0,
            "tilt_2": 0.0,
            "phi_12": 0.0,
            "phi_jl": 0.0,
            "network_snr": 0.43,  # ← FIXED: default mid-range SNR (15/35 ≈ 0.43)
        }
        try:
            tensor_data = []
            for detection in detections:
                vals = []
                for name in param_names:
                    try:
                        v = detection.get(name, defaults[name])
                        if isinstance(v, dict):
                            v = v.get("median", v.get("mean", defaults[name]))
                        v = float(v)
                        if not np.isfinite(v):
                            v = defaults[name]

                        # ← CRITICAL FIX (Nov 15): Use log-scale for distance to capture physics (SNR ∝ 1/d)
                        # Log-space encoding ensures small distances get high sensitivity (steep gradient)
                        if name == "luminosity_distance":
                            # Use log-scale: log10(d) maps [50, 800] → [1.7, 2.9] → [0, 1] via normalization
                            # This makes SNR changes (inversely proportional to distance) very sensitive
                            # Example: 150→200 Mpc is (log10(200)-log10(150))/(log10(800)-log10(50)) = 0.064/1.2 = 5.3% (vs 3.4% linear)
                            v_log = np.log10(max(v, 50.0))
                            log_min = np.log10(50.0)   # 1.699
                            log_max = np.log10(800.0)  # 2.903
                            norm = (v_log - log_min) / (log_max - log_min)
                            vals.append(np.clip(norm, 0.0, 1.0))
                            continue  # Skip the general normalization below

                        # ← FIXED (Nov 19): Enhanced SNR scaling for fine-grained discrimination
                        if name == "network_snr":
                            # Use tighter upper bound (25 instead of 35) for better resolution of close SNRs
                            # Example: SNR 14.5-15.0 differs by 0.5 → in [0,25] that's 2% vs 0.143% in [0,35]
                            # 14.0x improvement in sensitivity for close values
                            v = min(float(v), 25.0) / 25.0
                            vals.append(np.clip(v, 0.0, 1.0))
                            continue  # Skip the general normalization below

                        mn, mx = ranges[name]
                        norm = (v - mn) / (mx - mn)
                        vals.append(np.clip(norm, 0.0, 1.0))
                    except:
                        mn, mx = ranges[name]
                        dv = defaults[name]
                        norm = (dv - mn) / (mx - mn)
                        vals.append(np.clip(norm, 0.0, 1.0))
                tensor_data.append(vals)
            return torch.tensor(tensor_data, dtype=torch.float32)
        except Exception as e:
            logging.error(f"Tensor conversion failed: {e}")
            n = len(detections)
            return torch.full((n, 16), 0.5, dtype=torch.float32)  # ← NEW: 16 dims

    def _snr_fallback_ranking(self, detections: List[Dict]) -> List[int]:
            """
            Fallback ranking by available SNR-like fields when model ranking fails.

            Args:
                detections: List of detection dicts possibly containing 'network_snr', 'target_snr', or 'snr'.

            Returns:
                Indices sorted by descending SNR proxy; identity order on failure.
            """
            try:
                snr_scores = []
                for i, detection in enumerate(detections):
                    snr = detection.get(
                        "network_snr", detection.get("target_snr", detection.get("snr", 15.0))
                    )
                    snr = float(snr) if np.isfinite(snr) else 0.0
                    snr_scores.append((i, snr))
                snr_scores.sort(key=lambda x: x[1], reverse=True)
                return [idx for idx, _ in snr_scores]
            except:
                return list(range(len(detections)))


class PriorityNetTrainer:
    """
    Trainer for PriorityNet with warmup, ReduceLROnPlateau, and composite loss.

    Handles optimizer/scheduler setup, gradient clipping, batch-wise training over
    variable-length scenarios, and exposes an affine calibration mode to fine-tune
    a global gain/bias for rapid scale alignment when MAE plateaus.
    """

    def __init__(self, model: PriorityNet, config=None):
        """
        Initialize the trainer components from configuration.

        Args:
            model: An instance of PriorityNet to be trained.
            config: Dict/object providing learning rate, weight decay, warmup and scheduler settings,
                    loss weights (ranking/mse/uncertainty), label_smoothing, gradient_clip_norm.
        """
        self.model = model
        self.config = config
        self.current_epoch = 0

        def get_config(key, default):
            if hasattr(config, "get"):
                return config.get(key, default)
            return getattr(config, key, default) if config is not None else default

        lr = get_config("learning_rate", 5e-4)
        weight_decay = get_config("weight_decay", 1e-5)
        self.warmup_epochs = get_config("warmup_epochs", 5)
        warmup_start_factor = get_config("warmup_start_factor", 0.1)
        scheduler_patience = get_config("scheduler_patience", 5)
        scheduler_factor = get_config("scheduler_factor", 0.5)
        scheduler_threshold = float(get_config("scheduler_threshold", 1e-3))
        min_lr = get_config("min_lr", 1e-6)

        # Loss weights and smoothing
        self.ranking_weight = get_config("ranking_weight", 0.80)  # Updated Nov 24: 0.7 → 0.80
        self.mse_weight = get_config("mse_weight", 0.05)  # Updated Nov 24: 0.2 → 0.05
        self.uncertainty_weight = get_config("uncertainty_weight", 0.15)  # Updated Nov 24: 0.35 → 0.15
        self.use_snr_weighting = get_config("use_snr_weighting", True)
        self.label_smoothing = get_config("label_smoothing", 0.0)
        lambda_calib = get_config("lambda_calib", 1e-4)
        
        # Uncertainty bounds (Nov 13 fix)
        uncertainty_lower_bound = get_config("uncertainty_lower_bound", 0.01)
        uncertainty_upper_bound = get_config("uncertainty_upper_bound", 0.50)
        uncertainty_bounds_weight = get_config("uncertainty_bounds_weight", 0.05)
        
        # Calibration weights for range expansion (Nov 15 DEEP FIX)
        calib_mean_weight = get_config("calib_mean_weight", 0.30)
        calib_max_weight = get_config("calib_max_weight", 2.50)
        calib_range_weight = get_config("calib_range_weight", 2.00)

        self.gradient_clip_norm = get_config("gradient_clip_norm", 1.0)

        # Class-based loss with σ calibration and bucket-5 emphasis
        self.loss_fn = PriorityLoss(
            ranking_weight=self.ranking_weight,
            mse_weight=self.mse_weight,
            uncertainty_weight=self.uncertainty_weight,
            use_snr_weighting=self.use_snr_weighting,
            label_smoothing=self.label_smoothing,
            lambda_calib=lambda_calib,
            uncertainty_lower_bound=uncertainty_lower_bound,
            uncertainty_upper_bound=uncertainty_upper_bound,
            uncertainty_bounds_weight=uncertainty_bounds_weight,
            calib_mean_weight=calib_mean_weight,
            calib_max_weight=calib_max_weight,
            calib_range_weight=calib_range_weight,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Warmup scheduler (stepped in outer loop)
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )

        # Main scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=scheduler_threshold,  # configurable threshold
            threshold_mode="rel",  # relative improvement required
            cooldown=0,
            min_lr=min_lr,
        )
        self.main_scheduler = self.scheduler

        logging.info(f"✅ Trainer initialized:")
        logging.info(f"   LR: {lr:.2e}")
        logging.info(f"   Weight decay: {weight_decay:.2e}")
        logging.info(f"   Warmup epochs: {self.warmup_epochs}")
        logging.info(f"   Warmup start factor: {warmup_start_factor}")
        logging.info(
            f"   Scheduler patience: {scheduler_patience}, threshold: {scheduler_threshold:.2e}"
        )
        logging.info(f"   Min LR: {min_lr:.2e}")
        logging.info(
            f"   Loss weights: R={self.ranking_weight}, M={self.mse_weight}, U={self.uncertainty_weight}"
        )
        logging.info(
            f"   Uncertainty bounds: [{uncertainty_lower_bound:.3f}, {uncertainty_upper_bound:.3f}], weight={uncertainty_bounds_weight:.3f}"
        )
        logging.info(f"   Gradient clip: {self.gradient_clip_norm}")

    def set_affine_calibration(
        self, enable: bool, base_lr: Optional[float] = None, weight_decay: Optional[float] = None
    ):
        """
        Enter or exit affine calibration mode on the model's priority head.

        When enabled, freezes all model parameters except prio_gain and prio_bias, and
        rebuilds the optimizer to update only these two parameters (using current LR).
        When disabled, restores the full optimizer over all parameters from the config.

        Args:
            enable: True to enter affine mode, False to exit and restore full training.
            base_lr: Optional LR to use in affine mode; defaults to current optimizer LR.
            weight_decay: Optional weight decay to use in affine mode (defaults to 0.0).
        """
        # Freeze/unfreeze all params
        for p in self.model.parameters():
            p.requires_grad = not enable

        # Always allow affine params
        self.model.prio_gain.requires_grad = True
        self.model.prio_bias.requires_grad = True

        if enable:
            # Use current LR; do not force an increase
            lr = base_lr if base_lr is not None else self.optimizer.param_groups[0]["lr"]
            wd = 0.0 if weight_decay is None else weight_decay
            self.optimizer = torch.optim.AdamW(
                [self.model.prio_gain, self.model.prio_bias], lr=lr, weight_decay=wd
            )
        else:
            # Restore full optimizer from config (dict or object)
            get_cfg = (
                self.config.get
                if hasattr(self.config, "get")
                else lambda k, d: getattr(self.config, k, d)
            )
            lr = get_cfg("learning_rate", 7e-4)
            wd = get_cfg("weight_decay", 1.5e-5)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def train_step(
        self,
        detections_batch: List[List[Dict]],
        priorities_batch: List[torch.Tensor],
        strain_batch: Optional[List[torch.Tensor]] = None,
        edge_type_ids_batch: Optional[List[torch.Tensor]] = None,
        snr_values_batch: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Perform a single training step over a batch of scenarios.

        Accumulates loss over variable-length scenarios, backpropagates once,
        applies gradient clipping, and updates optimizer.

        Args:
            detections_batch: List of lists of detection dicts per scenario.
            priorities_batch: List of tensors of target priorities per scenario.
            strain_batch: Optional list of strain tensors per scenario.
            edge_type_ids_batch: Optional list of edge-type id tensors per scenario.
            snr_values_batch: Optional list of per-signal SNR tensors per scenario.

        Returns:
            Dict of scalar metrics averaged over valid scenarios in the batch.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_losses = {"total": 0.0, "mse": 0.0, "ranking": 0.0}
        valid_batches = 0

        # CONSISTENT accumulator for a single backward at the end
        accumulated_loss_tensor = None

        for i, (detections, target_priorities) in enumerate(
            zip(detections_batch, priorities_batch)
        ):
            if not detections or len(target_priorities) == 0:
                continue
            try:
                strain_segments = None if strain_batch is None else strain_batch[i]
                edge_ids = None if edge_type_ids_batch is None else edge_type_ids_batch[i]

                preds, sigma = self.model(detections, strain_segments, edge_type_ids=edge_ids)

                m = min(len(preds), len(target_priorities))
                if m == 0:
                    continue
                preds = preds[:m]
                targets = target_priorities[:m].to(preds.device)
                sigma = sigma[:m]

                snr_vals = None if snr_values_batch is None else snr_values_batch[i]
                if snr_vals is not None:
                    snr_vals = snr_vals[:m].to(preds.device)

                losses = self.loss_fn(preds, targets, sigma, snr_values=snr_vals)
                loss = losses["total"]

                # Accumulate graph-carrying loss tensor
                if accumulated_loss_tensor is None:
                    accumulated_loss_tensor = loss
                else:
                    accumulated_loss_tensor = accumulated_loss_tensor + loss

                # Track scalars for logging
                total_losses["total"] += float(loss.detach().cpu())
                total_losses["mse"] += float(losses["mse"].detach().cpu())
                total_losses["ranking"] += float(losses["ranking"].detach().cpu())
                valid_batches += 1

            except Exception as e:
                logging.debug(f"train_step error on batch {i}: {e}")
                continue

        grad_norm = 0.0
        if valid_batches > 0 and accumulated_loss_tensor is not None:
            avg_loss_tensor = accumulated_loss_tensor / valid_batches
            avg_loss_tensor.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            )
            self.optimizer.step()

        # Log affine parameters (Nov 19: track range expansion capability)
        with torch.no_grad():
            gain_val = float(self.model.prio_gain.detach().cpu())
            bias_val = float(self.model.prio_bias.detach().cpu())

        return {
            "loss": total_losses["total"] / max(1, valid_batches),
            "mse": total_losses["mse"] / max(1, valid_batches),
            "ranking_loss": total_losses["ranking"] / max(1, valid_batches),
            "priority_loss": total_losses["mse"] / max(1, valid_batches),
            "uncertainty": 0.0,
            "grad_norm": grad_norm,
            "valid_batches": valid_batches,
            "affine_gain": gain_val,  # NEW: Track affine gain training
            "affine_bias": bias_val,  # NEW: Track affine bias training
        }


# Backward compatibility: alias old class names
PriorityNet = PriorityNet
PriorityNetTrainer = PriorityNetTrainer