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
    """CNN + BiLSTM + Attention encoder for whitened strain segments."""

    def __init__(self, input_length: int = 2048, n_detectors: int = 2, hidden_dim: int = 128):
        super().__init__()

        # Multi-scale CNN for time-frequency features
        # Architecture based on LIGO ML pipelines (2048 Hz, 1s segments)
        self.conv_blocks = nn.ModuleList([
            self._conv_block(n_detectors, 32, kernel_size=64, stride=4),
            self._conv_block(32, 64, kernel_size=32, stride=4),
            self._conv_block(64, 128, kernel_size=16, stride=2),
            self._conv_block(128, 128, kernel_size=8, stride=2),
        ])

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
            dropout=0.2
        )

        # Multi-head self-attention for long-range dependencies
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Projection to fixed dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )

    def _conv_block(self, in_channels, out_channels, kernel_size, stride):
        """Convolutional block with normalization."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, strain_segments: torch.Tensor) -> torch.Tensor:
        """
        Args:
            strain_segments: [batch, n_detectors, time_samples]
        Returns:
            encoded: [batch, 64] temporal features
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
    """Computes pairwise overlap features for multi-signal scenarios."""

    def __init__(self):
        super().__init__()

        # Learnable overlap feature extractor
        self.overlap_net = nn.Sequential(
            nn.Linear(8, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16)
        )

    def forward(self, params_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            params_batch: [n_signals, 15] normalized parameters
        Returns:
            overlap_features: [n_signals, 16] cross-signal features
        """
        n_signals = params_batch.shape[0]

        if n_signals < 2:
            # No overlap for single signal
            return torch.zeros(n_signals, 16, device=params_batch.device)

        overlap_scores = []

        for i in range(n_signals):
            # Compute overlap metrics with all other signals
            pairwise_features = []

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

                mc_i = (m1_i * m2_i)**(3/5) / (m1_i + m2_i)**(1/5)
                mc_j = (m1_j * m2_j)**(3/5) / (m1_j + m2_j)**(1/5)
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

                pairwise_features.append(torch.stack([
                    dt, sky_sep, mass_similarity, freq_overlap, 
                    dist_ratio, dpsi, dra, ddec
                ]))

            if len(pairwise_features) > 0:
                # Average overlap with all other signals
                pairwise_tensor = torch.stack(pairwise_features)
                mean_overlap = torch.mean(pairwise_tensor, dim=0)
                overlap_scores.append(mean_overlap)
            else:
                overlap_scores.append(torch.zeros(8, device=params_batch.device))

        overlap_tensor = torch.stack(overlap_scores)
        overlap_features = self.overlap_net(overlap_tensor)

        return overlap_features


class SignalFeatureExtractor(nn.Module):
    """Enhanced feature extractor with deeper architecture and layer normalization."""

    def __init__(self, input_dim: int = 15):
        super().__init__()

        # Deeper physics-aware feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(15, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32)
        )

        # Physics-derived features extractor
        self.physics_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 16)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 15] normalized parameters
        Returns:
            features: [batch, 48] (32 from network + 16 from physics)
        """
        # Network-learned features
        network_features = self.feature_processor(x)

        # Physics-derived features
        physics_features = self._compute_physics_features(x)
        physics_encoded = self.physics_encoder(physics_features)

        # Concatenate both feature types
        combined = torch.cat([network_features, physics_encoded], dim=1)

        return combined

    def _compute_physics_features(self, params: torch.Tensor) -> torch.Tensor:
        """Compute derived physics quantities."""

        batch_size = params.shape[0]
        physics_features = torch.zeros(batch_size, 8, device=params.device)

        try:
            # Extract basic parameters (denormalized)
            m1 = params[:, 0] * 95 + 5  # [5, 100] solar masses
            m2 = params[:, 1] * 95 + 5
            distance = params[:, 2] * 2950 + 50  # [50, 3000] Mpc

            # Derived quantities
            total_mass = m1 + m2
            chirp_mass = (m1 * m2)**(3/5) / total_mass**(1/5)
            mass_ratio = torch.minimum(m1, m2) / torch.maximum(m1, m2)
            eta = m1 * m2 / (total_mass**2)

            # SNR estimation (approximate)
            estimated_snr = 8.0 * (chirp_mass / 30.0)**(5/6) * (400.0 / torch.clamp(distance, min=50.0))

            # Frequency estimates
            f_isco = 220.0 / total_mass

            # Effective spin (if available)
            if params.shape[1] >= 11:
                a1 = params[:, 9] * 0.99  # [0, 0.99]
                a2 = params[:, 10] * 0.99
                chi_eff = (m1 * a1 + m2 * a2) / total_mass
            else:
                chi_eff = torch.zeros_like(m1)

            # Detection difficulty score
            difficulty = torch.log(distance / 100.0) - torch.log(estimated_snr / 10.0 + 1e-6)

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


class EnhancedPriorityNet(nn.Module):
    """Enhanced PriorityNet with temporal encoding and uncertainty quantification."""

    def __init__(self, config=None, use_strain: bool = True):
        super().__init__()

        self.config = config or self._default_config()
        self.use_strain = use_strain

        # Temporal strain encoder (optional, for when strain data is available)
        if self.use_strain:
            self.strain_encoder = TemporalStrainEncoder(
                input_length=2048,  # 1s at 2048 Hz
                n_detectors=2,      # H1, L1 (can extend to 3 for Virgo)
                hidden_dim=128
            )
            temporal_dim = 64
        else:
            temporal_dim = 0

        # Cross-signal overlap analyzer
        self.cross_signal_analyzer = CrossSignalAnalyzer()

        # Enhanced metadata feature extraction
        self.signal_encoder = SignalFeatureExtractor()

        # Fusion layer (combines temporal + metadata + overlap features)
        fusion_input_dim = 48 + 16 + temporal_dim  # metadata + overlap + temporal

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Priority prediction head (outputs priority + uncertainty)
        self.priority_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 2)  # [priority, log_uncertainty]
        )

        logging.info(f"Enhanced PriorityNet initialized (use_strain={use_strain})")

    def _default_config(self):
        return type('Config', (), {
            'hidden_dims': [256, 128, 64],
            'dropout': 0.1,
            'learning_rate': 1e-3,
            'use_strain': True
        })()

    # def forward(self, detections: List[Dict], strain_segments: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Forward pass with uncertainty quantification.

    #     Args:
    #         detections: List of detection dictionaries with parameters
    #         strain_segments: Optional [n_signals, n_detectors, time_samples] whitened strain

    #     Returns:
    #         priorities: [n_signals] priority scores
    #         uncertainties: [n_signals] uncertainty estimates
    #     """

    #     if not detections:
    #         return torch.empty(0), torch.empty(0)

    #     try:
    #         # Convert metadata to tensor
    #         signal_tensor = self._detections_to_tensor(detections)

    #         if signal_tensor is None or signal_tensor.numel() == 0:
    #             n = len(detections)
    #             return torch.zeros(n), torch.ones(n)

    #         # Extract metadata features
    #         metadata_features = self.signal_encoder(signal_tensor)

    #         # Extract cross-signal overlap features
    #         overlap_features = self.cross_signal_analyzer(signal_tensor)

    #         # Extract temporal features from strain (if available)
    #         if self.use_strain and strain_segments is not None:
    #             temporal_features = self.strain_encoder(strain_segments)
    #         else:
    #             temporal_features = torch.zeros(
    #                 signal_tensor.shape[0], 0, 
    #                 device=signal_tensor.device
    #             )

    #         # Fuse all features
    #         if temporal_features.shape[1] > 0:
    #             combined_features = torch.cat([
    #                 metadata_features, overlap_features, temporal_features
    #             ], dim=1)
    #         else:
    #             combined_features = torch.cat([
    #                 metadata_features, overlap_features
    #             ], dim=1)


    #         print(f"metadata_features.shape = {metadata_features.shape}")
    #         print(f"overlap_features.shape = {overlap_features.shape}")
    #         print(f"temporal_features.shape = {temporal_features.shape}")
    #         fused = self.fusion_layer(combined_features)

    #         # Predict priority and uncertainty
    #         output = self.priority_head(fused)
    #         priorities = torch.sigmoid(output[:, 0])  # [0, 1]
    #         log_uncertainties = output[:, 1]
    #         uncertainties = torch.exp(log_uncertainties)  # Convert to positive

    #         return priorities, uncertainties

    #     except Exception as e:
    #         logging.error(f"Forward pass error: {e}")
    #         n = len(detections)
    #         return torch.zeros(n), torch.ones(n)


    def forward(self, detections: List[Dict], strain_segments: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty quantification.

        Args:
            detections: List of detection dictionaries with parameters
            strain_segments: Optional [n_signals, n_detectors, time_samples] whitened strain

        Returns:
            priorities: [n_signals] priority scores
            uncertainties: [n_signals] uncertainty estimates
        """
        
        if not detections:
            return torch.empty(0), torch.empty(0)

        try:
            signal_tensor = self._detections_to_tensor(detections)
            if signal_tensor is None or signal_tensor.numel() == 0:
                n = len(detections)
                return torch.zeros(n), torch.ones(n)

            metadata_features = self.signal_encoder(signal_tensor)
            overlap_features = self.cross_signal_analyzer(signal_tensor)

            if self.use_strain and strain_segments is not None and strain_segments.shape[1] > 0:
                temporal_features = self.strain_encoder(strain_segments)
                if temporal_features.shape[1] == 0:
                    # Replace zero-dim temporal_features with zeros tensor of shape [batch_size, 64]
                    temporal_features = torch.zeros((signal_tensor.shape[0], 64), device=signal_tensor.device)
            else:
                temporal_features = torch.zeros((signal_tensor.shape[0], 64), device=signal_tensor.device)

            combined_features = torch.cat([metadata_features, overlap_features, temporal_features], dim=1)

            fused = self.fusion_layer(combined_features)
            output = self.priority_head(fused)
            priorities = torch.sigmoid(output[:, 0])
            log_uncertainties = output[:, 1]
            uncertainties = torch.exp(log_uncertainties)

            return priorities, uncertainties

        except Exception as e:
            logging.error(f"Forward pass error: {e}")
            n = len(detections)
            return torch.zeros(n), torch.ones(n)

    def rank_detections(self, detections: List[Dict], strain_segments: Optional[torch.Tensor] = None) -> List[int]:
        """Rank detections by priority, accounting for uncertainty."""

        if not detections:
            return []

        try:
            with torch.no_grad():
                priorities, uncertainties = self.forward(detections, strain_segments)

                if priorities.numel() == 0 or len(priorities) != len(detections):
                    return list(range(len(detections)))

                # Uncertainty-aware ranking: penalize high uncertainty
                # Use UCB-like approach: score = priority - beta * uncertainty
                beta = 0.1  # Exploration parameter
                scores = priorities - beta * uncertainties

                ranked_indices = torch.argsort(scores, descending=True).tolist()
                return ranked_indices

        except Exception as e:
            logging.warning(f"Ranking failed: {e}")
            return self._snr_fallback_ranking(detections)

    def _detections_to_tensor(self, detections: List[Dict]) -> torch.Tensor:
        """Convert detection dictionaries to tensor format."""

        param_names = [
            'mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
            'geocent_time', 'theta_jn', 'psi', 'phase', 'a_1', 'a_2',
            'tilt_1', 'tilt_2', 'phi_12', 'phi_jl'
        ]

        ranges = {
            'mass_1': (5.0, 100.0), 'mass_2': (5.0, 100.0),
            'luminosity_distance': (50.0, 3000.0),
            'ra': (0.0, 2*np.pi), 'dec': (-np.pi/2, np.pi/2),
            'geocent_time': (-0.1, 0.1),
            'theta_jn': (0.0, np.pi), 'psi': (0.0, np.pi),
            'phase': (0.0, 2*np.pi),
            'a_1': (0.0, 0.99), 'a_2': (0.0, 0.99),
            'tilt_1': (0.0, np.pi), 'tilt_2': (0.0, np.pi),
            'phi_12': (0.0, 2*np.pi), 'phi_jl': (0.0, 2*np.pi)
        }

        defaults = {
            'mass_1': 35.0, 'mass_2': 30.0, 'luminosity_distance': 500.0,
            'ra': 1.0, 'dec': 0.0, 'geocent_time': 0.0,
            'theta_jn': np.pi/2, 'psi': 0.0, 'phase': 0.0,
            'a_1': 0.0, 'a_2': 0.0, 'tilt_1': 0.0, 'tilt_2': 0.0,
            'phi_12': 0.0, 'phi_jl': 0.0
        }

        try:
            tensor_data = []

            for detection in detections:
                signal_params = []

                for param_name in param_names:
                    try:
                        value = detection.get(param_name, defaults[param_name])

                        if isinstance(value, dict):
                            value = value.get('median', value.get('mean', defaults[param_name]))

                        value = float(value)

                        if not np.isfinite(value):
                            value = defaults[param_name]

                        # Normalize to [0, 1]
                        min_val, max_val = ranges[param_name]
                        normalized = (value - min_val) / (max_val - min_val)
                        normalized = np.clip(normalized, 0.0, 1.0)

                        signal_params.append(normalized)

                    except:
                        min_val, max_val = ranges[param_name]
                        default_val = defaults[param_name]
                        normalized = (default_val - min_val) / (max_val - min_val)
                        signal_params.append(np.clip(normalized, 0.0, 1.0))

                tensor_data.append(signal_params)

            return torch.tensor(tensor_data, dtype=torch.float32)

        except Exception as e:
            logging.error(f"Tensor conversion failed: {e}")
            n_detections = len(detections)
            return torch.full((n_detections, 15), 0.5, dtype=torch.float32)

    def _snr_fallback_ranking(self, detections: List[Dict]) -> List[int]:
        """Fallback SNR-based ranking."""

        try:
            snr_scores = []
            for i, detection in enumerate(detections):
                snr = detection.get('network_snr', 0.0)
                if not np.isfinite(snr):
                    snr = 0.0
                snr_scores.append((i, snr))

            snr_scores.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in snr_scores]

        except:
            return list(range(len(detections)))


class AdaptiveRankingLoss(nn.Module):
    """Adaptive pairwise ranking loss with learned margins."""

    def __init__(self, base_margin: float = 0.1):
        super().__init__()
        self.base_margin = base_margin

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                uncertainties: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute adaptive ranking loss with uncertainty weighting.

        Args:
            predictions: [n] predicted priorities
            targets: [n] ground truth priorities
            uncertainties: [n] optional uncertainty estimates
        """
        n_samples = predictions.shape[0]

        if n_samples < 2:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        loss = 0.0
        count = 0

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Adaptive margin based on target separation
                target_diff = torch.abs(targets[i] - targets[j])
                margin = self.base_margin * torch.clamp(target_diff, min=0.1, max=1.0)

                # Uncertainty weighting (down-weight uncertain predictions)
                if uncertainties is not None:
                    weight = 1.0 / (1.0 + uncertainties[i] + uncertainties[j])
                else:
                    weight = 1.0

                # Ranking constraint
                if targets[i] > targets[j]:
                    diff = predictions[j] - predictions[i] + margin
                    loss += weight * torch.clamp(diff, min=0.0)
                    count += 1
                elif targets[j] > targets[i]:
                    diff = predictions[i] - predictions[j] + margin
                    loss += weight * torch.clamp(diff, min=0.0)
                    count += 1

        return loss / max(count, 1)


class EnhancedPriorityLoss(nn.Module):
    """Enhanced loss with adaptive ranking, MSE, and uncertainty regularization."""

    def __init__(self, ranking_weight: float = 0.5, mse_weight: float = 0.4, 
                 uncertainty_weight: float = 0.1):
        super().__init__()
        self.ranking_weight = ranking_weight
        self.mse_weight = mse_weight
        self.uncertainty_weight = uncertainty_weight
        self.ranking_loss_fn = AdaptiveRankingLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                uncertainties: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Combined loss with uncertainty regularization.

        Returns:
            Dict with total loss and individual components
        """
        # MSE component
        mse_loss = F.mse_loss(predictions, targets)

        # Adaptive ranking loss
        ranking_loss = self.ranking_loss_fn(predictions, targets, uncertainties)

        # Uncertainty regularization (penalize overconfidence and underconfidence)
        # Encourage uncertainty to be correlated with prediction error
        pred_error = torch.abs(predictions - targets)
        uncertainty_loss = F.mse_loss(uncertainties, pred_error.detach())

        # Combined loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.ranking_weight * ranking_loss +
                     self.uncertainty_weight * uncertainty_loss)

        return {
            'total': total_loss,
            'mse': mse_loss.detach(),
            'ranking': ranking_loss.detach(),
            'uncertainty': uncertainty_loss.detach()
        }


class EnhancedPriorityNetTrainer:
    """Trainer for Enhanced PriorityNet with improved stability."""

    def __init__(self, model: EnhancedPriorityNet, config=None):
        self.model = model
        self.config = config

        # Extract learning rate
        if config and hasattr(config, 'learning_rate'):
            lr = config.learning_rate
        else:
            lr = 1e-3

        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode = "min",
            factor = 0.5,
            patience = 15,
            min_lr = 1e-6
        )

        self.criterion = EnhancedPriorityLoss(
                ranking_weight=0.3,  # Reduced from 0.5
                mse_weight=0.6,      # Increased from 0.4
                uncertainty_weight=0.1
            )

    def train_step(self, detections_batch: List[List[Dict]], 
              priorities_batch: List[torch.Tensor],
              strain_batch: Optional[List[torch.Tensor]] = None) -> Dict[str, float]:
        """
        Single training step with optional strain data.
        
        Args:
            detections_batch: List of detection lists
            priorities_batch: List of target priority tensors
            strain_batch: Optional list of strain segment tensors
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        total_losses = {'total': 0.0, 'mse': 0.0, 'ranking': 0.0, 'uncertainty': 0.0}
        valid_batches = 0
        accumulated_loss_tensor = None
        
        for idx, (detections, target_priorities) in enumerate(zip(detections_batch, priorities_batch)):
            if not detections or len(target_priorities) == 0:
                continue
            
            try:
                # Get strain segments if available
                strain_segments = strain_batch[idx] if strain_batch is not None else None
                
                # Forward pass
                predicted_priorities, uncertainties = self.model(detections, strain_segments)
                
                if predicted_priorities.numel() == 0:
                    continue
                
                # Match lengths
                min_len = min(len(predicted_priorities), len(target_priorities))
                if min_len == 0:
                    continue
                
                pred_slice = predicted_priorities[:min_len]
                target_slice = target_priorities[:min_len].to(pred_slice.device)
                unc_slice = uncertainties[:min_len]
                
                # Compute loss
                losses = self.criterion(pred_slice, target_slice, unc_slice)
                
                if accumulated_loss_tensor is None:
                    accumulated_loss_tensor = losses['total']
                else:
                    accumulated_loss_tensor = accumulated_loss_tensor + losses['total']
                
                total_losses['total'] += float(losses['total'].item())
                total_losses['mse'] += float(losses['mse'].item())
                total_losses['ranking'] += float(losses['ranking'].item())
                total_losses['uncertainty'] += float(losses['uncertainty'].item())
                valid_batches += 1
                
            except Exception as e:
                logging.debug(f"Training step error: {e}")
                continue
        
        # Compute gradients and update
        grad_norm = 0.0
        if valid_batches > 0 and accumulated_loss_tensor is not None:
            avg_loss_tensor = accumulated_loss_tensor / valid_batches
            avg_loss_tensor.backward()
            
            # Gradient clipping and get norm
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
            
            self.optimizer.step()
            
            return {
                'loss': total_losses['total'] / valid_batches,
                'mse': float(total_losses['mse'] / valid_batches),
                'ranking_loss': float(total_losses['ranking'] / valid_batches),
                'priority_loss': float(total_losses['mse'] / valid_batches),
                'uncertainty': float(total_losses['uncertainty'] / valid_batches),
                'grad_norm': grad_norm,
                'valid_batches': valid_batches
            }
        else:
            return {
                'loss': 0.0,
                'mse': 0.0,
                'ranking_loss': 0.0,
                'priority_loss': 0.0,
                'uncertainty': 0.0,
                'grad_norm': 0.0,
                'valid_batches': 0
            }

    def train_epoch(self, data_loader) -> Dict[str, float]:
        """Train for one epoch."""

        epoch_losses = {'total': 0.0, 'mse': 0.0, 'ranking': 0.0, 'uncertainty': 0.0}
        num_batches = 0

        for batch in data_loader:
            # Handle different batch formats
            if len(batch) == 2:
                detections, priorities = batch
                strain_segments = None
            elif len(batch) == 3:
                detections, priorities, strain_segments = batch
            else:
                continue

            if not detections or len(priorities) == 0:
                continue

            loss_info = self.train_step(detections, priorities, strain_segments)

            epoch_losses['total'] += loss_info['loss']
            epoch_losses['mse'] += loss_info['mse']
            epoch_losses['ranking'] += loss_info['ranking']
            epoch_losses['uncertainty'] += loss_info['uncertainty']
            num_batches += 1

        # Step scheduler
        self.scheduler.step()

        if num_batches > 0:
            return {
                'epoch_loss': epoch_losses['total'] / num_batches,
                'epoch_mse': epoch_losses['mse'] / num_batches,
                'epoch_ranking': epoch_losses['ranking'] / num_batches,
                'epoch_uncertainty': epoch_losses['uncertainty'] / num_batches,
                'lr': self.optimizer.param_groups[0]['lr']
            }
        else:
            return {'epoch_loss': 0.0, 'epoch_mse': 0.0, 'epoch_ranking': 0.0, 
                   'epoch_uncertainty': 0.0, 'lr': 0.0}


# Backward compatibility: alias old class names
PriorityNet = EnhancedPriorityNet
PriorityNetTrainer = EnhancedPriorityNetTrainer
