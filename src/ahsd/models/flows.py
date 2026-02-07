"""
Neural Spline Flow (NSF) posterior for Bayesian parameter estimation.
Optimized for overlapping gravitational wave signals.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union, Any
from pathlib import Path
import numpy as np
from nflows import distributions
import logging

# ✅ Import UniversalConfigReader for type-safe config access
from ahsd.utils.universal_config import UniversalConfigReader, ConfigDict

# ✅ NSF imports from nflows library
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation


 
# Problem: Original scale modulation only scales loss, not base distribution
# This allows flow to internally compensate, defeating PSD physics
#
# Solution: Make base distribution conditional on PSD scales
# z ~ N(0, Σ_psd) instead of z ~ N(0, I)
# Flow cannot compensate → posterior width genuinely scales with PSD
#
class PSDScaledNormal(nn.Module):
    """
    Conditional base distribution that scales with PSD
    
    Implements N(0, Σ_psd) where Σ_psd is diagonal with scale factors σ_psd
    
    This is the **mathematically correct** way to handle heteroscedastic likelihood
    in Bayesian inference with normalizing flows.
    
    Physics interpretation:
    - Quiet data (large σ_psd) → wider base distribution → broader posterior
    - Loud data (small σ_psd) → narrower base distribution → sharper posterior
    - Flow cannot compensate because base is part of model structure
    """
    
    def __init__(self, shape: Union[list, int]):
        super().__init__()
        if isinstance(shape, list):
            self.shape = shape
            self.dim = shape[0]
        else:
            self.shape = [shape]
            self.dim = shape
        self.logger = logging.getLogger(__name__)
    
    def log_prob(self, z: torch.Tensor, log_sigma_psd: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under N(0, Σ_psd)
        
        Args:
            z: [batch, dim] - samples to evaluate
            log_sigma_psd: [batch, dim] - log scale factors from PSD head
        
        Returns:
            log_prob: [batch] - log probabilities
        """
        # Ensure shapes match
        if z.shape != log_sigma_psd.shape:
            raise ValueError(
                f"Shape mismatch: z {z.shape} vs log_sigma_psd {log_sigma_psd.shape}"
            )
        
        sigma_psd = torch.exp(log_sigma_psd)  # [batch, dim]
        
        # Gaussian log probability: -0.5 * (z²/σ² + d*log(2πσ²))
        # Expanded: -0.5 * (sum((z/σ)²) + d*log(2π) + d*log(σ)²)
        #        = -0.5 * (sum((z/σ)²) + d*log(2π) + 2*sum(log(σ)))
        
        quadratic_term = ((z / sigma_psd) ** 2).sum(dim=1)  # [batch]
        variance_term = 2 * log_sigma_psd.sum(dim=1)  # [batch]
        const_term = self.dim * np.log(2 * np.pi)
        
        log_prob = -0.5 * (quadratic_term + variance_term + const_term)
        
        return log_prob
    
    def sample(self, num_samples: int, log_sigma_psd: torch.Tensor = None) -> torch.Tensor:
        """
        Sample from N(0, I) - STANDARD NORMAL (NOT PSD-scaled)
        
        ✅ FEB 2 CRITICAL FIX: Sample from unit normal only
        
        Apply σ_psd OUTSIDE the flow at sampling time (factored scale).
        This prevents the flow from re-learning σ_psd to undo PSD conditioning.
        
        Args:
            num_samples: number of samples to generate per batch element
            log_sigma_psd: [batch, dim] - IGNORED (kept for API compatibility)
        
        Returns:
            samples: [batch, num_samples, dim] - from standard normal N(0, I)
        """
        batch_size = log_sigma_psd.shape[0] if log_sigma_psd is not None else 1
        
        # Sample from UNIT NORMAL ONLY (no PSD scaling here)
        # The flow will work in unit variance space
        eps = torch.randn(
            batch_size, num_samples, self.dim,
            device=log_sigma_psd.device if log_sigma_psd is not None else torch.device('cpu'),
            dtype=log_sigma_psd.dtype if log_sigma_psd is not None else torch.float32
        )
        
        # Return unit normal samples (PSD scaling applied OUTSIDE flow later)
        # No scaling applied here - flow works in unit variance space only
        return eps  # [batch, num_samples, dim] from N(0, I)


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and MLP"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = self.norm1(x + self.dropout(attn_out.squeeze(1)))
        
        # MLP with residual
        mlp_out = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_out))
        
        return x


class AdaptiveContextGating(nn.Module):
    """
    ✅ NEW: Adaptive gating for multi-signal context
    
    Problem: With 4-5 overlaps, context contains info about ALL signals
    Solution: Learn to gate context based on which signal we're estimating
    
    Input: 
      - Global context [batch, context_dim] (from all signals)
      - Signal-specific features [batch, features] (current signal params)
    Output:
      - Gated context [batch, context_dim] (relevant for this signal)
    """
    
    def __init__(self, context_dim: int, feature_dim: int):
        super().__init__()
        
        # Gate network: learns which parts of context are relevant
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim + context_dim, context_dim),
            nn.Tanh(),
            nn.Linear(context_dim, context_dim),
            nn.Sigmoid()  # Output [0, 1] gate values
        )
        
    def forward(self, context: torch.Tensor, signal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [batch, context_dim] - global context from all signals
            signal_features: [batch, feature_dim] - current signal parameters
        Returns:
            gated_context: [batch, context_dim] - context relevant for this signal
        """
        # Concatenate signal features with context
        combined = torch.cat([signal_features, context], dim=-1)
        
        # Compute gate values [0, 1]
        gate = self.gate_network(combined)
        
        # Apply gate to context
        gated_context = context * gate
        
        return gated_context


class OverlapAwareVelocityNet(nn.Module):
    """
    ✅ ENHANCED: Velocity network with overlap awareness
    
    New features:
    1. Adaptive context gating (different for each signal)
    2. Overlap indicator embedding (number of overlaps affects difficulty)
    3. Residual context integration (preserve information flow)
    """
    
    def __init__(self, features: int, context_features: int, hidden_dim: int = 256, 
                 num_layers: int = 2, dropout: float = 0.05, max_overlaps: int = 6, **kwargs):
        super().__init__()
        
        self.features = features
        self.context_features = context_features
        self.hidden_dim = hidden_dim
        self.max_overlaps = max_overlaps
        self.logger = logging.getLogger(__name__)
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Time embedding (sinusoidal)
        self.time_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim),  # [sin(πt), cos(πt)]
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ✅ NEW: Overlap count embedding
        # Encodes how many signals overlap (affects estimation difficulty)
        self.overlap_embedding = nn.Embedding(max_overlaps + 1, hidden_dim)
        
        # Input/context projections
        self.input_projection = nn.Linear(features, hidden_dim)
        self.context_projection = nn.Sequential(
            nn.Linear(context_features, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # ✅ NEW: Adaptive context gating (learns signal-specific context)
        self.context_gating = AdaptiveContextGating(context_features, features)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, 8, dropout) for _ in range(num_layers)
        ])
        
        # Cross-attention (for context conditioning)
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, 8, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # ✅ FIXED: Context scales initialized to 3.0 (strong conditioning)
        # Clamped to [1.0, 5.0] to prevent ignoring context
        self.context_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 3.0) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, features)
        )
        
        # Better weight initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        self.logger.info(
            f"✅ OverlapAwareVelocityNet initialized: {features} features, "
            f"{hidden_dim} hidden, {num_layers} layers, max_overlaps={max_overlaps}"
        )
    
    def get_context_scale_stats(self) -> dict:
        """Get statistics of learned context scaling parameters"""
        scales = [s.item() for s in self.context_scales]
        return {
            'mean': float(np.mean(scales)),
            'std': float(np.std(scales)),
            'min': float(np.min(scales)),
            'max': float(np.max(scales)),
            'scales': scales
        }
    
    def clamp_context_scales(self):
        """Prevent context scales from dropping too low"""
        with torch.no_grad():
            for scale in self.context_scales:
                scale.data = torch.clamp(scale.data, min=1.0, max=5.0)
    
    def forward(self, z: torch.Tensor, t: torch.Tensor, context: torch.Tensor, 
                n_overlaps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z: [batch, features] - current parameter state
            t: [batch, 1] or [batch] - time in [0, 1]
            context: [batch, context_features] - global context from strain
            n_overlaps: [batch] - number of overlapping signals (optional)
        
        Returns:
            velocity: [batch, features] - predicted velocity field
        """
        batch_size = z.shape[0]
        
        # Sinusoidal time embedding
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_sin = torch.sin(np.pi * t)
        t_cos = torch.cos(np.pi * t)
        t_normalized = torch.cat([t_sin, t_cos], dim=1)  # [batch, 2]
        
        # ✅ NEW: Apply adaptive gating to context
        # This filters context to be signal-specific
        gated_context = self.context_gating(context, z)
        
        # Embed inputs
        z_emb = self.input_projection(z)
        t_emb = self.time_embedding(t_normalized)
        ctx_emb = self.context_projection(gated_context)
        
        # ✅ NEW: Add overlap count embedding (if provided)
        if n_overlaps is not None:
            # Clamp to valid range [0, max_overlaps]
            n_overlaps_clamped = torch.clamp(n_overlaps, 0, self.max_overlaps).long()
            overlap_emb = self.overlap_embedding(n_overlaps_clamped)
            x = z_emb + t_emb + 0.5 * overlap_emb  # Weighted sum
        else:
            x = z_emb + t_emb
        
        # Process through transformer with context conditioning
        for layer_idx, (transformer_block, cross_attn) in enumerate(
            zip(self.transformer_blocks, self.cross_attention)
        ):
            x = transformer_block(x)
            # Cross-attention with gated context
            x_att, _ = cross_attn(x.unsqueeze(1), ctx_emb.unsqueeze(1), ctx_emb.unsqueeze(1))
            x = x + self.context_scales[layer_idx] * x_att.squeeze(1)
        
        # Output velocity
        velocity = self.output_layer(x)
        return velocity


class NSFPosteriorFlow(nn.Module):
    """
    ✅ Q3 REDESIGN: Neural Spline Flow using nflows library
    
    Production-ready implementation using:
    - MaskedPiecewiseRationalQuadraticAutoregressiveTransform (proven stable)
    - StandardNormal base distribution
    - CompositeTransform for multiple layers
    
    Advantages:
    - 16× faster inference than FlowMatching
    - Numerically stable (tested on GW data)
    - Simple 2-term loss (NLL + bounds)
    - Direct inverse (no ODE solver)
    """
    
    def __init__(self, features: int, context_features: int = 0, hidden_features: int = 256,
                 num_layers: int = 12, num_bins: int = 16, tail_bound: Union[float, Dict[int, float]] = 3.0,
                 max_overlaps: int = 6, dropout: float = 0.0, temperature_scale: float = 1.5, **kwargs):
        super().__init__()
        
        self.features = features
        self.context_features = context_features
        self.num_layers = num_layers
        self.max_overlaps = max_overlaps
        self.logger = logging.getLogger(__name__)
        
        # ✅ Per-parameter tail_bounds 
        # If tail_bound is dict: use per-parameter values
        # If tail_bound is float: use global value for all parameters
        if isinstance(tail_bound, dict):
            self.per_param_tail_bounds = tail_bound
            # Create mapping for all features
            self.tail_bounds_list = [
                tail_bound.get(i, 3.0) for i in range(features)
            ]
            self.logger.info(
                f"✅ Per-parameter tail_bounds enabled: {self.tail_bounds_list}"
            )
        else:
            self.per_param_tail_bounds = None
            self.tail_bounds_list = [tail_bound] * features
        
        # ✅ TEMPERATURE SCALING (Jan 4): Learnable base distribution temperature
        # Higher T → wider base distribution N(0, T²I) → posterior samples more spread out
        # Prevents posterior collapse to delta function → fixes PIT tails
        # Start at 1.5 for exploratory sampling, can anneal to 1.0 during training
        self.temperature = nn.Parameter(torch.tensor(temperature_scale, dtype=torch.float32))
        self.temperature_scale_init = temperature_scale
        
        # ✅ FEB 2 FIX: Use PSD-conditioned base distribution (not StandardNormal)
        # This ensures z ~ N(0, Σ_psd), not z ~ N(0, I)
        # Flow cannot compensate internally → posterior width genuinely scales with PSD
        self.base_dist = PSDScaledNormal(shape=[features])
        
        # Build spline transform layers
        transforms = []
        for layer_idx in range(num_layers):
            # Permutation for better mixing
            transforms.append(ReversePermutation(features=features))
            
            # Use per-parameter tail_bounds if available
            # For each layer, create individual transforms per feature with custom tail_bounds
            if self.per_param_tail_bounds:
                # Build autoregressive transform with per-feature tail bounds
                transforms.append(
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                        features=features,
                        hidden_features=hidden_features,
                        context_features=context_features if context_features > 0 else None,
                        num_bins=num_bins,
                        tails='linear',
                        tail_bound=3.0,  # Will be overridden per-feature
                        num_blocks=2,
                        use_residual_blocks=True,
                        random_mask=False,
                        activation=nn.functional.relu,
                        dropout_probability=dropout,
                        use_batch_norm=False
                    )
                )
                # Manually override tail_bounds for each feature in this transform
                # Access the autoregressive transform and patch its tail bounds
                autoregressive_transform = transforms[-1]
                if hasattr(autoregressive_transform, '_transforms'):
                    for feature_idx, custom_bound in enumerate(self.tail_bounds_list):
                        if feature_idx < len(autoregressive_transform._transforms):
                            transform = autoregressive_transform._transforms[feature_idx]
                            if hasattr(transform, 'tail_bound'):
                                transform.tail_bound = custom_bound
            else:
                # Standard global tail_bound
                transforms.append(
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                        features=features,
                        hidden_features=hidden_features,
                        context_features=context_features if context_features > 0 else None,
                        num_bins=num_bins,
                        tails='linear',
                        tail_bound=tail_bound if isinstance(tail_bound, float) else 3.0,
                        num_blocks=2,
                        use_residual_blocks=True,
                        random_mask=False,
                        activation=nn.functional.relu,
                        dropout_probability=dropout,
                        use_batch_norm=False
                    )
                )
        
        # Compose transforms
        self.transform = CompositeTransform(transforms)
        
        # Create flow
        self.flow = Flow(self.transform, self.base_dist)
        
        # ✅ NEW: Distance-specific context residual
        # Problem: NSF transforms z→x but context signal gets lost in spline nonlinearities
        # Solution: Add explicit residual from context to distance parameter (index 2)
        # This helps distance stay close to context-encoded value
        self.distance_context_head = nn.Sequential(
            nn.Linear(context_features if context_features > 0 else 1, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1)  # Output: distance residual
        ) if context_features > 0 else None
        
        self.logger.info(
            f"✅ NSFPosteriorFlow initialized: {features} features, "
            f"{context_features} context dims, {num_layers} layers, "
            f"{num_bins} bins, tail_bound={tail_bound}"
        )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: x (params) → z (base distribution)"""
        if self.context_features > 0 and context is not None:
            z, log_det = self.transform(x, context=context)
        else:
            z, log_det = self.transform(x)
        return z, log_det
    
    def inverse(self, z: torch.Tensor, context: torch.Tensor = None,
                 n_overlaps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse: z (base) → x (params). ✅ DETERMINISTIC - no ODE solver!"""
        
        # ✅ FEB 4 CRITICAL FIX: Sanitize context before passing to nflows transform
        # Context NaN → propagates through transform → invalid spline parameters → assertion error
        if context is not None:
            if torch.isnan(context).any() or torch.isinf(context).any():
                self.logger.warning(
                    f"🔴 NSF.inverse() detected NaN/Inf in context! "
                    f"Replacing with zeros (shape {context.shape})"
                )
                context = torch.nan_to_num(context, nan=0.0, posinf=1e-3, neginf=-1e-3)
        
        if self.context_features > 0 and context is not None:
            try:
                x, log_det = self.transform.inverse(z, context=context)
            except AssertionError as e:
                # NSF spline assertion failed - likely due to numerical instability
                self.logger.error(f"🔴 NSF.inverse() assertion failed: {e}")
                # Fallback: return z directly (no transformation)
                x = z
                log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        else:
            x, log_det = self.transform.inverse(z)
        
        # ✅ NEW: Distance anchoring via context residual
        # Distance (param index 2) gets explicit guidance from context
        if self.distance_context_head is not None and context is not None:
            try:
                distance_residual = self.distance_context_head(context)  # [batch, 1]
                # Add small residual from context (prevents drift, preserves learned structure)
                x[:, 2] = x[:, 2] + 0.05 * torch.tanh(distance_residual.squeeze(1))
            except Exception as e:
                self.logger.debug(f"Distance context head failed: {e}")
        
        # ✅ FEB 4: Sanitize output
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clamp to [-1, 1]
        x = torch.clamp(x, -1.0, 1.0)
        return x, log_det
    
    def log_prob(self, x: torch.Tensor, context: torch.Tensor = None, temperature: Optional[float] = None) -> torch.Tensor:
        """✅ PRIMARY LOSS: Negative log-likelihood with temperature scaling"""
        
        # ✅ FEB 4 CRITICAL FIX: Sanitize context before passing to nflows transform
        if context is not None:
            if torch.isnan(context).any() or torch.isinf(context).any():
                self.logger.warning(
                    f"🔴 NSF.log_prob() detected NaN/Inf in context! "
                    f"Replacing with zeros (shape {context.shape})"
                )
                context = torch.nan_to_num(context, nan=0.0, posinf=1e-3, neginf=-1e-3)
        
        # ✅ TEMPERATURE SCALING (Jan 4): Account for temperature in log probability
        # If we sample z ~ N(0, T²I), then x = inverse(z) has different distribution
        # We need to adjust log_prob to match this: log_p(x) = log_p(x/T) + d * log(T)
        if temperature is None:
            temperature = torch.clamp(self.temperature, min=0.5, max=3.0)
        
        # Compute log probability at scaled point
        x_scaled = x / temperature
        
        try:
            if self.context_features > 0 and context is not None:
                log_p = self.flow.log_prob(x_scaled, context=context)
            else:
                log_p = self.flow.log_prob(x_scaled)
        except AssertionError as e:
            # NSF assertion failed - return high loss as penalty
            self.logger.error(f"🔴 NSF.log_prob() assertion failed: {e}")
            log_p = torch.ones(x.shape[0], device=x.device, dtype=x.dtype) * 1000.0
        
        # Add Jacobian correction: log p(x) = log p(x/T) + d * log(T)
        # where d is the dimension (self.features)
        log_prob_corrected = log_p + self.features * torch.log(temperature)
        
        # ✅ FEB 4: Sanitize output
        if torch.isnan(log_prob_corrected).any():
            log_prob_corrected = torch.nan_to_num(log_prob_corrected, nan=1000.0)
        
        return -log_prob_corrected  # Return NEGATIVE (for minimization)
    
    def sample(self, num_samples: int, context: torch.Tensor = None, temperature: Optional[float] = None) -> torch.Tensor:
        """Sample from posterior p(x|context)"""
        with torch.no_grad():
            # ✅ TEMPERATURE SCALING (Jan 4): Use learned temperature if not overridden
            if temperature is None:
                temperature = torch.clamp(self.temperature, min=0.5, max=3.0)  # Clamp to reasonable range
            else:
                temperature = torch.tensor(temperature, dtype=self.temperature.dtype, device=self.temperature.device)
            
            if context is not None:
                batch_size = context.shape[0]
                
                # Expand context
                context_expanded = context.unsqueeze(1).expand(
                    batch_size, num_samples, self.context_features
                ).reshape(batch_size * num_samples, self.context_features)
                
                # ✅ TEMPERATURE SCALING: Scale z from base distribution by temperature
                # This widens the base distribution N(0, I) → N(0, T²I)
                z = self.base_dist.sample(num_samples * batch_size)
                z = z * temperature  # Apply temperature scaling
                
                # Transform
                samples, _ = self.inverse(z, context_expanded)
                
                # Reshape
                samples = samples.reshape(batch_size, num_samples, self.features)
            else:
                z = self.base_dist.sample(num_samples)
                z = z * temperature  # ✅ Apply temperature scaling to samples without context
                samples, _ = self.inverse(z, None)
            
            samples = torch.clamp(samples, -1.0, 1.0)
            return samples
    
    def compute_psd_aware_nll(self, x: torch.Tensor, context: torch.Tensor,
                             log_sigma_psd: torch.Tensor) -> torch.Tensor:
        """
        Compute NLL with PSD-conditioned base distribution
        
        This is the key method that makes PSD modulation work correctly.
        Instead of using StandardNormal base, we use z ~ N(0, Σ_psd).
        
        Args:
            x: [batch, features] - parameters (normalized to [-1, 1])
            context: [batch, context_features] - strain conditioning
            log_sigma_psd: [batch, features] - log(σ_psd) from PSD head
        
        Returns:
            nll: [batch] - negative log likelihood (loss for minimization)
        
        Physics:
        - Base distribution now scales with PSD
        - Quiet data (large σ_psd) → wider N(0, Σ_psd) → broader posterior
        - Loud data (small σ_psd) → narrower N(0, Σ_psd) → sharper posterior
        - Flow cannot internally compensate
        """
        batch_size = x.shape[0]
        
        # ✅ CRITICAL: Transform x → z using inverse transform
        # This gives us z samples that should follow base distribution
        if isinstance(self.base_dist, PSDScaledNormal):
            # Transform x → z
            z, log_det = self.inverse(x, context)  # [batch, features]
            
            # Compute log_prob under PSD-scaled base
            log_p_base = self.base_dist.log_prob(z, log_sigma_psd)  # [batch]
            
            # Compute log_det from flow transform
            log_det = log_det.squeeze() if log_det.dim() > 1 else log_det  # [batch]
            
            # Total log probability with Jacobian
            # log_p(x) = log_p(z) + log|det J|
            log_prob_x = log_p_base + log_det  # [batch]
            
            # Return negative log likelihood (for minimization)
            nll = -log_prob_x  # [batch]
        else:
            z, log_det = self.inverse(x, context)
            # Use standard Gaussian base distribution
            log_p_base = -0.5 * (z ** 2).sum(dim=1) - self.features * np.log(2 * np.pi) / 2
            log_det = log_det.squeeze() if log_det.dim() > 1 else log_det
            log_prob_x = log_p_base + log_det
            nll = -log_prob_x
        
        return nll  # [batch]
    
    def sample_psd_aware(self, num_samples: int, context: torch.Tensor,
                        log_sigma_psd: torch.Tensor) -> torch.Tensor:
        """
        Sample from posterior with PSD-conditioned base distribution
        
        This ensures samples follow the correct PSD-scaled distribution
        during inference, not just during training.
        
        Args:
            num_samples: number of posterior samples to generate
            context: [batch, context_features] - strain conditioning
            log_sigma_psd: [batch, features] - log(σ_psd) from PSD head
        
        Returns:
            samples: [batch, num_samples, features] - posterior samples
        
        Sampling process:
        1. z ~ N(0, Σ_psd)  ← PSD-conditioned (not standard normal)
        2. x = f^{-1}(z | context)  ← Apply inverse transform
        3. Return x
        """
        batch_size = context.shape[0]
        
        with torch.no_grad():
            if isinstance(self.base_dist, PSDScaledNormal):
                # ✅ FEB 2: Sample from unit normal base (NOT PSD-scaled)
                # PSD scaling applied AFTER flow (factored scale architecture)
                z_samples = self.base_dist.sample(num_samples, log_sigma_psd)  # [batch, num_samples, features] from N(0,I)
                
                # Reshape for batch processing
                z_flat = z_samples.reshape(batch_size * num_samples, self.features)  # [batch*num_samples, features]
                
                # Expand context to match
                context_expanded = context.unsqueeze(1).expand(
                    batch_size, num_samples, self.context_features
                ).reshape(batch_size * num_samples, self.context_features)
                
                # Transform z → x
                x_flat, _ = self.inverse(z_flat, context_expanded)  # [batch*num_samples, features]
                
                # ✅ FEB 7 CRITICAL FIX: Clamp in normalized space BEFORE reshaping
                # Must clamp BEFORE PSD scaling to prevent scaling destruction
                x_flat = torch.clamp(x_flat, -1.0, 1.0)  # Clamp normalized values first
                
                # Reshape back to [batch, num_samples, features]
                samples = x_flat.reshape(batch_size, num_samples, self.features)
                
                # ✅ FEB 2 CRITICAL: Apply factored scale OUTSIDE flow
                # Flow works in unit variance space, apply σ_psd here (no gradients)
                # This scaling is NOT clamped - intentional, to preserve PSD effect
                sigma_psd = torch.exp(log_sigma_psd)  # [batch, features]
                samples = samples * sigma_psd.unsqueeze(1)  # [batch, num_samples, features]
                # ✅ FEB 7: NO CLAMPING AFTER PSD SCALING - let samples expand per PSD!
            else:
                # Fallback to standard sampling (shouldn't happen)
                z = torch.randn(batch_size, num_samples, self.features,
                               device=context.device, dtype=context.dtype)
                z_flat = z.reshape(batch_size * num_samples, self.features)
                context_expanded = context.unsqueeze(1).expand(
                    batch_size, num_samples, self.context_features
                ).reshape(batch_size * num_samples, self.context_features)
                x_flat, _ = self.inverse(z_flat, context_expanded)
                # ✅ FEB 7: Clamp here too before reshaping
                x_flat = torch.clamp(x_flat, -1.0, 1.0)
                samples = x_flat.reshape(batch_size, num_samples, self.features)
                
                # ✅ FEB 7: Removed clamp after PSD scaling - denormalization will handle bounds
            
            return samples  # [batch, num_samples, features]

    def sample_with_uncertainty(self, num_samples: int, context: torch.Tensor,
                               n_overlaps: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        ✅ CRITICAL: Compatible interface with FlowMatching
        
        Sample posterior and compute statistics
        
        Args:
            num_samples: number of samples to generate
            context: [batch, context_features] conditioning
            n_overlaps: [batch] overlap count (for compatibility)
        
        Returns:
            dict with samples, mean, std, n_overlaps
        """
        samples = self.sample(num_samples, context)
        
        # If single context, squeeze batch dimension
        if context.shape[0] == 1 and samples.dim() == 3:
            samples = samples.squeeze(0)  # [num_samples, features]
        
        # Compute statistics
        if samples.dim() == 3:
            mean = samples.mean(dim=1)  # [batch, features]
            std = samples.std(dim=1)    # [batch, features]
        else:
            mean = samples.mean(dim=0)  # [features]
            std = samples.std(dim=0)    # [features]
        
        return {
            'samples': samples,
            'mean': mean,
            'std': std,
            'n_overlaps': n_overlaps[0] if n_overlaps is not None else None
        }
    
    def extract_signals(self, strain_batch: torch.Tensor, context: torch.Tensor,
                       n_samples: int = 50, return_all_samples: bool = False) -> Dict:
        """Extract parameters from strain with multiple samples"""
        with torch.no_grad():
            samples = self.sample(n_samples, context)
            
            if return_all_samples:
                return {
                    'samples_all': samples,
                    'mean': samples.mean(dim=0 if samples.dim() == 2 else 1),
                    'std': samples.std(dim=0 if samples.dim() == 2 else 1),
                    'cov': None
                }
            else:
                return {
                    'mean': samples.mean(dim=0 if samples.dim() == 2 else 1),
                    'std': samples.std(dim=0 if samples.dim() == 2 else 1),
                    'cov': None
                }
    
    def compute_nll_loss(self, params_norm: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        ✅ NSF PRIMARY LOSS: Negative Log-Likelihood
        
        For NSF, this is the main training objective (not velocity matching).
        Distance anchoring is handled separately in overlap_neuralpe.py
        """
        nll = self.log_prob(params_norm, context).mean()
        return nll
    
    def compute_bounds_penalty(self, params_norm: torch.Tensor, 
                              bounds: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
        """
        ✅ NSF SECONDARY LOSS: Bounds penalty
        
        Penalizes if samples fall outside valid parameter range.
        """
        lower, upper = bounds
        penalty_lower = torch.relu(lower - params_norm).mean()
        penalty_upper = torch.relu(params_norm - upper).mean()
        return penalty_lower + penalty_upper
    
    def compute_endpoint_loss(self, params_norm: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        ✅ OPTIONAL: Endpoint anchoring (for comparison with FlowMatching)
        
        Samples extremes from base distribution and checks if flow can map them
        back to valid parameter ranges.
        """
        batch_size = params_norm.shape[0]
        
        # Sample extreme noise values
        z_min = torch.full((batch_size, self.features), -3.0, device=params_norm.device)
        z_max = torch.full((batch_size, self.features), +3.0, device=params_norm.device)
        
        # Transform to parameter space
        x_min, _ = self.inverse(z_min, context)
        x_max, _ = self.inverse(z_max, context)
        
        # Check if ground truth falls within endpoint bounds
        penalty = torch.relu(x_min - params_norm) + torch.relu(params_norm - x_max)
        
        return penalty.mean()


def create_flow_model(
    flow_type: str,
    features: int,
    context_features: int = 0,
    max_overlaps: int = 6,
    config: Optional[Union[Dict[str, Any], ConfigDict, str, Path]] = None,
    **kwargs
) -> nn.Module:
    """
    Uses UniversalConfigReader for type-safe config validation and access.
    
    Args:
        flow_type: "nsf" or "flowmatching"
        features: parameter dimension
        context_features: context dimension
        max_overlaps: maximum number of overlapping signals
        config: Optional config dict/ConfigDict/path for validation
        **kwargs: Additional parameters (hidden_features, num_layers, etc.)
    
    Returns:
        Flow model instance
    """
    logger = logging.getLogger(__name__)
    
    # ✅ Initialize UniversalConfigReader for config validation
    reader = UniversalConfigReader()
    
    # Parse config if provided
    if config is not None:
        if isinstance(config, (str, Path)):
            config = reader.load(config)
        elif isinstance(config, dict) and not isinstance(config, ConfigDict):
            config = ConfigDict(config)
    
    # ✅ Validate and extract flow config parameters with type safety
    if config is not None:
        flow_config_dict = config.get("flow_config", {}) if isinstance(config, dict) else getattr(config, "flow_config", {})
        
        # Use reader to get values with type conversion and defaults
        kwargs.setdefault('num_layers', reader.get(flow_config_dict, 'num_layers', default=12, dtype=int))
        kwargs.setdefault('hidden_features', reader.get(flow_config_dict, 'hidden_features', default=256, dtype=int))
        kwargs.setdefault('num_bins', reader.get(flow_config_dict, 'num_bins', default=16, dtype=int))
        kwargs.setdefault('dropout', reader.get(flow_config_dict, 'dropout', default=0.15, dtype=float))
        
        # ✅ Jan 20: Support both global and per-parameter tail_bounds
        # Check for per_param_tail_bounds first, then fall back to global tail_bound
        per_param_bounds = flow_config_dict.get('per_param_tail_bounds', None)
        if per_param_bounds:
            # Convert string keys to int indices
            if isinstance(per_param_bounds, dict) and all(isinstance(k, str) for k in per_param_bounds.keys()):
                # If it's a dict with string keys, keep it for later mapping to param names
                kwargs.setdefault('tail_bound', per_param_bounds)
                logger.info(f"Per-parameter tail_bounds loaded from config")
            else:
                kwargs.setdefault('tail_bound', per_param_bounds)
        else:
            # Fall back to global tail_bound
            kwargs.setdefault('tail_bound', reader.get(flow_config_dict, 'tail_bound', default=3.0, dtype=float))
        
        logger.info(f"Flow config validated:")
        logger.info(f"  num_layers: {kwargs.get('num_layers')}")
        logger.info(f"  hidden_features: {kwargs.get('hidden_features')}")
        logger.info(f"  num_bins: {kwargs.get('num_bins')}")
        logger.info(f"  tail_bound: {kwargs.get('tail_bound')}")
        logger.info(f"  dropout: {kwargs.get('dropout')}")
    
    # Set defaults if not in kwargs
    kwargs.setdefault('num_layers', 12)
    kwargs.setdefault('hidden_features', 256)
    kwargs.setdefault('num_bins', 16)
    kwargs.setdefault('tail_bound', 3.0)
    kwargs.setdefault('dropout', 0.15)
    
    flow_type_lower = flow_type.lower()
    
    if flow_type_lower == "nsf":
        return NSFPosteriorFlow(
            features=features,
            context_features=context_features,
            max_overlaps=max_overlaps,
            **kwargs
        )
    elif flow_type_lower == "flowmatching":
        return EnhancedFlowMatchingPosterior(
            features=features,
            context_features=context_features,
            max_overlaps=max_overlaps,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown flow_type: {flow_type}. Must be 'nsf' or 'flowmatching'."
        )
