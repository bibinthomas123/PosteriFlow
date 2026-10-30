"""
Neural Spline Flow (NSF) posterior for Bayesian parameter estimation.
Optimized for overlapping gravitational wave signals.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union, Any, List
from pathlib import Path
import numpy as np
from nflows import distributions
import logging

from ahsd.utils.universal_config import UniversalConfigReader, ConfigDict

# Single source of truth for the normalizing flow's expected input/output range.
# Imported from parameter_scalers so both sides of the normalize→flow boundary
# always reference the same constant.
from ahsd.models.parameter_scalers import FLOW_NORM_BOUND

from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation


class PSDScaledNormal(nn.Module):
    """
    Base distribution supporting an optional per-parameter width rescaling,
    N(0, Sigma_psd) where Sigma_psd = diag(exp(log_sigma_psd)).

    As of the uncertainty-conditioning redesign (see
    docs/ARCHITECTURE_uncertainty_conditioning.md), OverlapNeuralPE always calls
    this with log_sigma_psd == 0, which makes it mathematically identical to a
    standard N(0, I) base distribution (sigma_psd = exp(0) = 1, so log_prob's
    quadratic/variance terms reduce exactly to the standard normal log-density).
    The external module that used to predict a non-trivial log_sigma_psd
    (`psd_head`, keyed off a redundant/weak SNR proxy) was removed; this class
    is kept as-is — rather than replaced with nflows' `StandardNormal` and a
    flows.py rewrite — because it is already exactly equivalent for the only
    input it now receives, so keeping it avoids any risk to the flow's
    forward/inverse transform machinery.
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
        Sample from N(0, I), standard normal (not PSD-scaled).

        σ_psd is applied outside the flow at sampling time (factored scale),
        which prevents the flow from re-learning σ_psd to undo PSD conditioning.

        Args:
            num_samples: number of samples to generate per batch element
            log_sigma_psd: [batch, dim] - IGNORED (kept for API compatibility)

        Returns:
            samples: [batch, num_samples, dim] - from standard normal N(0, I)
        """
        batch_size = log_sigma_psd.shape[0] if log_sigma_psd is not None else 1

        eps = torch.randn(
            batch_size, num_samples, self.dim,
            device=log_sigma_psd.device if log_sigma_psd is not None else torch.device('cpu'),
            dtype=log_sigma_psd.dtype if log_sigma_psd is not None else torch.float32
        )

        return eps  # [batch, num_samples, dim] from N(0, I)


class NSFPosteriorFlow(nn.Module):
    """
    Neural Spline Flow using the nflows library.

    Implementation using:
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
                 num_layers: int = 12, num_bins: int = 16, tail_bound: Union[float, Dict[int, float]] = FLOW_NORM_BOUND,
                 max_overlaps: int = 6, dropout: float = 0.0, temperature_scale: float = 1.5,
                 **kwargs):
        super().__init__()

        self.features = features
        self.context_features = context_features
        self.num_layers = num_layers
        self.max_overlaps = max_overlaps
        self.logger = logging.getLogger(__name__)

        # tail_bound may be a single float (global) or a dict mapping feature
        # index -> tail_bound (per-parameter)
        if isinstance(tail_bound, dict):
            self.per_param_tail_bounds = tail_bound
            self.tail_bounds_list = [
                tail_bound.get(i, FLOW_NORM_BOUND) for i in range(features)
            ]
            self.logger.info(
                f"Per-parameter tail_bounds enabled: {self.tail_bounds_list}"
            )
        else:
            self.per_param_tail_bounds = None
            self.tail_bounds_list = [tail_bound] * features

        # Base-distribution temperature: higher T -> wider base N(0, T^2 I) ->
        # more spread-out posterior samples, preventing collapse to a delta
        # function. A non-trainable buffer, not an nn.Parameter: it was
        # originally learnable but every caller (LeanNPE included) freezes it
        # immediately after construction, so making it a buffer removes the
        # dead "trainable-looking but never trained" parameter from
        # model.parameters()/the optimizer without changing the checkpoint's
        # state_dict key or shape (existing checkpoints still load unchanged).
        # log_prob()/sample() still accept an explicit `temperature` override
        # per call -- only the LEARNED-default mechanism is gone.
        self.register_buffer("temperature", torch.tensor(temperature_scale, dtype=torch.float32))
        self.temperature_scale_init = temperature_scale

        # PSD-conditioned base distribution (not StandardNormal): z ~ N(0, Sigma_psd)
        self.base_dist = PSDScaledNormal(shape=[features])

        # Build spline transform layers
        transforms = []
        self._ar_transforms = []  # kept for re-patching tail_bounds if the autoregressive order changes
        for layer_idx in range(num_layers):
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
                        tail_bound=FLOW_NORM_BOUND,  # Will be overridden per-feature
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
                self._ar_transforms.append(autoregressive_transform)
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
                        tail_bound=tail_bound if isinstance(tail_bound, float) else FLOW_NORM_BOUND,
                        num_blocks=2,
                        use_residual_blocks=True,
                        random_mask=False,
                        activation=nn.functional.relu,
                        dropout_probability=dropout,
                        use_batch_norm=False
                    )
                )
                self._ar_transforms.append(transforms[-1])

        # Compose transforms
        self.transform = CompositeTransform(transforms)

        # Create flow
        self.flow = Flow(self.transform, self.base_dist)

        # Autoregressive order: a fixed permutation applied to x on the way in
        # (forward/log_prob/compute_psd_aware_nll) and undone on the way out
        # (inverse/sample/sample_psd_aware). Identity by default -- callers
        # opt into a different order via set_autoregressive_order(). This
        # keeps every OTHER part of the codebase (param_names, scalers,
        # bounds, evaluation) working in the original semantic parameter
        # order; only the flow's internal degree ordering changes.
        self.register_buffer("_ar_perm", torch.arange(features, dtype=torch.long))
        self.register_buffer("_ar_inv_perm", torch.arange(features, dtype=torch.long))

        # Output clamp for inverse(). MUST track this flow's OWN spline domain,
        # not the module-level FLOW_NORM_BOUND default: LeanNPE constructs this
        # class with tail_bound=5.0 while FLOW_NORM_BOUND is 3.0, so a fixed
        # clamp silently truncated the outer 40% of the spline's support. Only
        # latent today (LeanNPE's targets live in [-1,1] and ParamScaler.wrap()
        # clamps there anyway), but it made the three bounds in this path --
        # normalize()'s [-1,1], tail_bound, and FLOW_NORM_BOUND -- disagree.
        self._out_clamp = float(max(self.tail_bounds_list))

        self.logger.info(
            f"NSFPosteriorFlow initialized: {features} features, "
            f"{context_features} context dims, {num_layers} layers, "
            f"{num_bins} bins, tail_bound={tail_bound}"
        )
    
    def set_autoregressive_order(self, order: List[int]) -> None:
        """
        Reorder which physical parameter occupies which autoregressive
        position. `order[i]` = original feature index placed at
        autoregressive position i (e.g. order=[2,0,1,3,...] puts feature 2
        first, feature 0 second, feature 1 third, then the rest unchanged).

        Only affects the flow's internal degree ordering (via a fixed
        permutation applied in forward()/undone in inverse()) -- param_names,
        scalers, bounds, and every other caller keep operating in the
        original semantic order. Also re-patches per-position tail_bounds
        (if per_param_tail_bounds is configured) so each parameter keeps its
        own configured tail_bound regardless of which position it now sits
        at, since that patching was originally applied by position index.

        Existing checkpoints are NOT compatible with a changed order (same
        tensor shapes, but the learned per-position weights would then refer
        to the wrong parameter) -- this is meant for a fresh training run.
        """
        features = self.features
        if sorted(order) != list(range(features)):
            raise ValueError(f"order must be a permutation of range({features}), got {order}")

        device = self._ar_perm.device
        perm = torch.tensor(order, dtype=torch.long, device=device)
        inv_perm = torch.argsort(perm)
        self._ar_perm = perm
        self._ar_inv_perm = inv_perm

        if self.per_param_tail_bounds:
            for ar_transform in self._ar_transforms:
                if hasattr(ar_transform, "_transforms"):
                    for position, orig_feature_idx in enumerate(order):
                        if position < len(ar_transform._transforms):
                            sub_transform = ar_transform._transforms[position]
                            if hasattr(sub_transform, "tail_bound"):
                                sub_transform.tail_bound = self.tail_bounds_list[orig_feature_idx]

        self.logger.info(f"Autoregressive order set: position -> original feature index = {order}")

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: x (params) → z (base distribution)"""
        x = x[:, self._ar_perm]
        if self.context_features > 0 and context is not None:
            z, log_det = self.transform(x, context=context)
        else:
            z, log_det = self.transform(x)
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None,
                 n_overlaps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse: z (base) -> x (params). Deterministic, no ODE solver."""

        # Context NaN/Inf propagates through the transform into invalid spline
        # parameters and raises an assertion error, so sanitize first.
        if context is not None:
            if torch.isnan(context).any() or torch.isinf(context).any():
                self.logger.warning(
                    f"NSF.inverse() detected NaN/Inf in context! "
                    f"Replacing with zeros (shape {context.shape})"
                )
                context = torch.nan_to_num(context, nan=0.0, posinf=1e-3, neginf=-1e-3)

        if self.context_features > 0 and context is not None:
            try:
                x, log_det = self.transform.inverse(z, context=context)
            except AssertionError as e:
                # NSF spline assertion failed - likely due to numerical instability
                self.logger.error(f"NSF.inverse() assertion failed: {e}")
                x = z
                log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        else:
            x, log_det = self.transform.inverse(z)

        # Undo the autoregressive-order permutation applied in forward(), so
        # every caller sees x back in the original semantic parameter order.
        x = x[:, self._ar_inv_perm]

        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clamp to this flow's own spline domain (max tail_bound), not the
        # module-level FLOW_NORM_BOUND — see _out_clamp in __init__.
        x = torch.clamp(x, -self._out_clamp, self._out_clamp)
        return x, log_det
    
    def log_prob(self, x: torch.Tensor, context: torch.Tensor = None, temperature: Optional[float] = None) -> torch.Tensor:
        """Primary loss: negative log-likelihood with temperature scaling."""

        # Sanitize context before passing to the nflows transform (NaN/Inf would
        # otherwise propagate into invalid spline parameters).
        if context is not None:
            if torch.isnan(context).any() or torch.isinf(context).any():
                self.logger.warning(
                    f"NSF.log_prob() detected NaN/Inf in context! "
                    f"Replacing with zeros (shape {context.shape})"
                )
                context = torch.nan_to_num(context, nan=0.0, posinf=1e-3, neginf=-1e-3)

        # If z ~ N(0, T^2 I), x = inverse(z) has a different distribution, so
        # log_prob must be adjusted: log_p(x) = log_p(x/T) + d * log(T)
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
            self.logger.error(f"NSF.log_prob() assertion failed: {e}")
            log_p = torch.ones(x.shape[0], device=x.device, dtype=x.dtype) * 1000.0

        # Change-of-variables: y=x/T -> log p_x(x) = log p_y(x/T) + log|det dy/dx|
        # det dy/dx = T^-d -> log|det| = -d*log(T), so log p_x(x) = log p_y(x/T) - d*log(T)
        log_prob_corrected = log_p - self.features * torch.log(temperature)

        if torch.isnan(log_prob_corrected).any():
            log_prob_corrected = torch.nan_to_num(log_prob_corrected, nan=1000.0)

        return -log_prob_corrected  # negative, for minimization

    def sample(self, num_samples: int, context: torch.Tensor = None, temperature: Optional[float] = None) -> torch.Tensor:
        """Sample from posterior p(x|context)"""
        with torch.no_grad():
            if temperature is None:
                temperature = torch.clamp(self.temperature, min=0.5, max=3.0)
            else:
                temperature = torch.tensor(temperature, dtype=self.temperature.dtype, device=self.temperature.device)

            if context is not None:
                batch_size = context.shape[0]

                context_expanded = context.unsqueeze(1).expand(
                    batch_size, num_samples, self.context_features
                ).reshape(batch_size * num_samples, self.context_features)

                # Scale z from the base distribution by temperature, widening
                # N(0, I) -> N(0, T^2 I)
                z = self.base_dist.sample(num_samples * batch_size)
                z = z * temperature

                samples, _ = self.inverse(z, context_expanded)
                samples = samples.reshape(batch_size, num_samples, self.features)
            else:
                z = self.base_dist.sample(num_samples)
                z = z * temperature
                samples, _ = self.inverse(z, None)
            
            samples = torch.clamp(samples, -FLOW_NORM_BOUND, FLOW_NORM_BOUND)
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
        
        # Transform x (data) → z (latent) using the FORWARD transform.
        # Change-of-variables: log p(x) = log p(z) + log|det ∂z/∂x|
        # self.forward() returns (z, log|det ∂z/∂x|) — correct for density evaluation.
        # self.inverse() maps z→x (sampling direction) and must NOT be used here.
        if isinstance(self.base_dist, PSDScaledNormal):
            # Transform x → z (data → latent, correct direction)
            z, log_det = self.forward(x, context)  # [batch, features]

            # Compute log_prob under PSD-scaled base
            log_p_base = self.base_dist.log_prob(z, log_sigma_psd)  # [batch]

            # Compute log_det from flow transform
            log_det = log_det.squeeze() if log_det.dim() > 1 else log_det  # [batch]

            # Total log probability with Jacobian
            # log_p(x) = log_p(z) + log|det ∂z/∂x|
            log_prob_x = log_p_base + log_det  # [batch]

            # Return negative log likelihood (for minimization)
            nll = -log_prob_x  # [batch]
        else:
            z, log_det = self.forward(x, context)
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
                # Sample from the unit normal base (not PSD-scaled); PSD scaling
                # is applied after the flow (factored-scale architecture).
                z_samples = self.base_dist.sample(num_samples, log_sigma_psd)  # [batch, num_samples, features] from N(0,I)

                z_flat = z_samples.reshape(batch_size * num_samples, self.features)

                context_expanded = context.unsqueeze(1).expand(
                    batch_size, num_samples, self.context_features
                ).reshape(batch_size * num_samples, self.context_features)

                x_flat, _ = self.inverse(z_flat, context_expanded)

                # Clamp in normalized space before PSD scaling, using ±FLOW_NORM_BOUND
                # to match tail_bound and normalize_batch's output range.
                x_flat = torch.clamp(x_flat, -FLOW_NORM_BOUND, FLOW_NORM_BOUND)

                samples = x_flat.reshape(batch_size, num_samples, self.features)

                # Apply the factored scale outside the flow (no gradients): the flow
                # works in unit-variance space, sigma_psd is applied here. Not clamped
                # afterward — intentional, so samples can expand per PSD and
                # denormalization handles final bounds.
                sigma_psd = torch.exp(log_sigma_psd)  # [batch, features]
                samples = samples * sigma_psd.unsqueeze(1)  # [batch, num_samples, features]
            else:
                # Fallback to standard sampling (shouldn't happen)
                z = torch.randn(batch_size, num_samples, self.features,
                               device=context.device, dtype=context.dtype)
                z_flat = z.reshape(batch_size * num_samples, self.features)
                context_expanded = context.unsqueeze(1).expand(
                    batch_size, num_samples, self.context_features
                ).reshape(batch_size * num_samples, self.context_features)
                x_flat, _ = self.inverse(z_flat, context_expanded)
                x_flat = torch.clamp(x_flat, -FLOW_NORM_BOUND, FLOW_NORM_BOUND)
                samples = x_flat.reshape(batch_size, num_samples, self.features)

            return samples  # [batch, num_samples, features]

    def sample_with_uncertainty(self, num_samples: int, context: torch.Tensor,
                               n_overlaps: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compatible interface with FlowMatching.

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
        Primary NSF loss: negative log-likelihood.

        For NSF, this is the main training objective (not velocity matching).
        Distance anchoring is handled separately in overlap_neuralpe.py
        """
        nll = self.log_prob(params_norm, context).mean()
        return nll
    
    def compute_bounds_penalty(self, params_norm: torch.Tensor,
                              bounds: Tuple[float, float] = (-FLOW_NORM_BOUND, FLOW_NORM_BOUND)) -> torch.Tensor:
        """
        Secondary NSF loss: bounds penalty.

        Penalizes if samples fall outside valid parameter range.
        """
        lower, upper = bounds
        penalty_lower = torch.relu(lower - params_norm).mean()
        penalty_upper = torch.relu(params_norm - upper).mean()
        return penalty_lower + penalty_upper
    
    def compute_endpoint_loss(self, params_norm: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Optional endpoint anchoring (for comparison with FlowMatching).

        Samples extremes from base distribution and checks if flow can map them
        back to valid parameter ranges.
        """
        batch_size = params_norm.shape[0]

        z_min = torch.full((batch_size, self.features), -FLOW_NORM_BOUND, device=params_norm.device)
        z_max = torch.full((batch_size, self.features), +FLOW_NORM_BOUND, device=params_norm.device)

        x_min, _ = self.inverse(z_min, context)
        x_max, _ = self.inverse(z_max, context)

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
        flow_type: "nsf" (the only supported model)
        features: parameter dimension
        context_features: context dimension
        max_overlaps: maximum number of overlapping signals
        config: Optional config dict/ConfigDict/path for validation
        **kwargs: Additional parameters (hidden_features, num_layers, etc.)
    
    Returns:
        Flow model instance
    """
    logger = logging.getLogger(__name__)

    reader = UniversalConfigReader()

    if config is not None:
        if isinstance(config, (str, Path)):
            config = reader.load(config)
        elif isinstance(config, dict) and not isinstance(config, ConfigDict):
            config = ConfigDict(config)

    if config is not None:
        flow_config_dict = config.get("flow_config", {}) if isinstance(config, dict) else getattr(config, "flow_config", {})

        kwargs.setdefault('num_layers', reader.get(flow_config_dict, 'num_layers', default=12, dtype=int))
        kwargs.setdefault('hidden_features', reader.get(flow_config_dict, 'hidden_features', default=256, dtype=int))
        kwargs.setdefault('num_bins', reader.get(flow_config_dict, 'num_bins', default=16, dtype=int))
        kwargs.setdefault('dropout', reader.get(flow_config_dict, 'dropout', default=0.15, dtype=float))

        # Check for per_param_tail_bounds first, then fall back to global tail_bound
        per_param_bounds = flow_config_dict.get('per_param_tail_bounds', None)
        if per_param_bounds:
            # A dict with string keys maps param names -> tail_bound, kept as-is
            # for later mapping to param names
            if isinstance(per_param_bounds, dict) and all(isinstance(k, str) for k in per_param_bounds.keys()):
                kwargs.setdefault('tail_bound', per_param_bounds)
                logger.info(f"Per-parameter tail_bounds loaded from config")
            else:
                kwargs.setdefault('tail_bound', per_param_bounds)
        else:
            # Fall back to global tail_bound
            kwargs.setdefault('tail_bound', reader.get(flow_config_dict, 'tail_bound', default=FLOW_NORM_BOUND, dtype=float))
        
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
    kwargs.setdefault('tail_bound', FLOW_NORM_BOUND)
    kwargs.setdefault('dropout', 0.15)
    
    flow_type_lower = flow_type.lower()

    if flow_type_lower != "nsf":
        raise ValueError(
            f"Unknown flow_type: {flow_type}. Only 'nsf' is supported."
        )

    return NSFPosteriorFlow(
        features=features,
        context_features=context_features,
        max_overlaps=max_overlaps,
        **kwargs
    )
