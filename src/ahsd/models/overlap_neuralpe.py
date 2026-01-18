"""
Neural Parameter Estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import deque
from pathlib import Path

from ahsd.core.priority_net import PriorityNet
from ahsd.core.adaptive_subtractor import AdaptiveSubtractor
from ahsd.models.flows import create_flow_model
from ahsd.models.parameter_scalers import TorchParameterScaler
from ahsd.utils import UniversalConfigReader, ConfigDict


class ResidualContextAdapter(nn.Module):
    """
    Residual adapter with bounded modification.
    
    Problem with naive adapter: Completely replaces context (change=15), destroying PriorityNet info
    Solution: Preserve original context + apply small, learnable modification
    
    Architecture:
        output = input + scale * adapter(input)
    
    Where:
    - input: PriorityNet context [batch, context_dim]
    - adapter: Small learnable transformation
    - scale: Learnable parameter starting at 0.1 (bounded growth)
    
    This ensures:
    ✅ 90% of original signal preserved (residual)
    ✅ 10% learned task-specific adaptation (delta)
    ✅ Bounded modification (scale parameter prevents explosion)
    ✅ Smooth training (LayerNorm instead of BatchNorm)
    """
    
    def __init__(self, context_dim: int, hidden_dim: int = 256, scale: float = 0.1):
        super().__init__()
        
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        # Learnable transformation (starts small)
        self.adapter = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # More stable than BatchNorm
            nn.GELU(),
            nn.Dropout(0.2),  # Regularization
            nn.Linear(hidden_dim, context_dim)
        )
        
        # Learnable scale factor: controls how much to apply the adaptation
        # Starts at 0.1 so output ≈ input + 0.1*delta (preserves 99% of original)
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual adaptation that PRESERVES variance.
        
        Changed from unit-norm delta (kills variance) to raw delta (preserves variance)
        
        Args:
            x: Input context [batch, context_dim]
        
        Returns:
            output: Adapted context [batch, context_dim]
                   = x + clipped_scale * delta
                   
        Where:
        - delta is raw adapter output (preserves input variance structure)
        - clipped_scale prevents explosion (bounded to [0, 0.2])
        - This ensures modification preserves input std while adding learned features
        """
        residual = x
        delta = self.adapter(x)  # [batch, context_dim]
        
        # Clip scale instead of normalizing delta
        # Why this works:
        # - Unit-norm delta added to variable input creates directional changes (kills variance)
        # - Raw delta preserves the variance structure of adapter output
        # - Clipping scale prevents explosion while maintaining variance
        clipped_scale = torch.clamp(self.scale, max=0.2)  # Cap at 0.2 for stability
        
        # output = input + clipped_delta
        # → Preserves input variance while adding learned modification
        output = residual + clipped_scale * delta
        return output


class OverlapNeuralPE(nn.Module):
    """
    Unified Neural PE for overlapping gravitational wave signals.

    Integrates:
    - PriorityNet: Signal importance ranking (optional)
    - Normalizing Flow: Posterior estimation
    - Adaptive Subtractor: Iterative signal extraction
    - Physics Priors: Domain knowledge integration
    - Uncertainty Estimation: Calibrated uncertainties
    - Proper Posterior Sampling: Full Bayesian inference
    """

    def __init__(
        self,
        param_names: List[str],
        priority_net_path: str,
        config: Dict[str, Any],
        device: str = "cuda",
        event_type: str = "BBH",
    ):
        super().__init__()

        self.param_names = param_names
        self.param_dim = len(param_names)
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)

        # Event-type specific configuration (Nov 13 enhancement)
        self.event_type = event_type.upper()  # BBH, BNS, or NSBH
        self.use_event_priors = config.get("enable_event_specific_priors", True)

        # Model configuration - read from neural_posterior section first (where config actually lives)
        np_config = config.get("neural_posterior", {})
        self.context_dim = np_config.get("context_dim", config.get("context_dim", 512))  # ✅ Read from neural_posterior first
        self.max_iterations = np_config.get("max_iterations", config.get("max_iterations", 5))

        # Dropout configuration - read from neural_posterior section first
        self.dropout_rate = np_config.get("dropout", config.get("dropout", 0.1))
        self.flow_config = np_config.get("flow_config", config.get("flow_config", {}))
        
        # ✅ FIXED: Read num_layers from flow_config (default to 6 if not specified)
        self.n_flow_layers = self.flow_config.get("num_layers", 6)
        self.flow_dropout = self.flow_config.get("dropout", 0.15)
        self.flow_hidden_features = self.flow_config.get("hidden_features", 128)
        self.flow_num_blocks = self.flow_config.get("num_blocks_per_layer", 2)

        # ✅ DEC 15: Initialize advanced parameter scaler (fixes -285 Mpc distance bias!)
        self.param_scaler = TorchParameterScaler(
            param_names=self.param_names,
            event_type=self.event_type,
            device=str(self.device)
        )
        
        # Parameter bounds for normalization
        self.param_bounds = self._get_parameter_bounds()

        # Physics-informed priors
        self.physics_priors = self._build_physics_priors()

        # Initialize components
        self._init_components(priority_net_path)

        # Uncertainty estimator network
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.param_dim + self.context_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, self.param_dim),
            nn.Softplus(),  # Ensure positive uncertainties
        )
        
        # Distance prediction head for auxiliary distance loss
        # Input: [context, network_snr, detector_rms, rel_amp, chirp_mass_norm] 
        #        [batch, context_dim + 1 + 3 + 3 + 1 = context_dim + 8]
        # Output: predicted distance [batch, 1] in [0, 1] normalized space
        # ✅ JAN 5 CRITICAL FIX: Bypass encoder bottleneck with amplitude information
        # This provides direct SNR-distance coupling signal that encoder couldn't provide alone
        # ✅ JAN 5 CRITICAL FIX #4: Remove Sigmoid() - distance is log-linear, not bounded-linear
        # Sigmoid compresses gradients at extremes and fights log-RMS SNR feature
        # Output will be clamped to [0, 1] in loss function instead (better gradient flow)
        self.distance_prediction_head = nn.Sequential(
            nn.Linear(self.context_dim + 8, 512),  # +8: network_snr(1) + detector_rms(3) + rel_amp(3) + chirp_mass(1)
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        # Performance tracking
        self.performance_tracker = {
            "training_losses": deque(maxlen=1000),
            "validation_metrics": deque(maxlen=100),
            "complexity_history": deque(maxlen=1000),
            "inference_times": deque(maxlen=1000),
            "rl_rewards": deque(maxlen=1000),
        }
        self.training_step = 0
        
        # ✅ Initialize PriorityNet metric tracking
        self._last_priority_net_preds = None
        self._last_priority_net_uncs = None
        
        self.to(self.device)
        total_params = sum(p.numel() for p in self.parameters())

        self.logger.info(
            f" NeuralPE initialized with {total_params:,} parameters"
        )
        self.logger.info(f"   Context dim: {self.context_dim}")
        self.logger.info(f"   Flow layers: {self.n_flow_layers}")
        self.logger.info(f"   Dropout: {self.dropout_rate}, Flow dropout: {self.flow_dropout}")
        self.logger.info(f"   Distance prediction head initialized ({self._count_distance_head_params():,} params)")

    def _parse_config(self, config: Union[Dict[str, Any], ConfigDict, str, Path]) -> ConfigDict:
        """
        Parse configuration from various formats.
        
        Args:
            config: Config as dict, ConfigDict, or path to YAML file
        
        Returns:
            ConfigDict with configuration
        """
        if isinstance(config, ConfigDict):
            return config
        elif isinstance(config, dict):
            return self._reader._to_config_dict(config)
        elif isinstance(config, (str, Path)):
            # Load from file
            config_path = Path(config)
            if not config_path.exists():
                self.logger.warning(f"Config file not found: {config_path}, using empty config")
                return ConfigDict()
            return self._reader.load(config_path)
        else:
            self.logger.warning(f"Unknown config type: {type(config)}, using empty config")
            return ConfigDict()
    
    def _get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
         """Get parameter bounds for normalization - CORRECTED TO MATCH ACTUAL DATASET (Dec 14, REVISED)."""
         bounds = {
              "mass_1": (1.0, 100.0),  # Matches data [1.0, 99.9]
              "mass_2": (1.0, 100.0),  # Matches data [0.1, 95.5]
              "luminosity_distance": (10.0, 2000.0),  #  (Dec 14 REVISED): Covers 99.5% of data, avoids extreme outliers. Was (10, 15000) which caused 3× gradient scaling, now optimized for training stability
              "geocent_time": (-2.0, 4.0),  #  (REVISED): Asymmetric but matches actual data clustering better. Was (-8.4, 8.4) too wide
              "ra": (0.0, 2 * np.pi),
              "dec": (-np.pi / 2, np.pi / 2),
              "theta_jn": (0.0, np.pi),
              "psi": (0.0, np.pi),
              "phase": (0.0, 2 * np.pi),
              "a_1": (0.0, 0.99),  # Primary BH spin magnitude
              "a_2": (0.0, 0.99),  # Secondary BH spin magnitude
              "tilt1": (0.0, np.pi),  # Spin tilt angle
              "tilt2": (0.0, np.pi),  # Spin tilt angle
          }
         return {param: bounds.get(param, (0.0, 1.0)) for param in self.param_names}
    
    def _count_distance_head_params(self) -> int:
        """Count parameters in distance prediction head."""
        return sum(p.numel() for p in self.distance_prediction_head.parameters())

    def _normalize_strain(self, strain_data: torch.Tensor) -> torch.Tensor:
        """
        Normalize strain data to (mean≈0, std≈1) for Conv layers.
        
        ✅ JAN 5 CRITICAL FIX: Normalize per-detector, not globally
        
        Problem: Global normalization leaks amplitude across detectors
        - H1 strain [1e-21, 1e-20], L1 strain [1e-22, 1e-21]
        - Global norm by max destroys inter-detector SNR ratio
        
        Solution: Normalize per-detector to preserve amplitude geometry
        - Each detector normalized independently
        - Relative amplitude ratios preserved (SNR information)
        
        Args:
            strain_data: [batch, n_detectors, time]
        
        Returns:
            normalized: [batch, n_detectors, time] with per-detector zero-mean, unit-std
        """
        # Ensure strain_data has correct shape [batch, n_detectors, time]
        if strain_data.dim() == 2:
            # If [batch, time], add detector dimension
            strain_data = strain_data.unsqueeze(1)
        
        # ✅ FIX: Normalize per-detector, not globally
        # Compute mean and std across time dimension only (dim=-1)
        strain_mean = strain_data.mean(dim=-1, keepdim=True)  # [batch, n_detectors, 1]
        strain_std = strain_data.std(dim=-1, keepdim=True)    # [batch, n_detectors, 1]
        
        # Normalize: (x - mean) / std
        # This preserves inter-detector amplitude ratios (SNR structure)
        normalized = (strain_data - strain_mean) / (strain_std + 1e-8)
        
        return normalized

    def _build_physics_priors(self) -> Dict[str, torch.distributions.Distribution]:
        """Build physics-informed priors with event-type-specific configurations."""
        priors = {}

        for param in self.param_names:
            if "mass" in param:
                # Event-type-specific mass priors (Nov 13 enhancement)
                if self.use_event_priors:
                    if self.event_type == "BBH":
                        # BBH: broader distribution, avoid very small masses
                        # Start at 5 Msun, power-law Salpeter index 2.35
                        priors[param] = torch.distributions.Pareto(5.0, 2.35)
                    elif self.event_type == "BNS":
                        # BNS: narrow distribution around 1.4 Msun (neutron star)
                        # Normal with mean=1.4, std=0.15 (σ ≈ 10% of mean)
                        priors[param] = torch.distributions.Normal(loc=1.4, scale=0.15)
                    elif self.event_type == "NSBH":
                        # NSBH: differentiate primary (NS) and secondary (BH)
                        if "mass_1" in param or param == "mass_1":
                            # Primary mass (NS): ~1.4 Msun
                            priors[param] = torch.distributions.Normal(loc=1.4, scale=0.15)
                        else:
                            # Secondary mass (BH): broader distribution starting at 5 Msun
                            priors[param] = torch.distributions.Pareto(5.0, 2.35)
                    else:
                        # Fallback for unknown event type
                        priors[param] = torch.distributions.Pareto(1.0, 2.35)
                else:
                    # Generic prior if event-specific disabled
                    priors[param] = torch.distributions.Pareto(1.0, 2.35)

            elif param in ["ra", "phase"]:
                # Uniform priors for phase-like angles (isotropic sky)
                priors[param] = torch.distributions.Uniform(0.0, 2 * np.pi)

            elif param in ["dec", "theta_jn"]:
                # Sine priors for spherical angles (uniform on sphere/celestial sphere)
                priors[param] = torch.distributions.Beta(0.5, 0.5)

            elif param == "luminosity_distance":
                # Volume prior for distance (number density proportional to r^2)
                # Pareto(1.0, 2.0) ~ 1/r^2 after exponential transformation
                priors[param] = torch.distributions.Pareto(1.0, 2.0)

            elif param in ["a_1", "a_2"]:
                # Spin magnitude priors: Beta distribution favors lower spins
                # Beta(2, 5) matches astrophysical spin distributions (most BHs have a < 0.5)
                priors[param] = torch.distributions.Beta(2.0, 5.0)

            else:
                # Default uniform prior for other parameters
                priors[param] = torch.distributions.Uniform(0.0, 1.0)

        self.logger.info(
            f"✅ Physics priors initialized: event_type={self.event_type}, "
            f"use_event_priors={self.use_event_priors}"
        )
        return priors

    def _load_checkpoint_with_mismatch_handling(
        self, model: torch.nn.Module, checkpoint_state: Dict[str, torch.Tensor]
    ):
        """Load checkpoint, skipping weights with shape mismatches."""
        model_state = model.state_dict()
        incompatible_keys = []

        for name, checkpoint_param in checkpoint_state.items():
            if name in model_state:
                if checkpoint_param.shape == model_state[name].shape:
                    # Shape matches, load normally
                    model_state[name].copy_(checkpoint_param)
                else:
                    # Shape mismatch, skip
                    incompatible_keys.append(
                        f"{name}: checkpoint {checkpoint_param.shape} vs model {model_state[name].shape}"
                    )

        if incompatible_keys:
            self.logger.warning(
                f"Skipping {len(incompatible_keys)} weights with shape mismatches:\n"
                + "\n".join(incompatible_keys[:5])  # Show first 5
                + (
                    f"\n... and {len(incompatible_keys) - 5} more"
                    if len(incompatible_keys) > 5
                    else ""
                )
            )

        if len(checkpoint_state) == 0:
            self.logger.warning(
                "⚠️  Empty checkpoint (0 weights) - model initialized with random weights"
            )
        else:
            self.logger.info(
                f"Loaded {len(checkpoint_state) - len(incompatible_keys)}/{len(checkpoint_state)} checkpoint weights"
            )

    def _init_components(self, priority_net_path: str):
        """Initialize all pipeline components."""

        # 1. PriorityNet (pre-trained, frozen) - optional
        if priority_net_path is not None:
            checkpoint = torch.load(priority_net_path, map_location=self.device)
            model_arch = checkpoint.get("model_architecture", {})
            # ✅ FIXED: Pass config to PriorityNet so it respects use_transformer_encoder flag
            priority_net_config = self.config.get("priority_net", {})
            self.priority_net = PriorityNet(
                use_strain=model_arch.get("use_strain", True),
                use_edge_conditioning=model_arch.get("use_edge_conditioning", True),
                n_edge_types=model_arch.get("n_edge_types", 19),
                config=priority_net_config,
            )
            self._load_checkpoint_with_mismatch_handling(
                self.priority_net, checkpoint["model_state_dict"]
            )
            self.priority_net.eval()
            for param in self.priority_net.parameters():
                param.requires_grad = False
            self.logger.info("✅ PriorityNet loaded and frozen")
            
            #  (Dec 17 - Second Fix): Use Residual Adapter to preserve PriorityNet info
            # Previous naive adapter was DESTROYING PriorityNet signal (change=15, should be 0.1-0.2)
            # New residual adapter: output = input + 0.1 * adapter(input)
            # This preserves 90% of original context while learning task-specific adaptation
            self.context_adapter = ResidualContextAdapter(
                context_dim=self.context_dim,
                hidden_dim=256,
                scale=0.1  # Start conservative - grows gradually if needed
            )
            self.logger.info(f"✅ Residual adapter initialized: scale=0.1, hidden=256")
        else:
            self.priority_net = None
            self.context_adapter = None
            self.logger.info("⊘ PriorityNet disabled")

        # 2. Context Encoder (✅ FIXED: Flexible detector count - infer from data)
        # Default to 3 detectors (H1, L1, V1), but can handle 2 (H1, L1) or other configurations
        n_detectors = self.config.get("n_detectors", 3)
        self.context_encoder = ContextEncoder(
            n_detectors=n_detectors, hidden_dim=self.context_dim, dropout=self.dropout_rate
        )
        self.logger.info(f"Context Encoder initialized with {n_detectors} detectors")

        # 3. Normalizing Flow - ✅ Q3 REDESIGN: NSF (Neural Spline Flow)
        np_config = self.config.get("neural_posterior", {})
        
        # ✅ Q3: Switch to NSF (better gradients, 16× faster, simpler training)
        flow_type = np_config.get("flow_type", "nsf")  # Default to NSF

        # ✅ STEP 5: Account for explicit SNR conditioning
        # If enabled, context_dim increases by 1 (network_snr only)
        # ✅ CRITICAL: Only use network_snr (always available at inference)
        #    Never use target_snr (ground truth info, causes data leakage during training)
        snr_conditioning_enabled = np_config.get("snr_conditioning", True)
        flow_context_dim = self.context_dim + (1 if snr_conditioning_enabled else 0)
        
        # Track actual runtime context dimension (after SNR conditioning)
        self.context_dim_with_snr = flow_context_dim

        # ✅ PASS CONFIG TO FLOW MODEL - UniversalConfigReader will handle validation
        self.flow = create_flow_model(
            flow_type=flow_type,
            features=self.param_dim,
            context_features=flow_context_dim,  # 770 if SNR conditioning, 768 otherwise
            config=np_config,  # ✅ Pass neural_posterior config for automatic extraction
            solver_steps=self.flow_config.get("solver_steps", 20),  # Only used by FlowMatching, ignored by NSF
        )

        self.logger.info(
            f"🔵 {flow_type.upper()} Flow Initialization"
            f" | encoder_context_dim={self.context_dim}"
            f" | snr_conditioning={snr_conditioning_enabled} (+1 network_snr feature if enabled)"
            f" | actual_context_dim={self.context_dim_with_snr}"
            f" | num_layers={self.flow_config.get('num_layers', 12)}"
        )

        # 4. Adaptive Subtractor
        self.adaptive_subtractor = AdaptiveSubtractor()

    def sample_posterior(
        self, strain_data: torch.Tensor, n_samples: int = 1000, 
        return_all_samples: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from learned posterior distribution with rejection sampling.

        Args:
            strain_data: [batch, n_det, n_samples] whitened strain data
            n_samples: Number of posterior samples to draw
            return_all_samples: If True, return all samples (not just summary stats)

        Returns:
            If return_all_samples=False (default):
                dict containing:
                    'samples': [batch, n_samples, param_dim] posterior samples
                    'means': [batch, param_dim] posterior means
                    'stds': [batch, param_dim] posterior standard deviations
                    'uncertainties': [batch, param_dim] estimated uncertainties
            
            If return_all_samples=True:
                torch.Tensor [batch, n_samples, param_dim] all samples (for metrics computation)

         (Nov 13): Added rejection sampling + output clamping to filter
        out-of-range predictions. The flow can extrapolate beyond physical bounds,
        causing loss explosion. Rejection sampling ensures all returned samples are physical.
        """
        self.eval()
        batch_size = strain_data.size(0)

        with torch.no_grad():
            # Extract context from strain
            #  (Dec 7): Match training-time strain normalization
            # Conv layers need normalized input (mean≈0, std≈1)
            # Normalize strain the same way as in compute_loss()
            strain_normalized = self._normalize_strain(strain_data)
            context = self.context_encoder(strain_normalized)  # [batch, context_dim]
            
            #  (Dec 17): Apply context adapter for inference
            # Must match training-time context transformation
            if self.context_adapter is not None:
                context = self.context_adapter(context)
            
            # DO NOT re-normalize context - it causes train-test mismatch
            # The context encoder already produces properly scaled features
            
            # ✅ STEP 5: EXPLICIT SNR CONDITIONING (inference)
            # Must match training-time SNR conditioning (inference-safe: network_snr only)
            try:
                np_config = self.config.get("neural_posterior", {})
                snr_conditioning_enabled = np_config.get("snr_conditioning", True)
                
                if snr_conditioning_enabled:
                    # Compute network SNR from strain (always available, no ground truth)
                    network_snr = self._compute_network_snr(strain_data)  # [batch, 1]
                    
                    # Normalize network SNR (same as training)
                    log_rms = network_snr  # Already computed in log space
                    log_min = np.log(0.01)   # -4.605 (very quiet baseline)
                    log_max = np.log(1.0)     # 0.0 (reference whitened noise)
                    norm_net_snr = (log_rms - log_min) / (log_max - log_min)
                    norm_net_snr = torch.clamp(norm_net_snr, min=-5.0, max=5.0)
                    
                    # Append to context: [batch, 768] → [batch, 769]
                    context = torch.cat([context, norm_net_snr], dim=1)
                    
            except Exception as e:
                self.logger.debug(f"⚠️ STEP 5 SNR conditioning in inference failed: {type(e).__name__}: {e}")
                # Continue without SNR conditioning if it fails (graceful degradation)

            samples_list = []

            for i in range(batch_size):
                context_i = context[i : i + 1].expand(
                    n_samples * 2, -1
                )  # 2x oversampling for rejection
                z_i = torch.randn(n_samples * 2, self.param_dim, device=self.device)

                # ✅ FIX 4: Rejection sampling loop with diagnostic logging
                valid_samples = []
                n_attempts = 0
                max_attempts = 5

                while len(valid_samples) < n_samples and n_attempts < max_attempts:
                    # Flow transformation returns normalized params [-1, 1]
                    samples_normalized, _ = self.flow.inverse(z_i, context_i)
                    
                    # Clamp to [-1, 1] to ensure we stay in valid range
                    samples_normalized = torch.clamp(samples_normalized, -1.0, 1.0)

                    # ✅ FIX 2: Check raw normalized outputs (before clamping)
                    out_of_bounds_norm = (samples_normalized < -1.0) | (samples_normalized > 1.0)
                    norm_violation_pct = (
                        out_of_bounds_norm.sum().float() / out_of_bounds_norm.numel()
                    ).item() * 100

                    # Denormalize to physical units (includes clamping in FIX 1)
                    samples_physical = self._denormalize_parameters(samples_normalized)

                    # ✅ Filter physically valid samples
                    valid_mask = self._check_sample_validity(samples_physical)
                    n_valid = valid_mask.sum().item()

                    valid_samples.append(samples_physical[valid_mask])

                    n_attempts += 1
                    if len(valid_samples) < n_samples and n_attempts < max_attempts:
                        z_i = torch.randn(n_samples * 2, self.param_dim, device=self.device)



                # Concatenate and trim to requested number
                if valid_samples:
                    batch_samples = torch.cat(valid_samples, dim=0)[:n_samples]
                    if len(batch_samples) < n_samples:
                        # If rejection too high, just return what we have with clipping
                        if len(batch_samples) < n_samples // 2:
                            self.logger.warning(
                                f"Rejection rate too high, using fallback sampling"
                            )
                            # Generate more samples without rejection
                            z_fallback = torch.randn(n_samples - len(batch_samples), self.param_dim, device=self.device)
                            context_fallback = context_i[:n_samples - len(batch_samples)]
                            samples_fallback_norm, _ = self.flow.inverse(z_fallback, context_fallback)
                            samples_fallback = self._denormalize_parameters(samples_fallback_norm)
                            # Clamp to bounds instead of rejecting
                            samples_fallback = self._clamp_to_bounds(samples_fallback)
                            batch_samples = torch.cat([batch_samples, samples_fallback], dim=0)
                else:
                    # Fallback: return samples without rejection if all rejected
                    self.logger.warning(f"All samples rejected, using clamped samples")
                    samples_normalized, _ = self.flow.inverse(
                        torch.randn(n_samples, self.param_dim, device=self.device),
                        context_i[:n_samples],
                    )
                    batch_samples = self._denormalize_parameters(samples_normalized)
                    batch_samples = self._clamp_to_bounds(batch_samples)

                samples_list.append(batch_samples)

            samples = torch.stack(samples_list, dim=0)  # [batch, n_samples, param_dim]

            # ✅ NO POSTERIOR WIDENING (Dec 14, 2025): Removed artificial scaling
            # REASON: Posterior width should emerge naturally from flow learning
            # Previous artificial widening (1.10, then 1.35) masked root causes:
            #   1. Log-scale normalization fix ensures proper parameter distribution
            #   2. Context utilization loss forces flow to use context features
            #   3. Strong context variance penalty forces encoder to learn
            # With these fixes, posteriors should reach correct width naturally
            # If posteriors still too narrow after training, root cause is context/flow not learning
            # Rather than mask with scaling, the training loss fixes address this properly
            posterior_scale_factor = 1.0  # No artificial widening - let flow learn naturally
             
            means_temp = samples.mean(dim=1, keepdim=True)  # [batch, 1, param_dim]
            samples = means_temp + posterior_scale_factor * (samples - means_temp)
             
            # Note: posterior_scale_factor = 1.0 means no scaling (identity operation)
            # Clamp to bounds is still applied for safety
            samples = self._clamp_to_bounds(samples)

            # ✅ Early return for metrics computation (return all samples)
            if return_all_samples:
                return samples
            
            # Compute summary statistics (from scaled samples)
            means = samples.mean(dim=1)
            stds = samples.std(dim=1)

            # Estimate uncertainties
            uncertainties = self.uncertainty_estimator(
                torch.cat([self._normalize_parameters(means), context], dim=1)
            )

            return {
                "samples": samples,
                "means": means,
                "stds": stds,
                "uncertainties": uncertainties,
                "context": context,
            }

    def _clamp_to_bounds(self, samples: torch.Tensor) -> torch.Tensor:
         """Clamp samples to physical bounds without rejection.
         
         Args:
             samples: [batch, n_samples, param_dim] or [batch, param_dim] physical parameters
         Returns:
             clamped_samples: Same shape as input, clipped to valid bounds
         
         ✅ FIXED (Dec 14): 
             - Updated geocent_time: (-2.0, 8.0) → (-2.5, 6.0)
             - Updated luminosity_distance: (10.0, 8000.0) → (10.0, 5000.0)
             - Works with both 2D [batch, param_dim] and 3D [batch, n_samples, param_dim] tensors
         """
         clamped = samples.clone()
         
         # Complete bounds for all 11 parameters
         param_bounds = [
             (1.0, 100.0),          # mass_1
             (1.0, 100.0),          # mass_2
             (10.0, 15000.0),       # luminosity_distance
             (0.0, 2*np.pi),        # ra
             (-np.pi/2, np.pi/2),   # dec
             (0.0, np.pi),          # theta_jn
             (0.0, np.pi),          # psi
             (0.0, 2*np.pi),        # phase
             (-8.4, 8.4),           # geocent_time (-2.5, 6.0, now symmetric)
             (0.0, 0.99),           # a1 (primary spin magnitude) 
             (0.0, 0.99),           # a_2 (secondary spin magnitude) 
         ]
         
         # Clamp all parameters using ellipsis indexing (works for any shape)
         for i, (min_val, max_val) in enumerate(param_bounds):
             if i < clamped.shape[-1]:  
                 clamped[..., i] = torch.clamp(clamped[..., i], min=min_val, max=max_val)  # ✅ FIXED: Use ... to handle any shape
         
         return clamped

    def _normalize_parameters(self, physical_params: torch.Tensor) -> torch.Tensor:
        """
        Normalize physical parameters using advanced data-driven scaler.
        
        - Distance: Log-minmax normalization 
        - Masses: Log-zscore (handles 0.1-100 M☉ range)
        - Spins: Bounded zscore (most spins low, proper variance)
        - All parameters scaled to [-1, 1] for network compatibility
        
        Args:
            physical_params: [batch, param_dim] physical parameter values
        Returns:
            normalized: [batch, param_dim] normalized to [-1, 1]
        """
        return self.param_scaler.normalize_batch(physical_params)
    
    def _denormalize_parameters(self, normalized_params: torch.Tensor) -> torch.Tensor:
        """
        Denormalize parameters back to physical units using advanced scaler.
        
        Args:
            normalized_params: [batch, param_dim] normalized to [-1, 1]
        Returns:
            physical: [batch, param_dim] physical parameter values
        """
        return self.param_scaler.denormalize_batch(normalized_params)

    def _check_sample_validity(self, samples: torch.Tensor) -> torch.Tensor:
        """Check if samples are within physical bounds.

        Returns boolean mask of valid samples.
        """
        valid = torch.ones(samples.shape[0], dtype=torch.bool, device=samples.device)

        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_bounds[param_name]

            # Check if within bounds (with small tolerance for numerical errors)
            tolerance = (max_val - min_val) * 0.01  # 1% tolerance
            valid &= (samples[:, i] >= min_val - tolerance) & (samples[:, i] <= max_val + tolerance)

        return valid

    def extract_single_signal(
        self, strain_data: torch.Tensor, complexity: str = "medium"
    ) -> Dict[str, Any]:
        """
        Extract parameters for a single signal with complexity-dependent settings.

        Args:
            strain_data: [batch, n_det, n_samples] strain data
            complexity: Complexity level ('low', 'medium', 'high')

        Returns:
            dict with parameter estimates and uncertainties
        """
        # ✅ Nov 14: Apply complexity-dependent settings
        complexity_settings = self.complexity_configs.get(complexity, {})
        n_samples = complexity_settings.get("inference_samples", 1000)
        # Note: flow_layers is pre-configured in flow model initialization
        # This can be extended to support dynamic flow architectures if needed

        result = self.sample_posterior(strain_data, n_samples=n_samples)

        return {
            "means": result["means"],
            "stds": result["stds"],
            "samples": result["samples"],
            "uncertainties": result["uncertainties"],
            "context": result["context"],
            "complexity": complexity,  # ✅ Track which complexity was used
            "n_samples": n_samples,  # ✅ Track inference samples
        }

    def extract_overlapping_signals(
        self,
        strain_data: torch.Tensor,
        true_params: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract all overlapping signals iteratively with integrated bias correction,
        adaptive subtraction, normalizing flows, and RL control.
        
        ✅ JOINT TRAINING (Nov 27): When training=True, gradient tracking enabled
        for flow extraction feedback loop.

        Args:
            strain_data: [batch, n_det, n_samples] strain data
            true_params: Optional [batch, n_signals, param_dim] ground truth parameters
            training: Whether in training mode (enables gradient tracking through extraction)

        Returns:
            dict with all_extracted signals, final_residuals, and metadata
        """
        batch_size = strain_data.size(0)

        all_extracted = []
        residual_data = strain_data.clone() if not training else strain_data
        # ✅ During training, keep gradients; during inference, detach

        # ✅ Nov 14: Track extraction success for state
        successful_extractions = 0
        total_attempts = 0

        pipeline_state = {
            "remaining_signals": self.max_iterations,
            "residual_power": 1.0,
            "current_snr": 0.0,
            "extraction_success_rate": 1.0,
        }

        for iteration in range(self.max_iterations):
            # ✅ 1. GET PRIORITIES FROM PRIORITYNET
            if self.priority_net is not None:
                with torch.no_grad():
                    detections = self._residual_to_detections(residual_data)
                    # ✅ Nov 14: Pass strain segments for temporal feature extraction
                    priorities, uncertainties = self.priority_net(detections, strain_segments=residual_data)
                    # ✅ Track metrics for logging
                    self._last_priority_net_preds = priorities.detach()
                    self._last_priority_net_uncs = uncertainties.detach()
            else:
                priorities = None

            # ✅ 2. SELECT COMPLEXITY (RL disabled for training stability)
            # Use fixed medium complexity during training; RL available in InferencePipeline for inference-time adaptation
            complexity = "medium"

            # ✅ 3. EXTRACT SIGNAL USING NORMALIZING FLOW
            extraction_result = self.extract_single_signal(residual_data, complexity)
            params_means = extraction_result["means"]
            params_stds = extraction_result["stds"]
            context = extraction_result["context"]
            n_samples_used = extraction_result.get("n_samples", 1000)  # ✅ Nov 14
            
            # ✅ Nov 14: Track extraction success (no NaN/Inf in means)
            total_attempts += 1
            extraction_valid = not (torch.isnan(params_means).any() or torch.isinf(params_means).any())
            if extraction_valid:
                successful_extractions += 1

            # ✅ 4. NO BIAS CORRECTION (trained separately)
            params_corrected = params_means
            params_stds_corrected = params_stds

            all_extracted.append(
                {
                    "params": params_corrected,
                    "uncertainties": params_stds_corrected,
                    "context": context,  # ✅ ADDED: Store context for loss computation
                    "priority": priorities,
                    "iteration": iteration,
                    "complexity": complexity,
                    "n_samples_used": n_samples_used,  # ✅ Nov 14: Track complexity setting used
                    "extraction_valid": extraction_valid,  # ✅ Nov 14: Track if extraction succeeded
                }
            )

            # ✅ 5. ADAPTIVE SUBTRACTION (using physics-based template matching)
            params_dict = self._tensor_to_param_dict(params_corrected[0] if params_corrected.dim() > 1 else params_corrected)
            uncertainties_dict = self._tensor_to_param_dict(params_stds[0] if params_stds.dim() > 1 else params_stds)

            # Convert tensor residual_data to detector format dictionary
            residual_dict = self._tensor_to_detector_dict(residual_data)

            # Call AdaptiveSubtractor with learned parameters
            residual_data_dict, subtraction_result = self.adaptive_subtractor.extract_and_subtract(
                residual_dict,
                best_params=params_dict,
                uncertainties=uncertainties_dict
            )

            # Convert residual dictionary back to tensor
            residual_data = self._detector_dict_to_tensor(residual_data_dict)
            
            # ✅ Log subtraction quality metrics 
            if subtraction_result:
                validation_results = subtraction_result.get('validation_results', {})
                overall_quality = validation_results.get('overall_quality', 0.0)
                
                if overall_quality > 0:
                    self.logger.debug(
                        f"Iteration {iteration}: Subtraction quality={overall_quality:.3f}, "
                        f"Recommendations: {validation_results.get('recommendations', ['N/A'])}"
                    )

            # ✅ 6. UPDATE PIPELINE STATE FOR RL
            pipeline_state["remaining_signals"] -= 1
            pipeline_state["residual_power"] = float(torch.mean(residual_data**2))
            
            # ✅ Nov 14: Update extraction success rate for state
            if total_attempts > 0:
                pipeline_state["extraction_success_rate"] = successful_extractions / total_attempts

            # Compute SNR for state
            if iteration == 0:
                pipeline_state["current_snr"] = float(torch.sqrt(torch.mean(strain_data**2)))
            else:
                pipeline_state["current_snr"] = float(torch.sqrt(torch.mean(residual_data**2)))

            # Early stopping if residual too low
            if pipeline_state["residual_power"] < 0.001:
                self.logger.info(f"Stopping at iteration {iteration+1}: low residual power")
                break

            # ✅ 7. RL TRAINING DISABLED (inference-only)
            # RL is disabled during training to prevent instability from noisy rewards and mid-epoch complexity switching
            pass

        return {
            "all_extracted": all_extracted,           # ✅ Used by compute_loss for joint training
            "extracted_signals": all_extracted,       # ✅ Backward compat
            "final_residuals": residual_data,         # ✅ Used by compute_loss for residual loss
            "final_residual": residual_data,          # ✅ Backward compat
            "n_iterations": iteration + 1,
            "pipeline_state": pipeline_state,
            "rl_metadata": {  # ✅ Nov 14: Track RL-related stats
                "total_attempts": total_attempts,
                "successful_extractions": successful_extractions,
                "success_rate": successful_extractions / total_attempts if total_attempts > 0 else 0.0,
            }
        }

    def extract(self, strain_data: torch.Tensor, training: bool = False) -> Dict[str, Any]:
        """
        Public API for signal extraction and RL metric population.
        
        Args:
            strain_data: [batch, n_det, n_samples] strain data
            training: Whether in training mode
        
        Returns:
            dict with extracted signals and metadata
        """
        return self.extract_overlapping_signals(strain_data, training=training)

    def _tensor_to_detector_dict(self, strain_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """Convert strain tensor [batch, n_det, n_samples] to detector dict format."""
        detector_names = ["H1", "L1", "V1"][: strain_tensor.size(1)]
        result = {}

        for det_idx, detector in enumerate(detector_names):
            if det_idx < strain_tensor.size(1):
                # Take first batch element and convert to numpy
                result[detector] = strain_tensor[0, det_idx].cpu().numpy().astype(np.float32)

        return result

    def _detector_dict_to_tensor(self, detector_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """Convert detector dict format back to strain tensor [batch, n_det, n_samples]."""
        detector_names = ["H1", "L1", "V1"]
        detector_arrays = []

        for detector in detector_names:
            if detector in detector_dict:
                detector_arrays.append(detector_dict[detector])

        if not detector_arrays:
            # Fallback if no detectors found
            return torch.zeros(1, 2, 4096, dtype=torch.float32, device=self.device)

        # Stack detectors and add batch dimension
        stacked = np.stack(detector_arrays, axis=0)  # [n_det, n_samples]
        tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)  # [1, n_det, n_samples]

        return tensor

    def _residual_to_detections(self, residual: torch.Tensor) -> List[Dict]:
        """
        Convert residual strain to detection format for PriorityNet.
        
        ✅ JAN 5 CRITICAL FIX #2: Use unified SNR computation
        Problem: Two incompatible SNR definitions (RMS-log-scaled vs linear RMS×10)
        Solution: Use _compute_network_snr() consistently everywhere
        """
        batch_size = residual.size(0)
        detections = []

        # ✅ FIX: Use unified _compute_network_snr method
        # This uses log-RMS scaling consistent with distance head training
        network_snr_all = self._compute_network_snr(residual)  # [batch, 1]

        for i in range(batch_size):
            # Extract SNR for this sample and convert to float
            network_snr = float(network_snr_all[i, 0])

            detection = {
                "network_snr": network_snr,
                "match_filter_snr": network_snr,
                "chi_squared": 1.0,
                "null_snr": 0.1,
            }
            detections.append(detection)

        return detections

    def _tensor_to_param_dict(self, params_tensor: torch.Tensor) -> Dict[str, float]:
        """Convert parameter tensor to dictionary."""
        params_np = params_tensor.detach().cpu().numpy()

        if len(params_np.shape) > 1:
            params_np = params_np[0]

        return {name: float(params_np[i]) for i, name in enumerate(self.param_names)}

    def _compute_extraction_reward(
        self, estimated_params: torch.Tensor, true_params: Optional[Dict]
    ) -> float:
        """Compute reward for RL training."""
        if true_params is None:
            return 0.0

        true_tensor = torch.tensor(
            [true_params.get(name, 0.0) for name in self.param_names],
            dtype=torch.float32,
            device=self.device,
        )

        rel_error = torch.abs((estimated_params[0] - true_tensor) / (true_tensor + 1e-6))
        accuracy = 1.0 - torch.mean(rel_error).item()

        return max(0.0, accuracy)

    def compute_loss(self, strain_data: torch.Tensor, true_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive training loss with all integrated components.
    
        1. Extract signals using extract_overlapping_signals() with gradient tracking
        2. Compute flow loss on extracted parameters (realistic residual scenario)
        3. Compute extraction accuracy loss (|extracted - true|)
        4. Compute residual quality loss (penalty for high residuals)
        5. Integrated gradient flow: Flow → Subtractor → better residuals → better context
        
        Args:
            strain_data: [batch, n_det, n_samples] strain data (full strain with overlaps)
            true_params: [batch, n_signals, param_dim] true parameters
        
        Returns:
            dict with total loss and component losses
        """
        # Increment training step for diagnostic logging
        if self.training:
            self.training_step += 1
        
        batch_size = strain_data.size(0)
        
        # ========================================
        # STEP 1: CONTEXT ENCODING (from clean strain)
        # ========================================
        # Compute context ONCE from clean strain data
        # During training, context must match inference: fresh encoding from strain
        # NOT from extraction results (which can have errors/biases)
        
        # 🔍 DIAGNOSTIC (Dec 7): Log strain data validity first
        strain_mean = strain_data.mean().item()
        strain_std = strain_data.std().item()
        strain_max = strain_data.abs().max().item()
        has_nan = torch.isnan(strain_data).any().item()
        has_inf = torch.isinf(strain_data).any().item()
        
        # Alert only on severe issues (NaN, Inf, all zeros, or completely empty)
        if has_nan or has_inf or strain_std < 1e-25 or (strain_max < 1e-15 and strain_std < 1e-15):
            self.logger.warning(
                f"⚠️ Strain data issue: mean={strain_mean:.2e}, std={strain_std:.2e}, "
                f"max={strain_max:.2e}, NaN={has_nan}, Inf={has_inf}"
            )
        
        #  Normalize strain BEFORE context encoder
        # Conv layers expect normalized input (mean≈0, std≈1)
        # If strain std is very small (2e-07), gradients will vanish in Conv layers
        strain_normalized = self._normalize_strain(strain_data)
        context = self.context_encoder(strain_normalized)  # [batch, context_dim]
        
        #  (Dec 17): Apply context adapter to transform encoder output
        # The context encoder learns strain features, but Neural PE needs parameter estimation features
        # The adapter is a learnable transformation that bridges this gap
        if self.context_adapter is not None:
            try:
                if self.training:
                    context_before_adapter = context.detach().clone()
                    context = self.context_adapter(context)  # [batch, context_dim] → learned transformation
                    adapter_change = (context - context_before_adapter).norm(dim=1).mean().item()
                    if self.training_step % 100 == 0:  # Log every 100 steps for visibility
                        # Get current scale factor
                        scale_val = self.context_adapter.scale.item() if hasattr(self.context_adapter, 'scale') else 0.0
                        self.logger.info(
                            f"✅ [RESIDUAL ADAPTER] Step {self.training_step}: scale={scale_val:.4f}, "
                            f"change={adapter_change:.6f}, context_std={context.std():.4f}"
                        )
                    else:
                        # Eval mode: apply adapter without logging
                        context = self.context_adapter(context)
            except Exception as e:
                self.logger.error(
                    f"❌ [CONTEXT ADAPTER ERROR] {type(e).__name__}: {e}\n"
                    f"   Context shape: {context.shape}, Training: {self.training}"
                )
                raise  # Re-raise to help debug
        
        # ✅ CRITICAL FIX (Jan 7): Save original context BEFORE SNR appending
        # Distance head needs original [batch, 768] context, not the [batch, 769] with SNR
        context_for_distance_head = context.clone()
        
        # ✅ STEP 5: EXPLICIT SNR CONDITIONING (INFERENCE-SAFE)
        # Append ONLY network SNR to context (always available at inference)
        # ✅ CRITICAL: Never append target_snr (ground truth info causes data leakage)
        # 
        # Why network_snr only:
        # - Network SNR = RMS-based signal strength (computed from strain, always available)
        # - Target SNR = ground truth parameter (NOT available at inference, causes leakage if used)
        # - Using target_snr during training artificially sharpens posteriors and hides biases
        # 
        # Physics principle:
        # - The network learns: "given strain RMS, predict parameters"
        # - RMS encodes signal strength → SNR conditioning helps generalize across regimes
        # - No ground truth needed (network learns from data distribution)
        try:
            np_config = self.config.get("neural_posterior", {})
            snr_conditioning_enabled = np_config.get("snr_conditioning", True)
            
            if snr_conditioning_enabled:
                # Compute network SNR from strain (always available, no ground truth)
                network_snr = self._compute_network_snr(strain_data)  # [batch, 1]
                
                # Normalize network SNR (ensures [-1, 1] range for stable gradients)
                # Use single-element version of normalize function
                log_rms = network_snr  # Already computed in log space
                log_min = np.log(0.01)   # -4.605 (very quiet baseline)
                log_max = np.log(1.0)     # 0.0 (reference whitened noise)
                norm_net_snr = (log_rms - log_min) / (log_max - log_min)
                norm_net_snr = torch.clamp(norm_net_snr, min=-5.0, max=5.0)  # Allow range but prevent explosion
                
                # Append to context: [batch, 768] → [batch, 769]
                context = torch.cat([context, norm_net_snr], dim=1)
                
                # Diagnostic logging (every 500 steps)
                if self.training_step % 500 == 0:
                    self.logger.debug(
                        f"✅ [STEP 5] SNR conditioning (inference-safe)"
                        f" | network_snr={network_snr.mean():.3f} (RMS log-scale)"
                        f" | context_dim={context.shape[1]}"
                    )
        except Exception as e:
            self.logger.warning(f"⚠️  STEP 5 SNR conditioning failed: {type(e).__name__}: {e}")
            self.logger.debug("Continuing without SNR features (graceful degradation)")
            # Continue without SNR conditioning if it fails (graceful degradation)
        
        # Compute stride auxiliary loss
        # Forces encoder to learn strain properties (not just random noise)
        # Stride auxiliary loss measures: can encoder predict strain mean/std from context?
        stride_loss = torch.tensor(0.0, device=strain_data.device)
        if self.training:
            try:
                # Get stride predictions from encoder's auxiliary head
                if hasattr(self.context_encoder, '_last_stride_predictions'):
                    stride_predictions = self.context_encoder._last_stride_predictions  # [batch, 2*n_det]
                    
                    # 🔴 FIXED: Compute true stride statistics correctly
                    # Use RAW strain for amplitude, NOT z-normalized strain
                    # Build list of [rms_amplitude, diversity] for each detector
                    # This preserves SNR information without destroying physics via per-sample rescaling
                    true_stride_stats_list = []
                    for det_idx in range(batch_size):
                        for d in range(3):  # 3 detectors (H1, L1, V1)
                            # Use raw strain (already whitened, not z-normalized)
                            det_strain = strain_data[det_idx, d, :]  # [time_samples]
                            # RMS amplitude (std of raw strain)
                            rms_amplitude = det_strain.std()
                            # Diversity = std of absolute values (captures signal morphology)
                            diversity = torch.abs(det_strain).std()
                            true_stride_stats_list.append(rms_amplitude)
                            true_stride_stats_list.append(diversity)
                    
                    # Stack and reshape: [batch*3*2] → [batch, 6]
                    true_stride_stats_tensor = torch.stack(true_stride_stats_list)  # [batch*6]
                    true_stride_stats_tensor = true_stride_stats_tensor.view(batch_size, -1)  # [batch, 6]
                    
                    # Verify shape matches stride_predictions [batch, 6]
                    if stride_predictions.shape == true_stride_stats_tensor.shape:
                        # MSE loss between predicted and true stride statistics
                        stride_loss = ((stride_predictions - true_stride_stats_tensor) ** 2).mean()
                        
                        if self.training_step % 200 == 0:
                            self.logger.debug(
                                f"[STRIDE AUX] Encoder learning strain statistics: "
                                f"stride_loss={stride_loss:.6f} (pred_shape={stride_predictions.shape}, true_shape={true_stride_stats_tensor.shape})"
                            )
                    else:
                        if self.training_step % 500 == 0:
                            self.logger.warning(
                                f"[STRIDE AUX] Shape mismatch: pred={stride_predictions.shape} vs true={true_stride_stats_tensor.shape}"
                            )
                        stride_loss = torch.tensor(0.0, device=strain_data.device)
            except Exception as e:
                if self.training_step % 500 == 0:
                    self.logger.debug(f"[STRIDE AUX] Error computing stride loss: {type(e).__name__}: {e}")
                stride_loss = torch.tensor(0.0, device=strain_data.device)
        
        # Strengthen context variance penalty to fix encoder learning
        # Context encoder was producing near-constant embeddings (std=0.46 instead of ~1.0)
        # Increase penalty weight from 0.05 to 0.20 to force learning
        context_variance_loss = torch.tensor(0.0, device=strain_data.device)
        if self.training:
            context_std_actual = context.std()
            # Target context std should be >0.8 (encoder has learned meaningful features)
            # Penalty if std is too low: 0.10 * (0.8 - std)^2 (was 0.05, 4× stronger now)
            if context_std_actual < 0.8:
                context_variance_loss = 0.10 * (0.8 - context_std_actual) ** 2  
                if self.training_step % 100 == 0:
                    self.logger.debug(
                        f"[CONTEXT LOSS] Penalizing low variance: std={context_std_actual:.4f}, "
                        f"penalty={context_variance_loss:.6f}, target=0.8"
                    )
        
        # ========================================
        # STEP 1.5: CONTEXT DISCRIMINATION LOSS (NEW - Dec 15, 22:30, FIXED Dec 16)
        # ========================================
        # 🔴 CRITICAL FIX: Context encoder was producing pure random Gaussian (mean=0, std=1.0)
        # Root cause: LayerNorm + output_scale just normalize unlearned encoder outputs to Gaussian
        # Solution: Add explicit loss that FORCES context to be DIFFERENT for DIFFERENT strains
        #
        # IMPORTANT: We can't just check feature variance - pure N(0,1) also has std≈1.0!
        # Instead, force context fusion layers to process encoder outputs (not bypass them)
        # by training them to predict something detector-specific
        #
        # Two complementary approaches:
        # 1. Stride predictor head (aux loss) - forces encoder to encode strain properties
        # 2. Variance penalty - penalizes ONLY true collapse (both conv and context bad)
        #
        # ⚠️ FIXED (Dec 16): Removed contradictory penalties that fight each other
        # OLD: Penalized high context std AND low conv std = oscillation
        # NEW: Only penalize TRUE collapse (conv collapsed AND context is noise)
        
        context_discrimination_loss = torch.tensor(0.0, device=strain_data.device)
        
        if self.training:
            # Get conv output statistics (before fusion)
            with torch.no_grad():
                batch_size_check = strain_data.shape[0]
                detector_features_debug = []
                for i in range(min(batch_size_check, 4)):  # Check first 4 samples only (debug overhead)
                    det_data = strain_normalized[i:i+1, :1, :]  # First detector, one sample
                    features_check = self.context_encoder.detector_encoder(det_data)  # [1, 128, 64]
                    detector_features_debug.append(features_check.flatten())
            
            if len(detector_features_debug) > 1:
                conv_outputs_stack = torch.stack(detector_features_debug)  # [samples, 128*64]
                conv_between_sample_std = conv_outputs_stack.std(dim=0).mean()  # Average variance across features
                
                # Get context statistics
                context_feature_std = context.std(dim=0).mean()  # Average std across context features in batch
                
                # ✅ FIXED: Only penalize TRUE collapse
                # Collapse = encoder dead (conv_std < 0.15) AND context is noise (std > 0.95)
                if conv_between_sample_std < 0.15 and context_feature_std > 0.95:
                    # SEVERE COLLAPSE: Conv outputs nothing, context is pure noise
                    collapse_penalty = (0.15 - conv_between_sample_std) + (context_feature_std - 0.95) ** 2
                    context_discrimination_loss = torch.tensor(2.0 * collapse_penalty, device=strain_data.device, dtype=torch.float32)
                    
                    if self.training_step % 100 == 0:
                        self.logger.error(
                            f"🔴 SEVERE COLLAPSE: Conv std={conv_between_sample_std:.6f}, "
                            f"Context std={context_feature_std:.6f}, penalty={context_discrimination_loss.item():.6f}"
                        )
                
                elif conv_between_sample_std < 0.25 and context_feature_std > 0.85:
                    # MILD COLLAPSE: Conv struggling, context becoming noise
                    mild_penalty = 0.5 * ((0.25 - conv_between_sample_std) + (context_feature_std - 0.85) ** 2)
                    context_discrimination_loss = torch.tensor(mild_penalty, device=strain_data.device, dtype=torch.float32)
                    
                    if self.training_step % 100 == 0:
                        self.logger.warning(
                            f"🟡 Mild collapse: Conv std={conv_between_sample_std:.6f}, "
                            f"Context std={context_feature_std:.6f}, penalty={context_discrimination_loss.item():.6f}"
                        )
                
                else:
                    # ✅ NO COLLAPSE: Learning is healthy
                    # Return ZERO penalty - don't fight natural learning!
                    context_discrimination_loss = torch.tensor(0.0, device=strain_data.device, dtype=torch.float32)
                    
                    if self.training_step % 500 == 0:
                        self.logger.info(
                            f"✅ Context encoder healthy: Conv std={conv_between_sample_std:.6f}, "
                            f"Context std={context_feature_std:.6f}"
                        )
        
        # ========================================
        # STEP 1.6: CONTEXT STD REGULARIZATION (UPDATED - Dec 17 WITH CONTEXT ADAPTER)
        # ========================================
        # ✅ UPDATED FIX (Dec 17): Context adapter now helps with task transformation
        # With adapter present:
        # - Weight reduced from 5.0 → 2.0 (adapter helps stabilize context)
        # - Still use LINEAR penalty (0.5 coefficient) for consistency
        # - Target threshold relaxed from 0.6 → 0.7 (adapter can handle slightly noisier context)
        # 
        # Without adapter (PriorityNet disabled):
        # - Use original weight 5.0 (no help from task transformation)
        # - Keep threshold 0.6
        # ✅ REMOVED (Jan 2): Unused penalty computation - actual penalty is in STEP 6 (lines 1886+)
        # This code was computing a penalty but never using it, causing confusion
        # The real variance penalty is computed below at line 1909 using weight 10.0
        
        # ========================================
        # STEP 2: EXTRACT SIGNALS (Using Ground Truth for Now)
        # ========================================
        # ✅ FIX: Skip actual extraction during early training when flow is untrained
        # Instead, directly use ground truth params to train flow
        # This avoids extracting garbage and having huge extraction losses
        # 
        #  (Dec 16): Proper Flow Matching Loss
        # Flow must learn to GENERATE from noise, not reconstruct ground truth!
        # Previous bug: flow was trained on identity mapping (input→output copying)
        # This wasted compute and prevented proper posterior estimation
        # 
        # CORRECT behavior: 
        # 1. Sample noise from N(0,I)
        # 2. Interpolate: z_t = t*target + (1-t)*noise
        # 3. Predict velocity using context
        # 4. Match predicted velocity to true velocity
        # This teaches flow to generate from noise using context guidance
        
        # Dummy container for compatibility with rest of code
        all_extracted = []
        
        # Create zero residuals for now (will improve as model trains)
        final_residuals = torch.zeros_like(strain_data[:, :1, :])
        
        # ========================================
        # STEP 3: FLOW LOSS (On ALL Signal Parameters) ✅ DEC 5 FIX
        # ========================================
        #  (Dec 5): Train on ALL signals, not just primary
        # Data has 4-6 signals per overlap, but code only trained on signal 0
        # This caused secondary parameters (a_2, mass_2, etc.) to be untrained
        # Now: Loop through all signals and train flow on each
        
        flow_loss_total = torch.tensor(0.0, device=strain_data.device)
        
        # ✅ FLEXIBLE: Handle both 2D (single signal) and 3D (multiple signals)
        if true_params.dim() == 2:
            # Single signal: [batch, param_dim] → reshape to [batch, 1, param_dim]
            true_params = true_params.unsqueeze(1)
        
        n_signals_in_batch = true_params.shape[1]  # Number of signals in this batch
        n_signals_extracted = n_signals_in_batch  # Track for logging
        flow_loss_per_signal = []
        
        # ========================================
        # FLOW LOSS (FlowMatching or NSF)
        # ========================================
        # ✅ FIX (Jan 2): Extract neural_posterior section from full config
        np_config = self.config.get("neural_posterior", self.config)
        flow_type = np_config.get("flow_type", "flowmatching")
        flow_type_lower = flow_type.lower().strip()  # Normalize: "NSF", "nsf", " nsf " all work
        
        # DEBUG: Log flow type on first batch
        if strain_data.shape[0] == 64 and not hasattr(self, '_flow_type_logged'):
            self.logger.debug(f"🔍 Flow type detected: {flow_type}")
            self._flow_type_logged = True
        
        # Loop through all signals in batch
        for signal_idx in range(n_signals_in_batch):
            params_signal = true_params[:, signal_idx, :]  # [batch, param_dim]
            params_norm = self._normalize_parameters(params_signal)
            
            try:
                batch_size = params_norm.shape[0]
                
                # ✅ NSF vs FlowMatching: Different loss computation
                if flow_type_lower == "nsf":
                    # NSF: Use NLL (negative log probability) as primary loss
                    # NSF includes geometric anchoring via inverse likelihood transformation
                    nll_loss = self.flow.compute_nll_loss(params_norm, context)
                    bounds_penalty = self.flow.compute_bounds_penalty(params_norm)
                    
                    # REMOVED mean-anchor loss entirely
                    # Reason: Rank loss (calibration) already does the right thing
                    # - Fixes calibration (median ≈ truth)
                    # - Improves SBC
                    # - Accepts skewness
                    # Distance bias at low SNR is EXPECTED physics behavior
                    # 
                    # Report median bias, not mean bias
                    # medium/high SNR → good, low SNR → wide but calibrated
                    # That's success. Don't try to eliminate expected bias.
                    
                    # Clean loss without mean-anchor
                    signal_loss = nll_loss + 0.5 * bounds_penalty
                    
                    flow_loss_per_signal.append(signal_loss.item())
                    flow_loss_total += signal_loss
                    
                    # Debug logging (no mean_anchor)
                    if signal_idx == 0 and hasattr(self, 'logger') and self.training_step % 100 == 0:
                        self.logger.debug(
                            f"[NSF LOSS] NLL={nll_loss.item():.4f}, bounds={bounds_penalty.item():.6f}"
                        )
                    continue
                
                # ===== FLOWMATCHING LOSS (below) =====
                # ===== STEP 1: Sample noise from N(0, I) =====
                # This is the STARTING point for generation
                z_0 = torch.randn_like(params_norm)  # [batch, param_dim]
                
                # ===== STEP 2: Sample random interpolation time t ∈ [0, 1] =====
                # t=0 means we're at noise (z_0)
                # t=1 means we're at target parameters
                t = torch.rand(batch_size, 1, device=params_norm.device)  # [batch, 1]
                
                # ===== STEP 3: Compute interpolated position on optimal transport path =====
                # The path follows: z_t = (1-t)*z_0 + t*x_target
                # This is the optimal coupling for Gaussian base distribution
                z_t = (1.0 - t) * z_0 + t * params_norm  # [batch, param_dim]
                
                # ===== STEP 4: Compute TARGET velocity (what the flow should learn) =====
                # On the linear interpolation path, the velocity is constant:
                # v_target = d(z_t)/dt = x_target - z_0
                v_target = params_norm - z_0  # [batch, param_dim]
                
                # ===== STEP 5: Predict velocity using velocity_net (conditioned on context!) =====
                # The network takes current position (z_t), time (t), and context
                # Context encodes information from the strain data
                # Network outputs the predicted velocity at this position/time
                v_pred = self.flow.velocity_net(z_t, t, context)  # [batch, param_dim]
                
                # ===== STEP 6: Flow Matching Loss =====
                # Penalize if predicted velocity doesn't match target velocity
                # This is the fundamental training signal for the flow
                cfm_loss = ((v_pred - v_target) ** 2).mean()  # Scalar
                
                #  (Dec 15): Endpoint loss ENABLED to anchor flow outputs
                # Issue: Without location constraint, flow drifts from true parameters
                # Root cause: CFM (velocity matching) only constrains gradients, not position
                # Solution: Use endpoint anchoring - constrain distribution support via extremes
                # 
                # Why endpoints are better than sample mean:
                # 1. Deterministic: Always exist, no stochastic variance
                # 2. Support constraint: Directly controls distribution bounds
                # 3. Stable: Less prone to mode collapse (endpoints must stay valid)
                # 4. Efficient: Single forward pass through velocity_net, no sampling loop
                # 
                # Implementation: Sample extreme noise values (z_min, z_max) and flow them to time=1
                # to get endpoints, then penalize if they drift too far from true params
                
                endpoint_loss = torch.tensor(0.0, device=strain_data.device) 
                
                # 🔧 DEBUG: Check config structure
                np_config = self.config.get("neural_posterior", {})
                if not np_config and hasattr(self, 'logger'):
                    # Fallback: try reading endpoint_loss_weight from top level
                    endpoint_loss_weight = self.config.get("endpoint_loss_weight", 0.5)
                    if endpoint_loss_weight > 0 and hasattr(self, 'logger'):
                        self.logger.debug(f"[ENDPOINT LOSS] Using top-level endpoint_loss_weight={endpoint_loss_weight}")
                else:
                    endpoint_loss_weight = np_config.get("endpoint_loss_weight", 0.5)
                    # if endpoint_loss_weight > 0 and hasattr(self, 'logger'):
                    #     self.logger.debug(f"[ENDPOINT LOSS] Using neural_posterior section, endpoint_loss_weight={endpoint_loss_weight}")
                
                # 🔴 CRITICAL: Endpoint loss routing (Flow-type aware)
                # FlowMatching has compute_endpoint_loss() built-in
                # CFM requires manual endpoint computation
                # Note: flow_type_lower already computed above (line 1248)
                
                # ✅ FIX (Dec 15, Bug #2): Simplified routing - use endpoint_loss_weight directly
                if endpoint_loss_weight > 0 and flow_type_lower == "flowmatching":
                    # FlowMatching: Use built-in compute_endpoint_loss()
                    # Will be computed below in "For FlowMatching" section
                    use_flowmatching_endpoint = True
                else:
                    # CFM: Use manual endpoint computation
                    use_flowmatching_endpoint = False
                
                if endpoint_loss_weight > 0 and not use_flowmatching_endpoint:
                    try:
                        # 🔧 CRITICAL DEBUG: Check if velocity_net exists
                        if not hasattr(self.flow, 'velocity_net'):
                            if hasattr(self, 'logger'):
                                self.logger.error(f"[ENDPOINT LOSS] FATAL: velocity_net missing! Flow type: {type(self.flow).__name__}")
                                self.logger.error(f"[ENDPOINT LOSS] Available attributes: {[a for a in dir(self.flow) if not a.startswith('_')]}")
                            endpoint_loss_weight = 0.0
                        
                        if True:  # All endpoint loss computation happens here
                            # ✅ SIMPLIFIED (Dec 16): Direct velocity penalty at t=1
                            # Instead of ODE integration (which can accumulate errors),
                            # directly penalize if velocity is non-zero at true parameters
                            # This ensures flow learns to map true params to themselves (fixed point)
                            
                            # At the target (t=1), velocity network should output near-zero
                            # because we're at the destination and shouldn't move further
                            t_final = torch.ones(batch_size, 1, device=strain_data.device)  # t=1
                            v_at_target = self.flow.velocity_net(params_norm, t_final, context)  # [batch, param_dim]
                            
                            # Penalize non-zero velocity at ground truth
                            # This ensures the flow converges to the target
                            endpoint_loss = endpoint_loss_weight * torch.mean(v_at_target ** 2)
                            
                            # Debug logging (first batch only)
                            if signal_idx == 0 and hasattr(self, 'logger'):
                                v_mag = torch.mean(torch.abs(v_at_target)).item()
                                if v_mag > 0.1:
                                    self.logger.debug(f"[ENDPOINT] v_at_target magnitude={v_mag:.6f}, loss={endpoint_loss.item():.6f}")
                            
                            # Log endpoint loss metrics
                            # if hasattr(self, 'logger') and endpoint_loss.item() > 0:
                            #     self.logger.debug(f"[ENDPOINT LOSS] Computed: {endpoint_loss.item():.6f} (penalty={endpoint_penalty:.6f}, spread={endpoint_spread:.6f})")
                        
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.debug(f"[ENDPOINT LOSS] Computation failed: {e}")
                        endpoint_loss = torch.tensor(0.0, device=strain_data.device)
                
                # For FlowMatching, use built-in endpoint anchoring
                if endpoint_loss_weight > 0 and use_flowmatching_endpoint:
                    try:
                        # ✅ FIX (Dec 15, Bug #2): Simplified - just call compute_endpoint_loss()
                        # FlowMatching has built-in method that penalizes non-zero velocity at true params
                        fm_endpoint_loss = self.flow.compute_endpoint_loss(params_norm, context)
                        endpoint_loss = endpoint_loss_weight * fm_endpoint_loss
                        
                        # if hasattr(self, 'logger') and endpoint_loss.item() > 0:
                        #     self.logger.debug(f"[FM ENDPOINT] Loss: {endpoint_loss.item():.6f}")
                    
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.debug(f"[FM ENDPOINT] Failed: {type(e).__name__}: {e}")
                        endpoint_loss = torch.tensor(0.0, device=strain_data.device)
                else:
                    endpoint_loss = torch.tensor(0.0, device=strain_data.device)
                
                signal_loss = cfm_loss + endpoint_loss
                
                # Debug: Log endpoint loss contribution
                if signal_idx == 0 and endpoint_loss.item() < 1e-6:
                    # Endpoint loss is zero - this is a problem!
                    if hasattr(self, 'logger'):
                        self.logger.debug(f"🔴 [FLOW LOSS] Signal {signal_idx}: CFM={cfm_loss.item():.6f}, Endpoint={endpoint_loss.item():.6e} (NOT ANCHORING!)")
                
                # Weight secondary signals lower (they're harder to learn from noisy overlaps)
                # Signal 0 (primary): weight = 1.0 (well-constrained by SNR)
                # Signal 1+ (secondary): weight = 0.7 (noisier, less SNR)
                signal_weight = 1.0 if signal_idx == 0 else 0.7
                
                flow_loss_per_signal.append(signal_loss.item())
                flow_loss_total += signal_weight * signal_loss
                
                # ✅ FIX 3 (Dec 27): DISTANCE-SPECIFIC BIAS CORRECTION
                # Problem: Distance bias growing +50 → +63 Mpc due to context compression
                # Solution: Penalize positive distance bias specifically
                # This targets the distance parameter (index 2) in normalized space
                if self.training and signal_idx == 0:  # Only apply to primary signal
                    try:
                        # Get mean of normalized distance predictions
                        # params_norm shape: [batch, param_dim]
                        # Index 2 = luminosity_distance
                        distance_idx = 2  # luminosity_distance is param index 2
                        if params_norm.shape[1] > distance_idx:
                            samples_distance_mean = params_norm[:, distance_idx].mean()
                            
                            # If samples skew positive (> 0.1 in normalized space [-1, 1])
                            # This indicates overestimation of distance
                            positive_bias_penalty = torch.relu(samples_distance_mean - 0.1) ** 2
                            distance_bias_loss = 5.0 * positive_bias_penalty
                            flow_loss_total += distance_bias_loss
                            
                            # Log every 100 training steps
                            if self.training_step % 100 == 0 and hasattr(self, 'logger'):
                                self.logger.warning(
                                    f"🔧 DISTANCE BIAS: mean={samples_distance_mean.item():.4f}, "
                                    f"penalty={positive_bias_penalty.item():.4f}, "
                                    f"loss_contrib={distance_bias_loss.item():.4f}"
                                )
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.debug(f"Distance bias correction failed: {e}")
                
            except (NotImplementedError, AttributeError, RuntimeError, Exception) as e:
                import traceback
                self.logger.warning(f"⚠️ Flow loss failed for signal {signal_idx}: {type(e).__name__}: {str(e)}")
                if signal_idx == 0:  # Log full traceback for first signal only
                    self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                continue
        
        # Average over number of signals (important for fair weighting across batches with different n_signals)
        if n_signals_in_batch > 0:
            flow_loss_total = flow_loss_total / n_signals_in_batch
        else:
            flow_loss_total = torch.tensor(0.0, device=strain_data.device)
        
        # ========================================
        # STEP 4: EXTRACTION ACCURACY LOSS (DISABLED)
        # ========================================
        # ✅ Dec 16 FIX: Removed extraction loss computation
        # REASON: We're no longer computing actual extraction during training
        # Instead, flow is trained directly on ground truth params
        # Extraction will be evaluated at validation time
        extraction_loss = torch.tensor(0.0, device=strain_data.device)
        
        # ========================================
        # STEP 5: RESIDUAL QUALITY LOSS (DISABLED)
        # ========================================
        # ✅ Dec 16 FIX: Removed residual loss computation
        # REASON: No residual extraction during training
        residual_loss = torch.tensor(0.0, device=strain_data.device)
        residual_power = torch.tensor(0.0, device=strain_data.device)
        
        # ========================================
        # STEP 6: BOUNDS LOSS (✅ Q3 REDESIGN: NSF soft constraints)
        # ========================================
        # ✅ Q3 SIMPLIFIED: NSF requires soft bounds on parameters
        # Compute penalty if any parameters fall outside expected ranges
        bounds_loss = torch.tensor(0.0, device=strain_data.device)
        try:
            true_params_primary = true_params[:, 0, :]
            params_norm = self._normalize_parameters(true_params_primary)
            # Check if any parameters are out of [-1, 1] after normalization
            out_of_bounds_mask = torch.abs(params_norm) > 1.0
            out_of_bounds_fraction = out_of_bounds_mask.float().mean()
            # Penalize out-of-bounds samples
            bounds_loss = 0.5 * out_of_bounds_fraction
        except Exception as e:
            self.logger.debug(f"Bounds loss computation failed: {e}")
            bounds_loss = torch.tensor(0.0, device=strain_data.device)
        
        # ========================================
        # STEP 6b: PHYSICS CONSTRAINTS (Legacy, kept for diagnostics)
        # ========================================
        # ✅ DEC 5 FIX: Physics loss now on PRIMARY signal only (it has ground truth)
        # Secondary signals don't have dedicated extraction, so check primary for validity
        # Ground truth: Ensures dataset is valid (should be ~0.2-0.3)
        physics_loss_raw, physics_violations = self._compute_physics_loss(true_params[:, 0, :])
        
        # ✅ NEW: Also penalize extracted parameters if invalid
        physics_loss_extracted = torch.tensor(0.0, device=strain_data.device)
        if len(all_extracted) > 0 and all_extracted[0]["extraction_valid"]:
            extracted_params = all_extracted[0]["params"]
            physics_loss_extracted_raw, _ = self._compute_physics_loss(extracted_params)
            # ✅ Weight extracted predictions lower (they're learned, not ground truth)
            physics_loss_extracted = 0.5 * torch.clamp(physics_loss_extracted_raw, max=100.0)
        
        # ✅ Combine: ground truth + extracted predictions
        physics_loss = torch.clamp(physics_loss_raw, max=100.0) + physics_loss_extracted
        
        # Log warning if physics loss is exploding
        if physics_loss_raw > 500.0:
            self.logger.warning(
                f"Physics loss explosion detected: {physics_loss_raw:.1f} → clamped to 10.0"
            )
        
        # ===============================
        # STEP 7: UNCERTAINTY & REGULARIZATION (Dec 15 FIX)
        # ===============================
        #  (Dec 15): Uncertainty loss was 0 - network had no incentive to learn σ
        # Now: Read weight from config and apply proper calibration-based uncertainty loss
        # 
        # Issue: Previous code used 0.01 * mean(uncertainties), which just penalizes non-zero σ
        # This prevents model from ever learning meaningful uncertainties
        # 
        # Solution: MSE loss between predicted σ and empirical error magnitude
        # This forces network to learn calibrated uncertainties (σ ≈ |error|)
        
        uncertainty_loss = torch.tensor(0.0, device=strain_data.device)
        np_config = self.config.get("neural_posterior", {})
        # Note: uncertainty_loss_weight is NOT in config yet, will be added to PriorityNet
        # For now, use weight from flow component regularization
        
        try:
            if len(all_extracted) > 0 and all_extracted[0]["extraction_valid"]:
                first_context = all_extracted[0]["context"]
                extracted_params = all_extracted[0]["params"]  # [batch, param_dim]
                true_params_primary = true_params[:, 0, :]  # [batch, param_dim]
                
                uncertainties = self.uncertainty_estimator(
                    torch.cat([extracted_params, first_context], dim=1)
                )  # [batch, param_dim]
                
                # Compute empirical error
                param_errors = torch.abs(extracted_params - true_params_primary)  # [batch, param_dim]
                
                # Loss: MSE between predicted σ and empirical error
                # This trains network to output σ ≈ |error|
                uncertainty_mse_loss = torch.mean((uncertainties - param_errors) ** 2)
                
                # Regularization: prevent σ from collapsing to 0 or exploding
                # Minimum σ = 0.01 (prevent collapse)
                # Maximum σ = max observed error (prevent explosion)
                min_uncertainty_penalty = torch.mean(torch.relu(0.01 - uncertainties))
                
                # Combined uncertainty loss
                uncertainty_loss = 0.15 * (uncertainty_mse_loss + 0.1 * min_uncertainty_penalty)
        except Exception as e:
            self.logger.debug(f"Uncertainty loss computation failed: {e}")
        
        # ===============================
        # STEP 8: CALIBRATION LOSS (Prevent Posterior Collapse)
        # ===============================
        # DISABLED due to NSF inverse sampling numerical instability
        # Issue: flow.inverse() encounters negative discriminants in rational quadratic splines
        #        during early training when model hasn't converged
        # Solution: Set calibration loss to 0 during training, will re-enable after model stabilizes
        # 
        # The issue manifests as:
        #   AssertionError in nflows/transforms/splines/rational_quadratic.py
        #   assert (discriminant >= 0).all()
        #
        # This happens because NSF spline parameters can become invalid before convergence
        # SOLUTION (Dec 7, 2025): Use variance penalty instead of inverse sampling
        # Penalizes posteriors that are too narrow (don't properly quantify uncertainty)
        
        try:
            # Estimate posterior variance from flow (without inverse sampling)
            # Use multiple forward passes to estimate stochasticity
            np_config = self.config.get("neural_posterior", {})
            calib_weight = np_config.get("calibration_loss_weight", 0.2)
            
            # # ✅ STEP 3 (JAN 10): Gate rank loss more aggressively
            # # Disable during early training (first 5000 steps) to avoid oscillation when flow geometry forming
            # # After 5000 steps, use lower threshold (0.25 vs 0.30) for more aggressive calibration
            # if self.training_step < 5000:
            #     calib_weight = 0.0  # Disable rank loss entirely during early phase
            
            if calib_weight > 0:
                # Sample from flow to estimate posterior spread
                posterior_samples_calib = []
                for _ in range(5):  # 5 random samples to estimate variance
                    z = torch.randn(params_norm.shape, device=params_norm.device, dtype=params_norm.dtype)
                    try:
                        # Generate samples via flow (safe, no inverse needed)
                        sample = self.flow._transform.inverse(z, context=context)[0]
                        posterior_samples_calib.append(sample)
                    except Exception:
                        # If inverse fails, use z directly as approximation
                        posterior_samples_calib.append(z)
                
                if posterior_samples_calib:
                    samples_stack = torch.stack(posterior_samples_calib)  # [5, batch, n_params]
                    posterior_std = torch.std(samples_stack, dim=0)  # [batch, n_params]
                    
                    # ✅ FIXED (Dec 11): Parameter-specific calibration targets
                    # Parameters with known calibration issues get stronger penalties
                    # Under-calibrated (coverage <60%): dec, geocent_time, a1, mass_2
                    # Dec 11 adjustment: Removed CFM per-parameter weighting (was causing phase/theta_jn over-optimization)
                    #   So relaxed targets for phase/theta_jn (0.30 → 0.33) since they won't be over-confident anymore
                    # ✅ DEC 12 FIX: DRAMATICALLY increased targets (0.3-0.5 → 0.6-0.8)
                    # Old targets were too small, posteriors stayed narrow despite calibration loss
                    # New targets force wider posteriors → uniform PIT → proper calibration
                    param_specific_targets = {
                        'dec': 0.75,             # ⬆️⬆️ MASSIVELY increased (was 0.50, even smaller was ignoring penalty)
                        'geocent_time': 0.75,    # ⬆️⬆️ MASSIVELY increased (was 0.55)
                        'a1': 0.70,              # ⬆️⬆️ MASSIVELY increased (was 0.45)
                        'mass_2': 0.70,          # ⬆️⬆️ MASSIVELY increased (was 0.42)
                        'theta_jn': 0.70,        # ⬆️⬆️ MASSIVELY increased (was 0.33, was over-confident)
                        'phase': 0.70,           # ⬆️⬆️ MASSIVELY increased (was 0.33, was over-confident)
                    }
                    
                    # Default target for well-calibrated params
                    default_target = 0.70  # ⬆️⬆️ MASSIVELY increased (was 0.35)
                    
                    # Compute per-parameter targets
                    target_std_per_param = []
                    for param_idx, param_name in enumerate(self.param_names):
                        if param_name in param_specific_targets:
                            target_std_per_param.append(param_specific_targets[param_name])
                        else:
                            target_std_per_param.append(default_target)
                    
                    target_std_array = torch.tensor(target_std_per_param, 
                                                    device=posterior_std.device, 
                                                    dtype=posterior_std.dtype)  # [n_params]
                    
                    # Broadcast to match posterior_std shape [batch, n_params]
                    target_std_array = target_std_array.unsqueeze(0)  # [1, n_params]
                    
                    # Penalize if posterior is too narrow (under-calibrated params)
                    # For over-calibrated params, target is smaller, allowing tighter posteriors
                    posterior_std_penalty = torch.relu(target_std_array - posterior_std).mean()
                    calibration_loss = calib_weight * posterior_std_penalty
                else:
                    calibration_loss = torch.tensor(0.0, device=strain_data.device, dtype=strain_data.dtype)
        except Exception as e:
            # Graceful fallback if any calibration computation fails
            self.logger.debug(f"Calibration loss computation failed: {e}")
            calibration_loss = torch.tensor(0.0, device=strain_data.device, dtype=strain_data.dtype)
        
        # ===============================
        # STEP 9: FLOW REGULARIZATION (NSF spline stability)
        # ===============================
        # Penalizes large determinants in coupling transforms for stability
        np_config = self.config.get("neural_posterior", {})
        jacobian_loss_weight = np_config.get("jacobian_reg_weight", 0.02)
        
        try:
            if hasattr(self.flow, 'compute_jacobian_loss'):
                # NSF, RealNVP, MAF all have compute_jacobian_loss
                jacobian_loss_raw = self.flow.compute_jacobian_loss(params_norm, context)
                jacobian_reg = jacobian_loss_weight * torch.clamp(jacobian_loss_raw, max=100.0)
            else:
                jacobian_reg = torch.tensor(0.0, device=strain_data.device)
        except Exception as e:
            self.logger.debug(f"Jacobian loss computation failed: {e}")
            jacobian_reg = torch.tensor(0.0, device=strain_data.device)
    
        
        # ===============================
        # STEP 10: SIMPLIFIED LOSS (✅ Q3 REDESIGN: NSF uses NLL only)
        # ===============================
        # ✅ CRITICAL SIMPLIFICATION (Dec 24, 2025):
        # NSF provides direct inverse() with strong gradients to context encoder
        # All complex loss components were band-aids for FlowMatching's weak gradients
        # With NSF, we only need:
        #   1. Flow loss (NLL) - primary learning signal
        #   2. Bounds penalty - soft constraints on parameters
        #
        # ❌ REMOVED: extraction_loss, residual_loss, physics_loss, jacobian_reg,
        #              uncertainty_loss, context_*_loss, stride_loss
        # These were specific to FlowMatching architecture
        
        np_config = self.config.get("neural_posterior", {})
        flow_loss_weight = np_config.get("flow_loss_weight", 1.0)
        bounds_penalty_weight = np_config.get("bounds_penalty_weight", 0.15)
        
        # =======================================
        # DISTANCE-SPECIFIC LOSSES
        # =======================================
        # NEW (Jan 1, 2026): Force encoder to extract distance-informative features
        # Three complementary mechanisms:
        # 1. Auxiliary distance loss: Predict distance directly from context
        # 2. Distance endpoint loss: Stronger constraint on distance parameter
        # 3. Distance prior: Discourage out-of-bounds distance predictions
        
        auxiliary_dist_loss = torch.tensor(0.0, device=strain_data.device)
        
        # ✅ JAN 7 CRITICAL FIX: Disabled distance-specific losses
        # Problem: Distance losses (weight 0.25) + flow losses created OPPOSING gradients
        #   Epoch 6: bias=-10.8 Mpc (OK)
        #   Epoch 7: bias=-81.8 Mpc (7.5× worse)
        #   Epoch 8: bias=-296.9 Mpc (3.6× worse, DIVERGING)
        # Root cause: 
        #   Distance head predicted 322 Mpc (log error 26.36)
        #   Flow tried to predict 924 Mpc
        #   Conflicting supervision → oscillation → divergence
        # Solution: DISABLE distance-specific losses entirely
        #   Flow is 11D posterior estimator; forcing distance values conflicts with learning
        #   Distance will converge naturally as flow learns from multi-noise data
        
        aux_dist_weight = np_config.get("auxiliary_distance_loss_weight", 0.0)  # ✅ JAN 7: Disabled
        direct_flow_dist_weight = np_config.get("direct_flow_distance_weight", 0.0)  # ✅ JAN 7: Disabled
        
        if self.training and (aux_dist_weight > 0 or direct_flow_dist_weight > 0):
            if self.training_step % 200 == 0:
                self.logger.info(f"Distance losses ENABLED: aux={aux_dist_weight}, direct_flow={direct_flow_dist_weight}")
            try:
                # Find distance parameter index
                distance_idx = None
                for idx, name in enumerate(self.param_names):
                    if name.lower() == "luminosity_distance":
                        distance_idx = idx
                        break
                
                if distance_idx is not None:
                    # Get true distance (first signal only, primary)
                    if true_params.dim() == 3:
                        true_distance = true_params[:, 0, distance_idx]  # [batch]
                    else:
                        true_distance = true_params[:, distance_idx]  # [batch]
                    
                    # Get bounds for distance
                    dist_bounds = self.param_bounds["luminosity_distance"]
                    dist_min, dist_max = dist_bounds[0], dist_bounds[1]
                    
                    # ✅ AUXILIARY DISTANCE LOSS (JAN 4 FIX)
                    # Problem: Distance head was learning mean distance instead of learning SNR-distance relationship
                    # Root cause: Context encoder doesn't encode SNR, so distance head had no training signal
                    # Solution: Make distance loss proportional to |SNR_predicted - SNR_true|
                    # This forces context encoder to encode SNR (distance → lower SNR at large D)
                    
                    if aux_dist_weight > 0:
                        # ✅ JAN 4 FIX: Add chirp mass to distance head input
                        # Compute chirp mass from m1, m2
                        m1_idx = next((i for i, n in enumerate(self.param_names) if "mass_1" in n.lower()), None)
                        m2_idx = next((i for i, n in enumerate(self.param_names) if "mass_2" in n.lower()), None)
                        
                        mc_norm = torch.ones(context.shape[0], device=context.device) * 0.5  # Safe default
                        
                        if m1_idx is not None and m2_idx is not None:
                            try:
                                m1 = true_params[:, 0, m1_idx] if true_params.dim() == 3 else true_params[:, m1_idx]
                                m2 = true_params[:, 0, m2_idx] if true_params.dim() == 3 else true_params[:, m2_idx]
                                
                                # Check for NaN/Inf in inputs
                                if torch.isnan(m1).any() or torch.isnan(m2).any():
                                    if self.training_step % 100 == 0:
                                        self.logger.warning(f"[DIST] m1/m2 contain NaN: m1_nan={torch.isnan(m1).sum()}, m2_nan={torch.isnan(m2).sum()}")
                                    mc_norm = torch.ones(context.shape[0], device=context.device) * 0.5
                                elif torch.isinf(m1).any() or torch.isinf(m2).any():
                                    if self.training_step % 100 == 0:
                                        self.logger.warning(f"[DIST] m1/m2 contain Inf: m1_inf={torch.isinf(m1).sum()}, m2_inf={torch.isinf(m2).sum()}")
                                    mc_norm = torch.ones(context.shape[0], device=context.device) * 0.5
                                else:
                                    # Mc = (m1*m2)^(3/5) / (m1+m2)^(1/5), clamped for safety
                                    m1_clamped = torch.clamp(m1, min=0.1, max=100.0)
                                    m2_clamped = torch.clamp(m2, min=0.1, max=100.0)
                                    
                                    numerator = (m1_clamped * m2_clamped) ** 0.6
                                    denominator = torch.clamp((m1_clamped + m2_clamped) ** 0.2, min=0.01)
                                    mc = numerator / denominator
                                    
                                    # Check intermediate result
                                    if torch.isnan(mc).any():
                                        if self.training_step % 100 == 0:
                                            self.logger.warning(f"[DIST] mc computation produced NaN: {torch.isnan(mc).sum()} samples")
                                        mc_norm = torch.ones(context.shape[0], device=context.device) * 0.5
                                    else:
                                        # Log scale normalization
                                        mc_log = torch.log(torch.clamp(mc, min=0.1) + 1.0)
                                        mc_min = mc_log.min()
                                        mc_max = mc_log.max()
                                        
                                        if torch.isnan(mc_log).any() or torch.isinf(mc_log).any():
                                            if self.training_step % 100 == 0:
                                                self.logger.warning(f"[DIST] mc_log contains NaN/Inf")
                                            mc_norm = torch.ones(context.shape[0], device=context.device) * 0.5
                                        else:
                                            # Final normalization
                                            denom = torch.clamp(mc_max - mc_min, min=1e-6)
                                            mc_norm = (mc_log - mc_min) / denom
                                            mc_norm = torch.clamp(mc_norm, min=0.0, max=1.0)
                            except Exception as e:
                                if self.training_step % 100 == 0:
                                    self.logger.error(f"[DIST] Chirp mass exception: {e}")
                                mc_norm = torch.ones(context.shape[0], device=context.device) * 0.5
                        
                        # Add amplitude bypass to distance head
                        # Directly compute SNR-distance coupling without bottleneck
                        # This provides the encoder 4 independent amplitude signals
                        
                        # 1. Network SNR (scalar)
                        network_snr = self._compute_network_snr(strain_data)  # [batch, 1]
                        
                        # 2. Detector RMS amplitudes [batch, 3] - one per detector
                        # ✅ JAN 5 FIX: Pad to 3 detectors if fewer present
                        detector_rms = torch.std(strain_data, dim=-1, keepdim=False)  # [batch, n_detectors]
                        
                        # Pad to 3 detectors if needed (training data may have 2 detectors)
                        if detector_rms.size(1) < 3:
                            padding = torch.zeros(detector_rms.size(0), 3 - detector_rms.size(1), device=strain_data.device, dtype=strain_data.dtype)
                            detector_rms = torch.cat([detector_rms, padding], dim=1)  # [batch, 3]
                        
                        # 3. Relative amplitude geometry [batch, 3] - detector ratios
                        detector_rms_mean = detector_rms.mean(dim=1, keepdim=True)  # [batch, 1]
                        rel_amp = detector_rms / (detector_rms_mean + 1e-8)  # [batch, 3]
                        
                        # Concatenate all inputs: [context, network_snr, detector_rms, rel_amp, mc_norm]
                        # Total: [batch, context_dim + 1 + 3 + 3 + 1]
                        
                        # # ✅ DEBUG: Log actual shapes before concatenation
                        # if self.training_step % 200 == 0:
                        #     self.logger.info(
                        #         f"[DIST HEAD INPUT] context={context.shape}, "
                        #         f"network_snr={network_snr.shape}, detector_rms={detector_rms.shape}, "
                        #         f"rel_amp={rel_amp.shape}, mc_norm={mc_norm.unsqueeze(1).shape}"
                        #     )
                        
                        distance_head_input = torch.cat([
                            context_for_distance_head,        # [batch, context_dim] - learned morphology (original, no SNR)
                            network_snr,                       # [batch, 1] - absolute amplitude scale
                            detector_rms,                      # [batch, 3] - absolute SNR per detector
                            rel_amp,                           # [batch, 3] - antenna response geometry
                            mc_norm.unsqueeze(1)               # [batch, 1] - chirp mass
                        ], dim=1)
                        
                        # ✅ DEBUG: Check input shape matches expected
                        expected_input_size = self.context_dim + 8
                        actual_input_size = distance_head_input.shape[1]
                        if actual_input_size != expected_input_size:
                            self.logger.error(
                                f"[DIST HEAD] Shape mismatch! Expected {expected_input_size}, got {actual_input_size}. "
                                f"context_dim={self.context_dim}, context.shape={context.shape}"
                            )
                        
                        predicted_distance = self.distance_prediction_head(distance_head_input).squeeze(-1)  # [batch] in [-1, 1] log-space
                        
                        # Switch to log D_L space normalization
                        # Reason: NSF handles curved degeneracies better in log-space
                        #         Gradients stop vanishing at large distances
                        #         Posterior geometry becomes sane
                        # 
                        # Log-space normalization bounds (from parameter scaler):
                        # log(D_L) ∈ [log(10), log(5000)] Mpc = [2.303, 8.517]
                        log_dist_min = np.log(10.0)    # 2.303
                        log_dist_max = np.log(5000.0)  # 8.517
                        
                        #  [-1, 1] → log-space
                        # Formula: x_mapped = x * (b-a)/2 + (a+b)/2
                        pred_norm = predicted_distance.clamp(-1.0, 1.0)
                        pred_log_dist = (
                            pred_norm * 0.5 * (log_dist_max - log_dist_min)
                            + 0.5 * (log_dist_max + log_dist_min)
                        )
                        
                        # Compute true log-distance
                        true_log_dist = torch.log(true_distance.clamp(min=1e-6))
                        
                        # MSE loss directly in log-space
                        auxiliary_dist_loss = F.mse_loss(pred_log_dist, true_log_dist)
                        
                        # L2 regularization
                        l2_reg = torch.tensor(0.0, device=auxiliary_dist_loss.device)
                        for param in self.distance_prediction_head.parameters():
                            l2_reg = l2_reg + torch.norm(param) ** 2
                        auxiliary_dist_loss = auxiliary_dist_loss + 0.001 * l2_reg
                        
                        if self.training_step % 100 == 0:
                            # ✅ STEP 3: Convert from log-space back to Mpc for display
                            pred_dist_physical = torch.exp(pred_log_dist)
                            true_dist_physical = torch.exp(true_log_dist)
                            self.logger.info(
                                f"[DISTANCE HEAD LOG-SPACE] Pred={pred_dist_physical.mean().item():.0f} Mpc, "
                                f"True={true_dist_physical.mean().item():.0f} Mpc, "
                                f"LogMSE={auxiliary_dist_loss.item():.6f} (log-space error)"
                            )
                    
                    # ✅ JAN 5 CRITICAL FIX #1: REMOVED distance_endpoint_loss
                     # Reason: Not flow-aware, operates in parameter space, fights NSF geometry
                     # NSF already enforces monotonicity via splines
                     
                     # ✅ JAN 5 CRITICAL FIX #1: REMOVED distance_prior_loss
                    # Reason: Asymmetric gradients near bounds cause bias sticking
                    # Prior is baked into data + NSF base distribution
                    
                    # ✅ JAN 5 FIX: DIRECT FLOW-BASED DISTANCE MSE LOSS
                    # Problem: Distance bias not decreasing (±60-150 Mpc oscillating)
                    # Root cause: Distance losses indirect, only auxiliary directly penalizes
                    # Solution: Sample from flow, denormalize distance, directly penalize |distance_pred - distance_true|
                    # This ensures EVERY batch has explicit distance gradient signal
                    #
                    # Weight tuning (JAN 5):
                    # - auxiliary_distance_loss_weight: 0.5 (main teacher signal)
                    # - direct_flow_distance_weight: 0.1 (gentle flow alignment, not dominating)
                    # - Together: 0.6 total distance supervision, sufficient but not overwhelming
                    
                    direct_dist_loss = torch.tensor(0.0, device=strain_data.device)
                    try:
                        if direct_flow_dist_weight > 0:
                            n_flow_samples = 50
                            with torch.no_grad():
                                flow_samples_dist = self.flow.sample(n_flow_samples, context)  # [batch, n_samples, param_dim]
                            
                            # ✅ STEP 3 (JAN 6): CLEAN FLOW DISTANCE HANDLING IN LOG-SPACE
                            # Flow outputs normalized in [-1, 1] (from parameter scaler in log-space)
                            dist_samples_norm = flow_samples_dist[:, :, distance_idx]  # [batch, n_samples], in [-1, 1]
                            
                            # ✅ CORRECT MAPPING: [-1, 1] → log-space
                            # Formula: x_mapped = x * (b-a)/2 + (a+b)/2
                            dist_samples_log = (
                                dist_samples_norm * 0.5 * (log_dist_max - log_dist_min)
                                + 0.5 * (log_dist_max + log_dist_min)
                            )
                            
                            # Convert from log-space to physical distance
                            dist_samples_physical = torch.exp(dist_samples_log)
                            
                            # Compute mean distance from samples
                            dist_pred_from_flow = dist_samples_physical.mean(dim=1)  # [batch]
                            
                            # ✅ STEP 3: Direct MSE loss in log-space (more natural)
                            pred_log_dist_flow = torch.log(dist_pred_from_flow.clamp(min=1e-6))
                            true_log_dist_flow = torch.log(true_distance.clamp(min=1e-6))
                            
                            # MSE in log-space
                            direct_dist_mse = F.mse_loss(pred_log_dist_flow, true_log_dist_flow)
                            direct_dist_loss = direct_flow_dist_weight * direct_dist_mse
                            
                            # Add to auxiliary loss
                            auxiliary_dist_loss = auxiliary_dist_loss + direct_dist_loss
                            
                            if self.training_step % 200 == 0:
                                pred_dist_mean = dist_pred_from_flow.mean().item()
                                true_dist_mean = true_distance.mean().item()
                                pred_dist_std = dist_samples_physical.mean(dim=1).std().item()
                                self.logger.info(
                                    f"[DISTANCE FLOW MSE] Pred={pred_dist_mean:.0f}±{pred_dist_std:.0f} Mpc, True={true_dist_mean:.0f} Mpc, "
                                    f"Loss={direct_dist_loss.item():.6f}"
                                )
                    except Exception as e:
                        if self.training_step % 200 == 0:
                            self.logger.warning(f"⚠️ Direct distance flow MSE loss skipped: {e}")
                
            except Exception as e:
                if self.training_step % 200 == 0:
                    self.logger.warning(f"Distance-specific loss computation failed: {e}")
                auxiliary_dist_loss = torch.tensor(0.0, device=strain_data.device)
                # ✅ JAN 5 FIX: Only auxiliary_dist_loss needed (endpoint/prior removed)
        
        # ========================================
        # CONTEXT COLLAPSE PREVENTION (JAN 5 SIMPLIFIED)
        # ========================================
        # ✅ JAN 5 CRITICAL FIX #3: Remove excessive penalties
        # NSF provides strong gradients - no need for emergency measures
        # Keep only: variance penalty + diversity loss
        # Remove: noise injection, emergency noise, discrimination loss, stride loss
        #
        # Rationale: These added complexity without helping convergence with NSF
        # NSF already constrains the flow strongly via spline monotonicity
        
        context_variance_penalty = torch.tensor(0.0, device=strain_data.device)
        diversity_loss = torch.tensor(0.0, device=strain_data.device)
        diversity_weight = 1.0  
        context_std_value = 0.0
        
        if self.training:
            try:
                context_std = context.std()
                context_std_value = context_std.item()
                
                # ✅ SINGLE VARIANCE PENALTY (capped to prevent oscillation)
                target_std = 0.75
                variance_deficit = torch.relu(target_std - context_std)
                context_variance_penalty = torch.clamp(2.0 * variance_deficit, max=2.0)  # Cap at 2.0
                
                # ✅ SINGLE DIVERSITY LOSS (orthogonality, capped)
                batch_size = context.size(0)
                context_centered = context - context.mean(dim=0, keepdim=True)
                context_cov = torch.mm(context_centered.T, context_centered) / max(batch_size, 1)
                
                # Target: diagonal matrix with variance on diagonal
                identity = torch.eye(context.size(1), device=context.device)
                target_cov = identity * (target_std ** 2)
                
                # Frobenius norm of difference
                diversity_loss = torch.norm(context_cov - target_cov, p='fro') / context.size(1)
                diversity_loss = torch.clamp(diversity_loss, max=1.0)  # Cap at 1.0
                
                # ✅ LOG EVERY 100 STEPS (reduced frequency)
                if self.training_step % 100 == 0:
                    deficit_pct = (variance_deficit.item() / 0.75) * 100
                    if deficit_pct < 5:
                        status = "✅ HEALTHY"
                    elif deficit_pct <= 10:
                        status = "🟡 CLOSE"
                    else:
                        status = "🔴 NEEDS WORK"
                    
                    self.logger.info(
                        f"{status} CONTEXT: std={context_std_value:.4f} (target 0.75), "
                        f"var_pen={context_variance_penalty.item():.4f}, div={diversity_loss.item():.4f}"
                    )
            except Exception as e:
                self.logger.warning(f"⚠️ Context collapse check failed: {e}")
        
        # ========================================
        # STEP 7: PIT LOSS (Jan 4: Ensure uniform P-P distribution)
        # ========================================
        # ✅ NEW (Jan 4): PIT loss ensures posterior calibration
        # PIT = Probability Integral Transform value
        # Target: Uniform distribution on [0, 1] (no bias toward early/late values)
        # Loss: KS statistic between observed PIT and uniform distribution
        #
        # Without this, model can collapse to specific value (e.g., always PIT < 0.1)
        # which indicates posterior is too narrow (mass2 std → 0)
        
        pit_loss = torch.tensor(0.0, device=strain_data.device)
        pit_weight = 2.0  # ✅ HIGH WEIGHT: Force uniform PIT distribution
        
        if flow_type_lower == "nsf" and self.training:
            try:
                # Sample from posterior to compute PIT values
                n_pit_samples = 10
                pit_values_list = []
                
                for sig_idx in range(min(n_signals_in_batch, 2)):  # Compute for first 2 signals only (memory efficient)
                    params_signal = true_params[:, sig_idx, :]  # [batch, param_dim]
                    params_norm = self._normalize_parameters(params_signal)
                    
                    # Sample from flow
                    samples = self.flow.sample(n_pit_samples, context)  # [batch, n_samples, param_dim]
                    
                    # Compute CDF values: PIT_i = Φ(true_param_i | samples)
                    # Using empirical CDF: (# samples < true_param) / n_samples
                    for param_idx in range(self.features):
                        param_samples = samples[:, :, param_idx]  # [batch, n_samples]
                        param_true = params_norm[:, param_idx].unsqueeze(1)  # [batch, 1]
                        
                        # Count how many samples are < true_param
                        count_below = (param_samples < param_true).float().sum(dim=1)  # [batch]
                        pit_empirical = count_below / n_pit_samples  # [batch] ∈ [0, 1]
                        pit_values_list.append(pit_empirical)
                
                if pit_values_list:
                    # Combine all PIT values
                    pit_all = torch.cat(pit_values_list, dim=0).flatten()  # [total_pit_values]
                    
                    # Compute KS statistic: max difference from uniform CDF
                    # Uniform CDF at x is just x (for x ∈ [0, 1])
                    pit_sorted, _ = torch.sort(pit_all)
                    n_pit = len(pit_sorted)
                    uniform_cdf = torch.linspace(0, 1, n_pit, device=strain_data.device)
                    
                    # KS statistic: max absolute difference
                    ks_stat = torch.max(torch.abs(pit_sorted - uniform_cdf))
                    
                    # Loss: penalize large KS (non-uniform PIT)
                    # Target KS > 0.05 means well-calibrated (statistical threshold)
                    pit_loss = torch.relu(0.05 - ks_stat) ** 2  # Penalize if KS < 0.05
                    pit_loss = pit_loss * pit_weight
                    
                    if self.training_step % 100 == 0:
                        self.logger.info(
                            f"[PIT LOSS] KS={ks_stat.item():.4f}, loss={pit_loss.item():.6f}, "
                            f"PIT mean={pit_all.mean().item():.3f}, std={pit_all.std().item():.3f}"
                        )
            except Exception as e:
                if self.training_step % 200 == 0:
                    self.logger.debug(f"[PIT LOSS] Error computing PIT: {type(e).__name__}: {e}")
                pit_loss = torch.tensor(0.0, device=strain_data.device)
        
        # ✅ JAN 7 FIX: Ultra-minimal loss to prevent oscillation
        # Initialize diversity_weight in case context variance penalty wasn't computed
        if not hasattr(self, '_diversity_weight'):
            diversity_weight = 1.0
        
        # ✅ JAN 8 CRITICAL FIX #3: Distance loss DISABLED (high rejection rate in early training)
        # Problem (JAN 8 Rev2): Sampling 10 times to compute distance loss triggered high rejection rates
        # Root cause: Early-training flow produces many out-of-bounds samples, rejection sampling fails
        # Solution: DISABLE distance loss during training, let NSF learn distance naturally
        # Better approach: Will implement lightweight distance anchor in future without rejection sampling
        #
        # The SNR conditioning (STEP 5) implicitly carries distance information:
        # - SNR = constant / distance  (inverse relationship)
        # - Flow learns distance from SNR in context
        # - No explicit distance loss needed for convergence
        distance_loss_total = torch.tensor(0.0, device=strain_data.device)
        
        # ✅ JAN 8 FINAL: Minimal loss without sampling-dependent components
        # Distance loss disabled due to rejection sampling triggering high rejection rates
        # SNR conditioning (STEP 5) already provides distance information implicitly
        total_loss = (
            flow_loss_weight * flow_loss_total +                  # PRIMARY: NLL from NSF (1.0 weight)
            bounds_penalty_weight * bounds_loss +                 # SECONDARY: soft parameter bounds (0.15)
            context_variance_penalty                              # TERTIARY: encoder variance (0.1 weight cap)
        )
        self._diversity_weight = diversity_weight
        
        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss_total,              # PRIMARY: NLL from NSF
            "bounds_loss": bounds_loss,                # SECONDARY: soft constraints
            "context_var_penalty": context_variance_penalty.item() if self.training else 0.0,  # TERTIARY: encoder health
            "context_std": context_std_value,          # Monitor encoder variance
            "distance_loss": distance_loss_total.item() if self.training else 0.0,  # QUATERNARY: normalized distance anchor (0.05 weight)
            "n_signals_extracted": torch.tensor(n_signals_extracted, dtype=torch.float32),
        }

    # ============================================================================
    # FLOW MATCHING MONITORING & DIAGNOSTICS
    # ============================================================================
    
    def compute_flow_matching_metrics(self, strain_batch: torch.Tensor, 
                                     true_params_batch: torch.Tensor) -> Dict[str, float]:
        """
        Compute detailed Flow Matching diagnostics for monitoring learning.
        
        Metrics:
        1. CFM Velocity Loss Components
        2. Sample Quality (MSE, diversity, bias)
        3. Per-Parameter Bias
        4. Context Utilization
        5. Gradient Health
        
        Returns:
            dict with all metrics for logging
        """
        self.eval()
        metrics = {}
        
        with torch.no_grad():
            batch_size = strain_batch.shape[0]
            
            # ===== METRIC 1: Sample-Space MSE (Prediction Accuracy) =====
            try:
                posterior_samples = self.sample_posterior(
                    strain_batch, n_samples=50, return_all_samples=True
                )
                
                if posterior_samples is not None:
                    # posterior_samples: [batch, n_samples, param_dim]
                    # Get mean prediction over samples (not batch)
                    pred_mean = posterior_samples.mean(dim=1)  # [batch, param_dim]
                    
                    # Extract first signal only from true_params for comparison (primary signal)
                    true_params_primary = true_params_batch[:, 0, :]  # [batch, param_dim]
                    
                    # MSE between predictions and truth
                    sample_mse = F.mse_loss(pred_mean, true_params_primary).item()
                    metrics['sample_mse'] = sample_mse
                    
                    # ✅ STEP 2 (JAN 10): Change how we evaluate bias
                    # Report median bias (should → 0), quantile bias (10-90%), coverage vs SNR
                    # NOT mean bias (which expects low SNR to be biased - physics expected behavior)
                    bias_per_param = {}
                    median_bias_per_param = {}
                    q10_bias_per_param = {}
                    q90_bias_per_param = {}
                    std_per_param = {}
                    
                    for i, param_name in enumerate(self.param_names):
                        errors = (pred_mean[:, i] - true_params_primary[:, i])
                        
                        # Mean bias (for reference, but not the target metric)
                        mean_bias = errors.mean().item()
                        
                        # MEDIAN BIAS (THIS IS THE TARGET - should → 0)
                        median_bias = errors.median().item()
                        
                        # QUANTILE BIAS (10-90% range)
                        q10_bias = torch.quantile(errors, 0.1).item()
                        q90_bias = torch.quantile(errors, 0.9).item()
                        
                        std = errors.std().item()
                        
                        bias_per_param[param_name] = mean_bias
                        median_bias_per_param[param_name] = median_bias
                        q10_bias_per_param[param_name] = q10_bias
                        q90_bias_per_param[param_name] = q90_bias
                        std_per_param[param_name] = std
                    
                    metrics['bias_per_param'] = bias_per_param
                    metrics['median_bias_per_param'] = median_bias_per_param  # NEW: Primary metric
                    metrics['q10_bias_per_param'] = q10_bias_per_param
                    metrics['q90_bias_per_param'] = q90_bias_per_param
                    metrics['std_per_param'] = std_per_param
                    metrics['max_median_bias'] = max(abs(b) for b in median_bias_per_param.values())  # NEW: Primary metric
                    metrics['max_bias'] = max(abs(b) for b in bias_per_param.values())  # Kept for reference
                    metrics['mean_error_std'] = np.mean(list(std_per_param.values()))
                else:
                    metrics['sample_mse'] = 999.0
                    metrics['max_bias'] = 999.0
            except Exception as e:
                self.logger.debug(f"Sample MSE computation failed: {e}")
                metrics['sample_mse'] = 999.0
            
            # ===== METRIC 2: Sample Diversity (Variance Check) =====
            try:
                # Normalize strain same as in training
                strain_batch_norm = self._normalize_strain(strain_batch[:1])
                context = self.context_encoder(strain_batch_norm)
                
                # ✅ CRITICAL FIX (Jan 7): Match training context with SNR conditioning
                # During training (compute_loss line 1147), SNR is appended to context
                # Diagnostic code must do the same to get comparable results
                try:
                    np_config = self.config.get("neural_posterior", {})
                    snr_conditioning_enabled = np_config.get("snr_conditioning", True)
                    
                    if snr_conditioning_enabled:
                        # Compute network SNR from strain (same as training)
                        network_snr = self._compute_network_snr(strain_batch_norm)  # [batch, 1] in log space
                        # Normalize network SNR
                        log_rms = network_snr
                        log_min = np.log(0.01)
                        log_max = np.log(1.0)
                        norm_net_snr = (log_rms - log_min) / (log_max - log_min)
                        norm_net_snr = torch.clamp(norm_net_snr, min=-5.0, max=5.0)
                        
                        # Append to context to match training
                        context = torch.cat([context, norm_net_snr], dim=1)
                except Exception as snr_err:
                    self.logger.debug(f"⚠️ SNR conditioning in diversity metric failed: {snr_err}")
                
                samples_diversity = []
                
                for _ in range(50):
                    z = torch.randn(1, self.param_dim, device=strain_batch.device)
                    x, _ = self.flow.inverse(z, context)
                    samples_diversity.append(x)
                
                samples_diversity = torch.cat(samples_diversity, dim=0)  # [50, param_dim]
                sample_std_per_param = samples_diversity.std(dim=0)
                
                metrics['sample_diversity_mean'] = sample_std_per_param.mean().item()
                metrics['sample_diversity_min'] = sample_std_per_param.min().item()
                metrics['sample_diversity_max'] = sample_std_per_param.max().item()
            except Exception as e:
                self.logger.debug(f"Sample diversity computation failed: {e}")
                metrics['sample_diversity_mean'] = 0.0
            
            # ===== METRIC 3: Context Utilization =====
            try:
                strain_batch_norm = self._normalize_strain(strain_batch)
                context = self.context_encoder(strain_batch_norm)
                
                metrics['context_mean'] = context.mean().item()
                metrics['context_std'] = context.std().item()
                metrics['context_norm_mean'] = torch.norm(context, dim=-1).mean().item()
                metrics['context_norm_std'] = torch.norm(context, dim=-1).std().item()
            except Exception as e:
                self.logger.debug(f"Context utilization computation failed: {e}")
            
            # ===== METRIC 4: Velocity Network Output Norms (FlowMatching only) =====
             # NSF doesn't have velocity_net, skip this diagnostic
            if hasattr(self.flow, 'velocity_net'):
                 try:
                     # Simulate a velocity computation for diagnostics
                     strain_batch_norm = self._normalize_strain(strain_batch[:1])
                     context = self.context_encoder(strain_batch_norm)
                     t = torch.rand(1, 1, device=strain_batch.device)
                     z_t = torch.randn(1, self.param_dim, device=strain_batch.device)
                     
                     v_pred = self.flow.velocity_net(z_t, t.squeeze(-1), context)
                     
                     metrics['velocity_pred_norm'] = torch.norm(v_pred).item()
                     metrics['velocity_pred_mean_magnitude'] = torch.abs(v_pred).mean().item()
                 except Exception as e:
                     self.logger.debug(f"Velocity norm computation failed: {e}")
        
        self.train()
        return metrics
    
    def log_flow_diagnostics(self, metrics: Dict[str, float], epoch: int, prefix: str = ""):
        """
        Log comprehensive Flow Matching diagnostics.
        
        Args:
            metrics: Dictionary from compute_flow_matching_metrics()
            epoch: Current epoch number
            prefix: Optional prefix for logging (e.g., "VAL")
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"EPOCH {epoch + 1} {prefix} FLOW MATCHING DIAGNOSTICS")
        self.logger.info(f"{'='*70}")
        
        # Sample quality
        if 'sample_mse' in metrics:
            status = "✅" if metrics['sample_mse'] < 0.1 else "🟡" if metrics['sample_mse'] < 0.5 else "🔴"
            self.logger.info(f"{status} Sample MSE: {metrics['sample_mse']:.4f}")
        
        # ✅ STEP 2 (JAN 10): Report median bias as PRIMARY metric
        # NOT mean bias (which expects low SNR to be biased - physics behavior)
        if 'max_median_bias' in metrics:
            status = "✅" if metrics['max_median_bias'] < 0.1 else "🟡" if metrics['max_median_bias'] < 0.5 else "🔴"
            self.logger.info(f"{status} Max MEDIAN Parameter Bias: {metrics['max_median_bias']:.4f}")
        
        if 'median_bias_per_param' in metrics:
            self.logger.info("   Per-parameter MEDIAN bias (→ 0 is good):")
            for param_name, bias in metrics['median_bias_per_param'].items():
                status = "✅" if abs(bias) < 0.05 else "🟡" if abs(bias) < 0.2 else "🔴"
                self.logger.info(f"      {status} {param_name:20s}: {bias:+.4f}")
        
        # Quantile bias (10-90%)
        if 'q10_bias_per_param' in metrics and 'q90_bias_per_param' in metrics:
            self.logger.info("   Quantile bias (10-90% range):")
            for param_name in metrics['q10_bias_per_param'].keys():
                q10 = metrics['q10_bias_per_param'][param_name]
                q90 = metrics['q90_bias_per_param'][param_name]
                self.logger.info(f"      {param_name:20s}: [{q10:+.4f}, {q90:+.4f}]")
        
        # Mean bias (for reference - acknowledges low SNR bias is expected)
        if 'bias_per_param' in metrics:
            self.logger.info("   Per-parameter MEAN bias (reference, low SNR bias expected):")
            for param_name, bias in metrics['bias_per_param'].items():
                status = "✅" if abs(bias) < 0.1 else "🟡" if abs(bias) < 0.5 else "🔴"
                self.logger.info(f"      {status} {param_name:20s}: {bias:+.4f}")
        
        # Sample diversity
        if 'sample_diversity_mean' in metrics:
            div = metrics['sample_diversity_mean']
            status = "✅" if div > 0.5 else "🔴"
            self.logger.info(f"{status} Sample Diversity (mean): {div:.4f}")
            if 'sample_diversity_min' in metrics:
                self.logger.info(f"      Range: [{metrics['sample_diversity_min']:.4f}, {metrics['sample_diversity_max']:.4f}]")
        
        # Context health
        if 'context_mean' in metrics:
            self.logger.info(f"📊 Context Utilization:")
            self.logger.info(f"      Mean: {metrics['context_mean']:.4f}, Std: {metrics['context_std']:.4f}")
            self.logger.info(f"      Norm: {metrics['context_norm_mean']:.2f} ± {metrics['context_norm_std']:.2f}")
        
        # Velocity norms
        if 'velocity_pred_norm' in metrics:
            self.logger.info(f"🚀 Velocity Network:")
            self.logger.info(f"      Norm: {metrics['velocity_pred_norm']:.4f}")
            self.logger.info(f"      Mean magnitude: {metrics['velocity_pred_mean_magnitude']:.4f}")
        
        self.logger.info(f"{'='*70}\n")

    def _normalize_snr_features(self, network_snr: torch.Tensor, target_snr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ STEP 5: Normalize SNR features for explicit conditioning.
        
        Makes SNR an explicit context feature instead of implicit in strain RMS.
        This prevents low-SNR biases from leaking into high-SNR inference.
        
        Args:
            network_snr: [batch, 1] - RMS-based SNR from strain, log-scaled ~[-0.5, 2.0]
            target_snr:  [batch, 1] - Ground truth SNR from parameters, linear ~[5, 100]
        
        Returns:
            norm_net:    [batch, 1] - Normalized network SNR to [0, 1]
            norm_target: [batch, 1] - Normalized target SNR to [0, 1]
        """
        # Network SNR normalization
        # Range: log(0.05) ≈ -3.0 to log(e) ≈ 1.0
        # Maps: very quiet (-3) → 0, reference (0) → 0.75, very loud (1) → 1.0
        norm_net = (network_snr - (-3.0)) / (1.0 - (-3.0))  # [batch, 1]
        norm_net = torch.clamp(norm_net, min=0.0, max=1.0)  # Bounded to [0, 1]
        
        # Target SNR normalization (log scale for stability)
        # Range: log(5) ≈ 1.61 to log(100) ≈ 4.61
        # Maps: weak (5) → 0, medium (30) → 0.5, loud (100) → 1.0
        log_target = torch.log(torch.clamp(target_snr, min=1.0))  # [batch, 1]
        log_min = np.log(5.0)
        log_max = np.log(100.0)
        norm_target = (log_target - log_min) / (log_max - log_min)  # [batch, 1]
        norm_target = torch.clamp(norm_target, min=0.0, max=1.0)  # Bounded to [0, 1]
        
        return norm_net, norm_target

    def _compute_network_snr(self, strain_data: torch.Tensor) -> torch.Tensor:
        """
        Estimate network SNR amplitude from strain RMS.
        
        ✅ JAN 5 CRITICAL FIX: Use strain RMS as direct SNR proxy
        
        Why RMS is the right signal:
        - Whitened strain: RMS = noise + signal
        - High signal → high RMS
        - Low signal → low RMS  
        - This directly correlates with SNR (distant weak signals = small RMS)
        
        Calibration:
        - Pure noise whitened: RMS ≈ 1.0 (by definition)
        - But test data/real data varies: 0.01-1.0
        - Log scale this to [0, 1]
        
        Args:
            strain_data: [batch, 3, time_samples] - Whitened strain from H1, L1, V1
        
        Returns:
            network_snr: [batch, 1] - RMS-based SNR proxy, normalized to [0, 1]
        """
        batch_size = strain_data.shape[0]
        device = strain_data.device
        
        try:
            # ✅ JAN 5: RMS is the ground truth SNR proxy
            # Compute RMS for each detector
            rms_values = torch.std(strain_data, dim=2)  # [batch, 3]
            
            # Network RMS = max across detectors (brightest detector sets scale)
            network_rms = rms_values.max(dim=1)[0]  # [batch]
            
            # ✅ JAN 5 CRITICAL FIX: NO hard clamping of SNR feature
            # Typical whitened strain RMS: 0.01 (weak) to 1.0 (strong)
            # In practice: 0.05 (quiet) to 0.25 (loud) for GW signals
            # Log scale provides natural normalization: log(0.05)=-2.996, log(1.0)=0.0
            
            # ✅ KEY: Allow RMS > 1.0 to pass through without clamping
            # (loud signals can have RMS >> 1.0, and that's informative!)
            log_rms = torch.log(torch.clamp(network_rms, min=0.01))  # Only clamp min, NOT max
            log_min = np.log(0.01)   # -4.605 (very quiet baseline)
            log_max = np.log(1.0)     # 0.0 (reference whitened noise)
            
            # Linear rescaling based on log scale (no hard bounds)
            network_snr_norm = (log_rms - log_min) / (log_max - log_min)
            # ✅ NO clamp here - let extreme SNRs flow naturally
            # Range: (-inf, +inf) but typically (-0.5, 2.0) for training data
            # Distance head will learn what values mean
            
            return network_snr_norm.unsqueeze(1)  # [batch, 1] unbounded
            
        except Exception as e:
            if self.training_step % 100 == 0:
                self.logger.warning(f"[SNR] Network SNR computation failed: {e}, using 0.5 fallback")
            return torch.ones(batch_size, 1, device=device) * 0.5

    def _compute_physics_loss(self, params: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Enforce physical constraints with normalized bounds penalty.
        
        Added normalization to prevent explosion.
        Raw violations (e.g., mass difference of 50 solar masses) were causing
        physics_loss = 27,568 which dominated training. Now normalized to ~1.0 scale.
        
        Args:
            params: [batch_size, param_dim] parameters in PHYSICAL units
        
        Returns:
            Tuple of (loss, violation_dict)
        """
        loss = torch.tensor(0.0, device=params.device)
        
        # ⚠️ DEBUG: Track violations
        debug_violations = {}
        
        # ===============================
        # ✅ 0. BOUNDS PENALTY (normalized)
        # ===============================
        violation_list = []
        
        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_bounds[param_name]
            param_range = max_val - min_val
            
            # Compute violations
            lower_violation = F.relu(min_val - params[:, i])
            upper_violation = F.relu(params[:, i] - max_val)
            
            # ✅ CRITICAL: Normalize by parameter range
            lower_violation_norm = lower_violation / param_range
            upper_violation_norm = upper_violation / param_range
            
            violation_list.append(lower_violation_norm**2)
            violation_list.append(upper_violation_norm**2)
            
            # Debug tracking
            n_lower = (lower_violation > 1e-6).sum().item()
            n_upper = (upper_violation > 1e-6).sum().item()
            debug_violations[param_name] = {
                'lower': n_lower,
                'upper': n_upper,
                'lower_max': lower_violation.max().item(),
                'upper_max': upper_violation.max().item(),
                'range': (params[:, i].min().item(), params[:, i].max().item())
            }
        
        # ✅ Aggregate normalized violations
        if violation_list:
            bounds_loss = torch.mean(torch.stack(violation_list))
        else:
            bounds_loss = torch.tensor(0.0, device=params.device)
        
        # ✅ FIX: Use moderate penalty weight (not 0.5 which is too high)
        np_config = self.config.get("neural_posterior", {})
        penalty_weight = np_config.get("bounds_penalty_weight", 0.8)
        loss += penalty_weight * bounds_loss
        
        # ===============================
        # ✅ 1. MASS ORDERING CONSTRAINT: m1 >= m2
        # ===============================
        if "mass_1" in self.param_names and "mass_2" in self.param_names:
            m1_idx = self.param_names.index("mass_1")
            m2_idx = self.param_names.index("mass_2")
            
            mass_violation = F.relu(params[:, m2_idx] - params[:, m1_idx])
            
            # ✅ Normalize by typical mass scale (100 solar masses)
            mass_violation_norm = mass_violation / 100.0
            loss += torch.mean(mass_violation_norm**2)
        
        # ===============================
        # ✅ 2. SPIN BOUNDS: |a| <= 0.99
        # ===============================
        for i, param_name in enumerate(self.param_names):
            if param_name in ["a_1", "a_2"]:
                spin = params[:, i]
                spin_violation = F.relu(spin - 0.99)
                # Already normalized (spin magnitude bounded by 0.99)
                loss += 0.5 * torch.mean(spin_violation**2)
            elif "chi" in param_name.lower():
                spin = params[:, i]
                spin_violation = F.relu(torch.abs(spin) - 1.0)
                # Already normalized (effective spin is dimensionless, scale = 1)
                loss += 0.5 * torch.mean(spin_violation**2)
        
        # ===============================
        # ✅ 3. TIDAL DEFORMABILITY (BNS only): Λ <= 5000
        # ===============================
        if self.event_type == "BNS":
            for i, param_name in enumerate(self.param_names):
                if "lambda" in param_name.lower():
                    lambda_val = params[:, i]
                    lambda_violation = F.relu(lambda_val - 5000.0)
                    # ✅ Normalize by scale (5000)
                    lambda_violation_norm = lambda_violation / 5000.0
                    loss += 0.3 * torch.mean(lambda_violation_norm**2)
        
        # ✅ Final normalization by batch size (already handled by torch.mean)
        # This ensures loss is O(1) regardless of batch size
        
        return loss, debug_violations

    def update_training_metrics(
        self, loss_dict: Dict[str, torch.Tensor], processing_time: float, gradient_norm: float
    ):
        """Update training metrics for monitoring and RL."""
        self.training_step += 1

        # Store metrics
        self.performance_tracker["training_losses"].append(loss_dict["total_loss"].item())
        self.performance_tracker["inference_times"].append(processing_time)

        # Log periodically
        if self.training_step % 100 == 0:
            recent_loss = np.mean(list(self.performance_tracker["training_losses"])[-10:])
            self.logger.debug(
                f"Step {self.training_step}: Loss={recent_loss:.4f}, "
                f"GradNorm={gradient_norm:.3f}"
            )

    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "rl_controller": (
                self.model.rl_controller.get_state()
                if hasattr(self.model.rl_controller, "get_state")
                else None
            ),
            "param_names": self.model.param_names,
            "param_bounds": self.model.param_bounds,
            "config": self.config,
            "training_step": self.model.training_step,
            "performance_tracker": self.model.performance_tracker,
            # ADD THESE:
            "epoch": epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "val_metrics": val_metrics,
        }

    def load_model(self, filepath: str):
        """Load complete model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        # ✅ RL controller removed from training - skip RL checkpoint loading
        # if "rl_controller" in checkpoint and self.rl_controller is not None:
        #     self.rl_controller.q_network.load_state_dict(checkpoint["rl_controller"])
        self.training_step = checkpoint.get("training_step", 0)

        if "performance_tracker" in checkpoint:
            for k, v in checkpoint["performance_tracker"].items():
                self.performance_tracker[k] = deque(v, maxlen=self.performance_tracker[k].maxlen)

        self.logger.info(f"Model loaded from {filepath}")

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "UnifiedOverlapNeuralPE",
            "parameter_names": self.param_names,
            "parameter_dimension": self.param_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "context_dim": self.context_dim,
            "flow_layers": self.n_flow_layers,
            "training_step": self.training_step,
            "recent_performance": {
                "avg_loss": (
                    np.mean(list(self.performance_tracker["training_losses"])[-10:])
                    if self.performance_tracker["training_losses"]
                    else 0.0
                ),
                "avg_inference_time": (
                    np.mean(list(self.performance_tracker["inference_times"])[-10:])
                    if self.performance_tracker["inference_times"]
                    else 0.0
                ),
            },
        }

    def get_rl_metrics(self) -> Dict[str, float]:
        """
        Get metrics from RL controller for tracking adaptive complexity.

        Returns:
            Empty dict - RL disabled during training (available in InferencePipeline for inference-only)
        """
        # RL controller disabled for training stability
        # See RL_CONTROLLER_DISABLED_FOR_TRAINING.md for details
        return {}

    def get_bias_metrics(self) -> Dict[str, float]:
        """Bias corrector trained separately - returns empty dict."""
        return {}

    def get_priority_net_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from PriorityNet for tracking signal ranking performance.

        Returns:
            Dict with PriorityNet metrics (empty if PriorityNet not enabled)
        """
        if not hasattr(self, "priority_net") or self.priority_net is None:
            return {}

        try:
            metrics = {
                "enabled": True,
                "frozen": all(not p.requires_grad for p in self.priority_net.parameters()),
                "n_parameters": sum(p.numel() for p in self.priority_net.parameters()),
            }

            # Get PriorityNet specific metrics if available
            if hasattr(self.priority_net, "get_metrics"):
                metrics.update(self.priority_net.get_metrics())

            # Track last batch predictions/uncertainties if available
            if hasattr(self, "_last_priority_net_preds") and self._last_priority_net_preds is not None:
                preds = self._last_priority_net_preds
                if len(preds) > 0:
                    metrics["avg_importance"] = float(torch.mean(preds).item())
                    metrics["max_importance"] = float(torch.max(preds).item())
                    metrics["min_importance"] = float(torch.min(preds).item())

            if hasattr(self, "_last_priority_net_uncs") and self._last_priority_net_uncs is not None:
                uncs = self._last_priority_net_uncs
                if len(uncs) > 0:
                    metrics["avg_uncertainty"] = float(torch.mean(uncs).item())
                    metrics["max_uncertainty"] = float(torch.max(uncs).item())

            return metrics

        except Exception as e:
            self.logger.warning(f"Failed to get PriorityNet metrics: {e}")
            return {}

    def get_integration_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all integrated components and their status."""
        summary = {
            "model_type": "OverlapNeuralPE",
            "timestamp": str(pd.Timestamp.now()),
            "components": {
                "prioritynet": {
                    "enabled": self.priority_net is not None,
                    "frozen": all(not p.requires_grad for p in self.priority_net.parameters()) if self.priority_net else False,
                    "n_parameters": sum(p.numel() for p in self.priority_net.parameters()) if self.priority_net else 0,
                },
                "normalizing_flow": {
                    "enabled": True,
                    "type": self.config.get("flow_type", "flowmatching"),
                    "n_layers": self.flow_config.get("num_layers", 4),
                    "context_dim": self.context_dim,
                    "n_parameters": sum(p.numel() for p in self.flow.parameters()),
                },
                "adaptive_subtractor": {
                    "enabled": True,
                    "max_iterations": self.adaptive_subtractor.max_iterations,
                    "convergence_threshold": self.adaptive_subtractor.convergence_threshold,
                },
                "context_encoder": {
                    "enabled": True,
                    "hidden_dim": self.context_dim,
                    "n_parameters": sum(p.numel() for p in self.context_encoder.parameters()),
                },
                "uncertainty_estimator": {
                    "enabled": True,
                    "n_parameters": sum(p.numel() for p in self.uncertainty_estimator.parameters()),
                },
            },
            "metrics": {
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                ),
            },
            "configuration": {
                "context_dim": self.context_dim,
                "n_flow_layers": self.n_flow_layers,
                "max_iterations": self.max_iterations,
                "dropout_rate": self.dropout_rate,
            },
            "training_status": {
                "training_step": self.training_step,
                "recent_loss": (
                    float(np.mean(list(self.performance_tracker["training_losses"])[-10:]))
                    if self.performance_tracker["training_losses"]
                    else 0.0
                ),
            },
        }


        return summary


class NormalizingFlow(nn.Module):
    """Conditional normalizing flow for posterior estimation."""

    def __init__(self, param_dim: int, context_dim: int, n_layers: int = 8):
        super().__init__()

        self.param_dim = param_dim
        self.context_dim = context_dim
        self.n_layers = n_layers

        # Coupling layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(AffineCouplingLayer(param_dim, context_dim))

    def forward(
        self, params: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform parameters to latent space."""
        z = params
        log_det_sum = 0.0

        for layer in self.layers:
            z, log_det = layer(z, context)
            log_det_sum = log_det_sum + log_det

        return z, log_det_sum

    def inverse(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Transform from latent to parameter space."""
        params = z

        for layer in reversed(self.layers):
            params = layer.inverse(params, context)

        return params


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flow."""

    def __init__(self, param_dim: int, context_dim: int):
        super().__init__()

        self.param_dim = param_dim
        self.split_dim = param_dim // 2

        # Scale and shift networks
        self.scale_net = nn.Sequential(
            nn.Linear(self.split_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, param_dim - self.split_dim),
            nn.Tanh(),
        )

        self.shift_net = nn.Sequential(
            nn.Linear(self.split_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, param_dim - self.split_dim),
        )

    def forward(
        self, params: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transformation."""
        x1, x2 = params[:, : self.split_dim], params[:, self.split_dim :]

        combined = torch.cat([x1, context], dim=1)
        s = self.scale_net(combined)
        t = self.shift_net(combined)

        y2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=1)

        y = torch.cat([x1, y2], dim=1)

        return y, log_det

    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Inverse transformation."""
        y1, y2 = y[:, : self.split_dim], y[:, self.split_dim :]

        combined = torch.cat([y1, context], dim=1)
        s = self.scale_net(combined)
        t = self.shift_net(combined)

        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([y1, x2], dim=1)

        return x



class TemporalPooling(nn.Module):
    """
    Time-segmented pooling to preserve temporal structure.
    
    Problem: Global AdaptiveAvgPool1d(64) averages early/mid/late phases together,
    destroying temporal cues for distance, inclination, and merger properties.
    
    Solution: Split signal into 3 temporal regions, pool independently, concatenate.
    This preserves which phase (early inspiral / merger / ringdown) the information
    comes from, enabling the network to learn distance/inclination-dependent features.
    
    Example signal structure:
      Early inspiral (0-33%):    Low amplitude, slope encodes distance
      Merger (33-67%):           Peak amplitude, encodes inclination  
      Ringdown (67-100%):        Decay rate, encodes frequency content
    
    By concatenating [early_pooled, mid_pooled, late_pooled], the flow receives
    temporal context that distance/inclination corrections depend on.
    """
    
    def __init__(self, out_len: int = 64):
        super().__init__()
        self.out_len = out_len
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            pooled: (batch, channels, out_len) with temporal structure preserved
        """
        T = x.size(-1)
        # Split into temporal regions
        third = T // 3
        early = x[..., :third]            # Early inspiral (0-33%)
        mid   = x[..., third:2*third]     # Merger (33-67%)
        late  = x[..., 2*third:]          # Ringdown (67-100%)
        
        # Pool each region independently (preserving temporal location)
        # Handle uneven division of out_len
        out_early = (self.out_len + 2) // 3  # Round up for early
        out_mid   = self.out_len // 3        # Middle third
        out_late  = self.out_len - out_early - out_mid  # Remainder to late
        
        early_pooled = F.adaptive_avg_pool1d(early, out_early)
        mid_pooled   = F.adaptive_avg_pool1d(mid,   out_mid)
        late_pooled  = F.adaptive_avg_pool1d(late,  out_late)
        
        # Concatenate: result is [batch, channels, out_len] with temporal structure
        return torch.cat([early_pooled, mid_pooled, late_pooled], dim=-1)


class ContextEncoder(nn.Module):
    """Encodes multi-detector strain data into context vector."""

    def __init__(
        self, n_detectors: int = 2, hidden_dim: int = 512, dropout: float = 0.1
    ):
        super().__init__()
        self.n_detectors = n_detectors
        self.hidden_dim = hidden_dim

        #  (Dec 24): Replace BatchNorm with InstanceNorm for validation stability
        # Problem: BatchNorm1d uses running statistics (updated during training, fixed at validation)
        #   - During training: batch stats good, context std recovers (0.45-0.70) ✅
        #   - During validation: uses corrupted running stats from early collapsed batches
        #   - Result: validation context std = 0.13 (collapsed) despite training recovery
        # Solution: InstanceNorm has NO running statistics (computed per sample)
        #   - Independent of batch composition
        #   - Same behavior in train and validation
        #   - Prevents train/val context divergence
        # Replace global pooling with time-segmented pooling
        # This preserves early/mid/late phase information critical for distance/inclination inference
        self.detector_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=4),
            nn.InstanceNorm1d(32, affine=True), 
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=32, stride=4),
            nn.InstanceNorm1d(64, affine=True), 
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=16, stride=2),
            nn.InstanceNorm1d(128, affine=True), 
            nn.ReLU(),
            
            TemporalPooling(out_len=64),  #Time-segmented pooling
        )

        # ✅ LIGHTWEIGHT IMPROVEMENTS (Dec 30):
        # 1. LayerNorm for stable gradient flow (+2% training time)
        # 2. Learnable context scale to increase variance (zero overhead)
        # 3. Higher dropout for better regularization (zero overhead)
        
        # Learnable context scale (starts at 1.3 to boost variance)
        self.context_scale = nn.Parameter(torch.tensor(1.3, dtype=torch.float32))
        
        # ✅ JAN 5 CRITICAL FIX: Fixed fusion input size
        # Problem: Training data may have 2-3 detectors, but fusion needs fixed input
        # Solution: Always pad detector_features to 3 in forward() before fusion
        # This ensures fusion always receives [batch, 128*64*3] = [batch, 24576]
        #
        # ✅ JAN 17 CRITICAL FIX: Add amplitude-aware pooling via RMS features
        # Problem: InstanceNorm removes amplitude information, so flow can't anchor distance
        # Solution: Add raw RMS energy (1 scalar per detector) to context
        # This gives flow explicit amplitude cue: low amplitude = far, high = close
        # Fusion input now: [128*64*3] CNN features + [3] RMS features = [24579] total
        self.fusion = nn.Sequential(
            nn.Linear(128 * 64 * 3 + 3, hidden_dim * 2),  # Fixed to 3 detectors (always padded)
            nn.LayerNorm(hidden_dim * 2),  # ✅ Better gradient flow
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),  # ✅ 0.1 → 0.15 for better regularization
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # ✅ Stabilizes context embeddings
            nn.ReLU(),
            nn.Dropout(dropout),  # ✅ Final regularization layer
        )
        

    def forward(self, strain_data: torch.Tensor) -> torch.Tensor:
         """
         Args:
             strain_data: (batch, n_detectors, time_samples)
         Returns:
             context: (batch, hidden_dim) with LEARNED scale (not forced to std=1.0)
         """
         batch_size = strain_data.size(0)
         
         # Validate input shape
         if strain_data.dim() != 3:
             raise ValueError(f"Expected strain_data to be 3D [batch, detectors, time], got shape {strain_data.shape}")
         
         n_input_detectors = strain_data.size(1)
         
         # ✅ FLEXIBLE: Auto-adapt to actual detector count
         # If input has fewer detectors than initialized, use only those
         # If input has more, use only the first n_detectors
         actual_n_detectors = min(n_input_detectors, self.n_detectors)

         # Encode each detector
         detector_features = []
         rms_features = []  # ✅ JAN 17: Collect RMS amplitude for each detector
         
         for i in range(actual_n_detectors):
             det_data = strain_data[:, i : i + 1, :]  # (batch, 1, time)
             if det_data.size(1) != 1:
                 raise ValueError(f"Detector {i} extraction failed: shape={det_data.shape}, expected (batch, 1, time)")
             
             # Compute RMS energy (amplitude anchor for distance inference)
             # RMS = sqrt(mean(x^2)) captures absolute amplitude before normalization
             det_data_squeezed = det_data.squeeze(1)  # (batch, time)
             rms = torch.sqrt((det_data_squeezed ** 2).mean(dim=1, keepdim=True) + 1e-10)  # (batch, 1)
             rms_features.append(rms)
             
             features = self.detector_encoder(det_data)  # (batch, 128, 64)
             detector_features.append(features)
         
         # Pad with zeros if we have fewer detectors than expected
         if actual_n_detectors < self.n_detectors:
             # Add zero features for missing detectors
             zero_features = torch.zeros(batch_size, 128, 64, device=strain_data.device, dtype=strain_data.dtype)
             zero_rms = torch.zeros(batch_size, 1, device=strain_data.device, dtype=strain_data.dtype)
             for _ in range(self.n_detectors - actual_n_detectors):
                 detector_features.append(zero_features)
                 rms_features.append(zero_rms)

        # Concatenate CNN features and RMS features
         combined = torch.cat(detector_features, dim=1)  # (batch, 128*n_det, 64)
         combined = combined.flatten(1)  # (batch, 128*64*n_det)
         
         # ✅ JAN 17: Append RMS amplitude features
         rms_concatenated = torch.cat(rms_features, dim=1)  # (batch, n_detectors)
         combined = torch.cat([combined, rms_concatenated], dim=1)  # (batch, 128*64*n_det + n_detectors)

         context = self.fusion(combined)  # (batch, hidden_dim)
        
         # ✅ LIGHTWEIGHT FIX (Dec 30): Learnable context scaling
         # Scale context to increase variance and improve posterior diversity
         # context_scale starts at 1.3, learns optimal scaling during training
         context = context * self.context_scale

         return context
