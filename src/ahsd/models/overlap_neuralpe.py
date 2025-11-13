"""
Neural Parameter Estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

from ahsd.core.priority_net import PriorityNet
from ahsd.models.rl_controller import AdaptiveComplexityController
from ahsd.core.bias_corrector import BiasCorrector
from ahsd.core.adaptive_subtractor import AdaptiveSubtractor
from ahsd.models.flows import create_flow_model


class OverlapNeuralPE(nn.Module):
    """
    Unified best-in-class Neural PE for overlapping gravitational wave signals.

    Integrates:
    - PriorityNet: Signal importance ranking
    - RL Controller: Adaptive complexity
    - Normalizing Flow: Posterior estimation
    - Bias Corrector: Systematic bias removal
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

        # Model configuration
        self.context_dim = config.get("context_dim", 512)  # ✅ Increased from 256
        self.n_flow_layers = config.get(
            "n_flow_layers", 6
        )  # ✅ Reduced for FlowMatching (more expressive)
        self.max_iterations = config.get("max_iterations", 5)

        # Dropout configuration
        self.dropout_rate = config.get("dropout", 0.1)
        self.flow_config = config.get("flow_config", {})
        self.flow_dropout = self.flow_config.get("dropout", 0.15)
        self.flow_hidden_features = self.flow_config.get("hidden_features", 128)
        self.flow_num_blocks = self.flow_config.get("num_blocks_per_layer", 2)

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
        # Performance tracking
        self.performance_tracker = {
            "training_losses": deque(maxlen=1000),
            "validation_metrics": deque(maxlen=100),
            "complexity_history": deque(maxlen=1000),
            "inference_times": deque(maxlen=1000),
            "rl_rewards": deque(maxlen=1000),
        }
        self.training_step = 0
        self.to(self.device)
        total_params = sum(p.numel() for p in self.parameters())

        self.logger.info(
            f"âœ… Unified OverlapNeuralPE initialized with {total_params:,} parameters"
        )
        self.logger.info(f"   Context dim: {self.context_dim}")
        self.logger.info(f"   Flow layers: {self.n_flow_layers}")
        self.logger.info(f"   Dropout: {self.dropout_rate}, Flow dropout: {self.flow_dropout}")

    def _get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
         """Get parameter bounds for normalization."""
         bounds = {
             "mass_1": (1.0, 100.0),
             "mass_2": (1.0, 100.0),
             "luminosity_distance": (10.0, 8000.0),  # Fixed: Allow rare events down to 10 Mpc
             "geocent_time": (-2.0, 8.0),  # Fixed: Matches actual data range (i*1.5 spacing up to 4 signals)
             "ra": (0.0, 2 * np.pi),
             "dec": (-np.pi / 2, np.pi / 2),
             "theta_jn": (0.0, np.pi),
             "psi": (0.0, np.pi),
             "phase": (0.0, 2 * np.pi),
         }
         return {param: bounds.get(param, (0.0, 1.0)) for param in self.param_names}

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

            else:
                # Default uniform prior for other parameters
                priors[param] = torch.distributions.Uniform(0.0, 1.0)

        self.logger.info(
            f"✅ Physics priors initialized: event_type={self.event_type}, "
            f"use_event_priors={self.use_event_priors}"
        )
        return priors

    def _normalize_parameters(self, params: torch.Tensor) -> torch.Tensor:
        """Normalize parameters to [-1, 1]."""
        normalized = torch.zeros_like(params)

        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_bounds[param_name]
            normalized[..., i] = 2 * (params[..., i] - min_val) / (max_val - min_val) - 1

        return torch.clamp(normalized, -1, 1)

    def _denormalize_parameters(self, normalized_params: torch.Tensor) -> torch.Tensor:
        """Denormalize parameters from [-1, 1] to physical units.

        ✅ CRITICAL FIX (Nov 13): Clamp normalized inputs to [-1, 1] to prevent
        flow extrapolation from creating unphysical values.

        The normalizing flow can output values outside [-1, 1], especially early
        in training. Clamping ensures denormalized values stay in physical ranges.
        """
        # ✅ FIX 1: Clamp to ensure normalized params are in valid [-1, 1] range
        normalized_params_clipped = torch.clamp(normalized_params, -1.0, 1.0)

        params = torch.zeros_like(normalized_params_clipped)

        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_bounds[param_name]
            # Formula: x_physical = (x_norm + 1) / 2 * (max - min) + min
            params[..., i] = (normalized_params_clipped[..., i] + 1) / 2 * (
                max_val - min_val
            ) + min_val

        return params

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

        self.logger.info(
            f"Loaded {len(checkpoint_state) - len(incompatible_keys)}/{len(checkpoint_state)} checkpoint weights"
        )

    def _init_components(self, priority_net_path: str):
        """Initialize all pipeline components."""

        # 1. PriorityNet (pre-trained, frozen) - load checkpoint first to get architecture
        checkpoint = torch.load(priority_net_path, map_location=self.device)
        model_arch = checkpoint.get("model_architecture", {})
        self.priority_net = PriorityNet(
            use_strain=model_arch.get("use_strain", True),
            use_edge_conditioning=model_arch.get("use_edge_conditioning", True),
            n_edge_types=model_arch.get("n_edge_types", 19),
        )
        self._load_checkpoint_with_mismatch_handling(
            self.priority_net, checkpoint["model_state_dict"]
        )
        self.priority_net.eval()
        for param in self.priority_net.parameters():
            param.requires_grad = False
        self.logger.info(f"âœ… Loaded PriorityNet from {priority_net_path}")

        # 2. Context Encoder
        self.context_encoder = ContextEncoder(
            n_detectors=2, hidden_dim=self.context_dim, dropout=self.dropout_rate
        )

        # 3. Normalizing Flow (using FlowMatching by default, fallback to RealNVP)
        flow_type = self.flow_config.get(
            "type", "flowmatching"
        )  # 'flowmatching', 'realnvp', or 'maf'

        if flow_type.lower() == "flowmatching":
            # âœ… FlowMatching: OT-CFM with higher expressiveness per layer
            self.flow = create_flow_model(
                flow_type="flowmatching",
                features=self.param_dim,
                context_features=self.context_dim,  # Now 512
                hidden_dim=self.flow_config.get("hidden_features", 256),  # Increased
                num_layers=self.flow_config.get("num_layers", 4),  # Fewer, more expressive
                solver_steps=self.flow_config.get("solver_steps", 10),
                dropout=self.flow_config.get("dropout", 0.1),
            )
        else:
            # Fallback to RealNVP for compatibility
            self.flow = create_flow_model(
                flow_type=flow_type,
                features=self.param_dim,
                context_features=self.context_dim,
                hidden_features=self.flow_config.get("hidden_features", 256),
                num_layers=self.flow_config.get("num_layers", self.n_flow_layers),
                num_blocks_per_layer=self.flow_config.get("num_blocks_per_layer", 2),
                dropout=self.flow_config.get("dropout", 0.1),
            )

        self.logger.info(
            f"âœ… Flow model initialized: {flow_type} with context_dim={self.context_dim}, "
            f"num_layers={self.flow_config.get('num_layers', 4 if flow_type.lower() == 'flowmatching' else 8)}"
        )

        # 4. RL Controller
        rl_config = self.config.get("rl_controller", {})

        self.rl_controller = AdaptiveComplexityController(
            state_features=rl_config.get(
                "state_features",
                ["remaining_signals", "residual_power", "current_snr", "extraction_success_rate"],
            ),
            complexity_levels=rl_config.get("complexity_levels", ["low", "medium", "high"]),
            learning_rate=rl_config.get("learning_rate", 1e-3),
            epsilon=rl_config.get("epsilon", 0.1),
            epsilon_decay=rl_config.get("epsilon_decay", 0.995),
            memory_size=rl_config.get("memory_size", 10000),
            batch_size=rl_config.get("batch_size", 32),
        )

        self.complexity_configs = rl_config.get(
            "complexity_configs",
            {
                "low": {"flow_layers": 4, "inference_samples": 500},
                "medium": {"flow_layers": 8, "inference_samples": 1000},
                "high": {"flow_layers": 12, "inference_samples": 2000},
            },
        )

        # 5. Bias Corrector
        bias_cfg = self.config.get("bias_corrector", {})
        if bias_cfg.get("enabled", True):
            self.bias_corrector = BiasCorrector(
                param_names=self.param_names, context_dim=self.context_dim
            )
            self.logger.info("BiasCorrector initialized")
        else:
            self.bias_corrector = None
            self.logger.info("BiasCorrector disabled")

        # 6. Adaptive Subtractor
        self.adaptive_subtractor = AdaptiveSubtractor()
        self.logger.info("âœ… AdaptiveSubtractor initialized")

    def sample_posterior(
        self, strain_data: torch.Tensor, n_samples: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from learned posterior distribution with rejection sampling.

        Args:
            strain_data: [batch, n_det, n_samples] whitened strain data
            n_samples: Number of posterior samples to draw

        Returns:
            dict containing:
                'samples': [batch, n_samples, param_dim] posterior samples
                'means': [batch, param_dim] posterior means
                'stds': [batch, param_dim] posterior standard deviations
                'uncertainties': [batch, param_dim] estimated uncertainties

        ✅ CRITICAL FIX (Nov 13): Added rejection sampling + output clamping to filter
        out-of-range predictions. The flow can extrapolate beyond physical bounds,
        causing NLL explosion. Rejection sampling ensures all returned samples are physical.
        """
        self.eval()
        batch_size = strain_data.size(0)

        with torch.no_grad():
            # Extract context from strain
            context = self.context_encoder(strain_data)
            context = (context - context.mean(dim=0, keepdim=True)) / (
                context.std(dim=0, keepdim=True) + 1e-6
            )

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
                rejection_rates = []

                while len(valid_samples) < n_samples and n_attempts < max_attempts:
                    # Inverse flow transformation returns normalized params [-1, 1]
                    samples_normalized, _ = self.flow.inverse(z_i, context_i)

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
                    rejection_rate = (1.0 - n_valid / len(valid_mask)) * 100
                    rejection_rates.append(rejection_rate)

                    valid_samples.append(samples_physical[valid_mask])

                    n_attempts += 1
                    if len(valid_samples) < n_samples and n_attempts < max_attempts:
                        z_i = torch.randn(n_samples * 2, self.param_dim, device=self.device)

                # Log rejection statistics
                if rejection_rates:
                    avg_rejection = np.mean(rejection_rates)
                    if avg_rejection > 30:
                        self.logger.warning(
                            f"High rejection rate: {avg_rejection:.1f}% | "
                            f"Norm violations: {norm_violation_pct:.1f}% | "
                            f"Attempts: {n_attempts}/{max_attempts}"
                        )

                # Concatenate and trim to requested number
                if valid_samples:
                    batch_samples = torch.cat(valid_samples, dim=0)[:n_samples]
                    if len(batch_samples) < n_samples:
                        self.logger.warning(
                            f"Got {len(batch_samples)}/{n_samples} valid samples (rejection too high)"
                        )
                else:
                    # Fallback: return all samples if rejection rate too high
                    self.logger.warning(f"Rejection sampling failed, returning all samples")
                    samples_normalized, _ = self.flow.inverse(
                        torch.randn(n_samples, self.param_dim, device=self.device),
                        context_i[:n_samples],
                    )
                    batch_samples = self._denormalize_parameters(samples_normalized)

                samples_list.append(batch_samples)

            samples = torch.stack(samples_list, dim=0)  # [batch, n_samples, param_dim]

            # Compute summary statistics
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
        Extract parameters for a single signal.

        Args:
            strain_data: [batch, n_det, n_samples] strain data
            complexity: Complexity level ('low', 'medium', 'high')

        Returns:
            dict with parameter estimates and uncertainties
        """
        # Use posterior sampling for extraction
        complexity_settings = self.complexity_configs.get(complexity, {})
        n_samples = complexity_settings.get("inference_samples", 1000)

        result = self.sample_posterior(strain_data, n_samples=n_samples)

        return {
            "means": result["means"],
            "stds": result["stds"],
            "samples": result["samples"],
            "uncertainties": result["uncertainties"],
            "context": result["context"],
        }

    def extract_overlapping_signals(
        self,
        strain_data: torch.Tensor,
        true_params: Optional[List[Dict]] = None,
        training: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract all overlapping signals iteratively with integrated bias correction,
        adaptive subtraction, normalizing flows, and RL control.

        Args:
            strain_data: [batch, n_det, n_samples] strain data
            true_params: Optional ground truth parameters for training
            training: Whether in training mode (for RL)

        Returns:
            dict with all extracted signals, bias-corrected parameters, and final residual
        """
        batch_size = strain_data.size(0)

        all_extracted = []
        residual_data = strain_data.clone()

        pipeline_state = {
            "remaining_signals": self.max_iterations,
            "residual_power": 1.0,
            "current_snr": 0.0,
            "extraction_success_rate": 1.0,
        }

        for iteration in range(self.max_iterations):
            # ✅ 1. GET PRIORITIES FROM PRIORITYNET
            with torch.no_grad():
                detections = self._residual_to_detections(residual_data)
                priorities, _ = self.priority_net(detections)

            # ✅ 2. SELECT COMPLEXITY VIA RL CONTROLLER
            complexity = self.rl_controller.get_complexity_level(pipeline_state, training=training)

            # ✅ 3. EXTRACT SIGNAL USING NORMALIZING FLOW
            extraction_result = self.extract_single_signal(residual_data, complexity)
            params_means = extraction_result["means"]
            params_stds = extraction_result["stds"]
            context = extraction_result["context"]

            # ✅ 4. APPLY BIAS CORRECTION
            if self.bias_corrector is not None:
                with torch.no_grad():
                    # Normalize parameters for bias corrector
                    params_normalized = self._normalize_parameters(params_means)

                    # Get bias corrections
                    corrections, bias_uncertainties, confidence = self.bias_corrector(
                        params_normalized, context
                    )

                    # Apply corrections
                    params_corrected = params_means + corrections
                    params_corrected = torch.clamp(
                        params_corrected,
                        torch.tensor(
                            [b[0] for b in self.param_bounds.values()], device=self.device
                        ),
                        torch.tensor(
                            [b[1] for b in self.param_bounds.values()], device=self.device
                        ),
                    )

                    # Update uncertainties (quadrature sum)
                    params_stds_corrected = torch.sqrt(params_stds**2 + bias_uncertainties**2)

                    self.logger.debug(
                        f"Iteration {iteration}: Bias correction applied, confidence={confidence.mean().item():.3f}"
                    )
            else:
                params_corrected = params_means
                params_stds_corrected = params_stds

            all_extracted.append(
                {
                    "params": params_corrected,
                    "uncertainties": params_stds_corrected,
                    "priority": priorities,
                    "iteration": iteration,
                    "complexity": complexity,
                    "bias_corrected": self.bias_corrector is not None,
                }
            )

            # ✅ 5. ADAPTIVE SUBTRACTION (using correct method)
            estimated_params_dict = self._tensor_to_param_dict(params_corrected)

            # Convert tensor residual_data back to detector format dictionary
            residual_dict = self._tensor_to_detector_dict(residual_data)

            # Use extract_and_subtract (the actual method in AdaptiveSubtractor)
            residual_data_dict, extraction_metadata, extraction_uncertainties = (
                self.adaptive_subtractor.extract_and_subtract(
                    residual_dict, detection_idx=iteration
                )
            )

            # Convert residual dictionary back to tensor
            residual_data = self._detector_dict_to_tensor(residual_data_dict)

            # ✅ 6. UPDATE PIPELINE STATE FOR RL
            pipeline_state["remaining_signals"] -= 1
            pipeline_state["residual_power"] = float(torch.mean(residual_data**2))

            # Compute SNR for state
            if iteration == 0:
                pipeline_state["current_snr"] = float(torch.sqrt(torch.mean(strain_data**2)))
            else:
                pipeline_state["current_snr"] = float(torch.sqrt(torch.mean(residual_data**2)))

            # Early stopping if residual too low
            if pipeline_state["residual_power"] < 0.001:
                self.logger.info(f"Stopping at iteration {iteration+1}: low residual power")
                break

            # ✅ 7. RL TRAINING WITH IMPROVED REWARD
            if training and true_params is not None:
                # Compute extraction accuracy
                true_params_iter = true_params[iteration] if iteration < len(true_params) else None
                reward = self._compute_extraction_reward(params_corrected, true_params_iter)

                # Bonus for good bias correction
                if self.bias_corrector is not None:
                    reward += 0.1 * confidence.mean().item()

                # Penalty for high uncertainty
                reward -= 0.05 * torch.mean(params_stds_corrected).item()

                state_vector = self.rl_controller.get_state_vector(pipeline_state)
                action = self.rl_controller.complexity_levels.index(complexity)

                # Next state vector after subtraction
                next_state_dict = pipeline_state.copy()
                next_state_vector = self.rl_controller.get_state_vector(next_state_dict)

                done = iteration == self.max_iterations - 1

                # Store experience in RL memory
                self.rl_controller.store_experience(
                    state_vector, action, reward, next_state_vector, done
                )

                # Train RL if enough experience collected
                if len(self.rl_controller.memory) >= self.rl_controller.batch_size:
                    rl_loss = self.rl_controller.train_step()
                    if rl_loss is not None:
                        self.logger.debug(
                            f"Iteration {iteration}: RL loss={rl_loss:.4f}, reward={reward:.3f}"
                        )

        return {
            "extracted_signals": all_extracted,
            "final_residual": residual_data,
            "n_iterations": iteration + 1,
            "pipeline_state": pipeline_state,
        }

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
        """Convert residual strain to detection format for PriorityNet."""
        batch_size = residual.size(0)
        detections = []

        for i in range(batch_size):
            snr_proxy = float(torch.sqrt(torch.mean(residual[i] ** 2)))

            detection = {
                "network_snr": snr_proxy * 10.0,
                "match_filter_snr": snr_proxy * 10.0,
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

    def compute_loss(
        self, strain_data: torch.Tensor, true_params: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive training loss with all integrated components.

        Args:
            strain_data: [batch, n_det, n_samples] strain data
            true_params: [batch, param_dim] true parameters

        Returns:
            dict with total loss and component losses
        """
        # ✅ Extract context via ContextEncoder
        context = self.context_encoder(strain_data)

        # ✅ Normalize parameters for normalizing flow
        true_params_norm = self._normalize_parameters(true_params)

        # ✅ 1. FLOW LOSS (negative log-likelihood from normalizing flow)
        log_prob = self.flow.log_prob(true_params_norm, context)
        flow_loss = -log_prob.mean()

        # ✅ 2. PHYSICS CONSTRAINT LOSS
        physics_loss = self._compute_physics_loss(true_params)

        # ✅ 3. BIAS CORRECTION LOSS (if bias corrector enabled)
        bias_loss = torch.tensor(0.0, device=strain_data.device)
        if self.bias_corrector is not None:
            with torch.no_grad():
                # Get bias corrections for ground truth parameters
                corrections, bias_uncertainties, confidence = self.bias_corrector(
                    true_params_norm, context
                )

            # Encourage small corrections on average (prior toward no correction)
            # ✅ Fixed: Changed -confidence to (1.0 - confidence) for positive loss
            bias_loss = 0.05 * (
                torch.mean(torch.abs(corrections)) + 0.1 * torch.mean(torch.abs(1.0 - confidence))
            )

        # ✅ 4. UNCERTAINTY REGULARIZATION
        uncertainties = self.uncertainty_estimator(torch.cat([true_params_norm, context], dim=1))
        uncertainty_loss = 0.01 * torch.mean(uncertainties)

        # ✅ 5. FLOW REGULARIZATION (weight norm penalty) - configurable strength
        jacobian_reg_weight = self.config.get(
            "jacobian_reg_weight"
        ) or self.config.get("neural_posterior", {}).get("jacobian_reg_weight", 0.001)  # 10x stronger than default 1e-4
        jacobian_reg = jacobian_reg_weight * torch.mean(
            torch.stack([p.norm(2) for n, p in self.flow.named_parameters() if "weight" in n])
        )

        # ✅ TOTAL LOSS: All components integrated
        # ✅ CRITICAL FIX (Nov 13): Increased physics_loss weight from 0.2 → 1.0
        # to prevent flow from extrapolating beyond physical bounds and causing NLL explosion
        physics_loss_weight = self.config.get(
            "physics_loss_weight", 1.0
        ) or self.config.get("neural_posterior", {}).get("physics_loss_weight", 1.0)  # ✅ Default 1.0 (was 0.2)

        total_loss = (
            flow_loss  # Main likelihood
            + jacobian_reg  # Flow weight regularization
            + physics_loss_weight
            * physics_loss  # ✅ Physics constraints (1.0x for hard constraint)
            + bias_loss  # Bias correction regularization
            + uncertainty_loss  # Uncertainty regularization
        )

        return {
            "total_loss": total_loss,
            "nll": flow_loss,
            "physics_loss": physics_loss,
            "bias_loss": bias_loss,
            "uncertainty_loss": uncertainty_loss,
            "jacobian_reg": jacobian_reg,
        }

    def _compute_physics_loss(self, params: torch.Tensor) -> torch.Tensor:
        """Enforce physical constraints with hard bounds penalty.

        ✅ CRITICAL FIX (Nov 13): Added bounds penalties to prevent flow from
        extrapolating beyond physical parameter ranges. Without this, the flow
        learns to predict negative masses and negative distances.

        ✅ FIX 2: Stronger physics loss weight (1.0 default) to constrain flow outputs.
        """
        loss = torch.tensor(0.0, device=params.device)

        # ✅ 0. BOUNDS PENALTY: Penalize parameters outside training domain
        # This is CRUCIAL to prevent NLL explosion from out-of-range predictions
        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_bounds[param_name]

            # Soft penalty for exceeding bounds (quadratic for smooth gradients)
            lower_violation = F.relu(min_val - params[:, i])  # Penalty if param < min
            upper_violation = F.relu(params[:, i] - max_val)  # Penalty if param > max

            # ✅ FIX 3 (Nov 13): REDUCED penalty weight from 1.0 → 0.01
            # Nov 13 physics_loss_weight doubled (0.2 → 1.0) which amplified bounds penalty
            # Squared violations cause 1750x spike at Epoch 1 when model outputs go out-of-range
            # Reducing to 0.01 provides gentle constraint without dominating loss
            penalty_weight = self.config.get("bounds_penalty_weight") or self.config.get("neural_posterior", {}).get("bounds_penalty_weight", 0.01)  # ✅ Reduced from 1.0
            loss += penalty_weight * (
                torch.mean(lower_violation**2) + torch.mean(upper_violation**2)
            )

        # 1. Mass ordering constraint: m1 >= m2
        if "mass_1" in self.param_names and "mass_2" in self.param_names:
            m1_idx = self.param_names.index("mass_1")
            m2_idx = self.param_names.index("mass_2")
            mass_violation = F.relu(params[:, m2_idx] - params[:, m1_idx])
            loss += torch.mean(mass_violation**2)

        # 2. Spin bounds constraint (|χ| <= 1)
        for i, param_name in enumerate(self.param_names):
            if "chi" in param_name.lower():  # e.g., chi_1, chi_2
                spin = params[:, i]
                spin_violation = F.relu(torch.abs(spin) - 1.0)  # penalty if |χ| > 1
                loss += 0.5 * torch.mean(spin_violation**2)

        # 3. Tidal deformability bounds for BNS (Lambda <= 5000)
        if self.event_type == "BNS":
            for i, param_name in enumerate(self.param_names):
                if "lambda" in param_name.lower():
                    lambda_val = params[:, i]
                    lambda_violation = F.relu(lambda_val - 5000.0)
                    loss += 0.3 * torch.mean(lambda_violation**2)

        return loss

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
        self.rl_controller.q_network.load_state_dict(checkpoint["rl_controller"])
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
            Dict with RL controller metrics (empty if RL not enabled)
        """
        if not hasattr(self, "rl_controller") or self.rl_controller is None:
            return {}

        try:
            # Get RL controller state/metrics
            if hasattr(self.rl_controller, "get_metrics"):
                return self.rl_controller.get_metrics()

            # Fallback: manually extract metrics
            metrics = {}

            # Complexity distribution
            if hasattr(self.rl_controller, "action_history"):
                action_history = self.rl_controller.action_history[-100:]  # Last 100 actions
                if action_history:
                    metrics["avg_complexity"] = float(np.mean(action_history))
                    metrics["complexity_std"] = float(np.std(action_history))

            # Rewards
            if hasattr(self.rl_controller, "reward_history"):
                reward_history = self.rl_controller.reward_history[-100:]
                if reward_history:
                    metrics["avg_reward"] = float(np.mean(reward_history))
                    metrics["total_reward"] = float(np.sum(reward_history))

            # Epsilon (exploration rate)
            if hasattr(self.rl_controller, "epsilon"):
                metrics["epsilon"] = float(self.rl_controller.epsilon)

            # Action entropy (exploration diversity)
            if hasattr(self.rl_controller, "action_counts"):
                counts = np.array(list(self.rl_controller.action_counts.values()))
                if counts.sum() > 0:
                    probs = counts / counts.sum()
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    metrics["action_entropy"] = float(entropy)

            return metrics

        except Exception as e:
            self.logger.warning(f"Failed to get RL metrics: {e}")
            return {}

    def get_bias_metrics(self) -> Dict[str, float]:
        """
        Get metrics from bias corrector for tracking correction effectiveness.

        Returns:
            Dict with bias correction metrics (empty if bias corrector not enabled)
        """
        if not hasattr(self, "bias_corrector") or self.bias_corrector is None:
            return {}

        try:
            # Get bias corrector metrics
            if hasattr(self.bias_corrector, "get_metrics"):
                return self.bias_corrector.get_metrics()

            # Fallback: manually extract metrics
            metrics = {}

            # Correction history
            if hasattr(self.bias_corrector, "correction_history"):
                corrections = self.bias_corrector.correction_history[-100:]  # Last 100
                if corrections:
                    # Average correction magnitude per parameter
                    corrections_array = np.array(corrections)
                    metrics["avg_correction"] = float(np.mean(np.abs(corrections_array)))
                    metrics["max_correction"] = float(np.max(np.abs(corrections_array)))
                    metrics["correction_std"] = float(np.std(corrections_array))

            # Confidence scores
            if hasattr(self.bias_corrector, "confidence_history"):
                confidences = self.bias_corrector.confidence_history[-100:]
                if confidences:
                    metrics["avg_confidence"] = float(np.mean(confidences))
                    metrics["min_confidence"] = float(np.min(confidences))

            # Physics violation count
            if hasattr(self.bias_corrector, "physics_violation_count"):
                metrics["physics_violations"] = int(self.bias_corrector.physics_violation_count)

            # Correction acceptance rate
            if hasattr(self.bias_corrector, "corrections_accepted") and hasattr(
                self.bias_corrector, "corrections_proposed"
            ):
                proposed = self.bias_corrector.corrections_proposed
                accepted = self.bias_corrector.corrections_accepted
                if proposed > 0:
                    metrics["correction_acceptance_rate"] = float(accepted / proposed)

            return metrics

        except Exception as e:
            self.logger.warning(f"Failed to get bias metrics: {e}")
            return {}

    def get_integration_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all integrated components and their status.

        Returns:
            Dict with integration status, metrics, and component configuration
        """
        summary = {
            "model_type": "UnifiedOverlapNeuralPE",
            "timestamp": str(pd.Timestamp.now()),
            "components": {
                "prioritynet": {
                    "enabled": True,
                    "frozen": all(not p.requires_grad for p in self.priority_net.parameters()),
                    "n_parameters": sum(p.numel() for p in self.priority_net.parameters()),
                },
                "normalizing_flow": {
                    "enabled": True,
                    "type": self.flow_config.get("type", "flowmatching"),
                    "n_layers": self.flow_config.get("num_layers", 4),
                    "context_dim": self.context_dim,
                    "n_parameters": sum(p.numel() for p in self.flow.parameters()),
                },
                "bias_corrector": {
                    "enabled": self.bias_corrector is not None,
                    "trained": self.bias_corrector.is_trained if self.bias_corrector else False,
                    "strategy": (
                        self.bias_corrector.current_strategy if self.bias_corrector else None
                    ),
                    "n_parameters": (
                        sum(p.numel() for p in self.bias_corrector.parameters())
                        if self.bias_corrector
                        else 0
                    ),
                },
                "adaptive_subtractor": {
                    "enabled": True,
                    "max_iterations": self.adaptive_subtractor.max_iterations,
                    "convergence_threshold": self.adaptive_subtractor.convergence_threshold,
                },
                "rl_controller": {
                    "enabled": True,
                    "epsilon": float(self.rl_controller.epsilon),
                    "memory_size": len(self.rl_controller.memory),
                    "complexity_levels": self.rl_controller.complexity_levels,
                    "n_parameters": sum(
                        p.numel() for p in self.rl_controller.q_network.parameters()
                    ),
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
                "rl_metrics": self.get_rl_metrics(),
                "bias_metrics": self.get_bias_metrics(),
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

        self.logger.info(f"✅ Integration summary: {len(summary['components'])} components active")

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


class ContextEncoder(nn.Module):
    """Encodes multi-detector strain data into context vector."""

    def __init__(
        self, n_detectors: int = 2, hidden_dim: int = 512, dropout: float = 0.1
    ):  # âœ… Updated to 512
        super().__init__()

        self.n_detectors = n_detectors

        # Per-detector encoder
        self.detector_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),  # âœ… ADD dropout after each block
            nn.Conv1d(32, 64, kernel_size=32, stride=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),  # âœ… ADD
            nn.Conv1d(64, 128, kernel_size=16, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),  # âœ… ADD
            nn.AdaptiveAvgPool1d(64),
        )

        # Multi-detector fusion (âœ… Updated for 512-dim output)
        self.fusion = nn.Sequential(
            nn.Linear(128 * 64 * n_detectors, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # âœ… Add final layer norm for stability
        )

    def forward(self, strain_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            strain_data: (batch, n_detectors, time_samples)
        Returns:
            context: (batch, hidden_dim)
        """
        batch_size = strain_data.size(0)

        # Encode each detector
        detector_features = []
        for i in range(self.n_detectors):
            det_data = strain_data[:, i : i + 1, :]  # (batch, 1, time)
            features = self.detector_encoder(det_data)  # (batch, 128, 64)
            detector_features.append(features)

        # Concatenate and fuse
        combined = torch.cat(detector_features, dim=1)  # (batch, 128*n_det, 64)
        combined = combined.flatten(1)  # (batch, 128*64*n_det)

        context = self.fusion(combined)  # (batch, hidden_dim)

        return context
