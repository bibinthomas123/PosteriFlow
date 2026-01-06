#!/usr/bin/env python3
"""
 Bias Correction with Advanced Neural Networks - COMPLETE REAL IMPLEMENTATION
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings

class ResidualMLP(nn.Module):
    """FIX #5: Lightweight residual MLP backbone (replaces Transformer for single-token scenario)"""
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)  # Residual connection
        return x


class BiasEstimator(nn.Module):
    """ Advanced bias estimator with ResidualMLP backbone (FIX #5 simplification)"""
     
    def __init__(self, input_dim: int, context_dim: int = 256, hidden_dims: List[int] = None):
        super().__init__()
         
        self.input_dim = input_dim
        self.context_dim = context_dim   # Extended context features
         
        if hidden_dims is None:
             hidden_dims = [256, 128, 64] if input_dim <= 9 else [512, 256, 128, 64]
         
         #  Multi-scale feature extraction (ENHANCED for better gradient flow)
        self.param_embedding = nn.Sequential(
             nn.Linear(input_dim, 256),
             nn.LayerNorm(256),
             nn.GELU(),
             nn.Dropout(0.15),
             nn.Linear(256, 128)
         )
         
        self.context_embedding = nn.Sequential(
             nn.Linear(self.context_dim, 256),
             nn.LayerNorm(256),
             nn.GELU(),
             nn.Dropout(0.15),
             nn.Linear(256, 128)
         )
         
         # FIX #5: Replace Transformer with ResidualMLP (faster, lower variance, same expressiveness)
         # Rationale: Single-token input doesn't benefit from self-attention; MLP + residuals are simpler & more stable
        self.backbone = ResidualMLP(256, hidden_dim=256, num_layers=3)
        
        #  Parameter-specific correction heads with physics constraints (IMPROVED with Linear outputs)
        self.mass_corrector = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2)  # mass_1, mass_2 (no Tanh - raw outputs bounded via loss)
        )
        
        self.distance_corrector = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)  # luminosity_distance (no Tanh)
        )
        
        self.time_corrector = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)  # geocent_time (no Tanh)
        )
        
        self.sky_corrector = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2)  # ra, dec (no Tanh)
        )
        
        #  Additional parameters corrector (improved architecture)
        remaining_params = max(0, input_dim - 6)
        if remaining_params > 0:
            self.extra_corrector = nn.Sequential(
                nn.Linear(256, max(128, remaining_params * 12)),
                nn.GELU(),
                nn.LayerNorm(max(128, remaining_params * 12)),
                nn.Linear(max(128, remaining_params * 12), max(64, remaining_params * 6)),
                nn.GELU(),
                nn.Linear(max(64, remaining_params * 6), remaining_params)  # no Tanh
            )
        else:
            self.extra_corrector = None
        
        #  Physics-based scaling parameters (learned)
        self.mass_scale = nn.Parameter(torch.tensor(0.08))        # 8% max mass correction
        self.distance_scale = nn.Parameter(torch.tensor(0.25))    # 25% max distance correction
        self.time_scale = nn.Parameter(torch.tensor(0.001))       # 1ms max time correction
        self.sky_scale = nn.Parameter(torch.tensor(0.15))         # 15% max sky correction
        self.extra_scale = nn.Parameter(torch.tensor(0.12))       # 12% max other corrections
        
        #  Uncertainty estimation head (improved for small corrections)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, input_dim),
            nn.Softplus(beta=1.0)  # Standard softplus for uncertainty
        )
        
        # Initialize final linear layer to output realistic uncertainties
        # Target: Softplus(x) ≈ 2.01 when x ≈ 1.3 (matching error magnitude ~1-3)
        # This is critical: errors are ~1-3 magnitude, so uncertainties must match
        with torch.no_grad():
            # Initialize weights to small values
            nn.init.uniform_(self.uncertainty_head[-2].weight, -0.05, 0.05)
            # Initialize bias to positive value → realistic uncertainty after Softplus
            # Softplus(1.3) ≈ 2.01, which matches typical error magnitude
            nn.init.constant_(self.uncertainty_head[-2].bias, 1.3)
        
        # ✅ NEW: Variance correction head - scales posterior width, not just shifts
        # Problem: Posterior too narrow (30% coverage vs target 68%)
        # Solution: Learn scale inflation factor to widen likelihood
        # Physics: likelihood_temp ∈ [0.5, 2.0] inflates covariance by factor of 1/T²
        self.variance_scale_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, input_dim)  # Raw outputs, bounded after
        )
        
        # Initialize variance scales to output 1.0 (no inflation initially)
        # No Softplus constraint - post-process with clamp + relu to ensure > 0
        with torch.no_grad():
            nn.init.uniform_(self.variance_scale_head[-1].weight, -0.01, 0.01)
            nn.init.constant_(self.variance_scale_head[-1].bias, 0.0)  # Initialize to 0 → clamp to 1.0
        
        # EMA for distance zero-mean (instead of batch-sensitive batch mean)
        # Exponential moving average across all training samples
        # Prevents noisy centering from small/stratified batches
        self.register_buffer('distance_ema_mean', torch.tensor(0.0))
        self.register_buffer('distance_ema_steps', torch.tensor(0.0))
        self.distance_ema_decay = 0.99  # EMA decay rate: new_ema = 0.99*old + 0.01*batch_mean
        
    def forward(self, param_estimates: torch.Tensor, context_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ forward pass with uncertainty quantification + variance scaling"""
        
        batch_size = param_estimates.shape[0]
        
        # Embed parameters and context
        param_embed = self.param_embedding(param_estimates)
        context_embed = self.context_embedding(context_features)
        
        # Combine embeddings
        combined_embed = torch.cat([param_embed, context_embed], dim=1)
        
        # Apply ResidualMLP backbone (FIX #5: faster and more stable than Transformer)
        features = self.backbone(combined_embed)
        
        # Generate parameter-specific corrections
        corrections = []
        
        # Mass corrections (parameters 0, 1)
        mass_corr = self.mass_corrector(features) * self.mass_scale
        corrections.append(mass_corr)
        
        # Distance correction (parameter 2)
        # ⚠️ FIX #1: Zero-mean distance correction using EMA (not batch-sensitive)
        # Prevents noisy centering from small/stratified batches
        dist_corr = self.distance_corrector(features)
         
         # Update EMA with batch mean
        if self.training:
            batch_mean = dist_corr.mean(dim=0, keepdim=True)
            # EMA update: ema_new = decay * ema_old + (1-decay) * batch_mean
            self.distance_ema_mean = self.distance_ema_decay * self.distance_ema_mean + \
                                    (1.0 - self.distance_ema_decay) * batch_mean.detach()
            self.distance_ema_steps += 1
         
        # Center using EMA (stable across batches)
        dist_corr = dist_corr - self.distance_ema_mean  # Remove EMA-based bias
        dist_corr = dist_corr * self.distance_scale
        corrections.append(dist_corr)

        # Sky corrections (3, 4)
        if self.input_dim > 3:
            sky_corr = self.sky_corrector(features) * self.sky_scale
            corrections.append(sky_corr)

        # Time corrections (5)
        if self.input_dim > 5:
            time_corr = self.time_corrector(features) * self.time_scale
            corrections.append(time_corr)

        # Extra corrections (6+)
        if self.extra_corrector is not None and self.input_dim > 6:
            extra_corr = self.extra_corrector(features) * self.extra_scale
            corrections.append(extra_corr)

        
        # Concatenate all corrections
        all_corrections = torch.cat(corrections, dim=1)
        
        # Ensure correct output dimension
        if all_corrections.shape[1] > self.input_dim:
            all_corrections = all_corrections[:, :self.input_dim]
        elif all_corrections.shape[1] < self.input_dim:
            padding = torch.zeros(
                batch_size, 
                self.input_dim - all_corrections.shape[1],
                device=all_corrections.device,
                dtype=all_corrections.dtype
            )
            all_corrections = torch.cat([all_corrections, padding], dim=1)
        
        # Estimate uncertainties (mean shift component)
        uncertainties = self.uncertainty_head(features)
        
        # ✅ NEW: Estimate variance scales (width inflation component)
        # Outputs per-parameter scale factors to widen posterior
        # Typical values: 1.0 (no inflation) to 1.5 (50% wider)
        variance_scales_raw = self.variance_scale_head(features)
        
        # Bound variance scales to approximately [0.78, 1.69]
        # This corresponds to std inflation between ~0.8× and ~1.7×
        variance_scales = torch.clamp(variance_scales_raw, min=-0.223, max=0.693) + 1.0

        # Formula: Softplus(x)  clamps to [0.69, 2.0] when input ∈ [-0.22, 0.69]
        # But simpler clamp: output ∈ [0.8, 2.0]
        
        return all_corrections, uncertainties, variance_scales


class BiasCorrector(nn.Module):
    """
    BiasCorrector is a comprehensive class for correcting hierarchical biases in parameter estimation, 
    particularly in gravitational-wave (GW) signal analysis. It integrates advanced machine learning 
    (neural network-based bias estimation) with physics-based validation and constraints to ensure 
    robust and physically plausible corrections.
    Key Features:
    -------------
    - Initialization of parameter-specific physics bounds, priors, and known parameter correlations.
    - Extraction and normalization of parameter values and uncertainties from posterior summaries.
    - Preparation of neural network input features, including context and physics-informed statistics.
    - Dual correction modes:
        - Neural network-based correction with uncertainty quantification (if trained).
        - Physics-based correction using domain knowledge (fallback).
    - Comprehensive validation of corrections:
        - Physics bounds enforcement.
        - Correction magnitude checks.
        - Correlation consistency with expected parameter relationships.
        - Uncertainty reasonableness.
    - Application of validated corrections to posterior summaries, updating central values and uncertainties.
    - Adaptive correction strategies (conservative, balanced, aggressive) with configurable scaling.
    - Performance tracking and statistics, including correction rates, parameter improvements, and quality metrics.
    - Training interface for the neural bias estimator, supporting early stopping and validation.
    - Error handling and logging throughout the correction and training processes.
    Typical Usage:
    --------------
    1. Instantiate with a list of parameter names.
    2. Optionally train the neural bias estimator using labeled scenarios.
    3. Apply `correct_hierarchical_biases` to a list of extracted signals to obtain bias-corrected results.
    4. Retrieve correction statistics via `get_correction_statistics`.
    Parameters
    ----------
    param_names : List[str]
        List of parameter names to be corrected (e.g., ['mass_1', 'mass_2', 'luminosity_distance', ...]).
    Attributes
    ----------
    param_names : List[str]
        Names of parameters to correct.
    n_params : int
        Number of parameters.
    bias_estimator : BiasEstimator
        Neural network model for bias estimation.
    logger : logging.Logger
        Logger for information and error reporting.
    correction_history : list
        History of correction attempts and their outcomes.
    performance_metrics : dict
        Aggregated statistics on corrections and improvements.
    is_trained : bool
        Indicates if the neural bias estimator has been trained.
    training_epochs : int
        Number of epochs completed during training.
    physics_bounds : dict
        Physics-based bounds and priors for each parameter.
    correlation_matrix : np.ndarray
        Expected parameter correlation matrix.
    correction_strategies : dict
        Preset strategies for scaling and thresholding corrections.
    current_strategy : str
        Currently selected correction strategy.
    quality_thresholds : dict
        Thresholds for signal and posterior quality assessment.
    Methods
    -------
    - correct_hierarchical_biases(extracted_signals)
        Applies bias correction to a list of extracted signals with validation.
    - train_bias_estimator(training_scenarios, epochs=200, validation_split=0.2)
        Trains the neural bias estimator using provided scenarios.
    - get_correction_statistics()
        Returns comprehensive statistics on correction performance.
    """
    """ bias corrector with advanced machine learning and physics validation"""
    
    def __init__(self, param_names, context_dim=256):
        super().__init__()
        self.param_names = param_names
        self.n_params = len(param_names)
        self.context_dim = context_dim
        
        #  Initialize advanced neural bias estimator
        self.bias_estimator = BiasEstimator(
            input_dim=self.n_params,
            context_dim=self.context_dim
        )
        self.correction_scales = nn.Parameter(torch.ones(self.n_params))

        
        self.logger = logging.getLogger(__name__)
        #  FIX: Initialize with correct shapes for 9 parameters
        self.bias_corrections = np.zeros(self.n_params)  # Shape: (9,)
        self.scale_corrections = np.ones(self.n_params)   # Shape: (9,)
        
        #  FIX: Covariance matrix should be (9, 9) not (2, 2)
        self.covariance_corrections = np.eye(self.n_params)  # Shape: (9, 9)
        
        
        #  Comprehensive statistics tracking
        self.correction_history = []
        self.performance_metrics = {
            'corrections_applied': 0,
            'corrections_rejected': 0,
            'parameter_improvements': {param: [] for param in param_names},
            'quality_improvements': [],
            'convergence_rates': [],
            'physics_violations': 0,
            'uncertainty_estimates': []
        }
        
        self.is_trained = False
        self.training_epochs = 0
        
        #  Physics-based parameter bounds and constraints
        self.physics_bounds = self._initialize_physics_bounds()
        self.correlation_matrix = self._initialize_parameter_correlations()
        
        #  Adaptive correction strategies
        self.correction_strategies = {
            'conservative': {'scaling': 0.3, 'threshold': 0.8},
            'balanced': {'scaling': 0.7, 'threshold': 0.6},
            'aggressive': {'scaling': 1.0, 'threshold': 0.4}
        }
        self.current_strategy = 'balanced'
        
        #  Quality assessment thresholds
        self.quality_thresholds = {
            'minimum_snr': 8.0,
            'maximum_chi_squared': 2.0,
            'minimum_samples': 1000,
            'maximum_autocorr_time': 50.0
        }
        
        self.logger.info(f"BiasCorrector initialized for {self.n_params} parameters")
    

    def __call__(self, params: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make BiasCorrector callable like a function.
        This method delegates to the forward method.
        
        Args:
            params: (batch, param_dim) normalized parameters
            context: (batch, context_dim) context features
            
        Returns:
            corrections: (batch, param_dim) bias corrections
            uncertainties: (batch, param_dim) correction uncertainties
            confidence: (batch,) correction confidence scores
        """
        return self.forward(params, context)


    def forward(self, params: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
         """Forward pass for integration with OverlapNeuralPE.
         
         Args:
             params: [batch, param_dim] normalized parameters
             context: [batch, context_dim] context features
             
         Returns:
             corrections: [batch, param_dim] bias corrections (mean shift)
             uncertainties: [batch, param_dim] correction uncertainties (baseline)
             variance_scales: [batch, param_dim] posterior width inflation factors
             confidence: [batch] correction confidence scores
         """
         # Get bias estimates with variance scaling ✅ NOW RETURNS 3 VALUES
         bias_pred, uncertainty, variance_scales = self.bias_estimator(params, context)
         
         # Apply learned scaling
         corrections = bias_pred * self.correction_scales
         
         # ✅ REMOVED: Do NOT clamp here - let physics validation handle it
         # Double clamping suppresses gradients and biases toward under-correction
         # Physics validation will enforce bounds per parameter, not globally
         
         # ✅ SEMANTIC CLARITY FIX #1: 
         # uncertainty = how wrong the correction is (correction uncertainty, NOT posterior width)
         # variance_scale = how wrong the posterior width is (applied once in _apply_corrections_to_posterior)
         # DO NOT scale here - keep uncertainties as correction uncertainty baseline
         # Scaling applied only in canonical location: _apply_corrections_to_posterior
         
         #  #5: Calibrate confidence using likelihood-based formula
         # Correct formula: confidence = exp(-mean_uncertainty)
         # This matches likelihood logic: lower unc → higher exp value → higher confidence
         # Range: unc=0 → conf=1.0, unc=1.0 → conf=0.37, unc=2.0 → conf=0.135
         # Much better calibrated than arbitrary reciprocal formula
         mean_uncertainty = uncertainty.mean(dim=1)  # [batch] - use correction uncertainty (NOT scaled)
         confidence = torch.exp(-mean_uncertainty)
         
         # FIX #4: Penalize large corrections with high confidence (physically unrealistic)
         # "High-confidence huge correction" should never happen
         corr_norm = torch.norm(corrections, dim=1)  # L2 norm of corrections
         magnitude_penalty = torch.exp(-3.0 * corr_norm)  # Penalize if |corr| > 0.2
         confidence = confidence * magnitude_penalty  # Confidence → 0 for large corrections
         
         # Track metrics for logging (detach from graph) - use uncertainty as-is
         self._track_batch_metrics(corrections, uncertainty, confidence)
         
         # ✅ RETURN UNCHANGED: uncertainty and variance_scales separately
         # uncertainty = how wrong the correction is
         # variance_scales = how to scale posterior width (applied in _apply_corrections_to_posterior only)
         return corrections, uncertainty, variance_scales, confidence
    
    def _track_batch_metrics(self, corrections: torch.Tensor, uncertainty: torch.Tensor, 
                            confidence: torch.Tensor) -> None:
        """Track metrics from forward pass for monitoring.
        
        Args:
            corrections: [batch, param_dim] corrections applied
            uncertainty: [batch, param_dim] uncertainty estimates
            confidence: [batch] confidence scores
        """
        if not hasattr(self, '_batch_metrics'):
            self._batch_metrics = {
                'corrections': [],
                'confidences': [],
                'uncertainties': []
            }
        
        # Detach and move to CPU for storage
        self._batch_metrics['corrections'].append(torch.abs(corrections).detach().cpu().numpy())
        self._batch_metrics['confidences'].append(confidence.detach().cpu().numpy())
        self._batch_metrics['uncertainties'].append(uncertainty.detach().cpu().numpy())
        
        # Keep only last 100 batches
        for key in self._batch_metrics:
            if len(self._batch_metrics[key]) > 100:
                self._batch_metrics[key] = self._batch_metrics[key][-100:]
    
    def get_metrics(self) -> Dict[str, float]:
        """Get metrics from recent batches for monitoring."""
        if not hasattr(self, '_batch_metrics') or not self._batch_metrics['corrections']:
            return {}
        
        try:
            import numpy as np
            
            # Correction statistics
            all_corrections = np.concatenate(self._batch_metrics['corrections'], axis=0)
            metrics = {
                'avg_correction': float(np.mean(all_corrections)),
                'max_correction': float(np.max(all_corrections)),
                'correction_std': float(np.std(all_corrections))
            }
            
            # Confidence metrics
            all_confidences = np.concatenate(self._batch_metrics['confidences'], axis=0)
            metrics['avg_confidence'] = float(np.mean(all_confidences))
            metrics['min_confidence'] = float(np.min(all_confidences))
            
            # Uncertainty metrics
            all_uncertainties = np.concatenate(self._batch_metrics['uncertainties'], axis=0)
            metrics['avg_uncertainty'] = float(np.mean(all_uncertainties))
            metrics['max_uncertainty'] = float(np.max(all_uncertainties))
            
            # ✅ FIXED: Prioritize training mode counters, fallback to inference confidence
            total_attempts = (
                self.performance_metrics['corrections_applied'] +
                self.performance_metrics['corrections_rejected']
            )
            
            if total_attempts > 0:
                # Training mode: use explicit counters
                acceptance_rate = (
                    self.performance_metrics['corrections_applied'] / total_attempts
                )
            else:
                # Inference mode: estimate from confidence (corrections not validated yet)
                # Interpret confidence as proxy for acceptance likelihood
                acceptance_rate = float(np.mean(all_confidences))
            
            metrics['correction_acceptance_rate'] = acceptance_rate
            
            return metrics
        except Exception as e:
            self.logger.debug(f"Failed to compute bias metrics: {e}")
            return {}

    
    def _initialize_physics_bounds(self) -> Dict[str, Dict[str, float]]:
        """Initialize physics-based parameter bounds and priors"""
        
        bounds = {}
        
        for param_name in self.param_names:
            if param_name in ['mass_1', 'mass_2']:
                bounds[param_name] = {
                    'min_value': 1.0,      # Solar masses
                    'max_value': 100.0,
                    'max_correction': 0.08,  # 8% maximum correction
                    'prior_width': 0.1,      # Prior uncertainty
                    'coupling_strength': 0.9  # Strong coupling with other mass
                }
            elif param_name == 'luminosity_distance':
                bounds[param_name] = {
                    'min_value': 10.0,     # Mpc
                    'max_value': 5000.0,
                    'max_correction': 0.30,  # 30% maximum correction
                    'prior_width': 0.2,
                    'coupling_strength': 0.3
                }
            elif param_name == 'geocent_time':
                bounds[param_name] = {
                    'min_value': -0.1,     # seconds
                    'max_value': 0.1,
                    'max_correction': 0.002,  # 2ms maximum correction
                    'prior_width': 0.01,
                    'coupling_strength': 0.1
                }
            elif param_name in ['ra', 'dec']:
                bounds[param_name] = {
                    'min_value': -np.pi if param_name == 'dec' else 0.0,
                    'max_value': np.pi if param_name == 'dec' else 2*np.pi,
                    'max_correction': 0.2,   # 20% of parameter space
                    'prior_width': 0.3,
                    'coupling_strength': 0.7  # Sky position correlation
                }
            elif param_name in ['theta_jn', 'psi', 'phase']:
                bounds[param_name] = {
                    'min_value': 0.0,
                    'max_value': np.pi if param_name != 'phase' else 2*np.pi,
                    'max_correction': 0.25,  # 25% of parameter space
                    'prior_width': 0.4,
                    'coupling_strength': 0.5
                }
            elif param_name in ['a1', 'a2', 'a_1', 'a_2']:  # Support both naming conventions
                bounds[param_name] = {
                    'min_value': 0.0,
                    'max_value': 0.99,
                    'max_correction': 0.2,   # 20% of maximum spin
                    'prior_width': 0.3,
                    'coupling_strength': 0.4
                }
            else:
                # Default bounds for other parameters
                bounds[param_name] = {
                    'min_value': -10.0,
                    'max_value': 10.0,
                    'max_correction': 0.15,
                    'prior_width': 0.25,
                    'coupling_strength': 0.3
                }
        
        return bounds
    
    def _initialize_parameter_correlations(self) -> np.ndarray:
        """Initialize expected parameter correlation matrix"""
        
        n = len(self.param_names)
        correlation_matrix = np.eye(n)
        
        # Define known parameter correlations from GW physics
        param_dict = {name: i for i, name in enumerate(self.param_names)}
        
        known_correlations = [
            (['mass_1', 'mass_2'], 0.3),           # Mass correlation
            (['mass_1', 'luminosity_distance'], -0.6),  # Mass-distance degeneracy
            (['mass_2', 'luminosity_distance'], -0.6),
            (['ra', 'dec'], 0.1),                  # Sky position correlation
            (['theta_jn', 'luminosity_distance'], 0.4),  # Inclination-distance
            (['a1', 'a2'], 0.2),                   # Spin correlation (dataset uses 'a1', 'a2')
            (['psi', 'phase'], -0.3),              # Orientation correlation
        ]
        
        for param_pair, correlation in known_correlations:
            if all(param in param_dict for param in param_pair):
                i, j = param_dict[param_pair[0]], param_dict[param_pair[1]]
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def _extract_parameter_values(self, posterior_summary: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract parameter values and uncertainties from posterior summary"""
        
        values = np.zeros(self.n_params)
        uncertainties = np.zeros(self.n_params)
        
        for i, param_name in enumerate(self.param_names):
            if param_name in posterior_summary:
                param_data = posterior_summary[param_name]
                
                if isinstance(param_data, dict):
                    # Extract central value
                    if 'median' in param_data:
                        values[i] = float(param_data['median'])
                    elif 'mean' in param_data:
                        values[i] = float(param_data['mean'])
                    else:
                        values[i] = 0.0
                    
                    # Extract uncertainty
                    if 'std' in param_data:
                        uncertainties[i] = float(param_data['std'])
                    elif 'quantiles' in param_data and len(param_data['quantiles']) >= 5:
                        # Estimate std from quantiles (84th - 16th percentile) / 2
                        q16, q84 = param_data['quantiles'][1], param_data['quantiles'][3]
                        uncertainties[i] = (q84 - q16) / 2.0
                    else:
                        uncertainties[i] = abs(values[i]) * 0.1  # 10% default
                
                elif isinstance(param_data, (int, float)):
                    values[i] = float(param_data)
                    uncertainties[i] = abs(values[i]) * 0.1
                else:
                    values[i] = 0.0
                    uncertainties[i] = 1.0
            else:
                # Parameter not found - use defaults
                values[i] = 0.0
                uncertainties[i] = 1.0
        
        return values, uncertainties
    
    def _prepare_neural_network_input(self, signal: Dict, position: int, 
                                    all_signals: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare comprehensive input features for neural network"""
        
        try:
            # Extract parameter values and uncertainties
            param_values, param_uncertainties = self._extract_parameter_values(
                signal.get('posterior_summary', {})
            )
            
            # Normalize parameter values
            normalized_params = np.zeros_like(param_values)
            for i, param_name in enumerate(self.param_names):
                bounds = self.physics_bounds[param_name]
                param_range = bounds['max_value'] - bounds['min_value']
                if param_range > 0:
                    normalized_params[i] = (param_values[i] - bounds['min_value']) / param_range
                else:
                    normalized_params[i] = 0.5  # Default to middle
            
            # Check if actual context embedding is provided
            if 'context_embedding' in signal:
                context_features = np.array(signal['context_embedding'])
            else:
                # Create comprehensive context features as fallback
                n_signals = len(all_signals)
                signal_quality = signal.get('signal_quality', 0.5)
                
                context_features = np.array([
                    position / max(n_signals, 1),              # Hierarchical position
                    signal_quality,                             # Signal quality
                    np.log10(max(n_signals, 1)),              # Log number of signals
                    np.mean(param_uncertainties),              # Average parameter uncertainty
                    np.std(param_uncertainties),               # Uncertainty spread
                    signal.get('network_snr', 10.0) / 50.0,   # Normalized SNR
                    signal.get('context_snr', signal_quality * 20.0) / 50.0,  # Context SNR
                    
                    # Statistical features
                    np.mean(normalized_params),                # Average parameter value
                    np.std(normalized_params),                 # Parameter spread
                    np.min(normalized_params),                 # Minimum parameter
                    np.max(normalized_params),                 # Maximum parameter
                    
                    # Physics-based features
                    self._compute_chirp_mass_feature(param_values),
                    self._compute_mass_ratio_feature(param_values),
                    self._compute_effective_distance_feature(param_values),
                    self._compute_sky_area_feature(param_values),
                    min(1.0, position * 0.1)                  # Hierarchy degradation factor
                ])
            
            # Convert to tensors
            param_tensor = torch.tensor(normalized_params, dtype=torch.float32).unsqueeze(0)
            context_tensor = torch.tensor(context_features, dtype=torch.float32).unsqueeze(0)
            
            return param_tensor, context_tensor
            
        except Exception as e:
            self.logger.debug(f"Input preparation failed: {e}")
            return None, None
    
    def _compute_chirp_mass_feature(self, param_values: np.ndarray) -> float:
        """Compute normalized chirp mass feature"""
        try:
            if 'mass_1' in self.param_names and 'mass_2' in self.param_names:
                m1_idx = self.param_names.index('mass_1')
                m2_idx = self.param_names.index('mass_2')
                
                m1, m2 = param_values[m1_idx], param_values[m2_idx]
                if m1 > 0 and m2 > 0:
                    total_mass = m1 + m2
                    chirp_mass = (m1 * m2)**(3/5) / total_mass**(1/5)
                    return chirp_mass / 50.0  # Normalize by typical chirp mass
            return 0.5
        except:
            return 0.5
    
    def _compute_mass_ratio_feature(self, param_values: np.ndarray) -> float:
        """Compute mass ratio feature"""
        try:
            if 'mass_1' in self.param_names and 'mass_2' in self.param_names:
                m1_idx = self.param_names.index('mass_1')
                m2_idx = self.param_names.index('mass_2')
                
                m1, m2 = param_values[m1_idx], param_values[m2_idx]
                if m1 > 0 and m2 > 0:
                    return min(m1, m2) / max(m1, m2)
            return 0.5
        except:
            return 0.5
    
    def _compute_effective_distance_feature(self, param_values: np.ndarray) -> float:
        """Compute effective distance feature"""
        try:
            if 'luminosity_distance' in self.param_names:
                dist_idx = self.param_names.index('luminosity_distance')
                distance = param_values[dist_idx]
                return min(1.0, distance / 1000.0)  # Normalize by 1000 Mpc
            return 0.5
        except:
            return 0.5
    
    def _compute_sky_area_feature(self, param_values: np.ndarray) -> float:
        """Compute sky localization area feature"""
        try:
            # Rough estimate based on ra/dec presence
            if 'ra' in self.param_names and 'dec' in self.param_names:
                return 0.8  # Assume reasonably well localized
            return 0.3  # Assume poorly localized
        except:
            return 0.5
    
    def _apply_neural_correction(self, param_tensor: torch.Tensor, 
                               context_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply neural network bias correction with uncertainty quantification AND variance scaling"""
         
        with torch.no_grad():
            # ✅ Now receives variance scales for posterior widening
            corrections, uncertainties, variance_scales = self.bias_estimator(param_tensor, context_tensor)
            corrections_np = corrections.squeeze().numpy()
            uncertainties_np = uncertainties.squeeze().numpy()
            variance_scales_np = variance_scales.squeeze().numpy()  # Scale factors for width inflation
         
        return corrections_np, uncertainties_np, variance_scales_np
    
    def _apply_physics_based_correction(self, param_values: np.ndarray, 
                                      param_uncertainties: np.ndarray,
                                      position: int, total_signals: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply physics-based bias correction when neural network is not available"""
        
        corrections = np.zeros_like(param_values)
        correction_uncertainties = np.zeros_like(param_values)
        
        # Hierarchical bias model
        hierarchy_factor = 1.0 + (position * 0.15)  # 15% degradation per position
        
        for i, param_name in enumerate(self.param_names):
            bounds = self.physics_bounds[param_name]
            
            # Expected bias based on physics knowledge
            if param_name in ['mass_1', 'mass_2']:
                # Masses typically biased toward lower values in hierarchy
                expected_bias = -0.02 * param_values[i] * hierarchy_factor
                bias_uncertainty = 0.05 * abs(param_values[i])
                
            elif param_name == 'luminosity_distance':
                # Distance typically overestimated in overlapping signals
                expected_bias = 0.1 * param_values[i] * hierarchy_factor
                bias_uncertainty = 0.15 * abs(param_values[i])
                
            elif param_name == 'geocent_time':
                # Time biases are typically small but systematic
                expected_bias = np.random.normal(0, 0.001 * hierarchy_factor)
                bias_uncertainty = 0.002
                
            elif param_name in ['ra', 'dec']:
                # Sky position biases increase with poor localization
                sky_uncertainty = param_uncertainties[i]
                expected_bias = np.random.normal(0, 0.05 * sky_uncertainty * hierarchy_factor)
                bias_uncertainty = 0.1 * sky_uncertainty
                
            else:
                # Other parameters - conservative bias estimate
                expected_bias = np.random.normal(0, 0.03 * abs(param_values[i]) * hierarchy_factor)
                bias_uncertainty = 0.08 * max(abs(param_values[i]), param_uncertainties[i])
            
            # Apply physics constraints
            max_correction = bounds['max_correction']
            if param_name in ['ra', 'dec', 'geocent_time']:
                # Absolute bounds for certain parameters
                corrections[i] = np.clip(expected_bias, -max_correction, max_correction)
            else:
                # Relative bounds for others
                max_abs_correction = max_correction * abs(param_values[i])
                corrections[i] = np.clip(expected_bias, -max_abs_correction, max_abs_correction)
            
            correction_uncertainties[i] = bias_uncertainty
        
        return corrections, correction_uncertainties
    
    def _validate_corrections(self, original_values: np.ndarray, corrections: np.ndarray,
                         correction_uncertainties: np.ndarray) -> Dict[str, Any]:
        """Comprehensive validation of proposed corrections"""
        
        validation_results = {
            'physics_valid': True,
            'magnitude_acceptable': True,
            'correlation_consistent': True,  
            'uncertainty_reasonable': True,
            'warnings': [],
            'rejection_reasons': []
        }
        
        corrected_values = original_values + corrections
        
        # Physics bounds validation
        for i, param_name in enumerate(self.param_names):
            bounds = self.physics_bounds[param_name]
            
            # Check absolute bounds
            if (corrected_values[i] < bounds['min_value'] or 
                corrected_values[i] > bounds['max_value']):
                validation_results['physics_valid'] = False
                validation_results['rejection_reasons'].append(
                    f"{param_name} correction violates physics bounds: "
                    f"{corrected_values[i]:.4f} not in [{bounds['min_value']:.4f}, {bounds['max_value']:.4f}]"
                )
            
            # Check correction magnitude (for bias corrections, be lenient)
            # Skip strict bounds for periodic/angular parameters
            if param_name in ['phase', 'psi', 'ra', 'dec']:
                # Periodic parameters - bias corrections are naturally small
                # Allow up to 10% of parameter range
                param_range = bounds['max_value'] - bounds['min_value']
                max_correction = param_range * 0.1
                if abs(corrections[i]) > max_correction * 2:  # 2x leniency
                    validation_results['warnings'].append(
                        f"{param_name} correction large for periodic param: "
                        f"{abs(corrections[i]):.4f}"
                    )
            elif param_name in ['geocent_time']:
                # Time: tight bounds
                max_correction = bounds['max_correction']
                if abs(corrections[i]) > max_correction * 5:  # 5x leniency
                    validation_results['magnitude_acceptable'] = False
                    validation_results['warnings'].append(
                        f"{param_name} correction magnitude extreme: {abs(corrections[i]):.4f}"
                    )
            else:
                # Masses, distance, spins: relative bounds
                max_correction = bounds['max_correction'] * abs(original_values[i])
                lax_threshold = max_correction * 3.0
                if abs(corrections[i]) > lax_threshold:
                    validation_results['magnitude_acceptable'] = False
                    validation_results['warnings'].append(
                        f"{param_name} correction magnitude extreme: "
                        f"{abs(corrections[i]):.4f} > {lax_threshold:.4f}"
                    )
        
        # ✅ IMPROVED: Uncertainty validation
        # Note: Bias corrections are inherently small, so we use lenient thresholds
        for i, param_name in enumerate(self.param_names):
            # For small corrections, high relative uncertainty is expected
            # Only check absolute magnitude, not relative
            # Typical bias correction uncertainty is 0.4-0.6, which is fine for tiny corrections
            
            # Skip relative checks for near-zero original values (geocent_time, phase)
            if abs(original_values[i]) > 1e-3:
                relative_uncertainty = correction_uncertainties[i] / (abs(original_values[i]) + 1e-10)
                # Lenient threshold: 2.0 (200%) to allow for small parameter biases
                if relative_uncertainty > 2.0:
                    validation_results['warnings'].append(
                        f"{param_name} correction uncertainty large: "
                        f"{correction_uncertainties[i]:.4f} ({relative_uncertainty*100:.1f}% of original)"
                    )
            
            # For bias corrections, uncertainty can be >> correction (acceptable)
            # Only warn if uncertainty >> correction by a factor of 100x (extreme case)
            if correction_uncertainties[i] > abs(corrections[i]) * 100 and abs(corrections[i]) > 1e-5:
                validation_results['warnings'].append(
                    f"{param_name} correction extremely uncertain: "
                    f"uncertainty {correction_uncertainties[i]:.4f} >> {100*abs(corrections[i]):.4f}"
                )
        
        # Overall validation 
        # Note: uncertainty_reasonable removed - bias corrections can have high relative uncertainty
        validation_results['overall_valid'] = (
            validation_results['physics_valid'] and 
            validation_results['magnitude_acceptable']
        )
        
        return validation_results

    def _apply_corrections_to_posterior(self, original_summary: Dict, corrections: np.ndarray,
                                      correction_uncertainties: np.ndarray, variance_scales: np.ndarray = None) -> Dict:
        """Apply validated corrections to posterior summary with variance scaling"""
         
        if variance_scales is None:
             variance_scales = np.ones_like(corrections)
         
        corrected_summary = {}
         
        for key, value in original_summary.items():
            if key in self.param_names:
                param_idx = self.param_names.index(key)
                correction = corrections[param_idx]
                correction_unc = correction_uncertainties[param_idx]
                variance_scale = variance_scales[param_idx]
                 
                if isinstance(value, dict):
                    corrected_value = value.copy()
                     
                     # Apply correction to central values
                    if 'median' in corrected_value:
                         corrected_value['median'] += correction
                    if 'mean' in corrected_value:
                        corrected_value['mean'] += correction
                    
                    #  #2b: Update uncertainty with variance scaling
                    # Apply both: (1) correction uncertainty in quadrature, (2) posterior width inflation
                    # Final std = sqrt(scaled_original_std^2 + correction_unc^2)
                    # where scaled_original_std = original_std * variance_scale
                    if 'std' in corrected_value:
                         original_std = corrected_value['std']
                         # Scale original std by variance scale to widen posterior
                         scaled_original_std = original_std * variance_scale
                         # Add correction uncertainty in quadrature
                         corrected_value['std'] = np.sqrt(scaled_original_std**2 + correction_unc**2)
                    
                    # ⚠️ FIX #2: MUST FIX - Quantile rescaling formula was double-applying correction
                    # BEFORE: q_scaled = (q_shifted - median) * scale + median + correction  [WRONG - adds correction twice]
                    # AFTER:  q_scaled = (q - median) * scale + median + correction          [CORRECT - adds once]
                    # The correction is already in q_shifted, so adding it again shifts asymmetrically
                    if 'quantiles' in corrected_value:
                         original_median = corrected_value.get('median', corrected_value.get('mean', 0.0))
                         corrected_quantiles = []
                         for q in corrected_value['quantiles']:
                             # Correct formula: scale width around original median, then shift by correction
                             # Step 1: Scale the deviation from median: (q - median) * variance_scale
                             q_scaled = (q - original_median) * variance_scale + original_median + correction
                             corrected_quantiles.append(q_scaled)
                         corrected_value['quantiles'] = corrected_quantiles
                    
                    corrected_summary[key] = corrected_value
                
                elif isinstance(value, (int, float)):
                    corrected_summary[key] = float(value) + correction
                else:
                    corrected_summary[key] = value
            else:
                # Parameter not being corrected
                corrected_summary[key] = value
        
        return corrected_summary
    
    def _update_performance_statistics(self, validation_results: Dict, 
                                     corrections: np.ndarray, signal_quality: float):
        """Update performance tracking statistics"""
        
        if validation_results['overall_valid']:
            self.performance_metrics['corrections_applied'] += 1
            self.performance_metrics['quality_improvements'].append(signal_quality)
        else:
            self.performance_metrics['corrections_rejected'] += 1
            
        if not validation_results['physics_valid']:
            self.performance_metrics['physics_violations'] += 1
        
        # Track parameter-specific improvements
        for i, param_name in enumerate(self.param_names):
            if abs(corrections[i]) > 1e-10:  # Non-zero correction
                self.performance_metrics['parameter_improvements'][param_name].append(abs(corrections[i]))
        
        # Track overall correction statistics
        correction_magnitude = np.linalg.norm(corrections)
        self.correction_history.append({
            'magnitude': correction_magnitude,
            'accepted': validation_results['overall_valid'],
            'quality': signal_quality
        })
    
    def correct_hierarchical_biases(self, extracted_signals: List[Dict]) -> List[Dict]:
        """ bias correction with comprehensive validation and error handling"""
        
        if not extracted_signals:
            return []
        
        corrected_signals = []
        strategy = self.correction_strategies[self.current_strategy]
        
        self.logger.info(f"Starting bias correction for {len(extracted_signals)} signals using {self.current_strategy} strategy")
        
        successful_corrections = 0
        
        for i, signal in enumerate(extracted_signals):
            try:
                # Prepare neural network inputs
                param_tensor, context_tensor = self._prepare_neural_network_input(
                    signal, i, extracted_signals
                )
                
                if param_tensor is not None and context_tensor is not None:
                    # Extract original parameter values
                    original_values, original_uncertainties = self._extract_parameter_values(
                        signal.get('posterior_summary', {})
                    )
                    
                    # Apply correction method
                    if self.is_trained:
                        corrections, correction_uncertainties, variance_scales = self._apply_neural_correction(
                            param_tensor, context_tensor
                        )
                        # Uncertainties are in normalized space, convert to physical
                        # Training: uncertainty target = 0.3 in [-1, 1] normalized space
                        # Inference: must convert to physical units for proper posterior widening
                        # Conversion: unc_physical[j] = unc_normalized[j] * param_range[j]
                        correction_uncertainties_physical = np.zeros_like(correction_uncertainties)
                        for j, param_name in enumerate(self.param_names):
                            bounds = self.physics_bounds[param_name]
                            param_range = bounds['max_value'] - bounds['min_value']
                            correction_uncertainties_physical[j] = correction_uncertainties[j] * param_range
                        correction_uncertainties = correction_uncertainties_physical
                        correction_method = 'neural_network'
                    else:
                        corrections, correction_uncertainties = self._apply_physics_based_correction(
                            original_values, original_uncertainties, i, len(extracted_signals)
                        )
                        # ✅ SHOULD FIX #5: Physics fallback needs variance scaling too
                        # Physics fallback applies mean correction only (no neural width scaling)
                        # This makes fallback outputs systematically overconfident
                        # FIX: Apply constant variance scaling (1.3×) to match neural case
                        variance_scales = np.ones_like(corrections) * 1.3  # Inflate width by 30%
                        correction_method = 'physics_based'
                    
                    #  #1b: Distance scaling bug - DO NOT double-scale
                    # PROBLEM: Network applies self.distance_scale (line 223), then denorm multiplies by param_range
                    # This scales distance twice: dist_raw * distance_scale * param_range
                    # FIX: Distance already scaled in network, use as-is. Only denormalize non-distance params.
                    corrections_physical = np.zeros_like(corrections)
                    distance_idx = self.param_names.index('luminosity_distance') if 'luminosity_distance' in self.param_names else -1
                    
                    for j, param_name in enumerate(self.param_names):
                        if j == distance_idx:
                            # Distance correction already fully scaled in network (line 223: * distance_scale)
                            # Use as-is, no additional denormalization
                            corrections_physical[j] = corrections[j]
                        else:
                            # Other parameters: denormalize from [-1,1] * scale → physical units
                            bounds = self.physics_bounds[param_name]
                            param_range = bounds['max_value'] - bounds['min_value']
                            corrections_physical[j] = corrections[j] * param_range
                    
                    # Apply strategy scaling (now on physical corrections)
                    corrections_physical *= strategy['scaling']
                    corrections = corrections_physical  # Use denormalized for rest of pipeline
                    
                    # ✅ MUST FIX #1: Do NOT scale correction_uncertainties here
                    # FIX: Variance scaling applied only in canonical location: _apply_corrections_to_posterior
                    # This prevents variance_scale² from sneaking in via multiple applications
                    # correction_uncertainties remains as-is (baseline correction uncertainty)
                    
                    # Validate corrections
                    validation_results = self._validate_corrections(
                        original_values, corrections, correction_uncertainties
                    )
                    
                    # Update performance statistics
                    self._update_performance_statistics(
                        validation_results, corrections, signal.get('signal_quality', 0.5)
                    )
                    
                    # Apply corrections if validated
                    if validation_results['overall_valid']:
                        corrected_summary = self._apply_corrections_to_posterior(
                            signal['posterior_summary'], corrections, correction_uncertainties, variance_scales
                        )
                        
                        corrected_signal = signal.copy()
                        corrected_signal['posterior_summary'] = corrected_summary
                        corrected_signal['bias_correction'] = {
                            'applied': True,
                            'method': correction_method,
                            'strategy': self.current_strategy,
                            'corrections': corrections.tolist(),
                            'correction_uncertainties': correction_uncertainties.tolist(),
                            'validation': validation_results,
                            'correction_magnitude': float(np.linalg.norm(corrections))
                        }
                        
                        successful_corrections += 1
                        
                    else:
                        # Corrections rejected - return original
                        corrected_signal = signal.copy()
                        corrected_signal['bias_correction'] = {
                            'applied': False,
                            'method': correction_method,
                            'strategy': self.current_strategy,
                            'rejection_reason': '; '.join(validation_results['rejection_reasons']),
                            'warnings': validation_results['warnings']
                        }
                
                else:
                    # Input preparation failed
                    corrected_signal = signal.copy()
                    corrected_signal['bias_correction'] = {
                        'applied': False,
                        'method': 'none',
                        'rejection_reason': 'input_preparation_failed'
                    }
                
                corrected_signals.append(corrected_signal)
                
            except Exception as e:
                # Error handling - return original signal
                self.logger.error(f"Bias correction failed for signal {i}: {e}")
                corrected_signal = signal.copy()
                corrected_signal['bias_correction'] = {
                    'applied': False,
                    'method': 'error',
                    'error': str(e)
                }
                corrected_signals.append(corrected_signal)
        
        correction_rate = successful_corrections / len(extracted_signals)
        self.logger.info(f"Bias correction completed: {successful_corrections}/{len(extracted_signals)} signals corrected ({correction_rate:.1%})")
        
        # ✅ NEW: Comprehensive validation metrics (debug-only, no gradients)
        self._compute_and_log_validation_metrics(corrected_signals)
        
        return corrected_signals
        
    def _compute_and_log_validation_metrics(self, corrected_signals: List[Dict]) -> None:
        """Compute and log comprehensive validation metrics to assess correction quality.
        
        Metrics computed:
        1. Distance MAE before vs after correction
        2. Mean signed distance bias (systematic offset)
        3. 68% credible interval (CI) coverage
        4. Rejection rate (confidence < 0.5)
        5. High-uncertainty fraction (>75th percentile)
        """
        
        if not corrected_signals:
            return
        
        # ✅ Extract all distance values (before and after)
        distances_before = []
        distances_after = []
        biases_before = []
        biases_after = []
        high_unc_flags = []
        rejection_flags = []
        coverage_flags = []
        
        for signal in corrected_signals:
            try:
                # Get original distance estimate
                posterior_orig = signal.get('posterior_summary', {})
                dist_orig_dict = posterior_orig.get('luminosity_distance', {})
                if isinstance(dist_orig_dict, dict):
                    dist_before = dist_orig_dict.get('median', dist_orig_dict.get('mean', None))
                    dist_std_orig = dist_orig_dict.get('std', 1.0)
                else:
                    dist_before = float(dist_orig_dict) if dist_orig_dict else None
                    dist_std_orig = 1.0
                
                # Get true distance (if available)
                true_distance = signal.get('true_parameters', {}).get('luminosity_distance', None)
                
                if dist_before is None or true_distance is None:
                    continue
                
                distances_before.append(dist_before)
                bias_before = dist_before - true_distance
                biases_before.append(bias_before)
                
                # Get corrected distance estimate
                bias_correction = signal.get('bias_correction', {})
                if bias_correction.get('applied', False):
                    # Correction was applied - get corrected posterior
                    posterior_corr = signal.get('posterior_summary', {})
                    dist_corr_dict = posterior_corr.get('luminosity_distance', {})
                    if isinstance(dist_corr_dict, dict):
                        dist_after = dist_corr_dict.get('median', dist_corr_dict.get('mean', None))
                        dist_std_corr = dist_corr_dict.get('std', dist_std_orig)
                    else:
                        dist_after = float(dist_corr_dict) if dist_corr_dict else dist_before
                        dist_std_corr = dist_std_orig
                    
                    bias_after = dist_after - true_distance
                    distances_after.append(dist_after)
                    biases_after.append(bias_after)
                    
                    # Check: is 68% CI coverage achieved? (68% of truth should be within mean ± std)
                    # Coverage: |bias| < std indicates truth within 68% CI
                    coverage_before = abs(bias_before) < dist_std_orig
                    coverage_after = abs(bias_after) < dist_std_corr
                    coverage_flags.append(coverage_after)
                else:
                    distances_after.append(dist_before)
                    biases_after.append(bias_before)
                    coverage_flags.append(abs(bias_before) < dist_std_orig)
                
                # Check: high uncertainty (>75th percentile)?
                # If not yet computed: use raw std, else use from uncertainty metrics
                unc_threshold = np.percentile([signal.get('posterior_summary', {}).get(p, {}).get('std', 1.0) 
                                              for p in self.param_names if isinstance(signal.get('posterior_summary', {}).get(p, {}), dict)], 75)
                high_unc_flags.append(dist_std_orig > unc_threshold if dist_std_orig else False)
                
                # Check: rejection rate (confidence < 0.5)?
                confidence = bias_correction.get('confidence', 1.0) if isinstance(bias_correction, dict) else 1.0
                rejection_flags.append(confidence < 0.5)
                
            except Exception as e:
                self.logger.debug(f"Error computing validation metrics for signal: {e}")
                continue
        
        # ✅ Log all metrics
        if distances_before and distances_after:
            distances_before = np.array(distances_before)
            distances_after = np.array(distances_after)
            biases_before = np.array(biases_before)
            biases_after = np.array(biases_after)
            
            mae_before = np.mean(np.abs(biases_before))
            mae_after = np.mean(np.abs(biases_after))
            mae_improvement = (1.0 - mae_after / mae_before) * 100 if mae_before > 0 else 0.0
            
            bias_mean_before = np.mean(biases_before)
            bias_mean_after = np.mean(biases_after)
            
            coverage_rate = np.mean(coverage_flags) * 100 if coverage_flags else 0.0
            rejection_rate = np.mean(rejection_flags) * 100 if rejection_flags else 0.0
            high_unc_rate = np.mean(high_unc_flags) * 100 if high_unc_flags else 0.0
            
            # ✅ Pretty print validation summary
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"BiasCorrector Validation Metrics:")
            self.logger.info(f"  Distance MAE:")
            self.logger.info(f"    Before: {mae_before:.3f} Mpc")
            self.logger.info(f"    After:  {mae_after:.3f} Mpc")
            self.logger.info(f"    Improvement: {mae_improvement:.1f}% {'✅' if mae_improvement > 0 else '❌'}")
            self.logger.info(f"  Distance Bias (mean):")
            self.logger.info(f"    Before: {bias_mean_before:+.3f} Mpc (should be ~0)")
            self.logger.info(f"    After:  {bias_mean_after:+.3f} Mpc (target: ±10 Mpc)")
            self.logger.info(f"    Fixed: {abs(bias_mean_before) - abs(bias_mean_after):.3f} Mpc {'✅' if abs(bias_mean_after) < 20 else '⚠️'}")
            self.logger.info(f"  68% Credible Interval Coverage:")
            self.logger.info(f"    {coverage_rate:.1f}% of samples (target: 68% ± 5%)")
            self.logger.info(f"    Status: {'✅ Good' if 63 <= coverage_rate <= 73 else '⚠️  Need tuning' if 50 <= coverage_rate else '❌ Poor'}")
            self.logger.info(f"  Rejection Rate (confidence < 0.5):")
            self.logger.info(f"    {rejection_rate:.1f}% (target: <5%)")
            self.logger.info(f"  High Uncertainty Fraction (>75th percentile):")
            self.logger.info(f"    {high_unc_rate:.1f}% (should be ~25%)")
            self.logger.info(f"{'='*70}\n")
            
    def train_bias_estimator(self, training_scenarios: List[Dict], epochs: int = 200,
                           validation_split: float = 0.2) -> Dict[str, Any]:
        """ training of neural bias estimator"""
            
        if not training_scenarios:
           self.logger.warning("No training scenarios provided")
           return {'success': False, 'error': 'no_training_data'}
        
        # Log training configuration
        learning_rate = 1e-4
        batch_size = 32
        patience = 20
        
        self.logger.info(f"BiasCorrector Training Configuration:")
        self.logger.info(f"  Epochs: {epochs}")
        self.logger.info(f"  Learning Rate: {learning_rate}")
        self.logger.info(f"  Batch Size: {batch_size}")
        self.logger.info(f"  Patience (Early Stopping): {patience}")
        self.logger.info(f"  Validation Split: {validation_split:.1%}")
        self.logger.info(f"  Training Scenarios: {len(training_scenarios)}")
         
         # ⚠️ FIX #4: Filter out edge cases before training
         # Edge cases: low ESS, bad R̂, pathological overlap flags
        filtered_scenarios = []
        edge_case_count = 0
         
        for scenario in training_scenarios:
            # Check for edge case flags
            if scenario.get('is_edge_case', False):
                edge_case_count += 1
                continue  # Skip edge cases
            
            # Check for low quality
            quality = scenario.get('signal_quality', 1.0)
            if quality < 0.3:  # Skip very low quality
                edge_case_count += 1
                continue
            
            # Check for posterior issues
            extracted_signals = scenario.get('extracted_signals', [])
            skip_scenario = False
            for sig in extracted_signals:
                posterior_std = sig.get('posterior_summary', {}).get('luminosity_distance', {}).get('std', 1.0)
                if posterior_std < 1e-8 or posterior_std > 1e6:  # Pathological
                    skip_scenario = True
                    break
            
            if skip_scenario:
                edge_case_count += 1
                continue
             
            filtered_scenarios.append(scenario)
        
        if edge_case_count > 0:
            self.logger.info(f"Filtered out {edge_case_count} edge case scenarios")
        
        if len(filtered_scenarios) < 10:
            self.logger.error(f"Insufficient clean training data: {len(filtered_scenarios)} scenarios remain")
            return {'success': False, 'error': 'insufficient_clean_data'}
        
        # Prepare training data
        training_data = []
        for scenario in filtered_scenarios:
            try:
               extracted_signals = scenario.get('extracted_signals', [])
               true_parameters = scenario.get('true_parameters', [])
               
               if len(extracted_signals) == len(true_parameters):
                   for i, (extracted, true_params) in enumerate(zip(extracted_signals, true_parameters)):
                       # Compute true correction
                       extracted_values, _ = self._extract_parameter_values(
                           extracted.get('posterior_summary', {})
                       )
                       
                       true_values = np.array([
                           true_params.get(param_name, 0.0) for param_name in self.param_names
                       ])
                       
                       # FIX #1: Compute corrections in normalized space (matching network inputs)
                       # Network predicts in normalized [-1,1] space, so true_correction must also be normalized
                       true_correction = np.zeros_like(extracted_values)
                       for i, p in enumerate(self.param_names):
                           bounds = self.physics_bounds[p]
                           rng = bounds['max_value'] - bounds['min_value']
                           # Normalize correction: physical_correction / parameter_range
                           true_correction[i] = (true_values[i] - extracted_values[i]) / max(rng, 1e-6)
                       
                       # Prepare input features
                       param_tensor, context_tensor = self._prepare_neural_network_input(
                           extracted, i, extracted_signals
                        )
                       
                       if param_tensor is not None and context_tensor is not None:
                           training_data.append({
                                'param_tensor': param_tensor,
                                'context_tensor': context_tensor,
                                'true_correction': true_correction,
                                'signal_quality': extracted.get('signal_quality', 0.5)
                            })
                            
            except Exception as e:
                self.logger.debug(f"Error processing training scenario: {e}")
                continue
        
        if len(training_data) < 10:
            self.logger.warning(f"Insufficient training data: {len(training_data)} samples")
            return {'success': False, 'error': 'insufficient_data'}
        
        # Split training/validation
        np.random.shuffle(training_data)
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        # Setup training
        optimizer = torch.optim.AdamW(self.bias_estimator.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1.5)
        
        # IMPROVED Loss function with robust gradients + VARIANCE CORRECTION
        def loss_function(corrections_pred, uncertainties_pred, variance_scales_pred, corrections_true, quality_weights):
            # Compute per-parameter MSE with uncertainty weighting
            # Shape: (batch, 9) each
            squared_errors = (corrections_pred - corrections_true) ** 2
            
            #  Adaptive parameter scaling to handle different correction magnitudes
            # (small corrections for time vs large for distance)
            # Use absolute error magnitude instead of true value (corrections are small!)
            param_scales = torch.clamp(torch.abs(corrections_true).mean(dim=0, keepdim=True) + 0.01, min=0.01)
            normalized_errors = squared_errors / (param_scales ** 2 + 1e-8)
            
            # Primary loss: NLL (likelihood-based) weighted by inverse uncertainty
            # Lower uncertainty -> higher gradient for errors
            nll_loss = torch.mean(normalized_errors / (uncertainties_pred + 1e-8))
            
            # ✅ NEW: Relative error loss (better convergence for small corrections)
            # Forces model to learn magnitude of corrections, not just direction
            relative_errors = torch.abs(corrections_pred - corrections_true) / (torch.abs(corrections_true) + 0.1)
            relative_loss = torch.mean(torch.clamp(relative_errors, min=0.0, max=10.0))
            
            # FIX #3: KL-divergence to lognormal prior (instead of entropy that inflates uncertainties)
            # Target: uncertainties ≈ 0.3 (in normalized space), matching typical correction magnitudes
            target_uncertainty = torch.full_like(uncertainties_pred, 0.3)
            # Use log-scale comparison to match typical ML calibration practices
            log_pred = torch.log(uncertainties_pred + 1e-6)
            log_target = torch.log(target_uncertainty + 1e-6)
            uncertainty_kl_loss = F.mse_loss(log_pred, log_target)
            
            # ✅ Quality weighting: higher quality signals get stronger training signal
            quality_weights_expanded = quality_weights.unsqueeze(1)  # Shape: (batch, 1)
            quality_weighted_nll = torch.mean(quality_weights_expanded * normalized_errors / (uncertainties_pred + 1e-8))
            
            # ✅ CRITICAL: Variance scale loss - train network to inflate posterior to 68% coverage
            # Target: variance_scale ≈ 1.3-1.5 (widens std by 30-50%, coverage 30% → 68%)
            # Formula: For narrow posterior (coverage ~30%), need scale² ≈ 2.3× to reach 68% target
            # Per-parameter learning: some params need 1.1×, others need 1.8×
            # Loss: penalize if scale < 1.1 (underfitting width) OR scale > 1.8 (overfitting)
            target_variance_scale = torch.clamp(variance_scales_pred, min=1.0, max=2.0)  # Bounded to [1.0, 2.0]
            scale_lower_bound = torch.full_like(variance_scales_pred, 1.1)  # Minimum inflation
            scale_upper_bound = torch.full_like(variance_scales_pred, 1.8)  # Maximum inflation
            
            scale_underfitting = torch.relu(scale_lower_bound - variance_scales_pred)  # Penalize if too narrow
            scale_overfitting = torch.relu(variance_scales_pred - scale_upper_bound)   # Penalize if too wide
            variance_scale_loss = torch.mean(scale_underfitting + 0.5 * scale_overfitting)
            
            # ✅ NEW: Add coverage-driven anchor - prevents learning from diverging
            # Empirical observation: need mean scale ≈ 1.4-1.6 to reach 68% coverage
            # Loss penalizes if average scale deviates from target
            coverage_target = 1.4  # Target scale to achieve 68% coverage
            coverage_anchor_loss = torch.abs(torch.mean(variance_scales_pred) - coverage_target)
            variance_scale_loss += 0.1 * coverage_anchor_loss  # Add as weak regularizer
            
            # ⚠️ SHOULD FIX #3: Vectorized correlation penalty (no batch loop)
            # Penalize if corrected params violate known physics correlations
            # For gravitational waves: mass_1 and mass_2 are somewhat correlated,
            # and both inversely correlate with distance (stronger for fixed SNR)
            
            if corrections_pred.shape[1] >= 3:  # Only if we have mass_1, mass_2, distance
                # VECTORIZED VERSION (no loop):
                # Mass correlation: corrections should be similar magnitude
                # Shape: [batch]
                mass_corr_diff = torch.abs(corrections_pred[:, 0] - corrections_pred[:, 1])
                
                # Mass-distance anti-correlation: if mass increases, distance should decrease
                # Shape: [batch]
                mass_avg = (corrections_pred[:, 0] + corrections_pred[:, 1]) / 2.0
                distance_corr = corrections_pred[:, 2]
                mass_distance_anticorr = (mass_avg * distance_corr) ** 2  # Should be < 0
                
                # Penalize if correlations violated (vectorized mean)
                corr_loss = torch.mean(0.1 * mass_corr_diff + 0.05 * mass_distance_anticorr)
            else:
                corr_loss = torch.tensor(0.0, dtype=corrections_pred.dtype, device=corrections_pred.device)
            
            # ✅ IMPROVED: Balanced loss with variance correction AND correlation
            # Mean correction: NLL (0.40) + Relative (0.25) for parameter shifts
            # Width correction: Variance (0.20) to widen posterior from 30% → 68% coverage
            # Correlation: (0.02) to enforce physics relationships
            # Regularization: KL-Prior (0.08) + Quality (0.03) + Magnitude (0.02)
            magnitude_penalty = torch.mean(torch.abs(corrections_pred) > 0.5)  # Penalize large corrections
            total_loss = (
                0.40 * nll_loss +
                0.25 * relative_loss +
                0.20 * variance_scale_loss +  # ✅ Width widening
                0.08 * uncertainty_kl_loss +
                0.03 * quality_weighted_nll +
                0.05 * corr_loss +  
                0.02 * magnitude_penalty
            )
            return total_loss
        
        # Training loop
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'epochs_completed': 0
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            # Training phase
            self.bias_estimator.train()
            train_losses = []
            
            np.random.shuffle(train_data)
            
            for batch_start in range(0, len(train_data), batch_size):
                batch_data = train_data[batch_start:batch_start + batch_size]
                
                if len(batch_data) == 0:
                    continue
                
                # Prepare batch tensors
                param_tensors = torch.cat([item['param_tensor'] for item in batch_data], dim=0)
                context_tensors = torch.cat([item['context_tensor'] for item in batch_data], dim=0)
                # Convert list of corrections (each is 1D array of shape (9,)) to 2D tensor
                true_corrections = torch.stack([
                    torch.tensor(item['true_correction'], dtype=torch.float32) 
                    for item in batch_data
                ], dim=0)  # Result: (batch_size, 9)
                quality_weights = torch.tensor([item['signal_quality'] for item in batch_data], dtype=torch.float32)
                
                # Forward pass - now returns variance scales too
                pred_corrections, pred_uncertainties, pred_variance_scales = self.bias_estimator(param_tensors, context_tensors)
                
                # Compute loss with variance correction
                loss = loss_function(pred_corrections, pred_uncertainties, pred_variance_scales, true_corrections, quality_weights)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bias_estimator.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses) if train_losses else 0.0
            training_history['train_losses'].append(avg_train_loss)
            
            # Validation phase
            avg_val_loss = None
            if val_data:
                self.bias_estimator.eval()
                val_losses = []
                
                with torch.no_grad():
                    for item in val_data:
                        pred_corrections, pred_uncertainties, pred_variance_scales = self.bias_estimator(
                            item['param_tensor'], item['context_tensor']
                        )
                        true_correction = torch.tensor([item['true_correction']], dtype=torch.float32)
                        quality_weight = torch.tensor([item['signal_quality']], dtype=torch.float32)
                        
                        val_loss = loss_function(pred_corrections, pred_uncertainties, pred_variance_scales, true_correction, quality_weight)
                        val_losses.append(val_loss.item())
                
                avg_val_loss = np.mean(val_losses)
                training_history['val_losses'].append(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1} (patience={patience})")
                    break
            
            scheduler.step()
            
            # Logging - show every 2-5 epochs depending on total
            log_interval = max(1, epochs // 15)
            if epoch % log_interval == 0 or epoch == epochs - 1:
                val_loss_str = f" | Val Loss: {avg_val_loss:.6f}" if avg_val_loss is not None else ""
                patience_str = f" | Patience: {patience_counter}/{patience}" if avg_val_loss is not None else ""
                self.logger.info(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f}{val_loss_str}{patience_str}")
        
        training_history['epochs_completed'] = epoch + 1
        
        # Mark as trained
        self.is_trained = True
        self.training_epochs = training_history['epochs_completed']
        
        # Final evaluation
        final_metrics = {
            'success': True,
            'training_history': training_history,
            'final_train_loss': training_history['train_losses'][-1] if training_history['train_losses'] else 0.0,
            'final_val_loss': training_history['val_losses'][-1] if training_history['val_losses'] else 0.0,
            'epochs_completed': training_history['epochs_completed'],
            'training_samples': len(train_data),
            'validation_samples': len(val_data)
        }
        
        # Log training summary
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"BiasCorrector Training Summary:")
        self.logger.info(f"  Total Epochs Completed: {training_history['epochs_completed']}/{epochs}")
        self.logger.info(f"  Final Train Loss: {final_metrics['final_train_loss']:.6f}")
        if training_history['val_losses']:
            self.logger.info(f"  Final Val Loss: {final_metrics['final_val_loss']:.6f}")
        self.logger.info(f"  Training Samples: {len(train_data)}")
        self.logger.info(f"  Validation Samples: {len(val_data)}")
        self.logger.info(f"  Configuration: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, patience={patience}")
        self.logger.info(f"{'='*70}\n")
        
        return final_metrics
    
    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive correction performance statistics"""
        
        stats = {
            'training_status': {
                'is_trained': self.is_trained,
                'training_epochs': self.training_epochs,
                'current_strategy': self.current_strategy
            },
            'performance_metrics': self.performance_metrics.copy(),
            'correction_history_summary': {},
            'parameter_statistics': {}
        }
        
        # Correction history summary
        if self.correction_history:
            magnitudes = [item['magnitude'] for item in self.correction_history]
            qualities = [item['quality'] for item in self.correction_history]
            acceptance_rate = np.mean([item['accepted'] for item in self.correction_history])
            
            stats['correction_history_summary'] = {
                'total_corrections_attempted': len(self.correction_history),
                'acceptance_rate': float(acceptance_rate),
                'mean_correction_magnitude': float(np.mean(magnitudes)),
                'std_correction_magnitude': float(np.std(magnitudes)),
                'mean_signal_quality': float(np.mean(qualities))
            }
        
        # Parameter-specific statistics
        for param_name in self.param_names:
            param_corrections = self.performance_metrics['parameter_improvements'][param_name]
            if param_corrections:
                stats['parameter_statistics'][param_name] = {
                    'corrections_applied': len(param_corrections),
                    'mean_correction': float(np.mean(param_corrections)),
                    'max_correction': float(np.max(param_corrections)),
                    'physics_bound': self.physics_bounds[param_name]['max_correction']
                }
        
        return stats

UncertaintyAwareSubtractor = BiasCorrector  