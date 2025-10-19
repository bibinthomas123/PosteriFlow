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

class BiasEstimator(nn.Module):
    """ advanced bias estimator with transformer architecture"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.context_dim = 16  # Extended context features
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64] if input_dim <= 9 else [512, 256, 128, 64]
        
        #  Multi-scale feature extraction
        self.param_embedding = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        self.context_embedding = nn.Sequential(
            nn.Linear(self.context_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        #  Transformer encoder for parameter correlations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=96,  # 64 + 32
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        #  Parameter-specific correction heads with physics constraints
        self.mass_corrector = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # mass_1, mass_2
            nn.Tanh()
        )
        
        self.distance_corrector = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.LayerNorm(48),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),  # luminosity_distance
            nn.Tanh()
        )
        
        self.time_corrector = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # geocent_time
            nn.Tanh()
        )
        
        self.sky_corrector = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.LayerNorm(48),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 2),  # ra, dec
            nn.Tanh()
        )
        
        #  Additional parameters corrector
        remaining_params = max(0, input_dim - 6)
        if remaining_params > 0:
            self.extra_corrector = nn.Sequential(
                nn.Linear(96, max(64, remaining_params * 8)),
                nn.ReLU(),
                nn.LayerNorm(max(64, remaining_params * 8)),
                nn.Linear(max(64, remaining_params * 8), max(32, remaining_params * 4)),
                nn.ReLU(),
                nn.Linear(max(32, remaining_params * 4), remaining_params),
                nn.Tanh()
            )
        else:
            self.extra_corrector = None
        
        #  Physics-based scaling parameters (learned)
        self.mass_scale = nn.Parameter(torch.tensor(0.08))        # 8% max mass correction
        self.distance_scale = nn.Parameter(torch.tensor(0.25))    # 25% max distance correction
        self.time_scale = nn.Parameter(torch.tensor(0.001))       # 1ms max time correction
        self.sky_scale = nn.Parameter(torch.tensor(0.15))         # 15% max sky correction
        self.extra_scale = nn.Parameter(torch.tensor(0.12))       # 12% max other corrections
        
        #  Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Softplus()  # Ensures positive uncertainties
        )
        
    def forward(self, param_estimates: torch.Tensor, context_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ forward pass with uncertainty quantification"""
        
        batch_size = param_estimates.shape[0]
        
        # Embed parameters and context
        param_embed = self.param_embedding(param_estimates)
        context_embed = self.context_embedding(context_features)
        
        # Combine embeddings
        combined_embed = torch.cat([param_embed, context_embed], dim=1)
        combined_embed = combined_embed.unsqueeze(1)  # Add sequence dimension
        
        # Apply transformer
        transformer_out = self.transformer(combined_embed)
        features = transformer_out.squeeze(1)
        
        # Generate parameter-specific corrections
        corrections = []
        
        # Mass corrections (parameters 0, 1)
        mass_corr = self.mass_corrector(features) * self.mass_scale
        corrections.append(mass_corr)
        
        # Distance correction (parameter 2)
        dist_corr = self.distance_corrector(features) * self.distance_scale
        corrections.append(dist_corr)
        
        # Time correction (parameter 5 if present)
        if self.input_dim > 5:
            time_corr = self.time_corrector(features) * self.time_scale
            corrections.append(time_corr)
        
        # Sky corrections (parameters 3, 4)
        if self.input_dim > 3:
            sky_corr = self.sky_corrector(features) * self.sky_scale
            corrections.append(sky_corr)
        
        # Extra parameter corrections
        if self.extra_corrector is not None and self.input_dim > 6:
            extra_corr = self.extra_corrector(features) * self.extra_scale
            corrections.append(extra_corr)
        
        # Concatenate all corrections
        all_corrections = torch.cat(corrections, dim=1)
        
        # Ensure correct output dimension
        if all_corrections.shape[1] > self.input_dim:
            all_corrections = all_corrections[:, :self.input_dim]
        elif all_corrections.shape[1] < self.input_dim:
            padding = torch.zeros(batch_size, self.input_dim - all_corrections.shape[1])
            all_corrections = torch.cat([all_corrections, padding], dim=1)
        
        # Estimate uncertainties
        uncertainties = self.uncertainty_head(features)
        
        return all_corrections, uncertainties


class BiasCorrector:
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
    
    def __init__(self, param_names: List[str]):
        self.param_names = param_names
        self.n_params = len(param_names)
        
        #  Initialize advanced neural bias estimator
        self.bias_estimator = BiasEstimator(self.n_params)
        
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
    
    
    def forward(self, params: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for integration with OverlapNeuralPE.
        
        Args:
            params: [batch, param_dim] normalized parameters
            context: [batch, context_dim] context features
            
        Returns:
            corrections: [batch, param_dim] bias corrections
            uncertainties: [batch, param_dim] correction uncertainties
            confidence: [batch] correction confidence scores
        """
        # Concatenate params and context
        combined = torch.cat([params, context], dim=1)
        
        # Get bias estimates
        bias_pred, uncertainty, confidence = self.bias_estimator(combined)
        
        # Apply learned scaling
        corrections = bias_pred * self.correction_scales
        
        # Clamp to reasonable range
        corrections = torch.clamp(corrections, -0.2, 0.2)  # Max 20% correction
        
        return corrections, uncertainty, confidence
    
    
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
            elif param_name in ['a_1', 'a_2']:
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
            (['a_1', 'a_2'], 0.2),                 # Spin correlation
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
            
            # Create comprehensive context features
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
                               context_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Apply neural network bias correction with uncertainty quantification"""
        
        with torch.no_grad():
            corrections, uncertainties = self.bias_estimator(param_tensor, context_tensor)
            corrections_np = corrections.squeeze().numpy()
            uncertainties_np = uncertainties.squeeze().numpy()
        
        return corrections_np, uncertainties_np
    
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
                    f"{param_name} correction violates physics bounds"
                )
            
            # Check correction magnitude
            if param_name in ['ra', 'dec', 'geocent_time']:
                max_correction = bounds['max_correction']
            else:
                max_correction = bounds['max_correction'] * abs(original_values[i])
            
            if abs(corrections[i]) > max_correction:
                validation_results['magnitude_acceptable'] = False
                validation_results['warnings'].append(
                    f"{param_name} correction magnitude too large: {abs(corrections[i]):.4f} > {max_correction:.4f}"
                )
        
        # Parameter correlation validation
        expected_correlations = self.correlation_matrix
        correction_correlations = np.corrcoef(corrections.reshape(1, -1), corrections.reshape(1, -1))
        
        if corrections.shape[0] > 1:  # Need multiple parameters for correlation
            correlation_deviation = np.mean(np.abs(correction_correlations - expected_correlations))
            if correlation_deviation > 0.5:
                validation_results['correlation_consistent'] = False
                validation_results['warnings'].append(
                    f"Correction correlations deviate significantly from expected: {correlation_deviation:.3f}"
                )
        
        # Uncertainty reasonableness
        for i, param_name in enumerate(self.param_names):
            if correction_uncertainties[i] > abs(corrections[i]) * 5:  # Uncertainty > 5x correction
                validation_results['uncertainty_reasonable'] = False
                validation_results['warnings'].append(
                    f"{param_name} correction uncertainty suspiciously large"
                )
        
        # Overall validation
        validation_results['overall_valid'] = (
            validation_results['physics_valid'] and 
            validation_results['magnitude_acceptable'] and
            validation_results['correlation_consistent']
        )
        
        return validation_results
    
    def _apply_corrections_to_posterior(self, original_summary: Dict, corrections: np.ndarray,
                                      correction_uncertainties: np.ndarray) -> Dict:
        """Apply validated corrections to posterior summary"""
        
        corrected_summary = {}
        
        for key, value in original_summary.items():
            if key in self.param_names:
                param_idx = self.param_names.index(key)
                correction = corrections[param_idx]
                correction_unc = correction_uncertainties[param_idx]
                
                if isinstance(value, dict):
                    corrected_value = value.copy()
                    
                    # Apply correction to central values
                    if 'median' in corrected_value:
                        corrected_value['median'] += correction
                    if 'mean' in corrected_value:
                        corrected_value['mean'] += correction
                    
                    # Update uncertainty (quadrature sum)
                    if 'std' in corrected_value:
                        original_std = corrected_value['std']
                        corrected_value['std'] = np.sqrt(original_std**2 + correction_unc**2)
                    
                    # Update quantiles if present
                    if 'quantiles' in corrected_value:
                        corrected_value['quantiles'] = [q + correction for q in corrected_value['quantiles']]
                    
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
                        corrections, correction_uncertainties = self._apply_neural_correction(
                            param_tensor, context_tensor
                        )
                        correction_method = 'neural_network'
                    else:
                        corrections, correction_uncertainties = self._apply_physics_based_correction(
                            original_values, original_uncertainties, i, len(extracted_signals)
                        )
                        correction_method = 'physics_based'
                    
                    # Apply strategy scaling
                    corrections *= strategy['scaling']
                    
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
                            signal['posterior_summary'], corrections, correction_uncertainties
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
        
        return corrected_signals
    
    def train_bias_estimator(self, training_scenarios: List[Dict], epochs: int = 200, 
                           validation_split: float = 0.2) -> Dict[str, Any]:
        """ training of neural bias estimator"""
        
        if not training_scenarios:
            self.logger.warning("No training scenarios provided")
            return {'success': False, 'error': 'no_training_data'}
        
        self.logger.info(f"Training bias estimator on {len(training_scenarios)} scenarios for {epochs} epochs")
        
        # Prepare training data
        training_data = []
        for scenario in training_scenarios:
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
                        
                        true_correction = true_values - extracted_values
                        
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
        optimizer = torch.optim.AdamW(self.bias_estimator.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Loss function with uncertainty
        def loss_function(corrections_pred, uncertainties_pred, corrections_true, quality_weights):
            # Negative log-likelihood assuming Gaussian corrections
            mse_loss = torch.mean(((corrections_pred - corrections_true) ** 2) / (uncertainties_pred + 1e-8))
            uncertainty_reg = torch.mean(torch.log(uncertainties_pred + 1e-8))  # Encourage reasonable uncertainties
            quality_weighting = torch.mean(quality_weights * ((corrections_pred - corrections_true) ** 2))
            
            return mse_loss + 0.1 * uncertainty_reg + 0.3 * quality_weighting
        
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
            
            for batch_start in range(0, len(train_data), 32):  # Batch size 32
                batch_data = train_data[batch_start:batch_start + 32]
                
                if len(batch_data) == 0:
                    continue
                
                # Prepare batch tensors
                param_tensors = torch.cat([item['param_tensor'] for item in batch_data], dim=0)
                context_tensors = torch.cat([item['context_tensor'] for item in batch_data], dim=0)
                true_corrections = torch.tensor([item['true_correction'] for item in batch_data], dtype=torch.float32)
                quality_weights = torch.tensor([item['signal_quality'] for item in batch_data], dtype=torch.float32)
                
                # Forward pass
                pred_corrections, pred_uncertainties = self.bias_estimator(param_tensors, context_tensors)
                
                # Compute loss
                loss = loss_function(pred_corrections, pred_uncertainties, true_corrections, quality_weights)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bias_estimator.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses) if train_losses else 0.0
            training_history['train_losses'].append(avg_train_loss)
            
            # Validation phase
            if val_data:
                self.bias_estimator.eval()
                val_losses = []
                
                with torch.no_grad():
                    for item in val_data:
                        pred_corrections, pred_uncertainties = self.bias_estimator(
                            item['param_tensor'], item['context_tensor']
                        )
                        true_correction = torch.tensor([item['true_correction']], dtype=torch.float32)
                        quality_weight = torch.tensor([item['signal_quality']], dtype=torch.float32)
                        
                        val_loss = loss_function(pred_corrections, pred_uncertainties, true_correction, quality_weight)
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
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            scheduler.step()
            
            # Logging
            if epoch % 20 == 0 or epoch == epochs - 1:
                val_loss_str = f", Val Loss: {avg_val_loss:.6f}" if val_data else ""
                self.logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}{val_loss_str}")
        
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
        
        self.logger.info(f" Bias estimator training completed successfully in {training_history['epochs_completed']} epochs")
        
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