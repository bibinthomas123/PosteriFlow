"""
overlap_neuralpe.py - Unified Best-in-Class Neural Parameter Estimation
Combines OverlapNeuralPE + BestInClassNeuralPE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

from ahsd.core.priority_net import PriorityNet
from ahsd.models.rl_controller import AdaptiveComplexityController
from ahsd.core.bias_corrector import BiasCorrector
from ahsd.core.adaptive_subtractor import AdaptiveSubtractor
from ahsd.models.flows import ConditionalRealNVP


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
    
    
    def __init__(self, param_names: List[str], priority_net_path: str, 
                 config: Dict[str, Any], device: str = 'cuda'):
        super().__init__()
        
        self.param_names = param_names
        self.param_dim = len(param_names)
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.context_dim = config.get('context_dim', 256)
        self.n_flow_layers = config.get('n_flow_layers', 8)
        self.max_iterations = config.get('max_iterations', 5)
        
        # Dropout configuration
        self.dropout_rate = config.get('dropout', 0.1)
        flow_config = config.get('flow_config', {})
        self.flow_dropout = flow_config.get('dropout', 0.15)
        self.flow_hidden_features = flow_config.get('hidden_features', 128)
        self.flow_num_blocks = flow_config.get('num_blocks_per_layer', 2)
        
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
            nn.Softplus()  # Ensure positive uncertainties
        )
        
        # Performance tracking
        self.performance_tracker = {
            'training_losses': deque(maxlen=1000),
            'validation_metrics': deque(maxlen=100),
            'complexity_history': deque(maxlen=1000),
            'inference_times': deque(maxlen=1000),
            'rl_rewards': deque(maxlen=1000)
        }
        
        self.training_step = 0
        
        self.to(self.device)
        total_params = sum(p.numel() for p in self.parameters())
        self.logger.info(f"✅ Unified OverlapNeuralPE initialized with {total_params:,} parameters")
        self.logger.info(f"   Context dim: {self.context_dim}")
        self.logger.info(f"   Flow layers: {self.n_flow_layers}")
        self.logger.info(f"   Dropout: {self.dropout_rate}, Flow dropout: {self.flow_dropout}")
    
    def _get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for normalization."""
        bounds = {
            'mass_1': (1.0, 100.0),
            'mass_2': (1.0, 100.0),
            'luminosity_distance': (20.0, 8000.0),
            'geocent_time': (-0.1, 0.1),
            'ra': (0.0, 2*np.pi),
            'dec': (-np.pi/2, np.pi/2),
            'theta_jn': (0.0, np.pi),
            'psi': (0.0, np.pi),
            'phase': (0.0, 2*np.pi)
        }
        return {param: bounds.get(param, (0.0, 1.0)) for param in self.param_names}
    
    def _build_physics_priors(self) -> Dict[str, torch.distributions.Distribution]:
        """Build physics-informed priors."""
        priors = {}
        
        for param in self.param_names:
            if 'mass' in param:
                # Power-law priors for masses (Salpeter IMF)
                priors[param] = torch.distributions.Pareto(1.0, 2.35)
            elif param in ['ra', 'phase']:
                # Uniform priors for phase-like angles
                priors[param] = torch.distributions.Uniform(0.0, 2*np.pi)
            elif param in ['dec', 'theta_jn']:
                # Sine priors for spherical angles
                priors[param] = torch.distributions.Beta(0.5, 0.5)
            elif param == 'luminosity_distance':
                # Volume prior for distance
                priors[param] = torch.distributions.Pareto(1.0, 2.0)
            else:
                # Default uniform prior
                priors[param] = torch.distributions.Uniform(0.0, 1.0)
        
        return priors
    
    def _normalize_parameters(self, params: torch.Tensor) -> torch.Tensor:
        """Normalize parameters to [-1, 1]."""
        normalized = torch.zeros_like(params)
        
        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_bounds[param_name]
            normalized[..., i] = 2 * (params[..., i] - min_val) / (max_val - min_val) - 1
            
        return torch.clamp(normalized, -1, 1)
    
    def _denormalize_parameters(self, normalized_params: torch.Tensor) -> torch.Tensor:
        """Denormalize parameters from [-1, 1] to physical units."""
        params = torch.zeros_like(normalized_params)
        
        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_bounds[param_name]
            params[..., i] = (normalized_params[..., i] + 1) / 2 * (max_val - min_val) + min_val
            
        return params
    
    def _init_components(self, priority_net_path: str):
        """Initialize all pipeline components."""
        
        # 1. PriorityNet (pre-trained, frozen)
        self.priority_net = PriorityNet(use_strain=True)
        checkpoint = torch.load(priority_net_path, map_location=self.device)
        self.priority_net.load_state_dict(checkpoint['model_state_dict'])
        self.priority_net.eval()
        for param in self.priority_net.parameters():
            param.requires_grad = False
        self.logger.info(f"✅ Loaded PriorityNet from {priority_net_path}")
        
        # 2. Context Encoder
        self.context_encoder = ContextEncoder(
            n_detectors=2,
            hidden_dim=self.context_dim,
            dropout=self.dropout_rate
        )
        
        # 3. Normalizing Flow
        self.flow = ConditionalRealNVP(
            features=self.param_dim,
            context_features=self.context_dim,
            hidden_features=self.flow_hidden_features,
            max_layers=self.n_flow_layers,
            num_blocks_per_layer=self.flow_num_blocks,
            dropout=self.flow_dropout
        )
        
        self.logger.info(f"✅ Flow model initialized: {self.n_flow_layers} layers")
        
        # 4. RL Controller
        rl_config = self.config.get('rl_controller', {})
        
        self.rl_controller = AdaptiveComplexityController(
            state_features=rl_config.get('state_features', [
                'remaining_signals', 'residual_power', 
                'current_snr', 'extraction_success_rate'
            ]),
            complexity_levels=rl_config.get('complexity_levels', ['low', 'medium', 'high']),
            learning_rate=rl_config.get('learning_rate', 1e-3),
            epsilon=rl_config.get('epsilon', 0.1),
            epsilon_decay=rl_config.get('epsilon_decay', 0.995),
            memory_size=rl_config.get('memory_size', 10000),
            batch_size=rl_config.get('batch_size', 32)
        )
        
        self.complexity_configs = rl_config.get('complexity_configs', {
            'low': {'flow_layers': 4, 'inference_samples': 500},
            'medium': {'flow_layers': 8, 'inference_samples': 1000},
            'high': {'flow_layers': 12, 'inference_samples': 2000}
        })
        
        # 5. Bias Corrector
        bias_cfg = self.config.get('bias_corrector', {})
        if bias_cfg.get('enabled', True):
            self.bias_corrector = BiasCorrector(param_names=self.param_names)
            self.logger.info("✅ BiasCorrector initialized")
        else:
            self.bias_corrector = None
            self.logger.info("⚠️  BiasCorrector disabled")
        
        # 6. Adaptive Subtractor
        self.adaptive_subtractor = AdaptiveSubtractor()
        self.logger.info("✅ AdaptiveSubtractor initialized")
    
    def sample_posterior(self, strain_data: torch.Tensor, 
                        n_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """
        Sample from learned posterior distribution.
        
        Args:
            strain_data: [batch, n_det, n_samples] whitened strain data
            n_samples: Number of posterior samples to draw
            
        Returns:
            dict containing:
                'samples': [batch, n_samples, param_dim] posterior samples
                'means': [batch, param_dim] posterior means
                'stds': [batch, param_dim] posterior standard deviations
                'uncertainties': [batch, param_dim] estimated uncertainties
        """
        self.eval()
        batch_size = strain_data.size(0)
        
        with torch.no_grad():
            # Extract context from strain
            context = self.context_encoder(strain_data)
            
            # Sample from base distribution (standard normal)
            z = torch.randn(batch_size, n_samples, self.param_dim, device=self.device)
            
            # Transform through flow (inverse direction for sampling)
            samples_list = []
            for i in range(batch_size):
                context_i = context[i:i+1].expand(n_samples, -1)
                z_i = z[i]
                
                # Inverse flow transformation
                samples_norm_i, _ = self.flow.inverse(z_i, context_i)
                
                # Apply bias correction
                if self.bias_corrector is not None:
                    corrections, _, _ = self.bias_corrector(samples_norm_i, context_i)
                    samples_norm_i = samples_norm_i + corrections
                
                # Denormalize to physical units
                samples_physical = self._denormalize_parameters(samples_norm_i)
                samples_list.append(samples_physical)
            
            samples = torch.stack(samples_list, dim=0)  # [batch, n_samples, param_dim]
            
            # Compute summary statistics
            means = samples.mean(dim=1)
            stds = samples.std(dim=1)
            
            # Estimate uncertainties
            uncertainties = self.uncertainty_estimator(
                torch.cat([self._normalize_parameters(means), context], dim=1)
            )
            
            return {
                'samples': samples,
                'means': means,
                'stds': stds,
                'uncertainties': uncertainties,
                'context': context
            }
    
    def extract_single_signal(self, strain_data: torch.Tensor, 
                             complexity: str = 'medium') -> Dict[str, Any]:
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
        n_samples = complexity_settings.get('inference_samples', 1000)
        
        result = self.sample_posterior(strain_data, n_samples=n_samples)
        
        return {
            'means': result['means'],
            'stds': result['stds'],
            'samples': result['samples'],
            'uncertainties': result['uncertainties'],
            'context': result['context']
        }
    
    def extract_overlapping_signals(self, strain_data: torch.Tensor,
                                   true_params: Optional[List[Dict]] = None,
                                   training: bool = False) -> Dict[str, Any]:
        """
        Extract all overlapping signals iteratively.
        
        Args:
            strain_data: [batch, n_det, n_samples] strain data
            true_params: Optional ground truth parameters for training
            training: Whether in training mode (for RL)
            
        Returns:
            dict with all extracted signals and final residual
        """
        batch_size = strain_data.size(0)
        
        all_extracted = []
        residual_data = strain_data.clone()
        
        pipeline_state = {
            'remaining_signals': self.max_iterations,
            'residual_power': 1.0,
            'current_snr': 0.0,
            'extraction_success_rate': 1.0
        }
        
        for iteration in range(self.max_iterations):
            # Get priorities from PriorityNet
            with torch.no_grad():
                detections = self._residual_to_detections(residual_data)
                priorities, _ = self.priority_net(detections)
            
            # Select complexity level via RL
            complexity = self.rl_controller.get_complexity_level(
                pipeline_state,
                training=training
            )
            
            # Extract signal
            extraction_result = self.extract_single_signal(residual_data, complexity)
            
            all_extracted.append({
                'params': extraction_result['means'],
                'uncertainties': extraction_result['stds'],
                'priority': priorities,
                'iteration': iteration,
                'complexity': complexity
            })
            
            # Subtract extracted signal
            estimated_params_dict = self._tensor_to_param_dict(extraction_result['means'])
            subtraction_result = self.adaptive_subtractor.subtract(
                residual_data.cpu().numpy(),
                estimated_params_dict
            )
            
            residual_data = torch.tensor(
                subtraction_result['residual'],
                dtype=torch.float32,
                device=self.device
            )
            
            # Update pipeline state
            pipeline_state['remaining_signals'] -= 1
            pipeline_state['residual_power'] = float(torch.mean(residual_data ** 2))
            
            # Early stopping if residual too low
            if pipeline_state['residual_power'] < 0.01:
                self.logger.info(f"Stopping at iteration {iteration+1}: low residual power")
                break
            
            # RL training if enabled
            if training and true_params is not None:
                reward = self._compute_extraction_reward(
                    extraction_result['means'],
                    true_params[iteration] if iteration < len(true_params) else None
                )
                
                state_vector = self.rl_controller.get_state_vector(pipeline_state)
                action = self.rl_controller.complexity_levels.index(complexity)
                next_state_vector = self.rl_controller.get_state_vector(pipeline_state)
                done = (iteration == self.max_iterations - 1)
                
                self.rl_controller.store_experience(
                    state_vector, action, reward, next_state_vector, done
                )
                
                if len(self.rl_controller.memory) >= self.rl_controller.batch_size:
                    self.rl_controller.train_step()
        
        return {
            'extracted_signals': all_extracted,
            'final_residual': residual_data,
            'n_iterations': iteration + 1
        }
    
    def _residual_to_detections(self, residual: torch.Tensor) -> List[Dict]:
        """Convert residual strain to detection format for PriorityNet."""
        batch_size = residual.size(0)
        detections = []
        
        for i in range(batch_size):
            snr_proxy = float(torch.sqrt(torch.mean(residual[i] ** 2)))
            
            detection = {
                'network_snr': snr_proxy * 10.0,
                'match_filter_snr': snr_proxy * 10.0,
                'chi_squared': 1.0,
                'null_snr': 0.1
            }
            detections.append(detection)
        
        return detections
    
    def _tensor_to_param_dict(self, params_tensor: torch.Tensor) -> Dict[str, float]:
        """Convert parameter tensor to dictionary."""
        params_np = params_tensor.detach().cpu().numpy()
        
        if len(params_np.shape) > 1:
            params_np = params_np[0]
        
        return {name: float(params_np[i]) for i, name in enumerate(self.param_names)}
    
    def _compute_extraction_reward(self, estimated_params: torch.Tensor,
                                   true_params: Optional[Dict]) -> float:
        """Compute reward for RL training."""
        if true_params is None:
            return 0.0
        
        true_tensor = torch.tensor(
            [true_params.get(name, 0.0) for name in self.param_names],
            dtype=torch.float32,
            device=self.device
        )
        
        rel_error = torch.abs((estimated_params[0] - true_tensor) / (true_tensor + 1e-6))
        accuracy = 1.0 - torch.mean(rel_error).item()
        
        return max(0.0, accuracy)
    
    def compute_loss(self, strain_data: torch.Tensor, 
                    true_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive training loss.
        
        Args:
            strain_data: [batch, n_det, n_samples] strain data
            true_params: [batch, param_dim] true parameters
            
        Returns:
            dict with total loss and component losses
        """
        # Extract context
        context = self.context_encoder(strain_data)
        
        # Normalize parameters
        true_params_norm = self._normalize_parameters(true_params)
        
        # Flow loss (negative log-likelihood)
        log_prob = self.flow.log_prob(true_params_norm, context)
        flow_loss = -log_prob.mean()
        
        # Physics constraint loss
        physics_loss = self._compute_physics_loss(true_params)
        
        # Uncertainty regularization
        uncertainties = self.uncertainty_estimator(
            torch.cat([true_params_norm, context], dim=1)
        )
        uncertainty_loss = 0.01 * torch.mean(uncertainties)
        
        # Total loss
        total_loss = flow_loss + 0.1 * physics_loss + uncertainty_loss
        
        return {
            'total_loss': total_loss,
            'nll': flow_loss,
            'physics_loss': physics_loss,
            'uncertainty_loss': uncertainty_loss
        }
    
    def _compute_physics_loss(self, params: torch.Tensor) -> torch.Tensor:
        """Enforce physical constraints."""
        loss = torch.tensor(0.0, device=params.device)
        
        # Mass ordering constraint: m1 >= m2
        if 'mass_1' in self.param_names and 'mass_2' in self.param_names:
            m1_idx = self.param_names.index('mass_1')
            m2_idx = self.param_names.index('mass_2')
            mass_violation = F.relu(params[:, m2_idx] - params[:, m1_idx])
            loss += torch.mean(mass_violation**2)
        
        return loss
    
    def update_training_metrics(self, loss_dict: Dict[str, torch.Tensor],
                               processing_time: float, gradient_norm: float):
        """Update training metrics for monitoring and RL."""
        self.training_step += 1
        
        # Store metrics
        self.performance_tracker['training_losses'].append(loss_dict['total_loss'].item())
        self.performance_tracker['inference_times'].append(processing_time)
        
        # Log periodically
        if self.training_step % 100 == 0:
            recent_loss = np.mean(list(self.performance_tracker['training_losses'])[-10:])
            self.logger.debug(f"Step {self.training_step}: Loss={recent_loss:.4f}, " 
                            f"GradNorm={gradient_norm:.3f}")
    
    def save_model(self, filepath: str):
        """Save complete model state."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'rl_controller': self.rl_controller.q_network.state_dict(),
            'param_names': self.param_names,
            'param_bounds': self.param_bounds,
            'config': self.config,
            'training_step': self.training_step,
            'performance_tracker': {
                k: list(v) for k, v in self.performance_tracker.items()
            }
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load complete model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.rl_controller.q_network.load_state_dict(checkpoint['rl_controller'])
        self.training_step = checkpoint.get('training_step', 0)
        
        if 'performance_tracker' in checkpoint:
            for k, v in checkpoint['performance_tracker'].items():
                self.performance_tracker[k] = deque(v, maxlen=self.performance_tracker[k].maxlen)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'UnifiedOverlapNeuralPE',
            'parameter_names': self.param_names,
            'parameter_dimension': self.param_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'context_dim': self.context_dim,
            'flow_layers': self.n_flow_layers,
            'training_step': self.training_step,
            'recent_performance': {
                'avg_loss': np.mean(list(self.performance_tracker['training_losses'])[-10:]) 
                           if self.performance_tracker['training_losses'] else 0.0,
                'avg_inference_time': np.mean(list(self.performance_tracker['inference_times'])[-10:]) 
                                     if self.performance_tracker['inference_times'] else 0.0
            }
        }


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
    
    def forward(self, params: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            nn.Tanh()
        )
        
        self.shift_net = nn.Sequential(
            nn.Linear(self.split_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, param_dim - self.split_dim)
        )
    
    def forward(self, params: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transformation."""
        x1, x2 = params[:, :self.split_dim], params[:, self.split_dim:]
        
        combined = torch.cat([x1, context], dim=1)
        s = self.scale_net(combined)
        t = self.shift_net(combined)
        
        y2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=1)
        
        y = torch.cat([x1, y2], dim=1)
        
        return y, log_det
    
    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Inverse transformation."""
        y1, y2 = y[:, :self.split_dim], y[:, self.split_dim:]
        
        combined = torch.cat([y1, context], dim=1)
        s = self.scale_net(combined)
        t = self.shift_net(combined)
        
        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([y1, x2], dim=1)
        
        return x


class ContextEncoder(nn.Module):
    """Encodes multi-detector strain data into context vector."""
    
    def __init__(self, n_detectors: int = 2, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.n_detectors = n_detectors
        
        # Per-detector encoder
        self.detector_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),  # ✅ ADD dropout after each block
            nn.Conv1d(32, 64, kernel_size=32, stride=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),  # ✅ ADD
            nn.Conv1d(64, 128, kernel_size=16, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),  # ✅ ADD
            nn.AdaptiveAvgPool1d(64)
        )
        
        # Multi-detector fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 * 64 * n_detectors, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
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
            det_data = strain_data[:, i:i+1, :]  # (batch, 1, time)
            features = self.detector_encoder(det_data)  # (batch, 128, 64)
            detector_features.append(features)
        
        # Concatenate and fuse
        combined = torch.cat(detector_features, dim=1)  # (batch, 128*n_det, 64)
        combined = combined.flatten(1)  # (batch, 128*64*n_det)
        
        context = self.fusion(combined)  # (batch, hidden_dim)
        
        return context

