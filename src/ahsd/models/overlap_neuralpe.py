#!/usr/bin/env python3
"""
PosteriFlow: Neural Posterior Estimator for Overlapping Gravitational Waves

Complete integration of:
- PriorityNet (signal ranking)
- RLController (adaptive complexity)
- NeuralPE (parameter estimation)
- BiasCorrector (systematic error correction)
- AdaptiveSubtractor (iterative signal removal)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Import your existing components
from ahsd.core.priority_net import PriorityNet
from ahsd.models.rl_controller import AdaptiveComplexityController
from ahsd.core.bias_corrector import BiasCorrector
from ahsd.core.adaptive_subtractor import AdaptiveSubtractor
from ahsd.models.flows import ConditionalRealNVP


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


class OverlapNeuralPE(nn.Module):
    """
    Complete Neural PE system for overlapping gravitational waves.
    
    Integrates:
    - PriorityNet: Ranks signals by importance
    - RLController: Adapts model complexity
    - NeuralPE: Estimates parameters via normalizing flows
    - BiasCorrector: Corrects systematic biases
    - AdaptiveSubtractor: Removes extracted signals
    """
    
    def __init__(self, param_names, priority_net_path, config, device='cuda'):
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
        
        # ✅ Read dropout configs
        self.dropout_rate = config.get('dropout', 0.1)
        flow_config = config.get('flow_config', {})
        self.flow_dropout = flow_config.get('dropout', 0.15)  # ✅ Store as instance variable
        self.flow_hidden_features = flow_config.get('hidden_features', 128)
        self.flow_num_blocks = flow_config.get('num_blocks_per_layer', 2)
        
        # Log configuration
        self.logger.info(f"📋 Model Configuration:")
        self.logger.info(f"  Context dim: {self.context_dim}")
        self.logger.info(f"  Flow layers: {self.n_flow_layers}")
        self.logger.info(f"  Dropout: {self.dropout_rate}")
        self.logger.info(f"  Flow dropout: {self.flow_dropout}")
        
        # Parameter normalization
        self.param_ranges = {
            'mass_1': (1.0, 100.0),
            'mass_2': (1.0, 100.0),
            'luminosity_distance': (20.0, 8000.0),
            'ra': (0.0, 2*3.14159),
            'dec': (-1.5708, 1.5708),
            'theta_jn': (0.0, 3.14159),
            'psi': (0.0, 3.14159),
            'phase': (0.0, 2*3.14159),
            'geocent_time': (-0.1, 0.1)
        }
        
        # Register as buffers so they move to GPU with model
        self.register_buffer('param_means', torch.tensor([
            (self.param_ranges[name][0] + self.param_ranges[name][1]) / 2.0
            for name in param_names
        ], dtype=torch.float32))
        
        self.register_buffer('param_stds', torch.tensor([
            (self.param_ranges[name][1] - self.param_ranges[name][0]) / 4.0
            for name in param_names
        ], dtype=torch.float32))
        
        # Initialize components
        self._init_components(priority_net_path)
        
        self.to(self.device)
        self.logger.info(f"OverlapNeuralPE initialized with {sum(p.numel() for p in self.parameters()):,} parameters")

    def _init_components(self, priority_net_path: str):
        """Initialize all pipeline components."""
        
        # 1. PriorityNet (pre-trained)
        self.priority_net = PriorityNet(use_strain=False)
        checkpoint = torch.load(priority_net_path, map_location=self.device)
        self.priority_net.load_state_dict(checkpoint['model_state_dict'])
        self.priority_net.eval()
        for param in self.priority_net.parameters():
            param.requires_grad = False
        self.logger.info(f"✅ Loaded PriorityNet from {priority_net_path}")
        
        # 2. Context Encoder with dropout
        self.context_encoder = ContextEncoder(
            n_detectors=2,
            hidden_dim=self.context_dim,
            dropout=self.dropout_rate  # ✅ Use config dropout
        )
        
        # 3. Normalizing Flow with dropout
        # ✅ IMPORTANT: ConditionalRealNVP doesn't have dropout parameter in __init__
        # The dropout is applied inside the coupling networks automatically
        # We just need to make sure the coupling networks use it
        
        self.flow = ConditionalRealNVP(
        features=self.param_dim,
        context_features=self.context_dim,
        hidden_features=self.flow_hidden_features,
        max_layers=self.n_flow_layers,
        num_blocks_per_layer=self.flow_num_blocks,
        dropout=self.flow_dropout  # ✅ NOW PASSED!
    )
    
        self.logger.info(f"✅ Flow model initialized: {self.n_flow_layers} layers, dropout={self.flow_dropout}")
        
        # ✅ Set dropout in flow coupling networks
        # ConditionalRealNVP creates ContextAwareCouplingNet internally
        # We need to ensure dropout is applied there
        for transform in self.flow.all_transforms:
            if hasattr(transform, 'transform_net_create_fn'):
                # The dropout is already in ContextAwareCouplingNet (0.1 hardcoded)
                # If you want to use config dropout, you need to modify
                # the flow.py file to accept dropout parameter
                pass
        
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
        
        # Store complexity configs
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
            self.logger.info("⚠️ BiasCorrector disabled")
        
        # 6. Adaptive Subtractor
        self.adaptive_subtractor = AdaptiveSubtractor()
        self.logger.info("✅ AdaptiveSubtractor initialized")
    
    def extract_single_signal(self,
                             strain_data: torch.Tensor,
                             complexity: str = 'medium') -> Dict[str, Any]:
        """Extract parameters for a single signal."""
        context = self.context_encoder(strain_data)
        
        # Get number of samples from config
        complexity_settings = self.complexity_configs.get(complexity, {})
        n_samples = complexity_settings.get('inference_samples', 1000)
        
        # Sample from flow
        samples_norm = self.flow.sample(n_samples, context)
        
        # Denormalize
        samples = samples_norm * self.param_stds + self.param_means
        
        # Reshape if needed
        if samples.dim() == 2:
            samples = samples.unsqueeze(0)
        
        # Compute statistics
        means = samples.mean(dim=1)
        stds = samples.std(dim=1)
        
        # Apply bias correction
        if self.bias_corrector is not None:
            means = self.bias_corrector.correct(means, context)
        
        return {
            'means': means,
            'stds': stds,
            'samples': samples,
            'context': context
        }
    
    def extract_overlapping_signals(self,
                                   strain_data: torch.Tensor,
                                   true_params: Optional[List[Dict]] = None,
                                   training: bool = False) -> Dict[str, Any]:
        """Extract all overlapping signals iteratively."""
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
            with torch.no_grad():
                detections = self._residual_to_detections(residual_data)
                priorities, _ = self.priority_net(detections)
            
            complexity = self.rl_controller.get_complexity_level(
                pipeline_state,
                training=training
            )
            
            extraction_result = self.extract_single_signal(residual_data, complexity)
            
            all_extracted.append({
                'params': extraction_result['means'],
                'uncertainties': extraction_result['stds'],
                'priority': priorities,
                'iteration': iteration,
                'complexity': complexity
            })
            
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
            
            pipeline_state['remaining_signals'] -= 1
            pipeline_state['residual_power'] = float(torch.mean(residual_data ** 2))
            
            if pipeline_state['residual_power'] < 0.01:
                self.logger.info(f"Stopping at iteration {iteration+1}: residual power too low")
                break
            
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
    
    def _compute_extraction_reward(self,
                                   estimated_params: torch.Tensor,
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
    
    def compute_loss(self,
                    strain_data: torch.Tensor,
                    true_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute training loss."""
        context = self.context_encoder(strain_data)
        
        # Normalize parameters
        true_params_norm = (true_params - self.param_means) / self.param_stds
        
        # Use flow's log_prob method
        log_prob = self.flow.log_prob(true_params_norm, context)
        
        # Negative log-likelihood
        nll = -log_prob.mean()
        
        return {
            'total_loss': nll,
            'nll': nll
        }
    
    def save_model(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'rl_controller': self.rl_controller.q_network.state_dict(),
            'param_names': self.param_names,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.rl_controller.q_network.load_state_dict(checkpoint['rl_controller'])
        self.logger.info(f"Model loaded from {filepath}")


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


