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


class ContextEncoder(nn.Module):
    """Encodes multi-detector strain data into context vector."""
    
    def __init__(self, n_detectors: int = 2, hidden_dim: int = 256):
        super().__init__()
        
        self.n_detectors = n_detectors
        
        # Per-detector encoder
        self.detector_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=32, stride=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=16, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )
        
        # Multi-detector fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 * 64 * n_detectors, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
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
    
    def __init__(self, 
                 param_names: List[str],
                 priority_net_path: str,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
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
        
        # 2. Context Encoder
        self.context_encoder = ContextEncoder(
            n_detectors=2,
            hidden_dim=self.context_dim
        )
        
        # 3. Normalizing Flow
        self.flow = NormalizingFlow(
            param_dim=self.param_dim,
            context_dim=self.context_dim,
            n_layers=self.n_flow_layers
        )
        
        # 4. RL Controller
        rl_state_features = [
            'remaining_signals',
            'residual_power',
            'current_snr',
            'extraction_success_rate'
        ]
        self.rl_controller = AdaptiveComplexityController(
            state_features=rl_state_features,
            complexity_levels=['low', 'medium', 'high'],
            learning_rate=self.config.get('rl_learning_rate', 1e-3)
        )
        
        # 5. Bias Corrector
        self.bias_corrector = BiasCorrector(
            input_dim=self.param_dim,
            hidden_dims=[256, 128, 64]
        )
        
        # 6. Adaptive Subtractor
        self.adaptive_subtractor = AdaptiveSubtractor(
            param_names=self.param_names,
            complexity_level='medium'
        )
    
    def extract_single_signal(self,
                             strain_data: torch.Tensor,
                             complexity: str = 'medium') -> Dict[str, Any]:
        """
        Extract parameters for a single signal.
        
        Args:
            strain_data: (batch, n_detectors, time_samples)
            complexity: Model complexity level
            
        Returns:
            Dictionary with estimated parameters and uncertainties
        """
        # Encode context
        context = self.context_encoder(strain_data)
        
        # Sample from posterior
        n_samples = {'low': 500, 'medium': 1000, 'high': 2000}[complexity]
        
        # Sample from standard normal
        z = torch.randn(strain_data.size(0), n_samples, self.param_dim, device=self.device)
        
        # Transform to parameter space
        param_samples = []
        for i in range(z.size(1)):
            params = self.flow.inverse(z[:, i, :], context)
            param_samples.append(params)
        
        param_samples = torch.stack(param_samples, dim=1)  # (batch, n_samples, param_dim)
        
        # Compute statistics
        means = param_samples.mean(dim=1)
        stds = param_samples.std(dim=1)
        
        # Apply bias correction
        corrected_means = self.bias_corrector(means, context)
        
        return {
            'means': corrected_means,
            'stds': stds,
            'samples': param_samples,
            'context': context
        }
    
    def extract_overlapping_signals(self,
                                   strain_data: torch.Tensor,
                                   true_params: Optional[List[Dict]] = None,
                                   training: bool = False) -> Dict[str, Any]:
        """
        Extract all overlapping signals iteratively.
        
        Args:
            strain_data: (batch, n_detectors, time_samples)
            true_params: Ground truth parameters (for training)
            training: Whether in training mode
            
        Returns:
            Dictionary with all extracted signals
        """
        batch_size = strain_data.size(0)
        
        # Storage for extracted signals
        all_extracted = []
        residual_data = strain_data.clone()
        
        # Pipeline state for RL controller
        pipeline_state = {
            'remaining_signals': self.max_iterations,
            'residual_power': 1.0,
            'current_snr': 0.0,
            'extraction_success_rate': 1.0
        }
        
        for iteration in range(self.max_iterations):
            # Rank remaining signals with PriorityNet
            with torch.no_grad():
                # Convert residual to detection format
                detections = self._residual_to_detections(residual_data)
                priorities, _ = self.priority_net(detections)
            
            # Select complexity level with RL
            complexity = self.rl_controller.get_complexity_level(
                pipeline_state,
                training=training
            )
            
            # Extract highest priority signal
            extraction_result = self.extract_single_signal(residual_data, complexity)
            
            # Store result
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
            
            # Check stopping condition
            if pipeline_state['residual_power'] < 0.01:
                self.logger.info(f"Stopping at iteration {iteration+1}: residual power too low")
                break
            
            # RL training step
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
            # Compute SNR proxy
            snr_proxy = float(torch.sqrt(torch.mean(residual[i] ** 2)))
            
            detection = {
                'network_snr': snr_proxy * 10.0,  # Scale to realistic SNR range
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
            params_np = params_np[0]  # Take first batch item
        
        return {name: float(params_np[i]) for i, name in enumerate(self.param_names)}
    
    def _compute_extraction_reward(self,
                                   estimated_params: torch.Tensor,
                                   true_params: Optional[Dict]) -> float:
        """Compute reward for RL training."""
        if true_params is None:
            return 0.0
        
        # Convert true params to tensor
        true_tensor = torch.tensor(
            [true_params.get(name, 0.0) for name in self.param_names],
            dtype=torch.float32,
            device=self.device
        )
        
        # Compute relative error
        rel_error = torch.abs((estimated_params[0] - true_tensor) / (true_tensor + 1e-6))
        accuracy = 1.0 - torch.mean(rel_error).item()
        
        return max(0.0, accuracy)
    
    def compute_loss(self,
                    strain_data: torch.Tensor,
                    true_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            strain_data: (batch, n_detectors, time_samples)
            true_params: (batch, param_dim)
            
        Returns:
            Dictionary of losses
        """
        # Encode context
        context = self.context_encoder(strain_data)
        
        # Forward through flow (compute negative log-likelihood)
        z, log_det = self.flow(true_params, context)
        
        # Standard normal prior
        log_prior = -0.5 * torch.sum(z ** 2, dim=1)
        
        # Negative log-likelihood
        nll = -(log_prior + log_det).mean()
        
        # Total loss
        total_loss = nll
        
        return {
            'total_loss': total_loss,
            'nll': nll,
            'log_det': log_det.mean(),
            'log_prior': log_prior.mean()
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
