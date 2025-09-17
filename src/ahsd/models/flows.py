"""
Normalizing flows for posterior approximation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from nflows import transforms, distributions, flows
import logging

class ConditionalRealNVP(nn.Module):
    """Conditional Real NVP for posterior estimation."""
    
    def __init__(self,
                 features: int,
                 context_features: int,
                 hidden_features: int = 128,
                 num_layers: int = 8,
                 num_blocks_per_layer: int = 2):
        super().__init__()
        
        self.features = features
        self.context_features = context_features
        self.logger = logging.getLogger(__name__)
        
        # Base distribution
        self.base_distribution = distributions.StandardNormal([features])
        
        # Transform layers
        transform_layers = []
        for i in range(num_layers):
            # Coupling transform
            transform_layers.append(
                transforms.PiecewiseRationalQuadraticCouplingTransform(
                    mask=self._create_alternating_mask(features, i % 2 == 0),
                    transform_net_create_fn=lambda in_features, out_features: 
                        self._create_coupling_net(in_features, out_features, 
                                                context_features, hidden_features, 
                                                num_blocks_per_layer)
                )
            )
            
            # Permutation
            if i < num_layers - 1:
                transform_layers.append(transforms.RandomPermutation(features))
        
        # Combine transforms
        self.transform = transforms.CompositeTransform(transform_layers)
        
    def _create_alternating_mask(self, features: int, even: bool) -> torch.Tensor:
        """Create alternating binary mask."""
        mask = torch.zeros(features)
        if even:
            mask[::2] = 1
        else:
            mask[1::2] = 1
        return mask.bool()
    
    def _create_coupling_net(self,
                           in_features: int,
                           out_features: int, 
                           context_features: int,
                           hidden_features: int,
                           num_blocks: int) -> nn.Module:
        """Create coupling network."""
        
        layers = []
        
        # Input layer (features + context)
        layers.extend([
            nn.Linear(in_features + context_features, hidden_features),
            nn.ReLU()
        ])
        
        # Hidden layers
        for _ in range(num_blocks):
            layers.extend([
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU()
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_features, out_features))
        
        return nn.Sequential(*layers)
    
    def forward(self, inputs: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transformation."""
        return self.transform(inputs, context=context)
    
    def inverse(self, inputs: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse transformation."""  
        return self.transform.inverse(inputs, context=context)
    
    def log_prob(self, inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        transformed_inputs, log_abs_det = self.forward(inputs, context)
        log_prob_base = self.base_distribution.log_prob(transformed_inputs)
        return log_prob_base + log_abs_det
    
    def sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        """Sample from the distribution."""
        # Expand context for multiple samples
        if context.dim() == 1:
            context = context.unsqueeze(0)
        context_expanded = context.repeat(num_samples, 1)
        
        # Sample from base distribution
        base_samples = self.base_distribution.sample(num_samples)
        
        # Transform samples
        samples, _ = self.inverse(base_samples, context_expanded)
        return samples

class MaskedAutoregressiveFlow(nn.Module):
    """Masked Autoregressive Flow for density estimation."""
    
    def __init__(self,
                 features: int,
                 context_features: int = 0,
                 hidden_features: int = 128,
                 num_layers: int = 8,
                 num_blocks: int = 2):
        super().__init__()
        
        self.features = features
        self.context_features = context_features
        
        # Base distribution
        self.base_distribution = distributions.StandardNormal([features])
        
        # Transform layers
        transform_layers = []
        for _ in range(num_layers):
            transform_layers.append(
                transforms.MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    num_blocks=num_blocks
                )
            )
            transform_layers.append(transforms.RandomPermutation(features))
        
        # Remove last permutation
        transform_layers = transform_layers[:-1]
        
        self.transform = transforms.CompositeTransform(transform_layers)
        
    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        return self.transform(inputs, context=context)
    
    def inverse(self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass."""
        return self.transform.inverse(inputs, context=context)
    
    def log_prob(self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Log probability."""
        transformed_inputs, log_abs_det = self.forward(inputs, context)
        log_prob_base = self.base_distribution.log_prob(transformed_inputs)
        return log_prob_base + log_abs_det
    
    def sample(self, num_samples: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from distribution."""
        base_samples = self.base_distribution.sample(num_samples)
        
        if context is not None:
            if context.dim() == 1:
                context = context.unsqueeze(0)
            context = context.repeat(num_samples, 1)
        
        samples, _ = self.inverse(base_samples, context)
        return samples

def create_flow_model(flow_type: str, 
                     features: int,
                     context_features: int = 0,
                     **kwargs) -> nn.Module:
    """Factory function to create flow models."""
    
    if flow_type.lower() == "realnvp":
        return ConditionalRealNVP(
            features=features,
            context_features=context_features,
            **kwargs
        )
    elif flow_type.lower() == "maf":
        return MaskedAutoregressiveFlow(
            features=features,
            context_features=context_features,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")
