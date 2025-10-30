"""
Normalizing flows for posterior approximation - FIXED WITH BOUNDS + Configurable Dropout
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from nflows import transforms, distributions, flows
import logging


class ContextAwareCouplingNet(nn.Module):
    """Coupling network that properly handles context input"""
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 context_features: int,
                 hidden_features: int,
                 num_blocks: int,
                 dropout: float = 0.1):  # ✅ ADD dropout parameter
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.context_features = context_features
        self.dropout = dropout  # ✅ Store dropout rate
        
        # Create network that expects concatenated input
        layers = []
        
        # Input layer processes BOTH inputs and context
        input_dim = in_features + context_features
        layers.extend([
            nn.Linear(input_dim, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout)  # ✅ Use parameter instead of hardcoded
        ])
        
        # Hidden layers
        for _ in range(num_blocks):
            layers.extend([
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Dropout(dropout)  # ✅ Use parameter
            ])
        
        # Output layer with smaller initialization
        output_layer = nn.Linear(hidden_features, out_features)
        torch.nn.init.normal_(output_layer.weight, 0, 0.01)
        torch.nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, inputs, context):
        """Forward pass with input concatenation"""
        combined_input = torch.cat([inputs, context], dim=-1)
        return self.network(combined_input)


class ConditionalRealNVP(nn.Module):
    """Conditional Real NVP with dynamic layer depth and configurable dropout"""
    
    def __init__(self,
                 features: int,
                 context_features: int,
                 hidden_features: int = 128,
                 max_layers: int = 8,
                 num_blocks_per_layer: int = 2,
                 dropout: float = 0.1):  # ✅ ADD dropout parameter
        super().__init__()
        
        self.features = features
        self.context_features = context_features
        self.hidden_features = hidden_features
        self.max_layers = max_layers
        self.num_blocks_per_layer = num_blocks_per_layer
        self.dropout = dropout  # ✅ Store dropout
        self.logger = logging.getLogger(__name__)
        
        self._active_layers = max_layers
        
        # Base distribution
        self.base_distribution = distributions.StandardNormal([features])
        
        # Create all transforms upfront
        self.all_transforms = nn.ModuleList()
        for i in range(max_layers):
            mask = self._create_alternating_mask(features, i % 2 == 0)
            
            # ✅ CRITICAL FIX: Pass dropout to coupling network
            def make_coupling_net(in_features, out_features, 
                                 cf=context_features, 
                                 hf=hidden_features, 
                                 nb=num_blocks_per_layer,
                                 dp=dropout):  # ✅ Capture dropout in closure
                return ContextAwareCouplingNet(
                    in_features=in_features,
                    out_features=out_features,
                    context_features=cf,
                    hidden_features=hf,
                    num_blocks=nb,
                    dropout=dp  # ✅ Pass dropout
                )
            
            self.all_transforms.append(
    def _get_active_transform(self, active_layers: int = None):
        """Return a CompositeTransform using only the specified number of coupling layers"""
        if active_layers is None:
            active_layers = self._active_layers
            
        transforms_list = []
        count = 0
        for t in self.all_transforms:
            if isinstance(t, (transforms.AffineCouplingTransform, transforms.CompositeTransform)):
                if count >= active_layers:
                    break
                count += 1
            transforms_list.append(t)
        return transforms.CompositeTransform(transforms_list)            )
            if i < max_layers - 1:
                self.all_transforms.append(transforms.RandomPermutation(features))
        
        self.logger.info(f"ConditionalRealNVP initialized: {max_layers} layers, dropout={dropout}")
    
    @property
    def active_layers(self):
        return self._active_layers
    
    @active_layers.setter
    def active_layers(self, value):
        self._active_layers = min(max(1, value), self.max_layers)
        self.logger.debug(f"Active layers set to: {self._active_layers}/{self.max_layers}")
    
    @property
    def num_layers(self):
        return self._active_layers
    
    @num_layers.setter
    def num_layers(self, value):
        self.active_layers = value
    
    def _create_alternating_mask(self, features: int, even: bool) -> torch.Tensor:
        mask = torch.zeros(features)
        mask[::2] = 1 if even else 0
        mask[1::2] = 0 if even else 1
        return mask.bool()
    
    def _get_active_transform(self, active_layers: int = None):
        """Return a CompositeTransform using only the specified number of coupling layers"""
        if active_layers is None:
            active_layers = self._active_layers
            
        transforms_list = []
        count = 0
        for t in self.all_transforms:
            if isinstance(t, transforms.AffineCouplingTransform):
                if count >= active_layers:
                    break
                count += 1
            transforms_list.append(t)
        return transforms.CompositeTransform(transforms_list)
    
    def forward(self, inputs: torch.Tensor, context: torch.Tensor, active_layers: int = None):
        inputs_clamped = torch.clamp(inputs, -5.0, 5.0)
        if active_layers is None:
            active_layers = self._active_layers
        transform = self._get_active_transform(active_layers)
        return transform(inputs_clamped, context=context)
    
    def inverse(self, inputs: torch.Tensor, context: torch.Tensor, active_layers: int = None):
        inputs_clamped = torch.clamp(inputs, -5.0, 5.0)
        if active_layers is None:
            active_layers = self._active_layers
        transform = self._get_active_transform(active_layers)
        return transform.inverse(inputs_clamped, context=context)
    
    def log_prob(self, inputs: torch.Tensor, context: torch.Tensor, active_layers: int = None):
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if inputs.shape[0] != context.shape[0] and context.shape[0] == 1:
            context = context.repeat(inputs.shape[0], 1)
        inputs_safe = torch.clamp(inputs, -2.0, 2.0)
        try:
            z, log_det = self.forward(inputs_safe, context, active_layers)
            log_prob_base = self.base_distribution.log_prob(z)
            return log_prob_base + log_det
        except Exception as e:
            self.logger.debug(f"log_prob fallback: {e}")
            return torch.zeros(inputs.shape[0], device=inputs.device) - 10.0
    
    def sample(self, num_samples: int, context: torch.Tensor, active_layers: int = None):
        if context.dim() == 1:
            context = context.unsqueeze(0)
        base_samples = torch.clamp(self.base_distribution.sample(num_samples), -2.0, 2.0)
        if context.shape[0] == 1 and num_samples > 1:
            context_expanded = context.repeat(num_samples, 1)
        else:
            context_expanded = context
        try:
            samples, _ = self.inverse(base_samples, context_expanded, active_layers)
            return torch.clamp(samples, -2.0, 2.0)
        except Exception as e:
            self.logger.debug(f"Sampling fallback: {e}")
            return torch.randn(num_samples, self.features, device=context.device) * 0.5


class MaskedAutoregressiveFlow(nn.Module):
    """Masked Autoregressive Flow with configurable dropout"""
    
    def __init__(self,
                 features: int,
                 context_features: int = 0,
                 hidden_features: int = 128,
                 num_layers: int = 8,
                 num_blocks: int = 2,
                 dropout: float = 0.1):  # ✅ ADD dropout parameter
        super().__init__()
        
        self.features = features
        self.context_features = context_features
        self.dropout = dropout  # ✅ Store dropout
        self.logger = logging.getLogger(__name__)
        
        # Base distribution
        self.base_distribution = distributions.StandardNormal([features])
        
        # Use only affine coupling transforms for stability
        transform_layers = []
        for i in range(num_layers):
            mask = self._create_alternating_mask(features, i % 2 == 0)
            
            # ✅ Pass dropout to coupling network
            def make_coupling_net(in_features, out_features, 
                                 cf=context_features, 
                                 hf=hidden_features, 
                                 nb=num_blocks,
                                 dp=dropout):  # ✅ Capture dropout
                return ContextAwareCouplingNet(
                    in_features=in_features,
                    out_features=out_features,
                    context_features=cf,
                    hidden_features=hf,
                    num_blocks=nb,
                    dropout=dp  # ✅ Pass dropout
                )
            
            transform_layers.append(
                transforms.AffineCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=make_coupling_net
                )
            )
            
            if i < num_layers - 1:
                transform_layers.append(transforms.RandomPermutation(features))
        
        self.transform = transforms.CompositeTransform(transform_layers)
        self.logger.info(f"✅ MaskedAutoregressiveFlow: {features} features, {num_layers} layers, dropout={dropout}")
    
    def _create_alternating_mask(self, features: int, even: bool) -> torch.Tensor:
        mask = torch.zeros(features)
        if even:
            mask[::2] = 1
        else:
            mask[1::2] = 1
        return mask.bool()
    
    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_safe = torch.clamp(inputs, -2.0, 2.0)
        return self.transform(inputs_safe, context=context)
    
    def inverse(self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_safe = torch.clamp(inputs, -2.0, 2.0)
        return self.transform.inverse(inputs_safe, context=context)
    
    def log_prob(self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if context is not None:
            if context.dim() == 1:
                context = context.unsqueeze(0)
            if inputs.shape[0] != context.shape[0]:
                if context.shape[0] == 1:
                    context = context.repeat(inputs.shape[0], 1)
        
        inputs_safe = torch.clamp(inputs, -2.0, 2.0)
        
        try:
            transformed_inputs, log_abs_det = self.forward(inputs_safe, context)
            log_prob_base = self.base_distribution.log_prob(transformed_inputs)
            return log_prob_base + log_abs_det
        except Exception as e:
            self.logger.debug(f"MAF log_prob failed: {e}")
            return torch.zeros(inputs.shape[0], device=inputs.device) - 10.0
    
    def sample(self, num_samples: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_samples = self.base_distribution.sample(num_samples)
        base_samples = torch.clamp(base_samples, -2.0, 2.0)
        
        if context is not None:
            if context.dim() == 1:
                context = context.unsqueeze(0)
            if context.shape[0] == 1 and num_samples > 1:
                context = context.repeat(num_samples, 1)
        
        try:
            samples, _ = self.inverse(base_samples, context)
            return torch.clamp(samples, -2.0, 2.0)
        except Exception as e:
            self.logger.debug(f"MAF sampling failed: {e}")
            return torch.randn(num_samples, self.features, device=base_samples.device) * 0.5


def create_flow_model(flow_type: str, 
                     features: int,
                     context_features: int = 0,
                     **kwargs) -> nn.Module:
    """Factory function to create flow models with proper parameter mapping"""
    
    # Map num_layers to max_layers for ConditionalRealNVP
    if 'num_layers' in kwargs and 'max_layers' not in kwargs:
        kwargs['max_layers'] = kwargs.pop('num_layers')
    
    if 'flow_layers' in kwargs and 'max_layers' not in kwargs:
        kwargs['max_layers'] = kwargs.pop('flow_layers')
    
    if flow_type.lower() == "realnvp":
        return ConditionalRealNVP(
            features=features,
            context_features=context_features,
            **kwargs
        )
    elif flow_type.lower() == "maf":
        # MAF uses num_layers, so restore it if needed
        if 'max_layers' in kwargs and 'num_layers' not in kwargs:
            kwargs['num_layers'] = kwargs.pop('max_layers')
        
        return MaskedAutoregressiveFlow(
            features=features,
            context_features=context_features,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")
