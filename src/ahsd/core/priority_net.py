#!/usr/bin/env python3
"""
Production PriorityNet for intelligent signal extraction ordering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import logging

class SignalFeatureExtractor(nn.Module):
    """Production-level signal feature extractor"""
    
    def __init__(self, input_dim: int = 15):
        super().__init__()
        
        # Simple but effective feature processing - NO BatchNorm to avoid single sample issues
        self.feature_processor = nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)  # Output 32 features
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass"""
        return self.feature_processor(x)
    
    def _compute_physics_features(self, params: torch.Tensor) -> torch.Tensor:
        """Compute derived physics quantities"""
        
        batch_size = params.shape[0]
        physics_features = torch.zeros(batch_size, 8)
        
        try:
            # Extract basic parameters (denormalized)
            m1 = params[:, 0] * 95 + 5  # [5, 100] solar masses
            m2 = params[:, 1] * 95 + 5
            distance = params[:, 2] * 2950 + 50  # [50, 3000] Mpc
            
            # Derived quantities
            total_mass = m1 + m2
            chirp_mass = (m1 * m2)**(3/5) / total_mass**(1/5)
            mass_ratio = torch.minimum(m1, m2) / torch.maximum(m1, m2)
            eta = m1 * m2 / (total_mass**2)
            
            # SNR estimation
            estimated_snr = 8.0 * (chirp_mass / 30.0)**(5/6) * (400.0 / torch.clamp(distance, min=50.0))
            
            # Frequency estimates
            f_isco = 220.0 / total_mass
            
            # Effective spin (if available)
            if params.shape[1] >= 11:
                a1 = params[:, 9] * 0.99  # [0, 0.99]
                a2 = params[:, 10] * 0.99
                chi_eff = (m1 * a1 + m2 * a2) / total_mass
            else:
                chi_eff = torch.zeros_like(m1)
            
            # Detection difficulty
            difficulty = torch.log(distance / 100.0) - torch.log(estimated_snr / 10.0)
            
            # Normalize features for network
            physics_features[:, 0] = chirp_mass / 50.0
            physics_features[:, 1] = mass_ratio
            physics_features[:, 2] = eta * 4
            physics_features[:, 3] = torch.clamp(estimated_snr / 50.0, 0, 1)
            physics_features[:, 4] = torch.clamp(f_isco / 1000.0, 0, 1)
            physics_features[:, 5] = torch.clamp((chi_eff + 1) / 2, 0, 1)
            physics_features[:, 6] = torch.clamp((difficulty + 5) / 10, 0, 1)
            physics_features[:, 7] = total_mass / 200.0
            
        except Exception as e:
            logging.debug(f"Physics feature computation failed: {e}")
            physics_features.fill_(0.5)
        
        return physics_features

class PriorityNet(nn.Module):
    """Production-level PriorityNet for signal ranking"""
    
    def __init__(self, config=None):
        super().__init__()
        
        self.config = config or self._default_config()
        
        # Feature extraction
        self.signal_encoder = SignalFeatureExtractor()
        
        # Simple priority prediction
        self.priority_predictor = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        logging.info("✅ PriorityNet initialized successfully")
        
    def _default_config(self):
        return type('Config', (), {
            'hidden_dims': [256, 128, 64],
            'dropout': 0.1,
            'learning_rate': 1e-3
        })()
    
    def forward(self, detections: List[Dict]) -> torch.Tensor:
        """Forward pass for list of detection dictionaries"""
        
        if not detections:
            return torch.empty(0)
        
        try:
            # Convert to tensor
            signal_tensor = self._detections_to_tensor(detections)
            
            if signal_tensor is None or signal_tensor.numel() == 0:
                return torch.zeros(len(detections))
            
            # Extract features
            features = self.signal_encoder(signal_tensor)
            
            # Predict priorities
            priorities = self.priority_predictor(features)
            
            return priorities.squeeze(-1)
            
        except Exception as e:
            logging.debug(f"Forward pass error: {e}")
            return torch.zeros(len(detections))
    
    def rank_detections(self, detections: List[Dict]) -> List[int]:
        """Rank detections by priority"""
        
        if not detections:
            return []
        
        try:
            with torch.no_grad():
                scores = self.forward(detections)
                
                if scores.numel() == 0 or len(scores) != len(detections):
                    return list(range(len(detections)))
                
                ranked_indices = torch.argsort(scores, descending=True).tolist()
                return ranked_indices
                
        except Exception as e:
            logging.warning(f"Ranking failed: {e}")
            return self._snr_fallback_ranking(detections)
    
    def _detections_to_tensor(self, detections: List[Dict]) -> torch.Tensor:
        """Convert detection dictionaries to tensor format"""
        
        param_names = [
            'mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec',
            'geocent_time', 'theta_jn', 'psi', 'phase', 'a_1', 'a_2',
            'tilt_1', 'tilt_2', 'phi_12', 'phi_jl'
        ]
        
        ranges = {
            'mass_1': (5.0, 100.0), 'mass_2': (5.0, 100.0),
            'luminosity_distance': (50.0, 3000.0),
            'ra': (0.0, 2*np.pi), 'dec': (-np.pi/2, np.pi/2),
            'geocent_time': (-0.1, 0.1),
            'theta_jn': (0.0, np.pi), 'psi': (0.0, np.pi),
            'phase': (0.0, 2*np.pi),
            'a_1': (0.0, 0.99), 'a_2': (0.0, 0.99),
            'tilt_1': (0.0, np.pi), 'tilt_2': (0.0, np.pi),
            'phi_12': (0.0, 2*np.pi), 'phi_jl': (0.0, 2*np.pi)
        }
        
        defaults = {
            'mass_1': 35.0, 'mass_2': 30.0, 'luminosity_distance': 500.0,
            'ra': 1.0, 'dec': 0.0, 'geocent_time': 0.0,
            'theta_jn': np.pi/2, 'psi': 0.0, 'phase': 0.0,
            'a_1': 0.0, 'a_2': 0.0, 'tilt_1': 0.0, 'tilt_2': 0.0,
            'phi_12': 0.0, 'phi_jl': 0.0
        }
        
        try:
            tensor_data = []
            
            for detection in detections:
                signal_params = []
                
                for param_name in param_names:
                    try:
                        value = detection.get(param_name, defaults[param_name])
                        
                        if isinstance(value, dict):
                            value = value.get('median', value.get('mean', defaults[param_name]))
                        
                        value = float(value)
                        
                        if not np.isfinite(value):
                            value = defaults[param_name]
                        
                        # Normalize to [0, 1]
                        min_val, max_val = ranges[param_name]
                        normalized = (value - min_val) / (max_val - min_val)
                        normalized = np.clip(normalized, 0.0, 1.0)
                        
                        signal_params.append(normalized)
                        
                    except:
                        min_val, max_val = ranges[param_name]
                        default_val = defaults[param_name]
                        normalized = (default_val - min_val) / (max_val - min_val)
                        signal_params.append(np.clip(normalized, 0.0, 1.0))
                
                tensor_data.append(signal_params)
            
            return torch.tensor(tensor_data, dtype=torch.float32)
            
        except Exception as e:
            logging.error(f"Tensor conversion failed: {e}")
            n_detections = len(detections)
            return torch.full((n_detections, 15), 0.5, dtype=torch.float32)
    
    def _snr_fallback_ranking(self, detections: List[Dict]) -> List[int]:
        """Fallback SNR-based ranking"""
        
        try:
            snr_scores = []
            for i, detection in enumerate(detections):
                snr = detection.get('network_snr', 0.0)
                if not np.isfinite(snr):
                    snr = 0.0
                snr_scores.append((i, snr))
            
            snr_scores.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in snr_scores]
            
        except:
            return list(range(len(detections)))

class CombinedPriorityLoss(nn.Module):
    """Production loss function for priority training"""
    
    def __init__(self, ranking_weight=0.6, mse_weight=0.4):
        super().__init__()
        self.ranking_weight = ranking_weight
        self.mse_weight = mse_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # MSE component
        mse_loss = nn.functional.mse_loss(predictions, targets)
        
        # Ranking loss component
        ranking_loss = self._compute_ranking_loss(predictions, targets)
        
        # Combined loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.ranking_weight * ranking_loss)
        
        return total_loss
    
    def _compute_ranking_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute pairwise ranking loss"""
        
        n_samples = predictions.shape[0]
        if n_samples < 2:
            return torch.tensor(0.0, requires_grad=True)
        
        loss = 0.0
        count = 0
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if targets[i] > targets[j]:
                    diff = predictions[j] - predictions[i] + 0.1
                    loss += torch.clamp(diff, min=0.0)
                    count += 1
                elif targets[j] > targets[i]:
                    diff = predictions[i] - predictions[j] + 0.1
                    loss += torch.clamp(diff, min=0.0)
                    count += 1
        
        return loss / max(count, 1)

class PriorityNetTrainer:
    """Production trainer for PriorityNet"""
    
    def __init__(self, model: PriorityNet, config=None):
        self.model = model
        self.config = config
        
        # Extract learning rate from config if available
        if config and hasattr(config, 'learning_rate'):
            lr = config.learning_rate
        elif config and hasattr(config, '__dict__') and 'learning_rate' in config.__dict__:
            lr = config.__dict__['learning_rate']
        else:
            lr = 1e-3
            
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=500, eta_min=1e-6
        )
        self.criterion = CombinedPriorityLoss()
        
    def train_step(self, detections_batch: List[List[Dict]], priorities_batch: List[torch.Tensor]) -> Dict[str, float]:
        """Single training step over a batch of batches (as expected by the training script)"""
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        valid_batches = 0
        for detections, target_priorities in zip(detections_batch, priorities_batch):
            if not detections or len(target_priorities) == 0:
                continue
            try:
                predicted_priorities = self.model(detections)
                if predicted_priorities.numel() == 0:
                    continue
                min_len = min(len(predicted_priorities), len(target_priorities))
                if min_len == 0:
                    continue
                pred_slice = predicted_priorities[:min_len]
                target_slice = target_priorities[:min_len]
                loss = self.criterion(pred_slice, target_slice)
                total_loss += loss
                valid_batches += 1
            except Exception as e:
                logging.debug(f"Training step error: {e}")
                continue
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            return {'loss': float(avg_loss)}
        else:
            return {'loss': 0.0}
    def train_epoch(self, data_loader) -> Dict[str, float]:
        """Train for one epoch"""
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            detections, priorities = batch
            
            # Skip empty batches
            if not detections or len(priorities) == 0:
                continue
                
            loss_info = self.train_step(detections, priorities)
            total_loss += loss_info['loss']
            num_batches += 1
            
        # Step the learning rate scheduler once per epoch
        self.scheduler.step()
        
        if num_batches > 0:
            return {'epoch_loss': total_loss / num_batches}
        else:
            return {'epoch_loss': 0.0}
