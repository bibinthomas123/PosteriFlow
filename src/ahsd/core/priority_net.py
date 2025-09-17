import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
from scipy import signal as scipy_signal
import logging

class SignalFeatureExtractor:
    """Extract astrophysically meaningful features from GW candidates"""
    
    def __init__(self):
        self.feature_names = [
            'network_snr', 'chirp_mass_source', 'total_mass_source', 
            'mass_1', 'mass_2', 'luminosity_distance',
            'ra', 'dec', 'a_1', 'a_2',
            'astrophysical_significance', 'frequency_overlap', 
            'time_overlap', 'sky_area_90'
        ]
        
    def extract_features(self, detections: List[Dict]) -> torch.Tensor:
        """Extract exactly 14 features for priority ranking."""
        if not detections:
            return torch.empty(0, 14)
        
        features = []
        for detection in detections:
            feature_vector = [
                detection.get('network_snr', 10.0),           # 1
                detection.get('mass_1', 35.0),                # 2
                detection.get('mass_2', 30.0),                # 3
                detection.get('chirp_mass_source', 30.0),     # 4
                detection.get('total_mass_source', 60.0),     # 5
                detection.get('luminosity_distance', 500.0),  # 6
                detection.get('ra', 0.0),                     # 7
                detection.get('dec', 0.0),                    # 8
                detection.get('a_1', 0.0),                    # 9
                detection.get('a_2', 0.0),                    # 10
                detection.get('theta_jn', 1.57),              # 11
                detection.get('psi', 0.0),                    # 12
                self._compute_frequency_overlap(detection, detections),  # 13
                self._compute_time_overlap(detection, detections)        # 14
            ]
            features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _compute_mass_ratio(self, detection: Dict) -> float:
        """Compute mass ratio from component masses"""
        m1 = detection.get('mass_1_source', detection.get('mass_1', 30.0))
        m2 = detection.get('mass_2_source', detection.get('mass_2', 30.0))
        
        if m1 <= 0 or m2 <= 0:
            return 1.0
        
        return min(m1, m2) / max(m1, m2)
    
    def _compute_redshift(self, distance: float) -> float:
        """Convert luminosity distance to redshift using cosmology"""
        # Simplified conversion: z ≈ H0 * D_L / c
        H0 = 67.4  # km/s/Mpc
        c = 299792.458  # km/s
        
        return H0 * distance / c if distance > 0 else 0.0
    
    def _compute_frequency_overlap(self, signal: Dict, all_signals: List[Dict]) -> float:
        """Compute frequency overlap using chirp mass"""
        overlaps = []
        mc1 = signal.get('chirp_mass_source', signal.get('chirp_mass', 30.0))
        
        for other in all_signals:
            if other != signal:
                mc2 = other.get('chirp_mass_source', other.get('chirp_mass', 30.0))
                if mc1 > 0 and mc2 > 0:
                    # Frequency overlap based on chirp mass similarity
                    overlap = np.exp(-abs(mc1 - mc2) / (0.1 * max(mc1, mc2)))
                    overlaps.append(overlap)
        
        return max(overlaps) if overlaps else 0.0
    
    def _compute_time_overlap(self, signal: Dict, all_signals: List[Dict]) -> float:
        """Compute time overlap using GPS times"""
        overlaps = []
        t1 = signal.get('synthetic_gps_time', signal.get('gps_time', 0.0))
        
        for other in all_signals:
            if other != signal:
                t2 = other.get('synthetic_gps_time', other.get('gps_time', 0.0))
                # Time overlap for signals within 1 second
                overlap = np.exp(-abs(t1 - t2)) if abs(t1 - t2) < 10 else 0.0
                overlaps.append(overlap)
        
        return max(overlaps) if overlaps else 0.0
    
    def _compute_astrophysical_significance(self, signal: Dict) -> float:
        """Compute astrophysical significance score"""
        # Factors that make signals more astrophysically interesting
        
        # SNR contribution
        snr = signal.get('network_snr', 0.0)
        snr_score = min(snr / 20.0, 1.0)  # Normalize to max of 20
        
        # Mass contribution (prefer intermediate mass BBH)
        total_mass = signal.get('total_mass_source', signal.get('total_mass', 60.0))
        mass_score = np.exp(-0.001 * abs(total_mass - 60))  # Peak at 60 solar masses
        
        # Mass ratio contribution (prefer equal mass)
        q = self._compute_mass_ratio(signal)
        q_score = q  # Higher for more equal masses
        
        # Distance contribution (prefer closer events)
        distance = signal.get('luminosity_distance', 500.0)
        distance_score = np.exp(-distance / 1000.0)
        
        return (snr_score + mass_score + q_score + distance_score) / 4.0

class PriorityNet(nn.Module):
    """Neural network for ranking GW signal extraction priority"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Build network architecture
        layers = []
        input_dim = 14  # Match actual feature count  # Match actual feature count  # Number of features
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_dim = hidden_dim
            
        # Output layer - single priority score
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())  # Ensure output is between 0 and 1
        
        self.network = nn.Sequential(*layers)
        self.feature_extractor = SignalFeatureExtractor()
    
    def forward(self, detections: List[Dict]) -> torch.Tensor:
        """Forward pass - compute priority scores"""
        features = self.feature_extractor.extract_features(detections)
        
        # Handle batch normalization for single samples
        if len(features) == 1:
            features = features.repeat(2, 1)
            priority_scores = self.network(features)[0:1]
        else:
            priority_scores = self.network(features)
            
        return priority_scores.squeeze(-1)
    
    def rank_detections(self, detections: List[Dict]) -> List[int]:
        """Return detection indices ranked by priority (highest first)"""
        if len(detections) <= 1:
            return list(range(len(detections)))
            
        with torch.no_grad():
            scores = self.forward(detections)
            ranked_indices = torch.argsort(scores, descending=True).tolist()
        return ranked_indices

class PriorityNetTrainer:
    """Training logic for PriorityNet with real data"""
    
    def __init__(self, model: PriorityNet, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
        self.criterion = nn.MSELoss()
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            if len(batch) == 0:
                continue
                
            batch_loss = 0
            valid_samples = 0
            
            for sample in batch:
                detections = sample['detections']
                targets = sample['target_priorities']
                
                if len(detections) == 0:
                    continue
                
                try:
                    # Forward pass
                    predictions = self.model(detections)
                    loss = self.criterion(predictions, targets)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    batch_loss += loss.item()
                    valid_samples += 1
                    
                except Exception as e:
                    self.logger.warning(f"Skipping batch due to error: {e}")
                    continue
            
            if valid_samples > 0:
                total_loss += batch_loss / valid_samples
                n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')
        self.scheduler.step(avg_loss)
        
        return avg_loss
