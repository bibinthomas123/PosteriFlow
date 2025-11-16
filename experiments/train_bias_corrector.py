#!/usr/bin/env python3
"""
Train Bias Corrector: Advanced Neural Network-based Parameter Bias Correction
Trains the BiasEstimator to learn corrections for hierarchical biases in GW parameter estimation.

Usage:
    python experiments/train_bias_corrector.py --config configs/enhanced_training.yaml
    python experiments/train_bias_corrector.py --epochs 100 --batch_size 64 --learning_rate 1e-4
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import yaml
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import json
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ahsd.core.bias_corrector import BiasCorrector, BiasEstimator
from ahsd.utils.config_loader import load_enhanced_config


def setup_logging(log_file: str = "train_bias_corrector.log", verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class BiasDataset(Dataset):
    """Streaming dataset for bias correction training - loads pre-generated biases from disk"""
    
    def __init__(self, data_dir: str, param_names: List[str], 
                 context_dim: int = 256, split: str = 'train', seed: int = 42):
        """
        Args:
            data_dir: Path to data directory with train/validation/test splits
            param_names: List of parameter names
            context_dim: Context feature dimension
            split: 'train', 'validation', or 'test'
            seed: Random seed for reproducible bias generation (used in preprocessing)
        """
        self.data_dir = Path(data_dir) / split
        self.param_names = param_names
        self.context_dim = context_dim
        self.seed = seed
        self.logger = logging.getLogger(__name__)
        
        # Collect all chunk files
        self.batch_files = sorted(self.data_dir.glob('chunk_*.pkl'))
        if not self.batch_files:
            self.logger.warning(f"No chunk files found in {self.data_dir}")
        
        # Collect pre-generated bias files (created during preprocessing)
        self.bias_files = sorted(self.data_dir.glob('biases_*.pkl'))
        self.has_pregenerated_biases = len(self.bias_files) > 0
        if not self.has_pregenerated_biases:
            self.logger.warning(
                "No pre-generated bias files found. Biases will be generated on-the-fly (not reproducible). "
                "Run preprocessing script to generate biases: "
                "python scripts/preprocess_biases.py"
            )
        
        # Build index: (batch_idx, sample_idx_in_batch) -> sample
        self.sample_index = []
        self.pregenerated_biases = {}  # Load if available
        self._build_index()
        
        if self.has_pregenerated_biases:
            self._load_pregenerated_biases()
        
    def _build_index(self):
        """Build index of all samples across all chunks"""
        self.logger.info(f"Building sample index from {len(self.batch_files)} chunks...")
        
        for batch_idx, batch_file in enumerate(self.batch_files):
            try:
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f)
                
                if isinstance(batch, dict) and 'samples' in batch:
                    samples = batch['samples']
                else:
                    samples = batch if isinstance(batch, list) else []
                
                for sample_idx in range(len(samples)):
                    self.sample_index.append((batch_idx, sample_idx))
                    
            except Exception as e:
                self.logger.warning(f"Failed to index batch {batch_file}: {e}")
                continue
        
        self.logger.info(f"Built index with {len(self.sample_index)} total samples")
    
    def _load_pregenerated_biases(self):
        """Load pre-generated biases from disk"""
        self.logger.info(f"Loading pre-generated biases from {len(self.bias_files)} files...")
        
        for bias_file in self.bias_files:
            try:
                with open(bias_file, 'rb') as f:
                    biases = pickle.load(f)
                
                if isinstance(biases, dict):
                    self.pregenerated_biases.update(biases)
            except Exception as e:
                self.logger.warning(f"Failed to load biases from {bias_file}: {e}")
        
        self.logger.info(f"Loaded {len(self.pregenerated_biases)} pre-generated bias entries")
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx):
        """Load sample on-demand"""
        batch_idx, sample_idx = self.sample_index[idx]
        batch_file = self.batch_files[batch_idx]
        
        try:
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f)
            
            if isinstance(batch, dict) and 'samples' in batch:
                samples = batch['samples']
            else:
                samples = batch if isinstance(batch, list) else []
            
            sample = samples[sample_idx]
            
            # Process sample
            if sample is None:
                return self._get_empty_item()
            
            params_list = sample.get('parameters', [])
            if not isinstance(params_list, list):
                params_list = [params_list]
            
            if not params_list:
                return self._get_empty_item()
            
            params = params_list[0] if isinstance(params_list[0], dict) else sample
            
            # Extract true parameters
            true_params = np.array([
                float(params.get(param_name, 0.0)) for param_name in self.param_names
            ])
            
            # Load or generate biased estimates (reproducibility via pre-generated biases)
            sample_key = f"{batch_idx}_{sample_idx}"
            
            if self.has_pregenerated_biases and sample_key in self.pregenerated_biases:
                # ✅ Load pre-generated bias values (reproducible)
                bias_data = self.pregenerated_biases[sample_key]
                estimated_params = bias_data.get('estimated_params', true_params.copy())
                true_correction = bias_data.get('true_correction', true_params - estimated_params)
            else:
                # Fallback: generate on-the-fly (non-reproducible, only when preprocessing not run)
                self.logger.debug(f"Sample {sample_key} not found in pre-generated biases. Generating on-the-fly.")
                estimated_params = true_params.copy()
                for j, param_name in enumerate(self.param_names):
                    if param_name in ['mass_1', 'mass_2']:
                        bias = np.random.normal(0, 0.08)
                    elif param_name == 'luminosity_distance':
                        bias = np.random.normal(0, 0.15)
                    elif param_name == 'geocent_time':
                        bias = np.random.normal(0, 0.001)
                    elif param_name in ['ra', 'dec']:
                        bias = np.random.normal(0, 0.1)
                    else:
                        bias = np.random.normal(0, 0.05)
                    
                    estimated_params[j] += bias * true_params[j] if abs(true_params[j]) > 1e-3 else bias
                
                true_correction = true_params - estimated_params
            
            # Normalize parameters
            param_min = np.array([-100, -100, 10, -np.pi, -np.pi, -1, -1, -1, -1])
            param_max = np.array([100, 100, 5000, np.pi, np.pi, 1, 1, 1, 1])
            
            normalized_params = 2 * (estimated_params - param_min) / (param_max - param_min) - 1
            normalized_params = np.clip(normalized_params, -1, 1)
            
            # Extract strain data from detector_data if available
            strain_segments = {}
            detector_data = sample.get('detector_data', {})
            for detector in ['H1', 'L1', 'V1']:
                if detector in detector_data:
                    det_data = detector_data[detector]
                    if isinstance(det_data, dict):
                        # Try multiple keys: 'strain', 'whitened_data', or direct array
                        if 'strain' in det_data:
                            strain_segments[detector] = det_data['strain']
                        elif 'whitened_data' in det_data:
                            strain_segments[detector] = det_data['whitened_data']
                        elif isinstance(det_data, (list, np.ndarray)):
                            strain_segments[detector] = det_data
                    else:
                        # Direct array
                        strain_segments[detector] = det_data
            
            # Signal quality from SNR
            signal_quality = float(params.get('target_snr', 15.0)) / 50.0
            
            # ✅ FIXED: Encode strain segments to context tensor (for trainer compatibility)
            context_tensor = self._encode_strain_to_context(strain_segments)
            
            return {
                'param_tensor': torch.tensor(normalized_params, dtype=torch.float32),
                'context_tensor': context_tensor,  # ✅ Now returns context_tensor as trainer expects
                'true_correction': torch.tensor(true_correction, dtype=torch.float32),
                'signal_quality': torch.tensor(signal_quality, dtype=torch.float32)
            }
            
        except Exception as e:
            self.logger.debug(f"Error loading sample {idx}: {e}")
            return self._get_empty_item()
    
    def _encode_strain_to_context(self, strain_segments: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Encode strain segments to context tensor [context_dim].
        
        Simple implementation: concatenate detector strain power as features.
        In production, use TransformerStrainEncoder for sophisticated encoding.
        
        Args:
            strain_segments: Dict of {detector: strain_array}
        
        Returns:
            context_tensor: [context_dim] feature vector
        """
        try:
            features = []
            
            # Extract power features from each detector
            for detector in ['H1', 'L1', 'V1']:
                if detector in strain_segments:
                    strain = np.array(strain_segments[detector])
                    # Simple features: RMS, variance, peak, spectral properties
                    rms = float(np.sqrt(np.mean(strain**2))) if len(strain) > 0 else 0.0
                    var = float(np.var(strain)) if len(strain) > 0 else 0.0
                    peak = float(np.max(np.abs(strain))) if len(strain) > 0 else 0.0
                    features.extend([rms, var, peak])
                else:
                    features.extend([0.0, 0.0, 0.0])  # Zeros for missing detector
            
            # Pad or truncate to context_dim
            features_array = np.array(features, dtype=np.float32)
            
            if len(features_array) >= self.context_dim:
                context = features_array[:self.context_dim]
            else:
                # Pad with zeros to reach context_dim
                context = np.pad(features_array, (0, self.context_dim - len(features_array)))
            
            return torch.tensor(context, dtype=torch.float32)
        
        except Exception as e:
            self.logger.debug(f"Context encoding failed: {e}")
            return torch.zeros(self.context_dim, dtype=torch.float32)
    
    def _get_empty_item(self):
        """Return empty/default item on error"""
        return {
            'param_tensor': torch.zeros(len(self.param_names), dtype=torch.float32),
            'context_tensor': torch.zeros(self.context_dim, dtype=torch.float32),  # ✅ Empty context tensor
            'true_correction': torch.zeros(len(self.param_names), dtype=torch.float32),
            'signal_quality': torch.tensor(0.3, dtype=torch.float32)
        }


class BiasCorrectorTrainer:
    """Trainer class for BiasCorrector"""
    
    def __init__(self, bias_corrector: BiasCorrector, config: Dict[str, Any], 
                 device: str = 'cpu', logger: logging.Logger = None):
        """Initialize trainer
        
        Args:
            bias_corrector: BiasCorrector module to train
            config: Training configuration
            device: Device to train on ('cpu' or 'cuda')
            logger: Logger instance
        """
        self.bias_corrector = bias_corrector
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        self.bias_corrector.to(device)
        
        # Training hyperparameters
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 200)
        self.patience = config.get('patience', 20)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.grad_clip = config.get('grad_clip', 1.0)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.bias_corrector.bias_estimator.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs
        )
        
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_uncertainties': [],
            'val_uncertainties': [],
            'train_max_error': [],
            'val_max_error': [],
            'learning_rates': [],
            # ✅ Acceptance & Correlation Metrics
            'train_acceptance_rate': [],
            'val_acceptance_rate': [],
            'train_avg_confidence': [],
            'val_avg_confidence': [],
            'train_min_confidence': [],
            'val_min_confidence': [],
            'train_correction_std': [],
            'val_correction_std': [],
            'train_correction_bias': [],
            'val_correction_bias': [],
            'train_correction_correlation': [],
            'val_correction_correlation': [],
            'gradient_norms': [],
            'train_val_gap': []
        }
        
    def loss_function(self, pred_corrections: torch.Tensor, 
                     pred_uncertainties: torch.Tensor,
                     true_corrections: torch.Tensor,
                     quality_weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss with uncertainty quantification and acceptance metrics
        
        Args:
            pred_corrections: Predicted bias corrections [batch, param_dim]
            pred_uncertainties: Predicted uncertainties [batch, param_dim]
            true_corrections: Ground truth corrections [batch, param_dim]
            quality_weights: Signal quality weights [batch]
            
        Returns:
            loss: Total loss
            metrics: Dict of loss components and acceptance metrics
        """
        # Negative log-likelihood (Gaussian) - clamped to prevent explosion
        residuals = pred_corrections - true_corrections
        # Clamp uncertainties to reasonable range to prevent NLL explosion
        unc_clamped = torch.clamp(pred_uncertainties, min=0.01, max=1.0)
        mse_loss = torch.mean((residuals ** 2) / (unc_clamped + 1e-8))
        mse_loss = torch.clamp(mse_loss, max=100.0)  # Cap max loss
        
        # Uncertainty regularization (prevent collapse to zero)
        uncertainty_reg = torch.mean(torch.log(unc_clamped + 1e-8))
        
        # Magnitude penalty (prevent large corrections)
        magnitude_penalty = torch.mean(torch.abs(pred_corrections))
        
        # Quality-weighted loss (normalized by batch)
        batch_size = quality_weights.shape[0]
        quality_weights_expanded = quality_weights.view(-1, 1).expand_as(residuals)
        quality_weighted = torch.mean(quality_weights_expanded * (residuals ** 2))
        quality_weighted = torch.clamp(quality_weighted, max=50.0)  # Cap max loss
        
        # Combined loss with balanced weighting
        loss = (
            0.6 * torch.clamp(mse_loss, max=50.0) +
            0.15 * uncertainty_reg +
            0.10 * magnitude_penalty +
            0.15 * quality_weighted
        )
        
        # Compute accuracy metrics
        abs_errors = torch.abs(residuals)
        mae = torch.mean(abs_errors).item()
        rmse = torch.sqrt(torch.mean(residuals ** 2)).item()
        max_error = torch.max(abs_errors).item()
        mean_uncertainty = torch.mean(pred_uncertainties).item()
        
        # ✅ Acceptance metrics (from phase3a_neural_pe.py)
        # Acceptance based on uncertainty: corrections with low uncertainty are accepted
        if pred_uncertainties.numel() > 0:
            # ✅ FIXED: Handle small batches or empty predictions
            uncertainty_threshold = torch.quantile(pred_uncertainties.flatten(), 0.75)  # Top 25% confidence
            accepted_mask = pred_uncertainties < uncertainty_threshold
            acceptance_rate = float(torch.mean(accepted_mask.float()).item())
            
            # Average confidence (inverse of uncertainty)
            avg_confidence = float(torch.mean(1.0 / (pred_uncertainties + 1e-8)).item())
            min_confidence = float(torch.min(1.0 / (pred_uncertainties + 1e-8)).item())
        else:
            # Handle empty batch gracefully
            acceptance_rate = 0.0
            avg_confidence = 0.0
            min_confidence = 0.0
        
        # Correction statistics
        correction_mean = float(torch.mean(torch.abs(pred_corrections)).item())
        correction_std = float(torch.std(torch.abs(pred_corrections)).item())
        
        # Bias in corrections (should be close to zero)
        correction_bias = float(torch.mean(pred_corrections).item())
        
        # Correlation between predicted and true corrections (should be high)
        try:
            pred_flat = pred_corrections.detach().cpu().numpy().flatten()
            true_flat = true_corrections.detach().cpu().numpy().flatten()
            corr, _ = pearsonr(pred_flat, true_flat)
            correction_correlation = float(corr)
        except:
            correction_correlation = 0.0
        
        return loss, {
            'mse': mse_loss.item(),
            'uncertainty_reg': uncertainty_reg.item(),
            'magnitude_penalty': magnitude_penalty.item(),
            'quality_weighted': quality_weighted.item(),
            'mae': mae,
            'rmse': rmse,
            'max_error': max_error,
            'mean_uncertainty': mean_uncertainty,
            # ✅ Acceptance & Correlation Metrics
            'acceptance_rate': acceptance_rate,
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'correction_std': correction_std,
            'correction_bias': correction_bias,
            'correction_correlation': correction_correlation
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """Train one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            avg_loss: Average loss for epoch
            metrics: Dict of loss components and acceptance metrics
        """
        self.bias_corrector.bias_estimator.train()
        train_losses = []
        gradient_norms = []
        metrics_sum = {
            'mse': [],
            'uncertainty_reg': [],
            'magnitude_penalty': [],
            'quality_weighted': [],
            'mae': [],
            'rmse': [],
            'max_error': [],
            'mean_uncertainty': [],
            # ✅ Acceptance & Correlation Metrics
            'acceptance_rate': [],
            'avg_confidence': [],
            'min_confidence': [],
            'correction_std': [],
            'correction_bias': [],
            'correction_correlation': []
        }
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            param_tensors = batch['param_tensor'].to(self.device)
            context_tensors = batch['context_tensor'].to(self.device)
            true_corrections = batch['true_correction'].to(self.device)
            quality_weights = batch['signal_quality'].to(self.device)
            
            # Forward pass
            pred_corrections, pred_uncertainties = self.bias_corrector.bias_estimator(
                param_tensors, context_tensors
            )
            
            # Compute loss
            loss, metrics = self.loss_function(
                pred_corrections, pred_uncertainties,
                true_corrections, quality_weights
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Compute gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.bias_corrector.bias_estimator.parameters(),
                self.grad_clip
            )
            gradient_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            
            self.optimizer.step()
            
            train_losses.append(loss.item())
            for key in metrics_sum:
                metrics_sum[key].append(metrics[key])
        
        avg_loss = np.mean(train_losses)
        avg_metrics = {key: np.mean(values) for key, values in metrics_sum.items()}
        avg_metrics['avg_gradient_norm'] = np.mean(gradient_norms)
        
        return avg_loss, avg_metrics
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            avg_loss: Average validation loss
            metrics: Dict of metrics including acceptance metrics
        """
        self.bias_corrector.bias_estimator.eval()
        val_losses = []
        metrics_sum = {
            'mse': [],
            'uncertainty_reg': [],
            'magnitude_penalty': [],
            'quality_weighted': [],
            'mae': [],
            'rmse': [],
            'max_error': [],
            'mean_uncertainty': [],
            # ✅ Acceptance & Correlation Metrics
            'acceptance_rate': [],
            'avg_confidence': [],
            'min_confidence': [],
            'correction_std': [],
            'correction_bias': [],
            'correction_correlation': []
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                param_tensors = batch['param_tensor'].to(self.device)
                context_tensors = batch['context_tensor'].to(self.device)
                true_corrections = batch['true_correction'].to(self.device)
                quality_weights = batch['signal_quality'].to(self.device)
                
                pred_corrections, pred_uncertainties = self.bias_corrector.bias_estimator(
                    param_tensors, context_tensors
                )
                
                loss, metrics = self.loss_function(
                    pred_corrections, pred_uncertainties,
                    true_corrections, quality_weights
                )
                
                val_losses.append(loss.item())
                for key in metrics_sum:
                    metrics_sum[key].append(metrics[key])
        
        avg_loss = np.mean(val_losses)
        avg_metrics = {key: np.mean(values) for key, values in metrics_sum.items()}
        
        return avg_loss, avg_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None) -> Dict:
        """
        Main training loop with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            training_history: Dict with training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
             # Training phase
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.training_history['train_losses'].append(train_loss)
            self.training_history['train_mae'].append(train_metrics.get('mae', 0))
            self.training_history['train_rmse'].append(train_metrics.get('rmse', 0))
            self.training_history['train_max_error'].append(train_metrics.get('max_error', 0))
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['gradient_norms'].append(train_metrics.get('avg_gradient_norm', 0))
            # ✅ Acceptance & Correlation Metrics
            self.training_history['train_acceptance_rate'].append(train_metrics.get('acceptance_rate', 0))
            self.training_history['train_avg_confidence'].append(train_metrics.get('avg_confidence', 0))
            self.training_history['train_min_confidence'].append(train_metrics.get('min_confidence', 0))
            self.training_history['train_correction_std'].append(train_metrics.get('correction_std', 0))
            self.training_history['train_correction_bias'].append(train_metrics.get('correction_bias', 0))
            self.training_history['train_correction_correlation'].append(train_metrics.get('correction_correlation', 0))
            
            # Validation phase
            val_loss = None
            if val_loader is not None:
                val_loss, val_metrics = self.validate(val_loader)
                self.training_history['val_losses'].append(val_loss)
                self.training_history['val_mae'].append(val_metrics.get('mae', 0))
                self.training_history['val_rmse'].append(val_metrics.get('rmse', 0))
                self.training_history['val_max_error'].append(val_metrics.get('max_error', 0))
                # ✅ Acceptance & Correlation Metrics
                self.training_history['val_acceptance_rate'].append(val_metrics.get('acceptance_rate', 0))
                self.training_history['val_avg_confidence'].append(val_metrics.get('avg_confidence', 0))
                self.training_history['val_min_confidence'].append(val_metrics.get('min_confidence', 0))
                self.training_history['val_correction_std'].append(val_metrics.get('correction_std', 0))
                self.training_history['val_correction_bias'].append(val_metrics.get('correction_bias', 0))
                self.training_history['val_correction_correlation'].append(val_metrics.get('correction_correlation', 0))
                
                # Train-Val gap
                train_val_gap = train_loss - val_loss
                self.training_history['train_val_gap'].append(train_val_gap)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu() for k, v in self.bias_corrector.state_dict().items()}
                else:
                    patience_counter += 1
            
            self.scheduler.step()
            
            # Logging
            if (epoch + 1) % 1 == 0 or epoch == 0:
                msg = f"Epoch {epoch+1:3d}/{self.epochs} | Train Loss: {train_loss:.6f} | MAE: {train_metrics.get('mae', 0):.6f} | RMSE: {train_metrics.get('rmse', 0):.6f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.6f} | Val MAE: {val_metrics.get('mae', 0):.6f} | Patience: {patience_counter}/{self.patience}"
                self.logger.info(msg)
                
                # ✅ Log acceptance metrics
                self.logger.info(f"  ✓ Acceptance Rate: Train={train_metrics.get('acceptance_rate', 0):.2%} | Val={val_metrics.get('acceptance_rate', 0):.2%}" if val_loss is not None else f"  ✓ Acceptance Rate: Train={train_metrics.get('acceptance_rate', 0):.2%}")
                self.logger.info(f"  ⚡ Avg Confidence: Train={train_metrics.get('avg_confidence', 0):.4f} | Val={val_metrics.get('avg_confidence', 0):.4f}" if val_loss is not None else f"  ⚡ Avg Confidence: Train={train_metrics.get('avg_confidence', 0):.4f}")
                self.logger.info(f"  📊 Correction Correlation: Train={train_metrics.get('correction_correlation', 0):.4f} | Val={val_metrics.get('correction_correlation', 0):.4f}" if val_loss is not None else f"  📊 Correction Correlation: Train={train_metrics.get('correction_correlation', 0):.4f}")
                
                # Early stopping check
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.bias_corrector.load_state_dict(best_model_state)
            self.logger.info(f"Restored best model with val_loss={best_val_loss:.6f}")
        
        return self.training_history





def main():
    parser = argparse.ArgumentParser(
        description="Train BiasCorrector neural network for parameter bias correction"
    )
    
    parser.add_argument('--config', type=str, default='configs/enhanced_training.yaml',
                       help='Path to training config')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                       help='Patience for early stopping')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda or cpu)')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of training samples to generate')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--output_dir', type=str, default='models/bias_corrector',
                       help='Output directory for checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    logger.info("="*70)
    logger.info("BiasCorrector Training")
    logger.info("="*70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    try:
        config = load_enhanced_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    except Exception as e:
        logger.warning(f"Failed to load config: {e}. Using defaults.")
        config = {}
    
    # Parameter names
    param_names = [
        'mass_1', 'mass_2', 'luminosity_distance',
        'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time'
    ]
    
    # Initialize bias corrector
    context_dim = config.get('bias_corrector', {}).get('context_dim', 256)
    bias_corrector = BiasCorrector(param_names=param_names, context_dim=context_dim)
    logger.info(f"Initialized BiasCorrector with {len(param_names)} parameters, context_dim={context_dim}")
    
    # Create streaming datasets (loads pre-generated biases from disk)
    data_dir = 'data/dataset'
    
    train_dataset = BiasDataset(
        data_dir=data_dir,
        param_names=param_names,
        context_dim=context_dim,
        split='train',
        seed=args.seed
    )
    
    val_dataset = BiasDataset(
        data_dir=data_dir,
        param_names=param_names,
        context_dim=context_dim,
        split='validation',
        seed=args.seed
    )
    
    logger.info(f"Training set: {len(train_dataset)} samples (streamed)")
    logger.info(f"Validation set: {len(val_dataset)} samples (streamed)")
    
    if len(train_dataset) < 10:
        logger.error("Insufficient training data")
        return
    
    # Create data loaders with streaming (loads batches on-the-fly)
    # ✅ FIXED: persistent_workers=True requires num_workers >= 1; use num_workers=0 for no workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # No multiprocessing for streaming (dataset loads on-demand)
        pin_memory=torch.cuda.is_available(),  # Only pin if using GPU
        persistent_workers=False  # ✅ Removed invalid persistent_workers=True with num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    ) if len(val_dataset) > 0 else None
    
    # Initialize trainer
    train_config = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'weight_decay': 1e-5,
        'grad_clip': 1.0
    }
    
    trainer = BiasCorrectorTrainer(
        bias_corrector=bias_corrector,
        config=train_config,
        device=args.device,
        logger=logger
    )
    
    logger.info("="*70)
    logger.info("Training Configuration:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.learning_rate}")
    logger.info(f"  Patience: {args.patience}")
    logger.info(f"  Device: {args.device}")
    logger.info("="*70)
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    # Save checkpoint
    checkpoint_path = output_dir / 'bias_corrector_best.pth'
    torch.save(bias_corrector.state_dict(), checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    # Convert numpy values to native Python types for JSON serialization
    history_json = {
        k: [float(v) for v in vs] if isinstance(vs, list) else vs
        for k, vs in history.items()
    }
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    
    # Plot training curves with metrics (2x3 grid)
    plot_path = output_dir / 'training_curves.png'
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_losses'], label='Train Loss', linewidth=2)
    if history['val_losses']:
        axes[0, 0].plot(history['val_losses'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE curves
    axes[0, 1].plot(history['train_mae'], label='Train MAE', linewidth=2)
    if history['val_mae']:
        axes[0, 1].plot(history['val_mae'], label='Val MAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].set_title('MAE (Lower is Better)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Acceptance Rate - ✅ NEW
    axes[0, 2].plot(history['train_acceptance_rate'], label='Train Acceptance', linewidth=2, color='green')
    if history['val_acceptance_rate']:
        axes[0, 2].plot(history['val_acceptance_rate'], label='Val Acceptance', linewidth=2, color='darkgreen')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Acceptance Rate')
    axes[0, 2].set_title('Correction Acceptance Rate (Higher is Better)')
    axes[0, 2].set_ylim([0, 1.0])
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # RMSE curves
    axes[1, 0].plot(history['train_rmse'], label='Train RMSE', linewidth=2)
    if history['val_rmse']:
        axes[1, 0].plot(history['val_rmse'], label='Val RMSE', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Root Mean Squared Error')
    axes[1, 0].set_title('RMSE (Lower is Better)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confidence metrics - ✅ NEW
    axes[1, 1].plot(history['train_avg_confidence'], label='Train Avg Confidence', linewidth=2, color='purple')
    if history['val_avg_confidence']:
        axes[1, 1].plot(history['val_avg_confidence'], label='Val Avg Confidence', linewidth=2, color='darkviolet')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Confidence (1/Uncertainty)')
    axes[1, 1].set_title('Average Confidence (Higher is Better)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Correlation - ✅ NEW
    axes[1, 2].plot(history['train_correction_correlation'], label='Train Correlation', linewidth=2, color='red')
    if history['val_correction_correlation']:
        axes[1, 2].plot(history['val_correction_correlation'], label='Val Correlation', linewidth=2, color='darkred')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Pearson Correlation')
    axes[1, 2].set_title('Correction Prediction Correlation (Higher is Better)')
    axes[1, 2].set_ylim([-1.0, 1.0])
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved training curves to {plot_path}")
    plt.close()
    
    # Final summary
    logger.info("="*70)
    logger.info("Training Summary:")
    logger.info(f"  Total Epochs: {len(history['train_losses'])}")
    logger.info(f"  Final Train Loss: {history['train_losses'][-1]:.6f}")
    logger.info(f"  Final Train MAE: {history['train_mae'][-1]:.6f}")
    logger.info(f"  Final Train RMSE: {history['train_rmse'][-1]:.6f}")
    if history['val_losses']:
        logger.info(f"  Final Val Loss: {history['val_losses'][-1]:.6f}")
        logger.info(f"  Final Val MAE: {history['val_mae'][-1]:.6f}")
        logger.info(f"  Best Val Loss: {min(history['val_losses']):.6f}")
        logger.info(f"  Best Val MAE: {min(history['val_mae']):.6f}")
    
    # ✅ Summary of Acceptance & Correlation Metrics
    logger.info("\n  ✅ Bias Corrector Quality Metrics:")
    logger.info(f"    Acceptance Rate (Train): {history['train_acceptance_rate'][-1]:.2%}")
    if history['val_acceptance_rate']:
        logger.info(f"    Acceptance Rate (Val): {history['val_acceptance_rate'][-1]:.2%}")
        logger.info(f"    Best Val Acceptance: {max(history['val_acceptance_rate']):.2%}")
    
    logger.info(f"    Avg Confidence (Train): {history['train_avg_confidence'][-1]:.4f}")
    if history['val_avg_confidence']:
        logger.info(f"    Avg Confidence (Val): {history['val_avg_confidence'][-1]:.4f}")
    
    logger.info(f"    Correction Correlation (Train): {history['train_correction_correlation'][-1]:.4f}")
    if history['val_correction_correlation']:
        logger.info(f"    Correction Correlation (Val): {history['val_correction_correlation'][-1]:.4f}")
        logger.info(f"    Best Val Correlation: {max(history['val_correction_correlation']):.4f}")
    
    logger.info(f"    Correction Std (Train): {history['train_correction_std'][-1]:.6f}")
    logger.info(f"    Correction Bias (Train): {history['train_correction_bias'][-1]:.6f}")
    if history['val_correction_bias']:
        logger.info(f"    Correction Bias (Val): {history['val_correction_bias'][-1]:.6f}")
    
    logger.info(f"\n  Checkpoint: {checkpoint_path}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
