#!/usr/bin/env python3
"""
Train Bias Corrector: Advanced Neural Network-based Parameter Bias Correction
Trains the BiasEstimator to learn corrections for hierarchical biases in GW parameter estimation.

Uses frozen PriorityNet and NeuralPE models + precomputed posterior statistics for fast training.

Workflow:
    1. Precompute posterior stats: python experiments/precompute_posterior_stats.py
    2. Train BiasCorrector: python experiments/train_bias_corrector.py

Usage:
    python experiments/train_bias_corrector.py --config configs/enhanced_training.yaml
    python experiments/train_bias_corrector.py --epochs 100 --batch_size 64 --learning_rate 1e-4
    python experiments/train_bias_corrector.py --priority_net_path models/priority_net/priority_net_best.pth --neural_pe_path models/neural_pe/best_model.pth
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ahsd.core.bias_corrector import BiasCorrector, BiasEstimator
from ahsd.utils.config_loader import load_enhanced_config
from ahsd.core.priority_net import PriorityNet
from ahsd.models.overlap_neuralpe import OverlapNeuralPE


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
    """Streaming dataset for bias correction training - loads pre-computed posterior statistics from disk"""
    
    def __init__(self, data_dir: str, param_names: List[str], 
                 context_dim: int = 256, split: str = 'train', seed: int = 42,
                 physics_bounds: Dict = None):
        """
        Args:
            data_dir: Path to data directory with train/validation/test splits
            param_names: List of parameter names
            context_dim: Context feature dimension
            split: 'train', 'validation', or 'test'
            seed: Random seed for reproducible bias generation (used in preprocessing)
            physics_bounds: Physics bounds dict for parameter normalization
        """
        self.data_dir = Path(data_dir)  # ✅ Root data dir (not split subdir)
        self.split_dir = self.data_dir / split
        self.param_names = param_names
        self.context_dim = context_dim
        self.seed = seed
        self.physics_bounds = physics_bounds or {}  # ✅ FIX #1: Store bounds for normalization
        self.logger = logging.getLogger(__name__)
        
        # Collect all chunk files from split subdirectory
        self.batch_files = sorted(self.split_dir.glob('chunk_*.pkl'))
        if not self.batch_files:
            self.logger.warning(f"No chunk files found in {self.split_dir}")
        
        # Load precomputed posterior statistics (faster than on-the-fly sampling)
        self.posterior_stats = {}
        self._load_posterior_stats(split)
        
        # Build index: (batch_idx, sample_idx_in_batch) -> sample
        self.sample_index = []
        self._build_index()
        
    def _build_index(self):
        """Build index of all non-edge-case samples across all chunks"""
        self.logger.info(f"Building sample index from {len(self.batch_files)} chunks (excluding edge cases)...")
        
        edge_case_count = 0
        
        for batch_idx, batch_file in enumerate(self.batch_files):
            try:
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f)
                
                if isinstance(batch, dict) and 'samples' in batch:
                    samples = batch['samples']
                else:
                    samples = batch if isinstance(batch, list) else []
                
                for sample_idx in range(len(samples)):
                    sample = samples[sample_idx]
                    if sample.get("is_edge_case"):
                        edge_case_count += 1
                        continue
                    self.sample_index.append((batch_idx, sample_idx))
                    
            except Exception as e:
                self.logger.warning(f"Failed to index batch {batch_file}: {e}")
                continue
        
        self.logger.info(f"Built index with {len(self.sample_index)} clean samples (excluded {edge_case_count} edge cases)")
    
    def _load_posterior_stats(self, split: str):
        """Load precomputed posterior statistics (mean, std) from disk
        
        Stats file: data_dir/posterior_stats_{split}.pkl
        Created by: python experiments/precompute_posterior_stats.py
        
        Each entry: {sample_idx: {'mean': array, 'std': array, 'snr': float, 'true_params': array}}
        """
        stats_path = self.data_dir / f'posterior_stats_{split}.pkl'
        
        if not stats_path.exists():
            self.logger.warning(
                f"⚠️  Posterior stats not found at {stats_path}\n"
                f"    Biases will be generated on-the-fly (slower, less reproducible)\n"
                f"    To precompute: python experiments/precompute_posterior_stats.py "
                f"--data_dir {self.data_dir} --device cuda"
            )
            return
        
        try:
            with open(stats_path, 'rb') as f:
                self.posterior_stats = pickle.load(f)
            self.logger.info(f"✅ Loaded {len(self.posterior_stats)} precomputed posterior stats from {stats_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load posterior stats from {stats_path}: {e}")
            self.posterior_stats = {}
    
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
            
            # Load precomputed posterior statistics or generate on-the-fly
            # Posterior stats provide realistic biased estimates using Neural PE samples
            sample_key = idx  # Use linear index (matches precompute_posterior_stats.py)

            
            if sample_key in self.posterior_stats:
                # ✅ Use precomputed posterior stats to inform bias magnitude
                # Posterior stats contain aggregate mean/std from Neural PE posterior samples
                stats = self.posterior_stats[sample_key]
                
                # Extract posterior statistics for magnitude guidance
                post_mean_scalar = stats.get('mean', 0.0)  # Scalar mean across all params
                post_std_scalar = stats.get('std', 1.0)     # Scalar std across all params
                
                # Ensure scalars are float (not array, tensor, or other types)
                if hasattr(post_mean_scalar, '__len__'):
                    post_mean_scalar = float(np.mean(post_mean_scalar)) if len(post_mean_scalar) > 0 else 0.0
                if hasattr(post_std_scalar, '__len__'):
                    post_std_scalar = float(np.mean(post_std_scalar)) if len(post_std_scalar) > 0 else 1.0
                
                post_mean_scalar = float(post_mean_scalar)
                post_std_scalar = float(post_std_scalar)
                
                #  (Jan 2): Use posterior std DIRECTLY as uncertainty scale
                # posterior_std tells us how much Neural PE's posterior spread
                # Larger spread = larger hierarchical bias → need bigger corrections
                # DO NOT clamp - preserve full distribution information
                # With large posterior_std (e.g., 300+), we need proportionally large biases
                uncertainty_scale = max(post_std_scalar, 0.5) if post_std_scalar > 0 else 1.0
                # Remove upper clamp - let it scale with actual posterior uncertainty
                
                # self.logger.debug(f"[DATASET] Sample {idx}: post_std_scalar={post_std_scalar:.4f}, uncertainty_scale={uncertainty_scale:.4f}")
                
                estimated_params = true_params.copy()
                bias_magnitudes = []  # DEBUG: Track bias magnitudes
                for j, param_name in enumerate(self.param_names):
                    if param_name in ['mass_1', 'mass_2']:
                        # Mass: 10-20% relative error, scaled by posterior uncertainty
                        rel_bias = np.random.normal(0, 0.12 * uncertainty_scale)
                        estimated_params[j] = true_params[j] * (1.0 + rel_bias)
                        bias_magnitudes.append((param_name, abs(true_params[j] - estimated_params[j])))
                    elif param_name == 'luminosity_distance':
                        # Distance: 10-20% relative error, scaled by posterior uncertainty
                        rel_bias = np.random.normal(0, 0.15 * uncertainty_scale)
                        if float(abs(true_params[j])) > 1.0:
                            estimated_params[j] = true_params[j] * (1.0 + rel_bias)
                        else:
                            estimated_params[j] = true_params[j]
                    elif param_name == 'geocent_time':
                        # Time: ±10ms bias
                        bias = np.random.normal(0, 0.01 * uncertainty_scale)
                        estimated_params[j] = true_params[j] + bias
                    elif param_name in ['ra', 'dec']:
                        # Sky: ±0.1 rad bias
                        bias = np.random.normal(0, 0.06 * uncertainty_scale)
                        estimated_params[j] = true_params[j] + bias
                    else:
                        # Spins, angles: 10-20% relative error
                        if float(abs(true_params[j])) > 1e-3:
                            rel_bias = np.random.normal(0, 0.12 * uncertainty_scale)
                            estimated_params[j] = true_params[j] * (1.0 + rel_bias)
                        else:
                            bias = np.random.normal(0, 0.10 * uncertainty_scale)
                            estimated_params[j] = true_params[j] + bias
                
                # ✅ FIX #1: Normalize corrections to [-1, 1] space matching network predictions
                true_correction = np.zeros_like(estimated_params)
                for j, param_name in enumerate(self.param_names):
                    # Use safe bounds retrieval with sensible defaults
                    if self.physics_bounds and param_name in self.physics_bounds:
                        bounds = self.physics_bounds[param_name]
                        rng = bounds.get('max_value', 1.0) - bounds.get('min_value', -1.0)
                    else:
                        # Default bounds if not available
                        rng = 2.0  # Assume [-1, 1] range by default
                    # Normalize: physical_correction / parameter_range
                    true_correction[j] = (true_params[j] - estimated_params[j]) / max(rng, 1e-6)
            else:
                # ✅ Fallback when posterior stats not available
                # Generate realistic biased estimates with LARGER magnitudes (10-20% error)
                if idx < 3:
                    self.logger.info(f"[DATASET] Sample {idx}: NOT in posterior_stats, using fallback")
                self.logger.debug(f"Sample {idx} not in posterior stats. Generating realistic biases (fallback).")
                
                estimated_params = true_params.copy()
                for j, param_name in enumerate(self.param_names):
                    if param_name in ['mass_1', 'mass_2']:
                        # Mass: 10-20% relative error
                        rel_bias = np.random.normal(0, 0.12)
                        estimated_params[j] = true_params[j] * (1.0 + rel_bias)
                    elif param_name == 'luminosity_distance':
                        # Distance: 10-20% relative error
                        rel_bias = np.random.normal(0, 0.15)
                        if abs(true_params[j]) > 1.0:
                            estimated_params[j] = true_params[j] * (1.0 + rel_bias)
                        else:
                            estimated_params[j] = true_params[j]
                    elif param_name == 'geocent_time':
                        # Time: ±10ms bias (larger)
                        bias = np.random.normal(0, 0.01)
                        estimated_params[j] = true_params[j] + bias
                    elif param_name in ['ra', 'dec']:
                        # Sky: ±0.1 rad bias (larger)
                        bias = np.random.normal(0, 0.06)
                        estimated_params[j] = true_params[j] + bias
                    else:
                        # Spins, angles: 10-20% relative error
                        if abs(true_params[j]) > 1e-3:
                            rel_bias = np.random.normal(0, 0.12)
                            estimated_params[j] = true_params[j] * (1.0 + rel_bias)
                        else:
                            bias = np.random.normal(0, 0.10)
                            estimated_params[j] = true_params[j] + bias
                
                # ✅ FIX #1: Normalize corrections to [-1, 1] space matching network predictions
                true_correction = np.zeros_like(estimated_params)
                for j, param_name in enumerate(self.param_names):
                    # Use safe bounds retrieval with sensible defaults
                    if self.physics_bounds and param_name in self.physics_bounds:
                        bounds = self.physics_bounds[param_name]
                        rng = bounds.get('max_value', 1.0) - bounds.get('min_value', -1.0)
                    else:
                        # Default bounds if not available
                        rng = 2.0  # Assume [-1, 1] range by default
                    # Normalize: physical_correction / parameter_range
                    true_correction[j] = (true_params[j] - estimated_params[j]) / max(rng, 1e-6)
                
                # ✅ DEBUG: Log correction magnitude
                if idx == 0:
                    tc_mean = np.mean(np.abs(true_correction))
                    tc_max = np.max(np.abs(true_correction))
                    self.logger.info(f"[DATASET DEBUG] Sample 0 true_correction: mean={tc_mean:.9f}, max={tc_max:.9f}, fallback={'posterior_stats' not in self.posterior_stats}")
            
            # ✅ Normalize parameters (estimated params used for context only, not direct correction target)
            # Normalization is for context tensor encoding
            # 11 parameters: mass_1, mass_2, luminosity_distance, ra, dec, theta_jn, psi, phase, geocent_time, a_1, a_2
            param_min = np.array([-100, -100, 10, -np.pi, -np.pi, -1, -1, -1, -1, 0.0, 0.0])
            param_max = np.array([100, 100, 5000, np.pi, np.pi, 1, 1, 1, 1, 0.99, 0.99])
            
            # Clamp parameters to valid ranges before normalization
            estimated_params_clamped = np.clip(estimated_params, param_min, param_max)
            normalized_params = 2 * (estimated_params_clamped - param_min) / (param_max - param_min) - 1
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
            import traceback
            self.logger.error(f"❌ Error loading sample {idx}: {e}")
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
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
                 device: str = 'cpu', logger: logging.Logger = None, use_wandb: bool = False,
                 priority_net_path: str = None, neural_pe_path: str = None, 
                 resume_checkpoint: str = None, scheduler_type: str = 'cosine'):
        """Initialize trainer
        
        Args:
            bias_corrector: BiasCorrector module to train
            config: Training configuration
            device: Device to train on ('cpu' or 'cuda')
            logger: Logger instance
            use_wandb: Whether to use Weights & Biases logging
            priority_net_path: Path to frozen PriorityNet checkpoint
            neural_pe_path: Path to frozen NeuralPE checkpoint
            resume_checkpoint: Path to checkpoint to resume from
            scheduler_type: 'cosine' or 'plateau' (default: 'cosine')
        """
        self.bias_corrector = bias_corrector
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.resume_checkpoint = resume_checkpoint
        self.scheduler_type = scheduler_type
        self.start_epoch = 0
        
        self.bias_corrector.to(device)
        
        # Load and freeze PriorityNet
        self.priority_net = None
        if priority_net_path:
            try:
                self.priority_net = PriorityNet()
                checkpoint = torch.load(priority_net_path, map_location=device, weights_only=False)
                
                # Handle both dict checkpoints and direct state_dicts
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Load with strict=False to handle architecture changes
                self.priority_net.load_state_dict(state_dict, strict=False)
                self.priority_net.to(device)
                self.priority_net.eval()
                # Freeze all parameters
                for param in self.priority_net.parameters():
                    param.requires_grad = False
                self.logger.info(f"✅ Loaded frozen PriorityNet from {priority_net_path}")
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to load PriorityNet: {str(e)[:200]}...")
                self.logger.info("   Training will continue without PriorityNet (optional component)")
                self.priority_net = None
        
        # Load and freeze NeuralPE
        self.neural_pe = None
        if neural_pe_path:
            try:
                # Load checkpoint first to extract training config
                checkpoint = torch.load(neural_pe_path, map_location=device, weights_only=False)
                
                # Extract saved training config to match checkpoint architecture
                saved_config = checkpoint.get('config', {})
                
                # ✅ CRITICAL: Use param_names from neural_posterior section (includes spins a1, a2)
                neural_posterior_config = saved_config.get('neural_posterior', {})
                param_names = neural_posterior_config.get('param_names', [
                    'mass_1', 'mass_2', 'luminosity_distance',
                    'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time',
                    'a1', 'a2' 
                ])
                
                # Get context_dim from neural_posterior config or default
                saved_context_dim = neural_posterior_config.get('context_dim', 768)
                
                # Build config for NeuralPE initialization to match checkpoint
                # Flow config is nested under neural_posterior section
                flow_config = neural_posterior_config.get('flow_config', {})
                neural_pe_config = {
                    'context_dim': saved_context_dim,
                    'flow_type': neural_posterior_config.get('flow_type', 'nsf'),
                    'num_layers': flow_config.get('num_layers', 12),
                    'hidden_features': flow_config.get('hidden_features', 256),
                    'num_bins': flow_config.get('num_bins', 16),
                    'tail_bound': flow_config.get('tail_bound', 3.0),
                    'dropout': flow_config.get('dropout', 0.15)
                }
                
                # Use None for priority_net_path if not available
                priority_net_for_pe = priority_net_path if self.priority_net is not None else None
                
                self.neural_pe = OverlapNeuralPE(
                    param_names=param_names,
                    priority_net_path=priority_net_for_pe,
                    config=neural_pe_config
                )
                
                # Extract state_dict
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Load with strict=False to handle flow architecture differences
                # (checkpoint may have different number of flow transforms)
                missing, unexpected = self.neural_pe.load_state_dict(state_dict, strict=False)
                if unexpected:
                    self.logger.debug(f"Unexpected keys in checkpoint (ignored): {len(unexpected)} keys")
                if missing:
                    self.logger.debug(f"Missing keys in current model (using defaults): {len(missing)} keys")
                self.neural_pe.to(device)
                self.neural_pe.eval()
                # Freeze all parameters
                for param in self.neural_pe.parameters():
                    param.requires_grad = False
                self.logger.info(f"✅ Loaded frozen NeuralPE from {neural_pe_path}")
                self.logger.info(f"   Context dim: {saved_context_dim}, Flow: {neural_pe_config['flow_type']}, Layers: {neural_pe_config['num_layers']}")
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to load NeuralPE: {str(e)[:300]}...")
                self.logger.info("   Training will continue without NeuralPE (optional component)")
                self.neural_pe = None
        
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
        
        # Setup scheduler - ✅ ADDED: Support both cosine and plateau schedulers
        if self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=1e-7
            )
        else:  # 'plateau' - reduce LR when loss plateaus
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=False,
                min_lr=1e-7
            )
        
        # ✅ ADDED: Resume from checkpoint if provided
        if self.resume_checkpoint and Path(self.resume_checkpoint).exists():
            self._load_checkpoint(self.resume_checkpoint)
        
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
            # ✅ BIAS FIXING METRICS
            'train_bias_before': [],
            'train_bias_after': [],
            'train_bias_improvement_pct': [],
            'train_bias_reduction_ratio': [],
            'val_bias_before': [],
            'val_bias_after': [],
            'val_bias_improvement_pct': [],
            'val_bias_reduction_ratio': [],
            # ✅ NEW VALIDATION-ONLY DISTANCE METRICS
            'val_distance_mae_before': [],
            'val_distance_mae_after': [],
            'val_distance_improvement': [],
            'val_distance_bias_signed': [],
            'val_ci_coverage_68': [],
            'val_rejection_rate': [],
            'val_high_unc_fraction': [],
            'gradient_norms': [],
            'train_val_gap': []
        }
        
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.bias_corrector.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.bias_corrector.load_state_dict(checkpoint)
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.start_epoch = checkpoint.get('epoch', 0)
            
            # Load training history if available
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            
            self.logger.info(f"✅ Resumed from checkpoint: {checkpoint_path}")
            self.logger.info(f"   Starting from epoch {self.start_epoch + 1}")
            if 'training_history' in checkpoint:
                self.logger.info(f"   Restored training history with {len(self.training_history['train_losses'])} epochs")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load checkpoint {checkpoint_path}: {e}")
            self.logger.info("   Starting fresh training")
    
    def _save_checkpoint(self, checkpoint_path: str, epoch: int):
        """Save checkpoint for resuming training
        
        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch number
        """
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.bias_corrector.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
                'training_history': self.training_history,
            }
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def loss_function(self, pred_corrections: torch.Tensor,
                     pred_uncertainties: torch.Tensor,
                     true_corrections: torch.Tensor,
                     quality_weights: torch.Tensor,
                     ) -> Tuple[torch.Tensor, Dict]:
        """
         (Dec 9, 2025): Removed variance penalty explosion
        
        Problem: magnitude_loss = (pred_std - true_std)^2 was EXPLODING
        - pred_std ≈ 0.02 (tiny predictions)
        - true_std ≈ 74 (large targets)
        - magnitude_loss ≈ (0.02 - 74)^2 ≈ 5476 (CATASTROPHIC!)
        
        Solution: Replace variance penalty with DIRECT MSE on predictions
        This rewards the model for outputting LARGER corrections, not penalizing small ones.
        
        Target: 68% acceptance rate (P(|error| < σ) for well-calibrated Gaussian)
        
        Expects:
        - pred_corrections: [batch_size, num_params] from BiasEstimator
        - pred_uncertainties: [batch_size, num_params] from BiasEstimator  
        - true_corrections: [batch_size, num_params] ground truth corrections
        - quality_weights: [batch_size, 1] or [batch_size]
        """
        # Ensure shapes match - expand true_corrections if needed
        if true_corrections.dim() == 1:
            true_corrections = true_corrections.unsqueeze(1)  # [batch] → [batch, 1]
        
        # Expand quality weights to match
        if quality_weights.dim() == 1:
            quality_weights = quality_weights.unsqueeze(1)  # [batch] → [batch, 1]
        
        errors = torch.abs(pred_corrections - true_corrections)
        true_mag = torch.mean(torch.abs(true_corrections)).detach()
        pred_mag = torch.mean(torch.abs(pred_corrections))
        
        # ✅ PRIMARY (50%): Direct MSE loss - reward predictions toward true corrections
        # This is the CORE learning signal: pred_corrections should match true_corrections
        mse_loss = torch.mean((pred_corrections - true_corrections) ** 2)
        
        # ✅ SECONDARY (30%): Calibration loss - ensure uncertainty covers errors
        # For well-calibrated Gaussian: P(|error| < σ) = 0.68 exactly
        # Penalize overconfident (unc too small) and underconfident (unc too large)
        overconfident = torch.relu(0.68 * pred_uncertainties - errors)
        underconfident = torch.relu(errors - 1.5 * pred_uncertainties)
        calibration_loss = torch.mean(overconfident + underconfident)
        
        # ✅ TERTIARY (15%): Uncertainty bounds regularization
        # Keep uncertainties in reasonable range [0.01, 20.0]
        # Allows model to cover large correction outliers (error up to 1000+)
        # Prevents collapse (unc→0) and mild explosion (unc→∞)
        unc_bounds_loss = torch.mean(
            torch.relu(pred_uncertainties - 20.0) +     # Penalty if unc > 20.0
            torch.relu(0.01 - pred_uncertainties)       # Penalty if unc < 0.01
        )
        
        # ✅ QUATERNARY (5%): Magnitude incentive - bonus when predictions are non-zero
        # Prevent collapse to pred ≈ 0
        # Simple: penalize if mean(|pred|) < 0.1 * mean(|true|)
        pred_mag_mean = torch.mean(torch.abs(pred_corrections))
        true_mag_mean = torch.mean(torch.abs(true_corrections))
        min_pred_threshold = 0.1 * true_mag_mean
        magnitude_incentive = torch.relu(min_pred_threshold - pred_mag_mean)
        
        # Final loss: 4-term, direct MSE as PRIMARY
        # MSE is the ONLY backprop signal directly pushing toward true corrections
        # Everything else guides HOW to correct (calibration, bounds, incentives)
        loss = (
            0.50 * mse_loss +                # ← PRIMARY: Direct prediction matching
            0.30 * calibration_loss +        # Calibration guides uncertainty
            0.15 * unc_bounds_loss +         # Bounds prevent collapse/explosion
            0.05 * magnitude_incentive       # Magnitude incentive prevents pred→0
        )
        
        # Legacy metrics for compatibility with existing tracking code
        uncertainty_reg = torch.tensor(0.0, device=pred_corrections.device)
        magnitude_penalty = magnitude_incentive  # Was variance penalty, now incentive
        quality_weighted = torch.tensor(0.0, device=pred_corrections.device)
        correlation_penalty = torch.tensor(0.0, device=pred_corrections.device)  # Computed at epoch level
        large_unc_penalty = torch.relu(pred_uncertainties.mean() - 2.0)
        magnitude_expansion_penalty = torch.tensor(0.0, device=pred_corrections.device)
        target_magnitude = 1.0
        mae_loss = torch.mean(torch.abs(pred_corrections - true_corrections))  # For logging only
        
        # ✅ REMOVED: acceptance_penalty was causing loss to be disconnected from MSE
        # Now acceptance improvement comes naturally from MSE reduction
        
        # Compute accuracy metrics
        residuals = pred_corrections - true_corrections
        abs_errors = torch.abs(residuals)
        mae = torch.mean(abs_errors).item()
        rmse = torch.sqrt(torch.mean(residuals ** 2)).item()
        max_error = torch.max(abs_errors).item()
        mean_uncertainty = torch.mean(pred_uncertainties).item()
        
        # ✅ FIXED (Nov 30): Acceptance metrics based on CALIBRATION, not quantile
        # Acceptance = fraction of predictions where |error| < predicted_uncertainty (68% target)
        # This is the proper Gaussian calibration metric
        # Initialize debug metrics
        err_mean = 0.0
        err_max = 0.0
        unc_mean = 0.0
        unc_min = 0.0
        unc_max = 0.0
         
        if pred_uncertainties.numel() > 0:
             # Compute errors
            errors = torch.abs(residuals)
            
            # ✅ DEBUG: Check if uncertainties are too large
            err_mean = torch.mean(errors).item()
            err_max = torch.max(errors).item()
            unc_mean = torch.mean(pred_uncertainties).item()
            unc_min = torch.min(pred_uncertainties).item()
            unc_max = torch.max(pred_uncertainties).item()
            
            # ✅ FIX: Use CLAMPED uncertainties for acceptance metric
            # Raw pred_uncertainties may be too small, clamp to reasonable minimum
            min_uncertainty_target = 0.01  # Minimum uncertainty threshold for acceptance
            within_bounds = errors < (torch.clamp(pred_uncertainties, min=min_uncertainty_target) + 1e-8)
            acceptance_rate = float(torch.mean(within_bounds.float()).item())
            
            # ✅ DEBUG: Log this
            # (Only log once per batch, not in every call)
            
            # Average confidence (inverse of uncertainty) - for monitoring only
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
            
            # Check for NaN/Inf values
            if np.isnan(pred_flat).any() or np.isinf(pred_flat).any():
                correction_correlation = 0.0
            elif np.isnan(true_flat).any() or np.isinf(true_flat).any():
                correction_correlation = 0.0
            else:
                # Check for zero variance (early training: all predictions same)
                pred_std = np.std(pred_flat)
                true_std = np.std(true_flat)
                
                if pred_std > 1e-6 and true_std > 1e-6:
                    corr, _ = pearsonr(pred_flat, true_flat)
                    # Handle NaN result from pearsonr
                    if np.isnan(corr):
                        correction_correlation = 0.0
                    else:
                        correction_correlation = float(corr)
                else:
                    # Constant predictions or targets - correlation undefined
                    correction_correlation = 0.0
        except:
            correction_correlation = 0.0
        
        return loss, {
            # Loss components (what we're actually optimizing) - 4-PART LOSS
            'mse': mse_loss.item(),                              # 50% - PRIMARY: Direct prediction matching
            'calibration_loss': calibration_loss.item(),         # 30% - Calibration guides uncertainty
            'unc_bounds_loss': unc_bounds_loss.item(),           # 15% - Bounds prevent collapse/explosion
            'magnitude_penalty': magnitude_incentive.item(),      # 5% - Magnitude incentive prevents pred→0
            # Legacy metrics for backward compatibility
            'uncertainty_reg': uncertainty_reg.item(),
            'quality_weighted': quality_weighted.item(),
            'correlation_penalty': correlation_penalty.item(),
            'large_unc_penalty': large_unc_penalty.item(),
            'magnitude_expansion_penalty': magnitude_expansion_penalty.item(),
            'total_loss': loss.item(),
            # Accuracy metrics
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
            'correction_correlation': correction_correlation,
            # ✅ DEBUG: Error vs Uncertainty comparison
             'debug_err_mean': err_mean,
             'debug_err_max': err_max,
             'debug_unc_mean': unc_mean,
             'debug_unc_min': unc_min,
             'debug_unc_max': unc_max,
            # Magnitude tracking (for debugging output capping)
            'true_mag': true_mag.item(),
            'pred_mag': pred_mag.item(),
            'target_magnitude': target_magnitude,  # Target for magnitude expansion
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
            'correlation_penalty': [],
            'large_unc_penalty': [],
            'magnitude_expansion_penalty': [],
            'total_loss': [],
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
            'correction_correlation': [],
            'true_mag': [],
            'pred_mag': [],
            'target_magnitude': []
            }
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            param_tensors = batch['param_tensor'].to(self.device)
            context_tensors = batch['context_tensor'].to(self.device)
            true_corrections = batch['true_correction'].to(self.device)
            quality_weights = batch['signal_quality'].to(self.device)
            
            # ✅ DEBUG: Check if targets have meaningful magnitude
            if batch_idx == 0:
                tc_mean = torch.abs(true_corrections).mean().item()
                tc_max = torch.abs(true_corrections).max().item()
                tc_std = torch.abs(true_corrections).std().item()
                self.logger.info(f"[BATCH 0 DEBUG] true_corrections: mean={tc_mean:.9f}, max={tc_max:.9f}, std={tc_std:.9f}")
            
            # Forward pass - now returns 3 values (corrections, uncertainties, variance_scales)
            pred_corrections, pred_uncertainties, pred_variance_scales = self.bias_corrector.bias_estimator(
                param_tensors, context_tensors
            )
            
            # ✅ DEBUG: Check if predictions have reasonable magnitude
            if batch_idx == 0:
                pc_mean = torch.abs(pred_corrections).mean().item()
                pc_max = torch.abs(pred_corrections).max().item()
                unc_mean = pred_uncertainties.mean().item()
                unc_max = pred_uncertainties.max().item()
                vs_mean = pred_variance_scales.mean().item()
                vs_min = pred_variance_scales.min().item()
                vs_max = pred_variance_scales.max().item()
                self.logger.info(f"[BATCH 0 DEBUG] pred_corrections: mean={pc_mean:.9f}, max={pc_max:.9f}")
                self.logger.info(f"[BATCH 0 DEBUG] pred_uncertainties: mean={unc_mean:.9f}, max={unc_max:.9f}")
                self.logger.info(f"[BATCH 0 DEBUG] variance_scales: mean={vs_mean:.4f}, range=[{vs_min:.4f}, {vs_max:.4f}]")
            
            # Compute loss
            loss, metrics = self.loss_function(
                pred_corrections, pred_uncertainties,
                true_corrections, quality_weights
            )
            
            # ✅ DEBUG: Check loss components (all of them!)
            if batch_idx == 0:
                loss_components = {k: v for k, v in metrics.items() if k in [
                    'mse', 'uncertainty_reg', 'magnitude_penalty', 'quality_weighted',
                    'correlation_penalty', 'large_unc_penalty', 'magnitude_expansion_penalty', 'total_loss'
                ]}
                self.logger.info(f"[BATCH 0 DEBUG] Loss components: {loss_components}")
                
                mag_info = {k: v for k, v in metrics.items() if k in ['true_mag', 'pred_mag', 'correction_correlation']}
                self.logger.info(f"[BATCH 0 DEBUG] Magnitude & Corr: true_mag={mag_info.get('true_mag', 0):.6f}, pred_mag={mag_info.get('pred_mag', 0):.6f}, corr={mag_info.get('correction_correlation', 0):.6f}")
            
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
        
        # ✅ BIAS FIXING METRICS (DISABLED)
        # Note: Training data only contains corrections (true_params - estimated_params)
        # Does not contain estimated_params or true_params separately
        # Would need changes to data loader to track before/after bias
        # For now, loss components provide sufficient monitoring
        
        return avg_loss, avg_metrics
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            avg_loss: Average validation loss
            metrics: Dict of metrics including acceptance metrics + new validation-only metrics
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
        
        # ✅ NEW VALIDATION-ONLY METRICS (no gradients)
        distance_mae_before = []  # MAE before correction
        distance_mae_after = []   # MAE after correction
        distance_bias_signed = []  # Mean signed distance bias (Mpc)
        ci_coverage_68 = []        # 68% credible interval coverage
        rejection_count = 0        # Count of low-confidence predictions
        high_unc_count = 0         # Count of high-uncertainty predictions
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                param_tensors = batch['param_tensor'].to(self.device)
                context_tensors = batch['context_tensor'].to(self.device)
                true_corrections = batch['true_correction'].to(self.device)
                quality_weights = batch['signal_quality'].to(self.device)
                
                # Now returns 3 values (corrections, uncertainties, variance_scales)
                pred_corrections, pred_uncertainties, pred_variance_scales = self.bias_corrector.bias_estimator(
                    param_tensors, context_tensors
                )
                
                loss, metrics = self.loss_function(
                    pred_corrections, pred_uncertainties,
                    true_corrections, quality_weights
                )
                
                val_losses.append(loss.item())
                for key in metrics_sum:
                    metrics_sum[key].append(metrics[key])
                
                # ✅ NEW VALIDATION-ONLY METRICS (no backward pass)
                # These use parameter_names[2] = 'luminosity_distance' (index 2)
                distance_idx = 2  # luminosity_distance is at index 2
                
                # Extract distance corrections (in normalized space)
                true_dist_corr = true_corrections[:, distance_idx].cpu().numpy()
                pred_dist_corr = pred_corrections[:, distance_idx].cpu().numpy()
                uncertainties = pred_uncertainties.cpu().numpy()
                
                # MAE before correction (true error magnitude)
                distance_mae_before.append(np.mean(np.abs(true_dist_corr)))
                
                # MAE after correction (residual error after applying prediction)
                # Residual = true_correction - pred_correction
                residual_dist = true_dist_corr - pred_dist_corr
                distance_mae_after.append(np.mean(np.abs(residual_dist)))
                
                # Mean signed distance bias (positive = overestimate distance)
                distance_bias_signed.append(np.mean(residual_dist))
                
                # 68% CI coverage: proportion of samples where |true_error| < 1.0 × uncertainty
                # (1σ covers ~68% of normal distribution)
                covered = np.sum(np.abs(true_dist_corr) < (1.0 * uncertainties[:, 0]))
                coverage = covered / len(true_dist_corr)
                ci_coverage_68.append(coverage)
                
                # Rejection rate: confidence < 0.5 (inverse of uncertainty)
                confidence = 1.0 / (uncertainties[:, 0] + 1e-6)
                rejection_count += np.sum(confidence < 0.5)
                
                # High-uncertainty fraction: uncertainty > 75th percentile
                high_unc_count += np.sum(uncertainties[:, 0] > np.percentile(uncertainties[:, 0], 75))
        
        avg_loss = np.mean(val_losses)
        avg_metrics = {key: np.mean(values) for key, values in metrics_sum.items()}
        
        # ✅ ADD NEW VALIDATION-ONLY METRICS
        avg_metrics['distance_mae_before'] = np.mean(distance_mae_before)
        avg_metrics['distance_mae_after'] = np.mean(distance_mae_after)
        avg_metrics['distance_improvement'] = avg_metrics['distance_mae_before'] - avg_metrics['distance_mae_after']
        avg_metrics['distance_bias_signed'] = np.mean(distance_bias_signed)
        avg_metrics['ci_coverage_68'] = np.mean(ci_coverage_68)
        avg_metrics['rejection_rate'] = rejection_count / (len(val_loader.dataset) + 1e-6)
        avg_metrics['high_unc_fraction'] = high_unc_count / (len(val_loader.dataset) + 1e-6)
        
        return avg_loss, avg_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, 
              checkpoint_dir: str = None) -> Dict:
        """
        Main training loop with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            checkpoint_dir: Directory to save periodic checkpoints (optional)
            
        Returns:
            training_history: Dict with training history
        """
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        best_val_acceptance = 0.0  # Track best acceptance rate
        best_val_correlation = -1.0  # Track best correlation
        patience_counter = 0
        patience_metric = 'loss'  # 'loss', 'acceptance', or 'correlation'
        best_model_state = None
        
        for epoch in range(self.start_epoch, self.epochs):
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
            # ✅ BIAS FIXING METRICS
            self.training_history['train_bias_before'].append(train_metrics.get('bias_before_mean', 0))
            self.training_history['train_bias_after'].append(train_metrics.get('bias_after_mean', 0))
            self.training_history['train_bias_improvement_pct'].append(train_metrics.get('bias_improvement_pct', 0))
            self.training_history['train_bias_reduction_ratio'].append(train_metrics.get('bias_reduction_ratio', 1.0))
             
             # Validation phase
            val_loss = None
            val_metrics = {}
            val_acceptance = 0.0
            val_correlation = 0.0
            
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
                # ✅ BIAS FIXING METRICS
                self.training_history['val_bias_before'].append(val_metrics.get('bias_before_mean', 0))
                self.training_history['val_bias_after'].append(val_metrics.get('bias_after_mean', 0))
                self.training_history['val_bias_improvement_pct'].append(val_metrics.get('bias_improvement_pct', 0))
                self.training_history['val_bias_reduction_ratio'].append(val_metrics.get('bias_reduction_ratio', 1.0))
                
                # ✅ NEW VALIDATION-ONLY METRICS (distance calibration)
                self.training_history['val_distance_mae_before'].append(val_metrics.get('distance_mae_before', 0))
                self.training_history['val_distance_mae_after'].append(val_metrics.get('distance_mae_after', 0))
                self.training_history['val_distance_improvement'].append(val_metrics.get('distance_improvement', 0))
                self.training_history['val_distance_bias_signed'].append(val_metrics.get('distance_bias_signed', 0))
                self.training_history['val_ci_coverage_68'].append(val_metrics.get('ci_coverage_68', 0))
                self.training_history['val_rejection_rate'].append(val_metrics.get('rejection_rate', 0))
                self.training_history['val_high_unc_fraction'].append(val_metrics.get('high_unc_fraction', 0))
                
                # ✅ CORRECTED EARLY STOPPING (Dec 2, 2025, Epoch-Level)
                # Only use reliable metrics: Loss and Acceptance
                # Correlation is monitored but not used for early stopping (batch-level noise)
                val_acceptance = val_metrics.get('acceptance_rate', 0)
                val_correlation = val_metrics.get('correction_correlation', 0)
             
            # Train-Val gap
            train_val_gap = train_loss - val_loss if val_loss is not None else 0.0
            self.training_history['train_val_gap'].append(train_val_gap)
            
            improved = False
            reason = ""
            
            # Primary metric: Loss (most reliable, only metric in backward pass)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True
                patience_metric = 'loss'
                reason = f"loss {best_val_loss:.6f}"
            
            # Secondary metric: Acceptance rate (monitor but don't stop early on it)
            # Only update if loss didn't improve
            if not improved and val_acceptance > best_val_acceptance:
                best_val_acceptance = val_acceptance
                improved = True
                patience_metric = 'acceptance'
                reason = f"acceptance {best_val_acceptance:.2%}"
            
            # Tertiary: Track correlation progress (monitor only, not decision criterion)
            if val_correlation > best_val_correlation and val_correlation > 0.0:
                best_val_correlation = val_correlation
            
            if improved:
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in self.bias_corrector.state_dict().items()}
                
                # ✅ SAVE BEST MODEL WITH EPOCH NUMBER
                if checkpoint_dir:
                    best_model_path = checkpoint_dir / f'best_model.pth'
                    self._save_checkpoint(str(best_model_path), epoch)
                    self.logger.info(f"[Epoch {epoch+1:3d}] 🎯 NEW BEST MODEL saved: {best_model_path} | Metric: {reason}")
            else:
                patience_counter += 1
            
            # ✅ ADDED: Step scheduler based on type
            if self.scheduler_type == 'plateau' and val_loss is not None:
                self.scheduler.step(val_loss)
            elif self.scheduler_type == 'cosine':
                self.scheduler.step()
            
            # ✅ ADDED: Save periodic checkpoints (every 10 epochs)
            if checkpoint_dir and (epoch + 1) % 10 == 0:
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pth'
                self._save_checkpoint(str(checkpoint_path), epoch)
                self.logger.debug(f"[Epoch {epoch+1:3d}] Saved periodic checkpoint: {checkpoint_path}")
            
            # Logging
            if (epoch + 1) % 1 == 0 or epoch == 0:
                msg = f"Epoch {epoch+1:3d}/{self.epochs} | Train Loss: {train_loss:.6f} | MAE: {train_metrics.get('mae', 0):.6f} | RMSE: {train_metrics.get('rmse', 0):.6f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.6f} | Val MAE: {val_metrics.get('mae', 0):.6f} | Patience: {patience_counter}/{self.patience}"
                self.logger.info(msg)
            
            # ✅ Log acceptance metrics with epoch number
            self.logger.info(f"[Epoch {epoch+1:3d}] ✓ Acceptance Rate: Train={train_metrics.get('acceptance_rate', 0):.2%} | Val={val_metrics.get('acceptance_rate', 0):.2%}" if val_loss is not None else f"[Epoch {epoch+1:3d}] ✓ Acceptance Rate: Train={train_metrics.get('acceptance_rate', 0):.2%}")
            self.logger.info(f"[Epoch {epoch+1:3d}] ⚡ Avg Confidence: Train={train_metrics.get('avg_confidence', 0):.4f} | Val={val_metrics.get('avg_confidence', 0):.4f}" if val_loss is not None else f"[Epoch {epoch+1:3d}] ⚡ Avg Confidence: Train={train_metrics.get('avg_confidence', 0):.4f}")
            self.logger.info(f"[Epoch {epoch+1:3d}] 📊 Correction Correlation: Train={train_metrics.get('correction_correlation', 0):.4f} | Val={val_metrics.get('correction_correlation', 0):.4f}" if val_loss is not None else f"[Epoch {epoch+1:3d}] 📊 Correction Correlation: Train={train_metrics.get('correction_correlation', 0):.4f}")
            
            # ✅ Log BIAS FIXING METRICS with epoch number
            if 'bias_before_mean' in train_metrics and 'bias_after_mean' in train_metrics:
                train_improvement = train_metrics.get('bias_improvement_pct', 0)
                train_ratio = train_metrics.get('bias_reduction_ratio', 1.0)
                self.logger.info(f"[Epoch {epoch+1:3d}] 🔧 BIAS FIXING (Train): Before={train_metrics.get('bias_before_mean', 0):.6f} → After={train_metrics.get('bias_after_mean', 0):.6f} | Improvement={train_improvement:.1f}% | Reduction={train_ratio:.2f}x")
            
            if val_loss is not None and 'bias_before_mean' in val_metrics and 'bias_after_mean' in val_metrics:
                val_improvement = val_metrics.get('bias_improvement_pct', 0)
                val_ratio = val_metrics.get('bias_reduction_ratio', 1.0)
                self.logger.info(f"[Epoch {epoch+1:3d}] 🔧 BIAS FIXING (Val):   Before={val_metrics.get('bias_before_mean', 0):.6f} → After={val_metrics.get('bias_after_mean', 0):.6f} | Improvement={val_improvement:.1f}% | Reduction={val_ratio:.2f}x")
            
            # ✅ NEW: Log distance-specific calibration metrics (validation-only)
            if val_loss is not None:
                dist_before = val_metrics.get('distance_mae_before', 0)
                dist_after = val_metrics.get('distance_mae_after', 0)
                dist_improve = val_metrics.get('distance_improvement', 0)
                dist_bias = val_metrics.get('distance_bias_signed', 0)
                ci_cov = val_metrics.get('ci_coverage_68', 0)
                rej_rate = val_metrics.get('rejection_rate', 0)
                high_unc = val_metrics.get('high_unc_fraction', 0)
                
                self.logger.info(f"[Epoch {epoch+1:3d}] 📏 DISTANCE: MAE Before={dist_before:.6f} → After={dist_after:.6f} | Improvement={dist_improve:.6f} | Bias={dist_bias:.6f} Mpc")
                self.logger.info(f"[Epoch {epoch+1:3d}] 📊 CALIBRATION: CI@68%={ci_cov:.2%} | Rejection={rej_rate:.2%} | High-Unc={high_unc:.2%}")
            
            # # ✅ DEBUG: Log error vs uncertainty magnitudes
            # if epoch < 3:  # Only log first 3 epochs for brevity
            #     debug_info = f"  🔍 DEBUG: Err[mean={train_metrics.get('debug_err_mean', 0):.9f}, max={train_metrics.get('debug_err_max', 0):.9f}] Unc[mean={train_metrics.get('debug_unc_mean', 0):.9f}, min={train_metrics.get('debug_unc_min', 0):.9f}]"
            #     self.logger.info(debug_info)
            
            # 🎯 Log to Weights & Biases
            if self.use_wandb:
                wandb_log = {
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/mae': train_metrics.get('mae', 0),
                    'train/rmse': train_metrics.get('rmse', 0),
                    'train/max_error': train_metrics.get('max_error', 0),
                    'train/mean_uncertainty': train_metrics.get('mean_uncertainty', 0),
                    'train/acceptance_rate': train_metrics.get('acceptance_rate', 0),
                    'train/avg_confidence': train_metrics.get('avg_confidence', 0),
                    'train/correction_correlation': train_metrics.get('correction_correlation', 0),
                    'train/correction_std': train_metrics.get('correction_std', 0),
                    'train/correction_bias': train_metrics.get('correction_bias', 0),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'gradient_norm': train_metrics.get('avg_gradient_norm', 0),
                }
                
                if val_loss is not None:
                    wandb_log.update({
                        'val/loss': val_loss,
                        'val/mae': val_metrics.get('mae', 0),
                        'val/rmse': val_metrics.get('rmse', 0),
                        'val/max_error': val_metrics.get('max_error', 0),
                        'val/mean_uncertainty': val_metrics.get('mean_uncertainty', 0),
                        'val/acceptance_rate': val_metrics.get('acceptance_rate', 0),
                        'val/avg_confidence': val_metrics.get('avg_confidence', 0),
                        'val/correction_correlation': val_metrics.get('correction_correlation', 0),
                        'val/correction_std': val_metrics.get('correction_std', 0),
                        'val/correction_bias': val_metrics.get('correction_bias', 0),
                        'train_val_gap': train_val_gap,
                        # ✅ BIAS FIXING METRICS
                        'train/bias_before': train_metrics.get('bias_before_mean', 0),
                        'train/bias_after': train_metrics.get('bias_after_mean', 0),
                        'train/bias_improvement_pct': train_metrics.get('bias_improvement_pct', 0),
                        'train/bias_reduction_ratio': train_metrics.get('bias_reduction_ratio', 1.0),
                        'val/bias_before': val_metrics.get('bias_before_mean', 0),
                        'val/bias_after': val_metrics.get('bias_after_mean', 0),
                        'val/bias_improvement_pct': val_metrics.get('bias_improvement_pct', 0),
                        'val/bias_reduction_ratio': val_metrics.get('bias_reduction_ratio', 1.0),
                    })
                
                wandb.log(wandb_log)
            
            # Early stopping check
            if patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                self.logger.info(f"  Metrics at stop:")
                self.logger.info(f"    - Acceptance: {val_acceptance:.2%} (target: 68%)")
                self.logger.info(f"    - Correlation: {val_correlation:.4f} (target: >0.5)")
                self.logger.info(f"    - Loss: {val_loss:.6f}")
                self.logger.info(f"  Last metric: {patience_metric}")
                if patience_counter - self.patience > 0:
                    self.logger.info(f"  (patience exceeded by {patience_counter - self.patience} epochs)")
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
    parser.add_argument('--patience', type=int, default=40,
                       help='Patience for early stopping (increased to 40 for acceptance convergence)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda or cpu)')
    parser.add_argument('--num_samples', type=int, default=20000,
                       help='Number of training samples to generate')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--output_dir', type=str, default='models/bias_corrector',
                       help='Output directory for checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--wandb', action='store_true',
                       help='Log metrics to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='posterflow',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Weights & Biases entity (username/team)')
    parser.add_argument('--data_path', type=str, default='data/dataset',
                       help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default="models/neuralpe2/best_model.pth",
                        help='Path to pre-trained Neural PE model')
    parser.add_argument('--priority_net_path', type=str, default='models/prioritynet/priority_net_best.pth',
                        help='Path to frozen PriorityNet checkpoint')
    parser.add_argument('--neural_pe_path', type=str, default='models/neuralpe/best_model.pth',
                        help='Path to frozen NeuralPE checkpoint')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'plateau'],
                        help='Learning rate scheduler type: cosine (smooth decay) or plateau (adaptive)')
    
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
    
    # Initialize Weights & Biases
    use_wandb = False
    if args.wandb:
        if WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"bias_corrector_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'patience': args.patience,
                    'device': args.device,
                    'seed': args.seed,
                    'data_path': args.data_path,
                    'model_path': args.model_path,
                }
            )
            use_wandb = True
            logger.info("✅ Weights & Biases logging initialized")
        else:
            logger.warning("⚠️ Weights & Biases not installed. Install with: pip install wandb")
    
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
    
    # Parameter names (11D: orbital + spins)
    # ✅ NOW USING 11D: Spins a1, a2 are generated in the dataset
    # This matches Neural PE 11D parameter space
    # NOTE: Dataset uses 'a1', 'a2' (not 'a_1', 'a_2') from parameter_sampler.py
    param_names = [
        'mass_1', 'mass_2', 'luminosity_distance',
        'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time',
        'a1', 'a2'  # ✅ FIXED: Use 'a1', 'a2' to match dataset field names
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
        seed=args.seed,
        physics_bounds=bias_corrector.physics_bounds  # ✅ FIX #1: Pass bounds for normalization
    )
    
    val_dataset = BiasDataset(
        data_dir=data_dir,
        param_names=param_names,
        context_dim=context_dim,
        split='validation',
        seed=args.seed,
        physics_bounds=bias_corrector.physics_bounds  # ✅ FIX #1: Pass bounds for normalization
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
        logger=logger,
        use_wandb=use_wandb,
        priority_net_path=args.priority_net_path if Path(args.priority_net_path).exists() else None,
        neural_pe_path=args.neural_pe_path if Path(args.neural_pe_path).exists() else None,
        resume_checkpoint=args.resume_checkpoint,  # ✅ ADDED
        scheduler_type=args.scheduler_type  # ✅ ADDED
    )
    
    logger.info("="*70)
    logger.info("Training Configuration:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.learning_rate}")
    logger.info(f"  Patience: {args.patience}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Scheduler: {args.scheduler_type} (cosine decay or plateau adaptive)")
    if trainer.start_epoch > 0:
        logger.info(f"  Resume: Yes (from epoch {trainer.start_epoch + 1})")
    else:
        logger.info(f"  Resume: No (fresh start)")
    logger.info("\n  Frozen Models:")
    if trainer.priority_net is not None:
        logger.info(f"    ✅ PriorityNet: {args.priority_net_path} (frozen, no gradients)")
    else:
        logger.info(f"    ⚠️  PriorityNet: Not loaded")
    if trainer.neural_pe is not None:
        logger.info(f"    ✅ NeuralPE: {args.neural_pe_path} (frozen, no gradients)")
    else:
        logger.info(f"    ⚠️  NeuralPE: Not loaded")
    logger.info("="*70)
    
    # ✅ ADDED: Create checkpoint directory inside output_dir
    checkpoint_dir = output_dir / 'checkpoints'
    
    # Train
    history = trainer.train(train_loader, val_loader, checkpoint_dir=str(checkpoint_dir))
    
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
    
    # Finish wandb run
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        logger.info("✅ Weights & Biases run finished")


if __name__ == '__main__':
    main()
