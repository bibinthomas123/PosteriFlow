#!/usr/bin/env python3
"""
Training script for OverlapNeuralPE with FlowMatching posterior.

Features:
- Multi-signal extraction with PriorityNet
- Flow Matching (OT-CFM) for posterior estimation
- RL-controlled adaptive complexity
- Bias correction and adaptive subtraction
- Comprehensive metrics and evaluation
- WandB integration for experiment tracking

Changes from RealNVP version:
- Flow type: 'flowmatching' instead of 'realnvp'
- Context dim: 512 (increased from 256)
- Fewer flow layers (4 instead of 8) - FlowMatching is more expressive
- Velocity-based training instead of affine coupling
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import logging
import time
import pickle
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from ahsd.models.overlap_neuralpe import OverlapNeuralPE
from experiments.train_priority_net import ChunkedGWDataLoader

# ✅ WANDB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  WandB not available - install with: pip install wandb")


def setup_logging(output_dir: Path, verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'training.log'

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


class OverlapGWDataset(Dataset):
    """Dataset for training OverlapNeuralPE with FlowMatching."""

    def __init__(self, data_loader: ChunkedGWDataLoader, param_names: List[str], config: Dict[str, Any]):
        self.param_names = param_names
        self.config = config
        self.logger = logging.getLogger(__name__)

        # ✅ Read augmentation config
        aug_config = config.get('data_augmentation', {})
        self.augmentation_enabled = aug_config.get('enabled', False)
        self.noise_scaling = aug_config.get('noise_scaling', [0.95, 1.05])
        self.time_shifts = aug_config.get('time_shifts', [-0.005, 0.005])
        self.apply_probability = aug_config.get('apply_probability', 0.3)

        # Load samples
        self.samples = list(data_loader.iter_all_samples())
        self.logger.info(f"✅ Loaded {len(self.samples)} samples from {data_loader.split}")
        if self.augmentation_enabled:
            self.logger.info(f"✅ Data augmentation enabled (p={self.apply_probability})")

    def __len__(self):
        return len(self.samples)

    def _apply_augmentation(self, strain_data: np.ndarray) -> np.ndarray:
        """Apply data augmentation to strain data."""
        if not self.augmentation_enabled:
            return strain_data

        # Only apply with probability
        if np.random.random() > self.apply_probability:
            return strain_data

        augmented = strain_data.copy()

        # Noise scaling
        noise_scale = np.random.uniform(self.noise_scaling[0], self.noise_scaling[1])
        augmented = augmented * noise_scale

        # Time shifts
        time_shift = np.random.uniform(self.time_shifts[0], self.time_shifts[1])
        if abs(time_shift) > 1e-6:
            shift_samples = int(time_shift * 4096)
            augmented = np.roll(augmented, shift_samples, axis=-1)

        return augmented

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract data
        strain_data = []
        for det in ['H1', 'L1']:
            det_strain = sample['detector_data'][det]['strain'].astype(np.float32)
            det_strain = self._apply_augmentation(det_strain)
            strain_data.append(det_strain)

        strain_tensor = torch.tensor(np.array(strain_data), dtype=torch.float32)

        # Extract parameters
        params_list = [sample['parameters'].get(pn, 0.0) for pn in self.param_names]
        params_tensor = torch.tensor(params_list, dtype=torch.float32)

        return {
            'strain': strain_tensor,
            'parameters': params_tensor,
            'sample_id': sample.get('sample_id', idx)
        }


class OverlapNeuralPETrainer:
    """Trainer for OverlapNeuralPE with FlowMatching."""

    def __init__(self, model: OverlapNeuralPE, config: Dict[str, Any], device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)

        # Training configuration
        self.learning_rate = config.get('training', {}).get('learning_rate', 1e-3)
        self.weight_decay = config.get('training', {}).get('weight_decay', 1e-5)
        self.max_epochs = config.get('training', {}).get('max_epochs', 50)
        self.grad_clip = config.get('training', {}).get('grad_clip', 1.0)
        self.log_interval = config.get('training', {}).get('log_interval', 100)

        # ✅ FlowMatching specific config
        self.flow_config = config.get('flow_config', {})
        self.solver_steps = self.flow_config.get('solver_steps', 10)

        # Optimizer
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            threshold_mode='rel',
            threshold=1e-3
        )

        # Metrics
        self.training_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        self.logger.info(f"✅ Trainer initialized: lr={self.learning_rate}, epochs={self.max_epochs}")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'total_loss': [],
            'nll': [],
            'physics_loss': [],
            'uncertainty_loss': []
        }

        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch+1}')
        for batch in pbar:
            strain_data = batch['strain'].to(self.device)
            parameters = batch['parameters'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            loss_dict = self.model.compute_loss(strain_data, parameters)

            # Backward pass
            loss_dict['total_loss'].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Update metrics
            for key, value in loss_dict.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value.detach().cpu().item())

            # Log
            self.training_step += 1
            if self.training_step % self.log_interval == 0:
                avg_loss = np.mean(epoch_metrics['total_loss'][-self.log_interval:])
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'grad_norm': f'{grad_norm:.3f}'})

            # Update model metrics
            self.model.update_training_metrics(loss_dict, time.time(), grad_norm)

        # Average metrics
        avg_metrics = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
        return avg_metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_metrics = {
            'total_loss': [],
            'nll': [],
            'physics_loss': [],
            'uncertainty_loss': []
        }

        for batch in tqdm(val_loader, desc='Validation'):
            strain_data = batch['strain'].to(self.device)
            parameters = batch['parameters'].to(self.device)

            loss_dict = self.model.compute_loss(strain_data, parameters)

            for key, value in loss_dict.items():
                if key in val_metrics:
                    val_metrics[key].append(value.cpu().item())

        # Average metrics
        avg_metrics = {k: float(np.mean(v)) for k, v in val_metrics.items()}
        return avg_metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader, output_dir: Path):
        """Full training loop."""
        self.logger.info("🚀 Starting training with FlowMatching posterior...")

        for epoch in range(self.max_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.logger.info(f"Epoch {epoch+1} - Train Loss: {train_metrics['total_loss']:.4f}")

            # Validate
            val_metrics = self.validate(val_loader)
            self.logger.info(f"Epoch {epoch+1} - Val Loss: {val_metrics['total_loss']:.4f}")

            # Scheduler step
            self.scheduler.step(val_metrics['total_loss'])

            # Save checkpoint
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.save_checkpoint(output_dir / 'best_model.pt', is_best=True)
                self.logger.info(f"✅ New best model! Val loss: {self.best_val_loss:.4f}")

            # Periodic save
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(output_dir / f'model_epoch_{epoch+1}.pt')

            # Log to WandB
            if WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'val_loss': val_metrics['total_loss'],
                    'train_nll': train_metrics['nll'],
                    'val_nll': val_metrics['nll'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

        self.logger.info("✅ Training complete!")
        return self.best_val_loss

    def save_checkpoint(self, filepath: Path, is_best: bool = False):
        """Save model checkpoint."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'model_config': {
                'context_dim': self.model.context_dim,
                'param_names': self.model.param_names,
                'flow_type': 'flowmatching',
                'solver_steps': self.solver_steps
            },
            'best_val_loss': self.best_val_loss,
            'training_step': self.training_step
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")

        if is_best:
            # Also save as best model
            best_path = filepath.parent / 'best_model_flowmatching.pt'
            torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser(description='Train OverlapNeuralPE with FlowMatching')
    parser.add_argument('--config', type=str, default='configs/enhanced_training.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs/phase3a_flowmatching',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--priority_net_path', type=str, required=True,
                        help='Path to trained PriorityNet checkpoint')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Update config with command-line args
    config['training']['max_epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size

    # ✅ Ensure FlowMatching is configured
    if 'flow_config' not in config:
        config['flow_config'] = {}
    config['flow_config']['type'] = 'flowmatching'
    config['flow_config']['hidden_features'] = 256
    config['flow_config']['num_layers'] = 4
    config['flow_config']['solver_steps'] = 10
    config['context_dim'] = 512

    logger.info(f"✅ Config loaded: {args.config}")
    logger.info(f"   Flow type: {config['flow_config']['type']}")
    logger.info(f"   Context dim: {config['context_dim']}")
    logger.info(f"   Flow layers: {config['flow_config']['num_layers']}")

    # Initialize WandB
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project='posterflow-phase3a',
            name='flowmatching',
            config=config,
            tags=['flowmatching', 'neural-pe']
        )
        logger.info("✅ WandB initialized")

    # Load data
    logger.info("📊 Loading data...")
    from ahsd.data.dataset_generator import DatasetGenerator

    gen = DatasetGenerator(config.get('data_config', {}))
    param_names = config.get('param_names', [
        'mass_1', 'mass_2', 'luminosity_distance', 'geocent_time',
        'ra', 'dec', 'theta_jn', 'psi', 'phase'
    ])

    train_data = ChunkedGWDataLoader(gen, 'train', config.get('data_config', {}))
    val_data = ChunkedGWDataLoader(gen, 'validation', config.get('data_config', {}))

    train_dataset = OverlapGWDataset(train_data, param_names, config)
    val_dataset = OverlapGWDataset(val_data, param_names, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"✅ Data loaded: {len(train_dataset)} train, {len(val_dataset)} val")

    # Initialize model
    logger.info("🧠 Initializing OverlapNeuralPE with FlowMatching...")
    model = OverlapNeuralPE(
        param_names=param_names,
        priority_net_path=args.priority_net_path,
        config=config,
        device=args.device
    )

    summary = model.get_model_summary()
    logger.info(f"   Total params: {summary['total_parameters']:,}")
    logger.info(f"   Trainable params: {summary['trainable_parameters']:,}")

    # Initialize trainer
    trainer = OverlapNeuralPETrainer(model, config, device=args.device)

    # Train
    logger.info("🚀 Starting training...")
    best_loss = trainer.train(train_loader, val_loader, output_dir)

    logger.info(f"✅ Training complete! Best validation loss: {best_loss:.4f}")
    logger.info(f"   Output directory: {output_dir}")

    # Cleanup WandB
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
