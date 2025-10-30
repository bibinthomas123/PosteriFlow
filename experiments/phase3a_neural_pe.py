#!/usr/bin/env python3
"""
Training script for OverlapNeuralPE - Complete pipeline for overlapping GW signals.

Features:
- Multi-signal extraction with PriorityNet
- RL-controlled adaptive complexity
- Bias correction and adaptive subtraction
- Comprehensive metrics and evaluation
- WandB integration for experiment tracking
- Fully config-driven
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

# âœ… WANDB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("âš ï¸  WandB not available - install with: pip install wandb")


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
    """Dataset for training OverlapNeuralPE with optional data augmentation."""

    def __init__(self, data_loader: ChunkedGWDataLoader, param_names: List[str], config: Dict[str, Any]):
        self.param_names = param_names
        self.config = config
        self.logger = logging.getLogger(__name__)

        # âœ… Read augmentation config
        aug_config = config.get('data_augmentation', {})
        self.augmentation_enabled = aug_config.get('enabled', False)
        self.noise_scaling = aug_config.get('noise_scaling', [0.95, 1.05])
        self.time_shifts = aug_config.get('time_shifts', [-0.005, 0.005])
        self.apply_probability = aug_config.get('apply_probability', 0.3)

        self.samples = data_loader.samples
        self.logger.info(f"âœ… Loaded {len(self.samples)} samples from {data_loader.split}")
        if self.augmentation_enabled:
            self.logger.info(f"âœ… Data augmentation enabled (p={self.apply_probability})")

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
        shift_samples = int(time_shift * 4096)  # Assuming 4096 Hz sampling
        if shift_samples != 0:
            augmented = np.roll(augmented, shift_samples, axis=1)

        return augmented

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract strain data
        strain_dict = sample.get('whitened_data', {})
        h1_strain = strain_dict.get('H1', np.zeros(4096))
        l1_strain = strain_dict.get('L1', np.zeros(4096))
        strain_data = np.stack([h1_strain, l1_strain], axis=0)

        # âœ… FIX: Resize to fixed length BEFORE augmentation
        target_length = 16384
        if strain_data.shape[1] != target_length:
            if strain_data.shape[1] > target_length:
                # Crop from center
                start_idx = (strain_data.shape[1] - target_length) // 2
                strain_data = strain_data[:, start_idx:start_idx + target_length]
            else:
                # Pad with zeros
                padding = target_length - strain_data.shape[1]
                pad_left = padding // 2
                pad_right = padding - pad_left
                strain_data = np.pad(strain_data, ((0, 0), (pad_left, pad_right)), mode='constant')

        # Apply augmentation AFTER resizing
        strain_data = self._apply_augmentation(strain_data)

        # Extract metadata
        metadata = sample.get('metadata', {})
        signal_params_list = metadata.get('signal_parameters', [])

        # Scale parameters
        all_params = []
        if signal_params_list and len(signal_params_list) > 0:
            for sig_params in signal_params_list:
                param_vector = self._scale_parameters(sig_params)
                all_params.append(param_vector)

        # Pad to max_signals
        max_signals = 5
        while len(all_params) < max_signals:
            all_params.append([0.0] * len(self.param_names))
        all_params = all_params[:max_signals]

        return {
            'strain_data': torch.tensor(strain_data, dtype=torch.float32),  # Always [2, 16384]
            'parameters': torch.tensor(all_params, dtype=torch.float32),
            'n_signals': len(signal_params_list) if signal_params_list else 0,
            'metadata': metadata
        }

    def _scale_parameters(self, params_dict):
        """Return parameters in physical units (no scaling)."""
        return [params_dict.get(name, 0.0) for name in self.param_names]


def collate_fn(batch):
    """Simple collate - data already fixed-size from __getitem__."""
    strain_data = torch.stack([item['strain_data'] for item in batch])
    parameters = torch.stack([item['parameters'] for item in batch])
    n_signals = torch.tensor([item['n_signals'] for item in batch])
    metadata = [item['metadata'] for item in batch]

    return strain_data, parameters, n_signals, metadata



class OverlapNeuralPETrainer:
    """Trainer for OverlapNeuralPE - Fully config-driven."""

    def __init__(self, model: OverlapNeuralPE, config: Dict[str, Any], use_wandb: bool = True):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.use_wandb = use_wandb and WANDB_AVAILABLE


        # âœ… ALL VALUES FROM CONFIG
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.batch_size = config.get('batch_size', 16)
        self.epochs = config.get('epochs', 100)
        self.patience = config.get('patience', 20)
        self.scheduler_patience = config.get('scheduler_patience', 10)
        self.scheduler_factor = config.get('scheduler_factor', 0.5)
        self.scheduler_min_lr = config.get('min_lr', 1e-6)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.gradient_clip = config.get('gradient_clip', 1.0)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.scheduler_min_lr
        )

        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'gradient_norms': [],
            'train_val_gap': []
        }

        self.device = model.device
        self.global_step = 0

        # Monitoring config
        monitoring_config = config.get('monitoring', {})
        self.save_frequency = monitoring_config.get('save_frequency', 10)
        self.log_frequency = monitoring_config.get('log_frequency', 1)

        # Log configuration
        self.logger.info(f"ðŸ“‹ Trainer Configuration:")
        self.logger.info(f"  Learning Rate: {self.learning_rate}")
        self.logger.info(f"  Weight Decay: {self.weight_decay}")
        self.logger.info(f"  Gradient Clip: {self.gradient_clip}")
        self.logger.info(f"  Batch Size: {self.batch_size}")
        self.logger.info(f"  Epochs: {self.epochs}")
        self.logger.info(f"  Early Stop Patience: {self.patience}")
        self.logger.info(f"  Scheduler Patience: {self.scheduler_patience}")
        self.logger.info(f"  Scheduler Factor: {self.scheduler_factor}")
        self.logger.info(f"  Min LR: {self.scheduler_min_lr}")

    def load_model(self, filepath: str):
        self.model.load_model(filepath)
        self.logger.info(f"Trainer loaded model state from {filepath}")

    def get_model_summary(self) -> Dict[str, Any]:
        return self.model.get_model_summary()

    def save_checkpoint(self, filepath: str, epoch: int, val_metrics: Dict[str, float], is_best: bool = False):
        """Save a full training checkpoint including optimizer/scheduler and trainer state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'epoch': epoch,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'global_step': self.global_step,
            'training_step': getattr(self.model, 'training_step', None),
            'val_metrics': val_metrics,
            'config': self.config
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")

        # Also save model-only if model provides a save_model compat method
        try:
            if hasattr(self.model, 'save_model'):
                model_only_path = str(Path(filepath).with_name(Path(filepath).stem + '_model_only.pth'))
                self.model.save_model(model_only_path)
                self.logger.info(f"Model-only file saved: {model_only_path}")
        except Exception:
            # Non-critical: don't fail checkpointing because model.save_model is incompatible
            self.logger.debug("Model-only save failed or not supported; continuing.")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load a full checkpoint and restore model, optimizer, scheduler and trainer metadata.

        If only a model file (model-only) was provided, this will attempt to load the model weights
        and leave optimizer/scheduler state untouched.
        """
        self.logger.info(f"Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback: try model.load_model if provided (model-only file)
            if hasattr(self.model, 'load_model'):
                try:
                    self.model.load_model(filepath)
                except Exception as e:
                    self.logger.warning(f"Failed fallback model.load_model: {e}")

        # Restore optimizer and scheduler
        if load_optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None and self.scheduler is not None:
                    try:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except Exception as e:
                        self.logger.warning(f"Failed to load scheduler state: {e}")
            except Exception as e:
                self.logger.warning(f"Failed to load optimizer state: {e}")

        # Restore training metadata
        self.best_val_loss = checkpoint.get('best_val_loss', self.best_val_loss)
        self.patience_counter = checkpoint.get('patience_counter', self.patience_counter)
        self.best_epoch = checkpoint.get('best_epoch', self.best_epoch)
        self.history = checkpoint.get('history', self.history)
        self.global_step = checkpoint.get('global_step', self.global_step)

        # If epoch present, return next epoch to start from
        start_epoch = checkpoint.get('epoch', None)
        if start_epoch is not None:
            self.logger.info(f"Resuming from epoch {start_epoch+1}")
            return start_epoch + 1

        return None

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        epoch_losses = []
        epoch_metrics = {
            'nll': [],
            'gradient_norm': []
        }

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, (strain_data, parameters, n_signals, metadata) in enumerate(progress_bar):
            self.optimizer.zero_grad()
            strain_data = strain_data.to(self.device)
            parameters = parameters.to(self.device)

            target_params = parameters[:, 0, :]

            # Compute loss
            loss_dict = self.model.compute_loss(strain_data, target_params)

            # Backward pass
            loss = loss_dict['total_loss']
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()
            # Store metrics
            epoch_losses.append(loss_dict['total_loss'].item())
            epoch_metrics['nll'].append(loss_dict['nll'].item())
            epoch_metrics['gradient_norm'].append(grad_norm.item())

            # âœ… WANDB: Log batch metrics
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'batch/train_loss': loss_dict['total_loss'].item(),
                    'batch/nll': loss_dict['nll'].item(),
                    'batch/gradient_norm': grad_norm.item(),
                    'batch/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })

            self.global_step += 1

            progress_bar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'NLL': f"{loss_dict['nll'].item():.4f}",
                'GradNorm': f"{grad_norm.item():.3f}"
            })

        # Compute epoch averages
        avg_metrics = {
            'avg_loss': np.mean(epoch_losses),
            'avg_nll': np.mean(epoch_metrics['nll']),
            'avg_gradient_norm': np.mean(epoch_metrics['gradient_norm'])
        }

        return avg_metrics

    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate one epoch."""
        self.model.eval()

        epoch_losses = []
        epoch_nlls = []

        with torch.no_grad():
            for strain_data, parameters, n_signals, metadata in tqdm(val_loader, desc="Validating"):
                strain_data = strain_data.to(self.device)
                parameters = parameters.to(self.device)

                target_params = parameters[:, 0, :]
                loss_dict = self.model.compute_loss(strain_data, target_params)

                epoch_losses.append(loss_dict['total_loss'].item())
                epoch_nlls.append(loss_dict['nll'].item())

        return {
            'avg_loss': np.mean(epoch_losses),
            'avg_nll': np.mean(epoch_nlls)
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader, output_dir: Path, start_epoch: int = 0):
        """Complete training loop."""
        self.logger.info("ðŸš€ Starting OverlapNeuralPE training")
        self.logger.info(f"Epochs: {self.epochs}, Batch size: {self.batch_size}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")

        if self.use_wandb:
            self.logger.info("âœ… WandB logging enabled")

    for epoch in range(start_epoch, self.epochs):
            epoch_start = time.time()

            # Training
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validation
            val_metrics = self.validate_epoch(val_loader, epoch)

            # Compute train-val gap
            train_val_gap = train_metrics['avg_loss'] - val_metrics['avg_loss']

            # Scheduler step
            self.scheduler.step(val_metrics['avg_loss'])

            rl_metrics = self.model.get_rl_metrics() if hasattr(self.model, 'get_rl_metrics') else {}
            bias_metrics = self.model.get_bias_metrics() if hasattr(self.model, 'get_bias_metrics') else {}


            # Store history
            self.history['train_loss'].append(train_metrics['avg_loss'])
            self.history['val_loss'].append(val_metrics['avg_loss'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.history['gradient_norms'].append(train_metrics['avg_gradient_norm'])
            self.history['train_val_gap'].append(train_val_gap)

            epoch_time = time.time() - epoch_start

            # âœ… WANDB: Log epoch metrics with patience tracking
            if self.use_wandb:
                log_dict = {
                    'epoch/train_loss': train_metrics['avg_loss'],
                    'epoch/train_nll': train_metrics['avg_nll'],
                    'epoch/train_gradient_norm': train_metrics['avg_gradient_norm'],
                    'epoch/val_loss': val_metrics['avg_loss'],
                    'epoch/val_nll': val_metrics['avg_nll'],
                    'epoch/train_val_gap': train_val_gap,
                    'epoch/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch/time': epoch_time,
                    'epoch/patience_counter': self.patience_counter,
                    'epoch/epochs_since_best': epoch - self.best_epoch,
                    'epoch': epoch + 1,
                    'config/early_stop_patience': self.patience,
                    'config/scheduler_patience': self.scheduler_patience,
                }
                # Merge RL and bias metrics dicts into main log_dict
                if rl_metrics:
                    log_dict.update({f'rl/{key}': val for key, val in rl_metrics.items()})
                if bias_metrics:
                    log_dict.update({f'bias/{key}': val for key, val in bias_metrics.items()})

                wandb.log(log_dict)


            # Logging
            self.logger.info(f"\nEpoch {epoch+1}/{self.epochs}")
            self.logger.info(f"  Train Loss: {train_metrics['avg_loss']:.4f} (NLL: {train_metrics['avg_nll']:.4f})")
            self.logger.info(f"  Val Loss: {val_metrics['avg_loss']:.4f} (NLL: {val_metrics['avg_nll']:.4f})")
            self.logger.info(f"  Train-Val Gap: {train_val_gap:.4f}")
            self.logger.info(f"  Gradient Norm: {train_metrics['avg_gradient_norm']:.3f}")
            self.logger.info(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            self.logger.info(f"  Patience: {self.patience_counter}/{self.patience}")
            self.logger.info(f"  Time: {epoch_time:.1f}s")

            # RL metrics logging
            if rl_metrics:
                self.logger.info(f"  RL Controller:")
                for key, value in rl_metrics.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"    {key}: {value:.4f}")
                    else:
                        self.logger.info(f"    {key}: {value}")

            # Bias Corrector metrics logging
            if bias_metrics:
                self.logger.info(f"  Bias Corrector:")
                for key, value in bias_metrics.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"    {key}: {value:.4f}")
                    else:
                        self.logger.info(f"    {key}: {value}")


            # Early stopping
            if val_metrics['avg_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['avg_loss']
                self.patience_counter = 0
                self.best_epoch = epoch

                # Save best model
                best_path = output_dir / 'best_model.pth'
                # Save full checkpoint (model + optimizer + scheduler + metadata)
                self.save_checkpoint(str(best_path), epoch, val_metrics, is_best=True)
                self.logger.info(f"ðŸ’¾ Best model saved: {val_metrics['avg_loss']:.4f}")

                # âœ… WANDB: Log best model
                if self.use_wandb:
                    wandb.run.summary['best_val_loss'] = self.best_val_loss
                    wandb.run.summary['best_epoch'] = epoch + 1
                    wandb.run.summary['best_train_loss'] = train_metrics['avg_loss']
                    wandb.run.summary['best_gradient_norm'] = train_metrics['avg_gradient_norm']

            else:
                self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    self.logger.info(f"ðŸ›‘ Early stopping at epoch {epoch+1}")
                    self.logger.info(f"   Best model was at epoch {self.best_epoch+1}")
                    break

            # Save checkpoint
            if (epoch + 1) % self.save_frequency == 0:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
                self.save_checkpoint(str(checkpoint_path), epoch, val_metrics)

        # Final save
        final_path = output_dir / 'final_model.pth'
        self.save_checkpoint(str(final_path), epoch, val_metrics)
        self.logger.info("\nâœ… Training completed!")
        self.logger.info(f"ðŸŽ¯ Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")

        # âœ… WANDB: Log final summary
        if self.use_wandb:
            wandb.run.summary['total_epochs'] = epoch + 1
            wandb.run.summary['stopped_early'] = self.patience_counter >= self.patience

        return self.history


def main():
    parser = argparse.ArgumentParser(description='Train OverlapNeuralPE')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--priority_net', type=str, required=True,
                       help='Path to trained PriorityNet checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--wandb_project', type=str, default='overlap-neural-pe',
                       help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='WandB run name')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable WandB logging')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Parameter names
    param_names = config.get('param_names', [
        'mass_1', 'mass_2', 'luminosity_distance',
        'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time'
    ])


    logger.info(f"Parameters: {param_names}")

    # âœ… WANDB: Initialize with full config
    use_wandb = not args.no_wandb and WANDB_AVAILABLE
    if use_wandb:
        # Add training metadata to config
        config['training_metadata'] = {
            'data_dir': args.data_dir,
            'priority_net': args.priority_net,
            'output_dir': str(output_dir),
            'param_names': param_names
        }

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config,
            tags=['overlap-neural-pe', 'gravitational-waves', 'normalizing-flows']
        )
        logger.info(f"âœ… WandB initialized: {wandb.run.name}")

    try:
        # Initialize model
        logger.info("ðŸ”§ Initializing OverlapNeuralPE")
        model = OverlapNeuralPE(
            param_names=param_names,
            priority_net_path=args.priority_net,
            config=config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # âœ… WANDB: Watch model
        if use_wandb:
            wandb.watch(model, log='all', log_freq=100)

        # Load data
        logger.info("ðŸ“Š Loading datasets")

        train_data_loader = ChunkedGWDataLoader(
            dataset_path=args.data_dir,
            split='train',
            max_samples=None
        )

        val_data_loader = ChunkedGWDataLoader(
            dataset_path=args.data_dir,
            split='validation',
            max_samples=None
        )

        train_dataset = OverlapGWDataset(train_data_loader, param_names, config)
        val_dataset = OverlapGWDataset(val_data_loader, param_names, config)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=1,
            pin_memory=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1,
            pin_memory=False
        )
        aug_config = config.get('data_augmentation', {})
        if aug_config.get('enabled', False):
            logger.info(f"ðŸ“Š Data augmentation enabled:")
            logger.info(f"  Noise scaling: {aug_config.get('noise_scaling')}")
            logger.info(f"  Time shifts: {aug_config.get('time_shifts')}")
            logger.info(f"  Apply probability: {aug_config.get('apply_probability')}")

        # âœ… Log dropout config
        dropout = config.get('dropout', 0.1)
        flow_dropout = config.get('flow_config', {}).get('dropout', 0.15)
        logger.info(f"ðŸ”§ Regularization:")
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Flow dropout: {flow_dropout}")

        # Train
        trainer = OverlapNeuralPETrainer(model, config, use_wandb=use_wandb)

        if args.resume_checkpoint is not None:
            logger.info(f"ðŸ“‚ Resuming from checkpoint: {args.resume_checkpoint}")
            # Load full checkpoint (model + optimizer + scheduler + trainer metadata)
            start_epoch = trainer.load_checkpoint(args.resume_checkpoint)
            if start_epoch is None:
                start_epoch = 0
        else:
            start_epoch = 0
        history = trainer.train(train_loader, val_loader, output_dir, start_epoch=start_epoch)

        summary = trainer.get_model_summary()
        print(summary)


        # Save training history
        history_path = output_dir / 'training_history.yaml'
        with open(history_path, 'w') as f:
            yaml.dump(history, f)

        logger.info(f"ðŸŽ‰ All done! Results saved to {output_dir}")


        # âœ… WANDB: Save history artifact
        if use_wandb:
            artifact = wandb.Artifact('training_history', type='training_log')
            artifact.add_file(str(history_path))
            wandb.log_artifact(artifact)
            wandb.finish()

    except Exception as e:
        logger.error(f"âŒ Training failed: {e}", exc_info=True)
        if use_wandb:
            wandb.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
