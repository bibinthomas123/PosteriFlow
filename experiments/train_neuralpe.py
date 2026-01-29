#!/usr/bin/env python3
"""
Training script for OverlapNeuralPE - Complete pipeline for overlapping GW signals.

Features:
- Multi-signal extraction with PriorityNet
- FlowMatching (OT-CFM) or RealNVP normalizing flow (controlled via config flag)
- Comprehensive metrics and evaluation
- WandB integration for experiment tracking
- Fully config-driven

Flow Type Control:
  Set in config: neural_posterior.flow_type = "flowmatching" or "realnvp"
  Auto-configures: num_layers, context_dim based on flow type
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
from ahsd.utils.universal_config import UniversalConfigReader, load_config
from ahsd.utils.noise_marginalization import marginalize_loss_by_theta, should_marginalize

# ✅ WANDB
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available - install with: pip install wandb")


def setup_logging(output_dir: Path, verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def log_distance_calibration(d_true, d_samples, snr, event_type, logger):
    """
    Log distance-specific calibration metrics.
    
    Args:
        d_true: Ground truth distances [batch,]
        d_samples: Posterior samples [batch, n_samples, 1] or [batch, n_samples]
        snr: Network SNR values [batch,]
        event_type: Event type array [batch,] (0=BNS, 1=NSBH, 2=BBH)
        logger: Logger instance
    """
    # Ensure 1D
    d_true = np.atleast_1d(d_true).flatten()
    snr = np.atleast_1d(snr).flatten()
    event_type = np.atleast_1d(event_type).flatten()
    
    # Handle d_samples shape [batch, n_samples, 1] or [batch, n_samples]
    if d_samples.ndim == 3:
        d_samples = d_samples[:, :, 0]  # [batch, n_samples]
    
    # Compute quantiles
    q10 = np.percentile(d_samples, 10, axis=1)
    q50 = np.percentile(d_samples, 50, axis=1)
    q90 = np.percentile(d_samples, 90, axis=1)
    
    logger.info("=" * 70)
    logger.info("📊 DISTANCE CALIBRATION METRICS (MANDATORY INSPECTION)")
    logger.info("=" * 70)
    
    # ✅ METRIC 1: Per-event-type median bias
    logger.info("\n1️⃣ DISTANCE MEDIAN BIAS (per event type):")
    event_names = {0: "BNS ", 1: "NSBH", 2: "BBH "}
    event_targets = {0: 40, 1: 100, 2: 200}  # Mpc
    
    for etype in [0, 1, 2]:
        mask = (event_type == etype)
        if np.sum(mask) > 0:
            median_bias = np.median(q50[mask] - d_true[mask])
            status = "✅" if abs(median_bias) < event_targets[etype] else "🔴"
            logger.info(f"   {status} {event_names[etype]}: {median_bias:+7.1f} Mpc (target: < {event_targets[etype]} Mpc)")
        else:
            logger.info(f"   ⚠️  {event_names[etype]}: NO SAMPLES")
    
    # ✅ METRIC 2: Coverage (quantile-based, NOT raw width)
    logger.info("\n2️⃣ DISTANCE 10-90% COVERAGE (should be 75-85%):")
    coverage_global = np.mean((d_true >= q10) & (d_true <= q90))
    status = "✅" if 0.75 <= coverage_global <= 0.85 else "🔴" if coverage_global > 0.90 else "🟡"
    logger.info(f"   {status} Overall: {100*coverage_global:.1f}%")
    
    for etype in [0, 1, 2]:
        mask = (event_type == etype)
        if np.sum(mask) > 0:
            coverage_type = np.mean((d_true[mask] >= q10[mask]) & (d_true[mask] <= q90[mask]))
            status = "✅" if 0.75 <= coverage_type <= 0.85 else "🔴" if coverage_type > 0.90 else "🟡"
            logger.info(f"   {status} {event_names[etype]}: {100*coverage_type:.1f}%")
    
    # ✅ METRIC 3: Width vs SNR bins (coarse)
    logger.info("\n3️⃣ DISTANCE WIDTH vs SNR (should be: Weak > Medium > Loud):")
    snr_bins = {
        "Weak  ": (0, 15),
        "Medium": (15, 50),
        "Loud  ": (50, 1000),
    }
    
    widths = q90 - q10
    for bin_name, (snr_min, snr_max) in snr_bins.items():
        mask = (snr >= snr_min) & (snr < snr_max)
        if np.sum(mask) > 0:
            median_width = np.median(widths[mask])
            logger.info(f"   {bin_name}: median width = {median_width:7.0f} Mpc (n={np.sum(mask)})")
        else:
            logger.info(f"   {bin_name}: NO SAMPLES")
    
    logger.info("=" * 70)
    logger.info("\n🎯 INTERPRETATION:")
    logger.info("   • Metric 1 → Distance unbiased per event type?")
    logger.info("   • Metric 2 → Posteriors well-calibrated (68% rule)?")
    logger.info("   • Metric 3 → SNR-dependent widths correct physics?")
    logger.info("")


class GroupedBatchSampler:
    """
    Groups samples by base_id to keep K noise realizations together in batches.
    
    Problem: K=3 noise variants spread across chunks → standard shuffle scatters them
    Solution: Keep all variants of same θ in same batch via grouped iteration
    """
    
    def __init__(self, dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        self._group_by_base_id()
    
    def _group_by_base_id(self):
        """Group dataset indices by base sample ID."""
        from collections import defaultdict
        from src.ahsd.utils.noise_marginalization import extract_base_id
        
        groups = defaultdict(list)
        for idx in self.indices:
            sample = self.dataset[idx]
            sample_id = sample.get('sample_id', str(idx))
            base_id = extract_base_id(sample_id)
            groups[base_id].append(idx)
        
        self.groups = groups
        self.group_list = list(groups.items())
    
    def __iter__(self):
        """Yield batches of grouped indices."""
        batch = []
        for base_id, group_indices in self.group_list:
            if len(batch) + len(group_indices) > self.batch_size and batch:
                yield batch
                batch = []
            batch.extend(group_indices)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch
    
    def __len__(self):
        return max(1, len(self.dataset) // self.batch_size)


class OverlapGWDataset(Dataset):
    """Dataset for training OverlapNeuralPE with optional data augmentation."""

    def __init__(
        self, data_loader: ChunkedGWDataLoader, param_names: List[str], config: Dict[str, Any]
    ):
        self.param_names = param_names
        self.config = config
        self.logger = logging.getLogger(__name__)

        # ✅ Read augmentation config
        aug_config = config.get("data_augmentation", {})
        self.augmentation_enabled = aug_config.get("enabled", False)
        self.noise_scaling = aug_config.get("noise_scaling", [0.95, 1.05])
        self.time_shifts = aug_config.get("time_shifts", [-0.005, 0.005])
        self.apply_probability = aug_config.get("apply_probability", 0.3)

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
        shift_samples = int(time_shift * 4096)  # Assuming 4096 Hz sampling
        if shift_samples != 0:
            augmented = np.roll(augmented, shift_samples, axis=1)

        return augmented

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract strain data - try detector_data first (new format), then whitened_data (legacy)
        detector_data = sample.get("detector_data", {})
        if detector_data:
            # New format: detector_data with 'strain' key
            h1_strain = detector_data.get("H1", {}).get("strain", np.zeros(16384))
            l1_strain = detector_data.get("L1", {}).get("strain", np.zeros(16384))
        else:
            # Legacy format: whitened_data
            strain_dict = sample.get("whitened_data", {})
            h1_strain = strain_dict.get("H1", np.zeros(16384))
            l1_strain = strain_dict.get("L1", np.zeros(16384))

        # Ensure proper numpy arrays
        h1_strain = np.asarray(h1_strain, dtype=np.float32)
        l1_strain = np.asarray(l1_strain, dtype=np.float32)
        strain_data = np.stack([h1_strain, l1_strain], axis=0)

        # ✅ FIX: Resize to fixed length BEFORE augmentation
        target_length = 16384
        if strain_data.shape[1] != target_length:
            if strain_data.shape[1] > target_length:
                # Crop from center
                start_idx = (strain_data.shape[1] - target_length) // 2
                strain_data = strain_data[:, start_idx : start_idx + target_length]
            else:
                # Pad with zeros
                padding = target_length - strain_data.shape[1]
                pad_left = padding // 2
                pad_right = padding - pad_left
                strain_data = np.pad(strain_data, ((0, 0), (pad_left, pad_right)), mode="constant")

        # Apply augmentation AFTER resizing
        strain_data = self._apply_augmentation(strain_data)

        # Extract parameters from sample (already in correct format)
        sample_params = sample.get("parameters", [])

        # Build parameter vectors for each signal
        all_params = []
        if sample_params:
            # sample_params is either a list of dicts or a single dict
            if isinstance(sample_params, dict):
                sample_params = [sample_params]

            for sig_params in sample_params:
                if isinstance(sig_params, dict):
                    param_vector = self._scale_parameters(sig_params)
                    all_params.append(param_vector)

        # Pad to max_signals
        max_signals = 5
        while len(all_params) < max_signals:
            all_params.append([0.0] * len(self.param_names))
        all_params = all_params[:max_signals]

        # Get n_signals from sample or metadata
        n_signals = sample.get(
            "n_signals", len([p for p in all_params if p != [0.0] * len(self.param_names)])
        )

        return {
            "strain_data": torch.tensor(strain_data, dtype=torch.float32),  # Always [2, 16384]
            "parameters": torch.tensor(all_params, dtype=torch.float32),
            "n_signals": n_signals,
            "metadata": sample.get("metadata", {}),
            "sample_id": sample.get("sample_id", "unknown"),  # ✅ NEW: For noise marginalization
        }

    def _scale_parameters(self, params_dict):
        """Return parameters in physical units (no scaling)."""
        return [params_dict.get(name, 0.0) for name in self.param_names]


def collate_fn(batch):
    """Simple collate - data already fixed-size from __getitem__."""
    strain_data = torch.stack([item["strain_data"] for item in batch])
    parameters = torch.stack([item["parameters"] for item in batch])
    n_signals = torch.tensor([item["n_signals"] for item in batch])
    metadata = [item["metadata"] for item in batch]
    sample_ids = [item.get("sample_id", "unknown") for item in batch]  # ✅ NEW: For marginalization

    return strain_data, parameters, n_signals, metadata, sample_ids


class OverlapNeuralPETrainer:
    """Trainer for OverlapNeuralPE - Fully config-driven with UniversalConfigReader."""

    def __init__(self, model: OverlapNeuralPE, config: Dict[str, Any], use_wandb: bool = True):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # ✅ Initialize UniversalConfigReader for type-safe config access
        reader = UniversalConfigReader()
        np_config = config.neural_posterior if hasattr(config, 'neural_posterior') else config.get("neural_posterior", {})

        # ✅ ALL VALUES FROM CONFIG - Read from neural_posterior section with UniversalConfigReader
        self.learning_rate = reader.get(np_config, "learning_rate", default=3e-5, dtype=float, section="neural_posterior")
        self.batch_size = reader.get(np_config, "batch_size", default=64, dtype=int, section="neural_posterior")
        self.epochs = reader.get(np_config, "epochs", default=50, dtype=int, section="neural_posterior")
        self.patience = reader.get(np_config, "patience", default=15, dtype=int, section="neural_posterior")
        self.scheduler_patience = reader.get(np_config, "scheduler_patience", default=8, dtype=int, section="neural_posterior")
        self.scheduler_factor = reader.get(np_config, "scheduler_factor", default=0.6, dtype=float, section="neural_posterior")
        self.scheduler_min_lr = reader.get(np_config, "min_lr", default=1e-6, dtype=float, section="neural_posterior")
        self.weight_decay = reader.get(np_config, "weight_decay", default=1e-4, dtype=float, section="neural_posterior")
        self.gradient_clip = reader.get(np_config, "gradient_clip", default=20.0, dtype=float, section="neural_posterior")
        self.endpoint_loss_weight = reader.get(np_config, "endpoint_loss_weight", default=1.5, dtype=float, section="neural_posterior")

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.scheduler_min_lr,
        )

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_epoch = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "gradient_norms": [],
            "train_val_gap": [],
        }

        self.device = model.device
        self.global_step = 0

        # Monitoring config
        monitoring_config = config.get("monitoring", {})
        self.save_frequency = monitoring_config.get("save_frequency", 10)
        self.log_frequency = monitoring_config.get("log_frequency", 1)

        # Log configuration
        self.logger.info(f"✅ Trainer Configuration:")
        self.logger.info(f"  Learning Rate: {self.learning_rate}")
        self.logger.info(f"  Weight Decay: {self.weight_decay}")
        self.logger.info(f"  Gradient Clip: {self.gradient_clip}")
        self.logger.info(f"  Endpoint Loss Weight: {self.endpoint_loss_weight}")
        self.logger.info(f"  Batch Size: {self.batch_size}")
        self.logger.info(f"  Epochs: {self.epochs}")
        self.logger.info(f"  Early Stop Patience: {self.patience}")
        self.logger.info(f"  Scheduler Patience: {self.scheduler_patience}")
        self.logger.info(f"  Scheduler Factor: {self.scheduler_factor}")
        self.logger.info(f"  Min LR: {self.scheduler_min_lr}")
        
        # Log flow configuration
        flow_config_dict = np_config.get("flow_config", {}) if isinstance(np_config, dict) else getattr(np_config, "flow_config", {})
        self.logger.info(f"Flow Configuration:")
        self.logger.info(f"  Flow Type: nsf (from neural_posterior.flow_type)")
        if isinstance(flow_config_dict, dict):
            self.logger.info(f"  Num Layers: {flow_config_dict.get('num_layers', 12)}")
            self.logger.info(f"  Hidden Features: {flow_config_dict.get('hidden_features', 256)}")
            self.logger.info(f"  Num Bins: {flow_config_dict.get('num_bins', 16)}")
            self.logger.info(f"  Dropout: {flow_config_dict.get('dropout', 0.15)}")
            self.logger.info(f"  Tail Bound: {flow_config_dict.get('tail_bound', 3.0)}")

    def load_model(self, filepath: str):
        self.model.load_model(filepath)
        self.logger.info(f"Trainer loaded model state from {filepath}")

    def get_model_summary(self) -> Dict[str, Any]:
        return self.model.get_model_summary()

    def save_checkpoint(
        self, filepath: str, epoch: int, val_metrics: Dict[str, float], is_best: bool = False
    ):
        """Save a full training checkpoint including optimizer/scheduler and trainer state."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "global_step": self.global_step,
            "training_step": getattr(self.model, "training_step", None),
            "val_metrics": val_metrics,
            "config": self.config,
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")

        # Also save model-only if model provides a save_model compat method
        try:
            if hasattr(self.model, "save_model"):
                model_only_path = str(
                    Path(filepath).with_name(Path(filepath).stem + "_model_only.pth")
                )
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
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Fallback: try model.load_model if provided (model-only file)
            if hasattr(self.model, "load_model"):
                try:
                    self.model.load_model(filepath)
                except Exception as e:
                    self.logger.warning(f"Failed fallback model.load_model: {e}")

        # Restore optimizer and scheduler
        if (
            load_optimizer
            and "optimizer_state_dict" in checkpoint
            and checkpoint["optimizer_state_dict"] is not None
        ):
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if (
                    "scheduler_state_dict" in checkpoint
                    and checkpoint["scheduler_state_dict"] is not None
                    and self.scheduler is not None
                ):
                    try:
                        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    except Exception as e:
                        self.logger.warning(f"Failed to load scheduler state: {e}")
            except Exception as e:
                self.logger.warning(f"Failed to load optimizer state: {e}")

        # Restore training metadata
        self.best_val_loss = checkpoint.get("best_val_loss", self.best_val_loss)
        self.patience_counter = checkpoint.get("patience_counter", self.patience_counter)
        self.best_epoch = checkpoint.get("best_epoch", self.best_epoch)
        self.history = checkpoint.get("history", self.history)
        self.global_step = checkpoint.get("global_step", self.global_step)

        # If epoch present, return next epoch to start from
        start_epoch = checkpoint.get("epoch", None)
        if start_epoch is not None:
            self.logger.info(f"Resuming from epoch {start_epoch+1}")
            return start_epoch + 1

        return None

    def _log_gradient_statistics(self, epoch: int):
        """✅ Nov 13: Log per-layer gradient norms for debugging vanishing gradients."""
        self.logger.info(f"\n=== Gradient Statistics (Epoch {epoch+1}, Batch 1) ===")

        vanishing_layers = []
        flow_grad_found = False

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                param_norm = param.data.norm(2).item() + 1e-8
                ratio = grad_norm / param_norm

                # Flag flow-related layers
                if "flow" in name or "velocity" in name:
                    flow_grad_found = True
                    status = "✅" if grad_norm > 1e-6 else "⚠️ "
                    self.logger.info(
                        f"{status} {name}: grad_norm={grad_norm:.6f}, "
                        f"param_norm={param_norm:.6f}, ratio={ratio:.6f}"
                    )

                    if ratio < 1e-6 and param_norm > 1e-2:
                        vanishing_layers.append(name)

        # Summary
        if not flow_grad_found:
            self.logger.warning("⚠️ No gradients found in flow layers!")

        if vanishing_layers:
            self.logger.error(f"🔴 VANISHING GRADIENTS IN: {vanishing_layers}")
        else:
            self.logger.info("✅ Gradient flow OK (no vanishing gradients detected)")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch with integrated RL controller training."""
        self.model.train()

        epoch_losses = []
        epoch_metrics = {
            "flow_loss": [],
            "gradient_norm": [],
            "physics_loss": [],
            "bias_loss": [],
            "uncertainty_loss": [],
            "jacobian_reg": [],
            "rl_loss": [],  # ✅ Track RL losses
            "rl_complexity": [],  # Track complexity distribution
        }

        # RL state tracking
        prev_loss = float("inf")
        batch_time_start = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch_output in enumerate(progress_bar):
            try:
                batch_start = time.time()
                self.optimizer.zero_grad()
                
                # ✅ Handle new sample_ids return value
                if len(batch_output) == 5:
                    strain_data, parameters, n_signals, metadata, sample_ids = batch_output
                else:
                    strain_data, parameters, n_signals, metadata = batch_output
                    sample_ids = [f"batch_{i}" for i in range(len(strain_data))]  # Fallback
                
                strain_data = strain_data.to(self.device)
                parameters = parameters.to(self.device)

                target_params = parameters[:, 0, :]

                # Compute loss
                loss_dict = self.model.compute_loss(strain_data, target_params)
                
                # ✅ JAN 7: APPLY NOISE MARGINALIZATION (log every 50 batches)
                if should_marginalize(sample_ids):
                    loss_dict = marginalize_loss_by_theta(
                        loss_dict, sample_ids, verbose=True, batch_idx=batch_idx, log_frequency=100
                    )

                # Backward pass
                loss = loss_dict["total_loss"]
                loss.backward()

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.optimizer.step()

                # Store metrics
                epoch_losses.append(loss_dict["total_loss"].item())
                epoch_metrics["flow_loss"].append(loss_dict["flow_loss"].item())
                epoch_metrics["gradient_norm"].append(grad_norm.item())

                # Store component losses
                if "physics_loss" in loss_dict:
                    epoch_metrics["physics_loss"].append(loss_dict["physics_loss"].item())
                if "bias_loss" in loss_dict:
                    epoch_metrics["bias_loss"].append(loss_dict["bias_loss"].item())
                if "uncertainty_loss" in loss_dict:
                    epoch_metrics["uncertainty_loss"].append(loss_dict["uncertainty_loss"].item())
                if "jacobian_reg" in loss_dict:
                    epoch_metrics["jacobian_reg"].append(loss_dict["jacobian_reg"].item())

                # ✅ WANDB: Log batch metrics
                if self.use_wandb and batch_idx % 10 == 0:
                    batch_log = {
                        "batch/train_loss": loss_dict["total_loss"].item(),
                        "batch/flow_loss": loss_dict["flow_loss"].item(),
                        "batch/gradient_norm": grad_norm.item(),
                        "batch/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "global_step": self.global_step,
                    }
                    # Log component losses if available
                    if "physics_loss" in loss_dict:
                        batch_log["batch/physics_loss"] = loss_dict["physics_loss"].item()
                    if "bias_loss" in loss_dict:
                        batch_log["batch/bias_loss"] = loss_dict["bias_loss"].item()
                    if "uncertainty_loss" in loss_dict:
                        batch_log["batch/uncertainty_loss"] = loss_dict["uncertainty_loss"].item()
                    if "jacobian_reg" in loss_dict:
                        batch_log["batch/jacobian_reg"] = loss_dict["jacobian_reg"].item()

                    # Log RL metrics if available
                    if epoch_metrics["rl_loss"]:
                        batch_log["batch/rl_loss"] = epoch_metrics["rl_loss"][-1]

                    wandb.log(batch_log)

                self.global_step += 1

                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss_dict['total_loss'].item():.4f}",
                        "FlowLoss": f"{loss_dict['flow_loss'].item():.4f}",
                        "GradNorm": f"{grad_norm.item():.3f}",
                    }
                )

            except Exception as e:
                self.logger.error(f"🔴 Error in batch {batch_idx}: {str(e)}", exc_info=True)
                self.logger.error(f"Strain shape: {strain_data.shape if 'strain_data' in locals() else 'N/A'}")
                self.logger.error(f"Parameters shape: {parameters.shape if 'parameters' in locals() else 'N/A'}")
                raise

        # Compute epoch averages
        avg_metrics = {
            "avg_loss": np.mean(epoch_losses),
            "avg_flow_loss": np.mean(epoch_metrics["flow_loss"]),
            "avg_gradient_norm": np.mean(epoch_metrics["gradient_norm"]),
        }

        # Add component loss averages if available
        if epoch_metrics["physics_loss"]:
            avg_metrics["avg_physics_loss"] = np.mean(epoch_metrics["physics_loss"])
        if epoch_metrics["bias_loss"]:
            avg_metrics["avg_bias_loss"] = np.mean(epoch_metrics["bias_loss"])
        if epoch_metrics["uncertainty_loss"]:
            avg_metrics["avg_uncertainty_loss"] = np.mean(epoch_metrics["uncertainty_loss"])
        if epoch_metrics["jacobian_reg"]:
            avg_metrics["avg_jacobian_reg"] = np.mean(epoch_metrics["jacobian_reg"])

        # ✅ Nov 14: Add RL metrics
        if epoch_metrics["rl_loss"]:
            avg_metrics["avg_rl_loss"] = np.mean(epoch_metrics["rl_loss"])

        # Add complexity distribution
        if epoch_metrics["rl_complexity"]:
            complexity_array = np.array(epoch_metrics["rl_complexity"])
            avg_metrics["avg_complexity"] = float(np.mean(complexity_array))
            avg_metrics["complexity_std"] = float(np.std(complexity_array))
            # Count selections
            avg_metrics["complexity_low_count"] = int(np.sum(complexity_array == 0))
            avg_metrics["complexity_medium_count"] = int(np.sum(complexity_array == 1))
            avg_metrics["complexity_high_count"] = int(np.sum(complexity_array == 2))

        return avg_metrics

    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
         """Validate one epoch."""
         self.model.eval()

         epoch_losses = []
         epoch_nlls = []
         epoch_physics_losses = []
         
         # Collect data for final diagnostics
         all_strain_data = []
         all_parameters = []
         all_target_snrs = []  # Collect target_snr from metadata

         with torch.no_grad():
             for batch_idx, batch_output in enumerate(
                 tqdm(val_loader, desc="Validating")
             ):
                 # ✅ Handle new sample_ids return value
                 if len(batch_output) == 5:
                     strain_data, parameters, n_signals, metadata, sample_ids = batch_output
                 else:
                     strain_data, parameters, n_signals, metadata = batch_output
                     sample_ids = [f"batch_{i}" for i in range(len(strain_data))]
                 
                 strain_data = strain_data.to(self.device)
                 parameters = parameters.to(self.device)

                 target_params = parameters[:, 0, :]
                 loss_dict = self.model.compute_loss(strain_data, target_params)
                 
                 # ✅ JAN 7: APPLY NOISE MARGINALIZATION IN VALIDATION TOO (log every 10 batches)
                 if should_marginalize(sample_ids):
                     loss_dict = marginalize_loss_by_theta(
                         loss_dict, sample_ids, verbose=True, batch_idx=batch_idx, log_frequency=100
                     )

                 epoch_losses.append(loss_dict["total_loss"].item())
                 epoch_nlls.append(loss_dict["flow_loss"].item())
                 if "physics_loss" in loss_dict:
                     epoch_physics_losses.append(loss_dict["physics_loss"].item())
                 
                 # Collect first batch worth of data for diagnostics
                 if len(all_strain_data) == 0:
                     all_strain_data = strain_data
                     all_parameters = parameters
                     # Extract target_snr from metadata
                     if isinstance(metadata, dict) and 'target_snr' in metadata:
                         all_target_snrs = metadata['target_snr']
                     elif isinstance(metadata, dict):
                         # Fallback: compute from strain RMS if target_snr not available
                         all_target_snrs = np.sqrt(np.mean(strain_data.cpu().numpy()**2, axis=(1, 2)))
                         all_target_snrs = np.maximum(all_target_snrs, 1.0)

         # ✅ FINAL: Compute and log diagnostics after entire validation epoch
         try:
             # Only run diagnostics if we collected data
             if len(all_strain_data) > 0 and len(all_parameters) > 0:
                 # Use up to 256 samples for diagnostics (one batch, much faster)
                 n_diag = min(256, len(all_strain_data) if isinstance(all_strain_data, torch.Tensor) else len(all_strain_data))
                 subset_strain = all_strain_data[:n_diag] if isinstance(all_strain_data, torch.Tensor) else all_strain_data
                 subset_params = all_parameters[:n_diag] if isinstance(all_parameters, torch.Tensor) else all_parameters
                 flow_metrics = self.model.compute_flow_matching_metrics(subset_strain, subset_params)
                 self.model.log_flow_diagnostics(flow_metrics, epoch, prefix="VALIDATION (EPOCH FINAL)")
                 
                 # ✅ JAN 28: Log distance calibration every 5 epochs (not every 10, and not every epoch!)
                 if epoch % 5 == 0 or epoch == 0:
                     self.logger.info(f"\n🔍 EPOCH {epoch}: Computing distance calibration metrics (n={n_diag})...")
                     try:
                         # Sample posterior with all samples returned [batch, n_samples, 11 params]
                         with torch.no_grad():
                             posterior_samples = self.model.sample_posterior(
                                 subset_strain, n_samples=256, return_all_samples=True
                             )
                         
                         if posterior_samples is not None and posterior_samples.shape[0] > 0:
                             # Extract distance (parameter index 2: luminosity_distance)
                             d_true = subset_params[:, 0, 2].cpu().numpy()  # [batch,]
                             d_samples = posterior_samples[:, :, 2].cpu().numpy()  # [batch, n_samples]
                             
                             # Use target_snr from metadata (much more accurate than computing from strain RMS)
                             if isinstance(all_target_snrs, (list, np.ndarray)):
                                 snr = np.array(all_target_snrs[:len(d_true)]).astype(float)
                             else:
                                 # Fallback: compute from strain RMS if target_snr not available
                                 snr = np.sqrt(np.mean(subset_strain.cpu().numpy()**2, axis=(1, 2)))
                                 snr = np.maximum(snr, 1.0)
                             
                             # Event type distribution
                             event_type = (np.arange(len(d_true)) % 3).astype(int)
                             
                             log_distance_calibration(d_true, d_samples, snr, event_type, self.logger)
                         else:
                             self.logger.warning("⚠️  No posterior samples returned at epoch %d", epoch)
                     except Exception as e:
                         self.logger.warning(f"⚠️  Distance calibration skipped (epoch {epoch}): {type(e).__name__}: {e}")
             else:
                 self.logger.debug("No validation data collected for diagnostics")
         except Exception as e:
             self.logger.error(f"Flow diagnostics failed: {type(e).__name__}: {e}")

         val_metrics = {"avg_loss": np.mean(epoch_losses), "avg_flow_loss": np.mean(epoch_nlls)}
         if epoch_physics_losses:
             val_metrics["avg_physics_loss"] = np.mean(epoch_physics_losses)

         return val_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
        start_epoch: int = 0,
    ):
        """Complete training loop."""
        self.logger.info("Starting OverlapNeuralPE training")
        self.logger.info(f"Epochs: {self.epochs}, Batch size: {self.batch_size}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")

        if self.use_wandb:
            self.logger.info("✅ WandB logging enabled")

        for epoch in range(start_epoch, self.epochs):
            epoch_start = time.time()

            # Training
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validation
            val_metrics = self.validate_epoch(val_loader, epoch)

            # Compute train-val gap
            train_val_gap = train_metrics["avg_loss"] - val_metrics["avg_loss"]

            # Scheduler step
            self.scheduler.step(val_metrics["avg_loss"])

            rl_metrics = (
                self.model.get_rl_metrics() if hasattr(self.model, "get_rl_metrics") else {}
            )
            bias_metrics = (
                self.model.get_bias_metrics() if hasattr(self.model, "get_bias_metrics") else {}
            )
            priority_net_metrics = (
                self.model.get_priority_net_metrics()
                if hasattr(self.model, "get_priority_net_metrics")
                else {}
            )

            # Store history
            self.history["train_loss"].append(train_metrics["avg_loss"])
            self.history["val_loss"].append(val_metrics["avg_loss"])
            self.history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])
            self.history["gradient_norms"].append(train_metrics["avg_gradient_norm"])
            self.history["train_val_gap"].append(train_val_gap)

            epoch_time = time.time() - epoch_start

            # ✅ WANDB: Log epoch metrics with patience tracking
            if self.use_wandb:
                log_dict = {
                    "epoch/train_loss": train_metrics["avg_loss"],
                    "epoch/train_flow_loss": train_metrics["avg_flow_loss"],
                    "epoch/train_gradient_norm": train_metrics["avg_gradient_norm"],
                    "epoch/val_loss": val_metrics["avg_loss"],
                    "epoch/val_flow_loss": val_metrics["avg_flow_loss"],
                    "epoch/train_val_gap": train_val_gap,
                    "epoch/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "epoch/time": epoch_time,
                    "epoch/patience_counter": self.patience_counter,
                    "epoch/epochs_since_best": epoch - self.best_epoch,
                    "epoch": epoch + 1,
                    "config/early_stop_patience": self.patience,
                    "config/scheduler_patience": self.scheduler_patience,
                }
                # ✅ Nov 14: Merge RL training metrics
                if "avg_rl_loss" in train_metrics:
                    log_dict["epoch/rl_loss"] = train_metrics["avg_rl_loss"]

                # Merge RL, bias, and priority net metrics dicts into main log_dict
                if rl_metrics:
                    log_dict.update({f"rl/{key}": val for key, val in rl_metrics.items()})
                if bias_metrics:
                    log_dict.update({f"bias/{key}": val for key, val in bias_metrics.items()})
                if priority_net_metrics:
                    log_dict.update(
                        {f"priority_net/{key}": val for key, val in priority_net_metrics.items()}
                    )

                wandb.log(log_dict)

            # Logging
            self.logger.info(f"\nEpoch {epoch+1}/{self.epochs}")
            self.logger.info(
                f"  Train Loss: {train_metrics['avg_loss']:.4f} (Flow Loss: {train_metrics['avg_flow_loss']:.4f})"
            )
            self.logger.info(
                f"  Val Loss: {val_metrics['avg_loss']:.4f} (Flow Loss: {val_metrics['avg_flow_loss']:.4f})"
            )
            self.logger.info(f"  Train-Val Gap: {train_val_gap:.4f}")
            self.logger.info(f"  Gradient Norm: {train_metrics['avg_gradient_norm']:.3f}")
            self.logger.info(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            self.logger.info(f"  Patience: {self.patience_counter}/{self.patience}")
            self.logger.info(f"  Time: {epoch_time:.1f}s")

            # Component loss breakdown
            self.logger.info(f"  Loss Components:")
            if "avg_physics_loss" in train_metrics:
                self.logger.info(f"    Physics Loss: {train_metrics['avg_physics_loss']:.6f}")
            if "avg_bias_loss" in train_metrics:
                self.logger.info(f"    Bias Loss: {train_metrics['avg_bias_loss']:.6f}")
            if "avg_uncertainty_loss" in train_metrics:
                self.logger.info(
                    f"    Uncertainty Loss: {train_metrics['avg_uncertainty_loss']:.6f}"
                )
            if "avg_jacobian_reg" in train_metrics:
                self.logger.info(f"    Jacobian Reg: {train_metrics['avg_jacobian_reg']:.6f}")
            # ✅ Nov 14: Log RL loss and complexity
            if "avg_rl_loss" in train_metrics:
                self.logger.info(f"    RL Loss: {train_metrics['avg_rl_loss']:.6f}")
            if "avg_complexity" in train_metrics:
                self.logger.info(
                    f"    Complexity (avg±std): {train_metrics['avg_complexity']:.2f}±{train_metrics['complexity_std']:.2f}"
                )
                self.logger.info(
                    f"      Low: {train_metrics.get('complexity_low_count', 0)}, Medium: {train_metrics.get('complexity_medium_count', 0)}, High: {train_metrics.get('complexity_high_count', 0)}"
                )

            # Integration summary logging
            if hasattr(self.model, "get_integration_summary"):
                try:
                    summary = self.model.get_integration_summary()
                    metrics = summary.get("metrics", {})
                    bias_metrics_summary = metrics.get("bias_metrics", {})
                    rl_metrics_summary = metrics.get("rl_metrics", {})

                    self.logger.info(f"  Integration Summary:")
                    self.logger.info(
                        f"    Total Parameters: {metrics.get('total_parameters', 0):,}"
                    )
                    self.logger.info(
                        f"    Trainable Parameters: {metrics.get('trainable_parameters', 0):,}"
                    )

                    if rl_metrics_summary:
                        self.logger.info(f"  RL State:")
                        if "epsilon" in rl_metrics_summary:
                            self.logger.info(
                                f"    Exploration (ε): {rl_metrics_summary['epsilon']:.4f}"
                            )
                        if "avg_complexity" in rl_metrics_summary:
                            self.logger.info(
                                f"    Avg Complexity: {rl_metrics_summary['avg_complexity']:.2f}"
                            )
                        if "action_entropy" in rl_metrics_summary:
                            self.logger.info(
                                f"    Action Entropy: {rl_metrics_summary['action_entropy']:.4f}"
                            )
                        if "avg_reward" in rl_metrics_summary:
                            self.logger.info(
                                f"    Avg Reward: {rl_metrics_summary['avg_reward']:.4f}"
                            )

                    if bias_metrics_summary:
                        self.logger.info(f"  Bias State:")
                        if "avg_correction" in bias_metrics_summary:
                            self.logger.info(
                                f"    Magnitude: {bias_metrics_summary['avg_correction']:.6f}"
                            )
                        if "correction_acceptance_rate" in bias_metrics_summary:
                            accept_rate = bias_metrics_summary["correction_acceptance_rate"]
                            self.logger.info(f"    Acceptance: {accept_rate:.2%}")
                except Exception as e:
                    self.logger.debug(f"Could not log integration summary: {e}")

            # Early stopping
            if val_metrics["avg_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["avg_loss"]
                self.patience_counter = 0
                self.best_epoch = epoch

                # Save best model
                best_path = output_dir / "best_model.pth"
                # Save full checkpoint (model + optimizer + scheduler + metadata)
                self.save_checkpoint(str(best_path), epoch, val_metrics, is_best=True)
                self.logger.info(f"ðŸ’¾ Best model saved: {val_metrics['avg_loss']:.4f}")

                # ✅ WANDB: Log best model
                if self.use_wandb:
                    wandb.run.summary["best_val_loss"] = self.best_val_loss
                    wandb.run.summary["best_epoch"] = epoch + 1
                    wandb.run.summary["best_train_loss"] = train_metrics["avg_loss"]
                    wandb.run.summary["best_gradient_norm"] = train_metrics["avg_gradient_norm"]

            else:
                self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    self.logger.info(f"ðŸ›‘ Early stopping at epoch {epoch+1}")
                    self.logger.info(f"   Best model was at epoch {self.best_epoch+1}")
                    break

            # Save checkpoint
            if (epoch + 1) % self.save_frequency == 0:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
                self.save_checkpoint(str(checkpoint_path), epoch, val_metrics)

        # Final save
        final_path = output_dir / "final_model.pth"
        self.save_checkpoint(str(final_path), epoch, val_metrics)
        self.logger.info("\n✅ Training completed!")
        self.logger.info(
            f"ðŸŽ¯ Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}"
        )

        # ✅ WANDB: Log final summary
        if self.use_wandb:
            wandb.run.summary["total_epochs"] = epoch + 1
            wandb.run.summary["stopped_early"] = self.patience_counter >= self.patience

        return self.history


def main():
    parser = argparse.ArgumentParser(description="Train OverlapNeuralPE")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data")
    parser.add_argument(
        "--priority_net", type=str, required=True, help="Path to trained PriorityNet checkpoint"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="overlap-neural-pe", help="WandB project name"
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    # ✅ Load config using UniversalConfigReader
    logger.info("Loading config with UniversalConfigReader...")
    config = load_config(args.config)  # Returns ConfigDict with dual access
    logger.info(f"Config loaded successfully from {args.config}")
    
    # ✅ FIX (Jan 2): Pass FULL config, not just neural_posterior section
    # Model will extract neural_posterior section internally with:
    #   np_config = self.config.get("neural_posterior", {})
    # If we pass only neural_posterior section here, the second extraction returns {}

    # Parameter names (stored in neural_posterior section)
    np_config = config.get("neural_posterior", {})
    param_names = np_config.get(
       "param_names",
       [
           "mass_1",
           "mass_2",
           "luminosity_distance",
           "ra",
           "dec",
           "theta_jn",
           "psi",
           "phase",
           "geocent_time",
           "a1",
           "a2",
           "tilt1",
           "tilt2",
       ],
    )

    logger.info(f"Parameters: {param_names}")
    
    # ✅ Validate critical config values using UniversalConfigReader
    reader = UniversalConfigReader()
    np_config = config.neural_posterior
    learning_rate = reader.get(np_config, "learning_rate", default=4e-4, dtype=float, section="neural_posterior")
    batch_size = reader.get(np_config, "batch_size", default=64, dtype=int, section="neural_posterior")
    epochs = reader.get(np_config, "epochs", default=100, dtype=int, section="neural_posterior")
    patience = reader.get(np_config, "patience", default=20, dtype=int, section="neural_posterior")
    logger.info(f"✅ Config validation (neural_posterior section):")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Patience: {patience}")

    # ✅ WANDB: Initialize with full config
    use_wandb = not args.no_wandb and WANDB_AVAILABLE
    if use_wandb:
        # Add training metadata to config
        config["training_metadata"] = {
            "data_dir": args.data_dir,
            "priority_net": args.priority_net,
            "output_dir": str(output_dir),
            "param_names": param_names,
        }

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config,
            tags=["overlap-neural-pe", "gravitational-waves", "normalizing-flows"],
        )
        logger.info(f"✅ WandB initialized: {wandb.run.name}")

    try:
        # Initialize model
        logger.info("Initializing OverlapNeuralPE")
        model = OverlapNeuralPE(
            param_names=param_names,
            priority_net_path=args.priority_net,
            config=config,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # ✅ WANDB: Watch model
        if use_wandb:
            wandb.watch(model, log="all", log_freq=100)

        # Load data
        logger.info("Loading datasets")

        train_data_loader = ChunkedGWDataLoader(
            dataset_path=args.data_dir, split="train", max_samples=None
        )

        val_data_loader = ChunkedGWDataLoader(
            dataset_path=args.data_dir, split="validation", max_samples=None
        )

        train_dataset = OverlapGWDataset(train_data_loader, param_names, config)
        val_dataset = OverlapGWDataset(val_data_loader, param_names, config)

        # Get batch size from config
        batch_size = config.get("batch_size", 64)

        # Create data loaders with grouped batch sampler
        # Use GroupedBatchSampler to keep K=3 noise variants together
        # Problem: K=3 variants spread across chunks, shuffle=True scatters them
        # Solution: GroupedBatchSampler groups by base_id, keeps variants in same batch
        grouped_sampler = GroupedBatchSampler(train_dataset, batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=grouped_sampler,
            collate_fn=collate_fn,
            num_workers=1,
            pin_memory=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1,
            pin_memory=False,
        )
        aug_config = config.get("data_augmentation", {})
        if aug_config.get("enabled", False):
            logger.info(f"Data augmentation enabled:")
            logger.info(f"  Noise scaling: {aug_config.get('noise_scaling')}")
            logger.info(f"  Time shifts: {aug_config.get('time_shifts')}")
            logger.info(f"  Apply probability: {aug_config.get('apply_probability')}")

        # ✅ Log dropout config
        dropout = config.get("dropout", 0.1)
        flow_dropout = config.get("flow_config", {}).get("dropout", 0.15)
        logger.info(f"Regularization:")
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
        history_path = output_dir / "training_history.yaml"
        with open(history_path, "w") as f:
            yaml.dump(history, f)

        logger.info(f"ðŸŽ‰ All done! Results saved to {output_dir}")

        # ✅ WANDB: Save history artifact
        if use_wandb:
            artifact = wandb.Artifact("training_history", type="training_log")
            artifact.add_file(str(history_path))
            wandb.log_artifact(artifact)
            wandb.finish()

    except Exception as e:
        logger.error(f"âŒ Training failed: {e}", exc_info=True)
        if use_wandb:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
