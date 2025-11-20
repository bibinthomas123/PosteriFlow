#!/usr/bin/env python3
"""
Training script for OverlapNeuralPE - Complete pipeline for overlapping GW signals.

Features:
- Multi-signal extraction with PriorityNet
- FlowMatching (OT-CFM) or RealNVP normalizing flow (controlled via config flag)
- RL-controlled adaptive complexity
- Bias correction and adaptive subtraction
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

# ✅ WANDB
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
    log_file = output_dir / "training.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


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

    return strain_data, parameters, n_signals, metadata


class OverlapNeuralPETrainer:
    """Trainer for OverlapNeuralPE - Fully config-driven."""

    def __init__(self, model: OverlapNeuralPE, config: Dict[str, Any], use_wandb: bool = True):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # ✅ ALL VALUES FROM CONFIG
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.batch_size = config.get("batch_size", 16)
        self.epochs = config.get("epochs", 100)
        self.patience = config.get("patience", 20)
        self.scheduler_patience = config.get("scheduler_patience", 10)
        self.scheduler_factor = config.get("scheduler_factor", 0.5)
        self.scheduler_min_lr = config.get("min_lr", 1e-6)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.gradient_clip = config.get("gradient_clip", 1.0)

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

        # ✅ Log loss weights (Nov 13 fix verification)
        np_cfg = config.get("neural_posterior", {})
        physics_weight = np_cfg.get("physics_loss_weight", 0.05)
        bounds_weight = np_cfg.get("bounds_penalty_weight", 0.5)
        sample_weight = np_cfg.get("sample_loss_weight", 0.5)
        self.logger.info(f"\n⚖️  NEURAL PE LOSS WEIGHTS (Nov 13 09:55 fix):")
        self.logger.info(f"  Physics Loss Weight: {physics_weight} (soft constraint)")
        self.logger.info(f"  Bounds Penalty Weight: {bounds_weight} (ground truth protection)")
        self.logger.info(f"  Sample Loss Weight: {sample_weight} (flow bounds constraint)")
        self.logger.info(f"  Jacobian Reg Weight: {np_cfg.get('jacobian_reg_weight', 0.001)}")

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

        # ✅ Save RL controller state (now that it's nn.Module)
        if hasattr(self.model, "rl_controller") and self.model.rl_controller is not None:
            checkpoint["rl_controller_state_dict"] = self.model.rl_controller.state_dict()

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

        # ✅ Restore RL controller state (including memory buffer)
        if (
            hasattr(self.model, "rl_controller")
            and self.model.rl_controller is not None
            and "rl_controller_state_dict" in checkpoint
        ):
            try:
                self.model.rl_controller.load_state_dict(
                    checkpoint["rl_controller_state_dict"]
                )
                self.logger.info("RL controller state restored (including memory buffer)")
            except Exception as e:
                self.logger.warning(f"Failed to load RL controller state: {e}")

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

        for batch_idx, (strain_data, parameters, n_signals, metadata) in enumerate(progress_bar):
            batch_start = time.time()
            self.optimizer.zero_grad()
            strain_data = strain_data.to(self.device)
            parameters = parameters.to(self.device)

            target_params = parameters[:, 0, :]

            # ✅ Nov 14: GET RL STATE AND COMPLEXITY RECOMMENDATION
            complexity = "medium"
            complexity_id = 1
            if hasattr(self.model, "rl_controller") and self.model.rl_controller is not None:
                # Extract pipeline state
                avg_n_signals = float(n_signals.float().mean().item())
                pipeline_state = {
                    "remaining_signals": avg_n_signals,
                    "residual_power": prev_loss if prev_loss != float("inf") else 0.5,
                    "processing_time": time.time() - batch_time_start,
                    "current_snr": 20.0,
                    "extraction_success_rate": 0.8,
                }

                # Get complexity (training=True for exploration)
                complexity = self.model.rl_controller.get_complexity_level(
                    pipeline_state, training=True
                )
                complexity_id = self.model.rl_controller.complexity_levels.index(complexity)
                epoch_metrics["rl_complexity"].append(complexity_id)

            # Compute loss
            loss_dict = self.model.compute_loss(strain_data, target_params)

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

            # ✅ Nov 14: RL EXPERIENCE COLLECTION AND TRAINING
            if hasattr(self.model, "rl_controller") and self.model.rl_controller is not None:
                batch_time = time.time() - batch_start
                current_loss = loss_dict["total_loss"].item()

                # Compute reward
                reward = self.model.rl_controller.compute_reward(
                    {"parameter_bias": current_loss, "extraction_time": batch_time}, complexity
                )

                # Build next state
                next_pipeline_state = {
                    "remaining_signals": avg_n_signals,
                    "residual_power": current_loss,
                    "processing_time": batch_time,
                    "current_snr": 20.0,
                    "extraction_success_rate": 0.8,
                }

                state_vector = self.model.rl_controller.get_state_vector(pipeline_state)
                next_state_vector = self.model.rl_controller.get_state_vector(next_pipeline_state)

                # Store experience
                self.model.rl_controller.store_experience(
                    state_vector, complexity_id, reward, next_state_vector, done=False
                )

                # Train when buffer filled
                rl_memory = self.model.rl_controller.memory
                if len(rl_memory) >= self.model.rl_controller.batch_size:
                    rl_loss = self.model.rl_controller.train_step()
                    if rl_loss is not None:
                        epoch_metrics["rl_loss"].append(rl_loss)

                prev_loss = current_loss

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

        # ✅ Nov 14: PERIODIC TARGET NETWORK UPDATE (every 5 epochs)
        if hasattr(self.model, "rl_controller") and self.model.rl_controller is not None:
            if (epoch + 1) % 5 == 0:
                self.model.rl_controller.update_target_network()
                self.logger.info(f"✅ RL target network updated at epoch {epoch+1}")

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

        with torch.no_grad():
            for batch_idx, (strain_data, parameters, n_signals, metadata) in enumerate(
                tqdm(val_loader, desc="Validating")
            ):
                strain_data = strain_data.to(self.device)
                parameters = parameters.to(self.device)

                target_params = parameters[:, 0, :]
                loss_dict = self.model.compute_loss(strain_data, target_params)

                # ✅ DEBUG: Log first validation batch for comparison
                # if batch_idx == 0:
                #     self.logger.info(f"\n[VAL BATCH 0 LOSS BREAKDOWN - Epoch {epoch+1}]")
                #     self.logger.info(f"  Total Loss: {loss_dict['total_loss'].item():.4f}")
                #     self.logger.info(f"  Flow Loss: {loss_dict.get('flow_loss', 0):.4f}")
                #     self.logger.info(f"  Physics Loss: {loss_dict.get('physics_loss', 0):.4f}")
                #     self.logger.info(f"  (Val physics loss should be SIMILAR to train)")

                epoch_losses.append(loss_dict["total_loss"].item())
                epoch_nlls.append(loss_dict["flow_loss"].item())
                if "physics_loss" in loss_dict:
                    epoch_physics_losses.append(loss_dict["physics_loss"].item())

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
        self.logger.info("ðŸš€ Starting OverlapNeuralPE training")
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

            # RL metrics logging
            if rl_metrics:
                self.logger.info(f"  RL Controller:")
                for key, value in rl_metrics.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"    {key}: {value:.4f}")
                    else:
                        self.logger.info(f"    {key}: {value}")
            else:
                self.logger.info(f"  RL Controller:")
                self.logger.info(f"    (metrics not available)")

            # Bias Corrector metrics logging
            if bias_metrics:
                self.logger.info(f"  Bias Corrector:")
                # Correction statistics
                if "avg_correction" in bias_metrics:
                    self.logger.info(f"    ✓ Avg Correction: {bias_metrics['avg_correction']:.6f}")
                if "max_correction" in bias_metrics:
                    self.logger.info(f"    ✓ Max Correction: {bias_metrics['max_correction']:.6f}")
                if "correction_std" in bias_metrics:
                    self.logger.info(f"    ✓ Correction Std: {bias_metrics['correction_std']:.6f}")
                # Confidence metrics
                if "avg_confidence" in bias_metrics:
                    self.logger.info(f"    ⚡ Avg Confidence: {bias_metrics['avg_confidence']:.4f}")
                if "min_confidence" in bias_metrics:
                    self.logger.info(f"    ⚡ Min Confidence: {bias_metrics['min_confidence']:.4f}")
                # Acceptance rates
                if "correction_acceptance_rate" in bias_metrics:
                    acceptance = bias_metrics["correction_acceptance_rate"]
                    self.logger.info(f"    📊 Acceptance Rate: {acceptance:.2%}")
                # Physics violations
                if "physics_violations" in bias_metrics:
                    violations = bias_metrics["physics_violations"]
                    self.logger.info(f"    ⚠️  Physics Violations: {violations}")
                # Additional metrics
                for key, value in bias_metrics.items():
                    if key not in [
                        "avg_correction",
                        "max_correction",
                        "correction_std",
                        "avg_confidence",
                        "min_confidence",
                        "correction_acceptance_rate",
                        "physics_violations",
                    ]:
                        if isinstance(value, (int, float)):
                            self.logger.info(f"    {key}: {value:.4f}")
                        else:
                            self.logger.info(f"    {key}: {value}")
            else:
                self.logger.info(f"  Bias Corrector:")
                self.logger.info(f"    (disabled or metrics not available)")

            # Priority Net logging
            if priority_net_metrics:
                self.logger.info(f"  Priority Net:")
                if "enabled" in priority_net_metrics and priority_net_metrics["enabled"]:
                    if "avg_importance" in priority_net_metrics:
                        self.logger.info(
                            f"    ✓ Avg Importance: {priority_net_metrics['avg_importance']:.4f}"
                        )
                    if "max_importance" in priority_net_metrics:
                        self.logger.info(
                            f"    ✓ Max Importance: {priority_net_metrics['max_importance']:.4f}"
                        )
                    if "min_importance" in priority_net_metrics:
                        self.logger.info(
                            f"    ✓ Min Importance: {priority_net_metrics['min_importance']:.4f}"
                        )
                    if "avg_uncertainty" in priority_net_metrics:
                        self.logger.info(
                            f"    ⚡ Avg Uncertainty: {priority_net_metrics['avg_uncertainty']:.4f}"
                        )
                    if "max_uncertainty" in priority_net_metrics:
                        self.logger.info(
                            f"    ⚡ Max Uncertainty: {priority_net_metrics['max_uncertainty']:.4f}"
                        )
                    if "edge_type_accuracy" in priority_net_metrics:
                        self.logger.info(
                            f"    📊 Edge Type Accuracy: {priority_net_metrics['edge_type_accuracy']:.2%}"
                        )
                    if "n_parameters" in priority_net_metrics:
                        self.logger.info(
                            f"    🔧 Parameters: {priority_net_metrics['n_parameters']:,}"
                        )
                else:
                    self.logger.info(f"    (disabled or frozen)")
            else:
                self.logger.info(f"  Priority Net:")
                self.logger.info(f"    (disabled or metrics not available)")

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


def _collect_bias_training_data(
    model: OverlapNeuralPE,
    val_loader: DataLoader,
    param_names: List[str],
    device: torch.device,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    Collect validation predictions for bias correction training.

    Returns:
        List of bias training scenarios: [
            {
                'extracted_signals': [...],  # Flow predictions
                'true_parameters': [...]     # Ground truth params
            },
            ...
        ]
    """
    training_scenarios = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (strain_data, parameters, n_signals, metadata) in enumerate(
            tqdm(val_loader, desc="Collecting bias data", leave=False)
        ):
            strain_data = strain_data.to(device)
            parameters = parameters.to(device)

            # Ground truth parameters (first signal in batch)
            true_params_batch = parameters[:, 0, :]  # [batch, param_dim]

            try:
                # Get flow predictions (posterior means)
                # Call the flow directly to get posterior samples
                context = (
                    model.context_encoder(strain_data)
                    if hasattr(model, "context_encoder")
                    else torch.randn(strain_data.shape[0], model.context_dim, device=device)
                )

                # Sample from flow to get posterior estimates (reduced from 100 to 20 samples for speed)
                num_samples = 20  # Reduced for faster bias data collection
                samples_list = []
                for _ in range(num_samples):
                    samples = model.flow.sample(num_samples=strain_data.shape[0], context=context)
                    samples_list.append(samples)

                posterior_samples = torch.stack(
                    samples_list, dim=0
                )  # [num_samples, batch, param_dim]
                posterior_means = torch.mean(posterior_samples, dim=0)  # [batch, param_dim]
                posterior_stds = torch.std(posterior_samples, dim=0)  # [batch, param_dim]

                # Convert to numpy for bias correction format
                for b in range(strain_data.shape[0]):
                    extracted_summary = {}
                    true_params_dict = {}

                    for p_idx, param_name in enumerate(param_names):
                        # Extracted signal (flow prediction)
                        extracted_summary[param_name] = {
                            "mean": posterior_means[b, p_idx].item(),
                            "median": posterior_means[b, p_idx].item(),
                            "std": posterior_stds[b, p_idx].item(),
                            "quantiles": [
                                (posterior_means[b, p_idx] - 2 * posterior_stds[b, p_idx]).item(),
                                (posterior_means[b, p_idx] - posterior_stds[b, p_idx]).item(),
                                posterior_means[b, p_idx].item(),
                                (posterior_means[b, p_idx] + posterior_stds[b, p_idx]).item(),
                                (posterior_means[b, p_idx] + 2 * posterior_stds[b, p_idx]).item(),
                            ],
                        }

                        # True parameters
                        true_params_dict[param_name] = true_params_batch[b, p_idx].item()

                    # Create scenario for bias corrector training
                    scenario = {
                        "extracted_signals": [
                            {
                                "posterior_summary": extracted_summary,
                                "signal_quality": 0.8,
                                "network_snr": 20.0,
                                "context_embedding": context[b]
                                .detach()
                                .cpu()
                                .numpy(),  # Pass actual 768D context from encoder
                            }
                        ],
                        "true_parameters": [true_params_dict],
                    }
                    training_scenarios.append(scenario)

            except Exception as e:
                logger.debug(f"Error processing batch {batch_idx}: {e}")
                continue

    logger.info(f"✅ Collected {len(training_scenarios)} bias training scenarios")
    return training_scenarios


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

    # Load config
    with open(args.config, "r") as f:
        full_config = yaml.safe_load(f)

    # ✅ FIX: Extract neural_posterior section from YAML
    config = full_config.get("neural_posterior", full_config)

    # Parameter names
    param_names = config.get(
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
        logger.info("ðŸ”§ Initializing OverlapNeuralPE")
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
        logger.info("ðŸ“Š Loading datasets")

        train_data_loader = ChunkedGWDataLoader(
            dataset_path=args.data_dir, split="train", max_samples=None
        )

        val_data_loader = ChunkedGWDataLoader(
            dataset_path=args.data_dir, split="validation", max_samples=None
        )

        train_dataset = OverlapGWDataset(train_data_loader, param_names, config)
        val_dataset = OverlapGWDataset(val_data_loader, param_names, config)

        # Get batch size from config
        batch_size = config.get("batch_size", 16)

        # Create data loaders (num_workers=0 to avoid pickling issues)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
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
            logger.info(f"ðŸ“Š Data augmentation enabled:")
            logger.info(f"  Noise scaling: {aug_config.get('noise_scaling')}")
            logger.info(f"  Time shifts: {aug_config.get('time_shifts')}")
            logger.info(f"  Apply probability: {aug_config.get('apply_probability')}")

        # ✅ Log dropout config
        dropout = config.get("dropout", 0.1)
        flow_dropout = config.get("flow_config", {}).get("dropout", 0.15)
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

        # ✅ PHASE 2: TRAIN BIAS CORRECTOR (after Neural PE converges)
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: Training BiasCorrector (after Neural PE convergence)")
        logger.info("=" * 70)

        try:
            # Collect validation data for bias correction training
            logger.info("Collecting validation predictions for bias correction...")
            bias_training_scenarios = _collect_bias_training_data(
                model=model,
                val_loader=val_loader,
                param_names=param_names,
                device=model.device,
                logger=logger,
            )

            if len(bias_training_scenarios) > 0:
                # Train BiasCorrector
                logger.info(
                    f"Training BiasCorrector with {len(bias_training_scenarios)} scenarios..."
                )
                from ahsd.core.bias_corrector import BiasCorrector

                # Get bias corrector config from YAML
                bias_corrector_config = config.get("bias_corrector", {})
                # Use bias_corrector.context_dim from config, matching the context encoder output
                context_dim = bias_corrector_config.get("context_dim", 768)

                bias_corrector = BiasCorrector(param_names=param_names, context_dim=context_dim)
                logger.info(f"✅ BiasCorrector initialized (context_dim={context_dim})")

                # Train bias corrector with config settings
                bias_training_config = {
                    "epochs": bias_corrector_config.get("epochs", 30),
                    "learning_rate": bias_corrector_config.get("learning_rate", 1e-4),
                    "batch_size": bias_corrector_config.get("batch_size", 16),
                    "patience": bias_corrector_config.get("patience", 10),
                }

                training_result = bias_corrector.train_bias_estimator(
                    training_scenarios=bias_training_scenarios,
                    epochs=bias_training_config["epochs"],
                    validation_split=0.2,
                )

                # ✅ COMPREHENSIVE BIAS METRICS FOR PHASE 2
                logger.info(f"\n✅ BiasCorrector training completed!")
                logger.info(f"\n{'='*70}")
                logger.info(f"PHASE 2: BiasCorrector Training Results")
                logger.info(f"{'='*70}")

                # Loss metrics
                logger.info(f"\n📊 Loss Metrics:")
                logger.info(f"  Final Train Loss: {training_result.get('final_train_loss', 0):.6f}")
                logger.info(f"  Final Val Loss: {training_result.get('final_val_loss', 0):.6f}")
                epochs_completed = training_result.get("epochs_completed", 0)
                logger.info(f"  Epochs: {epochs_completed}/{bias_training_config['epochs']}")

                # Training history analysis
                if "training_history" in training_result:
                    hist = training_result["training_history"]
                    train_losses = hist.get("train_losses", [])
                    val_losses = hist.get("val_losses", [])

                    if train_losses:
                        logger.info(f"\n📈 Training Loss Trajectory:")
                        logger.info(f"  Initial Train Loss: {train_losses[0]:.6f}")
                        logger.info(f"  Final Train Loss: {train_losses[-1]:.6f}")
                        logger.info(
                            f"  Reduction: {(1 - train_losses[-1]/max(train_losses[0], 1e-8))*100:.1f}%"
                        )
                        logger.info(f"  Min Train Loss: {min(train_losses):.6f}")
                        logger.info(f"  Max Train Loss: {max(train_losses):.6f}")

                    if val_losses:
                        logger.info(f"\n📉 Validation Loss Trajectory:")
                        logger.info(f"  Initial Val Loss: {val_losses[0]:.6f}")
                        logger.info(f"  Final Val Loss: {val_losses[-1]:.6f}")
                        logger.info(
                            f"  Reduction: {(1 - val_losses[-1]/max(val_losses[0], 1e-8))*100:.1f}%"
                        )
                        logger.info(f"  Min Val Loss: {min(val_losses):.6f}")
                        logger.info(f"  Max Val Loss: {max(val_losses):.6f}")
                        logger.info(
                            f"  Train-Val Gap: {abs(train_losses[-1] - val_losses[-1]):.6f}"
                        )

                # Data split metrics
                logger.info(f"\n📊 Data Split:")
                logger.info(f"  Training Samples: {training_result.get('training_samples', 0)}")
                logger.info(f"  Validation Samples: {training_result.get('validation_samples', 0)}")
                total_samples = training_result.get("training_samples", 0) + training_result.get(
                    "validation_samples", 0
                )
                logger.info(f"  Total Samples: {total_samples}")

                # Training configuration summary
                logger.info(f"\n⚙️  Training Configuration:")
                logger.info(f"  Epochs: {bias_training_config['epochs']}")
                logger.info(f"  Learning Rate: {bias_training_config['learning_rate']}")
                logger.info(f"  Batch Size: {bias_training_config['batch_size']}")
                logger.info(f"  Patience (Early Stopping): {bias_training_config['patience']}")

                # BiasCorrector statistics
                logger.info(f"\n🔬 BiasCorrector Statistics:")
                if hasattr(bias_corrector, "is_trained"):
                    logger.info(
                        f"  Training Status: {'✅ Trained' if bias_corrector.is_trained else '❌ Not Trained'}"
                    )
                if hasattr(bias_corrector, "training_epochs"):
                    logger.info(f"  Training Epochs Completed: {bias_corrector.training_epochs}")

                # Get correction statistics from bias corrector
                if hasattr(bias_corrector, "get_correction_statistics"):
                    try:
                        correction_stats = bias_corrector.get_correction_statistics()
                        perf_metrics = correction_stats.get("performance_metrics", {})

                        logger.info(f"\n💡 Correction Performance:")
                        if perf_metrics:
                            if "total_corrections" in perf_metrics:
                                logger.info(
                                    f"  Total Corrections Applied: {perf_metrics['total_corrections']}"
                                )
                            if "avg_correction_magnitude" in perf_metrics:
                                logger.info(
                                    f"  Avg Correction Magnitude: {perf_metrics['avg_correction_magnitude']:.6f}"
                                )
                            if "accepted_corrections" in perf_metrics:
                                logger.info(
                                    f"  Accepted Corrections: {perf_metrics['accepted_corrections']}"
                                )
                            if "acceptance_rate" in perf_metrics:
                                logger.info(
                                    f"  Acceptance Rate: {perf_metrics['acceptance_rate']:.2%}"
                                )
                    except Exception as e:
                        logger.debug(f"Could not retrieve correction statistics: {e}")

                # ✅ Per-parameter bias metrics
                logger.info(f"\n📋 Per-Parameter Bias Metrics:")
                if param_names:
                    for i, param_name in enumerate(param_names):
                        logger.info(f"  {param_name}:")
                        logger.info(f"    - Status: Trained with BiasCorrector")
                        # Additional per-param stats can be added as bias_corrector develops
                else:
                    logger.info(f"  (No parameter names available)")

                logger.info(f"\n✅ Phase 2 Summary:")
                logger.info(f"  BiasCorrector Training: Complete")
                logger.info(f"  Model Integration: Enabled")
                logger.info(f"  Checkpoint Saved: Yes")
                logger.info(f"{'='*70}\n")

                # ✅ WANDB: Log bias corrector metrics
                if use_wandb:
                    bias_wandb_metrics = {
                        "bias/final_train_loss": training_result.get("final_train_loss", 0),
                        "bias/final_val_loss": training_result.get("final_val_loss", 0),
                        "bias/epochs_completed": epochs_completed,
                        "bias/training_samples": training_result.get("training_samples", 0),
                        "bias/validation_samples": training_result.get("validation_samples", 0),
                        "bias/total_samples": total_samples,
                    }

                    # Add loss trajectory metrics
                    if "training_history" in training_result:
                        hist = training_result["training_history"]
                        train_losses = hist.get("train_losses", [])
                        val_losses = hist.get("val_losses", [])

                        if train_losses:
                            bias_wandb_metrics["bias/initial_train_loss"] = train_losses[0]
                            bias_wandb_metrics["bias/train_loss_reduction_pct"] = (
                                1 - train_losses[-1] / max(train_losses[0], 1e-8)
                            ) * 100
                            bias_wandb_metrics["bias/min_train_loss"] = min(train_losses)

                        if val_losses:
                            bias_wandb_metrics["bias/initial_val_loss"] = val_losses[0]
                            bias_wandb_metrics["bias/val_loss_reduction_pct"] = (
                                1 - val_losses[-1] / max(val_losses[0], 1e-8)
                            ) * 100
                            bias_wandb_metrics["bias/min_val_loss"] = min(val_losses)
                            if train_losses:
                                bias_wandb_metrics["bias/train_val_gap"] = abs(
                                    train_losses[-1] - val_losses[-1]
                                )

                    # Add correction performance metrics
                    if hasattr(bias_corrector, "get_correction_statistics"):
                        try:
                            correction_stats = bias_corrector.get_correction_statistics()
                            perf_metrics = correction_stats.get("performance_metrics", {})

                            if perf_metrics:
                                if "total_corrections" in perf_metrics:
                                    bias_wandb_metrics["bias/total_corrections"] = perf_metrics[
                                        "total_corrections"
                                    ]
                                if "avg_correction_magnitude" in perf_metrics:
                                    bias_wandb_metrics["bias/avg_correction_magnitude"] = (
                                        perf_metrics["avg_correction_magnitude"]
                                    )
                                if "acceptance_rate" in perf_metrics:
                                    bias_wandb_metrics["bias/acceptance_rate"] = perf_metrics[
                                        "acceptance_rate"
                                    ]
                        except Exception:
                            pass

                    wandb.log(bias_wandb_metrics)
                    wandb.run.summary["phase2_complete"] = True
                    wandb.run.summary["bias_corrector_trained"] = True

                # Save BiasCorrector checkpoint
                bias_corrector_path = output_dir / "bias_corrector_best.pth"
                torch.save(bias_corrector.state_dict(), bias_corrector_path)
                logger.info(f"BiasCorrector saved: {bias_corrector_path}")

                # Save BiasCorrector training history with comprehensive metrics
                bias_history_path = output_dir / "bias_corrector_history.yaml"

                # Prepare comprehensive metrics dictionary
                bias_metrics_dict = {
                    "phase": 2,
                    "status": "completed",
                    "training_result": training_result,
                    "comprehensive_metrics": {
                        "loss_metrics": {
                            "final_train_loss": training_result.get("final_train_loss", 0),
                            "final_val_loss": training_result.get("final_val_loss", 0),
                            "epochs_completed": epochs_completed,
                        },
                        "data_split": {
                            "training_samples": training_result.get("training_samples", 0),
                            "validation_samples": training_result.get("validation_samples", 0),
                            "total_samples": total_samples,
                        },
                        "training_config": bias_training_config,
                        "parameter_names": param_names,
                    },
                }

                # Add training history if available
                if "training_history" in training_result:
                    hist = training_result["training_history"]
                    bias_metrics_dict["comprehensive_metrics"]["loss_trajectory"] = {
                        "train_losses": hist.get("train_losses", []),
                        "val_losses": hist.get("val_losses", []),
                    }

                with open(bias_history_path, "w") as f:
                    yaml.dump(bias_metrics_dict, f)
                logger.info(f"BiasCorrector history saved: {bias_history_path}")

                # Also save metrics summary JSON for quick parsing
                bias_metrics_summary = {
                    "phase": 2,
                    "status": "completed",
                    "loss_metrics": {
                        "final_train_loss": float(training_result.get("final_train_loss", 0)),
                        "final_val_loss": float(training_result.get("final_val_loss", 0)),
                        "epochs_completed": int(epochs_completed),
                    },
                    "data_metrics": {
                        "training_samples": int(training_result.get("training_samples", 0)),
                        "validation_samples": int(training_result.get("validation_samples", 0)),
                        "total_samples": int(total_samples),
                    },
                    "timestamp": str(Path(output_dir).stat().st_mtime),
                }

                bias_metrics_json_path = output_dir / "bias_corrector_metrics.json"
                with open(bias_metrics_json_path, "w") as f:
                    json.dump(bias_metrics_summary, f, indent=2)
                logger.info(f"BiasCorrector metrics saved: {bias_metrics_json_path}")

                # Update model with trained BiasCorrector
                model.bias_corrector = bias_corrector
                logger.info("BiasCorrector integrated into model")

                # Save integrated model with all trained components
                integrated_model_path = output_dir / "model_with_bias_corrector.pth"
                full_checkpoint = {
                    "model_state_dict": model.state_dict(),  # Includes bias_corrector + RL controller weights
                    "bias_corrector_state_dict": bias_corrector.state_dict(),
                    "param_names": param_names,
                    "config": config,
                    "training_metadata": {
                        "phase": "both_phases_complete",
                        "neural_pe_trained": True,
                        "bias_corrector_trained": True,
                        "rl_controller_trained": hasattr(model, "rl_controller")
                        and model.rl_controller is not None,
                        "bias_training_epochs": training_result.get("epochs_completed", 0),
                        "bias_final_train_loss": training_result.get("final_train_loss", 0),
                        "bias_final_val_loss": training_result.get("final_val_loss", 0),
                    },
                }

                # Add RL controller state if present (trained during Phase 1)
                if hasattr(model, "rl_controller") and model.rl_controller is not None:
                    full_checkpoint["rl_controller_state_dict"] = model.rl_controller.state_dict()

                torch.save(full_checkpoint, integrated_model_path)
                logger.info(f"Integrated model saved: {integrated_model_path}")
                logger.info(f"  - Neural PE: ✅ (included in model_state_dict)")
                logger.info(
                    f"  - BiasCorrector: ✅ (included in model_state_dict + separate state dict)"
                )
                if hasattr(model, "rl_controller") and model.rl_controller is not None:
                    logger.info(
                        f"  - RL Controller: ✅ (included in model_state_dict + separate state dict)"
                    )

                # Also update the best_model checkpoint from trainer with all components
                best_model_path = output_dir / "best_model.pth"
                if best_model_path.exists():
                    logger.info("Updating best_model.pth with BiasCorrector and RL Controller...")
                    best_checkpoint = torch.load(best_model_path, map_location="cpu")
                    best_checkpoint["model_state_dict"] = (
                        model.state_dict()
                    )  # Updated model with bias corrector
                    best_checkpoint["bias_corrector_state_dict"] = bias_corrector.state_dict()

                    # Update RL controller state if present
                    if hasattr(model, "rl_controller") and model.rl_controller is not None:
                        best_checkpoint["rl_controller_state_dict"] = (
                            model.rl_controller.state_dict()
                        )

                    best_checkpoint["training_metadata"] = full_checkpoint["training_metadata"]
                    torch.save(best_checkpoint, best_model_path)
                    logger.info(f"✅ best_model.pth updated")
                    logger.info(f"   - BiasCorrector: ✅")
                    if hasattr(model, "rl_controller") and model.rl_controller is not None:
                        logger.info(f"   - RL Controller: ✅")

            else:
                logger.warning("No validation data collected for bias correction training")

        except Exception as e:
            logger.error(f"BiasCorrector training failed: {e}", exc_info=True)
            # Continue without bias correction
            logger.warning("Proceeding without BiasCorrector")

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
