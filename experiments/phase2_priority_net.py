#!/usr/bin/env python3
"""
COMPLETE Phase 2 Training Script with Enhanced PriorityNet
Integrates improved architecture with existing chunked dataset infrastructure
Full working implementation - no code reuse
"""

import sys
import os
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import argparse
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import yaml
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import random
from scipy.stats import spearmanr, kendalltau

# Setup project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def safe_float(value):
    """Convert tensor or float to Python float safely"""
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)

# Import Enhanced PriorityNet
from ahsd.core.priority_net import (
    EnhancedPriorityNet,
    EnhancedPriorityNetTrainer
)

def setup_logging(verbose: bool = False):
    """Initialize logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('phase2_enhanced_training.log'),
            logging.StreamHandler()
        ]
    )

class ChunkedGWDataLoader:
    """
    Loads chunked gravitational wave dataset from newDataset directory structure.
    Handles train/validation/test splits with proper chunk loading.
    """

    def __init__(self, dataset_path: str, split: str = 'train', max_samples: Optional[int] = None):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.max_samples = max_samples
        self.logger = logging.getLogger(__name__)

        self._load_split_info()
        self.samples = self._load_all_samples()

        self.logger.info(f"✅ {split.upper()} dataset loaded: {len(self.samples)} samples")

    def _load_split_info(self):
        """Load split metadata"""
        split_dir = self.dataset_path / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        split_info_file = split_dir / 'split_info.json'
        if split_info_file.exists():
            with open(split_info_file, 'r') as f:
                self.split_info = json.load(f)
        else:
            chunk_files = list(split_dir.glob('chunk_*.pkl'))
            self.split_info = {
                'n_chunks': len(chunk_files),
                'chunk_size': 500,
                'file_pattern': 'chunk_XXXX.pkl'
            }

        self.n_chunks = self.split_info['n_chunks']
        self.chunk_size = self.split_info['chunk_size']

        self.logger.info(f"📊 {self.split}: {self.n_chunks} chunks, {self.chunk_size} samples per chunk")

    def _load_all_samples(self) -> List[Dict]:
        """Load all samples from chunks"""
        all_samples = []
        total_loaded = 0

        self.logger.info(f"💾 Loading {self.split} chunks...")

        for chunk_id in tqdm(range(self.n_chunks), desc=f"Loading {self.split}"):
            chunk_file = self.dataset_path / self.split / f'chunk_{chunk_id:04d}.pkl'

            if chunk_file.exists():
                try:
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)

                    for sample in chunk_data:
                        all_samples.append(sample)
                        total_loaded += 1

                        if self.max_samples and total_loaded >= self.max_samples:
                            self.logger.info(f"⏹️ Reached max samples limit: {self.max_samples}")
                            return all_samples

                except Exception as e:
                    self.logger.warning(f"Failed to load chunk {chunk_id}: {e}")
                    continue

        return all_samples

    def convert_to_priority_scenarios(self, create_overlaps: bool = True,
                                     overlap_probability: float = 0.3) -> List[Dict]:
        """Convert GW samples to PriorityNet training scenarios"""

        self.logger.info("🔄 Converting GW samples to PriorityNet scenarios...")

        scenarios = []
        single_signal_samples = []

        # Separate by type
        for sample in self.samples:
            metadata = sample.get('metadata', {})
            event_type = metadata.get('event_type', 'unknown')
            n_signals = metadata.get('n_signals', 1)

            if event_type == 'noise':
                scenario = self._convert_noise_sample(sample)
                if scenario:
                    scenarios.append(scenario)
            elif n_signals == 1:
                single_signal_samples.append(sample)
                scenario = self._convert_single_sample(sample)
                if scenario:
                    scenarios.append(scenario)
            else:
                scenario = self._convert_multi_sample(sample)
                if scenario:
                    scenarios.append(scenario)

        # Create artificial overlaps for training
        if create_overlaps and len(single_signal_samples) >= 2:
            if self.split == 'train':
                n_artificial = min(6000, len(single_signal_samples) // 2)
            elif self.split == 'validation':
                n_artificial = min(400, len(single_signal_samples) // 2)
            elif self.split == 'test':
                n_artificial = min(400, len(single_signal_samples) // 2)
            else:
                n_artificial = 0

            if n_artificial > 0:
                self.logger.info(f"   Creating {n_artificial} artificial overlaps for {self.split}...")
                created = 0
                attempts = 0
                max_attempts = n_artificial * 2

                while created < n_artificial and attempts < max_attempts:
                    attempts += 1
                    scenario = self._create_artificial_overlap(single_signal_samples)
                    if scenario:
                        scenarios.append(scenario)
                        created += 1

                self.logger.info(f"✅ Created {created} artificial overlaps")

        self.logger.info(f"✅ Created {len(scenarios)} total scenarios")
        return scenarios

    def _convert_noise_sample(self, sample: Dict) -> Optional[Dict]:
        """Convert noise sample to scenario"""
        try:
            metadata = sample.get('metadata', {})
            noise_param = {
                'mass_1': 0.0, 'mass_2': 0.0, 'luminosity_distance': 0.0,
                'network_snr': 0.0, 'ra': 0.0, 'dec': 0.0,
                'theta_jn': 0.0, 'psi': 0.0, 'phase': 0.0,
                'geocent_time': 0.0, 'a_1': 0.0, 'a_2': 0.0,
                'tilt_1': 0.0, 'tilt_2': 0.0, 'phi_12': 0.0, 'phi_jl': 0.0,
                'event_type': 'noise'
            }

            return {
                'scenario_id': sample.get('sample_id', 'unknown'),
                'true_parameters': [noise_param],
                'baseline_biases': [],
                'detector_data': sample.get('detector_data', {}),
                'metadata': metadata
            }
        except:
            return None

    def _convert_single_sample(self, sample: Dict) -> Optional[Dict]:
        """Convert single GW sample to scenario"""
        try:
            metadata = sample.get('metadata', {})
            signal_parameters = metadata.get('signal_parameters', [])

            if not signal_parameters:
                return None

            sig_param = signal_parameters[0]
            event_type = metadata.get('event_type', 'BBH')

            priority_param = {
                'mass_1': sig_param.get('mass_1', 30.0),
                'mass_2': sig_param.get('mass_2', 25.0),
                'luminosity_distance': sig_param.get('luminosity_distance', 500.0),
                'network_snr': metadata.get('network_snr', 10.0),
                'ra': sig_param.get('ra', 0.0),
                'dec': sig_param.get('dec', 0.0),
                'theta_jn': sig_param.get('theta_jn', 0.0),
                'psi': sig_param.get('psi', 0.0),
                'phase': sig_param.get('phase', 0.0),
                'geocent_time': sig_param.get('geocent_time', 0.0),
                'a_1': sig_param.get('a1', 0.0),
                'a_2': sig_param.get('a2', 0.0),
                'tilt_1': sig_param.get('tilt1', 0.0),
                'tilt_2': sig_param.get('tilt2', 0.0),
                'phi_12': sig_param.get('phi_12', 0.0),
                'phi_jl': sig_param.get('phi_jl', 0.0),
                'event_type': event_type
            }

            return {
                'scenario_id': sample.get('sample_id', 'unknown'),
                'true_parameters': [priority_param],
                'baseline_biases': [],
                'detector_data': sample.get('detector_data', {}),
                'metadata': metadata
            }
        except:
            return None

    def _convert_multi_sample(self, sample: Dict) -> Optional[Dict]:
        """Convert multi-signal sample to scenario"""
        try:
            metadata = sample.get('metadata', {})
            signal_parameters = metadata.get('signal_parameters', [])

            if len(signal_parameters) < 2:
                return self._convert_single_sample(sample)

            true_parameters = []
            total_snr = metadata.get('network_snr', 20.0)
            n_signals = len(signal_parameters)

            for i, sig_param in enumerate(signal_parameters):
                individual_snr = total_snr / np.sqrt(n_signals)
                event_type = metadata.get('event_type', 'BBH')

                priority_param = {
                    'mass_1': sig_param.get('mass_1', 30.0),
                    'mass_2': sig_param.get('mass_2', 25.0),
                    'luminosity_distance': sig_param.get('luminosity_distance', 500.0),
                    'network_snr': individual_snr,
                    'ra': sig_param.get('ra', 0.0),
                    'dec': sig_param.get('dec', 0.0),
                    'theta_jn': sig_param.get('theta_jn', 0.0),
                    'psi': sig_param.get('psi', 0.0),
                    'phase': sig_param.get('phase', 0.0),
                    'geocent_time': sig_param.get('geocent_time', i * 0.5),
                    'a_1': sig_param.get('a1', 0.0),
                    'a_2': sig_param.get('a2', 0.0),
                    'tilt_1': sig_param.get('tilt1', 0.0),
                    'tilt_2': sig_param.get('tilt2', 0.0),
                    'phi_12': sig_param.get('phi_12', 0.0),
                    'phi_jl': sig_param.get('phi_jl', 0.0),
                    'event_type': event_type,
                    'is_overlap': True,
                    'n_overlapping_signals': n_signals
                }
                true_parameters.append(priority_param)

            return {
                'scenario_id': sample.get('sample_id', 'unknown'),
                'true_parameters': true_parameters,
                'baseline_biases': [],
                'detector_data': sample.get('detector_data', {}),
                'metadata': metadata
            }
        except:
            return None

    def _create_artificial_overlap(self, single_samples: List[Dict]) -> Optional[Dict]:
        """Create artificial overlap scenario"""
        try:
            n_signals = random.choice([2, 3])
            if len(single_samples) < n_signals:
                return None

            selected_samples = random.sample(single_samples, n_signals)
            combined_parameters = []

            for i, sample in enumerate(selected_samples):
                metadata = sample.get('metadata', {})
                signal_params = metadata.get('signal_parameters', [])

                if not signal_params:
                    continue

                sig_param = signal_params[0]
                time_offset = i * random.uniform(0.2, 1.0)
                snr_reduction = random.uniform(0.6, 0.8)
                event_type = metadata.get('event_type', 'BBH')

                priority_param = {
                    'mass_1': sig_param.get('mass_1', 30.0),
                    'mass_2': sig_param.get('mass_2', 25.0),
                    'luminosity_distance': sig_param.get('luminosity_distance', 500.0),
                    'network_snr': metadata.get('network_snr', 10.0) * snr_reduction,
                    'ra': sig_param.get('ra', 0.0),
                    'dec': sig_param.get('dec', 0.0),
                    'theta_jn': sig_param.get('theta_jn', 0.0),
                    'psi': sig_param.get('psi', 0.0),
                    'phase': sig_param.get('phase', 0.0),
                    'geocent_time': time_offset,
                    'a_1': sig_param.get('a1', 0.0),
                    'a_2': sig_param.get('a2', 0.0),
                    'tilt_1': sig_param.get('tilt1', 0.0),
                    'tilt_2': sig_param.get('tilt2', 0.0),
                    'phi_12': sig_param.get('phi_12', 0.0),
                    'phi_jl': sig_param.get('phi_jl', 0.0),
                    'event_type': event_type,
                    'is_overlap': True,
                    'artificial': True,
                    'n_overlapping_signals': n_signals
                }
                combined_parameters.append(priority_param)

            return {
                'scenario_id': f"artificial_overlap_{random.randint(1000, 9999)}",
                'true_parameters': combined_parameters,
                'baseline_biases': [],
                'detector_data': {},
                'metadata': {'event_type': 'overlap', 'n_signals': len(combined_parameters)}
            }
        except:
            return None


class PriorityNetDataset(Dataset):
    """PyTorch Dataset for Enhanced PriorityNet training"""

    def __init__(self, scenarios: List[Dict], split_name: str = 'train'):
        self.data = []
        self.split_name = split_name
        self.logger = logging.getLogger(__name__)

        for scenario in scenarios:
            try:
                true_params = scenario.get('true_parameters', [])
                if not true_params:
                    continue

                detections = []
                for signal in true_params:
                    detection = {
                        'mass_1': signal.get('mass_1', 30.0),
                        'mass_2': signal.get('mass_2', 25.0),
                        'luminosity_distance': signal.get('luminosity_distance', 500.0),
                        'network_snr': signal.get('network_snr', 10.0),
                        'ra': signal.get('ra', 0.0),
                        'dec': signal.get('dec', 0.0),
                        'theta_jn': signal.get('theta_jn', 0.0),
                        'psi': signal.get('psi', 0.0),
                        'phase': signal.get('phase', 0.0),
                        'geocent_time': signal.get('geocent_time', 0.0),
                        'a_1': signal.get('a_1', 0.0),
                        'a_2': signal.get('a_2', 0.0),
                        'tilt_1': signal.get('tilt_1', 0.0),
                        'tilt_2': signal.get('tilt_2', 0.0),
                        'phi_12': signal.get('phi_12', 0.0),
                        'phi_jl': signal.get('phi_jl', 0.0),
                        'event_type': signal.get('event_type', 'BBH')
                    }
                    detections.append(detection)

                priorities = self._compute_priorities(true_params)
                if priorities is None or len(priorities) == 0:
                    continue

                scenario_data = {
                    'scenario_id': scenario.get('scenario_id', 'unknown'),
                    'detections': detections,
                    'priorities': priorities,
                    'metadata': scenario.get('metadata', {})
                }

                self.data.append(scenario_data)

            except Exception as e:
                self.logger.debug(f"Error processing scenario: {e}")
                continue

        self.logger.info(f"✅ {split_name.upper()} dataset: {len(self.data)} scenarios")

    def _compute_priorities(self, signals: List[Dict]) -> torch.Tensor:
        """Compute extraction priorities for signals"""
        n_signals = len(signals)
        priorities = torch.zeros(n_signals)

        for i, signal in enumerate(signals):
            try:
                event_type = str(signal.get('event_type', 'BBH')).upper()

                if event_type == 'NOISE':
                    priorities[i] = 0.0
                    continue

                snr = max(0.1, float(signal.get('network_snr', 10.0)))
                distance = max(1.0, float(signal.get('luminosity_distance', 500.0)))
                m1 = max(0.1, float(signal.get('mass_1', 30.0)))
                m2 = max(0.1, float(signal.get('mass_2', 25.0)))

                if m2 > m1:
                    m1, m2 = m2, m1

                total_mass = m1 + m2
                chirp_mass = (m1 * m2)**(3/5) / total_mass**(1/5) if total_mass > 0 else 1.0

                # SNR priority
                if snr >= 20.0:
                    snr_priority = 1.0 + 0.15 * np.log10(snr / 20.0)
                elif snr >= 12.0:
                    snr_priority = snr / 12.0
                else:
                    snr_priority = 0.7 + 0.3 * snr / 12.0
                snr_priority = min(snr_priority, 1.5)

                # Mass priority
                if 15.0 <= total_mass <= 50.0:
                    mass_priority = 1.0
                elif 50.0 < total_mass <= 200.0:
                    mass_priority = 1.1
                else:
                    mass_priority = 0.9

                # Distance priority
                base_horizon = 1000.0 if event_type == 'BBH' else 200.0
                effective_horizon = base_horizon * (chirp_mass / 30.0)**(5/6)

                if distance <= effective_horizon:
                    distance_priority = 1.0
                elif distance <= 2.0 * effective_horizon:
                    distance_priority = 0.6 + 0.4 * (2.0 * effective_horizon - distance) / effective_horizon
                else:
                    distance_priority = max(0.1, effective_horizon / distance)

                # Combined priority
                base_priority = (
                    0.40 * snr_priority +
                    0.30 * distance_priority +
                    0.30 * mass_priority
                )

                # Overlap bonus
                if signal.get('is_overlap', False):
                    base_priority += 0.05

                final_priority = max(0.3, min(base_priority, 2.0))
                priorities[i] = final_priority

            except Exception as e:
                self.logger.debug(f"Priority calculation error: {e}")
                priorities[i] = 0.5
                continue

        return torch.clamp(priorities, min=0.01, max=2.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_priority_batch(batch: List[Dict]) -> Tuple:
    """Collate function for variable-length sequences"""
    detections_batch = []
    priorities_batch = []

    for item in batch:
        detections_batch.append(item['detections'])
        priorities_batch.append(item['priorities'])

    return detections_batch, priorities_batch


def train_enhanced_priority_net(config, train_dataset, val_dataset, output_dir: Path) -> Dict:
    """
    Enhanced training with validation monitoring and comprehensive diagnostics
    """

    logging.info("🧠 Training Enhanced PriorityNet...")

    # Initialize Enhanced model
    use_strain = getattr(config, 'use_strain', False)
    model = EnhancedPriorityNet(config, use_strain=use_strain)
    trainer = EnhancedPriorityNetTrainer(model, config)

    logging.info(f"📊 Model configuration:")
    logging.info(f"   Architecture: Enhanced PriorityNet")
    logging.info(f"   Strain encoding: {'Enabled' if use_strain else 'Disabled'}")
    logging.info(f"   Parameters: ~{sum(p.numel() for p in model.parameters()):,}")

    # Training parameters
    batch_size = getattr(config, 'batch_size', 32)
    n_epochs = getattr(config, 'epochs', 500)
    patience = getattr(config, 'patience', 30)

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_priority_batch, num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_priority_batch, num_workers=0
    )

    # Training state
    best_val_loss = float('inf')
    patience_counter = 0

    training_metrics = {
        'train_losses': [], 'train_mse': [], 'train_ranking': [], 'train_uncertainty': [],
        'val_losses': [], 'val_mse': [], 'val_ranking': [], 'val_uncertainty': [],
        'epochs_completed': 0, 'best_epoch': 0, 'learning_rates': []
    }

    logging.info(f"🚀 Starting training: {n_epochs} epochs, patience={patience}")

    # Training loop
    for epoch in range(n_epochs):
        # TRAINING PHASE
        model.train()
        train_losses = {'total': [], 'mse': [], 'ranking': [], 'uncertainty': []}

        train_pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{n_epochs}')
        for detections_batch, priorities_batch in train_pbar:
            loss_info = trainer.train_step(detections_batch, priorities_batch)

            train_losses['total'].append(safe_float(loss_info['loss']))
            train_losses['mse'].append(safe_float(loss_info['mse']))
            train_losses['ranking'].append(safe_float(loss_info['ranking']))
            train_losses['uncertainty'].append(safe_float(loss_info['uncertainty']))

            train_pbar.set_postfix({
                'Loss': f"{loss_info['loss']:.4f}",
                'MSE': f"{loss_info['mse']:.3f}",
                'Rank': f"{loss_info['ranking']:.3f}",
                'Unc': f"{loss_info['uncertainty']:.3f}"
            })

        avg_train_loss = np.mean(train_losses['total']) if train_losses['total'] else 0.0
        avg_train_mse = np.mean(train_losses['mse']) if train_losses['mse'] else 0.0
        avg_train_rank = np.mean(train_losses['ranking']) if train_losses['ranking'] else 0.0
        avg_train_unc = np.mean(train_losses['uncertainty']) if train_losses['uncertainty'] else 0.0

        training_metrics['train_losses'].append(avg_train_loss)
        training_metrics['train_mse'].append(avg_train_mse)
        training_metrics['train_ranking'].append(avg_train_rank)
        training_metrics['train_uncertainty'].append(avg_train_unc)

        # VALIDATION PHASE
        model.eval()
        val_losses = {'total': [], 'mse': [], 'ranking': [], 'uncertainty': []}

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{n_epochs}')
            for detections_batch, priorities_batch in val_pbar:
                for detections, target_priorities in zip(detections_batch, priorities_batch):
                    if not detections or len(target_priorities) == 0:
                        continue

                    try:
                        predicted_priorities, uncertainties = model(detections)

                        if predicted_priorities.numel() == 0:
                            continue

                        min_len = min(len(predicted_priorities), len(target_priorities))
                        if min_len == 0:
                            continue

                        pred_slice = predicted_priorities[:min_len]
                        target_slice = target_priorities[:min_len].to(pred_slice.device)
                        unc_slice = uncertainties[:min_len]

                        losses = trainer.criterion(pred_slice, target_slice, unc_slice)

                        val_losses['total'].append(float(losses['total']))
                        val_losses['mse'].append(float(losses['mse']))
                        val_losses['ranking'].append(float(losses['ranking']))
                        val_losses['uncertainty'].append(float(losses['uncertainty']))

                    except Exception as e:
                        logging.debug(f"Validation error: {e}")
                        continue

                if val_losses['total']:
                    val_pbar.set_postfix({
                        'Loss': f"{np.mean(val_losses['total']):.4f}",
                        'MSE': f"{np.mean(val_losses['mse']):.3f}",
                        'Rank': f"{np.mean(val_losses['ranking']):.3f}",
                        'Unc': f"{np.mean(val_losses['uncertainty']):.3f}"
                    })

        avg_val_loss = np.mean(val_losses['total']) if val_losses['total'] else 0.0
        avg_val_mse = np.mean(val_losses['mse']) if val_losses['mse'] else 0.0
        avg_val_rank = np.mean(val_losses['ranking']) if val_losses['ranking'] else 0.0
        avg_val_unc = np.mean(val_losses['uncertainty']) if val_losses['uncertainty'] else 0.0

        training_metrics['val_losses'].append(avg_val_loss)
        training_metrics['val_mse'].append(avg_val_mse)
        training_metrics['val_ranking'].append(avg_val_rank)
        training_metrics['val_uncertainty'].append(avg_val_unc)
        training_metrics['epochs_completed'] = epoch + 1
        training_metrics['learning_rates'].append(trainer.optimizer.param_groups[0]['lr'])

        # Enhanced logging
        log_msg = (f"Epoch {epoch:3d}: "
                f"Train={avg_train_loss:.6f} (MSE:{avg_train_mse:.4f}, R:{avg_train_rank:.4f}, U:{avg_train_unc:.4f}), "
                f"Val={avg_val_loss:.6f} (MSE:{avg_val_mse:.4f}, R:{avg_val_rank:.4f}, U:{avg_val_unc:.4f})")

        logging.info(log_msg)

        # Early stopping
        if avg_val_loss < best_val_loss - 1e-6:
            improvement = best_val_loss - avg_val_loss
            best_val_loss = avg_val_loss
            training_metrics['best_epoch'] = epoch
            patience_counter = 0

            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config.__dict__ if hasattr(config, '__dict__') else {},
                'training_metrics': training_metrics,
                'model_type': 'EnhancedPriorityNet',
                'use_strain': use_strain
            }, output_dir / 'priority_net_best.pth')

            logging.info(f"💾 Best model saved (val_loss: {avg_val_loss:.6f}, improvement: {improvement:.6f})")
        else:
            patience_counter += 1

            if patience_counter >= patience:
                logging.info(f"⏹️ Early stopping (best epoch: {training_metrics['best_epoch']}, "
                           f"best val loss: {best_val_loss:.6f})")
                break

        # Learning rate scheduling
        if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
            trainer.scheduler.step()

    # Plot training curves
    plot_training_curves(training_metrics, output_dir)

    return training_metrics


def plot_training_curves(metrics: Dict, output_dir: Path):
    """Plot comprehensive training diagnostics"""

    epochs = range(1, len(metrics['train_losses']) + 1)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Enhanced PriorityNet Training Diagnostics', fontsize=16, fontweight='bold')

    # Overall Loss
    ax = axes[0, 0]
    ax.plot(epochs, metrics['train_losses'], 'b-', label='Training', linewidth=2)
    ax.plot(epochs, metrics['val_losses'], 'r-', label='Validation', linewidth=2)
    if 'best_epoch' in metrics:
        ax.axvline(x=metrics['best_epoch'] + 1, color='g', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Overall Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MSE Component
    ax = axes[0, 1]
    ax.plot(epochs, metrics['train_mse'], 'b-', label='Training', linewidth=2)
    ax.plot(epochs, metrics['val_mse'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('MSE Component')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ranking Component
    ax = axes[1, 0]
    ax.plot(epochs, metrics['train_ranking'], 'b-', label='Training', linewidth=2)
    ax.plot(epochs, metrics['val_ranking'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Ranking Loss')
    ax.set_title('Ranking Component')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Uncertainty Component
    ax = axes[1, 1]
    ax.plot(epochs, metrics['train_uncertainty'], 'b-', label='Training', linewidth=2)
    ax.plot(epochs, metrics['val_uncertainty'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Uncertainty Loss')
    ax.set_title('Uncertainty Component')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Generalization Gap
    ax = axes[2, 0]
    loss_diff = np.array(metrics['val_losses']) - np.array(metrics['train_losses'])
    ax.plot(epochs, loss_diff, 'purple', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.fill_between(epochs, 0, loss_diff, where=(loss_diff >= 0), alpha=0.3, color='red', label='Overfitting')
    ax.fill_between(epochs, 0, loss_diff, where=(loss_diff < 0), alpha=0.3, color='green', label='Underfitting')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss - Train Loss')
    ax.set_title('Generalization Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[2, 1]
    if 'learning_rates' in metrics and metrics['learning_rates']:
        ax.plot(epochs, metrics['learning_rates'], 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'LR schedule not tracked', ha='center', va='center')
        ax.set_title('Learning Rate Schedule')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = output_dir / 'training_curves_enhanced.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"📊 Training curves saved to {save_path}")


def evaluate_enhanced_model(model, dataset, split_name: str) -> Dict:
    """Evaluate Enhanced PriorityNet with uncertainty tracking"""

    logging.info(f"📊 Evaluating on {split_name} set...")

    model.eval()

    correlations = []
    precisions = []
    uncertainties_list = []
    successful = 0
    total = 0

    with torch.no_grad():
        for item in tqdm(dataset, desc=f"Evaluating {split_name}"):
            total += 1

            try:
                detections = item['detections']

                if len(detections) <= 1:
                    continue

                true_priorities = item['priorities']
                if isinstance(true_priorities, torch.Tensor):
                    true_priorities = true_priorities.detach().cpu().numpy()
                else:
                    true_priorities = np.asarray(true_priorities, dtype=np.float32)

                # Get predictions with uncertainty
                predicted_priorities, uncertainties = model(detections)
                predicted_priorities = predicted_priorities.detach().cpu().numpy()
                uncertainties = uncertainties.detach().cpu().numpy()

                m = min(len(predicted_priorities), len(true_priorities))
                if m <= 1:
                    continue

                pred_slice = predicted_priorities[:m]
                true_slice = true_priorities[:m]
                unc_slice = uncertainties[:m]

                if not (np.isfinite(pred_slice).all() and np.isfinite(true_slice).all()):
                    continue

                # Correlation
                if m <= 3:
                    tau, _ = kendalltau(true_slice, pred_slice)
                    corr = float(tau if np.isfinite(tau) else 0.0)
                else:
                    rho, _ = spearmanr(true_slice, pred_slice)
                    corr = float(rho if np.isfinite(rho) else 0.0)

                correlations.append(corr)

                # Precision@k
                k = min(3, m)
                true_top_k = set(np.argsort(true_slice)[::-1][:k])
                pred_top_k = set(np.argsort(pred_slice)[::-1][:k])
                precision = len(true_top_k & pred_top_k) / k
                precisions.append(float(precision))

                # Track uncertainty
                uncertainties_list.append(float(np.mean(unc_slice)))

                successful += 1

            except Exception as e:
                logging.debug(f"Evaluation error: {e}")
                continue

    results = {
        'split': split_name,
        'n_samples': successful,
        'total_attempts': total,
        'success_rate': successful / max(total, 1)
    }

    if correlations:
        results.update({
            'avg_ranking_correlation': float(np.mean(correlations)),
            'std_ranking_correlation': float(np.std(correlations)),
            'median_ranking_correlation': float(np.median(correlations)),
            'avg_top_k_precision': float(np.mean(precisions)),
            'avg_uncertainty': float(np.mean(uncertainties_list)),
            'std_uncertainty': float(np.std(uncertainties_list))
        })

    logging.info(f"📈 {split_name.upper()}: {successful}/{total} successful")
    if correlations:
        logging.info(f"   Correlation: {results['avg_ranking_correlation']:.3f} ± {results['std_ranking_correlation']:.3f}")
        logging.info(f"   Precision@3: {results['avg_top_k_precision']:.3f}")
        logging.info(f"   Avg Uncertainty: {results['avg_uncertainty']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Enhanced PriorityNet Training')
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--dataset_path', required=True, help='Path to newDataset')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples per split')
    parser.add_argument('--create_overlaps', action='store_true', help='Create artificial overlaps')
    parser.add_argument('--use_strain', action='store_true', help='Enable strain encoding')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    setup_logging(args.verbose)
    logging.info("🚀 Starting Enhanced Phase 2 Training")

    # Validate paths
    if not Path(args.dataset_path).exists():
        logging.error(f"❌ Dataset not found: {args.dataset_path}")
        return

    # Load config
    try:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)

        priority_config = config_dict.get('priority_net', {})

        defaults = {
            'hidden_dims': [256, 128, 64, 32],
            'dropout': 0.1,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 32,
            'epochs': 500,
            'patience': 30,
            'use_strain': args.use_strain,
            'ranking_weight': 0.5,
            'mse_weight': 0.4,
            'uncertainty_weight': 0.1
        }

        final_config = {}
        for key, default_value in defaults.items():
            config_value = priority_config.get(key, default_value)
            if isinstance(default_value, (int, float)):
                try:
                    final_config[key] = type(default_value)(float(config_value))
                except:
                    final_config[key] = default_value
            else:
                final_config[key] = config_value

        logging.info("📋 Configuration:")
        for key, value in final_config.items():
            logging.info(f"   {key}: {value}")

        config = type('Config', (), final_config)()

    except Exception as e:
        logging.error(f"Config error: {e}, using defaults")
        config = type('Config', (), {
            'hidden_dims': [256, 128, 64, 32],
            'dropout': 0.1,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 32,
            'epochs': 500,
            'patience': 30,
            'use_strain': False
        })()

    # Load datasets
    logging.info("📊 Loading datasets...")

    train_loader = ChunkedGWDataLoader(args.dataset_path, 'train', args.max_samples)
    val_loader = ChunkedGWDataLoader(args.dataset_path, 'validation', args.max_samples)
    test_loader = ChunkedGWDataLoader(args.dataset_path, 'test', args.max_samples)

    # Convert to scenarios
    logging.info("🔄 Creating scenarios...")

    train_scenarios = train_loader.convert_to_priority_scenarios(create_overlaps=args.create_overlaps)
    val_scenarios = val_loader.convert_to_priority_scenarios(create_overlaps=False)
    test_scenarios = test_loader.convert_to_priority_scenarios(create_overlaps=False)

    # Create datasets
    train_dataset = PriorityNetDataset(train_scenarios, "train")
    val_dataset = PriorityNetDataset(val_scenarios, "validation")
    test_dataset = PriorityNetDataset(test_scenarios, "test")

    if len(train_dataset) == 0:
        logging.error("❌ No training data")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    training_metrics = train_enhanced_priority_net(config, train_dataset, val_dataset, output_dir)

    # Load best model
    best_checkpoint = torch.load(output_dir / 'priority_net_best.pth', weights_only=False, map_location='cpu')
    model = EnhancedPriorityNet(config, use_strain=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    # Evaluate
    train_results = evaluate_enhanced_model(model, train_dataset, "train")
    val_results = evaluate_enhanced_model(model, val_dataset, "validation")
    test_results = evaluate_enhanced_model(model, test_dataset, "test")

    # Save results
    final_results = {
        'training_metrics': training_metrics,
        'evaluation': {
            'train': train_results,
            'validation': val_results,
            'test': test_results
        },
        'config': config.__dict__ if hasattr(config, '__dict__') else {}
    }

    with open(output_dir / 'complete_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)

    # Summary
    print("\n" + "="*80)
    print("✅ ENHANCED PHASE 2 COMPLETE")
    print("="*80)
    print(f"🎯 TEST Correlation: {test_results.get('avg_ranking_correlation', 0):.1%}")
    print(f"📊 VAL Correlation: {val_results.get('avg_ranking_correlation', 0):.1%}")
    print(f"📁 Results: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
