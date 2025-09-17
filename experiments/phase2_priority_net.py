import os
#!/usr/bin/env python3
"""
Phase 2: Train PriorityNet for intelligent signal ranking
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
import pickle
from pathlib import Path
import logging
import wandb
from tqdm import tqdm
import yaml
from typing import List, Dict

from ahsd.utils.config import AHSDConfig
from ahsd.core.priority_net import PriorityNet, PriorityNetTrainer

class PriorityNetDataset(Dataset):
    """Dataset for training PriorityNet with real and simulated data."""
    
    def __init__(self, scenarios: List[Dict], baseline_results: List[Dict]):
        self.data = []
        self.logger = logging.getLogger(__name__)
        
        # Process training examples
        valid_pairs = 0
        for scenario, baseline in zip(scenarios, baseline_results):
            try:
                # Create ranking targets based on bias outcomes
                target_priorities = self._compute_optimal_priorities(
                    scenario['true_parameters'],
                    baseline.get('baseline_biases', [])
                )
                
                # Prepare detection features
                detections = self._prepare_detections(scenario)
                
                if len(detections) > 0 and len(target_priorities) > 0:
                    self.data.append({
                        'detections': detections,
                        'target_priorities': target_priorities,
                        'scenario_id': scenario['scenario_id'],
                        'n_signals': scenario['n_signals'],
                        'data_type': scenario.get('data_type', 'simulated')
                    })
                    valid_pairs += 1
                    
            except Exception as e:
                self.logger.debug(f"Skipping scenario {scenario.get('scenario_id', 'unknown')}: {e}")
                continue
        
        self.logger.info(f"Created dataset with {valid_pairs} valid training examples")
    
    def _compute_optimal_priorities(self, true_params: List[Dict], biases: List[Dict]) -> torch.Tensor:
        """Compute optimal extraction priorities based on bias outcomes."""
        
        priorities = []
        
        for i, true_param in enumerate(true_params):
            if i < len(biases) and biases[i]:
                # Lower bias → higher priority
                bias_values = list(biases[i].values())
                if bias_values:
                    avg_bias = np.mean([abs(b) for b in bias_values])
                    priority = 1.0 / (1.0 + avg_bias)
                else:
                    priority = 0.5  # Default moderate priority
            else:
                # Signal not recovered or no bias data
                priority = 0.1
            
            # Add SNR-based component
            snr_component = min(true_param.get('network_snr', 10.0) / 20.0, 1.0)
            combined_priority = 0.7 * priority + 0.3 * snr_component
            
            priorities.append(combined_priority)
        
        return torch.tensor(priorities, dtype=torch.float32)
    
    def _prepare_detections(self, scenario: Dict) -> List[Dict]:
        """Prepare detection candidates with realistic features."""
        
        detections = []
        
        for i, signal_params in enumerate(scenario['true_parameters']):
            # Start with true parameters
            detection = signal_params.copy()
            
            # Add measurement uncertainties
            detection = self._add_detection_noise(detection)
            
            # Add SNR estimate
            detection['network_snr'] = self._estimate_snr(signal_params, scenario)
            detection['coherent_snr'] = detection['network_snr'] * np.random.uniform(0.8, 1.0)
            detection['null_snr'] = np.random.normal(1.0, 0.3)
            
            # Add derived quantities
            detection['chirp_mass_source'] = self._compute_chirp_mass(
                detection['mass_1'], detection['mass_2']
            )
            detection['total_mass_source'] = detection['mass_1'] + detection['mass_2']
            
            # Add sky localization uncertainty
            detection['sky_area_90'] = np.random.exponential(50)  # deg^2
            
            # Add detection metadata
            detection['detection_id'] = i
            
            detections.append(detection)
        
        return detections
    
    def _add_detection_noise(self, params: Dict) -> Dict:
        """Add realistic measurement uncertainties to parameters."""
        
        noisy_params = params.copy()
        
        # Mass uncertainties (10-20%)
        for mass_param in ['mass_1', 'mass_2']:
            if mass_param in params:
                true_val = params[mass_param]
                uncertainty = true_val * np.random.uniform(0.1, 0.2)
                noisy_params[mass_param] = max(1.0, np.random.normal(true_val, uncertainty))
        
        # Distance uncertainty (20-50%)
        if 'luminosity_distance' in params:
            true_val = params['luminosity_distance']
            uncertainty = true_val * np.random.uniform(0.2, 0.5)
            noisy_params['luminosity_distance'] = max(10.0, np.random.normal(true_val, uncertainty))
        
        # Angular uncertainties
        for angle_param in ['ra', 'dec', 'theta_jn', 'psi', 'phase']:
            if angle_param in params:
                true_val = params[angle_param]
                uncertainty = np.random.uniform(0.1, 0.5)
                noisy_params[angle_param] = np.random.normal(true_val, uncertainty)
        
        # Time uncertainty (1-10 ms)
        if 'geocent_time' in params:
            true_val = params['geocent_time']
            uncertainty = np.random.uniform(0.001, 0.01)
            noisy_params['geocent_time'] = np.random.normal(true_val, uncertainty)
        
        # Spin uncertainties
        for spin_param in ['a_1', 'a_2']:
            if spin_param in params:
                true_val = params[spin_param]
                uncertainty = 0.2
                noisy_params[spin_param] = np.clip(np.random.normal(true_val, uncertainty), 0, 0.99)
        
        return noisy_params
    
    def _estimate_snr(self, signal_params: Dict, scenario: Dict) -> float:
        """Estimate network SNR for signal."""
        
        # Use target SNR if available
        if 'target_snrs' in scenario:
            signal_id = signal_params.get('signal_id', 0)
            if signal_id < len(scenario['target_snrs']):
                return scenario['target_snrs'][signal_id] + np.random.normal(0, 1)
        
        # Estimate based on masses and distance
        m1 = signal_params.get('mass_1', 30)
        m2 = signal_params.get('mass_2', 30)
        distance = signal_params.get('luminosity_distance', 500)
        
        # Rough SNR scaling
        chirp_mass = self._compute_chirp_mass(m1, m2)
        snr = 1000 * (chirp_mass / 30)**(5/6) / (distance / 400)
        
        # Add noise and ensure reasonable range
        snr = snr + np.random.normal(0, 2)
        return max(8.0, min(50.0, snr))
    
    def _compute_chirp_mass(self, m1: float, m2: float) -> float:
        """Compute chirp mass."""
        return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Custom collate function for variable-size detection lists."""
    return batch

def train_priority_net(config: AHSDConfig, train_dataset: PriorityNetDataset, 
                      val_dataset: PriorityNetDataset, args) -> PriorityNet:
    """Train the PriorityNet model."""
    
    logging.info("Initializing PriorityNet training...")
    
    # Initialize model and trainer
    model = PriorityNet(config.priority_net)
    trainer = PriorityNetTrainer(model, config.priority_net)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.priority_net.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.priority_net.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Training loop
    n_epochs = 200
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{n_epochs}'):
            batch_loss = 0
            valid_samples = 0
            
            for sample in batch:
                try:
                    detections = sample['detections']
                    targets = sample['target_priorities']
                    
                    if len(detections) == 0 or len(targets) == 0:
                        continue
                    
                    # Forward pass
                    predictions = model(detections)
                    
                    # Ensure same length
                    min_len = min(len(predictions), len(targets))
                    predictions = predictions[:min_len]
                    targets = targets[:min_len]
                    
                    loss = trainer.criterion(predictions, targets)
                    
                    # Backward pass
                    trainer.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    trainer.optimizer.step()
                    
                    batch_loss += loss.item()
                    valid_samples += 1
                    
                except Exception as e:
                    logging.debug(f"Skipping batch sample: {e}")
                    continue
            
            if valid_samples > 0:
                train_losses.append(batch_loss / valid_samples)
        
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{n_epochs}'):
                for sample in batch:
                    try:
                        detections = sample['detections']
                        targets = sample['target_priorities']
                        
                        if len(detections) == 0 or len(targets) == 0:
                            continue
                        
                        predictions = model(detections)
                        
                        min_len = min(len(predictions), len(targets))
                        predictions = predictions[:min_len]
                        targets = targets[:min_len]
                        
                        loss = trainer.criterion(predictions, targets)
                        val_losses.append(loss.item())
                        
                    except Exception as e:
                        continue
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        logging.info(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
        
        # Learning rate scheduling
        trainer.scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_priority_net.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience or (avg_val_loss == float("inf") and epoch > 5):
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': trainer.optimizer.param_groups[0]['lr']
            })
    
    # Load best model
    if os.path.exists('best_priority_net.pth'):
        model.load_state_dict(torch.load('best_priority_net.pth', weights_only=True))
    else:
        logging.warning("Best model file not found, using current model")
    
    return model

def evaluate_priority_net(model: PriorityNet, test_dataset: PriorityNetDataset) -> Dict:
    """Evaluate PriorityNet performance."""
    
    logging.info("Evaluating PriorityNet...")
    
    model.eval()
    metrics = {
        'ranking_correlations': [],
        'top_k_precisions': [],
        'priority_accuracy': []
    }
    
    with torch.no_grad():
        for sample in tqdm(test_dataset.data, desc="Evaluation"):
            try:
                detections = sample['detections']
                true_priorities = sample['target_priorities']
                
                if len(detections) == 0:
                    continue
                
                # Get model predictions
                predicted_ranking = model.rank_detections(detections)
                true_ranking = torch.argsort(true_priorities, descending=True).tolist()
                
                # Ranking correlation (Spearman)
                from scipy.stats import spearmanr
                if len(predicted_ranking) > 1 and len(true_ranking) > 1:
                    correlation, _ = spearmanr(predicted_ranking, true_ranking)
                    if not np.isnan(correlation):
                        metrics['ranking_correlations'].append(correlation)
                
                # Top-k precision
                k = min(2, len(detections))
                if k > 0:
                    top_k_predicted = set(predicted_ranking[:k])
                    top_k_true = set(true_ranking[:k])
                    precision = len(top_k_predicted.intersection(top_k_true)) / k
                    metrics['top_k_precisions'].append(precision)
                
                # Priority prediction accuracy
                predicted_priorities = model(detections)
                priority_mae = torch.mean(torch.abs(predicted_priorities - true_priorities)).item()
                metrics['priority_accuracy'].append(1.0 / (1.0 + priority_mae))
                
            except Exception as e:
                logging.debug(f"Evaluation error: {e}")
                continue
    
    # Compute summary statistics
    summary_metrics = {}
    for metric_name, values in metrics.items():
        if values:
            summary_metrics[f'avg_{metric_name[:-1]}'] = np.mean(values)
            summary_metrics[f'std_{metric_name[:-1]}'] = np.std(values)
        else:
            summary_metrics[f'avg_{metric_name[:-1]}'] = 0.0
            summary_metrics[f'std_{metric_name[:-1]}'] = 0.0
    
    return summary_metrics

def main():
    parser = argparse.ArgumentParser(description='Train PriorityNet')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--data_dir', required=True, help='Training data directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split fraction')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project="ahsd-priority-net", config=vars(args))
    
    # Load configuration
    config = AHSDConfig.from_yaml(args.config)
    
    # Load training data
    data_dir = Path(args.data_dir)
    
    with open(data_dir / 'training_scenarios.pkl', 'rb') as f:
        scenarios = pickle.load(f)
    
    with open(data_dir / 'baseline_results.pkl', 'rb') as f:
        baseline_results = pickle.load(f)
    
    logging.info(f"Loaded {len(scenarios)} scenarios and {len(baseline_results)} baseline results")
    
    # Split data
    n_total = len(scenarios)
    n_val = int(args.val_split * n_total)
    n_test = n_val
    n_train = n_total - n_val - n_test
    
    # Shuffle data
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    # Create datasets
    train_scenarios = [scenarios[i] for i in train_indices]
    train_baseline = [baseline_results[i] for i in train_indices]
    train_dataset = PriorityNetDataset(train_scenarios, train_baseline)
    
    val_scenarios = [scenarios[i] for i in val_indices]
    val_baseline = [baseline_results[i] for i in val_indices]
    val_dataset = PriorityNetDataset(val_scenarios, val_baseline)
    
    test_scenarios = [scenarios[i] for i in test_indices]
    test_baseline = [baseline_results[i] for i in test_indices]
    test_dataset = PriorityNetDataset(test_scenarios, test_baseline)
    
    logging.info(f"Split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Train model
    trained_model = train_priority_net(config, train_dataset, val_dataset, args)
    
    # Evaluate model
    evaluation_metrics = evaluate_priority_net(trained_model, test_dataset)
    logging.info(f"Evaluation metrics: {evaluation_metrics}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(trained_model.state_dict(), output_dir / 'priority_net.pth')
    
    with open(output_dir / 'priority_net_evaluation.pkl', 'wb') as f:
        pickle.dump(evaluation_metrics, f)
    
    # Save model config
    model_config = {
        'config': config,
        'evaluation_metrics': evaluation_metrics,
        'training_args': vars(args)
    }
    
    with open(output_dir / 'priority_net_config.yaml', 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)
    
    logging.info(f"PriorityNet training completed! Model saved to {output_dir}")

if __name__ == '__main__':
    main()
