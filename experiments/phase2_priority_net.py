
#!/usr/bin/env python3
"""
Phase 2: Train PriorityNet for intelligent signal ranking
"""


import os
from xml.parsers.expat import model
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
from ahsd.core.priority_net import PriorityNet

class CombinedPriorityLoss(nn.Module):
    """Enhanced loss function emphasizing ranking with diversity regularization"""
    
    def __init__(self, mse_weight=0.1, ranking_weight=0.9, diversity_weight=0.1, margin=0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.ranking_weight = ranking_weight
        self.diversity_weight = diversity_weight
        self.margin = margin
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        mse_loss = self.mse_loss(predictions, targets)
        ranking_loss = self.compute_ranking_loss(predictions, targets)
        diversity_loss = self.compute_diversity_loss(predictions, targets)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.ranking_weight * ranking_loss + 
                     self.diversity_weight * diversity_loss)
        
        return total_loss
    
    def compute_ranking_loss(self, predictions, targets):
        """Enhanced ranking loss with larger margin"""
        if len(predictions) < 2:
            return torch.tensor(0.0, requires_grad=True)
        
        ranking_loss = 0.0
        pair_count = 0
        
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                target_diff = targets[i] - targets[j]
                pred_diff = predictions[i] - predictions[j]
                
                # Use larger margin to force bigger prediction differences
                if target_diff > self.margin:
                    loss = torch.clamp(self.margin - pred_diff, min=0.0)
                elif target_diff < -self.margin:
                    loss = torch.clamp(self.margin + pred_diff, min=0.0)
                else:
                    # Small penalty for similar priorities to encourage discrimination
                    loss = 0.1 * torch.abs(pred_diff)
                
                ranking_loss += loss
                pair_count += 1
        
        return ranking_loss / max(pair_count, 1)
    
    def compute_diversity_loss(self, predictions, targets):
        """Encourage prediction variance to match target variance"""
        if len(predictions) < 2:
            return torch.tensor(0.0, requires_grad=True)
        
        pred_var = torch.var(predictions)
        target_var = torch.var(targets)
        
        # Penalize when prediction variance is much lower than target variance
        diversity_loss = torch.clamp(target_var - pred_var, min=0.0)
        
        return diversity_loss



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
        """CORRECTED: Higher bias = Higher priority"""
        
        priorities = []
        all_biases = []
        
        # Collect bias magnitudes for normalization
        for i, true_param in enumerate(true_params):
            if i < len(biases) and biases[i]:
                bias_values = list(biases[i].values())
                if bias_values:
                    avg_bias = np.mean([abs(b) for b in bias_values])
                    all_biases.append(avg_bias)
        
        # Compute bias percentiles
        if all_biases:
            bias_90 = np.percentile(all_biases, 90)
        else:
            bias_90 = 2.0
        
        # Compute corrected priorities
        for i, true_param in enumerate(true_params):
            if i < len(biases) and biases[i]:
                bias_values = list(biases[i].values())
                if bias_values:
                    avg_bias = np.mean([abs(b) for b in bias_values])
                    # CORRECTED: Higher bias → Higher priority
                    normalized_bias = np.clip(avg_bias / bias_90, 0, 1)
                    bias_priority = 0.3 + 0.7 * normalized_bias
                else:
                    bias_priority = 0.5
            else:
                bias_priority = 1.0  # Signal not recovered gets highest priority
            
            # SNR penalty (easier signals get lower priority)
            snr = true_param.get('network_snr', 10.0)
            snr_penalty = max(0.1, 1.0 - (snr - 8.0) / 15.0)
            
            combined_priority = 0.8 * bias_priority + 0.2 * snr_penalty
            scaled_priority = (combined_priority - 0.5) * 4  # Scale to roughly [-1, 1]
            priorities.append(scaled_priority)
        
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

        return self.validate_and_clean_detections(detections)

    def _add_detection_noise(self, params: Dict) -> Dict:
        """ Realistic parameter uncertainties based on Fisher matrix scaling"""
        
        noisy_params = params.copy()
        snr = self._estimate_snr(params, {})
        
        # SNR-dependent uncertainties (better SNR = lower uncertainty)
        snr_factor = max(0.1, 8.0 / max(snr, 8.0))
        
        # Mass uncertainties (Fisher matrix scaling: 5-15% depending on SNR)
        for mass_param in ['mass_1', 'mass_2']:
            if mass_param in params:
                true_val = params[mass_param]
                uncertainty = true_val * (0.05 + 0.10 * snr_factor)
                noisy_params[mass_param] = max(1.0, np.random.normal(true_val, uncertainty))
        
        # Distance uncertainty (scales as SNR^-1: 15-50%)
        if 'luminosity_distance' in params:
            true_val = params['luminosity_distance']
            uncertainty = true_val * (0.15 + 0.35 * snr_factor)
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
        """ Realistic SNR estimation with detector modeling"""
        
        # Physical parameters
        m1, m2 = signal_params.get('mass_1', 30), signal_params.get('mass_2', 30)
        distance = signal_params.get('luminosity_distance', 500)
        
        # Chirp mass scaling
        chirp_mass = self._compute_chirp_mass(m1, m2)
        mass_factor = (chirp_mass / 30.0)**(5.0/6.0)
        
        # Distance and inclination effects
        distance_factor = 400.0 / distance
        inclination = signal_params.get('theta_jn', np.pi/2)
        inclination_factor = np.sqrt((1 + 6*np.cos(inclination)**2 + np.cos(inclination)**4) / 8)
        
        # Combined SNR with realistic base value
        base_snr = 12.0
        snr = base_snr * mass_factor * distance_factor * inclination_factor
        
        return max(6.0, min(50.0, snr + np.random.normal(0, 1.0)))


    def _compute_chirp_mass(self, m1: float, m2: float) -> float:
        """Compute chirp mass."""
        return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    
    def validate_and_clean_detections(self, detections: List[Dict]) -> List[Dict]:
        """Validate and clean detection data"""
        cleaned_detections = []
        
        for detection in detections:
            # Check required parameters
            required_params = ['mass_1', 'mass_2', 'luminosity_distance']
            if not all(param in detection for param in required_params):
                continue
            
            # Validate physical constraints
            if not self.validate_physical_parameters(detection):
                continue
            
            # Check for NaN/inf values
            has_invalid = any(
                isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value))
                for value in detection.values()
            )
            if has_invalid:
                continue
            
            cleaned_detections.append(self.normalize_detection_parameters(detection))
        
        return cleaned_detections

    def validate_physical_parameters(self, params: Dict) -> bool:
        """Validate parameter constraints"""
        m1, m2 = params.get('mass_1', 0), params.get('mass_2', 0)
        if m1 <= 0 or m2 <= 0 or (m1 + m2) > 200 or (m1 + m2) < 2:
            return False
        
        distance = params.get('luminosity_distance', 0)
        if distance <= 0 or distance > 10000:
            return False
        
        return True

    def normalize_detection_parameters(self, detection: Dict) -> Dict:
        """Normalize parameters to reasonable ranges"""
        normalized = detection.copy()
        normalized['mass_1'] = np.clip(detection['mass_1'], 1.0, 100.0)
        normalized['mass_2'] = np.clip(detection['mass_2'], 1.0, 100.0)
        
        # Ensure m1 >= m2
        if normalized['mass_1'] < normalized['mass_2']:
            normalized['mass_1'], normalized['mass_2'] = normalized['mass_2'], normalized['mass_1']
        
        normalized['luminosity_distance'] = np.clip(detection['luminosity_distance'], 10.0, 5000.0)
        return normalized


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Custom collate function for variable-size detection lists."""
    return batch

def train_priority_net(config: AHSDConfig, train_dataset: PriorityNetDataset,
                      val_dataset: PriorityNetDataset, args) -> PriorityNet:
    """
    Train the PriorityNet model with corrected bias logic and improved loss function.
    Uses PriorityNet model from priority_net.py but ignores PriorityNetTrainer.
    """

    logging.info("Initializing PriorityNet training...")

    #   Initialize model (from your priority_net.py)
    model = PriorityNet(config.priority_net)
    
    #   Use our improved combined loss function
    criterion = CombinedPriorityLoss(
    mse_weight=0.1,       # Keep MSE low since it's just a guide
    ranking_weight=0.7,   # Ranking is still the main driver
    diversity_weight=0.2  # Boosted from 0.1 → 0.2 for more diversity
)

   
    #   Initialize optimizer with regularization
    optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=5e-4,  # Match config learning_rate: 0.0005
    weight_decay=1e-4
)

    
    #   Initialize cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=60, 
        eta_min=1e-5
    )

    #   Create data loaders with optimal batch size
    batch_size = min(16, getattr(config.priority_net, 'batch_size', 32))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )

    #   Training parameters
    n_epochs = 60
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 8
    
    logging.info(f"Training setup: {n_epochs} epochs, batch_size={batch_size}, patience={patience}")

    #   Training loop
    for epoch in range(n_epochs):
        # Training phase
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

                    # Forward pass using PriorityNet model
                    predictions = model(detections)

                    # Ensure same length
                    min_len = min(len(predictions), len(targets))
                    predictions = predictions[:min_len]
                    targets = targets[:min_len]

                    #   Use our CombinedPriorityLoss (not trainer's MSELoss)
                    loss = criterion(predictions, targets)

                    # Backward pass with our optimizer
                    optimizer.zero_grad()
                    loss.backward()
                    
                    #   Gradient clipping (reduced from 1.0 to 0.5)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    
                    optimizer.step()

                    batch_loss += loss.item()
                    valid_samples += 1

                except Exception as e:
                    logging.debug(f"Skipping batch sample: {e}")
                    continue

            if valid_samples > 0:
                train_losses.append(batch_loss / valid_samples)

        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')

        # Validation phase
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

                        #   Use our criterion for validation too
                        loss = criterion(predictions, targets)
                        val_losses.append(loss.item())

                    except Exception as e:
                        continue

        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')

        logging.info(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')

        #   Realistic convergence check (removed false "perfect convergence")
        if avg_val_loss < 0.02 and abs(avg_train_loss - avg_val_loss) < 0.01:
            if epoch > 25:  # Ensure minimum training
                logging.info("  GOOD CONVERGENCE ACHIEVED!")
                logging.info(f"📊 Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                logging.info(f"🎯 Training complete at epoch {epoch+1} - good generalization!")
                torch.save(model.state_dict(), 'converged_priority_net.pth')
                break

        #   Use our cosine annealing scheduler
        scheduler.step()

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_priority_net.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break

        #   Log to wandb with correct learning rate access
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            })

    # Load best model
    if os.path.exists('best_priority_net.pth'):
        model.load_state_dict(torch.load('best_priority_net.pth', weights_only=True))
        logging.info("Loaded best model from checkpoint")
    else:
        logging.warning("Best model file not found, using current model")

    return model

def evaluate_priority_net(model: PriorityNet, test_dataset: PriorityNetDataset) -> Dict:
    """Evaluate PriorityNet performance with debugging."""

    logging.info("Evaluating PriorityNet...")

    model.eval()
    metrics = {
        'ranking_correlations': [],
        'top_k_precisions': [],
        'priority_accuracy': []
    }

    debug_count = 0
    with torch.no_grad():
        for sample in tqdm(test_dataset.data, desc="Evaluation"):
            try:
                detections = sample['detections']
                true_priorities = sample['target_priorities']

                if len(detections) == 0:
                    continue

                # Get model predictions
                predicted_priorities = model(detections)
                predicted_ranking = model.rank_detections(detections)
                true_ranking = torch.argsort(true_priorities, descending=True).tolist()

                # DEBUG: Print first few samples to understand the issue
                if debug_count < 5:
                    print(f"\n=== DEBUG SAMPLE {debug_count + 1} ===")
                    print(f"Number of detections: {len(detections)}")
                    print(f"True priorities: {true_priorities.tolist()}")
                    print(f"Predicted priorities: {predicted_priorities.tolist()}")
                    print(f"True ranking: {true_ranking}")
                    print(f"Predicted ranking: {predicted_ranking}")
                    
                    # Check variance
                    true_var = torch.var(true_priorities).item()
                    pred_var = torch.var(predicted_priorities).item()
                    print(f"True priorities variance: {true_var:.6f}")
                    print(f"Predicted priorities variance: {pred_var:.6f}")

                # Ranking correlation (Spearman)
                from scipy.stats import spearmanr
                if len(predicted_ranking) > 1 and len(true_ranking) > 1:
                    correlation, p_value = spearmanr(predicted_ranking, true_ranking)
                    
                    # DEBUG: Print correlation details
                    if debug_count < 5:
                        print(f"Spearman correlation: {correlation:.6f} (p={p_value:.6f})")
                    
                    if not np.isnan(correlation):
                        metrics['ranking_correlations'].append(correlation)
                    else:
                        if debug_count < 5:
                            print("WARNING: NaN correlation!")

                # Top-k precision
                k = min(2, len(detections))
                if k > 0:
                    top_k_predicted = set(predicted_ranking[:k])
                    top_k_true = set(true_ranking[:k])
                    precision = len(top_k_predicted.intersection(top_k_true)) / k
                    metrics['top_k_precisions'].append(precision)
                    
                    if debug_count < 5:
                        print(f"Top-{k} precision: {precision:.6f}")

                # Priority prediction accuracy
                priority_mae = torch.mean(torch.abs(predicted_priorities - true_priorities)).item()
                metrics['priority_accuracy'].append(1.0 / (1.0 + priority_mae))
                
                if debug_count < 5:
                    print(f"Priority MAE: {priority_mae:.6f}")
                
            
                if debug_count < 3:  # For first few samples
                    print(f"\n=== FEATURE ANALYSIS SAMPLE {debug_count + 1} ===")
                    for i, detection in enumerate(detections):
                        print(f"Detection {i} features:")
                        feature_values = []
                        for key, value in detection.items():
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value:.6f}")
                                feature_values.append(value)
                        
                        # Check feature variance within this detection
                        if len(feature_values) > 1:
                            feature_variance = np.var(feature_values)
                            print(f"  Feature variance: {feature_variance:.6f}")
                    
                    # Check how different the detections are from each other
                    if len(detections) > 1:
                        print(f"Number of detections in scenario: {len(detections)}")
                        print("Are detections sufficiently different?")
                        all_features = np.array([[v for k, v in det.items() if isinstance(v, (int, float))] for det in detections])
                        feature_diffs = np.std(all_features, axis=0)
                        print(f"Feature standard deviations across detections: {feature_diffs}")
                        

                debug_count += 1

            except Exception as e:
                logging.debug(f"Evaluation error: {e}")
                if debug_count < 5:
                    print(f"ERROR in sample {debug_count}: {e}")
                continue

    # Compute summary statistics
    summary_metrics = {}
    for metric_name, values in metrics.items():
        if values:
            summary_metrics[f'avg_{metric_name[:-1]}'] = np.mean(values)
            summary_metrics[f'std_{metric_name[:-1]}'] = np.std(values)
            
            # DEBUG: Print distribution info
            print(f"\n{metric_name} distribution:")
            print(f"  Mean: {np.mean(values):.6f}")
            print(f"  Std: {np.std(values):.6f}")
            print(f"  Min: {np.min(values):.6f}")
            print(f"  Max: {np.max(values):.6f}")
            print(f"  Count: {len(values)}")
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

    logging.info(f"Original lengths: scenarios={len(scenarios)}, baseline_results={len(baseline_results)}")

    # Align scenarios with baseline_results by scenario_id
    baseline_ids = {b['scenario_id'] for b in baseline_results}
    aligned_scenarios = [s for s in scenarios if s['scenario_id'] in baseline_ids]

    # Reorder aligned_scenarios to match baseline_results order
    baseline_dict = {b['scenario_id']: b for b in baseline_results}
    aligned_scenarios_sorted = [s for s in scenarios if s['scenario_id'] in baseline_dict]
    aligned_baseline_sorted = [baseline_dict[s['scenario_id']] for s in aligned_scenarios_sorted]

    logging.info(f"After alignment: scenarios={len(aligned_scenarios_sorted)}, baseline_results={len(aligned_baseline_sorted)}")

    # Split data
    n_total = len(aligned_scenarios_sorted)
    n_val = int(args.val_split * n_total)
    n_test = n_val
    n_train = n_total - n_val - n_test

    # Shuffle
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]

    # Create datasets
    train_scenarios = [aligned_scenarios_sorted[i] for i in train_indices]
    train_baseline = [aligned_baseline_sorted[i] for i in train_indices]
    train_dataset = PriorityNetDataset(train_scenarios, train_baseline)

    val_scenarios = [aligned_scenarios_sorted[i] for i in val_indices]
    val_baseline = [aligned_baseline_sorted[i] for i in val_indices]
    val_dataset = PriorityNetDataset(val_scenarios, val_baseline)

    test_scenarios = [aligned_scenarios_sorted[i] for i in test_indices]
    test_baseline = [aligned_baseline_sorted[i] for i in test_indices]
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
