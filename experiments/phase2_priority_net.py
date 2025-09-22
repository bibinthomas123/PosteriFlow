#!/usr/bin/env python3
"""
PRODUCTION Phase 2: Train PriorityNet for intelligent signal extraction ordering
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
from typing import List, Dict, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from ahsd.core
from ahsd.core.priority_net import PriorityNet, PriorityNetTrainer
print('PriorityNetTrainer loaded from:', sys.modules['ahsd.core.priority_net'].__file__)

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('phase2_production_priority_net.log'),
            logging.StreamHandler()
        ]
    )

class PriorityNetDataset(Dataset):
    """Production dataset for PriorityNet training"""
    
    def __init__(self, scenarios: List[Dict]):
        self.data = []
        self.logger = logging.getLogger(__name__)
        
        for scenario_id, scenario in enumerate(scenarios):
            try:
                true_params = scenario.get('true_parameters', [])
                baseline_results = scenario.get('baseline_biases', [])
                
                if not true_params:
                    continue
                
                # Compute physics-based priorities
                priorities = self._compute_extraction_priorities(true_params, baseline_results)
                
                if priorities is not None and len(priorities) > 0:
                    self.data.append({
                        'scenario_id': scenario_id,
                        'detections': true_params,
                        'priorities': priorities
                    })
                    
            except Exception as e:
                self.logger.debug(f"Error processing scenario {scenario_id}: {e}")
                continue
        
        self.logger.info(f"✅ Created production dataset with {len(self.data)} scenarios")
    
    def _compute_extraction_priorities(self, signals: List[Dict], 
                                     baseline_biases: List[Dict] = None) -> torch.Tensor:
        """Compute extraction priorities based on signal characteristics"""
        
        n_signals = len(signals)
        priorities = torch.zeros(n_signals)
        
        for i, signal in enumerate(signals):
            # Base priority from signal characteristics
            snr = signal.get('network_snr', 10.0)
            m1 = signal.get('mass_1', 30.0)
            m2 = signal.get('mass_2', 25.0)
            distance = signal.get('luminosity_distance', 500.0)
            
            # SNR component (normalized)
            snr_priority = min(snr / 25.0, 1.0)
            
            # Mass-based difficulty
            total_mass = m1 + m2
            if 25 <= total_mass <= 75:
                mass_priority = 1.0  # Optimal range
            elif 15 <= total_mass <= 100:
                mass_priority = 0.7  # Moderate
            else:
                mass_priority = 0.4  # Difficult
            
            # Distance-based difficulty
            distance_priority = max(0.2, min(1.0, 800.0 / distance))
            
            # Bias information if available
            bias_penalty = 0.0
            if baseline_biases and i < len(baseline_biases) and baseline_biases[i]:
                try:
                    bias_magnitude = np.mean([abs(b) for b in baseline_biases[i].values() if isinstance(b, (int, float))])
                    bias_penalty = min(0.3, bias_magnitude * 0.5)
                except:
                    bias_penalty = 0.0
            
            # Combined priority
            base_priority = (0.4 * snr_priority + 
                           0.3 * mass_priority + 
                           0.2 * distance_priority + 
                           0.1 * bias_penalty)
            
            # Hierarchical penalty (later extractions are harder)
            hierarchy_penalty = i * 0.05
            
            final_priority = max(0.1, base_priority - hierarchy_penalty)
            priorities[i] = final_priority
        
        return priorities
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_priority_batch(batch: List[Dict]) -> Tuple[List[List[Dict]], List[torch.Tensor]]:
    """Collate function for variable-length sequences"""
    
    detections_batch = []
    priorities_batch = []
    
    for item in batch:
        detections_batch.append(item['detections'])
        priorities_batch.append(item['priorities'])
    
    return detections_batch, priorities_batch

def train_priority_net(config, dataset: PriorityNetDataset, output_dir: Path) -> Dict[str, Any]:
    """Production training function for PriorityNet"""
    
    logging.info("🧠 Phase 2: Training Production PriorityNet...")
    
    # Initialize model and trainer
    model = PriorityNet(config)
    trainer = PriorityNetTrainer(model, config)
    
    
    # Data loader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_priority_batch,
        num_workers=0
    )
    
    # Training parameters
    n_epochs = 300  # Reduced for faster training
    best_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    training_metrics = {
        'losses': [],
        'epochs_completed': 0
    }
    
    # Training loop
    for epoch in range(n_epochs):
        epoch_losses = []
        
        for detections_batch, priorities_batch in tqdm(dataloader, desc=f'Epoch {epoch+1}'):
            loss_info = trainer.train_step(detections_batch, priorities_batch)
            epoch_losses.append(loss_info['loss'])
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        training_metrics['losses'].append(avg_loss)
        training_metrics['epochs_completed'] = epoch + 1
        
        # Logging
        if epoch % 25 == 0 or epoch == n_epochs - 1:
            logging.info(f"Epoch {epoch:3d}: Average Loss = {avg_loss:.6f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'config': config.__dict__ if hasattr(config, '__dict__') else {}
            }, output_dir / 'priority_net_best.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
    model.eval()
    evaluation_metrics = evaluate_priority_net(model, dataset)
    
    # Save final results
    final_results = {
        'training_metrics': training_metrics,
        'evaluation_metrics': evaluation_metrics,
        'model_config': config.__dict__ if hasattr(config, '__dict__') else {},
        'total_epochs': training_metrics['epochs_completed'],
        'final_loss': training_metrics['losses'][-1] if training_metrics['losses'] else 0.0
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_results': final_results
    }, output_dir / 'priority_net_final.pth')
    
    with open(output_dir / 'priority_net_evaluation.pkl', 'wb') as f:
        pickle.dump(evaluation_metrics, f)
    
    logging.info("✅ Production PriorityNet training completed!")
    
    return evaluation_metrics

def evaluate_priority_net(model: PriorityNet, dataset: PriorityNetDataset) -> Dict[str, Any]:
    """Evaluate trained PriorityNet"""
    
    logging.info("📊 Evaluating PriorityNet performance...")
    
    model.eval()
    
    correlations = []
    precisions = []
    accuracies = []
    
    with torch.no_grad():
        for item in dataset:
            detections = item['detections']
            true_priorities = item['priorities']
            
            if len(detections) <= 1:
                continue
            
            try:
                # Get model predictions
                pred_priorities = model.forward(detections)
                
                if len(pred_priorities) != len(true_priorities):
                    continue
                
                # Ranking correlation
                true_ranking = torch.argsort(true_priorities, descending=True)
                pred_ranking = torch.argsort(pred_priorities, descending=True)
                
                # Spearman correlation approximation
                n = len(true_ranking)
                if n > 1:
                    correlation = 1.0 - 6 * torch.sum((true_ranking.float() - pred_ranking.float())**2) / (n * (n**2 - 1))
                    correlations.append(float(correlation))
                
                # Top-k precision
                k = min(3, len(detections))
                true_top_k = set(true_ranking[:k].tolist())
                pred_top_k = set(pred_ranking[:k].tolist())
                precision = len(true_top_k & pred_top_k) / k
                precisions.append(precision)
                
                # Priority accuracy
                priority_error = torch.mean(torch.abs(pred_priorities - true_priorities))
                accuracy = 1.0 / (1.0 + priority_error)
                accuracies.append(float(accuracy))
                
            except Exception as e:
                logging.debug(f"Evaluation error: {e}")
                continue
    
    # Compile results
    results = {}
    
    if correlations:
        results.update({
            'avg_ranking_correlation': np.mean(correlations),
            'std_ranking_correlation': np.std(correlations),
            'count_ranking_correlation': len(correlations)
        })
    
    if precisions:
        results.update({
            'avg_top_k_precision': np.mean(precisions),
            'std_top_k_precision': np.std(precisions),
            'count_top_k_precision': len(precisions)
        })
    
    if accuracies:
        results.update({
            'avg_priority_accuracy': np.mean(accuracies),
            'std_priority_accuracy': np.std(accuracies),
            'count_priority_accuracy': len(accuracies)
        })
    
    logging.info(f"📈 Evaluation completed: {len(correlations)} samples evaluated")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='hase32: Train Production PriorityNet for AHSD')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--data_dir', required=True, help='Training data directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logging.info("🚀 Starting Phase 2: Production PriorityNet Training")
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = type('Config', (), config_dict.get('priority_net', {
            'hidden_dims': [256, 128, 64],
            'dropout': 0.1,
            'learning_rate': 1e-3
        }))()
        
    except Exception as e:
        logging.warning(f"Could not load config: {e}, using defaults")
        config = type('Config', (), {
            'hidden_dims': [256, 128, 64],
            'dropout': 0.1,
            'learning_rate': 1e-3
        })()
    
    # Load training data
    data_dir = Path(args.data_dir)
    
    try:
        with open(data_dir / 'training_scenarios.pkl', 'rb') as f:
            scenarios = pickle.load(f)
        
        logging.info(f"✅ Loaded {len(scenarios)} training scenarios")
        
    except Exception as e:
        logging.error(f"❌ Failed to load training data: {e}")
        return
    
    # Create dataset
    dataset = PriorityNetDataset(scenarios)
    
    if len(dataset) == 0:
        logging.error("❌ No valid training data for PriorityNet")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    evaluation_metrics = train_priority_net(config, dataset, output_dir)
    
    # Print results
    logging.info("✅ Phase 2: Production PriorityNet Training COMPLETED")
    for key, value in evaluation_metrics.items():
        if isinstance(value, float):
            logging.info(f"📊 {key}: {value:.4f}")
    
    print("\n" + "="*60)
    print("✅ PHASE 2 COMPLETE - PRODUCTION PRIORITYNET")
    print("="*60)
    
    if 'avg_ranking_correlation' in evaluation_metrics:
        print(f"📈 Ranking Correlation: {evaluation_metrics['avg_ranking_correlation']:.1%}")
    if 'avg_top_k_precision' in evaluation_metrics:
        print(f"🎯 Top-K Precision: {evaluation_metrics['avg_top_k_precision']:.1%}")
    if 'avg_priority_accuracy' in evaluation_metrics:
        print(f"✅ Priority Accuracy: {evaluation_metrics['avg_priority_accuracy']:.1%}")
    
    print("="*60)

if __name__ == '__main__':
    main()
