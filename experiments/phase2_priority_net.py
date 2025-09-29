#!/usr/bin/env python3
"""
FIXED Phase 2: Train PriorityNet for intelligent signal extraction ordering
Optimized for all signal types (BBH, BNS, NSBH) with better generalization
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
from typing import List, Dict, Tuple, Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ahsd.core.priority_net import PriorityNet, PriorityNetTrainer
print('PriorityNetTrainer loaded from:', sys.modules['ahsd.core.priority_net'].__file__)

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('phase2_priority_net_fixed.log'),
            logging.StreamHandler()
        ]
    )

class PriorityNetDataset(Dataset):
    """FIXED: Production dataset for PriorityNet training with signal-type awareness"""
    
    def __init__(self, scenarios: List[Dict]):
        self.data = []
        self.logger = logging.getLogger(__name__)
        
        # Track signal type distribution
        bbh_count = bns_count = nsbh_count = 0
        
        for scenario_id, scenario in enumerate(scenarios):
            try:
                true_params = scenario.get('true_parameters', [])
                baseline_results = scenario.get('baseline_biases', [])
                
                if not true_params:
                    continue
                
                # FIXED: Enhanced priority computation
                priorities = self._compute_extraction_priorities(true_params, baseline_results)
                
                if priorities is not None and len(priorities) > 0:
                    # Count signal types for this scenario
                    for signal in true_params:
                        m1 = signal.get('mass_1', 30.0)
                        m2 = signal.get('mass_2', 25.0)
                        
                        if m1 < 3.0 or m2 < 3.0:
                            if m1 < 3.0 and m2 < 3.0:
                                bns_count += 1
                            else:
                                nsbh_count += 1
                        else:
                            bbh_count += 1
                    
                    self.data.append({
                        'scenario_id': scenario_id,
                        'detections': true_params,
                        'priorities': priorities
                    })
                    
            except Exception as e:
                self.logger.debug(f"Error processing scenario {scenario_id}: {e}")
                continue
        
        total = bbh_count + bns_count + nsbh_count
        self.logger.info(f"✅ FIXED Priority dataset created: {len(self.data)} scenarios")
        if total > 0:
            self.logger.info(f"   BBH: {bbh_count} ({bbh_count/total:.1%})")
            self.logger.info(f"   BNS: {bns_count} ({bns_count/total:.1%})")
            self.logger.info(f"   NSBH: {nsbh_count} ({nsbh_count/total:.1%})")
    
    def _compute_extraction_priorities(self, signals: List[Dict], 
                                     baseline_biases: Optional[List[Dict]] = None) -> torch.Tensor:
        """FIXED: Enhanced priority computation for all signal types"""
        
        n_signals = len(signals)
        priorities = torch.zeros(n_signals)
        
        for i, signal in enumerate(signals):
            # Basic signal properties
            snr = signal.get('network_snr', 10.0)
            m1 = signal.get('mass_1', 30.0)
            m2 = signal.get('mass_2', 25.0)
            distance = signal.get('luminosity_distance', 500.0)
            
            # FIXED: Signal type aware prioritization
            if m1 < 3.0 or m2 < 3.0:
                if m1 < 3.0 and m2 < 3.0:
                    signal_type = 'BNS'
                    type_bonus = 0.25  # Highest priority for rare BNS signals
                else:
                    signal_type = 'NSBH'
                    type_bonus = 0.15  # Medium priority for NSBH
            else:
                signal_type = 'BBH'
                type_bonus = 0.0   # Standard priority for BBH
            
            # FIXED: SNR component (more conservative for NS)
            if signal_type in ['BNS', 'NSBH']:
                snr_priority = min(snr / 15.0, 1.0)  # Lower threshold for NS
            else:
                snr_priority = min(snr / 20.0, 1.0)  # Standard for BBH
            
            # FIXED: Mass-based difficulty (signal-type aware)
            total_mass = m1 + m2
            if signal_type == 'BNS':
                # BNS: lower masses are easier to extract
                if total_mass <= 5.0:
                    mass_priority = 1.0
                elif total_mass <= 6.0:
                    mass_priority = 0.9
                else:
                    mass_priority = 0.7
            elif signal_type == 'NSBH':
                # NSBH: moderate complexity
                if 8.0 <= total_mass <= 40.0:
                    mass_priority = 1.0
                else:
                    mass_priority = 0.8
            else:  # BBH
                # BBH: moderate masses easier
                if 25 <= total_mass <= 80:
                    mass_priority = 1.0
                elif 15 <= total_mass <= 120:
                    mass_priority = 0.8
                else:
                    mass_priority = 0.6
            
            # FIXED: Distance priority (signal-type aware)
            if signal_type == 'BNS':
                # BNS typically closer
                distance_priority = max(0.4, min(1.0, 400.0 / distance))
            elif signal_type == 'NSBH':
                # NSBH intermediate
                distance_priority = max(0.3, min(1.0, 500.0 / distance))
            else:  # BBH
                # BBH can be more distant
                distance_priority = max(0.2, min(1.0, 600.0 / distance))
            
            # Baseline bias penalty (if available)
            bias_penalty = 0.0
            if baseline_biases and i < len(baseline_biases) and baseline_biases[i]:
                try:
                    bias_values = [abs(b) for b in baseline_biases[i].values() 
                                 if isinstance(b, (int, float))]
                    if bias_values:
                        bias_magnitude = np.mean(bias_values)
                        bias_penalty = min(0.2, bias_magnitude * 0.3)
                except:
                    bias_penalty = 0.0
            
            # FIXED: Combined priority with signal type awareness
            base_priority = (0.35 * snr_priority + 
                           0.25 * mass_priority + 
                           0.20 * distance_priority + 
                           0.20 * type_bonus - 
                           0.10 * bias_penalty)
            
            # FIXED: Reduced hierarchical penalty
            hierarchy_penalty = i * 0.02  # Smaller penalty
            
            # FIXED: Higher minimum priority, especially for NS
            if signal_type in ['BNS', 'NSBH']:
                min_priority = 0.20
            else:
                min_priority = 0.15
            
            final_priority = max(min_priority, base_priority - hierarchy_penalty)
            priorities[i] = final_priority
            
            # Debug logging for first few signals
            if i < 3:
                self.logger.debug(f"Signal {i}: {signal_type}, Priority={final_priority:.3f}")
        
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
    """FIXED: Enhanced training function for PriorityNet"""
    
    logging.info("🧠 Phase 2: Training FIXED PriorityNet...")
    
    # Initialize model and trainer
    model = PriorityNet(config)
    trainer = PriorityNetTrainer(model, config)
    
    # Data loader
    dataloader = DataLoader(
        dataset,
        batch_size=12,  # Increased batch size for stability
        shuffle=True,
        collate_fn=collate_priority_batch,
        num_workers=0
    )
    
    # FIXED: Training parameters for better generalization
    n_epochs = 250  # Moderate number of epochs
    best_loss = float('inf')
    patience = 25   # Increased patience
    patience_counter = 0
    
    training_metrics = {
        'losses': [],
        'epochs_completed': 0
    }
    
    # FIXED: Enhanced training loop
    for epoch in range(n_epochs):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f'FIXED Epoch {epoch+1}/{n_epochs}')
        for detections_batch, priorities_batch in pbar:
            loss_info = trainer.train_step(detections_batch, priorities_batch)
            epoch_losses.append(loss_info['loss'])
            
            # Update progress bar
            pbar.set_postfix({'Loss': f"{loss_info['loss']:.4f}"})
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        training_metrics['losses'].append(avg_loss)
        training_metrics['epochs_completed'] = epoch + 1
        
        # Logging
        if epoch % 20 == 0 or epoch == n_epochs - 1:
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
                'config': config.__dict__ if hasattr(config, '__dict__') else {},
                'fixed_version': True
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
        'final_loss': training_metrics['losses'][-1] if training_metrics['losses'] else 0.0,
        'fixed_version': True
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_results': final_results
    }, output_dir / 'priority_net_final.pth')
    
    with open(output_dir / 'priority_net_evaluation.pkl', 'wb') as f:
        pickle.dump(evaluation_metrics, f)
    
    logging.info("✅ FIXED PriorityNet training completed!")
    
    return evaluation_metrics

def evaluate_priority_net(model: PriorityNet, dataset: PriorityNetDataset) -> Dict[str, Any]:
    """FIXED: Enhanced evaluation with signal-type breakdown"""
    
    logging.info("📊 Evaluating FIXED PriorityNet performance...")
    
    model.eval()
    
    # Overall metrics
    correlations = []
    precisions = []
    accuracies = []
    
    # Signal-type specific metrics
    bbh_correlations = []
    bns_correlations = []
    nsbh_correlations = []
    
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
                
                # Overall ranking correlation
                true_ranking = torch.argsort(true_priorities, descending=True)
                pred_ranking = torch.argsort(pred_priorities, descending=True)
                
                # Spearman correlation approximation
                n = len(true_ranking)
                if n > 1:
                    correlation = 1.0 - 6 * torch.sum((true_ranking.float() - pred_ranking.float())**2) / (n * (n**2 - 1))
                    correlations.append(float(correlation))
                    
                    # FIXED: Signal-type specific correlations
                    for i, detection in enumerate(detections):
                        m1 = detection.get('mass_1', 30.0)
                        m2 = detection.get('mass_2', 25.0)
                        
                        if m1 < 3.0 or m2 < 3.0:
                            if m1 < 3.0 and m2 < 3.0:
                                bns_correlations.append(float(correlation))
                            else:
                                nsbh_correlations.append(float(correlation))
                        else:
                            bbh_correlations.append(float(correlation))
                
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
    
    # FIXED: Signal-type specific results
    if bbh_correlations:
        results['bbh_correlation'] = np.mean(bbh_correlations)
        results['bbh_count'] = len(bbh_correlations)
    
    if bns_correlations:
        results['bns_correlation'] = np.mean(bns_correlations)
        results['bns_count'] = len(bns_correlations)
    
    if nsbh_correlations:
        results['nsbh_correlation'] = np.mean(nsbh_correlations)
        results['nsbh_count'] = len(nsbh_correlations)
    
    logging.info(f"📈 FIXED Evaluation completed: {len(correlations)} samples evaluated")
    logging.info(f"   Overall correlation: {results.get('avg_ranking_correlation', 0):.3f}")
    logging.info(f"   BBH correlation: {results.get('bbh_correlation', 0):.3f}")
    logging.info(f"   BNS correlation: {results.get('bns_correlation', 0):.3f}")
    logging.info(f"   NSBH correlation: {results.get('nsbh_correlation', 0):.3f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Phase 2: FIXED PriorityNet for AHSD')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--data_dir', required=True, help='Training data directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logging.info("🚀 Starting Phase 2: FIXED PriorityNet Training")
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # FIXED: Optimized configuration
        config = type('Config', (), config_dict.get('priority_net', {
            'hidden_dims': [128, 64, 32],  # Reduced from [256, 128, 64]
            'dropout': 0.15,               # Increased dropout
            'learning_rate': 8e-4          # Slightly reduced LR
        }))()
        
    except Exception as e:
        logging.warning(f"Could not load config: {e}, using FIXED defaults")
        config = type('Config', (), {
            'hidden_dims': [128, 64, 32],
            'dropout': 0.15,
            'learning_rate': 8e-4
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
    
    # Create FIXED dataset
    dataset = PriorityNetDataset(scenarios)
    
    if len(dataset) == 0:
        logging.error("❌ No valid training data for PriorityNet")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train FIXED model
    evaluation_metrics = train_priority_net(config, dataset, output_dir)
    
    # Print results
    logging.info("✅ Phase 2: FIXED PriorityNet Training COMPLETED")
    for key, value in evaluation_metrics.items():
        if isinstance(value, float):
            logging.info(f"📊 {key}: {value:.4f}")
    
    print("\n" + "="*60)
    print("✅ PHASE 2 COMPLETE - FIXED PRIORITYNET")
    print("="*60)
    
    if 'avg_ranking_correlation' in evaluation_metrics:
        corr = evaluation_metrics['avg_ranking_correlation']
        print(f"📈 Overall Ranking Correlation: {corr:.1%}")
        
        if corr > 0.75:
            print("🏆 OUTSTANDING correlation achieved!")
        elif corr > 0.65:
            print("🎉 EXCELLENT correlation achieved!")
        elif corr > 0.55:
            print("✅ GOOD correlation achieved!")
        else:
            print("🟡 LEARNING - continue improvements")
    
    if 'avg_top_k_precision' in evaluation_metrics:
        prec = evaluation_metrics['avg_top_k_precision']
        print(f"🎯 Top-K Precision: {prec:.1%}")
    
    if 'avg_priority_accuracy' in evaluation_metrics:
        acc = evaluation_metrics['avg_priority_accuracy']
        print(f"✅ Priority Accuracy: {acc:.1%}")
    
    # Signal-type breakdown
    print(f"\n🔬 Signal-Type Performance:")
    if 'bbh_correlation' in evaluation_metrics:
        print(f"   BBH: {evaluation_metrics['bbh_correlation']:.3f}")
    if 'bns_correlation' in evaluation_metrics:
        print(f"   BNS: {evaluation_metrics['bns_correlation']:.3f}")  
    if 'nsbh_correlation' in evaluation_metrics:
        print(f"   NSBH: {evaluation_metrics['nsbh_correlation']:.3f}")
    
    print("="*60)

if __name__ == '__main__':
    main()
