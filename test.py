# #!/usr/bin/env python3
# """
# PRODUCTION Phase 2: Train PriorityNet for intelligent signal extraction ordering with NS support
# Enhanced to handle BBH, BNS, and NSBH systems with proper tidal effects and binary-type-aware prioritization
# """


# import sys
# import os
# import numpy as np 
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader 
# import argparse
# import pickle
# from pathlib import Path
# import logging
# from tqdm import tqdm
# import yaml
# from typing import List, Dict, Tuple, Any, Optional
# import warnings
# warnings.filterwarnings('ignore')


# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))


# try:
#     from ahsd.core.priority_net import PriorityNet, PriorityNetTrainer
#     print('PriorityNetTrainer loaded from:', sys.modules['ahsd.core.priority_net'].__file__)
#     IMPORTS_OK = True
# except ImportError as e:
#     print(f"Warning: Could not import PriorityNet modules: {e}")
#     IMPORTS_OK = False


# def setup_logging(verbose: bool = False):
#     level = logging.DEBUG if verbose else logging.INFO
#     logging.basicConfig(
#         level=level,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler('phase2_production_priority_net_ns.log'),
#             logging.StreamHandler()
#         ]
#     )


# class EnhancedPriorityNet(nn.Module):
#     """Enhanced PriorityNet with comprehensive NS support for BBH, BNS, and NSBH systems"""
    
#     def __init__(self, config):
#         super().__init__()
        
#         # Enhanced input dimension for NS features
#         self.input_dim = 15  # Expanded feature set including NS characteristics
        
#         hidden_dims = getattr(config, 'hidden_dims', [512, 256, 128, 64])
#         dropout = getattr(config, 'dropout', 0.15)
        
#         layers = []
#         prev_dim = self.input_dim
        
#         for i, hidden_dim in enumerate(hidden_dims):
#             layers.extend([
#                 nn.Linear(prev_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.5)
#             ])
#             prev_dim = hidden_dim
        
#         # Output layer for priority prediction
#         layers.append(nn.Linear(prev_dim, 1))
#         layers.append(nn.Sigmoid())  # Priority between 0 and 1
        
#         self.network = nn.Sequential(*layers)
        
#         # NS-specific parameters
#         self.ns_weight_factor = getattr(config, 'ns_weight_factor', 1.2)
#         self.tidal_bonus = getattr(config, 'tidal_bonus', 0.1)
#         self.binary_type_aware = getattr(config, 'binary_type_aware', True)
        
#         logging.info(f"✅ Enhanced PriorityNet initialized with {self.input_dim} input features")
#         logging.info(f"   Hidden layers: {hidden_dims}")
#         logging.info(f"   NS weight factor: {self.ns_weight_factor}")
#         logging.info(f"   Tidal bonus: {self.tidal_bonus}")
    
#     def extract_signal_features(self, signal: Dict) -> torch.Tensor:
#         """Extract enhanced features including NS characteristics"""
        
#         # Basic parameters
#         snr = signal.get('network_snr', 10.0)
#         m1 = signal.get('mass_1', 30.0)
#         m2 = signal.get('mass_2', 25.0) 
#         distance = signal.get('luminosity_distance', 500.0)
        
#         # Binary type encoding
#         binary_type = signal.get('binary_type', 'BBH')
#         is_bbh = 1.0 if binary_type == 'BBH' else 0.0
#         is_bns = 1.0 if binary_type == 'BNS' else 0.0
#         is_nsbh = 1.0 if binary_type == 'NSBH' else 0.0
        
#         # Tidal parameters
#         lambda_1 = signal.get('lambda_1', 0.0)
#         lambda_2 = signal.get('lambda_2', 0.0)
#         has_tidal = 1.0 if (lambda_1 > 0 or lambda_2 > 0) else 0.0
        
#         # Approximant encoding
#         approximant = signal.get('approximant', 'IMRPhenomPv2')
#         is_tidal_approx = 1.0 if 'NRTidal' in approximant else 0.0
        
#         # Derived features
#         total_mass = m1 + m2
#         chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
#         mass_ratio = min(m1, m2) / max(m1, m2)
        
#         # NS-specific derived features
#         ns_mass_count = sum(1 for m in [m1, m2] if m <= 3.0)  # Count NS components
#         effective_lambda = (lambda_1 + lambda_2) / 2.0 if (lambda_1 > 0 or lambda_2 > 0) else 0.0
        
#         # Difficulty indicators
#         difficulty = signal.get('difficulty', 'medium')
#         is_hard = 1.0 if difficulty in ['hard', 'extreme'] else 0.0
        
#         # Additional physics features
#         symmetric_mass_ratio = (m1 * m2) / (m1 + m2)**2
        
#         features = torch.tensor([
#             # Basic parameters (log-normalized for better learning)
#             np.log10(snr + 1e-6) / 2.0,  # Normalize to ~[0,1]
#             np.log10(total_mass) / 2.5,   # Normalize to ~[0,1]  
#             np.log10(chirp_mass) / 2.5,   # Normalize to ~[0,1]
#             mass_ratio,                   # Already [0,1]
#             np.log10(distance) / 4.0,     # Normalize to ~[0,1]
            
#             # Binary type features
#             is_bbh, is_bns, is_nsbh,
            
#             # NS-specific features  
#             ns_mass_count / 2.0,          # 0, 0.5, or 1.0
#             has_tidal,                    # 0 or 1
#             np.log10(effective_lambda + 1.0) / 4.0,  # Log of tidal parameter, normalized
#             is_tidal_approx,              # 0 or 1
            
#             # Additional features
#             symmetric_mass_ratio * 4.0,   # Scale symmetric mass ratio
#             is_hard,                      # Difficulty indicator
            
#             # Cross-features for NS systems
#             is_bns * np.log10(effective_lambda + 1.0) / 4.0,  # BNS-tidal interaction
#         ], dtype=torch.float32)
        
#         return features
    
#     def forward(self, detections: List[Dict]) -> torch.Tensor:
#         """Forward pass with enhanced NS-aware processing"""
        
#         if not detections:
#             return torch.tensor([])
        
#         # Extract enhanced features for each detection
#         features_list = []
#         for detection in detections:
#             features = self.extract_signal_features(detection)
#             features_list.append(features)
        
#         # Stack and process
#         if len(features_list) > 1:
#             batch_features = torch.stack(features_list)
#             priorities = self.network(batch_features).squeeze(-1)
#         else:
#             priorities = self.network(features_list[0].unsqueeze(0)).squeeze()
#             if priorities.dim() == 0:
#                 priorities = priorities.unsqueeze(0)
        
#         # Apply NS-specific weighting if enabled
#         if self.binary_type_aware:
#             ns_weights = torch.ones_like(priorities)
#             for i, detection in enumerate(detections):
#                 binary_type = detection.get('binary_type', 'BBH')
#                 if binary_type in ['BNS', 'NSBH']:
#                     ns_weights[i] *= self.ns_weight_factor
            
#             priorities = priorities * ns_weights
        
#         return priorities


# class EnhancedPriorityNetTrainer:
#     """Enhanced trainer for NS-aware PriorityNet"""
    
#     def __init__(self, model: EnhancedPriorityNet, config):
#         self.model = model
#         self.config = config
        
#         # Enhanced optimizer for NS systems
#         self.optimizer = torch.optim.AdamW(
#             model.parameters(),
#             lr=getattr(config, 'learning_rate', 0.0008),
#             weight_decay=getattr(config, 'weight_decay', 1e-4),
#             betas=(0.9, 0.999)
#         )
        
#         # Learning rate scheduler
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer,
#             mode='min',
#             factor=0.7,
#             patience=15,
#             verbose=True
#         )
        
#         # Enhanced loss function
#         self.criterion = nn.MSELoss()
#         self.ranking_loss_weight = getattr(config, 'ranking_loss_weight', 0.3)
        
#         logging.info("✅ Enhanced PriorityNet trainer initialized")
    
#     def ranking_loss(self, pred_priorities: torch.Tensor, true_priorities: torch.Tensor) -> torch.Tensor:
#         """Ranking-aware loss function for better priority ordering"""
        
#         if len(pred_priorities) <= 1:
#             return torch.tensor(0.0)
        
#         # Get rankings
#         true_ranking = torch.argsort(true_priorities, descending=True)
#         pred_ranking = torch.argsort(pred_priorities, descending=True)
        
#         # Ranking loss (penalize incorrect ordering)
#         ranking_error = 0.0
#         n = len(true_ranking)
        
#         for i in range(n):
#             for j in range(i + 1, n):
#                 true_i, true_j = true_ranking[i], true_ranking[j]
#                 pred_i_pos = torch.where(pred_ranking == true_i)[0]
#                 pred_j_pos = torch.where(pred_ranking == true_j)[0]
                
#                 if len(pred_i_pos) > 0 and len(pred_j_pos) > 0:
#                     if pred_i_pos[0] > pred_j_pos[0]:  # Wrong order
#                         ranking_error += 1.0
        
#         ranking_error = ranking_error / (n * (n - 1) / 2) if n > 1 else 0.0
#         return torch.tensor(ranking_error)
    
#     def train_step(self, detections_batch: List[List[Dict]], 
#                   priorities_batch: List[torch.Tensor]) -> Dict[str, float]:
#         """Enhanced training step with NS-aware loss computation"""
        
#         self.model.train()
#         self.optimizer.zero_grad()
        
#         total_loss = 0.0
#         mse_loss_total = 0.0
#         ranking_loss_total = 0.0
#         batch_size = len(detections_batch)
        
#         for detections, true_priorities in zip(detections_batch, priorities_batch):
#             if len(detections) == 0:
#                 continue
            
#             try:
#                 # Forward pass
#                 pred_priorities = self.model(detections)
                
#                 if len(pred_priorities) != len(true_priorities):
#                     continue
                
#                 # MSE loss
#                 mse_loss = self.criterion(pred_priorities, true_priorities)
                
#                 # Ranking loss
#                 ranking_loss = self.ranking_loss(pred_priorities, true_priorities)
                
#                 # Combined loss
#                 combined_loss = mse_loss + self.ranking_loss_weight * ranking_loss
                
#                 total_loss += combined_loss
#                 mse_loss_total += mse_loss.item()
#                 ranking_loss_total += ranking_loss.item() if isinstance(ranking_loss, torch.Tensor) else ranking_loss
                
#             except Exception as e:
#                 logging.debug(f"Training step error: {e}")
#                 continue
        
#         if total_loss > 0:
#             total_loss.backward()
            
#             # Gradient clipping for stability
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
#             self.optimizer.step()
        
#         return {
#             'loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
#             'mse_loss': mse_loss_total / batch_size,
#             'ranking_loss': ranking_loss_total / batch_size
#         }


# class PriorityNetDataset(Dataset):
#     """Enhanced production dataset for PriorityNet training with NS support"""
    
#     def __init__(self, scenarios: List[Dict]):
#         self.data = []
#         self.logger = logging.getLogger(__name__)
        
#         # Statistics tracking
#         self.stats = {
#             'total_scenarios': len(scenarios),
#             'valid_scenarios': 0,
#             'bbh_scenarios': 0,
#             'bns_scenarios': 0,
#             'nsbh_scenarios': 0,
#             'tidal_systems': 0
#         }
        
#         for scenario_id, scenario in enumerate(scenarios):
#             try:
#                 true_params = scenario.get('true_parameters', [])
#                 baseline_results = scenario.get('baseline_biases', [])
                
#                 if not true_params:
#                     continue
                
#                 # Track binary types
#                 binary_types = [p.get('binary_type', 'BBH') for p in true_params]
#                 for bt in binary_types:
#                     self.stats[f'{bt.lower()}_scenarios'] += 1
                
#                 # Track tidal systems
#                 tidal_count = sum(1 for p in true_params if 'lambda_1' in p or 'lambda_2' in p)
#                 self.stats['tidal_systems'] += tidal_count
                
#                 # Compute enhanced physics-based priorities
#                 priorities = self._compute_enhanced_extraction_priorities(true_params, baseline_results)
                
#                 if priorities is not None and len(priorities) > 0:
#                     self.data.append({
#                         'scenario_id': scenario_id,
#                         'detections': true_params,
#                         'priorities': priorities,
#                         'binary_types': binary_types,
#                         'has_ns': any(bt in ['BNS', 'NSBH'] for bt in binary_types)
#                     })
#                     self.stats['valid_scenarios'] += 1
                    
#             except Exception as e:
#                 self.logger.debug(f"Error processing scenario {scenario_id}: {e}")
#                 continue
        
#         self.logger.info(f"✅ Enhanced PriorityNet dataset created:")
#         self.logger.info(f"   Valid scenarios: {self.stats['valid_scenarios']}/{self.stats['total_scenarios']}")
#         self.logger.info(f"   BBH scenarios: {self.stats['bbh_scenarios']}")
#         self.logger.info(f"   BNS scenarios: {self.stats['bns_scenarios']}")
#         self.logger.info(f"   NSBH scenarios: {self.stats['nsbh_scenarios']}")
#         self.logger.info(f"   Tidal systems: {self.stats['tidal_systems']}")
    
#     def _compute_enhanced_extraction_priorities(self, signals: List[Dict], 
#                                               baseline_biases: Optional[List[Dict]] = None) -> torch.Tensor:
#         """Enhanced extraction priorities with comprehensive NS support"""
        
#         n_signals = len(signals)
#         priorities = torch.zeros(n_signals)
        
#         for i, signal in enumerate(signals):
#             # Get signal parameters
#             snr = signal.get('network_snr', 10.0)
#             m1 = signal.get('mass_1', 30.0)
#             m2 = signal.get('mass_2', 25.0)
#             distance = signal.get('luminosity_distance', 500.0)
#             binary_type = signal.get('binary_type', 'BBH')
            
#             # SNR component (enhanced scaling)
#             if binary_type == 'BNS':
#                 snr_priority = min(snr / 20.0, 1.0)  # BNS typically lower SNR threshold
#             elif binary_type == 'NSBH':
#                 snr_priority = min(snr / 22.0, 1.0)  # NSBH intermediate threshold
#             else:
#                 snr_priority = min(snr / 25.0, 1.0)  # BBH standard threshold
            
#             # Binary-type-specific mass priority
#             total_mass = m1 + m2
            
#             if binary_type == 'BNS':
#                 # BNS systems: 2-5 M☉ total mass
#                 if 2.5 <= total_mass <= 4.0:
#                     mass_priority = 1.0  # Optimal BNS range
#                 elif 2.0 <= total_mass <= 5.0:
#                     mass_priority = 0.8  # Good BNS range
#                 else:
#                     mass_priority = 0.5  # Unusual BNS mass
                    
#             elif binary_type == 'NSBH':
#                 # NSBH systems: 6-50 M☉ total mass
#                 if 8.0 <= total_mass <= 30.0:
#                     mass_priority = 0.9  # Good NSBH range
#                 elif 6.0 <= total_mass <= 50.0:
#                     mass_priority = 0.7  # Acceptable NSBH range
#                 else:
#                     mass_priority = 0.4  # Extreme NSBH mass
                    
#             else:  # BBH systems
#                 # BBH systems: 10-200 M☉ total mass
#                 if 25 <= total_mass <= 75:
#                     mass_priority = 1.0  # Optimal BBH range
#                 elif 15 <= total_mass <= 100:
#                     mass_priority = 0.7  # Moderate BBH
#                 else:
#                     mass_priority = 0.4  # Difficult BBH
            
#             # Binary-type-specific distance scaling
#             if binary_type == 'BNS':
#                 # BNS are typically closer and harder to detect at distance
#                 distance_priority = max(0.3, min(1.0, 200.0 / distance))
#             elif binary_type == 'NSBH':
#                 # NSBH intermediate distance sensitivity
#                 distance_priority = max(0.25, min(1.0, 500.0 / distance))
#             else:  # BBH
#                 # BBH can be detected at larger distances
#                 distance_priority = max(0.2, min(1.0, 800.0 / distance))
            
#             # Tidal effects bonus for NS systems
#             tidal_bonus = 0.0
#             if binary_type in ['BNS', 'NSBH']:
#                 # NS systems get priority bonus for tidal physics
#                 tidal_bonus = 0.1
                
#                 # Extra bonus if tidal parameters present
#                 lambda_1 = signal.get('lambda_1', 0.0)
#                 lambda_2 = signal.get('lambda_2', 0.0)
#                 if lambda_1 > 0 or lambda_2 > 0:
#                     tidal_bonus += 0.05
                    
#                     # Bonus for realistic tidal values
#                     effective_lambda = (lambda_1 + lambda_2) / 2.0
#                     if 100 <= effective_lambda <= 2000:
#                         tidal_bonus += 0.03  # Realistic tidal range bonus
            
#             # Approximant-based priority
#             approximant = signal.get('approximant', 'IMRPhenomPv2')
#             approx_priority = 0.0
#             if 'NRTidal' in approximant:
#                 approx_priority = 0.05  # Tidal approximants are more complex
            
#             # Difficulty-based adjustment
#             difficulty = signal.get('difficulty', 'medium')
#             difficulty_factor = {
#                 'easy': 1.1,
#                 'medium': 1.0,
#                 'hard': 0.8,
#                 'extreme': 0.6
#             }.get(difficulty, 1.0)
            
#             # Bias penalty (enhanced)
#             bias_penalty = 0.0
#             if baseline_biases and i < len(baseline_biases) and baseline_biases[i]:
#                 try:
#                     bias_values = [abs(b) for b in baseline_biases[i].values() if isinstance(b, (int, float))]
#                     if bias_values:
#                         bias_magnitude = np.mean(bias_values)
#                         bias_penalty = min(0.3, bias_magnitude * 0.5)
#                 except:
#                     bias_penalty = 0.0
            
#             # Combined priority with enhanced NS considerations
#             base_priority = (0.30 * snr_priority + 
#                            0.25 * mass_priority + 
#                            0.20 * distance_priority + 
#                            0.10 * tidal_bonus +
#                            0.05 * approx_priority +
#                            0.05 * bias_penalty) * difficulty_factor
            
#             # Hierarchical penalty (later extractions are harder)
#             hierarchy_penalty = i * 0.06  # Slightly increased for NS complexity
            
#             # NS system bonus
#             if binary_type in ['BNS', 'NSBH']:
#                 base_priority *= 1.1  # 10% bonus for NS systems
            
#             final_priority = max(0.1, min(1.0, base_priority - hierarchy_penalty))
#             priorities[i] = final_priority
        
#         return priorities
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data[idx]


# def collate_priority_batch(batch: List[Dict]) -> Tuple[List[List[Dict]], List[torch.Tensor]]:
#     """Enhanced collate function for variable-length sequences"""
    
#     detections_batch = []
#     priorities_batch = []
    
#     for item in batch:
#         detections_batch.append(item['detections'])
#         priorities_batch.append(item['priorities'])
    
#     return detections_batch, priorities_batch


# def train_enhanced_priority_net(config, dataset: PriorityNetDataset, output_dir: Path) -> Dict[str, Any]:
#     """Enhanced production training function for NS-aware PriorityNet"""
    
#     logging.info("🧠 Phase 2: Training Enhanced PriorityNet with NS Support...")
    
#     # Initialize enhanced model and trainer
#     model = EnhancedPriorityNet(config)
#     trainer = EnhancedPriorityNetTrainer(model, config)
    
#     # Enhanced data loader
#     dataloader = DataLoader(
#         dataset,
#         batch_size=getattr(config, 'batch_size', 16),
#         shuffle=True,
#         collate_fn=collate_priority_batch,
#         num_workers=0,
#         drop_last=False
#     )
    
#     # Training parameters
#     n_epochs = getattr(config, 'n_epochs', 400)
#     best_loss = float('inf')
#     patience = getattr(config, 'patience', 40)
#     patience_counter = 0
    
#     training_metrics = {
#         'losses': [],
#         'mse_losses': [],
#         'ranking_losses': [],
#         'epochs_completed': 0,
#         'best_epoch': 0
#     }
    
#     logging.info(f"🚀 Starting training: {n_epochs} epochs, patience={patience}")
    
#     # Training loop
#     for epoch in range(n_epochs):
#         epoch_losses = []
#         epoch_mse_losses = []
#         epoch_ranking_losses = []
        
#         progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1:3d}/{n_epochs}', leave=False)
        
#         for detections_batch, priorities_batch in progress_bar:
#             loss_info = trainer.train_step(detections_batch, priorities_batch)
            
#             epoch_losses.append(loss_info['loss'])
#             epoch_mse_losses.append(loss_info['mse_loss'])
#             epoch_ranking_losses.append(loss_info['ranking_loss'])
            
#             # Update progress bar
#             progress_bar.set_postfix({
#                 'Loss': f"{loss_info['loss']:.4f}",
#                 'MSE': f"{loss_info['mse_loss']:.4f}",
#                 'Rank': f"{loss_info['ranking_loss']:.4f}"
#             })
        
#         # Epoch statistics
#         avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
#         avg_mse_loss = np.mean(epoch_mse_losses) if epoch_mse_losses else 0.0
#         avg_ranking_loss = np.mean(epoch_ranking_losses) if epoch_ranking_losses else 0.0
        
#         training_metrics['losses'].append(avg_loss)
#         training_metrics['mse_losses'].append(avg_mse_loss)
#         training_metrics['ranking_losses'].append(avg_ranking_loss)
#         training_metrics['epochs_completed'] = epoch + 1
        
#         # Learning rate scheduling
#         trainer.scheduler.step(avg_loss)
        
#         # Logging
#         if epoch % 5 == 0 or epoch == n_epochs - 1:
#             logging.info(f"Epoch {epoch:3d}: Loss={avg_loss:.6f}, MSE={avg_mse_loss:.6f}, Rank Loss={avg_ranking_loss:.6f}")
        
#         # Enhanced early stopping
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             patience_counter = 0
#             training_metrics['best_epoch'] = epoch
            
#             # Save best model
#             torch.save({
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': trainer.optimizer.state_dict(),
#                 'scheduler_state_dict': trainer.scheduler.state_dict(),
#                 'epoch': epoch,
#                 'loss': best_loss,
#                 'config': config.__dict__ if hasattr(config, '__dict__') else {},
#                 'dataset_stats': dataset.stats
#             }, output_dir / 'priority_net_best.pth')
#         else:
#             patience_counter += 1
            
#         if patience_counter >= patience:
#             logging.info(f"Early stopping at epoch {epoch} (best epoch: {training_metrics['best_epoch']})")
#             break
    
#     # Final evaluation
#     logging.info("🔍 Performing final evaluation...")
#     model.eval()
#     evaluation_metrics = evaluate_enhanced_priority_net(model, dataset)
    
#     # Save final results
#     final_results = {
#         'training_metrics': training_metrics,
#         'evaluation_metrics': evaluation_metrics,
#         'model_config': config.__dict__ if hasattr(config, '__dict__') else {},
#         'dataset_stats': dataset.stats,
#         'total_epochs': training_metrics['epochs_completed'],
#         'best_epoch': training_metrics['best_epoch'],
#         'final_loss': training_metrics['losses'][-1] if training_metrics['losses'] else 0.0,
#         'best_loss': best_loss
#     }
    
#     # Save final model
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'final_results': final_results
#     }, output_dir / 'priority_net_final.pth')
    
#     # Save evaluation results
#     with open(output_dir / 'priority_net_evaluation.pkl', 'wb') as f:
#         pickle.dump(evaluation_metrics, f)
    
#     # Save training curves
#     training_curves = {
#         'epochs': list(range(training_metrics['epochs_completed'])),
#         'total_loss': training_metrics['losses'],
#         'mse_loss': training_metrics['mse_losses'],
#         'ranking_loss': training_metrics['ranking_losses']
#     }
    
#     with open(output_dir / 'training_curves.pkl', 'wb') as f:
#         pickle.dump(training_curves, f)
    
#     logging.info("✅ Enhanced PriorityNet training completed!")
    
#     return evaluation_metrics


# def evaluate_enhanced_priority_net(model: EnhancedPriorityNet, dataset: PriorityNetDataset) -> Dict[str, Any]:
#     """Enhanced evaluation with NS-specific performance analysis"""
    
#     logging.info("📊 Evaluating Enhanced PriorityNet performance...")
    
#     model.eval()
    
#     # Overall metrics
#     correlations = []
#     precisions = []
#     accuracies = []
    
#     # Binary-type-specific metrics
#     bbh_metrics = {'correlations': [], 'precisions': [], 'accuracies': [], 'count': 0}
#     bns_metrics = {'correlations': [], 'precisions': [], 'accuracies': [], 'count': 0}
#     nsbh_metrics = {'correlations': [], 'precisions': [], 'accuracies': [], 'count': 0}
#     mixed_metrics = {'correlations': [], 'precisions': [], 'accuracies': [], 'count': 0}
    
#     with torch.no_grad():
#         for item in dataset:
#             detections = item['detections']
#             true_priorities = item['priorities']
#             binary_types = item['binary_types']
#             has_ns = item['has_ns']
            
#             if len(detections) <= 1:
#                 continue
            
#             try:
#                 # Get model predictions
#                 pred_priorities = model(detections)
                
#                 if len(pred_priorities) != len(true_priorities):
#                     continue
                
#                 # Ranking correlation (Spearman-like)
#                 true_ranking = torch.argsort(true_priorities, descending=True)
#                 pred_ranking = torch.argsort(pred_priorities, descending=True)
                
#                 n = len(true_ranking)
#                 if n > 1:
#                     rank_diffs = (true_ranking.float() - pred_ranking.float())**2
#                     correlation = 1.0 - 6 * torch.sum(rank_diffs) / (n * (n**2 - 1))
#                     correlation = float(correlation)
#                     correlations.append(correlation)
                    
#                     # Top-k precision
#                     k = min(3, len(detections))
#                     true_top_k = set(true_ranking[:k].tolist())
#                     pred_top_k = set(pred_ranking[:k].tolist())
#                     precision = len(true_top_k & pred_top_k) / k
#                     precisions.append(precision)
                    
#                     # Priority accuracy
#                     priority_error = torch.mean(torch.abs(pred_priorities - true_priorities))
#                     accuracy = 1.0 / (1.0 + priority_error)
#                     accuracies.append(float(accuracy))
                    
#                     # Categorize by binary type composition
#                     unique_types = set(binary_types)
                    
#                     if len(unique_types) == 1:
#                         binary_type = list(unique_types)[0]
#                         if binary_type == 'BBH':
#                             bbh_metrics['correlations'].append(correlation)
#                             bbh_metrics['precisions'].append(precision)
#                             bbh_metrics['accuracies'].append(accuracy)
#                             bbh_metrics['count'] += 1
#                         elif binary_type == 'BNS':
#                             bns_metrics['correlations'].append(correlation)
#                             bns_metrics['precisions'].append(precision)
#                             bns_metrics['accuracies'].append(accuracy)
#                             bns_metrics['count'] += 1
#                         elif binary_type == 'NSBH':
#                             nsbh_metrics['correlations'].append(correlation)
#                             nsbh_metrics['precisions'].append(precision)
#                             nsbh_metrics['accuracies'].append(accuracy)
#                             nsbh_metrics['count'] += 1
#                     else:
#                         # Mixed scenario
#                         mixed_metrics['correlations'].append(correlation)
#                         mixed_metrics['precisions'].append(precision)
#                         mixed_metrics['accuracies'].append(accuracy)
#                         mixed_metrics['count'] += 1
                
#             except Exception as e:
#                 logging.debug(f"Evaluation error: {e}")
#                 continue
    
#     # Compile comprehensive results
#     def compute_stats(values):
#         if not values:
#             return {'mean': 0.0, 'std': 0.0, 'count': 0}
#         return {
#             'mean': float(np.mean(values)),
#             'std': float(np.std(values)),
#             'count': len(values)
#         }
    
#     results = {
#         # Overall performance
#         'overall_performance': {
#             'ranking_correlation': compute_stats(correlations),
#             'top_k_precision': compute_stats(precisions),
#             'priority_accuracy': compute_stats(accuracies)
#         },
        
#         # Binary-type-specific performance
#         'bbh_performance': {
#             'ranking_correlation': compute_stats(bbh_metrics['correlations']),
#             'top_k_precision': compute_stats(bbh_metrics['precisions']),
#             'priority_accuracy': compute_stats(bbh_metrics['accuracies']),
#             'scenario_count': bbh_metrics['count']
#         },
#         'bns_performance': {
#             'ranking_correlation': compute_stats(bns_metrics['correlations']),
#             'top_k_precision': compute_stats(bns_metrics['precisions']),
#             'priority_accuracy': compute_stats(bns_metrics['accuracies']),
#             'scenario_count': bns_metrics['count']
#         },
#         'nsbh_performance': {
#             'ranking_correlation': compute_stats(nsbh_metrics['correlations']),
#             'top_k_precision': compute_stats(nsbh_metrics['precisions']),
#             'priority_accuracy': compute_stats(nsbh_metrics['accuracies']),
#             'scenario_count': nsbh_metrics['count']
#         },
#         'mixed_performance': {
#             'ranking_correlation': compute_stats(mixed_metrics['correlations']),
#             'top_k_precision': compute_stats(mixed_metrics['precisions']),
#             'priority_accuracy': compute_stats(mixed_metrics['accuracies']),
#             'scenario_count': mixed_metrics['count']
#         },
        
#         # Summary statistics
#         'evaluation_summary': {
#             'total_scenarios_evaluated': len(correlations),
#             'ns_scenarios_evaluated': sum(1 for item in dataset if item['has_ns']),
#             'bbh_only_scenarios': bbh_metrics['count'],
#             'bns_scenarios': bns_metrics['count'],
#             'nsbh_scenarios': nsbh_metrics['count'],
#             'mixed_scenarios': mixed_metrics['count']
#         }
#     }
    
#     logging.info(f"📈 Enhanced evaluation completed:")
#     logging.info(f"   Total scenarios evaluated: {len(correlations)}")
#     logging.info(f"   BBH-only scenarios: {bbh_metrics['count']}")
#     logging.info(f"   BNS scenarios: {bns_metrics['count']}")
#     logging.info(f"   NSBH scenarios: {nsbh_metrics['count']}")
#     logging.info(f"   Mixed scenarios: {mixed_metrics['count']}")
    
#     return results


# def main():
#     parser = argparse.ArgumentParser(description='Phase 2: Train Enhanced PriorityNet for AHSD with NS Support')
#     parser.add_argument('--config', required=True, help='Config file path')
#     parser.add_argument('--data_dir', required=True, help='Training data directory')
#     parser.add_argument('--output_dir', required=True, help='Output directory')
#     parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
#     args = parser.parse_args()
    
#     setup_logging(args.verbose)
#     logging.info("🚀 Starting Phase 2: Enhanced PriorityNet Training with NS Support")
    
#     # Load enhanced configuration
#     try:
#         with open(args.config, 'r') as f:
#             config_dict = yaml.safe_load(f)
        
#         priority_config = config_dict.get('priority_net', {})
        
#         config = type('Config', (), {
#             # Enhanced architecture
#             'hidden_dims': priority_config.get('hidden_dims', [1024,512, 256, 128, 64]),
#             'dropout': priority_config.get('dropout', 0.1),
#             'learning_rate': priority_config.get('learning_rate', 0.0008),
#             'weight_decay': priority_config.get('weight_decay', 1e-4),
#             'batch_size': priority_config.get('batch_size', 16),
#             'n_epochs': priority_config.get('n_epochs', 400),
#             'patience': priority_config.get('patience', 40),
            
#             # NS-specific parameters
#             'ns_weight_factor': priority_config.get('ns_weight_factor', 1.5),
#             'tidal_bonus': priority_config.get('tidal_bonus', 0.15),
#             'binary_type_aware': priority_config.get('binary_type_aware', True),
#             'ranking_loss_weight': priority_config.get('ranking_loss_weight', 0.6),
#         })()
        
#         logging.info("✅ Enhanced configuration loaded from file")
        
#     except Exception as e:
#         logging.warning(f"Could not load config: {e}, using enhanced defaults")
#         config = type('Config', (), {
#             'hidden_dims': [512, 256, 128, 64],
#             'dropout': 0.15,
#             'learning_rate': 0.0008,
#             'weight_decay': 1e-4,
#             'batch_size': 16,
#             'n_epochs': 400,
#             'patience': 40,
#             'ns_weight_factor': 1.2,
#             'tidal_bonus': 0.1,
#             'binary_type_aware': True,
#             'ranking_loss_weight': 0.3,
#         })()
    
#     # Load training data
#     data_dir = Path(args.data_dir)
    
#     try:
#         # Try different possible filenames
#         possible_files = [
#             'training_scenarios.pkl',
#             'diversified_dataset_ns_enhanced.pkl',
#             'train_ns_enhanced.pkl'
#         ]
        
#         scenarios = None
#         for filename in possible_files:
#             filepath = data_dir / filename
#             if filepath.exists():
#                 with open(filepath, 'rb') as f:
#                     scenarios = pickle.load(f)
#                 logging.info(f"✅ Loaded {len(scenarios)} training scenarios from {filename}")
#                 break
        
#         if scenarios is None:
#             raise FileNotFoundError("No training data file found")
        
#     except Exception as e:
#         logging.error(f"❌ Failed to load training data: {e}")
#         return
    
#     # Create enhanced dataset
#     dataset = PriorityNetDataset(scenarios)
    
#     if len(dataset) == 0:
#         logging.error("❌ No valid training data for Enhanced PriorityNet")
#         return
    
#     # Create output directory
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Train enhanced model
#     evaluation_metrics = train_enhanced_priority_net(config, dataset, output_dir)
    
#     # Print comprehensive results
#     logging.info("✅ Phase 2: Enhanced PriorityNet Training COMPLETED")
    
#     def print_performance(perf_dict, name):
#         if 'ranking_correlation' in perf_dict and perf_dict['ranking_correlation']['count'] > 0:
#             corr = perf_dict['ranking_correlation']['mean']
#             prec = perf_dict['top_k_precision']['mean']
#             acc = perf_dict['priority_accuracy']['mean']
#             count = perf_dict['ranking_correlation']['count']
#             logging.info(f"📊 {name} ({count} scenarios): Corr={corr:.3f}, Prec={prec:.3f}, Acc={acc:.3f}")
    
#     # Print detailed results
#     if 'overall_performance' in evaluation_metrics:
#         print_performance(evaluation_metrics['overall_performance'], "Overall")
#         print_performance(evaluation_metrics['bbh_performance'], "BBH-only")
#         print_performance(evaluation_metrics['bns_performance'], "BNS")
#         print_performance(evaluation_metrics['nsbh_performance'], "NSBH")
#         print_performance(evaluation_metrics['mixed_performance'], "Mixed")
    
#     print("\n" + "="*80)
#     print("✅ PHASE 2 COMPLETE: ENHANCED PRIORITYNET WITH NS SUPPORT")
#     print("="*80)
    
#     if 'overall_performance' in evaluation_metrics:
#         overall = evaluation_metrics['overall_performance']
#         if overall['ranking_correlation']['count'] > 0:
#             print(f"🎯 Overall Ranking Correlation: {overall['ranking_correlation']['mean']:.1%}")
#             print(f"📈 Overall Top-K Precision: {overall['top_k_precision']['mean']:.1%}")
#             print(f"✅ Overall Priority Accuracy: {overall['priority_accuracy']['mean']:.1%}")
    
#     summary = evaluation_metrics.get('evaluation_summary', {})
#     if summary:
#         print(f"📊 Scenarios Evaluated: {summary.get('total_scenarios_evaluated', 0)}")
#         print(f"🌟 NS Systems Evaluated: {summary.get('ns_scenarios_evaluated', 0)}")
#         print(f"   BBH-only: {summary.get('bbh_only_scenarios', 0)}")
#         print(f"   BNS: {summary.get('bns_scenarios', 0)}")
#         print(f"   NSBH: {summary.get('nsbh_scenarios', 0)}")
#         print(f"   Mixed: {summary.get('mixed_scenarios', 0)}")
    
#     print("="*80)


# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
"""
Phase 3B .pth File Inspector
Inspects and displays the contents of your Phase 3B model file
"""

import torch
import sys
from pathlib import Path
import numpy as np
from typing import Any, Dict

def inspect_phase3b_file(file_path: str):
    """Comprehensive inspection of Phase 3B .pth file"""
    
    print(f"🔍 INSPECTING PHASE 3B FILE: {file_path}")
    print("="*80)
    
    try:
        # Load the file
        checkpoint = torch.load(file_path, map_location='cpu')
        print(f"✅ File loaded successfully")
        print(f"📊 File type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"📁 Dictionary with {len(checkpoint)} keys")
            
            # Display all top-level keys
            print(f"\n🗂️  TOP-LEVEL KEYS:")
            for i, key in enumerate(checkpoint.keys(), 1):
                value = checkpoint[key]
                value_type = type(value).__name__
                
                if hasattr(value, 'shape'):
                    size_info = f"shape: {value.shape}"
                elif hasattr(value, '__len__') and not isinstance(value, str):
                    size_info = f"length: {len(value)}"
                else:
                    size_info = f"value: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}"
                
                print(f"   {i:2d}. '{key}' -> {value_type} ({size_info})")
            
            # Detailed inspection of important keys
            important_keys = [
                'neural_pe_model', 'subtractor_model', 'param_names', 
                'pe_results', 'results', 'config', 'enhanced_config',
                'training_metrics', 'evaluation_metrics', 'final_results'
            ]
            
            print(f"\n🔍 DETAILED INSPECTION:")
            
            for key in important_keys:
                if key in checkpoint:
                    print(f"\n📋 KEY: '{key}'")
                    value = checkpoint[key]
                    
                    if key.endswith('_model'):
                        # Model inspection
                        print(f"   Type: {type(value)}")
                        if hasattr(value, 'state_dict'):
                            print(f"   Has state_dict: ✅")
                            state_dict = value.state_dict()
                            print(f"   Parameters: {len(state_dict)} layers")
                            total_params = sum(p.numel() for p in value.parameters())
                            print(f"   Total parameters: {total_params:,}")
                        
                        if hasattr(value, '__class__'):
                            print(f"   Model class: {value.__class__.__name__}")
                        
                        if hasattr(value, 'eval'):
                            print(f"   Model can be evaluated: ✅")
                    
                    elif key == 'param_names':
                        print(f"   Parameters: {value}")
                        print(f"   Count: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                    
                    elif key in ['pe_results', 'results', 'training_metrics', 'evaluation_metrics']:
                        # Results inspection
                        if isinstance(value, dict):
                            print(f"   Result keys: {list(value.keys())}")
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, (int, float)):
                                    print(f"     {sub_key}: {sub_value}")
                                elif isinstance(sub_value, list) and len(sub_value) > 0:
                                    print(f"     {sub_key}: list[{len(sub_value)}] (first: {sub_value[0]})")
                                else:
                                    print(f"     {sub_key}: {type(sub_value).__name__}")
                        else:
                            print(f"   Value: {value}")
                    
                    elif key in ['config', 'enhanced_config']:
                        # Config inspection
                        if isinstance(value, dict):
                            print(f"   Config keys: {list(value.keys())}")
                            for sub_key, sub_value in value.items():
                                print(f"     {sub_key}: {sub_value}")
                        else:
                            print(f"   Config type: {type(value)}")
                            print(f"   Config value: {value}")
                    
                    else:
                        # General inspection
                        if isinstance(value, dict):
                            print(f"   Dictionary keys: {list(value.keys())}")
                        elif isinstance(value, list):
                            print(f"   List length: {len(value)}")
                            if len(value) > 0:
                                print(f"   First item: {value[0]}")
                        elif isinstance(value, (int, float, str)):
                            print(f"   Value: {value}")
                        else:
                            print(f"   Type: {type(value)}")
            
            # Check for any model that can be tested
            print(f"\n🧪 MODEL TESTING:")
            
            if 'neural_pe_model' in checkpoint:
                neural_pe = checkpoint['neural_pe_model']
                try:
                    neural_pe.eval()
                    test_input = torch.randn(1, 2, 4096)  # Standard GW data shape
                    with torch.no_grad():
                        output = neural_pe(test_input)
                        if isinstance(output, tuple):
                            params, uncert = output
                            print(f"   Neural PE test: ✅ Output shape: {params.shape}, {uncert.shape}")
                            print(f"   Neural PE output range: {torch.min(params):.3f} to {torch.max(params):.3f}")
                        else:
                            print(f"   Neural PE test: ✅ Output shape: {output.shape}")
                            print(f"   Neural PE output range: {torch.min(output):.3f} to {torch.max(output):.3f}")
                except Exception as e:
                    print(f"   Neural PE test: ❌ Error: {e}")
            
            if 'subtractor_model' in checkpoint:
                subtractor = checkpoint['subtractor_model']
                try:
                    subtractor.eval()
                    test_data = torch.randn(1, 2, 4096)
                    test_uncertainties = torch.randn(1, 9)
                    with torch.no_grad():
                        cleaned, confidence = subtractor(test_data, test_uncertainties)
                        print(f"   Subtractor test: ✅ Cleaned shape: {cleaned.shape}, Confidence: {confidence.shape}")
                except Exception as e:
                    print(f"   Subtractor test: ❌ Error: {e}")
            
            # Summary
            print(f"\n📊 SUMMARY:")
            print(f"   File contains: {len(checkpoint)} main components")
            has_neural_pe = 'neural_pe_model' in checkpoint
            has_subtractor = 'subtractor_model' in checkpoint
            has_params = 'param_names' in checkpoint
            has_results = any(key in checkpoint for key in ['pe_results', 'results', 'training_metrics'])
            
            print(f"   Neural PE model: {'✅' if has_neural_pe else '❌'}")
            print(f"   Subtractor model: {'✅' if has_subtractor else '❌'}")
            print(f"   Parameter names: {'✅' if has_params else '❌'}")
            print(f"   Training results: {'✅' if has_results else '❌'}")
            
            if has_neural_pe and has_subtractor and has_params:
                print(f"   🏆 Complete Phase 3B system detected!")
            elif has_neural_pe and has_params:
                print(f"   🧠 Neural PE system detected")
            else:
                print(f"   ⚠️  Partial system detected")
        
        else:
            print(f"📊 Non-dictionary content: {type(checkpoint)}")
            if hasattr(checkpoint, 'shape'):
                print(f"   Shape: {checkpoint.shape}")
            elif hasattr(checkpoint, '__len__'):
                print(f"   Length: {len(checkpoint)}")
            else:
                print(f"   Value: {str(checkpoint)[:200]}{'...' if len(str(checkpoint)) > 200 else ''}")
    
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    print("="*80)
    return checkpoint

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Inspect Phase 3B .pth file contents')
    parser.add_argument('--file', required=True, help='Path to .pth file')
    parser.add_argument('--save_summary', help='Save summary to text file')
    
    args = parser.parse_args()
    
    # Inspect the file
    checkpoint = inspect_phase3b_file(args.file)
    
    # Save summary if requested
    if args.save_summary and checkpoint:
        summary_path = Path(args.save_summary)
        with open(summary_path, 'w') as f:
            f.write(f"Phase 3B File Inspection Summary\n")
            f.write(f"File: {args.file}\n")
            f.write(f"="*50 + "\n")
            
            if isinstance(checkpoint, dict):
                f.write(f"Keys: {list(checkpoint.keys())}\n\n")
                
                for key, value in checkpoint.items():
                    f.write(f"{key}: {type(value).__name__}\n")
                    if key == 'param_names':
                        f.write(f"  Parameters: {value}\n")
                    elif key in ['pe_results', 'results'] and isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                f.write(f"  {sub_key}: {sub_value}\n")
                    f.write("\n")
        
        print(f"📄 Summary saved to: {summary_path}")

if __name__ == '__main__':
    main()
