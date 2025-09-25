# #!/usr/bin/env python3
# """
# Phase 3A: Neural Parameter Estimation Training - OPTIMIZED
# High performance training with proper model saving
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
# from typing import List, Dict, Tuple, Any
# from scipy import signal
# import warnings
# import gc
# warnings.filterwarnings('ignore')

# # Add project root to path
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# # Import from ahsd.core
# try:
#     from ahsd.core.priority_net import PriorityNet
#     from ahsd.core.bias_corrector import BiasCorrector
# except ImportError:
#     print("Warning: Could not import AHSD modules. Continuing without them.")

# def setup_logging(verbose: bool = False):
#     level = logging.DEBUG if verbose else logging.INFO
#     logging.basicConfig(
#         level=level,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler('phase3a_neural_pe_optimized.log'),
#             logging.StreamHandler()
#         ]
#     )

# class NeuralPENetwork(nn.Module):
#     """OPTIMIZED Neural PE Network - Enhanced for 90%+ accuracy"""
    
#     def __init__(self, param_names: List[str], data_length: int = 4096):
#         super().__init__()
        
#         self.param_names = param_names
#         self.n_params = len(param_names)
#         self.data_length = data_length
        
#         # ✅ OPTIMIZED: More powerful feature extractor
#         self.feature_extractor = nn.Sequential(
#             nn.BatchNorm1d(2),
#             nn.Conv1d(2, 64, kernel_size=16, stride=2, padding=7),  # Increased from 32
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Dropout(0.1),
#             nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Dropout(0.1),
#             nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),  # Increased from 128
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(16),  # Increased from 8
#             nn.Dropout(0.1),
#             nn.Flatten(),
#         )
        
#         # Calculate exact feature size: 256 channels * 16 length = 4096
#         self.feature_size = 4096
        
#         # ✅ OPTIMIZED: More expressive predictor
#         self.param_predictor = nn.Sequential(
#             nn.Linear(self.feature_size, 512),  # Increased from 256
#             nn.ReLU(),
#             nn.Dropout(0.15),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.n_params),
#             nn.Tanh()
#         )
        
#         self.uncertainty_predictor = nn.Sequential(
#             nn.Linear(self.feature_size, 128),  # Increased from 64
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, self.n_params),
#             nn.Sigmoid()
#         )
        
#         self.apply(self._init_weights)
        
#         # ✅ OPTIMIZED: Better final layer initialization
#         with torch.no_grad():
#             final_layer = self.param_predictor[-2]
#             torch.nn.init.uniform_(final_layer.bias, -0.05, 0.05)  # Reduced from [-0.1, 0.1]
        
#         logging.info(f"✅ Optimized Neural PE Network initialized for {self.n_params} parameters")
#         logging.info(f"   Feature size: {self.feature_size}")
        
#         # Count total parameters
#         total_params = sum(p.numel() for p in self.parameters())
#         logging.info(f"   Total parameters: {total_params:,}")
    
#     def _init_weights(self, module):
#         """Enhanced weight initialization"""
#         if isinstance(module, nn.Linear):
#             torch.nn.init.xavier_normal_(module.weight, gain=1.2)  # Increased gain
#             if module.bias is not None:
#                 torch.nn.init.constant_(module.bias, 0.0)
#         elif isinstance(module, nn.Conv1d):
#             torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
#             if module.bias is not None:
#                 torch.nn.init.constant_(module.bias, 0.0)
#         elif isinstance(module, nn.BatchNorm1d):
#             torch.nn.init.constant_(module.weight, 1.0)
#             torch.nn.init.constant_(module.bias, 0.0)
    
#     def forward(self, waveform_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         waveform_data = torch.clamp(waveform_data, min=-1e3, max=1e3)
#         features = self.feature_extractor(waveform_data)
#         predicted_params = self.param_predictor(features)
#         predicted_uncertainties = 0.01 + 1.99 * self.uncertainty_predictor(features)
#         return predicted_params, predicted_uncertainties

# class AdaptiveSubtractorDataset(Dataset):
#     """OPTIMIZED dataset with better data quality"""
    
#     def __init__(self, scenarios: List[Dict], param_names: List[str]):
#         self.scenarios = scenarios
#         self.param_names = param_names
#         self.data = []
#         self.logger = logging.getLogger(__name__)
        
#         valid_scenarios = 0
#         processed_signals = 0
        
#         # ✅ OPTIMIZED: Process more scenarios for better training
#         max_scenarios = min(len(scenarios), 2000)  # Limit for memory efficiency
        
#         for scenario_id, scenario in enumerate(scenarios[:max_scenarios]):
#             try:
#                 true_params = scenario.get('true_parameters', [])
#                 if true_params:
#                     valid_scenarios += 1
#                     for signal_idx, params in enumerate(true_params):
#                         waveform_data = self._generate_synthetic_waveform(params, scenario)
#                         if waveform_data is not None:
#                             param_vector = self._extract_parameter_vector(params)
#                             if param_vector is not None:
#                                 quality = self._compute_synthetic_quality(params)
#                                 # ✅ OPTIMIZED: Only use high-quality samples
#                                 if quality > 0.3:  # Filter out poor quality
#                                     self.data.append({
#                                         'scenario_id': scenario_id,
#                                         'signal_index': signal_idx,
#                                         'waveform_data': waveform_data,
#                                         'true_parameters': param_vector,
#                                         'signal_quality': quality
#                                     })
#                                     processed_signals += 1
#             except Exception as e:
#                 self.logger.debug(f"Error processing scenario {scenario_id}: {e}")
#                 continue
        
#         self.logger.info(f"✅ Optimized Dataset: {valid_scenarios} scenarios, {processed_signals} signals, {len(self.data)} samples")

#     def _generate_synthetic_waveform(self, params: Dict, scenario: Dict) -> np.ndarray:
#         try:
#             mass_1 = max(5.0, min(100.0, self._extract_param_value(params, 'mass_1', 35.0)))
#             mass_2 = max(5.0, min(100.0, self._extract_param_value(params, 'mass_2', 30.0)))
#             distance = max(50.0, min(5000.0, self._extract_param_value(params, 'luminosity_distance', 500.0)))
            
#             t = np.linspace(0, 4.0, 4096)
#             total_mass = mass_1 + mass_2
#             chirp_mass = max(10.0, min(100.0, (mass_1 * mass_2)**(3/5) / total_mass**(1/5)))
            
#             f_start = 20.0
#             f_end = min(200.0, 220.0 / total_mass)
#             frequency = f_start + (f_end - f_start) * (t / 4.0)
#             frequency = np.clip(frequency, f_start, f_end)
            
#             base_amplitude = max(1e-24, min(1e-20, chirp_mass / distance))
#             amplitude = base_amplitude * np.exp(-t / 8.0)
#             phase = 2 * np.pi * np.cumsum(frequency) * (4.0 / 4096)
            
#             inclination = max(0.1, min(np.pi-0.1, self._extract_param_value(params, 'theta_jn', np.pi/4)))
            
#             h_plus = amplitude * (1 + np.cos(inclination)**2) * np.cos(phase)
#             h_cross = amplitude * 2 * np.cos(inclination) * np.sin(phase)
            
#             # ✅ OPTIMIZED: Better noise modeling
#             noise_level = base_amplitude * 0.005  # Reduced noise
#             h_plus += np.random.normal(0, noise_level, len(h_plus))
#             h_cross += np.random.normal(0, noise_level, len(h_cross))
            
#             h_plus = np.nan_to_num(h_plus, nan=0.0, posinf=base_amplitude, neginf=-base_amplitude)
#             h_cross = np.nan_to_num(h_cross, nan=0.0, posinf=base_amplitude, neginf=-base_amplitude)
            
#             max_val = base_amplitude * 10
#             h_plus = np.clip(h_plus, -max_val, max_val)
#             h_cross = np.clip(h_cross, -max_val, max_val)
            
#             waveform_data = np.zeros((2, 4096))
#             waveform_data[0] = h_plus
#             waveform_data[1] = h_cross
            
#             return waveform_data
            
#         except Exception:
#             t = np.linspace(0, 4, 4096)
#             clean_amplitude = 1e-23
#             clean_wave = clean_amplitude * np.sin(2 * np.pi * 50.0 * t)
#             noise = np.random.normal(0, clean_amplitude * 0.005, 4096)
            
#             waveform_data = np.zeros((2, 4096))
#             waveform_data[0] = clean_wave + noise
#             waveform_data[1] = (clean_wave + noise) * 0.7
#             return waveform_data

#     def _extract_param_value(self, params: Dict, param_name: str, default: float) -> float:
#         try:
#             value = params.get(param_name, default)
#             if isinstance(value, dict):
#                 for key in ['median', 'mean', 'value']:
#                     if key in value:
#                         return float(value[key])
#                 return default
#             return float(value)
#         except:
#             return default
    
#     def _extract_parameter_vector(self, params: Dict) -> np.ndarray:
#         try:
#             param_vector = []
            
#             defaults = {
#                 'mass_1': 35.0, 'mass_2': 30.0, 'luminosity_distance': 500.0,
#                 'ra': 0.0, 'dec': 0.0, 'geocent_time': 0.0,
#                 'theta_jn': np.pi/4, 'psi': 0.0, 'phase': 0.0
#             }

#             param_ranges = {
#                 'mass_1': (5.0, 80.0), 'mass_2': (3.0, 50.0),
#                 'luminosity_distance': (50.0, 2000.0),
#                 'ra': (0.0, 2 * np.pi), 'dec': (-np.pi/2, np.pi/2),
#                 'geocent_time': (-2.0, 2.0), 'theta_jn': (0.0, np.pi),
#                 'psi': (0.0, np.pi), 'phase': (0.0, 2 * np.pi)
#             }

#             for param_name in self.param_names:
#                 raw_value = self._extract_param_value(params, param_name, defaults.get(param_name, 0.0))

#                 if param_name in param_ranges:
#                     min_val, max_val = param_ranges[param_name]
#                     raw_value = max(min_val, min(max_val, raw_value))
                    
#                     if param_name in ['luminosity_distance']:
#                         log_val = np.log10(raw_value)
#                         log_min, log_max = np.log10(min_val), np.log10(max_val)  
#                         normalized_value = 2.0 * (log_val - log_min) / (log_max - log_min) - 1.0
#                     else:
#                         normalized_value = 2.0 * (raw_value - min_val) / (max_val - min_val) - 1.0
                    
#                     normalized_value = np.clip(normalized_value, -0.98, 0.98)
#                 else:
#                     normalized_value = np.tanh(raw_value / 10.0)

#                 param_vector.append(normalized_value)

#             return np.array(param_vector, dtype=np.float32)
#         except Exception as e:
#             self.logger.debug(f"Parameter extraction failed: {e}")
#             return None # type: ignore

#     def _compute_synthetic_quality(self, params: Dict) -> float:
#         try:
#             mass_1 = self._extract_param_value(params, 'mass_1', 35.0)
#             mass_2 = self._extract_param_value(params, 'mass_2', 30.0)
#             distance = self._extract_param_value(params, 'luminosity_distance', 500.0)
            
#             chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
#             estimated_snr = 8.0 * (chirp_mass / 30.0)**(5/6) * (400.0 / distance)
            
#             snr_quality = min(1.0, estimated_snr / 20.0)
#             mass_quality = 1.0 if 10.0 <= mass_1 <= 60.0 and 5.0 <= mass_2 <= 40.0 else 0.7
#             distance_quality = 1.0 if 100.0 <= distance <= 1500.0 else 0.8
            
#             combined_quality = 0.6 * snr_quality + 0.2 * mass_quality + 0.2 * distance_quality
#             return max(0.1, min(1.0, combined_quality))
#         except:
#             return 0.5
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data[idx]

# def collate_subtractor_batch(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     waveforms = torch.stack([torch.tensor(item['waveform_data'], dtype=torch.float32) for item in batch])
#     parameters = torch.stack([torch.tensor(item['true_parameters'], dtype=torch.float32) for item in batch])
#     qualities = torch.tensor([item['signal_quality'] for item in batch], dtype=torch.float32)
#     return waveforms, parameters, qualities

# def train_neural_pe(neural_pe: NeuralPENetwork, dataset: AdaptiveSubtractorDataset, 
#                    epochs: int = 40) -> Dict[str, Any]:
#     """OPTIMIZED Neural PE training for 90%+ accuracy"""
    
#     logging.info("🧠 Training Optimized Neural PE Network...")
    
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=48,  # Optimized batch size
#         shuffle=True,
#         collate_fn=collate_subtractor_batch, 
#         num_workers=0, 
#         pin_memory=True
#     )
    
#     # ✅ OPTIMIZED: Better optimizer settings
#     optimizer = torch.optim.AdamW(
#         neural_pe.parameters(), 
#         lr=1e-3,  # REDUCED from 3e-3 (was too high!)
#         weight_decay=1e-6,
#         betas=(0.9, 0.999)  # More conservative momentum
#     )
    
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode='max',  # Watch accuracy (max)
#     factor=0.5,  # Reduce LR by half
#     patience=5,  # Wait 5 epochs for improvement
#     min_lr=1e-6
# )
    
#     def optimized_neural_pe_loss(pred_params, pred_uncertainties, true_params, quality_weights):
#         device = pred_params.device
        
#         # ✅ OPTIMIZED: Less aggressive clamping
#         pred_params = torch.clamp(pred_params, min=-3.0, max=3.0)
#         pred_uncertainties = torch.clamp(pred_uncertainties, min=1e-6, max=10.0)
#         true_params = torch.clamp(true_params, min=-3.0, max=3.0)
#         quality_weights = torch.clamp(quality_weights, min=1e-6, max=1.0)
        
#         # Main MSE loss with quality weighting
#         param_errors = (pred_params - true_params) ** 2
#         weighted_mse = torch.mean(quality_weights.unsqueeze(1) * param_errors)
        
#         # Uncertainty regularization
#         uncertainty_loss = torch.mean(pred_uncertainties)
        
#         # ✅ OPTIMIZED: Lighter physics constraint
#         physics_penalty = torch.tensor(0.0, device=device)
#         if pred_params.size(1) >= 2:
#             mass_violation = torch.relu(pred_params[:, 1] - pred_params[:, 0])
#             physics_penalty = torch.mean(mass_violation ** 2) * 0.005
        
#         # ✅ OPTIMIZED: Scale-aware loss
#         scale_penalty = torch.tensor(0.0, device=device)
#         pred_scale = torch.mean(torch.abs(pred_params))
#         true_scale = torch.mean(torch.abs(true_params))
#         if true_scale > 0:
#             scale_diff = torch.abs(pred_scale - true_scale) / true_scale
#             scale_penalty = scale_diff * 0.1
        
#         total_loss = weighted_mse + 0.005 * uncertainty_loss + physics_penalty + scale_penalty
        
#         # Anti-vanishing gradients
#         if torch.abs(pred_params).max() < 1e-6:
#             total_loss += 2.0
        
#         if torch.isnan(total_loss) or torch.isinf(total_loss):
#             total_loss = weighted_mse
        
#         return total_loss

#     training_history = {'losses': [], 'accuracies': []}
#     debug_samples = []
#     best_accuracy = 0.0
    
#     for epoch in range(epochs):
#         epoch_losses = []
#         epoch_accuracies = []
        
#         neural_pe.train()
        
#         pbar = tqdm(dataloader, desc=f'Neural PE Epoch {epoch+1}', leave=False)
        
#         for batch_idx, (waveforms, true_params, qualities) in enumerate(pbar):
            
#             pred_params, pred_uncertainties = neural_pe(waveforms)
#             loss = optimized_neural_pe_loss(pred_params, pred_uncertainties, true_params, qualities)
            
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(neural_pe.parameters(), 1.0)
#             optimizer.step()
            
#             epoch_losses.append(loss.item())
            
#             with torch.no_grad():
#                 param_errors = torch.mean((pred_params - true_params) ** 2, dim=1)
#                 accuracy = 1.0 / (1.0 + torch.mean(param_errors))
#                 epoch_accuracies.append(float(accuracy))
            
#             # Update progress bar
#             if batch_idx % 10 == 0:
#                 pbar.set_postfix({
#                     'loss': loss.item(),
#                     'acc': np.mean(epoch_accuracies[-10:]) if epoch_accuracies else 0.0,
#                     'lr': scheduler.get_last_lr()[0]
#                 })
            
#             # Collect debug samples
#             if epoch == 0 and batch_idx == 0:
#                 debug_samples.append({
#                     'pred_params': pred_params[0].detach().cpu().numpy(),
#                     'true_params': true_params[0].detach().cpu().numpy(),
#                     'pred_uncertainties': pred_uncertainties[0].detach().cpu().numpy(),
#                     'pred_range': [pred_params.min().item(), pred_params.max().item()],
#                     'true_range': [true_params.min().item(), true_params.max().item()]
#                 })
        
#         avg_loss = np.mean(epoch_losses)
#         avg_accuracy = np.mean(epoch_accuracies)
        
#         training_history['losses'].append(avg_loss)
#         training_history['accuracies'].append(avg_accuracy)
        
#         early_stop_patience = 10
#         epochs_without_improvement = 0

#         if avg_accuracy > best_accuracy:
#             best_accuracy = avg_accuracy
#             epochs_without_improvement = 0
#         else:
#             epochs_without_improvement += 1

#         if epochs_without_improvement >= early_stop_patience:
#             logging.info(f"Early stopping at epoch {epoch}")
#             break
        
#         if avg_accuracy > best_accuracy:
#             best_accuracy = avg_accuracy
        
#         if epoch % 5 == 0:
#             logging.info(f"Neural PE Epoch {epoch}: Loss = {avg_loss:.6f}, Accuracy = {avg_accuracy:.3f}, Best = {best_accuracy:.3f}")
    
#     final_accuracy = training_history['accuracies'][-1] if training_history['accuracies'] else 0.0
    
#     # Final prediction check
#     with torch.no_grad():
#         sample_batch = next(iter(dataloader))
#         sample_preds, sample_uncertainties = neural_pe(sample_batch[0][:1])
#         pred_magnitude = torch.abs(sample_preds).max().item()
    
#     # Memory cleanup
#     del dataloader
#     gc.collect()
    
#     # Output results
#     print("\n" + "="*60)
#     print("📊 PHASE 3A OPTIMIZED NEURAL PE RESULTS")
#     print("="*60)
#     print(f"✅ Final Accuracy: {final_accuracy:.3f} ({final_accuracy:.1%})")
#     print(f"🏆 Best Accuracy: {best_accuracy:.3f} ({best_accuracy:.1%})")
#     print(f"✅ Training Completed: {epochs} epochs")
#     print(f"✅ Prediction Magnitude: {pred_magnitude:.6f}")
    
#     if debug_samples:
#         sample = debug_samples[0]
#         print("\n🔍 SAMPLE PREDICTION DEBUG:")
#         print(f"Predicted params: {sample['pred_params']}")
#         print(f"True params: {sample['true_params']}")
#         print(f"Pred range: [{sample['pred_range'][0]:.3f}, {sample['pred_range'][1]:.3f}]")
#         print(f"True range: [{sample['true_range'][0]:.3f}, {sample['true_range'][1]:.3f}]")
#         print(f"Uncertainties: {sample['pred_uncertainties']}")
        
#         if np.abs(sample['pred_params']).max() < 0.01:
#             print("⚠️  WARNING: Predictions are near zero!")
#         elif np.abs(sample['pred_params']).max() > 10:
#             print("⚠️  WARNING: Predictions are extreme!")
#         else:
#             print("✅ Predictions look reasonable")
    
#     print("="*60)
    
#     return {
#         'training_history': training_history,
#         'final_accuracy': final_accuracy,
#         'best_accuracy': best_accuracy,
#         'prediction_magnitude': pred_magnitude,
#         'debug_samples': debug_samples
#     }

# def main():
#     parser = argparse.ArgumentParser(description='Phase 3A: Optimized Neural PE Training')
#     parser.add_argument('--config', required=True, help='Config file path')
#     parser.add_argument('--data_dir', required=True, help='Training data directory')
#     parser.add_argument('--output_dir', required=True, help='Output directory')
#     parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
#     args = parser.parse_args()
    
#     setup_logging(args.verbose)
#     logging.info("🚀 Starting Phase 3A: Optimized Neural PE Training")
    
#     # Load configuration
#     try:
#         with open(args.config, 'r') as f:
#             config_dict = yaml.safe_load(f)
#         logging.info("✅ Configuration loaded")
#     except Exception as e:
#         logging.warning(f"⚠️ Could not load config: {e}. Using defaults.")
#         config_dict = {}
    
#     # Load training data
#     data_dir = Path(args.data_dir)
    
#     try:
#         training_file = None
#         possible_files = [
#             'training_scenarios.pkl',
#             'scenarios.pkl'
#         ]
        
#         for filename in possible_files:
#             if (data_dir / filename).exists():
#                 training_file = data_dir / filename
#                 break
        
#         if training_file is None:
#             raise FileNotFoundError("No training scenario file found")
        
#         with open(training_file, 'rb') as f:
#             scenarios = pickle.load(f)
#         logging.info(f"✅ Loaded {len(scenarios)} training scenarios from {training_file.name}")
        
#     except Exception as e:
#         logging.error(f"❌ Failed to load training data: {e}")
#         return
    
#     # Define parameter names
#     param_names = [
#         'mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 'geocent_time',
#         'theta_jn', 'psi', 'phase'
#     ]
    
#     # Create dataset
#     dataset = AdaptiveSubtractorDataset(scenarios, param_names)
    
#     if len(dataset) == 0:
#         logging.error("❌ No valid training data")
#         return
    
#     # Create output directory
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Initialize and train Neural PE
#     neural_pe = NeuralPENetwork(param_names)
#     pe_results = train_neural_pe(neural_pe, dataset, epochs=40)
    
#     # ✅ FIXED: Proper model saving without dataset (causes issues)
#     try:
#         # Save with minimal, safe data
#         output_data = {
#             'neural_pe_state_dict': neural_pe.state_dict(),
#             'pe_results': pe_results,
#             'param_names': param_names,
#             'model_architecture': {
#                 'n_params': neural_pe.n_params,
#                 'feature_size': neural_pe.feature_size,
#                 'data_length': neural_pe.data_length
#             },
#             'dataset_info': {
#                 'size': len(dataset),
#                 'param_names': param_names
#             }
#         }
        
#         # Save main model file
#         main_output_file = output_dir / 'phase3a_neural_pe_output.pth'
#         torch.save(output_data, main_output_file, pickle_protocol=pickle.HIGHEST_PROTOCOL)
#         logging.info(f"✅ Model saved to {main_output_file}")
        
#         # ✅ ADDITIONAL: Save just the model for easy loading
#         model_only = {
#             'model_state_dict': neural_pe.state_dict(),
#             'param_names': param_names,
#             'final_accuracy': pe_results['final_accuracy'],
#             'best_accuracy': pe_results['best_accuracy']
#         }
        
#         model_file = output_dir / 'neural_pe_model_only.pth'
#         torch.save(model_only, model_file)
#         logging.info(f"✅ Model-only saved to {model_file}")
        
#     except Exception as e:
#         logging.error(f"❌ Error saving model: {e}")
#         # Fallback: save just the essentials
#         try:
#             fallback_data = {
#                 'model_state_dict': neural_pe.state_dict(),
#                 'param_names': param_names,
#                 'final_accuracy': pe_results['final_accuracy']
#             }
#             fallback_file = output_dir / 'neural_pe_fallback.pth'
#             torch.save(fallback_data, fallback_file)
#             logging.info(f"✅ Fallback model saved to {fallback_file}")
#         except Exception as e2:
#             logging.error(f"❌ Fallback save also failed: {e2}")
    
#     # Save readable results
#     try:
#         with open(output_dir / 'phase3a_results.txt', 'w') as f:
#             f.write("PHASE 3A OPTIMIZED NEURAL PE RESULTS\n")
#             f.write("="*50 + "\n")
#             f.write(f"Final Accuracy: {pe_results['final_accuracy']:.3f}\n")
#             f.write(f"Best Accuracy: {pe_results['best_accuracy']:.3f}\n")
#             f.write(f"Prediction Magnitude: {pe_results['prediction_magnitude']:.6f}\n")
#             f.write(f"Dataset Size: {len(dataset)} samples\n")
#             f.write(f"Parameters: {param_names}\n")
#             if pe_results.get('debug_samples'):
#                 sample = pe_results['debug_samples'][0]
#                 f.write(f"Sample Prediction Range: [{sample['pred_range'][0]:.3f}, {sample['pred_range'][1]:.3f}]\n")
#                 f.write(f"Sample True Range: [{sample['true_range'][0]:.3f}, {sample['true_range'][1]:.3f}]\n")
#         logging.info("✅ Results saved to phase3a_results.txt")
#     except Exception as e:
#         logging.error(f"❌ Error saving results file: {e}")
    
#     # Performance assessment
#     final_acc = pe_results['final_accuracy']
#     if final_acc >= 0.90:
#         print("\n🏆 EXCELLENT: Ready for high-performance Phase 3B!")
#     elif final_acc >= 0.85:
#         print("\n✅ GOOD: Ready for Phase 3B!")
#     elif final_acc >= 0.80:
#         print("\n⚠️  ACCEPTABLE: Should work for Phase 3B")
#     else:
#         print("\n❌ LOW: Consider retraining or adjusting parameters")
    
#     logging.info("✅ Phase 3A optimized training completed!")

# if __name__ == '__main__':
#     main()


import torch
import numpy as np

# Simulate the problem
contaminated = torch.randn(32, 2, 4096) * 1e-21  # GW scale
clean_target = torch.randn(32, 2, 4096) * 1e-21  # GW scale  
cleaned_output = contaminated * 0.9  # Slight change

# Power calculation
cont_power = torch.mean(contaminated ** 2, dim=(1,2))
clean_power = torch.mean(cleaned_output ** 2, dim=(1,2)) 
target_power = torch.mean(clean_target ** 2, dim=(1,2))

print(f"Contaminated power: {cont_power}")
print(f"Cleaned power: {clean_power}")
print(f"Target power: {target_power}")
print(f"All essentially the same? {torch.allclose(cont_power, clean_power, atol=1e-45)}")

# This shows why efficiency = 0!
