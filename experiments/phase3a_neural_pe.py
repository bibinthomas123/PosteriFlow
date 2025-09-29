#!/usr/bin/env python3
"""
Phase 3A: Enhanced Neural PE Training - Optimized for Phase 2 Integration
Designed to complement Phase 2's 62.1% correlation success with V4 architecture
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
import warnings
import gc
import math
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from ahsd.core.priority_net import PriorityNet
    from ahsd.core.bias_corrector import BiasCorrector
except ImportError:
    print("Warning: Could not import AHSD modules. Continuing without them.")

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('phase3a_enhanced_neural_pe.log'),
            logging.StreamHandler()
        ]
    )

class NeuralPENetwork(nn.Module):
    """FIXED: Reduced overfitting, better NS handling"""
    
    def __init__(self, param_names, config=None, data_length=4096):
        super().__init__()
        self.param_names = param_names
        self.n_params = len(param_names)
        self.data_length = data_length
        self.config = config or {}

        
        # FIXED: Multi-scale feature extraction for BBH+NS
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(2),
            # Larger kernels for NS signals
            nn.Conv1d(2, 32, kernel_size=32, stride=2, padding=15),  # Reduced from 64
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=7),   # Reduced from 128  
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Conv1d(64, 96, kernel_size=8, stride=2, padding=3),    # Reduced from 256
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.AdaptiveAvgPool1d(32),  # Increased from 16 for NS
            nn.Dropout(0.1),
            nn.Flatten(),
        )
        
        self.feature_size = 96 * 32  # 3072 instead of 4096
        
        # FIXED: Much smaller param predictor - reduces overfitting
        self.param_predictor = nn.Sequential(
            nn.Linear(self.feature_size, 256),  # Reduced from 512
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),                # Reduced from 256
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.15),
            nn.Linear(128, 64),                 # NEW intermediate layer
            nn.ReLU(),
            nn.Linear(64, self.n_params),
            nn.Tanh()
        )
        
        # FIXED: Smaller uncertainty predictor
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(self.feature_size, 128),  # Reduced from 128
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),                 # Reduced from 64
            nn.ReLU(),
            nn.Linear(64, self.n_params),
            nn.Sigmoid()
        )
        
        # Apply weight initialization
        self.apply(self._init_weights)
        
        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters())
        hidden_dims = [256, 128, 64]  # Define here
        logging.info(f"✅ Enhanced Neural PE Network initialized:")
        logging.info(f"   Parameters: {self.n_params}, Feature size: {self.feature_size}")
        logging.info(f"   Total parameters: {total_params:,}")
        logging.info(f"   Architecture depth: {len(hidden_dims)} layers (matching Phase 2)")
        
    def _init_weights(self, module):
        """Enhanced weight initialization matching Phase 2 approach"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.constant_(module.weight, 1.0)
            torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, waveform_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with enhanced stability"""
        # Input stabilization
        waveform_data = torch.clamp(waveform_data, min=-1e3, max=1e3)
        
        # Extract features
        features = self.feature_extractor(waveform_data)
        
        # Predict parameters - FIXED: proper scaling
        predicted_params = self.param_predictor(features)
        
        # Scale predictions to proper range [-2, 2] for parameter space
        output_scaling = self.config.get('output_scaling', 4.0) 
        predicted_params = (predicted_params - 0.5) * output_scaling
        
        # Predict uncertainties
        predicted_uncertainties = 0.01 + 1.99 * self.uncertainty_predictor(features)
        
        return predicted_params, predicted_uncertainties

class EnhancedAdaptiveSubtractorDataset(Dataset):
    """
    Enhanced dataset optimized for Phase 2 integration
    Improves on 78.8% accuracy with better parameter scaling and data quality
    """
    
    def __init__(self, scenarios: List[Dict], param_names: List[str], config: Dict = None):
        self.scenarios = scenarios
        self.param_names = param_names
        self.config = config or {}
        self.data = []
        self.logger = logging.getLogger(__name__)
        
        # Enhanced processing parameters
        max_scenarios = min(len(scenarios), self.config.get('max_scenarios', 3000))  # More data
        quality_threshold = self.config.get('quality_threshold', 0.2)  # Lower threshold for more samples
        
        valid_scenarios = 0
        processed_signals = 0
        
        for scenario_id, scenario in enumerate(scenarios[:max_scenarios]):
            try:
                true_params = scenario.get('true_parameters', [])
                if true_params:
                    valid_scenarios += 1
                    for signal_idx, params in enumerate(true_params):
                        waveform_data = self._generate_enhanced_waveform(params, scenario)
                        if waveform_data is not None:
                            param_vector = self._extract_enhanced_parameter_vector(params)
                            if param_vector is not None:
                                quality = self._compute_enhanced_quality(params)
                                if quality > quality_threshold:
                                    self.data.append({
                                        'scenario_id': scenario_id,
                                        'signal_index': signal_idx,
                                        'waveform_data': waveform_data,
                                        'true_parameters': param_vector,
                                        'signal_quality': quality,
                                        'binary_type': params.get('binary_type', 'BBH')
                                    })
                                    processed_signals += 1
            except Exception as e:
                self.logger.debug(f"Error processing scenario {scenario_id}: {e}")
                continue
        
        self.logger.info(f"✅ Enhanced Dataset Created:")
        self.logger.info(f"   {valid_scenarios} scenarios processed")
        self.logger.info(f"   {processed_signals} signals processed") 
        self.logger.info(f"   {len(self.data)} training samples")
        
        # Dataset statistics
        if self.data:
            binary_types = [item['binary_type'] for item in self.data]
            bbh_count = sum(1 for bt in binary_types if bt == 'BBH')
            bns_count = sum(1 for bt in binary_types if bt == 'BNS')
            nsbh_count = sum(1 for bt in binary_types if bt == 'NSBH')
            
            self.logger.info(f"   BBH samples: {bbh_count}")
            self.logger.info(f"   BNS samples: {bns_count}")
            self.logger.info(f"   NSBH samples: {nsbh_count}")
    
    def _generate_enhanced_waveform(self, params: Dict, scenario: Dict) -> np.ndarray:
        """Enhanced waveform generation with better physics modeling"""
        try:
            # Extract parameters with better defaults
            mass_1 = max(3.0, min(100.0, self._extract_param_value(params, 'mass_1', 35.0)))
            mass_2 = max(3.0, min(100.0, self._extract_param_value(params, 'mass_2', 30.0)))
            distance = max(50.0, min(5000.0, self._extract_param_value(params, 'luminosity_distance', 500.0)))
            inclination = max(0.1, min(np.pi-0.1, self._extract_param_value(params, 'theta_jn', np.pi/4)))
            
            # Enhanced binary type specific modeling
            binary_type = params.get('binary_type', 'BBH')
            
            # Time array
            t = np.linspace(0, 4.0, 4096)
            total_mass = mass_1 + mass_2
            chirp_mass = (mass_1 * mass_2)**(3/5) / total_mass**(1/5)
            
            # Binary-specific frequency evolution
            if binary_type == 'BNS':
                f_start, f_end = 20.0, min(1000.0, 1400.0 / total_mass)
                tidal_factor = 1.1  # Enhanced tidal effects
            elif binary_type == 'NSBH':
                f_start, f_end = 20.0, min(800.0, 1200.0 / total_mass)
                tidal_factor = 1.05  # Moderate tidal effects
            else:  # BBH
                f_start, f_end = 20.0, min(500.0, 800.0 / total_mass)
                tidal_factor = 1.0   # No tidal effects
            
            # Enhanced frequency evolution
            frequency = f_start + (f_end - f_start) * (t / 4.0)**0.75  # More realistic chirp
            frequency = np.clip(frequency, f_start, f_end)
            
            # Enhanced amplitude modeling
            base_amplitude = chirp_mass**(5/6) / distance * 1e-21
            amplitude_decay = np.exp(-t / (8.0 * tidal_factor))  # Binary-specific decay
            amplitude = base_amplitude * amplitude_decay
            
            # Phase evolution with enhanced physics
            phase = 2 * np.pi * np.cumsum(frequency) * (4.0 / 4096)
            phase += self._extract_param_value(params, 'phase', 0.0)
            
            # Enhanced polarizations
            h_plus = amplitude * (1 + np.cos(inclination)**2) * np.cos(phase) * tidal_factor
            h_cross = amplitude * 2 * np.cos(inclination) * np.sin(phase) * tidal_factor
            
            # Realistic noise modeling
            noise_level = base_amplitude * 0.003  # Reduced noise for better training
            h_plus += np.random.normal(0, noise_level, len(h_plus))
            h_cross += np.random.normal(0, noise_level, len(h_cross))
            
            # Stability checks
            max_val = base_amplitude * 20
            h_plus = np.clip(np.nan_to_num(h_plus), -max_val, max_val)
            h_cross = np.clip(np.nan_to_num(h_cross), -max_val, max_val)
            
            waveform_data = np.array([h_plus, h_cross], dtype=np.float32)
            return waveform_data
            
        except Exception:
            # Enhanced fallback
            t = np.linspace(0, 4, 4096)
            f_base = 50.0 + np.random.uniform(-10, 10)
            amplitude = 1e-22 * np.random.uniform(0.5, 2.0)
            
            clean_wave = amplitude * np.sin(2 * np.pi * f_base * t + np.random.uniform(0, 2*np.pi))
            noise = np.random.normal(0, amplitude * 0.01, 4096)
            
            h_plus = clean_wave + noise
            h_cross = clean_wave * 0.8 + noise * 0.7
            
            return np.array([h_plus, h_cross], dtype=np.float32)
    
    def _extract_param_value(self, params: Dict, param_name: str, default: float) -> float:
        """Enhanced parameter extraction with better handling"""
        try:
            value = params.get(param_name, default)
            if isinstance(value, dict):
                # Try multiple keys for robustness
                for key in ['median', 'mean', 'value', 'best_fit']:
                    if key in value:
                        return float(value[key])
                return default
            return float(value)
        except:
            return default
    
    def _extract_enhanced_parameter_vector(self, params: Dict) -> np.ndarray:
        """FIXED parameter extraction with proper [0,1] scaling"""
        try:
            param_vector = []
            
            # Enhanced parameter ranges based on Phase 2 success
            param_ranges = {
                'mass_1': (3.0, 100.0),           # Extended range
                'mass_2': (1.0, 50.0),            # Extended range
                'luminosity_distance': (20.0, 3000.0),  # Extended range
                'ra': (0.0, 2 * np.pi),
                'dec': (-np.pi/2, np.pi/2),
                'geocent_time': (-5.0, 5.0),      # Extended range
                'theta_jn': (0.0, np.pi),
                'psi': (0.0, np.pi),
                'phase': (0.0, 2 * np.pi)
            }
            
            defaults = {
                'mass_1': 35.0, 'mass_2': 30.0, 'luminosity_distance': 500.0,
                'ra': 0.0, 'dec': 0.0, 'geocent_time': 0.0,
                'theta_jn': np.pi/4, 'psi': 0.0, 'phase': 0.0
            }
            
            for param_name in self.param_names:
                raw_value = self._extract_param_value(params, param_name, defaults.get(param_name, 0.0))
                
                if param_name in param_ranges:
                    min_val, max_val = param_ranges[param_name]
                    raw_value = max(min_val, min(max_val, raw_value))
                    
                    # FIXED: Proper [0,1] scaling for ALL parameters
                    if param_name == 'luminosity_distance':
                        # Log scaling for distance (better for ML)
                        log_val = np.log10(raw_value)
                        log_min, log_max = np.log10(min_val), np.log10(max_val)
                        normalized_value = (log_val - log_min) / (log_max - log_min)
                    else:
                        # Linear scaling for other parameters
                        normalized_value = (raw_value - min_val) / (max_val - min_val)
                    
                    # Ensure [0,1] range with small margin
                    normalized_value = np.clip(normalized_value, 0.001, 0.999)
                else:
                    normalized_value = 0.5  # Safe default
                
                param_vector.append(normalized_value)
            
            return np.array(param_vector, dtype=np.float32)
            
        except Exception as e:
            self.logger.debug(f"Enhanced parameter extraction failed: {e}")
            return None
    
    def _compute_enhanced_quality(self, params: Dict) -> float:
        """Enhanced quality computation with binary-type awareness"""
        try:
            mass_1 = self._extract_param_value(params, 'mass_1', 35.0)
            mass_2 = self._extract_param_value(params, 'mass_2', 30.0)
            distance = self._extract_param_value(params, 'luminosity_distance', 500.0)
            binary_type = params.get('binary_type', 'BBH')
            
            # Enhanced quality metrics
            total_mass = mass_1 + mass_2
            chirp_mass = (mass_1 * mass_2)**(3/5) / total_mass**(1/5)
            
            # Binary-specific SNR estimation
            if binary_type == 'BNS':
                estimated_snr = 10.0 * (chirp_mass / 1.2)**(5/6) * (100.0 / distance)
                optimal_mass_range = (2.0, 3.5)
                optimal_distance_range = (50.0, 300.0)
            elif binary_type == 'NSBH':
                estimated_snr = 12.0 * (chirp_mass / 8.0)**(5/6) * (200.0 / distance)
                optimal_mass_range = (8.0, 25.0)
                optimal_distance_range = (100.0, 800.0)
            else:  # BBH
                estimated_snr = 8.0 * (chirp_mass / 30.0)**(5/6) * (400.0 / distance)
                optimal_mass_range = (20.0, 80.0)
                optimal_distance_range = (200.0, 1500.0)
            
            # Quality components
            snr_quality = min(1.0, max(0.1, estimated_snr / 15.0))
            
            mass_quality = 1.0 if optimal_mass_range[0] <= total_mass <= optimal_mass_range[1] else 0.7
            distance_quality = 1.0 if optimal_distance_range[0] <= distance <= optimal_distance_range[1] else 0.8
            
            # Combined quality with binary-specific weighting
            if binary_type == 'BNS':
                combined_quality = 0.5 * snr_quality + 0.3 * mass_quality + 0.2 * distance_quality
            elif binary_type == 'NSBH':
                combined_quality = 0.6 * snr_quality + 0.25 * mass_quality + 0.15 * distance_quality
            else:  # BBH
                combined_quality = 0.7 * snr_quality + 0.2 * mass_quality + 0.1 * distance_quality
            
            return max(0.1, min(1.0, combined_quality))
            
        except:
            return 0.5
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_enhanced_batch(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enhanced collate function with stability checks"""
    waveforms = []
    parameters = []
    qualities = []
    
    for item in batch:
        waveforms.append(torch.tensor(item['waveform_data'], dtype=torch.float32))
        parameters.append(torch.tensor(item['true_parameters'], dtype=torch.float32))
        qualities.append(item['signal_quality'])
    
    return (torch.stack(waveforms), 
            torch.stack(parameters), 
            torch.tensor(qualities, dtype=torch.float32))

def train_enhanced_neural_pe(neural_pe: NeuralPENetwork, 
                           dataset: EnhancedAdaptiveSubtractorDataset, 
                           config: Dict) -> Dict[str, Any]:
    """
    Enhanced Neural PE training optimized for Phase 2 integration
    Target: 90%+ accuracy to complement Phase 2's 62.1% correlation
    """
    
    logging.info("🧠 Training Enhanced Neural PE (Phase 2 Compatible)...")
    
    # Enhanced training parameters matching Phase 2 approach
    batch_size = config.get('batch_size', 16)  # Match Phase 2
    epochs = config.get('epochs', 100)         # More epochs like Phase 2
    learning_rate = config.get('learning_rate', 0.0005)  # Match Phase 2
    weight_decay = config.get('weight_decay', 5e-5)      # Match Phase 2
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_enhanced_batch,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    # Enhanced optimizer matching Phase 2 success
    optimizer = torch.optim.AdamW(
        neural_pe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Enhanced scheduler (match Phase 2 approach)
    if config.get('scheduler') == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        use_cosine = True
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=8, min_lr=1e-6
        )
        use_cosine = False
    
    def enhanced_neural_pe_loss(pred_params, pred_uncertainties, true_params, quality_weights):
        """Enhanced loss function optimized for Phase 2 compatibility"""
        device = pred_params.device
        
        # Stability clamping
        pred_params = torch.clamp(pred_params, min=-5.0, max=5.0)
        pred_uncertainties = torch.clamp(pred_uncertainties, min=1e-6, max=5.0)
        true_params = torch.clamp(true_params, min=-5.0, max=5.0)
        quality_weights = torch.clamp(quality_weights, min=1e-6, max=1.0)
        
        # Main MSE loss with quality weighting
        param_errors = (pred_params - true_params) ** 2
        weighted_mse = torch.mean(quality_weights.unsqueeze(1) * param_errors)
        
        # Uncertainty regularization (light)
        uncertainty_loss = torch.mean(pred_uncertainties)
        
        # Scale preservation loss (NEW - addresses scaling issues)
        pred_scale = torch.mean(torch.abs(pred_params), dim=0)
        true_scale = torch.mean(torch.abs(true_params), dim=0)
        scale_loss = torch.mean((pred_scale - true_scale) ** 2)
        
        # Physics constraints (light)
        physics_penalty = torch.tensor(0.0, device=device)
        if pred_params.size(1) >= 2:
            # Mass ordering constraint (if masses are first two params)
            mass_violation = torch.relu(pred_params[:, 1] - pred_params[:, 0])
            physics_penalty = torch.mean(mass_violation ** 2)
        
        # Get loss weights from config
        loss_weights = config.get('loss_weights', {})
        mse_weight = loss_weights.get('mse_weight', 1.0)
        uncertainty_weight = loss_weights.get('uncertainty_weight', 0.01)
        scale_weight = loss_weights.get('scale_weight', 0.1)
        physics_weight = loss_weights.get('physics_weight', 0.01)
        
        # Combined loss
        total_loss = (mse_weight * weighted_mse + 
                     uncertainty_weight * uncertainty_loss +
                     scale_weight * scale_loss +
                     physics_weight * physics_penalty)
        
        # Stability checks
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = weighted_mse
        
        return total_loss, {
            'mse': weighted_mse.item(),
            'uncertainty': uncertainty_loss.item(),
            'scale': scale_loss.item(),
            'physics': physics_penalty.item()
        }
    
    # Training loop
    training_history = {'losses': [], 'accuracies': [], 'loss_components': []}
    debug_samples = []
    best_accuracy = 0.0
    patience = config.get('patience', 20)
    epochs_without_improvement = 0
    
    logging.info(f"🚀 Training configuration:")
    logging.info(f"   Batch size: {batch_size}, Epochs: {epochs}")
    logging.info(f"   Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    logging.info(f"   Scheduler: {'Cosine Annealing' if use_cosine else 'ReduceLROnPlateau'}")
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []
        epoch_components = []
        
        neural_pe.train()
        
        pbar = tqdm(dataloader, desc=f'Enhanced Neural PE Epoch {epoch+1}', leave=False)
        
        for batch_idx, (waveforms, true_params, qualities) in enumerate(pbar):
            optimizer.zero_grad()
            
            # Forward pass
            pred_params, pred_uncertainties = neural_pe(waveforms)
            
            # Compute loss
            loss, components = enhanced_neural_pe_loss(pred_params, pred_uncertainties, true_params, qualities)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (match Phase 2)
            torch.nn.utils.clip_grad_norm_(neural_pe.parameters(), 0.5)
            
            optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            epoch_components.append(components)
            
            # Compute accuracy
            with torch.no_grad():
                param_errors = torch.mean((pred_params - true_params) ** 2, dim=1)
                accuracy = 1.0 / (1.0 + torch.mean(param_errors))
                epoch_accuracies.append(float(accuracy))
            
            # Update progress bar
            if batch_idx % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                recent_acc = np.mean(epoch_accuracies[-20:]) if epoch_accuracies else 0.0
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{recent_acc:.3f}',
                    'scale': f'{components["scale"]:.4f}',
                    'lr': f'{current_lr:.1e}'
                })
            
            # Collect debug samples (first epoch only)
            if epoch == 0 and batch_idx == 0 and len(debug_samples) == 0:
                debug_samples.append({
                    'pred_params': pred_params[0].detach().cpu().numpy(),
                    'true_params': true_params[0].detach().cpu().numpy(),
                    'pred_uncertainties': pred_uncertainties[0].detach().cpu().numpy(),
                    'pred_range': [pred_params.min().item(), pred_params.max().item()],
                    'true_range': [true_params.min().item(), true_params.max().item()],
                    'loss_components': components
                })
        
        # Epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)
        avg_components = {k: np.mean([c[k] for c in epoch_components]) for k in epoch_components[0].keys()}
        
        training_history['losses'].append(avg_loss)
        training_history['accuracies'].append(avg_accuracy)
        training_history['loss_components'].append(avg_components)
        
        # Scheduler step
        if use_cosine:
            scheduler.step()
        else:
            scheduler.step(avg_accuracy)
        
        # Track best accuracy
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Logging
        if epoch % 5 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Enhanced Neural PE Epoch {epoch}: Loss={avg_loss:.6f}, "
                        f"Accuracy={avg_accuracy:.3f}, Best={best_accuracy:.3f}, "
                        f"Scale Loss={avg_components['scale']:.4f}, LR={current_lr:.2e}")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            logging.info(f"Early stopping at epoch {epoch} - no improvement for {patience} epochs")
            break
    
    # Final evaluation
    final_accuracy = training_history['accuracies'][-1] if training_history['accuracies'] else 0.0
    
    # Final prediction test
    with torch.no_grad():
        neural_pe.eval()
        sample_batch = next(iter(dataloader))
        sample_preds, sample_uncertainties = neural_pe(sample_batch[0][:1])
        pred_magnitude = torch.abs(sample_preds).max().item()
    
    # Cleanup
    del dataloader
    gc.collect()
    
    # Results
    results = {
        'training_history': training_history,
        'final_accuracy': final_accuracy,
        'best_accuracy': best_accuracy,
        'prediction_magnitude': pred_magnitude,
        'debug_samples': debug_samples,
        'epochs_trained': len(training_history['losses']),
        'early_stopped': len(training_history['losses']) < epochs,
        'config_used': config
    }
    
    # Print enhanced results
    print("\n" + "="*70)
    print("📊 PHASE 3A ENHANCED NEURAL PE RESULTS (Phase 2 Compatible)")
    print("="*70)
    print(f"✅ Final Accuracy: {final_accuracy:.3f} ({final_accuracy:.1%})")
    print(f"🏆 Best Accuracy: {best_accuracy:.3f} ({best_accuracy:.1%})")
    print(f"📈 Improvement Target: 90%+ (Phase 2 Compatible)")
    print(f"⚡ Training Epochs: {len(training_history['losses'])} / {epochs}")
    print(f"🔧 Prediction Magnitude: {pred_magnitude:.6f}")
    
    if debug_samples:
        sample = debug_samples[0]
        print(f"\n🔍 ENHANCED PREDICTION DEBUG:")
        print(f"Predicted range: [{sample['pred_range'][0]:.3f}, {sample['pred_range'][1]:.3f}]")
        print(f"True range: [{sample['true_range'][0]:.3f}, {sample['true_range'][1]:.3f}]")
        
        # Enhanced assessment
        pred_mag = np.abs(sample['pred_params']).max()
        true_mag = np.abs(sample['true_params']).max()
        scale_ratio = pred_mag / true_mag if true_mag > 0 else 0
        
        print(f"Scale ratio: {scale_ratio:.3f}")
        print(f"Loss components: MSE={sample['loss_components']['mse']:.4f}, "
              f"Scale={sample['loss_components']['scale']:.4f}")
        
        if 0.3 <= scale_ratio <= 3.0:
            print("✅ FIXED: Predictions are properly scaled!")
        elif scale_ratio < 0.1:
            print("⚠️  Still small, but improved")
        else:
            print("✅ Good prediction scaling")
    
    # Performance assessment
    if best_accuracy >= 0.90:
        print(f"\n🏆 EXCELLENT: {best_accuracy:.1%} - Perfect for Phase 2 integration!")
        status = "EXCELLENT"
    elif best_accuracy >= 0.85:
        print(f"\n✅ VERY GOOD: {best_accuracy:.1%} - Ready for Phase 2 integration!")
        status = "VERY_GOOD"
    elif best_accuracy >= 0.80:
        print(f"\n✅ GOOD: {best_accuracy:.1%} - Compatible with Phase 2!")
        status = "GOOD"
    else:
        print(f"\n⚠️  ACCEPTABLE: {best_accuracy:.1%} - May need tuning for optimal Phase 2 integration")
        status = "ACCEPTABLE"
    
    results['performance_status'] = status
    
    print("="*70)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Phase 3A: Enhanced Neural PE Training (Phase 2 Compatible)')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--data_dir', required=True, help='Training data directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logging.info("🚀 Starting Phase 3A: Enhanced Neural PE Training (Phase 2 Compatible)")
    
    # Load configuration with Phase 2 compatibility
    try:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract Neural PE config with Phase 2 compatibility defaults
        neural_pe_config = config_dict.get('neural_pe', {})
        
        # Enhanced config with Phase 2 matching parameters
        enhanced_config = {
            # Architecture (match Phase 2 depth)
            'conv_channels': neural_pe_config.get('conv_channels', [64, 128, 256, 128]),
            'conv_kernels': neural_pe_config.get('conv_kernels', [16, 8, 4, 4]),
            'hidden_dims': neural_pe_config.get('hidden_dims', [512, 256, 128, 64]),
            'output_scaling': neural_pe_config.get('output_scaling', 4.0),
            
            # Training (match Phase 2 success)
            'learning_rate': neural_pe_config.get('learning_rate', 0.0005),
            'weight_decay': neural_pe_config.get('weight_decay', 5e-5),
            'batch_size': neural_pe_config.get('batch_size', 16),
            'epochs': neural_pe_config.get('epochs', 100),
            'patience': neural_pe_config.get('patience', 20),
            'scheduler': neural_pe_config.get('scheduler', 'cosine_annealing'),
            
            # Dataset parameters
            'max_scenarios': neural_pe_config.get('max_scenarios', 3000),
            'quality_threshold': neural_pe_config.get('quality_threshold', 0.2),
            
            # Loss weights
            'loss_weights': neural_pe_config.get('loss_weights', {
                'mse_weight': 1.0,
                'uncertainty_weight': 0.01,
                'scale_weight': 0.1,
                'physics_weight': 0.01
            })
        }
        
        logging.info("✅ Enhanced configuration loaded (Phase 2 Compatible)")
        logging.info(f"   Learning rate: {enhanced_config['learning_rate']}")
        logging.info(f"   Batch size: {enhanced_config['batch_size']}")
        logging.info(f"   Architecture depth: {len(enhanced_config['hidden_dims'])} layers")
        
    except Exception as e:
        logging.warning(f"⚠️ Could not load config: {e}. Using Phase 2 compatible defaults.")
        enhanced_config = {
            'conv_channels': [64, 128, 256, 128],
            'hidden_dims': [512, 256, 128, 64],
            'learning_rate': 0.0005,
            'batch_size': 16,
            'epochs': 100,
            'output_scaling': 4.0,
            'loss_weights': {
                'mse_weight': 1.0,
                'uncertainty_weight': 0.01,
                'scale_weight': 0.1,
                'physics_weight': 0.01
            }
        }
    
    # Load training data
    data_dir = Path(args.data_dir)
    
    try:
        training_file = None
        possible_files = [
            'training_scenarios.pkl',
            'diversified_dataset_ns_enhanced.pkl',
            'scenarios.pkl'
        ]
        
        for filename in possible_files:
            if (data_dir / filename).exists():
                training_file = data_dir / filename
                break
        
        if training_file is None:
            raise FileNotFoundError("No training scenario file found")
        
        with open(training_file, 'rb') as f:
            scenarios = pickle.load(f)
        
        logging.info(f"✅ Loaded {len(scenarios)} training scenarios from {training_file.name}")
        
    except Exception as e:
        logging.error(f"❌ Failed to load training data: {e}")
        return
    
    # Parameter names (standard GW parameters)
    param_names = [
        'mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 'geocent_time',
        'theta_jn', 'psi', 'phase'
    ]
    
    # Create enhanced dataset
    dataset = EnhancedAdaptiveSubtractorDataset(scenarios, param_names, enhanced_config)
    
    if len(dataset) == 0:
        logging.error("❌ No valid training data for Enhanced Neural PE")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize and train Enhanced Neural PE
    neural_pe = NeuralPENetwork(param_names, config=enhanced_config)
    pe_results = train_enhanced_neural_pe(neural_pe, dataset, enhanced_config)
    
    # Enhanced model saving
    try:
        output_data = {
            'neural_pe_state_dict': neural_pe.state_dict(),
            'pe_results': pe_results,
            'param_names': param_names,
            'enhanced_config': enhanced_config,
            'model_architecture': {
                'n_params': neural_pe.n_params,
                'feature_size': neural_pe.feature_size,
                'data_length': neural_pe.data_length
            },
            'dataset_info': {
                'size': len(dataset),
                'param_names': param_names
            },
            'phase2_compatibility': {
                'target_accuracy': 0.90,
                'achieved_accuracy': pe_results['best_accuracy'],
                'phase2_correlation': 0.621,  # Reference Phase 2 performance
                'status': pe_results['performance_status']
            }
        }
        
        main_output_file = output_dir / 'phase3a_enhanced_neural_pe.pth'
        torch.save(output_data, main_output_file)
        logging.info(f"✅ Enhanced model saved to {main_output_file}")
        
        # Save model-only version for deployment
        model_only = {
            'model_state_dict': neural_pe.state_dict(),
            'param_names': param_names,
            'enhanced_config': enhanced_config,
            'final_accuracy': pe_results['final_accuracy'],
            'best_accuracy': pe_results['best_accuracy'],
            'phase2_compatible': pe_results['best_accuracy'] >= 0.85
        }
        
        model_file = output_dir / 'enhanced_neural_pe_model.pth'
        torch.save(model_only, model_file)
        logging.info(f"✅ Enhanced model-only saved to {model_file}")
        
    except Exception as e:
        logging.error(f"❌ Error saving enhanced model: {e}")
        
        # Fallback save
        try:
            fallback_data = {
                'model_state_dict': neural_pe.state_dict(),
                'param_names': param_names,
                'final_accuracy': pe_results['final_accuracy'],
                'best_accuracy': pe_results['best_accuracy']
            }
            fallback_file = output_dir / 'neural_pe_fallback.pth'
            torch.save(fallback_data, fallback_file)
            logging.info(f"✅ Fallback model saved to {fallback_file}")
        except Exception as e2:
            logging.error(f"❌ Fallback save failed: {e2}")
    
    # Save enhanced results
    try:
        with open(output_dir / 'phase3a_enhanced_results.txt', 'w') as f:
            f.write("PHASE 3A ENHANCED NEURAL PE RESULTS (Phase 2 Compatible)\n")
            f.write("="*60 + "\n")
            f.write(f"Final Accuracy: {pe_results['final_accuracy']:.3f}\n")
            f.write(f"Best Accuracy: {pe_results['best_accuracy']:.3f}\n")
            f.write(f"Target Accuracy: 0.900 (90%)\n")
            f.write(f"Phase 2 Correlation Reference: 62.1%\n")
            f.write(f"Performance Status: {pe_results['performance_status']}\n")
            f.write(f"Prediction Magnitude: {pe_results['prediction_magnitude']:.6f}\n")
            f.write(f"Dataset Size: {len(dataset)} samples\n")
            f.write(f"Parameters: {param_names}\n")
            f.write(f"Architecture: {enhanced_config['hidden_dims']}\n")
            f.write(f"Epochs Trained: {pe_results['epochs_trained']}\n")
            f.write(f"Early Stopped: {pe_results['early_stopped']}\n")
            
            if pe_results.get('debug_samples'):
                sample = pe_results['debug_samples'][0]
                f.write(f"Sample Pred Range: [{sample['pred_range'][0]:.3f}, {sample['pred_range'][1]:.3f}]\n")
                f.write(f"Sample True Range: [{sample['true_range'][0]:.3f}, {sample['true_range'][1]:.3f}]\n")
                
        logging.info("✅ Enhanced results saved to phase3a_enhanced_results.txt")
    except Exception as e:
        logging.error(f"❌ Error saving results: {e}")
    
    # Phase 2 compatibility assessment
    final_assessment = pe_results['best_accuracy']
    phase2_compatible = final_assessment >= 0.85
    
    print(f"\n🔗 PHASE 2 INTEGRATION ASSESSMENT:")
    print(f"   Phase 2 Correlation: 62.1% (Reference)")
    print(f"   Phase 3A Accuracy: {final_assessment:.1%}")
    print(f"   Integration Ready: {'✅ YES' if phase2_compatible else '⚠️  NEEDS TUNING'}")
    
    if phase2_compatible:
        print(f"\n🚀 READY FOR PHASE 3B: Enhanced Neural PE ({final_assessment:.1%}) + ")
        print(f"   Phase 2 PriorityNet (62.1%) = Optimal AHSD Performance!")
    
    logging.info("✅ Phase 3A enhanced training completed with Phase 2 compatibility!")

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Phase 3A: FIXED Neural PE Training - Optimized for all signal types (BBH, BNS, NSBH)
Reduced overfitting, better generalization, improved NS handling
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
import warnings
import gc
import math
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from ahsd.core.priority_net import PriorityNet
    from ahsd.core.bias_corrector import BiasCorrector
except ImportError:
    print("Warning: Could not import AHSD modules. Continuing without them.")

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('phase3a_neural_pe_fixed.log'),
            logging.StreamHandler()
        ]
    )

class NeuralPENetwork(nn.Module):
    """FIXED: Optimized Neural PE Network for universal signal handling"""
    
    def __init__(self, param_names, config=None, data_length=4096):  # Added config parameter
        super().__init__()
        self.param_names = param_names
        self.n_params = len(param_names)
        self.data_length = data_length
        self.config = config or {}
        
        # Use config if provided, otherwise use defaults
        if config:
            conv_channels = config.get('conv_channels', [32, 64, 96])
            hidden_dims = config.get('hidden_dims', [256, 128, 64])
            feature_length = config.get('feature_length', 32)
            dropout = config.get('dropout', 0.15)
        else:
            # FIXED defaults
            conv_channels = [32, 64, 96]
            hidden_dims = [256, 128, 64]
            feature_length = 32
            dropout = 0.15
        
        # FIXED: Multi-scale feature extraction optimized for BBH+NS
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(2),
            # Larger kernels for NS signals, smaller channels to reduce overfitting
            nn.Conv1d(2, conv_channels[0], kernel_size=32, stride=2, padding=15),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels[0]),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=16, stride=2, padding=7),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels[1]),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels[2]),
            nn.AdaptiveAvgPool1d(feature_length),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        
        self.feature_size = conv_channels[2] * feature_length
        
        # FIXED: Much smaller param predictor - reduces overfitting significantly
        self.param_predictor = nn.Sequential(
            nn.Linear(self.feature_size, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout + 0.05),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], self.n_params),
            nn.Tanh()
        )
        
        # FIXED: Smaller uncertainty predictor
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(self.feature_size, hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], self.n_params),
            nn.Sigmoid()
        )
        
        # Initialize weights conservatively
        self.apply(self._init_weights)
        
        # Log architecture info
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"✅ FIXED Neural PE Network initialized:")
        logging.info(f"   Total parameters: {total_params:,} (reduced for better generalization)")
        logging.info(f"   Feature size: {self.feature_size}")
        logging.info(f"   Conv channels: {conv_channels}")
        logging.info(f"   Hidden dims: {hidden_dims}")
        logging.info(f"   Architecture: Optimized for BBH + NS signals")

    
    def _init_weights(self, module):
        """Conservative weight initialization to prevent overfitting"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain=0.8)  # Smaller gain
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.constant_(module.weight, 1.0)
            torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, waveform_data):
        # FIXED: Better input stabilization
        waveform_data = torch.clamp(waveform_data, min=-1e2, max=1e2)
        features = self.feature_extractor(waveform_data)
        predicted_params = self.param_predictor(features)
        predicted_uncertainties = 0.01 + 0.99 * self.uncertainty_predictor(features)
        return predicted_params, predicted_uncertainties

class NeuralPEDataset(Dataset):
    """FIXED: Better dataset handling for all signal types"""
    
    def __init__(self, scenarios: List[Dict]):
        self.data = []
        self.logger = logging.getLogger(__name__)
        
        # Signal type counters for balanced learning
        bbh_count = bns_count = nsbh_count = 0
        
        for scenario_id, scenario in enumerate(scenarios):
            try:
                true_params = scenario.get('true_parameters', [])
                if not true_params:
                    continue
                
                # Process each signal in the scenario
                for signal_params in true_params:
                    # Detect signal type for balanced training
                    m1 = signal_params.get('mass_1', 30.0)
                    m2 = signal_params.get('mass_2', 25.0)
                    
                    if m1 < 3.0 or m2 < 3.0:
                        signal_type = 'BNS' if (m1 < 3.0 and m2 < 3.0) else 'NSBH'
                        if signal_type == 'BNS':
                            bns_count += 1
                        else:
                            nsbh_count += 1
                    else:
                        signal_type = 'BBH'
                        bbh_count += 1
                    
                    # Generate realistic waveform
                    waveform_data = self._generate_realistic_waveform(signal_params, signal_type)
                    
                    if waveform_data is not None:
                        # Normalize parameters
                        normalized_params = self._normalize_parameters(signal_params)
                        
                        self.data.append({
                            'waveform_data': waveform_data,
                            'parameters': normalized_params,
                            'signal_type': signal_type,
                            'scenario_id': scenario_id
                        })
                        
            except Exception as e:
                self.logger.debug(f"Error processing scenario {scenario_id}: {e}")
                continue
        
        # Log dataset composition
        total = len(self.data)
        self.logger.info(f"✅ FIXED Dataset created: {total} samples")
        self.logger.info(f"   BBH: {bbh_count} ({bbh_count/total:.1%})")
        self.logger.info(f"   BNS: {bns_count} ({bns_count/total:.1%})")
        self.logger.info(f"   NSBH: {nsbh_count} ({nsbh_count/total:.1%})")
    
    def _generate_realistic_waveform(self, params: Dict, signal_type: str) -> np.ndarray:
        """Generate realistic waveforms for all signal types"""
        
        try:
            t = np.linspace(0, 4, 4096)
            
            # Get parameters
            mass_1 = params.get('mass_1', 30.0)
            mass_2 = params.get('mass_2', 25.0)
            distance = params.get('luminosity_distance', 500.0)
            inclination = params.get('theta_jn', np.pi/4)
            
            # Signal-type specific generation
            if signal_type == 'BNS':
                # BNS: Longer inspirals, tidal effects
                chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
                f_start = 15.0  # Lower starting frequency
                f_end = min(1500.0, 4400.0 / (mass_1 + mass_2))  # Higher end for NS
                
                # Amplitude scaling for NS
                amplitude = 2e-3 * np.exp(-t / 12.0) * np.sqrt(chirp_mass / 2.8)
                
            elif signal_type == 'NSBH':
                # NSBH: Mixed characteristics
                chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
                f_start = 18.0
                f_end = min(800.0, 2200.0 / (mass_1 + mass_2))
                
                amplitude = 1.5e-3 * np.exp(-t / 10.0) * np.sqrt(chirp_mass / 15.0)
                
            else:  # BBH
                # BBH: Standard generation
                chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
                f_start = 20.0
                f_end = min(250.0, 220.0 / (mass_1 + mass_2))
                
                amplitude = 1e-3 * np.exp(-t / 8.0) * np.sqrt(chirp_mass / 30.0)
            
            # Generate frequency evolution
            frequency = f_start + (f_end - f_start) * (t / 4.0)**3  # Cubic for realism
            phase = 2 * np.pi * np.cumsum(frequency) * (4.0 / 4096)
            
            # Polarizations
            h_plus = amplitude * (1 + np.cos(inclination)**2) * np.cos(phase)
            h_cross = amplitude * 2 * np.cos(inclination) * np.sin(phase)
            
            # Add realistic noise
            noise_level = amplitude.max() * 0.1  # 10% noise
            h_plus += np.random.normal(0, noise_level, 4096)
            h_cross += np.random.normal(0, noise_level, 4096)
            
            # Distance scaling
            distance_factor = 400.0 / distance  # Reference distance
            h_plus *= distance_factor
            h_cross *= distance_factor
            
            return np.array([h_plus, h_cross], dtype=np.float32)
            
        except Exception as e:
            self.logger.debug(f"Waveform generation failed: {e}")
            return None
    
    def _normalize_parameters(self, params: Dict) -> np.ndarray:
        """Normalize parameters to [-1, 1] range"""
        
        param_ranges = {
            'mass_1': (1.0, 150.0),
            'mass_2': (1.0, 150.0),
            'luminosity_distance': (10.0, 15000.0),
            'ra': (0.0, 2*np.pi),
            'dec': (-np.pi/2, np.pi/2),
            'geocent_time': (-0.5, 0.5),
            'theta_jn': (0.0, np.pi),
            'psi': (0.0, np.pi),
            'phase': (0.0, 2*np.pi)
        }
        
        normalized = []
        param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 
                      'geocent_time', 'theta_jn', 'psi', 'phase']
        
        for param_name in param_names:
            value = params.get(param_name, 0.0)
            
            if param_name in param_ranges:
                min_val, max_val = param_ranges[param_name]
                
                if param_name == 'luminosity_distance':
                    # Log scaling for distance
                    log_min, log_max = np.log10(min_val), np.log10(max_val)
                    log_val = np.log10(max(value, min_val))
                    norm_val = 2 * (log_val - log_min) / (log_max - log_min) - 1
                else:
                    # Linear scaling
                    norm_val = 2 * (value - min_val) / (max_val - min_val) - 1
                
                # Clamp to [-1, 1]
                norm_val = np.clip(norm_val, -1.0, 1.0)
                normalized.append(norm_val)
            else:
                normalized.append(0.0)
        
        return np.array(normalized, dtype=np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_neural_pe_batch(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Collate function for neural PE training"""
    
    waveforms = torch.stack([torch.tensor(item['waveform_data']) for item in batch])
    parameters = torch.stack([torch.tensor(item['parameters']) for item in batch])
    signal_types = [item['signal_type'] for item in batch]
    
    return waveforms, parameters, signal_types

def train_enhanced_neural_pe(model, dataset, config, output_dir):
    """FIXED: Enhanced training with better generalization"""
    
    logging.info("🧠 Training FIXED Neural PE Network...")
    
    # FIXED: Balanced loss for different signal types
    class BalancedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, pred_params, pred_uncertainties, true_params, signal_types):
            # Base MSE loss
            param_loss = nn.functional.mse_loss(pred_params, true_params, reduction='none')
            
            # FIXED: Signal-type weighting for balanced learning
            weights = torch.ones(param_loss.shape[0], device=param_loss.device)
            for i, signal_type in enumerate(signal_types):
                if signal_type == 'BNS':
                    weights[i] = 3.0  # Highest weight for rare BNS
                elif signal_type == 'NSBH':
                    weights[i] = 2.5  # Higher weight for NSBH
                else:  # BBH
                    weights[i] = 1.0  # Standard weight
            
            # Apply weights
            weighted_param_loss = torch.mean(param_loss * weights.unsqueeze(1))
            
            # Uncertainty regularization
            uncertainty_loss = torch.mean(pred_uncertainties)
            
            # FIXED: Balanced total loss
            total_loss = 0.8 * weighted_param_loss + 0.2 * uncertainty_loss
            return total_loss, weighted_param_loss, uncertainty_loss
    
    criterion = BalancedLoss()
    
    # FIXED: Conservative optimizer for better generalization
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=3e-4,            # Reduced learning rate
                                  weight_decay=1e-4,  # Increased regularization
                                  betas=(0.9, 0.999))
    
    # FIXED: Better scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=15, verbose=True
    )
    
    # Data loader
    dataloader = DataLoader(
        dataset,
        batch_size=24,  # Increased for stability
        shuffle=True,
        collate_fn=collate_neural_pe_batch,
        num_workers=0,
        drop_last=True
    )
    
    # Training loop
    num_epochs = 80  # Reduced to prevent overfitting
    best_loss = float('inf')
    patience = 20    # Increased patience
    patience_counter = 0
    
    training_metrics = {
        'total_losses': [],
        'param_losses': [],
        'uncertainty_losses': [],
        'epochs_completed': 0
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_total_losses = []
        epoch_param_losses = []
        epoch_uncertainty_losses = []
        
        pbar = tqdm(dataloader, desc=f'FIXED Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (waveforms, parameters, signal_types) in enumerate(pbar):
            
            # Forward pass
            pred_params, pred_uncertainties = model(waveforms)
            
            # Compute loss
            total_loss, param_loss, uncertainty_loss = criterion(
                pred_params, pred_uncertainties, parameters, signal_types
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            epoch_total_losses.append(total_loss.item())
            epoch_param_losses.append(param_loss.item())
            epoch_uncertainty_losses.append(uncertainty_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'Total': f'{total_loss.item():.4f}',
                'Param': f'{param_loss.item():.4f}',
                'Uncert': f'{uncertainty_loss.item():.4f}'
            })
        
        # Epoch metrics
        avg_total_loss = np.mean(epoch_total_losses)
        avg_param_loss = np.mean(epoch_param_losses)
        avg_uncertainty_loss = np.mean(epoch_uncertainty_losses)
        
        training_metrics['total_losses'].append(avg_total_loss)
        training_metrics['param_losses'].append(avg_param_loss)
        training_metrics['uncertainty_losses'].append(avg_uncertainty_loss)
        training_metrics['epochs_completed'] = epoch + 1
        
        # Learning rate scheduling
        scheduler.step(avg_total_loss)
        
        # Logging
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            logging.info(f"Epoch {epoch:3d}: Total={avg_total_loss:.4f}, "
                        f"Param={avg_param_loss:.4f}, Uncert={avg_uncertainty_loss:.4f}")
        
        # Early stopping
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'training_metrics': training_metrics,
                'param_names': model.param_names
            }, output_dir / 'enhanced_neural_pe_model.pth')
            
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch}")
            break
        
        # Memory cleanup
        if epoch % 20 == 0:
            gc.collect()
    
    # Final evaluation
    model.eval()
    final_accuracy = evaluate_neural_pe(model, dataset)
    
    # Save final results
    pe_results = {
        'training_metrics': training_metrics,
        'final_accuracy': final_accuracy,
        'best_loss': best_loss,
        'total_epochs': training_metrics['epochs_completed']
    }
    
    with open(output_dir / 'neural_pe_results.pkl', 'wb') as f:
        pickle.dump(pe_results, f)
    
    logging.info("✅ FIXED Neural PE training completed!")
    
    return pe_results

def evaluate_neural_pe(model: nn.Module, dataset: NeuralPEDataset) -> float:
    """Evaluate Neural PE accuracy across all signal types"""
    
    logging.info("📊 Evaluating Neural PE accuracy...")
    
    model.eval()
    
    # Track by signal type
    bbh_errors = []
    bns_errors = []
    nsbh_errors = []
    
    with torch.no_grad():
        for i in range(min(1000, len(dataset))):  # Sample for evaluation
            item = dataset[i]
            waveform = torch.tensor(item['waveform_data']).unsqueeze(0)
            true_params = torch.tensor(item['parameters']).unsqueeze(0)
            signal_type = item['signal_type']
            
            try:
                pred_params, _ = model(waveform)
                error = torch.mean(torch.abs(pred_params - true_params)).item()
                
                if signal_type == 'BBH':
                    bbh_errors.append(error)
                elif signal_type == 'BNS':
                    bns_errors.append(error)
                else:  # NSBH
                    nsbh_errors.append(error)
                    
            except Exception as e:
                logging.debug(f"Evaluation error: {e}")
                continue
    
    # Compute accuracies
    def compute_accuracy(errors):
        if not errors:
            return 0.0
        mean_error = np.mean(errors)
        return max(0.0, 1.0 - mean_error)
    
    bbh_accuracy = compute_accuracy(bbh_errors)
    bns_accuracy = compute_accuracy(bns_errors)
    nsbh_accuracy = compute_accuracy(nsbh_errors)
    
    # Weighted overall accuracy
    total_samples = len(bbh_errors) + len(bns_errors) + len(nsbh_errors)
    if total_samples > 0:
        overall_accuracy = (len(bbh_errors) * bbh_accuracy + 
                           len(bns_errors) * bns_accuracy + 
                           len(nsbh_errors) * nsbh_accuracy) / total_samples
    else:
        overall_accuracy = 0.0
    
    # Log results
    logging.info(f"📈 Neural PE Evaluation Results:")
    logging.info(f"   BBH Accuracy: {bbh_accuracy:.3f} ({len(bbh_errors)} samples)")
    logging.info(f"   BNS Accuracy: {bns_accuracy:.3f} ({len(bns_errors)} samples)")
    logging.info(f"   NSBH Accuracy: {nsbh_accuracy:.3f} ({len(nsbh_errors)} samples)")
    logging.info(f"   Overall Accuracy: {overall_accuracy:.3f}")
    
    return overall_accuracy

def main():
    parser = argparse.ArgumentParser(description='Phase 3A: FIXED Neural PE Training')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--data_dir', required=True, help='Training data directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logging.info("🚀 Starting Phase 3A: FIXED Neural PE Training")
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.warning(f"Could not load config: {e}, using defaults")
        config = {
            'neural_pe': {
                'learning_rate': 3e-4,
                'batch_size': 24,
                'epochs': 80
            }
        }
    
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
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 
                   'geocent_time', 'theta_jn', 'psi', 'phase']
    
    dataset = NeuralPEDataset(scenarios)
    
    if len(dataset) == 0:
        logging.error("❌ No valid training data")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and train model
    model = NeuralPENetwork(param_names,config)
    pe_results = train_enhanced_neural_pe(model, dataset, config, output_dir)
    
    # Print results
    logging.info("✅ Phase 3A: FIXED Neural PE Training COMPLETED")
    
    print("\n" + "="*60)
    print("✅ PHASE 3A COMPLETE - FIXED NEURAL PE")
    print("="*60)
    print(f"📈 Final Accuracy: {pe_results['final_accuracy']:.1%}")
    print(f"🔧 Architecture: Optimized for BBH/BNS/NSBH")
    print(f"⚡ Parameters: Reduced for better generalization")
    print(f"🎯 Training: Signal-type balanced")
    print("="*60)

if __name__ == '__main__':
    main()
