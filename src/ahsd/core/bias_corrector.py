import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
import pickle
from pathlib import Path

class BiasEstimator(nn.Module):
    """Neural network to estimate residual bias after hierarchical subtraction"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        current_dim = input_dim + 5  # Add context features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15)
            ])
            current_dim = hidden_dim
            
        # Output bias correction for each parameter
        layers.extend([
            nn.Linear(current_dim, input_dim),
            nn.Tanh()  # Bound corrections to reasonable range
        ])
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, param_estimates: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """Predict bias corrections based on parameter estimates and extraction context"""
        
        # Handle batch normalization for single samples
        if len(param_estimates) == 1:
            param_estimates = param_estimates.repeat(2, 1)
            context_features = context_features.repeat(2, 1)
            
            combined_input = torch.cat([param_estimates, context_features], dim=1)
            output = self.network(combined_input)[0:1]
        else:
            combined_input = torch.cat([param_estimates, context_features], dim=1)
            output = self.network(combined_input)
            
        return output

class BiasCorrector:
    """Correct hierarchical biases in extracted parameters using real data patterns"""
    
    def __init__(self, param_names: List[str]):
        self.param_names = param_names
        self.bias_estimator = BiasEstimator(
            input_dim=len(param_names),
            hidden_dims=[256, 128, 64]
        )
        self.logger = logging.getLogger(__name__)
        
        # Bias correction statistics from real data
        self.bias_history = []
        self.correction_stats = {}
        self.is_trained = False
        
    def correct_hierarchical_biases(self, extracted_signals: List[Dict]) -> List[Dict]:
        """Apply learned bias corrections to extracted parameters"""
        
        if not extracted_signals:
            return []
            
        corrected_signals = []
        
        try:
            for i, signal in enumerate(extracted_signals):
                # Prepare input for bias estimator
                param_tensor, context_tensor = self._prepare_bias_correction_input(
                    signal, i, extracted_signals
                )
                
                if param_tensor is not None and self.is_trained:
                    # Predict bias correction
                    with torch.no_grad():
                        bias_correction = self.bias_estimator(param_tensor, context_tensor)
                        
                    # Apply correction to posterior summary
                    corrected_summary = self._apply_bias_correction(
                        signal['posterior_summary'], 
                        bias_correction.squeeze().numpy(),
                        signal.get('signal_quality', 1.0)
                    )
                else:
                    # No correction available
                    corrected_summary = signal['posterior_summary'].copy()
                    bias_correction = np.zeros(len(self.param_names))
                
                corrected_signal = signal.copy()
                corrected_signal['posterior_summary'] = corrected_summary
                corrected_signal['bias_correction'] = bias_correction if isinstance(bias_correction, np.ndarray) else np.array(bias_correction)
                
                corrected_signals.append(corrected_signal)
                
        except Exception as e:
            self.logger.error(f"Bias correction failed: {e}")
            # Return original signals if correction fails
            return extracted_signals
        
        return corrected_signals
    
    def _prepare_bias_correction_input(self, signal: Dict, position: int, all_signals: List[Dict]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare input tensors for bias estimation"""
        
        try:
            posterior_summary = signal.get('posterior_summary', {})
            
            # Extract parameter medians
            param_values = []
            for param_name in self.param_names:
                if param_name in posterior_summary:
                    median_value = posterior_summary[param_name].get('median', 0.0)
                    param_values.append(float(median_value))
                else:
                    # Use default values for missing parameters
                    defaults = {
                        'mass_1': 30.0, 'mass_2': 30.0, 'luminosity_distance': 500.0,
                        'geocent_time': 0.0, 'ra': 1.0, 'dec': 0.0
                    }
                    param_values.append(defaults.get(param_name, 0.0))
            
            param_tensor = torch.tensor(param_values, dtype=torch.float32).unsqueeze(0)
            
            # Prepare context features
            context_features = [
                position / max(len(all_signals), 1),  # Normalized extraction position
                len(all_signals),  # Total number of overlapping signals
                signal.get('signal_quality', 0.5),  # Signal quality score
                self._compute_snr_estimate(posterior_summary),  # SNR estimate
                self._compute_mass_ratio(posterior_summary)  # Mass ratio
            ]
            
            context_tensor = torch.tensor(context_features, dtype=torch.float32).unsqueeze(0)
            
            return param_tensor, context_tensor
            
        except Exception as e:
            self.logger.debug(f"Failed to prepare bias correction input: {e}")
            return None, None
    
    def _compute_snr_estimate(self, posterior_summary: Dict) -> float:
        """Estimate SNR from posterior summary"""
        
        if 'network_snr' in posterior_summary:
            return float(posterior_summary['network_snr'].get('median', 10.0))
        
        # Estimate from distance and masses
        try:
            distance = posterior_summary.get('luminosity_distance', {}).get('median', 500.0)
            m1 = posterior_summary.get('mass_1', {}).get('median', 30.0)
            m2 = posterior_summary.get('mass_2', {}).get('median', 30.0)
            
            # Rough SNR scaling
            chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
            estimated_snr = 1000.0 * (chirp_mass / 30.0)**(5/6) / (distance / 400.0)
            
            return max(float(estimated_snr), 8.0)
            
        except:
            return 10.0  # Default
    
    def _compute_mass_ratio(self, posterior_summary: Dict) -> float:
        """Compute mass ratio from posterior"""
        
        try:
            m1 = posterior_summary.get('mass_1', {}).get('median', 30.0)
            m2 = posterior_summary.get('mass_2', {}).get('median', 30.0)
            
            if m1 > 0 and m2 > 0:
                return min(m1, m2) / max(m1, m2)
            else:
                return 1.0
                
        except:
            return 1.0
    
    def _apply_bias_correction(self, posterior_summary: Dict, 
                             bias_correction: np.ndarray, 
                             signal_quality: float) -> Dict:
        """Apply bias correction to posterior summary statistics"""
        
        corrected_summary = {}
        
        # Scale correction by signal quality
        scaled_correction = bias_correction * signal_quality * 0.5  # Conservative scaling
        
        for i, param_name in enumerate(self.param_names):
            if param_name in posterior_summary and i < len(scaled_correction):
                original = posterior_summary[param_name].copy()
                correction = float(scaled_correction[i])
                
                # Apply correction to median
                corrected_summary[param_name] = original.copy()
                
                # Get parameter-specific scaling
                param_scale = self._get_parameter_scale(param_name, original.get('median', 0.0))
                scaled_param_correction = correction * param_scale
                
                corrected_summary[param_name]['median'] += scaled_param_correction
                
                # Adjust quantiles if available
                if 'quantiles' in original:
                    corrected_quantiles = np.array(original['quantiles']) + scaled_param_correction
                    corrected_summary[param_name]['quantiles'] = corrected_quantiles.tolist()
                
                # Adjust mean if available
                if 'mean' in original:
                    corrected_summary[param_name]['mean'] += scaled_param_correction
                    
            else:
                # Parameter not found, copy original
                if param_name in posterior_summary:
                    corrected_summary[param_name] = posterior_summary[param_name].copy()
        
        # Copy any additional parameters not in the correction list
        for param_name, param_data in posterior_summary.items():
            if param_name not in corrected_summary:
                corrected_summary[param_name] = param_data.copy()
                
        return corrected_summary
    
    def _get_parameter_scale(self, param_name: str, current_value: float) -> float:
        """Get parameter-specific scaling factor for corrections"""
        
        scales = {
            'mass_1': min(abs(current_value) * 0.1, 10.0),
            'mass_2': min(abs(current_value) * 0.1, 10.0),
            'luminosity_distance': min(abs(current_value) * 0.2, 200.0),
            'geocent_time': 0.01,
            'ra': 0.1,
            'dec': 0.1,
            'theta_jn': 0.1,
            'psi': 0.1,
            'phase': 0.2,
            'a_1': 0.1,
            'a_2': 0.1,
            'tilt_1': 0.1,
            'tilt_2': 0.1,
            'phi_12': 0.2,
            'phi_jl': 0.2
        }
        
        return scales.get(param_name, 0.1)
    
    def train_bias_estimator(self, training_scenarios: List[Dict]) -> None:
        """Train the bias estimation network using real data patterns"""
        
        if not training_scenarios:
            self.logger.warning("No training data provided for bias estimator")
            return
        
        self.logger.info(f"Training bias estimator on {len(training_scenarios)} scenarios")
        
        try:
            # Prepare training data
            inputs, targets, weights = self._prepare_training_data(training_scenarios)
            
            if len(inputs) == 0:
                self.logger.warning("No valid training data for bias estimator")
                return
            
            # Convert to tensors
            input_params = torch.tensor(inputs['params'], dtype=torch.float32)
            input_context = torch.tensor(inputs['context'], dtype=torch.float32)
            target_corrections = torch.tensor(targets, dtype=torch.float32)
            sample_weights = torch.tensor(weights, dtype=torch.float32)
            
            # Training setup
            optimizer = torch.optim.AdamW(self.bias_estimator.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
            criterion = nn.MSELoss(reduction='none')
            
            # Training loop
            n_epochs = min(1000, len(training_scenarios) * 2)
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                
                predictions = self.bias_estimator(input_params, input_context)
                loss_per_sample = criterion(predictions, target_corrections).mean(dim=1)
                weighted_loss = (loss_per_sample * sample_weights).mean()
                
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bias_estimator.parameters(), 1.0)
                optimizer.step()
                scheduler.step(weighted_loss)
                
                # Early stopping
                if weighted_loss.item() < best_loss:
                    best_loss = weighted_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter > 100:
                    break
                
                if epoch % 100 == 0:
                    self.logger.info(f"Bias correction training epoch {epoch}, loss: {weighted_loss.item():.6f}")
            
            self.is_trained = True
            self.logger.info("Bias estimator training completed")
            
        except Exception as e:
            self.logger.error(f"Bias estimator training failed: {e}")
            self.is_trained = False
    
    def _prepare_training_data(self, training_scenarios: List[Dict]) -> Tuple[Dict, List, List]:
        """Prepare training data for bias estimator"""
        
        param_inputs = []
        context_inputs = []
        bias_targets = []
        sample_weights = []
        
        for scenario in training_scenarios:
            try:
                true_params = scenario.get('true_parameters', {})
                extracted_params = scenario.get('extracted_parameters', {})
                
                if not true_params or not extracted_params:
                    continue
                
                # Compute parameter estimates and biases
                param_estimates = []
                bias_corrections = []
                
                for param_name in self.param_names:
                    if param_name in true_params and param_name in extracted_params:
                        true_val = float(true_params[param_name])
                        extracted_dict = extracted_params[param_name]
                        
                        if isinstance(extracted_dict, dict):
                            estimated_val = float(extracted_dict.get('median', 0.0))
                            std_val = float(extracted_dict.get('std', 1.0))
                        else:
                            estimated_val = float(extracted_dict)
                            std_val = 1.0
                        
                        param_estimates.append(estimated_val)
                        
                        # Compute bias correction needed (normalized)
                        if std_val > 0:
                            bias_correction = (true_val - estimated_val) / std_val
                        else:
                            bias_correction = true_val - estimated_val
                            
                        bias_corrections.append(bias_correction)
                    else:
                        # Missing parameter
                        param_estimates.append(0.0)
                        bias_corrections.append(0.0)
                
                # Context features
                context_features = [
                    scenario.get('extraction_position', 0) / max(scenario.get('total_signals', 1), 1),
                    scenario.get('total_signals', 1),
                    scenario.get('signal_quality', 0.5),
                    scenario.get('estimated_snr', 10.0) / 20.0,  # Normalized
                    scenario.get('mass_ratio', 1.0)
                ]
                
                # Weight by signal quality and SNR
                weight = scenario.get('signal_quality', 0.5) * min(scenario.get('estimated_snr', 10.0) / 15.0, 1.0)
                
                param_inputs.append(param_estimates)
                context_inputs.append(context_features)
                bias_targets.append(bias_corrections)
                sample_weights.append(weight)
                
            except Exception as e:
                self.logger.debug(f"Skipping training scenario due to error: {e}")
                continue
        
        inputs = {
            'params': param_inputs,
            'context': context_inputs
        }
        
        return inputs, bias_targets, sample_weights
    
    def save_model(self, filepath: str) -> None:
        """Save trained bias correction model"""
        
        try:
            model_data = {
                'state_dict': self.bias_estimator.state_dict(),
                'param_names': self.param_names,
                'is_trained': self.is_trained,
                'correction_stats': self.correction_stats
            }
            
            torch.save(model_data, filepath)
            self.logger.info(f"Bias correction model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save bias correction model: {e}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained bias correction model"""
        
        try:
            model_data = torch.load(filepath, map_location='cpu')
            
            self.bias_estimator.load_state_dict(model_data['state_dict'])
            self.param_names = model_data['param_names']
            self.is_trained = model_data.get('is_trained', False)
            self.correction_stats = model_data.get('correction_stats', {})
            
            self.logger.info(f"Bias correction model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load bias correction model: {e}")
            self.is_trained = False
