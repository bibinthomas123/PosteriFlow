#!/usr/bin/env python3
"""
Phase 3: Train and test adaptive subtractor components
"""

import numpy as np
import torch
import torch.nn as nn
import argparse
import pickle
from pathlib import Path
import logging
import wandb
from tqdm import tqdm
import yaml
from typing import List, Dict, Tuple

from ahsd.utils.config import AHSDConfig
from ahsd.models.neural_pe import NeuralPosteriorEstimator
from ahsd.core.adaptive_subtractor import AdaptiveSubtractor, UncertaintyAwareSubtractor
from ahsd.core.priority_net import PriorityNet

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('phase3_adaptive_subtractor.log'),
            logging.StreamHandler()
        ]
    )

class NeuralPETrainer:
    """Trainer for Neural Posterior Estimator."""
    
    def __init__(self, neural_pe: NeuralPosteriorEstimator, learning_rate: float = 1e-3):
        self.neural_pe = neural_pe
        self.optimizer = torch.optim.Adam(neural_pe.flow.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
        self.logger = logging.getLogger(__name__)
    
    def train_on_scenarios(self, training_scenarios: List[Dict], n_epochs: int = 100) -> Dict:
        """Train neural PE on training scenarios."""
        
        self.logger.info(f"Training Neural PE on {len(training_scenarios)} scenarios for {n_epochs} epochs")
        
        training_losses = []
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            for scenario in tqdm(training_scenarios, desc=f'Epoch {epoch+1}/{n_epochs}'):
                try:
                    # Prepare training data from scenario
                    data_contexts, parameter_samples = self._prepare_training_data(scenario)
                    
                    if len(data_contexts) == 0:
                        continue
                    
                    for context, samples in zip(data_contexts, parameter_samples):
                        # Forward pass through flow
                        log_prob = self.neural_pe.flow.log_prob(samples, context=context)
                        loss = -torch.mean(log_prob)
                        
                        # Handle NaN/inf losses
                        if torch.isnan(loss) or torch.isinf(loss):
                            continue
                        
                        # Backward pass
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.neural_pe.flow.parameters(), 1.0)
                        self.optimizer.step()
                        
                        epoch_losses.append(loss.item())
                        
                except Exception as e:
                    self.logger.debug(f"Training error for scenario {scenario.get('scenario_id', 'unknown')}: {e}")
                    continue
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            training_losses.append(avg_loss)
            
            self.scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
        
        return {'training_losses': training_losses}
    
    def _prepare_training_data(self, scenario: Dict) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Prepare training data from scenario."""
        
        data_contexts = []
        parameter_samples = []
        
        try:
            # Extract data features
            injected_data = scenario.get('injected_data', {})
            true_parameters = scenario.get('true_parameters', [])
            
            for i, params in enumerate(true_parameters):
                # Prepare data context vector
                context = self._extract_data_features(injected_data)
                
                # Prepare parameter tensor
                param_vector = self._params_to_tensor(params)
                
                if context is not None and param_vector is not None:
                    data_contexts.append(context)
                    parameter_samples.append(param_vector.unsqueeze(0))
            
        except Exception as e:
            self.logger.debug(f"Data preparation error: {e}")
        
        return data_contexts, parameter_samples
    
    def _extract_data_features(self, data: Dict) -> torch.Tensor:
        """Extract features from strain data for conditioning."""
        
        try:
            features = []
            
            for det_name, strain in data.items():
                if hasattr(strain, '__len__') and len(strain) > 0:
                    strain_array = np.array(strain)
                    
                    # Basic statistical features
                    features.extend([
                        np.mean(strain_array),
                        np.std(strain_array),
                        np.max(np.abs(strain_array)),
                        np.median(strain_array)
                    ])
                    
                    # Spectral features
                    fft = np.fft.fft(strain_array)
                    psd = np.abs(fft)**2
                    
                    # Power in different frequency bands
                    freqs = np.fft.fftfreq(len(strain_array))
                    low_freq_power = np.sum(psd[np.abs(freqs) < 0.01])
                    mid_freq_power = np.sum(psd[(np.abs(freqs) >= 0.01) & (np.abs(freqs) < 0.1)])
                    high_freq_power = np.sum(psd[np.abs(freqs) >= 0.1])
                    
                    features.extend([low_freq_power, mid_freq_power, high_freq_power])
                    
                    # Downsample PSD
                    psd_downsampled = psd[::max(1, len(psd)//50)][:50]
                    features.extend(psd_downsampled.tolist())
            
            # Pad or truncate to fixed size
            target_size = 300
            if len(features) > target_size:
                features = features[:target_size]
            else:
                features.extend([0.0] * (target_size - len(features)))
            
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
        except Exception as e:
            self.logger.debug(f"Feature extraction error: {e}")
            return None
    
    def _params_to_tensor(self, params: Dict) -> torch.Tensor:
        """Convert parameter dict to tensor."""
        
        try:
            param_order = [
                'mass_1', 'mass_2', 'luminosity_distance', 'theta_jn', 'psi',
                'phase', 'geocent_time', 'ra', 'dec', 'a_1', 'a_2',
                'tilt_1', 'tilt_2', 'phi_12', 'phi_jl'
            ]
            
            param_values = []
            for param_name in param_order:
                if param_name in params:
                    param_values.append(float(params[param_name]))
                else:
                    # Default values
                    defaults = {
                        'mass_1': 30.0, 'mass_2': 30.0, 'luminosity_distance': 500.0,
                        'theta_jn': 0.0, 'psi': 0.0, 'phase': 0.0, 'geocent_time': 0.0,
                        'ra': 0.0, 'dec': 0.0, 'a_1': 0.0, 'a_2': 0.0,
                        'tilt_1': 0.0, 'tilt_2': 0.0, 'phi_12': 0.0, 'phi_jl': 0.0
                    }
                    param_values.append(defaults.get(param_name, 0.0))
            
            return torch.tensor(param_values, dtype=torch.float32)
            
        except Exception as e:
            self.logger.debug(f"Parameter tensor conversion error: {e}")
            return None

def train_adaptive_subtractor(config: AHSDConfig, training_scenarios: List[Dict], args) -> AdaptiveSubtractor:
    """Train the adaptive subtractor components."""
    
    logging.info("Training Adaptive Subtractor...")
    
    # Initialize neural PE
    param_names = [
        'mass_1', 'mass_2', 'luminosity_distance', 'theta_jn', 'psi',
        'phase', 'geocent_time', 'ra', 'dec', 'a_1', 'a_2',
        'tilt_1', 'tilt_2', 'phi_12', 'phi_jl'
    ]
    
    neural_pe = NeuralPosteriorEstimator(param_names, config.adaptive_subtractor.neural_pe)
    
    # Train neural PE
    pe_trainer = NeuralPETrainer(neural_pe, learning_rate=1e-4)
    training_results = pe_trainer.train_on_scenarios(training_scenarios[:100], n_epochs=50)  # Subset for training
    
    # Initialize uncertainty-aware subtractor
    waveform_generator = setup_waveform_generator(config)
    uncertainty_subtractor = UncertaintyAwareSubtractor(waveform_generator)
    
    # Combine into adaptive subtractor
    adaptive_subtractor = AdaptiveSubtractor(neural_pe, uncertainty_subtractor)
    
    logging.info("Adaptive subtractor training completed")
    
    return adaptive_subtractor

def setup_waveform_generator(config: AHSDConfig):
    """Setup bilby waveform generator."""
    
    import bilby
    
    return bilby.gw.WaveformGenerator(
        duration=config.waveform.duration,
        sampling_frequency=config.detectors[0].sampling_rate,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant=config.waveform.approximant,
            reference_frequency=config.waveform.f_ref,
        )
    )

def test_adaptive_subtractor(adaptive_subtractor: AdaptiveSubtractor, 
                           test_scenarios: List[Dict],
                           priority_net: PriorityNet = None) -> Dict:
    """Test adaptive subtractor performance."""
    
    logging.info(f"Testing adaptive subtractor on {len(test_scenarios)} scenarios...")
    
    test_results = []
    
    for scenario in tqdm(test_scenarios, desc="Testing"):
        try:
            # Prepare data and detections
            data = scenario['injected_data']
            true_parameters = scenario['true_parameters']
            
            # Create detection candidates
            detections = create_detection_candidates(true_parameters)
            
            # Get priority ranking
            if priority_net is not None:
                priority_ranking = priority_net.rank_detections(detections)
            else:
                # Fallback to SNR-based ranking
                priority_ranking = list(range(len(detections)))
            
            # Test subtraction for each signal
            scenario_results = []
            current_data = {det: np.array(strain) for det, strain in data.items()}
            
            for rank, detection_idx in enumerate(priority_ranking):
                try:
                    # Extract and subtract
                    residual_data, extraction_result, subtraction_uncertainty = \
                        adaptive_subtractor.extract_and_subtract(current_data, detection_idx)
                    
                    # Evaluate extraction quality
                    if detection_idx < len(true_parameters):
                        true_params = true_parameters[detection_idx]
                        extraction_metrics = evaluate_extraction(
                            true_params, extraction_result, subtraction_uncertainty
                        )
                    else:
                        extraction_metrics = {'error': 'no_ground_truth'}
                    
                    scenario_results.append({
                        'detection_idx': detection_idx,
                        'rank': rank,
                        'extraction_result': extraction_result,
                        'metrics': extraction_metrics
                    })
                    
                    # Update data for next extraction
                    current_data = residual_data
                    
                except Exception as e:
                    logging.debug(f"Extraction failed for detection {detection_idx}: {e}")
                    continue
            
            test_results.append({
                'scenario_id': scenario['scenario_id'],
                'n_signals': scenario['n_signals'],
                'extractions': scenario_results,
                'data_type': scenario.get('data_type', 'simulated')
            })
            
        except Exception as e:
            logging.warning(f"Test failed for scenario {scenario.get('scenario_id', 'unknown')}: {e}")
            continue
    
    # Compute summary metrics
    summary_metrics = compute_test_metrics(test_results)
    
    return {
        'detailed_results': test_results,
        'summary_metrics': summary_metrics
    }

def create_detection_candidates(true_parameters: List[Dict]) -> List[Dict]:
    """Create realistic detection candidates."""
    
    candidates = []
    
    for i, true_params in enumerate(true_parameters):
        candidate = true_params.copy()
        
        # Add detection uncertainties
        for param in ['mass_1', 'mass_2']:
            if param in candidate:
                true_val = candidate[param]
                candidate[param] = np.random.normal(true_val, true_val * 0.15)
        
        if 'luminosity_distance' in candidate:
            true_val = candidate['luminosity_distance']
            candidate['luminosity_distance'] = np.random.normal(true_val, true_val * 0.3)
        
        # Add SNR estimate
        candidate['network_snr'] = np.random.uniform(10, 20)
        candidate['detection_id'] = i
        
        candidates.append(candidate)
    
    return candidates

def evaluate_extraction(true_params: Dict, extraction_result: Dict, 
                       subtraction_uncertainty: Dict) -> Dict:
    """Evaluate extraction quality against ground truth."""
    
    try:
        posterior_summary = extraction_result.get('posterior_summary', {})
        
        # Parameter biases
        biases = {}
        for param_name in ['mass_1', 'mass_2', 'luminosity_distance']:
            if param_name in true_params and param_name in posterior_summary:
                true_val = true_params[param_name]
                est_val = posterior_summary[param_name]['median']
                std_val = posterior_summary[param_name]['std']
                
                if std_val > 0:
                    normalized_bias = abs(est_val - true_val) / std_val
                else:
                    normalized_bias = abs(est_val - true_val)
                
                biases[param_name] = normalized_bias
        
        # Overall quality metrics
        avg_bias = np.mean(list(biases.values())) if biases else float('inf')
        signal_quality = extraction_result.get('signal_quality', 0.0)
        
        # Subtraction quality
        if subtraction_uncertainty:
            avg_uncertainty = np.mean([np.mean(unc) for unc in subtraction_uncertainty.values()])
        else:
            avg_uncertainty = 0.0
        
        return {
            'parameter_biases': biases,
            'avg_parameter_bias': avg_bias,
            'signal_quality': signal_quality,
            'subtraction_uncertainty': avg_uncertainty,
            'extraction_success': avg_bias < 3.0  # Less than 3-sigma bias
        }
        
    except Exception as e:
        return {'error': str(e), 'extraction_success': False}

def compute_test_metrics(test_results: List[Dict]) -> Dict:
    """Compute summary metrics from test results."""
    
    all_biases = []
    success_rates = []
    signal_qualities = []
    
    for result in test_results:
        scenario_biases = []
        scenario_successes = []
        scenario_qualities = []
        
        for extraction in result['extractions']:
            metrics = extraction['metrics']
            
            if 'avg_parameter_bias' in metrics:
                if np.isfinite(metrics['avg_parameter_bias']):
                    scenario_biases.append(metrics['avg_parameter_bias'])
                    all_biases.append(metrics['avg_parameter_bias'])
            
            if 'extraction_success' in metrics:
                scenario_successes.append(metrics['extraction_success'])
            
            if 'signal_quality' in metrics:
                scenario_qualities.append(metrics['signal_quality'])
        
        if scenario_biases:
            success_rates.append(np.mean(scenario_successes))
            signal_qualities.extend(scenario_qualities)
    
    return {
        'mean_parameter_bias': np.mean(all_biases) if all_biases else float('inf'),
        'median_parameter_bias': np.median(all_biases) if all_biases else float('inf'),
        'extraction_success_rate': np.mean(success_rates) if success_rates else 0.0,
        'mean_signal_quality': np.mean(signal_qualities) if signal_qualities else 0.0,
        'n_test_scenarios': len(test_results),
        'n_total_extractions': sum(len(r['extractions']) for r in test_results)
    }

def main():
    parser = argparse.ArgumentParser(description='Train and test adaptive subtractor')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--data_dir', required=True, help='Training data directory')
    parser.add_argument('--priority_net', help='Trained PriorityNet model path')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--n_train', type=int, default=200, help='Number of training scenarios')
    parser.add_argument('--n_test', type=int, default=50, help='Number of test scenarios')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    config = AHSDConfig.from_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project="ahsd-adaptive-subtractor", config=vars(args))
    
    # Load training data
    data_dir = Path(args.data_dir)
    with open(data_dir / 'training_scenarios.pkl', 'rb') as f:
        all_scenarios = pickle.load(f)
    
    logging.info(f"Loaded {len(all_scenarios)} scenarios")
    
    # Split data
    np.random.shuffle(all_scenarios)
    train_scenarios = all_scenarios[:args.n_train]
    test_scenarios = all_scenarios[args.n_train:args.n_train + args.n_test]
    
    # Load PriorityNet if available
    priority_net = None
    if args.priority_net:
        try:
            priority_net = PriorityNet(config.priority_net)
            priority_net.load_state_dict(torch.load(args.priority_net))
            priority_net.eval()
            logging.info("Loaded pre-trained PriorityNet")
        except Exception as e:
            logging.warning(f"Failed to load PriorityNet: {e}")
    
    # Train adaptive subtractor
    adaptive_subtractor = train_adaptive_subtractor(config, train_scenarios, args)
    
    # Test adaptive subtractor
    test_results = test_adaptive_subtractor(adaptive_subtractor, test_scenarios, priority_net)
    
    # Save results
    with open(output_dir / 'adaptive_subtractor_results.pkl', 'wb') as f:
        pickle.dump(test_results, f)
    
    # Save model
    torch.save(adaptive_subtractor.neural_pe.flow.state_dict(), 
               output_dir / 'neural_pe_model.pth')
    
    # Log results
    metrics = test_results['summary_metrics']
    logging.info("Adaptive Subtractor Test Results:")
    logging.info(f"  Mean Parameter Bias: {metrics['mean_parameter_bias']:.3f}")
    logging.info(f"  Extraction Success Rate: {metrics['extraction_success_rate']:.3f}")
    logging.info(f"  Mean Signal Quality: {metrics['mean_signal_quality']:.3f}")
    
    if args.use_wandb:
        wandb.log(metrics)
    
    # Save summary
    with open(output_dir / 'subtractor_summary.yaml', 'w') as f:
        yaml.dump({
            'test_metrics': metrics,
            'config': vars(args)
        }, f, default_flow_style=False)
    
    logging.info(f"Phase 3 completed! Results saved to {output_dir}")

if __name__ == '__main__':
    main()
