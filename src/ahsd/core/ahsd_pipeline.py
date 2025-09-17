from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
import time
from .priority_net import PriorityNet
from .adaptive_subtractor import AdaptiveSubtractor  
from .bias_corrector import BiasCorrector
from ..utils.config import AHSDConfig

class AHSDPipeline:
    """Complete AHSD pipeline for real gravitational wave data"""
    
    def __init__(self, config: AHSDConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.priority_net = PriorityNet(config.priority_net)
        self.adaptive_subtractor = self._setup_adaptive_subtractor()
        self.bias_corrector = BiasCorrector(self._get_param_names())
        
        # Pipeline state
        self.extraction_history = []
        self.performance_metrics = {}
        self.ground_truth = None
        
    def _setup_adaptive_subtractor(self) -> AdaptiveSubtractor:
        """Setup adaptive subtractor with neural PE"""
        from ..models.neural_pe import NeuralPosteriorEstimator
        from .adaptive_subtractor import UncertaintyAwareSubtractor
        
        # Initialize neural PE
        param_names = self._get_param_names()
        neural_pe = NeuralPosteriorEstimator(param_names, self.config.adaptive_subtractor.neural_pe)
        
        # Initialize uncertainty-aware subtractor
        waveform_generator = self._setup_waveform_generator()
        uncertainty_subtractor = UncertaintyAwareSubtractor(waveform_generator)
        
        return AdaptiveSubtractor(neural_pe, uncertainty_subtractor)
    
    def _get_param_names(self) -> List[str]:
        """Get parameter names for analysis"""
        return [
            'mass_1', 'mass_2', 'luminosity_distance', 'theta_jn', 'psi',
            'phase', 'geocent_time', 'ra', 'dec', 'a_1', 'a_2',
            'tilt_1', 'tilt_2', 'phi_12', 'phi_jl'
        ]
    
    def _setup_waveform_generator(self):
        """Setup bilby waveform generator"""
        import bilby
        
        return bilby.gw.WaveformGenerator(
            duration=self.config.waveform.duration,
            sampling_frequency=self.config.detectors.sampling_rate,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(
                waveform_approximant=self.config.waveform.approximant,
                reference_frequency=self.config.waveform.f_ref,
            )
        )
    
    def decompose_overlapping_signals(self, 
                                    data: Dict, 
                                    initial_detections: List[Dict]) -> Dict:
        """Main AHSD algorithm"""
        
        start_time = time.time()
        self.logger.info(f"Starting AHSD decomposition of {len(initial_detections)} signals")
        
        # Reset pipeline state
        self.extraction_history = []
        
        # Phase 1: Intelligent Signal Prioritization
        self.logger.info("Phase 1: Signal prioritization")
        try:
            priority_ranking = self.priority_net.rank_detections(initial_detections)
            self.logger.info(f"Priority ranking: {priority_ranking}")
        except Exception as e:
            self.logger.error(f"Priority ranking failed: {e}")
            # Fallback to SNR-based ranking
            priority_ranking = self._fallback_ranking(initial_detections)
        
        # Phase 2: Adaptive Hierarchical Extraction
        self.logger.info("Phase 2: Hierarchical signal extraction")
        extracted_signals = []
        residual_data = {det: np.array(strain) for det, strain in data.items()}
        
        for rank, detection_idx in enumerate(priority_ranking):
            detection = initial_detections[detection_idx]
            
            self.logger.info(f"Extracting signal {detection_idx} (rank {rank+1}/{len(priority_ranking)})")
            
            try:
                # Extract signal with uncertainty quantification
                residual_data, extraction_result, subtraction_uncertainty = \
                    self.adaptive_subtractor.extract_and_subtract(residual_data, detection_idx)
                
                # Update model complexity for remaining signals
                remaining_signals = len(priority_ranking) - rank - 1
                self._update_model_complexity(residual_data, remaining_signals)
                
                # Record extraction metrics
                extraction_time = time.time()
                extraction_snr = self._compute_extraction_snr(extraction_result)
                
                extracted_signals.append(extraction_result)
                self.extraction_history.append({
                    'rank': rank,
                    'signal_id': detection_idx,
                    'extraction_snr': extraction_snr,
                    'extraction_time': extraction_time - start_time,
                    'residual_statistics': self._compute_residual_stats(residual_data),
                    'remaining_signals': remaining_signals
                })
                
                self.logger.info(f"Extracted signal {detection_idx} with SNR ~{extraction_snr:.1f}")
                
            except Exception as e:
                self.logger.error(f"Failed to extract signal {detection_idx}: {e}")
                # Continue with next signal
                continue
        
        # Phase 3: Hierarchical Bias Correction
        self.logger.info("Phase 3: Bias correction")
        try:
            corrected_signals = self.bias_corrector.correct_hierarchical_biases(extracted_signals)
        except Exception as e:
            self.logger.error(f"Bias correction failed: {e}")
            corrected_signals = extracted_signals
        
        # Compute final performance metrics
        total_time = time.time() - start_time
        final_metrics = self._compute_performance_metrics(corrected_signals, residual_data, total_time)
        
        self.logger.info(f"AHSD decomposition completed in {total_time:.2f}s")
        self.logger.info(f"Extracted {len(corrected_signals)} signals")
        
        return {
            'extracted_signals': corrected_signals,
            'residual_data': residual_data,
            'extraction_history': self.extraction_history,
            'priority_ranking': priority_ranking,
            'performance_metrics': final_metrics,
            'total_processing_time': total_time
        }
    
    def _fallback_ranking(self, detections: List[Dict]) -> List[int]:
        """Fallback ranking based on SNR"""
        snrs = []
        for i, det in enumerate(detections):
            snr = det.get('network_snr', det.get('snr', 0.0))
            snrs.append((snr, i))
        
        # Sort by SNR (descending)
        snrs.sort(reverse=True)
        return [idx for _, idx in snrs]
    
    def _update_model_complexity(self, residual_data: Dict, remaining_signals: int):
        """Adaptively update model complexity"""
        
        # Compute residual power
        total_power = sum(np.var(strain) for strain in residual_data.values())
        
        # Adjust neural PE complexity
        if remaining_signals > 2 and total_power > 1e-44:
            self.adaptive_subtractor.neural_pe.set_complexity("high")
        elif remaining_signals > 0:
            self.adaptive_subtractor.neural_pe.set_complexity("medium") 
        else:
            self.adaptive_subtractor.neural_pe.set_complexity("low")
        
        self.logger.debug(f"Model complexity updated for {remaining_signals} remaining signals")
    
    def _compute_extraction_snr(self, extraction_result: Dict) -> float:
        """Estimate extraction SNR from posterior"""
        posterior_summary = extraction_result.get('posterior_summary', {})
        
        # Try to get direct SNR estimate
        if 'network_snr' in posterior_summary:
            return posterior_summary['network_snr']['median']
        
        # Estimate from distance and masses
        try:
            distance = posterior_summary.get('luminosity_distance', {}).get('median', 500)
            m1 = posterior_summary.get('mass_1', {}).get('median', 30)
            m2 = posterior_summary.get('mass_2', {}).get('median', 30)
            
            # Rough SNR scaling: SNR ∝ Mc^5/6 / D
            chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
            estimated_snr = 1000 * (chirp_mass / 30)**(5/6) / (distance / 400)
            
            return max(estimated_snr, 8.0)  # Minimum threshold
            
        except:
            return 10.0  # Default estimate
    
    def _compute_residual_stats(self, residual_data: Dict) -> Dict:
        """Compute residual data statistics"""
        stats = {}
        
        for det_name, strain in residual_data.items():
            try:
                stats[det_name] = {
                    'rms': float(np.sqrt(np.mean(strain**2))),
                    'max_abs': float(np.max(np.abs(strain))),
                    'mean': float(np.mean(strain)),
                    'std': float(np.std(strain)),
                    'power': float(np.sum(strain**2))
                }
            except:
                stats[det_name] = {
                    'rms': 0.0, 'max_abs': 0.0, 'mean': 0.0, 'std': 0.0, 'power': 0.0
                }
        
        return stats
    
    def _compute_performance_metrics(self, extracted_signals: List[Dict], 
                                   residual_data: Dict, total_time: float) -> Dict:
        """Compute overall performance metrics"""
        
        metrics = {
            'n_extracted_signals': len(extracted_signals),
            'total_extraction_time': total_time,
            'residual_power': sum(stats.get('power', 0) for stats in 
                                self._compute_residual_stats(residual_data).values()),
            'extraction_snrs': [self._compute_extraction_snr(signal) for signal in extracted_signals],
            'mean_extraction_snr': 0.0,
            'processing_rate': len(extracted_signals) / total_time if total_time > 0 else 0.0
        }
        
        # Compute mean SNR
        if metrics['extraction_snrs']:
            metrics['mean_extraction_snr'] = np.mean(metrics['extraction_snrs'])
        
        # Add bias metrics if ground truth available
        if self.ground_truth is not None:
            try:
                bias_metrics = self._compute_bias_metrics(extracted_signals)
                metrics.update(bias_metrics)
            except Exception as e:
                self.logger.warning(f"Failed to compute bias metrics: {e}")
        
        return metrics
    
    def _compute_bias_metrics(self, extracted_signals: List[Dict]) -> Dict:
        """Compute parameter estimation biases against ground truth"""
        
        bias_metrics = {}
        all_biases = []
        
        for i, signal in enumerate(extracted_signals):
            if i < len(self.ground_truth['signals']):
                true_params = self.ground_truth['signals'][i]
                posterior_summary = signal.get('posterior_summary', {})
                
                signal_biases = {}
                for param_name in self._get_param_names():
                    if param_name in true_params and param_name in posterior_summary:
                        true_val = true_params[param_name]
                        est_val = posterior_summary[param_name]['median']
                        std_val = posterior_summary[param_name]['std']
                        
                        if std_val > 0:
                            normalized_bias = abs(est_val - true_val) / std_val
                            signal_biases[param_name] = normalized_bias
                            all_biases.append(normalized_bias)
                
                bias_metrics[f'signal_{i}_biases'] = signal_biases
        
        # Overall bias statistics
        if all_biases:
            bias_metrics.update({
                'mean_parameter_bias': np.mean(all_biases),
                'median_parameter_bias': np.median(all_biases),
                'max_parameter_bias': np.max(all_biases),
                'bias_std': np.std(all_biases)
            })
        
        return bias_metrics
    
    def set_ground_truth(self, ground_truth: Dict):
        """Set ground truth for evaluation"""
        self.ground_truth = ground_truth
        self.logger.info("Ground truth set for bias evaluation")
    
    def save_pipeline_state(self, filepath: str):
        """Save trained pipeline state"""
        import torch
        
        state = {
            'priority_net_state': self.priority_net.state_dict(),
            'bias_corrector_state': self.bias_corrector.bias_estimator.state_dict(),
            'config': self.config,
            'extraction_history': self.extraction_history,
            'performance_metrics': self.performance_metrics
        }
        
        torch.save(state, filepath)
        self.logger.info(f"Pipeline state saved to {filepath}")
    
    def load_pipeline_state(self, filepath: str):
        """Load trained pipeline state"""
        import torch
        
        try:
            state = torch.load(filepath, map_location='cpu')
            
            self.priority_net.load_state_dict(state['priority_net_state'])
            self.bias_corrector.bias_estimator.load_state_dict(state['bias_corrector_state'])
            
            self.extraction_history = state.get('extraction_history', [])
            self.performance_metrics = state.get('performance_metrics', {})
            
            self.logger.info(f"Pipeline state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load pipeline state: {e}")
