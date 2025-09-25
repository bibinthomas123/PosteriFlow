#!/usr/bin/env python3
"""
AHSD Pipeline -  PHYSICS-BASED IMPLEMENTATION
"""

import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Any, Optional
import logging

# Import real components
from .adaptive_subtractor import AdaptiveSubtractor
from .bias_corrector import BiasCorrector

class AHSDPipeline:
    """REAL AHSD Pipeline with complete physics-based implementation"""
    
    def __init__(self, config=None, priority_net=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize REAL components
        param_names = [
            'mass_1', 'mass_2', 'luminosity_distance', 
            'geocent_time', 'ra', 'dec', 'theta_jn', 'psi', 'phase'
        ]
        
        self.adaptive_subtractor = AdaptiveSubtractor()
        self.bias_corrector = BiasCorrector(param_names)
        self.priority_net = priority_net
        
        # Pipeline configuration
        self.max_iterations = 10
        self.convergence_threshold = 0.01
        self.quality_threshold = 0.3
        
        self.logger.info("✅ REAL AHSDPipeline initialized with physics-based components")
    
    def decompose_overlapping_signals(self, data: Dict, initial_detections: List[Dict]) -> Dict:
        """REAL AHSD pipeline analysis with complete physics implementation."""
        
        start_time = time.time()
        
        try:
            # ✅ REAL: Intelligent signal prioritization using trained PriorityNet
            if self.priority_net is not None:
                prioritized_detections = self._prioritize_signals_with_net(initial_detections)
            else:
                prioritized_detections = self._prioritize_signals_heuristic(initial_detections)
            
            # ✅ REAL: Iterative signal extraction with adaptive subtraction
            extracted_signals = []
            remaining_data = data.copy()
            
            for iteration, detection in enumerate(prioritized_detections):
                if iteration >= self.max_iterations:
                    self.logger.warning(f"Reached maximum iterations ({self.max_iterations})")
                    break
                
                # ✅ REAL: Extract signal using adaptive subtractor
                residual_data, extraction_result, uncertainties = self.adaptive_subtractor.extract_and_subtract(
                    remaining_data, iteration
                )
                
                # ✅ REAL: Quality assessment
                signal_quality = extraction_result.get('signal_quality', 0.0)
                
                if signal_quality >= self.quality_threshold:
                    # ✅ REAL: Apply bias correction
                    corrected_signals = self.bias_corrector.correct_hierarchical_biases([extraction_result])
                    
                    if corrected_signals:
                        final_signal = corrected_signals[0]
                        final_signal.update({
                            'iteration': iteration,
                            'priority_score': detection.get('priority_score', 0.0),
                            'original_detection': detection
                        })
                        extracted_signals.append(final_signal)
                        
                        # Update remaining data for next iteration
                        remaining_data = residual_data
                        
                        self.logger.debug(f"Successfully extracted signal {iteration} with quality {signal_quality:.3f}")
                    else:
                        self.logger.warning(f"Bias correction failed for signal {iteration}")
                else:
                    self.logger.debug(f"Signal {iteration} quality too low ({signal_quality:.3f} < {self.quality_threshold})")
            
            processing_time = time.time() - start_time
            
            # ✅ REAL: Compute comprehensive performance metrics
            performance_metrics = self._compute_comprehensive_performance_metrics(
                extracted_signals, initial_detections, processing_time
            )
            
            return {
                'extracted_signals': extracted_signals,
                'method': 'ahsd_real_physics',
                'performance_metrics': performance_metrics,
                'pipeline_metadata': {
                    'initial_detections': len(initial_detections),
                    'prioritized_detections': len(prioritized_detections),
                    'iterations_completed': len(extracted_signals),
                    'convergence_achieved': len(extracted_signals) == len(prioritized_detections),
                    'quality_threshold': self.quality_threshold,
                    'used_priority_net': self.priority_net is not None
                }
            }
            
        except Exception as e:
            self.logger.error(f"REAL AHSD pipeline failed: {e}")
            
            # Fallback result
            return {
                'extracted_signals': [],
                'method': 'ahsd_real_physics',
                'error': str(e),
                'performance_metrics': {
                    'total_extraction_time': time.time() - start_time,
                    'n_extracted_signals': 0,
                    'pipeline_failed': True
                }
            }
    
    def _prioritize_signals_with_net(self, detections: List[Dict]) -> List[Dict]:
        """REAL signal prioritization using trained PriorityNet."""
        
        try:
            # Use PriorityNet to rank detections
            ranked_indices = self.priority_net.rank_detections(detections)
            
            # Add priority scores
            prioritized = []
            for rank, idx in enumerate(ranked_indices):
                if idx < len(detections):
                    detection = detections[idx].copy()
                    detection['priority_score'] = 1.0 - (rank / len(ranked_indices))  # Higher score = higher priority
                    detection['priority_rank'] = rank
                    prioritized.append(detection)
            
            self.logger.debug(f"PriorityNet ranked {len(prioritized)} signals")
            return prioritized
            
        except Exception as e:
            self.logger.warning(f"PriorityNet prioritization failed: {e}, using heuristic")
            return self._prioritize_signals_heuristic(detections)
    
    def _prioritize_signals_heuristic(self, detections: List[Dict]) -> List[Dict]:
        """REAL heuristic signal prioritization based on physics principles."""
        
        try:
            scored_detections = []
            
            for i, detection in enumerate(detections):
                # ✅ REAL: Multi-factor priority scoring
                priority_score = 0.0
                
                # SNR factor (40% weight) - higher SNR = higher priority
                snr = detection.get('network_snr', 10.0)
                snr_score = min(snr / 20.0, 1.0)  # Normalize to [0,1]
                priority_score += 0.4 * snr_score
                
                # Mass factor (25% weight) - moderate masses easier to extract
                m1 = detection.get('mass_1', 30.0)
                m2 = detection.get('mass_2', 30.0)
                total_mass = m1 + m2
                
                # Optimal mass range is 40-80 solar masses
                if 40 <= total_mass <= 80:
                    mass_score = 1.0
                elif 20 <= total_mass <= 120:
                    mass_score = 0.7
                else:
                    mass_score = 0.4
                
                priority_score += 0.25 * mass_score
                
                # Distance factor (20% weight) - closer signals are easier
                distance = detection.get('luminosity_distance', 500.0)
                distance_score = max(0.2, min(1.0, 1000.0 / distance))  # Inverse relationship
                priority_score += 0.2 * distance_score
                
                # Sky localization factor (15% weight) - better localized signals prioritized
                if 'ra' in detection and 'dec' in detection:
                    # Assume better localized signals have more complete parameters
                    localization_score = 1.0
                else:
                    localization_score = 0.5
                
                priority_score += 0.15 * localization_score
                
                scored_detections.append({
                    **detection,
                    'priority_score': priority_score,
                    'snr_score': snr_score,
                    'mass_score': mass_score,
                    'distance_score': distance_score,
                    'localization_score': localization_score
                })
            
            # Sort by priority score (highest first)
            prioritized = sorted(scored_detections, key=lambda x: x['priority_score'], reverse=True)
            
            # Add priority ranks
            for rank, detection in enumerate(prioritized):
                detection['priority_rank'] = rank
            
            self.logger.debug(f"Heuristically prioritized {len(prioritized)} signals")
            return prioritized
            
        except Exception as e:
            self.logger.error(f"Heuristic prioritization failed: {e}")
            return detections  # Return original order as fallback
    
    def _compute_comprehensive_performance_metrics(self, extracted_signals: List[Dict], 
                                                 initial_detections: List[Dict],
                                                 processing_time: float) -> Dict[str, Any]:
        """REAL comprehensive performance metrics computation."""
        
        metrics = {
            'timing_metrics': {},
            'extraction_metrics': {},
            'quality_metrics': {},
            'efficiency_metrics': {}
        }
        
        try:
            n_initial = len(initial_detections)
            n_extracted = len(extracted_signals)
            
            # ✅ REAL: Timing metrics
            metrics['timing_metrics'] = {
                'total_extraction_time': float(processing_time),
                'average_time_per_signal': float(processing_time / max(n_extracted, 1)),
                'extraction_rate_hz': float(n_extracted / max(processing_time, 1e-6)),
                'time_per_input_signal': float(processing_time / max(n_initial, 1))
            }
            
            # ✅ REAL: Extraction metrics
            metrics['extraction_metrics'] = {
                'n_initial_detections': n_initial,
                'n_extracted_signals': n_extracted,
                'extraction_rate': float(n_extracted / max(n_initial, 1)),
                'signals_processed': n_extracted,
                'processing_efficiency': float(n_extracted / max(n_initial, 1))
            }
            
            # ✅ REAL: Quality metrics
            if extracted_signals:
                qualities = [s.get('signal_quality', 0.0) for s in extracted_signals]
                bias_corrections = [s.get('bias_correction', {}) for s in extracted_signals]
                
                # Count successful bias corrections
                successful_corrections = sum(1 for bc in bias_corrections 
                                           if isinstance(bc, dict) and bc.get('applied', False))
                
                metrics['quality_metrics'] = {
                    'mean_signal_quality': float(np.mean(qualities)),
                    'std_signal_quality': float(np.std(qualities)),
                    'min_signal_quality': float(np.min(qualities)),
                    'max_signal_quality': float(np.max(qualities)),
                    'high_quality_signals': sum(1 for q in qualities if q >= 0.7),
                    'bias_corrections_applied': successful_corrections,
                    'bias_correction_rate': float(successful_corrections / n_extracted)
                }
                
                # Signal quality distribution
                quality_bins = [
                    sum(1 for q in qualities if q >= 0.8),  # Excellent
                    sum(1 for q in qualities if 0.6 <= q < 0.8),  # Good
                    sum(1 for q in qualities if 0.4 <= q < 0.6),  # Fair
                    sum(1 for q in qualities if q < 0.4)  # Poor
                ]
                
                metrics['quality_metrics']['quality_distribution'] = {
                    'excellent': quality_bins[0],
                    'good': quality_bins[1], 
                    'fair': quality_bins[2],
                    'poor': quality_bins[3]
                }
            
            # ✅ REAL: Efficiency metrics
            if n_initial > 0:
                # Computational efficiency
                theoretical_time = n_initial * 0.5  # Assume 0.5s per signal baseline
                computational_efficiency = theoretical_time / max(processing_time, 0.1)
                
                # Extraction efficiency
                extraction_efficiency = n_extracted / n_initial
                
                # Overall efficiency
                overall_efficiency = computational_efficiency * extraction_efficiency
                
                metrics['efficiency_metrics'] = {
                    'computational_efficiency': float(computational_efficiency),
                    'extraction_efficiency': float(extraction_efficiency),
                    'overall_efficiency': float(overall_efficiency),
                    'theoretical_baseline_time': float(theoretical_time),
                    'efficiency_grade': self._compute_efficiency_grade(overall_efficiency)
                }
            
            # ✅ REAL: Advanced metrics
            if extracted_signals:
                # SNR statistics
                snr_estimates = []
                for signal in extracted_signals:
                    context_snr = signal.get('context_snr', 0.0)
                    if context_snr > 0:
                        snr_estimates.append(context_snr)
                
                if snr_estimates:
                    metrics['snr_statistics'] = {
                        'mean_estimated_snr': float(np.mean(snr_estimates)),
                        'std_estimated_snr': float(np.std(snr_estimates)),
                        'snr_range': [float(np.min(snr_estimates)), float(np.max(snr_estimates))]
                    }
                
                # Method performance by iteration
                iteration_performance = {}
                for signal in extracted_signals:
                    iteration = signal.get('iteration', 0)
                    quality = signal.get('signal_quality', 0.0)
                    
                    if iteration not in iteration_performance:
                        iteration_performance[iteration] = []
                    iteration_performance[iteration].append(quality)
                
                avg_quality_by_iteration = {
                    iter_num: float(np.mean(qualities))
                    for iter_num, qualities in iteration_performance.items()
                }
                
                metrics['iteration_performance'] = avg_quality_by_iteration
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def _compute_efficiency_grade(self, efficiency: float) -> str:
        """Compute efficiency grade."""
        
        if efficiency >= 1.5:
            return 'excellent'
        elif efficiency >= 1.0:
            return 'good'
        elif efficiency >= 0.7:
            return 'fair'
        else:
            return 'poor'
    
    def train_bias_corrector(self, training_scenarios: List[Dict]):
        """Train the bias corrector component."""
        
        try:
            self.logger.info("Training AHSD bias corrector...")
            self.bias_corrector.train_bias_estimator(training_scenarios)
            self.logger.info("✅ AHSD bias corrector training completed")
            
        except Exception as e:
            self.logger.error(f"Bias corrector training failed: {e}")
    
    def set_priority_net(self, priority_net):
        """Set the trained PriorityNet."""
        
        self.priority_net = priority_net
        self.logger.info("✅ PriorityNet set for AHSD pipeline")
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """Get current pipeline state and configuration."""
        
        return {
            'has_priority_net': self.priority_net is not None,
            'bias_corrector_trained': self.bias_corrector.is_trained,
            'max_iterations': self.max_iterations,
            'quality_threshold': self.quality_threshold,
            'convergence_threshold': self.convergence_threshold,
            'component_status': {
                'adaptive_subtractor': 'initialized',
                'bias_corrector': 'trained' if self.bias_corrector.is_trained else 'initialized',
                'priority_net': 'loaded' if self.priority_net is not None else 'not_loaded'
            }
        }
    
    def save_pipeline_state(self, filepath: str):
        """Save pipeline state to file."""
        
        try:
            import torch
            
            state = {
                'config': self.config,
                'max_iterations': self.max_iterations,
                'quality_threshold': self.quality_threshold,
                'convergence_threshold': self.convergence_threshold,
                'bias_corrector_state': {
                    'param_names': self.bias_corrector.param_names,
                    'is_trained': self.bias_corrector.is_trained,
                    'correction_stats': self.bias_corrector.correction_stats
                }
            }
            
            # Save PriorityNet state if available
            if self.priority_net is not None:
                state['priority_net_state'] = self.priority_net.state_dict()
            
            # Save bias corrector model if trained
            if self.bias_corrector.is_trained:
                state['bias_estimator_state'] = self.bias_corrector.bias_estimator.state_dict()
            
            torch.save(state, filepath)
            self.logger.info(f"✅ Pipeline state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline state: {e}")
    
    def load_pipeline_state(self, filepath: str):
        """Load pipeline state from file."""
        
        try:
            import torch
            
            state = torch.load(filepath, map_location='cpu')
            
            # Restore configuration
            self.max_iterations = state.get('max_iterations', 10)
            self.quality_threshold = state.get('quality_threshold', 0.3)
            self.convergence_threshold = state.get('convergence_threshold', 0.01)
            
            # Restore bias corrector state
            bias_state = state.get('bias_corrector_state', {})
            if bias_state:
                self.bias_corrector.is_trained = bias_state.get('is_trained', False)
                self.bias_corrector.correction_stats = bias_state.get('correction_stats', {})
            
            # Restore bias estimator model if available
            if 'bias_estimator_state' in state and self.bias_corrector.is_trained:
                self.bias_corrector.bias_estimator.load_state_dict(state['bias_estimator_state'])
            
            # Note: PriorityNet would need to be loaded separately
            
            self.logger.info(f"✅ Pipeline state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load pipeline state: {e}")
    
    def validate_pipeline_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of the pipeline components."""
        
        validation_results = {
            'overall_status': 'unknown',
            'component_checks': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check adaptive subtractor
            try:
                test_data = {'H1': np.random.normal(0, 1e-23, 1000)}
                _, _, _ = self.adaptive_subtractor.extract_and_subtract(test_data, 0)
                validation_results['component_checks']['adaptive_subtractor'] = 'pass'
            except Exception as e:
                validation_results['component_checks']['adaptive_subtractor'] = 'fail'
                validation_results['errors'].append(f"Adaptive subtractor test failed: {e}")
            
            # Check bias corrector
            try:
                test_signal = {
                    'posterior_summary': {
                        'mass_1': {'median': 30.0, 'std': 3.0},
                        'mass_2': {'median': 25.0, 'std': 2.5}
                    },
                    'signal_quality': 0.8
                }
                corrected = self.bias_corrector.correct_hierarchical_biases([test_signal])
                validation_results['component_checks']['bias_corrector'] = 'pass'
                
                if not self.bias_corrector.is_trained:
                    validation_results['warnings'].append("Bias corrector not trained - using physics-based corrections")
                    
            except Exception as e:
                validation_results['component_checks']['bias_corrector'] = 'fail'
                validation_results['errors'].append(f"Bias corrector test failed: {e}")
            
            # Check PriorityNet
            if self.priority_net is not None:
                try:
                    test_detections = [
                        {'mass_1': 35.0, 'mass_2': 30.0, 'network_snr': 15.0},
                        {'mass_1': 25.0, 'mass_2': 20.0, 'network_snr': 12.0}
                    ]
                    ranking = self.priority_net.rank_detections(test_detections)
                    validation_results['component_checks']['priority_net'] = 'pass'
                except Exception as e:
                    validation_results['component_checks']['priority_net'] = 'fail'
                    validation_results['errors'].append(f"PriorityNet test failed: {e}")
            else:
                validation_results['component_checks']['priority_net'] = 'not_loaded'
                validation_results['warnings'].append("PriorityNet not loaded - using heuristic prioritization")
            
            # Determine overall status
            if validation_results['errors']:
                validation_results['overall_status'] = 'fail'
            elif validation_results['warnings']:
                validation_results['overall_status'] = 'pass_with_warnings'
            else:
                validation_results['overall_status'] = 'pass'
            
            # Configuration validation
            config_issues = []
            
            if self.quality_threshold < 0.1:
                config_issues.append("Quality threshold very low - may accept poor extractions")
            elif self.quality_threshold > 0.8:
                config_issues.append("Quality threshold very high - may reject good extractions")
            
            if self.max_iterations > 20:
                config_issues.append("Max iterations very high - may be computationally expensive")
            elif self.max_iterations < 3:
                config_issues.append("Max iterations very low - may miss signals")
            
            if config_issues:
                validation_results['warnings'].extend(config_issues)
            
            validation_results['configuration'] = {
                'max_iterations': self.max_iterations,
                'quality_threshold': self.quality_threshold,
                'convergence_threshold': self.convergence_threshold
            }
            
        except Exception as e:
            validation_results['overall_status'] = 'error'
            validation_results['errors'].append(f"Validation process failed: {e}")
        
        return validation_results
    
    def optimize_pipeline_parameters(self, test_scenarios: List[Dict]) -> Dict[str, Any]:
        """Optimize pipeline parameters based on test scenarios."""
        
        optimization_results = {
            'original_parameters': {
                'quality_threshold': self.quality_threshold,
                'max_iterations': self.max_iterations
            },
            'optimization_process': [],
            'recommended_parameters': {},
            'performance_improvement': {}
        }
        
        try:
            if not test_scenarios:
                optimization_results['error'] = 'no_test_scenarios_provided'
                return optimization_results
            
            # Test different parameter combinations
            test_quality_thresholds = [0.2, 0.3, 0.4, 0.5]
            test_max_iterations = [5, 8, 10, 12]
            
            best_performance = -1
            best_params = {}
            
            original_quality_threshold = self.quality_threshold
            original_max_iterations = self.max_iterations
            
            for quality_thresh in test_quality_thresholds:
                for max_iters in test_max_iterations:
                    
                    # Update parameters
                    self.quality_threshold = quality_thresh
                    self.max_iterations = max_iters
                    
                    # Test on scenarios
                    performance_scores = []
                    
                    for i, scenario in enumerate(test_scenarios[:min(10, len(test_scenarios))]):  # Limit for speed
                        try:
                            true_params = scenario.get('true_parameters', [])
                            injected_data = scenario.get('injected_data', {})
                            
                            if not true_params or not injected_data:
                                continue
                            
                            # Create mock detections
                            detections = []
                            for params in true_params:
                                detection = params.copy()
                                # Add some noise
                                for key in detection:
                                    if isinstance(detection[key], (int, float)) and key != 'signal_id':
                                        detection[key] *= (1 + np.random.normal(0, 0.05))
                                detections.append(detection)
                            
                            # Run pipeline
                            result = self.decompose_overlapping_signals(injected_data, detections)
                            
                            # Compute performance score
                            n_extracted = len(result.get('extracted_signals', []))
                            n_true = len(true_params)
                            
                            if n_extracted > 0:
                                extraction_time = result.get('performance_metrics', {}).get('total_extraction_time', 1.0)
                                avg_quality = np.mean([s.get('signal_quality', 0.0) 
                                                     for s in result.get('extracted_signals', [])])
                                
                                # Combined score: recovery rate * quality / time
                                recovery_rate = n_extracted / max(n_true, 1)
                                performance_score = (recovery_rate * avg_quality) / max(extraction_time, 0.1)
                                performance_scores.append(performance_score)
                            
                        except Exception as e:
                            self.logger.debug(f"Optimization test failed for scenario {i}: {e}")
                            continue
                    
                    # Average performance for this parameter combination
                    if performance_scores:
                        avg_performance = np.mean(performance_scores)
                        
                        optimization_results['optimization_process'].append({
                            'quality_threshold': quality_thresh,
                            'max_iterations': max_iters,
                            'avg_performance': float(avg_performance),
                            'n_test_scenarios': len(performance_scores)
                        })
                        
                        if avg_performance > best_performance:
                            best_performance = avg_performance
                            best_params = {
                                'quality_threshold': quality_thresh,
                                'max_iterations': max_iters
                            }
            
            # Restore original parameters
            self.quality_threshold = original_quality_threshold
            self.max_iterations = original_max_iterations
            
            # Set recommendations
            if best_params:
                optimization_results['recommended_parameters'] = best_params
                
                # Compute improvement
                original_performance = None
                for test in optimization_results['optimization_process']:
                    if (test['quality_threshold'] == original_quality_threshold and 
                        test['max_iterations'] == original_max_iterations):
                        original_performance = test['avg_performance']
                        break
                
                if original_performance:
                    improvement = (best_performance - original_performance) / original_performance * 100
                    optimization_results['performance_improvement'] = {
                        'original_performance': float(original_performance),
                        'best_performance': float(best_performance),
                        'improvement_percent': float(improvement)
                    }
                
                self.logger.info(f"✅ Parameter optimization completed. Best performance: {best_performance:.6f}")
            else:
                optimization_results['error'] = 'no_valid_parameter_combinations_tested'
            
        except Exception as e:
            optimization_results['error'] = str(e)
            
            # Restore original parameters on error
            self.quality_threshold = optimization_results['original_parameters']['quality_threshold']
            self.max_iterations = optimization_results['original_parameters']['max_iterations']
        
        return optimization_results
    
    def get_performance_report(self, recent_results: List[Dict] = None) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        
        report = {
            'pipeline_status': self.get_pipeline_state(),
            'validation_results': self.validate_pipeline_integrity(),
            'performance_statistics': {},
            'recommendations': []
        }
        
        try:
            if recent_results:
                # Analyze recent performance
                extraction_times = []
                quality_scores = []
                recovery_rates = []
                
                for result in recent_results:
                    perf_metrics = result.get('performance_metrics', {})
                    
                    # Timing
                    time = perf_metrics.get('total_extraction_time', 0)
                    if time > 0:
                        extraction_times.append(time)
                    
                    # Quality
                    signals = result.get('extracted_signals', [])
                    if signals:
                        qualities = [s.get('signal_quality', 0.0) for s in signals]
                        quality_scores.extend(qualities)
                        
                        # Recovery rate (approximated)
                        n_extracted = len(signals)
                        n_detections = perf_metrics.get('n_initial_detections', n_extracted)
                        if n_detections > 0:
                            recovery_rates.append(n_extracted / n_detections)
                
                # Performance statistics
                if extraction_times:
                    report['performance_statistics']['timing'] = {
                        'mean_extraction_time': float(np.mean(extraction_times)),
                        'std_extraction_time': float(np.std(extraction_times)),
                        'min_extraction_time': float(np.min(extraction_times)),
                        'max_extraction_time': float(np.max(extraction_times))
                    }
                
                if quality_scores:
                    report['performance_statistics']['quality'] = {
                        'mean_signal_quality': float(np.mean(quality_scores)),
                        'std_signal_quality': float(np.std(quality_scores)),
                        'min_signal_quality': float(np.min(quality_scores)),
                        'max_signal_quality': float(np.max(quality_scores))
                    }
                
                if recovery_rates:
                    report['performance_statistics']['recovery'] = {
                        'mean_recovery_rate': float(np.mean(recovery_rates)),
                        'std_recovery_rate': float(np.std(recovery_rates)),
                        'min_recovery_rate': float(np.min(recovery_rates)),
                        'max_recovery_rate': float(np.max(recovery_rates))
                    }
                
                # Generate recommendations
                recommendations = []
                
                # Timing recommendations
                if extraction_times:
                    avg_time = np.mean(extraction_times)
                    if avg_time > 5.0:
                        recommendations.append("Consider reducing max_iterations for faster processing")
                    elif avg_time < 0.5:
                        recommendations.append("Processing very fast - consider increasing quality_threshold")
                
                # Quality recommendations
                if quality_scores:
                    avg_quality = np.mean(quality_scores)
                    if avg_quality < 0.5:
                        recommendations.append("Low signal quality - consider improving neural PE training")
                    elif avg_quality > 0.9:
                        recommendations.append("Excellent signal quality - pipeline performing well")
                
                # Recovery recommendations
                if recovery_rates:
                    avg_recovery = np.mean(recovery_rates)
                    if avg_recovery < 0.7:
                        recommendations.append("Low recovery rate - consider lowering quality_threshold")
                    elif avg_recovery > 0.95:
                        recommendations.append("Excellent recovery rate - consider increasing quality_threshold")
                
                report['recommendations'] = recommendations
            
            report['report_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
        except Exception as e:
            report['error'] = str(e)
        
        return report
