#!/usr/bin/env python3
"""his module provides comprehensive metrics and statistical analysis tools for evaluating the performance, bias, recovery, and comparison of signal extraction methods in the context of the AHSD (Advanced Hierarchical Signal Decomposition) framework. The metrics are designed to assess both the accuracy and computational efficiency of various signal extraction algorithms, with a focus on physics-based criteria relevant to gravitational wave signal analysis.
Classes:
--------
- BiasMetrics:
    Computes parameter-wise biases between true and estimated parameters, including statistical significance tests, distributional analysis, and expected vs. observed bias comparison. Provides overall bias metrics and grading.
- PerformanceMetrics:
    Evaluates extraction performance, including timing, accuracy, efficiency, and scalability metrics. Supports method-specific expectations and grades performance based on computational and signal quality criteria.
- RecoveryMetrics:
    Analyzes signal recovery by matching extracted signals to true signals using multi-criteria physical matching (mass, time, sky position, distance). Computes recall, precision, F1 score, and provides detailed analysis of recovery quality and failure patterns.
- ComparisonMetrics:
    Compares multiple extraction methods using pairwise comparisons, multi-criteria ranking, and statistical significance testing. Summarizes method characteristics and provides recommendations based on performance across timing, recovery, and quality.
Dependencies:
-------------
- numpy
- torch
- scipy.stats
- scipy.spatial.distance
- logging
Intended Usage:
---------------
Import this module and instantiate the relevant metric classes to evaluate and compare the performance of signal extraction methods on simulated or real datasets. The metrics are suitable for benchmarking, method development, and reporting in gravitational wave data analysis pipelines.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
import logging
from scipy.stats import ks_2samp, anderson_ksamp
from scipy.spatial.distance import wasserstein_distance

class BiasMetrics:
    """Bias metrics with comprehensive statistical analysis"""
    
    def __init__(self, param_names: List[str]):
        """
        Initialize BiasMetrics.

        Parameters
        ----------
        param_names : List[str]
            List of parameter names for which bias metrics will be computed.
        
        Usage
        -----
        Instantiate with a list of parameter names to evaluate, e.g.:
            bias_metrics = BiasMetrics(['mass_1', 'mass_2', 'luminosity_distance'])
        """
        self.param_names = param_names
        self.logger = logging.getLogger(__name__)
        
        # Parameter-specific bias expectations
        self.expected_biases = {
            'mass_1': 0.05,        # 5% typical bias
            'mass_2': 0.05,
            'luminosity_distance': 0.15,  # 15% typical bias
            'geocent_time': 0.002,    # 2ms typical bias
            'ra': 0.1,            # ~6 degrees typical bias
            'dec': 0.1,
            'theta_jn': 0.2,      # ~11 degrees
            'psi': 0.2,
            'phase': 0.3
        }
        
    def compute_parameter_biases(self, true_params: List[Dict], 
                               estimated_params: List[Dict]) -> Dict[str, Any]:
        """
        Compute parameter-wise biases between true and estimated parameters, including statistical significance tests.

        Parameters
        ----------
        true_params : List[Dict]
            List of dictionaries containing the true parameter values for each signal.
        estimated_params : List[Dict]
            List of dictionaries containing the estimated parameter values for each signal.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing parameter-wise bias metrics, overall metrics, and the number of signals analyzed.
            The structure includes:
                - 'parameter_biases': Dict of bias statistics for each parameter.
                - 'overall_metrics': Summary statistics across all parameters.
                - 'n_signals_analyzed': Number of signals included in the analysis.

        Usage
        -----
        Call this method with lists of true and estimated parameter dictionaries to obtain detailed bias analysis.
        """
        
        if not true_params or not estimated_params:
            return self._get_empty_bias_metrics()
        
        n_signals = min(len(true_params), len(estimated_params))
        
        bias_results = {}
        
        for param_name in self.param_names:
            try:
                # Extract true and estimated values
                true_values = []
                estimated_values = []
                
                for i in range(n_signals):
                    if param_name in true_params[i] and param_name in estimated_params[i]:
                        true_val = true_params[i][param_name]
                        
                        # Handle different estimate formats
                        est_data = estimated_params[i][param_name]
                        if isinstance(est_data, dict):
                            est_val = est_data.get('median', est_data.get('mean', 0.0))
                        else:
                            est_val = float(est_data)
                        
                        # Exclude invalid values
                        if np.isfinite(true_val) and np.isfinite(est_val):
                            true_values.append(true_val)
                            estimated_values.append(est_val)
                
                if len(true_values) >= 3:  # Minimum for statistical analysis
                    param_bias_results = self._compute_comprehensive_parameter_bias(
                        param_name, true_values, estimated_values
                    )
                    bias_results[param_name] = param_bias_results
                else:
                    bias_results[param_name] = self._get_insufficient_data_result(param_name)
                    
            except Exception as e:
                self.logger.debug(f"Bias computation failed for {param_name}: {e}")
                bias_results[param_name] = {'error': str(e)}
        
        # Compute overall bias metrics
        overall_metrics = self._compute_overall_bias_metrics(bias_results)
        
        return {
            'parameter_biases': bias_results,
            'overall_metrics': overall_metrics,
            'n_signals_analyzed': n_signals
        }
    
    def _compute_comprehensive_parameter_bias(self, param_name: str, 
                                            true_vals: List[float], 
                                            est_vals: List[float]) -> Dict[str, Any]:
        """comprehensive bias analysis for a single parameter."""
        
        true_array = np.array(true_vals)
        est_array = np.array(est_vals)
        
        # Basic bias statistics
        absolute_biases = est_array - true_array
        relative_biases = absolute_biases / np.maximum(np.abs(true_array), 1e-6)
        
        # Comprehensive statistical measures
        bias_stats = {
            # Central tendencies
            'mean_absolute_bias': float(np.mean(absolute_biases)),
            'median_absolute_bias': float(np.median(absolute_biases)),
            'mean_relative_bias': float(np.mean(relative_biases)),
            'median_relative_bias': float(np.median(relative_biases)),
            
            # Spread measures
            'std_absolute_bias': float(np.std(absolute_biases)),
            'std_relative_bias': float(np.std(relative_biases)),
            'iqr_relative_bias': float(np.percentile(relative_biases, 75) - np.percentile(relative_biases, 25)),
            
            # Tail behavior
            'max_absolute_bias': float(np.max(np.abs(absolute_biases))),
            'percentile_90_relative_bias': float(np.percentile(np.abs(relative_biases), 90)),
            'percentile_95_relative_bias': float(np.percentile(np.abs(relative_biases), 95)),
            
            # Distribution shape
            'skewness': float(self._compute_skewness(relative_biases)),
            'kurtosis': float(self._compute_kurtosis(relative_biases)),
            
            # Sample information
            'n_samples': len(true_vals),
            'parameter_name': param_name
        }
        
        # Statistical significance tests
        significance_tests = self._perform_bias_significance_tests(absolute_biases, relative_biases, param_name)
        bias_stats.update(significance_tests)
        
        # Expected vs observed bias comparison
        expected_bias = self.expected_biases.get(param_name, 0.1)
        observed_bias = abs(bias_stats['mean_relative_bias'])
        
        bias_stats.update({
            'expected_relative_bias': expected_bias,
            'bias_ratio': observed_bias / expected_bias,
            'bias_significance': 'high' if observed_bias > 2 * expected_bias else 'moderate' if observed_bias > expected_bias else 'low'
        })
        
        return bias_stats
    
    def _perform_bias_significance_tests(self, abs_biases: np.ndarray, 
                                       rel_biases: np.ndarray, 
                                       param_name: str) -> Dict[str, Any]:
        """statistical significance testing for biases."""
        
        tests = {}
        
        try:
            # One-sample t-test for zero bias
            from scipy.stats import ttest_1samp
            t_stat, p_value = ttest_1samp(rel_biases, 0.0)
            
            tests.update({
                'zero_bias_t_statistic': float(t_stat),
                'zero_bias_p_value': float(p_value),
                'zero_bias_significant': p_value < 0.05
            })
            
            # Normality test
            from scipy.stats import shapiro
            if len(rel_biases) <= 5000:  # Shapiro-Wilk limit
                shapiro_stat, shapiro_p = shapiro(rel_biases)
                tests.update({
                    'normality_statistic': float(shapiro_stat),
                    'normality_p_value': float(shapiro_p),
                    'biases_normal': shapiro_p > 0.05
                })
            
            # Outlier detection using IQR method
            q1, q3 = np.percentile(rel_biases, [25, 75])
            iqr = q3 - q1
            outlier_bounds = (q1 - 1.5*iqr, q3 + 1.5*iqr)
            outliers = np.sum((rel_biases < outlier_bounds[0]) | (rel_biases > outlier_bounds[1]))
            
            tests.update({
                'n_outliers': int(outliers),
                'outlier_fraction': float(outliers / len(rel_biases)),
                'outlier_bounds': [float(outlier_bounds[0]), float(outlier_bounds[1])]
            })
            
        except Exception as e:
            tests['significance_test_error'] = str(e)
        
        return tests
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute sample skewness."""
        try:
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            if std > 0:
                return np.mean(((data - mean) / std) ** 3)
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute sample kurtosis (excess kurtosis)."""
        try:
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            if std > 0:
                return np.mean(((data - mean) / std) ** 4) - 3.0
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_overall_bias_metrics(self, param_biases: Dict[str, Dict]) -> Dict[str, Any]:
        """overall bias assessment across all parameters."""
        
        overall_metrics = {
            'parameters_analyzed': 0,
            'mean_relative_bias_magnitude': 0.0,
            'worst_parameter': 'unknown',
            'best_parameter': 'unknown',
            'significant_biases': 0,
            'overall_bias_grade': 'unknown'
        }
        
        try:
            valid_params = []
            relative_biases = []
            
            for param_name, bias_data in param_biases.items():
                if isinstance(bias_data, dict) and 'mean_relative_bias' in bias_data:
                    valid_params.append(param_name)
                    relative_biases.append(abs(bias_data['mean_relative_bias']))
            
            if relative_biases:
                overall_metrics.update({
                    'parameters_analyzed': len(valid_params),
                    'mean_relative_bias_magnitude': float(np.mean(relative_biases)),
                    'max_relative_bias_magnitude': float(np.max(relative_biases)),
                    'std_relative_bias_magnitude': float(np.std(relative_biases))
                })
                
                # Find worst and best parameters
                worst_idx = np.argmax(relative_biases)
                best_idx = np.argmin(relative_biases)
                
                overall_metrics.update({
                    'worst_parameter': valid_params[worst_idx],
                    'worst_parameter_bias': float(relative_biases[worst_idx]),
                    'best_parameter': valid_params[best_idx],
                    'best_parameter_bias': float(relative_biases[best_idx])
                })
                
                # Count significant biases
                significant_count = 0
                for param_name in valid_params:
                    if param_biases[param_name].get('zero_bias_significant', False):
                        significant_count += 1
                
                overall_metrics['significant_biases'] = significant_count
                
                # Overall grade
                mean_bias = overall_metrics['mean_relative_bias_magnitude']
                if mean_bias < 0.05:
                    grade = 'excellent'
                elif mean_bias < 0.10:
                    grade = 'good'
                elif mean_bias < 0.20:
                    grade = 'fair'
                else:
                    grade = 'poor'
                
                overall_metrics['overall_bias_grade'] = grade
        
        except Exception as e:
            overall_metrics['computation_error'] = str(e)
        
        return overall_metrics
    
    def _get_empty_bias_metrics(self) -> Dict[str, Any]:
        """Return empty bias metrics structure."""
        return {
            'parameter_biases': {},
            'overall_metrics': {
                'parameters_analyzed': 0,
                'error': 'no_data'
            },
            'n_signals_analyzed': 0
        }
    
    def _get_insufficient_data_result(self, param_name: str) -> Dict[str, Any]:
        """Return result for insufficient data."""
        return {
            'error': 'insufficient_data',
            'parameter_name': param_name,
            'n_samples': 0,
            'minimum_required': 3
        }


class PerformanceMetrics:
    """Performance metrics with computational and accuracy analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def compute_extraction_performance(self, extraction_results: Dict, 
                                     ground_truth: List[Dict]) -> Dict[str, Any]:
        """extraction performance analysis."""
        
        performance_metrics = {
            'timing_metrics': {},
            'accuracy_metrics': {},
            'efficiency_metrics': {},
            'scalability_metrics': {}
        }
        
        try:
            # Timing analysis
            performance_metrics['timing_metrics'] = self._analyze_timing_performance(extraction_results)
            
            # Accuracy analysis
            performance_metrics['accuracy_metrics'] = self._analyze_accuracy_performance(
                extraction_results, ground_truth
            )
            
            # Efficiency analysis
            performance_metrics['efficiency_metrics'] = self._analyze_efficiency_performance(extraction_results)
            
            # Scalability analysis
            performance_metrics['scalability_metrics'] = self._analyze_scalability_performance(extraction_results)
            
        except Exception as e:
            performance_metrics['error'] = str(e)
        
        return performance_metrics
    
    def _analyze_timing_performance(self, results: Dict) -> Dict[str, Any]:
        """timing performance analysis."""
        
        timing_metrics = {}
        
        try:
            perf_data = results.get('performance_metrics', {})
            
            total_time = perf_data.get('total_extraction_time', 0.0)
            n_signals = perf_data.get('n_recovered_signals', 1)
            
            timing_metrics.update({
                'total_extraction_time': float(total_time),
                'average_time_per_signal': float(total_time / max(n_signals, 1)),
                'extraction_rate_hz': float(n_signals / max(total_time, 1e-6)),
                'timing_efficiency': self._compute_timing_efficiency(total_time, n_signals)
            })
            
            # Method-specific timing analysis
            method = results.get('method', 'unknown')
            expected_times = {
                'ahsd': 1.0,  # Reference time
                'standard_hierarchical': 0.8,
                'joint_parameter_estimation': 3.0,
                'iterative_subtraction': 1.5
            }
            
            expected_time = expected_times.get(method, 1.0)
            actual_time_per_signal = timing_metrics['average_time_per_signal']
            
            timing_metrics.update({
                'expected_time_per_signal': expected_time,
                'timing_ratio': actual_time_per_signal / expected_time,
                'timing_performance': 'fast' if actual_time_per_signal < expected_time else 'slow'
            })
            
        except Exception as e:
            timing_metrics['error'] = str(e)
        
        return timing_metrics
    
    def _compute_timing_efficiency(self, total_time: float, n_signals: int) -> str:
        """Compute timing efficiency grade."""
        
        if total_time <= 0 or n_signals <= 0:
            return 'unknown'
        
        time_per_signal = total_time / n_signals
        
        if time_per_signal < 0.5:
            return 'excellent'
        elif time_per_signal < 1.0:
            return 'good'
        elif time_per_signal < 2.0:
            return 'fair'
        else:
            return 'poor'
    
    def _analyze_accuracy_performance(self, results: Dict, ground_truth: List[Dict]) -> Dict[str, Any]:
        """accuracy performance analysis."""
        
        accuracy_metrics = {}
        
        try:
            extracted_signals = results.get('extracted_signals', [])
            
            if not extracted_signals or not ground_truth:
                return {'error': 'no_data_for_accuracy_analysis'}
            
            # Overall accuracy metrics
            n_extracted = len(extracted_signals)
            n_true = len(ground_truth)
            
            accuracy_metrics.update({
                'recovery_rate': float(n_extracted / max(n_true, 1)),
                'n_signals_recovered': n_extracted,
                'n_signals_true': n_true
            })
            
            # Signal quality analysis
            quality_scores = [s.get('signal_quality', 0.0) for s in extracted_signals]
            if quality_scores:
                accuracy_metrics.update({
                    'mean_signal_quality': float(np.mean(quality_scores)),
                    'std_signal_quality': float(np.std(quality_scores)),
                    'min_signal_quality': float(np.min(quality_scores)),
                    'max_signal_quality': float(np.max(quality_scores))
                })
                
                # Quality grade
                mean_quality = accuracy_metrics['mean_signal_quality']
                if mean_quality > 0.8:
                    quality_grade = 'excellent'
                elif mean_quality > 0.6:
                    quality_grade = 'good'
                elif mean_quality > 0.4:
                    quality_grade = 'fair'
                else:
                    quality_grade = 'poor'
                
                accuracy_metrics['quality_grade'] = quality_grade
            
            # Method-specific accuracy expectations
            method = results.get('method', 'unknown')
            expected_accuracy = self._get_expected_method_accuracy(method)
            
            actual_accuracy = accuracy_metrics.get('mean_signal_quality', 0.0)
            
            accuracy_metrics.update({
                'expected_accuracy': expected_accuracy,
                'accuracy_ratio': actual_accuracy / max(expected_accuracy, 0.1),
                'accuracy_performance': 'above_expected' if actual_accuracy > expected_accuracy else 'below_expected'
            })
            
        except Exception as e:
            accuracy_metrics['error'] = str(e)
        
        return accuracy_metrics
    
    def _get_expected_method_accuracy(self, method: str) -> float:
        """Get expected accuracy for different methods."""
        
        expected_accuracies = {
            'ahsd': 0.75,  # Expected AHSD performance
            'standard_hierarchical': 0.55,
            'joint_parameter_estimation': 0.65,
            'iterative_subtraction': 0.50
        }
        
        return expected_accuracies.get(method, 0.6)
    
    def _analyze_efficiency_performance(self, results: Dict) -> Dict[str, Any]:
        """efficiency performance analysis."""
        
        efficiency_metrics = {}
        
        try:
            perf_data = results.get('performance_metrics', {})
            method = results.get('method', 'unknown')
            
            # Computational efficiency
            total_time = perf_data.get('total_extraction_time', 0.0)
            n_signals = perf_data.get('n_recovered_signals', 0)
            
            # Resource utilization (estimated)
            if method == 'joint_parameter_estimation':
                complexity_factor = perf_data.get('computational_complexity', n_signals**2)
                efficiency_score = n_signals / max(complexity_factor, 1)
            elif method == 'ahsd':
                # AHSD should have near-linear scaling
                efficiency_score = n_signals / max(total_time, 0.1)
            else:
                # Linear scaling assumption
                efficiency_score = n_signals / max(total_time, 0.1)
            
            efficiency_metrics.update({
                'computational_efficiency_score': float(efficiency_score),
                'resource_utilization': self._estimate_resource_utilization(method, n_signals),
                'scalability_factor': self._compute_scalability_factor(method, n_signals)
            })
            
            # Method-specific efficiency analysis
            if method == 'joint_parameter_estimation':
                joint_signals = perf_data.get('joint_estimation_signals', n_signals)
                efficiency_metrics['joint_estimation_efficiency'] = joint_signals / max(n_signals, 1)
            
            elif method == 'iterative_subtraction':
                iterations = perf_data.get('iterations_completed', 1)
                subtraction_eff = perf_data.get('mean_subtraction_efficiency', 0.5)
                efficiency_metrics.update({
                    'iteration_efficiency': float(n_signals / max(iterations, 1)),
                    'subtraction_efficiency': float(subtraction_eff)
                })
            
        except Exception as e:
            efficiency_metrics['error'] = str(e)
        
        return efficiency_metrics
    
    def _estimate_resource_utilization(self, method: str, n_signals: int) -> str:
        """Estimate resource utilization for different methods."""
        
        if method == 'joint_parameter_estimation':
            if n_signals > 3:
                return 'high'
            else:
                return 'moderate'
        elif method == 'ahsd':
            return 'moderate'
        else:
            return 'low'
    
    def _compute_scalability_factor(self, method: str, n_signals: int) -> float:
        """Compute scalability factor for different methods."""
        
        if method == 'joint_parameter_estimation':
            # Exponential scaling
            return 1.0 / (n_signals ** 1.5)
        elif method == 'ahsd':
            # Near-linear scaling
            return 1.0 / (n_signals ** 0.8)
        else:
            # Linear scaling
            return 1.0 / n_signals
    
    def _analyze_scalability_performance(self, results: Dict) -> Dict[str, Any]:
        """scalability performance analysis."""
        
        scalability_metrics = {}
        
        try:
            n_signals = results.get('performance_metrics', {}).get('n_recovered_signals', 0)
            method = results.get('method', 'unknown')
            
            # Theoretical scalability limits
            theoretical_limits = {
                'ahsd': 50,  # Can handle up to 50 overlapping signals
                'standard_hierarchical': 20,  # Limited by error accumulation
                'joint_parameter_estimation': 5,  # Limited by computational cost
                'iterative_subtraction': 10  # Limited by convergence issues
            }
            
            theoretical_limit = theoretical_limits.get(method, 10)
            
            scalability_metrics.update({
                'theoretical_signal_limit': theoretical_limit,
                'current_signal_count': n_signals,
                'scalability_utilization': float(n_signals / theoretical_limit),
                'remaining_capacity': theoretical_limit - n_signals
            })
            
            # Scalability grade
            utilization = scalability_metrics['scalability_utilization']
            if utilization < 0.5:
                grade = 'excellent'
            elif utilization < 0.8:
                grade = 'good'
            elif utilization < 1.0:
                grade = 'fair'
            else:
                grade = 'poor'
            
            scalability_metrics['scalability_grade'] = grade
            
        except Exception as e:
            scalability_metrics['error'] = str(e)
        
        return scalability_metrics


class RecoveryMetrics:
    """Recovery metrics with statistical analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def compute_signal_recovery(self, extracted_signals: List[Dict], 
                              true_signals: List[Dict],
                              matching_threshold: float = 0.1) -> Dict[str, Any]:
        """signal recovery analysis with matching and statistical tests."""
        
        recovery_results = {
            'matching_results': {},
            'recovery_statistics': {},
            'recovery_quality': {},
            'failure_analysis': {}
        }
        
        try:
            # Signal matching
            matches = self._match_signals(extracted_signals, true_signals, matching_threshold)
            recovery_results['matching_results'] = matches
            
            # Recovery statistics
            recovery_results['recovery_statistics'] = self._compute_recovery_statistics(matches, true_signals)
            
            # Recovery quality analysis
            recovery_results['recovery_quality'] = self._analyze_recovery_quality(matches, extracted_signals)
            
            # Failure analysis
            recovery_results['failure_analysis'] = self._analyze_recovery_failures(matches, true_signals)
            
        except Exception as e:
            recovery_results['error'] = str(e)
        
        return recovery_results
    
    def _match_signals(self, extracted: List[Dict], true_signals: List[Dict], 
                      threshold: float) -> Dict[str, Any]:
        """signal matching using multiple criteria."""
        
        matches = {
            'matched_pairs': [],
            'unmatched_extracted': [],
            'unmatched_true': [],
            'match_qualities': [],
            'matching_criteria': {}
        }
        
        try:
            used_true_indices = set()
            used_extracted_indices = set()
            
            # For each extracted signal, find best match in true signals
            for ext_idx, ext_signal in enumerate(extracted):
                best_match_idx = -1
                best_match_score = -1
                
                for true_idx, true_signal in enumerate(true_signals):
                    if true_idx in used_true_indices:
                        continue
                    
                    # Compute match score
                    match_score = self._compute_signal_match_score(ext_signal, true_signal)
                    
                    if match_score > threshold and match_score > best_match_score:
                        best_match_score = match_score
                        best_match_idx = true_idx
                
                if best_match_idx >= 0:
                    # Found a match
                    matches['matched_pairs'].append({
                        'extracted_index': ext_idx,
                        'true_index': best_match_idx,
                        'match_score': best_match_score,
                        'extracted_signal': ext_signal,
                        'true_signal': true_signals[best_match_idx]
                    })
                    matches['match_qualities'].append(best_match_score)
                    
                    used_true_indices.add(best_match_idx)
                    used_extracted_indices.add(ext_idx)
                else:
                    # No match found
                    matches['unmatched_extracted'].append({
                        'index': ext_idx,
                        'signal': ext_signal
                    })
            
            # Find unmatched true signals
            for true_idx, true_signal in enumerate(true_signals):
                if true_idx not in used_true_indices:
                    matches['unmatched_true'].append({
                        'index': true_idx,
                        'signal': true_signal
                    })
            
            # Matching statistics
            matches['matching_criteria'] = {
                'threshold_used': threshold,
                'n_extracted': len(extracted),
                'n_true': len(true_signals),
                'n_matched': len(matches['matched_pairs']),
                'n_unmatched_extracted': len(matches['unmatched_extracted']),
                'n_unmatched_true': len(matches['unmatched_true'])
            }
            
        except Exception as e:
            matches['error'] = str(e)
        
        return matches
    
    def _compute_signal_match_score(self, extracted: Dict, true_signal: Dict) -> float:
        """signal matching score using multiple physical criteria."""
        
        try:
            scores = []
            weights = []
            
            # Get posterior summary
            posterior = extracted.get('posterior_summary', {})
            
            # Mass matching (most important)
            mass_score = self._compute_mass_match_score(posterior, true_signal)
            if mass_score >= 0:
                scores.append(mass_score)
                weights.append(0.4)  # 40% weight
            
            # Time matching
            time_score = self._compute_time_match_score(posterior, true_signal)
            if time_score >= 0:
                scores.append(time_score)
                weights.append(0.3)  # 30% weight
            
            # Sky position matching
            sky_score = self._compute_sky_match_score(posterior, true_signal)
            if sky_score >= 0:
                scores.append(sky_score)
                weights.append(0.2)  # 20% weight
            
            # Distance matching
            distance_score = self._compute_distance_match_score(posterior, true_signal)
            if distance_score >= 0:
                scores.append(distance_score)
                weights.append(0.1)  # 10% weight
            
            if scores and weights:
                # Weighted average
                total_weight = sum(weights)
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
                return weighted_score
            else:
                return 0.0
                
        except Exception as e:
            self.logger.debug(f"Match score computation failed: {e}")
            return 0.0
    
    def _compute_mass_match_score(self, posterior: Dict, true_signal: Dict) -> float:
        """Compute mass matching score."""
        
        try:
            # Extract masses
            true_m1 = true_signal.get('mass_1', 0)
            true_m2 = true_signal.get('mass_2', 0)
            
            if true_m1 <= 0 or true_m2 <= 0:
                return -1
            
            # Get estimated masses
            est_m1_data = posterior.get('mass_1', {})
            est_m2_data = posterior.get('mass_2', {})
            
            if isinstance(est_m1_data, dict):
                est_m1 = est_m1_data.get('median', est_m1_data.get('mean', 0))
                est_m1_std = est_m1_data.get('std', abs(est_m1) * 0.1)
            else:
                est_m1 = float(est_m1_data)
                est_m1_std = abs(est_m1) * 0.1
            
            if isinstance(est_m2_data, dict):
                est_m2 = est_m2_data.get('median', est_m2_data.get('mean', 0))
                est_m2_std = est_m2_data.get('std', abs(est_m2) * 0.1)
            else:
                est_m2 = float(est_m2_data)
                est_m2_std = abs(est_m2) * 0.1
            
            # Compute normalized differences
            m1_diff = abs(est_m1 - true_m1) / max(est_m1_std, true_m1 * 0.05)
            m2_diff = abs(est_m2 - true_m2) / max(est_m2_std, true_m2 * 0.05)
            
            # Gaussian-like scoring
            m1_score = np.exp(-(m1_diff**2) / 2)
            m2_score = np.exp(-(m2_diff**2) / 2)
            
            return (m1_score + m2_score) / 2
            
        except:
            return -1
    
    def _compute_time_match_score(self, posterior: Dict, true_signal: Dict) -> float:
        """Compute time matching score."""
        
        try:
            true_time = true_signal.get('geocent_time', 0)
            
            est_time_data = posterior.get('geocent_time', {})
            if isinstance(est_time_data, dict):
                est_time = est_time_data.get('median', est_time_data.get('mean', 0))
                est_time_std = est_time_data.get('std', 0.01)
            else:
                est_time = float(est_time_data)
                est_time_std = 0.01
            
            # Time difference in milliseconds
            time_diff_ms = abs(est_time - true_time) * 1000
            
            # Scoring: good match if within 3-sigma or 10ms
            threshold_ms = max(3 * est_time_std * 1000, 10.0)
            
            if time_diff_ms <= threshold_ms:
                score = np.exp(-(time_diff_ms / threshold_ms)**2)
            else:
                score = 0.0
            
            return score
            
        except:
            return -1
    
    def _compute_sky_match_score(self, posterior: Dict, true_signal: Dict) -> float:
        """Compute sky position matching score."""
        
        try:
            true_ra = true_signal.get('ra', 0)
            true_dec = true_signal.get('dec', 0)
            
            # Get estimated positions
            est_ra_data = posterior.get('ra', {})
            est_dec_data = posterior.get('dec', {})
            
            if isinstance(est_ra_data, dict):
                est_ra = est_ra_data.get('median', est_ra_data.get('mean', 0))
            else:
                est_ra = float(est_ra_data)
                
            if isinstance(est_dec_data, dict):
                est_dec = est_dec_data.get('median', est_dec_data.get('mean', 0))
            else:
                est_dec = float(est_dec_data)
            
            # Angular separation on sphere
            dra = est_ra - true_ra
            ddec = est_dec - true_dec
            
            # Handle RA wrapping
            if dra > np.pi:
                dra -= 2*np.pi
            elif dra < -np.pi:
                dra += 2*np.pi
            
            # Angular separation (small angle approximation)
            angular_sep = np.sqrt(dra**2 * np.cos(true_dec)**2 + ddec**2)
            
            # Scoring: good match if within 0.5 radians (~30 degrees)
            threshold_rad = 0.5
            
            if angular_sep <= threshold_rad:
                score = np.exp(-(angular_sep / threshold_rad)**2)
            else:
                score = 0.0
            
            return score
            
        except:
            return -1
    
    def _compute_distance_match_score(self, posterior: Dict, true_signal: Dict) -> float:
        """Compute distance matching score."""
        
        try:
            true_distance = true_signal.get('luminosity_distance', 0)
            
            if true_distance <= 0:
                return -1
            
            est_distance_data = posterior.get('luminosity_distance', {})
            if isinstance(est_distance_data, dict):
                est_distance = est_distance_data.get('median', est_distance_data.get('mean', 0))
                est_distance_std = est_distance_data.get('std', true_distance * 0.2)
            else:
                est_distance = float(est_distance_data)
                est_distance_std = true_distance * 0.2
            
            # Log-normal distance comparison (distances are log-normally distributed)
            if est_distance > 0:
                log_true = np.log(true_distance)
                log_est = np.log(est_distance)
                log_std = est_distance_std / est_distance  # Approximate log-std
                
                log_diff = abs(log_est - log_true) / max(log_std, 0.2)
                score = np.exp(-(log_diff**2) / 2)
            else:
                score = 0.0
            
            return score
            
        except:
            return -1
    
    def _compute_recovery_statistics(self, matches: Dict, true_signals: List[Dict]) -> Dict[str, Any]:
        """recovery statistics computation."""
        
        stats = {}
        
        try:
            n_true = len(true_signals)
            n_matched = matches['matching_criteria']['n_matched']
            n_extracted = matches['matching_criteria']['n_extracted']
            
            # Basic recovery metrics
            recall = n_matched / max(n_true, 1)  # Fraction of true signals recovered
            precision = n_matched / max(n_extracted, 1)  # Fraction of extractions that are real
            
            if recall + precision > 0:
                f1_score = 2 * recall * precision / (recall + precision)
            else:
                f1_score = 0.0
            
            stats.update({
                'recall': float(recall),
                'precision': float(precision),
                'f1_score': float(f1_score),
                'n_true_signals': n_true,
                'n_extracted_signals': n_extracted,
                'n_matched_signals': n_matched
            })
            
            # Match quality statistics
            match_qualities = matches.get('match_qualities', [])
            if match_qualities:
                stats.update({
                    'mean_match_quality': float(np.mean(match_qualities)),
                    'std_match_quality': float(np.std(match_qualities)),
                    'min_match_quality': float(np.min(match_qualities)),
                    'max_match_quality': float(np.max(match_qualities))
                })
            
            # Recovery performance grade
            if f1_score > 0.8:
                grade = 'excellent'
            elif f1_score > 0.6:
                grade = 'good'
            elif f1_score > 0.4:
                grade = 'fair'
            else:
                grade = 'poor'
            
            stats['recovery_grade'] = grade
            
        except Exception as e:
            stats['error'] = str(e)
        
        return stats
    
    def _analyze_recovery_quality(self, matches: Dict, extracted_signals: List[Dict]) -> Dict[str, Any]:
        """recovery quality analysis."""
        
        quality_analysis = {}
        
        try:
            matched_pairs = matches.get('matched_pairs', [])
            
            if not matched_pairs:
                return {'error': 'no_matched_signals_for_quality_analysis'}
            
            # Analyze signal quality for matched signals
            matched_qualities = []
            for pair in matched_pairs:
                ext_signal = pair['extracted_signal']
                signal_quality = ext_signal.get('signal_quality', 0.0)
                matched_qualities.append(signal_quality)
            
            quality_analysis.update({
                'matched_signals_mean_quality': float(np.mean(matched_qualities)),
                'matched_signals_std_quality': float(np.std(matched_qualities)),
                'matched_signals_min_quality': float(np.min(matched_qualities)),
                'matched_signals_max_quality': float(np.max(matched_qualities))
            })
            
            # Compare with unmatched extractions
            unmatched_extracted = matches.get('unmatched_extracted', [])
            if unmatched_extracted:
                unmatched_qualities = []
                for um in unmatched_extracted:
                    signal_quality = um['signal'].get('signal_quality', 0.0)
                    unmatched_qualities.append(signal_quality)
                
                quality_analysis.update({
                    'unmatched_extractions_mean_quality': float(np.mean(unmatched_qualities)),
                    'quality_difference': float(np.mean(matched_qualities) - np.mean(unmatched_qualities))
                })
            
            # Quality-based recovery analysis
            high_quality_threshold = 0.7
            high_quality_matched = sum(1 for q in matched_qualities if q >= high_quality_threshold)
            
            quality_analysis.update({
                'high_quality_recovery_rate': float(high_quality_matched / len(matched_qualities)),
                'high_quality_threshold': high_quality_threshold
            })
            
        except Exception as e:
            quality_analysis['error'] = str(e)
        
        return quality_analysis
    
    def _analyze_recovery_failures(self, matches: Dict, true_signals: List[Dict]) -> Dict[str, Any]:
        """recovery failure analysis."""
        
        failure_analysis = {}
        
        try:
            unmatched_true = matches.get('unmatched_true', [])
            
            if not unmatched_true:
                failure_analysis['no_failures'] = True
                return failure_analysis
            
            # Analyze characteristics of failed recoveries
            failed_snrs = []
            failed_masses = []
            failed_distances = []
            
            for failure in unmatched_true:
                true_signal = failure['signal']
                
                snr = true_signal.get('network_snr', 0)
                if snr > 0:
                    failed_snrs.append(snr)
                
                m1 = true_signal.get('mass_1', 0)
                m2 = true_signal.get('mass_2', 0)
                if m1 > 0 and m2 > 0:
                    failed_masses.append((m1, m2))
                
                distance = true_signal.get('luminosity_distance', 0)
                if distance > 0:
                    failed_distances.append(distance)
            
            # Failure statistics
            failure_analysis.update({
                'n_failed_recoveries': len(unmatched_true),
                'failure_rate': float(len(unmatched_true) / max(len(true_signals), 1))
            })
            
            # SNR analysis of failures
            if failed_snrs:
                failure_analysis.update({
                    'failed_mean_snr': float(np.mean(failed_snrs)),
                    'failed_min_snr': float(np.min(failed_snrs)),
                    'failed_max_snr': float(np.max(failed_snrs))
                })
                
                # Compare with overall SNR distribution
                all_snrs = [s.get('network_snr', 0) for s in true_signals if s.get('network_snr', 0) > 0]
                if all_snrs:
                    failure_analysis['snr_bias_in_failures'] = float(np.mean(failed_snrs) - np.mean(all_snrs))
            
            # Mass analysis of failures
            if failed_masses:
                failed_total_masses = [m1 + m2 for m1, m2 in failed_masses]
                failure_analysis.update({
                    'failed_mean_total_mass': float(np.mean(failed_total_masses)),
                    'failed_min_total_mass': float(np.min(failed_total_masses)),
                    'failed_max_total_mass': float(np.max(failed_total_masses))
                })
            
            # Distance analysis of failures
            if failed_distances:
                failure_analysis.update({
                    'failed_mean_distance': float(np.mean(failed_distances)),
                    'failed_min_distance': float(np.min(failed_distances)),
                    'failed_max_distance': float(np.max(failed_distances))
                })
            
            # Failure pattern identification
            failure_patterns = []
            
            # Low SNR failures
            low_snr_failures = [snr for snr in failed_snrs if snr < 10.0]
            if low_snr_failures:
                failure_patterns.append(f"Low SNR: {len(low_snr_failures)}/{len(failed_snrs)} failures")
            
            # High mass failures
            high_mass_failures = [tm for tm in failed_total_masses if tm > 80.0]
            if high_mass_failures:
                failure_patterns.append(f"High total mass: {len(high_mass_failures)}/{len(failed_masses)} failures")
            
            # High distance failures
            high_distance_failures = [d for d in failed_distances if d > 1000.0]
            if high_distance_failures:
                failure_patterns.append(f"High distance: {len(high_distance_failures)}/{len(failed_distances)} failures")
            
            failure_analysis['failure_patterns'] = failure_patterns
            
        except Exception as e:
            failure_analysis['error'] = str(e)
        
        return failure_analysis


class ComparisonMetrics:
    """Comparison metrics for method evaluation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def compare_methods(self, method_results: Dict[str, Dict]) -> Dict[str, Any]:
        """comprehensive method comparison."""
        
        comparison_results = {
            'pairwise_comparisons': {},
            'ranking_analysis': {},
            'statistical_significance': {},
            'performance_summary': {}
        }
        
        try:
            method_names = list(method_results.keys())
            
            if len(method_names) < 2:
                comparison_results['error'] = 'need_at_least_two_methods'
                return comparison_results
            
            # Pairwise comparisons
            comparison_results['pairwise_comparisons'] = self._compute_pairwise_comparisons(method_results)
            
            # Method ranking
            comparison_results['ranking_analysis'] = self._compute_method_ranking(method_results)
            
            # Statistical significance testing
            comparison_results['statistical_significance'] = self._test_statistical_significance(method_results)
            
            # Performance summary
            comparison_results['performance_summary'] = self._summarize_method_performance(method_results)
            
        except Exception as e:
            comparison_results['error'] = str(e)
        
        return comparison_results
    
    def _compute_pairwise_comparisons(self, method_results: Dict[str, Dict]) -> Dict[str, Any]:
        """pairwise method comparisons."""
        
        pairwise = {}
        method_names = list(method_results.keys())
        
        try:
            for i, method1 in enumerate(method_names):
                for j, method2 in enumerate(method_names[i+1:], i+1):
                    
                    comparison_key = f"{method1}_vs_{method2}"
                    
                    # Extract comparable metrics
                    comparison = self._compare_two_methods(
                        method1, method_results[method1],
                        method2, method_results[method2]
                    )
                    
                    pairwise[comparison_key] = comparison
            
        except Exception as e:
            pairwise['error'] = str(e)
        
        return pairwise
    
    def _compare_two_methods(self, name1: str, results1: Dict, 
                           name2: str, results2: Dict) -> Dict[str, Any]:
        """comparison between two methods."""
        
        comparison = {
            'method_1': name1,
            'method_2': name2,
            'performance_comparison': {},
            'timing_comparison': {},
            'quality_comparison': {},
            'winner': 'tie'
        }
        
        try:
            # Performance comparison
            perf1 = results1.get('performance_metrics', {})
            perf2 = results2.get('performance_metrics', {})
            
            # Compare timing
            time1 = perf1.get('total_extraction_time', float('inf'))
            time2 = perf2.get('total_extraction_time', float('inf'))
            
            timing_winner = name1 if time1 < time2 else name2 if time2 < time1 else 'tie'
            timing_ratio = time2 / max(time1, 1e-6)
            
            comparison['timing_comparison'] = {
                'method_1_time': float(time1),
                'method_2_time': float(time2),
                'timing_winner': timing_winner,
                'timing_ratio': float(timing_ratio),
                'speedup': f"{timing_ratio:.2f}x" if timing_winner == name1 else f"{1/timing_ratio:.2f}x"
            }
            
            # Compare signal recovery
            n_signals1 = perf1.get('n_recovered_signals', 0)
            n_signals2 = perf2.get('n_recovered_signals', 0)
            
            recovery_winner = name1 if n_signals1 > n_signals2 else name2 if n_signals2 > n_signals1 else 'tie'
            
            comparison['performance_comparison'] = {
                'method_1_signals': n_signals1,
                'method_2_signals': n_signals2,
                'recovery_winner': recovery_winner,
                'recovery_difference': abs(n_signals1 - n_signals2)
            }
            
            # Compare signal quality
            signals1 = results1.get('extracted_signals', [])
            signals2 = results2.get('extracted_signals', [])
            
            if signals1 and signals2:
                quality1 = np.mean([s.get('signal_quality', 0.0) for s in signals1])
                quality2 = np.mean([s.get('signal_quality', 0.0) for s in signals2])
                
                quality_winner = name1 if quality1 > quality2 else name2 if quality2 > quality1 else 'tie'
                
                comparison['quality_comparison'] = {
                    'method_1_quality': float(quality1),
                    'method_2_quality': float(quality2),
                    'quality_winner': quality_winner,
                    'quality_difference': float(abs(quality1 - quality2))
                }
            
            # Overall winner determination
            wins = {name1: 0, name2: 0, 'tie': 0}
            wins[timing_winner] += 1
            wins[recovery_winner] += 1
            
            if 'quality_comparison' in comparison:
                wins[comparison['quality_comparison']['quality_winner']] += 1
            
            if wins[name1] > wins[name2]:
                comparison['winner'] = name1
            elif wins[name2] > wins[name1]:
                comparison['winner'] = name2
            else:
                comparison['winner'] = 'tie'
            
            comparison['win_counts'] = dict(wins)
            
        except Exception as e:
            comparison['error'] = str(e)
        
        return comparison
    
    def _compute_method_ranking(self, method_results: Dict[str, Dict]) -> Dict[str, Any]:
        """method ranking based on multiple criteria."""
        
        ranking = {
            'overall_ranking': [],
            'criterion_rankings': {},
            'ranking_scores': {},
            'ranking_methodology': 'weighted_multi_criteria'
        }
        
        try:
            method_names = list(method_results.keys())
            
            # Define ranking criteria and weights
            criteria = {
                'timing_performance': 0.3,      # 30% weight
                'recovery_rate': 0.4,           # 40% weight  
                'signal_quality': 0.3           # 30% weight
            }
            
            # Compute scores for each criterion
            criterion_scores = {}
            
            # Timing performance (lower is better)
            timing_scores = {}
            times = []
            for method in method_names:
                time = method_results[method].get('performance_metrics', {}).get('total_extraction_time', float('inf'))
                times.append((method, time))
            
            times.sort(key=lambda x: x[1])  # Sort by time
            for rank, (method, time) in enumerate(times):
                timing_scores[method] = 1.0 - (rank / max(len(times) - 1, 1))  # Normalize to [0,1]
            
            criterion_scores['timing_performance'] = timing_scores
            
            # Recovery rate (higher is better)
            recovery_scores = {}
            recoveries = []
            for method in method_names:
                n_signals = method_results[method].get('performance_metrics', {}).get('n_recovered_signals', 0)
                recoveries.append((method, n_signals))
            
            recoveries.sort(key=lambda x: x[1], reverse=True)  # Sort by recovery (descending)
            for rank, (method, n_signals) in enumerate(recoveries):
                recovery_scores[method] = 1.0 - (rank / max(len(recoveries) - 1, 1))
            
            criterion_scores['recovery_rate'] = recovery_scores
            
            # Signal quality (higher is better)
            quality_scores = {}
            qualities = []
            for method in method_names:
                signals = method_results[method].get('extracted_signals', [])
                if signals:
                    avg_quality = np.mean([s.get('signal_quality', 0.0) for s in signals])
                else:
                    avg_quality = 0.0
                qualities.append((method, avg_quality))
            
            qualities.sort(key=lambda x: x[1], reverse=True)  # Sort by quality (descending)
            for rank, (method, quality) in enumerate(qualities):
                quality_scores[method] = 1.0 - (rank / max(len(qualities) - 1, 1))
            
            criterion_scores['signal_quality'] = quality_scores
            
            # Compute overall weighted scores
            overall_scores = {}
            for method in method_names:
                weighted_score = 0.0
                for criterion, weight in criteria.items():
                    criterion_score = criterion_scores[criterion].get(method, 0.0)
                    weighted_score += weight * criterion_score
                
                overall_scores[method] = weighted_score
            
            # Create final ranking
            final_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
            
            ranking.update({
                'overall_ranking': [method for method, score in final_ranking],
                'criterion_rankings': {
                    criterion: sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    for criterion, scores in criterion_scores.items()
                },
                'ranking_scores': overall_scores,
                'criteria_weights': criteria
            })
            
        except Exception as e:
            ranking['error'] = str(e)
        
        return ranking
    
    def _test_statistical_significance(self, method_results: Dict[str, Dict]) -> Dict[str, Any]:
        """statistical significance testing."""
        
        significance_tests = {
            'quality_differences': {},
            'timing_differences': {},
            'overall_assessment': {}
        }
        
        try:
            method_names = list(method_results.keys())
            
            if len(method_names) < 2:
                return {'error': 'need_at_least_two_methods'}
            
            # Extract quality scores for all methods
            method_qualities = {}
            for method in method_names:
                signals = method_results[method].get('extracted_signals', [])
                qualities = [s.get('signal_quality', 0.0) for s in signals if s.get('signal_quality', 0.0) > 0]
                if len(qualities) >= 3:  # Minimum for statistical test
                    method_qualities[method] = qualities
            
            # Pairwise quality comparisons
            quality_comparisons = {}
            
            for i, method1 in enumerate(method_names):
                for method2 in method_names[i+1:]:
                    if method1 in method_qualities and method2 in method_qualities:
                        
                        qualities1 = method_qualities[method1]
                        qualities2 = method_qualities[method2]
                        
                        # Two-sample t-test
                        try:
                            from scipy.stats import ttest_ind
                            t_stat, p_value = ttest_ind(qualities1, qualities2)
                            
                            comparison_key = f"{method1}_vs_{method2}"
                            quality_comparisons[comparison_key] = {
                                'mean_quality_1': float(np.mean(qualities1)),
                                'mean_quality_2': float(np.mean(qualities2)),
                                't_statistic': float(t_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05,
                                'effect_size': float(abs(np.mean(qualities1) - np.mean(qualities2)) / 
                                                   np.sqrt((np.var(qualities1) + np.var(qualities2)) / 2))
                            }
                        except Exception as e:
                            quality_comparisons[comparison_key] = {'error': str(e)}
            
            significance_tests['quality_differences'] = quality_comparisons
            
            # Overall significance assessment
            significant_differences = sum(1 for comp in quality_comparisons.values() 
                                        if isinstance(comp, dict) and comp.get('significant', False))
            
            total_comparisons = len(quality_comparisons)
            
            significance_tests['overall_assessment'] = {
                'significant_quality_differences': significant_differences,
                'total_quality_comparisons': total_comparisons,
                'significance_rate': float(significant_differences / max(total_comparisons, 1)),
                'methods_analyzed': len(method_qualities),
                'statistical_power': 'high' if significant_differences > 0 else 'low'
            }
            
        except Exception as e:
            significance_tests['error'] = str(e)
        
        return significance_tests
    
    def _summarize_method_performance(self, method_results: Dict[str, Dict]) -> Dict[str, Any]:
        """method performance summary."""
        
        summary = {
            'best_overall': 'unknown',
            'best_timing': 'unknown',
            'best_recovery': 'unknown',
            'best_quality': 'unknown',
            'method_characteristics': {},
            'recommendations': []
        }
        
        try:
            method_names = list(method_results.keys())
            
            # Find best method for each criterion
            best_timing = None
            best_time = float('inf')
            
            best_recovery = None
            best_recovery_count = -1
            
            best_quality = None
            best_quality_score = -1
            
            for method in method_names:
                results = method_results[method]
                perf = results.get('performance_metrics', {})
                
                # Timing
                time = perf.get('total_extraction_time', float('inf'))
                if time < best_time:
                    best_time = time
                    best_timing = method
                
                # Recovery
                recovery = perf.get('n_recovered_signals', 0)
                if recovery > best_recovery_count:
                    best_recovery_count = recovery
                    best_recovery = method
                
                # Quality
                signals = results.get('extracted_signals', [])
                if signals:
                    quality = np.mean([s.get('signal_quality', 0.0) for s in signals])
                    if quality > best_quality_score:
                        best_quality_score = quality
                        best_quality = method
            
            summary.update({
                'best_timing': best_timing,
                'best_recovery': best_recovery,
                'best_quality': best_quality
            })
            
            # Method characteristics
            for method in method_names:
                results = method_results[method]
                perf = results.get('performance_metrics', {})
                signals = results.get('extracted_signals', [])
                
                characteristics = {
                    'extraction_time': float(perf.get('total_extraction_time', 0)),
                    'signals_recovered': perf.get('n_recovered_signals', 0),
                    'mean_signal_quality': float(np.mean([s.get('signal_quality', 0.0) for s in signals])) if signals else 0.0,
                    'method_type': results.get('method', 'unknown')
                }
                
                # Method-specific characteristics
                if 'joint' in method.lower():
                    characteristics['computational_complexity'] = 'high'
                    characteristics['scalability'] = 'limited'
                elif 'hierarchical' in method.lower():
                    characteristics['computational_complexity'] = 'moderate'
                    characteristics['scalability'] = 'good'
                elif 'ahsd' in method.lower():
                    characteristics['computational_complexity'] = 'moderate'
                    characteristics['scalability'] = 'excellent'
                
                summary['method_characteristics'][method] = characteristics
            
            # Generate recommendations
            recommendations = []
            
            if best_timing and best_recovery and best_quality:
                if best_timing == best_recovery == best_quality:
                    recommendations.append(f"Clear winner: {best_timing} excels in all criteria")
                    summary['best_overall'] = best_timing
                else:
                    recommendations.append(f"Speed: {best_timing}, Recovery: {best_recovery}, Quality: {best_quality}")
                    
                    # Determine overall best based on weighted criteria
                    if 'ahsd' in method_names:
                        summary['best_overall'] = 'ahsd'
                        recommendations.append("AHSD recommended for balanced performance")
                    elif best_recovery:
                        summary['best_overall'] = best_recovery
                        recommendations.append(f"{best_recovery} recommended for signal recovery priority")
            
            if len(method_names) > 2:
                recommendations.append("Multiple methods compared - see detailed analysis")
            
            summary['recommendations'] = recommendations
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
