"""
Evaluation metrics for AHSD performance assessment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_curve, auc

class BiasMetrics:
    """Compute parameter estimation bias metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compute_parameter_bias(self,
                              true_values: Union[List, np.ndarray],
                              estimated_values: Union[List, np.ndarray],
                              uncertainties: Optional[Union[List, np.ndarray]] = None) -> Dict:
        """
        Compute bias metrics for parameter estimation.
        
        Parameters:
        -----------
        true_values : array-like
            True parameter values
        estimated_values : array-like  
            Estimated parameter values
        uncertainties : array-like, optional
            Parameter uncertainties (std dev)
            
        Returns:
        --------
        dict
            Bias metrics
        """
        
        true_values = np.array(true_values)
        estimated_values = np.array(estimated_values)
        
        if uncertainties is not None:
            uncertainties = np.array(uncertainties)
        
        # Absolute bias
        absolute_bias = estimated_values - true_values
        
        # Relative bias
        relative_bias = absolute_bias / np.where(true_values != 0, true_values, 1)
        
        # Normalized bias (if uncertainties available)
        if uncertainties is not None:
            normalized_bias = absolute_bias / np.where(uncertainties > 0, uncertainties, 1)
        else:
            normalized_bias = absolute_bias
        
        metrics = {
            'mean_absolute_bias': float(np.mean(np.abs(absolute_bias))),
            'std_absolute_bias': float(np.std(absolute_bias)),
            'mean_relative_bias': float(np.mean(np.abs(relative_bias))),
            'std_relative_bias': float(np.std(relative_bias)),
            'mean_normalized_bias': float(np.mean(np.abs(normalized_bias))),
            'std_normalized_bias': float(np.std(normalized_bias)),
            'max_absolute_bias': float(np.max(np.abs(absolute_bias))),
            'rms_bias': float(np.sqrt(np.mean(absolute_bias**2))),
            'bias_outliers': int(np.sum(np.abs(normalized_bias) > 3.0))  # > 3σ
        }
        
        return metrics
    
    def compute_coverage_probability(self,
                                   true_values: Union[List, np.ndarray],
                                   credible_intervals: List[Tuple[float, float]],
                                   confidence_level: float = 0.9) -> Dict:
        """Compute coverage probability of credible intervals."""
        
        true_values = np.array(true_values)
        
        covered = []
        for i, (lower, upper) in enumerate(credible_intervals):
            if i < len(true_values):
                covered.append(lower <= true_values[i] <= upper)
        
        coverage = np.mean(covered) if covered else 0.0
        expected_coverage = confidence_level
        
        return {
            'coverage_probability': float(coverage),
            'expected_coverage': float(expected_coverage),
            'coverage_difference': float(coverage - expected_coverage),
            'n_covered': int(np.sum(covered)),
            'n_total': len(covered)
        }

class PerformanceMetrics:
    """Compute computational performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compute_timing_metrics(self, processing_times: List[float]) -> Dict:
        """Compute timing performance metrics."""
        
        times = np.array(processing_times)
        
        return {
            'mean_time': float(np.mean(times)),
            'median_time': float(np.median(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'total_time': float(np.sum(times)),
            'percentile_95': float(np.percentile(times, 95))
        }
    
    def compute_throughput_metrics(self,
                                 n_signals_processed: List[int],
                                 processing_times: List[float]) -> Dict:
        """Compute throughput metrics."""
        
        signals = np.array(n_signals_processed)
        times = np.array(processing_times)
        
        # Avoid division by zero
        throughput = signals / np.where(times > 0, times, 1e-6)
        
        return {
            'mean_throughput': float(np.mean(throughput)),
            'median_throughput': float(np.median(throughput)),
            'total_signals_processed': int(np.sum(signals)),
            'total_processing_time': float(np.sum(times)),
            'overall_throughput': float(np.sum(signals) / np.sum(times)) if np.sum(times) > 0 else 0.0
        }
    
    def compute_resource_metrics(self, memory_usage: List[float]) -> Dict:
        """Compute resource utilization metrics."""
        
        memory = np.array(memory_usage)
        
        return {
            'mean_memory_mb': float(np.mean(memory)),
            'peak_memory_mb': float(np.max(memory)),
            'memory_efficiency': float(np.mean(memory) / np.max(memory)) if np.max(memory) > 0 else 0.0
        }

class RecoveryMetrics:
    """Compute signal recovery metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compute_detection_metrics(self,
                                true_signals: List[bool],
                                detected_signals: List[bool]) -> Dict:
        """Compute detection performance metrics."""
        
        true_signals = np.array(true_signals, dtype=bool)
        detected_signals = np.array(detected_signals, dtype=bool)
        
        # Ensure same length
        min_len = min(len(true_signals), len(detected_signals))
        true_signals = true_signals[:min_len]
        detected_signals = detected_signals[:min_len]
        
        # Confusion matrix components
        tp = np.sum(true_signals & detected_signals)
        fp = np.sum(~true_signals & detected_signals) 
        fn = np.sum(true_signals & ~detected_signals)
        tn = np.sum(~true_signals & ~detected_signals)
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'accuracy': float(accuracy),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn)
        }
    
    def compute_snr_recovery_rate(self,
                                 true_snrs: List[float],
                                 detected_flags: List[bool],
                                 snr_bins: Optional[List[float]] = None) -> Dict:
        """Compute recovery rate as function of SNR."""
        
        if snr_bins is None:
            snr_bins = [8, 10, 12, 15, 20, 30]
        
        true_snrs = np.array(true_snrs)
        detected_flags = np.array(detected_flags)
        
        recovery_rates = []
        bin_centers = []
        
        for i in range(len(snr_bins) - 1):
            low_snr = snr_bins[i]
            high_snr = snr_bins[i + 1]
            
            # Signals in this SNR bin
            in_bin = (true_snrs >= low_snr) & (true_snrs < high_snr)
            
            if np.sum(in_bin) > 0:
                recovery_rate = np.mean(detected_flags[in_bin])
                recovery_rates.append(recovery_rate)
                bin_centers.append((low_snr + high_snr) / 2)
        
        return {
            'snr_bins': snr_bins,
            'bin_centers': bin_centers,
            'recovery_rates': recovery_rates,
            'overall_recovery_rate': float(np.mean(detected_flags))
        }

class ComparisonMetrics:
    """Compare different methods."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compare_methods(self,
                       method_results: Dict[str, Dict]) -> Dict:
        """Compare multiple methods across metrics."""
        
        comparison = {}
        
        # Get all metric keys
        all_metrics = set()
        for method_result in method_results.values():
            all_metrics.update(method_result.keys())
        
        # Compare each metric
        for metric in all_metrics:
            metric_values = {}
            for method, results in method_results.items():
                if metric in results:
                    metric_values[method] = results[metric]
            
            if len(metric_values) > 1:
                comparison[metric] = self._compute_metric_comparison(metric_values)
        
        return comparison
    
    def _compute_metric_comparison(self, metric_values: Dict[str, float]) -> Dict:
        """Compare values for single metric."""
        
        values = list(metric_values.values())
        methods = list(metric_values.keys())
        
        # Find best/worst
        best_idx = np.argmax(values)
        worst_idx = np.argmin(values)
        
        comparison = {
            'values': metric_values,
            'best_method': methods[best_idx],
            'best_value': values[best_idx],
            'worst_method': methods[worst_idx],
            'worst_value': values[worst_idx],
            'relative_improvement': (values[best_idx] - values[worst_idx]) / values[worst_idx] if values[worst_idx] != 0 else 0.0
        }
        
        return comparison

def aggregate_scenario_metrics(scenario_results: List[Dict], 
                             metric_names: List[str]) -> Dict:
    """Aggregate metrics across multiple scenarios."""
    
    aggregated = {}
    
    for metric_name in metric_names:
        values = []
        
        for result in scenario_results:
            if metric_name in result:
                values.append(result[metric_name])
        
        if values:
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
    
    return aggregated
