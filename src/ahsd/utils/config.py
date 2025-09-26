from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
import numpy as np

@dataclass
class DetectorConfig:
    name: str
    psd_file: Optional[str] = None
    sampling_rate: int = 4096
    duration: float = 8.0  

@dataclass 
class WaveformConfig:
    approximant: str = "IMRPhenomPv2"
    f_lower: float = 20.0
    f_ref: float = 20.0
    duration: float = 8.0  

@dataclass
class PriorityNetConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128, 64])  
    dropout: float = 0.25  
    learning_rate: float = 0.0005  
    batch_size: int = 16  

@dataclass
class AdaptiveSubtractorConfig:
    neural_pe: Dict = field(default_factory=lambda: {
        'flow_layers': 8,
        'hidden_features': 256,  
        'num_blocks': 3,         
        'context_features': 300  
    })
    uncertainty_realizations: int = 200  

@dataclass
class BiasCorrectorConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])  
    training_epochs: int = 1500  

@dataclass
class AHSDConfig:
    detectors: List[DetectorConfig]
    waveform: WaveformConfig
    priority_net: PriorityNetConfig
    adaptive_subtractor: AdaptiveSubtractorConfig
    bias_corrector: BiasCorrectorConfig
    max_overlapping_signals: int = 5
    snr_threshold: float = 8.0
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """: Better error handling and defaults"""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            config_dict = {}
        
        # Convert detector configs with better defaults
        detector_configs = []
        for det in config_dict.get('detectors', []):
            detector_configs.append(DetectorConfig(**det))
        
        if not detector_configs:
            # Default detectors
            detector_configs = [
                DetectorConfig('H1'),
                DetectorConfig('L1'), 
                DetectorConfig('V1')
            ]
        
        return cls(
            detectors=detector_configs,
            waveform=WaveformConfig(**config_dict.get('waveform', {})),
            priority_net=PriorityNetConfig(**config_dict.get('priority_net', {})),
            adaptive_subtractor=AdaptiveSubtractorConfig(**config_dict.get('adaptive_subtractor', {})),
            bias_corrector=BiasCorrectorConfig(**config_dict.get('bias_corrector', {})),
            max_overlapping_signals=config_dict.get('max_overlapping_signals', 5),
            snr_threshold=config_dict.get('snr_threshold', 8.0),
            seed=config_dict.get('seed', 42)
        )

# Add missing utility functions that other files expect
def add_parameter_noise(param_name: str, true_value: float) -> float:
    """Add realistic noise to parameter estimates"""
    noise_levels = {
        'geocent_time': 0.01,  
        'ra': 0.1,  # ~6 degree uncertainty
        'dec': 0.1,
        'psi': 0.5,
        'phase': 1.0,
        'theta_jn': 0.3,
        'a_1': 0.1,
        'a_2': 0.1,
        'tilt_1': 0.3,
        'tilt_2': 0.3,
        'phi_12': 1.0,
        'phi_jl': 1.0
    }
    
    std = noise_levels.get(param_name, 0.1)
    return np.random.normal(true_value, std)

def compute_detailed_metrics(true_parameters: List[Dict], 
                           analysis_result: Dict, 
                           method_name: str) -> Dict:
    """Compute detailed performance metrics for a method"""
    
    extracted_signals = analysis_result.get('extracted_signals', [])
    
    # Parameter bias metrics
    parameter_biases = []
    coverage_probabilities = []
    
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'geocent_time', 'ra', 'dec']
    
    for i, true_params in enumerate(true_parameters):
        if i < len(extracted_signals):
            extracted = extracted_signals[i]
            
            for param_name in param_names:
                if param_name in true_params and param_name in extracted.get('posterior_summary', {}):
                    true_val = true_params[param_name]
                    posterior = extracted['posterior_summary'][param_name]
                    
                    # Normalized bias
                    estimated_val = posterior.get('median', posterior.get('mean', 0))
                    std_val = posterior.get('std', 1)
                    bias = abs(estimated_val - true_val) / std_val if std_val > 0 else 0
                    parameter_biases.append(bias)
                    
                    # Coverage probability (90% credible interval)
                    if 'quantiles' in posterior and len(posterior['quantiles']) >= 2:
                        q5, q95 = posterior['quantiles'][0], posterior['quantiles'][-1]
                        coverage = 1.0 if q5 <= true_val <= q95 else 0.0
                        coverage_probabilities.append(coverage)
    
    # Computational efficiency metrics
    extraction_time = analysis_result.get('performance_metrics', {}).get('total_extraction_time', 0)
    
    # Signal recovery metrics
    n_true_signals = len(true_parameters)
    n_recovered_signals = len(extracted_signals)
    recovery_rate = n_recovered_signals / n_true_signals if n_true_signals > 0 else 0
    
    return {
        'method_name': method_name,
        'parameter_biases': parameter_biases,
        'mean_parameter_bias': np.mean(parameter_biases) if parameter_biases else np.inf,
        'coverage_probabilities': coverage_probabilities,
        'mean_coverage': np.mean(coverage_probabilities) if coverage_probabilities else 0,
        'extraction_time': extraction_time,
        'recovery_rate': recovery_rate,
        'n_true_signals': n_true_signals,
        'n_recovered_signals': n_recovered_signals
    }

def compute_summary_metrics(evaluation_results: Dict) -> Dict:
    """Compute summary statistics across all test scenarios"""
    
    methods = ['ahsd', 'hierarchical_baseline', 'joint_pe_baseline']
    summary = {}
    
    for method in methods:
        results_key = f'{method}_results'
        if results_key in evaluation_results:
            method_results = evaluation_results[results_key]
            
            # Aggregate metrics across scenarios
            all_biases = []
            all_coverages = []
            all_times = []
            all_recovery_rates = []
            
            for result in method_results:
                if 'metrics' in result:
                    metrics = result['metrics']
                    all_biases.extend(metrics.get('parameter_biases', []))
                    all_coverages.extend(metrics.get('coverage_probabilities', []))
                    all_times.append(metrics.get('extraction_time', 0))
                    all_recovery_rates.append(metrics.get('recovery_rate', 0))
            
            summary[method] = {
                'mean_bias': np.mean(all_biases) if all_biases else np.inf,
                'median_bias': np.median(all_biases) if all_biases else np.inf,
                'bias_std': np.std(all_biases) if all_biases else 0,
                'mean_coverage': np.mean(all_coverages) if all_coverages else 0,
                'mean_extraction_time': np.mean(all_times) if all_times else 0,
                'mean_recovery_rate': np.mean(all_recovery_rates) if all_recovery_rates else 0
            }
    
    # Compute relative improvements
    if 'ahsd' in summary and 'hierarchical_baseline' in summary:
        ahsd_bias = summary['ahsd']['mean_bias']
        hierarchical_bias = summary['hierarchical_baseline']['mean_bias']
        
        if hierarchical_bias > 0 and not np.isinf(hierarchical_bias):
            bias_improvement = (hierarchical_bias - ahsd_bias) / hierarchical_bias
        else:
            bias_improvement = 0
        
        if summary['hierarchical_baseline']['mean_extraction_time'] > 0:
            time_ratio = summary['ahsd']['mean_extraction_time'] / summary['hierarchical_baseline']['mean_extraction_time']
        else:
            time_ratio = 1
            
        summary['ahsd_vs_hierarchical'] = {
            'bias_improvement': bias_improvement,
            'time_ratio': time_ratio
        }
    
    return summary
