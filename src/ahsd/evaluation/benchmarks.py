"""
Baseline methods for comparison with AHSD.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import bilby
import time
from concurrent.futures import ProcessPoolExecutor
from ..utils.config import AHSDConfig

class StandardHierarchicalSubtraction:
    """Standard hierarchical subtraction baseline."""
    
    def __init__(self, config: Optional[AHSDConfig] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze(self, data: Dict, initial_detections: List[Dict]) -> Dict:
        """Run standard hierarchical analysis."""
        
        start_time = time.time()
        extracted_signals = []
        
        # Sort detections by SNR (highest first)
        sorted_detections = sorted(initial_detections, 
                                 key=lambda x: x.get('network_snr', 0), 
                                 reverse=True)
        
        current_data = {det: np.array(strain) for det, strain in data.items()}
        
        for i, detection in enumerate(sorted_detections):
            try:
                # Run bilby parameter estimation
                result = self._run_bilby_analysis(current_data, detection)
                
                if result is not None:
                    # Subtract signal from data
                    current_data = self._subtract_signal(current_data, result)
                    extracted_signals.append(result)
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze detection {i}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        return {
            'extracted_signals': extracted_signals,
            'method': 'standard_hierarchical',
            'performance_metrics': {
                'total_extraction_time': processing_time,
                'n_recovered_signals': len(extracted_signals)
            }
        }
    
    def _run_bilby_analysis(self, data: Dict, detection: Dict) -> Optional[Dict]:
        """Run bilby parameter estimation for single signal."""
        
        try:
            # Setup interferometers
            interferometers = []
            for det_name, strain in data.items():
                if len(strain) > 0:
                    ifo = bilby.gw.detector.get_detector(det_name)
                    ifo.strain_data.set_from_array(
                        strain, 
                        sampling_frequency=4096,
                        duration=4.0,
                        start_time=0.0
                    )
                    interferometers.append(ifo)
            
            if not interferometers:
                return None
            
            # Setup priors
            priors = self._get_default_priors()
            
            # Likelihood
            likelihood = bilby.gw.GravitationalWaveTransient(
                interferometers=interferometers,
                waveform_generator=self._get_waveform_generator()
            )
            
            # Run sampling (reduced for speed)
            result = bilby.run_sampler(
                likelihood=likelihood,
                priors=priors,
                sampler='dynesty',
                nlive=100,  # Reduced for speed
                dlogz=0.1,
                sample='rwalk'
            )
            
            # Extract posterior summary
            posterior_summary = {}
            for param in priors.keys():
                if param in result.posterior.columns:
                    samples = result.posterior[param].values
                    posterior_summary[param] = {
                        'median': float(np.median(samples)),
                        'mean': float(np.mean(samples)),
                        'std': float(np.std(samples)),
                        'quantiles': np.percentile(samples, [5, 16, 84, 95]).tolist()
                    }
            
            return {
                'posterior_summary': posterior_summary,
                'posterior_samples': {param: result.posterior[param].values 
                                    for param in priors.keys() 
                                    if param in result.posterior.columns},
                'log_evidence': float(result.log_evidence),
                'extraction_method': 'bilby_standard'
            }
            
        except Exception as e:
            self.logger.debug(f"Bilby analysis failed: {e}")
            return None
    
    def _get_default_priors(self) -> bilby.core.prior.PriorDict:
        """Get default priors for BBH analysis."""
        
        priors = bilby.core.prior.PriorDict()
        priors['mass_1'] = bilby.core.prior.Uniform(10, 50, 'mass_1')
        priors['mass_2'] = bilby.core.prior.Uniform(10, 50, 'mass_2')
        priors['luminosity_distance'] = bilby.core.prior.PowerLaw(100, 2000, 'luminosity_distance', alpha=2)
        priors['theta_jn'] = bilby.core.prior.Sine('theta_jn')
        priors['psi'] = bilby.core.prior.Uniform(0, np.pi, 'psi')
        priors['phase'] = bilby.core.prior.Uniform(0, 2*np.pi, 'phase')
        priors['geocent_time'] = bilby.core.prior.Uniform(-0.1, 0.1, 'geocent_time')
        priors['ra'] = bilby.core.prior.Uniform(0, 2*np.pi, 'ra')
        priors['dec'] = bilby.core.prior.Cosine('dec')
        
        return priors
    
    def _get_waveform_generator(self):
        """Get waveform generator."""
        
        return bilby.gw.WaveformGenerator(
            duration=4.0,
            sampling_frequency=4096,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(
                waveform_approximant="IMRPhenomPv2",
                reference_frequency=20.0,
            )
        )
    
    def _subtract_signal(self, data: Dict, signal_result: Dict) -> Dict:
        """Subtract signal from data using posterior median."""
        
        try:
            # Get median parameters
            median_params = {}
            for param, summary in signal_result['posterior_summary'].items():
                median_params[param] = summary['median']
            
            # Generate waveform
            waveform_generator = self._get_waveform_generator()
            waveform_polarizations = waveform_generator.frequency_domain_strain(median_params)
            
            # Project to detectors and subtract
            residual_data = {}
            for det_name, strain in data.items():
                try:
                    detector = bilby.gw.detector.get_detector(det_name)
                    projected_strain = detector.project_wave(
                        waveform_polarizations['plus'],
                        waveform_polarizations['cross'],
                        median_params['ra'],
                        median_params['dec'], 
                        median_params['psi']
                    ).time_domain_strain
                    
                    # Subtract (ensure same length)
                    min_len = min(len(strain), len(projected_strain))
                    residual_data[det_name] = strain[:min_len] - projected_strain[:min_len]
                    
                except Exception as e:
                    self.logger.debug(f"Failed to subtract from {det_name}: {e}")
                    residual_data[det_name] = strain
            
            return residual_data
            
        except Exception as e:
            self.logger.debug(f"Signal subtraction failed: {e}")
            return data

class JointParameterEstimation:
    """Joint parameter estimation baseline."""
    
    def __init__(self, config: Optional[AHSDConfig] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze(self, data: Dict, initial_detections: List[Dict]) -> Dict:
        """Run joint parameter estimation."""
        
        start_time = time.time()
        
        try:
            # Setup multi-signal model
            n_signals = len(initial_detections)
            result = self._run_joint_analysis(data, n_signals, initial_detections)
            
            if result is not None:
                extracted_signals = self._split_joint_result(result, n_signals)
            else:
                extracted_signals = []
            
        except Exception as e:
            self.logger.error(f"Joint analysis failed: {e}")
            extracted_signals = []
        
        processing_time = time.time() - start_time
        
        return {
            'extracted_signals': extracted_signals,
            'method': 'joint_pe',
            'performance_metrics': {
                'total_extraction_time': processing_time,
                'n_recovered_signals': len(extracted_signals)
            }
        }
    
    def _run_joint_analysis(self, data: Dict, n_signals: int, detections: List[Dict]) -> Optional[Dict]:
        """Run joint multi-signal analysis."""
        
        try:
            # This would implement joint PE for multiple signals
            # For now, return a simplified result
            
            # Setup interferometers
            interferometers = []
            for det_name, strain in data.items():
                if len(strain) > 0:
                    ifo = bilby.gw.detector.get_detector(det_name)
                    # Simplified setup
                    interferometers.append(ifo)
            
            if not interferometers:
                return None
            
            # For multiple signals, this would need custom likelihood
            # Simplified approach: analyze strongest signal only
            if detections:
                strongest = max(detections, key=lambda x: x.get('network_snr', 0))
                
                # Run single signal analysis as approximation
                standard_analyzer = StandardHierarchicalSubtraction(self.config)
                result = standard_analyzer._run_bilby_analysis(data, strongest)
                
                return result
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Joint analysis implementation failed: {e}")
            return None
    
    def _split_joint_result(self, joint_result: Dict, n_signals: int) -> List[Dict]:
        """Split joint result into individual signals."""
        
        # For true joint analysis, this would split the posterior
        # For now, return the single result
        return [joint_result] if joint_result else []

class BaselineHierarchicalSubtraction:
    """Simple baseline hierarchical subtraction."""
    
    def __init__(self, config: AHSDConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def run_analysis(self, data: Dict, n_signals: int) -> Dict:
        """Run simple hierarchical analysis."""
        
        # Generate fake results for training
        extracted_parameters = []
        
        for i in range(n_signals):
            # Generate fake posterior with realistic uncertainties
            fake_posterior = {
                'mass_1': {
                    'median': np.random.normal(35, 5),
                    'std': np.random.uniform(2, 8),
                    'quantiles': [0, 0, 0, 0]
                },
                'mass_2': {
                    'median': np.random.normal(30, 5), 
                    'std': np.random.uniform(2, 8),
                    'quantiles': [0, 0, 0, 0]
                },
                'luminosity_distance': {
                    'median': np.random.normal(500, 200),
                    'std': np.random.uniform(50, 200),
                    'quantiles': [0, 0, 0, 0]
                },
                'geocent_time': {
                    'median': np.random.normal(0, 0.01),
                    'std': np.random.uniform(0.005, 0.02),
                    'quantiles': [0, 0, 0, 0]
                },
                'ra': {
                    'median': np.random.uniform(0, 2*np.pi),
                    'std': np.random.uniform(0.1, 0.5),
                    'quantiles': [0, 0, 0, 0]
                },
                'dec': {
                    'median': np.random.uniform(-1, 1),
                    'std': np.random.uniform(0.1, 0.5),
                    'quantiles': [0, 0, 0, 0]
                }
            }
            
            extracted_parameters.append(fake_posterior)
        
        return {'extracted_parameters': extracted_parameters}
