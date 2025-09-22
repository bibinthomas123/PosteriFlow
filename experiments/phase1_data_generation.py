#!/usr/bin/env python3
"""
 Phase 1: Maximum diversity data generation for optimal PriorityNet training
INTEGRATED with  AHSD pipeline
"""
import sys
import os
import numpy as np
import argparse
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import yaml
from typing import List, Dict, Tuple
from scipy.stats import powerlaw, norm
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Safe imports with fallbacks
try:
    from ahsd.utils.config import AHSDConfig
    from ahsd.utils.data_format import standardize_strain_data
    IMPORTS_OK = True
except ImportError as e:
    print(f"Warning: Package imports failed: {e}")
    IMPORTS_OK = False
    
    # Fallback config implementation
    from dataclasses import dataclass, field
    from typing import List, Optional
    
    @dataclass
    class DetectorConfig:
        name: str
        sampling_rate: int = 4096
        duration: float = 8.0
    
    @dataclass
    class WaveformConfig:
        approximant: str = "IMRPhenomPv2"
        f_lower: float = 20.0
        duration: float = 8.0
    
    @dataclass
    class AHSDConfig:
        detectors: List[DetectorConfig] = field(default_factory=lambda: [
            DetectorConfig('H1'), DetectorConfig('L1'), DetectorConfig('V1')
        ])
        waveform: WaveformConfig = field(default_factory=WaveformConfig)
        
        @classmethod
        def from_yaml(cls, config_path: str):
            try:
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                return cls()  # Use defaults if loading fails
            except:
                return cls()

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('phase1_enhanced.log'),
            logging.StreamHandler()
        ]
    )

class OptimalParameterGenerator:
    """Generate parameters optimized for PriorityNet training diversity."""
    
    def __init__(self, config: AHSDConfig):
        self.config = config
        
    def generate_scenario_parameters(self, n_signals: int, scenario_id: int) -> List[Dict]:
        """Generate a complete set of diverse parameters for one scenario."""
        
        signal_parameters = []
        
        # First, determine the "difficulty ranking" for this scenario
        difficulty_levels = self._assign_difficulty_levels(n_signals)
        
        for sig_idx in range(n_signals):
            difficulty = difficulty_levels[sig_idx]
            params = self._generate_single_signal(sig_idx, difficulty, scenario_id)
            signal_parameters.append(params)
        
        # Ensure mutual diversity
        signal_parameters = self._enforce_signal_separation(signal_parameters)
        
        return signal_parameters
    
    def _assign_difficulty_levels(self, n_signals: int) -> List[str]:
        """Assign difficulty levels to create ranking targets."""
        
        if n_signals == 2:
            options = [
                ['easy', 'hard'],
                ['medium', 'hard'],
                ['easy', 'medium']
            ]
            return options[np.random.randint(len(options))]
        elif n_signals == 3:
            options = [
                ['easy', 'medium', 'hard'],
                ['easy', 'hard', 'hard'], 
                ['medium', 'medium', 'hard']
            ]
            return options[np.random.randint(len(options))]
        elif n_signals == 4:
            options = [
                ['easy', 'medium', 'hard', 'hard'],
                ['easy', 'easy', 'medium', 'hard'],
                ['medium', 'medium', 'hard', 'hard']
            ]
            return options[np.random.randint(len(options))]
        else:  # 5 signals
            return ['easy', 'easy', 'medium', 'hard', 'hard']
    
    def _generate_single_signal(self, sig_idx: int, difficulty: str, scenario_id: int) -> Dict:
        """Generate parameters for a single signal with specified difficulty."""
        
        # Mass parameters based on difficulty
        if difficulty == 'easy':
            mass_1 = np.random.uniform(20, 50)  # Intermediate mass
            mass_ratio = np.random.uniform(0.5, 1.0)  # High mass ratio
        elif difficulty == 'medium':
            mass_1 = np.random.uniform(10, 35)  # Lower mass  
            mass_ratio = np.random.uniform(0.3, 0.8)  # Medium mass ratio
        else:  # hard
            mass_1 = np.random.uniform(8, 25)   # Low mass
            mass_ratio = np.random.uniform(0.1, 0.5)  # Low mass ratio
            
        mass_2 = mass_1 * mass_ratio
        
        # Ensure m1 >= m2
        if mass_2 > mass_1:
            mass_1, mass_2 = mass_2, mass_1
        
        # Distance and SNR based on difficulty
        if difficulty == 'easy':
            distance = np.random.uniform(150, 600)   # Close (high SNR)
        elif difficulty == 'medium':
            distance = np.random.uniform(500, 1200)  # Intermediate
        else:  # hard
            distance = np.random.uniform(1000, 2500) # Far (low SNR)
        
        # Compute realistic SNR
        network_snr = self._compute_realistic_snr(mass_1, mass_2, distance)
        
        # Sky position (spread signals across sky)
        ra = np.random.uniform(0, 2 * np.pi)
        dec = np.arcsin(np.random.uniform(-1, 1))
        
        # Temporal separation
        geocent_time = np.random.uniform(-2.0, 2.0) + sig_idx * 0.2
        
        # Orientation parameters
        theta_jn = np.arccos(np.random.uniform(-1, 1))
        psi = np.random.uniform(0, np.pi) 
        phase = np.random.uniform(0, 2 * np.pi)
        
        # Spin parameters
        a_1 = np.random.uniform(0, 0.9)
        a_2 = np.random.uniform(0, 0.9)
        tilt_1 = np.arccos(np.random.uniform(-1, 1))
        tilt_2 = np.arccos(np.random.uniform(-1, 1))
        phi_12 = np.random.uniform(0, 2 * np.pi)
        phi_jl = np.random.uniform(0, 2 * np.pi)
        
        return {
            'mass_1': float(mass_1),
            'mass_2': float(mass_2),
            'luminosity_distance': float(distance),
            'geocent_time': float(geocent_time),
            'ra': float(ra),
            'dec': float(dec),
            'theta_jn': float(theta_jn),
            'psi': float(psi),
            'phase': float(phase),
            'a_1': float(a_1),
            'a_2': float(a_2),
            'tilt_1': float(tilt_1),
            'tilt_2': float(tilt_2),
            'phi_12': float(phi_12),
            'phi_jl': float(phi_jl),
            'signal_id': sig_idx,
            'network_snr': float(network_snr),
            'difficulty': difficulty
        }
    
    def _compute_realistic_snr(self, mass_1: float, mass_2: float, distance: float) -> float:
        """Compute realistic SNR with physical scaling."""
        
        # Chirp mass effect
        chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
        mass_factor = (chirp_mass / 25.0)**(5/6)
        
        # Distance scaling
        distance_factor = 400.0 / distance
        
        # Base SNR for optimal sensitivity
        base_snr = 20.0
        snr = base_snr * mass_factor * distance_factor
        
        # Add realistic scatter
        snr *= np.random.uniform(0.7, 1.3)
        
        return float(np.clip(snr, 6.0, 60.0))
    
    def _enforce_signal_separation(self, signal_parameters: List[Dict]) -> List[Dict]:
        """Ensure signals are sufficiently separated in parameter space."""
        
        if len(signal_parameters) < 2:
            return signal_parameters
        
        # Sort by SNR for consistent ordering
        signal_parameters.sort(key=lambda x: x['network_snr'], reverse=True)
        
        # Ensure minimum separations
        for i in range(1, len(signal_parameters)):
            prev_sig = signal_parameters[i-1]
            curr_sig = signal_parameters[i]
            
            # Ensure minimum mass separation
            mass_diff = abs((prev_sig['mass_1'] + prev_sig['mass_2']) - 
                          (curr_sig['mass_1'] + curr_sig['mass_2']))
            if mass_diff < 10.0:
                # Adjust current signal mass
                adjustment = 10.0 + np.random.uniform(0, 5)
                if curr_sig['mass_1'] + curr_sig['mass_2'] > 50:
                    curr_sig['mass_1'] -= adjustment / 2
                    curr_sig['mass_2'] -= adjustment / 2
                else:
                    curr_sig['mass_1'] += adjustment / 2
                    curr_sig['mass_2'] += adjustment / 2
                
                # Ensure m1 >= m2
                if curr_sig['mass_2'] > curr_sig['mass_1']:
                    curr_sig['mass_1'], curr_sig['mass_2'] = curr_sig['mass_2'], curr_sig['mass_1']
            
            # Ensure minimum distance separation
            dist_diff = abs(prev_sig['luminosity_distance'] - curr_sig['luminosity_distance'])
            if dist_diff < 200.0:
                adjustment = 200.0 + np.random.uniform(0, 100)
                curr_sig['luminosity_distance'] += adjustment
                # Recompute SNR
                curr_sig['network_snr'] = self._compute_realistic_snr(
                    curr_sig['mass_1'], curr_sig['mass_2'], curr_sig['luminosity_distance']
                )
        
        return signal_parameters

def generate_optimal_training_scenarios(config: AHSDConfig, n_scenarios: int) -> List[Dict]:
    """Generate optimized training scenarios for maximum learning."""
    
    logging.info(f"📊 Phase 1: Generating {n_scenarios} optimal training scenarios...")
    
    generator = OptimalParameterGenerator(config)
    scenarios = []
    
    for scenario_id in tqdm(range(n_scenarios), desc="Generating scenarios"):
        # Weighted signal number distribution (favor harder cases)
        n_signals = np.random.choice([2, 3, 4, 5], p=[0.35, 0.35, 0.20, 0.10])
        
        # Generate scenario parameters
        signal_parameters = generator.generate_scenario_parameters(n_signals, scenario_id)
        
        # Create scenario
        scenario = {
            'scenario_id': scenario_id,
            'true_parameters': signal_parameters,
            'injected_data': create_synthetic_data_(signal_parameters, config),
            'n_signals': n_signals,
            'data_type': 'simulated',
            'quality_metrics': compute_scenario_quality_metrics(signal_parameters)
        }
        
        scenarios.append(scenario)
    
    # Filter for quality - less restrictive threshold
    high_quality_scenarios = [s for s in scenarios if s['quality_metrics']['diversity_score'] > 0.3]
    
    logging.info(f"✅ Phase 1: Generated {len(high_quality_scenarios)} high-quality scenarios out of {len(scenarios)}")
    return high_quality_scenarios

def create_synthetic_data_(signal_parameters: List[Dict], config: AHSDConfig) -> Dict:
    """✅ : Create synthetic data structure that matches AHSD pipeline expectations."""
    
    duration = config.waveform.duration
    sample_rate = 4096
    n_samples = int(duration * sample_rate)
    
    # ✅ CRITICAL FIX: Create data format that matches pipeline expectations
    data = {}
    for detector in ['H1', 'L1', 'V1']:
        # Generate noise + signals
        noise = np.random.normal(0, 1e-23, n_samples)
        
        # Add signal components with proper GW physics
        signal_sum = np.zeros(n_samples)
        t = np.linspace(0, duration, n_samples)
        
        for params in signal_parameters:
            try:
                # More realistic chirp waveform
                m1, m2 = params['mass_1'], params['mass_2']
                chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
                
                # Realistic frequency evolution: f ∝ (tc - t)^(-3/8)
                tc = duration + params.get('geocent_time', 0.0)
                time_to_merger = np.maximum(tc - t, 0.01)  # Avoid division by zero
                frequency = 35.0 * (time_to_merger / 1.0)**(-3/8)
                frequency = np.clip(frequency, 35.0, 512.0)  # Physical limits
                
                # Amplitude scaling
                distance = params['luminosity_distance']
                amplitude = params['network_snr'] * 1e-23 / (distance / 400.0)
                
                # Generate waveform
                phase = 2 * np.pi * np.cumsum(frequency) * (t[1] - t[0])
                signal = amplitude * np.sin(phase + params['phase'])
                
                # Apply detector response (simplified)
                signal *= np.random.uniform(0.5, 1.5)  # Simple antenna pattern
                
                signal_sum += signal
                
            except Exception as e:
                logging.debug(f"Signal generation failed: {e}")
                continue
        
        # ✅ : Return simple arrays, not dicts - this is what PriorityNet expects
        data[detector] = noise + signal_sum
    
    return data

def compute_scenario_quality_metrics(signal_parameters: List[Dict]) -> Dict:
    """Compute quality metrics for scenario diversity."""
    
    if len(signal_parameters) < 2:
        return {'diversity_score': 1.0, 'mass_diversity': 0.0, 'distance_diversity': 0.0, 'snr_diversity': 0.0}
    
    try:
        # Mass diversity
        masses = [p['mass_1'] + p['mass_2'] for p in signal_parameters]
        mass_diversity = np.std(masses) / max(np.mean(masses), 1.0)
        
        # Distance diversity  
        distances = [p['luminosity_distance'] for p in signal_parameters]
        distance_diversity = np.std(distances) / max(np.mean(distances), 1.0)
        
        # SNR diversity
        snrs = [p['network_snr'] for p in signal_parameters]
        snr_diversity = np.std(snrs) / max(np.mean(snrs), 1.0)
        
        # Overall diversity score
        diversity_score = np.mean([
            min(mass_diversity, 1.0),
            min(distance_diversity, 1.0), 
            min(snr_diversity, 1.0)
        ])
        
        return {
            'diversity_score': float(diversity_score),
            'mass_diversity': float(mass_diversity),
            'distance_diversity': float(distance_diversity),
            'snr_diversity': float(snr_diversity)
        }
    except Exception as e:
        logging.debug(f"Quality metrics computation failed: {e}")
        return {'diversity_score': 0.5, 'mass_diversity': 0.0, 'distance_diversity': 0.0, 'snr_diversity': 0.0}

def generate_physics_based_biases(scenarios: List[Dict]) -> List[Dict]:
    """Generate realistic parameter biases based on physics and SNR."""
    
    logging.info("📊 Computing physics-based parameter biases...")
    
    baseline_results = []
    
    for scenario in tqdm(scenarios, desc="Computing biases"):
        biases = []
        
        for params in scenario['true_parameters']:
            try:
                snr = params.get('network_snr', 15.0)
                difficulty = params.get('difficulty', 'medium')
                
                # SNR-dependent bias scaling (more realistic)
                if snr > 20:
                    bias_scale = 0.3  # Low bias for high SNR
                elif snr > 12:
                    bias_scale = 0.7  # Medium bias
                else:
                    bias_scale = 1.2  # High bias for low SNR
                    
                # Difficulty-dependent bias
                if difficulty == 'easy':
                    diff_scale = 0.8
                elif difficulty == 'medium':
                    diff_scale = 1.0
                else:  # hard
                    diff_scale = 1.3
                
                total_scale = bias_scale * diff_scale
                
                # ✅ : Generate realistic biases (SMALLER values to prevent issues)
                param_bias = {
                    'mass_1': np.random.normal(0, 0.05 * total_scale),        # Reduced from 0.1
                    'mass_2': np.random.normal(0, 0.05 * total_scale),        # Reduced from 0.1
                    'luminosity_distance': np.random.normal(0, 0.15 * total_scale),  # Reduced from 0.3
                    'ra': np.random.normal(0, 0.1 * total_scale),             # Reduced from 0.2
                    'dec': np.random.normal(0, 0.1 * total_scale),            # Reduced from 0.2
                    'geocent_time': np.random.normal(0, 0.0005 * total_scale) # Reduced from 0.001
                }
                
                biases.append(param_bias)
                
            except Exception as e:
                logging.debug(f"Bias computation failed: {e}")
                # Fallback bias
                biases.append({
                    'mass_1': 0.0, 'mass_2': 0.0, 'luminosity_distance': 0.0,
                    'ra': 0.0, 'dec': 0.0, 'geocent_time': 0.0
                })
        
        baseline_results.append({
            'scenario_id': scenario['scenario_id'],
            'baseline_biases': biases,
            'n_signals': scenario['n_signals'],
            'data_type': scenario.get('data_type', 'simulated')
        })
    
    return baseline_results

def validate_scenario(scenario: Dict) -> bool:
    """✅ : Validate scenario has all required components."""
    required_keys = ['true_parameters', 'injected_data', 'n_signals']
    
    if not all(key in scenario for key in required_keys):
        return False
    
    # Validate parameters
    for params in scenario['true_parameters']:
        if not all(key in params for key in ['mass_1', 'mass_2', 'luminosity_distance']):
            return False
        
        # Check reasonable parameter ranges
        if not (5 <= params['mass_1'] <= 100):
            return False
        if not (5 <= params['mass_2'] <= 100):
            return False
        if not (50 <= params['luminosity_distance'] <= 3000):
            return False
    
    return True

def save_enhanced_training_data(scenarios: List[Dict], baseline_results: List[Dict], output_dir: Path):
    """Save enhanced training data with quality metrics."""
    
    # Validate scenarios before saving
    valid_scenarios = [s for s in scenarios if validate_scenario(s)]
    valid_baseline = baseline_results[:len(valid_scenarios)]
    
    logging.info(f"✅ Saving {len(valid_scenarios)} valid scenarios out of {len(scenarios)}")
    
    # Save scenarios
    with open(output_dir / 'training_scenarios.pkl', 'wb') as f:
        pickle.dump(valid_scenarios, f)
    
    # Save baseline results
    with open(output_dir / 'baseline_results.pkl', 'wb') as f:
        pickle.dump(valid_baseline, f)
    
    # Compute enhanced statistics
    if valid_scenarios:
        diversity_scores = [s['quality_metrics']['diversity_score'] for s in valid_scenarios]
        snr_ranges = []
        mass_ranges = []
        
        for s in valid_scenarios:
            try:
                snrs = [p['network_snr'] for p in s['true_parameters']]
                masses = [p['mass_1'] + p['mass_2'] for p in s['true_parameters']]
                snr_ranges.append(max(snrs) - min(snrs) if len(snrs) > 1 else 0)
                mass_ranges.append(max(masses) - min(masses) if len(masses) > 1 else 0)
            except:
                snr_ranges.append(0)
                mass_ranges.append(0)
        
        stats = {
            'total_scenarios': len(valid_scenarios),
            'signal_distribution': {
                '2_signals': len([s for s in valid_scenarios if s['n_signals'] == 2]),
                '3_signals': len([s for s in valid_scenarios if s['n_signals'] == 3]),
                '4_signals': len([s for s in valid_scenarios if s['n_signals'] == 4]),
                '5_signals': len([s for s in valid_scenarios if s['n_signals'] == 5]),
            },
            'quality_metrics': {
                'avg_diversity_score': float(np.mean(diversity_scores)) if diversity_scores else 0.0,
                'avg_snr_range': float(np.mean(snr_ranges)) if snr_ranges else 0.0,
                'avg_mass_range': float(np.mean(mass_ranges)) if mass_ranges else 0.0,
                'high_quality_fraction': float(np.mean([s > 0.7 for s in diversity_scores])) if diversity_scores else 0.0
            }
        }
    else:
        stats = {'total_scenarios': 0, 'signal_distribution': {}, 'quality_metrics': {}}
    
    with open(output_dir / 'enhanced_dataset_statistics.yaml', 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    logging.info(f"✅ Enhanced training data saved to {output_dir}")
    logging.info(f"📊 Dataset statistics: {stats}")

def main():
    parser = argparse.ArgumentParser(description='Generate enhanced AHSD training data')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--n_simulated', type=int, default=5000, help='Number of simulated scenarios')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    config = AHSDConfig.from_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("🚀 Starting Enhanced Phase 1 Data Generation")
    
    # Generate optimal scenarios
    scenarios = generate_optimal_training_scenarios(config, args.n_simulated)
    
    if not scenarios:
        logging.error("❌ No valid scenarios generated!")
        return
    
    # Generate physics-based biases
    baseline_results = generate_physics_based_biases(scenarios)
    
    # Save enhanced data
    save_enhanced_training_data(scenarios, baseline_results, output_dir)
    
    logging.info("✅ Enhanced Phase 1 data generation completed successfully!")

if __name__ == '__main__':
    main()
