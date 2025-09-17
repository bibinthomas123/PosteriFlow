#!/usr/bin/env python3
"""
Phase 1: Generate training data for AHSD system using both simulated and real data
"""

import numpy as np
import argparse
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import yaml

from src.ahsd.utils.config import AHSDConfig
from src.ahsd.data.simulation import OverlappingSignalSimulator
from src.ahsd.data.gwtc_loader import GWTCDataLoader
from src.ahsd.data.preprocessing import DataPreprocessor
from src.ahsd.data.injection import RealDataSignalInjector

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('phase1_data_generation.log'),
            logging.StreamHandler()
        ]
    )

def generate_simulated_scenarios(config: AHSDConfig, n_scenarios: int = 500) -> List[Dict]:
    """Generate pure simulation scenarios."""
    
    logging.info("Generating simulated overlapping scenarios...")
    
    simulator = OverlappingSignalSimulator(config)
    scenarios = simulator.create_training_dataset(n_scenarios, "data/simulated")
    
    logging.info(f"Generated {len(scenarios)} simulated scenarios")
    return scenarios

def generate_real_data_scenarios(config: AHSDConfig, n_scenarios: int = 200) -> List[Dict]:
    """Generate scenarios using real GWTC data backgrounds."""
    
    logging.info("Generating real data injection scenarios...")
    
    # Load real GWTC data
    data_loader = GWTCDataLoader("data/raw")
    events_df = data_loader.get_gwtc_events()
    
    if events_df.empty:
        logging.warning("No GWTC events loaded, skipping real data scenarios")
        return []
    
    # Create synthetic overlapping scenarios
    overlap_scenarios = data_loader.create_synthetic_overlaps(events_df, n_scenarios)
    
    # Setup injection
    injector = RealDataSignalInjector(config)
    preprocessor = DataPreprocessor(config)
    
    real_scenarios = []
    
    for scenario in tqdm(overlap_scenarios, desc="Processing real data scenarios"):
        try:
            # Load background strain
            strain_data = data_loader.load_strain_for_overlap(scenario)
            
            if not strain_data:
                continue
            
            # Get signal parameters from events
            signal_parameters = []
            for event in scenario['events']:
                params = {
                    'mass_1': event.get('mass_1_source', 35.0),
                    'mass_2': event.get('mass_2_source', 30.0),
                    'luminosity_distance': event.get('luminosity_distance', 500.0),
                    'geocent_time': np.random.uniform(-0.5, 0.5),
                    'ra': np.random.uniform(0, 2*np.pi),
                    'dec': np.arcsin(np.random.uniform(-1, 1)),
                    'theta_jn': np.arccos(np.random.uniform(-1, 1)),
                    'psi': np.random.uniform(0, np.pi),
                    'phase': np.random.uniform(0, 2*np.pi),
                    'a_1': np.random.uniform(0, 0.5),
                    'a_2': np.random.uniform(0, 0.5),
                    'tilt_1': np.arccos(np.random.uniform(-1, 1)),
                    'tilt_2': np.arccos(np.random.uniform(-1, 1)),
                    'phi_12': np.random.uniform(0, 2*np.pi),
                    'phi_jl': np.random.uniform(0, 2*np.pi),
                    'signal_id': len(signal_parameters)
                }
                signal_parameters.append(params)
            
            # Create injection
            target_snrs = [np.random.uniform(10, 20) for _ in signal_parameters]
            injected_data, signal_contributions = injector.create_overlapping_injection(
                strain_data, signal_parameters, target_snrs
            )
            
            # Preprocess
            processed_data = preprocessor.preprocess(injected_data)
            
            # Create training scenario
            real_scenario = {
                'scenario_id': scenario['scenario_id'] + 10000,  # Offset for real data
                'true_parameters': signal_parameters,
                'injected_data': processed_data,
                'background_data': strain_data,
                'signal_contributions': signal_contributions,
                'n_signals': len(signal_parameters),
                'source_events': scenario['events'],
                'data_type': 'real_background'
            }
            
            real_scenarios.append(real_scenario)
            
        except Exception as e:
            logging.warning(f"Failed to process real scenario {scenario['scenario_id']}: {e}")
            continue
    
    logging.info(f"Generated {len(real_scenarios)} real data scenarios")
    return real_scenarios

def run_baseline_analysis(scenarios: List[Dict], config: AHSDConfig) -> List[Dict]:
    """Run baseline hierarchical subtraction analysis."""
    
    from src.ahsd.evaluation.benchmarks import BaselineHierarchicalSubtraction
    
    logging.info("Running baseline analysis for comparison...")
    
    baseline = BaselineHierarchicalSubtraction(config)
    results = []
    
    for scenario in tqdm(scenarios, desc="Baseline analysis"):
        try:
            # Run baseline analysis
            baseline_result = baseline.run_analysis(
                scenario['injected_data'],
                scenario['n_signals']
            )
            
            # Compute biases against ground truth
            biases = compute_parameter_biases(
                scenario['true_parameters'],
                baseline_result['extracted_parameters']
            )
            
            results.append({
                'scenario_id': scenario['scenario_id'],
                'baseline_result': baseline_result,
                'baseline_biases': biases,
                'n_signals': scenario['n_signals'],
                'data_type': scenario.get('data_type', 'simulated')
            })
            
        except Exception as e:
            logging.warning(f"Baseline analysis failed for scenario {scenario['scenario_id']}: {e}")
            continue
    
    return results

def compute_parameter_biases(true_params: List[Dict], estimated_params: List[Dict]) -> List[Dict]:
    """Compute parameter estimation biases."""
    
    biases = []
    param_names = ['mass_1', 'mass_2', 'luminosity_distance', 'geocent_time', 'ra', 'dec']
    
    for i in range(len(true_params)):
        if i < len(estimated_params):
            param_bias = {}
            true_param = true_params[i]
            est_param = estimated_params[i]
            
            for param_name in param_names:
                if param_name in true_param and param_name in est_param:
                    true_val = true_param[param_name]
                    
                    if isinstance(est_param[param_name], dict):
                        est_val = est_param[param_name]['median']
                        std_val = est_param[param_name]['std']
                    else:
                        est_val = est_param[param_name]
                        std_val = abs(est_val * 0.1)  # Assume 10% uncertainty
                    
                    # Normalized bias
                    if std_val > 0:
                        bias = (est_val - true_val) / std_val
                    else:
                        bias = est_val - true_val
                        
                    param_bias[param_name] = bias
                    
            biases.append(param_bias)
        else:
            biases.append({})  # Signal not recovered
            
    return biases

def save_training_data(scenarios: List[Dict], baseline_results: List[Dict], output_dir: Path):
    """Save all training data."""
    
    # Save scenarios
    with open(output_dir / 'training_scenarios.pkl', 'wb') as f:
        pickle.dump(scenarios, f)
    
    # Save baseline results
    with open(output_dir / 'baseline_results.pkl', 'wb') as f:
        pickle.dump(baseline_results, f)
    
    # Save dataset statistics
    stats = {
        'total_scenarios': len(scenarios),
        'simulated_scenarios': len([s for s in scenarios if s.get('data_type', 'simulated') == 'simulated']),
        'real_data_scenarios': len([s for s in scenarios if s.get('data_type') == 'real_background']),
        'signal_distribution': {
            '2_signals': len([s for s in scenarios if s['n_signals'] == 2]),
            '3_signals': len([s for s in scenarios if s['n_signals'] == 3]),
            '4_signals': len([s for s in scenarios if s['n_signals'] == 4]),
            '5_signals': len([s for s in scenarios if s['n_signals'] == 5]),
        }
    }
    
    with open(output_dir / 'dataset_statistics.yaml', 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    logging.info(f"Training data saved to {output_dir}")
    logging.info(f"Dataset statistics: {stats}")

def main():
    parser = argparse.ArgumentParser(description='Generate AHSD training data')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--n_simulated', type=int, default=500, help='Number of simulated scenarios')
    parser.add_argument('--n_real', type=int, default=200, help='Number of real data scenarios')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline analysis')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    config = AHSDConfig.from_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate simulated scenarios
    simulated_scenarios = generate_simulated_scenarios(config, args.n_simulated)
    
    # Generate real data scenarios
    real_scenarios = generate_real_data_scenarios(config, args.n_real)
    
    # Combine scenarios
    all_scenarios = simulated_scenarios + real_scenarios
    
    if not all_scenarios:
        logging.error("No scenarios generated!")
        return
    
    # Run baseline analysis
    if not args.skip_baseline:
        baseline_results = run_baseline_analysis(all_scenarios, config)
    else:
        baseline_results = []
    
    # Save data
    save_training_data(all_scenarios, baseline_results, output_dir)
    
    logging.info("Phase 1 data generation completed successfully!")

if __name__ == '__main__':
    main()
