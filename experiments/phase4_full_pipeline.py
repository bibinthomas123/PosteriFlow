from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
#!/usr/bin/env python3
"""
Complete AHSD pipeline using real LIGO-Virgo data from GWTC-4.0
"""

import argparse
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import wandb

# AHSD imports
from ahsd.utils.config import AHSDConfig
from ahsd.data.gwtc_loader import GWTCDataLoader
from ahsd.data.preprocessing import DataPreprocessor
from ahsd.data.injection import RealDataSignalInjector
from ahsd.core.ahsd_pipeline import AHSDPipeline
from ahsd.utils.config import compute_detailed_metrics, compute_summary_metrics

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ahsd_real_data.log'),
            logging.StreamHandler()
        ]
    )

def load_real_gwtc_data(n_scenarios: int = 50) -> Tuple[List[Dict], pd.DataFrame]:
    """Load and prepare real GWTC data for AHSD training"""
    
    logging.info("Loading GWTC-4.0 data...")
    
    # Initialize data loader
    data_loader = GWTCDataLoader()
    
    # Get GWTC events
    events_df = data_loader.get_gwtc_events()
    logging.info(f"Loaded {len(events_df)} GWTC events")
    
    # Create synthetic overlapping scenarios using real event parameters
    overlapping_scenarios = data_loader.create_synthetic_overlaps(events_df, n_overlaps=n_scenarios)
    
    # Load strain data for each scenario
    processed_scenarios = []
    
    for scenario in tqdm(overlapping_scenarios, desc="Processing scenarios"):
        try:
            # Load background strain data
            strain_data = data_loader.load_strain_for_overlap(
                scenario, 
                detectors=['H1', 'L1'], 
                duration=4,
                sampling_rate=4096
            )
            
            if strain_data:
                scenario['background_data'] = strain_data
                processed_scenarios.append(scenario)
            
        except Exception as e:
            logging.warning(f"Failed to load data for scenario {scenario['scenario_id']}: {e}")
            continue
    
    logging.info(f"Successfully processed {len(processed_scenarios)} scenarios with real data")
    return processed_scenarios, events_df

def create_injection_scenarios(scenarios: List[Dict], config: AHSDConfig) -> List[Dict]:
    """Create injection scenarios using real background data"""
    
    logging.info("Creating signal injection scenarios...")
    
    injector = RealDataSignalInjector(config)
    preprocessor = DataPreprocessor(config)
    
    injection_scenarios = []
    
    for scenario in tqdm(scenarios, desc="Creating injections"):
        try:
            # Get true signal parameters from GWTC events
            true_parameters = []
            for event in scenario['events']:
                params = {
                    'mass_1': event.get('mass_1_source', 35.0),
                    'mass_2': event.get('mass_2_source', 30.0),
                    'luminosity_distance': event.get('luminosity_distance', 500.0),
                    'geocent_time': np.random.uniform(-0.5, 0.5),  # Random offset
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
                    'signal_id': len(true_parameters)
                }
                true_parameters.append(params)
            
            # Create injection
            background_data = scenario['background_data']
            target_snrs = [np.random.uniform(10, 20) for _ in true_parameters]
            
            injected_data, signal_contributions = injector.create_overlapping_injection(
                background_data, true_parameters, target_snrs
            )
            
            # Preprocess injected data
            processed_data = preprocessor.preprocess(injected_data)
            
            # Create training scenario
            training_scenario = {
                'scenario_id': scenario['scenario_id'],
                'true_parameters': true_parameters,
                'injected_data': processed_data,
                'background_data': background_data,
                'signal_contributions': signal_contributions,
                'target_snrs': target_snrs,
                'n_signals': len(true_parameters),
                'source_events': scenario['events']
            }
            
            injection_scenarios.append(training_scenario)
            
        except Exception as e:
            logging.error(f"Failed to create injection for scenario {scenario['scenario_id']}: {e}")
            continue
    
    logging.info(f"Created {len(injection_scenarios)} injection scenarios")
    return injection_scenarios

def prepare_detection_candidates(scenario: Dict) -> List[Dict]:
    """Prepare realistic detection candidates from injection scenario"""
    
    candidates = []
    
    for i, true_params in enumerate(scenario['true_parameters']):
        # Add realistic noise to parameters to simulate detection uncertainty
        candidate = {}
        
        for param_name, true_value in true_params.items():
            if param_name in ['mass_1', 'mass_2']:
                # Mass uncertainty ~10-20%
                std = true_value * np.random.uniform(0.1, 0.2)
                candidate[param_name] = np.random.normal(true_value, std)
            elif param_name == 'luminosity_distance':
                # Distance uncertainty ~20-50%
                std = true_value * np.random.uniform(0.2, 0.5)
                candidate[param_name] = max(50.0, np.random.normal(true_value, std))
            elif param_name in ['ra', 'dec']:
                # Sky location uncertainty
                std = np.random.uniform(0.1, 0.5)
                candidate[param_name] = np.random.normal(true_value, std)
            else:
                # Other parameters with moderate uncertainty
                std = np.random.uniform(0.1, 0.3)
                candidate[param_name] = np.random.normal(true_value, std)
        
        # Add detection-specific parameters
        target_snr = scenario.get('target_snrs', [15.0])[i] if i < len(scenario.get('target_snrs', [])) else 15.0
        candidate.update({
            'network_snr': target_snr + np.random.normal(0, 2),
            'coherent_snr': target_snr * np.random.uniform(0.8, 0.95),
            'null_snr': np.random.normal(1.0, 0.3),
            'detection_id': i,
            'chirp_mass_source': (true_params['mass_1'] * true_params['mass_2'])**(3/5) / (true_params['mass_1'] + true_params['mass_2'])**(1/5),
            'total_mass_source': true_params['mass_1'] + true_params['mass_2']
        })
        
        candidates.append(candidate)
    
    return candidates

def train_ahsd_on_real_data(config: AHSDConfig, scenarios: List[Dict], output_dir: Path):
    """Train AHSD pipeline on real data scenarios"""
    
    logging.info("Training AHSD pipeline on real data...")
    
    # Initialize pipeline
    pipeline = AHSDPipeline(config)
    
    # Collect training data for bias corrector
    bias_training_data = []
    
    # Split scenarios for training/testing
    n_train = int(0.7 * len(scenarios))
    train_scenarios = scenarios[:n_train]
    test_scenarios = scenarios[n_train:]
    
    logging.info(f"Using {len(train_scenarios)} scenarios for training, {len(test_scenarios)} for testing")
    
    # Generate training data for bias correction
    logging.info("Generating bias correction training data...")
    
    for scenario in tqdm(train_scenarios[:20], desc="Bias training data"):  # Use subset for training
        try:
            # Prepare detection candidates
            initial_detections = prepare_detection_candidates(scenario)
            
            # Run pipeline without bias correction to get raw results
            pipeline.bias_corrector.is_trained = False
            
            result = pipeline.decompose_overlapping_signals(
                scenario['injected_data'],
                initial_detections
            )
            
            # Compute training examples for bias correction
            true_params = scenario['true_parameters']
            extracted_signals = result['extracted_signals']
            
            for i, (true_param, extracted_signal) in enumerate(zip(true_params, extracted_signals)):
                bias_training_example = {
                    'true_parameters': true_param,
                    'extracted_parameters': extracted_signal.get('posterior_summary', {}),
                    'extraction_position': i,
                    'total_signals': len(true_params),
                    'signal_quality': extracted_signal.get('signal_quality', 0.5),
                    'estimated_snr': scenario.get('target_snrs', [15.0])[i] if i < len(scenario.get('target_snrs', [])) else 15.0
                }
                bias_training_data.append(bias_training_example)
        
        except Exception as e:
            logging.error(f"Failed to process training scenario {scenario['scenario_id']}: {e}")
            continue
    
    # Train bias corrector
    if bias_training_data:
        logging.info(f"Training bias corrector on {len(bias_training_data)} examples...")
        pipeline.bias_corrector.train_bias_estimator(bias_training_data)
        
        # Save trained model
        bias_model_path = output_dir / 'bias_corrector_model.pth'
        pipeline.bias_corrector.save_model(str(bias_model_path))
    
    return pipeline, test_scenarios

def evaluate_ahsd_performance(pipeline: AHSDPipeline, test_scenarios: List[Dict]) -> Dict:
    """Evaluate AHSD performance on test scenarios"""
    
    logging.info("Evaluating AHSD performance...")
    
    results = []
    
    for scenario in tqdm(test_scenarios, desc="Evaluating"):
        try:
            # Set ground truth for evaluation
            pipeline.set_ground_truth({'signals': scenario['true_parameters']})
            
            # Prepare detection candidates
            initial_detections = prepare_detection_candidates(scenario)
            
            # Run AHSD analysis
            result = pipeline.decompose_overlapping_signals(
                scenario['injected_data'],
                initial_detections
            )
            
            # Compute detailed metrics
            metrics = compute_detailed_metrics(
                scenario['true_parameters'],
                result,
                'AHSD'
            )
            
            results.append({
                'scenario_id': scenario['scenario_id'],
                'result': result,
                'metrics': metrics
            })
            
        except Exception as e:
            logging.error(f"Evaluation failed for scenario {scenario['scenario_id']}: {e}")
            continue
    
    # Compute summary metrics
    evaluation_summary = {
        'ahsd_results': results
    }
    
    summary_metrics = compute_summary_metrics(evaluation_summary)
    
    return {
        'detailed_results': results,
        'summary_metrics': summary_metrics
    }

def main():
    parser = argparse.ArgumentParser(description='AHSD Real Data Pipeline')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--n_scenarios', type=int, default=50, help='Number of scenarios to process')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project="ahsd-real-data", config=vars(args))
    
    # Load configuration
    config = AHSDConfig.from_yaml(args.config)
    
    # Load real GWTC data
    scenarios, events_df = load_real_gwtc_data(args.n_scenarios)
    
    # Save GWTC events for reference
    events_df.to_csv(output_dir / 'gwtc_events.csv', index=False)
    
    # Create injection scenarios
    injection_scenarios = create_injection_scenarios(scenarios, config)
    
    # Save scenarios
    with open(output_dir / 'injection_scenarios.pkl', 'wb') as f:
        pickle.dump(injection_scenarios, f)
    
    # Train AHSD pipeline
    trained_pipeline, test_scenarios = train_ahsd_on_real_data(config, injection_scenarios, output_dir)
    
    # Save trained pipeline
    trained_pipeline.save_pipeline_state(str(output_dir / 'trained_ahsd_pipeline.pth'))
    
    # Evaluate performance
    evaluation_results = evaluate_ahsd_performance(trained_pipeline, test_scenarios)
    
    # Save results
    with open(output_dir / 'evaluation_results.pkl', 'wb') as f:
        pickle.dump(evaluation_results, f)
    
    # Generate report
    generate_performance_report(evaluation_results, output_dir)
    
    logging.info("AHSD real data pipeline completed successfully!")

def generate_performance_report(results: Dict, output_dir: Path):
    """Generate comprehensive performance report"""
    
    summary = results['summary_metrics']
    
    report_lines = [
        "AHSD Real Data Performance Report",
        "=" * 50,
        "",
        f"Total Test Scenarios: {len(results['detailed_results'])}",
        "",
        "Performance Metrics:",
    ]
    
    if 'ahsd' in summary:
        metrics = summary['ahsd']
        report_lines.extend([
            f"  Mean Parameter Bias: {metrics['mean_bias']:.3f}",
            f"  Median Parameter Bias: {metrics['median_bias']:.3f}",
            f"  Mean Coverage Probability: {metrics['mean_coverage']:.3f}",
            f"  Mean Extraction Time: {metrics['mean_extraction_time']:.2f}s",
            f"  Mean Recovery Rate: {metrics['mean_recovery_rate']:.3f}",
        ])
    
    # Per-scenario breakdown
    report_lines.extend([
        "",
        "Per-Scenario Results:",
        "-" * 30
    ])
    
    for result in results['detailed_results'][:10]:  # Show first 10
        metrics = result['metrics']
        report_lines.append(
            f"Scenario {result['scenario_id']}: "
            f"Bias={metrics['mean_parameter_bias']:.3f}, "
            f"Recovery={metrics['recovery_rate']:.2f}, "
            f"Time={metrics['extraction_time']:.2f}s"
        )
    
    # Save report
    with open(output_dir / 'performance_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))

if __name__ == '__main__':
    main()
