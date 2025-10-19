#!/usr/bin/env python3
"""
Complete Dataset Generation Script for AHSD Overlapping GW Analysis
====================================================================

Generates production-ready datasets with:
- Configurable overlapping scenarios
- Multiple detector networks
- Realistic astrophysical distributions
- Quality validation
- Multiple output formats

Usage:
    python generate_dataset.py --config config.yaml
    python generate_dataset.py --n-samples 10000 --output-dir data/my_dataset
    ahsd-generate --n-samples 10000 --overlap-fraction 0.1
"""

import argparse
import logging
import sys
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# Import AHSD data modules
from ahsd.data import GWDatasetGenerator
from ahsd.data.config import (
    SAMPLE_RATE, DURATION, DETECTORS,
    EVENT_TYPE_DISTRIBUTION, SNR_DISTRIBUTION,
    OVERLAP_FRACTION, EDGE_CASE_FRACTION,SNR_RANGES
)
from ahsd.data.io_utils import DatasetReader, MetadataManager


def setup_logging(output_dir: Path, verbose: bool = False):
    """Setup comprehensive logging"""
    
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"generation_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    
    return logger


def load_config(config_file: Optional[str]) -> Dict:
    """Load configuration from YAML file"""
    
    if config_file is None:
        return {}
    
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: Dict) -> Dict:
    """Validate and fill in default values"""
    
    validated = {
        'n_samples': config.get('n_samples', 1000),
        'sample_rate': config.get('sample_rate', SAMPLE_RATE),
        'duration': config.get('duration', DURATION),
        'detectors': config.get('detectors', DETECTORS),
        'output_dir': config.get('output_dir', 'data/output'),
        'output_format': config.get('output_format', 'pkl_compressed'),
        'overlap_fraction': config.get('overlap_fraction', OVERLAP_FRACTION),
        'edge_case_fraction': config.get('edge_case_fraction', EDGE_CASE_FRACTION),
        'save_batch_size': config.get('save_batch_size', 100),
        'add_glitches': config.get('add_glitches', True),
        'preprocess': config.get('preprocess', True),
        'save_complete': config.get('save_complete', True),
        'validate_output': config.get('validate_output', True),
        'random_seed': config.get('random_seed', None),
        'event_distribution': config.get('event_distribution', EVENT_TYPE_DISTRIBUTION),
        'snr_distribution': config.get('snr_distribution', SNR_DISTRIBUTION),
        'create_splits': config.get('create_splits', True),
        'train_frac': config.get('train_frac', 0.8),
        'val_frac': config.get('val_frac', 0.1),
        'test_frac': config.get('test_frac', 0.1),
    }
    
    # Validate ranges
    if validated['n_samples'] < 1:
        raise ValueError("n_samples must be >= 1")
    
    if not 0 <= validated['overlap_fraction'] <= 1:
        raise ValueError("overlap_fraction must be in [0, 1]")
    
    if validated['output_format'] not in ['hdf5', 'pkl', 'pkl_compressed', 'both']:
        raise ValueError(f"Invalid output_format: {validated['output_format']}")
    
    total_frac = validated['train_frac'] + validated['val_frac'] + validated['test_frac']
    if abs(total_frac - 1.0) > 0.01:
        raise ValueError(f"Split fractions must sum to 1.0, got {total_frac}")
    
    return validated


def apply_custom_distributions(config: Dict):
    """Apply custom event and SNR distributions if specified"""
    
    if 'event_distribution' in config:
        import ahsd.data.config as data_config
        data_config.EVENT_TYPE_DISTRIBUTION = config['event_distribution']
        logging.info(f"Applied custom event distribution: {config['event_distribution']}")
    
    if 'snr_distribution' in config:
        import ahsd.data.config as data_config
        data_config.SNR_DISTRIBUTION = config['snr_distribution']
        logging.info(f"Applied custom SNR distribution: {config['snr_distribution']}")


def generate_dataset_from_config(config: Dict) -> Dict:
    """
    Main dataset generation function with comprehensive logging
    
    Args:
        config: Validated configuration dictionary
        
    Returns:
        Generation summary dictionary
    """
    
    logger = logging.getLogger(__name__)
    
    # ========================================================================
    # CONFIGURATION DISPLAY
    # ========================================================================
    logger.info("=" * 70)
    logger.info("DATASET GENERATION CONFIGURATION")
    logger.info("=" * 70)
    
    logger.info("")
    logger.info("Dataset Parameters:")
    logger.info(f"  Total samples:        {config['n_samples']:,}")
    logger.info(f"  Overlap fraction:     {config['overlap_fraction']:.1%} ({int(config['n_samples'] * config['overlap_fraction'])} samples)")
    logger.info(f"  Edge case fraction:   {config['edge_case_fraction']:.1%} ({int(config['n_samples'] * config['edge_case_fraction'])} samples)")
    logger.info(f"  Save batch size:      {config['save_batch_size']}")
    
    logger.info("")
    logger.info("Signal Processing:")
    logger.info(f"  Sample rate:          {config['sample_rate']} Hz")
    logger.info(f"  Duration:             {config['duration']} seconds")
    logger.info(f"  Add glitches:         {config['add_glitches']}")
    logger.info(f"  Preprocessing:        {config['preprocess']}")
    
    logger.info("")
    logger.info("Detector Network:")
    logger.info(f"  Detectors:            {', '.join(config['detectors'])}")
    
    logger.info("")
    logger.info("Event Type Distribution:")
    for event_type, fraction in config.get('event_distribution', {}).items():
        expected_count = int(config['n_samples'] * fraction * (1 - config['overlap_fraction']))
        logger.info(f"  {event_type:6s}:  {fraction:5.1%}  (~{expected_count:,} samples)")
    
    logger.info("")
    logger.info("SNR Distribution:")
    for snr_regime, fraction in config.get('snr_distribution', {}).items():
        snr_range = SNR_RANGES.get(snr_regime, (0, 0))
        expected_count = int(config['n_samples'] * fraction)
        logger.info(f"  {snr_regime:6s}:  {fraction:5.1%}  (SNR {snr_range[0]}-{snr_range[1]}, ~{expected_count:,} samples)")
    
    logger.info("")
    logger.info("Train/Val/Test Splits:")
    logger.info(f"  Create splits:        {config.get('create_splits', True)}")
    if config.get('create_splits', True):
        logger.info(f"  Train fraction:       {config['train_frac']:.1%} ({int(config['n_samples'] * config['train_frac'])} samples)")
        logger.info(f"  Val fraction:         {config['val_frac']:.1%} ({int(config['n_samples'] * config['val_frac'])} samples)")
        logger.info(f"  Test fraction:        {config['test_frac']:.1%} ({int(config['n_samples'] * config['test_frac'])} samples)")
        logger.info(f"  Chunk size:           {config.get('chunk_size', 100)}")
        
        aug_k = config.get('noise_augmentation_k', 1)
        if aug_k > 1:
            train_original = int(config['n_samples'] * config['train_frac'])
            train_augmented = train_original * aug_k
            logger.info(f"  Train augmentation:   {aug_k}x ({train_original:,} → {train_augmented:,} samples)")
    
    logger.info("")
    logger.info("Output Configuration:")
    logger.info(f"  Output directory:     {config['output_dir']}")
    logger.info(f"  Output format:        {config['output_format']}")
    logger.info(f"  Save complete file:   {config.get('save_complete', True)}")
    logger.info(f"  Validate output:      {config.get('validate_output', True)}")
    
    if config.get('random_seed') is not None:
        logger.info(f"  Random seed:          {config['random_seed']}")
    
    logger.info("=" * 70)
    logger.info("")
    
    # Set random seed for reproducibility
    if config['random_seed'] is not None:
        np.random.seed(config['random_seed'])
        logger.info(f"✓ Random seed set to: {config['random_seed']}")
    
    # Apply custom distributions
    apply_custom_distributions(config)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = output_dir / 'generation_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"✓ Configuration saved to: {config_file}")
    logger.info("")
    
    # Initialize generator
    logger.info("=" * 70)
    logger.info("INITIALIZING DATASET GENERATOR")
    logger.info("=" * 70)
    
    generator = GWDatasetGenerator(
        output_dir=str(output_dir),
        sample_rate=config['sample_rate'],
        duration=config['duration'],
        detectors=config['detectors'],
        output_format=config['output_format']
    )
    
    logger.info("")
    
    # Estimate generation time
    estimated_time_per_sample = 0.5  # seconds (conservative estimate)
    total_estimated_time = config['n_samples'] * estimated_time_per_sample
    hours = int(total_estimated_time // 3600)
    minutes = int((total_estimated_time % 3600) // 60)
    
    logger.info("=" * 70)
    logger.info("STARTING DATASET GENERATION")
    logger.info("=" * 70)
    logger.info(f"Estimated time: ~{hours}h {minutes}m (assuming {estimated_time_per_sample}s per sample)")
    logger.info(f"Actual time may vary based on system performance")
    logger.info("=" * 70)
    logger.info("")
    
    start_time = time.time()
    
    # Track progress milestones
    milestones = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    milestone_idx = 0
    
    try:
        summary = generator.generate_dataset(
            n_samples=config['n_samples'],
            overlap_fraction=config['overlap_fraction'],
            edge_case_fraction=config['edge_case_fraction'],
            save_batch_size=config['save_batch_size'],
            add_glitches=config['add_glitches'],
            preprocess=config['preprocess'],
            save_complete=config.get('save_complete', True),
            create_splits=config.get('create_splits', True),
            train_frac=config['train_frac'],
            val_frac=config['val_frac'],
            test_frac=config['test_frac'],
            chunk_size=config.get('chunk_size', 100),
            noise_augmentation_k=config.get('noise_augmentation_k', 2)
        )
        
        elapsed = time.time() - start_time
        
        # Add configuration to summary
        summary['configuration'] = config
        summary['total_time_seconds'] = elapsed
        summary['success'] = True
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("DATASET GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"✓ Generated {summary['n_samples']:,} samples")
        logger.info(f"✓ Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m) @ {summary['samples_per_second']:.2f} samples/s")
        logger.info(f"✓ Output directory: {summary['output_dir']}")
        logger.info(f"✓ Batches created: {summary['n_batches']}")
        
        if 'splits' in summary and summary['splits']:
            logger.info("")
            logger.info("Split Summary:")
            splits = summary['splits']
            if isinstance(splits['train'], dict):
                logger.info(f"  Train:  {splits['train']['original']:,} original → {splits['train']['augmented']:,} augmented ({splits['train']['augmentation_factor']}x)")
            else:
                logger.info(f"  Train:  {splits['train']:,} samples")
            logger.info(f"  Validation:    {splits['validation']:,} samples")
            logger.info(f"  Test:   {splits['test']:,} samples")
        
        logger.info("=" * 70)
        logger.info("")
        
        # Validate output if requested
        if config['validate_output']:
            logger.info("=" * 70)
            logger.info("VALIDATING GENERATED DATASET")
            logger.info("=" * 70)
            validation_report = validate_generated_dataset(output_dir, config)
            summary['validation'] = validation_report
            
            if validation_report['passed']:
                logger.info("✓ Validation PASSED")
            else:
                logger.warning(f"✗ Validation FAILED")
                if validation_report['errors']:
                    logger.error("Errors:")
                    for error in validation_report['errors']:
                        logger.error(f"  - {error}")
            
            if validation_report['warnings']:
                logger.warning("Warnings:")
                for warning in validation_report['warnings']:
                    logger.warning(f"  - {warning}")
            
            logger.info("=" * 70)
            logger.info("")
        
        # Save final summary
        summary_file = output_dir / 'generation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"✓ Summary saved to: {summary_file}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}", exc_info=True)
        raise

def validate_generated_dataset(output_dir: Path, config: Dict) -> Dict:
    """
    Validate generated dataset
    
    Checks:
    - File existence
    - Sample counts
    - Data integrity
    - Parameter distributions
    """
    
    logger = logging.getLogger(__name__)
    reader = DatasetReader()
    
    validation_report = {
        'passed': True,
        'errors': [],
        'warnings': [],
        'checks': {}
    }
    
    # Check 1: Output files exist
    logger.info("  [1/5] Checking output files...")
    
    if config['output_format'] in ['pkl', 'pkl_compressed']:
        if config['save_complete']:
            complete_file = output_dir / 'complete_dataset.pkl'
            if config['output_format'] == 'pkl_compressed':
                complete_file = output_dir / 'complete_dataset.pkl.gz'
            
            if complete_file.exists():
                validation_report['checks']['complete_file_exists'] = True
            else:
                validation_report['passed'] = False
                validation_report['errors'].append(f"Complete dataset file not found: {complete_file}")
    
    # Check 2: Load and verify sample count
    logger.info("  [2/5] Verifying sample count...")
    
    try:
        if config['save_complete'] and config['output_format'] in ['pkl', 'pkl_compressed']:
            dataset = reader.load_pkl(str(complete_file))
            
            actual_samples = len(dataset.get('samples', []))
            expected_samples = config['n_samples']
            
            if actual_samples == expected_samples:
                validation_report['checks']['sample_count'] = True
                logger.info(f"    ✓ Sample count matches: {actual_samples}")
            else:
                validation_report['passed'] = False
                validation_report['errors'].append(
                    f"Sample count mismatch: expected {expected_samples}, got {actual_samples}"
                )
        else:
            validation_report['checks']['sample_count'] = 'skipped'
    
    except Exception as e:
        validation_report['passed'] = False
        validation_report['errors'].append(f"Failed to load dataset: {e}")
    
    # Check 3: Verify overlap fraction
    logger.info("  [3/5] Checking overlap distribution...")
    
    try:
        if 'dataset' in locals():
            samples = dataset['samples']
            n_overlap = sum(1 for s in samples if s.get('is_overlap', False))
            actual_overlap_frac = n_overlap / len(samples) if samples else 0
            expected_overlap_frac = config['overlap_fraction']
            
            tolerance = 0.02  # 2% tolerance
            if abs(actual_overlap_frac - expected_overlap_frac) < tolerance:
                validation_report['checks']['overlap_fraction'] = True
                logger.info(f"    ✓ Overlap fraction: {actual_overlap_frac:.2%}")
            else:
                validation_report['warnings'].append(
                    f"Overlap fraction deviation: expected {expected_overlap_frac:.2%}, "
                    f"got {actual_overlap_frac:.2%}"
                )
    except Exception as e:
        validation_report['warnings'].append(f"Could not verify overlap fraction: {e}")
    
    # Check 4: Data integrity
    
    logger.info("  [4/5] Checking data integrity...")

    try:
        if 'dataset' in locals() and dataset is not None and 'samples' in dataset:
            samples = dataset['samples']
            
            integrity_checks = {
                'has_strain_data': 0,
                'has_parameters': 0,
                'has_valid_snr': 0,
                'has_detector_data': 0
            }
            
            check_count = min(100, len(samples))
            
            for i, sample in enumerate(samples[:check_count]):
                # Safety check: skip if sample is None
                if sample is None:
                    continue
                
                # Check detector data
                if 'detector_data' in sample and sample['detector_data'] is not None:
                    integrity_checks['has_detector_data'] += 1
                    
                    for det_name, det_data in sample['detector_data'].items():
                        if det_data is None:
                            continue
                        
                        if 'strain' in det_data and det_data['strain'] is not None:
                            integrity_checks['has_strain_data'] += 1
                            
                            # Check for NaN/Inf
                            strain = det_data['strain']
                            if not np.all(np.isfinite(strain)):
                                validation_report['errors'].append(
                                    f"Sample {sample.get('sample_id', i)} contains NaN/Inf in {det_name}"
                                )
                                validation_report['passed'] = False
                
                # Check parameters
                if 'parameters' in sample and sample['parameters'] is not None:
                    integrity_checks['has_parameters'] += 1
                    
                    # Check SNR if available
                    params = sample['parameters']
                    if isinstance(params, list):
                        params = params[0] if len(params) > 0 else None
                    
                    if params is not None and 'target_snr' in params:
                        if params['target_snr'] > 0:
                            integrity_checks['has_valid_snr'] += 1
            
            validation_report['checks']['integrity'] = integrity_checks
            logger.info(f"    ✓ Integrity checks passed ({check_count} samples checked)")
        else:
            validation_report['warnings'].append("Could not access dataset samples for integrity check")

    except Exception as e:
        validation_report['warnings'].append(f"Data integrity check failed: {e}")

    # Check 5: Statistics
    logger.info("  [5/5] Computing statistics...")
    
    try:
        if 'dataset' in locals() and 'statistics' in dataset:
            stats = dataset['statistics']
            validation_report['checks']['statistics'] = stats
            
            logger.info(f"    ✓ Event types: {stats.get('event_types', {})}")
            logger.info(f"    ✓ SNR range: {stats.get('snr_distribution', {})}")
    
    except Exception as e:
        validation_report['warnings'].append(f"Statistics computation failed: {e}")
    
    return validation_report


def parse_arguments():
    """Parse command-line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Generate AHSD Overlapping GW Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from config file
  %(prog)s --config my_config.yaml
  
  # Quick generation with defaults
  %(prog)s --n-samples 1000 --output-dir data/test
  
  # Production dataset
  %(prog)s --n-samples 50000 --overlap-fraction 0.1 --output-format pkl_compressed
  
  # Custom detector network
  %(prog)s --n-samples 5000 --detectors H1 L1 V1
        """
    )
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    
    # Dataset parameters
    parser.add_argument('--n-samples', type=int, help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, help='Output directory path')
    parser.add_argument('--output-format', type=str, 
                       choices=['hdf5', 'pkl', 'pkl_compressed', 'both'],
                       help='Output file format')
    
    # Signal parameters
    parser.add_argument('--overlap-fraction', type=float, 
                       help='Fraction of samples with overlapping signals (0-1)')
    parser.add_argument('--edge-case-fraction', type=float,
                       help='Fraction of edge case samples (0-1)')
    
    # Detector configuration
    parser.add_argument('--detectors', nargs='+', 
                       choices=['H1', 'L1', 'V1'],
                       help='Detector network (e.g., H1 L1)')
    parser.add_argument('--sample-rate', type=int, help='Sample rate in Hz')
    parser.add_argument('--duration', type=float, help='Sample duration in seconds')
    
    # Processing options
    parser.add_argument('--no-glitches', action='store_true',
                       help='Disable glitch injection')
    parser.add_argument('--no-preprocess', action='store_true',
                       help='Disable preprocessing')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip output validation')
    parser.add_argument('--no-save-complete', action='store_true',
                       help='Do not save complete dataset file')
    
    # Advanced options
    parser.add_argument('--save-batch-size', type=int, help='Batch size for saving')
    parser.add_argument('--random-seed', type=int, help='Random seed for reproducibility')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal logging')
    
    #splits
    parser.add_argument('--no-splits', action='store_true',
                       help='Do not create train/val/test splits')
    parser.add_argument('--train-frac', type=float, default=0.8,
                       help='Training set fraction (default: 0.8)')
    parser.add_argument('--val-frac', type=float, default=0.1,
                       help='Validation set fraction (default: 0.1)')
    parser.add_argument('--test-frac', type=float, default=0.1,
                       help='Test set fraction (default: 0.1)')
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='Samples per chunk file (default: 100)')
    parser.add_argument('--noise-augmentation', type=int, default=1,
                       help='Noise augmentation factor for training set (default: 1, no augmentation)')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    
    args = parse_arguments()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = {}
    
    # Override with command-line arguments
    if args.n_samples:
        config['n_samples'] = args.n_samples
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.output_format:
        config['output_format'] = args.output_format
    if args.overlap_fraction is not None:
        config['overlap_fraction'] = args.overlap_fraction
    if args.edge_case_fraction is not None:
        config['edge_case_fraction'] = args.edge_case_fraction
    if args.detectors:
        config['detectors'] = args.detectors
    if args.sample_rate:
        config['sample_rate'] = args.sample_rate
    if args.duration:
        config['duration'] = args.duration
    if args.save_batch_size:
        config['save_batch_size'] = args.save_batch_size
    if args.random_seed is not None:
        config['random_seed'] = args.random_seed
    
    # Processing flags
    if args.no_glitches:
        config['add_glitches'] = False
    if args.no_preprocess:
        config['preprocess'] = False
    if args.no_validate:
        config['validate_output'] = False
    if args.no_save_complete:
        config['save_complete'] = False
    
    # Validate configuration
    try:
        config = validate_config(config)
    except Exception as e:
        print(f"Error: Invalid configuration - {e}")
        sys.exit(1)
    
    # Setup logging
    output_dir = Path(config['output_dir'])
    logger = setup_logging(output_dir, verbose=args.verbose)
    
    # Generate dataset
    try:
        summary = generate_dataset_from_config(config)
        
        logger.info("=" * 70)
        logger.info("SUCCESS: Dataset generation completed successfully!")
        logger.info("=" * 70)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("\nDataset generation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"FAILED: Dataset generation encountered an error", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
