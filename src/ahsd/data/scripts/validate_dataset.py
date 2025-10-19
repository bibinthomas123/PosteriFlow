#!/usr/bin/env python3
"""
Dataset Validation Script for AHSD
Validates generated datasets and provides comprehensive quality reports
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

from ahsd.data.io_utils import DatasetReader


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def validate_dataset(dataset_dir: str, verbose: bool = False) -> Dict:
    """
    Validate dataset and generate comprehensive report
    
    Args:
        dataset_dir: Path to dataset directory
        verbose: Verbose output
        
    Returns:
        Validation report dictionary
    """
    
    logger = setup_logging(verbose)
    reader = DatasetReader()
    
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return {'passed': False, 'error': 'Directory not found'}
    
    logger.info("=" * 70)
    logger.info("AHSD DATASET VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info("")
    
    report = {
        'dataset_dir': str(dataset_dir),
        'passed': True,
        'errors': [],
        'warnings': [],
        'checks': {}
    }
    
    # Check 1: Find dataset files
    logger.info("[1/7] Locating dataset files...")
    
    pkl_files = list(dataset_path.glob("*.pkl")) + list(dataset_path.glob("*.pkl.gz"))
    h5_files = list(dataset_path.glob("*.h5"))
    batch_files = list(dataset_path.glob("batch_*.pkl")) + list(dataset_path.glob("batch_*.pkl.gz"))
    
    complete_file = None
    for f in pkl_files:
        if 'complete_dataset' in f.name:
            complete_file = f
            break
    
    report['checks']['files'] = {
        'complete_file': str(complete_file) if complete_file else None,
        'n_batch_files': len(batch_files),
        'n_pkl_files': len(pkl_files),
        'n_h5_files': len(h5_files)
    }
    
    if complete_file:
        logger.info(f"  ✓ Complete file: {complete_file.name}")
    else:
        logger.warning(f"  ⚠ No complete dataset file found")
        report['warnings'].append("No complete dataset file")
    
    logger.info(f"  ✓ Batch files: {len(batch_files)}")
    logger.info("")
    
    # Check 2: Load and validate structure
    logger.info("[2/7] Loading dataset...")
    
    dataset = None
    try:
        if complete_file:
            dataset = reader.load_pkl(str(complete_file))
            logger.info(f"  ✓ Loaded {len(dataset.get('samples', []))} samples")
        else:
            logger.warning("  ⚠ Skipping - no complete file")
            
    except Exception as e:
        logger.error(f"  ✗ Failed to load: {e}")
        report['errors'].append(f"Load failed: {e}")
        report['passed'] = False
    
    logger.info("")
    
    # Check 3: Validate metadata
    logger.info("[3/7] Validating metadata...")
    
    if dataset and 'metadata' in dataset:
        metadata = dataset['metadata']
        required_fields = ['n_samples', 'format_version']
        
        for field in required_fields:
            if field in metadata:
                logger.info(f"  ✓ {field}: {metadata[field]}")
            else:
                logger.warning(f"  ⚠ Missing metadata field: {field}")
                report['warnings'].append(f"Missing metadata: {field}")
        
        report['checks']['metadata'] = metadata
    else:
        logger.warning("  ⚠ No metadata found")
    
    logger.info("")
    
    # Check 4: Sample validation
    logger.info("[4/7] Validating samples...")
    
    if dataset and 'samples' in dataset:
        samples = dataset['samples']
        
        sample_checks = {
            'n_total': len(samples),
            'n_valid': 0,
            'n_with_strain': 0,
            'n_with_parameters': 0,
            'event_types': {},
            'issues': []
        }
        
        for i, sample in enumerate(samples[:min(1000, len(samples))]):  # Check first 1000
            if sample is None:
                sample_checks['issues'].append(f"Sample {i} is None")
                continue
            
            sample_checks['n_valid'] += 1
            
            # Check strain data
            if 'detector_data' in sample:
                has_strain = False
                for det, det_data in sample['detector_data'].items():
                    if det_data and 'strain' in det_data:
                        strain = det_data['strain']
                        if strain is not None and len(strain) > 0:
                            has_strain = True
                            
                            # Check for NaN/Inf
                            if not np.all(np.isfinite(strain)):
                                sample_checks['issues'].append(
                                    f"Sample {i} has NaN/Inf in {det}"
                                )
                
                if has_strain:
                    sample_checks['n_with_strain'] += 1
            
            # Check parameters
            if 'parameters' in sample and sample['parameters']:
                sample_checks['n_with_parameters'] += 1
            
            # Count event types
            event_type = sample.get('type', 'unknown')
            sample_checks['event_types'][event_type] = \
                sample_checks['event_types'].get(event_type, 0) + 1
        
        report['checks']['samples'] = sample_checks
        
        logger.info(f"  ✓ Valid samples: {sample_checks['n_valid']}/{sample_checks['n_total']}")
        logger.info(f"  ✓ With strain data: {sample_checks['n_with_strain']}")
        logger.info(f"  ✓ With parameters: {sample_checks['n_with_parameters']}")
        logger.info(f"  ✓ Event types: {sample_checks['event_types']}")
        
        if sample_checks['issues']:
            logger.warning(f"  ⚠ Issues found: {len(sample_checks['issues'])}")
            for issue in sample_checks['issues'][:5]:  # Show first 5
                logger.warning(f"    - {issue}")
    
    logger.info("")
    
    # Check 5: Statistics validation
    logger.info("[5/7] Checking statistics...")
    
    if dataset and 'statistics' in dataset:
        stats = dataset['statistics']
        
        logger.info(f"  ✓ Total samples: {stats.get('n_total', 'N/A')}")
        
        if 'event_types' in stats:
            logger.info("  ✓ Event distribution:")
            for event_type, count in stats['event_types'].items():
                logger.info(f"    - {event_type}: {count}")
        
        if 'overlap_statistics' in stats:
            overlap = stats['overlap_statistics']
            logger.info(f"  ✓ Overlapping: {overlap.get('n_overlapping', 0)}")
            logger.info(f"  ✓ Single: {overlap.get('n_single', 0)}")
        
        report['checks']['statistics'] = stats
    else:
        logger.warning("  ⚠ No statistics found")
    
    logger.info("")
    
    # Check 6: Data quality metrics
    logger.info("[6/7] Computing quality metrics...")
    
    if dataset and 'samples' in dataset:
        samples = dataset['samples']
        
        quality_metrics = {
            'strain_rms': [],
            'strain_max': [],
            'strain_length': []
        }
        
        for sample in samples[:100]:  # Sample first 100
            if sample and 'detector_data' in sample:
                for det, det_data in sample['detector_data'].items():
                    if det_data and 'strain' in det_data:
                        strain = det_data['strain']
                        if strain is not None and len(strain) > 0:
                            quality_metrics['strain_rms'].append(np.sqrt(np.mean(strain**2)))
                            quality_metrics['strain_max'].append(np.max(np.abs(strain)))
                            quality_metrics['strain_length'].append(len(strain))
        
        if quality_metrics['strain_rms']:
            logger.info(f"  ✓ Strain RMS (mean): {np.mean(quality_metrics['strain_rms']):.2e}")
            logger.info(f"  ✓ Strain max (mean): {np.mean(quality_metrics['strain_max']):.2e}")
            logger.info(f"  ✓ Strain length (mode): {max(set(quality_metrics['strain_length']), key=quality_metrics['strain_length'].count)}")
        
        report['checks']['quality_metrics'] = {
            'strain_rms_mean': float(np.mean(quality_metrics['strain_rms'])) if quality_metrics['strain_rms'] else None,
            'strain_max_mean': float(np.mean(quality_metrics['strain_max'])) if quality_metrics['strain_max'] else None
        }
    
    # Check 7: Checking splits (FIXED for chunked storage)
    logger.info("[7/7] Checking splits...")

    split_dirs = {
        'train': dataset_path / 'train',
        'validation': dataset_path / 'validation',
        'test': dataset_path / 'test'
    }

    splits_found = {}
    for split_name, split_dir in split_dirs.items():
        if split_dir.exists() and split_dir.is_dir():
            # Check for chunk files
            chunk_files = list(split_dir.glob('chunk_*.pkl')) + list(split_dir.glob('chunk_*.pkl.gz'))
            
            # Check for split_info.json
            split_info_file = split_dir / 'split_info.json'
            
            if chunk_files:
                logger.info(f"  ✓ {split_name} split found ({len(chunk_files)} chunks)")
                
                # Load split info if available
                if split_info_file.exists():
                    import json
                    with open(split_info_file, 'r') as f:
                        split_info = json.load(f)
                    logger.info(f"    - Total samples: {split_info.get('n_samples', 'N/A')}")
                    logger.info(f"    - Chunks: {split_info.get('n_chunks', len(chunk_files))}")
                    if split_info.get('augmentation_factor', 1) > 1:
                        logger.info(f"    - Augmentation: {split_info['augmentation_factor']}x")
                
                splits_found[split_name] = {
                    'found': True,
                    'n_chunks': len(chunk_files),
                    'has_metadata': split_info_file.exists()
                }
            else:
                logger.warning(f"  ⚠ {split_name} directory exists but no chunk files found")
                splits_found[split_name] = {'found': False, 'reason': 'no_chunks'}
        else:
            logger.warning(f"  ⚠ {split_name} split directory not found")
            splits_found[split_name] = {'found': False, 'reason': 'no_directory'}

    report['checks']['splits'] = splits_found

    # Check split_indices.json
    split_indices_file = dataset_path / 'split_indices.json'
    if split_indices_file.exists():
        logger.info("  ✓ split_indices.json found")
        import json
        with open(split_indices_file, 'r') as f:
            indices = json.load(f)
        logger.info(f"    - Train indices: {len(indices.get('train', []))}")
        logger.info(f"    - Val indices: {len(indices.get('val', []))}")
        logger.info(f"    - Test indices: {len(indices.get('test', []))}")
        if 'train_augmentation_factor' in indices:
            logger.info(f"    - Train augmentation: {indices['train_augmentation_factor']}x")
    else:
        logger.warning("  ⚠ split_indices.json not found")

    logger.info("")

    
    # Final verdict
    logger.info("=" * 70)
    if report['passed'] and not report['errors']:
        logger.info("✓ VALIDATION PASSED")
    elif report['warnings'] and not report['errors']:
        logger.info("⚠ VALIDATION PASSED WITH WARNINGS")
    else:
        logger.error("✗ VALIDATION FAILED")
        report['passed'] = False
    
    logger.info("=" * 70)
    
    if report['errors']:
        logger.error("\nErrors:")
        for error in report['errors']:
            logger.error(f"  - {error}")
    
    if report['warnings']:
        logger.warning("\nWarnings:")
        for warning in report['warnings']:
            logger.warning(f"  - {warning}")
    
    return report


def parse_arguments():
    """Parse command-line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Validate AHSD Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str,
                       help='Output path for validation report (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    
    args = parse_arguments()
    
    try:
        report = validate_dataset(args.dataset_dir, args.verbose)
        
        # Save report if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nValidation report saved to: {output_path}")
        
        # Exit with appropriate code
        sys.exit(0 if report['passed'] else 1)
        
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
