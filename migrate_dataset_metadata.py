#!/usr/bin/env python3
"""
Migrate existing dataset to add new SNR/distance metadata fields.

This script loads the old dataset chunks and adds the missing metadata fields
that were added in the critical physics fixes (Jan 27, 2026).

Usage:
    python migrate_dataset_metadata.py --input-dir data/dataset/train --output-dir data/dataset_migrated/train
"""

import argparse
import pickle
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def migrate_chunk(chunk, verbose=False):
    """Add missing metadata fields to a chunk of samples."""
    if chunk is None or not isinstance(chunk, list):
        return None
    
    migrated = []
    for sample in chunk:
        if sample is None or not isinstance(sample, dict):
            migrated.append(sample)
            continue
        
        # Get parameters to extract SNR/distance
        params_list = sample.get('parameters', [])
        if not params_list:
            migrated.append(sample)
            continue
        
        params = params_list[0] if isinstance(params_list, list) else params_list
        if not isinstance(params, dict):
            migrated.append(sample)
            continue
        
        # Add missing metadata fields if not already present
        if 'metadata' not in sample:
            sample['metadata'] = {}
        
        # Add SNR/distance to metadata if not already there
        if 'target_snr' not in sample['metadata']:
            sample['metadata']['target_snr'] = float(params.get('target_snr', 0.0))
        
        if 'luminosity_distance' not in sample['metadata']:
            sample['metadata']['luminosity_distance'] = float(params.get('luminosity_distance', 0.0))
        
        if 'chirp_mass' not in sample['metadata']:
            sample['metadata']['chirp_mass'] = float(params.get('chirp_mass', 0.0))
        
        migrated.append(sample)
    
    return migrated


def migrate_dataset(input_dir, output_dir, verbose=False):
    """Migrate all chunks in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_files = sorted(input_dir.glob('chunk_*.pkl'))
    logger.info(f"Found {len(chunk_files)} chunks to migrate")
    
    total_samples = 0
    total_migrated = 0
    
    for chunk_file in tqdm(chunk_files, desc="Migrating chunks"):
        try:
            # Load chunk
            with open(chunk_file, 'rb') as f:
                chunk = pickle.load(f)
            
            # Migrate metadata
            migrated = migrate_chunk(chunk, verbose)
            
            if migrated is not None:
                # Count samples
                if isinstance(migrated, list):
                    total_samples += len(chunk) if isinstance(chunk, list) else 1
                    total_migrated += len(migrated)
                
                # Save migrated chunk
                output_file = output_dir / chunk_file.name
                with open(output_file, 'wb') as f:
                    pickle.dump(migrated, f)
                
                if verbose:
                    logger.info(f"  ✓ {chunk_file.name}: {len(migrated)} samples")
        
        except Exception as e:
            logger.warning(f"Error migrating {chunk_file.name}: {e}")
            continue
    
    logger.info(f"\nMigration complete:")
    logger.info(f"  Total chunks: {len(chunk_files)}")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Migrated samples: {total_migrated}")
    logger.info(f"  Output directory: {output_dir}")
    
    return total_migrated > 0


def main():
    parser = argparse.ArgumentParser(
        description='Migrate dataset to add SNR/distance metadata fields'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input data directory (e.g., data/dataset/train)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output data directory (e.g., data/dataset_migrated/train)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("DATASET METADATA MIGRATION")
    logger.info("="*70)
    logger.info(f"\nInput:  {args.input_dir}")
    logger.info(f"Output: {args.output_dir}\n")
    
    success = migrate_dataset(args.input_dir, args.output_dir, args.verbose)
    
    if success:
        logger.info("\n✅ Migration successful!")
        logger.info(f"\nYou can now verify with:")
        logger.info(f"  python verify_snr_distance_correlation.py --data-path {args.output_dir}")
    else:
        logger.error("\n❌ Migration failed!")
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
