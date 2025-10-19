"""
Main Dataset Generator
Orchestrates complete dataset generation pipeline
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import time


from .config import (
    SAMPLE_RATE, DURATION, DETECTORS, EVENT_TYPE_DISTRIBUTION,
    SNR_DISTRIBUTION, SNR_RANGES, OVERLAP_FRACTION, EDGE_CASE_FRACTION
)
from .psd_manager import PSDManager
from .parameter_sampler import ParameterSampler
from .waveform_generator import WaveformGenerator
from .noise_generator import NoiseGenerator
from .injection import SignalInjector
from .preprocessing import DataPreprocessor
from .io_utils import DatasetWriter, MetadataManager

# Add to the GWDatasetGenerator class

class GWDatasetGenerator:
    """
    Main class for generating complete GW datasets
    Supports HDF5 and PKL output formats
    """
    
    def __init__(self, 
                 output_dir: str = "data/output",
                 sample_rate: int = SAMPLE_RATE,
                 duration: float = DURATION,
                 detectors: List[str] = None,
                 output_format: str = 'pkl'):  # NEW: Add format parameter
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.duration = duration
        self.detectors = detectors or DETECTORS
        self.output_format = output_format.lower()  # 'hdf5', 'pkl', or 'both'
        
        # Validate format
        if self.output_format not in ['hdf5', 'pkl', 'pkl_compressed', 'both']:
            raise ValueError(f"Invalid output_format: {self.output_format}")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.logger.info("Initializing dataset generator components...")
        self.psd_manager = PSDManager(sample_rate, duration)
        self.parameter_sampler = ParameterSampler()
        self.waveform_generator = WaveformGenerator(sample_rate, duration)
        self.noise_generator = NoiseGenerator(sample_rate, duration)
        self.injector = SignalInjector(sample_rate, duration)
        self.preprocessor = DataPreprocessor(sample_rate, duration)
        self.writer = DatasetWriter(output_dir, format=output_format)
        self.metadata_manager = MetadataManager()
        
        # Load PSDs
        self.logger.info(f"Loading PSDs for detectors: {self.detectors}")
        self.psds = self.psd_manager.load_detector_psds(self.detectors)
        
        self.logger.info(f"✓ Dataset generator initialized (output format: {output_format})")
    
    def create_noise_augmentations(self, sample: Dict, k: int) -> List[Dict]:
        """
        Create k augmented versions with different noise realizations
        
        Args:
            sample: Original sample
            k: Number of augmentations (including original)
            
        Returns:
            List of k augmented samples
        """
        
        if k <= 1:
            return [sample]
        
        augmented_samples = [sample]  # Include original
        
        # Get signal for this sample (if not noise-only)
        has_signal = sample['type'] != 'noise'
        
        for aug_idx in range(1, k):
            # Create augmented copy
            aug_sample = {
                'sample_id': f"{sample['sample_id']}_aug{aug_idx}",
                'type': sample['type'],
                'is_overlap': sample.get('is_overlap', False),
                'is_edge_case': sample.get('is_edge_case', False),
                'parameters': sample.get('parameters'),
                'detector_data': {}
            }
            
            # Generate new noise realization for each detector
            for det_name in self.detectors:
                psd_dict = self.psds[det_name]
                
                # Generate fresh noise
                new_noise = self.noise_generator.generate_colored_noise(psd_dict)
                
                if has_signal and 'detector_data' in sample:
                    # Extract signal from original (assuming signal + noise structure)
                    # For simplicity, we regenerate the signal with same params
                    original_data = sample['detector_data'].get(det_name, {})
                    
                    if sample.get('is_overlap', False):
                        # Overlapping signals
                        params_list = sample['parameters']
                        injected, metadata_list = self.injector.inject_overlapping_signals(
                            new_noise, params_list, det_name, psd_dict
                        )
                    else:
                        # Single signal
                        params = sample['parameters']
                        if params:
                            injected, metadata = self.injector.inject_signal(
                                new_noise, params, det_name, psd_dict
                            )
                        else:
                            injected = new_noise
                            metadata = {'noise_only': True}
                else:
                    # Noise-only
                    injected = new_noise
                    metadata = {'noise_only': True}
                
                # Preprocess if needed
                if self.preprocessor:
                    injected = self.preprocessor.preprocess(injected, psd_dict)
                
                aug_sample['detector_data'][det_name] = {
                    'strain': injected.astype(np.float32),
                    'metadata': metadata if not sample.get('is_overlap') else metadata_list
                }
            
            augmented_samples.append(aug_sample)
        
        return augmented_samples
    
    
    def generate_dataset(self,
                    n_samples: int = 1000,
                    overlap_fraction: float = OVERLAP_FRACTION,
                    edge_case_fraction: float = EDGE_CASE_FRACTION,
                    save_batch_size: int = 100,
                    add_glitches: bool = True,
                    preprocess: bool = True,
                    save_complete: bool = True,
                    create_splits: bool = True,
                    train_frac: float = 0.8,
                    val_frac: float = 0.1,
                    test_frac: float = 0.1,
                    chunk_size: int = 100,
                    noise_augmentation_k: int = 1) -> Dict:
        """
        Generate complete dataset with train/val/test splits, chunked storage, and optional augmentation
        
        Args:
            n_samples: Total number of samples to generate
            overlap_fraction: Fraction with overlapping signals (0-1)
            edge_case_fraction: Fraction of edge cases (0-1)
            save_batch_size: Save to disk every N samples
            add_glitches: Add realistic glitches to noise
            preprocess: Apply preprocessing (whitening, filtering)
            save_complete: Save complete dataset to single file (PKL only)
            create_splits: Create train/val/test splits
            train_frac: Training set fraction (default 0.8)
            val_frac: Validation set fraction (default 0.1)
            test_frac: Test set fraction (default 0.1)
            chunk_size: Samples per chunk file
            noise_augmentation_k: Number of noise augmentations for train set (1 = no augmentation)
            
        Returns:
            Dictionary with dataset statistics and generation summary
        """
        
        self.logger.info("=" * 60)
        self.logger.info("Starting dataset generation")
        self.logger.info(f"Target samples: {n_samples:,}")
        self.logger.info(f"Overlap fraction: {overlap_fraction:.2%}")
        self.logger.info(f"Edge case fraction: {edge_case_fraction:.2%}")
        self.logger.info(f"Output format: {self.output_format}")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Calculate sample counts
        n_overlap = int(n_samples * overlap_fraction)
        n_edge = int(n_samples * edge_case_fraction)
        n_regular = n_samples - n_overlap
        
        self.logger.info("Sample breakdown:")
        self.logger.info(f"  - Overlapping: {n_overlap:,}")
        self.logger.info(f"  - Regular: {n_regular:,}")
        self.logger.info(f"  - Edge cases: {n_edge:,}")
        self.logger.info("")
        
        # Store preprocessing flag for augmentation
        self.preprocess_enabled = preprocess
        
        # Generate samples
        samples = []
        all_samples = []  # For complete dataset and splits
        batch_id = 0
        
        # Progress tracking
        last_log_time = time.time()
        log_interval = 60  # Log every 60 seconds
        samples_at_last_log = 0
        
        with tqdm(total=n_samples, desc="Generating samples", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            # Generate overlapping samples
            for i in range(n_overlap):
                is_edge = i < (n_edge // 2)
                sample = self._generate_overlapping_sample(
                    i, is_edge, add_glitches, preprocess
                )
                samples.append(sample)
                all_samples.append(sample)
                pbar.update(1)
                
                # Periodic progress logging
                current_time = time.time()
                if current_time - last_log_time >= log_interval:
                    elapsed = current_time - start_time
                    samples_generated = len(all_samples)
                    samples_since_last = samples_generated - samples_at_last_log
                    current_rate = samples_since_last / log_interval
                    remaining_samples = n_samples - samples_generated
                    eta = remaining_samples / current_rate if current_rate > 0 else 0
                    
                    self.logger.info(
                        f"[PROGRESS] {samples_generated:,}/{n_samples:,} ({samples_generated/n_samples*100:.1f}%) | "
                        f"Rate: {current_rate:.2f} samples/s | "
                        f"ETA: {eta/60:.1f}m | "
                        f"Elapsed: {elapsed/60:.1f}m"
                    )
                    
                    last_log_time = current_time
                    samples_at_last_log = samples_generated
                
                # Save batch
                if len(samples) >= save_batch_size:
                    self._save_batch(batch_id, samples)
                    self.logger.debug(f"Batch {batch_id} saved ({len(samples)} samples)")
                    batch_id += 1
                    samples = []
            
            # Generate regular samples
            for i in range(n_regular):
                is_edge = (i + n_overlap) < n_edge
                sample = self._generate_single_sample(
                    i + n_overlap, is_edge, add_glitches, preprocess
                )
                samples.append(sample)
                all_samples.append(sample)
                pbar.update(1)
                
                # Periodic progress logging (same as above)
                current_time = time.time()
                if current_time - last_log_time >= log_interval:
                    elapsed = current_time - start_time
                    samples_generated = len(all_samples)
                    samples_since_last = samples_generated - samples_at_last_log
                    current_rate = samples_since_last / log_interval
                    remaining_samples = n_samples - samples_generated
                    eta = remaining_samples / current_rate if current_rate > 0 else 0
                    
                    self.logger.info(
                        f"[PROGRESS] {samples_generated:,}/{n_samples:,} ({samples_generated/n_samples*100:.1f}%) | "
                        f"Rate: {current_rate:.2f} samples/s | "
                        f"ETA: {eta/60:.1f}m | "
                        f"Elapsed: {elapsed/60:.1f}m"
                    )
                    
                    last_log_time = current_time
                    samples_at_last_log = samples_generated
                
                # Save batch
                if len(samples) >= save_batch_size:
                    self._save_batch(batch_id, samples)
                    self.logger.debug(f"Batch {batch_id} saved ({len(samples)} samples)")
                    batch_id += 1
                    samples = []
        
        # Save remaining samples
        if samples:
            self._save_batch(batch_id, samples)
            self.logger.debug(f"Final batch {batch_id} saved ({len(samples)} samples)")
        
        generation_time = time.time() - start_time
        self.logger.info("")
        self.logger.info(f"✓ Sample generation complete: {len(all_samples):,} samples in {generation_time/60:.1f}m")
        self.logger.info("")
        
        # Save complete dataset if requested (before splitting)
        if save_complete and self.output_format in ['pkl', 'pkl_compressed']:
            self.logger.info("Saving complete dataset to single file...")
            
            metadata = self.metadata_manager.create_metadata(
                n_samples=n_samples,
                config={
                    'sample_rate': self.sample_rate,
                    'duration': self.duration,
                    'detectors': self.detectors,
                    'overlap_fraction': overlap_fraction,
                    'edge_case_fraction': edge_case_fraction,
                    'output_format': self.output_format,
                    'generation_time_seconds': generation_time
                }
            )
            
            compress = (self.output_format == 'pkl_compressed')
            self.writer.save_complete_dataset_pkl(
                'complete_dataset.pkl',
                all_samples,
                metadata,
                compress=compress
            )
        
        # Create train/val/test splits
        splits_info = None
        if create_splits:
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("Creating train/validation/test splits...")
            self.logger.info("=" * 60)
            
            splits = self._create_splits(
                all_samples,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                stratify=True
            )
            
            # Apply noise augmentation to training set
            original_train_size = len(splits['train']['samples'])
            
            if noise_augmentation_k > 1:
                self.logger.info("")
                self.logger.info(f"Applying {noise_augmentation_k}x noise augmentation to training set...")
                self.logger.info(f"  Original train size: {original_train_size:,} samples")
                
                augmented_train = []
                aug_start_time = time.time()
                
                with tqdm(total=original_train_size, desc="Augmenting train",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                    for sample in splits['train']['samples']:
                        aug_samples = self.create_noise_augmentations(sample, noise_augmentation_k)
                        augmented_train.extend(aug_samples)
                        pbar.update(1)
                
                aug_time = time.time() - aug_start_time
                splits['train']['samples'] = augmented_train
                splits['train']['n_samples'] = len(augmented_train)
                
                self.logger.info(f"✓ Training set augmented: {original_train_size:,} → {len(augmented_train):,} samples in {aug_time/60:.1f}m")
                self.logger.info("")
            
            # Save splits in chunks
            self.logger.info("Saving splits to disk...")
            compress = (self.output_format == 'pkl_compressed')
            
            for split_name in ['train', 'validation', 'test']:
                split_samples = splits[split_name]['samples']
                
                split_metadata = {
                    'split': split_name,
                    'sample_rate': self.sample_rate,
                    'duration': self.duration,
                    'detectors': self.detectors,
                    'parent_dataset_size': n_samples,
                    'augmentation_factor': noise_augmentation_k if split_name == 'train' else 1,
                    'original_samples': len(splits[split_name]['indices']),
                    'total_samples': len(split_samples),
                    'chunk_size': chunk_size
                }
                
                self.writer.save_split_chunks(
                    split_name,
                    split_samples,
                    split_metadata,
                    chunk_size=chunk_size,
                    compress=compress
                )
            
            # Save split indices
            split_indices = {
                'train': splits['train']['indices'],
                'validation': splits['validation']['indices'],
                'test': splits['test']['indices'],
                'train_augmentation_factor': noise_augmentation_k,
                'train_original_size': original_train_size,
                'train_augmented_size': splits['train']['n_samples']
            }
            
            import json
            with open(self.output_dir / 'split_indices.json', 'w') as f:
                json.dump(split_indices, f, indent=2)
            
            self.logger.info("✓ All splits saved in chunks")
            
            # Store splits info for summary
            splits_info = {
                'train': {
                    'original': original_train_size,
                    'augmented': splits['train']['n_samples'],
                    'augmentation_factor': noise_augmentation_k
                },
                'validation': splits['validation']['n_samples'],
                'test': splits['test']['n_samples']
            }
        
        # Save PSDs
        self.logger.info("")
        self.logger.info("Saving detector PSDs...")
        psd_dir = self.output_dir / 'detector_psds'
        psd_dir.mkdir(exist_ok=True)
        
        for detector_name, psd_info in self.psds.items():
            psd_file = psd_dir / f'{detector_name}_psd.npz'
            psd_array = psd_info['psd']
            if hasattr(psd_array, 'numpy'):
                psd_array = psd_array.numpy()
            
            np.savez(psd_file,
                    frequencies=psd_info['frequencies'],
                    psd=psd_array,
                    source=psd_info['source'],
                    name=psd_info['name'])
        
        self.logger.info("✓ PSDs saved")
        
        # Generate summary
        elapsed_time = time.time() - start_time
        
        summary = {
            'n_samples': n_samples,
            'n_batches': batch_id + 1,
            'elapsed_time': elapsed_time,
            'generation_time': generation_time,
            'samples_per_second': n_samples / generation_time,
            'output_dir': str(self.output_dir),
            'output_format': self.output_format,
            'configuration': {
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'detectors': self.detectors,
                'overlap_fraction': overlap_fraction,
                'edge_case_fraction': edge_case_fraction,
                'add_glitches': add_glitches,
                'preprocess': preprocess,
                'chunk_size': chunk_size,
                'noise_augmentation_k': noise_augmentation_k,
                'train_frac': train_frac,
                'val_frac': val_frac,
                'test_frac': test_frac
            }
        }
        
        if splits_info:
            summary['splits'] = splits_info
        
        # Final log
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("DATASET GENERATION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Total samples:     {n_samples:,}")
        self.logger.info(f"Generation time:   {generation_time/60:.1f}m ({generation_time:.1f}s)")
        self.logger.info(f"Total time:        {elapsed_time/60:.1f}m ({elapsed_time:.1f}s)")
        self.logger.info(f"Generation rate:   {summary['samples_per_second']:.2f} samples/s")
        self.logger.info(f"Output directory:  {self.output_dir}")
        self.logger.info(f"Batches saved:     {batch_id + 1}")
        
        if create_splits:
            self.logger.info("")
            self.logger.info("Split Summary:")
            self.logger.info(f"  Train:  {splits_info['train']['original']:,} samples (original)")
            if noise_augmentation_k > 1:
                self.logger.info(f"          {splits_info['train']['augmented']:,} samples (after {noise_augmentation_k}x augmentation)")
            self.logger.info(f"  Validation:    {splits_info['validation']:,} samples")
            self.logger.info(f"  Test:   {splits_info['test']:,} samples")
        
        self.logger.info("=" * 60)
        
        # Save summary
        self.writer.save_json('generation_summary.json', summary)
        
        return summary

    def _save_batch(self, batch_id: int, samples: List[Dict]):
        """Save batch to disk in specified format"""
        
        metadata = {
            'batch_id': batch_id,
            'n_samples': len(samples),
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'detectors': self.detectors
        }
        
        if self.output_format == 'hdf5':
            self.writer.save_batch_hdf5(batch_id, samples, metadata)
        elif self.output_format == 'pkl':
            self.writer.save_batch_pkl(batch_id, samples, metadata, compress=False)
        elif self.output_format == 'pkl_compressed':
            self.writer.save_batch_pkl(batch_id, samples, metadata, compress=True)
        elif self.output_format == 'both':
            self.writer.save_batch_hdf5(batch_id, samples, metadata)
            self.writer.save_batch_pkl(batch_id, samples, metadata, compress=True)

    def _create_splits(self, 
                   all_samples: List[Dict],
                   train_frac: float = 0.8,
                   val_frac: float = 0.1,
                   test_frac: float = 0.1,
                   stratify: bool = True) -> Dict:
        """
        Split dataset into train/val/test with optional stratification
        """
        
        import random
        
        n_total = len(all_samples)
        
        if stratify:
            # Group samples by event type
            type_groups = {}
            for i, sample in enumerate(all_samples):
                event_type = sample.get('type', 'unknown')
                if event_type not in type_groups:
                    type_groups[event_type] = []
                type_groups[event_type].append(i)
            
            # Split each group proportionally
            train_indices = []
            val_indices = []
            test_indices = []
            
            for event_type, indices in type_groups.items():
                random.shuffle(indices)
                n_type = len(indices)
                
                n_train = int(n_type * train_frac)
                n_val = int(n_type * val_frac)
                
                train_indices.extend(indices[:n_train])
                val_indices.extend(indices[n_train:n_train + n_val])
                test_indices.extend(indices[n_train + n_val:])
            
            # Shuffle the splits
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)
            
        else:
            # Simple random split
            indices = list(range(n_total))
            random.shuffle(indices)
            
            n_train = int(n_total * train_frac)
            n_val = int(n_total * val_frac)
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
        
        # Create splits with 'validation' key (not 'val')
        splits = {
            'train': {
                'samples': [all_samples[i] for i in train_indices],
                'indices': train_indices,
                'n_samples': len(train_indices)
            },
            'validation': {  # ← CHANGE FROM 'val' TO 'validation'
                'samples': [all_samples[i] for i in val_indices],
                'indices': val_indices,
                'n_samples': len(val_indices)
            },
            'test': {
                'samples': [all_samples[i] for i in test_indices],
                'indices': test_indices,
                'n_samples': len(test_indices)
            }
        }
        
        self.logger.info("Dataset splits created:")
        self.logger.info(f"  Train:      {len(train_indices)} samples ({len(train_indices)/n_total*100:.1f}%)")
        self.logger.info(f"  Validation: {len(val_indices)} samples ({len(val_indices)/n_total*100:.1f}%)")  # ← UPDATE LOG
        self.logger.info(f"  Test:       {len(test_indices)} samples ({len(test_indices)/n_total*100:.1f}%)")
        
        return splits

    def _generate_single_sample(self, sample_id: int, is_edge_case: bool,
                           add_glitches: bool, preprocess: bool) -> Dict:
        """Generate single non-overlapping sample"""
        
        event_type = self._sample_event_type()
        snr_regime = self._sample_snr_regime()
        
        # Generate parameters
        if event_type == 'BBH':
            params = self.parameter_sampler.sample_bbh_parameters(snr_regime, is_edge_case)
        elif event_type == 'BNS':
            params = self.parameter_sampler.sample_bns_parameters(snr_regime, is_edge_case)
        elif event_type == 'NSBH':
            params = self.parameter_sampler.sample_nsbh_parameters(snr_regime, is_edge_case)
        else:
            params = None
        
        # Generate data for each detector
        detector_data = {}
        for detector_name in self.detectors:
            psd_dict = self.psds[detector_name]
            
            # Generate noise
            noise = self.noise_generator.generate_colored_noise(psd_dict)
            
            # Add glitches
            if add_glitches:
                noise = self.noise_generator.add_glitches(noise, glitch_prob=0.3)
            
            # Inject signal
            if params:
                injected, metadata = self.injector.inject_signal(
                    noise, params, detector_name, psd_dict
                )
            else:
                injected = noise
                metadata = {'detector': detector_name, 'noise_only': True}
            
            # Preprocess
            if preprocess:
                injected = self.preprocessor.preprocess(injected, psd_dict)
            
            detector_data[detector_name] = {
                'strain': injected.astype(np.float32),
                'metadata': metadata
            }
        
        # Create sample with comprehensive metadata (like original script)
        sample = {
            'sample_id': f'single_{sample_id:06d}',
            'type': event_type,
            'is_overlap': False,
            'is_edge_case': is_edge_case,
            'parameters': params,
            'detector_data': detector_data,
            'metadata': {                                      # ← ADD THIS BLOCK
                'sample_id': f'single_{sample_id:06d}',
                'event_type': event_type,
                'detector_network': self.detectors,
                'snr_regime': snr_regime,
                'signal_parameters': [params] if params else [],  # ← KEY FIX
                'is_edge_case': is_edge_case,
                'overlap_type': 'single'
            }
        }
        
        return sample

    def _generate_overlapping_sample(self, sample_id: int, is_edge_case: bool,
                                    add_glitches: bool, preprocess: bool) -> Dict:
        """Generate sample with overlapping signals"""
        
        n_signals = np.random.choice([2, 3, 4], p=[0.6, 0.3, 0.1])
        
        signal_params_list = []
        target_snrs = []
        
        # Generate parameters for each signal
        for i in range(n_signals):
            event_type = self._sample_event_type()
            snr_regime = self._sample_snr_regime()
            
            if event_type == 'BBH':
                params = self.parameter_sampler.sample_bbh_parameters(snr_regime, is_edge_case)
            elif event_type == 'BNS':
                params = self.parameter_sampler.sample_bns_parameters(snr_regime, is_edge_case)
            else:
                params = self.parameter_sampler.sample_nsbh_parameters(snr_regime, is_edge_case)
            
            params['time_offset'] = np.random.uniform(-0.25, 0.25) if i > 0 else 0.0
            signal_params_list.append(params)
            
            # Compute SNR using the first detector's PSD (outside detector loop)
            waveform = self.waveform_generator.generate_waveform(params)
            reference_psd = self.psds[self.detectors[0]]  # Use first detector as reference
            target_snr = self.injector._compute_optimal_snr(waveform, reference_psd)
            target_snrs.append(float(target_snr))
        
        # Generate data for each detector (NOW psd_dict is defined here)
        detector_data = {}
        for detector_name in self.detectors:
            psd_dict = self.psds[detector_name]
            
            noise = self.noise_generator.generate_colored_noise(psd_dict)
            
            if add_glitches:
                noise = self.noise_generator.add_glitches(noise, glitch_prob=0.2)
            
            injected, metadata_list = self.injector.inject_overlapping_signals(
                noise, signal_params_list, detector_name, psd_dict
            )
            
            if preprocess:
                injected = self.preprocessor.preprocess(injected, psd_dict)
            
            detector_data[detector_name] = {
                'strain': injected.astype(np.float32),
                'metadata': metadata_list
            }
        
        sample = {
            'sample_id': f'overlap_{sample_id:06d}',
            'type': 'overlap',
            'is_overlap': True,
            'n_signals': n_signals,
            'is_edge_case': is_edge_case,
            'parameters': signal_params_list,
            'priorities': target_snrs,
            'detector_data': detector_data,
            'metadata': {
                'sample_id': f'overlap_{sample_id:06d}',
                'event_type': 'overlap',
                'detector_network': self.detectors,
                'n_signals': n_signals,
                'signal_parameters': signal_params_list,
                'is_edge_case': is_edge_case,
                'overlap_type': 'multi_signal'
            }
        }
        
        return sample

    def _sample_event_type(self) -> str:
        """Sample event type from distribution"""
        types = list(EVENT_TYPE_DISTRIBUTION.keys())
        probs = list(EVENT_TYPE_DISTRIBUTION.values())
        return np.random.choice(types, p=probs)
    
    def _sample_snr_regime(self) -> str:
        """Sample SNR regime from distribution"""
        regimes = list(SNR_DISTRIBUTION.keys())
        probs = list(SNR_DISTRIBUTION.values())
        return np.random.choice(regimes, p=probs)
