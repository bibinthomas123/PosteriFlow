"""
I/O Utilities for Dataset Management
HDF5, pickle, and metadata handling with PKL support
"""

import numpy as np
import h5py
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import gzip
import time
from datetime import datetime, timezone

class DatasetWriter:
    """
    Write datasets to HDF5 or PKL format with comprehensive metadata
    """
    
    def __init__(self, output_dir: str, format: str = 'hdf5'):
        """
        Args:
            output_dir: Output directory path
            format: 'hdf5', 'pkl', or 'pkl_compressed'
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format.lower()
        self.logger = logging.getLogger(__name__)
    
    def save_batch_pkl(self,
                      batch_id: int,
                      samples: List[Dict],
                      metadata: Dict = None,
                      compress: bool = False) -> Path:
        """
        Save batch of samples to pickle file
        
        Args:
            batch_id: Batch identifier
            samples: List of sample dictionaries
            metadata: Metadata dictionary
            compress: Use gzip compression
            
        Returns:
            Path to saved file
        """
        
        if compress:
            filename = f"batch_{batch_id:05d}.pkl.gz"
        else:
            filename = f"batch_{batch_id:05d}.pkl"
        
        batch_dir = self.output_dir / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        filepath = batch_dir / filename
        
        try:
            # Create batch data structure
            batch_data = {
                'metadata': metadata or {},
                'samples': samples,
                'batch_id': batch_id,
                'n_samples': len(samples)
            }
            
            # Add metadata fields
            if metadata:
                batch_data['metadata']['batch_id'] = batch_id
                batch_data['metadata']['n_samples'] = len(samples)
            
            # Save with or without compression
            if compress:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.debug(f"✓ Batch {batch_id} saved to PKL ({len(samples)} samples)")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save PKL batch {batch_id}: {e}")
            raise
    
    def save_complete_dataset_pkl(self,
                                  filename: str,
                                  all_samples: List[Dict],
                                  metadata: Dict = None,
                                  compress: bool = True) -> Path:
        """
        Save entire dataset to single pickle file
        
        Args:
            filename: Output filename
            all_samples: List of all samples
            metadata: Dataset metadata
            compress: Use gzip compression
            
        Returns:
            Path to saved file
        """
        
        if compress and not filename.endswith('.gz'):
            if not filename.endswith('.pkl'):
                filename = filename + '.pkl.gz'
            else:
                filename = filename + '.gz'
        elif not compress and not filename.endswith('.pkl'):
            filename = filename + '.pkl'
        
        filepath = self.output_dir / filename
        
        try:
            
            # Ensure metadata has format_version
            if metadata is None:
                metadata = {}
            
            if 'format_version' not in metadata:
                metadata['format_version'] = '1.0.0'
            
            if 'creation_time' not in metadata:
                from datetime import datetime
                metadata['creation_time'] = datetime.now().isoformat()
                
                
            dataset = {
                'metadata': metadata or {},
                'samples': all_samples,
                'n_samples': len(all_samples),
                'format_version': '1.0.0'
            }
            
            # Add statistics
            from .io_utils import MetadataManager
            meta_manager = MetadataManager()
            stats = meta_manager.compute_dataset_statistics(all_samples)
            dataset['statistics'] = stats
            
            # Save
            if compress:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                self.logger.info(f"✓ Complete dataset saved to {filepath} (compressed)")
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                self.logger.info(f"✓ Complete dataset saved to {filepath}")
            
            # Get file size
            size_mb = filepath.stat().st_size / (1024 * 1024)
            self.logger.info(f"  File size: {size_mb:.2f} MB")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save complete dataset: {e}")
            raise
    
    def save_batch_hdf5(self,
                       batch_id: int,
                       samples: List[Dict],
                       metadata: Dict = None) -> Path:
        """Save batch of samples to HDF5"""
        
        filename = f"batch_{batch_id:05d}.h5"
        filepath = self.output_dir / filename
        
        try:
            with h5py.File(filepath, 'w') as f:
                # Metadata
                if metadata:
                    meta_group = f.create_group('metadata')
                    self._write_dict_to_group(meta_group, metadata)
                    meta_group.attrs['n_samples'] = len(samples)
                
                # Samples
                samples_group = f.create_group('samples')
                
                for i, sample in enumerate(samples):
                    sample_group = samples_group.create_group(f'sample_{i:05d}')
                    
                    # Sample data
                    for key, value in sample.items():
                        if isinstance(value, np.ndarray):
                            sample_group.create_dataset(key, data=value, compression='gzip')
                        elif isinstance(value, dict):
                            subgroup = sample_group.create_group(key)
                            self._write_dict_to_group(subgroup, value)
                        elif isinstance(value, (int, float, str, bool)):
                            sample_group.attrs[key] = value
            
            self.logger.debug(f"✓ Batch {batch_id} saved to HDF5 ({len(samples)} samples)")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save HDF5 batch {batch_id}: {e}")
            raise
    
    def _write_dict_to_group(self, group: h5py.Group, data_dict: Dict):
        """Recursively write dictionary to HDF5 group"""
        
        for key, value in data_dict.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._write_dict_to_group(subgroup, value)
            elif isinstance(value, (np.ndarray, list)):
                group.create_dataset(key, data=value)
            elif isinstance(value, (int, float, str, bool, type(None))):
                group.attrs[key] = value if value is not None else 'None'
            elif isinstance(value, (tuple, set)):
                group.attrs[key] = str(value)
    
    def save_pickle(self, filename: str, data: Any, compress: bool = False) -> Path:
        """Save arbitrary data to pickle file"""
        
        if compress and not filename.endswith('.gz'):
            filename = filename + '.gz'
        
        filepath = self.output_dir / filename
        
        try:
            if compress:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f"✓ Saved pickle to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save pickle: {e}")
            raise
    
    def save_json(self, filename: str, data: Dict) -> Path:
        """Save data to JSON file"""
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"✓ Saved JSON to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON: {e}")
            raise
    
    def save_yaml(self, filename: str, data: Dict) -> Path:
        """Save data to YAML file"""
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            self.logger.info(f"✓ Saved YAML to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save YAML: {e}")
            raise

    def save_split_chunks(self,
                     split_name: str,
                     samples: List[Dict],
                     metadata: Dict = None,
                     chunk_size: int = 100,
                     compress: bool = True) -> Path:
        """
        Save dataset split in chunks with comprehensive metadata.
        
        Args:
            split_name: 'train', 'validation', or 'test'
            samples: List of samples for this split
            metadata: Split metadata
            chunk_size: Samples per chunk file
            compress: Use gzip compression
            
        Returns:
            Path to split directory
        """
        
        split_dir = self.output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        n_chunks = (len(samples) + chunk_size - 1) // chunk_size
        
        self.logger.info(f"Saving {split_name} split: {len(samples)} samples in {n_chunks} chunks")
        
        # Save chunks
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(samples))
            chunk_samples = samples[start_idx:end_idx]
            
            # Save chunk
            if compress:
                chunk_file = split_dir / f'chunk_{chunk_idx:04d}.pkl.gz'
                with gzip.open(chunk_file, 'wb') as f:
                    pickle.dump(chunk_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                chunk_file = split_dir / f'chunk_{chunk_idx:04d}.pkl'
                with open(chunk_file, 'wb') as f:
                    pickle.dump(chunk_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.debug(f"  Chunk {chunk_idx+1}/{n_chunks}: {len(chunk_samples)} samples")
        
        # Prepare comprehensive metadata
        split_metadata = {
            'split': split_name,  
            'n_samples': len(samples),
            'n_chunks': n_chunks,
            'chunk_size': chunk_size,
            'compressed': compress,
            'file_pattern': 'chunk_XXXX.pkl.gz' if compress else 'chunk_XXXX.pkl',
            'created_at': datetime.now(timezone.utc).isoformat(),  # ✅ UTC ISO format
            'chunk_files': [f'chunk_{i:04d}.pkl' + ('.gz' if compress else '') 
                        for i in range(n_chunks)]
        }
        
        if metadata:
            split_metadata.update(metadata)
        
        json_file = split_dir / 'split_info.json'
        with open(json_file, 'w') as f:
            json.dump(split_metadata, f, indent=2, default=str)
        
        pkl_file = split_dir / f'{split_name}_metadata.pkl'
        with open(pkl_file, 'wb') as f:
            pickle.dump(split_metadata, f)
        
        self.logger.info(f"✓ {split_name} split saved ({n_chunks} chunks)")
        self.logger.info(f"  - Metadata: {json_file.name}, {pkl_file.name}")
        
        return split_dir

class DatasetReader:
    """
    Read datasets from various formats including pickle
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_pkl(self, filepath: str, compressed: bool = None) -> Dict:
        """
        Load dataset from pickle file
        
        Args:
            filepath: Path to pickle file
            compressed: True if gzipped, None for auto-detect
            
        Returns:
            Dataset dictionary
        """
        
        filepath = Path(filepath)
        
        # Auto-detect compression
        if compressed is None:
            compressed = filepath.suffix == '.gz' or str(filepath).endswith('.pkl.gz')
        
        try:
            if compressed:
                with gzip.open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            
            self.logger.info(f"✓ Loaded from {filepath}")
            
            # Log dataset info
            if isinstance(data, dict):
                if 'n_samples' in data:
                    self.logger.info(f"  Samples: {data['n_samples']}")
                if 'samples' in data:
                    self.logger.info(f"  Samples: {len(data['samples'])}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load pickle: {e}")
            raise
    
    def load_batch_pkl(self, filepath: str) -> Dict:
        """Load batch from pickle file"""
        return self.load_pkl(filepath)
    
    def load_hdf5(self, filepath: str) -> Dict:
        """Load dataset from HDF5 file"""
        
        try:
            with h5py.File(filepath, 'r') as f:
                data = {}
                
                # Load metadata
                if 'metadata' in f:
                    data['metadata'] = self._read_group_to_dict(f['metadata'])
                
                # Load data
                if 'data' in f:
                    data['data'] = self._read_group_to_dict(f['data'])
                
                # Load samples if present
                if 'samples' in f:
                    data['samples'] = self._read_samples_group(f['samples'])
            
            self.logger.info(f"✓ Loaded from {filepath}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load HDF5: {e}")
            raise
    
    def _read_group_to_dict(self, group: h5py.Group) -> Dict:
        """Recursively read HDF5 group to dictionary"""
        
        result = {}
        
        # Read attributes
        for key, value in group.attrs.items():
            if value == 'None':
                result[key] = None
            else:
                result[key] = value
        
        # Read datasets and subgroups
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                result[key] = np.array(item)
            elif isinstance(item, h5py.Group):
                result[key] = self._read_group_to_dict(item)
        
        return result
    
    def _read_samples_group(self, samples_group: h5py.Group) -> List[Dict]:
        """Read samples group to list of dicts"""
        
        samples = []
        
        for sample_key in sorted(samples_group.keys()):
            sample_group = samples_group[sample_key]
            sample = self._read_group_to_dict(sample_group)
            samples.append(sample)
        
        return samples
    
    def load_pickle(self, filepath: str) -> Any:
        """Load arbitrary data from pickle file"""
        return self.load_pkl(filepath)
    
    def load_json(self, filepath: str) -> Dict:
        """Load data from JSON file"""
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.logger.info(f"✓ Loaded JSON from {filepath}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON: {e}")
            raise
    
    def load_yaml(self, filepath: str) -> Dict:
        """Load data from YAML file"""
        
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            self.logger.info(f"✓ Loaded YAML from {filepath}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load YAML: {e}")
            raise

    def load_split_chunks(self, split_dir: str) -> List[Dict]:
        """
        Load all chunks from a split directory
        
        Args:
            split_dir: Path to split directory (e.g., 'data/dataset/train')
            
        Returns:
            List of all samples
        """
        
        split_path = Path(split_dir)
        
        # Load metadata
        metadata_file = split_path / 'split_info.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.logger.info(f"Loading {metadata['n_samples']} samples from {metadata['n_chunks']} chunks")
        
        # Find all chunk files
        chunk_files = sorted(split_path.glob('chunk_*.pkl*'))
        
        all_samples = []
        for chunk_file in chunk_files:
            samples = self.load_pkl(str(chunk_file))
            all_samples.extend(samples)
        
        self.logger.info(f"✓ Loaded {len(all_samples)} samples from {split_path.name}")
        
        return all_samples


class MetadataManager:
    """
    Manage dataset metadata and statistics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def create_metadata(self,
                   n_samples: int,
                   config: Dict,
                   statistics: Dict = None) -> Dict:
        """
        Create comprehensive dataset metadata dictionary.
        
        Includes creation timestamp (UTC), configuration, version info,
        system details, and optional statistics.
        
        Args:
            n_samples: Total number of samples in dataset
            config: Generation configuration dictionary
            statistics: Optional statistics dict (sample counts, distributions, etc.)
        
        Returns:
            Metadata dictionary with all relevant dataset information
        
        Example:
            >>> metadata = gen.create_metadata(
            ...     n_samples=50000,
            ...     config={'duration': 4.0, 'sample_rate': 4096},
            ...     statistics={'mean_snr': 18.5, 'n_overlaps': 15000}
            ... )
        """
        
        from datetime import datetime, timezone
        import sys
        import platform
        
        # ========================================================================
        # CORE METADATA
        # ========================================================================
        metadata = {
            # Timestamps
            'creation_time': datetime.now(timezone.utc).isoformat(),  # ✅ UTC ISO format
            'creation_timestamp_utc': datetime.now(timezone.utc).timestamp(),  # Unix timestamp
            
            # Dataset info
            'n_samples': n_samples,
            'format': 'AHSD-GW-Dataset',
            'format_version': '1.0.0',
            'schema_version': '1.0.0',
            
            # Configuration
            'configuration': config.copy() if config else {},
            
            # Generator info
            'generator': {
                'name': 'GWDatasetGenerator',
                'version': '1.0.0',
                'class': self.__class__.__name__
            }
        }
        
        # ========================================================================
        # SYSTEM INFORMATION (for reproducibility)
        # ========================================================================
        metadata['system_info'] = {
            'platform': sys.platform,
            'platform_version': platform.platform(),
            'python_version': sys.version.split()[0],
            'python_implementation': platform.python_implementation(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
        
        # ========================================================================
        # GIT INFORMATION (optional, for reproducibility)
        # ========================================================================
        git_commit = self._get_git_commit()
        if git_commit:
            metadata['git_info'] = {
                'commit': git_commit,
                'branch': self._get_git_branch(),
                'dirty': self._is_git_dirty()
            }
        
        # ========================================================================
        # DETECTOR CONFIGURATION
        # ========================================================================
        metadata['detector_config'] = {
            'detectors': self.detectors,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'n_samples_per_segment': int(self.duration * self.sample_rate)
        }
        
        # ========================================================================
        # STATISTICS (optional)
        # ========================================================================
        if statistics:
            metadata['statistics'] = statistics
        
        # ========================================================================
        # PACKAGE VERSIONS (for reproducibility)
        # ========================================================================
        metadata['dependencies'] = self._get_package_versions()
        
        return metadata


    # ========================================================================
    # HELPER METHODS FOR METADATA
    # ========================================================================

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            return result.stdout.strip()
        except Exception:
            return None


    def _get_git_branch(self) -> Optional[str]:
        """Get current git branch name."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            return result.stdout.strip()
        except Exception:
            return None


    def _is_git_dirty(self) -> bool:
        """Check if git working directory has uncommitted changes."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            return len(result.stdout.strip()) > 0
        except Exception:
            return False


    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key dependencies."""
        versions = {}
        
        packages = [
            'numpy',
            'scipy',
            'pycbc',
            'lalsuite',
            'h5py',
            'torch',
            'gwpy'
        ]
        
        for package in packages:
            try:
                import importlib
                mod = importlib.import_module(package)
                versions[package] = getattr(mod, '__version__', 'unknown')
            except ImportError:
                versions[package] = 'not_installed'
            except Exception:
                versions[package] = 'unknown'
        
        return versions


    def compute_dataset_statistics(self, samples: List[Dict]) -> Dict:
        """Compute comprehensive dataset statistics"""
        
        stats = {
            'n_total': len(samples),
            'event_types': {},
            'snr_distribution': {},
            'overlap_statistics': {},
            'detector_coverage': {}
        }
        
        # Event type distribution
        event_types = [s.get('type', 'unknown') for s in samples]
        for et in set(event_types):
            stats['event_types'][et] = event_types.count(et)
        
        # SNR statistics
        snrs = [s.get('target_snr', 0) for s in samples if 'target_snr' in s]
        if snrs:
            stats['snr_distribution'] = {
                'mean': float(np.mean(snrs)),
                'std': float(np.std(snrs)),
                'min': float(np.min(snrs)),
                'max': float(np.max(snrs)),
                'median': float(np.median(snrs))
            }
        
        # Overlap statistics
        n_overlaps = sum(1 for s in samples if s.get('is_overlap', False))
        stats['overlap_statistics'] = {
            'n_overlapping': n_overlaps,
            'n_single': len(samples) - n_overlaps,
            'overlap_fraction': n_overlaps / len(samples) if samples else 0
        }
        
        return stats
