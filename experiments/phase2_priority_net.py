#!/usr/bin/env python3
"""
COMPLETE Phase 2: PriorityNet with Integrated Dataset Loading
Reads directly from your 20K chunked dataset structure
Enhanced with train/validation/test splits and comprehensive evaluation
"""

import sys
import os
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import argparse
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import yaml
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ahsd.core.priority_net import PriorityNet, PriorityNetTrainer
print('PriorityNetTrainer loaded from:', sys.modules['ahsd.core.priority_net'].__file__)

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('phase2_priority_net_complete.log'),
            logging.StreamHandler()
        ]
    )

class ChunkedGWDataLoader:
    """
    Data loader for your 20K chunked GW dataset structure
    Converts GW samples to PriorityNet training scenarios
    """
    
    def __init__(self, dataset_path: str, split: str = 'train', 
                 max_samples: Optional[int] = None):
        """
        Initialize chunked GW dataset loader
        
        Args:
            dataset_path: Path to 'newDataset' directory
            split: 'train', 'validation', or 'test'
            max_samples: Maximum samples to load (None = all)
        """
        
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.max_samples = max_samples
        self.logger = logging.getLogger(__name__)
        
        # Load split information
        self._load_split_info()
        
        # Load all samples from chunks
        self.samples = self._load_all_samples()
        
        self.logger.info(f" {split.upper()} dataset loaded: {len(self.samples)} samples")

    def _load_split_info(self):
        """Load split information"""
        
        split_dir = self.dataset_path / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Load split info
        split_info_file = split_dir / 'split_info.json'
        if split_info_file.exists():
            with open(split_info_file, 'r') as f:
                self.split_info = json.load(f)
        else:
            # Fallback: count chunk files
            chunk_files = list(split_dir.glob('chunk_*.pkl'))
            self.split_info = {
                'n_chunks': len(chunk_files),
                'chunk_size': 500,
                'file_pattern': 'chunk_XXXX.pkl'
            }
        
        self.n_chunks = self.split_info['n_chunks']
        self.chunk_size = self.split_info['chunk_size']
        
        self.logger.info(f"ðŸ“Š {self.split}: {self.n_chunks} chunks, {self.chunk_size} samples per chunk")

    def _load_all_samples(self) -> List[Dict]:
        """Load all samples from chunks"""
        
        all_samples = []
        total_loaded = 0
        
        self.logger.info(f"ðŸ’¾ Loading {self.split} chunks...")
        
        for chunk_id in tqdm(range(self.n_chunks), desc=f"Loading {self.split}"):
            chunk_file = self.dataset_path / self.split / f'chunk_{chunk_id:04d}.pkl'
            
            if chunk_file.exists():
                try:
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    for sample in chunk_data:
                        all_samples.append(sample)
                        total_loaded += 1
                        
                        # Check max samples limit
                        if self.max_samples and total_loaded >= self.max_samples:
                            self.logger.info(f"â¹ï¸ Reached max samples limit: {self.max_samples}")
                            return all_samples
                            
                except Exception as e:
                    self.logger.warning(f"Failed to load chunk {chunk_id}: {e}")
                    continue
            else:
                self.logger.warning(f"Chunk file not found: {chunk_file}")
        
        return all_samples
    
    
    def _convert_noise_sample_to_scenario(self, sample: Dict) -> Optional[Dict]:
        """FIXED: Convert noise sample to PriorityNet scenario"""
        
        try:
            metadata = sample.get('metadata', {})
            
            # Create noise parameter entry
            noise_param = {
                'mass_1': 0.0,
                'mass_2': 0.0,
                'luminosity_distance': 0.0,
                'network_snr': 0.0,
                'individual_snrs': {det: 0.0 for det in metadata.get('detector_network', ['H1', 'L1'])},
                'ra': 0.0,
                'dec': 0.0,
                'theta_jn': 0.0,
                'psi': 0.0,
                'phase': 0.0,
                'geocent_time': 0.0,
                'a1': 0.0,
                'a2': 0.0,
                'approximant': 'noise',
                'event_type': 'noise',
                'edge_case': False,
                'edge_case_type': None,
                'detector_network': metadata.get('detector_network', ['H1', 'L1']),
                'sample_id': sample.get('sample_id', 'unknown'),
                'noise_type': metadata.get('noise_type', 'gaussian'),
                'glitch_present': metadata.get('glitch_present', False)
            }
            
            scenario = {
                'scenario_id': sample.get('sample_id', 'unknown'),
                'true_parameters': [noise_param],
                'baseline_biases': [],
                'detector_data': sample.get('detector_data', {}),
                'whitened_data': sample.get('whitened_data', {}),
                'metadata': metadata
            }
            
            return scenario
            
        except Exception as e:
            self.logger.debug(f"Failed to convert noise sample: {e}")
            return None


    def convert_to_priority_scenarios(self, create_overlaps: bool = True,
                                   overlap_probability: float = 0.3) -> List[Dict]:
        """
        Convert GW samples to PriorityNet training scenarios with artificial overlaps
        for ALL splits (train, validation, test)
        """
        
        self.logger.info("ðŸ”„ Converting GW samples to PriorityNet scenarios...")
        
        scenarios = []
        single_signal_samples = []
        multi_signal_samples = []
        noise_samples = []
        overlap_samples = []
        
        # Separate samples by type
        for sample in self.samples:
            metadata = sample.get('metadata', {})
            event_type = metadata.get('event_type', 'unknown')
            overlap_type = metadata.get('overlap_type', 'single')
            n_signals = metadata.get('n_signals', 1)
            
            if event_type == 'noise':
                noise_samples.append(sample)
            elif event_type == 'overlap' or overlap_type == 'multi_signal' or n_signals > 1:
                overlap_samples.append(sample)
            elif n_signals == 1:
                single_signal_samples.append(sample)
            else:
                multi_signal_samples.append(sample)
        
        self.logger.info(f"ðŸ“Š Found {len(single_signal_samples)} single-signal, "
                        f"{len(multi_signal_samples)} multi-signal, "
                        f"{len(noise_samples)} noise, "
                        f"{len(overlap_samples)} overlap samples")
        
        # Convert noise samples
        for sample in tqdm(noise_samples, desc="Converting noise"):
            scenario = self._convert_noise_sample_to_scenario(sample)
            if scenario:
                scenarios.append(scenario)
        
        # Convert single-signal samples
        for sample in tqdm(single_signal_samples, desc="Converting single signals"):
            scenario = self._convert_single_sample_to_scenario(sample)
            if scenario:
                scenarios.append(scenario)
        
        # Convert multi-signal samples
        for sample in tqdm(multi_signal_samples, desc="Converting multi signals"):
            scenario = self._convert_multi_sample_to_scenario(sample)
            if scenario:
                scenarios.append(scenario)
        
        # Convert original overlap samples
        for sample in tqdm(overlap_samples, desc="Converting overlaps"):
            scenario = self._convert_overlap_sample_to_scenario(sample)
            if scenario:
                scenarios.append(scenario)
        
        #  FIXED: Create artificial overlaps for ALL splits (not just training)
        if create_overlaps and len(single_signal_samples) >= 2:
            self.logger.info(f"ðŸ”„ Creating artificial overlap scenarios for {self.split}...")
            
            #  Split-specific overlap counts
            if self.split == 'train':
                n_artificial_overlaps = min(6000, len(single_signal_samples) // 2)
            elif self.split == 'validation':
                n_artificial_overlaps = min(400, len(single_signal_samples) // 2)
            elif self.split == 'test':
                n_artificial_overlaps = min(400, len(single_signal_samples) // 2)
            else:
                n_artificial_overlaps = 0
            
            if n_artificial_overlaps > 0:
                self.logger.info(f"   Target: {n_artificial_overlaps} artificial overlaps")
                created_count = 0
                attempts = 0
                max_attempts = n_artificial_overlaps * 2  # Allow 2x attempts
                
                pbar = tqdm(total=n_artificial_overlaps, desc="Creating artificial overlaps")
                
                while created_count < n_artificial_overlaps and attempts < max_attempts:
                    attempts += 1
                    
                    # Create artificial overlap
                    scenario = self._create_artificial_overlap_scenario(single_signal_samples)
                    if scenario:
                        scenarios.append(scenario)
                        created_count += 1
                        pbar.update(1)
                
                pbar.close()
                
                self.logger.info(f" Created {created_count} artificial overlap scenarios ({attempts} attempts)")
        
        self.logger.info(f" Created {len(scenarios)} total PriorityNet scenarios")
        
        return scenarios

    def _convert_overlap_sample_to_scenario(self, sample: Dict) -> Optional[Dict]:
        """FIXED: Convert original overlap sample to PriorityNet scenario"""
        
        try:
            metadata = sample.get('metadata', {})
            signal_parameters = metadata.get('signal_parameters', [])
            
            if len(signal_parameters) < 1:
                return None
            
            # Convert each signal to PriorityNet format
            true_parameters = []
            
            # Get total network SNR and distribute among signals
            total_network_snr = metadata.get('network_snr', 20.0)
            n_signals = len(signal_parameters)
            
            for i, sig_param in enumerate(signal_parameters):
                # Distribute SNR among signals (approximate)
                if n_signals > 1:
                    individual_snr = total_network_snr / np.sqrt(n_signals)
                else:
                    individual_snr = total_network_snr
                
                # Use original event type if available, otherwise classify by masses
                original_event_type = sig_param.get('event_type', metadata.get('event_type', 'overlap'))
                if original_event_type == 'overlap':
                    # Classify by masses for overlap cases
                    m1, m2 = sig_param.get('mass_1', 30.0), sig_param.get('mass_2', 25.0)
                    if m1 <= 3.0 and m2 <= 3.0:
                        original_event_type = 'BNS'
                    elif (m1 <= 3.0 and m2 > 3.0) or (m1 > 3.0 and m2 <= 3.0):
                        original_event_type = 'NSBH'
                    else:
                        original_event_type = 'BBH'
                
                priority_param = {
                    'mass_1': sig_param.get('mass_1', 30.0),
                    'mass_2': sig_param.get('mass_2', 25.0),
                    'luminosity_distance': sig_param.get('luminosity_distance', 500.0),
                    'network_snr': individual_snr,
                    'individual_snrs': {det: individual_snr * 0.8 for det in metadata.get('detector_network', ['H1', 'L1'])},
                    'ra': sig_param.get('ra', 0.0),
                    'dec': sig_param.get('dec', 0.0),
                    'theta_jn': sig_param.get('theta_jn', 0.0),
                    'psi': sig_param.get('psi', 0.0),
                    'phase': sig_param.get('phase', 0.0),
                    'geocent_time': sig_param.get('geocent_time', i * 0.5),
                    'a1': sig_param.get('a1', 0.0),
                    'a2': sig_param.get('a2', 0.0),
                    'approximant': sig_param.get('approximant', 'IMRPhenomD'),
                    'event_type': original_event_type,  # FIXED: Proper event type
                    'edge_case': sig_param.get('edge_case', False),
                    'edge_case_type': sig_param.get('edge_case_type'),
                    'detector_network': metadata.get('detector_network', ['H1', 'L1']),
                    'sample_id': f"{sample.get('sample_id', 'unknown')}_signal_{i}",
                    # Overlap-specific information
                    'is_overlap': True,
                    'overlap_index': i,
                    'n_overlapping_signals': n_signals,
                    'time_separation': sig_param.get('geocent_time', i * 0.5),
                    'snr_in_overlap': individual_snr
                }
                true_parameters.append(priority_param)
            
            scenario = {
                'scenario_id': sample.get('sample_id', 'unknown'),
                'true_parameters': true_parameters,
                'baseline_biases': [],
                'detector_data': sample.get('detector_data', {}),
                'whitened_data': sample.get('whitened_data', {}),
                'metadata': {
                    **metadata,
                    'scenario_type': 'original_overlap',
                    'n_signals': n_signals
                }
            }
            
            return scenario
            
        except Exception as e:
            self.logger.debug(f"Failed to convert overlap sample: {e}")
            return None
    
    def _convert_single_sample_to_scenario(self, sample: Dict) -> Optional[Dict]:
        """FIXED: Convert single GW sample to PriorityNet scenario"""
        
        try:
            metadata = sample.get('metadata', {})
            signal_parameters = metadata.get('signal_parameters', [])
            
            # FIXED: Handle samples without signal_parameters (shouldn't happen for non-noise)
            if not signal_parameters:
                # Check if this is actually a mislabeled noise sample
                event_type = metadata.get('event_type', 'unknown')
                if event_type == 'noise':
                    return self._convert_noise_sample_to_scenario(sample)
                else:
                    self.logger.warning(f"Non-noise sample without signal_parameters: {sample.get('sample_id')}")
                    return None
            
            # Convert to PriorityNet format
            true_parameters = []
            
            for sig_param in signal_parameters:
                # FIXED: Use original event_type from metadata, not mass-based classification
                original_event_type = metadata.get('event_type', 'BBH')
                
                priority_param = {
                    'mass_1': sig_param.get('mass_1', 30.0),
                    'mass_2': sig_param.get('mass_2', 25.0),
                    'luminosity_distance': sig_param.get('luminosity_distance', 500.0),
                    'network_snr': metadata.get('network_snr', 10.0),
                    'individual_snrs': metadata.get('individual_snrs', {}),
                    'ra': sig_param.get('ra', 0.0),
                    'dec': sig_param.get('dec', 0.0),
                    'theta_jn': sig_param.get('theta_jn', 0.0),
                    'psi': sig_param.get('psi', 0.0),
                    'phase': sig_param.get('phase', 0.0),
                    'geocent_time': sig_param.get('geocent_time', 0.0),
                    'a1': sig_param.get('a1', 0.0),
                    'a2': sig_param.get('a2', 0.0),
                    'tilt1': sig_param.get('tilt1', 0.0),
                    'tilt2': sig_param.get('tilt2', 0.0),
                    'approximant': sig_param.get('approximant', 'IMRPhenomD'),
                    'event_type': original_event_type,  # FIXED: Use original classification
                    'edge_case': sig_param.get('edge_case', False),
                    'edge_case_type': sig_param.get('edge_case_type'),
                    'detector_network': metadata.get('detector_network', ['H1', 'L1']),
                    'sample_id': sample.get('sample_id', 'unknown'),
                    # Additional parameters
                    'total_mass': sig_param.get('total_mass', sig_param.get('mass_1', 30.0) + sig_param.get('mass_2', 25.0)),
                    'chirp_mass': sig_param.get('chirp_mass', 0.0),
                    'effective_spin': sig_param.get('effective_spin', 0.0),
                    'lambda_1': sig_param.get('lambda_1', 0),
                    'lambda_2': sig_param.get('lambda_2', 0),
                    'distance_mpc': sig_param.get('luminosity_distance', 500.0),
                    'snr_regime': metadata.get('snr_regime', 'medium'),
                    'difficulty_assessment': metadata.get('difficulty_assessment', 'medium')
                }
                true_parameters.append(priority_param)
            
            scenario = {
                'scenario_id': sample.get('sample_id', 'unknown'),
                'true_parameters': true_parameters,
                'baseline_biases': [],  # Could be computed if needed
                'detector_data': sample.get('detector_data', {}),
                'whitened_data': sample.get('whitened_data', {}),
                'metadata': metadata
            }
            
            return scenario
            
        except Exception as e:
            self.logger.debug(f"Failed to convert single sample {sample.get('sample_id', 'unknown')}: {e}")
            return None

    def _convert_multi_sample_to_scenario(self, sample: Dict) -> Optional[Dict]:
        """FIXED: Convert multi-signal GW sample to PriorityNet scenario (legacy support)"""
        
        try:
            metadata = sample.get('metadata', {})
            signal_parameters = metadata.get('signal_parameters', [])
            
            if len(signal_parameters) < 2:
                # If it's actually a single signal, convert as single
                if len(signal_parameters) == 1:
                    return self._convert_single_sample_to_scenario(sample)
                else:
                    return None
            
            # This is essentially the same as overlap conversion
            return self._convert_overlap_sample_to_scenario(sample)
            
        except Exception as e:
            self.logger.debug(f"Failed to convert multi-sample: {e}")
            return None
    
    def _create_artificial_overlap_scenario(self, single_samples: List[Dict]) -> Optional[Dict]:
        """FIXED: Create artificial overlap scenario with proper event type tracking"""
        
        try:
            # Randomly select 2-3 samples to combine
            n_signals = random.choice([2, 3])
            if len(single_samples) < n_signals:
                return None
                
            selected_samples = random.sample(single_samples, n_signals)
            
            combined_parameters = []
            combined_detector_data = {}
            combined_whitened_data = {}
            
            # Get detector list from first sample
            detectors = list(selected_samples[0].get('detector_data', {}).keys())
            
            # Initialize combined data
            for detector in detectors:
                combined_detector_data[detector] = np.zeros_like(
                    selected_samples[0]['detector_data'][detector])
                combined_whitened_data[detector] = np.zeros_like(
                    selected_samples[0]['whitened_data'][detector])
            
            # Combine samples
            for i, sample in enumerate(selected_samples):
                metadata = sample.get('metadata', {})
                signal_params = metadata.get('signal_parameters', [])
                
                if signal_params:
                    sig_param = signal_params[0]
                    
                    # Adjust timing and SNR for overlap
                    time_offset = i * random.uniform(0.2, 1.0)
                    snr_reduction = random.uniform(0.6, 0.8)
                    
                    # Use original event type from the sample
                    original_event_type = metadata.get('event_type', 'BBH')
                    
                    priority_param = {
                        'mass_1': sig_param.get('mass_1', 30.0),
                        'mass_2': sig_param.get('mass_2', 25.0),
                        'luminosity_distance': sig_param.get('luminosity_distance', 500.0),
                        'network_snr': metadata.get('network_snr', 10.0) * snr_reduction,
                        'individual_snrs': {det: metadata.get('network_snr', 10.0) * snr_reduction * 0.8 
                                        for det in detectors},
                        'ra': sig_param.get('ra', 0.0),
                        'dec': sig_param.get('dec', 0.0),
                        'theta_jn': sig_param.get('theta_jn', 0.0),
                        'psi': sig_param.get('psi', 0.0),
                        'phase': sig_param.get('phase', 0.0),
                        'geocent_time': time_offset,
                        'a1': sig_param.get('a1', 0.0),
                        'a2': sig_param.get('a2', 0.0),
                        'approximant': sig_param.get('approximant', 'IMRPhenomD'),
                        'event_type': original_event_type,  # FIXED: Preserve original type
                        'edge_case': sig_param.get('edge_case', False),
                        'detector_network': detectors,
                        'sample_id': f"artificial_overlap_{i}",
                        'is_overlap': True,
                        'artificial': True,
                        'overlap_index': i,
                        'n_overlapping_signals': n_signals
                    }
                    combined_parameters.append(priority_param)
                    
                    # Add detector data (simplified combination)
                    for detector in detectors:
                        if detector in sample.get('detector_data', {}):
                            combined_detector_data[detector] += sample['detector_data'][detector] * snr_reduction
                            combined_whitened_data[detector] += sample['whitened_data'][detector] * snr_reduction
            
            # Create combined scenario
            scenario = {
                'scenario_id': f"artificial_overlap_{random.randint(1000, 9999)}",
                'true_parameters': combined_parameters,
                'baseline_biases': [],
                'detector_data': combined_detector_data,
                'whitened_data': combined_whitened_data,
                'metadata': {
                    'event_type': 'overlap',
                    'n_signals': len(combined_parameters),
                    'overlap_type': 'multi_signal',
                    'detector_network': detectors,
                    'network_snr': sum(p['network_snr'] for p in combined_parameters),
                    'artificial': True,
                    'scenario_type': 'artificial_overlap'
                }
            }
            
            return scenario
            
        except Exception as e:
            self.logger.debug(f"Failed to create artificial overlap: {e}")
            return None
        
    
class PriorityNetDataset(Dataset):
    """Enhanced PriorityNet dataset with signal-type awareness"""
    
    def __init__(self, scenarios: List[Dict], split_name: str = 'train'):
        self.data = []
        self.split_name = split_name
        self.logger = logging.getLogger(__name__)
        
        bbh_count = bns_count = nsbh_count = noise_count = overlap_count = 0
        
        for scenario_id, scenario in enumerate(scenarios):
            try:
                true_params = scenario.get('true_parameters', [])
                baseline_results = scenario.get('baseline_biases', [])
                
                if not true_params:
                    continue
                
                #  FIXED: Extract ALL detections, not just first
                detections = []
                class_ids = []
                
                for signal in true_params:  # Process ALL signals
                    # Extract detection parameters
                    detection = {
                        'mass_1': signal.get('mass_1', 30.0),
                        'mass_2': signal.get('mass_2', 25.0),
                        'luminosity_distance': signal.get('luminosity_distance', 500.0),
                        'network_snr': signal.get('network_snr', 10.0),
                        'individual_snrs': signal.get('individual_snrs', {}),
                        'ra': signal.get('ra', 0.0),
                        'dec': signal.get('dec', 0.0),
                        'theta_jn': signal.get('theta_jn', 0.0),
                        'psi': signal.get('psi', 0.0),
                        'phase': signal.get('phase', 0.0),
                        'geocent_time': signal.get('geocent_time', 0.0),
                        'a_1': signal.get('a_1', 0.0),
                        'a_2': signal.get('a_2', 0.0),
                        'tilt_1': signal.get('tilt_1', 0.0),
                        'tilt_2': signal.get('tilt_2', 0.0),
                        'phi_12': signal.get('phi_12', 0.0),
                        'phi_jl': signal.get('phi_jl', 0.0),
                        'approximant': signal.get('approximant', 'IMRPhenomD'),
                        'event_type': signal.get('event_type', 'BBH'),
                        'edge_case': signal.get('edge_case', False),
                        'detector_network': signal.get('detector_network', ['H1', 'L1']),
                        'sample_id': signal.get('sample_id', 'unknown')
                    }
                    detections.append(detection)
                    
                    # Get event type for classification
                    evt_raw = signal.get('event_type', None)
                    evt = str(evt_raw).strip().lower() if evt_raw is not None else None
                    
                    if evt and 'bbh' in evt:
                        class_id = 0
                        bbh_count += 1
                    elif evt and 'bns' in evt:
                        class_id = 1
                        bns_count += 1
                    elif evt and 'nsbh' in evt:
                        class_id = 2
                        nsbh_count += 1
                    else:
                        class_id = 3  # noise or unknown
                        noise_count += 1
                    
                    class_ids.append(class_id)
                
                #  FIXED: Compute priorities for ALL detections
                priorities = self._compute_extraction_priorities(true_params, baseline_results)
                
                if priorities is None or len(priorities) == 0:
                    continue
                
                #  Verify all have same length
                if len(detections) != len(priorities) or len(detections) != len(class_ids):
                    self.logger.warning(f"Length mismatch in scenario {scenario_id}: "
                                      f"detections={len(detections)}, priorities={len(priorities)}, "
                                      f"class_ids={len(class_ids)}")
                    continue
                
                # Ensure priorities are tensor
                if not isinstance(priorities, torch.Tensor):
                    priorities = torch.tensor(priorities, dtype=torch.float32)
                
                # Clean priorities (no NaN/Inf)
                priorities = torch.where(torch.isnan(priorities), torch.tensor(0.5), priorities)
                priorities = torch.where(torch.isinf(priorities), torch.tensor(1.0), priorities)
                
                # Store complete scenario
                scenario_data = {
                    'scenario_id': scenario.get('scenario_id', f'scenario_{scenario_id}'),
                    'detections': detections,  #  ALL detections
                    'priorities': priorities,   #  ALL priorities
                    'class_ids': torch.tensor(class_ids, dtype=torch.long),  #  ALL class IDs
                    'metadata': scenario.get('metadata', {})
                }
                
                self.data.append(scenario_data)
                
                # Track overlaps
                if len(detections) > 1:
                    overlap_count += 1
                
            except Exception as e:
                self.logger.warning(f"Error processing scenario {scenario_id}: {e}")
                continue
        
        # Log statistics
        total = bbh_count + bns_count + nsbh_count + noise_count
        self.logger.info(f" {split_name.upper()} PriorityNet dataset created: {len(self.data)} scenarios")
        self.logger.info(f"   BBH: {bbh_count} ({bbh_count/max(total,1)*100:.1f}%)")
        self.logger.info(f"   BNS: {bns_count} ({bns_count/max(total,1)*100:.1f}%)")
        self.logger.info(f"   NSBH: {nsbh_count} ({nsbh_count/max(total,1)*100:.1f}%)")
        self.logger.info(f"   Noise: {noise_count} ({noise_count/max(total,1)*100:.1f}%)")
        self.logger.info(f"   Overlap: {overlap_count} ({overlap_count/max(len(self.data),1)*100:.1f}%)")
    
    def _compute_extraction_priorities(self, signals: List[Dict], 
                                baseline_biases: Optional[List[Dict]] = None) -> torch.Tensor:
        """
        COMPLETE FIXED: Enhanced priority computation for all signal types
        
        Handles:
        - Noise samples (lowest priority)
        - BBH, BNS, NSBH with optimized parameters
        - Overlap scenarios
        - Edge cases
        - Original event type preservation
        - Distance/mass/SNR scaling
        - Bias corrections
        """
        
        n_signals = len(signals)
        priorities = torch.zeros(n_signals)
         
        for i, signal in enumerate(signals):
            try:
                # FIXED: Use original event_type, handle noise properly (normalize to upper)
                evt_raw = signal.get('event_type', 'BBH')
                event_type = str(evt_raw).strip().upper() if evt_raw is not None else 'BBH'
                
                # Handle noise samples - give them lowest but non-zero priority
                if event_type == 'NOISE':
                    # Noise gets consistent low priority  with small random variation
                    priorities[i] = 0.0 
                    
                    # Do not proceed to further classification/logging for noise
                    continue
                
                # Basic signal properties with safety checks (robust casting)
                def _as_float(val, default):
                    try:
                        return float(val)
                    except Exception:
                        return float(default)
                    
                    
                snr = max(0.1, _as_float(signal.get('network_snr', 10.0), 10.0))
                m1 = max(0.1, _as_float(signal.get('mass_1', 30.0), 30.0))
                m2 = max(0.1, _as_float(signal.get('mass_2', 25.0), 25.0))
                distance = max(1.0, _as_float(signal.get('luminosity_distance', 500.0), 500.0))
                
                # Ensure proper mass ordering
                if m2 > m1:
                    m1, m2 = m2, m1
                
                # Use original event type instead of mass-based classification
                if event_type not in ['BBH', 'BNS', 'NSBH']:
                    # Fallback to mass-based classification for unknown/invalid types
                    if m1 <= 3.0 and m2 <= 3.0:
                        signal_type = 'BNS'
                    elif (m1 <= 3.0 and m2 > 3.0) or (m1 > 3.0 and m2 <= 3.0):
                        signal_type = 'NSBH'
                    else:
                        signal_type = 'BBH'
                    
                    # Avoid misleading logs for noise which is handled above
                    if event_type not in ['NOISE']:
                        self.logger.debug(f"Signal {i}: Unknown event_type '{event_type}', classified as {signal_type}")
                else:
                    signal_type = event_type
                
                # Derived quantities with safety checks
                total_mass = m1 + m2
                if total_mass > 0:
                    chirp_mass = (m1 * m2)**(3/5) / total_mass**(1/5)
                    mass_ratio = m2 / m1
                    symmetric_mass_ratio = (m1 * m2) / total_mass**2
                else:
                    chirp_mass = 1.0
                    mass_ratio = 0.5
                    symmetric_mass_ratio = 0.25
                
                #  ENHANCED: SNR priority with logarithmic scaling for high SNR
                if snr >= 50.0:
                    snr_priority = 1.4 + 0.1 * np.log10(snr / 50.0)  # Extra bonus for very high SNR
                elif snr >= 20.0:
                    snr_priority = 1.0 + 0.15 * np.log10(snr / 20.0)  # Bonus for high SNR
                elif snr >= 12.0:
                    snr_priority = snr / 12.0  # Linear scaling in medium range
                elif snr >= 8.0:
                    snr_priority = 0.7 + 0.3 * (snr - 8.0) / 4.0  # Gentle scaling for low SNR
                else:
                    snr_priority = 0.4 + 0.3 * snr / 8.0  # Very low SNR handling
                
                snr_priority = min(snr_priority, 1.5)  # Cap at 1.5
                
                #  ENHANCED: Mass priority optimized for each signal type
                if signal_type == 'BNS':
                    # BNS: Favor canonical masses, don't heavily penalize outliers
                    if 2.0 <= total_mass <= 3.5:  # Canonical BNS range
                        mass_priority = 1.0
                    elif 1.8 <= total_mass < 2.0:  # Light BNS
                        mass_priority = 0.95
                    elif 3.5 < total_mass <= 4.5:  # Heavy BNS
                        mass_priority = 0.9
                    elif 4.5 < total_mass <= 6.0:  # Very heavy BNS (still detectable)
                        mass_priority = 0.8
                    else:  # Extreme cases
                        mass_priority = 0.7
                        
                elif signal_type == 'NSBH':
                    # NSBH: Wide mass range acceptance
                    if 4.0 <= total_mass <= 35.0:  # Standard NSBH
                        mass_priority = 1.0
                    elif 2.5 <= total_mass < 4.0:  # Light NSBH
                        mass_priority = 0.9
                    elif 35.0 < total_mass <= 80.0:  # Heavy NSBH
                        mass_priority = 0.95
                    elif 80.0 < total_mass <= 150.0:  # Very heavy NSBH
                        mass_priority = 0.85
                    else:  # Extreme cases
                        mass_priority = 0.75
                        
                else:  # BBH
                    #  ENHANCED: Excellent support for all BBH masses
                    if 15.0 <= total_mass <= 50.0:  # Stellar mass BBH (optimal)
                        mass_priority = 1.0
                    elif 50.0 < total_mass <= 100.0:  # Intermediate mass BBH
                        mass_priority = 1.05  # Slight bonus for intermediate mass
                    elif 100.0 < total_mass <= 200.0:  # Heavy BBH (GW190521-like)
                        mass_priority = 1.1   # Bonus for heavy BBH (astrophysically interesting)
                    elif 200.0 < total_mass <= 400.0:  # Very heavy BBH
                        mass_priority = 1.0   # Still very valuable
                    elif 8.0 <= total_mass < 15.0:  # Light BBH
                        mass_priority = 0.95
                    elif 5.0 <= total_mass < 8.0:  # Very light BBH
                        mass_priority = 0.9
                    else:  # Extreme masses
                        mass_priority = 0.8
                
                #  ENHANCED: Distance priority with better distant signal support
                if chirp_mass > 0:
                    # Chirp mass scaling for detection range
                    chirp_mass_factor = (chirp_mass / 30.0)**(5/6)
                    
                    # Signal-type dependent horizon scaling
                    if signal_type == 'BBH':
                        base_horizon = 1000.0  # BBH detectable further
                    elif signal_type == 'NSBH':
                        base_horizon = 800.0   # NSBH intermediate
                    else:  # BNS
                        base_horizon = 200.0   # BNS closer detection
                    
                    effective_horizon = base_horizon * chirp_mass_factor
                else:
                    effective_horizon = 500.0
                
                if distance <= effective_horizon:
                    distance_priority = 1.0
                elif distance <= 2.0 * effective_horizon:
                    # Gradual falloff for distant signals
                    distance_priority = 0.6 + 0.4 * (2.0 * effective_horizon - distance) / effective_horizon
                elif distance <= 5.0 * effective_horizon:
                    # Extended range for very high mass systems
                    distance_priority = 0.3 + 0.3 * (5.0 * effective_horizon - distance) / (3.0 * effective_horizon)
                else:
                    # Very distant signals still get some priority
                    distance_priority = max(0.1, effective_horizon / distance)
                
                #  ENHANCED: Advanced detectability factors
                base_detectability = 1.0
                
                # Chirp mass bonus (higher chirp mass = louder signal)
                if chirp_mass >= 60.0:  # Very high chirp mass
                    chirp_bonus = 0.2
                elif chirp_mass >= 40.0:  # High chirp mass
                    chirp_bonus = 0.15
                elif chirp_mass >= 25.0:  # Moderate high chirp mass
                    chirp_bonus = 0.1
                elif chirp_mass >= 15.0:  # Standard chirp mass
                    chirp_bonus = 0.05
                elif chirp_mass >= 5.0:   # Low chirp mass
                    chirp_bonus = 0.02
                else:  # Very low chirp mass
                    chirp_bonus = 0.0
                
                # Mass ratio factor (symmetric masses generally easier to detect)
                if mass_ratio >= 0.9:  # Nearly equal masses
                    mass_ratio_bonus = 0.08
                elif mass_ratio >= 0.8:  # Close to equal
                    mass_ratio_bonus = 0.05
                elif mass_ratio >= 0.6:  # Moderate asymmetry
                    mass_ratio_bonus = 0.03
                elif mass_ratio >= 0.4:  # High asymmetry
                    mass_ratio_bonus = 0.01
                elif mass_ratio >= 0.2:  # Very high asymmetry
                    mass_ratio_bonus = 0.0
                else:  # Extreme asymmetry
                    mass_ratio_bonus = -0.02  # Small penalty for extreme asymmetry
                
                # Symmetric mass ratio factor (peaks at 0.25)
                eta_factor = 4.0 * symmetric_mass_ratio * (1.0 - symmetric_mass_ratio)
                eta_bonus = 0.05 * eta_factor
                
                detectability = base_detectability + chirp_bonus + mass_ratio_bonus + eta_bonus
                
                #  ENHANCED: Special handling for extreme and interesting cases
                extreme_bonus = 0.0
                
                # Very high mass bonus (interesting astrophysics)
                if total_mass >= 150.0:
                    extreme_bonus += 0.15  # Very interesting for astrophysics
                elif total_mass >= 80.0:
                    extreme_bonus += 0.1   # High mass bonus
                elif total_mass >= 50.0:
                    extreme_bonus += 0.05  # Moderate mass bonus
                
                # Very distant but high SNR bonus (rare but important)
                if distance >= 3000.0 and snr >= 15.0:
                    extreme_bonus += 0.15  # Exceptional detection
                elif distance >= 2000.0 and snr >= 12.0:
                    extreme_bonus += 0.1   # Very good distant detection
                elif distance >= 1000.0 and snr >= 10.0:
                    extreme_bonus += 0.05  # Good distant detection
                
                # Very high SNR bonus (regardless of other factors)
                if snr >= 100.0:
                    extreme_bonus += 0.2   # Exceptional SNR
                elif snr >= 50.0:
                    extreme_bonus += 0.15  # Very high SNR
                elif snr >= 30.0:
                    extreme_bonus += 0.1   # High SNR bonus
                
                # Low frequency bonus (longer inspiral, more information)
                f_lower = _as_float(signal.get('f_lower', 20.0), 20.0)
                if f_lower <= 10.0:
                    extreme_bonus += 0.1   # Very low frequency start
                elif f_lower <= 15.0:
                    extreme_bonus += 0.05  # Low frequency start
                
                # Edge case bonus
                if signal.get('edge_case', False):
                    edge_case_type = signal.get('edge_case_type', 'unknown')
                    if edge_case_type in ['high_spin', 'eccentric']:
                        extreme_bonus += 0.08  # Higher bonus for challenging cases
                    elif edge_case_type in ['short_bbh', 'long_bns']:
                        extreme_bonus += 0.06  # Medium bonus
                    else:
                        extreme_bonus += 0.04  # Standard edge case bonus
                
                # Overlap handling bonus
                if signal.get('is_overlap', False):
                    n_overlapping = signal.get('n_overlapping_signals', 1)
                    if n_overlapping >= 3:
                        extreme_bonus += 0.1   # Complex overlap scenario
                    elif n_overlapping == 2:
                        extreme_bonus += 0.05  # Standard overlap
                    
                    # Time separation factor
                    time_sep = abs(signal.get('time_separation', 0.0))
                    if time_sep < 0.5:
                        extreme_bonus += 0.05  # Close in time (harder)
                
                # Spin magnitude bonus
                a1 = _as_float(signal.get('a1', 0.0), 0.0)
                a2 = _as_float(signal.get('a2', 0.0), 0.0)
                max_spin = max(abs(a1), abs(a2))
                if max_spin >= 0.9:
                    extreme_bonus += 0.08  # Very high spin
                elif max_spin >= 0.7:
                    extreme_bonus += 0.05  # High spin
                elif max_spin >= 0.5:
                    extreme_bonus += 0.02  # Moderate spin
                
                # Tidal parameter bonus (for BNS/NSBH)
                if signal_type in ['BNS', 'NSBH']:
                    lambda_1 = _as_float(signal.get('lambda_1', 0), 0.0)
                    lambda_2 = _as_float(signal.get('lambda_2', 0), 0.0)
                    max_lambda = max(lambda_1, lambda_2)
                    if max_lambda > 1000:
                        extreme_bonus += 0.05  # High tidal deformability
                    elif max_lambda > 500:
                        extreme_bonus += 0.03  # Moderate tidal effects
                
                #  ENHANCED: Baseline bias penalty (if available)
                bias_penalty = 0.0
                if baseline_biases and i < len(baseline_biases) and baseline_biases[i]:
                    try:
                        bias_values = [abs(float(b)) for b in baseline_biases[i].values() 
                                    if isinstance(b, (int, float)) and not np.isnan(float(b))]
                        if bias_values:
                            bias_magnitude = np.mean(bias_values)
                            # Scale penalty based on bias severity
                            if bias_magnitude > 0.5:
                                bias_penalty = 0.15  # Severe bias
                            elif bias_magnitude > 0.3:
                                bias_penalty = 0.12  # High bias
                            elif bias_magnitude > 0.1:
                                bias_penalty = 0.08  # Moderate bias
                            else:
                                bias_penalty = 0.04  # Small bias
                    except Exception as e:
                        self.logger.debug(f"Bias calculation error for signal {i}: {e}")
                        bias_penalty = 0.0
                
                #  ENHANCED: SNR regime bonus/penalty
                snr_regime = signal.get('snr_regime', 'medium')
                snr_regime_modifier = 0.0
                if snr_regime == 'loud':
                    snr_regime_modifier = 0.05   # Loud signals get small bonus
                elif snr_regime == 'weak':
                    snr_regime_modifier = 0.03   # Weak signals get small bonus (challenging)
                elif snr_regime == 'low':
                    snr_regime_modifier = 0.02   # Low SNR get small bonus
                
                #  OPTIMIZED: Final priority formula
                # Weights: SNR is most important, then distance, then mass
                base_priority = (
                    0.40 * snr_priority +      # SNR weight slightly reduced
                    0.30 * distance_priority + # Distance weight increased
                    0.25 * mass_priority +     # Mass weight
                    0.05 * detectability       # Detectability factors
                ) * (1.0 + extreme_bonus + snr_regime_modifier) - bias_penalty
                
                #  REDUCED: Even smaller hierarchy penalty to avoid artificial ordering
                hierarchy_penalty = i * 0.002  # Very small penalty
                
                #  ENHANCED: Adaptive minimum priority based on signal quality
                if snr >= 15.0:
                    min_priority = 0.4  # High SNR signals
                elif snr >= 10.0:
                    min_priority = 0.35  # Medium SNR signals
                elif snr >= 8.0:
                    min_priority = 0.3   # Low SNR signals
                else:
                    min_priority = 0.25  # Very low SNR signals
                
                # Special minimum for edge cases and overlaps
                if signal.get('edge_case', False) or signal.get('is_overlap', False):
                    min_priority = max(min_priority, 0.35)
                
                final_priority = max(min_priority, base_priority - hierarchy_penalty)
                
                # Ensure reasonable bounds
                final_priority = min(max(final_priority, 0.1), 2.0)
                
                priorities[i] = final_priority
                
                # Enhanced debug logging for first few samples
                # if i < 5 or (i < 20 and event_type in ['noise', 'overlap']):
                #     self.logger.debug(
                #         f"Signal {i}: {signal_type}, M={total_mass:.1f}, D={distance:.0f}, "
                #         f"SNR={snr:.1f}, P={final_priority:.3f} "
                #         f"(SNR:{snr_priority:.2f}, Mass:{mass_priority:.2f}, Dist:{distance_priority:.2f})"
                #     )
            
            except Exception as e:
                self.logger.warning(f"Error computing priority for signal {i}: {e}")
                # Assign safe default priority
                priorities[i] = 0.5
                continue
        
        # Final validation and normalization
        priorities = torch.clamp(priorities, min=0.01, max=2.0)
        
        # Ensure no NaN or infinite values
        priorities = torch.where(torch.isnan(priorities), torch.tensor(0.5), priorities)
        priorities = torch.where(torch.isinf(priorities), torch.tensor(1.0), priorities)
        
        return priorities

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """FIXED: Ensure proper tensor types returned"""
        
        item = self.data[idx].copy()  
        
        #  FIXED: Ensure priorities are proper tensors
        priorities = item.get('priorities')
        if not isinstance(priorities, torch.Tensor):
            if isinstance(priorities, (list, tuple)):
                item['priorities'] = torch.tensor(priorities, dtype=torch.float32)
            else:
                item['priorities'] = torch.tensor([priorities], dtype=torch.float32)
        
        #  FIXED: Ensure class_ids are proper tensors  
        class_ids = item.get('class_ids')
        if not isinstance(class_ids, torch.Tensor):
            if isinstance(class_ids, (list, tuple)):
                item['class_ids'] = torch.tensor(class_ids, dtype=torch.long)
            else:
                item['class_ids'] = torch.tensor([class_ids], dtype=torch.long)
        
        return item



def _config_get(cfg: Any, key: str, default: Any) -> Any:
    """Safely read a config value from either an object or a dict."""
    try:
        if hasattr(cfg, key):
            return getattr(cfg, key)
        if isinstance(cfg, dict):
            return cfg.get(key, default)
    except Exception:
        pass
    return default

def collate_priority_batch(batch: List[Dict]) -> Tuple[List[List[Dict]], List[torch.Tensor], List[torch.Tensor]]:
    """Collate function for variable-length sequences"""
    
    detections_batch = []
    priorities_batch = []
    class_batch = []
    
    for item in batch:
        detections_batch.append(item['detections'])
        priorities_batch.append(item['priorities'])
        class_batch.append(item.get('class_ids', torch.zeros(len(item['detections']), dtype=torch.long)))
    
    return detections_batch, priorities_batch, class_batch

def create_data_splits(scenarios: List[Dict], train_ratio: float = 0.7, 
                      val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create stratified train/validation/test splits"""
    
    logging.info(f"ðŸ“Š Creating scenario splits: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test")
    
    # First split: train vs (val + test)
    train_scenarios, temp_scenarios = train_test_split(
        scenarios, 
        test_size=(val_ratio + test_ratio), 
        random_state=42
    )
    
    # Second split: val vs test
    val_scenarios, test_scenarios = train_test_split(
        temp_scenarios,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=42
    )
    
    logging.info(f" Scenario splits created:")
    logging.info(f"   Train: {len(train_scenarios)} scenarios")
    logging.info(f"   Validation: {len(val_scenarios)} scenarios") 
    logging.info(f"   Test: {len(test_scenarios)} scenarios")
    
    return train_scenarios, val_scenarios, test_scenarios

def train_priority_net_with_validation(config, train_dataset: PriorityNetDataset, 
                                      val_dataset: PriorityNetDataset, 
                                      output_dir: Path) -> Dict[str, Any]:
    """Enhanced training with validation monitoring"""
    
    
    logging.info("ðŸ§  Training PriorityNet with validation...")
    
    # Initialize model and trainer
    model = PriorityNet(config)
    trainer = PriorityNetTrainer(model, config)
    
    # Trainer hyperparameters from config (attribute or dict style)
    batch_size = _config_get(config, 'batch_size', 8)
    n_epochs = _config_get(config, 'epochs', 200)
    patience = _config_get(config, 'patience', 25)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # from config
        shuffle=True,
        collate_fn=collate_priority_batch,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_priority_batch,
        num_workers=0
    )
    
    # Training parameters
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Track metrics
    training_metrics = {
        'train_losses': [],
        'val_losses': [],
        'epochs_completed': 0,
        'best_epoch': 0
    }
    
    logging.info(f"ðŸš€ Starting training: {n_epochs} epochs, patience={patience}")
    
    # Enhanced Training loop with comprehensive diagnostics
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_ranking_losses = []
        train_priority_losses = []
        train_grad_norms = []
        
        train_pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{n_epochs}')
        for detections_batch, priorities_batch, in train_pbar:
            loss_info = trainer.train_step(detections_batch, priorities_batch)
            
            train_losses.append(loss_info['loss'])
            train_ranking_losses.append(loss_info.get('ranking_loss', 0))
            train_priority_losses.append(loss_info.get('priority_loss', 0))
            train_grad_norms.append(loss_info.get('grad_norm', 0))
            
            #  Enhanced progress bar with component breakdown
            train_pbar.set_postfix({
                'Loss': f"{loss_info['loss']:.4f}",
                'Rank': f"{loss_info.get('ranking_loss', 0):.3f}",
                'Prior': f"{loss_info.get('priority_loss', 0):.3f}",
                'Grad': f"{loss_info.get('grad_norm', 0):.1e}",
                'Valid': loss_info.get('valid_batches', 0)
            })
        
        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        avg_ranking_loss = np.mean(train_ranking_losses) if train_ranking_losses else 0.0
        avg_priority_loss = np.mean(train_priority_losses) if train_priority_losses else 0.0
        avg_grad_norm = np.mean(train_grad_norms) if train_grad_norms else 0.0
        
        training_metrics['train_losses'].append(avg_train_loss)
        training_metrics.setdefault('train_ranking_losses', []).append(avg_ranking_loss)
        training_metrics.setdefault('train_priority_losses', []).append(avg_priority_loss)
        training_metrics.setdefault('grad_norms', []).append(avg_grad_norm)
        
        #  Validation phase (fixed to match training)
        model.eval()
        val_losses = []
        val_ranking_losses = []
        val_priority_losses = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{n_epochs}')
            for detections_batch, priorities_batch, class_batch in val_pbar:
                batch_total = 0.0
                batch_ranking = 0.0
                batch_priority = 0.0
                valid_batches = 0
                
                for detections, target_priorities in zip(detections_batch, priorities_batch):
                    if not detections or len(target_priorities) == 0:
                        continue
                    
                    try:
                        predicted_priorities = model(detections)
                        
                        # Handle tuple output (priorities, class_logits)
                        if isinstance(predicted_priorities, tuple):
                            predicted_priorities = predicted_priorities[0]
                        
                        if predicted_priorities.numel() == 0:
                            continue
                        
                        min_len = min(len(predicted_priorities), len(target_priorities))
                        if min_len == 0:
                            continue
                        
                        pred_slice = predicted_priorities[:min_len]
                        target_slice = target_priorities[:min_len].to(pred_slice.device)
                        
    
                        # Combined loss with same weights as training
                        if len(predicted_priorities) >= 2:
                            # Multi-detection: use ranking + priority (same as training)
                            ranking_loss = trainer._compute_ranking_loss(predicted_priorities, target_priorities)
                            priority_loss = trainer._compute_priority_loss(predicted_priorities, target_priorities)

                            combined_loss = (
                                0.85 * ranking_loss +    #  Changed from 0.5
                                0.13 * priority_loss     #  Same
                            )
                        else:
                            # Single detection: only priority (same as training)
                            priority_loss = trainer._compute_priority_loss(predicted_priorities, target_priorities)
                            ranking_loss = torch.tensor(0.0)
                            
                            combined_loss = (
                                0.75 * priority_loss     #  Match training single-detection weight
                            )

                        batch_total += float(combined_loss)
                        batch_ranking += float(ranking_loss)
                        batch_priority += float(priority_loss)
                        valid_batches += 1


                        
                    except Exception as e:
                        logging.debug(f"Validation step error: {e}")
                        continue
                
                if valid_batches > 0:
                    avg_batch_loss = batch_total / valid_batches
                    avg_batch_ranking = batch_ranking / valid_batches
                    avg_batch_priority = batch_priority / valid_batches
                    
                    val_losses.append(avg_batch_loss)
                    val_ranking_losses.append(avg_batch_ranking)
                    val_priority_losses.append(avg_batch_priority)
                    
                    val_pbar.set_postfix({
                        'Loss': f"{avg_batch_loss:.4f}",
                        'Rank': f"{avg_batch_ranking:.3f}",
                        'Prior': f"{avg_batch_priority:.3f}",
                        'Valid': valid_batches
                    })
        
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        avg_val_ranking = np.mean(val_ranking_losses) if val_ranking_losses else 0.0
        avg_val_priority = np.mean(val_priority_losses) if val_priority_losses else 0.0
        
        training_metrics['val_losses'].append(avg_val_loss)
        training_metrics.setdefault('val_ranking_losses', []).append(avg_val_ranking)
        training_metrics.setdefault('val_priority_losses', []).append(avg_val_priority)
        training_metrics['epochs_completed'] = epoch + 1
        
        #  Enhanced logging with diagnostics
        log_msg = (f"Epoch {epoch:3d}: "
                f"Train={avg_train_loss:.6f} (R:{avg_ranking_loss:.4f}, P:{avg_priority_loss:.4f}), "
                f"Val={avg_val_loss:.6f} (R:{avg_val_ranking:.4f}, P:{avg_val_priority:.4f}), "
                f"Grad={avg_grad_norm:.2e}")
        
        logging.info(log_msg)
        
        #  Critical diagnostic warnings
        if epoch == 0:
            if avg_grad_norm < 1e-6:
                logging.error("ðŸš¨ CRITICAL: Vanishing gradients detected at epoch 0! Model not learning.")
                logging.error("   Possible causes:")
                logging.error("   1. Target priorities have no variation")
                logging.error("   2. Model outputs are constant")
                logging.error("   3. Loss function returns zero gradients")
            
            if avg_ranking_loss < 1e-6 and avg_priority_loss < 1e-6:
                logging.error("ðŸš¨ CRITICAL: Both loss components near zero! Check:")
                logging.error("   1. Are target priorities meaningful?")
                logging.error("   2. Is data being loaded correctly?")
            
            if avg_train_loss == avg_val_loss:
                logging.warning("âš ï¸ Train and val losses identical - possible data leak or broken split")
        
        # Check for learning stagnation
        if epoch > 10:
            recent_train = training_metrics['train_losses'][-10:]
            if max(recent_train) - min(recent_train) < 1e-6:
                logging.warning(f"âš ï¸ Training loss flat for 10 epochs: {avg_train_loss:.6f}")
        
        # Early stopping with improved logic
        if avg_val_loss < best_val_loss - 1e-6:  #  Add minimum improvement threshold
            improvement = best_val_loss - avg_val_loss
            best_val_loss = avg_val_loss
            training_metrics['best_epoch'] = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_ranking_loss': avg_ranking_loss,
                'train_priority_loss': avg_priority_loss,
                'grad_norm': avg_grad_norm,
                'config': config.__dict__ if hasattr(config, '__dict__') else {},
                'training_metrics': training_metrics
            }, output_dir / 'priority_net_best.pth')
            
            logging.info(f"ðŸ’¾ Best model saved at epoch {epoch} "
                        f"(val_loss: {avg_val_loss:.6f}, improvement: {improvement:.6f})")
        else:
            patience_counter += 1
            
            if patience_counter >= patience:
                logging.info(f"â¹ï¸ Early stopping at epoch {epoch} "
                            f"(best epoch: {training_metrics['best_epoch']}, "
                            f"best val loss: {best_val_loss:.6f})")
                break
        
        #  Learning rate scheduling (if available)
        if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
            if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                trainer.scheduler.step(avg_val_loss)
                current_lr = trainer.optimizer.param_groups[0]['lr']
                if epoch > 0 and current_lr != training_metrics.get('last_lr', current_lr):
                    logging.info(f"ðŸ“‰ Learning rate reduced to {current_lr:.2e}")
                training_metrics['last_lr'] = current_lr

    #  Plot enhanced training curves
    plot_enhanced_training_curves(training_metrics, output_dir)

    return training_metrics

def plot_enhanced_training_curves(metrics: Dict, output_dir: Path):
    """
    Plot comprehensive training diagnostics with 6 subplots
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('PriorityNet Training Diagnostics', fontsize=16, fontweight='bold')
    
    # ========================================
    # 1. Overall Loss (top-left)
    # ========================================
    ax = axes[0, 0]
    ax.plot(epochs, metrics['train_losses'], 'b-', label='Training Loss', alpha=0.7, linewidth=2)
    ax.plot(epochs, metrics['val_losses'], 'r-', label='Validation Loss', alpha=0.7, linewidth=2)
    
    if 'best_epoch' in metrics:
        best_epoch = metrics['best_epoch'] + 1
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=2, label=f'Best Epoch ({best_epoch})')
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Overall Training Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ========================================
    # 2. Loss Components (top-middle)
    # ========================================
    ax = axes[0, 1]
    if 'train_ranking_losses' in metrics and metrics['train_ranking_losses']:
        ax.plot(epochs, metrics['train_ranking_losses'], 'b-', label='Ranking Loss', alpha=0.7, linewidth=2)
        ax.plot(epochs, metrics['train_priority_losses'], 'r-', label='Priority Loss', alpha=0.7, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Loss Components (Training)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Loss components\nnot tracked', 
                ha='center', va='center', fontsize=12)
        ax.set_title('Loss Components', fontsize=12, fontweight='bold')
    
    # ========================================
    # 3. Gradient Norms (top-right)
    # ========================================
    ax = axes[0, 2]
    if 'grad_norms' in metrics and metrics['grad_norms']:
        ax.plot(epochs, metrics['grad_norms'], 'purple', alpha=0.7, linewidth=2)
        ax.axhline(y=1e-6, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Vanishing (1e-6)')
        ax.axhline(y=100, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Exploding (100)')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Gradient Norm (log scale)', fontsize=11)
        ax.set_title('Gradient Norms', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Gradient norms\nnot tracked', 
                ha='center', va='center', fontsize=12)
        ax.set_title('Gradient Norms', fontsize=12, fontweight='bold')
    
    # ========================================
    # 4. Generalization Gap (bottom-left)
    # ========================================
    ax = axes[1, 0]
    loss_diff = np.array(metrics['val_losses']) - np.array(metrics['train_losses'])
    ax.plot(epochs, loss_diff, 'purple', alpha=0.7, linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1.5)
    ax.fill_between(epochs, 0, loss_diff, where=(loss_diff >= 0), alpha=0.3, color='red', label='Overfitting')
    ax.fill_between(epochs, 0, loss_diff, where=(loss_diff < 0), alpha=0.3, color='green', label='Underfitting')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Val Loss - Train Loss', fontsize=11)
    ax.set_title('Generalization Gap', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ========================================
    # 5. Loss Ratio (bottom-middle)
    # ========================================
    ax = axes[1, 1]
    loss_ratio = np.array(metrics['val_losses']) / (np.array(metrics['train_losses']) + 1e-10)
    ax.plot(epochs, loss_ratio, 'orange', alpha=0.7, linewidth=2)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='Ideal (1.0)')
    ax.fill_between(epochs, 1.0, loss_ratio, where=(loss_ratio >= 1.0), alpha=0.2, color='red')
    ax.fill_between(epochs, loss_ratio, 1.0, where=(loss_ratio < 1.0), alpha=0.2, color='green')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Val Loss / Train Loss', fontsize=11)
    ax.set_title('Loss Ratio (Val/Train)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 2.0])  # Reasonable range
    
    # ========================================
    # 6. Per-Epoch Improvement (bottom-right)
    # ========================================
    ax = axes[1, 2]
    train_improvements = -np.diff(metrics['train_losses'], prepend=metrics['train_losses'][0])
    val_improvements = -np.diff(metrics['val_losses'], prepend=metrics['val_losses'][0])
    
    ax.plot(epochs, train_improvements, 'b-', label='Train Improvement', alpha=0.7, linewidth=2)
    ax.plot(epochs, val_improvements, 'r-', label='Val Improvement', alpha=0.7, linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss Improvement', fontsize=11)
    ax.set_title('Per-Epoch Improvement', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ========================================
    # Finalize and save
    # ========================================
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    
    save_path = output_dir / 'training_curves_enhanced.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"ðŸ“Š Enhanced training curves saved to {save_path}")
    
    # ========================================
    # Also create simplified version
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Simple loss plot
    ax = axes[0]
    ax.plot(epochs, metrics['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, metrics['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    if 'best_epoch' in metrics:
        ax.axvline(x=metrics['best_epoch'] + 1, color='g', linestyle='--', alpha=0.5, linewidth=2, label='Best Epoch')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Simple gap plot
    ax = axes[1]
    loss_diff = np.array(metrics['val_losses']) - np.array(metrics['train_losses'])
    ax.plot(epochs, loss_diff, 'purple', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax.set_title('Generalization Gap', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path_simple = output_dir / 'training_curves.png'
    plt.savefig(save_path_simple, dpi=200, bbox_inches='tight')
    plt.close()
    
    logging.info(f"ðŸ“Š Simple training curves saved to {save_path_simple}")

# def evaluate_priority_net(model: PriorityNet, dataset: PriorityNetDataset,
#                          split_name: str) -> Dict[str, Any]:
#     """
#     FINAL FIX: Robust evaluation for PriorityNet
#     - Handles variable-length (jagged) model outputs
#     - Converts all priorities to flat 1D numpy float arrays
#     - Aligns lengths safely and computes Spearman correlation
#     - Skips invalid samples gracefully with debug logs
#     """
    
#     logging.info(f"ðŸ“Š Evaluating PriorityNet on {split_name} set...")
    
#     model.eval()
    
#     correlations = []
#     precisions = []
#     accuracies = []
#     successful_evaluations = 0
#     total_attempts = 0
    
#     # Helper: normalize model outputs to 1D float array
#     def flatten_pred_priorities(pred_output) -> np.ndarray:
#         """
#         Normalize arbitrary PriorityNet model outputs into flat 1D numpy array.
#         Handles torch.Tensor, list/tuple, nested structures.
#         """
#         scalars = []
        
#         def collect(x):
#             if isinstance(x, torch.Tensor):
#                 if x.numel() == 0:
#                     return
#                 # Take first element if multi-element tensor
#                 scalars.append(float(x.flatten()[0].item()))
#                 return
            
#             if isinstance(x, np.ndarray):
#                 if x.size == 0:
#                     return
#                 scalars.append(float(x.flatten()[0]))
#                 return
            
#             if isinstance(x, (list, tuple)):
#                 for elem in x:
#                     collect(elem)
#                 return
            
#             if isinstance(x, (int, float)):
#                 scalars.append(float(x))
#                 return
        
#         collect(pred_output)
#         return np.asarray(scalars, dtype=np.float32)
    
#     # Import scipy stats
#     from scipy.stats import spearmanr, kendalltau
    
#     with torch.no_grad():
#         for item_idx, item in enumerate(tqdm(dataset, desc=f"Evaluating {split_name}")):
#             total_attempts += 1
            
#             try:
#                 detections = item['detections']
                
#                 # Skip single detection scenarios (ranking undefined)
#                 if len(detections) <= 1:
#                     continue
                
#                 # True priorities -> 1D numpy float array
#                 try:
#                     tp = item['priorities']
#                     if isinstance(tp, torch.Tensor):
#                         true_priorities = tp.detach().cpu().numpy().astype(np.float32).flatten()
#                     elif isinstance(tp, (list, tuple, np.ndarray)):
#                         true_priorities = np.asarray(tp, dtype=np.float32).flatten()
#                     else:
#                         true_priorities = np.asarray([float(tp)], dtype=np.float32)
#                 except Exception as e:
#                     logging.debug(f"True priorities extraction error {item_idx}: {e}")
#                     continue
                
#                 # Predictions -> flatten robustly (handles jagged outputs)
#                 try:
#                     raw_pred = model.forward(detections)
#                     pred_priorities = flatten_pred_priorities(raw_pred)
#                 except Exception as e:
#                     logging.debug(f"Model prediction error {item_idx}: {e}")
#                     continue
                
#                 # Align lengths safely
#                 try:
#                     true_priorities = np.atleast_1d(true_priorities).astype(np.float32)
#                     pred_priorities = np.atleast_1d(pred_priorities).astype(np.float32)
                    
#                     m = min(len(true_priorities), len(pred_priorities))
#                     if m <= 1:
#                         continue
                    
#                     true_priorities = true_priorities[:m]
#                     pred_priorities = pred_priorities[:m]
                    
#                     #  FIXED: Only check finiteness
#                     if not (np.isfinite(true_priorities).all() and np.isfinite(pred_priorities).all()):
#                         continue
                    
#                 except Exception as e:
#                     logging.debug(f"Array processing error {item_idx}: {e}")
#                     continue
                
#                 #  FIXED: Normalize predictions and use Kendall for m==2
#                 try:
#                     # Normalize predictions per-sample
#                     pred_priorities = (pred_priorities - pred_priorities.mean()) / (pred_priorities.std() + 1e-8)
                    
#                     # Use Kendall tau for pairs, Spearman for larger sets
#                     if m == 2:
#                         tau, _ = kendalltau(true_priorities, pred_priorities)
#                         corr = 0.0 if np.isnan(tau) else float(tau)
#                     else:
#                         corr, _ = spearmanr(true_priorities, pred_priorities)
#                         if np.isnan(corr) or np.isinf(corr):
#                             tr = np.argsort(np.argsort(true_priorities))
#                             pr = np.argsort(np.argsort(pred_priorities))
#                             corr = np.corrcoef(tr, pr)[0, 1]
#                             if np.isnan(corr):
#                                 corr = 0.0
                    
#                     correlations.append(float(corr))
                    
#                     #  FIXED: Use k=1 for pairs
#                     k = 1 if m == 2 else min(3, m)
#                     if k > 0:
#                         tr_idx = np.argsort(true_priorities)[::-1][:k]
#                         pr_idx = np.argsort(pred_priorities)[::-1][:k]
#                         precision = len(set(tr_idx) & set(pr_idx)) / k
#                         precisions.append(float(precision))
                    
#                     # Priority accuracy (MSE-based transformed)
#                     pe = np.mean(np.abs(pred_priorities - true_priorities))
#                     acc = 1.0 / (1.0 + float(pe))
#                     accuracies.append(float(acc))
                    
#                     successful_evaluations += 1
                    
#                     if successful_evaluations <= 5:
#                         logging.debug(f" Sample {item_idx}: len={m}, corr={corr:.3f}")
#                         logging.debug(f"   true[:3]={true_priorities[:3]}, pred[:3]={pred_priorities[:3]}")
                    
#                 except Exception as e:
#                     logging.debug(f"Correlation calculation error {item_idx}: {e}")
#                     continue
                
#             except Exception as e:
#                 logging.debug(f"General evaluation error {item_idx}: {e}")
#                 continue
    
#     # Compile results
#     results = {
#         'split': split_name,
#         'n_samples': successful_evaluations,
#         'total_attempts': total_attempts,
#         'success_rate': successful_evaluations / max(total_attempts, 1)
#     }
    
#     if correlations:
#         cor = np.asarray(correlations, dtype=np.float32)
#         results.update({
#             'avg_ranking_correlation': float(np.mean(cor)),
#             'std_ranking_correlation': float(np.std(cor)),
#             'median_ranking_correlation': float(np.median(cor)),
#             'min_correlation': float(np.min(cor)),
#             'max_correlation': float(np.max(cor))
#         })
    
#     if precisions:
#         pr = np.asarray(precisions, dtype=np.float32)
#         results.update({
#             'avg_top_k_precision': float(np.mean(pr)),
#             'std_top_k_precision': float(np.std(pr))
#         })
    
#     if accuracies:
#         ac = np.asarray(accuracies, dtype=np.float32)
#         results.update({
#             'avg_priority_accuracy': float(np.mean(ac)),
#             'std_priority_accuracy': float(np.std(ac))
#         })
    
#     # Log summary
#     logging.info(f"ðŸ“ˆ {split_name.upper()} evaluation: {successful_evaluations}/{total_attempts} successful")
#     if correlations:
#         logging.info(f"   Average correlation: {results['avg_ranking_correlation']:.3f} Â± {results['std_ranking_correlation']:.3f}")
#         logging.info(f"   Range: [{results['min_correlation']:.3f}, {results['max_correlation']:.3f}]")
#     else:
#         logging.warning("   âŒ No successful correlations computed")
    
#     return results

import numpy as np
import torch

from scipy.stats import spearmanr, kendalltau
import os, time

def evaluate_priority_net(model: PriorityNet, dataset: PriorityNetDataset,
                          split_name: str,
                          debug_plots: bool = False,
                          out_dir: str = "outputs/priority_net") -> Dict[str, Any]:
    """
    Enhanced evaluation for PriorityNet with:
      - Deterministic seeds
      - Per-scenario z-score normalization
      - Spearman (m>3) / Kendall (m<=3)
      - Vectorized Local Rank Drift (LRD)
      - Robust degenerate-case skips
      - Rich summary (separate Spearman/Kendall, success/failure rate)
      - Optional .npy dumps and timing
    """

    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs(out_dir, exist_ok=True)

    logging.info(f"ðŸ“Š Evaluating PriorityNet on {split_name} set...")

    start_time = time.time()

    model.eval()
    device = next(model.parameters()).device

    corrs = []           # per-scenario correlation (Spearman if m>3, else Kendall)
    corr_types = []      # "spearman" or "kendall" per scenario
    kendalls_all = []    # store Kendall for all scenarios (for reporting)
    spearmans_all = []   # store Spearman for all scenarios (for reporting)
    lrd_list = []
    precisions = []
    lens = []            # store m per scenario for clearer reporting
    successful_evaluations = 0
    total_multi_det = 0

    with torch.no_grad():
        for item_idx, item in enumerate(tqdm(dataset, desc=f"Evaluating {split_name}")):
            try:
                detections = item['detections']
                true_priorities = item['priorities']

                # Skip single detection (can't compute correlation)
                if len(detections) <= 1:
                    continue

                total_multi_det += 1

                # Tensor conversion via model utility
                try:
                    detection_tensor = model._detections_to_tensor(detections)
                    if detection_tensor is None or detection_tensor.numel() == 0:
                        if item_idx < 5:
                            logging.debug(f"Empty tensor for scenario {item_idx}")
                        continue
                    detection_tensor = detection_tensor.to(device)
                except Exception as e:
                    if item_idx < 5:
                        logging.error(f"Tensor conversion failed at {item_idx}: {e}")
                    continue

                # Forward pass (priority head only)
                try:
                    features = model.signal_encoder(detection_tensor)
                    trunk_feats = model.trunk(features)
                    pred_priorities = model.priority_head(trunk_feats).squeeze(-1)
                    pred_priorities = pred_priorities.detach().cpu().numpy()
                except Exception as e:
                    if item_idx < 5:
                        logging.error(f"Forward pass failed at {item_idx}: {e}")
                    continue

                # True priorities to numpy
                try:
                    if isinstance(true_priorities, torch.Tensor):
                        true_priorities = true_priorities.detach().cpu().numpy()
                    else:
                        true_priorities = np.asarray(true_priorities, dtype=np.float64)
                except Exception as e:
                    if item_idx < 5:
                        logging.error(f"True priorities conversion failed at {item_idx}: {e}")
                    continue

                # Align lengths and validate arrays
                try:
                    m = min(len(pred_priorities), len(true_priorities))
                    if m <= 1:
                        continue

                    pred_priorities = pred_priorities[:m].astype(np.float64, copy=False)
                    true_priorities = true_priorities[:m].astype(np.float64, copy=False)

                    if not (np.isfinite(pred_priorities).all() and np.isfinite(true_priorities).all()):
                        continue

                    # Edge-case: all-equal priorities in either vector â‡’ skip (undefined rank)
                    if np.allclose(true_priorities, true_priorities[0]) or np.allclose(pred_priorities, pred_priorities[0]):
                        if item_idx < 5:
                            logging.debug(f"Degenerate equal-priority scenario {item_idx}, skipping")
                        continue

                    # Avoid zero-variance degenerate cases
                    if np.std(pred_priorities) < 1e-8 or np.std(true_priorities) < 1e-8:
                        if item_idx < 5:
                            logging.debug(f"No variation in scenario {item_idx}")
                        continue
                except Exception as e:
                    if item_idx < 5:
                        logging.error(f"Array processing failed at {item_idx}: {e}")
                    continue

                # Per-scenario z-score normalization for robust correlation
                t_mean, t_std = true_priorities.mean(), true_priorities.std()
                p_mean, p_std = pred_priorities.mean(), pred_priorities.std()
                t_z = (true_priorities - t_mean) / (t_std + 1e-8)
                p_z = (pred_priorities - p_mean) / (p_std + 1e-8)

                # Correlations: compute both; select reporting by length
                try:
                    sp_s, _ = spearmanr(t_z, p_z)
                    kd_t, _ = kendalltau(t_z, p_z)
                    # Store both
                    spearmans_all.append(float(sp_s if np.isfinite(sp_s) else 0.0))
                    kendalls_all.append(float(kd_t if np.isfinite(kd_t) else 0.0))

                    if m <= 3:
                        corr_val = float(kd_t if np.isfinite(kd_t) else 0.0)
                        corr_types.append("kendall")
                    else:
                        corr_val = float(sp_s if np.isfinite(sp_s) else 0.0)
                        corr_types.append("spearman")

                    # Fallback to rank corr if needed
                    if not np.isfinite(corr_val):
                        true_ranks = np.argsort(np.argsort(t_z))
                        pred_ranks = np.argsort(np.argsort(p_z))
                        corr_val = float(np.corrcoef(true_ranks, pred_ranks)[0, 1])
                        if not np.isfinite(corr_val):
                            corr_val = 0.0

                    corrs.append(corr_val)
                    lens.append(m)
                except Exception as e:
                    if item_idx < 5:
                        logging.error(f"Correlation failed at {item_idx}: {e}")
                    continue

                # Vectorized Local Rank Drift (pairwise inversion fraction)
                try:
                    diff_true = np.subtract.outer(t_z, t_z)
                    diff_pred = np.subtract.outer(p_z, p_z)
                    mask = (diff_true * diff_pred) < 0.0
                    mask = np.triu(mask, k=1)  # use upper triangle only
                    lrd = float(np.mean(mask)) if mask.size > 0 else 0.0
                    lrd_list.append(lrd)
                except Exception as e:
                    if item_idx < 5:
                        logging.error(f"LRD failed at {item_idx}: {e}")

                # Precision@k (k = min(3, m))
                k = min(3, m)
                true_top_k = set(np.argsort(true_priorities)[::-1][:k])
                pred_top_k = set(np.argsort(pred_priorities)[::-1][:k])
                precision = len(true_top_k & pred_top_k) / k
                precisions.append(float(precision))

                successful_evaluations += 1

                # Optional debug plots for first few scenarios
                if debug_plots and item_idx < 3:
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(4, 3))
                        plt.scatter(true_priorities, pred_priorities, s=18, alpha=0.8)
                        plt.xlabel("True Priority")
                        plt.ylabel("Predicted Priority")
                        plt.title(f"{split_name} #{item_idx} | Spearman={sp_s:.2f} | Tau={kd_t:.2f}")
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, f"eval_debug_{split_name}_{item_idx}.png"))
                        plt.close()
                    except Exception as e:
                        if item_idx < 5:
                            logging.debug(f"Plot failed at {item_idx}: {e}")

                if successful_evaluations <= 5:
                    logging.debug(f" Scenario {item_idx}: m={m}, corr={corr_val:.3f}, prec={precision:.3f}")
                    logging.debug(f"   True:  {true_priorities[:min(3,m)]}")
                    logging.debug(f"   Pred:  {pred_priorities[:min(3,m)]}")

            except Exception as e:
                if item_idx < 5:
                    logging.error(f"General error at {item_idx}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                continue

    # Compile results
    success_rate = successful_evaluations / max(total_multi_det, 1)
    results = {
        'split': split_name,
        'n_samples': successful_evaluations,
        'total_multi_det': total_multi_det,
        'total_scenarios': len(dataset),
        'success_rate': success_rate,
        'failure_rate': 1.0 - success_rate,  # explicit
        'eval_time_sec': float(time.time() - start_time),
    }

    # Aggregate metrics
    if corrs:
        corrs_arr = np.asarray(corrs, dtype=np.float64)
        lrd_arr = np.asarray(lrd_list, dtype=np.float64) if len(lrd_list) else np.array([])
        prec_arr = np.asarray(precisions, dtype=np.float64)

        # Separate Spearman vs Kendall reporting
        lens_arr = np.asarray(lens, dtype=np.int32)
        spears_arr = np.asarray(spearmans_all, dtype=np.float64) if len(spearmans_all) else np.array([])
        kend_arr = np.asarray(kendalls_all, dtype=np.float64) if len(kendalls_all) else np.array([])

        results.update({
            'avg_corr_selected': float(np.mean(corrs_arr)),
            'std_corr_selected': float(np.std(corrs_arr)),
            'median_corr_selected': float(np.median(corrs_arr)),
            'min_corr_selected': float(np.min(corrs_arr)),
            'max_corr_selected': float(np.max(corrs_arr)),
            'fraction_positive_corr': float(np.mean(corrs_arr > 0.0)),
            'median_minus_mean_corr': float(np.median(corrs_arr) - np.mean(corrs_arr)),
            'avg_spearman': float(np.nanmean(spears_arr[lens_arr > 3])) if spears_arr.size else 0.0,
            'avg_kendall': float(np.nanmean(kend_arr[lens_arr <= 3])) if kend_arr.size else 0.0,
            'avg_local_rank_drift': float(np.nanmean(lrd_arr)) if lrd_arr.size else 0.0,
            'avg_top_k_precision': float(np.mean(prec_arr)),
            'std_top_k_precision': float(np.std(prec_arr))
        })

        # Optional reproducibility dumps
        try:
            np.save(os.path.join(out_dir, f"{split_name}_corrs.npy"), corrs_arr)
            np.save(os.path.join(out_dir, f"{split_name}_precisions.npy"), prec_arr)
            if lrd_arr.size:
                np.save(os.path.join(out_dir, f"{split_name}_lrd.npy"), lrd_arr)
        except Exception as e:
            logging.debug(f"Metric dump failed: {e}")

    # Log summary
    logging.info(f"ðŸ“ˆ {split_name.upper()} evaluation: {successful_evaluations}/{total_multi_det} multi-detection scenarios")
    logging.info(f"   Total scenarios: {len(dataset)} | Success: {results['success_rate']:.3f} | Failure: {results['failure_rate']:.3f}")
    if corrs:
        logging.info(f"   Corr (selected): {results['avg_corr_selected']:.3f} Â± {results['std_corr_selected']:.3f} | median Î” mean: {results['median_minus_mean_corr']:.3f}")
        logging.info(f"   Spearman(avg m>3): {results['avg_spearman']:.3f} | Kendall(avg m<=3): {results['avg_kendall']:.3f} | LRD: {results['avg_local_rank_drift']:.3f}")
        logging.info(f"   Precision@3: {results['avg_top_k_precision']:.3f} | ðŸ•’ {results['eval_time_sec']:.2f}s")
    else:
        logging.warning("   âŒ No successful correlations computed")
        logging.warning(f"   Multi-detection scenarios found: {total_multi_det}")
        logging.warning("   Check: Are priorities being predicted correctly?")

    return results


def main():
    parser = argparse.ArgumentParser(description='Complete Phase 2: PriorityNet with Integrated Dataset Loading')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--dataset_path', required=True, help='Path to newDataset directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples per split (None = all)')
    parser.add_argument('--create_overlaps', action='store_true', help='Create artificial overlap scenarios')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training scenarios ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation scenarios ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test scenarios ratio')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logging.info("ðŸš€ Starting Complete Phase 2: PriorityNet with Integrated Dataset Loading")
    
    # Validate arguments
    if not Path(args.dataset_path).exists():
        logging.error(f"âŒ Dataset path not found: {args.dataset_path}")
        return
    
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        logging.error("âŒ Split ratios must sum to 1.0")
        return
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        #  FIXED: Extract priority_net config with proper defaults
        priority_config = config_dict.get('priority_net', {})
        
        # Set defaults
        defaults = {
            'hidden_dims': [256, 128, 64, 32],
            'dropout': 0.1,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'batch_size': 32,
            'epochs': 200,
            'patience': 30,
            'scheduler_patience': 15,
            'min_lr': 1e-6
        }
        
        # Merge with config, ensuring proper types
        final_config = {}
        for key, default_value in defaults.items():
            config_value = priority_config.get(key, default_value)
            
            #  Type conversion based on default type
            if isinstance(default_value, (int, float)):
                try:
                    final_config[key] = type(default_value)(float(config_value))
                except (ValueError, TypeError):
                    final_config[key] = default_value
                    logging.warning(f"Using default for {key}: {default_value}")
            elif isinstance(default_value, list):
                if isinstance(config_value, (list, tuple)):
                    final_config[key] = list(config_value)
                else:
                    final_config[key] = default_value
            else:
                final_config[key] = config_value
        
        #  Log loaded config for verification
        logging.info("ðŸ“‹ PriorityNet Configuration:")
        logging.info(f"   hidden_dims: {final_config['hidden_dims']}")
        logging.info(f"   dropout: {final_config['dropout']}")
        logging.info(f"   learning_rate: {final_config['learning_rate']:.2e}")
        logging.info(f"   weight_decay: {final_config['weight_decay']:.2e}")
        logging.info(f"   batch_size: {final_config['batch_size']}")
        logging.info(f"   epochs: {final_config['epochs']}")
        logging.info(f"   patience: {final_config['patience']}")
        
        # Create config object
        config = type('Config', (), final_config)()
        
    except Exception as e:
        logging.error(f"Could not load config: {e}, using defaults")
        # Fallback defaults
        config = type('Config', (), {
            'hidden_dims': [256, 128, 64, 32],
            'dropout': 0.1,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'batch_size': 32,
            'epochs': 200,
            'patience': 30
        })()

    
    # Load GW datasets
    logging.info("ðŸ“Š Loading GW datasets...")
    
    train_loader = ChunkedGWDataLoader(args.dataset_path, split='train', max_samples=args.max_samples)
    val_loader = ChunkedGWDataLoader(args.dataset_path, split='validation', max_samples=args.max_samples)
    test_loader = ChunkedGWDataLoader(args.dataset_path, split='test', max_samples=args.max_samples)
    
    # Convert to PriorityNet scenarios
    logging.info("ðŸ”„ Converting to PriorityNet scenarios...")
    
    train_scenarios = train_loader.convert_to_priority_scenarios(
        create_overlaps=args.create_overlaps, overlap_probability=0.3
    )
    val_scenarios = val_loader.convert_to_priority_scenarios(
        create_overlaps=False  # Don't create artificial overlaps for validation
    )
    test_scenarios = test_loader.convert_to_priority_scenarios(
        create_overlaps=False  # Don't create artificial overlaps for test
    )
    
    # Create PriorityNet datasets
    train_dataset = PriorityNetDataset(train_scenarios, "train")
    val_dataset = PriorityNetDataset(val_scenarios, "validation")
    test_dataset = PriorityNetDataset(test_scenarios, "test")
    
    if len(train_dataset) == 0:
        logging.error("âŒ No valid training scenarios for PriorityNet")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    logging.info("\n" + "="*80)
    logging.info("ðŸ” MULTI-DETECTION DIAGNOSTIC")
    logging.info("="*80)

    # Check actual detection counts
    multi_det_count = 0
    single_det_count = 0
    detection_counts = {}

    for item in train_dataset:
        n_det = len(item.get('detections', []))
        detection_counts[n_det] = detection_counts.get(n_det, 0) + 1
        
        if n_det >= 2:
            multi_det_count += 1
        elif n_det == 1:
            single_det_count += 1

    logging.info(f"\nðŸ“Š Detection Count Distribution:")
    for n_det in sorted(detection_counts.keys()):
        count = detection_counts[n_det]
        pct = count / len(train_dataset) * 100
        logging.info(f"   {n_det} detection(s): {count} ({pct:.1f}%)")

    logging.info(f"\nðŸ“‹ Summary:")
    logging.info(f"   Multi-detection (2+): {multi_det_count} ({multi_det_count/len(train_dataset)*100:.1f}%)")
    logging.info(f"   Single-detection (1): {single_det_count} ({single_det_count/len(train_dataset)*100:.1f}%)")

    if multi_det_count < len(train_dataset) * 0.1:
        logging.error(f"\nðŸš¨ CRITICAL: Only {multi_det_count/len(train_dataset)*100:.1f}% multi-detection!")
        logging.error("   Ranking loss will be ZERO most of the time.")
        logging.error("   STOP and fix before training!")
    elif multi_det_count < len(train_dataset) * 0.3:
        logging.warning(f"\nâš ï¸ WARNING: Only {multi_det_count/len(train_dataset)*100:.1f}% multi-detection.")
        logging.warning("   Ranking supervision will be limited but training can proceed.")
    else:
        logging.info(f"\n Good! {multi_det_count/len(train_dataset)*100:.1f}% multi-detection scenarios.")

    logging.info("="*80 + "\n")

    
    # Train model
    training_metrics = train_priority_net_with_validation(config, train_dataset, val_dataset, output_dir)
    
    # Load best model for evaluation
    best_checkpoint = torch.load(output_dir / 'priority_net_best.pth', weights_only=False, map_location='cpu')
    model = PriorityNet(config)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Comprehensive evaluation
    train_results = evaluate_priority_net(model, train_dataset, "train")
    val_results = evaluate_priority_net(model, val_dataset, "validation")
    test_results = evaluate_priority_net(model, test_dataset, "test")
    
    # Combine all results
    final_results = {
        'training_metrics': training_metrics,
        'evaluation_results': {
            'train': train_results,
            'validation': val_results,
            'test': test_results
        },
        'model_config': config.__dict__ if hasattr(config, '__dict__') else {},
        'dataset_info': {
            'train_scenarios': len(train_scenarios),
            'val_scenarios': len(val_scenarios),
            'test_scenarios': len(test_scenarios),
            'train_samples': len(train_loader.samples),
            'val_samples': len(val_loader.samples),
            'test_samples': len(test_loader.samples)
        }
    }
    
    # Save results
    with open(output_dir / 'complete_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    # Print final summary
    print("\n" + "="*80)
    print(" COMPLETE PHASE 2 - PRIORITYNET WITH INTEGRATED DATASET LOADING")
    print("="*80)
    
    test_corr = test_results.get('avg_ranking_correlation', 0)
    print(f"ðŸŽ¯ TEST SET Ranking Correlation: {test_corr:.1%}")
    
    val_corr = val_results.get('avg_ranking_correlation', 0)
    print(f"ðŸ“Š VALIDATION SET Ranking Correlation: {val_corr:.1%}")
    
    if test_corr > 0.8:
        print("ðŸ† OUTSTANDING performance achieved!")
    elif test_corr > 0.7:
        print("ðŸŽ‰ EXCELLENT performance achieved!")
    elif test_corr > 0.6:
        print(" GOOD performance achieved!")
    else:
        print("ðŸŸ¡ LEARNING - continue improvements")
    
    print(f"\nðŸ“Š DATASET STATISTICS:")
    print(f"   Training: {len(train_loader.samples):,} samples â†’ {len(train_scenarios):,} scenarios")
    print(f"   Validation: {len(val_loader.samples):,} samples â†’ {len(val_scenarios):,} scenarios")
    print(f"   Test: {len(test_loader.samples):,} samples â†’ {len(test_scenarios):,} scenarios")
    
    print(f"\nðŸ“ Results saved to: {output_dir}")
    print(f"ðŸ“Š See training_curves.png for training visualization")
    print("="*80)

if __name__ == '__main__':
    main()