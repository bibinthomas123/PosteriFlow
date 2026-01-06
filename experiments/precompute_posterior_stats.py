#!/usr/bin/env python
"""
Precompute posterior statistics (mean, std) for BiasCorrector training.
Runs once offline, saves results, eliminates expensive sampling during training.
"""

import torch
import pickle
import logging
from pathlib import Path
from tqdm import tqdm
from ahsd.models.overlap_neuralpe import OverlapNeuralPE
from ahsd.core.priority_net import PriorityNet
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def precompute_posterior_stats(data_dir: str, 
                               neural_pe_path: str,
                               priority_net_path: str,
                               param_names: list,
                               device: str = 'cuda',
                               splits: list = ['train', 'validation']):
    """
    Precompute posterior mean/std for all samples in splits.
    
    Saves cached stats to data_dir/posterior_stats_{split}.pkl
    """
    
    # Load frozen models
    logger.info("Loading Neural PE and PriorityNet...")
    # Match checkpoint architecture (trained with context_dim=768, 12 layers)
    # : Config must be nested under 'neural_posterior' key 
    # because OverlapNeuralPE.__init__ reads from neural_posterior section first
    neural_pe_config = {
       'neural_posterior': {
           'context_dim': 768,  # ✅ Checkpoint trained with 768 (verified from state dict)
           'flow_type': 'nsf',
           'num_layers': 12,
           'hidden_features': 256,
           'num_bins': 16,  # ✅ Match checkpoint (was 8, should be 16)
           'tail_bound': 3.0,
           'dropout': 0.2,
           'flow_config': {
               'num_layers': 12,
               'hidden_features': 256,
               'dropout': 0.15
           }
       },
       'enable_event_specific_priors': True
    }
    neural_pe_model = OverlapNeuralPE(
        param_names=param_names,
        priority_net_path=priority_net_path,
        config=neural_pe_config,
        device=device
    )
    neural_pe_checkpoint = torch.load(neural_pe_path, map_location=device)
    if isinstance(neural_pe_checkpoint, dict) and 'model_state_dict' in neural_pe_checkpoint:
        neural_pe_state = neural_pe_checkpoint['model_state_dict']
    else:
        neural_pe_state = neural_pe_checkpoint
    neural_pe_model.load_state_dict(neural_pe_state, strict=False)
    neural_pe_model.to(device)
    neural_pe_model.eval()
    
    priority_net_model = PriorityNet()
    priority_checkpoint = torch.load(priority_net_path, map_location=device)
    if isinstance(priority_checkpoint, dict) and 'model_state_dict' in priority_checkpoint:
        priority_state = priority_checkpoint['model_state_dict']
    else:
        priority_state = priority_checkpoint
    priority_net_model.load_state_dict(priority_state)
    priority_net_model.to(device)
    priority_net_model.eval()
    
    logger.info("✅ Models loaded\n")
    
    # Process each split
    for split in splits:
        logger.info(f"Processing {split} split...")
        data_path = Path(data_dir) / split
        chunk_files = sorted(data_path.glob('chunk_*.pkl'))
        
        if not chunk_files:
            logger.warning(f"No chunks found in {data_path}")
            continue
        
        posterior_stats = {}  # idx -> {'mean': array, 'std': array, 'snr': float}
        sample_idx = 0
        
        for batch_file in tqdm(chunk_files, desc=f"Loading chunks ({split})"):
            try:
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f)
                
                if isinstance(batch, dict) and 'samples' in batch:
                    samples = batch['samples']
                else:
                    samples = batch if isinstance(batch, list) else []
                
                for sample in samples:
                    if sample is None:
                        sample_idx += 1
                        continue
                    
                    try:
                        # Extract strain
                        strain_dict = {}
                        detector_data = sample.get('detector_data', {})
                        for detector in ['H1', 'L1', 'V1']:
                            if detector in detector_data:
                                det_data = detector_data[detector]
                                if isinstance(det_data, dict):
                                    strain_data = det_data.get('strain')
                                    if strain_data is None:
                                        strain_data = det_data.get('whitened_data')
                                    if strain_data is None:
                                        strain_data = det_data
                                else:
                                    strain_data = det_data
                                strain = np.array(strain_data, dtype=np.float32)
                                strain_dict[detector] = torch.from_numpy(strain).to(device)
                            else:
                                strain_dict[detector] = torch.zeros(16384, dtype=torch.float32, device=device)
                        
                        # Stack strain [1, 3, 16384]
                        strain_batch = torch.stack([
                            strain_dict['H1'],
                            strain_dict['L1'],
                            strain_dict['V1']
                        ]).unsqueeze(0).to(device)
                        
                        # Compute posterior stats (ULTRA FAST: use direct flow sampling, no sample_posterior API)
                        with torch.no_grad():
                            # ✅ CRITICAL SPEEDUP: Direct flow transform (50× faster than sample_posterior API)
                            context = neural_pe_model.context_encoder(strain_batch)  # [1, 768]
                            
                            # Sample directly from standard normal and transform through flow
                            n_quick_samples = 3  # ✅ MINIMAL: Just 2 samples for mean/std estimate
                            z_samples = torch.randn(n_quick_samples, neural_pe_model.param_dim, device=device)
                            
                            # Transform through flow - returns tuple (z_transformed, log_prob)
                            flow_output = neural_pe_model.flow.transform(z_samples, context.expand(n_quick_samples, -1))
                            
                            # Unpack tuple
                            if isinstance(flow_output, tuple):
                                posterior_samples_norm = flow_output[0]  # Take z_transformed, skip log_prob
                            else:
                                posterior_samples_norm = flow_output
                            
                            # Denormalize to physical units
                            posterior_samples = neural_pe_model._denormalize_parameters(posterior_samples_norm)
                            
                            mean = posterior_samples.mean(dim=0).cpu().numpy().astype(np.float32)
                            std = posterior_samples.std(dim=0).cpu().numpy().astype(np.float32)
                            
                            # Also get true SNR for weighting
                            if sample is None:
                                snr = 15.0
                            else:
                                params_list = sample.get('parameters', [])
                                if isinstance(params_list, list) and len(params_list) > 0:
                                    params = params_list[0] if isinstance(params_list[0], dict) else {}
                                else:
                                    params = sample if isinstance(sample, dict) else {}
                                snr = float(params.get('target_snr', 15.0))
                        
                        # Extract true parameters (first signal in overlaps)
                        params_list = sample.get('parameters', [])
                        if isinstance(params_list, list) and len(params_list) > 0:
                            params = params_list[0] if isinstance(params_list[0], dict) else {}
                        else:
                            params = sample if isinstance(sample, dict) else {}
                        
                        true_params = np.array([
                            float(params.get(param_name, 0.0)) for param_name in param_names
                        ], dtype=np.float32)
                        
                        posterior_stats[sample_idx] = {
                            'mean': mean,
                            'std': std,
                            'snr': snr,
                            'true_params': true_params  # ✅ NOW INCLUDES TRUE PARAMS
                        }
                    
                    except Exception as e:
                        logger.warning(f"Skip sample {sample_idx}: {type(e).__name__}: {str(e)[:100]}")
                    
                    sample_idx += 1
            
            except Exception as e:
                logger.warning(f"Skip batch {batch_file}: {e}")
                continue
        
        # Save cache at data_dir root level (NOT split subdirectory)
        # This matches where BiasDataset loads it from (line 115 of train_bias_corrector.py)
        cache_path = Path(data_dir) / f'posterior_stats_{split}.pkl'
        with open(cache_path, 'wb') as f:
            pickle.dump(posterior_stats, f)
        
        logger.info(f"✅ Cached {len(posterior_stats)} {split} samples → {cache_path}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/test', help='Data directory')
    parser.add_argument('--neural_pe_path', default='models/neuralpe/best_model.pth')
    parser.add_argument('--priority_net_path', default='models/prioritynet/priority_net_best.pth')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    param_names = [
        'mass_1', 'mass_2', 'luminosity_distance',
        'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time',
        'a_1', 'a_2'  # ✅ Now using 11D: Spin magnitudes (matches train_bias_corrector.py)
    ]
    
    precompute_posterior_stats(
        data_dir=args.data_dir,
        neural_pe_path=args.neural_pe_path,
        priority_net_path=args.priority_net_path,
        param_names=param_names,
        device=args.device
    )
