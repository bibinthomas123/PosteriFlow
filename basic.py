#!/usr/bin/env python3
"""
Simple Neural PE Test - Following PriorityNet loading style
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
import pickle
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SIMPLE NEURAL PE MODEL (Mimics your architecture)
# ============================================================================

class SimpleNeuralPE(nn.Module):
    """Simplified Neural PE for testing"""
    
    def __init__(self, n_params=9, context_dim=256):
        super().__init__()
        self.n_params = n_params
        self.context_dim = context_dim
        
        # Context encoder (strain -> context vector)
        self.context_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=16, stride=4),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, context_dim),
            nn.ReLU()
        )
        
    def forward(self, strain):
        """Forward pass through context encoder"""
        # strain: [batch, time_samples]
        if strain.dim() == 2:
            strain = strain.unsqueeze(1)  # [batch, 1, time]
        
        context = self.context_encoder(strain)
        return context


# ============================================================================
# DATASET
# ============================================================================

class TestDataset(torch.utils.data.Dataset):
    """Load test data from chunked pickle files"""
    
    def __init__(self, data_path, split='test', max_samples=None):
        self.data_path = Path(data_path)
        self.split = split
        self.samples = self._load_samples(max_samples)
        
        logger.info(f"✅ Loaded {len(self.samples)} {split} samples")
    
    def _load_samples(self, max_samples):
        samples = []
        
        # Try split subdirectory first
        split_dir = self.data_path / self.split
        if not split_dir.exists():
            split_dir = self.data_path
            logger.info(f"Using data directory: {split_dir}")
        
        chunk_files = sorted(split_dir.glob('chunk_*.pkl'))
        
        if not chunk_files:
            logger.warning(f"No chunk_*.pkl files found in {split_dir}")
            return []
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    samples.extend(chunk_data)
                    
                    if max_samples and len(samples) >= max_samples:
                        return samples[:max_samples]
            except Exception as e:
                logger.warning(f"Failed to load {chunk_file}: {e}")
                continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract strain from H1 or first available detector
        detector_data = sample.get('whitened_data', sample.get('detector_data', {}))
        
        if 'H1' in detector_data:
            strain = np.array(detector_data['H1'])
        elif detector_data:
            first_det = list(detector_data.keys())[0]
            strain = np.array(detector_data[first_det])
        else:
            strain = np.zeros(16384)
        
        # Ensure correct length (pad or truncate)
        if len(strain) < 16384:
            strain = np.pad(strain, (0, 16384 - len(strain)))
        elif len(strain) > 16384:
            strain = strain[:16384]
        
        # Extract metadata
        metadata = sample.get('metadata', {})
        event_type = metadata.get('event_type', 'unknown')
        network_snr = metadata.get('network_snr', 0.0)
        
        return torch.FloatTensor(strain), event_type, network_snr


# ============================================================================
# TEST FUNCTION (Like test_priority_net)
# ============================================================================

def test_neural_pe(model_path, test_data_dir, output_dir, max_samples=100):
    """Test trained Neural PE on test set - same style as PriorityNet test."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model (SAME STYLE AS PRIORITYNET)
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model configuration from checkpoint
    n_params = len(checkpoint.get('param_names', [])) if 'param_names' in checkpoint else 9
    context_dim = checkpoint.get('config', {}).get('context_dim', 256)
    
    # Initialize model
    model = SimpleNeuralPE(n_params=n_params, context_dim=context_dim)
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load state dict strictly: {e}")
        logger.info("⚠️  Loading with strict=False")
        
        # Load what we can
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"   Loaded {len(pretrained_dict)}/{len(model_dict)} layers")
    
    model.to(device).eval()
    
    # Load test data
    logger.info(f"Loading test data from {test_data_dir}")
    test_dataset = TestDataset(test_data_dir, split='test', max_samples=max_samples)
    
    if len(test_dataset) == 0:
        logger.error("No test data loaded!")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Run inference
    logger.info("Running inference on test set...")
    results = {
        'successful': 0,
        'total': 0,
        'by_type': {}
    }
    
    with torch.no_grad():
        for strain, event_types, snrs in tqdm(test_loader, desc="Testing"):
            strain = strain.to(device)
            batch_size = strain.shape[0]
            results['total'] += batch_size
            
            try:
                # Forward pass
                context = model(strain)
                
                # Check for valid output
                if not torch.isnan(context).any() and not torch.isinf(context).any():
                    results['successful'] += batch_size
                    
                    # Track by event type
                    for et in event_types:
                        if et not in results['by_type']:
                            results['by_type'][et] = {'success': 0, 'total': 0}
                        results['by_type'][et]['success'] += 1
                        results['by_type'][et]['total'] += 1
                else:
                    for et in event_types:
                        if et not in results['by_type']:
                            results['by_type'][et] = {'success': 0, 'total': 0}
                        results['by_type'][et]['total'] += 1
                        
            except Exception as e:
                logger.warning(f"Batch inference failed: {e}")
                for et in event_types:
                    if et not in results['by_type']:
                        results['by_type'][et] = {'success': 0, 'total': 0}
                    results['by_type'][et]['total'] += 1
    
    # Print results
    success_rate = results['successful'] / results['total'] if results['total'] > 0 else 0
    
    logger.info("\n" + "="*70)
    logger.info("TEST RESULTS")
    logger.info("="*70)
    logger.info(f"Total samples:     {results['total']}")
    logger.info(f"Successful:        {results['successful']}")
    logger.info(f"Success rate:      {success_rate*100:.1f}%")
    
    logger.info("\nBy Event Type:")
    for event_type, stats in results['by_type'].items():
        type_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        logger.info(f"  {event_type:10s}: {stats['success']:3d}/{stats['total']:3d} ({type_rate*100:.1f}%)")
    
    if success_rate > 0.95:
        logger.info("\n✅ EXCELLENT - Model working correctly")
    elif success_rate > 0.80:
        logger.info("\n✅ GOOD - Model mostly functional")
    elif success_rate > 0.50:
        logger.info("\n⚠️  ACCEPTABLE - Some issues detected")
    else:
        logger.warning("\n❌ POOR - Significant problems")
    
    # Save results
    import json
    results_file = output_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'total': results['total'],
            'successful': results['successful'],
            'success_rate': success_rate,
            'by_type': results['by_type']
        }, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {results_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Neural PE model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='test_outputs',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum samples to test')
    
    args = parser.parse_args()
    
    test_neural_pe(
        model_path=args.model_path,
        test_data_dir=args.data_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
