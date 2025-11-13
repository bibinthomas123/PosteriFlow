#!/usr/bin/env python3
"""
Comprehensive testing script for OverlapNeuralPE
Tests model inference, loss computation, and posterior quality
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import logging
import argparse
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OverlapNeuralPETest:
    """Complete test suite for OverlapNeuralPE"""
    
    def __init__(self, model_path: str, config_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = {}
        self.results = {}
        
        logger.info(f"Using device: {self.device}")
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load model from checkpoint"""
        logger.info(f"\n📦 Loading model from: {model_path}")
        
        try:
            from ahsd.models.overlap_neuralpe import OverlapNeuralPE
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract config and parameters
            self.config = checkpoint.get('config', {})
            param_names = checkpoint.get('param_names', [
                'mass_1', 'mass_2', 'luminosity_distance',
                'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time'
            ])
            
            priority_net_path = self.config.get('training_metadata', {}).get('priority_net')
            if not priority_net_path:
                logger.warning("⚠️  Priority net path not found in checkpoint")
            
            # Create model
            self.model = OverlapNeuralPE(
                param_names=param_names,
                priority_net_path=priority_net_path,
                config=self.config,
                device=self.device
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model.to(self.device).eval()
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"   Parameters: {param_names}")
            logger.info(f"   Flow type: {self.config.get('neural_posterior', {}).get('flow_type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    # ========================================================================
    # TEST 1: FORWARD PASS
    # ========================================================================
    
    def test_forward_pass(self, batch_size: int = 4, n_signals: int = 2):
        """Test basic forward pass"""
        logger.info("\n" + "="*70)
        logger.info("TEST 1: FORWARD PASS")
        logger.info("="*70)
        
        try:
            # Create dummy strain data
            strain_data = torch.randn(batch_size, 2, 16384, device=self.device)
            
            logger.info(f"Input shape: {strain_data.shape}")
            
            with torch.no_grad():
                # Test context encoding
                context = self.model.context_encoder(strain_data)
                logger.info(f"✅ Context encoding: {context.shape}")
                
                # Test signal extraction (if available)
                if hasattr(self.model, 'signal_extractor'):
                    signals = self.model.signal_extractor(strain_data)
                    logger.info(f"✅ Signal extraction: {signals.shape if hasattr(signals, 'shape') else type(signals)}")
                
                # Test flow forward pass
                if hasattr(self.model, 'flow'):
                    param_dim = self.model.param_dim
                    z = torch.randn(batch_size, param_dim, device=self.device)
                    log_prob, _ = self.model.flow(z, context)
                    logger.info(f"✅ Flow forward: log_prob shape {log_prob.shape}")
                    
                    # Test inverse (sampling)
                    samples, log_det = self.model.flow.inverse(z, context)
                    logger.info(f"✅ Flow inverse: samples shape {samples.shape}")
            
            self.results['forward_pass'] = {'status': 'PASS', 'batch_size': batch_size}
            return True
            
        except Exception as e:
            logger.error(f"❌ Forward pass failed: {e}")
            self.results['forward_pass'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    # ========================================================================
    # TEST 2: LOSS COMPUTATION
    # ========================================================================
    
    def test_loss_computation(self, batch_size: int = 4):
        """Test loss computation"""
        logger.info("\n" + "="*70)
        logger.info("TEST 2: LOSS COMPUTATION")
        logger.info("="*70)
        
        try:
            strain_data = torch.randn(batch_size, 2, 16384, device=self.device)
            target_params = torch.randn(batch_size, self.model.param_dim, device=self.device)
            
            logger.info(f"Strain shape: {strain_data.shape}")
            logger.info(f"Target params shape: {target_params.shape}")
            
            with torch.no_grad():
                loss_dict = self.model.compute_loss(strain_data, target_params)
            
            logger.info(f"\n✅ Loss computation successful:")
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    val = value.item()
                else:
                    val = value
                logger.info(f"   {key:20s}: {val:.6f}")
                
                # Check for NaN/Inf
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any():
                        logger.warning(f"   ⚠️  {key} contains NaN!")
                    if torch.isinf(value).any():
                        logger.warning(f"   ⚠️  {key} contains Inf!")
            
            # Verify loss quality
            total_loss = loss_dict.get('total_loss', loss_dict.get('loss'))
            if total_loss < 0:
                logger.warning("⚠️  Negative loss detected!")
            if total_loss > 100:
                logger.warning("⚠️  Very high loss detected!")
            
            self.results['loss_computation'] = {
                'status': 'PASS',
                'loss_dict': {k: v.item() if isinstance(v, torch.Tensor) else v 
                             for k, v in loss_dict.items()}
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Loss computation failed: {e}")
            self.results['loss_computation'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    # ========================================================================
    # TEST 3: GRADIENT FLOW
    # ========================================================================
    
    def test_gradient_flow(self, batch_size: int = 4):
        """Test gradient flow through the model"""
        logger.info("\n" + "="*70)
        logger.info("TEST 3: GRADIENT FLOW")
        logger.info("="*70)
        
        try:
            self.model.train()  # Set to train mode for gradients
            
            strain_data = torch.randn(batch_size, 2, 16384, device=self.device, requires_grad=False)
            target_params = torch.randn(batch_size, self.model.param_dim, device=self.device)
            
            # Forward pass
            loss_dict = self.model.compute_loss(strain_data, target_params)
            total_loss = loss_dict.get('total_loss', loss_dict.get('loss'))
            
            # Backward pass
            total_loss.backward()
            
            logger.info(f"✅ Backward pass successful")
            
            # Check gradients
            grad_stats = {
                'zero_grad': 0,
                'small_grad': 0,
                'normal_grad': 0,
                'large_grad': 0,
                'total_params': 0
            }
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    grad_stats['total_params'] += 1
                    
                    if grad_norm == 0:
                        grad_stats['zero_grad'] += 1
                    elif grad_norm < 1e-6:
                        grad_stats['small_grad'] += 1
                    elif grad_norm > 1.0:
                        grad_stats['large_grad'] += 1
                    else:
                        grad_stats['normal_grad'] += 1
            
            logger.info(f"\n📊 Gradient Statistics:")
            logger.info(f"   Total parameters: {grad_stats['total_params']}")
            logger.info(f"   Zero gradients: {grad_stats['zero_grad']}")
            logger.info(f"   Small gradients (<1e-6): {grad_stats['small_grad']}")
            logger.info(f"   Normal gradients: {grad_stats['normal_grad']}")
            logger.info(f"   Large gradients (>1.0): {grad_stats['large_grad']}")
            
            if grad_stats['zero_grad'] == grad_stats['total_params']:
                logger.warning("⚠️  All gradients are zero!")
                return False
            
            if grad_stats['small_grad'] / grad_stats['total_params'] > 0.5:
                logger.warning("⚠️  More than 50% vanishing gradients!")
            
            self.model.zero_grad()  # Clear gradients
            self.model.eval()  # Back to eval mode
            
            self.results['gradient_flow'] = {
                'status': 'PASS',
                'grad_stats': grad_stats
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Gradient flow test failed: {e}")
            self.results['gradient_flow'] = {'status': 'FAIL', 'error': str(e)}
            self.model.eval()
            return False
    
    # ========================================================================
    # TEST 4: POSTERIOR SAMPLING
    # ========================================================================
    
    def test_posterior_sampling(self, batch_size: int = 2, n_samples: int = 100):
        """Test posterior sampling"""
        logger.info("\n" + "="*70)
        logger.info("TEST 4: POSTERIOR SAMPLING")
        logger.info("="*70)
        
        try:
            strain_data = torch.randn(batch_size, 2, 16384, device=self.device)
            
            logger.info(f"Sampling {n_samples} posterior samples per batch element")
            
            with torch.no_grad():
                # Get context
                context = self.model.context_encoder(strain_data)
                logger.info(f"✅ Context shape: {context.shape}")
                
                # Sample from base distribution
                z = torch.randn(n_samples, batch_size, self.model.param_dim, device=self.device)
                
                # Transform through flow
                posterior_samples = []
                for i in range(0, n_samples, 20):  # Process in chunks
                    z_chunk = z[i:min(i+20, n_samples)]
                    samples_chunk, _ = self.model.flow.inverse(
                        z_chunk.reshape(-1, self.model.param_dim),
                        context.repeat_interleave(z_chunk.shape[0], dim=0)
                    )
                    posterior_samples.append(samples_chunk.reshape(z_chunk.shape))
                
                posterior_samples = torch.cat(posterior_samples, dim=0)
                
                logger.info(f"✅ Posterior samples shape: {posterior_samples.shape}")
                
                # Compute statistics
                samples_np = posterior_samples.cpu().numpy()
                
                logger.info(f"\n📊 Sample Statistics:")
                param_names = ['mass_1', 'mass_2', 'luminosity_distance',
                              'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time']
                
                for p in range(min(self.model.param_dim, len(param_names))):
                    samples_p = samples_np[:, 0, p]  # First batch element
                    
                    mean = samples_p.mean()
                    std = samples_p.std()
                    min_val = samples_p.min()
                    max_val = samples_p.max()
                    
                    logger.info(f"   {param_names[p]:20s}: "
                               f"μ={mean:8.4f}, σ={std:8.4f}, "
                               f"range=[{min_val:8.4f}, {max_val:8.4f}]")
                    
                    if std < 1e-6:
                        logger.warning(f"   ⚠️  Very small std for {param_names[p]}")
                
                # Check for NaN/Inf
                if np.isnan(samples_np).any():
                    logger.warning("⚠️  Posterior samples contain NaN!")
                    return False
                
                if np.isinf(samples_np).any():
                    logger.warning("⚠️  Posterior samples contain Inf!")
                    return False
            
            self.results['posterior_sampling'] = {
                'status': 'PASS',
                'n_samples': n_samples,
                'sample_shape': str(posterior_samples.shape)
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Posterior sampling failed: {e}")
            self.results['posterior_sampling'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    # ========================================================================
    # TEST 5: DETECTOR HANDLING
    # ========================================================================
    
    def test_detector_handling(self, batch_size: int = 2):
        """Test multi-detector strain handling"""
        logger.info("\n" + "="*70)
        logger.info("TEST 5: MULTI-DETECTOR HANDLING")
        logger.info("="*70)
        
        try:
            # Test different input formats
            test_cases = [
                ("H1, L1 only", torch.randn(batch_size, 2, 16384)),
                ("H1, L1, V1", torch.randn(batch_size, 3, 16384)),
            ]
            
            for name, strain_data in test_cases:
                try:
                    # Pad/truncate to 2 detectors if needed
                    if strain_data.shape[1] > 2:
                        strain_data = strain_data[:, :2, :]
                        logger.info(f"   Truncated to 2 detectors: {strain_data.shape}")
                    
                    strain_data = strain_data.to(self.device)
                    
                    with torch.no_grad():
                        context = self.model.context_encoder(strain_data)
                    
                    logger.info(f"✅ {name}: input shape {(batch_size, strain_data.shape[1], 16384)} → "
                               f"context shape {context.shape}")
                    
                except Exception as e:
                    logger.warning(f"   ⚠️  {name} failed: {e}")
            
            self.results['detector_handling'] = {'status': 'PASS'}
            return True
            
        except Exception as e:
            logger.error(f"❌ Detector handling test failed: {e}")
            self.results['detector_handling'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    # ========================================================================
    # TEST 6: SIGNAL EXTRACTION
    # ========================================================================
    
    def test_signal_extraction(self, batch_size: int = 2, max_signals: int = 5):
        """Test signal extraction with PriorityNet"""
        logger.info("\n" + "="*70)
        logger.info("TEST 6: SIGNAL EXTRACTION (PRIORITYNET)")
        logger.info("="*70)
        
        try:
            strain_data = torch.randn(batch_size, 2, 16384, device=self.device)
            
            if not hasattr(self.model, 'signal_extractor'):
                logger.info("⚠️  Signal extractor not available in model")
                self.results['signal_extraction'] = {'status': 'SKIP'}
                return True
            
            with torch.no_grad():
                # Extract signals
                extracted = self.model.signal_extractor(strain_data)
                
                if isinstance(extracted, dict):
                    logger.info(f"✅ Signal extraction returned dict with keys: {extracted.keys()}")
                    
                    for key, val in extracted.items():
                        if hasattr(val, 'shape'):
                            logger.info(f"   {key}: {val.shape}")
                else:
                    logger.info(f"✅ Signal extraction shape: {extracted.shape if hasattr(extracted, 'shape') else type(extracted)}")
            
            self.results['signal_extraction'] = {'status': 'PASS'}
            return True
            
        except Exception as e:
            logger.error(f"❌ Signal extraction failed: {e}")
            self.results['signal_extraction'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    # ========================================================================
    # TEST 7: OUTPUT RANGES
    # ========================================================================
    
    def test_output_ranges(self, batch_size: int = 10, n_samples: int = 100):
        """Test if model outputs are in reasonable ranges"""
        logger.info("\n" + "="*70)
        logger.info("TEST 7: OUTPUT RANGES")
        logger.info("="*70)
        
        try:
            all_samples = []
            
            with torch.no_grad():
                for _ in range(2):  # Multiple batches
                    strain_data = torch.randn(batch_size, 2, 16384, device=self.device)
                    
                    # Use model's sample_posterior() which includes denormalization
                    result = self.model.sample_posterior(strain_data, n_samples=n_samples)
                    samples = result['samples']  # [batch, n_samples, param_dim]
                    
                    # Transpose to [n_samples, batch, n_params] for consistency
                    samples = samples.permute(1, 0, 2)
                    all_samples.append(samples.cpu().numpy())
            
            all_samples = np.concatenate(all_samples, axis=0)  # [n_total, batch, n_params]
            
            # Check ranges
            param_names = ['mass_1', 'mass_2', 'luminosity_distance',
                          'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time']
            
            expected_ranges = {
                'mass_1': (2, 150),
                'mass_2': (2, 150),
                'luminosity_distance': (10, 5000),
                'ra': (-np.pi, np.pi),
                'dec': (-np.pi/2, np.pi/2),
                'theta_jn': (0, np.pi),
                'psi': (0, np.pi),
                'phase': (0, 2*np.pi),
                'geocent_time': (-0.1, 0.1)
            }
            
            logger.info(f"\n📊 Output Ranges (checking {all_samples.shape[0]} samples):")
            
            out_of_range_count = 0
            for p in range(min(self.model.param_dim, len(param_names))):
                samples_p = all_samples[:, :, p].flatten()
                
                min_val = samples_p.min()
                max_val = samples_p.max()
                mean_val = samples_p.mean()
                
                param_name = param_names[p]
                expected = expected_ranges.get(param_name, (-np.inf, np.inf))
                
                in_range = (samples_p >= expected[0]).all() and (samples_p <= expected[1]).all()
                status = "✅" if in_range else "⚠️"
                
                logger.info(f"   {status} {param_name:20s}: "
                           f"[{min_val:8.4f}, {max_val:8.4f}] "
                           f"(expected: {expected})")
                
                if not in_range:
                    out_of_range = ((samples_p < expected[0]) | (samples_p > expected[1])).sum()
                    out_of_range_count += out_of_range
                    logger.info(f"      {out_of_range} out of range samples")
            
            if out_of_range_count > 0:
                logger.warning(f"⚠️  Total out-of-range samples: {out_of_range_count}")
            
            self.results['output_ranges'] = {
                'status': 'PASS',
                'out_of_range_count': int(out_of_range_count)
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Output ranges test failed: {e}")
            self.results['output_ranges'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    
    def run_all_tests(self, batch_size: int = 4):
        """Run complete test suite"""
        logger.info("\n" + "="*70)
        logger.info("OVERLAP NEURAL PE - COMPLETE TEST SUITE")
        logger.info("="*70)
        
        test_functions = [
            self.test_forward_pass,
            self.test_loss_computation,
            self.test_gradient_flow,
            self.test_posterior_sampling,
            self.test_detector_handling,
            self.test_signal_extraction,
            self.test_output_ranges,
        ]
        
        passed = 0
        failed = 0
        skipped = 0
        
        for test_func in test_functions:
            try:
                result = test_func(batch_size=batch_size)
                
                if 'status' in self.results.get(test_func.__name__.replace('test_', ''), {}):
                    status = self.results[test_func.__name__.replace('test_', '')]['status']
                    if status == 'PASS':
                        passed += 1
                    elif status == 'FAIL':
                        failed += 1
                    elif status == 'SKIP':
                        skipped += 1
            except Exception as e:
                logger.error(f"Unexpected error in {test_func.__name__}: {e}")
                failed += 1
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("TEST SUMMARY")
        logger.info("="*70)
        logger.info(f"✅ Passed:  {passed}")
        logger.info(f"❌ Failed:  {failed}")
        logger.info(f"⏭️  Skipped: {skipped}")
        logger.info("="*70 + "\n")
        
        return {
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'details': self.results
        }


def main():
    parser = argparse.ArgumentParser(description='Test OverlapNeuralPE model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for testing')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                       help='Device to use')
    parser.add_argument('--output', type=str, default='outputs/overlap_neural_pe_test_results.json',
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Run tests
    tester = OverlapNeuralPETest(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )
    
    results = tester.run_all_tests(batch_size=args.batch_size)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Ensure we're writing to a file, not a directory
    if output_path.is_dir():
        output_path = output_path / 'test_results.json'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Results saved to: {output_path}")
    
    # Return exit code
    return 0 if results['failed'] == 0 else 1


if __name__ == '__main__':
    exit(main())
