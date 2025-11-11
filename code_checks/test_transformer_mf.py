#!/usr/bin/env python3
"""
Test script for Transformer encoder and matched-filter overlap metric integration.
"""

import torch
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

from ahsd.core.priority_net import PriorityNet, CrossSignalAnalyzer

def test_matched_filter_overlap():
    """Test the new matched-filter overlap metric in CrossSignalAnalyzer."""
    print("\n" + "="*70)
    print("TEST 1: Matched-Filter Overlap Metric")
    print("="*70)
    
    analyzer = CrossSignalAnalyzer()
    
    # Create sample batch: 3 signals with 15 normalized parameters
    batch_size = 3
    params_batch = torch.randn(batch_size, 15)
    
    # Ensure parameters are in valid ranges [0, 1]
    params_batch = torch.clamp(params_batch, 0, 1)
    
    print(f"Input shape: {params_batch.shape}")
    print(f"Param ranges: min={params_batch.min():.3f}, max={params_batch.max():.3f}")
    
    # Forward pass
    overlap_features = analyzer(params_batch)
    
    print(f"Output shape: {overlap_features.shape}")
    print(f"Output ranges: min={overlap_features.min():.3f}, max={overlap_features.max():.3f}")
    
    assert overlap_features.shape == (batch_size, 16), f"Expected shape (3, 16), got {overlap_features.shape}"
    assert torch.isfinite(overlap_features).all(), "Non-finite values in output!"
    
    print("✅ Matched-filter overlap metric test PASSED")
    return True

def test_transformer_encoder():
    """Test the new Transformer encoder option."""
    print("\n" + "="*70)
    print("TEST 2: Transformer Encoder Option")
    print("="*70)
    
    # Test with Transformer encoder
    try:
        print("\nInitializing PriorityNet with Transformer encoder...")
        config = type('Config', (), {
            'use_strain': True,
            'use_transformer_encoder': True,
            'use_edge_conditioning': True,
            'n_edge_types': 17,
            'hidden_dims': [512, 384, 256, 128],
            'dropout': 0.15,
            'learning_rate': 5e-4,
        })()
        
        model = PriorityNet(config=config, use_transformer_encoder=True)
        model.eval()
        
        print("✅ Transformer encoder initialization PASSED")
        
        # Test forward pass
        print("\nTesting forward pass with Transformer encoder...")
        batch_size = 2
        n_signals_per_scenario = [2, 3]
        
        for scenario_idx, n_signals in enumerate(n_signals_per_scenario):
            # Create sample detections
            detections = []
            for sig_idx in range(n_signals):
                detection = {
                    'mass_1': 30.0 + sig_idx * 5,
                    'mass_2': 25.0 + sig_idx * 3,
                    'luminosity_distance': 500.0,
                    'ra': 1.5 + sig_idx * 0.1,
                    'dec': 0.5 + sig_idx * 0.1,
                    'geocent_time': sig_idx * 0.01,
                    'theta_jn': 1.0,
                    'psi': 0.5,
                    'phase': 1.2,
                    'a_1': 0.1,
                    'a_2': 0.2,
                    'tilt_1': 0.5,
                    'tilt_2': 0.5,
                    'phi_12': 1.0,
                    'phi_jl': 1.0,
                    'network_snr': 20.0
                }
                detections.append(detection)
            
            # Create strain tensor [n_signals, n_detectors, time_samples]
            # With Transformer expecting 3 detectors (H1, L1, V1)
            strain_segments = torch.randn(n_signals, 3, 2048)
            
            with torch.no_grad():
                priorities, uncertainties = model(
                    detections,
                    strain_segments=strain_segments
                )
            
            print(f"  Scenario {scenario_idx+1} ({n_signals} signals):")
            print(f"    Priorities shape: {priorities.shape}, range: [{priorities.min():.3f}, {priorities.max():.3f}]")
            print(f"    Uncertainties shape: {uncertainties.shape}, range: [{uncertainties.min():.3f}, {uncertainties.max():.3f}]")
            
            assert priorities.shape == (n_signals,), f"Expected shape ({n_signals},), got {priorities.shape}"
            assert uncertainties.shape == (n_signals,), f"Expected shape ({n_signals},), got {uncertainties.shape}"
            assert torch.isfinite(priorities).all(), "Non-finite priorities!"
            assert torch.isfinite(uncertainties).all(), "Non-finite uncertainties!"
        
        print("✅ Transformer encoder forward pass test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Transformer encoder test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cnn_lstm_backward_compatibility():
    """Test that CNN+LSTM encoder still works (backward compatibility)."""
    print("\n" + "="*70)
    print("TEST 3: CNN+LSTM Backward Compatibility")
    print("="*70)
    
    print("\nInitializing PriorityNet with CNN+LSTM encoder (default)...")
    config = type('Config', (), {
        'use_strain': True,
        'use_transformer_encoder': False,  # Use CNN+LSTM
        'use_edge_conditioning': True,
        'n_edge_types': 17,
        'hidden_dims': [512, 384, 256, 128],
        'dropout': 0.15,
        'learning_rate': 5e-4,
    })()
    
    model = PriorityNet(config=config, use_transformer_encoder=False)
    model.eval()
    
    print("✅ CNN+LSTM encoder initialization PASSED")
    
    # Quick forward pass test
    detections = [
        {
            'mass_1': 30.0, 'mass_2': 25.0,
            'luminosity_distance': 500.0, 'ra': 1.5, 'dec': 0.5,
            'geocent_time': 0.0, 'theta_jn': 1.0, 'psi': 0.5,
            'phase': 1.2, 'a_1': 0.1, 'a_2': 0.2,
            'tilt_1': 0.5, 'tilt_2': 0.5, 'phi_12': 1.0, 'phi_jl': 1.0,
            'network_snr': 20.0
        },
        {
            'mass_1': 35.0, 'mass_2': 28.0,
            'luminosity_distance': 600.0, 'ra': 1.6, 'dec': 0.6,
            'geocent_time': 0.01, 'theta_jn': 1.2, 'psi': 0.6,
            'phase': 1.3, 'a_1': 0.15, 'a_2': 0.25,
            'tilt_1': 0.6, 'tilt_2': 0.6, 'phi_12': 1.1, 'phi_jl': 1.1,
            'network_snr': 25.0
        }
    ]
    
    strain_segments = torch.randn(2, 3, 2048)
    
    with torch.no_grad():
        priorities, uncertainties = model(detections, strain_segments=strain_segments)
    
    print(f"Priorities: {priorities.shape}, range: [{priorities.min():.3f}, {priorities.max():.3f}]")
    print(f"Uncertainties: {uncertainties.shape}, range: [{uncertainties.min():.3f}, {uncertainties.max():.3f}]")
    
    assert torch.isfinite(priorities).all() and torch.isfinite(uncertainties).all()
    print("✅ CNN+LSTM backward compatibility test PASSED")
    return True

if __name__ == '__main__':
    print("\n" + "="*70)
    print("TRANSFORMER ENCODER + MATCHED-FILTER OVERLAP METRIC TESTS")
    print("="*70)
    
    results = []
    
    try:
        results.append(test_matched_filter_overlap())
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        results.append(False)
    
    try:
        results.append(test_transformer_encoder())
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        results.append(False)
    
    try:
        results.append(test_cnn_lstm_backward_compatibility())
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        results.append(False)
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
