#!/usr/bin/env python3
"""
Simple test script for Transformer encoder and matched-filter overlap metric.
Avoids circular imports by testing components directly.
"""

import torch
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

def test_transformer_encoder_direct():
    """Test TransformerStrainEncoder directly (avoiding circular imports)."""
    print("\n" + "="*70)
    print("TEST 1: TransformerStrainEncoder (Direct Import)")
    print("="*70)
    
    from ahsd.models.transformer_encoder import TransformerStrainEncoder
    
    encoder = TransformerStrainEncoder(
        use_whisper=True,
        freeze_layers=4,
        input_length=2048,
        n_detectors=3,
        output_dim=64
    )
    
    encoder.eval()
    
    # Test forward pass
    batch_size = 2
    strain_data = torch.randn(batch_size, 3, 2048)  # [batch, n_detectors, time_samples]
    
    with torch.no_grad():
        features = encoder(strain_data)
    
    print(f"Input shape: {strain_data.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output range: [{features.min():.4f}, {features.max():.4f}]")
    
    assert features.shape == (batch_size, 64), f"Expected (2, 64), got {features.shape}"
    assert torch.isfinite(features).all(), "Non-finite values in output!"
    
    print("✅ TransformerStrainEncoder test PASSED")
    return True

def test_matched_filter_overlap():
    """Test matched-filter overlap metric (direct import)."""
    print("\n" + "="*70)
    print("TEST 2: Matched-Filter Overlap Metric")
    print("="*70)
    
    # Import only the core module which doesn't have circular dependencies
    import sys
    import importlib.util
    
    # Load priority_net directly to avoid __init__.py circular import
    spec = importlib.util.spec_from_file_location(
        "priority_net_direct",
        "/home/bibinathomas/PosteriFlow/src/ahsd/core/priority_net.py"
    )
    priority_net_module = importlib.util.module_from_spec(spec)
    
    # Mock ahsd to prevent circular import
    import sys
    from unittest.mock import MagicMock
    sys.modules['ahsd'] = MagicMock()
    sys.modules['ahsd.models'] = MagicMock()
    sys.modules['ahsd.models.transformer_encoder'] = MagicMock()
    
    # Now we can load the module
    spec.loader.exec_module(priority_net_module)
    
    CrossSignalAnalyzer = priority_net_module.CrossSignalAnalyzer
    
    analyzer = CrossSignalAnalyzer()
    
    # Create sample batch
    batch_size = 3
    params_batch = torch.randn(batch_size, 15)
    params_batch = torch.clamp(params_batch, 0, 1)
    
    print(f"Input shape: {params_batch.shape}")
    
    # Forward pass
    overlap_features = analyzer(params_batch)
    
    print(f"Output shape: {overlap_features.shape}")
    print(f"Output range: [{overlap_features.min():.4f}, {overlap_features.max():.4f}]")
    
    assert overlap_features.shape == (batch_size, 16), f"Expected (3, 16), got {overlap_features.shape}"
    assert torch.isfinite(overlap_features).all(), "Non-finite values!"
    
    print("✅ Matched-filter overlap metric test PASSED")
    return True

def test_transformer_encoder_feature_comparison():
    """Compare CNN+LSTM vs Transformer output dimensions."""
    print("\n" + "="*70)
    print("TEST 3: Encoder Architecture Comparison")
    print("="*70)
    
    from ahsd.models.transformer_encoder import TransformerStrainEncoder
    import sys
    import importlib.util
    from unittest.mock import MagicMock
    
    # Load TemporalStrainEncoder
    sys.modules['ahsd'] = MagicMock()
    sys.modules['ahsd.models'] = MagicMock()
    sys.modules['ahsd.models.transformer_encoder'] = MagicMock()
    
    spec = importlib.util.spec_from_file_location(
        "priority_net_direct2",
        "/home/bibinathomas/PosteriFlow/src/ahsd/core/priority_net.py"
    )
    priority_net_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(priority_net_module)
    
    TemporalStrainEncoder = priority_net_module.TemporalStrainEncoder
    
    # Test both encoders with same input
    batch_size = 2
    strain_data = torch.randn(batch_size, 3, 2048)
    
    print("\n--- CNN+BiLSTM Encoder ---")
    cnn_encoder = TemporalStrainEncoder(input_length=2048, n_detectors=3, hidden_dim=128)
    cnn_encoder.eval()
    
    with torch.no_grad():
        cnn_features = cnn_encoder(strain_data)
    
    print(f"Output shape: {cnn_features.shape}")
    print(f"Output range: [{cnn_features.min():.4f}, {cnn_features.max():.4f}]")
    
    print("\n--- Transformer Encoder ---")
    transformer_encoder = TransformerStrainEncoder(
        use_whisper=True,
        freeze_layers=4,
        input_length=2048,
        n_detectors=3,
        output_dim=64
    )
    transformer_encoder.eval()
    
    with torch.no_grad():
        transformer_features = transformer_encoder(strain_data)
    
    print(f"Output shape: {transformer_features.shape}")
    print(f"Output range: [{transformer_features.min():.4f}, {transformer_features.max():.4f}]")
    
    # Both should output 64-D features
    assert cnn_features.shape == (batch_size, 64), "CNN output dimension mismatch!"
    assert transformer_features.shape == (batch_size, 64), "Transformer output dimension mismatch!"
    
    print("\n✅ Both architectures produce 64-D output (compatible!)")
    return True

if __name__ == '__main__':
    print("\n" + "="*70)
    print("TRANSFORMER ENCODER + MATCHED-FILTER OVERLAP TESTS (SIMPLIFIED)")
    print("="*70)
    
    results = []
    
    try:
        results.append(test_transformer_encoder_direct())
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)
    
    try:
        results.append(test_matched_filter_overlap())
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)
    
    try:
        results.append(test_transformer_encoder_feature_comparison())
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
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
        print("\n⚠️  SOME TESTS FAILED (likely due to unrelated circular imports)")
        sys.exit(1)
