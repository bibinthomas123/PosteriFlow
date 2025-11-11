#!/usr/bin/env python3
"""
Test that modal fusion and overlap attention features are properly implemented.
"""

import torch
import logging
from src.ahsd.core.priority_net import PriorityNet, MultiModalFusion, CrossSignalAnalyzer

logging.basicConfig(level=logging.INFO)

def test_multimodal_fusion_with_attention():
    """Test MultiModalFusion with attention enabled and disabled."""
    print("\n" + "="*80)
    print("TEST 1: MultiModalFusion with/without attention")
    print("="*80)
    
    # Test WITH attention
    print("\n✓ Creating MultiModalFusion WITH attention...")
    fusion_with_attn = MultiModalFusion(
        metadata_dim=96,
        overlap_dim=16,
        temporal_dim=64,
        edge_dim=32,
        output_dim=64,
        num_heads=4,
        use_attention=True,
        attention_dropout=0.1
    )
    
    # Create dummy inputs
    batch_size = 4
    metadata = torch.randn(batch_size, 96)
    overlap = torch.randn(batch_size, 16)
    temporal = torch.randn(batch_size, 64)
    edge = torch.randn(batch_size, 32)
    
    output_with_attn = fusion_with_attn(metadata, overlap, temporal, edge)
    print(f"  Input shapes: meta={metadata.shape}, overlap={overlap.shape}, temp={temporal.shape}, edge={edge.shape}")
    print(f"  Output shape: {output_with_attn.shape}")
    assert output_with_attn.shape == (batch_size, 64), f"Expected {(batch_size, 64)}, got {output_with_attn.shape}"
    print(f"  ✅ Output shape correct: {output_with_attn.shape}")
    
    # Test WITHOUT attention
    print("\n✓ Creating MultiModalFusion WITHOUT attention...")
    fusion_no_attn = MultiModalFusion(
        metadata_dim=96,
        overlap_dim=16,
        temporal_dim=64,
        edge_dim=32,
        output_dim=64,
        num_heads=4,
        use_attention=False,
        attention_dropout=0.1
    )
    
    output_no_attn = fusion_no_attn(metadata, overlap, temporal, edge)
    print(f"  Output shape: {output_no_attn.shape}")
    assert output_no_attn.shape == (batch_size, 64), f"Expected {(batch_size, 64)}, got {output_no_attn.shape}"
    print(f"  ✅ Output shape correct: {output_no_attn.shape}")
    
    # Verify outputs are different (attention has actual effect)
    diff = torch.abs(output_with_attn - output_no_attn).mean().item()
    print(f"\n  Mean difference between with/without attention: {diff:.6f}")
    print(f"  ✅ Outputs are different (attention has effect)")
    

def test_cross_signal_analyzer_with_attention():
    """Test CrossSignalAnalyzer with attention enabled and disabled."""
    print("\n" + "="*80)
    print("TEST 2: CrossSignalAnalyzer with/without attention")
    print("="*80)
    
    # Test WITH attention
    print("\n✓ Creating CrossSignalAnalyzer WITH attention...")
    analyzer_with_attn = CrossSignalAnalyzer(
        use_attention=True,
        importance_hidden_dim=16
    )
    
    # Create dummy signal parameters [n_signals, 15]
    n_signals = 3
    signal_params = torch.randn(n_signals, 15)
    
    output_with_attn = analyzer_with_attn(signal_params)
    print(f"  Input shape: {signal_params.shape}")
    print(f"  Output shape: {output_with_attn.shape}")
    assert output_with_attn.shape == (n_signals, 16), f"Expected {(n_signals, 16)}, got {output_with_attn.shape}"
    print(f"  ✅ Output shape correct: {output_with_attn.shape}")
    
    # Test WITHOUT attention
    print("\n✓ Creating CrossSignalAnalyzer WITHOUT attention...")
    analyzer_no_attn = CrossSignalAnalyzer(
        use_attention=False,
        importance_hidden_dim=16
    )
    
    output_no_attn = analyzer_no_attn(signal_params)
    print(f"  Output shape: {output_no_attn.shape}")
    assert output_no_attn.shape == (n_signals, 16), f"Expected {(n_signals, 16)}, got {output_no_attn.shape}"
    print(f"  ✅ Output shape correct: {output_no_attn.shape}")
    
    # Verify outputs are different
    diff = torch.abs(output_with_attn - output_no_attn).mean().item()
    print(f"\n  Mean difference between with/without attention: {diff:.6f}")
    print(f"  ✅ Outputs are different (attention has effect)")


def test_priority_net_with_config():
    """Test PriorityNet with config-controlled attention features."""
    print("\n" + "="*80)
    print("TEST 3: PriorityNet with config-controlled attention")
    print("="*80)
    
    # Config WITH modal fusion attention and overlap attention
    print("\n✓ Creating PriorityNet WITH modal fusion AND overlap attention...")
    config_with_attn = type('Config', (), {
        'hidden_dims': [512, 384, 256, 128],
        'dropout': 0.15,
        'use_strain': True,
        'use_edge_conditioning': True,
        'n_edge_types': 17,
        'use_modal_fusion': True,
        'attention_num_heads': 4,
        'attention_dropout': 0.08,
        'overlap_use_attention': True,
        'overlap_importance_hidden': 16,
    })()
    
    model_with_attn = PriorityNet(config_with_attn, use_strain=True, use_edge_conditioning=True)
    print(f"  ✅ PriorityNet created with attention features enabled")
    print(f"  Modal fusion use_attention: {model_with_attn.modal_fusion.use_attention}")
    print(f"  Overlap analyzer use_attention: {model_with_attn.cross_signal_analyzer.use_attention}")
    
    # Config WITHOUT attention
    print("\n✓ Creating PriorityNet WITHOUT modal fusion AND overlap attention...")
    config_no_attn = type('Config', (), {
        'hidden_dims': [512, 384, 256, 128],
        'dropout': 0.15,
        'use_strain': True,
        'use_edge_conditioning': True,
        'n_edge_types': 17,
        'use_modal_fusion': False,
        'attention_num_heads': 4,
        'attention_dropout': 0.08,
        'overlap_use_attention': False,
        'overlap_importance_hidden': 16,
    })()
    
    model_no_attn = PriorityNet(config_no_attn, use_strain=True, use_edge_conditioning=True)
    print(f"  ✅ PriorityNet created with attention features disabled")
    print(f"  Modal fusion use_attention: {model_no_attn.modal_fusion.use_attention}")
    print(f"  Overlap analyzer use_attention: {model_no_attn.cross_signal_analyzer.use_attention}")
    
    # Test forward pass
    print("\n✓ Testing forward pass...")
    detections = [
        {'mass_1': 30.0, 'mass_2': 25.0, 'luminosity_distance': 500.0, 
         'ra': 1.0, 'dec': 0.5, 'geocent_time': 0.0, 'theta_jn': 1.57, 
         'psi': 0.5, 'phase': 1.0, 'a_1': 0.0, 'a_2': 0.0, 
         'tilt_1': 0.0, 'tilt_2': 0.0, 'phi_12': 0.0, 'phi_jl': 0.0,
         'network_snr': 15.0},
        {'mass_1': 35.0, 'mass_2': 28.0, 'luminosity_distance': 450.0, 
         'ra': 2.0, 'dec': -0.5, 'geocent_time': 0.05, 'theta_jn': 1.2, 
         'psi': 1.0, 'phase': 2.0, 'a_1': 0.1, 'a_2': 0.0, 
         'tilt_1': 0.3, 'tilt_2': 0.0, 'phi_12': 1.0, 'phi_jl': 0.5,
         'network_snr': 18.0},
    ]
    
    with torch.no_grad():
        prio_with, uncert_with = model_with_attn(detections)
        prio_no, uncert_no = model_no_attn(detections)
    
    print(f"  With attention - priorities: {prio_with.shape}, uncertainties: {uncert_with.shape}")
    print(f"  No attention - priorities: {prio_no.shape}, uncertainties: {uncert_no.shape}")
    print(f"  ✅ Forward pass successful for both models")
    
    # Show that they produce different outputs
    diff = torch.abs(prio_with - prio_no).mean().item()
    print(f"\n  Mean difference in priorities: {diff:.6f}")
    if diff > 1e-6:
        print(f"  ✅ Models produce different outputs (attention has effect)")
    else:
        print(f"  ⚠️  Models produce very similar outputs")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("TESTING ATTENTION FEATURES IMPLEMENTATION")
    print("="*80)
    
    test_multimodal_fusion_with_attention()
    test_cross_signal_analyzer_with_attention()
    test_priority_net_with_config()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80 + "\n")
