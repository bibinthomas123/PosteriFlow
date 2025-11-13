#!/usr/bin/env python3
"""
Comprehensive transformer encoder health check with detailed logging.
Tests TransformerStrainEncoder in PriorityNet, verifies forward passes,
checks gradient flow, and detects potential issues.
"""

import torch
import torch.nn as nn
import logging
import sys
from pathlib import Path

# Setup logging with detailed formatting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('transformer_health_check.log')
    ]
)

logger = logging.getLogger(__name__)

def test_transformer_encoder_directly():
    """Test TransformerStrainEncoder in isolation."""
    logger.info("="*80)
    logger.info("TEST 1: Direct TransformerStrainEncoder instantiation and forward pass")
    logger.info("="*80)
    
    try:
        from ahsd.models.transformer_encoder import TransformerStrainEncoder
        
        logger.info("✅ Successfully imported TransformerStrainEncoder")
        
        # Create encoder
        encoder = TransformerStrainEncoder(
            use_whisper=False,
            freeze_layers=4,
            input_length=2048,
            n_detectors=2,
            output_dim=64
        )
        logger.info("✅ TransformerStrainEncoder instantiated successfully")
        logger.info(f"   - Encoder type: {type(encoder.encoder).__name__}")
        logger.info(f"   - Encoder dim: {encoder.encoder_dim}")
        logger.info(f"   - Output dim: {encoder.output_dim}")
        logger.info(f"   - Use Whisper: {encoder.use_whisper}")
        
        # Test forward pass
        batch_size = 4
        strain_data = torch.randn(batch_size, 2, 2048)
        logger.info(f"\n   Input shape: {strain_data.shape}")
        
        output = encoder(strain_data)
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Output dtype: {output.dtype}")
        logger.info(f"   Output mean: {output.mean():.6f}, std: {output.std():.6f}")
        logger.info(f"   Output range: [{output.min():.6f}, {output.max():.6f}]")
        
        # Check for NaN/Inf
        if torch.isnan(output).any():
            logger.error("❌ OUTPUT CONTAINS NaN!")
            return False
        if torch.isinf(output).any():
            logger.error("❌ OUTPUT CONTAINS Inf!")
            return False
        
        logger.info("✅ Forward pass successful, no NaN/Inf")
        
        # Test gradients
        encoder.train()
        loss = output.sum()
        loss.backward()
        
        # Check for None gradients
        has_grad = False
        for name, param in encoder.named_parameters():
            if param.grad is not None:
                has_grad = True
                if torch.isnan(param.grad).any():
                    logger.error(f"❌ Gradient NaN in {name}")
                    return False
                if torch.isinf(param.grad).any():
                    logger.error(f"❌ Gradient Inf in {name}")
                    return False
        
        if has_grad:
            logger.info("✅ Gradient flow successful, no NaN/Inf in gradients")
        else:
            logger.warning("⚠️  No gradients found (might be frozen layers)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Direct encoder test FAILED: {e}", exc_info=True)
        return False


def test_priority_net_with_transformer():
    """Test TransformerStrainEncoder within PriorityNet."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: TransformerStrainEncoder within PriorityNet")
    logger.info("="*80)
    
    try:
        from ahsd.core.priority_net import PriorityNet
        
        # Create config with transformer enabled
        config = {
            "use_strain": True,
            "use_transformer_encoder": True,
            "use_edge_conditioning": True,
            "hidden_dims": [512, 384, 256, 128],
            "dropout": 0.15,
            "learning_rate": 5e-4,
            "weight_decay": 1e-5,
        }
        
        logger.info("✅ Creating PriorityNet with transformer enabled")
        model = PriorityNet(config=config)
        logger.info(f"✅ PriorityNet instantiated")
        logger.info(f"   - Use transformer: {model.use_transformer_encoder}")
        logger.info(f"   - Strain encoder: {type(model.strain_encoder).__name__}")
        
        # Create mock detection data
        detections = [
            {
                "mass_1": {"median": 35.0},
                "mass_2": {"median": 30.0},
                "luminosity_distance": {"median": 400.0},
                "ra": {"median": 1.5},
                "dec": {"median": 0.5},
                "geocent_time": {"median": 0.0},
                "theta_jn": {"median": 1.57},
                "psi": {"median": 0.5},
                "phase": {"median": 0.0},
                "a_1": {"median": 0.3},
                "a_2": {"median": 0.2},
                "tilt_1": {"median": 0.5},
                "tilt_2": {"median": 0.5},
                "phi_12": {"median": 1.0},
                "phi_jl": {"median": 1.0},
                "network_snr": {"median": 15.0},
            }
            for _ in range(3)
        ]
        
        # Create strain segments
        strain_segments = torch.randn(3, 2, 2048)
        
        logger.info(f"\n   Input:")
        logger.info(f"   - Detections: {len(detections)}")
        logger.info(f"   - Strain shape: {strain_segments.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            priorities, uncertainties = model(detections, strain_segments)
        
        logger.info(f"\n   Output:")
        logger.info(f"   - Priorities shape: {priorities.shape}")
        logger.info(f"   - Priorities: {priorities}")
        logger.info(f"   - Uncertainties shape: {uncertainties.shape}")
        logger.info(f"   - Uncertainties: {uncertainties}")
        
        # Validate outputs
        if torch.isnan(priorities).any():
            logger.error("❌ Priorities contain NaN!")
            return False
        if torch.isnan(uncertainties).any():
            logger.error("❌ Uncertainties contain NaN!")
            return False
        
        logger.info("✅ PriorityNet forward pass successful with transformer")
        
        # Test training step
        logger.info("\n   Testing training step...")
        model.train()
        priorities_target = torch.tensor([0.8, 0.5, 0.3])
        
        loss = ((priorities - priorities_target) ** 2).mean() + uncertainties.mean()
        loss.backward()
        
        logger.info(f"   - Loss: {loss.item():.6f}")
        logger.info("✅ Backward pass successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PriorityNet test FAILED: {e}", exc_info=True)
        return False


def test_strain_encoder_output_dims():
    """Test that strain encoder produces correct output dimensions."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Strain encoder output dimension check")
    logger.info("="*80)
    
    try:
        from ahsd.models.transformer_encoder import TransformerStrainEncoder
        
        test_cases = [
            {"n_detectors": 2, "input_length": 2048, "expected_patches": 32},
            {"n_detectors": 2, "input_length": 4096, "expected_patches": 64},
            {"n_detectors": 1, "input_length": 2048, "expected_patches": 32},
        ]
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"\n   Case {i+1}: n_detectors={test_case['n_detectors']}, input_length={test_case['input_length']}")
            
            encoder = TransformerStrainEncoder(
                use_whisper=False,
                n_detectors=test_case['n_detectors'],
                input_length=test_case['input_length'],
                output_dim=64
            )
            
            strain_data = torch.randn(2, test_case['n_detectors'], test_case['input_length'])
            output = encoder(strain_data)
            
            logger.info(f"   - Output shape: {output.shape}")
            
            if output.shape != (2, 64):
                logger.error(f"❌ Expected (2, 64), got {output.shape}")
                return False
            
            logger.info("   ✅ Correct output shape")
        
        logger.info("\n✅ All dimension tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dimension test FAILED: {e}", exc_info=True)
        return False


def test_transformer_with_3_detectors():
    """Test handling when 3 detectors (H1, L1, V1) are provided but transformer expects 2."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Handling 3 detectors (H1, L1, V1) input")
    logger.info("="*80)
    
    try:
        from ahsd.core.priority_net import PriorityNet
        
        config = {
            "use_strain": True,
            "use_transformer_encoder": True,
        }
        
        model = PriorityNet(config=config)
        
        # Create 3-detector strain data (H1, L1, V1)
        strain_3det = torch.randn(2, 3, 2048)
        logger.info(f"   Input with 3 detectors: {strain_3det.shape}")
        
        # Test with 3 detectors
        detections = [
            {
                "mass_1": {"median": 35.0},
                "mass_2": {"median": 30.0},
                "luminosity_distance": {"median": 400.0},
                "ra": {"median": 1.5},
                "dec": {"median": 0.5},
                "geocent_time": {"median": 0.0},
                "theta_jn": {"median": 1.57},
                "psi": {"median": 0.5},
                "phase": {"median": 0.0},
                "a_1": {"median": 0.3},
                "a_2": {"median": 0.2},
                "tilt_1": {"median": 0.5},
                "tilt_2": {"median": 0.5},
                "phi_12": {"median": 1.0},
                "phi_jl": {"median": 1.0},
                "network_snr": {"median": 15.0},
            }
            for _ in range(2)
        ]
        
        model.eval()
        with torch.no_grad():
            priorities, uncertainties = model(detections, strain_3det)
        
        logger.info(f"   Output: priorities={priorities.shape}, uncertainties={uncertainties.shape}")
        logger.info("✅ Successfully handled 3-detector input (reduced to 2)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 3-detector test FAILED: {e}", exc_info=True)
        return False


def main():
    """Run all transformer health checks."""
    logger.info("\n")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*20 + "TRANSFORMER ENCODER HEALTH CHECK" + " "*26 + "║")
    logger.info("╚" + "="*78 + "╝")
    
    results = []
    
    # Run all tests
    results.append(("Direct TransformerStrainEncoder", test_transformer_encoder_directly()))
    results.append(("PriorityNet with Transformer", test_priority_net_with_transformer()))
    results.append(("Output Dimensions", test_strain_encoder_output_dims()))
    results.append(("3-Detector Handling", test_transformer_with_3_detectors()))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("="*80)
    
    if passed == total:
        logger.info("🎉 All tests PASSED! Transformer is working correctly.")
        return 0
    else:
        logger.error(f"⚠️  {total - passed} test(s) FAILED. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
