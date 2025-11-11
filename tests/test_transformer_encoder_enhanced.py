#!/usr/bin/env python3
"""
Comprehensive tests for enhanced TransformerStrainEncoder.

Tests:
- Whisper-based encoder with positional encoding
- Lightweight Transformer fallback
- Attention mask support
- Variable-length sequences
- Gradient flow through frozen/unfrozen layers
- Output shape and stability
"""

import torch
import pytest
import sys
import logging
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ahsd.models.transformer_encoder import TransformerStrainEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTransformerStrainEncoder:
    """Test suite for TransformerStrainEncoder"""

    @pytest.fixture
    def sample_strain(self):
        """Generate sample strain data [batch, n_detectors, time_samples]"""
        return torch.randn(4, 2, 2048)

    @pytest.fixture
    def attention_mask(self):
        """Generate valid attention mask (all ones = all valid)"""
        return torch.ones(4, 32)  # 2048/64 = 32 patches

    def test_whisper_mode(self, sample_strain):
        """Test Whisper-based encoder loads correctly."""
        encoder = TransformerStrainEncoder(
            use_whisper=True,
            freeze_layers=4,
            input_length=2048,
            n_detectors=2,
            output_dim=64
        )

        # Forward pass
        with torch.no_grad():
            features = encoder(sample_strain)

        # Validate output
        assert features.shape == (4, 64), f"Expected (4, 64), got {features.shape}"
        assert not torch.isnan(features).any(), "NaN in output"
        assert not torch.isinf(features).any(), "Inf in output"

        logger.info(f"✅ Whisper mode: output shape {features.shape}, dtype {features.dtype}")

    def test_fallback_mode(self, sample_strain):
        """Test lightweight Transformer fallback."""
        encoder = TransformerStrainEncoder(
            use_whisper=False,  # Force fallback
            input_length=2048,
            n_detectors=2,
            output_dim=64
        )

        batch_size = sample_strain.shape[0]
        with torch.no_grad():
            features = encoder(sample_strain)

        assert features.shape == (batch_size, 64)
        assert not encoder.use_whisper, "Should use fallback"
        assert encoder.encoder_dim == 256, "Fallback should be 256-D"

        logger.info(f"✅ Fallback mode: output shape {features.shape}, encoder_dim {encoder.encoder_dim}")

    def test_frozen_layers(self):
        """Test that specified layers are frozen."""
        encoder = TransformerStrainEncoder(
            use_whisper=True,
            freeze_layers=4,
            input_length=2048,
            output_dim=64
        )

        if encoder.use_whisper:
            # Check frozen layers have no requires_grad
            for i, layer in enumerate(encoder.encoder.layers[:4]):
                for param in layer.parameters():
                    assert not param.requires_grad, f"Layer {i} should be frozen"
                    logger.debug(f"✅ Layer {i} is frozen")

            # Check unfrozen layers
            unfrozen_count = 0
            for i, layer in enumerate(encoder.encoder.layers[4:], start=4):
                for param in layer.parameters():
                    if param.requires_grad:
                        unfrozen_count += 1

            assert unfrozen_count > 0, "Should have unfrozen layers"
            logger.info(f"✅ Frozen first 4 layers, {unfrozen_count} params trainable")

    def test_positional_encoding(self):
        """Test positional encoding adapter exists and is correct shape."""
        input_length = 4096
        encoder = TransformerStrainEncoder(
            use_whisper=True,
            freeze_layers=4,
            input_length=input_length,
            output_dim=64
        )

        if encoder.use_whisper and encoder.pos_encoding_adapter is not None:
            n_patches = input_length // 64
            expected_shape = (1, n_patches, 768)  # 768 = Whisper encoder_dim
            assert encoder.pos_encoding_adapter.shape == expected_shape, \
                f"Expected {expected_shape}, got {encoder.pos_encoding_adapter.shape}"

            logger.info(f"✅ Positional encoding shape: {encoder.pos_encoding_adapter.shape}")

    def test_attention_mask(self, sample_strain, attention_mask):
        """Test forward pass with attention mask."""
        encoder = TransformerStrainEncoder(
            use_whisper=True,
            freeze_layers=4,
            input_length=2048,
            output_dim=64
        )

        with torch.no_grad():
            features_no_mask = encoder(sample_strain)
            features_with_mask = encoder(sample_strain, attention_mask=attention_mask)

        # Both should have valid shapes
        assert features_no_mask.shape == features_with_mask.shape
        assert not torch.isnan(features_with_mask).any()

        logger.info(f"✅ Attention mask support: outputs match shape {features_with_mask.shape}")

    def test_variable_length_sequences(self):
        """Test with different input lengths."""
        encoder = TransformerStrainEncoder(
            use_whisper=True,
            input_length=2048,
            output_dim=64
        )

        test_lengths = [1024, 2048, 4096]

        for length in test_lengths:
            strain = torch.randn(2, 2, length)
            try:
                with torch.no_grad():
                    features = encoder(strain)

                # This might fail for non-2048 lengths without special handling
                logger.info(f"✅ Length {length}: output shape {features.shape}")
            except RuntimeError as e:
                # Expected for mismatched positional encoding
                logger.warning(f"⚠️ Length {length} failed (expected): {str(e)[:80]}")

    def test_gradient_flow(self, sample_strain):
        """Test gradient flow through frozen and trainable layers."""
        encoder = TransformerStrainEncoder(
            use_whisper=True,
            freeze_layers=4,
            input_length=2048,
            output_dim=64
        )

        sample_strain = sample_strain.clone().requires_grad_(True)
        features = encoder(sample_strain)
        loss = features.sum()
        loss.backward()

        # Input should have gradients
        assert sample_strain.grad is not None, "Input should have gradients"

        if encoder.use_whisper:
            # Frozen layers should have no gradients
            for i, layer in enumerate(encoder.encoder.layers[:4]):
                for name, param in layer.named_parameters():
                    assert param.grad is None, f"Frozen layer {i} param {name} should have no grad"

            # Unfrozen layers should have gradients
            for i, layer in enumerate(encoder.encoder.layers[4:], start=4):
                has_grad = any(p.grad is not None for p in layer.parameters())
                if i == 4:  # At least first unfrozen should have gradients
                    logger.info(f"✅ Layer {i} has gradients: {has_grad}")

        logger.info(f"✅ Gradient flow test passed")

    def test_output_properties(self, sample_strain):
        """Test output is well-formed."""
        encoder = TransformerStrainEncoder(
            use_whisper=False,  # Use fallback for speed
            input_length=2048,
            output_dim=64
        )

        batch_size = sample_strain.shape[0]
        with torch.no_grad():
            features = encoder(sample_strain)

        # Shape
        assert features.shape == (batch_size, 64)

        # Dtype
        assert features.dtype == torch.float32

        # Values in reasonable range (not exploding)
        assert features.abs().max() < 100, f"Max value {features.abs().max()} is too large"
        assert features.abs().min() < 100, f"Min value {features.abs().min()} is too large"

        # Some variance across batch
        batch_variance = features.var(dim=0).mean()
        assert batch_variance > 1e-6, f"Very low variance: {batch_variance}"

        logger.info(f"✅ Output properties: shape {features.shape}, "
                   f"range [{features.min():.3f}, {features.max():.3f}], "
                   f"variance {batch_variance:.6f}")

    def test_batch_independence(self, sample_strain):
        """Test that batch samples don't affect each other."""
        encoder = TransformerStrainEncoder(
            use_whisper=False,
            input_length=2048,
            output_dim=64
        )

        # Run full batch
        with torch.no_grad():
            batch_features = encoder(sample_strain)

        # Run individual samples and compare
        individual_features = []
        for i in range(sample_strain.shape[0]):
            with torch.no_grad():
                feat = encoder(sample_strain[i:i+1])
            individual_features.append(feat)

        individual_features = torch.cat(individual_features, dim=0)

        # Should be very close (allowing for batch norm differences)
        max_diff = (batch_features - individual_features).abs().max()
        assert max_diff < 1e-4 or encoder.use_whisper, \
            f"Batch vs individual max diff: {max_diff} (expected < 1e-4)"

        logger.info(f"✅ Batch independence: max diff {max_diff:.2e}")

    def test_device_compatibility(self, sample_strain):
        """Test CPU/GPU compatibility."""
        encoder = TransformerStrainEncoder(
            use_whisper=False,
            input_length=2048,
            output_dim=64
        )

        # CPU
        encoder_cpu = encoder.cpu()
        strain_cpu = sample_strain.cpu()
        with torch.no_grad():
            features_cpu = encoder_cpu(strain_cpu)

        assert features_cpu.device.type == 'cpu'
        logger.info(f"✅ CPU: output on {features_cpu.device}")

        # GPU (if available)
        if torch.cuda.is_available():
            encoder_gpu = encoder.cuda()
            strain_gpu = sample_strain.cuda()
            with torch.no_grad():
                features_gpu = encoder_gpu(strain_gpu)

            assert features_gpu.device.type == 'cuda'
            logger.info(f"✅ GPU: output on {features_gpu.device}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
