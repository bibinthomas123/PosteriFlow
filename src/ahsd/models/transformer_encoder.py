#!/usr/bin/env python3
"""
Transformer-based strain encoder using Whisper audio encoder.
Replaces CNN+LSTM architecture with more efficient Transformer for GW strain analysis.

Features:
- Positional encoding adapter for GW sequence lengths
- Attention mask support for variable-length sequences
- Graceful fallback to lightweight Transformer
- Mixed precision training compatible
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

try:
    from transformers import WhisperModel, WhisperConfig

    HAS_WHISPER = True
    HAS_WHISPER_CONFIG = True
except ImportError:
    HAS_WHISPER = False
    HAS_WHISPER_CONFIG = False
    logging.warning("Whisper not available; will use lightweight Transformer fallback")


class TransformerStrainEncoder(nn.Module):
    """
    Transformer-based strain encoder for gravitational wave analysis.

    Uses either:
    - Pre-trained Whisper audio encoder (efficient, leverages speech similarity to GW)
    - Custom lightweight Transformer (when Whisper unavailable)

    Converts 2-channel strain data (H1, L1 detectors) to fixed 64-D feature vector.
    """

    def __init__(
        self,
        use_whisper: bool = True,
        freeze_layers: int = 4,
        input_length: int = 2048,
        n_detectors: int = 2,
        output_dim: int = 64,
    ):
        """
        Initialize the Transformer strain encoder.

        Args:
            use_whisper: If True, use pre-trained Whisper; else train from scratch.
            freeze_layers: Number of Whisper encoder layers to freeze (generic features).
            input_length: Number of time samples per detector (e.g., 2048 for 1s at 2048 Hz).
            n_detectors: Number of detector channels (2 for H1, L1).
            output_dim: Output feature dimension (default 64).
        """
        super().__init__()

        self.use_whisper = use_whisper and HAS_WHISPER
        self.freeze_layers = freeze_layers
        self.output_dim = output_dim
        self.input_length = input_length
        self.n_detectors = n_detectors

        if self.use_whisper:
            # Load pre-trained Whisper audio encoder
            try:
                whisper_model = WhisperModel.from_pretrained("openai/whisper-small")
                self.encoder = whisper_model.encoder
                encoder_dim = self.encoder.config.d_model  # 768 for whisper-small


                config = self.encoder.config
                logging.info(
                    f"✅ Whisper encoder loaded:\n"
                    f"   - Dimension: {config.d_model}\n"
                    f"   - Layers: {config.encoder_layers} (freezing {freeze_layers})\n"
                    f"   - Attention heads: {config.encoder_attention_heads}\n"
                    f"   - FFN dimension: {config.encoder_ffn_dim}\n"
                    f"   - Dropout: {config.dropout}\n"
                    f"   - Parameters: {sum(p.numel() for p in self.encoder.parameters()):,}"
                )
                
                # Freeze early layers (generic audio features)
                if freeze_layers > 0:
                    for layer in self.encoder.layers[:freeze_layers]:
                        for param in layer.parameters():
                            param.requires_grad = False

                # ✅ FIX: Add positional encoding adapter for GW sequences
                # Whisper expects ~1500 positions, GW has ~32 (2048/64)
                n_patches = input_length // 64
                self.pos_encoding_adapter = nn.Parameter(
                    torch.randn(1, n_patches, encoder_dim) * 0.02
                )

                logging.info(
                    f"✅ Whisper encoder loaded: {encoder_dim}-D, "
                    f"froze {freeze_layers}/{len(self.encoder.layers)} layers, "
                    f"pos_encoding for {n_patches} patches"
                )
            except Exception as e:
                logging.error(f"❌ Whisper loading FAILED: {e}")
                logging.warning(f"⚠️  Falling back to lightweight Transformer encoder")
                logging.warning(f"   This will create a checkpoint with mismatched encoder architecture!")
                logging.warning(f"   The checkpoint will use 256-D lightweight, not 768-D Whisper!")
                self.use_whisper = False
                self.encoder = None
                self.pos_encoding_adapter = None

        if not self.use_whisper:
            # Lightweight Transformer (6 layers, d_model=256) - using standard PyTorch
            self.encoder = LightweightTransformerEncoder(d_model=256, num_heads=8, num_layers=6)
            encoder_dim = 256
            self.pos_encoding_adapter = None
            logging.info("✅ Using lightweight Transformer encoder (6 layers, d_model=256)")

        self.encoder_dim = encoder_dim if self.encoder else 128

        # Patch embedding: convert strain channels to transformer-compatible patches
        # 64-sample patches at 2048 Hz = 31.25 ms windows
        self.patch_embed = nn.Conv1d(
            n_detectors, self.encoder_dim, kernel_size=64, stride=64, padding=0
        )

        # Projection to fixed output dimension
        self.output_proj = nn.Sequential(
            nn.Linear(self.encoder_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )

        logging.info(
            f"TransformerStrainEncoder: input_length={input_length}, "
            f"patch_dim={self.encoder_dim}, output_dim={output_dim}"
        )

    def forward(
        self, strain_data: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode whitened strain segments into a 64-D temporal representation.

        Args:
            strain_data: Tensor [batch, n_detectors, time_samples]
                         (e.g., [B, 2, 2048] for H1+L1 at 2048 Hz)
            attention_mask: Optional [batch, n_patches] mask (1=valid, 0=padding)
                           For variable-length sequences

        Returns:
            features: Tensor [batch, output_dim] temporal feature vector
        """
        batch_size = strain_data.shape[0]
        device = strain_data.device

        try:
            # ✅ FIX: Sanitize input for NaN/Inf before processing
            has_invalid = torch.isnan(strain_data).any() or torch.isinf(strain_data).any()
            if has_invalid:
                logging.warning("Detected NaN/Inf in strain_data input; sanitizing")
            strain_data = torch.nan_to_num(strain_data, nan=0.0, posinf=1e2, neginf=-1e2)
            strain_data = torch.clamp(strain_data, min=-1e2, max=1e2)
            
            # Patch embedding: [batch, n_detectors, time_samples] → [batch, encoder_dim, n_patches]
            patches = self.patch_embed(strain_data)  # [B, encoder_dim, n_patches]
            patches = patches.transpose(1, 2)  # [B, n_patches, encoder_dim]

            # ✅ FIX: Add positional encoding for Whisper
            if self.use_whisper and self.pos_encoding_adapter is not None:
                patches = patches + self.pos_encoding_adapter

            # Pass through Transformer encoder
            if self.encoder is not None:
                try:
                    if self.use_whisper:
                        # Whisper pathway with optional attention mask
                        encoder_output = self.encoder(
                            inputs_embeds=patches,
                            attention_mask=attention_mask,  # ✅ ADD MASK SUPPORT
                            return_dict=True,
                        )
                    else:
                        # Lightweight Transformer
                        encoder_output = self.encoder(inputs_embeds=patches, return_dict=True)
                    
                    hidden_states = encoder_output.last_hidden_state  # [B, n_patches, encoder_dim]
                except Exception as e:
                    logging.error(f"Transformer encoder forward failed: {e}")
                    hidden_states = patches
            else:
                hidden_states = patches

            # ✅ IMPROVED: Masked average pooling over time dimension
            if attention_mask is not None:
                # Masked pooling: ignore padding tokens
                mask_expanded = attention_mask.unsqueeze(-1)  # [B, n_patches, 1]
                sum_pooled = (hidden_states * mask_expanded).sum(dim=1)  # [B, encoder_dim]
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1]
                pooled = sum_pooled / sum_mask
            else:
                # Standard global average pooling
                pooled = hidden_states.mean(dim=1)  # [B, encoder_dim]

            # Project to output dimension
            features = self.output_proj(pooled)  # [B, output_dim]

            return features
            
        except Exception as e:
            logging.error(f"Transformer encoder forward failed: {e}", exc_info=True)
            # Return zeros as fallback
            return torch.zeros(batch_size, self.output_dim, device=device, dtype=strain_data.dtype)


class LightweightTransformerEncoder(nn.Module):
    """
    Lightweight Transformer encoder fallback (no external dependencies).

    Single transformer block with self-attention and FFN.
    Used when Whisper is unavailable or for resource-constrained scenarios.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8, num_layers: int = 2):
        """
        Initialize lightweight Transformer.

        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer blocks.
        """
        super().__init__()
        self.d_model = d_model

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, inputs_embeds: torch.Tensor, return_dict: bool = True):
        """
        Args:
            inputs_embeds: Tensor [batch, seq_len, d_model]
            return_dict: If True, return dict-like object with 'last_hidden_state'

        Returns:
            Dict with 'last_hidden_state' key (compatible with Whisper), or Tensor
        """
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)

        if return_dict:
            # Return object compatible with Whisper encoder output format
            class EncoderOutput:
                def __init__(self, last_hidden_state):
                    self.last_hidden_state = last_hidden_state

            return EncoderOutput(x)
        return x
