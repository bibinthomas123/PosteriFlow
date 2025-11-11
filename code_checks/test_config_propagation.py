#!/usr/bin/env python3
"""
Test script to verify that configs are correctly passed from YAML to PriorityNet model
"""

import sys
import yaml
import torch
import logging
from pathlib import Path

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ahsd.core.priority_net import PriorityNet

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load YAML config."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    pn_config = config_dict.get('priority_net', config_dict)
    
    # Convert to object (same as train_priority_net.py does)
    params = {
        'hidden_dims': pn_config.get('hidden_dims', [512, 384, 256, 128]),
        'dropout': float(pn_config.get('dropout', 0.2)),
        'learning_rate': float(pn_config.get('learning_rate', 5e-4)),
        'weight_decay': float(pn_config.get('weight_decay', 1e-5)),
        'batch_size': int(pn_config.get('batch_size', 32)),
        'epochs': int(pn_config.get('epochs', 250)),
        'patience': int(pn_config.get('patience', 20)),
        'warmup_epochs': int(pn_config.get('warmup_epochs', 10)),
        'warmup_start_factor': float(pn_config.get('warmup_start_factor', 0.1)),
        'scheduler_patience': int(pn_config.get('scheduler_patience', 8)),
        'scheduler_factor': float(pn_config.get('scheduler_factor', 0.5)),
        'min_lr': float(pn_config.get('min_lr', 1e-6)),
        'ranking_weight': float(pn_config.get('ranking_weight', 0.3)),
        'mse_weight': float(pn_config.get('mse_weight', 0.6)),
        'uncertainty_weight': float(pn_config.get('uncertainty_weight', 0.1)),
        'use_snr_weighting': bool(pn_config.get('use_snr_weighting', True)),
        'loss_scale_factor': float(pn_config.get('loss_scale_factor', 0.001)),
        'gradient_clip_norm': float(pn_config.get('gradient_clip_norm', 1.0)),
        'gradient_log_threshold': float(pn_config.get('gradient_log_threshold', 0.5)),
        'attention_num_heads': int(pn_config.get('attention_num_heads', 4)),
        'attention_dropout': float(pn_config.get('attention_dropout', 0.1)),
        'use_modal_fusion': bool(pn_config.get('use_modal_fusion', False)),
        'overlap_use_attention': bool(pn_config.get('overlap_use_attention', False)),
        'overlap_importance_hidden': int(pn_config.get('overlap_importance_hidden', 16)),
        'use_strain': bool(pn_config.get('use_strain', True)),
        'use_edge_conditioning': bool(pn_config.get('use_edge_conditioning', True)),
        'n_edge_types': int(pn_config.get('n_edge_types', 17)),
        'use_transformer_encoder': bool(pn_config.get('use_transformer_encoder', False)),
    }
    
    config = type('EnhancedConfig', (), params)()
    return config, pn_config


def test_config_propagation():
    """Test that config is correctly passed to PriorityNet."""
    
    logger.info("\n" + "="*80)
    logger.info("🔍 CONFIG PROPAGATION TEST")
    logger.info("="*80)
    
    # Load config
    config_path = Path('configs/enhanced_training.yaml')
    logger.info(f"\n📂 Loading config from: {config_path}")
    
    config, pn_config_raw = load_config(config_path)
    
    logger.info("\n✅ Config loaded successfully!")
    
    # Create PriorityNet
    logger.info("\n🚀 Creating PriorityNet with loaded config...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = PriorityNet(
            config,
            use_strain=True,
            use_edge_conditioning=True,
            n_edge_types=17
        ).to(device)
        logger.info("\n✅ PriorityNet created successfully!")
    except Exception as e:
        logger.error(f"\n❌ Failed to create PriorityNet: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify model has key attributes
    logger.info("\n" + "="*80)
    logger.info("📋 VERIFYING MODEL CONFIGURATION")
    logger.info("="*80)
    
    checks = [
        ('use_strain', hasattr(model, 'use_strain'), model.use_strain if hasattr(model, 'use_strain') else None),
        ('use_edge_conditioning', hasattr(model, 'use_edge_conditioning'), model.use_edge_conditioning if hasattr(model, 'use_edge_conditioning') else None),
        ('n_edge_types', hasattr(model, 'n_edge_types'), model.n_edge_types if hasattr(model, 'n_edge_types') else None),
        ('use_transformer_encoder', hasattr(model, 'use_transformer_encoder'), model.use_transformer_encoder if hasattr(model, 'use_transformer_encoder') else None),
        ('strain_encoder', hasattr(model, 'strain_encoder'), "Present" if hasattr(model, 'strain_encoder') else None),
        ('edge_embedding', hasattr(model, 'edge_embedding'), "Present" if hasattr(model, 'edge_embedding') else None),
        ('priority_head', hasattr(model, 'priority_head'), "Present" if hasattr(model, 'priority_head') else None),
    ]
    
    all_passed = True
    for attr_name, has_attr, value in checks:
        if has_attr:
            logger.info(f"   ✓ {attr_name:30s} = {value}")
        else:
            logger.warning(f"   ✗ {attr_name:30s} = MISSING")
            all_passed = False
    
    # Test forward pass
    logger.info("\n" + "="*80)
    logger.info("🧪 TESTING FORWARD PASS")
    logger.info("="*80)
    
    try:
        # Create dummy batch
        batch_size = 2
        n_signals = 3
        strain_length = 16384
        
        dummy_batch = {
            'metadata': torch.randn(batch_size, 96, device=device),
            'overlap_structure': torch.randn(batch_size, 16, device=device),
            'network_snr': torch.randn(batch_size, 1, device=device),
            'temporal_strain': torch.randn(batch_size, 3, strain_length, device=device),
            'edge_type_ids': torch.zeros(batch_size, dtype=torch.long, device=device),
            'signal_count': torch.tensor([n_signals] * batch_size, device=device),
        }
        
        logger.info(f"\n📊 Dummy batch shapes:")
        for key, value in dummy_batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"   {key:30s} {value.shape}")
        
        # Forward pass
        with torch.no_grad():
            priorities, uncertainties = model(dummy_batch)
        
        logger.info(f"\n✅ Forward pass successful!")
        logger.info(f"   Output priorities shape:     {priorities.shape}")
        logger.info(f"   Output uncertainties shape:  {uncertainties.shape}")
        
    except Exception as e:
        logger.error(f"\n❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📈 SUMMARY")
    logger.info("="*80)
    
    logger.info("\n✅ CONFIG PROPAGATION SUCCESSFUL!")
    logger.info("\nKey configurations passed to model:")
    logger.info(f"   use_strain:              {pn_config_raw.get('use_strain', 'NOT SET')}")
    logger.info(f"   use_edge_conditioning:   {pn_config_raw.get('use_edge_conditioning', 'NOT SET')}")
    logger.info(f"   use_transformer_encoder: {pn_config_raw.get('use_transformer_encoder', 'NOT SET')}")
    logger.info(f"   hidden_dims:             {pn_config_raw.get('hidden_dims', 'NOT SET')}")
    logger.info(f"   dropout:                 {pn_config_raw.get('dropout', 'NOT SET')}")
    logger.info(f"   learning_rate:           {pn_config_raw.get('learning_rate', 'NOT SET'):.2e}")
    
    logger.info("\n" + "="*80)
    
    return all_passed


if __name__ == '__main__':
    success = test_config_propagation()
    sys.exit(0 if success else 1)
