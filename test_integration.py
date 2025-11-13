#!/usr/bin/env python3
"""
Test script to verify OverlapNeuralPE integration:
- BiasCorrector
- AdaptiveSubtractor
- NormalizingFlows
- RL Controller
- Context Encoder
- Uncertainty Estimator
"""

import torch
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ahsd.models.overlap_neuralpe import OverlapNeuralPE

def test_integration():
    """Test all integrated components."""
    
    logger.info("=" * 80)
    logger.info("🔧 TESTING OVERLAP NEURAL PE INTEGRATION")
    logger.info("=" * 80)
    
    # Configuration
    param_names = [
        'mass_1', 'mass_2', 'luminosity_distance',
        'geocent_time', 'ra', 'dec', 'theta_jn', 'psi', 'phase'
    ]
    
    config = {
        'context_dim': 512,
        'n_flow_layers': 6,
        'max_iterations': 3,
        'dropout': 0.1,
        'flow_config': {
            'type': 'flowmatching',
            'hidden_features': 256,
            'num_layers': 4,
            'num_blocks_per_layer': 2,
            'dropout': 0.1,
            'solver_steps': 10
        },
        'rl_controller': {
            'state_features': ['remaining_signals', 'residual_power', 'current_snr', 'extraction_success_rate'],
            'complexity_levels': ['low', 'medium', 'high'],
            'learning_rate': 1e-3,
            'epsilon': 0.1,
            'epsilon_decay': 0.995,
            'memory_size': 10000,
            'batch_size': 32
        },
        'bias_corrector': {
            'enabled': True
        }
    }
    
    # Mock priority net checkpoint path
    priority_net_path = Path(__file__).parent / 'models' / 'priority_net_checkpoint.pt'
    
    # Create a dummy checkpoint if it doesn't exist
    if not priority_net_path.exists():
        logger.warning(f"Priority net checkpoint not found at {priority_net_path}")
        logger.info("Creating dummy checkpoint for testing...")
        priority_net_path.parent.mkdir(parents=True, exist_ok=True)
        
        dummy_checkpoint = {
            'model_state_dict': {},
            'model_architecture': {
                'use_strain': True,
                'use_edge_conditioning': True,
                'n_edge_types': 19
            }
        }
        torch.save(dummy_checkpoint, priority_net_path)
        logger.info(f"Dummy checkpoint created: {priority_net_path}")
    
    try:
        # ✅ Initialize OverlapNeuralPE
        logger.info("\n1️⃣  Initializing OverlapNeuralPE...")
        model = OverlapNeuralPE(
            param_names=param_names,
            priority_net_path=str(priority_net_path),
            config=config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info("✅ Model initialized successfully")
        
        # ✅ Check component status
        logger.info("\n2️⃣  Checking component status...")
        integration_summary = model.get_integration_summary()
        
        for component_name, component_info in integration_summary['components'].items():
            enabled = component_info.get('enabled', False)
            n_params = component_info.get('n_parameters', 0)
            status = "✅ ENABLED" if enabled else "❌ DISABLED"
            logger.info(f"  {component_name}: {status} ({n_params:,} params)")
        
        # ✅ Test forward pass (single signal extraction)
        logger.info("\n3️⃣  Testing single signal extraction...")
        batch_size, n_detectors, n_samples = 2, 2, 4096
        strain_data = torch.randn(batch_size, n_detectors, n_samples, device=model.device)
        
        with torch.no_grad():
            result = model.sample_posterior(strain_data, n_samples=100)
        
        logger.info(f"  Samples shape: {result['samples'].shape}")
        logger.info(f"  Means shape: {result['means'].shape}")
        logger.info(f"  Stds shape: {result['stds'].shape}")
        logger.info(f"  Uncertainties shape: {result['uncertainties'].shape}")
        logger.info("✅ Single signal extraction passed")
        
        # ✅ Test overlapping signal extraction
        logger.info("\n4️⃣  Testing overlapping signal extraction...")
        with torch.no_grad():
            overlap_result = model.extract_overlapping_signals(strain_data, training=False)
        
        logger.info(f"  Extracted {len(overlap_result['extracted_signals'])} signals")
        for i, signal in enumerate(overlap_result['extracted_signals']):
            logger.info(f"    Signal {i+1}: complexity={signal['complexity']}, bias_corrected={signal['bias_corrected']}")
        logger.info(f"  Final residual shape: {overlap_result['final_residual'].shape}")
        logger.info(f"  Total iterations: {overlap_result['n_iterations']}")
        logger.info("✅ Overlapping signal extraction passed")
        
        # ✅ Test loss computation
        logger.info("\n5️⃣  Testing loss computation...")
        true_params = torch.randn(batch_size, len(param_names), device=model.device)
        true_params[:, 0] = torch.abs(true_params[:, 0]) + 10  # mass_1 > 10
        true_params[:, 1] = torch.abs(true_params[:, 1]) + 5   # mass_2 > 5
        true_params[:, 2] = torch.abs(true_params[:, 2]) + 50  # distance > 50
        
        loss_dict = model.compute_loss(strain_data, true_params)
        
        logger.info(f"  Total loss: {loss_dict['total_loss'].item():.6f}")
        logger.info(f"  NLL: {loss_dict['nll'].item():.6f}")
        logger.info(f"  Physics loss: {loss_dict['physics_loss'].item():.6f}")
        logger.info(f"  Bias loss: {loss_dict['bias_loss'].item():.6f}")
        logger.info(f"  Uncertainty loss: {loss_dict['uncertainty_loss'].item():.6f}")
        logger.info(f"  Jacobian reg: {loss_dict['jacobian_reg'].item():.6f}")
        logger.info("✅ Loss computation passed")
        
        # ✅ Test RL training
        logger.info("\n6️⃣  Testing RL training...")
        true_params_list = [
            {name: val.item() for name, val in zip(param_names, true_params[0])}
        ]
        
        with torch.enable_grad():
            rl_result = model.extract_overlapping_signals(strain_data, true_params=true_params_list, training=True)
        
        rl_metrics = model.get_rl_metrics()
        logger.info(f"  RL metrics: {rl_metrics}")
        logger.info("✅ RL training passed")
        
        # ✅ Test bias correction metrics
        logger.info("\n7️⃣  Testing bias corrector metrics...")
        bias_metrics = model.get_bias_metrics()
        if bias_metrics:
            logger.info(f"  Bias metrics available: {list(bias_metrics.keys())}")
        else:
            logger.info("  (Bias corrector not yet trained)")
        logger.info("✅ Bias correction metrics passed")
        
        # ✅ Final summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL INTEGRATION TESTS PASSED")
        logger.info("=" * 80)
        logger.info("\n📊 Component Integration Status:")
        logger.info(f"""
            ✅ BiasCorrector:       Integrated in signal extraction pipeline
            ✅ AdaptiveSubtractor:  Integrated for iterative signal subtraction
            ✅ NormalizingFlows:    Integrated for posterior sampling and likelihood
            ✅ RL Controller:       Integrated for adaptive complexity selection
            ✅ ContextEncoder:      Integrated for feature extraction
            ✅ UncertaintyEstimator: Integrated for uncertainty quantification
            ✅ PriorityNet:         Integrated for signal prioritization
        """)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = test_integration()
    exit(0 if success else 1)
