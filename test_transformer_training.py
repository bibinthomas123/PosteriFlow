#!/usr/bin/env python3
"""
Test TransformerStrainEncoder during actual training with detailed logging.
Simulates a real training step to verify gradient flow and loss computation.
"""

import torch
import torch.nn as nn
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('transformer_training_test.log')
    ]
)

logger = logging.getLogger(__name__)

def test_training_with_transformer():
    """Test a full training step with transformer encoder."""
    logger.info("="*80)
    logger.info("TRAINING TEST: TransformerStrainEncoder with PriorityNet")
    logger.info("="*80)
    
    try:
        from ahsd.core.priority_net import PriorityNet, PriorityNetTrainer
        
        # Create config with transformer
        config = {
            "use_strain": True,
            "use_transformer_encoder": True,
            "use_edge_conditioning": True,
            "hidden_dims": [512, 384, 256, 128],
            "dropout": 0.15,
            "learning_rate": 5e-4,
            "weight_decay": 1e-5,
            "ranking_weight": 0.50,
            "mse_weight": 0.35,
            "uncertainty_weight": 0.15,
            "warmup_epochs": 2,
            "scheduler_patience": 5,
        }
        
        logger.info("✅ Creating PriorityNet with transformer...")
        model = PriorityNet(config=config)
        logger.info(f"✅ Model created: {type(model).__name__}")
        logger.info(f"   - Use transformer: {model.use_transformer_encoder}")
        
        # Create trainer
        logger.info("\n✅ Creating PriorityNetTrainer...")
        trainer = PriorityNetTrainer(model, config)
        logger.info("✅ Trainer initialized")
        
        # Create mock batch of data
        logger.info("\n" + "="*80)
        logger.info("CREATING MOCK TRAINING DATA")
        logger.info("="*80)
        
        batch_size = 2
        n_signals_per_batch = [3, 2]  # Variable batch sizes
        
        # Create detection batches
        detections_batch = []
        for n_signals in n_signals_per_batch:
            detections = [
                {
                    "mass_1": {"median": 30.0 + i * 5},
                    "mass_2": {"median": 25.0 + i * 3},
                    "luminosity_distance": {"median": 350.0 + i * 100},
                    "ra": {"median": 1.5 + i * 0.1},
                    "dec": {"median": 0.5 + i * 0.1},
                    "geocent_time": {"median": 0.0 + i * 0.01},
                    "theta_jn": {"median": 1.57},
                    "psi": {"median": 0.5 + i * 0.1},
                    "phase": {"median": 0.0 + i * 0.1},
                    "a_1": {"median": 0.3 + i * 0.05},
                    "a_2": {"median": 0.2 + i * 0.05},
                    "tilt_1": {"median": 0.5 + i * 0.1},
                    "tilt_2": {"median": 0.5 + i * 0.1},
                    "phi_12": {"median": 1.0 + i * 0.1},
                    "phi_jl": {"median": 1.0 + i * 0.1},
                    "network_snr": {"median": 12.0 + i * 2},
                }
                for i in range(n_signals)
            ]
            detections_batch.append(detections)
        
        # Create target priorities
        priorities_batch = [
            torch.tensor([0.8, 0.5, 0.2]),  # 3 signals
            torch.tensor([0.7, 0.3]),        # 2 signals
        ]
        
        # Create strain segments (one per scenario)
        strain_batch = [
            torch.randn(3, 2, 2048),  # 3 signals, 2 detectors (H1, L1), 2048 samples
            torch.randn(2, 2, 2048),  # 2 signals, 2 detectors, 2048 samples
        ]
        
        logger.info(f"✅ Created batch with {batch_size} scenarios")
        for i, (dets, prio, strain) in enumerate(zip(detections_batch, priorities_batch, strain_batch)):
            logger.info(f"   Scenario {i+1}: {len(dets)} detections, priorities shape {prio.shape}, strain shape {strain.shape}")
        
        # Training step
        logger.info("\n" + "="*80)
        logger.info("TRAINING STEP")
        logger.info("="*80)
        
        losses = trainer.train_step(
            detections_batch=detections_batch,
            priorities_batch=priorities_batch,
            strain_batch=strain_batch,
        )
        
        logger.info(f"\n✅ Training step completed!")
        logger.info(f"   Loss metrics:")
        for key, val in losses.items():
            logger.info(f"      {key}: {val:.6f}")
        
        # Check for NaN/Inf in losses
        for key, val in losses.items():
            if not isinstance(val, (int, float)):
                continue
            if not (val == val):  # NaN check
                logger.error(f"   ❌ {key} is NaN!")
                return False
            if val == float('inf') or val == float('-inf'):
                logger.error(f"   ❌ {key} is Inf!")
                return False
        
        logger.info("✅ No NaN/Inf in loss values")
        
        # Test forward pass in eval mode
        logger.info("\n" + "="*80)
        logger.info("EVALUATION MODE")
        logger.info("="*80)
        
        model.eval()
        with torch.no_grad():
            priorities, uncertainties = model(detections_batch[0], strain_batch[0])
        
        logger.info(f"✅ Eval forward pass successful")
        logger.info(f"   Priorities: {priorities}")
        logger.info(f"   Uncertainties: {uncertainties}")
        
        if torch.isnan(priorities).any() or torch.isnan(uncertainties).any():
            logger.error("   ❌ NaN in output!")
            return False
        
        logger.info("✅ No NaN/Inf in predictions")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training test FAILED: {e}", exc_info=True)
        return False


def test_inference_vs_training_mode():
    """Test that transformer behaves correctly in both train and eval modes."""
    logger.info("\n" + "="*80)
    logger.info("MODE TRANSITION TEST")
    logger.info("="*80)
    
    try:
        from ahsd.core.priority_net import PriorityNet
        
        config = {
            "use_strain": True,
            "use_transformer_encoder": True,
        }
        
        model = PriorityNet(config=config)
        
        # Create test data
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
        strain = torch.randn(2, 2, 2048)
        
        # Test training mode
        logger.info("\n   Testing TRAINING mode...")
        model.train()
        priorities_train, uncertainties_train = model(detections, strain)
        logger.info(f"   ✅ Training mode: priorities={priorities_train.shape}, uncertainties={uncertainties_train.shape}")
        
        # Test eval mode
        logger.info("\n   Testing EVAL mode...")
        model.eval()
        with torch.no_grad():
            priorities_eval, uncertainties_eval = model(detections, strain)
        logger.info(f"   ✅ Eval mode: priorities={priorities_eval.shape}, uncertainties={uncertainties_eval.shape}")
        
        # Check shapes match
        if priorities_train.shape != priorities_eval.shape:
            logger.error(f"   ❌ Shape mismatch: {priorities_train.shape} vs {priorities_eval.shape}")
            return False
        
        logger.info("✅ Mode transitions successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mode transition test FAILED: {e}", exc_info=True)
        return False


def main():
    """Run all training tests."""
    logger.info("\n")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*15 + "TRANSFORMER TRAINING & INFERENCE TEST" + " "*25 + "║")
    logger.info("╚" + "="*78 + "╝")
    
    results = []
    
    results.append(("Training with Transformer", test_training_with_transformer()))
    results.append(("Mode Transitions", test_inference_vs_training_mode()))
    
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
        logger.info("🎉 All training tests PASSED!")
        logger.info("\n⚠️  IMPORTANT: Review the log files for transformer debug output:")
        logger.info("   - transformer_training_test.log (detailed execution trace)")
        logger.info("   - transformer_health_check.log (earlier health check)")
        return 0
    else:
        logger.error(f"⚠️  {total - passed} test(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
