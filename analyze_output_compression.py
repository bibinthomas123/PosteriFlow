#!/usr/bin/env python3
"""
Analyze output compression problem in PriorityNet training.
Goal: Understand why predictions are compressed to narrow band [0.085, 0.767] vs target [0.11, 0.95]
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def analyze_compression():
    """Analyze the output compression issue from logs"""
    
    # Data from logs
    epoch_38 = {
        'pred_min': 7.529e-02,
        'pred_max': 7.727e-01,
        'tgt_min': 1.101e-01,
        'tgt_max': 9.500e-01,
    }
    
    epoch_39 = {
        'pred_min': 8.521e-02,
        'pred_max': 7.673e-01,
        'tgt_min': 1.101e-01,
        'tgt_max': 9.500e-01,
    }
    
    logger.info("=" * 80)
    logger.info("OUTPUT COMPRESSION ANALYSIS")
    logger.info("=" * 80)
    
    for epoch, data in [('38', epoch_38), ('39', epoch_39)]:
        pred_range = data['pred_max'] - data['pred_min']
        tgt_range = data['tgt_max'] - data['tgt_min']
        compression = pred_range / tgt_range
        
        logger.info(f"\nEpoch {epoch}:")
        logger.info(f"  Predictions: [{data['pred_min']:.4f}, {data['pred_max']:.4f}]")
        logger.info(f"  Targets:     [{data['tgt_min']:.4f}, {data['tgt_max']:.4f}]")
        logger.info(f"  Pred range:  {pred_range:.4f}")
        logger.info(f"  Tgt range:   {tgt_range:.4f}")
        logger.info(f"  Compression: {compression:.1%} (should be ~100%)")
        logger.info(f"  Gap:")
        logger.info(f"    - Min gap: {data['pred_min'] - data['tgt_min']:+.4f}")
        logger.info(f"    - Max gap: {data['pred_max'] - data['tgt_max']:+.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("ROOT CAUSE ANALYSIS")
    logger.info("=" * 80)
    
    logger.info("""
OBSERVATION: Predictions compressed to 69.3% of target range (only 69% of expansion achieved)

CAUSES:
1. **Affine Transformation Underutilization**
   - Config: gain ∈ [1.2, 2.5], bias ∈ [-0.2, 0.05]
   - If gain=1.2: only 20% expansion above base output
   - If gain=2.5: 150% expansion (should achieve full range)
   - Issue: Affine is either NOT trained properly or clamped to low gain (≈1.2)
   
2. **Calibration Loss Too Weak**
   - calib_max_weight: 0.50 → max_gap penalty is small
   - calib_range_weight: 0.40 → range_gap penalty is small
   - ratio_penalty = max(0, 1 - pred_max/tgt_max) 
     = max(0, 1 - 0.767/0.950)
     = max(0, 1 - 0.807)
     = 0.193
   - With weight 0.50: loss contribution only 0.50 × 0.193 = 0.097
   - TOO WEAK to override other loss components!
   
3. **Competing Loss Components**
   - mse_weight: 0.20 (penalizes large predictions → shrinkage)
   - ranking_weight: 0.70 (cares about ordering, not range)
   - uncertainty_weight: 0.10
   - Total: 6 loss components competing for gradients
   - Compression forces lower MSE (safe zone near 0.5)
   
4. **Affine Clamping Preventing Range**
   - Even if unclamped gain is 2.5, clamping to [1.2, 2.5] might reduce it
   - If trained gain is pushed toward 1.2 by MSE, range stays compressed

SOLUTION STRATEGY:
1. **Increase calibration weight by 5-10×**
   - calib_max_weight: 0.50 → 2.0-3.0
   - calib_range_weight: 0.40 → 1.5-2.0
   - Make range expansion DOMINANT over MSE during training

2. **Reduce MSE weight** to prevent shrinkage pressure
   - mse_weight: 0.20 → 0.05-0.10
   - Ranking loss should be primary ordering signal

3. **Increase minimum variance penalty** further
   - min_variance_penalty: 2.0 → 5.0-10.0
   - Prevent predictions from clustering

4. **Remove or relax upper clipping** on predictions
   - Line 566: prio = torch.clamp(prio, 0, 1)
   - This hard-clips predictions; if affine tries to expand beyond 1.0,
     it gets clipped, killing gradient signal
   - Solution: Move clipping to loss penalty only (keep it soft via penalty)

5. **Monitor affine parameters in training**
   - Log self.prio_gain and self.prio_bias per epoch
   - If gain stays ≈1.2, it's not being trained properly
   - If gain is large but clipping kills gradient, that's the issue
""")

def estimate_required_weights():
    """Estimate weights needed to achieve full range expansion"""
    logger.info("\n" + "=" * 80)
    logger.info("WEIGHT TUNING GUIDANCE")
    logger.info("=" * 80)
    
    logger.info("""
Current compression: 69.3% (pred_range / tgt_range)
Target: 95%+ compression (pred_range ≥ 0.95 × tgt_range)

Gradient analysis:
- To increase pred_max from 0.767 to 0.95:
  - Δ = 0.95 - 0.767 = 0.183
  - Requires: 183 mV increase or 23.8% range expansion
  - Using gain: gain = (0.95 - bias) / (base_output + ε)
  - If base ≈ 0.4, bias ≈ -0.1, then gain ≈ (0.95 + 0.1) / 0.4 ≈ 2.625
  
Loss gradient magnitude estimate:
  - MSE gradient: O(0.2 × 2 × error) = O(0.4 × error) ≈ O(0.01-0.1) for small errors
  - Calibration gradient: O(0.50 × ∂ratio_penalty) = O(0.50 × 0.2) ≈ O(0.1)
  - Ratio: MSE_gradient ≈ 2× Calibration_gradient
  - Result: MSE pulling down outweighs calibration pulling up

Required weight ratio:
  - To make calibration dominant: calib_weight ≥ 2-3 × mse_weight
  - If mse_weight = 0.10: calib_weights ≥ 0.20-0.30 minimum
  
Recommended config changes:
1. calib_max_weight: 0.50 → 2.5 (5× increase)
2. calib_range_weight: 0.40 → 2.0 (5× increase)
3. mse_weight: 0.20 → 0.05 (4× decrease)
4. min_variance_penalty multiplier: 2.0 → 5.0 (still reasonable)
""")

if __name__ == '__main__':
    analyze_compression()
    estimate_required_weights()
