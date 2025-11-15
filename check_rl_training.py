#!/usr/bin/env python3
"""
Diagnose RL Controller training issues.

The RL Controller is initialized but not receiving any training data because:
1. Training only happens when extract_overlapping_signals() is called with training=True AND true_params
2. During normal inference/validation, these conditions are not met
3. The deques (action_history, reward_history) remain empty
4. Metrics show all zeros
"""

import sys
sys.path.insert(0, '/home/bibinathomas/PosteriFlow')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.ahsd.models.rl_controller import AdaptiveComplexityController
import numpy as np

# Test 1: Initialize controller
logger.info("=" * 80)
logger.info("TEST 1: RL Controller Initialization")
logger.info("=" * 80)

controller = AdaptiveComplexityController(
    state_features=["remaining_signals", "residual_power", "current_snr", "extraction_success_rate"],
    complexity_levels=["low", "medium", "high"],
    learning_rate=1e-3,
    epsilon=0.1,
    epsilon_decay=0.995,
    memory_size=10000,
    batch_size=32
)

metrics = controller.get_metrics()
logger.info("\nInitial metrics:")
for key, value in metrics.items():
    logger.info(f"  {key}: {value}")

logger.info(f"\nMemory size: {len(controller.memory)}")
logger.info(f"Action history length: {len(controller.action_history)}")
logger.info(f"Reward history length: {len(controller.reward_history)}")
logger.info(f"Action counts: {controller.action_counts}")

# Test 2: Simulate some experiences
logger.info("\n" + "=" * 80)
logger.info("TEST 2: Store & Train Experiences")
logger.info("=" * 80)

pipeline_state = {
    "remaining_signals": 5,
    "residual_power": 0.8,
    "current_snr": 15.0,
    "extraction_success_rate": 0.9,
}

for i in range(50):  # 50 experiences
    state = controller.get_state_vector(pipeline_state)
    action = controller.select_action(state, training=True)
    reward = np.random.normal(0.5, 0.2)  # Random reward
    next_state = controller.get_state_vector(pipeline_state)
    done = (i % 10 == 0)
    
    controller.store_experience(state, action, reward, next_state, done)
    
    logger.debug(f"Experience {i}: action={action}, reward={reward:.3f}")

logger.info(f"\nAfter storing 50 experiences:")
logger.info(f"  Memory size: {len(controller.memory)}")
logger.info(f"  Action history length: {len(controller.action_history)}")
logger.info(f"  Reward history length: {len(controller.reward_history)}")

# Test 3: Check metrics after experiences
metrics = controller.get_metrics()
logger.info("\nMetrics after storing experiences:")
for key, value in metrics.items():
    logger.info(f"  {key}: {value}")

# Test 4: Train the RL controller
logger.info("\n" + "=" * 80)
logger.info("TEST 3: Training Loop")
logger.info("=" * 80)

for epoch in range(5):
    # Add more experiences each epoch
    for i in range(50):
        state = controller.get_state_vector(pipeline_state)
        action = controller.select_action(state, training=True)
        reward = np.random.normal(0.5, 0.2)
        next_state = controller.get_state_vector(pipeline_state)
        done = (i % 10 == 0)
        controller.store_experience(state, action, reward, next_state, done)
    
    # Train step
    loss = controller.train_step()
    
    # Log metrics
    metrics = controller.get_metrics()
    logger.info(f"\nEpoch {epoch}:")
    logger.info(f"  RL Loss: {loss:.6f}" if loss else "  RL Loss: None")
    logger.info(f"  Epsilon: {metrics.get('epsilon', 0):.4f}")
    logger.info(f"  Avg Complexity: {metrics.get('avg_complexity', 0):.2f}")
    logger.info(f"  Avg Reward: {metrics.get('avg_reward', 0):.4f}")
    logger.info(f"  Action Entropy: {metrics.get('action_entropy', 0):.4f}")
    logger.info(f"  Memory Size: {metrics.get('memory_size', 0)}")

# Test 5: Issues found
logger.info("\n" + "=" * 80)
logger.info("DIAGNOSIS: ROOT CAUSES")
logger.info("=" * 80)

logger.info("""
The RL Controller is not working during normal training because:

1. ❌ TRAINING CONDITION: RL experiences are only stored in extract_overlapping_signals()
   when BOTH conditions are true:
   - training=True (must be explicitly set)
   - true_params is not None (must have ground truth)

2. ❌ DURING INFERENCE: When the model does normal inference:
   - extract_overlapping_signals() is called with training=False
   - NO experiences are stored
   - RL controller metrics remain at 0.0000

3. ❌ REWARD SIGNAL: Rewards depend on true_params for accuracy computation
   - Without ground truth, the reward signal is undefined

4. ❌ MISSING TRAINING SIGNAL: The RL controller can learn from:
   - Extraction accuracy (requires ground truth)
   - Pipeline efficiency (residual power, SNR)
   - Complexity-quality tradeoff

SOLUTIONS:
1. Add synthetic reward function based on residual power (no ground truth needed)
2. Ensure training=True during training epochs
3. Log when experiences ARE being collected vs when they're NOT
4. Make RL training happen during inference with weaker signals
""")

logger.info("\nRECOMMENDED FIX:")
logger.info("""
In overlap_neuralpe.py extract_overlapping_signals():

# Change from:
if training and true_params is not None:
    # Store experience

# Change to:
# Option A: Always collect RL data, but weight reward differently
if len(self.rl_controller.memory) < self.rl_controller.memory.maxlen // 2:
    # During warm-up phase, collect data regardless of training
    if true_params is not None:
        reward = self._compute_extraction_reward(params_corrected, true_params_iter)
    else:
        # Inference mode: use residual power as proxy reward
        reward = 1.0 - torch.mean(residual_data**2).item()
    
    # Always store during initial phase
    self.rl_controller.store_experience(...)
    self.rl_controller.train_step()
    
# Option B: Keep separate reward for inference
elif training and true_params is not None:
    # Training mode with ground truth
    reward = self._compute_extraction_reward(...)
    self.rl_controller.store_experience(...)
    self.rl_controller.train_step()
else:
    # Inference mode: still collect data for adaptation
    reward = 1.0 - torch.mean(residual_data**2).item()
    self.rl_controller.store_experience(...)
    if len(self.rl_controller.memory) >= self.rl_controller.batch_size:
        self.rl_controller.train_step()
""")
