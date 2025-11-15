"""
Reinforcement learning controller for adaptive model complexity.
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
import random


class DQNController(nn.Module):
  """Deep Q-Network for complexity control."""
  
  def __init__(self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 64]):
    super().__init__()
    
    self.state_dim = state_dim
    self.action_dim = action_dim
    
    # Build network
    layers = []
    input_dim = state_dim
    
    for hidden_dim in hidden_dims:
      layers.extend([
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1)
      ])
      input_dim = hidden_dim
    
    layers.append(nn.Linear(input_dim, action_dim))
    
    self.network = nn.Sequential(*layers)
    
  def forward(self, state: torch.Tensor) -> torch.Tensor:
    """Forward pass."""
    return self.network(state)


class AdaptiveComplexityController(nn.Module):
  """RL-based adaptive complexity controller."""
  
  def __init__(self,
        state_features: List[str],
        complexity_levels: List[str] = ["low", "medium", "high"],
        learning_rate: float = 1e-3,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32):
    super().__init__()
    
    self.state_features = state_features
    self.complexity_levels = complexity_levels
    self.state_dim = len(state_features)
    self.action_dim = len(complexity_levels)
    
    # RL parameters
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.batch_size = batch_size
    self.gamma = 0.95 # Discount factor
    
    # Networks
    self.q_network = DQNController(self.state_dim, self.action_dim)
    self.target_network = DQNController(self.state_dim, self.action_dim)
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
    
    # Experience replay
    self.memory = deque(maxlen=memory_size)
    
    # Metrics tracking
    self.action_history = deque(maxlen=100)  # Last 100 actions
    self.reward_history = deque(maxlen=100)  # Last 100 rewards
    self.action_counts = {i: 0 for i in range(self.action_dim)}  # Action frequency
    
    # Update target network
    self.update_target_network()
    
    # Logging
    self.logger = logging.getLogger(__name__)
    
  def get_state_vector(self, pipeline_state: Dict) -> torch.Tensor:
    """Convert pipeline state to feature vector."""
    
    state_values = []
    
    for feature in self.state_features:
      if feature == "remaining_signals":
        state_values.append(pipeline_state.get("remaining_signals", 0))
      elif feature == "residual_power":
        state_values.append(pipeline_state.get("residual_power", 0.0))
      elif feature == "processing_time":
        state_values.append(pipeline_state.get("processing_time", 0.0))
      elif feature == "current_snr":
        state_values.append(pipeline_state.get("current_snr", 0.0))
      elif feature == "extraction_success_rate":
        state_values.append(pipeline_state.get("extraction_success_rate", 0.0))
      else:
        state_values.append(0.0) # Default value
    
    return torch.tensor(state_values, dtype=torch.float32)
  
  def select_action(self, state: torch.Tensor, training: bool = True) -> int:
    """Select action using epsilon-greedy policy."""
    
    if training and np.random.random() < self.epsilon:
      # Random action
      return np.random.randint(self.action_dim)
    else:
      # Greedy action
      with torch.no_grad():
        q_values = self.q_network(state.unsqueeze(0))
        return q_values.argmax().item()
  
  def get_complexity_level(self, pipeline_state: Dict, training: bool = True) -> str:
    """Get complexity level recommendation."""
    
    state_vector = self.get_state_vector(pipeline_state)
    action = self.select_action(state_vector, training)
    return self.complexity_levels[action]
  
  def store_experience(self,
             state: torch.Tensor,
             action: int,
             reward: float,
             next_state: torch.Tensor,
             done: bool):
     """Store experience in replay buffer."""
     
     self.memory.append((state, action, reward, next_state, done))
     
     # Track metrics
     self.action_history.append(action)
     self.reward_history.append(reward)
     self.action_counts[action] += 1
  
  def compute_reward(self,
           pipeline_metrics: Dict,
           complexity_level: str) -> float:
    """Compute reward based on pipeline performance."""
    
    # Base reward components
    accuracy_reward = 0.0
    speed_reward = 0.0
    complexity_penalty = 0.0
    
    # Accuracy component (higher is better)
    if "parameter_bias" in pipeline_metrics:
      bias = pipeline_metrics["parameter_bias"]
      accuracy_reward = max(0, 1.0 - bias) # Reward for low bias
    
    # Speed component (faster is better)
    if "extraction_time" in pipeline_metrics:
      time_taken = pipeline_metrics["extraction_time"]
      speed_reward = max(0, 1.0 - time_taken / 60.0) # Normalize by 1 minute
    
    # Complexity penalty (lower complexity is better for similar performance)
    complexity_costs = {"low": 0.0, "medium": 0.1, "high": 0.2}
    complexity_penalty = complexity_costs.get(complexity_level, 0.0)
    
    # Combined reward
    total_reward = accuracy_reward + speed_reward - complexity_penalty
    
    return float(total_reward)
  
  def train_step(self) -> Optional[float]:
    """Perform one training step."""
    
    if len(self.memory) < self.batch_size:
      return None
    
    # Sample batch from memory
    batch = random.sample(self.memory, self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Convert to tensors
    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.stack(next_states)
    dones = torch.tensor(dones, dtype=torch.bool)
    
    # Current Q values
    current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
    
    # Next Q values from target network
    with torch.no_grad():
      next_q_values = self.target_network(next_states).max(1)[0]
      target_q_values = rewards + (self.gamma * next_q_values * ~dones)
    
    # Compute loss
    loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
    
    # Optimize
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
    self.optimizer.step()
    
    # Decay epsilon
    self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
    
    return loss.item()
  
  def update_target_network(self):
     """Update target network with main network weights."""
     self.target_network.load_state_dict(self.q_network.state_dict())
  
  def get_metrics(self) -> Dict[str, float]:
     """Get current RL controller metrics for monitoring."""
     
     metrics = {}
     
     # Epsilon (exploration rate)
     metrics["epsilon"] = float(self.epsilon)
     
     # Complexity distribution from recent actions
     if self.action_history:
       action_array = np.array(list(self.action_history))
       metrics["avg_complexity"] = float(np.mean(action_array))
       metrics["complexity_std"] = float(np.std(action_array))
     else:
       metrics["avg_complexity"] = 0.0
       metrics["complexity_std"] = 0.0
     
     # Rewards
     if self.reward_history:
       reward_array = np.array(list(self.reward_history))
       metrics["avg_reward"] = float(np.mean(reward_array))
       metrics["total_reward"] = float(np.sum(reward_array))
     else:
       metrics["avg_reward"] = 0.0
       metrics["total_reward"] = 0.0
     
     # Action entropy (measures exploration diversity)
     counts = np.array(list(self.action_counts.values()))
     if counts.sum() > 0:
       probs = counts / counts.sum()
       entropy = -np.sum(probs * np.log(probs + 1e-10))
       metrics["action_entropy"] = float(entropy)
     else:
       metrics["action_entropy"] = 0.0
     
     # Memory size
     metrics["memory_size"] = len(self.memory)
     
     return metrics
  
  def state_dict(self, destination=None, prefix='', keep_vars=False):
    """Override state_dict to include non-tensor attributes."""
    # Get module state dict
    module_state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
    
    # Add non-tensor attributes as metadata
    state = {
      'module_state': module_state,
      'epsilon': self.epsilon,
      'epsilon_decay': self.epsilon_decay,
      'state_features': self.state_features,
      'complexity_levels': self.complexity_levels,
    }
    return state
  
  def load_state_dict(self, state_dict, strict=True):
    """Override load_state_dict to restore non-tensor attributes."""
    if isinstance(state_dict, dict) and 'module_state' in state_dict:
      # New format with metadata
      module_state = state_dict['module_state']
      super().load_state_dict(module_state, strict=strict)
      
      self.epsilon = state_dict.get('epsilon', self.epsilon)
      self.epsilon_decay = state_dict.get('epsilon_decay', self.epsilon_decay)
    else:
      # Fallback: assume it's pure module state
      super().load_state_dict(state_dict, strict=strict)
  
  def save_model(self, filepath: str):
    """Save controller model."""
    torch.save({
      'q_network_state_dict': self.q_network.state_dict(),
      'target_network_state_dict': self.target_network.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'epsilon': self.epsilon,
      'state_features': self.state_features,
      'complexity_levels': self.complexity_levels
    }, filepath)
    
    self.logger.info(f"RL controller saved to {filepath}")
  
  def load_model(self, filepath: str):
    """Load controller model."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.epsilon = checkpoint['epsilon']
    
    self.logger.info(f"RL controller loaded from {filepath}")