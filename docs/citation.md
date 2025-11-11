Mass ratio of binary black holes determined from LIGO/Virgo data restricted to small false alarm rate Open Access
Tomoya Kinugawa, Takashi Nakamura, Hiroyuki Nakano

Suggested Citation Text 📄
For METHODS Section:

"Our training dataset exhibits moderate distance-SNR correlation for BBH systems (r = -0.586), consistent with the amplitude ∝ 1/distance relationship and well-documented GW selection effects. This selection function reflects realistic LIGO O3 detection sensitivity."​

For DISCUSSION Section (where this paper fits):

"Recent observational studies demonstrate that mass and redshift distributions are fundamentally coupled to detector sensitivity [this paper, 13, 18], with predictions that higher-redshift observations will exhibit different selection patterns. Extending our framework to next-generation detectors (ET, CE) will require accounting for these evolving population properties."


Raymond, V. (2025). Data for Simulation-based Inference for Gravitational-waves from Intermediate-Mass Binary Black Holes in Real Noise [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16752757




# ========================= AHSD: Adaptive Hierarchical Signal Decomposition =========================
# Complete PriorityNet training configuration with all validated parameters
# Updated: 2025-11-10

experiment_name: "AHSD_Phase2_PriorityNet"
random_seed: 420
device: "cpu"  # auto, cuda, cpu

# ==============================================================================
# PRIORITY NET CONFIGURATION (Core model parameters)
# ==============================================================================
priority_net:
  
  # --- ARCHITECTURE ---
  hidden_dims: [512, 384, 256, 128]           # ✅ REDUCED from [640, 512, 384, 256]
  dropout: 0.20                           # Dropout in priority head
  
  # --- MODEL FLAGS ---
  use_strain: true                       # Enable temporal strain encoder
  use_edge_conditioning: true            # Enable edge-type embedding
  n_edge_types: 19                        # Number of edge case types
  use_transformer_encoder: true          # Disable Transformer (too slow on CPU, 4.8s/batch→0.8s/batch)
  
  # --- OPTIMIZER ---
  optimizer: "AdamW"
  learning_rate: 1.0e-4
  weight_decay: 2.0e-5
  
  # --- TRAINING SCHEDULE ---
  batch_size: 32
  epochs: 250
  patience: 20                            # Early stopping patience (was 1, too strict)
  
  # --- WARMUP ---
  warmup_epochs: 15
  warmup_start_factor: 0.03
  
  # --- LEARNING RATE SCHEDULER ---
  scheduler: "ReduceLROnPlateau"
  scheduler_patience: 5
  scheduler_factor: 0.5
  min_lr: 1.0e-6
  
  # --- LOSS FUNCTION ---
  ranking_weight: 0.75                      # Ranking loss weight
  mse_weight: 0.20                        # MSE loss weight
  uncertainty_weight: 0.05                # Uncertainty loss weight
  use_snr_weighting: true                 # Weight losses by SNR
  loss_scale_factor: 0.001                # Scale factor for loss terms
  
  # --- GRADIENT MANAGEMENT ---
  gradient_clip_norm: 2.0                # Gradient clipping threshold
  gradient_log_threshold: 0.5             # Threshold for gradient norm logging
  
  # --- LABEL SMOOTHING ---
  label_smoothing: 0.0
  
  # --- ATTENTION/MODAL FUSION ---
  use_modal_fusion: true                 # Enable multi-modal attention fusion
  attention_num_heads: 4
  attention_dropout: 0.1
  
  # --- OVERLAP HANDLING ---
  overlap_use_attention: true            # Use attention for overlap encoding
  overlap_importance_hidden: 16           # Hidden dim for importance network


# ==============================================================================
# NEURAL POSTERIOR ESTIMATION (Flow-based parameter inference)
# ==============================================================================
neural_posterior:
  
  # --- PARAMETERS (9-dimensional parameter space) ---
  param_names:
    - mass_1                              # Primary mass
    - mass_2                              # Secondary mass
    - luminosity_distance                 # Distance to source
    - ra                                  # Right ascension
    - dec                                 # Declination
    - theta_jn                            # Inclination angle
    - psi                                 # Polarization angle
    - phase                               # Coalescence phase
    - geocent_time                        # Time at geocenter
  
  # --- ARCHITECTURE ---
  context_dim: 256                        # Context embedding dimension
  n_flow_layers: 14                       # Number of normalizing flow layers
  max_iterations: 5                       # Max adaptive refinement iterations
  
  # --- NORMALIZING FLOW ---
  flow_config:
    type: "realnvp"                       # realnvp or maf
    hidden_features: 256                  # Hidden features per flow block
    num_blocks_per_layer: 2               # Coupling blocks per layer
    dropout: 0.02                         # Minimal dropout for flow stability
  
  # --- TRAINING ---
  learning_rate: 3.0e-4
  batch_size: 64                          # Flow training batch size
  epochs: 40                              # Training epochs
  patience: 10
  weight_decay: 1.0e-5
  gradient_clip: 5.0
  optimizer: "AdamW"
  
  # --- LEARNING RATE SCHEDULER ---
  scheduler: "ReduceLROnPlateau"
  scheduler_patience: 8
  scheduler_factor: 0.6
  min_lr: 1.0e-6
  
  # --- DATA AUGMENTATION ---
  data_augmentation:
    enabled: true
    noise_scaling: [0.98, 1.02]          # Noise level variation
    time_shifts: [-0.002, 0.002]         # Time shift range (seconds)
    apply_probability: 0.3                # Probability of augmentation


# ==============================================================================
# ADAPTIVE SUBTRACTION (Signal removal for overlaps)
# ==============================================================================
adaptive_subtractor:
  complexity_level: "medium"              # low, medium, high
  waveform_approximant: "IMRPhenomPv2"   # Waveform model
  post_newtonian_order: "3.5PN"          # PN order for waveforms
  
  # --- PHYSICS ---
  spin_effects: true                      # Include spin effects
  tidal_effects: true                     # Include tidal effects
  
  # --- SUBTRACTION STRENGTH ---
  base_strength: 0.1                      # Base subtraction strength
  max_strength: 0.8                       # Maximum strength
  uncertainty_threshold: 0.3              # Uncertainty threshold for adaptation
  match_threshold: 0.7                    # Match threshold for acceptance


# ==============================================================================
# REINFORCEMENT LEARNING CONTROLLER (Adaptive complexity)
# ==============================================================================
rl_controller:
  enabled: true
  
  # --- STATE ---
  state_features:
    - remaining_signals
    - residual_power
    - current_snr
    - extraction_success_rate
  
  # --- COMPLEXITY LEVELS ---
  complexity_levels: ["low", "medium", "high"]
  complexity_configs:
    low:
      flow_layers: 4
      inference_samples: 500
    medium:
      flow_layers: 8
      inference_samples: 1000
    high:
      flow_layers: 14
      inference_samples: 2000
  
  # --- LEARNING ---
  learning_rate: 1.0e-3
  epsilon: 0.1                            # Exploration rate
  epsilon_decay: 0.995
  memory_size: 10000
  batch_size: 32


# ==============================================================================
# BIAS CORRECTION NETWORK (Systematic error correction)
# ==============================================================================
bias_corrector:
  enabled: true
  
  # --- ARCHITECTURE ---
  hidden_dims: [256, 128, 64]
  context_dim: 16
  dropout: 0.10
  
  # --- TRAINING ---
  learning_rate: 1.0e-4
  batch_size: 64
  epochs: 15
  patience: 8


# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================
data:
  # --- SAMPLING ---
  sample_rate: 4096                       # Hz
  segment_duration: 4.0                   # seconds
  
  # --- FREQUENCY RANGE ---
  f_low: 20.0                             # Hz (low frequency cutoff)
  f_high: 1024.0                          # Hz (high frequency cutoff)
  
  # --- SPLITS ---
  validation_split: 0.15                  # Validation fraction
  test_split: 0.05                        # Test fraction
  
  # --- SIGNAL DISTRIBUTION ---
  signal_distribution:
    BBH: 0.70                             # Binary Black Hole
    BNS: 0.20                             # Binary Neutron Star
    NSBH: 0.10                            # Neutron Star - Black Hole


# ==============================================================================
# MONITORING & OUTPUT
# ==============================================================================
monitoring:
  save_frequency: 5                       # Save checkpoint every N epochs
  log_frequency: 1                        # Log metrics every N epochs
  early_stopping: true
  
  # --- METRICS TO TRACK ---
  metrics:
    - nll_loss
    - parameter_accuracy
    - extraction_efficiency
    - train_val_gap
    - gradient_norm
  
  # --- RL MONITORING ---
  rl_monitoring:
    track_complexity_changes: true
    log_rl_rewards: true
    complexity_distribution: true
  
  # --- BIAS CORRECTION MONITORING ---
  bias_monitoring:
    track_correction_effectiveness: true
    log_physics_violations: true

output:
  save_best_only: true                    # Save only best checkpoint
  save_intermediate: true                 # Save intermediate checkpoints
  generate_plots: true                    # Generate training plots


# ==============================================================================
# VALIDATION FLAGS (True/False Summary)
# ==============================================================================
# These flags are logged during training startup for debugging:
#
# PRIORITY NET:
#  ✓ use_strain: true
#  ✓ use_edge_conditioning: true
#  ✓ use_transformer_encoder: true
#  ✓ use_snr_weighting: true
#  ✓ use_modal_fusion: false (disabled - use attention fusion disabled)
#  ✓ overlap_use_attention: false (disabled - standard overlap encoding)
#  ✓ early_stopping: true
#  ✓ label_smoothing: 0.02
#
# NEURAL POSTERIOR:
#  ✓ data_augmentation.enabled: true
#
# ADAPTIVE SUBTRACTION:
#  ✓ spin_effects: true
#  ✓ tidal_effects: true
#
# RL CONTROLLER:
#  ✓ enabled: true
#  ✓ track_complexity_changes: true
#  ✓ log_rl_rewards: true
#
# BIAS CORRECTOR:
#  ✓ enabled: true
#  ✓ track_correction_effectiveness: true
#  ✓ log_physics_violations: true
#
# OUTPUT:
#  ✓ save_best_only: true
#  ✓ save_intermediate: true
#  ✓ generate_plots: true
