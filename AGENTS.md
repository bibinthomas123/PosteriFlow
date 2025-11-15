# Agent Guidelines for PosteriFlow

## Project Overview
PosteriFlow is a gravitational wave astronomy pipeline implementing Adaptive Hierarchical Signal Decomposition (AHSD). It processes overlapping gravitational wave signals using neural posterior estimation and adaptive subtraction techniques to detect and analyze multiple concurrent events.

**Key Components:**
- PriorityNet: Core neural network for signal prioritization and detection
- Neural posterior estimation for parameter inference
- Adaptive signal subtraction for handling overlapping sources
- Real-time analysis pipeline for LIGO/Virgo data

## Environment Setup

**IMPORTANT: The conda environment 'ahsd' already exists. NEVER recreate it.**

- `conda init` - init the conda 
- `conda activate ahsd` - Activate existing environment (always use this)
- `pip install -e . --no-deps` - Install/update package in development mode

**IMPORTANT** Always run `pip install -e . --no-deps` after making changes to the codebase.

If dependencies need updating, use `conda install <package>` or `pip install <package>` within the activated environment.

Do not create a new Document every time just read the old doc and update it so that we can keep a track in the end 

## Build/Lint/Test Commands

**Testing:**
- `pytest` - Run all tests
- `pytest tests/test_file.py::TestClass::test_method` - Run specific test
- `pytest --cov=ahsd --cov-report=html` - Tests with coverage report
- `pytest -v -s` - Verbose output with print statements

**Code Quality:**
- `black .` - Format code (100 char line length)
- `isort .` - Sort imports (black profile)
- `flake8 .` - Lint code
- `mypy .` - Type checking (configured for partial checking)

**Entry Points:**
- `ahsd-generate` - Generate training/validation datasets
- `ahsd-validate` - Validate model performance
- `ahsd-train` - Train models
- `ahsd-test` - Run inference and evaluation

**New Tools (Nov 2025):**
- **TransformerStrainEncoder (✅ VERIFIED WORKING - Nov 13):**
  - `python validate_transformer_encoder.py` - Validate TransformerStrainEncoder implementation
  - `pytest tests/test_transformer_encoder_enhanced.py -v` - Run encoder tests
  - `python scripts/benchmark_encoder.py --iterations 100` - Benchmark encoder performance
  - `python scripts/benchmark_encoder.py --amp --masks --iterations 100` - Full benchmark with AMP/masks
  - **Health Check (NEW):**
    - `python check_transformer_health.py` - Comprehensive health check (forward pass, gradients, dimensions)
    - `python test_transformer_training.py` - Full training simulation with detailed logging
  - **Enable in training:** Set `use_transformer_encoder: true` in config YAML
  - **Logging:** Use DEBUG level to see detailed transformer execution traces
  - **Status:** ✅ Working correctly - forward pass, gradient flow, loss computation all verified
  - **Checkpoint Loading (Nov 13 FIXED):** Transformer-trained checkpoints now load perfectly (strict=True, perfect match)

- **Neural Noise Integration (10,000× speedup):**
  - `python test_neural_noise_integration.py` - Validate neural noise generation (expects ✓ PASS)

- **Real Noise Cache Integration (10-25× speedup):**
   - Pre-downloaded GWOSC segments from `gw_segments_cleaned/` folder
   - 133 real noise segments (H1: 59, L1: 58, V1: 16) automatically loaded at startup
   - Set `use_real_noise_prob: 0.1` in `configs/data_config.yaml` to enable (10% real noise)

- **Neural Spline Flow (NSF) - Nov 14, 2025 (✅ RECOMMENDED):**
   - State-of-the-art posterior flow: 3 days → 0.8 seconds inference on 9D space
   - `python experiments/phase3a_neural_pe.py --epochs 50 --device cuda` - Train neural PE with NSF
   - Set `flow_type: "nsf"` in configs/enhanced_training.yaml (✅ Already set as default)
   - NSF uses monotonic rational quadratic splines (invertible by construction, no ODE approximation)
   - Alternative flows: "flowmatching" (ODE-based), "realnvp", "maf"
   - Expected NLL convergence: 4-5 bits by epoch 10 (vs FlowMatching 8+ bits plateau)
   - **Flow Loss Stuck at 0.1 - CRITICAL FIX (Nov 14 23:50):**
     - **Issue**: Loss plateaued at 0.1000, gradients vanished (0.010), no learning possible
     - **Root cause**: NLL loss clamped at max=-0.1, forcing loss floor at 0.1 (perfectly flat landscape)
     - **Fix**: Removed upper clamp from log_prob - let NLL loss be natural (2-10 nats)
     - **Code change**: `overlap_neuralpe.py` line 874: Removed `max=-0.1` from torch.clamp()
     - **Gradient handling**: Increased gradient_clip 1.0 → 5.0 to handle natural loss spikes without suppressing learning
     - **Result**: Loss now decreases naturally, gradients flow properly, convergence resumes

- **Config Loading for Neural PE (✅ FIXED - Nov 14):**
    - **Issue**: Trainer was reading hard-coded defaults instead of YAML values; loss computation using wrong flow_type
    - **Root cause**: YAML has `neural_posterior:` section, but code was reading top-level config; Multiple places read flow_type from `flow_config.get("type")` instead of top-level `flow_type`
    - **Fixes**:
      1. Config extraction (phase3a_neural_pe.py line 831): Extract `neural_posterior` section from YAML
      2. Flow initialization (overlap_neuralpe.py lines 280-283): Read `flow_type` from top-level config
      3. Loss computation (overlap_neuralpe.py lines 806-807): Read `flow_type` for correct loss selection (CFM vs NLL)
      4. Logging/metadata (overlap_neuralpe.py line 1285): Read `flow_type` from top-level config
    - **Verification**: All 22 critical parameters load correctly ✅
      - Trainer: learning_rate=1e-5, batch_size=64, epochs=50, patience=15, gradient_clip=5.0
      - Flow: flow_type=nsf, context_dim=768, num_layers=8, hidden_features=256, tail_bound=3.0
      - Loss: physics_loss_weight=0.0, bounds_penalty_weight=0.5, sample_loss_weight=0.1
      - Loss computation: flow_type=nsf → NLL loss (no velocity_net AttributeError) ✅

- **RL-based Adaptive Complexity Controller (✅ INTEGRATED - Nov 15):**
   - **Purpose**: Dynamically adjust signal processing complexity based on data characteristics (remaining signals, residual power, SNR, success rate)
   - **DQN-based control**: Epsilon-greedy policy with experience replay for optimal complexity selection
   - **Integration in InferencePipeline**: 
     - `src/ahsd/inference/inference_pipeline.py` - Unified inference API with RL adaptation
     - `src/ahsd/models/rl_controller.py` - Core RL controller (DQNController + AdaptiveComplexityController)
   - **Usage**:
     - `python src/ahsd/inference/inference_pipeline.py --use-rl --rl-controller models/rl_controller.pt`
     - `pipeline = InferencePipeline(model_path, config_path, inference_config=InferenceConfig(use_rl_controller=True))`
   - **Features**:
     - State: remaining_signals, residual_power, processing_time, current_snr, extraction_success_rate
     - Actions: low/medium/high complexity levels
     - Rewards: accuracy_reward (1-bias) + speed_reward (1-time/60) - complexity_penalty
     - Metrics: epsilon (exploration), avg_complexity, avg_reward, action_entropy, memory_size
   - **API Methods**:
     - `extract(use_rl_adaptation=True)` - Extract with RL-controlled complexity
     - `get_rl_metrics()` - Monitor RL controller learning
     - `save_rl_controller(filepath)` - Save trained controller
   - **Output Fields**:
     - `rl_complexity_level`: "low", "medium", or "high" recommendation
     - `rl_pipeline_state`: State dict used for decision
     - `rl_metrics`: Controller performance metrics
     - `refined`: Boolean indicating if high-complexity refinement applied


## Architecture & Codebase Structure

**Directory Layout:**
src/ahsd/
├── core/ # PriorityNet and core detection logic
├── data/ # Dataset classes, data loading, preprocessing
├── models/ # Neural network architectures
├── utils/ # Configuration, logging, helper functions
└── evaluation/ # Metrics, plotting, analysis tools

experiments/ # Training/inference scripts and notebooks
configs/ # YAML configuration files
tests/ # Unit and integration tests
data/ # Generated datasets (not in git)
models/ # Model checkpoints (not in git)
outputs/ # Experiment results (not in git)

text

**Key Dependencies:**
- Deep Learning: PyTorch, NumPy, SciPy
- GW Analysis: PyCBC, GWpy, Bilby
- Data: Pandas, h5py, HDF5
- Dev Tools: pytest, black, mypy, isort, flake8

## Code Style Guidelines

**Formatting:**
- Black formatter with 100 character line limit
- Unix LF line endings
- isort with black profile for import sorting

**Type Hints:**
- Always use type hints for function signatures
- Use `from __future__ import annotations` for forward references
- Mypy configuration allows missing imports but warns on missing return types

**Import Organization:**
1. Standard library imports
2. Third-party imports (PyTorch, NumPy, PyCBC, etc.)
3. Local package imports
- Prefer absolute imports over relative
- Group related imports together

**Naming Conventions:**
- Classes: `PascalCase` (e.g., `PriorityNet`, `WaveformDataset`)
- Functions/methods: `snake_case` (e.g., `train_model`, `compute_snr`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_EPOCHS`, `DEFAULT_SAMPLE_RATE`)
- Modules: `snake_case` (e.g., `priority_net.py`, `data_utils.py`)
- Private members: prefix with `_` (e.g., `_internal_helper`)

**Error Handling:**
- Use specific exception types, not bare `except:`
- Log errors with context using Python logging
- Validate inputs early in functions
- Raise `ValueError` for invalid parameters, `RuntimeError` for runtime issues

**Configuration:**
- All experiments use YAML config files in `configs/`
- Load configs through `utils/config.py`
- Document all config parameters with comments
- Never hardcode hyperparameters in code

## Testing Instructions

Only when asked to test please follow the below conditions 

**Test Structure:**
- Unit tests for individual functions/classes
- Integration tests for end-to-end workflows
- Use pytest fixtures for common setup (e.g., mock waveforms, datasets)
- Test files mirror source structure: `tests/test_<module>.py`

**What to Test:**
- Data preprocessing and augmentation
- Model forward passes with known inputs
- Loss functions and metrics
- Config loading and validation
- Edge cases: empty data, boundary conditions, invalid inputs

**Mocking:**
- Mock expensive operations (waveform generation, model training)
- Use `pytest.fixture` for reusable test data
- Mock external dependencies (PyCBC, GWOSC data fetching)

## Common Pitfalls & Best Practices


**Dont create documents for explaining it untill you make any changes to code**
**DONT TRAIN THE MODEL I WILL DO IT AND SHARE THE OUTPUT**
**Data Handling:**
- Always check for NaN/Inf in tensors before training
- Verify dataset serialization includes all required fields (e.g., `network_snr`)
- Scale SNR values appropriately for downstream targets
- Watch for embedding padding issues and near-zero std
- **Distance-SNR Correlation** (FIXED - Nov 2025): Ensures strong negative correlation between distance and SNR:
  - Distance is now derived directly from target_snr using chirp mass scaling: `d = d_ref * (M_c/M_c_ref)^(5/6) * (SNR_ref/target_SNR)`
  - Reference parameters in ParameterSampler: `reference_snr=35`, `reference_distance=400 Mpc`, `reference_mass=30 M_sun`
  - **Jitter removed** to preserve correlation fidelity (was weakening tight SNR-distance relationship)
  - **Non-edge samples (94% of dataset)** show strong correlations: BBH r≈-0.79, BNS r≈-0.87, NSBH r≈-0.67 ✓
  - Edge cases (6% of dataset) intentionally modify parameters for training robustness, lowering overall correlation to BBH r≈-0.60, BNS r≈-0.23, NSBH r≈-0.21
  - The `attach_network_snr()` function uses priority order:
    1. Already-sampled `target_snr` (from ParameterSampler)
    2. Per-detector matched-filter SNRs  
    3. Proxy formula (mass/distance-based)
  - Always filter `is_edge_case==False` when evaluating physics correctness checks
- **Mass-Distance Correlation** (FIXED - Nov 2025): Physics-aware correlation for BBH/BNS, mass-agnostic SNR for NSBH:
  - **BBH**: Mass distribution widened sigma=0.30-0.32 (was 0.20-0.25), clipping 8.0-60.0 Msun → r≈0.38-0.39 ✓
  - **BNS**: Narrow mass range (1.0-2.5 M☉) naturally creates minimal correlation → r≈0.25 ✓
  - **NSBH**: Mass-aware SNR adjustment (light BH: baseline, medium: +25%, heavy: +55%) decouples mass from distance → r≈0.32 ✓
  - Rationale: NSBH BH mass diversity would create spurious distance correlation via chirp mass scaling, so target_snr is mass-adjusted to keep distances uniform
  - SNR-distance anticorrelation maintained: BBH r≈-0.79, BNS r≈-0.89, NSBH r≈-0.62 (all strong)
- **Missing Noise Data in Samples** (FIXED - Nov 10, 2025): Original noise arrays are now stored in samples:
  - Issue: Noise was generated and used for signal injection but discarded after combining with signal
  - Fix: Added `"noise"` field to `detector_data[detector_name]` dictionaries in 8 sample generation methods
  - Implementation: Store noise before preprocessing using `detector_data[detector_name] = {"noise": noise.astype(np.float32), ...}`
  - Affected methods: `_generate_single_sample`, `_generate_overlapping_sample`, `_generate_psd_drift_sample`, `_generate_sky_position_extreme_sample`, `_generate_pre_merger_sample`, `_generate_sample_from_params`, `_generate_partial_overlap_sample`
  - Verification: All samples now include noise for H1, L1, V1 detectors (shape: 16384 float32 @ 4096 Hz, 4s duration)

- **PriorityNet Dimension Mismatch** (FIXED - Nov 10, 2025): Fixed forward pass shape errors:
- Issue: "mat1 and mat2 shapes cannot be multiplied (5x16 and 15x640)" errors during training
- Root cause: (1) CrossSignalAnalyzer hardcoded dimension in importance_net, (2) SignalFeatureExtractor expecting 15 dims but receiving 16 (network_snr added)
- Fix: (1) Changed `nn.Linear(16, 1)` to `nn.Linear(importance_hidden_dim, 1)` in CrossSignalAnalyzer.importance_net, (2) Updated SignalFeatureExtractor default `input_dim` from 15 to 16
- Verification: Forward pass now succeeds with 5 signals, 16 features

- **Neural Noise Model Path Resolution** (FIXED - Nov 10, 2025): Auto-resolve relative paths to project root:
    - Issue: Neural noise models not loading - "No model path provided" message even with valid config
    - Root cause: Relative paths in config (e.g., `"data/Gaussian_network.pickle"`) not resolved to absolute paths
    - Fix: Added automatic path resolution in `dataset_generator.py` (lines 276-303) that finds project root via `.git/` directory
    - Works from any working directory - paths automatically resolved relative to project root
    - Graceful fallback to colored Gaussian noise if models unavailable or sbigw missing
    - No config changes needed - existing YAML config works transparently

- **Real Noise Cache Integration** (FIXED - Nov 11, 2025): Pre-downloaded GWOSC segments for 10-25× speedup:
    - Pre-downloaded segments stored in `gw_segments_cleaned/` folder (H1: 59, L1: 58, V1: 16 segments)
    - Loaded automatically at dataset generator startup via `_load_cached_noise_segments()`
    - Three-level priority: cached segments → on-demand fetching → synthetic noise
    - Set `use_real_noise_prob: 0.1` in config to enable (10% of samples use real noise)
    - Backward compatible: gracefully falls back if cache directory doesn't exist
    - Memory efficient: ~21.5 MB total for all 133 segments
- **PriorityNet Edge Conditioning & Calibration** (FIXED - Nov 12, 2025): Validation dataset and loss weight tuning:
     - **Edge ID Issue**: Validation set was generated with `create_overlaps=False`, causing all samples to have edge_type_id=0 (variance=0)
     - **Fix**: Updated `train_priority_net.py` lines 2564-2569 to pass `create_overlaps=args.create_overlaps` to validation/test loaders
     - **Calibration Issue**: Model predictions max out at 0.557 vs true max 0.950 (gap=0.393) due to ranking loss dominance
     - **Loss Rebalancing** in `configs/enhanced_training.yaml`: ranking=0.50 (↓0.70), mse=0.35 (↑0.20), uncertainty=0.15 (↑0.10)
     - **Expected Results**: Edge variance >5, max gap <0.10, uncertainty correlation >0.30, distance sensitivity <-0.01
     - **Retraining**: Run with `--create_overlaps` flag for proper multi-detection validation
- **LR Scheduler Patience Reset Bug** (FIXED - Nov 12, 2025): ReduceLROnPlateau spurious counter resets:
     - **Issue**: `num_bad_epochs` counter reset to 0 without reducing LR, causing monitoring errors
     - **Root cause**: `threshold_mode='abs'` with very small losses (~1e-3) caused floating-point precision errors in PyTorch's comparison logic
     - **Fix**: Changed to `threshold_mode='rel'` (relative mode) in `src/ahsd/core/priority_net.py` lines 1228-1237
     - **Changes**: `threshold=1e-4, threshold_mode='abs'` → `threshold=1e-3, threshold_mode='rel'` (0.1% relative improvement threshold)
     - **Why**: Relative comparison `(best - current) / abs(best) > threshold` is numerically stable vs direct subtraction of tiny numbers
     - **Verification**: `num_bad_epochs` now increments consistently without spurious resets
- **Checkpoint Encoder Type Mismatch** (FIXED - Nov 12, 2025): Config nesting issue in checkpoint validation + PriorityNet config reading:
      - **Issue**: Spurious "Encoder type mismatch: checkpoint=True, config=False" during training resume; state_dict shape mismatches (missing CNN conv_blocks, unexpected Transformer encoder layers)
      - **Root cause**: Config loader returns nested dict with `priority_net` top-level key, but (1) checkpoint loader only checked top level, (2) PriorityNet's `cfg_get()` also only checked top level, so even after validation passed, model initialized with wrong encoder
      - **Fix**: (1) Updated `load_checkpoint()` in `experiments/train_priority_net.py` lines 2195-2210 to search both top level and nested `priority_net` section, (2) Updated `cfg_get()` in `src/ahsd/core/priority_net.py` lines 760-777 to search both levels when reading config
      - **Behavior**: Checkpoint validation passes when types match; PriorityNet correctly initializes TransformerStrainEncoder when `use_transformer_encoder: true`
      - **Verification**: Checkpoints resume without warnings, encoder type matches config, state_dict loads cleanly

- **Prediction Compression/Saturation** (FIXED - Nov 12, 2025): Output range severely limited to 25% of target range:
       - **Issue**: Predictions stuck in (0.297, 0.471) while targets span (0.263, 0.950) — compression ratio only 25%
       - **Root causes**: (1) Calibration penalties weak (0.05 each), (2) Output bias initialized to 0.2 not 0.5, (3) Weight init std too small (0.01 not 0.05), (4) No range regularization, (5) Affine params unconstrained
       - **Fix**: (1) Config-driven calibration weights (calib_mean_weight=0.15, calib_max_weight=0.40), (2) Output init from config (bias=0.5, std=0.05), (3) Range regularization loss term, (4) Affine param clamping (gain [0.7,1.5], bias [-0.1,0.1]), (5) Stronger penalties in loss function
       - **Config changes** in `configs/enhanced_training.yaml` (lines 59-71): 6 new parameters for calibration control
       - **Code changes** in `src/ahsd/core/priority_net.py`: 6 locations across PriorityLoss, PriorityNet, TrainerForPriorityNet
       - **Expected improvement**: Range 0.174 → ≥0.50 (287% increase), max_gap 0.687 → <0.10 (93% reduction)
       - **Timeline**: Epoch 30 target (prediction range ≥0.60, max_gap <0.10), full convergence by epoch 40-50

- **Uncertainty Calibration** (FIXED - Nov 13, 2025): Model uncertainty estimates not correlating with actual errors:
        - **Issue**: Block 5️⃣ failure - corr(|error|, unc)=0.096 (target: ≥0.15); uncertainty head undertrained
        - **Root causes**: (1) Weak loss weight (0.10), (2) Naive MSE-only loss, (3) Insufficient gradient flow (Softplus beta=1.0)
        - **Fix**: (1) Increased uncertainty_weight 0.10 → 0.35 (3.5x), (2) Two-part loss: MSE toward error + log-scale calibration, (3) Added bounds penalty [0.01, 0.50] to prevent collapse/explosion, (4) Increased Softplus beta 1.0 → 2.0
        - **Config changes** in `configs/enhanced_training.yaml` (line 48): uncertainty_weight=0.35; lines 51-53: new bounds/weight params
        - **Code changes** in `src/ahsd/core/priority_net.py`: PriorityLoss.__init__ (add bounds params), forward (two-part uncertainty loss + bounds penalty), PriorityNet (beta=2.0), PriorityNetTrainer (pass config params)
        - **Expected improvement**: corr(|error|, unc) → ≥0.20 by epoch 15-20; convergence by epoch 50
        - **Verification**: Run `python experiments/test_priority_net.py` → Block 5️⃣ should pass
        
- **Neural PE Output Denormalization** (FIXED - Nov 13, 2025): Posterior samples returned in normalized form instead of physical units:
         - **Issue**: Model outputs were in normalized range [-1, 1] instead of physical parameters (e.g., mass in Msun, distance in Mpc)
         - **Root cause**: `sample_posterior()` called `flow.inverse()` but didn't denormalize results. Comment claimed "flow trained on physical units" but actually trained on normalized params
         - **Fix**: Added `_denormalize_parameters()` call in `src/ahsd/models/overlap_neuralpe.py` lines 341-345
         - **Code changes**: (1) Line 342: renamed `samples_physical` → `samples_normalized`, (2) Line 345: added `samples_physical = self._denormalize_parameters(samples_normalized)`
         - **Test update**: Updated `test_overlap_neural_pe.py` lines 430-441 to use `sample_posterior()` API instead of calling `flow.inverse()` directly
         - **Results**: All 9 parameters now in physical units (e.g., mass_1: [-61, 162] Msun vs [-2.1, 2.2] before)
         - **Verification**: Run `python test_overlap_neural_pe.py --model_path models/neural_pe/best_model.pth --device cpu` → TEST 7 shows physical units

- **Geocent_time & Luminosity_distance Bounds Mismatch** (FIXED - Nov 13, 2025): Parameter bounds not matching actual data generated:
          - **Issue**: Physics loss penalty detected massive violations (70%+ of validation data) - validation loss 12.4× higher than training
          - **Root cause**: OverlapNeuralPE bounds were too restrictive: `geocent_time [-0.1, 0.1]s` vs actual data `[-1.77, 6.63]s`; `luminosity_distance [20, 8000]` Mpc vs actual `[15.9, 1170]` Mpc
          - **Dataset generator reality**: Edge case samples intentionally create out-of-bounds timing (lines 2674, 4351, 4522, 4662, 4665 in `dataset_generator.py`); line 4665 uses `i*1.5` spacing for overlapping signals
          - **Fix**: Updated bounds in `src/ahsd/models/overlap_neuralpe.py` lines 114-115:
             - `geocent_time: (-0.1, 0.1)` → `(-2.0, 8.0)` (covers 99th percentile 6.05s with safety margin for i*1.5 spacing)
             - `luminosity_distance: (20.0, 8000.0)` → `(10.0, 8000.0)` (allows rare nearby events)
          - **Verification**: All 9 parameters now within bounds: mass_1 [1.2, 73] ⊆ [1, 100], geocent_time [-1.77, 6.63] ⊆ [-2.0, 8.0], etc. ✓
          - **Impact**: Eliminates spurious physics penalties on valid edge case samples; training convergence improves; loss reflects real vs false violations
- **NLL Explosion (8.78→12.1 bits) - Physics Loss Dominance** (FIXED - Nov 13 09:55, 2025): Neural posterior NLL catastrophically high due to physics loss weight imbalance:
           - **Issue**: Train NLL = 12.1 bits (catastrophic, target 1-3 bits), Physics Loss = 5513.75 (99.8% of total loss!), Train-Val gap = 5517
           - **Root cause**: Nov 13 morning fix increased `physics_loss_weight: 0.2 → 1.0` as hard constraint. This backfired: physics loss became 1000× larger than NLL, giving optimizer zero incentive to minimize likelihood
           - **Previous intermediate fix** (Nov 13 early): Added sample_loss (0.1 weight) to constrain flow, but too weak vs hard physics constraint (1.0)
           - **Final fix** (Nov 13 09:55): Rebalanced all weights in `configs/enhanced_training.yaml` (lines 142-146):
              - `physics_loss_weight: 1.0 → 0.05` (soft guidance, not hard constraint)
              - `bounds_penalty_weight: 0.1 → 0.5` (strong protection for ground truth)
              - `sample_loss_weight: 0.1 → 0.5` (strong flow regularization)
           - **Rationale**: Physics loss should guide optimization gently (0.05), while flow regularization must be strong (0.5) to learn bounded outputs. Loss balance: all components comparable magnitude
           - **Expected improvement**: NLL 12.1 → 2-4 bits by epoch 15, Physics loss 5513 → <10, Train-Val gap 5517 → <2
           - **Timeline**: Epoch 1 - Physics drops dramatically; Epoch 5 - NLL < 6; Epoch 15 - NLL < 3 bits
           - **Physics Loss - First Signal Only** (FIXED - Nov 13 10:30, 2025): Physics loss was penalizing secondary signals in overlaps:
           - **Issue**: Physics loss raw magnitude 27568 (99.8% of total after weight fix) applying to all batch signals
           - **Root cause**: Secondary signals in overlapping data are edge cases intentionally out-of-bounds for training robustness; shouldn't constrain posterior flow
           - **Dataset context**: `dataset_generator.py` lines 4660-4665 creates overlaps with spacing i*1.5s; secondary signals intentionally at parameter extremes
           - **Fix**: Restrict physics loss to first signal only (ground truth) in `src/ahsd/models/overlap_neuralpe.py` line 765:
              - BEFORE: `physics_loss = self._compute_physics_loss(true_params)`
              - AFTER: `physics_loss, physics_violations = self._compute_physics_loss(true_params[:1, :])`
           - **Code changes**: (1) Line 765: Pass `true_params[:1, :]` instead of `true_params` to physics loss, (2) Line 908: Return tuple `(loss, debug_violations)` for logging, (3) Lines 433-442 in phase3a_neural_pe.py: Added debug logging for parameter violations
           - **Debug logging**: Each epoch prints parameter ranges and violation counts to identify which params cause issues
           - **Expected improvement**: Physics loss raw 27568 → 1-10 (single clean sample), allows NLL to properly optimize, Train-Val gap closes
           - **Timeline**: Epoch 1 - Physics loss tiny, NLL dominates; Epoch 5 - Smooth convergence visible; Epoch 15 - NLL <3 bits target
           - **Verification**: Run `python experiments/phase3a_neural_pe.py --epochs 10 --log_level DEBUG` and check Epoch 1 Batch 0 logging shows zero violations
- **Context Encoder Dimension Mismatch** (FIXED - Nov 13 21:45, 2025): Hardcoded default context_dim caused untrained context encoder:
         - **Issue**: Context encoder outputting [16, 512] instead of [16, 768], statistics showed mean≈0, std≈1.0 (random)
         - **Root cause**: Line 57 in `src/ahsd/models/overlap_neuralpe.py` had default `context_dim=512` but config specifies `context_dim: 768`
         - **Fix**: Changed default from 512 → 768 (line 57) + added diagnostic logging to verify output dimensions on init
         - **Code changes**: (1) Line 57: Default 512 → 768, (2) Lines 267-282: Verify context encoder output matches config, (3) Lines 311-319: Verify flow receives correct context_dim
         - **Impact**: Context encoder now properly initialized with 768 dimensions, flow receives correct embedding size
         - **Verification**: Test confirms `actual_context_dim == 768 ✅`
         - **Result**: NLL improved from 18.2 → 8.3 bits in 15 epochs (working, but slower than expected)
- **Flow Capacity Increase for Convergence** (APPLIED - Nov 13 21:50, 2025): Doubled flow parameters to improve posterior learning:
         - **Issue**: NLL plateau at 8.32 bits (target <3.0) after 15 epochs - suggests capacity bottleneck
         - **Analysis**: Original 4-layer, 256-dim flow may be too weak for 9D posterior with 768D context
         - **Fix**: Increased flow capacity 4.2× in `configs/enhanced_training.yaml` (lines 117-119):
            - `hidden_features: 256 → 512` (2x)
            - `num_layers: 4 → 8` (2x)
            - `solver_steps: 10 → 20` (2x accuracy)
         - **Impact**: Flow parameters 2.2M → 9.3M; Total model ~43M params; Training ~120-150s/epoch (2-2.5x slower)
         - **Verification**: Forward pass confirms 9.3M flow parameters ✅
         - **Expected improvement**: NLL should drop rapidly by epoch 20 if capacity was bottleneck; if plateau persists, architectural redesign needed
         - **Timeline**: Monitor epoch 20 for decision point (continue training vs pivot)

           **Model Training:**
- Monitor calibration and output dynamic range
- Use decoy injection for better real/false separation
- Check model behavior on edge cases and real events
- Validate on both synthetic and real GWOSC/GWTC events

**Performance:**
- GW data processing is I/O intensive - batch operations where possible
- Use PyTorch DataLoader with multiple workers
- Profile long training runs with nohup on AWS EC2
- Monitor memory usage for large datasets

**Debugging:**
- Log intermediate shapes and statistics during development
- Visualize embeddings and predictions to spot issues
- Test on small subsets before full training runs
- Save checkpoints frequently for long experiments

## Git Workflow

- Create feature branches for new development
- Write descriptive commit messages explaining "why" not just "what"
- Run tests and linters before committing
- Keep commits atomic and focused

## Security & Data

- Never commit API keys, credentials, or tokens
- Real gravitational wave data from LIGO/Virgo is public but cite properly
- Model checkpoints can be large - use Git LFS or exclude from repo
- AWS credentials should be in environment variables, not config files