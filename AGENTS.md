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

**Data Handling:**
- Always check for NaN/Inf in tensors before training
- Verify dataset serialization includes all required fields (e.g., `network_snr`)
- Scale SNR values appropriately for downstream targets
- Watch for embedding padding issues and near-zero std
- **Distance-SNR Correlation** (FIXED - Nov 2025): Ensures strong negative correlation between distance and SNR:
  - Distance is now derived directly from target_snr using chirp mass scaling: `d = d_ref * (M_c/M_c_ref)^(5/6) * (SNR_ref/target_SNR)`
  - Reference parameters in ParameterSampler: `reference_snr=35`, `reference_distance=400 Mpc`, `reference_mass=30 M_sun`
  - Jitter reduced to 0.1% (0.999-1.001) to preserve correlation fidelity
  - Achieved correlations: BBH r≈-0.75, BNS r≈-0.86, NSBH r≈-0.67 (all negative as required)
  - The `attach_network_snr()` function uses priority order:
    1. Already-sampled `target_snr` (from ParameterSampler)
    2. Per-detector matched-filter SNRs  
    3. Proxy formula (mass/distance-based)
  - NSBH has lower correlation due to wider mass distribution inherent to the physics

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