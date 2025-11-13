# OverlapNeuralPE Test Suite

Comprehensive testing script for validating OverlapNeuralPE model performance.

## Quick Start

```bash
python test_overlap_neural_pe.py \
    --model_path models/best_model.pth \
    --batch_size 4 \
    --device cuda
```

## Features

### Test 1: Forward Pass
- Validates context encoding
- Tests signal extraction (PriorityNet)
- Verifies flow forward/inverse passes

### Test 2: Loss Computation
- Computes full loss dict (NLL + components)
- Checks for NaN/Inf values
- Validates loss magnitude (should be <15 for good models)

### Test 3: Gradient Flow
- Tests backward pass through entire model
- Checks for vanishing gradients
- Reports gradient statistics per layer
- Flag: ⚠️ warns if >50% vanishing gradients

### Test 4: Posterior Sampling
- Samples 100+ posterior samples per batch element
- Validates sample shapes and statistics
- Checks parameter std dev (should be >1e-6)

### Test 5: Multi-Detector Handling
- Tests H1, L1 strain handling
- Validates shape compatibility
- Tests detector truncation/padding

### Test 6: Signal Extraction
- Validates PriorityNet integration
- Tests multi-signal decomposition
- Reports extraction output shapes

### Test 7: Output Ranges
- Validates physical parameter ranges:
  - mass_1, mass_2: [2, 150] M☉
  - luminosity_distance: [10, 5000] Mpc
  - ra, dec, psi, phase: standard angles
  - theta_jn: [0, π]
- Reports out-of-range statistics

## Usage

### Basic test with default settings
```bash
python test_overlap_neural_pe.py --model_path models/best_model.pth
```

### Custom batch size and device
```bash
python test_overlap_neural_pe.py \
    --model_path models/best_model.pth \
    --batch_size 8 \
    --device cuda
```

### Save results to custom location
```bash
python test_overlap_neural_pe.py \
    --model_path models/best_model.pth \
    --output /tmp/test_results.json
```

## Output Format

### Console Output
Clear pass/fail status for each test with diagnostic details:
```
TEST 1: FORWARD PASS
✅ Forward pass successful
   Batch size: 4
   Input shape: (4, 2, 16384)
   Context encoding: torch.Size([4, 128])
   Flow forward: log_prob shape torch.Size([4])
   Flow inverse: samples shape torch.Size([4, 9])
```

### JSON Results File
```json
{
  "passed": 7,
  "failed": 0,
  "skipped": 0,
  "details": {
    "forward_pass": {
      "status": "PASS",
      "batch_size": 4
    },
    "loss_computation": {
      "status": "PASS",
      "loss_dict": {
        "total_loss": 8.234,
        "nll": 7.892,
        "physics_loss": 0.342
      }
    },
    ...
  }
}
```

## Interpreting Results

### Good Results
- ✅ All tests pass
- NLL: 6.0-8.5 (publication quality)
- Gradient flow: >80% normal/large gradients
- Parameter std: >0.01 per dimension
- Out-of-range: <5% of samples

### Warning Signs
- ⚠️ Vanishing gradients: flow layers not training
- ⚠️ High NLL: >10 indicates poor model quality
- ⚠️ Small parameter std: model undershooting uncertainty
- ⚠️ Out-of-range samples: model extrapolating beyond priors

## Configuration

All model configuration is loaded from the checkpoint. To customize:

1. Edit configs/enhanced_training.yaml
2. Retrain model with new config
3. Re-run tests with new checkpoint

Key parameters affecting results:
- `flow_type`: "flowmatching" or "realnvp"
- `context_dim`: encoder output dimension
- `n_layers`: flow network depth
- `dropout`: regularization strength

## Example: Full Testing Workflow

```bash
# After training a model
python test_overlap_neural_pe.py \
    --model_path models/best_model.pth \
    --batch_size 8 \
    --device cuda \
    --output outputs/test_results.json

# Check results
cat outputs/test_results.json | python -m json.tool

# Verify key metrics
python -c "
import json
with open('outputs/test_results.json') as f:
    results = json.load(f)
print(f'Passed: {results[\"passed\"]}/{results[\"passed\"] + results[\"failed\"]}')
print(f'NLL: {results[\"details\"][\"loss_computation\"][\"loss_dict\"][\"nll\"]:.4f}')
"
```

## Troubleshooting

### Model loading fails
- Check checkpoint path is correct
- Ensure priority_net path is available (warning is OK)
- Try `--device cpu` if CUDA unavailable

### Loss is NaN
- Model weights may be corrupted
- Try with fresh checkpoint
- Check input data validity

### Vanishing gradients
- May indicate training issues
- Check if model was trained with batch norm
- Consider learning rate adjustment

### Out-of-range samples
- Physics loss may need strengthening
- Consider adding bounds penalties
- May indicate insufficient training

## Extending the Tests

To add custom tests, subclass `OverlapNeuralPETest`:

```python
class MyTest(OverlapNeuralPETest):
    def test_my_feature(self, batch_size: int = 4):
        """My custom test"""
        logger.info("\n" + "="*70)
        logger.info("TEST: MY FEATURE")
        logger.info("="*70)
        
        # Your test logic here
        
        self.results['my_feature'] = {'status': 'PASS'}
        return True

# Use it
tester = MyTest(model_path="...")
results = tester.run_all_tests()
```

## Integration with Training

The test suite can be run during training:

```bash
# After each training checkpoint
python test_overlap_neural_pe.py \
    --model_path models/checkpoint_epoch_10.pth \
    --output outputs/checkpoint_10_test.json
```

Monitor results across epochs to track model improvement.
