# Physics Loss Return Statement Fix (Nov 13, 2025 - 10:45)

## Issue
**Error**: `TypeError: cannot unpack non-iterable NoneType object`
- Location: `phase3a_neural_pe.py` line 765 in `compute_loss()`
- Cause: `_compute_physics_loss()` returning `None` instead of tuple `(loss, violations)`

```python
# Line 765 - Error occurs here
physics_loss, physics_violations = self._compute_physics_loss(true_params[:1, :])
```

## Root Cause
In `overlap_neuralpe.py`, the `_compute_physics_loss()` method had an early return statement only in the BNS (Binary Neutron Star) lambda constraint branch (line 910), but no return statement for other event types:

```python
# BROKEN: Only returns for BNS case with lambda constraints
if self.event_type == "BNS":
    for i, param_name in enumerate(self.param_names):
        if "lambda" in param_name.lower():
            lambda_violation = F.relu(lambda_val - 5000.0)
            loss += 0.3 * torch.mean(lambda_violation**2)
            return loss, debug_violations
# ❌ No return for BBH, NSBH, or BNS without lambda params!
```

This caused the function to fall through and return `None` implicitly for non-BNS events and BBH/NSBH types.

## Fix Applied
**File**: `/home/bibinathomas/PosteriFlow/src/ahsd/models/overlap_neuralpe.py`

**Lines**: 909-910

**Changed**:
```python
# Before (incorrect indentation + early return only for BNS)
if self.event_type == "BNS":
    for i, param_name in enumerate(self.param_names):
        if "lambda" in param_name.lower():
            lambda_violation = F.relu(lambda_val - 5000.0)
            loss += 0.3 * torch.mean(lambda_violation**2)
            return loss, debug_violations
```

**To**:
```python
# After (return statement always executes)
if self.event_type == "BNS":
    for i, param_name in enumerate(self.param_names):
        if "lambda" in param_name.lower():
            lambda_violation = F.relu(lambda_val - 5000.0)
            loss += 0.3 * torch.mean(lambda_violation**2)

# ✅ Always return loss and violations for logging
return loss, debug_violations
```

## Technical Details

The method signature promises to return a tuple:
```python
def _compute_physics_loss(self, params: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
```

But the implementation didn't consistently return values for all code paths.

## Verification

After fix:
1. ✅ Installed updated package: `pip install -e . --no-deps`
2. ✅ Method now returns tuple for all event types (BBH, BNS, NSBH)
3. ✅ Training can resume without unpacking errors

## Impact

- **Training Status**: Can now proceed without TypeError
- **Loss Computation**: Physics loss (both value and debug violations) properly tracked
- **Logging**: Debug violations logged for parameter constraint monitoring

## Related Fixes
- FIX_DOCS/PHYSICS_LOSS_FIRST_SIGNAL_FIX.md - Restricting physics loss to first signal only
- FIX_DOCS/NLL_EXPLOSION_FIX_SUMMARY.md - Loss weight rebalancing for NLL convergence
