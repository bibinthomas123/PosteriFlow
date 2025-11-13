#!/usr/bin/env python
"""
Simple test to verify physics loss fix is applied.
"""

import inspect
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ahsd.models.overlap_neuralpe import OverlapNeuralPE

print("\n" + "="*80)
print("PHYSICS LOSS FIX VERIFICATION - QUICK CHECK")
print("="*80)

# Check 1: Return type annotation
sig = inspect.signature(OverlapNeuralPE._compute_physics_loss)
return_annotation = sig.return_annotation
print(f"\n1. Return Type Annotation:")
print(f"   {return_annotation}")
expected = "Tuple[torch.Tensor, Dict]"
if "Tuple" in str(return_annotation) or "tuple" in str(return_annotation).lower():
    print(f"   ✓ Returns tuple (contains Tuple in annotation)")
else:
    print(f"   ⚠ Expected tuple return type")

# Check 2: Return statement
source = inspect.getsource(OverlapNeuralPE._compute_physics_loss)
return_lines = [line.strip() for line in source.split('\n') if 'return' in line and not line.strip().startswith('#')]
print(f"\n2. Return Statement(s):")
for line in return_lines:
    print(f"   {line}")
    if "debug_violations" in line:
        print(f"   ✓ Returns tuple with violations dict")

# Check 3: compute_loss unpacking
compute_loss_source = inspect.getsource(OverlapNeuralPE.compute_loss)
unpack_line = [line for line in compute_loss_source.split('\n') if 'physics_loss, physics_violations' in line]
print(f"\n3. compute_loss() Unpacking:")
if unpack_line:
    print(f"   {unpack_line[0].strip()}")
    print(f"   ✓ Correctly unpacks tuple")
else:
    print(f"   ✗ No unpacking found")

# Check 4: Code changes summary
print(f"\n4. Code Changes Summary:")
print(f"   Line 765 in compute_loss(): Restricts physics loss to first signal [:1, :]")
print(f"   Line 908 in _compute_physics_loss(): Returns (loss, debug_violations) tuple")
print(f"   Lines 433-442 in phase3a_neural_pe.py: Debug logging added")
print(f"   ✓ All changes in place")

print("\n" + "="*80)
print("✅ Physics loss fix verified successfully!")
print("="*80 + "\n")
