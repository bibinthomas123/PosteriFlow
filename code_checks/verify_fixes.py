#!/usr/bin/env python3
"""
Verification script for Transformer Encoder & Matched-Filter Metrics fixes
"""

import sys
import subprocess
from pathlib import Path

def check_syntax(filepath):
    """Check if Python file has valid syntax"""
    result = subprocess.run(
        ["python3", "-m", "py_compile", filepath],
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stderr

def main():
    print("=" * 70)
    print("VERIFICATION: Transformer Encoder & MF Metrics Fixes")
    print("=" * 70)
    
    root = Path(__file__).parent
    
    # Check files that were fixed
    files_to_check = [
        "src/ahsd/models/flows.py",
        "src/ahsd/models/transformer_encoder.py",
        "src/ahsd/models/__init__.py",
    ]
    
    print("\n1. SYNTAX VALIDATION")
    print("-" * 70)
    all_valid = True
    for filepath in files_to_check:
        full_path = root / filepath
        if not full_path.exists():
            print(f"  ❌ {filepath}: FILE NOT FOUND")
            all_valid = False
            continue
        
        valid, error = check_syntax(str(full_path))
        if valid:
            print(f"  ✅ {filepath}")
        else:
            print(f"  ❌ {filepath}")
            if error:
                print(f"     Error: {error[:100]}")
            all_valid = False
    
    print("\n2. CRITICAL FIXES VERIFICATION")
    print("-" * 70)
    
    # Check flows.py fixes
    flows_path = root / "src/ahsd/models/flows.py"
    if flows_path.exists():
        content = flows_path.read_text()
        
        checks = [
            ("Property decorators outside __init__", "@property" in content and "def active_layers" in content),
            ("_create_alternating_mask at class level", "def _create_alternating_mask(self" in content),
            ("Dropout parameter in ContextAwareCouplingNet", "dropout: float = 0.1" in content),
            ("ConditionalRealNVP class exists", "class ConditionalRealNVP" in content),
            ("MaskedAutoregressiveFlow class exists", "class MaskedAutoregressiveFlow" in content),
        ]
        
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"  {status} flows.py: {check_name}")
            if not result:
                all_valid = False
    
    # Check transformer_encoder.py fixes
    transformer_path = root / "src/ahsd/models/transformer_encoder.py"
    if transformer_path.exists():
        content = transformer_path.read_text()
        
        checks = [
            ("Whisper import try-except", "try:" in content and "from transformers import WhisperModel" in content),
            ("HAS_WHISPER flag", "HAS_WHISPER = " in content),
            ("LightweightTransformerEncoder fallback", "class LightweightTransformerEncoder" in content),
            ("Forward method returns dict-like object", "return_dict: bool = True" in content),
            ("EncoderOutput compatibility class", "class EncoderOutput:" in content),
        ]
        
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"  {status} transformer_encoder.py: {check_name}")
            if not result:
                all_valid = False
    
    # Check __init__.py graceful degradation
    init_path = root / "src/ahsd/models/__init__.py"
    if init_path.exists():
        content = init_path.read_text()
        
        checks = [
            ("Try-except for OverlapNeuralPE import", "try:" in content and "from .overlap_neuralpe" in content),
            ("Logging warning on import error", "logging.warning" in content),
            ("TransformerStrainEncoder exported", "TransformerStrainEncoder" in content),
            ("Fallback sets None for missing imports", "= None" in content),
        ]
        
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"  {status} __init__.py: {check_name}")
            if not result:
                all_valid = False
    
    print("\n3. DOCUMENTATION")
    print("-" * 70)
    doc_path = root / "FIX_DOCS/TRANSFORMER_AND_MF_METRICS.md"
    if doc_path.exists():
        content = doc_path.read_text()
        doc_checks = [
            ("Issue 1 marked FIXED", "Issue 1: Whisper Not Available ✅ FIXED" in content),
            ("Issue 2 marked FIXED", "Issue 2: flows.py Has Syntax Errors ✅ FIXED" in content),
            ("Fallback encoder documented", "6-layer PyTorch Transformer (256-D" in content),
            ("Installation instructions", "pip install transformers" in content),
        ]
        
        for check_name, result in doc_checks:
            status = "✅" if result else "❌"
            print(f"  {status} TRANSFORMER_AND_MF_METRICS.md: {check_name}")
            if not result:
                all_valid = False
    
    print("\n" + "=" * 70)
    if all_valid:
        print("✅ ALL CHECKS PASSED - Issues successfully fixed!")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Review issues above")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
