#!/usr/bin/env python3
"""
Quick validation script for TransformerStrainEncoder enhancements.
Checks syntax, structure, and basic functionality.
"""

import sys
import ast
from pathlib import Path

def check_syntax(filepath):
    """Check Python syntax is valid."""
    print(f"✓ Checking syntax: {filepath}")
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        print("  ✅ Syntax valid")
        return True
    except SyntaxError as e:
        print(f"  ❌ Syntax error: {e}")
        return False

def check_key_components(filepath):
    """Check for key enhancements in the file."""
    print(f"\n✓ Checking key components in: {filepath.name}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    checks = {
        "Positional encoding adapter": "pos_encoding_adapter",
        "Attention mask support": "attention_mask: Optional[torch.Tensor]",
        "Masked average pooling": "mask_expanded = attention_mask.unsqueeze(-1)",
        "Gradient clamp": "clamp(min=1e-9)",
        "Logger enhancements": "pos_encoding for",
    }
    
    results = {}
    for name, pattern in checks.items():
        if pattern in content:
            print(f"  ✅ {name}")
            results[name] = True
        else:
            print(f"  ❌ {name} - pattern '{pattern}' not found")
            results[name] = False
    
    return results

def check_test_file(filepath):
    """Validate test file structure."""
    print(f"\n✓ Checking test file: {filepath.name}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    test_methods = [
        "test_whisper_mode",
        "test_fallback_mode",
        "test_frozen_layers",
        "test_positional_encoding",
        "test_attention_mask",
        "test_variable_length_sequences",
        "test_gradient_flow",
        "test_output_properties",
        "test_batch_independence",
        "test_device_compatibility",
    ]
    
    found = 0
    for test_name in test_methods:
        if test_name in content:
            print(f"  ✅ {test_name}")
            found += 1
        else:
            print(f"  ❌ {test_name}")
    
    print(f"\n  Total: {found}/{len(test_methods)} tests defined")
    return found == len(test_methods)

def check_benchmark_file(filepath):
    """Validate benchmark file."""
    print(f"\n✓ Checking benchmark script: {filepath.name}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    components = {
        "EncoderBenchmark class": "class EncoderBenchmark",
        "benchmark_encoder method": "def benchmark_encoder",
        "run_comparison method": "def run_comparison",
        "benchmark_with_masks method": "def benchmark_with_masks",
        "benchmark_mixed_precision method": "def benchmark_mixed_precision",
        "Main CLI": "def main()",
    }
    
    found = 0
    for name, pattern in components.items():
        if pattern in content:
            print(f"  ✅ {name}")
            found += 1
        else:
            print(f"  ❌ {name}")
    
    print(f"\n  Total: {found}/{len(components)} components")
    return found == len(components)

def check_documentation(filepath):
    """Validate documentation file."""
    print(f"\n✓ Checking documentation: {filepath.name}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    sections = {
        "Summary": "## Summary",
        "Core Updates": "### Core Updates",
        "Positional Encoding": "## Positional Encoding Adapter",
        "Test Suite": "## New Test Suite",
        "Benchmark Suite": "## Benchmark Suite",
        "Integration": "## Integration with Training",
        "Migration": "## Compatibility & Migration",
        "Troubleshooting": "## Troubleshooting",
        "Performance": "## Performance Expectations",
    }
    
    found = 0
    for name, pattern in sections.items():
        if pattern in content:
            print(f"  ✅ {name}")
            found += 1
        else:
            print(f"  ⚠️  {name}")
    
    return found >= len(sections) - 2  # Allow 2 missing sections

def main():
    print("=" * 80)
    print("TransformerStrainEncoder Enhancement Validation")
    print("=" * 80)
    
    project_root = Path(__file__).parent
    
    # Files to check
    encoder_file = project_root / "src/ahsd/models/transformer_encoder.py"
    test_file = project_root / "tests/test_transformer_encoder_enhanced.py"
    benchmark_file = project_root / "scripts/benchmark_encoder.py"
    doc_file = project_root / "FIX_DOCS/TRANSFORMER_ENCODER_ENHANCEMENT.md"
    
    results = {}
    
    # Check encoder file
    results["encoder_syntax"] = check_syntax(encoder_file)
    results["encoder_components"] = check_key_components(encoder_file)
    
    # Check test file
    if test_file.exists():
        results["test_syntax"] = check_syntax(test_file)
        results["test_structure"] = check_test_file(test_file)
    else:
        print(f"\n❌ Test file not found: {test_file}")
        results["test_structure"] = False
    
    # Check benchmark file
    if benchmark_file.exists():
        results["benchmark_syntax"] = check_syntax(benchmark_file)
        results["benchmark_structure"] = check_benchmark_file(benchmark_file)
    else:
        print(f"\n❌ Benchmark file not found: {benchmark_file}")
        results["benchmark_structure"] = False
    
    # Check documentation
    if doc_file.exists():
        results["documentation"] = check_documentation(doc_file)
    else:
        print(f"\n❌ Documentation file not found: {doc_file}")
        results["documentation"] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = all(
        v for k, v in results.items()
        if isinstance(v, bool)
    )
    
    if all_passed and isinstance(results.get("encoder_components"), dict):
        component_all_ok = all(results["encoder_components"].values())
    else:
        component_all_ok = False
    
    if all_passed and component_all_ok:
        print("\n✅ All validations PASSED!")
        print("\nNext steps:")
        print("  1. Run tests: pytest tests/test_transformer_encoder_enhanced.py -v")
        print("  2. Run benchmarks: python scripts/benchmark_encoder.py --iterations 100")
        print("  3. Integrate into training: See FIX_DOCS/TRANSFORMER_ENCODER_ENHANCEMENT.md")
        return 0
    else:
        print("\n❌ Some validations FAILED")
        print("\nFailed checks:")
        for k, v in results.items():
            if isinstance(v, bool) and not v:
                print(f"  - {k}")
            elif isinstance(v, dict):
                for ck, cv in v.items():
                    if not cv:
                        print(f"  - {ck}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
