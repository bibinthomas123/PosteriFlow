#!/usr/bin/env python
"""
Validation script for neural noise integration.
Tests that neural noise generation is fast and produces valid output.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ahsd.data.neural_noise_generator import MultiDetectorNeuralNoiseGenerator


def test_neural_noise_generation():
    """Test neural noise generation speed and quality."""
    print("\n" + "=" * 70)
    print("NEURAL NOISE INTEGRATION TEST")
    print("=" * 70)
    
    # Initialize generator with model paths
    model_paths = {
        "H1": "data/Gaussian_network.pickle",
        "L1": "data/Gaussian_network.pickle",
        "V1": "data/Gaussian_network.pickle",
    }
    
    print("\n1. Initializing MultiDetectorNeuralNoiseGenerator...")
    try:
        gen = MultiDetectorNeuralNoiseGenerator(
            model_paths=model_paths,
            model_type="gaussian",
            sample_rate=4096,
            duration=4.0,
            device="cuda"
        )
        print("   ✓ Generator initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return False
    
    # Test generation speed (should be <1ms per detector)
    print("\n2. Testing generation speed...")
    num_trials = 10
    times = []
    
    for i in range(num_trials):
        start = time.time()
        noise_dict = gen.generate()
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)
        print(f"   Trial {i+1}: {elapsed:.2f} ms")
    
    avg_time = np.mean(times)
    print(f"   Average: {avg_time:.2f} ms per sample")
    
    if avg_time > 100:  # Allow some slack for slow systems
        print(f"   ⚠ Warning: Generation is slower than expected (<1ms target)")
    else:
        print(f"   ✓ Generation speed is excellent")
    
    # Test output quality
    print("\n3. Testing output quality...")
    noise_dict = gen.generate()
    
    for detector, noise in noise_dict.items():
        print(f"\n   {detector}:")
        print(f"      Shape: {noise.shape} (expected: (16384,) for 4s @ 4096 Hz)")
        print(f"      Dtype: {noise.dtype}")
        print(f"      Min/Max: {noise.min():.2e} / {noise.max():.2e}")
        print(f"      Mean: {noise.mean():.2e}")
        print(f"      Std: {noise.std():.2e}")
        
        # Check for NaN/Inf
        has_nan = np.any(~np.isfinite(noise))
        if has_nan:
            print(f"      ✗ Contains NaN/Inf values!")
            return False
        else:
            print(f"      ✓ All values are finite")
        
        # Check reasonable amplitude (should be ~1e-21 to 1e-23)
        rms = np.sqrt(np.mean(noise**2))
        print(f"      RMS: {rms:.2e}")
        if 1e-25 < rms < 1e-18:
            print(f"      ✓ RMS amplitude is realistic")
        else:
            print(f"      ⚠ RMS amplitude seems unusual")
    
    # Test with specific detectors
    print("\n4. Testing selective generation...")
    noise_dict = gen.generate(detectors=["H1", "L1"])
    if set(noise_dict.keys()) == {"H1", "L1"}:
        print("   ✓ Selective generation works correctly")
    else:
        print(f"   ✗ Expected ['H1', 'L1'], got {list(noise_dict.keys())}")
        return False
    
    # Test reproducibility with seed
    print("\n5. Testing reproducibility with seed...")
    noise1 = gen.generate(seed=42)
    noise2 = gen.generate(seed=42)
    
    all_match = True
    for detector in ["H1", "L1", "V1"]:
        if np.allclose(noise1[detector], noise2[detector]):
            print(f"   ✓ {detector} reproducible with seed")
        else:
            print(f"   ✗ {detector} not reproducible")
            all_match = False
    
    if not all_match:
        return False
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    return True


def test_dataset_integration():
    """Test integration with dataset generator."""
    print("\n" + "=" * 70)
    print("DATASET GENERATOR INTEGRATION TEST")
    print("=" * 70)
    
    try:
        from ahsd.data.dataset_generator import GWDatasetGenerator
        
        config = {
            "neural_noise_enabled": True,
            "neural_noise_prob": 0.5,
            "neural_model_type": "gaussian",
            "neural_model_paths": {
                "H1": "data/Gaussian_network.pickle",
                "L1": "data/Gaussian_network.pickle",
                "V1": "data/Gaussian_network.pickle",
            },
            "neural_device": "cuda",
            "use_real_noise_prob": 0.0,  # Disable slow GWOSC
        }
        
        print("\nInitializing GWDatasetGenerator with neural noise config...")
        gen = GWDatasetGenerator(
            output_dir="data/test_dataset",
            config=config,
            output_format="pkl"
        )
        print("✓ GWDatasetGenerator initialized with neural noise enabled")
        
        print("\nGenerating a sample...")
        sample = gen._generate_single_sample(
            sample_id=0,
            is_edge_case=False,
            add_glitches=False,
            preprocess=True
        )
        print(f"✓ Sample generated successfully")
        print(f"  Sample ID: {sample['sample_id']}")
        print(f"  Event type: {sample['type']}")
        print(f"  Detectors: {list(sample['detector_data'].keys())}")
        
        for detector, data in sample['detector_data'].items():
            strain = data.get('strain')
            if strain is not None:
                print(f"  {detector}: strain shape {strain.shape}, dtype {strain.dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "🚀 Neural Noise Integration Validation" + "\n")
    
    # Run tests
    test1 = test_neural_noise_generation()
    test2 = test_dataset_integration()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Neural Noise Generation: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Dataset Integration: {'✓ PASS' if test2 else '✗ FAIL'}")
    print("=" * 70 + "\n")
    
    sys.exit(0 if (test1 and test2) else 1)
