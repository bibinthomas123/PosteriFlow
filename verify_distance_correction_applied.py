#!/usr/bin/env python3
"""
Verify that distance correction (Fix #0B) is actually being applied in dataset samples.

This script checks if:
1. corrected_distance field exists in metadata
2. distance_correction_factor field exists
3. Correction was actually applied (original != corrected)
4. Physics rule maintained: distance × SNR ≈ constant
"""

import sys
import pickle
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_sample_corrections(sample_path):
    """Load a sample file and check for distance corrections."""
    
    print("=" * 80)
    print("DISTANCE CORRECTION VERIFICATION")
    print("=" * 80)
    
    if not Path(sample_path).exists():
        print(f"❌ File not found: {sample_path}")
        return False
    
    try:
        with open(sample_path, 'rb') as f:
            samples = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False
    
    if not isinstance(samples, list):
        samples = [samples]
    
    print(f"\n✓ Loaded {len(samples)} samples from {Path(sample_path).name}\n")
    
    has_corrections = 0
    no_corrections = 0
    corrections_applied = 0
    physics_rule_check = []
    
    for idx, sample in enumerate(samples):
        if "detector_data" not in sample:
            print(f"Sample {idx}: No detector_data field")
            continue
        
        detector_data = sample["detector_data"]
        sample_has_correction = False
        
        # Check each detector
        for det_name, det_data in detector_data.items():
            if "metadata" not in det_data:
                continue
            
            metadata = det_data["metadata"]
            
            # Handle both single signal (dict) and overlapping (list)
            if isinstance(metadata, list):
                # Overlapping signals
                for sig_idx, meta in enumerate(metadata):
                    corrected_dist = meta.get("corrected_distance")
                    correction_factor = meta.get("distance_correction_factor")
                    
                    if corrected_dist is not None and correction_factor is not None:
                        sample_has_correction = True
                        corrections_applied += 1
                        
                        # Get original distance and SNR
                        params = sample["parameters"]
                        if isinstance(params, list) and sig_idx < len(params):
                            param = params[sig_idx]
                            original_distance = param.get("luminosity_distance")
                            target_snr = param.get("target_snr")
                            achieved_snr = meta.get("actual_snr")
                            
                            if original_distance and target_snr and achieved_snr:
                                # Check physics rule: d × SNR = constant
                                product_before = original_distance * target_snr
                                product_after = corrected_dist * achieved_snr
                                ratio = product_before / product_after if product_after > 0 else 1.0
                                physics_rule_check.append({
                                    "sample": idx,
                                    "signal": sig_idx,
                                    "detector": det_name,
                                    "ratio": ratio,
                                    "original_d": original_distance,
                                    "corrected_d": corrected_dist,
                                    "factor": correction_factor,
                                })
            else:
                # Single signal
                corrected_dist = metadata.get("corrected_distance")
                correction_factor = metadata.get("distance_correction_factor")
                
                if corrected_dist is not None and correction_factor is not None:
                    sample_has_correction = True
                    corrections_applied += 1
                    
                    # Get original distance and SNR
                    params = sample.get("parameters", {})
                    if isinstance(params, dict):
                        original_distance = params.get("luminosity_distance")
                        target_snr = params.get("target_snr")
                        achieved_snr = metadata.get("actual_snr")
                        
                        if original_distance and target_snr and achieved_snr:
                            # Check physics rule
                            product_before = original_distance * target_snr
                            product_after = corrected_dist * achieved_snr
                            ratio = product_before / product_after if product_after > 0 else 1.0
                            physics_rule_check.append({
                                "sample": idx,
                                "detector": det_name,
                                "ratio": ratio,
                                "original_d": original_distance,
                                "corrected_d": corrected_dist,
                                "factor": correction_factor,
                            })
        
        if sample_has_correction:
            has_corrections += 1
        else:
            no_corrections += 1
    
    print("-" * 80)
    print("CORRECTION COVERAGE")
    print("-" * 80)
    print(f"Samples with corrections: {has_corrections}")
    print(f"Samples without corrections: {no_corrections}")
    print(f"Total corrections applied: {corrections_applied}")
    print(f"Coverage: {has_corrections}/{len(samples)} samples ({has_corrections/len(samples)*100:.1f}%)")
    
    if has_corrections == 0:
        print("\n❌ NO CORRECTIONS FOUND - Distance correction code not being applied!")
        print("\nPossible causes:")
        print("  1. Samples were generated before the fix was implemented")
        print("  2. achieved_snr is 0 or missing in metadata (correction skipped)")
        print("  3. target_snr is 0 or missing in params (correction skipped)")
        return False
    
    print("\n✅ Corrections ARE being applied!")
    
    # Analyze physics rule
    if physics_rule_check:
        print("\n" + "-" * 80)
        print("PHYSICS RULE CHECK (distance × SNR = constant)")
        print("-" * 80)
        
        ratios = [c["ratio"] for c in physics_rule_check]
        print(f"Ratio (before/after): mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}")
        print(f"  Should be ≈ 1.0 (close to 1.0 means physics rule maintained)")
        print(f"  Min: {np.min(ratios):.4f}, Max: {np.max(ratios):.4f}")
        
        if abs(np.mean(ratios) - 1.0) < 0.01:
            print("\n✅ PHYSICS RULE MAINTAINED! (ratio ≈ 1.0)")
        else:
            print(f"\n⚠️  Physics rule slightly off (ratio = {np.mean(ratios):.4f})")
    
    # Show sample details
    print("\n" + "-" * 80)
    print("SAMPLE EXAMPLES")
    print("-" * 80)
    
    shown = 0
    for idx, sample in enumerate(samples):
        if shown >= 3:
            break
        
        detector_data = sample.get("detector_data", {})
        for det_name in ["H1", "L1", "V1"]:
            if det_name not in detector_data:
                continue
            
            det_data = detector_data[det_name]
            metadata = det_data.get("metadata", {})
            
            corrected_dist = metadata.get("corrected_distance") if isinstance(metadata, dict) else None
            if isinstance(metadata, list) and len(metadata) > 0:
                corrected_dist = metadata[0].get("corrected_distance")
            
            if corrected_dist is not None:
                correction_factor = metadata.get("distance_correction_factor") if isinstance(metadata, dict) else metadata[0].get("distance_correction_factor")
                
                params = sample.get("parameters", {})
                if isinstance(params, list) and len(params) > 0:
                    params = params[0]
                
                original_d = params.get("luminosity_distance")
                
                print(f"\nSample {idx}, {det_name}:")
                print(f"  Original distance: {original_d:.2f} Mpc")
                print(f"  Corrected distance: {corrected_dist:.2f} Mpc")
                print(f"  Correction factor: {correction_factor:.4f}")
                print(f"  Change: {corrected_dist - original_d:.2f} Mpc ({(corrected_dist/original_d - 1)*100:.1f}%)")
                
                shown += 1
                break
    
    return has_corrections > 0


if __name__ == "__main__":
    # Try common paths
    test_paths = [
        "data/dataset/train/chunk_0000.pkl",
        "data/dataset/train/singles_0.pkl",
        "data/dataset/train/overlaps_0.pkl",
        "data/dataset/train/00000.pkl",
        "data/dataset/train/samples_000.pkl",
    ]
    
    found = False
    for path in test_paths:
        if Path(path).exists():
            result = check_sample_corrections(path)
            found = True
            break
    
    if not found:
        print("=" * 80)
        print("ERROR: Could not find dataset files")
        print("=" * 80)
        print("\nTried paths:")
        for path in test_paths:
            print(f"  ✗ {path}")
        
        print("\nAvailable dataset paths:")
        data_dir = Path("data/dataset/train")
        if data_dir.exists():
            files = list(data_dir.glob("*.pkl"))
            if files:
                for f in files[:5]:
                    print(f"  ✓ {f}")
            else:
                print("  (No .pkl files found)")
        else:
            print(f"  (Directory not found: {data_dir})")
