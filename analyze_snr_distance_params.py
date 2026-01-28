#!/usr/bin/env python3
"""
Deep analysis of SNR-distance relationship in the generated dataset.

This checks:
1. Are target_snr values actually being sampled from all regimes?
2. Are distance values properly distributed?
3. Is the SNR-distance formula being used during parameter sampling?
"""

import sys
import pickle
from pathlib import Path
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src"))


def analyze_snr_distance_params():
    """Analyze SNR and distance parameters in dataset samples."""
    
    print("=" * 80)
    print("SNR-DISTANCE PARAMETER ANALYSIS")
    print("=" * 80)
    
    # Load samples
    chunk_path = "data/dataset/train/chunk_0000.pkl"
    if not Path(chunk_path).exists():
        print(f"❌ File not found: {chunk_path}")
        return
    
    with open(chunk_path, 'rb') as f:
        samples = pickle.load(f)
    
    if not isinstance(samples, list):
        samples = [samples]
    
    print(f"\n✓ Loaded {len(samples)} samples\n")
    
    snr_values = []
    distance_values = []
    regimes_count = defaultdict(int)
    event_types_count = defaultdict(int)
    
    # Collect all SNR and distance values
    for idx, sample in enumerate(samples):
        params = sample.get("parameters", {})
        
        # Handle both single signal (dict) and overlapping (list)
        if isinstance(params, list):
            # Overlapping signals
            for param in params:
                if isinstance(param, dict):
                    target_snr = param.get("target_snr")
                    distance = param.get("luminosity_distance")
                    event_type = param.get("type", "unknown")
                    
                    if target_snr and distance:
                        snr_values.append(target_snr)
                        distance_values.append(distance)
                        event_types_count[event_type] += 1
        else:
            # Single signal
            if isinstance(params, dict):
                target_snr = params.get("target_snr")
                distance = params.get("luminosity_distance")
                event_type = params.get("type", "unknown")
                
                if target_snr and distance:
                    snr_values.append(target_snr)
                    distance_values.append(distance)
                    event_types_count[event_type] += 1
    
    snr_values = np.array(snr_values)
    distance_values = np.array(distance_values)
    
    print("-" * 80)
    print("SNR DISTRIBUTION")
    print("-" * 80)
    print(f"Count: {len(snr_values)}")
    print(f"Mean: {np.mean(snr_values):.2f}")
    print(f"Median: {np.median(snr_values):.2f}")
    print(f"Std: {np.std(snr_values):.2f}")
    print(f"Min: {np.min(snr_values):.2f}")
    print(f"Max: {np.max(snr_values):.2f}")
    
    # Regime distribution
    weak_mask = (snr_values >= 8) & (snr_values < 12)
    low_mask = (snr_values >= 12) & (snr_values < 20)
    medium_mask = (snr_values >= 20) & (snr_values < 40)
    high_mask = (snr_values >= 40) & (snr_values < 70)
    loud_mask = (snr_values >= 70)
    
    print(f"\nRegime distribution:")
    print(f"  Weak (8-12):    {np.sum(weak_mask):4d} ({np.sum(weak_mask)/len(snr_values)*100:5.1f}%)")
    print(f"  Low (12-20):    {np.sum(low_mask):4d} ({np.sum(low_mask)/len(snr_values)*100:5.1f}%)")
    print(f"  Medium (20-40): {np.sum(medium_mask):4d} ({np.sum(medium_mask)/len(snr_values)*100:5.1f}%)")
    print(f"  High (40-70):   {np.sum(high_mask):4d} ({np.sum(high_mask)/len(snr_values)*100:5.1f}%)")
    print(f"  Loud (>70):     {np.sum(loud_mask):4d} ({np.sum(loud_mask)/len(snr_values)*100:5.1f}%)")
    
    print("\n" + "-" * 80)
    print("DISTANCE DISTRIBUTION")
    print("-" * 80)
    print(f"Count: {len(distance_values)}")
    print(f"Mean: {np.mean(distance_values):.2f} Mpc")
    print(f"Median: {np.median(distance_values):.2f} Mpc")
    print(f"Std: {np.std(distance_values):.2f} Mpc")
    print(f"Min: {np.min(distance_values):.2f} Mpc")
    print(f"Max: {np.max(distance_values):.2f} Mpc")
    
    print("\n" + "-" * 80)
    print("EVENT TYPE DISTRIBUTION")
    print("-" * 80)
    for event_type, count in sorted(event_types_count.items(), key=lambda x: -x[1]):
        print(f"  {event_type:6s}: {count:4d}")
    
    # Check for SNR-distance relationship WITHIN each event type
    print("\n" + "-" * 80)
    print("SNR-DISTANCE CORRELATION BY EVENT TYPE")
    print("-" * 80)
    
    for idx, sample in enumerate(samples):
        params = sample.get("parameters", {})
        
        if isinstance(params, list):
            for param in params:
                if isinstance(param, dict):
                    event_type = param.get("type")
                    target_snr = param.get("target_snr")
                    distance = param.get("luminosity_distance")
                    
                    # Store for analysis by type
                    if event_type:
                        if not hasattr(analyze_snr_distance_params, 'by_type'):
                            analyze_snr_distance_params.by_type = defaultdict(lambda: {"snr": [], "d": []})
                        analyze_snr_distance_params.by_type[event_type]["snr"].append(target_snr)
                        analyze_snr_distance_params.by_type[event_type]["d"].append(distance)
        else:
            if isinstance(params, dict):
                event_type = params.get("type")
                target_snr = params.get("target_snr")
                distance = params.get("luminosity_distance")
                
                if event_type:
                    if not hasattr(analyze_snr_distance_params, 'by_type'):
                        analyze_snr_distance_params.by_type = defaultdict(lambda: {"snr": [], "d": []})
                    analyze_snr_distance_params.by_type[event_type]["snr"].append(target_snr)
                    analyze_snr_distance_params.by_type[event_type]["d"].append(distance)
    
    if hasattr(analyze_snr_distance_params, 'by_type'):
        for event_type in ["BBH", "BNS", "NSBH"]:
            if event_type in analyze_snr_distance_params.by_type:
                data = analyze_snr_distance_params.by_type[event_type]
                snr_arr = np.array(data["snr"])
                d_arr = np.array(data["d"])
                
                if len(snr_arr) > 1:
                    corr = np.corrcoef(snr_arr, d_arr)[0, 1]
                    print(f"\n{event_type}: n={len(snr_arr)}, r={corr:.4f}")
                    print(f"  SNR range: {np.min(snr_arr):.2f} - {np.max(snr_arr):.2f}")
                    print(f"  Distance range: {np.min(d_arr):.2f} - {np.max(d_arr):.2f} Mpc")
                    
                    # Check if distance scales with SNR inversely
                    # For each SNR, what's the median distance?
                    snr_regimes = [
                        ("Weak (8-12)", (8, 12)),
                        ("Low (12-20)", (12, 20)),
                        ("Medium (20-40)", (20, 40)),
                        ("High (40-70)", (40, 70)),
                        ("Loud (>70)", (70, 200)),
                    ]
                    
                    print(f"  Distance by SNR regime:")
                    for regime_name, (snr_min, snr_max) in snr_regimes:
                        mask = (snr_arr >= snr_min) & (snr_arr < snr_max)
                        if np.sum(mask) > 0:
                            med_d = np.median(d_arr[mask])
                            print(f"    {regime_name:15s}: median distance = {med_d:.2f} Mpc")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("""
If distances are INDEPENDENT of SNR within each event type,
the formula in parameter_sampler.py is NOT being used:

    distance = ref_distance × (Mc/ref_mass)^(5/6) × (ref_snr / target_snr)

This formula REQUIRES:
  - Different Mc → different distance (for same SNR)
  - Different target_snr → DIFFERENT distance (for same Mc)

Check: Do high-SNR samples have SMALLER distances than low-SNR samples?
""")


if __name__ == "__main__":
    analyze_snr_distance_params()
