#!/usr/bin/env python3
"""
Dataset Validation Script - Checks for corruption from decoy injection fix
Run: python validate_dataset_fix.py
"""

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

def check_1_mismatches(dataset_dir="data/dataset"):
    """Check for n_signals ≠ len(parameters) mismatches"""
    print("\n" + "="*60)
    print("CHECK 1: Signal Count Mismatches")
    print("="*60)
    
    mismatches = 0
    total = 0
    mismatch_details = []
    
    for split in ['train', 'validation', 'test']:
        split_path = Path(dataset_dir) / split
        if not split_path.exists():
            continue
        
        chunks = sorted(split_path.glob("chunk_*.pkl"))
        for chunk_file in chunks:
            with open(chunk_file, 'rb') as f:
                data = pickle.load(f)
            
            for idx, sample in enumerate(data):
                n_signals = sample.get('n_signals')
                n_params = len(sample.get('parameters', []))
                n_prios = len(sample.get('priorities', []))
                
                total += 1
                if n_params != n_signals or n_prios != n_signals:
                    mismatches += 1
                    if len(mismatch_details) < 5:
                        mismatch_details.append(
                            f"  n_signals={n_signals}, len(params)={n_params}, len(prios)={n_prios}"
                        )
    
    pct = 100 * mismatches / total if total > 0 else 0
    status = "✅ PASS" if mismatches == 0 else "❌ FAIL"
    
    print(f"Total samples checked: {total}")
    print(f"Mismatches found: {mismatches} ({pct:.1f}%)")
    if mismatch_details:
        print("Examples:")
        for detail in mismatch_details:
            print(detail)
    print(f"Result: {status}")
    
    return mismatches == 0

def check_2_decoys(dataset_dir="data/dataset"):
    """Check for decoy signals (duplicates with similar mass)"""
    print("\n" + "="*60)
    print("CHECK 2: Decoy Signal Detection")
    print("="*60)
    
    suspected_decoys = 0
    samples_checked = 0
    
    for split in ['train', 'validation', 'test']:
        split_path = Path(dataset_dir) / split
        if not split_path.exists():
            continue
        
        chunks = sorted(split_path.glob("chunk_*.pkl"))
        for chunk_file in chunks:
            with open(chunk_file, 'rb') as f:
                data = pickle.load(f)
            
            for sample in data:
                params = sample.get('parameters', [])
                samples_checked += 1
                
                if len(params) < 2:
                    continue
                
                # Decoys: similar mass to first signal
                first_m1 = params[0].get('mass_1', 0)
                last_m1 = params[-1].get('mass_1', 0)
                
                if abs(first_m1 - last_m1) < 1.0:  # Very similar mass
                    suspected_decoys += 1
    
    status = "✅ PASS" if suspected_decoys < 50 else "⚠️  CHECK"
    print(f"Total samples checked: {samples_checked}")
    print(f"Suspected decoys: {suspected_decoys}")
    print(f"Result: {status}")
    
    return suspected_decoys < 50

def check_3_priority_alignment(dataset_dir="data/dataset"):
    """Check priority-parameter alignment and value ranges"""
    print("\n" + "="*60)
    print("CHECK 3: Priority Alignment & Ranges")
    print("="*60)
    
    errors = 0
    out_of_range = 0
    samples_checked = 0
    
    for split in ['train', 'validation', 'test']:
        split_path = Path(dataset_dir) / split
        if not split_path.exists():
            continue
        
        chunks = sorted(split_path.glob("chunk_*.pkl"))
        for chunk_file in chunks:
            with open(chunk_file, 'rb') as f:
                data = pickle.load(f)
            
            for sample in data:
                params = sample.get('parameters', [])
                prios = sample.get('priorities', [])
                samples_checked += 1
                
                # Length match
                if len(params) != len(prios):
                    errors += 1
                    continue
                
                # Value ranges [0, 1]
                for p in prios:
                    if not (0.0 <= p <= 1.0):
                        out_of_range += 1
    
    status = "✅ PASS" if (errors == 0 and out_of_range == 0) else "❌ FAIL"
    print(f"Total samples checked: {samples_checked}")
    print(f"Length mismatches: {errors}")
    print(f"Out-of-range priorities: {out_of_range}")
    print(f"Result: {status}")
    
    return errors == 0 and out_of_range == 0

def check_4_statistics(dataset_dir="data/dataset"):
    """Check dataset statistics"""
    print("\n" + "="*60)
    print("CHECK 4: Dataset Statistics")
    print("="*60)
    
    stats = {
        'single': 0,
        'pair': 0,
        'triple_quad': 0,
        'dense': 0,
        'priorities': [],
        'spreads': [],
    }
    
    for split in ['train', 'validation', 'test']:
        split_path = Path(dataset_dir) / split
        if not split_path.exists():
            continue
        
        chunks = sorted(split_path.glob("chunk_*.pkl"))
        for chunk_file in chunks:
            with open(chunk_file, 'rb') as f:
                data = pickle.load(f)
            
            for sample in data:
                n = sample.get('n_signals')
                prios = sample.get('priorities', [])
                
                if n == 1:
                    stats['single'] += 1
                elif n == 2:
                    stats['pair'] += 1
                elif n in [3, 4]:
                    stats['triple_quad'] += 1
                else:
                    stats['dense'] += 1
                
                if prios:
                    stats['priorities'].extend(prios)
                    stats['spreads'].append(max(prios) - min(prios) if len(prios) > 1 else 0)
    
    total = sum(stats[k] for k in ['single', 'pair', 'triple_quad', 'dense'])
    
    print(f"Total samples: {total}")
    print(f"  Single signal: {stats['single']} ({100*stats['single']/total:.1f}%)")
    print(f"  2-signal overlap: {stats['pair']} ({100*stats['pair']/total:.1f}%)")
    print(f"  3-4 signal overlap: {stats['triple_quad']} ({100*stats['triple_quad']/total:.1f}%)")
    print(f"  5+ signal overlap: {stats['dense']} ({100*stats['dense']/total:.1f}%)")
    
    if stats['priorities']:
        print(f"\nPriority statistics:")
        print(f"  Mean: {np.mean(stats['priorities']):.3f}")
        print(f"  Std: {np.std(stats['priorities']):.3f}")
        print(f"  Min: {np.min(stats['priorities']):.3f}")
        print(f"  Max: {np.max(stats['priorities']):.3f}")
        print(f"  Avg spread: {np.mean(stats['spreads']):.3f}")
    
    # Check if distribution and statistics are reasonable
    single_pct = 100 * stats['single'] / total if total > 0 else 0
    overlap_pct = 100 * (stats['pair'] + stats['triple_quad'] + stats['dense']) / total if total > 0 else 0
    
    # Validate priority statistics
    if stats['priorities']:
        mean_priority = np.mean(stats['priorities'])
        std_priority = np.std(stats['priorities'])
        min_priority = np.min(stats['priorities'])
        max_priority = np.max(stats['priorities'])
        
        # Check if priorities are well-distributed and in valid range
        valid_range = (0.05 <= min_priority) and (max_priority <= 1.0)
        valid_spread = (std_priority > 0.05)  # Not all same value
        reasonable_mean = (0.2 <= mean_priority <= 0.9)  # Centered, not extreme
        
        reasonable = valid_range and valid_spread and reasonable_mean
    else:
        reasonable = True  # No priorities to check
    
    # Check if signal distribution is sensible (should have both singles and overlaps)
    has_singles = stats['single'] > 0
    has_overlaps = (stats['pair'] + stats['triple_quad'] + stats['dense']) > 0
    
    reasonable = reasonable and has_singles and has_overlaps
    
    status = "✅ PASS" if reasonable else "⚠️  CHECK"
    print(f"\nResult: {status}")
    
    return reasonable

def check_5_zero_anomalies(dataset_dir="data/dataset"):
    """Check for near-zero priority anomalies (should be minimal with log scaling)"""
    print("\n" + "="*60)
    print("CHECK 5: Near-Zero Priority Anomalies")
    print("="*60)
    
    samples_with_very_low = 0
    total_samples = 0
    
    for split in ['train', 'validation', 'test']:
        split_path = Path(dataset_dir) / split
        if not split_path.exists():
            continue
        
        chunks = sorted(split_path.glob("chunk_*.pkl"))
        for chunk_file in chunks:
            with open(chunk_file, 'rb') as f:
                data = pickle.load(f)
            
            for sample in data:
                prios = sample.get('priorities', [])
                total_samples += 1
                
                # Count priorities < 0.05 (should be minimal with log scaling)
                very_low_count = sum(1 for p in prios if p < 0.05)
                if very_low_count > 0:
                    samples_with_very_low += 1
    
    threshold = max(50, int(total_samples * 0.05))  # 5% of dataset
    status = "✅ PASS" if samples_with_very_low < threshold else "⚠️  INVESTIGATE"
    print(f"Total samples: {total_samples}")
    print(f"Samples with priorities < 0.05: {samples_with_very_low}")
    print(f"Threshold: {threshold}")
    print(f"Result: {status}")
    
    return samples_with_very_low < threshold

def main():
    print("\n" + "="*60)
    print("DATASET VALIDATION - Decoy Injection Fix Check")
    print("="*60)
    
    results = {}
    results['check_1'] = check_1_mismatches()
    results['check_2'] = check_2_decoys()
    results['check_3'] = check_3_priority_alignment()
    results['check_4'] = check_4_statistics()
    results['check_5'] = check_5_zero_anomalies()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Check 1 (Mismatches):        {'✅ PASS' if results['check_1'] else '❌ FAIL'}")
    print(f"Check 2 (Decoys):            {'✅ PASS' if results['check_2'] else '⚠️  CHECK'}")
    print(f"Check 3 (Alignment):         {'✅ PASS' if results['check_3'] else '❌ FAIL'}")
    print(f"Check 4 (Statistics):        {'✅ PASS' if results['check_4'] else '⚠️  CHECK'}")
    print(f"Check 5 (Zero Anomalies):    {'✅ PASS' if results['check_5'] else '⚠️  CHECK'}")
    
    all_pass = all(results.values())
    print(f"\n{'='*60}")
    print(f"Overall: {'✅ DATASET CLEAN' if all_pass else '⚠️  NEEDS INVESTIGATION'}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
