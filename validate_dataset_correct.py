#!/usr/bin/env python3
"""
Corrected validation script for AHSD dataset (50K samples)
Validates distribution metrics, physical constraints, and data quality
Loads from data/output/ (where ahsd-generate saves files)
"""

import pickle
import numpy as np
import glob
from pathlib import Path
from collections import defaultdict

def load_dataset_from_output(split='train'):
    """Load all samples from data/output/{split}/*.pkl"""
    distances, snrs, masses, types, params_list = [], [], [], [], []
    
    # Updated path: ahsd-generate saves to data/output/{train,val,test}/*.pkl
    pattern = f'data/dataset/{split}/*.pkl'
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"❌ No files found matching: {pattern}")
        return np.array([]), np.array([]), np.array([]), [], []
    
    print(f"📂 Loading {len(files)} files from {split} split...")
    
    for pkl_file in files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Handle both list of samples and batch dict format
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict):
                samples = data.get('samples', [])
            else:
                continue
            
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                
                # Extract parameters (primary signal only for now)
                params = sample.get('parameters', {})
                if isinstance(params, list) and len(params) > 0:
                    params = params[0]  # First signal only
                
                if not isinstance(params, dict):
                    continue
                
                d = params.get('luminosity_distance')
                snr = params.get('target_snr')
                mc = params.get('chirp_mass')
                et = params.get('type')
                
                # Validate required fields
                if d is not None and snr is not None and mc is not None and et is not None:
                    try:
                        d = float(d)
                        snr = float(snr)
                        mc = float(mc)
                        
                        # Sanity checks
                        if d > 0 and snr > 0 and mc > 0 and et in ['BBH', 'BNS', 'NSBH']:
                            distances.append(d)
                            snrs.append(snr)
                            masses.append(mc)
                            types.append(et)
                            params_list.append(params)
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            print(f"⚠️  Error loading {pkl_file}: {e}")
            continue
    
    return np.array(distances), np.array(snrs), np.array(masses), types, params_list

def validate_physical_constraints(params_list):
    """Check physical constraints on parameters"""
    issues = []
    
    for i, p in enumerate(params_list):
        if not isinstance(p, dict):
            continue
        
        # Mass ordering: m1 >= m2
        m1 = p.get('mass_1')
        m2 = p.get('mass_2')
        if m1 is not None and m2 is not None:
            if m1 < m2:
                issues.append(f"Sample {i}: m1 < m2 ({m1:.2f} < {m2:.2f})")
        
        # Spin magnitude constraint: |a| <= 0.99
        for spin_key in ['a1', 'a2']:
            spin = p.get(spin_key)
            if spin is not None:
                spin = float(spin)
                if spin > 0.99 or spin < 0.0:
                    issues.append(f"Sample {i}: {spin_key}={spin:.3f} outside [0, 0.99]")
        
        # Distance positivity
        d = p.get('luminosity_distance')
        if d is not None and float(d) <= 0:
            issues.append(f"Sample {i}: negative/zero distance {d}")
        
        # SNR positivity
        snr = p.get('target_snr')
        if snr is not None and float(snr) <= 0:
            issues.append(f"Sample {i}: negative/zero SNR {snr}")
    
    return issues

def print_separator(char='=', length=80):
    print(char * length)

def main():
    print_separator('=')
    print("FINAL DATASET VALIDATION")
    print_separator('=')
    
    # Load dataset
    distances, snrs, masses, types, params_list = load_dataset_from_output('train')
    
    if len(distances) == 0:
        print("❌ ERROR: No data found! Check data/output/train/")
        return False
    
    print(f"\n✅ Loaded {len(distances):,} samples\n")
    
    # Overall statistics
    print_separator('-')
    print("📊 OVERALL STATISTICS")
    print_separator('-')
    
    overall_corr = np.corrcoef(distances, snrs)[0, 1]
    
    print(f"Total samples:        {len(distances):,}")
    print(f"Distance mean:        {distances.mean():.0f} Mpc")
    print(f"Distance median:      {np.median(distances):.0f} Mpc")
    print(f"Distance std:         {distances.std():.0f} Mpc")
    print(f"Distance CV:          {distances.std()/distances.mean():.3f}")
    print(f"Distance range:       [{distances.min():.0f}, {distances.max():.0f}] Mpc")
    print(f"SNR mean:             {snrs.mean():.1f}")
    print(f"SNR median:           {np.median(snrs):.1f}")
    print(f"SNR range:            [{snrs.min():.1f}, {snrs.max():.1f}]")
    print(f"SNR-Distance corr:    {overall_corr:.3f} [target: -0.78]")
    
    # Per event-type statistics
    print(f"\n")
    print_separator('-')
    print("📊 PER EVENT-TYPE STATISTICS")
    print_separator('-')
    
    targets = {
        'BBH': {'mean': 1300, 'cv': 0.55, 'corr': -0.75},
        'BNS': {'mean': 130, 'cv': 0.55, 'corr': -0.75},
        'NSBH': {'mean': 400, 'cv': 0.55, 'corr': -0.75}
    }
    
    metrics = {}
    
    for event_type in ['BBH', 'BNS', 'NSBH']:
        mask = np.array([t == event_type for t in types])
        n = mask.sum()
        
        if n == 0:
            print(f"\n⚠️  {event_type}: No samples found!")
            continue
        
        d_et = distances[mask]
        snr_et = snrs[mask]
        m_et = masses[mask]
        
        mean_d = d_et.mean()
        cv = d_et.std() / mean_d if mean_d > 0 else 0
        corr = np.corrcoef(d_et, snr_et)[0, 1] if len(d_et) > 1 else 0
        
        metrics[event_type] = {
            'n': n,
            'mean': mean_d,
            'cv': cv,
            'corr': corr,
            'median': np.median(d_et),
            'min': d_et.min(),
            'max': d_et.max(),
            'snr_mean': snr_et.mean()
        }
        
        target = targets[event_type]
        mean_pct = (mean_d / target['mean']) * 100
        cv_pct = (cv / target['cv']) * 100
        corr_pct = (abs(corr) / abs(target['corr'])) * 100 if target['corr'] != 0 else 0
        
        # Status indicators
        mean_status = '✅' if 85 <= mean_pct <= 115 else '⚠️ ' if 70 <= mean_pct <= 130 else '❌'
        cv_status = '✅' if 85 <= cv_pct <= 115 else '⚠️ ' if 70 <= cv_pct <= 130 else '❌'
        corr_status = '✅' if corr_pct >= 90 else '⚠️ ' if corr_pct >= 80 else '❌'
        
        print(f"\n{event_type} (n={n:,}, {n/len(distances)*100:.1f}%)")
        print(f"  Distance:  mean={mean_d:7.0f} Mpc [target: {target['mean']:4.0f}] {mean_status} ({mean_pct:.0f}%)")
        print(f"             median={np.median(d_et):5.0f} Mpc")
        print(f"             range=[{d_et.min():5.0f}, {d_et.max():5.0f}] Mpc")
        print(f"  CV:        {cv:.3f} [target: {target['cv']:.2f}] {cv_status} ({cv_pct:.0f}%)")
        print(f"  Corr:      {corr:.3f} [target: {target['corr']:.2f}] {corr_status} ({corr_pct:.0f}%)")
        print(f"  SNR:       mean={snr_et.mean():.1f}, range=[{snr_et.min():.1f}, {snr_et.max():.1f}]")
        print(f"  Mass:      mean={m_et.mean():.1f} M☉, range=[{m_et.min():.1f}, {m_et.max():.1f}] M☉")
    
    # SNR regime distribution (for all event types)
    print(f"\n")
    print_separator('-')
    print("📊 SNR REGIME DISTRIBUTION")
    print_separator('-')
    
    regime_targets = {
        'weak (5-10)': (5, 10, 5.0),
        'low (10-20)': (10, 20, 35.0),
        'medium (20-40)': (20, 40, 45.0),
        'high (40-70)': (40, 70, 12.0),
        'loud (>70)': (70, 200, 3.0)
    }
    
    for et in ['BBH', 'BNS', 'NSBH']:
        et_mask = np.array([t == et for t in types])
        et_snrs = snrs[et_mask]
        
        if len(et_snrs) == 0:
            continue
        
        print(f"\n{et}:")
        for regime, (low, high, target_pct) in regime_targets.items():
            count = ((et_snrs >= low) & (et_snrs < high)).sum()
            actual_pct = (count / len(et_snrs)) * 100
            
            status = '✅' if abs(actual_pct - target_pct) < 5 else '⚠️ ' if abs(actual_pct - target_pct) < 10 else '❌'
            
            print(f"  {regime:16s}: {actual_pct:5.1f}% (target: {target_pct:5.1f}%) {status}")
    
    # Physical constraints validation
    print(f"\n")
    print_separator('-')
    print("🔬 PHYSICAL CONSTRAINTS CHECK")
    print_separator('-')
    
    issues = validate_physical_constraints(params_list)
    
    if len(issues) == 0:
        print("✅ All physical constraints satisfied!")
        constraints_pass = True
    else:
        print(f"⚠️  Found {len(issues)} constraint violations:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues)-10} more")
        constraints_pass = False
    
    # Data quality checks
    print(f"\n")
    print_separator('-')
    print("🔍 DATA QUALITY CHECKS")
    print_separator('-')
    
    # Check for NaN/Inf
    nan_count = np.isnan(distances).sum() + np.isnan(snrs).sum() + np.isnan(masses).sum()
    inf_count = np.isinf(distances).sum() + np.isinf(snrs).sum() + np.isinf(masses).sum()
    
    print(f"NaN values:           {nan_count} {'✅' if nan_count == 0 else '❌'}")
    print(f"Inf values:           {inf_count} {'✅' if inf_count == 0 else '❌'}")
    
    # Check for duplicates (approximately)
    unique_distances = len(np.unique(np.round(distances, 1)))
    duplicate_pct = (1 - unique_distances / len(distances)) * 100
    print(f"Duplicate distances:  {duplicate_pct:.2f}% {'✅' if duplicate_pct < 5 else '⚠️ '}")
    
    quality_pass = nan_count == 0 and inf_count == 0
    
    # Final verdict
    print(f"\n")
    print_separator('=')
    print("FINAL VERDICT")
    print_separator('=')
    
    all_pass = True
    checks = []
    
    # Check each event type
    for et in ['BBH', 'BNS', 'NSBH']:
        if et in metrics:
            m = metrics[et]
            t = targets[et]
            
            mean_ok = 0.70 <= m['mean']/t['mean'] <= 1.30
            cv_ok = 0.70 <= m['cv']/t['cv'] <= 1.30
            corr_ok = abs(m['corr']) >= 0.65
            
            checks.append(f"{et:4s} mean:        {'✅ PASS' if mean_ok else '❌ FAIL'}")
            checks.append(f"{et:4s} CV:          {'✅ PASS' if cv_ok else '❌ FAIL'}")
            checks.append(f"{et:4s} correlation: {'✅ PASS' if corr_ok else '⚠️ WARN' if corr_ok is False else '✅ PASS'}")
            
            all_pass = all_pass and mean_ok and cv_ok and corr_ok
    
    # Other checks
    checks.append(f"Physical constraints: {'✅ PASS' if constraints_pass else '⚠️ WARN'}")
    checks.append(f"Data quality:         {'✅ PASS' if quality_pass else '❌ FAIL'}")
    
    for check in checks:
        print(check)
    
    print()
    if all_pass and quality_pass:
        print("🎉 DATASET IS PRODUCTION READY! Proceed with training.")
        return True
    elif all_pass:
        print("⚠️  DATASET IS ACCEPTABLE with minor issues. Review before training.")
        return True
    else:
        print("❌ DATASET HAS ISSUES. Review and regenerate if necessary.")
        return False
    
    print_separator('=')

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
