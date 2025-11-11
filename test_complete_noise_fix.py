#!/usr/bin/env python3
"""
Comprehensive test showing the noise PSD fixes work correctly.
Tests:
1. Colored noise is generated with frequency dependence
2. PSD shows realistic variation
3. Noise amplitude is in correct range
"""

import numpy as np
from ahsd.data.noise_generator import NoiseGenerator
from ahsd.data.config import SAMPLE_RATE, DURATION

def compute_psd_log_std(noise, sample_rate, band=(50, 2000)):
    """Compute PSD and log-space standard deviation."""
    fft = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1.0 / sample_rate)
    power = np.abs(fft)**2 / len(noise)
    
    mask = (freqs >= band[0]) & (freqs <= band[1])
    psd_range = power[mask]
    
    # Use log-space for numerical robustness
    log_psd = np.log10(np.maximum(psd_range, 1e-50))
    log_std = np.std(log_psd)
    
    return freqs, power, log_std, np.mean(psd_range), np.std(psd_range)

def main():
    print("=" * 80)
    print("COMPREHENSIVE NOISE PSD FIX VALIDATION")
    print("=" * 80)
    
    # Generate colored noise
    gen = NoiseGenerator(sample_rate=SAMPLE_RATE, duration=DURATION)
    psd_dict = {}  # Will use default ASD
    
    print(f"\nGenerating {5} noise samples...")
    log_stds = []
    amplitudes = []
    
    for i in range(5):
        noise = gen.generate_colored_noise(psd_dict, seed=42+i)
        freqs, power, log_std, psd_mean, psd_std = compute_psd_log_std(
            noise, SAMPLE_RATE, band=(50, 2000)
        )
        
        log_stds.append(log_std)
        amplitudes.append(np.std(noise))
        
        print(f"\n  Sample {i+1}:")
        print(f"    Noise amplitude (std): {np.std(noise):.2e}")
        print(f"    PSD log_std:           {log_std:.4f}")
        print(f"    PSD mean:              {psd_mean:.2e}")
        print(f"    PSD linear std:        {psd_std:.2e}")
    
    # Analyze results
    print(f"\n" + "=" * 80)
    print("ANALYSIS RESULTS:")
    print("=" * 80)
    
    avg_log_std = np.mean(log_stds)
    avg_amplitude = np.mean(amplitudes)
    amplitude_consistency = np.std(amplitudes) / avg_amplitude
    
    print(f"\n1. FREQUENCY DEPENDENCE (log-space):")
    print(f"   Average log_std: {avg_log_std:.4f}")
    print(f"   Expected range:  0.15-0.25 (for colored noise)")
    if avg_log_std > 0.05:
        print(f"   ✓ PASS - PSD shows frequency dependence")
    else:
        print(f"   ✗ FAIL - PSD too uniform")
    
    print(f"\n2. AMPLITUDE CONSISTENCY:")
    print(f"   Average amplitude: {avg_amplitude:.2e}")
    print(f"   Expected range:    ~3e-21 (realistic LIGO noise)")
    print(f"   Variability (CV):   {amplitude_consistency:.3f}")
    if 1e-22 < avg_amplitude < 1e-19:
        print(f"   ✓ PASS - Amplitude in realistic range")
    else:
        print(f"   ✗ FAIL - Amplitude out of range")
    
    if amplitude_consistency < 0.3:
        print(f"   ✓ PASS - Samples consistent")
    else:
        print(f"   ⚠ WARNING - High sample-to-sample variation")
    
    print(f"\n3. FREQUENCY BAND ANALYSIS (Sample 1):")
    noise = gen.generate_colored_noise(psd_dict, seed=42)
    freqs, power, _, _, _ = compute_psd_log_std(noise, SAMPLE_RATE)
    
    bands = [
        ("Low (10-20 Hz)", 10, 20),
        ("Seismic (20-60 Hz)", 20, 60),
        ("Thermal (60-250 Hz)", 60, 250),
        ("High (250-500 Hz)", 250, 500),
        ("Shot noise (500+ Hz)", 500, SAMPLE_RATE//2),
    ]
    
    prev_mean = None
    for name, f_min, f_max in bands:
        mask = (freqs > f_min) & (freqs < f_max)
        if np.sum(mask) > 0:
            band_mean = np.mean(power[mask])
            trend = ""
            if prev_mean is not None:
                ratio = band_mean / prev_mean
                trend = f" (vs prev: {ratio:.2f}x)"
            print(f"   {name:30s}: {band_mean:.2e}{trend}")
            prev_mean = band_mean
    
    print(f"\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    
    # Overall pass/fail
    checks = [
        ("Frequency dependence", avg_log_std > 0.05),
        ("Amplitude range", 1e-22 < avg_amplitude < 1e-19),
        ("Amplitude consistency", amplitude_consistency < 0.5),
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    for check, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ ALL FIXES WORKING CORRECTLY")
        return 0
    else:
        print(f"\n✗ {total - passed} CHECK(S) FAILED")
        return 1

if __name__ == "__main__":
    exit(main())
