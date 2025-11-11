#!/usr/bin/env python3
"""
Quick test to verify PSD coloring is working after the fix.
"""

import numpy as np
from ahsd.data.noise_generator import NoiseGenerator
from ahsd.data.config import SAMPLE_RATE, DURATION

def compute_psd(noise, sample_rate):
    """Compute power spectral density of noise signal."""
    # FFT
    fft = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1.0 / sample_rate)
    
    # Power (magnitude squared)
    power = np.abs(fft)**2 / len(noise)
    
    return freqs, power

def analyze_psd():
    """Analyze PSD of generated noise."""
    print("=" * 80)
    print("NOISE PSD ANALYSIS - TESTING FIX")
    print("=" * 80)
    
    # Generate colored noise
    gen = NoiseGenerator(sample_rate=SAMPLE_RATE, duration=DURATION)
    
    # Create a simple PSD dict (will use default ASD fallback)
    psd_dict = {}
    
    print(f"\n1. Generating colored noise ({SAMPLE_RATE} Hz, {DURATION}s)...")
    noise = gen.generate_colored_noise(psd_dict, seed=42)
    print(f"   Shape: {noise.shape}, dtype: {noise.dtype}")
    print(f"   Min: {np.min(noise):.2e}, Max: {np.max(noise):.2e}, Mean: {np.mean(noise):.2e}")
    print(f"   Std: {np.std(noise):.2e}")
    
    # Compute PSD
    print(f"\n2. Computing PSD...")
    freqs, power = compute_psd(noise, SAMPLE_RATE)
    
    # Analyze in 50-2000 Hz band (where we check for flatness)
    mask = (freqs > 50) & (freqs < 2000)
    psd_in_band = power[mask]
    freqs_in_band = freqs[mask]
    
    psd_mean = np.mean(psd_in_band)
    psd_std = np.std(psd_in_band)
    psd_min = np.min(psd_in_band)
    psd_max = np.max(psd_in_band)
    
    print(f"   Frequency band: 50-2000 Hz ({np.sum(mask)} samples)")
    print(f"   PSD mean: {psd_mean:.2e}")
    print(f"   PSD std:  {psd_std:.2e}")
    print(f"   PSD min:  {psd_min:.2e}")
    print(f"   PSD max:  {psd_max:.2e}")
    
    # Check ratio (this was 0.0000 before the fix, should be > 0.2 now)
    ratio = psd_std / psd_mean if psd_mean > 0 else 0
    print(f"   Ratio (std/mean): {ratio:.4f}")
    
    # Check frequency dependence at specific bands
    print(f"\n3. Frequency band analysis:")
    bands = [
        ("Low (10-20 Hz)", 10, 20),
        ("Seismic transition (20-60 Hz)", 20, 60),
        ("Thermal (60-250 Hz)", 60, 250),
        ("High transition (250-500 Hz)", 250, 500),
        ("Shot noise (>500 Hz)", 500, SAMPLE_RATE//2),
    ]
    
    for name, f_min, f_max in bands:
        mask_band = (freqs > f_min) & (freqs < f_max)
        if np.sum(mask_band) > 0:
            psd_band = power[mask_band]
            mean_band = np.mean(psd_band)
            print(f"   {name:30s}: {mean_band:.2e}")
    
    # Results
    print(f"\n4. Validation:")
    passed = ratio > 0.2
    if passed:
        print(f"   ✓ PASS - PSD shows realistic frequency dependence (ratio={ratio:.4f})")
    else:
        print(f"   ✗ FAIL - PSD still too flat (ratio={ratio:.4f}, need > 0.2)")
    
    print("=" * 80)
    return passed

if __name__ == "__main__":
    try:
        passed = analyze_psd()
        exit(0 if passed else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
