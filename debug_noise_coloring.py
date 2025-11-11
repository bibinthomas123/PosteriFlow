#!/usr/bin/env python3
"""
Debug script to understand why PSD coloring isn't working.
"""

import numpy as np
from ahsd.data.config import SAMPLE_RATE, DURATION

# Replicate the noise generation step-by-step
sample_rate = SAMPLE_RATE
duration = DURATION
n_samples = int(sample_rate * duration)

print(f"Parameters: {sample_rate} Hz, {duration}s, {n_samples} samples")
print("=" * 80)

# Step 1: Generate white Gaussian noise
np.random.seed(42)
white_noise = np.random.randn(n_samples)
print(f"\n1. White noise: min={np.min(white_noise):.3f}, max={np.max(white_noise):.3f}, std={np.std(white_noise):.3f}")

# Step 2: FFT to frequency domain
white_fft = np.fft.rfft(white_noise)
freqs = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
print(f"\n2. FFT: shape={white_fft.shape}, freq range=[{freqs[0]:.1f}, {freqs[-1]:.1f}] Hz")

# Step 3: Generate ASD (same as in default_asd)
def default_asd(frequencies):
    f = np.maximum(frequencies, 1.0)
    asd = np.zeros_like(f, dtype=float)
    
    # Low frequency: seismic wall
    mask_low = f <= 20
    asd[mask_low] = 1e-21 * (f[mask_low] / 10.0) ** (-2.07)
    
    # Transition
    mask_trans = (f > 20) & (f < 60)
    if np.any(mask_trans):
        f_trans = f[mask_trans]
        asd[mask_trans] = 1e-21 + (3e-24 - 1e-21) * ((f_trans - 20) / 40) ** 2
    
    # Mid
    mask_mid = (f >= 60) & (f <= 250)
    asd[mask_mid] = 3e-24 * (1 + 0.1 * np.log(f[mask_mid] / 100.0))
    
    # High trans
    mask_high_trans = (f > 250) & (f < 500)
    if np.any(mask_high_trans):
        f_high = f[mask_high_trans]
        asd[mask_high_trans] = 3e-24 * (1 + 0.5 * ((f_high - 250) / 250) ** 1.5)
    
    # Very high
    mask_vhigh = f >= 500
    asd[mask_vhigh] = 1e-23 * (f[mask_vhigh] / 200.0) ** 0.8
    
    asd = np.maximum(asd, 1e-24)
    return asd

asd = default_asd(freqs)
print(f"\n3. ASD: shape={asd.shape}")
print(f"   ASD stats: min={np.min(asd):.2e}, max={np.max(asd):.2e}, mean={np.mean(asd):.2e}")
print(f"   ASD samples at key frequencies:")
for f_val in [10, 50, 100, 250, 500, 1000, 2000]:
    idx = np.argmin(np.abs(freqs - f_val))
    print(f"     {freqs[idx]:7.1f} Hz: {asd[idx]:.2e}")

# Step 4: Apply coloring (the current fix)
print(f"\n4. Applying coloring: colored_fft = white_fft * asd")
colored_fft = white_fft * asd

# Check if scaling helps
print(f"   white_fft magnitude: min={np.min(np.abs(white_fft)):.2e}, max={np.max(np.abs(white_fft)):.2e}")
print(f"   colored_fft magnitude: min={np.min(np.abs(colored_fft)):.2e}, max={np.max(np.abs(colored_fft)):.2e}")

# Step 5: Transform back to time domain
colored_noise = np.fft.irfft(colored_fft, n=n_samples)
print(f"\n5. Inverse FFT:")
print(f"   colored_noise: min={np.min(colored_noise):.2e}, max={np.max(colored_noise):.2e}, std={np.std(colored_noise):.2e}")

# Step 6: Check PSD of colored noise
print(f"\n6. PSD of colored noise:")
colored_fft_check = np.fft.rfft(colored_noise)
psd = np.abs(colored_fft_check)**2 / n_samples
psd_in_band = psd[(freqs > 50) & (freqs < 2000)]
print(f"   PSD (50-2000 Hz): mean={np.mean(psd_in_band):.2e}, std={np.std(psd_in_band):.2e}")
print(f"   Ratio: {np.std(psd_in_band) / np.mean(psd_in_band):.4f}")

# Step 7: Expected PSD (what we want)
print(f"\n7. Expected PSD (squared ASD):")
expected_psd = asd**2
expected_in_band = expected_psd[(freqs > 50) & (freqs < 2000)]
print(f"   Expected PSD (50-2000 Hz): mean={np.mean(expected_in_band):.2e}, std={np.std(expected_in_band):.2e}")
print(f"   Ratio: {np.std(expected_in_band) / np.mean(expected_in_band):.4f}")

# DIAGNOSIS
print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

# The issue: when we multiply white noise FFT by ASD, we're not properly accounting for
# the magnitude of the white noise. The white noise FFT has huge dynamic range.

# Better approach: we should multiply by sqrt(PSD) or use a different formula
print("\nProblem: white_fft * asd doesn't properly apply power scaling")
print("The white_fft magnitude is ~4096 (sqrt of sample count)")
print(f"  actual white_fft max: {np.max(np.abs(white_fft)):.2e}")
print(f"  expected (sqrt(n_samples)): {np.sqrt(n_samples):.2e}")

print("\nWhen we multiply by tiny ASD values (1e-24 to 1e-21):")
print(f"  colored_fft becomes ~ 4096 * 1e-24 = {4096 * 1e-24:.2e}")
print(f"  actual colored_fft max: {np.max(np.abs(colored_fft)):.2e}")

print("\nAfter IRFFT, this becomes time-domain amplitude of ~1e-21")
print(f"  actual colored_noise std: {np.std(colored_noise):.2e}")

print("\nBUT: the resulting time-domain values are correct (1e-21 range)")
print("     the problem is the PSD appears flat because the noise is too small!")
print("     The numerical precision is lost - many frequencies become zero.")
