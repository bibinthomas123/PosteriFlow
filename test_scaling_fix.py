#!/usr/bin/env python3
"""
Test the scaling fix - check if it preserves frequency dependence.
"""

import numpy as np
from ahsd.data.noise_generator import NoiseGenerator
from ahsd.data.config import SAMPLE_RATE, DURATION

print(f"SAMPLE_RATE={SAMPLE_RATE}, DURATION={DURATION}")

# Generate noise
gen = NoiseGenerator(sample_rate=SAMPLE_RATE, duration=DURATION)
noise = gen.generate_colored_noise({}, seed=42)

print(f"\nNoise stats:")
print(f"  Std: {np.std(noise):.2e}")
print(f"  Min: {np.min(noise):.2e}")
print(f"  Max: {np.max(noise):.2e}")

# Compute PSD
fft = np.fft.rfft(noise)
freqs = np.fft.rfftfreq(len(noise), 1.0 / SAMPLE_RATE)
power = np.abs(fft)**2 / len(noise)

# Band analysis
print(f"\nPower at different frequencies:")
for f in [10, 100, 500, 1000, 2000]:
    idx = np.argmin(np.abs(freqs - f))
    print(f"  {f:4d} Hz: {power[idx]:.2e}")

# Compute log_std properly (avoiding zeros)
mask = (freqs > 50) & (freqs < 2000)
power_in_band = power[mask]

# Filter out exactly zero values
nonzero = power_in_band[power_in_band > 0]
print(f"\nPSD in 50-2000 Hz band:")
print(f"  Total bins: {len(power_in_band)}")
print(f"  Non-zero bins: {len(nonzero)}")
print(f"  Zero bins: {np.sum(power_in_band == 0)}")
print(f"  Mean (all): {np.mean(power_in_band):.2e}")
print(f"  Mean (non-zero): {np.mean(nonzero):.2e} if nonzero else 0")

if len(nonzero) > 10:
    log_power = np.log10(nonzero)
    log_std = np.std(log_power)
    print(f"  Log std (nonzero): {log_std:.4f}")
else:
    print(f"  Log std: N/A (not enough non-zero bins)")
