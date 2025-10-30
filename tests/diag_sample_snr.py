"""
Simple diagnostic that samples ParameterSampler many times and prints regime fractions.
Run: python -m tests.diag_sample_snr (or pytest tests/diag_sample_snr.py)
"""

import collections
from ahsd.data.parameter_sampler import ParameterSampler


def main(n=50000):
    sampler = ParameterSampler()
    counts = collections.Counter()
    for _ in range(n):
        # draw regime using sampler internals
        regime = sampler._sample_snr_regime()
        counts[regime] += 1

    total = sum(counts.values())
    print(f"Sampled {total} target SNR regimes:\n")
    for r in sorted(sampler.snr_ranges.keys()):
        c = counts.get(r, 0)
        print(f"  {r:8s}: {c:8d} ({c/total*100:5.2f}%)")

    print("\nConfigured SNR_DISTRIBUTION:")
    print(sampler.snr_distribution)


if __name__ == '__main__':
    main()
