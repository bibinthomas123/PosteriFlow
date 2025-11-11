#!/usr/bin/env python3
"""Test that real noise is being properly cached and loaded."""

import logging
from pathlib import Path
import tempfile
from ahsd.data.noise_generator import RealNoiseGenerator

logging.basicConfig(level=logging.DEBUG)

# Test with temporary cache directory
with tempfile.TemporaryDirectory() as tmpdir:
    cache_dir = Path(tmpdir) / "noise_cache"
    
    print(f"\n{'='*80}")
    print("TEST 1: First initialization (should create cache)")
    print(f"{'='*80}\n")
    
    gen1 = RealNoiseGenerator(detector="H1", cache_dir=str(cache_dir))
    print(f"✓ RealNoiseGenerator created")
    print(f"  Cache directory: {cache_dir}")
    print(f"  Cached segments: {len(gen1.noise_segments)}")
    
    cache_file = cache_dir / "H1_segments_catalog.pkl"
    print(f"  Cache file exists: {cache_file.exists()}")
    print(f"  Cache file path: {cache_file}")
    
    if cache_file.exists():
        import pickle
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        print(f"  ✓ Cache file contains {len(cached)} segments")
    else:
        print(f"  ✗ Cache file NOT found - this is the issue!")
    
    print(f"\n{'='*80}")
    print("TEST 2: Second initialization (should load from cache)")
    print(f"{'='*80}\n")
    
    gen2 = RealNoiseGenerator(detector="H1", cache_dir=str(cache_dir))
    print(f"✓ RealNoiseGenerator created from cache")
    print(f"  Cached segments: {len(gen2.noise_segments)}")
    
    if len(gen2.noise_segments) > 0:
        print(f"  ✓ SUCCESS: Segments loaded from cache!")
    else:
        print(f"  ✗ FAILED: No segments loaded")

print("\n" + "="*80)
print("Tests complete!")
print("="*80)
