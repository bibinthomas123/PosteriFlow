#!/usr/bin/env python3
"""
Benchmark TransformerStrainEncoder vs fallback Transformer.

Measures:
- Inference latency (ms per batch)
- Throughput (samples/sec)
- Memory usage (GPU)
- Speed comparison: Whisper vs Fallback
"""

import torch
import torch.cuda
import time
import sys
import logging
from pathlib import Path
import argparse
from typing import Dict, Tuple

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ahsd.models.transformer_encoder import TransformerStrainEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EncoderBenchmark:
    """Benchmark TransformerStrainEncoder variants."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.results = {}

    def benchmark_encoder(
        self,
        use_whisper: bool,
        batch_size: int = 16,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        input_length: int = 2048
    ) -> Dict[str, float]:
        """
        Benchmark encoder inference performance.

        Args:
            use_whisper: If True, use Whisper; else use fallback
            batch_size: Batch size for inference
            num_iterations: Number of inference iterations to benchmark
            warmup_iterations: Number of warmup iterations
            input_length: Time samples per detector

        Returns:
            Dict with 'latency_ms', 'throughput_samples_sec', 'memory_mb'
        """

        # Initialize encoder
        encoder = TransformerStrainEncoder(
            use_whisper=use_whisper,
            freeze_layers=4,
            input_length=input_length,
            n_detectors=2,
            output_dim=64
        )
        encoder = encoder.to(self.device).eval()

        # Create dummy input
        dummy_input = torch.randn(batch_size, 2, input_length, device=self.device)

        # Warmup
        logger.info(f"Warming up with {warmup_iterations} iterations...")
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = encoder(dummy_input)

        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Benchmark
        logger.info(f"Benchmarking with {num_iterations} iterations...")
        start_time = time.time()

        for _ in range(num_iterations):
            with torch.no_grad():
                _ = encoder(dummy_input)

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start_time

        # Calculate metrics
        total_samples = num_iterations * batch_size
        latency_ms = (elapsed / num_iterations) * 1000
        throughput = total_samples / elapsed

        # Memory estimate
        memory_mb = 0
        if self.device == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        return {
            "latency_ms": latency_ms,
            "throughput": throughput,
            "memory_mb": memory_mb,
            "total_time_sec": elapsed
        }

    def run_comparison(
        self,
        batch_sizes: list = None,
        num_iterations: int = 100
    ):
        """
        Run comprehensive benchmark comparing Whisper vs Fallback.

        Args:
            batch_sizes: List of batch sizes to test
            num_iterations: Number of iterations per test
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]

        logger.info(f"🚀 TransformerStrainEncoder Benchmark")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Input length: 2048 samples (1s @ 2048Hz)")
        logger.info(f"   Output dim: 64")
        print("=" * 90)

        for batch_size in batch_sizes:
            logger.info(f"\n📊 Batch size: {batch_size}")
            logger.info("-" * 90)

            results_whisper = None
            results_fallback = None

            try:
                logger.info("Testing Whisper encoder...")
                results_whisper = self.benchmark_encoder(
                    use_whisper=True,
                    batch_size=batch_size,
                    num_iterations=num_iterations
                )
            except Exception as e:
                logger.warning(f"Whisper benchmark failed: {e}")

            try:
                logger.info("Testing Lightweight Transformer...")
                results_fallback = self.benchmark_encoder(
                    use_whisper=False,
                    batch_size=batch_size,
                    num_iterations=num_iterations
                )
            except Exception as e:
                logger.warning(f"Fallback benchmark failed: {e}")

            # Print results
            print(f"{'Mode':<15} | {'Latency (ms)':<15} | {'Throughput':<15} | {'Memory (MB)':<12}")
            print("-" * 70)

            if results_whisper:
                print(
                    f"{'Whisper':<15} | "
                    f"{results_whisper['latency_ms']:>13.2f} | "
                    f"{results_whisper['throughput']:>13.1f} | "
                    f"{results_whisper['memory_mb']:>10.1f}"
                )

            if results_fallback:
                print(
                    f"{'Fallback':<15} | "
                    f"{results_fallback['latency_ms']:>13.2f} | "
                    f"{results_fallback['throughput']:>13.1f} | "
                    f"{results_fallback['memory_mb']:>10.1f}"
                )

            # Calculate speedup
            if results_whisper and results_fallback:
                speedup = results_fallback['latency_ms'] / results_whisper['latency_ms']
                print(f"\n⚡ Speedup: {speedup:.2f}x (Whisper is {speedup:.2f}x faster)")

            self.results[batch_size] = {
                "whisper": results_whisper,
                "fallback": results_fallback
            }

    def benchmark_with_masks(self, batch_size: int = 16, num_iterations: int = 100):
        """
        Benchmark with and without attention masks.
        """
        logger.info("\n🎭 Attention Mask Performance")
        print("=" * 90)

        encoder = TransformerStrainEncoder(
            use_whisper=True,
            freeze_layers=4,
            input_length=2048,
            output_dim=64
        )
        encoder = encoder.to(self.device).eval()

        dummy_input = torch.randn(batch_size, 2, 2048, device=self.device)
        attention_mask = torch.ones(batch_size, 32, device=self.device)  # 2048/64 = 32

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = encoder(dummy_input)
                _ = encoder(dummy_input, attention_mask=attention_mask)

        if self.device == "cuda":
            torch.cuda.synchronize()

        # Benchmark without mask
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = encoder(dummy_input)
        if self.device == "cuda":
            torch.cuda.synchronize()
        time_no_mask = time.time() - start

        # Benchmark with mask
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = encoder(dummy_input, attention_mask=attention_mask)
        if self.device == "cuda":
            torch.cuda.synchronize()
        time_with_mask = time.time() - start

        latency_no_mask = (time_no_mask / num_iterations) * 1000
        latency_with_mask = (time_with_mask / num_iterations) * 1000
        overhead = ((time_with_mask - time_no_mask) / time_no_mask) * 100

        print(f"No mask:     {latency_no_mask:>8.2f} ms/batch")
        print(f"With mask:   {latency_with_mask:>8.2f} ms/batch")
        print(f"Overhead:    {overhead:>8.2f}%")

    def benchmark_mixed_precision(self, batch_size: int = 16, num_iterations: int = 100):
        """
        Benchmark mixed precision (AMP) training speedup.
        """
        if not torch.cuda.is_available():
            logger.warning("Mixed precision benchmark requires CUDA")
            return

        logger.info("\n🎯 Mixed Precision (AMP) Performance")
        print("=" * 90)

        from torch.cuda.amp import autocast

        encoder = TransformerStrainEncoder(
            use_whisper=True,
            freeze_layers=4,
            input_length=2048,
            output_dim=64
        )
        encoder = encoder.to("cuda").eval()

        dummy_input = torch.randn(batch_size, 2, 2048, device="cuda")

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = encoder(dummy_input)
            with torch.no_grad():
                with autocast():
                    _ = encoder(dummy_input)

        torch.cuda.synchronize()

        # FP32
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = encoder(dummy_input)
        torch.cuda.synchronize()
        time_fp32 = time.time() - start

        # FP32 + AMP
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                with autocast():
                    _ = encoder(dummy_input)
        torch.cuda.synchronize()
        time_amp = time.time() - start

        latency_fp32 = (time_fp32 / num_iterations) * 1000
        latency_amp = (time_amp / num_iterations) * 1000
        speedup = time_fp32 / time_amp

        print(f"FP32:        {latency_fp32:>8.2f} ms/batch")
        print(f"AMP (FP16):  {latency_amp:>8.2f} ms/batch")
        print(f"Speedup:     {speedup:>8.2f}x")

    def print_summary(self):
        """Print benchmark summary."""
        logger.info("\n" + "=" * 90)
        logger.info("📈 SUMMARY")
        logger.info("=" * 90)

        for batch_size, results in self.results.items():
            if results["whisper"] and results["fallback"]:
                speedup = results["fallback"]["latency_ms"] / results["whisper"]["latency_ms"]
                logger.info(
                    f"Batch {batch_size:>2}: Whisper is {speedup:>5.2f}x faster than Fallback"
                )


def main():
    parser = argparse.ArgumentParser(description="Benchmark TransformerStrainEncoder")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32],
                       help="Batch sizes to test")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of benchmark iterations")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on (cuda/cpu)")
    parser.add_argument("--amp", action="store_true", help="Benchmark mixed precision")
    parser.add_argument("--masks", action="store_true", help="Benchmark attention masks")

    args = parser.parse_args()

    benchmark = EncoderBenchmark(device=args.device)

    # Run main benchmarks
    benchmark.run_comparison(batch_sizes=args.batch_sizes, num_iterations=args.iterations)

    # Optional: AMP benchmark
    if args.amp and torch.cuda.is_available():
        benchmark.benchmark_mixed_precision(num_iterations=args.iterations)

    # Optional: Mask benchmark
    if args.masks:
        benchmark.benchmark_with_masks(num_iterations=args.iterations)

    # Summary
    benchmark.print_summary()


if __name__ == "__main__":
    main()
