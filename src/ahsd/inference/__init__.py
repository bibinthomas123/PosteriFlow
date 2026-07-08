"""
Production inference pipeline for LeanNPE.

    from ahsd.inference import infer
    posterior = infer(event="GW150914", num_samples=10000)
    posterior.summary()
    posterior.corner(save="corner.png")
    posterior.save("results/GW150914")

Preprocessing exactly matches training (same BilbyPreprocessor for simulated
data; the noise-bank whitening recipe for real data). See pipeline.infer for
the full API and infer.py at the repo root for the CLI.
"""

from ahsd.inference.pipeline import infer, load_model
from ahsd.inference.preprocessing import (prepare_real, prepare_simulated,
                                          fetch_gwosc, PreparedData)
from ahsd.inference.result import PosteriorResult

__all__ = ["infer", "load_model", "prepare_real", "prepare_simulated",
           "fetch_gwosc", "PreparedData", "PosteriorResult"]
