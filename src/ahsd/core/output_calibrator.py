#!/usr/bin/env python3
"""
Post-Training Output Calibration for PriorityNet

Calibrates model predictions to match target range without retraining.
Uses linear affine transformation: y = gain * x + bias

Three approaches:
1. Percentile matching: match pred quantiles to target quantiles
2. Min-max scaling: stretch pred range to target range
3. Learned calibration: train affine params on validation set
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

L = logging.getLogger("output_calibrator")


class OutputCalibrator(nn.Module):
    """Simple learned affine calibration: y = gain * x + bias"""

    def __init__(self, gain: float = 1.0, bias: float = 0.0, clamp: bool = True):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(gain))
        self.bias = nn.Parameter(torch.tensor(bias))
        self.clamp = clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gain * x + self.bias
        if self.clamp:
            y = torch.clamp(y, 0.0, 1.0)
        return y

    def calibrate_static(
        self, preds: np.ndarray, targets: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fit affine transform using percentile matching.

        Parameters
        ----------
        preds : np.ndarray
            Model predictions [N]
        targets : np.ndarray
            Target values [N]

        Returns
        -------
        gain, bias : float
            Calibration parameters
        """
        # Use 5th and 95th percentiles for robustness
        p05, p95 = 5, 95
        pred_p05 = np.percentile(preds, p05)
        pred_p95 = np.percentile(preds, p95)
        targ_p05 = np.percentile(targets, p05)
        targ_p95 = np.percentile(targets, p95)

        # Solve: targ_p05 = gain * pred_p05 + bias
        #        targ_p95 = gain * pred_p95 + bias
        if pred_p95 - pred_p05 < 1e-6:
            L.warning("Prediction range too small; using min-max instead")
            return self._calibrate_minmax(preds, targets)

        gain = (targ_p95 - targ_p05) / (pred_p95 - pred_p05)
        bias = targ_p05 - gain * pred_p05

        L.info(f"Percentile-matched calibration:")
        L.info(f"  Pred {p05}%={pred_p05:.4f}, {p95}%={pred_p95:.4f} → range={pred_p95-pred_p05:.4f}")
        L.info(f"  Targ {p05}%={targ_p05:.4f}, {p95}%={targ_p95:.4f} → range={targ_p95-targ_p05:.4f}")
        L.info(f"  Fitted: y = {gain:.4f} * x + {bias:.4f}")

        return float(gain), float(bias)

    def _calibrate_minmax(self, preds: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
        """Fallback: match min/max values"""
        pred_min, pred_max = preds.min(), preds.max()
        targ_min, targ_max = targets.min(), targets.max()

        gain = (targ_max - targ_min) / (pred_max - pred_min + 1e-8)
        bias = targ_min - gain * pred_min

        L.info(f"Min-max calibration:")
        L.info(f"  Pred range: [{pred_min:.4f}, {pred_max:.4f}]")
        L.info(f"  Targ range: [{targ_min:.4f}, {targ_max:.4f}]")
        L.info(f"  Fitted: y = {gain:.4f} * x + {bias:.4f}")

        return float(gain), float(bias)


def calibrate_predictions(
    model_preds: np.ndarray,
    targets: np.ndarray,
    method: str = "percentile",
    val_split: float = 0.2,
) -> Tuple[OutputCalibrator, dict]:
    """
    Calibrate model predictions post-training.

    Parameters
    ----------
    model_preds : np.ndarray
        Predictions from model [N]
    targets : np.ndarray
        Ground truth targets [N]
    method : str
        'percentile' (default), 'minmax', or 'learned'
    val_split : float
        Fraction for validation (only for 'learned')

    Returns
    -------
    calibrator : OutputCalibrator
        Callable calibration module
    metrics : dict
        Before/after metrics
    """
    L.info(f"\n{'='*80}\n📊 OUTPUT CALIBRATION ({method})\n{'='*80}")

    # Compute baseline metrics
    before = {
        "pred_mean": model_preds.mean(),
        "pred_std": model_preds.std(),
        "pred_min": model_preds.min(),
        "pred_max": model_preds.max(),
        "pred_range": model_preds.max() - model_preds.min(),
        "targ_mean": targets.mean(),
        "targ_std": targets.std(),
        "targ_min": targets.min(),
        "targ_max": targets.max(),
        "targ_range": targets.max() - targets.min(),
        "mse": np.mean((model_preds - targets) ** 2),
        "mae": np.mean(np.abs(model_preds - targets)),
    }

    L.info("Before calibration:")
    L.info(f"  Pred: mean={before['pred_mean']:.4f} std={before['pred_std']:.4f}")
    L.info(f"  Pred: range=[{before['pred_min']:.4f}, {before['pred_max']:.4f}]")
    L.info(f"  Targ: mean={before['targ_mean']:.4f} std={before['targ_std']:.4f}")
    L.info(f"  Targ: range=[{before['targ_min']:.4f}, {before['targ_max']:.4f}]")
    L.info(f"  Gap: {before['targ_range'] - before['pred_range']:.4f} (target_range - pred_range)")

    calibrator = OutputCalibrator()

    if method == "percentile":
        gain, bias = calibrator.calibrate_static(model_preds, targets)
        calibrator.gain.data = torch.tensor(gain)
        calibrator.bias.data = torch.tensor(bias)

    elif method == "minmax":
        gain, bias = calibrator._calibrate_minmax(model_preds, targets)
        calibrator.gain.data = torch.tensor(gain)
        calibrator.bias.data = torch.tensor(bias)

    elif method == "learned":
        # Split data
        n = len(model_preds)
        n_val = int(n * val_split)
        idx = np.random.permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        pred_train = torch.tensor(model_preds[train_idx], dtype=torch.float32)
        targ_train = torch.tensor(targets[train_idx], dtype=torch.float32)

        # Train calibrator
        opt = torch.optim.LBFGS(calibrator.parameters(), lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            cal_preds = calibrator(pred_train)
            loss = torch.nn.functional.mse_loss(cal_preds, targ_train)
            loss.backward()
            return loss

        opt.step(closure)

        L.info(f"Learned calibration on {len(train_idx)} samples")
        L.info(f"  Gain: {calibrator.gain.item():.4f}")
        L.info(f"  Bias: {calibrator.bias.item():.4f}")

    else:
        raise ValueError(f"Unknown calibration method: {method}")

    # Apply calibration
    with torch.no_grad():
        cal_preds = calibrator(torch.tensor(model_preds, dtype=torch.float32))
        cal_preds = cal_preds.cpu().numpy()

    after = {
        "pred_mean": cal_preds.mean(),
        "pred_std": cal_preds.std(),
        "pred_min": cal_preds.min(),
        "pred_max": cal_preds.max(),
        "pred_range": cal_preds.max() - cal_preds.min(),
        "mse": np.mean((cal_preds - targets) ** 2),
        "mae": np.mean(np.abs(cal_preds - targets)),
    }

    L.info("\nAfter calibration:")
    L.info(f"  Pred: mean={after['pred_mean']:.4f} std={after['pred_std']:.4f}")
    L.info(f"  Pred: range=[{after['pred_min']:.4f}, {after['pred_max']:.4f}]")
    L.info(f"  Gap reduction: {before['targ_range'] - before['pred_range']:.4f} → {before['targ_range'] - after['pred_range']:.4f}")
    L.info(f"  MSE: {before['mse']:.6f} → {after['mse']:.6f}")
    L.info(f"  MAE: {before['mae']:.6f} → {after['mae']:.6f}")

    metrics = {
        "before": before,
        "after": after,
        "gain": float(calibrator.gain.item()),
        "bias": float(calibrator.bias.item()),
    }

    return calibrator, metrics
