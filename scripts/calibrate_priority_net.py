#!/usr/bin/env python3
"""
Post-Training Calibration Script for PriorityNet

Fixes output range compression by fitting affine transform:
  calibrated_pred = gain * raw_pred + bias

Usage:
  python scripts/calibrate_priority_net.py \
    --model models/priority_net/priority_net_best.pth \
    --data_dir data/dataset \
    --method percentile \
    --output models/priority_net/calibrator.pt
"""

import sys
import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments"))

from train_priority_net import PriorityNet, PriorityNetDataset, ChunkedGWDataLoader
from src.ahsd.core.output_calibrator import calibrate_predictions, OutputCalibrator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
L = logging.getLogger("calibrate")


def load_prioritynet(checkpoint_path, device="cpu"):
    """Load PriorityNet from checkpoint"""
    if Path(checkpoint_path).is_dir():
        for fname in ["priority_net_best.pth", "checkpoint_best.pt", "model.pt"]:
            candidate = Path(checkpoint_path) / fname
            if candidate.exists():
                checkpoint_path = candidate
                break

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("model_config") or ckpt.get("config") or {}

    if "hidden_dims" not in cfg_dict or not cfg_dict["hidden_dims"]:
        sd = ckpt["model_state_dict"]
        dims = []
        w0 = sd.get("signal_encoder.input_embed.weight", None)
        if w0 is not None:
            dims.append(int(w0.shape[0]))
        i = 0
        while f"signal_encoder.blocks.{i}.linear.weight" in sd:
            dims.append(int(sd[f"signal_encoder.blocks.{i}.linear.weight"].shape[0]))
            i += 1
        if len(dims) >= 2 and dims[0] == dims[1]:
            dims = [dims[0]] + dims[2:]
        cfg_dict["hidden_dims"] = dims if dims else [512, 384, 256, 128]

    cfg_dict.setdefault("dropout", 0.2)
    cfg_dict.setdefault("use_strain", True)
    cfg_dict.setdefault("use_edge_conditioning", True)
    cfg_dict.setdefault("n_edge_types", 19)

    cfg = SimpleNamespace(**cfg_dict)
    model = PriorityNet(
        cfg,
        use_strain=cfg.use_strain,
        use_edge_conditioning=cfg.use_edge_conditioning,
        n_edge_types=cfg.n_edge_types,
    ).to(device).eval()

    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        L.info("✅ Loaded state_dict strict=True")
    except:
        msd, sd = model.state_dict(), ckpt["model_state_dict"]
        filtered = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        model.load_state_dict(filtered, strict=False)
        L.warning(f"⚠️  Loaded filtered: {len(filtered)} matched")

    return model


def collect_predictions(model, dataset, device="cpu", max_samples=None):
    """Collect predictions and targets from dataset"""
    preds_list = []
    targets_list = []

    count = 0
    with torch.no_grad():
        for i in range(min(max_samples or len(dataset), len(dataset))):
            sample = dataset[i]
            detections = sample["detections"]
            targets = sample["priorities"].numpy()

            # Predict
            pred, _ = model(detections)
            pred = pred.cpu().numpy()

            preds_list.extend(pred)
            targets_list.extend(targets)

            count += len(pred)
            if i % 100 == 0:
                L.info(f"  Processed {i+1} scenarios ({count} detections)")

    return np.array(preds_list), np.array(targets_list)


def main():
    ap = argparse.ArgumentParser(
        description="Post-training calibration for PriorityNet output range"
    )
    ap.add_argument("--model", required=True, help="Model checkpoint path")
    ap.add_argument("--data_dir", required=True, help="Dataset directory")
    ap.add_argument(
        "--split", default="validation", help="Dataset split (train/validation/test)"
    )
    ap.add_argument(
        "--method",
        default="percentile",
        choices=["percentile", "minmax", "learned"],
        help="Calibration method",
    )
    ap.add_argument("--max_samples", type=int, default=None, help="Max samples to use")
    ap.add_argument("--output", default=None, help="Output calibrator path")
    ap.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = ap.parse_args()

    # Load model
    L.info(f"Loading model from {args.model}")
    model = load_prioritynet(args.model, device=args.device)

    # Load dataset
    L.info(f"Loading {args.split} dataset from {args.data_dir}")
    loader = ChunkedGWDataLoader(
        args.data_dir, split=args.split, max_samples=args.max_samples, verbose=False
    )
    scenarios = loader.convert_to_priority_scenarios(create_overlaps=True)
    dataset = PriorityNetDataset(scenarios, args.split)
    L.info(f"✅ Loaded {len(dataset)} scenarios")

    # Collect predictions
    L.info("\n📊 Collecting predictions from model...")
    preds, targets = collect_predictions(model, dataset, args.device, args.max_samples)
    L.info(f"✅ Collected {len(preds)} predictions")

    # Calibrate
    calibrator, metrics = calibrate_predictions(
        preds, targets, method=args.method, val_split=0.2
    )

    # Save calibrator
    output_path = args.output or str(
        Path(args.model).parent / "output_calibrator.pt"
    )
    torch.save(
        {
            "gain": calibrator.gain.item(),
            "bias": calibrator.bias.item(),
            "method": args.method,
            "metrics": metrics,
        },
        output_path,
    )
    L.info(f"\n✅ Saved calibrator to {output_path}")

    # Summary
    L.info("\n" + "=" * 80)
    L.info("📈 CALIBRATION SUMMARY")
    L.info("=" * 80)
    L.info(f"Method: {args.method}")
    L.info(f"Gain (slope): {metrics['gain']:.4f}")
    L.info(f"Bias (offset): {metrics['bias']:.4f}")
    L.info(f"\nPrediction range improvement:")
    L.info(
        f"  Before: [{metrics['before']['pred_min']:.4f}, {metrics['before']['pred_max']:.4f}] (width={metrics['before']['pred_range']:.4f})"
    )
    L.info(
        f"  After:  [{metrics['after']['pred_min']:.4f}, {metrics['after']['pred_max']:.4f}] (width={metrics['after']['pred_range']:.4f})"
    )
    L.info(f"  Target: [{metrics['before']['targ_min']:.4f}, {metrics['before']['targ_max']:.4f}] (width={metrics['before']['targ_range']:.4f})")
    L.info(f"\nAccuracy improvement:")
    L.info(f"  MSE:  {metrics['before']['mse']:.6f} → {metrics['after']['mse']:.6f}")
    L.info(f"  MAE:  {metrics['before']['mae']:.6f} → {metrics['after']['mae']:.6f}")

    if metrics["before"]["mse"] > 0:
        mse_improvement = (
            (metrics["before"]["mse"] - metrics["after"]["mse"])
            / metrics["before"]["mse"]
            * 100
        )
        L.info(f"  MSE improvement: {mse_improvement:.1f}%")


if __name__ == "__main__":
    main()
