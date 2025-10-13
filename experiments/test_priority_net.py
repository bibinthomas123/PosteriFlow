#!/usr/bin/env python3
"""
Test script for trained PriorityNet on held-out test set.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import logging

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ahsd.core.prioritynet import PriorityNet
from ahsd.core.data_loader import ChunkedGWDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_priority_net(model_path, test_data_dir, output_dir):
    """Test trained PriorityNet on test set."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = PriorityNet(use_strain=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    logger.info(f"  Best val loss: {checkpoint['val_loss']:.5f}")
    logger.info(f"  Best val corr: {checkpoint['val_correlation']:.3f}")
    
    # Load test data
    logger.info(f"\nLoading test data from {test_data_dir}")
    test_loader = ChunkedGWDataLoader(
        dataset_path=test_data_dir,
        split='test',
        max_samples=None,
        shuffle_globally=False
    )
    
    test_scenarios = test_loader.convert_to_priority_scenarios(create_overlaps=True)
    logger.info(f"Created {len(test_scenarios)} test scenarios")
    
    # Test on scenarios
    logger.info(f"\nTesting on scenarios...")
    all_predictions = []
    all_targets = []
    all_errors = []
    scenario_correlations = []
    
    for scenario in test_scenarios:
        # Each scenario is already a dict with 'detections' and 'priorities'
        detections = scenario['detections']
        true_priorities = torch.tensor(scenario['priorities'], dtype=torch.float32)
        
        if len(detections) < 2:
            continue  # Skip single-detection scenarios
        
        with torch.no_grad():
            pred_priorities, uncertainties = model(detections)
        
        pred_np = pred_priorities.cpu().numpy()
        true_np = true_priorities.numpy()
        
        # Match lengths
        min_len = min(len(pred_np), len(true_np))
        pred_np = pred_np[:min_len]
        true_np = true_np[:min_len]
        
        # Compute correlation for this scenario
        if len(pred_np) >= 2:
            try:
                rho, _ = spearmanr(true_np, pred_np)
                if np.isfinite(rho):
                    scenario_correlations.append(rho)
            except Exception as e:
                pass
        
        # Accumulate for global metrics
        all_predictions.extend(pred_np.tolist())
        all_targets.extend(true_np.tolist())
        all_errors.extend(np.abs(pred_np - true_np).tolist())
    
    logger.info(f"Processed {len(scenario_correlations)} multi-signal scenarios")
    
    # Check if we have data
    if len(all_predictions) == 0:
        logger.warning("No predictions collected! Check if overlaps were created properly.")
        logger.warning("Test set may have only single-detection scenarios.")
        return None
    
    # Compute metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_errors = np.array(all_errors)
    
    global_corr, _ = spearmanr(all_targets, all_predictions)
    mae = np.mean(all_errors)
    rmse = np.sqrt(np.mean(all_errors**2))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total predictions: {len(all_predictions)}")
    logger.info(f"Multi-signal scenarios: {len(scenario_correlations)}")
    logger.info(f"\nGlobal Metrics:")
    logger.info(f"  Spearman correlation: {global_corr:.4f}")
    logger.info(f"  Mean Absolute Error:  {mae:.4f}")
    logger.info(f"  Root Mean Sq Error:   {rmse:.4f}")
    
    if len(scenario_correlations) > 0:
        logger.info(f"\nPer-Scenario Metrics:")
        logger.info(f"  Mean correlation: {np.mean(scenario_correlations):.4f}")
        logger.info(f"  Std correlation:  {np.std(scenario_correlations):.4f}")
        logger.info(f"  Min correlation:  {np.min(scenario_correlations):.4f}")
        logger.info(f"  Max correlation:  {np.max(scenario_correlations):.4f}")
    logger.info(f"{'='*60}\n")
    
    # Visualizations
    create_test_visualizations(
        all_predictions, all_targets, all_errors,
        scenario_correlations, global_corr, output_dir
    )
    
    # Save results
    results = {
        'global_correlation': float(global_corr),
        'mae': float(mae),
        'rmse': float(rmse),
        'scenario_correlations': {
            'mean': float(np.mean(scenario_correlations)) if scenario_correlations else 0.0,
            'std': float(np.std(scenario_correlations)) if scenario_correlations else 0.0,
            'min': float(np.min(scenario_correlations)) if scenario_correlations else 0.0,
            'max': float(np.max(scenario_correlations)) if scenario_correlations else 0.0
        },
        'n_predictions': len(all_predictions),
        'n_scenarios': len(scenario_correlations)
    }
    
    import json
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    return results


def create_test_visualizations(predictions, targets, errors, 
                               scenario_corrs, global_corr, output_dir):
    """Create test result visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Predicted vs True
    ax = axes[0, 0]
    ax.scatter(targets, predictions, alpha=0.5, s=20)
    ax.plot([targets.min(), targets.max()], 
            [targets.min(), targets.max()], 'r--', lw=2)
    ax.set_xlabel('True Priority')
    ax.set_ylabel('Predicted Priority')
    ax.set_title(f'Predicted vs True (ρ={global_corr:.3f})')
    ax.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax = axes[0, 1]
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(errors), color='r', linestyle='--', 
               label=f'Mean={np.mean(errors):.3f}')
    ax.axvline(np.median(errors), color='g', linestyle='--',
               label=f'Median={np.median(errors):.3f}')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Per-scenario correlation distribution
    ax = axes[1, 0]
    if len(scenario_corrs) > 0:
        ax.hist(scenario_corrs, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(scenario_corrs), color='r', linestyle='--',
                   label=f'Mean={np.mean(scenario_corrs):.3f}')
        ax.set_xlabel('Spearman Correlation')
        ax.set_ylabel('Count')
        ax.set_title('Per-Scenario Correlation Distribution')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No multi-signal scenarios', 
                ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    
    # 4. Residuals vs True
    ax = axes[1, 1]
    residuals = predictions - targets
    ax.scatter(targets, residuals, alpha=0.5, s=20)
    ax.axhline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('True Priority')
    ax.set_ylabel('Residual (Pred - True)')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {output_dir / 'test_results.png'}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='results/prioritynet_test',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    test_priority_net(args.model, args.data_dir, args.output_dir)
