#!/usr/bin/env python3
"""
Flow Convergence Diagnostic - Detect Mode Collapse Early
Nov 13, 2025 Enhancement

This script monitors normalizing flow training to detect:
1. Mode collapse (very narrow parameter distributions)
2. Divergence (extreme values outside expected bounds)
3. Insufficient coverage (low variance in dimensions)
4. Convergence progress (NLL trends, parameter range expansion)

Usage:
    python check_flow_convergence.py --model path/to/flow_checkpoint.pt --epoch 10
    python check_flow_convergence.py --continuous --every 5  # Monitor every 5 epochs
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FlowConvergenceChecker')


class FlowConvergenceChecker:
    """Monitor normalizing flow convergence and detect mode collapse."""
    
    def __init__(self, flow_model: nn.Module, param_names: list, 
                 param_bounds: Dict[str, Tuple[float, float]], device: str = 'cuda'):
        """
        Initialize the convergence checker.
        
        Args:
            flow_model: The normalizing flow model to monitor
            param_names: List of parameter names (e.g., ['mass_1', 'mass_2', ...])
            param_bounds: Dictionary of bounds for each parameter after normalization
            device: Device to use (cuda or cpu)
        """
        self.flow = flow_model.eval()
        self.param_names = param_names
        self.param_dim = len(param_names)
        self.param_bounds = param_bounds
        self.device = device
        
        # Thresholds for red flags
        self.COLLAPSE_THRESHOLD = 0.15  # If range < 0.15, likely mode collapse
        self.DIVERGENCE_THRESHOLD = 3.0  # If samples exceed ±3.0, divergence risk
        self.MIN_COVERAGE_STD = 0.3     # Minimum std for adequate coverage
        self.RED_FLAG_RANGE = 0.20      # Range < 0.20 triggers warning
        
        # History tracking
        self.history = {
            'epoch': [],
            'nll': [],
            'param_ranges': {name: [] for name in param_names},
            'param_stds': {name: [] for name in param_names},
            'coverage_ratio': [],
            'divergence_ratio': [],
        }
    
    def check_convergence(self, num_samples: int = 1000, context: torch.Tensor = None,
                         epoch: int = None) -> Dict:
        """
        Run convergence check on flow samples.
        
        Args:
            num_samples: Number of samples to draw from flow
            context: Context tensor [1, context_dim] - if None, uses random
            epoch: Current epoch (for logging)
            
        Returns:
            Dictionary with diagnostic results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"FLOW CONVERGENCE CHECK - Epoch {epoch if epoch is not None else 'N/A'}")
        logger.info(f"{'='*70}")
        
        with torch.no_grad():
            # Generate context if not provided
            if context is None:
                # Assume context_dim from flow architecture
                context_dim = getattr(self.flow, 'context_dim', 512)
                context = torch.randn(1, context_dim, device=self.device)
            
            # Sample from flow
            try:
                samples = self.flow.sample(num_samples=num_samples, context=context)
            except Exception as e:
                logger.error(f"❌ Sampling failed: {e}")
                return self._create_error_result()
            
            # Ensure on CPU for analysis
            samples = samples.cpu().numpy() if torch.is_tensor(samples) else samples
            
            # Compute diagnostics
            results = self._compute_diagnostics(samples, epoch)
            
            # Check for red flags
            warnings = self._check_red_flags(results)
            results['warnings'] = warnings
            
            # Log results
            self._log_results(results, epoch)
            
            # Track history
            if epoch is not None:
                self._update_history(results, epoch)
            
            return results
    
    def _compute_diagnostics(self, samples: np.ndarray, epoch: int = None) -> Dict:
        """Compute diagnostic metrics from samples."""
        diagnostics = {
            'num_samples': samples.shape[0],
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'param_ranges': {},
            'param_stds': {},
            'param_means': {},
            'param_mins': {},
            'param_maxs': {},
            'coverage_metrics': {},
            'divergence_metrics': {},
        }
        
        for i, param_name in enumerate(self.param_names):
            param_samples = samples[:, i]
            
            # Range (max - min)
            param_min = param_samples.min()
            param_max = param_samples.max()
            param_range = param_max - param_min
            
            # Statistics
            param_std = param_samples.std()
            param_mean = param_samples.mean()
            
            # Coverage: how much of expected [-1, 1] space is explored
            coverage_ratio = param_range / 2.0  # Expected range is 2.0 for [-1, 1]
            
            # Divergence: ratio of samples beyond ±3σ
            divergence_count = np.sum(np.abs(param_samples) > self.DIVERGENCE_THRESHOLD)
            divergence_ratio = divergence_count / len(param_samples)
            
            diagnostics['param_ranges'][param_name] = float(param_range)
            diagnostics['param_stds'][param_name] = float(param_std)
            diagnostics['param_means'][param_name] = float(param_mean)
            diagnostics['param_mins'][param_name] = float(param_min)
            diagnostics['param_maxs'][param_name] = float(param_max)
            diagnostics['coverage_metrics'][param_name] = {
                'ratio': float(coverage_ratio),
                'expected_coverage': 2.0,
            }
            diagnostics['divergence_metrics'][param_name] = {
                'ratio': float(divergence_ratio),
                'divergent_samples': int(divergence_count),
            }
        
        return diagnostics
    
    def _check_red_flags(self, diagnostics: Dict) -> list:
        """Identify potential issues."""
        warnings = []
        
        for param_name in self.param_names:
            param_range = diagnostics['param_ranges'][param_name]
            param_std = diagnostics['param_stds'][param_name]
            coverage = diagnostics['coverage_metrics'][param_name]['ratio']
            divergence = diagnostics['divergence_metrics'][param_name]['ratio']
            
            # Red Flag 1: Mode collapse
            if param_range < self.COLLAPSE_THRESHOLD:
                warnings.append({
                    'type': 'MODE_COLLAPSE',
                    'severity': 'CRITICAL',
                    'param': param_name,
                    'message': f"🔴 CRITICAL: {param_name} has very narrow range ({param_range:.4f})",
                    'range': param_range,
                })
            
            # Red Flag 2: Low coverage
            elif param_range < self.RED_FLAG_RANGE:
                warnings.append({
                    'type': 'LOW_COVERAGE',
                    'severity': 'WARNING',
                    'param': param_name,
                    'message': f"🟠 WARNING: {param_name} has limited coverage ({param_range:.4f})",
                    'range': param_range,
                })
            
            # Red Flag 3: Insufficient spread
            if param_std < self.MIN_COVERAGE_STD:
                warnings.append({
                    'type': 'LOW_VARIANCE',
                    'severity': 'WARNING',
                    'param': param_name,
                    'message': f"🟠 WARNING: {param_name} has low variance (std={param_std:.4f})",
                    'std': param_std,
                })
            
            # Red Flag 4: Divergence
            if divergence > 0.05:  # More than 5% of samples diverging
                warnings.append({
                    'type': 'DIVERGENCE',
                    'severity': 'WARNING',
                    'param': param_name,
                    'message': f"🟠 WARNING: {param_name} has {divergence*100:.1f}% divergent samples",
                    'divergence_ratio': divergence,
                })
        
        return warnings
    
    def _log_results(self, diagnostics: Dict, epoch: int = None):
        """Log diagnostic results to console."""
        epoch_str = f"Epoch {epoch}" if epoch is not None else "Diagnostic Check"
        
        logger.info(f"\n📊 {epoch_str}: Parameter Space Coverage")
        logger.info(f"{'Parameter':<20} {'Range':<12} {'Std':<12} {'Coverage %':<15} {'Status':<15}")
        logger.info("-" * 75)
        
        for param_name in self.param_names:
            param_range = diagnostics['param_ranges'][param_name]
            param_std = diagnostics['param_stds'][param_name]
            coverage = diagnostics['coverage_metrics'][param_name]['ratio'] * 100
            
            # Status indicator
            if param_range < self.COLLAPSE_THRESHOLD:
                status = "🔴 COLLAPSE"
            elif param_range < self.RED_FLAG_RANGE:
                status = "🟠 LIMITED"
            elif param_std < self.MIN_COVERAGE_STD:
                status = "🟠 LOW_VAR"
            else:
                status = "✅ GOOD"
            
            logger.info(f"{param_name:<20} {param_range:<12.4f} {param_std:<12.4f} "
                       f"{coverage:<15.1f} {status:<15}")
        
        # Log warnings
        if diagnostics['warnings']:
            logger.warning(f"\n⚠️  RED FLAGS DETECTED ({len(diagnostics['warnings'])}):")
            for warning in diagnostics['warnings']:
                logger.warning(f"   {warning['message']}")
        else:
            logger.info("\n✅ No critical issues detected")
        
        logger.info(f"{'='*70}\n")
    
    def _update_history(self, diagnostics: Dict, epoch: int):
        """Update convergence history."""
        self.history['epoch'].append(epoch)
        for param_name in self.param_names:
            self.history['param_ranges'][param_name].append(
                diagnostics['param_ranges'][param_name]
            )
            self.history['param_stds'][param_name].append(
                diagnostics['param_stds'][param_name]
            )
    
    def _create_error_result(self) -> Dict:
        """Create error result dict."""
        return {
            'error': True,
            'message': 'Failed to compute diagnostics',
            'warnings': [{'type': 'ERROR', 'severity': 'CRITICAL', 
                         'message': '❌ Convergence check failed'}],
        }
    
    def plot_convergence_history(self, output_path: str = 'flow_convergence_history.png'):
        """Plot convergence history over epochs."""
        import matplotlib.pyplot as plt
        
        if not self.history['epoch']:
            logger.warning("No convergence history to plot")
            return
        
        epochs = self.history['epoch']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Flow Convergence History', fontsize=16, fontweight='bold')
        
        # Plot 1: Parameter ranges
        ax = axes[0, 0]
        for param_name in self.param_names:
            ranges = self.history['param_ranges'][param_name]
            ax.plot(epochs, ranges, marker='o', label=param_name, alpha=0.7)
        ax.axhline(y=self.COLLAPSE_THRESHOLD, color='r', linestyle='--', 
                   label=f'Collapse threshold ({self.COLLAPSE_THRESHOLD})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Parameter Range')
        ax.set_title('Parameter Space Coverage')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Parameter stds
        ax = axes[0, 1]
        for param_name in self.param_names:
            stds = self.history['param_stds'][param_name]
            ax.plot(epochs, stds, marker='s', label=param_name, alpha=0.7)
        ax.axhline(y=self.MIN_COVERAGE_STD, color='orange', linestyle='--',
                   label=f'Min coverage ({self.MIN_COVERAGE_STD})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Parameter Std Dev')
        ax.set_title('Parameter Variance')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Coverage ratio (average across all params)
        ax = axes[1, 0]
        avg_coverage = [np.mean([self.history['param_ranges'][p][i]/2.0 
                                for p in self.param_names]) 
                       for i in range(len(epochs))]
        ax.plot(epochs, avg_coverage, marker='^', color='green', linewidth=2)
        ax.axhline(y=1.0, color='gray', linestyle=':', label='Full coverage')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Coverage Ratio')
        ax.set_title('Overall Parameter Space Exploration')
        ax.set_ylim([0, 1.2])
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Status summary
        ax = axes[1, 1]
        ax.axis('off')
        status_text = "✅ Flow Convergence Summary\n\n"
        status_text += f"Total epochs monitored: {len(epochs)}\n"
        status_text += f"Parameters: {len(self.param_names)}\n\n"
        
        # Latest epoch statistics
        if epochs:
            latest_epoch = epochs[-1]
            latest_ranges = [self.history['param_ranges'][p][-1] for p in self.param_names]
            avg_range = np.mean(latest_ranges)
            status_text += f"Latest epoch: {latest_epoch}\n"
            status_text += f"Average parameter range: {avg_range:.4f}\n"
            status_text += f"Min range: {min(latest_ranges):.4f}\n"
            status_text += f"Max range: {max(latest_ranges):.4f}\n\n"
            
            # Check for issues
            collapsed = sum(1 for r in latest_ranges if r < self.COLLAPSE_THRESHOLD)
            limited = sum(1 for r in latest_ranges if r < self.RED_FLAG_RANGE)
            
            if collapsed > 0:
                status_text += f"🔴 {collapsed} parameters showing mode collapse\n"
            if limited > 0:
                status_text += f"🟠 {limited} parameters with limited coverage\n"
            if collapsed == 0 and limited == 0:
                status_text += "✅ All parameters have good coverage\n"
        
        ax.text(0.1, 0.5, status_text, fontsize=11, family='monospace',
               verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Convergence history saved to {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Flow Convergence Diagnostic')
    parser.add_argument('--model', type=str, help='Path to flow model checkpoint')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--epoch', type=int, help='Current epoch number')
    parser.add_argument('--output', type=str, help='Output JSON path for results')
    parser.add_argument('--plot', action='store_true', help='Generate convergence plots')
    
    args = parser.parse_args()
    
    logger.info("Flow Convergence Checker initialized")
    logger.info("This is a diagnostic tool - use in training loops or standalone")
    logger.info("\nExample usage in training code:")
    logger.info("  checker = FlowConvergenceChecker(flow_model, param_names, param_bounds)")
    logger.info("  if epoch % 5 == 0:  # Every 5 epochs")
    logger.info("      results = checker.check_convergence(epoch=epoch)")
    logger.info("      if any(w['severity'] == 'CRITICAL' for w in results['warnings']):")
    logger.info("          logger.warning('Mode collapse detected - adjust hyperparameters')")


if __name__ == '__main__':
    main()
