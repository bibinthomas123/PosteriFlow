#!/usr/bin/env python3
"""
Unified Inference Pipeline for OverlapNeuralPE

Combines neural posterior estimation with bias correction and quality assessment.
Provides clean API for inference testing and deployment.

Usage:
    pipeline = InferencePipeline('models/neural_pe/best_model.pth', 'configs/enhanced_training.yaml')
    result = pipeline.extract(strain_data)
    posteriors = pipeline.get_posteriors(strain_data, n_samples=1000)
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
from dataclasses import dataclass

# Add project root to path
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ahsd.models.overlap_neuralpe import OverlapNeuralPE
from src.ahsd.models.rl_controller import AdaptiveComplexityController


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""

    device: str = "cuda"
    batch_size: int = 1
    n_posterior_samples: int = 1000
    return_posterior_samples: bool = False
    compute_credible_intervals: bool = True
    credibility: float = 0.90
    verbose: bool = False
    # RL controller settings
    use_rl_controller: bool = True
    rl_controller_path: Optional[str] = None
    rl_state_features: List[str] = None
    rl_complexity_levels: List[str] = None


class InferencePipeline:
    """
    High-level inference pipeline for gravitational wave signal extraction.

    Provides unified interface to OverlapNeuralPE for:
    - Iterative signal extraction with adaptive subtraction
    - Posterior distribution sampling
    - Parameter uncertainty quantification
    - Credible interval computation
    - Quality metrics
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: Optional[str] = None,
        inference_config: Optional[InferenceConfig] = None,
    ):
        """
        Initialize inference pipeline.

        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration YAML
            device: Device to run on ('cuda' or 'cpu')
            inference_config: Optional inference configuration
        """
        self.logger = logging.getLogger(__name__)

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.inference_config = inference_config or InferenceConfig(device=str(self.device))
        self.logger.info(f"Using device: {self.device}")

        # Load configuration
        config_path = Path(config_path)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.logger.info(f"Loaded config from {config_path}")

        # Parameter names
        param_names = [
            "mass_1",
            "mass_2",
            "luminosity_distance",
            "ra",
            "dec",
            "theta_jn",
            "psi",
            "phase",
            "geocent_time",
        ]

        # Priority net path
        priority_net_path = Path("models/priority_net_checkpoint.pt")

        # Initialize model
        self.model = OverlapNeuralPE(
            param_names=param_names,
            priority_net_path=str(priority_net_path),
            config=self.config,
            device=str(self.device),
        )

        # Load checkpoint
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # ✅ PyTorch 2.6+ requires weights_only=False for legacy checkpoints
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load with shape compatibility handling
        try:
            incompatible = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                self.logger.warning(
                    f"Checkpoint compatibility: {len(incompatible.missing_keys)} missing, "
                    f"{len(incompatible.unexpected_keys)} unexpected keys"
                )
        except RuntimeError as e:
            # Handle shape mismatches by loading only compatible keys
            self.logger.warning(f"Shape mismatch in checkpoint: {str(e)[:200]}...")
            self.logger.info("Attempting selective load of compatible weights...")
            
            state_dict = self.model.state_dict()
            checkpoint_dict = checkpoint["model_state_dict"]
            loaded_count = 0
            
            for key in list(checkpoint_dict.keys()):
                if key in state_dict:
                    if state_dict[key].shape == checkpoint_dict[key].shape:
                        state_dict[key] = checkpoint_dict[key]
                        loaded_count += 1
            
            self.logger.info(f"Selectively loaded {loaded_count} compatible weights")
            self.model.load_state_dict(state_dict)
        
        self.model.eval()
        
        # ✅ Load RL controller state if present in checkpoint (it's now nn.Module)
        if hasattr(self.model, "rl_controller") and self.model.rl_controller is not None:
            try:
                rl_state_dict = checkpoint.get("rl_controller_state_dict")
                if rl_state_dict:
                    self.model.rl_controller.load_state_dict(rl_state_dict)
                    self.logger.info("✅ Loaded RL controller state from checkpoint")
                else:
                    self.logger.info("⊘ No RL controller state in checkpoint (training from scratch)")
            except Exception as e:
                self.logger.warning(f"Failed to load RL controller state: {e}")
        
        self.logger.info(f"Loaded model from {model_path}")

        # Parameter names for reference
        self.param_names = [
            "mass_1",
            "mass_2",
            "luminosity_distance",
            "ra",
            "dec",
            "theta_jn",
            "psi",
            "phase",
            "geocent_time",
        ]

        # Initialize RL controller for adaptive complexity
        self.rl_controller = None
        if self.inference_config.use_rl_controller:
            state_features = self.inference_config.rl_state_features or [
                "remaining_signals",
                "residual_power",
                "processing_time",
                "current_snr",
                "extraction_success_rate",
            ]
            complexity_levels = self.inference_config.rl_complexity_levels or [
                "low",
                "medium",
                "high",
            ]

            self.rl_controller = AdaptiveComplexityController(
                state_features=state_features, complexity_levels=complexity_levels
            )

            # Load saved RL controller if available
            if self.inference_config.rl_controller_path:
                rl_path = Path(self.inference_config.rl_controller_path)
                if rl_path.exists():
                    self.rl_controller.load_model(str(rl_path))
                    self.logger.info(f"Loaded RL controller from {rl_path}")

            self.logger.info(f"Initialized RL controller with {len(state_features)} features")

    def _get_pipeline_state(
        self, strain_data: torch.Tensor, extraction_result: Dict
    ) -> Dict[str, float]:
        """
        Build pipeline state from strain data and extraction results.

        Args:
            strain_data: Input strain tensor
            extraction_result: Result from signal extraction

        Returns:
            State dict for RL controller
        """
        state = {
            "remaining_signals": float(len(extraction_result.get("extracted_signals", []))),
            "residual_power": float(
                torch.mean(extraction_result.get("residual", strain_data) ** 2).item()
            ),
            "processing_time": float(extraction_result.get("metrics", {}).get("total_time", 0.0)),
            "current_snr": float(
                np.mean(extraction_result.get("metrics", {}).get("snr_values", [0.0]))
            ),
            "extraction_success_rate": float(
                extraction_result.get("metrics", {}).get("extraction_quality", 0.0)
            ),
        }
        return state

    def _apply_rl_adjusted_extraction(self, strain_data: torch.Tensor) -> Dict[str, Any]:
        """
        Extract signals with RL-controlled complexity adaptation.

        Args:
            strain_data: [batch, n_detectors, n_samples] strain tensor

        Returns:
            Extraction result with RL-based complexity management
        """
        if not self.rl_controller:
            # Fall back to standard extraction
            return self.model.extract_overlapping_signals(
                strain_data=strain_data, true_params=None, training=False
            )

        # Run initial extraction
        result = self.model.extract_overlapping_signals(
            strain_data=strain_data, true_params=None, training=False
        )

        # Get pipeline state and RL recommendation
        pipeline_state = self._get_pipeline_state(strain_data, result)
        complexity_level = self.rl_controller.get_complexity_level(pipeline_state, training=False)

        self.logger.info(f"RL controller recommends complexity: {complexity_level}")
        self.logger.debug(f"Pipeline state: {pipeline_state}")

        # Apply complexity-based refinement
        result["rl_complexity_level"] = complexity_level
        result["rl_pipeline_state"] = pipeline_state
        result["rl_metrics"] = self.rl_controller.get_metrics()

        # High complexity: apply additional refinement
        if complexity_level == "high":
            self.logger.debug("Applying high-complexity refinement...")
            # Could trigger additional signal processing, more posterior samples, etc.
            result["refined"] = True
        elif complexity_level == "low":
            self.logger.debug("Using low-complexity fast extraction...")
            result["refined"] = False

        return result

    def extract(
        self,
        strain_data: torch.Tensor,
        return_posterior_samples: Optional[bool] = None,
        use_rl_adaptation: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract signals from strain data using iterative neural posterior estimation.

        Args:
            strain_data: [batch, n_detectors, n_samples] strain tensor
            return_posterior_samples: Override config for returning samples
            use_rl_adaptation: Apply RL-based complexity adaptation

        Returns:
            {
                'extracted_signals': List of extracted signal dicts,
                'residual': Remaining strain after extraction,
                'n_iterations': Number of extraction iterations,
                'metrics': {
                    'total_time': float,
                    'extraction_quality': float,
                    'snr_values': List[float]
                },
                'posterior_samples': (optional) [batch, n_samples, 9],
                'rl_complexity_level': (optional) complexity recommendation,
                'rl_pipeline_state': (optional) pipeline state used by RL
            }
        """
        if strain_data.device != self.device:
            strain_data = strain_data.to(self.device)

        self.logger.info(f"Extracting signals from strain data: {strain_data.shape}")

        with torch.no_grad():
            # Run extraction with optional RL adaptation
            if use_rl_adaptation and self.rl_controller:
                result = self._apply_rl_adjusted_extraction(strain_data)
            else:
                result = self.model.extract_overlapping_signals(
                    strain_data=strain_data, true_params=None, training=False
                )

            # Get posterior samples if requested
            return_samples = (
                return_posterior_samples or self.inference_config.return_posterior_samples
            )
            if return_samples:
                self.logger.info(
                    f"Sampling posterior ({self.inference_config.n_posterior_samples} samples)..."
                )
                posterior = self.model.sample_posterior(
                    strain_data=strain_data, n_samples=self.inference_config.n_posterior_samples
                )
                result["posterior_samples"] = posterior["samples"]
                result["posterior_means"] = posterior["means"]
                result["posterior_stds"] = posterior["stds"]

            return result

    def get_posteriors(
        self, strain_data: torch.Tensor, n_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get posterior distribution over parameters.

        Args:
            strain_data: [batch, n_detectors, n_samples] strain tensor
            n_samples: Number of posterior samples (uses config default if None)

        Returns:
            {
                'samples': [batch, n_samples, 9],
                'means': [batch, 9],
                'stds': [batch, 9],
                'context': [batch, context_dim]
            }
        """
        if strain_data.device != self.device:
            strain_data = strain_data.to(self.device)

        n_samples = n_samples or self.inference_config.n_posterior_samples

        self.logger.info(f"Sampling posterior: {n_samples} samples")

        with torch.no_grad():
            posterior = self.model.sample_posterior(strain_data=strain_data, n_samples=n_samples)
            return posterior

    def get_credible_intervals(
        self, strain_data: torch.Tensor, credibility: Optional[float] = None, n_samples: int = 5000
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute credible intervals for parameters.

        Args:
            strain_data: [batch, n_detectors, n_samples] strain tensor
            credibility: Credibility level (uses config default if None)
            n_samples: Number of posterior samples for estimation

        Returns:
            {
                'mass_1': {'lower': float, 'median': float, 'upper': float},
                'mass_2': {...},
                ...
            }
        """
        credibility = credibility or self.inference_config.credibility

        self.logger.info(f"Computing {credibility:.0%} credible intervals (n_samples={n_samples})")

        # Get posterior samples
        posterior = self.get_posteriors(strain_data, n_samples=n_samples)
        samples = posterior["samples"][0].cpu().numpy()  # [n_samples, 9]

        intervals = {}
        lower_percentile = (1 - credibility) / 2 * 100
        upper_percentile = (1 + credibility) / 2 * 100

        for i, param_name in enumerate(self.param_names):
            param_samples = samples[:, i]
            intervals[param_name] = {
                "lower": float(np.percentile(param_samples, lower_percentile)),
                "median": float(np.percentile(param_samples, 50)),
                "upper": float(np.percentile(param_samples, upper_percentile)),
                "mean": float(np.mean(param_samples)),
                "std": float(np.std(param_samples)),
            }

        return intervals

    def get_posterior_statistics(
        self, strain_data: torch.Tensor, n_samples: int = 5000
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute detailed posterior statistics.

        Args:
            strain_data: [batch, n_detectors, n_samples] strain tensor
            n_samples: Number of posterior samples

        Returns:
            {
                'mass_1': {
                    'mean': float,
                    'median': float,
                    'std': float,
                    'skewness': float,
                    'kurtosis': float,
                    'min': float,
                    'max': float
                },
                ...
            }
        """
        from scipy import stats

        posterior = self.get_posteriors(strain_data, n_samples=n_samples)
        samples = posterior["samples"][0].cpu().numpy()  # [n_samples, 9]

        statistics = {}
        for i, param_name in enumerate(self.param_names):
            param_samples = samples[:, i]
            statistics[param_name] = {
                "mean": float(np.mean(param_samples)),
                "median": float(np.median(param_samples)),
                "std": float(np.std(param_samples)),
                "skewness": float(stats.skew(param_samples)),
                "kurtosis": float(stats.kurtosis(param_samples)),
                "min": float(np.min(param_samples)),
                "max": float(np.max(param_samples)),
            }

        return statistics

    def compare_to_truth(
        self, strain_data: torch.Tensor, true_params: np.ndarray, n_samples: int = 5000
    ) -> Dict[str, Any]:
        """
        Compare posterior to ground truth parameters.

        Args:
            strain_data: [batch, n_detectors, n_samples] strain tensor
            true_params: [9,] array of true parameters
            n_samples: Number of posterior samples

        Returns:
            {
                'parameters': {
                    'mass_1': {
                        'true': 35.0,
                        'posterior_mean': 34.8,
                        'posterior_median': 34.9,
                        'within_credible_interval': True,
                        'error': 0.2,
                        'fractional_error': 0.006
                    },
                    ...
                },
                'global_metrics': {
                    'parameters_in_ci': 7/9,
                    'mean_error': 0.5,
                    'rms_error': 0.6
                }
            }
        """
        intervals = self.get_credible_intervals(strain_data, n_samples=n_samples)
        posterior = self.get_posteriors(strain_data, n_samples=n_samples)

        comparison = {"parameters": {}, "global_metrics": {}}

        errors = []
        in_ci = 0

        for i, param_name in enumerate(self.param_names):
            true_val = float(true_params[i])
            ci = intervals[param_name]
            posterior_mean = posterior["means"][0, i].item()

            # Check if within credible interval
            within_ci = ci["lower"] <= true_val <= ci["upper"]

            # Compute errors
            error = posterior_mean - true_val
            frac_error = error / (abs(true_val) + 1e-10)

            comparison["parameters"][param_name] = {
                "true": true_val,
                "posterior_mean": posterior_mean,
                "posterior_median": ci["median"],
                "within_credible_interval": bool(within_ci),
                "credible_interval": {"lower": ci["lower"], "upper": ci["upper"]},
                "error": error,
                "fractional_error": frac_error,
            }

            errors.append(abs(error))
            if within_ci:
                in_ci += 1

        comparison["global_metrics"] = {
            "parameters_in_ci": in_ci / len(self.param_names),
            "mean_absolute_error": float(np.mean(errors)),
            "rms_error": float(np.sqrt(np.mean(np.array(errors) ** 2))),
            "max_error": float(np.max(errors)),
            "max_fractional_error": float(
                np.max(
                    np.abs(
                        [comparison["parameters"][p]["fractional_error"] for p in self.param_names]
                    )
                )
            ),
        }

        return comparison

    def get_model_metrics(self) -> Dict[str, float]:
        """Get current model and component metrics."""
        metrics = self.model.get_metrics()

        # Add RL controller metrics if available
        if self.rl_controller:
            rl_metrics = self.rl_controller.get_metrics()
            metrics["rl_controller"] = rl_metrics

        return metrics

    def get_rl_metrics(self) -> Optional[Dict[str, float]]:
        """Get RL controller metrics."""
        if not self.rl_controller:
            return None
        return self.rl_controller.get_metrics()

    def save_rl_controller(self, filepath: str) -> None:
        """
        Save trained RL controller.

        Args:
            filepath: Path to save controller model
        """
        if self.rl_controller:
            self.rl_controller.save_model(filepath)
            self.logger.info(f"Saved RL controller to {filepath}")
        else:
            self.logger.warning("No RL controller to save")

    @staticmethod
    def from_checkpoint(
        checkpoint_path: str, config_path: str, device: Optional[str] = None
    ) -> "InferencePipeline":
        """
        Create InferencePipeline from checkpoint path.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to configuration YAML
            device: Device to use

        Returns:
            InferencePipeline instance
        """
        return InferencePipeline(checkpoint_path, config_path, device=device)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def load_strain_data(
    strain_dict: Dict[str, np.ndarray], target_length: int = 16384
) -> torch.Tensor:
    """
    Load strain data from detector dictionary.

    Args:
        strain_dict: {'H1': array, 'L1': array, ...}
        target_length: Expected sample length

    Returns:
        [1, n_detectors, target_length] tensor
    """
    detector_names = ["H1", "L1", "V1"]
    detector_arrays = []

    for det_name in detector_names:
        if det_name in strain_dict:
            strain = np.array(strain_dict[det_name])

            # Pad or truncate to target length
            if len(strain) < target_length:
                strain = np.pad(strain, (0, target_length - len(strain)))
            elif len(strain) > target_length:
                strain = strain[:target_length]

            detector_arrays.append(strain)

    strain_array = np.stack(detector_arrays)  # [n_detectors, n_samples]
    return torch.tensor(strain_array, dtype=torch.float32).unsqueeze(0)


def load_parameters_from_dict(param_dict: Dict[str, float]) -> np.ndarray:
    """
    Load parameters from dictionary.

    Args:
        param_dict: {'mass_1': 35.0, 'mass_2': 30.0, ...}

    Returns:
        [9,] array in standard order
    """
    param_names = [
        "mass_1",
        "mass_2",
        "luminosity_distance",
        "ra",
        "dec",
        "theta_jn",
        "psi",
        "phase",
        "geocent_time",
    ]

    return np.array([param_dict.get(name, 0.0) for name in param_names])


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference with OverlapNeuralPE + RL adaptation"
    )
    parser.add_argument("--model", type=str, default="models/neural_pe/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/enhanced_training.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument(
        "--rl-controller", type=str, default=None, help="Path to saved RL controller model"
    )
    parser.add_argument(
        "--use-rl", action="store_true", default=True, help="Enable RL-based adaptive complexity"
    )
    parser.add_argument(
        "--save-rl", type=str, default=None, help="Path to save RL controller after inference"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Initialize pipeline with RL controller
    inference_config = InferenceConfig(
        n_posterior_samples=args.n_samples,
        device=args.device,
        use_rl_controller=args.use_rl,
        rl_controller_path=args.rl_controller,
    )

    pipeline = InferencePipeline(
        args.model, args.config, device=args.device, inference_config=inference_config
    )

    # Generate test data
    strain_h1 = torch.randn(1, 16384) * 1e-23
    strain_l1 = torch.randn(1, 16384) * 1e-23
    strain_data = torch.stack([strain_h1, strain_l1], dim=1)

    print("Running inference with RL adaptation...")
    result = pipeline.extract(
        strain_data, return_posterior_samples=True, use_rl_adaptation=args.use_rl
    )
    print(f"✅ Extracted {len(result['extracted_signals'])} signals")

    # Show RL controller metrics
    if "rl_complexity_level" in result:
        print(f"  RL Complexity Level: {result['rl_complexity_level']}")

    if pipeline.rl_controller:
        rl_metrics = pipeline.get_rl_metrics()
        if rl_metrics:
            print("\nRL Controller Metrics:")
            print(f"  Epsilon (exploration): {rl_metrics.get('epsilon', 0.0):.4f}")
            print(f"  Avg Complexity: {rl_metrics.get('avg_complexity', 0.0):.2f}")
            print(f"  Avg Reward: {rl_metrics.get('avg_reward', 0.0):.2f}")

    print("\nComputing credible intervals...")
    intervals = pipeline.get_credible_intervals(strain_data)
    for param_name, interval in list(intervals.items())[:3]:
        print(
            f"  {param_name}: {interval['median']:.2f} [{interval['lower']:.2f}, {interval['upper']:.2f}]"
        )

    # Save RL controller if requested
    if args.save_rl:
        pipeline.save_rl_controller(args.save_rl)
        print(f"\nRL controller saved to {args.save_rl}")
