"""
Parameter Normalization & Denormalization
====================================================
Implements data-driven parameter scalers based on empirical analysis of 8,370 single-signal samples.

Key Features:
- Event-type-specific scaling strategies
- Log-space normalization for wide dynamic ranges (distances, masses, SNR)
- Periodic angle embedding (cos/sin for RA, phase)
- Bounded angle scaling for declination
- Physics-aware scaling (chirp mass, effective spin)
- Distance: Log-minmax normalization
- Masses: Log-zscore for wide dynamic range (0.1-100 M☉)
- Spins: Bounded scaling [0, 1] with proper variance handling
- All parameters scaled to [-1, 1] or [0, 1] for neural network compatibility
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List
import logging


logger = logging.getLogger(__name__)


class ParameterScaler:
    """
    Centralized parameter normalization/denormalization with data-driven statistics.
    
    Attributes:
        param_names: List of parameter names (e.g., ['mass_1', 'mass_2', ...])
        scalers: Dictionary mapping param_name -> scaler config
        event_type: Event type ('BBH', 'BNS', 'NSBH') for optional event-specific scaling
    """
    
    def __init__(self, param_names: List[str], event_type: str = "BBH"):
        self.param_names = param_names
        self.event_type = event_type.upper()
        self.scalers = self._build_scalers()
        self._validate_scalers()  # ✅ DEC 15: Sanity check on initialization
        
    def _build_scalers(self) -> Dict[str, Dict[str, Any]]:
        """Build parameter-specific scaler configurations based on empirical analysis."""
        
        scalers = {}
        
        for param in self.param_names:
            # ============================================================
            # MASSES: Log-space + Z-score (handles 0.1-100 M☉ range)
            # ============================================================
            if param == "mass_1":
                scalers[param] = {
                    "type": "log_zscore",
                    "log_mean": 2.908,   # log(18.3 M☉)
                    "log_std": 0.719,
                    "min": 1.0,
                    "max": 100.0,
                }
            elif param == "mass_2":
                scalers[param] = {
                    "type": "log_zscore",
                    "log_mean": 1.733,   # log(5.6 M☉)
                    "log_std": 1.053,
                    "min": 0.1,
                    "max": 100.0,
                }
            elif param in ["chirp_mass", "chirp_m"]:
                scalers[param] = {
                    "type": "log_zscore",
                    "log_mean": 2.089,   # log(8.1 M☉)
                    "log_std": 0.934,
                    "min": 0.8,
                    "max": 85.0,
                }
            
            # ============================================================
            # DISTANCE: Log-minmax (CRITICAL FIX for -285 Mpc bias!)
            # ============================================================
            elif param == 'luminosity_distance':
                scalers[param] = {
                    'type': 'log_minmax',
                    'log_min': 2.345,      # log(10.4 Mpc)
                    'log_max': 8.987,      # log(8000 Mpc) 
                    'scaleto': (-1, 1),
                }

            elif param == "redshift":
                scalers[param] = {
                    "type": "log_zscore",
                    "log_mean": -2.996,  # log(0.05)
                    "log_std": 0.916,
                    "min": 0.002,
                    "max": 0.4,
                }
            
            # ============================================================
            # SPINS: Bounded [0, 1], use zscore (most spins are low)
            # ============================================================
            elif param in ["a_1", "a_2", "a1", "a2", "spin1_aligned", "spin2_aligned"]:
                scalers[param] = {
                    "type": "bounded_zscore",
                    "min": 0.0,
                    "max": 1.0,
                    "mean": 0.215 if "1" in param else 0.166,
                    "std": 0.230 if "1" in param else 0.193,
                }
            elif param == "effective_spin":
                scalers[param] = {
                    "type": "zscore",  # Can be negative
                    "mean": 0.046,
                    "std": 0.162,
                    "min": -0.5,
                    "max": 1.0,
                }
            
            # ============================================================
            # ANGLES - SKY: Periodic embedding (RA, phase)
            # ============================================================
            elif param in ["ra", "phase"]:
                scalers[param] = {
                    "type": "periodic",
                    "period": 2 * np.pi,
                    "embedding": "cos_sin",  # Return [cos, sin] for NN input
                }
            
            # ============================================================
            # ANGLES - BOUNDED: Declination, inclination, polarization
            # ============================================================
            elif param == "dec":
                scalers[param] = {
                    "type": "bounded_angle",
                    "min": -np.pi / 2,
                    "max": np.pi / 2,
                    "normalize_to": [-1, 1],
                }
            elif param in ["theta_jn", "inclination"]:
                scalers[param] = {
                    "type": "bounded_angle",
                    "min": 0.0,
                    "max": np.pi,
                    "normalize_to": [0, 1],
                }
            elif param == "psi":
                scalers[param] = {
                    "type": "bounded_angle",
                    "min": 0.0,
                    "max": np.pi,
                    "normalize_to": [0, 1],
                }
            elif param in ["tilt1", "tilt2", "tilt_1", "tilt_2"]:
                scalers[param] = {
                    "type": "bounded_angle",
                    "min": 0.0,
                    "max": np.pi,
                    "normalize_to": [0, 1],
                }
            
            # ============================================================
            # TIME: Z-score (can be negative, includes edge cases)
            # ============================================================
            elif param == "geocent_time":
                scalers[param] = {
                    "type": "zscore",
                    "mean": 1.371,
                    "std": 2.267,
                    "clip": [-2.0, 7.0],  # Clip extreme outliers
                }
            elif param == "time_delay":
                scalers[param] = {
                    "type": "zscore",
                    "mean": 0.0,
                    "std": 0.01,  # Typical light-travel time
                }
            
            # ============================================================
            # SNR: Log-space (ranges 5-100+)
            # ============================================================
            elif param in ["network_snr", "snr"]:
                scalers[param] = {
                    "type": "log_zscore",
                    "log_mean": 3.359,   # log(28.7)
                    "log_std": 0.405,
                    "min": 5.0,
                    "max": 100.0,
                }
            
            # ============================================================
            # DEFAULT: Linear minmax [-1, 1]
            # ============================================================
            else:
                scalers[param] = {
                    "type": "linear_minmax",
                    "min": 0.0,
                    "max": 1.0,
                    "scale_to": [-1, 1],
                }
        
        return scalers
        
    def _validate_scalers(self) -> None:
        """
        Validate that all parameters have properly configured scalers.
        
        ✅ DEC 15: Sanity check on initialization to catch configuration errors early.
        """
        for param in self.param_names:
            if param not in self.scalers:
                raise ValueError(
                    f"No scaler defined for parameter: {param}. "
                    f"Available: {list(self.scalers.keys())}"
                )
            
            scaler = self.scalers[param]
            
            # Check required keys
            if "type" not in scaler:
                raise ValueError(f"Scaler for {param} missing required key: 'type'")
            
            # Validate scaler type
            valid_types = [
                "log_zscore", "log_minmax", "bounded_zscore", "zscore",
                "periodic", "bounded_angle", "linear_minmax"
            ]
            if scaler["type"] not in valid_types:
                raise ValueError(
                    f"Invalid scaler type '{scaler['type']}' for {param}. "
                    f"Valid types: {valid_types}"
                )
        
        logger.info(f"✅ Validated scalers for {len(self.param_names)} parameters ({self.event_type} event type)")
        
    def normalize(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Normalize parameters using empirical statistics.
        
        Args:
            params: Dict mapping param_name -> value
        
        Returns:
            Dict mapping param_name -> normalized_value (or [cos, sin] for periodic angles)
        """
        normalized = {}
        
        for param_name, value in params.items():
            if param_name not in self.scalers:
                # Unknown parameter - return as-is with warning
                logger.warning(f"Unknown parameter: {param_name}, returning as-is")
                normalized[param_name] = value
                continue
            
            scaler = self.scalers[param_name]
            normalized[param_name] = self._normalize_single(param_name, value, scaler)
        
        return normalized
    
    def _normalize_single(self, param_name: str, value: float, scaler: Dict) -> Any:
        """Normalize a single parameter using its scaler config."""
        
        scaler_type = scaler["type"]
        
        # ============================================================
        # Log-space Z-score: log(value) → (log(value) - mean) / std
        # ============================================================
        if scaler_type == "log_zscore":
            # Clamp to valid range
            clamped = np.clip(value, scaler["min"], scaler["max"])
            # Log transform
            log_val = np.log(clamped + 1e-10)  # Avoid log(0)
            # Z-score normalization
            norm = (log_val - scaler["log_mean"]) / (scaler["log_std"] + 1e-8)
            return float(np.clip(norm, -5.0, 5.0))  # Clip at ±5σ
        
        # ============================================================
        # Log-space MinMax: log-linear scaling to target range
        # ============================================================
        elif scaler_type == "log_minmax":
            # Clamp to valid range
            clamped = np.clip(value, np.exp(scaler["log_min"]), np.exp(scaler["log_max"]))
            # Log transform
            log_val = np.log(clamped)
            # Scale to target range (default [-1, 1])
            scale_to = scaler.get("scale_to", [-1, 1])
            norm = 2.0 * (log_val - scaler["log_min"]) / (scaler["log_max"] - scaler["log_min"]) - 1.0
            return float(np.clip(norm, scale_to[0], scale_to[1]))
        
        # ============================================================
        # Bounded Z-score: Clamp then Z-score normalize
        # ============================================================
        elif scaler_type == "bounded_zscore":
            # Clamp to bounds
            clamped = np.clip(value, scaler["min"], scaler["max"])
            # Z-score normalization
            if scaler["std"] > 1e-8:
                norm = (clamped - scaler["mean"]) / scaler["std"]
            else:
                norm = 0.0
            return float(np.clip(norm, -5.0, 5.0))
        
        # ============================================================
        # Z-score: (value - mean) / std
        # ============================================================
        elif scaler_type == "zscore":
            clamped = value
            if "clip" in scaler:
                clamped = np.clip(value, scaler["clip"][0], scaler["clip"][1])
            if scaler["std"] > 1e-8:
                norm = (clamped - scaler["mean"]) / scaler["std"]
            else:
                norm = 0.0
            return float(np.clip(norm, -5.0, 5.0))
        
        # ============================================================
        # Periodic: Return [cos(value), sin(value)]
        # ============================================================
        elif scaler_type == "periodic":
            period = scaler["period"]
            # Normalize angle to [0, period)
            angle = value % period
            if scaler.get("embedding") == "cos_sin":
                return {
                    "cos": float(np.cos(angle)),
                    "sin": float(np.sin(angle)),
                }
            else:
                # Simple angle normalization to [-1, 1]
                return float(2.0 * angle / period - 1.0)
        
        # ============================================================
        # Bounded angle: Map to target range
        # ============================================================
        elif scaler_type == "bounded_angle":
            min_val = scaler["min"]
            max_val = scaler["max"]
            scale_to = scaler["normalize_to"]
            # Clamp to bounds
            clamped = np.clip(value, min_val, max_val)
            # Linear map to target range
            norm = scale_to[0] + (clamped - min_val) / (max_val - min_val) * (scale_to[1] - scale_to[0])
            return float(norm)
        
        # ============================================================
        # Linear MinMax: Linear scaling to target range
        # ============================================================
        elif scaler_type == "linear_minmax":
            min_val = scaler["min"]
            max_val = scaler["max"]
            scale_to = scaler.get("scale_to", [-1, 1])
            # Clamp to bounds
            clamped = np.clip(value, min_val, max_val)
            # Linear scale
            if abs(max_val - min_val) > 1e-8:
                norm = scale_to[0] + (clamped - min_val) / (max_val - min_val) * (scale_to[1] - scale_to[0])
            else:
                norm = (scale_to[0] + scale_to[1]) / 2.0
            return float(norm)
        
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}")
            return value
    
    def denormalize(self, normalized_params: Dict[str, float]) -> Dict[str, float]:
        """
        Denormalize parameters back to physical units.
        
        Args:
            normalized_params: Dict mapping param_name -> normalized_value
        
        Returns:
            Dict mapping param_name -> physical_value
        """
        denormalized = {}
        
        for param_name, norm_value in normalized_params.items():
            if param_name not in self.scalers:
                logger.warning(f"Unknown parameter: {param_name}, returning as-is")
                denormalized[param_name] = norm_value
                continue
            
            scaler = self.scalers[param_name]
            denormalized[param_name] = self._denormalize_single(param_name, norm_value, scaler)
        
        return denormalized
    
    def _denormalize_single(self, param_name: str, norm_value: float, scaler: Dict) -> float:
        """Denormalize a single parameter using its scaler config."""
        
        scaler_type = scaler["type"]
        
        if scaler_type == "log_zscore":
            # Reverse: norm = (log(val) - mean) / std → log(val) = norm * std + mean
            log_val = norm_value * scaler["log_std"] + scaler["log_mean"]
            val = np.exp(log_val)
            # Clamp to bounds
            return float(np.clip(val, scaler["min"], scaler["max"]))
        
        elif scaler_type == "log_minmax":
            # Reverse: norm = 2*(log(val) - log_min) / (log_max - log_min) - 1
            # → log(val) = (norm + 1) / 2 * (log_max - log_min) + log_min
            log_val = (norm_value + 1.0) / 2.0 * (scaler["log_max"] - scaler["log_min"]) + scaler["log_min"]
            val = np.exp(log_val)
            # Clamp to physical range
            return float(np.clip(val, np.exp(scaler["log_min"]), np.exp(scaler["log_max"])))
        
        elif scaler_type == "bounded_zscore":
            # Reverse: norm = (val - mean) / std → val = norm * std + mean
            val = norm_value * scaler["std"] + scaler["mean"]
            # Clamp to bounds
            return float(np.clip(val, scaler["min"], scaler["max"]))
        
        elif scaler_type == "zscore":
            # Reverse: norm = (val - mean) / std → val = norm * std + mean
            val = norm_value * scaler["std"] + scaler["mean"]
            # Clamp if specified
            if "clip" in scaler:
                val = np.clip(val, scaler["clip"][0], scaler["clip"][1])
            return float(val)
        
        elif scaler_type == "periodic":
            # For periodic angles stored as cos/sin, reconstruct using atan2
            if isinstance(norm_value, dict) and "cos" in norm_value and "sin" in norm_value:
                angle = np.arctan2(norm_value["sin"], norm_value["cos"])
                # Ensure positive angle [0, 2π)
                if angle < 0:
                    angle += 2 * np.pi
                return float(angle)
            else:
                # Simple normalized angle [-1, 1] → [0, 2π]
                period = scaler["period"]
                angle = (norm_value + 1.0) / 2.0 * period
                return float(angle % period)
        
        elif scaler_type == "bounded_angle":
            min_val = scaler["min"]
            max_val = scaler["max"]
            scale_to = scaler["normalize_to"]
            # Reverse linear map
            val = min_val + (norm_value - scale_to[0]) / (scale_to[1] - scale_to[0] + 1e-8) * (max_val - min_val)
            # Clamp
            return float(np.clip(val, min_val, max_val))
        
        elif scaler_type == "linear_minmax":
            min_val = scaler["min"]
            max_val = scaler["max"]
            scale_to = scaler.get("scale_to", [-1, 1])
            # Reverse linear scale
            val = min_val + (norm_value - scale_to[0]) / (scale_to[1] - scale_to[0]) * (max_val - min_val)
            return float(np.clip(val, min_val, max_val))
        
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}")
            return norm_value
    
    def validate_normalization(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate normalization/denormalization round-trip.
        
        Returns:
            Dict with validation results for non-periodic parameters
        """
        normalized = self.normalize(params)
        denormalized = self.denormalize(normalized)
        
        errors = {}
        for param_name in params.keys():
            if param_name not in self.scalers:
                continue
            
            scaler = self.scalers[param_name]
            if scaler["type"] == "periodic":
                continue  # Skip periodic angles for validation
            
            original = params[param_name]
            recovered = denormalized[param_name]
            relative_error = abs(original - recovered) / (abs(original) + 1e-8)
            errors[param_name] = {
                "original": original,
                "recovered": recovered,
                "abs_error": abs(original - recovered),
                "rel_error": relative_error
            }
        
        return errors


class TorchParameterScaler(nn.Module):
    """PyTorch-native parameter scaler for batch operations."""
    
    def __init__(self, param_names: List[str], event_type: str = "BBH", device: str = "cpu"):
        super().__init__()
        self.param_names = param_names
        self.event_type = event_type.upper()
        self.device = torch.device(device)
        
        # Build scaler using numpy version, then convert to torch
        np_scaler = ParameterScaler(param_names, event_type)
        self.scalers = np_scaler.scalers
        
        # Register scaler parameters as buffers (not trainable)
        self._register_scaler_buffers()
    
    def _register_scaler_buffers(self):
        """Register scaler statistics as non-trainable buffers."""
        for param_name, scaler in self.scalers.items():
            scaler_type = scaler.get("type", "linear_minmax")
            
            if "log_mean" in scaler:
                self.register_buffer(
                    f"{param_name}_log_mean",
                    torch.tensor(scaler["log_mean"], device=self.device)
                )
            if "log_std" in scaler:
                self.register_buffer(
                    f"{param_name}_log_std",
                    torch.tensor(scaler["log_std"], device=self.device)
                )
            if "mean" in scaler:
                self.register_buffer(
                    f"{param_name}_mean",
                    torch.tensor(scaler["mean"], device=self.device)
                )
            if "std" in scaler:
                self.register_buffer(
                    f"{param_name}_std",
                    torch.tensor(scaler["std"], device=self.device)
                )
            if "log_min" in scaler:
                self.register_buffer(
                    f"{param_name}_log_min",
                    torch.tensor(scaler["log_min"], device=self.device)
                )
            if "log_max" in scaler:
                self.register_buffer(
                    f"{param_name}_log_max",
                    torch.tensor(scaler["log_max"], device=self.device)
                )
    
    def normalize_batch(self, params: torch.Tensor) -> torch.Tensor:
        """
        Normalize batch of parameters.
        
        Args:
            params: [batch, param_dim] tensor of physical parameters
        
        Returns:
            normalized: [batch, param_dim] tensor of normalized parameters
        """
        normalized = params.clone()
        
        for i, param_name in enumerate(self.param_names):
            scaler = self.scalers[param_name]
            scaler_type = scaler.get("type", "linear_minmax")
            
            if scaler_type == "log_zscore":
                min_val = torch.tensor(scaler["min"], dtype=params.dtype, device=params.device)
                max_val = torch.tensor(scaler["max"], dtype=params.dtype, device=params.device)
                log_mean = getattr(self, f"{param_name}_log_mean")
                log_std = getattr(self, f"{param_name}_log_std")
                
                clamped = torch.clamp(params[:, i], min_val, max_val)
                log_val = torch.log(clamped + 1e-10)
                normalized[:, i] = torch.clamp(
                    (log_val - log_mean) / (log_std + 1e-8), -5.0, 5.0
                )
            
            elif scaler_type == "log_minmax":
                log_min = getattr(self, f"{param_name}_log_min")
                log_max = getattr(self, f"{param_name}_log_max")
                
                min_val = torch.exp(log_min)
                max_val = torch.exp(log_max)
                clamped = torch.clamp(params[:, i], min_val, max_val)
                log_val = torch.log(clamped)
                norm = 2.0 * (log_val - log_min) / (log_max - log_min) - 1.0
                normalized[:, i] = torch.clamp(norm, -1.0, 1.0)
            
            elif scaler_type == "zscore":
                mean = getattr(self, f"{param_name}_mean")
                std = getattr(self, f"{param_name}_std")
                
                if "clip" in scaler:
                    clamped = torch.clamp(
                        params[:, i], scaler["clip"][0], scaler["clip"][1]
                    )
                else:
                    clamped = params[:, i]
                
                normalized[:, i] = torch.clamp(
                    (clamped - mean) / (std + 1e-8), -5.0, 5.0
                )
            
            elif scaler_type == "bounded_zscore":
                min_val = torch.tensor(scaler["min"], device=self.device)
                max_val = torch.tensor(scaler["max"], device=self.device)
                mean = getattr(self, f"{param_name}_mean")
                std = getattr(self, f"{param_name}_std")
                
                clamped = torch.clamp(params[:, i], min_val, max_val)
                normalized[:, i] = torch.clamp(
                    (clamped - mean) / (std + 1e-8), -5.0, 5.0
                )
            
            # ✅ FIXED (Dec 15): Add missing scaler types
            elif scaler_type == "bounded_angle":
                min_val = torch.tensor(scaler["min"], device=self.device)
                max_val = torch.tensor(scaler["max"], device=self.device)
                scale_to = scaler["normalize_to"]
                
                clamped = torch.clamp(params[:, i], min_val, max_val)
                norm = scale_to[0] + (clamped - min_val) / (max_val - min_val + 1e-8) * (scale_to[1] - scale_to[0])
                normalized[:, i] = norm
            
            elif scaler_type == "linear_minmax":
                min_val = torch.tensor(scaler["min"], device=self.device)
                max_val = torch.tensor(scaler["max"], device=self.device)
                scale_to = scaler.get("scale_to", [-1, 1])
                
                clamped = torch.clamp(params[:, i], min_val, max_val)
                range_size = max_val - min_val
                if range_size > 1e-8:
                    norm = scale_to[0] + (clamped - min_val) / range_size * (scale_to[1] - scale_to[0])
                else:
                    norm = torch.ones_like(clamped) * (scale_to[0] + scale_to[1]) / 2.0
                normalized[:, i] = norm
            
            elif scaler_type == "periodic":
                # For periodic angles in batch mode: just normalize to [-1, 1]
                # ✅ Note: This doesn't expand to cos/sin (that requires different handling)
                
                # Only warn once per scaler instance (not every batch!)
                if not hasattr(self, '_periodic_warning_shown'):
                    logger.info(f"ℹ️  Periodic angles (ra, phase) using simple normalization in batch mode (not cos/sin)")
                    self._periodic_warning_shown = True
                
                period = torch.tensor(scaler["period"], device=self.device)
                angle = params[:, i] % period
                norm = 2.0 * angle / period - 1.0
                normalized[:, i] = norm

        
        return normalized
    
    def denormalize_batch(self, normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize batch of parameters.
        
        Args:
            normalized: [batch, param_dim] tensor of normalized parameters
        
        Returns:
            params: [batch, param_dim] tensor of physical parameters
        """
        params = normalized.clone()
        
        for i, param_name in enumerate(self.param_names):
            scaler = self.scalers[param_name]
            scaler_type = scaler.get("type", "linear_minmax")
            
            if scaler_type == "log_zscore":
                log_mean = getattr(self, f"{param_name}_log_mean")
                log_std = getattr(self, f"{param_name}_log_std")
                min_val = torch.tensor(scaler["min"], device=self.device)
                max_val = torch.tensor(scaler["max"], device=self.device)
                
                log_val = normalized[:, i] * log_std + log_mean
                val = torch.exp(log_val)
                params[:, i] = torch.clamp(val, min_val, max_val)
            
            elif scaler_type == "log_minmax":
                log_min = getattr(self, f"{param_name}_log_min")
                log_max = getattr(self, f"{param_name}_log_max")
                
                log_val = (normalized[:, i] + 1.0) / 2.0 * (log_max - log_min) + log_min
                val = torch.exp(log_val)
                min_val = torch.exp(log_min)
                max_val = torch.exp(log_max)
                params[:, i] = torch.clamp(val, min_val, max_val)
            
            elif scaler_type == "zscore":
                mean = getattr(self, f"{param_name}_mean")
                std = getattr(self, f"{param_name}_std")
                
                val = normalized[:, i] * std + mean
                if "clip" in scaler:
                    val = torch.clamp(val, scaler["clip"][0], scaler["clip"][1])
                params[:, i] = val
            
            elif scaler_type == "bounded_zscore":
                mean = getattr(self, f"{param_name}_mean")
                std = getattr(self, f"{param_name}_std")
                min_val = torch.tensor(scaler["min"], device=self.device)
                max_val = torch.tensor(scaler["max"], device=self.device)
                
                val = normalized[:, i] * std + mean
                params[:, i] = torch.clamp(val, min_val, max_val)
            
            # ✅ FIXED (Dec 15): Add missing denormalize cases
            elif scaler_type == "bounded_angle":
                min_val = torch.tensor(scaler["min"], device=self.device)
                max_val = torch.tensor(scaler["max"], device=self.device)
                scale_to = scaler["normalize_to"]
                
                val = min_val + (normalized[:, i] - scale_to[0]) / (scale_to[1] - scale_to[0] + 1e-8) * (max_val - min_val)
                params[:, i] = torch.clamp(val, min_val, max_val)
            
            elif scaler_type == "linear_minmax":
                min_val = torch.tensor(scaler["min"], device=self.device)
                max_val = torch.tensor(scaler["max"], device=self.device)
                scale_to = scaler.get("scale_to", [-1, 1])
                
                val = min_val + (normalized[:, i] - scale_to[0]) / (scale_to[1] - scale_to[0] + 1e-8) * (max_val - min_val)
                params[:, i] = torch.clamp(val, min_val, max_val)
            
            elif scaler_type == "periodic":
                # Reverse of periodic normalization
                period = torch.tensor(scaler["period"], device=self.device)
                angle = (normalized[:, i] + 1.0) / 2.0 * period
                params[:, i] = angle % period
        
        return params
