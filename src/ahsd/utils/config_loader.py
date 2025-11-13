"""
Unified configuration loader for PriorityNet training.
Handles YAML parsing, nested configs, and provides clean attribute access.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class ConfigDict(dict):
    """
    A dictionary that also supports attribute access.
    Allows config['key'] and config.key to work interchangeably.
    """
    
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value
    
    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """
        Get nested value using dot notation.
        Example: config.get_nested('priority_net.learning_rate', 1e-4)
        """
        keys = key_path.split('.')
        value = self
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML file
    
    Returns:
        Dictionary with raw YAML content
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError("Config file is empty")
        
        return config
    
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML: {e}")


def extract_priority_net_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract PriorityNet configuration from full config YAML.
    Returns nested dict with all priority_net settings.
    """
    if 'priority_net' in config_dict:
        return config_dict['priority_net']
    return config_dict


def create_config_object(config_dict: Dict[str, Any]) -> ConfigDict:
    """
    Convert nested dictionary to ConfigDict with recursive conversion.
    Allows config.key and config['key'] access patterns.
    """
    def _convert(obj):
        if isinstance(obj, dict):
            return ConfigDict({k: _convert(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [_convert(item) for item in obj]
        return obj
    
    return _convert(config_dict)


def load_enhanced_config(config_path: Union[str, Path]) -> ConfigDict:
    """
    Load and parse enhanced training configuration.
    
    Args:
        config_path: Path to enhanced_training.yaml
    
    Returns:
        ConfigDict with all parameters accessible via dot notation.
        If 'priority_net' section exists, returns the full config with priority_net nested;
        otherwise returns priority_net directly for backward compatibility.
    """
    logger = logging.getLogger(__name__)
    
    # Load raw YAML
    config_dict = load_yaml_config(config_path)
    
    # If config has 'priority_net' section, return full config with nesting preserved
    if 'priority_net' in config_dict:
        config = create_config_object(config_dict)
    else:
        # Fallback: return priority_net config directly for flat configs
        pn_config = extract_priority_net_config(config_dict)
        config = create_config_object(pn_config)
    
    return config


def log_config(config: ConfigDict, logger: Optional[logging.Logger] = None) -> None:
    """
    Log all configuration parameters in organized sections.
    
    Args:
        config: Configuration object (can be full config with 'priority_net' nested, or flat config)
        logger: Logger instance (creates one if None)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # If config has 'priority_net' nested, use that section for logging
    config_to_log = config.get('priority_net', config) if 'priority_net' in config else config
    
    logger.info("\n" + "=" * 80)
    logger.info("📋 COMPLETE CONFIGURATION")
    logger.info("=" * 80)
    
    # Architecture
    if 'hidden_dims' in config_to_log or 'dropout' in config_to_log:
        logger.info("\n🏗️  ARCHITECTURE:")
        logger.info(f"   hidden_dims: {config_to_log.get('hidden_dims', [512, 384, 256, 128])}")
        logger.info(f"   dropout: {config_to_log.get('dropout', 0.2)}")
        logger.info(f"   use_strain: {config_to_log.get('use_strain', True)}")
        logger.info(f"   use_edge_conditioning: {config_to_log.get('use_edge_conditioning', True)}")
        logger.info(f"   use_transformer_encoder: {config_to_log.get('use_transformer_encoder', True)}")
        logger.info(f"   n_edge_types: {config_to_log.get('n_edge_types', 19)}")
    
    # Optimizer
    if 'learning_rate' in config_to_log or 'weight_decay' in config_to_log:
        logger.info("\n🎯 OPTIMIZER:")
        logger.info(f"   learning_rate: {config_to_log.get('learning_rate', 5e-4):.2e}")
        logger.info(f"   weight_decay: {config_to_log.get('weight_decay', 1e-5):.2e}")
        logger.info(f"   optimizer: {config_to_log.get('optimizer', 'AdamW')}")
    
    # Training
    if 'batch_size' in config_to_log or 'epochs' in config_to_log:
        logger.info("\n📊 TRAINING:")
        logger.info(f"   batch_size: {config_to_log.get('batch_size', 32)}")
        logger.info(f"   epochs: {config_to_log.get('epochs', 250)}")
        logger.info(f"   patience: {config_to_log.get('patience', 20)}")
        logger.info(f"   warmup_epochs: {config_to_log.get('warmup_epochs', 10)}")
        logger.info(f"   warmup_start_factor: {config_to_log.get('warmup_start_factor', 0.1)}")
    
    # Learning rate scheduler
    if 'scheduler' in config_to_log:
        logger.info("\n📈 LEARNING RATE SCHEDULER:")
        logger.info(f"   scheduler: {config_to_log.get('scheduler', 'ReduceLROnPlateau')}")
        logger.info(f"   scheduler_patience: {config_to_log.get('scheduler_patience', 5)}")
        logger.info(f"   scheduler_factor: {config_to_log.get('scheduler_factor', 0.5)}")
        logger.info(f"   min_lr: {config_to_log.get('min_lr', 1e-6):.2e}")
    
    # Loss function
    if 'ranking_weight' in config_to_log:
        logger.info("\n🎲 LOSS FUNCTION:")
        logger.info(f"   ranking_weight: {config_to_log.get('ranking_weight', 0.3)}")
        logger.info(f"   mse_weight: {config_to_log.get('mse_weight', 0.6)}")
        logger.info(f"   uncertainty_weight: {config_to_log.get('uncertainty_weight', 0.1)}")
        logger.info(f"   use_snr_weighting: {config_to_log.get('use_snr_weighting', True)}")
        logger.info(f"   loss_scale_factor: {config_to_log.get('loss_scale_factor', 0.001)}")
        logger.info(f"   label_smoothing: {config_to_log.get('label_smoothing', 0.02)}")
    
    # Gradient management
    if 'gradient_clip_norm' in config_to_log:
        logger.info("\n🔄 GRADIENT MANAGEMENT:")
        logger.info(f"   gradient_clip_norm: {config_to_log.get('gradient_clip_norm', 1.0)}")
        logger.info(f"   gradient_log_threshold: {config_to_log.get('gradient_log_threshold', 0.5)}")
    
    # Attention/Modal fusion
    if 'use_modal_fusion' in config_to_log:
        logger.info("\n✨ ATTENTION/MODAL FUSION:")
        logger.info(f"   use_modal_fusion: {config_to_log.get('use_modal_fusion', False)}")
        logger.info(f"   attention_num_heads: {config_to_log.get('attention_num_heads', 4)}")
        logger.info(f"   attention_dropout: {config_to_log.get('attention_dropout', 0.1)}")
    
    # Overlap handling
    if 'overlap_use_attention' in config_to_log:
        logger.info("\n🔗 OVERLAP HANDLING:")
        logger.info(f"   overlap_use_attention: {config_to_log.get('overlap_use_attention', False)}")
        logger.info(f"   overlap_importance_hidden: {config_to_log.get('overlap_importance_hidden', 16)}")
    
    logger.info("=" * 80 + "\n")


def get_config_value(
    config: Union[ConfigDict, dict],
    key: str,
    default: Any = None,
    dtype: Optional[type] = None
) -> Any:
    """
    Safely get config value with optional type conversion.
    Checks nested 'priority_net' section if key not found at top level.
    
    Args:
        config: Configuration object or dict (can have nested 'priority_net')
        key: Configuration key
        default: Default value if key not found
        dtype: Type to convert value to (int, float, bool, etc.)
    
    Returns:
        Configuration value
    """
    # Try to get value from config
    if isinstance(config, dict):
        value = config.get(key)
        # If not found at top level and 'priority_net' exists, check there
        if value is None and 'priority_net' in config:
            value = config['priority_net'].get(key)
        if value is None:
            value = default
    else:
        value = getattr(config, key, None)
        # If not found and config has 'priority_net', check there
        if value is None and hasattr(config, 'priority_net'):
            priority_net = getattr(config, 'priority_net', None)
            if priority_net is not None:
                value = getattr(priority_net, key, None)
        if value is None:
            value = default
    
    # Type conversion
    if value is not None and dtype is not None:
        try:
            value = dtype(value)
        except (ValueError, TypeError):
            logging.warning(f"Failed to convert {key}={value} to {dtype.__name__}, using default")
            value = default
    
    return value


def validate_config(config: ConfigDict, logger: Optional[logging.Logger] = None) -> bool:
    """
    Validate critical configuration parameters.
    
    Args:
        config: Configuration object (can be full config with 'priority_net' nested, or flat config)
        logger: Logger instance
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # If config has 'priority_net' nested, use that section for validation
    config_to_validate = config.get('priority_net', config) if 'priority_net' in config else config
    
    # Validate learning rate
    lr = config_to_validate.get('learning_rate', 5e-4)
    if not (1e-6 <= float(lr) <= 1e-2):
        raise ValueError(f"Learning rate {lr} out of range [1e-6, 1e-2]")
    
    # Validate batch size
    batch_size = config_to_validate.get('batch_size', 32)
    if not (1 <= int(batch_size) <= 512):
        raise ValueError(f"Batch size {batch_size} out of range [1, 512]")
    
    # Validate loss weights sum to ~1.0
    ranking = float(config_to_validate.get('ranking_weight', 0.3))
    mse = float(config_to_validate.get('mse_weight', 0.6))
    uncertainty = float(config_to_validate.get('uncertainty_weight', 0.1))
    weight_sum = ranking + mse + uncertainty
    
    if abs(weight_sum - 1.0) > 0.01:
        logger.warning(f"⚠️  Loss weights sum to {weight_sum:.3f}, expected ~1.0")
    
    # Validate warmup
    warmup = int(config_to_validate.get('warmup_epochs', 10))
    epochs = int(config_to_validate.get('epochs', 250))
    if warmup >= epochs:
        raise ValueError(f"Warmup epochs ({warmup}) must be < total epochs ({epochs})")
    
    # Validate patience
    patience = int(config_to_validate.get('patience', 20))
    if patience > epochs // 2:
        logger.warning(f"⚠️  Patience ({patience}) is > half of epochs ({epochs})")
    
    logger.info("✅ Configuration validated successfully\n")
    return True
