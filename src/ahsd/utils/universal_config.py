"""
Universal Configuration Reader and Parser for PosteriFlow.

Provides consistent, centralized config management across all models and training scripts.
Handles nested configs, type validation, defaults, and dot-notation access.

Key Features:
- Single source of truth for all configuration
- Recursive nested access with dot notation
- Type validation and conversion
- Comprehensive logging and validation
- Backward-compatible fallbacks
- Environment variable substitution
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """
    A dictionary supporting both dict and attribute access patterns.
    
    Examples:
        config['learning_rate']  # Dict access
        config.learning_rate     # Attribute access
        config.get_nested('neural_posterior.flow_type')  # Nested access
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
        
        Args:
            key_path: Dot-separated path (e.g., 'neural_posterior.flow_type')
            default: Default value if not found
        
        Returns:
            Value at nested path or default
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
    
    def set_nested(self, key_path: str, value: Any) -> None:
        """
        Set nested value using dot notation, creating dicts as needed.
        
        Args:
            key_path: Dot-separated path (e.g., 'neural_posterior.flow_type')
            value: Value to set
        """
        keys = key_path.split('.')
        current = self
        for key in keys[:-1]:
            if key not in current:
                current[key] = ConfigDict()
            current = current[key]
        current[keys[-1]] = value
    
    def to_dict(self, recursive: bool = True) -> dict:
        """
        Convert ConfigDict back to plain dict.
        
        Args:
            recursive: Recursively convert nested ConfigDicts
        
        Returns:
            Plain dictionary
        """
        if not recursive:
            return dict(self)
        
        result = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                result[k] = v.to_dict(recursive=True)
            elif isinstance(v, dict):
                result[k] = {_k: _v.to_dict(recursive=True) if isinstance(_v, ConfigDict) else _v
                            for _k, _v in v.items()}
            else:
                result[k] = v
        return result


class UniversalConfigReader:
    """
    Central configuration reader for PosteriFlow.
    
    Provides unified interface for loading, parsing, and accessing configurations
    across all models and training scripts.
    
    Usage:
        # Load from YAML
        reader = UniversalConfigReader()
        config = reader.load('configs/enhanced_training.yaml')
        
        # Access config
        lr = config.get('priority_net.learning_rate', 1e-4)
        flow_type = config.get_nested('neural_posterior.flow_type')
        
        # Validate
        reader.validate(config)
        
        # Get section
        pn_config = reader.get_section(config, 'priority_net')
    """
    
    # Default configuration sections that should exist
    EXPECTED_SECTIONS = {
        'priority_net',
        'neural_posterior',
        'data',
        'training',
        'optimizer',
        'loss',
    }
    
    # Default values for critical parameters
    DEFAULTS = {
        'device': 'auto',
        'random_seed': 42,
        'experiment_name': 'default_experiment',
        'priority_net.learning_rate': 1e-4,
        'priority_net.batch_size': 32,
        'priority_net.epochs': 50,
        'neural_posterior.flow_type': 'nsf',
        'neural_posterior.context_dim': 768,
        'neural_posterior.num_layers': 8,
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize ConfigReader.
        
        Args:
            logger: Logger instance (creates one if None)
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def load(self, config_path: Union[str, Path], env_substitution: bool = True) -> ConfigDict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
            env_substitution: Replace ${VAR_NAME} with environment variables
        
        Returns:
            ConfigDict with configuration
        
        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Environment variable substitution
            if env_substitution:
                content = self._substitute_env_vars(content)
            
            config_dict = yaml.safe_load(content)
            
            if config_dict is None:
                self.logger.warning(f"Config file {config_path} is empty, using defaults")
                config_dict = {}
            
            # Convert to ConfigDict recursively
            config = self._to_config_dict(config_dict)
            
            self.logger.info(f"✅ Loaded config from {config_path}")
            return config
        
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML {config_path}: {e}")
        except Exception as e:
            raise Exception(f"Failed to load config {config_path}: {e}")
    
    def _substitute_env_vars(self, content: str) -> str:
        """
        Replace ${VAR_NAME} with environment variables.
        
        Args:
            content: YAML content string
        
        Returns:
            Content with env vars substituted
        """
        import re
        
        def replace_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        
        return re.sub(r'\$\{(\w+)\}', replace_var, content)
    
    def _to_config_dict(self, obj: Any) -> Any:
        """
        Recursively convert dict to ConfigDict.
        
        Args:
            obj: Object to convert
        
        Returns:
            ConfigDict or converted object
        """
        if isinstance(obj, dict):
            return ConfigDict({k: self._to_config_dict(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [self._to_config_dict(item) for item in obj]
        return obj
    
    def get(
        self,
        config: Union[ConfigDict, dict],
        key: str,
        default: Any = None,
        dtype: Optional[type] = None,
        section: Optional[str] = None
    ) -> Any:
        """
        Get configuration value with nested access and type conversion.
        
        Args:
            config: Configuration dict
            key: Configuration key (supports dot notation)
            default: Default value if not found
            dtype: Type to convert to (int, float, bool, etc.)
            section: Look in specific section first (e.g., 'priority_net')
        
        Returns:
            Configuration value
        """
        value = None
        
        # Try specific section if provided
        if section and section in config:
            value = config[section].get(key) if isinstance(config[section], dict) else None
        
        # Try nested access at top level
        if value is None and '.' in key:
            value = config.get_nested(key) if isinstance(config, ConfigDict) else None
        
        # Try direct access
        if value is None:
            value = config.get(key) if isinstance(config, dict) else None
        
        # Use default if not found
        if value is None:
            value = self.DEFAULTS.get(key, default)
        
        # Type conversion
        if value is not None and dtype is not None:
            try:
                value = dtype(value)
            except (ValueError, TypeError) as e:
                self.logger.warning(
                    f"Failed to convert {key}={value} to {dtype.__name__}: {e}, using default"
                )
                value = default
        
        return value
    
    def get_section(self, config: ConfigDict, section: str) -> ConfigDict:
        """
        Get configuration section (e.g., 'priority_net', 'neural_posterior').
        
        Args:
            config: Full configuration
            section: Section name
        
        Returns:
            Section config or empty ConfigDict if not found
        """
        if section in config:
            section_config = config[section]
            if isinstance(section_config, ConfigDict):
                return section_config
            elif isinstance(section_config, dict):
                return self._to_config_dict(section_config)
        
        self.logger.warning(f"Section '{section}' not found in config")
        return ConfigDict()
    
    def validate(
        self,
        config: ConfigDict,
        required_sections: Optional[List[str]] = None,
        raise_on_error: bool = False
    ) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            required_sections: Sections that must exist
            raise_on_error: Raise exception on validation error
        
        Returns:
            True if valid
        """
        errors = []
        warnings = []
        
        # Check required sections
        sections_to_check = required_sections or ['priority_net', 'neural_posterior']
        for section in sections_to_check:
            if section not in config:
                warnings.append(f"Missing section: {section}")
        
        # Validate learning rate
        lr = self.get(config, 'learning_rate', section='priority_net')
        if lr and not (1e-6 <= float(lr) <= 1e-1):
            errors.append(f"Learning rate {lr} out of range [1e-6, 1e-1]")
        
        # Validate batch size
        batch_size = self.get(config, 'batch_size', section='priority_net')
        if batch_size and not (1 <= int(batch_size) <= 1024):
            errors.append(f"Batch size {batch_size} out of range [1, 1024]")
        
        # Validate epochs
        epochs = self.get(config, 'epochs', section='priority_net')
        if epochs and int(epochs) <= 0:
            errors.append(f"Epochs must be positive, got {epochs}")
        
        # Validate loss weights if present
        if 'neural_posterior' in config:
            np_config = config['neural_posterior']
            weights = []
            for key in ['flow_loss_weight', 'extraction_loss_weight', 'residual_loss_weight',
                       'physics_loss_weight', 'bounds_penalty_weight', 'sample_loss_weight']:
                if key in np_config:
                    weights.append(float(np_config[key]))
            
            if weights and sum(weights) > 5.0:
                warnings.append(f"Loss weights sum to {sum(weights)}, consider reducing")
        
        # Log validation results
        if errors:
            self.logger.error("❌ Configuration validation FAILED:")
            for error in errors:
                self.logger.error(f"   - {error}")
            if raise_on_error:
                raise ValueError(f"Configuration validation failed: {errors}")
        
        if warnings:
            for warning in warnings:
                self.logger.warning(f"⚠️  {warning}")
        
        if not errors:
            self.logger.info("✅ Configuration validated successfully")
        
        return len(errors) == 0
    
    def log_config(self, config: ConfigDict, max_depth: int = 3) -> None:
        """
        Log configuration in organized format.
        
        Args:
            config: Configuration to log
            max_depth: Maximum nesting depth to display
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📋 CONFIGURATION")
        self.logger.info("=" * 80)
        self._log_dict(config, depth=0, max_depth=max_depth)
        self.logger.info("=" * 80 + "\n")
    
    def _log_dict(self, obj: dict, prefix: str = "", depth: int = 0, max_depth: int = 3) -> None:
        """
        Recursively log dictionary contents.
        
        Args:
            obj: Dictionary to log
            prefix: Prefix for logging
            depth: Current nesting depth
            max_depth: Maximum nesting depth
        """
        if depth > max_depth:
            return
        
        indent = "  " * depth
        for key, value in obj.items():
            if isinstance(value, dict):
                self.logger.info(f"{indent}{key}:")
                self._log_dict(value, prefix=prefix, depth=depth + 1, max_depth=max_depth)
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    self.logger.info(f"{indent}{key}: [{len(value)} items]")
                else:
                    value_str = str(value)[:100]
                    self.logger.info(f"{indent}{key}: {value_str}")
            else:
                self.logger.info(f"{indent}{key}: {value}")
    
    def merge(self, base: ConfigDict, override: ConfigDict) -> ConfigDict:
        """
        Merge override config into base config.
        
        Args:
            base: Base configuration
            override: Override configuration
        
        Returns:
            Merged configuration
        """
        result = ConfigDict(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge(ConfigDict(result[key]), ConfigDict(value))
            else:
                result[key] = value
        
        return result
    
    def save(self, config: ConfigDict, output_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            output_path: Path to save YAML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"✅ Saved config to {output_path}")


# Singleton instance for global access
_reader: Optional[UniversalConfigReader] = None


def get_reader() -> UniversalConfigReader:
    """Get or create singleton config reader."""
    global _reader
    if _reader is None:
        _reader = UniversalConfigReader()
    return _reader


def load_config(config_path: Union[str, Path]) -> ConfigDict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        ConfigDict with configuration
    """
    return get_reader().load(config_path)


def get_config_value(
    config: Union[ConfigDict, dict],
    key: str,
    default: Any = None,
    dtype: Optional[type] = None,
    section: Optional[str] = None
) -> Any:
    """
    Get configuration value.
    
    Args:
        config: Configuration dict
        key: Configuration key
        default: Default value
        dtype: Type to convert to
        section: Section to look in
    
    Returns:
        Configuration value
    """
    return get_reader().get(config, key, default=default, dtype=dtype, section=section)


def validate_config(config: ConfigDict) -> bool:
    """
    Validate configuration.
    
    Args:
        config: Configuration to validate
    
    Returns:
        True if valid
    """
    return get_reader().validate(config)
