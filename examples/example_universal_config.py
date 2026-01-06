"""
Comprehensive examples of using the Universal Configuration Reader.

Demonstrates all major features and usage patterns.
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_loading():
    """Example 1: Load and access configuration."""
    from ahsd.utils import load_config
    
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Basic Loading and Access")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config('configs/enhanced_training.yaml')
    
    # Access values using dot notation
    lr = config.get('priority_net.learning_rate')
    batch_size = config.get('priority_net.batch_size')
    flow_type = config.get_nested('neural_posterior.flow_type')
    
    logger.info(f"Learning Rate: {lr}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Flow Type: {flow_type}")


def example_2_reader_api():
    """Example 2: Using UniversalConfigReader directly."""
    from ahsd.utils import UniversalConfigReader
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: UniversalConfigReader API")
    logger.info("=" * 80)
    
    reader = UniversalConfigReader(logger=logger)
    config = reader.load('configs/enhanced_training.yaml')
    
    # Validate configuration
    try:
        reader.validate(config, raise_on_error=True)
        logger.info("✅ Configuration is valid")
    except ValueError as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return
    
    # Log configuration
    reader.log_config(config, max_depth=2)
    
    # Get specific values
    lr = reader.get(config, 'learning_rate', section='priority_net', dtype=float)
    batch_size = reader.get(config, 'batch_size', section='priority_net', dtype=int)
    
    logger.info(f"Learning Rate (float): {lr}")
    logger.info(f"Batch Size (int): {batch_size}")


def example_3_section_access():
    """Example 3: Access configuration sections."""
    from ahsd.utils import UniversalConfigReader
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: Section-Based Access")
    logger.info("=" * 80)
    
    reader = UniversalConfigReader(logger=logger)
    config = reader.load('configs/enhanced_training.yaml')
    
    # Get entire sections
    pn_config = reader.get_section(config, 'priority_net')
    np_config = reader.get_section(config, 'neural_posterior')
    
    logger.info("\n📋 Priority Net Configuration:")
    for key, value in list(pn_config.items())[:5]:  # Show first 5
        logger.info(f"   {key}: {value}")
    
    logger.info("\n📋 Neural Posterior Configuration:")
    for key, value in list(np_config.items())[:5]:  # Show first 5
        logger.info(f"   {key}: {value}")


def example_4_with_defaults():
    """Example 4: Using defaults and type conversion."""
    from ahsd.utils import UniversalConfigReader
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 4: Defaults and Type Conversion")
    logger.info("=" * 80)
    
    reader = UniversalConfigReader(logger=logger)
    config = reader.load('configs/enhanced_training.yaml')
    
    # Get with defaults
    lr = reader.get(config, 'learning_rate', default=1e-4, dtype=float, section='priority_net')
    batch_size = reader.get(config, 'batch_size', default=32, dtype=int, section='priority_net')
    custom_param = reader.get(config, 'nonexistent_param', default=99.9, dtype=float)
    
    logger.info(f"Learning Rate: {lr} (type: {type(lr).__name__})")
    logger.info(f"Batch Size: {batch_size} (type: {type(batch_size).__name__})")
    logger.info(f"Custom Param (default): {custom_param} (type: {type(custom_param).__name__})")


def example_5_nested_access():
    """Example 5: Deep nested configuration access."""
    from ahsd.utils import load_config
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 5: Nested Configuration Access")
    logger.info("=" * 80)
    
    config = load_config('configs/enhanced_training.yaml')
    
    # Nested dot notation
    flow_type = config.get_nested('neural_posterior.flow_type')
    context_dim = config.get_nested('neural_posterior.context_dim')
    num_layers = config.get_nested('neural_posterior.num_layers')
    
    logger.info(f"Flow Type: {flow_type}")
    logger.info(f"Context Dimension: {context_dim}")
    logger.info(f"Number of Layers: {num_layers}")
    
    # With defaults for deep nesting
    custom_value = config.get_nested('custom.very.deep.param', default='default_value')
    logger.info(f"Custom Deep Param (default): {custom_value}")


def example_6_config_dict_features():
    """Example 6: ConfigDict special features."""
    from ahsd.utils import ConfigDict
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 6: ConfigDict Features")
    logger.info("=" * 80)
    
    # Create ConfigDict
    config = ConfigDict({
        'learning_rate': 1e-4,
        'batch_size': 32,
        'nested': {
            'flow_type': 'nsf',
            'context_dim': 768
        }
    })
    
    # Dict access
    lr_dict = config['learning_rate']
    logger.info(f"Dict access - LR: {lr_dict}")
    
    # Attribute access
    batch_size = config.batch_size
    logger.info(f"Attribute access - Batch Size: {batch_size}")
    
    # Nested access
    flow_type = config.get_nested('nested.flow_type')
    logger.info(f"Nested access - Flow Type: {flow_type}")
    
    # Set nested
    config.set_nested('nested.new_param', 42)
    new_param = config.get_nested('nested.new_param')
    logger.info(f"Set nested - New Param: {new_param}")
    
    # Convert to plain dict
    plain_dict = config.to_dict()
    logger.info(f"Converted to dict - Type: {type(plain_dict)}")


def example_7_merge_configs():
    """Example 7: Merge configurations."""
    from ahsd.utils import UniversalConfigReader, ConfigDict
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 7: Configuration Merging")
    logger.info("=" * 80)
    
    reader = UniversalConfigReader(logger=logger)
    
    # Load base config
    base_config = reader.load('configs/enhanced_training.yaml')
    
    # Create override config
    override = ConfigDict({
        'priority_net': {
            'learning_rate': 5e-5,
            'batch_size': 64,
            'epochs': 100
        },
        'new_param': 'new_value'
    })
    
    logger.info("Original LR:", reader.get(base_config, 'learning_rate', section='priority_net'))
    logger.info("Override LR:", override.priority_net.learning_rate)
    
    # Merge
    merged = reader.merge(base_config, override)
    
    logger.info("Merged LR:", reader.get(merged, 'learning_rate', section='priority_net'))
    logger.info("Merged batch size:", reader.get(merged, 'batch_size', section='priority_net'))
    logger.info("Merged new_param:", merged.get('new_param'))


def example_8_save_config():
    """Example 8: Save configuration to YAML."""
    from ahsd.utils import UniversalConfigReader, ConfigDict
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 8: Saving Configuration")
    logger.info("=" * 80)
    
    reader = UniversalConfigReader(logger=logger)
    config = reader.load('configs/enhanced_training.yaml')
    
    # Modify config
    config.set_nested('priority_net.learning_rate', 1e-5)
    config.set_nested('priority_net.batch_size', 64)
    
    # Save to new file
    output_path = Path('configs/enhanced_training_modified.yaml')
    reader.save(config, output_path)
    
    logger.info(f"✅ Config saved to {output_path}")
    
    # Verify by reloading
    reloaded = reader.load(output_path)
    reloaded_lr = reader.get(reloaded, 'learning_rate', section='priority_net')
    logger.info(f"Reloaded LR: {reloaded_lr}")


def example_9_validation():
    """Example 9: Configuration validation."""
    from ahsd.utils import UniversalConfigReader
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 9: Configuration Validation")
    logger.info("=" * 80)
    
    reader = UniversalConfigReader(logger=logger)
    config = reader.load('configs/enhanced_training.yaml')
    
    # Validate with specific sections
    try:
        is_valid = reader.validate(
            config,
            required_sections=['priority_net', 'neural_posterior'],
            raise_on_error=True
        )
        logger.info(f"✅ Config validation passed: {is_valid}")
    except ValueError as e:
        logger.error(f"❌ Validation failed: {e}")


def example_10_error_handling():
    """Example 10: Error handling."""
    from ahsd.utils import UniversalConfigReader
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 10: Error Handling")
    logger.info("=" * 80)
    
    reader = UniversalConfigReader(logger=logger)
    
    # Missing file
    try:
        config = reader.load('nonexistent_config.yaml')
    except FileNotFoundError as e:
        logger.error(f"❌ File error: {e}")
    
    # Invalid YAML
    try:
        invalid_yaml = "invalid: yaml: content:"
        with open('/tmp/invalid.yaml', 'w') as f:
            f.write(invalid_yaml)
        config = reader.load('/tmp/invalid.yaml')
    except Exception as e:
        logger.error(f"❌ Parse error handled gracefully")
    
    # Type conversion error
    config = reader.load('configs/enhanced_training.yaml')
    result = reader.get(
        config,
        'learning_rate',
        default=1e-4,
        dtype=int,  # Wrong type - will use default
        section='priority_net'
    )
    logger.info(f"Type conversion error handled - result: {result}")


def example_11_model_integration():
    """Example 11: Integration with models."""
    from ahsd.utils import UniversalConfigReader
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 11: Model Integration")
    logger.info("=" * 80)
    
    reader = UniversalConfigReader(logger=logger)
    config = reader.load('configs/enhanced_training.yaml')
    
    # Get model configs
    pn_config = reader.get_section(config, 'priority_net')
    np_config = reader.get_section(config, 'neural_posterior')
    
    # Extract key parameters for models
    logger.info("\n🧠 PriorityNet Parameters:")
    logger.info(f"  Learning Rate: {reader.get(pn_config, 'learning_rate', dtype=float)}")
    logger.info(f"  Batch Size: {reader.get(pn_config, 'batch_size', dtype=int)}")
    logger.info(f"  Epochs: {reader.get(pn_config, 'epochs', dtype=int)}")
    
    logger.info("\n🌊 Neural PE Parameters:")
    logger.info(f"  Flow Type: {reader.get(np_config, 'flow_type')}")
    logger.info(f"  Context Dim: {reader.get(np_config, 'context_dim', dtype=int)}")
    logger.info(f"  Num Layers: {reader.get(np_config, 'num_layers', dtype=int)}")
    
    # These would be passed to model initialization
    # model = SomeModel(
    #     learning_rate=reader.get(pn_config, 'learning_rate', dtype=float),
    #     batch_size=reader.get(pn_config, 'batch_size', dtype=int),
    #     config=config
    # )


def main():
    """Run all examples."""
    
    # Check if config exists
    config_path = Path('configs/enhanced_training.yaml')
    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        logger.error("Please run from project root directory")
        return
    
    examples = [
        ("Basic Loading", example_1_basic_loading),
        ("Reader API", example_2_reader_api),
        ("Section Access", example_3_section_access),
        ("Defaults & Types", example_4_with_defaults),
        ("Nested Access", example_5_nested_access),
        ("ConfigDict Features", example_6_config_dict_features),
        ("Merge Configs", example_7_merge_configs),
        ("Save Config", example_8_save_config),
        ("Validation", example_9_validation),
        ("Error Handling", example_10_error_handling),
        ("Model Integration", example_11_model_integration),
    ]
    
    logger.info("\n" + "🚀" * 40)
    logger.info("UNIVERSAL CONFIG READER EXAMPLES")
    logger.info("🚀" * 40)
    
    for i, (name, example_func) in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            logger.error(f"❌ Example {i} ({name}) failed: {e}", exc_info=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ All examples completed")
    logger.info("=" * 80 + "\n")


if __name__ == '__main__':
    main()
