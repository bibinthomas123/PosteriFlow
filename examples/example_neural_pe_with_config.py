"""
Example: Using OverlapNeuralPE with Universal Config Reader.

Demonstrates how the Neural PE model now integrates with the universal
configuration system for consistent, centralized config management.
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_load_and_initialize():
    """Example 1: Load config and initialize Neural PE model."""
    from ahsd.utils import load_config
    from ahsd.models.overlap_neuralpe import OverlapNeuralPE
    
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Load Config and Initialize Neural PE")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config('configs/enhanced_training.yaml')
    
    # Define parameter names (example)
    param_names = [
        'mass_1', 'mass_2', 'luminosity_distance',
        'geocent_time', 'ra', 'dec',
        'theta_jn', 'psi', 'phase', 'a_1', 'a_2'
    ]
    
    # Initialize model with config
    try:
        model = OverlapNeuralPE(
            param_names=param_names,
            priority_net_path='models/priority_net/priority_net_best.pth',
            config=config,  # Pass loaded ConfigDict
            device='cpu',  # Use CPU for example
            event_type='BBH'
        )
        
        logger.info(f"✅ Model initialized successfully")
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"   Context dim: {model.context_dim}")
        logger.info(f"   Flow layers: {model.n_flow_layers}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}", exc_info=True)


def example_2_from_yaml_path():
    """Example 2: Initialize Neural PE directly from YAML path."""
    from ahsd.models.overlap_neuralpe import OverlapNeuralPE
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: Initialize from YAML Path")
    logger.info("=" * 80)
    
    param_names = [
        'mass_1', 'mass_2', 'luminosity_distance',
        'geocent_time', 'ra', 'dec',
        'theta_jn', 'psi', 'phase', 'a_1', 'a_2'
    ]
    
    try:
        # Pass YAML path directly - model will load it internally
        model = OverlapNeuralPE(
            param_names=param_names,
            priority_net_path='models/priority_net/priority_net_best.pth',
            config='configs/enhanced_training.yaml',  # Path to YAML
            device='cpu',
            event_type='BNS'
        )
        
        logger.info(f"✅ Model initialized from YAML path")
        logger.info(f"   Event type: {model.event_type}")
        logger.info(f"   Dropout: {model.dropout_rate}")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}", exc_info=True)


def example_3_config_override():
    """Example 3: Load config with overrides."""
    from ahsd.utils import UniversalConfigReader, ConfigDict
    from ahsd.models.overlap_neuralpe import OverlapNeuralPE
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: Config with Overrides")
    logger.info("=" * 80)
    
    reader = UniversalConfigReader(logger=logger)
    
    # Load base config
    config = reader.load('configs/enhanced_training.yaml')
    
    # Create override
    override = ConfigDict({
        'neural_posterior': {
            'context_dim': 512,  # Override: smaller context
            'num_layers': 4,     # Override: fewer flow layers
            'dropout': 0.2,      # Override: higher dropout
        }
    })
    
    # Merge configs
    merged = reader.merge(config, override)
    
    logger.info(f"Original context_dim: {reader.get(config, 'context_dim', section='neural_posterior')}")
    logger.info(f"Merged context_dim: {reader.get(merged, 'context_dim', section='neural_posterior')}")
    
    param_names = [
        'mass_1', 'mass_2', 'luminosity_distance',
        'geocent_time', 'ra', 'dec',
        'theta_jn', 'psi', 'phase', 'a_1', 'a_2'
    ]
    
    try:
        model = OverlapNeuralPE(
            param_names=param_names,
            priority_net_path='models/priority_net/priority_net_best.pth',
            config=merged,
            device='cpu',
            event_type='NSBH'
        )
        
        logger.info(f"✅ Model with merged config initialized")
        logger.info(f"   Context dim: {model.context_dim}")
        logger.info(f"   Flow layers: {model.n_flow_layers}")
        logger.info(f"   Dropout: {model.dropout_rate}")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}", exc_info=True)


def example_4_access_model_config():
    """Example 4: Access configuration from initialized model."""
    from ahsd.utils import load_config
    from ahsd.models.overlap_neuralpe import OverlapNeuralPE
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 4: Access Configuration from Model")
    logger.info("=" * 80)
    
    config = load_config('configs/enhanced_training.yaml')
    
    param_names = [
        'mass_1', 'mass_2', 'luminosity_distance',
        'geocent_time', 'ra', 'dec',
        'theta_jn', 'psi', 'phase', 'a_1', 'a_2'
    ]
    
    try:
        model = OverlapNeuralPE(
            param_names=param_names,
            priority_net_path='models/priority_net/priority_net_best.pth',
            config=config,
            device='cpu',
            event_type='BBH'
        )
        
        # Access config from model
        logger.info("Config accessible from model:")
        logger.info(f"  model.config type: {type(model.config)}")
        logger.info(f"  model.context_dim: {model.context_dim}")
        logger.info(f"  model.n_flow_layers: {model.n_flow_layers}")
        
        # Use reader to access more config values
        reader = model._reader
        
        flow_type = reader.get(
            model.config,
            'flow_type',
            default='nsf',
            section='neural_posterior'
        )
        logger.info(f"  Flow type: {flow_type}")
        
        # Get entire section
        np_section = reader.get_section(model.config, 'neural_posterior')
        logger.info(f"  Neural posterior section keys: {list(np_section.keys())[:5]}...")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}", exc_info=True)


def example_5_validate_config():
    """Example 5: Validate config before using it."""
    from ahsd.utils import UniversalConfigReader
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 5: Validate Configuration")
    logger.info("=" * 80)
    
    reader = UniversalConfigReader(logger=logger)
    config = reader.load('configs/enhanced_training.yaml')
    
    # Validate
    try:
        is_valid = reader.validate(
            config,
            required_sections=['priority_net', 'neural_posterior'],
            raise_on_error=True
        )
        logger.info(f"✅ Configuration is valid: {is_valid}")
    except ValueError as e:
        logger.error(f"❌ Config validation failed: {e}")


def example_6_log_config():
    """Example 6: Log configuration for debugging."""
    from ahsd.utils import UniversalConfigReader
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 6: Log Configuration")
    logger.info("=" * 80)
    
    reader = UniversalConfigReader(logger=logger)
    config = reader.load('configs/enhanced_training.yaml')
    
    # Log full config (limited depth)
    reader.log_config(config, max_depth=2)


def example_7_different_event_types():
    """Example 7: Initialize models for different event types with same config."""
    from ahsd.utils import load_config
    from ahsd.models.overlap_neuralpe import OverlapNeuralPE
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 7: Different Event Types")
    logger.info("=" * 80)
    
    config = load_config('configs/enhanced_training.yaml')
    
    param_names = [
        'mass_1', 'mass_2', 'luminosity_distance',
        'geocent_time', 'ra', 'dec',
        'theta_jn', 'psi', 'phase', 'a_1', 'a_2'
    ]
    
    event_types = ['BBH', 'BNS', 'NSBH']
    models = {}
    
    for event_type in event_types:
        try:
            model = OverlapNeuralPE(
                param_names=param_names,
                priority_net_path='models/priority_net/priority_net_best.pth',
                config=config,
                device='cpu',
                event_type=event_type
            )
            models[event_type] = model
            logger.info(f"✅ {event_type}: Context={model.context_dim}, Layers={model.n_flow_layers}")
        except Exception as e:
            logger.error(f"❌ {event_type} failed: {e}")
    
    logger.info(f"\n✅ Created {len(models)} models")


def example_8_config_consistency():
    """Example 8: Ensure config consistency across components."""
    from ahsd.utils import UniversalConfigReader
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 8: Config Consistency")
    logger.info("=" * 80)
    
    reader = UniversalConfigReader(logger=logger)
    config = reader.load('configs/enhanced_training.yaml')
    
    # Check consistency
    logger.info("Checking config consistency...")
    
    # Neural PE parameters
    np_context_dim = reader.get(config, 'context_dim', section='neural_posterior', dtype=int)
    np_num_layers = reader.get(config, 'num_layers', section='neural_posterior', dtype=int)
    
    # Flow parameters
    flow_context_dim = reader.get(config, 'context_dim', section='flow_config', dtype=int) or np_context_dim
    
    logger.info(f"  Neural PE context_dim: {np_context_dim}")
    logger.info(f"  Flow context_dim: {flow_context_dim}")
    logger.info(f"  ✅ Consistency: {np_context_dim == flow_context_dim}")
    
    logger.info(f"  Neural PE num_layers: {np_num_layers}")
    
    # Training parameters
    pn_learning_rate = reader.get(config, 'learning_rate', section='priority_net', dtype=float)
    logger.info(f"  PriorityNet learning_rate: {pn_learning_rate:.2e}")


def main():
    """Run all examples."""
    
    # Check if config exists
    config_path = Path('configs/enhanced_training.yaml')
    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        logger.error("Please run from project root directory")
        return
    
    examples = [
        ("Load and Initialize", example_1_load_and_initialize),
        ("From YAML Path", example_2_from_yaml_path),
        ("Config Overrides", example_3_config_override),
        ("Access from Model", example_4_access_model_config),
        ("Validate Config", example_5_validate_config),
        ("Log Config", example_6_log_config),
        ("Different Event Types", example_7_different_event_types),
        ("Config Consistency", example_8_config_consistency),
    ]
    
    logger.info("\n" + "🚀" * 40)
    logger.info("NEURAL PE WITH UNIVERSAL CONFIG EXAMPLES")
    logger.info("🚀" * 40)
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            logger.error(f"❌ Example ({name}) failed: {e}", exc_info=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ All examples completed")
    logger.info("=" * 80 + "\n")


if __name__ == '__main__':
    main()
