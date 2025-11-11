#!/usr/bin/env python3
"""
Configuration Audit Tool
Validates all config parameters and checks if they're correctly passed to PriorityNet
"""

import yaml
import sys
import logging
from pathlib import Path
import json
from typing import Dict, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# EXPECTED CONFIG STRUCTURE (Ground truth)
# ==============================================================================

PRIORITY_NET_REQUIRED = {
    'hidden_dims': (list, "Architecture: Hidden layer dimensions"),
    'dropout': (float, "Architecture: Dropout rate"),
    'use_strain': (bool, "Flag: Enable strain encoder"),
    'use_edge_conditioning': (bool, "Flag: Enable edge conditioning"),
    'n_edge_types': (int, "Architecture: Number of edge types"),
    'use_transformer_encoder': (bool, "Flag: Use Transformer encoder"),
    'optimizer': (str, "Optimizer type"),
    'learning_rate': (float, "Learning rate"),
    'weight_decay': (float, "Weight decay"),
    'batch_size': (int, "Batch size"),
    'epochs': (int, "Number of epochs"),
    'patience': (int, "Early stopping patience"),
    'warmup_epochs': (int, "Warmup epochs"),
    'warmup_start_factor': (float, "Warmup start factor"),
    'scheduler': (str, "Scheduler type"),
    'scheduler_patience': (int, "Scheduler patience"),
    'scheduler_factor': (float, "Scheduler factor"),
    'min_lr': (float, "Minimum learning rate"),
    'ranking_weight': (float, "Loss: Ranking weight"),
    'mse_weight': (float, "Loss: MSE weight"),
    'uncertainty_weight': (float, "Loss: Uncertainty weight"),
    'use_snr_weighting': (bool, "Flag: Use SNR weighting"),
    'loss_scale_factor': (float, "Loss scale factor"),
    'gradient_clip_norm': (float, "Gradient clipping norm"),
    'gradient_log_threshold': (float, "Gradient log threshold"),
    'label_smoothing': (float, "Label smoothing"),
    'attention_num_heads': (int, "Attention: Number of heads"),
    'attention_dropout': (float, "Attention: Dropout"),
    'overlap_importance_hidden': (int, "Overlap: Importance hidden dim"),
}

PRIORITY_NET_OPTIONAL = {
    'use_modal_fusion': (bool, "Flag: Use modal fusion (can be false)"),
    'overlap_use_attention': (bool, "Flag: Use overlap attention (can be false)"),
}

NEURAL_POSTERIOR_REQUIRED = {
    'param_names': (list, "Parameter names for inference"),
    'context_dim': (int, "Context dimension"),
    'n_flow_layers': (int, "Number of flow layers"),
    'max_iterations': (int, "Max iterations"),
}

DATA_REQUIRED = {
    'sample_rate': (int, "Sample rate (Hz)"),
    'segment_duration': (float, "Segment duration (s)"),
    'f_low': (float, "Low frequency (Hz)"),
    'f_high': (float, "High frequency (Hz)"),
    'validation_split': (float, "Validation split fraction"),
    'test_split': (float, "Test split fraction"),
}

MONITORING_REQUIRED = {
    'save_frequency': (int, "Save frequency"),
    'log_frequency': (int, "Log frequency"),
    'early_stopping': (bool, "Early stopping enabled"),
}

OUTPUT_REQUIRED = {
    'save_best_only': (bool, "Save best only"),
    'save_intermediate': (bool, "Save intermediate"),
    'generate_plots': (bool, "Generate plots"),
}


def validate_type(key: str, value: Any, expected_type: type) -> Tuple[bool, str]:
    """Validate single config value type."""
    if isinstance(value, expected_type):
        return True, ""
    
    # Special case: int can be float
    if expected_type == float and isinstance(value, int):
        return True, f"(auto-converted int→float)"
    
    return False, f"Expected {expected_type.__name__}, got {type(value).__name__}"


def validate_section(config: Dict, section_name: str, schema: Dict, required: bool = True) -> Tuple[bool, list]:
    """Validate a config section."""
    errors = []
    
    section = config.get(section_name)
    if section is None:
        if required:
            errors.append(f"❌ MISSING SECTION: {section_name}")
        return len(errors) == 0, errors
    
    if not isinstance(section, dict):
        errors.append(f"❌ {section_name} must be a dictionary, got {type(section).__name__}")
        return False, errors
    
    for key, (expected_type, description) in schema.items():
        if key not in section:
            errors.append(f"❌ MISSING KEY: {section_name}.{key} ({description})")
        else:
            value = section[key]
            is_valid, msg = validate_type(key, value, expected_type)
            if not is_valid:
                errors.append(f"❌ INVALID TYPE: {section_name}.{key} - {msg}")
            else:
                if msg:
                    logger.debug(f"   {key}: {msg}")
    
    return len(errors) == 0, errors


def print_config_audit(config: Dict) -> None:
    """Print a detailed audit of all config values."""
    logger.info("\n" + "="*80)
    logger.info("📋 CONFIGURATION AUDIT - ALL VALUES")
    logger.info("="*80)
    
    pn_config = config.get('priority_net', {})
    
    # PRIORITY NET
    logger.info("\n🔴 PRIORITY NET (critical flags):")
    flags_to_check = [
        ('use_strain', "Temporal strain encoder"),
        ('use_edge_conditioning', "Edge-case conditioning"),
        ('use_transformer_encoder', "Transformer-based encoder"),
        ('use_snr_weighting', "SNR-weighted loss"),
        ('use_modal_fusion', "Multi-modal fusion (attention)"),
        ('overlap_use_attention', "Overlap attention"),
        ('label_smoothing', "Label smoothing applied"),
    ]
    
    for key, desc in flags_to_check:
        value = pn_config.get(key, "NOT SET")
        status = "✓" if value is True else "✗" if value is False else "⚠"
        logger.info(f"   {status} {key:30s} = {str(value):10s} ({desc})")
    
    # ARCHITECTURE
    logger.info("\n🏗️  ARCHITECTURE:")
    logger.info(f"   hidden_dims:              {pn_config.get('hidden_dims', 'NOT SET')}")
    logger.info(f"   dropout:                  {pn_config.get('dropout', 'NOT SET')}")
    logger.info(f"   n_edge_types:             {pn_config.get('n_edge_types', 'NOT SET')}")
    
    # OPTIMIZER
    logger.info("\n🎯 OPTIMIZER:")
    logger.info(f"   learning_rate:            {pn_config.get('learning_rate', 'NOT SET'):.2e}")
    logger.info(f"   weight_decay:             {pn_config.get('weight_decay', 'NOT SET'):.2e}")
    logger.info(f"   optimizer:                {pn_config.get('optimizer', 'NOT SET')}")
    
    # TRAINING
    logger.info("\n📊 TRAINING:")
    logger.info(f"   batch_size:               {pn_config.get('batch_size', 'NOT SET')}")
    logger.info(f"   epochs:                   {pn_config.get('epochs', 'NOT SET')}")
    logger.info(f"   patience:                 {pn_config.get('patience', 'NOT SET')}")
    
    # WARMUP
    logger.info("\n🔥 WARMUP:")
    logger.info(f"   warmup_epochs:            {pn_config.get('warmup_epochs', 'NOT SET')}")
    logger.info(f"   warmup_start_factor:      {pn_config.get('warmup_start_factor', 'NOT SET')}")
    
    # SCHEDULER
    logger.info("\n⏱️  SCHEDULER:")
    logger.info(f"   scheduler:                {pn_config.get('scheduler', 'NOT SET')}")
    logger.info(f"   scheduler_patience:       {pn_config.get('scheduler_patience', 'NOT SET')}")
    logger.info(f"   scheduler_factor:         {pn_config.get('scheduler_factor', 'NOT SET')}")
    logger.info(f"   min_lr:                   {pn_config.get('min_lr', 'NOT SET'):.2e}")
    
    # LOSS
    logger.info("\n🎲 LOSS FUNCTION:")
    logger.info(f"   ranking_weight:           {pn_config.get('ranking_weight', 'NOT SET')}")
    logger.info(f"   mse_weight:               {pn_config.get('mse_weight', 'NOT SET')}")
    logger.info(f"   uncertainty_weight:       {pn_config.get('uncertainty_weight', 'NOT SET')}")
    weight_sum = (pn_config.get('ranking_weight', 0) + 
                  pn_config.get('mse_weight', 0) + 
                  pn_config.get('uncertainty_weight', 0))
    logger.info(f"   ↳ Sum of weights:         {weight_sum:.3f} (should be ~1.0)")
    logger.info(f"   use_snr_weighting:        {pn_config.get('use_snr_weighting', 'NOT SET')}")
    logger.info(f"   loss_scale_factor:        {pn_config.get('loss_scale_factor', 'NOT SET'):.4f}")
    
    # GRADIENT
    logger.info("\n🔄 GRADIENT:")
    logger.info(f"   gradient_clip_norm:       {pn_config.get('gradient_clip_norm', 'NOT SET')}")
    logger.info(f"   gradient_log_threshold:   {pn_config.get('gradient_log_threshold', 'NOT SET')}")
    
    # ATTENTION
    logger.info("\n✨ ATTENTION:")
    logger.info(f"   attention_num_heads:      {pn_config.get('attention_num_heads', 'NOT SET')}")
    logger.info(f"   attention_dropout:        {pn_config.get('attention_dropout', 'NOT SET')}")
    
    # OVERLAP
    logger.info("\n🔗 OVERLAP:")
    logger.info(f"   overlap_importance_hidden: {pn_config.get('overlap_importance_hidden', 'NOT SET')}")
    
    # DATA
    logger.info("\n📂 DATA:")
    data_config = config.get('data', {})
    logger.info(f"   sample_rate:              {data_config.get('sample_rate', 'NOT SET')} Hz")
    logger.info(f"   segment_duration:         {data_config.get('segment_duration', 'NOT SET')} s")
    logger.info(f"   f_low:                    {data_config.get('f_low', 'NOT SET')} Hz")
    logger.info(f"   f_high:                   {data_config.get('f_high', 'NOT SET')} Hz")
    
    # MONITORING
    logger.info("\n📈 MONITORING:")
    monitoring = config.get('monitoring', {})
    logger.info(f"   early_stopping:           {monitoring.get('early_stopping', 'NOT SET')}")
    logger.info(f"   save_frequency:           {monitoring.get('save_frequency', 'NOT SET')}")
    
    # OUTPUT
    logger.info("\n💾 OUTPUT:")
    output = config.get('output', {})
    logger.info(f"   save_best_only:           {output.get('save_best_only', 'NOT SET')}")
    logger.info(f"   save_intermediate:        {output.get('save_intermediate', 'NOT SET')}")
    logger.info(f"   generate_plots:           {output.get('generate_plots', 'NOT SET')}")
    
    logger.info("="*80 + "\n")


def main():
    config_path = Path('configs/enhanced_training.yaml')
    
    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config
    logger.info(f"📂 Loading config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validate sections
    logger.info("\n" + "="*80)
    logger.info("🔍 VALIDATION CHECKS")
    logger.info("="*80)
    
    all_errors = []
    
    # Check priority_net (required)
    valid, errors = validate_section(config, 'priority_net', PRIORITY_NET_REQUIRED, required=True)
    all_errors.extend(errors)
    
    # Check optional flags
    pn_config = config.get('priority_net', {})
    for key, (expected_type, desc) in PRIORITY_NET_OPTIONAL.items():
        if key in pn_config:
            value = pn_config[key]
            is_valid, msg = validate_type(key, value, expected_type)
            if not is_valid:
                all_errors.append(f"❌ INVALID TYPE: priority_net.{key} - {msg}")
    
    # Check neural_posterior
    valid_np, errors_np = validate_section(config, 'neural_posterior', NEURAL_POSTERIOR_REQUIRED, required=False)
    all_errors.extend(errors_np)
    
    # Check data
    valid_data, errors_data = validate_section(config, 'data', DATA_REQUIRED, required=True)
    all_errors.extend(errors_data)
    
    # Check monitoring
    valid_mon, errors_mon = validate_section(config, 'monitoring', MONITORING_REQUIRED, required=False)
    all_errors.extend(errors_mon)
    
    # Check output
    valid_out, errors_out = validate_section(config, 'output', OUTPUT_REQUIRED, required=False)
    all_errors.extend(errors_out)
    
    # Print validation results
    if all_errors:
        logger.error("\n⚠️  VALIDATION FAILED:\n")
        for error in all_errors:
            logger.error(f"   {error}")
        logger.error("")
    else:
        logger.info("\n✅ All required sections and keys are present!")
    
    # Print detailed audit
    print_config_audit(config)
    
    # Check for "stupid" configurations
    logger.info("="*80)
    logger.info("🚨 SANITY CHECKS (Removing stupid configs)")
    logger.info("="*80)
    
    warnings = []
    
    pn = pn_config
    
    # Check loss weight sum
    loss_sum = pn.get('ranking_weight', 0) + pn.get('mse_weight', 0) + pn.get('uncertainty_weight', 0)
    if abs(loss_sum - 1.0) > 0.05:
        warnings.append(f"⚠️  Loss weights sum to {loss_sum:.3f}, not 1.0 (ADJUST ranking/mse/uncertainty weights)")
    
    # Check batch size vs total samples (approx)
    if pn.get('batch_size', 32) > 128:
        warnings.append(f"⚠️  batch_size={pn.get('batch_size')} is very large (may cause memory issues)")
    
    # Check epochs vs patience
    if pn.get('patience', 1) == 1:
        warnings.append(f"⚠️  patience=1 is very strict (will stop training after 1 epoch of no improvement)")
    
    # Check warmup vs epochs
    warmup = pn.get('warmup_epochs', 5)
    epochs = pn.get('epochs', 250)
    if warmup >= epochs * 0.3:
        warnings.append(f"⚠️  warmup_epochs={warmup} is {warmup/epochs*100:.0f}% of total epochs (too high)")
    
    # Check scheduler patience vs epochs
    sched_patience = pn.get('scheduler_patience', 5)
    if sched_patience > epochs // 10:
        warnings.append(f"⚠️  scheduler_patience={sched_patience} is very high relative to epochs={epochs}")
    
    # Check learning rates
    lr = pn.get('learning_rate', 1e-4)
    if lr < 1e-6 or lr > 1e-2:
        warnings.append(f"⚠️  learning_rate={lr:.2e} is outside typical range [1e-6, 1e-2]")
    
    # Check dropout
    dropout = pn.get('dropout', 0.12)
    if dropout > 0.5:
        warnings.append(f"⚠️  dropout={dropout} is very high (may underfit)")
    
    # Check hidden dims decreasing
    hidden_dims = pn.get('hidden_dims', [])
    if hidden_dims and hidden_dims != sorted(hidden_dims, reverse=True):
        warnings.append(f"⚠️  hidden_dims={hidden_dims} should be in decreasing order for stability")
    
    if warnings:
        logger.warning("")
        for w in warnings:
            logger.warning(f"   {w}")
    else:
        logger.info("\n✅ All sanity checks passed!")
    
    logger.info("="*80)
    
    return 0 if not all_errors else 1


if __name__ == '__main__':
    sys.exit(main())
