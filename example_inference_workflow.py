#!/usr/bin/env python3
"""
Example: Real Gravitational Wave Data → Parameter Estimation

This shows the complete workflow:
1. Load real strain data (or simulate realistic data)
2. Run inference through trained model
3. Get predicted parameters with uncertainties
"""

import torch
import numpy as np
import logging
from pathlib import Path
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# STEP 1: PREPARE REAL STRAIN DATA
# ==============================================================================

def prepare_real_strain_data():
    """
    Example 1: Simulated realistic gravitational wave data
    (In practice, you'd load actual GWOSC/LIGO data)
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Prepare Real Strain Data")
    logger.info("=" * 80)
    
    # Real GW data specs from LIGO/Virgo
    sample_rate = 4096  # Hz
    duration = 4        # seconds
    n_samples = sample_rate * duration  # 16384 samples
    
    # Realistic noise level (LIGO sensitivity)
    noise_std = 1e-23  # strain units
    
    # Create mock strain data for H1 and L1 detectors
    # In reality, you'd download this from GWOSC or load from files
    strain_h1 = np.random.normal(0, noise_std, n_samples).astype(np.float32)
    strain_l1 = np.random.normal(0, noise_std, n_samples).astype(np.float32)
    
    # Optionally inject a real signal (simulate a GW event)
    # For this example, let's inject a realistic BBH signal
    inject_signal = True
    if inject_signal:
        # Parameters of injected signal (like a real GW event)
        true_params = {
            'mass_1': 35.0,           # Primary mass (solar masses)
            'mass_2': 30.0,           # Secondary mass (solar masses)
            'distance': 400.0,        # Luminosity distance (Mpc)
            'ra': 2.5,                # Right ascension (radians)
            'dec': -0.3,              # Declination (radians)
            'theta_jn': 1.2,          # Inclination angle (radians)
            'psi': 0.8,               # Polarization angle (radians)
            'phase': 1.5,             # Orbital phase (radians)
            'geocent_time': 0.5       # Time offset (seconds)
        }
        
        logger.info(f"✓ Injected simulated GW signal with parameters:")
        for key, val in true_params.items():
            logger.info(f"    {key:20s}: {val:10.3f}")
        
        # Generate and inject signal (simplified - just add amplitude modulation)
        signal_amplitude = 1e-21 * (35.0 * 30.0) / (400.0 ** 1.5)
        t = np.arange(n_samples) / sample_rate
        signal = signal_amplitude * np.sin(2 * np.pi * 100 * t)  # 100 Hz carrier
        signal = signal * np.exp(-(t - true_params['geocent_time'])**2 / 0.1)  # Gaussian envelope
        
        strain_h1 += signal.astype(np.float32)
        strain_l1 += signal.astype(np.float32) * 0.9  # Slightly different amplitude at L1
    
    # Convert to PyTorch tensor
    # Shape: [batch=1, detectors=2, samples=16384]
    strain_data = torch.tensor(
        [[strain_h1, strain_l1]],
        dtype=torch.float32
    )
    
    logger.info(f"✓ Created strain data:")
    logger.info(f"    Shape: {strain_data.shape} (batch, detectors, samples)")
    logger.info(f"    Detectors: H1, L1")
    logger.info(f"    Duration: {duration}s at {sample_rate} Hz")
    logger.info(f"    Noise level: {noise_std:.2e} strain")
    
    return strain_data, true_params if inject_signal else None


# ==============================================================================
# STEP 2: LOAD TRAINED MODEL
# ==============================================================================

def load_trained_model(device='cuda'):
    """Load the trained OverlapNeuralPE model"""
    from src.ahsd.models.overlap_neuralpe import OverlapNeuralPE
    
    logger.info("=" * 80)
    logger.info("STEP 2: Load Trained Model")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        config_path = Path('configs/enhanced_training.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✓ Loaded config from {config_path}")
        
        # Define parameter names
        param_names = [
            'mass_1', 'mass_2', 'luminosity_distance',
            'ra', 'dec', 'theta_jn', 'psi', 'phase', 'geocent_time'
        ]
        
        # Initialize model with required arguments
        priority_net_path = Path('models/priority_net_checkpoint.pt')
        model = OverlapNeuralPE(
            param_names=param_names,
            priority_net_path=str(priority_net_path),
            config=config,
            device=device
        )
        logger.info("✓ Initialized OverlapNeuralPE model")
        
        # Load trained checkpoint
        checkpoint_path = Path('models/neural_pe/best_model.pth')
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✓ Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning(f"⚠ Checkpoint not found at {checkpoint_path}")
            logger.warning("  Model will use random initialization")
        
        model.eval()  # Set to evaluation mode
        logger.info("✓ Model set to evaluation mode")
        
        return model
    
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        logger.error(f"Error details: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# ==============================================================================
# STEP 3: RUN INFERENCE
# ==============================================================================

def run_inference(model, strain_data, device='cuda'):
    """Run inference to extract signals and estimate parameters"""
    logger.info("=" * 80)
    logger.info("STEP 3: Run Inference")
    logger.info("=" * 80)
    
    strain_data = strain_data.to(device)
    
    with torch.no_grad():
        # Method 1: Extract overlapping signals with iterative subtraction
        logger.info("\n[Method 1] Iterative Signal Extraction")
        logger.info("-" * 40)
        
        extraction_result = model.extract_overlapping_signals(
            strain_data=strain_data,
            true_params=None,  # No ground truth for real data
            training=False
        )
        
        logger.info(f"✓ Extraction complete:")
        logger.info(f"    Signals extracted: {extraction_result['iterations']}")
        logger.info(f"    Residual power: {extraction_result['residual'].abs().max():.3e}")
        
        # Method 2: Sample posterior distribution
        logger.info("\n[Method 2] Posterior Sampling")
        logger.info("-" * 40)
        
        posterior = model.sample_posterior(
            strain_data=strain_data,
            n_samples=1000,  # Draw 1000 samples from posterior
            batch_size=100
        )
        
        logger.info(f"✓ Posterior sampling complete:")
        logger.info(f"    Samples shape: {posterior['samples'].shape}")
        logger.info(f"    (batch=1, n_samples=1000, n_params=9)")
    
    return extraction_result, posterior


# ==============================================================================
# STEP 4: EXTRACT AND DISPLAY RESULTS
# ==============================================================================

def display_results(extraction_result, posterior, true_params=None):
    """Extract and display predicted parameters with uncertainties"""
    logger.info("=" * 80)
    logger.info("STEP 4: Results - Predicted Parameters")
    logger.info("=" * 80)
    
    # Parameter names and units
    param_names = [
        'mass_1',           # Solar masses
        'mass_2',           # Solar masses
        'luminosity_distance',  # Mpc
        'ra',               # Radians
        'dec',              # Radians
        'theta_jn',         # Radians
        'psi',              # Radians
        'phase',            # Radians
        'geocent_time'      # Seconds
    ]
    
    units = [
        'M☉', 'M☉', 'Mpc',
        'rad', 'rad', 'rad', 'rad', 'rad', 's'
    ]
    
    # Get posterior samples for first event in batch
    samples = posterior['samples'][0]  # [1000, 9]
    
    # Compute statistics
    logger.info("\n📊 PARAMETER ESTIMATES:")
    logger.info("-" * 100)
    logger.info(f"{'Parameter':<25} {'Median':<15} {'Std Dev':<15} {'90% Credible':<30}")
    logger.info("-" * 100)
    
    results = {}
    
    for i, (name, unit) in enumerate(zip(param_names, units)):
        param_samples = samples[:, i].cpu().numpy()
        
        # Compute statistics
        median = np.median(param_samples)
        std = np.std(param_samples)
        lower = np.percentile(param_samples, 5)
        upper = np.percentile(param_samples, 95)
        
        results[name] = {
            'median': median,
            'std': std,
            'lower_90': lower,
            'upper_90': upper,
            'samples': param_samples
        }
        
        credible_interval = f"[{lower:.3f}, {upper:.3f}]"
        
        display_name = f"{name} ({unit})"
        logger.info(
            f"{display_name:<25} {median:>12.3f}±{std:>8.3f}  {credible_interval:>30}"
        )
    
    # Compare with injected values if available
    if true_params:
        logger.info("\n" + "=" * 100)
        logger.info("📈 COMPARISON WITH INJECTED VALUES:")
        logger.info("-" * 100)
        logger.info(f"{'Parameter':<25} {'Injected':<15} {'Predicted':<15} {'Error':<15} {'Within 90%?'}")
        logger.info("-" * 100)
        
        for name in param_names:
            if name in true_params:
                true_val = true_params[name]
                pred_val = results[name]['median']
                error = abs(pred_val - true_val)
                within = (results[name]['lower_90'] <= true_val <= results[name]['upper_90'])
                
                status = "✓ YES" if within else "✗ NO"
                
                logger.info(
                    f"{name:<25} {true_val:>12.3f}     {pred_val:>12.3f}     "
                    f"{error:>12.3f}     {status:>10}"
                )
    
    return results


# ==============================================================================
# STEP 5: VISUALIZATION
# ==============================================================================

def visualize_results(posterior, results, save_path='outputs/inference_results.pdf'):
    """Create visualization of posterior distributions"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualization")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Visualization")
    logger.info("=" * 80)
    
    param_names = list(results.keys())
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, name in enumerate(param_names):
        ax = axes[idx]
        samples = results[name]['samples']
        
        # Plot histogram
        ax.hist(samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add median and credible interval
        median = results[name]['median']
        lower = results[name]['lower_90']
        upper = results[name]['upper_90']
        
        ax.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median: {median:.3f}')
        ax.axvline(lower, color='orange', linestyle=':', linewidth=1.5, label='90% CI')
        ax.axvline(upper, color='orange', linestyle=':', linewidth=1.5)
        
        ax.set_xlabel(name)
        ax.set_ylabel('Probability Density')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Create output directory if needed
    Path('outputs').mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization to {save_path}")
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "EXAMPLE: Real GW Data → Parameter Estimation with Neural PE".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}\n")
    
    # Step 1: Prepare strain data
    strain_data, true_params = prepare_real_strain_data()
    
    # Step 2: Load model
    model = load_trained_model(device=device)
    
    if model is not None:
        # Step 3: Run inference
        extraction_result, posterior = run_inference(model, strain_data, device=device)
        
        # Step 4: Display results
        results = display_results(extraction_result, posterior, true_params)
        
        # Step 5: Visualize
        visualize_results(posterior, results)
    
    logger.info("\n✅ Inference complete!")
    logger.info("\nOutput includes:")
    logger.info("  • Predicted parameter values (median)")
    logger.info("  • Parameter uncertainties (standard deviation)")
    logger.info("  • 90% credible intervals (5th-95th percentiles)")
    logger.info("  • Full posterior distribution (1000 samples)")
    logger.info("  • Comparison with true injected values")
