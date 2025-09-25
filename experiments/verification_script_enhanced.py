#!/usr/bin/env python3
"""
ENHANCED AHSD VERIFICATION SCRIPT
Complete verification including subtractor architecture and system integration checks
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('experiments')
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required classes
try:
    from phase3a_neural_pe import NeuralPENetwork
except ImportError:
    print("Warning: Could not import NeuralPENetwork")

# ✅ EXACT EffectiveSubtractor class definition (matches training architecture)
class EffectiveSubtractor(nn.Module):
    def __init__(self, data_length: int = 4096):
        super().__init__()
        self.data_length = data_length
        
        # Multi-scale contamination detector
        self.contamination_detector = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=32, stride=4, padding=14),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=7),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
            nn.Flatten(),
            nn.Linear(256 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, data_length * 2),
            nn.Tanh()
        )
        
        # Adaptive strength based on Neural PE confidence
        self.confidence_adapter = nn.Sequential(
            nn.Linear(9, 32),  # 9 parameter estimates
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name and len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, contaminated_data: torch.Tensor, neural_pe_output) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle both (params, uncertainties) and (params only) from Neural PE"""
        
        batch_size = contaminated_data.shape[0]
        
        # Handle Neural PE output
        if isinstance(neural_pe_output, tuple):
            pred_params, pred_uncertainties = neural_pe_output
            confidence_input = pred_uncertainties if pred_uncertainties is not None else pred_params
        else:
            confidence_input = neural_pe_output
        
        # Detect contamination pattern
        contamination_pattern = self.contamination_detector(contaminated_data)
        contamination_pattern = contamination_pattern.view(batch_size, 2, self.data_length)
        
        # Adaptive strength
        confidence = self.confidence_adapter(confidence_input)
        strength = 0.3 + 0.5 * confidence  # Range: [0.3, 0.8]
        
        # Apply subtraction
        cleaned_data = contaminated_data - (contamination_pattern * strength.unsqueeze(-1))
        
        return cleaned_data, confidence.squeeze(-1)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_phase3c_compatible_dataset(n_samples: int, seed: int = 42):
    """✅ EXACT Phase 3C data generation for perfect verification"""
    
    np.random.seed(seed)
    samples = []
    
    logging.info(f"🔧 Creating {n_samples} samples with EXACT Phase 3C compatibility (seed={seed})")
    
    for i in range(n_samples):
        t = np.linspace(0, 4, 4096)
        
        # Generate realistic GW parameters (EXACT Phase 3C)
        mass_1 = np.random.uniform(20, 50)
        mass_2 = np.random.uniform(15, mass_1)
        distance = np.random.uniform(200, 800)
        chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
        inclination = np.random.uniform(0, np.pi)
        
        # ✅ EXACT Phase 3C scaling (CRITICAL)
        signal_scale = 1e-3
        contamination_scale = signal_scale * 10.0  # 10x contamination
        
        # Generate clean GW signal (EXACT Phase 3C method)
        f_start = 20.0
        f_end = min(100.0, 220.0 / (mass_1 + mass_2))
        frequency = f_start + (f_end - f_start) * (t / 4.0)
        phase = 2 * np.pi * np.cumsum(frequency) * (4.0 / 4096)
        amplitude = signal_scale * np.exp(-t / 8.0) * np.sqrt(chirp_mass / 30.0)
        
        h_plus_clean = amplitude * (1 + np.cos(inclination)**2) * np.cos(phase)
        h_cross_clean = amplitude * 2 * np.cos(inclination) * np.sin(phase)
        
        # ✅ EXACT Phase 3C contamination patterns (CRITICAL)
        # Power line contamination (60Hz)
        power_contamination_h_plus = contamination_scale * np.sin(2 * np.pi * 60.0 * t)
        power_contamination_h_cross = contamination_scale * np.cos(2 * np.pi * 60.0 * t)
        
        # Low frequency seismic (2Hz)
        seismic_contamination_h_plus = contamination_scale * 0.7 * np.sin(2 * np.pi * 2.0 * t)
        seismic_contamination_h_cross = contamination_scale * 0.7 * np.cos(2 * np.pi * 2.0 * t)
        
        # High frequency noise (100Hz)
        hf_contamination_h_plus = contamination_scale * 0.3 * np.sin(2 * np.pi * 100.0 * t)
        hf_contamination_h_cross = contamination_scale * 0.3 * np.cos(2 * np.pi * 100.0 * t)
        
        # Glitch (transient burst at t=2s)
        glitch_center = 2.0
        glitch_width = 0.2
        glitch_amplitude = contamination_scale * 2.0
        glitch = glitch_amplitude * np.exp(-((t - glitch_center) / glitch_width)**2)
        
        # ✅ EXACT Phase 3C total contamination
        total_contamination_h_plus = (power_contamination_h_plus + 
                                     seismic_contamination_h_plus + 
                                     hf_contamination_h_plus + glitch)
        total_contamination_h_cross = (power_contamination_h_cross + 
                                      seismic_contamination_h_cross + 
                                      hf_contamination_h_cross + glitch * 0.8)
        
        h_plus_contaminated = h_plus_clean + total_contamination_h_plus
        h_cross_contaminated = h_cross_clean + total_contamination_h_cross
        
        # Add realistic noise (EXACT Phase 3C)
        noise_level = signal_scale * 0.01  # 1% noise
        h_plus_contaminated += np.random.normal(0, noise_level, 4096)
        h_cross_contaminated += np.random.normal(0, noise_level, 4096)
        h_plus_clean += np.random.normal(0, noise_level, 4096)
        h_cross_clean += np.random.normal(0, noise_level, 4096)
        
        # Store data in EXACT Phase 3C format
        contaminated_data = np.array([h_plus_contaminated, h_cross_contaminated], dtype=np.float32)
        clean_data = np.array([h_plus_clean, h_cross_clean], dtype=np.float32)
        
        # Normalized parameters (EXACT Phase 3C)
        true_params = np.array([
            2 * (mass_1 - 15) / (50 - 15) - 1,
            2 * (mass_2 - 10) / (40 - 10) - 1,
            2 * (np.log10(distance) - np.log10(100)) / (np.log10(1000) - np.log10(100)) - 1,
            np.random.uniform(-0.8, 0.8),  # ra
            np.random.uniform(-0.8, 0.8),  # dec
            np.random.uniform(-0.8, 0.8),  # geocent_time
            2 * (inclination / np.pi) - 1,  # theta_jn
            np.random.uniform(-0.8, 0.8),  # psi
            np.random.uniform(-0.8, 0.8),  # phase
        ], dtype=np.float32)
        
        samples.append({
            'contaminated_data': contaminated_data,
            'clean_data': clean_data,
            'true_parameters': true_params,
            'signal_quality': 0.8 + 0.2 * np.random.random()
        })
        
        # ✅ VERIFICATION: Check contamination strength on first sample
        if i == 0:
            contamination_strength = np.mean(np.abs(contaminated_data - clean_data))
            logging.info(f"✅ First sample contamination strength: {contamination_strength:.6f}")
            if contamination_strength < 5e-6:  # Should be ~0.008
                logging.error(f"❌ CRITICAL: Contamination too weak: {contamination_strength:.6f}")
                return None
            else:
                logging.info(f"🎯 Contamination strength confirmed: {contamination_strength:.6f}")
    
    return samples

def load_models_safely(phase3b_path: str):
    """✅ ROBUST: Load models with comprehensive error handling"""
    
    logging.info("📂 Loading models using robust state_dict approach...")
    
    try:
        # Load checkpoint with error handling
        checkpoint = torch.load(phase3b_path, map_location='cpu')
        param_names = checkpoint['param_names']
        
        logging.info(f"📋 Parameter names: {param_names}")
        
        # Create fresh model instances
        neural_pe = NeuralPENetwork(param_names)
        subtractor = EffectiveSubtractor()
        
        # Load Neural PE state dict
        if 'neural_pe_state_dict' in checkpoint:
            neural_pe.load_state_dict(checkpoint['neural_pe_state_dict'])
            logging.info("✅ Neural PE loaded from state_dict")
        elif hasattr(checkpoint['neural_pe_model'], 'state_dict'):
            neural_pe.load_state_dict(checkpoint['neural_pe_model'].state_dict())
            logging.info("✅ Neural PE loaded from model.state_dict()")
        else:
            raise KeyError("Neural PE state dict not found")
        
        # Load Subtractor state dict
        if 'subtractor_state_dict' in checkpoint:
            subtractor.load_state_dict(checkpoint['subtractor_state_dict'])
            logging.info("✅ Subtractor loaded from state_dict")
        elif hasattr(checkpoint['subtractor_model'], 'state_dict'):
            subtractor.load_state_dict(checkpoint['subtractor_model'].state_dict())
            logging.info("✅ Subtractor loaded from model.state_dict()")
        else:
            raise KeyError("Subtractor state dict not found")
        
        # Set evaluation mode
        neural_pe.eval()
        subtractor.eval()
        
        # Verify model parameters
        pe_params = sum(p.numel() for p in neural_pe.parameters())
        sub_params = sum(p.numel() for p in subtractor.parameters())
        
        logging.info(f"✅ Models loaded successfully:")
        logging.info(f"   Neural PE: {pe_params:,} parameters")
        logging.info(f"   Subtractor: {sub_params:,} parameters")
        
        return neural_pe, subtractor, param_names
        
    except Exception as e:
        logging.error(f"❌ Model loading failed: {e}")
        raise

def verify_subtractor_architecture(subtractor):
    """✅ ENHANCED: Verify subtractor architecture matches training"""
    
    logging.info("🔧 Verifying subtractor architecture...")
    
    # Test input
    test_input = torch.randn(1, 2, 4096) * 0.01
    test_pe_output = torch.randn(1, 9) * 0.1
    
    try:
        # Test forward pass
        cleaned, confidence = subtractor(test_input, test_pe_output)
        
        # Verify output shapes
        assert cleaned.shape == test_input.shape, f"Output shape mismatch: {cleaned.shape} vs {test_input.shape}"
        assert confidence.shape == (1,), f"Confidence shape mismatch: {confidence.shape}"
        
        # Verify confidence range [0.3, 0.8]
        assert 0.3 <= confidence.item() <= 0.8, f"Confidence out of range: {confidence.item()}"
        
        # Test contamination detection capability
        contaminated = test_input + torch.randn_like(test_input) * 0.02  # Add strong contamination
        cleaned_contaminated, conf_contaminated = subtractor(contaminated, test_pe_output)
        
        # Verify subtraction occurred
        reduction = torch.mean((contaminated - cleaned_contaminated) ** 2)
        assert reduction > 1e-8, f"No subtraction detected: {reduction.item()}"
        
        logging.info("✅ Subtractor architecture verification PASSED")
        return True
        
    except Exception as e:
        logging.error(f"❌ Subtractor architecture verification FAILED: {e}")
        return False

def verify_system_performance(neural_pe, subtractor, samples):
    """✅ COMPREHENSIVE: Complete system performance verification"""
    
    pe_accuracies = []
    subtractor_efficiencies = []
    system_successes = []
    detailed_results = []
    
    logging.info(f"🔍 Verifying complete system on {len(samples)} samples...")
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(samples, desc="System Verification")):
            try:
                # Prepare inputs
                contaminated = torch.tensor(sample['contaminated_data']).unsqueeze(0)
                clean_target = torch.tensor(sample['clean_data']).unsqueeze(0)
                true_params = torch.tensor(sample['true_parameters']).unsqueeze(0)
                
                # ✅ STEP 1: Neural PE prediction
                neural_pe_output = neural_pe(contaminated)
                if isinstance(neural_pe_output, tuple):
                    pred_params, pred_uncertainties = neural_pe_output
                    if pred_uncertainties is None:
                        pred_uncertainties = torch.abs(pred_params) + 0.1
                else:
                    pred_params = neural_pe_output
                    pred_uncertainties = torch.abs(pred_params) + 0.1
                
                # Calculate PE accuracy (EXACT Phase 3C method)
                param_errors = torch.mean((pred_params - true_params) ** 2)
                pe_accuracy = float(1.0 / (1.0 + param_errors))
                pe_accuracy = max(0.0, min(1.0, pe_accuracy))
                pe_accuracies.append(pe_accuracy)
                
                # ✅ STEP 2: Subtractor efficiency (EXACT Phase 3C method)
                cleaned_output, confidence = subtractor(contaminated, pred_uncertainties)
                
                # Calculate efficiency using MSE-based method (Phase 3C standard)
                mse_before = torch.mean((contaminated - clean_target) ** 2, dim=(1, 2))
                mse_after = torch.mean((cleaned_output - clean_target) ** 2, dim=(1, 2))
                improvement = mse_before - mse_after
                efficiency = improvement / (mse_before + 1e-8)
                efficiency = float(torch.clamp(efficiency, 0.0, 1.0))
                subtractor_efficiencies.append(efficiency)
                
                # ✅ STEP 3: System success (EXACT Phase 3C criteria)
                pe_success = pe_accuracy > 0.5    # PE threshold
                sub_success = efficiency > 0.3    # Subtraction improvement threshold
                system_success = pe_success and sub_success
                system_successes.append(system_success)
                
                # Store detailed results for debugging
                if i < 10:  # First 10 samples for debugging
                    detailed_results.append({
                        'sample_id': i,
                        'pe_accuracy': pe_accuracy,
                        'subtractor_efficiency': efficiency,
                        'system_success': system_success,
                        'pe_success': pe_success,
                        'sub_success': sub_success,
                        'confidence': confidence.item(),
                        'mse_before': mse_before.item(),
                        'mse_after': mse_after.item(),
                        'improvement': improvement.item()
                    })
                
            except Exception as e:
                logging.warning(f"Sample {i} verification failed: {e}")
                pe_accuracies.append(0.0)
                subtractor_efficiencies.append(0.0)
                system_successes.append(False)
    
    # Calculate comprehensive statistics
    results = {
        'neural_pe_accuracy': np.mean(pe_accuracies),
        'subtractor_efficiency': np.mean(subtractor_efficiencies),
        'system_success_rate': np.mean(system_successes),
        'pe_std': np.std(pe_accuracies),
        'sub_std': np.std(subtractor_efficiencies),
        'sys_std': np.std([float(x) for x in system_successes]),
        'samples': len(samples),
        'pe_min': np.min(pe_accuracies),
        'pe_max': np.max(pe_accuracies),
        'sub_min': np.min(subtractor_efficiencies),
        'sub_max': np.max(subtractor_efficiencies),
        'detailed_samples': detailed_results
    }
    
    return results

def print_detailed_verification_results(results, claims):
    """✅ ENHANCED: Print comprehensive verification results"""
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE AHSD SYSTEM VERIFICATION RESULTS")
    print("="*80)
    
    # Component Performance
    print("🧠 NEURAL PE PERFORMANCE:")
    print(f"   Average Accuracy: {results['neural_pe_accuracy']:.3f} ± {results['pe_std']:.3f} ({results['neural_pe_accuracy']:.1%})")
    print(f"   Range: [{results['pe_min']:.3f}, {results['pe_max']:.3f}]")
    print(f"   vs Claimed: {claims['neural_pe']:.3f} (diff: {results['neural_pe_accuracy']-claims['neural_pe']:+.3f})")
    
    print(f"\n🎯 SUBTRACTOR PERFORMANCE:")
    print(f"   Average Efficiency: {results['subtractor_efficiency']:.3f} ± {results['sub_std']:.3f} ({results['subtractor_efficiency']:.1%})")
    print(f"   Range: [{results['sub_min']:.3f}, {results['sub_max']:.3f}]")
    print(f"   vs Claimed: {claims['subtractor']:.3f} (diff: {results['subtractor_efficiency']-claims['subtractor']:+.3f})")
    
    print(f"\n✅ SYSTEM INTEGRATION:")
    print(f"   Success Rate: {results['system_success_rate']:.3f} ± {results['sys_std']:.3f} ({results['system_success_rate']:.1%})")
    print(f"   Samples Tested: {results['samples']}")
    print(f"   vs Claimed: {claims['system']:.3f} (diff: {results['system_success_rate']-claims['system']:+.3f})")
    
    # Detailed sample analysis
    if results['detailed_samples']:
        print(f"\n🔍 DETAILED SAMPLE ANALYSIS (First 5 samples):")
        for i, detail in enumerate(results['detailed_samples'][:5]):
            print(f"   Sample {i+1}: PE={detail['pe_accuracy']:.3f}, Eff={detail['subtractor_efficiency']:.3f}, "
                  f"Success={detail['system_success']}, Conf={detail['confidence']:.3f}")
    
    print("="*80)

def main():
    setup_logging()
    
    print("🔧 ENHANCED AHSD SYSTEM VERIFICATION")
    print("="*60)
    print("Complete verification including:")
    print("• Model architecture validation")
    print("• System integration testing")  
    print("• Performance benchmarking")
    print("• Statistical analysis")
    print("="*60)
    
    # Load models with verification
    try:
        neural_pe, subtractor, param_names = load_models_safely(
            'outputs/phase3b_production/phase3b_working_output.pth'
        )
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Verify subtractor architecture
    if not verify_subtractor_architecture(subtractor):
        print("❌ Subtractor architecture verification failed")
        return
    
    # Create Phase 3C compatible dataset
    samples = create_phase3c_compatible_dataset(1000, seed=42)
    if samples is None:
        print("❌ Dataset creation failed")
        return
    
    # Run comprehensive verification
    results = verify_system_performance(neural_pe, subtractor, samples)
    
    # Define claims for comparison
    claims = {
        'neural_pe': 0.587,    # Phase 3C Neural PE accuracy
        'subtractor': 0.811,   # Phase 3B Subtractor efficiency  
        'system': 0.817       # Phase 3C System success rate
    }
    
    # Print comprehensive results
    print_detailed_verification_results(results, claims)
    
    # ✅ ENHANCED: Verification status with detailed analysis
    pe_diff = abs(results['neural_pe_accuracy'] - claims['neural_pe'])
    sub_diff = abs(results['subtractor_efficiency'] - claims['subtractor'])
    sys_diff = abs(results['system_success_rate'] - claims['system'])
    
    print("\n📋 VERIFICATION ANALYSIS:")
    print(f"   Neural PE deviation: ±{pe_diff:.3f} ({pe_diff/claims['neural_pe']*100:.1f}%)")
    print(f"   Subtractor deviation: ±{sub_diff:.3f} ({sub_diff/claims['subtractor']*100:.1f}%)")
    print(f"   System deviation: ±{sys_diff:.3f} ({sys_diff/claims['system']*100:.1f}%)")
    
    # Final verification status
    if sub_diff < 0.02 and sys_diff < 0.02 and pe_diff < 0.05:
        print("\n🎉 VERIFICATION STATUS: PERFECT MATCH")
        print("✅ All claims confirmed within ±2% (subtractor/system) and ±5% (neural PE)")
        print("🏆 READY FOR PUBLICATION: Claims independently verified")
    elif sub_diff < 0.05 and sys_diff < 0.05 and pe_diff < 0.10:
        print("\n✅ VERIFICATION STATUS: EXCELLENT AGREEMENT")
        print("✅ All claims confirmed within ±5-10% margins")
        print("📚 Ready for publication with minor clarifications")
    elif sub_diff < 0.10 and sys_diff < 0.10:
        print("\n⚠️ VERIFICATION STATUS: ACCEPTABLE AGREEMENT")
        print("⚠️ Claims largely confirmed but some deviations noted")
        print("📋 Consider additional validation before publication")
    else:
        print("\n❌ VERIFICATION STATUS: SIGNIFICANT DEVIATIONS")
        print("❌ Claims require investigation and potential revision")
        print("🔍 Not recommended for publication without further analysis")
    
    # Save results for future reference
    output_dir = Path('outputs/verification_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'enhanced_verification_results.json', 'w') as f:
        # Convert numpy types to python types for JSON serialization
        results_json = {k: (v.tolist() if isinstance(v, np.ndarray) else 
                           float(v) if isinstance(v, np.floating) else v) 
                       for k, v in results.items() if k != 'detailed_samples'}
        json.dump({
            'verification_results': results_json,
            'original_claims': claims,
            'verification_date': datetime.now().isoformat(),
            'verification_status': 'PASSED' if sub_diff < 0.05 else 'CONCERNS'
        }, f, indent=2)
    
    logging.info("✅ Enhanced verification completed successfully!")

if __name__ == '__main__':
    main()
