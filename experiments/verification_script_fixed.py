#!/usr/bin/env python3
"""
FIXED AHSD VERIFICATION SCRIPT
Uses EXACT same data generation as Phase 3C validation
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

# ✅ FIXED: Add EffectiveSubtractor class definition
class EffectiveSubtractor(nn.Module):
    def __init__(self, data_length: int = 4096):
        super().__init__()
        self.data_length = data_length
        
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
        
        self.confidence_adapter = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, contaminated_data, neural_pe_output):
        batch_size = contaminated_data.shape[0]
        
        if isinstance(neural_pe_output, tuple):
            pred_params, pred_uncertainties = neural_pe_output
            confidence_input = pred_uncertainties if pred_uncertainties is not None else pred_params
        else:
            confidence_input = neural_pe_output
        
        contamination_pattern = self.contamination_detector(contaminated_data)
        contamination_pattern = contamination_pattern.view(batch_size, 2, self.data_length)
        confidence = self.confidence_adapter(confidence_input)
        strength = 0.3 + 0.5 * confidence
        cleaned_data = contaminated_data - (contamination_pattern * strength.unsqueeze(-1))
        
        return cleaned_data, confidence.squeeze(-1)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_phase3c_compatible_dataset(n_samples: int, seed: int = 42):
    """✅ FIXED: Use EXACT Phase 3C data generation"""
    
    np.random.seed(seed)
    samples = []
    
    logging.info(f"Creating {n_samples} samples with Phase 3C compatibility (seed={seed})")
    
    for i in range(n_samples):
        t = np.linspace(0, 4, 4096)
        
        # Generate parameters (same as Phase 3C)
        mass_1 = np.random.uniform(20, 50)
        mass_2 = np.random.uniform(15, mass_1)
        distance = np.random.uniform(200, 800)
        chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
        inclination = np.random.uniform(0, np.pi)
        
        # ✅ FIXED: EXACT Phase 3C scaling
        signal_scale = 1e-3
        contamination_scale = signal_scale * 10.0
        
        # Generate clean signal (same as Phase 3C)
        f_start = 20.0
        f_end = min(100.0, 220.0 / (mass_1 + mass_2))
        frequency = f_start + (f_end - f_start) * (t / 4.0)
        phase = 2 * np.pi * np.cumsum(frequency) * (4.0 / 4096)
        amplitude = signal_scale * np.exp(-t / 8.0) * np.sqrt(chirp_mass / 30.0)
        
        h_plus_clean = amplitude * (1 + np.cos(inclination)**2) * np.cos(phase)
        h_cross_clean = amplitude * 2 * np.cos(inclination) * np.sin(phase)
        
        # ✅ FIXED: EXACT Phase 3C contamination patterns
        power_contamination_h_plus = contamination_scale * np.sin(2 * np.pi * 60.0 * t)
        power_contamination_h_cross = contamination_scale * np.cos(2 * np.pi * 60.0 * t)
        
        seismic_contamination_h_plus = contamination_scale * 0.7 * np.sin(2 * np.pi * 2.0 * t)
        seismic_contamination_h_cross = contamination_scale * 0.7 * np.cos(2 * np.pi * 2.0 * t)
        
        hf_contamination_h_plus = contamination_scale * 0.3 * np.sin(2 * np.pi * 100.0 * t)
        hf_contamination_h_cross = contamination_scale * 0.3 * np.cos(2 * np.pi * 100.0 * t)
        
        # Glitch
        glitch_center = 2.0
        glitch_width = 0.2
        glitch_amplitude = contamination_scale * 2.0
        glitch = glitch_amplitude * np.exp(-((t - glitch_center) / glitch_width)**2)
        
        # Complete contamination (EXACT Phase 3C)
        total_contamination_h_plus = (power_contamination_h_plus + 
                                     seismic_contamination_h_plus + 
                                     hf_contamination_h_plus + glitch)
        total_contamination_h_cross = (power_contamination_h_cross + 
                                      seismic_contamination_h_cross + 
                                      hf_contamination_h_cross + glitch * 0.8)
        
        h_plus_contaminated = h_plus_clean + total_contamination_h_plus
        h_cross_contaminated = h_cross_clean + total_contamination_h_cross
        
        # Add noise (same as Phase 3C)
        noise_level = signal_scale * 0.01
        h_plus_contaminated += np.random.normal(0, noise_level, 4096)
        h_cross_contaminated += np.random.normal(0, noise_level, 4096)
        h_plus_clean += np.random.normal(0, noise_level, 4096)
        h_cross_clean += np.random.normal(0, noise_level, 4096)
        
        # Store data
        contaminated_data = np.array([h_plus_contaminated, h_cross_contaminated], dtype=np.float32)
        clean_data = np.array([h_plus_clean, h_cross_clean], dtype=np.float32)
        
        # Normalized parameters
        true_params = np.array([
            2 * (mass_1 - 15) / (50 - 15) - 1,
            2 * (mass_2 - 10) / (40 - 10) - 1,
            2 * (np.log10(distance) - np.log10(100)) / (np.log10(1000) - np.log10(100)) - 1,
            np.random.uniform(-0.8, 0.8),
            np.random.uniform(-0.8, 0.8),
            np.random.uniform(-0.8, 0.8),
            2 * (inclination / np.pi) - 1,
            np.random.uniform(-0.8, 0.8),
            np.random.uniform(-0.8, 0.8),
        ], dtype=np.float32)
        
        samples.append({
            'contaminated_data': contaminated_data,
            'clean_data': clean_data,
            'true_parameters': true_params,
        })
        
        # Verify contamination strength on first sample
        if i == 0:
            contamination_strength = np.mean(np.abs(contaminated_data - clean_data))
            logging.info(f"✅ First sample contamination strength: {contamination_strength:.6f}")
            if contamination_strength < 1e-6:
                logging.error("❌ Contamination still too weak!")
                return None
    
    return samples

def load_models_safely(phase3b_path: str):
    """✅ FIXED: Load models with state_dict approach"""
    
    logging.info("Loading models using state_dict approach...")
    
    # Load checkpoint
    checkpoint = torch.load(phase3b_path, map_location='cpu')
    param_names = checkpoint['param_names']
    
    # Create fresh model instances
    neural_pe = NeuralPENetwork(param_names)
    subtractor = EffectiveSubtractor()
    
    # Load state dicts if available
    if 'neural_pe_state_dict' in checkpoint:
        neural_pe.load_state_dict(checkpoint['neural_pe_state_dict'])
    elif hasattr(checkpoint['neural_pe_model'], 'state_dict'):
        neural_pe.load_state_dict(checkpoint['neural_pe_model'].state_dict())
    
    if 'subtractor_state_dict' in checkpoint:
        subtractor.load_state_dict(checkpoint['subtractor_state_dict'])
    elif hasattr(checkpoint['subtractor_model'], 'state_dict'):
        subtractor.load_state_dict(checkpoint['subtractor_model'].state_dict())
    
    neural_pe.eval()
    subtractor.eval()
    
    logging.info("✅ Models loaded successfully via state_dict")
    return neural_pe, subtractor, param_names

def verify_system_performance(neural_pe, subtractor, samples):
    """Verify system performance on Phase 3C compatible data"""
    
    pe_accuracies = []
    subtractor_efficiencies = []
    system_successes = []
    
    logging.info(f"Verifying system on {len(samples)} samples...")
    
    with torch.no_grad():
        for sample in tqdm(samples, desc="Verification"):
            contaminated = torch.tensor(sample['contaminated_data']).unsqueeze(0)
            clean_target = torch.tensor(sample['clean_data']).unsqueeze(0)
            true_params = torch.tensor(sample['true_parameters']).unsqueeze(0)
            
            # Neural PE prediction
            neural_pe_output = neural_pe(contaminated)
            if isinstance(neural_pe_output, tuple):
                pred_params, pred_uncertainties = neural_pe_output
                if pred_uncertainties is None:
                    pred_uncertainties = torch.abs(pred_params) + 0.1
            else:
                pred_params = neural_pe_output
                pred_uncertainties = torch.abs(pred_params) + 0.1
            
            # PE accuracy
            param_errors = torch.mean((pred_params - true_params) ** 2)
            pe_accuracy = float(1.0 / (1.0 + param_errors))
            pe_accuracies.append(max(0.0, min(1.0, pe_accuracy)))
            
            # Subtractor efficiency (same calculation as Phase 3C)
            cleaned_output, confidence = subtractor(contaminated, pred_uncertainties)
            
            mse_before = torch.mean((contaminated - clean_target) ** 2, dim=(1, 2))
            mse_after = torch.mean((cleaned_output - clean_target) ** 2, dim=(1, 2))
            improvement = mse_before - mse_after
            efficiency = improvement / (mse_before + 1e-8)
            efficiency = float(torch.clamp(efficiency, 0.0, 1.0))
            subtractor_efficiencies.append(efficiency)
            
            # System success (same thresholds as Phase 3C)
            pe_success = pe_accuracy > 0.5
            sub_success = efficiency > 0.3
            system_success = pe_success and sub_success
            system_successes.append(system_success)
    
    return {
        'neural_pe_accuracy': np.mean(pe_accuracies),
        'subtractor_efficiency': np.mean(subtractor_efficiencies),
        'system_success_rate': np.mean(system_successes),
        'pe_std': np.std(pe_accuracies),
        'sub_std': np.std(subtractor_efficiencies),
        'samples': len(samples)
    }

def main():
    setup_logging()
    
    print("🔧 FIXED AHSD VERIFICATION")
    print("="*50)
    
    # Load models
    try:
        neural_pe, subtractor, param_names = load_models_safely(
            'outputs/phase3b_production/phase3b_working_output.pth'
        )
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Create Phase 3C compatible dataset
    samples = create_phase3c_compatible_dataset(1000, seed=42)
    if samples is None:
        print("❌ Dataset creation failed")
        return
    
    # Run verification
    results = verify_system_performance(neural_pe, subtractor, samples)
    
    # Print results
    print("\n" + "="*60)
    print("📊 FIXED VERIFICATION RESULTS")
    print("="*60)
    print(f"Neural PE Accuracy: {results['neural_pe_accuracy']:.3f} ± {results['pe_std']:.3f}")
    print(f"Subtractor Efficiency: {results['subtractor_efficiency']:.3f} ± {results['sub_std']:.3f}")
    print(f"System Success Rate: {results['system_success_rate']:.3f}")
    print("="*60)
    
    # Compare with claims
    claims = {'neural_pe': 0.587, 'subtractor': 0.811, 'system': 0.817}
    
    print("COMPARISON WITH ORIGINAL CLAIMS:")
    print(f"Neural PE: {results['neural_pe_accuracy']:.3f} vs {claims['neural_pe']:.3f} (diff: {results['neural_pe_accuracy']-claims['neural_pe']:+.3f})")
    print(f"Subtractor: {results['subtractor_efficiency']:.3f} vs {claims['subtractor']:.3f} (diff: {results['subtractor_efficiency']-claims['subtractor']:+.3f})")
    print(f"System: {results['system_success_rate']:.3f} vs {claims['system']:.3f} (diff: {results['system_success_rate']-claims['system']:+.3f})")
    
    # Verification status
    sub_diff = abs(results['subtractor_efficiency'] - claims['subtractor'])
    sys_diff = abs(results['system_success_rate'] - claims['system'])
    
    if sub_diff < 0.05 and sys_diff < 0.05:
        print("\n🎉 VERIFICATION PASSED: Claims confirmed within ±5%")
    elif sub_diff < 0.1 and sys_diff < 0.1:
        print("\n✅ VERIFICATION ACCEPTABLE: Claims confirmed within ±10%")
    else:
        print("\n⚠️ VERIFICATION CONCERNS: Significant deviations detected")

if __name__ == '__main__':
    main()
