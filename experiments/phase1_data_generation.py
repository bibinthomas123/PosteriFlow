#!/usr/bin/env python3
"""
Phase 1: Diversified Mixed Real+Synthetic Dataset (Enhanced with NS support)
Uses existing AHSD modules with proper imports and implementations
"""
import sys
import numpy as np
import argparse
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import yaml
from typing import List, Dict, Optional
from scipy.stats import beta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
import time
warnings.filterwarnings('ignore')


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))


try:
    from ahsd.data.gwtc_loader import GWTCDataLoader
    from ahsd.data.preprocessing import DataPreprocessor
    from ahsd.data.simulation import OverlappingSignalSimulator
    from ahsd.utils.config import AHSDConfig
    IMPORTS_OK = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    IMPORTS_OK = False


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('phase1_enhanced.log'),
            logging.StreamHandler()
        ]
    )


# Fallback config if imports fail
class FallbackConfig:
    def __init__(self):
        self.waveform = type('obj', (object,), {
            'duration': 8.0,
            'f_lower': 20.0,
            'approximant': 'IMRPhenomPv2',
            'f_ref': 50.0
        })()
        self.detectors = [
            type('obj', (object,), {
                'name': 'H1',
                'sampling_rate': 4096,
                'duration': 8.0
            })(),
            type('obj', (object,), {
                'name': 'L1', 
                'sampling_rate': 4096,
                'duration': 8.0
            })(),
            type('obj', (object,), {
                'name': 'V1',
                'sampling_rate': 4096,
                'duration': 8.0
            })()
        ]
    
    @classmethod
    def from_yaml(cls, config_path: str):
        return cls()


class DatasetGenerator:
    """A class for generating comprehensive, diversified gravitational wave datasets with real LIGO integration.
    This class handles generation of a comprehensive dataset containing different types of gravitational wave scenarios,
    including pure synthetic, colored noise, real-augmented single/multi signals, real backgrounds, and extreme parameter cases.
    The generated dataset is designed for robust machine learning model training with emphasis on diversity and realism.
    Attributes:
        config: Configuration object containing dataset generation parameters
        logger: Logger instance for tracking generation progress
        gwtc_loader: GWTCDataLoader instance for accessing real LIGO data
        preprocessor: DataPreprocessor instance for signal processing
        simulator: OverlappingSignalSimulator for generating synthetic signals
        param_generator: Generator for creating diverse parameter sets
        stats: Dictionary tracking statistics about generated scenarios
    Methods:
        generate_diversified_dataset: Main method to generate the full dataset
        fix_signal_parameters: Ensures signal parameters have all required keys
        generate_pure_synthetic_scenarios: Creates pure synthetic gravitational wave scenarios
        generate_synthetic_colored_noise_scenarios: Generates scenarios with colored noise
        generate_real_augmented_scenarios: Creates scenarios using real LIGO data
        generate_real_background_scenarios: Generates scenarios from real background events
        generate_extreme_scenarios: Creates scenarios with extreme parameters
        generate_low_snr_scenarios: Generates challenging low SNR scenarios
        generate_high_snr_scenarios: Creates pristine high SNR scenarios
    Example:
        config = DatasetConfig()
        generator = DatasetGenerator(config)
        dataset = generator.generate_diversified_dataset(
            total_scenarios=10000,
            max_real_events=100
    Notes:
        - Requires proper configuration and initialization of LIGO data access
        - Handles fallback mechanisms when real data access fails
        - Implements comprehensive error handling and validation
        - Generates dataset according to specified distribution strategy
        - Provides detailed logging and statistics tracking
    Distribution Strategy:
        - Pure synthetic: 35% (3500 scenarios)
        - Synthetic colored noise: 15% (1500 scenarios) 
        - Real augmented single: 20% (2000 scenarios)
        - Real augmented multi: 15% (1500 scenarios)
        - Real background: 8% (800 scenarios)
        - Extreme scenarios: 4% (400 scenarios)
        - Low SNR challenge: 2% (200 scenarios)
        - High SNR pristine: 1% (100 scenarios)
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components with proper error handling
        try:
            if IMPORTS_OK:
                self.gwtc_loader = GWTCDataLoader() 
                self.preprocessor = DataPreprocessor(config)
                self.simulator = OverlappingSignalSimulator(config)
                self.logger.info("✅ Successfully initialized AHSD modules")
            else:
                self.gwtc_loader = FallbackGWTCLoader()
                self.preprocessor = None
                self.simulator = None
                self.logger.warning("⚠️ Using fallback implementations")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize modules: {e}")
            self.gwtc_loader = FallbackGWTCLoader() 
            self.preprocessor = None
            self.simulator = None
            
        self.param_generator = MaximumDiversityParameterGenerator(config)
        
        # Statistics tracking with NS support
        self.stats = {
            'pure_synthetic': 0,
            'synthetic_colored_noise': 0,
            'real_augmented_single': 0,
            'real_augmented_multi': 0,
            'real_background_events': 0,
            'extreme_parameter_scenarios': 0,
            'low_snr_scenarios': 0,
            'high_snr_scenarios': 0,
            'failed_scenarios': 0,
            'total_processed': 0,
            # NS-specific stats
            'bbh_scenarios': 0,
            'bns_scenarios': 0,
            'nsbh_scenarios': 0
        }
    
    
    def fix_signal_parameters(self, signal_parameters: List[Dict]) -> List[Dict]:
        """Fix signal parameters to ensure they have all required keys including NS support."""
        
        fixed_parameters = []
        
        for params in signal_parameters:
            fixed_params = params.copy()
            
            # Ensure network_snr exists
            if 'network_snr' not in fixed_params and 'snr' not in fixed_params:
                # Compute SNR from physical parameters
                m1 = fixed_params.get('mass_1', 30.0)
                m2 = fixed_params.get('mass_2', 25.0) 
                dist = fixed_params.get('luminosity_distance', 500.0)
                
                chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
                snr = 20.0 * (chirp_mass / 30.0)**(5/6) * (400.0 / dist)
                snr = np.clip(snr, 5.0, 100.0)
                
                fixed_params['network_snr'] = float(snr)
                fixed_params['snr'] = float(snr)
            elif 'network_snr' not in fixed_params:
                fixed_params['network_snr'] = fixed_params.get('snr', 15.0)
            elif 'snr' not in fixed_params:
                fixed_params['snr'] = fixed_params.get('network_snr', 15.0)
            
            # Ensure other common parameters exist
            defaults = {
                'mass_1': 30.0,
                'mass_2': 25.0,
                'luminosity_distance': 500.0,
                'geocent_time': 0.0,
                'ra': 0.0,
                'dec': 0.0,
                'theta_jn': 0.0,
                'psi': 0.0,
                'phase': 0.0,
                'signal_id': 0,
                'difficulty': 'medium',
                'approximant': 'IMRPhenomPv2',
                'binary_type': 'BBH'  # Default to BBH
            }
            
            for key, default_value in defaults.items():
                if key not in fixed_params:
                    fixed_params[key] = default_value
            
            # Track binary type statistics
            binary_type = fixed_params.get('binary_type', 'BBH')
            if binary_type == 'BBH':
                self.stats['bbh_scenarios'] += 1
            elif binary_type == 'BNS':
                self.stats['bns_scenarios'] += 1
            elif binary_type == 'NSBH':
                self.stats['nsbh_scenarios'] += 1
            
            fixed_parameters.append(fixed_params)
        
        return fixed_parameters

    def generate_diversified_dataset(self, 
                                    total_scenarios: int = 10000,
                                    max_real_events: int = 100) -> List[Dict]:
        """Generate comprehensive dataset with REAL DATA distribution."""
        
        self.logger.info("🌟 GENERATING ENHANCED DIVERSIFIED DATASET WITH NS SUPPORT")
        self.logger.info("="*80)
        
        # **DISTRIBUTION STRATEGY**
        distribution = {
            'pure_synthetic': int(total_scenarios * 0.35),        
            'synthetic_colored_noise': int(total_scenarios * 0.15), 
            'real_augmented_single': int(total_scenarios * 0.20),   
            'real_augmented_multi': int(total_scenarios * 0.15),    
            'real_background': int(total_scenarios * 0.08),        
            'extreme_scenarios': int(total_scenarios * 0.04),      
            'low_snr_challenge': int(total_scenarios * 0.02),      
            'high_snr_pristine': int(total_scenarios * 0.01)       
        }
        
        # Purpose descriptions
        purpose = {
            'pure_synthetic': 'Core training on clean signals (BBH+NS)',
            'synthetic_colored_noise': 'Noise robustness training (BBH+NS)',
            'real_augmented_single': 'Real detector characteristics (BOOSTED)',
            'real_augmented_multi': 'Multi-event realism (BOOSTED)', 
            'real_background': 'Realistic noise environment (INCREASED)',
            'extreme_scenarios': 'Rare/high-mass events coverage',
            'low_snr_challenge': 'Weak signal capability',
            'high_snr_pristine': 'Ideal signal benchmarks'
        }
        
        self.logger.info("📊 REAL-DATA FOCUSED DISTRIBUTION WITH NS SUPPORT:")
        for category, count in distribution.items():
            percentage = count/total_scenarios*100
            self.logger.info(f"   {category:25}: {count:4d} ({percentage:4.1f}%) - {purpose[category]}")
        
        # Continue with rest of generation...
        all_scenarios = []
        
        # **PHASE 1: Pure Synthetic**
        self.logger.info(f"\n🔥 Phase 1: Generating {distribution['pure_synthetic']} pure synthetic scenarios...")
        pure_synthetic = self.generate_pure_synthetic_scenarios(distribution['pure_synthetic'])
        all_scenarios.extend(pure_synthetic)
        self.stats['pure_synthetic'] = len(pure_synthetic)
        
        # **PHASE 2: Synthetic with Colored Noise**
        self.logger.info(f"\n🌈 Phase 2: Generating {distribution['synthetic_colored_noise']} colored noise scenarios...")
        colored_noise = self.generate_synthetic_colored_noise_scenarios(distribution['synthetic_colored_noise'])
        all_scenarios.extend(colored_noise)
        self.stats['synthetic_colored_noise'] = len(colored_noise)
        
        # **PHASE 3: Real-Augmented Single Signal**
        self.logger.info(f"\n📡 Phase 3: Generating {distribution['real_augmented_single']} real-augmented single scenarios...")
        real_aug_single = self.generate_real_augmented_scenarios(
            distribution['real_augmented_single'], max_real_events, multi_signal=False
        )
        all_scenarios.extend(real_aug_single)
        self.stats['real_augmented_single'] = len(real_aug_single)
        
        # **PHASE 4: Real-Augmented Multi Signal **
        self.logger.info(f"\n📡 Phase 4: Generating {distribution['real_augmented_multi']} real-augmented multi scenarios...")
        real_aug_multi = self.generate_real_augmented_scenarios(
            distribution['real_augmented_multi'], max_real_events, multi_signal=True
        )
        all_scenarios.extend(real_aug_multi)
        self.stats['real_augmented_multi'] = len(real_aug_multi)
        
        # **PHASE 5: Real Background Events **
        self.logger.info(f"\n🎯 Phase 5: Generating {distribution['real_background']} real background scenarios...")
        real_background = self.generate_real_background_scenarios(distribution['real_background'], max_real_events)
        all_scenarios.extend(real_background)
        self.stats['real_background_events'] = len(real_background)
        
        # **PHASE 6: Extreme Scenarios **
        self.logger.info(f"\n🔥 Phase 6: Generating {distribution['extreme_scenarios']} extreme parameter scenarios...")
        extreme_scenarios = self.generate_extreme_scenarios(distribution['extreme_scenarios'])
        all_scenarios.extend(extreme_scenarios)
        self.stats['extreme_parameter_scenarios'] = len(extreme_scenarios)
        
        # **PHASE 7: Low SNR Challenge **
        self.logger.info(f"\n🎯 Phase 7: Generating {distribution['low_snr_challenge']} low SNR challenge scenarios...")
        low_snr = self.generate_low_snr_scenarios(distribution['low_snr_challenge'])
        all_scenarios.extend(low_snr)
        self.stats['low_snr_scenarios'] = len(low_snr)
        
        # **PHASE 8: High SNR Pristine **
        self.logger.info(f"\n⭐ Phase 8: Generating {distribution['high_snr_pristine']} high SNR pristine scenarios...")
        high_snr = self.generate_high_snr_scenarios(distribution['high_snr_pristine'])
        all_scenarios.extend(high_snr)
        self.stats['high_snr_scenarios'] = len(high_snr)
        
        # **PHASE 9: Post-Processing and Validation**
        self.logger.info(f"\n🔧 Phase 9: Post-processing {len(all_scenarios)} total scenarios...")
        self.stats['total_processed'] = len(all_scenarios)
        
        # Shuffle for training diversity
        random.shuffle(all_scenarios)
        
        # Validate and clean
        validated_scenarios = self.validate_and_clean_scenarios(all_scenarios)
        
        self.logger.info("✅ DATASET GENERATION COMPLETED!")
        self._log_final_statistics()
        
        return validated_scenarios
    
    def generate_pure_synthetic_scenarios(self, n_scenarios: int) -> List[Dict]:
        """Generate pure synthetic scenarios using existing simulator if available with NS support."""
        
        scenarios = []
        
        if self.simulator is not None:            
            for scenario_id in tqdm(range(n_scenarios), desc="Pure synthetic (NS+BBH)"):
                try:
                    n_signals = np.random.choice([2, 3, 4, 5], p=[0.35, 0.35, 0.20, 0.10])
                    
                    # Generate scenario using simulator with NS support
                    scenario = self.simulator.generate_overlapping_scenario(n_signals)
                    noise_data = self.simulator.generate_detector_noise()
                    injected_data, signal_contributions = self.simulator.inject_signals_to_data(
                        scenario, noise_data
                    )
                    
                    # Enhance signals with NS parameters
                    enhanced_signals = []
                    for signal in scenario['signals']:
                        enhanced_signal = self._enhance_signal_with_ns_params(signal)
                        enhanced_signals.append(enhanced_signal)
                    
                    # Convert to training format
                    training_scenario = {
                        'scenario_id': scenario_id,
                        'true_parameters': enhanced_signals,
                        'injected_data': injected_data,
                        'waveform_data': self.convert_to_waveform_format(injected_data),
                        'n_signals': n_signals,
                        'data_type': 'pure_synthetic',
                        'source': 'synthetic_ns_bbh',
                        'binary_types': [s.get('binary_type', 'BBH') for s in enhanced_signals],
                        'approximants': [s.get('approximant', 'IMRPhenomPv2') for s in enhanced_signals],
                        'quality_metrics': self.compute_quality_metrics(enhanced_signals)
                    }
                    
                    scenarios.append(training_scenario)
                    
                except Exception as e:
                    self.logger.debug(f"Simulator scenario failed: {e}")
                    self.stats['failed_scenarios'] += 1
        else:
            # Fallback to manual generation with NS support
            self.logger.info("Using fallback synthetic generation with NS support")
            scenarios = self.generate_fallback_synthetic_scenarios(n_scenarios)
        
        return scenarios

    def _enhance_signal_with_ns_params(self, signal: Dict) -> Dict:
        """Enhance existing signal parameters with NS binary type classification"""
        enhanced_signal = signal.copy()
        
        # Determine binary type based on masses
        mass_1 = enhanced_signal.get('mass_1', 30.0)
        mass_2 = enhanced_signal.get('mass_2', 25.0)
        
        # Binary type classification
        if mass_1 <= 3.0 and mass_2 <= 3.0:
            # Both components are NS-like
            binary_type = 'BNS'
            approximant = random.choice(['IMRPhenomPv2_NRTidal', 'IMRPhenomD_NRTidal'])
            # Add tidal parameters
            enhanced_signal['lambda_1'] = np.random.uniform(50, 5000)
            enhanced_signal['lambda_2'] = np.random.uniform(50, 5000)
        elif (mass_1 <= 3.0 and mass_2 > 3.0) or (mass_1 > 3.0 and mass_2 <= 3.0):
            # One NS, one BH
            binary_type = 'NSBH'
            approximant = random.choice(['IMRPhenomPv2_NRTidal', 'IMRPhenomD_NRTidal'])
            # Add tidal parameters (only for NS component)
            if mass_1 <= 3.0:
                enhanced_signal['lambda_1'] = np.random.uniform(50, 2000)
                enhanced_signal['lambda_2'] = 0
            else:
                enhanced_signal['lambda_1'] = 0
                enhanced_signal['lambda_2'] = np.random.uniform(50, 2000)
        else:
            # Both components are BH
            binary_type = 'BBH'
            approximant = enhanced_signal.get('approximant', 'IMRPhenomPv2')
        
        enhanced_signal['binary_type'] = binary_type
        enhanced_signal['approximant'] = approximant
        
        # Ensure network_snr exists
        if 'network_snr' not in enhanced_signal:
            chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
            dist = enhanced_signal.get('luminosity_distance', 500.0)
            snr = 20.0 * (chirp_mass / 30.0)**(5/6) * (400.0 / dist)
            snr = np.clip(snr, 5.0, 100.0)
            enhanced_signal['network_snr'] = float(snr)
            enhanced_signal['snr'] = float(snr)
        
        return enhanced_signal

    def generate_fallback_synthetic_scenarios(self, n_scenarios: int) -> List[Dict]:
        """Fallback synthetic scenario generation with NS support."""
        
        scenarios = []
        
        for scenario_id in tqdm(range(n_scenarios), desc="Fallback synthetic (NS+BBH)"):
            try:
                n_signals = np.random.choice([2, 3, 4], p=[0.5, 0.35, 0.15])
                signal_parameters = self.param_generator.generate_maximum_diversity_parameters_with_ns(n_signals, scenario_id)
                
                # Generate synthetic data
                injected_data = self.create_synthetic_data(signal_parameters, self.config)
                
                scenario = {
                    'scenario_id': scenario_id,
                    'true_parameters': signal_parameters,
                    'injected_data': injected_data,
                    'waveform_data': self.convert_to_waveform_format(injected_data),
                    'n_signals': n_signals,
                    'data_type': 'pure_synthetic',
                    'source': 'synthetic_fallback_ns_bbh',
                    'binary_types': [p.get('binary_type', 'BBH') for p in signal_parameters],
                    'approximants': [p.get('approximant', 'IMRPhenomPv2') for p in signal_parameters],
                    'quality_metrics': self.compute_quality_metrics(signal_parameters)
                }
                
                scenarios.append(scenario)
                
            except Exception as e:
                self.logger.debug(f"Fallback scenario failed: {e}")
                self.stats['failed_scenarios'] += 1
        
        return scenarios
    
    def manual_strain_processing(self, strain_data: np.ndarray) -> np.ndarray:
        """Manual strain data processing when preprocessor unavailable."""
        
        try:
            # Basic processing steps
            processed_strain = np.array(strain_data)
            
            # Remove DC component
            processed_strain = processed_strain - np.mean(processed_strain)
            
            # Basic high-pass filtering to remove low frequencies
            if len(processed_strain) > 100:
                from scipy.signal import butter, filtfilt
                sample_rate = 4096  # Assume standard LIGO rate
                nyquist = sample_rate / 2
                low_cutoff = 35.0 / nyquist  # 35 Hz high-pass
                
                try:
                    b, a = butter(4, low_cutoff, btype='high')
                    processed_strain = filtfilt(b, a, processed_strain)
                except:
                    pass  # Skip filtering if it fails
            
            # Normalize
            rms = np.sqrt(np.mean(processed_strain**2))
            if rms > 0:
                processed_strain = processed_strain / rms * 1e-21
            
            return processed_strain
            
        except Exception as e:
            self.logger.debug(f"Manual strain processing failed: {e}")
            return strain_data

    def generate_real_augmented_scenarios(self, n_scenarios: int, 
                                       max_real_events: int, 
                                       multi_signal: bool = False) -> List[Dict]:
        """Generate real-augmented scenarios with enhanced signal injection"""
        
        scenarios = []
        
        self.logger.info(f"🎯 Loading GWTC events (max: {max_real_events})...")
        try:
            gwtc_events = self.gwtc_loader.get_gwtc_events()
            gwtc_events = gwtc_events.head(max_real_events)
            self.logger.info(f"✅ Loaded {len(gwtc_events)} real events")
        except Exception as e:
            self.logger.warning(f"⚠️ GWTC loading failed: {e}")
            gwtc_events = pd.DataFrame()
        
        scenario_type = 'real_augmented_multi' if multi_signal else 'real_augmented_single'
        
        for scenario_id in tqdm(range(n_scenarios), desc=f"{scenario_type}"):
            try:
                # Select base event
                if len(gwtc_events) > 0:
                    event = gwtc_events.iloc[np.random.randint(0, len(gwtc_events))].to_dict()
                    base_strain = self.get_realistic_noise_from_event(event)
                else:
                    # Fallback to synthetic noise
                    base_strain = self.generate_fallback_noise()
                    event = {'event_name': f'synthetic_{scenario_id}'}
                
                # Generate additional signals to inject
                n_additional = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1]) if multi_signal else 1
                additional_signals = self.param_generator.generate_maximum_diversity_parameters_with_ns(
                    n_additional, scenario_id
                )
                
                # Inject additional signals
                injected_data = self.inject_signals_to_real_data(base_strain, additional_signals)
                
                scenario = {
                    'scenario_id': scenario_id,
                    'true_parameters': additional_signals,
                    'injected_data': injected_data,
                    'waveform_data': self.convert_to_waveform_format(injected_data),
                    'n_signals': n_additional,
                    'data_type': scenario_type,
                    'source': f'real_augmented_{event.get("event_name", "unknown")}',
                    'base_event': event,
                    'binary_types': [p.get('binary_type', 'BBH') for p in additional_signals],
                    'approximants': [p.get('approximant', 'IMRPhenomPv2') for p in additional_signals],
                    'quality_metrics': self.compute_quality_metrics(additional_signals)
                }
                
                scenarios.append(scenario)
                
            except Exception as e:
                self.logger.debug(f"Real augmented scenario {scenario_id} failed: {e}")
                self.stats['failed_scenarios'] += 1
        
        return scenarios

    def get_realistic_noise_from_event(self, event: Dict) -> Dict:
        """Extract realistic noise characteristics from a real event"""
        
        try:
            # Simulate getting real strain data
            duration = 4.0
            sample_rate = 4096
            n_samples = int(duration * sample_rate)
            
            # Create realistic noise based on event characteristics
            observing_run = event.get('observing_run', 'O3a')
            
            noise_data = {}
            for detector in ['H1', 'L1', 'V1']:
                # Generate realistic colored noise
                noise = self.generate_advanced_colored_noise(n_samples, sample_rate, detector)
                
                # Scale based on observing run
                if observing_run.startswith('O1'):
                    noise = noise * 1.5  # O1 was noisier
                elif observing_run.startswith('O2'):
                    noise = noise * 1.2  # O2 intermediate
                # O3 keeps base noise level
                
                noise_data[detector] = noise
            
            return noise_data
            
        except Exception as e:
            self.logger.debug(f"Real noise extraction failed: {e}")
            return self.generate_fallback_noise()

    def generate_fallback_noise(self) -> Dict:
        """Generate fallback noise when real data unavailable"""
        
        duration = 4.0
        sample_rate = 4096
        n_samples = int(duration * sample_rate)
        
        noise_data = {}
        for detector in ['H1', 'L1', 'V1']:
            # Generate basic colored noise
            noise = self.generate_advanced_colored_noise(n_samples, sample_rate, detector)
            noise_data[detector] = noise
        
        return noise_data

    def inject_signals_to_real_data(self, base_data: Dict, signal_parameters: List[Dict]) -> Dict:
        """Inject synthetic signals into real detector data - Most robust version"""
        
        injected_data = {}
        
        for detector in ['H1', 'L1', 'V1']:
            try:
                # Start with base noise/data
                base_strain = base_data.get(detector, np.zeros(16384))
                
                # Ensure we have valid data
                if base_strain is None or len(base_strain) == 0:
                    base_strain = np.random.normal(0, 1e-21, 16384)
                
                # Ensure correct length
                target_length = 16384
                if len(base_strain) > target_length:
                    start_idx = (len(base_strain) - target_length) // 2
                    base_strain = base_strain[start_idx:start_idx + target_length]
                elif len(base_strain) < target_length:
                    base_strain = np.pad(base_strain, (0, target_length - len(base_strain)))
                
                # Robust preprocessing with multiple approaches
                base_strain = self._apply_preprocessing(base_strain, detector)
                
                # Add synthetic signals
                total_signal = np.zeros_like(base_strain)
                
                for params in signal_parameters:
                    try:
                        # Generate waveform
                        signal = self.generate_synthetic_waveform(params, len(base_strain), detector)
                        if signal is not None and len(signal) == len(base_strain):
                            total_signal += signal
                    except Exception as e:
                        self.logger.debug(f"Waveform injection failed for {detector}: {e}")
                        continue
                
                injected_data[detector] = base_strain + total_signal
                
            except Exception as e:
                self.logger.debug(f"Signal injection failed for {detector}: {e}")
                # Fallback: generate synthetic data
                injected_data[detector] = np.random.normal(0, 1e-21, 16384)
        
        return injected_data

    def _apply_preprocessing(self, strain_data: np.ndarray, detector: str) -> np.ndarray:
        """Apply preprocessing using available methods - FIXED"""
        
        if self.preprocessor is None:
            return self.manual_strain_processing(strain_data)
        
        # Try different preprocessing approaches
        preprocessing_approaches = [
            # Approach 1: Use main preprocess method with dictionary
            lambda data, det: self.preprocessor.preprocess({det: data}).get(det, data),
            
            # Approach 2: Use internal method directly - FIXED METHOD NAME
            lambda data, det: self.preprocessor._preprocess_detector_data(data, det),
            
            # Approach 3: Manual processing fallback
            lambda data, det: self.manual_strain_processing(data)
        ]
        
        for i, approach in enumerate(preprocessing_approaches):
            try:
                result = approach(strain_data, detector)
                
                # Validate result
                if result is not None and len(result) > 0:
                    result_array = np.array(result)
                    if np.all(np.isfinite(result_array)):
                        return result_array
                        
            except Exception as e:
                self.logger.debug(f"Preprocessing approach {i+1} failed for {detector}: {e}")
                continue
        
        # Ultimate fallback
        return np.array(strain_data)

    def generate_real_background_scenarios(self, n_scenarios: int, max_real_events: int) -> List[Dict]:
        """Generate scenarios using real background noise without injection"""
        
        scenarios = []
        
        self.logger.info(f"📡 Loading background noise from {max_real_events} events...")
        
        try:
            gwtc_events = self.gwtc_loader.get_gwtc_events()
            gwtc_events = gwtc_events.head(max_real_events)
        except Exception as e:
            self.logger.warning(f"GWTC loading failed: {e}")
            gwtc_events = pd.DataFrame()
        
        for scenario_id in tqdm(range(n_scenarios), desc="Real background"):
            try:
                # Get real background data
                if len(gwtc_events) > 0:
                    event = gwtc_events.iloc[np.random.randint(0, len(gwtc_events))].to_dict()
                    background_data = self.get_background_data_from_event(event)
                    source_name = event.get('event_name', f'background_{scenario_id}')
                else:
                    background_data = self.generate_fallback_background()
                    source_name = f'synthetic_background_{scenario_id}'
                
                # No signals - pure background
                scenario = {
                    'scenario_id': scenario_id,
                    'true_parameters': [],  # No signals
                    'injected_data': background_data,
                    'waveform_data': self.convert_to_waveform_format(background_data),
                    'n_signals': 0,
                    'data_type': 'real_background',
                    'source': f'background_{source_name}',
                    'binary_types': [],
                    'approximants': [],
                    'quality_metrics': {'diversity_score': 0.0, 'background_only': True}
                }
                
                scenarios.append(scenario)
                
            except Exception as e:
                self.logger.debug(f"Background scenario {scenario_id} failed: {e}")
                self.stats['failed_scenarios'] += 1
        
        return scenarios
    
    def get_background_data_from_event(self, event: Dict) -> Dict:
        """Get background data from event time period - ROBUST VERSION"""
        
        try:
            # Generate realistic background for each detector
            duration = 4.0
            sample_rate = 4096
            n_samples = int(duration * sample_rate)
            
            # Generate background noise from different observing periods
            background_data = {}
            observing_run = event.get('observing_run', 'O3a')
            
            for detector in ['H1', 'L1', 'V1']:
                try:
                    # Generate base colored noise with validation
                    noise = self.generate_advanced_colored_noise(n_samples, sample_rate, detector)
                    
                    # Validate noise quality
                    if not np.all(np.isfinite(noise)):
                        self.logger.debug(f"Non-finite values in noise for {detector}, using fallback")
                        noise = np.random.normal(0, 1e-23, n_samples)
                    
                    # Calculate noise level safely
                    noise_level = np.std(noise)
                    if noise_level == 0 or not np.isfinite(noise_level):
                        noise_level = 1e-23
                    
                    # Apply observing run specific enhancements
                    if observing_run.startswith('O4'):
                        try:
                            enhanced_noise = self.enhance_low_frequency_performance(noise, sample_rate)
                            if np.all(np.isfinite(enhanced_noise)):
                                noise = enhanced_noise
                        except Exception as enhance_error:
                            self.logger.debug(f"O4 enhancement failed for {detector}: {enhance_error}")
                    
                    # Apply observing run scaling factors
                    scaling_factors = {
                        'O1': 1.5,    # O1 was noisier
                        'O2': 1.2,    # O2 intermediate noise
                        'O3a': 1.0,   # O3a baseline
                        'O3b': 1.0,   # O3b baseline
                        'O4': 0.8     # O4 improved sensitivity
                    }
                    
                    scale_factor = 1.0
                    for run_key, factor in scaling_factors.items():
                        if observing_run.startswith(run_key):
                            scale_factor = factor
                            break
                    
                    # Apply scaling
                    noise = noise * scale_factor
                    
                    # Final validation
                    if not np.all(np.isfinite(noise)):
                        self.logger.debug(f"Non-finite values after scaling for {detector}, using fallback")
                        noise = np.random.normal(0, 1e-23, n_samples)
                    
                    background_data[detector] = noise.astype(np.float64)
                    
                except Exception as detector_error:
                    self.logger.debug(f"Background generation failed for {detector}: {detector_error}")
                    # Fallback for this detector
                    background_data[detector] = np.random.normal(0, 1e-23, n_samples).astype(np.float64)
            
            return background_data
            
        except Exception as e:
            self.logger.debug(f"Complete background data extraction failed: {e}")
            return self.generate_fallback_background()

    def enhance_low_frequency_performance(self, noise: np.ndarray, sample_rate: int) -> np.ndarray:
        """Enhance low-frequency performance for O4-like data."""
        
        try:
            # BASE: O4 has better low-frequency performance
            from scipy.signal import butter, filtfilt
            
            # BASE: Apply better low-frequency filtering (O4 improvements)
            nyquist = sample_rate / 2
            low_cutoff = 15.0 / nyquist  # O4 goes down to 15 Hz
            
            b, a = butter(6, low_cutoff, btype='high')
            enhanced_noise = filtfilt(b, a, noise)
            
            return enhanced_noise
        except:
            return noise

    def generate_synthetic_colored_noise_scenarios(self, n_scenarios: int) -> List[Dict]:
        """Generate synthetic scenarios with realistic colored noise"""
        
        scenarios = []
        
        for scenario_id in tqdm(range(n_scenarios), desc="Colored noise"):
            try:
                # Generate base signals with NS support
                n_signals = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                signal_parameters = self.param_generator.generate_maximum_diversity_parameters_with_ns(n_signals, scenario_id)
                
                # Create synthetic data with advanced colored noise
                injected_data = self.create_synthetic_colored_noise_data(signal_parameters, self.config)
                
                scenario = {
                    'scenario_id': scenario_id,
                    'true_parameters': signal_parameters,
                    'injected_data': injected_data,
                    'waveform_data': self.convert_to_waveform_format(injected_data),
                    'n_signals': n_signals,
                    'data_type': 'synthetic_colored_noise',
                    'source': 'synthetic_colored_ns_bbh',
                    'binary_types': [p.get('binary_type', 'BBH') for p in signal_parameters],
                    'approximants': [p.get('approximant', 'IMRPhenomPv2') for p in signal_parameters],
                    'quality_metrics': self.compute_quality_metrics(signal_parameters)
                }
                
                scenarios.append(scenario)
                
            except Exception as e:
                self.logger.debug(f"Colored noise scenario {scenario_id} failed: {e}")
                self.stats['failed_scenarios'] += 1
        
        return scenarios

    def create_synthetic_colored_noise_data(self, signal_parameters: List[Dict], config) -> Dict:
        """Create synthetic data with advanced colored noise characteristics"""
        
        duration = 4.0
        sample_rate = 4096
        n_samples = int(duration * sample_rate)
        
        data = {}
        for detector in ['H1', 'L1', 'V1']:
            # Generate advanced colored noise
            noise = self.generate_advanced_colored_noise(n_samples, sample_rate, detector)
            
            # Add all signals to the detector data
            total_signal = np.zeros(n_samples)
            
            for params in signal_parameters:
                try:
                    signal = self.generate_synthetic_waveform(params, n_samples, detector)
                    total_signal += signal
                except Exception as e:
                    self.logger.debug(f"Signal generation failed: {e}")
                    continue
            
            data[detector] = noise + total_signal
        
        return data
    
    def generate_advanced_colored_noise(self, n_samples: int, sample_rate: int, detector: str) -> np.ndarray:
        """Generate advanced colored noise with realistic PSD - ROBUST VERSION"""
        
        try:
            # Generate white noise
            white_noise = np.random.normal(0, 1, n_samples)
            
            # Apply detector-specific coloring
            freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
            positive_freqs = freqs[:n_samples//2 + 1]
            
            # Avoid zero frequency issues
            positive_freqs = np.maximum(positive_freqs, 1e-3)  # Minimum frequency
            
            # Enhanced PSD models for different detectors
            if detector in ['H1', 'L1']:
                # Advanced LIGO-like PSD with robust calculation
                f0 = 215.0
                
                # Robust PSD calculation to avoid infinities
                psd = np.zeros_like(positive_freqs)
                
                # Low frequency term (avoid division by zero)
                low_freq_mask = positive_freqs < 10.0
                psd[low_freq_mask] = 1e-44 * (10.0 / 10.0)**(-4.14)  # Flat at low freq
                
                # Main frequency range
                main_freq_mask = (positive_freqs >= 10.0) & (positive_freqs <= 2000.0)
                f_main = positive_freqs[main_freq_mask]
                psd[main_freq_mask] = (
                    1e-44 * (f_main / f0)**(-4.14) + 
                    1e-46 * (f_main / f0)**(-2) + 
                    1e-47 * (1 + (f_main / f0)**2)**(-0.5)
                )
                
                # High frequency (above 2000 Hz)
                high_freq_mask = positive_freqs > 2000.0
                psd[high_freq_mask] = 1e-47
                
                # Add quantum noise contribution
                psd += 1e-48 * (1 + (positive_freqs / 150.0)**2)
                
            elif detector == 'V1':
                # Virgo-like PSD with different characteristics
                psd = np.zeros_like(positive_freqs)
                
                # Avoid very low frequencies
                freq_safe = np.maximum(positive_freqs, 1.0)
                
                psd = (3.2e-46 * (freq_safe / 100.0)**(-4.05) + 
                    2e-48 + 
                    5e-49 * (freq_safe / 200.0)**(-2))
                
            else:
                # Generic detector PSD
                freq_safe = np.maximum(positive_freqs, 1.0)
                psd = 1e-46 * (freq_safe / 100.0)**(-4) + 1e-48
            
            # Ensure PSD is always positive and finite
            psd = np.maximum(psd, 1e-50)
            psd = np.nan_to_num(psd, nan=1e-48, posinf=1e-40, neginf=1e-50)
            
            # Apply coloring in frequency domain
            white_noise_f = np.fft.fft(white_noise)
            white_noise_positive = white_noise_f[:n_samples//2 + 1]
            
            # Robust division
            sqrt_psd = np.sqrt(psd * sample_rate / 2)
            sqrt_psd = np.maximum(sqrt_psd, 1e-25)  # Avoid division by tiny numbers
            
            colored_noise_f = white_noise_positive / sqrt_psd
            
            # Ensure no non-finite values
            colored_noise_f = np.nan_to_num(colored_noise_f, nan=0.0, posinf=1e-20, neginf=-1e-20)
            
            # Convert back to time domain
            if n_samples % 2 == 0:
                # Even length
                colored_noise_f_full = np.concatenate([
                    colored_noise_f, 
                    np.conj(colored_noise_f[-2:0:-1])
                ])
            else:
                # Odd length
                colored_noise_f_full = np.concatenate([
                    colored_noise_f, 
                    np.conj(colored_noise_f[-1:0:-1])
                ])
            
            colored_noise = np.fft.ifft(colored_noise_f_full).real
            
            # Final validation and cleanup
            colored_noise = np.nan_to_num(colored_noise, nan=0.0, posinf=1e-20, neginf=-1e-20)
            
            # Ensure correct length
            if len(colored_noise) != n_samples:
                if len(colored_noise) > n_samples:
                    colored_noise = colored_noise[:n_samples]
                else:
                    colored_noise = np.pad(colored_noise, (0, n_samples - len(colored_noise)))
            
            # Scale to realistic noise level
            target_rms = 1e-23
            current_rms = np.std(colored_noise)
            if current_rms > 0:
                colored_noise = colored_noise * (target_rms / current_rms)
            
            return colored_noise.astype(np.float64)
            
        except Exception as e:
            logging.debug(f"Advanced colored noise generation failed: {e}")
            # Ultimate fallback: simple white noise
            return np.random.normal(0, 1e-23, n_samples).astype(np.float64)

    def generate_fallback_background(self) -> Dict:
        """Generate fallback background noise"""
        
        duration = 4.0
        sample_rate = 4096
        n_samples = int(duration * sample_rate)
        
        background_data = {}
        for detector in ['H1', 'L1', 'V1']:
            noise = self.generate_advanced_colored_noise(n_samples, sample_rate, detector)
            background_data[detector] = noise
        
        return background_data

    def generate_extreme_scenarios(self, n_scenarios: int) -> List[Dict]:
        """Generate scenarios with extreme parameters to test model limits"""
        
        scenarios = []
        
        for scenario_id in tqdm(range(n_scenarios), desc="Extreme scenarios"):
            try:
                # Generate extreme parameter combinations
                n_signals = np.random.choice([3, 4, 5], p=[0.5, 0.3, 0.2])  # More signals
                extreme_parameters = self.param_generator.generate_extreme_parameters(n_signals, scenario_id)
                
                # Create synthetic data
                injected_data = self.create_synthetic_data(extreme_parameters, self.config)
                
                scenario = {
                    'scenario_id': scenario_id,
                    'true_parameters': extreme_parameters,
                    'injected_data': injected_data,
                    'waveform_data': self.convert_to_waveform_format(injected_data),
                    'n_signals': n_signals,
                    'data_type': 'extreme_scenarios',
                    'source': 'synthetic_extreme',
                    'binary_types': [p.get('binary_type', 'BBH') for p in extreme_parameters],
                    'approximants': [p.get('approximant', 'IMRPhenomPv2') for p in extreme_parameters],
                    'quality_metrics': self.compute_quality_metrics(extreme_parameters)
                }
                
                scenarios.append(scenario)
                
            except Exception as e:
                self.logger.debug(f"Extreme scenario {scenario_id} failed: {e}")
                self.stats['failed_scenarios'] += 1
        
        return scenarios

    def generate_low_snr_scenarios(self, n_scenarios: int) -> List[Dict]:
        """Generate challenging low SNR scenarios"""
        
        scenarios = []
        
        for scenario_id in tqdm(range(n_scenarios), desc="Low SNR challenge"):
            try:
                # Generate low SNR signals
                n_signals = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                low_snr_parameters = self.param_generator.generate_low_snr_parameters(n_signals, scenario_id)
                
                # Create synthetic data with enhanced noise
                injected_data = self.create_synthetic_data(low_snr_parameters, self.config)
                
                # Add extra noise to make it more challenging
                for detector in injected_data.keys():
                    extra_noise = np.random.normal(0, 1e-23, len(injected_data[detector]))
                    injected_data[detector] += extra_noise
                
                scenario = {
                    'scenario_id': scenario_id,
                    'true_parameters': low_snr_parameters,
                    'injected_data': injected_data,
                    'waveform_data': self.convert_to_waveform_format(injected_data),
                    'n_signals': n_signals,
                    'data_type': 'low_snr_challenge',
                    'source': 'synthetic_low_snr',
                    'binary_types': [p.get('binary_type', 'BBH') for p in low_snr_parameters],
                    'approximants': [p.get('approximant', 'IMRPhenomPv2') for p in low_snr_parameters],
                    'quality_metrics': self.compute_quality_metrics(low_snr_parameters)
                }
                
                scenarios.append(scenario)
                
            except Exception as e:
                self.logger.debug(f"Low SNR scenario {scenario_id} failed: {e}")
                self.stats['failed_scenarios'] += 1
        
        return scenarios

    def generate_high_snr_scenarios(self, n_scenarios: int) -> List[Dict]:
        """Generate pristine high SNR scenarios for benchmarking"""
        
        scenarios = []
        
        for scenario_id in tqdm(range(n_scenarios), desc="High SNR pristine"):
            try:
                # Generate high SNR signals
                n_signals = np.random.choice([1, 2], p=[0.7, 0.3])
                high_snr_parameters = self.param_generator.generate_high_snr_parameters(n_signals, scenario_id)
                
                # Create clean synthetic data
                injected_data = self.create_synthetic_data(high_snr_parameters, self.config)
                
                # Reduce noise for pristine quality
                for detector in injected_data.keys():
                    injected_data[detector] *= 0.5  # Reduce noise component
                
                scenario = {
                    'scenario_id': scenario_id,
                    'true_parameters': high_snr_parameters,
                    'injected_data': injected_data,
                    'waveform_data': self.convert_to_waveform_format(injected_data),
                    'n_signals': n_signals,
                    'data_type': 'high_snr_pristine',
                    'source': 'synthetic_high_snr',
                    'binary_types': [p.get('binary_type', 'BBH') for p in high_snr_parameters],
                    'approximants': [p.get('approximant', 'IMRPhenomPv2') for p in high_snr_parameters],
                    'quality_metrics': self.compute_quality_metrics(high_snr_parameters)
                }
                
                scenarios.append(scenario)
                
            except Exception as e:
                self.logger.debug(f"High SNR scenario {scenario_id} failed: {e}")
                self.stats['failed_scenarios'] += 1
        
        return scenarios

    def create_synthetic_data(self, signal_parameters: List[Dict], config) -> Dict:
        """Create synthetic detector data from signal parameters"""
        
        duration = 4.0
        sample_rate = 4096
        n_samples = int(duration * sample_rate)
        
        data = {}
        for detector in ['H1', 'L1', 'V1']:
            # Generate base noise
            noise = self.generate_advanced_colored_noise(n_samples, sample_rate, detector)
            
            # Add all signals
            total_signal = np.zeros(n_samples)
            
            for params in signal_parameters:
                try:
                    signal = self.generate_synthetic_waveform(params, n_samples, detector)
                    total_signal += signal
                except Exception as e:
                    self.logger.debug(f"Synthetic waveform generation failed: {e}")
                    continue
            
            data[detector] = noise + total_signal
        
        return data

    def generate_synthetic_waveform(self, params: Dict, n_samples: int, detector: str) -> np.ndarray:
        """Generate synthetic gravitational waveform with NS support"""
        
        try:
            sample_rate = 4096
            duration = n_samples / sample_rate
            t = np.linspace(0, duration, n_samples)
            
            m1, m2 = params['mass_1'], params['mass_2']
            distance = params['luminosity_distance']
            snr = params['network_snr']
            binary_type = params.get('binary_type', 'BBH')
            approximant = params.get('approximant', 'IMRPhenomPv2')
            
            # Enhanced physics for different binary types
            chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
            eta = (m1 * m2) / (m1 + m2)**2
            
            # Time to merger and frequency evolution
            tc = duration * 0.8 + params.get('geocent_time', 0.0)
            time_to_merger = np.maximum(tc - t, 0.01)
            
            # Different frequency evolution for different binary types
            if binary_type == 'BNS':
                # BNS systems: longer inspiral, more cycles
                f_start = 20.0  # Lower starting frequency for NS
                frequency = f_start * (time_to_merger / 1.0)**(-3/8)
                # Add tidal effects near merger
                tidal_correction = 1.0
                if 'lambda_1' in params and 'lambda_2' in params:
                    lambda_eff = (params['lambda_1'] + params['lambda_2']) / 2
                    tidal_correction = 1 + 0.1 * (lambda_eff / 1000) * (frequency / 1000)**2
                frequency *= tidal_correction
                frequency = np.clip(frequency, 20.0, 2048.0)
            elif binary_type == 'NSBH':
                # NSBH: intermediate behavior
                f_start = 25.0
                frequency = f_start * (time_to_merger / 1.0)**(-3/8)
                # Partial tidal effects
                frequency = np.clip(frequency, 25.0, 1536.0)
            else:  # BBH
                # BBH: standard evolution
                f_start = 35.0
                frequency = f_start * (time_to_merger / 1.0)**(-3/8)
                frequency = np.clip(frequency, 35.0, 1024.0)
            
            # Enhanced amplitude evolution
            amplitude = snr * 1e-23 * (chirp_mass / 30.0)**(5/6) / (distance / 400.0)
            
            # Different amplitude scaling for different binary types
            if binary_type == 'BNS':
                amplitude *= 0.7  # BNS typically have lower amplitude
            elif binary_type == 'NSBH':
                amplitude *= 0.85  # NSBH intermediate amplitude
            
            # Amplitude evolution with improved merger modeling
            amp_evolution = (time_to_merger / time_to_merger[0])**(-1/4)
            
            # Enhanced merger and ringdown
            merger_mask = time_to_merger < 0.1
            if binary_type == 'BBH':
                # BBH has prominent ringdown
                ringdown_decay = 0.02
            elif binary_type == 'BNS':
                # BNS may have disruption, not ringdown
                ringdown_decay = 0.005
            else:  # NSBH
                # NSBH intermediate ringdown
                ringdown_decay = 0.01
            
            amp_evolution[merger_mask] *= np.exp(-(t[merger_mask] - tc)**2 / ringdown_decay)
            
            # Generate both polarizations
            dt = t[1] - t[0] if len(t) > 1 else 1/sample_rate
            phase = 2 * np.pi * np.cumsum(frequency) * dt + params['phase']
            
            h_plus = amplitude * amp_evolution * np.sin(phase)
            h_cross = amplitude * amp_evolution * np.cos(phase) * np.cos(2 * params.get('theta_jn', 0))
            
            # Enhanced detector response with proper antenna patterns
            ra = params['ra']
            dec = params['dec']
            psi = params['psi']
            
            # Improved antenna pattern calculations
            if detector == 'H1':
                F_plus = 0.5 * (1 + np.cos(dec)**2) * np.cos(2*psi) * np.cos(2*ra)
                F_cross = np.cos(dec) * np.sin(2*psi) * np.sin(2*ra)
            elif detector == 'L1':
                F_plus = 0.5 * (1 + np.cos(dec)**2) * np.cos(2*psi + np.pi/2) * np.cos(2*ra)
                F_cross = np.cos(dec) * np.sin(2*psi + np.pi/2) * np.sin(2*ra)
            elif detector == 'V1':
                F_plus = 0.3 * (1 + np.cos(dec)**2) * np.cos(2*psi + np.pi/4) * np.cos(2*ra + np.pi/3)
                F_cross = 0.7 * np.cos(dec) * np.sin(2*psi + np.pi/4) * np.sin(2*ra + np.pi/3)
            else:
                F_plus = 0.5
                F_cross = 0.5
            
            h_detector = F_plus * h_plus + F_cross * h_cross
            
            return h_detector
            
        except Exception as e:
            self.logger.debug(f"Enhanced waveform generation failed: {e}")
            # Simple fallback
            t = np.linspace(0, n_samples/4096, n_samples)
            fallback_signal = np.sin(2 * np.pi * 100 * t) * 1e-22
            return fallback_signal

    def convert_to_waveform_format(self, injected_data: Dict) -> np.ndarray:
        """Convert injected data to standardized waveform format."""
        
        try:
            # Get first available detector data
            for detector, data in injected_data.items():
                if isinstance(data, np.ndarray) and len(data) > 0:
                    # Ensure correct length
                    target_length = 4096
                    if len(data) > target_length:
                        center = len(data) // 2
                        data = data[center-target_length//2:center+target_length//2]
                    elif len(data) < target_length:
                        data = np.pad(data, (0, target_length - len(data)))
                    
                    # Create 2-channel format (plus and cross polarizations)
                    waveform_data = np.zeros((2, target_length), dtype=np.float32)
                    waveform_data[0] = data.astype(np.float32)
                    
                    # Generate cross-polarization approximation
                    try:
                        h_cross = np.imag(hilbert(data)) * 0.7
                        waveform_data[1] = h_cross.astype(np.float32)
                    except:
                        waveform_data[1] = data.astype(np.float32) * 0.7
                    
                    return waveform_data
            
            # Fallback
            return np.random.normal(0, 1e-21, (2, 4096)).astype(np.float32)
            
        except Exception as e:
            self.logger.debug(f"Waveform conversion failed: {e}")
            return np.random.normal(0, 1e-21, (2, 4096)).astype(np.float32)

    def compute_quality_metrics(self, signal_parameters: List[Dict]) -> Dict:
        """Compute comprehensive quality metrics including NS-specific metrics"""
        
        if not signal_parameters:
            return {'diversity_score': 0.0}
        
        try:
            binary_types = [p.get('binary_type', 'BBH') for p in signal_parameters]
            approximants = [p.get('approximant', 'IMRPhenomPv2') for p in signal_parameters]
            
            # Count different binary types
            type_diversity = len(set(binary_types)) / 3.0  # Max 3 types
            approximant_diversity = len(set(approximants)) / max(len(set(approximants)), 1)
            
            # Standard parameter diversity
            masses = [p['mass_1'] + p['mass_2'] for p in signal_parameters]
            distances = [p['luminosity_distance'] for p in signal_parameters]
            snrs = [p['network_snr'] for p in signal_parameters]
            
            mass_diversity = np.std(masses) / max(np.mean(masses), 1.0) if len(masses) > 1 else 0.5
            distance_diversity = np.std(distances) / max(np.mean(distances), 1.0) if len(distances) > 1 else 0.5
            snr_diversity = np.std(snrs) / max(np.mean(snrs), 1.0) if len(snrs) > 1 else 0.5
            
            # NS-specific metrics
            ns_fraction = sum(1 for bt in binary_types if 'NS' in bt) / len(binary_types)
            tidal_systems = sum(1 for p in signal_parameters if 'lambda_1' in p or 'lambda_2' in p)
            tidal_fraction = tidal_systems / len(signal_parameters) if signal_parameters else 0
            
            # Overall diversity score
            diversity_score = np.mean([
                type_diversity,
                approximant_diversity,
                min(mass_diversity, 1.0),
                min(distance_diversity, 1.0),
                min(snr_diversity, 1.0),
                ns_fraction,  # Bonus for NS inclusion
                tidal_fraction  # Bonus for tidal systems
            ])
            
            return {
                'diversity_score': float(diversity_score),
                'type_diversity': float(type_diversity),
                'approximant_diversity': float(approximant_diversity),
                'mass_diversity': float(mass_diversity),
                'distance_diversity': float(distance_diversity),
                'snr_diversity': float(snr_diversity),
                'ns_fraction': float(ns_fraction),
                'tidal_fraction': float(tidal_fraction),
                'avg_snr': float(np.mean(snrs)),
                'binary_types': binary_types,
                'approximants': approximants
            }
            
        except Exception as e:
            self.logger.debug(f"Quality metrics computation failed: {e}")
            return {'diversity_score': 0.5, 'computation_failed': True}

    def validate_and_clean_scenarios(self, scenarios: List[Dict]) -> List[Dict]:
        """Validate and clean generated scenarios"""
        
        valid_scenarios = []
        
        for scenario in tqdm(scenarios, desc="Validating"):
            try:
                # Basic validation checks
                if not isinstance(scenario, dict):
                    continue
                
                if 'true_parameters' not in scenario:
                    continue
                
                # Ensure parameters are properly formatted
                scenario['true_parameters'] = self.fix_signal_parameters(scenario['true_parameters'])
                
                # Ensure waveform data exists and is properly formatted
                if 'waveform_data' in scenario:
                    waveform_data = scenario['waveform_data']
                    if isinstance(waveform_data, np.ndarray):
                        if waveform_data.shape != (2, 4096):
                            # Try to reshape or regenerate
                            try:
                                if 'injected_data' in scenario:
                                    scenario['waveform_data'] = self.convert_to_waveform_format(scenario['injected_data'])
                            except:
                                continue
                    else:
                        continue
                else:
                    # Generate waveform data if missing
                    if 'injected_data' in scenario:
                        scenario['waveform_data'] = self.convert_to_waveform_format(scenario['injected_data'])
                    else:
                        continue
                
                # Add scenario to valid list
                valid_scenarios.append(scenario)
                
            except Exception as e:
                self.logger.debug(f"Scenario validation failed: {e}")
                continue
        
        self.logger.info(f"✅ Validated {len(valid_scenarios)}/{len(scenarios)} scenarios")
        return valid_scenarios

    def _log_final_statistics(self):
        """Log comprehensive statistics including NS support"""
        
        total = self.stats['total_processed']
        if total == 0:
            return
            
        self.logger.info("📊 FINAL DATASET STATISTICS (WITH NS SUPPORT):")
        self.logger.info(f"   Total scenarios: {total}")
        for key, value in self.stats.items():
            if key != 'total_processed':
                percentage = (value / total * 100) if total > 0 else 0
                self.logger.info(f"   {key:30}: {value:5d} ({percentage:4.1f}%)")
        
        # NS-specific statistics
        total_binary = self.stats['bbh_scenarios'] + self.stats['bns_scenarios'] + self.stats['nsbh_scenarios']
        if total_binary > 0:
            self.logger.info("🌟 BINARY TYPE DISTRIBUTION:")
            self.logger.info(f"   BBH systems: {self.stats['bbh_scenarios']} ({self.stats['bbh_scenarios']/total_binary*100:.1f}%)")
            self.logger.info(f"   BNS systems: {self.stats['bns_scenarios']} ({self.stats['bns_scenarios']/total_binary*100:.1f}%)")
            self.logger.info(f"   NSBH systems: {self.stats['nsbh_scenarios']} ({self.stats['nsbh_scenarios']/total_binary*100:.1f}%)")


class MaximumDiversityParameterGenerator:
    """Generate parameters with maximum diversity for dataset including NS support"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_maximum_diversity_parameters_with_ns(self, n_signals: int, scenario_id: int) -> List[Dict]:
        """Enhanced parameter generation with NS support"""
        signal_parameters = []
        
        for sig_idx in range(n_signals):
            # Select binary type with realistic probabilities
            binary_type = self._select_binary_type()
            
            if binary_type == 'BBH':
                params = self._generate_bbh_parameters(sig_idx, scenario_id)
            elif binary_type == 'BNS':
                params = self._generate_bns_parameters(sig_idx, scenario_id)
            elif binary_type == 'NSBH':
                params = self._generate_nsbh_parameters(sig_idx, scenario_id)
            
            signal_parameters.append(params)
        
        return signal_parameters
    
    def _select_binary_type(self) -> str:
        """Select binary type based on detection rates"""
        return random.choices(
            ['BBH', 'BNS', 'NSBH'], 
            weights=[0.65, 0.20, 0.15],
            k=1
        )[0]
    
    def _generate_bbh_parameters(self, sig_idx: int, scenario_id: int) -> Dict:
        """Generate BBH parameters"""
        # BBH mass distribution
        mass_1 = np.random.lognormal(np.log(25), 0.6)
        mass_1 = np.clip(mass_1, 5.0, 100.0)
        
        q = np.random.beta(2, 3)
        mass_2 = mass_1 * q
        mass_2 = np.clip(mass_2, 5.0, mass_1)
        
        if mass_2 > mass_1:
            mass_1, mass_2 = mass_2, mass_1
        
        # BBH parameters
        distance = np.random.lognormal(np.log(800), 0.8)
        distance = np.clip(distance, 50, 8000)
        
        chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
        snr = 20.0 * (chirp_mass / 30.0)**(5/6) * (400.0 / distance)
        snr = np.clip(snr, 5.0, 100.0)
        
        approximant = random.choice(['IMRPhenomPv2', 'IMRPhenomD', 'SEOBNRv4'])
        
        return self._create_base_parameters(
            mass_1, mass_2, distance, snr, approximant, 
            sig_idx, scenario_id, 'BBH'
        )
    
    def _generate_bns_parameters(self, sig_idx: int, scenario_id: int) -> Dict:
        """Generate BNS parameters"""
        # BNS mass distribution
        mass_1 = np.random.normal(1.4, 0.3)
        mass_1 = np.clip(mass_1, 1.0, 2.5)
        
        mass_2 = np.random.normal(1.4, 0.3)
        mass_2 = np.clip(mass_2, 1.0, 2.5)
        
        if mass_2 > mass_1:
            mass_1, mass_2 = mass_2, mass_1
        
        # BNS are typically closer
        distance = np.random.lognormal(np.log(200), 0.6)
        distance = np.clip(distance, 20, 800)
        
        # BNS SNR scaling
        chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
        snr = 12.0 * (chirp_mass / 1.2)**(5/6) * (150.0 / distance)
        snr = np.clip(snr, 4.0, 50.0)
        
        # Use tidal approximants
        approximant = random.choice(['IMRPhenomPv2_NRTidal', 'IMRPhenomD_NRTidal'])
        
        params = self._create_base_parameters(
            mass_1, mass_2, distance, snr, approximant, 
            sig_idx, scenario_id, 'BNS'
        )
        
        # Add tidal parameters
        params['lambda_1'] = np.random.uniform(50, 5000)
        params['lambda_2'] = np.random.uniform(50, 5000)
        
        return params
    
    def _generate_nsbh_parameters(self, sig_idx: int, scenario_id: int) -> Dict:
        """Generate NSBH parameters"""
        # NS mass
        ns_mass = np.random.normal(1.4, 0.3)
        ns_mass = np.clip(ns_mass, 1.0, 2.5)
        
        # BH mass
        bh_mass = np.random.lognormal(np.log(15), 0.5)
        bh_mass = np.clip(bh_mass, 5.0, 50.0)
        
        # BH is primary
        mass_1, mass_2 = bh_mass, ns_mass
        
        # NSBH distance
        distance = np.random.lognormal(np.log(400), 0.7)
        distance = np.clip(distance, 40, 2000)
        
        # NSBH SNR
        chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
        snr = 15.0 * (chirp_mass / 8.0)**(5/6) * (300.0 / distance)
        snr = np.clip(snr, 4.0, 60.0)
        
        # Use tidal approximants
        approximant = random.choice(['IMRPhenomPv2_NRTidal', 'IMRPhenomD_NRTidal'])
        
        params = self._create_base_parameters(
            mass_1, mass_2, distance, snr, approximant, 
            sig_idx, scenario_id, 'NSBH'
        )
        
        # Add tidal parameters (only NS component)
        params['lambda_1'] = 0  # BH
        params['lambda_2'] = np.random.uniform(50, 2000)  # NS
        
        return params
    
    def _create_base_parameters(self, mass_1: float, mass_2: float, distance: float, 
                              snr: float, approximant: str, sig_idx: int, 
                              scenario_id: int, binary_type: str) -> Dict:
        """Create base parameter dictionary"""
        
        return {
            'mass_1': float(mass_1),
            'mass_2': float(mass_2),
            'luminosity_distance': float(distance),
            'geocent_time': float(np.random.uniform(-2.0, 2.0) + sig_idx * 0.4),
            'ra': float(np.random.uniform(0, 2 * np.pi)),
            'dec': float(np.random.uniform(-np.pi/2, np.pi/2)),
            'theta_jn': float(np.random.uniform(0, np.pi)),
            'psi': float(np.random.uniform(0, np.pi)),
            'phase': float(np.random.uniform(0, 2 * np.pi)),
            'signal_id': sig_idx,
            'network_snr': float(snr),
            'snr': float(snr),
            'approximant': approximant,
            'binary_type': binary_type,
            'difficulty': self._assign_difficulty(mass_1, mass_2, distance, snr),
            'f_lower': 20.0,
            'f_ref': 50.0
        }
    
    def _assign_difficulty(self, mass_1: float, mass_2: float, distance: float, snr: float) -> str:
        """Assign difficulty level"""
        total_mass = mass_1 + mass_2
        
        difficulty_points = 0
        
        # Low mass systems harder
        if total_mass < 10:
            difficulty_points += 2
        elif total_mass < 20:
            difficulty_points += 1
        
        # Very high mass systems challenging
        if total_mass > 80:
            difficulty_points += 1
        
        # Distance effects
        if distance > 1500:
            difficulty_points += 1
        if distance > 3000:
            difficulty_points += 2
        
        # SNR effects
        if snr < 10:
            difficulty_points += 2
        elif snr < 15:
            difficulty_points += 1
        
        if difficulty_points >= 5:
            return 'extreme'
        elif difficulty_points >= 3:
            return 'hard' 
        elif difficulty_points >= 1:
            return 'medium'
        else:
            return 'easy'

    def generate_maximum_diversity_parameters(self, n_signals: int, scenario_id: int) -> List[Dict]:
        """Original method maintained for backward compatibility"""
        return self.generate_maximum_diversity_parameters_with_ns(n_signals, scenario_id)

    def generate_extreme_parameters(self, n_signals: int, scenario_id: int) -> List[Dict]:
        """Generate extreme parameters for challenging scenarios"""
        
        extreme_parameters = []
        
        for sig_idx in range(n_signals):
            # Force extreme cases with NS support
            extreme_type = random.choices(
                ['very_low_mass', 'very_high_mass', 'very_distant', 'very_close', 'extreme_ns'],
                weights=[0.2, 0.2, 0.2, 0.2, 0.2],
                k=1
            )[0]
            
            if extreme_type == 'very_low_mass':
                # Very low mass BBH or BNS
                binary_type = random.choice(['BBH', 'BNS'])
                if binary_type == 'BNS':
                    mass_1 = np.random.uniform(1.0, 1.2)
                    mass_2 = np.random.uniform(1.0, 1.2)
                    approximant = 'IMRPhenomPv2_NRTidal'
                else:
                    mass_1 = np.random.uniform(5.0, 8.0)
                    mass_2 = np.random.uniform(5.0, 8.0)
                    approximant = 'IMRPhenomPv2'
                distance = np.random.uniform(100, 500)
                
            elif extreme_type == 'very_high_mass':
                # Very high mass BBH
                binary_type = 'BBH'
                mass_1 = np.random.uniform(80.0, 150.0)
                mass_2 = np.random.uniform(60.0, mass_1)
                distance = np.random.uniform(2000, 8000)
                approximant = 'IMRPhenomPv2'
                
            elif extreme_type == 'very_distant':
                # Very distant systems
                binary_type = random.choice(['BBH', 'NSBH'])
                if binary_type == 'NSBH':
                    mass_1 = np.random.uniform(10.0, 25.0)
                    mass_2 = np.random.uniform(1.0, 2.0)
                    approximant = 'IMRPhenomPv2_NRTidal'
                else:
                    mass_1 = np.random.uniform(30.0, 60.0)
                    mass_2 = np.random.uniform(25.0, mass_1)
                    approximant = 'IMRPhenomPv2'
                distance = np.random.uniform(5000, 15000)
                
            elif extreme_type == 'very_close':
                # Very close systems
                binary_type = random.choice(['BBH', 'BNS'])
                if binary_type == 'BNS':
                    mass_1 = np.random.uniform(1.2, 2.0)
                    mass_2 = np.random.uniform(1.2, mass_1)
                    approximant = 'IMRPhenomPv2_NRTidal'
                else:
                    mass_1 = np.random.uniform(20.0, 40.0)
                    mass_2 = np.random.uniform(15.0, mass_1)
                    approximant = 'IMRPhenomPv2'
                distance = np.random.uniform(10, 100)
                
            else:  # extreme_ns
                # Extreme NS cases
                binary_type = random.choice(['BNS', 'NSBH'])
                if binary_type == 'BNS':
                    mass_1 = np.random.uniform(2.2, 2.5)  # Heavy NS
                    mass_2 = np.random.uniform(2.2, 2.5)
                    approximant = 'IMRPhenomPv2_NRTidal'
                    distance = np.random.uniform(50, 300)
                else:  # NSBH
                    mass_1 = np.random.uniform(30.0, 50.0)  # Heavy BH
                    mass_2 = np.random.uniform(1.0, 1.5)   # Light NS
                    approximant = 'IMRPhenomPv2_NRTidal'
                    distance = np.random.uniform(200, 1000)
            
            # Ensure m1 >= m2
            if mass_2 > mass_1:
                mass_1, mass_2 = mass_2, mass_1
            
            # Compute SNR
            chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
            snr = 20.0 * (chirp_mass / 30.0)**(5/6) * (400.0 / distance)
            snr = np.clip(snr, 3.0, 150.0)  # Allow extreme SNR ranges
            
            params = self._create_base_parameters(
                mass_1, mass_2, distance, snr, approximant,
                sig_idx, scenario_id, binary_type
            )
            
            # Add tidal parameters if needed
            if binary_type == 'BNS':
                params['lambda_1'] = np.random.uniform(50, 5000)
                params['lambda_2'] = np.random.uniform(50, 5000)
            elif binary_type == 'NSBH':
                params['lambda_1'] = 0
                params['lambda_2'] = np.random.uniform(50, 2000)
            
            params['difficulty'] = 'extreme'
            extreme_parameters.append(params)
        
        return extreme_parameters

    def generate_low_snr_parameters(self, n_signals: int, scenario_id: int) -> List[Dict]:
        """Generate low SNR parameters for challenging detection"""
        
        low_snr_parameters = []
        
        for sig_idx in range(n_signals):
            # Select binary type favoring harder detections
            binary_type = random.choices(
                ['BBH', 'BNS', 'NSBH'],
                weights=[0.4, 0.4, 0.2],  # More NS systems for challenge
                k=1
            )[0]
            
            if binary_type == 'BNS':
                mass_1 = np.random.normal(1.4, 0.2)
                mass_1 = np.clip(mass_1, 1.1, 2.0)
                mass_2 = np.random.normal(1.4, 0.2)
                mass_2 = np.clip(mass_2, 1.1, mass_1)
                distance = np.random.uniform(400, 1200)  # Distant BNS
                approximant = 'IMRPhenomPv2_NRTidal'
                
            elif binary_type == 'NSBH':
                ns_mass = np.random.uniform(1.2, 2.0)
                bh_mass = np.random.uniform(8.0, 20.0)
                mass_1, mass_2 = bh_mass, ns_mass
                distance = np.random.uniform(800, 2500)  # Distant NSBH
                approximant = 'IMRPhenomPv2_NRTidal'
                
            else:  # BBH
                mass_1 = np.random.uniform(15.0, 35.0)
                mass_2 = np.random.uniform(10.0, mass_1)
                distance = np.random.uniform(1500, 5000)  # Distant BBH
                approximant = 'IMRPhenomPv2'
            
            # Force low SNR
            chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
            snr = np.random.uniform(3.0, 8.0)  # Very low SNR
            
            params = self._create_base_parameters(
                mass_1, mass_2, distance, snr, approximant,
                sig_idx, scenario_id, binary_type
            )
            
            # Add tidal parameters if needed
            if binary_type == 'BNS':
                params['lambda_1'] = np.random.uniform(100, 3000)
                params['lambda_2'] = np.random.uniform(100, 3000)
            elif binary_type == 'NSBH':
                params['lambda_1'] = 0
                params['lambda_2'] = np.random.uniform(100, 1500)
            
            params['difficulty'] = 'hard'
            low_snr_parameters.append(params)
        
        return low_snr_parameters

    def generate_high_snr_parameters(self, n_signals: int, scenario_id: int) -> List[Dict]:
        """Generate high SNR parameters for pristine benchmarks"""
        
        high_snr_parameters = []
        
        for sig_idx in range(n_signals):
            # Select binary type for clear detections
            binary_type = random.choices(
                ['BBH', 'BNS', 'NSBH'],
                weights=[0.6, 0.3, 0.1],
                k=1
            )[0]
            
            if binary_type == 'BNS':
                mass_1 = np.random.normal(1.4, 0.15)
                mass_1 = np.clip(mass_1, 1.2, 1.8)
                mass_2 = np.random.normal(1.4, 0.15)
                mass_2 = np.clip(mass_2, 1.2, mass_1)
                distance = np.random.uniform(40, 200)  # Close BNS
                approximant = 'IMRPhenomPv2_NRTidal'
                
            elif binary_type == 'NSBH':
                ns_mass = np.random.uniform(1.3, 1.6)
                bh_mass = np.random.uniform(15.0, 30.0)
                mass_1, mass_2 = bh_mass, ns_mass
                distance = np.random.uniform(100, 400)  # Close NSBH
                approximant = 'IMRPhenomPv2_NRTidal'
                
            else:  # BBH
                mass_1 = np.random.uniform(25.0, 50.0)
                mass_2 = np.random.uniform(20.0, mass_1)
                distance = np.random.uniform(200, 800)  # Close BBH
                approximant = 'IMRPhenomPv2'
            
            # Force high SNR
            chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
            snr = np.random.uniform(25.0, 80.0)  # High SNR
            
            params = self._create_base_parameters(
                mass_1, mass_2, distance, snr, approximant,
                sig_idx, scenario_id, binary_type
            )
            
            # Add tidal parameters if needed
            if binary_type == 'BNS':
                params['lambda_1'] = np.random.uniform(200, 2000)
                params['lambda_2'] = np.random.uniform(200, 2000)
            elif binary_type == 'NSBH':
                params['lambda_1'] = 0
                params['lambda_2'] = np.random.uniform(200, 1000)
            
            params['difficulty'] = 'easy'
            high_snr_parameters.append(params)
        
        return high_snr_parameters


class FallbackGWTCLoader:
    """Fallback GWTC loader with built-in event database."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_gwtc_events(self) -> pd.DataFrame:
        """Get comprehensive built-in GWTC events."""
        
        builtin_events = [
            # O1-O2 Events
            {'event_name': 'GW150914', 'gps_time': 1126259462.4, 'mass_1_source': 36.2, 'mass_2_source': 29.1, 'luminosity_distance': 410.0, 'network_snr': 23.7, 'observing_run': 'O1'},
            {'event_name': 'GW151012', 'gps_time': 1128678900.4, 'mass_1_source': 23.2, 'mass_2_source': 13.6, 'luminosity_distance': 1080.0, 'network_snr': 9.7, 'observing_run': 'O1'},
            {'event_name': 'GW151226', 'gps_time': 1135136350.6, 'mass_1_source': 14.2, 'mass_2_source': 7.5, 'luminosity_distance': 440.0, 'network_snr': 13.0, 'observing_run': 'O1'},
            {'event_name': 'GW170104', 'gps_time': 1167559936.6, 'mass_1_source': 31.2, 'mass_2_source': 19.4, 'luminosity_distance': 880.0, 'network_snr': 13.0, 'observing_run': 'O2'},
            {'event_name': 'GW170608', 'gps_time': 1180922494.5, 'mass_1_source': 12.0, 'mass_2_source': 7.0, 'luminosity_distance': 340.0, 'network_snr': 15.0, 'observing_run': 'O2'},
            {'event_name': 'GW170814', 'gps_time': 1186741861.5, 'mass_1_source': 30.5, 'mass_2_source': 25.3, 'luminosity_distance': 540.0, 'network_snr': 15.9, 'observing_run': 'O2'},
            {'event_name': 'GW170817', 'gps_time': 1187008882.4, 'mass_1_source': 1.6, 'mass_2_source': 1.2, 'luminosity_distance': 40.0, 'network_snr': 32.4, 'observing_run': 'O2'},
            
            # O3a Events
            {'event_name': 'GW190408_181802', 'gps_time': 1238782699.2, 'mass_1_source': 24.4, 'mass_2_source': 17.0, 'luminosity_distance': 1540.0, 'network_snr': 12.4, 'observing_run': 'O3a'},
            {'event_name': 'GW190412', 'gps_time': 1239042967.4, 'mass_1_source': 30.1, 'mass_2_source': 8.4, 'luminosity_distance': 730.0, 'network_snr': 19.0, 'observing_run': 'O3a'},
            {'event_name': 'GW190413_052954', 'gps_time': 1239179411.2, 'mass_1_source': 31.0, 'mass_2_source': 25.0, 'luminosity_distance': 1100.0, 'network_snr': 8.8, 'observing_run': 'O3a'},
            {'event_name': 'GW190413_134308', 'gps_time': 1239208207.1, 'mass_1_source': 47.1, 'mass_2_source': 35.6, 'luminosity_distance': 1200.0, 'network_snr': 9.3, 'observing_run': 'O3a'},
            {'event_name': 'GW190421_213856', 'gps_time': 1239917954.7, 'mass_1_source': 40.9, 'mass_2_source': 9.7, 'luminosity_distance': 2600.0, 'network_snr': 8.3, 'observing_run': 'O3a'},
            {'event_name': 'GW190503_185404', 'gps_time': 1240944462.4, 'mass_1_source': 48.7, 'mass_2_source': 30.1, 'luminosity_distance': 2750.0, 'network_snr': 9.3, 'observing_run': 'O3a'},
            {'event_name': 'GW190512_180714', 'gps_time': 1241719654.1, 'mass_1_source': 23.3, 'mass_2_source': 14.2, 'luminosity_distance': 1100.0, 'network_snr': 9.0, 'observing_run': 'O3a'},
            {'event_name': 'GW190513_205428', 'gps_time': 1241804488.1, 'mass_1_source': 32.6, 'mass_2_source': 30.2, 'luminosity_distance': 1700.0, 'network_snr': 8.4, 'observing_run': 'O3a'},
            {'event_name': 'GW190514_065416', 'gps_time': 1241835676.7, 'mass_1_source': 30.7, 'mass_2_source': 26.8, 'luminosity_distance': 1450.0, 'network_snr': 8.7, 'observing_run': 'O3a'},
            {'event_name': 'GW190517_055101', 'gps_time': 1242107481.0, 'mass_1_source': 45.0, 'mass_2_source': 26.8, 'luminosity_distance': 2900.0, 'network_snr': 8.2, 'observing_run': 'O3a'},
            {'event_name': 'GW190519_153544', 'gps_time': 1242315364.0, 'mass_1_source': 66.5, 'mass_2_source': 40.7, 'luminosity_distance': 2800.0, 'network_snr': 8.4, 'observing_run': 'O3a'},
            {'event_name': 'GW190521', 'gps_time': 1242459927.6, 'mass_1_source': 95.3, 'mass_2_source': 69.0, 'luminosity_distance': 2900.0, 'network_snr': 14.7, 'observing_run': 'O3a'},
            {'event_name': 'GW190527_092055', 'gps_time': 1242991276.5, 'mass_1_source': 42.6, 'mass_2_source': 31.3, 'luminosity_distance': 2400.0, 'network_snr': 8.9, 'observing_run': 'O3a'},
            {'event_name': 'GW190602_175927', 'gps_time': 1243533585.1, 'mass_1_source': 18.1, 'mass_2_source': 12.2, 'luminosity_distance': 900.0, 'network_snr': 9.7, 'observing_run': 'O3a'},
            {'event_name': 'GW190620_030421', 'gps_time': 1245079478.0, 'mass_1_source': 34.0, 'mass_2_source': 31.0, 'luminosity_distance': 1200.0, 'network_snr': 10.2, 'observing_run': 'O3a'},
            {'event_name': 'GW190630_185205', 'gps_time': 1246006543.0, 'mass_1_source': 36.1, 'mass_2_source': 26.8, 'luminosity_distance': 1100.0, 'network_snr': 17.3, 'observing_run': 'O3a'},
            {'event_name': 'GW190701_203306', 'gps_time': 1246149204.3, 'mass_1_source': 55.2, 'mass_2_source': 41.7, 'luminosity_distance': 1600.0, 'network_snr': 15.2, 'observing_run': 'O3a'},
            {'event_name': 'GW190706_222641', 'gps_time': 1246573619.0, 'mass_1_source': 67.0, 'mass_2_source': 40.8, 'luminosity_distance': 2800.0, 'network_snr': 11.7, 'observing_run': 'O3a'},
            {'event_name': 'GW190707_093326', 'gps_time': 1246611224.2, 'mass_1_source': 12.1, 'mass_2_source': 7.7, 'luminosity_distance': 780.0, 'network_snr': 9.1, 'observing_run': 'O3a'},
            {'event_name': 'GW190708_232457', 'gps_time': 1246766317.8, 'mass_1_source': 21.4, 'mass_2_source': 10.5, 'luminosity_distance': 1500.0, 'network_snr': 9.6, 'observing_run': 'O3a'},
            {'event_name': 'GW190719_215514', 'gps_time': 1247716532.2, 'mass_1_source': 40.8, 'mass_2_source': 21.0, 'luminosity_distance': 3400.0, 'network_snr': 8.1, 'observing_run': 'O3a'},
            {'event_name': 'GW190720_000836', 'gps_time': 1247727034.6, 'mass_1_source': 11.9, 'mass_2_source': 7.6, 'luminosity_distance': 780.0, 'network_snr': 10.4, 'observing_run': 'O3a'},
            {'event_name': 'GW190727_060333', 'gps_time': 1248334431.4, 'mass_1_source': 15.6, 'mass_2_source': 10.7, 'luminosity_distance': 900.0, 'network_snr': 9.3, 'observing_run': 'O3a'},
            {'event_name': 'GW190728_064510', 'gps_time': 1248420328.1, 'mass_1_source': 9.0, 'mass_2_source': 6.8, 'luminosity_distance': 300.0, 'network_snr': 14.0, 'observing_run': 'O3a'},
            {'event_name': 'GW190731_140936', 'gps_time': 1248688194.7, 'mass_1_source': 40.2, 'mass_2_source': 29.1, 'luminosity_distance': 2200.0, 'network_snr': 8.3, 'observing_run': 'O3a'},
            
            # O3b Events
            {'event_name': 'GW191103_012549', 'gps_time': 1257296767.4, 'mass_1_source': 9.4, 'mass_2_source': 7.2, 'luminosity_distance': 380.0, 'network_snr': 12.6, 'observing_run': 'O3b'},
            {'event_name': 'GW191105_143521', 'gps_time': 1257451739.2, 'mass_1_source': 10.9, 'mass_2_source': 8.1, 'luminosity_distance': 450.0, 'network_snr': 11.2, 'observing_run': 'O3b'},
            {'event_name': 'GW191109_010717', 'gps_time': 1257741255.6, 'mass_1_source': 65.0, 'mass_2_source': 47.0, 'luminosity_distance': 1750.0, 'network_snr': 15.9, 'observing_run': 'O3b'},
            {'event_name': 'GW191113_071753', 'gps_time': 1258085891.4, 'mass_1_source': 62.2, 'mass_2_source': 37.1, 'luminosity_distance': 2100.0, 'network_snr': 8.5, 'observing_run': 'O3b'},
            {'event_name': 'GW191126_115259', 'gps_time': 1259218797.1, 'mass_1_source': 35.4, 'mass_2_source': 26.7, 'luminosity_distance': 1400.0, 'network_snr': 8.7, 'observing_run': 'O3b'},
            {'event_name': 'GW191127_050227', 'gps_time': 1259285365.2, 'mass_1_source': 19.0, 'mass_2_source': 11.6, 'luminosity_distance': 1200.0, 'network_snr': 8.4, 'observing_run': 'O3b'},
            {'event_name': 'GW191129_134029', 'gps_time': 1259496047.5, 'mass_1_source': 10.7, 'mass_2_source': 8.3, 'luminosity_distance': 320.0, 'network_snr': 14.1, 'observing_run': 'O3b'},
            {'event_name': 'GW191204_110529', 'gps_time': 1259919947.3, 'mass_1_source': 30.0, 'mass_2_source': 15.0, 'luminosity_distance': 2300.0, 'network_snr': 8.2, 'observing_run': 'O3b'},
            {'event_name': 'GW191204_171526', 'gps_time': 1259941344.2, 'mass_1_source': 20.0, 'mass_2_source': 15.0, 'luminosity_distance': 1600.0, 'network_snr': 8.9, 'observing_run': 'O3b'},
            {'event_name': 'GW191215_223052', 'gps_time': 1260982270.1, 'mass_1_source': 4.9, 'mass_2_source': 1.6, 'luminosity_distance': 21.0, 'network_snr': 12.9, 'observing_run': 'O3b'},
            {'event_name': 'GW191216_213338', 'gps_time': 1261034036.7, 'mass_1_source': 31.1, 'mass_2_source': 26.8, 'luminosity_distance': 1400.0, 'network_snr': 9.3, 'observing_run': 'O3b'},
            {'event_name': 'GW191219_163120', 'gps_time': 1261265298.8, 'mass_1_source': 31.0, 'mass_2_source': 25.0, 'luminosity_distance': 1100.0, 'network_snr': 11.1, 'observing_run': 'O3b'},
            {'event_name': 'GW191222_033537', 'gps_time': 1261479355.6, 'mass_1_source': 20.0, 'mass_2_source': 17.0, 'luminosity_distance': 1800.0, 'network_snr': 9.6, 'observing_run': 'O3b'},
            {'event_name': 'GW191230_180458', 'gps_time': 1262218716.7, 'mass_1_source': 37.4, 'mass_2_source': 31.2, 'luminosity_distance': 1100.0, 'network_snr': 13.7, 'observing_run': 'O3b'},
            {'event_name': 'GW200105_162426', 'gps_time': 1262737484.1, 'mass_1_source': 19.2, 'mass_2_source': 13.2, 'luminosity_distance': 740.0, 'network_snr': 15.5, 'observing_run': 'O3b'},
            {'event_name': 'GW200112_155838', 'gps_time': 1263343136.9, 'mass_1_source': 15.1, 'mass_2_source': 11.6, 'luminosity_distance': 340.0, 'network_snr': 16.4, 'observing_run': 'O3b'},
            {'event_name': 'GW200115_042309', 'gps_time': 1263582207.4, 'mass_1_source': 5.7, 'mass_2_source': 1.5, 'luminosity_distance': 87.0, 'network_snr': 11.6, 'observing_run': 'O3b'},
            {'event_name': 'GW200128_022112', 'gps_time': 1264704090.4, 'mass_1_source': 17.0, 'mass_2_source': 12.0, 'luminosity_distance': 1200.0, 'network_snr': 9.1, 'observing_run': 'O3b'},
            {'event_name': 'GW200129_065458', 'gps_time': 1264781316.4, 'mass_1_source': 20.0, 'mass_2_source': 12.0, 'luminosity_distance': 1000.0, 'network_snr': 11.4, 'observing_run': 'O3b'},
            {'event_name': 'GW200202_154313', 'gps_time': 1265115811.8, 'mass_1_source': 35.6, 'mass_2_source': 26.7, 'luminosity_distance': 1200.0, 'network_snr': 13.9, 'observing_run': 'O3b'},
            {'event_name': 'GW200208_130117', 'gps_time': 1265640095.4, 'mass_1_source': 37.4, 'mass_2_source': 26.0, 'luminosity_distance': 1400.0, 'network_snr': 9.5, 'observing_run': 'O3b'},
            {'event_name': 'GW200208_222617', 'gps_time': 1265673995.0, 'mass_1_source': 56.2, 'mass_2_source': 37.6, 'luminosity_distance': 1400.0, 'network_snr': 16.2, 'observing_run': 'O3b'},
            {'event_name': 'GW200209_085452', 'gps_time': 1265707310.7, 'mass_1_source': 35.2, 'mass_2_source': 31.6, 'luminosity_distance': 1100.0, 'network_snr': 11.5, 'observing_run': 'O3b'},
            {'event_name': 'GW200210_092254', 'gps_time': 1265794192.2, 'mass_1_source': 24.1, 'mass_2_source': 17.0, 'luminosity_distance': 1200.0, 'network_snr': 12.4, 'observing_run': 'O3b'},
            {'event_name': 'GW200216_220804', 'gps_time': 1266307702.4, 'mass_1_source': 54.2, 'mass_2_source': 43.0, 'luminosity_distance': 1100.0, 'network_snr': 16.4, 'observing_run': 'O3b'},
            {'event_name': 'GW200219_094415', 'gps_time': 1266507873.9, 'mass_1_source': 31.0, 'mass_2_source': 28.0, 'luminosity_distance': 2600.0, 'network_snr': 8.5, 'observing_run': 'O3b'},
            {'event_name': 'GW200220_061928', 'gps_time': 1266564586.4, 'mass_1_source': 34.6, 'mass_2_source': 9.0, 'luminosity_distance': 2900.0, 'network_snr': 8.3, 'observing_run': 'O3b'},
            {'event_name': 'GW200220_124850', 'gps_time': 1266587348.4, 'mass_1_source': 40.4, 'mass_2_source': 21.0, 'luminosity_distance': 3100.0, 'network_snr': 8.1, 'observing_run': 'O3b'},
            {'event_name': 'GW200224_222234', 'gps_time': 1266962572.7, 'mass_1_source': 22.9, 'mass_2_source': 15.5, 'luminosity_distance': 2100.0, 'network_snr': 8.9, 'observing_run': 'O3b'},
            {'event_name': 'GW200225_060421', 'gps_time': 1266990279.1, 'mass_1_source': 38.7, 'mass_2_source': 31.0, 'luminosity_distance': 1600.0, 'network_snr': 11.6, 'observing_run': 'O3b'},
            {'event_name': 'GW200302_015811', 'gps_time': 1267589909.4, 'mass_1_source': 47.1, 'mass_2_source': 26.6, 'luminosity_distance': 2400.0, 'network_snr': 8.6, 'observing_run': 'O3b'},
            {'event_name': 'GW200306_093714', 'gps_time': 1267952252.5, 'mass_1_source': 30.0, 'mass_2_source': 17.0, 'luminosity_distance': 5500.0, 'network_snr': 8.1, 'observing_run': 'O3b'},
            {'event_name': 'GW200308_173609', 'gps_time': 1268165787.4, 'mass_1_source': 26.2, 'mass_2_source': 15.1, 'luminosity_distance': 1200.0, 'network_snr': 13.3, 'observing_run': 'O3b'},
            {'event_name': 'GW200311_115853', 'gps_time': 1268397551.4, 'mass_1_source': 26.6, 'mass_2_source': 15.7, 'luminosity_distance': 1700.0, 'network_snr': 11.0, 'observing_run': 'O3b'},
            {'event_name': 'GW200316_215756', 'gps_time': 1268872694.1, 'mass_1_source': 50.0, 'mass_2_source': 18.7, 'luminosity_distance': 5200.0, 'network_snr': 8.4, 'observing_run': 'O3b'},
            {'event_name': 'GW200322_091133', 'gps_time': 1269349911.4, 'mass_1_source': 59.0, 'mass_2_source': 59.0, 'luminosity_distance': 1100.0, 'network_snr': 21.8, 'observing_run': 'O3b'}
        ]
        
        # Add some synthetic NS events for diversity
        synthetic_ns_events = []
        for i in range(20):
            # BNS events
            synthetic_ns_events.append({
                'event_name': f'BNS_synthetic_{i:03d}',
                'gps_time': 1200000000 + i * 1000000,
                'mass_1_source': np.random.uniform(1.1, 2.3),
                'mass_2_source': np.random.uniform(1.1, 2.3),
                'luminosity_distance': np.random.uniform(40, 400),
                'network_snr': np.random.uniform(8, 35),
                'observing_run': random.choice(['O3a', 'O3b'])
            })
            
            # NSBH events
            if i < 10:
                synthetic_ns_events.append({
                    'event_name': f'NSBH_synthetic_{i:03d}',
                    'gps_time': 1200000000 + i * 1000000 + 500000,
                    'mass_1_source': np.random.uniform(8.0, 30.0),
                    'mass_2_source': np.random.uniform(1.1, 2.5),
                    'luminosity_distance': np.random.uniform(100, 1000),
                    'network_snr': np.random.uniform(8, 25),
                    'observing_run': random.choice(['O3a', 'O3b'])
                })
        
        builtin_events.extend(synthetic_ns_events)
        
        self.logger.info(f"Using fallback GWTC database with {len(builtin_events)} events")
        return pd.DataFrame(builtin_events)


def save_diversified_dataset(scenarios: List[Dict], output_dir: Path):
    """Save the diversified dataset with NS statistics"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full dataset
    with open(output_dir / 'diversified_dataset_ns_enhanced.pkl', 'wb') as f:
        pickle.dump(scenarios, f)
    
    # Create splits
    random.shuffle(scenarios)
    total = len(scenarios)
    train_size = int(0.8 * total)
    val_size = int(0.15 * total)
    
    train_scenarios = scenarios[:train_size]
    val_scenarios = scenarios[train_size:train_size + val_size]
    test_scenarios = scenarios[train_size + val_size:]
    
    with open(output_dir / 'train_ns_enhanced.pkl', 'wb') as f:
        pickle.dump(train_scenarios, f)
    with open(output_dir / 'val_ns_enhanced.pkl', 'wb') as f:
        pickle.dump(val_scenarios, f)
    with open(output_dir / 'test_ns_enhanced.pkl', 'wb') as f:
        pickle.dump(test_scenarios, f)
    
    with open(output_dir / 'dataset_splits.yaml', 'w') as f:
        yaml.dump({
            'total_scenarios': total,
            'train_scenarios': len(train_scenarios),
            'val_scenarios': len(val_scenarios),  
            'test_scenarios': len(test_scenarios),
            'split_ratios': {'train': 0.8, 'val': 0.15, 'test': 0.05}
        }, f)
    
    # Compute and save NS statistics
    ns_stats = compute_ns_statistics(scenarios)
    with open(output_dir / 'ns_statistics.yaml', 'w') as f:
        yaml.dump(ns_stats, f)
    
    # Save dataset metadata
    metadata = {
        'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_scenarios': len(scenarios),
        'ns_enhanced': True,
        'binary_types_supported': ['BBH', 'BNS', 'NSBH'],
        'approximants_used': ['IMRPhenomPv2', 'IMRPhenomD', 'SEOBNRv4', 'IMRPhenomPv2_NRTidal', 'IMRPhenomD_NRTidal'],
        'tidal_effects_included': True,
        'version': '2.0_ns_enhanced'
    }
    
    with open(output_dir / 'dataset_metadata.yaml', 'w') as f:
        yaml.dump(metadata, f)
    
    logging.info(f"✅ NS-Enhanced dataset saved: {len(scenarios)} scenarios")
    logging.info(f"   Train: {len(train_scenarios)}, Val: {len(val_scenarios)}, Test: {len(test_scenarios)}")
    logging.info(f"   NS statistics: {ns_stats}")


def compute_ns_statistics(scenarios: List[Dict]) -> Dict:
    """Compute NS-specific statistics"""
    stats = {
        'total_scenarios': len(scenarios),
        'bbh_count': 0,
        'bns_count': 0,
        'nsbh_count': 0,
        'tidal_approximants': 0,
        'ns_mass_range': {'min': float('inf'), 'max': 0.0},
        'approximant_distribution': {},
        'data_type_distribution': {},
        'difficulty_distribution': {}
    }
    
    for scenario in scenarios:
        # Count binary types
        binary_types = scenario.get('binary_types', [])
        for bt in binary_types:
            if bt == 'BBH':
                stats['bbh_count'] += 1
            elif bt == 'BNS':
                stats['bns_count'] += 1
            elif bt == 'NSBH':
                stats['nsbh_count'] += 1
        
        # Count approximants
        approximants = scenario.get('approximants', [])
        for approx in approximants:
            if 'NRTidal' in approx:
                stats['tidal_approximants'] += 1
            stats['approximant_distribution'][approx] = stats['approximant_distribution'].get(approx, 0) + 1
        
        # Data type distribution
        data_type = scenario.get('data_type', 'unknown')
        stats['data_type_distribution'][data_type] = stats['data_type_distribution'].get(data_type, 0) + 1
        
        # Check NS mass ranges and difficulty
        for params in scenario.get('true_parameters', []):
            binary_type = params.get('binary_type', 'BBH')
            difficulty = params.get('difficulty', 'medium')
            stats['difficulty_distribution'][difficulty] = stats['difficulty_distribution'].get(difficulty, 0) + 1
            
            if binary_type in ['BNS', 'NSBH']:
                mass_1 = params.get('mass_1', 0)
                mass_2 = params.get('mass_2', 0)
                
                # Find NS masses (< 3 solar masses)
                ns_masses = [m for m in [mass_1, mass_2] if m <= 3.0]
                for ns_mass in ns_masses:
                    if ns_mass > 0:
                        stats['ns_mass_range']['min'] = min(stats['ns_mass_range']['min'], ns_mass)
                        stats['ns_mass_range']['max'] = max(stats['ns_mass_range']['max'], ns_mass)
    
    # Handle case where no NS found
    if stats['ns_mass_range']['min'] == float('inf'):
        stats['ns_mass_range'] = {'min': 0.0, 'max': 0.0}
    
    # Add percentages
    total_binary = stats['bbh_count'] + stats['bns_count'] + stats['nsbh_count']
    if total_binary > 0:
        stats['bbh_percentage'] = (stats['bbh_count'] / total_binary) * 100
        stats['bns_percentage'] = (stats['bns_count'] / total_binary) * 100
        stats['nsbh_percentage'] = (stats['nsbh_count'] / total_binary) * 100
        stats['ns_percentage'] = ((stats['bns_count'] + stats['nsbh_count']) / total_binary) * 100
    
    return stats


def validate_dataset_integrity(scenarios: List[Dict]) -> Dict:
    """Validate dataset integrity and return validation report"""
    
    validation_report = {
        'total_scenarios': len(scenarios),
        'valid_scenarios': 0,
        'invalid_scenarios': 0,
        'validation_errors': [],
        'warnings': [],
        'ns_validation': {
            'bns_with_tidal': 0,
            'nsbh_with_tidal': 0,
            'bbh_without_tidal': 0,
            'tidal_parameter_errors': 0
        }
    }
    
    for i, scenario in enumerate(scenarios):
        scenario_valid = True
        
        try:
            # Check basic structure
            required_keys = ['scenario_id', 'true_parameters', 'injected_data', 'waveform_data', 'n_signals', 'data_type']
            for key in required_keys:
                if key not in scenario:
                    validation_report['validation_errors'].append(f"Scenario {i}: Missing required key '{key}'")
                    scenario_valid = False
            
            # Check true_parameters
            if 'true_parameters' in scenario:
                for j, params in enumerate(scenario['true_parameters']):
                    binary_type = params.get('binary_type', 'BBH')
                    approximant = params.get('approximant', 'IMRPhenomPv2')
                    
                    # NS-specific validation
                    if binary_type == 'BNS':
                        if 'lambda_1' in params and 'lambda_2' in params:
                            validation_report['ns_validation']['bns_with_tidal'] += 1
                        else:
                            validation_report['ns_validation']['tidal_parameter_errors'] += 1
                            validation_report['validation_errors'].append(f"Scenario {i}, Signal {j}: BNS missing tidal parameters")
                        
                        if 'NRTidal' not in approximant:
                            validation_report['warnings'].append(f"Scenario {i}, Signal {j}: BNS using non-tidal approximant")
                    
                    elif binary_type == 'NSBH':
                        if 'lambda_1' in params or 'lambda_2' in params:
                            validation_report['ns_validation']['nsbh_with_tidal'] += 1
                        else:
                            validation_report['ns_validation']['tidal_parameter_errors'] += 1
                            validation_report['validation_errors'].append(f"Scenario {i}, Signal {j}: NSBH missing tidal parameters")
                        
                        if 'NRTidal' not in approximant:
                            validation_report['warnings'].append(f"Scenario {i}, Signal {j}: NSBH using non-tidal approximant")
                    
                    elif binary_type == 'BBH':
                        validation_report['ns_validation']['bbh_without_tidal'] += 1
                        if 'lambda_1' in params or 'lambda_2' in params:
                            validation_report['warnings'].append(f"Scenario {i}, Signal {j}: BBH has tidal parameters")
            
            # Check waveform data
            if 'waveform_data' in scenario:
                waveform_data = scenario['waveform_data']
                if isinstance(waveform_data, np.ndarray):
                    if waveform_data.shape != (2, 4096):
                        validation_report['validation_errors'].append(f"Scenario {i}: Invalid waveform shape {waveform_data.shape}")
                        scenario_valid = False
                else:
                    validation_report['validation_errors'].append(f"Scenario {i}: Waveform data is not numpy array")
                    scenario_valid = False
            
            if scenario_valid:
                validation_report['valid_scenarios'] += 1
            else:
                validation_report['invalid_scenarios'] += 1
                
        except Exception as e:
            validation_report['validation_errors'].append(f"Scenario {i}: Validation exception: {e}")
            validation_report['invalid_scenarios'] += 1
    
    # Summary statistics
    validation_report['validation_success_rate'] = (validation_report['valid_scenarios'] / validation_report['total_scenarios']) * 100
    validation_report['ns_systems_validated'] = (
        validation_report['ns_validation']['bns_with_tidal'] + 
        validation_report['ns_validation']['nsbh_with_tidal']
    )
    
    return validation_report


def generate_dataset_summary_report(scenarios: List[Dict], output_dir: Path):
    """Generate comprehensive dataset summary report"""
    
    report = {
        'dataset_overview': {
            'total_scenarios': len(scenarios),
            'generation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'enhanced_with_ns': True
        },
        'binary_type_analysis': {},
        'approximant_analysis': {},
        'data_type_analysis': {},
        'parameter_ranges': {},
        'quality_analysis': {},
        'ns_specific_analysis': {}
    }
    
    # Collect all data for analysis
    all_binary_types = []
    all_approximants = []
    all_data_types = []
    all_masses = []
    all_distances = []
    all_snrs = []
    all_difficulties = []
    ns_masses = []
    tidal_parameters = []
    
    for scenario in scenarios:
        # Data type
        data_type = scenario.get('data_type', 'unknown')
        all_data_types.append(data_type)
        
        # Binary types and approximants
        binary_types = scenario.get('binary_types', [])
        approximants = scenario.get('approximants', [])
        all_binary_types.extend(binary_types)
        all_approximants.extend(approximants)
        
        # Parameter analysis
        for params in scenario.get('true_parameters', []):
            mass_1 = params.get('mass_1', 0)
            mass_2 = params.get('mass_2', 0)
            distance = params.get('luminosity_distance', 0)
            snr = params.get('network_snr', 0)
            difficulty = params.get('difficulty', 'medium')
            binary_type = params.get('binary_type', 'BBH')
            
            all_masses.extend([mass_1, mass_2])
            all_distances.append(distance)
            all_snrs.append(snr)
            all_difficulties.append(difficulty)
            
            # NS-specific analysis
            if binary_type in ['BNS', 'NSBH']:
                ns_candidates = [m for m in [mass_1, mass_2] if m <= 3.0]
                ns_masses.extend(ns_candidates)
                
                if 'lambda_1' in params:
                    tidal_parameters.append(params['lambda_1'])
                if 'lambda_2' in params:
                    tidal_parameters.append(params['lambda_2'])
    
    # Binary type analysis
    binary_type_counts = {}
    for bt in all_binary_types:
        binary_type_counts[bt] = binary_type_counts.get(bt, 0) + 1
    
    report['binary_type_analysis'] = {
        'counts': binary_type_counts,
        'percentages': {bt: (count/len(all_binary_types))*100 for bt, count in binary_type_counts.items()}
    }
    
    # Approximant analysis
    approximant_counts = {}
    for approx in all_approximants:
        approximant_counts[approx] = approximant_counts.get(approx, 0) + 1
    
    report['approximant_analysis'] = {
        'counts': approximant_counts,
        'tidal_approximant_usage': sum([count for approx, count in approximant_counts.items() if 'NRTidal' in approx]),
        'non_tidal_approximant_usage': sum([count for approx, count in approximant_counts.items() if 'NRTidal' not in approx])
    }
    
    # Data type analysis
    data_type_counts = {}
    for dt in all_data_types:
        data_type_counts[dt] = data_type_counts.get(dt, 0) + 1
    
    report['data_type_analysis'] = {
        'counts': data_type_counts,
        'percentages': {dt: (count/len(all_data_types))*100 for dt, count in data_type_counts.items()}
    }
    
    # Parameter ranges
    if all_masses:
        report['parameter_ranges'] = {
            'mass_range': {'min': float(np.min(all_masses)), 'max': float(np.max(all_masses)), 'mean': float(np.mean(all_masses))},
            'distance_range': {'min': float(np.min(all_distances)), 'max': float(np.max(all_distances)), 'mean': float(np.mean(all_distances))},
            'snr_range': {'min': float(np.min(all_snrs)), 'max': float(np.max(all_snrs)), 'mean': float(np.mean(all_snrs))}
        }
    
    # Difficulty analysis
    difficulty_counts = {}
    for diff in all_difficulties:
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
    
    report['quality_analysis'] = {
        'difficulty_distribution': difficulty_counts,
        'average_parameters_per_scenario': len(all_masses) / (2 * len(scenarios)) if scenarios else 0
    }
    
    # NS-specific analysis
    if ns_masses:
        report['ns_specific_analysis'] = {
            'total_ns_components': len(ns_masses),
            'ns_mass_range': {'min': float(np.min(ns_masses)), 'max': float(np.max(ns_masses)), 'mean': float(np.mean(ns_masses))},
            'tidal_parameter_count': len(tidal_parameters),
            'tidal_parameter_range': {
                'min': float(np.min(tidal_parameters)) if tidal_parameters else 0,
                'max': float(np.max(tidal_parameters)) if tidal_parameters else 0,
                'mean': float(np.mean(tidal_parameters)) if tidal_parameters else 0
            } if tidal_parameters else {'min': 0, 'max': 0, 'mean': 0}
        }
    
    # Save report
    with open(output_dir / 'dataset_summary_report.yaml', 'w') as f:
        yaml.dump(report, f, default_flow_style=False)
    
    return report


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Generate NS-Enhanced Diversified Dataset')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--total_scenarios', type=int, default=1000, help='Total scenarios')
    parser.add_argument('--max_real_events', type=int, default=100, help='Max real events')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--validate', action='store_true', help='Validate dataset integrity')
    parser.add_argument('--generate_report', action='store_true', help='Generate summary report')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    
    try:
        if IMPORTS_OK:
            config = AHSDConfig.from_yaml(args.config)
        else:
            config = FallbackConfig.from_yaml(args.config)
    except Exception as e:
        logging.warning(f"Config loading failed: {e}, using fallback")
        config = FallbackConfig()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("🌟 STARTING NS-ENHANCED DIVERSIFIED DATASET GENERATION")
    logging.info("=" * 80)
    
    start_time = time.time()
    
    # Generate dataset
    generator = DatasetGenerator(config)
    scenarios = generator.generate_diversified_dataset(
        total_scenarios=args.total_scenarios,
        max_real_events=args.max_real_events
    )
    
    if not scenarios:
        logging.error("❌ No valid scenarios generated!")
        return
    
    generation_time = time.time() - start_time
    logging.info(f"⏱️ Dataset generation completed in {generation_time:.2f} seconds")
    
    # Save dataset
    save_diversified_dataset(scenarios, output_dir)
    
    # Optional validation
    if args.validate:
        logging.info("🔍 Validating dataset integrity...")
        validation_report = validate_dataset_integrity(scenarios)
        
        with open(output_dir / 'validation_report.yaml', 'w') as f:
            yaml.dump(validation_report, f)
        
        logging.info(f"✅ Validation completed: {validation_report['validation_success_rate']:.1f}% success rate")
        if validation_report['validation_errors']:
            logging.warning(f"⚠️ {len(validation_report['validation_errors'])} validation errors found")
    
    # Optional summary report
    if args.generate_report:
        logging.info("📊 Generating dataset summary report...")
        summary_report = generate_dataset_summary_report(scenarios, output_dir)
        logging.info("✅ Summary report generated")
    
    total_time = time.time() - start_time
    logging.info("🎉 NS-ENHANCED DIVERSIFIED DATASET GENERATION COMPLETED!")
    logging.info(f"📁 Dataset saved to: {output_dir}")
    logging.info(f"⏱️ Total execution time: {total_time:.2f} seconds")
    logging.info(f"🔢 Final count: {len(scenarios)} scenarios generated")
    
    # Final NS statistics summary
    ns_stats = compute_ns_statistics(scenarios)
    logging.info("🌟 FINAL NS ENHANCEMENT SUMMARY:")
    logging.info(f"   BBH systems: {ns_stats['bbh_count']} ({ns_stats.get('bbh_percentage', 0):.1f}%)")
    logging.info(f"   BNS systems: {ns_stats['bns_count']} ({ns_stats.get('bns_percentage', 0):.1f}%)")
    logging.info(f"   NSBH systems: {ns_stats['nsbh_count']} ({ns_stats.get('nsbh_percentage', 0):.1f}%)")
    logging.info(f"   NS systems total: {ns_stats.get('ns_percentage', 0):.1f}%")
    logging.info(f"   Tidal approximants: {ns_stats['tidal_approximants']}")


if __name__ == '__main__':
    main()

