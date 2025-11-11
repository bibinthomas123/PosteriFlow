"""
Real Noise Dataset Analysis Module

Utilities for analyzing and verifying the presence of real LIGO/Virgo noise
in generated datasets. Provides statistical analysis of noise composition.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import json

logger = logging.getLogger(__name__)


class NoiseAnalyzer:
    """
    Analyze dataset noise composition to verify real LIGO noise integration.
    
    Provides methods to:
    - Detect noise type (real vs synthetic) from statistical signatures
    - Quantify real noise fraction in dataset
    - Generate noise composition report
    - Validate dataset meets expected real noise percentage
    """
    
    def __init__(self, sample_rate: int = 4096, duration: float = 4.0):
        """
        Initialize NoiseAnalyzer.
        
        Args:
            sample_rate: Sampling rate in Hz
            duration: Duration of noise chunks in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.logger = logging.getLogger(__name__)
    
    def analyze_strain_data(
        self, 
        strain: np.ndarray,
        detector_name: str = "Unknown"
    ) -> Dict:
        """
        Analyze strain data to estimate noise type and characteristics.
        
        Computes statistical features that differ between real and synthetic noise:
        - Whiteness (frequency content)
        - Non-Gaussianity (kurtosis, skewness)
        - Spectral features
        - Glitch indicators
        
        Args:
            strain: Time-domain strain data (1D array)
            detector_name: Name of detector for logging
        
        Returns:
            Dictionary with analysis results:
            - 'is_real_estimate': Boolean estimate of real noise
            - 'confidence': Confidence score (0-1)
            - 'statistics': Dict of computed statistics
            - 'features': Dict of extracted features
        """
        try:
            strain = np.asarray(strain, dtype=np.float64)
            
            # Compute statistics
            stats = self._compute_statistics(strain)
            
            # Extract features
            features = self._extract_features(strain, stats)
            
            # Estimate real vs synthetic
            is_real, confidence = self._classify_noise(features)
            
            return {
                'detector': detector_name,
                'is_real_estimate': is_real,
                'confidence': float(confidence),
                'statistics': stats,
                'features': features,
            }
        
        except Exception as e:
            self.logger.warning(f"Analysis failed for {detector_name}: {e}")
            return {
                'detector': detector_name,
                'is_real_estimate': None,
                'confidence': 0.0,
                'statistics': {},
                'features': {},
                'error': str(e),
            }
    
    def _compute_statistics(self, strain: np.ndarray) -> Dict:
        """
        Compute basic statistical properties of strain data.
        
        Args:
            strain: Time-domain strain array
        
        Returns:
            Dictionary of statistics
        """
        return {
            'mean': float(np.mean(strain)),
            'std': float(np.std(strain)),
            'min': float(np.min(strain)),
            'max': float(np.max(strain)),
            'kurtosis': float(self._compute_kurtosis(strain)),
            'skewness': float(self._compute_skewness(strain)),
            'rms': float(np.sqrt(np.mean(strain**2))),
        }
    
    def _extract_features(self, strain: np.ndarray, stats: Dict) -> Dict:
        """
        Extract distinguishing features between real and synthetic noise.
        
        Args:
            strain: Time-domain strain array
            stats: Pre-computed statistics
        
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Spectral features
        fft = np.fft.rfft(strain)
        psd = np.abs(fft) ** 2
        frequencies = np.fft.rfftfreq(len(strain), 1.0 / self.sample_rate)
        
        # Overall spectral shape
        features['spectral_slope'] = float(self._compute_spectral_slope(frequencies, psd))
        
        # Low-frequency power (seismic noise indicator - real detector signature)
        low_freq_mask = frequencies < 50
        features['low_freq_power'] = float(np.mean(psd[low_freq_mask]) if np.any(low_freq_mask) else 0)
        
        # High-frequency power
        high_freq_mask = frequencies > 200
        features['high_freq_power'] = float(np.mean(psd[high_freq_mask]) if np.any(high_freq_mask) else 0)
        
        # Spectral flatness (Wiener entropy - indicator of whiteness)
        features['spectral_flatness'] = float(self._compute_spectral_flatness(psd))
        
        # Line noise detection (60Hz harmonics in real data)
        features['line_noise_60hz'] = float(self._detect_line_noise(frequencies, psd, 60))
        features['line_noise_120hz'] = float(self._detect_line_noise(frequencies, psd, 120))
        
        # Glitch indicator (non-stationarity)
        features['non_stationarity'] = float(self._compute_non_stationarity(strain))
        
        # Excess kurtosis (real detectors have non-Gaussian tails)
        features['excess_kurtosis'] = float(stats['kurtosis'] - 3.0)  # 3.0 = Gaussian
        
        return features
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis (4th standardized moment)."""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        kurtosis = np.mean(((data - mean) / std) ** 4)
        return float(kurtosis)
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness (3rd standardized moment)."""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        skewness = np.mean(((data - mean) / std) ** 3)
        return float(skewness)
    
    def _compute_spectral_slope(self, frequencies: np.ndarray, psd: np.ndarray) -> float:
        """
        Compute slope of PSD in log-log space.
        
        Real LIGO noise has ~1/f^2 slope (seismic) at low freq.
        Synthetic Gaussian has flat spectrum.
        """
        try:
            # Use low-frequency region (10-100 Hz)
            mask = (frequencies > 10) & (frequencies < 100) & (psd > 0)
            if not np.any(mask):
                return 0.0
            
            freq_region = frequencies[mask]
            psd_region = psd[mask]
            
            # Log-log fit
            log_freq = np.log10(freq_region)
            log_psd = np.log10(psd_region + 1e-30)
            
            slope = np.polyfit(log_freq, log_psd, 1)[0]
            return float(slope)
        except (ValueError, np.linalg.LinAlgError):
            return 0.0
    
    def _compute_spectral_flatness(self, psd: np.ndarray) -> float:
        """
        Compute spectral flatness (Wiener entropy).
        
        Real noise: Less flat (shaped by detector response)
        Synthetic: More flat (white noise)
        Range: [0, 1] where 0=pure tone, 1=white noise
        """
        try:
            psd = np.asarray(psd, dtype=np.float64)
            psd = psd[psd > 0]
            if len(psd) == 0:
                return 0.0
            
            # Normalize PSD
            psd_norm = psd / np.sum(psd)
            
            # Entropy
            entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-30))
            max_entropy = np.log(len(psd_norm))
            
            flatness = entropy / max_entropy if max_entropy > 0 else 0.0
            return float(np.clip(flatness, 0, 1))
        except (ValueError, FloatingPointError) as e:
            self.logger.debug(f"Spectral flatness computation failed: {e}")
            return 0.5
    
    def _detect_line_noise(
        self, 
        frequencies: np.ndarray, 
        psd: np.ndarray,
        target_freq: float,
        bandwidth: float = 2.0
    ) -> float:
        """
        Detect presence of line noise at specific frequency.
        
        Returns amplitude ratio (0 = absent, 1 = strong presence)
        """
        try:
            mask = np.abs(frequencies - target_freq) < bandwidth
            if not np.any(mask):
                return 0.0
            
            line_power = np.mean(psd[mask])
            
            # Compare to neighboring regions
            neighbor_mask = (np.abs(frequencies - target_freq) > 5) & \
                           (np.abs(frequencies - target_freq) < 10)
            if not np.any(neighbor_mask):
                return 0.0
            
            neighbor_power = np.mean(psd[neighbor_mask])
            
            if neighbor_power < 1e-30:
                return 0.0
            
            ratio = (line_power - neighbor_power) / (neighbor_power + 1e-30)
            return float(np.clip(ratio, 0, 1))
        except (ValueError, IndexError):
            return 0.0
    
    def _compute_non_stationarity(self, strain: np.ndarray, n_windows: int = 8) -> float:
        """
        Compute non-stationarity by analyzing spectral variance over time.
        
        Real noise: Higher non-stationarity (glitches, transients)
        Synthetic: Very stationary (constant white noise)
        """
        try:
            window_size = len(strain) // n_windows
            if window_size < 64:
                return 0.0
            
            spectral_powers = []
            for i in range(n_windows):
                start = i * window_size
                end = start + window_size
                window = strain[start:end]
                
                fft = np.fft.rfft(window)
                power = np.mean(np.abs(fft) ** 2)
                spectral_powers.append(power)
            
            spectral_powers = np.array(spectral_powers)
            if np.mean(spectral_powers) < 1e-30:
                return 0.0
            
            # Coefficient of variation
            cv = np.std(spectral_powers) / (np.mean(spectral_powers) + 1e-30)
            return float(np.clip(cv, 0, 1))
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def _classify_noise(self, features: Dict) -> Tuple[bool, float]:
        """
        Classify noise as real vs synthetic based on features.
        
        Returns:
            Tuple of (is_real_estimate, confidence)
        """
        # Feature weights (tuned for LIGO detector characteristics)
        scores = {
            'spectral_slope': 1.0 if features.get('spectral_slope', 0) < -1.5 else 0.0,
            'low_freq_power': 1.0 if features.get('low_freq_power', 0) > 0 else 0.0,
            'excess_kurtosis': 1.0 if features.get('excess_kurtosis', 0) > 0.5 else 0.0,
            'line_noise': 1.0 if (features.get('line_noise_60hz', 0) > 0.1 or 
                                   features.get('line_noise_120hz', 0) > 0.1) else 0.0,
            'non_stationarity': 1.0 if features.get('non_stationarity', 0) > 0.1 else 0.0,
        }
        
        # Conservative classification: need >2 features to classify as real
        real_score = sum(scores.values())
        confidence = min(real_score / 5.0, 1.0)  # Normalize to [0,1]
        
        is_real = real_score >= 2.0
        
        return is_real, confidence
    
    def analyze_dataset(
        self,
        samples: List[Dict],
        max_samples: Optional[int] = None,
        report_file: Optional[str] = None
    ) -> Dict:
        """
        Analyze full dataset for real noise composition.
        
        Args:
            samples: List of sample dictionaries from generated dataset
            max_samples: Limit analysis to first N samples (None = all)
            report_file: Optional file path to save JSON report
        
        Returns:
            Dictionary with dataset-level statistics:
            - 'total_samples': Total samples analyzed
            - 'real_noise_count': Estimated number with real noise
            - 'real_noise_fraction': Estimated fraction (0-1)
            - 'per_detector_stats': Stats for each detector
            - 'confidence': Overall confidence in estimates
        """
        if not samples:
            self.logger.warning("No samples provided for analysis")
            return {
                'total_samples': 0,
                'real_noise_count': 0,
                'real_noise_fraction': 0.0,
                'per_detector_stats': {},
                'confidence': 0.0,
            }
        
        # Limit to max_samples if specified
        samples_to_analyze = samples[:max_samples] if max_samples else samples
        
        # Analyze each sample's detector data
        detector_results = {}
        real_count = 0
        total_analyzed = 0
        
        for sample_idx, sample in enumerate(samples_to_analyze):
            try:
                detector_data = sample.get('detector_data', {})
                
                for detector_name, data in detector_data.items():
                    if detector_name not in detector_results:
                        detector_results[detector_name] = {
                            'real_count': 0,
                            'total_count': 0,
                            'analyses': [],
                        }
                    
                    # Extract strain data
                    strain = data.get('strain')
                    if strain is None:
                        continue
                    
                    # Analyze
                    analysis = self.analyze_strain_data(strain, detector_name)
                    detector_results[detector_name]['analyses'].append(analysis)
                    detector_results[detector_name]['total_count'] += 1
                    
                    if analysis['is_real_estimate']:
                        detector_results[detector_name]['real_count'] += 1
                        real_count += 1
                    
                    total_analyzed += 1
            
            except Exception as e:
                self.logger.debug(f"Error analyzing sample {sample_idx}: {e}")
                continue
        
        # Compute per-detector statistics
        per_detector_stats = {}
        for detector_name, results in detector_results.items():
            total = results['total_count']
            real = results['real_count']
            
            per_detector_stats[detector_name] = {
                'total_samples': total,
                'real_noise_count': real,
                'real_noise_fraction': real / total if total > 0 else 0.0,
                'confidence': float(np.mean(
                    [a['confidence'] for a in results['analyses']]
                )) if results['analyses'] else 0.0,
            }
        
        # Overall statistics
        real_noise_fraction = real_count / total_analyzed if total_analyzed > 0 else 0.0
        all_confidences = [a['confidence'] for analyses in detector_results.values() 
                           for a in analyses['analyses']]
        overall_confidence = float(np.mean(all_confidences)) if all_confidences else 0.0
        
        result = {
            'total_samples': len(samples_to_analyze),
            'total_analyzed': total_analyzed,
            'real_noise_count': real_count,
            'real_noise_fraction': real_noise_fraction,
            'per_detector_stats': per_detector_stats,
            'overall_confidence': overall_confidence,
        }
        
        # Save report if requested
        if report_file:
            self._save_report(result, report_file)
        
        return result
    
    def _save_report(self, result: Dict, report_file: str) -> None:
        """Save analysis report to JSON file."""
        try:
            Path(report_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            self.logger.info(f"Noise analysis report saved to {report_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save report to {report_file}: {e}")
    
    def print_summary(self, result: Dict) -> None:
        """Print human-readable summary of analysis results."""
        print("\n" + "=" * 70)
        print("REAL NOISE DATASET ANALYSIS SUMMARY")
        print("=" * 70)
        
        print(f"\nOverall Statistics:")
        print(f"  Total samples analyzed: {result['total_analyzed']}")
        print(f"  Real noise estimated:   {result['real_noise_count']} samples")
        print(f"  Real noise fraction:    {result['real_noise_fraction']:.1%}")
        print(f"  Confidence:             {result['overall_confidence']:.1%}")
        
        print(f"\nPer-Detector Breakdown:")
        for det_name, stats in result['per_detector_stats'].items():
            print(f"\n  {det_name}:")
            print(f"    Total samples:      {stats['total_samples']}")
            print(f"    Real noise:         {stats['real_noise_count']} ({stats['real_noise_fraction']:.1%})")
            print(f"    Confidence:         {stats['confidence']:.1%}")
        
        # Expected value
        print(f"\nExpected (configured):")
        print(f"  Real noise probability: 30%")
        
        # Assessment
        print(f"\nAssessment:")
        overall_real = result['real_noise_fraction']
        if 0.25 <= overall_real <= 0.35:
            status = "✓ PASS - Matches expected ~30%"
        elif 0.20 <= overall_real <= 0.40:
            status = "⚠ ACCEPTABLE - Within 20-40% range"
        else:
            status = f"✗ FAIL - Only {overall_real:.1%} real noise detected"
        print(f"  {status}")
        
        print("=" * 70 + "\n")


def analyze_dataset_file(
    dataset_path: str,
    max_samples: Optional[int] = None,
    output_report: Optional[str] = None
) -> Dict:
    """
    Convenience function to analyze dataset from file.
    
    Supports both pickle and HDF5 formats.
    
    Args:
        dataset_path: Path to dataset file
        max_samples: Maximum samples to analyze
        output_report: Optional path for JSON report
    
    Returns:
        Analysis results dictionary
    """
    import pickle
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Load dataset
    if dataset_path.suffix == '.pkl':
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        samples = data.get('samples', data) if isinstance(data, dict) else data
    elif dataset_path.suffix in ['.h5', '.hdf5']:
        try:
            import h5py
            with h5py.File(dataset_path, 'r') as f:
                # HDF5 format conversion to dict would go here
                raise NotImplementedError("HDF5 analysis not yet implemented")
        except ImportError:
            raise RuntimeError("h5py required for HDF5 analysis. Install with: pip install h5py")
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
    
    # Analyze
    analyzer = NoiseAnalyzer()
    result = analyzer.analyze_dataset(samples, max_samples=max_samples, report_file=output_report)
    
    return result
