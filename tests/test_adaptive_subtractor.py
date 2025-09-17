import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahsd.core.adaptive_subtractor import AdaptiveSubtractor, UncertaintyAwareSubtractor
from ahsd.models.neural_pe import NeuralPosteriorEstimator

@pytest.fixture
def mock_neural_pe():
    """Mock neural posterior estimator"""
    neural_pe = Mock(spec=NeuralPosteriorEstimator)
    
    # Mock posterior samples and summary
    mock_samples = {
        'mass_1': np.array([35.0, 36.0, 34.0]),
        'mass_2': np.array([30.0, 29.0, 31.0]),
        'luminosity_distance': np.array([500.0, 480.0, 520.0])
    }
    
    mock_summary = {
        'mass_1': {'median': 35.0, 'std': 1.0, 'quantiles': [33.0, 34.0, 36.0, 37.0]},
        'mass_2': {'median': 30.0, 'std': 1.0, 'quantiles': [28.0, 29.0, 31.0, 32.0]},
        'luminosity_distance': {'median': 500.0, 'std': 20.0, 'quantiles': [470.0, 485.0, 515.0, 530.0]}
    }
    
    neural_pe.quick_estimate.return_value = (mock_samples, mock_summary)
    
    return neural_pe

@pytest.fixture
def mock_waveform_generator():
    """Mock bilby waveform generator"""
    waveform_gen = Mock()
    
    # Mock waveform polarizations
    mock_polarizations = {
        'plus': Mock(),
        'cross': Mock()
    }
    
    waveform_gen.frequency_domain_strain.return_value = mock_polarizations
    
    return waveform_gen

@pytest.fixture
def mock_uncertainty_subtractor(mock_waveform_generator):
    """Mock uncertainty-aware subtractor"""
    subtractor = UncertaintyAwareSubtractor(mock_waveform_generator)
    
    # Mock the waveform generation to avoid bilby dependencies
    def mock_generate_waveform_realizations(samples, n_realizations):
        # Return mock waveform realizations
        mock_realizations = {}
        for det in ['H1', 'L1']:
            # Generate some fake waveform data
            n_samples = 16384  # 4 seconds at 4096 Hz
            realizations = np.random.normal(0, 1e-22, (n_realizations, n_samples))
            mock_realizations[det] = realizations
        return mock_realizations
    
    subtractor._generate_waveform_realizations = mock_generate_waveform_realizations
    
    return subtractor

@pytest.fixture
def sample_data():
    """Sample strain data"""
    n_samples = 16384  # 4 seconds at 4096 Hz
    return {
        'H1': np.random.normal(0, 1e-22, n_samples),
        'L1': np.random.normal(0, 1e-22, n_samples)
    }

def test_adaptive_subtractor_initialization(mock_neural_pe, mock_uncertainty_subtractor):
    """Test AdaptiveSubtractor initialization"""
    subtractor = AdaptiveSubtractor(mock_neural_pe, mock_uncertainty_subtractor)
    
    assert subtractor.neural_pe == mock_neural_pe
    assert subtractor.uncertainty_subtractor == mock_uncertainty_subtractor

def test_extract_and_subtract(mock_neural_pe, mock_uncertainty_subtractor, sample_data):
    """Test signal extraction and subtraction"""
    subtractor = AdaptiveSubtractor(mock_neural_pe, mock_uncertainty_subtractor)
    
    # Mock the uncertainty subtractor return
    mock_residual = {det: strain * 0.5 for det, strain in sample_data.items()}
    mock_uncertainty = {det: np.random.normal(0, 1e-23, len(strain)) 
                       for det, strain in sample_data.items()}
    
    mock_uncertainty_subtractor.subtract_with_uncertainty.return_value = (
        mock_residual, mock_uncertainty
    )
    
    residual_data, extraction_result, subtraction_uncertainty = subtractor.extract_and_subtract(
        sample_data, signal_priority=0
    )
    
    # Check outputs
    assert isinstance(residual_data, dict)
    assert isinstance(extraction_result, dict)
    assert isinstance(subtraction_uncertainty, dict)
    
    # Check extraction result structure
    assert 'signal_id' in extraction_result
    assert 'posterior_samples' in extraction_result
    assert 'posterior_summary' in extraction_result
    assert extraction_result['signal_id'] == 0

def test_uncertainty_aware_subtraction(mock_waveform_generator, sample_data):
    """Test uncertainty-aware subtraction"""
    subtractor = UncertaintyAwareSubtractor(mock_waveform_generator)
    
    # Mock posterior samples
    posterior_samples = {
        'mass_1': np.array([35.0, 36.0, 34.0]),
        'mass_2': np.array([30.0, 29.0, 31.0]),
        'ra': np.array([1.0, 1.1, 0.9]),
        'dec': np.array([0.5, 0.4, 0.6]),
        'psi': np.array([0.0, 0.1, -0.1])
    }
    
    # Mock waveform generation
    def mock_generate_realizations(samples, n_real):
        mock_realizations = {}
        for det in sample_data.keys():
            # Generate realistic waveform realizations
            n_samples = len(sample_data[det])
            realizations = np.random.normal(0, 1e-22, (n_real, n_samples))
            mock_realizations[det] = realizations
        return mock_realizations
    
    subtractor._generate_waveform_realizations = mock_generate_realizations
    
    residual_data, uncertainty = subtractor.subtract_with_uncertainty(
        sample_data, posterior_samples, n_realizations=10
    )
    
    # Check outputs
    assert isinstance(residual_data, dict)
    assert isinstance(uncertainty, dict)
    
    for det in sample_data.keys():
        assert det in residual_data
        assert det in uncertainty
        assert len(residual_data[det]) > 0
        assert len(uncertainty[det]) > 0

@patch('bilby.gw.detector.get_detector')
def test_generate_detector_waveform(mock_get_detector, mock_waveform_generator):
    """Test detector waveform generation"""
    # Mock detector
    mock_detector = Mock()
    mock_strain = Mock()
    mock_strain.time_domain_strain = np.random.normal(0, 1e-22, 1000)
    mock_detector.project_wave.return_value = mock_strain
    mock_get_detector.return_value = mock_detector
    
    subtractor = UncertaintyAwareSubtractor(mock_waveform_generator)
    
    params = {
        'mass_1': 35.0,
        'mass_2': 30.0,
        'ra': 1.0,
        'dec': 0.5,
        'psi': 0.0
    }
    
    strain = subtractor._generate_detector_waveform(params, 'H1')
    
    assert strain is not None
    assert len(strain) == 1000

def test_signal_quality_assessment(mock_neural_pe, mock_uncertainty_subtractor):
    """Test signal quality assessment"""
    subtractor = AdaptiveSubtractor(mock_neural_pe, mock_uncertainty_subtractor)
    
    # High quality posterior
    high_quality_summary = {
        'mass_1': {'median': 35.0, 'std': 1.0},
        'mass_2': {'median': 30.0, 'std': 1.0},
        'network_snr': {'median': 20.0, 'std': 1.0}
    }
    
    # Low quality posterior
    low_quality_summary = {
        'mass_1': {'median': 35.0, 'std': 10.0},
        'mass_2': {'median': 30.0, 'std': 10.0},
        'network_snr': {'median': 8.0, 'std': 2.0}
    }
    
    high_quality = subtractor._assess_signal_quality(high_quality_summary, {})
    low_quality = subtractor._assess_signal_quality(low_quality_summary, {})
    
    assert high_quality > low_quality
    assert 0 <= high_quality <= 1
    assert 0 <= low_quality <= 1

def test_fallback_behavior(mock_neural_pe, mock_uncertainty_subtractor, sample_data):
    """Test fallback behavior when extraction fails"""
    # Make neural PE fail
    mock_neural_pe.quick_estimate.side_effect = Exception("Neural PE failed")
    
    subtractor = AdaptiveSubtractor(mock_neural_pe, mock_uncertainty_subtractor)
    
    residual_data, extraction_result, subtraction_uncertainty = subtractor.extract_and_subtract(
        sample_data, signal_priority=0
    )
    
    # Should return fallback result
    assert extraction_result['extraction_method'] == 'fallback'
    assert 'posterior_summary' in extraction_result

def test_empty_data_handling(mock_neural_pe, mock_uncertainty_subtractor):
    """Test handling of empty or invalid data"""
    subtractor = AdaptiveSubtractor(mock_neural_pe, mock_uncertainty_subtractor)
    
    empty_data = {}
    
    residual_data, extraction_result, subtraction_uncertainty = subtractor.extract_and_subtract(
        empty_data, signal_priority=0
    )
    
    # Should handle gracefully
    assert isinstance(extraction_result, dict)
    assert 'signal_id' in extraction_result

def test_multiple_detectors(mock_neural_pe, mock_uncertainty_subtractor):
    """Test with multiple detector data"""
    subtractor = AdaptiveSubtractor(mock_neural_pe, mock_uncertainty_subtractor)
    
    multi_detector_data = {
        'H1': np.random.normal(0, 1e-22, 1000),
        'L1': np.random.normal(0, 1e-22, 1000),
        'V1': np.random.normal(0, 1e-22, 1000)
    }
    
    # Mock the uncertainty subtractor return
    mock_residual = {det: strain * 0.5 for det, strain in multi_detector_data.items()}
    mock_uncertainty = {det: np.random.normal(0, 1e-23, len(strain)) 
                       for det, strain in multi_detector_data.items()}
    
    mock_uncertainty_subtractor.subtract_with_uncertainty.return_value = (
        mock_residual, mock_uncertainty
    )
    
    residual_data, extraction_result, subtraction_uncertainty = subtractor.extract_and_subtract(
        multi_detector_data, signal_priority=0
    )
    
    # Should handle all detectors
    assert len(residual_data) == 3
    assert 'H1' in residual_data
    assert 'L1' in residual_data
    assert 'V1' in residual_data

def test_parameter_validation():
    """Test parameter validation in waveform generation"""
    subtractor = UncertaintyAwareSubtractor(Mock())
    
    # Test parameter completion
    incomplete_params = {'mass_1': 35.0, 'mass_2': 30.0}
    complete_params = subtractor.uncertainty_subtractor._complete_parameters(incomplete_params)
    
    required_params = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 
                      'theta_jn', 'psi', 'phase', 'geocent_time']
    
    for param in required_params:
        assert param in complete_params
    
    # Test parameter bounds
    assert 1.0 <= complete_params['mass_1'] <= 100.0
    assert 1.0 <= complete_params['mass_2'] <= 100.0
    assert 10.0 <= complete_params['luminosity_distance'] <= 5000.0

def test_snr_scaling():
    """Test SNR scaling functionality"""
    subtractor = UncertaintyAwareSubtractor(Mock())
    
    # Mock waveform and background
    waveform = np.random.normal(0, 1e-22, 1000)
    background = np.random.normal(0, 1e-23, 1000)
    target_snr = 15.0
    
    scaled_waveform = subtractor._scale_to_target_snr(
        waveform, target_snr, background, 'H1'
    )
    
    # Should return a scaled waveform
    assert len(scaled_waveform) == len(waveform)
    assert np.any(scaled_waveform != waveform)  # Should be different after scaling

if __name__ == "__main__":
    pytest.main([__file__])
