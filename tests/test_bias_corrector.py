import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahsd.core.bias_corrector import BiasCorrector, BiasEstimator

@pytest.fixture
def param_names():
    return ['mass_1', 'mass_2', 'luminosity_distance', 'geocent_time', 'ra', 'dec']

@pytest.fixture
def bias_corrector(param_names):
    return BiasCorrector(param_names)

@pytest.fixture
def sample_extracted_signals():
    return [
        {
            'signal_id': 0,
            'posterior_summary': {
                'mass_1': {'median': 35.0, 'std': 2.0, 'quantiles': [31.0, 33.0, 37.0, 39.0]},
                'mass_2': {'median': 30.0, 'std': 2.0, 'quantiles': [26.0, 28.0, 32.0, 34.0]},
                'luminosity_distance': {'median': 500.0, 'std': 100.0, 'quantiles': [300.0, 400.0, 600.0, 700.0]},
                'geocent_time': {'median': 0.01, 'std': 0.005, 'quantiles': [0.0, 0.005, 0.015, 0.02]},
                'ra': {'median': 1.0, 'std': 0.2, 'quantiles': [0.6, 0.8, 1.2, 1.4]},
                'dec': {'median': 0.5, 'std': 0.1, 'quantiles': [0.3, 0.4, 0.6, 0.7]}
            },
            'signal_quality': 0.8
        },
        {
            'signal_id': 1,
            'posterior_summary': {
                'mass_1': {'median': 40.0, 'std': 3.0, 'quantiles': [34.0, 37.0, 43.0, 46.0]},
                'mass_2': {'median': 25.0, 'std': 3.0, 'quantiles': [19.0, 22.0, 28.0, 31.0]},
                'luminosity_distance': {'median': 800.0, 'std': 200.0, 'quantiles': [400.0, 600.0, 1000.0, 1200.0]},
                'geocent_time': {'median': 0.05, 'std': 0.01, 'quantiles': [0.03, 0.04, 0.06, 0.07]},
                'ra': {'median': 2.0, 'std': 0.3, 'quantiles': [1.4, 1.7, 2.3, 2.6]},
                'dec': {'median': -0.3, 'std': 0.15, 'quantiles': [-0.6, -0.45, -0.15, 0.0]}
            },
            'signal_quality': 0.6
        }
    ]

def test_bias_corrector_initialization(bias_corrector, param_names):
    """Test BiasCorrector initialization"""
    assert bias_corrector.param_names == param_names
    assert isinstance(bias_corrector.bias_estimator, BiasEstimator)
    assert not bias_corrector.is_trained

def test_bias_estimator_forward():
    """Test BiasEstimator forward pass"""
    input_dim = 6
    bias_estimator = BiasEstimator(input_dim, [64, 32])
    
    # Test with batch size > 1
    param_estimates = torch.randn(5, input_dim)
    context_features = torch.randn(5, 5)
    
    output = bias_estimator(param_estimates, context_features)
    
    assert output.shape == (5, input_dim)
    assert torch.all(torch.isfinite(output))
    assert torch.all(torch.abs(output) <= 1)  # Tanh output bounds

def test_bias_estimator_single_sample():
    """Test BiasEstimator with single sample"""
    input_dim = 6
    bias_estimator = BiasEstimator(input_dim, [64, 32])
    
    # Test with single sample (batch size 1)
    param_estimates = torch.randn(1, input_dim)
    context_features = torch.randn(1, 5)
    
    output = bias_estimator(param_estimates, context_features)
    
    assert output.shape == (1, input_dim)
    assert torch.all(torch.isfinite(output))

def test_prepare_bias_correction_input(bias_corrector, sample_extracted_signals):
    """Test preparation of bias correction input"""
    signal = sample_extracted_signals[0]
    all_signals = sample_extracted_signals
    
    param_tensor, context_tensor = bias_corrector._prepare_bias_correction_input(
        signal, 0, all_signals
    )
    
    assert param_tensor is not None
    assert context_tensor is not None
    assert param_tensor.shape == (1, len(bias_corrector.param_names))
    assert context_tensor.shape[0] == 1
    assert context_tensor.shape[1] == 5  # 5 context features

def test_snr_estimation(bias_corrector):
    """Test SNR estimation from posterior"""
    # With explicit SNR
    posterior_with_snr = {
        'network_snr': {'median': 15.0}
    }
    snr = bias_corrector._compute_snr_estimate(posterior_with_snr)
    assert snr == 15.0
    
    # Without explicit SNR - estimate from masses and distance
    posterior_without_snr = {
        'mass_1': {'median': 35.0},
        'mass_2': {'median': 30.0},
        'luminosity_distance': {'median': 500.0}
    }
    snr = bias_corrector._compute_snr_estimate(posterior_without_snr)
    assert snr >= 8.0  # Should be reasonable estimate

def test_mass_ratio_computation(bias_corrector):
    """Test mass ratio computation"""
    posterior_summary = {
        'mass_1': {'median': 40.0},
        'mass_2': {'median': 20.0}
    }
    
    mass_ratio = bias_corrector._compute_mass_ratio(posterior_summary)
    assert mass_ratio == 0.5  # 20/40
    
    # Edge case: equal masses
    posterior_equal = {
        'mass_1': {'median': 30.0},
        'mass_2': {'median': 30.0}
    }
    
    mass_ratio_equal = bias_corrector._compute_mass_ratio(posterior_equal)
    assert mass_ratio_equal == 1.0

def test_parameter_scaling(bias_corrector):
    """Test parameter-specific scaling factors"""
    # Mass parameters
    mass_scale = bias_corrector._get_parameter_scale('mass_1', 35.0)
    assert mass_scale <= 10.0  # Capped at 10
    
    # Distance parameter
    dist_scale = bias_corrector._get_parameter_scale('luminosity_distance', 500.0)
    assert dist_scale <= 200.0  # Capped at 200
    
    # Time parameter
    time_scale = bias_corrector._get_parameter_scale('geocent_time', 0.01)
    assert time_scale == 0.01

def test_apply_bias_correction(bias_corrector, sample_extracted_signals):
    """Test application of bias corrections"""
    posterior_summary = sample_extracted_signals[0]['posterior_summary']
    bias_correction = np.array([1.0, -0.5, 10.0, 0.001, 0.05, -0.02])
    signal_quality = 0.8
    
    corrected_summary = bias_corrector._apply_bias_correction(
        posterior_summary, bias_correction, signal_quality
    )
    
    # Check structure is preserved
    assert set(corrected_summary.keys()) == set(posterior_summary.keys())
    
    # Check corrections were applied (scaled by quality)
    for i, param_name in enumerate(bias_corrector.param_names):
        if param_name in posterior_summary:
            original_median = posterior_summary[param_name]['median']
            corrected_median = corrected_summary[param_name]['median']
            
            # Should be different (corrected)
            assert corrected_median != original_median

def test_correct_hierarchical_biases(bias_corrector, sample_extracted_signals):
    """Test hierarchical bias correction"""
    # Mock trained model
    bias_corrector.is_trained = True
    
    # Mock bias estimator
    def mock_forward(param_tensor, context_tensor):
        return torch.zeros(param_tensor.shape[0], len(bias_corrector.param_names))
    
    bias_corrector.bias_estimator.forward = mock_forward
    
    corrected_signals = bias_corrector.correct_hierarchical_biases(sample_extracted_signals)
    
    assert len(corrected_signals) == len(sample_extracted_signals)
    
    for i, corrected_signal in enumerate(corrected_signals):
        assert 'posterior_summary' in corrected_signal
        assert 'bias_correction' in corrected_signal
        assert corrected_signal['signal_id'] == sample_extracted_signals[i]['signal_id']

def test_training_data_preparation(bias_corrector):
    """Test training data preparation"""
    training_scenarios = [
        {
            'true_parameters': {'mass_1': 35.0, 'mass_2': 30.0, 'luminosity_distance': 500.0},
            'extracted_parameters': {
                'mass_1': {'median': 36.0, 'std': 2.0},
                'mass_2': {'median': 29.0, 'std': 2.0},
                'luminosity_distance': {'median': 520.0, 'std': 50.0}
            },
            'extraction_position': 0,
            'total_signals': 2,
            'signal_quality': 0.8,
            'estimated_snr': 15.0
        },
        {
            'true_parameters': {'mass_1': 40.0, 'mass_2': 25.0, 'luminosity_distance': 800.0},
            'extracted_parameters': {
                'mass_1': {'median': 38.0, 'std': 3.0},
                'mass_2': {'median': 27.0, 'std': 3.0},
                'luminosity_distance': {'median': 750.0, 'std': 100.0}
            },
            'extraction_position': 1,
            'total_signals': 2,
            'signal_quality': 0.6,
            'estimated_snr': 12.0
        }
    ]
    
    inputs, targets, weights = bias_corrector._prepare_training_data(training_scenarios)
    
    assert len(inputs['params']) == len(training_scenarios)
    assert len(inputs['context']) == len(training_scenarios)
    assert len(targets) == len(training_scenarios)
    assert len(weights) == len(training_scenarios)
    
    # Check dimensions
    assert len(inputs['params'][0]) == len(bias_corrector.param_names)
    assert len(inputs['context'][0]) == 5  # Context features

def test_untrained_model_handling(bias_corrector, sample_extracted_signals):
    """Test handling when model is not trained"""
    # Ensure model is not trained
    bias_corrector.is_trained = False
    
    corrected_signals = bias_corrector.correct_hierarchical_biases(sample_extracted_signals)
    
    # Should return signals without bias correction
    assert len(corrected_signals) == len(sample_extracted_signals)
    
    for corrected_signal in corrected_signals:
        assert 'bias_correction' in corrected_signal
        # Bias correction should be zeros
        assert np.allclose(corrected_signal['bias_correction'], 0.0)

def test_empty_signals_handling(bias_corrector):
    """Test handling of empty signal list"""
    corrected_signals = bias_corrector.correct_hierarchical_biases([])
    assert corrected_signals == []

def test_missing_parameters_handling(bias_corrector):
    """Test handling of signals with missing parameters"""
    incomplete_signal = {
        'signal_id': 0,
        'posterior_summary': {
            'mass_1': {'median': 35.0, 'std': 2.0},
            # Missing other parameters
        },
        'signal_quality': 0.5
    }
    
    # Should not crash and should handle gracefully
    corrected_signals = bias_corrector.correct_hierarchical_biases([incomplete_signal])
    assert len(corrected_signals) == 1
    assert 'posterior_summary' in corrected_signals[0]

def test_model_save_load(bias_corrector, tmp_path):
    """Test model save and load functionality"""
    model_path = str(tmp_path / "test_bias_model.pth")
    
    # Modify some state
    bias_corrector.is_trained = True
    bias_corrector.correction_stats = {'test': 'data'}
    
    # Save model
    bias_corrector.save_model(model_path)
    
    # Create new corrector and load
    new_corrector = BiasCorrector(bias_corrector.param_names)
    new_corrector.load_model(model_path)
    
    assert new_corrector.is_trained == True
    assert new_corrector.correction_stats == {'test': 'data'}
    assert new_corrector.param_names == bias_corrector.param_names

def test_bias_estimation_network_training():
    """Test that bias estimation network can be trained"""
    bias_corrector = BiasCorrector(['mass_1', 'mass_2'])
    
    # Simple training scenario
    training_scenarios = [{
        'true_parameters': {'mass_1': 35.0, 'mass_2': 30.0},
        'extracted_parameters': {
            'mass_1': {'median': 36.0, 'std': 2.0},
            'mass_2': {'median': 29.0, 'std': 2.0}
        },
        'extraction_position': 0,
        'total_signals': 1,
        'signal_quality': 0.8,
        'estimated_snr': 15.0
    }]
    
    # Should not crash during training
    bias_corrector.train_bias_estimator(training_scenarios)
    
    # Should be marked as trained
    assert bias_corrector.is_trained

if __name__ == "__main__":
    pytest.main([__file__])
