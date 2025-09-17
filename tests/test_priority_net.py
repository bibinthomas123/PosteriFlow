import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahsd.core.priority_net import PriorityNet, SignalFeatureExtractor
from ahsd.utils.config import PriorityNetConfig

@pytest.fixture
def priority_net_config():
    return PriorityNetConfig(
        hidden_dims=[64, 32],
        dropout=0.1,
        learning_rate=1e-3,
        batch_size=32
    )

@pytest.fixture
def sample_detections():
    return [
        {
            'network_snr': 15.0,
            'chirp_mass_source': 30.0,
            'total_mass_source': 60.0,
            'mass_1': 35.0,
            'mass_2': 25.0,
            'luminosity_distance': 500.0,
            'geocent_time': 0.0,
            'ra': 1.0,
            'dec': 0.5,
            'a_1': 0.1,
            'a_2': 0.05
        },
        {
            'network_snr': 12.0,
            'chirp_mass_source': 25.0,
            'total_mass_source': 50.0,
            'mass_1': 30.0,
            'mass_2': 20.0,
            'luminosity_distance': 800.0,
            'geocent_time': 0.1,
            'ra': 1.2,
            'dec': 0.3,
            'a_1': 0.05,
            'a_2': 0.02
        }
    ]

def test_feature_extraction(sample_detections):
    """Test feature extraction for priority ranking"""
    extractor = SignalFeatureExtractor()
    
    features = extractor.extract_features(sample_detections)
    
    assert features.shape[0] == len(sample_detections)
    assert features.shape[1] == len(extractor.feature_names)
    assert torch.all(torch.isfinite(features))

def test_priority_net_forward(priority_net_config, sample_detections):
    """Test PriorityNet forward pass"""
    model = PriorityNet(priority_net_config)
    
    scores = model.forward(sample_detections)
    
    assert scores.shape[0] == len(sample_detections)
    assert torch.all(torch.isfinite(scores))
    assert torch.all(scores >= 0) and torch.all(scores <= 1)  # Sigmoid output

def test_priority_ranking(priority_net_config, sample_detections):
    """Test priority ranking functionality"""
    model = PriorityNet(priority_net_config)
    
    ranking = model.rank_detections(sample_detections)
    
    assert len(ranking) == len(sample_detections)
    assert set(ranking) == set(range(len(sample_detections)))

def test_priority_consistency(priority_net_config, sample_detections):
    """Test that higher SNR signals get higher priority (generally)"""
    model = PriorityNet(priority_net_config)
    
    # Make first signal clearly stronger
    sample_detections[0]['network_snr'] = 25.0
    sample_detections[1]['network_snr'] = 8.0
    
    # Run multiple times due to randomness
    first_ranked_first = 0
    n_trials = 10
    
    for _ in range(n_trials):
        ranking = model.rank_detections(sample_detections)
        if ranking[0] == 0:
            first_ranked_first += 1
    
    # Should be ranked first most of the time (allowing for some randomness)
    assert first_ranked_first >= n_trials // 2

def test_single_detection(priority_net_config):
    """Test ranking with single detection"""
    model = PriorityNet(priority_net_config)
    
    single_detection = [{
        'network_snr': 15.0,
        'chirp_mass_source': 30.0,
        'total_mass_source': 60.0,
        'mass_1': 35.0,
        'mass_2': 25.0
    }]
    
    ranking = model.rank_detections(single_detection)
    assert ranking == [0]

def test_empty_detections(priority_net_config):
    """Test ranking with no detections"""
    model = PriorityNet(priority_net_config)
    
    ranking = model.rank_detections([])
    assert ranking == []

def test_feature_extractor_missing_values():
    """Test feature extraction with missing parameter values"""
    extractor = SignalFeatureExtractor()
    
    incomplete_detection = [{
        'network_snr': 15.0,
        'mass_1': 35.0
        # Missing many parameters
    }]
    
    features = extractor.extract_features(incomplete_detection)
    
    assert features.shape[0] == 1
    assert features.shape[1] == len(extractor.feature_names)
    assert torch.all(torch.isfinite(features))

def test_astrophysical_significance():
    """Test astrophysical significance computation"""
    extractor = SignalFeatureExtractor()
    
    # High mass, high SNR signal
    high_sig_detection = {
        'network_snr': 20.0,
        'total_mass_source': 60.0,
        'mass_1': 35.0,
        'mass_2': 30.0,
        'luminosity_distance': 200.0
    }
    
    # Low mass, low SNR signal
    low_sig_detection = {
        'network_snr': 8.0,
        'total_mass_source': 20.0,
        'mass_1': 12.0,
        'mass_2': 8.0,
        'luminosity_distance': 1000.0
    }
    
    high_sig = extractor._compute_astrophysical_significance(high_sig_detection)
    low_sig = extractor._compute_astrophysical_significance(low_sig_detection)
    
    assert high_sig > low_sig

def test_overlap_computations():
    """Test frequency and time overlap computations"""
    extractor = SignalFeatureExtractor()
    
    signals = [
        {'chirp_mass_source': 30.0, 'geocent_time': 0.0},
        {'chirp_mass_source': 32.0, 'geocent_time': 0.1},  # Similar mass, close time
        {'chirp_mass_source': 50.0, 'geocent_time': 1.0}   # Different mass, far time
    ]
    
    freq_overlap_01 = extractor._compute_frequency_overlap(signals[0], signals)
    freq_overlap_02 = extractor._compute_frequency_overlap(signals[0], [signals[0], signals[2]])
    
    time_overlap_01 = extractor._compute_time_overlap(signals[0], signals)
    time_overlap_02 = extractor._compute_time_overlap(signals[0], [signals[0], signals[2]])
    
    # Should have higher overlap with similar signal
    assert freq_overlap_01 > freq_overlap_02
    assert time_overlap_01 > time_overlap_02

if __name__ == "__main__":
    pytest.main([__file__])
