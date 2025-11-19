import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahsd.core.priority_net import PriorityNet, SignalFeatureExtractor
from ahsd.utils.config import PriorityNetConfig


# ============================================================================
# Distribution Separation Metrics
# ============================================================================

class DistributionMetrics:
    """Compute distribution separation, entropy, and sharpness metrics"""
    
    @staticmethod
    def compute_entropy(probabilities):
        """
        Compute Shannon entropy of a probability distribution.
        Lower entropy = sharper/more concentrated distribution
        """
        # Clip to avoid log(0)
        probs = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        return entropy
    
    @staticmethod
    def compute_sharpness(scores, window_size=5):
        """
        Compute sharpness as inverse of smoothness.
        Higher sharpness = more concentrated/decisive predictions
        Measured as negative of moving average variance
        """
        if len(scores) < window_size:
            return 0.0
        
        # Calculate variance of moving average (smoothness indicator)
        smoothness = np.var(np.convolve(scores, 
                                         np.ones(window_size)/window_size, 
                                         mode='valid'))
        # Sharpness is inverse of smoothness
        sharpness = 1.0 / (1.0 + smoothness)
        return sharpness
    
    @staticmethod
    def compute_separation_auc(high_priority_scores, low_priority_scores):
        """
        Compute AUC for separating high vs low priority signals.
        Higher AUC = better separation between distributions
        AUC = 1.0 is perfect separation, 0.5 is random separation
        """
        if len(high_priority_scores) == 0 or len(low_priority_scores) == 0:
            return 0.5
        
        # Create binary labels: 1 for high priority, 0 for low priority
        y_true = np.concatenate([np.ones(len(high_priority_scores)), 
                                 np.zeros(len(low_priority_scores))])
        y_scores = np.concatenate([high_priority_scores, low_priority_scores])
        
        # Handle edge cases
        if len(np.unique(y_scores)) < 2:
            return 0.5
        
        try:
            auc = roc_auc_score(y_true, y_scores)
            return auc
        except:
            return 0.5
    
    @staticmethod
    def compute_distribution_divergence(dist1, dist2):
        """
        Compute KL divergence between two distributions.
        Lower divergence = more similar distributions
        """
        # Normalize distributions
        p = np.clip(dist1 / (np.sum(dist1) + 1e-10), 1e-10, 1.0)
        q = np.clip(dist2 / (np.sum(dist2) + 1e-10), 1e-10, 1.0)
        
        # KL(p||q) = sum(p * log(p/q))
        kl_divergence = np.sum(p * np.log(p / (q + 1e-10)))
        return kl_divergence
    
    @staticmethod
    def compute_wasserstein_distance(scores1, scores2):
        """
        Compute 1D Wasserstein distance between two distributions.
        Lower distance = more similar distributions
        """
        return stats.wasserstein_distance(scores1, scores2)


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


# ============================================================================
# Distribution Separation Tests (AUC, Entropy, Sharpness)
# ============================================================================

def test_distribution_separation_auc():
    """Test AUC metric for separating high vs low priority signals"""
    # Perfect separation case
    high_priority = np.array([0.9, 0.85, 0.95, 0.88, 0.92])
    low_priority = np.array([0.1, 0.15, 0.05, 0.12, 0.08])
    
    auc = DistributionMetrics.compute_separation_auc(high_priority, low_priority)
    
    # Should be near perfect separation (AUC ≈ 1.0)
    assert auc > 0.95, f"Expected AUC > 0.95, got {auc}"
    assert auc <= 1.0, f"Expected AUC <= 1.0, got {auc}"


def test_distribution_separation_auc_overlapping():
    """Test AUC metric with overlapping distributions"""
    # Overlapping case - more realistic
    high_priority = np.array([0.6, 0.7, 0.65, 0.75, 0.68])
    low_priority = np.array([0.4, 0.35, 0.45, 0.38, 0.42])
    
    auc = DistributionMetrics.compute_separation_auc(high_priority, low_priority)
    
    # Should show good separation but not perfect
    assert 0.6 < auc <= 1.0, f"Expected 0.6 < AUC <= 1.0, got {auc}"


def test_distribution_separation_auc_poor():
    """Test AUC metric with poor/no separation"""
    # Poor separation case
    high_priority = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    low_priority = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    auc = DistributionMetrics.compute_separation_auc(high_priority, low_priority)
    
    # Should be near random (AUC ≈ 0.5)
    assert abs(auc - 0.5) < 0.1, f"Expected AUC ≈ 0.5, got {auc}"


def test_entropy_sharp_distribution():
    """Test entropy metric on sharp/concentrated distributions"""
    # Sharp distribution: most probability mass at peak
    sharp_dist = np.array([0.01, 0.05, 0.88, 0.05, 0.01])
    
    entropy = DistributionMetrics.compute_entropy(sharp_dist)
    
    # Sharp distribution should have low entropy
    assert entropy < 1.0, f"Expected entropy < 1.0 for sharp distribution, got {entropy}"


def test_entropy_uniform_distribution():
    """Test entropy metric on uniform distribution"""
    # Uniform distribution: maximum entropy
    uniform_dist = np.ones(5) / 5
    
    entropy = DistributionMetrics.compute_entropy(uniform_dist)
    
    # Uniform distribution should have maximum entropy
    expected_entropy = np.log(5)  # log(N) for uniform distribution
    assert abs(entropy - expected_entropy) < 0.1, \
        f"Expected entropy ≈ {expected_entropy}, got {entropy}"


def test_sharpness_metric():
    """Test sharpness metric for decision decisiveness"""
    # Sharp/decisive scores
    sharp_scores = np.array([0.95, 0.94, 0.96, 0.95, 0.97, 0.96, 0.94, 0.95])
    
    # Blurry/uncertain scores
    blurry_scores = np.array([0.4, 0.6, 0.35, 0.65, 0.5, 0.55, 0.45, 0.58])
    
    sharp_sharpness = DistributionMetrics.compute_sharpness(sharp_scores)
    blurry_sharpness = DistributionMetrics.compute_sharpness(blurry_scores)
    
    # Sharp scores should have higher sharpness metric
    assert sharp_sharpness > blurry_sharpness, \
        f"Expected sharp > blurry, got {sharp_sharpness} vs {blurry_sharpness}"


def test_kl_divergence_identical_distributions():
    """Test KL divergence between identical distributions"""
    dist = np.array([0.2, 0.3, 0.3, 0.2])
    
    kl = DistributionMetrics.compute_distribution_divergence(dist, dist)
    
    # KL divergence between identical distributions should be near 0
    assert kl < 1e-5, f"Expected KL ≈ 0 for identical distributions, got {kl}"


def test_kl_divergence_different_distributions():
    """Test KL divergence between different distributions"""
    dist1 = np.array([0.8, 0.1, 0.05, 0.05])  # Concentrated
    dist2 = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform
    
    kl = DistributionMetrics.compute_distribution_divergence(dist1, dist2)
    
    # KL divergence should be positive and non-trivial
    assert kl > 0.5, f"Expected significant KL divergence, got {kl}"


def test_wasserstein_distance():
    """Test Wasserstein distance between score distributions"""
    # Two distinct distributions
    scores1 = np.array([0.9, 0.85, 0.95, 0.88, 0.92])
    scores2 = np.array([0.1, 0.15, 0.05, 0.12, 0.08])
    
    distance = DistributionMetrics.compute_wasserstein_distance(scores1, scores2)
    
    # Distance should be significant for well-separated distributions
    assert distance > 0.5, f"Expected large Wasserstein distance, got {distance}"


def test_wasserstein_distance_similar():
    """Test Wasserstein distance for similar distributions"""
    scores1 = np.array([0.5, 0.52, 0.48, 0.51, 0.49])
    scores2 = np.array([0.51, 0.49, 0.50, 0.52, 0.48])
    
    distance = DistributionMetrics.compute_wasserstein_distance(scores1, scores2)
    
    # Distance should be small for similar distributions
    assert distance < 0.1, f"Expected small Wasserstein distance, got {distance}"


def test_priority_net_separation_quality(priority_net_config, sample_detections):
    """Test overall distribution separation quality of PriorityNet"""
    model = PriorityNet(priority_net_config)
    
    # Create high and low SNR signal pairs
    high_snr_signals = []
    low_snr_signals = []
    
    for _ in range(10):
        high_snr = dict(sample_detections[0])
        high_snr['network_snr'] = 20.0 + np.random.rand() * 10
        high_snr_signals.append(high_snr)
        
        low_snr = dict(sample_detections[1])
        low_snr['network_snr'] = 5.0 + np.random.rand() * 5
        low_snr_signals.append(low_snr)
    
    # Get priority scores
    high_scores = model.forward(high_snr_signals).detach().numpy()
    low_scores = model.forward(low_snr_signals).detach().numpy()
    
    # Compute separation metrics
    auc = DistributionMetrics.compute_separation_auc(high_scores, low_scores)
    high_entropy = DistributionMetrics.compute_entropy(high_scores)
    low_entropy = DistributionMetrics.compute_entropy(low_scores)
    
    # Metrics should indicate good separation
    assert auc > 0.6, f"Expected AUC > 0.6, got {auc}"
    assert high_entropy < low_entropy or high_entropy < 1.5, \
        f"High SNR entropy should be reasonable, got {high_entropy}"


def test_distribution_metrics_batch_consistency():
    """Test that distribution metrics are stable across batches"""
    np.random.seed(42)
    
    # Generate batches of scores
    batch1_high = np.random.beta(8, 2, 20)  # High priority distribution
    batch1_low = np.random.beta(2, 8, 20)   # Low priority distribution
    
    batch2_high = np.random.beta(8, 2, 20)
    batch2_low = np.random.beta(2, 8, 20)
    
    # Compute metrics
    auc1 = DistributionMetrics.compute_separation_auc(batch1_high, batch1_low)
    auc2 = DistributionMetrics.compute_separation_auc(batch2_high, batch2_low)
    
    # Metrics should be consistent across similar distributions
    assert abs(auc1 - auc2) < 0.15, \
        f"Expected consistent AUC across batches, got {auc1} vs {auc2}"


def test_entropy_sharpness_correlation():
    """Test that entropy and sharpness are inversely correlated"""
    # Generate test scores
    np.random.seed(42)
    
    # Sharp scores (low entropy, high sharpness)
    sharp_scores = np.clip(np.random.normal(0.9, 0.05, 50), 0, 1)
    
    # Blurry scores (high entropy, low sharpness)
    blurry_scores = np.random.uniform(0.2, 0.8, 50)
    
    # Compute metrics
    sharp_entropy = DistributionMetrics.compute_entropy(sharp_scores)
    blurry_entropy = DistributionMetrics.compute_entropy(blurry_scores)
    
    sharp_sharpness = DistributionMetrics.compute_sharpness(sharp_scores)
    blurry_sharpness = DistributionMetrics.compute_sharpness(blurry_scores)
    
    # Sharp should have lower entropy and higher sharpness metric
    assert sharp_entropy < blurry_entropy, \
        f"Sharp entropy {sharp_entropy} should be < blurry {blurry_entropy}"
    assert sharp_sharpness > blurry_sharpness, \
        f"Sharp sharpness {sharp_sharpness} should be > blurry {blurry_sharpness}"

if __name__ == "__main__":
    pytest.main([__file__])
