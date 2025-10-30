"""
Configuration parameters for AHSD data generation
Optimized for PriorityNet and Neural Posterior Estimation
"""

# Acquisition parameters
SAMPLE_RATE = 4096  # Hz
DURATION = 4.0  # seconds

# Detector configuration
DETECTORS = ['H1', 'L1', 'V1']
DEFAULT_DETECTOR_NETWORK = ['H1', 'L1']

# ✅ FIXED: Correct SNR ranges matching dataset_generator.py
SNR_RANGES = {
    'weak': (7, 10),      # ✅ SNR 7-10
    'low': (10, 15),      # ✅ SNR 10-15
    'medium': (15, 25),   # ✅ SNR 15-25 - MOST TRAINING HERE
    'high': (25, 40),     # ✅ SNR 25-40 - Good for priority ranking
    'loud': (40, 80)      # ✅ SNR 40-80 - Rare but important
}

# Approximants by event type
APPROXIMANTS = {
    'BBH': {
        'non_precessing': ['IMRPhenomD'],
        'precessing': ['IMRPhenomPv2'],
        'tidal': []
    },
    'BNS': {
        'non_precessing': ['TaylorF2'],
        'precessing': [],
        'tidal': ['IMRPhenomD_NRTidal']
    },
    'NSBH': {
        'non_precessing': ['SEOBNRv4_ROM_NRTidal'],
        'precessing': ['IMRPhenomPv2_NRTidal'],
        'tidal': ['IMRPhenomD_NRTidal']
    }
}

# ✅ Event type distribution - Astrophysically realistic
EVENT_TYPE_DISTRIBUTION = {
    'BBH': 0.50,    # Most common astrophysically
    'BNS': 0.28,    # Less common than BBH
    'NSBH': 0.17,   # Intermediate
    'noise': 0.05   # For robustness
}

# ✅ SNR distribution - Balanced for training
SNR_DISTRIBUTION = {
    'weak': 0.15,   # Weak signals (detectable)
    'low': 0.30,    # Low SNR (training)
    'medium': 0.30, # MOST TRAINING HERE for clean priority learning
    'high': 0.20,   # Good for priority ranking
    'loud': 0.05    # Rare but important
}

# Special cases
OVERLAP_FRACTION = 0.40    # 40% overlap for PriorityNet training
EDGE_CASE_FRACTION = 0.12  # 12% edge cases (balanced)

# Frequency bounds
F_LOWER = 20.0  # Hz
F_UPPER = 2048.0  # Hz

# Distance ranges (Mpc)
DISTANCE_RANGES = {
    'BBH': (100.0, 4000.0),
    'BNS': (10.0, 500.0),
    'NSBH': (20.0, 1500.0)
}

# Mass ranges (solar masses)
MASS_RANGES = {
    'BBH': {'m1': (5.0, 100.0), 'm2': (5.0, 100.0)},
    'BNS': {'m1': (1.0, 2.5), 'm2': (1.0, 2.5)},
    'NSBH': {'m1': (3.0, 100.0), 'm2': (1.0, 2.5)}
}

# Cosmology (Planck 2018)
COSMO_H0 = 67.4  # km/s/Mpc
COSMO_OMEGA_M = 0.315
COSMO_OMEGA_LAMBDA = 0.685

# I/O parameters
CHUNK_SIZE = 100
SAVE_BATCH_SIZE = 100

# Strict validation
STRICT_SNR_ACCEPTANCE = True
SNR_TOLERANCE_LOW = 0.3
SNR_TOLERANCE_HIGH = 3.0

# Train/val/test split
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
CREATE_SPLITS = True
STRATIFY_SPLITS = True
