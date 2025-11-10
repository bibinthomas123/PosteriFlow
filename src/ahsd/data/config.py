"""
AHSD Configuration Module
=========================
Production configuration for realistic O4 / Design LIGO sensitivity,
optimized for Neural Posterior Estimation and PriorityNet training.
Version: 2.0 - Physics-Validated
Generated: 2025-11-06
"""


# Sampling and duration
SAMPLE_RATE = 4096  # Hz
DURATION = 4.0  # seconds

# Detector configuration
DETECTORS = ['H1', 'L1', 'V1']

# Frequency range
F_LOWER = 20.0  # Hz
F_UPPER = 1024.0  # Hz

# Event type distribution
# These values are used throughout the code for:
# - Sampling event types in single/overlap/edge case generation
# - Validation logging to compare actual vs expected distributions

EVENT_TYPE_DISTRIBUTION = {
    'BBH': 0.46,    # Binary Black Hole mergers (46%)
    'BNS': 0.32,    # Binary Neutron Star mergers (32%)
    'NSBH': 0.17,   # Neutron Star-Black Hole mergers (17%)
    'noise': 0.05   # Noise-only samples (5%)
}

# SNR regime distribution
# Target SNR distribution (matches O4 detection rates)
SNR_DISTRIBUTION = {
    'weak': 0.05,     # 5%
    'low': 0.35,      # 35%
    'medium': 0.45,   # 45%
    'high': 0.12,     # 12%
    'loud': 0.03      # 3%
}

# SNR ranges for each regime
# Expanded ranges to allow stronger distance-SNR correlation
SNR_RANGES = { 
    'weak': (10.0, 15.0),
    'low': (15.0, 25.0),
    'medium': (25.0, 40.0),
    'high': (40.0, 60.0),
    'loud': (60.0, 80.0)

}

# Mass ranges (solar masses)
MASS_RANGES = {
    'BBH': (5.0, 100.0),
    'BNS': (1.0, 2.5),
    'NSBH': (1.0, 100.0)
}

# Distance ranges (Mpc) - O4/design sensitivity calibrated
# BBH: O4 horizon ~1000-1200 Mpc, observed GWTC-1 events up to 2840 Mpc
# BNS: O4 realistic range ~150-170 Mpc (design ~170 Mpc)
# NSBH: Intermediate mass systems scaling between BNS and BBH
DISTANCE_RANGES = {
    'BBH': (100.0, 2000.0),
    'BNS': (10.0, 300.0),
    'NSBH': (20.0, 800.0)
}


# Overlap and edge case configuration
OVERLAP_FRACTION = 0.35     # 35% overlapping signals
EDGE_CASE_FRACTION = 0.08   # 8% of total are edge cases


# Cosmology parameters
COSMO_H0 = 67.9  # km/s/Mpc
COSMO_OMEGA_M = 0.3065
COSMO_OMEGA_LAMBDA = 0.6935

# Approximants
APPROXIMANTS = {
    'BBH': {
        'non_precessing': ['SEOBNRv4', 'IMRPhenomXAS', 'IMRPhenomD'],
        'precessing': ['SEOBNRv4P', 'IMRPhenomPv2']
    },
    'BNS': {
        'tidal': ['TaylorF2', 'IMRPhenomD_NRTidal']
    },
    'NSBH': {
        'non_precessing': ['IMRPhenomD_NRTidal', 'IMRPhenomXAS', 'IMRPhenomD'],
        'precessing': ['IMRPhenomPv2_NRTidal', 'IMRPhenomPv2']
    }
}



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