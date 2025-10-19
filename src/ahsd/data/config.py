"""
Configuration parameters for AHSD data generation
Extracted from main script configuration
"""

# Acquisition parameters
SAMPLE_RATE = 4096  # Hz
DURATION = 4.0  # seconds
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# Detector configuration
DETECTORS = ['H1', 'L1', 'V1']
DEFAULT_DETECTOR_NETWORK = ['H1', 'L1']

# SNR regimes
SNR_RANGES = {
    'weak': (8, 10),
    'low': (10, 14),
    'medium': (14, 20),
    'high': (20, 30),
    'loud': (30, 50)
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

# Event type distribution
EVENT_TYPE_DISTRIBUTION = {
    'BBH': 0.55,
    'BNS': 0.25,
    'NSBH': 0.15,
    'noise': 0.05
}

# SNR distribution
SNR_DISTRIBUTION = {
    'weak': 0.15,
    'low': 0.35,
    'medium': 0.30,
    'high': 0.15,
    'loud': 0.05
}

# Special cases
OVERLAP_FRACTION = 0.05
EDGE_CASE_FRACTION = 0.15

# Frequency bounds
F_LOWER = 20.0  # Hz
F_UPPER = 2048.0  # Hz

# Distance ranges (Mpc)
DISTANCE_RANGES = {
    'BBH': (100.0, 2000.0),
    'BNS': (10.0, 300.0),
    'NSBH': (20.0, 800.0)
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

TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
CREATE_SPLITS = True
STRATIFY_SPLITS = True
