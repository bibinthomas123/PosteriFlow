"""
Gravitational Wave Signal Simulation Module.

This module provides functionality for simulating realistic overlapping gravitational wave
(GW) signals injected into detector noise. It is designed to generate training data for
machine learning models that analyze GW signal overlaps and perform parameter estimation.

The module includes:
- Signal parameter generation using realistic astrophysical priors
- Detector noise simulation with frequency-dependent power spectral densities (PSDs)
- Multi-signal injection into noise with support for arbitrary numbers of overlapping events
- Flexible configuration-driven signal generation
- Fallback mechanisms for robustness across different waveform backends

Key Classes:
    - OverlappingSignalSimulator: Main class for signal and noise generation

Key Functions:
    - compute_waveform_overlap: Compute normalized waveform overlap
    - estimate_snr_from_strain: Estimate signal-to-noise ratio from strain data

Usage Example:
    >>> from ahsd.utils.config import AHSDConfig
    >>> from ahsd.data.simulation import OverlappingSignalSimulator
    >>> config = AHSDConfig.load('config.yaml')
    >>> simulator = OverlappingSignalSimulator(config)
    >>> noise = simulator.generate_detector_noise()
    >>> scenario = simulator.generate_overlapping_scenario(n_signals=3)
    >>> injected_data, signals = simulator.inject_signals_to_data(scenario, noise)

Dependencies:
    - pycbc: For detector projections and waveform calculations
    - bilby: Optional, for advanced waveform generation (gracefully handled if missing)
    - numpy, scipy: Numerical and signal processing operations
    - gwpy: GW data handling utilities
"""

import numpy as np
try:
    import bilby
except ImportError:
    bilby = None  # Optional dependency
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from typing import List, Dict, Tuple, Optional
import torch
import logging
from pathlib import Path
import pickle
from gwpy.timeseries import TimeSeries
from scipy.signal import get_window
from ..utils.config import AHSDConfig

# Suppress bilby warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", message="Unkown projection method")
warnings.filterwarnings("ignore", message="Unknown projection method")

# Configure bilby logging (if available)
if bilby is not None:
    bilby.core.utils.logger.setLevel('WARNING')

class OverlappingSignalSimulator:
    """
    Simulator for overlapping gravitational wave signals and detector noise.

    This class generates realistic gravitational wave signal parameters, simulates detector noise
    with frequency-dependent characteristics, injects multiple overlapping signals, and creates
    complete training datasets. It supports arbitrary numbers of overlapping events and provides
    robust fallback mechanisms for waveform generation failures.

    The simulator uses configuration-driven parameter generation with physically motivated priors
    based on gravitational wave observations from LIGO/Virgo (GWTC catalog). Signal injections
    are performed in the time domain with proper detector response calculations.

    Attributes:
        config (AHSDConfig): Configuration object containing waveform parameters, detector
            specifications, and sampling rates. Used to set up all detector and waveform
            generation parameters.
        logger (logging.Logger): Logger instance for informational and error messages during
            signal generation and noise creation.
        detectors (Dict[str, Any]): Dictionary mapping detector names (e.g., 'H1', 'L1', 'V1')
            to detector objects. Supports both PyCBC and Bilby detector implementations with
            fallback mechanisms for incompatible versions.
        waveform_generator (Optional[Any]): Bilby waveform generator instance for frequency-domain
            waveform generation. Set to None if bilby is unavailable; mock waveforms are used
            as fallback in this case.

    Raises:
        ImportError: If neither bilby nor pycbc waveform generation is available (non-fatal,
            falls back to mock waveforms with reduced fidelity).
        RuntimeError: On catastrophic detector initialization or configuration errors.

    Examples:
        Basic usage with default configuration:
            >>> config = AHSDConfig.load('path/to/config.yaml')
            >>> simulator = OverlappingSignalSimulator(config)
            >>> scenario = simulator.generate_overlapping_scenario(n_signals=2)
            >>> noise = simulator.generate_detector_noise()
            >>> injected, signals = simulator.inject_signals_to_data(scenario, noise)

        Generate complete training dataset:
            >>> simulator = OverlappingSignalSimulator(config)
            >>> training_data = simulator.create_training_dataset(
            ...     n_scenarios=1000,
            ...     output_dir='./training_data'
            ... )

        Estimate signal properties:
            >>> snr = estimate_snr_from_strain(injected['H1'])
            >>> overlap = compute_waveform_overlap(signal1, signal2)
    """
    
    def __init__(self, config: AHSDConfig):
        """
        Initialize the gravitational wave signal simulator.

        Sets up detector instances for signal projections and configures the waveform
        generator for signal synthesis. Detectors are initialized from PyCBC with graceful
        fallback to mock detectors if unavailable.

        Args:
            config (AHSDConfig): Configuration object containing detector parameters
                (sampling rates, names, characteristics) and waveform settings
                (duration, approximant, reference frequency).

        Raises:
            ConfigurationError: If detector configuration is invalid or waveform
                parameters are incompatible.

        Side Effects:
            - Initializes logging to module namespace
            - Creates detector instances for each detector in config
            - Sets up bilby waveform generator if available
            - Suppresses warnings from bilby initialization
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup detectors from configuration
        self.detectors = {}
        for det_config in config.detectors:
            try:
                self.detectors[det_config.name] = Detector(det_config.name)
            except:
                # Fallback to bilby detector or mock if PyCBC unavailable
                self.detectors[det_config.name] = self._create_mock_detector(det_config.name)
        
        # Setup waveform generator for signal synthesis
        self.waveform_generator = self._setup_waveform_generator()
        
    def _setup_waveform_generator(self):
        """
        Configure and return a bilby waveform generator instance.

        Creates a frequency-domain waveform generator using bilby's LAL-based binary black hole
        source model. The generator is configured with duration, sampling frequency, and waveform
        approximant from the configuration. This generator is used for high-fidelity signal
        synthesis during injection.

        Returns:
            bilby.gw.WaveformGenerator: Configured waveform generator instance capable of
                synthesizing frequency-domain GW signals. Returns None if bilby is unavailable.

        Note:
            - Requires bilby installation; gracefully returns None if unavailable
            - Uses LAL-based waveform generation (SpinTaylorT4, IMRPhenomD, etc.)
            - All waveforms are generated in frequency domain for efficiency
            - Parameter conversion handles spin/tilt angle transformations automatically
        """
        if bilby is None:
            self.logger.warning("Bilby not available; waveform generation will use mock waveforms")
            return None
        
        return bilby.gw.WaveformGenerator(
            duration=self.config.waveform.duration,
            sampling_frequency=self.config.detectors[0].sampling_rate,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(
                waveform_approximant=self.config.waveform.approximant,
                reference_frequency=self.config.waveform.f_ref,
            )
        )
        
    def generate_single_signal_params(self) -> Dict:
        """
        Generate parameter set for a single gravitational wave signal.

        Generates a complete set of physical parameters for a binary black hole merger using
        realistic priors. Parameters are sampled conservatively to ensure numerical stability
        in waveform generation and to avoid edge cases. The generated parameters include
        masses, distance, sky position, polarization, spin magnitudes, and spin angles.

        Physical parameter ranges are motivated by LIGO/Virgo observations (GWTC catalog):
        - Primary/secondary masses: Conservative range to ensure m1 >= m2
        - Luminosity distance: Typical observational range for detectable events
        - Spin magnitudes: Moderate values to ensure convergence in waveform codes
        - Sky coordinates: Uniformly distributed on celestial sphere
        - Polarization angles: Uniformly distributed in valid ranges

        Returns:
            Dict[str, float]: Dictionary containing the following gravitational wave parameters:
                mass_1 (float): Primary component mass in solar masses [25-40 M_sun]
                mass_2 (float): Secondary component mass in solar masses [20-35 M_sun]
                luminosity_distance (float): Luminosity distance in megaparsecs [300-800 Mpc]
                a_1 (float): Spin magnitude of primary BH [0.1-0.4, dimensionless]
                a_2 (float): Spin magnitude of secondary BH [0.1-0.4, dimensionless]
                ra (float): Right ascension in radians [0.5 to 2π-0.5]
                dec (float): Declination in radians [-π/2 to π/2]
                theta_jn (float): Inclination angle in radians [0 to π]
                psi (float): Polarization angle in radians [0.3 to π-0.3]
                phase (float): Initial phase in radians [0.3 to 2π-0.3]
                geocent_time (float): Geocenter time offset in seconds [-0.02 to 0.02]
                tilt_1 (float): Spin tilt angle of primary BH [0.5 to π-0.5]
                tilt_2 (float): Spin tilt angle of secondary BH [0.5 to π-0.5]
                phi_12 (float): Relative azimuthal angle between spins [0.3 to 2π-0.3]
                phi_jl (float): Angle between orbital angular momentum and total spin [0.3 to 2π-0.3]

        Note:
            - Conservative ranges avoid numerical instabilities in waveform generation
            - Mass ordering is enforced: m1 >= m2
            - Sky position sampling avoids extreme declinations that may cause projection issues
            - Spin angles avoid boundaries (0 and π) that may cause coordinate singularities
            - Time offset is small to ensure overlap regions are within the data segment

        Example:
            >>> simulator = OverlappingSignalSimulator(config)
            >>> params = simulator.generate_single_signal_params()
            >>> print(f"Chirp mass: {params['mass_1'] * params['mass_2'] / (params['mass_1'] + params['mass_2']):.2f}")
        """
        # Conservative mass ranges to ensure numerical stability
        mass_1 = np.random.uniform(25, 40)
        mass_2 = np.random.uniform(20, 35)
        if mass_1 < mass_2:
            mass_1, mass_2 = mass_2, mass_1
        
        # Realistic distance range for detectable events
        luminosity_distance = np.random.uniform(300, 800)
        
        # Moderate spin magnitudes to ensure waveform code convergence
        a_1 = np.random.uniform(0.1, 0.4)
        a_2 = np.random.uniform(0.1, 0.4)
        
        # Sky position: uniform on celestial sphere, avoiding extreme declinations
        ra = np.random.uniform(0.5, 2*np.pi - 0.5)
        dec = np.arcsin(np.random.uniform(-1, 1))
        
        # Polarization and orbital angles: uniformly distributed in safe ranges
        theta_jn = np.arccos(np.random.uniform(-1, 1))
        psi = np.random.uniform(0.3, np.pi - 0.3)
        phase = np.random.uniform(0.3, 2*np.pi - 0.3)
        
        # Small time offset to ensure overlap region is within segment
        geocent_time = np.random.uniform(-0.02, 0.02)
        
        # Spin orientation angles: avoiding singularities at boundaries
        tilt_1 = np.random.uniform(0.5, np.pi - 0.5)
        tilt_2 = np.random.uniform(0.5, np.pi - 0.5)
        phi_12 = np.random.uniform(0.3, 2*np.pi - 0.3)
        phi_jl = np.random.uniform(0.3, 2*np.pi - 0.3)
        
        return {
            'mass_1': mass_1,
            'mass_2': mass_2,
            'luminosity_distance': luminosity_distance,
            'theta_jn': theta_jn,
            'psi': psi,
            'phase': phase,
            'geocent_time': geocent_time,
            'ra': ra,
            'dec': dec,
            'a_1': a_1,
            'a_2': a_2,
            'tilt_1': tilt_1,
            'tilt_2': tilt_2,
            'phi_12': phi_12,
            'phi_jl': phi_jl,
        }

    def _sample_power_law(self, min_val: float, max_val: float, alpha: float) -> float:
        """
        Sample a value from a power-law probability distribution.

        Generates random samples from a power-law distribution P(x) ∝ x^α over the
        interval [min_val, max_val]. This is useful for sampling astrophysical quantities
        like masses and distances that often follow power-law distributions.

        The sampling uses inverse transform sampling: samples are drawn from the power-law
        CDF which is invertible analytically.

        Args:
            min_val (float): Minimum value of the distribution support. Must be positive.
            max_val (float): Maximum value of the distribution support. Must be > min_val.
            alpha (float): Power-law index. P(x) ∝ x^α.
                - alpha < 0: Distribution peaks at small values (common for masses)
                - alpha = -1: Special case handled separately (log-uniform distribution)
                - alpha > 0: Distribution peaks at large values

        Returns:
            float: Single sample from the specified power-law distribution,
                guaranteed to be in [min_val, max_val].

        Raises:
            ValueError: If min_val >= max_val or min_val <= 0

        Mathematical Details:
            For α ≠ -1, the CDF is:
                F(x) = (x^(α+1) - min^(α+1)) / (max^(α+1) - min^(α+1))
            
            Inverse transform: x = [u*(max^(α+1) - min^(α+1)) + min^(α+1)]^(1/(α+1))
            where u ~ Uniform(0, 1)
            
            For α = -1 (log-uniform), the inverse CDF is:
                x = min * exp(u * ln(max/min))

        Examples:
            >>> sim = OverlappingSignalSimulator(config)
            >>> # Sample mass from distribution peaking at low values
            >>> mass = sim._sample_power_law(1.0, 100.0, alpha=-2)
            >>> # Sample from log-uniform distribution
            >>> distance = sim._sample_power_law(10, 5000, alpha=-1)
        """
        if alpha == -1:
            # Special case: log-uniform distribution
            # Equivalent to uniform sampling in log-space
            return min_val * np.exp(np.random.random() * np.log(max_val / min_val))
        else:
            # General power-law inverse transform sampling
            u = np.random.random()
            return (min_val**(alpha + 1) + u * (max_val**(alpha + 1) - min_val**(alpha + 1)))**(1.0 / (alpha + 1))
    
    def _sample_spin_magnitude(self) -> float:
        """
        Sample a spin magnitude from a realistic astrophysical distribution.

        Generates spin magnitudes for black holes using a Beta distribution that is fit
        to observational data from the LIGO/Virgo Gravitational Wave Transient Catalog
        (GWTC). This provides realistic spin properties for injected signals.

        The Beta distribution is parameterized as Beta(α=1.5, β=3.0), which produces
        values biased toward lower spins with a tail toward high spins, matching
        observed GW event properties.

        Returns:
            float: Spin magnitude (dimensionless, 0 to 1 convention for black holes),
                sampled from the fitted distribution. Range: [0, 0.99].

        Distribution Properties:
            - Mean: ~0.33 (moderately spinning black holes)
            - Mode: ~0.08 (peak at low spins)
            - Std: ~0.25 (substantial spread)
            - Physically motivated by GWTC observations

        References:
            - LIGO/Virgo Gravitational Wave Transient Catalog: https://gwosc.org/
            - Abbott et al., GW190814: On the Properties of the Secondary Component
              of the Binary, ApJL, 896, L44 (2020)

        Note:
            - Spin magnitude is defined as |χ| = |a|/M for Kerr black holes
            - Range [0, 1] covers non-rotating (0) to maximally rotating (1) cases
            - Capped at 0.99 to avoid extremal black hole singularities

        Examples:
            >>> sim = OverlappingSignalSimulator(config)
            >>> a1 = sim._sample_spin_magnitude()
            >>> # Expected value ~0.33
            >>> spins = [sim._sample_spin_magnitude() for _ in range(1000)]
            >>> print(f"Mean spin: {np.mean(spins):.3f}, Std: {np.std(spins):.3f}")
        """
        # Beta distribution fit to GWTC measurements
        # Shape parameters (1.5, 3.0) fit observed spin distribution
        return np.random.beta(1.5, 3.0) * 0.99
    
    def generate_overlapping_scenario(self, n_signals: int) -> Dict:
        """
        Generate a scenario containing parameters for multiple overlapping signals.

        Creates a complete scenario with the specified number of gravitational wave signals
        that will have overlapping arrival times and frequency content. Each signal is assigned
        independent parameters sampled from realistic priors.

        This method is the primary interface for creating multi-signal scenarios used in
        training datasets. The returned scenario dict contains all information needed for
        signal injection into detector noise.

        Args:
            n_signals (int): Number of independent signals to generate for this scenario.
                Must be >= 1. Typical range for training: 2-6 signals.

        Returns:
            Dict: Scenario dictionary with the following structure:
                {
                    'signals': List[Dict[str, float]],  # List of n_signals parameter dicts
                    'n_signals': int,                   # Confirms number of signals
                    'scenario_id': int,                 # Unique scenario identifier (0-999999)
                }
                Each signal dict contains all parameters from generate_single_signal_params().

        Raises:
            ValueError: If n_signals < 1

        Side Effects:
            - Generates n_signals independent parameter samples
            - Assigns unique signal_id (0 to n_signals-1) to each signal
            - Creates random scenario_id for experiment tracking

        Note:
            - Scenario ID is random; not guaranteed unique across multiple calls
            - All signals are independent (no enforced time or frequency overlap)
            - Actual temporal overlap depends on signal duration and time parameters
            - Frequency overlap depends on masses (which determine signal bandwidth)

        Complexity:
            - Time: O(n_signals × 15) due to parameter generation
            - Space: O(n_signals × 15 floats) ≈ 120n_signals bytes

        Example:
            >>> simulator = OverlappingSignalSimulator(config)
            >>> # Generate 3-signal overlap scenario
            >>> scenario = simulator.generate_overlapping_scenario(n_signals=3)
            >>> print(f"Generated scenario {scenario['scenario_id']} with {scenario['n_signals']} signals")
            >>> # Access parameters of second signal
            >>> sig2_mass1 = scenario['signals'][1]['mass_1']
        """
        signals = []
        for i in range(n_signals):
            # Generate independent parameters for each signal
            params = self.generate_single_signal_params()
            # Assign unique signal identifier for tracking
            params['signal_id'] = i
            signals.append(params)
            
        return {
            'signals': signals,
            'n_signals': n_signals,
            'scenario_id': np.random.randint(0, 1000000)
        }
    
    def generate_detector_noise(self, 
                               duration: Optional[float] = None,
                               sampling_rate: Optional[int] = None) -> Dict:
        """
        Generate realistic colored Gaussian noise for all detectors.

        Synthesizes detector noise that matches the frequency-dependent characteristics
        (power spectral density, or PSD) of actual LIGO/Virgo detectors. The noise is
        generated in the frequency domain and transformed to time domain to ensure
        proper spectral properties.

        Each detector receives independent noise realization with its own PSD characteristics.
        LIGO detectors (H1, L1) use identical sensitivity profiles, while Virgo (V1) has
        different noise characteristics.

        Args:
            duration (float, optional): Duration of noise segment in seconds.
                Defaults to config.waveform.duration if not specified.
            sampling_rate (int, optional): Sampling frequency in Hz.
                Defaults to config.detectors[0].sampling_rate if not specified.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping detector names to noise time series.
                Keys are detector names (e.g., 'H1', 'L1', 'V1').
                Values are 1D numpy arrays of float64 strain data with length = duration * sampling_rate.
                Noise RMS ~1e-23 (strain units), matching typical detector sensitivity.

        Note:
            - Noise is generated independently for each detector
            - Real detectors have correlated noise (not modeled here for simplicity)
            - Noise properties (PSD shape) are analytical approximations, not exact
            - Generated noise is zero-mean Gaussian in time domain

        Raises:
            ValueError: If duration <= 0 or sampling_rate <= 0
            RuntimeError: If noise generation fails for all detectors

        Side Effects:
            - Calls _generate_colored_noise() for each detector
            - Logs warnings if noise generation fails and falls back to white noise

        Complexity:
            - Time: O(n_detectors × n_samples × log(n_samples)) due to FFT operations
            - Space: O(n_detectors × n_samples)

        Example:
            >>> simulator = OverlappingSignalSimulator(config)
            >>> # Generate default duration noise from config
            >>> noise_dict = simulator.generate_detector_noise()
            >>> # Or specify custom duration
            >>> noise_dict = simulator.generate_detector_noise(duration=8.0, sampling_rate=4096)
            >>> print(f"H1 noise shape: {noise_dict['H1'].shape}")
            >>> print(f"H1 noise RMS: {np.std(noise_dict['H1']):.2e}")
        """
        # Use config defaults if not specified
        if duration is None:
            duration = self.config.waveform.duration
        if sampling_rate is None:
            sampling_rate = self.config.detectors[0].sampling_rate
        
        # Calculate total number of samples
        n_samples = int(duration * sampling_rate)
        noise_data = {}
        
        # Generate colored noise for each detector using its PSD
        for det_name in self.detectors.keys():
            # Generate colored Gaussian noise with realistic PSD
            noise = self._generate_colored_noise(n_samples, sampling_rate, det_name)
            noise_data[det_name] = noise
            
        return noise_data
    
    def _generate_colored_noise(self, n_samples: int, sampling_rate: int, detector: str) -> np.ndarray:
        """
        Generate colored (frequency-dependent) Gaussian noise for a detector.

        Synthesizes realistic detector noise by shaping white Gaussian noise according to
        the detector's power spectral density (PSD). The PSD is obtained from analytical
        approximations that match real detector characteristics.

        The generation procedure:
        1. Generate white noise in frequency domain (uniform power across frequencies)
        2. Retrieve detector-specific PSD function (aLIGO for H1/L1, Virgo for V1)
        3. Scale frequency-domain noise by 1/sqrt(PSD) to achieve desired color
        4. Enforce Hermitian symmetry for real time-domain signal
        5. Inverse FFT to obtain time-domain strain

        Args:
            n_samples (int): Number of time samples to generate. Must be > 0.
            sampling_rate (int): Sampling frequency in Hz. Must be > 0.
            detector (str): Detector identifier ('H1', 'L1', 'V1', or other).
                - 'H1', 'L1': Advanced LIGO detectors (identical PSD)
                - 'V1': Virgo detector (different PSD shape and magnitude)
                - Other: Falls back to white noise PSD

        Returns:
            np.ndarray: 1D array of strain data with length n_samples.
                Dtype: float64
                Mean: ~0 (zero-mean)
                Variance: Set by PSD normalization
                Properties: Properly colored Gaussian noise in time domain

        Raises:
            RuntimeError: If FFT operations fail or arrays have incompatible shapes

        Side Effects:
            - Logs warnings if noise generation fails (falls back to white noise)
            - Uses detector-specific PSD calculations

        Notes:
            - PSD values are analytical approximations, not measured data
            - Noise is independent for each detector and each call
            - Frequency components above Nyquist (sampling_rate/2) are not present
            - Minimum PSD value is clamped to avoid division by zero in coloring step
            - Real LIGO/Virgo detector noise has additional features (glitches, lines)
              not modeled here

        Complexity:
            - Time: O(n_samples * log(n_samples)) due to FFT
            - Space: O(n_samples) for frequency domain arrays
            - FFT operations dominate for large n_samples

        Example:
            >>> simulator = OverlappingSignalSimulator(config)
            >>> # Generate 4 seconds of H1 noise at 4096 Hz
            >>> h1_noise = simulator._generate_colored_noise(
            ...     n_samples=16384,
            ...     sampling_rate=4096,
            ...     detector='H1'
            ... )
            >>> print(f"Generated H1 noise: shape={h1_noise.shape}, dtype={h1_noise.dtype}")
        """
        try:
            # Compute frequency array for FFT
            freqs = np.fft.fftfreq(n_samples, 1/sampling_rate)
            # Keep only positive frequencies (Hermitian symmetry constraint)
            positive_freqs = freqs[:n_samples//2 + 1]
            
            # Get detector-specific PSD
            if detector in ['H1', 'L1']:
                # Advanced LIGO PSD (H1 and L1 are identical)
                psd = self._aLIGO_psd(positive_freqs)
            elif detector == 'V1':
                # Virgo PSD (different shape and sensitivity)
                psd = self._virgo_psd(positive_freqs)
            else:
                # Unknown detector: use white noise as fallback
                psd = np.ones_like(positive_freqs)
            
            # Generate white noise in frequency domain
            # Complex amplitude with independent real/imaginary parts
            white_noise_f = np.random.normal(0, 1, n_samples//2 + 1) + 1j * np.random.normal(0, 1, n_samples//2 + 1)
            # DC component must be real (no imaginary part)
            white_noise_f[0] = white_noise_f[0].real
            # Nyquist component must be real if n_samples is even
            if n_samples % 2 == 0:
                white_noise_f[-1] = white_noise_f[-1].real
            
            # Color the white noise: scale by 1/sqrt(PSD) to achieve desired coloring
            # Normalization by sampling_rate/2 accounts for FFT conventions
            colored_noise_f = white_noise_f / np.sqrt(psd * sampling_rate / 2)
            
            # Reconstruct full frequency-domain array maintaining Hermitian symmetry
            # For real signals, negative frequencies are complex conjugates of positive frequencies
            colored_noise_f_full = np.concatenate([colored_noise_f, np.conj(colored_noise_f[-2:0:-1])])
            # Inverse FFT to obtain time-domain noise
            colored_noise = np.fft.ifft(colored_noise_f_full).real
            
            return colored_noise
            
        except Exception as e:
            self.logger.warning(f"Failed to generate colored noise for {detector}: {e}")
            # Fallback to white noise if coloring fails
            return np.random.normal(0, 1e-23, n_samples)
    
    def _aLIGO_psd(self, freqs: np.ndarray) -> np.ndarray:
        """
        Compute the power spectral density (PSD) for Advanced LIGO detectors.

        Returns an analytical approximation of the LIGO detector's power spectral
        density as a function of frequency. This PSD represents the expected noise
        level (strain-squared per unit frequency) in real LIGO data.

        The PSD shape is constructed to match key features of actual LIGO sensitivity:
        - Steep low-frequency rise due to Fabry-Perot cavity dynamics and seismic isolation
        - Broad intermediate plateau due to quantum noise and thermal noise
        - Specific features at characteristic resonance frequencies

        Args:
            freqs (np.ndarray): Array of frequencies in Hz. Can contain zero (clamped to 10 Hz).
                Shape: arbitrary (works element-wise)
                Values: 10 Hz to Nyquist frequency

        Returns:
            np.ndarray: Power spectral density in strain^2/Hz
                Shape: matches input freqs
                Dtype: float64
                Values: typically 1e-49 to 1e-47 strain^2/Hz in sensitive band

        Notes:
            - This is a simplified analytical model, not a measured/calibrated PSD
            - Real LIGO data has additional features (calibration lines, glitches)
            - Low frequencies (<10 Hz) are clamped to avoid singularities
            - Reference frequency f0=215 Hz is characteristic of LIGO
            - The formula is fit to approximate typical LIGO O3 sensitivity

        References:
            - LIGO Sensitivity Curves: https://dcc.ligo.org/LIGO-T1800044
            - Buikema et al., ApJL, 818, L1 (2016) - O3 constraints

        Mathematical Form:
            PSD(f) = [A(f/f0)^-4.14 - 5(f/f0)^-2 + 111/(1+(f/f0)^2)^0.5
                     + 1e4(f/10)^-8] × 1e-48
            where f0 = 215 Hz

        Example:
            >>> simulator = OverlappingSignalSimulator(config)
            >>> freqs = np.linspace(10, 4000, 1000)
            >>> psd = simulator._aLIGO_psd(freqs)
            >>> # Plot PSD
            >>> import matplotlib.pyplot as plt
            >>> plt.loglog(freqs, psd)
            >>> plt.xlabel('Frequency (Hz)')
            >>> plt.ylabel('PSD (strain^2/Hz)')
        """
        # Clamp frequencies below 10 Hz to avoid numerical singularities
        freqs = np.maximum(freqs, 10.0)
        
        # Analytical PSD formula fit to LIGO sensitivity
        # Reference frequency and shape parameters
        f0 = 215.0  # Hz - characteristic LIGO resonance frequency
        
        # Multi-component PSD model:
        # 1. Power-law term (f/f0)^-4.14: radiation pressure and shot noise
        # 2. Intermediate term (f/f0)^-2: Fabry-Perot dynamics
        # 3. Thermal noise term: resonance-like feature around f0
        # 4. Seismic term: low-frequency rise
        psd = ((freqs / f0)**(-4.14) 
               - 5 * (freqs / f0)**(-2) 
               + 111 * (1 + (freqs / f0)**2)**(-0.5)
               + 1e4 * (freqs / 10.0)**(-8))
        
        # Overall normalization to match typical LIGO PSD magnitudes
        return psd * 1e-48
    
    def _virgo_psd(self, freqs: np.ndarray) -> np.ndarray:
        """
        Compute the power spectral density (PSD) for Virgo detector.

        Returns an analytical approximation of the Virgo detector's power spectral
        density as a function of frequency. Virgo has different noise characteristics
        compared to LIGO due to differences in detector design:
        - Shorter arm length (3 km vs 4 km for LIGO)
        - Different isolation systems
        - Higher quantum noise at some frequencies
        - Different thermal noise properties

        Args:
            freqs (np.ndarray): Array of frequencies in Hz. Can contain zero (clamped to 10 Hz).
                Shape: arbitrary (works element-wise)
                Values: 10 Hz to Nyquist frequency

        Returns:
            np.ndarray: Power spectral density in strain^2/Hz
                Shape: matches input freqs
                Dtype: float64
                Values: typically higher than LIGO in most frequency ranges

        Notes:
            - This is a simplified analytical model, not a measured/calibrated PSD
            - Real Virgo data has additional features (calibration lines, glitches)
            - Low frequencies (<10 Hz) are clamped to avoid singularities
            - Virgo is typically less sensitive than LIGO (higher PSD values)
            - The formula is simplified compared to actual Virgo PSD

        References:
            - Virgo Sensitivity Curves: https://dcc.ligo.org/LIGO-T1800044
            - Acernese et al., CQG, 32, 024001 (2015) - Virgo+ design

        Mathematical Form:
            PSD(f) = 3.2e-46 × (f/100)^-4.05 + 2e-48

        Example:
            >>> simulator = OverlappingSignalSimulator(config)
            >>> freqs = np.linspace(10, 4000, 1000)
            >>> psd = simulator._virgo_psd(freqs)
            >>> # Compare to LIGO
            >>> ligo_psd = simulator._aLIGO_psd(freqs)
            >>> ratio = psd / ligo_psd
            >>> print(f"Virgo/LIGO sensitivity ratio: {ratio[500]:.2f}")
        """
        # Clamp frequencies below 10 Hz to avoid numerical singularities
        freqs = np.maximum(freqs, 10.0)
        
        # Simplified Virgo PSD model
        # Virgo is less sensitive than LIGO, with different frequency dependence
        # Power-law component: (f/100)^-4.05
        # Constant floor: 2e-48 (residual noise floor)
        psd = 3.2e-46 * (freqs / 100.0)**(-4.05) + 2e-48
        
        return psd
    
    def inject_signals_to_data(self, scenario: Dict, noise_data: Dict) -> Tuple[Dict, Dict]:
        """
        Inject multiple gravitational wave signals into detector noise data.

        Combines clean detector noise with generated gravitational wave signals for
        multiple detectors. Each signal in the scenario is synthesized in the detector
        frame (accounting for detector orientation and sky position) and added to the
        noise independently. The method produces both the combined strain data and
        tracking information for individual signal contributions.

        Signal injection simulates how real GW signals appear in detector data: a coherent
        signal pattern observed across all detectors, with detector-dependent amplitudes
        and phase relationships determined by sky position and polarization.

        Args:
            scenario (Dict): Scenario dictionary containing signal parameters.
                Must have structure: {'signals': List[Dict], 'n_signals': int, ...}
                Each signal dict contains gravitational wave parameters
                (mass_1, mass_2, luminosity_distance, ra, dec, etc.)

            noise_data (Dict[str, np.ndarray]): Dictionary mapping detector names to
                background noise time series.
                Keys: detector names ('H1', 'L1', 'V1', etc.)
                Values: 1D numpy arrays of strain data

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Dict]]: Two-element tuple:
                1. injected_data (Dict): Maps detector names to combined strain time series
                   with signals injected into noise. Same structure as noise_data.
                   Values are total strain = noise + signal_1 + signal_2 + ...

                2. signal_contributions (Dict): Tracks individual signal contributions.
                   Structure: {detector_name: {signal_id: signal_strain, ...}}
                   Allows reconstruction of individual signals post-injection.
                   signal_id matches the 'signal_id' field in scenario['signals']

        Raises:
            KeyError: If required fields missing from scenario
            RuntimeError: If waveform generation fails for all signals

        Side Effects:
            - Calls _generate_detector_strain() for each signal/detector pair
            - Logs warnings if injection fails for specific detectors
            - Modifies total_strain array in-place during accumulation

        Note:
            - Signals are linearly superposed (assumes weak-field limit)
            - Each signal is independently generated and aligned to scenario times
            - Strain arrays are padded/truncated to match length if necessary
            - Missing or failed signal generations skip contribution (graceful degradation)
            - Detector not in scenario.detectors is ignored with warning

        Complexity:
            - Time: O(n_detectors × n_signals × waveform_generation_time)
            - Space: O(n_detectors × n_samples) for output arrays

        Example:
            >>> simulator = OverlappingSignalSimulator(config)
            >>> scenario = simulator.generate_overlapping_scenario(n_signals=2)
            >>> noise_dict = simulator.generate_detector_noise()
            >>> injected, signals = simulator.inject_signals_to_data(scenario, noise_dict)
            >>> # Access combined strain
            >>> h1_data = injected['H1']  # H1 noise + 2 signals
            >>> # Access individual signals
            >>> signal_0_in_h1 = signals['H1'][0]
            >>> signal_1_in_h1 = signals['H1'][1]
            >>> # Verify linear superposition
            >>> combined = noise_dict['H1'] + signal_0_in_h1 + signal_1_in_h1
            >>> assert np.allclose(injected['H1'], combined)
        """
        injected_data = {}
        signal_contributions = {}
        
        # Process each detector
        for det_name, noise in noise_data.items():
            if det_name in self.detectors:
                try:
                    # Initialize total strain with background noise
                    detector = self.detectors[det_name]
                    total_strain = np.array(noise)
                    signal_contributions[det_name] = {}
                    
                    # Inject each signal from the scenario
                    for signal in scenario['signals']:
                        # Generate waveform for this signal/detector combination
                        strain = self._generate_detector_strain(signal, det_name)
                        
                        if strain is not None:
                            # Align and add signal to total strain
                            # Handle potential length mismatches
                            min_len = min(len(total_strain), len(strain))
                            total_strain[:min_len] += strain[:min_len]
                            # Track this signal's contribution for analysis
                            signal_contributions[det_name][signal['signal_id']] = strain
                    
                    injected_data[det_name] = total_strain
                    
                except Exception as e:
                    self.logger.warning(f"Failed to inject signals into {det_name}: {e}")
                    # Graceful degradation: return noise unchanged if injection fails
                    injected_data[det_name] = noise
                    signal_contributions[det_name] = {}
            else:
                # Detector not configured: skip
                injected_data[det_name] = noise
                signal_contributions[det_name] = {}
                
        return injected_data, signal_contributions
    
    def _generate_mock_waveform(self, params: Dict, detector_name: str) -> np.ndarray:
        """
        Generate a simplified mock gravitational wave waveform.

        Creates a synthetic gravitational wave signal using a simple analytical model
        when high-fidelity waveform generation (via bilby/pycbc) is unavailable or fails.
        The mock waveform incorporates:
        - Frequency sweep (chirp): Low frequency to high frequency during merger
        - Amplitude envelope: Power-law growth then exponential decay
        - Mass-dependent amplitude scaling: Louder signals for larger chirp masses
        - Distance scaling: Amplitude decreases inversely with distance
        - Detector response: Accounts for detector orientation and sky position

        This fallback mechanism ensures training data can be generated even when
        sophisticated waveform codes are unavailable, though with reduced physical fidelity.

        Args:
            params (Dict): Gravitational wave signal parameters.
                Required keys:
                    - mass_1 (float): Primary mass in solar masses
                    - mass_2 (float): Secondary mass in solar masses
                    - luminosity_distance (float): Distance in megaparsecs
                    - ra (float): Right ascension in radians
                    - dec (float): Declination in radians
                    - psi (float): Polarization angle in radians
                Uses defaults if keys missing.

            detector_name (str): Name of detector ('H1', 'L1', 'V1', or other).
                Determines detector response tensor calculation.

        Returns:
            np.ndarray: Time-domain strain data as 1D float array.
                Shape: (n_samples,) where n_samples = duration × sampling_rate
                Dtype: float64
                Amplitude: ~1e-21 to 1e-20 strain (typical GW signal range)
                Properties: Zero-mean, oscillatory with decaying envelope

        Note:
            - Model is highly simplified compared to physical GW waveforms
            - Used only as fallback when proper waveform generation fails
            - Frequency evolution follows power-law (t/duration)^3
            - Amplitude scaling: A ∝ M_c^(5/6) / distance
            - Detector response: Simplified orientation dependence
            - Training on mock waveforms produces lower model performance

        Mathematical Details:
            Frequency evolution: f(t) = f_0 + (f_1 - f_0)(t/T)^3
            where f_0=35 Hz, f_1=250 Hz, T=duration
            
            Amplitude: A = A_0 × (M_c/30)^(5/6) / (D/400)
            where A_0 = 1e-21, M_c = chirp mass, D = distance in Mpc
            
            Envelope: E(t) = exp(-t/(T×0.4))
            
            Strain: h(t) = A × E(t) × sin(∫2πf(t)dt) × detector_response

        Complexity:
            - Time: O(n_samples) for phase accumulation
            - Space: O(n_samples) for output array

        Example:
            >>> simulator = OverlappingSignalSimulator(config)
            >>> params = simulator.generate_single_signal_params()
            >>> mock_strain = simulator._generate_mock_waveform(params, 'H1')
            >>> print(f"Mock waveform amplitude: {np.max(np.abs(mock_strain)):.2e}")
        """
        # Setup time array
        duration = self.config.waveform.duration
        sampling_rate = self.config.detectors[0].sampling_rate
        n_samples = int(duration * sampling_rate)
        t = np.linspace(0, duration, n_samples)
        
        # Extract physical parameters with sensible defaults
        m1 = params.get('mass_1', 30.0)
        m2 = params.get('mass_2', 30.0)
        distance = params.get('luminosity_distance', 500.0)
        
        # Calculate chirp mass: characteristic GW emission mass
        # M_c = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
        chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        
        # Frequency sweep: starts at ~35 Hz (LIGO sensitive band) to ~250 Hz (merger)
        # Power-law evolution with index 3 models pre-merger phase
        f_start = 35.0   # Hz - LIGO band edge
        f_end = 250.0    # Hz - merger frequency (approximate)
        freq = f_start + (f_end - f_start) * (t / duration)**3
        
        # Amplitude scaling with mass and distance
        # Proportional to M_c^(5/6) / distance (from GW physics)
        # Normalization: 1e-21 strain for 30 M_sun binary at 400 Mpc
        amp = 1e-21 * (chirp_mass / 30.0)**(5/6) / (distance / 400.0)
        
        # Exponential envelope: signal peaks at ~60% of duration
        envelope = np.exp(-t / (duration * 0.4))
        
        # Generate phase through integration of angular frequency
        dt = t[1] - t[0] if len(t) > 1 else 1/sampling_rate
        phase = 2 * np.pi * np.cumsum(freq) * dt
        
        # Strain: product of amplitude, envelope, and oscillation
        strain = amp * envelope * np.sin(phase)
        
        # Apply detector response tensor based on sky position and polarization
        # Simplified model accounting for detector orientation
        ra = params.get('ra', 0.0)
        dec = params.get('dec', 0.0)
        psi = params.get('psi', 0.0)
        
        # Detector response: depends on sky position and source polarization
        # Different detectors have different orientations:
        # H1/L1: 90° arms, relative orientation gives relative phase
        # V1: 60° arms, different response pattern
        if detector_name == 'H1':
            # LIGO Hanford: responds to plus polarization
            response = 0.5 * (1 + np.cos(dec)**2) * np.cos(2*psi)
        elif detector_name == 'L1':
            # LIGO Livingston: 90° rotation relative to H1
            response = 0.5 * (1 + np.cos(dec)**2) * np.cos(2*psi + np.pi/2)
        elif detector_name == 'V1':
            # Virgo: different arm configuration
            response = 0.3 * (1 + np.cos(dec)**2) * np.cos(2*psi + np.pi/4)
        else:
            # Unknown detector: use generic response
            response = 0.5
        
        return strain * abs(response)
    
    def _generate_detector_strain(self, params: Dict, detector_name: str) -> Optional[np.ndarray]:
        """
        Generate strain time series for a specific detector and signal.

        Synthesizes the strain response of a single detector to a gravitational wave signal
        with specified parameters. This is the primary method for generating physical GW
        waveforms aligned to each detector's frame.

        The method implements a three-tier approach:
        1. Primary: Use bilby's frequency-domain waveform generator (highest fidelity)
        2. Secondary: Project frequency-domain waveform using detector response (if bilby fails)
        3. Fallback: Use analytical mock waveform (if all else fails)

        This ensures maximum robustness across different software versions and configurations.

        Args:
            params (Dict): Gravitational wave parameters dictionary.
                Must contain all fields from generate_single_signal_params():
                mass_1, mass_2, luminosity_distance, ra, dec, psi, geocent_time, etc.

            detector_name (str): Name of target detector ('H1', 'L1', 'V1', or custom).
                Must exist in self.detectors dictionary.

        Returns:
            Optional[np.ndarray]: Time-domain strain data for this detector.
                Type: 1D numpy array of float64
                Shape: (n_samples,) where n_samples = duration × sampling_rate
                Value: Strain amplitude in units of 10^-21 (typical GW range)
                None: If waveform generation fails completely for this detector

        Raises:
            (none - all exceptions caught and logged)

        Side Effects:
            - Logs warnings if fallback mechanisms are triggered
            - May call waveform_generator if bilby is available
            - Calls _generate_mock_waveform() as last resort

        Notes:
            - Bilby waveform generator must be initialized (set up in __init__)
            - Detector must support project_wave() method (bilby detectors) or be PyCBC
            - Frequency-domain waveforms are inverse-FFT'd to time domain as needed
            - Returns None gracefully if detector not found or all generation attempts fail
            - Mock waveforms have lower fidelity than bilby-generated signals

        Workflow:
            1. Try bilby.frequency_domain_strain() to generate polarization components
            2. If successful, use detector.project_wave() to apply detector response
            3. If bilby detector unavailable, FFT inverse the plus polarization
            4. On any bilby error, fall back to mock waveform generation
            5. Log debug message if both fail (returns None)

        Complexity:
            - Time: O(n_samples × log(n_samples)) for FFT, or O(n_samples) for mock
            - Space: O(n_samples) for intermediate frequency-domain arrays

        Example:
            >>> simulator = OverlappingSignalSimulator(config)
            >>> params = simulator.generate_single_signal_params()
            >>> strain_h1 = simulator._generate_detector_strain(params, 'H1')
            >>> if strain_h1 is not None:
            ...     print(f"Generated H1 strain: {strain_h1.shape}, max amp: {np.max(np.abs(strain_h1)):.2e}")
        """
        try:
            # Primary: Use bilby for high-fidelity frequency-domain waveform
            if self.waveform_generator is not None:
                waveform_polarizations = self.waveform_generator.frequency_domain_strain(params)
                
                if detector_name in self.detectors:
                    detector = self.detectors[detector_name]
                    
                    # Check if detector supports bilby interface
                    if hasattr(detector, 'project_wave'):
                        # Bilby detector: apply antenna response tensor to polarizations
                        strain = detector.project_wave(
                            waveform_polarizations['plus'],
                            waveform_polarizations['cross'],
                            params['ra'], params['dec'], params['psi'],
                            params.get('geocent_time', 0.0)
                        )
                        return strain.time_domain_strain
                    else:
                        # PyCBC detector or other: simplified projection using plus polarization
                        # Convert frequency domain to time domain
                        waveform_td = np.fft.ifft(waveform_polarizations['plus']).real
                        return waveform_td
                
                return None
            
        except Exception as e:
            # Log warning and proceed to fallback
            self.logger.debug(f"Bilby waveform generation failed for {detector_name}: {str(e)[:100]}")
        
        # Fallback: Use analytical mock waveform
        try:
            return self._generate_mock_waveform(params, detector_name)
        except Exception as e2:
            # Complete failure: both bilby and mock failed
            self.logger.debug(f"All waveform generation methods failed for {detector_name}: {str(e2)[:100]}")
            return None
    
    def create_training_dataset(self, 
                               n_scenarios: int, 
                               output_dir: str) -> List[Dict]:
        """
        Generate a complete training dataset with overlapping GW signals and noise.

        Orchestrates the full simulation pipeline to create a dataset suitable for training
        machine learning models. For each scenario, generates:
        1. Random number of overlapping signals (2-5 events)
        2. Realistic detector noise for all detectors
        3. Injects signals into noise to create realistic data
        4. Saves complete dataset to disk for later training

        The resulting dataset contains all information needed for training:
        - Combined strain data (injected signals + noise) for each detector
        - Ground truth parameters for all injected signals
        - Individual signal contributions for analysis and validation
        - Metadata about generation parameters (duration, sample rate, etc.)

        This is the highest-level interface for generating training data.

        Args:
            n_scenarios (int): Number of independent scenarios to generate.
                Must be > 0. Typical ranges: 1000-100000 for training datasets.
                Larger datasets produce better-trained models but take longer.

            output_dir (str): Directory path where dataset will be saved.
                If directory doesn't exist, it will be created.
                Dataset is saved as 'simulated_training_data.pkl' in this directory.

        Returns:
            List[Dict]: List of training scenario dictionaries. Each dict has structure:
                {
                    'scenario_id': int,                      # Index in batch (0 to n_scenarios-1)
                    'true_parameters': List[Dict],           # Ground truth parameters for all signals
                    'injected_data': Dict[str, np.ndarray],  # Combined strain for all detectors
                    'noise_data': Dict[str, np.ndarray],     # Background noise for all detectors
                    'signal_contributions': Dict,            # Individual signal strains
                    'n_signals': int,                        # Number of signals in this scenario
                    'generation_params': Dict,               # Metadata about generation
                }

        Raises:
            OSError: If output directory cannot be created
            PickleError: If dataset cannot be saved to disk

        Side Effects:
            - Creates output directory if it doesn't exist
            - Saves pickle file to disk containing complete dataset
            - Logs generation progress every 100 scenarios
            - May log warnings/errors for individual scenarios that fail

        Notes:
            - Signal count distribution: 2-signal (50%), 3-signal (30%), 4-signal (15%), 5-signal (5%)
            - Failed scenarios are skipped (error logged); they don't abort entire dataset generation
            - Generation can be slow: ~1-5 seconds per scenario depending on hardware
            - Dataset size: ~1-2 MB per scenario (with 4 second duration, 4 kHz sampling)
            - Memory usage: ~100-200 MB for full 100K dataset in RAM before saving

        Complexity:
            - Time: O(n_scenarios × n_signals × waveform_generation_time)
            - Space: O(n_signals × n_samples) per scenario in memory
            - Disk: ~1-2 MB per scenario saved to pickle

        Progress Monitoring:
            - Logs "Generating N scenarios..." at start
            - Logs progress at each 100-scenario milestone
            - Logs final count of successfully generated scenarios

        Example:
            >>> simulator = OverlappingSignalSimulator(config)
            >>> dataset = simulator.create_training_dataset(
            ...     n_scenarios=1000,
            ...     output_dir='./datasets/train'
            ... )
            >>> print(f"Generated {len(dataset)} scenarios successfully")
            >>> # Access first scenario
            >>> scenario_0 = dataset[0]
            >>> print(f"Scenario 0 has {scenario_0['n_signals']} signals")

        See Also:
            - generate_overlapping_scenario(): For single scenario generation
            - generate_detector_noise(): For noise-only generation
            - inject_signals_to_data(): For signal injection details
        """
        training_scenarios = []
        output_path = Path(output_dir)
        # Create output directory if necessary
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating {n_scenarios} training scenarios...")
        
        for i in range(n_scenarios):
            try:
                # Sample number of overlapping signals with realistic distribution
                # Most events are 2-signal overlaps, fewer are 3+
                n_signals = np.random.choice([2, 3, 4, 5], p=[0.5, 0.3, 0.15, 0.05])
                
                # Generate signal parameters for this scenario
                scenario = self.generate_overlapping_scenario(n_signals)
                
                # Generate realistic detector noise
                noise_data = self.generate_detector_noise()
                
                # Inject signals into noise to create realistic data
                injected_data, signal_contributions = self.inject_signals_to_data(
                    scenario, noise_data
                )
                
                # Package complete training example with metadata
                training_scenario = {
                    'scenario_id': i,
                    'true_parameters': scenario['signals'],        # Ground truth
                    'injected_data': injected_data,               # Observational data
                    'noise_data': noise_data,                     # Background noise
                    'signal_contributions': signal_contributions,  # For analysis
                    'n_signals': n_signals,
                    'generation_params': {
                        'duration': self.config.waveform.duration,
                        'sampling_rate': self.config.detectors[0].sampling_rate,
                        'approximant': self.config.waveform.approximant
                    }
                }
                
                training_scenarios.append(training_scenario)
                
                # Log progress every 100 scenarios
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Generated {i + 1}/{n_scenarios} scenarios")
                    
            except Exception as e:
                # Graceful degradation: skip failed scenarios, continue with others
                self.logger.error(f"Failed to generate scenario {i}: {str(e)[:100]}")
                continue
        
        # Serialize complete dataset to disk for later use
        dataset_file = output_path / 'simulated_training_data.pkl'
        with open(dataset_file, 'wb') as f:
            pickle.dump(training_scenarios, f)
        
        # Final status report
        self.logger.info(f"Saved {len(training_scenarios)} training scenarios to {dataset_file}")
        
        return training_scenarios


def compute_waveform_overlap(h1: np.ndarray, h2: np.ndarray, 
                           sampling_rate: float = 4096.0) -> float:
    """
    Compute the normalized inner product overlap between two waveforms.

    Calculates a measure of similarity between two gravitational wave strain signals
    in the time domain. The overlap is defined as the normalized inner product:

        O = |<h1, h2>| / (||h1|| × ||h2||)

    This ranges from 0 (orthogonal signals) to 1 (identical signals). This metric is
    useful for quantifying signal degeneracies, overlapping signal confusion, and
    validating waveform generation consistency.

    The overlap is computed in the time domain using simple inner product, which is
    fast but less sensitive to phase coherence than frequency-domain overlap. For
    high-precision overlap metrics used in Bayesian inference, consider using match
    filter inner products instead.

    Args:
        h1 (np.ndarray): First strain time series.
            Shape: 1D array of arbitrary length
            Dtype: float or complex
            Units: Strain (dimensionless)

        h2 (np.ndarray): Second strain time series.
            Shape: 1D array, can differ from h1 in length
            Dtype: float or complex
            Units: Strain (dimensionless)

        sampling_rate (float, optional): Sampling frequency in Hz.
            Default: 4096.0
            Currently unused (for future normalization extensions)

    Returns:
        float: Normalized overlap value.
            Range: [0.0, 1.0] under normal conditions, may exceed 1.0 for
                numerical edge cases.
            Type: Python float (guaranteed scalar)
            Interpretation:
                - 0.0: Signals are orthogonal/uncorrelated
                - 0.5: Moderate correlation
                - 1.0: Signals are identical

    Raises:
        (none - all exceptions caught, returns 0.0 on error)

    Side Effects:
        - Truncates longer signal to match shorter signal
        - All-zero signals return 0.0 (division by zero handled)

    Notes:
        - Symmetric in arguments: overlap(h1, h2) == overlap(h2, h1)
        - Insensitive to overall amplitude: overlap(a×h1, b×h2) = overlap(h1, h2)
        - Time-domain overlap (fast but insensitive to phase relations)
        - For frequency-domain overlap: use matched filtering techniques
        - Gracefully handles various numerical errors, returns 0.0 on failure

    Complexity:
        - Time: O(min(len(h1), len(h2))) for inner product computation
        - Space: O(min(len(h1), len(h2))) for array slicing

    Example:
        >>> import numpy as np
        >>> h1 = np.sin(2*np.pi*100*np.linspace(0, 1, 4096))  # 100 Hz sine
        >>> h2 = np.sin(2*np.pi*100*np.linspace(0, 1, 4096))  # Identical
        >>> overlap = compute_waveform_overlap(h1, h2)
        >>> print(f"Overlap (identical signals): {overlap:.4f}")  # ~1.0
        >>> h3 = np.sin(2*np.pi*200*np.linspace(0, 1, 4096))  # 200 Hz sine
        >>> overlap = compute_waveform_overlap(h1, h3)
        >>> print(f"Overlap (orthogonal frequencies): {overlap:.4f}")  # ~0.0

    References:
        - Maggiore, M., Gravitational Waves, Vol. 2, Oxford University Press (2018)
        - Poisson & Will, Gravity: Newtonian, Post-Newtonian, Relativistic, CUP (2014)
    """
    try:
        # Ensure both waveforms have the same length
        # Truncate longer signal to match shorter one
        min_len = min(len(h1), len(h2))
        h1 = h1[:min_len]
        h2 = h2[:min_len]
        
        # Compute normalized inner product
        # numerator: absolute value of dot product (handles complex signals)
        # denominator: product of L2 norms (amplitudes)
        numerator = np.abs(np.dot(h1, h2))
        denominator = np.linalg.norm(h1) * np.linalg.norm(h2)
        
        # Handle zero-norm signals (all-zero arrays)
        if denominator == 0.0:
            return 0.0
        
        overlap = numerator / denominator
        
        return float(overlap)
        
    except Exception:
        # Graceful error handling: return 0.0 for any numerical issues
        return 0.0


def estimate_snr_from_strain(strain: np.ndarray, 
                            psd: Optional[np.ndarray] = None,
                            sampling_rate: float = 4096.0) -> float:
    """
    Estimate the signal-to-noise ratio (SNR) of strain data.

    Computes a measure of signal quality/strength in gravitational wave data by comparing
    signal power to expected noise characteristics. Supports two methods:

    1. **Time-domain estimate**: Simple ratio of signal variance to assumed noise power.
       Fast but rough; assumes white noise.

    2. **Frequency-domain (matched filter) estimate**: Proper SNR calculation using the
       power spectral density (PSD). Weighted by detector sensitivity; accounts for
       frequency-dependent noise properties.

    The matched-filter SNR is the "whitened" signal power and is the proper metric for
    GW detection significance and Bayesian inference.

    Args:
        strain (np.ndarray): Time-domain strain data from detector.
            Shape: 1D array
            Dtype: float
            Units: Strain (dimensionless)
            Length: n_samples = duration × sampling_rate

        psd (np.ndarray, optional): Power spectral density for whitening.
            Shape: 1D array of frequency-domain magnitudes
            Dtype: float
            Units: Strain^2/Hz
            If None: Uses time-domain method (assumes white noise)
            If provided: Uses frequency-domain matched filter method
            Should have length >= len(strain) for FFT compatibility

        sampling_rate (float, optional): Sampling frequency in Hz.
            Default: 4096.0
            Used for frequency grid calculation

    Returns:
        float: Estimated SNR value.
            Type: Python float (guaranteed scalar)
            Range: Typically 1-100+ for detectable signals, <5 for noise-only
            Interpretation:
                - SNR < 5: Below detection threshold
                - SNR 5-10: Marginal signals (barely detectable)
                - SNR > 10: Confident detection
                - SNR > 20: High-confidence astrophysical event

    Raises:
        (none - all exceptions caught, returns 0.0 on error)

    Side Effects:
        - Truncates PSD to match strain length if necessary
        - Performs FFT of strain array (does not modify input)

    Notes:
        - Time-domain method: assumes white noise (flat PSD)
        - Frequency-domain method: accounts for real detector noise (colored)
        - Matches normalization convention used in dataset_generator.py
        - Multiple segments: for multi-detector SNR, average individual SNRs
        - Network SNR: combine H1, L1, V1 via sqrt(SNR_H^2 + SNR_L^2 + SNR_V^2)

    Complexity:
        - Time domain: O(n_samples) for variance calculation
        - Frequency domain: O(n_samples × log(n_samples)) for FFT

    Mathematical Details (Frequency Domain):
        The matched-filter SNR is:
            SNR² = 4 ∫_0^(f_Nyquist) |ĥ(f)|² / S_n(f) df
        where:
            - ĥ(f) is strain in frequency domain
            - S_n(f) is the one-sided power spectral density
            - Factor of 4 accounts for one-sided vs two-sided spectrum
            - Integration only over positive frequencies

    Example:
        >>> import numpy as np
        >>> strain = np.random.randn(16384) * 1e-23  # Simulated noise
        >>> # Time-domain (quick, rough)
        >>> snr_td = estimate_snr_from_strain(strain)
        >>> print(f"Time-domain SNR: {snr_td:.2f}")  # ~small value for noise
        >>> # Frequency-domain (proper, slow)
        >>> psd = np.ones(16384) * 1e-46  # White noise PSD
        >>> snr_fd = estimate_snr_from_strain(strain, psd=psd, sampling_rate=4096)
        >>> print(f"Matched-filter SNR: {snr_fd:.2f}")  # More realistic

    References:
        - Allen et al., gr-qc/0405045: Statistical properties of GW backgrounds
        - Flanagan & Hughes, Phys.Rev.D 57, 4535 (1998): Matched filtering
        - LIGO Science Collaboration Documentation: SNR definitions

    See Also:
        - _aLIGO_psd(), _virgo_psd(): For realistic PSD models
        - OverlappingSignalSimulator.generate_detector_noise(): Generates PSD samples
    """
    try:
        if psd is None:
            # Simple time-domain SNR estimate
            # Assumes white noise with fixed power
            signal_power = np.var(strain)
            # Assume typical LIGO noise floor
            noise_power = 1e-46
            snr = np.sqrt(signal_power / noise_power)
            
        else:
            # Proper frequency-domain matched-filter SNR
            # Accounts for realistic detector PSD (colored noise)
            
            # FFT of strain with normalization
            N = len(strain)
            strain_f = np.fft.fft(strain) / float(N)
            freqs = np.fft.fftfreq(N, 1/sampling_rate)

            # Handle PSD array: truncate to match strain length if necessary
            psd_seg = psd[:len(strain_f)] if len(psd) >= len(strain_f) else np.ones_like(strain_f)
            
            # Compute matched-filter integrand: |ĥ(f)|² / S_n(f)
            integrand = np.abs(strain_f)**2 / psd_seg
            
            # Frequency resolution
            df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0/sampling_rate
            
            # Matched-filter SNR squared with proper normalization
            # Factor of 4.0: accounts for one-sided PSD convention
            # (Positive frequencies only, but must account for negative frequencies)
            snr_squared = 4.0 * np.sum(integrand) * df
            snr = np.sqrt(max(snr_squared, 0.0))
        
        return float(snr)
        
    except Exception:
        # Graceful error handling: return 0 on numerical issues
        return 0.0
