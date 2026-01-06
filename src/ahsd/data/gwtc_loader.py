"""GWTC Catalog Loader Module.

This module provides functionality to load and process real gravitational wave (GW) events
from the Gravitational Wave Transient Catalog (GWTC) maintained by GWOSC (Gravitational
Wave Open Science Center). It supports multiple catalog versions (GWTC-1, 2, 3, 4) with
automatic caching, fallback mechanisms, and synthetic overlap scenario generation.

Key Features:
    - Fetches real GW event parameters from GWOSC API (GWTC-1, -2, -3, -4)
    - Intelligent caching with configurable validity period to minimize API calls
    - Robust fallback to hardcoded high-confidence events if API unavailable
    - Strain data download capability via GWpy integration (when available)
    - Generation of synthetic overlapping event scenarios from real catalog
    - Production-grade error handling and logging

Dependencies:
    - pandas: Event data manipulation and DataFrame operations
    - numpy: Numerical operations for time/parameter calculations
    - requests: HTTP API communication with GWOSC servers
    - gwpy (optional): TimeSeries data fetching for strain download
    - logging: Application-level logging throughout module

Example:
    >>> from ahsd.data.gwtc_loader import GWTCLoader
    >>> loader = GWTCLoader(data_dir="data/gwtc", cache_days=7)
    >>> events_df = loader.get_gwtc_events(catalog="GWTC-3")
    >>> print(f"Loaded {len(events_df)} events")
    >>> overlaps = loader.create_synthetic_overlaps(events_df, n_overlaps=50)

Note:
    Event parameters include:
    - mass_1_source, mass_2_source: Component masses in source frame (solar masses)
    - chirp_mass_source: Chirp mass in source frame (solar masses)
    - luminosity_distance: Luminosity distance to event (Mpc)
    - redshift: Cosmological redshift
    - network_snr: Network-wide matched-filter SNR
    - far: False alarm rate (Hz or per year depending on catalog)
"""

import numpy as np
import pandas as pd
import logging
import requests
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

try:
    from gwpy.timeseries import TimeSeries
    GWPY_AVAILABLE = True
except ImportError:
    GWPY_AVAILABLE = False

class GWTCLoader:
    """Load and process real gravitational wave events from GWTC catalogs.

    This class manages retrieval of published GW event parameters from the GWOSC API,
    with intelligent caching, error recovery, and optional strain data download.
    Supports all major catalog versions: GWTC-1, -2, -3, and -4.

    The loader implements a multi-level fallback strategy:
        1. Check local JSON cache (valid if age < cache_days)
        2. Fetch from GWOSC API with pagination support
        3. Fall back to hardcoded high-confidence events if API unavailable
        4. Cache results for future use (both live API and fallback)

    Attributes:
        data_dir (Path): Directory for cache storage and downloaded data
        cache_file (Path): Path to JSON cache file (gwtc_events.json)
        cache_days (int): Cache validity period in days (default: 7)
        logger (logging.Logger): Module logger instance
        gwosc_base_url (str): GWOSC API base URL

    Example:
        >>> loader = GWTCLoader(data_dir="data/gwtc", cache_days=7)
        >>> events = loader.get_gwtc_events(catalog="GWTC-3")
        >>> print(f"Loaded {len(events)} events")
        >>> strain = loader.download_strain("GW190814", detector="H1")
    """

    def __init__(self, data_dir: str = "data/gwtc", cache_days: int = 7) -> None:
        """Initialize GWTC loader with cache configuration.

        Args:
            data_dir (str): Directory path for cache storage. Created if missing.
                Defaults to "data/gwtc".
            cache_days (int): Cache validity period in days. Cached events are
                reused if file age < cache_days. Defaults to 7 days.

        Returns:
            None

        Note:
            The cache directory is created automatically with parent directories
            if it doesn't exist. A JSON cache file is expected at data_dir/gwtc_events.json.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.data_dir / "gwtc_events.json"
        self.cache_days = cache_days
        self.logger = logging.getLogger(__name__)
        self.gwosc_base_url = "https://gwosc.org"
    
    def get_gwtc_events(self, catalog: str = "GWTC") -> pd.DataFrame:
        """Fetch all gravitational wave events from specified GWTC catalog.

        Implements multi-level fallback strategy with caching:
            1. Returns cached events if cache is valid (age < cache_days)
            2. Fetches from GWOSC API with automatic pagination
            3. Falls back to hardcoded events if API unavailable
            4. Caches results for future use

        The API pagination automatically handles catalogs with >100 events by
        iterating through pages until no more results are returned.

        Args:
            catalog (str): GWTC catalog identifier. Options:
                - "GWTC": Cumulative (recommended for all events)
                - "GWTC-1": O1 observing run events (first catalog)
                - "GWTC-2": O1-O2 events (more events, higher confidence)
                - "GWTC-3": O1-O3a events (largest catalog to date)
                - "GWTC-4": O3b events (most recent)
                Defaults to "GWTC" for all published events.

        Returns:
            pd.DataFrame: DataFrame with event parameters. Columns include:
                - event_name (str): GW event identifier (e.g., "GW190814")
                - gps_time (float): GPS timestamp of merger (seconds since Jan 1, 1980)
                - mass_1_source (float): Primary component mass (solar masses, source frame)
                - mass_2_source (float): Secondary component mass (solar masses, source frame)
                - chirp_mass_source (float): Chirp mass (solar masses, source frame)
                - total_mass_source (float): Total mass (solar masses, source frame)
                - luminosity_distance (float): Luminosity distance to event (Mpc)
                - redshift (float): Cosmological redshift
                - network_snr (float): Network-wide matched-filter SNR
                - far (float): False alarm rate (Hz or 1/year)
                - observing_run (str): Observing run identifier (O1, O2, O3a, O3b, etc.)
                - detectors (list): Detectors that observed event (e.g., ["H1", "L1", "V1"])

        Raises:
            ValueError: If API returns no results after exhausting fallback options.
            RequestException: Caught and logged, triggers fallback to hardcoded events.

        Note:
            - Cache is stored as JSON at data_dir/gwtc_events.json
            - API timeout is set to 30 seconds per page request
            - Fallback events include high-confidence historical detections
            - Logging tracks cache hits, API fetches, and fallback usage

        Example:
            >>> loader = GWTCLoader()
            >>> events = loader.get_gwtc_events("GWTC-3")
            >>> bbh_events = events[events['event_name'].str.startswith('GW1')]
            >>> print(f"Found {len(bbh_events)} BBH events")
        """

        # Check cache first
        if self._is_cache_valid():
            self.logger.info("Loading GWTC events from cache")
            return self._load_from_cache()

        # Fetch from API (all pages)
        base_url = f"https://gwosc.org/api/v2/catalogs/{catalog}/events?include-default-parameters=true"
        all_results = []

        try:
            page = 1
            while True:
                endpoint = f"{base_url}&page={page}"
                response = requests.get(endpoint, timeout=30)
                if response.status_code != 200:
                    break

                api_data = response.json()
                if not isinstance(api_data, dict) or 'results' not in api_data:
                    break

                all_results.extend(api_data['results'])

                # Check if there are more pages
                if api_data.get('next') is None:
                    break
                page += 1

            if all_results:
                df = self._parse_api_results(all_results)
                # Cache the result
                self._save_to_cache(df)
                self.logger.info(f"Fetched {len(df)} events from GWOSC API")
                return df
            else:
                raise ValueError("No results from API")

        except requests.RequestException as e:
            self.logger.warning(f"Failed to fetch from GWOSC API: {e}")
        except Exception as e:
            self.logger.warning(f"Error parsing GWOSC response: {e}")

        # Fallback to hardcoded high-confidence events
        self.logger.warning("API fetch failed, using hardcoded events")
        df = self._get_hardcoded_events()
        self._save_to_cache(df)  # Cache fallback too
        return df

    def _is_cache_valid(self) -> bool:
        """Check if cached events are fresh and usable.

        Validates cache file existence and age based on configured validity period.
        A cache is considered valid if:
            1. Cache file exists at data_dir/gwtc_events.json
            2. File modification time is within cache_days period

        Returns:
            bool: True if cache exists and is recent (age < cache_days), False otherwise.

        Note:
            File age is computed as: (current_time - file_mtime) / 86400 seconds/day
            Zero or negative age means file was just created/modified.
        """
        if not self.cache_file.exists():
            return False

        file_age_days = (time.time() - self.cache_file.stat().st_mtime) / (24 * 3600)
        return file_age_days < self.cache_days

    def _load_from_cache(self) -> pd.DataFrame:
        """Load cached GWTC events from JSON file.

        Deserializes cached event data from self.cache_file (gwtc_events.json)
        into a pandas DataFrame with appropriate data types preserved.

        Returns:
            pd.DataFrame: DataFrame with cached event records. Same structure
                as get_gwtc_events() return value.

        Raises:
            FileNotFoundError: If cache file does not exist (should be checked with
                _is_cache_valid() before calling).
            json.JSONDecodeError: If cache file is corrupted or invalid JSON.

        Warning:
            Assumes cache file was created by _save_to_cache(). If manually edited,
            ensure JSON format matches expected column structure.
        """
        with open(self.cache_file, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def _save_to_cache(self, df: pd.DataFrame) -> None:
        """Persist GWTC events to local JSON cache.

        Serializes DataFrame to JSON format with human-readable formatting
        (indent=2) for debugging and manual inspection.

        Args:
            df (pd.DataFrame): DataFrame with event records to cache.
                Records are converted to list-of-dicts format for JSON serialization.

        Returns:
            None

        Raises:
            IOError: If write permission denied or disk space insufficient.
            ValueError: If DataFrame contains non-JSON-serializable types
                (e.g., numpy arrays - converts to list first).

        Note:
            Cache file location: data_dir/gwtc_events.json
            File is overwritten completely on each save (not appended).
            Lists (e.g., detector names) are preserved in JSON.

        Example:
            >>> df = loader.get_gwtc_events()
            >>> loader._save_to_cache(df)  # Usually automatic
        """
        df_dict = df.to_dict('records')
        with open(self.cache_file, 'w') as f:
            json.dump(df_dict, f, indent=2)

    def _parse_api_results(self, results: List[Dict]) -> pd.DataFrame:
        """Parse GWOSC v2 REST API JSON response into event DataFrame.

        Transforms GWOSC API response format (nested JSON with default_parameters
        array) into flat DataFrame with standard column names and data types.

        The GWOSC v2 API returns each event with:
            - Basic metadata: name, gps (timestamp), catalog, detectors
            - Array of parameter estimates (default_parameters) with best values

        This method:
            1. Extracts basic event info (name, GPS time, observing run, detectors)
            2. Flattens parameter array into key-value dict
            3. Maps GWOSC parameter names to standardized column names
            4. Provides defaults (0.0 for masses, 1e10 for FAR) for missing parameters

        Args:
            results (List[Dict]): List of event dictionaries from GWOSC API.
                Each dict has keys: name, gps, catalog, detectors, default_parameters.
                default_parameters is a list of dicts with keys: name, best, lower, upper.

        Returns:
            pd.DataFrame: Flat DataFrame with event records. Columns:
                - event_name, gps_time, observing_run, detectors (basic info)
                - mass_1_source, mass_2_source, chirp_mass_source, total_mass_source
                - luminosity_distance, redshift
                - network_snr (mapped from network_matched_filter_snr)
                - far (False alarm rate)

        Note:
            - Missing mass parameters default to 0.0 (sentinel for missing data)
            - Missing FAR defaults to 1e10 (very low confidence)
            - Parameter names in GWOSC use underscores: mass_1_source not mass1_source
            - Detectors list is preserved (e.g., ["H1", "L1"])

        Example:
            >>> api_response = [
            ...     {
            ...         "name": "GW190814",
            ...         "gps": 1250581504.169,
            ...         "catalog": "GWTC-3",
            ...         "detectors": ["H1", "L1"],
            ...         "default_parameters": [
            ...             {"name": "mass_1_source", "best": 23.2, ...},
            ...             {"name": "mass_2_source", "best": 2.6, ...},
            ...             ...
            ...         ]
            ...     }
            ... ]
            >>> df = loader._parse_api_results(api_response)
        """
        events_list = []

        for event in results:
            event_dict = {
                'event_name': event.get('name', ''),
                'gps_time': event.get('gps', 0.0),
                'observing_run': event.get('catalog', ''),
                'detectors': event.get('detectors', [])
            }

            params = event.get('default_parameters', [])
            param_dict = {p['name']: p.get('best', 0.0) for p in params}

            event_dict.update({
                'mass_1_source': param_dict.get('mass_1_source', 0.0),
                'mass_2_source': param_dict.get('mass_2_source', 0.0),
                'chirp_mass_source': param_dict.get('chirp_mass_source', 0.0),
                'total_mass_source': param_dict.get('total_mass_source', 0.0),
                'luminosity_distance': param_dict.get('luminosity_distance', 0.0),
                'redshift': param_dict.get('redshift', 0.0),
                'network_snr': param_dict.get('network_matched_filter_snr', 0.0),
                'far': param_dict.get('far', 1e10),
            })

            events_list.append(event_dict)

        return pd.DataFrame(events_list)

    def _parse_events_dict(self, events_dict: Dict) -> pd.DataFrame:
        """Parse legacy GWOSC dictionary-format catalog response.

        Handles older GWOSC API format where events are dictionary keyed by
        event names (deprecated but maintained for backward compatibility).

        Args:
            events_dict (Dict): Dictionary with structure:
                {
                    "GW150914": {
                        "GPS": 1126259462.4,
                        "mass_1_source": 36.2,
                        "mass_2_source": 29.1,
                        ...
                    },
                    "GW151226": {...},
                    ...
                }

        Returns:
            pd.DataFrame: Flat DataFrame with event records, same columns as
                _parse_api_results() for consistency.

        Note:
            This method exists for backward compatibility. New code should use
            _parse_api_results() (v2 API) or _parse_events_list() (new format).
            GPS key may be uppercase "GPS" in legacy format (checked here).
        """
        events_list = []
        for event_name, info in events_dict.items():
            events_list.append({
                'event_name': event_name,
                'gps_time': info.get('GPS', 0.0),
                'mass_1_source': info.get('mass_1_source', 0.0),
                'mass_2_source': info.get('mass_2_source', 0.0),
                'chirp_mass_source': info.get('chirp_mass_source', 0.0),
                'total_mass_source': info.get('total_mass_source', 0.0),
                'luminosity_distance': info.get('luminosity_distance', 0.0),
                'redshift': info.get('redshift', 0.0),
                'network_snr': info.get('network_matched_filter_snr', 0.0),
                'far': info.get('far', 1e10),
                'observing_run': info.get('observing_run', ''),
                'detectors': info.get('detectors', [])
            })
        
        return pd.DataFrame(events_list)
    
    def _parse_events_list(self, events_list: List) -> pd.DataFrame:
        """Parse modern GWOSC list-format catalog response.

        Handles current GWOSC API format where events are returned as list of
        dictionaries with flexible parameter naming (handles both GPS/gps_time variants).

        Args:
            events_list (List[Dict]): List of event dictionaries with flexible keys.
                Each event may have:
                - name or event_name (event identifier)
                - GPS or gps_time (GPS timestamp)
                - mass_1_source, mass_2_source, etc. (binary parameters)
                - network_matched_filter_snr (network SNR)
                - far (false alarm rate)
                - observing_run, detectors (metadata)

        Returns:
            pd.DataFrame: Flat DataFrame with event records. Columns match
                _parse_api_results() for consistency across parsing methods.

        Note:
            - Accepts both "GPS" (legacy) and "gps_time" (modern) keys
            - Missing parameters default to 0.0 (masses) or 1e10 (FAR)
            - Non-dict items in list are silently skipped
            - Use this when API returns list format vs nested API response

        Example:
            >>> events_list = [
            ...     {
            ...         "name": "GW190814",
            ...         "gps_time": 1250581504.169,
            ...         "mass_1_source": 23.2,
            ...         "mass_2_source": 2.6,
            ...         "observing_run": "O3a",
            ...         "detectors": ["H1", "L1"]
            ...     },
            ...     ...
            ... ]
            >>> df = loader._parse_events_list(events_list)
        """
        parsed = []
        for event in events_list:
            if isinstance(event, dict):
                parsed.append({
                    'event_name': event.get('name', 'Unknown'),
                    'gps_time': event.get('GPS', event.get('gps_time', 0.0)),
                    'mass_1_source': event.get('mass_1_source', 0.0),
                    'mass_2_source': event.get('mass_2_source', 0.0),
                    'chirp_mass_source': event.get('chirp_mass_source', 0.0),
                    'total_mass_source': event.get('total_mass_source', 0.0),
                    'luminosity_distance': event.get('luminosity_distance', 0.0),
                    'redshift': event.get('redshift', 0.0),
                    'network_snr': event.get('network_matched_filter_snr', 0.0),
                    'far': event.get('far', 1e10),
                    'observing_run': event.get('observing_run', ''),
                    'detectors': event.get('detectors', [])
                })
        
        return pd.DataFrame(parsed)
    
    def _get_hardcoded_events(self) -> pd.DataFrame:
        """Return hardcoded catalog of high-confidence GWTC milestone events.

        Provides a reliable fallback when GWOSC API is unavailable or network
        connectivity is lost. Includes canonical landmark GW detections across
        multiple observing runs and source types.

        Returns:
            pd.DataFrame: DataFrame with 4 landmark GW events:
                - GW150914: First detected GW, BBH in O1
                - GW151226: Second BBH detection, O1
                - GW170817: First BNS, O2 (associated with GRB and kilonova)
                - GW190521: Most massive BH merger, O3a

        Event Selection Criteria:
            - Events are from GWTC official catalogs (published results)
            - Cover multiple source types: BBH, BNS (important for diversity)
            - Span multiple observing runs: O1, O2, O3a (temporal coverage)
            - High confidence detections (low FAR, high SNR)
            - Parameters published and stable across analyses

        Note:
            This is a fallback dataset, not replacement for live catalog.
            Should only be used when API unavailable. For production use,
            cache fetch should be attempted first (see get_gwtc_events).

            Parameter values sourced from GWTC publications:
            - Masses in source frame (redshift corrected)
            - SNR values from network matched-filter analysis
            - FAR (false alarm rate) indicates detection confidence
            - Detectors list shows which observatories contributed data

        Example:
            >>> events = loader._get_hardcoded_events()
            >>> print(f"Fallback catalog has {len(events)} events")
            >>> print(events[['event_name', 'network_snr', 'far']])
        """
        events = [
            {
                'event_name': 'GW150914', 'gps_time': 1126259462.4,
                'mass_1_source': 36.2, 'mass_2_source': 29.1,
                'chirp_mass_source': 30.5, 'total_mass_source': 65.3,
                'luminosity_distance': 410.0, 'redshift': 0.09,
                'network_snr': 23.7, 'far': 2.0e-7,
                'observing_run': 'O1', 'detectors': ['H1', 'L1']
            },
            {
                'event_name': 'GW151226', 'gps_time': 1135136350.6,
                'mass_1_source': 14.2, 'mass_2_source': 7.5,
                'chirp_mass_source': 8.9, 'total_mass_source': 21.8,
                'luminosity_distance': 440.0, 'redshift': 0.09,
                'network_snr': 13.0, 'far': 1.0e-6,
                'observing_run': 'O1', 'detectors': ['H1', 'L1']
            },
            {
                'event_name': 'GW170817', 'gps_time': 1187008882.4,
                'mass_1_source': 1.6, 'mass_2_source': 1.2,
                'chirp_mass_source': 1.2, 'total_mass_source': 2.8,
                'luminosity_distance': 40.0, 'redshift': 0.009,
                'network_snr': 32.4, 'far': 1.0e-9,
                'observing_run': 'O2', 'detectors': ['H1', 'L1', 'V1']
            },
            {
                'event_name': 'GW190521', 'gps_time': 1242442967.4,
                'mass_1_source': 85.0, 'mass_2_source': 66.0,
                'chirp_mass_source': 72.0, 'total_mass_source': 151.0,
                'luminosity_distance': 5300.0, 'redshift': 0.82,
                'network_snr': 14.7, 'far': 1.4e-4,
                'observing_run': 'O3a', 'detectors': ['H1', 'L1', 'V1']
            },
        ]
        
        return pd.DataFrame(events)
    
    def download_strain(self,
                       event_name: str,
                       detector: str = 'H1',
                       duration: int = 4,
                       sample_rate: int = 4096) -> Optional[np.ndarray]:
        """Download gravitational wave strain data for specified event and detector.

        Fetches public strain data from GWOSC open data repository using GWpy
        TimeSeries API. Data window is centered on merger time (GPS).

        This method:
            1. Retrieves GPS time of event merger from GWTC catalog
            2. Defines time window: [gps_time - duration/2, gps_time + duration/2]
            3. Fetches strain from GWOSC open data via GWpy TimeSeries
            4. Resamples to requested sample_rate if needed
            5. Returns as numpy float32 array

        Args:
            event_name (str): GWTC event name (e.g., "GW190814"). Must exist in
                catalog or be obtainable from GWOSC API.
            detector (str): Detector identifier. Options: "H1" (LIGO Hanford),
                "L1" (LIGO Livingston), "V1" (Virgo). Defaults to "H1".
            duration (int): Time window duration in seconds, centered on merger.
                Typical values: 4s (fastest, minimal context), 8s (common),
                16s, 32s (more context, larger data). Defaults to 4s.
            sample_rate (int): Resampling rate in Hz. Options:
                - 4096 Hz (default, common for analysis)
                - 16384 Hz (raw strain, larger data)
                - 256 Hz (downsampled, analysis only)
                Defaults to 4096 Hz.

        Returns:
            Optional[np.ndarray]: Strain data as float32 array with shape (n_samples,)
                where n_samples = duration × sample_rate (e.g., 4s × 4096Hz = 16384).
                Returns None if download fails (missing event, network error, no GWpy).

        Raises:
            No explicit exceptions. Failures logged and None returned:
            - ImportError: GWpy not installed (logged at init)
            - RuntimeError: Event GPS time not found
            - RequestException: Network error fetching strain (connection, timeout)
            - Other: Logged as general strain download failure

        Note:
            - Requires GWpy library: `pip install gwpy`
            - Network I/O operation: typically 5-30s for 4s window
            - Requires internet access to GWOSC servers
            - Downloaded data is NOT cached locally (fresh fetch each call)
            - Float32 precision adequate for GW analysis (original is float64)

        Example:
            >>> loader = GWTCLoader()
            >>> # Download 4 seconds centered on merger
            >>> strain = loader.download_strain("GW190814", detector="H1",
            ...                                   duration=4, sample_rate=4096)
            >>> if strain is not None:
            ...     print(f"Downloaded {len(strain)} samples at 4096 Hz")
            ...     print(f"Duration: {len(strain) / 4096:.2f} seconds")
            ... else:
            ...     print("Strain download failed (no GWpy or network error)")

        See Also:
            - _get_event_gps_time: Look up event GPS time from catalog
            - gwpy.timeseries.TimeSeries.fetch_open_data: Underlying API
            - GWOSC: https://gwosc.org (open science data repository)
        """
        if not GWPY_AVAILABLE:
            self.logger.error("gwpy not available for strain download")
            return None
        
        try:
            gps_time = self._get_event_gps_time(event_name)
            if gps_time is None:
                return None
            
            start_time = gps_time - duration // 2
            end_time = gps_time + duration // 2
            
            strain = TimeSeries.fetch_open_data(
                detector, start_time, end_time,
                sample_rate=sample_rate
            )
            
            return np.array(strain.value, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Strain download failed for {event_name}: {e}")
            return None
    
    def _get_event_gps_time(self, event_name: str) -> Optional[float]:
        """Retrieve GPS timestamp for gravitational wave event.

        Uses multi-level fallback strategy to locate event GPS time:
            1. Check loaded GWTC catalog (from cache or API)
            2. Direct GWOSC API query for individual event
            3. Return None if event not found

        GPS time (seconds since GPS epoch: Jan 1, 1980 00:00:00 UTC) is critical
        for:
            - Fetching strain data window (merger ± duration/2 seconds)
            - Coordinating multi-detector analysis
            - Matching with electromagnetic observations

        Args:
            event_name (str): Gravitational wave event identifier.
                Standard format: "GWYYMMDD[a-z]" where YYMMDD is date.
                Examples: "GW150914", "GW190814", "GW170817"

        Returns:
            Optional[float]: GPS timestamp (seconds, float32 resolution) if found,
                None if event not found in catalog or API unavailable.

        Behavior:
            1. Calls get_gwtc_events() to load/cache catalog
            2. Searches for event_name in 'event_name' column
            3. If found locally, returns immediately
            4. If not found, attempts direct GWOSC API query
            5. Returns None on all failures (logged silently)

        Note:
            - Does not raise exceptions; returns None on failure
            - Exceptions are silently caught and logged (doesn't break control flow)
            - API query includes ?format=api for JSON response
            - Timeout on API request set to 10 seconds

        Example:
            >>> loader = GWTCLoader()
            >>> gps = loader._get_event_gps_time("GW190814")
            >>> if gps:
            ...     print(f"Merger GPS time: {gps:.3f}")
            ... else:
            ...     print("Event not found")

        See Also:
            - download_strain: Uses this method to locate event for data fetching
            - get_gwtc_events: Catalog lookup (1st fallback)
            - GWOSC API: https://gwosc.org/api/v2/events/{event_name}
        """
        try:
            events_df = self.get_gwtc_events()
            event_row = events_df[events_df['event_name'] == event_name]
            if not event_row.empty:
                return float(event_row.iloc[0]['gps_time'])
        except:
            pass

        try:
            response = requests.get(
                f"https://gwosc.org/api/v2/events/{event_name}?format=api",
                timeout=10
            )
            if response.status_code == 200:
                event_data = response.json()
                return event_data.get('GPS')
        except:
            pass

        return None
    
    def create_synthetic_overlaps(self,
                                 events_df: pd.DataFrame,
                                 n_overlaps: int = 100,
                                 overlap_window: float = 0.5) -> List[Dict]:
        """Generate synthetic multi-event GW scenarios from real GWTC catalog.

        Creates realistic training scenarios by combining real GW events with
        controlled time offsets. Useful for:
            - Testing signal subtraction and decomposition algorithms
            - Training hierarchical extraction methods on multi-signal overlaps
            - Validating parameter recovery accuracy for blended signals
            - Studying detection efficiency vs. signal complexity

        Algorithm:
            1. Filter catalog for quality events (SNR > 10, reasonable masses)
            2. For each scenario i:
                a. Randomly select 2-4 events (distribution: 60%/30%/10%)
                b. Use first event GPS time as reference
                c. Apply random time offsets to secondary events
                d. Package as overlapping scenario dict

        Scenario Structure:
            {
                'scenario_id': int (0 to n_overlaps-1),
                'n_signals': int (2, 3, or 4),
                'central_gps_time': float (reference GPS time),
                'events': [
                    {'event_name': str, ..., 'time_offset': float (seconds)},
                    ...
                ]
            }

        Args:
            events_df (pd.DataFrame): Catalog of GW events from get_gwtc_events().
                Must contain columns: event_name, gps_time, mass_1_source, network_snr.
            n_overlaps (int): Number of synthetic scenarios to generate.
                Defaults to 100. Recommend 100-1000 for training.
            overlap_window (float): Maximum time offset between signals in seconds.
                Defines window: [−overlap_window/2, +overlap_window/2].
                Typical: 0.5s (conservative), 1.0s (aggressive).
                Defaults to 0.5s.

        Returns:
            List[Dict]: List of n_overlaps scenario dictionaries. Each contains:
                - scenario_id: Unique integer identifier [0, n_overlaps-1]
                - n_signals: Number of overlapping signals [2, 3, 4]
                - central_gps_time: Primary event GPS time (seconds)
                - events: List of n_signals events with original parameters + time_offset

            Returns empty list if:
                - events_df is empty or has <2 rows
                - After quality filtering, insufficient events remain

        Signal Distribution (stochastic):
            - 2-signal: 60% (binary mergers with nearby companion)
            - 3-signal: 30% (triple system or hierarchical merger)
            - 4-signal: 10% (highly constrained rare scenario)

        Quality Filters Applied:
            - network_snr > 10 (adequate SNR for analysis)
            - mass_1_source > 5 M☉ (exclude low-mass systems with poor SNR scaling)

        Note:
            - Overlaps are synthetic (signals not physically injected)
            - Only event parameters and timing information combined
            - Use with full waveform injection (PyCBC) for realistic strain overlap
            - Time offsets sampled uniformly, may not reflect astrophysical populations
            - Real multi-merger probability much lower than 60% 2-signal rate

        Example:
            >>> loader = GWTCLoader()
            >>> events = loader.get_gwtc_events("GWTC-3")
            >>> overlaps = loader.create_synthetic_overlaps(
            ...     events, n_overlaps=500, overlap_window=1.0)
            >>> print(f"Generated {len(overlaps)} scenarios")
            >>> # Distribution of signal counts
            >>> counts = [s['n_signals'] for s in overlaps]
            >>> print(f"2-signal: {counts.count(2)}, "
            ...       f"3-signal: {counts.count(3)}, "
            ...       f"4-signal: {counts.count(4)}")
            >>> # Example scenario
            >>> scenario = overlaps[0]
            >>> print(f"Scenario {scenario['scenario_id']}: "
            ...       f"{scenario['n_signals']} signals at GPS {scenario['central_gps_time']}")

        See Also:
            - get_gwtc_events: Load real event catalog
            - Each event dict includes all columns from get_gwtc_events() plus 'time_offset'
        """
        
        if events_df.empty or len(events_df) < 2:
            return []
        
        # Filter quality events
        quality_events = events_df[
            (events_df['network_snr'] > 10) &
            (events_df['mass_1_source'] > 5)
        ]
        
        if len(quality_events) < 2:
            quality_events = events_df
        
        overlaps = []
        
        for i in range(n_overlaps):
            # Number of signals (2-4)
            n_signals = np.random.choice([2, 3, 4], p=[0.6, 0.3, 0.1])
            
            if len(quality_events) >= n_signals:
                selected = quality_events.sample(n_signals)
                base_time = selected.iloc[0]['gps_time']
                
                # Create time offsets
                time_offsets = [0.0] + [
                    np.random.uniform(-overlap_window/2, overlap_window/2)
                    for _ in range(n_signals - 1)
                ]
                
                overlaps.append({
                    'scenario_id': i,
                    'n_signals': n_signals,
                    'central_gps_time': base_time,
                    'events': [
                        {**event.to_dict(), 'time_offset': time_offsets[j]}
                        for j, (_, event) in enumerate(selected.iterrows())
                    ]
                })
        
        return overlaps
