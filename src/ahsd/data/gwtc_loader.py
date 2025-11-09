"""
GWTC Catalog Loader
Load real gravitational wave events from GWTC catalogs
"""

import numpy as np
import pandas as pd
import logging
import requests
import json
from typing import Dict, List, Optional
from pathlib import Path
import time

try:
    from gwpy.timeseries import TimeSeries
    GWPY_AVAILABLE = True
except ImportError:
    GWPY_AVAILABLE = False

class GWTCLoader:
    """
    Load and process real GW event data from GWTC catalogs
    Supports GWTC-1, GWTC-2, GWTC-3, GWTC-4
    """

    def __init__(self, data_dir: str = "data/gwtc", cache_days: int = 7):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.data_dir / "gwtc_events.json"
        self.cache_days = cache_days  # Cache validity in days
        self.logger = logging.getLogger(__name__)
        self.gwosc_base_url = "https://gwosc.org"
    
    def get_gwtc_events(self, catalog: str = "GWTC") -> pd.DataFrame:
        """
        Fetch all events from GWTC catalog

        Args:
            catalog: Catalog name (GWTC-1, GWTC-2, GWTC-3, GWTC-4, or GWTC for cumulative)

        Returns:
            DataFrame with event parameters
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
        """Check if cache file exists and is recent enough"""
        if not self.cache_file.exists():
            return False

        # Check file age
        file_age_days = (time.time() - self.cache_file.stat().st_mtime) / (24 * 3600)
        return file_age_days < self.cache_days

    def _load_from_cache(self) -> pd.DataFrame:
        """Load events from cache file"""
        with open(self.cache_file, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def _save_to_cache(self, df: pd.DataFrame) -> None:
        """Save events to cache file"""
        df_dict = df.to_dict('records')
        with open(self.cache_file, 'w') as f:
            json.dump(df_dict, f, indent=2)

    def _parse_api_results(self, results: List[Dict]) -> pd.DataFrame:
        """Parse GWOSC v2 API results"""
        events_list = []

        for event in results:
            # Extract basic info
            event_dict = {
                'event_name': event.get('name', ''),
                'gps_time': event.get('gps', 0.0),
                'observing_run': event.get('catalog', ''),
                'detectors': event.get('detectors', [])
            }

            # Extract parameters from default_parameters
            params = event.get('default_parameters', [])
            param_dict = {p['name']: p.get('best', 0.0) for p in params}

            # Map to expected columns
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
        """Parse GWOSC old dict format"""
        
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
        """Parse GWOSC new list format"""
        
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
        """Hardcoded high-confidence GWTC events"""
        
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
        """Download strain data for event"""
        
        if not GWPY_AVAILABLE:
            self.logger.error("gwpy not available for strain download")
            return None
        
        try:
            # Get event GPS time
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
        """Get GPS time for event"""

        # First try from cached events
        try:
            events_df = self.get_gwtc_events()
            event_row = events_df[events_df['event_name'] == event_name]
            if not event_row.empty:
                return float(event_row.iloc[0]['gps_time'])
        except:
            pass

        # Fallback to direct API call
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
        """
        Create synthetic overlapping scenarios from real events
        
        Args:
            events_df: DataFrame with GWTC events
            n_overlaps: Number of overlapping scenarios to create
            overlap_window: Time window for overlaps (seconds)
            
        Returns:
            List of overlapping scenario dicts
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
