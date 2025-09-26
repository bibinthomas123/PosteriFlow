import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityFlag
import requests
import json
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import h5py


class GWTCDataLoader:
    """
    GWTCDataLoader
    A utility class for loading and processing real gravitational-wave event data from the LIGO-Virgo-KAGRA GWTC catalogs (up to GWTC-4.0), as well as downloading strain data and generating synthetic overlapping event scenarios.
    Main Features:
    --------------
    - Fetches event metadata from GWOSC API endpoints, with robust fallback to hardcoded event lists if APIs are unavailable.
    - Parses event data from various API response formats (old dict, new list, paginated).
    - Downloads strain data for specific events and detectors using gwpy.
    - Identifies real events with overlapping GPS times within a configurable window.
    - Generates synthetic overlapping event scenarios for simulation and testing.
    - Loads background strain data for overlapping scenarios, with fallback to synthetic Gaussian noise if real data is unavailable.
    Parameters
    ----------
    data_dir : str, optional
        Directory to store downloaded data (default: "data/raw").
    Attributes
    ----------
    data_dir : pathlib.Path
        Path object for the data directory.
    logger : logging.Logger
        Logger for status and error messages.
    gwosc_base_url : str
        Base URL for GWOSC API endpoints.
    Methods
    -------
    get_gwtc_events(catalog: str = "GWTC-4") -> pd.DataFrame
        Fetch all events from the specified GWTC catalog.
    download_strain_data(event_name: str, detector: str = "H1", duration: int = 32, sampling_rate: int = 4096) -> Optional[TimeSeries]
        Download strain data for a specific event and detector.
    load_overlapping_candidates(time_window: float = 1.0, min_events: int = 2) -> List[Dict]
        Find groups of events with overlapping GPS times.
    create_synthetic_overlaps(events_df: pd.DataFrame, n_overlaps: int = 100) -> List[Dict]
        Generate synthetic overlapping event scenarios from real events.
    load_strain_for_overlap(overlap_scenario: Dict, detectors: List[str] = ['H1', 'L1'], duration: int = 4, sampling_rate: int = 4096) -> Dict
        Load background strain data for a given overlapping scenario.
    Notes
    -----
    - Requires `requests`, `pandas`, `numpy`, and `gwpy` packages.
    - Handles changes in GWOSC API formats and provides robust fallbacks.
    - Designed for use in gravitational-wave data analysis and simulation pipelines.
    """
    """Load real LIGO-Virgo-KAGRA data from GWTC-4.0"""
    
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Updated GWOSC API endpoints for 2025
        self.gwosc_base_url = "https://gwosc.org"
        
    def get_gwtc_events(self, catalog: str = "GWTC-4") -> pd.DataFrame:
        """Get all events from GWTC catalog using updated API"""
        
        # Try multiple endpoint patterns
        endpoints_to_try = [
            f"https://gwosc.org/eventapi/json/{catalog}/",
            f"https://gwosc.org/eventapi/json/GWTC-4.0/",
            "https://gwosc.org/eventapi/json/confident/",
            "https://gwosc.org/eventapi/json/GWTC-3/",
            "https://gwosc.org/api/v1/events/{}/".format(catalog),
        ]
        
        for endpoint in endpoints_to_try:
            try:
                self.logger.info(f"Trying endpoint: {endpoint}")
                response = requests.get(endpoint, timeout=10)
                
                if response.status_code == 200:
                    try:
                        events_data = response.json()
                        
                        # Handle different response formats
                        if 'events' in events_data:
                            # Old format
                            return self._parse_events_old_format(events_data['events'])
                        elif isinstance(events_data, list):
                            # New list format
                            return self._parse_events_list_format(events_data)
                        elif 'results' in events_data:
                            # Paginated format
                            return self._parse_events_paginated_format(events_data)
                        else:
                            self.logger.warning(f"Unknown response format from {endpoint}")
                            continue
                            
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"JSON decode error for {endpoint}: {e}")
                        continue
                        
                else:
                    self.logger.debug(f"HTTP {response.status_code} from {endpoint}")
                    continue
                    
            except requests.RequestException as e:
                self.logger.warning(f"Request failed for {endpoint}: {e}")
                continue
        
        # If all endpoints fail, try the hardcoded event list
        self.logger.warning("All GWTC endpoints failed, using hardcoded event list")
        return self._get_hardcoded_gwtc_events()
    
    def _parse_events_old_format(self, events_dict: Dict) -> pd.DataFrame:
        """Parse events from old API format"""
        events_list = []
        
        for event_name, event_info in events_dict.items():
            event_dict = {
                'event_name': event_name,
                'gps_time': event_info.get('GPS', 0),
                'mass_1_source': event_info.get('mass_1_source', 0),
                'mass_2_source': event_info.get('mass_2_source', 0),
                'luminosity_distance': event_info.get('luminosity_distance', 0),
                'network_snr': event_info.get('network_matched_filter_snr', 0),
                'total_mass_source': event_info.get('total_mass_source', 0),
                'chirp_mass_source': event_info.get('chirp_mass_source', 0),
                'final_mass_source': event_info.get('final_mass_source', 0),
                'observing_run': event_info.get('observing_run', ''),
                'detectors': event_info.get('detectors', [])
            }
            events_list.append(event_dict)
            
        return pd.DataFrame(events_list)
    
    def _parse_events_list_format(self, events_list: List) -> pd.DataFrame:
        """Parse events from new list API format"""
        parsed_events = []
        
        for event in events_list:
            if isinstance(event, dict):
                event_dict = {
                    'event_name': event.get('name', event.get('event', 'Unknown')),
                    'gps_time': event.get('GPS', event.get('gps_time', 0)),
                    'mass_1_source': event.get('mass_1_source', 0),
                    'mass_2_source': event.get('mass_2_source', 0),
                    'luminosity_distance': event.get('luminosity_distance', 0),
                    'network_snr': event.get('network_matched_filter_snr', event.get('snr', 0)),
                    'total_mass_source': event.get('total_mass_source', 0),
                    'chirp_mass_source': event.get('chirp_mass_source', 0),
                    'final_mass_source': event.get('final_mass_source', 0),
                    'observing_run': event.get('observing_run', event.get('run', '')),
                    'detectors': event.get('detectors', [])
                }
                parsed_events.append(event_dict)
                
        return pd.DataFrame(parsed_events)
    
    def _parse_events_paginated_format(self, response_data: Dict) -> pd.DataFrame:
        """Parse events from paginated API format"""
        events_list = response_data.get('results', [])
        return self._parse_events_list_format(events_list)
    
    def _get_hardcoded_gwtc_events(self) -> pd.DataFrame:
        """Fallback hardcoded event list based on GWTC-4.0"""
        
        # Some well-known events from GWTC catalogs
        hardcoded_events = [
            {
                'event_name': 'GW150914',
                'gps_time': 1126259462.4,
                'mass_1_source': 36.2,
                'mass_2_source': 29.1,
                'luminosity_distance': 410.0,
                'network_snr': 23.7,
                'total_mass_source': 65.3,
                'chirp_mass_source': 28.6,
                'final_mass_source': 62.3,
                'observing_run': 'O1',
                'detectors': ['H1', 'L1']
            },
            {
                'event_name': 'GW190521',
                'gps_time': 1242442967.4,
                'mass_1_source': 85.0,
                'mass_2_source': 66.0,
                'luminosity_distance': 5300.0,
                'network_snr': 14.7,
                'total_mass_source': 151.0,
                'chirp_mass_source': 65.0,
                'final_mass_source': 142.0,
                'observing_run': 'O3a',
                'detectors': ['H1', 'L1', 'V1']
            },
            {
                'event_name': 'GW231123',  # New high-mass event from O4
                'gps_time': 1384950870.4,
                'mass_1_source': 120.0,
                'mass_2_source': 110.0,
                'luminosity_distance': 3000.0,
                'network_snr': 18.0,
                'total_mass_source': 230.0,
                'chirp_mass_source': 100.0,
                'final_mass_source': 225.0,
                'observing_run': 'O4',
                'detectors': ['H1', 'L1', 'V1']
            },
            {
                'event_name': 'GW170817',  # Famous neutron star merger
                'gps_time': 1187008882.4,
                'mass_1_source': 1.6,
                'mass_2_source': 1.2,
                'luminosity_distance': 40.0,
                'network_snr': 32.4,
                'total_mass_source': 2.8,
                'chirp_mass_source': 1.2,
                'final_mass_source': 2.7,
                'observing_run': 'O2',
                'detectors': ['H1', 'L1', 'V1']
            },
            {
                'event_name': 'GW200115_042309',
                'gps_time': 1263084207.3,
                'mass_1_source': 5.9,
                'mass_2_source': 1.4,
                'luminosity_distance': 300.0,
                'network_snr': 15.3,
                'total_mass_source': 7.3,
                'chirp_mass_source': 2.1,
                'final_mass_source': 7.0,
                'observing_run': 'O3b',
                'detectors': ['H1', 'L1', 'V1']
            }
        ]
        
        self.logger.info(f"Using {len(hardcoded_events)} hardcoded events")
        return pd.DataFrame(hardcoded_events)

    def download_strain_data(self, 
                           event_name: str, 
                           detector: str = "H1", 
                           duration: int = 32,
                           sampling_rate: int = 4096) -> Optional[TimeSeries]:
        """Download strain data for specific event"""
        try:
            # Get GPS time for event
            gps_time = self._get_event_gps_time(event_name)
            
            if gps_time is None:
                return None
                
            # Download strain data using gwpy
            start_time = gps_time - duration // 2
            end_time = gps_time + duration // 2
            
            strain = TimeSeries.fetch_open_data(
                detector, 
                start_time, 
                end_time,
                sample_rate=sampling_rate
            )
            
            return strain
            
        except Exception as e:
            self.logger.error(f"Failed to download strain data for {event_name}: {e}")
            return None
    
    def _get_event_gps_time(self, event_name: str) -> Optional[float]:
        """Get GPS time for specific event"""
        try:
            # Try new API format
            response = requests.get(f"https://gwosc.org/api/v1/events/{event_name}/")
            if response.status_code == 200:
                event_data = response.json()
                return event_data.get('GPS', event_data.get('gps_time'))
            
            # Fallback to old format
            response = requests.get(f"https://gwosc.org/eventapi/json/{event_name}/")
            if response.status_code == 200:
                event_data = response.json()
                return event_data.get('GPS', None)
                
            return None
        except:
            return None

    def load_overlapping_candidates(self, 
                                  time_window: float = 1.0,
                                  min_events: int = 2) -> List[Dict]:
        """Find events that could have overlapping signals"""
        events_df = self.get_gwtc_events()
        
        if events_df.empty:
            return []
        
        # Sort by GPS time
        events_df = events_df.sort_values('gps_time')
        
        overlapping_groups = []
        
        for i, event1 in events_df.iterrows():
            group = [event1.to_dict()]
            
            # Look for events within time window
            for j, event2 in events_df.iterrows():
                if i != j and abs(event1['gps_time'] - event2['gps_time']) <= time_window:
                    group.append(event2.to_dict())
            
            if len(group) >= min_events:
                overlapping_groups.append({
                    'group_id': len(overlapping_groups),
                    'events': group,
                    'central_gps_time': event1['gps_time']
                })
        
        return overlapping_groups

    def create_synthetic_overlaps(self, 
                                events_df: pd.DataFrame, 
                                n_overlaps: int = 100) -> List[Dict]:
        """Create synthetic overlapping scenarios from real events"""
        
        if events_df.empty:
            return []
            
        overlaps = []
        
        # Select high-quality events
        quality_events = events_df[
            (events_df['network_snr'] > 10) & 
            (events_df['mass_1_source'] > 5) & 
            (events_df['mass_2_source'] > 5)
        ]
        
        if len(quality_events) < 2:
            self.logger.warning("Not enough quality events for synthetic overlaps")
            quality_events = events_df  # Use all events
        
        for i in range(n_overlaps):
            # Randomly sample 2-4 events
            n_signals = np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2])
            
            if len(quality_events) >= n_signals:
                selected_events = quality_events.sample(n_signals)
                
                # Create synthetic GPS times close together
                base_time = selected_events.iloc[0]['gps_time']
                synthetic_times = base_time + np.random.uniform(-0.5, 0.5, n_signals)
                
                overlap_scenario = {
                    'scenario_id': i,
                    'n_signals': n_signals,
                    'central_gps_time': base_time,
                    'events': []
                }
                
                for j, (_, event) in enumerate(selected_events.iterrows()):
                    event_dict = event.to_dict()
                    event_dict['synthetic_gps_time'] = synthetic_times[j]
                    overlap_scenario['events'].append(event_dict)
                
                overlaps.append(overlap_scenario)
        
        return overlaps

    def load_strain_for_overlap(self, 
                              overlap_scenario: Dict, 
                              detectors: List[str] = ['H1', 'L1'],
                              duration: int = 4,
                              sampling_rate: int = 4096) -> Dict:
        """Load strain data for overlapping scenario"""
        
        strain_data = {}
        central_time = overlap_scenario['central_gps_time']
        
        for detector in detectors:
            try:
                # Load background noise from quiet period
                noise_start = central_time - 100  # 100s before event
                noise_end = noise_start + duration
                
                background = TimeSeries.fetch_open_data(
                    detector,
                    noise_start,
                    noise_end, 
                    sample_rate=sampling_rate
                )
                
                strain_data[detector] = background.value
                
            except Exception as e:
                self.logger.warning(f"Using synthetic noise for {detector}: {e}")
                # Use synthetic Gaussian noise if real data unavailable
                n_samples = duration * sampling_rate
                strain_data[detector] = np.random.normal(0, 1e-22, n_samples)
        
        return strain_data
