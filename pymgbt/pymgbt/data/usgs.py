"""
USGS data retrieval utilities for streamflow data.

This module provides functions to retrieve streamflow data from USGS NWIS
(National Water Information System) web services, including peak flow data
and daily flow data for MGBT analysis.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple, Union
import warnings
import io


class USGSDataError(Exception):
    """Exception raised for errors in USGS data retrieval."""
    pass


def get_usgs_peaks(
    site_no: str,
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None,
    parameter_cd: str = "00060",
    timeout: int = 30
) -> pd.DataFrame:
    """
    Retrieve annual peak streamflow data from USGS NWIS.
    
    Parameters
    ----------
    site_no : str
        USGS site number (e.g., '01646500')
    start_date : str or date, optional
        Start date for data retrieval (YYYY-MM-DD format)
    end_date : str or date, optional
        End date for data retrieval (YYYY-MM-DD format)
    parameter_cd : str, default='00060'
        USGS parameter code ('00060' for discharge)
    timeout : int, default=30
        Request timeout in seconds
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['site_no', 'peak_dt', 'peak_va', 'peak_cd', 'water_year']
        
    Raises
    ------
    USGSDataError
        If data retrieval fails or no data is found
        
    Notes
    -----
    This function retrieves data from the USGS Peak-flow data service.
    Peak flow data represents the maximum instantaneous discharge for each water year.
    """
    # Construct URL for USGS peak flow service
    base_url = "https://nwis.waterdata.usgs.gov/nwis/peak"
    
    params = {
        'site_no': site_no,
        'agency_cd': 'USGS',
        'format': 'rdb',
        'peak_dt_acy_cd': '3,2,1,6,7,8,9',  # Include various accuracy codes
        'parameter_cd': parameter_cd
    }
    
    # Add date range if specified
    if start_date is not None:
        if isinstance(start_date, date):
            start_date = start_date.strftime('%Y-%m-%d')
        params['begin_date'] = start_date
        
    if end_date is not None:
        if isinstance(end_date, date):
            end_date = end_date.strftime('%Y-%m-%d')
        params['end_date'] = end_date
    
    try:
        # Make request to USGS
        response = requests.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        
        # Parse the RDB format response
        content = response.text
        
        # Find the data section (after the header comments)
        lines = content.split('\n')
        data_start = None
        
        for i, line in enumerate(lines):
            if line.startswith('agency_cd'):
                data_start = i
                break
        
        if data_start is None:
            raise USGSDataError(f"No data header found for site {site_no}")
        
        # Skip the format line (contains data types)
        data_lines = lines[data_start+2:]
        
        # Filter out empty lines and comments
        data_lines = [line for line in data_lines if line.strip() and not line.startswith('#')]
        
        if not data_lines:
            raise USGSDataError(f"No peak flow data found for site {site_no}")
        
        # Create DataFrame from the data
        header = lines[data_start].split('\t')
        data_rows = []
        
        for line in data_lines:
            if line.strip():
                data_rows.append(line.split('\t'))
        
        if not data_rows:
            raise USGSDataError(f"No valid data rows found for site {site_no}")
        
        df = pd.DataFrame(data_rows, columns=header)
        
        # Clean and process the data
        df = df.replace('', np.nan)
        
        # Convert peak date to datetime
        df['peak_dt'] = pd.to_datetime(df['peak_dt'], errors='coerce')
        
        # Convert peak value to numeric
        df['peak_va'] = pd.to_numeric(df['peak_va'], errors='coerce')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['peak_dt', 'peak_va'])
        
        if df.empty:
            raise USGSDataError(f"No valid peak flow records after data cleaning for site {site_no}")
        
        # Add water year column
        df['water_year'] = df['peak_dt'].apply(lambda x: x.year if x.month < 10 else x.year + 1)
        
        # Select and rename relevant columns
        result_columns = ['site_no', 'peak_dt', 'peak_va', 'peak_cd', 'water_year']
        available_columns = [col for col in result_columns if col in df.columns]
        
        if 'peak_cd' not in df.columns:
            df['peak_cd'] = ''
            available_columns.append('peak_cd')
        
        df = df[available_columns].copy()
        
        # Sort by water year
        df = df.sort_values('water_year').reset_index(drop=True)
        
        return df
        
    except requests.RequestException as e:
        raise USGSDataError(f"Failed to retrieve data from USGS: {str(e)}")
    except Exception as e:
        raise USGSDataError(f"Error processing USGS peak flow data: {str(e)}")


def get_usgs_daily_flow(
    site_no: str,
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None,
    parameter_cd: str = "00060",
    timeout: int = 30
) -> pd.DataFrame:
    """
    Retrieve daily streamflow data from USGS NWIS.
    
    Parameters
    ----------
    site_no : str
        USGS site number
    start_date : str or date, optional
        Start date for data retrieval
    end_date : str or date, optional
        End date for data retrieval
    parameter_cd : str, default='00060'
        USGS parameter code ('00060' for discharge)
    timeout : int, default=30
        Request timeout in seconds
        
    Returns
    -------
    pd.DataFrame
        DataFrame with daily flow data
        
    Raises
    ------
    USGSDataError
        If data retrieval fails
    """
    base_url = "https://nwis.waterdata.usgs.gov/nwis/dv"
    
    params = {
        'cb_00060': 'on',
        'format': 'rdb',
        'site_no': site_no,
        'referred_module': 'sw',
        'period': '',
        'begin_date': start_date or '',
        'end_date': end_date or ''
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        
        # Parse RDB format
        content = response.text
        lines = content.split('\n')
        
        # Find data start
        data_start = None
        for i, line in enumerate(lines):
            if line.startswith('agency_cd'):
                data_start = i
                break
        
        if data_start is None:
            raise USGSDataError(f"No data found for site {site_no}")
        
        # Skip format line and get data
        data_lines = lines[data_start+2:]
        data_lines = [line for line in data_lines if line.strip() and not line.startswith('#')]
        
        if not data_lines:
            raise USGSDataError(f"No daily flow data found for site {site_no}")
        
        # Create DataFrame
        header = lines[data_start].split('\t')
        data_rows = [line.split('\t') for line in data_lines if line.strip()]
        
        df = pd.DataFrame(data_rows, columns=header)
        df = df.replace('', np.nan)
        
        # Convert date and flow columns
        date_col = [col for col in df.columns if 'datetime' in col.lower()][0]
        flow_col = [col for col in df.columns if parameter_cd in col and '_va' in col][0]
        
        df[date_col] = pd.to_datetime(df[date_col])
        df[flow_col] = pd.to_numeric(df[flow_col], errors='coerce')
        
        # Rename columns for consistency
        df = df.rename(columns={
            date_col: 'date',
            flow_col: 'discharge'
        })
        
        # Add water year
        df['water_year'] = df['date'].apply(lambda x: x.year if x.month < 10 else x.year + 1)
        
        return df[['site_no', 'date', 'discharge', 'water_year']].dropna()
        
    except requests.RequestException as e:
        raise USGSDataError(f"Failed to retrieve daily flow data: {str(e)}")
    except Exception as e:
        raise USGSDataError(f"Error processing daily flow data: {str(e)}")


def search_usgs_sites(
    state_cd: Optional[str] = None,
    county_cd: Optional[str] = None,
    huc_cd: Optional[str] = None,
    site_type: str = "ST",
    has_peak_data: bool = True,
    timeout: int = 30
) -> pd.DataFrame:
    """
    Search for USGS streamflow sites.
    
    Parameters
    ----------
    state_cd : str, optional
        State code (e.g., 'VA' for Virginia)
    county_cd : str, optional
        County code
    huc_cd : str, optional
        Hydrologic Unit Code
    site_type : str, default='ST'
        Site type ('ST' for stream)
    has_peak_data : bool, default=True
        Filter for sites with peak flow data
    timeout : int, default=30
        Request timeout in seconds
        
    Returns
    -------
    pd.DataFrame
        DataFrame with site information
        
    Raises
    ------
    USGSDataError
        If search fails
    """
    base_url = "https://nwis.waterdata.usgs.gov/nwis/inventory"
    
    params = {
        'format': 'rdb',
        'site_tp_cd': site_type,
        'group_key': 'NONE',
        'sitefile_output_format': 'html_table',
        'column_name': 'agency_cd,site_no,station_nm,dec_lat_va,dec_long_va,drain_area_va',
        'list_of_search_criteria': 'lat_long_bounding_box,site_tp_cd'
    }
    
    if state_cd:
        params['state_cd'] = state_cd
    if county_cd:
        params['county_cd'] = county_cd
    if huc_cd:
        params['huc_cd'] = huc_cd
    
    if has_peak_data:
        params['peak_count_nu'] = '1'
    
    try:
        response = requests.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        
        # Parse the response similar to other functions
        content = response.text
        lines = content.split('\n')
        
        # Find data start
        data_start = None
        for i, line in enumerate(lines):
            if line.startswith('agency_cd'):
                data_start = i
                break
        
        if data_start is None:
            return pd.DataFrame()  # Return empty DataFrame if no sites found
        
        # Process data
        data_lines = lines[data_start+2:]
        data_lines = [line for line in data_lines if line.strip() and not line.startswith('#')]
        
        if not data_lines:
            return pd.DataFrame()
        
        header = lines[data_start].split('\t')
        data_rows = [line.split('\t') for line in data_lines if line.strip()]
        
        df = pd.DataFrame(data_rows, columns=header)
        df = df.replace('', np.nan)
        
        # Convert numeric columns
        numeric_cols = ['dec_lat_va', 'dec_long_va', 'drain_area_va']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except requests.RequestException as e:
        raise USGSDataError(f"Failed to search USGS sites: {str(e)}")
    except Exception as e:
        raise USGSDataError(f"Error processing site search: {str(e)}")


def validate_site_number(site_no: str) -> str:
    """
    Validate and format USGS site number.
    
    Parameters
    ----------
    site_no : str
        USGS site number
        
    Returns
    -------
    str
        Validated site number
        
    Raises
    ------
    ValueError
        If site number is invalid
    """
    if not isinstance(site_no, str):
        site_no = str(site_no)
    
    # Remove any non-numeric characters except leading zeros
    site_no = site_no.strip()
    
    # USGS site numbers are typically 8-15 digits
    if not site_no.isdigit():
        raise ValueError(f"Site number must contain only digits: {site_no}")
    
    if len(site_no) < 8 or len(site_no) > 15:
        warnings.warn(f"Site number length ({len(site_no)}) is outside typical range (8-15 digits)")
    
    return site_no


def get_site_info(site_no: str, timeout: int = 30) -> Dict:
    """
    Get basic information about a USGS site.
    
    Parameters
    ----------
    site_no : str
        USGS site number
    timeout : int, default=30
        Request timeout in seconds
        
    Returns
    -------
    dict
        Site information including name, location, drainage area
        
    Raises
    ------
    USGSDataError
        If site information cannot be retrieved
    """
    site_no = validate_site_number(site_no)
    
    base_url = "https://nwis.waterdata.usgs.gov/nwis/inventory"
    
    params = {
        'site_no': site_no,
        'format': 'rdb',
        'group_key': 'NONE',
        'column_name': 'agency_cd,site_no,station_nm,dec_lat_va,dec_long_va,drain_area_va,contrib_drain_area_va'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        
        content = response.text
        lines = content.split('\n')
        
        # Find and parse data
        data_start = None
        for i, line in enumerate(lines):
            if line.startswith('agency_cd'):
                data_start = i
                break
        
        if data_start is None or len(lines) <= data_start + 2:
            raise USGSDataError(f"Site {site_no} not found")
        
        # Get the first data line
        data_line = lines[data_start + 2].strip()
        if not data_line:
            raise USGSDataError(f"No data found for site {site_no}")
        
        header = lines[data_start].split('\t')
        data = data_line.split('\t')
        
        site_info = dict(zip(header, data))
        
        # Convert numeric fields
        numeric_fields = ['dec_lat_va', 'dec_long_va', 'drain_area_va', 'contrib_drain_area_va']
        for field in numeric_fields:
            if field in site_info and site_info[field]:
                try:
                    site_info[field] = float(site_info[field])
                except ValueError:
                    site_info[field] = None
        
        return site_info
        
    except requests.RequestException as e:
        raise USGSDataError(f"Failed to retrieve site information: {str(e)}")
    except Exception as e:
        raise USGSDataError(f"Error processing site information: {str(e)}")
