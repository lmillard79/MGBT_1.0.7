"""
Data processing module for MGBT analysis.

This module provides utilities for retrieving and processing hydrological data,
including USGS streamflow data retrieval and water year calculations.
"""

from .usgs import (
    get_usgs_peaks,
    get_usgs_daily_flow,
    search_usgs_sites,
    USGSDataError
)
from .water_year import (
    make_water_year,
    water_year_to_calendar,
    get_water_year_range,
    filter_by_water_year
)

__all__ = [
    'get_usgs_peaks',
    'get_usgs_daily_flow', 
    'search_usgs_sites',
    'USGSDataError',
    'make_water_year',
    'water_year_to_calendar',
    'get_water_year_range',
    'filter_by_water_year'
]
