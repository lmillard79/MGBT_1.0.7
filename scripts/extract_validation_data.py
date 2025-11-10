"""
Extract annual maxima series from flike_bayes validation files.

This utility parses FLIKE output files to extract:
1. Gauged annual maximum discharge data
2. Censored flow observations
3. Model parameters and thresholds

The extracted data is used for validating the pymgbt package against R MGBT results.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationData:
    """Container for validation data extracted from FLIKE files."""
    
    station_id: str
    gauged_flows: pd.DataFrame
    censored_flows: pd.DataFrame
    zero_flow_threshold: float
    flood_model: str
    n_gauged: int
    n_censored: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'station_id': self.station_id,
            'gauged_flows': self.gauged_flows.to_dict('records'),
            'censored_flows': self.censored_flows.to_dict('records'),
            'zero_flow_threshold': self.zero_flow_threshold,
            'flood_model': self.flood_model,
            'n_gauged': self.n_gauged,
            'n_censored': self.n_censored
        }


def extract_station_id(filepath: Path) -> str:
    """Extract station ID from filename."""
    filename = filepath.stem
    
    # Pattern: flike_Bayes_Out_416040 or flike_Bayes_416040
    match = re.search(r'(\d{6}[A-Z]?)', filename)
    if match:
        return match.group(1)
    
    # Fallback to filename
    return filename.replace('flike_Bayes_Out_', '').replace('flike_Bayes_', '')


def parse_flike_file(filepath: Path) -> Optional[ValidationData]:
    """
    Parse a FLIKE output file to extract validation data.
    
    Parameters
    ----------
    filepath : Path
        Path to the flike_bayes file
        
    Returns
    -------
    ValidationData or None
        Extracted validation data, or None if parsing fails
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        station_id = extract_station_id(filepath)
        logger.info(f"Processing {station_id}")
        
        # Extract flood model
        model_match = re.search(r'Flood model:\s*(.+?)(?:\s{2,}|\n)', content)
        flood_model = model_match.group(1).strip() if model_match else "Unknown"
        
        # Extract zero flow threshold
        threshold_match = re.search(r'Zero flow threshold:\s*([\d.]+)', content)
        zero_threshold = float(threshold_match.group(1)) if threshold_match else 0.0
        
        # Extract gauged annual maximum discharge data
        gauged_flows = extract_gauged_flows(content)
        
        # Extract censored flows
        censored_flows = extract_censored_flows(content)
        
        if gauged_flows.empty:
            logger.warning(f"No gauged flows found in {filepath.name}")
            return None
        
        return ValidationData(
            station_id=station_id,
            gauged_flows=gauged_flows,
            censored_flows=censored_flows,
            zero_flow_threshold=zero_threshold,
            flood_model=flood_model,
            n_gauged=len(gauged_flows),
            n_censored=len(censored_flows)
        )
        
    except Exception as e:
        logger.error(f"Failed to parse {filepath.name}: {e}")
        return None


def extract_gauged_flows(content: str) -> pd.DataFrame:
    """
    Extract gauged annual maximum discharge data.
    
    Expected format:
    Gauged Annual Maximum Discharge Data
     Obs   Discharge Year AEP plot AEP
                          position 1 in Y yrs
     ----------------------------------------
       1     1894.38 2012  0.97872      47.00
       2     1221.90 2023  0.94326      17.63
    """
    # Find the section
    pattern = r'Gauged Annual Maximum Discharge Data.*?-{10,}(.*?)(?=The following gauged flows were censored:|Flood model:|$)'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return pd.DataFrame()
    
    section = match.group(1)
    
    # Parse each line
    data = []
    for line in section.strip().split('\n'):
        # Match pattern: obs discharge year [optional: aep_plot aep_value]
        parts = line.split()
        if len(parts) >= 3:
            try:
                obs = int(parts[0])
                discharge = float(parts[1])
                year = int(parts[2])
                
                data.append({
                    'obs': obs,
                    'discharge': discharge,
                    'year': year
                })
            except (ValueError, IndexError):
                continue
    
    return pd.DataFrame(data)


def extract_censored_flows(content: str) -> pd.DataFrame:
    """
    Extract censored flow observations.
    
    Expected format:
    The following gauged flows were censored:
     Obs   Discharge Year
     --------------------
      29        0.00 1994
      30        0.00 1995
    """
    # Find the section
    pattern = r'The following gauged flows were censored:.*?-{10,}(.*?)(?=Flood model:|$)'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return pd.DataFrame()
    
    section = match.group(1)
    
    # Parse each line
    data = []
    for line in section.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 3:
            try:
                obs = int(parts[0])
                discharge = float(parts[1])
                year = int(parts[2])
                
                data.append({
                    'obs': obs,
                    'discharge': discharge,
                    'year': year
                })
            except (ValueError, IndexError):
                continue
    
    return pd.DataFrame(data)


def extract_all_validation_data(unit_tests_dir: Path) -> Dict[str, ValidationData]:
    """
    Extract validation data from all flike_bayes files in directory.
    
    Parameters
    ----------
    unit_tests_dir : Path
        Directory containing flike_bayes files
        
    Returns
    -------
    Dict[str, ValidationData]
        Dictionary mapping station_id to ValidationData
    """
    validation_data = {}
    
    # Find all flike_bayes files (excluding _Model files for now)
    pattern = 'flike_Bayes*.txt'
    files = list(unit_tests_dir.glob(pattern))
    
    # Filter out _Model files
    files = [f for f in files if '_Model' not in f.name and 'Pars' not in f.name]
    
    logger.info(f"Found {len(files)} flike_bayes files to process")
    
    for filepath in sorted(files):
        data = parse_flike_file(filepath)
        if data:
            validation_data[data.station_id] = data
            logger.info(f"  {data.station_id}: {data.n_gauged} gauged, {data.n_censored} censored")
    
    return validation_data


def create_test_database(validation_data: Dict[str, ValidationData], 
                         output_dir: Path) -> None:
    """
    Create a database of test cases for MGBT validation.
    
    Parameters
    ----------
    validation_data : Dict[str, ValidationData]
        Extracted validation data
    output_dir : Path
        Directory to save test database
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary CSV
    summary_data = []
    for station_id, data in validation_data.items():
        summary_data.append({
            'station_id': station_id,
            'n_gauged': data.n_gauged,
            'n_censored': data.n_censored,
            'n_total': data.n_gauged + data.n_censored,
            'zero_threshold': data.zero_flow_threshold,
            'flood_model': data.flood_model
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'validation_summary.csv', index=False)
    logger.info(f"Saved summary to {output_dir / 'validation_summary.csv'}")
    
    # Save individual station data
    for station_id, data in validation_data.items():
        station_dir = output_dir / station_id
        station_dir.mkdir(exist_ok=True)
        
        # Combine gauged and censored flows
        all_flows = pd.concat([
            data.gauged_flows.assign(censored=False),
            data.censored_flows.assign(censored=True)
        ], ignore_index=True).sort_values('year')
        
        all_flows.to_csv(station_dir / 'annual_maxima.csv', index=False)
        
        # Save just the flow values for easy MGBT input
        flows_array = all_flows['discharge'].values
        np.savetxt(station_dir / 'flows.txt', flows_array, fmt='%.2f')
        
        # Save metadata
        with open(station_dir / 'metadata.txt', 'w') as f:
            f.write(f"Station ID: {station_id}\n")
            f.write(f"Flood Model: {data.flood_model}\n")
            f.write(f"Zero Flow Threshold: {data.zero_flow_threshold}\n")
            f.write(f"Gauged Flows: {data.n_gauged}\n")
            f.write(f"Censored Flows: {data.n_censored}\n")
            f.write(f"Expected Censored Count (R MGBT): {data.n_censored}\n")
    
    logger.info(f"Saved {len(validation_data)} station datasets to {output_dir}")


def main():
    """Main execution function."""
    # Set up paths
    repo_root = Path(__file__).parent.parent
    unit_tests_dir = repo_root / 'UnitTests'
    output_dir = repo_root / 'data' / 'validation'
    
    logger.info(f"Repository root: {repo_root}")
    logger.info(f"Unit tests directory: {unit_tests_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    if not unit_tests_dir.exists():
        logger.error(f"Unit tests directory not found: {unit_tests_dir}")
        return
    
    # Extract validation data
    validation_data = extract_all_validation_data(unit_tests_dir)
    
    if not validation_data:
        logger.error("No validation data extracted")
        return
    
    logger.info(f"\nExtracted data from {len(validation_data)} stations")
    
    # Create test database
    create_test_database(validation_data, output_dir)
    
    logger.info("\nValidation data extraction complete")
    logger.info(f"Test database created in: {output_dir}")


if __name__ == '__main__':
    main()
