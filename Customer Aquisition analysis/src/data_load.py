"""Functions to load and save datasets."""
import pandas as pd
from pathlib import Path

# Define DATA_DIR relative to project root
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_raw(path: str = None) -> pd.DataFrame:
    """
    Load the raw customer acquisition CSV.
    
    Parameters:
    - path: optional path to CSV. If None, defaults to DATA_DIR/raw/customer_acquisition.csv
    
    Returns:
    - pd.DataFrame
    """
    p = DATA_DIR / "raw" / "customer_acquisition.csv" if path is None else Path(path)
    data = pd.read_csv(p, low_memory=False)
    return data


def save_processed(data: pd.DataFrame, filename: str = "processed.csv") -> None:
    """
    Save a processed dataframe to the processed folder.
    
    Parameters:
    - df: DataFrame to save
    - filename: CSV file name (default: processed.csv)
    """
    proc_dir = DATA_DIR / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(proc_dir / filename, index=False)
