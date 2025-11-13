"""Small helper functions."""

import pandas as pd


def ensure_columns(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Ensures that the DataFrame contains all columns in `cols`.
    Adds missing columns with None values.
    
    Parameters:
    - df: input DataFrame
    - cols: list of column names to ensure
    
    Returns:
    - DataFrame with all specified columns
    """
    data = data.copy()
    
    for c in cols:
        if c not in data.columns:
            data[c] = None
    
    return data
