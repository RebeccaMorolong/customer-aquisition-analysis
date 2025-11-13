"""Small helper functions."""

import pandas as pd


def ensure_columns(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    data = data.copy()
    for c in cols:
        if c not in data.columns:
            data[c] = None
    return data

