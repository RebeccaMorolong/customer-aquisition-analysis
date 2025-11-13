"""Data cleaning utilities."""

import pandas as pd
import numpy as np


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [c.strip().lower().replace(' ', '_') for c in data.columns]
    return data


def parse_dates(data: pd.DataFrame, date_cols: list = None) -> pd.DataFrame:
    data = data.copy()
    if date_cols is None:
        candidates = [c for c in data.columns if 'date' in c or 'joined' in c or 'acq' in c]
    else:
        candidates = date_cols
    for c in candidates:
        try:
            data[c] = pd.to_datetime(data[c], errors='coerce')
        except Exception:
            pass
    return data


def drop_duplicates(data: pd.DataFrame, id_col: str = 'customer_id') -> pd.DataFrame:
    data= data.copy()
    if id_col in data.columns:
        data = data.drop_duplicates(subset=id_col)
    return data


def impute_numeric(data: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    data = data.copy()
    if cols is None:
        cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for c in cols:
        data[c] = data[c].fillna(data[c].median())
    return data





