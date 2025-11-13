"""Package initializer for src."""

from .cleaning import normalize_columns, parse_dates, drop_duplicates, impute_numeric
from .features import add_acquisition_features, compute_first_purchase_metrics, bucketize_features
from .metrics import calc_cac, ltv_by_cohort
from .modeling import train_model, predict
from .helpers import ensure_columns

__all__ = [
    'normalize_columns', 'parse_dates', 'drop_duplicates', 'impute_numeric',
    'add_acquisition_features', 'compute_first_purchase_metrics', 'bucketize_features',
    'calc_cac', 'ltv_by_cohort', 'train_model', 'predict', 'ensure_columns'
]


