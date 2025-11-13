"""Business metrics: CAC, LTV, retention."""
import pandas as pd


def calc_cac(
    spend_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    campaign_col: str = 'campaign',
    cost_col: str = 'spend',
    customer_col: str = 'customer_id'
) -> pd.DataFrame:
    """
    Calculate Customer Acquisition Cost (CAC) per campaign.

    Parameters:
    - spend_df: DataFrame with campaign spend data
    - customers_df: DataFrame with customer acquisition data
    - campaign_col: column name for campaign
    - cost_col: column name for spend/cost
    - customer_col: column name for customer ID

    Returns:
    - DataFrame with CAC per campaign
    """
    s = spend_df.groupby(campaign_col).agg(campaign_spend=(cost_col,'sum')).reset_index()
    c = customers_df.groupby(campaign_col).agg(new_customers=(customer_col,'nunique')).reset_index()
    out = s.merge(c, on=campaign_col, how='left')
    out['CAC'] = out['campaign_spend'] / out['new_customers']
    return out


def ltv_by_cohort(
    trans_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    cohort_period: str = 'M',
    window_days: int = 180,
    customer_col: str = 'customer_id',
    trans_amount_col: str = 'amount',
    trans_date_col: str = 'transaction_date',
    acquisition_date_col: str = 'acquisition_date'
) -> pd.DataFrame:
    """
    Calculate LTV per customer cohort within a time window.

    Parameters:
    - trans_df: transactions DataFrame
    - customers_df: customer DataFrame
    - cohort_period: 'M' for month, 'W' for week, etc.
    - window_days: number of days to include for each customer
    - customer_col: column name for customer ID
    - trans_amount_col: column for transaction amounts
    - trans_date_col: column for transaction date
    - acquisition_date_col: column for acquisition date

    Returns:
    - DataFrame with LTV per cohort
    """
    trans = trans_df.copy()
    trans[trans_date_col] = pd.to_datetime(trans[trans_date_col], errors='coerce')
    customers_df[acquisition_date_col] = pd.to_datetime(customers_df[acquisition_date_col], errors='coerce')

    # Create cohort
    customers_df['cohort'] = customers_df[acquisition_date_col].dt.to_period(cohort_period).astype(str)

    merged = trans.merge(customers_df[[customer_col, 'cohort']], on=customer_col, how='left')
    merged = merged.merge(customers_df[[customer_col, acquisition_date_col]], on=customer_col, how='left')

    merged['days_since_acq'] = (merged[trans_date_col] - merged[acquisition_date_col]).dt.days
    merged = merged[merged['days_since_acq'].between(0, window_days)]

    ltv = merged.groupby('cohort').agg(
        total_revenue=(trans_amount_col,'sum'),
        n_customers=(customer_col,'nunique')
    ).reset_index()
    ltv['ltv_per_customer'] = ltv['total_revenue'] / ltv['n_customers']
    return ltv

