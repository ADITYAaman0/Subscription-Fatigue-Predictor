"""
Helper functions and utilities.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def normalize_series(series):
    """Normalize a pandas Series to [0, 1]."""
    return (series - series.min()) / (series.max() - series.min())


def standardize_series(series):
    """Standardize a pandas Series (z-score normalization)."""
    return (series - series.mean()) / series.std()


def calculate_pct_change(old_value, new_value):
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 0
    return ((new_value - old_value) / abs(old_value)) * 100


def get_date_ranges(start_date, end_date, freq='M'):
    """Generate date ranges for analysis."""
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def create_lagged_features(df, column, lags=8):
    """Create lagged features for time series."""
    for lag in range(1, lags + 1):
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df


def handle_missing_values(df, method='forward_fill'):
    """Handle missing values in dataframe."""
    if method == 'forward_fill':
        return df.fillna(method='ffill')
    elif method == 'backward_fill':
        return df.fillna(method='bfill')
    elif method == 'mean':
        return df.fillna(df.mean())
    else:
        return df.dropna()


def identify_outliers(series, method='iqr', threshold=1.5):
    """Identify outliers using IQR method or z-score."""
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    else:  # z-score
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold


def get_last_n_weeks(df, date_column, n_weeks=52):
    """Filter dataframe to last n weeks."""
    cutoff_date = datetime.now() - timedelta(weeks=n_weeks)
    return df[pd.to_datetime(df[date_column]) >= cutoff_date]
