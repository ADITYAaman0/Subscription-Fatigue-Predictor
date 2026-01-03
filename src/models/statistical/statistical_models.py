"""
Statistical models for change point detection and causality analysis.
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from statsmodels.tsa.stattools import grangercausalitytests


class ChangePointDetector:
    """
    Detects structural breaks in time series using multiple algorithms.
    """
    
    def __init__(self, min_size=30):
        self.min_size = min_size
        self.change_points = []
    
    def pelt_detection(self, series, penalty=10):
        """
        PELT (Pruned Exact Linear Time) - Fast and accurate.
        
        Args:
            series: pandas Series with DatetimeIndex
            penalty: Higher = fewer breakpoints detected
        """
        # Normalize the data
        data_normalized = (series.values - np.mean(series)) / (np.std(series) + 1e-8)
        
        # Detect change points
        algo = rpt.Pelt(min_size=self.min_size).fit(data_normalized.reshape(-1, 1))
        change_points = algo.predict(pen=penalty)
        
        # Convert indices to dates
        if len(change_points) > 0:
            change_dates = [series.index[cp - 1] for cp in change_points[:-1]]
        else:
            change_dates = []
        
        return change_dates
    
    def binary_segmentation(self, series, n_bkps=3):
        """
        Binary Segmentation - Faster but less accurate.
        """
        data_normalized = (series.values - np.mean(series)) / (np.std(series) + 1e-8)
        
        algo = rpt.Binseg(min_size=self.min_size).fit(data_normalized.reshape(-1, 1))
        change_points = algo.predict(n_bkps=n_bkps)
        
        if len(change_points) > 0:
            change_dates = [series.index[cp - 1] for cp in change_points[:-1]]
        else:
            change_dates = []
        
        return change_dates
    
    def detect_changes(self, series, method='pelt', penalty=10, n_bkps=3):
        """
        Main method to detect changes.
        """
        if method == 'pelt':
            return self.pelt_detection(series, penalty)
        elif method == 'binseg':
            return self.binary_segmentation(series, n_bkps)
        else:
            raise ValueError(f"Unknown method: {method}")


class CausalAnalyzer:
    """
    Analyzes correlation between price changes and churn signals.
    """
    
    def __init__(self, pricing_df, search_trends_df):
        """
        Args:
            pricing_df: DataFrame with 'date' and 'price' columns
            search_trends_df: DataFrame with 'date' and 'search_volume' columns
        """
        self.pricing = pricing_df.sort_values('date')
        self.trends = search_trends_df.sort_values('date')
    
    def create_lagged_features(self, max_lag=8):
        """Create lagged versions of search volume."""
        merged = pd.merge(
            self.pricing[['date', 'price']],
            self.trends[['date', 'search_volume']],
            on='date', how='outer'
        ).sort_values('date')
        
        merged['price_change'] = merged['price'].diff()
        
        # Create lags
        for lag in range(0, max_lag + 1):
            merged[f'search_volume_lag_{lag}'] = merged['search_volume'].shift(lag)
        
        return merged.fillna(method='ffill').fillna(method='bfill')
    
    def find_optimal_lag(self):
        """Find lag with strongest correlation between price and search volume."""
        lagged = self.create_lagged_features()
        lagged = lagged.dropna()
        
        correlations = {}
        
        for lag in range(0, 9):
            if f'search_volume_lag_{lag}' in lagged.columns:
                corr = lagged['price_change'].corr(lagged[f'search_volume_lag_{lag}'])
                correlations[lag] = corr if not np.isnan(corr) else 0
        
        if correlations:
            optimal_lag = max(correlations, key=lambda x: abs(correlations[x]))
        else:
            optimal_lag = 0
        
        return {
            'optimal_lag_weeks': optimal_lag,
            'correlation': correlations.get(optimal_lag, 0),
            'all_correlations': correlations
        }
