import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

def calculate_health_logic(df):
    """Refined logic extracted from dashboard.py for testing."""
    if df.empty:
        return 0, 0
    
    # Deduplication logic
    if 'table_name' in df.columns and 'ingestion_timestamp' in df.columns:
        latest = df.sort_values('ingestion_timestamp', ascending=False).drop_duplicates('table_name')
    else:
        latest = df
        
    source_counts = latest['source_type'].value_counts()
    total = source_counts.sum()
    
    synthetic = source_counts.get('synthetic', 0)
    real = total - synthetic
    
    real_pct = (real / total * 100) if total > 0 else 0
    syn_pct = (synthetic / total * 100) if total > 0 else 0
    
    return real_pct, syn_pct

def test_data_health_deduplication():
    # Setup data: 1 real table, 1 synthetic table, BUT synthetic has 2 records (one old failure, one new)
    now = datetime.now()
    data = [
        {'table_name': 'pricing', 'source_type': 'real', 'ingestion_timestamp': now - timedelta(days=2)},
        {'table_name': 'metrics', 'source_type': 'synthetic', 'ingestion_timestamp': now - timedelta(days=1)},
        {'table_name': 'metrics', 'source_type': 'synthetic', 'ingestion_timestamp': now}, # Latest for metrics
    ]
    df = pd.DataFrame(data)
    
    # Without deduplication: 1 real, 2 synthetic -> 33% real
    # With deduplication: 1 real (pricing), 1 synthetic (metrics latest) -> 50% real
    
    real_pct, syn_pct = calculate_health_logic(df)
    
    assert real_pct == 50.0
    assert syn_pct == 50.0

def test_data_health_empty():
    df = pd.DataFrame()
    real_pct, syn_pct = calculate_health_logic(df)
    assert real_pct == 0
    assert syn_pct == 0

def test_data_health_all_real():
    data = [
        {'table_name': 't1', 'source_type': 'real', 'ingestion_timestamp': datetime.now()},
        {'table_name': 't2', 'source_type': 'kaggle', 'ingestion_timestamp': datetime.now()},
    ]
    df = pd.DataFrame(data)
    real_pct, _ = calculate_health_logic(df)
    assert real_pct == 100.0

if __name__ == "__main__":
    pytest.main([__file__])
