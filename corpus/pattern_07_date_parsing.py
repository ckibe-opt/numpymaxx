"""
Pattern 7: Date string parsing and filtering
Common pattern: "Extract date components and filter by month/year"
Source type: r/pandas / time series questions
"""
import pandas as pd
import numpy as np
from datetime import datetime


def slow_date_filter(df, target_month=6):
    """Filter rows where month matches target - slow string parsing."""
    matching_indices = []
    for idx, row in df.iterrows():
        date_str = row['date_str']
        # Parse string like "2024-06-15" and extract month
        parts = date_str.split('-')
        month = int(parts[1])
        if month == target_month:
            matching_indices.append(idx)
    return matching_indices


def _numpymaxx_inputs():
    """Fixture for date filter test."""
    rng = np.random.default_rng(7)
    n = 5000
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    df = pd.DataFrame({
        'date_str': [d.strftime('%Y-%m-%d') for d in dates],
        'value': rng.standard_normal(n),
    })
    return ([df, 6], {})
