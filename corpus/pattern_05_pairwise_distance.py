"""
Pattern 5: Pairwise distance calculation
Common pattern: "Calculate distance between consecutive points"
Source type: r/datascience / geospatial analysis
"""
import pandas as pd
import numpy as np


def slow_pairwise_distance(df):
    """Calculate Euclidean distance between consecutive points - slow loop."""
    distances = []
    for idx in range(len(df) - 1):
        x1, y1 = df.iloc[idx]['x'], df.iloc[idx]['y']
        x2, y2 = df.iloc[idx + 1]['x'], df.iloc[idx + 1]['y']
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances.append(dist)
    return distances


def _numpymaxx_inputs():
    """Fixture for pairwise distance test."""
    rng = np.random.default_rng(5)
    n = 3000
    df = pd.DataFrame({
        'x': np.cumsum(rng.standard_normal(n)),  # Random walk
        'y': np.cumsum(rng.standard_normal(n)),
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='min'),
    })
    return ([df], {})
