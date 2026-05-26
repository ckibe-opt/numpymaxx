"""
Pattern 8: Rolling/moving average calculation
Common pattern: "Calculate moving average with custom window"
Source type: r/learnpython / SO "pandas rolling window without rolling()"
"""
import pandas as pd
import numpy as np


def slow_rolling_average(values, window=5):
    """Calculate rolling average using explicit loop - slow version."""
    result = []
    for i in range(len(values)):
        # Calculate window boundaries
        start = max(0, i - window + 1)
        end = i + 1
        # Sum values in window
        window_sum = 0
        for j in range(start, end):
            window_sum += values[j]
        avg = window_sum / (end - start)
        result.append(avg)
    return result


def _numpymaxx_inputs():
    """Fixture for rolling average test."""
    rng = np.random.default_rng(8)
    values = rng.standard_normal(2000)
    return ([values, 10], {})
