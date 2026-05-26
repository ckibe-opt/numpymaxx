"""
Pattern 1: Cumulative sum with repeated indexing
Common Stack Overflow pattern: "How to calculate running total in pandas"
Source type: r/learnpython / SO beginner question
"""
import pandas as pd
import numpy as np


def slow_cumulative_sum(df):
    """Calculate cumulative sum of 'value' column using loop (slow)."""
    result = []
    running_total = 0
    for idx, row in df.iterrows():
        running_total += row['value']
        result.append(running_total)
    return result


def _numpymaxx_inputs():
    """Fixture for cumulative sum test."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        'value': rng.standard_normal(5000),
        'id': range(5000),
    })
    return ([df], {})
