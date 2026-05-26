import pandas as pd
import numpy as np


def slow_filter_and_square(df):
    """Filter rows where x > 0 and square the values - using iterrows (slow)."""
    result = []
    for idx, row in df.iterrows():
        if row['x'] > 0:
            result.append(row['x'] ** 2)
    return result


def slow_string_processing(df):
    """Process strings using iterrows (slow)."""
    processed = []
    for idx, row in df.iterrows():
        text = row['message']
        if 'error' in text.lower():
            processed.append(text.upper())
    return processed


def _numpymaxx_inputs():
    """Fixture: inputs for slow_filter_and_square (default function)."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'x': rng.standard_normal(2000),
        'y': rng.standard_normal(2000),
    })
    return ([df], {})
