"""
Pattern 2: String filtering with iterrows
Common pattern: "Filter dataframe rows where string column contains substring"
Source type: r/pandas / SO
"""
import pandas as pd
import numpy as np


def slow_filter_by_substring(df, substring="error"):
    """Filter rows where 'message' contains substring - slow iterrows version."""
    matching = []
    for idx, row in df.iterrows():
        message = row['message']
        if substring.lower() in message.lower():
            matching.append(message)
    return matching


def _numpymaxx_inputs():
    """Fixture for string filter test."""
    rng = np.random.default_rng(2)
    messages = [
        "INFO: Process started successfully",
        "ERROR: Connection failed",
        "WARNING: Low memory",
        "DEBUG: Variable x = 42",
        "ERROR: Database timeout",
        "INFO: Task completed",
    ]
    df = pd.DataFrame({
        'message': rng.choice(messages, 3000),
        'timestamp': pd.date_range('2024-01-01', periods=3000, freq='s'),
    })
    return ([df, "error"], {})
