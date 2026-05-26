"""
Pattern 6: Group-wise normalization
Common pattern: "Normalize values within each group"
Source type: r/pandas / data preprocessing questions
"""
import pandas as pd
import numpy as np


def slow_group_normalize(df):
    """Normalize 'value' within each 'group' using loop - slow version."""
    normalized = []
    for idx, row in df.iterrows():
        group = row['group']
        value = row['value']
        # Calculate mean and std for this group (expensive repeated calc)
        group_mean = df[df['group'] == group]['value'].mean()
        group_std = df[df['group'] == group]['value'].std()
        norm_value = (value - group_mean) / (group_std + 1e-10)
        normalized.append(norm_value)
    return normalized


def _numpymaxx_inputs():
    """Fixture for group normalization test."""
    rng = np.random.default_rng(6)
    n = 2500
    df = pd.DataFrame({
        'group': rng.choice(['A', 'B', 'C', 'D', 'E'], n),
        'value': rng.standard_normal(n),
    })
    return ([df], {})
