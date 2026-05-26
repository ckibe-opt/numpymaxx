"""
Pattern 3: Conditional new column assignment
Common pattern: "Create new column based on condition in another column"
Source type: r/learnpython
"""
import pandas as pd
import numpy as np


def slow_conditional_column(df):
    """Create 'category' column based on 'score' value - slow loop version."""
    categories = []
    for idx, row in df.iterrows():
        score = row['score']
        if score < 30:
            categories.append('low')
        elif score < 70:
            categories.append('medium')
        else:
            categories.append('high')
    return categories


def _numpymaxx_inputs():
    """Fixture for conditional column test."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        'score': rng.uniform(0, 100, 4000),
        'name': [f'item_{i}' for i in range(4000)],
    })
    return ([df], {})
