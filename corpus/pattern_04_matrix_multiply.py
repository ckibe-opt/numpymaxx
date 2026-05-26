"""
Pattern 4: NumPy nested loop matrix multiplication
Common pattern: "Matrix multiplication without @ operator"
Source type: r/learnpython / numpy tutorials
"""
import numpy as np


def slow_matrix_multiply(A, B):
    """Multiply two matrices using nested loops (O(n³) slow)."""
    n = A.shape[0]
    m = B.shape[1]
    p = A.shape[1]
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i, j] += A[i, k] * B[k, j]
    return C


def _numpymaxx_inputs():
    """Fixture for matrix multiply test (smaller size due to O(n³))."""
    rng = np.random.default_rng(4)
    A = rng.standard_normal((50, 50))
    B = rng.standard_normal((50, 50))
    return ([A, B], {})
