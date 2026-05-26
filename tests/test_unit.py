"""
Unit tests for NumpyMaxx — no LLM calls required.
Covers: extract_function, detect_pattern, extract_code, results_match,
        _synthesize_inputs, exec_and_resolve.
"""

import sys
import os
import tempfile
import textwrap
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from numpymaxx.cli import (
    extract_function,
    detect_pattern,
    extract_code,
    results_match,
    exec_and_resolve,
    _synthesize_inputs,
    build_prompt,
)


# ------------------------------------------------------------------ #
# extract_function
# ------------------------------------------------------------------ #
def _write_tmp(code: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    f.write(textwrap.dedent(code))
    f.close()
    return f.name


def test_extract_function_by_name():
    path = _write_tmp("""
        def slow_thing(df):
            return df
        def other(x):
            return x
    """)
    src, name = extract_function(path, "slow_thing")
    assert name == "slow_thing"
    assert "def slow_thing" in src
    os.unlink(path)


def test_extract_function_first_public():
    path = _write_tmp("""
        def _private(x):
            return x
        def public_one(df):
            return df
    """)
    src, name = extract_function(path, None)
    assert name == "public_one"
    os.unlink(path)


def test_extract_function_not_found_raises():
    path = _write_tmp("def foo(x): return x\n")
    with pytest.raises(ValueError, match="missing_fn"):
        extract_function(path, "missing_fn")
    os.unlink(path)


# ------------------------------------------------------------------ #
# detect_pattern
# ------------------------------------------------------------------ #
def test_detect_pattern_iterrows():
    src = "def f(df):\n  for idx, row in df.iterrows():\n    pass\n"
    assert detect_pattern(src) == "pandas_iterrows"


def test_detect_pattern_string_loop():
    src = "def f(df):\n  for idx, row in df.iterrows():\n    if 'error' in row['message'].lower():\n      pass\n"
    assert detect_pattern(src) == "string_loop"


def test_detect_pattern_numpy():
    src = "def f(A):\n  for i in range(len(A)):\n    np.sqrt(A[i])\n"
    assert detect_pattern(src) == "numpy_matrix"


# ------------------------------------------------------------------ #
# extract_code
# ------------------------------------------------------------------ #
def test_extract_code_with_fence():
    raw = "Here is the answer:\n```python\ndef foo(x):\n    return x * 2\n```"
    code = extract_code(raw)
    assert code.startswith("def foo")


def test_extract_code_bare_fence():
    raw = "```\ndef foo(x):\n    return x\n```"
    code = extract_code(raw)
    assert "def foo" in code


def test_extract_code_no_fence():
    raw = "def foo(x):\n    return x"
    assert extract_code(raw) == raw.strip()


# ------------------------------------------------------------------ #
# results_match
# ------------------------------------------------------------------ #
def test_results_match_lists():
    assert results_match([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert not results_match([1.0, 2.0], [1.0, 3.0])


def test_results_match_numpy():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0 + 1e-8])
    assert results_match(a, b)
    assert not results_match(a, np.array([1.0, 2.0, 4.0]))


def test_results_match_dataframe():
    df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    df2 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert results_match(df1, df2)


def test_results_match_dataframe_shape_mismatch():
    df1 = pd.DataFrame({"a": [1.0, 2.0]})
    df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert not results_match(df1, df2)


def test_results_match_scalar():
    assert results_match(42, 42)
    assert not results_match(42, 43)


# ------------------------------------------------------------------ #
# exec_and_resolve
# ------------------------------------------------------------------ #
def test_exec_and_resolve_same_name():
    code = "import numpy as np\ndef fast_fn(x):\n    return x * 2\n"
    func = exec_and_resolve(code, "fast_fn")
    assert func(3) == 6


def test_exec_and_resolve_renamed():
    # LLM renamed the function — should still be found as the only top-level def
    code = "def optimized_fn(x):\n    return x + 1\n"
    func = exec_and_resolve(code, "original_fn")
    assert func(10) == 11


def test_exec_and_resolve_not_found_raises():
    code = "x = 42\n"
    with pytest.raises(ValueError):
        exec_and_resolve(code, "missing")


# ------------------------------------------------------------------ #
# _synthesize_inputs
# ------------------------------------------------------------------ #
def test_synthesize_inputs_df_param():
    src = "def f(df):\n    return df\n"
    args, kwargs = _synthesize_inputs(src, n=100)
    assert len(args) == 1
    assert isinstance(args[0], pd.DataFrame)
    assert len(args[0]) == 100


def test_synthesize_inputs_matrix_param():
    src = "def f(A, B):\n    return A @ B\n"
    args, kwargs = _synthesize_inputs(src, n=100)
    assert len(args) == 2
    assert isinstance(args[0], np.ndarray)
    assert args[0].ndim == 2


def test_synthesize_inputs_unknown_param_raises():
    src = "def f(zorblax):\n    return zorblax\n"
    with pytest.raises(RuntimeError, match="_numpymaxx_inputs"):
        _synthesize_inputs(src, n=100)


# ------------------------------------------------------------------ #
# build_prompt
# ------------------------------------------------------------------ #
def test_build_prompt_contains_hint():
    src = "def f(df):\n    for idx, row in df.iterrows():\n        pass\n"
    prompt = build_prompt(src, "f", "pandas_iterrows")
    assert "iterrows" in prompt
    assert "def f" in prompt
    assert "f" in prompt


def test_build_prompt_contains_function_name_constraint():
    src = "def slow_fn(df):\n    pass\n"
    prompt = build_prompt(src, "slow_fn", "pandas_iterrows")
    assert "slow_fn" in prompt
