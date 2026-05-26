#!/usr/bin/env python3
"""
NumpyMaxx CLI — single-shot LLM vectorizer.

Usage:
    numpymaxx optimize <file.py> [--function NAME] [--n N] [--output out.py]
    python -m numpymaxx <same args>

How it works:
    1. Parse the target function from the user's file.
    2. Build real benchmark inputs (_numpymaxx_inputs() or auto-synthesized).
    3. Time the original (median of 5 runs after 2 warmups).
    4. Single LLM call: domain hint + user code -> vectorized version.
    5. Validate correctness and time the result.
    6. Report speedup.

No evolution loop, no SQLite database, no subprocess.  ~5-15s total.
"""

import sys
import os
import ast
import re
import time
import json
import argparse
import importlib.util
import types
import urllib.request
import urllib.error
import statistics

# Windows console UTF-8 safety
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# .env loader (optional — graceful if python-dotenv not installed)
# ------------------------------------------------------------------ #
def _load_env():
    """Load .env from cwd, script dir, or any parent (up to 6 levels)."""
    try:
        from dotenv import load_dotenv
        # Try cwd and its parents first
        search_dirs = []
        cwd = os.getcwd()
        for _ in range(6):
            search_dirs.append(cwd)
            parent = os.path.dirname(cwd)
            if parent == cwd:
                break
            cwd = parent
        # Also try the directory containing this file and its parents
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):
            if here not in search_dirs:
                search_dirs.append(here)
            parent = os.path.dirname(here)
            if parent == here:
                break
            here = parent
        for d in search_dirs:
            candidate = os.path.join(d, ".env")
            if os.path.exists(candidate):
                load_dotenv(candidate)
                return
    except ImportError:
        pass

_load_env()

# ------------------------------------------------------------------ #
# LLM configuration (reads from environment / .env)
# ------------------------------------------------------------------ #
_LLM_PROVIDER   = os.environ.get("LLM_PROVIDER",      "openrouter").lower()
_OR_MODEL       = os.environ.get("OPENROUTER_MODEL",   "openai/gpt-oss-120b:free")
_OR_API_KEY     = os.environ.get("OPENROUTER_API_KEY", "")
_OLLAMA_URL     = os.environ.get("OLLAMA_API_URL",     "http://localhost:11434/api/generate")
_OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL",       "llama3")
_OR_THROTTLE    = os.path.join(os.getcwd(), ".or_last_call")
_OR_MIN_GAP     = 2.1   # seconds between OpenRouter calls

_openrouter_client = None
if _LLM_PROVIDER == "openrouter" and _OR_API_KEY:
    try:
        from openai import OpenAI as _OpenAI
        _openrouter_client = _OpenAI(
            api_key=_OR_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
        print(f"[SYSTEM] OpenRouter provider active: {_OR_MODEL}")
    except ImportError:
        print("[SYSTEM] openai package not found — falling back to Ollama")


# ------------------------------------------------------------------ #
# Domain hints (vectorization-specific, inlined from prompts.py)
# ------------------------------------------------------------------ #
_DOMAIN_HINTS = {
    "pandas_iterrows": """DOMAIN HINT (Pandas loop optimization):
- df.iterrows() is the slowest way to iterate. Replace with vectorized operations.
- Boolean indexing: df[df['column'] > 0] beats explicit loop+if checks.
- .apply(lambda) is often slower than direct vectorized math operations.
- String operations: use .str accessor (df['col'].str.contains('error')) not Python string methods in loop.
- Aggregation: groupby().agg() or direct column operations beat manual dict accumulation.
- Example:
  OLD: for idx, row in df.iterrows(): if row['x'] > 0: result.append(row['x'] ** 2)
  NEW: (df.loc[df['x'] > 0, 'x'] ** 2).tolist()""",

    "numpy_matrix": """DOMAIN HINT (NumPy loop optimization):
- Replace nested loops with broadcasting: a[:, None] * b[None, :] for outer products.
- Use np.where(condition, a, b) instead of if/else in loops.
- Use ufuncs (np.add, np.multiply, np.sqrt) instead of Python operators in loops.
- Matrix multiplication: use np.dot() or @ operator, not manual triple loops.
- Example:
  OLD: for i in range(n): for j in range(m): C[i,j] = A[i,j] * B[i,j]
  NEW: C = A * B  (element-wise) or  C = A @ B  (matrix multiply)""",

    "string_loop": """DOMAIN HINT (String loop optimization):
- df.iterrows() with string operations is extremely slow.
- Use .str accessor for vectorized string operations: df['col'].str.contains(), .str.upper(), .str.lower()
- Boolean masking: mask = df['col'].str.contains('pattern', case=False, na=False)
- Example:
  OLD: for idx, row in df.iterrows(): if 'error' in text.lower(): processed.append(text.upper())
  NEW: df.loc[df['message'].str.lower().str.contains('error'), 'message'].str.upper().tolist()""",
}

def _get_domain_hint(pattern: str) -> str:
    for key, hint in _DOMAIN_HINTS.items():
        if key in pattern.lower():
            return "\n" + hint + "\n"
    return ""


# ------------------------------------------------------------------ #
# Step 1: parse user function
# ------------------------------------------------------------------ #
_SYNTHETIC_N_DEFAULT = 1000

def extract_function(filepath: str, func_name: str | None = None):
    """Return (source_str, name) for the requested function."""
    with open(filepath, "r", encoding="utf-8") as fh:
        src = fh.read()

    tree = ast.parse(src)
    lines = src.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                continue
            if func_name is None or node.name == func_name:
                start = node.lineno - 1
                end = node.end_lineno
                return "\n".join(lines[start:end]), node.name

    raise ValueError(
        f"Function '{func_name}' not found in {filepath}"
        if func_name
        else f"No public top-level function found in {filepath}"
    )


# ------------------------------------------------------------------ #
# Step 2: resolve inputs
# ------------------------------------------------------------------ #
def _synthesize_inputs(func_source: str, n: int):
    """Guess inputs from parameter names."""
    tree = ast.parse(func_source)
    func_node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
    )
    params = [a.arg for a in func_node.args.args]

    rng = np.random.default_rng(42)
    args = []
    for p in params:
        pl = p.lower()
        if any(k in pl for k in ("df", "data", "frame", "table")):
            args.append(pd.DataFrame({
                "x":       rng.standard_normal(n),
                "y":       rng.standard_normal(n),
                "score":   rng.integers(0, 100, n).astype(float),
                "value":   rng.standard_normal(n),
                "message": np.where(rng.random(n) > 0.5, "error log", "info log"),
                "group":   np.where(rng.random(n) > 0.5, "A", "B"),
            }))
        elif any(k in pl for k in ("matrix", "mat", "a", "b")):
            m = max(int(n ** 0.5), 10)
            args.append(rng.standard_normal((m, m)))
        elif any(k in pl for k in ("arr", "array", "values", "vec")):
            args.append(rng.standard_normal(n).tolist())
        elif pl in ("n", "size", "length"):
            args.append(n)
        elif pl in ("window", "k", "w"):
            args.append(5)
        else:
            raise RuntimeError(
                f"Cannot synthesize input for parameter '{p}'. "
                f"Add a _numpymaxx_inputs() function to your file."
            )
    return args, {}


def resolve_inputs(filepath: str, func_source: str, func_name: str, n: int):
    """Priority: _numpymaxx_inputs() from file > synthesize from param names."""
    spec = importlib.util.spec_from_file_location("_user_mod", filepath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass

    if hasattr(mod, "_numpymaxx_inputs"):
        args, kwargs = mod._numpymaxx_inputs()
        print(f"[inputs] Using _numpymaxx_inputs() from file (n={len(args)} args).")
        return args, kwargs

    print(f"[inputs] No _numpymaxx_inputs() found — synthesizing from parameter names (n={n}).")
    return _synthesize_inputs(func_source, n)


# ------------------------------------------------------------------ #
# Step 3: time a function
# ------------------------------------------------------------------ #
def time_function(func, args, kwargs, runs=5, warmups=2):
    for _ in range(warmups):
        func(*args, **kwargs)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return result, statistics.median(times)


# ------------------------------------------------------------------ #
# Step 4: detect pattern and build prompt
# ------------------------------------------------------------------ #
def detect_pattern(func_source: str) -> str:
    src = func_source.lower()
    if "iterrows" in src or ("for" in src and ("df" in src or "dataframe" in src)):
        if "str." in src or "string" in src or "contains" in src or "upper" in src or "lower" in src:
            return "string_loop"
        return "pandas_iterrows"
    if "for" in src and ("np." in func_source or "numpy" in src):
        return "numpy_matrix"
    if "iterrows" in src or "apply" in src:
        return "pandas_iterrows"
    return "pandas_iterrows"


def build_prompt(func_source: str, func_name: str, pattern: str) -> str:
    hint = _get_domain_hint(pattern).strip()
    return f"""\
You are an expert Python performance engineer specializing in Pandas and NumPy vectorization.

{hint}

Rewrite the following function to be as fast as possible using vectorized operations.
Keep the function name exactly as '{func_name}'.
Return only the complete Python function inside a single ```python ... ``` code block.
Do not include any explanation outside the code block.

```python
{func_source}
```
"""


# ------------------------------------------------------------------ #
# Step 5: call LLM (OpenRouter -> Ollama fallback, self-contained)
# ------------------------------------------------------------------ #
_or_rate_limited = False


def _or_throttle():
    try:
        if os.path.exists(_OR_THROTTLE):
            with open(_OR_THROTTLE) as f:
                last = float(f.read().strip())
            elapsed = time.time() - last
            if elapsed < _OR_MIN_GAP:
                time.sleep(_OR_MIN_GAP - elapsed)
        with open(_OR_THROTTLE, "w") as f:
            f.write(str(time.time()))
    except Exception:
        pass


def _call_openrouter(prompt: str, temp: float = 0.3, max_attempts: int = 2) -> str:
    global _or_rate_limited
    if _openrouter_client is None or _or_rate_limited:
        return ""
    _or_throttle()
    for attempt in range(max_attempts):
        try:
            print("    [LLM] ", end="", flush=True)
            completion = _openrouter_client.chat.completions.create(
                model=_OR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=1200,
            )
            text = completion.choices[0].message.content or ""
            print(text)
            try:
                with open(_OR_THROTTLE, "w") as f:
                    f.write(str(time.time()))
            except Exception:
                pass
            return text
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                m = re.search(r"'retry_after_seconds':\s*([\d\.]+)", err)
                if not m:
                    m = re.search(r"Please try again in ([\.\d]+)s", err)
                wait = float(m.group(1)) + 2 if m else 30
                wait = min(wait, 90)
                print(f"\n    [WARN] OpenRouter rate limit. Retrying in {wait:.0f}s...")
                time.sleep(wait)
                continue
            if attempt < max_attempts - 1:
                time.sleep(5 * (attempt + 1))
                continue
            print(f"\n    [WARN] OpenRouter failed: {err[:120]}. Trying Ollama...")
            return ""
    _or_rate_limited = True
    return ""


def _call_ollama(prompt: str, temp: float = 0.3) -> str:
    data = json.dumps({
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temp, "num_ctx": 8192, "num_predict": 2048},
    }).encode("utf-8")
    req = urllib.request.Request(
        _OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        chunks = []
        with urllib.request.urlopen(req, timeout=120) as r:
            print("    [LLM] ", end="", flush=True)
            for raw_line in r:
                chunk = json.loads(raw_line.decode("utf-8"))
                token = chunk.get("response", "")
                if token:
                    print(token, end="", flush=True)
                    chunks.append(token)
                if chunk.get("done"):
                    break
        print()
        return "".join(chunks)
    except Exception as e:
        print(f"\n    [ERROR] Ollama call failed: {e}")
        return ""


def call_llm(prompt: str) -> str:
    t0 = time.perf_counter()
    if _LLM_PROVIDER == "openrouter":
        result = _call_openrouter(prompt)
        if result:
            print(f"      LLM responded in {time.perf_counter()-t0:.1f}s")
            return result
        print("[SYSTEM] Falling back to Ollama...")
    result = _call_ollama(prompt)
    print(f"      LLM responded in {time.perf_counter()-t0:.1f}s")
    return result


# ------------------------------------------------------------------ #
# Step 6: extract, exec, and resolve callable
# ------------------------------------------------------------------ #
def extract_code(llm_response: str) -> str:
    """Pull Python source from markdown fences."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", llm_response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return llm_response.strip()


def exec_and_resolve(code: str, preferred_name: str):
    """exec code, return the callable for preferred_name or the only top-level def."""
    ns = {"pd": pd, "np": np, "__builtins__": __builtins__}
    exec(code, ns)
    if preferred_name in ns and callable(ns[preferred_name]):
        return ns[preferred_name]
    # LLM may have renamed the function
    tree = ast.parse(code)
    defs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    if len(defs) == 1 and defs[0] in ns:
        return ns[defs[0]]
    raise ValueError(f"Cannot resolve callable '{preferred_name}' from optimized code.")


# ------------------------------------------------------------------ #
# Step 7: tolerant equality check
# ------------------------------------------------------------------ #
def results_match(a, b) -> bool:
    try:
        if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
            if a.shape != b.shape:
                return False
            for col in a.columns:
                if col not in b.columns:
                    return False
                if pd.api.types.is_numeric_dtype(a[col]):
                    if not np.allclose(a[col].values, b[col].values, rtol=1e-4, atol=1e-6, equal_nan=True):
                        return False
                else:
                    if not (a[col].values == b[col].values).all():
                        return False
            return True
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            return np.allclose(a.values, b.values, rtol=1e-4, atol=1e-6, equal_nan=True)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a.shape == b.shape and np.allclose(a, b, rtol=1e-4, atol=1e-6, equal_nan=True)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(results_match(x, y) for x, y in zip(a, b))
        if isinstance(a, float) or isinstance(b, float):
            return abs(float(a) - float(b)) < 1e-4
        return a == b
    except Exception:
        return False


# ------------------------------------------------------------------ #
# Main optimize() — callable from both CLI and Python API
# ------------------------------------------------------------------ #
def optimize(
    filepath: str,
    func_name: str | None = None,
    n: int = _SYNTHETIC_N_DEFAULT,
    output: str | None = None,
    show_speedup: bool = True,
) -> dict:
    """
    Optimize a function in filepath.
    Returns a dict with keys: success, correct, speedup, orig_time, opt_time, code.
    """
    filepath = os.path.abspath(filepath)

    # Step 1
    try:
        func_source, func_name = extract_function(filepath, func_name)
    except ValueError as e:
        print(f"ERROR: {e}")
        return {"success": False, "error": str(e)}

    print(f"\nFunction : {func_name}")
    print(f"File     : {os.path.basename(filepath)}")
    print(f"\nOriginal code:")
    print("-" * 40)
    print(func_source)
    print("-" * 40)

    # Step 2
    try:
        args, kwargs = resolve_inputs(filepath, func_source, func_name, n)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return {"success": False, "error": str(e)}

    # Load original callable
    spec = importlib.util.spec_from_file_location("_user_mod_orig", filepath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"ERROR: Could not import {filepath}: {e}")
        return {"success": False, "error": str(e)}
    orig_func = getattr(mod, func_name, None)
    if orig_func is None:
        print(f"ERROR: function '{func_name}' not found after import")
        return {"success": False, "error": f"function '{func_name}' not found"}

    # Step 3
    print("\n[1/4] Timing original function...")
    try:
        baseline_result, orig_time = time_function(orig_func, args, kwargs)
    except Exception as e:
        print(f"ERROR: Original function raised: {e}")
        return {"success": False, "error": str(e)}
    print(f"      {orig_time * 1000:.3f} ms")

    # Step 4
    pattern = detect_pattern(func_source)
    print(f"\n[2/4] Building prompt (pattern: {pattern})...")
    prompt = build_prompt(func_source, func_name, pattern)

    # Step 5
    print("[3/4] Calling LLM...")
    raw = call_llm(prompt)
    if not raw:
        print("ERROR: LLM returned empty response.")
        return {"success": False, "error": "empty LLM response"}

    # Step 6
    print("[4/4] Validating optimized code...")
    opt_code = extract_code(raw)
    try:
        ast.parse(opt_code)
    except SyntaxError as e:
        print(f"ERROR: Optimized code has syntax error: {e}")
        return {"success": False, "error": f"syntax error: {e}"}

    try:
        opt_func = exec_and_resolve(opt_code, func_name)
    except Exception as e:
        print(f"ERROR: Could not resolve optimized function: {e}")
        return {"success": False, "error": str(e)}

    try:
        opt_result, opt_time = time_function(opt_func, args, kwargs)
    except Exception as e:
        print(f"ERROR: Optimized function raised: {e}")
        return {"success": False, "error": str(e)}

    correct = results_match(baseline_result, opt_result)
    speedup = orig_time / opt_time if opt_time > 0 else 0.0

    # Report
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Original  : {orig_time * 1000:.3f} ms")
    print(f"Optimized : {opt_time * 1000:.3f} ms")
    print(f"Speedup   : {speedup:.1f}x")
    print(f"Correct   : {'PASS' if correct else 'FAIL'}")
    print(f"\nOptimized code:")
    print("-" * 40)
    print(opt_code)
    print("-" * 40)

    if output and correct:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(opt_code)
        print(f"\nSaved to: {output}")

    return {
        "success": True,
        "correct": correct,
        "speedup": speedup,
        "orig_time": orig_time,
        "opt_time": opt_time,
        "code": opt_code,
    }


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        prog="numpymaxx",
        description="NumpyMaxx - single-shot LLM vectorization optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  numpymaxx optimize my_script.py
  numpymaxx optimize my_script.py --function slow_fn
  numpymaxx optimize my_script.py --function slow_fn --n 5000 --output fast.py
        """,
    )
    sub = parser.add_subparsers(dest="command")

    opt_p = sub.add_parser("optimize", help="Optimize a function")
    opt_p.add_argument("file", help="Python file containing the function")
    opt_p.add_argument("--function", "-f", default=None, help="Function name (default: first public def)")
    opt_p.add_argument("--n", type=int, default=_SYNTHETIC_N_DEFAULT, help="Synthetic input size (default: 1000)")
    opt_p.add_argument("--output", "-o", default=None, help="Write optimized code to this file")

    args = parser.parse_args()

    if args.command == "optimize":
        print("NumpyMaxx - Vectorization Optimizer")
        print("=" * 50)
        optimize(
            filepath=args.file,
            func_name=args.function,
            n=args.n,
            output=args.output,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
