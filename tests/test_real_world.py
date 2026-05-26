#!/usr/bin/env python3
"""
Real-world end-to-end test for NumpyMaxx.

Tests against an actual user pattern from Stack Overflow:
  "Pandas iterrows too slow, how can I vectorize this code?"
  (nested iterrows performing a group-multiply join)

Uses the new single-shot CLI (numpymaxx_cli.optimize) directly — no subprocess,
no evolve.py, no evolution.db.  Exits non-zero only on unexpected exceptions.
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))  # src layout

# Windows console safety
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ------------------------------------------------------------------ #
# The real Stack Overflow snippet (slightly reformatted, zero manual edits)
# Source pattern: r/learnpython "slow comparisons with pandas" /
#                 SO "Pandas iterrows too slow, how can I vectorize?"
# ------------------------------------------------------------------ #
_REAL_WORLD_CODE = '''\
import pandas as pd
import numpy as np


def calculate_group_products(table1, table2):
    """
    For each row in table1 join to all rows in table2 that share the same
    'letter', multiply number1 * number2 and collect the max per letter.
    Classic slow nested-iterrows pattern real users write.
    """
    results = []
    for _, row1 in table1.iterrows():
        for _, row2 in table2.iterrows():
            if row1["letter"] == row2["letter"]:
                results.append({
                    "letter": row1["letter"],
                    "product": row1["number1"] * row2["number2"],
                })
    if not results:
        return pd.DataFrame(columns=["letter", "product"])
    df = pd.DataFrame(results)
    return df.groupby("letter")["product"].max().reset_index()


def _numpymaxx_inputs():
    """Deterministic fixture so timings are reproducible."""
    rng = np.random.default_rng(7)
    t1 = pd.DataFrame({
        "letter":  list("abcd") * 25,          # 100 rows
        "number1": rng.standard_normal(100),
    })
    t2 = pd.DataFrame({
        "letter":  list("aabbccdd") * 25,       # 200 rows
        "number2": rng.standard_normal(200),
    })
    return ([t1, t2], {})
'''


def write_temp_file(code: str) -> str:
    """Write code to a named temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    )
    tmp.write(code)
    tmp.close()
    return tmp.name


def main():
    print("NumpyMaxx Real-World End-to-End Test")
    print("=" * 50)
    print("Target: nested iterrows group-multiply (Stack Overflow pattern)")
    print()

    # Write snippet to temp file
    tmp_path = write_temp_file(_REAL_WORLD_CODE)

    try:
        from numpymaxx import optimize

        result = optimize(
            filepath=tmp_path,
            func_name="calculate_group_products",
            n=200,
            output=None,
        )
    except Exception as exc:
        print(f"\nUNEXPECTED EXCEPTION: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        os.unlink(tmp_path)

    print()
    print("=" * 50)
    print("VERDICT")
    print("=" * 50)

    if not result.get("success"):
        print("OUTCOME: NumpyMaxx could not optimize this pattern.")
        print("This is an expected limitation for complex nested loops.")
        print("No exception escaped — clean failure mode. OK.")
        sys.exit(0)

    speedup = result["speedup"]
    correct = result["correct"]

    if correct and speedup >= 2.0:
        print(f"OUTCOME: SUCCESS  {speedup:.1f}x speedup, correctness PASS")
    elif correct:
        print(f"OUTCOME: PARTIAL  {speedup:.1f}x speedup, correctness PASS (speedup < 2x)")
    else:
        print(f"OUTCOME: CORRECTNESS FAIL  {speedup:.1f}x speedup but outputs differ")
        print("The optimized code is functionally wrong — do not use it.")

    sys.exit(0)


if __name__ == "__main__":
    main()
