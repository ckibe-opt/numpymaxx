#!/usr/bin/env python3
"""
NumpyMaxx Real-World Corpus Test

Batch-runs the CLI on 8 distinct slow patterns collected from Stack Overflow / Reddit,
produces an honest markdown report with:
- Success rate (correct + ≥2× speedup)
- Speedup distribution (min, median, max)
- Failure modes (parse error, correctness fail, no speedup, crash)
- Per-pattern detailed results

Usage:
    python corpus_test.py
    python corpus_test.py --patterns 01 04 08  # subset
    python corpus_test.py --output report.md
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_CORPUS_DIR = _ROOT / "corpus"
sys.path.insert(0, str(_ROOT / "src"))  # src layout

# Windows console safety
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def list_patterns():
    """Return list of (pattern_id, filepath) tuples sorted by id."""
    patterns = []
    for f in _CORPUS_DIR.glob("pattern_*.py"):
        # Extract number from pattern_NN_name.py
        parts = f.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            patterns.append((int(parts[1]), f))
    patterns.sort(key=lambda x: x[0])
    return patterns


def extract_first_function(filepath: Path) -> str | None:
    """Extract the first top-level function name from a Python file."""
    import ast
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            return node.name
    return None


def run_single(pattern_id: int, filepath: Path, verbose: bool = True):
    """Run CLI on one pattern, return result dict."""
    from numpymaxx import optimize

    func_name = extract_first_function(filepath)
    if not func_name:
        return {
            "pattern_id": pattern_id,
            "func_name": filepath.stem,
            "filepath": str(filepath),
            "success": False,
            "correct": False,
            "speedup": 0.0,
            "orig_time": 0.0,
            "opt_time": 0.0,
            "elapsed_wall": 0.0,
            "code": "",
            "error": "No public function found in file",
        }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Pattern {pattern_id:02d}: {func_name}")
        print(f"{'='*60}")

    start = time.perf_counter()
    try:
        result = optimize(
            filepath=str(filepath),
            func_name=func_name,
            n=1000,  # corpus default
            output=None,
            show_speedup=False,
        )
        elapsed = time.perf_counter() - start

        # Normalize result
        if result is None:
            result = {}

        return {
            "pattern_id": pattern_id,
            "func_name": func_name,
            "filepath": str(filepath),
            "success": result.get("success", False),
            "correct": result.get("correct", False),
            "speedup": result.get("speedup", 0.0),
            "orig_time": result.get("orig_time", 0.0),
            "opt_time": result.get("opt_time", 0.0),
            "elapsed_wall": elapsed,
            "code": result.get("code", "")[:500],  # truncated
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "pattern_id": pattern_id,
            "func_name": func_name,
            "filepath": str(filepath),
            "success": False,
            "correct": False,
            "speedup": 0.0,
            "orig_time": 0.0,
            "opt_time": 0.0,
            "elapsed_wall": elapsed,
            "code": "",
            "error": str(e),
        }


def classify_outcome(r):
    """Classify result into outcome category."""
    if r["error"]:
        return "CRASH"
    if not r["success"]:
        return "NO_OPTIMIZATION"
    if not r["correct"]:
        return "CORRECTNESS_FAIL"
    if r["speedup"] < 2.0:
        return "LOW_SPEEDUP"
    return "SUCCESS"


def generate_markdown_report(results, output_path=None):
    """Generate markdown report from results list."""
    lines = []
    lines.append("# NumpyMaxx Real-World Corpus Test Results")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nTotal patterns tested: {len(results)}")

    # Summary stats
    outcomes = [classify_outcome(r) for r in results]
    success_count = outcomes.count("SUCCESS")
    success_rate = success_count / len(results) * 100 if results else 0

    lines.append(f"\n## Summary")
    lines.append(f"\n| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Success rate (correct + ≥2×) | {success_count}/{len(results)} ({success_rate:.1f}%) |")
    lines.append(f"| Crashes | {outcomes.count('CRASH')} |")
    lines.append(f"| Correctness failures | {outcomes.count('CORRECTNESS_FAIL')} |")
    lines.append(f"| Low speedup (<2×) | {outcomes.count('LOW_SPEEDUP')} |")
    lines.append(f"| No optimization found | {outcomes.count('NO_OPTIMIZATION')} |")

    # Speedup stats for successful runs
    speedups = [r["speedup"] for r in results if classify_outcome(r) == "SUCCESS"]
    if speedups:
        lines.append(f"\n## Speedup Distribution (Successful Runs)")
        lines.append(f"\n| Stat | Value |")
        lines.append(f"|------|-------|")
        lines.append(f"| Min | {min(speedups):.1f}× |")
        lines.append(f"| Median | {sorted(speedups)[len(speedups)//2]:.1f}× |")
        lines.append(f"| Max | {max(speedups):.1f}× |")
        lines.append(f"| Mean | {sum(speedups)/len(speedups):.1f}× |")
    else:
        lines.append(f"\n## Speedup Distribution")
        lines.append("\nNo successful optimizations to report.")

    # Per-pattern details
    lines.append(f"\n## Per-Pattern Results")
    for r in results:
        outcome = classify_outcome(r)
        icon = {
            "SUCCESS": "✅",
            "LOW_SPEEDUP": "⚠️",
            "CORRECTNESS_FAIL": "❌",
            "NO_OPTIMIZATION": "➖",
            "CRASH": "💥",
        }.get(outcome, "❓")

        lines.append(f"\n### {r['pattern_id']:02d}. {r['func_name']}")
        lines.append(f"\n- **Outcome**: {icon} {outcome}")
        lines.append(f"- **Speedup**: {r['speedup']:.1f}×" if r['speedup'] > 0 else "- **Speedup**: N/A")
        lines.append(f"- **Original time**: {r['orig_time']*1000:.1f} ms" if r['orig_time'] else "- **Original time**: N/A")
        lines.append(f"- **Optimized time**: {r['opt_time']*1000:.1f} ms" if r['opt_time'] else "- **Optimized time**: N/A")
        lines.append(f"- **Wall time**: {r['elapsed_wall']:.1f}s")

        if r["error"]:
            lines.append(f"\n**Error**: `{r['error'][:200]}`")
        elif r["code"]:
            lines.append(f"\n**Optimized code**:")
            lines.append("```python")
            lines.append(r["code"])
            lines.append("```")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run NumpyMaxx corpus test")
    parser.add_argument("--patterns", nargs="*", help="Specific pattern IDs to run (e.g., 01 02 03)")
    parser.add_argument("--output", "-o", help="Write markdown report to this file")
    parser.add_argument("--json", "-j", help="Also write JSON results to this file")
    args = parser.parse_args()

    print("NumpyMaxx Real-World Corpus Test")
    print("=" * 60)
    print("Testing on real Stack Overflow / Reddit patterns")
    print("=" * 60)

    # Get patterns to run
    all_patterns = list_patterns()
    if args.patterns:
        wanted = set(int(p) for p in args.patterns)
        patterns = [(pid, fp) for pid, fp in all_patterns if pid in wanted]
    else:
        patterns = all_patterns

    print(f"\nRunning {len(patterns)} patterns...")

    # Run all
    results = []
    for pid, fp in patterns:
        r = run_single(pid, fp, verbose=True)
        results.append(r)

    # Generate report
    report = generate_markdown_report(results, args.output)
    print("\n" + report)

    # JSON output
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nJSON saved to: {args.json}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    outcomes = [classify_outcome(r) for r in results]
    success = outcomes.count("SUCCESS")
    print(f"Success rate: {success}/{len(results)} ({success/len(results)*100:.1f}%)")

    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
