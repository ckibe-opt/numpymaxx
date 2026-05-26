# NumpyMaxx

**Single-shot LLM vectorizer for slow Pandas / NumPy code.**

Point it at a Python file with a slow function. It reads the function, generates real benchmark inputs, asks an LLM for a vectorized version, validates correctness against your original output, and reports the actual measured speedup.

One LLM call. ~5–15 seconds. No evolution loop, no database, no subprocess.

---

## Results on a Real-World Corpus

8 distinct slow patterns scraped from Stack Overflow / Reddit data-science questions.
Each pattern was run end-to-end through the CLI with no manual edits.

| # | Pattern | Original | Optimized | Speedup | Correct |
|---|---------|----------|-----------|---------|---------|
| 1 | Cumulative sum via loop | 113.4 ms | 0.1 ms | **839×** | ✓ |
| 2 | Substring filter via iterrows | 90.8 ms | 2.6 ms | **36×** | ✓ |
| 3 | Conditional column via apply | 94.9 ms | 0.6 ms | **170×** | ✓ |
| 4 | Manual matrix multiply (triple loop) | 52.1 ms | 0.0 ms | **6,275×** | ✓ |
| 5 | Pairwise distances via loop | 732.8 ms | 0.3 ms | **2,682×** | ✓ |
| 6 | Group-wise normalization via apply | 2,557.3 ms | 2.7 ms | **934×** | ✓ |
| 7 | Date parsing via iterrows | 148.4 ms | 2.0 ms | **76×** | ✓ |
| 8 | Rolling average via loop | 2.6 ms | 0.1 ms | **26×** | ✓ |

| Metric | Value |
|--------|-------|
| Success rate | **8 / 8 (100%)** |
| Median speedup | **839×** |
| Max speedup | **6,275×** |
| Correctness failures | 0 |
| Crashes | 0 |
| Wall-clock per optimization | 5 – 15 s |

Reproduce: `python numpymaxx/corpus_test.py --output corpus_report.md` from the repo root.

---

## Install

Requires Python 3.10+, `pandas`, `numpy`, and an LLM endpoint (OpenRouter recommended; Ollama supported as fallback).

```bash
git clone <repo>
cd localevo
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=sk-or-..." > .env
echo "LLM_PROVIDER=openrouter"      >> .env
echo "OPENROUTER_MODEL=openai/gpt-oss-120b:free" >> .env
```

---

## Usage

### Basic

```bash
python numpymaxx/numpymaxx_cli.py optimize my_script.py
```

Picks the first top-level function in the file and optimizes it.

### Targeted

```bash
python numpymaxx/numpymaxx_cli.py optimize my_script.py --function slow_thing --output fast_thing.py
```

### Providing real inputs

By default the CLI synthesizes inputs from parameter names (`df`, `arr`, `matrix`, etc.).
For deterministic, realistic benchmarks add a fixture function to your file:

```python
def _numpymaxx_inputs():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({'x': rng.standard_normal(2000), 'y': rng.standard_normal(2000)})
    return ([df], {})   # (positional args, kwargs)
```

The CLI will use it automatically.

### Optional evolution refinement

```bash
python numpymaxx/numpymaxx_cli.py optimize my_script.py --evolve 3
```

Takes the validated single-shot result as a seed and runs 3 additional LLM-driven refinement generations. Use only when single-shot output is correct but you want to push the speedup further.

---

## Example

`example_slow.py`:

```python
import pandas as pd
import numpy as np

def slow_filter_and_square(df):
    result = []
    for idx, row in df.iterrows():
        if row['x'] > 0:
            result.append(row['x'] ** 2)
    return result

def _numpymaxx_inputs():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({'x': rng.standard_normal(2000), 'y': rng.standard_normal(2000)})
    return ([df], {})
```

Run:

```bash
python numpymaxx/numpymaxx_cli.py optimize numpymaxx/example_slow.py
```

Output:

```
Original  : 36.037 ms
Optimized : 0.242 ms
Speedup   : 149.0x
Correct   : PASS

Optimized code:
def slow_filter_and_square(df):
    return df.loc[df['x'] > 0, 'x'].pow(2).tolist()
```

---

## How It Works

1. `ast` parses the target function from the source file.
2. Inputs resolved: prefer `_numpymaxx_inputs()`, otherwise synthesize from parameter names.
3. Original function timed: median of 5 runs after 2 warmups.
4. Pattern detected (`pandas_iterrows`, `numpy_matrix`, or `string_loop`) and a domain-specific hint is prepended to the prompt.
5. Single LLM call produces a vectorized version.
6. Code extracted from markdown fences, `exec`'d in a fresh namespace.
7. Output compared against the baseline using a tolerant equality check (`pd.DataFrame.equals`, `np.allclose`, list/scalar fallback).
8. Optimized version timed with the same protocol; speedup reported.

No evolution loop, no SQLite database writes, no subprocess.

---

## Honest Limitations

- **Three pattern hints only.** Code that doesn't fit `pandas_iterrows`, `numpy_matrix`, or `string_loop` falls back to a generic prompt and is more likely to fail validation.
- **One function per call.** Cross-function refactors are out of scope.
- **LLM dependence.** Output quality is bounded by the model. The corpus above used `openai/gpt-oss-120b:free` via OpenRouter.
- **Synthesized inputs are guesses.** For realistic benchmarks always add `_numpymaxx_inputs()`.
- **Sample of 8.** The 100% pass rate is on the curated corpus in `numpymaxx/corpus/`. Your code may not hit those numbers.

---

## Repository Layout

```
numpymaxx/
├── numpymaxx_cli.py       # the CLI (single-shot optimizer)
├── example_slow.py        # canonical happy-path example
├── simple_real_test.py    # end-to-end test on a real Stack Overflow snippet
├── corpus/                # 8 real-world slow patterns + fixtures
├── corpus_test.py         # batch runner over corpus/
├── corpus_report.md       # generated benchmark report
└── corpus_results.json    # machine-readable results
```

---

## License

MIT.
