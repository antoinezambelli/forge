# ADR-007: Report Views

**Status:** Planned (README roadmap item 5)

## Problem

`report.py` prints to stdout only — ASCII table + list view. Analyzing eval results requires re-running with different `--ablation` and `--exclude-scenario` flags, manually scanning 50+ rows to answer questions like "what's the best Ollama config?" or "how does Ministral 8B compare across backends?" There's no persistent output and no way to slice data interactively.

With ablation runs completing and BFCL on the horizon, the volume of configs will grow. Stdout scanning doesn't scale.

## Decision

Two output modes, both generated from the same JSONL data and reusing the existing aggregation logic in `report.py`:

1. **HTML dashboard** — single self-contained file with interactive filters and sortable columns
2. **Markdown snapshots** — pre-sliced static views for embedding in docs or quick reference

### HTML dashboard

One self-contained HTML file. No build step, no dependencies, no server. Open in any browser.

**Architecture:**
- `report.py` reads JSONL, runs `compute_config_metrics()` for all configs
- Aggregated data embedded as `const DATA = [...]` in a `<script>` tag
- [Pico CSS](https://picocss.com/) inlined in `<style>` for styling (~10KB minified, classless — semantic HTML looks good automatically, dark mode built in)
- ~100-150 lines of vanilla JS for interactivity

**Filter dimensions** (derived from JSONL fields, not hardcoded):
- **Backend**: ollama, llamaserver, llamafile, anthropic
- **Mode**: native, prompt
- **Model family**: extracted from model name (llama3.1, ministral-8b, ministral-14b, qwen3-8b, qwen3-14b, mistral-v0.3, mistral-nemo, claude)
- **Quant**: extracted from model name (q4_K_M, q8_0, n/a for API models)
- **Ablation**: full, bare, bare+any, etc.
- **Scenario**: multi-select to include/exclude specific scenarios from aggregates

Each filter is a multi-select dropdown (or checkbox group). Filters compose with AND — selecting "ollama" + "ministral-8b" shows only Ollama Ministral 8B configs.

**Table features:**
- Click column header to sort (asc/desc toggle)
- Same columns as current ASCII table: Score, Accuracy, Completeness, Efficiency, Wasted, Speed, N, per-scenario accuracy
- Row count updates as filters change
- Optional: highlight cells by value (green/yellow/red heat map for accuracy)

**Compare mode** (HTML-only):
- Checkbox column on each row (max 2 selectable)
- When 2 rows are checked, a comparison panel appears below the table
- Layout: Config A | Delta | Config B
- Delta column shows colored diffs: green = B is better, red = B is worse (A is the baseline)
- All metrics get deltas: Score `+3%`, Efficiency `-0.02`, Speed `+1.2s`, etc.
- Per-scenario accuracy: same colored delta treatment
- "Swap A↔B" button to flip the baseline
- Clearing either checkbox dismisses the panel
- ~30-40 lines of additional JS; no changes to data model — purely a view concern

**Generation:**
```
python -m tests.eval.report eval_results.jsonl --html docs/results/dashboard.html
```

### Markdown snapshots

Static `.md` files with ASCII tables, generated in one batch. Each file is a pre-filtered view.

**Views:**

| File | Filter | Purpose |
|------|--------|---------|
| `all.md` | None (complete configs only) | Full leaderboard |
| `ollama.md` | backend=ollama | What works on Ollama |
| `llamaserver.md` | backend=llamaserver | llama-server results |
| `llamafile.md` | backend=llamafile | Llamafile results |
| `anthropic.md` | backend=anthropic | Anthropic baseline |
| `ablation.md` | All ablation presets | Guardrail impact comparison |
| `native-vs-prompt.md` | backend=llamaserver, grouped by model | Native FC vs prompt-injected on llama-server |
| `by-family.md` | Grouped by model family (sub-tables, best-scoring family first) | Same model across backends/modes |
| `budget.md` | Compaction scenarios only | Tight-budget performance |

Each file includes the legend and a timestamp. `index.md` links to all views.

`by-family.md` renders as sequential sub-tables — one per model family, ordered by the best-scoring config in each family. Within each sub-table, rows are sorted by score descending.

**Generation:**
```
python -m tests.eval.report eval_results.jsonl --markdown docs/results/
```

**Both modes can run together:**
```
python -m tests.eval.report eval_results.jsonl --html docs/results/dashboard.html --markdown docs/results/
```

## Implementation

### Changes to `report.py`

The existing code structure is clean for extension:
- `load_jsonl()` / `group_rows()` / `compute_config_metrics()` — reused as-is
- `print_table()` / `print_list()` — stdout stays, new renderers added alongside

New functions:

```python
def write_html(
    metrics_list: list[ConfigMetrics],
    rows: list[dict],
    scenarios: list[str],
    output_path: Path,
) -> None:
    """Write self-contained HTML dashboard."""
    # Serialize aggregated metrics + raw filter dimensions to JSON
    # Template into HTML string with inlined Pico CSS + JS
    # Write to output_path

def write_markdown_views(
    all_metrics: list[ConfigMetrics],
    grouped: dict[ConfigKey, dict[str, list[dict]]],
    scenarios: list[str],
    output_dir: Path,
) -> None:
    """Write pre-filtered markdown view files."""
    # For each view definition: filter metrics, render ASCII table to string
    # Write each .md file
    # Write index.md linking to all views
```

### HTML template structure

```html
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <title>Forge Eval Dashboard</title>
  <style>/* Pico CSS inlined here (~10KB) */</style>
  <style>/* Custom overrides: table font-size, filter layout, etc. */</style>
</head>
<body>
  <main class="container">
    <h1>Forge Eval Dashboard</h1>
    <p id="summary">50 configs, 24448 runs</p>

    <!-- Filter bar -->
    <fieldset>
      <legend>Filters</legend>
      <div id="filters">
        <!-- JS populates filter controls from DATA dimensions -->
      </div>
    </fieldset>

    <!-- Results table -->
    <figure>
      <table id="results" role="grid">
        <thead><!-- Column headers, clickable for sort; first col is compare checkbox --></thead>
        <tbody><!-- Rows, filtered/sorted by JS --></tbody>
      </table>
    </figure>

    <!-- Compare panel (hidden until 2 rows checked) -->
    <article id="compare" hidden>
      <header>
        <span id="compare-title">Comparing: A vs B</span>
        <button id="compare-swap">Swap A↔B</button>
      </header>
      <table id="compare-table">
        <thead><tr><th>Metric</th><th>Config A</th><th>Delta</th><th>Config B</th></tr></thead>
        <tbody><!-- JS renders metric rows with colored deltas --></tbody>
      </table>
    </article>
  </main>

  <script>
    const DATA = /* report.py injects JSON here */;
    const SCENARIOS = /* scenario list */;
    // ~130 lines: render filters, render table, sort/filter handlers, compare logic
  </script>
</body>
</html>
```

Pico CSS gives this semantic HTML automatic styling: responsive table, dark mode, form elements for filters, clean typography. No classes needed.

### Model family extraction

Derive from model name string rather than requiring a field in JSONL:

```python
def extract_family(model: str) -> str:
    """Extract model family from Ollama-style model name."""
    if "claude" in model:
        return "claude"
    if "llama3.1" in model:
        return "llama3.1"
    if model.startswith("qwen3:"):
        size = model.split(":")[1].split("-")[0]  # "8b", "14b"
        return f"qwen3-{size}"
    if "ministral" in model:
        # ministral-3:8b-instruct -> ministral-8b, ministral-3:14b -> ministral-14b
        size = model.split(":")[1].split("-")[0]
        return f"ministral-{size}"
    if "mistral-nemo" in model:
        return "mistral-nemo"
    if "mistral:" in model:
        return "mistral-v0.3"
    return model.split(":")[0]

def extract_quant(model: str) -> str:
    """Extract quantization from model name."""
    if "q4_K_M" in model:
        return "q4_K_M"
    if "q8_0" in model:
        return "q8_0"
    return "n/a"
```

These are heuristics — good enough for the current model set, easily extended.

### CLI additions

```python
parser.add_argument(
    "--html", metavar="PATH",
    help="Write interactive HTML dashboard to PATH",
)
parser.add_argument(
    "--markdown", metavar="DIR",
    help="Write pre-filtered markdown views to DIR",
)
```

Existing stdout behavior unchanged. New flags are additive.

## Touch Points

| Component | Change | Size |
|-----------|--------|------|
| `report.py` | Add `--html` and `--markdown` flags, `write_html()`, `write_markdown_views()` | Medium |
| `report.py` | `extract_family()`, `extract_quant()` helpers | Small |
| HTML template | Inline Pico CSS + vanilla JS filter/sort/compare logic | Medium |
| `docs/results/` | New directory for generated output | Trivial |
| `.gitignore` | Optionally ignore `docs/results/*.html` if dashboard is ephemeral | Trivial |

## What Doesn't Change

- Existing `print_table()` / `print_list()` stdout output — still the default
- `load_jsonl()` / `group_rows()` / `compute_config_metrics()` — reused, not modified
- `ConfigKey` / `ConfigMetrics` dataclasses — consumed as-is
- JSONL format — no new fields needed
- `batch_eval.py` / `eval_runner.py` — writers unchanged

## Resolved Questions

1. **HTML dashboard: committed.** It's a snapshot of latest results, same as the README table. Regenerate before meaningful pushes.

2. **Markdown views: manual regeneration.** Batch runs take hours — inspect stdout results before committing views. No auto-generation in `batch_eval.py`.

3. **Native-vs-prompt view: added.** Effectively slices llama-server in two (it's the only backend with both modes at scale). Added as `native-vs-prompt.md`.

4. **No top-N view.** That's just `all.md` truncated — not worth a separate file.

5. **`by-family.md` uses sub-tables.** One table per model family, ordered by best-scoring config in each family. Within each sub-table, rows sorted by score descending.

6. **Compare mode: HTML-only.** Checkbox-based, max 2 configs, colored deltas. Not feasible in static markdown (combinatorial). Details in the HTML dashboard section above.
