# Eval Guide

Internal tooling for measuring how reliably a model + backend combo navigates multi-step tool-calling workflows. Not a test suite — run manually against a live backend.

## Eval Harness

### Quick Start

```bash
# Ollama — all scenarios, 10 runs each
python -m tests.eval.eval_runner --backend ollama --model "ministral-3:8b-instruct-2512-q4_K_M" --runs 10 --stream --verbose

# llama-server — start server in one terminal, run eval in another
llama-server --jinja -m path/to/model.gguf -ngl 999 --port 8080
python -m tests.eval.eval_runner --backend llamafile --llamafile-mode native --model ministral-14b-instruct-q4_k_m --runs 10 --stream --verbose

# Anthropic API
python -m tests.eval.eval_runner --backend anthropic --model claude-haiku-4-5-20251001 --runs 5 --stream --verbose
```

### eval_runner Flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--backend` | `ollama`, `llamafile`, `anthropic` | `ollama` | Backend to target |
| `--model` | string | *(required)* | Model name (Ollama-style or Anthropic model ID) |
| `--runs` | int | `10` | Runs per scenario |
| `--stream` | flag | off | Use streaming mode |
| `--verbose`, `-v` | flag | off | Print live per-message trace |
| `--tags` | `plumbing`, `model_quality`, `compaction`, `stateful`, `reasoning` | all | Filter scenarios by tag |
| `--scenario` | name(s) | all | Run specific scenario(s) by name |
| `--llamafile-mode` | `native`, `prompt`, `auto` | `auto` | FC mode for llamafile/llama-server backend |
| `--think` | `true`, `false`, `auto` | `auto` | Thinking mode. Ollama: controls `think` param. Llamafile: captures `[THINK]` tags and `reasoning_content` |
| `--budget-mode` | `backend`, `manual`, `forge-full`, `forge-fast` | `forge-full` | Context budget strategy. Compaction scenarios always override with their own budget |
| `--num-ctx` | int | none | Exact token budget (requires `--budget-mode manual`) |
| `--no-history` | flag | off | Disable message history collection (lighter, fewer metrics) |
| `--probe` | flag | off | Print resolved budget from backend and exit (no eval run) |
| `--base-url` | URL | none | Override backend base URL |
| `--ablation` | `reforged`, `no_rescue`, `no_nudge`, `no_steps`, `no_recovery`, `no_compact`, `bare` | `reforged` | Ablation preset: selectively disable guardrails |
| `--tool-choice` | `auto`, `any` | none | Anthropic `tool_choice` type. `any` forces tool calls |
| `--no-cache-prompt` | flag | off | Disable llama-server prompt caching |
| `--compact-strategy` | `tiered`, `sliding`, `none` | auto | Override compaction strategy for all scenarios |

### Scenarios

22 scenarios across four categories:

**Plumbing** (does forge's tool-calling loop work?):
- `basic_2step`, `sequential_3step`, `error_recovery`

**Model quality** (does the model reason correctly?):
- `tool_selection`, `argument_fidelity`, `sequential_reasoning`, `conditional_routing`, `data_gap_recovery`, `relevance_detection`

**Compaction chain** (multi-phase compaction retention):
- `compaction_chain_baseline`, `compaction_chain_p1`, `compaction_chain_p2`, `compaction_chain_p3`

**Stateful variants** (state carries between calls — wrong arguments cascade):
- `basic_2step_stateful`, `sequential_3step_stateful`, `error_recovery_stateful`, `tool_selection_stateful`, `argument_fidelity_stateful`, `sequential_reasoning_stateful`, `conditional_routing_stateful`, `data_gap_recovery_stateful`, `relevance_detection_stateful`

**Lambda vs stateful:** Lambda scenarios use hardcoded echo tools — tool arguments don't affect the result. Stateful scenarios use backend classes where arguments matter and state carries between calls. The delta between lambda and stateful scores for the same model isolates model reasoning quality from forge correctness.

### Examples

```bash
# Filter by tag
python -m tests.eval.eval_runner --backend ollama --model "qwen3:8b-q4_K_M" --runs 5 --tags plumbing

# Specific scenarios
python -m tests.eval.eval_runner --backend ollama --model "ministral-3:8b-instruct-2512-q4_K_M" --runs 10 --scenario basic_2step sequential_3step

# Qwen3 with thinking on llama-server
llama-server --jinja -m path/to/Qwen3-8B-Q4_K_M.gguf -ngl 999 --port 8080 --reasoning-format auto
python -m tests.eval.eval_runner --backend llamafile --llamafile-mode native --model qwen3-8b-q4_k_m --runs 10 --stream --think true

# Probe budget without running eval
python -m tests.eval.eval_runner --backend ollama --model "ministral-3:8b-instruct-2512-q4_K_M" --probe

# Ablation — bare (all guardrails off)
python -m tests.eval.eval_runner --backend anthropic --model claude-haiku-4-5-20251001 --runs 5 --stream --ablation bare
```

All non-compaction scenarios (copy-paste friendly):

```
--scenario basic_2step sequential_3step error_recovery tool_selection argument_fidelity sequential_reasoning conditional_routing data_gap_recovery relevance_detection
```

All stateful scenarios (copy-paste friendly):

```
--scenario basic_2step_stateful sequential_3step_stateful error_recovery_stateful tool_selection_stateful argument_fidelity_stateful sequential_reasoning_stateful conditional_routing_stateful data_gap_recovery_stateful relevance_detection_stateful
```

---

## Batch Eval

Run large-scale model comparisons across all backends. Results append to JSONL with automatic resume. Ollama auto-loads models, llama-server is auto-managed (start/stop/health check per GGUF), llamafile binaries require a manual server.

### batch_eval Flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--config` | `all`, `ollama`, `llamaserver`, `llamafile`, `llamaserver-native`, `llamaserver-prompt`, `anthropic`, `anthropic-any`, `haiku`, `sonnet`, `opus`, `haiku-any`, `sonnet-any`, `opus-any` | `all` | Config set to run |
| `--runs` | int | `50` | Runs per scenario |
| `--output` | path | `eval_results.jsonl` | JSONL output path |
| `--scenario` | name(s) | all | Run specific scenario(s) |
| `--tags` | tag(s) | all | Filter scenarios by tag |
| `--budget-mode` | `backend`, `manual`, `forge-full`, `forge-fast` | `forge-full` | Context budget strategy |
| `--num-ctx` | int | none | Exact token budget (requires `--budget-mode manual`) |
| `--ablation` | preset name | `reforged` | Ablation preset |
| `--model` | substring | none | Filter configs to models containing this substring |
| `--dry-run` | flag | off | Show what would run without executing |
| `--verbose`, `-v` | flag | off | Print per-run details |

### Examples

```bash
# Ollama (11 models, fully unattended)
python -m tests.eval.batch_eval --config ollama --runs 50

# llama-server (auto-managed, starts/stops per GGUF)
python -m tests.eval.batch_eval --config llamaserver --runs 50

# Anthropic (costs money)
python -m tests.eval.batch_eval --config anthropic --runs 50

# Dry run
python -m tests.eval.batch_eval --config all --runs 50 --dry-run

# Filter to specific model
python -m tests.eval.batch_eval --config llamaserver --model 8b-reasoning --runs 20

# Specific scenarios only
python -m tests.eval.batch_eval --config ollama --runs 50 --scenario basic_2step sequential_reasoning
```

Resume is automatic: re-run the same command and it skips completed scenarios.

---

## Reports

### Forge eval report

```bash
# Full table + list
python -m tests.eval.report eval_results.jsonl

# Progress (for incomplete runs)
python -m tests.eval.report eval_results.jsonl --progress

# Compact list only (phone-friendly)
python -m tests.eval.report eval_results.jsonl --list-only

# Include partially-completed configs
python -m tests.eval.report eval_results.jsonl --include-partial

# Filter by ablation
python -m tests.eval.report eval_results.jsonl --ablation reforged bare

# Filter by scenario tag
python -m tests.eval.report eval_results.jsonl --tags stateful

# Exclude specific scenarios
python -m tests.eval.report eval_results.jsonl --exclude-scenario error_recovery

# HTML dashboard (requires Node.js)
python -m tests.eval.report eval_results.jsonl --html docs/results/dashboard.html

# Markdown views
python -m tests.eval.report eval_results.jsonl --markdown docs/results/

# Both
python -m tests.eval.report eval_results.jsonl --html docs/results/dashboard.html --markdown docs/results/
```

### report Flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `jsonl` | path | `eval_results.jsonl` | JSONL input file (positional, optional) |
| `--list-only` | flag | off | Skip table, show list view only |
| `--progress` | flag | off | Show progress for all configs (including incomplete) |
| `--include-partial` | flag | off | Include configs that haven't finished all scenarios |
| `--ablation` | preset name(s) | all | Filter to specific ablation preset(s) |
| `--exclude-scenario` | name(s) | none | Exclude scenario(s) from aggregates and columns |
| `--tags` | `stateful`, `lambda`, `compaction` | all | Filter to scenarios matching tag(s) |
| `--html` | path | none | Write interactive HTML dashboard |
| `--markdown` | dir | none | Write pre-filtered markdown views |

---

## BFCL Benchmark (removed)

Forge previously included a [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) v4 integration (11 categories, ~2,183 entries). It was removed in favor of forge's own eval harness, which measures multi-step workflow completion rather than single-call argument matching. Last commit with BFCL code: [`a9b0257`](https://github.com/antoinezambelli/forge/commit/a9b0257).

---

## Ablation Presets

Ablation selectively disables forge guardrails to isolate their contribution to model performance.

| Preset | Rescue | Retry Nudge | Step Enforcement | Error Recovery | Compaction |
|--------|--------|-------------|------------------|----------------|------------|
| `reforged` | yes | yes (5 retries) | yes | yes (2 errors) | yes |
| `no_rescue` | **no** | yes | yes | yes | yes |
| `no_nudge` | **no** | **no** | yes | yes | yes |
| `no_steps` | yes | yes | **no** | yes | yes |
| `no_recovery` | yes | yes | yes | **no** | yes |
| `no_compact` | yes | yes | yes | yes | **no** |
| `bare` | **no** | **no** | **no** | **no** | **no** |

---

## Backend Notes

See [BACKEND_SETUP.md](BACKEND_SETUP.md) for installation, server launch, and verification instructions for each backend (Ollama, llama-server, llamafile).

**Key points for eval:**
- Ollama runs as a background service — no manual server launch needed
- llama-server needs `--jinja` for native function calling; use `--backend llamafile --llamafile-mode native`
- llamafile has no native FC — use `--llamafile-mode prompt`
- Anthropic needs `ANTHROPIC_API_KEY` env var; compaction scenarios are skipped (200K context)
