# Contributing to forge

Thanks for your interest in contributing. This guide covers how to get set up, run tests, and where to look when adding new functionality.

## Setup

```bash
git clone https://github.com/antoinezambelli/forge.git
cd forge
python -m venv .venv
pip install -e ".[dev]"
```

## Running Tests

Unit tests are fully deterministic — no LLM backend required.

```bash
# Full suite (865 tests)
python -m pytest tests/unit/ -v --tb=short

# With coverage
python -m pytest tests/unit/ --cov=forge --cov-report=term-missing

# Single file
python -m pytest tests/unit/test_runner.py -v
```

Integration tests (`@pytest.mark.integration`) require a running backend. Skip them with:

```bash
python -m pytest tests/ -m "not integration"
```

## Project Layout

```
src/forge/           # Library source
  clients/           # LLM backend adapters (one per backend)
  core/              # Workflow, runner, messages, steps
  context/           # Context management and compaction
  prompts/           # Prompt templates and nudges
tests/
  unit/              # Deterministic tests
  eval/              # Eval harness (requires live backends)
    scenarios/       # Eval scenario definitions
    dashboard/       # React-based HTML dashboard (separate npm build)
docs/                # User-facing documentation
  decisions/         # Architecture Decision Records (ADRs)
  results/           # Eval results and raw data tables
```

## Common Contribution Areas

### Adding a new LLM backend client

1. Create `src/forge/clients/yourbackend.py`
2. Implement the `LLMClient` protocol defined in `src/forge/clients/base.py` — it requires `send()` and `send_stream()` methods
3. Add unit tests in `tests/unit/test_yourbackend_client.py`
4. Export from `src/forge/__init__.py`
5. Add backend setup instructions to `docs/BACKEND_SETUP.md`

### Adding an eval scenario

1. Pick the right file in `tests/eval/scenarios/`:
   - `_plumbing.py` — basic tool-calling mechanics
   - `_model_quality.py` — model reasoning and argument fidelity
   - `_compaction.py` / `_compaction_chain.py` — context window pressure
   - `_stateful_*.py` — stateful variants of the above
2. Define an `EvalScenario` with a `Workflow`, validation function, and tags
3. Register it in `ALL_SCENARIOS` (see existing patterns in each file)
4. Run it: `python -m tests.eval.eval_runner --scenarios your_scenario --runs 5`

### Adding or modifying guardrails

Guardrails live in the runner (`src/forge/core/runner.py`) and nudge templates (`src/forge/prompts/nudges.py`). Each guardrail can be independently toggled via ablation presets in `tests/eval/ablation.py`. If you add a new guardrail:

1. Add the toggle to `AblationConfig`
2. Create a new ablation preset that isolates it
3. Run eval with and without to measure impact

## Eval Dashboard

The interactive HTML dashboard is a React app at `tests/eval/dashboard/`. It's a separate build:

```bash
cd tests/eval/dashboard
npm install
npm run build
```

The built output is embedded into `docs/results/dashboard.html` via `report.py --html`.

## Architecture Decision Records

Design decisions are documented in `docs/decisions/`. If you're proposing a significant change, consider writing an ADR first. See existing ones for the format.

## Code Style

- Python 3.12+ — use modern syntax (type unions with `|`, etc.)
- `asyncio` throughout — all client methods and the runner are async
- Pydantic for tool parameter schemas
- No external formatting/linting tools enforced yet — match the style of surrounding code

## Questions

Open an issue on GitHub if something is unclear.
