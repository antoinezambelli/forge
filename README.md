# forge

[![Tests](https://github.com/antoinezambelli/forge/actions/workflows/tests.yml/badge.svg)](https://github.com/antoinezambelli/forge/actions/workflows/tests.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Python framework for self-hosted LLM tool-calling and multi-step agentic workflows. Define tools, pick a backend, run structured agent loops on consumer hardware.

Supports Ollama, llama-server (llama.cpp), Llamafile, and Anthropic as backends, with VRAM-aware context budgets, tiered compaction, and guardrail-based reliability (step enforcement, retry nudges, rescue loops).

## Requirements

- Python 3.12+
- A running LLM backend (see below)

## Install

```bash
git clone https://github.com/antoinezambelli/forge.git
cd forge
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -e .                # core only
pip install -e ".[anthropic]"   # + Anthropic client
pip install -e ".[dev]"         # + test/eval dependencies
```

### Backend setup (pick one)

**Ollama** (easiest):
```bash
# Install from https://ollama.com/download
ollama pull ministral-3:8b-instruct-2512-q4_K_M
```

**llama-server** (best performance):
```bash
# Install from https://github.com/ggml-org/llama.cpp/releases
llama-server -m path/to/Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf --jinja -ngl 999 --port 8080
```

**Anthropic** (API, no local GPU needed):
```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-...
```

See [Backend Setup](docs/BACKEND_SETUP.md) for full instructions and [Model Guide](docs/MODEL_GUIDE.md) for which model fits your hardware.

## Quick Start

```python
import asyncio
from forge import (
    Workflow, ToolDef, ToolSpec, ToolParam,
    WorkflowRunner, OllamaClient,
    ContextManager, TieredCompact,
)

def get_weather(city: str) -> str:
    return f"72°F and sunny in {city}"

workflow = Workflow(
    name="weather",
    description="Look up weather for a city.",
    tools={
        "get_weather": ToolDef(
            spec=ToolSpec(
                name="get_weather",
                description="Get current weather",
                parameters=[ToolParam("city", "string", "City name", required=True)],
            ),
            callable=get_weather,
        ),
    },
    terminal_tool="get_weather",
)

async def main():
    client = OllamaClient(model="ministral-3:8b-instruct-2512-q4_K_M")
    ctx = ContextManager(strategy=TieredCompact(keep_recent=2), budget_tokens=8192)
    runner = WorkflowRunner(client=client, context_manager=ctx)
    await runner.run(workflow, "What's the weather in Paris?")

asyncio.run(main())
```

For multi-step workflows, multi-turn conversations, and backend auto-management, see the [User Guide](docs/USER_GUIDE.md).

## Backends

| Backend | Best for | Native FC? |
|---------|----------|------------|
| **Ollama** | Easiest setup, model management built-in | Yes |
| **llama-server** | Best performance, full control | Yes (with `--jinja`) |
| **Llamafile** | Single binary, zero dependencies | No (prompt-injected) |
| **Anthropic** | Frontier baseline, hybrid workflows | Yes |

See [Backend Setup](docs/BACKEND_SETUP.md) for installation and [Model Guide](docs/MODEL_GUIDE.md) for which model to pick.

## Running Tests

```bash
python -m pytest tests/ -v --tb=short
```

```bash
python -m pytest tests/ --cov=forge --cov-report=term-missing
```

## Eval Harness

29 scenarios measuring how reliably a model + backend combo navigates multi-step tool-calling workflows. See [Eval Guide](docs/EVAL_GUIDE.md) for full CLI reference.

```bash
# Ollama
python -m tests.eval.eval_runner --backend ollama --model "ministral-3:8b-instruct-2512-q4_K_M" --runs 10 --stream --verbose

# Batch eval (JSONL output, automatic resume)
python -m tests.eval.batch_eval --config all --runs 50

# Reports (ASCII table, HTML dashboard, markdown views)
python -m tests.eval.report eval_results.jsonl
```

### BFCL Benchmark

Run forge against [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) v4 tasks (11 categories, ~2,183 entries). See [Eval Guide](docs/EVAL_GUIDE.md) for details.

## Project Structure

```
src/forge/
  __init__.py          # Public API exports
  errors.py            # ForgeError hierarchy
  server.py            # setup_backend(), ServerManager, BudgetMode
  core/
    messages.py        # Message, MessageRole, MessageType, MessageMeta
    workflow.py        # ToolParam, ToolSpec, ToolDef, ToolCall, TextResponse, Workflow
    runner.py          # WorkflowRunner — the agentic loop
    steps.py           # StepTracker
  clients/
    base.py            # ChunkType, StreamChunk, LLMClient protocol
    ollama.py          # OllamaClient (native FC)
    llamafile.py       # LlamafileClient (native FC or prompt-injected)
    anthropic.py       # AnthropicClient (frontier baseline)
  context/
    manager.py         # ContextManager, CompactEvent
    strategies.py      # CompactStrategy, NoCompact, TieredCompact, SlidingWindowCompact
    hardware.py        # HardwareProfile, detect_hardware()
  prompts/
    templates.py       # Tool prompt builders (prompt-injected path)
    nudges.py          # Retry and step-enforcement nudge templates
tests/
  unit/                # 562 deterministic tests — no LLM backend required
  eval/                # Eval harness — model qualification against real backends
```

## Roadmap

1. **Multi-model routing** — Model pool for managing N backends simultaneously. See [`docs/decisions/MULTI_MODEL_ROUTING.md`](docs/decisions/MULTI_MODEL_ROUTING.md).
2. **Tool prerequisites** — Conditional tool dependencies. See [`docs/decisions/006-tool-prerequisites.md`](docs/decisions/006-tool-prerequisites.md).
3. **Context window self-awareness** — Inject remaining context budget so the model knows compaction is approaching.
4. **Compaction tiers** — Consumer-configurable per-phase compaction thresholds.

## Documentation

- [User Guide](docs/USER_GUIDE.md) — Usage patterns, multi-turn, context management, guardrails
- [Model Guide](docs/MODEL_GUIDE.md) — Which model and backend for your hardware
- [Backend Setup](docs/BACKEND_SETUP.md) — Backend installation and server setup
- [Eval Guide](docs/EVAL_GUIDE.md) — Eval harness CLI reference, batch eval, BFCL benchmark
- [Architecture](docs/ARCHITECTURE.md) — Full design document
- [Workflow Internals](docs/WORKFLOW.md) — Workflow design and runner internals
- [Contributing](CONTRIBUTING.md) — How to set up, test, and add new backends or scenarios

## License

[MIT](LICENSE) — Copyright (c) 2025-2026 Antoine Zambelli
