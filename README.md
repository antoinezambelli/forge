# forge

[![Tests](https://github.com/antoinezambelli/forge/actions/workflows/tests.yml/badge.svg)](https://github.com/antoinezambelli/forge/actions/workflows/tests.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A reliability layer for self-hosted LLM tool-calling. Forge takes an 8B model from ~38% to ~99% on multi-step agentic workflows through guardrails (rescue parsing, retry nudges, step enforcement) and context management (VRAM-aware budgets, tiered compaction).

Three ways to use it:

- **WorkflowRunner** — Define tools, pick a backend, run structured agent loops. Forge manages the full lifecycle: system prompts, tool execution, context compaction, and guardrails. **SlotWorker** adds priority-queued access to a shared inference slot with auto-preemption — for multi-agent architectures where specialist workflows share a GPU slot. Best when you're building on forge directly.

- **Guardrails middleware** — Use forge's reliability stack ([composable middleware](examples/foreign_loop.py)) inside your own orchestration loop. You control the loop; forge validates responses, rescues malformed tool calls, and enforces required steps.

- **Proxy server** — Drop-in OpenAI-compatible proxy (`python -m forge.proxy`) that sits between any client (opencode, Continue, aider, etc.) and a local model server. Applies guardrails transparently — the client thinks it's talking to a smarter model.

Supports Ollama, llama-server (llama.cpp), Llamafile, and Anthropic as backends.

## Requirements

- Python 3.12+
- A running LLM backend (see below)

## Install

```bash
pip install forge-llm                # core only
pip install "forge-llm[anthropic]"   # + Anthropic client
```

For development:

```bash
git clone https://github.com/antoinezambelli/forge.git
cd forge
pip install -e ".[dev]"
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
from pydantic import BaseModel, Field
from forge import (
    Workflow, ToolDef, ToolSpec,
    WorkflowRunner, OllamaClient,
    ContextManager, TieredCompact,
)

def get_weather(city: str) -> str:
    return f"72°F and sunny in {city}"

class GetWeatherParams(BaseModel):
    city: str = Field(description="City name")

workflow = Workflow(
    name="weather",
    description="Look up weather for a city.",
    tools={
        "get_weather": ToolDef(
            spec=ToolSpec(
                name="get_weather",
                description="Get current weather",
                parameters=GetWeatherParams,
            ),
            callable=get_weather,
        ),
    },
    required_steps=[],
    terminal_tool="get_weather",
    system_prompt_template="You are a helpful assistant. Use the available tools to answer the user.",
)

async def main():
    client = OllamaClient(model="ministral-3:8b-instruct-2512-q4_K_M")
    ctx = ContextManager(strategy=TieredCompact(keep_recent=2), budget_tokens=8192)
    runner = WorkflowRunner(client=client, context_manager=ctx)
    await runner.run(workflow, "What's the weather in Paris?")

asyncio.run(main())
```

For multi-step workflows, multi-turn conversations, and backend auto-management, see the [User Guide](docs/USER_GUIDE.md). If you're building a long-running session (CLI, chat server, voice assistant), see the [long-running session advisory](docs/USER_GUIDE.md#long-running-sessions-filtering-transient-messages) for important guidance on filtering transient messages.

## Proxy Server

Drop-in replacement for a local model server. Point any OpenAI-compatible client at the proxy and get forge's guardrails for free.

```bash
# External mode — you manage llama-server, forge proxies it
python -m forge.proxy --backend-url http://localhost:8080 --port 8081

# Managed mode — forge starts llama-server and the proxy together
python -m forge.proxy --backend llamaserver --gguf path/to/model.gguf --port 8081
```

Then configure your client to use `http://localhost:8081/v1` as the API base URL.

**Note:** The proxy automatically injects a synthetic `respond` tool when tools are present in the request. The model calls `respond(message="...")` instead of producing bare text, keeping it in tool-calling mode where forge's full guardrail stack applies. The `respond` call is stripped from the outbound response — the client sees a normal text response (`finish_reason: "stop"`) and never knows the tool exists. This is essential for small local models (~8B), which cannot be trusted to choose correctly between text and tool calls — guiding them to a tool is a must. See [ADR-013](docs/decisions/013-text-response-intent.md) for the full analysis.

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

22 scenarios measuring how reliably a model + backend combo navigates multi-step tool-calling workflows. See [Eval Guide](docs/EVAL_GUIDE.md) for full CLI reference.

```bash
# Ollama
python -m tests.eval.eval_runner --backend ollama --model "ministral-3:8b-instruct-2512-q4_K_M" --runs 10 --stream --verbose

# Batch eval (JSONL output, automatic resume)
python -m tests.eval.batch_eval --config all --runs 50

# Reports (ASCII table, HTML dashboard, markdown views)
python -m tests.eval.report eval_results.jsonl
```

## Project Structure

```
src/forge/
  __init__.py          # Public API exports
  errors.py            # ForgeError hierarchy
  server.py            # setup_backend(), ServerManager, BudgetMode
  core/
    messages.py        # Message, MessageRole, MessageType, MessageMeta
    workflow.py        # ToolSpec, ToolDef, ToolCall, TextResponse, Workflow
    inference.py       # run_inference() — shared front half (compact, fold, validate, retry)
    runner.py          # WorkflowRunner — the agentic loop
    slot_worker.py     # SlotWorker — priority-queued slot access
    steps.py           # StepTracker
  guardrails/
    nudge.py           # Nudge dataclass
    response_validator.py  # ResponseValidator, ValidationResult
    step_enforcer.py   # StepEnforcer, StepCheck
    error_tracker.py   # ErrorTracker
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
  tools/
    respond.py         # Synthetic respond tool (respond_tool(), respond_spec())
  proxy/
    proxy.py           # ProxyServer — programmatic start/stop API
    server.py          # Raw asyncio HTTP server, SSE streaming
    handler.py         # Request handler — bridge between HTTP and run_inference
    convert.py         # OpenAI messages ↔ forge Messages conversion
tests/
  unit/                # 638 deterministic tests — no LLM backend required
  eval/                # Eval harness — model qualification against real backends
```

## Documentation

- [User Guide](docs/USER_GUIDE.md) — Usage patterns, multi-turn, context management, guardrails, slot worker, long-running session advisory
- [Model Guide](docs/MODEL_GUIDE.md) — Which model and backend for your hardware
- [Backend Setup](docs/BACKEND_SETUP.md) — Backend installation and server setup
- [Eval Guide](docs/EVAL_GUIDE.md) — Eval harness CLI reference, batch eval
- [Architecture](docs/ARCHITECTURE.md) — Full design document
- [Workflow Internals](docs/WORKFLOW.md) — Workflow design and runner internals
- [Contributing](CONTRIBUTING.md) — How to set up, test, and add new backends or scenarios

## License

[MIT](LICENSE) — Copyright (c) 2025-2026 Antoine Zambelli
