# forge

[![PyPI](https://img.shields.io/pypi/v/forge-guardrails.svg)](https://pypi.org/project/forge-guardrails/)
[![Tests](https://github.com/antoinezambelli/forge/actions/workflows/tests.yml/badge.svg)](https://github.com/antoinezambelli/forge/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/antoinezambelli/forge/branch/main/graph/badge.svg)](https://codecov.io/gh/antoinezambelli/forge)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A reliability layer for self-hosted LLM tool-calling. You give forge a set of tools; the model calls whichever it wants in whatever order. Workflow structure is opt-in — `required_steps`, `prerequisites`, and `terminal_tool` let you constrain the loop when you need to, but forge's guardrails (rescue parsing, retry nudges, response validation) apply with zero required steps too.

Forge takes an 8B local model from single digits to 84% across forge's 26-scenario v0.7.0 eval suite — and even lifts Sonnet 4.6 from 85% to 98% on the same workload (Anthropic numbers measured in v0.6.0; not re-run in v0.7.0 since the cost is non-trivial).

**What forge isn't:**
- **Not an agent orchestrator.** Forge sits inside one agentic loop and makes its tool calls reliable. Multi-agent graphs, DAG planners, and cross-agent coordination are out of scope.
- **Not a coding harness.** Forge is domain-agnostic. If you're building a coding agent (or already using one like opencode, aider, Cline), [proxy mode](#proxy-server) lifts your existing harness with forge's guardrails — no rewrite.

**Three ways to use it:**

- **Proxy server** — Drop-in proxy (`python -m forge.proxy`) speaking both the OpenAI chat-completions and Anthropic Messages (`/v1/messages`) APIs, sitting between any client and a local model server. Point OpenAI-compatible tools (opencode, Continue, aider) **or Claude Code** at it and forge applies guardrails transparently — the client thinks it's talking to a smarter model. Most popular entry point.

- **WorkflowRunner** — Define tools, pick a backend, run structured agent loops. Forge manages the full lifecycle: system prompts, tool execution, context compaction, and guardrails. **SlotWorker** adds priority-queued access to a shared inference slot with auto-preemption — for multi-agent architectures where specialist workflows share a GPU slot. Best when you're building on forge directly.

- **Guardrails middleware** — Use forge's reliability stack ([composable middleware](examples/foreign_loop.py)) inside your own orchestration loop. You control the loop; forge validates responses, rescues malformed tool calls, and enforces required steps.

Supports Ollama, llama-server (llama.cpp), Llamafile, vLLM, and Anthropic as backends.

## Requirements

- Python 3.12+
- A running LLM backend (see below)

## Install

```bash
pip install forge-guardrails                # core only
pip install "forge-guardrails[anthropic]"   # + Anthropic client
```

For development:

```bash
git clone https://github.com/antoinezambelli/forge.git
cd forge
pip install -e ".[dev]"
```

### Backend setup (pick one)

**llama-server** (recommended — top 10 eval configs all run on llama-server):
```bash
# Install from https://github.com/ggml-org/llama.cpp/releases
llama-server -m path/to/Ministral-3-8B-Instruct-2512-Q8_0.gguf --jinja -ngl 999 --port 8080
```

**Ollama** (alternative — easier setup, slightly weaker on harder workloads):
```bash
# Install from https://ollama.com/download
ollama pull ministral-3:8b-instruct-2512-q4_K_M
```

**Anthropic** (API, no local GPU needed):
```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-...
```

See [Backend Setup](docs/BACKEND_SETUP.md) for full instructions and [Model Guide](docs/MODEL_GUIDE.md) for which model fits your hardware.

## Quick Start

Start llama-server however you normally do (e.g. in a separate shell):

```bash
llama-server -m path/to/Ministral-3-8B-Instruct-2512-Q8_0.gguf --jinja -ngl 999 --port 8080
```

Then the Python you'll run (e.g. from another shell):

```python
import asyncio
from pydantic import BaseModel, Field
from forge import (
    Workflow, ToolDef, ToolSpec,
    WorkflowRunner, LlamafileClient,
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
    client = LlamafileClient(
        gguf_path="path/to/Ministral-3-8B-Instruct-2512-Q8_0.gguf",
        mode="native",
        recommended_sampling=True,
    )
    ctx = ContextManager(strategy=TieredCompact(keep_recent=2), budget_tokens=8192)
    runner = WorkflowRunner(client=client, context_manager=ctx)
    await runner.run(workflow, "What's the weather in Paris?")

asyncio.run(main())
```

For multi-step workflows, multi-turn conversations, and backend auto-management, see the [User Guide](docs/USER_GUIDE.md). If you're building a long-running session (CLI, chat server, voice assistant), see the [long-running session advisory](docs/USER_GUIDE.md#long-running-sessions-filtering-transient-messages) for important guidance on filtering transient messages.

## Proxy Server

Drop-in proxy that sits between any client and a local model server, speaking both the OpenAI chat-completions API and the Anthropic Messages API (`/v1/messages`). Point your client at the proxy (e.g. `http://localhost:8081/v1`) and forge applies its guardrails transparently — the client thinks it's talking to a smarter model.

This is the path for **using forge with an existing harness** (opencode, Continue, aider, Cline, anything that speaks the OpenAI chat-completions schema — or Claude Code, which speaks the Anthropic Messages API). No Python rewrite.

```bash
# External mode — you manage the backend, forge proxies it
python -m forge.proxy --backend-url http://localhost:8080 --port 8081

# Managed mode — forge starts the backend and the proxy together
python -m forge.proxy --backend llamaserver --gguf path/to/model.gguf --port 8081

# Managed vLLM — pass a model directory or HF repo id via --model-path
python -m forge.proxy --backend vllm --model-path /path/to/awq-dir --port 8081
```

Then configure your client to use `http://localhost:8081/v1` as the API base URL.

**Claude Code:** the proxy also serves the Anthropic Messages API on `POST /v1/messages`, so you can point Claude Code at a forge-guarded local model — set `ANTHROPIC_BASE_URL=http://localhost:8081` and `ANTHROPIC_AUTH_TOKEN=anything` for the `claude` process. See [Using forge with Claude Code](docs/USER_GUIDE.md#using-forge-with-claude-code) for the full setup (native-vs-prompt FC, Anthropic-shape downstreams, `cache_control`).

**Backend compatibility:**

- **Managed mode** spins up the backend for you. Supported backends: `llamaserver`, `llamafile`, `ollama`, `vllm` (use `--backend <name>` with `--gguf` for the GGUF-based backends, `--model-path` for vllm, or `--model` for ollama).
- **External mode** is backend-agnostic — forge talks `POST /v1/chat/completions` to whatever you point `--backend-url` at, as long as it speaks the OpenAI schema. Tool calls must come back in OpenAI `tool_calls` format or in one of forge's rescue-parsed formats (Mistral `[TOOL_CALLS]`, Qwen `<tool_call>` XML, fenced JSON). For a vLLM server, add `--backend vllm` so the proxy adopts vLLM's `--served-model-name` (vLLM 404s on a mismatched `model` field, unlike llama.cpp).

### What proxy mode fortifies

On every `POST /v1/chat/completions`, forge applies (in order):

1. **Response validation** — each tool call in the model's response is checked against the `tools` array in the request. Calls to unknown tool names or with malformed shapes are caught before the response returns to your client.
2. **Rescue parsing** — when the model emits tool calls in the wrong format (JSON in a code fence, Mistral's `[TOOL_CALLS]name{args}`, Qwen's `<tool_call>...</tool_call>` XML), forge extracts the structured call and re-emits it in the canonical OpenAI `tool_calls` schema. Biggest practical lift for Mistral-family models.
3. **Retry loop with error tracking** — if validation fails, forge retries inference up to `--max-retries` (default 3) with a corrective tool-result message on the canonical channel, rather than returning a malformed response. From your client's perspective the proxy looks like a single request that just took a few extra ms.
4. **Synthetic `respond` tool injection** — when tools are present in the request, forge injects a synthetic `respond` tool the model calls instead of producing bare text. The `respond` call is stripped from the outbound response — the client sees a normal text response (`finish_reason: "stop"`) and never knows the tool exists. Essential for small local models (~8B) that can't be trusted to choose correctly between text and tool calls. See [ADR-013](docs/decisions/013-text-response-intent.md) for the full analysis.

### What proxy mode does *not* do

Proxy mode is single-shot per request; some forge features need multi-turn workflow state that the OpenAI chat-completions schema doesn't carry:

- **Prerequisite enforcement and step-ordering** — these need a workflow definition spanning turns. Available in `WorkflowRunner`.
- **Context compaction and session memory** — proxy mode forwards the inbound message list as-is; managing the rolling window is the client's job.
- **VRAM-aware budget detection** — opt in with `--budget-mode forge-full` or `--budget-mode forge-fast`; otherwise proxy uses the backend's reported budget.

For the full guardrail surface, use `WorkflowRunner` directly. The proxy trades depth for "use forge with your existing setup, no rewrite."

### Useful flags

| Flag | Default | Purpose |
|---|---|---|
| `--max-retries N` | 3 | Retry budget per validation failure |
| `--no-rescue` | (rescue on) | Disable rescue parsing (debugging only) |
| `--budget-mode {backend,manual,forge-full,forge-fast}` | `backend` | Context budget source |
| `--budget-tokens N` | — | Manual token budget (requires `--budget-mode manual`) |
| `--serialize` / `--no-serialize` | auto | Force request serialization (single-slot backends) |

### Docker

You can run the forge proxy as a Docker container.

**Build the image:**

```bash
docker build -t forge-proxy .
```

**Run the container:**

```bash
# Connect to an external backend (e.g. vLLM hosted on the same machine)
docker run -p 8081:8081 forge-proxy --backend-url http://host.docker.internal:8000 --backend vllm --budget-mode manual --budget-tokens 8192
```

Note: If your backend is running on `localhost` of the host machine, use `http://host.docker.internal:PORT` (on macOS/Windows) or the host's IP address to allow the container to reach it.

## Backends

| Backend | Best for | Native FC? |
|---------|----------|------------|
| **Ollama** | Easiest setup, model management built-in | Yes |
| **llama-server** | Best performance, full control | Yes (with `--jinja`) |
| **Llamafile** | Single binary, zero dependencies | No (prompt-injected) |
| **vLLM** | High-throughput serving, AWQ/GPTQ weights | Yes (server-side parser) |
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

26 scenarios measuring how reliably a model + backend combo navigates multi-step tool-calling workflows — split into an OG-18 baseline tier and an 8-scenario advanced_reasoning tier for top-end separation. See [Eval Guide](docs/EVAL_GUIDE.md) for full CLI reference.

```bash
# llama-server (start in another terminal first; see Eval Guide)
python -m tests.eval.eval_runner --backend llamafile --llamafile-mode prompt --gguf "path/to/Ministral-3-8B-Instruct-2512-Q8_0.gguf" --runs 10 --stream --verbose

# Batch eval (JSONL output, automatic resume)
python -m tests.eval.batch_eval --config all --runs 50

# Reports — ASCII table by default; --html / --markdown export views
python -m tests.eval.report eval_results.jsonl
python -m tests.eval.report eval_results.jsonl --html docs/results/dashboard.html
python -m tests.eval.report eval_results.jsonl --markdown docs/results/raw/
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
    guardrails.py      # Guardrails facade — applies the full stack in foreign loops
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
    __main__.py        # CLI entry point: python -m forge.proxy
    proxy.py           # ProxyServer — programmatic start/stop API
    server.py          # Raw asyncio HTTP server, SSE streaming
    handler.py         # Request handler — bridge between HTTP and run_inference
    convert.py         # OpenAI messages ↔ forge Messages conversion
tests/
  unit/                # 865 deterministic tests — no LLM backend required
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

## Paper

The forge guardrail framework and ablation study are published as:

> Zambelli, A. *Forge: A Reliability Layer for Self-Hosted LLM Tool-Calling.*
> [https://doi.org/10.1145/3786335.3813193](https://doi.org/10.1145/3786335.3813193)

A pre-publication preprint is also available at [docs/forge_ieee_preprint.pdf](docs/forge_ieee_preprint.pdf) — kept as a historical artifact. Cite the published version above; the DOI link may not resolve immediately depending on the publisher's release timing.

## License

[MIT](LICENSE) — Copyright (c) 2025-2026 Antoine Zambelli
