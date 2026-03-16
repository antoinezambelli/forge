# forge

A Python framework for self-hosted LLM tool-calling and multi-step agentic workflows. Supports Ollama, llama-server (llama.cpp), Llamafile, and Anthropic as backends, with VRAM-aware context budgets, tiered compaction, and guardrail-based reliability (step enforcement, retry nudges, rescue loops).

## Requirements

- Python 3.12+

## Setup

```bash
python -m venv .venv
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest tests/ -v --tb=short
```

### Coverage

```bash
python -m pytest tests/ --cov=forge --cov-report=term-missing
```

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
    scenarios/         # 29 eval scenarios (lambda, stateful, compaction chain)
      _base.py         # EvalScenario dataclass, ALL_SCENARIOS registry
      _plumbing.py     # basic_2step, sequential_3step, error_recovery, compaction_stress
      _model_quality.py # tool_selection, argument_fidelity, sequential_reasoning, etc.
      _compaction.py   # phase2_compaction, relevance_detection
      _compaction_chain.py # Multi-phase compaction retention (baseline, P1, P2, P3)
      _stateful_plumbing.py    # Stateful variants of plumbing scenarios
      _stateful_model_quality.py # Stateful variants of model quality scenarios
      _stateful_compaction.py  # Stateful compaction + inventory_audit, supplier_deep_dive
    eval_runner.py     # Run scenarios N times, collect per-run results
    metrics.py         # Aggregate results, compute metrics, print report
    batch_eval.py      # Batch runner — iterate configs, JSONL output, resume
    report.py          # ASCII table + list report from JSONL, HTML dashboard, markdown views
    ablation.py        # AblationConfig presets for guardrail isolation
    dashboard/         # React-based interactive HTML dashboard (built via report.py --html)
    bfcl/              # BFCL v4 benchmark integration
      runner.py        # Single-turn + multi-turn BFCL runner
      scorer.py        # Pass/fail scoring against ground truth
      schema_adapter.py # BFCL JSON → forge ToolDef conversion
      executors.py     # BFCL backend execution wrappers
      backends/        # BFCL multi-turn backend implementations (14 APIs)
      checker/         # AST-based call validation, multi-turn state checking
      batch_runner.py  # Batch runner — all configs, JSONL output, resume
      bfcl_report.py   # ASCII table report from BFCL JSONL
      smoke_test.py    # Quick sanity check against a live backend
```

## Usage

### Single-Turn

```python
from forge.core.workflow import Workflow, ToolDef, ToolSpec, ToolParam
from forge.core.runner import WorkflowRunner
from forge.clients.llamafile import LlamafileClient
from forge.server import setup_backend, BudgetMode

# Define tools
def get_weather(city: str) -> str:
    return f"72°F and sunny in {city}"

def report_weather(city: str, weather: str) -> str:
    return f"Weather report: {weather}"

workflow = Workflow(
    name="weather",
    description="Look up weather and report it.",
    tools={
        "get_weather": ToolDef(
            spec=ToolSpec(
                name="get_weather",
                description="Get current weather for a city",
                parameters=[ToolParam("city", "string", "City name", required=True)],
            ),
            callable=get_weather,
        ),
        "report_weather": ToolDef(
            spec=ToolSpec(
                name="report_weather",
                description="Report the weather",
                parameters=[
                    ToolParam("city", "string", "City name", required=True),
                    ToolParam("weather", "string", "Weather description", required=True),
                ],
            ),
            callable=report_weather,
        ),
    },
    required_steps=["get_weather"],
    terminal_tool="report_weather",
)

# setup_backend() auto-manages llama-server: starts the process, health-checks,
# resolves a VRAM-aware context budget, and returns a ContextManager ready to use.
server, ctx = await setup_backend(
    backend="llamaserver",
    model="ministral-8b-instruct",
    gguf_path="path/to/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
    budget_mode=BudgetMode.FORGE_FULL,
)
# Or manage the server yourself and create the ContextManager directly:
# ctx = ContextManager(strategy=TieredCompact(keep_recent=2), budget_tokens=8192)

client = LlamafileClient(model="ministral-8b-instruct", mode="native")
runner = WorkflowRunner(client=client, context_manager=ctx, stream=True)
await runner.run(workflow, "What's the weather in Paris?")
await server.stop()
```

### Multi-Turn Conversations

`WorkflowRunner` accepts an optional `on_message` callback that fires each time a `Message` is appended to the conversation during `run()`. This is the primary observability hook — use it for logging, eval metric collection, or building conversation history for multi-turn flows.

- **Single-turn (default):** `on_message` fires for every message the runner creates — system prompt, user input, assistant responses, tool results, nudges.
- **Multi-turn (`initial_messages`):** `run()` accepts an optional `initial_messages` parameter that seeds the conversation with prior history. `on_message` fires **only for new messages created during this turn**, not for the replayed history.

`WorkflowRunner` does not manage server lifecycle or track conversation history across `run()` calls — both are the consumer's responsibility.

```python
from forge.server import setup_backend, BudgetMode
from forge.core.runner import WorkflowRunner
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType

# 1. Start server once — stays up for the lifetime of the consumer
client = OllamaClient(model="ministral-3:8b-instruct-2512-q4_K_M")
server, ctx = await setup_backend(
    backend="ollama", model="ministral-3:8b-instruct-2512-q4_K_M",
    budget_mode=BudgetMode.FORGE_FULL, client=client,
)

# 2. Consumer owns the conversation history
conversation: list[Message] = []

# Turn 0 — normal run, on_message collects everything (system prompt, user input, etc.)
runner = WorkflowRunner(client=client, context_manager=ctx,
                        on_message=lambda msg: conversation.append(msg))
await runner.run(workflow, "first question")

# Turn 1+ — seed with full history, append new user message
turn_messages: list[Message] = []
runner = WorkflowRunner(client=client, context_manager=ctx,
                        on_message=lambda msg: turn_messages.append(msg))
seed = list(conversation)
seed.append(Message(MessageRole.USER, "follow-up question",
                    MessageMeta(MessageType.USER_INPUT)))
await runner.run(workflow, "follow-up question", initial_messages=seed)
conversation.extend(turn_messages)

# 3. Shut down when the consumer is done (not per-turn)
await server.stop()
```

The system prompt lives in `conversation` from turn 0 — it is not rebuilt or duplicated on subsequent turns. `StepTracker` and `tool_call_counter` reset each `run()` call since they are per-turn state.

## Eval Harness

The eval harness measures how reliably a model + backend combo navigates multi-step tool-calling workflows. 29 scenarios across plumbing, model quality, compaction, and stateful categories. See [docs/EVAL_GUIDE.md](docs/EVAL_GUIDE.md) for full CLI reference, flag documentation, and examples.

Quick start:

```bash
# Ollama
python -m tests.eval.eval_runner --backend ollama --model "ministral-3:8b-instruct-2512-q4_K_M" --runs 10 --stream --verbose

# llama-server (start server first: llama-server --jinja -m model.gguf -ngl 999 --port 8080)
python -m tests.eval.eval_runner --backend llamafile --llamafile-mode native --model model-name --runs 10 --stream --verbose

# Batch eval (JSONL output, automatic resume)
python -m tests.eval.batch_eval --config all --runs 50

# Reports (ASCII table, HTML dashboard, markdown views)
python -m tests.eval.report eval_results.jsonl
```

### BFCL Benchmark

Run forge against [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) v4 tasks (11 categories, ~2,183 entries). See [docs/EVAL_GUIDE.md](docs/EVAL_GUIDE.md) for details.

## Roadmap

All three backends (Ollama, llama-server, Llamafile) are stable for Ministral, Qwen3, Llama 3.1, and Mistral Nemo. Open items:

1. **Multi-model routing** — Model pool for managing N backends simultaneously; consumer orchestrates which client each workflow uses. See [`docs/decisions/MULTI_MODEL_ROUTING.md`](docs/decisions/MULTI_MODEL_ROUTING.md).
2. **Tool prerequisites** — Conditional tool dependencies ("if you call B, you must have called A"). See [`docs/decisions/006-tool-prerequisites.md`](docs/decisions/006-tool-prerequisites.md).
3. **Context window self-awareness** — Inject remaining context budget at configurable thresholds so the model knows compaction is approaching before it fires. The compaction chain scenarios (P1/P2/P3) provide the eval harness to measure whether self-awareness improves retention.
4. **Compaction tiers** — Consumer-configurable per-phase compaction thresholds. Currently all 3 phases share one trigger (budget x 0.75). Expose phase thresholds (e.g. P1=60%, P2=75%, P3=90%) so consumers can tune the aggressiveness/retention tradeoff.

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Full design document
- [WORKFLOW.md](docs/WORKFLOW.md) — Workflow design and runner internals
- [EVAL_GUIDE.md](docs/EVAL_GUIDE.md) — Eval harness CLI reference, batch eval, BFCL benchmark
- [BACKEND_SETUP.md](docs/BACKEND_SETUP.md) — Backend installation and server setup
