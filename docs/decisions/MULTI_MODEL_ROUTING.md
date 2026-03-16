# Multi-Model Routing — Concept Doc

## Goal

Allow forge to manage multiple model backends simultaneously and expose
them as named clients to the consumer. Forge handles the pool (lifecycle,
health, budgets). The consumer handles orchestration (which workflow uses
which model, when to swap, event dispatch).

## Motivation

### Multi-workflow, multi-backend

A user with two GPUs — or one GPU with capacity for two small models —
wants to run different workflows against different models. Today this
requires manual client construction and port management. Forge should
make the pool management straightforward.

**Example — home assistant:** A lightweight conversational model runs
continuously for natural language interaction. A vision model (Gemma)
polls cameras for events. When something triggers (UPS delivery, motion
alert), the orchestrator spins up a workflow on a heavyweight reasoning
model to analyze and act. Three models, three roles — but the
orchestrator logic (event dispatch, model eviction, priority) is
consumer code. Forge just keeps the pool healthy.

### Single workflow, single model (today)

Nothing changes for the common case. A consumer with one model uses
forge exactly as today. The pool is a generalization, not a replacement.

## Current State

### What works today

- All three clients accept `base_url` as a constructor parameter —
  `OllamaClient(base_url="http://localhost:11434")`,
  `LlamafileClient(base_url="http://localhost:8080/v1")`,
  `AnthropicClient(model="claude-haiku-4-5-20251001")`
- Multiple client instances can coexist in the same process
- `ServerManager` accepts a `port` parameter for llama-server/llamafile

### What doesn't work

- `ServerManager` manages one server process at a time — starting a new
  model stops the old one
- No concept of a named model pool or client registry
- No per-model budget resolution when multiple models share VRAM
- `WorkflowRunner` takes a single `client` — consumer must construct
  separate runners for separate models

## Scope: What Forge Owns vs Consumer Owns

| Concern | Owner | Notes |
|---------|-------|-------|
| Start/stop model backends | **Forge** (model pool) | Health checks, port assignment, process lifecycle |
| VRAM-aware budget per model | **Forge** (pool + budget) | Each pool entry gets its own context budget |
| Named client registry | **Forge** (pool interface) | Consumer references models by user-supplied name |
| Which workflow uses which model | **Consumer** | Forge doesn't assign roles or route |
| Model eviction/swapping on single GPU | **Consumer** | Consumer calls pool.stop("A") then pool.start("C") |
| Event dispatch (camera trigger, timer, user input) | **Consumer** | Application-level orchestration |
| Mid-workflow model switching | **Consumer** | Consumer constructs workflow with the right client |
| Multi-agent coordination | **Consumer** | Forge runs single-model workflows; agents are consumer-level |

Forge is a library, not a framework. It manages infrastructure (the
pool) and execution (the runner). Policy decisions — which model for
which task, when to swap, how to orchestrate — belong to the consumer.

## Design

### Model Pool

A `ModelPool` replaces (or wraps) `ServerManager`. It manages N backend
processes, each identified by a user-supplied name:

```python
pool = ModelPool()

# Start multiple models
await pool.start("conversational", ModelConfig(
    backend="ollama",
    model="ministral-3:8b-instruct-2512-q4_K_M",
))
await pool.start("reasoning", ModelConfig(
    backend="llamaserver",
    gguf_path="/models/Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
    port=8081,
))
await pool.start("vision", ModelConfig(
    backend="llamaserver",
    gguf_path="/models/gemma-vision.gguf",
    port=8082,
))

# Get a client by name
client = pool.client("conversational")  # → OllamaClient
runner = WorkflowRunner(client=client)
result = await runner.run(workflow, user_message)

# Consumer manages lifecycle
await pool.stop("conversational")  # free VRAM
await pool.start("heavy", ModelConfig(...))  # load new model
```

Names are opaque strings — forge doesn't interpret them. The consumer
assigns meaning ("conversational", "vision", "heavy" — or "model_a",
"model_b", whatever).

### ModelConfig

Declarative per-model configuration:

```python
@dataclass
class ModelConfig:
    backend: str                    # "ollama", "llamaserver", "llamafile"
    model: str                      # model name or identifier
    gguf_path: Path | None = None   # for llamaserver/llamafile
    port: int = 8080                # for llamaserver/llamafile
    mode: str = "native"            # "native" or "prompt"
    budget_mode: BudgetMode = BudgetMode.FORGE_FULL
    extra_flags: list[str] | None = None  # e.g. ["--jinja", "--reasoning-format", "auto"]
```

### Pool lifecycle

- `start(name, config)` — starts the backend process (or connects to
  Ollama), resolves the context budget, creates the client, stores it
- `stop(name)` — stops the process, removes from pool
- `stop_all()` — shuts down everything
- `client(name)` → `LLMClient` — returns the client for a named model
- `budget(name)` → `int` — returns the resolved context budget
- `health(name)` → `bool` — checks if the backend is responsive
- `list()` → `dict[str, ModelConfig]` — lists active models

### ServerManager relationship

`ModelPool` wraps multiple `ServerManager`-like instances internally.
Each pool entry gets its own process tracking, health check, and budget
resolution. The existing `ServerManager` can be refactored into this,
or `ModelPool` can compose multiple `ServerManager` instances.

### Budget per model

Each pool entry gets its own context budget, resolved independently via
the existing `resolve_budget()` logic:

- Ollama entries: VRAM-tier budget (existing logic)
- llama-server entries: `/props` endpoint (existing logic)
- API entries (Anthropic): no VRAM constraint

When multiple local models share one GPU, VRAM budgets are **not
automatically partitioned** — forge doesn't know the consumer's intent.
The consumer manages this:
- Load both models, let each backend auto-tune its own budget
- Or explicitly set budgets via `BudgetMode.MANUAL`
- Or stop one model before starting another (sequential)

### WorkflowRunner: unchanged

The runner still takes a single `client`. No multi-client registry in
the runner, no routing logic. The consumer picks the client from the
pool and passes it to the runner:

```python
# Consumer orchestration code
client = pool.client("reasoning")
ctx = ContextManager(strategy=TieredCompact(), budget_tokens=pool.budget("reasoning"))
runner = WorkflowRunner(client=client, context_manager=ctx)
result = await runner.run(analysis_workflow, trigger_data)
```

This keeps the runner simple and the routing decision in consumer code
where it belongs.

## Design Space: Things We Considered But Deferred

### Mid-workflow model switching

A single workflow that switches models at step boundaries (e.g., text
model for gathering, vision model for analysis, text model for
synthesis). This would require the runner to accept multiple clients and
switch between them based on routing rules.

**Why deferred:** No concrete use case yet. The consumer can achieve
this today by chaining multiple `runner.run()` calls, passing context
between them. A first-class runner feature adds complexity to the
runner's core loop for a pattern that may be rare.

**If we revisit:** The natural trigger is step-gated — "after these
tools have been called, switch to client X." Today steps and tools are
1:1 in StepTracker, but they could diverge into composite steps
("research phase = search + read_file"). Routing would key off step
completion. Message history handoff works naturally since forge's
internal `Message` format is api_format-agnostic — `to_api_dict()`
re-serializes for each client.

### VRAM-aware auto-partitioning

Forge could detect total VRAM, subtract each model's weight, and
automatically compute per-model context budgets. This is complex
(requires knowing model sizes before loading) and fragile (backends
have their own VRAM management). Better to let each backend auto-tune
and let the consumer manage loading order.

### Model eviction policies

Automatic eviction (LRU, priority-based) when VRAM is full. This is
orchestration policy — consumer code. The pool provides `stop()` and
`start()`; the consumer decides when to call them.

## Touch Points

| Component | Change | Size |
|-----------|--------|------|
| `ModelPool` | New class, wraps ServerManager lifecycle for N models | Medium |
| `ModelConfig` | New dataclass for per-model configuration | Small |
| `ServerManager` | Refactor to support composition (or replace with pool internals) | Medium |
| `setup_backend()` | Update to work with pool or deprecate in favor of pool | Small |

### What doesn't change

- `WorkflowRunner` — still single-client, no routing
- `LLMClient` protocol — clients are unchanged
- `ContextManager` — still per-runner, budget set at construction
- `Workflow` / `ToolDef` / `ToolSpec` — no routing annotations
- Eval harness — batch_eval already manages sequential model switching

## Sequencing

Independent of other roadmap items. The pool is pure infrastructure —
no interaction with parallel tools, prerequisites, or BFCL.

Could be built incrementally:
1. `ModelConfig` dataclass + `ModelPool` with start/stop/client
2. Multi-port `ServerManager` (or pool-internal process tracking)
3. Per-model budget resolution
4. Migrate `batch_eval.py` to use pool (natural first consumer)

## References

- `src/forge/server.py` — ServerManager, BudgetMode, setup_backend()
- `src/forge/clients/` — OllamaClient, LlamafileClient, AnthropicClient
- `src/forge/core/runner.py` — WorkflowRunner
- `tests/eval/batch_eval.py` — current sequential model switching
- `src/forge/context/hardware.py` — detect_hardware(), HardwareProfile
