# User Guide

Practical usage patterns for forge — from single-turn tool calling to multi-turn conversations.

For model and backend selection, see [MODEL_GUIDE.md](MODEL_GUIDE.md). For backend installation, see [BACKEND_SETUP.md](BACKEND_SETUP.md).

---

## Integration Modes

Forge's guardrail stack (retry nudges, step enforcement, error recovery, context compaction, VRAM budgeting) can be consumed in three ways. All three share the same underlying guardrail logic.

### At a glance

Each mode trades control for convenience. WorkflowRunner handles everything; the proxy applies guardrails transparently but drops workflow-level features; the middleware gives you building blocks and nothing else.

| Feature | WorkflowRunner | Proxy | Middleware |
|---------|:-:|:-:|:-:|
| Validation + rescue parsing | Yes | Yes | Yes |
| Retry nudges | Yes | Yes | Yes |
| Respond tool | Caller adds | Auto-injected | Caller adds |
| Step enforcement | Yes | No | Yes (caller wires) |
| Prerequisites | Yes | No | Yes (caller wires) |
| Max iterations | Yes | Bounded by max_retries | Caller's responsibility |
| Context compaction | Yes | Yes | Caller wires ContextManager |
| Context threshold warnings | Yes | No | Caller wires ContextManager |
| Cancellation | Between iterations | Between retries | Caller's responsibility |
| Streaming (token-by-token) | Yes | Post-hoc SSE | Caller's responsibility |
| Tool execution | Yes | No (client executes) | No (caller executes) |
| Callbacks (on_message, on_compact) | Yes | No | No |

The proxy is intentionally bare-bones — it applies response-quality guardrails (validation, rescue, retry, respond tool) without requiring workflow knowledge. Features like step enforcement and prerequisites require workflow structure that doesn't exist in the OpenAI chat completions API. See [Proxy design boundaries](#proxy-design-boundaries) for details.

### Mode 1: Standalone Runner (batteries included)

Forge owns the full agentic loop — LLM communication, guardrail policy, tool execution, and orchestration. You provide tools and a task, forge handles everything.

```python
from forge import WorkflowRunner

runner = WorkflowRunner(client=client, context_manager=ctx)
result = await runner.run(workflow, "What's the weather in Paris?")
```

**Best for:** Projects where forge is the primary framework. Scripts, pipelines, and applications built around forge from the start. See [Single-Turn Workflow](#single-turn-workflow) and [Multi-Turn Conversations](#multi-turn-conversations) below.

### Mode 2: Proxy Server (drop-in, zero code changes)

Forge sits between any OpenAI-compatible client and your model server, intercepting requests and applying guardrails transparently. The client doesn't know forge is there.

```bash
# External mode — you manage the backend
python -m forge.proxy --backend-url http://localhost:8080 --port 8081

# Managed mode — forge starts llama-server and the proxy together
python -m forge.proxy --backend llamaserver --gguf path/to/model.gguf --port 8081
```

Then point any client at forge instead of the model server:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8081/v1")
```

**Best for:** Adding guardrails to existing tools without modifying them. Works with any tool that speaks the OpenAI-compatible API — no per-client wrappers needed.

**Reliability note:** The proxy automatically injects a synthetic `respond` tool when tools are present in the request. The model calls `respond(message="...")` instead of producing bare text, keeping it in tool-calling mode where forge's full guardrail stack applies. The `respond` call is stripped from the outbound response — the client sees a normal text response and never knows the tool exists. This is essential for small local models (~8B), which cannot be trusted to choose correctly between text and tool calls — eval testing showed that trusting the model's text intent dropped workflow completion from 100% to as low as 4%. Guiding the model to a tool is a must. See [ADR-013](decisions/013-text-response-intent.md) for the full analysis.

#### Proxy design boundaries

The proxy is intentionally bare-bones: it applies response-quality guardrails without requiring workflow knowledge. The following features are available in WorkflowRunner but not in the proxy, by design:

- **Step enforcement and prerequisites.** These require workflow structure (required steps, terminal tool, tool dependencies) that doesn't exist in the OpenAI chat completions API. The proxy receives tool definitions per request but has no concept of workflow progression. If you need step enforcement, use WorkflowRunner or the middleware directly.

- **Max iterations.** The proxy calls `run_inference` once per request. Each call is bounded at `max_retries + 1` LLM attempts (default 4). There is no outer loop — a runaway model cannot loop indefinitely. This is sufficient for the proxy's single-request model.

- **Real streaming.** The proxy accepts `stream=true` and returns SSE events, but the full inference completes before SSE conversion. Token-by-token streaming during inference would require validating partial responses, which is incompatible with guardrails that need complete responses (rescue parsing, retry nudges). The guardrail-first design is the proxy's value proposition.

- **Context threshold warnings.** The proxy is stateless — the client sends the full conversation history in every request and decides what to include. Context pressure is the client's concern. Compaction still fires when the budget is exceeded.

- **Cancellation on disconnect.** Client disconnects are detected but do not cancel in-flight inference. This is the same granularity as WorkflowRunner, which checks `cancel_event` between loop iterations but does not interrupt a running LLM call. The worst case is `max_retries + 1` wasted calls (default 4) for a disconnected client.

### Mode 3: Middleware (composable guardrails)

Import forge's guardrail components directly into your own orchestration loop. You own the loop, forge provides the reliability logic.

**Simple API** (two calls -- covers most use cases):

```python
from forge.guardrails import Guardrails

guardrails = Guardrails(
    tool_names=["search", "lookup", "answer"],
    required_steps=["search", "lookup"],
    terminal_tool="answer",
)

# After each LLM response:
result = guardrails.check(response)

if result.action in ("retry", "step_blocked"):
    messages.append({"role": result.nudge.role, "content": result.nudge.content})
    continue

if result.action == "fatal":
    raise RuntimeError(result.reason)

# result.action == "execute" -- run the tools, then tell forge what succeeded:
execute(result.tool_calls)
done = guardrails.record([tc.tool for tc in result.tool_calls])
```

**Granular API** (individual components for custom control):

```python
from forge.guardrails import ResponseValidator, StepEnforcer, ErrorTracker

validator = ResponseValidator(tool_names=["search", "lookup", "answer"])
enforcer = StepEnforcer(required_steps=["search", "lookup"], terminal_tools=frozenset(["answer"]))
errors = ErrorTracker(max_retries=3, max_tool_errors=2)

# Inside your loop:
result = validator.validate(response)
if result.needs_retry:
    errors.record_retry()
    messages.append({"role": result.nudge.role, "content": result.nudge.content})
    continue

step_check = enforcer.check(result.tool_calls)
if step_check.needs_nudge:
    messages.append({"role": step_check.nudge.role, "content": step_check.nudge.content})
    continue

for tc in result.tool_calls:
    ok = execute(tc)
    enforcer.record(tc.tool)
    errors.record_result(success=ok)
```

**What you own:** The middleware provides validation, rescue parsing, retry nudges, and step enforcement. Your loop is responsible for: iteration caps, cancellation, context management (including compaction and threshold callbacks), and streaming. These are handled automatically by WorkflowRunner but are intentionally left to the caller in middleware mode — the middleware is an advisory layer, not an execution engine.

**Best for:** Framework developers embedding forge's guardrails inside a custom agent, a proprietary pipeline, or another open-source framework. For a complete runnable example showing both APIs, see [`examples/foreign_loop.py`](../examples/foreign_loop.py). For design rationale, see [ADR-011](decisions/011-guardrail-middleware.md).

### How they relate

```
forge.guardrails/            <-- extracted guardrail logic
    ^                ^
forge.server         forge.core.runner
(proxy mode)         (standalone mode)
```

The middleware layer is the foundation. Both the proxy server and the standalone runner compose the same guardrail components internally. The proxy wraps them behind an OpenAI-compatible API. The runner wraps them in a complete agentic loop. The middleware exposes them as building blocks.

| | Standalone | Proxy | Middleware |
|---|---|---|---|
| Who owns the loop? | Forge | Forge (transparent) | You |
| Code changes needed? | Build on forge | Change one URL | Import + integrate |
| Works with existing tools? | No | Yes | Depends on integration |
| Best for | New projects | Existing toolchains | Framework developers |

---

## Concepts

A forge workflow has four main pieces:

- **Tools** — Python functions the LLM can call, each described by a `ToolSpec` with typed parameters.
- **Workflow** — A named bundle of tools, with optional `required_steps` (tools the LLM *must* call) and a `terminal_tool` (the tool or tools that end the workflow — accepts `str` or `list[str]`).
- **Client** — An LLM backend adapter (`OllamaClient`, `LlamafileClient`, `AnthropicClient`).
- **Runner** — `WorkflowRunner` drives the agentic loop: send messages, parse tool calls, execute tools, enforce guardrails, manage compaction.

---

## Single-Turn Workflow

A two-step weather workflow: look up weather, then report it.

```python
from pydantic import BaseModel, Field
from forge.core.workflow import Workflow, ToolDef, ToolSpec
from forge.core.runner import WorkflowRunner
from forge.clients.llamafile import LlamafileClient
from forge.server import setup_backend, BudgetMode

# Define tools
def get_weather(city: str) -> str:
    return f"72°F and sunny in {city}"

def report_weather(city: str, weather: str) -> str:
    return f"Weather report: {weather}"

class GetWeatherParams(BaseModel):
    city: str = Field(description="City name")

class ReportWeatherParams(BaseModel):
    city: str = Field(description="City name")
    weather: str = Field(description="Weather description")

workflow = Workflow(
    name="weather",
    description="Look up weather and report it.",
    tools={
        "get_weather": ToolDef(
            spec=ToolSpec(
                name="get_weather",
                description="Get current weather for a city",
                parameters=GetWeatherParams,
            ),
            callable=get_weather,
        ),
        "report_weather": ToolDef(
            spec=ToolSpec(
                name="report_weather",
                description="Report the weather",
                parameters=ReportWeatherParams,
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

### What happens under the hood

1. `setup_backend()` starts the server, detects available VRAM, and calculates a context budget.
2. `WorkflowRunner.run()` builds a system prompt describing the available tools.
3. The LLM calls `get_weather(city="Paris")` — forge executes it and feeds the result back.
4. Step enforcement verifies `get_weather` was called (it's in `required_steps`).
5. The LLM calls `report_weather(...)` — forge executes it, sees it's the `terminal_tool`, and ends the loop.
6. If any step fails: retry nudges, rescue loops, and error recovery kick in automatically.

---

## Multi-Turn Conversations

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

The system prompt lives in `conversation` from turn 0 — it is not rebuilt or duplicated on subsequent turns. `StepEnforcer` and `tool_call_counter` reset each `run()` call since they are per-turn state.

### Long-Running Sessions: Filtering Transient Messages

`on_message` emits everything the runner creates during a turn, including transient retry artifacts — failed bare text responses, retry nudges, step nudges, and prerequisite nudges. This is by design: consumers get full visibility for logging and debugging.

For long-running sessions where conversation history persists across turns, these transient messages accumulate. The model sees its own past failures and corrective nudges on every subsequent turn, polluting effective context and degrading coherence — especially on smaller models (8-14B).

**Who's affected:** Any consumer that appends all `on_message` outputs to a persistent message list and reuses it via `initial_messages` on subsequent turns.

**Not affected:** Single-shot workflows, eval scenarios, or consumers that rebuild the message list from scratch each turn.

**Fix:** Filter transient message types before persisting. The metadata already tags these:

```python
from forge.core.messages import MessageType

TRANSIENT_TYPES = {
    MessageType.RETRY_NUDGE,
    MessageType.STEP_NUDGE,
    MessageType.PREREQUISITE_NUDGE,
    MessageType.TEXT_RESPONSE,
}

def on_message(self, msg: Message) -> None:
    if msg.metadata.type not in TRANSIENT_TYPES:
        self.messages.append(msg)
```

`TEXT_RESPONSE` is included because in tool-calling workflows, bare text is always a failed attempt that triggered a retry — the successful response comes as a `TOOL_CALL`. Consumers using the respond tool for conversational replies should keep `TEXT_RESPONSE` in their persist list.

**Why not fix this in forge?** The runner's job is to emit everything — within a turn, retry nudges are useful (the model needs to see the nudge to self-correct). The distinction between "within a turn" and "across turns" is a consumer concern. Compaction handles context overflow but doesn't proactively clean up transient messages — it fires based on token budget pressure, not session hygiene.

---

## Choosing a Backend

| Backend | Best for | Native FC? | Setup |
|---------|----------|------------|-------|
| **Ollama** | Easiest setup, model management built-in | Yes | `ollama serve` |
| **llama-server** | Best performance, full control | Yes (with `--jinja`) | `llama-server -m model.gguf --jinja` |
| **Llamafile** | Single binary, zero dependencies | No (prompt-injected) | Download and run |
| **Anthropic** | Frontier baseline, hybrid workflows | Yes | API key only |

See [BACKEND_SETUP.md](BACKEND_SETUP.md) for full installation instructions and [MODEL_GUIDE.md](MODEL_GUIDE.md) for which model to pick.

---

## Context Management

Forge automatically manages the context window. When the conversation approaches the budget limit, tiered compaction fires:

- **Phase 1** — Summarize older tool results, keep recent messages intact.
- **Phase 2** — Compress mid-conversation exchanges, preserve system prompt and recent context.
- **Phase 3** — Aggressive compression, retain only system prompt and last few exchanges.

You can configure this via the `ContextManager`:

```python
from forge.context import ContextManager, TieredCompact, NoCompact

# Default: tiered compaction with 2 recent messages preserved
ctx = ContextManager(strategy=TieredCompact(keep_recent=2), budget_tokens=8192)

# No compaction (for short workflows that won't hit the limit)
ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
```

Or let `setup_backend()` handle it — it detects your VRAM and calculates the budget automatically.

---

## Guardrails

Forge's guardrail stack runs automatically. Each layer can be independently disabled via [ablation presets](../tests/eval/ablation.py) for testing:

| Guardrail | What it does |
|-----------|-------------|
| **Step enforcement** | Verifies required tools were called before the terminal tool fires |
| **Prerequisites** | Enforces conditional tool dependencies (e.g. must read before edit) |
| **Retry nudges** | Prompts the LLM to try again when a tool call fails validation |
| **Rescue loops** | Recovers malformed tool calls from the LLM's text output |
| **Error recovery** | Re-prompts after tool execution errors instead of crashing |
| **Compaction** | Prevents context overflow in long conversations |

The eval harness measures each guardrail's contribution — see [EVAL_GUIDE.md](EVAL_GUIDE.md) for ablation results.

---

## Tool Prerequisites

Tools can declare conditional dependencies — "if you call this tool, you must have called tool X first." This is enforced at runtime via nudge-and-retry, the same pattern as step enforcement.

```python
ToolDef(
    spec=edit_spec,
    callable=edit_file,
    # Name-only: any prior call to read_file satisfies it
    prerequisites=["read_file"],
)

ToolDef(
    spec=edit_spec,
    callable=edit_file,
    # Arg-matched: must have called read_file with the same path
    prerequisites=[{"tool": "read_file", "match_arg": "path"}],
)
```

If the model calls a tool without satisfying its prerequisites, the runner blocks the batch, emits a `PREREQUISITE_NUDGE`, and the model retries. After `max_prereq_violations` (default 2) consecutive violations, `PrerequisiteError` is raised.

Prerequisites are not included in the tool schema — the model discovers constraints via nudge, same as step enforcement.

---

## Multiple Terminal Tools

Workflows can have multiple valid exit points. Pass a list to `terminal_tool`:

```python
workflow = Workflow(
    ...
    terminal_tool=["set_ac", "no_action"],  # either can end the workflow
)
```

Internally normalized to a `frozenset` for O(1) membership checks. A single string is still accepted and works as before.

---

## Cancellation

`WorkflowRunner.run()` accepts an optional `cancel_event` parameter for cooperative cancellation:

```python
import asyncio

cancel = asyncio.Event()

# In another coroutine or callback:
cancel.set()

try:
    result = await runner.run(workflow, "task", cancel_event=cancel)
except WorkflowCancelledError as e:
    print(f"Cancelled at iteration {e.iteration}")
    print(f"Completed steps: {e.completed_steps}")
    print(f"Messages so far: {len(e.messages)}")
```

The runner checks the event once per iteration, before the inference call. This is cooperative — if the model is mid-inference, the runner waits for it to finish before checking. The `WorkflowCancelledError` includes the full conversation state for the caller to resume, discard, or log.

---

## SlotWorker — Shared Slot Access

`SlotWorker` serializes workflow execution on a single inference slot with priority-based queuing and auto-preemption. Use it when multiple callers need to share a slot — for example, a home assistant's specialist workflows (calendar, AC management, escalation) all sharing slot 1 while the main conversation runs on slot 0.

### Basic usage (FIFO)

```python
from forge import SlotWorker, WorkflowRunner

# One runner pinned to a slot, one worker wrapping it
runner = WorkflowRunner(client=client, context_manager=ctx)
worker = SlotWorker(runner)
await worker.start()

# From anywhere — multiple concurrent callers are serialized
result = await worker.submit(workflow, "do the thing")
```

### Priority

Priority is an `int` — lower values run first. Forge imposes no semantics; the consumer defines what the levels mean:

```python
# Consumer defines their own levels
USER = 0
ESCALATED = 1
ROUTINE = 2

# User-initiated request — highest priority
result = await worker.submit(calendar_wf, "what's on my schedule?", priority=USER)

# Background cron — lowest priority, can be preempted
result = await worker.submit(ac_wf, "check temperature", priority=ROUTINE)
```

Without an explicit priority, all tasks default to 0 (pure FIFO).

### Auto-preemption

If a higher-priority task is submitted while a lower-priority task is running, the running task is automatically cancelled and the higher-priority task takes over. The cancelled task's `submit()` raises `WorkflowCancelledError`.

```python
# Routine AC check is running (priority=2)...
# User asks about calendar (priority=0) — AC check is auto-cancelled
result = await worker.submit(calendar_wf, "what's next?", priority=0)
```

You can also cancel manually:

```python
worker.cancel_current()  # cancels whatever is running
```

### Multi-slot architecture

For multi-slot setups (e.g., with `--kv-unified`), create one `SlotWorker` per shared slot. The main conversation slot typically doesn't need a worker — it's dedicated to one persistent session.

```python
# Slot 0: main conversation (no worker needed — dedicated)
main_client = LlamafileClient(model="...", slot_id=0)
main_runner = WorkflowRunner(client=main_client, context_manager=ctx)

# Slot 1: shared specialist slot (needs a worker)
service_client = LlamafileClient(model="...", slot_id=1)
service_runner = WorkflowRunner(client=service_client, context_manager=ctx)
service_worker = SlotWorker(service_runner)
await service_worker.start()

# Tools route through the worker
async def query_calendar(**kwargs):
    return await service_worker.submit(calendar_wf, kwargs["query"], priority=0)
```

