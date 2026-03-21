# User Guide

Practical usage patterns for forge — from single-turn tool calling to multi-turn conversations.

For model and backend selection, see [MODEL_GUIDE.md](MODEL_GUIDE.md). For backend installation, see [BACKEND_SETUP.md](BACKEND_SETUP.md).

---

## Integration Modes

Forge's guardrail stack (retry nudges, step enforcement, error recovery, context compaction, VRAM budgeting) can be consumed in three ways. All three share the same underlying guardrail logic.

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

**Reliability note:** The proxy sets `trust_text_intent=True`, meaning it trusts the backend's finish reason when the model responds with text instead of calling tools. This eliminates retry latency on conversational turns (e.g. the user says "hi" in a tool-equipped session) but means the proxy won't nudge the model if it should have called a tool. In eval testing with an 8B model (Ministral 3 8B Reasoning Q4_K_M), unconditionally trusting intent dropped workflow completion from 100% to as low as 4% on reasoning-heavy scenarios. WorkflowRunner and middleware default to `trust_text_intent=False` and are not affected. See [ADR-013](../decisions/013-text-response-intent.md) for the full analysis.

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
enforcer = StepEnforcer(required_steps=["search", "lookup"], terminal_tool="answer")
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
- **Workflow** — A named bundle of tools, with optional `required_steps` (tools the LLM *must* call) and a `terminal_tool` (the tool that ends the workflow).
- **Client** — An LLM backend adapter (`OllamaClient`, `LlamafileClient`, `AnthropicClient`).
- **Runner** — `WorkflowRunner` drives the agentic loop: send messages, parse tool calls, execute tools, enforce guardrails, manage compaction.

---

## Single-Turn Workflow

A two-step weather workflow: look up weather, then report it.

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
| **Retry nudges** | Prompts the LLM to try again when a tool call fails validation |
| **Rescue loops** | Recovers malformed tool calls from the LLM's text output |
| **Error recovery** | Re-prompts after tool execution errors instead of crashing |
| **Compaction** | Prevents context overflow in long conversations |

The eval harness measures each guardrail's contribution — see [EVAL_GUIDE.md](EVAL_GUIDE.md) for ablation results.

