# User Guide

Practical usage patterns for forge — from single-turn tool calling to multi-turn conversations.

For model and backend selection, see [MODEL_GUIDE.md](MODEL_GUIDE.md). For backend installation, see [BACKEND_SETUP.md](BACKEND_SETUP.md). Upgrading a Proxy deployment from pre-0.9? Start with [Migrating to Forge 0.9](MIGRATING_TO_0.9.md).

---

## Integration Modes

Forge's guardrail stack (retry nudges, step enforcement, error recovery, context compaction, VRAM budgeting) can be consumed in three ways. All three share the same underlying guardrail logic.

### At a glance

Each mode trades control for convenience. WorkflowRunner handles everything; the proxy applies guardrails transparently but drops workflow-level features; the middleware gives you building blocks and nothing else.

| Feature | WorkflowRunner | Proxy | Middleware |
|---------|:-:|:-:|:-:|
| Validation + rescue parsing | Yes | Yes | Yes |
| Retry nudges | Yes | Yes | Yes |
| Respond tool | Caller adds | Opt-in flag | Caller adds |
| Step enforcement | Yes | No | Yes (caller wires) |
| Prerequisites | Yes | No | Yes (caller wires) |
| Max iterations | Yes | Bounded by max_retries | Caller's responsibility |
| Context compaction | Yes | No | Caller wires ContextManager |
| Last-completed context reporting | Caller observes | `/forge/usage` | Caller observes |
| Context threshold warnings | Yes | No | Caller wires ContextManager |
| Cancellation | Between iterations | Between retries | Caller's responsibility |
| Streaming (token-by-token) | Yes | Post-hoc SSE | Caller's responsibility |
| Tool execution | Yes | No (client executes) | No (caller executes) |
| Callbacks (on_message, on_compact) | Yes | No | No |

The proxy is intentionally bare-bones — it applies response-quality guardrails
(validation, rescue, retry, optional respond tool) to tool-bearing requests
without requiring workflow knowledge. Tool-free requests bypass those
guardrails and call the selected backend directly. Features like step
enforcement and prerequisites require workflow structure that doesn't exist in
the OpenAI chat completions API. See [Proxy design boundaries](#proxy-design-boundaries) for details.

### Mode 1: Standalone Runner (batteries included)

Forge owns the full agentic loop — LLM communication, guardrail policy, tool execution, and orchestration. You provide tools and a task, forge handles everything.

```python
from forge import WorkflowRunner

runner = WorkflowRunner(client=client, context_manager=ctx)
result = await runner.run(workflow, "What's the weather in Paris?")
```

**Best for:** Projects where forge is the primary framework. Scripts, pipelines, and applications built around forge from the start. See [Single-Turn Workflow](#single-turn-workflow) and [Multi-Turn Conversations](#multi-turn-conversations) below.

### Mode 2: Proxy Server (drop-in, zero code changes)

Forge sits between any client and your model server, applying guardrails to
tool-bearing requests and directly relaying tool-free inference through the
selected adapter. It speaks both the OpenAI chat-completions API and the
Anthropic Messages API (`/v1/messages`), so OpenAI-compatible tools and Claude
Code both work. The client doesn't know Forge is there.

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

**Best for:** Adding guardrails to existing tools without modifying them. Works with any tool that speaks the OpenAI-compatible API, plus Claude Code via the Anthropic Messages API — no per-client wrappers needed.

**Reliability note:** `--inject-respond-tool` explicitly opts into Forge's synthetic `respond` tool when tools are present; the default is off. When enabled, the call is stripped from the outbound response and the client sees normal text. See [ADR-013](decisions/013-text-response-intent.md) for the rationale and eval evidence.

#### Using forge with Claude Code

Claude Code speaks the Anthropic Messages API, which the proxy serves on `POST /v1/messages` — so you can point Claude Code at a forge-guarded local model. Start the proxy against any backend, then set two environment variables for the Claude Code process:

```bash
# Start the proxy (managed mode against a local GGUF; native FC by default)
python -m forge.proxy --backend llamaserver --gguf path/to/model.gguf --port 8081

# Point Claude Code at it — scope these to the claude process only
ANTHROPIC_BASE_URL=http://localhost:8081 \
ANTHROPIC_AUTH_TOKEN=forge \
claude
```

For an ungated local backend, `ANTHROPIC_AUTH_TOKEN` may be any non-empty
placeholder required by Claude Code. Forge still applies its zero-or-one
credential-identity forwarding rule; it does not authenticate the caller.
Response `model` is the effective configured, pinned, discovered, or
request-routed backend identity, not a fabricated Forge alias.

**Function-calling capability.** `--backend-capability native` (default) uses the backend's chat-template tool-calling and is the smoother default for Claude Code's heavy multi-turn tool use. `--backend-capability prompt` injects the tool surface into the prompt for llama.cpp/llamafile backends without a tool-calling template; whether a model stays coherent across multi-turn tool results in prompt mode varies by model — and tends to degrade on more complex, multi-step interactions — so prefer native whenever the backend supports it. The capability is declared at startup and frozen.

**Reasoning replay.** Reasoning-capable backends may return hidden reasoning alongside tool calls. Forge captures that reasoning for observability, then controls how much is replayed to the backend on later turns with `--reasoning-replay {full,keep-last,none}`. The default is `none`: captured reasoning stays out of backend-facing history entirely. This is the most token-efficient policy, and on forge's eval suite it is statistically indistinguishable from replay-all (no aggregate score cost; see [reasoning-replay results](results/raw/reasoning-replay.md)). `keep-last` replays only the latest captured reasoning block. `full` preserves the historical behavior and replays every captured reasoning block. In OpenAI-compatible proxy responses, `keep-last` exposes current reasoning as `reasoning_content` instead of normal assistant `content` so clients that preserve reasoning fields can replay only the latest block without turning it into plain text; under the default `none`, proxy responses omit captured reasoning. Anthropic proxy responses only emit reasoning text under `full`; Forge does not synthesize signed Anthropic thinking blocks, so default Anthropic proxy responses do not expose replayable reasoning. See [ADR-017](decisions/017-reasoning-replay-policy.md) for the policy design and the eval evidence behind the default.

**Downstream profile.**

- **Local/OpenAI-shaped downstream (external omission or `--backend openai`)** — forge translates Claude Code's Anthropic requests to OpenAI and converts the reply back to Anthropic SSE. Anthropic-only fields with no OpenAI analog (`cache_control`, `thinking`, `document` blocks) are dropped at that boundary; see [ADR-015](decisions/015-cache-control-preservation-path1.md).
- **Anthropic-shaped downstream (`--backend anthropic`, external mode)** — forge forwards to an Anthropic Messages endpoint (for example LiteLLM or the Anthropic API), preserving applicable unknown fields and `litellm_session_id`. This path uses HTTP directly, with no Anthropic SDK dependency. Native inbound requests retain their `anthropic-version` through retries; missing versions and cross-protocol requests default to `2023-06-01`. The `2023-06-01` and `2026-10-21` contracts share the same body/response machinery; Forge does not silently rewrite invalid fields to fit a different API version.

#### Proxy design boundaries

The proxy is intentionally bare-bones: it applies response-quality guardrails
to tool-bearing requests without requiring workflow knowledge; tool-free
requests bypass validation, rescue, and retry. The following features are
available in WorkflowRunner but not in the proxy, by design:

- **Step enforcement and prerequisites.** These require workflow structure (required steps, terminal tool, tool dependencies) that doesn't exist in the OpenAI chat completions API. The proxy receives tool definitions per request but has no concept of workflow progression. If you need step enforcement, use WorkflowRunner or the middleware directly.

- **Max iterations.** The proxy calls `run_inference` once per request. Each call is bounded at `max_retries + 1` LLM attempts (default 4). There is no outer loop — a runaway model cannot loop indefinitely. This is sufficient for the proxy's single-request model.

- **Real streaming.** The proxy accepts `stream=true` and returns SSE events, but the full inference completes before SSE conversion. Token-by-token streaming during inference would require validating partial responses, which is incompatible with guardrails that need complete responses (rescue parsing, retry nudges). The guardrail-first design is the proxy's value proposition.

- **Context mutation.** The client sends the full conversation history in every request and decides what to include. Proxy always uses `NoCompact`; budgets and metadata report context but never trim caller history. `/forge/usage` exposes only the one last-completed eligible process-local snapshot (or 204), not session memory or a persistent ledger. In unmanaged mode the operator owns overflow, model swaps, and backend failures.

- **Cancellation on disconnect.** Client disconnects are detected but do not cancel in-flight inference. This is the same granularity as WorkflowRunner, which checks `cancel_event` between loop iterations but does not interrupt a running LLM call. The worst case is `max_retries + 1` wasted calls (default 4) for a disconnected client.

**Operations and observability.** Use `/forge/health` for Forge liveness and
the forwarded `/health` for backend readiness. The honest backend catalog is
forwarded at `/v1/models`; `/v1/health`, `/models`, and `/props` are also on the
closed read-only forwarding allowlist. `/forge/usage` is a last-completed
process-local snapshot, not a live meter, ledger, or durable session API.

Forge Proxy is a per-operator sidecar and does not authenticate callers.
`--backend-api-key` is a convenience for authenticating Forge to its backend,
not caller authorization. Put an authentication gateway in front for a
centralized or multi-tenant deployment. See [Backend Setup](BACKEND_SETUP.md)
for the zero-or-one credential-identity rule.

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

The `Guardrails` facade covers validation, retry nudges, and step enforcement. It does **not** enforce tool prerequisites — for those, use the Granular API (`StepEnforcer.check_prerequisites`) shown next.

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

- **Tools** — Python functions the LLM can call, each described by a `ToolDefinition` with a complete input schema.
- **Workflow** — A named bundle of tools, with optional `required_steps` (tools the LLM *must* call) and a `terminal_tool` (the tool or tools that end the workflow — accepts `str` or `list[str]`).
- **Client** — An LLM backend adapter (`OpenAICompatClient`, `OllamaClient`,
  `LlamafileClient`, `VLLMClient`, or `AnthropicClient`).
- **Runner** — `WorkflowRunner` drives the agentic loop: send messages, parse tool calls, execute tools, enforce guardrails, manage compaction.

---

## Single-Turn Workflow

A two-step weather workflow: look up weather, then report it.

```python
from forge.core.workflow import Workflow, ToolDefinition
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
        "get_weather": ToolDefinition(
            name="get_weather",
            description="Get current weather for a city",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        ),
        "report_weather": ToolDefinition(
            name="report_weather",
            description="Report the weather",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "weather": {"type": "string", "description": "Weather description"},
                },
                "required": ["city", "weather"],
            },
        ),
    },
    handlers={"get_weather": get_weather, "report_weather": report_weather},
    required_steps=["get_weather"],
    terminal_tool="report_weather",
    system_prompt_template="Use the available tools to answer the user.",
)

# setup_backend() auto-manages llama-server: starts the process, health-checks,
# resolves a VRAM-aware context budget, and returns a ContextManager ready to use.
server, ctx = await setup_backend(
    backend="llamaserver",
    gguf_path="path/to/Ministral-3-8B-Instruct-2512-Q8_0.gguf",
    budget_mode=BudgetMode.FORGE_FULL,
)
# Or manage the server yourself and create the ContextManager directly:
# ctx = ContextManager(strategy=TieredCompact(keep_recent=2), budget_tokens=8192)

client = LlamafileClient(
    gguf_path="path/to/Ministral-3-8B-Instruct-2512-Q8_0.gguf",
    mode="native",
    recommended_sampling=True,
)
runner = WorkflowRunner(client=client, context_manager=ctx, stream=True)
await runner.run(workflow, "What's the weather in Paris?")
await server.stop()
```

### Experimental llama.cpp RPC

`setup_backend()` can own one remote `ggml-rpc-server` worker and one local
`llama-server` coordinator. The worker runs in the foreground of an SSH
connection, so normal stop and restart target the exact SSH child Forge
started. An abruptly lost SSH transport that leaves a remote worker behind
must be cleared by the operator.

```python
from forge import (
    BudgetMode,
    LlamaCppRpcConfig,
    LlamaCppRpcWorkerConfig,
    setup_backend,
)

rpc = LlamaCppRpcConfig(
    worker=LlamaCppRpcWorkerConfig(
        ssh_target="model-worker@10.0.0.5",
        rpc_host="10.0.0.5",
        executable="/opt/llama.cpp/build/bin/ggml-rpc-server",
        device="Vulkan0",
        environment=(("AMD_VULKAN_ICD", "RADV"),),
    ),
    coordinator_executable="/opt/llama.cpp/build/bin/llama-server",
    coordinator_environment=(("AMD_VULKAN_ICD", "RADV"),),
    devices=("Vulkan0", "RPC0"),
    tensor_split=(1, 1),
    startup_timeout=1800,  # allow a large distributed model to load
    log_directory="logs/llama-rpc",
)

server, ctx = await setup_backend(
    backend="llamaserver",
    gguf_path="/models/model-00001-of-00005.gguf",
    budget_mode=BudgetMode.MANUAL,
    manual_tokens=262_144,
    cache_type_k="q8_0",
    cache_type_v="q8_0",
    extra_flags=["--fit", "off", "--no-mmap", "-fa", "on"],
    rpc=rpc,
)

# The latest worker/coordinator logs remain available after stop or failure.
print(server.rpc_log_paths)
await server.stop()
```

### What happens under the hood

1. `setup_backend()` starts the server, detects available VRAM, and calculates a context budget.
2. `WorkflowRunner.run()` builds a system prompt describing the available tools.
3. The LLM calls `get_weather(city="Paris")` — forge executes it and feeds the result back.
4. Step enforcement verifies `get_weather` was called (it's in `required_steps`).
5. The LLM calls `report_weather(...)` — forge executes it, sees it's the `terminal_tool`, and ends the loop.
6. If any step fails: retry nudges, rescue loops, and error recovery kick in automatically. Step-enforcement and prerequisite violations surface as tool-error responses on the tool channel (the same wire shape models are trained on for "tool call failed, try again"); bare-text retry nudges still arrive as user messages.

---

## Multi-Turn Conversations

`WorkflowRunner` uses one canonical `ConversationState` during each run.
`on_message` receives each new `RichMessage` once, including fresh system/user
seeds, recorded assistant replies, execution results and corrections. Supplied
`initial_messages` entries are shallow-list seeded without re-emission or an
extra system/user pair. Include the system prompt and new user turn yourself.

```python
from forge import WorkflowRunner, RichMessage, MessageRole, TextPart

conversation: list[RichMessage] = []

def collect(message: RichMessage) -> None:
    # Warnings are transient events, not part of canonical history.
    if message.forge_purpose != "context_warning":
        conversation.append(message)

runner = WorkflowRunner(client, ctx, on_message=collect)
value = await runner.run(workflow, "first question")

seed = [*conversation,
        RichMessage(MessageRole.USER, [TextPart("follow-up question")])]
# Store the new user turn once; supplied seeds are silent.
conversation = list(seed)
value = await runner.run(workflow, "follow-up question", initial_messages=seed)
```

The system prompt is not rebuilt when seeds are supplied. Step enforcement
and the generated-call counter reset per run, as before. The consumer still
owns server lifecycle and history between runs. Callbacks are event delivery,
so they can include messages later removed by configured compaction.

### Migrating Python callers

| Old usage | New usage |
|---|---|
| `Message(role, content, MessageMeta(...))` | `RichMessage(role, [TextPart(content)])` |
| `message.metadata.step_index` | `message.step_index` |
| `MessageType.REASONING` | Ordered `ReasoningPart` values in an assistant message |
| `message.tool_calls` / `ToolCallInfo` | `ToolCallPart` values in `message.parts` |
| `ToolCall(tool, args)` / `TextResponse(content)` replies | Assistant `RichMessage` with typed parts |
| `tc.tool`, `tc.args` | `tc.name`, `tc.arguments.value` after validation |
| `message.tool_call_id` / result string | `ToolResultPart.call_id`, `.content`, `.outcome` |
| Retry/step/prerequisite classifications | `forge_purpose` values `retry`, `step`, `prerequisite` |
| `run_inference(messages=..., reasoning_replay=...)` | `run_inference(state=..., metadata=RequestMetadata(reasoning_replay=...))` |

Removed imports fail; string entries, dictionaries and malformed seed forms
raise `TypeError`. There is no alternate legacy seed API. Definitions are
`ToolDefinition` objects; `Workflow.handlers` and `Workflow.prerequisites`
remain separate execution data.

```python
from forge import ReasoningPart, ToolCallPart, ToolResultPart

def inspect_message(message: RichMessage) -> None:
    for part in message.parts:
        if isinstance(part, TextPart):
            print("visible", part.text)
        elif isinstance(part, ReasoningPart):
            print("reasoning", part.text)
        elif isinstance(part, ToolCallPart):
            print("call", part.name, part.arguments, part.call_id)
        elif isinstance(part, ToolResultPart):
            print("result", part.call_id, part.outcome,
                  [text.text for text in part.content])
```

One assistant message can contain all visible text, multiple reasoning parts,
and parallel calls. Do not split or reconstruct that reply. The accepted reply's
`step_index` is the completed iteration index, shared with Runner corrections
and execution results; rejected attempts and retry corrections retain their
inference-call grouping. `on_chunk` is a side channel: streamed `FINAL` arrives
before reply callbacks and tool execution, including on exhaustion.

`reasoning_replay` changes only the rendered view: `none` omits reasoning,
`keep-last` includes every reasoning part in the last reasoning-bearing assistant
turn, and `full` includes all reasoning. Canonical history and callbacks retain
it until explicit compaction.

Terminal handlers return their actual Python value after the whole call batch
and error bookkeeping finish. Results in history hold model-visible text only:
successful non-string values use `json.dumps(value)`. Each result uses the shared
inference-generated call ID. Callable errors, resolution errors and workflow
corrections retain their existing prefixes/channels and distinct budgets.

### Persistent sessions

Keep correlated call/result exchanges together when choosing history for later
turns. Corrections have `forge_purpose`; ordinary roles and parts describe the
rest. Context warnings are transient callback events and should not be added to
persistent seeds. Consumer persistence policy remains separate from Runner's
within-turn corrections and configured compaction.

---

## Choosing a Backend

See [BACKEND_SETUP.md](BACKEND_SETUP.md) for the supported-backend table, boot commands, and client snippets. [MODEL_GUIDE.md](MODEL_GUIDE.md) covers which model to pick.

### Sampling Parameters

Each model family has its own recommended temperature / top_p / top_k — and those recommendations differ substantially across families. Running everything at a single default is a measurable handicap for most models. Forge ships a per-model recommendations map that consumers opt into explicitly via a constructor flag:

```python
from forge.clients import LlamafileClient

client = LlamafileClient(
    gguf_path="path/to/Qwen3.5-27B-Q4_K_M.gguf",
    mode="native",
    recommended_sampling=True,
)
```

For local-server backends, the GGUF (or llamafile) path is the canonical model identity — its filename stem (e.g. `Qwen3.5-27B-Q4_K_M`) is what forge uses for sampling-defaults lookup, the wire-format `model` field, and JSONL eval rows. For vLLM the equivalent is `model_path` (a model directory or HF repo id), whose trailing segment serves the same role. Use Ollama-style strings only with `OllamaClient`.

The flag is opt-in. Default behavior (`recommended_sampling=False`) leaves sampling to backend defaults; if forge has opinions about the model, it logs a one-shot INFO message pointing the caller at the flag. With `recommended_sampling=True`, an unknown model raises `UnsupportedModelError`.

#### Proxy mode

The proxy does not consult the recommendations map. For generic OpenAI/llama
and vLLM adapters, the OpenAI-shaped per-call dictionary carries
`temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`,
`presence_penalty`, and `seed`. Ollama translates supported values into its
native options. `AnthropicClient` ignores this OpenAI-shaped dictionary; on a
clean Anthropic-to-Anthropic request, caller-authored Anthropic body fields are
preserved instead. The proxy's pre-built client remains a blank slate.

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:27b-q4_K_M",
    "messages": [{"role": "user", "content": "hi"}],
    "temperature": 1.0,
    "top_p": 0.95,
    "presence_penalty": 1.5
  }'
```

To get card-recommended sampling in proxy mode, the calling client looks up
`forge.clients.get_sampling_defaults(model)` and supplies values supported by
the selected adapter. Forge does not apply the recommendations map itself.

See [MODEL_GUIDE.md#sampling-parameters](MODEL_GUIDE.md#sampling-parameters) for the supported-models table, source citations, and override patterns.

---

## Context Management

Context strategies consume `ConversationState` and return `(compacted_state,
phase_reached)` from `compact(state, budget_tokens, step_hint="")`.
`ContextManager.maybe_compact(state, step_index=0, step_hint="")` returns the
state to apply. `NoCompact` returns the input unchanged. The other strategies
return a compacted copy, preserving the first two messages and recent
`step_index` iteration groups.

Tiered compaction removes eligible correction text/results, truncates each old
result to 200 original text characters across its ordered blocks, then drops
old results and finally old assistant text/reasoning as pressure increases.
Calls and unrelated parts survive category removal. Truncation preserves result
correlation, outcome and retained block fields, and reports removed characters.
Backend usage takes priority; after an actual rewrite the fallback counts only
legacy content characters divided by four: newline-joined visible text per
message, reasoning text, and newline-joined text per result. Arguments, schemas
and extras do not contribute, and reasoning replay does not filter accounting.

Native `WorkflowRunner`/`ContextManager` integrations can automatically manage the context window. This section does not describe Proxy, which is always no-compaction. When a native conversation approaches the budget limit, tiered compaction fires:

- **Phase 1** — Drop old corrections and truncate old results to 200 text characters plus a notice.
- **Phase 2** — Drop old results while preserving assistant text, reasoning and calls.
- **Phase 3** — Also drop old assistant text and reasoning, retaining calls and protected history.

You can configure this via the `ContextManager`:

```python
from forge.context import ContextManager, TieredCompact, NoCompact

# Default: tiered compaction with 2 recent iterations preserved
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

Workflow can declare conditional tool dependencies — "if you call this tool, you must have called tool X first." This is enforced at runtime via nudge-and-retry, the same pattern as step enforcement.

```python
# Workflow owns these mappings separately from its tool definitions:
handlers = {"read_file": read_file, "edit_file": edit_file}

# Name-only: any prior call to read_file satisfies it.
prerequisites = {"edit_file": ["read_file"]}

# Or arg-matched: must have called read_file with the same path.
prerequisites = {"edit_file": [{"tool": "read_file", "match_arg": "path"}]}
# Supply handlers=handlers and prerequisites=prerequisites to Workflow.
```

If the model calls a tool without satisfying its prerequisites, the runner blocks the batch and emits one tool-error response per blocked tool call (`[PrereqError] ...` on the tool channel, with `PREREQUISITE_NUDGE` message type for compaction prioritization). The model retries off the canonical "tool failed" wire shape rather than a trailing user message — friendlier to OpenAI-tool-trained models. After `max_prereq_violations` (default 2) consecutive violations, `PrerequisiteError` is raised.

Prerequisites are not included in the tool schema — the model discovers constraints via the tool-error reply, same as step enforcement.

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
main_client = LlamafileClient(gguf_path="path/to/model.gguf", id_slot=0)
main_runner = WorkflowRunner(client=main_client, context_manager=ctx)

# Slot 1: shared specialist slot (needs a worker)
service_client = LlamafileClient(gguf_path="path/to/model.gguf", id_slot=1)
service_runner = WorkflowRunner(client=service_client, context_manager=ctx)
service_worker = SlotWorker(service_runner)
await service_worker.start()

# Tools route through the worker
async def query_calendar(**kwargs):
    return await service_worker.submit(calendar_wf, kwargs["query"], priority=0)
```
