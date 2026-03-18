# ADR-011: Guardrail Middleware — Composable Reliability Without Loop Ownership

**Status:** Accepted (March 2026) — Phase 1-3 implemented, Phase 4 docs updated

## Problem

Forge's guardrail stack is the core value — it takes an 8B model from 38% to 99%. But the guardrails are embedded inside `WorkflowRunner.run()`, which also owns the orchestration loop. Any system that wants forge's reliability without forge's loop can't consume the guardrails independently.

This creates integration friction with:
- **Eval harnesses** (BFCL) that own the model interaction loop
- **Agent frameworks** (LangChain, CrewAI, AutoGen) that own orchestration
- **Custom loops** where the consumer wants selective guardrails

The guardrail logic is already *mostly* separated at the function/class level (`StepTracker`, `retry_nudge()`, `rescue_tool_call()`, `ContextManager`), but the wiring and decision logic — when to retry, when to nudge, when to escalate — lives in the runner's 240-line loop body.

## Decision

Extract guardrails into `forge.guardrails` as a set of composable objects with **multiple entry points mapping to natural loop phases**. The consumer calls them in sequence inside their own loop. `WorkflowRunner` becomes a batteries-included consumer of the same API.

### Why Multiple Entry Points (Not One)

A single `process()` entry point bundles all guardrails into one call. This forces all-or-nothing adoption and hides what's happening behind one opaque interface that gets called twice per turn (pre-execution, post-execution) doing completely different things each time.

Multiple entry points let consumers:
- **Adopt selectively** — maybe they only want rescue parsing and retry nudges, not step enforcement
- **Understand what each call does** — each method has one job
- **Insert framework-specific logic between phases** — logging, metrics, custom validation

The tradeoff is ordering: the consumer must call phases in sequence. This is acceptable because the phases map to the natural structure of any agent loop (got response → validated it → executed tools → checked results → managed context), so the ordering is intuitive rather than arbitrary.

### Consumer Integration

```python
from forge.guardrails import ResponseValidator, StepEnforcer, ErrorTracker, Nudge
from forge.context import ContextManager, TieredCompact

# Setup (once per session)
validator = ResponseValidator(
    tool_names=["search", "lookup", "answer"],
    rescue_enabled=True,
)
enforcer = StepEnforcer(
    required_steps=["search", "lookup"],
    terminal_tool="answer",
)
errors = ErrorTracker(max_retries=3, max_tool_errors=2)
context = ContextManager(strategy=TieredCompact(keep_recent=2), budget_tokens=8192)

def inject(nudge: Nudge):
    """Convert a forge Nudge to the host framework's message format."""
    messages.append({"role": nudge.role, "content": nudge.content})

# Inside the consumer's loop:
while not done:
    # Phase 0: compact context if approaching budget
    messages = context.maybe_compact(messages)

    # Phase 1: call the model (consumer's responsibility)
    response = their_framework.call_model(messages)

    # Phase 2: validate response → tool calls or retry
    result = validator.validate(response)
    if result.needs_retry:
        errors.record_retry()
        if errors.retries_exhausted:
            raise ...
        inject(result.nudge)
        continue
    tool_calls = result.tool_calls

    # Phase 3: check step requirements before execution
    step_check = enforcer.check(tool_calls)
    if step_check.needs_nudge:
        inject(step_check.nudge)
        continue

    # Phase 4: execute tools (consumer's responsibility)
    for tc in tool_calls:
        result = their_framework.execute(tc)
        enforcer.record(tc.tool)
        errors.record_result(success=result.ok, is_soft_error=result.is_resolution_error)

    # Phase 5: check error budget
    if errors.tool_errors_exhausted:
        raise ...

    # Terminal detection (consumer's responsibility)
    if enforcer.terminal_reached(tool_calls):
        done = True
```

### What Gets Extracted

| Component | Source | Middleware Class | Stateful? |
|---|---|---|---|
| Rescue tool call from free text | `templates.py` `rescue_tool_call()` | `ResponseValidator` | No |
| Retry nudge on TextResponse | `nudges.py` `retry_nudge()` | `ResponseValidator` | No |
| Unknown tool nudge | `nudges.py` `unknown_tool_nudge()` | `ResponseValidator` | No |
| Step tracking | `steps.py` `StepTracker` | `StepEnforcer` | Yes |
| Premature terminal escalation | `runner.py` L241-272 (tiered nudges) | `StepEnforcer` | Yes |
| Consecutive retry counting | `runner.py` `consecutive_retries` | `ErrorTracker` | Yes |
| Consecutive tool error counting | `runner.py` `consecutive_tool_errors` | `ErrorTracker` | Yes |
| Context compaction | `context/manager.py` `ContextManager` | (already standalone) | Yes |

### What Stays in `WorkflowRunner`

| Component | Reason |
|---|---|
| The loop itself | Prescriptive mode — composes guardrails internally |
| Tool execution dispatch | Middleware doesn't execute tools — the host framework does |
| LLM client send/stream | Middleware is client-agnostic |
| Message serialization (api_messages, reasoning folding) | Wire format is a client concern |

### What Stays in `forge.clients`

Unchanged. Middleware users won't need it (they have their own client). Standalone users still use it.

## Component Design

### `Nudge`

Lightweight dataclass returned by all guardrail components when they need a message injected into conversation history.

```python
@dataclass(frozen=True)
class Nudge:
    role: str           # "user", "system", or "tool"
    content: str        # the nudge text
    kind: str           # "retry", "unknown_tool", "step" — identifies the nudge type
    tier: int = 0       # escalation level for step nudges (0 = N/A, 1-3 = escalating)
```

**Why not plain strings.** The consumer needs the role to insert the nudge correctly — injecting a nudge as `"assistant"` would be catastrophic. But without a role, the consumer has to guess.

**Why not forge's `Message` type.** `Message` carries `MessageRole` (enum), `MessageMeta` (with `MessageType`, `step_index`, `original_type`, `token_estimate`), plus optional `tool_calls`, `tool_name`, `tool_call_id`. That's six forge-internal types just to read a nudge. Middleware consumers shouldn't need any of that.

**The `kind` field.** Identifies what generated the nudge (`"retry"`, `"unknown_tool"`, `"step"`). Middleware consumers can use this for logging, metrics, or selective handling. `WorkflowRunner` uses it to map back to the appropriate `MessageType` when constructing internal `Message` objects.

**Role mapping.** All current nudges use role `"user"` — the runner injects them as user messages to prompt the model to try again. This could change in the future (e.g., a system-level nudge), so `role` is a plain string rather than hardcoded.

#### WorkflowRunner impact

In Phase 2, `WorkflowRunner` receives `Nudge` objects from the guardrail components and converts them to internal `Message` objects:

```python
# Before (current runner, inline nudge construction):
_emit(Message(
    MessageRole.USER,
    retry_nudge(response.content),
    MessageMeta(MessageType.RETRY_NUDGE, step_index=iteration),
))

# After (runner consumes guardrail Nudge):
nudge = result.nudge  # Nudge(role="user", content="...", kind="retry")
_emit(Message(
    MessageRole(nudge.role),
    nudge.content,
    MessageMeta(_NUDGE_KIND_TO_TYPE[nudge.kind], step_index=iteration),
))
```

Where `_NUDGE_KIND_TO_TYPE` maps `kind` strings to `MessageType` values:

```python
_NUDGE_KIND_TO_TYPE = {
    "retry": MessageType.RETRY_NUDGE,
    "unknown_tool": MessageType.RETRY_NUDGE,  # currently uses RETRY_NUDGE
    "step": MessageType.STEP_NUDGE,
}
```

This preserves the `MessageType` tagging that `TieredCompact` relies on for compaction prioritization — nudge messages remain expendable during context pressure.

### `ResponseValidator`

Stateless. Takes a raw LLM response and returns either parsed tool calls or a `Nudge`.

```python
@dataclass
class ValidationResult:
    tool_calls: list[ToolCall] | None  # None if needs_retry
    nudge: Nudge | None                # Nudge if needs_retry
    needs_retry: bool

class ResponseValidator:
    def __init__(self, tool_names: list[str], rescue_enabled: bool = True): ...
    def validate(self, response: TextResponse | list[ToolCall]) -> ValidationResult: ...
```

Internally calls `rescue_tool_call()` on TextResponse, checks tool names against `tool_names`, generates appropriate `Nudge`:

- TextResponse with no rescuable tool call → `Nudge(role="user", content=retry_nudge(...), kind="retry")`
- Unknown tool name → `Nudge(role="user", content=unknown_tool_nudge(...), kind="unknown_tool")`

### `StepEnforcer`

Stateful (per-session). Wraps `StepTracker` and adds premature terminal escalation.

```python
@dataclass
class StepCheck:
    nudge: Nudge | None
    needs_nudge: bool

class StepEnforcer:
    def __init__(self, required_steps: list[str], terminal_tool: str): ...
    def check(self, tool_calls: list[ToolCall]) -> StepCheck: ...
    def record(self, tool_name: str) -> None: ...
    def is_satisfied(self) -> bool: ...
    def terminal_reached(self, tool_calls: list[ToolCall]) -> bool: ...
```

Internally uses `StepTracker` for state + `step_nudge()` for message generation. Manages the `premature_terminal_attempts` counter and tier escalation that currently lives in the runner. Returns `Nudge(role="user", content=step_nudge(...), kind="step", tier=N)`.

The `tier` on `StepCheck` was removed — it now lives on the `Nudge` itself, which is the right place since the consumer shouldn't need to know the escalation level separately from the nudge they're injecting.

### `ErrorTracker`

Stateful (per-session). Tracks consecutive retries and consecutive tool errors with their respective limits.

```python
class ErrorTracker:
    def __init__(self, max_retries: int = 3, max_tool_errors: int = 2): ...
    def record_retry(self) -> None: ...
    def reset_retries(self) -> None: ...       # called on successful validation
    def record_result(self, success: bool, is_soft_error: bool = False) -> None: ...
    def reset_errors(self) -> None: ...        # called on fully clean batch

    @property
    def retries_exhausted(self) -> bool: ...
    @property
    def tool_errors_exhausted(self) -> bool: ...
```

The `is_soft_error` flag handles the `ToolResolutionError` case: the host framework executes the tool, catches its own equivalent of a resolution error, and reports it as a soft error. Soft errors don't increment `consecutive_tool_errors`. This pushes classification to the host (Option A from design discussion), which is correct — the middleware is an advisory layer, not an execution engine.

`ErrorTracker` does not produce nudges — it only tracks counts and exposes exhaustion flags. The consumer decides what to do (raise, log, bail out). This is deliberate: error budget exhaustion is a terminal condition, not a "try again with this message" situation.

### `ContextManager`

Already standalone in `forge.context.manager`. The only coupling is that `maybe_compact()` operates on forge `Message` objects. For middleware consumers using their own message format, two options:

1. Consumers convert to/from forge `Message` at the boundary (simple, works today)
2. Future: add a `compact_dicts()` method that works on plain dicts (deferred to when a real consumer needs it)

No changes for v1. `ContextManager` is already importable and usable independently.

## ToolResolutionError in Middleware Mode

Currently, the runner catches `ToolResolutionError` during tool execution to distinguish soft errors from hard crashes. In middleware mode, the host framework executes tools — forge never sees the exception.

**Decision: convention-based reporting.** The host reports soft errors via `ErrorTracker.record_result(success=False, is_soft_error=True)`. The host is responsible for classifying their own errors.

This is the right boundary because:
- The middleware is advisory, not an execution engine
- Different frameworks have different error taxonomies — forge can't catch exceptions it doesn't control
- The host already knows which errors are retryable in their domain

The `ToolResolutionError` exception type remains available for standalone mode (`WorkflowRunner`) and for consumers who use forge tool callables directly.

## Message Format

Nudge messages are returned as `Nudge` dataclasses — a minimal type with `role`, `content`, `kind`, and `tier`. The consumer maps this to their framework's message format:

```python
# Plain OpenAI-style
messages.append({"role": nudge.role, "content": nudge.content})

# LangChain
msg_cls = HumanMessage if nudge.role == "user" else SystemMessage
messages.append(msg_cls(content=nudge.content))
```

`Nudge` lives in `forge.guardrails` and has no dependencies on forge internals (`Message`, `MessageRole`, `MessageMeta`, `MessageType`). It carries just enough information for the consumer to inject correctly and for `WorkflowRunner` to reconstruct full `Message` objects internally.

The `kind` field (`"retry"`, `"unknown_tool"`, `"step"`) serves two purposes:
1. **Middleware consumers** can use it for logging, metrics, or conditional handling (e.g., "log all step nudges but not retries")
2. **WorkflowRunner** maps it back to `MessageType` for compaction prioritization — `TieredCompact` treats nudge messages as expendable under context pressure

## Forge Types in the Middleware API

The middleware API uses two existing forge types: `ToolCall` and `TextResponse` from `forge.core.workflow`. Both are lightweight Pydantic models:

```python
class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any]
    reasoning: str | None = None

class TextResponse(BaseModel):
    content: str
```

These are the right boundary types — they describe what the model said, not how forge processes it. A consumer converting from their framework's representation is one line:

```python
# From OpenAI-style dict
tc = ToolCall(tool=d["function"]["name"], args=json.loads(d["function"]["arguments"]))

# From LangChain ToolMessage
tc = ToolCall(tool=lc_call.name, args=lc_call.args)
```

### Types middleware consumers DON'T need

| Type | Why it stays internal |
|---|---|
| `ToolDef` | Binds schema to callable — middleware doesn't execute tools |
| `ToolSpec` | Pydantic parameter schema for LLM tool descriptions — middleware doesn't talk to the LLM |
| `Workflow` | Full declarative definition — middleware doesn't own orchestration |
| `Message` | Internal message with `MessageMeta`, `ToolCallInfo`, serialization — middleware returns `Nudge` instead |
| `MessageRole`, `MessageType`, `MessageMeta` | Internal metadata for compaction — middleware consumers use `Nudge.role` and `Nudge.kind` |
| `ToolCallInfo` | Wire-format tool call with `call_id` — an internal bookkeeping type |

### No rework needed

`ToolCall` and `TextResponse` don't need changes. They're already minimal, framework-agnostic types. The middleware just uses them as-is. The heavier types (`ToolDef`, `ToolSpec`, `Workflow`, `Message`) stay in their current modules, used only by `WorkflowRunner` and clients.

### Future: framework adapters (v2)

v1 uses forge types (`ToolCall`, `TextResponse`) at both the input and output boundaries of the middleware API. This is a deliberate consistency choice — if we return `ToolCall`, we should accept `ToolCall`. The consumer constructs them from their framework's format (one-line conversions, shown above).

If real consumers later report that even this lightweight conversion is friction, the right solution is per-framework adapters — thin wrappers that handle format conversion at the edges:

```python
# Hypothetical v2
from forge.guardrails.adapters import LangChainAdapter

adapter = LangChainAdapter(validator, enforcer, errors)
result = adapter.validate(langchain_ai_message)  # accepts LangChain types directly
```

Adapters would live in `forge.guardrails.adapters`, one module per framework (langchain, openai, crewai). Each adapter converts the framework's types to/from forge types before delegating to the core middleware classes — no changes to the middleware itself.

This is explicitly deferred until a real consumer asks for it. Building adapters speculatively means tracking framework version changes and message format evolution with no signal on what anyone actually needs. The forge types are the honest, stable boundary for v1.

## Batch Tool Call Handling

The runner currently processes multiple tool calls per LLM response. The middleware API handles this naturally:

- `ResponseValidator.validate()` returns `list[ToolCall]` (preserves batches)
- `StepEnforcer.check()` takes `list[ToolCall]` (checks terminal across the batch)
- `ErrorTracker.record_result()` is called per-tool (host loops over results)

No special batch abstraction needed.

## Implementation Plan

### Phase 1: Additive extraction (no existing code changes)

1. Create `src/forge/guardrails/__init__.py`
2. Implement `ResponseValidator` — wraps `rescue_tool_call()`, `retry_nudge()`, `unknown_tool_nudge()`
3. Implement `StepEnforcer` — wraps `StepTracker` + premature terminal escalation logic
4. Implement `ErrorTracker` — consecutive retry/error counting
5. Unit tests for all three in isolation (no LLM, no backend)
6. Export from `forge.__init__`

### Phase 2: Runner refactor

7. Refactor `WorkflowRunner.run()` to internally compose `ResponseValidator`, `StepEnforcer`, `ErrorTracker`
8. Run full unit test suite (562 tests) — no regressions
9. Run eval on one model config — before/after scores match

### Phase 3: Proof of concept

10. Write a minimal "foreign loop" example using the middleware API
11. (Optional) BFCL adapter as real-world validation

### Phase 4: Documentation

12. Update README (add middleware to feature list)
13. Update USER_GUIDE (middleware usage section)
14. Update ARCHITECTURE.md

## Risk

**Low structural risk.** The guardrail logic is already independently toggleable (the ablation study depends on this). The extraction is primarily moving decision logic across module boundaries, not rewriting it.

**Medium regression risk in Phase 2.** `WorkflowRunner` is the most complex single module. Changing how it calls into guardrail logic could introduce subtle ordering bugs. Mitigation: 562 unit tests + before/after eval run.

**Low API stability risk.** The middleware API starts narrow (three classes) and can expand based on actual integration feedback. No need to over-design before someone tries to use it.

## References

- `WorkflowRunner.run()`: `src/forge/core/runner.py`
- `StepTracker`: `src/forge/core/steps.py`
- Nudge templates: `src/forge/prompts/nudges.py`
- Rescue parsing: `src/forge/prompts/templates.py`
- Context management: `src/forge/context/manager.py`
- Error hierarchy: `src/forge/errors.py`
- Paper: guardrail ablation findings (10-55% accuracy improvement)
- External design conversation: `middleware_refactor.md` (March 2026)
