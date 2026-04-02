# Architecture: Agentic Tool-Calling Library

## Overview

A reusable Python library for self-hosted LLM tool-calling and multi-step agentic workflows. The library owns the tool-calling loop — retry logic, context management, step enforcement, and client adapters. It does not own intent routing, model selection, or domain logic; downstream projects build those on top.

**Tested models:** Ministral 8B/14B (Instruct + Reasoning), Qwen3 8B/14B, Llama 3.1 8B, Mistral 7B v0.3, Mistral Nemo 12B, Claude Haiku/Sonnet/Opus
**Target hardware:** 12–32GB VRAM (consumer GPUs)
**Backends:** Ollama, llama-server (llama.cpp), Llamafile, Anthropic API (baseline)
**License:** MIT

---

## Design Principles

### 1. Fail Fast, Fail Loud

No defensive coding. No silent `try/except` blocks, no fallback defaults, no swallowed errors. If the model returns garbage, the retry loop handles it explicitly. If retries are exhausted, the library raises a typed exception with full context (attempt count, last error, last raw response). Silent failures in agentic loops are devastating — a swallowed error at step 3 corrupts every subsequent step.

```python
# BAD — defensive
try:
    tool_call = parse_tool_call(response)
except Exception:
    tool_call = ToolCall(tool="fallback", args={})  # Silent corruption

# GOOD — fail fast with context
try:
    tool_call = parse_tool_call(response)
except ParseError as e:
    raise ToolCallError(
        f"Failed to parse tool call on attempt {attempt}/{max_retries}",
        raw_response=response,
        cause=e,
    )
```

### 2. Explicit Over Implicit

All data schemas defined with Pydantic models. All LLM outputs validated against schemas before execution. No magic strings — use enums and constants. Configuration is explicit: if the library auto-detects hardware, it logs what it detected and what budget it chose. The user can always override.

**Why this matters for self-hosted:** Cloud APIs absorb ambiguity gracefully. A 14B model at Q4 does not. Every implicit assumption is a failure mode.

### 3. Control Flow Is Not Memory

The library separates *what the model remembers* (message history, subject to compaction) from *what the runner enforces* (step completion, iteration count, terminal conditions). The model's context is a resource to be managed. Control-flow state is authoritative and lives outside the message history. See [P0-1 Decision](#p0-1-compaction--required-steps-interaction).

### 4. The Client Adapter Is the Abstraction Boundary

The library does not know whether the LLM supports native function calling, prompt-injected tool calling, or some future protocol. The client adapter is responsible for translating between the library's internal `ToolCall` representation and whatever the backend expects. The tool-calling loop receives validated `ToolCall` objects and never parses raw text. See [P0-2 Decision](#p0-2-native-function-calling-vs-prompt-injected-tool-calling).

### 5. Context Is a First-Class Resource

On consumer hardware, KV cache competes with model weights for VRAM. A 15-step workflow can easily hit 10–20K tokens, pushing a 14B model at Q4 off GPU and into RAM (5–20x slower). Context management is not optional — it's load-bearing infrastructure. The library budgets context, compacts proactively, and exposes the strategy to downstream consumers.

---

## P0 Design Decisions

### P0-1: Compaction × Required Steps Interaction

**Decision:** Step completion is tracked in a `StepTracker` on the workflow runner, outside the message history.

**Rationale:** The message history is the model's scratchpad — compaction is allowed to reshape it. Step completion is a control-flow fact. The runner checks `completed_steps` authoritatively and blocks premature termination regardless of what the model "remembers." When compaction summarizes or drops a tool result, the summary includes a hint (`[Steps completed: get_public_pricing, get_company_history]`), but enforcement does not depend on the model reading that hint.

**Tradeoff:** The model may redundantly re-call a tool whose result was aggressively compacted. This wastes an iteration but does not corrupt the workflow. Idempotent tools are a downstream responsibility — the library documents this expectation.

### P0-2: Native Function Calling vs. Prompt-Injected Tool Calling

**Decision:** Native FC is the primary path. Prompt-injected tool calling is a supported fallback behind the same `LLMClient` protocol.

**Rationale:** Mistral models with native FC achieve 90–100% reliability; prompt-injected calling is competitive but requires the prompt-injection and JSON extraction machinery. The library defaults to native FC. The prompt-injected path exists because: (1) some models don't expose native FC through all backends, and (2) it enables llama-server without `--jinja` support and older Llamafile builds.

The abstraction boundary is the client adapter. A `LLMClient` returns `list[ToolCall] | TextResponse` — never raw text. The native FC adapter parses structured API responses. The prompt-injected adapter does JSON extraction and retry internally. The tool-calling loop is path-agnostic.

**Tradeoff:** Two code paths to test and maintain. Mitigated by keeping the prompt-injected path minimal (fallback, not first-class) and documenting which models use which path.

### P0-3: Message Metadata (`_meta` Tagging)

**Decision:** Internal `Message` dataclass with typed metadata. Serialize to plain dicts at the API boundary only.

**Rationale:** The concept doc's approach of stuffing `_meta` into message dicts alongside `role`/`content` works internally but breaks on APIs that validate schema strictly. The library defines a `Message` model with `role`, `content`, and `metadata: MessageMeta`. All internal code works with `Message` objects. A `serialize_for_api()` function strips metadata at the call site, producing clean `{"role": ..., "content": ...}` dicts.

Downstream consumers who inspect message history (logging, debugging, eval) get the rich `Message` objects with compaction type, step index, timestamps, etc.

**Tradeoff:** Every message goes through a wrapper instead of being a raw dict. Thin allocation cost. Consumers can't pass raw dicts without wrapping. Worth it — the `_meta`-in-dict approach is a silent API-compatibility bomb.

---

## Component Boundaries and Interfaces

### Dependency Graph

```
┌──────────────────────────────────────────────────────────┐
│                    Downstream Project                      │
│    (defines tools, prompts, workflows, routing)           │
└──────────────────┬─────────────────────────────────────┘
                   │ uses
┌──────────────────▼─────────────────────────────────────┐
│                      forge library                       │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ WorkflowRun- │  │   Context    │  │   Server     │  │
│  │ ner (loop)   │──│   Manager    │  │   Manager    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │           │
│         │          ┌──────▼───────┐  ┌──────▼───────┐  │
│         │          │  Compact     │  │  Hardware    │  │
│         │          │  Strategy    │  │  Profile     │  │
│         │          └──────────────┘  └──────────────┘  │
│  ┌──────▼──────────────────────────┐                    │
│  │  Guardrails (middleware)        │  ┌──────────────┐  │
│  │  ┌──────────────────────────┐   │  │   Message    │  │
│  │  │ ResponseValidator        │   │  │   Types      │  │
│  │  │ (rescue, retry, unknown) │   │  └──────────────┘  │
│  │  ├──────────────────────────┤   │                    │
│  │  │ StepEnforcer             │   │                    │
│  │  │ (wraps StepTracker,      │   │                    │
│  │  │  premature escalation)   │   │                    │
│  │  ├──────────────────────────┤   │                    │
│  │  │ ErrorTracker             │   │                    │
│  │  │ (retry + tool budgets)   │   │                    │
│  │  └──────────────────────────┘   │                    │
│  └─────────────┬───────────────────┘                    │
│                │                                         │
│  ┌─────────────▼────────────────────────────────────┐   │
│  │              LLMClient (Protocol)                │   │
│  │  ┌─────────────┐ ┌──────────────┐ ┌───────────┐ │   │
│  │  │OllamaClient │ │LlamafileClnt │ │Anthropic- │ │   │
│  │  │(native FC)  │ │(native/prompt)│ │Client     │ │   │
│  │  └─────────────┘ └──────────────┘ └───────────┘ │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Core Protocols and Types

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from pydantic import BaseModel


# ── Message Types ──────────────────────────────────────────────


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageType(str, Enum):
    """Metadata tag for compaction prioritization."""
    SYSTEM_PROMPT = "system_prompt"
    USER_INPUT = "user_input"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"       # Chain-of-thought from Reasoning variants
    TEXT_RESPONSE = "text_response"  # Failed tool call attempt (free text)
    STEP_NUDGE = "step_nudge"     # Runner-injected "you must call X first"
    RETRY_NUDGE = "retry_nudge"   # Runner-injected "invalid format, try again"
    SUMMARY = "summary"           # Compacted content


@dataclass(frozen=True)
class MessageMeta:
    """Metadata attached to a message. Never sent to the API."""
    type: MessageType
    step_index: int | None = None
    original_type: MessageType | None = None  # If this is a summary, what it replaced
    token_estimate: int | None = None


@dataclass(frozen=True)
class ToolCallInfo:
    """One tool call within an assistant message."""
    name: str
    args: dict[str, Any]
    call_id: str


@dataclass
class Message:
    role: MessageRole
    content: str                           # Human-readable text (used by logging, eval, compaction)
    metadata: MessageMeta
    tool_name: str | None = None           # For role="tool" result messages
    tool_call_id: str | None = None        # For OpenAI-format tool call/result correlation
    tool_calls: list[ToolCallInfo] | None = None  # For assistant tool-call messages (1+ entries)

    def to_api_dict(self, format: str = "ollama") -> dict[str, Any]:
        """Serialize for LLM API. Strips metadata.

        format="ollama": arguments as dict, no "type" field in tool_calls,
            tool results use "tool_name".
        format="openai": arguments as JSON string, "type": "function" and
            "id" required on tool_calls, tool results use "tool_call_id"
            and "name".
        """


# ── Tool Types ─────────────────────────────────────────────────


class ToolSpec(BaseModel):
    """Declarative tool schema — what the LLM sees.

    parameters is a Pydantic BaseModel subclass (type[BaseModel]).
    ToolSpec.from_json_schema() builds one dynamically from a raw JSON
    Schema dict via pydantic.create_model(). get_json_schema() calls
    parameters.model_json_schema() for the wire format.
    """
    name: str
    description: str
    parameters: type[BaseModel]


@dataclass
class ToolDef:
    """Binds a tool schema to its implementation. Single source of truth.

    Downstream projects define tools as ToolDefs. The Workflow holds these
    in a dict keyed by name and derives the spec list (for the LLM) and
    callable lookup (for execution) internally.

    Prerequisites express conditional dependencies: "if you call this tool,
    you must have called tool X first." Enforced via nudge-and-retry at
    the runner level.
    """
    spec: ToolSpec
    callable: Callable[..., Any]       # sync or async
    prerequisites: list[str | dict[str, str]]  # name-only or arg-matched

    @property
    def name(self) -> str:
        return self.spec.name


class ToolCall(BaseModel):
    """Validated tool invocation returned by an LLMClient."""
    tool: str
    args: dict[str, Any]
    reasoning: str | None = None  # Model's chain-of-thought alongside tool call


class TextResponse(BaseModel):
    """Non-tool-call response from the model (reasoning trace, refusal, etc.)."""
    content: str


# Type alias for what the client returns
type LLMResponse = list[ToolCall] | TextResponse


# ── Streaming Types ────────────────────────────────────────────


class ChunkType(str, Enum):
    """What kind of partial data a stream chunk carries."""
    TEXT_DELTA = "text_delta"          # Partial text (reasoning trace, refusal, etc.)
    TOOL_CALL_DELTA = "tool_call_delta"  # Partial tool call (name or args building up)
    FINAL = "final"                    # Stream complete — carries the resolved LLMResponse
    RETRY = "retry"                    # Previous stream was malformed; client is retrying


@dataclass(frozen=True)
class StreamChunk:
    """A single chunk from a streaming LLM response.

    Consumers (UI, logging) process TEXT_DELTA and TOOL_CALL_DELTA as they arrive.
    The runner ignores all chunks except FINAL, which carries the resolved response.
    On RETRY, consumers should discard the partial output from the failed attempt.
    """
    type: ChunkType
    content: str = ""                  # Partial text for deltas, empty for FINAL/RETRY
    response: LLMResponse | None = None  # Only set when type == FINAL


# ── LLM Client Protocol ───────────────────────────────────────


@runtime_checkable
class LLMClient(Protocol):
    """Interface that client adapters implement.

    The client is responsible for:
    1. Sending messages to the LLM backend
    2. Parsing the response into ToolCall or TextResponse
    3. Handling native FC or prompt-injected calling internally
    4. Optionally streaming partial responses via send_stream()

    The client does NOT retry. Retry logic lives in the WorkflowRunner,
    which has the context to decide whether a response is retryable and
    to track attempt counts for eval metrics.
    """

    api_format: str
    """Wire format for Message.to_api_dict(): 'ollama' or 'openai'."""

    async def send(
        self,
        messages: list[dict[str, Any]],    # Already serialized via Message.to_api_dict()
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Send messages and return a parsed response.

        Returns list[ToolCall] if the model produced one or more valid tool
        invocations. Returns TextResponse if the model produced text
        (reasoning, refusal, or malformed output that couldn't be parsed
        as a tool call).

        The runner inspects the response and decides whether to retry.
        """
        ...

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Send messages and yield streaming chunks.

        Yields TEXT_DELTA or TOOL_CALL_DELTA chunks as they arrive.
        The final chunk has type FINAL and carries the resolved LLMResponse
        (same list[ToolCall] | TextResponse as send() would return).

        The runner forwards chunks to its on_chunk callback for UI/logging,
        then inspects the FINAL chunk and decides whether to retry.
        """
        ...

    async def get_context_length(self) -> int | None:
        """Query the backend for its configured context window size.

        Returns the context length in tokens, or None if the backend
        does not report it. Used by budget resolution to determine the
        hard ceiling on context size.
        """
        ...


# ── Compaction Types ───────────────────────────────────────────


@dataclass(frozen=True)
class CompactEvent:
    """Emitted by ContextManager when compaction fires.

    Provides enough information for logging, debugging, and UI display.
    """
    step_index: int                    # Which loop iteration triggered compaction
    tokens_before: int                 # Estimated tokens before compaction
    tokens_after: int                  # Estimated tokens after compaction
    budget_tokens: int                 # The context budget
    messages_before: int               # Message count before
    messages_after: int                # Message count after
    phase_reached: int                 # Compaction phase reached (0 = none, 1+ = strategy-defined)


# ── Compaction Strategy ────────────────────────────────────────


class CompactStrategy(ABC):
    """Interface for context compaction strategies.

    Implementations decide when and how to compress the message history.
    The ContextManager calls these methods; strategies never mutate
    messages in place.
    """

    @abstractmethod
    def compact(self, messages: list[Message], trigger_tokens: int, *, step_hint: str = "") -> tuple[list[Message], int]:
        """Return a compacted copy of the message history and the phase reached.

        Returns a tuple of (compacted_messages, phase_reached). The phase int
        indicates how aggressively the strategy compacted: 0 means no
        compaction was applied, 1+ is implementation-defined. Strategies
        without internal phases should return 1.

        trigger_tokens is the threshold that triggered compaction. For tiered
        strategies, each phase applies its structural changes, then checks
        whether the result is under trigger_tokens before escalating to the
        next phase. This is NOT a target to compact down to — phases remove
        what they remove structurally.

        Must preserve (never cut):
        - The system prompt (messages[0])
        - The original user input (messages[1])

        Recommended compaction priority (cut first → preserve longest):
        1. step_nudge, retry_nudge — ephemeral corrections, no long-term value
        2. tool_result — truncate to first ~200 chars; raw data is expendable once processed
        3. text_response — failed tool call attempt, expendable after retry corrects model
        4. reasoning — preserve as long as possible; this is the model's interpretive context
        5. tool_call — preserved in all phases (no transformation)
        6. Recent iterations (within keep_recent window) — fully intact
        """
        ...


# ── Context Manager ────────────────────────────────────────────


class ContextManager:
    """Manages context window budget and triggers compaction.

    Does not own the strategy — receives it via constructor or
    factory method.
    """

    def __init__(
        self,
        strategy: CompactStrategy,
        budget_tokens: int,
        compact_threshold: float = 0.75,
        on_compact: Callable[[CompactEvent], None] | None = None,
    ):
        """
        Args:
            on_compact: Callback invoked when compaction fires. Receives a
                CompactEvent with before/after token counts, phase reached,
                and which messages were affected. Use for logging, debugging,
                or surfacing compaction to a UI.
        """
        ...

    def maybe_compact(
        self,
        messages: list[Message],
        step_index: int = 0,
        step_hint: str = "",
    ) -> list[Message]:
        """If estimated tokens exceed budget * compact_threshold, trigger compaction.

        The threshold serves as both the trigger point and the phase escalation
        check — each compaction phase applies its structural changes, then checks
        whether the result is under the threshold before escalating further.

        Args:
            step_index: Which loop iteration triggered compaction. Forwarded to
                CompactEvent for logging/debugging.
            step_hint: Output of StepTracker.summary_hint(). Forwarded to the
                strategy — TieredCompact Phase 3 uses it as checkpoint content.

        Returns the original list unchanged if under threshold.
        Invokes on_compact callback if compaction occurred.
        """
        ...

    def estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate token count. Uses char/4 heuristic by default."""
        ...


# ── Hardware Profile ───────────────────────────────────────────


@dataclass
class HardwareProfile:
    """Detected GPU capabilities (total VRAM only — a stable value)."""
    gpu_name: str
    vram_total_mb: int

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_mb / 1024


def detect_hardware() -> HardwareProfile | None:
    """Auto-detect GPU via nvidia-smi. Returns None if detection fails.

    Reads total VRAM (a stable number that doesn't change with
    allocations). Used by ServerManager for VRAM tier lookup.
    """
    ...


# ── Step Tracker ───────────────────────────────────────────────


@dataclass
class StepTracker:
    """Tracks which required steps have been completed.

    Lives on the WorkflowRunner, outside the message history.
    Compaction cannot invalidate step completion. See P0-1.
    """
    required_steps: list[str]
    completed_steps: dict[str, None] = field(default_factory=dict)

    def record(self, tool_name: str) -> None:
        """Record a tool call as completed."""
        self.completed_steps[tool_name] = None

    def is_satisfied(self) -> bool:
        """True if all required steps have been called."""
        return all(s in self.completed_steps for s in self.required_steps)

    def pending(self) -> list[str]:
        """Return required steps not yet completed, preserving original order."""
        return [s for s in self.required_steps if s not in self.completed_steps]

    def summary_hint(self) -> str:
        """Human-readable hint for injection into compacted summaries."""
        if not self.completed_steps:
            return "[No steps completed yet]"
        return f"[Steps completed: {', '.join(self.completed_steps)}]"


# ── Guardrail Middleware ──────────────────────────────────────
#
# Composable middleware extracted from WorkflowRunner internals.
# The runner instantiates these per-run; they are also importable
# for use in foreign orchestration loops (see examples/foreign_loop.py).


@dataclass
class Nudge:
    """A corrective message to inject into the conversation."""
    role: str           # Always "user"
    content: str        # The nudge text
    kind: str           # "retry", "unknown_tool", or "step"
    tier: int = 0       # Escalation tier (step nudges: 1/2/3)


@dataclass
class ValidationResult:
    """Result of validating an LLM response.

    Exactly one of ``tool_calls`` or ``nudge`` is set:
    - If ``needs_retry`` is False, ``tool_calls`` contains the validated calls.
    - If ``needs_retry`` is True, ``nudge`` contains the message to inject.
    """
    tool_calls: list[ToolCall] | None
    nudge: Nudge | None
    needs_retry: bool


class ResponseValidator:
    """Validates LLM responses: rescues tool calls from text, checks tool names.

    Stateless — safe to reuse across turns and sessions.

    Args:
        tool_names: Valid tool names for this workflow.
        rescue_enabled: If True, attempt to parse tool calls from TextResponse
            before generating a retry nudge.
    """

    def __init__(self, tool_names: list[str], rescue_enabled: bool = True) -> None: ...

    def validate(self, response: LLMResponse) -> ValidationResult:
        """Validate an LLM response.

        Returns ValidationResult with tool_calls on success, or a Nudge on failure.
        TextResponse path: try rescue_tool_call(), then retry nudge.
        list[ToolCall] path: check for unknown tool names → unknown_tool_nudge.
        """
        ...


@dataclass
class StepCheck:
    """Result of checking tool calls against step requirements."""
    nudge: Nudge | None
    needs_nudge: bool


class StepEnforcer:
    """Tracks required steps, enforces them with escalating nudges,
    and enforces tool prerequisites.

    Wraps StepTracker internally. Stateful — instantiate per session/task.

    Args:
        required_steps: Tool names that must be called before the terminal tool.
        terminal_tools: The tools that can end the workflow (frozenset).
        tool_prerequisites: Map of tool name to its ToolDef.prerequisites list.
        max_premature_attempts: How many premature terminal attempts before
            the enforcer signals exhaustion.
        max_prereq_violations: How many consecutive prerequisite violations
            before the enforcer signals exhaustion.
    """

    def __init__(self, required_steps: list[str], terminal_tools: frozenset[str],
                 tool_prerequisites: dict | None = None,
                 max_premature_attempts: int = 3,
                 max_prereq_violations: int = 2) -> None: ...

    def check(self, tool_calls: list[ToolCall]) -> StepCheck:
        """Check whether tool calls include a premature terminal call.

        Returns StepCheck with escalating nudge (tier 1/2/3) if premature.
        """
        ...

    def check_prerequisites(self, tool_calls: list[ToolCall]) -> StepCheck:
        """Check whether any tool call has unsatisfied prerequisites.

        Evaluates against pre-batch state. Any violation blocks the entire batch.
        """
        ...

    def record(self, tool_name: str, args: dict | None = None) -> None: ...
    def is_satisfied(self) -> bool: ...
    def pending(self) -> list[str]: ...
    def reset_premature(self) -> None: ...
    def reset_prereq_violations(self) -> None: ...
    def summary_hint(self) -> str: ...

    @property
    def premature_attempts(self) -> int: ...
    @property
    def premature_exhausted(self) -> bool: ...
    @property
    def prereq_violations(self) -> int: ...
    @property
    def prereq_exhausted(self) -> bool: ...
    @property
    def completed_steps(self) -> dict[str, None]: ...


class ErrorTracker:
    """Tracks consecutive retry and tool error counts against limits.

    Stateful — instantiate per session/task.

    Args:
        max_retries: Consecutive formatting/validation failures before exhaustion.
        max_tool_errors: Consecutive tool execution errors before exhaustion.
    """

    def __init__(self, max_retries: int = 3, max_tool_errors: int = 2) -> None: ...

    def record_retry(self) -> None: ...
    def reset_retries(self) -> None: ...
    def record_result(self, success: bool, is_soft_error: bool = False) -> None: ...
    def reset_errors(self) -> None: ...

    @property
    def retries_exhausted(self) -> bool: ...
    @property
    def tool_errors_exhausted(self) -> bool: ...


# ── Workflow Definition ────────────────────────────────────────


@dataclass
class Workflow:
    """Declarative workflow definition. Provided by downstream projects.

    The Workflow holds ToolDefs in an ordered dict keyed by tool name.
    Keys must match ToolDef.spec.name — validated at construction time.
    It does NOT contain execution logic — that's the WorkflowRunner's job.
    """
    name: str
    description: str
    tools: dict[str, ToolDef]            # Keyed by tool name, validated for consistency
    required_steps: list[str]            # Tools that must be called before terminal
    terminal_tool: str | list[str]       # Tool(s) that can end the workflow
    system_prompt_template: str          # May contain {placeholders}
    terminal_tools: frozenset[str]       # Normalized from terminal_tool (init=False)

    def __post_init__(self) -> None:
        """Normalize terminal_tool to frozenset. Validate tool key/name
        consistency, required_steps, terminal_tools, and prerequisites."""
        # Normalize terminal_tool → terminal_tools (frozenset)
        if isinstance(self.terminal_tool, str):
            self.terminal_tools = frozenset([self.terminal_tool])
        else:
            self.terminal_tools = frozenset(self.terminal_tool)

        for key, tool_def in self.tools.items():
            if key != tool_def.name:
                raise ValueError(...)
        tool_names = set(self.tools.keys())
        for step in self.required_steps:
            if step not in tool_names:
                raise ValueError(...)
        for tt in self.terminal_tools:
            if tt not in tool_names:
                raise ValueError(...)
            if tt in self.required_steps:
                raise ValueError(...)
        for key, tool_def in self.tools.items():
            for prereq in tool_def.prerequisites:
                prereq_name = prereq if isinstance(prereq, str) else prereq["tool"]
                if prereq_name not in tool_names:
                    raise ValueError(...)

    def build_system_prompt(self, **kwargs: str) -> str:
        """Render the system prompt with user-provided values."""
        ...

    def get_tool_specs(self) -> list[ToolSpec]:
        """Return all tool specs for passing to the LLM client."""
        ...

    def get_callable(self, tool_name: str) -> Callable[..., Any]:
        """Return the callable for a tool by name. Raises KeyError if not found."""
        ...


# ── Workflow Runner ────────────────────────────────────────────


class WorkflowRunner:
    """Executes a Workflow against an LLMClient with context management.

    This is the core agentic loop. It:
    1. Builds the initial message list (system prompt + user input)
    2. Initializes guardrail middleware (ResponseValidator, StepEnforcer, ErrorTracker)
    3. Delegates inference to run_inference() — the shared "front half" that
       handles compaction, reasoning folding, serialization, sending, validation,
       and retry. Both the runner and the proxy consume this function.
    4. Delegates step enforcement to StepEnforcer (premature terminal escalation)
    5. Delegates error budgets to ErrorTracker (consecutive retries and tool errors)
    6. Executes tool calls and manages conversation history
    7. Terminates on terminal tool or max iterations

    The runner composes middleware internally — it instantiates
    ResponseValidator, StepEnforcer, and ErrorTracker per run(). These same
    components are importable for use in foreign orchestration loops (see
    examples/foreign_loop.py and ADR-011).

    Every LLM call — whether it returns a list[ToolCall] or TextResponse —
    consumes one iteration (retries within run_inference also consume
    iterations). When a model returns multiple tool calls in one response,
    the runner executes all of them in the same iteration, emitting one
    TOOL_CALL message (with N ToolCallInfo entries) and N TOOL_RESULT
    messages.
    """

    def __init__(
        self,
        client: LLMClient,
        context_manager: ContextManager,
        max_iterations: int = 10,
        max_retries_per_step: int = 3,
        max_tool_errors: int = 2,
        stream: bool = False,
        on_chunk: Callable[[StreamChunk], Awaitable[None]] | None = None,
        on_message: Callable[[Message], None] | None = None,
        rescue_enabled: bool = True,
    ):
        """
        Args:
            max_iterations: Hard ceiling on total LLM round trips. Each
                iteration = one LLM call, whether it returns a ToolCall or
                TextResponse. Retries consume iterations.
            max_retries_per_step: Maximum consecutive formatting failures
                (TextResponse or unknown tool name) before raising ToolCallError.
                Resets on any valid ToolCall (known tool name).
            max_tool_errors: Maximum consecutive tool execution errors before
                raising ToolExecutionError. When a tool callable raises, the
                error is fed back to the model as a tool result so it can
                self-correct. Resets on any successful tool execution.
            stream: If True, uses send_stream() path on the client. The runner
                still waits for the FINAL chunk before acting — streaming is a
                side channel, not a control-flow change.
            on_chunk: Async callback awaited for each StreamChunk during
                streaming. Receives TEXT_DELTA, TOOL_CALL_DELTA, and FINAL
                chunks. Typical use: pipe to a terminal UI or SSE endpoint.
                Ignored if stream=False.
            on_message: Callback invoked each time a Message is appended to
                the conversation history. Receives the Message object with
                full metadata (type, step_index, etc.). Use for observability,
                eval metrics, or logging. Does not affect runner behavior.
                Defaults to None (no overhead when unused).
            rescue_enabled: If False, skip rescue_tool_call() — TextResponse
                goes straight to retry nudge (or failure if retries=0).
                Used by ablation configs to measure guardrail impact.
        """
        ...

    async def run(
        self,
        workflow: Workflow,
        user_message: str,
        prompt_vars: dict[str, str] | None = None,
        initial_messages: list[Message] | None = None,
    ) -> Any:
        """Execute the workflow and return the terminal tool's result.

        Args:
            workflow: The workflow to execute.
            user_message: The user's input message.
            prompt_vars: Variables for the system prompt template.
            initial_messages: If provided, seeds the conversation with these
                messages instead of building a fresh system prompt + user
                input. The on_message callback fires only for NEW messages
                created during this run, not the replayed history. The caller
                must include the system prompt and new user message in the
                seed. See README § on_message Callback and Multi-Turn
                Conversations.

        Raises:
            MaxIterationsError: If max_iterations exceeded without terminal tool.
            ToolCallError: If max_retries_per_step exhausted on consecutive
                formatting failures (TextResponse or unknown tool name).
            ToolExecutionError: If a tool callable raised and the model failed
                to self-correct after max_tool_errors consecutive attempts.
            StreamError: If streaming mode and FINAL chunk is missing.
        """
        ...


# ── Exceptions ─────────────────────────────────────────────────


class ForgeError(Exception):
    """Base exception for the library."""
    pass

class ToolCallError(ForgeError):
    """LLM failed to produce a valid tool call after retries."""
    def __init__(self, message: str, raw_response: str | None = None, cause: Exception | None = None):
        ...

class ToolExecutionError(ForgeError):
    """A tool callable raised and the model failed to self-correct after max_tool_errors."""
    def __init__(self, tool_name: str, cause: Exception):
        ...

class ToolResolutionError(Exception):
    """Tool arguments were valid but the data didn't resolve.

    Not a ForgeError — this is a tool-author exception. The runner catches
    it explicitly: feeds the message back to the model without counting
    toward consecutive_tool_errors and without marking the step as completed.
    """
    def __init__(self, message: str, tool_name: str | None = None):
        ...

class MaxIterationsError(ForgeError):
    """Workflow exceeded max_iterations without calling the terminal tool."""
    def __init__(self, iterations: int, completed_steps: dict[str, None], pending_steps: list[str]):
        ...

class StepEnforcementError(ForgeError):
    """Model repeatedly tried to skip required steps (3 escalating nudges exhausted)."""
    def __init__(self, terminal_tool: str, attempts: int, pending_steps: list[str]):
        ...

class PrerequisiteError(ForgeError):
    """Model repeatedly called a tool without satisfying its prerequisites."""
    def __init__(self, tool_name: str, violations: int, missing_prereqs: list[str]):
        ...

class WorkflowCancelledError(ForgeError):
    """Workflow was cancelled via cancel_event before completion."""
    def __init__(self, messages: list, completed_steps: dict[str, None], iteration: int):
        ...

class ContextBudgetExceeded(ForgeError):
    """Context exceeded budget even after compaction. Unrecoverable."""
    def __init__(self, estimated_tokens: int, budget_tokens: int):
        ...

class BudgetResolutionError(ForgeError):
    """No context budget could be determined from any source."""
    def __init__(self, cause: Exception | None = None):
        ...

class HardwareDetectionError(ForgeError):
    """nvidia-smi responded but output couldn't be parsed."""
    def __init__(self, cause: Exception):
        ...

class ContextDiscoveryError(ForgeError):
    """Backend context length response couldn't be parsed."""
    def __init__(self, cause: Exception):
        ...

class BackendError(ForgeError):
    """Unexpected HTTP error from the LLM backend."""
    def __init__(self, status_code: int, body: str):
        ...

class ThinkingNotSupportedError(BackendError):
    """Model does not support thinking mode, but think=True was explicitly requested."""
    def __init__(self, model: str, status_code: int = 400, body: str = ""):
        ...

class StreamError(ForgeError):
    """Stream ended without producing a FINAL chunk."""
    def __init__(self, message: str = "Stream ended without FINAL chunk"):
        ...
```

---

## Data Flow

### Full Workflow Execution

```
Downstream Project
│
│  workflow = Workflow(tools=..., required_steps=..., terminal_tool=...)  # str or list[str]
│  runner = WorkflowRunner(client=ollama_client, context_manager=ctx_mgr)
│  result = await runner.run(workflow, "Generate a quote for part X", cancel_event=event)
│
▼
WorkflowRunner.run()
│
├─ 1. Build initial messages
│     [Message(SYSTEM, rendered_prompt, meta=SYSTEM_PROMPT)]
│     [Message(USER, user_input, meta=USER_INPUT)]
│
├─ 2. Initialize guardrail middleware
│     validator = ResponseValidator(tool_names, rescue_enabled)
│     step_enforcer = StepEnforcer(required_steps, terminal_tools, tool_prerequisites)
│     error_tracker = ErrorTracker(max_retries, max_tool_errors)
│
└─ 3. Loop (up to max_iterations):
      │
      ├─ 3.0. Cancel check: if cancel_event is set → raise WorkflowCancelledError
      │
      ├─ 3a. ContextManager.maybe_compact(messages)
      │       │
      │       ├─ Under budget? → pass through
      │       └─ Over threshold? → CompactStrategy.compact()
      │             │
      │             ├─ Phase 1: Drop nudges, truncate old tool_results to ~200 chars
      │             ├─ Phase 2: Drop old tool_results entirely (reasoning preserved)
      │             └─ Phase 3: Drop reasoning + text_response (tool_call skeleton only)
      │
      ├─ 3b. Serialize and send (one LLM call per iteration):
      │       │
      │       ├─ stream=False: LLMClient.send(serialized, tools=...)
      │       └─ stream=True:  LLMClient.send_stream(serialized, tools=...)
      │                        → forward chunks to on_chunk callback
      │                        → collect FINAL chunk (raise StreamError if missing)
      │
      ├─ 3c. ResponseValidator.validate(response)
      │       │
      │       ├─ TextResponse → try rescue_tool_call() (skip if rescue_enabled=False)
      │       │    ├─ Rescued? → ValidationResult(tool_calls=rescued, needs_retry=False)
      │       │    └─ Not rescued → ValidationResult(nudge=retry_nudge, needs_retry=True)
      │       ├─ list[ToolCall] with unknown tool → ValidationResult(nudge=unknown_tool_nudge, needs_retry=True)
      │       └─ list[ToolCall] all known → ValidationResult(tool_calls=..., needs_retry=False)
      │
      │       If needs_retry:
      │       ├─ error_tracker.record_retry()
      │       ├─ error_tracker.retries_exhausted → raise ToolCallError
      │       └─ Emit assistant content + nudge message, continue to next iteration
      │
      ├─ 3d. Valid tool calls — error_tracker.reset_retries()
      │
      ├─ 3e. StepEnforcer.check(tool_calls)
      │       │
      │       ├─ Terminal present + steps NOT satisfied → StepCheck(nudge=step_nudge, needs_nudge=True)
      │       │    ├─ step_enforcer.premature_exhausted → raise StepEnforcementError
      │       │    └─ Emit tool call + escalating step nudge (tier 1/2/3), continue
      │       └─ No premature terminal → StepCheck(needs_nudge=False), continue to 3e.2
      │
      ├─ 3e.2. StepEnforcer.check_prerequisites(tool_calls)
      │       │
      │       ├─ Any prereq unsatisfied → StepCheck(nudge=prerequisite_nudge, needs_nudge=True)
      │       │    ├─ step_enforcer.prereq_exhausted → raise PrerequisiteError
      │       │    └─ Emit tool call + prerequisite nudge, continue
      │       └─ All prereqs satisfied → continue to 3f
      │
      ├─ 3f. Execute ALL tool calls in the batch sequentially
      │       │
      │       Emit one TOOL_CALL message with N ToolCallInfo entries, then
      │       execute each tool and emit N individual TOOL_RESULT messages.
      │       │
      │       ├─ ToolResolutionError → feed back to model, don't count error,
      │       │    don't record step
      │       ├─ Other exception → feed back as [ToolError], track last_error
      │       │    batch_had_error flag set, continue to next tool in batch
      │       ├─ Success → step_enforcer.record(tool, args), append result as TOOL_RESULT
      │       └─ Terminal tool in batch + succeeded → stash result for return
      │
      ├─ 3g. Post-batch bookkeeping:
      │       │
      │       ├─ batch_had_error → error_tracker.record_result(success=False)
      │       │    error_tracker.tool_errors_exhausted → raise ToolExecutionError
      │       ├─ No errors → error_tracker.reset_errors(), step_enforcer.reset_premature(),
      │       │    step_enforcer.reset_prereq_violations()
      │       └─ Terminal tool succeeded → return terminal result
      │       │
      │       Messages appended per iteration:
      │       [Message(ASSISTANT, reasoning, meta=REASONING, step=N)]  ← only if reasoning present
      │       [Message(ASSISTANT, "", meta=TOOL_CALL, step=N, tool_calls=[...])]
      │       [Message(TOOL, result_1, meta=TOOL_RESULT, step=N)]
      │       [Message(TOOL, result_2, meta=TOOL_RESULT, step=N)]  ← one per tool call
      │       ...
      │       REASONING messages are folded into the following TOOL_CALL message's
      │       content field on the wire, keeping the internal list separate for compaction.
      │
      └─ 3h. Continue loop
```

### Message Lifecycle

```
                          Internal (Message objects)
                         ┌─────────────────────────────┐
Created by runner ──────▶│ role: "assistant"            │
                         │ content: ""                  │
                         │ metadata:                    │
                         │   type: TOOL_CALL            │
                         │   step_index: 3              │
                         │ tool_calls:                  │
                         │   [ToolCallInfo(name, args,  │
                         │    call_id), ...]            │
                         └────────────┬────────────────┘
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                     ▼
           Sent to LLM API                     Kept for logging/eval
        ┌──────────────────┐              ┌──────────────────────┐
        │ {"role": "asst", │              │ Full Message object  │
        │  "tool_calls":   │              │ with all metadata    │
        │   [{...}, ...]}  │              │ and structured data  │
        └──────────────────┘              └──────────────────────┘
           (metadata stripped              (available to hooks,
            by to_api_dict();              debuggers, eval harness)
            tool_calls list may
            contain 1+ entries)
```

---

## Compaction Strategies (Built-In)

The library ships three `CompactStrategy` implementations. Downstream projects can provide custom strategies via the same interface.

### NoCompact

Passthrough. Preserves the entire message history. Use when VRAM is abundant (32GB+) or workflows are short.

### SlidingWindowCompact

Keeps the system prompt, original user input, and the last N iterations. Uses `step_index` to identify iteration boundaries (handles variable-size parallel tool batches). Simple and predictable. Good baseline for testing.

### TieredCompact (Default)

Three-phase compaction with an explicit priority order. Each phase fires only if the previous phase didn't reduce tokens below the trigger threshold. `keep_recent` controls how many recent loop iterations (each iteration = one assistant message + N tool result messages) are fully preserved before older content is eligible for compaction.

```python
class TieredCompact(CompactStrategy):
    TRUNCATE_CHARS = 200

    def __init__(self, keep_recent: int = 2):
        """
        Args:
            keep_recent: Number of recent loop iterations to keep fully intact.
                Tune based on workflow depth — shallow workflows (3-5 steps)
                can use 2-3, deep workflows (8-10+) may need 4-6.
        """
        ...
```

**Compaction priority** (what gets cut first → what's preserved longest):

| Priority | Message Type | Phase 1 Treatment | Phase 2 Treatment | Phase 3 Treatment |
|----------|-------------|-------------------|-------------------|-------------------|
| Cut first | `step_nudge`, `retry_nudge` | Drop entirely | Dropped | Dropped |
| Cut second | `tool_result` (older than `keep_recent`) | **Truncate** to first ~200 chars + `[Truncated — N chars removed]` | **Dropped entirely** | Dropped |
| Cut third | `text_response` | Preserved | Preserved | **Dropped** |
| Cut fourth | `reasoning` | Preserved | **Preserved fully** | **Dropped** |
| Preserved | `tool_call` (older than `keep_recent`) | Preserved | Preserved | **Preserved** (full, no transformation) |
| Never cut | `system_prompt`, `user_input` | Preserved | Preserved | Preserved |
| Never cut | Recent iterations (within `keep_recent`) | Preserved in full | Preserved in full | Preserved in full |

**Key design choice:** Reasoning traces survive through Phase 2. The model's chain-of-thought from step 3 ("pricing is $10.69 at MOQ 100, but history shows $9.50 — price below web but above historical") is what informs decisions at step 5+. Losing raw tool results is recoverable; losing the model's interpretation of those results is not. `text_response` (a failed tool call attempt) is expendable after the retry nudge corrects the model.

**Phase 3** is the emergency cutoff. Reasoning and text_response are dropped, leaving only tool_call messages (fully intact, no transformation) in the eligible window. Only the `keep_recent` most recent iterations retain full context. This should only fire under extreme VRAM pressure.

**All three phases are deterministic text manipulation** — no LLM calls, no inference cost, sub-millisecond. They're lossy by design, but the loss is structured and predictable.

---

## Client Adapters

### OllamaClient (Native FC)

Uses Ollama's `/api/chat` endpoint with the `tools` parameter for native function calling. Parses structured `tool_calls` from the response. Primary path for Mistral models. Passes `role="tool"` through to the backend unchanged (Ollama supports it). `api_format = "ollama"`.

**Reasoning gating:** `_resolve_reasoning()` gates reasoning capture on the `_think` flag — when `_think=False`, all reasoning is discarded (returns `None`). This prevents models like Qwen3 from leaking `<think>` content into reasoning fields when the user opted out via `--think false`.

**Timeout handling:** `httpx.ReadTimeout` is caught in both `send()` and `_iter_stream()` and re-raised as `BackendError(408, ...)`, which the eval runner's `ForgeError` handler catches gracefully (records a failed run, batch continues).

```python
class OllamaClient:
    """Native function calling via Ollama's tools API.

    think parameter controls Ollama's thinking/reasoning mode:
        None (default) — auto-detect from model name, fall back on error
        True  — always send think=True (error if model doesn't support it)
        False — never send think
    """

    api_format: str = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        timeout: float = 300.0,
        think: bool | None = None,
    ):
        ...

    def set_num_ctx(self, num_ctx: int | None) -> None:
        """Set the num_ctx override sent on every request."""
        ...

    async def get_context_length(self) -> int | None:
        """Return num_ctx if set via set_num_ctx(), None otherwise.
        Budget resolution lives in ServerManager, not here.
        """
        ...
```

### LlamafileClient (Native FC or Prompt-Injected)

Uses the OpenAI-compatible API (`api_format = "openai"`). Supports native FC via `tools` parameter (used by llama-server with `--jinja`); falls back to prompt-injected JSON extraction otherwise. Also used as the client for llama-server (same OpenAI-compatible API).

`_downgrade_messages()` is applied in the prompt-injected path and does two things: downgrades `role="tool"` to `role="user"` (backend doesn't support tool role), and flattens structured `tool_calls` on assistant messages to `{"tool": ..., "args": ...}` JSON (matching the prompt instruction format, so history acts as few-shot examples).

`_merge_consecutive()` ensures strict user/assistant alternation for Jinja template parity — llama-server's Mistral Jinja template counts only plain user and plain assistant messages (ignoring tool_calls and tool role), so consecutive same-role plain messages at visible positions must be merged.

**`mode="auto"` fallback is explicit, not silent.** Per Design Principle #2, when auto mode detects that native FC is unavailable and falls back to prompt-injected calling, it:
1. Sets `self.resolved_mode` to the actual mode in use (inspectable by the caller)
2. The fallback decision happens once on first `send()`, not per-request

```python
class LlamafileClient:
    """OpenAI-compatible client for Llamafile / llama-server.

    mode="native" uses the tools parameter (requires --jinja or FC support).
    mode="prompt" injects tool descriptions into the prompt and extracts JSON.
    mode="auto" tries native first, falls back to prompt on HTTP error.
    """

    api_format: str = "openai"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        temperature: float = 0.7,
        mode: str = "auto",   # "native", "prompt", or "auto"
        timeout: float = 300.0,
        think: bool | None = None,
        cache_prompt: bool = True,  # Enable llama-server prompt caching
    ):
        ...

    # resolved_mode: str | None — set on first send(). "native" or "prompt".
    # Callers can inspect this to know which path is active.

    async def get_context_length(self) -> int | None:
        """Query the /props endpoint for n_ctx. Strips /v1 from base_url."""
        ...
```

**Think tag handling:** Supports `[THINK]...[/THINK]` (Mistral Reasoning) and `<think>...</think>` (Qwen3, DeepSeek) via `_extract_think_tags()`. The `think` parameter controls capture: `None` (default) auto-captures, `True` always captures, `False` discards reasoning. Server-side `reasoning_content` field takes priority over client-side tag extraction.

### AnthropicClient (Baseline)

Uses the official `anthropic` SDK for Claude models. Serves as a frontier baseline for eval comparisons. `api_format = "openai"` — the runner serializes OpenAI-style messages, and the client converts them to Anthropic format internally.

```python
class AnthropicClient:
    """Anthropic Messages API client for Claude models.

    The runner serializes messages in OpenAI format; this client converts
    them to Anthropic format before each API call.
    """

    api_format: str = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 300.0,
        max_retries: int = 3,
        tool_choice: str | None = None,   # "auto", "any", or None
    ):
        ...

    # last_usage: dict[str, int] | None — populated after each API call
    #   with {"input_tokens": ..., "output_tokens": ...}

    async def get_context_length(self) -> int | None:
        """Claude models have 200K context."""
        return 200_000
```

**Key conversion details:**
- System messages extracted to separate `system=` kwarg
- `tool_calls` → `tool_use` content blocks
- `role="tool"` → `tool_result` content blocks inside user messages
- Unpaired `tool_use` (step/unknown-tool nudges where tool was never executed) get synthetic error `tool_result` blocks injected
- Consecutive same-role messages merged (Anthropic requires strict alternation)

---

## Consumer Validation

Brief check that the library interfaces support two downstream use cases without contortion.

### 1. Home Assistant — Simple Intent Routing with Agentic Escalation

The home assistant uses a simple intent router for 80–90% of requests ("add to calendar," "turn off lights"). Complex requests ("plan my week based on weather and schedule") escalate to this library.

**How it plugs in:**
- Downstream defines tools: `get_calendar`, `get_weather`, `get_tasks`, `create_plan`
- Downstream defines a `Workflow` with `required_steps=["get_calendar", "get_weather"]`, `terminal_tool="create_plan"`
- The intent router is downstream code — it calls `WorkflowRunner.run()` only for complex requests
- Hardware: 12GB GPU → `ContextManager` with tight budget, `TieredCompact` strategy

**Interface friction:** None. The library doesn't assume anything about the routing layer. The `Workflow` is self-contained.

### 2. KILBOTS Vision Pipeline — Phase 2 Sub-Agent Targeting

KILBOTS Phase 2 uses targeted VLM prompts to identify subsystems after initial scene classification. This is conceptually a tool-calling workflow: the VLM calls tools like `identify_radar`, `prioritize_targets`, `select_aim_point`, then terminates with `lock_target`.

**How it plugs in:**
- Tools wrap VLM inference calls (each "tool" is a different targeted prompt evaluated against the current frame)
- `Workflow` with `required_steps=["classify_target", "identify_subsystems"]`, `terminal_tool="lock_target"`
- Hardware: Jetson Orin NX 16GB — tight context budget, aggressive compaction
- Client adapter: custom `LLMClient` implementation wrapping local TensorRT-LLM inference instead of Ollama/Llamafile

**Interface friction:** The `LLMClient` protocol is backend-agnostic, so a TensorRT-LLM adapter works. The library doesn't assume HTTP-based backends. Tool callables can be async (frame capture + inference). `detect_hardware()` returns `None` on non-NVIDIA systems, and budget can be provided explicitly via `ServerManager` or direct `ContextManager` construction.

---

## File Structure

```
forge/
├── src/forge/
│   ├── __init__.py                # Public API exports
│   │
│   ├── core/
│   │   ├── inference.py           # run_inference() — shared front half (compact, fold, validate, retry)
│   │   ├── runner.py              # WorkflowRunner — the agentic loop (back half)
│   │   ├── workflow.py            # Workflow, ToolSpec, ToolCall, TextResponse
│   │   ├── steps.py               # StepTracker
│   │   └── messages.py            # Message, MessageMeta, MessageRole, MessageType
│   │
│   ├── context/
│   │   ├── manager.py             # ContextManager, CompactEvent
│   │   ├── strategies.py          # CompactStrategy ABC + NoCompact, SlidingWindowCompact, TieredCompact
│   │   └── hardware.py            # HardwareProfile, detect_hardware()
│   │
│   ├── clients/
│   │   ├── base.py                # LLMClient protocol, ChunkType, StreamChunk, format_tool()
│   │   ├── ollama.py              # OllamaClient (native FC via Ollama API)
│   │   ├── llamafile.py           # LlamafileClient (native FC or prompt-injected, OpenAI-compatible)
│   │   └── anthropic.py           # AnthropicClient (Claude baseline via anthropic SDK)
│   │
│   ├── guardrails/
│   │   ├── __init__.py            # Public API: ResponseValidator, StepEnforcer, ErrorTracker, Nudge
│   │   ├── nudge.py               # Nudge dataclass (role, content, kind, tier)
│   │   ├── response_validator.py  # ResponseValidator, ValidationResult (rescue + retry + unknown tool)
│   │   ├── step_enforcer.py       # StepEnforcer, StepCheck (wraps StepTracker + premature escalation)
│   │   └── error_tracker.py       # ErrorTracker (consecutive retry/tool error budgets)
│   │
│   ├── prompts/
│   │   ├── templates.py           # build_tool_prompt(), extract_tool_call(), rescue_tool_call()
│   │   └── nudges.py              # retry_nudge(), unknown_tool_nudge(), step_nudge() (3 tiers)
│   │
│   ├── proxy/
│   │   ├── proxy.py               # ProxyServer — programmatic start/stop API
│   │   ├── server.py              # Raw asyncio HTTP server, queue serialization, SSE streaming
│   │   ├── handler.py             # Request handler — bridge between HTTP and run_inference
│   │   └── convert.py             # OpenAI messages ↔ forge Messages conversion
│   │
│   ├── tools/
│   │   ├── __init__.py            # Built-in tool exports
│   │   └── respond.py             # Synthetic respond tool (respond_tool(), respond_spec())
│   │
│   ├── server.py                  # ServerManager, BudgetMode, setup_backend()
│   └── errors.py                  # ForgeError hierarchy
│
├── tests/
│   ├── unit/                      # Deterministic tests — no LLM backend required
│   │   ├── test_messages.py       # Message serialization, metadata handling
│   │   ├── test_steps.py          # StepTracker logic
│   │   ├── test_strategies.py     # Compaction strategies (deterministic)
│   │   ├── test_hardware.py       # HardwareProfile, detect_hardware()
│   │   ├── test_workflow.py       # Workflow definition, prompt rendering
│   │   ├── test_runner.py         # WorkflowRunner with mock LLMClient
│   │   ├── test_ollama_client.py  # OllamaClient with mocked HTTP
│   │   ├── test_llamafile_client.py # LlamafileClient with mocked HTTP
│   │   ├── test_anthropic_client.py # AnthropicClient format conversion
│   │   ├── test_server.py         # ServerManager budget resolution
│   │   ├── test_context_manager.py # ContextManager compaction triggers
│   │   ├── test_templates.py      # Prompt builders, extract/rescue
│   │   ├── test_nudges.py         # Nudge templates
│   │   ├── test_response_validator.py # ResponseValidator (rescue, retry, unknown tool)
│   │   ├── test_step_enforcer.py  # StepEnforcer (premature terminal, escalation, reset)
│   │   ├── test_error_tracker.py  # ErrorTracker (retry/tool error budgets)
│   │   ├── test_eval_budget.py    # Eval budget override logic
│   │   ├── test_bfcl_backends.py  # BFCL backend wiring
│   │   ├── test_bfcl_e2e.py       # BFCL end-to-end
│   │   ├── test_bfcl_executors.py # BFCL executor dispatch
│   │   ├── test_bfcl_report.py    # BFCL report generation
│   │   ├── test_bfcl_runner.py    # BFCL runner logic
│   │   ├── test_bfcl_schema_adapter.py # BFCL schema conversion
│   │   └── test_bfcl_scorer.py    # BFCL scoring logic
│   │
│   └── eval/                      # Eval harness — model qualification against real backends
│       ├── scenarios/             # EvalScenario dataclass, 29 scenarios, ALL_SCENARIOS
│       │   ├── _base.py           # EvalScenario dataclass
│       │   ├── _plumbing.py       # basic_2step, sequential_3step, error_recovery, compaction_stress
│       │   ├── _model_quality.py  # tool_selection, argument_fidelity, sequential_reasoning, etc.
│       │   ├── _compaction.py     # phase2_compaction
│       │   ├── _stateful_plumbing.py      # Stateful variants of plumbing scenarios
│       │   ├── _stateful_model_quality.py  # Stateful variants of model quality scenarios
│       │   ├── _stateful_compaction.py     # Stateful compaction + inventory/supplier scenarios
│       │   └── _compaction_chain.py        # 10-step medical investigation chain (4 budget variants)
│       ├── eval_runner.py         # RunResult, EvalConfig, run_scenario, run_eval, CLI
│       ├── metrics.py             # HistoryStats, ScenarioMetrics, compute_metrics, print_report
│       ├── batch_eval.py          # BatchConfig, batch runner, JSONL output, resume
│       ├── report.py              # ASCII table + list + HTML + Markdown report from JSONL
│       ├── ablation.py            # AblationConfig, ABLATION_PRESETS (reforged/bare/no_*)
│       └── bfcl/                  # Berkeley Function Calling Leaderboard eval
│           ├── runner.py          # BFCL runner
│           ├── batch_runner.py    # BFCL batch runner
│           ├── scorer.py          # BFCL scoring
│           ├── schema_adapter.py  # BFCL schema conversion
│           ├── executors.py       # BFCL executor dispatch
│           ├── backend_wiring.py  # BFCL backend config
│           ├── bfcl_report.py     # BFCL report generation
│           └── data/              # BFCL test data
│
└── scripts/
    └── purge_jsonl.py             # JSONL cleanup utility
```

---

## Out of Scope

| Feature | Why Not Now | When It Might Matter |
|---------|-------------|----------------------|
| **Model-assisted compaction** | Using the LLM itself to summarize context is elegant but costs tokens and adds latency. The heuristic strategies (TieredCompact) are good enough. | When heuristic compaction causes the model to lose critical context in deep workflows (10+ steps). |
| **Multi-model routing** | The library serves one model per workflow run. Routing between Instruct and Reasoning variants, or between fast/deep models, is a downstream concern. | When a project needs to mix models within a single workflow. |
| **Async `on_message` callback** | `on_message` is still sync (fires once per message append, not in hot SSE loop). Low priority — blocking cost is negligible. | When a real async consumer of per-message events emerges. |
| **Token counting via model tokenizer** | Uses char/4 heuristic. Accurate but not precise. | When running very close to context limits on constrained hardware and the ~20% estimation error matters. |
| **Tool argument validation** | Validates that the tool exists and the response parses. Does not validate argument types/values against the `ToolParam` schema before execution. | When tools have complex argument schemas. Downstream can validate in the tool callable for now. |
| ~~**Conversation memory across workflow runs**~~ | **Done.** `run()` accepts `initial_messages` to seed conversation with prior history. Multi-turn callers collect messages via `on_message`, carry them forward, and pass them as `initial_messages` on subsequent turns. See README § `on_message` Callback and Multi-Turn Conversations. | — |

---

## Test Strategy

### Unit Tests

Deterministic, no LLM or backend required. 562 tests across 21 test files, all run natively on Windows (no WSL needed).

| Component | What's Tested | Key Cases |
|-----------|--------------|-----------|
| `Message` | Serialization, `to_api_dict()` strips metadata, both `"ollama"` and `"openai"` formats | Metadata never appears in API dict; all roles round-trip; tool-call messages emit structured `tool_calls` list (1+ `ToolCallInfo` entries) with correct format-specific fields; tool-result messages include `tool_name` or `tool_call_id` |
| `StepTracker` | `record()`, `is_satisfied()`, `pending()`, `summary_hint()` | Empty tracker; partial completion; all steps satisfied; duplicate records |
| `CompactStrategy` implementations | Deterministic compaction behavior; compaction priority order | System prompt always preserved; user input always preserved; nudges dropped first; tool_results truncated (P1) then dropped (P2); reasoning preserved through Phase 2, dropped in Phase 3; `keep_recent` iterations untouched |
| `CompactEvent` / `on_compact` | ContextManager invokes callback with correct event data | `on_compact` called when compaction fires; not called when under budget; event contains accurate before/after counts and phase reached; `on_compact=None` works (no error) |
| `HardwareProfile` | `detect_hardware()` output parsing | nvidia-smi CSV parsing; detection failure returns None |
| `Workflow` | `build_system_prompt()`, `get_tool_specs()`, validation | Template rendering; tool spec completeness; key/name mismatch; terminal in required_steps |
| `WorkflowRunner` (mocked client) | Loop logic, step enforcement, rescue, compaction triggers, terminal tool handling, error propagation, parallel batch execution | Escalating nudge tiers 1/2/3 → `StepEnforcementError`; `rescue_tool_call` salvages TextResponse; counter resets on progress; max iterations → exception; tool raises → `ToolExecutionError`; `rescue_enabled=False` skips rescue; async `on_chunk` awaited per chunk; `on_message` fires for all message types; parallel batches emit 1 TOOL_CALL + N TOOL_RESULT messages |
| `OllamaClient` | Mocked HTTP: send, send_stream, thinking mode | Tool call parsing; reasoning capture gated by `_think` flag; think auto-detect/fallback; `set_num_ctx` wiring; `BackendError`/`ThinkingNotSupportedError`; `httpx.ReadTimeout` → `BackendError(408)` |
| `LlamafileClient` | Mocked HTTP: native/prompt/auto mode, streaming | Mode auto-resolution; `_downgrade_messages` format; `_merge_consecutive` alternation; think tag extraction; `/props` context discovery |
| `AnthropicClient` | Format conversion: OpenAI → Anthropic | System extraction; tool_calls→tool_use; role=tool→tool_result; unpaired tool_use→synthetic error; consecutive same-role merging; `tool_choice` wiring |
| `ServerManager` | Budget resolution logic | VRAM tier lookup; FORGE_FULL/FORGE_FAST/MANUAL/BACKEND modes |
| `Templates` | `build_tool_prompt()`, `extract_tool_call()`, `rescue_tool_call()` | Prompt format; JSON extraction from code fences; rehearsal syntax `tool[ARGS]{...}` |
| `Nudges` | `retry_nudge()`, `unknown_tool_nudge()`, `step_nudge()` tiers | Escalating tier content; available tools listed |
| `BFCL` (7 files) | Schema adapter, scorer, executors, runner, backends, report, E2E | BFCL schema conversion; nested dict/list scoring; executor dispatch; backend wiring |

### Eval Harness

**Purpose:** A model qualification gate, not a consumer test suite. The eval harness answers "how does model X perform against the forge runner plumbing" in isolation — no consumer variability, no real tool complexity. Tools return deterministic hardcoded results; the only variable is whether the model can navigate a multi-step tool-calling workflow N times out of M.

**This is internal development tooling**, not part of the library's public API. It lives under `tests/eval/`, not `src/forge/`. Scenarios are defined in Python (not YAML) for faster iteration.

Not traditional tests — these measure reliability rates against real models. Run manually or on a schedule.

**Running the eval:**

```bash
# All scenarios, 10 runs each, against Ollama
python -m tests.eval.eval_runner --backend ollama --model "ministral-3:8b-instruct-2512-q4_K_M" --runs 10 --stream --verbose

# Filter by tag (plumbing or model_quality)
python -m tests.eval.eval_runner --backend ollama --model "..." --runs 5 --tags plumbing

# Run specific scenarios
python -m tests.eval.eval_runner --backend ollama --model "..." --runs 10 --scenario basic_2step sequential_3step

# llama-server (start server separately, use llamafile backend with native mode)
python -m tests.eval.eval_runner --backend llamafile --llamafile-mode native --model "..." --runs 10 --stream

# Thinking mode (Qwen3, Ministral Reasoning)
python -m tests.eval.eval_runner --backend ollama --model "qwen3:8b-q4_K_M" --runs 10 --stream --think true

# Anthropic baseline
python -m tests.eval.eval_runner --backend anthropic --model claude-haiku-4-5-20251001 --runs 5 --stream

# Ablation (disable specific guardrails)
python -m tests.eval.eval_runner --backend ollama --model "..." --runs 10 --ablation bare

# Override compaction strategy (tiered, sliding, none)
python -m tests.eval.eval_runner --backend llamafile --llamafile-mode native --model "..." --runs 50 --stream --compact-strategy tiered --tags compaction

# Probe context length only (no eval run)
python -m tests.eval.eval_runner --backend ollama --model "..." --probe
```

**Scenarios (29):**

Core scenarios (11):

| Scenario | Tags | Ideal | What It Tests |
|----------|------|-------|---------------|
| `basic_2step` | plumbing | 2 | Basic FC — does the model call tools at all? |
| `sequential_3step` | plumbing | 3 | Required step enforcement via nudges |
| `compaction_stress` | plumbing, compaction | 3 | Completion after compaction fires (2048 token budget) |
| `error_recovery` | plumbing | 2 | Tool error self-correction (TypeError on bad args) |
| `tool_selection` | model_quality | 3 | Correct tool routing among 8 tools (5 distractors) |
| `argument_fidelity` | model_quality | 3 | Extracting args from tool results (entity_id=42) |
| `sequential_reasoning` | model_quality | 4 | 4-step chain with data dependency |
| `conditional_routing` | model_quality, reasoning | 4 | Incident triage with data correlation |
| `data_gap_recovery` | model_quality, reasoning | 5 | Dead-end recovery and tool name mapping |
| `phase2_compaction` | compaction, reasoning | 6 | 5-supplier audit under tight budget (925 tokens) |
| `relevance_detection` | model_quality | 1 | Hallucination resistance — refuses irrelevant queries |

Stateful variants (14) — same logic with `build_workflow` factories that return fresh state per run:

| Scenario | Tags |
|----------|------|
| `basic_2step_stateful` | stateful, plumbing |
| `basic_2step_stateful_tre` | stateful, plumbing |
| `sequential_3step_stateful` | stateful, plumbing |
| `error_recovery_stateful` | stateful, plumbing |
| `tool_selection_stateful` | stateful, model_quality |
| `argument_fidelity_stateful` | stateful, model_quality |
| `sequential_reasoning_stateful` | stateful, model_quality |
| `conditional_routing_stateful` | stateful, model_quality, reasoning |
| `data_gap_recovery_stateful` | stateful, model_quality, reasoning |
| `compaction_stress_stateful` | stateful, compaction |
| `phase2_compaction_stateful` | stateful, compaction, reasoning |
| `inventory_audit` | stateful, compaction |
| `supplier_deep_dive` | stateful, compaction, reasoning |
| `relevance_detection_stateful` | stateful, model_quality |

Compaction chain (4) — 10-step medical investigation with per-phase budget variants:

| Scenario | Tags | What It Tests |
|----------|------|---------------|
| `compaction_chain_baseline` | stateful | Full budget, no compaction pressure |
| `compaction_chain_p1` | stateful, compaction | Budget triggers Phase 1 compaction |
| `compaction_chain_p2` | stateful, compaction | Budget triggers Phase 2 compaction |
| `compaction_chain_p3` | stateful, compaction | Budget triggers Phase 3 compaction |

Each scenario includes a `validate` function for correctness checking — the model must not only complete the workflow but produce accurate terminal tool arguments.

**Metrics collected per run:**

| Metric | Source | How Measured |
|--------|--------|-------------|
| **Completeness** | `RunResult.completeness` | Did the run reach the terminal tool? |
| **Accuracy** | `RunResult.accuracy` | Did the terminal tool args pass validation? |
| **Iterations used** | `RunResult.iterations_used` | LLM round trips (ideal = scenario-defined) |
| **Wasted calls** | iterations - ideal | Extra calls beyond minimum needed |
| **Elapsed time** | Wall clock | Per-run timing |
| **Compaction events** | `ContextManager.on_compact` | Runs where compaction fired, phases reached |
| **Stream retries** | `RunResult.stream_retries` | Client-level stream retry count |
| **Token usage** | `RunResult.input_tokens/output_tokens` | Anthropic API usage tracking |
| **Cost** | `RunResult.cost_usd` | Computed from Anthropic pricing |
| **Retry/step nudges** | `WorkflowRunner.on_message` | Count per run (when history collection is enabled) |

**Aggregate metrics (ScenarioMetrics):**

| Metric | Definition |
|--------|-----------|
| **Score** | correct / total — blended success rate (primary sort key) |
| **Accuracy** | correct / validated — among runs that completed, how many were correct |
| **Completeness** | completed / total — % reaching terminal tool |
| **Efficiency** | ideal_iterations / actual_iterations — 1.0 = perfect |
| **Wasted calls** | avg extra calls beyond minimum |
| **Speed** | avg seconds per run |

### Batch Eval

Run large-scale model comparisons across all backends. Results append to JSONL with automatic resume. Ollama auto-loads models, llama-server is auto-managed (start/stop/health check per GGUF), llamafile binaries require a manual server.

```bash
# Ollama (11 models, fully unattended)
python -m tests.eval.batch_eval --config ollama --runs 50

# llama-server (auto-managed per GGUF)
python -m tests.eval.batch_eval --config llamaserver --runs 50

# Anthropic baselines
python -m tests.eval.batch_eval --config anthropic --runs 50

# Ablation runs
python -m tests.eval.batch_eval --config ollama --runs 50 --ablation bare

# Check progress
python -m tests.eval.report eval_results.jsonl --progress
```

Configs: `all`, `ollama`, `llamaserver`, `llamaserver-native`, `llamaserver-prompt`, `llamafile`, `anthropic`, `anthropic-any`, `haiku`, `sonnet`, `opus`, `haiku-any`, `sonnet-any`, `opus-any`.

### Ablation

The ablation framework measures guardrail impact by selectively disabling runner features:

```python
class AblationConfig:
    name: str                          # Preset name
    rescue_enabled: bool = True        # rescue_tool_call() on TextResponse
    max_retries_per_step: int = 5      # 0 = no retry/unknown-tool nudge
    step_enforcement_enabled: bool = True
    max_tool_errors: int = 2           # 0 = no error recovery
    compaction_enabled: bool = True
```

**Presets:** `reforged` (baseline, all features enabled), `no_rescue`, `no_nudge`, `no_steps`, `no_recovery`, `no_compact`, `bare` (all off).

### Report Generator

ASCII table, phone-friendly list view, HTML dashboard, and Markdown exports from JSONL results. Sorts by score (correct/total), shows per-scenario accuracy breakdown.

```bash
# Full table + list
python -m tests.eval.report eval_results.jsonl

# Compact list only
python -m tests.eval.report eval_results.jsonl --list-only

# Include partially-completed configs
python -m tests.eval.report eval_results.jsonl --include-partial

# Filter by ablation preset or tags
python -m tests.eval.report eval_results.jsonl --ablation reforged bare --tags stateful

# HTML dashboard
python -m tests.eval.report eval_results.jsonl --html dashboard.html

# Markdown reports
python -m tests.eval.report eval_results.jsonl --markdown reports/
```

**Use cases:**
- Qualifying a new model before building a consumer on it
- Regression testing after runner logic changes
- Comparing native FC vs prompt-injected mode for a given model
- Measuring guardrail impact via ablation

---

## Server Management and Budget Resolution

`ServerManager` (`server.py`) owns backend lifecycle (start/stop processes, health polling) and resolves context budgets. It is the single point of truth for "how much context can I use?" — clients just send messages.

### BudgetMode

```python
class BudgetMode(str, Enum):
    BACKEND = "backend"      # Trust the backend's default. No override sent.
    MANUAL = "manual"        # User specifies exact token count.
    FORGE_FULL = "forge-full"  # Max safe context (server auto-tune / Ollama tier).
    FORGE_FAST = "forge-fast"  # Half of full. Trades context for faster attention.
```

### ServerManager

```python
class ServerManager:
    def __init__(self, backend: str, port: int = 8080, models_dir: str | Path | None = None):
        """
        Args:
            backend: "ollama" | "llamaserver" | "llamafile"
        """
        ...

    async def start(self, model, gguf_path, mode="native", extra_flags=None,
                    ctx_override=None, cache_type_k=None, cache_type_v=None,
                    n_slots=None) -> None:
        """Start a llama-server/llamafile process. No-op for Ollama.
        Reuses existing process if same model+mode+ctx+flags+cache+slots.
        cache_type_k/v: KV cache quantization (e.g. "q8_0") — halves cache VRAM.
        n_slots: number of concurrent slots (--parallel N)."""
        ...

    async def stop(self) -> None:
        """Stop server or unload Ollama model."""
        ...

    async def resolve_budget(self, mode: BudgetMode, manual_tokens: int | None = None) -> int:
        """Resolve context budget for the given mode."""
        ...

    async def start_with_budget(self, model, gguf_path, mode="native",
                                budget_mode=BudgetMode.BACKEND, ...,
                                cache_type_k=None, cache_type_v=None,
                                n_slots=None) -> int:
        """Start server + resolve budget in one call. Returns budget in tokens."""
        ...
```

### Budget Resolution by Backend

**Ollama:** Budget based on VRAM tier (same tiers Ollama uses internally):
- `<24GB` → 4,096 tokens
- `24–48GB` → 32,768 tokens
- `>=48GB` → 262,144 tokens

`OllamaClient.set_num_ctx()` sends `num_ctx` on every request via `_build_options()`.

**llama-server / Llamafile:** Budget read from the server's `/props` endpoint (`default_generation_settings.n_ctx`). llama-server auto-tunes context based on available VRAM when started without `-c` — no client-side VRAM computation needed.

**Anthropic:** 200K context, hardcoded in `AnthropicClient.get_context_length()`.

### One-Call Setup

```python
# setup_backend() wires ServerManager + ContextManager together:
client = OllamaClient(model="ministral-3:14b-instruct-2512-q4_K_M")
server, ctx = await setup_backend(
    backend="ollama",
    model="ministral-3:14b-instruct-2512-q4_K_M",
    budget_mode=BudgetMode.FORGE_FULL,
    client=client,  # set_num_ctx() called automatically for Ollama
)
runner = WorkflowRunner(client=client, context_manager=ctx)
# ... run workflows ...
await server.stop()
```

---

## Synthetic Respond Tool

When tools are present but the user sends a conversational message, small models must choose between calling a tool and responding with text. They frequently choose wrong — producing text when they should call tools, or vice versa. Small local models (~8B) cannot be trusted to make this choice correctly — eval testing showed that trusting the model's finish reason dropped workflow completion from 100% to as low as 4%. Guiding the model to a tool is a must.

The respond tool eliminates this ambiguity. The model calls `respond(message="...")` instead of producing bare text. From forge's perspective, every response is a valid tool call — no retries wasted on conversational turns, no accuracy loss on tool-calling turns.

### Three paths

**WorkflowRunner:** Use `respond_tool()` from `forge.tools` to get a ready-made `ToolDef`. Set it as the terminal tool. The callable returns the message string.

```python
from forge import respond_tool

tools = {
    "search": search_tool,
    "respond": respond_tool(),
}
workflow = Workflow(..., tools=tools, terminal_tool="respond")
```

**Proxy:** The respond tool is injected automatically when tools are present in the request. The client never sees it — respond calls are converted to plain text responses (`finish_reason: "stop"`) before returning.

**Middleware:** Include `"respond"` in `tool_names` when creating a `ResponseValidator` or `Guardrails` instance. The respond call passes validation like any other tool call. See `examples/foreign_loop.py` Part 3 for a complete example.

### Why this works for small models

Small models struggle with open-ended decisions ("should I use tools or chat?") but are good at structured choices ("which tool should I call?"). The respond tool converts an open-ended decision into a structured one. The model stays in tool-calling grammar/template at all times, which is where it performs best.

### KV Cache Quantization

`ServerManager.start()` and `setup_backend()` accept `cache_type_k` and `cache_type_v` parameters for KV cache quantization. Using Q8 (`cache_type_k="q8_0"`, `cache_type_v="q8_0"`) roughly halves KV cache VRAM, effectively doubling usable context for the same GPU memory. Measured: 36,864 → 68,608 tokens (1.86x) on Ministral 8B Q4 with no eval regression.

### Multi-Slot Support

`LlamafileClient` accepts a `slot_id` parameter to route requests to specific llama-server slots. `ServerManager.start()` accepts `n_slots` to start the server with `--parallel N`. This enables multi-agent architectures where each agent targets a dedicated KV cache slot.
