# Architecture

The *why* of forge. For *what* lives where, see [WORKFLOW.md](WORKFLOW.md) and the source. For specific past decisions, see [decisions/](decisions/).

## What forge is

A reusable Python library for self-hosted LLM tool-calling and multi-step agentic workflows. Forge owns the tool-calling loop — retry logic, context management, step enforcement, and client adapters. It does not own intent routing, model selection, or domain logic; downstream projects build those on top.

**Target hardware:** 12–32GB VRAM (consumer GPUs).
**Backends:** llama-server (llama.cpp), Llamafile, Ollama, Anthropic.
**Surfaced three ways:** `WorkflowRunner` (own the loop), OpenAI-compatible proxy (drop-in for existing harnesses), middleware (`Guardrails` facade for foreign loops).

---

## Design Principles

These are the load-bearing commitments. Most of forge's specific decisions fall out of these five.

### 1. Fail Fast, Fail Loud

No defensive coding. No silent `try/except`, no fallback defaults, no swallowed errors. If the model returns garbage, the retry loop handles it explicitly. If retries are exhausted, forge raises a typed exception with full context (attempt count, last error, last raw response). Silent failures in agentic loops are devastating — a swallowed error at step 3 corrupts every subsequent step.

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

All schemas defined with Pydantic. All LLM outputs validated before execution. Configuration is explicit: when forge auto-detects hardware, it logs what it detected and what budget it chose. The user can always override.

Cloud APIs absorb ambiguity gracefully. A 14B model at Q4 does not. Every implicit assumption is a failure mode.

Concrete consequence: `recommended_sampling=True` is opt-in, never default. With it on, an unknown model raises `UnsupportedModelError` rather than silently inheriting backend defaults. See [ADR-014](decisions/014-recommended-sampling-opt-in.md).

### 3. Control Flow Is Not Memory

Forge separates *what the model remembers* (message history, subject to compaction) from *what the runner enforces* (step completion, iteration count, terminal conditions). The model's context is a resource to be managed. Control-flow state is authoritative and lives outside the message history.

Concrete consequence: step completion is tracked in a `StepTracker` on the runner. Compaction may aggressively drop a tool result, but `StepEnforcer` checks `completed_steps` from the tracker, not from what the model "remembers."

**Tradeoff:** The model may redundantly re-call a tool whose result was compacted. This wastes an iteration but doesn't corrupt the workflow. Tools that are expensive to re-run should be idempotent — that's a downstream contract forge documents but doesn't enforce.

### 4. The Client Adapter Is the Abstraction Boundary

Forge doesn't know whether the LLM supports native function calling, prompt-injected tool calling, or some future protocol. The `LLMClient` adapter translates between forge's internal `ToolCall` representation and whatever the backend expects. The tool-calling loop receives validated `ToolCall` objects and never parses raw text.

This means rescue parsing (Mistral `[TOOL_CALLS]`, Qwen `<tool_call>` XML, fenced JSON, etc.) lives in the *client* — `runner.py` doesn't grow special cases per model family. See [ADR-005](decisions/005-parallel-tool-calls.md) for the batched-tool-call shape and [ADR-013](decisions/013-text-response-intent.md) for the synthetic respond tool's role in this boundary.

### 5. Context Is a First-Class Resource

On consumer hardware, KV cache competes with model weights for VRAM. A 15-step workflow can easily hit 10–20K tokens, pushing a 14B model at Q4 off GPU and into RAM (5–20× slower). Context management is not optional — it's load-bearing infrastructure.

Forge budgets context proactively. The compaction strategy is owned by the strategy object (not the manager), so swapping `TieredCompact` for `SlidingWindowCompact` or a custom strategy is a constructor change.

---

## Surface Modes

Three integration modes, three control / convenience tradeoffs. All three share the same underlying guardrail logic via the middleware layer.

```
forge.guardrails/            <-- extracted guardrail logic (shared)
    ^                ^
forge.proxy          forge.core.runner
(proxy mode)         (WorkflowRunner)
```

- **`WorkflowRunner`** (forge owns the loop) — full feature set: step enforcement, prerequisites, context compaction, threshold callbacks, cancellation, streaming, on_message observability. Best when building on forge directly.
- **Proxy server** (drop-in) — OpenAI-compatible `/v1/chat/completions` endpoint. Applies validation, rescue parsing, retry loop, and synthetic `respond` injection per request. Single-shot — workflow-spanning features (step enforcement, prerequisites, session memory) are out by design because the OpenAI chat-completions schema doesn't carry that state. See [ADR-012](decisions/012-openai-proxy.md).
- **Middleware** (`Guardrails` facade) — for callers running their own loop. Two-method API (`check()` / `record()`) wrapping `ResponseValidator`, `StepEnforcer`, `ErrorTracker`. Returns `Nudge` objects the caller routes however its framework expects. See [ADR-011](decisions/011-guardrail-middleware.md).

The middleware is the foundation; proxy and runner compose the same components.

---

## Guardrails: What They Are and Why

| Guardrail | What it catches | Why it exists |
|---|---|---|
| **Rescue parsing** | Model emits a tool call in the wrong format (fenced JSON, Mistral `[TOOL_CALLS]`, Qwen XML) | Modern API expects structured `tool_calls`; older models still emit inline JSON. Without rescue, the call dies before reaching the tool. |
| **Response validation** | Tool name unknown, tool args malformed | Validates the model's intent before tool execution; routes the corrective message back on the canonical wire shape. |
| **Retry nudges** | Bare text instead of a tool call | Surfaced on the `user` channel — the model needs a positive instruction ("try a tool call") rather than a tool-error reply pattern. |
| **Step enforcement** | Premature `terminal_tool` call | Surfaced as a tool-error reply (`role="tool"`, `[StepEnforcementError]` prefix). Tool-error shape is what OpenAI-tool-trained models pattern-match on for "your call failed, try again" — outperforms trailing user nudges in the wire. |
| **Prerequisites** | Tool A called before prerequisite tool B | Same wire shape as step enforcement (`[PrereqError]`). Constraint is enforced via tool-error reply, not via the tool schema — see [ADR-006](decisions/006-tool-prerequisites.md). |
| **Compaction** | Conversation approaching context budget | Tiered, deterministic; no LLM call. Strategies own their own threshold logic. |

Each guardrail can be independently disabled via ablation presets in `tests/eval/ablation.py` — that's what produces the per-guardrail contribution numbers in the eval reports.

---

## Compaction Strategy Choice

Three built-in strategies; downstream consumers can supply their own by implementing the `CompactStrategy` interface.

- **`NoCompact`** — passthrough. Use when VRAM is abundant or workflows are short.
- **`SlidingWindowCompact`** — keeps system prompt, original user input, and the last N iterations. Simple, predictable. Good baseline.
- **`TieredCompact`** (default) — three-phase escalating compaction with an explicit priority order:

| Priority | Type | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| Cut first | `step_nudge`, `prerequisite_nudge`, `retry_nudge` | Drop | Drop | Drop |
| Cut second | Older `tool_result` | Truncate ~200 chars | Drop | Drop |
| Cut third | `text_response` | Preserved | Preserved | Drop |
| Cut fourth | `reasoning` | Preserved | Preserved | Drop |
| Preserved | Older `tool_call` | Preserved | Preserved | Preserved (full) |
| Never cut | `system_prompt`, `user_input` | Preserved | Preserved | Preserved |
| Never cut | Recent iterations (`keep_recent`) | Preserved | Preserved | Preserved |

**Key design choice — reasoning survives through Phase 2.** The model's chain-of-thought from step 3 ("price below web but above historical") is what informs decisions at step 5+. Losing raw tool results is recoverable; losing the model's interpretation of those results is not. `text_response` (a failed tool call attempt) is expendable after the retry nudge corrects the model.

**Phase 3 is the emergency cutoff** — should only fire under extreme VRAM pressure.

All three phases are deterministic text manipulation — no LLM calls, sub-millisecond.

---

## The Synthetic `respond` Tool

Why it exists: when tools are present but the user sends a conversational message, small models must choose between calling a tool and responding with text. They frequently choose wrong. Eval testing showed that trusting the model's finish reason dropped workflow completion from 100% to as low as 4%.

The respond tool eliminates the open-ended choice. The model calls `respond(message="...")` instead of producing bare text. From forge's perspective, every response is a valid tool call — no retries wasted on conversational turns, no accuracy loss on tool-calling turns.

**Why this works for small models:** small models struggle with open-ended decisions ("tools or chat?") but are good at structured choices ("which tool?"). The respond tool converts an open-ended decision into a structured one. The model stays in tool-calling grammar at all times, which is where it performs best.

Full rationale and the bare-text eval data: [ADR-013](decisions/013-text-response-intent.md).

---

## Sampling Defaults

Each model family has its own card-recommended `temperature` / `top_p` / `top_k`. Running everything at a single default (the usual `0.7`) is a measurable handicap for most models. Forge ships a per-model recommendations map keyed on three identity forms: Ollama-style strings, GGUF stems, llamafile stems. Same value, three keys — vendor-specific guidance can diverge per backend without forcing alignment.

The flag is opt-in (`recommended_sampling=True`):
- **Off** (default) — forge stays out of the way; backend defaults apply. If forge has opinions about this model, it logs a one-shot INFO message pointing at the flag.
- **On, model known** — values applied; caller's explicit non-None kwargs win field-by-field.
- **On, model unknown** — raises `UnsupportedModelError`. Falling through to backend defaults silently would defeat the explicit opt-in.

Proxy mode doesn't consult the map — it plumbs whatever sampling params arrive in the request body. The calling client is expected to look up `get_sampling_defaults(model)` and include them in the body.

Full rationale: [ADR-014](decisions/014-recommended-sampling-opt-in.md). For supported models, citation links, and override patterns: [MODEL_GUIDE.md § Sampling Parameters](MODEL_GUIDE.md#sampling-parameters).

---

## Where to find things

- **What lives where in the code** — [WORKFLOW.md § Quick Reference](WORKFLOW.md#quick-reference)
- **Loop shape, message lifecycle, compaction flow** — diagrams in [WORKFLOW.md](WORKFLOW.md)
- **How to use forge** — [USER_GUIDE.md](USER_GUIDE.md)
- **Backends and boot commands** — [BACKEND_SETUP.md](BACKEND_SETUP.md)
- **Past decisions and rationale** — [decisions/](decisions/) (ADRs)
- **Class signatures and exact APIs** — source (`src/forge/`) is authoritative
