# ADR-013: Text Response Intent -- When the Model Chooses Not to Call Tools

**Status:** Superseded (April 2026) — `trust_text_intent` and `TextResponse.intentional` removed. The respond tool (Approach B) replaced them.

## Problem

Forge's guardrail stack treats every `TextResponse` as a failure when tools are present in the request. `ResponseValidator.validate()` sees text, tries rescue parsing, and if that fails, emits a retry nudge. This is correct when the model *failed* to produce a tool call (malformed JSON, forgot the format). It's wrong when the model *chose* to respond with text because no tool was needed.

Real-world example: a multi-turn conversation where tools are always available (bash, edit, grep, etc.), but the user says "hi" or asks a clarifying question. The model correctly responds with text. Forge retries 3 times, exhausts its budget, and returns the text response anyway -- wasting time and confusing the model with incorrect nudges.

This affects all three integration modes:

- **WorkflowRunner**: Masked by StepEnforcer/terminal_tool logic, but the validator still fires unnecessary retries before the runner can make the right decision.
- **Proxy**: No StepEnforcer. Every text response with tools present triggers retry.
- **Middleware/Guardrails facade**: `check()` returns `"retry"` for valid text responses.

## Context

Backends already signal the model's intent via finish/stop reason:

| Backend | Tool call signal | Text choice signal |
|---------|-----------------|-------------------|
| llama-server | `finish_reason: "tool_calls"` | `finish_reason: "stop"` |
| Ollama | `tool_calls` array present | `done: true`, no tool_calls |
| Anthropic | `stop_reason: "tool_use"` | `stop_reason: "end_turn"` |

All three clients currently discard this signal.

## Approaches

### Approach A: Add `intentional` flag to TextResponse

```python
class TextResponse(BaseModel):
    content: str
    intentional: bool = False  # True = model chose text (finish_reason=stop)
```

Changes: TextResponse, all three clients, ResponseValidator, proxy conversion.

ResponseValidator passes through if `intentional=True`. WorkflowRunner's StepEnforcer still catches cases where the model shouldn't have stopped. Proxy and middleware consumers get correct default behavior.

**Pros:** Architecturally correct, all modes benefit, no wire format changes.
**Cons:** Touches core type, all clients need updates.

### Approach B: Inject a synthetic `respond` tool

Inject `_forge_respond(message: str)` into the tool list. Model calls it instead of returning text. Strip on output.

**Pros:** No core type changes, isolated to proxy/facade.
**Cons:** Modifies wire format, new failure mode (model calls respond when it should use tools), only helps proxy/facade.

### Approach C: Respect `tool_choice` from the request

If `tool_choice: "auto"` (default), text responses are valid. If `"required"`, retry.

**Pros:** Uses existing API semantics.
**Cons:** Doesn't distinguish "model failed" from "model chose text" when auto.

## Decision

**Approach A implemented with caller-controlled trust.** The `intentional` flag on TextResponse is a structural signal from the backend's inference engine (EOS token / stop sequence), not a model judgment call. All three clients now read finish reason and set the flag. However, whether the flag is *respected* is controlled by `trust_text_intent` — a parameter on `ResponseValidator.validate()`, `run_inference()`, and `Guardrails.check()`.

### Why caller-controlled trust?

Eval testing (N=25, Ministral 8B reasoning Q4_K_M on llama-server) showed that unconditionally trusting the flag **destroyed workflow reliability** — completion rates dropped from 100% to as low as 4% on reasoning-heavy scenarios. Small models frequently emit `finish_reason: "stop"` when they should call tools. The retry nudge was saving them: the model would produce text, forge would nudge, and the model would self-correct on the next attempt.

With `trust_text_intent=True` (unconditional trust): sequential_reasoning dropped from 100% to 4%, relevance_detection from 100% to 16%, data_gap_recovery from 100% to 32%. With `trust_text_intent=False` (default): all scenarios returned to 100% completion.

### How it works

- **`trust_text_intent=False` (default):** validator ignores the `intentional` flag and retries text responses as before. WorkflowRunner behavior unchanged — full guardrail protection.
- **`trust_text_intent=True`:** validator respects `intentional=True` and passes text through without retry. The proxy sets this for conversational UX.

### Resolved questions

1. **Guardrails facade:** `CheckResult` has a new `action="text"` with a `text` field for intentional content (only when `trust_text_intent=True`). Callers can distinguish it from `"execute"` (tool calls) and `"retry"` (failure).
2. **WorkflowRunner:** does not set `trust_text_intent` — defaults to False. If intentional text somehow reaches the runner (e.g. a future consumer sets it), the runner emits it as a `TEXT_RESPONSE` message and continues the loop, consuming an iteration.
3. **Proxy:** sets `trust_text_intent=True`. Passes intentional text through immediately (no retries). This eliminates wasted retries on conversational turns but accepts the reliability tradeoff.

### Trust implications

The `intentional` flag is structurally correct — the model can't hallucinate a finish reason. But small models (~8B) frequently *choose wrong*: they produce text with `finish_reason: "stop"` when they should call tools. In WorkflowRunner and middleware, the retry nudge catches this. In proxy mode, the tradeoff is accepted for UX reasons — the client's own agentic loop is responsible for re-prompting.

## Superseded

The synthetic `respond` tool (Approach B) was implemented and fully replaces `trust_text_intent`. The `intentional` flag on TextResponse and the `trust_text_intent` parameter have been removed from all interfaces (ResponseValidator, Guardrails, run_inference, and all clients).

**Why:** Small local models (~8B) cannot be trusted to choose correctly between text and tool calls. Eval testing showed that trusting the model's finish reason dropped workflow completion from 100% to as low as 4%. The respond tool eliminates the ambiguity entirely — the model calls `respond(message="...")` instead of producing bare text, staying in tool-calling mode where forge's full guardrail stack applies. Guiding the model to a tool is a must; trusting its text intent is not a viable option for small models.
