# ADR-013: Text Response Intent -- When the Model Chooses Not to Call Tools

**Status:** Draft (March 2026)

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

## Recommendation

Approach A is the correct long-term fix. The information exists at the backend, gets discarded by clients, and is needed by the validator. Approach B is a viable proxy-specific interim fix.

## Open Questions

1. Should `intentional=True` text responses get a new `CheckResult.action` in the Guardrails facade?
2. For WorkflowRunner: when the model returns `intentional=True` text, should the runner emit it and continue the loop, or have a way to return text as a valid result?
3. Should the proxy support both approaches (intentional as default, tool injection as opt-in)?
