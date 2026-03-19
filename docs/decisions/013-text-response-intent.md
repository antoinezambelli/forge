# ADR-013: Text Response Intent — When the Model Chooses Not to Call Tools

**Status:** Draft (March 2026)

## Problem

Forge's guardrail stack treats every `TextResponse` as a failure when tools are present in the request. `ResponseValidator.validate()` sees text, tries rescue parsing, and if that fails, emits a retry nudge. This is correct when the model *failed* to produce a tool call (malformed JSON, forgot the format). It's wrong when the model *chose* to respond with text because no tool was needed.

Real-world example: a multi-turn conversation through opencode where tools are always available (bash, edit, grep, etc.), but the user says "hi" or asks a clarifying question. The model correctly responds with text. Forge retries 3 times, exhausts its budget, and returns the text response anyway — wasting time and confusing the model with incorrect nudges.

This affects all three integration modes:

- **WorkflowRunner**: Masked by StepEnforcer/terminal_tool logic, but the validator still fires unnecessary retries before the runner can make the right decision.
- **Proxy**: No StepEnforcer. Every text response with tools present triggers retry. Clients that mix chat and tool use (opencode, Continue, aider) hit this constantly.
- **Middleware/Guardrails facade**: `check()` returns `"retry"` for valid text responses. Integrators have no way to distinguish "model failed" from "model chose text."

## Context

Backends already signal the model's intent via finish/stop reason:

| Backend | Tool call signal | Text choice signal |
|---------|-----------------|-------------------|
| llama-server | `finish_reason: "tool_calls"` | `finish_reason: "stop"` |
| Ollama | `tool_calls` array present | `done: true`, no tool_calls |
| Anthropic | `stop_reason: "tool_use"` | `stop_reason: "end_turn"` |

All three clients (`OllamaClient`, `LlamafileClient`, `AnthropicClient`) currently discard this signal. They return `TextResponse(content=...)` with no indication of whether the model intended to respond with text or failed to call a tool.

## Approaches

### Approach A: Add `intentional` flag to TextResponse

Add a boolean to the core `TextResponse` type indicating whether the model chose to respond with text.

```python
class TextResponse(BaseModel):
    content: str
    intentional: bool = False  # True = model chose text (finish_reason=stop)
```

**Changes required:**
- `TextResponse` (core type)
- All three clients (map finish_reason/stop_reason/done to `intentional`)
- `ResponseValidator.validate()` (pass through if `intentional=True`)
- Proxy `parse_streamed_response` / `parse_batch_response` (read finish_reason)
- Tests across all affected components

**Behavior by mode:**
- **ResponseValidator**: `intentional=True` -> `needs_retry=False`, no nudge, `tool_calls=None`. Caller gets the text response and decides what to do.
- **WorkflowRunner**: ResponseValidator passes it through. Runner sees no tool calls. StepEnforcer has no tool calls to check. Runner emits the text as a `TEXT_RESPONSE` message and continues. If the workflow needs more tool calls, the model must produce them on the next turn — enforced by max_iterations and StepEnforcer on subsequent turns.
- **Proxy**: Text response passes through to client. No retry. Client decides if that's acceptable.
- **Guardrails facade**: `check()` returns a new action (e.g. `"text"`) with no tool calls and no nudge. Integrator handles it.

**Pros:**
- Architecturally correct — the information belongs on the type
- All three modes benefit
- No wire format changes — proxy stays transparent
- Works uniformly across all backends

**Cons:**
- Touches a core type used everywhere
- All three clients need updates
- `finish_reason` mapping isn't perfectly uniform across backends
- Default `intentional=False` means existing code (which doesn't set the flag) continues to retry — safe but requires all clients to be updated for the fix to take effect

### Approach B: Inject a synthetic `respond` tool

When tools are present, inject a `respond(message: str)` tool into the tool list. The model calls `respond` when it wants to reply with text. Forge sees a valid tool call, no retry needed. Strip the synthetic tool from the response before returning to the client.

```python
# Proxy injects on the way in:
tools.append({"type": "function", "function": {
    "name": "_forge_respond",
    "description": "Respond to the user with a text message (no tool needed)",
    "parameters": {"type": "object", "properties": {
        "message": {"type": "string"}
    }, "required": ["message"]},
}})

# Proxy strips on the way out:
if tool_call.name == "_forge_respond":
    return TextResponse(content=tool_call.args["message"])
```

**Changes required:**
- Proxy handler (inject/strip)
- No core type changes
- No client changes
- No ResponseValidator changes

**Behavior by mode:**
- **Proxy**: Inject on entry, strip on exit. Client never sees the synthetic tool.
- **WorkflowRunner**: Not applicable — runner has terminal_tool.
- **Middleware/Guardrails facade**: Integrator would need to inject the tool themselves, or the facade could do it.

**Pros:**
- No core type changes — completely isolated to proxy/facade
- ResponseValidator stays unchanged — every response is a tool call
- Small models may actually perform better with an explicit "respond" tool vs implicit text fallback
- Quick to implement

**Cons:**
- Modifies the client's tool array — could confuse clients that track tool names
- The model might call `respond` when it should call a real tool (new failure mode)
- It's a forge opinion injected into the wire format — proxy should be transparent
- Only helps proxy and facade consumers, not WorkflowRunner
- Frontier models and APIs handle `tool_choice: "auto"` correctly already — this solves a problem specific to small local models

### Approach C: Respect `tool_choice` from the request

The OpenAI API has `tool_choice` with values `"auto"` (model decides), `"required"` (must call a tool), `"none"` (no tools), or a specific tool name. If the client sends `tool_choice: "auto"` (or omits it, which defaults to auto), text responses are valid. If `tool_choice: "required"`, text responses should be retried.

**Changes required:**
- `ResponseValidator` accepts a `tool_choice` parameter
- Proxy passes `tool_choice` from the request to the validator
- No core type changes

**Pros:**
- Uses existing API semantics — no new concepts
- Simple implementation
- Client explicitly controls the behavior

**Cons:**
- Most clients don't set `tool_choice` explicitly (defaults to auto)
- Doesn't help WorkflowRunner (no `tool_choice` in its API)
- Doesn't capture the model's intent — only the client's preference
- Still doesn't distinguish "model failed" from "model chose text" when `tool_choice` is auto

## Recommendation

Approach A (`intentional` flag) is the correct long-term fix. The information exists at the backend, gets discarded by the clients, and is needed by the validator. Adding it to `TextResponse` is the right place architecturally — it flows naturally through the existing type system.

Approach B (tool injection) is a viable short-term proxy-specific fix if Approach A is too large for the current branch.

Approach C (`tool_choice`) is complementary — it could be added alongside either A or B to give clients explicit control, but it doesn't solve the core problem on its own.

## Open Questions

1. Should `intentional=True` text responses get a new `CheckResult.action` in the Guardrails facade (e.g. `"text"`) or reuse `"execute"` with `tool_calls=None`?
2. For WorkflowRunner: when the model returns `intentional=True` text, should the runner emit it as `TEXT_RESPONSE` and continue the loop, or should it have a way to return text as a valid result (not just terminal tool results)?
3. Should the proxy support both approaches — `intentional` flag as default, tool injection as opt-in via `--inject-respond-tool`?
