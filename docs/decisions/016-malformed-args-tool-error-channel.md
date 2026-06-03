# ADR-016: Malformed tool-call arguments ride the tool-error channel

**Status:** accepted (unreleased)

## Context

forge corrects a misbehaving model through one of **two channels** and counts it against one of **two budgets**:

- **Channels.** A *tool-error result* (`role="tool"`, anchored to the model's own `tool_call_id`) versus a *user nudge* (`role="user"`). The tool-error result rides the channel the model was pretrained on.
- **Budgets.** `max_tool_errors` (default 2) for tool-call faults — the same counter a runtime `FileNotFoundError` from a tool body drains — versus `max_retries` (default 3) for formatting/retry failures.

A malformed tool call — the model emits a structurally valid call whose **arguments** are wrong — comes in two flavors:

1. **Unparseable** — `arguments` is a string that is not valid JSON (`'{not json'`).
2. **Parseable but non-object** — valid JSON that decodes to a list, scalar, or `null` instead of an object.

The prior behavior was inconsistent across forge's clients and its three integration modes (WorkflowRunner, proxy, and the `Guardrails` middleware facade). Most OpenAI-shape clients normalized flavor (1) to a `TextResponse`, routing it back through the **retry** path (user nudge, `max_retries`); one client raised `ValueError`; and flavor (2) was never given a recovery path at all — it raised `ValidationError` at `ToolCall(BaseModel)` construction and crashed the parser. The dominant proxy use-case ran the full validate/retry loop but exposed only `max_retries`.

## Decision

Treat a malformed tool call as a **tool-call fault**, corrected on the **tool-error channel**, uniformly across all clients and all three integration modes.

- **Clients.** A single `decode_tool_args` helper (`clients/base.py`) is the one place every OpenAI-shape client decodes an `arguments` payload. JSON-string args are parsed; a malformed or non-object payload is **kept as-is** (a non-dict value) on the `ToolCall`. It is never coerced to `{}` (which would let the model proceed with empty args — a quiet false success), never collapsed to a `TextResponse`, and never raised. Backends that hand back already-decoded dicts (Ollama, the Anthropic SDK) pass through untouched.
- **Validation.** `ToolCall`/`TextResponse` are plain dataclasses (`args: Any`); construction no longer validates. `ResponseValidator` owns the args-shape check (`isinstance(args, dict)`) and, on failure, emits a `tool_arg_validation` nudge with `role="tool"`.
- **Routing.** Malformed args drain the **tool-error budget** (`max_tool_errors`) and are emitted as a tool-error result. An *unknown-tool* call also rides the tool channel (`role="tool"`) but keeps the **retry** budget — so two explicit kind-sets exist in `guardrails/nudge.py`: `TOOL_CHANNEL_KINDS` (channel) ⊃ `TOOL_ERROR_KINDS` (budget).
- **Consistency.** WorkflowRunner and the proxy share this via `run_inference`. The `Guardrails` facade returns a new `action="tool_error"` for tool-call faults so foreign loops account identically. The proxy exposes `--max-tool-errors` (default 2).

### Why the tool-error channel — and the honest scope

This is a **conditioning bet, not an ontology claim.** A malformed tool call is, strictly, a protocol/formatting failure. We route it over the tool-result channel because a small model **in native tool-calling mode** plausibly self-corrects better when the correction is anchored to its own tool call, on the channel it was pretrained on, rather than via a trailing user nudge that competes for attention in long context. We do **not** assert that malformed JSON "is" a runtime tool error.

Two consequences of that honesty:

- **Native mode only.** In prompt mode the backend cannot consume a `tool` role, so the tool-error result is downgraded to a user message (`llamafile._downgrade_messages`). The channel benefit does not apply there; behavior degrades to the prior retry shape.
- **Two tiers, one budget.** Structural faults (non-dict / unparseable) are caught here, at the validator. Schema/semantic faults (a valid dict with wrong or missing keys) are not checked by forge at all — it does not validate args against the tool's JSON schema — and surface only when the tool body runs. Both tiers already converge on the tool-error budget, which is the coherence this decision leans on.

## Consequences

- **Budget.** Malformed args get `max_tool_errors` (2) corrective turns rather than `max_retries` (3), and share the counter with genuine runtime tool errors. This coupling is deliberate but revisitable: a model that burns the budget on arg-shape mistakes has less room to recover from a real tool failure. If that proves harmful, arg-validation can be split into its own counter later; the kind-sets already make that a localized change.
- **Public surface.** `ToolCall`/`TextResponse` are exported dataclasses with `args: Any` — construction-time validation and the pydantic `.model_*` API are gone (callers that constructed these with keyword args are unaffected; only validation-on-construction is lost). `CheckResult.action` gains `"tool_error"` (no middleware consumers depend on the old vocabulary yet). No proxy wire-format change.
- **Granular safety.** `StepTracker.check_prerequisites` guards against a non-dict `args` (treats it as unsatisfied) so a caller that bypasses `check()` and dispatches directly cannot crash on `args.get`.

## Evidence and validation

This change surfaced from an as-yet-unpublished, higher-difficulty agentic eval suite, where the malformed-args error type appeared and moving corrections to the tool channel improved results. forge's own published eval did not surface the error type at meaningful rates. No regressions were seen in smoke tests, and an upcoming eval re-sweep will exercise this path further. Release messaging frames the change as a native-mode conditioning bet rather than a proven win.

## Alternatives considered

- **Keep the prior `TextResponse` → retry design.** Uniform "any malformed output → retry," dumber clients. Rejected because it (a) left flavor (2) as a hard crash at `ToolCall` construction, and (b) corrected on the user channel even in native mode, where the tool channel is available and (we bet) more effective. The unification it achieved is preserved here — one decode path, clients stop deciding — only the destination changed.
- **Decouple the tool-error budget.** Give arg-validation its own counter, separate from runtime tool errors. Cleaner separation of formatting faults from tool faults, but it expands the change surface; deferred in favor of splitting later if the coupling proves harmful.
