# ADR-010: ToolResolutionError

**Status:** Implemented (az/tre branch, Mar 2026)

## Context

Tool callables can fail in two fundamentally different ways:

1. **Hard errors** — bugs, crashes, permission failures. The tool itself is broken. These count toward `consecutive_tool_errors` and eventually raise `ToolExecutionError`.
2. **Resolution failures** — the call was well-formed and schema-valid, but the arguments didn't resolve against the underlying data. Wrong key, empty result set, unrecognized ID. The tool is fine; the model just guessed wrong.

Before this change, both cases hit the same `except Exception` handler: error counter incremented, `[ToolError]` fed back. A model that guessed wrong three times would trip `ToolExecutionError` and kill the workflow — even though the tool was healthy and the model might self-correct on the next try.

## Decision

Introduce `ToolResolutionError`, a new exception type that tool authors raise to signal "valid call, bad data — try again." The runner catches it explicitly, before the generic `Exception` handler, with different semantics:

| Behavior | `ToolResolutionError` | Generic `Exception` |
|---|---|---|
| Message fed back to model | `[ToolResolutionError] ...` | `[ToolError] TypeName: ...` |
| `consecutive_tool_errors` incremented | No | Yes |
| Step recorded as completed | No | No |
| Can trigger `ToolExecutionError` | No | Yes (after `max_tool_errors`) |
| Bounded by | `max_iterations` | `max_iterations` + `max_tool_errors` |

### Not a ForgeError

`ToolResolutionError` inherits from `Exception`, not `ForgeError`. This is deliberate — it's a **tool-author exception**, not a framework error. The consumer raises it from their tool callable; forge catches it in the runner. It sits outside the `ForgeError` hierarchy because it's not an internal failure mode.

### HTTP 4xx analogy

The mental model is HTTP status codes. A hard error is a 5xx — something is broken server-side. A resolution error is a 4xx — the request was valid but the resource doesn't exist. The client (model) should retry with different parameters, not give up.

## Alternatives Considered

- **Reuse the existing error path with a higher threshold** — Raise `max_tool_errors` to tolerate more guesses. But this also raises tolerance for real bugs, masking genuine failures. The two failure modes need separate counters, not a shared one with a higher cap.
- **Return a sentinel value instead of raising** — Tool returns `{"error": "not found"}` as a normal result. Works but is invisible to the framework — no way to distinguish "tool returned an error message" from "tool returned valid data that happens to contain the word error." The runner can't make policy decisions.
- **A `ForgeError` subclass** — Would work mechanically but misrepresents ownership. `ForgeError` means "the framework detected a problem." `ToolResolutionError` means "the tool author is telling the framework something." Keeping it outside the hierarchy makes the boundary clear.
- **Dedicated retry counter** — A separate `max_resolution_retries` cap instead of reusing `max_iterations`. Adds a knob for a problem that `max_iterations` already bounds. The model can't loop forever regardless — no new parameter needed.

## Consequences

- Tool authors have a clean way to signal "try different arguments" without side effects on the error budget
- The runner's error-counting logic is no longer polluted by resolution failures — `consecutive_tool_errors` accurately reflects tool health
- Retries are bounded by `max_iterations`, not a separate counter — no new configuration surface
- Exported in `forge.__init__` as part of the public API
- Eval scenario `basic_2step_stateful_tre` exercises the pattern with a `CountryFactsTRE` backend that raises on unknown keys

## References

- Exception class: `src/forge/errors.py`
- Runner catch branch: `src/forge/core/runner.py` (lines 310–323)
- Public export: `src/forge/__init__.py`
- Unit tests: `tests/unit/test_runner.py` (`TestToolResolutionError`, 9 tests)
- Eval scenario: `tests/eval/scenarios/_stateful_plumbing.py` (`basic_2step_stateful_tre`)
