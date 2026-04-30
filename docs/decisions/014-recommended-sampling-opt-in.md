# ADR-014: Recommended sampling — opt-in flag and proxy pass-through

**Status:** Implemented (v0.6.0, Apr 2026)

## Context

`MODEL_SAMPLING_DEFAULTS` (introduced in v0.5.x) holds per-model card-recommended sampling parameters sourced one entry at a time from HuggingFace. The first integration splatted `get_sampling_defaults(model)` into client constructors at every call site:

```python
client = LlamafileClient(model=m, **get_sampling_defaults(m))
```

Three problems surfaced in pre-v0.6.0 review:

1. **No first-class API.** Every consumer had to import the helper and splat manually. Unknown models silently returned `{}` — no signal that the map didn't have an entry. The "I want recommended params" intent was buried in unrelated kwargs.
2. **Proxy doc/code drift.** The docstring on `sampling_defaults.py` and `MODEL_GUIDE.md` claimed the proxy forwarded inbound sampling params. The handler discarded them. The proxy's pre-built client used whatever it was constructed with.
3. **Hardcoded `temperature=0.7`** in `OllamaClient` and `LlamafileClient` constructors meant "no sampling kwargs passed" silently meant `T=0.7`, not "backend defaults." Every prior eval run was sampling at 0.7 regardless of model card recommendation. This was the load-bearing bug — fixing it was necessary for any model-comparison work to be meaningful.

## Decision

Three connected changes shipped as a single PR:

### 1. Drop hardcoded `temperature=0.7`

`OllamaClient` and `LlamafileClient` constructors take `temperature: float | None = None`. Send sites only include the field in outbound bodies when non-None. `AnthropicClient` similarly stops sending its hardcoded temperature — Anthropic's API defaults apply.

**Behavior change:** `LlamafileClient(model=m)` previously sent `temperature=0.7`; now sends nothing for temperature, and the backend's own default applies. Callers depending on the implicit 0.7 must pass it explicitly.

### 2. First-class `recommended_sampling: bool = False` opt-in flag

A constructor flag on `OllamaClient` / `LlamafileClient`. Four-quadrant policy matrix:

| Caller passed | Model in map? | Behavior |
|---|---|---|
| `recommended_sampling=True` | yes | Splat map values; caller's explicit kwargs override field-by-field |
| `recommended_sampling=True` | no | **Raise** `UnsupportedModelError` |
| `recommended_sampling=False` (default) | yes | One-shot **INFO** log pointing at the flag; no behavior change |
| `recommended_sampling=False` (default) | no | Silent — same as no map |

Why opt-in, not opt-out: most forge consumers want backend-default sampling, and auto-application would silently change behavior for any consumer who upgrades. Explicit opt-in matches user mental model.

Why raise on opt-in + unknown model: when the caller declares "I want recommended sampling," falling through to backend defaults is a silent failure of intent. Raising forces a clear fix — either add the model to the map (with HF-card URL comment) or drop the flag.

`sampling_defaults.py` splits into two functions to separate lookup from policy:

- `get_sampling_defaults(model) -> dict` — pure lookup. Returns map value (copy) or `{}`. No logging, no raising.
- `apply_sampling_defaults(model, *, strict: bool) -> dict` — policy layer. Implements the four-quadrant matrix above.

Client constructors only call `apply_sampling_defaults`; `get_sampling_defaults` is the clean primitive for callers who want to introspect the map without triggering policy (e.g. proxy callers building request bodies).

### 3. Proxy plumbs body sampling through

`proxy/handler.py` extracts OpenAI-compatible sampling fields from the request body (`temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `seed`) and threads them as a per-call `sampling` dict through `client.send()` and `run_inference()`.

The new `sampling: dict | None = None` kwarg on `LLMClient.send` / `send_stream` and `run_inference` overrides the client's stored sampling **for that call only** — `self` is never mutated. Two proxy requests with different sampling don't see each other.

The proxy's pre-built client stays "blank slate." Body params are the only sampling source in proxy mode. To get card-recommended sampling in proxy mode, the calling client looks up `get_sampling_defaults(model)` and includes the values in the request body.

## Alternatives Considered

- **Auto-apply recommended params if model is in map.** Rejected — silently changes behavior on upgrade. Failed Principle #1: if the consumer didn't ask for it, forge shouldn't silently flip a knob.
- **Single `apply_sampling_defaults` function with `default=True`.** Rejected — entangles lookup with policy. Doc-generation, debugging, and proxy callers all need the lookup primitive without policy side effects.
- **Auto-detect model from a backend probe at construction time.** Rejected — out-of-scope. Consumers always know their model name at construction. Adds runtime dependency and failure modes.
- **Make proxy consult the map.** Rejected — the proxy is meant to be OpenAI-compatible. OpenAI honors body sampling; forge proxy should too. Map consultation in the proxy would surprise callers and create asymmetry between managed-mode and proxy-mode behavior.
- **One PR per issue.** Rejected — the three issues touch overlapping files (clients, sampling_defaults, MODEL_GUIDE, USER_GUIDE, tests). A split would mean reviewing the same files three times. Single PR with three commits in dependency order: drop hardcoded temp → opt-in flag → proxy pass-through.

## Consequences

- Caller-facing API is now explicit: `recommended_sampling=True` says "I want the card-recommended params" and either gets them or fails loud. `False` (default) is identical to pre-v0.6.0 behavior modulo the one-shot info log.
- Pre-v0.6.0 code that constructed clients without sampling kwargs sees one behavior change: `temperature=0.7` is no longer sent. Anything depending on that exact value must pass it explicitly. CHANGELOG calls this out.
- Proxy callers can now pass per-request sampling and have it actually take effect. The proxy is a true OpenAI-compatible pass-through for sampling fields.
- External consumers (forge-code, NORA, etc.) need a one-line update: `**get_sampling_defaults(m)` → `recommended_sampling=True`.
- Eval surface (`tests/eval/batch_eval.py`, `tests/eval/eval_runner.py`) migrated to `recommended_sampling=True`. Now goes through the policy layer — unknown models in the eval list raise instead of silently running with backend defaults.
- The pre-v0.6.0 hardcoded `T=0.7` was a real handicap on eval data: the v0.6.0 dataset (with per-model sampling) shows 3-8 point jumps over the v0.5.0 dataset on most 8B-class configs, attributable to this fix alone.

## References

- Sampling map and policy: `src/forge/clients/sampling_defaults.py`
- Constructor flag implementation: `src/forge/clients/ollama.py`, `src/forge/clients/llamafile.py`, `src/forge/clients/anthropic.py`
- Proxy plumbing: `src/forge/proxy/handler.py`, `src/forge/clients/base.py`, `src/forge/core/inference.py`
- Exception: `src/forge/errors.py` (`UnsupportedModelError`)
- Unit tests: `tests/unit/test_sampling_defaults.py`, `test_ollama_client.py` (`TestRecommendedSampling`, `TestPerCallSampling`), `test_llamafile_client.py`, `test_proxy_handler.py` (`TestSamplingPlumbing`)
- User-facing docs: [MODEL_GUIDE.md#sampling-parameters](../MODEL_GUIDE.md#sampling-parameters), [USER_GUIDE.md#sampling-parameters](../USER_GUIDE.md#sampling-parameters)
- Closes: #58, #59, #61
