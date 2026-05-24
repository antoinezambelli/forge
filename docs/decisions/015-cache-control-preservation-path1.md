# ADR-015: `cache_control` preservation in the Anthropic proxy (path 1)

**Status:** accepted (v0.7.1)

## Context

forge's proxy accepts Anthropic Messages API requests on `/v1/messages` and runs forge guardrails over the inference loop. Two paths exist:

- **Path 2** (default): inbound Anthropic → translate to OpenAI shape → run against an OpenAI-compatible backend (llama.cpp, vLLM, Ollama, etc.).
- **Path 1** (`--backend-protocol anthropic`): inbound Anthropic → run against an Anthropic-shape downstream (LiteLLM `/v1/messages`, the real Anthropic API, a self-hosted Anthropic proxy).

Anthropic's prompt caching is opt-in per content block: callers tag a block (typically the last block of a stable system prompt or tool definition) with `cache_control: {type: "ephemeral"}`. Anthropic hashes the prefix up to that block and reuses the cached prefix at ~10% of base input cost. Claude Code uses this aggressively to keep large stable contexts cheap across turns.

`cache_control` lives **inside content blocks** (e.g. on a `text` block within a `system` field or message). The forge internal type `forge.Message` does not represent per-block metadata. As the inbound request flows through forge:

1. The proxy parses the Anthropic body to `forge.Message` objects (no `cache_control` field exists to populate).
2. The runner serializes `forge.Message` back to OpenAI dicts internally (`api_format="openai"`) for client consumption.
3. `AnthropicClient._convert_messages` rebuilds Anthropic-shape blocks from the OpenAI dicts — `cache_control` is gone.

On path 2 this loss is intrinsic: OpenAI Chat Completions has no `cache_control` analog. On path 1 the loss is incidental to forge's deconstruct/rebuild architecture — both ends of the wire speak Anthropic, but forge in the middle doesn't preserve the field.

## Decision

For path 1 only, the proxy passes the **original inbound Anthropic body** through to `AnthropicClient` as a separate kwarg. When forge has not mutated the message list during the inference call, the client bypasses `_convert_messages` and sends the inbound body verbatim — preserving `cache_control` (and every other block-level Anthropic field).

The runner clears the verbatim opt-in on any forge-side mutation:

- `ContextManager.maybe_compact()` modifying the message list
- A context-threshold warning being injected
- A retry being triggered (rescue parse, unknown tool, step enforcement, etc.)

On those calls, the client falls back to the existing `_convert_messages` rebuild path. The next clean turn (next CC request) resumes verbatim emit.

## Consequences

| Scenario | Behavior |
|---|---|
| Clean turn, no forge mutation | Verbatim emit. `cache_control` preserved. Anthropic cache hit when prefix matches. |
| Forge retry (append-style mutation) | Rebuild on the retry call. Cache miss on **that call**. Anthropic's stored cache entry persists; next clean turn hits again. |
| Forge compaction (early-message removal) | Rebuild on calls after compaction. Cache miss on those calls. |
| Path 2 (OpenAI-shape downstream) | `cache_control` lost at the protocol boundary. Documented loss; no downstream support exists. |

Cost shape: a single CC turn that triggers a forge retry pays full price for that one retry call (no cache lookup), bounded per-call rather than per-session. Anthropic's content-hash cache on the server side is unaffected.

## Why not modify `forge.Message`

The architecturally complete answer is a per-block annotations field on `forge.Message` so block-level metadata round-trips through forge's internal serialization. That would touch a core primitive type for a benefit limited to one client and one path. Deferred until a concrete user case justifies it.

## Out of scope

- **LiteLLM → on-prem-OpenAI-shape models behind LiteLLM.** Those downstreams don't speak Anthropic's prompt-cache protocol regardless of what forge preserves. The path 1 cache-preservation work is moot for them. forge faithfully passes `cache_control` through to LiteLLM, which then strips it when translating to the on-prem model's protocol.
- **Path 2 `cache_control` preservation via OpenAI prompt-caching analogs** (some vendors have proprietary equivalents). Not currently in scope.

## How to verify

Per-call cache-hit ratio is observable in Anthropic's response `usage` field (`cache_read_input_tokens` vs `input_tokens`). A live integration test pointing CC at a v0.7.1 forge proxy → real Anthropic API would surface the working/non-working split between clean turns and forge-mutation turns.
