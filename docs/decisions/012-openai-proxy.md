# ADR-012: OpenAI-Compatible Proxy Server

**Status:** Implemented (March 2026)

## Problem

Forge's guardrail stack takes an 8B model from ~38% to ~99% reliability on multi-step tool-calling workflows. But the only way to use it today is through WorkflowRunner (you build on forge) or the middleware API (you import forge into your code). Both require Python and code changes.

Many existing tools — Continue, Cursor, aider, any OpenAI SDK consumer — already speak the OpenAI-compatible chat completions API to local model servers (llama-server, Ollama). These tools have their own agentic loops, tool execution, and UI. They don't need forge's runner. They need forge's response quality layer applied transparently to the model traffic.

The specific motivating case: using a reforged Ministral 8B model via llama-server with an external coding tool. The 8B model sometimes returns text instead of tool calls, calls nonexistent tools, or produces malformed output. The client tool's own retry logic (HTTP-level retries) just resends the same request — it doesn't give the model feedback about what went wrong.

## Decision

Build `python -m forge.proxy` — an HTTP proxy server that sits between any OpenAI-compatible client and a model server. The proxy intercepts chat completion requests, applies forge guardrails, and returns clean responses. The client doesn't know forge is there.

```
client  ──POST /v1/chat/completions──>  forge proxy (:8081)
                                            │
                                            ├─ forward to llama-server (:8080)
                                            ├─ buffer full response
                                            ├─ run guardrails (validate, rescue, retry)
                                            │   └─ if bad: inject nudge, re-request
                                            │      (may loop internally N times)
                                            └─ stream final clean response back

client  <──SSE stream (clean)──────────  forge proxy
```

### Backend lifecycle

The proxy supports two modes, mirroring how standalone mode works:

**Managed mode** — forge starts and manages the backend process. Same `ServerManager` + `setup_backend()` lifecycle as WorkflowRunner:
- Startup: `ServerManager.start()` launches llama-server, health-polls until ready
- Budget: resolved via `ServerManager.resolve_budget()` (reads `/props` for n_ctx)
- Shutdown: `ServerManager.stop()` terminates the backend cleanly

```bash
python -m forge.proxy --backend llamaserver --gguf path/to/model.gguf --port 8081
```

**External mode** — user manages the backend, proxy just connects to it:

```bash
python -m forge.proxy --backend-url http://localhost:8080 --port 8081
```

Budget auto-detected from `/props` endpoint, or overridden with `--budget`. `BudgetResolutionError` raised if auto-detection fails (no silent fallback).

### Buffer-then-stream

The proxy fully buffers each response from the backend before deciding what to do. If the response passes guardrails, it streams it back to the client. If not, it injects a nudge and sends a new request to the backend — the client never sees the failed attempt.

From the client's perspective, the proxy is just a slow LLM. This works because:
- **Most OpenAI-compatible clients have no LLM response timeout.** They pass a context and wait.
- **Local 8B models respond in 2-5 seconds.** Even with 2-3 retries, total latency stays under 15 seconds.
- **Simplicity.** No partial-stream-then-backtrack complexity.

### Internal architecture

Messages are converted to forge `Message` objects on entry and stay as forge types internally. OpenAI wire format is only used at two exit boundaries: serializing for the backend (`Message.to_api_dict(format="openai")`) and synthesizing client responses (for rescued tool calls).

This means:
- `ContextManager.maybe_compact()` works naturally (messages have proper `MessageType` tags)
- Retry nudges are appended as forge `Message` objects with correct compaction priority
- Reasoning content is detected (via `reasoning_content` field or `[THINK]`/`<think>` tags) and tagged as `REASONING` for compaction

### Which guardrails apply

| Guardrail | Applies? | Why |
|-----------|----------|-----|
| **Rescue parsing** | Yes | Model returns tool call as plain text -> proxy extracts it and returns structured tool_calls |
| **Retry with nudge** | Yes | Model returns garbage -> proxy injects a nudge into the conversation and re-requests |
| **Unknown tool check** | Yes | Model calls a tool not in the `tools` array -> proxy nudges and re-requests |
| **Context compaction** | Yes | Message history approaching budget -> proxy compacts before forwarding |
| **Step enforcement** | No | Proxy doesn't know the workflow. The client owns step ordering |
| **Error tracking** | Partial | Proxy tracks consecutive retries per request. Cross-request tracking doesn't apply |

### Reasoning / think tags

The proxy passes through reasoning content as-is. Behavior depends on the backend:

- **Streaming**: llama-server sends `reasoning_content` as a separate SSE delta field. The proxy replays raw chunks — consumers that understand `reasoning_content` get it, others ignore it. No API breakage either way.
- **Batch**: llama-server embeds `[THINK]...[/THINK]` tags in the `content` field. The proxy passes this through. This is llama-server's behavior, not forge's. Consumers that parse think tags handle it; others see raw tags in content.

Internally, the proxy detects reasoning in both formats and tags it as `REASONING` for compaction priority (preserved through Phase 2, dropped only in Phase 3 emergency).

### Configuration

```bash
# Managed mode (forge starts llama-server)
python -m forge.proxy \
  --backend llamaserver \
  --gguf path/to/model.gguf \
  --port 8081 \
  --max-retries 3 \
  --verbose

# External mode (user manages the backend)
python -m forge.proxy \
  --backend-url http://localhost:8080 \
  --port 8081 \
  --budget 8192 \
  --max-retries 3 \
  --no-rescue \
  --verbose

# Common flags
--port             Proxy listen port (default: 8081)
--host             Proxy bind address (default: 127.0.0.1)
--budget           Override context budget in tokens (default: auto-detect)
--keep-recent      Compaction: recent iterations to preserve (default: 2)
--max-retries      Guardrail retry attempts per request (default: 3)
--no-rescue        Disable rescue parsing
--no-compact       Disable context compaction
--backend-port     Backend port for managed mode (default: 8080)
--verbose          Debug logging
```

### Endpoints

| Endpoint | Behavior |
|----------|----------|
| `POST /v1/chat/completions` | Core proxy logic (guardrails + forward) |
| `GET /v1/models` | Pass-through to backend |
| `GET /health` | Proxy health check |
| Everything else | Pass-through to backend |

### What this is NOT

- **Not a model server.** Forge doesn't run models. It sits in front of one (or manages the process that does).
- **Not a router.** One backend per proxy instance. Multi-model routing is a separate roadmap item.
- **Not a tool executor.** The client executes tools. Forge only validates that the model's response is well-formed.
- **Not a session manager.** Each request is stateless (compaction derived from the messages array in each request, not stored server-side).

## Tradeoffs

**Added latency.** Every request gets at least one full buffer cycle before the client sees anything. For a local 8B model (2-5s generation), this means the user sees nothing for 2-5s instead of seeing tokens stream in real time. Acceptable for v1; streamable on happy path later.

**Rescue format mismatch.** If the model returns a tool call as plain text and forge rescues it, the proxy synthesizes a proper OpenAI tool_calls response. The client never knows it wasn't a native tool call.

**Compaction changes the messages.** The client sends N messages, forge compacts to M < N, the backend sees M messages. In practice, clients don't inspect what the backend received.

**No step enforcement.** The proxy can't enforce required steps because it doesn't know the workflow. The client owns step ordering.
