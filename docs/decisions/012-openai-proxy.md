# ADR-012: OpenAI-Compatible Proxy Server

**Status:** Draft (March 2026)

## Problem

Forge's guardrail stack takes an 8B model from ~38% to ~99% reliability on multi-step tool-calling workflows. But the only way to use it today is through WorkflowRunner (you build on forge) or the middleware API (you import forge into your code). Both require Python and code changes.

Many existing tools — opencode, Continue, Cursor, aider, any OpenAI SDK consumer — already speak the OpenAI-compatible chat completions API to local model servers (llama-server, Ollama). These tools have their own agentic loops, tool execution, and UI. They don't need forge's runner. They need forge's response quality layer applied transparently to the model traffic.

The specific motivating case: using opencode with a reforged Ministral 8B model via llama-server. opencode sends standard `POST /v1/chat/completions` with `tools` and `stream: true`. The 8B model sometimes returns text instead of tool calls, calls nonexistent tools, or produces malformed output. opencode's own retry logic (HTTP-level, up to 8 retries with backoff) just resends the same request — it doesn't give the model feedback about what went wrong.

## Decision

Build `forge serve` — an HTTP proxy server that sits between any OpenAI-compatible client and any OpenAI-compatible model server. The proxy intercepts chat completion requests, applies forge guardrails, and returns clean responses. The client doesn't know forge is there.

```
opencode  ──POST /v1/chat/completions──>  forge proxy (:8081)
                                              │
                                              ├─ forward to llama-server (:8080)
                                              ├─ buffer full response
                                              ├─ run guardrails (validate, rescue, retry)
                                              │   └─ if bad: inject nudge, re-request
                                              │      (may loop internally N times)
                                              └─ stream final clean response back

opencode  <──SSE stream (clean)──────────  forge proxy
```

### Buffer-then-stream (Option A)

The proxy fully buffers each response from the backend before deciding what to do. If the response passes guardrails, it streams it back to the client. If not, it injects a nudge and sends a new request to the backend — the client never sees the failed attempt.

From the client's perspective, the proxy is just a slow LLM. This works because:
- **opencode has no LLM response timeout.** It passes a context and waits. No timeout on first chunk, between chunks, or overall. Only user cancellation interrupts.
- **Local 8B models respond in 2-5 seconds.** Even with 2-3 retries, total latency stays under 15 seconds — well within what a user would expect from a "thinking" model.
- **Simplicity.** No partial-stream-then-backtrack complexity. The proxy either has a clean response or it doesn't.

A future optimization could stream through on the happy path (first response is good → forward chunks in real time, zero added latency). But for v1, buffer-then-stream is correct and simple.

### Which guardrails apply

The proxy doesn't know the client's workflow structure (what steps are required, what the terminal tool is). It only sees individual chat completion requests. This limits which guardrails are useful:

| Guardrail | Applies? | Why |
|-----------|----------|-----|
| **Rescue parsing** | Yes | Model returns tool call as plain text → proxy extracts it and returns structured tool_calls |
| **Retry with nudge** | Yes | Model returns garbage → proxy injects a nudge into the conversation and re-requests |
| **Unknown tool check** | Yes | Model calls a tool not in the `tools` array → proxy nudges and re-requests |
| **Context compaction** | Yes | Message history approaching budget → proxy compacts before forwarding. Critical for 8B models with 4-8K effective context |
| **Step enforcement** | No | Proxy doesn't know the workflow. The client owns step ordering |
| **Error tracking** | Partial | Proxy tracks consecutive retries per request and gives up after N failures (returns the best response it has). Cross-request error tracking doesn't apply |

So the proxy uses `ResponseValidator` + `ErrorTracker` per request, plus `ContextManager` across the session. `StepEnforcer` is not used.

### Request lifecycle

```
1. Client sends POST /v1/chat/completions
   {messages: [...], tools: [...], stream: true}

2. Proxy extracts tool_names from the tools array

3. Proxy runs context compaction on messages
   (ContextManager.maybe_compact — needs Message conversion)

4. Proxy forwards (possibly compacted) request to backend
   (stream: true internally for efficiency, but buffers fully)

5. Proxy gets complete response from backend

6. Proxy runs ResponseValidator.validate(response)
   a. If valid tool calls → go to step 7
   b. If rescued from text → go to step 7
   c. If needs retry:
      - ErrorTracker.record_retry()
      - If retries exhausted → go to step 7 with best response
      - Inject nudge into messages, go to step 4

7. Proxy streams final response back to client as SSE chunks
   (Replays buffered chunks, or synthesizes from rescued tool calls)
```

### What the proxy needs to handle

**Endpoints:**

| Endpoint | Behavior |
|----------|----------|
| `POST /v1/chat/completions` | Core proxy logic (guardrails + forward) |
| `GET /v1/models` | Pass-through to backend (opencode uses this for model discovery) |
| `GET /health` | Proxy health check |
| Everything else | Pass-through to backend |

**Request/response format:**

The proxy speaks the OpenAI chat completions wire format on both sides. It needs to:
- Parse `messages` and `tools` from the incoming request
- Convert messages to forge `Message` objects for compaction
- Convert back to OpenAI format after compaction
- Parse the backend's streamed response into `list[ToolCall] | TextResponse`
- If retrying, append the nudge as a user message and re-request
- Serialize the final response as OpenAI SSE chunks

This is similar to what `LlamafileClient` already does (OpenAI format in, forge types out, OpenAI format back), but as an HTTP server instead of a client.

**Streaming wire format (SSE):**

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"},"index":0}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"tool_calls":[{"id":"call_0","function":{"name":"grep","arguments":""}}]},"index":0}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"tool_calls":[{"id":"call_0","function":{"arguments":"{\"pattern\":"}}]},"index":0}]}

...

data: {"id":"chatcmpl-...","choices":[{"delta":{},"finish_reason":"tool_calls","index":0}],"usage":{"prompt_tokens":1234,"completion_tokens":56}}

data: [DONE]
```

### Configuration

```bash
# Minimal
forge serve --backend-url http://localhost:8080

# Full
forge serve \
  --port 8081 \
  --backend-url http://localhost:8080 \
  --budget 8192 \
  --max-retries 3 \
  --rescue / --no-rescue \
  --compact-strategy tiered \
  --keep-recent 2 \
  --verbose
```

The client (opencode) is configured to point at forge:
```bash
LOCAL_ENDPOINT=http://localhost:8081/v1
```

### Implementation plan

**Dependencies:** `httpx` (already in forge) for backend requests. Need a lightweight ASGI server — `uvicorn` + raw ASGI, or `starlette` for routing. Avoid heavy frameworks (FastAPI pulls in too much for what's essentially 3 endpoints).

**New files:**

```
src/forge/
  proxy/
    __init__.py
    server.py      # ASGI app, endpoint routing
    handler.py     # Chat completion handler (guardrail loop)
    convert.py     # OpenAI wire format <-> forge Message/ToolCall conversion
    stream.py      # SSE response synthesis (forge response -> SSE chunks)
  cli.py           # `forge serve` CLI entry point
```

**Phases:**

1. **Pass-through proxy** — forwards requests to backend, streams responses back. No guardrails. Proves the plumbing works with opencode.
2. **Response validation** — buffer response, run ResponseValidator, retry with nudge on failure. Core value.
3. **Context compaction** — convert messages to forge format, compact, convert back. Enables long sessions with 8B models.
4. **CLI + config** — `forge serve` entry point with flags.

### What this is NOT

- **Not a model server.** Forge doesn't run models. It sits in front of one.
- **Not a router.** One backend per proxy instance. Multi-model routing is a separate roadmap item.
- **Not a tool executor.** The client executes tools. Forge only validates that the model's response is well-formed.
- **Not a session manager.** Each request is stateless from the proxy's perspective (compaction state is derived from the messages array in each request, not stored server-side). The client owns conversation state.

## Tradeoffs

**Added latency.** Every request gets at least one full buffer cycle before the client sees anything. For a local 8B model (2-5s generation), this means the user sees nothing for 2-5s instead of seeing tokens stream in real time. Acceptable for v1; streamable on happy path later.

**Rescue format mismatch.** If the model returns a tool call as plain text and forge rescues it, the proxy needs to synthesize a proper OpenAI tool_calls response from the rescued ToolCall objects. The client never knows it wasn't a native tool call.

**Compaction changes the messages.** The client sends N messages, forge compacts to M < N, the backend sees M messages. If the client expects exact message round-tripping (unlikely but possible), this could surprise it. In practice, clients don't inspect what the backend received.

**No step enforcement.** The proxy can't enforce required steps because it doesn't know the workflow. If the model skips steps, that's the client's problem. This is fine — step enforcement is a workflow-level concern, and the client owns the workflow.
