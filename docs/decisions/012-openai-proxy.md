# ADR-012: OpenAI-Compatible Proxy Server

**Status:** Draft (March 2026) -- architecture revised after prototype

## Problem

Forge's guardrail stack takes an 8B model from ~38% to ~99% reliability on multi-step tool-calling workflows. But consuming it requires either WorkflowRunner (you build on forge) or the middleware API (you import forge into your code). Both require Python and code changes.

Many existing tools -- opencode, Continue, Cursor, aider, any OpenAI SDK consumer -- already speak the OpenAI-compatible chat completions API to local model servers (llama-server, Ollama). These tools have their own agentic loops, tool execution, and UI. They need forge's response quality layer applied transparently to the model traffic.

## Prototype Learnings

An initial prototype built custom request handling, message serialization, and retry logic in the proxy. This reimplemented WorkflowRunner's input processing poorly -- missing reasoning folding, consecutive message merging, and proper serialization. This caused Jinja template errors (`"roles must alternate user and assistant"`) when the conversation history had non-alternating roles from retries.

**Key insight:** WorkflowRunner got an 8B model to beat frontier. The proxy should reuse WorkflowRunner's exact logic, not reimplement it.

## Decision

Extract WorkflowRunner's input processing into a reusable component that both the runner and the proxy consume. The proxy is not a separate implementation -- it's WorkflowRunner's front half behind an HTTP server.

### WorkflowRunner's two halves

**Front half** (reusable -- proxy needs this):
1. Context compaction (`ContextManager.maybe_compact`)
2. Reasoning folding (standalone REASONING msg into TOOL_CALL content field)
3. Serialization (`to_api_dict` with format-appropriate handling)
4. Consecutive message merging (strict role alternation for Jinja templates)
5. Send to backend
6. Validate response (`ResponseValidator` -- rescue, retry nudge, unknown tool)
7. Retry with nudge if needed (loop back to 1)
8. Return clean `list[ToolCall] | TextResponse`

**Back half** (runner-only -- proxy doesn't need this):
1. Step enforcement (`StepEnforcer` -- proxy doesn't know the workflow)
2. Tool execution (proxy doesn't execute tools -- client does)
3. Message emission (`on_message` callback, internal history)
4. Terminal tool check and return
5. Post-batch bookkeeping (error tracking across tool executions)

### Proxy architecture

```
client  --POST /v1/chat/completions-->  forge proxy (:8081)
                                            |
                                            +-- convert OpenAI messages to forge Messages
                                            +-- run WorkflowRunner front half:
                                            |     compact -> fold -> serialize -> merge
                                            |     -> send to backend -> validate
                                            |     -> retry with nudge if needed
                                            +-- convert clean response to OpenAI format
                                            +-- stream back to client

client  <--SSE stream (clean)----------  forge proxy
```

### What the proxy handles

| Guardrail | Applies? | Why |
|-----------|----------|-----|
| **Rescue parsing** | Yes | Model returns tool call as text, proxy extracts it |
| **Retry with nudge** | Yes | Model returns garbage, proxy injects feedback and re-requests |
| **Unknown tool check** | Yes | Model calls nonexistent tool, proxy nudges |
| **Context compaction** | Yes | Message history approaching budget, proxy compacts |
| **Reasoning folding** | Yes | REASONING messages folded into TOOL_CALL for Jinja parity |
| **Message merging** | Yes | Consecutive same-role messages merged for strict alternation |
| **Step enforcement** | No | Proxy doesn't know the workflow |
| **Tool execution** | No | Client executes tools |

### Open design question: text response intent (ADR-013)

When tools are present but the model responds with text (e.g. user says "hi"), should the proxy retry or pass through? See ADR-013 for the full analysis. This affects ResponseValidator at the core level, not just the proxy.

### Backend lifecycle

Two modes, mirroring standalone:

**Managed mode** -- forge starts and manages the backend via ServerManager:
```python
proxy = ProxyServer(backend="llamaserver", gguf="path/to/model.gguf")
proxy.start()   # starts llama-server + proxy
proxy.stop()    # stops both
```

**External mode** -- user manages the backend:
```python
proxy = ProxyServer(backend_url="http://localhost:8080")
proxy.start()   # starts proxy only
proxy.stop()
```

### Request serialization

Single-GPU backends (llama-server, Ollama) can only process one inference at a time. The proxy serializes `/v1/chat/completions` requests via `asyncio.Lock` -- on by default for managed mode, off for external. Override with `--serialize` / `--no-serialize`.

### Buffer-then-stream

The proxy fully buffers each response from the backend before deciding what to do. If valid, streams it back to the client. If not, retries transparently. From the client's perspective, the proxy is just a slow LLM.

### Implementation plan

1. **Extract runner front half** -- factor out the input processing + validation loop from WorkflowRunner into a reusable component. Both runner and proxy consume it. This is the critical step -- get this right and the proxy is a thin HTTP wrapper.
2. **HTTP layer** -- raw ASGI server (asyncio.start_server, no framework dependencies), SSE buffering/replay, ProxyServer start/stop API. Most of this is proven from the prototype.
3. **Conversion layer** -- OpenAI messages to/from forge Messages. Inbound conversion is proven. Outbound uses the extracted serialization logic (no custom code).
4. **Client disconnect handling** -- detect TCP drop, cancel in-flight backend request, release inference lock.
5. **Testing** -- unit tests for extraction, integration tests with mock backend, smoke test with real llama-server.

### Revision: native-first, with opt-in prompt capability

The proxy is **native-first**. By default (`--backend-capability native`) it
targets backends that speak the native OpenAI tools API (llama.cpp with a
tool-calling chat template / `--jinja`, vLLM, Ollama, Anthropic) and forwards
the client's request verbatim (below).

Prompt-injection is available as an **explicit opt-in**
(`--backend-capability prompt`, llama.cpp/llamafile only) for non-FC backends —
it reuses the WorkflowRunner's prompt path (`build_tool_prompt`,
`_downgrade_messages`, `extract_tool_call`) so there is **one** prompt
implementation, not a proxy-specific fork. The capability is **declared once at
construction and frozen** — there is deliberately **no `mode="auto"` runtime
probe** (the old auto/HTTP-error fallback that mutated state mid-request was the
root of the original tangle; it is not reintroduced). In prompt capability the
verbatim passthrough is suppressed (`native_passthrough=False`): tools are
serialized into the prompt, so a raw native transcript would be meaningless.

History: this revision originally cut prompt mode from the proxy entirely
("native-only"). Prompt was then re-added as the opt-in capability above —
native-first is a cleaner story than a backwards-incompatible drop, and non-FC
backends (e.g. llamafile) stay usable through the proxy.

Rationale: the proxy is a transparent layer for an external agent that already
speaks native FC to a native-FC backend. A traced capture showed the native
path forwards the client's request byte-for-byte. The earlier eval regression
(prompt-mode proxy underperforming) was a prompt-injection artifact on an
FC-capable backend, not proxy overhead.

To preserve that transparency, the proxy forwards the client's **verbatim
OpenAI `tools` and `messages`** to the backend on the clean first attempt
(`raw_openai_tools` / `raw_openai_messages`), bypassing the lossy
`ToolSpec.from_json_schema` → `format_tool` round-trip that dropped schema
detail and leaked empty tool names. The parsed `ToolSpec` list is kept only as
forge's validation sidecar. On any forge mutation (retry / compaction / context
warning) the proxy falls back to the folded/serialized form — see the
`use_raw_messages` gate in `run_inference`, which mirrors the ADR-015
`inbound_anthropic_body` drop-on-mutation logic.

The synthetic `respond` tool is **opt-in** (`--inject-respond-tool`, default
off): the proxy forwards the client's tools untouched unless asked to inject it.

If a backend lacking native FC is placed behind the proxy, it degrades to
passing the model's text through (no auto-downgrade) — **bring an FC-capable
backend.**

### What this is NOT

- **Not a model server.** Forge sits in front of one.
- **Not a router.** One backend per proxy instance.
- **Not a tool executor.** Client executes tools.
- **Not a session manager.** Stateless per-request.
