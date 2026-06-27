# Backend Setup

How to point forge at a backend. Forge supports six:

| Backend | Forge client | Native FC | Default port | Best for |
|---|---|---|---|---|
| llama-server | `LlamafileClient` | Yes (with `--jinja`) | 8080 | Recommended — top-10 eval configs |
| llamafile | `LlamafileClient` | No (prompt-injected fallback) | 8080 | Single binary, zero setup |
| Ollama | `OllamaClient` | Yes | 11434 | Easiest model management |
| vLLM | `VLLMClient` | Yes (server-side parser) | 8000 | AWQ/GPTQ, high-throughput serving |
| OpenAI-compatible | `OpenAICompatClient` | Per-model | (caller URL) | Hosted providers (Cloudflare, OpenRouter, …) |
| Anthropic | `AnthropicClient` | Yes | (API) | Frontier baseline |

Install instructions for each backend live with the upstream project. Below is what forge expects once a backend is running.

---

## Authentication

forge carries **exactly one credential** to the backend, placed in the backend's
native auth header. forge does not validate the credential, manage its lifecycle
(expiry/refresh), or form any opinion on its value — it only relocates it into
the correct header slot for the target backend. Auth failures therefore surface
as the backend's own error (401/403), not a forge error.

**The one rule:** exactly one credential reaches the backend. If two are present
anywhere, forge **refuses the request** — it never merges, never picks a winner,
never silently drops one. (Design Principle #1: fail fast, fail loud.)

### WorkflowRunner (library use)

Supply the credential at construction, or per call for a rotating token:

```python
# Static credential (API key, service account): set once at construction.
client = OpenAICompatClient(model=..., base_url=..., api_key=API_KEY)

# Rotating credential (e.g. an SSO token refreshed out of band): per call.
await client.send(messages, extra_headers={"Authorization": f"Bearer {token()}"})
```

A construction credential **and** a per-call auth header on the same call is two
credentials → raises `MultipleCredentialsError`. Pass auth through one channel.

For a non-Bearer scheme, pass `extra_headers` alone (omit `api_key`); supplying
both `api_key` and an auth header at construction is also refused.

### Proxy

The proxy gets its one credential from one of two sources — never both:

1. **Inbound passthrough.** The caller's request already carries a credential;
   forge forwards it, relocating the header to the backend's protocol when they
   differ (see the table below). This is the SSO/forwarded-token case.
2. **Static `--backend-api-key`** (or the `FORGE_BACKEND_API_KEY` env var) for
   backends where the caller sends nothing — LM Studio, hosted providers,
   service accounts. Baked into the backend client at startup.

If an inbound auth header **and** `--backend-api-key` are both present, or a
single request carries **two** auth headers, the proxy refuses it with **HTTP
400** (a client error — the message names the conflicting slots, never a secret).
This holds for **streaming** requests too: the credential is resolved (and a
gated backend's context discovered) *before* the `200 OK` / SSE headers are
flushed, so a *conflict* (two credentials) or a discovery failure returns the
real status (400/401) rather than a stream that opens `200 OK` and then carries
an error event.

One streaming case is unavoidable today: an error that surfaces only when the
backend is actually called — the backend **rejecting the credential** (401), or
refusing a request with no credential — happens *after* the SSE headers are
flushed, because the proxy buffers the response rather than streaming it
incrementally. Such failures arrive as an error *event* inside the already-open
`200` stream, not as a `401` status. (Non-streaming requests always get the real
status; and the direct `WorkflowRunner` library path streams incrementally, so
this is specific to the buffered proxy.)

**Cross-protocol relocation.** forge moves the one credential into the target
backend's canonical auth slot (it never reads the secret value):

| Target backend | forge writes |
|---|---|
| OpenAI-wire (llama.cpp, vLLM, Ollama, hosted) | `Authorization: Bearer <token>` |
| Anthropic-wire | `x-api-key: <token>` (forge pins its own `anthropic-version`) |

Same protocol both ends → forwarded verbatim. Cross-protocol → the token is
normalized (a leading `Bearer ` is stripped/added as needed) and written to the
target slot. The common case — Claude Code (Anthropic-wire) in front of an
OpenAI backend — relocates `x-api-key` → `Authorization: Bearer` unambiguously.

> **One documented limitation:** an Anthropic *OAuth* token (which must ride
> `Authorization: Bearer`, not `x-api-key`) pushed through forge's *OpenAI*
> endpoint to an Anthropic backend is relocated to `x-api-key` and rejected by
> Anthropic. Coherent setups never hit this — OAuth callers use the Anthropic
> endpoint (`/v1/messages`), which is same-protocol passthrough.

forge forwards **only** the one credential header; it does not forward the rest
of the inbound header set (so client-set `anthropic-beta`, `OpenAI-Organization`,
etc. do not reach the backend — a future `--backend-header` may add this).

### Notes

- **Ambient `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN`** are read by the
  Anthropic SDK only for *direct* `AnthropicClient()` use (the eval path) — your
  deliberate single credential. The **proxy** neutralizes these env vars at
  construction so an ambient value can't become a hidden second credential.
- **Keyless passthrough to an auth-required backend:** the proxy discovers the
  backend's context length at startup, before any inbound credential exists. If
  that endpoint requires auth, pass `--budget-tokens` so startup doesn't need to
  call it.
- **DEBUG logging** (proxy `-v`) emits the forwarded credential's header *name*
  with the value redacted (`x-api-key: ***`). A raw secret is never logged.

---

## llama-server (recommended)

Upstream: [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)

Boot with `--jinja` for native function calling:

```bash
llama-server -m path/to/Ministral-3-8B-Instruct-2512-Q8_0.gguf --jinja -ngl 999 --port 8080
```

| Flag | Purpose |
|---|---|
| `--jinja` | **Required for native FC.** Without it, the `tools` parameter is ignored. |
| `-ngl 999` | Offload all layers to GPU |
| `-fa` | Flash attention (recommended if supported by your GPU) |
| `-c <N>` | Context size (defaults to model max) |
| `-hf <repo:quant>` | Pull model directly from HuggingFace instead of `-m <path>` |
| `--reasoning-budget 0` | Required for reasoning-tagged models on recent builds — see [Reasoning budget gotcha](#gotcha-reasoning-budget-on-recent-llamacpp-builds) |

Smoke-test the server is up:

```bash
curl http://localhost:8080/v1/models
```

Forge client:

```python
from forge.clients import LlamafileClient

client = LlamafileClient(
    gguf_path="path/to/Ministral-3-8B-Instruct-2512-Q8_0.gguf",
    mode="native",
    recommended_sampling=True,
)
```

The `gguf_path` is the canonical model identity — its file stem is used for sampling-defaults lookup and as the wire-format `model` field. The server itself ignores the wire `model` field, so the path doesn't need to resolve on the machine running forge if the server is remote — only the *file stem* needs to match.

---

## llamafile

Upstream: [llamafile releases](https://github.com/mozilla-ai/llamafile/releases)

Boot with a GGUF:

```bash
llamafile --server --nobrowser -m path/to/model.gguf --port 8080 -ngl 999
```

| Flag | Purpose |
|---|---|
| `--server` | Run in HTTP server mode |
| `--nobrowser` | Don't auto-open the web UI |
| `-ngl 999` | Offload all layers to GPU |
| `-m <path>` | Path to GGUF |

`LlamafileClient` is **native-first**: `mode="native"` (the default) forwards tools via the backend's `tools` parameter and requires native function calling (llama.cpp with `--jinja`). For a backend without native FC, declare `mode="prompt"` to inject tool descriptions into the prompt and parse the JSON call back out. The capability is declared at construction and frozen — there is no runtime auto-detection. Native-first is the default because local-model FC support has matured into the more reliable path; prompt-injection stays fully supported as an explicit opt-in, but note that on more complex, multi-step interactions models tend to struggle to drive the prompt-injected protocol reliably, so reach for it only when the backend leaves no alternative.

> **Proxy note:** the OpenAI-compatible proxy is **native-first**. By default (`--backend-capability native`) it forwards the client's tools verbatim to an FC-capable backend (llama.cpp with `--jinja`, vLLM, Ollama, Anthropic) — the recommended setup. For a non-FC llama.cpp/llamafile backend, opt into prompt-injection with `--backend-capability prompt` (strips tools into the prompt, parses the JSON call back; reuses the same prompt path as the WorkflowRunner). The choice is frozen at startup — there is no runtime auto-detect in the proxy. Reasoning replay is controlled separately with `--reasoning-replay {full,keep-last,none}`; the default `none` keeps captured reasoning out of backend-facing history (`keep-last` replays only the latest captured reasoning block, `full` replays everything). See ADR-012.

Smoke-test:

```bash
curl http://localhost:8080/v1/models
```

Forge client:

```python
from forge.clients import LlamafileClient

client = LlamafileClient(
    gguf_path="path/to/model.gguf",
    mode="prompt",  # default is "native"; use "prompt" only for non-FC backends
    recommended_sampling=True,
)
```

---

## Ollama

Upstream: [ollama.com/download](https://ollama.com/download)

For tool calling, pull a model whose registry page lists `tools` in its tags:

```bash
ollama pull ministral-3:8b-instruct-2512-q4_K_M
```

If the model you want isn't in the Ollama registry, you'll need to create it from a GGUF with a TEMPLATE block that includes the tool-calling tokens — see [Ollama's docs](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) for that workflow. Models without a tool-aware template will reject `tools` requests at the API level.

Smoke-test tool calling specifically:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "ministral-3:8b-instruct-2512-q4_K_M",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "tools": [{"type": "function", "function": {"name": "calc", "description": "Math", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}}],
  "stream": false
}'
```

A response containing `"tool_calls"` means tools are working.

Forge client:

```python
from forge.clients import OllamaClient

client = OllamaClient(
    model="ministral-3:8b-instruct-2512-q4_K_M",
    recommended_sampling=True,
)
```

Notes:
- Ollama lazy-loads models on the first inference request — first call can take 10-30s. `OllamaClient` uses a 300s timeout for this.
- Ollama's API is at `/api/chat`, not OpenAI-compatible. `OllamaClient` handles the conversion.

---

## vLLM

Upstream: [vLLM docs](https://docs.vllm.ai). vLLM is a separate install (not a forge extra) — follow vLLM's guide for your CUDA/ROCm setup.

Boot with server-side tool parsing for native function calling:

```bash
vllm serve /path/to/awq-dir \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --port 8000
```

| Flag | Purpose |
|---|---|
| `--enable-auto-tool-choice` | **Required for native FC.** Without it, the `tools` parameter 400s. |
| `--tool-call-parser <name>` | Parser matching the model family (`hermes`, `mistral`, `llama3_json`, …). |
| `--reasoning-parser <name>` | Splits thinking into a separate `reasoning` field (reasoning models). |
| `--max-model-len <N>` | Context size (forge reads it back from `/v1/models`). |
| `--served-model-name <name>` | Alias clients must send in the `model` field (vLLM 404s on a mismatch). |

vLLM parses tool calls and reasoning **server-side** (unlike llama.cpp's `--jinja` chat-template path), so there is no prompt-injection mode — `VLLMClient` is native-only.

Smoke-test the server is up:

```bash
curl http://localhost:8000/v1/models
```

Forge client:

```python
from forge.clients import VLLMClient

client = VLLMClient(model_path="/path/to/awq-dir")  # or a HuggingFace repo id
```

`model_path` is the canonical identity — a directory of safetensors/config or a HuggingFace repo id; its trailing segment is used for sampling-defaults lookup and the wire `model` field. Unlike llama.cpp, vLLM validates that field against its `--served-model-name`, so in proxy external mode forge auto-discovers the served name from `/v1/models` (pass `--backend vllm`).

---

## Anthropic

Anthropic is a published optional extra:

```bash
pip install "forge-guardrails[anthropic]"
```

Set the API key:

```bash
export ANTHROPIC_API_KEY=sk-...
```

Forge client:

```python
from forge.clients import AnthropicClient

client = AnthropicClient(model="claude-sonnet-4-6")
```

No server to smoke-test — first inference call surfaces auth/network issues.

---

## Hosted OpenAI-compatible providers

Any backend exposing `/v1/chat/completions` with bearer auth — Cloudflare Workers AI, Fireworks, OpenRouter, Together, OpenAI itself, and similar. The client is provider-agnostic: caller supplies the `base_url` and `api_key`; forge has no per-provider knowledge.

Forge client (Cloudflare Workers AI):

```python
from forge.clients import OpenAICompatClient

client = OpenAICompatClient(
    model="@cf/mistralai/mistral-small-3.1-24b-instruct",
    base_url=f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1",
    api_key=API_TOKEN,
)
```

Provider-specific request headers ride on `extra_headers` (e.g. OpenRouter's attribution):

```python
client = OpenAICompatClient(
    model="mistralai/mistral-small-3.1-24b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    extra_headers={"HTTP-Referer": "https://your-app.example", "X-Title": "Your App"},
)
```

Notes:
- **`get_context_length()` returns `None`.** Hosted providers don't expose `max_model_len`. Pass `budget_tokens` explicitly when constructing the `ContextManager` (or `--budget-tokens` to the proxy).
- **Native function calling is per-model, not per-provider.** Many hosted providers serve dozens of models; only the ones with a tool-calling chat template will return structured `tool_calls`. Check the provider's per-model capability docs.
- **Sampling defaults are opt-in.** `recommended_sampling=False` (default) skips the registry lookup, since hosted-provider model identifiers usually aren't in forge's per-model sampling map. Pass explicit `temperature` / `top_p` / etc. as needed.

---

## Gotcha: reasoning budget on recent llama.cpp builds

llama.cpp builds after April 10 2026 activate a reasoning budget sampler for models with thinking tags (Gemma 4, Qwen 3.5, Ministral Reasoning). The default budget is unlimited, which causes some runs to hang indefinitely or fill the KV cache until the server crashes.

Add `--reasoning-budget 0` to disable thinking, or set a specific cap (e.g. `--reasoning-budget 1024`):

```bash
llama-server -m model.gguf --jinja -ngl 999 --port 8080 --reasoning-budget 0
```

Affected models: Gemma 4 (all sizes), Qwen 3.5 (all sizes), Ministral Reasoning. Instruct-only models are not affected.

If you're using forge's managed mode (`setup_backend()` or `ServerManager`), pass this via `extra_flags=["--reasoning-budget", "0"]`.
