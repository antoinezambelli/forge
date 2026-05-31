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

> **Proxy note:** the OpenAI-compatible proxy is **native-first**. By default (`--backend-capability native`) it forwards the client's tools verbatim to an FC-capable backend (llama.cpp with `--jinja`, vLLM, Ollama, Anthropic) — the recommended setup. For a non-FC llama.cpp/llamafile backend, opt into prompt-injection with `--backend-capability prompt` (strips tools into the prompt, parses the JSON call back; reuses the same prompt path as the WorkflowRunner). The choice is frozen at startup — there is no runtime auto-detect in the proxy. See ADR-012.

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
