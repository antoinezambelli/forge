# Backend Setup

How to point forge at a backend. Forge supports four:

| Backend | Forge client | Native FC | Default port | Best for |
|---|---|---|---|---|
| llama-server | `LlamafileClient` | Yes (with `--jinja`) | 8080 | Recommended — top-10 eval configs |
| llamafile | `LlamafileClient` | No (prompt-injected fallback) | 8080 | Single binary, zero setup |
| Ollama | `OllamaClient` | Yes | 11434 | Easiest model management |
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

llamafile does **not** support native function calling — forge's `LlamafileClient` falls back to prompt-injected mode automatically (`mode="auto"`), or you can force it with `mode="prompt"`.

Smoke-test:

```bash
curl http://localhost:8080/v1/models
```

Forge client:

```python
from forge.clients import LlamafileClient

client = LlamafileClient(
    gguf_path="path/to/model.gguf",
    mode="prompt",  # or "auto" to try native first
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

## Gotcha: reasoning budget on recent llama.cpp builds

llama.cpp builds after April 10 2026 activate a reasoning budget sampler for models with thinking tags (Gemma 4, Qwen 3.5, Ministral Reasoning). The default budget is unlimited, which causes some runs to hang indefinitely or fill the KV cache until the server crashes.

Add `--reasoning-budget 0` to disable thinking, or set a specific cap (e.g. `--reasoning-budget 1024`):

```bash
llama-server -m model.gguf --jinja -ngl 999 --port 8080 --reasoning-budget 0
```

Affected models: Gemma 4 (all sizes), Qwen 3.5 (all sizes), Ministral Reasoning. Instruct-only models are not affected.

If you're using forge's managed mode (`setup_backend()` or `ServerManager`), pass this via `extra_flags=["--reasoning-budget", "0"]`.
