# Backend Setup Guide

How to install and run each LLM backend for forge eval and development. All instructions assume Windows 11 with an NVIDIA GPU (16GB VRAM).

## Model: Ministral 14B Reasoning

The reference model for forge eval is **Ministral-3-14B-Reasoning-2512** (Mistral's Dec 2025 refresh, reasoning variant). Apache 2.0 licensed, native function calling support, produces reasoning traces via `[THINK]` tags before tool calls.

- **HuggingFace (official GGUF):** https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512-GGUF
- **Community quants (more options):** https://huggingface.co/bartowski/mistralai_Ministral-3-14B-Reasoning-2512-GGUF
- **Recommended quant:** `Q4_K_M` (~8.24 GB) — fits comfortably in 16GB VRAM with room for KV cache
- **Alternative:** `Q5_K_M` (~9.6 GB) — slightly higher quality, still fits 16GB

Download the GGUF file and place it somewhere stable (e.g. `C:\models\`).

### Reasoning vs Instruct

| | Reasoning-2512 | Instruct-2512 |
|---|----------------|---------------|
| **Purpose** | Math, coding, STEM, multi-step reasoning | General chat, instruction-following |
| **Reasoning traces** | Yes — `[THINK]` tags, `reasoning_content` in stream | No |
| **Function calling** | Yes (native) | Yes (native) |
| **GGUF Q4_K_M size** | ~8.24 GB | ~8.24 GB |

The Reasoning variant emits thinking tokens before its final answer. This is what forge captures as `ToolCall.reasoning` — the model's chain-of-thought before deciding which tool to call. The Instruct variant almost certainly works too, but reasoning models are the target for forge's multi-step agentic workflows.

> **Open question:** How well do Ollama and llama.cpp surface `[THINK]` traces in their APIs? The model card describes this for vLLM specifically. The eval harness will surface any issues here.

---

## Backend 1: Ollama

**What it is:** Model management + inference server. Handles downloading, quantization selection, and serving behind a single CLI. Native function calling supported since v0.3.0.

**Forge client:** `OllamaClient` — talks to `http://localhost:11434/api/chat`
**Default model name in forge:** `ministral:14b`

### Install

1. Download installer from https://ollama.com/download/windows
2. Run `OllamaSetup.exe` (installs as a background service)
3. Verify: open a terminal and run `ollama --version`

### Option A: Pull from registry (preferred)

If Ollama's registry has your model with tool support, this is simplest:

```bash
ollama pull ministral-3:14b
```

Check `ollama.com/library/<model>` for `tools` in the model's tags. If the pull works, you're done — the template is included.

**Caveat:** Some models require a minimum Ollama version (e.g. `ministral-3` needs 0.13.1+). Check the model page.

### Option B: Create from local GGUF (requires Modelfile with template)

When you `ollama create` from a raw GGUF, Ollama assigns a generic chat template with **no tool-calling support**. The model itself may support tools, but Ollama doesn't know how to format them without the right template. You'll get:

```
{"error":"registry.ollama.ai/library/<model>:<tag> does not support tools"}
```

**The fix:** Supply a TEMPLATE block in your Modelfile that includes the model's tool-calling tokens. Copy it from the official registry model for your architecture.

For Ministral 14B, build a Modelfile manually:

1. Find the official template: `ollama show ministral-3:latest --modelfile` (if you have any version pulled), or browse the template blob at `ollama.com/library/ministral-3:latest`
2. Create a Modelfile with `FROM`, `TEMPLATE """..."""`, and `PARAMETER stop` directives
3. Run `ollama create ministral:14b -f Modelfile`

**Key detail for Mistral/Ministral:** The template must use the `$lastUserIndex` pattern to inject `[AVAILABLE_TOOLS]` before the last user message. Older templates used `le (len (slice ...))` which breaks on multi-round tool calls (tool results push the user message index back). See [ollama/ollama#13334](https://github.com/ollama/ollama/issues/13334).

**Storage note:** `ollama create` copies the weights into Ollama's blob storage (`%USERPROFILE%\.ollama\models\`). After it completes, the original GGUF is no longer referenced — you can move or delete it. This does mean ~8 GB of duplicate disk usage while both copies exist.

### Verify it works

```bash
ollama list                  # Should show ministral:14b
ollama show ministral:14b    # Shows model info, context length, quant
ollama run ministral:14b     # Interactive chat (ctrl+d to exit)
```

Verify tool support specifically:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "ministral:14b",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "tools": [{"type": "function", "function": {"name": "calc", "description": "Math", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}}],
  "stream": false
}'
```

If the response contains `"tool_calls"`, tools are working.

### Run the eval

Ollama serves automatically in the background after install. No need to start a server manually.

```bash
python -m tests.eval.eval_runner --backend ollama --runs 5 --tags plumbing
```

### Useful commands

| Command | Description |
|---------|-------------|
| `ollama serve` | Start server manually (if service isn't running) |
| `ollama list` | List local models |
| `ollama show <model>` | Show model info (quant, context length, etc.) |
| `ollama ps` | List currently loaded/running models |
| `ollama stop <model>` | Unload a model from memory |
| `ollama rm <model>` | Delete a model |
| `ollama pull <model>` | Download a model |
| `ollama create <name> -f <Modelfile>` | Create model from local GGUF |

### Notes

- Ollama auto-manages GPU offloading — no `-ngl` flag needed.
- Default context length depends on the model. Check with `ollama show`.
- API lives at `http://localhost:11434`. This is Ollama's own API format, not OpenAI-compatible (different endpoint structure).
- On Windows, Ollama installs as a background service — `ollama serve` is only needed if the service isn't running. Ctrl+C on `ollama serve` exits silently (no graceful shutdown message); this is normal.
- **Cold-loading:** Unlike llamafile (loads at server start), Ollama lazy-loads models into VRAM on the first inference request. The first API call can take 10-30s. Models stay warm for 5 minutes (default `keep_alive`) then unload. `OllamaClient` uses a 300s timeout to handle this.
- **Raw GGUF = no tools.** If you `ollama create` from a bare GGUF without a TEMPLATE block, tool calling will be rejected at the API level. See "Option B" above.

---

## Backend 2: llamafile

**What it is:** Single-binary LLM server from Mozilla. Runs a GGUF via an OpenAI-compatible API. Does NOT support native function calling as of v0.9.3 — forge uses prompt-injected tool calling as the fallback.

**Forge client:** `LlamafileClient` — talks to `http://localhost:8080/v1/chat/completions`
**Default mode in forge:** `auto` (tries native FC first, falls back to prompt-injected)

### Install

1. Go to https://github.com/mozilla-ai/llamafile/releases
2. Download the latest `llamafile-<version>` binary (e.g. `llamafile-0.9.3`)
3. Rename to `llamafile-0.9.3.exe` (Windows needs the `.exe` extension)
4. Place somewhere on your PATH or in a tools directory

### Start the server

```bash
llamafile-0.9.3.exe --server --nobrowser --host 127.0.0.1 --port 8080 -ngl 999 -m C:/models/Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf
```

Flags:
| Flag | Description |
|------|-------------|
| `--server` | Run in HTTP server mode |
| `--nobrowser` | Don't auto-open the web UI |
| `--host 127.0.0.1` | Bind address (use `0.0.0.0` for network access) |
| `--port 8080` | Listen port (matches forge's default) |
| `-ngl 999` | Offload all layers to GPU |
| `-m <path>` | Path to the GGUF model file |
| `-c <N>` | Context size (optional, defaults to model's max) |

### Verify it works

Server is ready when you see `listening on http://...` in the terminal. Then:

```bash
curl http://localhost:8080/v1/models
```

### Run the eval

Since llamafile lacks native FC, use prompt-injected mode explicitly:

```bash
# Prompt-injected (the only reliable path for llamafile)
python -m tests.eval.eval_runner --backend llamafile --runs 5 --llamafile-mode prompt --tags plumbing

# Auto mode (will attempt native, fail, fall back to prompt)
python -m tests.eval.eval_runner --backend llamafile --runs 5 --tags plumbing
```

### Standalone .llamafile vs llamafile binary + GGUF

| Aspect | Standalone `.llamafile` | Binary + separate GGUF |
|--------|------------------------|----------------------|
| Distribution | Single file with model weights baked in | Two separate files |
| Flexibility | Locked to one model | Swap models with `-m` |
| Use case | End-user distribution | Development (what we use) |

The command you had from a previous project was the binary + GGUF approach — that's the right pattern for development. The `-m` flag pointed at a `.llamafile.exe` which is unusual (that's a standalone binary, not a GGUF). For forge eval, always point `-m` at a `.gguf` file.

### Notes

- llamafile's API is OpenAI-compatible (`/v1/chat/completions`).
- No native tool calling — the `tools` param in the API is silently ignored or errors.
- Forge's `LlamafileClient` in `prompt` mode injects tool descriptions into the system prompt and parses JSON tool calls from the model's raw text output.
- The `/props` endpoint (on the base URL, not `/v1`) returns context length info.

---

## Backend 3: llama-server (llama.cpp)

**What it is:** The reference llama.cpp HTTP server. OpenAI-compatible API with native function calling support (via `--jinja` flag). More control than llamafile, slightly more setup.

**Forge client:** No dedicated client yet. The `LlamafileClient` in `native` mode should work since both speak the same OpenAI-compatible API on the same default port. Alternatively, use `prompt` mode as a safe fallback.

### Install

1. Go to https://github.com/ggml-org/llama.cpp/releases
2. Download the Windows build for your GPU:
   - NVIDIA: `llama-<build>-bin-win-cuda-cu12.2.0-x64.zip` (match your CUDA version)
   - CPU only: `llama-<build>-bin-win-cpu-x64.zip`
3. Extract the zip — you get `llama-server.exe` plus other utilities

### Start the server

```bash
# With native function calling support
llama-server.exe -m C:/models/Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf --jinja --host 127.0.0.1 --port 8080 -ngl 999

# With flash attention (recommended if supported)
llama-server.exe -m C:/models/Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf --jinja -fa --host 127.0.0.1 --port 8080 -ngl 999
```

Flags:
| Flag | Description |
|------|-------------|
| `-m <path>` | Path to the GGUF model file |
| `--jinja` | Enable Jinja templating — **required for native function calling** |
| `-fa` | Enable flash attention (faster, lower memory) |
| `--host 127.0.0.1` | Bind address |
| `--port 8080` | Listen port (matches forge's LlamafileClient default) |
| `-ngl 999` | Offload all layers to GPU |
| `-c <N>` | Context size (optional) |
| `-hf <repo:quant>` | Pull model directly from HuggingFace (alternative to `-m`) |

### Verify it works

```bash
curl http://localhost:8080/v1/models
curl http://localhost:8080/health
```

### Run the eval

Since llama-server speaks the same API as llamafile, use `--backend llamafile`:

```bash
# Native FC (llama-server supports it with --jinja)
python -m tests.eval.eval_runner --backend llamafile --runs 5 --llamafile-mode native --tags plumbing

# Prompt-injected fallback
python -m tests.eval.eval_runner --backend llamafile --runs 5 --llamafile-mode prompt --tags plumbing
```

### Notes

- The `--jinja` flag is critical — without it, the `tools` parameter is ignored and you get no function calling.
- llama-server and llamafile serve the same OpenAI-compatible API on the same default port (8080). They are interchangeable from forge's perspective.
- llama-server gives you native FC; llamafile does not. This is the key difference for eval.
- The `-hf` flag can download models directly: `llama-server.exe --jinja -fa -hf mistralai/Ministral-3-14B-Reasoning-2512-GGUF:Q4_K_M`

---

## Quick Reference

| | Ollama | llamafile | llama-server |
|---|--------|-----------|-------------|
| **Default port** | 11434 | 8080 | 8080 |
| **Forge client** | `OllamaClient` | `LlamafileClient` | `LlamafileClient` |
| **Native FC** | Yes | No | Yes (with `--jinja`) |
| **Prompt-injected FC** | N/A | Yes | Yes (via forge) |
| **API format** | Ollama-native | OpenAI-compatible | OpenAI-compatible |
| **GPU offload** | Automatic | `-ngl 999` | `-ngl 999` |
| **Eval command** | `--backend ollama` | `--backend llamafile` | `--backend llamafile` |

## VRAM Requirements

Weight sizes for all models tested with forge. Total VRAM usage will be higher (add ~0.5–2GB for KV cache and runtime overhead depending on context length).

### 8B-class models

| Model | Quant | Weights | Min VRAM | Recommended VRAM |
|-------|-------|---------|----------|-----------------|
| Ministral-8B Instruct | Q4_K_M | 4.8 GB | 6 GB | 8 GB |
| Ministral-8B Instruct | Q8_0 | 8.4 GB | 10 GB | 12 GB |
| Ministral-8B Reasoning | Q4_K_M | 4.8 GB | 6 GB | 8 GB |
| Ministral-8B Reasoning | Q8_0 | 8.4 GB | 10 GB | 12 GB |
| Qwen3-8B | Q4_K_M | 4.7 GB | 6 GB | 8 GB |
| Qwen3-8B | Q8_0 | 8.1 GB | 10 GB | 12 GB |
| Llama 3.1 8B Instruct | Q4_K_M | 4.6 GB | 6 GB | 8 GB |
| Llama 3.1 8B Instruct | Q8_0 | 8.0 GB | 10 GB | 12 GB |
| Mistral 7B v0.3 | Q4_K_M | 4.1 GB | 6 GB | 8 GB |
| Mistral 7B v0.3 | Q8_0 | 7.2 GB | 8 GB | 12 GB |

### 12B-class models

| Model | Quant | Weights | Min VRAM | Recommended VRAM |
|-------|-------|---------|----------|-----------------|
| Mistral Nemo 12B | Q4_K_M | 7.0 GB | 8 GB | 12 GB |

### 14B-class models

| Model | Quant | Weights | Min VRAM | Recommended VRAM |
|-------|-------|---------|----------|-----------------|
| Ministral-14B Instruct | Q4_K_M | 7.7 GB | 10 GB | 16 GB |
| Ministral-14B Reasoning | Q4_K_M | 7.7 GB | 10 GB | 16 GB |
| Qwen3-14B | Q4_K_M | 8.4 GB | 10 GB | 16 GB |

**Rule of thumb:** Weight size + 1–2 GB for KV cache at moderate context lengths (4–8K tokens). For longer contexts or batch inference, add more headroom.

---

## Smoke Test

After setting up any backend, verify forge can talk to it end-to-end:

```bash
# Run the two simplest eval scenarios (no batch, fast feedback)
python -m tests.eval.eval_runner --backend ollama --model "ministral-3:8b-instruct-2512-q4_K_M" --runs 1 --tags plumbing --scenarios basic_2step

# For llama-server / llamafile (start the server first)
python -m tests.eval.eval_runner --backend llamafile --llamafile-mode native --runs 1 --tags plumbing --scenarios basic_2step
```

If the scenario passes, your backend is wired correctly and ready for full eval runs.

---

## Eval matrix (what to test)

```bash
# 1. Ollama + native FC
python -m tests.eval.eval_runner --backend ollama --runs 10

# 2. llamafile + prompt-injected FC
llamafile-0.9.3.exe --server --nobrowser --port 8080 -ngl 999 -m C:/models/Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf
python -m tests.eval.eval_runner --backend llamafile --runs 10 --llamafile-mode prompt

# 3. llama-server + native FC
llama-server.exe -m C:/models/Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf --jinja --port 8080 -ngl 999
python -m tests.eval.eval_runner --backend llamafile --runs 10 --llamafile-mode native

# 4. llama-server + prompt-injected FC (baseline comparison)
llama-server.exe -m C:/models/Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf --port 8080 -ngl 999
python -m tests.eval.eval_runner --backend llamafile --runs 10 --llamafile-mode prompt
```

This gives you four data points: two backends x two FC modes (minus llamafile native which doesn't exist).
