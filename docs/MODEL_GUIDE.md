# Model Guide

Which model and backend to use with forge, based on your hardware and goals.

All numbers from forge's eval harness: 18 scenarios (9 lambda + 9 stateful) × 50 runs per config. 8B–14B models measured on RTX 5070 (12GB); 24B+ models measured on dual RTX 5070 Ti (16GB × 2). Full guardrail stack ("reforged") unless noted. See [EVAL_GUIDE.md](EVAL_GUIDE.md) for full scenario list (22 total including compaction chain) and methodology.

---

## The Short Answer

**12GB VRAM or less:** Ministral-8B Reasoning Q4_K_M on llama-server (native FC) — 99.3% score, 3.7s per workflow, 4.8 GB weights. Outperforms every 14B and 24B model tested.

**32GB VRAM:** Gemma4 31B IT Q4_K_M or Qwen3.5 27B Q4_K_M — both hit 100.0%, matching frontier APIs. First self-hosted models to achieve perfect scores on the eval suite.

If you want the simplest possible setup, **Ministral-8B Instruct Q4_K_M on Ollama** gets you 91% with zero server management.

---

## Quick Picks

| Goal | Model | Backend | Score | Speed |
|------|-------|---------|-------|-------|
| **Maximum reliability** | Claude Sonnet 4.6 | Anthropic API | 100.0% | 6.5s |
| **Best self-hosted (32GB)** | Gemma4 31B IT Q4_K_M | llama-server (native) | 100.0% | 15.3s |
| **Best self-hosted (32GB, fast)** | Qwen3.5 27B Q4_K_M | llama-server (native) | 100.0% | 11.4s |
| **Best self-hosted (12GB)** | Ministral-8B Reasoning Q4_K_M | llama-server (native) | 99.3% | 3.7s |
| **Easiest setup** | Ministral-8B Instruct Q4_K_M | Ollama | 91.2% | 4.2s |
| **No local GPU** | Claude Haiku 4.5 | Anthropic API | 99.6% | 4.0s |
| **Single binary, no deps** | Mistral Nemo 12B Q4_K_M | Llamafile | 82.6% | 4.2s |

---

## By VRAM Budget

### 12GB VRAM (RTX 5070 class)

8B and 14B models fit comfortably. Ministral-8B Reasoning is the top pick — it outperforms every 14B and 24B model tested. The extra headroom from 12GB vs 8GB is better spent on larger context windows (via `-c` flag) than bigger models.

| Model | Backend | Mode | Score | Speed | Notes |
|-------|---------|------|-------|-------|-------|
| Ministral-8B Reasoning Q4_K_M | llama-server | Native | 99.3% | 3.7s | **Best under 32GB** |
| Ministral-8B Reasoning Q8_0 | llama-server | Native | 99.2% | 4.6s | Higher precision, ~same score |
| Ministral-14B Instruct Q4_K_M | llama-server | Native | 98.8% | 3.5s | Marginal gain over 8B Instruct |
| Ministral-8B Instruct Q4_K_M | llama-server | Native | 96.3% | 3.1s | No reasoning overhead |
| Qwen3-14B Q4_K_M | Ollama | Native | 96.3% | 19.6s | Below 8B Reasoning, 5× slower |
| Ministral-8B Instruct Q8_0 | Ollama | Native | 94.6% | 9.1s | Simpler setup |
| Gemma4 E4B IT Q4_K_M | Ollama | Native | 93.0% | 6.6s | MoE, good efficiency |
| Ministral-8B Instruct Q4_K_M | Ollama | Native | 91.2% | 4.2s | Smallest footprint |

### 32GB VRAM (dual RTX 5070 Ti class)

The 32GB tier changes the story. Two self-hosted models achieve 100.0% — matching frontier APIs for the first time. These are the only self-hosted configs that leave zero room for improvement on the eval suite.

| Model | Backend | Mode | Score | Speed | Notes |
|-------|---------|------|-------|-------|-------|
| Gemma4 31B IT Q4_K_M | llama-server | Native | 100.0% | 15.3s | **Perfect score** |
| Gemma4 31B IT Q4_K_M | llama-server | Prompt | 100.0% | 16.8s | Both modes perfect |
| Gemma4 31B IT Q4_K_M | Ollama | Native | 100.0% | 17.0s | All backends perfect |
| Qwen3.5 27B Q4_K_M | llama-server | Native | 100.0% | 11.4s | **Perfect, fastest 100%** |
| Qwen3.5 27B Q4_K_M | llama-server | Prompt | 100.0% | 12.6s | Both modes perfect |
| Qwen3.5 27B Q4_K_M | Ollama | Native | 100.0% | 13.9s | All backends perfect |
| Gemma4 26B A4B IT Q4_K_M | llama-server | Prompt | 99.9% | 4.8s | MoE, near-perfect |
| Qwen3.5 35B A3B Q4_K_M | Ollama | Native | 99.9% | 5.2s | MoE, near-perfect |
| Qwen3.5 35B A3B Q4_K_M | llama-server | Native | 99.8% | 3.7s | **Fastest near-perfect** |

### API (no local GPU)

| Model | Score | Speed | Notes |
|-------|-------|-------|-------|
| Claude Sonnet 4.6 | 100.0% | 6.5s | Perfect score |
| Claude Opus 4.6 | 100.0% | 8.5s | Same score, slower |
| Claude Haiku 4.5 | 99.6% | 4.0s | Fastest, near-perfect |

Haiku for cost-sensitive workloads. Sonnet or Opus if you need the last 0.4%.

---

## Models to Avoid

| Model | Best Score | Why |
|-------|-----------|-----|
| Llama 3.1 8B | 68.8% | Consistently low accuracy across all backends |
| Mistral v0.3 7B | 64.7% | Highly variable, poor tool-call reliability |

These models work but fail too often for production-grade agentic workflows.

---

## The Backend Matters More Than You Think

The same model weights can produce dramatically different results depending on the serving backend. This is a hidden variable that no published benchmark we are aware of controls for.

| Model | Backend/Mode | Score | Notes |
|-------|-------------|-------|-------|
| Mistral Nemo 12B | Llamafile (prompt) | 82.6% | Best result for this model |
| Mistral Nemo 12B | llama-server (prompt) | 75.0% | Same weights, different backend |
| Mistral Nemo 12B | Ollama (native) | 44.6% | Same weights, 38% lower |
| Mistral Nemo 12B | llama-server (native) | 7.2% | Same weights, near-zero |

The same pattern appears across model families:

| Model | llama-server (native) | llama-server (prompt) | Ollama (native) |
|-------|-----------------------|----------------------|-----------------|
| Ministral-8B Reasoning Q4_K_M | 99.3% | 98.1% | — |
| Qwen3-14B Q4_K_M | 88.4% | 93.3% | 96.3% |
| Gemma4 26B A4B Q4_K_M | 96.9% | 99.9% | 85.1% |

**Takeaways:**
- **llama-server native is the best backend for most models** — but not all. Qwen3, Nemo, and Gemma4 MoE variants perform *worse* with native FC on llama-server than with prompt-injected or Ollama.
- **Always test your specific model/backend combination.** Don't assume native FC is better than prompt-injected — it depends on the model's training and the backend's template handling.
- **Forge's prompt-injection fallback is effective.** The gap between native and prompt-injected is often small (1–2%), and sometimes prompt wins.
- **Model availability varies by backend.** llama-server (via GGUF files from HuggingFace) has the widest model selection — any model with a GGUF release works, at any quantization. Ollama's registry is convenient but lags behind and is missing key models (including Ministral-8B Reasoning, the top self-hosted pick). Llamafile has the most limited selection and tends to trail further behind new releases. If you want access to the latest models, start with llama-server + GGUF.

---

## Sampling Parameters

Temperature, `top_p`, `top_k`, `min_p`, `repeat_penalty`, and `presence_penalty` control how the model samples the next token. **Every model family has its own recommended values, and the recommendations differ substantially.** Running all models at a single "default" temperature — which is what most evaluation harnesses do — compares each model outside the sampling zone its authors designed it for.

A few examples of how far recommendations spread:

| Model family | Card-recommended temperature | top_p | top_k |
|---|---|---|---|
| Qwen3 8B/14B (thinking) | 0.6 | 0.95 | 20 |
| Qwen3.5 / 3.6 (thinking, general) | **1.0** | 0.95 | 20 |
| Qwen3-Coder Instruct | 0.7 | 0.8 | 20 |

Running Qwen3.5 27B at the "standard" 0.7 temperature — instead of the card-recommended 1.0 — is a measurable handicap. The same logic applies across Gemma, Mistral, and Qwen families, all of which publish different recommendations.

### How forge handles it

Forge ships a per-model recommendations map at `forge.clients.sampling_defaults`. Each entry is sourced directly from the model's HuggingFace card (or, when the vendor has not published sampling on the card, from a secondary source that cites the vendor — Granite 4.0 is the current example), with the source URL as an inline comment. Values are verified one entry at a time — no best-effort or extrapolated entries.

```python
from forge.clients import LlamafileClient

# Managed mode — opt in to recommended defaults via constructor flag
client = LlamafileClient(
    model="qwen3.5:27b-q4_K_M",
    mode="native",
    recommended_sampling=True,
)
```

The flag is opt-in. Default behavior (`recommended_sampling=False`) leaves sampling to backend defaults; if forge has opinions about the model, it logs a one-shot INFO message pointing the caller at the flag. With `recommended_sampling=True`, an unknown model raises `UnsupportedModelError` — falling through to backend defaults silently would defeat the explicit opt-in.

Caller's explicit non-None sampling kwargs win field-by-field over the map:

```python
client = LlamafileClient(
    model="qwen3.5:27b-q4_K_M",
    mode="native",
    recommended_sampling=True,
    temperature=0.5,  # overrides the map's 1.0; other map fields still apply
)
```

For programmatic introspection without triggering policy, `forge.clients.get_sampling_defaults(model)` is a pure lookup — returns the map value (a fresh copy) or `{}` for unknown models. No logging, no raising.

**Unknown models** (not in the map): forge supports all models; it only has opinions about the ones in the map. Without `recommended_sampling=True`, an unknown model gets backend defaults silently. With it, you get a fail-loud `UnsupportedModelError`.

**Proxy mode** does not consult the map. The proxy plumbs whatever sampling params the inbound request body carries (OpenAI-compatible fields: `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `seed`) through to the backend on a per-call basis without mutating the proxy's pre-built client. To get recommended-sampling behavior in proxy mode, the calling client looks up `get_sampling_defaults(model)` and includes the values in the request body.

### Supported models

| Model | temp | top_p | top_k | min_p | repeat_penalty | presence_penalty | Source |
|---|---|---|---|---|---|---|---|
| `qwen3:4b-instruct-2507-q4_K_M` | 0.7 | 0.8 | 20 | 0.0 | — | — | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| `qwen3:4b-thinking-2507-q4_K_M` | 0.6 | 0.95 | 20 | 0.0 | — | — | [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) |
| `qwen3:8b-q4_K_M` | 0.6 | 0.95 | 20 | 0.0 | — | — | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| `qwen3:8b-q8_0` | 0.6 | 0.95 | 20 | 0.0 | — | — | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| `qwen3:14b-q4_K_M` | 0.6 | 0.95 | 20 | 0.0 | — | — | [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |
| `qwen3.5:27b-q4_K_M` | 1.0 | 0.95 | 20 | 0.0 | — | 1.5 | [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |
| `qwen3.5:35b-a3b-q4_K_M` | 1.0 | 0.95 | 20 | 0.0 | — | 1.5 | [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |
| `qwen3.6:35b-a3b-ud-q4_K_M` | 1.0 | 0.95 | 20 | 0.0 | — | 1.5 | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| `qwen3-coder:30b-a3b-instruct-q4_K_M` | 0.7 | 0.8 | 20 | — | 1.05 | — | [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| `gemma4:31b-it-q4_K_M` | 1.0 | 0.95 | 64 | — | — | — | [gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) |
| `gemma4:26b-a4b-it-q4_K_M` | 1.0 | 0.95 | 64 | — | — | — | [gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it) |
| `gemma4:26b-a4b-it-q8_0` | 1.0 | 0.95 | 64 | — | — | — | [gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it) |
| `gemma4:e4b-it-q4_K_M` | 1.0 | 0.95 | 64 | — | — | — | [gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it) |
| `gemma4:e4b-it-q8_0` | 1.0 | 0.95 | 64 | — | — | — | [gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it) |
| `mistral-small-3.2:24b-instruct-2506-q4_K_M` | 0.15 | — | — | — | — | — | [Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) |
| `mistral-small-3.2:24b-instruct-2506-q8_0` | 0.15 | — | — | — | — | — | [Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) |
| `devstral-small-2:24b-instruct-2512-q4_K_M` | 0.15 | — | — | — | — | — | [Devstral-Small-2-24B-Instruct-2512](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) |
| `devstral-small-2:24b-instruct-2512-q8_0` | 0.15 | — | — | — | — | — | [Devstral-Small-2-24B-Instruct-2512](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) |
| `ministral-3:8b-instruct-2512-q4_K_M` | 0.05¹ | — | — | — | — | — | [Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) |
| `ministral-3:8b-instruct-2512-q8_0` | 0.05¹ | — | — | — | — | — | [Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) |
| `ministral-3:14b-instruct-2512-q4_K_M` | 0.05¹ | — | — | — | — | — | [Ministral-3-14B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) |
| `ministral-3:8b-reasoning-2512-q4_K_M` | 0.7 | —² | — | — | — | — | [Ministral-3-8B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512) |
| `ministral-3:8b-reasoning-2512-q8_0` | 0.7 | —² | — | — | — | — | [Ministral-3-8B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512) |
| `ministral-3:14b-reasoning-2512-q4_K_M` | 1.0 | —² | — | — | — | — | [Ministral-3-14B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512) |
| `mistral-nemo:12b-instruct-2407-q4_K_M` | 0.3 | — | — | — | — | — | [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) |
| `granite-4.0:h-micro-q4_K_M` | 0.0³ | 1.0 | 0 | — | — | — | [Unsloth IBM-Granite-4.0 tutorial](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0) (cites IBM) |
| `granite-4.0:h-tiny-q4_K_M` | 0.0³ | 1.0 | 0 | — | — | — | [Unsloth IBM-Granite-4.0 tutorial](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0) (cites IBM) |

¹ Ministral-3 Instruct cards say "temperature below 0.1 for production"; 0.05 picked within that range.
² Ministral-3 Reasoning cards show `top_p=0.95` in code examples but do NOT include it in the formal "Recommended Settings" section — omitted here. Add it explicitly if you want to follow the examples.
³ Granite 4.0 sampling is greedy decoding (T=0); `top_p=1.0` and `top_k=0` are mathematical no-ops at T=0 but kept explicit to match the source recommendation. IBM's own HF cards, the granite-4.0-language-models GitHub repo, and the "Granite 4.0 Prompt engineering guide v2" do not publish sampling values directly — Unsloth's tutorial is a secondary source that cites IBM.

**Intentionally absent from the map** (no formal recommendation on the official card):
- **Llama 3.1 8B Instruct** — Meta's HF card, llama.com/docs, and llama-recipes are all silent on sampling.
- **Mistral 7B Instruct v0.3** — HF card has no "recommended settings" section; code examples use `temperature=0.0` (greedy) but explicitly note it's demo-only.

Rows using these models hit the unknown-model path and inherit backend defaults. Both are also in the [Models to Avoid](#models-to-avoid) section. The sparseness of official sampling guidance tracks with these being older or less-agentically-tuned releases.

A dash means the card does not specify a value for that parameter — forge sends nothing and the backend's default applies.

**Profile choices.** When a card gives multiple profiles (e.g. Qwen3.5 has separate "general" vs "precise coding" columns), forge uses the **general-tasks thinking-mode** profile. Consumers that know their workload better (code-focused harnesses, for instance) should override explicitly.

### Overriding

The simplest path: opt in and override individual fields. Caller's explicit non-None kwargs win over the map field-by-field.

```python
# Card-recommended general-tasks profile, but with the precise-coding (WebDev)
# profile's temperature and presence_penalty.
client = LlamafileClient(
    model="qwen3.5:27b-q4_K_M",
    mode="native",
    recommended_sampling=True,
    temperature=0.6,
    presence_penalty=0.0,
)
```

For programmatic access to the map without triggering policy:

```python
from forge.clients import get_sampling_defaults

defaults = get_sampling_defaults("qwen3.5:27b-q4_K_M")  # fresh dict, safe to mutate
defaults["temperature"] = 0.6
client = LlamafileClient(model="qwen3.5:27b-q4_K_M", mode="native", **defaults)
```

For fully manual control, pass sampling kwargs directly and skip the helpers.

---

## Key Findings

1. **Guardrails matter more than model size.** Forge's guardrail stack adds 10–79% accuracy depending on the model. Ministral-8B Instruct on Ollama jumps from 17% → 91% (+74 points). Claude Haiku drops from 99.6% → 66.3% without them. An 8B model *with* forge outperforms frontier APIs *without* forge.

2. **Bigger is not better — until 27B+.** Ministral-8B Reasoning (99.3%) outperforms every 14B and 24B model tested on 12GB hardware. Devstral 24B (96.3%), Mistral Small 24B (94.7%), and Ministral-14B Reasoning (95.6%) all fall short. The inflection point is ~27B on 32GB hardware: Qwen3.5 27B and Gemma4 31B both hit 100.0%. Below that threshold, reasoning-oriented fine-tuning at 8B produces better tool-calling discipline than scale alone.

3. **32GB unlocks frontier-matching self-hosted performance.** Gemma4 31B and Qwen3.5 27B achieve 100.0% on dual 5070 Ti (16GB × 2) — the first self-hosted models to match Claude Sonnet/Opus. This was not possible at any VRAM tier prior to this generation of models.

4. **The serving backend is a hidden variable.** The same weights produce 7% on one backend and 83% on another. Backend choice can swing accuracy more than model choice. Any evaluation that doesn't specify the backend may be producing misleading results.

5. **Error recovery is an architectural gap, not a capability gap.** Error recovery scores 0% for *every* model tested — local and frontier — without forge's retry mechanism. No model can self-correct from tool errors without a framework feeding errors back.

6. **Quantization impact is minimal.** Q4_K_M vs Q8_0 on the same model: <1% score difference in most cases. Use Q4_K_M to maximize context window headroom.

7. **Speed varies widely.** Ministral models cluster at 2.5–4.6s per workflow. Qwen3 and Gemma4 dense models are 3–5× slower (11–20s) despite competitive accuracy. MoE variants (Gemma4 A4B, Qwen3.5 A3B) are fast (2–5s) at near-perfect scores — the best speed/accuracy tradeoff at 32GB.

---

## Known Issues

**llama.cpp reasoning budget (builds after April 10 2026):** Gemma 4, Qwen 3.5, and Ministral Reasoning models can hang indefinitely on llama-server due to an unbounded reasoning budget sampler. Add `--reasoning-budget 0` to the server command line. See [BACKEND_SETUP.md](BACKEND_SETUP.md#reasoning-budget-llamacpp-builds-after-april-10-2026) for details.

---

## Setup

See [BACKEND_SETUP.md](BACKEND_SETUP.md) for installation and configuration of each backend.
