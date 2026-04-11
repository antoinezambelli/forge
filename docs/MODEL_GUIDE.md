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

## Key Findings

1. **Guardrails matter more than model size.** Forge's guardrail stack adds 10–79% accuracy depending on the model. Ministral-8B Instruct on Ollama jumps from 17% → 91% (+74 points). Claude Haiku drops from 99.6% → 66.3% without them. An 8B model *with* forge outperforms frontier APIs *without* forge.

2. **Bigger is not better — until 27B+.** Ministral-8B Reasoning (99.3%) outperforms every 14B and 24B model tested on 12GB hardware. Devstral 24B (96.3%), Mistral Small 24B (94.7%), and Ministral-14B Reasoning (95.6%) all fall short. The inflection point is ~27B on 32GB hardware: Qwen3.5 27B and Gemma4 31B both hit 100.0%. Below that threshold, reasoning-oriented fine-tuning at 8B produces better tool-calling discipline than scale alone.

3. **32GB unlocks frontier-matching self-hosted performance.** Gemma4 31B and Qwen3.5 27B achieve 100.0% on dual 5070 Ti (16GB × 2) — the first self-hosted models to match Claude Sonnet/Opus. This was not possible at any VRAM tier prior to this generation of models.

4. **The serving backend is a hidden variable.** The same weights produce 7% on one backend and 83% on another. Backend choice can swing accuracy more than model choice. Any evaluation that doesn't specify the backend may be producing misleading results.

5. **Error recovery is an architectural gap, not a capability gap.** Error recovery scores 0% for *every* model tested — local and frontier — without forge's retry mechanism. No model can self-correct from tool errors without a framework feeding errors back.

6. **Quantization impact is minimal.** Q4_K_M vs Q8_0 on the same model: <1% score difference in most cases. Use Q4_K_M to maximize context window headroom.

7. **Speed varies widely.** Ministral models cluster at 2.5–4.6s per workflow. Qwen3 and Gemma4 dense models are 3–5× slower (11–20s) despite competitive accuracy. MoE variants (Gemma4 A4B, Qwen3.5 A3B) are fast (2–5s) at near-perfect scores — the best speed/accuracy tradeoff at 32GB.

---

## Setup

See [BACKEND_SETUP.md](BACKEND_SETUP.md) for installation and configuration of each backend.
