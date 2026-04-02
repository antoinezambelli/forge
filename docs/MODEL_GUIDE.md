# Model Guide

Which model and backend to use with forge, based on your hardware and goals.

All numbers from forge's eval harness: 18 scenarios (9 lambda + 9 stateful) × 50 runs per config, measured 2026-03-12. Full guardrail stack ("reforged") unless noted. See [EVAL_GUIDE.md](EVAL_GUIDE.md) for full scenario list (22 total including compaction chain) and methodology.

---

## The Short Answer

**Ministral-8B Reasoning Q4_K_M on llama-server (native FC)** — 99.3% score, 3.7s per workflow, 4.8 GB weights, runs on any 8GB+ GPU.

This is an 8-billion-parameter model running on consumer hardware, scoring within 1 percentage point of frontier APIs (Claude Sonnet/Opus at 100%). It outperforms every 14B model tested, and it outperforms frontier APIs *without* forge guardrails (the best a consumer can achieve through API alone).

If you want the simplest possible setup and don't need the absolute best score, **Ministral-8B Instruct Q4_K_M on Ollama** gets you 91–95% with zero server management.

---

## Quick Picks

| Goal | Model | Backend | Score | Speed |
|------|-------|---------|-------|-------|
| **Maximum reliability** | Claude Sonnet 4.6 | Anthropic API | 100.0% | 6.5s |
| **Best self-hosted** | Ministral-8B Reasoning Q4_K_M | llama-server (native) | 99.3% | 3.7s |
| **Easiest setup** | Ministral-8B Instruct Q4_K_M | Ollama | 91.2% | 4.2s |
| **No local GPU** | Claude Haiku 4.5 | Anthropic API | 99.6% | 4.0s |
| **Single binary, no deps** | Mistral Nemo 12B Q4_K_M | Llamafile | 82.6% | 4.2s |

---

## By VRAM Budget

### 8GB VRAM (recommended)

8B-class models at Q4_K_M are the sweet spot — they fit comfortably in 8GB VRAM and deliver the best scores in the eval suite. Having more VRAM doesn't mean you should use a bigger model (see [Key Findings](#key-findings)).

| Model | Backend | Mode | Score | Notes |
|-------|---------|------|-------|-------|
| Ministral-8B Reasoning Q4_K_M | llama-server | Native | 99.3% | **Best overall** |
| Ministral-8B Reasoning Q8_0 | llama-server | Native | 99.2% | Higher precision, ~same score |
| Qwen3-8B Q8_0 | llama-server | Prompt | 95.7% | Good alternative, slower (17.8s) |
| Ministral-8B Instruct Q8_0 | Ollama | Native | 94.6% | Simpler setup |
| Ministral-8B Instruct Q4_K_M | Ollama | Native | 91.2% | Smallest footprint |

### 12–16GB VRAM

You *can* run 14B models, but the data shows 8B Reasoning outperforms 14B variants of the same family. The extra VRAM is better spent on larger context windows (via `-c` flag) than bigger models.

| Model | Backend | Mode | Score | Notes |
|-------|---------|------|-------|-------|
| Ministral-8B Reasoning Q4_K_M | llama-server | Native | 99.3% | Still the best choice |
| Ministral-14B Instruct Q4_K_M | llama-server | Native | 98.8% | Marginal, slower to load |
| Qwen3-14B Q4_K_M | Ollama | Native | 96.3% | Below 8B Reasoning, 5× slower |

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

**Takeaways:**
- **llama-server native is the best backend for most models** — but not all. Qwen3 and Nemo perform *worse* with native FC on llama-server than with prompt-injected or Ollama.
- **Always test your specific model/backend combination.** Don't assume native FC is better than prompt-injected — it depends on the model's training and the backend's template handling.
- **Forge's prompt-injection fallback is effective.** The gap between native and prompt-injected is often small (1–2%), and sometimes prompt wins.
- **Model availability varies by backend.** llama-server (via GGUF files from HuggingFace) has the widest model selection — any model with a GGUF release works, at any quantization. Ollama's registry is convenient but lags behind and is missing key models (including Ministral-8B Reasoning, the top self-hosted pick). Llamafile has the most limited selection and tends to trail further behind new releases. If you want access to the latest models, start with llama-server + GGUF.

---

## Key Findings

1. **Guardrails matter more than model size.** Forge's guardrail stack adds 10–55% accuracy depending on the model. Claude Haiku drops from 99.6% → 43.8% without them. An 8B model *with* forge outperforms frontier APIs *without* forge.

2. **Bigger is not better.** Ministral-8B Reasoning (99.3%) outperforms Ministral-14B Reasoning (95.7%) and Ministral-14B Instruct (98.8%). Qwen3-8B outperforms Qwen3-14B by up to 5.5% depending on backend. Reasoning-oriented fine-tuning at 8B produces better tool-calling discipline than scale alone at 14B.

3. **The serving backend is a hidden variable.** The same weights produce 7% on one backend and 83% on another. Backend choice can swing accuracy more than model choice. Any evaluation that doesn't specify the backend may be producing misleading results.

4. **Error recovery is an architectural gap, not a capability gap.** Error recovery scores 0% for *every* model tested — local and frontier — without forge's retry mechanism. No model can self-correct from tool errors without a framework feeding errors back.

5. **Quantization impact is minimal.** Q4_K_M vs Q8_0 on the same model: <1% score difference in most cases. Use Q4_K_M.

6. **Speed varies widely.** Ministral models cluster at 2.5–4.6s per workflow. Qwen3 is 4–5× slower (14–20s) despite competitive accuracy.

---

## Setup

See [BACKEND_SETUP.md](BACKEND_SETUP.md) for installation and configuration of each backend.
