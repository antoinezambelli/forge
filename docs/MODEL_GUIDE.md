# Model Guide

Which model and backend to use with forge, based on your hardware and goals.

All numbers from forge's eval harness: 29 scenarios × 50 runs per config, measured 2026-03-12. Full guardrail stack ("reforged") unless noted. See [EVAL_GUIDE.md](EVAL_GUIDE.md) for methodology.

---

## Quick Picks

| Goal | Model | Backend | Score | Speed |
|------|-------|---------|-------|-------|
| **Maximum reliability** | Claude Sonnet 4.6 | Anthropic API | 100.0% | 6.5s |
| **Best self-hosted** | Ministral-8B Reasoning Q4_K_M | llama-server (native) | 99.3% | 3.7s |
| **Best 16GB VRAM** | Ministral-14B Instruct Q4_K_M | llama-server (native) | 98.8% | 3.5s |
| **Easiest setup** | Qwen3-14B Q4_K_M | Ollama | 96.3% | 19.6s |
| **Budget 8GB VRAM** | Ministral-8B Instruct Q4_K_M | Ollama | 91.2% | 4.2s |
| **Single binary, no deps** | Mistral Nemo 12B Q4_K_M | Llamafile | 82.6% | 4.2s |

---

## By VRAM Budget

### 8GB VRAM

You can run 8B-class models at Q4_K_M, or 8B at Q8_0 if VRAM is fully available.

| Model | Backend | Mode | Score | Notes |
|-------|---------|------|-------|-------|
| Ministral-8B Reasoning Q4_K_M | llama-server | Native | 99.3% | Best overall value |
| Ministral-8B Reasoning Q8_0 | llama-server | Native | 99.2% | Slightly higher precision, ~same score |
| Ministral-8B Instruct Q8_0 | Ollama | Native | 94.6% | Simpler setup |
| Ministral-8B Instruct Q4_K_M | Ollama | Native | 91.2% | Smallest footprint |
| Qwen3-8B Q8_0 | llama-server | Prompt | 95.7% | Good alternative, slower (17.8s) |

**Recommendation:** Ministral-8B Reasoning on llama-server native. At 99.3% it rivals frontier models.

### 12–16GB VRAM

Room for 14B-class models at Q4_K_M (~8.2GB weights + KV cache headroom).

| Model | Backend | Mode | Score | Notes |
|-------|---------|------|-------|-------|
| Ministral-14B Instruct Q4_K_M | llama-server | Native | 98.8% | Fastest 14B option (3.5s) |
| Qwen3-14B Q4_K_M | Ollama | Native | 96.3% | Easiest setup, slower (19.6s) |
| Ministral-14B Instruct Q4_K_M | Ollama | Native | 96.1% | Good middle ground (4.5s) |

**Recommendation:** Ministral-14B Instruct on llama-server native for speed and accuracy. Qwen3-14B on Ollama if you want zero-config setup.

### API (no local GPU)

| Model | Score | Speed | Notes |
|-------|-------|-------|-------|
| Claude Sonnet 4.6 | 100.0% | 6.5s | Perfect score, best choice |
| Claude Opus 4.6 | 100.0% | 8.5s | Same score, slower |
| Claude Haiku 4.5 | 99.6% | 4.0s | Fastest, near-perfect |

**Recommendation:** Haiku for cost-sensitive workloads (99.6% at 4.0s). Sonnet or Opus if you need the last 0.4%.

---

## Models to Avoid

| Model | Best Score | Why |
|-------|-----------|-----|
| Llama 3.1 8B | 68.8% | Consistently low accuracy across all backends |
| Mistral v0.3 7B | 64.7% | Highly variable, poor tool-call reliability |

These models work but fail too often for production-grade agentic workflows.

---

## Backend Comparison

Same model, different backends — how much does the backend matter?

| Model | llama-server (native) | Ollama (native) | Llamafile (prompt) | Delta |
|-------|-----------------------|-----------------|--------------------|-------|
| Ministral-14B Instruct | 98.8% | 96.1% | — | 2.7% |
| Ministral-8B Instruct Q8_0 | 97.4% | 94.6% | — | 2.8% |
| Mistral Nemo 12B | — | — | 82.6% | — |

**Takeaways:**
- **llama-server vs Ollama:** 2–4% gap on the same model. llama-server wins on raw performance; Ollama wins on ease of use.
- **Native vs prompt-injected FC:** ~1–2% gap. Smaller than expected — forge's prompt-injection fallback is effective.
- **Llamafile:** Viable for portability but limited to prompt-injected mode. Best with Mistral Nemo.

---

## Key Findings

1. **Forge guardrails matter.** The full guardrail stack adds 10–55% accuracy depending on the model. Claude Haiku drops from 99.6% → 43.8% without them.
2. **Ministral is the sweet spot.** The 8B Reasoning variant at 99.3% rivals frontier API models while running on consumer hardware.
3. **Backend choice is secondary.** The model matters far more than the backend — the same model scores within 2–4% across backends.
4. **Speed varies widely.** Ministral models cluster at 2.5–4.6s per workflow. Qwen3 is 4–5× slower (14–20s) despite competitive accuracy.
5. **Quantization impact is minimal.** Q4_K_M vs Q8_0 on the same model: <1% score difference in most cases.

---

## Setup

See [BACKEND_SETUP.md](BACKEND_SETUP.md) for installation and configuration of each backend.
