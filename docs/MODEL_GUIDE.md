# Model Guide

Which model and backend to use with forge, based on your hardware and goals.

All numbers from forge's eval harness on the v0.6.0 consolidated dataset: 26 scenarios × 50 runs per config, 119,600 rows across 4 rigs. The suite splits into two tiers — **OG-18** (lambda + stateful baseline, 18 scenarios) and **advanced_reasoning** (8 harder scenarios designed as top-tier separators after the per-model sampling-params fix lifted 8B-class to 100% on OG-18). Reporting splits the two tiers in the dashboard's Suite scope. See [EVAL_GUIDE.md](EVAL_GUIDE.md) for full scenario list and methodology.

---

## Difficulty Tiers

Treat the eval suite as roughly three levels of difficulty:

- **Mechanical** — basic_2step, sequential_3step, error_recovery, tool_selection, argument_fidelity, relevance_detection. Every model in the recommended list should handle these. If a model fails here, it's not a candidate.
- **Mid** — sequential_reasoning, conditional_routing, data_gap_recovery, plus all stateful variants. Model reasoning starts to matter; 8B-class without good fine-tuning falls off here.
- **Hard** — `advanced_reasoning`: data_gap_recovery_extended, argument_transformation, inconsistent_api_recovery, grounded_synthesis (lambda + stateful, 8 total). Designed to spread top-tier 8B+ models from each other after sampling-defaults closed the OG-18 gap.

Mechanical + mid are roughly the OG-18 suite. The dashboard's Suite scope (`all` / `og18` / `advanced_reasoning`) cleanly separates them. **Most agentic flows you build in practice land closer to mechanical/mid than to hard** — the hard suite is intentionally adversarial.

---

## The Short Answer

**Best overall on the full suite:** Ministral-3 8B Instruct Q8_0 on llama-server / prompt — 86.5% across all 26 scenarios (91.1% OG-18, 76.0% advanced_reasoning), 4.7s per workflow. Wins overall, wins on hard. **Caveat:** has a binary failure mode — on `data_gap_recovery_extended` (lambda) it scores 0/50 and on the stateful variant 4/50, while everything else in advanced_reasoning is 90%+. Best-in-class or fail loudly.

**Most stable all-around:** Ministral-3 14B Reasoning Q4_K_M on llama-server / native — 81.5% overall, but **nothing at 0%**. #5 overall, #3 on hard. Pick this if your workload mixes execution and reasoning, or you don't yet know which tier your tasks will hit.

**Top 12 configs are all Ministral-3.** Top 10 are all llama-server. The first non-Mistral config — Qwen3 8B Q8_0 LS/prompt — appears at rank 13 (73.1% overall, 94.9% OG-18, 24.0% hard). It's also 7× slower than the Ministral configs ahead of it.

---

## Quick Picks

| Goal | Model | Backend / Mode | Overall | OG-18 | Hard | Speed |
|------|-------|----------------|---------|-------|------|-------|
| **Best overall, hard-task heavy** | Ministral-3 8B Instruct Q8_0 | llama-server / prompt | **86.5%** | 91.1% | **76.0%** | 4.7s |
| **Most stable, no zeros** | Ministral-3 14B Reasoning Q4_K_M | llama-server / native | 81.5% | 94.4% | 52.5% | 5.4s |
| **Best on OG-18 (perfect)** | Ministral-3 14B Instruct Q4_K_M | llama-server / native | 84.7% | **100.0%** | 50.2% | 3.9s |
| **Best on OG-18 at 8B (perfect)** | Ministral-3 8B Instruct Q8_0 | llama-server / native | 83.1% | **100.0%** | 45.0% | 4.1s |
| **Fastest top-12** | Ministral-3 8B Instruct Q4_K_M | llama-server / prompt | 78.0% | 88.9% | 53.5% | **2.8s** |
| **Best non-Mistral** | Qwen3 8B Q8_0 | llama-server / prompt | 73.1% | 94.9% | 24.0% | 33.6s |

Configs above are the slate that actually shipped in v0.6.0; for the full leaderboard see [docs/results/raw/reforged/all.md](results/raw/reforged/all.md) or the [interactive dashboard](results/dashboard.html).

---

## Top 12 — All Ministral-3, All llama-server

| Rank | Config | Overall | OG-18 | Hard | Speed |
|---:|---|---:|---:|---:|---:|
| 1 | ministral-3:8b-instruct-2512-q8_0 LS/P | 86.5% | 91.1% | 76.0% | 4.7s |
| 2 | ministral-3:14b-instruct-2512-q4_K_M LS/N | 84.7% | 100.0% | 50.2% | 3.9s |
| 3 | ministral-3:8b-instruct-2512-q8_0 LS/N | 83.1% | 100.0% | 45.0% | 4.1s |
| 4 | ministral-3:8b-reasoning-2512-q8_0 LS/P | 82.6% | 97.9% | 48.2% | 5.3s |
| 5 | ministral-3:14b-reasoning-2512-q4_K_M LS/N | 81.5% | 94.4% | 52.5% | 5.4s |
| 6 | ministral-3:8b-reasoning-2512-q4_K_M LS/P | 81.5% | 98.0% | 44.2% | 3.5s |
| 7 | ministral-3:8b-reasoning-2512-q8_0 LS/N | 79.6% | 100.0% | 33.8% | 6.4s |
| 8 | ministral-3:8b-reasoning-2512-q4_K_M LS/N | 79.1% | 99.9% | 32.2% | 4.3s |
| 9 | ministral-3:14b-reasoning-2512-q4_K_M LS/P | 79.0% | 94.6% | 44.0% | 4.3s |
| 10 | ministral-3:8b-instruct-2512-q4_K_M LS/P | 78.0% | 88.9% | 53.5% | 2.8s |
| 11 | ministral-3:14b-instruct-2512-q4_K_M OL/N | 76.8% | 100.0% | 24.8% | 6.6s |
| 12 | ministral-3:14b-instruct-2512-q4_K_M LS/P | 75.6% | 97.8% | 25.8% | 3.3s |

LS = llama-server, OL = Ollama. N = native function calling, P = prompt-injected.

**Three patterns worth noting:**

1. **Backend dominates.** 10 of the top 12 run on llama-server. The two Ollama exceptions (rank 11) come in materially behind. Ministral-3 14B Instruct Q4 is the *same model* at #2 on llama-server (84.7%) and #11 on Ollama (76.8%) — that's an 8-point gap from the serving layer alone.

2. **Q4 vs Q8 is largely a wash on OG-18, but Q8 helps on hard.** The top spot is Q8 with a 7-point lead on hard over the same model at Q4 (76.0% vs 53.5% on hard for 8B-instruct LS/P, ranks 1 and 10). At 14B, Q4 is the only quant tested.

3. **Native vs prompt is workload-dependent.** Native wins OG-18 (perfect 100% rows are all native). Prompt wins on hard (every top-3-on-hard config is LS/P). The model is the same; the wire format flips which suite it's stronger at.

---

## OG-18 — When Your Workload Is Closer to Mechanical

Most agentic flows in production look more like the OG-18 scenarios than the advanced_reasoning suite. If your tasks are 2-5 step tool chains with clear hand-offs and recoverable errors, the OG-18 view is the relevant ranking.

### Configs at 100.0% on OG-18

Five configs hit perfect on the OG-18 suite (50 runs × 18 scenarios = 900 trials, all correct):

| Config | OG-18 | Hard | Speed |
|---|---:|---:|---:|
| ministral-3:14b-instruct-2512-q4_K_M LS/N | 100.0% | 50.2% | 3.9s |
| ministral-3:8b-instruct-2512-q8_0 LS/N | 100.0% | 45.0% | 4.1s |
| ministral-3:8b-reasoning-2512-q8_0 LS/N | 100.0% | 33.8% | 6.4s |
| ministral-3:14b-instruct-2512-q4_K_M OL/N | 100.0% | 24.8% | 6.6s |
| ministral-3:8b-reasoning-2512-q4_K_M LS/N | 99.9% | 32.2% | 4.3s |

(99.9% included for the rounding break — 1 scenario miss across 900 trials.)

**Findings on OG-18:**

- **Native FC is the OG-18 winner.** All five 100%-tier configs use native function calling. Prompt-injected variants come in at 91-98%, still high but not perfect.
- **Sampling defaults closed the gap.** Pre-v0.6.0 evals (with hardcoded `temperature=0.7`) capped 8B-class around 95% on OG-18; the per-model sampling-defaults work in v0.6.0 lifted four 8B-class configs to perfect.
- **Ollama can hit 100% too.** Ministral-3 14B Instruct Q4 on Ollama scores 100% on OG-18 — the only Ollama config in the perfect tier, but slower than the LS variants.

If you're confident your workload is OG-18-shaped, any of the five configs above is a defensible pick. The split between them is speed and headroom: 8B Q8 if you want the smallest weights at perfect; 14B if you want reasoning headroom for adjacent harder tasks.

---

## Advanced Reasoning (Hard Suite)

The 8 advanced_reasoning scenarios are designed to spread top-tier models. The previous-generation winners that hit 100% on OG-18 fall to 33-53% on hard. **No self-hosted config tested cleared 80% on hard** — Claude Haiku 4.5 saturates the suite (next section).

### Top 5 on hard

| Config | Hard | OG-18 | Notes |
|---|---:|---:|---|
| ministral-3:8b-instruct-2512-q8_0 LS/P | 76.0% | 91.1% | Hard 0% on `data_gap_recovery_extended`, otherwise 90%+ — binary fail-loud mode |
| ministral-3:8b-instruct-2512-q4_K_M LS/P | 53.5% | 88.9% | 2.8s — fastest in top 12 |
| ministral-3:14b-reasoning-2512-q4_K_M LS/N | 52.5% | 94.4% | **No scenario at 0% — most stable across the suite** |
| ministral-3:14b-instruct-2512-q4_K_M LS/N | 50.2% | 100.0% | OG-18 perfect, hard middling |
| ministral-3:8b-reasoning-2512-q8_0 LS/P | 48.2% | 97.9% | Both modes (P/N) of this config place near the top on hard |

The #1-on-hard config has a hard failure mode worth understanding: on `data_gap_recovery_extended` (lambda) it scores 0/50, and on the stateful variant 4/50. Every other advanced_reasoning scenario for that config sits at 90-100%. If your workload includes data-gap-recovery patterns specifically, the #3 config (14B Reasoning Q4 LS/N, no zeros) is the safer pick at the cost of 23 points on the hard average.

### Why Ministral-3 dominates

Ministral-3 wins the top 12 across both Instruct and Reasoning variants, at 8B and 14B, in both quants, on both native and prompt modes. Two factors stand out:

1. **Tool-calling fine-tuning is more important than parameter count.** Ministral-3 8B Instruct Q4 (rank 10, 78.0%) outscores Qwen3 14B Q4 LS/N (rank 18, 68.9%). Throwing parameters at the problem stops paying after fine-tuning quality.
2. **Speed is competitive.** Top Ministral configs run at 2.8-6.6s per workflow; top Qwen3 configs at 28-35s — a 5-10× gap that compounds at scale.

---

## API Tier (No Local GPU)

API models still serve as the ceiling and the baseline:

| Model | Overall | OG-18 | Hard | Speed |
|-------|---:|---:|---:|---:|
| Claude Haiku 4.5 | ~95% | 99.6% | ~85% | 4.0s |
| Claude Sonnet 4.6 | ~99% | 100.0% | ~95% | 6.5s |
| Claude Opus 4.6 | ~99% | 100.0% | ~95% | 8.5s |

(API tier has full eval coverage on OG-18; selective coverage on advanced_reasoning — not all scenarios re-run on every API model. Numbers above are approximate from partial coverage.)

Haiku for cost-sensitive workloads. Sonnet or Opus for the last few points on hard. The gap between best-self-hosted and Haiku on the hard suite is real (~10 points) and is the current ceiling for self-hosted at 12-16GB VRAM.

---

## The Backend Matters More Than You Think

The same model weights can produce dramatically different results depending on the serving backend. This is a hidden variable that no published benchmark we are aware of controls for.

| Model | Backend / Mode | OG-18 | Notes |
|-------|----------------|---:|-------|
| Ministral-3 14B Instruct Q4 | LS / native | 100.0% | Rank 2 |
| Ministral-3 14B Instruct Q4 | Ollama / native | 100.0% | Rank 11 — same OG-18 score, slower, weaker on hard |
| Ministral-3 14B Instruct Q4 | LS / prompt | 97.8% | Rank 12 |
| Mistral Nemo 12B | LS / prompt | (~76%) | OG-18 only |
| Mistral Nemo 12B | LS / native | (~5%) | Same weights, 70+ point drop |

**Takeaways:**
- **llama-server is the right default for most models** — top 10 are all LS.
- **Native vs prompt depends on the model and the suite.** Native wins OG-18 perfects; prompt wins hard. Test both for your workload.
- **Ollama is convenient but slower and missing the top-tier model selection.** Ministral-3 8B Reasoning, the most accessible reasoning model, is not in the Ollama registry as of this writing — llama-server + GGUF is the only path.
- **Forge's prompt-injection fallback is real.** The gap between native and prompt is often small (1-2%), and prompt wins on the hardest scenarios. If your model has poor native FC support, prompt mode is not a downgrade.

---

## Sampling Parameters

Temperature, `top_p`, `top_k`, `min_p`, `repeat_penalty`, and `presence_penalty` control how the model samples the next token. **Every model family has its own recommended values, and the recommendations differ substantially.** Running all models at a single "default" temperature — which is what most evaluation harnesses do — compares each model outside the sampling zone its authors designed it for.

A few examples of how far recommendations spread:

| Model family | Card-recommended temperature | top_p | top_k |
|---|---|---|---|
| Qwen3 8B/14B (thinking) | 0.6 | 0.95 | 20 |
| Qwen3.5 / 3.6 (thinking, general) | **1.0** | 0.95 | 20 |
| Qwen3-Coder Instruct | 0.7 | 0.8 | 20 |
| Ministral-3 Instruct | 0.05 | — | — |
| Granite 4.0 | 0.0 (greedy) | 1.0 | 0 |

Running Ministral-3 Instruct at the "standard" 0.7 temperature — instead of the card-recommended 0.05 — is a measurable handicap. The v0.6.0 sampling-defaults work specifically targeted this gap; eval results jumped 3-8 points on most 8B-class configs after the fix.

### How forge handles it

Forge ships a per-model recommendations map at `forge.clients.sampling_defaults`. Each entry is sourced directly from the model's HuggingFace card (or, when the vendor has not published sampling on the card, from a secondary source that cites the vendor — Granite 4.0 is the current example), with the source URL as an inline comment. Values are verified one entry at a time — no best-effort or extrapolated entries.

```python
from forge.clients import LlamafileClient

# Managed mode — opt in to recommended defaults via constructor flag.
# For local-server backends, the GGUF / llamafile path *is* the model
# identity — its filename stem is the lookup key.
client = LlamafileClient(
    gguf_path="path/to/Ministral-3-8B-Instruct-2512-Q8_0.gguf",
    mode="native",
    recommended_sampling=True,
)
```

The flag is opt-in. Default behavior (`recommended_sampling=False`) leaves sampling to backend defaults; if forge has opinions about the model, it logs a one-shot INFO message pointing the caller at the flag. With `recommended_sampling=True`, an unknown model raises `UnsupportedModelError` — falling through to backend defaults silently would defeat the explicit opt-in.

Caller's explicit non-None sampling kwargs win field-by-field over the map:

```python
client = LlamafileClient(
    gguf_path="path/to/Ministral-3-8B-Instruct-2512-Q8_0.gguf",
    mode="native",
    recommended_sampling=True,
    temperature=0.1,  # overrides the map's 0.05; other map fields still apply
)
```

For programmatic introspection without triggering policy, `forge.clients.get_sampling_defaults(model)` is a pure lookup — returns the map value (a fresh copy) or `{}` for unknown models. No logging, no raising. Pass either an Ollama-style key (`"qwen3:8b-q8_0"`), a GGUF stem (`"Qwen3-8B-Q8_0"`), or a llamafile stem (`"Mistral-Nemo-Instruct-2407.Q4_K_M"`) — the map is keyed on all three identity forms.

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
    gguf_path="path/to/Qwen3.5-27B-Q4_K_M.gguf",
    mode="native",
    recommended_sampling=True,
    temperature=0.6,
    presence_penalty=0.0,
)
```

For programmatic access to the map without triggering policy:

```python
from forge.clients import get_sampling_defaults

defaults = get_sampling_defaults("Qwen3.5-27B-Q4_K_M")  # GGUF-stem lookup; fresh dict, safe to mutate
defaults["temperature"] = 0.6
client = LlamafileClient(gguf_path="path/to/Qwen3.5-27B-Q4_K_M.gguf", mode="native", **defaults)
```

For fully manual control, pass sampling kwargs directly and skip the helpers.

---

## Models to Avoid

Configs that score below 60% overall and aren't recommended for production agentic workloads:

| Model | Best Score | Why |
|-------|-----------|-----|
| Llama 3.1 8B | ~54% | Tool-call reliability falls off on stateful + hard scenarios |
| Mistral 7B v0.3 | ~46% | Older release, no formal sampling guidance, weak on multi-step workflows |
| Granite 4.0 h-micro / h-tiny | 26-65% | Hybrid architecture leaves reliability on the table even with full guardrails |

These models work but fail too often for production-grade agentic workflows. Forge's guardrails still help (Granite 4.0 lifts from low single digits to 65% with the full stack), but the floor isn't high enough to ship.

---

## Key Findings

1. **Guardrails matter more than model size.** Forge's guardrail stack adds 10-79 points depending on the model. The same 8B Ministral-3 Instruct that hits 86.5% with reforged guardrails drops to single digits on bare. An 8B model *with* forge outperforms most frontier APIs *without* forge.

2. **Tool-calling fine-tuning beats parameter count.** Ministral-3 8B Instruct outscores Qwen3 14B and Mistral Nemo 12B at the same backend. The Ministral-3 family was trained explicitly for agentic workflows; that fine-tuning quality carries further than 6B more parameters of general capability.

3. **The serving backend is a hidden variable.** Same weights, different backend, scores 70+ points apart. Backend choice can swing accuracy more than model choice. Any evaluation that doesn't specify the backend may be producing misleading results.

4. **Sampling defaults are a real lever.** Pre-v0.6.0, hardcoded `temperature=0.7` left ~3-8 points on the table for most 8B-class configs. Per-model card-recommended sampling (forge's `recommended_sampling=True`) closes that gap and lifts four 8B-class configs to perfect on OG-18.

5. **Error recovery is an architectural gap, not a capability gap.** Error recovery scores 0% for *every* model tested — local and frontier — without forge's retry mechanism. No model can self-correct from tool errors without a framework feeding errors back.

6. **Quantization impact is workload-dependent.** Q4_K_M vs Q8_0 on the same model: <2% on OG-18 in most cases, but Q8 helps on hard (the top-1 config gains 7-23 points on hard at Q8). Use Q4 for context window headroom; Q8 if your workload leans hard.

7. **Speed varies widely.** Top Ministral configs cluster at 2.8-6.6s per workflow. Top Qwen3 configs are 28-35s — a 5-10× gap that compounds at scale.

---

## Known Issues

**llama.cpp reasoning budget (builds after April 10 2026):** Gemma 4, Qwen 3.5, and Ministral Reasoning models can hang indefinitely on llama-server due to an unbounded reasoning budget sampler. Add `--reasoning-budget 0` to the server command line. See [BACKEND_SETUP.md](BACKEND_SETUP.md#reasoning-budget-llamacpp-builds-after-april-10-2026) for details.

---

## Setup

See [BACKEND_SETUP.md](BACKEND_SETUP.md) for installation and configuration of each backend.
