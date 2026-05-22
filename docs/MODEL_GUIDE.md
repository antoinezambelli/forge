# Model Guide

How to think about model choice with forge. **Current leaderboard, scores, and per-scenario breakdowns live in the [dashboard](results/dashboard.html) and the [markdown views](results/raw/)** — this doc is opinions and rationale, not numbers.

For the full list of models forge knows about (with sampling params + source links + eval status), see [MODEL_REGISTRY.md](MODEL_REGISTRY.md).

---

## Suite tiers

The 26-scenario suite splits into two tiers:

- **OG-18** — 18 scenarios covering plumbing, model quality, compaction, and stateful variants. The "is this model actually viable for agentic work" tier. Saturates on top-tier 8B+ configs.
- **advanced_reasoning** — 8 harder scenarios designed as top-tier separators after the per-model sampling-defaults fix lifted 8B-class to 100% on OG-18. **Most agentic flows you build in production land closer to OG-18 than to advanced_reasoning** — the hard suite is intentionally adversarial.

The dashboard's Suite scope (`all` / `og18` / `advanced_reasoning`) cleanly separates them. Most picking should be done on OG-18 unless your workload is known to land in the hard zone.

---

## How to pick

Three questions, in order:

1. **What's your VRAM budget?** Determines model size. See [BACKEND_SETUP.md](BACKEND_SETUP.md) for boot commands and the live [dashboard](results/dashboard.html) for which configs at your size are actually viable. Rough orientation: 8GB → 8B Q4; 12GB → 8B Q8 / 14B Q4; 16GB+ → 14B Q4 with headroom.

2. **What's your workload shape?** If your tasks are 2-5 step tool chains with clear hand-offs (most real workflows), filter the dashboard to OG-18. If your tasks involve adversarial transformation / multi-hop reasoning, look at the advanced_reasoning tier — the picture changes meaningfully there.

3. **How important is reliability vs raw speed?** Models with "no scenario at 0%" stability matter more for production than peak score. The dashboard's per-scenario columns let you spot binary-failure-mode configs (best-in-class on most scenarios, then 0% on one) — those are operationally riskier than slightly-lower-scoring stable configs.

---

## The backend matters more than you think

The same model weights produce dramatically different results depending on the serving backend. This is a hidden variable that no published benchmark we are aware of controls for.

- **llama-server is the right default for most models.** Almost every top-tier reforged config in forge's eval runs on llama-server, not Ollama.
- **Native vs prompt-injected is per-family, not workload-driven.** v0.7.0 data shows the OG-18 / hard split doesn't predict the winner: Ministral Instruct sees prompt wins big on hard (+20 to +34pts at 8B); Ministral Reasoning sees the opposite (native wins hard by ~13pts); Qwen3 mostly prefers prompt; Gemma-4-E4B mostly prefers native. Sensitivity is real and substantial — test both modes for your specific model.
- **Ollama is convenient but slower and often missing the top-tier models** (e.g. Ministral-3 Reasoning isn't in the Ollama registry — llama-server + GGUF is the only path).
- **Forge's prompt-injection fallback is real.** The gap between native and prompt is often small (1-2pts) and prompt sometimes wins on hard scenarios. If your model has poor native FC support, prompt mode is not a downgrade.

See the [native-vs-prompt markdown view](results/raw/native-vs-prompt.md) for the paired comparison on llama-server.

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
| Granite 4.x | 0.0 (greedy) | 1.0 | 0 |

Running Ministral-3 Instruct at the "standard" 0.7 temperature — instead of the card-recommended 0.05 — is a measurable handicap. v0.6.0's sampling-defaults work specifically targeted this gap; eval results jumped 3-8 points on most 8B-class configs after the fix.

### How forge handles it

Forge ships a per-model recommendations map at `forge.clients.sampling_defaults`. Each entry is sourced directly from the model's HuggingFace card (or, when the vendor has not published sampling on the card, from a secondary source that cites the vendor — Granite is the current example), with the source URL as an inline comment. Values are verified one entry at a time — no best-effort or extrapolated entries.

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

The flag is opt-in:
- **Off** (default) — leaves sampling to backend defaults; if forge has opinions about the model, it logs a one-shot INFO message pointing the caller at the flag.
- **On, model known** — values applied; caller's explicit non-None kwargs win field-by-field.
- **On, model unknown** — raises `UnsupportedModelError`. Falling through to backend defaults silently would defeat the explicit opt-in.

For full rationale see [ADR-014](decisions/014-recommended-sampling-opt-in.md). For the complete list of supported models, citation links, and per-model values, see [MODEL_REGISTRY.md](MODEL_REGISTRY.md).

### Proxy mode

The proxy does **not** consult the recommendations map. It plumbs whatever sampling params the inbound request body carries (OpenAI-compatible fields: `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `seed`) through to the backend on a per-call basis. The proxy's pre-built client is treated as a "blank slate" — body fields are the only sampling source.

To get card-recommended sampling in proxy mode, the calling client looks up `forge.clients.get_sampling_defaults(model)` and includes the values in the request body.

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

### Profile choices

When a card gives multiple profiles (e.g. Qwen3.5 has separate "general" vs "precise coding" columns), forge uses the **general-tasks thinking-mode** profile. Consumers that know their workload better (code-focused harnesses, for instance) should override explicitly.

---

## API tier (no local GPU)

Forge supports Anthropic models (`claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-6`) as a frontier baseline. Numbers in any v0.7.0 doc that cites API-tier scores are from the v0.6.0 dataset — the Anthropic ablation was not re-run in v0.7.0 because the cost is non-trivial (~$272 for the full 11,700-row matrix). Backend support is unchanged; the v0.6.0 numbers are the canonical reference.

The headline finding from the v0.6.0 Anthropic ablation worth carrying forward: **forge's guardrails lift frontier models too**, not just self-hosted. Sonnet bare → reforged was 85% → 98% overall; Haiku bare → reforged was 46% → 94%. This isn't a "guardrails are only for weak models" story.

---

## Known issues

**llama.cpp reasoning budget (builds after April 10 2026):** Gemma 4, Qwen 3.5, and Ministral Reasoning models can hang indefinitely on llama-server due to an unbounded reasoning budget sampler. Add `--reasoning-budget 0` to the server command line. See [BACKEND_SETUP.md](BACKEND_SETUP.md#gotcha-reasoning-budget-on-recent-llamacpp-builds) for details.

---

## Where the data lives

- **Live dashboard:** [`docs/results/dashboard.html`](results/dashboard.html) — interactive, sortable, filterable
- **Markdown leaderboards** (regenerate alongside the dashboard): [`docs/results/raw/`](results/raw/) — reforged-only, by-family, by-backend, reforged-vs-bare, native-vs-prompt
- **Raw JSONL:** `eval_results_v0.7.0.jsonl` at the repo root (LFS). Prior versions kept as `eval_results_vX.Y.Z.jsonl` for reproducibility.
- **Per-model sampling defaults:** [`src/forge/clients/sampling_defaults.py`](../src/forge/clients/sampling_defaults.py) — authoritative source
- **Model status by eval-suite:** [MODEL_REGISTRY.md](MODEL_REGISTRY.md)
