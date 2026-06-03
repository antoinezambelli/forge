# Model Registry

Every model forge knows about, classified by eval-suite status.

## Status meanings

- **Current** — in the published eval. The dashboard folds multiple eval *generations* into one view (the v0.7.0 8–14B lineup, plus the v0.7.4 32GB tier); runs not yet re-swept against the latest code — e.g. the Anthropic ablation — are carried forward and superscript-tagged. Numbers in [`docs/results/`](results/) and the [dashboard](results/dashboard.html).
- **Retired** — appeared in a prior eval suite, cut from the current one. Either too weak (bare scores below the threshold for informative comparison) or superseded by a newer family member. Sampling defaults retained for backward compatibility.
- **Unpublished** — sampling defaults are present, but no eval numbers have been published. Forge will work with these models; performance is undocumented.

Sampling values are sourced from the model's HuggingFace card unless noted. Values are verified one model at a time — see [`src/forge/clients/sampling_defaults.py`](../src/forge/clients/sampling_defaults.py) for the authoritative map.

---

## Current

| Model | Quants | temp | top_p | top_k | min_p | repeat_penalty | presence_penalty | Source |
|---|---|---|---|---|---|---|---|---|
| Ministral-3 8B Instruct 2512 | Q4_K_M, Q8_0 | 0.05¹ | — | — | — | — | — | [HF](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) |
| Ministral-3 14B Instruct 2512 | Q4_K_M | 0.05¹ | — | — | — | — | — | [HF](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) |
| Ministral-3 8B Reasoning 2512 | Q4_K_M, Q8_0 | 0.7 | —² | — | — | — | — | [HF](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512) |
| Ministral-3 14B Reasoning 2512 | Q4_K_M | 1.0 | —² | — | — | — | — | [HF](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512) |
| Qwen3 8B | Q4_K_M, Q8_0 | 0.6 | 0.95 | 20 | 0.0 | — | — | [HF](https://huggingface.co/Qwen/Qwen3-8B) |
| Qwen3 14B | Q4_K_M | 0.6 | 0.95 | 20 | 0.0 | — | — | [HF](https://huggingface.co/Qwen/Qwen3-14B) |
| Granite 4.1 8B | Q4_K_M, Q8_0 | 0.0³ | 1.0 | 0 | — | — | — | (IBM convention, unconfirmed) |
| Gemma-4 E4B-it | Q4_K_M, Q8_0 | 1.0 | 0.95 | 64 | — | — | — | [HF](https://huggingface.co/google/gemma-4-e4b-it) |
| Phi-4 | Q4_K_M | — | — | — | — | — | — | (no formal recommendation⁴) |
| Mistral Small 3.2 24B Instruct 2506 | Q4_K_M | 0.15 | — | — | — | — | — | [HF](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) |
| Qwen3.5 27B | Q4_K_M | 1.0 | 0.95 | 20 | 0.0 | — | 1.5 | [HF](https://huggingface.co/Qwen/Qwen3.5-27B) |
| Qwen3.5 35B-A3B | Q4_K_M | 1.0 | 0.95 | 20 | 0.0 | — | 1.5 | [HF](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |
| Qwen3.6 27B | Q4_K_M | 1.0 | 0.95 | 20 | 0.0 | — | 0.0⁶ | [HF](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Qwen3.6 35B-A3B UD | Q4_K_M | 1.0 | 0.95 | 20 | 0.0 | — | 1.5 | [HF](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Nemotron-3 Nano 30B-A3B | Q4_K_M | 0.6 | 0.95 | — | — | — | —⁷ | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) |
| Claude Haiku 4.5⁵ | — | — | — | — | — | — | — | (SDK-managed) |
| Claude Sonnet 4.6⁵ | — | — | — | — | — | — | — | (SDK-managed) |
| Claude Opus 4.6⁵ | — | — | — | — | — | — | — | (SDK-managed) |

¹ Ministral-3 Instruct cards say "temperature below 0.1 for production"; 0.05 picked within that range.
² Ministral-3 Reasoning cards show `top_p=0.95` in code examples but do NOT include it in the formal "Recommended Settings" section. Add explicitly if you want to follow the examples.
³ Granite 4.1 sampling mirrors the Granite 4.0 IBM convention (greedy decoding); marked unconfirmed pending IBM publication for the 4.1 family specifically.
⁴ Phi-4: no formal sampling recommendation from any official source (Microsoft HF card, model docs). Falls through to backend defaults.
⁵ **Claude numbers are carried forward from the v0.6.0 dataset** — gen 1 on the dashboard, superscript-tagged. The Anthropic ablation has not been re-run since, owing to cost (~$272 for the full 11,700-row matrix). Backend support is unchanged; numbers are stable to within tool-error-channel sensitivity (small).
⁶ Qwen3.6 27B (dense) deliberately diverges from its A3B siblings: its card drops the `presence_penalty=1.5` the MoE variants recommend, so forge sends `0.0` (no penalty).
⁷ Nemotron-3 Nano: the card splits sampling into a Reasoning preset (T=1.0, top_p=1.0) and a Tool-calling preset (T=0.6, top_p=0.95); the tool-calling preset is used here, with thinking enabled via `chat_template_kwargs`.

---

## Retired

| Model | Quants | temp | Why retired | Source |
|---|---|---|---|---|
| Llama 3.1 8B Instruct | Q4_K_M, Q8_0 | — (no formal rec) | Bare scores below threshold; superseded | [Meta HF (silent on sampling)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Mistral 7B Instruct v0.3 | Q4_K_M, Q8_0 | — (no formal rec) | Bare scores below threshold; older release, weak on multi-step | [HF (no recommended-settings section)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| Mistral Nemo 12B Instruct 2407 | Q4_K_M | 0.3 | Bare scores below threshold; superseded by Ministral-3 family | [HF](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) |
| Granite 4.0 h-micro | Q4_K_M | 0.0³ | Bare scores below threshold; superseded by Granite 4.1 8B | [Unsloth tutorial (cites IBM)](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0) |
| Granite 4.0 h-tiny | Q4_K_M | 0.0³ | Bare scores below threshold; superseded by Granite 4.1 8B | [Unsloth tutorial (cites IBM)](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0) |

³ Granite 4.0 sampling is greedy decoding (T=0); `top_p=1.0` and `top_k=0` are mathematical no-ops at T=0 but kept explicit to match the source recommendation.

---

## Unpublished

Sampling params staged, no eval data published. Forge supports these — performance is undocumented.

| Model | Quants | temp | top_p | top_k | min_p | Other | Source |
|---|---|---|---|---|---|---|---|
| Qwen3 4B Instruct 2507 | Q4_K_M | 0.7 | 0.8 | 20 | 0.0 | — | [HF](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| Qwen3 4B Thinking 2507 | Q4_K_M | 0.6 | 0.95 | 20 | 0.0 | — | [HF](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) |
| Qwen3.5 122B-A10B | Q4_K_M | 0.7 | 0.8 | 20 | — | balanced (instruct) preset; thinking mode is separate⁸ | [HF](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) |
| Qwen3-Coder 30B-A3B Instruct | Q4_K_M | 0.7 | 0.8 | 20 | — | repeat_penalty=1.05 | [HF](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| Qwen3-Coder-Next 80B-A3B | Q4_K_M | 1.0 | 0.95 | 40 | — | coder fine-tune over Qwen3-Next | [HF](https://huggingface.co/Qwen/Qwen3-Coder-Next) |
| Qwen3-Next 80B-A3B Instruct | Q4_K_M | 0.7 | 0.8 | 20 | 0.0 | hybrid-attention MoE; thinking-mode profile | [HF](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) |
| Devstral Small 2 24B Instruct 2512 | Q4_K_M, Q8_0 | 0.15 | — | — | — | — | [HF](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) |
| Mistral Small 4 119B 2603 UD | Q4_K_M | 0.7 | — | — | — | `chat_template_kwargs.reasoning_effort="high"`⁹ | [HF](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) |
| Gemma-4 26B-A4B-it | Q4_K_M (UD), Q8_0 | 1.0 | 0.95 | 64 | — | — | [HF](https://huggingface.co/google/gemma-4-26b-a4b-it) |
| Gemma-4 31B-it | Q4_K_M | 1.0 | 0.95 | 64 | — | — | [HF](https://huggingface.co/google/gemma-4-31b-it) |
| NVIDIA Nemotron-3 Super 120B-A12B UD | Q4_K_M | 1.0 | 0.95 | — | — | thinking on; low_effort, force_nonempty_content via `chat_template_kwargs`¹⁰ | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) |
| gpt-oss 120b | Q4_K_M | 1.0 | 1.0 | 0 | 0.0 | `chat_template_kwargs.reasoning_effort="medium"`; **do not** set repeat/presence penalties¹¹ | [HF](https://huggingface.co/openai/gpt-oss-120b) |

⁸ Qwen3.5-122B-A10B: values are the card's "balanced" instruct preset (T=0.7 / top_p=0.8 / top_k=20). Thinking mode uses different sampling and is tracked as a separate configuration.
⁹ Mistral-Small-4 card gives T=0.7 for `reasoning_effort="high"` and "between 0.0 and 0.7" for `reasoning_effort="none"` (task-dependent). High-effort profile picked as the safer default; top_p/top_k not specified on the card.
¹⁰ Nemotron-3 Super: card recommends T=1.0 / top_p=0.95 across all tasks. `low_effort` reins in over-thinking; `force_nonempty_content` so the model emits something substantive instead of empty `<think>` blocks.
¹¹ gpt-oss-120b: per the llama.cpp maintainer guide, **explicitly do not set `repeat_penalty` or `presence_penalty`** — they degrade output. Registry omission == None == field omitted from request body. `reasoning_effort` adjustable via `chat_template_kwargs`; "medium" is the current default, bring down if overthinking observed.

---

## Identity keys

Each model is keyed in `sampling_defaults.py` under all the identity forms a caller might use:
- Ollama-style strings (e.g. `qwen3:8b-q8_0`) — for `OllamaClient`
- GGUF file stems (e.g. `Qwen3-8B-Q8_0`) — for `LlamafileClient` with a `.gguf`
- Llamafile binary stems (e.g. `Mistral-Nemo-Instruct-2407.Q4_K_M`) — for `LlamafileClient` with a `.llamafile`

All forms point at independent rows starting as copies of the same HF card values, so vendor-specific guidance (e.g. an Ollama modelfile that diverges from the HF card) can be encoded without forcing alignment.

---

## Models intentionally absent from sampling defaults

Two models in the Retired tier have **no entry** in the sampling map at all:

- **Llama 3.1 8B Instruct** — Meta's HF card, llama.com/docs, and llama-recipes are all silent on sampling
- **Mistral 7B Instruct v0.3** — HF card has no "recommended settings" section; code examples use `temperature=0.0` (greedy) but explicitly note it's demo-only

These fall through to backend defaults. **Phi-4 (Current)** is in the same category — no formal recommendation — but kept in the Current tier because it's in the v0.7.0 eval and supported as a viable model.
