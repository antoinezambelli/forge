"""Per-model recommended sampling defaults sourced from HF model cards.

Two functions live here, separating lookup from policy:

- ``get_sampling_defaults(model)``: pure lookup. Returns the map value (a
  fresh copy) or ``{}`` for unknown models. No logging, no raising.
- ``apply_sampling_defaults(model, *, strict)``: policy layer. Used by
  client constructors when ``recommended_sampling=True``/``False``:

    | strict | model in map | behavior                                     |
    |--------|--------------|----------------------------------------------|
    | True   | yes          | return dict                                  |
    | True   | no           | raise ``UnsupportedModelError``              |
    | False  | yes          | one-shot INFO log; return ``{}``             |
    | False  | no           | return ``{}`` (silent)                       |

Proxy callers do not consult this map. The proxy plumbs through whatever
sampling params the inbound request carries (OpenAI-compatible body
fields: ``temperature``, ``top_p``, ``top_k``, ``min_p``,
``repeat_penalty``, ``presence_penalty``, ``seed``). For per-model
recommended sampling in proxy mode, the calling client looks up the map
and includes the params in the request body.

Each row carries an inline URL comment pointing at the HF model card the
values were pulled from. Values are verified one model at a time against
the live card — do not add entries without fetching the card.
"""

from __future__ import annotations

import logging

from forge.errors import UnsupportedModelError

log = logging.getLogger(__name__)

# Models for which we've already logged the "supported but not opted-in" info
# message this session. Keyed by model name; the log fires once per (model,
# process) pair.
_INFO_LOGGED: set[str] = set()


MODEL_SAMPLING_DEFAULTS: dict[str, dict[str, float | int]] = {
    # Each model is keyed once per identity form the caller might use:
    #   - Ollama-style string (for OllamaClient)
    #   - GGUF stem (for LlamafileClient with a .gguf — llama-server)
    #   - Llamafile stem (for LlamafileClient with a .llamafile binary)
    # All forms point at independent rows. They start as copies of the same
    # HF card values, but stay independent so vendor-specific guidance (e.g.
    # an Ollama-published modelfile that diverges from the HF card) can be
    # encoded without forcing alignment.

    # Qwen3 — thinking-mode values (card also lists non-thinking variant; forge
    # runs these in thinking mode by default via --reasoning-format auto).
    "qwen3:4b-instruct-2507-q4_K_M":      {"temperature": 0.7, "top_p": 0.8,  "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
    "qwen3:4b-thinking-2507-q4_K_M":      {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
    "qwen3:8b-q4_K_M":                    {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-8B
    "Qwen3-8B-Q4_K_M":                    {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-8B
    "qwen3:8b-q8_0":                      {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-8B
    "Qwen3-8B-Q8_0":                      {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-8B
    "qwen3:14b-q4_K_M":                   {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-14B
    "Qwen3-14B-Q4_K_M":                   {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-14B
    # Qwen3.5/3.6 — thinking-mode general-tasks profile. For precise-coding (WebDev) swap
    # temperature=0.6 and presence_penalty=0.0 (other keys unchanged).
    "qwen3.5:27b-q4_K_M":                 {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},                         # https://huggingface.co/Qwen/Qwen3.5-27B
    "Qwen3.5-27B-Q4_K_M":                 {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},                         # https://huggingface.co/Qwen/Qwen3.5-27B
    "qwen3.5:35b-a3b-q4_K_M":             {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},                         # https://huggingface.co/Qwen/Qwen3.5-35B-A3B
    "Qwen3.5-35B-A3B-Q4_K_M":             {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},                         # https://huggingface.co/Qwen/Qwen3.5-35B-A3B
    "qwen3.6:35b-a3b-ud-q4_K_M":          {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},                         # https://huggingface.co/Qwen/Qwen3.6-35B-A3B
    "Qwen3.6-35B-A3B-UD-Q4_K_M":          {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},                         # https://huggingface.co/Qwen/Qwen3.6-35B-A3B
    # Qwen3.6-27B (dense) — card explicitly differs from 35B-A3B on
    # presence_penalty (0.0 vs 1.5) for thinking-mode general tasks.
    "qwen3.6:27b-q4_K_M":                 {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},                         # https://huggingface.co/Qwen/Qwen3.6-27B
    "Qwen3.6-27B-Q4_K_M":                 {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},                         # https://huggingface.co/Qwen/Qwen3.6-27B
    # Qwen3-Coder — non-thinking instruct; card does not mention min_p or presence_penalty
    "qwen3-coder:30b-a3b-instruct-q4_K_M":  {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05},                                         # https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
    "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M":  {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05},                                         # https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
    # Qwen3-Next 80B-A3B-Instruct — hybrid attention MoE, 80B/3B-active; card lists thinking-mode profile
    "qwen3-next:80b-a3b-instruct-q4_K_M":   {"temperature": 0.7, "top_p": 0.8,  "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
    "Qwen3-Next-80B-A3B-Instruct-Q4_K_M":   {"temperature": 0.7, "top_p": 0.8,  "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
    # Qwen3-Coder-Next — coder fine-tune over Qwen3-Next-80B-A3B base; card differs notably from Qwen3-Coder-30B
    # (no min_p / repeat_penalty / presence_penalty recommendation)
    "qwen3-coder-next:80b-a3b-q4_K_M":      {"temperature": 1.0, "top_p": 0.95, "top_k": 40},                                                                # https://huggingface.co/Qwen/Qwen3-Coder-Next
    "Qwen3-Coder-Next-Q4_K_M":              {"temperature": 1.0, "top_p": 0.95, "top_k": 40},                                                                # https://huggingface.co/Qwen/Qwen3-Coder-Next
    # Gemma 4 — card gives one standardized profile for all use cases; no min_p/repeat/presence recommended
    "gemma4:31b-it-q4_K_M":               {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-31b-it
    "gemma-4-31B-it-Q4_K_M":              {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-31b-it
    "gemma4:26b-a4b-it-q4_K_M":           {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-26b-a4b-it
    "gemma-4-26B-A4B-it-UD-Q4_K_M":       {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-26b-a4b-it
    "gemma4:26b-a4b-it-q8_0":             {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-26b-a4b-it
    "gemma-4-26B-A4B-it-Q8_0":            {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-26b-a4b-it
    "gemma4:e4b-it-q4_K_M":               {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-e4b-it
    "gemma-4-E4B-it-Q4_K_M":              {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-e4b-it
    "gemma4:e4b-it-q8_0":                 {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-e4b-it
    "gemma-4-E4B-it-Q8_0":                {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-e4b-it
    # Mistral Small 4 — card gives T=0.7 for reasoning_effort="high" and "between 0.0 and 0.7" for
    # reasoning_effort="none" (task-dependent). Picking the high-effort profile (T=0.7 + explicit
    # reasoning_effort="high" via chat_template_kwargs envelope) as the safer default. top_p/top_k
    # not specified on the card.
    "mistral-small-4:119b-2603-q4_K_M":     {"temperature": 0.7, "chat_template_kwargs": {"reasoning_effort": "high"}},                                      # https://huggingface.co/mistralai/Mistral-Small-4-119B-2603
    "Mistral-Small-4-119B-2603-UD-Q4_K_M":  {"temperature": 0.7, "chat_template_kwargs": {"reasoning_effort": "high"}},                                      # https://huggingface.co/mistralai/Mistral-Small-4-119B-2603
    # Qwen3.5-122B-A10B — 122B / 10B active MoE. Smoke probe starts in instruct
    # mode (--reasoning-budget 0 in server flags). Card's "balanced" preset:
    # T=0.7, top_p=0.8, top_k=20. Bumping to thinking mode is a separate run if
    # smoke clean. enable_thinking left as template default (true server-side
    # but reasoning-budget=0 caps it to 0 tokens).
    "Qwen3.5-122B-A10B-Q4_K_M":             {"temperature": 0.7, "top_p": 0.8, "top_k": 20},                                                                 # https://huggingface.co/Qwen/Qwen3.5-122B-A10B
    # gpt-oss-120b — OpenAI open-weight MoE (117B total, 5.1B active). Reasoning model with three
    # discrete levels: "low" / "medium" / "high", controlled via chat_template_kwargs.reasoning_effort
    # (per llama.cpp guide: `--chat-template-kwargs '{"reasoning_effort": "high"}'`). Defaulting to
    # "high" — bring down if overthinking observed. Card says T=1.0, top_p=1.0; llama.cpp guide adds
    # top_k=0, min_p=0.0 (OpenAI's stated default; llama.cpp maintainer notes top_k=0 may add CPU
    # overhead but we're GPU). CRITICAL: do NOT set repeat_penalty/presence_penalty (guide explicitly
    # warns to disable repetition penalties) — registry omission == None == field omitted from body.
    "gpt-oss:120b-q4_K_M":  {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "chat_template_kwargs": {"reasoning_effort": "medium"}},  # https://huggingface.co/openai/gpt-oss-120b + https://github.com/ggml-org/llama.cpp/discussions/15396
    "gpt-oss-120b-Q4_K_M":  {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "chat_template_kwargs": {"reasoning_effort": "medium"}},  # https://huggingface.co/openai/gpt-oss-120b + https://github.com/ggml-org/llama.cpp/discussions/15396
    # NVIDIA Nemotron-3-Super-120B-A12B — hybrid Mamba-2 + Transformer + MoE. Reasoning model with
    # three states via chat_template_kwargs.enable_thinking: True (full default), True+low_effort
    # for lighter think, or False (off). Defaulting to enable_thinking=True (no low_effort) for full
    # effort, mirroring Mistral-Small-4 decision; bring down if overthinking. Card recommends
    # T=1.0, top_p=0.95 across all tasks. Coding-agent serving guidance also adds force_nonempty_content
    # so the model emits something substantive instead of empty <think> blocks. Native tool format is
    # Qwen3-coder-style (vLLM uses --tool-call-parser qwen3_coder); should work fine through forge's
    # existing prompt-mode rescue parsers, no new parser needed.
    "nemotron-3-super:120b-a12b-q4_K_M":            {"temperature": 1.0, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": True, "low_effort": True, "force_nonempty_content": True}},  # https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
    "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M":  {"temperature": 1.0, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": True, "low_effort": True, "force_nonempty_content": True}},  # https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
    # NVIDIA Nemotron-3-Nano-30B-A3B — same hybrid Mamba-2 + Transformer + MoE family as Super, 30B
    # total / ~3.5B active. Card splits sampling into two presets:
    #   - "Reasoning":    T=1.0, top_p=1.0   — wider nucleus than Super's 0.95
    #   - "Tool-calling": T=0.6, top_p=0.95  — deterministic preset (active below)
    # The "tool-calling" label is misleading: temperature is per-call and global, so the same low-T
    # preset reins in *all* generation (reasoning, tool emission, prose). 2026-05-05 N=5 probe with
    # the reasoning preset (T=1.0, top_p=1.0) gave decep% 16.04 driven entirely by F3 (9/15 silent
    # fails) — model confidently claimed completion while pipeline output diverged from golden.
    # Switched to the deterministic preset to test whether F3 deception is structural or temperature-
    # tunable. Native tool format is qwen3_coder XML — irrelevant for forge prompt-mode runs.
    # force_nonempty_content (Super's empty-<think> fix) is undocumented for Nano; add only if eval
    # shows empty thinking blocks. Per gpt_oss precedent, repeat_penalty/presence_penalty are
    # deliberately omitted (LlamafileClient drops missing fields entirely; explicit values would
    # degrade output for unfiltered models).
    "Nemotron-3-Nano-30B-A3B-Q4_K_M":               {"temperature": 0.6, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": True}},  # https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
    # Mistral Small 3.2 & Devstral Small 2 — cards only specify temperature; top_p/top_k/etc. left to backend defaults
    "mistral-small-3.2:24b-instruct-2506-q4_K_M":  {"temperature": 0.15},  # https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
    "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M":  {"temperature": 0.15},  # https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
    "mistral-small-3.2:24b-instruct-2506-q8_0":    {"temperature": 0.15},  # https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
    "Mistral-Small-3.2-24B-Instruct-2506-Q8_0":    {"temperature": 0.15},  # https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
    "devstral-small-2:24b-instruct-2512-q4_K_M":   {"temperature": 0.15},  # https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512
    "Devstral-Small-2-24B-Instruct-2512-Q4_K_M":   {"temperature": 0.15},  # https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512
    "devstral-small-2:24b-instruct-2512-q8_0":     {"temperature": 0.15},  # https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512
    "Devstral-Small-2-24B-Instruct-2512-Q8_0":     {"temperature": 0.15},  # https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512
    # Ministral-3 Instruct — card says "temperature below 0.1 for production"; 0.05 picked within that range.
    "ministral-3:8b-instruct-2512-q4_K_M":  {"temperature": 0.05},  # https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512 (card: temp<0.1)
    "Ministral-3-8B-Instruct-2512-Q4_K_M":  {"temperature": 0.05},  # https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512 (card: temp<0.1)
    "ministral-3:8b-instruct-2512-q8_0":    {"temperature": 0.05},  # https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512 (card: temp<0.1)
    "Ministral-3-8B-Instruct-2512-Q8_0":    {"temperature": 0.05},  # https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512 (card: temp<0.1)
    "ministral-3:14b-instruct-2512-q4_K_M": {"temperature": 0.05},  # https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512 (card: temp<0.1)
    "Ministral-3-14B-Instruct-2512-Q4_K_M": {"temperature": 0.05},  # https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512 (card: temp<0.1)
    # Ministral-3 Reasoning — cards give a specific temperature each (different per size). top_p=0.95 appears
    # in the cards' code examples but is NOT a formal recommendation, so it's omitted here.
    "ministral-3:8b-reasoning-2512-q4_K_M":  {"temperature": 0.7},  # https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
    "Ministral-3-8B-Reasoning-2512-Q4_K_M":  {"temperature": 0.7},  # https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
    "ministral-3:8b-reasoning-2512-q8_0":    {"temperature": 0.7},  # https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
    "Ministral-3-8B-Reasoning-2512-Q8_0":    {"temperature": 0.7},  # https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
    "ministral-3:14b-reasoning-2512-q4_K_M": {"temperature": 1.0},  # https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512
    "Ministral-3-14B-Reasoning-2512-Q4_K_M": {"temperature": 1.0},  # https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512
    # Mistral Nemo — formal recommendation is temp=0.3; code examples on the card use 0.35 (formal rec wins).
    "mistral-nemo:12b-instruct-2407-q4_K_M": {"temperature": 0.3},  # https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
    "Mistral-Nemo-Instruct-2407-Q4_K_M":     {"temperature": 0.3},  # https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407 (GGUF)
    "Mistral-Nemo-Instruct-2407.Q4_K_M":     {"temperature": 0.3},  # https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407 (llamafile)
    # Granite 4.0 — IBM-pointed reference cites greedy decoding (T=0); top_p/top_k effectively no-op at T=0
    # but kept explicit to match the source recommendation. IBM HF cards / GitHub repo / prompt-engineering
    # guide v2 themselves do not document sampling; Unsloth's tutorial cites IBM directly.
    "granite-4.0:h-micro-q4_K_M":  {"temperature": 0.0, "top_p": 1.0, "top_k": 0},  # https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0 (cites IBM)
    "granite-4.0-h-micro-Q4_K_M":  {"temperature": 0.0, "top_p": 1.0, "top_k": 0},  # https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0 (cites IBM)
    "granite-4.0:h-tiny-q4_K_M":   {"temperature": 0.0, "top_p": 1.0, "top_k": 0},  # https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0 (cites IBM)
    "granite-4.0-h-tiny-Q4_K_M":   {"temperature": 0.0, "top_p": 1.0, "top_k": 0},  # https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0 (cites IBM)
    # Granite 4.1 — UNCONFIRMED: no card-formal sampling recommendation from any source as of 2026-05-09
    # (HF model card, unsloth GGUF page, IBM blog, IBM granite-4.1-language-models GitHub repo all
    # silent on sampling; IBM granite/docs page returned 403; Unsloth tutorial 404). Mirroring Granite
    # 4.0 IBM convention (greedy decoding) since Granite 4.1 is confirmed non-thinking per the HF blog
    # ("without relying on long chains of thought") and IBM has been consistent across the family.
    # Replace with card-direct values if/when IBM publishes formal guidance.
    "granite4.1:8b-q4_K_M":        {"temperature": 0.0, "top_p": 1.0, "top_k": 0},  # unconfirmed; mirrors granite-4.0 IBM convention
    "granite-4.1-8b-Q4_K_M":       {"temperature": 0.0, "top_p": 1.0, "top_k": 0},  # unconfirmed; mirrors granite-4.0 IBM convention
    "granite4.1:8b-q8_0":          {"temperature": 0.0, "top_p": 1.0, "top_k": 0},  # unconfirmed; mirrors granite-4.0 IBM convention
    "granite-4.1-8b-Q8_0":         {"temperature": 0.0, "top_p": 1.0, "top_k": 0},  # unconfirmed; mirrors granite-4.0 IBM convention
    # Intentionally absent — no formal recommendation from any official source:
    #   llama3.1:*                Meta's HF card, llama.com/docs, and llama-recipes are all silent.
    #   mistral:7b-instruct-v0.3  HF card has no "recommended settings" section; code examples
    #                             use temperature=0.0 (greedy) but note it's demo-only, not a rec.
    #   phi-4 (base, non-reasoning)  HF card, Phi cookbook, Phi-4 technical report (2412.08905), and
    #                                Microsoft Foundry all silent on sampling; only T=0.0 demo
    #                                examples. Forge runs base phi-4 with recommended_sampling=False
    #                                (see batch_eval.py:_NO_RECOMMENDED_SAMPLING_MODELS).
    # These rows fall through to the unknown-model path (backend defaults).
}


def get_sampling_defaults(model: str) -> dict[str, float | int]:
    """Return recommended sampling params for ``model``.

    Pure lookup — no logging, no raising. For unknown models returns ``{}``.
    Use ``apply_sampling_defaults`` for the constructor-time policy layer
    that handles opt-in/opt-out, INFO logs, and ``UnsupportedModelError``.

    Args:
        model: Model name as used by the backend (e.g. ``"qwen3:8b-q4_K_M"``).

    Returns:
        Fresh dict of sampling kwargs (``temperature``, ``top_p``, ``top_k``,
        ``min_p``, ``repeat_penalty``, ``presence_penalty`` — subset per
        model). Empty dict for unknown models.
    """
    return dict(MODEL_SAMPLING_DEFAULTS.get(model, {}))


def apply_sampling_defaults(
    model: str,
    *,
    strict: bool,
) -> dict[str, float | int]:
    """Apply the recommended-sampling policy for ``model``.

    Called by client constructors at instantiation time. The four-quadrant
    behavior:

    - ``strict=True`` + known model: return the map value (dict copy).
    - ``strict=True`` + unknown model: raise ``UnsupportedModelError``.
    - ``strict=False`` + known model: one-shot INFO log; return ``{}``.
    - ``strict=False`` + unknown model: return ``{}`` (silent).

    The INFO log fires once per (model, process) pair to surface that
    forge has opinions about this model without spamming on every
    constructor call.

    Args:
        model: Model name (e.g. ``"qwen3:8b-q4_K_M"``).
        strict: If True, the caller declared ``recommended_sampling=True``;
            unknown models are an error. If False, recommended sampling
            was not opted into; the function returns ``{}`` and may log.

    Returns:
        Dict of sampling kwargs to splat onto the constructor, or ``{}``.

    Raises:
        UnsupportedModelError: ``strict=True`` and ``model`` is not in
            ``MODEL_SAMPLING_DEFAULTS``.
    """
    in_map = model in MODEL_SAMPLING_DEFAULTS
    if strict:
        if not in_map:
            raise UnsupportedModelError(model)
        return dict(MODEL_SAMPLING_DEFAULTS[model])

    # strict=False: caller did not opt in.
    if in_map and model not in _INFO_LOGGED:
        log.info(
            "Recommended sampling params exist for %r; "
            "pass recommended_sampling=True to use them.",
            model,
        )
        _INFO_LOGGED.add(model)
    return {}
