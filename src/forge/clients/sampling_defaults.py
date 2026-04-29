"""Per-model recommended sampling defaults sourced from HF model cards.

Used in managed mode (Engine, WorkflowRunner, batch_eval). Callers look up
defaults explicitly via ``get_sampling_defaults(model)`` and splat into the
client constructor.

Proxy/passthrough callers do NOT consult this map — they forward whatever
the caller sent.

An unknown model returns an empty dict rather than raising: forge allows all
models, but only officially supports those listed here. For unknown models
the backend's own defaults apply.

Each row carries an inline URL comment pointing at the HF model card the
values were pulled from. Values are verified one model at a time against
the live card — do not add entries without fetching the card.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Models for which we've already logged a "no defaults" warning this session.
_WARNED_UNKNOWN: set[str] = set()


MODEL_SAMPLING_DEFAULTS: dict[str, dict[str, float | int]] = {
    # Qwen3 — thinking-mode values (card also lists non-thinking variant; forge
    # runs these in thinking mode by default via --reasoning-format auto).
    "qwen3:4b-instruct-2507-q4_K_M":      {"temperature": 0.7, "top_p": 0.8,  "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
    "qwen3:4b-thinking-2507-q4_K_M":      {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
    "qwen3:8b-q4_K_M":                    {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-8B
    "qwen3:8b-q8_0":                      {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-8B
    "qwen3:14b-q4_K_M":                   {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},                                                  # https://huggingface.co/Qwen/Qwen3-14B
    # Qwen3.5/3.6 — thinking-mode general-tasks profile. For precise-coding (WebDev) swap
    # temperature=0.6 and presence_penalty=0.0 (other keys unchanged).
    "qwen3.5:27b-q4_K_M":                 {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},                         # https://huggingface.co/Qwen/Qwen3.5-27B
    "qwen3.5:35b-a3b-q4_K_M":             {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},                         # https://huggingface.co/Qwen/Qwen3.5-35B-A3B
    "qwen3.6:35b-a3b-ud-q4_K_M":          {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},                         # https://huggingface.co/Qwen/Qwen3.6-35B-A3B
    # Qwen3-Coder — non-thinking instruct; card does not mention min_p or presence_penalty
    "qwen3-coder:30b-a3b-instruct-q4_K_M": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05},                                          # https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
    # Gemma 4 — card gives one standardized profile for all use cases; no min_p/repeat/presence recommended
    "gemma4:31b-it-q4_K_M":       {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-31b-it
    "gemma4:26b-a4b-it-q4_K_M":   {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-26b-a4b-it
    "gemma4:26b-a4b-it-q8_0":     {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-26b-a4b-it
    "gemma4:e4b-it-q4_K_M":       {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-e4b-it
    "gemma4:e4b-it-q8_0":         {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # https://huggingface.co/google/gemma-4-e4b-it
    # Mistral Small 3.2 & Devstral Small 2 — cards only specify temperature; top_p/top_k/etc. left to backend defaults
    "mistral-small-3.2:24b-instruct-2506-q4_K_M":  {"temperature": 0.15},  # https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
    "mistral-small-3.2:24b-instruct-2506-q8_0":    {"temperature": 0.15},  # https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
    "devstral-small-2:24b-instruct-2512-q4_K_M":   {"temperature": 0.15},  # https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512
    "devstral-small-2:24b-instruct-2512-q8_0":     {"temperature": 0.15},  # https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512
    # Ministral-3 Instruct — card says "temperature below 0.1 for production"; 0.05 picked within that range.
    "ministral-3:8b-instruct-2512-q4_K_M":   {"temperature": 0.05},  # https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512 (card: temp<0.1)
    "ministral-3:8b-instruct-2512-q8_0":     {"temperature": 0.05},  # https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512 (card: temp<0.1)
    "ministral-3:14b-instruct-2512-q4_K_M":  {"temperature": 0.05},  # https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512 (card: temp<0.1)
    # Ministral-3 Reasoning — cards give a specific temperature each (different per size). top_p=0.95 appears
    # in the cards' code examples but is NOT a formal recommendation, so it's omitted here.
    "ministral-3:8b-reasoning-2512-q4_K_M":  {"temperature": 0.7},  # https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
    "ministral-3:8b-reasoning-2512-q8_0":    {"temperature": 0.7},  # https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
    "ministral-3:14b-reasoning-2512-q4_K_M": {"temperature": 1.0},  # https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512
    # Mistral Nemo — formal recommendation is temp=0.3; code examples on the card use 0.35 (formal rec wins).
    "mistral-nemo:12b-instruct-2407-q4_K_M": {"temperature": 0.3},  # https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
    # Granite 4.0 — IBM-pointed reference cites greedy decoding (T=0); top_p/top_k effectively no-op at T=0
    # but kept explicit to match the source recommendation. IBM HF cards / GitHub repo / prompt-engineering
    # guide v2 themselves do not document sampling; Unsloth's tutorial cites IBM directly.
    "granite-4.0:h-micro-q4_K_M": {"temperature": 0.0, "top_p": 1.0, "top_k": 0},  # https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0 (cites IBM)
    "granite-4.0:h-tiny-q4_K_M":  {"temperature": 0.0, "top_p": 1.0, "top_k": 0},  # https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0 (cites IBM)
    # Intentionally absent — no formal recommendation from any official source:
    #   llama3.1:*                Meta's HF card, llama.com/docs, and llama-recipes are all silent.
    #   mistral:7b-instruct-v0.3  HF card has no "recommended settings" section; code examples
    #                             use temperature=0.0 (greedy) but note it's demo-only, not a rec.
    # These rows fall through to the unknown-model path (backend defaults).
}


def get_sampling_defaults(model: str) -> dict[str, float | int]:
    """Return recommended sampling params for ``model``.

    For unknown models returns ``{}`` and logs a one-shot warning. The caller
    splats the result into the client constructor; an empty dict means no
    overrides, which lets the backend's own defaults apply.

    Args:
        model: Model name as used by the backend (e.g. ``"qwen3:8b-q4_K_M"``).

    Returns:
        Dict of sampling kwargs (``temperature``, ``top_p``, ``top_k``,
        ``min_p``, ``repeat_penalty`` — subset per model).
    """
    if model in MODEL_SAMPLING_DEFAULTS:
        return dict(MODEL_SAMPLING_DEFAULTS[model])

    if model not in _WARNED_UNKNOWN:
        log.warning(
            "No sampling defaults registered for model %r — backend defaults will apply",
            model,
        )
        _WARNED_UNKNOWN.add(model)
    return {}
