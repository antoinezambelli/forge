"""Literal batch server recipes and managed config roster parity."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import tests.eval.batch_eval as batch_eval


_EXPECTED_GGUFS = [
    ("Qwen3-8B-Q4_K_M", "Qwen3-8B-Q4_K_M.gguf"),
    ("Qwen3-8B-Q8_0", "Qwen3-8B-Q8_0.gguf"),
    ("Qwen3-14B-Q4_K_M", "Qwen3-14B-Q4_K_M.gguf"),
    ("Ministral-3-8B-Instruct-2512-Q4_K_M", "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf"),
    ("Ministral-3-8B-Instruct-2512-Q8_0", "Ministral-3-8B-Instruct-2512-Q8_0.gguf"),
    ("Ministral-3-14B-Instruct-2512-Q4_K_M", "Ministral-3-14B-Instruct-2512-Q4_K_M.gguf"),
    ("Ministral-3-8B-Reasoning-2512-Q4_K_M", "Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf"),
    ("Ministral-3-8B-Reasoning-2512-Q8_0", "Ministral-3-8B-Reasoning-2512-Q8_0.gguf"),
    ("Ministral-3-14B-Reasoning-2512-Q4_K_M", "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf"),
    ("gemma-4-E4B-it-Q4_K_M", "gemma-4-E4B-it-Q4_K_M.gguf"),
    ("gemma-4-E4B-it-Q8_0", "gemma-4-E4B-it-Q8_0.gguf"),
    ("granite-4.1-8b-Q4_K_M", "granite-4.1-8b-Q4_K_M.gguf"),
    ("granite-4.1-8b-Q8_0", "granite-4.1-8b-Q8_0.gguf"),
    ("phi-4-Q4_K_M", "phi-4-Q4_K_M.gguf"),
    ("Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M", "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf"),
    ("Qwen3.5-27B-Q4_K_M", "Qwen3.5-27B-Q4_K_M.gguf"),
    ("Qwen3.5-35B-A3B-Q4_K_M", "Qwen3.5-35B-A3B-Q4_K_M.gguf"),
    ("Qwen3.6-27B-Q4_K_M", "Qwen3.6-27B-Q4_K_M.gguf"),
    ("Qwen3.6-35B-A3B-UD-Q4_K_M", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    ("Nemotron-3-Nano-30B-A3B-Q4_K_M", "Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf"),
    ("Muse-Glimmer-30B-UD-Q4_K_XL", "Muse-Glimmer-30B-UD-Q4_K_XL.gguf"),
    ("gemma-4-26B-A4B-it-UD-Q4_K_M", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
    ("gemma-4-31B-it-Q4_K_M", "gemma-4-31B-it-Q4_K_M.gguf"),
    ("gpt-oss-120b-Q4_K_M", "gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"),
    ("Qwen3.5-122B-A10B-Q4_K_M", "Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"),
    (
        "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M",
        "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00001-of-00003.gguf",
    ),
    ("LFM2.5-8B-A1B-Q4_K_M", "LFM2.5-8B-A1B-Q4_K_M.gguf"),
    ("Mellum2-12B-A2.5B-Thinking-Q4_K_M", "Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf"),
    ("Mellum2-12B-A2.5B-Instruct-Q4_K_M", "Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf"),
]

_EXPECTED_LLAMAFILES = [
    ("Meta-Llama-3.1-8B-Instruct.Q4_K_M", "Meta-Llama-3.1-8B-Instruct.Q4_K_M.llamafile"),
    ("Meta-Llama-3.1-8B-Instruct.Q8_0", "Meta-Llama-3.1-8B-Instruct.Q8_0.llamafile"),
    ("Mistral-Nemo-Instruct-2407.Q4_K_M", "Mistral-Nemo-Instruct-2407.Q4_K_M.llamafile"),
    ("Mistral-7B-Instruct-v0.3.Q4_K_M", "Mistral-7B-Instruct-v0.3.Q4_K_M.llamafile"),
    ("Mistral-7B-Instruct-v0.3.Q8_0", "Mistral-7B-Instruct-v0.3.Q8_0.llamafile"),
]

_EXPECTED_OLLAMA_MODELS = [
    "qwen3:8b-q4_K_M",
    "qwen3:8b-q8_0",
    "qwen3:14b-q4_K_M",
    "ministral-3:8b-instruct-2512-q4_K_M",
    "ministral-3:8b-instruct-2512-q8_0",
    "ministral-3:14b-instruct-2512-q4_K_M",
    "gemma4:e4b-it-q4_K_M",
    "gemma4:e4b-it-q8_0",
    "granite4.1:8b-q4_K_M",
    "granite4.1:8b-q8_0",
]

_REASONING_FLAGS = ("--reasoning-format", "auto")
_GEMMA4_LARGE_FLAGS = (
    "--reasoning-format", "auto",
    "--ctx-checkpoints", "1", "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0", "-fa", "1",
    "--samplers", "temperature;top_p;top_k",
)
_GPT_OSS_120B_FLAGS = (
    "--reasoning-format", "auto",
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-fa", "1",
    "-ub", "2048", "-b", "2048",
    "--no-prefill-assistant", "--no-mmap",
)
_LARGE_120B_FLAGS = (
    "--reasoning-format", "auto",
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-fa", "1",
    "--no-prefill-assistant", "--no-mmap",
)
_GLIMMER_FLAGS = (
    "--reasoning", "on", "--reasoning-format", "auto",
    "--chat-template-kwargs", '{"reasoning_strength":"xhigh"}',
    "--ctx-checkpoints", "1", "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0", "-fa", "1",
    "--samplers", "temperature;top_p;top_k",
    "--spec-type", "draft-dflash", "--device-draft", "CUDA0",
    "--gpu-layers-draft", "all", "--spec-draft-n-max", "15",
)

_EXPECTED_SPECIAL_FLAGS = {
    "Qwen3-8B-Q4_K_M": _REASONING_FLAGS,
    "Qwen3-8B-Q8_0": _REASONING_FLAGS,
    "Qwen3-14B-Q4_K_M": _REASONING_FLAGS,
    "Qwen3.5-27B-Q4_K_M": _REASONING_FLAGS,
    "Qwen3.5-35B-A3B-Q4_K_M": _REASONING_FLAGS,
    "Qwen3.6-27B-Q4_K_M": _REASONING_FLAGS,
    "Qwen3.6-35B-A3B-UD-Q4_K_M": _REASONING_FLAGS,
    "Nemotron-3-Nano-30B-A3B-Q4_K_M": _REASONING_FLAGS,
    "Muse-Glimmer-30B-UD-Q4_K_XL": _GLIMMER_FLAGS,
    "LFM2.5-8B-A1B-Q4_K_M": _REASONING_FLAGS,
    "Mellum2-12B-A2.5B-Thinking-Q4_K_M": _REASONING_FLAGS,
    "gemma-4-26B-A4B-it-UD-Q4_K_M": _GEMMA4_LARGE_FLAGS,
    "gemma-4-31B-it-Q4_K_M": _GEMMA4_LARGE_FLAGS,
    "gpt-oss-120b-Q4_K_M": _GPT_OSS_120B_FLAGS,
    "Qwen3.5-122B-A10B-Q4_K_M": _LARGE_120B_FLAGS,
    "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M": _LARGE_120B_FLAGS,
}

_EXPECTED_REASONING_LEVELS = {
    "gpt-oss-120b-Q4_K_M": "medium",
    "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M": "low",
    "Muse-Glimmer-30B-UD-Q4_K_XL": "xhigh",
}


def _identity(config: batch_eval.BatchConfig) -> tuple[str, str, str, str, str | None]:
    return (
        config.model,
        config.backend,
        config.mode,
        config.reasoning_level,
        config.gguf_filename,
    )


def _expected_llamaserver_identities() -> list[tuple[str, str, str, str, str]]:
    identities: list[tuple[str, str, str, str, str]] = []
    for model, filename in _EXPECTED_GGUFS:
        reasoning_level = _EXPECTED_REASONING_LEVELS.get(model, "default")
        if model != "phi-4-Q4_K_M":
            identities.append((model, "llamaserver", "native", reasoning_level, filename))
        identities.append((model, "llamaserver", "prompt", reasoning_level, filename))
    return identities


def test_managed_config_roster_and_sets_are_pinned() -> None:
    llamaserver = _expected_llamaserver_identities()
    llamafile = [
        (model, "llamafile", "prompt", "default", filename)
        for model, filename in _EXPECTED_LLAMAFILES
    ]
    ollama = [
        (model, "ollama", "native", "default", None)
        for model in _EXPECTED_OLLAMA_MODELS
    ]
    reasoning_high = [
        (
            "gpt-oss-120b-Q4_K_M", "llamaserver", "native", "high",
            "gpt-oss-120b-Q4_K_M-00001-of-00002.gguf",
        ),
        (
            "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M",
            "llamaserver", "native", "high",
            "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00001-of-00003.gguf",
        ),
    ]
    deepseek_v4_rpc = [
        (
            "DeepSeek-V4-Flash-0731-UD-Q4_K_XL",
            "llamaserver", "native", "low",
            "DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf",
        ),
    ]
    new_models = {
        "LFM2.5-8B-A1B-Q4_K_M",
        "Mellum2-12B-A2.5B-Thinking-Q4_K_M",
        "Mellum2-12B-A2.5B-Instruct-Q4_K_M",
    }
    expected_sets = {
        "all": llamaserver + llamafile + ollama,
        "ollama": ollama,
        "llamaserver": llamaserver,
        "llamafile": llamafile,
        "llamaserver-native": [entry for entry in llamaserver if entry[2] == "native"],
        "llamaserver-prompt": [entry for entry in llamaserver if entry[2] == "prompt"],
        "reasoning-high": reasoning_high,
        "deepseek-v4-rpc": deepseek_v4_rpc,
        "new-models": [entry for entry in llamaserver if entry[0] in new_models],
        "new-models-native": [
            entry for entry in llamaserver
            if entry[0] in new_models and entry[2] == "native"
        ],
        "new-models-prompt": [
            entry for entry in llamaserver
            if entry[0] in new_models and entry[2] == "prompt"
        ],
    }

    assert list(batch_eval.CONFIG_SETS) == list(expected_sets)
    for name, expected in expected_sets.items():
        assert [_identity(config) for config in batch_eval.CONFIG_SETS[name]] == expected


def test_managed_recipe_mapping_matches_pinned_literals() -> None:
    llamaserver_configs = (
        batch_eval.LLAMASERVER_CONFIGS + batch_eval._REASONING_HIGH_CONFIGS
    )
    for config in llamaserver_configs:
        expected = _EXPECTED_SPECIAL_FLAGS.get(config.model, ())
        assert config.server_recipe.extra_flags == expected, _identity(config)
        assert "--jinja" not in config.server_recipe.extra_flags, _identity(config)

    for config in batch_eval.OLLAMA_CONFIGS + batch_eval.LLAMAFILE_CONFIGS:
        assert config.server_recipe.extra_flags == (), _identity(config)

    glimmer = next(
        config for config in batch_eval.LLAMASERVER_CONFIGS
        if config.model == "Muse-Glimmer-30B-UD-Q4_K_XL"
    )
    assert glimmer.server_recipe.draft_filename == "dflash-kquant.gguf"

    for model, reasoning_level in _EXPECTED_REASONING_LEVELS.items():
        configs = [
            config
            for config in batch_eval.LLAMASERVER_CONFIGS
            if config.model == model
        ]
        assert {config.mode for config in configs} == {"native", "prompt"}
        assert {config.reasoning_level for config in configs} == {reasoning_level}

    assert (
        batch_eval.get_sampling_defaults("gpt-oss-120b-Q4_K_M")
        ["chat_template_kwargs"]["reasoning_effort"]
        == "medium"
    )
    assert (
        batch_eval.get_sampling_defaults(
            "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M"
        )["chat_template_kwargs"]["low_effort"]
        is True
    )
    assert '{"reasoning_strength":"xhigh"}' in glimmer.server_recipe.extra_flags

    deepseek = batch_eval.DEEPSEEK_V4_RPC_CONFIGS[0]
    assert deepseek.server_recipe.extra_flags == (
        "--fit", "off",
        "-b", "2048", "-ub", "128",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--no-mmap", "-fa", "on",
        "--reasoning-budget", "32768", "--reasoning-format", "auto",
        "--no-prefill-assistant",
    )
    assert deepseek.server_recipe.rpc is None
    assert deepseek.sampling_override is None
    assert deepseek.reasoning_level == "low"
    assert (
        batch_eval.get_sampling_defaults(deepseek.model)
        ["chat_template_kwargs"]["reasoning_effort"]
        == deepseek.reasoning_level
    )
    for manager_owned in ("-ngl", "--jinja", "--rpc", "--device", "--tensor-split"):
        assert manager_owned not in deepseek.server_recipe.extra_flags

    recipe = batch_eval._BatchServerRecipe(("--reasoning-format", "auto"))
    with pytest.raises(FrozenInstanceError):
        recipe.extra_flags = ()  # type: ignore[misc]

    assert not hasattr(batch_eval, "_SERVER_EXTRA_FLAGS")
    assert not hasattr(batch_eval, "_get_server_flags")
