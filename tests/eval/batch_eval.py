"""Batch eval runner - iterate model/backend/mode configs, append JSONL.

Usage:
    python -m tests.eval.batch_eval [--runs 50] [--output results.jsonl]
                                     [--config CONFIG_NAME] [--dry-run]

Resumes automatically: for each (model, backend, mode, scenario) it counts
existing recorded attempts in the JSONL and only runs the remainder.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from forge.clients.sampling_defaults import get_sampling_defaults
from forge.core.reasoning import DEFAULT_REASONING_REPLAY, REASONING_REPLAY_CHOICES, ReasoningReplay
from forge.rpc import (
    LlamaCppRpcConfig,
    LlamaCppRpcWorkerConfig,
    render_rpc_coordinator_args,
    render_rpc_worker_command,
)
from forge.server import BudgetMode, ServerManager, setup_backend

from tests.eval.ablation import ABLATION_PRESETS, AblationConfig
from tests.eval.eval_runner import EvalConfig, RunResult, run_scenario
from tests.eval.generation import effective_generation, effective_reasoning_replay
from tests.eval.metrics import analyze_history, compute_metrics, count_wire_reasoning
from tests.eval.outcomes import (
    CANONICAL_DIALECT,
    OutcomeDialect,
    OutcomeSchemaError,
    RunOutcome,
    detect_outcome_dialect,
    read_outcome,
    write_outcome,
)
from tests.eval.scenarios import ALL_SCENARIOS, EvalScenario

# ── GGUF paths ──────────────────────────────────────────────────

MODELS_DIR_DEFAULT = Path("models")


def _eval_port() -> int:
    """llama-server port for eval workers; overridden by rig wrappers."""
    return int(os.environ.get("FORGE_EVAL_PORT", "8080"))

@dataclass(frozen=True)
class _BatchServerRecipe:
    """Literal server options owned by one managed batch configuration."""

    extra_flags: tuple[str, ...] = ()
    rpc: LlamaCppRpcConfig | None = None
    draft_filename: str | None = None


_DEFAULT_SERVER_RECIPE = _BatchServerRecipe()
_REASONING_SERVER_RECIPE = _BatchServerRecipe(("--reasoning-format", "auto"))
_QWEN38_SERVER_RECIPE = _BatchServerRecipe((
    "--reasoning-format", "auto",
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-fa", "1",
))
_GEMMA4_LARGE_SERVER_RECIPE = _BatchServerRecipe((
    "--reasoning-format", "auto",
    "--ctx-checkpoints", "1", "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0", "-fa", "1",
    "--samplers", "temperature;top_p;top_k",
))
_GPT_OSS_120B_SERVER_RECIPE = _BatchServerRecipe((
    "--reasoning-format", "auto",
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-fa", "1",
    "-ub", "2048", "-b", "2048",
    "--no-prefill-assistant", "--no-mmap",
))
_LARGE_120B_SERVER_RECIPE = _BatchServerRecipe((
    "--reasoning-format", "auto",
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-fa", "1",
    "--no-prefill-assistant", "--no-mmap",
))


def _glimmer_server_recipe(reasoning_strength: str) -> _BatchServerRecipe:
    return _BatchServerRecipe(
        extra_flags=(
            "--reasoning", "on", "--reasoning-format", "auto",
            "--chat-template-kwargs",
            json.dumps(
                {"reasoning_strength": reasoning_strength}, separators=(",", ":")
            ),
            "--ctx-checkpoints", "1", "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0", "-fa", "1",
            "--samplers", "temperature;top_p;top_k",
            "--spec-type", "draft-dflash", "--device-draft", "CUDA0",
            "--gpu-layers-draft", "all", "--spec-draft-n-max", "15",
        ),
        draft_filename="dflash-kquant.gguf",
    )


_GLIMMER_SERVER_RECIPE = _glimmer_server_recipe("xhigh")
_DEEPSEEK_V4_RPC_SERVER_RECIPE = _BatchServerRecipe((
    "--fit", "off",
    "-b", "2048", "-ub", "128",
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    "--no-mmap", "-fa", "on",
    "--reasoning-budget", "32768", "--reasoning-format", "auto",
    "--no-prefill-assistant",
))

_DEEPSEEK_V4_MODEL = "DeepSeek-V4-Flash-0731-UD-Q4_K_XL"
_DEEPSEEK_V4_GGUF = f"{_DEEPSEEK_V4_MODEL}-00001-of-00005.gguf"
_DEEPSEEK_V4_SAMPLING: dict[str, Any] = get_sampling_defaults(_DEEPSEEK_V4_MODEL)
_DEEPSEEK_V4_REASONING_LEVEL: str = _DEEPSEEK_V4_SAMPLING[
    "chat_template_kwargs"
]["reasoning_effort"]

# Effective reasoning levels for model configurations with an explicitly
# controlled effort axis.  "default" remains reserved for configurations that
# do not record a tested effort level; it is also the fallback for legacy rows
# written before the field existed.
_EFFECTIVE_REASONING_LEVELS: dict[str, str] = {
    "gpt-oss-120b-Q4_K_M": "medium",
    "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M": "low",
    "Muse-Glimmer-30B-UD-Q4_K_XL": "xhigh",
    "Qwen3.8-27B-UD-Q4_K_XL": "xhigh",
}


# GGUF files and their literal per-configuration server options. Mode-derived
# options such as --jinja remain owned by ServerManager.
_GGUF_FILES: list[tuple[str, _BatchServerRecipe]] = [
    ("Qwen3-8B-Q4_K_M.gguf", _REASONING_SERVER_RECIPE),
    ("Qwen3-8B-Q8_0.gguf", _REASONING_SERVER_RECIPE),
    ("Qwen3-14B-Q4_K_M.gguf", _REASONING_SERVER_RECIPE),
    ("Ministral-3-8B-Instruct-2512-Q4_K_M.gguf", _DEFAULT_SERVER_RECIPE),
    ("Ministral-3-8B-Instruct-2512-Q8_0.gguf", _DEFAULT_SERVER_RECIPE),
    ("Ministral-3-14B-Instruct-2512-Q4_K_M.gguf", _DEFAULT_SERVER_RECIPE),
    ("Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf", _DEFAULT_SERVER_RECIPE),
    ("Ministral-3-8B-Reasoning-2512-Q8_0.gguf", _DEFAULT_SERVER_RECIPE),
    ("Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf", _DEFAULT_SERVER_RECIPE),
    ("gemma-4-E4B-it-Q4_K_M.gguf", _DEFAULT_SERVER_RECIPE),
    ("gemma-4-E4B-it-Q8_0.gguf", _DEFAULT_SERVER_RECIPE),
    ("granite-4.1-8b-Q4_K_M.gguf", _DEFAULT_SERVER_RECIPE),
    ("granite-4.1-8b-Q8_0.gguf", _DEFAULT_SERVER_RECIPE),
    ("phi-4-Q4_K_M.gguf", _DEFAULT_SERVER_RECIPE),
    # 32GB tier (rig-02 v0.7.1 eval — the configs that ran)
    ("Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf", _DEFAULT_SERVER_RECIPE),
    ("Qwen3.5-27B-Q4_K_M.gguf", _REASONING_SERVER_RECIPE),
    ("Qwen3.5-35B-A3B-Q4_K_M.gguf", _REASONING_SERVER_RECIPE),
    ("Qwen3.6-27B-Q4_K_M.gguf", _REASONING_SERVER_RECIPE),
    ("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", _REASONING_SERVER_RECIPE),
    ("Qwen3.8-27B-UD-Q4_K_XL.gguf", _QWEN38_SERVER_RECIPE),
    ("Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf", _REASONING_SERVER_RECIPE),
    ("Muse-Glimmer-30B-UD-Q4_K_XL.gguf", _GLIMMER_SERVER_RECIPE),
    # Gemma-4 large (rig-04, az/eval-large): 26B-A4B MoE + 31B dense. Native FC;
    # literal serving recipe includes the SWA + q8-KV serving fixes.
    ("gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", _GEMMA4_LARGE_SERVER_RECIPE),
    ("gemma-4-31B-it-Q4_K_M.gguf", _GEMMA4_LARGE_SERVER_RECIPE),
    # 120B tier (rig-03, az/eval-large): multi-shard GGUFs — list the FIRST
    # shard; llama-server auto-loads siblings. The config loop strips the
    # -NNNNN-of-NNNNN suffix for clean model identity and sampling lookups.
    ("gpt-oss-120b-Q4_K_M-00001-of-00002.gguf", _GPT_OSS_120B_SERVER_RECIPE),
    ("Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf", _LARGE_120B_SERVER_RECIPE),
    ("NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00001-of-00003.gguf", _LARGE_120B_SERVER_RECIPE),
    # 16GB tier (rig-01) — LFM2.5 MoE + Mellum2 MoE (both variants). All
    # support native FC, so each gets native + prompt configs below.
    ("LFM2.5-8B-A1B-Q4_K_M.gguf", _REASONING_SERVER_RECIPE),
    ("Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf", _REASONING_SERVER_RECIPE),
    ("Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf", _DEFAULT_SERVER_RECIPE),
]

# Models that lack native function-calling support — only run prompt mode.
# Verified by curl test: model emits text output, no tool_calls field.
_PROMPT_ONLY_MODELS: set[str] = {
    "phi-4-Q4_K_M",  # phi-4 base; native FC not in training corpus
}

# Models with no formal sampling guidance from any authoritative source.
# Run with recommended_sampling=False so the strict-mode UnsupportedModelError
# doesn't fire. See sampling_defaults.py "Intentionally absent" comment block.
_NO_RECOMMENDED_SAMPLING_MODELS: set[str] = {
    "phi-4-Q4_K_M",
}

_LLAMAFILE_FILES: list[str] = [
    "Meta-Llama-3.1-8B-Instruct.Q4_K_M.llamafile",
    "Meta-Llama-3.1-8B-Instruct.Q8_0.llamafile",
    "Mistral-Nemo-Instruct-2407.Q4_K_M.llamafile",
    "Mistral-7B-Instruct-v0.3.Q4_K_M.llamafile",
    "Mistral-7B-Instruct-v0.3.Q8_0.llamafile",
]


# ── Config definitions ──────────────────────────────────────────


@dataclass
class BatchConfig:
    """A single eval configuration to run.

    The ``model`` field is the canonical identity used for JSONL row keys,
    resume matching, and display labels:
      - ollama: Ollama-style string (e.g. "qwen3:8b-q8_0")
      - llamaserver: GGUF stem (e.g. "Qwen3-8B-Q8_0")
      - llamafile: llamafile binary stem (e.g. "Mistral-Nemo-Instruct-2407.Q4_K_M")

    ``gguf_filename`` is the on-disk filename for llamaserver/llamafile
    backends (joined with ``models_dir`` to form the path passed to the
    server and to ``LlamafileClient(gguf_path=...)``). None for ollama.
    """

    model: str
    backend: str  # "ollama" | "llamaserver" | "llamafile"
    mode: str  # "native" | "prompt"
    think: bool | None  # None = auto
    tool_choice: str | None = None
    gguf_filename: str | None = None  # llamaserver/llamafile only
    server_recipe: _BatchServerRecipe = _DEFAULT_SERVER_RECIPE
    # Reasoning-effort axis. reasoning_level records the effective tested level
    # so variants of the same stem coexist in the resume key + report. "default"
    # means no explicit level was recorded (and is the legacy missing-field
    # fallback), not "look up the current registry value". sampling_override,
    # when set, bypasses recommended sampling in _build_client and passes an
    # explicit param set (its keys match LlamafileClient kwargs).
    reasoning_level: str = "default"
    sampling_override: dict[str, Any] | None = None


# Ollama configs: 10 instruct models, native FC, stream
OLLAMA_CONFIGS: list[BatchConfig] = [
    BatchConfig(model=m, backend="ollama", mode="native", think=None)
    for m in [
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
]

# llama-server configs: each GGUF × 2 modes (native + prompt), with native
# skipped for models in _PROMPT_ONLY_MODELS (no native FC training).
LLAMASERVER_CONFIGS: list[BatchConfig] = []
for _filename, _server_recipe in _GGUF_FILES:
    # Strip a multi-shard suffix (e.g. "-00001-of-00002") so a sharded model
    # keys on its clean stem for config/flags/sampling/row-identity, while
    # gguf_filename keeps the first-shard name that llama-server loads from.
    _stem = re.sub(r"-\d{5}-of-\d{5}$", "", Path(_filename).stem)
    if _stem not in _PROMPT_ONLY_MODELS:
        LLAMASERVER_CONFIGS.append(
            BatchConfig(
                model=_stem, backend="llamaserver", mode="native",
                think=None, gguf_filename=_filename,
                server_recipe=_server_recipe,
                reasoning_level=_EFFECTIVE_REASONING_LEVELS.get(_stem, "default"),
            )
        )
    LLAMASERVER_CONFIGS.append(
        BatchConfig(
            model=_stem, backend="llamaserver", mode="prompt",
            think=None, gguf_filename=_filename,
            server_recipe=_server_recipe,
            reasoning_level=_EFFECTIVE_REASONING_LEVELS.get(_stem, "default"),
        )
    )

# Llamafile binary configs: prompt only (no native FC support)
LLAMAFILE_CONFIGS: list[BatchConfig] = [
    BatchConfig(
        model=Path(filename).stem, backend="llamafile", mode="prompt",
        think=None, gguf_filename=filename,
    )
    for filename in _LLAMAFILE_FILES
]

ALL_CONFIGS: list[BatchConfig] = (
    LLAMASERVER_CONFIGS + LLAMAFILE_CONFIGS + OLLAMA_CONFIGS
)

# New models wired for the az/evals sweep (16GB tier): LFM2.5 MoE + Mellum2 MoE
# (both variants), each native + prompt. Subset lets a run target only these.
_NEW_MODEL_STEMS: set[str] = {
    "LFM2.5-8B-A1B-Q4_K_M",
    "Mellum2-12B-A2.5B-Thinking-Q4_K_M",
    "Mellum2-12B-A2.5B-Instruct-Q4_K_M",
}
NEW_MODEL_CONFIGS: list[BatchConfig] = [
    c for c in LLAMASERVER_CONFIGS if c.model in _NEW_MODEL_STEMS
]

QWEN38_CONFIGS: list[BatchConfig] = [
    c for c in LLAMASERVER_CONFIGS
    if c.model == "Qwen3.8-27B-UD-Q4_K_XL" and c.mode == "native"
]

_QWEN38_EFFORT_CONFIGS: dict[str, list[BatchConfig]] = {
    effort: [
        replace(
            QWEN38_CONFIGS[0],
            reasoning_level=effort,
            sampling_override={
                **get_sampling_defaults("Qwen3.8-27B-UD-Q4_K_XL"),
                "chat_template_kwargs": {"reasoning_effort": effort},
            },
        )
    ]
    for effort in ("medium", "low")
}

# Reasoning-effort axis (rig-03): gpt-oss@high + nemotron@high as PARALLEL
# configs to the medium/low-effort baselines. Same GGUF + native FC, but
# recommended sampling is bypassed (sampling_override) so chat_template_kwargs
# swaps the effort knob to HIGH while get_sampling_defaults preserves the rest
# of the registry baseline (temp/top_p/...). reasoning_level="high" tags the
# rows so they coexist with the explicitly stamped baseline rows in the resume
# key + report instead of colliding (silent-skip). Kept OUT of
# LLAMASERVER_CONFIGS / "all" so only an explicit --config reasoning-high runs
# them.
_REASONING_HIGH_CONFIGS: list[BatchConfig] = [
    BatchConfig(
        model="gpt-oss-120b-Q4_K_M", backend="llamaserver", mode="native",
        think=None, gguf_filename="gpt-oss-120b-Q4_K_M-00001-of-00002.gguf",
        server_recipe=_GPT_OSS_120B_SERVER_RECIPE,
        reasoning_level="high",
        sampling_override={
            **get_sampling_defaults("gpt-oss-120b-Q4_K_M"),
            "chat_template_kwargs": {"reasoning_effort": "high"},
        },
    ),
    BatchConfig(
        model="NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M", backend="llamaserver",
        mode="native", think=None,
        gguf_filename="NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00001-of-00003.gguf",
        server_recipe=_LARGE_120B_SERVER_RECIPE,
        reasoning_level="high",
        sampling_override={
            **get_sampling_defaults("NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M"),
            "chat_template_kwargs": {
                "enable_thinking": True, "low_effort": False, "force_nonempty_content": True,
            },
        },
    ),
]

_GLIMMER_EFFORT_CONFIGS: dict[str, list[BatchConfig]] = {
    effort: [
        BatchConfig(
            model="Muse-Glimmer-30B-UD-Q4_K_XL",
            backend="llamaserver",
            mode="native",
            think=None,
            gguf_filename="Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
            server_recipe=_glimmer_server_recipe(effort),
            reasoning_level=effort,
        )
    ]
    for effort in ("high", "medium", "low")
}

# Explicit large-model campaign. Machine-local RPC values are attached from
# --rpc-topology at invocation time; reasoning effort comes only from the
# sampling-registry row for this model.
DEEPSEEK_V4_RPC_CONFIGS: list[BatchConfig] = [
    BatchConfig(
        model=_DEEPSEEK_V4_MODEL,
        backend="llamaserver",
        mode="native",
        think=None,
        gguf_filename=_DEEPSEEK_V4_GGUF,
        server_recipe=_DEEPSEEK_V4_RPC_SERVER_RECIPE,
        reasoning_level=_DEEPSEEK_V4_REASONING_LEVEL,
    ),
]

# Named subsets for quick iteration
CONFIG_SETS: dict[str, list[BatchConfig]] = {
    "all": ALL_CONFIGS,
    "ollama": OLLAMA_CONFIGS,
    "llamaserver": LLAMASERVER_CONFIGS,
    "llamafile": LLAMAFILE_CONFIGS,
    "llamaserver-native": [c for c in LLAMASERVER_CONFIGS if c.mode == "native"],
    "llamaserver-prompt": [c for c in LLAMASERVER_CONFIGS if c.mode == "prompt"],
    "reasoning-high": _REASONING_HIGH_CONFIGS,
    "glimmer-high": _GLIMMER_EFFORT_CONFIGS["high"],
    "glimmer-medium": _GLIMMER_EFFORT_CONFIGS["medium"],
    "glimmer-low": _GLIMMER_EFFORT_CONFIGS["low"],
    "deepseek-v4-rpc": DEEPSEEK_V4_RPC_CONFIGS,
    "qwen38": QWEN38_CONFIGS,
    "qwen38-medium": _QWEN38_EFFORT_CONFIGS["medium"],
    "qwen38-low": _QWEN38_EFFORT_CONFIGS["low"],
    "new-models": NEW_MODEL_CONFIGS,
    "new-models-native": [c for c in NEW_MODEL_CONFIGS if c.mode == "native"],
    "new-models-prompt": [c for c in NEW_MODEL_CONFIGS if c.mode == "prompt"],
}


def _load_rpc_topology(path: Path) -> LlamaCppRpcConfig:
    """Load one explicit batch RPC topology from a small JSON file."""
    topology_data = dict(json.loads(path.read_text(encoding="utf-8")))
    worker_data = dict(topology_data.pop("worker"))
    if "environment" in worker_data:
        worker_data["environment"] = tuple(
            tuple(pair) for pair in worker_data["environment"]
        )
    if "ssh_options" in worker_data:
        worker_data["ssh_options"] = tuple(worker_data["ssh_options"])
    worker = LlamaCppRpcWorkerConfig(**worker_data)

    topology_data["worker"] = worker
    topology_data["devices"] = tuple(topology_data["devices"])
    topology_data["tensor_split"] = tuple(topology_data["tensor_split"])
    if "coordinator_environment" in topology_data:
        topology_data["coordinator_environment"] = tuple(
            tuple(pair) for pair in topology_data["coordinator_environment"]
        )
    return LlamaCppRpcConfig(**topology_data)


def _attach_deepseek_rpc_topology(
    configs: list[BatchConfig], rpc: LlamaCppRpcConfig,
) -> list[BatchConfig]:
    """Attach machine-local RPC values without mutating the registry config."""
    return [
        replace(
            config,
            server_recipe=replace(config.server_recipe, rpc=rpc),
        )
        if config.model == _DEEPSEEK_V4_MODEL else config
        for config in configs
    ]


def _resolve_server_recipe_flags(
    config: BatchConfig,
    models_dir: Path,
) -> list[str]:
    """Resolve one recipe's runtime paths and literal llama-server flags."""
    flags: list[str] = []
    if config.server_recipe.draft_filename is not None:
        flags.extend([
            "--model-draft",
            str(models_dir / config.server_recipe.draft_filename),
        ])
    flags.extend(config.server_recipe.extra_flags)
    return flags


def _print_rpc_recipe(
    config: BatchConfig,
    models_dir: Path,
    budget_mode: BudgetMode,
    manual_tokens: int | None,
) -> None:
    """Render the complete normalized launch recipe for an RPC batch config."""
    rpc = config.server_recipe.rpc
    assert rpc is not None
    assert config.gguf_filename is not None
    artifact = str(models_dir / config.gguf_filename)

    sampling = get_sampling_defaults(config.model)
    template_kwargs = sampling.get("chat_template_kwargs", {})
    effort = (
        template_kwargs.get("reasoning_effort")
        if isinstance(template_kwargs, dict) else None
    )

    coordinator_command = [
        "env",
        *(f"{key}={value}" for key, value in rpc.coordinator_environment),
        rpc.coordinator_executable,
        "-m", artifact,
        "-ngl", "999",
        "--port", str(_eval_port()),
        *render_rpc_coordinator_args(rpc),
    ]
    if config.mode == "native":
        coordinator_command.append("--jinja")
    coordinator_command.extend(_resolve_server_recipe_flags(config, models_dir))
    if budget_mode == BudgetMode.MANUAL and manual_tokens is not None:
        coordinator_command.extend(["-c", str(manual_tokens)])

    print("  Managed RPC recipe:")
    print(f"    model: {config.model}")
    print(f"    reasoning effort (sampling registry): {effort}")
    print(f"    worker command: {shlex.join(render_rpc_worker_command(rpc.worker))}")
    print(f"    coordinator command: {shlex.join(coordinator_command)}")
    print(f"    startup timeout: {rpc.startup_timeout:g}s")
    print(f"    log directory: {rpc.log_directory or '<temporary>'}")


# ── Anthropic pricing (USD per million tokens) ──────────────────

_ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    # model_id: (input_per_mtok, output_per_mtok)
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-6": (5.0, 25.0),
    # Opus 4.8 standard mode: $5 input / $25 output per Mtok (anthropic.com,
    # confirmed 2026-06). Same as 4-6. (Fast mode is 2× — $10/$50 — not used here.)
    "claude-opus-4-8": (5.0, 25.0),
}

# Prompt-cache token multipliers on the input rate, uniform across current
# Anthropic models: writes bill 1.25×, reads bill 0.1×.
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.1


def _compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Compute USD cost from token counts. Returns 0.0 for unknown models.

    ``input_tokens`` is the *uncached* input sliver; cached writes/reads are
    priced separately off the input rate so prompt caching is reflected
    accurately (the API reports these as distinct usage fields).
    """
    rates = _ANTHROPIC_PRICING.get(model)
    if not rates:
        return 0.0
    input_rate, output_rate = rates
    return (
        input_tokens * input_rate
        + cache_creation_tokens * input_rate * _CACHE_WRITE_MULTIPLIER
        + cache_read_tokens * input_rate * _CACHE_READ_MULTIPLIER
        + output_tokens * output_rate
    ) / 1_000_000


# ── JSONL helpers ───────────────────────────────────────────────


def _config_key(model: str, backend: str, mode: str) -> str:
    """Canonical key for resume matching."""
    return f"{model}|{backend}|{mode}"


def _run_key(
    model: str,
    backend: str,
    mode: str,
    ablation_name: str,
    tool_choice: str,
    reasoning_replay: str,
    reasoning_level: str,
    scenario: str,
) -> str:
    """Canonical per-run resume key.

    Single source of truth for the resume/dedup dimensions so the counting
    pass and every run-loop lookup stay in lockstep. reasoning_replay is part
    of the key: distinct policies (none/keep-last/full) on the same
    model+scenario are independent runs and must not collide. reasoning_level
    ("default"/"high"/...) is likewise part of the key so effort variants of one
    stem coexist instead of clobbering.
    """
    return (
        f"{model}|{backend}|{mode}"
        f"|{ablation_name}|{tool_choice}|{reasoning_replay}|{reasoning_level}|{scenario}"
    )


def _validate_generation(generation: Any, *, context: str = "generation") -> int:
    """Return a valid eval generation or fail closed."""
    if type(generation) is not int or generation < 0:
        raise ValueError(
            f"{context} must be a non-negative integer (bool is not allowed), "
            f"got {generation!r}"
        )
    return generation


def _parse_generation(value: str) -> int:
    """Argparse type for non-negative eval generations."""
    try:
        generation = int(value)
    except ValueError as exc:
        raise ValueError("generation must be a non-negative integer") from exc
    return _validate_generation(generation)


@dataclass(frozen=True)
class ResumePreflight:
    recorded_counts: dict[str, int]
    outcome_dialect: OutcomeDialect


def _preflight_recorded_runs(
    jsonl_path: Path,
    requested_generation: int,
    ablation_name: str = "reforged",
) -> ResumePreflight:
    """Validate append compatibility and count runs in one streaming pass.

    Every stored row participates in the single-generation check, while only
    rows for ``ablation_name`` contribute to the resume map. Historical rows
    without ``gen`` are generation 0 and rows without ``reasoning_replay`` used
    the historical ``full`` behavior.
    """
    requested_generation = _validate_generation(requested_generation)
    counts: dict[str, int] = {}
    if not jsonl_path.exists():
        return ResumePreflight(counts, CANONICAL_DIALECT)

    file_generation: int | None = None
    file_dialect: OutcomeDialect | None = None
    with jsonl_path.open("rb") as f:
        for line_number, raw_line in enumerate(f, 1):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"{jsonl_path}:{line_number}: invalid UTF-8 JSONL row"
                ) from exc
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{jsonl_path}:{line_number}: malformed JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{jsonl_path}:{line_number}: JSONL row must be an object"
                )
            try:
                row_dialect = detect_outcome_dialect(row)
                read_outcome(row, expected_dialect=row_dialect)
            except OutcomeSchemaError as exc:
                raise ValueError(
                    f"{jsonl_path}:{line_number}: invalid outcome schema: {exc}"
                ) from exc
            if file_dialect is None:
                file_dialect = row_dialect
            elif row_dialect != file_dialect:
                raise ValueError(
                    f"{jsonl_path}:{line_number}: mixed outcome dialects "
                    f"{file_dialect!r} and {row_dialect!r}"
                )
            try:
                row_generation = _validate_generation(
                    effective_generation(row), context="gen"
                )
            except ValueError as exc:
                raise ValueError(f"{jsonl_path}:{line_number}: {exc}") from exc
            if file_generation is None:
                file_generation = row_generation
            elif row_generation != file_generation:
                raise ValueError(
                    f"{jsonl_path}:{line_number}: mixed effective generations "
                    f"{file_generation} and {row_generation}"
                )

            row_ablation = row.get("ablation", "reforged")
            if row_ablation != ablation_name:
                continue
            try:
                key = _run_key(
                    row["model"], row["backend"], row["mode"],
                    row_ablation, row.get("tool_choice", "auto"),
                    effective_reasoning_replay(row),
                    row.get("reasoning_level", "default"), row["scenario"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{jsonl_path}:{line_number}: cannot build resume key: {exc}"
                ) from exc
            counts[key] = counts.get(key, 0) + 1

    if file_generation is not None and file_generation != requested_generation:
        raise ValueError(
            f"{jsonl_path}: existing effective generation {file_generation} "
            f"does not match requested generation {requested_generation}"
        )
    return ResumePreflight(counts, file_dialect or CANONICAL_DIALECT)


def _append_jsonl_row(jsonl_path: Path, row: dict[str, Any]) -> None:
    """Append one UTF-8 row, separating an unterminated existing final row."""
    payload = (json.dumps(row) + "\n").encode("utf-8")
    with jsonl_path.open("ab+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        separator = b""
        if size:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                separator = b"\n"
        f.seek(0, os.SEEK_END)
        f.write(separator + payload)


def _run_result_to_row(
    result: RunResult,
    config: BatchConfig,
    scenario: EvalScenario,
    run_idx: int,
    *,
    generation: int,
    budget_tokens: int | None = None,
    ablation_name: str = "reforged",
    reasoning_replay: str = DEFAULT_REASONING_REPLAY,
    outcome_dialect: OutcomeDialect = CANONICAL_DIALECT,
) -> dict[str, Any]:
    """Convert a RunResult into a flat dict for JSONL output."""
    generation = _validate_generation(generation)
    row: dict[str, Any] = {
        "gen": generation,
        "model": config.model,
        "backend": config.backend,
        "mode": config.mode,
        "ablation": ablation_name,
        "tool_choice": config.tool_choice or "auto",
        "reasoning_replay": reasoning_replay,
        "reasoning_level": config.reasoning_level,
        "scenario": result.scenario_name,
        "run": run_idx,
        "iterations": result.iterations_used,
        "elapsed_s": round(result.elapsed_seconds, 2),
        "error_type": result.error_type,
        "error_message": result.error_message,
        "compaction_events": len(result.compaction_events),
    }
    if budget_tokens is not None:
        row["budget_tokens"] = budget_tokens
    if result.stream_retries > 0:
        row["stream_retries"] = result.stream_retries

    # History-based stats
    if result.messages is not None:
        stats = analyze_history(result.messages)
        row["retry_nudges"] = stats.retry_nudges
        row["step_nudges"] = stats.step_nudges
        row["tool_errors"] = stats.tool_errors
        row["reasoning_msgs"] = stats.reasoning_messages
        # On-wire reasoning that survives the replay policy (independent
        # validation of the knob): none->0, keep-last->{0,1}, full->[0,total].
        # reasoning_wire_total is the denominator (non-empty reasoning blocks),
        # so reasoning_wire / reasoning_wire_total is the actual replay rate.
        wire_survived, wire_total = count_wire_reasoning(result.messages, reasoning_replay)
        row["reasoning_wire"] = wire_survived
        row["reasoning_wire_total"] = wire_total
    else:
        row["retry_nudges"] = None
        row["step_nudges"] = None
        row["tool_errors"] = None
        row["reasoning_msgs"] = None
        row["reasoning_wire"] = None
        row["reasoning_wire_total"] = None

    row.update(
        write_outcome(
            RunOutcome(
                correct=result.correct,
                completed=result.completed,
                validation_error=result.validation_error,
            ),
            outcome_dialect,
        )
    )

    # Wasted calls
    ideal = scenario.ideal_iterations or (len(scenario.workflow.required_steps) + 1)
    row["ideal_iterations"] = ideal
    if result.completed:
        row["wasted_calls"] = max(0, result.iterations_used - ideal)
    else:
        row["wasted_calls"] = None

    # Token usage and cost (Anthropic only — local backends report 0)
    if (
        result.input_tokens or result.output_tokens
        or result.cache_creation_tokens or result.cache_read_tokens
    ):
        row["input_tokens"] = result.input_tokens
        row["output_tokens"] = result.output_tokens
        row["cache_creation_input_tokens"] = result.cache_creation_tokens
        row["cache_read_input_tokens"] = result.cache_read_tokens
        row["cost_usd"] = round(
            _compute_cost(
                config.model, result.input_tokens, result.output_tokens,
                result.cache_creation_tokens, result.cache_read_tokens,
            ),
            6,
        )

    return row


# ── Model availability ──────────────────────────────────────────


def _ollama_models() -> set[str]:
    """Return set of locally available Ollama model names."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    models: set[str] = set()
    for line in result.stdout.strip().splitlines()[1:]:  # skip header
        name = line.split()[0] if line.strip() else ""
        if name:
            models.add(name)
    return models


def _check_model_available(
    config: "BatchConfig", models_dir: Path,
) -> str | None:
    """Return a skip reason if the model isn't available, or None if ready."""
    if config.backend in ("llamaserver", "llamafile"):
        if not config.gguf_filename:
            return f"no GGUF/llamafile filename on config for {config.model}"
        if not (models_dir / config.gguf_filename).exists():
            return f"file not found: {models_dir / config.gguf_filename}"
        draft_filename = config.server_recipe.draft_filename
        if draft_filename and not (models_dir / draft_filename).exists():
            return f"draft file not found: {models_dir / draft_filename}"
    elif config.backend == "ollama":
        available = _ollama_models()
        if config.model not in available:
            return f"not in ollama list"
    return None


# ── Run-level timeout ───────────────────────────────────────────

# Wall-clock cap per scenario run. p99 in historical data is ~38s and the
# longest legitimate 12GB-tier run is ~100s, so 300s is a safe hang guard.
_REQUEST_TIMEOUT = 300.0
_RUN_TIMEOUT = 300.0


async def _run_with_timeout(
    client: Any,
    scenario: EvalScenario,
    eval_config: EvalConfig,
    ablation: AblationConfig | None,
    run_timeout: float = _RUN_TIMEOUT,
) -> RunResult:
    """Run a scenario with a wall-clock cap.

    On timeout, synthesizes a failed RunResult with error_type='Timeout' so
    the batch keeps moving. No retry — one strike and we record the miss.
    """
    start = time.monotonic()
    try:
        return await asyncio.wait_for(
            run_scenario(client, scenario, eval_config, ablation=ablation),
            timeout=run_timeout,
        )
    except asyncio.TimeoutError:
        return RunResult(
            scenario_name=scenario.name,
            completed=False,
            iterations_used=0,
            error_type="Timeout",
            error_message=f"Exceeded {run_timeout:g}s",
            elapsed_seconds=time.monotonic() - start,
        )


# ── Server recovery ─────────────────────────────────────────────

_RECOVERY_BACKOFFS = [30, 60, 300]  # seconds: 30s, 60s, 5min


_INFRA_ERRORS = ("ConnectError", "RemoteProtocolError", "ReadTimeout", "WriteTimeout", "PoolTimeout")


def _is_server_error(result: "RunResult") -> bool:
    """Check if a run result indicates a server-side infrastructure failure."""
    if not result.error_message:
        return False
    return any(e in result.error_message for e in _INFRA_ERRORS)


async def _recover_server(
    server: "ServerManager",
    crash_count: int,
    budget_mode: BudgetMode,
    manual_tokens: int | None,
) -> int | None:
    """Attempt to restart the server after a crash.

    Restarts the manager's last successful launch, preserving its resolved
    topology and context arguments.

    Returns the recovered budget, or ``None`` if recovery failed or the circuit
    breaker tripped.
    """
    if crash_count > len(_RECOVERY_BACKOFFS):
        return None

    backoff = _RECOVERY_BACKOFFS[crash_count - 1]
    print(
        f"\n  [!] Server error detected (attempt {crash_count}/{len(_RECOVERY_BACKOFFS)}). "
        f"Waiting {backoff}s before restart...",
        flush=True,
    )

    # Kill any lingering process
    try:
        await server.stop()
    except Exception:
        pass

    await asyncio.sleep(backoff)

    try:
        await server.restart()
        resolved_budget = await server.resolve_budget(budget_mode, manual_tokens)
        print("  [!] Server restarted successfully.", flush=True)
        return resolved_budget
    except Exception as exc:
        print(f"  [!] Server restart failed: {exc}", flush=True)
        return None


# ── Client factory ──────────────────────────────────────────────


def _build_client(
    config: BatchConfig,
    models_dir: Path,
    base_url: str,
    request_timeout: float = _REQUEST_TIMEOUT,
) -> Any:
    """Build the appropriate LLM client for a BatchConfig.

    For llamaserver/llamafile, ``gguf_path`` is constructed from
    ``models_dir / config.gguf_filename``.
    """
    think_val = config.think
    recommended_sampling = config.model not in _NO_RECOMMENDED_SAMPLING_MODELS

    if config.backend == "ollama":
        from forge.clients.ollama import OllamaClient

        return OllamaClient(
            model=config.model, base_url=base_url, think=think_val,
            timeout=request_timeout,
            recommended_sampling=recommended_sampling,
        )

    elif config.backend == "llamaserver":
        from forge.clients.llamafile import LlamafileClient

        assert config.gguf_filename, f"llamaserver config missing gguf_filename: {config.model}"
        # Explicit per-config sampling (reasoning-effort axis): opt out of the
        # registry recommended row and pass the full param set. Its keys match
        # the constructor kwargs (temperature/top_p/top_k/min_p/
        # chat_template_kwargs); caller-explicit values win field-by-field and
        # chat_template_kwargs whole-replaces (llamafile.py:366-380).
        if config.sampling_override is not None:
            return LlamafileClient(
                gguf_path=str(models_dir / config.gguf_filename),
                mode=config.mode, think=think_val,
                base_url=base_url,
                timeout=request_timeout,
                recommended_sampling=False,
                **config.sampling_override,
            )
        return LlamafileClient(
            gguf_path=str(models_dir / config.gguf_filename),
            mode=config.mode, think=think_val,
            base_url=base_url,
            timeout=request_timeout,
            recommended_sampling=recommended_sampling,
        )

    elif config.backend == "llamafile":
        from forge.clients.llamafile import LlamafileClient

        assert config.gguf_filename, f"llamafile config missing gguf_filename: {config.model}"
        return LlamafileClient(
            gguf_path=str(models_dir / config.gguf_filename),
            mode=config.mode,
            think=think_val,
            base_url=base_url,
            timeout=request_timeout,
            recommended_sampling=recommended_sampling,
        )

    else:
        raise ValueError(f"Unknown backend: {config.backend}")


def _format_eta(total_ran: int, total_expected: int, batch_start: float) -> str:
    """Format a batch ETA string from run counts and start time."""
    if total_ran == 0 or total_expected <= total_ran:
        return ""
    elapsed = time.monotonic() - batch_start
    rate = total_ran / elapsed
    remaining = int((total_expected - total_ran) / rate)
    days, remainder = divmod(remaining, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        ts = f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        ts = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f" (batch ETA: {ts})"


# ── Main batch loop ─────────────────────────────────────────────


async def run_batch(
    configs: list[BatchConfig],
    runs_per_scenario: int,
    output_path: Path,
    models_dir: Path = MODELS_DIR_DEFAULT,
    dry_run: bool = False,
    verbose: bool = False,
    budget_mode: BudgetMode = BudgetMode.FORGE_FULL,
    manual_tokens: int | None = None,
    tags: list[str] | None = None,
    scenario_names: list[str] | None = None,
    ablation: AblationConfig | None = None,
    reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
    generation: int = 0,
    request_timeout: float = _REQUEST_TIMEOUT,
    run_timeout: float = _RUN_TIMEOUT,
) -> None:
    """Run all configs × scenarios, appending each result to JSONL.

    Budget resolution uses the public ``setup_backend()`` path. Compaction
    scenarios (compaction_stress, phase2_compaction) always override
    with their own hardcoded budget.
    """
    from forge.context.strategies import TieredCompact
    from tests.eval.eval_runner import _COMPACTION_SCENARIOS

    generation = _validate_generation(generation)
    supported_backends = {"ollama", "llamaserver", "llamafile"}
    unsupported_backends = sorted({
        config.backend for config in configs
        if config.backend not in supported_backends
    })
    if unsupported_backends:
        raise ValueError(
            "batch_eval supports only managed backends "
            f"{sorted(supported_backends)}; unsupported: {unsupported_backends}"
        )
    if any(
        config.model == _DEEPSEEK_V4_MODEL
        and config.server_recipe.rpc is None
        for config in configs
    ):
        raise ValueError(
            "DeepSeek V4 RPC batch config requires an attached RPC topology"
        )

    if scenario_names:
        name_set = set(scenario_names)
        scenarios = [s for s in ALL_SCENARIOS if s.name in name_set]
        missing = name_set - {s.name for s in scenarios}
        if missing:
            raise RuntimeError(f"Unknown scenarios: {', '.join(sorted(missing))}")
    elif tags:
        scenarios = [s for s in ALL_SCENARIOS if any(t in s.tags for t in tags)]
        if not scenarios:
            raise RuntimeError(f"No scenarios match tags: {tags}")
    else:
        scenarios = ALL_SCENARIOS

    ablation_name = ablation.name if ablation is not None else "reforged"
    preflight = _preflight_recorded_runs(
        output_path, generation, ablation_name=ablation_name
    )
    recorded_counts = preflight.recorded_counts
    outcome_dialect = preflight.outcome_dialect

    # Precompute total expected runs (excluding skips and unavailable models)
    total_expected = 0
    for config in configs:
        if _check_model_available(config, models_dir) is not None:
            continue
        tc_label_pre = config.tool_choice or "auto"
        for scenario in scenarios:
            skip_compaction = (
                ablation is not None and not ablation.compaction_enabled
            )
            if scenario.name in _COMPACTION_SCENARIOS and skip_compaction:
                continue
            key = _run_key(
                config.model, config.backend, config.mode,
                ablation_name, tc_label_pre, reasoning_replay,
                config.reasoning_level, scenario.name,
            )
            existing = recorded_counts.get(key, 0)
            total_expected += max(0, runs_per_scenario - existing)

    total_configs = len(configs)
    total_scenarios = len(scenarios)
    total_skipped = 0
    total_ran = 0
    total_failed_connect = 0
    batch_start = time.monotonic()
    for cfg_idx, config in enumerate(configs, 1):
        tc_label = config.tool_choice or "auto"
        config_label = f"{config.model} ({config.backend}/{config.mode})"
        if config.tool_choice:
            config_label += f" [tool_choice={config.tool_choice}]"
        print(
            f"\n{'='*70}\n"
            f"[{cfg_idx}/{total_configs}] {config_label}\n"
            f"{'='*70}",
            flush=True,
        )
        if (
            config.model == _DEEPSEEK_V4_MODEL
            and config.server_recipe.rpc is not None
        ):
            _print_rpc_recipe(config, models_dir, budget_mode, manual_tokens)

        # ── Dry run ───────────────────────────────────────
        if dry_run:
            for scenario in scenarios:
                skip_compaction = (
                    ablation is not None and not ablation.compaction_enabled
                )
                if scenario.name in _COMPACTION_SCENARIOS and skip_compaction:
                    print(f"  {scenario.name}: SKIP (compaction N/A)")
                    continue
                key = _run_key(
                    config.model, config.backend, config.mode,
                    ablation_name, tc_label, reasoning_replay,
                    config.reasoning_level, scenario.name,
                )
                existing = recorded_counts.get(key, 0)
                remaining = max(0, runs_per_scenario - existing)
                status = "SKIP" if remaining == 0 else f"RUN {remaining}"
                print(f"  {scenario.name}: {existing}/{runs_per_scenario} done -> {status}")
            continue

        # ── Model availability check ────────────────────
        skip_reason = _check_model_available(config, models_dir)
        if skip_reason:
            print(f"  SKIP ({skip_reason})", flush=True)
            total_skipped += total_scenarios
            continue

        # ── Check if any scenarios need runs ─────────────
        has_work = False
        for scenario in scenarios:
            skip_compaction = (
                ablation is not None and not ablation.compaction_enabled
            )
            if scenario.name in _COMPACTION_SCENARIOS and skip_compaction:
                continue
            key_check = _run_key(
                config.model, config.backend, config.mode,
                ablation_name, tc_label, reasoning_replay,
                config.reasoning_level, scenario.name,
            )
            if recorded_counts.get(key_check, 0) < runs_per_scenario:
                has_work = True
                break
        if not has_work:
            print("  SKIP (all requested attempts recorded)", flush=True)
            total_skipped += total_scenarios
            continue

        # Resolve GGUF/llamafile path for non-Ollama backends
        gguf_path: str | None = None
        if config.backend in ("llamaserver", "llamafile"):
            assert config.gguf_filename, f"missing gguf_filename: {config.model}"
            gguf_path = str(models_dir / config.gguf_filename)

        server_flags = _resolve_server_recipe_flags(config, models_dir)
        try:
            server, setup_context = await setup_backend(
                backend=config.backend,
                model=config.model if config.backend == "ollama" else None,
                gguf_path=gguf_path,
                mode=config.mode,
                port=_eval_port(),
                budget_mode=budget_mode,
                manual_tokens=manual_tokens,
                extra_flags=server_flags or None,
                rpc=config.server_recipe.rpc,
            )
        except RuntimeError:
            print(f"  SKIP (server failed to start)", flush=True)
            total_skipped += total_scenarios
            continue

        try:
            resolved_budget = setup_context.budget_tokens
            # Build one client for this isolated managed configuration.
            client = _build_client(
                config,
                models_dir,
                server.client_base_url,
                request_timeout,
            )
            if hasattr(client, "set_num_ctx"):
                client.set_num_ctx(resolved_budget)

            crash_count = 0
            config_aborted = False

            for sc_idx, scenario in enumerate(scenarios, 1):
                if config_aborted:
                    break

                # Skip compaction scenarios when ablation disables compaction
                if scenario.name in _COMPACTION_SCENARIOS and ablation is not None and not ablation.compaction_enabled:
                    total_skipped += 1
                    continue

                key = _run_key(
                    config.model, config.backend, config.mode,
                    ablation_name, tc_label, reasoning_replay,
                    config.reasoning_level, scenario.name,
                )
                existing = recorded_counts.get(key, 0)
                remaining = max(0, runs_per_scenario - existing)

                if remaining == 0:
                    total_skipped += 1
                    continue

                # Compaction scenarios use their own hardcoded budget
                if scenario.name in _COMPACTION_SCENARIOS:
                    scenario_budget = scenario.budget_tokens
                else:
                    scenario_budget = resolved_budget

                if hasattr(client, "set_num_ctx"):
                    client.set_num_ctx(scenario_budget)

                eval_config = EvalConfig(
                    runs_per_scenario=1,  # we loop ourselves
                    stream=True,
                    keep_message_history=True,
                    verbose=verbose,
                    budget_override=scenario_budget,
                    strategy_overrides={"compaction": TieredCompact(keep_recent=2)},
                    reasoning_replay=reasoning_replay,
                )

                eta = _format_eta(total_ran, total_expected, batch_start)
                print(
                    f"\n  [{sc_idx}/{total_scenarios}] {scenario.name} "
                    f"- {existing} done, running {remaining} more{eta}",
                    flush=True,
                )

                for run_idx in range(existing, existing + remaining):
                    result = await _run_with_timeout(
                        client, scenario, eval_config, ablation, run_timeout
                    )
                    total_ran += 1

                    # Server crash recovery
                    if _is_server_error(result):
                        crash_count += 1
                        print(
                            f"    run {run_idx+1}/{runs_per_scenario}: "
                            f"CRASH ({result.error_message.split(':')[0]})",
                            flush=True,
                        )
                        recovered_budget = await _recover_server(
                            server, crash_count, budget_mode, manual_tokens
                        )
                        if recovered_budget is None:
                            print(
                                f"\n  [!] Circuit breaker: {crash_count} crashes "
                                f"for {config_label}. Skipping remaining scenarios.",
                                flush=True,
                            )
                            config_aborted = True
                            break

                        # Rebuild client and retry the failed run
                        client = _build_client(
                            config,
                            models_dir,
                            server.client_base_url,
                            request_timeout,
                        )
                        resolved_budget = recovered_budget
                        if hasattr(client, "set_num_ctx"):
                            client.set_num_ctx(scenario_budget)

                        result = await _run_with_timeout(
                            client, scenario, eval_config, ablation, run_timeout
                        )
                        total_ran += 1

                    status = "OK" if result.completed else f"FAIL ({result.error_type})"
                    print(
                        f"    run {run_idx+1}/{runs_per_scenario}: {status} "
                        f"- {result.iterations_used} iters, "
                        f"{result.elapsed_seconds:.1f}s",
                        flush=True,
                    )

                    row = _run_result_to_row(
                        result, config, scenario, run_idx + 1,
                        generation=generation,
                        budget_tokens=scenario_budget,
                        ablation_name=ablation_name,
                        reasoning_replay=reasoning_replay,
                        outcome_dialect=outcome_dialect,
                    )
                    _append_jsonl_row(output_path, row)

                    # Update in-memory count for resume correctness
                    recorded_counts[key] = recorded_counts.get(key, 0) + 1
        finally:
            await server.stop()

    elapsed = time.monotonic() - batch_start
    print(
        f"\n{'='*70}\n"
        f"Batch complete - {total_ran} runs executed, "
        f"{total_skipped} scenario-slots skipped (already done), "
        f"{total_failed_connect} configs skipped (connection failed)\n"
        f"Total time: {elapsed/60:.1f} min\n"
        f"Results: {output_path}\n"
        f"{'='*70}",
        flush=True,
    )


# ── CLI ─────────────────────────────────────────────────────────


async def main() -> None:
    import argparse

    budget_choices = [m.value for m in BudgetMode]
    parser = argparse.ArgumentParser(description="Forge batch eval runner")
    parser.add_argument("--runs", type=int, default=50, help="Runs per scenario")
    parser.add_argument(
        "--generation",
        type=_parse_generation,
        default=0,
        help="Non-negative eval comparability generation (default: 0)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="JSONL output path"
    )
    parser.add_argument(
        "--config",
        choices=list(CONFIG_SETS.keys()),
        default="all",
        help="Which config set to run",
    )
    parser.add_argument(
        "--rpc-topology",
        type=str,
        default=None,
        help="JSON topology file required by --config deepseek-v4-rpc.",
    )
    parser.add_argument(
        "--scenario", nargs="*",
        help="Run specific scenarios by name (e.g. --scenario basic_2step sequential_reasoning)",
    )
    parser.add_argument(
        "--budget-mode",
        choices=budget_choices,
        default=BudgetMode.FORGE_FULL.value,
        help="Budget mode (prod BudgetMode). Compaction scenarios always override with their own budget.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Exact token budget (requires --budget-mode manual).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=_REQUEST_TIMEOUT,
        help="Per-request client timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=_RUN_TIMEOUT,
        help="Whole scenario-run timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--tags", nargs="*",
        help="Filter scenarios by tag (e.g. --tags plumbing model_quality)",
    )
    parser.add_argument(
        "--ablation",
        choices=list(ABLATION_PRESETS.keys()),
        default="reforged",
        help="Ablation preset: selectively disable guardrails (default: reforged = all enabled)",
    )
    parser.add_argument(
        "--reasoning-replay",
        choices=list(REASONING_REPLAY_CHOICES),
        default=DEFAULT_REASONING_REPLAY,
        help="How much captured reasoning to replay to the backend each turn: "
        "full (legacy), keep-last, none (default). Part of the resume key, so "
        "distinct policies for the same model/scenario are independent runs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter configs to models containing this substring (e.g. --model 8b-reasoning)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing GGUF and llamafile model files (default: models)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    budget_mode = BudgetMode(args.budget_mode)
    if budget_mode == BudgetMode.MANUAL and args.num_ctx is None:
        parser.error("--budget-mode manual requires --num-ctx")

    configs = CONFIG_SETS[args.config]
    if args.model:
        configs = [c for c in configs if args.model in c.model]
        if not configs:
            parser.error(f"No configs match --model '{args.model}' in set '{args.config}'")
    if args.config == "deepseek-v4-rpc":
        if args.rpc_topology is None:
            parser.error("--config deepseek-v4-rpc requires --rpc-topology")
        try:
            rpc_topology = _load_rpc_topology(Path(args.rpc_topology))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            parser.error(f"cannot load --rpc-topology: {exc}")
        configs = _attach_deepseek_rpc_topology(configs, rpc_topology)
    elif args.rpc_topology is not None:
        parser.error("--rpc-topology is only valid with --config deepseek-v4-rpc")
    output_path = Path(args.output) if args.output else Path("eval_results.jsonl")

    if args.scenario:
        scenario_count = len(args.scenario)
    elif args.tags:
        scenario_count = sum(1 for s in ALL_SCENARIOS if any(t in s.tags for t in args.tags))
    else:
        scenario_count = len(ALL_SCENARIOS)
    ablation = ABLATION_PRESETS[args.ablation]

    print(f"Forge Batch Eval")
    print(f"  Config set:    {args.config} ({len(configs)} configs)")
    print(f"  Budget mode:   {budget_mode.value}")
    print(f"  Ablation:      {ablation.name}")
    print(f"  Reasoning replay: {args.reasoning_replay}")
    print(f"  Generation:     {args.generation}")
    if args.scenario:
        print(f"  Scenarios:     {', '.join(args.scenario)}")
    elif args.tags:
        print(f"  Tags filter:   {', '.join(args.tags)}")
    print(f"  Scenarios:     {scenario_count}")
    print(f"  Runs/scenario: {args.runs}")
    print(f"  Output:        {output_path}")
    print(f"  Models dir:    {args.models_dir}")
    if args.rpc_topology:
        print(f"  RPC topology:  {args.rpc_topology}")
    print(f"  Total max runs: {len(configs) * scenario_count * args.runs}")

    models_dir = Path(args.models_dir)

    await run_batch(
        configs=configs,
        runs_per_scenario=args.runs,
        output_path=output_path,
        models_dir=models_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
        budget_mode=budget_mode,
        manual_tokens=args.num_ctx,
        tags=args.tags,
        scenario_names=args.scenario,
        ablation=ablation,
        reasoning_replay=args.reasoning_replay,
        generation=args.generation,
        request_timeout=args.request_timeout,
        run_timeout=args.run_timeout,
    )


if __name__ == "__main__":
    asyncio.run(main())
