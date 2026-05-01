"""Batch eval runner - iterate model/backend/mode configs, append JSONL.

Usage:
    python -m tests.eval.batch_eval [--runs 50] [--output results.jsonl]
                                     [--config CONFIG_NAME] [--dry-run]

Resumes automatically: for each (model, backend, mode, scenario) it counts
existing completed runs in the JSONL and only runs the remainder.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.server import BudgetMode, ServerManager

from tests.eval.ablation import ABLATION_PRESETS, AblationConfig
from tests.eval.eval_runner import EvalConfig, RunResult, run_scenario
from tests.eval.metrics import analyze_history, compute_metrics
from tests.eval.scenarios import ALL_SCENARIOS, EvalScenario

# ── GGUF paths ──────────────────────────────────────────────────

MODELS_DIR_DEFAULT = Path("models")

# GGUF and llamafile model files for local-server backends.
# Each entry is just the filename — paired into a BatchConfig below
# alongside the canonical identity (the file stem, no extension).
_GGUF_FILES: list[str] = [
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
    "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
    "Qwen3-8B-Q4_K_M.gguf",
    "Qwen3-8B-Q8_0.gguf",
    "Qwen3-14B-Q4_K_M.gguf",
    "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
    "Ministral-3-8B-Instruct-2512-Q8_0.gguf",
    "Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
    "Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf",
    "Ministral-3-8B-Reasoning-2512-Q8_0.gguf",
    "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf",
    # 32GB tier
    "gemma-4-31B-it-Q4_K_M.gguf",
    "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    "gemma-4-26B-A4B-it-Q8_0.gguf",
    "gemma-4-E4B-it-Q4_K_M.gguf",
    "gemma-4-E4B-it-Q8_0.gguf",
    "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
    "Mistral-Small-3.2-24B-Instruct-2506-Q8_0.gguf",
    "Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf",
    "Devstral-Small-2-24B-Instruct-2512-Q8_0.gguf",
    "Qwen3.5-27B-Q4_K_M.gguf",
    "Qwen3.5-35B-A3B-Q4_K_M.gguf",
    "granite-4.0-h-micro-Q4_K_M.gguf",
    "granite-4.0-h-micro-Q8_0.gguf",
    "granite-4.0-h-tiny-Q4_K_M.gguf",
    "granite-4.0-h-tiny-Q8_0.gguf",
]

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
      - anthropic: model ID (e.g. "claude-haiku-4-5-20251001")

    ``gguf_filename`` is the on-disk filename for llamaserver/llamafile
    backends (joined with ``models_dir`` to form the path passed to the
    server and to ``LlamafileClient(gguf_path=...)``). None for
    ollama/anthropic.
    """

    model: str
    backend: str  # "ollama" | "llamaserver" | "llamafile" | "anthropic"
    mode: str  # "native" | "prompt"
    think: bool | None  # None = auto
    tool_choice: str | None = None  # Anthropic only: "auto", "any"
    gguf_filename: str | None = None  # llamaserver/llamafile only


# Ollama configs: 11 instruct models, native FC, stream
OLLAMA_CONFIGS: list[BatchConfig] = [
    BatchConfig(model=m, backend="ollama", mode="native", think=None)
    for m in [
        "llama3.1:8b-instruct-q4_K_M",
        "llama3.1:8b-instruct-q8_0",
        "mistral-nemo:12b-instruct-2407-q4_K_M",
        "mistral:7b-instruct-v0.3-q4_K_M",
        "mistral:7b-instruct-v0.3-q8_0",
        "qwen3:8b-q4_K_M",
        "qwen3:8b-q8_0",
        "qwen3:14b-q4_K_M",
        "ministral-3:8b-instruct-2512-q4_K_M",
        "ministral-3:8b-instruct-2512-q8_0",
        "ministral-3:14b-instruct-2512-q4_K_M",
        # ── 32GB eval additions ──
        "gemma4:31b-it-q4_K_M",
        "gemma4:26b-a4b-it-q4_K_M",
        # gemma4:26b-a4b-it-q8_0 excluded — spills to CPU on Ollama (28GB weights)
        "gemma4:e4b-it-q4_K_M",
        "gemma4:e4b-it-q8_0",
        "qwen3.5:27b-q4_K_M",
        "qwen3.5:35b-a3b-q4_K_M",
        # Granite 4.0 (IBM) — q4 only on Ollama; matches LS coverage on rig-03
        "granite-4.0:h-micro-q4_K_M",
        "granite-4.0:h-tiny-q4_K_M",
    ]
]

# llama-server configs: each GGUF × 2 modes (native + prompt)
LLAMASERVER_CONFIGS: list[BatchConfig] = []
for _filename in _GGUF_FILES:
    _stem = Path(_filename).stem
    LLAMASERVER_CONFIGS.append(
        BatchConfig(
            model=_stem, backend="llamaserver", mode="native",
            think=None, gguf_filename=_filename,
        )
    )
    LLAMASERVER_CONFIGS.append(
        BatchConfig(
            model=_stem, backend="llamaserver", mode="prompt",
            think=None, gguf_filename=_filename,
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

ANTHROPIC_CONFIGS: list[BatchConfig] = [
    BatchConfig(model="claude-haiku-4-5-20251001", backend="anthropic", mode="native", think=None),
    BatchConfig(model="claude-sonnet-4-6", backend="anthropic", mode="native", think=None),
    BatchConfig(model="claude-opus-4-6", backend="anthropic", mode="native", think=None),
]

ANTHROPIC_ANY_CONFIGS: list[BatchConfig] = [
    BatchConfig(model="claude-haiku-4-5-20251001", backend="anthropic", mode="native", think=None, tool_choice="any"),
    BatchConfig(model="claude-sonnet-4-6", backend="anthropic", mode="native", think=None, tool_choice="any"),
    BatchConfig(model="claude-opus-4-6", backend="anthropic", mode="native", think=None, tool_choice="any"),
]

ALL_CONFIGS: list[BatchConfig] = (
    LLAMASERVER_CONFIGS + LLAMAFILE_CONFIGS + OLLAMA_CONFIGS
)

# Named subsets for quick iteration
# Note: "anthropic" is separate from "all" — it costs money per API call.
CONFIG_SETS: dict[str, list[BatchConfig]] = {
    "all": ALL_CONFIGS,
    "ollama": OLLAMA_CONFIGS,
    "llamaserver": LLAMASERVER_CONFIGS,
    "llamafile": LLAMAFILE_CONFIGS,
    "llamaserver-native": [c for c in LLAMASERVER_CONFIGS if c.mode == "native"],
    "llamaserver-prompt": [c for c in LLAMASERVER_CONFIGS if c.mode == "prompt"],
    "anthropic": ANTHROPIC_CONFIGS,
    "anthropic-any": ANTHROPIC_ANY_CONFIGS,
    "haiku": [c for c in ANTHROPIC_CONFIGS if "haiku" in c.model],
    "sonnet": [c for c in ANTHROPIC_CONFIGS if "sonnet" in c.model],
    "opus": [c for c in ANTHROPIC_CONFIGS if "opus" in c.model],
    "haiku-any": [c for c in ANTHROPIC_ANY_CONFIGS if "haiku" in c.model],
    "sonnet-any": [c for c in ANTHROPIC_ANY_CONFIGS if "sonnet" in c.model],
    "opus-any": [c for c in ANTHROPIC_ANY_CONFIGS if "opus" in c.model],
}


# ── Anthropic pricing (USD per million tokens) ──────────────────

_ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    # model_id: (input_per_mtok, output_per_mtok)
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-6": (5.0, 25.0),
}


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost from token counts. Returns 0.0 for unknown models."""
    rates = _ANTHROPIC_PRICING.get(model)
    if not rates:
        return 0.0
    input_rate, output_rate = rates
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


# ── JSONL helpers ───────────────────────────────────────────────


def _config_key(model: str, backend: str, mode: str) -> str:
    """Canonical key for resume matching."""
    return f"{model}|{backend}|{mode}"


def _count_completed_runs(
    jsonl_path: Path,
    ablation_name: str = "reforged",
) -> dict[str, int]:
    """Scan JSONL and count completed runs per (model, backend, mode, ablation, tool_choice, scenario).

    Returns dict mapping "model|backend|mode|ablation|tool_choice|scenario" → count.
    Records without an ablation field are treated as "reforged".
    Records without a tool_choice field are treated as "auto".
    """
    counts: dict[str, int] = {}
    if not jsonl_path.exists():
        return counts
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_ablation = row.get("ablation", "reforged")
            if row_ablation != ablation_name:
                continue
            row_tc = row.get("tool_choice", "auto")
            key = (
                f"{row['model']}|{row['backend']}|{row['mode']}"
                f"|{row_ablation}|{row_tc}|{row['scenario']}"
            )
            counts[key] = counts.get(key, 0) + 1
    return counts


def _run_result_to_row(
    result: RunResult,
    config: BatchConfig,
    scenario: EvalScenario,
    run_idx: int,
    budget_tokens: int | None = None,
    ablation_name: str = "reforged",
) -> dict[str, Any]:
    """Convert a RunResult into a flat dict for JSONL output."""
    row: dict[str, Any] = {
        "model": config.model,
        "backend": config.backend,
        "mode": config.mode,
        "ablation": ablation_name,
        "tool_choice": config.tool_choice or "auto",
        "scenario": result.scenario_name,
        "run": run_idx,
        "completeness": result.completeness,
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
    else:
        row["retry_nudges"] = None
        row["step_nudges"] = None
        row["tool_errors"] = None
        row["reasoning_msgs"] = None

    # Correctness
    row["accuracy"] = result.accuracy
    if result.validate_error:
        row["validate_error"] = result.validate_error

    # Wasted calls
    ideal = scenario.ideal_iterations or (len(scenario.workflow.required_steps) + 1)
    row["ideal_iterations"] = ideal
    if result.completeness:
        row["wasted_calls"] = max(0, result.iterations_used - ideal)
    else:
        row["wasted_calls"] = None

    # Token usage and cost (Anthropic only — local backends report 0)
    if result.input_tokens or result.output_tokens:
        row["input_tokens"] = result.input_tokens
        row["output_tokens"] = result.output_tokens
        row["cost_usd"] = round(
            _compute_cost(config.model, result.input_tokens, result.output_tokens),
            6,
        )

    return row


# ── llama-server flags ───────────────────────────────────────────

# Extra flags per model for llama-server, keyed by config.model (the GGUF
# stem for llamaserver configs).
# Qwen3 models: --reasoning-format auto (server-side <think> tag parsing)
# Everything else: no extra flags needed.
_SERVER_EXTRA_FLAGS: dict[str, list[str]] = {
    "Qwen3-8B-Q4_K_M": ["--reasoning-format", "auto"],
    "Qwen3-8B-Q8_0": ["--reasoning-format", "auto"],
    "Qwen3-14B-Q4_K_M": ["--reasoning-format", "auto"],
    "Qwen3.5-27B-Q4_K_M": ["--reasoning-format", "auto"],
    "Qwen3.5-35B-A3B-Q4_K_M": ["--reasoning-format", "auto"],
}


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
    elif config.backend == "ollama":
        available = _ollama_models()
        if config.model not in available:
            return f"not in ollama list"
    return None


def _get_server_flags(model: str, mode: str) -> list[str]:
    """Build llama-server CLI flags for a given model and mode."""
    flags: list[str] = []
    if mode == "native":
        flags.append("--jinja")
    flags.extend(_SERVER_EXTRA_FLAGS.get(model, []))
    return flags


# ── Run-level timeout ───────────────────────────────────────────

# Wall-clock cap per scenario run. p99 in historical data is ~38s and the
# longest legitimate 12GB-tier run is ~100s, so 300s is a safe hang guard.
_RUN_TIMEOUT = 300


async def _run_with_timeout(
    client: Any,
    scenario: EvalScenario,
    eval_config: EvalConfig,
    ablation: AblationConfig | None,
) -> RunResult:
    """Run a scenario with a wall-clock cap.

    On timeout, synthesizes a failed RunResult with error_type='Timeout' so
    the batch keeps moving. No retry — one strike and we record the miss.
    """
    start = time.monotonic()
    try:
        return await asyncio.wait_for(
            run_scenario(client, scenario, eval_config, ablation=ablation),
            timeout=_RUN_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return RunResult(
            scenario_name=scenario.name,
            completeness=False,
            iterations_used=0,
            error_type="Timeout",
            error_message=f"Exceeded {_RUN_TIMEOUT}s",
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
    config: BatchConfig,
    gguf_path: str,
    extra_flags: list[str] | None,
    crash_count: int,
) -> bool:
    """Attempt to restart the server after a crash.

    Returns True if recovery succeeded, False if circuit breaker tripped.
    """
    if crash_count > len(_RECOVERY_BACKOFFS):
        return False

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

    # Restart. Cache-equality identity: model string for ollama,
    # GGUF path for non-Ollama (matches run_batch and setup_backend).
    cache_identity = config.model if config.backend == "ollama" else gguf_path
    try:
        await server.start(
            model=cache_identity,
            gguf_path=gguf_path,
            mode=config.mode,
            extra_flags=extra_flags,
        )
        print("  [!] Server restarted successfully.", flush=True)
        return True
    except Exception as exc:
        print(f"  [!] Server restart failed: {exc}", flush=True)
        return False


# ── Client factory ──────────────────────────────────────────────


def _build_client(config: BatchConfig, models_dir: Path) -> Any:
    """Build the appropriate LLM client for a BatchConfig.

    For llamaserver/llamafile, ``gguf_path`` is constructed from
    ``models_dir / config.gguf_filename``.
    """
    think_val = config.think

    if config.backend == "ollama":
        from forge.clients.ollama import OllamaClient

        return OllamaClient(
            model=config.model, think=think_val,
            recommended_sampling=True,
        )

    elif config.backend == "llamaserver":
        from forge.clients.llamafile import LlamafileClient

        assert config.gguf_filename, f"llamaserver config missing gguf_filename: {config.model}"
        return LlamafileClient(
            gguf_path=str(models_dir / config.gguf_filename),
            mode=config.mode, think=think_val,
            recommended_sampling=True,
        )

    elif config.backend == "llamafile":
        from forge.clients.llamafile import LlamafileClient

        assert config.gguf_filename, f"llamafile config missing gguf_filename: {config.model}"
        return LlamafileClient(
            gguf_path=str(models_dir / config.gguf_filename),
            mode=config.mode,
            think=think_val,
            base_url="http://localhost:8080/v1",
            recommended_sampling=True,
        )

    elif config.backend == "anthropic":
        from forge.clients.anthropic import AnthropicClient

        return AnthropicClient(model=config.model, tool_choice=config.tool_choice)

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
) -> None:
    """Run all configs × scenarios, appending each result to JSONL.

    Budget resolution uses the prod ServerManager path. Compaction
    scenarios (compaction_stress, phase2_compaction) always override
    with their own hardcoded budget.
    """
    from forge.context.strategies import TieredCompact
    from tests.eval.eval_runner import _COMPACTION_SCENARIOS

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
    completed_counts = _count_completed_runs(output_path, ablation_name=ablation_name)

    # Precompute total expected runs (excluding skips and unavailable models)
    total_expected = 0
    for config in configs:
        if _check_model_available(config, models_dir) is not None:
            continue
        tc_label_pre = config.tool_choice or "auto"
        for scenario in scenarios:
            skip_compaction = (
                config.backend == "anthropic"
                or (ablation is not None and not ablation.compaction_enabled)
            )
            if scenario.name in _COMPACTION_SCENARIOS and skip_compaction:
                continue
            key = (
                f"{config.model}|{config.backend}|{config.mode}"
                f"|{ablation_name}|{tc_label_pre}|{scenario.name}"
            )
            existing = completed_counts.get(key, 0)
            total_expected += max(0, runs_per_scenario - existing)

    total_configs = len(configs)
    total_scenarios = len(scenarios)
    total_skipped = 0
    total_ran = 0
    total_failed_connect = 0
    batch_start = time.monotonic()
    server = ServerManager(backend="ollama", port=8080, models_dir=models_dir)
    prev_backend: str | None = None
    prev_server: ServerManager | None = None

    try:
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

            # ── Dry run ───────────────────────────────────────
            if dry_run:
                for scenario in scenarios:
                    skip_compaction = (
                        config.backend == "anthropic"
                        or (ablation is not None and not ablation.compaction_enabled)
                    )
                    if scenario.name in _COMPACTION_SCENARIOS and skip_compaction:
                        print(f"  {scenario.name}: SKIP (compaction N/A)")
                        continue
                    key = (
                        f"{config.model}|{config.backend}|{config.mode}"
                        f"|{ablation_name}|{tc_label}|{scenario.name}"
                    )
                    existing = completed_counts.get(key, 0)
                    remaining = max(0, runs_per_scenario - existing)
                    status = "SKIP" if remaining == 0 else f"RUN {remaining}"
                    print(f"  {scenario.name}: {existing}/{runs_per_scenario} done -> {status}")
                continue

            # ── Model availability check ────────────────────
            skip_reason = _check_model_available(config, models_dir)
            if skip_reason:
                # Stop Ollama before skipping so VRAM is clear for later configs
                if prev_backend == "ollama" and config.backend != "ollama":
                    if prev_server is not None:
                        await prev_server.stop()
                    prev_backend = None
                print(f"  SKIP ({skip_reason})", flush=True)
                total_skipped += total_scenarios
                continue

            # ── Anthropic cloud API path ─────────────────────
            # No server management, no GGUF, no VRAM budget.
            if config.backend == "anthropic":
                client = _build_client(config, models_dir)

                for sc_idx, scenario in enumerate(scenarios, 1):
                    if scenario.name in _COMPACTION_SCENARIOS:
                        total_skipped += 1
                        continue

                    key = (
                        f"{config.model}|{config.backend}|{config.mode}"
                        f"|{ablation_name}|{tc_label}|{scenario.name}"
                    )
                    existing = completed_counts.get(key, 0)
                    remaining = max(0, runs_per_scenario - existing)

                    if remaining == 0:
                        total_skipped += 1
                        continue

                    scenario_budget = scenario.budget_tokens

                    eval_config = EvalConfig(
                        runs_per_scenario=1,
                        stream=True,
                        keep_message_history=True,
                        verbose=verbose,
                        budget_override=scenario_budget,
                    )

                    eta = _format_eta(total_ran, total_expected, batch_start)
                    print(
                        f"\n  [{sc_idx}/{total_scenarios}] {scenario.name} "
                        f"- {existing} done, running {remaining} more{eta}",
                        flush=True,
                    )

                    for run_idx in range(existing, existing + remaining):
                        result = await _run_with_timeout(client, scenario, eval_config, ablation)
                        total_ran += 1
                        status = "OK" if result.completeness else f"FAIL ({result.error_type})"
                        print(
                            f"    run {run_idx+1}/{runs_per_scenario}: {status} "
                            f"- {result.iterations_used} iters, "
                            f"{result.elapsed_seconds:.1f}s",
                            flush=True,
                        )

                        row = _run_result_to_row(
                            result, config, scenario, run_idx + 1,
                            budget_tokens=scenario_budget,
                            ablation_name=ablation_name,
                        )
                        with output_path.open("a") as f:
                            f.write(json.dumps(row) + "\n")

                        completed_counts[key] = completed_counts.get(key, 0) + 1
                continue

            # ── Check if any scenarios need runs ─────────────
            has_work = False
            for scenario in scenarios:
                skip_compaction = (
                    ablation is not None and not ablation.compaction_enabled
                )
                if scenario.name in _COMPACTION_SCENARIOS and skip_compaction:
                    continue
                key_check = (
                    f"{config.model}|{config.backend}|{config.mode}"
                    f"|{ablation_name}|{tc_label}|{scenario.name}"
                )
                if completed_counts.get(key_check, 0) < runs_per_scenario:
                    has_work = True
                    break
            if not has_work:
                print(f"  SKIP (all scenarios complete)", flush=True)
                total_skipped += total_scenarios
                continue

            # ── Local backend path (server-managed) ──────────

            # Clean VRAM unload when switching away from Ollama
            if prev_backend == "ollama" and config.backend != "ollama":
                if prev_server is not None:
                    await prev_server.stop()

            # Create new ServerManager if backend changed
            if config.backend != prev_backend:
                if prev_server is not None and prev_backend != "ollama":
                    await prev_server.stop()
                server = ServerManager(
                    backend=config.backend, port=8080, models_dir=models_dir
                )

            # Resolve GGUF/llamafile path for non-Ollama backends
            gguf_path = ""
            if config.backend in ("llamaserver", "llamafile"):
                assert config.gguf_filename, f"missing gguf_filename: {config.model}"
                gguf_path = str(models_dir / config.gguf_filename)

            # Start server and get extra flags. For non-Ollama backends pass
            # the GGUF path as the cache-equality key (matches setup_backend
            # convention from server.py); for Ollama, pass the model string.
            extra_flags = _get_server_flags(config.model, config.mode)
            cache_identity = config.model if config.backend == "ollama" else gguf_path
            try:
                await server.start(
                    model=cache_identity,
                    gguf_path=gguf_path,
                    mode=config.mode,
                    extra_flags=extra_flags if extra_flags else None,
                )
            except RuntimeError:
                # Startup timeout — attempt recovery
                recovered = await _recover_server(
                    server, config, gguf_path,
                    extra_flags if extra_flags else None,
                    crash_count=1,
                )
                if not recovered:
                    print(f"  SKIP (server failed to start)", flush=True)
                    total_skipped += total_scenarios
                    continue

            prev_backend = config.backend
            prev_server = server

            # Build client
            client = _build_client(config, models_dir)

            # Resolve budget through prod ServerManager path
            resolved_budget = await server.resolve_budget(budget_mode, manual_tokens)
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

                key = (
                    f"{config.model}|{config.backend}|{config.mode}"
                    f"|{ablation_name}|{tc_label}|{scenario.name}"
                )
                existing = completed_counts.get(key, 0)
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
                )

                eta = _format_eta(total_ran, total_expected, batch_start)
                print(
                    f"\n  [{sc_idx}/{total_scenarios}] {scenario.name} "
                    f"- {existing} done, running {remaining} more{eta}",
                    flush=True,
                )

                for run_idx in range(existing, existing + remaining):
                    result = await _run_with_timeout(client, scenario, eval_config, ablation)
                    total_ran += 1

                    # Server crash recovery
                    if _is_server_error(result):
                        crash_count += 1
                        print(
                            f"    run {run_idx+1}/{runs_per_scenario}: "
                            f"CRASH ({result.error_message.split(':')[0]})",
                            flush=True,
                        )
                        recovered = await _recover_server(
                            server, config, gguf_path,
                            extra_flags if extra_flags else None,
                            crash_count,
                        )
                        if not recovered:
                            print(
                                f"\n  [!] Circuit breaker: {crash_count} crashes "
                                f"for {config_label}. Skipping remaining scenarios.",
                                flush=True,
                            )
                            config_aborted = True
                            break

                        # Rebuild client and retry the failed run
                        client = _build_client(config, models_dir)
                        resolved_budget = await server.resolve_budget(budget_mode, manual_tokens)
                        if hasattr(client, "set_num_ctx"):
                            client.set_num_ctx(scenario_budget)

                        result = await _run_with_timeout(client, scenario, eval_config, ablation)
                        total_ran += 1

                    status = "OK" if result.completeness else f"FAIL ({result.error_type})"
                    print(
                        f"    run {run_idx+1}/{runs_per_scenario}: {status} "
                        f"- {result.iterations_used} iters, "
                        f"{result.elapsed_seconds:.1f}s",
                        flush=True,
                    )

                    row = _run_result_to_row(
                        result, config, scenario, run_idx + 1,
                        budget_tokens=scenario_budget,
                        ablation_name=ablation_name,
                    )
                    with output_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")

                    # Update in-memory count for resume correctness
                    completed_counts[key] = completed_counts.get(key, 0) + 1

            # Free VRAM after finishing all scenarios for this Ollama config
            if config.backend == "ollama":
                await server.stop()
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
        "--output", type=str, default=None, help="JSONL output path"
    )
    parser.add_argument(
        "--config",
        choices=list(CONFIG_SETS.keys()),
        default="all",
        help="Which config set to run",
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
    if args.scenario:
        print(f"  Scenarios:     {', '.join(args.scenario)}")
    elif args.tags:
        print(f"  Tags filter:   {', '.join(args.tags)}")
    print(f"  Scenarios:     {scenario_count}")
    print(f"  Runs/scenario: {args.runs}")
    print(f"  Output:        {output_path}")
    print(f"  Models dir:    {args.models_dir}")
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
    )


if __name__ == "__main__":
    asyncio.run(main())
