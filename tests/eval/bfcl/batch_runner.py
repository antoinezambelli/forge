"""BFCL batch runner — run categories across model/backend/mode configs.

Mirrors the design of tests.eval.batch_eval: ablation is a CLI arg applied
uniformly, configs are model/backend/mode only, and the run loop handles
server lifecycle + resume.

Usage:
    python -m tests.eval.bfcl.batch_runner --config CONFIG [OPTIONS]

Examples:
    # Dry-run to see what would run
    python -m tests.eval.bfcl.batch_runner --config llamaserver-native --dry-run

    # Run simple_python only, bare ablation
    python -m tests.eval.bfcl.batch_runner --config llamaserver-native --ablation bare --categories simple_python

    # Filter to a specific model
    python -m tests.eval.bfcl.batch_runner --config llamaserver --model ministral-3:8b-instruct --categories simple_python
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.server import BudgetMode, ServerManager

from tests.eval.ablation import ABLATION_PRESETS, AblationConfig
from tests.eval.bfcl.runner import (
    BfclMultiTurnResult,
    BfclRunResult,
    run_multi_turn_entry,
    run_single_entry,
)
from tests.eval.bfcl.scorer import (
    ScoreResult,
    score_entry,
    score_multi_turn_entry,
)
from tests.eval.bfcl.schema_adapter import (
    CATEGORY_FILES,
    DATA_DIR,
    MULTI_TURN_CATEGORIES,
    load_jsonl,
    load_multi_turn_ground_truth,
)


# ── GGUF paths ──────────────────────────────────────────────────

MODELS_DIR = Path(os.environ.get("FORGE_MODELS_DIR", "models"))

GGUF_MAP: dict[str, str] = {
    "llama3.1:8b-instruct-q4_K_M": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "llama3.1:8b-instruct-q8_0": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    "mistral-nemo:12b-instruct-2407-q4_K_M": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
    "mistral:7b-instruct-v0.3-q4_K_M": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    "mistral:7b-instruct-v0.3-q8_0": "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
    "qwen3:8b-q4_K_M": "Qwen3-8B-Q4_K_M.gguf",
    "qwen3:8b-q8_0": "Qwen3-8B-Q8_0.gguf",
    "qwen3:14b-q4_K_M": "Qwen3-14B-Q4_K_M.gguf",
    "ministral-3:8b-instruct-2512-q4_K_M": "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
    "ministral-3:8b-instruct-2512-q8_0": "Ministral-3-8B-Instruct-2512-Q8_0.gguf",
    "ministral-3:14b-instruct-2512-q4_K_M": "Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
    "ministral-3:8b-reasoning-2512-q4_K_M": "Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf",
    "ministral-3:8b-reasoning-2512-q8_0": "Ministral-3-8B-Reasoning-2512-Q8_0.gguf",
    "ministral-3:14b-reasoning-2512-q4_K_M": "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf",
}

LLAMAFILE_MAP: dict[str, str] = {
    "llama3.1:8b-instruct-q4_K_M": "Meta-Llama-3.1-8B-Instruct.Q4_K_M.llamafile",
    "llama3.1:8b-instruct-q8_0": "Meta-Llama-3.1-8B-Instruct.Q8_0.llamafile",
    "mistral-nemo:12b-instruct-2407-q4_K_M": "Mistral-Nemo-Instruct-2407.Q4_K_M.llamafile",
    "mistral:7b-instruct-v0.3-q4_K_M": "Mistral-7B-Instruct-v0.3.Q4_K_M.llamafile",
    "mistral:7b-instruct-v0.3-q8_0": "Mistral-7B-Instruct-v0.3.Q8_0.llamafile",
}


# ── llama-server flags ──────────────────────────────────────────

_SERVER_EXTRA_FLAGS: dict[str, list[str]] = {
    "qwen3:8b-q4_K_M": ["--reasoning-format", "auto"],
    "qwen3:8b-q8_0": ["--reasoning-format", "auto"],
    "qwen3:14b-q4_K_M": ["--reasoning-format", "auto"],
}


def _get_server_flags(model: str, mode: str) -> list[str]:
    flags: list[str] = []
    if mode == "native":
        flags.append("--jinja")
    flags.extend(_SERVER_EXTRA_FLAGS.get(model, []))
    return flags


# ── Config definitions ──────────────────────────────────────────


@dataclass
class BfclBatchConfig:
    """One BFCL eval configuration (no ablation — that's a CLI arg)."""

    model: str
    backend: str   # "ollama" | "llamaserver" | "llamafile" | "anthropic"
    mode: str      # "native" | "prompt"
    think: bool | None = None  # None = auto
    tool_choice: str | None = None


# Ollama: representative sample, native FC
OLLAMA_CONFIGS: list[BfclBatchConfig] = [
    BfclBatchConfig(model=m, backend="ollama", mode="native")
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
    ]
]

# llama-server: all GGUFs × native + prompt
_LLAMASERVER_MODELS = list(GGUF_MAP.keys())

LLAMASERVER_CONFIGS: list[BfclBatchConfig] = []
for _m in _LLAMASERVER_MODELS:
    LLAMASERVER_CONFIGS.append(BfclBatchConfig(model=_m, backend="llamaserver", mode="native"))
    LLAMASERVER_CONFIGS.append(BfclBatchConfig(model=_m, backend="llamaserver", mode="prompt"))

# Llamafile: prompt only
LLAMAFILE_CONFIGS: list[BfclBatchConfig] = [
    BfclBatchConfig(model=m, backend="llamafile", mode="prompt")
    for m in LLAMAFILE_MAP
]

# Anthropic cloud
ANTHROPIC_CONFIGS: list[BfclBatchConfig] = [
    BfclBatchConfig(model="claude-haiku-4-5-20251001", backend="anthropic", mode="native"),
    BfclBatchConfig(model="claude-sonnet-4-6", backend="anthropic", mode="native"),
    BfclBatchConfig(model="claude-opus-4-6", backend="anthropic", mode="native"),
]

ANTHROPIC_ANY_CONFIGS: list[BfclBatchConfig] = [
    BfclBatchConfig(model="claude-haiku-4-5-20251001", backend="anthropic", mode="native", tool_choice="any"),
    BfclBatchConfig(model="claude-sonnet-4-6", backend="anthropic", mode="native", tool_choice="any"),
    BfclBatchConfig(model="claude-opus-4-6", backend="anthropic", mode="native", tool_choice="any"),
]

ALL_CONFIGS: list[BfclBatchConfig] = (
    OLLAMA_CONFIGS + LLAMASERVER_CONFIGS + LLAMAFILE_CONFIGS
)

CONFIG_SETS: dict[str, list[BfclBatchConfig]] = {
    "all": ALL_CONFIGS,
    "ollama": OLLAMA_CONFIGS,
    "llamaserver": LLAMASERVER_CONFIGS,
    "llamaserver-native": [c for c in LLAMASERVER_CONFIGS if c.mode == "native"],
    "llamaserver-prompt": [c for c in LLAMASERVER_CONFIGS if c.mode == "prompt"],
    "llamafile": LLAMAFILE_CONFIGS,
    "anthropic": ANTHROPIC_CONFIGS,
    "anthropic-any": ANTHROPIC_ANY_CONFIGS,
    "haiku": [c for c in ANTHROPIC_CONFIGS if "haiku" in c.model],
    "sonnet": [c for c in ANTHROPIC_CONFIGS if "sonnet" in c.model],
    "opus": [c for c in ANTHROPIC_CONFIGS if "opus" in c.model],
}

# All categories
SINGLE_TURN_CATEGORIES = list(CATEGORY_FILES.keys())
MULTI_TURN_CATS = list(MULTI_TURN_CATEGORIES.keys())
ALL_CATEGORIES = SINGLE_TURN_CATEGORIES + MULTI_TURN_CATS

DEFAULT_OUTPUT = Path("bfcl_results.jsonl")


# ── JSONL helpers ───────────────────────────────────────────────


def _entry_key(
    model: str, backend: str, mode: str,
    ablation: str, tool_choice: str,
    category: str, test_id: str,
) -> str:
    return f"{model}|{backend}|{mode}|{ablation}|{tool_choice}|{category}|{test_id}"


def _load_completed(jsonl_path: Path, ablation_name: str) -> set[str]:
    """Load completed entry keys from JSONL, filtered to ablation."""
    completed: set[str] = set()
    if not jsonl_path.exists():
        return completed
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
            key = _entry_key(
                row["model"], row["backend"], row["mode"],
                row_ablation, row.get("tool_choice", "auto"),
                row["category"], row["test_id"],
            )
            completed.add(key)
    return completed


def _result_to_row(
    config: BfclBatchConfig,
    ablation_name: str,
    category: str,
    test_id: str,
    valid: bool,
    error_type: str,
    errors: list[str],
    elapsed_s: float,
    iterations: int,
    completed: bool,
) -> dict[str, Any]:
    return {
        "model": config.model,
        "backend": config.backend,
        "mode": config.mode,
        "ablation": ablation_name,
        "tool_choice": config.tool_choice or "auto",
        "category": category,
        "test_id": test_id,
        "valid": valid,
        "completed": completed,
        "error_type": error_type,
        "errors": errors,
        "elapsed_s": round(elapsed_s, 2),
        "iterations": iterations,
    }


# ── Client factory ──────────────────────────────────────────────


def _build_client(config: BfclBatchConfig):
    """Build the appropriate LLM client for a BfclBatchConfig."""
    think_val = config.think

    if config.backend == "ollama":
        from forge.clients.ollama import OllamaClient

        return OllamaClient(model=config.model, think=think_val)

    elif config.backend == "llamaserver":
        from forge.clients.llamafile import LlamafileClient

        return LlamafileClient(
            model=config.model, mode=config.mode, think=think_val
        )

    elif config.backend == "llamafile":
        from forge.clients.llamafile import LlamafileClient

        return LlamafileClient(
            model=config.model,
            mode=config.mode,
            think=think_val,
            base_url="http://localhost:8080/v1",
        )

    elif config.backend == "anthropic":
        from forge.clients.anthropic import AnthropicClient

        return AnthropicClient(model=config.model, tool_choice=config.tool_choice)

    else:
        raise ValueError(f"Unknown backend: {config.backend}")


# ── Iteration limits ────────────────────────────────────────────


def _max_iters_single(ablation: AblationConfig) -> int:
    if ablation.max_retries_per_step == 0 and ablation.max_tool_errors == 0:
        return 8
    return 15


def _max_iters_multi(ablation: AblationConfig) -> int:
    if ablation.max_retries_per_step == 0 and ablation.max_tool_errors == 0:
        return 10
    return 20


# ── Core run loop ───────────────────────────────────────────────


async def run_batch(
    configs: list[BfclBatchConfig],
    categories: list[str],
    output_path: Path,
    ablation: AblationConfig,
    dry_run: bool = False,
    verbose: bool = False,
    budget_mode: BudgetMode = BudgetMode.FORGE_FULL,
    manual_tokens: int | None = None,
) -> None:
    """Run all configs × categories, appending each result to JSONL.

    Budget resolution uses the prod ServerManager path.
    """
    ablation_name = ablation.name
    completed_keys = _load_completed(output_path, ablation_name)

    total_configs = len(configs)
    total_entries = 0
    total_skipped = 0
    total_ran = 0
    total_valid = 0
    batch_start = time.monotonic()

    server = ServerManager(backend="ollama", port=8080, models_dir=MODELS_DIR)
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

            # ── Dry run ──────────────────────────────────────
            if dry_run:
                for category in categories:
                    is_multi = category in MULTI_TURN_CATEGORIES
                    filename = MULTI_TURN_CATEGORIES[category] if is_multi else CATEGORY_FILES[category]
                    entries = load_jsonl(DATA_DIR / filename)
                    done = 0
                    pending = 0
                    for entry in entries:
                        key = _entry_key(
                            config.model, config.backend, config.mode,
                            ablation_name, tc_label, category, entry["id"],
                        )
                        if key in completed_keys:
                            done += 1
                        else:
                            pending += 1
                    status = "SKIP" if pending == 0 else f"RUN {pending}"
                    print(f"  {category}: {done}/{len(entries)} done -> {status}")
                continue

            # ── Anthropic cloud API path ─────────────────────
            # No server management, no GGUF, no VRAM budget.
            if config.backend == "anthropic":
                client = _build_client(config)

                for category in categories:
                    is_multi = category in MULTI_TURN_CATEGORIES
                    filename = MULTI_TURN_CATEGORIES[category] if is_multi else CATEGORY_FILES[category]
                    data_path = DATA_DIR / filename
                    gt_path = DATA_DIR / "possible_answer" / filename
                    entries = load_jsonl(data_path)
                    gt_entries = (
                        {e["id"]: e for e in load_jsonl(gt_path)}
                        if gt_path.exists() else {}
                    )

                    pending = []
                    for entry in entries:
                        test_id = entry["id"]
                        key = _entry_key(
                            config.model, config.backend, config.mode,
                            ablation_name, tc_label, category, test_id,
                        )
                        if key in completed_keys:
                            total_skipped += 1
                        else:
                            pending.append(entry)
                        total_entries += 1

                    if not pending:
                        print(f"  {category}: all {len(entries)} done, skip")
                        continue

                    print(
                        f"  {category}: {len(pending)} pending "
                        f"({len(entries) - len(pending)} done)",
                        flush=True,
                    )

                    cat_valid = 0
                    cat_ran = 0
                    for i, entry in enumerate(pending):
                        test_id = entry["id"]
                        if verbose:
                            print(f"    [{i+1}/{len(pending)}] {test_id} ... ",
                                  end="", flush=True)

                        run_result, score = await _run_and_score(
                            client, entry, category, is_multi,
                            gt_entries, ablation,
                        )

                        row = _build_row(
                            config, ablation_name, category, test_id,
                            run_result, score, is_multi,
                        )
                        with output_path.open("a") as f:
                            f.write(json.dumps(row) + "\n")

                        total_ran += 1
                        cat_ran += 1
                        if score.valid:
                            total_valid += 1
                            cat_valid += 1

                        if verbose:
                            status = "PASS" if score.valid else f"FAIL ({score.error_type})"
                            print(f"{status} ({row['elapsed_s']}s)")

                    if not verbose and cat_ran > 0:
                        print(
                            f"    {cat_valid}/{cat_ran} passed "
                            f"({100*cat_valid/cat_ran:.0f}%)",
                            flush=True,
                        )
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
                    backend=config.backend, port=8080, models_dir=MODELS_DIR
                )

            # Resolve GGUF path for non-Ollama backends
            gguf_path = ""
            if config.backend in ("llamaserver", "llamafile"):
                file_map = LLAMAFILE_MAP if config.backend == "llamafile" else GGUF_MAP
                gguf_filename = file_map.get(config.model)
                if not gguf_filename:
                    raise ValueError(f"No GGUF mapping for model: {config.model}")
                gguf_path = str(MODELS_DIR / gguf_filename)

            # Start server
            extra_flags = _get_server_flags(config.model, config.mode)
            await server.start(
                model=config.model,
                gguf_path=gguf_path,
                mode=config.mode,
                extra_flags=extra_flags if extra_flags else None,
            )
            prev_backend = config.backend
            prev_server = server

            # Build client
            client = _build_client(config)

            # Resolve budget through prod ServerManager path
            resolved_budget = await server.resolve_budget(budget_mode, manual_tokens)
            if hasattr(client, "set_num_ctx"):
                client.set_num_ctx(resolved_budget)

            # ── Category loop ────────────────────────────────
            for category in categories:
                is_multi = category in MULTI_TURN_CATEGORIES

                # Load entries + ground truth
                filename = MULTI_TURN_CATEGORIES[category] if is_multi else CATEGORY_FILES[category]
                data_path = DATA_DIR / filename
                gt_path = DATA_DIR / "possible_answer" / filename
                entries = load_jsonl(data_path)
                gt_entries = (
                    {e["id"]: e for e in load_jsonl(gt_path)}
                    if gt_path.exists() else {}
                )

                # Filter to pending entries
                pending = []
                for entry in entries:
                    test_id = entry["id"]
                    key = _entry_key(
                        config.model, config.backend, config.mode,
                        ablation_name, tc_label, category, test_id,
                    )
                    if key in completed_keys:
                        total_skipped += 1
                    else:
                        pending.append(entry)
                    total_entries += 1

                if not pending:
                    print(f"  {category}: all {len(entries)} done, skip")
                    continue

                print(
                    f"  {category}: {len(pending)} pending "
                    f"({len(entries) - len(pending)} done)",
                    flush=True,
                )

                cat_valid = 0
                cat_ran = 0
                for i, entry in enumerate(pending):
                    test_id = entry["id"]
                    if verbose:
                        print(f"    [{i+1}/{len(pending)}] {test_id} ... ",
                              end="", flush=True)

                    run_result, score = await _run_and_score(
                        client, entry, category, is_multi,
                        gt_entries, ablation,
                    )

                    row = _build_row(
                        config, ablation_name, category, test_id,
                        run_result, score, is_multi,
                    )
                    with output_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")

                    total_ran += 1
                    cat_ran += 1
                    if score.valid:
                        total_valid += 1
                        cat_valid += 1

                    if verbose:
                        status = "PASS" if score.valid else f"FAIL ({score.error_type})"
                        print(f"{status} ({row['elapsed_s']}s)")

                if not verbose and cat_ran > 0:
                    print(
                        f"    {cat_valid}/{cat_ran} passed "
                        f"({100*cat_valid/cat_ran:.0f}%)",
                        flush=True,
                    )
    finally:
        await server.stop()

    elapsed = time.monotonic() - batch_start
    print(
        f"\n{'='*70}\n"
        f"Batch complete - {total_ran} entries executed, "
        f"{total_skipped} skipped (already done)\n",
        end="",
    )
    if total_ran > 0:
        print(f"  Valid: {total_valid}/{total_ran} ({100*total_valid/total_ran:.1f}%)")
    print(
        f"  Total time: {elapsed/60:.1f} min\n"
        f"  Results: {output_path}\n"
        f"{'='*70}",
        flush=True,
    )


# ── Run + score helpers ────────────────────────────────────────


async def _run_and_score(
    client: Any,
    entry: dict,
    category: str,
    is_multi: bool,
    gt_entries: dict[str, dict],
    ablation: AblationConfig,
) -> tuple[Any, ScoreResult]:
    """Run a single BFCL entry and score it. Returns (run_result, score)."""
    test_id = entry["id"]

    if is_multi:
        run_result = await run_multi_turn_entry(
            client, entry, category,
            stream=True,
            max_iterations=_max_iters_multi(ablation),
            max_retries_per_step=ablation.max_retries_per_step,
            max_tool_errors=ablation.max_tool_errors,
            rescue_enabled=ablation.rescue_enabled,
        )
        gt_entry = gt_entries.get(test_id)
        if gt_entry is None:
            score = ScoreResult(test_id, False, ["Missing GT"], "internal_error")
        else:
            ground_truth = load_multi_turn_ground_truth(gt_entry)
            score = score_multi_turn_entry(run_result, entry, ground_truth, category)
    else:
        run_result = await run_single_entry(
            client, entry, category,
            stream=True,
            max_iterations=_max_iters_single(ablation),
            max_retries_per_step=ablation.max_retries_per_step,
            max_tool_errors=ablation.max_tool_errors,
            rescue_enabled=ablation.rescue_enabled,
        )
        if not run_result.completed:
            score = ScoreResult(
                test_id, False,
                [f"Did not complete: {run_result.error}"],
                "incomplete",
            )
        else:
            gt_entry = gt_entries.get(test_id)
            ground_truth = gt_entry["ground_truth"] if gt_entry else []
            score = score_entry(
                test_id=test_id,
                func_descriptions=entry["function"],
                model_output=run_result.extracted_calls,
                ground_truth=ground_truth,
                category=category,
            )

    return run_result, score


def _build_row(
    config: BfclBatchConfig,
    ablation_name: str,
    category: str,
    test_id: str,
    run_result: Any,
    score: ScoreResult,
    is_multi: bool,
) -> dict[str, Any]:
    """Build a JSONL row from run result + score."""
    if is_multi:
        iterations = sum(len(tc) for tc in run_result.per_turn_calls)
        completed = run_result.completed
    else:
        iterations = run_result.iterations
        completed = run_result.completed

    return _result_to_row(
        config, ablation_name, category, test_id,
        valid=score.valid,
        error_type=score.error_type,
        errors=score.errors,
        elapsed_s=run_result.elapsed_seconds,
        iterations=iterations,
        completed=completed,
    )


# ── CLI ─────────────────────────────────────────────────────────


async def main() -> None:
    import argparse

    budget_choices = [m.value for m in BudgetMode]
    parser = argparse.ArgumentParser(description="BFCL batch runner")
    parser.add_argument(
        "--config",
        choices=list(CONFIG_SETS.keys()),
        default="all",
        help="Which config set to run",
    )
    parser.add_argument(
        "--categories", nargs="*", default=None,
        help=f"Categories to run (default: all). Choices: {', '.join(ALL_CATEGORIES)}",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"JSONL output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--budget-mode",
        choices=budget_choices,
        default=BudgetMode.FORGE_FULL.value,
        help="Budget mode (prod BudgetMode).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Exact token budget (requires --budget-mode manual).",
    )
    parser.add_argument(
        "--ablation",
        choices=list(ABLATION_PRESETS.keys()),
        default="reforged",
        help="Ablation preset (default: reforged = all guardrails enabled)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter configs to models containing this substring (e.g. --model 8b-reasoning)",
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

    categories = args.categories or ALL_CATEGORIES
    for cat in categories:
        if cat not in CATEGORY_FILES and cat not in MULTI_TURN_CATEGORIES:
            parser.error(f"Unknown category {cat!r}. Valid: {', '.join(ALL_CATEGORIES)}")

    ablation = ABLATION_PRESETS[args.ablation]

    print(f"BFCL Batch Runner")
    print(f"  Config set:    {args.config} ({len(configs)} configs)")
    print(f"  Budget mode:   {budget_mode.value}")
    print(f"  Ablation:      {ablation.name}")
    print(f"  Categories:    {len(categories)}: {', '.join(categories)}")
    print(f"  Output:        {args.output}")
    if args.dry_run:
        print(f"  *** DRY RUN ***")

    await run_batch(
        configs=configs,
        categories=categories,
        output_path=args.output,
        ablation=ablation,
        dry_run=args.dry_run,
        verbose=args.verbose,
        budget_mode=budget_mode,
        manual_tokens=args.num_ctx,
    )


if __name__ == "__main__":
    asyncio.run(main())
