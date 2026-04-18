"""Unattended ablation study runner.

Runs all 5 models × 5 ablation presets sequentially.
Retries failed configs up to MAX_RETRIES times, then skips.
Logs progress to ablation_progress.log.

Usage:
    python run_ablation.py
    python run_ablation.py --dry-run
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MAX_RETRIES = 3
RUNS_PER_SCENARIO = 50
TIMEOUT_S = 21600  # 6 hours per batch
LOG_FILE = Path("ablation_progress.log")

CONFIGS = [
    # (config_set, model_filter, label)
    # Original 5-model paper set
    ("llamaserver-native", "8b-reasoning-2512-q4", "ministral-8b-reasoning"),
    ("llamaserver-native", "14b-instruct-2512-q4", "ministral-14b-instruct"),
    ("ollama", "qwen3:14b-q4", "qwen3-14b"),
    ("llamafile", "nemo", "nemo-12b"),
    ("haiku", None, "haiku"),
    # 12GB-tier extension: (backend, mode) = best reforged per model
    ("llamaserver-native", "ministral-3:8b-instruct-2512-q4", "ministral-8b-instruct-q4"),
    ("llamaserver-native", "ministral-3:8b-instruct-2512-q8", "ministral-8b-instruct-q8"),
    ("llamaserver-native", "ministral-3:8b-reasoning-2512-q8", "ministral-8b-reasoning-q8"),
    ("llamaserver-prompt", "ministral-3:14b-reasoning-2512-q4", "ministral-14b-reasoning-q4"),
    ("llamaserver-native", "llama3.1:8b-instruct-q4", "llama3.1-8b-q4"),
    ("llamaserver-native", "llama3.1:8b-instruct-q8", "llama3.1-8b-q8"),
    ("llamafile", "mistral:7b-instruct-v0.3-q4", "mistral-7b-q4"),
    ("llamafile", "mistral:7b-instruct-v0.3-q8", "mistral-7b-q8"),
    ("llamaserver-prompt", "gemma4:e4b-it-q4", "gemma4-e4b-q4"),
    ("llamaserver-prompt", "gemma4:e4b-it-q8", "gemma4-e4b-q8"),
]

PRESETS = ["no_rescue", "no_nudge", "no_steps", "no_recovery", "no_compact"]

# Granite: tail-end best-effort. Runs only reforged + bare (headline delta).
# If the main batch finishes in time, these get picked up; if they hang or
# time out, the main results are already durable on disk.
GRANITE_CONFIGS = [
    ("llamaserver-native", "granite-4.0:h-micro-q8_0", "granite-4.0-h-micro-q8"),
    ("llamaserver-native", "granite-4.0:h-tiny-q4_K_M", "granite-4.0-h-tiny-q4"),
    ("llamaserver-native", "granite-4.0:h-tiny-q8_0", "granite-4.0-h-tiny-q8"),
]

GRANITE_PRESETS = ["reforged", "bare"]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_cmd(config: str, model: str | None, preset: str, models_dir: str | None) -> list[str]:
    cmd = [
        sys.executable, "-m", "tests.eval.batch_eval",
        "--config", config,
        "--ablation", preset,
        "--runs", str(RUNS_PER_SCENARIO),
        "--tags", "plumbing", "model_quality",
    ]
    if model:
        cmd.extend(["--model", model])
    if models_dir:
        cmd.extend(["--models-dir", models_dir])
    return cmd


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Unattended ablation study runner")
    parser.add_argument("--models-dir", default=None, help="Directory containing GGUF model files")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    dry_run = args.dry_run
    models_dir = args.models_dir

    main_total = len(CONFIGS) * len(PRESETS)
    granite_total = len(GRANITE_CONFIGS) * len(GRANITE_PRESETS)
    total = main_total + granite_total
    log(
        f"Ablation study: {main_total} main batches "
        f"+ {granite_total} Granite tail-end = {total} total"
    )
    if dry_run:
        log("DRY RUN - commands only, no execution")

    completed = 0
    skipped = 0
    failed = 0

    # ── Main batch ──────────────────────────────────────────
    for config, model, label in CONFIGS:
        for preset in PRESETS:
            batch_label = f"{label} / {preset}"
            cmd = build_cmd(config, model, preset, models_dir)

            if dry_run:
                log(f"  [{completed + 1}/{total}] {batch_label}: {' '.join(cmd)}")
                completed += 1
                continue

            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                log(f"  [{completed + 1}/{total}] {batch_label} (attempt {attempt}/{MAX_RETRIES})")
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=str(Path(__file__).parent.parent),
                        timeout=TIMEOUT_S,
                    )
                    if result.returncode == 0:
                        log(f"  OK: {batch_label}")
                        success = True
                        break
                    else:
                        log(f"  FAILED (exit {result.returncode}): {batch_label}")
                except subprocess.TimeoutExpired:
                    log(f"  TIMEOUT: {batch_label}")
                except Exception as e:
                    log(f"  ERROR: {batch_label} - {e}")

                if attempt < MAX_RETRIES:
                    log(f"  Retrying in 30s...")
                    time.sleep(30)

            if success:
                completed += 1
            else:
                log(f"  SKIPPING after {MAX_RETRIES} failures: {batch_label}")
                skipped += 1
                failed += 1

    # ── Granite tail-end (best-effort) ──────────────────────
    log("\n--- Granite tail-end batch ---")
    for config, model, label in GRANITE_CONFIGS:
        for preset in GRANITE_PRESETS:
            batch_label = f"{label} / {preset}"
            cmd = build_cmd(config, model, preset, models_dir)

            if dry_run:
                log(f"  [{completed + 1}/{total}] {batch_label}: {' '.join(cmd)}")
                completed += 1
                continue

            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                log(f"  [{completed + 1}/{total}] {batch_label} (attempt {attempt}/{MAX_RETRIES})")
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=str(Path(__file__).parent.parent),
                        timeout=TIMEOUT_S,
                    )
                    if result.returncode == 0:
                        log(f"  OK: {batch_label}")
                        success = True
                        break
                    else:
                        log(f"  FAILED (exit {result.returncode}): {batch_label}")
                except subprocess.TimeoutExpired:
                    log(f"  TIMEOUT: {batch_label}")
                except Exception as e:
                    log(f"  ERROR: {batch_label} - {e}")

                if attempt < MAX_RETRIES:
                    log(f"  Retrying in 30s...")
                    time.sleep(30)

            if success:
                completed += 1
            else:
                log(f"  SKIPPING after {MAX_RETRIES} failures: {batch_label}")
                skipped += 1
                failed += 1

    log(f"\nDone. Completed: {completed}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
