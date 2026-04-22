"""Unattended ablation study runner.

Runs all configured models × ablation presets sequentially.
Retries failed configs up to MAX_RETRIES times, then skips.
Logs progress to ablation_progress.log.

Usage:
    python run_ablation.py
    python run_ablation.py --dry-run
    python run_ablation.py --output eval_results_rig-00.jsonl
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

# Rig-00 plan for the model-params re-run: all Qwen3 variants × all backends,
# plus all llamafile configs. reforged + bare only.
QWEN_CONFIGS = [
    # (config_set, model_filter, label)
    ("ollama",            "qwen3:8b-q4",  "qwen3-8b-q4-ollama"),
    ("llamaserver-native", "qwen3:8b-q4", "qwen3-8b-q4-lls-native"),
    ("llamaserver-prompt", "qwen3:8b-q4", "qwen3-8b-q4-lls-prompt"),
    ("ollama",            "qwen3:8b-q8",  "qwen3-8b-q8-ollama"),
    ("llamaserver-native", "qwen3:8b-q8", "qwen3-8b-q8-lls-native"),
    ("llamaserver-prompt", "qwen3:8b-q8", "qwen3-8b-q8-lls-prompt"),
    ("ollama",            "qwen3:14b-q4", "qwen3-14b-q4-ollama"),
    ("llamaserver-native", "qwen3:14b-q4", "qwen3-14b-q4-lls-native"),
    ("llamaserver-prompt", "qwen3:14b-q4", "qwen3-14b-q4-lls-prompt"),
]

LLAMAFILE_CONFIGS = [
    ("llamafile", "llama3.1:8b-instruct-q4",      "llama3.1-8b-q4-lf"),
    ("llamafile", "llama3.1:8b-instruct-q8",      "llama3.1-8b-q8-lf"),
    ("llamafile", "mistral-nemo",                 "nemo-12b-lf"),
    ("llamafile", "mistral:7b-instruct-v0.3-q4",  "mistral-7b-q4-lf"),
    ("llamafile", "mistral:7b-instruct-v0.3-q8",  "mistral-7b-q8-lf"),
]

CONFIGS = QWEN_CONFIGS + LLAMAFILE_CONFIGS

PRESETS = ["reforged", "bare"]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_cmd(
    config: str,
    model: str | None,
    preset: str,
    models_dir: str | None,
    output: str | None,
) -> list[str]:
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
    if output:
        cmd.extend(["--output", output])
    return cmd


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Unattended ablation study runner")
    parser.add_argument("--models-dir", default=None, help="Directory containing GGUF model files")
    parser.add_argument(
        "--output",
        default=None,
        help="JSONL output path (forwarded to batch_eval; defaults to eval_results.jsonl)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    dry_run = args.dry_run
    models_dir = args.models_dir
    output = args.output

    total = len(CONFIGS) * len(PRESETS)
    log(f"Ablation study: {len(CONFIGS)} configs x {len(PRESETS)} presets = {total} batches")
    if output:
        log(f"Output: {output}")
    if dry_run:
        log("DRY RUN - commands only, no execution")

    completed = 0
    skipped = 0
    failed = 0

    for config, model, label in CONFIGS:
        for preset in PRESETS:
            batch_label = f"{label} / {preset}"
            cmd = build_cmd(config, model, preset, models_dir, output)

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
