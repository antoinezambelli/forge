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
RUNS_PER_SCENARIO = 10
TIMEOUT_S = 21600  # 6 hours per batch
LOG_FILE = Path("ablation_progress.log")

CONFIGS = [
    # (config_set, model_filter, label)
    ("llamaserver-native", "8b-reasoning-2512-q4", "ministral-8b-reasoning"),
    ("llamaserver-native", "14b-instruct-2512-q4", "ministral-14b-instruct"),
    ("ollama", "qwen3:14b-q4", "qwen3-14b"),
    ("llamafile", "nemo", "nemo-12b"),
    ("haiku", None, "haiku"),
]

PRESETS = ["no_rescue", "no_nudge", "no_steps", "no_recovery", "no_compact"]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_cmd(config: str, model: str | None, preset: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "tests.eval.batch_eval",
        "--config", config,
        "--ablation", preset,
        "--runs", str(RUNS_PER_SCENARIO),
        "--tags", "plumbing", "model_quality",
    ]
    if model:
        cmd.extend(["--model", model])
    return cmd


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    total = len(CONFIGS) * len(PRESETS)
    log(f"Ablation study: {len(CONFIGS)} models x {len(PRESETS)} presets = {total} batches")
    if dry_run:
        log("DRY RUN - commands only, no execution")

    completed = 0
    skipped = 0
    failed = 0

    for config, model, label in CONFIGS:
        for preset in PRESETS:
            batch_label = f"{label} / {preset}"
            cmd = build_cmd(config, model, preset)

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
                        cwd=str(Path(__file__).parent),
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
