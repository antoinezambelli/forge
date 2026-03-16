"""BFCL report generator — reads JSONL, prints ASCII table.

Usage:
    python -m tests.eval.bfcl.bfcl_report bfcl_results.jsonl
    python -m tests.eval.bfcl.bfcl_report bfcl_results.jsonl --progress
    python -m tests.eval.bfcl.bfcl_report bfcl_results.jsonl --list-only
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


# ── Short names ─────────────────────────────────────────────────

CATEGORY_SHORT_NAMES = {
    "simple_python": "spy",
    "simple_java": "sja",
    "simple_javascript": "sjs",
    "multiple": "mul",
    "parallel": "par",
    "parallel_multiple": "pmul",
    "irrelevance": "irr",
    "multi_turn_base": "mtb",
    "multi_turn_miss_func": "mtmf",
    "multi_turn_miss_param": "mtmp",
    "multi_turn_long_context": "mtlc",
}

# Preferred column order (categories appearing in data but not here sort last)
_CATEGORY_ORDER = list(CATEGORY_SHORT_NAMES.keys())

BACKEND_ABBREV = {
    "ollama": "OL",
    "llamaserver": "LS",
    "llamafile": "LF",
    "anthropic": "AN",
}

MODE_ABBREV = {"native": "N", "prompt": "P"}


def _shorten_model(model: str) -> str:
    """Shorten a model name for display."""
    # Strip quant suffixes
    for suffix in ["-q4_K_M", "-q8_0", "-Q4_K_M", "-Q8_0",
                   "_q4_K_M", "_q8_0", "_Q4_K_M", "_Q8_0"]:
        model = model.replace(suffix, "")
    # Strip known long suffixes
    for suffix in ["-instruct-2512", "-instruct-2407",
                   "-instruct-v0.3", "-instruct",
                   "-20251001", "-20250414"]:
        model = model.replace(suffix, "")
    return model


def _shorten_category(category: str) -> str:
    """Short column header for a category."""
    return CATEGORY_SHORT_NAMES.get(category, category[:4])


def _config_key(row: dict) -> str:
    """Unique key for a model/backend/mode/ablation/tool_choice combo."""
    tc = row.get("tool_choice", "auto")
    return f"{row['model']}|{row['backend']}|{row['mode']}|{row['ablation']}|{tc}"


def _config_label(row: dict) -> str:
    """Human-readable row label."""
    model = _shorten_model(row["model"])
    be = BACKEND_ABBREV.get(row["backend"], row["backend"][:2].upper())
    mode = MODE_ABBREV.get(row["mode"], row["mode"][0].upper())
    ablation = row["ablation"]
    tc = row.get("tool_choice", "auto")
    tc_suffix = f" tc={tc}" if tc != "auto" else ""
    return f"{model} {be}/{mode} [{ablation}{tc_suffix}]"


# ── Data loading ────────────────────────────────────────────────


def _load_rows(jsonl_path: Path) -> list[dict]:
    """Load all valid rows from JSONL."""
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ── Report building ─────────────────────────────────────────────


def build_report(rows: list[dict]) -> str:
    """Build ASCII table from JSONL rows.

    Columns:
      Scr  = score = correct / total
      Acc  = accuracy = correct / completed (excludes incomplete)
      Cmp  = completeness = completed / total
      Per-category columns show score (correct / total for that category).

    Returns the formatted string.
    """
    if not rows:
        return "No data."

    # Group by config
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_config[_config_key(row)].append(row)

    # Discover categories from data (in preferred order)
    all_cats = set()
    for row in rows:
        all_cats.add(row["category"])
    categories = [c for c in _CATEGORY_ORDER if c in all_cats]
    categories.extend(sorted(all_cats - set(categories)))

    # Build per-config stats
    config_stats = []
    for key, config_rows in by_config.items():
        label = _config_label(config_rows[0])

        # cat -> (valid, completed, total)
        cat_stats: dict[str, tuple[int, int, int]] = {}
        for cat in categories:
            cat_rows = [r for r in config_rows if r["category"] == cat]
            valid = sum(1 for r in cat_rows if r.get("valid", False))
            completed = sum(1 for r in cat_rows if r.get("completed", False))
            total = len(cat_rows)
            cat_stats[cat] = (valid, completed, total)

        total_valid = sum(v for v, _, _ in cat_stats.values())
        total_completed = sum(c for _, c, _ in cat_stats.values())
        total_entries = sum(t for _, _, t in cat_stats.values())

        score = total_valid / total_entries if total_entries > 0 else 0.0
        accuracy = total_valid / total_completed if total_completed > 0 else None
        completeness = total_completed / total_entries if total_entries > 0 else 0.0

        config_stats.append({
            "label": label,
            "score": score,
            "accuracy": accuracy,
            "completeness": completeness,
            "total_entries": total_entries,
            "cats": cat_stats,
        })

    # Sort by score desc -> completeness desc
    config_stats.sort(key=lambda x: (-x["score"], -x["completeness"]))

    # Build table
    cat_headers = [_shorten_category(c) for c in categories]
    label_width = max(len(s["label"]) for s in config_stats)
    label_width = max(label_width, len("Model/Backend"))

    # Header
    header = f"{'Model/Backend':<{label_width}}  {'Scr':>7}  {'Acc':>7}  {'Cmp':>7}"
    for ch in cat_headers:
        header += f"  {ch:>4}"
    sep = "\u2500" * len(header)

    lines = [sep, header, sep]

    for s in config_stats:
        scr_str = f"{s['score']*100:.1f}%"
        acc_str = f"{s['accuracy']*100:.1f}%" if s["accuracy"] is not None else "  \u2014"
        cmp_str = f"{s['completeness']*100:.1f}%"
        line = f"{s['label']:<{label_width}}  {scr_str:>7}  {acc_str:>7}  {cmp_str:>7}"
        for cat in categories:
            valid, completed, total = s["cats"].get(cat, (0, 0, 0))
            if total == 0:
                line += f"  {'\u2014':>4}"
            else:
                cat_pct = f"{100*valid/total:.0f}"
                line += f"  {cat_pct:>4}"
        lines.append(line)

    lines.append(sep)

    # Legend
    lines.append(
        "Scr=score(correct/total), Acc=accuracy(correct/completed), "
        "Cmp=completeness(completed/total)"
    )
    cat_legend = ", ".join(
        f"{_shorten_category(c)}={c}" for c in categories
    )
    lines.append(cat_legend)

    return "\n".join(lines)


def build_progress(rows: list[dict]) -> str:
    """Build progress summary — entries completed per config x category."""
    if not rows:
        return "No data."

    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_config[_config_key(row)].append(row)

    all_cats = set(r["category"] for r in rows)
    categories = [c for c in _CATEGORY_ORDER if c in all_cats]
    categories.extend(sorted(all_cats - set(categories)))

    lines = []
    for key, config_rows in sorted(by_config.items()):
        label = _config_label(config_rows[0])
        total = len(config_rows)
        lines.append(f"\n{label} ({total} entries)")
        for cat in categories:
            cat_rows = [r for r in config_rows if r["category"] == cat]
            if cat_rows:
                valid = sum(1 for r in cat_rows if r.get("valid", False))
                lines.append(f"  {cat}: {len(cat_rows)} done, {valid} valid")

    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="BFCL report from JSONL")
    parser.add_argument("jsonl", type=Path, help="Path to bfcl_results.jsonl")
    parser.add_argument("--progress", action="store_true",
                        help="Show progress (entries per config)")
    parser.add_argument("--list-only", action="store_true",
                        help="Compact list format (no table)")
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"File not found: {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    rows = _load_rows(args.jsonl)

    if args.progress:
        print(build_progress(rows))
    elif args.list_only:
        print(build_progress(rows))
    else:
        print(build_report(rows))


if __name__ == "__main__":
    main()
