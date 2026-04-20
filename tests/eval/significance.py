"""Statistical significance for ablation tables.

Pooled McNemar + Wilson 95% CI per (model, backend, mode) × ablation.

Usage:
    python -m tests.eval.significance eval_results.jsonl
    python -m tests.eval.significance eval_results.jsonl --model qwen3:14b-q4_K_M
    python -m tests.eval.significance eval_results.jsonl --deep-only

Pairing: reforged run i on scenario S vs ablation run i on scenario S.
Because ablation runs reuse the same (scenario, run_index) space, each trial
has a matched pair — McNemar's test is exactly the right tool.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


ABLATION_ORDER = ("reforged", "bare", "no_rescue", "no_nudge", "no_steps", "no_recovery", "no_compact")


def is_correct(row: dict) -> bool:
    """Same correctness signal report.py uses for `score`: accuracy==True on validated runs."""
    return bool(row.get("accuracy")) and not row.get("validate_error")


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval — two-sided 95% by default."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def normal_sf(z: float) -> float:
    """Survival function (1 - CDF) for standard normal, via erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value.

    b = pairs where reforged correct & ablation wrong
    c = pairs where reforged wrong & ablation correct
    Under H0, each discordant pair is a fair coin flip.
    """
    n = b + c
    if n == 0:
        return 1.0
    # Exact binomial tail, two-sided
    k = min(b, c)
    # P(X <= k) under Binomial(n, 0.5)
    log_half = math.log(0.5)
    log_tail = -math.inf
    log_comb = 0.0  # log C(n, 0) = 0
    for i in range(k + 1):
        if i > 0:
            log_comb += math.log(n - i + 1) - math.log(i)
        term = log_comb + n * log_half
        log_tail = term if log_tail == -math.inf else log_tail + math.log1p(math.exp(term - log_tail))
    one_tail = math.exp(log_tail)
    return min(1.0, 2 * one_tail)


def mcnemar_asymptotic(b: int, c: int) -> float:
    """Continuity-corrected McNemar chi-square, two-sided p-value."""
    n = b + c
    if n == 0:
        return 1.0
    chi = (abs(b - c) - 1) ** 2 / n
    # chi-square with 1 df: p = erfc(sqrt(chi/2))
    return math.erfc(math.sqrt(chi / 2))


def mcnemar_pvalue(b: int, c: int) -> float:
    """Pick exact for small discordant counts, asymptotic otherwise."""
    if b + c <= 25:
        return mcnemar_exact(b, c)
    return mcnemar_asymptotic(b, c)


def analyze_config(rows: list[dict]) -> dict:
    """Given all rows for one (model, backend, mode), return a per-ablation analysis table.

    Rows expected to have fields: ablation, scenario, run, accuracy, validate_error.
    Pairs are formed on (scenario, run) across ablations vs the reforged baseline.
    """
    # Build reforged correctness lookup: (scenario, run) -> bool
    ref_by_key: dict[tuple[str, int], bool] = {}
    for r in rows:
        if r.get("ablation", "reforged") == "reforged":
            ref_by_key[(r["scenario"], r["run"])] = is_correct(r)

    out: list[dict] = []
    for abl in ABLATION_ORDER:
        abl_rows = [r for r in rows if r.get("ablation", "reforged") == abl]
        if not abl_rows:
            continue

        n = len(abl_rows)
        correct = sum(1 for r in abl_rows if is_correct(r))
        lo, hi = wilson_ci(correct, n)

        entry = {
            "ablation": abl,
            "n": n,
            "correct": correct,
            "score": correct / n,
            "wilson_lo": lo,
            "wilson_hi": hi,
        }

        if abl != "reforged":
            # Pair each ablation run against the same (scenario, run) in reforged.
            b = 0  # reforged right, ablation wrong
            c = 0  # reforged wrong, ablation right
            paired_n = 0
            for r in abl_rows:
                key = (r["scenario"], r["run"])
                if key not in ref_by_key:
                    continue
                paired_n += 1
                ref_c = ref_by_key[key]
                abl_c = is_correct(r)
                if ref_c and not abl_c:
                    b += 1
                elif abl_c and not ref_c:
                    c += 1
            p = mcnemar_pvalue(b, c)
            ref_score = sum(ref_by_key.values()) / len(ref_by_key) if ref_by_key else 0
            entry.update({
                "paired_n": paired_n,
                "discordant_b": b,
                "discordant_c": c,
                "delta": (correct / n) - ref_score,
                "p_value": p,
                "test": "exact" if b + c <= 25 else "asymptotic",
            })

        out.append(entry)
    return out


def _p_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def print_report(label: str, analysis: list[dict]) -> None:
    print(f"\n{label}")
    print(f"{'ablation':14s} {'score':>8s} {'95% CI':>16s} {'delta':>9s}  {'disc(b/c)':>10s} {'p':>9s}  sig")
    print("-" * 80)
    for e in analysis:
        score_s = f"{e['score'] * 100:6.2f}%"
        ci_s = f"[{e['wilson_lo'] * 100:5.2f},{e['wilson_hi'] * 100:5.2f}]"
        if "p_value" in e:
            delta_s = f"{e['delta'] * 100:+6.2f}pt"
            disc_s = f"{e['discordant_b']}/{e['discordant_c']}"
            p_s = f"{e['p_value']:.2e}"
            sig = _p_stars(e["p_value"])
            print(f"{e['ablation']:14s} {score_s:>8s} {ci_s:>16s} {delta_s:>9s}  {disc_s:>10s} {p_s:>9s}  {sig}")
        else:
            print(f"{e['ablation']:14s} {score_s:>8s} {ci_s:>16s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", help="Path to eval_results.jsonl")
    parser.add_argument("--model", action="append", help="Filter to specific model(s)")
    parser.add_argument("--deep-only", action="store_true", help="Only show configs with at least one no_* ablation")
    args = parser.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Only tool_choice=auto for headline numbers (matches dashboard default)
    rows = [r for r in rows if r.get("tool_choice", "auto") == "auto"]

    by_cfg: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[(r["model"], r["backend"], r["mode"])].append(r)

    # Filter configs
    selected = []
    for key, cfg_rows in by_cfg.items():
        model, backend, mode = key
        if args.model and not any(m in model for m in args.model):
            continue
        if args.deep_only and not any(r.get("ablation", "reforged").startswith("no_") for r in cfg_rows):
            continue
        selected.append((key, cfg_rows))

    # Sort by best reforged score, descending
    def _reforged_score(cfg_rows: list[dict]) -> float:
        ref = [r for r in cfg_rows if r.get("ablation", "reforged") == "reforged"]
        if not ref:
            return 0.0
        return sum(1 for r in ref if is_correct(r)) / len(ref)

    selected.sort(key=lambda kv: _reforged_score(kv[1]), reverse=True)

    for (model, backend, mode), cfg_rows in selected:
        analysis = analyze_config(cfg_rows)
        print_report(f"{model}  ({backend}/{mode})", analysis)

    print("\nSignificance: *** p<.001  ** p<.01  * p<.05  n.s. p>=.05")
    print("Test: McNemar's test on paired (scenario, run) — exact for discordant n<=25, continuity-corrected chi-square otherwise")


if __name__ == "__main__":
    main()
