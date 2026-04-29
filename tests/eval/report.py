"""Eval report generator - reads JSONL, prints ASCII table + list summary.

Usage:
    python -m tests.eval.report [eval_results.jsonl]
    python -m tests.eval.report --list-only results.jsonl
    python -m tests.eval.report eval_tight.jsonl  # single-scenario budget files
    python -m tests.eval.report eval_results.jsonl --html docs/results/dashboard.html
    python -m tests.eval.report eval_results.jsonl --markdown docs/results/

Only shows model/backend/mode combos that have fully completed all scenarios.
Scenarios are derived from the data — works for both full 8-scenario batch
files and single-scenario budget eval files.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Scenarios and their ideal iteration counts (from scenarios.py)
# Hardcoded here so the report script has zero forge imports and can
# run on any machine with just Python + the JSONL file.
SCENARIO_NAMES = [
    "relevance_detection",
    "argument_fidelity",
    "tool_selection",
    "basic_2step",
    "sequential_3step",
    "conditional_routing",
    "sequential_reasoning",
    "error_recovery",
    "data_gap_recovery",
    "compaction_stress",
    "phase2_compaction",
    # Advanced reasoning (lambda)
    "data_gap_recovery_extended",
    "argument_transformation",
    "grounded_synthesis",
    "inconsistent_api_recovery",
    # Stateful scenarios (same order as lambda)
    "compaction_stress_stateful",
    "phase2_compaction_stateful",
    "inventory_audit",
    "supplier_deep_dive",
    "relevance_detection_stateful",
    "argument_fidelity_stateful",
    "tool_selection_stateful",
    "basic_2step_stateful",
    "sequential_3step_stateful",
    "conditional_routing_stateful",
    "sequential_reasoning_stateful",
    "error_recovery_stateful",
    "data_gap_recovery_stateful",
    # Advanced reasoning (stateful)
    "data_gap_recovery_extended_stateful",
    "argument_transformation_stateful",
    "grounded_synthesis_stateful",
    "inconsistent_api_recovery_stateful",
]

# Fallback ideal iterations for JSONL files that predate the ideal_iterations field.
# New runs write ideal_iterations per-row via batch_eval.py; this is only for old data.
_SCENARIO_IDEAL_FALLBACK: dict[str, int] = {
    "basic_2step": 2,
    "sequential_3step": 3,
    "compaction_stress": 3,
    "error_recovery": 2,
    "tool_selection": 3,
    "argument_fidelity": 3,
    "sequential_reasoning": 4,
    "conditional_routing": 4,
    "data_gap_recovery": 5,
    "phase2_compaction": 6,
    "relevance_detection": 1,
    # Stateful
    "basic_2step_stateful": 2,
    "sequential_3step_stateful": 3,
    "error_recovery_stateful": 3,
    "tool_selection_stateful": 3,
    "argument_fidelity_stateful": 3,
    "sequential_reasoning_stateful": 4,
    "conditional_routing_stateful": 4,
    "data_gap_recovery_stateful": 5,
    "compaction_stress_stateful": 3,
    "phase2_compaction_stateful": 6,
    "inventory_audit": 13,
    "supplier_deep_dive": 12,
    "relevance_detection_stateful": 1,
}

# Scenarios excluded from speed calculation (outlier timing)
SPEED_EXCLUDE = {"compaction_stress", "compaction_stress_stateful", "inventory_audit", "supplier_deep_dive"}

# Suite membership — advanced_reasoning vs og18. Used to drive the dashboard
# suite slicer; everything not in _AR_SCENARIOS is treated as og18.
_AR_SCENARIOS: set[str] = {
    "data_gap_recovery_extended", "data_gap_recovery_extended_stateful",
    "argument_transformation", "argument_transformation_stateful",
    "grounded_synthesis", "grounded_synthesis_stateful",
    "inconsistent_api_recovery", "inconsistent_api_recovery_stateful",
}


# ── Data loading ────────────────────────────────────────────────


@dataclass
class ConfigKey:
    model: str
    backend: str
    mode: str
    ablation: str = "reforged"
    tool_choice: str = "auto"

    @property
    def _tag(self) -> str:
        """Ablation + tool_choice tag, e.g. '[full]', '[bare]', '[bare+any]'."""
        if self.ablation != "reforged" and self.tool_choice != "auto":
            return f"[{self.ablation}+{self.tool_choice}]"
        return f"[{self.ablation}]"

    @property
    def label(self) -> str:
        return f"{self.model} ({self.backend}/{self.mode}) {self._tag}"

    @property
    def short_label(self) -> str:
        """Compact label for table display."""
        m = self.model
        if self.backend == "llamaserver":
            b = "LS"
        elif self.backend == "llamafile":
            b = "LF"
        elif self.backend == "anthropic":
            b = "AN"
        else:
            b = "OL"
        mode_char = "N" if self.mode == "native" else "P"
        return f"{m} {b}/{mode_char} {self._tag}"

    def __hash__(self) -> int:
        return hash((self.model, self.backend, self.mode, self.ablation, self.tool_choice))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConfigKey):
            return NotImplemented
        return (
            self.model == other.model
            and self.backend == other.backend
            and self.mode == other.mode
            and self.ablation == other.ablation
            and self.tool_choice == other.tool_choice
        )


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def group_rows(
    rows: list[dict],
) -> dict[ConfigKey, dict[str, list[dict]]]:
    """Group rows by config -> scenario -> list of run rows."""
    grouped: dict[ConfigKey, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        ablation = row.get("ablation", "reforged")
        tc = row.get("tool_choice", "auto")
        key = ConfigKey(row["model"], row["backend"], row["mode"], ablation, tc)
        grouped[key][row["scenario"]].append(row)
    return grouped


# ── Metrics ─────────────────────────────────────────────────────


@dataclass
class ConfigMetrics:
    key: ConfigKey
    runs_per_scenario: int
    total_runs: int
    total_completed: int
    completeness: float  # completed / total across all scenarios
    per_scenario_completeness: dict[str, float]  # scenario -> rate
    score: float  # correct / total across all scenarios
    accuracy: float | None  # correct / validated (excludes runs with accuracy=None)
    per_scenario_score: dict[str, float | None]  # scenario -> correct / total
    per_scenario_runs: dict[str, int]  # scenario -> total run count
    per_scenario_correct: dict[str, int]  # scenario -> correct run count
    per_scenario_completed: dict[str, int]  # scenario -> completed run count
    per_scenario_validated: dict[str, int]  # scenario -> count of runs with non-None accuracy (denom for accuracy)
    per_scenario_ideal_calls: dict[str, int]  # scenario -> sum of ideal_iterations across correct runs
    per_scenario_actual_calls: dict[str, int]  # scenario -> sum of iterations across correct runs
    per_scenario_wasted_sum: dict[str, float]  # scenario -> sum of wasted_calls across completed runs
    per_scenario_wasted_n: dict[str, int]  # scenario -> count of completed runs with non-None wasted
    per_scenario_speed_sum: dict[str, float]  # scenario -> sum of elapsed_s across completed runs (speed-eligible)
    per_scenario_speed_n: dict[str, int]  # scenario -> count of speed-eligible completed runs
    efficiency: float  # 1.0 = perfect (no wasted calls)
    avg_wasted: float
    speed: float  # avg seconds per run (excluding compaction_stress)
    complete: bool  # all scenarios have full run count
    stream_retries: int = 0  # total stream retries across all runs
    validate_errors: int = 0  # runs where validate() raised an exception


def _detect_scenarios(rows: list[dict]) -> list[str]:
    """Derive the scenario list from the data itself.

    If the data contains all 8 known scenarios, use the canonical order.
    Otherwise return only the scenarios present (e.g. budget eval files
    that only contain compaction_stress).
    """
    present = dict.fromkeys(r["scenario"] for r in rows)  # preserves order
    # If all known scenarios are present, use canonical order
    if all(sc in present for sc in SCENARIO_NAMES):
        return list(SCENARIO_NAMES)
    # Otherwise return only what's in the data, in canonical order where possible
    ordered = [sc for sc in SCENARIO_NAMES if sc in present]
    # Append any unknown scenarios (future-proofing)
    for sc in present:
        if sc not in ordered:
            ordered.append(sc)
    return ordered


def compute_config_metrics(
    key: ConfigKey,
    scenario_runs: dict[str, list[dict]],
    scenarios: list[str] | None = None,
    target_runs: dict[str, int] | None = None,
) -> ConfigMetrics:
    if scenarios is None:
        scenarios = SCENARIO_NAMES

    total_runs = 0
    total_completed = 0
    total_wasted = 0
    wasted_count = 0
    total_stream_retries = 0
    total_validate_errors = 0
    total_validated = 0
    total_correct = 0
    speed_times: list[float] = []
    per_scenario_completeness: dict[str, float] = {}
    per_scenario_score: dict[str, float | None] = {}
    per_scenario_runs: dict[str, int] = {}
    per_scenario_correct: dict[str, int] = {}
    per_scenario_completed: dict[str, int] = {}
    per_scenario_validated: dict[str, int] = {}
    per_scenario_ideal_calls: dict[str, int] = {}
    per_scenario_actual_calls: dict[str, int] = {}
    per_scenario_wasted_sum: dict[str, float] = {}
    per_scenario_wasted_n: dict[str, int] = {}
    per_scenario_speed_sum: dict[str, float] = {}
    per_scenario_speed_n: dict[str, int] = {}

    # A config is complete when every scenario it participates in (>0 runs)
    # has reached the global target run count for that scenario.
    run_counts = [len(scenario_runs.get(sc, [])) for sc in scenarios]
    max_runs = max(run_counts) if run_counts else 0
    if target_runs is not None:
        all_complete = all(
            len(scenario_runs.get(sc, [])) >= target_runs[sc]
            for sc in scenarios
            if len(scenario_runs.get(sc, [])) > 0
        ) and max_runs > 0
    else:
        all_complete = all(c == max_runs for c in run_counts) and max_runs > 0

    for scenario_name in scenarios:
        runs = scenario_runs.get(scenario_name, [])

        n = len(runs)
        completed = sum(1 for r in runs if r["completeness"])
        total_runs += n
        total_completed += completed
        per_scenario_completeness[scenario_name] = completed / n if n > 0 else 0.0
        per_scenario_runs[scenario_name] = n
        per_scenario_completed[scenario_name] = completed

        # Accuracy (correct / total runs — score semantics)
        tainted = sum(1 for r in runs if r.get("validate_error"))
        total_validate_errors += tainted
        validated = [
            r for r in runs
            if r.get("accuracy") is not None and not r.get("validate_error")
        ]
        sc_correct = sum(1 for r in validated if r["accuracy"])
        total_validated += len(validated)
        total_correct += sc_correct
        per_scenario_correct[scenario_name] = sc_correct
        per_scenario_validated[scenario_name] = len(validated)
        if n > 0:
            per_scenario_score[scenario_name] = sc_correct / n
        else:
            per_scenario_score[scenario_name] = None

        # Wasted calls (completed runs only)
        sc_wasted_sum = 0.0
        sc_wasted_n = 0
        for r in runs:
            if r["completeness"] and r.get("wasted_calls") is not None:
                total_wasted += r["wasted_calls"]
                wasted_count += 1
                sc_wasted_sum += r["wasted_calls"]
                sc_wasted_n += 1
        per_scenario_wasted_sum[scenario_name] = sc_wasted_sum
        per_scenario_wasted_n[scenario_name] = sc_wasted_n

        # Stream retries
        for r in runs:
            total_stream_retries += r.get("stream_retries", 0)

        # Speed (exclude compaction_stress)
        sc_speed_sum = 0.0
        sc_speed_n = 0
        if scenario_name not in SPEED_EXCLUDE:
            for r in runs:
                if r["completeness"]:
                    speed_times.append(r["elapsed_s"])
                    sc_speed_sum += r["elapsed_s"]
                    sc_speed_n += 1
        per_scenario_speed_sum[scenario_name] = sc_speed_sum
        per_scenario_speed_n[scenario_name] = sc_speed_n

        # Per-scenario efficiency components (correct runs only, same gating as global)
        sc_ideal = 0
        sc_actual = 0
        for r in runs:
            if r["completeness"] and r.get("accuracy"):
                ideal = r.get("ideal_iterations") or _SCENARIO_IDEAL_FALLBACK.get(scenario_name, 3)
                sc_ideal += ideal
                sc_actual += r["iterations"]
        per_scenario_ideal_calls[scenario_name] = sc_ideal
        per_scenario_actual_calls[scenario_name] = sc_actual

    completeness = total_completed / total_runs if total_runs > 0 else 0.0
    avg_wasted = total_wasted / wasted_count if wasted_count > 0 else 0.0

    # Efficiency: ratio of ideal calls to actual calls across correct runs only.
    # 1.0 = every correct run used exactly the ideal number of calls.
    # Only correct runs count — a model that fails fast shouldn't look efficient.
    total_ideal = 0
    total_actual = 0
    for scenario_name in scenarios:
        for r in scenario_runs.get(scenario_name, []):
            if r["completeness"] and r.get("accuracy"):
                ideal = r.get("ideal_iterations") or _SCENARIO_IDEAL_FALLBACK.get(scenario_name, 3)
                total_ideal += ideal
                total_actual += r["iterations"]
    efficiency = min(total_ideal / total_actual, 1.0) if total_actual > 0 else 0.0

    speed = sum(speed_times) / len(speed_times) if speed_times else 0.0

    accuracy = total_correct / total_validated if total_validated > 0 else None
    score = total_correct / total_runs if total_runs > 0 else 0.0

    return ConfigMetrics(
        key=key,
        runs_per_scenario=max_runs,
        total_runs=total_runs,
        total_completed=total_completed,
        completeness=completeness,
        per_scenario_completeness=per_scenario_completeness,
        score=score,
        accuracy=accuracy,
        per_scenario_score=per_scenario_score,
        per_scenario_runs=per_scenario_runs,
        per_scenario_correct=per_scenario_correct,
        per_scenario_completed=per_scenario_completed,
        per_scenario_validated=per_scenario_validated,
        per_scenario_ideal_calls=per_scenario_ideal_calls,
        per_scenario_actual_calls=per_scenario_actual_calls,
        per_scenario_wasted_sum=per_scenario_wasted_sum,
        per_scenario_wasted_n=per_scenario_wasted_n,
        per_scenario_speed_sum=per_scenario_speed_sum,
        per_scenario_speed_n=per_scenario_speed_n,
        efficiency=efficiency,
        avg_wasted=avg_wasted,
        speed=speed,
        complete=all_complete,
        stream_retries=total_stream_retries,
        validate_errors=total_validate_errors,
    )


# ── ASCII table ─────────────────────────────────────────────────


_SCENARIO_ABBREV: dict[str, str] = {
    "basic_2step": "b2s",
    "sequential_3step": "s3s",
    "compaction_stress": "cmp",
    "error_recovery": "err",
    "tool_selection": "tsl",
    "argument_fidelity": "arg",
    "sequential_reasoning": "srn",
    "conditional_routing": "crt",
    "data_gap_recovery": "dgr",
    "phase2_compaction": "p2c",
    "relevance_detection": "rel",
    # Advanced reasoning (lambda)
    "data_gap_recovery_extended": "dge",
    "argument_transformation": "art",
    "grounded_synthesis": "grs",
    "inconsistent_api_recovery": "iar",
    # Stateful (same abbrev + _s)
    "basic_2step_stateful": "b2s_s",
    "sequential_3step_stateful": "s3s_s",
    "error_recovery_stateful": "err_s",
    "tool_selection_stateful": "tsl_s",
    "argument_fidelity_stateful": "arg_s",
    "sequential_reasoning_stateful": "srn_s",
    "conditional_routing_stateful": "crt_s",
    "data_gap_recovery_stateful": "dgr_s",
    "compaction_stress_stateful": "cmp_s",
    "phase2_compaction_stateful": "p2c_s",
    "inventory_audit": "inv",
    "supplier_deep_dive": "sdd",
    "relevance_detection_stateful": "rel_s",
    # Advanced reasoning (stateful)
    "data_gap_recovery_extended_stateful": "dge_s",
    "argument_transformation_stateful": "art_s",
    "grounded_synthesis_stateful": "grs_s",
    "inconsistent_api_recovery_stateful": "iar_s",
}


# ── Model metadata extraction ─────────────────────────────────


def extract_family(model: str) -> str:
    """Extract model family from Ollama-style model name."""
    if "claude" in model:
        return "claude"
    if "llama3.1" in model:
        return "llama3.1"
    if model.startswith("qwen3:"):
        size = model.split(":")[1].split("-")[0]  # "8b", "14b"
        return f"qwen3-{size}"
    if "ministral" in model:
        size = model.split(":")[1].split("-")[0]
        return f"ministral-{size}"
    if "mistral-nemo" in model:
        return "mistral-nemo"
    if "mistral:" in model:
        return "mistral-v0.3"
    if model.startswith("granite-4.0:"):
        variant = model.split(":")[1].split("-")[0]  # "h", before splitting further
        # Names look like "granite-4.0:h-micro-q4_K_M" or "granite-4.0:h-tiny-q8_0"
        # We want family = "granite-4.0-h-micro" or "granite-4.0-h-tiny"
        parts = model.split(":")[1].split("-")
        # parts[0]="h", parts[1]="micro|tiny", rest is quant
        return f"granite-4.0-{parts[0]}-{parts[1]}"
    return model.split(":")[0]


def extract_quant(model: str) -> str:
    """Extract quantization from model name."""
    low = model.lower()
    if "q4_k_m" in low:
        return "q4_K_M"
    if "q8_0" in low:
        return "q8_0"
    return "n/a"


def _sort_metrics(metrics_list: list[ConfigMetrics]) -> list[ConfigMetrics]:
    """Sort by score desc → completeness desc → efficiency desc → speed asc."""
    return sorted(metrics_list, key=lambda m: (
        -round(m.score, 2),
        -round(m.completeness, 2),
        -round(m.efficiency, 2),
        m.speed,
    ))


def _legend_lines(scenarios: list[str]) -> list[str]:
    """Return legend lines for table footer."""
    lines = [
        "Scr=score(correct/total), Acc=accuracy(correct/total, excl validate errors), Cmp=completeness(completed/total), "
        "Eff=efficiency(ideal/actual calls), Wst=avg wasted calls, Spd=avg time(excl compaction)",
    ]
    abbrev_legend = ", ".join(
        f"{_SCENARIO_ABBREV.get(sc, sc[:3])}={sc}" for sc in scenarios
    )
    lines.append(abbrev_legend)
    lines.append(
        "Ablation: full=all guardrails, no_rescue=no rescue loop, no_nudge=no rescue/retry nudge, "
        "no_steps=no step enforcement, no_recovery=no error recovery, no_compact=no compaction, "
        "bare=all guardrails off"
    )
    return lines


def render_table_string(
    metrics_list: list[ConfigMetrics],
    scenarios: list[str] | None = None,
    include_legend: bool = True,
    presorted: bool = False,
) -> str:
    """Render ASCII table as a string (reused by print_table and markdown views)."""
    if not metrics_list:
        return "No fully completed configs found.\n"

    if scenarios is None:
        scenarios = SCENARIO_NAMES

    if not presorted:
        metrics_list = _sort_metrics(metrics_list)

    # Column widths
    label_w = max(len(m.key.short_label) for m in metrics_list)
    label_w = max(label_w, 10)

    # Build scenario column headers (width adapts to abbreviation length)
    sc_w = max(
        (len(_SCENARIO_ABBREV.get(sc, sc[:3])) for sc in scenarios),
        default=3,
    )
    sc_w = max(sc_w, 3)  # minimum 3 for percentage values
    sc_headers = " ".join(
        f"{_SCENARIO_ABBREV.get(sc, sc[:3]):>{sc_w}}" for sc in scenarios
    )

    header = (
        f"{'Model/Backend':<{label_w}}  "
        f"{'Scr':>7}  "
        f"{'Acc':>7}  "
        f"{'Cmp':>7}  "
        f"{'Eff':>5}  "
        f"{'Wst':>4}  "
        f"{'Spd':>5}  "
        f"{'N':>4}  "
        f"{sc_headers}"
    )
    sep = "-" * len(header)

    lines: list[str] = [sep, header, sep]

    for m in metrics_list:
        scr_str = f"{m.score*100:.1f}%"
        acc_str = f"{m.accuracy*100:.1f}%" if m.accuracy is not None else "  —"
        cmp_str = f"{m.completeness*100:.1f}%"
        eff_str = f"{m.efficiency*100:.0f}%"
        wst_str = f"{m.avg_wasted:.1f}"
        spd_str = f"{m.speed:.1f}s"
        n_str = f"{m.runs_per_scenario}"

        sc_strs = []
        for sc in scenarios:
            rate = m.per_scenario_score.get(sc)
            if rate is not None:
                sc_strs.append(f"{rate*100:{sc_w}.0f}")
            elif m.per_scenario_runs.get(sc, 0) == 0:
                sc_strs.append(f"{'I':>{sc_w}}")
            else:
                sc_strs.append(f"{'—':>{sc_w}}")

        row = (
            f"{m.key.short_label:<{label_w}}  "
            f"{scr_str:>7}  "
            f"{acc_str:>7}  "
            f"{cmp_str:>7}  "
            f"{eff_str:>5}  "
            f"{wst_str:>4}  "
            f"{spd_str:>5}  "
            f"{n_str:>4}  "
            f"{' '.join(sc_strs)}"
        )
        lines.append(row)

    lines.append(sep)

    if include_legend:
        lines.extend(_legend_lines(scenarios))

    lines.append("")
    return "\n".join(lines)


def print_table(
    metrics_list: list[ConfigMetrics],
    scenarios: list[str] | None = None,
) -> None:
    if not metrics_list:
        print("No fully completed configs found.")
        return
    print(f"\n{render_table_string(metrics_list, scenarios)}")


# ── List view (phone-friendly) ──────────────────────────────────


def print_list(metrics_list: list[ConfigMetrics]) -> None:
    if not metrics_list:
        print("No fully completed configs found.")
        return

    metrics_list = _sort_metrics(metrics_list)

    print(f"\n{'='*50}")
    print(f"  FORGE EVAL SUMMARY ({len(metrics_list)} configs)")
    print(f"{'='*50}")

    for i, m in enumerate(metrics_list, 1):
        cmp_pct = m.completeness * 100
        eff_pct = m.efficiency * 100
        pass_fail = f"{m.total_completed}/{m.total_runs}"

        print(f"\n  #{i} {m.key.label}")
        print(f"     Score:         {m.score*100:.1f}%")
        if m.accuracy is not None:
            print(f"     Accuracy:      {m.accuracy*100:.1f}%")
        print(f"     Completeness:  {pass_fail} ({cmp_pct:.1f}%)")
        print(f"     Efficiency: {eff_pct:.0f}% (avg {m.avg_wasted:.1f} wasted)")
        print(f"     Speed:      {m.speed:.1f}s avg (excl compaction)")

        # Flag weak scenarios (by completeness)
        weak = [
            sc for sc, rate in m.per_scenario_completeness.items()
            if rate < 1.0
        ]
        if weak:
            parts = [f"{sc}={m.per_scenario_completeness[sc]*100:.0f}%" for sc in weak]
            print(f"     Weak:       {', '.join(parts)}")
        else:
            print(f"     Weak:       none (all 100%)")

        if m.stream_retries > 0:
            print(f"     Stream retries: {m.stream_retries}")
        if m.validate_errors > 0:
            print(f"     Validate errors: {m.validate_errors}")

    print(f"\n{'='*50}\n")


# ── Progress view (for incomplete data) ─────────────────────────


def print_progress(
    grouped: dict[ConfigKey, dict[str, list[dict]]],
    scenarios: list[str] | None = None,
) -> None:
    """Show how many runs are done for each config, even if incomplete."""
    if scenarios is None:
        scenarios = SCENARIO_NAMES

    # Derive the target from the max run count seen for any single scenario
    global_max = 0
    for scenario_runs in grouped.values():
        for runs in scenario_runs.values():
            global_max = max(global_max, len(runs))

    print(f"\n{'='*50}")
    print(f"  PROGRESS (max N={global_max}, {len(scenarios)} scenarios)")
    print(f"{'='*50}")

    total_expected = 0
    total_done = 0

    for key in sorted(grouped.keys(), key=lambda k: k.label):
        scenario_runs = grouped[key]
        counts = [len(scenario_runs.get(sc, [])) for sc in scenarios]
        done = sum(counts)
        target = max(counts) if counts else global_max
        expected = len(scenarios) * target
        total_expected += expected
        total_done += done
        complete = "DONE" if done >= expected and target > 0 else f"{done}/{expected}"
        print(f"  {key.label}: {complete} (N={target})")

    print(f"\n  Total: {total_done}/{total_expected} runs")
    print(f"{'='*50}\n")


# ── HTML dashboard ─────────────────────────────────────────────


def _metrics_to_json_row(m: ConfigMetrics, scenarios: list[str]) -> dict:
    """Convert a ConfigMetrics to a JSON-serialisable dict for the HTML dashboard."""
    return {
        "label": m.key.short_label,
        "model": m.key.model,
        "backend": m.key.backend,
        "mode": m.key.mode,
        "ablation": m.key.ablation,
        "family": extract_family(m.key.model),
        "quant": extract_quant(m.key.model),
        "score": round(m.score * 100, 1),
        "accuracy": round(m.accuracy * 100, 1) if m.accuracy is not None else None,
        "completeness": round(m.completeness * 100, 1),
        "efficiency": round(m.efficiency * 100, 1),
        "wasted": round(m.avg_wasted, 1),
        "speed": round(m.speed, 1),
        "n": m.runs_per_scenario,
        "scenarios": {
            sc: round(m.per_scenario_score[sc] * 100) if m.per_scenario_score.get(sc) is not None else None
            for sc in scenarios
        },
        "scenarioRuns": {sc: m.per_scenario_runs.get(sc, 0) for sc in scenarios},
        "scenarioCorrect": {sc: m.per_scenario_correct.get(sc, 0) for sc in scenarios},
        "scenarioCompleted": {sc: m.per_scenario_completed.get(sc, 0) for sc in scenarios},
        "scenarioValidated": {sc: m.per_scenario_validated.get(sc, 0) for sc in scenarios},
        "scenarioIdealCalls": {sc: m.per_scenario_ideal_calls.get(sc, 0) for sc in scenarios},
        "scenarioActualCalls": {sc: m.per_scenario_actual_calls.get(sc, 0) for sc in scenarios},
        "scenarioWastedSum": {sc: round(m.per_scenario_wasted_sum.get(sc, 0.0), 2) for sc in scenarios},
        "scenarioWastedN": {sc: m.per_scenario_wasted_n.get(sc, 0) for sc in scenarios},
        "scenarioSpeedSum": {sc: round(m.per_scenario_speed_sum.get(sc, 0.0), 2) for sc in scenarios},
        "scenarioSpeedN": {sc: m.per_scenario_speed_n.get(sc, 0) for sc in scenarios},
    }


_DASHBOARD_DIR = Path(__file__).parent / "dashboard"


def _build_dashboard() -> Path:
    """Build the React dashboard and return the path to dist/index.html."""
    import shutil
    import subprocess

    dist_html = _DASHBOARD_DIR / "dist" / "index.html"
    npm = shutil.which("npm")
    if not npm:
        raise FileNotFoundError(
            "npm not found on PATH. Install Node.js to generate the HTML dashboard."
        )

    if not (_DASHBOARD_DIR / "node_modules").exists():
        print("Installing dashboard dependencies...")
        subprocess.run(
            [npm, "install"],
            cwd=_DASHBOARD_DIR,
            check=True,
            capture_output=True,
        )

    print("Building dashboard...")
    result = subprocess.run(
        [npm, "run", "build"],
        cwd=_DASHBOARD_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Dashboard build failed:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError("Dashboard build failed")

    if not dist_html.exists():
        raise FileNotFoundError(f"Dashboard build failed: {dist_html} not found")

    return dist_html


def write_html(
    metrics_list: list[ConfigMetrics],
    scenarios: list[str],
    output_path: Path,
) -> None:
    """Build React dashboard, inject data, write self-contained HTML."""
    import datetime

    template_path = _build_dashboard()
    template = template_path.read_text(encoding="utf-8")

    rows = [_metrics_to_json_row(m, scenarios) for m in _sort_metrics(metrics_list)]
    sc_abbrev = {sc: _SCENARIO_ABBREV.get(sc, sc[:3]) for sc in scenarios}
    sc_suite = {
        sc: "advanced_reasoning" if sc in _AR_SCENARIOS else "og18"
        for sc in scenarios
    }

    data_blob = json.dumps({
        "rows": rows,
        "scenarios": scenarios,
        "scenarioAbbrev": sc_abbrev,
        "scenarioSuite": sc_suite,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    })

    # Inject data before the closing </head> tag so it's available before React boots
    inject = f'<script>window.__FORGE_DATA__ = {data_blob};</script>'
    html = template.replace("</head>", f"{inject}\n</head>", 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"HTML dashboard written to {output_path}")


# ── Markdown views ─────────────────────────────────────────────


# Ordering for ablation rows in the Full Ablation screen / markdown.
# Mirrors ABLATION_ORDER in the dashboard's types.ts.
_ABLATION_ORDER = (
    "reforged", "bare",
    "no_rescue", "no_nudge", "no_steps", "no_recovery", "no_compact",
)


def _ablation_rank(name: str) -> int:
    """Rank for sorting ablation rows; unknowns land last."""
    try:
        return _ABLATION_ORDER.index(name)
    except ValueError:
        return len(_ABLATION_ORDER)


def write_markdown_views(
    all_metrics: list[ConfigMetrics],
    scenarios: list[str],
    output_dir: Path,
) -> None:
    """Write pre-filtered markdown view files mirroring the dashboard's three screens.

    Layout:
        index.md
        reforged/
            all.md          — flat leaderboard, reforged rows only
            by-family.md    — reforged rows grouped by model family
            by-backend.md   — reforged rows, same model across backends
        reforged-vs-bare.md — per-(model,backend,mode) reforged+bare pair
        ablation.md         — deep-ablation configs only, 7-row tower per config
        native-vs-prompt.md — llama-server paired native vs prompt (reforged)
        budget.md           — compaction scenarios only (reforged)
    """
    import datetime

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reforged_dir = raw_dir / "reforged"
    reforged_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    legend = "\n".join(_legend_lines(scenarios))

    written: list[tuple[str, str]] = []  # (relpath-from-raw, description)

    complete = [m for m in all_metrics if m.complete]
    reforged_only = [m for m in complete if m.key.ablation == "reforged"]

    def _flat_view(relpath: str, title: str, description: str, metrics: list[ConfigMetrics], sc: list[str] | None = None) -> None:
        if not metrics:
            return
        sc = sc or scenarios
        table = render_table_string(metrics, sc, include_legend=False)
        content = f"# {title}\n\n```\n{table}```\n\n{legend}\n\n*Generated {timestamp}*\n"
        (raw_dir / relpath).write_text(content, encoding="utf-8")
        written.append((relpath, description))

    def _grouped_view(
        relpath: str,
        title: str,
        description: str,
        groups_sorted: list[tuple[str, list[ConfigMetrics]]],
    ) -> None:
        if not groups_sorted:
            return
        parts: list[str] = [f"# {title}\n"]
        for heading, group in groups_sorted:
            table = render_table_string(group, scenarios, include_legend=False, presorted=True)
            parts.append(f"## {heading}\n\n```\n{table}```\n")
        parts.append(legend)
        parts.append(f"\n*Generated {timestamp}*\n")
        (raw_dir / relpath).write_text("\n".join(parts), encoding="utf-8")
        written.append((relpath, description))

    # ── Screen 1: reforged/ ───────────────────────────────────────
    # reforged/all.md — flat leaderboard of reforged-only rows
    _flat_view(
        "reforged/all.md",
        "Forge Eval — Reforged Leaderboard",
        "Leaderboard: forge-reforged configs only",
        reforged_only,
    )

    # reforged/by-family.md — grouped by model family, reforged only
    family_groups: dict[str, list[ConfigMetrics]] = defaultdict(list)
    for m in reforged_only:
        family_groups[extract_family(m.key.model)].append(m)
    sorted_families = sorted(
        family_groups.items(),
        key=lambda kv: max(m.score for m in kv[1]),
        reverse=True,
    )
    _grouped_view(
        "reforged/by-family.md",
        "Forge Eval — Reforged by Model Family",
        "Reforged results grouped by model family",
        [(family, sorted(group, key=lambda m: -m.score)) for family, group in sorted_families],
    )

    # reforged/by-backend.md — reforged only, grouped by model when >1 backend available
    backend_groups: dict[str, list[ConfigMetrics]] = defaultdict(list)
    for m in reforged_only:
        backend_groups[m.key.model].append(m)
    backend_pairs = {k: v for k, v in backend_groups.items() if len({m.key.backend for m in v}) > 1}
    sorted_bg = sorted(backend_pairs.items(), key=lambda kv: max(m.score for m in kv[1]), reverse=True)
    _grouped_view(
        "reforged/by-backend.md",
        "Forge Eval — Reforged by Backend",
        "Same model across backends (reforged only)",
        [(model, sorted(group, key=lambda m: -m.score)) for model, group in sorted_bg],
    )

    # ── Screen 2: reforged-vs-bare.md ─────────────────────────────
    # Per-(model, backend, mode) groups. Always exactly 2 rows (reforged + bare).
    rb_groups: dict[tuple[str, str, str], list[ConfigMetrics]] = defaultdict(list)
    for m in complete:
        if m.key.ablation in ("reforged", "bare"):
            rb_groups[(m.key.model, m.key.backend, m.key.mode)].append(m)
    # Only keep configs with both variants present
    rb_pairs = {k: v for k, v in rb_groups.items() if len({m.key.ablation for m in v}) == 2}
    sorted_rb = sorted(rb_pairs.items(), key=lambda kv: max(m.score for m in kv[1]), reverse=True)
    _grouped_view(
        "reforged-vs-bare.md",
        "Forge Eval — Reforged vs Bare",
        "Forge lift: reforged vs bare for each (model, backend, mode)",
        [
            (f"{model} ({backend}/{mode})", sorted(group, key=lambda m: _ablation_rank(m.key.ablation)))
            for (model, backend, mode), group in sorted_rb
        ],
    )

    # ── Screen 3: ablation.md ─────────────────────────────────────
    # Deep-ablation configs only — those with at least one no_* variant.
    # Each config gets a table with all its ablation rows in ABLATION_ORDER.
    abl_groups: dict[tuple[str, str, str], list[ConfigMetrics]] = defaultdict(list)
    for m in complete:
        abl_groups[(m.key.model, m.key.backend, m.key.mode)].append(m)
    deep_ablation = {
        k: v for k, v in abl_groups.items()
        if any(m.key.ablation.startswith("no_") for m in v)
    }
    sorted_abl = sorted(deep_ablation.items(), key=lambda kv: max(m.score for m in kv[1]), reverse=True)
    _grouped_view(
        "ablation.md",
        "Forge Eval — Full Ablation",
        "Per-guardrail ablation: each config shows all ablation variants",
        [
            (f"{model} ({backend}/{mode})", sorted(group, key=lambda m: _ablation_rank(m.key.ablation)))
            for (model, backend, mode), group in sorted_abl
        ],
    )

    # ── Orthogonal: native-vs-prompt.md ───────────────────────────
    # llama-server only, reforged only, grouped by model, both modes present.
    ls_reforged = [m for m in reforged_only if m.key.backend == "llamaserver"]
    ls_groups: dict[str, list[ConfigMetrics]] = defaultdict(list)
    for m in ls_reforged:
        ls_groups[m.key.model].append(m)
    ls_paired = {k: v for k, v in ls_groups.items() if {"native", "prompt"} <= {m.key.mode for m in v}}
    _grouped_view(
        "native-vs-prompt.md",
        "Forge Eval — Native vs Prompt (llama-server)",
        "llama-server native FC vs prompt-injected, reforged only",
        [
            (model, sorted(group, key=lambda m: m.key.mode))
            for model, group in sorted(ls_paired.items())
        ],
    )

    # ── Orthogonal: budget.md ─────────────────────────────────────
    compaction_scenarios = [sc for sc in scenarios if sc in {
        "compaction_stress", "phase2_compaction",
        "compaction_stress_stateful", "phase2_compaction_stateful",
        "inventory_audit", "supplier_deep_dive",
    }]
    if compaction_scenarios:
        _flat_view(
            "budget.md",
            "Forge Eval — Compaction / Tight Budget",
            "Compaction scenario results (reforged only)",
            reforged_only,
            sc=compaction_scenarios,
        )

    # ── index.md ──────────────────────────────────────────────────
    if written:
        # Section header → predicate over (relpath, desc) entries in `written`.
        # A section is only emitted if at least one entry matches.
        sections: list[tuple[str, callable]] = [
            ("## Reforged — which model should I run?", lambda rp: rp.startswith("reforged/")),
            ("## Reforged vs Bare — how much does forge lift a model?", lambda rp: rp == "reforged-vs-bare.md"),
            ("## Full Ablation — which guardrails do the work?", lambda rp: rp == "ablation.md"),
            ("## Other cross-cuts", lambda rp: rp in ("native-vs-prompt.md", "budget.md")),
        ]
        index_lines = [
            "# Forge Eval Reports\n",
            "For model and backend recommendations, see [Model Guide](../MODEL_GUIDE.md).\n",
        ]
        for header, pred in sections:
            entries = [(rp, desc) for rp, desc in written if pred(rp)]
            if not entries:
                continue
            index_lines.append(f"{header}\n")
            for rp, desc in entries:
                index_lines.append(f"- [{rp}](raw/{rp}) — {desc}")
            index_lines.append("")
        index_lines.append(f"*Generated {timestamp}*\n")
        (output_dir / "index.md").write_text("\n".join(index_lines), encoding="utf-8")
        print(f"Markdown views written to {output_dir}/raw/ ({len(written)} files + index.md)")


# ── CLI ─────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Forge eval report")
    parser.add_argument(
        "jsonl", nargs="?", default="eval_results.jsonl", help="JSONL input"
    )
    parser.add_argument(
        "--list-only", action="store_true", help="Skip table, list view only"
    )
    parser.add_argument(
        "--progress", action="store_true",
        help="Show progress for all configs (including incomplete)",
    )
    parser.add_argument(
        "--include-partial", action="store_true",
        help="Include configs that haven't finished all scenarios",
    )
    parser.add_argument(
        "--ablation", nargs="*",
        help="Filter to specific ablation preset(s) (e.g. --ablation reforged bare). "
        "Default: show all.",
    )
    parser.add_argument(
        "--exclude-scenario", nargs="*", metavar="NAME",
        help="Exclude scenario(s) from aggregates and columns "
        "(e.g. --exclude-scenario error_recovery).",
    )
    parser.add_argument(
        "--tags", nargs="*", metavar="TAG",
        help="Filter to scenarios matching tag(s): 'stateful', 'lambda', 'compaction'. "
        "Filters rows before scenario detection.",
    )
    parser.add_argument(
        "--html", metavar="PATH",
        help="Write interactive HTML dashboard to PATH",
    )
    parser.add_argument(
        "--markdown", metavar="DIR",
        help="Write pre-filtered markdown views to DIR",
    )
    args = parser.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows = load_jsonl(path)
    if not rows:
        print("No data in JSONL file.")
        sys.exit(0)

    # Filter by ablation preset if requested
    if args.ablation:
        ablation_set = set(args.ablation)
        rows = [r for r in rows if r.get("ablation", "reforged") in ablation_set]
        if not rows:
            print(f"No data for ablation preset(s): {', '.join(args.ablation)}")
            sys.exit(0)

    # Filter rows by scenario tag before detection
    if args.tags:
        _TAG_FILTERS = {
            "stateful": lambda sc: sc.endswith("_stateful"),
            "lambda": lambda sc: not sc.endswith("_stateful") and sc not in ("compaction_stress", "phase2_compaction"),
            "compaction": lambda sc: sc in ("compaction_stress", "phase2_compaction", "compaction_stress_stateful", "phase2_compaction_stateful", "inventory_audit", "supplier_deep_dive"),
        }
        tag_preds = [_TAG_FILTERS[t] for t in args.tags if t in _TAG_FILTERS]
        if tag_preds:
            rows = [r for r in rows if any(p(r["scenario"]) for p in tag_preds)]
            if not rows:
                print(f"No data for tag(s): {', '.join(args.tags)}")
                sys.exit(0)

    scenarios = _detect_scenarios(rows)
    if args.exclude_scenario:
        exclude_set = set(args.exclude_scenario)
        scenarios = [sc for sc in scenarios if sc not in exclude_set]

    grouped = group_rows(rows)
    print(
        f"Loaded {len(rows)} rows across {len(grouped)} model/backend combos"
        f" ({len(scenarios)} scenario{'s' if len(scenarios) != 1 else ''}: "
        f"{', '.join(scenarios)})"
    )

    if args.progress:
        print_progress(grouped, scenarios=scenarios)

    # Compute per-scenario target run count (max across all configs)
    target_runs: dict[str, int] = {}
    for sc in scenarios:
        target_runs[sc] = max(
            (len(scenario_runs.get(sc, [])) for scenario_runs in grouped.values()),
            default=0,
        )

    # Compute metrics for all configs
    all_metrics: list[ConfigMetrics] = []
    for key, scenario_runs in grouped.items():
        m = compute_config_metrics(key, scenario_runs, scenarios=scenarios, target_runs=target_runs)
        all_metrics.append(m)

    # Filter to complete configs only (unless --include-partial)
    if args.include_partial:
        display_metrics = all_metrics
    else:
        display_metrics = [m for m in all_metrics if m.complete]

    if not display_metrics and not args.progress:
        print(
            "\nNo configs have completed all scenarios yet."
        )
        print("Use --progress to see current status, or --include-partial.")
        sys.exit(0)

    if display_metrics:
        if not args.list_only:
            print_table(display_metrics, scenarios=scenarios)
        print_list(display_metrics)

    # HTML dashboard (complete configs only, same as ASCII table)
    if args.html:
        write_html(display_metrics, scenarios, Path(args.html))

    # Markdown views (uses all metrics — write_markdown_views filters internally)
    if args.markdown:
        write_markdown_views(all_metrics, scenarios, Path(args.markdown))


if __name__ == "__main__":
    main()
