"""Eval metrics — aggregate results and print report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from forge.core.messages import Message, MessageType

if TYPE_CHECKING:
    from tests.eval.eval_runner import RunResult
    from tests.eval.scenarios import EvalScenario


# ── History analysis ─────────────────────────────────────────────


@dataclass
class HistoryStats:
    """Stats extracted from a single run's message history."""

    total_tool_calls: int = 0
    unique_tools_called: set[str] = field(default_factory=set)
    retry_nudges: int = 0
    step_nudges: int = 0
    tool_errors: int = 0
    reasoning_messages: int = 0


def analyze_history(messages: list[Message]) -> HistoryStats:
    """Extract stats from a run's message history."""
    stats = HistoryStats()
    for msg in messages:
        match msg.metadata.type:
            case MessageType.TOOL_CALL:
                stats.total_tool_calls += 1
                # Extract tool name from "[tool_call] name({...})"
                content = msg.content
                if content.startswith("[tool_call] "):
                    name = content[len("[tool_call] "):].split("(", 1)[0]
                    stats.unique_tools_called.add(name)
            case MessageType.RETRY_NUDGE:
                stats.retry_nudges += 1
            case MessageType.STEP_NUDGE:
                stats.step_nudges += 1
            case MessageType.TOOL_RESULT:
                if "[ToolError]" in msg.content:
                    stats.tool_errors += 1
            case MessageType.REASONING:
                stats.reasoning_messages += 1
    return stats


# ── Aggregated metrics ───────────────────────────────────────────


@dataclass
class ScenarioMetrics:
    """Aggregated metrics for one scenario across all runs."""

    scenario_name: str
    total_runs: int
    completed_runs: int
    completion_rate: float
    avg_iterations: float
    max_iterations: int
    avg_elapsed_seconds: float
    correctness_rate: float | None = None
    error_counts: dict[str, int] = field(default_factory=dict)
    runs_with_compaction: int = 0
    avg_compaction_phases: float = 0.0
    avg_retry_nudges: float = 0.0
    avg_step_nudges: float = 0.0
    avg_tool_errors: float = 0.0
    avg_wasted_calls: float | None = None
    avg_reasoning_messages: float = 0.0
    correctness_with_reasoning: float | None = None
    correctness_without_reasoning: float | None = None
    has_history: bool = False


def compute_metrics(
    scenario: EvalScenario,
    results: list[RunResult],
) -> ScenarioMetrics:
    """Compute aggregated metrics from a list of RunResults."""
    total = len(results)
    completed = [r for r in results if r.completeness]
    completed_count = len(completed)
    completion_rate = completed_count / total if total > 0 else 0.0

    # Iteration stats (from completed runs)
    avg_iters = (
        sum(r.iterations_used for r in completed) / completed_count
        if completed_count > 0
        else 0.0
    )
    max_iters = max((r.iterations_used for r in results), default=0)

    # Timing
    avg_elapsed = (
        sum(r.elapsed_seconds for r in results) / total if total > 0 else 0.0
    )

    # Error breakdown
    error_counts: dict[str, int] = {}
    for r in results:
        if r.error_type:
            error_counts[r.error_type] = error_counts.get(r.error_type, 0) + 1

    # Compaction stats
    runs_with_compaction = sum(1 for r in results if r.compaction_events)
    compaction_phases = [
        max(e.phase_reached for e in r.compaction_events)
        for r in results
        if r.compaction_events
    ]
    avg_compaction_phases = (
        sum(compaction_phases) / len(compaction_phases)
        if compaction_phases
        else 0.0
    )

    # History-based metrics (from on_message callback)
    has_history = any(r.messages is not None for r in results)
    avg_retry_nudges = 0.0
    avg_step_nudges = 0.0
    avg_tool_errors = 0.0
    avg_reasoning_messages = 0.0
    avg_wasted_calls: float | None = None

    if has_history:
        history_results = [r for r in results if r.messages is not None]
        stats_list = [analyze_history(r.messages) for r in history_results]  # type: ignore[arg-type]
        n = len(stats_list)
        if n > 0:
            avg_retry_nudges = sum(s.retry_nudges for s in stats_list) / n
            avg_step_nudges = sum(s.step_nudges for s in stats_list) / n
            avg_tool_errors = sum(s.tool_errors for s in stats_list) / n
            avg_reasoning_messages = sum(s.reasoning_messages for s in stats_list) / n

        # Wasted calls (completed runs with history only)
        ideal_calls = scenario.ideal_iterations or (len(scenario.workflow.required_steps) + 1)
        completed_with_history = [
            r for r in completed if r.messages is not None
        ]
        if completed_with_history:
            wasted = [
                max(0, r.iterations_used - ideal_calls)
                for r in completed_with_history
            ]
            avg_wasted_calls = sum(wasted) / len(wasted)
    elif completed_count > 0:
        # Derive wasted calls from iteration count alone
        ideal_calls = scenario.ideal_iterations or (len(scenario.workflow.required_steps) + 1)
        wasted = [
            max(0, r.iterations_used - ideal_calls) for r in completed
        ]
        avg_wasted_calls = sum(wasted) / len(wasted)

    # Correctness (only if scenario has a validator)
    correctness_rate: float | None = None
    correctness_with_reasoning: float | None = None
    correctness_without_reasoning: float | None = None
    validated_runs = [r for r in results if r.accuracy is not None]
    if validated_runs:
        correctness_rate = sum(1 for r in validated_runs if r.accuracy) / len(validated_runs)

        # Reasoning vs correctness correlation
        if has_history:
            with_reasoning = [
                r for r in validated_runs
                if r.messages and analyze_history(r.messages).reasoning_messages > 0
            ]
            without_reasoning = [
                r for r in validated_runs
                if r.messages is not None
                and analyze_history(r.messages).reasoning_messages == 0
            ]
            if with_reasoning:
                correctness_with_reasoning = (
                    sum(1 for r in with_reasoning if r.accuracy)
                    / len(with_reasoning)
                )
            if without_reasoning:
                correctness_without_reasoning = (
                    sum(1 for r in without_reasoning if r.accuracy)
                    / len(without_reasoning)
                )

    return ScenarioMetrics(
        scenario_name=scenario.name,
        total_runs=total,
        completed_runs=completed_count,
        completion_rate=completion_rate,
        avg_iterations=avg_iters,
        max_iterations=max_iters,
        avg_elapsed_seconds=avg_elapsed,
        correctness_rate=correctness_rate,
        error_counts=error_counts,
        runs_with_compaction=runs_with_compaction,
        avg_compaction_phases=avg_compaction_phases,
        avg_retry_nudges=avg_retry_nudges,
        avg_step_nudges=avg_step_nudges,
        avg_tool_errors=avg_tool_errors,
        avg_reasoning_messages=avg_reasoning_messages,
        correctness_with_reasoning=correctness_with_reasoning,
        correctness_without_reasoning=correctness_without_reasoning,
        avg_wasted_calls=avg_wasted_calls,
        has_history=has_history,
    )


# ── Report printing ──────────────────────────────────────────────


def print_report(
    all_results: dict[str, list[RunResult]],
    scenarios: list[EvalScenario] | None = None,
    model_name: str | None = None,
) -> None:
    """Print a human-readable eval report to stdout.

    If scenarios is provided, uses them for ideal-call computation.
    Otherwise, metrics that require scenario info show N/A.
    """
    # Build a name→scenario lookup
    scenario_map: dict[str, EvalScenario] = {}
    if scenarios:
        scenario_map = {s.name: s for s in scenarios}

    title = "Forge Eval Report"
    if model_name:
        title += f" — {model_name}"
    width = max(62, len(title) + 4)
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")

    for name, results in all_results.items():
        scenario = scenario_map.get(name)
        if scenario:
            metrics = compute_metrics(scenario, results)
            tags_str = ", ".join(scenario.tags) if scenario.tags else ""
            ideal = scenario.ideal_iterations or (len(scenario.workflow.required_steps) + 1)
        else:
            # Minimal metrics without scenario info
            metrics = ScenarioMetrics(
                scenario_name=name,
                total_runs=len(results),
                completed_runs=sum(1 for r in results if r.completeness),
                completion_rate=(
                    sum(1 for r in results if r.completeness) / len(results)
                    if results
                    else 0.0
                ),
                avg_iterations=(
                    sum(r.iterations_used for r in results if r.completeness)
                    / max(1, sum(1 for r in results if r.completeness))
                ),
                max_iterations=max(
                    (r.iterations_used for r in results), default=0
                ),
                avg_elapsed_seconds=(
                    sum(r.elapsed_seconds for r in results) / len(results)
                    if results
                    else 0.0
                ),
            )
            tags_str = ""
            ideal = None

        header = f" {name}"
        if tags_str:
            header += f" ({tags_str})"
        print(f"\n--- {header} {'-' * max(1, width - len(header) - 5)}")

        rate_pct = metrics.completion_rate * 100
        print(
            f"  Completion rate:    {metrics.completed_runs}/{metrics.total_runs} "
            f"({rate_pct:.1f}%)"
        )

        if metrics.correctness_rate is not None:
            cor_pct = metrics.correctness_rate * 100
            validated = sum(
                1 for r in results if r.accuracy is not None
            )
            correct = sum(1 for r in results if r.accuracy)
            print(
                f"  Correctness rate:   {correct}/{validated} "
                f"({cor_pct:.1f}%)"
            )

        ideal_str = f" (ideal: {ideal})" if ideal is not None else ""
        print(f"  Avg iterations:     {metrics.avg_iterations:.1f}{ideal_str}")

        if metrics.avg_wasted_calls is not None:
            print(f"  Wasted calls:       {metrics.avg_wasted_calls:.1f} avg")

        print(f"  Avg time:           {metrics.avg_elapsed_seconds:.1f}s")

        # History-based stats
        if metrics.has_history:
            print(f"  Retry nudges:       {metrics.avg_retry_nudges:.1f} avg")
            print(f"  Step nudges:        {metrics.avg_step_nudges:.1f} avg")
            print(f"  Tool errors:        {metrics.avg_tool_errors:.1f} avg")
            if metrics.avg_reasoning_messages > 0:
                print(f"  Reasoning msgs:     {metrics.avg_reasoning_messages:.1f} avg")
                if (metrics.correctness_with_reasoning is not None
                        or metrics.correctness_without_reasoning is not None):
                    parts = []
                    if metrics.correctness_with_reasoning is not None:
                        parts.append(f"w/ {metrics.correctness_with_reasoning*100:.0f}%")
                    if metrics.correctness_without_reasoning is not None:
                        parts.append(f"w/o {metrics.correctness_without_reasoning*100:.0f}%")
                    print(f"  Reasoning->correct: {', '.join(parts)}")

        # Compaction
        if metrics.runs_with_compaction > 0:
            print(
                f"  Compaction fired:   {metrics.runs_with_compaction}/"
                f"{metrics.total_runs} runs"
            )
            print(
                f"  Avg compaction phases: {metrics.avg_compaction_phases:.1f}"
            )

        # Errors
        if metrics.error_counts:
            parts = [
                f"{etype}: {count}"
                for etype, count in metrics.error_counts.items()
            ]
            print(f"  Errors:             {', '.join(parts)}")
        else:
            print(f"  Errors:             none")

    print(f"\n{'=' * width}\n")
