"""Semantic regression tests for the public eval metric vocabulary."""

from __future__ import annotations

import pytest

from tests.eval import report, significance


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model": "model-q4",
        "backend": "llamaserver",
        "mode": "native",
        "scenario": "basic_2step",
        "run": 0,
        "correct": None,
        "completed": False,
        "validation_error": None,
        "iterations": 2,
        "ideal_iterations": 2,
        "wasted_calls": None,
        "elapsed_s": 1.0,
    }
    row.update(overrides)
    return row


def test_report_keeps_all_three_denominators_independent() -> None:
    scenario_rows = [
        _row(run=0, correct=True, completed=False),
        {
            **_row(run=1),
            "accuracy": False,
            "completeness": True,
            "validate_error": None,
        },
        _row(run=2, correct=None, completed=True),
        {
            **_row(run=3),
            "accuracy": False,
            "completeness": False,
            "validate_error": "ValueError",
        },
    ]
    for row in scenario_rows[1::2]:
        row.pop("correct")
        row.pop("completed")
        row.pop("validation_error")

    metrics = report.compute_config_metrics(
        report.ConfigKey("model-q4", "llamaserver", "native"),
        {"basic_2step": scenario_rows},
        scenarios=["basic_2step"],
    )

    assert metrics.attempted_count == 4
    assert metrics.correct_count == 1
    assert metrics.validated_count == 2
    assert metrics.completed_count == 2
    assert metrics.validation_error_count == 1
    assert metrics.score == pytest.approx(1 / 4)
    assert metrics.validated_accuracy == pytest.approx(1 / 2)
    assert metrics.completion_rate == pytest.approx(1 / 2)
    assert metrics.efficiency == 0.0  # correct-but-incomplete is not efficient work

    dashboard = report._metrics_to_json_row(metrics, ["basic_2step"])
    assert dashboard["score"] == 25.0
    assert dashboard["validatedAccuracy"] == 50.0
    assert dashboard["completionRate"] == 50.0
    assert dashboard["attemptedCount"] == 4
    assert dashboard["correctCount"] == 1
    assert dashboard["validatedCount"] == 2
    assert dashboard["completedCount"] == 2
    assert dashboard["scenarioAttempted"] == {"basic_2step": 4}
    assert "accuracy" not in dashboard
    assert "completeness" not in dashboard
    assert "scenarioRuns" not in dashboard


def test_report_labels_only_explicit_reasoning_levels() -> None:
    legacy = report.ConfigKey("model-q4", "llamaserver", "native")
    explicit_default = report.ConfigKey(
        "model-q4", "llamaserver", "native", reasoning_level="default"
    )

    assert legacy.short_label == explicit_default.short_label
    assert "@" not in legacy.short_label
    for level in ("low", "medium", "high", "xhigh"):
        key = report.ConfigKey(
            "model-q4", "llamaserver", "native", reasoning_level=level
        )
        assert f"@{level}]" in key.short_label


def test_significance_identity_does_not_pool_replay_or_reasoning() -> None:
    base = _row()
    none = significance.configuration_identity(
        {**base, "reasoning_replay": "none"}
    )
    full = significance.configuration_identity(
        {**base, "reasoning_replay": "full"}
    )
    high = significance.configuration_identity(
        {**base, "reasoning_replay": "none", "reasoning_level": "high"}
    )

    assert none != full
    assert none != high


def test_significance_reads_both_outcome_dialects() -> None:
    rows = [
        {**_row(run=0, correct=True, completed=True), "ablation": "reforged"},
        {**_row(run=1, correct=False, completed=True), "ablation": "reforged"},
        {
            **_row(run=0),
            "ablation": "bare",
            "accuracy": False,
            "completeness": True,
        },
        {
            **_row(run=1),
            "ablation": "bare",
            "accuracy": False,
            "completeness": True,
        },
    ]
    for row in rows[2:]:
        row.pop("correct")
        row.pop("completed")
        row.pop("validation_error")

    analysis = significance.analyze_config(rows)

    assert analysis[0]["ablation"] == "reforged"
    assert analysis[0]["score"] == 0.5
    assert analysis[1]["ablation"] == "bare"
    assert analysis[1]["score"] == 0.0
    assert analysis[1]["paired_n"] == 2
    assert analysis[1]["discordant_b"] == 1
    assert analysis[1]["discordant_c"] == 0
