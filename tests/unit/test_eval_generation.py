"""Regression coverage for the shared eval-generation contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.eval import report
from tests.eval.generation import (
    GenerationMaxima,
    accumulate_generation_maxima,
    effective_generation,
    effective_reasoning_replay,
    is_selected_generation,
    selection_maximum_generation,
    select_latest_generation,
)


def _row(marker: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "marker": marker,
        "model": "model-a",
        "backend": "backend-a",
        "mode": "native",
        "scenario": "basic_2step",
    }
    row.update(overrides)
    return row


def _assert_selected(rows: list[dict[str, Any]], expected_markers: list[str]) -> None:
    selected = select_latest_generation(rows)
    assert [row["marker"] for row in selected] == expected_markers


def _report_row(**overrides: Any) -> dict[str, Any]:
    row = _row(
        "report",
        completed=False,
        correct=None,
        validation_error=None,
    )
    row.update(overrides)
    return row


def test_effective_legacy_defaults() -> None:
    row = _row("legacy")

    assert effective_generation(row) == 0
    assert effective_reasoning_replay(row) == "full"
    assert effective_generation(_row("modern", gen=3)) == 3
    assert (
        effective_reasoning_replay(_row("modern", reasoning_replay="keep-last"))
        == "keep-last"
    )


@pytest.mark.parametrize(
    ("field", "default", "nondefault"),
    [
        ("reasoning_level", "default", "high"),
        ("ablation", "reforged", "bare"),
        ("tool_choice", "auto", "required"),
    ],
)
def test_selection_normalizes_defaults_but_separates_nondefaults(
    field: str, default: str, nondefault: str
) -> None:
    missing_old = _row("missing-old", gen=1)
    distinct_old = _row("distinct-old", gen=1, **{field: nondefault})
    explicit_default_new = _row("default-new", gen=2, **{field: default})

    _assert_selected(
        [missing_old, distinct_old, explicit_default_new],
        ["distinct-old", "default-new"],
    )


def test_missing_generation_and_equal_generation_legacy_rows_are_retained() -> None:
    missing = _row("missing")
    explicit_zero = _row("zero", gen=0)

    _assert_selected([missing, explicit_zero], ["missing", "zero"])


def test_newer_sweep_supersedes_legacy_row_regardless_of_replay_policies() -> None:
    legacy = _row("legacy", gen=1)
    newer_none = _row("none", gen=2, reasoning_replay="none")
    newer_keep = _row("keep", gen=2, reasoning_replay="keep-last")

    _assert_selected([legacy, newer_none, newer_keep], ["none", "keep"])


def test_explicit_arm_uses_same_policy_supersession_and_other_arm_carries() -> None:
    old_none = _row("old-none", gen=1, reasoning_replay="none")
    old_full = _row("old-full", gen=1, reasoning_replay="full")
    new_none = _row("new-none", gen=2, reasoning_replay="none")

    _assert_selected([old_none, old_full, new_none], ["old-full", "new-none"])


def test_older_configuration_without_newer_counterpart_carries_forward() -> None:
    old_unique = _row("old-unique", model="model-unique", gen=1)
    old_repeated = _row("old-repeated", gen=1)
    new_repeated = _row("new-repeated", gen=2)

    _assert_selected(
        [old_unique, old_repeated, new_repeated],
        ["old-unique", "new-repeated"],
    )


def test_equal_generation_explicit_rows_and_interleaved_order_are_retained() -> None:
    first = _row("first", gen=2, reasoning_replay="full")
    interleaved = _row("interleaved", model="model-b", gen=1)
    second = _row("second", gen=2, reasoning_replay="full")
    dropped = _row("dropped", model="model-b", gen=0)

    _assert_selected(
        [first, interleaved, second, dropped],
        ["first", "interleaved", "second"],
    )


def test_bounded_maxima_and_predicate_match_selector() -> None:
    old_none = _row("old-none", gen=1, reasoning_replay="none")
    carried_full = _row("carried-full", gen=1, reasoning_replay="full")
    legacy = _row("legacy", gen=1)
    new_none = _row("new-none", gen=2, reasoning_replay="none")
    rows = [old_none, carried_full, legacy, new_none]
    maxima = GenerationMaxima()
    for row in rows:
        accumulate_generation_maxima(maxima, row)

    assert [row for row in rows if is_selected_generation(maxima, row)] == [
        carried_full,
        new_none,
    ]
    assert selection_maximum_generation(maxima, old_none) == 2
    assert selection_maximum_generation(maxima, legacy) == 2


def test_report_generation_legend_exposes_provenance() -> None:
    legend = "\n".join(report._gen_legend_lines([SimpleNamespace(gen=2)], max_gen=3))

    assert "gen 2" in legend
    assert "655e1f6" in legend
    assert "2026-05-22" in legend


def test_report_groups_legacy_replay_as_full() -> None:
    legacy = _report_row()

    grouped = report.group_rows([legacy])
    key, scenarios = next(iter(grouped.items()))

    assert key.reasoning_replay == "full"
    assert scenarios["basic_2step"] == [legacy]


def test_report_replay_filter_includes_legacy_as_full_and_excludes_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "legacy.jsonl"
    source.write_text(json.dumps(_report_row()) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["report", str(source), "--reasoning-replay", "full", "--list-only"],
    )
    report.main()
    assert "Loaded 1 rows" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["report", str(source), "--reasoning-replay", "none", "--list-only"],
    )
    with pytest.raises(SystemExit) as exit_info:
        report.main()
    assert exit_info.value.code == 0
    assert "No data for reasoning_replay polic(ies): none" in capsys.readouterr().out


def test_report_metric_aggregation_uses_effective_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _report_row()
    grouped = report.group_rows([row])
    key, scenario_runs = next(iter(grouped.items()))
    monkeypatch.setattr(report, "effective_generation", lambda _row: 7)

    metrics = report.compute_config_metrics(
        key, scenario_runs, scenarios=["basic_2step"]
    )

    assert metrics.gen == 7
