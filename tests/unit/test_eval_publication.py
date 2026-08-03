"""Focused regression coverage for the eval publication contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.eval import publication, report
from tests.eval.provenance import GEN_INFO


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ablation": "reforged",
        "accuracy": True,
        "backend": "backend-a",
        "budget_tokens": 100,
        "compaction_events": 0,
        "completeness": True,
        "elapsed_s": 1.25,
        "error_message": None,
        "error_type": None,
        "gen": 1,
        "ideal_iterations": 2,
        "iterations": 2,
        "mode": "native",
        "model": "model-a",
        "reasoning_msgs": 1,
        "retry_nudges": 0,
        "run": 1,
        "scenario": "same-scenario",
        "step_nudges": 0,
        "tool_choice": "auto",
        "tool_errors": 0,
        "wasted_calls": 0,
    }
    row.update(overrides)
    return row


def _encode(row: Any) -> bytes:
    return json.dumps(
        row,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_source(
    root: Path,
    name: str,
    rows: list[dict[str, Any]],
    *,
    terminator: bytes = b"\n",
    final_terminator: bool = True,
) -> publication.SourceSpec:
    chunks = []
    for index, row in enumerate(rows):
        suffix = terminator if final_terminator or index < len(rows) - 1 else b""
        chunks.append(_encode(row) + suffix)
    payload = b"".join(chunks)
    (root / name).write_bytes(payload)
    generations = {row["gen"] for row in rows}
    assert len(generations) == 1
    generation = generations.pop()
    return publication.SourceSpec(
        name,
        "synthetic",
        generation,
        len(rows),
        hashlib.sha256(payload).hexdigest(),
    )


def _single_plan(
    tmp_path: Path,
    row: dict[str, Any] | None = None,
    *,
    terminator: bytes = b"\n",
    final_terminator: bool = True,
) -> tuple[publication.PublicationPlan, publication.SourceSpec]:
    source_row = row or _row()
    spec = _write_source(
        tmp_path,
        "synthetic.jsonl",
        [source_row],
        terminator=terminator,
        final_terminator=final_terminator,
    )
    return publication.build_publication_plan(tmp_path, source_specs=[spec]), spec


def test_schema_is_exact_and_shared_provenance_has_object_identity() -> None:
    assert len(publication.SOURCE_SCHEMA) == 33
    assert len(publication.NORMALIZED_SCHEMA) == 51
    assert [field.name for field in publication.SOURCE_SCHEMA] == [
        "ablation",
        "accuracy",
        "backend",
        "budget_tokens",
        "compaction_events",
        "completeness",
        "elapsed_s",
        "error_message",
        "error_type",
        "gen",
        "ideal_iterations",
        "iterations",
        "mode",
        "model",
        "reasoning_msgs",
        "retry_nudges",
        "run",
        "scenario",
        "step_nudges",
        "tool_choice",
        "tool_errors",
        "wasted_calls",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "cost_usd",
        "input_tokens",
        "output_tokens",
        "reasoning_level",
        "reasoning_replay",
        "reasoning_wire",
        "reasoning_wire_total",
        "rig",
        "stream_retries",
    ]
    assert report.GEN_INFO is publication.GEN_INFO is GEN_INFO

    fields = {field.name: field for field in publication.SOURCE_SCHEMA}
    assert {name for name, field in fields.items() if not field.source_required} == {
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "cost_usd",
        "input_tokens",
        "output_tokens",
        "reasoning_level",
        "reasoning_replay",
        "reasoning_wire",
        "reasoning_wire_total",
        "rig",
        "stream_retries",
    }
    assert {name for name, field in fields.items() if field.source_nullable} == {
        "accuracy",
        "error_message",
        "error_type",
        "reasoning_msgs",
        "reasoning_wire",
        "reasoning_wire_total",
        "retry_nudges",
        "step_nudges",
        "tool_errors",
        "wasted_calls",
    }
    assert {name for name, field in fields.items() if field.logical_type == "bool"} == {
        "accuracy",
        "completeness",
    }
    assert {name for name, field in fields.items() if field.logical_type == "float64"} == {
        "cost_usd",
        "elapsed_s",
    }
    assert fields["accuracy"].source_nullable
    assert fields["reasoning_wire"].source_nullable
    assert not fields["rig"].source_required
    assert not fields["rig"].source_nullable
    assert fields["rig"].normalized_nullable
    assert fields["run"].source_required
    assert not fields["run"].source_nullable

    manifest = publication.schema_manifest()
    assert manifest["sha256"] == publication.schema_fingerprint()
    assert manifest["enums"]["reasoning_replay"] == ["none", "keep-last", "full"]


def test_sparse_fields_normalize_to_null_and_run_is_untouched(tmp_path: Path) -> None:
    plan, _ = _single_plan(tmp_path, _row(run=-7, accuracy=None, wasted_calls=None))
    normalized = list(publication.iter_history(plan))[0]

    for name in (
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "cost_usd",
        "input_tokens",
        "output_tokens",
        "reasoning_level",
        "reasoning_replay",
        "reasoning_wire",
        "reasoning_wire_total",
        "rig",
        "stream_retries",
    ):
        assert normalized[name] is None
    assert normalized["effective_reasoning_replay"] == "full"
    assert normalized["effective_reasoning_level"] == "default"
    assert normalized["run"] == -7


@pytest.mark.parametrize(
    ("line", "category"),
    [
        (b"", "blank_line"),
        (b"{", "malformed_json"),
        (b"[]", "non_object"),
        (b'{"x":1,"x":2}', "duplicate_key"),
        (b'{"elapsed_s":NaN}', "non_standard_constant"),
        (b'{"elapsed_s":Infinity}', "non_standard_constant"),
        (b"\xff", "invalid_utf8"),
    ],
)
def test_strict_parser_rejects_invalid_json(
    line: bytes, category: str
) -> None:
    with pytest.raises(publication.PublicationError, match=category) as exc_info:
        publication.parse_source_line(line, "fixture.jsonl", 9)
    assert str(exc_info.value).startswith("fixture.jsonl:9:")


def test_overflow_float_is_rejected_as_non_finite() -> None:
    raw = _encode(_row()).replace(b'"elapsed_s":1.25', b'"elapsed_s":1e999')
    with pytest.raises(publication.PublicationError, match="non_finite"):
        publication.parse_source_line(raw, "fixture.jsonl", 1)


@pytest.mark.parametrize(
    ("mutate", "category"),
    [
        (lambda row: row.__setitem__("unknown", "secret-host-value"), "unknown_field"),
        (lambda row: row.pop("backend"), "missing_required_field"),
        (lambda row: row.__setitem__("budget_tokens", True), "wrong_type"),
        (lambda row: row.__setitem__("rig", None), "null_not_allowed"),
        (lambda row: row.__setitem__("reasoning_replay", "mystery"), "unknown_policy"),
    ],
)
def test_schema_errors_fail_closed_and_are_redacted(mutate: Any, category: str) -> None:
    row = _row()
    mutate(row)
    with pytest.raises(publication.PublicationError, match=category) as exc_info:
        publication.parse_source_line(_encode(row), "fixture.jsonl", 3)
    message = str(exc_info.value)
    assert "secret-host-value" not in message
    assert "model-a" not in message


def test_source_generation_and_generation_metadata_are_strict(tmp_path: Path) -> None:
    spec = _write_source(tmp_path, "wrong-gen.jsonl", [_row(gen=1)])
    mismatched = publication.SourceSpec(
        spec.source_file, spec.release, 2, spec.row_count, spec.sha256
    )
    with pytest.raises(
        publication.PublicationError, match="source_generation_mismatch"
    ):
        publication.build_publication_plan(tmp_path, source_specs=[mismatched])

    unknown = publication.SourceSpec("unused.jsonl", "synthetic", 99, 0, "0" * 64)
    with pytest.raises(publication.PublicationError, match="unknown_generation"):
        publication.build_publication_plan(tmp_path, source_specs=[unknown])


@pytest.mark.parametrize(
    ("terminator", "final_terminator"),
    [(b"\n", True), (b"\r\n", True), (b"\n", False)],
)
def test_physical_line_terminators_do_not_enter_source_row_hash(
    tmp_path: Path, terminator: bytes, final_terminator: bool
) -> None:
    row = _row(model="mödèl")
    plan, _ = _single_plan(
        tmp_path,
        row,
        terminator=terminator,
        final_terminator=final_terminator,
    )
    normalized = list(publication.iter_history(plan))[0]
    assert normalized["source_row_sha256"] == (
        "38ce2cddbf8797d3109338e2f0524e5a041890ad25a305cdfc6f9f26239eb9fc"
    )


def test_canonical_json_and_non_ascii_id_framing(tmp_path: Path) -> None:
    assert publication.canonical_json({"z": "é", "a": 1}, sort_keys=True) == (
        '{"a":1,"z":"é"}'
    )
    with pytest.raises(ValueError):
        publication.canonical_json({"bad": float("nan")}, sort_keys=True)

    plan, _ = _single_plan(tmp_path, _row(model="mödèl"))
    normalized = list(publication.iter_history(plan))[0]
    base_frame = '["mödèl","backend-a","native","reforged","auto","default"]'
    arm_frame = (
        '["mödèl","backend-a","native","reforged","auto","default",'
        '1,"legacy-inferred-full"]'
    )
    assert normalized["base_config_id"] == hashlib.sha256(
        base_frame.encode("utf-8")
    ).hexdigest()
    assert normalized["arm_id"] == hashlib.sha256(
        arm_frame.encode("utf-8")
    ).hexdigest()


def test_all_statuses_truth_table_order_and_repeated_labels(tmp_path: Path) -> None:
    gen1_rows = [
        _row(model="shared", gen=1),
        _row(model="shared", gen=1, reasoning_replay="full"),
        _row(model="shared", gen=1, reasoning_replay="none"),
        _row(model="carried", gen=1),
    ]
    gen2_rows = [
        _row(model="shared", gen=2, reasoning_replay="none"),
        _row(model="equal", gen=2, reasoning_replay="full"),
        _row(model="equal", gen=2, reasoning_replay="full"),
    ]
    specs = [
        _write_source(tmp_path, "gen1.jsonl", gen1_rows),
        _write_source(tmp_path, "gen2.jsonl", gen2_rows),
    ]
    plan = publication.build_publication_plan(tmp_path, source_specs=specs)
    rows = list(publication.iter_classified_history(plan))

    assert [row["selection_status"] for row in rows] == [
        "superseded_base",
        "carried",
        "superseded_same_policy",
        "carried",
        "latest",
        "latest",
        "latest",
    ]
    assert [row["source_file"] for row in rows] == ["gen1.jsonl"] * 4 + [
        "gen2.jsonl"
    ] * 3
    assert [row["source_line"] for row in rows] == [1, 2, 3, 4, 1, 2, 3]
    assert all(row["run"] == 1 and row["scenario"] == "same-scenario" for row in rows)

    legacy, explicit_full, old_none = rows[:3]
    assert legacy["superseded_by_scope"] == "base_configuration"
    assert legacy["superseded_by_generation"] == 2
    assert explicit_full["selected_for_snapshot"]
    assert explicit_full["superseded_by_scope"] is None
    assert old_none["superseded_by_scope"] == "explicit_policy"
    assert old_none["superseded_by_generation"] == 2
    assert legacy["base_config_id"] == explicit_full["base_config_id"]
    assert legacy["arm_id"] != explicit_full["arm_id"]
    assert all(
        row["comparable_to_latest"] == row["selected_for_latest"] for row in rows
    )

    assert list(plan.status_counts.values()) == [3, 2, 1, 1]
    assert len(list(publication.iter_snapshot(plan))) == 5
    assert len(list(publication.iter_latest(plan))) == 3
    assert [row["model"] for row in publication.iter_snapshot(plan)] == [
        "shared",
        "carried",
        "shared",
        "equal",
        "equal",
    ]


def test_plan_and_logical_digest_are_deterministic_golden(tmp_path: Path) -> None:
    plan, _ = _single_plan(tmp_path, _row(model="mödèl"), final_terminator=False)
    again = publication.build_publication_plan(
        tmp_path, source_specs=plan._source_specs
    )
    assert plan.to_json() == again.to_json()
    assert plan.views["history"].source_identity_sha256 == (
        "b5bd1fe2a7c3ad03dcdbe4d4d56fb19b7cb580efe2d8a9faeffd5b7fab447784"
    )
    assert plan.views["history"].source_payload_sha256 == (
        "13acbcacb2ce55e8a8e0e0d9b31eef9dcf2307d3990849a1092de77153166d11"
    )
    assert plan.views["history"].normalized_logical_sha256 == (
        "c682dd15cee85edd884f7cc7c89fef84201bd032f67c4355ed8fa6240b7d875f"
    )


def test_changed_or_truncated_source_fails_when_iterators_are_exhausted(
    tmp_path: Path,
) -> None:
    original = _row()
    plan, spec = _single_plan(tmp_path, original)
    (tmp_path / spec.source_file).write_bytes(_encode(_row(run=2)) + b"\n")
    with pytest.raises(publication.PublicationError, match="hash_mismatch"):
        list(publication.iter_classified_history(plan))

    plan, spec = _single_plan(tmp_path, original)
    (tmp_path / spec.source_file).write_bytes(b"")
    with pytest.raises(publication.PublicationError, match="count_mismatch"):
        list(publication.iter_snapshot(plan))


def test_unknown_view_fails() -> None:
    with pytest.raises(ValueError, match="unknown publication view"):
        publication.row_in_view({}, "other")
