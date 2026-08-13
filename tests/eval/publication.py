"""Strict, streaming publication contract for the released Forge eval corpus.

This module deliberately lives beside the eval harness and has no Forge
runtime or optional writer dependencies.  A publication plan takes two
bounded-memory passes: one to verify sources/build generation maxima, and one
to classify rows and calculate logical digests.  Iterators perform the same
strict verification again and must be exhausted for their final file
hash/count checks to run.

Run the read-only pinned-corpus audit with::

    python -m tests.eval.publication --source-root .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.eval.generation import (
    GenerationMaxima,
    accumulate_generation_maxima,
    base_configuration_identity,
    effective_generation,
    effective_reasoning_replay,
    is_selected_generation,
    selection_maximum_generation,
)
from tests.eval.provenance import GEN_INFO
from tests.eval.outcomes import (
    CANONICAL_DIALECT,
    LEGACY_DIALECT,
    OUTCOME_DIALECTS,
    OutcomeDialect,
    OutcomeSchemaError,
    read_outcome,
)


CONTRACT_VERSION = "forge-eval-publication-v2"
SCHEMA_VERSION = 2
REASONING_REPLAY_POLICIES = ("none", "keep-last", "full")
SELECTION_STATUSES = (
    "latest",
    "carried",
    "superseded_same_policy",
    "superseded_base",
)
SUPERSESSION_SCOPES = ("explicit_policy", "base_configuration")
VIEW_NAMES = ("history", "snapshot", "latest")

PINNED_VIEW_COUNTS = {
    "history": 529_100,
    "snapshot": 388_700,
    "latest": 270_400,
}
PINNED_VIEW_METRIC_COUNTS = {
    "history": (324_967, 439_472, 439_472),
    "snapshot": (236_020, 322_929, 322_929),
    "latest": (184_867, 238_244, 238_244),
}
PINNED_SNAPSHOT_SOURCE_IDENTITY_SHA256 = (
    "bfbf1cbc2fef6b35c71debf747ef5c8d0412450f48973f9c0629c20e489ae4b5"
)
PINNED_SNAPSHOT_SOURCE_PAYLOAD_SHA256 = (
    "a84e5428cfb058dcc71e403286fab4d6858fb495e6e33eba337d5d836bbda545"
)


@dataclass(frozen=True)
class SourceField:
    """One allowed source field and its strict JSON value contract.

    ``source_required`` controls omission in JSON. ``source_nullable`` controls
    an explicit JSON null.  Every omitted source field is present as null in a
    normalized row, so normalized nullability is the union of those flags.
    """

    name: str
    logical_type: str
    source_required: bool
    source_nullable: bool = False

    @property
    def normalized_nullable(self) -> bool:
        return self.source_nullable or not self.source_required


# Outcome aliases are dialect-specific; all other source fields are shared.
COMMON_SOURCE_SCHEMA: tuple[SourceField, ...] = (
    SourceField("ablation", "string", True),
    SourceField("backend", "string", True),
    SourceField("budget_tokens", "int64", True),
    SourceField("compaction_events", "int64", True),
    SourceField("elapsed_s", "float64", True),
    SourceField("error_message", "string", True, True),
    SourceField("error_type", "string", True, True),
    SourceField("gen", "int64", True),
    SourceField("ideal_iterations", "int64", True),
    SourceField("iterations", "int64", True),
    SourceField("mode", "string", True),
    SourceField("model", "string", True),
    SourceField("reasoning_msgs", "int64", True, True),
    SourceField("retry_nudges", "int64", True, True),
    SourceField("run", "int64", True),
    SourceField("scenario", "string", True),
    SourceField("step_nudges", "int64", True, True),
    SourceField("tool_choice", "string", True),
    SourceField("tool_errors", "int64", True, True),
    SourceField("wasted_calls", "int64", True, True),
    SourceField("cache_creation_input_tokens", "int64", False),
    SourceField("cache_read_input_tokens", "int64", False),
    SourceField("cost_usd", "float64", False),
    SourceField("input_tokens", "int64", False),
    SourceField("output_tokens", "int64", False),
    SourceField("reasoning_level", "string", False),
    SourceField("reasoning_replay", "string", False),
    SourceField("reasoning_wire", "int64", False, True),
    SourceField("reasoning_wire_total", "int64", False, True),
    SourceField("rig", "string", False),
    SourceField("stream_retries", "int64", False),
)

SOURCE_DIALECT_SCHEMAS: dict[OutcomeDialect, tuple[SourceField, ...]] = {
    LEGACY_DIALECT: (
        SourceField("accuracy", "bool", True, True),
        SourceField("completeness", "bool", True),
        SourceField("validate_error", "string", False, True),
    ),
    CANONICAL_DIALECT: (
        SourceField("correct", "bool", True, True),
        SourceField("completed", "bool", True),
        SourceField("validation_error", "string", False, True),
    ),
}


def source_schema(dialect: OutcomeDialect) -> tuple[SourceField, ...]:
    try:
        return COMMON_SOURCE_SCHEMA + SOURCE_DIALECT_SCHEMAS[dialect]
    except KeyError as exc:  # pragma: no cover - guarded by SourceSpec validation
        raise PublicationError(f"manifest:0: outcome dialect [unknown_dialect]") from exc


@dataclass(frozen=True)
class NormalizedField:
    """Stable logical type and nullability for one publication column."""

    name: str
    logical_type: str
    nullable: bool
    semantics: str


_COMMON_NORMALIZED_SCHEMA = tuple(
    NormalizedField(
        source.name,
        source.logical_type,
        source.normalized_nullable,
        "Source value preserved; an omitted sparse source field becomes null.",
    )
    for source in COMMON_SOURCE_SCHEMA
)

_OUTCOME_NORMALIZED_SCHEMA: tuple[NormalizedField, ...] = (
    NormalizedField(
        "correct",
        "bool",
        True,
        "Usable correctness verdict; null means the run was not validated.",
    ),
    NormalizedField(
        "completed",
        "bool",
        False,
        "Whether the workflow returned normally, independent of correctness.",
    ),
    NormalizedField(
        "validation_error",
        "string",
        True,
        "Validator exception category; non-null implies correct is null.",
    ),
)

_PROVENANCE_SCHEMA: tuple[NormalizedField, ...] = (
    NormalizedField("source_file", "string", False, "Pinned source basename."),
    NormalizedField("source_line", "int64", False, "One-based physical line."),
    NormalizedField(
        "source_file_sha256", "string", False, "SHA-256 of the complete source file."
    ),
    NormalizedField(
        "source_row_sha256",
        "string",
        False,
        "SHA-256 of physical line bytes excluding one LF or CRLF terminator.",
    ),
    NormalizedField("release", "string", False, "Pinned Forge release version."),
    NormalizedField(
        "generation_reference",
        "string",
        False,
        "Commit or tag reference from the shared generation provenance.",
    ),
    NormalizedField(
        "generation_date", "string", False, "Date from shared generation provenance."
    ),
    NormalizedField(
        "generation_note", "string", False, "Note from shared generation provenance."
    ),
    NormalizedField(
        "effective_reasoning_replay",
        "string",
        False,
        "Raw replay policy, or full for legacy rows where it is absent.",
    ),
    NormalizedField(
        "effective_reasoning_level",
        "string",
        False,
        "Raw reasoning level, or default where it is absent.",
    ),
    NormalizedField(
        "base_config_id",
        "string",
        False,
        "Full SHA-256 of the canonically framed base configuration identity.",
    ),
    NormalizedField(
        "arm_id",
        "string",
        False,
        "Full SHA-256 of base identity, generation, and replay provenance.",
    ),
    NormalizedField(
        "selection_status", "string", False, "One publication selection status."
    ),
    NormalizedField(
        "superseded_by_scope",
        "string",
        True,
        "Null when selected; otherwise explicit_policy or base_configuration.",
    ),
    NormalizedField(
        "superseded_by_generation",
        "int64",
        True,
        "Null when selected; otherwise the governing newer generation.",
    ),
    NormalizedField(
        "selected_for_snapshot",
        "bool",
        False,
        "True exactly when retained by the shared generation selector.",
    ),
    NormalizedField(
        "selected_for_latest",
        "bool",
        False,
        "Snapshot membership at the corpus-wide maximum generation.",
    ),
    NormalizedField(
        "comparable_to_latest",
        "bool",
        False,
        "Generation comparability; exactly equal to selected_for_latest.",
    ),
)

NORMALIZED_SCHEMA: tuple[NormalizedField, ...] = (
    _COMMON_NORMALIZED_SCHEMA + _OUTCOME_NORMALIZED_SCHEMA + _PROVENANCE_SCHEMA
)


@dataclass(frozen=True)
class SourceSpec:
    """One allowlisted source and its pinned release facts."""

    source_file: str
    release: str
    generation: int
    outcome_dialect: OutcomeDialect
    row_count: int
    sha256: str


PINNED_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "eval_results_v0.6.0.jsonl",
        "v0.6.0",
        1,
        LEGACY_DIALECT,
        131_300,
        "2e6a0135b278752cc1a1dee20f2ce20f019c515ad28642ed241960d251112e4e",
    ),
    SourceSpec(
        "eval_results_v0.7.0.jsonl",
        "v0.7.0",
        2,
        LEGACY_DIALECT,
        96_200,
        "0dbe1ee5f76c283edf07f2c7ced3c3f410733b4e5566580bf0e8b69365baeaf7",
    ),
    SourceSpec(
        "eval_results_v0.7.4.jsonl",
        "v0.7.4",
        2,
        LEGACY_DIALECT,
        31_200,
        "879ea1c11eae6cde6b9edb0ac2fdfe91013eb21cdcad7b289ea2c48d970835f5",
    ),
    SourceSpec(
        "eval_results_v0.7.5.jsonl",
        "v0.7.5",
        3,
        LEGACY_DIALECT,
        185_900,
        "906ad20c816d3248a8b8218d39db8f5a04ff81dcb037b8244573244b2d20b107",
    ),
    SourceSpec(
        "eval_results_v0.8.2.jsonl",
        "v0.8.2",
        3,
        LEGACY_DIALECT,
        74_100,
        "79cab7561117159c7e949c0f16126a5df6e20f9b408968cb9dbd27ba346842de",
    ),
    SourceSpec(
        "eval_results_v0.9.0.jsonl",
        "v0.9.0",
        3,
        CANONICAL_DIALECT,
        10_400,
        "c119d3318c0eb1b77c7f64708e649311c3a90a7f2a77c633137a5374ce19ae99",
    ),
)


class PublicationError(ValueError):
    """A compact, source-local publication-contract violation."""


class _DuplicateKey(ValueError):
    pass


class _NonStandardConstant(ValueError):
    pass


def canonical_json(value: Any, *, sort_keys: bool) -> str:
    """Serialize canonical compact JSON, preserving non-ASCII characters."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
        allow_nan=False,
    )


def canonical_json_bytes(value: Any, *, sort_keys: bool) -> bytes:
    """Return :func:`canonical_json` encoded as strict UTF-8."""
    return canonical_json(value, sort_keys=sort_keys).encode("utf-8")


def _schema_payload() -> dict[str, Any]:
    return {
        "source_dialects": {
            dialect: {
                "fields": [
                    {
                        "name": source.name,
                        "logical_type": source.logical_type,
                        "source_required": source.source_required,
                        "source_nullable": source.source_nullable,
                    }
                    for source in source_schema(dialect)
                ],
                "outcome_mapping": {
                    source_name: canonical_name
                    for source_name, canonical_name in zip(
                        ("accuracy", "completeness", "validate_error")
                        if dialect == LEGACY_DIALECT
                        else ("correct", "completed", "validation_error"),
                        ("correct", "completed", "validation_error"),
                        strict=True,
                    )
                },
            }
            for dialect in OUTCOME_DIALECTS
        },
        "normalized_fields": [
            {
                "name": normalized.name,
                "logical_type": normalized.logical_type,
                "nullable": normalized.nullable,
                "semantics": normalized.semantics,
            }
            for normalized in NORMALIZED_SCHEMA
        ],
        "enums": {
            "outcome_dialect": list(OUTCOME_DIALECTS),
            "reasoning_replay": list(REASONING_REPLAY_POLICIES),
            "selection_status": list(SELECTION_STATUSES),
            "supersession_scope": list(SUPERSESSION_SCOPES),
            "view": list(VIEW_NAMES),
        },
    }


def schema_fingerprint() -> str:
    """Return the SHA-256 fingerprint of the complete schema contract."""
    return hashlib.sha256(
        canonical_json_bytes(_schema_payload(), sort_keys=True)
    ).hexdigest()


def schema_manifest() -> dict[str, Any]:
    """Return the explicit source/normalized schema and enum vocabularies."""
    return {
        "version": SCHEMA_VERSION,
        "sha256": schema_fingerprint(),
        **_schema_payload(),
    }


def _error(source_file: str, source_line: int, invariant: str, category: str) -> None:
    raise PublicationError(
        f"{source_file}:{source_line}: {invariant} [{category}]"
    )


def _pairs_to_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise _NonStandardConstant


def _validate_string(
    value: Any, source_file: str, source_line: int, field_name: str
) -> None:
    if not isinstance(value, str):
        _error(source_file, source_line, f"field {field_name}", "wrong_type")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _error(source_file, source_line, f"field {field_name}", "invalid_unicode")


def _validate_value(
    value: Any, field_spec: SourceField, source_file: str, source_line: int
) -> None:
    if value is None:
        if not field_spec.source_nullable:
            _error(
                source_file,
                source_line,
                f"field {field_spec.name}",
                "null_not_allowed",
            )
        return

    logical_type = field_spec.logical_type
    if logical_type == "string":
        _validate_string(value, source_file, source_line, field_spec.name)
    elif logical_type == "bool":
        if type(value) is not bool:
            _error(source_file, source_line, f"field {field_spec.name}", "wrong_type")
    elif logical_type == "int64":
        if type(value) is not int:
            _error(source_file, source_line, f"field {field_spec.name}", "wrong_type")
        if not -(2**63) <= value < 2**63:
            _error(source_file, source_line, f"field {field_spec.name}", "out_of_range")
    elif logical_type == "float64":
        if type(value) not in (int, float):
            _error(source_file, source_line, f"field {field_spec.name}", "wrong_type")
        if not math.isfinite(value):
            _error(source_file, source_line, f"field {field_spec.name}", "non_finite")
    else:  # pragma: no cover - schema construction invariant
        raise AssertionError(f"unsupported logical type: {logical_type}")


def _validate_source_object(
    value: Any,
    source_file: str,
    source_line: int,
    outcome_dialect: OutcomeDialect,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _error(source_file, source_line, "source row", "non_object")

    schema = source_schema(outcome_dialect)
    fields = {field.name: field for field in schema}
    names = set(value)
    if names - fields.keys():
        _error(source_file, source_line, "source schema", "unknown_field")
    missing = {
        field.name
        for field in schema
        if field.source_required and field.name not in names
    }
    if missing:
        _error(source_file, source_line, "source schema", "missing_required_field")

    for name, field_value in value.items():
        _validate_value(field_value, fields[name], source_file, source_line)

    try:
        read_outcome(value, expected_dialect=outcome_dialect)
    except OutcomeSchemaError:
        _error(source_file, source_line, "outcome schema", "invalid_outcome")

    replay = value.get("reasoning_replay")
    if replay is not None and replay not in REASONING_REPLAY_POLICIES:
        _error(source_file, source_line, "field reasoning_replay", "unknown_policy")
    return value


def parse_source_line(
    line_bytes: bytes,
    source_file: str,
    source_line: int,
    *,
    outcome_dialect: OutcomeDialect,
) -> dict[str, Any]:
    """Strictly parse and validate one terminator-free physical source line."""
    if not line_bytes:
        _error(source_file, source_line, "source row", "blank_line")
    try:
        text = line_bytes.decode("utf-8")
    except UnicodeDecodeError:
        _error(source_file, source_line, "source row", "invalid_utf8")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_to_object,
            parse_constant=_reject_constant,
        )
    except _DuplicateKey:
        _error(source_file, source_line, "JSON object keys", "duplicate_key")
    except _NonStandardConstant:
        _error(source_file, source_line, "JSON number", "non_standard_constant")
    except (json.JSONDecodeError, RecursionError):
        _error(source_file, source_line, "source row", "malformed_json")
    return _validate_source_object(
        value, source_file, source_line, outcome_dialect
    )


@dataclass(frozen=True)
class _ParsedRow:
    source: dict[str, Any]
    source_line: int
    source_row_sha256: str


def _without_one_terminator(raw_line: bytes) -> bytes:
    if raw_line.endswith(b"\r\n"):
        return raw_line[:-2]
    if raw_line.endswith(b"\n"):
        return raw_line[:-1]
    return raw_line


def _generation_info(spec: SourceSpec) -> dict[str, str]:
    info = GEN_INFO.get(spec.generation)
    if info is None or set(info) != {"commit", "date", "note"}:
        _error(spec.source_file, 0, "generation metadata", "unknown_generation")
    return info


def _iter_validated_source(
    source_root: Path, spec: SourceSpec
) -> Iterator[_ParsedRow]:
    _generation_info(spec)
    path = source_root / spec.source_file
    try:
        source = path.open("rb")
    except OSError:
        _error(spec.source_file, 0, "source file", "unreadable")

    file_digest = hashlib.sha256()
    row_count = 0
    with source:
        for source_line, raw_line in enumerate(source, 1):
            row_count = source_line
            file_digest.update(raw_line)
            row_bytes = _without_one_terminator(raw_line)
            row = parse_source_line(
                row_bytes,
                spec.source_file,
                source_line,
                outcome_dialect=spec.outcome_dialect,
            )
            if effective_generation(row) != spec.generation:
                _error(
                    spec.source_file,
                    source_line,
                    "field gen",
                    "source_generation_mismatch",
                )
            yield _ParsedRow(
                source=row,
                source_line=source_line,
                source_row_sha256=hashlib.sha256(row_bytes).hexdigest(),
            )

    if row_count != spec.row_count:
        _error(spec.source_file, row_count, "source row count", "count_mismatch")
    if file_digest.hexdigest() != spec.sha256:
        _error(spec.source_file, row_count, "source file SHA-256", "hash_mismatch")


@dataclass(frozen=True)
class SourceFact:
    source_file: str
    release: str
    generation: int
    outcome_dialect: OutcomeDialect
    generation_reference: str
    generation_date: str
    generation_note: str
    row_count: int
    sha256: str

    @classmethod
    def from_spec(cls, spec: SourceSpec) -> SourceFact:
        info = _generation_info(spec)
        return cls(
            source_file=spec.source_file,
            release=spec.release,
            generation=spec.generation,
            outcome_dialect=spec.outcome_dialect,
            generation_reference=info["commit"],
            generation_date=info["date"],
            generation_note=info["note"],
            row_count=spec.row_count,
            sha256=spec.sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "release": self.release,
            "generation": self.generation,
            "outcome_dialect": self.outcome_dialect,
            "generation_reference": self.generation_reference,
            "generation_date": self.generation_date,
            "generation_note": self.generation_note,
            "row_count": self.row_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ViewStats:
    row_count: int
    attempted_count: int
    correct_count: int
    validated_count: int
    completed_count: int
    validation_error_count: int
    source_identity_sha256: str
    source_payload_sha256: str
    normalized_logical_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "attempted_count": self.attempted_count,
            "correct_count": self.correct_count,
            "validated_count": self.validated_count,
            "completed_count": self.completed_count,
            "validation_error_count": self.validation_error_count,
            "source_identity_sha256": self.source_identity_sha256,
            "source_payload_sha256": self.source_payload_sha256,
            "normalized_logical_sha256": self.normalized_logical_sha256,
        }


@dataclass(frozen=True)
class PublicationPlan:
    """Verified portable facts plus private state needed for row iteration."""

    sources: tuple[SourceFact, ...]
    maximum_generation: int
    base_configuration_count: int
    explicit_policy_count: int
    status_counts: Mapping[str, int]
    views: Mapping[str, ViewStats]
    _source_root: Path = field(repr=False, compare=False)
    _source_specs: tuple[SourceSpec, ...] = field(repr=False, compare=False)
    _maxima: GenerationMaxima = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical, host-independent machine-readable plan."""
        return {
            "contract_version": CONTRACT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "schema_sha256": schema_fingerprint(),
            "sources": [source.to_dict() for source in self.sources],
            "generation_maxima": {
                "corpus": self.maximum_generation,
                "base_configuration_count": self.base_configuration_count,
                "explicit_policy_count": self.explicit_policy_count,
            },
            "status_counts": {
                status: self.status_counts[status] for status in SELECTION_STATUSES
            },
            "views": {view: self.views[view].to_dict() for view in VIEW_NAMES},
        }

    def to_json(self) -> str:
        """Return byte-stable compact JSON without a trailing newline."""
        return canonical_json(self.to_dict(), sort_keys=True)


def _sha256_framed_json(value: Any, *, sort_keys: bool) -> str:
    return hashlib.sha256(canonical_json_bytes(value, sort_keys=sort_keys)).hexdigest()


def _normalize_row(
    parsed: _ParsedRow,
    spec: SourceSpec,
    maxima: GenerationMaxima,
    maximum_generation: int,
) -> dict[str, Any]:
    source = parsed.source
    generation = effective_generation(source)
    selected_for_snapshot = is_selected_generation(maxima, source)
    selected_for_latest = selected_for_snapshot and generation == maximum_generation

    if selected_for_latest:
        status = "latest"
        superseded_by_scope = None
        superseded_by_generation = None
    elif selected_for_snapshot:
        status = "carried"
        superseded_by_scope = None
        superseded_by_generation = None
    elif "reasoning_replay" in source:
        status = "superseded_same_policy"
        superseded_by_scope = "explicit_policy"
        superseded_by_generation = selection_maximum_generation(maxima, source)
    else:
        status = "superseded_base"
        superseded_by_scope = "base_configuration"
        superseded_by_generation = selection_maximum_generation(maxima, source)

    base_identity = list(base_configuration_identity(source))
    replay_provenance = (
        f"explicit:{source['reasoning_replay']}"
        if "reasoning_replay" in source
        else "legacy-inferred-full"
    )
    info = _generation_info(spec)
    outcome = read_outcome(source, expected_dialect=spec.outcome_dialect)

    normalized = {
        field.name: source.get(field.name) for field in COMMON_SOURCE_SCHEMA
    }
    normalized.update(
        {
            "correct": outcome.correct,
            "completed": outcome.completed,
            "validation_error": outcome.validation_error,
            "source_file": spec.source_file,
            "source_line": parsed.source_line,
            "source_file_sha256": spec.sha256,
            "source_row_sha256": parsed.source_row_sha256,
            "release": spec.release,
            "generation_reference": info["commit"],
            "generation_date": info["date"],
            "generation_note": info["note"],
            "effective_reasoning_replay": effective_reasoning_replay(source),
            "effective_reasoning_level": source.get("reasoning_level", "default"),
            "base_config_id": _sha256_framed_json(base_identity, sort_keys=False),
            "arm_id": _sha256_framed_json(
                base_identity + [generation, replay_provenance], sort_keys=False
            ),
            "selection_status": status,
            "superseded_by_scope": superseded_by_scope,
            "superseded_by_generation": superseded_by_generation,
            "selected_for_snapshot": selected_for_snapshot,
            "selected_for_latest": selected_for_latest,
            "comparable_to_latest": selected_for_latest,
        }
    )
    return normalized


@dataclass(frozen=True)
class _ClassifiedRecord:
    source: dict[str, Any]
    normalized: dict[str, Any]


def _iter_classified_records(
    source_root: Path,
    source_specs: Sequence[SourceSpec],
    maxima: GenerationMaxima,
    maximum_generation: int,
) -> Iterator[_ClassifiedRecord]:
    for spec in source_specs:
        for parsed in _iter_validated_source(source_root, spec):
            yield _ClassifiedRecord(
                source=parsed.source,
                normalized=_normalize_row(parsed, spec, maxima, maximum_generation),
            )


class _ViewAccumulator:
    def __init__(self) -> None:
        self.row_count = 0
        self.correct_count = 0
        self.validated_count = 0
        self.completed_count = 0
        self.validation_error_count = 0
        self.source_identity = hashlib.sha256()
        self.source_payload = hashlib.sha256()
        self.normalized_logical = hashlib.sha256()

    def update(self, source: dict[str, Any], normalized: dict[str, Any]) -> None:
        self.row_count += 1
        self.correct_count += int(normalized["correct"] is True)
        self.validated_count += int(normalized["correct"] is not None)
        self.completed_count += int(normalized["completed"])
        self.validation_error_count += int(
            normalized["validation_error"] is not None
        )
        self.source_identity.update(
            canonical_json_bytes(
                [normalized["source_file"], normalized["source_line"]],
                sort_keys=False,
            )
            + b"\n"
        )
        self.source_payload.update(
            canonical_json_bytes(source, sort_keys=True) + b"\n"
        )
        self.normalized_logical.update(
            canonical_json_bytes(normalized, sort_keys=True) + b"\n"
        )

    def finish(self) -> ViewStats:
        return ViewStats(
            row_count=self.row_count,
            attempted_count=self.row_count,
            correct_count=self.correct_count,
            validated_count=self.validated_count,
            completed_count=self.completed_count,
            validation_error_count=self.validation_error_count,
            source_identity_sha256=self.source_identity.hexdigest(),
            source_payload_sha256=self.source_payload.hexdigest(),
            normalized_logical_sha256=self.normalized_logical.hexdigest(),
        )


def _validate_source_specs(source_specs: tuple[SourceSpec, ...]) -> None:
    if not source_specs:
        raise PublicationError("manifest:0: source allowlist [empty]")
    names = [spec.source_file for spec in source_specs]
    if len(names) != len(set(names)):
        raise PublicationError("manifest:0: source allowlist [duplicate_source]")
    for spec in source_specs:
        if spec.outcome_dialect not in OUTCOME_DIALECTS:
            raise PublicationError(
                "manifest:0: source outcome dialect [unknown_dialect]"
            )
        _generation_info(spec)


def _enforce_pinned_plan(views: Mapping[str, ViewStats]) -> None:
    for view, expected_count in PINNED_VIEW_COUNTS.items():
        if views[view].row_count != expected_count:
            raise PublicationError(f"manifest:0: {view} row count [contract_mismatch]")
        expected_correct, expected_validated, expected_completed = (
            PINNED_VIEW_METRIC_COUNTS[view]
        )
        stats = views[view]
        if (
            stats.attempted_count != expected_count
            or stats.correct_count != expected_correct
            or stats.validated_count != expected_validated
            or stats.completed_count != expected_completed
            or stats.validation_error_count != 0
        ):
            raise PublicationError(
                f"manifest:0: {view} metric counts [contract_mismatch]"
            )
    snapshot = views["snapshot"]
    if (
        snapshot.source_identity_sha256
        != PINNED_SNAPSHOT_SOURCE_IDENTITY_SHA256
    ):
        raise PublicationError(
            "manifest:0: snapshot source identity digest [contract_mismatch]"
        )
    if (
        snapshot.source_payload_sha256
        != PINNED_SNAPSHOT_SOURCE_PAYLOAD_SHA256
    ):
        raise PublicationError(
            "manifest:0: snapshot source payload digest [contract_mismatch]"
        )


def build_publication_plan(
    source_root: str | Path,
    *,
    source_specs: Sequence[SourceSpec] = PINNED_SOURCES,
) -> PublicationPlan:
    """Verify, classify, and digest allowlisted sources in bounded memory."""
    root = Path(source_root)
    specs = tuple(source_specs)
    _validate_source_specs(specs)

    maxima = GenerationMaxima()
    maximum_generation = -1
    for spec in specs:
        for parsed in _iter_validated_source(root, spec):
            accumulate_generation_maxima(maxima, parsed.source)
            maximum_generation = max(
                maximum_generation, effective_generation(parsed.source)
            )

    accumulators = {view: _ViewAccumulator() for view in VIEW_NAMES}
    status_counts = {status: 0 for status in SELECTION_STATUSES}
    for record in _iter_classified_records(root, specs, maxima, maximum_generation):
        normalized = record.normalized
        status_counts[normalized["selection_status"]] += 1
        accumulators["history"].update(record.source, normalized)
        if normalized["selected_for_snapshot"]:
            accumulators["snapshot"].update(record.source, normalized)
        if normalized["selected_for_latest"]:
            accumulators["latest"].update(record.source, normalized)

    views = {view: accumulators[view].finish() for view in VIEW_NAMES}
    if specs == PINNED_SOURCES:
        _enforce_pinned_plan(views)

    return PublicationPlan(
        sources=tuple(SourceFact.from_spec(spec) for spec in specs),
        maximum_generation=maximum_generation,
        base_configuration_count=len(maxima.base),
        explicit_policy_count=len(maxima.explicit_policy),
        status_counts=status_counts,
        views=views,
        _source_root=root,
        _source_specs=specs,
        _maxima=maxima,
    )


def iter_classified_history(plan: PublicationPlan) -> Iterator[dict[str, Any]]:
    """Yield every normalized row in source order.

    Exhaust the iterator to complete its strict source hash/count verification.
    The iterator retains only one source and normalized row at a time.
    """
    for record in _iter_classified_records(
        plan._source_root,
        plan._source_specs,
        plan._maxima,
        plan.maximum_generation,
    ):
        yield record.normalized


def row_in_view(row: Mapping[str, Any], view: str) -> bool:
    """Return whether one classified history row belongs to a named view."""
    if view == "history":
        return True
    if view == "snapshot":
        return bool(row["selected_for_snapshot"])
    if view == "latest":
        return bool(row["selected_for_latest"])
    raise ValueError(f"unknown publication view: {view}")


def iter_view(plan: PublicationPlan, view: str) -> Iterator[dict[str, Any]]:
    """Project one view from the single classified-history stream."""
    if view not in VIEW_NAMES:
        raise ValueError(f"unknown publication view: {view}")
    for row in iter_classified_history(plan):
        if row_in_view(row, view):
            yield row


def iter_history(plan: PublicationPlan) -> Iterator[dict[str, Any]]:
    return iter_view(plan, "history")


def iter_snapshot(plan: PublicationPlan) -> Iterator[dict[str, Any]]:
    return iter_view(plan, "snapshot")


def iter_latest(plan: PublicationPlan) -> Iterator[dict[str, Any]]:
    return iter_view(plan, "latest")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and print the Forge eval publication plan."
    )
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Directory containing the six pinned eval JSONL files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        plan = build_publication_plan(args.source_root)
    except PublicationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.write(plan.to_json() + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a command
    raise SystemExit(main())
