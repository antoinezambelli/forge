"""Canonical eval outcome vocabulary and legacy-row compatibility.

One stored row is an attempted run.  New rows use ``correct`` / ``completed`` /
``validation_error``.  Released rows remain immutable and use the legacy
``accuracy`` / ``completeness`` / ``validate_error`` dialect.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal


OutcomeDialect = Literal["legacy", "canonical"]
LEGACY_DIALECT: OutcomeDialect = "legacy"
CANONICAL_DIALECT: OutcomeDialect = "canonical"
OUTCOME_DIALECTS: tuple[OutcomeDialect, ...] = (
    LEGACY_DIALECT,
    CANONICAL_DIALECT,
)

_DIALECT_FIELDS: dict[OutcomeDialect, tuple[str, str, str]] = {
    LEGACY_DIALECT: ("accuracy", "completeness", "validate_error"),
    CANONICAL_DIALECT: ("correct", "completed", "validation_error"),
}
_ALL_OUTCOME_FIELDS = frozenset(
    field for fields in _DIALECT_FIELDS.values() for field in fields
)


class OutcomeSchemaError(ValueError):
    """A row has an ambiguous or incomplete outcome shape."""


@dataclass(frozen=True)
class RunOutcome:
    """Canonical outcome for one attempted run."""

    correct: bool | None
    completed: bool
    validation_error: str | None = None

    @property
    def validated(self) -> bool:
        return self.correct is not None and self.validation_error is None

    @property
    def is_correct(self) -> bool:
        return self.validated and self.correct is True


@dataclass(frozen=True)
class OutcomeCounts:
    """Exact integer components for the three public aggregate metrics."""

    attempted_count: int = 0
    correct_count: int = 0
    validated_count: int = 0
    completed_count: int = 0
    validation_error_count: int = 0

    @property
    def score(self) -> float:
        return (
            self.correct_count / self.attempted_count
            if self.attempted_count
            else 0.0
        )

    @property
    def validated_accuracy(self) -> float | None:
        return (
            self.correct_count / self.validated_count
            if self.validated_count
            else None
        )

    @property
    def completion_rate(self) -> float:
        return (
            self.completed_count / self.attempted_count
            if self.attempted_count
            else 0.0
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "attempted_count": self.attempted_count,
            "correct_count": self.correct_count,
            "validated_count": self.validated_count,
            "completed_count": self.completed_count,
            "validation_error_count": self.validation_error_count,
        }


def outcome_fields(dialect: OutcomeDialect) -> tuple[str, str, str]:
    """Return ``(correct, completed, validation_error)`` source keys."""
    try:
        return _DIALECT_FIELDS[dialect]
    except KeyError as exc:  # pragma: no cover - guarded by typed callers
        raise OutcomeSchemaError(f"unknown outcome dialect: {dialect!r}") from exc


def detect_outcome_dialect(row: Mapping[str, Any]) -> OutcomeDialect:
    """Detect one exact outcome dialect, rejecting partial or mixed aliases."""
    present = set(row) & _ALL_OUTCOME_FIELDS
    legacy_correct, legacy_completed, legacy_error = _DIALECT_FIELDS[LEGACY_DIALECT]
    current_correct, current_completed, current_error = _DIALECT_FIELDS[CANONICAL_DIALECT]

    has_legacy_primary = {legacy_correct, legacy_completed} <= present
    has_current_primary = {current_correct, current_completed} <= present
    has_any_legacy = bool(present & {legacy_correct, legacy_completed, legacy_error})
    has_any_current = bool(present & {current_correct, current_completed, current_error})

    if has_legacy_primary and not has_any_current:
        return LEGACY_DIALECT
    if has_current_primary and not has_any_legacy:
        return CANONICAL_DIALECT
    raise OutcomeSchemaError("outcome fields must use one complete dialect")


def read_outcome(
    row: Mapping[str, Any],
    *,
    expected_dialect: OutcomeDialect | None = None,
) -> RunOutcome:
    """Read and type-check a row outcome into the canonical vocabulary."""
    dialect = detect_outcome_dialect(row)
    if expected_dialect is not None and dialect != expected_dialect:
        raise OutcomeSchemaError(
            f"outcome dialect {dialect!r} does not match expected {expected_dialect!r}"
        )
    correct_key, completed_key, error_key = outcome_fields(dialect)
    correct = row[correct_key]
    completed = row[completed_key]
    validation_error = row.get(error_key)
    if correct is not None and type(correct) is not bool:
        raise OutcomeSchemaError(f"{correct_key} must be bool or null")
    if type(completed) is not bool:
        raise OutcomeSchemaError(f"{completed_key} must be bool")
    if validation_error is not None and not isinstance(validation_error, str):
        raise OutcomeSchemaError(f"{error_key} must be string or null")
    # A validator exception means there is no usable correctness judgment,
    # including legacy state-validator rows that stored ``accuracy: false``.
    if validation_error is not None:
        correct = None
    return RunOutcome(
        correct=correct,
        completed=completed,
        validation_error=validation_error,
    )


def write_outcome(
    outcome: RunOutcome,
    dialect: OutcomeDialect = CANONICAL_DIALECT,
) -> dict[str, Any]:
    """Serialize one canonical outcome in an explicitly selected dialect."""
    correct_key, completed_key, error_key = outcome_fields(dialect)
    correct = None if outcome.validation_error is not None else outcome.correct
    row: dict[str, Any] = {
        correct_key: correct,
        completed_key: outcome.completed,
    }
    if outcome.validation_error is not None:
        row[error_key] = outcome.validation_error
    return row


def count_outcomes(outcomes: Iterable[RunOutcome]) -> OutcomeCounts:
    attempted = correct = validated = completed = validation_errors = 0
    for outcome in outcomes:
        attempted += 1
        completed += int(outcome.completed)
        validation_errors += int(outcome.validation_error is not None)
        if outcome.validated:
            validated += 1
            correct += int(outcome.correct is True)
    return OutcomeCounts(
        attempted_count=attempted,
        correct_count=correct,
        validated_count=validated,
        completed_count=completed,
        validation_error_count=validation_errors,
    )


def count_rows(rows: Iterable[Mapping[str, Any]]) -> OutcomeCounts:
    return count_outcomes(read_outcome(row) for row in rows)


def is_correct(row: Mapping[str, Any]) -> bool:
    return read_outcome(row).is_correct


def is_completed(row: Mapping[str, Any]) -> bool:
    return read_outcome(row).completed


def is_validated(row: Mapping[str, Any]) -> bool:
    return read_outcome(row).validated
