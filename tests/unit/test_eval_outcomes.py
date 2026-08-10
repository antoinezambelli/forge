"""Canonical eval outcome semantics and legacy compatibility."""

from __future__ import annotations

import pytest

from tests.eval.eval_runner import _evaluate_validators
from tests.eval.outcomes import (
    CANONICAL_DIALECT,
    LEGACY_DIALECT,
    OutcomeSchemaError,
    RunOutcome,
    count_outcomes,
    detect_outcome_dialect,
    read_outcome,
    write_outcome,
)


@pytest.mark.parametrize(
    ("row", "dialect", "expected"),
    [
        (
            {"accuracy": True, "completeness": False},
            LEGACY_DIALECT,
            RunOutcome(correct=True, completed=False),
        ),
        (
            {"correct": False, "completed": True},
            CANONICAL_DIALECT,
            RunOutcome(correct=False, completed=True),
        ),
        (
            {
                "accuracy": False,
                "completeness": True,
                "validate_error": "ValueError",
            },
            LEGACY_DIALECT,
            RunOutcome(
                correct=None,
                completed=True,
                validation_error="ValueError",
            ),
        ),
    ],
)
def test_read_outcome_normalizes_both_dialects(
    row: dict, dialect: str, expected: RunOutcome
) -> None:
    assert detect_outcome_dialect(row) == dialect
    assert read_outcome(row) == expected


@pytest.mark.parametrize(
    "row",
    [
        {"correct": True},
        {"accuracy": True, "completed": True},
        {
            "accuracy": True,
            "completeness": True,
            "correct": True,
            "completed": True,
        },
        {
            "correct": True,
            "completed": True,
            "validate_error": "wrong dialect",
        },
    ],
)
def test_partial_hybrid_and_dual_outcomes_fail_closed(row: dict) -> None:
    with pytest.raises(OutcomeSchemaError):
        read_outcome(row)


def test_outcome_truth_table_keeps_denominators_independent() -> None:
    counts = count_outcomes(
        [
            RunOutcome(correct=True, completed=False),
            RunOutcome(correct=False, completed=True),
            RunOutcome(correct=None, completed=True),
            RunOutcome(
                correct=None,
                completed=False,
                validation_error="validator failed",
            ),
        ]
    )

    assert counts.to_dict() == {
        "attempted_count": 4,
        "correct_count": 1,
        "validated_count": 2,
        "completed_count": 2,
        "validation_error_count": 1,
    }
    assert counts.score == 1 / 4
    assert counts.validated_accuracy == 1 / 2
    assert counts.completion_rate == 1 / 2


def test_outcome_writer_uses_only_selected_dialect() -> None:
    outcome = RunOutcome(
        correct=None, completed=True, validation_error="TypeError"
    )
    assert write_outcome(outcome) == {
        "correct": None,
        "completed": True,
        "validation_error": "TypeError",
    }
    assert write_outcome(outcome, LEGACY_DIALECT) == {
        "accuracy": None,
        "completeness": True,
        "validate_error": "TypeError",
    }


def test_outcome_writer_nulls_a_verdict_tainted_by_validation_error() -> None:
    assert write_outcome(
        RunOutcome(
            correct=True,
            completed=True,
            validation_error="validator failed",
        )
    )["correct"] is None


def test_validator_failures_always_produce_a_null_verdict() -> None:
    state_calls = 0

    def output_failure(_args: dict) -> bool:
        raise ValueError("bad output")

    def state_success() -> bool:
        nonlocal state_calls
        state_calls += 1
        return True

    assert _evaluate_validators(output_failure, {}, state_success) == (
        None,
        "ValueError",
    )
    assert state_calls == 0

    def state_failure() -> bool:
        raise RuntimeError("bad state")

    assert _evaluate_validators(lambda _args: True, {}, state_failure) == (
        None,
        "validate_state: RuntimeError",
    )


@pytest.mark.parametrize(
    ("output", "state", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_normal_validator_truth_table(
    output: bool, state: bool, expected: bool
) -> None:
    assert _evaluate_validators(
        lambda _args: output, {}, lambda: state
    ) == (expected, None)
