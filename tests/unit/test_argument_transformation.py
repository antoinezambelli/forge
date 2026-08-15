"""Behavioral contracts for the argument-transformation benchmark."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tests.eval.scenarios._model_reasoning import (
    _argument_transformation_tools,
    _validate_argument_transformation,
)
from tests.eval.scenarios._stateful_model_reasoning import (
    ExpenseAuditSystem,
    _build_argument_transformation_stateful,
    _validate_argument_transformation_stateful,
)


VALIDATORS: tuple[Callable[[dict[str, Any]], bool], ...] = (
    _validate_argument_transformation,
    _validate_argument_transformation_stateful,
)


def _call_tool(implementation: str, tool_name: str, kwargs: dict[str, Any]) -> str:
    if implementation == "lambda":
        return _argument_transformation_tools[tool_name].callable(**kwargs)
    return getattr(ExpenseAuditSystem(), tool_name)(**kwargs)


def _canonical_args() -> dict[str, str]:
    return {
        "transaction_ids": "TX-1001, TX-1005, TX-1006, TX-1008",
        "total_flagged_usd": "$28,980.00",
        "top_vendor": "Wonka Industries",
    }


def test_lambda_and_stateful_tools_share_output_contracts() -> None:
    transaction_signals = tuple(f"TX-10{i:02d}" for i in range(1, 14))
    cases = [
        (
            "Q4 transactions",
            "list_transactions",
            {"quarter": "Q4", "year": 2024},
            transaction_signals
            + (
                "EUR",
                "USD",
                "5,000.00 USD",
                "ACME Corp",
                "Wonka Industries",
            ),
            (),
        ),
        (
            "unknown quarter",
            "list_transactions",
            {"quarter": "Q3", "year": 2024},
            ("No transactions found",),
            ("TX-1001",),
        ),
        (
            "approved vendors",
            "get_approved_vendors",
            {},
            (
                "Acme Corp",
                "Globex Industries",
                "Initech Systems",
                "Umbrella Logistics",
                "Wayne Enterprises",
                "Stark Industries",
            ),
            ("ACME Corp",),
        ),
        (
            "alias resolution",
            "get_vendor_details",
            {"vendor_name": "ACME Corp"},
            ("alias of Acme Corp", "unified entity"),
            (),
        ),
        (
            "master vendor",
            "get_vendor_details",
            {"vendor_name": "Acme Corp"},
            ("master account", "ACME Corp"),
            (),
        ),
        (
            "unknown vendor",
            "get_vendor_details",
            {"vendor_name": "Cyberdyne LLC"},
            ("not found in vendor master",),
            (),
        ),
        (
            "flagged EUR conversion",
            "currency_convert",
            {
                "amount": 4800,
                "from_currency": "EUR",
                "to_currency": "USD",
            },
            ("5,280.00 USD", "1 EUR = 1.1 USD"),
            (),
        ),
        (
            "under-threshold EUR conversion",
            "currency_convert",
            {
                "amount": 2400,
                "from_currency": "EUR",
                "to_currency": "USD",
            },
            ("2,640.00 USD",),
            (),
        ),
        (
            "unsupported conversion",
            "currency_convert",
            {
                "amount": 100,
                "from_currency": "GBP",
                "to_currency": "JPY",
            },
            ("Unsupported",),
            (),
        ),
        (
            "categorization distractor",
            "categorize_expense",
            {"amount": 7500, "category": "contractor"},
            ("GL-",),
            ("TX-1001", "approved"),
        ),
        (
            "transaction lookup",
            "lookup_transaction",
            {"transaction_id": "TX-1006"},
            ("Pied Piper", "4,800.00 EUR"),
            (),
        ),
        (
            "unknown transaction",
            "lookup_transaction",
            {"transaction_id": "TX-9999"},
            ("No transaction found",),
            (),
        ),
        (
            "terminal acknowledgement",
            "submit_audit_report",
            {
                "transaction_ids": "TX-1001, TX-1005, TX-1006, TX-1008",
                "total_flagged_usd": "$28,980.00",
                "top_vendor": "Wonka Industries",
            },
            ("TX-1001", "Wonka Industries"),
            (),
        ),
    ]

    for implementation in ("lambda", "stateful"):
        for label, tool_name, kwargs, required, forbidden in cases:
            output = _call_tool(implementation, tool_name, kwargs)
            assert all(signal in output for signal in required), (
                implementation,
                label,
                output,
            )
            assert all(signal not in output for signal in forbidden), (
                implementation,
                label,
                output,
            )


def test_lambda_and_stateful_validators_accept_equivalent_formats() -> None:
    alternatives = [
        _canonical_args(),
        {
            "transaction_ids": "TX-1001 TX-1005 TX-1006 TX-1008",
            "total_flagged_usd": "28980 USD",
            "top_vendor": "wonka industries",
        },
    ]

    for validator in VALIDATORS:
        for args in alternatives:
            assert validator(args), (validator.__name__, args)


def test_lambda_and_stateful_validators_reject_benchmark_failure_modes() -> None:
    cases = [
        (
            "skipped EUR conversion",
            {
                "transaction_ids": "TX-1001, TX-1005, TX-1008",
                "total_flagged_usd": "$23,700",
            },
        ),
        (
            "ACME alias over-flag",
            {
                "transaction_ids": ("TX-1001, TX-1005, TX-1006, TX-1008, TX-1009"),
                "total_flagged_usd": "$35,480",
            },
        ),
        (
            "strict greater-than threshold",
            {
                "transaction_ids": "TX-1001, TX-1006, TX-1008",
                "total_flagged_usd": "$23,980",
            },
        ),
        ("wrong top vendor", {"top_vendor": "Cyberdyne LLC"}),
        (
            "missing transaction",
            {"transaction_ids": "TX-1001, TX-1005, TX-1008"},
        ),
        ("missing total", {"total_flagged_usd": ""}),
    ]

    for validator in VALIDATORS:
        for label, updates in cases:
            args = _canonical_args()
            args.update(updates)
            assert not validator(args), (validator.__name__, label)


def test_stateful_tools_record_successful_reasoning_path() -> None:
    system = ExpenseAuditSystem()

    assert "TX-1001" in system.list_transactions("Q4", 2024)
    assert "Acme Corp" in system.get_approved_vendors()
    assert "alias of Acme Corp" in system.get_vendor_details("ACME Corp")
    assert "5,280.00 USD" in system.currency_convert(4800, "EUR", "USD")
    system.submit_audit_report(
        transaction_ids="TX-1001",
        total_flagged_usd="$1.00",
        top_vendor="Anyone",
    )

    assert system.list_called_for == ("Q4", 2024)
    assert system.approved_called is True
    assert system.vendor_details_called_for == {"ACME Corp"}
    assert system.eur_conversion_called is True
    assert system.submitted_args == {
        "transaction_ids": "TX-1001",
        "total_flagged_usd": "$1.00",
        "top_vendor": "Anyone",
    }


def test_stateful_tools_do_not_record_rejected_paths() -> None:
    system = ExpenseAuditSystem()

    assert "No transactions found" in system.list_transactions("Q3", 2024)
    system.currency_convert(1000, "USD", "EUR")

    assert system.list_called_for is None
    assert system.eur_conversion_called is False


def _exercise_required_stateful_calls(workflow: Any) -> None:
    workflow.tools["list_transactions"].callable(quarter="Q4", year=2024)
    workflow.tools["get_approved_vendors"].callable()
    workflow.tools["currency_convert"].callable(
        amount=4800,
        from_currency="EUR",
        to_currency="USD",
    )
    workflow.tools["get_vendor_details"].callable(vendor_name="ACME Corp")


def test_validate_state_requires_full_reasoning_path_and_submission() -> None:
    workflow, validate_state = _build_argument_transformation_stateful()

    assert not validate_state()
    workflow.tools["list_transactions"].callable(quarter="Q4", year=2024)
    workflow.tools["get_approved_vendors"].callable()
    assert not validate_state()
    workflow.tools["currency_convert"].callable(
        amount=4800,
        from_currency="EUR",
        to_currency="USD",
    )
    assert not validate_state()
    workflow.tools["get_vendor_details"].callable(vendor_name="ACME Corp")
    assert not validate_state()
    workflow.tools["submit_audit_report"].callable(**_canonical_args())
    assert validate_state()


def test_validate_state_rejects_skipped_conversion() -> None:
    workflow, validate_state = _build_argument_transformation_stateful()

    workflow.tools["list_transactions"].callable(quarter="Q4", year=2024)
    workflow.tools["get_approved_vendors"].callable()
    workflow.tools["get_vendor_details"].callable(vendor_name="ACME Corp")
    workflow.tools["submit_audit_report"].callable(**_canonical_args())

    assert not validate_state()


def test_validate_state_rejects_wrong_submission() -> None:
    workflow, validate_state = _build_argument_transformation_stateful()

    _exercise_required_stateful_calls(workflow)
    args = _canonical_args()
    args["total_flagged_usd"] = "$99,999"
    workflow.tools["submit_audit_report"].callable(**args)

    assert not validate_state()
