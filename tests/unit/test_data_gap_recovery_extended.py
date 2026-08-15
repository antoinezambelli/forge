"""Behavioral contracts for the extended data-gap recovery benchmark."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tests.eval.scenarios._model_reasoning import (
    _data_gap_recovery_extended_tools,
    _validate_data_gap_recovery_extended,
)
from tests.eval.scenarios._stateful_model_reasoning import (
    HRRecordsSystemExtended,
    _build_data_gap_recovery_extended_stateful,
    _validate_data_gap_recovery_extended_stateful,
)


VALIDATORS: tuple[Callable[[dict[str, Any]], bool], ...] = (
    _validate_data_gap_recovery_extended,
    _validate_data_gap_recovery_extended_stateful,
)


def _call_tool(implementation: str, tool_name: str, kwargs: dict[str, Any]) -> str:
    if implementation == "lambda":
        return _data_gap_recovery_extended_tools[tool_name].callable(**kwargs)
    return getattr(HRRecordsSystemExtended(), tool_name)(**kwargs)


def _canonical_args() -> dict[str, str]:
    return {
        "employee_name": "Sarah Chen",
        "report": (
            "Sarah Chen — Engineering, Senior Backend Engineer. "
            "Clearance: L3 Confidential. "
            "Emergency contact: Michael Chen (spouse), (555) 867-5309. "
            "Salary band: B7. "
            "Active access groups: payments-prod-v3, internal-apis-v2."
        ),
    }


def test_lambda_and_stateful_tools_share_output_contracts() -> None:
    cases = [
        (
            "employee record",
            "get_employee",
            {"name": "Sarah Chen"},
            (
                "E-1847",
                "Engineering",
                "[RESTRICTED]",
                "[unavailable in this record]",
                "[PROTECTED]",
                "[managed externally]",
            ),
            (
                "security_audit",
                "personnel_file",
                "onboarding_records",
                "project_assignments",
            ),
        ),
        (
            "security soft trap",
            "security_audit",
            {"employee_id": "E-1847"},
            ("L3", "Confidential", "payments-prod", "internal-apis"),
            ("payments-prod-v3", "internal-apis-v2", "project_assignments"),
        ),
        (
            "emergency contact",
            "onboarding_records",
            {"employee_id": "E-1847"},
            ("Michael Chen", "867-5309"),
            (),
        ),
        (
            "domain-language redirect",
            "personnel_file",
            {"employee_id": "E-1847"},
            ("current pay system", "HR onboarding"),
            ("compensation_v2", "onboarding_records"),
        ),
        (
            "current compensation",
            "compensation_v2",
            {"employee_id": "E-1847"},
            ("B7",),
            (),
        ),
        (
            "current groups",
            "project_assignments",
            {"employee_id": "E-1847"},
            ("payments-prod-v3", "internal-apis-v2"),
            (),
        ),
        (
            "deprecated compensation",
            "compensation_history",
            {"employee_id": "E-1847"},
            ("DEPRECATED", "compensation_v2"),
            ("B7",),
        ),
        (
            "deprecated access",
            "access_control",
            {"employee_id": "E-1847"},
            ("DEPRECATED", "project_assignments"),
            ("payments-prod-v3",),
        ),
        (
            "legacy lure",
            "legacy_records",
            {"employee_id": "E-1847"},
            (
                "B5",
                "legacy-readonly",
                "legacy_v1",
                "compensation_v2",
                "project_assignments",
            ),
            ("B7", "payments-prod-v3"),
        ),
        (
            "compliance distractor",
            "compliance_check",
            {"employee_id": "E-1847"},
            (),
            ("B7", "payments-prod-v3", "Michael Chen"),
        ),
        (
            "directory distractor",
            "hr_directory",
            {"name": "Sarah Chen"},
            (),
            ("B7", "payments-prod-v3", "Michael Chen"),
        ),
        (
            "terminal acknowledgement",
            "submit_report",
            {"employee_name": "Sarah Chen", "report": "some report"},
            ("Sarah Chen",),
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


def test_lambda_and_stateful_validators_accept_current_facts() -> None:
    for validator in VALIDATORS:
        assert validator(_canonical_args())
        for current_group in ("internal-apis-v2", "payments-prod-v3"):
            args = _canonical_args()
            args["report"] = args["report"].replace(
                "payments-prod-v3, internal-apis-v2", current_group
            )
            assert validator(args), (validator.__name__, current_group)


def test_lambda_and_stateful_validators_reject_missing_or_lure_facts() -> None:
    cases = [
        ("missing department", (("Engineering", ""),)),
        ("missing clearance", (("L3", ""), ("Confidential", ""))),
        ("missing contact", (("Michael", ""),)),
        ("missing salary", (("B7", ""),)),
        (
            "missing groups",
            (("payments-prod-v3", ""), ("internal-apis-v2", "")),
        ),
        ("legacy salary lure", (("B7", "B5"),)),
        (
            "legacy group lure",
            (
                ("payments-prod-v3", "legacy-readonly"),
                (", internal-apis-v2", ""),
            ),
        ),
        (
            "static security groups",
            (
                ("payments-prod-v3", "payments-prod"),
                (", internal-apis-v2", ", internal-apis"),
            ),
        ),
    ]

    for validator in VALIDATORS:
        for label, replacements in cases:
            args = _canonical_args()
            for old, new in replacements:
                args["report"] = args["report"].replace(old, new)
            assert not validator(args), (validator.__name__, label)


def test_stateful_tools_record_only_successful_extended_fetches() -> None:
    records = HRRecordsSystemExtended()

    assert records.compensation_v2_fetched is None
    assert records.project_assignments_fetched is None
    assert "B7" in records.compensation_v2("E-1847")
    assert "payments-prod-v3" in records.project_assignments("E-1847")
    assert records.compensation_v2_fetched == "E-1847"
    assert records.project_assignments_fetched == "E-1847"

    rejected = HRRecordsSystemExtended()
    assert "No compensation_v2 record" in rejected.compensation_v2("E-9999")
    assert rejected.compensation_v2_fetched is None


def test_validate_state_requires_every_current_data_source() -> None:
    workflow, validate_state = _build_data_gap_recovery_extended_stateful()

    assert not validate_state()
    workflow.tools["get_employee"].callable(name="Sarah Chen")
    assert not validate_state()
    workflow.tools["security_audit"].callable(employee_id="E-1847")
    workflow.tools["onboarding_records"].callable(employee_id="E-1847")
    workflow.tools["compensation_v2"].callable(employee_id="E-1847")
    assert not validate_state()
    workflow.tools["project_assignments"].callable(employee_id="E-1847")
    assert validate_state()


def test_validate_state_rejects_legacy_only_path() -> None:
    workflow, validate_state = _build_data_gap_recovery_extended_stateful()

    workflow.tools["get_employee"].callable(name="Sarah Chen")
    workflow.tools["legacy_records"].callable(employee_id="E-1847")

    assert not validate_state()
