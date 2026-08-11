"""Tests for the data_gap_recovery_extended scenario.

Verifies:
- Each tool callable returns content with the expected unique markers
  (so the indirect-validation strategy actually works)
- The validator accepts a canonical-correct terminal call
- The validator rejects each failure mode (lure-trapped, missing fact,
  wrong groups, deprecated tool used)
- The stateful variant's HRRecordsSystemExtended tracks calls correctly
"""

from __future__ import annotations

from tests.eval.scenarios._model_reasoning import (
    _data_gap_recovery_extended_tools,
    _validate_data_gap_recovery_extended,
)
from tests.eval.scenarios._stateful_model_reasoning import (
    HRRecordsSystemExtended,
    _build_data_gap_recovery_extended_stateful,
    _validate_data_gap_recovery_extended_stateful,
)


# ── Lambda tool callables ───────────────────────────────────────


class TestLambdaTools:
    """Each tool returns the expected unique marker so the validator can detect
    whether the right downstream tool was reached."""

    def _call(self, tool_name: str, **kwargs: object) -> str:
        tool = _data_gap_recovery_extended_tools[tool_name]
        return tool.callable(**kwargs)

    def test_get_employee_returns_base_record_with_redacted_fields(self) -> None:
        out = self._call("get_employee", name="Sarah Chen")
        assert "E-1847" in out
        assert "Engineering" in out
        # All four fields are redacted with NO tool-name hints —
        # model must discover downstream tools by name.
        assert "[RESTRICTED]" in out
        assert "[unavailable in this record]" in out
        assert "[PROTECTED]" in out
        assert "[managed externally]" in out
        # No tool names should leak into get_employee output
        assert "security_audit" not in out
        assert "personnel_file" not in out
        assert "onboarding_records" not in out
        assert "project_assignments" not in out

    def test_security_audit_has_clearance_and_soft_trap_groups(self) -> None:
        out = self._call("security_audit", employee_id="E-1847")
        assert "L3" in out
        assert "Confidential" in out
        # Soft trap: provisioned groups look like a complete answer but
        # are missing the v3/v2 suffixes the validator wants.
        assert "payments-prod" in out
        assert "payments-prod-v3" not in out
        assert "internal-apis" in out
        assert "internal-apis-v2" not in out
        # No redirect hint to project_assignments — model must discover it
        assert "project_assignments" not in out

    def test_onboarding_has_emergency_contact(self) -> None:
        out = self._call("onboarding_records", employee_id="E-1847")
        assert "Michael Chen" in out
        assert "867-5309" in out

    def test_personnel_file_uses_domain_language_not_tool_names(self) -> None:
        out = self._call("personnel_file", employee_id="E-1847")
        # Domain language only — no explicit tool-name hints. Model must
        # infer compensation_v2 (vs deprecated compensation_history) and
        # discover onboarding_records by name.
        assert "current pay system" in out
        assert "HR onboarding" in out
        assert "compensation_v2" not in out
        assert "onboarding_records" not in out

    def test_compensation_v2_returns_b7_uniquely(self) -> None:
        out = self._call("compensation_v2", employee_id="E-1847")
        assert "B7" in out

    def test_project_assignments_returns_v3_groups_uniquely(self) -> None:
        out = self._call("project_assignments", employee_id="E-1847")
        assert "payments-prod-v3" in out
        assert "internal-apis-v2" in out

    def test_compensation_history_is_deprecated_redirect(self) -> None:
        out = self._call("compensation_history", employee_id="E-1847")
        assert "DEPRECATED" in out
        assert "compensation_v2" in out
        # Must NOT leak B7 — that's the whole point of this trap
        assert "B7" not in out

    def test_access_control_is_deprecated_redirect(self) -> None:
        out = self._call("access_control", employee_id="E-1847")
        assert "DEPRECATED" in out
        assert "project_assignments" in out
        assert "payments-prod-v3" not in out

    def test_legacy_records_lure_has_warning_and_stale_data(self) -> None:
        out = self._call("legacy_records", employee_id="E-1847")
        # Plausible-but-wrong data is present
        assert "B5" in out
        assert "legacy-readonly" in out
        # The warning / redirect note is also present (model must read it)
        assert "legacy_v1" in out
        assert "compensation_v2" in out
        assert "project_assignments" in out
        # Lure must not contain the right answers
        assert "B7" not in out
        assert "payments-prod-v3" not in out

    def test_distractors_have_no_required_facts(self) -> None:
        compliance = self._call("compliance_check", employee_id="E-1847")
        directory = self._call("hr_directory", name="Sarah Chen")
        for out in (compliance, directory):
            assert "B7" not in out
            assert "payments-prod-v3" not in out
            assert "Michael Chen" not in out

    def test_submit_report_acks_the_employee(self) -> None:
        out = self._call(
            "submit_report",
            employee_name="Sarah Chen",
            report="some report",
        )
        assert "Sarah Chen" in out


# ── Validator ───────────────────────────────────────────────────


class TestValidator:
    """All five facts must be present, and the unique markers must come from
    the right downstream tools (B7 not B5, v3/v2 groups not legacy-readonly)."""

    def _canonical_args(self) -> dict[str, str]:
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

    def test_canonical_args_pass(self) -> None:
        assert _validate_data_gap_recovery_extended(self._canonical_args())

    def test_missing_and_lure_trapped_facts_are_rejected(self) -> None:
        cases = (
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
                "static security-audit groups",
                (
                    ("payments-prod-v3", "payments-prod"),
                    (", internal-apis-v2", ", internal-apis"),
                ),
            ),
        )

        for case, replacements in cases:
            args = self._canonical_args()
            for old, new in replacements:
                args["report"] = args["report"].replace(old, new)
            assert not _validate_data_gap_recovery_extended(args), case

    def test_either_unique_current_group_marker_is_sufficient(self) -> None:
        for marker in ("internal-apis-v2", "payments-prod-v3"):
            args = self._canonical_args()
            args["report"] = args["report"].replace(
                "payments-prod-v3, internal-apis-v2", marker,
            )
            assert _validate_data_gap_recovery_extended(args), marker


# ── Stateful backend ────────────────────────────────────────────


class TestHRRecordsSystemExtended:
    """The subclass adds compensation_v2, project_assignments, and the
    deprecated/legacy endpoints. Existing methods stay functional."""

    def test_inherits_existing_methods(self) -> None:
        db = HRRecordsSystemExtended()
        # Inherited from HRRecordsSystem (with override)
        out = db.security_audit("E-1847")
        assert "L3" in out
        # Override drops the project_assignments redirect — soft trap:
        # provisioned groups look like a complete answer.
        assert "project_assignments" not in out
        assert "payments-prod" in out
        assert "payments-prod-v3" not in out

    def test_compensation_v2_returns_b7(self) -> None:
        db = HRRecordsSystemExtended()
        out = db.compensation_v2("E-1847")
        assert "B7" in out
        assert db.compensation_v2_fetched == "E-1847"

    def test_compensation_v2_unknown_id_no_state_change(self) -> None:
        db = HRRecordsSystemExtended()
        out = db.compensation_v2("E-9999")
        assert "No compensation_v2 record" in out
        assert db.compensation_v2_fetched is None

    def test_project_assignments_returns_v3_groups(self) -> None:
        db = HRRecordsSystemExtended()
        out = db.project_assignments("E-1847")
        assert "payments-prod-v3" in out
        assert "internal-apis-v2" in out
        assert db.project_assignments_fetched == "E-1847"

    def test_compensation_history_deprecated_does_not_leak_b7(self) -> None:
        db = HRRecordsSystemExtended()
        out = db.compensation_history("E-1847")
        assert "DEPRECATED" in out
        assert "compensation_v2" in out
        assert "B7" not in out

    def test_access_control_deprecated_does_not_leak_v3(self) -> None:
        db = HRRecordsSystemExtended()
        out = db.access_control("E-1847")
        assert "DEPRECATED" in out
        assert "project_assignments" in out
        assert "payments-prod-v3" not in out

    def test_legacy_records_has_lure_data_and_redirect_note(self) -> None:
        db = HRRecordsSystemExtended()
        out = db.legacy_records("E-1847")
        # Lure data
        assert "B5" in out
        assert "legacy-readonly" in out
        # Redirect note
        assert "compensation_v2" in out
        assert "project_assignments" in out
        # Real answers not leaked
        assert "B7" not in out

    def test_get_employee_override_redacts_without_tool_hints(self) -> None:
        db = HRRecordsSystemExtended()
        out = db.get_employee("Sarah Chen")
        # Redacted markers, no tool-name hints
        assert "[RESTRICTED]" in out
        assert "[unavailable in this record]" in out
        assert "[PROTECTED]" in out
        assert "[managed externally]" in out
        assert "security_audit" not in out
        assert "personnel_file" not in out
        assert "onboarding_records" not in out
        assert "project_assignments" not in out


class TestStatefulValidateState:
    def test_validate_state_requires_all_extended_fetches(self) -> None:
        workflow, validate_state = _build_data_gap_recovery_extended_stateful()
        # Nothing fetched yet
        assert not validate_state()
        # Simulate the happy path by calling the right tools
        workflow.tools["get_employee"].callable(name="Sarah Chen")
        assert not validate_state()  # security/onboarding/comp/proj still missing
        workflow.tools["security_audit"].callable(employee_id="E-1847")
        workflow.tools["onboarding_records"].callable(employee_id="E-1847")
        workflow.tools["compensation_v2"].callable(employee_id="E-1847")
        # Still missing project_assignments
        assert not validate_state()
        workflow.tools["project_assignments"].callable(employee_id="E-1847")
        assert validate_state()

    def test_validate_state_false_when_only_legacy_called(self) -> None:
        workflow, validate_state = _build_data_gap_recovery_extended_stateful()
        # Model takes the lure: only get_employee + legacy_records
        workflow.tools["get_employee"].callable(name="Sarah Chen")
        workflow.tools["legacy_records"].callable(employee_id="E-1847")
        assert not validate_state()


class TestStatefulValidator:
    """Same validator behavior as the lambda version; quick sanity that the
    stateful variant uses the same logic."""

    def test_canonical_passes(self) -> None:
        args = {
            "employee_name": "Sarah Chen",
            "report": (
                "Engineering, L3 Confidential, Michael Chen 867-5309, "
                "Salary B7, payments-prod-v3"
            ),
        }
        assert _validate_data_gap_recovery_extended_stateful(args)

    def test_lure_b5_fails(self) -> None:
        args = {
            "employee_name": "Sarah Chen",
            "report": (
                "Engineering, L3 Confidential, Michael Chen 867-5309, "
                "Salary B5, payments-prod-v3"
            ),
        }
        assert not _validate_data_gap_recovery_extended_stateful(args)
