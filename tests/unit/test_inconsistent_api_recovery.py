"""Tests for the inconsistent_api_recovery scenario.

Verifies:
- Each tool returns the canonical-correct payload when called with valid args
- Each tool returns a helpful ERROR message when called with the typical
  pattern-matching mistake from a prior tool's lesson (wrong format)
- The validator accepts a canonical-correct submitted report and rejects
  the obvious failure shapes (missing total, wrong total, wrong status)
- The stateful variant's LegacyAPISystem tracks per-tool calls and the
  validate_state lambda enforces the full canonical audit path
"""

from __future__ import annotations

import json

from tests.eval.scenarios import (
    inconsistent_api_recovery,
    inconsistent_api_recovery_stateful,
)
from tests.eval.scenarios._model_reasoning import (
    _inconsistent_api_recovery_tools,
    _validate_inconsistent_api_recovery,
)
from tests.eval.scenarios._stateful_model_reasoning import (
    LegacyAPISystem,
    _build_inconsistent_api_recovery_stateful,
    _validate_inconsistent_api_recovery_stateful,
)


# ── Lambda tool callables ───────────────────────────────────────


class TestLambdaTools:
    def _call(self, tool_name: str, **kwargs: object) -> str:
        return _inconsistent_api_recovery_tools[tool_name].callable(**kwargs)

    # legacy_list_accounts ----------------------------------------

    def test_list_accounts_canonical(self) -> None:
        out = self._call("legacy_list_accounts", page=1, page_size=10)
        assert "Acme Corp" in out
        assert "12345" in out
        assert "Globex Industries" in out

    def test_list_accounts_missing_args(self) -> None:
        out = self._call("legacy_list_accounts")
        assert "ERROR" in out
        assert "page" in out and "page_size" in out

    def test_list_accounts_zero_rejected(self) -> None:
        out = self._call("legacy_list_accounts", page=0, page_size=10)
        assert "ERROR" in out

    # legacy_get_balance ------------------------------------------

    def test_get_balance_canonical(self) -> None:
        out = self._call("legacy_get_balance", account_id="ACC-12345")
        assert "750000" in out
        assert "cents" in out
        assert "ACTIVE" in out

    def test_get_balance_missing_prefix_errors(self) -> None:
        # Pattern-match trap: model passes bare int from list_accounts.
        out = self._call("legacy_get_balance", account_id="12345")
        assert "ERROR" in out
        assert "ACC-" in out

    def test_get_balance_unknown_account(self) -> None:
        out = self._call("legacy_get_balance", account_id="ACC-99999")
        assert "ERROR" in out
        assert "not found" in out

    # legacy_get_transactions -------------------------------------

    def test_get_transactions_canonical(self) -> None:
        out = self._call(
            "legacy_get_transactions",
            account_id="ACC-12345", since="2024-10-01", until="2024-12-31",
        )
        assert "TXN/00042" in out
        assert "TXN/00043" in out
        assert "TXN/00044" in out
        assert "services" in out
        assert "hardware" in out

    def test_get_transactions_unix_ts_rejected(self) -> None:
        # Pattern-match trap: model reuses the unix ts from get_balance.
        out = self._call(
            "legacy_get_transactions",
            account_id="ACC-12345", since="1696000000", until="1704000000",
        )
        assert "ERROR" in out
        assert "ISO date" in out

    def test_get_transactions_missing_prefix(self) -> None:
        out = self._call(
            "legacy_get_transactions",
            account_id="12345", since="2024-10-01", until="2024-12-31",
        )
        assert "ERROR" in out
        assert "ACC-" in out

    # legacy_categorize_spend -------------------------------------

    def test_categorize_canonical(self) -> None:
        out = self._call(
            "legacy_categorize_spend", txn_id=42, category="SVCS",
        )
        assert "TXN/00042" in out
        assert "SVCS" in out
        assert "5000.00" in out  # cents->dollars conversion

    def test_categorize_full_string_id_rejected(self) -> None:
        # Pattern-match trap: pass the full TXN/00042 string.
        out = self._call(
            "legacy_categorize_spend", txn_id="TXN/00042", category="SVCS",
        )
        assert "ERROR" in out
        assert "numeric" in out

    def test_categorize_lowercase_category_rejected(self) -> None:
        # Pattern-match trap: pass the lowercase category from
        # get_transactions.
        out = self._call(
            "legacy_categorize_spend", txn_id=42, category="services",
        )
        assert "ERROR" in out
        assert "uppercase" in out

    def test_categorize_unknown_code_rejected(self) -> None:
        out = self._call(
            "legacy_categorize_spend", txn_id=42, category="FOOD",
        )
        assert "ERROR" in out

    # legacy_check_compliance -------------------------------------

    def test_compliance_canonical(self) -> None:
        out = self._call(
            "legacy_check_compliance", region="us", period="2024-Q4",
        )
        assert "PASS" in out
        assert "us" in out
        assert "2024-Q4" in out

    def test_compliance_uppercase_region_rejected(self) -> None:
        # Pattern-match trap: model just learned UPPERCASE codes.
        out = self._call(
            "legacy_check_compliance", region="US", period="2024-Q4",
        )
        assert "ERROR" in out
        assert "lowercase" in out

    def test_compliance_iso_date_rejected(self) -> None:
        # Pattern-match trap: model uses ISO date instead of quarter.
        out = self._call(
            "legacy_check_compliance", region="us", period="2024-12-31",
        )
        assert "ERROR" in out
        assert "quarter" in out

    # legacy_aggregate_subtotal -----------------------------------

    def test_aggregate_canonical(self) -> None:
        out = self._call(
            "legacy_aggregate_subtotal",
            amounts="5000.00|12500.00|8000.00",
        )
        assert "25500.00" in out
        assert "3 amounts" in out

    def test_aggregate_comma_separated_rejected(self) -> None:
        out = self._call(
            "legacy_aggregate_subtotal",
            amounts="5000.00,12500.00,8000.00",
        )
        assert "ERROR" in out
        assert "pipe-separated" in out

    def test_aggregate_empty_rejected(self) -> None:
        out = self._call("legacy_aggregate_subtotal", amounts="")
        assert "ERROR" in out

    def test_aggregate_non_numeric_rejected(self) -> None:
        out = self._call(
            "legacy_aggregate_subtotal", amounts="abc|def",
        )
        assert "ERROR" in out
        assert "non-numeric" in out

    # legacy_submit_audit -----------------------------------------

    def test_submit_audit_echoes(self) -> None:
        report = (
            '{"flagged_count": 0, "total_usd": "25500.00", '
            '"compliance_status": "PASS", "txn_count": 3}'
        )
        out = self._call("legacy_submit_audit", report=report)
        assert "Audit submitted" in out
        assert "25500.00" in out


# ── Validator (lambda) ──────────────────────────────────────────


class TestValidator:
    def _canonical_report(self) -> str:
        return (
            '{"flagged_count": 0, "total_usd": "25500.00", '
            '"compliance_status": "PASS", "txn_count": 3}'
        )

    def test_canonical_passes(self) -> None:
        args = {"report": self._canonical_report()}
        assert _validate_inconsistent_api_recovery(args) is True

    def test_missing_total_fails(self) -> None:
        args = {"report": '{"compliance_status": "PASS", "txn_count": 3}'}
        assert _validate_inconsistent_api_recovery(args) is False

    def test_wrong_total_fails(self) -> None:
        args = {"report": (
            '{"flagged_count": 0, "total_usd": "20500.00", '
            '"compliance_status": "PASS", "txn_count": 3}'
        )}
        assert _validate_inconsistent_api_recovery(args) is False

    def test_wrong_status_fails(self) -> None:
        args = {"report": (
            '{"flagged_count": 0, "total_usd": "25500.00", '
            '"compliance_status": "FAIL", "txn_count": 3}'
        )}
        assert _validate_inconsistent_api_recovery(args) is False

    def test_total_with_comma_passes(self) -> None:
        # _check strips commas; matcher is substring-AND on lowercased text.
        args = {"report": (
            '{"flagged_count": 0, "total_usd": "25,500.00", '
            '"compliance_status": "PASS", "txn_count": 3}'
        )}
        assert _validate_inconsistent_api_recovery(args) is True

    def test_empty_args_fail(self) -> None:
        assert _validate_inconsistent_api_recovery({}) is False


# ── Scenario shape ──────────────────────────────────────────────


class TestScenarioShape:
    def test_lambda_scenario_metadata(self) -> None:
        assert inconsistent_api_recovery.name == "inconsistent_api_recovery"
        assert "advanced_reasoning" in inconsistent_api_recovery.tags
        assert "error_recovery" in inconsistent_api_recovery.tags
        assert inconsistent_api_recovery.workflow.terminal_tool == "legacy_submit_audit"
        assert "legacy_list_accounts" in inconsistent_api_recovery.workflow.required_steps
        assert inconsistent_api_recovery.ideal_iterations == 8
        assert inconsistent_api_recovery.max_iterations == 20

    def test_lambda_scenario_has_seven_tools(self) -> None:
        tools = inconsistent_api_recovery.workflow.tools
        expected = {
            "legacy_list_accounts", "legacy_get_balance",
            "legacy_get_transactions", "legacy_categorize_spend",
            "legacy_check_compliance", "legacy_aggregate_subtotal",
            "legacy_submit_audit",
        }
        assert set(tools.keys()) == expected

    def test_stateful_scenario_metadata(self) -> None:
        assert inconsistent_api_recovery_stateful.name == "inconsistent_api_recovery_stateful"
        assert "stateful" in inconsistent_api_recovery_stateful.tags
        assert "advanced_reasoning" in inconsistent_api_recovery_stateful.tags
        assert "error_recovery" in inconsistent_api_recovery_stateful.tags
        assert inconsistent_api_recovery_stateful.build_workflow is not None


# ── Stateful backend ────────────────────────────────────────────


class TestStatefulBackend:
    def test_initial_state_is_empty(self) -> None:
        db = LegacyAPISystem()
        assert db.list_called is False
        assert db.balance_fetched_for == set()
        assert db.transactions_fetched_for == {}
        assert db.categorizations == []
        assert db.compliance_checked_for is None
        assert db.subtotal_amounts is None
        assert db.submitted_args is None

    def test_list_accounts_sets_flag(self) -> None:
        db = LegacyAPISystem()
        out = db.list_accounts(page=1, page_size=10)
        assert db.list_called is True
        assert "Acme Corp" in out

    def test_list_accounts_error_does_not_set_flag(self) -> None:
        db = LegacyAPISystem()
        out = db.list_accounts(page=None, page_size=None)
        assert db.list_called is False
        assert "ERROR" in out

    def test_get_balance_tracks_account(self) -> None:
        db = LegacyAPISystem()
        db.get_balance("ACC-12345")
        assert "ACC-12345" in db.balance_fetched_for

    def test_get_balance_error_does_not_track(self) -> None:
        db = LegacyAPISystem()
        db.get_balance("12345")  # missing prefix
        assert db.balance_fetched_for == set()

    def test_get_transactions_tracks_range(self) -> None:
        db = LegacyAPISystem()
        db.get_transactions("ACC-12345", "2024-10-01", "2024-12-31")
        assert db.transactions_fetched_for["ACC-12345"] == ("2024-10-01", "2024-12-31")

    def test_get_transactions_unix_error_does_not_track(self) -> None:
        db = LegacyAPISystem()
        out = db.get_transactions("ACC-12345", "1696000000", "1704000000")
        assert "ERROR" in out
        assert db.transactions_fetched_for == {}

    def test_categorize_spend_tracks_call(self) -> None:
        db = LegacyAPISystem()
        db.categorize_spend(42, "SVCS")
        assert (42, "SVCS") in db.categorizations

    def test_categorize_lowercase_error_does_not_track(self) -> None:
        db = LegacyAPISystem()
        db.categorize_spend(42, "services")
        assert db.categorizations == []

    def test_check_compliance_tracks(self) -> None:
        db = LegacyAPISystem()
        db.check_compliance("us", "2024-Q4")
        assert db.compliance_checked_for == ("us", "2024-Q4")

    def test_check_compliance_uppercase_error_does_not_track(self) -> None:
        db = LegacyAPISystem()
        db.check_compliance("US", "2024-Q4")
        assert db.compliance_checked_for is None

    def test_aggregate_subtotal_tracks_amounts(self) -> None:
        db = LegacyAPISystem()
        db.aggregate_subtotal("5000.00|12500.00|8000.00")
        assert db.subtotal_amounts == [5000.0, 12500.0, 8000.0]

    def test_aggregate_comma_error_does_not_track(self) -> None:
        db = LegacyAPISystem()
        db.aggregate_subtotal("5000.00,12500.00,8000.00")
        assert db.subtotal_amounts is None

    def test_submit_audit_captures_args(self) -> None:
        db = LegacyAPISystem()
        report = '{"flagged_count": 0, "total_usd": "25500.00"}'
        db.submit_audit(report)
        assert db.submitted_args == {"report": report}


# ── Stateful validate_state ─────────────────────────────────────


class TestStatefulValidateState:
    def _canonical_report(self) -> str:
        return json.dumps({
            "flagged_count": 0,
            "total_usd": "25500.00",
            "compliance_status": "PASS",
            "txn_count": 3,
        })

    def _drive_canonical(self, db: LegacyAPISystem) -> None:
        db.list_accounts(page=1, page_size=10)
        db.get_balance("ACC-12345")
        db.get_transactions("ACC-12345", "2024-10-01", "2024-12-31")
        db.categorize_spend(42, "SVCS")
        db.check_compliance("us", "2024-Q4")
        db.aggregate_subtotal("5000.00|12500.00|8000.00")
        db.submit_audit(self._canonical_report())

    def test_canonical_path_passes(self) -> None:
        workflow, validate_state = _build_inconsistent_api_recovery_stateful()
        # Walk the LegacyAPISystem instance backing this workflow's tools.
        # The backend instance is captured in the lambda closures — invoke
        # tools through the workflow to drive its state.
        tools = workflow.tools
        tools["legacy_list_accounts"].callable(page=1, page_size=10)
        tools["legacy_get_balance"].callable(account_id="ACC-12345")
        tools["legacy_get_transactions"].callable(
            account_id="ACC-12345", since="2024-10-01", until="2024-12-31",
        )
        tools["legacy_categorize_spend"].callable(txn_id=42, category="SVCS")
        tools["legacy_check_compliance"].callable(region="us", period="2024-Q4")
        tools["legacy_aggregate_subtotal"].callable(
            amounts="5000.00|12500.00|8000.00",
        )
        tools["legacy_submit_audit"].callable(report=self._canonical_report())
        assert validate_state() is True

    def test_skipping_categorize_fails(self) -> None:
        workflow, validate_state = _build_inconsistent_api_recovery_stateful()
        tools = workflow.tools
        tools["legacy_list_accounts"].callable(page=1, page_size=10)
        tools["legacy_get_balance"].callable(account_id="ACC-12345")
        tools["legacy_get_transactions"].callable(
            account_id="ACC-12345", since="2024-10-01", until="2024-12-31",
        )
        tools["legacy_check_compliance"].callable(region="us", period="2024-Q4")
        tools["legacy_aggregate_subtotal"].callable(
            amounts="5000.00|12500.00|8000.00",
        )
        tools["legacy_submit_audit"].callable(report=self._canonical_report())
        assert validate_state() is False

    def test_uppercase_region_in_state_fails(self) -> None:
        workflow, validate_state = _build_inconsistent_api_recovery_stateful()
        tools = workflow.tools
        tools["legacy_list_accounts"].callable(page=1, page_size=10)
        tools["legacy_get_balance"].callable(account_id="ACC-12345")
        tools["legacy_get_transactions"].callable(
            account_id="ACC-12345", since="2024-10-01", until="2024-12-31",
        )
        tools["legacy_categorize_spend"].callable(txn_id=42, category="SVCS")
        # Failed compliance check (uppercase region) — tool returns ERROR
        # and does not update state. compliance_checked_for stays None.
        tools["legacy_check_compliance"].callable(region="US", period="2024-Q4")
        tools["legacy_aggregate_subtotal"].callable(
            amounts="5000.00|12500.00|8000.00",
        )
        tools["legacy_submit_audit"].callable(report=self._canonical_report())
        assert validate_state() is False

    def test_wrong_subtotal_fails(self) -> None:
        workflow, validate_state = _build_inconsistent_api_recovery_stateful()
        tools = workflow.tools
        tools["legacy_list_accounts"].callable(page=1, page_size=10)
        tools["legacy_get_balance"].callable(account_id="ACC-12345")
        tools["legacy_get_transactions"].callable(
            account_id="ACC-12345", since="2024-10-01", until="2024-12-31",
        )
        tools["legacy_categorize_spend"].callable(txn_id=42, category="SVCS")
        tools["legacy_check_compliance"].callable(region="us", period="2024-Q4")
        # Wrong subtotal — only 2 of 3 amounts.
        tools["legacy_aggregate_subtotal"].callable(
            amounts="5000.00|12500.00",
        )
        tools["legacy_submit_audit"].callable(report=self._canonical_report())
        assert validate_state() is False

    def test_wrong_report_content_fails(self) -> None:
        workflow, validate_state = _build_inconsistent_api_recovery_stateful()
        tools = workflow.tools
        tools["legacy_list_accounts"].callable(page=1, page_size=10)
        tools["legacy_get_balance"].callable(account_id="ACC-12345")
        tools["legacy_get_transactions"].callable(
            account_id="ACC-12345", since="2024-10-01", until="2024-12-31",
        )
        tools["legacy_categorize_spend"].callable(txn_id=42, category="SVCS")
        tools["legacy_check_compliance"].callable(region="us", period="2024-Q4")
        tools["legacy_aggregate_subtotal"].callable(
            amounts="5000.00|12500.00|8000.00",
        )
        # Wrong report — missing "txn_count" key required by validator.
        tools["legacy_submit_audit"].callable(
            report='{"flagged_count": 0, "total_usd": "25500", "compliance_status": "PASS"}',
        )
        assert validate_state() is False
