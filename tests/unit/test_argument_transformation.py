"""Tests for the argument_transformation scenario.

Verifies:
- Each tool callable returns the expected content (transactions table,
  approved vendors, vendor alias resolution, currency conversion, etc.)
- The validator accepts a canonical-correct terminal call and rejects
  each top-tier failure mode (skipped currency conversion, vendor case-
  mismatch over-flag, strict > instead of >=, wrong top vendor)
- The stateful variant's ExpenseAuditSystem tracks calls correctly so
  validate_state can verify the model used the right reasoning path
"""

from __future__ import annotations

from tests.eval.scenarios import (
    argument_transformation,
    argument_transformation_stateful,
)
from tests.eval.scenarios._model_reasoning import (
    _argument_transformation_tools,
    _validate_argument_transformation,
)
from tests.eval.scenarios._stateful_model_reasoning import (
    ExpenseAuditSystem,
    _build_argument_transformation_stateful,
    _validate_argument_transformation_stateful,
)


# ── Lambda tool callables ───────────────────────────────────────


class TestLambdaTools:
    """Each tool returns the data the model needs to derive the answer."""

    def _call(self, tool_name: str, **kwargs: object) -> str:
        tool = _argument_transformation_tools[tool_name]
        return tool.callable(**kwargs)

    def test_list_transactions_returns_q4_2024_set(self) -> None:
        out = self._call("list_transactions", quarter="Q4", year=2024)
        # All 13 transaction IDs present
        for i in range(1, 14):
            assert f"TX-10{i:02d}" in out
        # Mixed currencies — both USD and EUR transactions visible
        assert "EUR" in out
        assert "USD" in out
        # Boundary case at $5,000 exactly
        assert "5,000.00 USD" in out
        # The case-mismatch alias is in the transaction list verbatim
        assert "ACME Corp" in out
        # The largest flagged transaction's vendor is in the list
        assert "Wonka Industries" in out

    def test_list_transactions_unknown_quarter_returns_empty(self) -> None:
        out = self._call("list_transactions", quarter="Q3", year=2024)
        assert "No transactions found" in out
        # No transaction IDs leaked
        assert "TX-1001" not in out

    def test_get_approved_vendors_returns_canonical_six(self) -> None:
        out = self._call("get_approved_vendors")
        for v in (
            "Acme Corp", "Globex Industries", "Initech Systems",
            "Umbrella Logistics", "Wayne Enterprises", "Stark Industries",
        ):
            assert v in out
        # Case-mismatch alias must NOT appear in the canonical list
        assert "ACME Corp" not in out

    def test_get_vendor_details_resolves_acme_alias(self) -> None:
        out = self._call("get_vendor_details", vendor_name="ACME Corp")
        # The whole point of this tool: model can disambiguate by reading
        # this. "alias of Acme Corp" is the key signal.
        assert "alias of Acme Corp" in out
        assert "unified entity" in out

    def test_get_vendor_details_acme_corp_lists_alias(self) -> None:
        out = self._call("get_vendor_details", vendor_name="Acme Corp")
        assert "master account" in out
        assert "ACME Corp" in out  # alias listed under master record

    def test_get_vendor_details_unknown_vendor_returns_not_found(self) -> None:
        out = self._call("get_vendor_details", vendor_name="Cyberdyne LLC")
        # Unapproved vendors return a no-record stub — model can still
        # proceed by trusting the approved-vendors list directly.
        assert "not found in vendor master" in out

    def test_currency_convert_eur_to_usd_uses_fixed_rate(self) -> None:
        out = self._call(
            "currency_convert", amount=4800, from_currency="EUR", to_currency="USD",
        )
        # 4800 * 1.10 = 5280
        assert "5,280.00 USD" in out
        assert "1 EUR = 1.1 USD" in out

    def test_currency_convert_under_threshold_eur(self) -> None:
        # TX-1011: 2400 EUR = 2640 USD < $5K (correctly excluded)
        out = self._call(
            "currency_convert", amount=2400, from_currency="EUR", to_currency="USD",
        )
        assert "2,640.00 USD" in out

    def test_currency_convert_unsupported_pair(self) -> None:
        out = self._call(
            "currency_convert", amount=100, from_currency="GBP", to_currency="JPY",
        )
        assert "Unsupported" in out

    def test_categorize_expense_is_distractor_no_audit_info(self) -> None:
        out = self._call("categorize_expense", amount=7500, category="contractor")
        # Distractor: no flagging signal, just GL bucket noise
        assert "GL-" in out
        assert "TX-1001" not in out
        assert "approved" not in out.lower()

    def test_lookup_transaction_redundant_helper(self) -> None:
        # Returns the same data the model already has from list_transactions
        out = self._call("lookup_transaction", transaction_id="TX-1006")
        assert "Pied Piper" in out
        assert "4,800.00 EUR" in out

    def test_lookup_transaction_unknown_id(self) -> None:
        out = self._call("lookup_transaction", transaction_id="TX-9999")
        assert "No transaction found" in out

    def test_submit_audit_report_acks_inputs(self) -> None:
        out = self._call(
            "submit_audit_report",
            transaction_ids="TX-1001, TX-1005, TX-1006, TX-1008",
            total_flagged_usd="$28,980.00",
            top_vendor="Wonka Industries",
        )
        assert "TX-1001" in out
        assert "Wonka Industries" in out


# ── Validator ───────────────────────────────────────────────────


class TestValidator:
    """All four IDs, the correct USD total, and the correct top vendor
    must appear in the terminal call args. Each top-tier failure mode
    must be rejected."""

    def _canonical_args(self) -> dict[str, str]:
        return {
            "transaction_ids": "TX-1001, TX-1005, TX-1006, TX-1008",
            "total_flagged_usd": "$28,980.00",
            "top_vendor": "Wonka Industries",
        }

    def test_canonical_args_pass(self) -> None:
        assert _validate_argument_transformation(self._canonical_args())

    def test_alt_format_no_comma_no_dollar_passes(self) -> None:
        # Validator should be permissive on formatting — what matters is
        # that the right values are present.
        args = {
            "transaction_ids": "TX-1001 TX-1005 TX-1006 TX-1008",
            "total_flagged_usd": "28980 USD",
            "top_vendor": "wonka industries",
        }
        assert _validate_argument_transformation(args)

    def test_skip_currency_conversion_fails(self) -> None:
        # Model treated TX-1006 as 4,800 EUR < $5K (didn't convert),
        # so excluded it. Total = 7500 + 5000 + 11200 = 23,700.
        args = self._canonical_args()
        args["transaction_ids"] = "TX-1001, TX-1005, TX-1008"
        args["total_flagged_usd"] = "$23,700"
        assert not _validate_argument_transformation(args)

    def test_case_mismatch_over_flag_fails(self) -> None:
        # Model failed to disambiguate ACME Corp / Acme Corp — over-
        # flagged TX-1009. IDs are still all there (substring AND tolerates
        # extras) but total is wrong: 28,980 + 6,500 = 35,480.
        args = self._canonical_args()
        args["transaction_ids"] = "TX-1001, TX-1005, TX-1006, TX-1008, TX-1009"
        args["total_flagged_usd"] = "$35,480"
        assert not _validate_argument_transformation(args)

    def test_strict_gt_threshold_fails(self) -> None:
        # Model interpreted "$5,000 or more" as strict >, missed TX-1005.
        args = self._canonical_args()
        args["transaction_ids"] = "TX-1001, TX-1006, TX-1008"
        args["total_flagged_usd"] = "$23,980"
        assert not _validate_argument_transformation(args)

    def test_wrong_top_vendor_fails(self) -> None:
        args = self._canonical_args()
        args["top_vendor"] = "Cyberdyne LLC"
        assert not _validate_argument_transformation(args)

    def test_missing_one_id_fails(self) -> None:
        args = self._canonical_args()
        args["transaction_ids"] = "TX-1001, TX-1005, TX-1008"
        assert not _validate_argument_transformation(args)

    def test_missing_total_fails(self) -> None:
        args = self._canonical_args()
        args["total_flagged_usd"] = ""
        assert not _validate_argument_transformation(args)


# ── Scenario shape ──────────────────────────────────────────────


class TestScenarioShape:
    def test_lambda_scenario_has_7_tools(self) -> None:
        assert len(argument_transformation.workflow.tools) == 7

    def test_lambda_scenario_terminal_is_submit_audit_report(self) -> None:
        assert argument_transformation.workflow.terminal_tool == "submit_audit_report"

    def test_lambda_scenario_required_steps(self) -> None:
        assert argument_transformation.workflow.required_steps == [
            "list_transactions", "get_approved_vendors",
        ]

    def test_lambda_scenario_iteration_budget(self) -> None:
        assert argument_transformation.ideal_iterations == 5
        assert argument_transformation.max_iterations == 15

    def test_lambda_scenario_tags(self) -> None:
        assert "model_quality" in argument_transformation.tags
        assert "reasoning" in argument_transformation.tags


# ── Stateful backend ────────────────────────────────────────────


class TestExpenseAuditSystem:
    """Backend tracks call state so validate_state can verify the model
    used currency_convert (for EUR) and get_vendor_details (for ACME Corp)."""

    def test_initial_state_empty(self) -> None:
        db = ExpenseAuditSystem()
        assert db.list_called_for is None
        assert db.approved_called is False
        assert db.vendor_details_called_for == set()
        assert db.eur_conversion_called is False
        assert db.submitted_args is None

    def test_list_transactions_q4_2024_records_state(self) -> None:
        db = ExpenseAuditSystem()
        out = db.list_transactions("Q4", 2024)
        assert "TX-1001" in out
        assert db.list_called_for == ("Q4", 2024)

    def test_list_transactions_wrong_quarter_no_state(self) -> None:
        db = ExpenseAuditSystem()
        out = db.list_transactions("Q3", 2024)
        assert "No transactions found" in out
        assert db.list_called_for is None

    def test_get_approved_vendors_records_state(self) -> None:
        db = ExpenseAuditSystem()
        out = db.get_approved_vendors()
        assert "Acme Corp" in out
        assert db.approved_called is True

    def test_get_vendor_details_acme_alias_records_state(self) -> None:
        db = ExpenseAuditSystem()
        out = db.get_vendor_details("ACME Corp")
        assert "alias of Acme Corp" in out
        assert "ACME Corp" in db.vendor_details_called_for

    def test_currency_convert_eur_records_state(self) -> None:
        db = ExpenseAuditSystem()
        out = db.currency_convert(4800, "EUR", "USD")
        assert "5,280.00 USD" in out
        assert db.eur_conversion_called is True

    def test_currency_convert_usd_to_eur_does_not_record_eur_flag(self) -> None:
        # The state flag is specifically for the EUR->USD path the model
        # needs for the audit.
        db = ExpenseAuditSystem()
        db.currency_convert(1000, "USD", "EUR")
        assert db.eur_conversion_called is False

    def test_submit_audit_report_records_args(self) -> None:
        db = ExpenseAuditSystem()
        db.submit_audit_report(
            transaction_ids="TX-1001",
            total_flagged_usd="$1.00",
            top_vendor="Anyone",
        )
        assert db.submitted_args == {
            "transaction_ids": "TX-1001",
            "total_flagged_usd": "$1.00",
            "top_vendor": "Anyone",
        }


class TestStatefulValidateState:
    def test_validate_state_requires_full_reasoning_path(self) -> None:
        workflow, validate_state = _build_argument_transformation_stateful()
        assert not validate_state()
        # Just calling list/approved is not enough
        workflow.tools["list_transactions"].callable(quarter="Q4", year=2024)
        workflow.tools["get_approved_vendors"].callable()
        assert not validate_state()
        # Need EUR conversion AND ACME Corp lookup
        workflow.tools["currency_convert"].callable(
            amount=4800, from_currency="EUR", to_currency="USD",
        )
        assert not validate_state()
        workflow.tools["get_vendor_details"].callable(vendor_name="ACME Corp")
        assert not validate_state()
        # Plus the canonical submit
        workflow.tools["submit_audit_report"].callable(
            transaction_ids="TX-1001, TX-1005, TX-1006, TX-1008",
            total_flagged_usd="$28,980",
            top_vendor="Wonka Industries",
        )
        assert validate_state()

    def test_validate_state_false_when_skipped_currency_convert(self) -> None:
        workflow, validate_state = _build_argument_transformation_stateful()
        workflow.tools["list_transactions"].callable(quarter="Q4", year=2024)
        workflow.tools["get_approved_vendors"].callable()
        workflow.tools["get_vendor_details"].callable(vendor_name="ACME Corp")
        # Skipped currency_convert — even if model guessed the right answer,
        # state validation fails.
        workflow.tools["submit_audit_report"].callable(
            transaction_ids="TX-1001, TX-1005, TX-1006, TX-1008",
            total_flagged_usd="$28,980",
            top_vendor="Wonka Industries",
        )
        assert not validate_state()

    def test_validate_state_false_when_skipped_vendor_details(self) -> None:
        workflow, validate_state = _build_argument_transformation_stateful()
        workflow.tools["list_transactions"].callable(quarter="Q4", year=2024)
        workflow.tools["get_approved_vendors"].callable()
        workflow.tools["currency_convert"].callable(
            amount=4800, from_currency="EUR", to_currency="USD",
        )
        # Skipped get_vendor_details for ACME Corp
        workflow.tools["submit_audit_report"].callable(
            transaction_ids="TX-1001, TX-1005, TX-1006, TX-1008",
            total_flagged_usd="$28,980",
            top_vendor="Wonka Industries",
        )
        assert not validate_state()

    def test_validate_state_false_when_wrong_args_submitted(self) -> None:
        workflow, validate_state = _build_argument_transformation_stateful()
        workflow.tools["list_transactions"].callable(quarter="Q4", year=2024)
        workflow.tools["get_approved_vendors"].callable()
        workflow.tools["currency_convert"].callable(
            amount=4800, from_currency="EUR", to_currency="USD",
        )
        workflow.tools["get_vendor_details"].callable(vendor_name="ACME Corp")
        # Used the right tools but submitted the wrong total
        workflow.tools["submit_audit_report"].callable(
            transaction_ids="TX-1001, TX-1005, TX-1006, TX-1008",
            total_flagged_usd="$99,999",
            top_vendor="Wonka Industries",
        )
        assert not validate_state()


class TestStatefulValidator:
    """Sanity check that the stateful validator behaves like the lambda one."""

    def test_canonical_passes(self) -> None:
        args = {
            "transaction_ids": "TX-1001, TX-1005, TX-1006, TX-1008",
            "total_flagged_usd": "$28,980.00",
            "top_vendor": "Wonka Industries",
        }
        assert _validate_argument_transformation_stateful(args)

    def test_skip_currency_fails(self) -> None:
        args = {
            "transaction_ids": "TX-1001, TX-1005, TX-1008",
            "total_flagged_usd": "$23,700",
            "top_vendor": "Wonka Industries",
        }
        assert not _validate_argument_transformation_stateful(args)


# ── Scenario registered in ALL_SCENARIOS ────────────────────────


class TestRegistration:
    def test_lambda_in_all_scenarios(self) -> None:
        from tests.eval.scenarios import ALL_SCENARIOS
        names = [s.name for s in ALL_SCENARIOS]
        assert "argument_transformation" in names

    def test_stateful_in_all_scenarios(self) -> None:
        from tests.eval.scenarios import ALL_SCENARIOS
        names = [s.name for s in ALL_SCENARIOS]
        assert "argument_transformation_stateful" in names
