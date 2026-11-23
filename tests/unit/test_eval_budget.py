"""Tests for budget_override in eval runner."""

from __future__ import annotations

from functools import partial
from typing import Any

import pytest

import tests.eval.eval_runner as eval_runner
from tests.canonical import assistant, call
from forge.core.messages import TextPart

from tests.conftest import MockClient
from tests.eval.eval_runner import EvalConfig, RunResult, run_scenario
from tests.eval.scenarios import compaction_chain_p1, basic_2step

# The eval-budget tests expect the client to return a "stuck" TextResponse
# once scripted calls run out (rather than raising) and to stream a single
# FINAL chunk with no text deltas.
_MockClient = partial(
    MockClient,
    on_exhausted=assistant(TextPart("stuck")),
    stream_mode="final",
)


class TestBudgetOverride:
    """Verify that EvalConfig.budget_override overrides scenario.budget_tokens."""

    async def test_override_changes_budget(self) -> None:
        """budget_override should replace scenario.budget_tokens in ContextManager."""
        # compaction_chain_p1 has budget_tokens=3600
        assert compaction_chain_p1.budget_tokens == 3600

        # Build a client that completes the 10-step chain
        client = _MockClient([
            assistant(call("patient_lookup", {"patient_name": "Margaret Chen"})),
            assistant(call("pull_records", {"mrn": "MRN-84201"})),
            assistant(call("order_labs", {"encounter_id": "ENC-20250305"})),
            assistant(call("review_imaging", {"lab_id": "LAB-7718"})),
            assistant(call("request_referral", {"imaging_id": "IMG-3304"})),
            assistant(call("check_pharmacy", {"referral_id": "REF-5521"})),
            assistant(call("verify_insurance", {"patient_mrn": "MRN-84201"})),
            assistant(call("request_prior_auth", {"plan_id": "PLAN-BC-4490", "referral_id": "REF-5521"})),
            assistant(call("schedule_appointment", {"auth_id": "AUTH-9917", "referral_id": "REF-5521"})),
            assistant(call("submit_treatment_plan", {"summary": "MRN-84201 HbA1c 9.2% cortical thinning Patel metformin"})),
        ])

        config = EvalConfig(
            runs_per_scenario=1,
            budget_override=16384,
        )

        result = await run_scenario(client, compaction_chain_p1, config)
        assert result.completed
        # With 16384 budget, no compaction should fire on short mock responses
        assert len(result.compaction_events) == 0

    async def test_no_override_uses_scenario_default(self) -> None:
        """Without budget_override, scenario.budget_tokens is used."""
        assert basic_2step.budget_tokens == 8192

        client = _MockClient([
            assistant(call("get_country_info", {"country": "France"})),
            assistant(call("summarize", {"content": "test"})),
        ])

        config = EvalConfig(
            runs_per_scenario=1,
            budget_override=None,
        )

        result = await run_scenario(client, basic_2step, config)
        assert result.completed

    async def test_tight_budget_triggers_compaction(self) -> None:
        """A very tight budget with long responses should trigger compaction."""
        client = _MockClient([
            assistant(call("patient_lookup", {"patient_name": "Margaret Chen"})),
            assistant(call("pull_records", {"mrn": "MRN-84201"})),
            assistant(call("order_labs", {"encounter_id": "ENC-20250305"})),
            assistant(call("review_imaging", {"lab_id": "LAB-7718"})),
            assistant(call("request_referral", {"imaging_id": "IMG-3304"})),
            assistant(call("check_pharmacy", {"referral_id": "REF-5521"})),
            assistant(call("verify_insurance", {"patient_mrn": "MRN-84201"})),
            assistant(call("request_prior_auth", {"plan_id": "PLAN-BC-4490", "referral_id": "REF-5521"})),
            assistant(call("schedule_appointment", {"auth_id": "AUTH-9917", "referral_id": "REF-5521"})),
            assistant(call("submit_treatment_plan", {"summary": "MRN-84201 HbA1c 9.2% cortical thinning Patel metformin"})),
        ])

        from forge.context.phase_strategies import TieredCompact

        config = EvalConfig(
            runs_per_scenario=1,
            budget_override=64,  # Extremely tight — system prompt alone exceeds this
            strategy_overrides={"compaction": TieredCompact(keep_recent=2)},
        )

        result = await run_scenario(client, compaction_chain_p1, config)
        assert result.completed
        assert len(result.compaction_events) > 0

    async def test_each_run_builds_a_fresh_context_manager(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        created: list[object] = []
        context_manager = eval_runner.ContextManager

        class RecordingContextManager(context_manager):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                created.append(self)

        monkeypatch.setattr(eval_runner, "ContextManager", RecordingContextManager)
        responses = [
            assistant(call("get_country_info", {"country": "France"})),
            assistant(call("summarize", {"content": "test"})),
        ]
        config = EvalConfig(runs_per_scenario=1, budget_override=4096)

        for _ in range(2):
            result = await run_scenario(_MockClient(responses), basic_2step, config)
            assert result.completed

        assert len(created) == 2
        assert created[0] is not created[1]
