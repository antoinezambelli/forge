"""Tests for Guardrails facade (bundled two-method API)."""

import pytest

from forge.clients.replies import text_reply
from tests.canonical import assistant, call
from forge.guardrails import CheckResult, Guardrails


# ── Helpers ──────────────────────────────────────────────────


def make_guardrails(**overrides):
    defaults = dict(
        tool_names=["search", "lookup", "answer"],
        required_steps=["search", "lookup"],
        terminal_tool="answer",
    )
    defaults.update(overrides)
    return Guardrails(**defaults)


# ── check(): response validation ─────────────────────────────


class TestCheckValidation:
    def test_text_response_triggers_retry(self):
        g = make_guardrails()
        result = g.check(text_reply("I think 42"))
        assert result.action == "retry"
        assert result.nudge is not None
        assert result.tool_calls is None

    def test_unknown_tool_triggers_tool_error(self):
        g = make_guardrails()
        result = g.check(assistant(call(name="nonexistent", arguments={})))
        assert result.action == "tool_error"
        assert result.nudge is not None
        assert result.nudge.role == "tool"

    def test_malformed_args_triggers_tool_error(self):
        g = make_guardrails()
        result = g.check(assistant(call(name="search", arguments="not a dict")))
        assert result.action == "tool_error"
        assert result.nudge is not None
        assert result.nudge.role == "tool"
        assert result.nudge.kind == "tool_arg_validation"

    def test_valid_tool_call_returns_execute(self):
        g = make_guardrails()
        result = g.check(assistant(call(name="search", arguments={"q": "test"})))
        assert result.action == "execute"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.nudge is None

    def test_rescue_configuration_controls_text_tool_parsing(self):
        text = '```json\n{"tool": "search", "args": {"q": "test"}}\n```'
        cases = [("enabled", True, "execute"), ("disabled", False, "retry")]
        for label, enabled, expected_action in cases:
            g = make_guardrails(rescue_enabled=enabled)
            result = g.check(text_reply(text))
            assert result.action == expected_action, label
            if enabled:
                assert result.tool_calls[0].name == "search", label


class TestCheckRetryExhaustion:
    def test_fatal_after_max_retries(self):
        g = make_guardrails(max_retries=2)
        # Burn through retries
        for _ in range(2):
            result = g.check(text_reply("nope"))
            assert result.action == "retry"
        # Next one should be fatal
        result = g.check(text_reply("still nope"))
        assert result.action == "fatal"
        assert "bad responses" in result.reason

    def test_retries_reset_on_valid_response(self):
        g = make_guardrails(max_retries=2)
        g.check(text_reply("nope"))
        g.check(text_reply("nope"))
        # Valid response resets the counter
        g.check(assistant(call(name="search", arguments={})))
        # So this is a retry again, not fatal
        result = g.check(text_reply("nope again"))
        assert result.action == "retry"


class TestCheckToolErrorBudget:
    """Malformed args drain the tool-error budget; unknown-tool drains retries.

    Mirrors run_inference so the middleware facade accounts identically.
    """

    def test_malformed_args_drains_tool_error_budget(self):
        g = make_guardrails(max_tool_errors=2, max_retries=5)
        for _ in range(2):
            r = g.check(assistant(call(name="search", arguments="bad")))
            assert r.action == "tool_error"
        r = g.check(assistant(call(name="search", arguments="bad")))
        assert r.action == "fatal"
        assert "tool-argument" in r.reason

    def test_malformed_args_do_not_drain_retry_budget(self):
        # Tool-arg errors use their own budget; the retry budget stays intact.
        g = make_guardrails(max_tool_errors=5, max_retries=2)
        for _ in range(3):
            assert g.check(assistant(call(name="search", arguments="bad"))).action == "tool_error"
        # Retry budget untouched → a bare-text failure is still a retry.
        assert g.check(text_reply("nope")).action == "retry"

    def test_unknown_tool_drains_retry_budget(self):
        # Unknown-tool rides the tool channel (action="tool_error") but still
        # drains the RETRY budget, matching run_inference.
        g = make_guardrails(max_retries=2, max_tool_errors=5)
        for _ in range(2):
            assert g.check(assistant(call(name="ghost", arguments={}))).action == "tool_error"
        r = g.check(assistant(call(name="ghost", arguments={})))
        assert r.action == "fatal"
        assert "bad responses" in r.reason


# ── check(): step enforcement ────────────────────────────────


class TestCheckStepEnforcement:
    def test_premature_terminal_blocked(self):
        g = make_guardrails()
        result = g.check(assistant(call(name="answer", arguments={})))
        assert result.action == "step_blocked"
        assert result.nudge is not None

    def test_terminal_allowed_after_steps(self):
        g = make_guardrails()
        # Complete required steps
        g.check(assistant(call(name="search", arguments={})))
        g.record(["search"])
        g.check(assistant(call(name="lookup", arguments={})))
        g.record(["lookup"])
        # Now terminal is allowed
        result = g.check(assistant(call(name="answer", arguments={})))
        assert result.action == "execute"

    def test_fatal_after_max_premature_attempts(self):
        g = make_guardrails(max_premature_attempts=2)
        for _ in range(2):
            result = g.check(assistant(call(name="answer", arguments={})))
            assert result.action == "step_blocked"
        result = g.check(assistant(call(name="answer", arguments={})))
        assert result.action == "fatal"
        assert "skipped required steps" in result.reason

    def test_no_required_steps_allows_terminal(self):
        g = make_guardrails(required_steps=[])
        result = g.check(assistant(call(name="answer", arguments={})))
        assert result.action == "execute"


# ── record() ─────────────────────────────────────────────────


class TestRecord:
    def test_returns_completion_across_workflow_lifecycle(self):
        g = make_guardrails()
        g.check(assistant(call(name="search", arguments={})))
        assert g.record(["search"]) is False
        g.check(assistant(call(name="lookup", arguments={})))
        g.record(["lookup"])
        g.check(assistant(call(name="answer", arguments={})))
        assert g.record(["answer"]) is True

    def test_returns_false_when_terminal_but_unsatisfied(self):
        """Edge case: if record() is called with the terminal tool
        but steps aren't satisfied (shouldn't happen if check() is
        used, but record() should still be safe)."""
        g = make_guardrails()
        done = g.record(["answer"])
        assert done is False

    def test_records_multiple_tools(self):
        g = make_guardrails()
        g.check(assistant(call(name="search", arguments={}), call(name="lookup", arguments={})))
        g.record(["search", "lookup"])
        result = g.check(assistant(call(name="answer", arguments={})))
        assert result.action == "execute"


# ── CheckResult dataclass ────────────────────────────────────


class TestCheckResult:
    def test_result_is_immutable(self):
        r = CheckResult(action="execute", tool_calls=[])
        with pytest.raises(AttributeError):
            r.action = "retry"


# ── Imports ──────────────────────────────────────────────────


class TestImports:
    def test_public_imports(self):
        from forge.guardrails import CheckResult, Guardrails

        assert CheckResult is not None
        assert Guardrails is not None
        from forge import CheckResult as TopLevelCheckResult
        from forge import Guardrails as TopLevelGuardrails

        assert TopLevelCheckResult is CheckResult
        assert TopLevelGuardrails is Guardrails


# ── Custom retry nudge ───────────────────────────────────────


class TestCustomRetryNudge:
    def test_default_nudge_used_when_none(self):
        g = make_guardrails()
        result = g.check(text_reply("bare text"))
        assert result.action == "retry"
        assert "tool call" in result.nudge.content.lower()

    def test_custom_nudge_receives_raw_response_and_supplies_content(self):
        captured = []

        def nudge_fn(raw):
            captured.append(raw)
            return f"Wrap this in respond: {raw}"

        g = make_guardrails(retry_nudge=nudge_fn, rescue_enabled=False)
        result = g.check(text_reply("my bare text"))
        assert result.action == "retry"
        assert result.nudge.content == "Wrap this in respond: my bare text"
        assert captured == ["my bare text"]
