"""Tests for Guardrails facade (bundled two-method API)."""

import pytest

from forge.core.workflow import TextResponse, ToolCall
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
        result = g.check(TextResponse(content="I think 42"))
        assert result.action == "retry"
        assert result.nudge is not None
        assert result.tool_calls is None

    def test_unknown_tool_triggers_retry(self):
        g = make_guardrails()
        result = g.check([ToolCall(tool="nonexistent", args={})])
        assert result.action == "retry"
        assert result.nudge is not None

    def test_valid_tool_call_returns_execute(self):
        g = make_guardrails()
        result = g.check([ToolCall(tool="search", args={"q": "test"})])
        assert result.action == "execute"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.nudge is None

    def test_rescue_parses_tool_call_from_text(self):
        g = make_guardrails(rescue_enabled=True)
        text = '```json\n{"tool": "search", "args": {"q": "test"}}\n```'
        result = g.check(TextResponse(content=text))
        assert result.action == "execute"
        assert result.tool_calls[0].tool == "search"

    def test_rescue_disabled_skips_parsing(self):
        g = make_guardrails(rescue_enabled=False)
        text = '```json\n{"tool": "search", "args": {"q": "test"}}\n```'
        result = g.check(TextResponse(content=text))
        assert result.action == "retry"


class TestCheckRetryExhaustion:
    def test_fatal_after_max_retries(self):
        g = make_guardrails(max_retries=2)
        # Burn through retries
        for _ in range(2):
            result = g.check(TextResponse(content="nope"))
            assert result.action == "retry"
        # Next one should be fatal
        result = g.check(TextResponse(content="still nope"))
        assert result.action == "fatal"
        assert "bad responses" in result.reason

    def test_retries_reset_on_valid_response(self):
        g = make_guardrails(max_retries=2)
        g.check(TextResponse(content="nope"))
        g.check(TextResponse(content="nope"))
        # Valid response resets the counter
        g.check([ToolCall(tool="search", args={})])
        # So this is a retry again, not fatal
        result = g.check(TextResponse(content="nope again"))
        assert result.action == "retry"


# ── check(): step enforcement ────────────────────────────────


class TestCheckStepEnforcement:
    def test_premature_terminal_blocked(self):
        g = make_guardrails()
        result = g.check([ToolCall(tool="answer", args={})])
        assert result.action == "step_blocked"
        assert result.nudge is not None

    def test_terminal_allowed_after_steps(self):
        g = make_guardrails()
        # Complete required steps
        g.check([ToolCall(tool="search", args={})])
        g.record(["search"])
        g.check([ToolCall(tool="lookup", args={})])
        g.record(["lookup"])
        # Now terminal is allowed
        result = g.check([ToolCall(tool="answer", args={})])
        assert result.action == "execute"

    def test_fatal_after_max_premature_attempts(self):
        g = make_guardrails(max_premature_attempts=2)
        for _ in range(2):
            result = g.check([ToolCall(tool="answer", args={})])
            assert result.action == "step_blocked"
        result = g.check([ToolCall(tool="answer", args={})])
        assert result.action == "fatal"
        assert "skipped required steps" in result.reason

    def test_no_required_steps_allows_terminal(self):
        g = make_guardrails(required_steps=[])
        result = g.check([ToolCall(tool="answer", args={})])
        assert result.action == "execute"


# ── record() ─────────────────────────────────────────────────


class TestRecord:
    def test_returns_false_for_non_terminal(self):
        g = make_guardrails()
        g.check([ToolCall(tool="search", args={})])
        done = g.record(["search"])
        assert done is False

    def test_returns_true_when_terminal_and_satisfied(self):
        g = make_guardrails()
        g.check([ToolCall(tool="search", args={})])
        g.record(["search"])
        g.check([ToolCall(tool="lookup", args={})])
        g.record(["lookup"])
        g.check([ToolCall(tool="answer", args={})])
        done = g.record(["answer"])
        assert done is True

    def test_returns_false_when_terminal_but_unsatisfied(self):
        """Edge case: if record() is called with the terminal tool
        but steps aren't satisfied (shouldn't happen if check() is
        used, but record() should still be safe)."""
        g = make_guardrails()
        done = g.record(["answer"])
        assert done is False

    def test_records_multiple_tools(self):
        g = make_guardrails()
        g.check([ToolCall(tool="search", args={}), ToolCall(tool="lookup", args={})])
        g.record(["search", "lookup"])
        result = g.check([ToolCall(tool="answer", args={})])
        assert result.action == "execute"


# ── CheckResult dataclass ────────────────────────────────────


class TestCheckResult:
    def test_is_frozen(self):
        r = CheckResult(action="execute", tool_calls=[])
        with pytest.raises(AttributeError):
            r.action = "retry"

    def test_defaults(self):
        r = CheckResult(action="retry")
        assert r.tool_calls is None
        assert r.nudge is None
        assert r.reason is None


# ── Imports ──────────────────────────────────────────────────


class TestImports:
    def test_importable_from_guardrails_package(self):
        from forge.guardrails import CheckResult, Guardrails
        assert CheckResult is not None
        assert Guardrails is not None

    def test_importable_from_forge_top_level(self):
        from forge import CheckResult, Guardrails
        assert CheckResult is not None
        assert Guardrails is not None
