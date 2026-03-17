"""Tests for StepEnforcer."""

import pytest

from forge.core.workflow import ToolCall
from forge.guardrails import StepEnforcer


class TestStepEnforcerCheck:
    """Premature terminal detection and escalation."""

    def setup_method(self):
        self.enforcer = StepEnforcer(
            required_steps=["search", "lookup"],
            terminal_tool="answer",
        )

    def test_no_terminal_no_nudge(self):
        calls = [ToolCall(tool="search", args={})]
        result = self.enforcer.check(calls)
        assert result.needs_nudge is False
        assert result.nudge is None

    def test_terminal_before_steps_nudges(self):
        calls = [ToolCall(tool="answer", args={})]
        result = self.enforcer.check(calls)
        assert result.needs_nudge is True
        assert result.nudge.kind == "step"
        assert result.nudge.role == "user"
        assert result.nudge.tier == 1

    def test_escalation_tiers(self):
        calls = [ToolCall(tool="answer", args={})]
        r1 = self.enforcer.check(calls)
        r2 = self.enforcer.check(calls)
        r3 = self.enforcer.check(calls)
        assert r1.nudge.tier == 1
        assert r2.nudge.tier == 2
        assert r3.nudge.tier == 3

    def test_tier_caps_at_3(self):
        calls = [ToolCall(tool="answer", args={})]
        for _ in range(5):
            result = self.enforcer.check(calls)
        assert result.nudge.tier == 3

    def test_terminal_after_steps_satisfied_no_nudge(self):
        self.enforcer.record("search")
        self.enforcer.record("lookup")
        calls = [ToolCall(tool="answer", args={})]
        result = self.enforcer.check(calls)
        assert result.needs_nudge is False

    def test_terminal_with_partial_steps_nudges(self):
        self.enforcer.record("search")
        calls = [ToolCall(tool="answer", args={})]
        result = self.enforcer.check(calls)
        assert result.needs_nudge is True
        assert "lookup" in result.nudge.content

    def test_non_terminal_tools_always_pass(self):
        calls = [ToolCall(tool="search", args={}), ToolCall(tool="lookup", args={})]
        result = self.enforcer.check(calls)
        assert result.needs_nudge is False

    def test_batch_with_terminal_and_others(self):
        calls = [
            ToolCall(tool="search", args={}),
            ToolCall(tool="answer", args={}),
        ]
        result = self.enforcer.check(calls)
        assert result.needs_nudge is True


class TestStepEnforcerRecord:
    """Step recording and satisfaction."""

    def setup_method(self):
        self.enforcer = StepEnforcer(
            required_steps=["search", "lookup"],
            terminal_tool="answer",
        )

    def test_initially_not_satisfied(self):
        assert self.enforcer.is_satisfied() is False

    def test_partial_not_satisfied(self):
        self.enforcer.record("search")
        assert self.enforcer.is_satisfied() is False
        assert self.enforcer.pending() == ["lookup"]

    def test_all_recorded_satisfied(self):
        self.enforcer.record("search")
        self.enforcer.record("lookup")
        assert self.enforcer.is_satisfied() is True
        assert self.enforcer.pending() == []

    def test_duplicate_record_harmless(self):
        self.enforcer.record("search")
        self.enforcer.record("search")
        assert self.enforcer.pending() == ["lookup"]

    def test_recording_non_required_tool_harmless(self):
        self.enforcer.record("other_tool")
        assert self.enforcer.pending() == ["search", "lookup"]


class TestStepEnforcerTerminalReached:
    """Terminal detection helper."""

    def setup_method(self):
        self.enforcer = StepEnforcer(
            required_steps=["search"],
            terminal_tool="answer",
        )

    def test_terminal_reached_when_satisfied(self):
        self.enforcer.record("search")
        calls = [ToolCall(tool="answer", args={})]
        assert self.enforcer.terminal_reached(calls) is True

    def test_terminal_not_reached_when_unsatisfied(self):
        calls = [ToolCall(tool="answer", args={})]
        assert self.enforcer.terminal_reached(calls) is False

    def test_no_terminal_in_batch(self):
        self.enforcer.record("search")
        calls = [ToolCall(tool="search", args={})]
        assert self.enforcer.terminal_reached(calls) is False


class TestStepEnforcerExhaustion:
    """Premature attempt exhaustion."""

    def test_premature_exhausted(self):
        enforcer = StepEnforcer(
            required_steps=["search"],
            terminal_tool="answer",
            max_premature_attempts=2,
        )
        calls = [ToolCall(tool="answer", args={})]
        enforcer.check(calls)
        enforcer.check(calls)
        assert enforcer.premature_exhausted is False
        enforcer.check(calls)
        assert enforcer.premature_exhausted is True

    def test_premature_attempts_count(self):
        enforcer = StepEnforcer(
            required_steps=["search"],
            terminal_tool="answer",
        )
        assert enforcer.premature_attempts == 0
        enforcer.check([ToolCall(tool="answer", args={})])
        assert enforcer.premature_attempts == 1


class TestStepEnforcerNoRequiredSteps:
    """Edge case: no required steps."""

    def test_always_satisfied(self):
        enforcer = StepEnforcer(
            required_steps=[], terminal_tool="answer"
        )
        assert enforcer.is_satisfied() is True

    def test_terminal_never_premature(self):
        enforcer = StepEnforcer(
            required_steps=[], terminal_tool="answer"
        )
        calls = [ToolCall(tool="answer", args={})]
        result = enforcer.check(calls)
        assert result.needs_nudge is False
