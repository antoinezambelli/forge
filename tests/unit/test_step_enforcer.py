"""Tests for StepEnforcer."""

import pytest

from forge.core.workflow import ToolCall
from forge.guardrails import StepEnforcer


class TestStepEnforcerCheck:
    """Premature terminal detection and escalation."""

    def setup_method(self):
        self.enforcer = StepEnforcer(
            required_steps=["search", "lookup"],
            terminal_tools=frozenset(["answer"]),
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
            terminal_tools=frozenset(["answer"]),
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
            terminal_tools=frozenset(["answer"]),
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
            terminal_tools=frozenset(["answer"]),
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
            terminal_tools=frozenset(["answer"]),
        )
        assert enforcer.premature_attempts == 0
        enforcer.check([ToolCall(tool="answer", args={})])
        assert enforcer.premature_attempts == 1


class TestStepEnforcerResetPremature:
    """Premature attempt counter reset."""

    def test_reset_clears_counter(self):
        enforcer = StepEnforcer(
            required_steps=["search"],
            terminal_tools=frozenset(["answer"]),
            max_premature_attempts=2,
        )
        calls = [ToolCall(tool="answer", args={})]
        enforcer.check(calls)
        enforcer.check(calls)
        assert enforcer.premature_attempts == 2
        enforcer.reset_premature()
        assert enforcer.premature_attempts == 0
        assert enforcer.premature_exhausted is False

    def test_reset_allows_fresh_attempts(self):
        enforcer = StepEnforcer(
            required_steps=["search"],
            terminal_tools=frozenset(["answer"]),
            max_premature_attempts=2,
        )
        calls = [ToolCall(tool="answer", args={})]
        enforcer.check(calls)
        enforcer.check(calls)
        enforcer.reset_premature()
        # Should get fresh tier-1 nudge after reset
        result = enforcer.check(calls)
        assert result.nudge.tier == 1


class TestStepEnforcerCompletedSteps:
    """Completed steps property."""

    def test_initially_empty(self):
        enforcer = StepEnforcer(
            required_steps=["search"], terminal_tools=frozenset(["answer"])
        )
        assert enforcer.completed_steps == {}

    def test_reflects_recordings(self):
        enforcer = StepEnforcer(
            required_steps=["search", "lookup"], terminal_tools=frozenset(["answer"])
        )
        enforcer.record("search")
        assert "search" in enforcer.completed_steps
        assert "lookup" not in enforcer.completed_steps


class TestStepEnforcerNoRequiredSteps:
    """Edge case: no required steps."""

    def test_always_satisfied(self):
        enforcer = StepEnforcer(
            required_steps=[], terminal_tools=frozenset(["answer"])
        )
        assert enforcer.is_satisfied() is True

    def test_terminal_never_premature(self):
        enforcer = StepEnforcer(
            required_steps=[], terminal_tools=frozenset(["answer"])
        )
        calls = [ToolCall(tool="answer", args={})]
        result = enforcer.check(calls)
        assert result.needs_nudge is False


class TestPrerequisiteCheckNameOnly:
    """Name-only prerequisite enforcement."""

    def setup_method(self):
        self.enforcer = StepEnforcer(
            required_steps=[],
            terminal_tools=frozenset(["respond"]),
            tool_prerequisites={"edit_file": ["read_file"]},
        )

    def test_blocks_without_prereq(self):
        calls = [ToolCall(tool="edit_file", args={"path": "foo.py"})]
        result = self.enforcer.check_prerequisites(calls)
        assert result.needs_nudge is True
        assert result.nudge.kind == "prerequisite"
        assert "read_file" in result.nudge.content

    def test_passes_after_prereq_satisfied(self):
        self.enforcer.record("read_file", {"path": "bar.py"})
        calls = [ToolCall(tool="edit_file", args={"path": "foo.py"})]
        result = self.enforcer.check_prerequisites(calls)
        assert result.needs_nudge is False

    def test_tool_without_prereqs_always_passes(self):
        calls = [ToolCall(tool="read_file", args={"path": "foo.py"})]
        result = self.enforcer.check_prerequisites(calls)
        assert result.needs_nudge is False


class TestPrerequisiteCheckArgMatched:
    """Arg-matched prerequisite enforcement."""

    def setup_method(self):
        self.enforcer = StepEnforcer(
            required_steps=[],
            terminal_tools=frozenset(["respond"]),
            tool_prerequisites={
                "edit_file": [{"tool": "read_file", "match_arg": "path"}],
            },
        )

    def test_blocks_without_matching_arg(self):
        self.enforcer.record("read_file", {"path": "other.py"})
        calls = [ToolCall(tool="edit_file", args={"path": "foo.py"})]
        result = self.enforcer.check_prerequisites(calls)
        assert result.needs_nudge is True

    def test_passes_with_matching_arg(self):
        self.enforcer.record("read_file", {"path": "foo.py"})
        calls = [ToolCall(tool="edit_file", args={"path": "foo.py"})]
        result = self.enforcer.check_prerequisites(calls)
        assert result.needs_nudge is False

    def test_blocks_when_prereq_never_called(self):
        calls = [ToolCall(tool="edit_file", args={"path": "foo.py"})]
        result = self.enforcer.check_prerequisites(calls)
        assert result.needs_nudge is True

    def test_multiple_files_tracked_independently(self):
        self.enforcer.record("read_file", {"path": "a.py"})
        self.enforcer.record("read_file", {"path": "b.py"})
        # a.py satisfied
        calls_a = [ToolCall(tool="edit_file", args={"path": "a.py"})]
        assert self.enforcer.check_prerequisites(calls_a).needs_nudge is False
        # b.py satisfied
        calls_b = [ToolCall(tool="edit_file", args={"path": "b.py"})]
        assert self.enforcer.check_prerequisites(calls_b).needs_nudge is False
        # c.py not satisfied
        calls_c = [ToolCall(tool="edit_file", args={"path": "c.py"})]
        assert self.enforcer.check_prerequisites(calls_c).needs_nudge is True


class TestPrerequisiteCheckMixed:
    """Mixed name-only and arg-matched prerequisites."""

    def test_both_must_be_satisfied(self):
        enforcer = StepEnforcer(
            required_steps=[],
            terminal_tools=frozenset(["respond"]),
            tool_prerequisites={
                "edit_file": [
                    "authenticate",
                    {"tool": "read_file", "match_arg": "path"},
                ],
            },
        )
        # Neither satisfied
        calls = [ToolCall(tool="edit_file", args={"path": "foo.py"})]
        result = enforcer.check_prerequisites(calls)
        assert result.needs_nudge is True
        assert "authenticate" in result.nudge.content

        # Only auth satisfied
        enforcer.record("authenticate", {})
        result = enforcer.check_prerequisites(calls)
        assert result.needs_nudge is True
        assert "read_file" in result.nudge.content

        # Both satisfied
        enforcer.record("read_file", {"path": "foo.py"})
        result = enforcer.check_prerequisites(calls)
        assert result.needs_nudge is False


class TestPrerequisiteBatchBlocking:
    """Whole-batch blocking on prerequisite violation."""

    def test_any_violation_blocks_entire_batch(self):
        enforcer = StepEnforcer(
            required_steps=[],
            terminal_tools=frozenset(["respond"]),
            tool_prerequisites={"edit_file": ["read_file"]},
        )
        calls = [
            ToolCall(tool="read_file", args={"path": "foo.py"}),
            ToolCall(tool="edit_file", args={"path": "foo.py"}),
        ]
        # edit_file prereq not yet satisfied (read_file hasn't been recorded)
        result = enforcer.check_prerequisites(calls)
        assert result.needs_nudge is True


class TestPrerequisiteExhaustion:
    """Consecutive prerequisite violation exhaustion."""

    def test_exhausted_after_max_violations(self):
        enforcer = StepEnforcer(
            required_steps=[],
            terminal_tools=frozenset(["respond"]),
            tool_prerequisites={"edit_file": ["read_file"]},
            max_prereq_violations=2,
        )
        calls = [ToolCall(tool="edit_file", args={"path": "foo.py"})]
        enforcer.check_prerequisites(calls)
        enforcer.check_prerequisites(calls)
        assert enforcer.prereq_exhausted is False
        enforcer.check_prerequisites(calls)
        assert enforcer.prereq_exhausted is True

    def test_violation_count_tracks(self):
        enforcer = StepEnforcer(
            required_steps=[],
            terminal_tools=frozenset(["respond"]),
            tool_prerequisites={"edit_file": ["read_file"]},
        )
        assert enforcer.prereq_violations == 0
        enforcer.check_prerequisites([ToolCall(tool="edit_file", args={})])
        assert enforcer.prereq_violations == 1

    def test_reset_clears_violations(self):
        enforcer = StepEnforcer(
            required_steps=[],
            terminal_tools=frozenset(["respond"]),
            tool_prerequisites={"edit_file": ["read_file"]},
            max_prereq_violations=1,
        )
        enforcer.check_prerequisites([ToolCall(tool="edit_file", args={})])
        enforcer.check_prerequisites([ToolCall(tool="edit_file", args={})])
        assert enforcer.prereq_exhausted is True
        enforcer.reset_prereq_violations()
        assert enforcer.prereq_violations == 0
        assert enforcer.prereq_exhausted is False


class TestPrerequisiteNoPrereqs:
    """Edge case: no prerequisites configured."""

    def test_always_passes(self):
        enforcer = StepEnforcer(
            required_steps=[],
            terminal_tools=frozenset(["respond"]),
        )
        calls = [ToolCall(tool="edit_file", args={"path": "foo.py"})]
        result = enforcer.check_prerequisites(calls)
        assert result.needs_nudge is False


class TestMultipleTerminalTools:
    """Multiple terminal tools support."""

    def setup_method(self):
        self.enforcer = StepEnforcer(
            required_steps=["gather_data"],
            terminal_tools=frozenset(["set_ac", "no_action"]),
        )

    def test_either_terminal_triggers_premature_check(self):
        calls_a = [ToolCall(tool="set_ac", args={})]
        result = self.enforcer.check(calls_a)
        assert result.needs_nudge is True
        assert "set_ac" in result.nudge.content

    def test_second_terminal_also_triggers(self):
        calls_b = [ToolCall(tool="no_action", args={})]
        result = self.enforcer.check(calls_b)
        assert result.needs_nudge is True
        assert "no_action" in result.nudge.content

    def test_either_terminal_succeeds_after_steps(self):
        self.enforcer.record("gather_data")
        calls_a = [ToolCall(tool="set_ac", args={})]
        assert self.enforcer.terminal_reached(calls_a) is True
        calls_b = [ToolCall(tool="no_action", args={})]
        assert self.enforcer.terminal_reached(calls_b) is True

    def test_non_terminal_does_not_trigger(self):
        calls = [ToolCall(tool="gather_data", args={})]
        result = self.enforcer.check(calls)
        assert result.needs_nudge is False

    def test_non_terminal_does_not_reach(self):
        self.enforcer.record("gather_data")
        calls = [ToolCall(tool="gather_data", args={})]
        assert self.enforcer.terminal_reached(calls) is False
