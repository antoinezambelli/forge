"""Tests for StepEnforcer."""

from forge.core.workflow import ToolCall
from forge.guardrails import StepEnforcer


class TestStepEnforcerCheck:
    """Premature terminal detection and escalation."""

    def setup_method(self):
        self.enforcer = StepEnforcer(
            required_steps=["search", "lookup"],
            terminal_tools=frozenset(["answer"]),
        )

    def test_terminal_before_steps_escalates_and_caps_nudge_tier(self):
        calls = [ToolCall(tool="answer", args={})]
        results = [self.enforcer.check(calls) for _ in range(5)]
        assert results[0].needs_nudge is True
        assert results[0].nudge.kind == "step"
        assert results[0].nudge.role == "user"
        assert [result.nudge.tier for result in results] == [1, 2, 3, 3, 3]

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

    def test_recording_lifecycle_and_forwarded_tracker_behavior(self):
        assert self.enforcer.is_satisfied() is False
        self.enforcer.record("search")
        self.enforcer.record("search")
        self.enforcer.record("other_tool")
        assert self.enforcer.is_satisfied() is False
        assert self.enforcer.pending() == ["lookup"]
        self.enforcer.record("lookup")
        assert self.enforcer.is_satisfied() is True
        assert self.enforcer.pending() == []


class TestStepEnforcerTerminalReached:
    """Terminal detection helper."""

    def setup_method(self):
        self.enforcer = StepEnforcer(
            required_steps=["search"],
            terminal_tools=frozenset(["answer"]),
        )

    def test_requires_both_satisfied_steps_and_terminal_call(self):
        terminal = [ToolCall(tool="answer", args={})]
        nonterminal = [ToolCall(tool="search", args={})]
        assert self.enforcer.terminal_reached(terminal) is False
        self.enforcer.record("search")
        assert self.enforcer.terminal_reached(nonterminal) is False
        assert self.enforcer.terminal_reached(terminal) is True


class TestStepEnforcerExhaustion:
    """Premature attempt exhaustion."""

    def test_premature_exhausted(self):
        enforcer = StepEnforcer(
            required_steps=["search"],
            terminal_tools=frozenset(["answer"]),
            max_premature_attempts=2,
        )
        calls = [ToolCall(tool="answer", args={})]
        assert enforcer.premature_attempts == 0
        enforcer.check(calls)
        enforcer.check(calls)
        assert enforcer.premature_attempts == 2
        assert enforcer.premature_exhausted is False
        enforcer.check(calls)
        assert enforcer.premature_exhausted is True


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
        result = enforcer.check(calls)
        assert result.nudge.tier == 1


class TestStepEnforcerCompletedSteps:
    """Completed steps property."""

    def test_reflects_recording_lifecycle(self):
        enforcer = StepEnforcer(
            required_steps=["search", "lookup"], terminal_tools=frozenset(["answer"])
        )
        assert enforcer.completed_steps == {}
        enforcer.record("search")
        assert "search" in enforcer.completed_steps
        assert "lookup" not in enforcer.completed_steps


class TestStepEnforcerNoRequiredSteps:
    """Edge case: no required steps."""

    def test_is_satisfied_and_never_blocks_terminal(self):
        enforcer = StepEnforcer(
            required_steps=[], terminal_tools=frozenset(["answer"])
        )
        assert enforcer.is_satisfied() is True
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

    def test_blocks_when_matching_prerequisite_is_missing(self):
        for label, recorded_path in [("never called", None), ("wrong arg", "other.py")]:
            enforcer = StepEnforcer(
                required_steps=[],
                terminal_tools=frozenset(["respond"]),
                tool_prerequisites={
                    "edit_file": [{"tool": "read_file", "match_arg": "path"}],
                },
            )
            if recorded_path is not None:
                enforcer.record("read_file", {"path": recorded_path})
            calls = [ToolCall(tool="edit_file", args={"path": "foo.py"})]
            assert enforcer.check_prerequisites(calls).needs_nudge is True, label

    def test_passes_with_matching_arg(self):
        self.enforcer.record("read_file", {"path": "foo.py"})
        calls = [ToolCall(tool="edit_file", args={"path": "foo.py"})]
        result = self.enforcer.check_prerequisites(calls)
        assert result.needs_nudge is False

    def test_non_dict_args_blocks_without_crash(self):
        # Malformed (non-dict) args can't satisfy an arg-match. ResponseValidator
        # normally rejects them before dispatch, but a granular caller may reach
        # here directly — it must block, not crash on ``args.get``.
        self.enforcer.record("read_file", {"path": "foo.py"})
        calls = [ToolCall(tool="edit_file", args="not a dict")]  # type: ignore[arg-type]
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
        assert enforcer.prereq_violations == 0
        enforcer.check_prerequisites(calls)
        enforcer.check_prerequisites(calls)
        assert enforcer.prereq_violations == 2
        assert enforcer.prereq_exhausted is False
        enforcer.check_prerequisites(calls)
        assert enforcer.prereq_exhausted is True

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


class TestMultipleTerminalTools:
    """Multiple terminal tools support."""

    def setup_method(self):
        self.enforcer = StepEnforcer(
            required_steps=["gather_data"],
            terminal_tools=frozenset(["set_ac", "no_action"]),
        )

    def test_each_terminal_triggers_premature_check(self):
        for terminal in ["set_ac", "no_action"]:
            enforcer = StepEnforcer(
                required_steps=["gather_data"],
                terminal_tools=frozenset(["set_ac", "no_action"]),
            )
            result = enforcer.check([ToolCall(tool=terminal, args={})])
            assert result.needs_nudge is True, terminal
            assert terminal in result.nudge.content, terminal

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
