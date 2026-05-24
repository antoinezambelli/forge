"""Guardrails format-agnostic tests.

Verifies that Forge guardrails (ResponseValidator, ErrorTracker,
StepEnforcer, Guardrails) work correctly regardless of encoder/decoder
combo. Guardrails operate on forge internal types (TextResponse, ToolCall)
and must work with any backend format.
"""

from __future__ import annotations

from forge.core.workflow import TextResponse, ToolCall
from forge.guardrails import ErrorTracker, Guardrails, ResponseValidator, StepEnforcer

# ── ResponseValidator works with all backends ─────────────────


class TestResponseValidatorAllBackends:
    """ResponseValidator validates forge types, not wire formats."""

    def test_validates_tool_calls_from_any_backend(self):
        """Valid tool calls pass regardless of backend format."""
        v = ResponseValidator(tool_names=["search", "answer"])
        calls = [ToolCall(tool="search", args={"q": "test"})]
        result = v.validate(calls)
        assert result.needs_retry is False
        assert result.tool_calls == calls

    def test_unknown_tool_detected_from_any_backend(self):
        """Unknown tool detection works regardless of backend."""
        v = ResponseValidator(tool_names=["search"])
        calls = [ToolCall(tool="unknown_tool", args={})]
        result = v.validate(calls)
        assert result.needs_retry is True
        assert result.nudge.kind == "unknown_tool"

    def test_rescue_works_from_any_backend(self):
        """Rescue parsing works regardless of backend format."""
        v = ResponseValidator(
            tool_names=["search", "answer"],
            rescue_enabled=True,
        )
        text = '{"tool": "search", "args": {"q": "rescued"}}'
        result = v.validate(TextResponse(content=text))
        assert result.needs_retry is False
        assert result.tool_calls[0].tool == "search"
        assert result.tool_calls[0].args == {"q": "rescued"}

    def test_retry_nudge_for_text_from_any_backend(self):
        """Text response triggers retry regardless of backend."""
        v = ResponseValidator(
            tool_names=["search", "answer"],
            rescue_enabled=False,
        )
        result = v.validate(TextResponse(content="I don't know"))
        assert result.needs_retry is True
        assert result.nudge.kind == "retry"


# ── ErrorTracker works with all backends ─────────────────────


class TestErrorTrackerAllBackends:
    """ErrorTracker tracks errors independently of wire format."""

    def test_retries_increment(self):
        et = ErrorTracker(max_retries=3)
        et.record_retry()
        et.record_retry()
        assert et.consecutive_retries == 2
        assert et.retries_exhausted is False
        et.record_retry()
        assert et.retries_exhausted is False  # 3 > 3 = False
        et.record_retry()
        assert et.retries_exhausted is True  # 4 > 3 = True

    def test_retries_reset_on_success(self):
        et = ErrorTracker(max_retries=3)
        et.record_retry()
        et.record_retry()
        et.reset_retries()
        assert et.consecutive_retries == 0
        assert et.retries_exhausted is False

    def test_tool_errors_increment(self):
        et = ErrorTracker(max_tool_errors=2)
        et.record_result(success=False, is_soft_error=False)
        et.record_result(success=False, is_soft_error=False)
        assert et.consecutive_tool_errors == 2
        assert et.tool_errors_exhausted is False  # 2 > 2 = False
        et.record_result(success=False, is_soft_error=False)
        assert et.tool_errors_exhausted is True  # 3 > 2 = True

    def test_soft_errors_do_not_increment(self):
        et = ErrorTracker(max_tool_errors=2)
        et.record_result(success=False, is_soft_error=True)
        assert et.consecutive_tool_errors == 0

    def test_success_does_not_reset(self):
        """Individual success doesn't reset — only clean batch does."""
        et = ErrorTracker(max_tool_errors=2)
        et.record_result(success=False)
        et.record_result(success=True)
        assert et.consecutive_tool_errors == 1

    def test_reset_errors_on_clean_batch(self):
        et = ErrorTracker(max_tool_errors=2)
        et.record_result(success=False)
        et.record_result(success=False)
        et.record_result(success=False)
        assert et.tool_errors_exhausted is True  # 3 > 2
        et.reset_errors()
        assert et.consecutive_tool_errors == 0
        assert et.tool_errors_exhausted is False


# ── StepEnforcer works with all backends ─────────────────────


class TestStepEnforcerAllBackends:
    """StepEnforcer enforces steps regardless of wire format."""

    def test_blocks_premature_terminal(self):
        se = StepEnforcer(
            required_steps=["search"],
            terminal_tools=frozenset(["answer"]),
        )
        result = se.check([ToolCall(tool="answer", args={})])
        assert result.needs_nudge is True

    def test_allows_terminal_after_steps(self):
        se = StepEnforcer(
            required_steps=["search"],
            terminal_tools=frozenset(["answer"]),
        )
        se.record("search")
        result = se.check([ToolCall(tool="answer", args={})])
        assert result.needs_nudge is False

    def test_prerequisites_blocked(self):
        se = StepEnforcer(
            required_steps=[],
            terminal_tools=frozenset(["respond"]),
            tool_prerequisites={"edit_file": ["read_file"]},
        )
        result = se.check_prerequisites([ToolCall(tool="edit_file", args={})])
        assert result.needs_nudge is True

    def test_prerequisites_passed(self):
        se = StepEnforcer(
            required_steps=[],
            terminal_tools=frozenset(["respond"]),
            tool_prerequisites={"edit_file": ["read_file"]},
        )
        se.record("read_file")
        result = se.check_prerequisites([ToolCall(tool="edit_file", args={})])
        assert result.needs_nudge is False


# ── Guardrails facade works with all backends ────────────────


class TestGuardrailsFacadeAllBackends:
    """Guardrails facade produces correct actions regardless of backend."""

    def make(self, **overrides):
        defaults = dict(
            tool_names=["search", "lookup", "answer"],
            required_steps=["search", "lookup"],
            terminal_tool="answer",
        )
        defaults.update(overrides)
        return Guardrails(**defaults)

    def test_valid_tool_execute(self):
        g = self.make()
        result = g.check([ToolCall(tool="search", args={"q": "x"})])
        assert result.action == "execute"

    def test_text_retry(self):
        g = self.make(rescue_enabled=False)
        result = g.check(TextResponse(content="bare text"))
        assert result.action == "retry"

    def test_premature_terminal_blocked(self):
        g = self.make()
        result = g.check([ToolCall(tool="answer", args={})])
        assert result.action == "step_blocked"

    def test_terminal_allowed_after_steps(self):
        g = self.make()
        g.check([ToolCall(tool="search", args={})])
        g.record(["search"])
        g.check([ToolCall(tool="lookup", args={})])
        g.record(["lookup"])
        result = g.check([ToolCall(tool="answer", args={})])
        assert result.action == "execute"

    def test_record_terminal_returns_done(self):
        g = self.make()
        g.check([ToolCall(tool="search", args={})])
        g.record(["search"])
        g.check([ToolCall(tool="lookup", args={})])
        g.record(["lookup"])
        g.check([ToolCall(tool="answer", args={})])
        assert g.record(["answer"]) is True

    def test_record_non_terminal_returns_not_done(self):
        g = self.make()
        g.check([ToolCall(tool="search", args={})])
        assert g.record(["search"]) is False


# ── Multi-turn agent loop with guardrails ────────────────────


class TestMultiTurnWithGuardrails:
    """Simulate a multi-turn agent loop with guardrails.

    This tests the full agentic loop pattern used by WorkflowRunner
    and the proxy handler, verifying guardrails work correctly
    regardless of encoder/decoder combo.
    """

    def test_two_step_tool_call_loop(self):
        """search → lookup → answer (with all guardrails)."""
        g = Guardrails(
            tool_names=["search", "lookup", "answer"],
            required_steps=["search", "lookup"],
            terminal_tool="answer",
        )

        # Turn 1: model calls search
        r1 = g.check([ToolCall(tool="search", args={"q": "weather"})])
        assert r1.action == "execute"
        g.record(["search"])

        # Turn 2: model calls lookup
        r2 = g.check([ToolCall(tool="lookup", args={"source": "api"})])
        assert r2.action == "execute"
        g.record(["lookup"])

        # Turn 3: model tries answer — should be allowed
        r3 = g.check([ToolCall(tool="answer", args={"text": "sunny"})])
        assert r3.action == "execute"
        assert g.record(["answer"]) is True

    def test_premature_answer_then_retry(self):
        """Model tries answer too early, gets blocked, then complies."""
        g = Guardrails(
            tool_names=["search", "answer"],
            required_steps=["search"],
            terminal_tool="answer",
        )

        # Premature
        r1 = g.check([ToolCall(tool="answer", args={"text": "hmm"})])
        assert r1.action == "step_blocked"

        # Now do the required step
        r2 = g.check([ToolCall(tool="search", args={"q": "x"})])
        assert r2.action == "execute"
        g.record(["search"])

        # Now answer is allowed
        r3 = g.check([ToolCall(tool="answer", args={"text": "result"})])
        assert r3.action == "execute"

    def test_error_tracker_with_tool_failures(self):
        """ErrorTracker correctly tracks tool execution errors."""
        et = ErrorTracker(max_retries=3, max_tool_errors=2)

        # Simulate: valid tool → execution error → valid tool → done
        et.record_result(success=False, is_soft_error=False)
        assert et.tool_errors_exhausted is False

        et.record_result(success=True)
        # Not reset yet

        et.reset_errors()
        assert et.consecutive_tool_errors == 0

    def test_retry_tracker_with_format_agnostic_responses(self):
        """ErrorTracker works regardless of whether backend returns
        TextResponse (OpenAI) or tool_use (Anthropic) internally."""
        et = ErrorTracker(max_retries=2)

        # Simulate repeated validation failures (format-agnostic)
        et.record_retry()
        et.record_retry()
        assert et.retries_exhausted is False  # 2 > 2 = False
        et.record_retry()
        assert et.retries_exhausted is True  # 3 > 2 = True

        # Reset on success
        et.reset_retries()
        assert et.retries_exhausted is False

    def test_step_tracker_with_arg_matched_prereqs(self):
        """StepEnforcer arg-matched prerequisites work correctly."""
        se = StepEnforcer(
            required_steps=[],
            terminal_tools=frozenset(["respond"]),
            tool_prerequisites={
                "edit_file": [{"tool": "read_file", "match_arg": "path"}],
            },
        )

        # Read different file — doesn't satisfy
        se.record("read_file", {"path": "other.py"})
        result = se.check_prerequisites([ToolCall(tool="edit_file", args={"path": "foo.py"})])
        assert result.needs_nudge is True

        # Read the right file — satisfies
        se.record("read_file", {"path": "foo.py"})
        result = se.check_prerequisites([ToolCall(tool="edit_file", args={"path": "foo.py"})])
        assert result.needs_nudge is False
